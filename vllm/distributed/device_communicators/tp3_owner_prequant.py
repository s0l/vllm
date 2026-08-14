# SPDX-License-Identifier: Apache-2.0
"""Default-off TP3 8/8/4 residual/RMSNorm/ARC-prequant seam."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed import ProcessGroup

from vllm._custom_ops import scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.nvfp4.arc import (
    _decode_e2m1,
    _encode_e2m1,
    _round_e2m1,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    convert_swizzled_to_linear,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)

from .tp3_exact_reduce import fixed_tp3_sum_fused


logger = init_logger(__name__)
WORLD = 3
HIDDEN = 5120
BLOCK = 1024
WIDTHS = (2048, 2048, 1024)
BLOCKS = (2, 2, 1)
OFFSETS = (0, 2048, 4096)

_owner_collective_workspace: torch.Tensor | None = None
_owner_replay_entries: dict[
    tuple[object, ...], tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]]
] = {}
_owner_replay_capture_streams: dict[int, torch.cuda.Stream] = {}
_owner_replay_pool: tuple[int, int] | None = None
_owner_replay_hits = 0
_owner_replay_misses = 0


def _clear_tp3_owner_prequant_replay_gate() -> None:
    global _owner_replay_hits, _owner_replay_misses, _owner_replay_pool
    _owner_replay_entries.clear()
    _owner_replay_capture_streams.clear()
    _owner_replay_pool = None
    _owner_replay_hits = 0
    _owner_replay_misses = 0


def get_tp3_owner_prequant_replay_gate_stats() -> dict[str, int]:
    """Return bounded lifecycle counters for startup/runtime evidence."""
    return {
        "entries": len(_owner_replay_entries),
        "hits": _owner_replay_hits,
        "misses": _owner_replay_misses,
    }


def set_tp3_owner_prequant_workspace(scratch: torch.Tensor | None) -> None:
    """Install the runner-owned sequential collective workspace.

    DCP prefill, target owner reduction, and MTP gate/up execute sequentially
    in one worker.  The runner therefore owns one persistent allocation and
    each consumer resolves checked, non-overlapping views for its own phase.
    """
    global _owner_collective_workspace
    if scratch is not None:
        if scratch.dtype != torch.bfloat16 or not scratch.is_contiguous():
            raise ValueError("TP3 owner workspace must be contiguous BF16")
        if scratch.ndim != 2:
            raise ValueError("TP3 owner workspace must be a 2D tensor")
    old_pointer = (
        _owner_collective_workspace.data_ptr()
        if _owner_collective_workspace is not None
        else None
    )
    new_pointer = scratch.data_ptr() if scratch is not None else None
    if old_pointer != new_pointer:
        _clear_tp3_owner_prequant_replay_gate()
    _owner_collective_workspace = scratch


def _tensor_replay_identity(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.index,
    )


def _owner_replay_max_rows() -> int:
    value = int(os.environ.get("AG2_VLLM_TP3_OWNER_REPLAY_GATE_MAX_ROWS", "256"))
    if value < 1:
        raise ValueError("AG2_VLLM_TP3_OWNER_REPLAY_GATE_MAX_ROWS must be positive")
    return value


def _owner_replay_min_rows() -> int:
    """Fail closed below the measured child-graph crossover.

    This check deliberately lives at the replay owner boundary rather than in
    the compiled model.  Dynamo traces the model branch before vLLM's backend
    partitions its no-guard compile ranges, so a Python shape predicate in the
    model alone cannot prevent small capture shapes from reaching this API.
    """
    value = int(os.environ.get("AG2_VLLM_TP3_OWNER_MIN_ROWS", "64"))
    if value < 1:
        raise ValueError("AG2_VLLM_TP3_OWNER_MIN_ROWS must be positive")
    return value


def _owner_replay_identity(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    input_scale_inv: torch.Tensor,
    selected_all: torch.Tensor,
    route0: torch.Tensor,
    route1: torch.Tensor,
    route2: torch.Tensor,
    inverse_order: torch.Tensor,
    route_counts: list[int],
    group: ProcessGroup,
    eps: float,
) -> tuple[object, ...]:
    return (
        id(group),
        float(eps),
        tuple(route_counts),
        *(
            _tensor_replay_identity(tensor)
            for tensor in (
                contribution,
                residual_owner,
                weight,
                input_scale_inv,
                selected_all,
                route0,
                route1,
                route2,
                inverse_order,
            )
        ),
    )


def _owner_collective_buffers(
    contribution: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scratch = _owner_collective_workspace
    if scratch is None:
        raise RuntimeError("TP3 owner workspace is enabled but not configured")
    if (
        scratch.device != contribution.device
        or scratch.dtype != contribution.dtype
        or not contribution.is_contiguous()
    ):
        raise RuntimeError(
            "TP3 owner workspace/input must be contiguous and match device/dtype"
        )
    rows = contribution.shape[0]
    send_elements = rows * HIDDEN
    receive_elements = WORLD * rows * WIDTHS[rank]
    required = send_elements + receive_elements
    flat = scratch.view(-1)
    if required > flat.numel():
        raise RuntimeError(
            "TP3 owner workspace is too small: "
            f"required_elements={required} capacity_elements={flat.numel()} "
            f"rows={rows} rank={rank}"
        )
    send = flat[:send_elements]
    receive = flat[send_elements:required]
    if send.untyped_storage().data_ptr() != receive.untyped_storage().data_ptr():
        raise RuntimeError("TP3 owner workspace views must share one allocation")
    send_end = send.data_ptr() + send.numel() * send.element_size()
    if send_end > receive.data_ptr():
        raise RuntimeError("TP3 owner workspace send/receive views overlap")
    return send, receive


@triton.jit
def _owner_stats(
    input_ptr,
    residual_ptr,
    stats_ptr,
    input_stride,
    residual_stride,
    stats_stride,
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_ptr + row * input_stride + cols).to(tl.float32)
    residual = tl.load(residual_ptr + row * residual_stride + cols).to(tl.float32)
    combined = value + residual
    tl.store(
        stats_ptr + row * stats_stride + block,
        tl.sum(combined * combined),
    )


@triton.jit
def _owner_norm(
    input_ptr,
    residual_ptr,
    weight_ptr,
    stats_ptr,
    output_ptr,
    residual_out_ptr,
    input_stride,
    residual_stride,
    stats_stride,
    output_stride,
    residual_out_stride,
    eps,
    N_BLOCKS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    total = tl.zeros([1], dtype=tl.float32)
    for index in range(N_BLOCKS):
        total += tl.load(stats_ptr + row * stats_stride + index)
    inv_rms = tl.rsqrt(total / HIDDEN_SIZE + eps)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(input_ptr + row * input_stride + cols).to(tl.float32)
    residual = tl.load(residual_ptr + row * residual_stride + cols).to(tl.float32)
    combined = value + residual
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    tl.store(residual_out_ptr + row * residual_out_stride + cols, combined)
    tl.store(output_ptr + row * output_stride + cols, combined * inv_rms * weight)


@triton.jit
def _write_sparse_tail(
    selected_values_ptr,
    selected_ptr,
    global_scale_ptr,
    packed_ptr,
    scales_ptr,
    selected_stride,
    selected_count: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    pair = tl.arange(0, 8)
    positions = group * 16 + pair * 2
    low_channels = tl.load(selected_ptr + positions)
    high_channels = tl.load(selected_ptr + positions + 1)
    augmented_k: tl.constexpr = HIDDEN_SIZE + selected_count
    packed_row = row * (augmented_k // 2)
    low_bytes = tl.load(packed_ptr + packed_row + low_channels // 2)
    high_bytes = tl.load(packed_ptr + packed_row + high_channels // 2)
    low_nibbles = tl.where(
        (low_channels & 1) == 0, low_bytes & 15, (low_bytes >> 4) & 15
    )
    high_nibbles = tl.where(
        (high_channels & 1) == 0, high_bytes & 15, (high_bytes >> 4) & 15
    )
    num_k_tiles: tl.constexpr = augmented_k // 64
    common = (row & 31) * 16 + ((row >> 5) & 3) * 4
    low_scale_k = low_channels // 16
    high_scale_k = high_channels // 16
    low_offsets = (
        ((row // 128) * num_k_tiles + low_scale_k // 4) * 512
        + common
        + (low_scale_k & 3)
    )
    high_offsets = (
        ((row // 128) * num_k_tiles + high_scale_k // 4) * 512
        + common
        + (high_scale_k & 3)
    )
    low_scales = (
        tl.load(scales_ptr + low_offsets).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    )
    high_scales = (
        tl.load(scales_ptr + high_offsets)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    low_qdq = _decode_e2m1(low_nibbles) * (low_scales / global_scale)
    high_qdq = _decode_e2m1(high_nibbles) * (high_scales / global_scale)
    low_source = tl.load(selected_values_ptr + row * selected_stride + positions).to(
        tl.float32
    )
    high_source = tl.load(
        selected_values_ptr + row * selected_stride + positions + 1
    ).to(tl.float32)
    low_residual = low_source - low_qdq
    high_residual = high_source - high_qdq
    vec_max = tl.maximum(
        tl.max(tl.abs(low_residual), axis=0),
        tl.max(tl.abs(high_residual), axis=0),
    )
    tail_scale = tl.clamp(global_scale * (vec_max / 6.0), -448.0, 448.0)
    tail_scale_fp8 = tail_scale.to(tl.float8e4nv)
    tail_scale_f32 = tail_scale_fp8.to(tl.float32)
    inverse = tl.where(tail_scale_f32 == 0.0, 0.0, global_scale / tail_scale_f32)
    low_quant = _round_e2m1(tl.clamp(low_residual * inverse, -6.0, 6.0))
    high_quant = _round_e2m1(tl.clamp(high_residual * inverse, -6.0, 6.0))
    tl.store(
        packed_ptr + packed_row + HIDDEN_SIZE // 2 + group * 8 + pair,
        _encode_e2m1(low_quant) | (_encode_e2m1(high_quant) << 4),
    )
    tail_scale_index = HIDDEN_SIZE // 16 + group
    tail_offset = (
        ((row // 128) * num_k_tiles + tail_scale_index // 4) * 512
        + common
        + (tail_scale_index & 3)
    )
    tl.store(
        scales_ptr + tail_offset,
        tail_scale_fp8.to(tl.uint8, bitcast=True),
    )


def _reduce_and_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if contribution.ndim != 2 or contribution.shape[1] != HIDDEN:
        raise ValueError(f"owner seam requires [M,{HIDDEN}], got {contribution.shape}")
    rank = dist.get_rank(group)
    rows = contribution.shape[0]
    if residual_owner.shape != (rows, WIDTHS[rank]):
        raise ValueError(
            f"rank {rank} residual owner shape {residual_owner.shape} "
            f"!= {(rows, WIDTHS[rank])}"
        )
    chunks = torch.split(contribution, WIDTHS, dim=1)
    send, receive = _owner_collective_buffers(contribution, rank)
    elements = [rows * width for width in WIDTHS]
    offset = 0
    for chunk, element_count, width in zip(chunks, elements, WIDTHS, strict=True):
        send[offset : offset + element_count].view(rows, width).copy_(chunk)
        offset += element_count
    owned_elements = elements[rank]
    if receive.numel() != WORLD * owned_elements:
        raise RuntimeError("TP3 owner workspace receive shape mismatch")
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[owned_elements] * WORLD,
        input_split_sizes=elements,
        group=group,
    )
    owned = fixed_tp3_sum_fused(receive.view(WORLD, rows, WIDTHS[rank]))
    local_stats = torch.zeros((rows, 2), dtype=torch.float32, device=owned.device)
    _owner_stats[(rows, BLOCKS[rank])](
        owned,
        residual_owner,
        local_stats,
        owned.stride(0),
        residual_owner.stride(0),
        local_stats.stride(0),
        N_BLOCKS=BLOCKS[rank],
        BLOCK_SIZE=BLOCK,
        num_warps=16,
    )
    gathered = torch.empty((WORLD * rows, 2), dtype=torch.float32, device=owned.device)
    dist.all_gather_single(gathered, local_stats, group=group)
    gathered = gathered.view(WORLD, rows, 2)
    global_stats = torch.cat(
        (gathered[0], gathered[1], gathered[2, :, :1]), dim=1
    ).contiguous()
    norm_owner = torch.empty_like(owned)
    residual_out = torch.empty_like(owned)
    weight_owner = weight[OFFSETS[rank] : OFFSETS[rank] + WIDTHS[rank]].contiguous()
    _owner_norm[(rows, BLOCKS[rank])](
        owned,
        residual_owner,
        weight_owner,
        global_stats,
        norm_owner,
        residual_out,
        owned.stride(0),
        residual_owner.stride(0),
        global_stats.stride(0),
        norm_owner.stride(0),
        residual_out.stride(0),
        eps,
        N_BLOCKS=HIDDEN // BLOCK,
        HIDDEN_SIZE=HIDDEN,
        BLOCK_SIZE=BLOCK,
        num_warps=16,
    )
    return norm_owner, residual_out


def _owner_residual_arc_prequant_direct(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    input_scale_inv: torch.Tensor,
    selected_all: torch.Tensor,
    route0: torch.Tensor,
    route1: torch.Tensor,
    route2: torch.Tensor,
    inverse_order: torch.Tensor,
    route_counts: list[int],
    group: ProcessGroup,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    rank = dist.get_rank(group)
    rows = contribution.shape[0]
    selected_count = selected_all.shape[1]
    if selected_all.shape != (WORLD, selected_count) or len(route_counts) != 9:
        raise ValueError("invalid ARC route metadata")
    norm_owner, residual_out = _reduce_and_norm(
        contribution, residual_owner, weight, group, eps
    )
    owner_q, owner_sf = scaled_fp4_quant(
        norm_owner,
        input_scale_inv,
        is_sf_swizzled_layout=True,
        backend="b12x",
    )
    owner_sf = convert_swizzled_to_linear(owner_sf, rows, WIDTHS[rank], 16).contiguous()
    q_elements = [rows * width // 2 for width in WIDTHS]
    sf_elements = [rows * width // 16 for width in WIDTHS]
    base_elements = [q + sf for q, sf in zip(q_elements, sf_elements, strict=True)]
    base = torch.cat((owner_q.reshape(-1), owner_sf.view(torch.uint8).reshape(-1)))
    routes = (route0, route1, route2)
    send_chunks = [
        torch.cat(
            (
                base,
                norm_owner.index_select(1, route)
                .contiguous()
                .view(torch.uint8)
                .reshape(-1),
            )
        )
        for route in routes
    ]
    input_splits = [chunk.numel() for chunk in send_chunks]
    output_splits = [
        base_elements[owner] + rows * route_counts[owner * 3 + rank] * 2
        for owner in range(WORLD)
    ]
    received = torch.empty(
        sum(output_splits), dtype=torch.uint8, device=contribution.device
    )
    dist.all_to_all_single(
        received,
        torch.cat(send_chunks),
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    q_parts, sf_parts, selected_parts = [], [], []
    for owner, payload in enumerate(received.split(output_splits)):
        base_owner = payload[: base_elements[owner]]
        q_parts.append(base_owner[: q_elements[owner]].view(rows, WIDTHS[owner] // 2))
        sf_parts.append(
            base_owner[q_elements[owner] :]
            .view(torch.float8_e4m3fn)
            .view(rows, WIDTHS[owner] // 16)
        )
        selected_parts.append(
            payload[base_elements[owner] :]
            .view(contribution.dtype)
            .view(rows, route_counts[owner * 3 + rank])
        )
    base_q = torch.cat(q_parts, dim=1)
    base_sf = swizzle_blockscale(torch.cat(sf_parts, dim=1).contiguous())
    selected_values = (
        torch.cat(selected_parts, dim=1).index_select(1, inverse_order).contiguous()
    )
    augmented_k = HIDDEN + selected_count
    q = torch.zeros((rows, augmented_k // 2), dtype=torch.uint8, device=base_q.device)
    q[:, : HIDDEN // 2].copy_(base_q)
    base_sf_linear = convert_swizzled_to_linear(base_sf, rows, HIDDEN, 16)
    sf_linear = torch.zeros(
        (rows, augmented_k // 16),
        dtype=torch.float8_e4m3fn,
        device=base_q.device,
    )
    sf_linear[:, : HIDDEN // 16].copy_(base_sf_linear)
    sf = swizzle_blockscale(sf_linear.contiguous())
    _write_sparse_tail[(rows, selected_count // 16)](
        selected_values,
        selected_all[rank],
        input_scale_inv,
        q,
        sf.view(torch.uint8),
        selected_values.stride(0),
        selected_count=selected_count,
        HIDDEN_SIZE=HIDDEN,
        num_warps=4,
    )
    # GDN has two consumers of the same normalized input.  qkvz owns the ARC
    # augmented K, while ba keeps the canonical K.  Both checkpoint projections
    # use one input scale (validated after weight loading), so the canonical
    # payload already assembled here is also the exact ba prequant input.
    return q, sf, base_q, base_sf, residual_out


def owner_residual_arc_prequant(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    input_scale_inv: torch.Tensor,
    selected_all: torch.Tensor,
    route0: torch.Tensor,
    route1: torch.Tensor,
    route2: torch.Tensor,
    inverse_order: torch.Tensor,
    route_counts: list[int],
    group: ProcessGroup,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the owner seam directly or replay one layer-local child graph.

    FULL/parent CUDA Graph capture remains authoritative: nested capture is
    forbidden, so an active outer capture records the direct implementation.
    PIECEWISE execution reaches this function outside stream capture and can
    use a graph keyed by the complete static tensor/route identity.
    """
    global _owner_replay_hits, _owner_replay_misses, _owner_replay_pool
    call_args = (
        contribution,
        residual_owner,
        weight,
        input_scale_inv,
        selected_all,
        route0,
        route1,
        route2,
        inverse_order,
        route_counts,
        group,
        eps,
    )
    if (
        os.environ.get("AG2_VLLM_TP3_OWNER_REPLAY_GATE", "0") != "1"
        or contribution.shape[0] < _owner_replay_min_rows()
        or contribution.shape[0] > _owner_replay_max_rows()
        or torch.cuda.is_current_stream_capturing()
    ):
        return _owner_residual_arc_prequant_direct(*call_args)

    key = _owner_replay_identity(*call_args)
    cached = _owner_replay_entries.get(key)
    if cached is not None:
        _owner_replay_hits += 1
        graph, output = cached
        graph.replay()
        if _owner_replay_hits & (_owner_replay_hits - 1) == 0:
            logger.warning(
                "TP3 owner replay gate hit: hits=%d misses=%d entries=%d rows=%d",
                _owner_replay_hits,
                _owner_replay_misses,
                len(_owner_replay_entries),
                contribution.shape[0],
            )
        return output  # type: ignore[return-value]

    limit = int(os.environ.get("AG2_VLLM_TP3_OWNER_REPLAY_GATE_MAX", "4096"))
    if len(_owner_replay_entries) >= limit:
        raise RuntimeError(
            "TP3 owner replay-gate cache limit exceeded: "
            f"entries={len(_owner_replay_entries)} limit={limit}"
        )

    device_index = contribution.device.index
    if device_index is None:
        raise RuntimeError("TP3 owner replay gate requires a CUDA device index")
    current_stream = torch.cuda.current_stream(contribution.device)
    capture_stream = _owner_replay_capture_streams.get(device_index)
    if capture_stream is None:
        capture_stream = torch.cuda.Stream(device=contribution.device)
        _owner_replay_capture_streams[device_index] = capture_stream
    capture_stream.wait_stream(current_stream)

    graph = torch.cuda.CUDAGraph()
    if _owner_replay_pool is None:
        _owner_replay_pool = torch.cuda.graph_pool_handle()
    with torch.cuda.stream(capture_stream):
        graph.capture_begin(pool=_owner_replay_pool)
        output = _owner_residual_arc_prequant_direct(*call_args)
        graph.capture_end()
    current_stream.wait_stream(capture_stream)
    _owner_replay_entries[key] = (graph, output)
    _owner_replay_misses += 1
    if _owner_replay_misses & (_owner_replay_misses - 1) == 0:
        logger.warning(
            "TP3 owner replay gate capture: hits=%d misses=%d entries=%d rows=%d",
            _owner_replay_hits,
            _owner_replay_misses,
            len(_owner_replay_entries),
            contribution.shape[0],
        )
    # Stream capture records work but does not materialize its outputs.  The
    # miss must execute once before returning the static tensors to a consumer.
    graph.replay()
    return output


def owner_terminal_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    norm_owner, _ = _reduce_and_norm(contribution, residual_owner, weight, group, eps)
    rank = dist.get_rank(group)
    rows = contribution.shape[0]
    elements = [rows * width for width in WIDTHS]
    send = norm_owner.reshape(-1).repeat(WORLD)
    received = torch.empty(
        sum(elements), dtype=contribution.dtype, device=contribution.device
    )
    dist.all_to_all_single(
        received,
        send,
        output_split_sizes=elements,
        input_split_sizes=[elements[rank]] * WORLD,
        group=group,
    )
    return torch.cat(
        [
            part.view(rows, width)
            for part, width in zip(received.split(elements), WIDTHS, strict=True)
        ],
        dim=1,
    )
