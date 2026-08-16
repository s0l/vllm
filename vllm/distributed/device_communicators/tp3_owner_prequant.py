# SPDX-License-Identifier: Apache-2.0
"""Default-off TP3 8/8/4 residual/RMSNorm/ARC-prequant seam."""

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed import ProcessGroup

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.kernels.linear.nvfp4.arc import (
    _decode_e2m1,
    _encode_e2m1,
    _round_e2m1,
    ag2_nvfp4_arc_quantize,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    convert_swizzled_to_linear,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)

from .tp3_exact_reduce import exact_weighted_owner_992_reduce, fixed_tp3_sum_fused

WORLD = 3
HIDDEN = 5120
BLOCK = 1024
WIDTHS = (2048, 2048, 1024)
V1_WIDTHS = WIDTHS
V1B5_WIDTHS = WIDTHS
BLOCKS = (2, 2, 1)
OFFSETS = (0, 2048, 4096)
V4_CLASS_STARTS = (0, 6, 11)
V4_CLASS_COUNTS = (6, 5, 5)
V4_WIDTHS = (1920, 1600, 1600)
V9_WIDTHS = V4_WIDTHS
# Triton requires tl.arange bounds to be powers of two.  At most 192 lanes are
# semantically active (rank 0, B8192); the remaining lanes stay masked/zero.
V4_MAX_STATS = 256
V8_WIDTHS = (3072, 1024, 1024)

_v4_column_index_cache: dict[tuple[int, str], torch.Tensor] = {}
_v8_column_index_cache: dict[tuple[int, str], torch.Tensor] = {}


def owner_geometry() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the installed runtime geometry after kernel capability check."""
    from .ag2_runtime_plan import get_runtime_plan

    plan = get_runtime_plan(required=False)
    if plan is None:
        # Standalone POCs load this module without a model runtime receipt.
        return WIDTHS, OFFSETS
    if plan.owner_widths != WIDTHS or plan.owner_offsets != OFFSETS:
        raise RuntimeError(
            "installed owner geometry exceeds the exact kernel capability: "
            f"plan={plan.owner_widths}/{plan.owner_offsets} "
            f"kernel={WIDTHS}/{OFFSETS}"
        )
    return plan.owner_widths, plan.owner_offsets


def v4_owner_column_indices(rank: int, device: torch.device) -> torch.Tensor:
    """Canonical hidden columns owned by one 6/5/5 FP4 block-class shard."""
    key = (rank, str(device))
    cached = _v4_column_index_cache.get(key)
    if cached is not None:
        return cached
    blocks = torch.arange(HIDDEN // 16, dtype=torch.int64, device=device)
    start = V4_CLASS_STARTS[rank]
    count = V4_CLASS_COUNTS[rank]
    selected = blocks[((blocks % 16) >= start) & ((blocks % 16) < start + count)]
    columns = (
        (
            selected[:, None] * 16
            + torch.arange(16, dtype=torch.int64, device=device)[None, :]
        )
        .reshape(-1)
        .contiguous()
    )
    if columns.numel() != V4_WIDTHS[rank]:
        raise RuntimeError("invalid v4 owner column geometry")
    _v4_column_index_cache[key] = columns
    return columns


def v9_owner_column_indices(rank: int, device: torch.device) -> torch.Tensor:
    return v4_owner_column_indices(rank, device)


def v1_owner_column_indices(rank: int, device: torch.device) -> torch.Tensor:
    return torch.arange(OFFSETS[rank], OFFSETS[rank] + WIDTHS[rank], device=device).to(
        torch.int64
    )


def v1b5_owner_column_indices(rank: int, device: torch.device) -> torch.Tensor:
    return v1_owner_column_indices(rank, device)


def v8_owner_column_indices(rank: int, device: torch.device) -> torch.Tensor:
    """B8192 reduction-subtree columns for 256/128/128 lane ownership."""
    key = (rank, str(device))
    cached = _v8_column_index_cache.get(key)
    if cached is not None:
        return cached
    if rank == 0:
        columns = torch.cat(
            (
                torch.arange(0, 2048, device=device),
                torch.arange(4096, HIDDEN, device=device),
            )
        )
    elif rank == 1:
        columns = torch.arange(2048, 3072, device=device)
    elif rank == 2:
        columns = torch.arange(3072, 4096, device=device)
    else:
        raise ValueError(f"invalid TP3 rank {rank}")
    columns = columns.to(torch.int64).contiguous()
    _v8_column_index_cache[key] = columns
    return columns


_owner_collective_workspace: torch.Tensor | None = None
_v10_lane_groups: tuple[ProcessGroup, ProcessGroup] | None = None
_v10_lane_workspaces: tuple[torch.Tensor, torch.Tensor] | None = None
_v10_lane_streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = None


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
    _owner_collective_workspace = scratch


def set_tp3_owner_v10_lane_resources(
    groups: tuple[ProcessGroup, ProcessGroup] | None,
    workspaces: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    """Install two non-aliasing communicator/workspace lanes for the v10 POC."""
    global _v10_lane_groups, _v10_lane_workspaces, _v10_lane_streams
    if groups is None or workspaces is None:
        _v10_lane_groups = None
        _v10_lane_workspaces = None
        _v10_lane_streams = None
        return
    if len(groups) != 2 or len(workspaces) != 2:
        raise ValueError("v10 requires exactly two groups and workspaces")
    if (
        workspaces[0].untyped_storage().data_ptr()
        == workspaces[1].untyped_storage().data_ptr()
    ):
        raise ValueError("v10 lane workspaces must not alias")
    for workspace in workspaces:
        if workspace.dtype != torch.bfloat16 or workspace.ndim != 2:
            raise ValueError("v10 lane workspace must be 2D BF16")
    device = workspaces[0].device
    if workspaces[1].device != device:
        raise ValueError("v10 lane workspaces must share one device")
    _v10_lane_groups = groups
    _v10_lane_workspaces = workspaces
    _v10_lane_streams = (
        torch.cuda.Stream(device=device),
        torch.cuda.Stream(device=device),
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


def _owner_collective_buffers_for_widths(
    contribution: torch.Tensor,
    rank: int,
    widths: tuple[int, int, int],
    scratch_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scratch = (
        scratch_override
        if scratch_override is not None
        else _owner_collective_workspace
    )
    if scratch is None:
        raise RuntimeError("TP3 owner workspace is enabled but not configured")
    rows = contribution.shape[0]
    send_elements = rows * HIDDEN
    receive_elements = WORLD * rows * widths[rank]
    required = send_elements + receive_elements
    flat = scratch.view(-1)
    if required > flat.numel():
        raise RuntimeError(
            f"TP3 owner workspace needs {required} elements, has {flat.numel()}"
        )
    return flat[:send_elements], flat[send_elements:required]


def _v4_rblock(rows: int) -> int:
    if rows <= 4:
        return 512
    if rows <= 64:
        return 1024
    return 8192


@triton.jit
def _v4_owner_thread_stats(
    input_ptr,
    residual_ptr,
    output_ptr,
    input_stride,
    residual_stride,
    output_stride,
    HIDDEN_SIZE: tl.constexpr,
    R_BLOCK: tl.constexpr,
    NUM_THREADS: tl.constexpr,
    CLASS_START: tl.constexpr,
    CLASS_COUNT: tl.constexpr,
    OWNER_THREADS: tl.constexpr,
    MAX_STATS: tl.constexpr,
    CANONICAL_OUTPUT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    local_thread = tl.arange(0, MAX_STATS)
    active = local_thread < OWNER_THREADS
    if R_BLOCK == 8192:
        values_per_thread: tl.constexpr = 8
    else:
        values_per_thread: tl.constexpr = R_BLOCK // NUM_THREADS
    threads_per_block: tl.constexpr = 16 // values_per_thread
    local_block = local_thread // threads_per_block
    thread_in_block = local_thread % threads_per_block
    global_block = (
        (local_block // CLASS_COUNT) * 16 + CLASS_START + local_block % CLASS_COUNT
    )
    global_thread = global_block * threads_per_block + thread_in_block
    acc = tl.zeros([MAX_STATS], dtype=tl.float32)
    if R_BLOCK == 8192:
        for lane in range(8):
            col = global_thread * 8 + lane
            block = col // 16
            local_owner_block = (block // 16) * CLASS_COUNT + (block % 16 - CLASS_START)
            local_col = local_owner_block * 16 + col % 16
            value = tl.load(
                input_ptr + row * input_stride + local_col,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + row * residual_stride + local_col,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            combined = value + residual
            acc += combined * combined
        for lane in range(8):
            col = 4096 + global_thread * 8 + lane
            valid = active & (col < HIDDEN_SIZE)
            block = col // 16
            local_owner_block = (block // 16) * CLASS_COUNT + (block % 16 - CLASS_START)
            local_col = local_owner_block * 16 + col % 16
            value = tl.load(
                input_ptr + row * input_stride + local_col,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + row * residual_stride + local_col,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            combined = value + residual
            acc += combined * combined
    else:
        for offset in range(0, HIDDEN_SIZE, R_BLOCK):
            for lane in range(values_per_thread):
                col = offset + global_thread * values_per_thread + lane
                valid = active & (col < HIDDEN_SIZE)
                block = col // 16
                local_owner_block = (block // 16) * CLASS_COUNT + (
                    block % 16 - CLASS_START
                )
                local_col = local_owner_block * 16 + col % 16
                value = tl.load(
                    input_ptr + row * input_stride + local_col,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                residual = tl.load(
                    residual_ptr + row * residual_stride + local_col,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                combined = value + residual
                acc += combined * combined
    if CANONICAL_OUTPUT:
        tl.store(output_ptr + row * output_stride + global_thread, acc, mask=active)
    else:
        tl.store(output_ptr + row * output_stride + local_thread, acc, mask=active)


@triton.jit
def _v4_fold_thread_stats(
    gathered_ptr,
    output_ptr,
    rank_stride,
    row_stride,
    R_BLOCK: tl.constexpr,
    NUM_THREADS: tl.constexpr,
    MAX_STATS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    thread = tl.arange(0, NUM_THREADS)
    if R_BLOCK == 8192:
        values_per_thread: tl.constexpr = 8
    else:
        values_per_thread: tl.constexpr = R_BLOCK // NUM_THREADS
    threads_per_block: tl.constexpr = 16 // values_per_thread
    block = thread // threads_per_block
    block_class = block % 16
    owner = tl.where(block_class < 6, 0, tl.where(block_class < 11, 1, 2))
    class_start = tl.where(owner == 0, 0, tl.where(owner == 1, 6, 11))
    class_count = tl.where(owner == 0, 6, 5)
    owner_block = (block // 16) * class_count + (block_class - class_start)
    owner_thread = owner_block * threads_per_block + thread % threads_per_block
    values = tl.load(
        gathered_ptr + owner * rank_stride + row * row_stride + owner_thread
    ).to(tl.float32)
    tl.store(output_ptr + row, tl.sum(values))


@triton.jit
def _v8_subtree_stats(
    input_ptr,
    residual_ptr,
    output_ptr,
    input_stride,
    residual_stride,
    RANK: tl.constexpr,
):
    """Reduce one complete B8192 native lane subtree to one FP32 root."""
    row = tl.program_id(0).to(tl.int64)
    thread = tl.arange(0, 512)
    if RANK == 0:
        active = thread < 256
        local_thread = thread
    elif RANK == 1:
        active = (thread >= 256) & (thread < 384)
        local_thread = thread - 256
    else:
        active = thread >= 384
        local_thread = thread - 384
    acc = tl.zeros([512], dtype=tl.float32)
    for lane in range(8):
        local_col = local_thread * 8 + lane
        value = tl.load(
            input_ptr + row * input_stride + local_col,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + row * residual_stride + local_col,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        combined = value + residual
        acc += combined * combined
    if RANK == 0:
        # Native B8192 reuses physical lanes 0..127 for columns 4096..5119.
        active_second = thread < 128
        for lane in range(8):
            local_col = 2048 + thread * 8 + lane
            value = tl.load(
                input_ptr + row * input_stride + local_col,
                mask=active_second,
                other=0.0,
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + row * residual_stride + local_col,
                mask=active_second,
                other=0.0,
            ).to(tl.float32)
            combined = value + residual
            acc += combined * combined
    tl.store(output_ptr + row, tl.sum(acc))


@triton.jit
def _v8_join_subtree_roots(gathered_ptr, output_ptr, rows):
    row = tl.program_id(0).to(tl.int64)
    active = row < rows
    root0 = tl.load(gathered_ptr + row, mask=active).to(tl.float32)
    root1 = tl.load(gathered_ptr + rows + row, mask=active).to(tl.float32)
    root2 = tl.load(gathered_ptr + 2 * rows + row, mask=active).to(tl.float32)
    tl.store(output_ptr + row, root0 + (root1 + root2), mask=active)


@triton.jit
def _v4_norm_owner(
    input_ptr,
    residual_ptr,
    weight_ptr,
    inv_rms_ptr,
    output_ptr,
    residual_out_ptr,
    input_stride,
    residual_stride,
    output_stride,
    residual_out_stride,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    value = tl.load(input_ptr + row * input_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    residual = tl.load(
        residual_ptr + row * residual_stride + cols, mask=mask, other=0.0
    ).to(tl.float32)
    combined = value + residual
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.load(inv_rms_ptr + row) / 5120.0 + eps)
    tl.store(
        residual_out_ptr + row * residual_out_stride + cols,
        combined,
        mask=mask,
    )
    tl.store(
        output_ptr + row * output_stride + cols,
        combined * inv_rms * weight,
        mask=mask,
    )


@triton.jit
def _v5_continue_thread_stats(
    input_ptr,
    residual_ptr,
    incoming_ptr,
    output_ptr,
    input_stride,
    residual_stride,
    stats_stride,
    OWNER_OFFSET: tl.constexpr,
    OWNER_WIDTH: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    R_BLOCK: tl.constexpr,
    NUM_THREADS: tl.constexpr,
    HAS_INCOMING: tl.constexpr,
):
    """Continue native per-thread accumulators over one contiguous owner."""
    row = tl.program_id(0).to(tl.int64)
    thread = tl.arange(0, NUM_THREADS)
    if HAS_INCOMING:
        acc = tl.load(incoming_ptr + row * stats_stride + thread).to(tl.float32)
    else:
        acc = tl.zeros([NUM_THREADS], dtype=tl.float32)
    if R_BLOCK == 8192:
        for lane in range(8):
            col = thread * 8 + lane
            valid = (col >= OWNER_OFFSET) & (col < OWNER_OFFSET + OWNER_WIDTH)
            local_col = col - OWNER_OFFSET
            value = tl.load(
                input_ptr + row * input_stride + local_col,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + row * residual_stride + local_col,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            combined = value + residual
            acc += combined * combined
        for lane in range(8):
            col = 4096 + thread * 8 + lane
            valid = (
                (col < HIDDEN_SIZE)
                & (col >= OWNER_OFFSET)
                & (col < OWNER_OFFSET + OWNER_WIDTH)
            )
            local_col = col - OWNER_OFFSET
            value = tl.load(
                input_ptr + row * input_stride + local_col,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + row * residual_stride + local_col,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            combined = value + residual
            acc += combined * combined
    else:
        values_per_thread: tl.constexpr = R_BLOCK // NUM_THREADS
        for offset in range(0, HIDDEN_SIZE, R_BLOCK):
            for lane in range(values_per_thread):
                col = offset + thread * values_per_thread + lane
                valid = (
                    (col < HIDDEN_SIZE)
                    & (col >= OWNER_OFFSET)
                    & (col < OWNER_OFFSET + OWNER_WIDTH)
                )
                local_col = col - OWNER_OFFSET
                value = tl.load(
                    input_ptr + row * input_stride + local_col,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                residual = tl.load(
                    residual_ptr + row * residual_stride + local_col,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                combined = value + residual
                acc += combined * combined
    tl.store(output_ptr + row * stats_stride + thread, acc)


@triton.jit
def _v5_fold_thread_stats(
    stats_ptr,
    output_ptr,
    stats_stride,
    NUM_THREADS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    thread = tl.arange(0, NUM_THREADS)
    values = tl.load(stats_ptr + row * stats_stride + thread).to(tl.float32)
    tl.store(output_ptr + row, tl.sum(values))


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
def _native_lane_rms_norm(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_out_ptr,
    input_stride,
    residual_stride,
    output_stride,
    residual_out_stride,
    eps,
    STORE_RESIDUAL: tl.constexpr,
):
    """Match Inductor's accepted 512-lane/10-chunk RMS reduction tree."""
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, 512)
    sum_sq = tl.zeros([512], dtype=tl.float32)
    for chunk in range(10):
        cols = chunk * 512 + lanes
        value = tl.load(input_ptr + row * input_stride + cols).to(tl.float32)
        residual = tl.load(residual_ptr + row * residual_stride + cols).to(tl.float32)
        combined = value + residual
        sum_sq += combined * combined
    inv_rms = tl.rsqrt(tl.sum(sum_sq) / 5120.0 + eps)
    for chunk in range(10):
        cols = chunk * 512 + lanes
        value = tl.load(input_ptr + row * input_stride + cols).to(tl.float32)
        residual = tl.load(residual_ptr + row * residual_stride + cols).to(tl.float32)
        combined = value + residual
        weight = tl.load(weight_ptr + cols).to(tl.float32)
        if STORE_RESIDUAL:
            tl.store(
                residual_out_ptr + row * residual_out_stride + cols,
                combined,
            )
        tl.store(
            output_ptr + row * output_stride + cols,
            combined * inv_rms * weight,
        )


@triton.jit
def _native_config_rms_norm(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_out_ptr,
    input_stride,
    residual_stride,
    output_stride,
    residual_out_stride,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    """Diagnostic copy of the native 2-D Inductor reduction family."""
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, R_BLOCK)[None, :]
    sum_sq = tl.zeros([1, R_BLOCK], dtype=tl.float32)
    for offset in range(0, HIDDEN_SIZE, R_BLOCK):
        cols = offset + lanes
        mask = cols < HIDDEN_SIZE
        value = tl.load(input_ptr + row * input_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        residual = tl.load(
            residual_ptr + row * residual_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        combined = value + residual
        sum_sq += combined * combined
    inv_rms = tl.rsqrt(tl.sum(sum_sq, axis=1) / 5120.0 + eps)
    for offset in range(0, HIDDEN_SIZE, R_BLOCK):
        cols = offset + lanes
        mask = cols < HIDDEN_SIZE
        value = tl.load(input_ptr + row * input_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        residual = tl.load(
            residual_ptr + row * residual_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        combined = value + residual
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            residual_out_ptr + row * residual_out_stride + cols,
            combined,
            mask=mask,
        )
        tl.store(
            output_ptr + row * output_stride + cols,
            combined * inv_rms * weight,
            mask=mask,
        )


@triton.jit
def _owner_lane_stats(
    input_ptr,
    residual_ptr,
    stats_ptr,
    input_stride,
    residual_stride,
    stats_stride,
    N_CHUNKS: tl.constexpr,
):
    """Keep the native 512 lanes intact across an owner partition."""
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, 512)
    sum_sq = tl.zeros([512], dtype=tl.float32)
    for chunk in range(N_CHUNKS):
        cols = chunk * 512 + lanes
        value = tl.load(input_ptr + row * input_stride + cols).to(tl.float32)
        residual = tl.load(residual_ptr + row * residual_stride + cols).to(tl.float32)
        combined = value + residual
        sum_sq += combined * combined
    tl.store(stats_ptr + row * stats_stride + lanes, sum_sq)


@triton.jit
def _owner_norm_lane_stats(
    input_ptr,
    residual_ptr,
    weight_ptr,
    gathered_stats_ptr,
    output_ptr,
    residual_out_ptr,
    input_stride,
    residual_stride,
    stats_rank_stride,
    stats_row_stride,
    output_stride,
    residual_out_stride,
    eps,
    N_CHUNKS: tl.constexpr,
):
    """Fold owner lane accumulators in rank order before the native tl.sum."""
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, 512)
    rank0 = tl.load(gathered_stats_ptr + row * stats_row_stride + lanes).to(tl.float32)
    rank1 = tl.load(
        gathered_stats_ptr + stats_rank_stride + row * stats_row_stride + lanes
    ).to(tl.float32)
    rank2 = tl.load(
        gathered_stats_ptr + 2 * stats_rank_stride + row * stats_row_stride + lanes
    ).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum((rank0 + rank1) + rank2) / 5120.0 + eps)
    for chunk in range(N_CHUNKS):
        cols = chunk * 512 + lanes
        value = tl.load(input_ptr + row * input_stride + cols).to(tl.float32)
        residual = tl.load(residual_ptr + row * residual_stride + cols).to(tl.float32)
        combined = value + residual
        weight = tl.load(weight_ptr + cols).to(tl.float32)
        tl.store(
            residual_out_ptr + row * residual_out_stride + cols,
            combined,
        )
        tl.store(
            output_ptr + row * output_stride + cols,
            combined * inv_rms * weight,
        )


@triton.jit
def _fixed_sum_owner_stats(
    gathered_ptr,
    residual_ptr,
    owned_ptr,
    stats_ptr,
    input_stride,
    residual_stride,
    stats_stride,
    source_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse the fixed TP3 sum with stats while preserving the BF16 seam."""
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = row * input_stride + cols
    rank0 = tl.load(gathered_ptr + offsets).to(tl.float32)
    rank1 = tl.load(gathered_ptr + source_elements + offsets).to(tl.float32)
    rank2 = tl.load(gathered_ptr + 2 * source_elements + offsets).to(tl.float32)
    summed = (rank0 + rank1) + rank2
    # The accepted contract rounds the fixed FP32 sum to BF16 before the
    # residual/RMSNorm consumer.  Keep that rounding even though both phases
    # share one kernel.
    rounded = summed.to(tl.bfloat16)
    tl.store(owned_ptr + offsets, rounded)
    combined = rounded.to(tl.float32) + tl.load(
        residual_ptr + row * residual_stride + cols
    ).to(tl.float32)
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


@triton.jit
def _v4_scatter_packed_blocks(
    received_ptr,
    base_q_ptr,
    augmented_q_ptr,
    base_sf_ptr,
    augmented_sf_ptr,
    payload1_offset,
    payload2_offset,
    augmented_q_stride,
    augmented_sf_stride,
    ROWS: tl.constexpr,
    Q_BLOCK: tl.constexpr,
):
    """Scatter three block-class payloads in one row kernel."""
    row = tl.program_id(0).to(tl.int64)
    q_col = tl.arange(0, Q_BLOCK)
    q_mask = q_col < 2560
    block = q_col // 8
    block_class = block % 16
    source = tl.where(block_class < 6, 0, tl.where(block_class < 11, 1, 2))
    class_start = tl.where(source == 0, 0, tl.where(source == 1, 6, 11))
    class_count = tl.where(source == 0, 6, 5)
    local_block = (block // 16) * class_count + block_class - class_start
    q_width = tl.where(source == 0, 960, 800)
    sf_width = tl.where(source == 0, 120, 100)
    payload_offset = tl.where(
        source == 0,
        0,
        tl.where(source == 1, payload1_offset, payload2_offset),
    )
    q_value = tl.load(
        received_ptr + payload_offset + row * q_width + local_block * 8 + q_col % 8,
        mask=q_mask,
    )
    tl.store(base_q_ptr + row * 2560 + q_col, q_value, mask=q_mask)
    tl.store(
        augmented_q_ptr + row * augmented_q_stride + q_col,
        q_value,
        mask=q_mask,
    )

    sf_mask = q_col < 320
    sf_block = q_col
    sf_class = sf_block % 16
    sf_source = tl.where(sf_class < 6, 0, tl.where(sf_class < 11, 1, 2))
    sf_start = tl.where(sf_source == 0, 0, tl.where(sf_source == 1, 6, 11))
    sf_count = tl.where(sf_source == 0, 6, 5)
    sf_local_block = (sf_block // 16) * sf_count + sf_class - sf_start
    sf_q_width = tl.where(sf_source == 0, 960, 800)
    sf_owner_width = tl.where(sf_source == 0, 120, 100)
    sf_payload_offset = tl.where(
        sf_source == 0,
        0,
        tl.where(sf_source == 1, payload1_offset, payload2_offset),
    )
    sf_value = tl.load(
        received_ptr
        + sf_payload_offset
        + ROWS * sf_q_width
        + row * sf_owner_width
        + sf_local_block,
        mask=sf_mask,
    )
    tl.store(base_sf_ptr + row * 320 + q_col, sf_value, mask=sf_mask)
    tl.store(
        augmented_sf_ptr + row * augmented_sf_stride + q_col,
        sf_value,
        mask=sf_mask,
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


def _reduce_and_norm_v2(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V2: fuse fixed sum and owner stats without changing exact semantics."""
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
    owned = torch.empty(
        (rows, WIDTHS[rank]), dtype=contribution.dtype, device=contribution.device
    )
    local_stats = torch.zeros((rows, 2), dtype=torch.float32, device=owned.device)
    _fixed_sum_owner_stats[(rows, BLOCKS[rank])](
        receive,
        residual_owner,
        owned,
        local_stats,
        owned.stride(0),
        residual_owner.stride(0),
        local_stats.stride(0),
        owned_elements,
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


def _reduce_and_norm_v3_lane512(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Owner transport with a 512-lane RMS tree close to native Inductor."""
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
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[owned_elements] * WORLD,
        input_split_sizes=elements,
        group=group,
    )
    owned = fixed_tp3_sum_fused(receive.view(WORLD, rows, WIDTHS[rank]))
    local_lanes = torch.empty((rows, 512), dtype=torch.float32, device=owned.device)
    _owner_lane_stats[(rows,)](
        owned,
        residual_owner,
        local_lanes,
        owned.stride(0),
        residual_owner.stride(0),
        local_lanes.stride(0),
        N_CHUNKS=WIDTHS[rank] // 512,
        num_warps=8,
    )
    gathered = torch.empty((WORLD, rows, 512), dtype=torch.float32, device=owned.device)
    dist.all_gather_single(gathered.view(WORLD * rows, 512), local_lanes, group=group)
    norm_owner = torch.empty_like(owned)
    residual_out = torch.empty_like(owned)
    weight_owner = weight[OFFSETS[rank] : OFFSETS[rank] + WIDTHS[rank]].contiguous()
    _owner_norm_lane_stats[(rows,)](
        owned,
        residual_owner,
        weight_owner,
        gathered,
        norm_owner,
        residual_out,
        owned.stride(0),
        residual_owner.stride(0),
        gathered.stride(0),
        gathered.stride(1),
        norm_owner.stride(0),
        residual_out.stride(0),
        eps,
        N_CHUNKS=WIDTHS[rank] // 512,
        num_warps=8,
    )
    return norm_owner, residual_out


def _reduce_and_norm_v4_block_class(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
    *,
    compact_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact native-tree RMS over 6/5/5 NVFP4 block-class owners."""
    if contribution.ndim != 2 or contribution.shape[1] != HIDDEN:
        raise ValueError(f"owner seam requires [M,{HIDDEN}], got {contribution.shape}")
    rank = dist.get_rank(group)
    rows = contribution.shape[0]
    width = V4_WIDTHS[rank]
    if residual_owner.shape != (rows, width):
        raise ValueError(
            f"rank {rank} v4 residual shape {residual_owner.shape} != {(rows, width)}"
        )
    indices = [
        v4_owner_column_indices(owner, contribution.device) for owner in range(WORLD)
    ]
    chunks = [contribution.index_select(1, index) for index in indices]
    send, receive = _owner_collective_buffers_for_widths(contribution, rank, V4_WIDTHS)
    elements = [rows * owner_width for owner_width in V4_WIDTHS]
    offset = 0
    for chunk, element_count, owner_width in zip(
        chunks, elements, V4_WIDTHS, strict=True
    ):
        send[offset : offset + element_count].view(rows, owner_width).copy_(chunk)
        offset += element_count
    owned_elements = elements[rank]
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[owned_elements] * WORLD,
        input_split_sizes=elements,
        group=group,
    )
    owned = fixed_tp3_sum_fused(receive.view(WORLD, rows, width))
    r_block = _v4_rblock(rows)
    num_threads = 512 if r_block >= 4096 else 256
    values_per_thread = 8 if r_block == 8192 else r_block // num_threads
    threads_per_block = 16 // values_per_thread
    blocks_per_rblock = 256 if r_block == 8192 else r_block // 16
    owner_blocks = blocks_per_rblock // 16 * V4_CLASS_COUNTS[rank]
    owner_threads = owner_blocks * threads_per_block
    stats_width = 128 if compact_stats and r_block != 8192 else V4_MAX_STATS
    if owner_threads > stats_width:
        raise RuntimeError(
            f"v4 owner stats need {owner_threads} lanes, have {stats_width}"
        )
    local_stats = torch.zeros(
        (rows, stats_width), dtype=torch.float32, device=owned.device
    )
    _v4_owner_thread_stats[(rows,)](
        owned,
        residual_owner,
        local_stats,
        owned.stride(0),
        residual_owner.stride(0),
        local_stats.stride(0),
        HIDDEN_SIZE=HIDDEN,
        R_BLOCK=r_block,
        NUM_THREADS=num_threads,
        CLASS_START=V4_CLASS_STARTS[rank],
        CLASS_COUNT=V4_CLASS_COUNTS[rank],
        OWNER_THREADS=owner_threads,
        MAX_STATS=stats_width,
        CANONICAL_OUTPUT=False,
        num_warps=8,
    )
    gathered = torch.empty(
        (WORLD, rows, stats_width), dtype=torch.float32, device=owned.device
    )
    dist.all_gather_single(
        gathered.view(WORLD * rows, stats_width), local_stats, group=group
    )
    sum_sq = torch.empty(rows, dtype=torch.float32, device=owned.device)
    _v4_fold_thread_stats[(rows,)](
        gathered,
        sum_sq,
        gathered.stride(0),
        gathered.stride(1),
        R_BLOCK=r_block,
        NUM_THREADS=num_threads,
        MAX_STATS=stats_width,
        num_warps=16 if num_threads == 512 else 8,
    )
    norm_owner = torch.empty_like(owned)
    residual_out = torch.empty_like(owned)
    weight_owner = weight.index_select(0, indices[rank]).contiguous()
    _v4_norm_owner[(rows,)](
        owned,
        residual_owner,
        weight_owner,
        sum_sq,
        norm_owner,
        residual_out,
        owned.stride(0),
        residual_owner.stride(0),
        norm_owner.stride(0),
        residual_out.stride(0),
        width,
        eps,
        BLOCK_SIZE=2048,
        num_warps=16,
    )
    return norm_owner, residual_out


def _reduce_and_norm_v9_disjoint_frontier(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
    *,
    scratch_override: torch.Tensor | None = None,
    r_block_override: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact v4 native frontier represented as one disjoint all-reduce.

    Every canonical native-thread slot is produced by exactly one owner.  The
    other ranks contribute zero to that slot, so the collective distributes
    leaves without changing their FP32 values or the final native ``tl.sum``
    tree.
    """
    if contribution.ndim != 2 or contribution.shape[1] != HIDDEN:
        raise ValueError(f"owner seam requires [M,{HIDDEN}], got {contribution.shape}")
    rank = dist.get_rank(group)
    rows = contribution.shape[0]
    width = V4_WIDTHS[rank]
    if residual_owner.shape != (rows, width):
        raise ValueError(
            f"rank {rank} v9 residual shape {residual_owner.shape} != {(rows, width)}"
        )
    indices = [
        v4_owner_column_indices(owner, contribution.device) for owner in range(WORLD)
    ]
    chunks = [contribution.index_select(1, index) for index in indices]
    send, receive = _owner_collective_buffers_for_widths(
        contribution, rank, V4_WIDTHS, scratch_override
    )
    elements = [rows * owner_width for owner_width in V4_WIDTHS]
    offset = 0
    for chunk, element_count, owner_width in zip(
        chunks, elements, V4_WIDTHS, strict=True
    ):
        send[offset : offset + element_count].view(rows, owner_width).copy_(chunk)
        offset += element_count
    owned_elements = elements[rank]
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[owned_elements] * WORLD,
        input_split_sizes=elements,
        group=group,
    )
    owned = fixed_tp3_sum_fused(receive.view(WORLD, rows, width))
    r_block = r_block_override if r_block_override is not None else _v4_rblock(rows)
    num_threads = 512 if r_block >= 4096 else 256
    values_per_thread = 8 if r_block == 8192 else r_block // num_threads
    threads_per_block = 16 // values_per_thread
    blocks_per_rblock = 256 if r_block == 8192 else r_block // 16
    owner_blocks = blocks_per_rblock // 16 * V4_CLASS_COUNTS[rank]
    owner_threads = owner_blocks * threads_per_block

    canonical_stats = torch.zeros(
        (rows, num_threads), dtype=torch.float32, device=owned.device
    )
    _v4_owner_thread_stats[(rows,)](
        owned,
        residual_owner,
        canonical_stats,
        owned.stride(0),
        residual_owner.stride(0),
        canonical_stats.stride(0),
        HIDDEN_SIZE=HIDDEN,
        R_BLOCK=r_block,
        NUM_THREADS=num_threads,
        CLASS_START=V4_CLASS_STARTS[rank],
        CLASS_COUNT=V4_CLASS_COUNTS[rank],
        OWNER_THREADS=owner_threads,
        MAX_STATS=num_threads,
        CANONICAL_OUTPUT=True,
        num_warps=8,
    )
    dist.all_reduce(canonical_stats, op=dist.ReduceOp.SUM, group=group)
    sum_sq = torch.empty(rows, dtype=torch.float32, device=owned.device)
    _v5_fold_thread_stats[(rows,)](
        canonical_stats,
        sum_sq,
        canonical_stats.stride(0),
        NUM_THREADS=num_threads,
        num_warps=16 if num_threads == 512 else 8,
    )
    norm_owner = torch.empty_like(owned)
    residual_out = torch.empty_like(owned)
    weight_owner = weight.index_select(0, indices[rank]).contiguous()
    _v4_norm_owner[(rows,)](
        owned,
        residual_owner,
        weight_owner,
        sum_sq,
        norm_owner,
        residual_out,
        owned.stride(0),
        residual_owner.stride(0),
        norm_owner.stride(0),
        residual_out.stride(0),
        width,
        eps,
        BLOCK_SIZE=2048,
        num_warps=16,
    )
    return norm_owner, residual_out


def _reduce_and_norm_v8_upper_tree(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact B8192 RMS with complete 256/128/128 reduction subtrees."""
    if contribution.ndim != 2 or contribution.shape[1] != HIDDEN:
        raise ValueError(f"owner seam requires [M,{HIDDEN}], got {contribution.shape}")
    rank = dist.get_rank(group)
    rows = contribution.shape[0]
    width = V8_WIDTHS[rank]
    if residual_owner.shape != (rows, width):
        raise ValueError(
            f"rank {rank} v8 residual shape {residual_owner.shape} != {(rows, width)}"
        )
    if _v4_rblock(rows) != 8192:
        raise ValueError("v8 upper-tree ownership is valid only for B8192 shapes")
    indices = [
        v8_owner_column_indices(owner, contribution.device) for owner in range(WORLD)
    ]
    chunks = [contribution.index_select(1, index) for index in indices]
    send, receive = _owner_collective_buffers_for_widths(contribution, rank, V8_WIDTHS)
    elements = [rows * owner_width for owner_width in V8_WIDTHS]
    offset = 0
    for chunk, element_count, owner_width in zip(
        chunks, elements, V8_WIDTHS, strict=True
    ):
        send[offset : offset + element_count].view(rows, owner_width).copy_(chunk)
        offset += element_count
    owned_elements = elements[rank]
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[owned_elements] * WORLD,
        input_split_sizes=elements,
        group=group,
    )
    owned = fixed_tp3_sum_fused(receive.view(WORLD, rows, width))
    local_root = torch.empty(rows, dtype=torch.float32, device=owned.device)
    _v8_subtree_stats[(rows,)](
        owned,
        residual_owner,
        local_root,
        owned.stride(0),
        residual_owner.stride(0),
        RANK=rank,
        num_warps=16,
    )
    gathered = torch.empty((WORLD, rows), dtype=torch.float32, device=owned.device)
    dist.all_gather_single(gathered, local_root, group=group)
    sum_sq = torch.empty(rows, dtype=torch.float32, device=owned.device)
    _v8_join_subtree_roots[(rows,)](gathered, sum_sq, rows, num_warps=1)
    norm_owner = torch.empty_like(owned)
    residual_out = torch.empty_like(owned)
    weight_owner = weight.index_select(0, indices[rank]).contiguous()
    _v4_norm_owner[(rows,)](
        owned,
        residual_owner,
        weight_owner,
        sum_sq,
        norm_owner,
        residual_out,
        owned.stride(0),
        residual_owner.stride(0),
        norm_owner.stride(0),
        residual_out.stride(0),
        width,
        eps,
        BLOCK_SIZE=4096,
        num_warps=16,
    )
    return norm_owner, residual_out


def v4_reduce_and_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public POC entry point for the v4 norm seam."""
    return _reduce_and_norm_v4_block_class(
        contribution, residual_owner, weight, group, eps
    )


def v1_reduce_and_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _reduce_and_norm(contribution, residual_owner, weight, group, eps)


def v1b5_reduce_and_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _reduce_and_norm(contribution, residual_owner, weight, group, eps)


def v8_reduce_and_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public POC entry point for the B8192 subtree-owner norm seam."""
    return _reduce_and_norm_v8_upper_tree(
        contribution, residual_owner, weight, group, eps
    )


def v9_reduce_and_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _reduce_and_norm_v9_disjoint_frontier(
        contribution, residual_owner, weight, group, eps
    )


def _reduce_and_norm_v5_contiguous_pipeline(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous owners with exact rank0 -> rank1 -> rank2 RMS accumulators."""
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
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[owned_elements] * WORLD,
        input_split_sizes=elements,
        group=group,
    )
    owned = fixed_tp3_sum_fused(receive.view(WORLD, rows, WIDTHS[rank]))
    r_block = _v4_rblock(rows)
    num_threads = 512 if r_block >= 4096 else 256
    incoming = torch.empty(
        (rows, num_threads), dtype=torch.float32, device=owned.device
    )
    outgoing = torch.empty_like(incoming)
    if rank > 0:
        source = dist.get_global_rank(group, rank - 1)
        request = dist.irecv(incoming, src=source, group=group)
        request.wait()
    _v5_continue_thread_stats[(rows,)](
        owned,
        residual_owner,
        incoming,
        outgoing,
        owned.stride(0),
        residual_owner.stride(0),
        outgoing.stride(0),
        OWNER_OFFSET=OFFSETS[rank],
        OWNER_WIDTH=WIDTHS[rank],
        HIDDEN_SIZE=HIDDEN,
        R_BLOCK=r_block,
        NUM_THREADS=num_threads,
        HAS_INCOMING=rank > 0,
        num_warps=16 if num_threads == 512 else 8,
    )
    if rank < WORLD - 1:
        destination = dist.get_global_rank(group, rank + 1)
        request = dist.isend(outgoing, dst=destination, group=group)
        request.wait()
    sum_sq = torch.empty(rows, dtype=torch.float32, device=owned.device)
    if rank == WORLD - 1:
        _v5_fold_thread_stats[(rows,)](
            outgoing,
            sum_sq,
            outgoing.stride(0),
            NUM_THREADS=num_threads,
            num_warps=16 if num_threads == 512 else 8,
        )
    dist.broadcast(
        sum_sq,
        src=dist.get_global_rank(group, WORLD - 1),
        group=group,
    )
    norm_owner = torch.empty_like(owned)
    residual_out = torch.empty_like(owned)
    weight_owner = weight[OFFSETS[rank] : OFFSETS[rank] + WIDTHS[rank]].contiguous()
    _v4_norm_owner[(rows,)](
        owned,
        residual_owner,
        weight_owner,
        sum_sq,
        norm_owner,
        residual_out,
        owned.stride(0),
        residual_owner.stride(0),
        norm_owner.stride(0),
        residual_out.stride(0),
        WIDTHS[rank],
        eps,
        BLOCK_SIZE=2048,
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
    reduce_and_norm=_reduce_and_norm,
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
    norm_owner, residual_out = reduce_and_norm(
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
    if selected_count == 0:
        return base_q, base_sf, base_q, base_sf, residual_out
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


def _owner_residual_arc_prequant_v4_block_class_direct(
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
    grouped_output: bool = False,
    compact_stats: bool = False,
    norm_impl=None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize complete block-class owners and scatter packed blocks at peers."""
    rank = dist.get_rank(group)
    rows = contribution.shape[0]
    selected_count = selected_all.shape[1]
    if selected_all.shape != (WORLD, selected_count) or len(route_counts) != 9:
        raise ValueError("invalid v4 ARC route metadata")
    if norm_impl is None:
        norm_owner, residual_out = _reduce_and_norm_v4_block_class(
            contribution,
            residual_owner,
            weight,
            group,
            eps,
            compact_stats=compact_stats,
        )
    else:
        norm_owner, residual_out = norm_impl(
            contribution, residual_owner, weight, group, eps
        )
    owner_q, owner_sf = scaled_fp4_quant(
        norm_owner,
        input_scale_inv,
        is_sf_swizzled_layout=True,
        backend="b12x",
    )
    owner_sf = convert_swizzled_to_linear(
        owner_sf, rows, V4_WIDTHS[rank], 16
    ).contiguous()
    q_elements = [rows * width // 2 for width in V4_WIDTHS]
    sf_elements = [rows * width // 16 for width in V4_WIDTHS]
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
        base_elements[source] + rows * route_counts[source * WORLD + rank] * 2
        for source in range(WORLD)
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
    q_parts = []
    sf_parts = []
    selected_parts = []
    for source, payload in enumerate(received.split(output_splits)):
        if grouped_output:
            base_source = payload[: base_elements[source]]
            q_parts.append(
                base_source[: q_elements[source]].view(rows, V4_WIDTHS[source] // 2)
            )
            sf_parts.append(
                base_source[q_elements[source] :]
                .view(torch.float8_e4m3fn)
                .view(rows, V4_WIDTHS[source] // 16)
            )
        selected_parts.append(
            payload[base_elements[source] :]
            .view(contribution.dtype)
            .view(rows, route_counts[source * WORLD + rank])
        )
    selected_values = (
        torch.cat(selected_parts, dim=1).index_select(1, inverse_order).contiguous()
    )
    augmented_k = HIDDEN + selected_count
    base_q = (
        torch.cat(q_parts, dim=1).contiguous()
        if grouped_output
        else torch.empty(
            (rows, HIDDEN // 2), dtype=torch.uint8, device=contribution.device
        )
    )
    q = torch.empty(
        (rows, augmented_k // 2), dtype=torch.uint8, device=contribution.device
    )
    base_sf_linear = (
        torch.cat(sf_parts, dim=1).contiguous()
        if grouped_output
        else torch.empty(
            (rows, HIDDEN // 16),
            dtype=torch.float8_e4m3fn,
            device=contribution.device,
        )
    )
    sf_linear = torch.empty(
        (rows, augmented_k // 16),
        dtype=torch.float8_e4m3fn,
        device=contribution.device,
    )
    if grouped_output:
        q[:, : HIDDEN // 2].copy_(base_q)
        sf_linear[:, : HIDDEN // 16].copy_(base_sf_linear)
    else:
        _v4_scatter_packed_blocks[(rows,)](
            received,
            base_q,
            q,
            base_sf_linear.view(torch.uint8),
            sf_linear.view(torch.uint8),
            output_splits[0],
            output_splits[0] + output_splits[1],
            q.stride(0),
            sf_linear.stride(0),
            ROWS=rows,
            Q_BLOCK=4096,
            num_warps=16,
        )
    base_sf = swizzle_blockscale(base_sf_linear.contiguous())
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
    return q, sf, base_q, base_sf, residual_out


def _full_reduce_and_norm(
    contribution: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accepted exact TP3 reduction plus native Inductor RMS arithmetic."""
    if contribution.ndim != 2 or contribution.shape[1] != HIDDEN:
        raise ValueError(f"full seam requires [M,{HIDDEN}], got {contribution.shape}")
    if residual.shape != contribution.shape:
        raise ValueError("full seam residual must match contribution")
    reduced = exact_weighted_owner_992_reduce(contribution, group)
    return native_fused_add_rms_norm(reduced, residual, weight, eps)


def native_fused_add_rms_norm(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the accepted generated 512-lane/10-chunk BF16 RMS seam."""
    if value.ndim != 2 or value.shape[1] != HIDDEN:
        raise ValueError(f"native RMS requires [M,{HIDDEN}], got {value.shape}")
    if residual.shape != value.shape or weight.shape != (HIDDEN,):
        raise ValueError("native RMS residual/weight shape mismatch")
    normalized = torch.empty_like(value)
    residual_out = torch.empty_like(value)
    _native_lane_rms_norm[(value.shape[0],)](
        value,
        residual,
        weight,
        normalized,
        residual_out,
        value.stride(0),
        residual.stride(0),
        normalized.stride(0),
        residual_out.stride(0),
        eps,
        STORE_RESIDUAL=True,
        num_warps=8,
    )
    return normalized, residual_out


def v1_full_block5_fused_add_rms_norm(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialized equivalent of contiguous owner-v1's five-root RMS tree."""
    if value.ndim != 2 or value.shape[1] != HIDDEN:
        raise ValueError(f"block5 RMS requires [M,{HIDDEN}], got {value.shape}")
    if residual.shape != value.shape or weight.shape != (HIDDEN,):
        raise ValueError("block5 RMS residual/weight shape mismatch")
    rows = value.shape[0]
    stats = torch.empty(
        (rows, HIDDEN // BLOCK), dtype=torch.float32, device=value.device
    )
    _owner_stats[(rows, HIDDEN // BLOCK)](
        value,
        residual,
        stats,
        value.stride(0),
        residual.stride(0),
        stats.stride(0),
        N_BLOCKS=HIDDEN // BLOCK,
        BLOCK_SIZE=BLOCK,
        num_warps=16,
    )
    normalized = torch.empty_like(value)
    residual_out = torch.empty_like(value)
    _owner_norm[(rows, HIDDEN // BLOCK)](
        value,
        residual,
        weight,
        stats,
        normalized,
        residual_out,
        value.stride(0),
        residual.stride(0),
        stats.stride(0),
        normalized.stride(0),
        residual_out.stride(0),
        eps,
        N_BLOCKS=HIDDEN // BLOCK,
        HIDDEN_SIZE=HIDDEN,
        BLOCK_SIZE=BLOCK,
        num_warps=16,
    )
    return normalized, residual_out


def native_fused_add_rms_norm_config(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    r_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one explicit native Inductor RMS configuration for controls."""
    if r_block not in (512, 1024, 4096, 8192):
        raise ValueError(f"unsupported native RMS R_BLOCK={r_block}")
    if value.ndim != 2 or value.shape[1] != HIDDEN:
        raise ValueError(f"native RMS requires [M,{HIDDEN}], got {value.shape}")
    if residual.shape != value.shape or weight.shape != (HIDDEN,):
        raise ValueError("native RMS residual/weight shape mismatch")
    normalized = torch.empty_like(value)
    residual_out = torch.empty_like(value)
    _native_config_rms_norm[(value.shape[0],)](
        value,
        residual,
        weight,
        normalized,
        residual_out,
        value.stride(0),
        residual.stride(0),
        normalized.stride(0),
        residual_out.stride(0),
        eps,
        HIDDEN_SIZE=HIDDEN,
        R_BLOCK=r_block,
        num_warps=16 if r_block >= 4096 else 8,
    )
    return normalized, residual_out


def full_residual_arc_prequant(
    contribution: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    input_scale_inv: torch.Tensor,
    selected: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized, residual_out = _full_reduce_and_norm(
        contribution, residual, weight, group, eps
    )
    base_q, base_sf = scaled_fp4_quant(
        normalized,
        input_scale_inv,
        is_sf_swizzled_layout=True,
        backend="b12x",
    )
    if selected.numel() == 0:
        return base_q, base_sf, base_q, base_sf, residual_out
    arc_q, arc_sf = ag2_nvfp4_arc_quantize(normalized, input_scale_inv, selected)
    return arc_q, arc_sf, base_q, base_sf, residual_out


def full_terminal_norm(
    contribution: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    normalized, _ = _full_reduce_and_norm(contribution, residual, weight, group, eps)
    return normalized


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
    """Execute one owner seam; the enclosing vLLM graph owns replay.

    This module deliberately has no child graphs and no tensor-pointer cache.
    When a FULL target graph is active, vLLM captures these operations once and
    keys replay by its bounded semantic ``BatchDescriptor``.
    """
    return _owner_residual_arc_prequant_direct(
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


def owner_residual_arc_prequant_v2(
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
    """Owner v2 with fused fixed-sum/statistics producer."""
    return _owner_residual_arc_prequant_direct(
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
        _reduce_and_norm_v2,
    )


def owner_residual_arc_prequant_v3_lane512(
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
    """Owner v3: retain native 512 reduction lanes across ranks."""
    return _owner_residual_arc_prequant_direct(
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
        _reduce_and_norm_v3_lane512,
    )


def owner_residual_arc_prequant_v4_block_class(
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
    """Owner v4: native-tree RMS plus complete NVFP4 block-class ownership."""
    return _owner_residual_arc_prequant_v4_block_class_direct(
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


def owner_residual_arc_prequant_v5_contiguous_pipeline(
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
    """Owner v5: contiguous ARC layout with exact sequential RMS handoff."""
    return _owner_residual_arc_prequant_direct(
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
        _reduce_and_norm_v5_contiguous_pipeline,
    )


def owner_residual_arc_prequant_v6_grouped_weight(
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
    """Owner v6: grouped block-class activation for startup-permuted weights."""
    return _owner_residual_arc_prequant_v4_block_class_direct(
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
        grouped_output=True,
    )


def owner_residual_arc_prequant_v7_compact_stats(
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
    """Owner v7: v6 grouped layout with shape-tight exact-stat payloads."""
    return _owner_residual_arc_prequant_v4_block_class_direct(
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
        grouped_output=True,
        compact_stats=True,
    )


def owner_residual_arc_prequant_v9_disjoint_frontier(
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
    """Owner v9: grouped v6 ARC with one disjoint exact RMS frontier."""
    return _owner_residual_arc_prequant_v4_block_class_direct(
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
        grouped_output=True,
        norm_impl=_reduce_and_norm_v9_disjoint_frontier,
    )


def owner_residual_arc_prequant_v10_two_lane(
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
    """Two request-aligned v9 waves on independent streams/communicators."""
    del group
    if (
        _v10_lane_groups is None
        or _v10_lane_workspaces is None
        or _v10_lane_streams is None
    ):
        raise RuntimeError("v10 lane resources are not configured")
    rows = contribution.shape[0]
    if rows < 8 or rows % 4:
        raise ValueError("v10 requires at least two qlen4 requests")
    first_rows = ((rows // 4 + 1) // 2) * 4
    boundaries = (0, first_rows, rows)
    parent_r_block = _v4_rblock(rows)
    current = torch.cuda.current_stream(contribution.device)
    results = []
    for lane, stream in enumerate(_v10_lane_streams):
        begin, end = boundaries[lane], boundaries[lane + 1]
        stream.wait_stream(current)
        lane_group = _v10_lane_groups[lane]
        lane_workspace = _v10_lane_workspaces[lane]

        def lane_norm(
            lane_contribution: torch.Tensor,
            lane_residual: torch.Tensor,
            lane_weight: torch.Tensor,
            norm_group: ProcessGroup,
            lane_eps: float,
            *,
            workspace: torch.Tensor = lane_workspace,
            r_block: int = parent_r_block,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return _reduce_and_norm_v9_disjoint_frontier(
                lane_contribution,
                lane_residual,
                lane_weight,
                norm_group,
                lane_eps,
                scratch_override=workspace,
                r_block_override=r_block,
            )

        with torch.cuda.stream(stream):
            result = _owner_residual_arc_prequant_v4_block_class_direct(
                contribution[begin:end],
                residual_owner[begin:end],
                weight,
                input_scale_inv,
                selected_all,
                route0,
                route1,
                route2,
                inverse_order,
                route_counts,
                lane_group,
                eps,
                grouped_output=True,
                norm_impl=lane_norm,
            )
        results.append(result)
    for stream in _v10_lane_streams:
        current.wait_stream(stream)
    q = torch.cat((results[0][0], results[1][0]), dim=0)
    augmented_k = HIDDEN + selected_all.shape[1]
    sf_linear = torch.cat(
        tuple(
            convert_swizzled_to_linear(result[1], end - begin, augmented_k, 16)
            for result, begin, end in zip(
                results, boundaries[:-1], boundaries[1:], strict=True
            )
        ),
        dim=0,
    ).contiguous()
    base_q = torch.cat((results[0][2], results[1][2]), dim=0)
    base_sf_linear = torch.cat(
        tuple(
            convert_swizzled_to_linear(result[3], end - begin, HIDDEN, 16)
            for result, begin, end in zip(
                results, boundaries[:-1], boundaries[1:], strict=True
            )
        ),
        dim=0,
    ).contiguous()
    residual_out = torch.cat((results[0][4], results[1][4]), dim=0)
    return (
        q,
        swizzle_blockscale(sf_linear),
        base_q,
        swizzle_blockscale(base_sf_linear),
        residual_out,
    )


def owner_terminal_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    norm_owner, _ = _reduce_and_norm(contribution, residual_owner, weight, group, eps)
    return materialize_owner_tensor(norm_owner, group)


def owner_terminal_norm_v2(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    norm_owner, _ = _reduce_and_norm_v2(
        contribution, residual_owner, weight, group, eps
    )
    return materialize_owner_tensor(norm_owner, group)


def owner_terminal_norm_v3_lane512(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    norm_owner, _ = _reduce_and_norm_v3_lane512(
        contribution, residual_owner, weight, group, eps
    )
    return materialize_owner_tensor(norm_owner, group)


def materialize_v4_owner_tensor(
    owner: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Replicate scattered 6/5/5 block-class owners in canonical column order."""
    rank = dist.get_rank(group)
    rows = owner.shape[0]
    if owner.shape != (rows, V4_WIDTHS[rank]):
        raise ValueError(
            f"rank {rank} v4 owner shape {owner.shape} != {(rows, V4_WIDTHS[rank])}"
        )
    padded = torch.zeros((rows, max(V4_WIDTHS)), dtype=owner.dtype, device=owner.device)
    padded[:, : V4_WIDTHS[rank]].copy_(owner)
    gathered = torch.empty(
        (WORLD, rows, max(V4_WIDTHS)), dtype=owner.dtype, device=owner.device
    )
    dist.all_gather_single(
        gathered.view(WORLD * rows, max(V4_WIDTHS)), padded, group=group
    )
    output = torch.empty((rows, HIDDEN), dtype=owner.dtype, device=owner.device)
    for source in range(WORLD):
        output.index_copy_(
            1,
            v4_owner_column_indices(source, owner.device),
            gathered[source, :, : V4_WIDTHS[source]],
        )
    return output


def materialize_v8_owner_tensor(
    owner: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Replicate B8192 subtree owners in canonical hidden-column order."""
    rank = dist.get_rank(group)
    rows = owner.shape[0]
    if owner.shape != (rows, V8_WIDTHS[rank]):
        raise ValueError(
            f"rank {rank} v8 owner shape {owner.shape} != {(rows, V8_WIDTHS[rank])}"
        )
    padded = torch.zeros((rows, max(V8_WIDTHS)), dtype=owner.dtype, device=owner.device)
    padded[:, : V8_WIDTHS[rank]].copy_(owner)
    gathered = torch.empty(
        (WORLD, rows, max(V8_WIDTHS)), dtype=owner.dtype, device=owner.device
    )
    dist.all_gather_single(
        gathered.view(WORLD * rows, max(V8_WIDTHS)), padded, group=group
    )
    output = torch.empty((rows, HIDDEN), dtype=owner.dtype, device=owner.device)
    for source in range(WORLD):
        output.index_copy_(
            1,
            v8_owner_column_indices(source, owner.device),
            gathered[source, :, : V8_WIDTHS[source]],
        )
    return output


def owner_terminal_norm_v4_block_class(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    """Finish once through the accepted full seam instead of owner RMS.

    Intermediate owner-normalized values are consumed only after NVFP4
    quantization, whose bytes are exact in the linked control.  The terminal
    value goes directly to the LM head, so even a quantization-invisible RMS
    ULP is semantic here.  Paying one full exact reduction/materialization at
    this single boundary preserves the native terminal contract without
    taxing all decoder seams.
    """
    residual = materialize_v4_owner_tensor(residual_owner, group)
    reduced = exact_weighted_owner_992_reduce(contribution, group)
    normalized, _ = native_fused_add_rms_norm_config(
        reduced, residual, weight, eps, _v4_rblock(contribution.shape[0])
    )
    return normalized


def owner_terminal_norm_v7_compact_stats(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    return owner_terminal_norm_v4_block_class(
        contribution, residual_owner, weight, group, eps
    )


def owner_terminal_norm_v9_disjoint_frontier(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    return owner_terminal_norm_v4_block_class(
        contribution, residual_owner, weight, group, eps
    )


def owner_terminal_norm_v10_two_lane(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    return owner_terminal_norm_v4_block_class(
        contribution, residual_owner, weight, group, eps
    )


def owner_terminal_norm_v5_contiguous_pipeline(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    group: ProcessGroup,
    eps: float,
) -> torch.Tensor:
    residual = materialize_owner_tensor(residual_owner, group)
    reduced = exact_weighted_owner_992_reduce(contribution, group)
    normalized, _ = native_fused_add_rms_norm_config(
        reduced, residual, weight, eps, _v4_rblock(contribution.shape[0])
    )
    return normalized


def materialize_owner_tensor(
    owner: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Replicate one valid 8/8/4 owner slice without reading carrier poison."""
    rank = dist.get_rank(group)
    rows = owner.shape[0]
    if owner.shape != (rows, WIDTHS[rank]):
        raise ValueError(
            f"rank {rank} owner shape {owner.shape} != {(rows, WIDTHS[rank])}"
        )
    elements = [rows * width for width in WIDTHS]
    send = owner.reshape(-1).repeat(WORLD)
    received = torch.empty(sum(elements), dtype=owner.dtype, device=owner.device)
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


def materialize_v1_owner_tensor(
    owner: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    return materialize_owner_tensor(owner, group)


def materialize_v1b5_owner_tensor(
    owner: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    return materialize_owner_tensor(owner, group)


def materialize_v9_owner_tensor(
    owner: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    return materialize_v4_owner_tensor(owner, group)
