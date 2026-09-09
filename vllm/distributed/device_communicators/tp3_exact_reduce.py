# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact TP3 reductions with fixed arithmetic and topology-aware ownership."""

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed import ProcessGroup


def fixed_tp3_sum(gathered: torch.Tensor) -> torch.Tensor:
    """Sum rank0, rank1, rank2 in FP32 and round to the input dtype once."""
    if gathered.shape[0] != 3:
        raise ValueError(f"expected three TP ranks, got {tuple(gathered.shape)}")
    result = gathered[0].float() + gathered[1].float()
    result.add_(gathered[2].float())
    return result.to(gathered.dtype)


@triton.jit
def _fixed_tp3_sum_kernel(
    gathered_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    rank0 = tl.load(gathered_ptr + offsets, mask=mask).to(tl.float32)
    rank1 = tl.load(gathered_ptr + n_elements + offsets, mask=mask).to(tl.float32)
    rank2 = tl.load(gathered_ptr + 2 * n_elements + offsets, mask=mask).to(tl.float32)
    rank01 = rank0 + rank1
    result = rank01 + rank2
    tl.store(output_ptr + offsets, result, mask=mask)


def fixed_tp3_sum_fused(gathered: torch.Tensor) -> torch.Tensor:
    """One-kernel form of the same fixed FP32 rank order."""
    if gathered.shape[0] != 3:
        raise ValueError(f"expected three TP ranks, got {tuple(gathered.shape)}")
    if not gathered.is_contiguous():
        gathered = gathered.contiguous()
    output = torch.empty_like(gathered[0])
    n_elements = output.numel()
    _fixed_tp3_sum_kernel[(triton.cdiv(n_elements, 256),)](
        gathered,
        output,
        n_elements,
        BLOCK_SIZE=256,
    )
    return output


def exact_all_gather_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Latency-oriented exact TP3 reducer for small packets."""
    rows, cols = tensor.shape
    gathered = torch.empty((3 * rows, cols), dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_single(gathered, tensor, group=group)
    return fixed_tp3_sum(gathered.view(3, rows, cols))


def exact_all_gather_fused_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """All-gather transport with one fixed-sum pointwise kernel."""
    rows, cols = tensor.shape
    gathered = torch.empty((3 * rows, cols), dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_single(gathered, tensor, group=group)
    return fixed_tp3_sum_fused(gathered.view(3, rows, cols))


def exact_fast_pair_owner_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Own equal output halves on ranks 0/1; rank 2 owns no output columns.

    Phase one sends each rank's left/right BF16 half to its owner. Owners sum
    the three contributions in fixed rank order and round once. Phase two
    all-gathers two completed halves plus a zero half from rank 2. The zero
    half keeps the second collective regular and graph-capturable.
    """
    if tensor.ndim != 2 or tensor.shape[1] % 2:
        raise ValueError(f"expected even-width matrix, got {tuple(tensor.shape)}")
    if dist.get_world_size(group) != 3:
        raise ValueError("fast-pair exact reduction requires TP=3")

    rank = dist.get_rank(group)
    rows, cols = tensor.shape
    half_cols = cols // 2
    shard_elements = rows * half_cols
    send = torch.cat(
        (
            tensor[:, :half_cols].reshape(-1),
            tensor[:, half_cols:].reshape(-1),
        )
    )
    input_splits = [shard_elements, shard_elements, 0]
    if rank < 2:
        receive = torch.empty(
            3 * shard_elements, dtype=tensor.dtype, device=tensor.device
        )
        output_splits = [shard_elements, shard_elements, shard_elements]
    else:
        receive = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        output_splits = [0, 0, 0]

    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    if rank < 2:
        owned = fixed_tp3_sum(receive.view(3, rows, half_cols))
    else:
        owned = torch.zeros((rows, half_cols), dtype=tensor.dtype, device=tensor.device)

    owner_halves = torch.empty(
        (3 * rows, half_cols), dtype=tensor.dtype, device=tensor.device
    )
    dist.all_gather_single(owner_halves, owned, group=group)
    owner_halves = owner_halves.view(3, rows, half_cols)
    return torch.cat((owner_halves[0], owner_halves[1]), dim=1)


def exact_direct_pair_owner_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Disseminate fixed-order owner halves without rank 2's zero payload."""
    if tensor.ndim != 2 or tensor.shape[1] % 2:
        raise ValueError(f"expected even-width matrix, got {tuple(tensor.shape)}")
    if dist.get_world_size(group) != 3:
        raise ValueError("direct-pair exact reduction requires TP=3")

    rank = dist.get_rank(group)
    rows, cols = tensor.shape
    half_cols = cols // 2
    shard_elements = rows * half_cols
    send = torch.cat(
        (
            tensor[:, :half_cols].reshape(-1),
            tensor[:, half_cols:].reshape(-1),
        )
    )
    input_splits = [shard_elements, shard_elements, 0]
    if rank < 2:
        receive = torch.empty(
            3 * shard_elements, dtype=tensor.dtype, device=tensor.device
        )
        output_splits = [shard_elements, shard_elements, shard_elements]
    else:
        receive = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        output_splits = [0, 0, 0]
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    if rank < 2:
        owned = fixed_tp3_sum(receive.view(3, rows, half_cols))
        disseminate = owned.reshape(-1).repeat(3)
        disseminate_splits = [shard_elements, shard_elements, shard_elements]
    else:
        disseminate = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        disseminate_splits = [0, 0, 0]
    completed = torch.empty(
        2 * shard_elements, dtype=tensor.dtype, device=tensor.device
    )
    dist.all_to_all_single(
        completed,
        disseminate,
        output_split_sizes=[shard_elements, shard_elements, 0],
        input_split_sizes=disseminate_splits,
        group=group,
    )
    return completed.view(2, rows, half_cols).movedim(0, 1).reshape(rows, cols)


def exact_direct_pair_owner_fused_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Direct owner dissemination with one fixed-sum pointwise kernel."""
    if tensor.ndim != 2 or tensor.shape[1] % 2:
        raise ValueError(f"expected even-width matrix, got {tuple(tensor.shape)}")
    if dist.get_world_size(group) != 3:
        raise ValueError("direct-pair fused exact reduction requires TP=3")

    rank = dist.get_rank(group)
    rows, cols = tensor.shape
    half_cols = cols // 2
    shard_elements = rows * half_cols
    send = torch.cat(
        (tensor[:, :half_cols].reshape(-1), tensor[:, half_cols:].reshape(-1))
    )
    if rank < 2:
        receive = torch.empty(
            3 * shard_elements, dtype=tensor.dtype, device=tensor.device
        )
        output_splits = [shard_elements, shard_elements, shard_elements]
    else:
        receive = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        output_splits = [0, 0, 0]
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=output_splits,
        input_split_sizes=[shard_elements, shard_elements, 0],
        group=group,
    )
    if rank < 2:
        owned = fixed_tp3_sum_fused(receive.view(3, rows, half_cols))
        disseminate = owned.reshape(-1).repeat(3)
        disseminate_splits = [shard_elements, shard_elements, shard_elements]
    else:
        disseminate = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        disseminate_splits = [0, 0, 0]
    completed = torch.empty(
        2 * shard_elements, dtype=tensor.dtype, device=tensor.device
    )
    dist.all_to_all_single(
        completed,
        disseminate,
        output_split_sizes=[shard_elements, shard_elements, 0],
        input_split_sizes=disseminate_splits,
        group=group,
    )
    return completed.view(2, rows, half_cols).movedim(0, 1).reshape(rows, cols)


def exact_pair_owner_broadcast_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Collect on two owners, then broadcast each completed half."""
    if tensor.ndim != 2 or tensor.shape[1] % 2:
        raise ValueError(f"expected even-width matrix, got {tuple(tensor.shape)}")
    if dist.get_world_size(group) != 3:
        raise ValueError("pair-broadcast exact reduction requires TP=3")
    rank = dist.get_rank(group)
    rows, cols = tensor.shape
    half_cols = cols // 2
    shard_elements = rows * half_cols
    send = torch.cat(
        (tensor[:, :half_cols].reshape(-1), tensor[:, half_cols:].reshape(-1))
    )
    receive = torch.empty(
        3 * shard_elements if rank < 2 else 0,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=(
            [shard_elements, shard_elements, shard_elements] if rank < 2 else [0, 0, 0]
        ),
        input_split_sizes=[shard_elements, shard_elements, 0],
        group=group,
    )
    left = (
        fixed_tp3_sum_fused(receive.view(3, rows, half_cols))
        if rank == 0
        else torch.empty((rows, half_cols), dtype=tensor.dtype, device=tensor.device)
    )
    right = (
        fixed_tp3_sum_fused(receive.view(3, rows, half_cols))
        if rank == 1
        else torch.empty((rows, half_cols), dtype=tensor.dtype, device=tensor.device)
    )
    dist.broadcast(left, src=0, group=group)
    dist.broadcast(right, src=1, group=group)
    return torch.cat((left, right), dim=1)


def exact_root_owner_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Gather to rank 0, fixed-sum once, then directly disseminate."""
    if tensor.ndim != 2:
        raise ValueError(f"expected matrix, got {tuple(tensor.shape)}")
    if dist.get_world_size(group) != 3:
        raise ValueError("root-owner exact reduction requires TP=3")
    rank = dist.get_rank(group)
    elements = tensor.numel()
    receive = torch.empty(
        3 * elements if rank == 0 else 0,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_to_all_single(
        receive,
        tensor.reshape(-1),
        output_split_sizes=[elements, elements, elements] if rank == 0 else [0, 0, 0],
        input_split_sizes=[elements, 0, 0],
        group=group,
    )
    if rank == 0:
        owned = fixed_tp3_sum_fused(receive.view(3, *tensor.shape))
        disseminate = owned.reshape(-1).repeat(3)
        disseminate_splits = [elements, elements, elements]
    else:
        disseminate = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        disseminate_splits = [0, 0, 0]
    output = torch.empty_like(tensor).reshape(-1)
    dist.all_to_all_single(
        output,
        disseminate,
        output_split_sizes=[elements, 0, 0],
        input_split_sizes=disseminate_splits,
        group=group,
    )
    return output.view_as(tensor)


def _exact_weighted_three_owner_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
    widths: tuple[int, int, int],
) -> torch.Tensor:
    """Own unequal column shards on all three ranks, then disseminate."""
    if tensor.ndim != 2 or sum(widths) != tensor.shape[1] or min(widths) < 1:
        raise ValueError(
            f"invalid owner widths {widths} for matrix {tuple(tensor.shape)}"
        )
    if dist.get_world_size(group) != 3:
        raise ValueError("weighted-owner exact reduction requires TP=3")
    rank = dist.get_rank(group)
    rows, _ = tensor.shape
    owner_chunks = torch.split(tensor, widths, dim=1)
    send = torch.cat([chunk.contiguous().view(-1) for chunk in owner_chunks])
    shard_elements = [rows * width for width in widths]
    owned_elements = shard_elements[rank]
    receive = torch.empty(3 * owned_elements, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[owned_elements] * 3,
        input_split_sizes=shard_elements,
        group=group,
    )
    owned = fixed_tp3_sum_fused(receive.view(3, rows, widths[rank]))
    disseminate = owned.reshape(-1).repeat(3)
    completed = torch.empty(
        sum(shard_elements), dtype=tensor.dtype, device=tensor.device
    )
    dist.all_to_all_single(
        completed,
        disseminate,
        output_split_sizes=shard_elements,
        input_split_sizes=[owned_elements] * 3,
        group=group,
    )
    chunks = [
        chunk.view(rows, width)
        for chunk, width in zip(completed.split(shard_elements), widths)
    ]
    return torch.cat(chunks, dim=1)


def exact_weighted_owner_884_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Use 40/40/20 percent ownership on ranks 0/1/2."""
    return _exact_weighted_three_owner_reduce(tensor, group, (2048, 2048, 1024))


def exact_weighted_owner_996_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Use 37.5/37.5/25 percent ownership on ranks 0/1/2."""
    return _exact_weighted_three_owner_reduce(tensor, group, (1920, 1920, 1280))


def exact_weighted_owner_992_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Use robust 45/45/10 percent ownership on ranks 0/1/2."""
    return _exact_weighted_three_owner_reduce(tensor, group, (2304, 2304, 512))


def candidate_fp32_all_reduce(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Fast falsifier: NCCL controls its FP32 reduction order."""
    output = tensor.float()
    dist.all_reduce(output, group=group)
    return output.to(tensor.dtype)


def exact_head_owner_reduce_scatter(
    tensor: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Reduce TP3 attention heads at their final owner in fixed rank order.

    ``tensor`` is [rows, global_heads, head_dim].  Every rank sends each
    owner's head slice directly to that owner.  The owner receives the three
    source-rank contributions in rank order, accumulates in FP32, and rounds
    once to BF16.  No all-gather is needed because reduce-scatter's result is
    already owner-local.
    """
    if tensor.ndim != 3 or tensor.shape[1] % 3:
        raise ValueError(
            "TP3 exact head-owner reduce-scatter requires "
            f"[rows,heads,dim] with heads divisible by 3, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.bfloat16:
        raise ValueError(
            f"TP3 exact head-owner reduce-scatter requires BF16, got {tensor.dtype}"
        )
    if dist.get_world_size(group) != 3:
        raise ValueError("exact head-owner reduce-scatter requires TP=3")

    rows, global_heads, head_dim = tensor.shape
    local_heads = global_heads // 3
    shard_elements = rows * local_heads * head_dim
    send = (
        tensor.contiguous()
        .view(rows, 3, local_heads, head_dim)
        .movedim(1, 0)
        .contiguous()
        .view(-1)
    )
    receive = torch.empty(
        3 * shard_elements,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    equal_splits = [shard_elements, shard_elements, shard_elements]
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=equal_splits,
        input_split_sizes=equal_splits,
        group=group,
    )
    return fixed_tp3_sum(receive.view(3, rows, local_heads, head_dim))
