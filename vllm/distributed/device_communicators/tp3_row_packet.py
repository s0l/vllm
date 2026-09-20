# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct aligned packet publication with unchanged FP4/ARC arithmetic.

Only the checkpoint-bound width determines a binary. Runtime row counts and
offsets must not create a new JIT variant on each first-use prefill shape.
"""

import torch
import torch.distributed as dist

from vllm.model_executor.kernels.linear.nvfp4.arc import ag2_nvfp4_arc_quantize
from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=("OFFSET", "ROWS", "WIRE"),
    do_not_specialize_on_alignment=("OFFSET", "ROWS", "WIRE"),
)
def _pack(
    Q,
    S,
    OUT,
    OFFSET,
    ROWS,
    K: tl.constexpr,
    WIRE,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    qbytes = ROWS * K // 2
    cols: tl.constexpr = K // 16
    j = i - qbytes
    row, col = j // cols, j % cols
    sf = ((row // 128 * (cols // 4) + col // 4) * 32 + row % 32) * 16
    sf += (row % 128 // 32) * 4 + col % 4
    code = tl.load(Q + i, i < qbytes, other=0)
    scale = tl.load(S + sf, (j >= 0) & (j < ROWS * cols), other=0)
    value = tl.where(i < qbytes, code, scale)
    tl.store(OUT + OFFSET + i, value, i < WIRE)


@triton.jit(
    do_not_specialize=("M", "N0", "N1", "N2", "W0", "W1"),
    do_not_specialize_on_alignment=("M", "N0", "N1", "N2", "W0", "W1"),
)
def _unpack(
    INCOMING,
    Q,
    S,
    M,
    K: tl.constexpr,
    N0,
    N1,
    N2,
    W0,
    W1,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    qcols: tl.constexpr = K // 2
    cols: tl.constexpr = K // 16
    row, col = i // qcols, i % qcols
    begin = tl.where(row < N0, 0, tl.where(row < N0 + N1, N0, N0 + N1))
    wire = tl.where(row < N0, 0, tl.where(row < N0 + N1, W0, W0 + W1))
    value = tl.load(
        INCOMING + wire + (row - begin) * qcols + col, i < M * qcols, other=0
    )
    tl.store(Q + i, value, i < M * qcols)
    # Inverse of the independently enumerated 128x4 scale permutation.
    inner = i % 4
    t = i // 4
    quarter, lane = t % 4, (t // 4) % 32
    block = (t // 128) % (cols // 4)
    tile = t // (128 * (cols // 4))
    row, col = tile * 128 + quarter * 32 + lane, block * 4 + inner
    begin = tl.where(row < N0, 0, tl.where(row < N0 + N1, N0, N0 + N1))
    count = tl.where(row < N0, N0, tl.where(row < N0 + N1, N1, N2))
    wire = tl.where(row < N0, 0, tl.where(row < N0 + N1, W0, W0 + W1))
    offset = wire + count * qcols + (row - begin) * cols + col
    scale = tl.load(INCOMING + offset, row < M, other=0)
    tl.store(S + i, scale, i < tl.cdiv(M, 128) * 128 * cols)


def unpack(incoming, plan, rank):
    width, rows = plan.augmented_k[rank], plan.total_rows
    recv = plan.wire_splits(rank, alignment=16)[1]
    if incoming.dtype != torch.uint8 or incoming.numel() != sum(recv):
        raise ValueError("aligned packet size/dtype mismatch")
    q = torch.empty((rows, width // 2), dtype=torch.uint8, device=incoming.device)
    sf = torch.empty(
        (triton.cdiv(rows, 128) * 128, width // 16),
        dtype=torch.float8_e4m3fn,
        device=incoming.device,
    )
    _unpack[(triton.cdiv(max(q.numel(), sf.numel()), 256),)](
        incoming, q, sf.view(torch.uint8), rows, width, *plan.rows, *recv[:2], 256
    )
    return q, sf


def publish_quantized(seam, normalized, divisor, selected):
    """Publish already-owned normalized rows into fresh functional outputs."""
    if (
        not callable(normalized)
        or len(selected) != 3
        or any(
            s.dtype != torch.int32
            or s.ndim != 1
            or not s.is_contiguous()
            or s.device != divisor.device
            or s.numel() + seam.plan.hidden != seam.plan.augmented_k[d]
            for d, s in enumerate(selected)
        )
    ):
        raise ValueError("destination ARC identity mismatch before collective")
    send, recv = seam.plan.wire_splits(seam.rank, alignment=16)
    outgoing = torch.empty(sum(send), dtype=torch.uint8, device=divisor.device)
    offset = 0
    for destination in range(3):
        value = normalized(destination)
        if (
            value.shape != (seam.count, seam.plan.hidden)
            or value.dtype != torch.bfloat16
            or value.device != divisor.device
            or not value.is_contiguous()
        ):
            raise ValueError("packet requires contiguous owned normalized rows")
        q, sf = ag2_nvfp4_arc_quantize(value, divisor, selected[destination])
        _pack[(triton.cdiv(send[destination], 256),)](
            q,
            sf.view(torch.uint8),
            outgoing,
            offset,
            seam.count,
            seam.plan.augmented_k[destination],
            send[destination],
            256,
        )
        offset += send[destination]
    incoming = torch.empty(sum(recv), dtype=torch.uint8, device=outgoing.device)
    dist.all_to_all_single(incoming, outgoing, list(recv), list(send), group=seam.group)
    return unpack(incoming, seam.plan, seam.rank)
