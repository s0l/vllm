# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton primitives for compact whole-expert E8 execution."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr


@tr.jit(
    do_not_specialize=["lanes"],
    do_not_specialize_on_alignment=["lanes"],
)
def _count_routes(
    ids,
    counts,
    expert_map,
    lanes,
    topk: tl.constexpr,
    begin: tl.constexpr,
    end: tl.constexpr,
    B: tl.constexpr,
    USE_EXPERT_MAP: tl.constexpr,
):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    expert = tl.load(ids + lane, lane < lanes, -1).to(tl.int32)
    keep = (lane < lanes) & (expert >= begin) & (expert < end)
    local = expert - begin
    if USE_EXPERT_MAP:
        local = tl.load(expert_map + local, mask=keep, other=0)
    tl.atomic_add(counts + local, 1, mask=keep)


@tr.jit(
    do_not_specialize=["lanes"],
    do_not_specialize_on_alignment=["lanes"],
)
def _compact_routes(
    ids,
    weights,
    cursor,
    tokens_out,
    weights_out,
    experts_out,
    expert_map,
    lanes,
    topk: tl.constexpr,
    begin: tl.constexpr,
    end: tl.constexpr,
    B: tl.constexpr,
    USE_EXPERT_MAP: tl.constexpr,
):
    lane = tl.program_id(0) * B + tl.arange(0, B)
    expert = tl.load(ids + lane, lane < lanes, -1).to(tl.int32)
    keep = (lane < lanes) & (expert >= begin) & (expert < end)
    local = expert - begin
    if USE_EXPERT_MAP:
        local = tl.load(expert_map + local, mask=keep, other=0)
    position = tl.atomic_add(cursor + local, 1, mask=keep)
    tl.store(tokens_out + position, lane // topk, mask=keep)
    tl.store(
        weights_out + position, tl.load(weights + lane, mask=keep, other=0.0), mask=keep
    )
    tl.store(experts_out + position, local, mask=keep)


@tr.jit
def _build_tasks(
    counts,
    offsets,
    task_counter,
    task_expert,
    task_begin,
    task_rows,
    owner: tl.constexpr,
    max_tiles: tl.constexpr,
    rows: tl.constexpr,
):
    expert = tl.program_id(0)
    tile = tl.program_id(1)
    count = tl.load(counts + expert)
    begin = tile * rows
    valid = (expert < owner) & (begin < count)
    task = tl.atomic_add(task_counter, 1, mask=valid)
    tl.store(task_expert + task, expert, mask=valid)
    tl.store(task_begin + task, tl.load(offsets + expert) + begin, mask=valid)
    tl.store(task_rows + task, tl.minimum(rows, count - begin), mask=valid)


@tr.jit
def _grouped_e8_mm(
    X,
    Q_RESIDENT,
    Q_COLD,
    CODEBOOK,
    Y,
    TASK_EXPERT,
    TASK_BEGIN,
    TASK_ROWS,
    layer: tl.constexpr,
    resident: tl.constexpr,
    cold: tl.constexpr,
    max_tasks: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    nb = tl.program_id(1)
    expert = tl.load(TASK_EXPERT + task, mask=task < max_tasks, other=-1)
    begin = tl.load(TASK_BEGIN + task, mask=task < max_tasks, other=0)
    rows = tl.load(TASK_ROWS + task, mask=task < max_tasks, other=0)
    om = begin + tl.arange(0, 8)
    row_mask = tl.arange(0, 8) < rows
    on = nb * BN + tl.arange(0, BN)
    acc = tl.zeros((8, BN), tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        oq = k0 // 8 + tl.arange(0, BK // 8)
        resident_expert = expert < resident
        rq = ((layer * resident + expert) * N + on[:, None]) * (K // 8) + oq[None, :]
        cq = ((expert - resident) * N + on[:, None]) * (K // 8) + oq[None, :]
        codes = tl.load(
            Q_RESIDENT + rq,
            mask=(expert >= 0) & resident_expert & (on[:, None] < N),
            other=0,
        ).to(tl.int32)
        cold_codes = tl.load(
            Q_COLD + cq,
            mask=(expert >= resident) & (expert < resident + cold) & (on[:, None] < N),
            other=0,
        ).to(tl.int32)
        codes = tl.where(resident_expert, codes, cold_codes)
        component = tl.arange(0, 8)
        w = tl.load(CODEBOOK + codes[:, :, None] * 8 + component[None, None, :])
        w = tl.reshape(w, (BN, BK))
        x = tl.load(
            X + om[:, None] * K + ok[None, :],
            mask=(expert >= 0) & row_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
    tl.store(
        Y + om[:, None] * N + on[None, :],
        acc,
        mask=(expert >= 0) & row_mask[:, None] & (on[None, :] < N),
    )


@tr.jit
def _persistent_grouped_e8_mm(
    X,
    Q_RESIDENT,
    Q_COLD,
    COLD_MAP,
    RESIDENT_MAP,
    CODEBOOK,
    Y,
    TASK_COUNTER,
    TASK_EXPERT,
    TASK_BEGIN,
    TASK_ROWS,
    layer: tl.constexpr,
    resident: tl.constexpr,
    cold: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WORKERS: tl.constexpr,
    USE_COLD_MAP: tl.constexpr,
    USE_RESIDENT_MAP: tl.constexpr,
):
    """Consume the device task count without launching its unused tail."""
    work = tl.program_id(0)
    output_tiles: tl.constexpr = tr.cdiv(N, BN)
    total = tl.load(TASK_COUNTER) * output_tiles
    while work < total:
        task = work // output_tiles
        nb = work - task * output_tiles
        expert = tl.load(TASK_EXPERT + task)
        begin = tl.load(TASK_BEGIN + task)
        rows = tl.load(TASK_ROWS + task)
        om = begin + tl.arange(0, BM)
        row_mask = tl.arange(0, BM) < rows
        on = nb * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            ok = k0 + tl.arange(0, BK)
            oq = k0 // 8 + tl.arange(0, BK // 8)
            resident_slot = (
                tl.load(RESIDENT_MAP + expert) if USE_RESIDENT_MAP else expert
            )
            resident_expert = (
                resident_slot >= 0 if USE_RESIDENT_MAP else expert < resident
            )
            rq = ((layer * resident + resident_slot) * N + on[:, None]) * (K // 8) + oq[
                None, :
            ]
            cold_expert = (
                tl.load(COLD_MAP + expert) if USE_COLD_MAP else expert - resident
            )
            cq = (cold_expert * N + on[:, None]) * (K // 8) + oq[None, :]
            codes = tl.load(
                Q_RESIDENT + rq,
                mask=resident_expert & (on[:, None] < N),
                other=0,
            ).to(tl.int32)
            cold_codes = tl.load(
                Q_COLD + cq,
                mask=(
                    ((~resident_expert) & (cold_expert >= 0))
                    if USE_RESIDENT_MAP
                    else (expert >= resident)
                    & (expert < resident + cold)
                    & (cold_expert >= 0)
                )
                & (on[:, None] < N),
                other=0,
            ).to(tl.int32)
            codes = tl.where(resident_expert, codes, cold_codes)
            component = tl.arange(0, 8)
            weight = tl.load(
                CODEBOOK + codes[:, :, None] * 8 + component[None, None, :]
            )
            weight = tl.reshape(weight, (BN, BK))
            value = tl.load(
                X + om[:, None] * K + ok[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(value, tl.trans(weight))
        tl.store(
            Y + om[:, None] * N + on[None, :],
            acc,
            mask=row_mask[:, None] & (on[None, :] < N),
        )
        work += WORKERS


@tr.jit
def _decode_e8_codes(codes, PACKED_ABS):
    """Exactly reconstruct E8P12 vectors from the 1 KiB packed abs grid."""
    signs = codes & 255
    parity = signs & 1
    parity = parity ^ ((signs >> 1) & 1)
    parity = parity ^ ((signs >> 2) & 1)
    parity = parity ^ ((signs >> 3) & 1)
    parity = parity ^ ((signs >> 4) & 1)
    parity = parity ^ ((signs >> 5) & 1)
    parity = parity ^ ((signs >> 6) & 1)
    parity = parity ^ ((signs >> 7) & 1)
    signs = signs ^ parity
    packed = tl.load(PACKED_ABS + (codes >> 8))
    component = tl.arange(0, 8)
    shuffled = component // 2 + (component % 2) * 4
    nibble = (packed[:, :, None] >> (4 * shuffled[None, None, :])) & 15
    magnitude = (nibble.to(tl.float32) - 8.0) * 0.5
    negative = ((signs[:, :, None] >> shuffled[None, None, :]) & 1) != 0
    signed = tl.where(negative, -magnitude, magnitude)
    offset = tl.where(parity[:, :, None] != 0, -0.25, 0.25)
    return (signed + offset).to(tl.float16)


@tr.jit
def _fwht_stage(x, N: tl.constexpr, groups: tl.constexpr, stride: tl.constexpr):
    shaped = tl.reshape(x, (groups, 2, stride))
    shaped = tl.trans(shaped, 0, 2, 1)
    left, right = tl.split(shaped)
    shaped = tl.join(left + right, left - right)
    shaped = tl.trans(shaped, 0, 2, 1)
    return tl.reshape(shaped, (N,))


@tr.jit
def _masked_hadamard_power2(
    SOURCE,
    SCALE,
    ROW_INDEX,
    EXPERT_INDEX,
    TMP,
    LANE_COUNT,
    SOURCE_STRIDE: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr,
    MAX_LANES: tl.constexpr,
    GATHER: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    lanes = tl.load(LANE_COUNT)
    valid = row < lanes
    source_row = tl.load(ROW_INDEX + row, mask=valid, other=0) if GATHER else row
    offset = group * P + tl.arange(0, P)
    value = tl.load(
        SOURCE + source_row * SOURCE_STRIDE + offset,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    if APPLY_SCALE:
        expert = tl.load(EXPERT_INDEX + row, mask=valid, other=0)
        value *= tl.load(
            SCALE + expert * N + offset,
            mask=valid,
            other=0.0,
        )
    if P >= 2:
        value = _fwht_stage(value, P, P // 2, 1)
    if P >= 4:
        value = _fwht_stage(value, P, P // 4, 2)
    if P >= 8:
        value = _fwht_stage(value, P, P // 8, 4)
    if P >= 16:
        value = _fwht_stage(value, P, P // 16, 8)
    if P >= 32:
        value = _fwht_stage(value, P, P // 32, 16)
    if P >= 64:
        value = _fwht_stage(value, P, P // 64, 32)
    if P >= 128:
        value = _fwht_stage(value, P, P // 128, 64)
    tl.store(TMP + row * N + offset, value, mask=valid)


@tr.jit
def _masked_hadamard20(
    TMP,
    HADAMARD20,
    OUTPUT,
    LANE_COUNT,
    N: tl.constexpr,
    P: tl.constexpr,
    MAX_LANES: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    BG: tl.constexpr,
    BP: tl.constexpr,
):
    row = tl.program_id(0)
    group_block = tl.program_id(1)
    column_block = tl.program_id(2)
    lanes = tl.load(LANE_COUNT)
    valid_row = row < lanes
    output_group = group_block * BG + tl.arange(0, BG)
    input_group = tl.arange(0, 32)
    column = column_block * BP + tl.arange(0, BP)
    if TRANSPOSE:
        h_offset = input_group[None, :] * 20 + output_group[:, None]
    else:
        h_offset = output_group[:, None] * 20 + input_group[None, :]
    h = tl.load(
        HADAMARD20 + h_offset,
        mask=(output_group[:, None] < 20) & (input_group[None, :] < 20),
        other=0.0,
    )
    value = tl.load(
        TMP + row * N + input_group[:, None] * P + column[None, :],
        mask=valid_row & (input_group[:, None] < 20) & (column[None, :] < P),
        other=0.0,
    )
    result = tl.dot(h, value, out_dtype=tl.float32) * (1.0 / tl.sqrt(float(N)))
    tl.store(
        OUTPUT + row * N + output_group[:, None] * P + column[None, :],
        result,
        mask=valid_row & (output_group[:, None] < 20) & (column[None, :] < P),
    )


@tr.jit
def _masked_e8_activation(
    GATE,
    UP,
    SV_GATE,
    SV_UP,
    SU_DOWN,
    EXPERT_INDEX,
    OUTPUT,
    LANE_COUNT,
    WIDTH: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    UP_STRIDE: tl.constexpr,
    MAX_LANES: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * B + tl.arange(0, B)
    lanes = tl.load(LANE_COUNT)
    valid = (row < lanes) & (column < WIDTH)
    expert = tl.load(EXPERT_INDEX + row, mask=row < lanes, other=0)
    gate = tl.load(GATE + row * GATE_STRIDE + column, mask=valid, other=0.0).to(
        tl.float32
    )
    up = tl.load(UP + row * UP_STRIDE + column, mask=valid, other=0.0).to(tl.float32)
    gate *= tl.load(SV_GATE + expert * WIDTH + column, mask=valid, other=0.0)
    up *= tl.load(SV_UP + expert * WIDTH + column, mask=valid, other=0.0)
    value = gate * tl.sigmoid(gate) * up
    value *= tl.load(SU_DOWN + expert * WIDTH + column, mask=valid, other=0.0)
    tl.store(OUTPUT + row * WIDTH + column, value, mask=valid)


@tr.jit
def _masked_e8_scatter(
    VALUE,
    SV_DOWN,
    TOKEN_INDEX,
    EXPERT_INDEX,
    ROUTE_WEIGHT,
    PARTIAL,
    LANE_COUNT,
    H: tl.constexpr,
    MAX_LANES: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * B + tl.arange(0, B)
    lanes = tl.load(LANE_COUNT)
    valid_row = row < lanes
    valid = valid_row & (column < H)
    token = tl.load(TOKEN_INDEX + row, mask=valid_row, other=0)
    expert = tl.load(EXPERT_INDEX + row, mask=valid_row, other=0)
    weight = tl.load(ROUTE_WEIGHT + row, mask=valid_row, other=0.0)
    value = tl.load(VALUE + row * H + column, mask=valid, other=0.0)
    scale = tl.load(SV_DOWN + expert * H + column, mask=valid, other=0.0)
    tl.atomic_add(PARTIAL + token * H + column, value * scale * weight, mask=valid)


@tr.jit
def _masked_e8_scatter_deterministic(
    VALUE,
    SV_DOWN,
    TOKEN_INDEX,
    EXPERT_INDEX,
    ROUTE_WEIGHT,
    PARTIAL,
    LANE_COUNT,
    H: tl.constexpr,
    TOKENS: tl.constexpr,
    MAX_LANES: tl.constexpr,
    B: tl.constexpr,
):
    """Reduce compact routes in a fixed order, one writer per output value."""
    token = tl.program_id(0)
    column = tl.program_id(1) * B + tl.arange(0, B)
    lanes = tl.load(LANE_COUNT)
    valid_column = column < H
    total = tl.zeros((B,), tl.float32)
    for row in tl.static_range(0, MAX_LANES):
        valid_row = row < lanes
        route_token = tl.load(TOKEN_INDEX + row, mask=valid_row, other=-1)
        selected = valid_row & (route_token == token)
        expert = tl.load(EXPERT_INDEX + row, mask=selected, other=0)
        weight = tl.load(ROUTE_WEIGHT + row, mask=selected, other=0.0)
        mask = selected & valid_column
        value = tl.load(VALUE + row * H + column, mask=mask, other=0.0)
        scale = tl.load(SV_DOWN + expert * H + column, mask=mask, other=0.0)
        total += value.to(tl.float32) * scale.to(tl.float32) * weight.to(tl.float32)
    tl.store(PARTIAL + token * H + column, total, mask=valid_column)


@tr.jit
def _persistent_grouped_e8_compact_mm(
    X,
    Q_RESIDENT,
    Q_COLD,
    PACKED_ABS,
    Y,
    TASK_COUNTER,
    TASK_EXPERT,
    TASK_BEGIN,
    TASK_ROWS,
    layer: tl.constexpr,
    resident: tl.constexpr,
    cold: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WORKERS: tl.constexpr,
):
    """Persistent grouped GEMM with exact algorithmic E8P12 decode."""
    work = tl.program_id(0)
    output_tiles: tl.constexpr = tr.cdiv(N, BN)
    total = tl.load(TASK_COUNTER) * output_tiles
    while work < total:
        task = work // output_tiles
        nb = work - task * output_tiles
        expert = tl.load(TASK_EXPERT + task)
        begin = tl.load(TASK_BEGIN + task)
        rows = tl.load(TASK_ROWS + task)
        om = begin + tl.arange(0, 8)
        row_mask = tl.arange(0, 8) < rows
        on = nb * BN + tl.arange(0, BN)
        acc = tl.zeros((8, BN), tl.float32)
        for k0 in range(0, K, BK):
            ok = k0 + tl.arange(0, BK)
            oq = k0 // 8 + tl.arange(0, BK // 8)
            resident_expert = expert < resident
            rq = ((layer * resident + expert) * N + on[:, None]) * (K // 8) + oq[
                None, :
            ]
            cq = ((expert - resident) * N + on[:, None]) * (K // 8) + oq[None, :]
            codes = tl.load(
                Q_RESIDENT + rq,
                mask=resident_expert & (on[:, None] < N),
                other=0,
            ).to(tl.int32)
            cold_codes = tl.load(
                Q_COLD + cq,
                mask=(expert >= resident)
                & (expert < resident + cold)
                & (on[:, None] < N),
                other=0,
            ).to(tl.int32)
            codes = tl.where(resident_expert, codes, cold_codes)
            weight = tl.reshape(_decode_e8_codes(codes, PACKED_ABS), (BN, BK))
            value = tl.load(
                X + om[:, None] * K + ok[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(value, tl.trans(weight))
        tl.store(
            Y + om[:, None] * N + on[None, :],
            acc,
            mask=row_mask[:, None] & (on[None, :] < N),
        )
        work += WORKERS


@tr.jit
def _expert_grid_e8_mm(
    X,
    Q_RESIDENT,
    Q_COLD,
    CODEBOOK,
    Y,
    ACTIVE_EXPERTS,
    COUNTS,
    OFFSETS,
    layer: tl.constexpr,
    resident: tl.constexpr,
    cold: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    packed_expert, row_tile = tl.program_id(0), tl.program_id(1)
    output_tiles: tl.constexpr = tr.cdiv(N, BN)
    active = packed_expert // output_tiles
    nb = packed_expert - active * output_tiles
    expert = tl.load(ACTIVE_EXPERTS + active)
    count = tl.load(COUNTS + expert)
    begin = tl.load(OFFSETS + expert) + row_tile * 8
    row_mask = row_tile * 8 + tl.arange(0, 8) < count
    om = begin + tl.arange(0, 8)
    on = nb * BN + tl.arange(0, BN)
    acc = tl.zeros((8, BN), tl.float32)
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        oq = k0 // 8 + tl.arange(0, BK // 8)
        resident_expert = expert < resident
        rq = ((layer * resident + expert) * N + on[:, None]) * (K // 8) + oq[None, :]
        cq = ((expert - resident) * N + on[:, None]) * (K // 8) + oq[None, :]
        codes = tl.load(
            Q_RESIDENT + rq,
            mask=resident_expert & (on[:, None] < N),
            other=0,
        ).to(tl.int32)
        cold_codes = tl.load(
            Q_COLD + cq,
            mask=(expert >= resident) & (expert < resident + cold) & (on[:, None] < N),
            other=0,
        ).to(tl.int32)
        codes = tl.where(resident_expert, codes, cold_codes)
        component = tl.arange(0, 8)
        weight = tl.load(CODEBOOK + codes[:, :, None] * 8 + component[None, None, :])
        weight = tl.reshape(weight, (BN, BK))
        value = tl.load(
            X + om[:, None] * K + ok[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(value, tl.trans(weight))
    tl.store(
        Y + om[:, None] * N + on[None, :],
        acc,
        mask=row_mask[:, None] & (on[None, :] < N),
    )


def compact_routes(
    ids,
    weights,
    *,
    begin,
    end,
    counts,
    offsets,
    cursor,
    tokens_out,
    weights_out,
    experts_out,
    synchronize_count=True,
    observer=None,
    expert_map=None,
) -> int | None:
    lanes, topk = ids.numel(), ids.shape[1]
    counts.zero_()
    if observer is not None:
        observer("counts_zero_done")
    _count_routes[(tr.cdiv(lanes, 256),)](
        ids,
        counts,
        counts if expert_map is None else expert_map,
        lanes,
        topk,
        begin,
        end,
        256,
        expert_map is not None,
        num_warps=4,
    )
    if observer is not None:
        observer("count_routes_done")
    offsets[0].zero_()
    torch.cumsum(counts, 0, out=offsets[1:])
    if observer is not None:
        observer("route_scan_done")
    cursor.copy_(offsets[:-1])
    if observer is not None:
        observer("cursor_done")
    tokens_out.zero_()
    weights_out.zero_()
    experts_out.zero_()
    if observer is not None:
        observer("compact_buffers_zero_done")
    _compact_routes[(tr.cdiv(lanes, 256),)](
        ids,
        weights,
        cursor,
        tokens_out,
        weights_out,
        experts_out,
        counts if expert_map is None else expert_map,
        lanes,
        topk,
        begin,
        end,
        256,
        expert_map is not None,
        num_warps=4,
    )
    if observer is not None:
        observer("compact_scatter_done")
    if synchronize_count:
        return int(offsets[-1].item())
    return None


def masked_hadamard(
    source,
    hadamard20,
    output,
    temporary,
    lane_count,
    *,
    max_lanes,
    transpose=False,
    scale=None,
    row_index=None,
    expert_index=None,
):
    """Graph-safe Hadamard over the device-published compact lane prefix."""
    n = output.shape[1]
    if n not in (640, 1280, 2560) or source.shape[1] != n:
        raise ValueError("unsupported masked Hadamard width")
    if output.shape[0] < max_lanes or temporary.shape != output.shape:
        raise ValueError("invalid masked Hadamard storage")
    gather = row_index is not None
    apply_scale = scale is not None
    if not (gather == apply_scale == (expert_index is not None)):
        raise ValueError("gathered scale requires row and expert indices")
    row_index = output if row_index is None else row_index
    expert_index = output if expert_index is None else expert_index
    scale = output if scale is None else scale
    p = n // 20
    _masked_hadamard_power2[(max_lanes, 20)](
        source,
        scale,
        row_index,
        expert_index,
        temporary,
        lane_count,
        source.stride(0),
        n,
        p,
        max_lanes,
        gather,
        apply_scale,
        num_warps=4,
    )
    _masked_hadamard20[(max_lanes, tr.cdiv(20, 16), tr.cdiv(p, 32))](
        temporary,
        hadamard20,
        output,
        lane_count,
        n,
        p,
        max_lanes,
        transpose,
        16,
        32,
        num_warps=4,
        num_stages=1,
    )


def masked_e8_activation(
    gate,
    up,
    sv_gate,
    sv_up,
    su_down,
    expert_index,
    output,
    lane_count,
    *,
    max_lanes,
):
    """Apply per-expert output scales, SiLU-times-up and down-input SU."""
    width = output.shape[1]
    if width != 640 or output.shape[0] < max_lanes:
        raise ValueError("invalid masked E8 activation storage")
    _masked_e8_activation[(max_lanes, tr.cdiv(width, 256))](
        gate,
        up,
        sv_gate,
        sv_up,
        su_down,
        expert_index,
        output,
        lane_count,
        width,
        gate.stride(0),
        up.stride(0),
        max_lanes,
        256,
        num_warps=4,
    )


def masked_e8_scatter(
    value,
    sv_down,
    token_index,
    expert_index,
    route_weight,
    partial,
    lane_count,
    *,
    max_lanes,
):
    """Scale and route valid compact lanes into the per-token FP32 partial."""
    hidden = partial.shape[1]
    if hidden != 2560 or value.shape[0] < max_lanes:
        raise ValueError("invalid masked E8 scatter storage")
    _masked_e8_scatter[(max_lanes, tr.cdiv(hidden, 256))](
        value,
        sv_down,
        token_index,
        expert_index,
        route_weight,
        partial,
        lane_count,
        hidden,
        max_lanes,
        256,
        num_warps=4,
    )


def masked_e8_scatter_deterministic(
    value,
    sv_down,
    token_index,
    expert_index,
    route_weight,
    partial,
    lane_count,
    *,
    max_lanes,
):
    """Fixed-order compact-route reduction for latency-critical small M."""
    hidden = partial.shape[1]
    tokens = partial.shape[0]
    if hidden != 2560 or value.shape[0] < max_lanes or not 1 <= tokens <= 4:
        raise ValueError("invalid deterministic E8 scatter storage")
    _masked_e8_scatter_deterministic[(tokens, tr.cdiv(hidden, 256))](
        value,
        sv_down,
        token_index,
        expert_index,
        route_weight,
        partial,
        lane_count,
        hidden,
        tokens,
        max_lanes,
        256,
        num_warps=4,
    )


def build_tasks(
    counts,
    offsets,
    task_counter,
    task_expert,
    task_begin,
    task_rows,
    *,
    max_tokens,
    rows=8,
):
    task_counter.zero_()
    task_expert.fill_(-1)
    task_begin.zero_()
    task_rows.zero_()
    _build_tasks[(counts.numel(), tr.cdiv(max_tokens, rows))](
        counts,
        offsets,
        task_counter,
        task_expert,
        task_begin,
        task_rows,
        counts.numel(),
        tr.cdiv(max_tokens, rows),
        rows,
        num_warps=1,
    )


def grouped_e8_mm(
    x,
    q_resident,
    q_cold,
    codebook,
    out,
    task_expert,
    task_begin,
    task_rows,
    *,
    layer,
    resident,
    block_n=32,
    block_k=64,
):
    max_tasks, n, k = task_expert.numel(), out.shape[1], x.shape[1]
    _grouped_e8_mm[(max_tasks, tr.cdiv(n, block_n))](
        x,
        q_resident,
        q_cold,
        codebook,
        out,
        task_expert,
        task_begin,
        task_rows,
        layer,
        resident,
        q_cold.shape[0],
        max_tasks,
        n,
        k,
        block_n,
        block_k,
        num_warps=4,
        num_stages=3,
    )


def persistent_grouped_e8_mm(
    x,
    q_resident,
    q_cold,
    codebook,
    out,
    task_counter,
    task_expert,
    task_begin,
    task_rows,
    *,
    layer,
    resident,
    workers=128,
    block_m=8,
    block_n=32,
    block_k=64,
    cold_map=None,
    resident_map=None,
):
    """Run only the tasks published by ``build_tasks`` on a fixed GPU grid."""
    n, k = out.shape[1], x.shape[1]
    _persistent_grouped_e8_mm[(workers,)](
        x,
        q_resident,
        q_cold,
        q_cold if cold_map is None else cold_map,
        q_cold if resident_map is None else resident_map,
        codebook,
        out,
        task_counter,
        task_expert,
        task_begin,
        task_rows,
        layer,
        resident,
        q_cold.shape[0],
        n,
        k,
        block_m,
        block_n,
        block_k,
        workers,
        cold_map is not None,
        resident_map is not None,
        num_warps=4,
        num_stages=3,
    )


def persistent_grouped_e8_compact_mm(
    x,
    q_resident,
    q_cold,
    packed_abs,
    out,
    task_counter,
    task_expert,
    task_begin,
    task_rows,
    *,
    layer,
    resident,
    workers=128,
    block_n=32,
    block_k=64,
):
    """Run exact E8P12 decode from the compact 256-entry representation."""
    n, k = out.shape[1], x.shape[1]
    _persistent_grouped_e8_compact_mm[(workers,)](
        x,
        q_resident,
        q_cold,
        packed_abs,
        out,
        task_counter,
        task_expert,
        task_begin,
        task_rows,
        layer,
        resident,
        q_cold.shape[0],
        n,
        k,
        block_n,
        block_k,
        workers,
        num_warps=4,
        num_stages=3,
    )


def expert_grid_e8_mm(
    x,
    q_resident,
    q_cold,
    codebook,
    out,
    active_experts,
    counts,
    offsets,
    *,
    layer,
    resident,
    max_rows,
    block_n=32,
    block_k=64,
):
    """Regular expert/row grid over compacted real-route rows."""
    n, k = out.shape[1], x.shape[1]
    _expert_grid_e8_mm[
        (active_experts.numel() * tr.cdiv(n, block_n), tr.cdiv(max_rows, 8))
    ](
        x,
        q_resident,
        q_cold,
        codebook,
        out,
        active_experts,
        counts,
        offsets,
        layer,
        resident,
        q_cold.shape[0],
        n,
        k,
        block_n,
        block_k,
        num_warps=4,
        num_stages=3,
    )
