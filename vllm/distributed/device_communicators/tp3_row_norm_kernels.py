"""Checkpoint-bound H5120 arithmetic forms for the opt-in row continuation.

Extracted from the accepted target AOT kernels; only decorators and function
names change. This module does not load executable compiler-cache files.
Launch configurations remain bound separately to the original destinations.
The row path is experimental; these bodies retain the accepted AOT arithmetic.
"""

# ruff: noqa: F841, E501
# Preserve generated unused assignments until native equivalence is proved.

import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


@triton.jit
def norm_9dffa1cf3ba0(
    in_ptr0,
    in_ptr1,
    out_ptr1,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(
            in_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(
            in_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp14 = tl.load(
            in_ptr1 + r0_1, r0_mask, eviction_policy="evict_last", other=0.0
        ).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 5120.0, tl.float32)
        tmp9 = tmp4 / tmp8
        tmp10 = tl.full([1, 1], 1e-06, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 1.0, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp13 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 5120 * x0), tmp19, r0_mask & xmask)


@triton.jit
def norm_27f5f8cce2f4(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr1,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(
            in_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp2 = tl.load(
            in_ptr1 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(
            in_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp11 = tl.load(
            in_ptr1 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp20 = tl.load(
            in_ptr2 + r0_1, r0_mask, eviction_policy="evict_last", other=0.0
        ).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 5120.0, tl.float32)
        tmp15 = tmp7 / tmp14
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 1.0, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp19 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 5120 * x0), tmp25, r0_mask & xmask)


@triton.jit
def norm_a89f489db113(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    in_ptr3,
    out_ptr1,
    out_ptr2,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(
            in_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp2 = tl.load(
            in_ptr1 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp4 = tl.load(
            in_ptr2 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 5120 * x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(
            in_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp17 = tl.load(
            in_ptr1 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp19 = tl.load(
            in_ptr2 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp31 = tl.load(
            in_ptr3 + r0_1, r0_mask, eviction_policy="evict_last", other=0.0
        ).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 5120.0, tl.float32)
        tmp26 = tmp12 / tmp25
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.full([1, 1], 1.0, tl.float32)
        tmp34 = tmp32 + tmp33
        tmp35 = tmp30 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 5120 * x0), tmp36, r0_mask & xmask)


@triton.jit
def norm_6331af9b3b5b(
    in_out_ptr0,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(
            in_out_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp2 = tl.load(
            in_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp4 = tl.load(
            in_ptr1 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_last",
            other=0.0,
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(
            in_out_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp16 = tl.load(
            in_ptr0 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp18 = tl.load(
            in_ptr1 + (r0_1 + 5120 * x0),
            r0_mask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        tmp30 = tl.load(
            in_ptr2 + r0_1, r0_mask, eviction_policy="evict_last", other=0.0
        ).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp15 + tmp22
        tmp24 = tl.full([1, 1], 5120.0, tl.float32)
        tmp25 = tmp12 / tmp24
        tmp26 = tl.full([1, 1], 1e-06, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.full([1, 1], 1.0, tl.float32)
        tmp33 = tmp31 + tmp32
        tmp34 = tmp29 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 5120 * x0), tmp35, r0_mask & xmask)


# Exact generated names are retained as immutable provenance keys.
KERNELS = {
    "9dffa1cf3ba0a7da380473e0919df6832ff25f8bd78776798e9b77cd6ba9f422": norm_9dffa1cf3ba0,
    "27f5f8cce2f4944dce69c22a86e5d108e6a64dea26bfd19683e20200e03646ec": norm_27f5f8cce2f4,
    "a89f489db113ed212a1e9f00d6a051c4189c083124dd160ae70eda3edd9669e6": norm_a89f489db113,
    "6331af9b3b5b2f0f7d7a27008e2575ffbadfd5134df834bd2338e019bc437d54": norm_6331af9b3b5b,
}
