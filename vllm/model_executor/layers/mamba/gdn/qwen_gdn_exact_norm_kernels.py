# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Original generated GDN norm arithmetic with explicit launch ownership.

POC only. Mechanical extraction from SHA-verified saved original target kernels.
Only names/decorators, xnumel binding and H15 row-count constants are generalized.
Do not change casts, expression associations, masks or floating-point literals.
"""

# ruff: noqa: F841, E501
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


@triton.jit
def gated_h18(
    in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr
):
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(
        in_ptr0 + (r0_1 + 128 * x0), xmask, eviction_policy="evict_first", other=0.0
    ).to(tl.float32)
    tmp13 = tl.load(
        in_ptr1 + (r0_1 + 128 * x0) % 2304 % 128,
        xmask,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp16 = tl.load(
        in_ptr2 + (3840 + 6144 * (x0 // 18) + (r0_1 + 128 * x0) % 2304),
        xmask,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([1, 1], 128.0, tl.float32)
    tmp8 = tmp6 / tmp7
    tmp9 = tl.full([1, 1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = -tmp17
    tmp19 = libdevice.exp(tmp18)
    tmp20 = tl.full([1, 1], 1.0, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp17 / tmp21
    tmp23 = tmp15 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 128 * x0), tmp24, xmask)


@triton.jit
def sum_h15(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(
        in_ptr0 + (r0_1 + 128 * x0), xmask, eviction_policy="evict_first", other=0.0
    ).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def gated_h15(
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rows, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2304
    x1 = xindex // 2304
    x2 = xindex
    tmp0 = x0.to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1], 1920, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(
        in_ptr0 + (128 * ((1920 * x1 + x0) // 128 % (15 * rows)) + x0 % 128),
        tmp6 & xmask,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.load(
        in_ptr1 + (1920 * x1 + x0) // 128 % (15 * rows),
        tmp6 & xmask,
        eviction_policy="evict_last",
        other=0.0,
    )
    tmp10 = tl.full([1], 128.0, tl.float32)
    tmp11 = tmp9 / tmp10
    tmp12 = tl.full([1], 1e-06, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp16 = tl.load(
        in_ptr2 + x0 % 128, tmp6 & xmask, eviction_policy="evict_last", other=0.0
    ).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 * tmp17
    tmp19 = tl.load(
        in_ptr3
        + (
            3840
            + 128 * ((1920 * x1 + x0) // 128 % (15 * rows) % 15)
            + 6144 * ((1920 * x1 + x0) // 1920 % rows)
            + x0 % 128
        ),
        tmp6 & xmask,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = -tmp20
    tmp22 = libdevice.exp(tmp21)
    tmp23 = tl.full([1], 1.0, tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp20 / tmp24
    tmp26 = tmp18 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp6, tmp27, tmp28)
    tmp30 = tmp0 >= tmp5
    tmp31 = tl.full([1], 2304, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.full([1], 0.0, tl.float32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp6, tmp29, tmp35)
    tl.store(out_ptr0 + x2, tmp36, xmask)
