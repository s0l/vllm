# SPDX-License-Identifier: Apache-2.0
"""Default-off ARC-style variable-K activation preparation for NVFP4 B12x."""

from __future__ import annotations

import torch

from vllm._custom_ops import create_fp4_output_tensors, scaled_fp4_quant
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _round_e2m1(x):
    sign = tl.where(x < 0.0, -1.0, 1.0)
    value = tl.abs(x)
    result = tl.where(value > 5.0, 6.0, 0.0)
    result = tl.where((value >= 3.5) & (value <= 5.0), 4.0, result)
    result = tl.where((value > 2.5) & (value < 3.5), 3.0, result)
    result = tl.where((value >= 1.75) & (value <= 2.5), 2.0, result)
    result = tl.where((value > 1.25) & (value < 1.75), 1.5, result)
    result = tl.where((value >= 0.75) & (value <= 1.25), 1.0, result)
    result = tl.where((value > 0.25) & (value < 0.75), 0.5, result)
    return result * sign


@triton.jit
def _decode_e2m1(nibble):
    magnitude = nibble & 7
    sign = (nibble >> 3) & 1
    value = tl.where(magnitude == 0, 0.0, 6.0)
    value = tl.where(magnitude == 1, 0.5, value)
    value = tl.where(magnitude == 2, 1.0, value)
    value = tl.where(magnitude == 3, 1.5, value)
    value = tl.where(magnitude == 4, 2.0, value)
    value = tl.where(magnitude == 5, 3.0, value)
    value = tl.where(magnitude == 6, 4.0, value)
    return tl.where(sign == 1, -value, value)


@triton.jit
def _encode_e2m1(value):
    magnitude = tl.abs(value)
    code = tl.where(magnitude == 0.0, 0, 7)
    code = tl.where(magnitude == 0.5, 1, code)
    code = tl.where(magnitude == 1.0, 2, code)
    code = tl.where(magnitude == 1.5, 3, code)
    code = tl.where(magnitude == 2.0, 4, code)
    code = tl.where(magnitude == 3.0, 5, code)
    code = tl.where(magnitude == 4.0, 6, code)
    return code | tl.where(value < 0.0, 8, 0)


@triton.jit
def _write_residual_tail_kernel(
    x_ptr,
    selected_ptr,
    global_scale_ptr,
    packed_ptr,
    scales_ptr,
    k: tl.constexpr,
    augmented_k: tl.constexpr,
):
    row = tl.program_id(0)
    residual_group = tl.program_id(1)
    pair_offsets = tl.arange(0, 8)
    low_channels = tl.load(selected_ptr + residual_group * 16 + pair_offsets * 2)
    high_channels = tl.load(
        selected_ptr + residual_group * 16 + pair_offsets * 2 + 1
    )
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
    low_scale_k = low_channels // 16
    high_scale_k = high_channels // 16
    common_scale_offset = (row & 31) * 16 + ((row >> 5) & 3) * 4
    low_scale_offsets = (
        ((row // 128) * num_k_tiles + low_scale_k // 4) * 512
        + common_scale_offset
        + (low_scale_k & 3)
    )
    high_scale_offsets = (
        ((row // 128) * num_k_tiles + high_scale_k // 4) * 512
        + common_scale_offset
        + (high_scale_k & 3)
    )
    low_scales = tl.load(scales_ptr + low_scale_offsets).to(
        tl.float8e4nv, bitcast=True
    ).to(tl.float32)
    high_scales = tl.load(scales_ptr + high_scale_offsets).to(
        tl.float8e4nv, bitcast=True
    ).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    low_qdq = _decode_e2m1(low_nibbles) * (low_scales / global_scale)
    high_qdq = _decode_e2m1(high_nibbles) * (high_scales / global_scale)
    low_source = tl.load(x_ptr + row * k + low_channels).to(tl.float32)
    high_source = tl.load(x_ptr + row * k + high_channels).to(tl.float32)
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
    low_quantized = _round_e2m1(tl.clamp(low_residual * inverse, -6.0, 6.0))
    high_quantized = _round_e2m1(tl.clamp(high_residual * inverse, -6.0, 6.0))
    packed_tail = _encode_e2m1(low_quantized) | (
        _encode_e2m1(high_quantized) << 4
    )
    tl.store(
        packed_ptr + packed_row + k // 2 + residual_group * 8 + pair_offsets,
        packed_tail,
    )
    tail_scale_k: tl.constexpr = k // 16
    tail_scale_index = tail_scale_k + residual_group
    tail_scale_offset = (
        ((row // 128) * num_k_tiles + tail_scale_index // 4) * 512
        + common_scale_offset
        + (tail_scale_index & 3)
    )
    tl.store(
        scales_ptr + tail_scale_offset,
        tail_scale_fp8.to(tl.uint8, bitcast=True),
    )


def _arc_quantize_impl(
    x: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_count = selected.shape[0] if selected.ndim == 1 else 0
    k = x.shape[-1]
    if (
        k <= 0
        or k % 64 != 0
        or selected_count not in (64, 256, 512)
        or selected.dtype != torch.int32
    ):
        raise ValueError(
            "ARC requires BF16/FP16 [...,K] with K divisible by 64 and "
            "int32 selected[64|256|512]"
        )
    original_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    packed, scales = scaled_fp4_quant(
        x,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend="b12x",
        padded_n=x.shape[1] + selected_count,
    )
    _write_residual_tail_kernel[(x.shape[0], selected_count // 16)](
        x,
        selected,
        input_global_scale_inv,
        packed,
        scales.view(torch.uint8),
        k=k,
        augmented_k=k + selected_count,
    )
    return packed.view(-1, (original_shape[-1] + selected_count) // 2), scales


def _arc_quantize_fake(
    x: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del input_global_scale_inv
    selected_count = selected.shape[0]
    m = x.numel() // x.shape[-1]
    packed, scales = create_fp4_output_tensors(
        m,
        x.shape[-1],
        x.device,
        True,
        padded_n=x.shape[-1] + selected_count,
    )
    return packed, scales.view(torch.float8_e4m3fn)


direct_register_custom_op(
    "ag2_nvfp4_arc_quantize",
    _arc_quantize_impl,
    fake_impl=_arc_quantize_fake,
)


def ag2_nvfp4_arc_quantize(
    x: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.ag2_nvfp4_arc_quantize(
        x, input_global_scale_inv, selected
    )
