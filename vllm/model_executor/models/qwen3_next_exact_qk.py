# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""POC exact Q/K sum boundary; normalization/RoPE still fuse outside the op."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from .qwen3_next_exact_qk_kernels import qk_sum_original


def validate_qkv(qkv):
    if (
        qkv.ndim != 2
        or qkv.shape[1] != 6144
        or not 1 <= qkv.shape[0] <= 4096
        or qkv.dtype != torch.bfloat16
        or not qkv.is_contiguous()
    ):
        raise ValueError("unproved ready QKV sum layout")


def validate_owner(owner, expected_device=None):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    if (
        owner.num_heads != 8
        or owner.num_kv_heads != 4
        or owner.head_dim != 256
        or not owner.attn_output_gate
        or owner.use_fused_qk_norm_rope_gate
    ):
        raise ValueError("unproved ready Q/K attention recipe")
    for norm in (owner.q_norm, owner.k_norm):
        if (
            type(norm) is not GemmaRMSNorm
            or norm.variance_epsilon != 1e-6
            or norm.weight.shape != (256,)
            or norm.weight.dtype != torch.bfloat16
            or not norm.weight.is_contiguous()
        ):
            raise ValueError("unproved ready Q/K norm recipe")
        if norm.weight.device != owner.q_norm.weight.device or (
            expected_device is not None and norm.weight.device != expected_device
        ):
            raise ValueError("ready Q/K norm device mismatch")


def exact_qk_sums(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_qkv(qkv)
    if not qkv.is_cuda:
        raise ValueError("exact Q/K sum is a CUDA-only recipe")
    m = qkv.shape[0]
    qs = torch.empty((m * 8,), dtype=torch.float32, device=qkv.device)
    ks = torch.empty((m * 4,), dtype=torch.float32, device=qkv.device)
    qk_sum_original[((m * 8 + 1) // 2 + (m * 4 + 1) // 2,)](
        qkv,
        qs,
        ks,
        m * 8,
        m * 4,
        XBLOCK=2,
        R0_BLOCK=256,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return qs, ks


def exact_qk_sums_fake(qkv):
    return (
        qkv.new_empty((qkv.shape[0] * 8,), dtype=torch.float32),
        qkv.new_empty((qkv.shape[0] * 4,), dtype=torch.float32),
    )


def normalize_from_sum(value, weight, sums, epsilon):
    dtype = value.dtype
    value = value.to(torch.float32)
    variance = sums.reshape(*value.shape[:-1], 1) / 256.0
    normalized = value * torch.rsqrt(variance + epsilon)
    return (normalized * (weight.float() + 1.0)).to(dtype)


direct_register_custom_op(
    op_name="ready_exact_qk_sums", op_func=exact_qk_sums, fake_impl=exact_qk_sums_fake
)
