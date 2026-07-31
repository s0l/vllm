# SPDX-License-Identifier: Apache-2.0
"""FP8-E4M3 draft-side lm_head read path (AG2 lever P2).

Draft token selection is argmax-only; at temperature 0 the emitted tokens are
always the target's argmax, so draft head precision affects acceptance
(speed) only, never output text. Offline oracle + real-hidden measurement:
argmax agreement 99.6%, flips only at margins <= 0.126.

The target's BF16 lm_head module is shared with the draft
(`_maybe_share_lm_head`) and must not be modified. This module keeps a
separate per-row-scaled FP8 copy (+~424 MiB/rank) and a Triton GEMV that
reads half the bytes of the BF16 path.

Enabled by AG2_VLLM_DRAFT_LMHEAD_FP8=1. Used only for small-M draft logits.
"""

from __future__ import annotations

import os

import torch
from torch import nn

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.triton_utils import tl, triton


@triton.jit
def _fp8_gemv_kernel(
    out_ptr,  # [M, V] fp32
    w_ptr,  # [V, K] fp8e4m3
    scale_ptr,  # [V] fp32
    x_ptr,  # [M, K] fp32
    V,
    K,
    stride_wv,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_v * ROWS + tl.arange(0, ROWS)
    row_mask = rows < V
    acc = tl.zeros((ROWS,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + pid_m * K + ks).to(tl.float32)
        w = tl.load(
            w_ptr + rows[:, None] * stride_wv + ks[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0)
    tl.store(out_ptr + pid_m * V + rows, acc * scale, mask=row_mask)


class Fp8DraftHead:
    """Per-row-scaled FP8 copy of a vocab-parallel lm_head shard."""

    def __init__(self, weight: torch.Tensor):
        w = weight.detach()
        amax = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
        self.scale = (amax / 448.0).squeeze(1).contiguous()
        self.w8 = (
            (w.float() / (amax / 448.0))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        self.vocab, self.k = w.shape

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("AG2_VLLM_DRAFT_LMHEAD_FP8", "0") == "1"

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.float().contiguous()
        m = x.shape[0]
        out = torch.empty(m, self.vocab, dtype=torch.float32, device=x.device)
        rows = 4
        grid = (triton.cdiv(self.vocab, rows), m)
        _fp8_gemv_kernel[grid](
            out,
            self.w8,
            self.scale,
            x,
            self.vocab,
            self.k,
            self.w8.stride(0),
            BLOCK_K=512,
            ROWS=rows,
        )
        return out


class Ag2SharedFp8LMHeadMethod(QuantizeMethodBase):
    """Per-channel W8A8 projection for a shared target/draft lm_head."""

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise RuntimeError("AG2 shared FP8 lm_head is installed after weight loading")

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_float = x.float()
        x_scale = x_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        x8 = (x_float / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        out = ops.cutlass_scaled_mm(
            x8,
            layer.weight.T,
            scale_a=x_scale.float(),
            scale_b=layer.ag2_fp8_weight_scale[None, :],
            out_dtype=torch.bfloat16,
        )
        if bias is not None:
            out = out + bias
        return out


def shared_fp8_head_enabled() -> bool:
    return os.environ.get("AG2_VLLM_SHARED_LMHEAD_FP8", "0") == "1"


def install_shared_fp8_lm_head(lm_head: nn.Module) -> tuple[int, int]:
    """Replace a loaded shared BF16 head with one per-channel FP8 copy."""
    if getattr(lm_head, "ag2_shared_fp8_installed", False):
        return 0, 0
    weight = lm_head.weight.detach()
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError(
            "AG2 shared FP8 lm_head requires a 2D BF16 source weight, got "
            f"{tuple(weight.shape)} {weight.dtype}"
        )
    source_bytes = weight.nbytes
    amax = weight.float().abs().amax(dim=1).clamp(min=1e-12)
    scale = (amax / 448.0).contiguous()
    weight8 = (
        (weight.float() / scale[:, None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    del lm_head.weight
    lm_head.register_parameter(
        "weight", nn.Parameter(weight8, requires_grad=False)
    )
    lm_head.register_buffer("ag2_fp8_weight_scale", scale)
    lm_head.quant_method = Ag2SharedFp8LMHeadMethod()
    lm_head.ag2_shared_fp8_installed = True
    installed_bytes = weight8.nbytes + scale.nbytes
    return source_bytes, installed_bytes
