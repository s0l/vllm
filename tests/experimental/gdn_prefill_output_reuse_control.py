#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-shape CUDA control for FLA GDN prefill output-buffer reuse."""

from __future__ import annotations

import gc
import json

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops.chunk import (
    chunk_gated_delta_rule,
)

TOKENS = 7056
KEY_HEADS = 6
# TP3 rank 0 owns six complete 1:3 K:V groups (18 value heads);
# ranks 1/2 own five groups (15 value heads). Rank 0 is the peak-memory shape.
VALUE_HEADS = 18
HEAD_DIM = 128


@torch.inference_mode()
def measure(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor | None,
    value_output: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    actual, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        core_attn_out=output,
        value_out=value_output,
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    return actual, final_state, peak


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    q = F.normalize(
        torch.randn(1, TOKENS, KEY_HEADS, HEAD_DIM, device=device, dtype=dtype).float(),
        dim=-1,
    ).to(dtype)
    k = F.normalize(torch.randn_like(q).float(), dim=-1).to(dtype)
    base_v = 0.1 * torch.randn(
        1, TOKENS, VALUE_HEADS, HEAD_DIM, device=device, dtype=dtype
    )
    g = -0.01 * torch.rand(1, TOKENS, VALUE_HEADS, device=device, dtype=torch.float32)
    beta = torch.sigmoid(
        torch.randn(1, TOKENS, VALUE_HEADS, device=device, dtype=torch.float32)
    )
    initial_state = 0.05 * torch.randn(
        1,
        VALUE_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )

    # Warm every involved kernel specialization before measuring.
    warm_reference_v = base_v.clone()
    warm_reference, warm_reference_state, _ = measure(
        q, k, warm_reference_v, g, beta, initial_state, None, None
    )
    del warm_reference, warm_reference_state, warm_reference_v

    warm_v = base_v.clone()
    warm_output = torch.empty_like(base_v)
    warm, warm_state, _ = measure(
        q, k, warm_v, g, beta, initial_state, warm_output, warm_v
    )
    del warm, warm_state, warm_output, warm_v

    reference_v = base_v.clone()
    reference, reference_state, allocating_peak = measure(
        q, k, reference_v, g, beta, initial_state, None, None
    )
    reference_cpu = reference.cpu()
    reference_state_cpu = reference_state.cpu()
    del reference, reference_state

    output_only_v = base_v.clone()
    output_only = torch.empty_like(base_v)
    output_reused, output_reused_state, output_reused_peak = measure(
        q, k, output_only_v, g, beta, initial_state, output_only, None
    )
    assert output_reused.data_ptr() == output_only.data_ptr()
    output_reused_cpu = output_reused.cpu()
    torch.testing.assert_close(output_reused_cpu, reference_cpu, rtol=0, atol=2e-3)
    output_reused_state_cpu = output_reused_state.cpu()
    output_only_state_error = (
        output_reused_state_cpu.float() - reference_state_cpu.float()
    ).abs()
    torch.testing.assert_close(
        output_reused_state_cpu,
        reference_state_cpu,
        rtol=1e-2,
        atol=1e-2,
    )
    assert output_only_state_error.mean().item() < 6e-4

    reused_v = base_v.clone()
    output = torch.empty_like(base_v)
    reused, reused_state, reused_peak = measure(
        q, k, reused_v, g, beta, initial_state, output, reused_v
    )
    assert reused.data_ptr() == output.data_ptr()
    reused_cpu = reused.cpu()
    output_error = (reused_cpu.float() - reference_cpu.float()).abs()
    torch.testing.assert_close(reused_cpu, reference_cpu, rtol=0, atol=2e-3)
    reused_state_cpu = reused_state.cpu()
    state_error = (reused_state_cpu.float() - reference_state_cpu.float()).abs()
    torch.testing.assert_close(
        reused_state_cpu, reference_state_cpu, rtol=1e-2, atol=1e-2
    )
    assert state_error.mean().item() < 6e-4

    output_bytes = base_v.numel() * base_v.element_size()
    output_saved_peak = allocating_peak - output_reused_peak
    value_saved_peak = output_reused_peak - reused_peak
    total_saved_peak = allocating_peak - reused_peak
    assert output_saved_peak >= output_bytes, (output_saved_peak, output_bytes)
    assert value_saved_peak >= output_bytes, (value_saved_peak, output_bytes)

    # Exercise irregular variable-length boundaries, including one-token,
    # exact-chunk and cross-chunk sequences.
    multiseq_tokens = 289
    multiseq_cu = torch.tensor(
        [0, 1, 64, 128, 193, multiseq_tokens],
        device=device,
        dtype=torch.int32,
    )
    multiseq_state = initial_state.expand(len(multiseq_cu) - 1, -1, -1, -1).clone()
    with torch.inference_mode():
        multiseq_reference, multiseq_reference_state = chunk_gated_delta_rule(
            q=q[:, :multiseq_tokens],
            k=k[:, :multiseq_tokens],
            v=base_v[:, :multiseq_tokens].clone(),
            g=g[:, :multiseq_tokens],
            beta=beta[:, :multiseq_tokens],
            initial_state=multiseq_state,
            output_final_state=True,
            cu_seqlens=multiseq_cu,
            use_qk_l2norm_in_kernel=False,
        )
        multiseq_v = base_v[:, :multiseq_tokens].clone()
        multiseq_output = torch.empty_like(multiseq_v)
        multiseq_reused, multiseq_reused_state = chunk_gated_delta_rule(
            q=q[:, :multiseq_tokens],
            k=k[:, :multiseq_tokens],
            v=multiseq_v,
            g=g[:, :multiseq_tokens],
            beta=beta[:, :multiseq_tokens],
            initial_state=multiseq_state,
            output_final_state=True,
            cu_seqlens=multiseq_cu,
            use_qk_l2norm_in_kernel=False,
            core_attn_out=multiseq_output,
            value_out=multiseq_v,
        )
    multiseq_output_error = (multiseq_reused.float() - multiseq_reference.float()).abs()
    multiseq_state_error = (
        multiseq_reused_state.float() - multiseq_reference_state.float()
    ).abs()
    torch.testing.assert_close(
        multiseq_reused, multiseq_reference, rtol=1e-2, atol=2e-2
    )
    torch.testing.assert_close(
        multiseq_reused_state,
        multiseq_reference_state,
        rtol=1e-2,
        atol=1e-2,
    )
    assert multiseq_state_error.mean().item() < 6e-4
    print(
        json.dumps(
            {
                "gdn_prefill_output_reuse": "passed",
                "shape": list(base_v.shape),
                "output_bytes": output_bytes,
                "allocating_peak_bytes": allocating_peak,
                "output_only_reused_peak_bytes": output_reused_peak,
                "reused_peak_bytes": reused_peak,
                "output_saved_peak_bytes": output_saved_peak,
                "value_saved_peak_bytes": value_saved_peak,
                "total_saved_peak_bytes": total_saved_peak,
                "output_max_abs_error": output_error.max().item(),
                "output_mean_abs_error": output_error.mean().item(),
                "state_max_abs_error": state_error.max().item(),
                "state_mean_abs_error": state_error.mean().item(),
                "multiseq_cu_seqlens": multiseq_cu.cpu().tolist(),
                "multiseq_output_max_abs_error": (multiseq_output_error.max().item()),
                "multiseq_output_mean_abs_error": (multiseq_output_error.mean().item()),
                "multiseq_state_max_abs_error": (multiseq_state_error.max().item()),
                "multiseq_state_mean_abs_error": (multiseq_state_error.mean().item()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
