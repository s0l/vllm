# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exactness oracle for a final-only GDN journal commit kernel.

Research POC only.  The candidate kernel intentionally omits query/output,
loads one committed FP32 recurrent state, replays the accepted 1..4 transition
inputs, and stores the matrix once.
"""

from __future__ import annotations

import json

import torch

from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.triton_utils import tl, triton

NUM_LAYERS = 3
NUM_REQUESTS = 4
QUERY_LEN = 4
NUM_K_HEADS = 6
NUM_V_HEADS = 18
HEAD_DIM = 128
COMMITTED_BLOCK_IDS = (7, 2, 11, 4)
POOL_BLOCKS = max(COMMITTED_BLOCK_IDS) + 1


@triton.jit
def _journal_commit_kernel(
    k_ptr,
    v_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    state_ptr,
    state_indices_ptr,
    accepted_ptr,
    scale,
    state_layer_stride: tl.constexpr,
    state_block_stride: tl.constexpr,
    journal_layer_k_stride: tl.constexpr,
    journal_layer_v_stride: tl.constexpr,
    journal_layer_gate_stride: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_lnh = tl.program_id(2)
    i_layer = i_lnh // (N * HV)
    i_nh = i_lnh % (N * HV)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(state_indices_ptr + i_n).to(tl.int64)
    if state_idx <= 0:
        return
    p_state = (
        state_ptr
        + i_layer * state_layer_stride
        + state_idx * state_block_stride
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(p_state, mask=mask_h, other=0).to(tl.float32)

    p_k = k_ptr + i_layer * journal_layer_k_stride + ((i_n * T) * H + i_h) * K + o_k
    p_v = v_ptr + i_layer * journal_layer_v_stride + ((i_n * T) * HV + i_hv) * V + o_v
    p_a = a_ptr + i_layer * journal_layer_gate_stride + (i_n * T) * HV + i_hv
    p_b = b_ptr + i_layer * journal_layer_gate_stride + (i_n * T) * HV + i_hv
    p_A_log = a_log_ptr + i_layer * HV + i_hv
    p_dt_bias = dt_bias_ptr + i_layer * HV + i_hv
    accepted = tl.load(accepted_ptr + i_n).to(tl.int32)

    for i_t in range(0, T):
        if i_t < accepted:
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_b = tl.load(p_b).to(tl.float32)

            x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
            softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
            b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x
            b_beta = tl.sigmoid(b_b)

            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            b_v *= b_beta
            b_h += b_v[:, None] * b_k[None, :]

        p_k += H * K
        p_v += HV * V
        p_a += HV
        p_b += HV

    tl.store(p_state, b_h, mask=mask_h)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "differing": int(torch.count_nonzero(delta).item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(212)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    state_dtype = torch.float32
    accepted = torch.arange(1, QUERY_LEN + 1, device=device, dtype=torch.int32)
    state_indices = torch.tensor(COMMITTED_BLOCK_IDS, device=device, dtype=torch.int32)
    state_indices_long = state_indices.to(torch.long)

    k = torch.randn(
        NUM_LAYERS,
        NUM_REQUESTS,
        QUERY_LEN,
        NUM_K_HEADS,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        NUM_LAYERS,
        NUM_REQUESTS,
        QUERY_LEN,
        NUM_V_HEADS,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(
        NUM_LAYERS,
        NUM_REQUESTS,
        QUERY_LEN,
        NUM_V_HEADS,
        device=device,
        dtype=dtype,
    )
    b = torch.randn_like(a)
    a_log = torch.randn(
        NUM_LAYERS, NUM_V_HEADS, device=device, dtype=torch.float32
    ).mul_(0.1)
    dt_bias = torch.randn_like(a_log).mul_(0.1)
    initial = torch.randn(
        NUM_LAYERS,
        NUM_REQUESTS,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=state_dtype,
    ).mul_(0.02)

    reference = torch.randn(
        NUM_LAYERS,
        POOL_BLOCKS,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=state_dtype,
    ).mul_(0.02)
    for layer_idx in range(NUM_LAYERS):
        reference[layer_idx].index_copy_(0, state_indices_long, initial[layer_idx])
    candidate = reference.clone()
    candidate_before = candidate.clone()

    # Reference uses the exact current kernel independently per layer, with
    # variable-length packed prefixes and repeated destination indices.
    replay_cu = torch.zeros(NUM_REQUESTS + 1, device=device, dtype=torch.int32)
    torch.cumsum(accepted, dim=0, out=replay_cu[1:])
    repeated_indices = state_indices.view(NUM_REQUESTS, 1).expand(-1, QUERY_LEN)
    for layer_idx in range(NUM_LAYERS):
        packed_k = torch.cat(
            [
                k[layer_idx, req_idx, :length]
                for req_idx, length in enumerate(accepted.tolist())
            ],
            dim=0,
        ).unsqueeze(0)
        packed_v = torch.cat(
            [
                v[layer_idx, req_idx, :length]
                for req_idx, length in enumerate(accepted.tolist())
            ],
            dim=0,
        ).unsqueeze(0)
        packed_a = torch.cat(
            [
                a[layer_idx, req_idx, :length]
                for req_idx, length in enumerate(accepted.tolist())
            ],
            dim=0,
        )
        packed_b = torch.cat(
            [
                b[layer_idx, req_idx, :length]
                for req_idx, length in enumerate(accepted.tolist())
            ],
            dim=0,
        )
        # Query has no effect on the recurrent state; zeros make that contract
        # explicit while preserving the current reference kernel.
        packed_q = torch.zeros_like(packed_k)
        fused_sigmoid_gating_delta_rule_update(
            A_log=a_log[layer_idx],
            a=packed_a,
            b=packed_b,
            dt_bias=dt_bias[layer_idx],
            q=packed_q,
            k=packed_k,
            v=packed_v,
            initial_state=reference[layer_idx],
            inplace_final_state=True,
            cu_seqlens=replay_cu,
            ssm_state_indices=repeated_indices,
            use_qk_l2norm_in_kernel=True,
        )

    grid = (
        triton.cdiv(HEAD_DIM, triton.next_power_of_2(HEAD_DIM)),
        triton.cdiv(HEAD_DIM, 32),
        NUM_LAYERS * NUM_REQUESTS * NUM_V_HEADS,
    )
    _journal_commit_kernel[grid](
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        candidate,
        state_indices,
        accepted,
        HEAD_DIM**-0.5,
        state_layer_stride=candidate.stride(0),
        state_block_stride=candidate.stride(1),
        journal_layer_k_stride=k.stride(0),
        journal_layer_v_stride=v.stride(0),
        journal_layer_gate_stride=a.stride(0),
        N=NUM_REQUESTS,
        H=NUM_K_HEADS,
        HV=NUM_V_HEADS,
        K=HEAD_DIM,
        V=HEAD_DIM,
        T=QUERY_LEN,
        BK=triton.next_power_of_2(HEAD_DIM),
        BV=32,
        num_warps=4,
        num_stages=3,
    )

    selected_reference = reference[:, state_indices_long]
    selected_candidate = candidate[:, state_indices_long]
    unmapped = torch.tensor(
        [idx for idx in range(POOL_BLOCKS) if idx not in COMMITTED_BLOCK_IDS],
        device=device,
    )
    selected_error = _error(selected_candidate, selected_reference)
    per_layer_request = {
        f"layer{layer_idx}.accept{req_idx + 1}": _error(
            selected_candidate[layer_idx, req_idx],
            selected_reference[layer_idx, req_idx],
        )
        for layer_idx in range(NUM_LAYERS)
        for req_idx in range(NUM_REQUESTS)
    }
    passed = bool(
        torch.equal(selected_candidate, selected_reference)
        and torch.equal(candidate[:, unmapped], candidate_before[:, unmapped])
    )
    result = {
        "passed": passed,
        "shape": {
            "layers": NUM_LAYERS,
            "requests": NUM_REQUESTS,
            "query_len": QUERY_LEN,
            "accepted": accepted.tolist(),
            "committed_block_ids": list(COMMITTED_BLOCK_IDS),
            "state_dtype": str(state_dtype),
            "journal_dtype": str(dtype),
        },
        "selected_state": selected_error,
        "per_layer_request": per_layer_request,
        "bit_exact": torch.equal(selected_candidate, selected_reference),
        "unmapped_blocks_untouched": torch.equal(
            candidate[:, unmapped], candidate_before[:, unmapped]
        ),
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
