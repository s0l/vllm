"""Prove the state contract for journal-based K3 GDN commit.

This is a research oracle, not production code.  It compares the current
four-snapshot recurrent path with replaying only the accepted prefix into one
committed state, and checks direct convolution-history commit for acceptance
lengths 1..4.  The intentionally off-by-one controls must differ.
"""

from __future__ import annotations

import json

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)


NUM_REQUESTS = 4
QUERY_LEN = 4
NUM_K_HEADS = 6
NUM_V_HEADS = 18
HEAD_DIM = 128
CONV_DIM = 3840
CONV_KERNEL_WIDTH = 4
CONV_HISTORY = CONV_KERNEL_WIDTH - 1
EXTENDED_CONV_STATE = CONV_HISTORY + QUERY_LEN - 1
COMMITTED_BLOCK_IDS = (7, 2, 11, 4)
POOL_BLOCKS = max(COMMITTED_BLOCK_IDS) + 1


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "differing": int(torch.count_nonzero(delta).item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def _pack_prefixes(
    tensor: torch.Tensor,
    accepted: torch.Tensor,
    *,
    has_batch_dim: bool,
) -> torch.Tensor:
    rows = []
    for req_idx, length in enumerate(accepted.tolist()):
        if has_batch_dim:
            start = req_idx * QUERY_LEN
            rows.append(tensor[:, start : start + length])
        else:
            start = req_idx * QUERY_LEN
            rows.append(tensor[start : start + length])
    return torch.cat(rows, dim=1 if has_batch_dim else 0)


@torch.inference_mode()
def recurrent_oracle(device: torch.device) -> dict[str, object]:
    dtype = torch.bfloat16
    state_dtype = torch.float32
    total_tokens = NUM_REQUESTS * QUERY_LEN
    accepted = torch.arange(1, QUERY_LEN + 1, device=device, dtype=torch.int32)
    full_cu = torch.arange(
        0,
        total_tokens + 1,
        QUERY_LEN,
        device=device,
        dtype=torch.int32,
    )

    q = torch.randn(
        1,
        total_tokens,
        NUM_K_HEADS,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1,
        total_tokens,
        NUM_V_HEADS,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(total_tokens, NUM_V_HEADS, device=device, dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(
        NUM_V_HEADS, device=device, dtype=torch.float32
    ).mul_(0.1)
    dt_bias = torch.randn_like(a_log).mul_(0.1)
    initial = torch.randn(
        NUM_REQUESTS,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=state_dtype,
    ).mul_(0.02)

    # Current topology: four physical state columns per request.  Column zero
    # initially owns the committed state and is overwritten with token one;
    # columns one through three receive the later speculative positions.
    current_pool = torch.zeros(
        1 + NUM_REQUESTS * QUERY_LEN,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=state_dtype,
    )
    current_indices = torch.arange(
        1,
        1 + NUM_REQUESTS * QUERY_LEN,
        device=device,
        dtype=torch.int32,
    ).view(NUM_REQUESTS, QUERY_LEN)
    for req_idx in range(NUM_REQUESTS):
        current_pool[current_indices[req_idx, 0]].copy_(initial[req_idx])

    current_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=current_pool,
        inplace_final_state=True,
        cu_seqlens=full_cu,
        ssm_state_indices=current_indices,
        num_accepted_tokens=torch.ones(
            NUM_REQUESTS, device=device, dtype=torch.int32
        ),
        use_qk_l2norm_in_kernel=True,
    )

    expected = torch.stack(
        [
            current_pool[current_indices[req_idx, accepted[req_idx] - 1]]
            for req_idx in range(NUM_REQUESTS)
        ]
    )

    # Proxy for the proposed verifier mode: identical recurrence and outputs,
    # but the input state is not overwritten.  The current API still allocates
    # full output states; production journal mode will suppress those stores.
    committed_indices = torch.tensor(
        COMMITTED_BLOCK_IDS, device=device, dtype=torch.int32
    )
    committed_indices_long = committed_indices.to(torch.long)
    verifier_pool = torch.randn(
        POOL_BLOCKS,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=state_dtype,
    ).mul_(0.02)
    verifier_pool.index_copy_(0, committed_indices_long, initial)
    verifier_indices = (
        committed_indices.view(NUM_REQUESTS, 1)
        .expand(-1, QUERY_LEN)
    )
    verifier_before = verifier_pool.clone()
    verifier_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=verifier_pool,
        inplace_final_state=False,
        cu_seqlens=full_cu,
        ssm_state_indices=verifier_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Journal commit: replay only each request's accepted prefix into its one
    # committed state.  Repeated indices store intermediate states to the same
    # slot; the register-resident recurrence does not reload them.
    replay_q = _pack_prefixes(q, accepted, has_batch_dim=True)
    replay_k = _pack_prefixes(k, accepted, has_batch_dim=True)
    replay_v = _pack_prefixes(v, accepted, has_batch_dim=True)
    replay_a = _pack_prefixes(a, accepted, has_batch_dim=False)
    replay_b = _pack_prefixes(b, accepted, has_batch_dim=False)
    replay_cu = torch.zeros(
        NUM_REQUESTS + 1, device=device, dtype=torch.int32
    )
    torch.cumsum(accepted, dim=0, out=replay_cu[1:])
    replay_indices = verifier_indices.clone()
    replay_pool = verifier_before.clone()
    fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=replay_a,
        b=replay_b,
        dt_bias=dt_bias,
        q=replay_q,
        k=replay_k,
        v=replay_v,
        initial_state=replay_pool,
        inplace_final_state=True,
        cu_seqlens=replay_cu,
        ssm_state_indices=replay_indices,
        use_qk_l2norm_in_kernel=True,
    )
    actual = replay_pool[committed_indices_long]

    # No-compaction form intended for integration.  Each request becomes an
    # accepted-prefix sequence followed by a rejected-tail sequence backed by
    # NULL_BLOCK_ID.  The tail advances cu_seqlens to the next fixed-stride
    # request boundary but returns before touching state.
    paired_cu = torch.empty(
        NUM_REQUESTS * 2 + 1, device=device, dtype=torch.int32
    )
    paired_cu[0::2] = torch.arange(
        0,
        NUM_REQUESTS * QUERY_LEN + 1,
        QUERY_LEN,
        device=device,
        dtype=torch.int32,
    )
    paired_cu[1::2] = (
        torch.arange(
            0,
            NUM_REQUESTS * QUERY_LEN,
            QUERY_LEN,
            device=device,
            dtype=torch.int32,
        )
        + accepted
    )
    paired_indices = torch.zeros(
        NUM_REQUESTS * 2,
        QUERY_LEN,
        device=device,
        dtype=torch.int32,
    )
    paired_indices[0::2] = verifier_indices
    fixed_stride_pool = verifier_before.clone()
    fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=torch.zeros_like(q),
        k=k,
        v=v,
        initial_state=fixed_stride_pool,
        inplace_final_state=True,
        cu_seqlens=paired_cu,
        ssm_state_indices=paired_indices,
        use_qk_l2norm_in_kernel=True,
    )
    fixed_stride_actual = fixed_stride_pool[committed_indices_long]

    wrong_accepted = accepted.roll(1)
    wrong_q = _pack_prefixes(q, wrong_accepted, has_batch_dim=True)
    wrong_k = _pack_prefixes(k, wrong_accepted, has_batch_dim=True)
    wrong_v = _pack_prefixes(v, wrong_accepted, has_batch_dim=True)
    wrong_a = _pack_prefixes(a, wrong_accepted, has_batch_dim=False)
    wrong_b = _pack_prefixes(b, wrong_accepted, has_batch_dim=False)
    wrong_cu = torch.zeros_like(replay_cu)
    torch.cumsum(wrong_accepted, dim=0, out=wrong_cu[1:])
    wrong_pool = verifier_before.clone()
    fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=wrong_a,
        b=wrong_b,
        dt_bias=dt_bias,
        q=wrong_q,
        k=wrong_k,
        v=wrong_v,
        initial_state=wrong_pool,
        inplace_final_state=True,
        cu_seqlens=wrong_cu,
        ssm_state_indices=replay_indices,
        use_qk_l2norm_in_kernel=True,
    )

    return {
        "verifier_output": _error(verifier_out, current_out),
        "verifier_input_state_unchanged": torch.equal(
            verifier_pool, verifier_before
        ),
        "accepted_commit": _error(actual, expected),
        "accepted_commit_bit_exact": torch.equal(actual, expected),
        "fixed_stride_commit": _error(fixed_stride_actual, expected),
        "fixed_stride_commit_bit_exact": torch.equal(
            fixed_stride_actual, expected
        ),
        "negative_wrong_acceptance": _error(
            wrong_pool[committed_indices_long], expected
        ),
        "negative_control_separates": not torch.equal(
            wrong_pool[committed_indices_long], expected
        ),
        "unmapped_blocks_untouched": torch.equal(
            replay_pool[
                torch.tensor(
                    [
                        idx
                        for idx in range(POOL_BLOCKS)
                        if idx not in COMMITTED_BLOCK_IDS
                    ],
                    device=device,
                )
            ],
            verifier_before[
                torch.tensor(
                    [
                        idx
                        for idx in range(POOL_BLOCKS)
                        if idx not in COMMITTED_BLOCK_IDS
                    ],
                    device=device,
                )
            ],
        ),
    }


@torch.inference_mode()
def convolution_oracle(device: torch.device) -> dict[str, object]:
    dtype = torch.bfloat16
    accepted = torch.arange(1, QUERY_LEN + 1, device=device, dtype=torch.int32)
    x = torch.randn(
        NUM_REQUESTS * QUERY_LEN, CONV_DIM, device=device, dtype=dtype
    )
    weights = torch.randn(
        CONV_DIM, CONV_KERNEL_WIDTH, device=device, dtype=dtype
    ).mul_(0.02)
    initial_history = torch.randn(
        NUM_REQUESTS, CONV_DIM, CONV_HISTORY, device=device, dtype=dtype
    )
    # Physical block zero is vLLM's NULL_BLOCK_ID and must never back a live
    # request.  Keep it as a sentinel exactly as the runtime allocator does.
    extended = torch.randn(
        POOL_BLOCKS,
        CONV_DIM,
        EXTENDED_CONV_STATE,
        device=device,
        dtype=dtype,
    )
    state_indices = torch.tensor(
        COMMITTED_BLOCK_IDS, device=device, dtype=torch.int32
    )
    state_indices_long = state_indices.to(torch.long)
    committed_conv = extended.index_select(0, state_indices_long)
    committed_conv[:, :, :CONV_HISTORY].copy_(initial_history)
    extended.index_copy_(0, state_indices_long, committed_conv)
    extended_before = extended.clone()
    query_start = torch.arange(
        0,
        NUM_REQUESTS * QUERY_LEN + 1,
        QUERY_LEN,
        device=device,
        dtype=torch.int32,
    )

    # The operation aliases/overwrites x, so retain the exact raw journal.
    raw_x = x.clone()
    causal_conv1d_update(
        x,
        extended,
        weights,
        None,
        "silu",
        conv_state_indices=state_indices,
        num_accepted_tokens=torch.ones(
            NUM_REQUESTS, device=device, dtype=torch.int32
        ),
        query_start_loc=query_start,
        max_query_len=QUERY_LEN,
        validate_data=False,
    )

    expected = []
    direct = []
    wrong = []
    for req_idx, length in enumerate(accepted.tolist()):
        bias = length - 1
        expected.append(
            extended[
                int(state_indices[req_idx].item()),
                :,
                bias : bias + CONV_HISTORY,
            ]
        )
        start = req_idx * QUERY_LEN
        accepted_inputs = raw_x[start : start + length].transpose(0, 1)
        direct.append(
            torch.cat((initial_history[req_idx], accepted_inputs), dim=1)[
                :, -CONV_HISTORY:
            ]
        )
        wrong_length = int(accepted.roll(1)[req_idx].item())
        wrong_inputs = raw_x[start : start + wrong_length].transpose(0, 1)
        wrong.append(
            torch.cat((initial_history[req_idx], wrong_inputs), dim=1)[
                :, -CONV_HISTORY:
            ]
        )
    expected_t = torch.stack(expected)
    direct_t = torch.stack(direct)
    wrong_t = torch.stack(wrong)
    unmapped = torch.tensor(
        [
            idx
            for idx in range(POOL_BLOCKS)
            if idx not in COMMITTED_BLOCK_IDS
        ],
        device=device,
    )

    return {
        "accepted_commit": _error(direct_t, expected_t),
        "accepted_commit_bit_exact": torch.equal(direct_t, expected_t),
        "negative_wrong_acceptance": _error(wrong_t, expected_t),
        "negative_control_separates": not torch.equal(wrong_t, expected_t),
        "unmapped_blocks_untouched": torch.equal(
            extended[unmapped], extended_before[unmapped]
        ),
    }


def main() -> None:
    torch.manual_seed(212)
    device = torch.device("cuda")
    recurrent = recurrent_oracle(device)
    convolution = convolution_oracle(device)
    passed = bool(
        recurrent["verifier_input_state_unchanged"]
        and recurrent["accepted_commit_bit_exact"]
        and recurrent["fixed_stride_commit_bit_exact"]
        and recurrent["negative_control_separates"]
        and recurrent["unmapped_blocks_untouched"]
        and recurrent["verifier_output"]["differing"] == 0
        and convolution["accepted_commit_bit_exact"]
        and convolution["negative_control_separates"]
        and convolution["unmapped_blocks_untouched"]
    )
    result = {
        "passed": passed,
        "shape": {
            "requests": NUM_REQUESTS,
            "query_len": QUERY_LEN,
            "accepted": [1, 2, 3, 4],
            "committed_block_ids": list(COMMITTED_BLOCK_IDS),
            "k_heads": NUM_K_HEADS,
            "v_heads": NUM_V_HEADS,
            "head_dim": HEAD_DIM,
            "conv_dim": CONV_DIM,
            "state_dtype": "float32",
            "activation_dtype": "bfloat16",
        },
        "recurrent": recurrent,
        "convolution": convolution,
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
