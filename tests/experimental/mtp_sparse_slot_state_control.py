#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic GPU controls for sparse MRV2 request-slot state.

This control deliberately separates batch rows from persistent request slots.
It covers three mutable state families implicated by a long-lived K2
request while neighbouring requests finish and reuse slots:

* committed-token accounting used by sampling penalties;
* mixed budget/plain thinking state;
* separate-pool GDN speculative-state commit.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from vllm import _C_stable_libtorch  # noqa: F401  # Register native runtime ops.
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import post_update
from vllm.v1.worker.gpu.sample.penalties import apply_penalties
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

MAX_REQUESTS = 44
VOCAB_SIZE = 257
SLOTS = (31, 2, 40)
START = 100
END = 200


def _reference_penalties(
    logits: torch.Tensor,
    *,
    slot: int,
    local_pos: int,
    request_start: int,
    input_ids: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_seen: torch.Tensor,
    output_counts: torch.Tensor,
) -> torch.Tensor:
    counts = output_counts[slot].clone()
    for prev_pos in range(local_pos):
        token = int(input_ids[request_start + prev_pos + 1])
        counts[token] += 1
    seen = prompt_seen[slot] | (counts > 0)
    rep = repetition_penalty[slot]
    scale = torch.where(seen, rep, 1.0)
    result = logits.float()
    result *= torch.where(result > 0, 1.0 / scale, scale)
    result -= frequency_penalty[slot] * counts
    result -= presence_penalty[slot] * (counts > 0)
    return result


@torch.inference_mode()
def penalties_and_commit_control(device: torch.device) -> dict[str, bool]:
    expanded_idx_mapping = torch.tensor(
        [SLOTS[0]] * 3 + [SLOTS[1]] * 3 + [SLOTS[2]] * 3,
        dtype=torch.int32,
        device=device,
    )
    expanded_local_pos = torch.tensor(
        [0, 1, 2] * 3,
        dtype=torch.int32,
        device=device,
    )
    input_ids = torch.tensor(
        [9, 11, 13, 17, 19, 23, 29, 31, 37],
        dtype=torch.int64,
        device=device,
    )
    base = torch.linspace(
        -4.0,
        4.0,
        steps=VOCAB_SIZE,
        dtype=torch.float32,
        device=device,
    ).repeat(9, 1)
    logits = base.clone()

    repetition = torch.ones(MAX_REQUESTS, dtype=torch.float32, device=device)
    frequency = torch.zeros_like(repetition)
    presence = torch.zeros_like(repetition)
    repetition[SLOTS[0]] = 1.05
    frequency[SLOTS[0]] = 0.2
    presence[SLOTS[0]] = 0.3
    repetition[SLOTS[1]] = 1.11
    frequency[SLOTS[1]] = 0.1

    prompt_seen = torch.zeros(MAX_REQUESTS, VOCAB_SIZE, dtype=torch.bool, device=device)
    prompt_seen[SLOTS[0], 7] = True
    prompt_seen[SLOTS[1], 19] = True
    prompt_bin_mask = torch.zeros(
        MAX_REQUESTS,
        (VOCAB_SIZE + 31) // 32,
        dtype=torch.int32,
        device=device,
    )
    for slot in SLOTS:
        for token in torch.nonzero(prompt_seen[slot], as_tuple=False).flatten():
            token_int = int(token)
            prompt_bin_mask[slot, token_int // 32] |= 1 << (token_int % 32)

    output_counts = torch.zeros(
        MAX_REQUESTS, VOCAB_SIZE, dtype=torch.int32, device=device
    )
    output_counts[SLOTS[0], 5] = 2
    output_counts[SLOTS[1], 17] = 3
    # Slot 40 is intentionally plain but contains stale-looking data. The
    # mixed kernel launch must leave all of its rows bit-identical.
    output_counts[SLOTS[2], 29] = 99

    expected = base.clone()
    for row in range(9):
        request_start = row - int(expanded_local_pos[row])
        expected[row] = _reference_penalties(
            base[row],
            slot=int(expanded_idx_mapping[row]),
            local_pos=int(expanded_local_pos[row]),
            request_start=request_start,
            input_ids=input_ids,
            repetition_penalty=repetition,
            frequency_penalty=frequency,
            presence_penalty=presence,
            prompt_seen=prompt_seen,
            output_counts=output_counts,
        )

    apply_penalties(
        logits,
        expanded_idx_mapping,
        input_ids,
        expanded_local_pos,
        repetition,
        frequency,
        presence,
        prompt_bin_mask,
        output_counts,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(logits, expected, rtol=0, atol=1e-6)
    assert torch.equal(logits[6:], base[6:])

    idx_mapping = torch.tensor(SLOTS, dtype=torch.int32, device=device)
    sampled = torch.tensor(
        [[41, -1, -1], [43, 47, -1], [53, 59, 61]],
        dtype=torch.int64,
        device=device,
    )
    num_sampled = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    num_rejected = torch.tensor([2, 1, 0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 6, 9], dtype=torch.int32, device=device)
    num_computed = torch.zeros(MAX_REQUESTS, dtype=torch.int32, device=device)
    last_sampled = torch.full((MAX_REQUESTS,), -1, dtype=torch.int64, device=device)
    all_token_ids = torch.full((MAX_REQUESTS, 16), -1, dtype=torch.int64, device=device)
    total_len = torch.zeros(MAX_REQUESTS, dtype=torch.int32, device=device)
    committed_counts = torch.zeros(
        MAX_REQUESTS, VOCAB_SIZE, dtype=torch.int32, device=device
    )
    untouched_slot = 17
    committed_counts[untouched_slot, 67] = 9

    post_update(
        idx_mapping,
        num_computed,
        last_sampled,
        committed_counts,
        sampled,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )
    torch.cuda.synchronize()

    for batch_row, slot in enumerate(SLOTS):
        count = int(num_sampled[batch_row])
        expected_tokens = sampled[batch_row, :count]
        assert torch.equal(all_token_ids[slot, :count], expected_tokens)
        assert int(total_len[slot]) == count
        assert int(last_sampled[slot]) == int(expected_tokens[-1])
        assert int(num_computed[slot]) == 3 - int(num_rejected[batch_row])
        for token in expected_tokens:
            assert int(committed_counts[slot, int(token)]) == 1
    assert int(committed_counts[untouched_slot, 67]) == 9
    assert int(total_len[untouched_slot]) == 0

    return {
        "penalties_sparse_mapping_exact": True,
        "plain_rows_bit_exact": True,
        "committed_counts_sparse_mapping_exact": True,
        "unmapped_slot_untouched": True,
    }


@torch.inference_mode()
def thinking_budget_control(device: torch.device) -> dict[str, bool]:
    req_states = RequestState(
        max_num_reqs=MAX_REQUESTS,
        max_model_len=64,
        max_num_batched_tokens=4,
        num_speculative_steps=2,
        vocab_size=VOCAB_SIZE,
        device=device,
    )
    histories = {
        SLOTS[0]: [START, 50, 51],
        SLOTS[2]: [70],
    }
    for slot, tokens in histories.items():
        req_states.all_token_ids.stage_write(slot, 0, tokens)
        req_states.total_len.stage_write_elem(slot, len(tokens))
    req_states.apply_staged_writes()

    state = ThinkingBudgetState(
        req_states,
        SimpleNamespace(
            reasoning_start_token_ids=[START],
            reasoning_end_token_ids=[END],
            natural_reasoning_end_token_ids=[END],
        ),
    )
    state.add_request(SLOTS[0], SamplingParams(thinking_token_budget=2))
    state.add_request(SLOTS[2], SamplingParams(thinking_token_budget=None))
    state.apply_staged_writes()

    idx_mapping = torch.tensor([SLOTS[0], SLOTS[2]], dtype=torch.int32, device=device)
    logits = torch.randn(2, VOCAB_SIZE, dtype=torch.float32, device=device)
    plain_before = logits[1].clone()
    state.apply(
        logits,
        idx_mapping,
        idx_mapping,
        np.array([SLOTS[0], SLOTS[2]], dtype=np.int32),
        torch.tensor([51, 70], dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize()
    assert logits[0, END].item() == 1.0e9
    assert torch.equal(logits[1], plain_before)

    # Reuse the sparse slot as a plain request. The newest staged state must
    # disable the old budget without scanning or mutating its logits.
    state.add_request(SLOTS[0], SamplingParams(thinking_token_budget=None))
    state.apply_staged_writes()
    assert not state.use_thinking_budget[SLOTS[0]]
    reuse = torch.randn(1, VOCAB_SIZE, dtype=torch.float32, device=device)
    reuse_before = reuse.clone()
    state.apply(
        reuse,
        torch.tensor([SLOTS[0]], dtype=torch.int32, device=device),
        torch.tensor([SLOTS[0]], dtype=torch.int32, device=device),
        np.array([SLOTS[0]], dtype=np.int32),
        torch.tensor([70], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    assert torch.equal(reuse, reuse_before)

    return {
        "mixed_sparse_plain_identity": True,
        "sparse_remove_reuse_identity": True,
    }


def _gdn_context(
    conv: torch.Tensor,
    temporal: torch.Tensor,
    block_table: torch.Tensor,
) -> MambaSpecDecodeGPUContext:
    device = conv.device
    return MambaSpecDecodeGPUContext(
        state_base_addrs=torch.tensor(
            [conv.data_ptr(), temporal.data_ptr()],
            dtype=torch.int64,
            device=device,
        ),
        state_block_strides=torch.tensor(
            [
                conv.stride(0) * conv.element_size(),
                temporal.stride(0) * temporal.element_size(),
            ],
            dtype=torch.int64,
            device=device,
        ),
        state_elem_sizes=torch.tensor(
            [conv.element_size(), temporal.element_size()],
            dtype=torch.int32,
            device=device,
        ),
        state_inner_sizes=torch.tensor(
            [conv.stride(1), temporal[0].numel()],
            dtype=torch.int64,
            device=device,
        ),
        state_conv_widths=torch.tensor([5, 0], dtype=torch.int32, device=device),
        state_group_indices=torch.zeros(2, dtype=torch.int32, device=device),
        state_dim_row_count=torch.zeros(2, dtype=torch.int32, device=device),
        state_dim_row_stride=torch.zeros(2, dtype=torch.int64, device=device),
        block_size=2352,
        num_states=2,
        mamba_group_ids=[0],
        num_groups=1,
        num_accepted_tokens_out=torch.zeros(
            MAX_REQUESTS, dtype=torch.int32, device=device
        ),
        block_table_ptrs=torch.tensor(
            [block_table.data_ptr()], dtype=torch.int64, device=device
        ),
        block_table_stride_req=block_table.stride(0),
        is_initialized=True,
    )


@torch.inference_mode()
def gdn_sparse_commit_control(device: torch.device) -> dict[str, bool]:
    idx_mapping = torch.tensor(SLOTS, dtype=torch.int32, device=device)
    num_reqs = len(SLOTS)
    num_blocks = 1 + num_reqs * 3
    block_table = (
        torch.arange(1, num_blocks, dtype=torch.int32, device=device)
        .view(num_reqs, 3)
        .contiguous()
    )
    torch.manual_seed(20260730)
    conv = torch.randn(num_blocks, 5, 3840, dtype=torch.bfloat16, device=device)
    temporal = torch.randn(num_blocks, 18, 128, 128, dtype=torch.float32, device=device)
    expected_conv = conv.clone()
    expected_temporal = temporal.clone()
    original_conv = conv.clone()
    original_temporal = temporal.clone()

    accepted = torch.ones(MAX_REQUESTS, dtype=torch.int32, device=device)
    accepted[SLOTS[0]] = 3
    accepted[SLOTS[1]] = 1
    accepted[SLOTS[2]] = 2
    for batch_row, slot in enumerate(SLOTS):
        bias = int(accepted[slot]) - 1
        if bias == 0:
            continue
        base_block = int(block_table[batch_row, 0])
        accepted_block = int(block_table[batch_row, bias])
        expected_conv[base_block, : 5 - bias] = original_conv[base_block, bias:]
        expected_temporal[base_block] = original_temporal[accepted_block]

    context = _gdn_context(conv, temporal, block_table)
    context.run_fused_postprocess_separate(num_reqs, accepted, idx_mapping)
    torch.cuda.synchronize()
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(temporal, expected_temporal, rtol=0, atol=0)
    assert torch.equal(
        accepted[list(SLOTS)],
        torch.ones(num_reqs, dtype=torch.int32, device=device),
    )
    assert int(accepted[17]) == 1

    return {
        "gdn_sparse_commit_bit_exact": True,
        "gdn_sparse_acceptance_reset_exact": True,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    result = {
        "penalties_and_commit": penalties_and_commit_control(device),
        "thinking_budget": thinking_budget_control(device),
        "gdn": gdn_sparse_commit_control(device),
    }
    print(json.dumps({"result": "pass", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
