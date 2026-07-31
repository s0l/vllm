# SPDX-License-Identifier: Apache-2.0
"""Deterministic oracle controls for MRV2 thinking-budget semantics."""

from copy import deepcopy
from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.sample.thinking_budget import (
    ReferenceThinkingState,
    advance_reference_state,
    initialize_reference_state,
    reference_forced_tokens,
)

START = [100]
END = [200, 201, 202]
THINK = 60
CONTENT = 50


def _advance(state: ReferenceThinkingState, tokens: list[int]) -> None:
    for token in tokens:
        advance_reference_state(state, token, START, END)


def test_budget_zero_forces_first_end_token_after_reasoning_start():
    state = initialize_reference_state(START, 0, START, END)
    assert reference_forced_tokens(state, [], START, END) == [END[0]]


def test_exact_budget_allows_n_tokens_then_forces_complete_multi_token_end():
    state = initialize_reference_state(START, 3, START, END)
    _advance(state, [THINK] * 3)
    assert reference_forced_tokens(state, [], START, END) == [END[0]]

    _advance(state, [END[0]])
    assert reference_forced_tokens(state, [], START, END) == [END[1]]
    _advance(state, [END[1]])
    assert reference_forced_tokens(state, [], START, END) == [END[2]]
    _advance(state, [END[2]])
    assert not state.in_think
    assert state.remaining == 3


def test_natural_end_before_budget_and_reentry_gets_fresh_block_budget():
    state = initialize_reference_state(START, 4, START, END)
    _advance(state, [THINK, *END, CONTENT, *START, THINK, THINK, THINK, THINK])
    assert state.in_think
    assert state.remaining == 0
    assert reference_forced_tokens(state, [], START, END) == [END[0]]


def test_prompt_already_inside_reasoning_is_initialized_from_full_history():
    state = initialize_reference_state(
        [CONTENT, *START, THINK, THINK],
        2,
        START,
        END,
    )
    assert state.in_think
    assert state.remaining == 0
    assert reference_forced_tokens(state, [], START, END) == [END[0]]


def test_k2_virtual_rows_force_at_exact_boundary_and_bonus():
    state = initialize_reference_state(START, 2, START, END)
    assert reference_forced_tokens(
        state, [THINK, THINK], START, END
    ) == [None, None, END[0]]

    state = initialize_reference_state([*START, THINK], 2, START, END)
    assert reference_forced_tokens(
        state, [THINK, END[0]], START, END
    ) == [None, END[0], END[1]]


def test_rejected_draft_suffix_has_no_persistent_effect():
    initial = initialize_reference_state([*START, THINK], 2, START, END)
    virtual = reference_forced_tokens(initial, [THINK, CONTENT], START, END)
    # Row 1 forces END[0]. The incompatible CONTENT proposal means row 2 is
    # unreachable after rejection; if inspected, its virtual path still starts
    # at END[0] because proposals are context, not forced target outputs.
    assert virtual == [None, END[0], END[0]]

    committed = deepcopy(initial)
    # The first proposal is accepted, the incompatible second proposal and its
    # suffix are rejected, and the target resamples the first end token.
    _advance(committed, [THINK, END[0]])
    assert committed.end_match == 1
    assert reference_forced_tokens(committed, [], START, END) == [END[1]]


def test_negative_control_detects_off_by_one_budget_enforcement():
    state = initialize_reference_state(START, 2, START, END)
    _advance(state, [THINK, THINK])
    correct = reference_forced_tokens(state, [], START, END)
    intentionally_broken = [None] if state.remaining == 0 else [END[0]]
    assert correct != intentionally_broken


@torch.inference_mode()
def test_gpu_k2_forcing_commit_rollback_and_plain_identity():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is required for the MRV2 Triton control")

    from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState

    device = torch.device("cuda")
    config = SimpleNamespace(
        reasoning_start_token_ids=START,
        reasoning_end_token_ids=END,
    )
    state = ThinkingBudgetState(
        reasoning_config=config,
        max_num_reqs=4,
        max_num_logits=12,
        max_committed_tokens=3,
        device=device,
    )
    budgeted = SimpleNamespace(thinking_token_budget=2)
    plain = SimpleNamespace(thinking_token_budget=None)
    state.add_request(0, budgeted, [*START, THINK])
    state.add_request(1, plain, [CONTENT])
    state.apply_staged_writes()

    vocab_size = 256
    logits = torch.randn(6, vocab_size, device=device)
    plain_before = logits[3:].clone()
    # Per-request flattened layout is [last committed input, draft0, draft1].
    draft_sampled = torch.tensor(
        [THINK, THINK, CONTENT, CONTENT, CONTENT, CONTENT],
        dtype=torch.int64,
        device=device,
    )
    expanded_idx_mapping = torch.tensor(
        [0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
    )
    expanded_local_pos = torch.tensor(
        [0, 1, 2, 0, 1, 2], dtype=torch.int32, device=device
    )
    state.apply_to_logits(
        logits,
        draft_sampled,
        expanded_idx_mapping,
        expanded_local_pos,
    )
    assert state.forced_tokens[:6].cpu().tolist() == [-1, END[0], END[0], -1, -1, -1]
    assert torch.equal(logits[3:], plain_before)
    assert logits[1, END[0]].item() == 0
    assert torch.isneginf(logits[1, :END[0]]).all()
    assert torch.isneginf(logits[1, END[0] + 1 :]).all()

    # Only accepted draft0 plus target-resampled END[0] is committed. The
    # incompatible draft suffix never reaches persistent state.
    sampled = torch.tensor(
        [[THINK, END[0], -1], [CONTENT, -1, -1]],
        dtype=torch.int64,
        device=device,
    )
    num_sampled = torch.tensor([2, 1], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    state.update_committed(idx_mapping, sampled, num_sampled)

    next_logits = torch.randn(1, vocab_size, device=device)
    state.apply_to_logits(
        next_logits,
        torch.tensor([END[0]], dtype=torch.int64, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
    )
    assert state.forced_tokens[0].item() == END[1]

    # A remove+reuse before staged writes must coalesce to the final row, not
    # race two writes to the same persistent slot.
    state.remove_request(0)
    state.add_request(0, SimpleNamespace(thinking_token_budget=5), [CONTENT])
    state.apply_staged_writes()
    reuse_logits = torch.randn(1, vocab_size, device=device)
    reuse_before = reuse_logits.clone()
    state.apply_to_logits(
        reuse_logits,
        torch.tensor([CONTENT], dtype=torch.int64, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
    )
    assert state.forced_tokens[0].item() == -1
    assert torch.equal(reuse_logits, reuse_before)

    # Completely unbudgeted batches take the early return and are bit-identical.
    state.remove_request(0)
    state.apply_staged_writes()
    assert not state.has_requests
    plain_logits = torch.randn(3, vocab_size, device=device)
    expected = plain_logits.clone()
    state.apply_to_logits(
        plain_logits,
        draft_sampled[:3],
        expanded_idx_mapping[:3],
        expanded_local_pos[:3],
    )
    assert torch.equal(plain_logits, expected)
