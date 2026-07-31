# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensorized thinking-budget state for Model Runner V2.

The target model is authoritative during speculative verification. Draft tokens
are used only to derive virtual state for each target-logit row; persistent
state advances exclusively from the prefix committed by rejection sampling.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig
    from vllm.sampling_params import SamplingParams


_ENABLED = 0
_BUDGET = 1
_REMAINING = 2
_IN_THINK = 3
_START_MATCH = 4
_END_MATCH = 5
_NUM_STATE_FIELDS = 6


def _prefix_function(pattern: list[int]) -> list[int]:
    failure = [0] * len(pattern)
    matched = 0
    for i in range(1, len(pattern)):
        while matched and pattern[i] != pattern[matched]:
            matched = failure[matched - 1]
        if pattern[i] == pattern[matched]:
            matched += 1
        failure[i] = matched
    return failure


def _advance_match(
    token_id: int,
    pattern: list[int],
    failure: list[int],
    matched: int,
) -> tuple[int, bool]:
    while matched and token_id != pattern[matched]:
        matched = failure[matched - 1]
    if token_id == pattern[matched]:
        matched += 1
    complete = matched == len(pattern)
    if complete:
        matched = failure[matched - 1]
    return matched, complete


@dataclass
class ReferenceThinkingState:
    """Small CPU oracle used for initialization and deterministic controls."""

    budget: int
    remaining: int
    in_think: bool = False
    start_match: int = 0
    end_match: int = 0


def advance_reference_state(
    state: ReferenceThinkingState,
    token_id: int,
    start_tokens: list[int],
    end_tokens: list[int],
) -> None:
    """Advance the public per-reasoning-block budget contract by one token."""
    _advance_reference_state_with_failure(
        state,
        token_id,
        start_tokens,
        _prefix_function(start_tokens),
        end_tokens,
        _prefix_function(end_tokens),
    )


def _advance_reference_state_with_failure(
    state: ReferenceThinkingState,
    token_id: int,
    start_tokens: list[int],
    start_failure: list[int],
    end_tokens: list[int],
    end_failure: list[int],
) -> None:
    if state.in_think:
        state.end_match, ended = _advance_match(
            token_id,
            end_tokens,
            end_failure,
            state.end_match,
        )
        if ended:
            state.in_think = False
            state.remaining = state.budget
            state.start_match = 0
            state.end_match = 0
        else:
            state.remaining = max(0, state.remaining - 1)
        return

    state.start_match, started = _advance_match(
        token_id,
        start_tokens,
        start_failure,
        state.start_match,
    )
    if started:
        state.in_think = True
        state.remaining = state.budget
        state.start_match = 0
        state.end_match = 0


def initialize_reference_state(
    token_ids: list[int],
    budget: int,
    start_tokens: list[int],
    end_tokens: list[int],
) -> ReferenceThinkingState:
    state = ReferenceThinkingState(budget=budget, remaining=budget)
    start_failure = _prefix_function(start_tokens)
    end_failure = _prefix_function(end_tokens)
    for token_id in token_ids:
        _advance_reference_state_with_failure(
            state,
            token_id,
            start_tokens,
            start_failure,
            end_tokens,
            end_failure,
        )
    return state


def reference_forced_tokens(
    state: ReferenceThinkingState,
    draft_tokens: list[int],
    start_tokens: list[int],
    end_tokens: list[int],
) -> list[int | None]:
    """Return target forcing for draft positions plus the bonus position."""
    virtual = ReferenceThinkingState(**vars(state))
    forced: list[int | None] = []
    for local_pos in range(len(draft_tokens) + 1):
        force_token = (
            end_tokens[virtual.end_match]
            if virtual.in_think and virtual.remaining <= 0
            else None
        )
        forced.append(force_token)
        if local_pos < len(draft_tokens):
            advance_reference_state(
                virtual, draft_tokens[local_pos], start_tokens, end_tokens
            )
    return forced


@triton.jit
def _advance_kmp(
    token_id,
    pattern_ptr,
    failure_ptr,
    matched,
    PATTERN_LEN: tl.constexpr,
):
    for _ in range(PATTERN_LEN):
        expected = tl.load(pattern_ptr + matched)
        mismatch = (matched > 0) & (token_id != expected)
        fallback_idx = tl.maximum(matched - 1, 0)
        fallback = tl.load(failure_ptr + fallback_idx)
        matched = tl.where(mismatch, fallback, matched)
    expected = tl.load(pattern_ptr + matched)
    matched += (token_id == expected).to(tl.int32)
    complete = matched == PATTERN_LEN
    fallback_idx = tl.maximum(matched - 1, 0)
    fallback = tl.load(failure_ptr + fallback_idx)
    matched = tl.where(complete, fallback, matched)
    return matched, complete


@triton.jit
def _advance_state(
    token_id,
    budget,
    remaining,
    in_think,
    start_match,
    end_match,
    start_tokens_ptr,
    start_failure_ptr,
    end_tokens_ptr,
    end_failure_ptr,
    START_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
):
    if in_think:
        end_match, ended = _advance_kmp(
            token_id,
            end_tokens_ptr,
            end_failure_ptr,
            end_match,
            PATTERN_LEN=END_LEN,
        )
        if ended:
            in_think = False
            remaining = budget
            start_match = 0
            end_match = 0
        else:
            remaining = tl.maximum(remaining - 1, 0)
    else:
        start_match, started = _advance_kmp(
            token_id,
            start_tokens_ptr,
            start_failure_ptr,
            start_match,
            PATTERN_LEN=START_LEN,
        )
        if started:
            in_think = True
            remaining = budget
            start_match = 0
            end_match = 0
    return remaining, in_think, start_match, end_match


@triton.jit
def _prepare_forced_tokens_kernel(
    forced_tokens_ptr,
    state_ptr,
    state_stride,
    draft_sampled_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    start_tokens_ptr,
    start_failure_ptr,
    end_tokens_ptr,
    end_failure_ptr,
    num_logits,
    START_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
):
    logit_idx = tl.program_id(0)
    if logit_idx >= num_logits:
        return

    req_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    local_pos = tl.load(expanded_local_pos_ptr + logit_idx)
    state_row = state_ptr + req_idx * state_stride
    enabled = tl.load(state_row + 0)
    budget = tl.load(state_row + 1)
    remaining = tl.load(state_row + 2)
    in_think = tl.load(state_row + 3) != 0
    start_match = tl.load(state_row + 4)
    end_match = tl.load(state_row + 5)

    request_start = logit_idx - local_pos
    for i in range(local_pos):
        token_id = tl.load(draft_sampled_ptr + request_start + i + 1)
        remaining, in_think, start_match, end_match = _advance_state(
            token_id,
            budget,
            remaining,
            in_think,
            start_match,
            end_match,
            start_tokens_ptr,
            start_failure_ptr,
            end_tokens_ptr,
            end_failure_ptr,
            START_LEN=START_LEN,
            END_LEN=END_LEN,
        )

    force_token = tl.full((), -1, tl.int64)
    if (enabled != 0) & in_think & (remaining <= 0):
        force_token = tl.load(end_tokens_ptr + end_match)
    tl.store(forced_tokens_ptr + logit_idx, force_token)


@triton.jit
def _apply_forced_tokens_kernel(
    logits_ptr,
    logits_stride,
    forced_tokens_ptr,
    num_logits,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    logit_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    if logit_idx >= num_logits:
        return
    force_token = tl.load(forced_tokens_ptr + logit_idx)
    if force_token < 0:
        return
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size
    values = tl.where(offsets == force_token, 0.0, float("-inf"))
    tl.store(logits_ptr + logit_idx * logits_stride + offsets, values, mask=mask)


@triton.jit
def _update_committed_state_kernel(
    state_ptr,
    state_stride,
    sampled_tokens_ptr,
    sampled_stride,
    num_sampled_ptr,
    idx_mapping_ptr,
    start_tokens_ptr,
    start_failure_ptr,
    end_tokens_ptr,
    end_failure_ptr,
    num_reqs,
    MAX_COMMITTED: tl.constexpr,
    START_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
):
    req_row = tl.program_id(0)
    if req_row >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + req_row)
    if req_idx < 0:
        return
    state_row = state_ptr + req_idx * state_stride
    enabled = tl.load(state_row + 0)
    if enabled == 0:
        return

    budget = tl.load(state_row + 1)
    remaining = tl.load(state_row + 2)
    in_think = tl.load(state_row + 3) != 0
    start_match = tl.load(state_row + 4)
    end_match = tl.load(state_row + 5)
    committed = tl.load(num_sampled_ptr + req_row)

    for i in range(MAX_COMMITTED):
        if i < committed:
            token_id = tl.load(sampled_tokens_ptr + req_row * sampled_stride + i)
            remaining, in_think, start_match, end_match = _advance_state(
                token_id,
                budget,
                remaining,
                in_think,
                start_match,
                end_match,
                start_tokens_ptr,
                start_failure_ptr,
                end_tokens_ptr,
                end_failure_ptr,
                START_LEN=START_LEN,
                END_LEN=END_LEN,
            )

    tl.store(state_row + 2, remaining)
    tl.store(state_row + 3, in_think.to(tl.int32))
    tl.store(state_row + 4, start_match)
    tl.store(state_row + 5, end_match)


class ThinkingBudgetState:
    """Persistent, slot-indexed Model Runner V2 thinking-budget state."""

    def __init__(
        self,
        reasoning_config: "ReasoningConfig | None",
        max_num_reqs: int,
        max_num_logits: int,
        max_committed_tokens: int,
        device: torch.device,
    ) -> None:
        self._active_slots: set[int] = set()
        self._pending_rows: dict[int, list[int]] = {}
        self.max_committed_tokens = max_committed_tokens
        self.state = StagedWriteTensor(
            (max_num_reqs, _NUM_STATE_FIELDS),
            dtype=torch.int32,
            device=device,
        )
        self.forced_tokens = torch.full(
            (max_num_logits,), -1, dtype=torch.int64, device=device
        )

        start_tokens = (
            list(reasoning_config.reasoning_start_token_ids)
            if reasoning_config is not None
            and reasoning_config.reasoning_start_token_ids
            else []
        )
        end_tokens = (
            list(reasoning_config.reasoning_end_token_ids)
            if reasoning_config is not None and reasoning_config.reasoning_end_token_ids
            else []
        )
        self.start_tokens_cpu = start_tokens
        self.end_tokens_cpu = end_tokens
        self.is_configured = bool(start_tokens and end_tokens)

        # Non-empty placeholders keep construction safe when reasoning is disabled;
        # kernels are never launched in that state.
        start_storage = start_tokens or [0]
        end_storage = end_tokens or [0]
        self.start_tokens = torch.tensor(
            start_storage, dtype=torch.int64, device=device
        )
        self.end_tokens = torch.tensor(end_storage, dtype=torch.int64, device=device)
        self.start_failure = torch.tensor(
            _prefix_function(start_storage), dtype=torch.int32, device=device
        )
        self.end_failure = torch.tensor(
            _prefix_function(end_storage), dtype=torch.int32, device=device
        )

    @property
    def has_requests(self) -> bool:
        return bool(self._active_slots)

    def add_request(
        self,
        req_idx: int,
        sampling_params: "SamplingParams",
        token_ids: list[int],
    ) -> None:
        budget = sampling_params.thinking_token_budget
        if budget is None:
            self.remove_request(req_idx)
            return
        if not self.is_configured:
            raise ValueError(
                "thinking_token_budget requires non-empty reasoning start and "
                "end token sequences"
            )
        initial = initialize_reference_state(
            token_ids,
            budget,
            self.start_tokens_cpu,
            self.end_tokens_cpu,
        )
        self._active_slots.add(req_idx)
        self._pending_rows[req_idx] = [
            1,
            budget,
            initial.remaining,
            int(initial.in_think),
            initial.start_match,
            initial.end_match,
        ]

    def remove_request(self, req_idx: int) -> None:
        if req_idx in self._active_slots:
            self._active_slots.remove(req_idx)
            self._pending_rows[req_idx] = [0, 0, 0, 0, 0, 0]

    def apply_staged_writes(self) -> None:
        for req_idx, row in self._pending_rows.items():
            self.state.stage_write(req_idx, 0, row)
        self._pending_rows.clear()
        self.state.apply_write()

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        draft_sampled: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self._active_slots:
            return
        num_logits, vocab_size = logits.shape
        _prepare_forced_tokens_kernel[(num_logits,)](
            self.forced_tokens,
            self.state.gpu,
            self.state.gpu.stride(0),
            draft_sampled,
            expanded_idx_mapping,
            expanded_local_pos,
            self.start_tokens,
            self.start_failure,
            self.end_tokens,
            self.end_failure,
            num_logits,
            START_LEN=len(self.start_tokens_cpu),
            END_LEN=len(self.end_tokens_cpu),
            num_warps=1,
        )
        block_size = 256
        _apply_forced_tokens_kernel[
            (num_logits, triton.cdiv(vocab_size, block_size))
        ](
            logits,
            logits.stride(0),
            self.forced_tokens,
            num_logits,
            vocab_size,
            BLOCK_SIZE=block_size,
        )

    def update_committed(
        self,
        idx_mapping: torch.Tensor,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        if not self._active_slots:
            return
        num_reqs = idx_mapping.shape[0]
        _update_committed_state_kernel[(num_reqs,)](
            self.state.gpu,
            self.state.gpu.stride(0),
            sampled_tokens,
            sampled_tokens.stride(0),
            num_sampled,
            idx_mapping,
            self.start_tokens,
            self.start_failure,
            self.end_tokens,
            self.end_failure,
            num_reqs,
            MAX_COMMITTED=self.max_committed_tokens,
            START_LEN=len(self.start_tokens_cpu),
            END_LEN=len(self.end_tokens_cpu),
            num_warps=1,
        )
