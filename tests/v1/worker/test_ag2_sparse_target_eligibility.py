# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState


class _ArrayState:
    def __init__(self, values: list[float] | list[int]):
        self.np = np.asarray(values)


def _make_sampler(*, active_budget_slots: tuple[int, ...] = ()) -> RejectionSampler:
    max_num_reqs = 16
    thinking_budget = np.zeros(max_num_reqs, dtype=bool)
    thinking_budget[list(active_budget_slots)] = True
    states = SimpleNamespace(
        top_k=_ArrayState([20] * max_num_reqs),
        temperature=_ArrayState([0.6] * max_num_reqs),
        min_p=_ArrayState([0.0] * max_num_reqs),
        max_num_logprobs=lambda idx: -1,
    )
    sampler = SimpleNamespace(
        compute_nans=False,
        sampling_states=states,
        logprob_token_ids_state=SimpleNamespace(max_num_token_ids=lambda idx: 0),
        logit_bias_state=SimpleNamespace(
            use_logit_bias=np.zeros(max_num_reqs, dtype=bool)
        ),
        bad_words_state=SimpleNamespace(
            num_bad_words=SimpleNamespace(np=np.zeros(max_num_reqs, dtype=np.int32))
        ),
        thinking_budget_state=SimpleNamespace(
            enabled=True,
            use_thinking_budget=thinking_budget,
        ),
    )
    rejection = RejectionSampler.__new__(RejectionSampler)
    rejection.sampler = sampler
    rejection._ag2_sparse_target_topk = True
    rejection._ag2_sparse_target_request_prefix = "chatcmpl-ag2-wide-sparse-"
    rejection._ag2_sparse_target_min_reqs = 8
    rejection._ag2_rejection_capture = None
    rejection._record_sparse_threshold_fallback = lambda num_reqs: None
    return rejection


def _make_input_batch() -> SimpleNamespace:
    slots = np.asarray([7, 1, 12, 3, 14, 5, 9, 0], dtype=np.int32)
    return SimpleNamespace(
        num_reqs=len(slots),
        req_ids=[f"chatcmpl-ag2-wide-sparse-{i}" for i in range(len(slots))],
        idx_mapping_np=slots,
    )


def test_sparse_target_accepts_plain_active_mrv2_slots() -> None:
    rejection = _make_sampler()
    assert rejection.can_use_sparse_target_topk(_make_input_batch())


def test_sparse_target_accepts_active_thinking_budget_slot() -> None:
    rejection = _make_sampler(active_budget_slots=(12,))
    assert rejection.can_use_sparse_target_topk(_make_input_batch())


def test_sparse_target_ignores_inactive_thinking_budget_slot() -> None:
    rejection = _make_sampler(active_budget_slots=(2,))
    assert rejection.can_use_sparse_target_topk(_make_input_batch())


def test_sparse_target_accepts_disabled_thinking_budget_state() -> None:
    rejection = _make_sampler()
    rejection.sampler.thinking_budget_state = SimpleNamespace(enabled=False)
    assert rejection.can_use_sparse_target_topk(_make_input_batch())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-backed UVA")
def test_sparse_target_tracks_real_mrv2_thinking_budget_slots() -> None:
    class _ReasoningConfig:
        reasoning_start_token_ids = [90]
        reasoning_end_token_ids = [91]
        natural_reasoning_end_token_ids = [91]

    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=16,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=3,
        vocab_size=128,
        device=device,
    )
    thinking_budget = ThinkingBudgetState(req_states, _ReasoningConfig())
    thinking_budget.add_request(12, SamplingParams(thinking_token_budget=32))
    thinking_budget.apply_staged_writes()

    rejection = _make_sampler()
    rejection.sampler.thinking_budget_state = thinking_budget
    batch = _make_input_batch()
    assert rejection.can_use_sparse_target_topk(batch)

    thinking_budget.add_request(12, SamplingParams(thinking_token_budget=None))
    thinking_budget.apply_staged_writes()
    assert rejection.can_use_sparse_target_topk(batch)
