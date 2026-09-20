# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_module
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState


class _ArrayState:
    def __init__(self, values: list[float] | list[int]):
        self.np = np.asarray(values)
        self.gpu = torch.from_numpy(self.np)


def _make_sampler(*, active_budget_slots: tuple[int, ...] = ()) -> RejectionSampler:
    max_num_reqs = 16
    thinking_budget = np.zeros(max_num_reqs, dtype=bool)
    thinking_budget[list(active_budget_slots)] = True
    states = SimpleNamespace(
        top_k=_ArrayState([20] * max_num_reqs),
        temperature=_ArrayState([0.6] * max_num_reqs),
        seeds=_ArrayState([0] * max_num_reqs),
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


@pytest.mark.parametrize("blocked", [None, "allowed", "bias", "restore"])
def test_sparse_target_min_tokens_only_admission(blocked):
    rejection = _make_sampler()
    state = LogitBiasState.__new__(LogitBiasState)
    state.use_logit_bias = np.ones(16, dtype=bool)
    state.num_allowed_token_ids = _ArrayState([0] * 16)
    state.num_logit_bias = _ArrayState([0] * 16)
    state.restore_when_all_masked = _ArrayState([0] * 16)
    if blocked is not None:
        field = {
            "allowed": state.num_allowed_token_ids,
            "bias": state.num_logit_bias,
            "restore": state.restore_when_all_masked,
        }[blocked]
        field.np[12] = 1
    rejection.sampler.logit_bias_state = state
    assert rejection.can_use_sparse_target_topk(_make_input_batch()) == (
        blocked is None
    )
    # An inactive slot cannot poison the active batch's domain.
    if blocked is not None:
        field.np[12] = 0
        field.np[2] = 1
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


@pytest.mark.parametrize("shadow", [False, True])
def test_sparse_target_forwards_sequence_length_bounds(monkeypatch, shadow) -> None:
    rejection = _make_sampler()
    rejection._ag2_sparse_target_shadow = shadow
    rejection._ag2_sparse_steps = 0
    rejection._ag2_sparse_fallbacks = 0
    rejection._ag2_sparse_shadow_steps = 0
    rejection.num_speculative_steps = 3
    rejection.synthetic_conditional_rates = None
    rejection.use_block_verification = False
    rejection.sampler.use_fp64_gumbel = False
    rejection.sampler.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int32))
    )

    seq_lens_upper_bound = torch.tensor([123], dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=1,
        input_ids=torch.zeros(1, dtype=torch.int64),
        logits_indices=torch.tensor([0]),
        positions=torch.tensor([0]),
        expanded_idx_mapping=torch.tensor([0]),
        idx_mapping=torch.tensor([0]),
        idx_mapping_np=np.asarray([0]),
        expanded_local_pos=torch.tensor([0]),
        seq_lens_cpu_upper_bound=seq_lens_upper_bound,
        cu_num_logits=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
    )
    rejection.can_use_sparse_target_topk = lambda _: True

    local_logits = torch.arange(70, dtype=torch.float32).unsqueeze(0)
    sparse_oracle = torch.full((1, 210), -float("inf"))
    sparse_oracle[:, 5:70] = local_logits[:, 5:70]
    seen_bounds = []

    def apply_sampling_params(
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        seq_lens_upper_bound_np,
        **kwargs,
    ):
        seen_bounds.append(seq_lens_upper_bound_np.copy())
        return local_logits if logits.shape[-1] == 70 else sparse_oracle

    rejection.sampler.apply_sampling_params = apply_sampling_params
    rejection.sampler.sampling_states.vocab_size = 210
    rejection.sampler.sampling_states.apply_top_k_top_p = lambda logits, *_: logits

    def all_gather(tensor, dim=-1):
        if tensor.shape[-1] == 70:
            return torch.cat((tensor, tensor, tensor), dim=dim)
        return torch.cat((tensor, tensor, tensor), dim=dim)

    monkeypatch.setattr(
        rejection_module, "tensor_model_parallel_all_gather", all_gather
    )
    monkeypatch.setattr(
        rejection_module,
        "rejection_sample",
        lambda *args, **kwargs: (
            torch.zeros((1, 1), dtype=torch.int64),
            torch.ones(1, dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        rejection_module,
        "get_num_sampled_and_rejected",
        lambda *args, **kwargs: (
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        ),
    )

    rejection.sample_sparse_target_topk(local_logits, 0, batch)

    assert len(seen_bounds) == (2 if shadow else 1)
    assert all(np.array_equal(value, np.asarray([123])) for value in seen_bounds)


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
