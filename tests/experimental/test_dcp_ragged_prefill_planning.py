# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.attention.backends.flashinfer import BatchDCPPrefillWrapper


class _PlanRecorder:
    def __init__(self) -> None:
        self.kwargs = None

    def plan(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_dcp_empty_context_plans_only_current_tokens(monkeypatch) -> None:
    monkeypatch.delenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS",
        raising=False,
    )
    wrapper = object.__new__(BatchDCPPrefillWrapper)
    wrapper._context = _PlanRecorder()
    wrapper._new_tokens = _PlanRecorder()
    wrapper._disable_split_kv_for_cuda_graph = False
    wrapper._ragged_fixed_split_size = -1
    wrapper._context_fixed_split_size = -1
    wrapper._ag2_request_tail_indices = None
    wrapper._ag2_request_tail_count = 0

    wrapper.plan(
        qo_indptr_cpu=torch.arange(65, dtype=torch.int32) * 4,
        # The live post-allocation warmup retains one dummy page reference per
        # request even though every semantic context length is zero.
        paged_kv_indptr_cpu=torch.arange(65, dtype=torch.int32),
        paged_kv_indices=torch.arange(64, dtype=torch.int32),
        paged_kv_last_page_len_cpu=torch.zeros(64, dtype=torch.int32),
        page_size=64,
        num_qo_heads=8,
        dcp_world_size=3,
        num_kv_heads=4,
        head_dim=256,
        sm_scale=256**-0.5,
        window_left=-1,
        logits_soft_cap=None,
        q_data_type=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        prefill_fixed_split_size=-1,
        disable_split_kv=False,
    )

    assert wrapper._ag2_context_is_empty is True
    assert wrapper._context.kwargs is None
    assert wrapper._new_tokens.kwargs is not None


def test_dcp_ragged_new_tokens_receives_fixed_split_size(monkeypatch) -> None:
    monkeypatch.delenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS",
        raising=False,
    )
    wrapper = object.__new__(BatchDCPPrefillWrapper)
    wrapper._context = _PlanRecorder()
    wrapper._new_tokens = _PlanRecorder()
    wrapper._disable_split_kv_for_cuda_graph = False
    wrapper._ragged_fixed_split_size = 4096
    wrapper._context_fixed_split_size = -1
    wrapper._ag2_request_tail_indices = None
    wrapper._ag2_request_tail_count = 0

    indptr = torch.tensor([0, 169, 338], dtype=torch.int32)
    wrapper.plan(
        qo_indptr_cpu=indptr,
        paged_kv_indptr_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_last_page_len_cpu=torch.tensor([1, 1], dtype=torch.int32),
        page_size=16,
        num_qo_heads=8,
        dcp_world_size=3,
        num_kv_heads=4,
        head_dim=256,
        sm_scale=256**-0.5,
        window_left=-1,
        logits_soft_cap=None,
        q_data_type=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        prefill_fixed_split_size=-1,
        disable_split_kv=False,
    )

    assert wrapper._context.kwargs is not None
    assert wrapper._new_tokens.kwargs is not None
    assert wrapper._context.kwargs["fixed_split_size"] == -1
    assert wrapper._new_tokens.kwargs["fixed_split_size"] == 4096


def test_dcp_ragged_fixed_split_falls_back_to_global_prefill_value(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS",
        raising=False,
    )
    wrapper = object.__new__(BatchDCPPrefillWrapper)
    wrapper._context = _PlanRecorder()
    wrapper._new_tokens = _PlanRecorder()
    wrapper._disable_split_kv_for_cuda_graph = False
    wrapper._ragged_fixed_split_size = -1
    wrapper._context_fixed_split_size = -1
    wrapper._ag2_request_tail_indices = None
    wrapper._ag2_request_tail_count = 0

    indptr = torch.tensor([0, 169], dtype=torch.int32)
    wrapper.plan(
        qo_indptr_cpu=indptr,
        paged_kv_indptr_cpu=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len_cpu=torch.tensor([1], dtype=torch.int32),
        page_size=16,
        num_qo_heads=8,
        dcp_world_size=3,
        num_kv_heads=4,
        head_dim=256,
        sm_scale=256**-0.5,
        window_left=-1,
        logits_soft_cap=None,
        q_data_type=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        prefill_fixed_split_size=-1,
        disable_split_kv=False,
    )

    assert wrapper._new_tokens.kwargs is not None
    assert wrapper._new_tokens.kwargs["fixed_split_size"] == -1


def test_dcp_context_fixed_split_is_independent_from_ragged(monkeypatch) -> None:
    monkeypatch.delenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS",
        raising=False,
    )
    wrapper = object.__new__(BatchDCPPrefillWrapper)
    wrapper._context = _PlanRecorder()
    wrapper._new_tokens = _PlanRecorder()
    wrapper._disable_split_kv_for_cuda_graph = False
    wrapper._ragged_fixed_split_size = 4096
    wrapper._context_fixed_split_size = 8
    wrapper._ag2_request_tail_indices = None
    wrapper._ag2_request_tail_count = 0

    indptr = torch.tensor([0, 4, 8], dtype=torch.int32)
    wrapper.plan(
        qo_indptr_cpu=indptr,
        paged_kv_indptr_cpu=torch.tensor([0, 8, 16], dtype=torch.int32),
        paged_kv_indices=torch.arange(16, dtype=torch.int32),
        paged_kv_last_page_len_cpu=torch.tensor([57, 57], dtype=torch.int32),
        page_size=64,
        num_qo_heads=8,
        dcp_world_size=3,
        num_kv_heads=4,
        head_dim=256,
        sm_scale=256**-0.5,
        window_left=-1,
        logits_soft_cap=None,
        q_data_type=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        prefill_fixed_split_size=-1,
        disable_split_kv=False,
    )

    assert wrapper._context.kwargs is not None
    assert wrapper._new_tokens.kwargs is not None
    assert wrapper._context.kwargs["fixed_split_size"] == 8
    assert wrapper._new_tokens.kwargs["fixed_split_size"] == 4096
    assert wrapper._context.kwargs["disable_split_kv"] is False
