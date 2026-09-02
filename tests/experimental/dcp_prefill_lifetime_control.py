"""Deterministic control for DCP prefill intermediate lifetimes."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.attention.backends import flashinfer


class _Group:
    def __init__(self) -> None:
        self.gathered_ref: weakref.ReferenceType[torch.Tensor] | None = None

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        gathered = torch.cat((tensor, tensor, tensor), dim=dim)
        self.gathered_ref = weakref.ref(gathered)
        return gathered


class _Context:
    def __init__(self) -> None:
        self.output_ref: weakref.ReferenceType[torch.Tensor] | None = None

    def run(self, query: torch.Tensor, *_args, **_kwargs):
        output = query + 1
        self.output_ref = weakref.ref(output)
        lse = torch.zeros(output.shape[:-1], dtype=torch.float32)
        return output, lse


class _NewTokens:
    def __init__(self, group: _Group, context: _Context) -> None:
        self.group = group
        self.context = context

    def run(self, query: torch.Tensor, *_args, **_kwargs):
        gc.collect()
        assert self.group.gathered_ref is not None
        assert self.group.gathered_ref() is None
        assert self.context.output_ref is not None
        assert self.context.output_ref() is None
        output = query + 2
        lse = torch.zeros(output.shape[:-1], dtype=torch.float32)
        return output, lse


def main() -> None:
    group = _Group()
    context = _Context()
    wrapper = flashinfer.BatchDCPPrefillWrapper.__new__(
        flashinfer.BatchDCPPrefillWrapper
    )
    wrapper._context = context
    wrapper._new_tokens = _NewTokens(group, context)
    wrapper._kv_layout = "NHD"
    wrapper._ag2_history_qo_indptr_cpu = None
    wrapper._ag2_history_paged_kv_indptr_cpu = None
    wrapper._ag2_history_paged_kv_indices = None
    wrapper._ag2_history_last_page_len_cpu = None
    wrapper._use_cuda_graph = False
    wrapper._ag2_request_trace_row = "tail"
    wrapper._absolute_segmented = False
    wrapper._canonical_paged = False
    wrapper._ag2_request_tail_indices = None
    wrapper._ag2_request_tail_count = 0
    wrapper._ag2_sm_scale = 1.0
    wrapper._convert_log2_lse_for_merge = False
    wrapper._dcp_combine = lambda output, lse, *_args, **_kwargs: (
        output[:, :2].clone(),
        lse[:, :2].clone(),
    )

    query = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    out = torch.empty_like(query)
    layer = SimpleNamespace(
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        _v_scale_float=1.0,
        dcp_full_kv_attention_heads=False,
    )

    def merge(
        destination: torch.Tensor,
        context_output: torch.Tensor,
        _context_lse: torch.Tensor,
        query_output: torch.Tensor,
        _query_lse: torch.Tensor,
    ) -> None:
        destination.copy_(context_output + query_output)

    with (
        patch.object(flashinfer, "get_dcp_group", return_value=group),
        patch.object(flashinfer, "merge_attn_states", side_effect=merge),
    ):
        actual = wrapper.run(layer, query, (query, query), query, query, out)

    torch.testing.assert_close(actual, (query + 1) + (query + 2))
    print(
        {
            "exact": True,
            "gathered_released_before_new_tokens": True,
            "context_output_released_before_new_tokens": True,
        }
    )


if __name__ == "__main__":
    main()
