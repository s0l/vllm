# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controls for the research-only FlashInfer DCP prefill CUDA Graph lane."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires CUDA.", allow_module_level=True)

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flashinfer import (
    BatchDCPPrefillWrapper,
    FlashInferMetadataBuilder,
    _semantic_attention_token_counts,
)


def _builder(*, enabled: bool = True) -> FlashInferMetadataBuilder:
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._dcp_prefill_cudagraph_enabled = enabled
    builder._dcp_pseudo_decode_query_len = 3
    builder.use_dcp = True
    return builder


def test_dynamic_trim_preserves_static_cudagraph_wrappers():
    builder = _builder()
    static_prefill = object()
    dynamic_prefill = object()
    static_decode = MagicMock()
    dynamic_decode = MagicMock()
    static_form = MagicMock(physical_rows=4)
    dynamic_form = MagicMock(physical_rows=8)
    builder._static_cudagraph_batch_sizes = frozenset({1, 4})
    builder._dcp_prefill_wrappers_cudagraph = {
        1: static_prefill,
        8: dynamic_prefill,
    }
    builder._dcp_pseudo_prefill_wrappers_cudagraph = {}
    builder._dcp_prefill_captured_wrappers = {
        1: static_prefill,
        8: dynamic_prefill,
    }
    builder._decode_wrappers_cudagraph = {4: static_decode, 8: dynamic_decode}
    builder._dcp_batched_decode_wrappers_cudagraph = {
        static_form: static_decode,
        dynamic_form: dynamic_decode,
    }

    removed = builder.trim_dynamic_cudagraph_wrappers(
        keep_request_batch_sizes=frozenset(),
        keep_token_batch_sizes=frozenset(),
    )

    assert removed == 2
    assert builder._dcp_prefill_wrappers_cudagraph == {1: static_prefill}
    assert builder._dcp_prefill_captured_wrappers == {1: static_prefill}
    assert builder._decode_wrappers_cudagraph == {4: static_decode}
    assert builder._dcp_batched_decode_wrappers_cudagraph == {
        static_form: static_decode
    }
    static_decode.retire_graph_lease.assert_not_called()
    dynamic_decode.retire_graph_lease.assert_called_once_with()


def test_dcp_overlapping_gqa_head_select_validates_before_cuda_launch():
    wrapper = BatchDCPPrefillWrapper.__new__(BatchDCPPrefillWrapper)
    wrapper._local_kv_head_index_tensors = {}
    key = torch.arange(2 * 4 * 8, dtype=torch.float32, device="cuda").view(2, 4, 8)
    value = key + 1000

    selected_key, selected_value = wrapper._select_local_kv_heads(
        key, value, (0, 0, 0, 1)
    )

    assert torch.equal(selected_key, key[:, (0, 0, 0, 1)])
    assert torch.equal(selected_value, value[:, (0, 0, 0, 1)])


def test_dcp_overlapping_gqa_head_select_rejects_invalid_geometry_synchronously():
    wrapper = BatchDCPPrefillWrapper.__new__(BatchDCPPrefillWrapper)
    wrapper._local_kv_head_index_tensors = {}
    key = torch.empty((2, 1, 128), dtype=torch.bfloat16, device="cuda")
    value = torch.empty_like(key)

    with pytest.raises(RuntimeError, match="head map exceeds"):
        wrapper._select_local_kv_heads(key, value, (0, 0, 0, 1))

    # The negative control must not enqueue a device-side assert.
    torch.cuda.synchronize()


def _metadata(
    query_start_loc: list[int],
    *,
    is_prefilling: list[bool],
    draft_counts: list[int] | None,
    causal: bool = True,
    full_cudagraph: bool = True,
):
    return type(
        "SyntheticMetadata",
        (),
        {
            "query_start_loc_cpu": torch.tensor(query_start_loc, dtype=torch.int32),
            "is_prefilling": torch.tensor(is_prefilling, dtype=torch.bool),
            "num_decode_draft_tokens_cpu": (
                None
                if draft_counts is None
                else torch.tensor(draft_counts, dtype=torch.int32)
            ),
            "causal": causal,
            "full_cudagraph": full_cudagraph,
        },
    )()


def test_dcp_prefill_cudagraph_accepts_uniform_mtp_with_padding():
    metadata = _metadata(
        [0, 3, 6, 6, 6],
        is_prefilling=[False, False, False, False],
        draft_counts=[2, 2, -1, -1],
    )

    assert (
        _builder()._dcp_prefill_cudagraph_batch_size(
            metadata,
            num_decodes=0,
            num_prefills=4,
        )
        == 4
    )


def test_dcp_prefill_piecewise_uses_replannable_wrapper():
    metadata = _metadata(
        [0, 3, 6],
        is_prefilling=[False, False],
        draft_counts=[2, 2],
        full_cudagraph=False,
    )

    assert (
        _builder()._dcp_prefill_cudagraph_batch_size(
            metadata,
            num_decodes=0,
            num_prefills=2,
        )
        is None
    )


def test_batched_q1_verifier_forces_qlen_greater_than_one_piecewise(monkeypatch):
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "1")

    assert (
        FlashInferMetadataBuilder.get_cudagraph_support(object(), object())
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


def test_dcp_prefill_cudagraph_capture_and_replay_use_same_wrapper_key():
    capture_metadata = _metadata(
        [0, 3, 6],
        is_prefilling=[False, False],
        draft_counts=None,
    )
    runtime_metadata = _metadata(
        [0, 3, 6],
        is_prefilling=[False, False],
        draft_counts=[2, 2],
    )
    builder = _builder()

    capture_key = builder._dcp_prefill_cudagraph_batch_size(
        capture_metadata,
        num_decodes=0,
        num_prefills=2,
        for_cudagraph_capture=True,
    )
    runtime_key = builder._dcp_prefill_cudagraph_batch_size(
        runtime_metadata,
        num_decodes=0,
        num_prefills=2,
    )

    assert capture_key == runtime_key == 2


def test_dcp_prefill_cudagraph_runtime_without_draft_markers_fails_closed():
    metadata = _metadata(
        [0, 3, 6],
        is_prefilling=[False, False],
        draft_counts=None,
    )

    assert (
        _builder()._dcp_prefill_cudagraph_batch_size(
            metadata,
            num_decodes=0,
            num_prefills=2,
        )
        is None
    )


def test_dcp_prefill_cudagraph_capture_and_runtime_wrapper_identity():
    builder = _builder()
    builder._dcp_prefill_captured_wrappers = {}
    wrapper = object()

    builder._verify_dcp_prefill_cudagraph_wrapper_identity(
        2, wrapper, for_cudagraph_capture=True
    )
    builder._verify_dcp_prefill_cudagraph_wrapper_identity(
        2, wrapper, for_cudagraph_capture=False
    )

    with pytest.raises(RuntimeError, match="wrapper mismatch"):
        builder._verify_dcp_prefill_cudagraph_wrapper_identity(
            2, object(), for_cudagraph_capture=False
        )

    with pytest.raises(RuntimeError, match="uncaptured wrapper"):
        builder._verify_dcp_prefill_cudagraph_wrapper_identity(
            3, object(), for_cudagraph_capture=False
        )


def test_dcp_prefill_cudagraph_uses_only_captured_runtime_shapes():
    builder = _builder()
    builder._dcp_prefill_captured_wrappers = {64: object()}

    assert (
        builder._select_captured_dcp_prefill_cudagraph_batch_size(
            64, for_cudagraph_capture=False
        )
        == 64
    )
    assert (
        builder._select_captured_dcp_prefill_cudagraph_batch_size(
            92, for_cudagraph_capture=False
        )
        is None
    )
    assert (
        builder._select_captured_dcp_prefill_cudagraph_batch_size(
            92, for_cudagraph_capture=True
        )
        == 92
    )


@pytest.mark.parametrize(
    ("metadata", "num_decodes"),
    [
        (
            _metadata(
                [0, 3, 5],
                is_prefilling=[False, False],
                draft_counts=[2, 2],
            ),
            0,
        ),
        (
            _metadata([0, 3], is_prefilling=[True], draft_counts=[2]),
            0,
        ),
        (
            _metadata(
                [0, 3],
                is_prefilling=[False],
                draft_counts=[2],
                causal=False,
            ),
            0,
        ),
        (
            _metadata([0, 3], is_prefilling=[False], draft_counts=[2]),
            1,
        ),
        (
            _metadata(
                [0, 3, 6],
                is_prefilling=[False, False],
                draft_counts=[2, -1],
            ),
            0,
        ),
    ],
)
def test_dcp_prefill_cudagraph_rejects_non_target_shapes(metadata, num_decodes):
    assert (
        _builder()._dcp_prefill_cudagraph_batch_size(
            metadata,
            num_decodes=num_decodes,
            num_prefills=1,
        )
        is None
    )


def test_dcp_prefill_cudagraph_is_default_off():
    metadata = _metadata([0, 3], is_prefilling=[False], draft_counts=[2])

    assert (
        _builder(enabled=False)._dcp_prefill_cudagraph_batch_size(
            metadata,
            num_decodes=0,
            num_prefills=1,
        )
        is None
    )


def test_padded_full_graph_uses_semantic_attention_rows():
    assert _semantic_attention_token_counts(
        92,
        torch.tensor([*range(0, 89, 4), 88], dtype=torch.int32),
        0,
    ) == (88, 0, 88)


def test_mixed_padded_graph_preserves_decode_prefill_boundary():
    assert _semantic_attention_token_counts(
        16,
        torch.tensor([0, 1, 2, 6, 10], dtype=torch.int32),
        2,
    ) == (10, 2, 8)


def test_semantic_attention_rows_cannot_exceed_physical_carrier():
    with pytest.raises(ValueError, match="exceed the physical graph carrier"):
        _semantic_attention_token_counts(
            8,
            torch.tensor([0, 4, 12], dtype=torch.int32),
            0,
        )


def test_native_decode_tail_is_not_a_semantic_prefill_graph():
    metadata = _metadata(
        list(range(12)),
        is_prefilling=[False] * 11,
        draft_counts=[0] * 11,
    )

    # Eleven live qlen1 rows may be carried by an M12 outer graph.  The
    # FlashInfer native-decode dispatch must preserve the physical M12 token
    # count; the semantic-row conversion belongs only to the DCP prefill lane.
    assert (
        _builder()._dcp_prefill_cudagraph_batch_size(
            metadata,
            num_decodes=12,
            num_prefills=0,
        )
        is None
    )
