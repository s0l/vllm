# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controls for the research-only FlashInfer DCP prefill CUDA Graph lane."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires CUDA.", allow_module_level=True)

from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder


def _builder(*, enabled: bool = True) -> FlashInferMetadataBuilder:
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._dcp_prefill_cudagraph_enabled = enabled
    builder._dcp_pseudo_decode_query_len = 3
    builder.use_dcp = True
    return builder


def _metadata(
    query_start_loc: list[int],
    *,
    is_prefilling: list[bool],
    draft_counts: list[int] | None,
    causal: bool = True,
):
    return type(
        "SyntheticMetadata",
        (),
        {
            "query_start_loc_cpu": torch.tensor(
                query_start_loc, dtype=torch.int32
            ),
            "is_prefilling": torch.tensor(is_prefilling, dtype=torch.bool),
            "num_decode_draft_tokens_cpu": (
                None
                if draft_counts is None
                else torch.tensor(draft_counts, dtype=torch.int32)
            ),
            "causal": causal,
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
    metadata = _metadata(
        [0, 3], is_prefilling=[False], draft_counts=[2]
    )

    assert (
        _builder(enabled=False)._dcp_prefill_cudagraph_batch_size(
            metadata,
            num_decodes=0,
            num_prefills=1,
        )
        is None
    )
