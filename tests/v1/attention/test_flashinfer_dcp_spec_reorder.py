# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer GQA builder: reorder threshold under DCP with spec decode."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flashinfer import (
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
    _dcp_pseudo_decode_rows,
)
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_flashinfer_gqa_dcp_spec_decode_clamps_reorder_threshold(monkeypatch):
    """trtllm-gen decode receives no cp_rank/global-seq-len information, so its
    end-aligned causal mask is wrong for q_len > 1 over the DCP-interleaved
    local KV shard. The builder must keep reorder_batch_threshold at 1 under
    DCP so spec queries take the (DCP-aware) prefill path instead.
    """
    vllm_config = create_vllm_config(max_model_len=1024)
    vllm_config.parallel_config.decode_context_parallel_size = 2
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )

    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: FlashInferDecodeKernel.TRTLLM_GEN),
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec,
            ["layer.0"],
            vllm_config,
            torch.device("cpu"),
        )

    # Guard against passing vacuously with the kernel disabled.
    assert (
        builder.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
    )
    assert builder.reorder_batch_threshold == 1


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, [14, 15, 15]),
        (1, [14, 14, 15]),
        (2, [14, 14, 14]),
    ],
)
def test_dcp_pseudo_decode_rows_preserve_per_token_causal_lengths(rank, expected):
    row_to_req, local_lens = _dcp_pseudo_decode_rows(
        seq_lens_cpu=torch.tensor([44], dtype=torch.int32),
        qo_indptr_cpu=torch.tensor([0, 3], dtype=torch.int32),
        dcp_world_size=3,
        dcp_rank=rank,
        dcp_kv_cache_interleave_size=1,
    )

    assert row_to_req.tolist() == [0, 0, 0]
    assert local_lens.tolist() == expected


def test_dcp_pseudo_decode_rows_map_multiple_requests_without_aliasing_lengths():
    row_to_req, local_lens = _dcp_pseudo_decode_rows(
        seq_lens_cpu=torch.tensor([44, 3553], dtype=torch.int32),
        qo_indptr_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        dcp_world_size=3,
        dcp_rank=0,
        dcp_kv_cache_interleave_size=1,
    )

    assert row_to_req.tolist() == [0, 0, 0, 1, 1, 1]
    assert local_lens.tolist() == [14, 15, 15, 1184, 1184, 1185]


def test_dcp_pseudo_decode_rows_preserve_padded_full_graph_carrier():
    row_to_req, local_lens = _dcp_pseudo_decode_rows(
        seq_lens_cpu=torch.tensor([44, 3553, 0], dtype=torch.int32),
        qo_indptr_cpu=torch.tensor([0, 3, 6, 6], dtype=torch.int32),
        dcp_world_size=3,
        dcp_rank=0,
        dcp_kv_cache_interleave_size=1,
        padded_num_rows=9,
    )

    assert row_to_req.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert local_lens.tolist() == [14, 15, 15, 1184, 1184, 1185, 14, 14, 14]


def test_dcp_pseudo_decode_rows_reject_too_small_physical_carrier():
    with pytest.raises(ValueError, match="smaller than its semantic rows"):
        _dcp_pseudo_decode_rows(
            seq_lens_cpu=torch.tensor([44], dtype=torch.int32),
            qo_indptr_cpu=torch.tensor([0, 3], dtype=torch.int32),
            dcp_world_size=3,
            dcp_rank=0,
            dcp_kv_cache_interleave_size=1,
            padded_num_rows=2,
        )


def test_dcp_pseudo_decode_keeps_dispatch_family_on_full_graph_tail():
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._dcp_special_decode_enabled = True
    builder._dcp_pseudo_decode_query_len = 3
    builder._dcp_pseudo_decode_max_rows = 9
    builder.use_dcp = True
    synthetic_tail = type(
        "SyntheticTail",
        (),
        {
            "query_start_loc_cpu": torch.tensor([0, 3, 6, 6], dtype=torch.int32),
            "num_actual_tokens": 9,
            "causal": True,
            "is_prefilling": torch.tensor([False, False, False]),
        },
    )()

    assert builder._can_use_dcp_pseudo_decode(0, synthetic_tail)


def test_dcp_pseudo_decode_rejects_non_verification_autotune_shape():
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._dcp_pseudo_decode_enabled = True
    builder._dcp_special_decode_enabled = True
    builder._dcp_pseudo_decode_query_len = 3
    builder._dcp_pseudo_decode_max_rows = 192
    builder.use_dcp = True

    synthetic_decode = type(
        "SyntheticDecode",
        (),
        {
            "query_start_loc_cpu": torch.tensor([0, 6656], dtype=torch.int32),
            "num_actual_tokens": 6656,
            "causal": True,
            "is_prefilling": torch.tensor([False]),
        },
    )()

    assert not builder._can_use_dcp_pseudo_decode(0, synthetic_decode)
