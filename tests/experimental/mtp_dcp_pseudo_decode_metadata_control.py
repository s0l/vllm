#!/usr/bin/env python3
"""GPU control for DCP speculative rows routed through paged prefill."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flashinfer import (
    BatchDCPPseudoPrefillWrapper,
    FIPrefill,
    FlashInferMetadataBuilder,
)


def main() -> None:
    device = torch.device("cuda")
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder.dcp_world_size = 3
    builder.dcp_rank = 0
    builder.dcp_kv_cache_interleave_size = 1
    builder.use_dcp = True
    builder._dcp_pseudo_decode_enabled = True
    builder._dcp_sequential_decode_enabled = False
    builder._dcp_batched_decode_enabled = False
    builder._dcp_special_decode_enabled = True
    builder._dcp_canonical_paged_prefill = False
    builder._dcp_absolute_segment_prefill = False
    builder._dcp_prefill_cudagraph_enabled = False
    builder._dcp_match_fp8_new_tokens = False
    builder._dcp_prefill_match_fp8_new_tokens = False
    builder._dcp_pseudo_decode_query_len = 3
    builder._dcp_pseudo_decode_max_rows = 192
    builder._dcp_pseudo_decode_block_tables = torch.empty(
        (192, 74),
        dtype=torch.int32,
        device=device,
    )
    builder._dcp_pseudo_decode_row_to_req = torch.empty(
        192,
        dtype=torch.int64,
        device=device,
    )

    source = torch.arange(
        2 * 74,
        dtype=torch.int32,
        device=device,
    ).reshape(2, 74)
    block_ptr = builder._dcp_pseudo_decode_block_tables.data_ptr()
    map_ptr = builder._dcp_pseudo_decode_row_to_req.data_ptr()

    blocks, seq_lens_cpu = builder._prepare_dcp_pseudo_decode(
        torch.tensor([44, 3553], dtype=torch.int32),
        torch.tensor([0, 3, 6], dtype=torch.int32),
        source,
    )
    torch.cuda.synchronize()
    assert blocks.shape == (6, 74)
    assert seq_lens_cpu.tolist() == [14, 15, 15, 1184, 1184, 1185]
    assert torch.equal(blocks[0], source[0])
    assert torch.equal(blocks[2], source[0])
    assert torch.equal(blocks[3], source[1])
    assert torch.equal(blocks[5], source[1])

    # Replay a different batch into the same addresses. CUDA Graph consumers
    # must never observe a newly allocated GPU metadata tensor.
    blocks_replay, seq_lens_replay = builder._prepare_dcp_pseudo_decode(
        torch.tensor([3555], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        source[:1],
    )
    torch.cuda.synchronize()
    assert blocks_replay.shape == (3, 74)
    assert seq_lens_replay.tolist() == [1185, 1185, 1185]
    assert builder._dcp_pseudo_decode_block_tables.data_ptr() == block_ptr
    assert builder._dcp_pseudo_decode_row_to_req.data_ptr() == map_ptr

    try:
        builder._prepare_dcp_pseudo_decode(
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32),
            torch.empty((1, 75), dtype=torch.int32, device=device),
        )
    except ValueError as exc:
        assert "block-table capacity is too small" in str(exc)
    else:
        raise AssertionError("Oversized block table was not rejected.")

    positive = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        num_actual_tokens=6,
        causal=True,
        is_prefilling=np.array([False, False]),
    )
    assert builder._can_use_dcp_pseudo_decode(0, positive)

    qlen1 = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_actual_tokens=2,
        causal=True,
        is_prefilling=np.array([False, False]),
    )
    assert not builder._can_use_dcp_pseudo_decode(0, qlen1)

    real_prefill = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        num_actual_tokens=3,
        causal=True,
        is_prefilling=np.array([True]),
    )
    assert not builder._can_use_dcp_pseudo_decode(0, real_prefill)
    assert not builder._can_use_dcp_pseudo_decode(64, positive)
    autotune = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 6656], dtype=torch.int32),
        num_actual_tokens=6656,
        causal=True,
        is_prefilling=np.array([False]),
    )
    assert not builder._can_use_dcp_pseudo_decode(0, autotune)

    # Exercise complete metadata dispatch while stubbing only wrapper.plan.
    builder.reorder_batch_threshold = 1
    builder.page_size = 1184
    builder.num_qo_heads = 24
    builder.num_kv_heads = 4
    builder.head_dim = 256
    builder.attention_config = SimpleNamespace(use_trtllm_attention=True)
    builder.cache_dtype = "fp8_e4m3"
    builder.q_data_type_prefill = torch.bfloat16
    builder.q_data_type_decode = torch.bfloat16
    builder.has_sinks = False
    builder.use_trtllm_decode_attention = False
    builder.global_hyperparameters = SimpleNamespace(
        has_same_window_lefts=True,
        has_same_all_params=True,
    )
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    builder.kv_cache_dtype = torch.float8_e4m3fn
    builder.is_kvcache_nvfp4 = False
    builder.enable_cuda_graph = False
    builder.prefill_fixed_split_size = -1
    builder.prefill_disable_split_kv = False
    builder.sm_scale = 0.0625
    builder.window_left = -1
    builder.logits_soft_cap = None
    builder.pin_memory = False
    builder.device = device
    builder.paged_kv_indptr = builder._make_buffer(193)
    builder.paged_kv_indptr_cpu_buffer = torch.zeros_like(
        builder.paged_kv_indptr.cpu
    )
    builder.paged_kv_indices = builder._make_buffer(64 * 222)
    builder.paged_kv_last_page_len = builder._make_buffer(192)
    planned: dict[str, object] = {}
    pseudo_wrapper = BatchDCPPseudoPrefillWrapper.__new__(
        BatchDCPPseudoPrefillWrapper
    )

    def capture_plan(**kwargs: object) -> None:
        planned.update(kwargs)

    pseudo_wrapper.plan = capture_plan
    builder._get_dcp_pseudo_prefill_wrapper = lambda: pseudo_wrapper

    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([44, 3553], dtype=torch.int32, device=device),
        num_reqs=2,
        num_actual_tokens=6,
        max_query_len=3,
        max_seq_len=3553,
        block_table_tensor=source,
        slot_mapping=torch.arange(6, dtype=torch.int64, device=device),
        causal=True,
        is_prefilling=np.array([False, False]),
        _seq_lens_cpu=torch.tensor([44, 3553], dtype=torch.int32),
    )
    metadata = builder.build(0, common)
    torch.cuda.synchronize()
    assert metadata.num_decodes == 0
    assert metadata.num_decode_tokens == 0
    assert metadata.num_prefills == 6
    assert metadata.num_prefill_tokens == 6
    assert isinstance(metadata.prefill, FIPrefill)
    assert metadata.prefill.wrapper is pseudo_wrapper
    assert builder.paged_kv_indptr.cpu[:7].tolist() == [0, 1, 2, 3, 4, 5, 7]
    assert builder.paged_kv_last_page_len.cpu[:6].tolist() == [
        14,
        15,
        15,
        1184,
        1184,
        1,
    ]
    assert builder.paged_kv_indices.gpu[:7].tolist() == [0, 0, 0, 74, 74, 74, 75]
    assert planned["qo_indptr_cpu"].tolist() == list(range(7))
    assert planned["num_qo_heads"] == 24
    assert planned["dcp_world_size"] == 3
    assert planned["q_data_type"] == torch.bfloat16
    assert planned["kv_cache_dtype"] == torch.float8_e4m3fn

    # Configured max_num_seqs=64 with K=2 expands to exactly 192 rows.
    max_source = torch.arange(
        64 * 74,
        dtype=torch.int32,
        device=device,
    ).reshape(64, 74)
    max_blocks, max_seq_lens = builder._prepare_dcp_pseudo_decode(
        torch.arange(3, 67, dtype=torch.int32),
        torch.arange(0, 193, 3, dtype=torch.int32),
        max_source,
    )
    torch.cuda.synchronize()
    assert max_blocks.shape == (192, 74)
    assert max_seq_lens.shape == (192,)
    assert torch.equal(max_blocks[189], max_source[63])
    assert torch.equal(max_blocks[191], max_source[63])

    print(
        "PASS: persistent pseudo-prefill metadata, aliased pages, causal row "
        "lengths, replay address stability, full 64-request capacity, builder "
        "dispatch containment, and negative controls"
    )


if __name__ == "__main__":
    main()
