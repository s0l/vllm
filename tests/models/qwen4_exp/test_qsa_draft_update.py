# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft metadata must retain addresses and consume live GPU state on replay."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig
from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
from vllm.models.qwen4_exp.nvidia.qsa_flashinfer import QSAFlashInferMetadataBuilder
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec


@pytest.mark.parametrize(
    "builder_cls", [QSAMetadataBuilder, QSAFlashInferMetadataBuilder]
)
def test_draft_update_rejects_non_decode_before_device_work(builder_cls):
    builder = object.__new__(builder_cls)
    metadata = SimpleNamespace(max_query_len=4, num_prefills=1, draft_common=None)
    with pytest.raises(ValueError, match="q1 decode"):
        builder.update_draft_decode_metadata(metadata)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph control")
@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("x", [1, 2, 4, 8, 16])
@torch.inference_mode()
def test_draft_update_matches_rebuild_across_boundaries_and_graph_tail(rank, x):
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.scheduler_config.max_num_batched_tokens = 64
    config.scheduler_config.max_num_seqs = 16
    config.parallel_config.tensor_parallel_size = 3
    config.parallel_config.decode_context_parallel_size = 3
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=3, parallel_drafting=False
    )
    config.model_config = SimpleNamespace(max_model_len=8192, dtype=torch.bfloat16)
    starts = torch.arange(x + 1, dtype=torch.int32, device=device)
    lengths = torch.tensor(
        ([3, 4, 7, 8, 191, 192, 193, 511] * 2)[:x], device=device, dtype=torch.int32
    )
    table = torch.arange(x * 16, device=device, dtype=torch.int32).reshape(x, 16)
    slots = torch.arange(x, device=device, dtype=torch.int64)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts.cpu(),
        seq_lens=lengths,
        num_reqs=x,
        num_actual_tokens=x,
        max_query_len=1,
        max_seq_len=8192,
        block_table_tensor=table,
        slot_mapping=slots,
    )
    specs = (
        CircularBufferSpec(
            block_size=8, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        ),
        MLAAttentionSpec(
            block_size=192,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            tokens_per_state=4,
        ),
    )
    builders = [
        QSAMetadataBuilder(spec, [str(i)], config, device)
        for i, spec in enumerate(specs)
    ]
    oracles = [
        QSAMetadataBuilder(spec, [str(i)], config, device)
        for i, spec in enumerate(specs)
    ]
    metadata = [builder.build(0, common) for builder in builders]
    main_builder = QSAFlashInferMetadataBuilder(specs[1], ["main"], config, device)
    main = main_builder.build(0, common)
    fields = (
        "token_to_req",
        "logical_positions",
        "visible_blocks",
        "slot_mapping",
        "k_work_metadata",
    )
    addresses = [[getattr(md, field).data_ptr() for field in fields] for md in metadata]

    def update():
        for builder, md in zip(builders, metadata, strict=True):
            builder.update_draft_decode_metadata(md)
        main_builder.update_draft_decode_metadata(main)

    def check():
        torch.accelerator.synchronize()
        # The control rebuild uses the current CPU query count; captured update
        # must derive the same count from live GPU state, including padded tails.
        common.query_start_loc_cpu = starts.cpu()
        for oracle, md, ptrs in zip(oracles, metadata, addresses, strict=True):
            expected = oracle.build(0, common)
            assert [getattr(md, field).data_ptr() for field in fields] == ptrs
            for field in fields:
                assert torch.equal(getattr(md, field), getattr(expected, field)), field
        assert main.block_table is table and main.slot_mapping is slots

    update()
    check()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        update()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph.replay()
    check()
    original = lengths.clone()
    for live, delta in ((x, 1), (max(1, x - 1), 2), (x, 4), (x, 0)):
        starts.copy_(torch.arange(x + 1, device=device).clamp_max(live))
        lengths.copy_(original + delta)
        lengths[live:] = 0
        slots.copy_(torch.arange(x, device=device))
        slots[rank::3] = -1  # DCP-local invalid slots are part of the input.
        slots[live:] = -1
        table.copy_(table.flip(0))
        graph.replay()
        check()
