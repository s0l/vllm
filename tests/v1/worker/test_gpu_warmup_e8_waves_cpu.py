# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU control for the two disjoint E8 synthetic warmup lifetimes."""

from types import SimpleNamespace
from typing import Any

import torch

from tests.v1.worker.test_gpu_warmup_blocks import (
    NUM_SPEC_STEPS,
    _attention_group,
    _make_runner,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.warmup import _reserved_block_count, warmup_kernels


def _separate_gdn_group() -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        ["gdn"],
        MambaSpec(
            block_size=192,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            separate_pool=True,
        ),
    )


def test_separate_gdn_warmup_reservation_is_one_slot_at_full_chunk():
    spec = _separate_gdn_group().kv_cache_spec
    runner = _make_runner([_attention_group()], NUM_SPEC_STEPS)
    runner.vllm_config.cache_config = SimpleNamespace(mamba_cache_mode="align")
    assert spec.max_num_blocks_per_req(runner.vllm_config, 262144) == 1
    assert (
        _reserved_block_count(
            4096,
            spec,
            num_lookahead_tokens=1,
            max_model_len=262144,
            max_encoder_len=0,
        )
        == 1
    )


def test_e8_large_prefill_restarts_block_ids_after_first_wave(monkeypatch):
    runner = _make_runner([_separate_gdn_group(), _attention_group()], NUM_SPEC_STEPS)
    runner.max_model_len = 128
    runner.scheduler_config.max_num_batched_tokens = 128
    runner.vllm_config.additional_config = {
        "flashnext_native_experts": {"e8_archive": {"max_tokens": 1024}}
    }
    runner.kv_cache_config.elastic_mapping_quantum = 0
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    events: list[tuple[Any, ...]] = []

    def execute(output):
        for req in output.scheduled_new_reqs:
            events.append(
                (
                    "new",
                    req.req_id,
                    tuple(tuple(ids) for ids in req.block_ids),
                )
            )
        if output.finished_req_ids:
            events.append(("finish", tuple(sorted(output.finished_req_ids))))

    warmup_kernels(runner, execute, lambda _grammar=None: None)

    large_index = next(
        i
        for i, event in enumerate(events)
        if event[:2] == ("new", "_warmup_e8_large_prefill_")
    )
    ordinary_finish_index = next(
        i
        for i, event in enumerate(events)
        if event[0] == "finish" and "_warmup_0_" in event[1]
    )
    assert ordinary_finish_index < large_index
    large_blocks = events[large_index][2]
    assert min(block for group in large_blocks for block in group) == 1
    assert len(large_blocks[0]) == 1
    assert len(large_blocks[1]) == 8
    large_finish_index = events.index(("finish", ("_warmup_e8_large_prefill_",)))
    overlap_index = next(
        i
        for i, event in enumerate(events)
        if event[:2] == ("new", "_warmup_e8_overlap_0_")
    )
    assert large_index < large_finish_index < overlap_index
    assert events[-1] == (
        "finish",
        ("_warmup_e8_overlap_0_", "_warmup_e8_overlap_1_"),
    )
