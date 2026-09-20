# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
)
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_elastic_max_resident_seqs,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


def _elastic_config(capacities: tuple[int, ...]) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=max(capacities),
        kv_cache_tensors=[],
        kv_cache_groups=[],
        elastic_gdn_initial_blocks=10,
        elastic_gdn_blocks_per_request=9,
        elastic_attention_capacity_by_gdn_blocks=capacities,
    )


def test_elastic_resident_cap_uses_gdn_and_writable_attention_tail() -> None:
    capacities = [100] * 190 + [21] * (578 - 190)
    # N=20 needs GDN block 181 and 21 attention blocks. N=21 needs GDN
    # block 190 and 22 attention blocks, so the exact hard cap is 20.
    capacities[190] = 21

    assert get_elastic_max_resident_seqs(_elastic_config(tuple(capacities)), 44) == 20


def test_non_elastic_config_preserves_configured_ceiling() -> None:
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    assert get_elastic_max_resident_seqs(config, 44) == 44


def test_initial_gdn_mapping_is_included_in_physical_cap() -> None:
    capacities = [100] * 10 + [1] * 10
    config = _elastic_config(tuple(capacities))
    config.elastic_gdn_initial_blocks = 10

    try:
        get_elastic_max_resident_seqs(config, 1)
    except ValueError as exc:
        assert "cannot host one decode request" in str(exc)
    else:
        raise AssertionError("initial GDN mapping must constrain attention capacity")


def test_rank_safe_cap_is_propagated_to_every_worker() -> None:
    def worker_config(attention_block_size: int, budget: int) -> KVCacheConfig:
        return KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=80,
                    shared_by=["attention"],
                    backing_id="elastic-attention-0",
                    mapping_quantum=4,
                    num_blocks=10,
                    logical_block_size=attention_block_size,
                ),
                KVCacheTensor(
                    size=40,
                    shared_by=["gdn"],
                    backing_id="elastic-gdn",
                    mapping_quantum=4,
                    num_blocks=8,
                    logical_block_size=5,
                ),
            ],
            kv_cache_groups=[],
            elastic_attention_stride=attention_block_size,
            elastic_gdn_stride=5,
            elastic_mapping_quantum=4,
            elastic_gdn_initial_blocks=2,
            elastic_gdn_blocks_per_request=3,
            elastic_budget_bytes=budget,
        )

    workers = [worker_config(5, 64), worker_config(7, 60)]
    scheduler = generate_scheduler_kv_cache_config(
        workers,
        configured_max_num_seqs=44,
        enable_auto_resident_cap=True,
    )

    assert scheduler.effective_max_resident_seqs > 0
    assert {worker.effective_max_resident_seqs for worker in workers} == {
        scheduler.effective_max_resident_seqs
    }


def test_v2_dense_full_shapes_do_not_expand_piecewise_family() -> None:
    manager = object.__new__(CudaGraphManager)
    manager.compilation_config = MagicMock(
        cudagraph_capture_sizes=[3, 6, 9, 24, 48, 72],
        max_cudagraph_capture_size=72,
    )
    manager.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    manager.decode_query_len = 3
    manager.full_decode_query_lens = {1, 3}
    manager.expand_dynamic_decode_query_lens = False
    manager.varlen_decode = False
    manager.defer_startup_graphs = False
    manager.dynamic_piecewise_coverage_sizes = ()
    manager.dynamic_piecewise_capture_sizes = ()
    manager.dynamic_piecewise_safety_sizes = ()
    manager.dynamic_full_capture_sizes = ()
    manager.max_num_reqs = 44
    manager.max_uniform_decode_reqs = 20
    manager.lora_capture_cases = [0]
    manager.vllm_config = MagicMock(spec=VllmConfig)
    manager.vllm_config.speculative_config = None
    manager.vllm_config.cache_config.use_kda_recoverssm = False
    manager._dynamic_graph_entries = {}
    manager._candidates = {}
    manager._capture_descs = {}

    manager._init_candidates()

    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    qlen3_reqs = {desc.num_reqs for desc in full_descs if desc.uniform_token_count == 3}
    qlen1_reqs = {desc.num_reqs for desc in full_descs if desc.uniform_token_count == 1}
    piecewise_tokens = {
        desc.num_tokens for desc in manager._capture_descs[CUDAGraphMode.PIECEWISE]
    }
    assert qlen3_reqs == set(range(1, 21))
    assert qlen1_reqs == {3, 6, 9, 20}
    assert piecewise_tokens == {3, 6, 9, 24, 48, 72}
