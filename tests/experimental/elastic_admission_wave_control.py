# SPDX-License-Identifier: Apache-2.0
"""CPU-only controls for elastic GDN admission-wave reservation."""

from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_coordinator import KVCacheBlockPoolRequirements
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.request import RequestStatus


def make_manager() -> KVCacheManager:
    block_size = 4
    mamba_spec = MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        separate_pool=True,
        separate_pool_num_blocks=577,
    )
    config = KVCacheConfig(
        num_blocks=66,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            *(KVCacheGroupSpec([f"gdn-{i}"], mamba_spec) for i in range(3)),
        ],
        elastic_attention_stride=81887232,
        elastic_gdn_stride=19488768,
        elastic_mapping_quantum=2 * 1024 * 1024,
        elastic_gdn_initial_blocks=10,
        elastic_gdn_blocks_per_request=9,
        elastic_budget_bytes=5623686329,
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=262144,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )


def main() -> None:
    manager = make_manager()
    coordinator = manager.coordinator
    attention = coordinator.block_pool
    gdn = coordinator.mamba_block_pool
    assert gdn is not None

    # Put high attention IDs at the head of the public free queue. Without a
    # wave reservation, early requests would pin the future shrink tail.
    history = attention.get_new_blocks(52)
    attention.free_blocks(reversed(history))
    assert attention.get_num_free_blocks() == 65

    assert coordinator.apply_elastic_admission_wave((1,) * 16) == 16
    assert (attention.active_num_gpu_blocks, gdn.active_num_gpu_blocks) == (34, 145)

    request_attention = []
    request_gdn = []
    for _ in range(16):
        a_blocks = attention.get_new_blocks(1)
        g_blocks = gdn.get_new_blocks(9)
        request_attention.append(a_blocks)
        request_gdn.append(g_blocks)
    allocated_ids = [blocks[0].block_id for blocks in request_attention]
    assert allocated_ids == list(range(1, 17))
    coordinator.rebalance_elastic_capacity()
    assert coordinator.take_elastic_transition() == (34, 145)
    assert coordinator.last_elastic_rejection is None

    # Keep one request and release the other fifteen. The arena must reclaim
    # the GDN wave and make the whole attention capacity available again.
    for a_blocks, g_blocks in zip(request_attention[1:], request_gdn[1:], strict=True):
        attention.free_blocks(reversed(a_blocks))
        gdn.free_blocks(reversed(g_blocks))
    coordinator.rebalance_elastic_capacity()
    assert (attention.active_num_gpu_blocks, gdn.active_num_gpu_blocks) == (66, 10)
    assert coordinator.take_elastic_transition() == (66, 10)

    long_tail = attention.get_new_blocks(64)
    assert attention.get_num_free_blocks() == 0
    attention.free_blocks(reversed(long_tail))
    attention.free_blocks(reversed(request_attention[0]))
    gdn.free_blocks(reversed(request_gdn[0]))
    coordinator.rebalance_elastic_capacity()
    assert coordinator.take_elastic_transition() is None

    # If no offered request is admitted, the speculative scheduler-side
    # reservation collapses before publication and emits no worker transition.
    unused = make_manager().coordinator
    assert unused.apply_elastic_admission_wave((1,) * 16) == 16
    unused.rebalance_elastic_capacity()
    assert unused.take_elastic_transition() is None

    # A long request at the head must reduce the wave, not be starved by a
    # layout sized as if every candidate needed only one primary block.
    long_first = make_manager().coordinator
    admitted = long_first.apply_elastic_admission_wave((40,) + (1,) * 15)
    assert 0 < admitted < 16
    assert long_first.block_pool.get_num_free_blocks() >= 40 + max(admitted - 1, 0)
    estimated = make_manager().estimate_uncached_full_sequence_requirements(
        SimpleNamespace(
            request_id="long",
            num_tokens=160,
            num_computed_tokens=0,
        )
    )
    assert estimated.primary == 40

    # Incremental-arrival control: a reservation can cover only requests
    # already offered to the scheduler. Elastic primary allocation must
    # therefore choose low logical IDs even when LRU history puts high IDs at
    # the queue head, leaving later shrink tails unreferenced.
    staged = make_manager().coordinator
    staged_attention = staged.block_pool
    staged_gdn = staged.mamba_block_pool
    assert staged_gdn is not None
    staged_history = staged_attention.get_new_blocks(52)
    staged_attention.free_blocks(reversed(staged_history))
    assert staged.apply_elastic_admission_wave((1,) * 14) == 14
    first_wave_attention = staged_attention.get_new_blocks(14)
    first_wave_gdn = staged_gdn.get_new_blocks(14 * 9)
    first_wave_ids = [block.block_id for block in first_wave_attention]
    assert first_wave_ids == list(range(1, 15)), first_wave_ids
    assert staged.apply_elastic_admission_wave((1, 1)) == 2
    second_wave_attention = staged_attention.get_new_blocks(2)
    second_wave_gdn = staged_gdn.get_new_blocks(2 * 9)
    assert [block.block_id for block in second_wave_attention] == [15, 16]
    assert staged.last_elastic_rejection is None
    staged_attention.free_blocks(reversed(second_wave_attention))
    staged_gdn.free_blocks(reversed(second_wave_gdn))
    staged_attention.free_blocks(reversed(first_wave_attention))
    staged_gdn.free_blocks(reversed(first_wave_gdn))

    # Reproduce the live chapter failure geometrically. A nine-request wave
    # leaves a logical attention tail active. If a shared-prefix hit then
    # references the first block in that tail, the incrementally arriving
    # tenth request cannot trade it for nine GDN blocks. Reserving one
    # lookahead slot with the first visible wave deactivates that block before
    # cache lookup. (The production geometry observed the same boundary at 42;
    # this reduced control reaches it at 46.)
    no_lookahead = make_manager().coordinator
    no_lookahead_attention = no_lookahead.block_pool
    no_lookahead_gdn = no_lookahead.mamba_block_pool
    assert no_lookahead_gdn is not None
    assert no_lookahead.apply_elastic_admission_wave((1,) * 9) == 9
    assert no_lookahead_attention.active_num_gpu_blocks == 49
    no_lookahead_gdn_leases = no_lookahead_gdn.get_new_blocks(9 * 9)
    pinned_shared_prefix = no_lookahead_attention.blocks[46]
    no_lookahead_attention.touch((pinned_shared_prefix,))
    assert no_lookahead.apply_elastic_admission_wave((1,)) == 0
    assert no_lookahead.last_elastic_rejection is None
    assert not no_lookahead.ensure_elastic_capacity(
        KVCacheBlockPoolRequirements(primary=1, mamba=9)
    )
    assert no_lookahead.last_elastic_rejection is not None
    assert no_lookahead.last_elastic_rejection["reason"] == "attention_tail_pinned"
    no_lookahead_attention.free_blocks((pinned_shared_prefix,))
    no_lookahead_gdn.free_blocks(reversed(no_lookahead_gdn_leases))

    with_lookahead = make_manager().coordinator
    assert with_lookahead.apply_elastic_admission_wave((1,) * 10) == 10
    assert with_lookahead.block_pool.active_num_gpu_blocks == 46

    # Scheduler integration: ready, already-promoted and still-compiling
    # grammar requests count toward the same wave. Failed grammars and
    # unbounded remote/streaming waits do not.
    reserved_candidates = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting = FCFSRequestQueue(
        [
            SimpleNamespace(status=RequestStatus.WAITING),
            SimpleNamespace(status=RequestStatus.WAITING),
        ]
    )
    ready = SimpleNamespace(
        request_id="ready",
        status=RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        structured_output_request=SimpleNamespace(grammar=object()),
    )
    pending = SimpleNamespace(
        request_id="pending",
        status=RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        structured_output_request=SimpleNamespace(grammar=None),
    )
    promoted = SimpleNamespace(
        request_id="promoted",
        status=RequestStatus.WAITING,
        structured_output_request=SimpleNamespace(grammar=object()),
    )
    broken = SimpleNamespace(
        request_id="broken",
        status=RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        structured_output_request=SimpleNamespace(grammar=ValueError("broken")),
    )
    remote = SimpleNamespace(
        request_id="remote",
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
    )
    streaming = SimpleNamespace(
        request_id="streaming",
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
        streaming_queue=[],
    )
    scheduler.skipped_waiting = FCFSRequestQueue(
        [ready, pending, promoted, broken, remote, streaming]
    )
    scheduler.max_num_running_reqs = 8
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.grammar_compile_error_reqs = set()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.kv_cache_manager = SimpleNamespace(
        kv_cache_config=SimpleNamespace(elastic_mapping_quantum=1),
        watermark_blocks=0,
        estimate_uncached_full_sequence_requirements=lambda request: SimpleNamespace(
            primary=1
        ),
        coordinator=SimpleNamespace(
            apply_elastic_admission_wave=lambda requirements: (
                reserved_candidates.append(requirements) or len(requirements)
            )
        ),
    )
    assert scheduler._apply_elastic_waiting_wave(token_budget=8) == 6
    assert reserved_candidates == [(1, 1, 1, 1, 1, 1)]
    assert ready.status == RequestStatus.WAITING
    assert pending.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert broken.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert remote.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert streaming.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.grammar_compile_error_reqs == {"broken"}

    # Preserve one incremental admission after the visible wave has drained
    # from WAITING into RUNNING. This is the cross-iteration half of the same
    # contract: end-of-step rebalance must not expose an attention tail that a
    # later arrival immediately needs to return to GDN.
    reserved_candidates.clear()
    scheduler.running = [SimpleNamespace()]
    scheduler.max_num_running_reqs = 2
    assert scheduler._apply_elastic_incremental_headroom() == 1
    assert reserved_candidates == [(1,)]
    scheduler.running.append(SimpleNamespace())
    assert scheduler._apply_elastic_incremental_headroom() == 0

    def worker_config(blocks_per_request: int) -> KVCacheConfig:
        return KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=80,
                    shared_by=["attention"],
                    backing_id="elastic-attention-0",
                    mapping_quantum=4,
                    num_blocks=10,
                    logical_block_size=5,
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
            elastic_attention_stride=5,
            elastic_gdn_stride=5,
            elastic_mapping_quantum=4,
            elastic_gdn_initial_blocks=2,
            elastic_gdn_blocks_per_request=blocks_per_request,
            elastic_budget_bytes=64,
        )

    rank_safe = generate_scheduler_kv_cache_config([worker_config(3), worker_config(3)])
    assert rank_safe.elastic_gdn_blocks_per_request == 3
    try:
        generate_scheduler_kv_cache_config([worker_config(3), worker_config(4)])
    except ValueError as exc:
        assert "blocks per request differs across workers" in str(exc)
    else:
        raise AssertionError("rank-local GDN topology mismatch was accepted")

    # Dynamic CUDA Graph memory is a step-scoped loan from the attention tail,
    # not a startup reserve. A pinned tail rejects the loan atomically; after
    # release, the same bytes reduce MaxX and returning them restores baseline.
    external = make_manager().coordinator
    baseline_attention = external.block_pool.active_num_gpu_blocks
    pinned_tail = external.block_pool.blocks[baseline_attention - 1]
    external.block_pool.touch((pinned_tail,))
    graph_bytes = 256 * 1024 * 1024
    assert not external.set_elastic_external_memory(graph_bytes)
    assert external.elastic_external_memory_bytes == 0
    assert external.block_pool.active_num_gpu_blocks == baseline_attention
    external.block_pool.free_blocks((pinned_tail,))

    assert external.set_elastic_external_memory(graph_bytes)
    borrowed_attention = external.block_pool.active_num_gpu_blocks
    assert borrowed_attention < baseline_attention
    graph_maxx = external.plan_elastic_admission_wave((1,) * 16)
    assert graph_maxx is not None
    assert graph_maxx.max_requests <= 16
    assert external.set_elastic_external_memory(0)
    assert external.block_pool.active_num_gpu_blocks == baseline_attention
    assert external.elastic_external_memory_bytes == 0

    print(
        "PASS: high-tail x16 admitted; attention IDs remained in-range; "
        "single-request capacity reclaimed; unused plan unpublished; "
        "long-head request reduced rather than deadlocked the wave; "
        "scheduler covered pending grammars, excluded unbounded waits, and "
        "obeyed token budget; "
        "rank topology mismatch failed closed; incremental high-tail arrival "
        "remained shrink-safe; graph bytes borrowed a free KV tail, changed "
        "MaxX, and returned the X1 baseline"
    )


if __name__ == "__main__":
    main()
