# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace

import pytest

from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


def test_mamba_align_chunk_split_uses_dcp_effective_block_size():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=1000,
        num_tokens=1000,
        shared_prefix_boundary=0,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        block_size=48,
        dcp_world_size=3,
        max_num_scheduled_tokens=128,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
        mamba_partial_cache_hit=False,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=128,
    )

    assert adjusted == 96
    effective_block_size = scheduler.cache_config.block_size * scheduler.dcp_world_size
    assert adjusted % effective_block_size == 0


def test_mamba_align_split_keeps_small_chunks_when_dcp_alignment_exceeds_budget():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=835,
        num_tokens=835,
        shared_prefix_boundary=0,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        block_size=624,
        dcp_world_size=3,
        max_num_scheduled_tokens=256,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=False,
        mamba_partial_cache_hit=False,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=256,
    )

    assert adjusted == 256


@pytest.mark.parametrize(
    ("num_new_tokens", "expected"),
    [(86, 64), (1488, 1488)],
)
def test_gdn_prefill_split_uses_backend_state_alignment(
    num_new_tokens: int, expected: int
):
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=1488,
        num_tokens=1488,
        shared_prefix_boundary=0,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2352),
        block_size=2352,
        max_num_scheduled_tokens=7056,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        hash_block_size=48,
        mamba_partial_cache_hit=False,
        mamba_state_update_alignment=64,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=num_new_tokens,
    )

    assert adjusted == expected


def test_gdn_prefill_alignment_keeps_natural_prompt_tail():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=1472,
        num_prompt_tokens=1513,
        num_tokens=1513,
        shared_prefix_boundary=0,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2352),
        block_size=2352,
        max_num_scheduled_tokens=7056,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        hash_block_size=48,
        mamba_partial_cache_hit=False,
        mamba_state_update_alignment=64,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=41,
    )

    assert adjusted == 41


def test_gdn_prefill_alignment_resumes_from_aligned_prefix():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=192,
        num_prompt_tokens=1000,
        num_tokens=1000,
        shared_prefix_boundary=0,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2496),
        block_size=2496,
        max_num_scheduled_tokens=7056,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        hash_block_size=192,
        mamba_partial_cache_hit=True,
        mamba_state_update_alignment=64,
    )

    adjusted = Scheduler._mamba_block_aligned_split(
        self=scheduler,
        request=request,
        num_new_tokens=86,
    )

    assert adjusted == 64


def test_gdn_prefill_alignment_rejects_unaligned_resume():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=48,
        num_prompt_tokens=1000,
        num_tokens=1000,
        shared_prefix_boundary=0,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2496),
        block_size=2496,
        max_num_scheduled_tokens=7056,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        hash_block_size=192,
        mamba_partial_cache_hit=True,
        mamba_state_update_alignment=64,
    )

    with pytest.raises(RuntimeError, match="unaligned state boundary"):
        Scheduler._mamba_block_aligned_split(
            self=scheduler,
            request=request,
            num_new_tokens=86,
        )


def test_gdn_prefill_alignment_does_not_change_decode():
    pytest.importorskip("torch")
    from vllm.v1.core.sched.scheduler import Scheduler

    request = SimpleNamespace(
        num_computed_tokens=1000,
        num_prompt_tokens=1000,
        num_tokens=1004,
        shared_prefix_boundary=0,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2496),
        block_size=2496,
        max_num_scheduled_tokens=7056,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True,
        hash_block_size=192,
        mamba_partial_cache_hit=True,
        mamba_state_update_alignment=64,
    )

    assert (
        Scheduler._mamba_block_aligned_split(
            self=scheduler,
            request=request,
            num_new_tokens=4,
        )
        == 4
    )


def test_gdn_async_x20_prefill_alignment_makes_progress():
    """Exercise the complete scheduler lifecycle that the helper POC missed.

    Twenty 1488-token prompts share the production 7056-token budget with
    async scheduling and K3 placeholders. Every artificial split must be on
    the GDN grid, natural prompt tails may be arbitrary, and all requests must
    reach terminal output without a zero-progress scheduler loop.
    """
    torch = pytest.importorskip("torch")
    block_size = 2496
    scheduler = create_scheduler(
        max_num_seqs=40,
        max_num_batched_tokens=7056,
        max_model_len=8192,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        block_size=block_size,
        num_blocks=10000,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
        async_scheduling=True,
        kv_cache_spec=MambaSpec(
            block_size=block_size,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            num_speculative_blocks=3,
            state_update_chunk_alignment=64,
        ),
    )
    scheduler.use_eagle = True
    # The generic test utility does not propagate the synthetic MambaSpec's
    # cache mode into CacheConfig. The real runtime does; activate the resolved
    # contract explicitly so this control exercises the intended scheduler path.
    scheduler.need_mamba_block_aligned_split = True
    scheduler.mamba_state_update_alignment = 64
    requests = create_requests(
        num_requests=20,
        num_tokens=1488,
        max_tokens=1,
        req_ids=[f"r{i}" for i in range(20)],
        block_size=block_size,
    )
    for request in requests:
        scheduler.add_request(request)

    inflight: deque[tuple[object, ModelRunnerOutput]] = deque()
    artificial_splits = 0

    def schedule_one() -> None:
        nonlocal artificial_splits
        output = scheduler.schedule()
        if not output.num_scheduled_tokens:
            return
        sampled_token_ids: list[list[int]] = []
        req_ids = list(output.num_scheduled_tokens)
        for req_id in req_ids:
            request = scheduler.requests[req_id]
            count = output.num_scheduled_tokens[req_id]
            end = request.num_computed_tokens
            start = end - count
            if end < request.num_prompt_tokens:
                artificial_splits += 1
                assert start % 64 == 0
                assert end % 64 == 0
                sampled_token_ids.append([])
            else:
                sampled_token_ids.append([50256])
        inflight.append(
            (
                output,
                ModelRunnerOutput(
                    req_ids=req_ids,
                    req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
                    sampled_token_ids=sampled_token_ids,
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
        )

    # Match the production async envelope: schedule two steps before the first
    # output is consumed, then keep a two-step FIFO in flight.
    schedule_one()
    schedule_one()
    for _ in range(256):
        if inflight:
            output, model_output = inflight.popleft()
            scheduler.update_from_output(output, model_output)
        if scheduler.get_num_unfinished_requests() == 0:
            break
        schedule_one()
        if not inflight:
            pytest.fail("scheduler made no progress with unfinished requests")
    else:
        pytest.fail("scheduler did not terminate within 256 transitions")

    assert artificial_splits > 0
    assert scheduler.get_num_unfinished_requests() == 0


def test_hybrid_dcp_resolves_global_scheduler_block_and_fine_hash_block():
    torch = pytest.importorskip("torch")
    block_size = 16
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
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
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=True,
            hash_block_size=None,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=3,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )

    assert scheduler_block_size == block_size * 3
    assert hash_block_size == block_size


def test_separate_gdn_pool_default_geometry_includes_state_alignment():
    from vllm.platforms.interface import Platform

    vllm_config = SimpleNamespace(
        additional_config={"gdn_separate_pool": True},
        cache_config=SimpleNamespace(
            block_size=16,
            mamba_block_size=None,
            prefix_match_unit=192,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=3),
    )

    Platform._align_hybrid_block_size(vllm_config, None)

    assert vllm_config.cache_config.block_size == 2496
    assert vllm_config.cache_config.mamba_block_size == 2496


@pytest.mark.parametrize(
    "additional_config,prefix_match_unit",
    [
        ({"gdn_separate_pool": True}, 48),
        (
            {
                "gdn_separate_pool": True,
                "gdn_attention_block_size": 2352,
            },
            192,
        ),
    ],
)
def test_separate_gdn_pool_rejects_incoherent_geometry(
    additional_config: dict[str, object], prefix_match_unit: int
):
    from vllm.platforms.interface import Platform

    vllm_config = SimpleNamespace(
        additional_config=additional_config,
        cache_config=SimpleNamespace(
            block_size=16,
            mamba_block_size=None,
            prefix_match_unit=prefix_match_unit,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=3),
    )

    with pytest.raises(ValueError):
        Platform._align_hybrid_block_size(vllm_config, None)
