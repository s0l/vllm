# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.utils import KVBlockZeroer, get_kv_caches_for_block_copy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        page_size_el,
        page_size_el,
        1,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


def test_block_copy_excludes_separate_mamba_pool():
    attention = torch.zeros(1)
    mamba = torch.zeros(1)
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((16,),),
        dtypes=(torch.float16,),
        separate_pool=True,
        separate_pool_num_blocks=4,
    )
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], attention_spec),
            KVCacheGroupSpec(["gdn"], mamba_spec),
        ],
    )

    selected = get_kv_caches_for_block_copy(
        [attention, mamba],
        {"attention": attention, "gdn": mamba},
        config,
    )

    assert selected == [attention]


def test_block_copy_preserves_shared_pool_behavior():
    attention = torch.zeros(1)
    mamba = torch.zeros(1)
    runner_caches = [attention, mamba]
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((16,),),
        dtypes=(torch.float16,),
        separate_pool=False,
    )
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["gdn"], mamba_spec)],
    )

    selected = get_kv_caches_for_block_copy(
        runner_caches,
        {"gdn": mamba},
        config,
    )

    assert selected is runner_caches
