# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifetime and accounting controls; GPU DMA has a separate native oracle."""

import gc

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia import expert_offload_pinned as module


@pytest.fixture
def registrations(monkeypatch):
    registered, removed = [], []
    monkeypatch.setattr(
        module, "_register", lambda tensor, device: registered.append(tensor.data_ptr())
    )
    monkeypatch.setattr(
        module, "_unregister", lambda tensor, device: removed.append(tensor.data_ptr())
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    yield registered, removed
    gc.collect()
    module.collect_retired()
    assert sorted(registered) == sorted(removed)


def sample(value=1):
    return {
        "weight": np.full((6000,), value, dtype=np.uint8),
        "scale": np.asarray(value, dtype=np.float32),
    }


def test_page_budget_base_views_hold_slots_and_close_defers_cuda(registrations):
    registered, removed = registrations
    pool = module.PinnedExpertPool(sample(), 12288, device=0, slab_rows=2)
    assert pool.capacity == 2 and not pool.slabs and pool.allocated == 0
    first, second = pool.retain(sample(1)), pool.retain(sample(2))
    assert len(registered) == 1 and pool.allocated == pool.limit
    assert registered[0] % module.mmap.PAGESIZE == 0
    assert all(not value.flags.writeable for value in first.values())
    with pytest.raises(ValueError):
        first["weight"].setflags(write=True)
    # np.asarray/view/reshape must not strip the lease, unlike subclass attrs.
    held = np.asarray(first["weight"]).view().reshape(2, -1)[1, :]
    held_bytes = np.frombuffer(second["weight"], dtype=np.uint8)
    del first, second
    gc.collect()
    assert pool.retain(sample(3)) is None
    assert np.all(held == 1) and np.all(held_bytes == 2)
    del held_bytes
    third = pool.retain(sample(3))
    assert third is not None and np.all(third["weight"] == 3)
    assert np.all(held == 1)
    pool.close()
    assert not removed and pool.allocated == pool.limit
    with pytest.raises(RuntimeError, match="closed"):
        pool.retain(sample())
    del held, third
    gc.collect()
    assert not removed  # Finalizers return slots but never call CUDA.
    module.collect_retired()
    assert pool.allocated == 0 and len(removed) == 1
    pool.close()  # Idempotent unregister.


@pytest.mark.parametrize("budget", [0, 6003, 6004, 8191, 8192, 12287, 12288, 45057])
def test_registration_and_retention_respect_rounded_budget(budget, registrations):
    pool = module.PinnedExpertPool(sample(), budget, device=0, slab_rows=2)
    rows = [pool.retain(sample()) for _ in range(pool.capacity + 1)]
    assert rows[-1] is None
    assert all(row is not None for row in rows[:-1])
    assert pool.allocated <= budget and pool.peak_allocated <= budget
    assert pool.allocated == sum(size for _, size in pool.planned)
    del rows
    pool.close()
    assert pool.allocated == 0


def test_descriptor_rejects_replaced_or_foreign_views(registrations):
    pool = module.PinnedExpertPool(sample(), 8192, device=0)
    other = module.PinnedExpertPool(sample(), 8192, device=0)
    row = pool.retain(sample())
    assert row.descriptor(pool) == (pool.slabs[0], 0)
    with pytest.raises(RuntimeError, match="foreign"):
        row.descriptor(other)
    row["weight"] = row["weight"].copy()
    with pytest.raises(RuntimeError, match="no longer owns"):
        row.descriptor(pool)
    del row
    pool.close()
    other.close()


def test_failed_registration_and_unregister_are_recoverable(registrations, monkeypatch):
    pool = module.PinnedExpertPool(sample(), 16384, device=0, slab_rows=1)
    first = pool.retain(sample())
    with monkeypatch.context() as patch:

        def fail(*args):
            raise RuntimeError("injected host registration error")

        patch.setattr(module, "_register", fail)
        with pytest.raises(RuntimeError, match="injected"):
            pool.retain(sample())
    assert pool.allocated == 8192 and len(pool.slabs) == 1
    del first
    with monkeypatch.context() as patch:
        patch.setattr(module, "_unregister", fail)
        with pytest.raises(RuntimeError, match="injected"):
            pool.close()
    assert pool.allocated == 8192 and len(pool.slabs) == 1
    pool.close()
    assert pool.allocated == 0


def test_source_pinned_cache_bypasses_held_rows_without_duplicate_retention(
    tmp_path, registrations
):
    from .test_expert_offload_source import store, write_source

    write_source(tmp_path)
    control = store(tmp_path, budget=0)
    source = store(tmp_path, budget=8192, cache_policy="lru", pin_cache=True)
    expected = control.get_many(0, (0, 1, 2)) + control.get_many(1, (0, 1, 2))
    actual = source.get_many(0, (0, 1, 2)) + source.get_many(1, (0, 1, 2))
    assert any(isinstance(row, module.PinnedExpertBundle) for row in actual)
    assert any(not isinstance(row, module.PinnedExpertBundle) for row in actual)
    assert source.pinned_pool.bypasses > 0 and source.cache_bypasses > 0
    assert all(
        isinstance(row, module.PinnedExpertBundle) for row in source.cache.values()
    )
    assert source.used <= source.pinned_pool.capacity * source.pinned_pool.row_bytes
    assert source.pinned_pool.allocated <= source.limit
    for first, second in zip(actual, expected):
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
    del first, second
    source.close()
    assert source.used == 0
    assert np.array_equal(actual[0]["w13_weight"], expected[0]["w13_weight"])
    del actual
    source.close()
    assert source.pinned_pool.allocated == 0
    control.close()


def test_pinned_source_epoch_failure_preserves_readers_then_fresh_source_recovers(
    tmp_path, registrations
):
    from .test_expert_offload_source import store, write_source

    write_source(tmp_path)
    source = store(tmp_path, budget=8192, pin_cache=True)
    before = source.get(0, 0)
    assert before["w2_weight_scale_2"] == 1
    write_source(tmp_path, scale=2.0)
    with pytest.raises(OSError, match="identity changed"):
        source.get(0, 0)
    assert source.closed and source.used == 0
    assert before["w2_weight_scale_2"] == 1
    del before
    source.close()
    recovered = store(tmp_path, budget=8192, pin_cache=True)
    assert recovered.get(0, 0)["w2_weight_scale_2"] == 4
    recovered.close()


def test_pinned_cache_disallows_registration_from_async_prefetch(
    tmp_path, registrations
):
    from .test_expert_offload_source import _prefetch_bank, store, write_source

    write_source(tmp_path)
    source = store(tmp_path, budget=8192, pin_cache=True)
    bank = _prefetch_bank(source)
    with pytest.raises(RuntimeError, match="excludes source prefetch"):
        bank.prefetch(0, (1,))
    assert bank.prefetch_reader is None
    source.close()


def test_failed_copy_traceback_releases_its_dma_rows_after_fence(
    tmp_path, registrations
):
    from threading import RLock
    from unittest.mock import Mock

    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import (
        NativeBankPlan,
        NativeExpertBank,
    )

    from .test_expert_offload_source import store, write_source

    write_source(tmp_path)
    source = store(tmp_path, budget=8192, pin_cache=True)
    source.get(0, 1)
    bank = NativeExpertBank.__new__(NativeExpertBank)
    bank.source, bank.lock, bank.state = source, RLock(), "LOADING"
    bank.plan = NativeBankPlan(1, 0, (0,), ((1, 0),))
    bank.completed, bank.prepared = set(), {}
    bank.staging, bank.prefetch_consumed = 2, 0
    bank.pinned_numpy = {}
    bank.fence = Mock(return_value=Mock())

    def fail(groups):
        raise RuntimeError("injected copy error")

    bank._copy_registered = fail
    with pytest.raises(RuntimeError, match="injected") as caught:
        bank.copy(bank.plan)
    assert bank.state == "POISONED"
    bank.fence.return_value.synchronize.assert_called_once_with()
    source.close()
    # Keep caught's exception/traceback alive while checking actual readers.
    assert caught.value.__traceback__ is not None
    assert source.pinned_pool.allocated == 0


@pytest.mark.parametrize("handle", [0, 456])
@pytest.mark.parametrize("failure", [False, True])
def test_batch_dma_bridges_null_stream_even_after_partial_failure(
    monkeypatch, handle, failure
):
    from collections import namedtuple
    from types import SimpleNamespace
    from weakref import WeakKeyDictionary

    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import (
        COMPONENTS,
        NativeExpertBank,
    )

    calls = []
    current = SimpleNamespace(
        cuda_stream=handle,
        wait_stream=lambda other: calls.append(("consumer_wait", other.cuda_stream)),
    )
    dma = SimpleNamespace(
        cuda_stream=123,
        wait_stream=lambda other: calls.append(("copy_wait", other.cuda_stream)),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: dma)
    Params = namedtuple("Params", ["stream_handle"])
    monkeypatch.setattr(
        module.cuda_mem_ops,
        "build_params",
        lambda src, dst, stream: Params(stream.cuda_stream),
    )

    def copy(src, dst, parameters):
        calls.append(("dma", parameters.stream_handle))
        if failure:
            raise RuntimeError("injected partial DMA")

    monkeypatch.setattr(module.cuda_mem_ops, "copy_blocks", copy)

    class Slab:
        views = dict.fromkeys(COMPONENTS)

    bank = NativeExpertBank.__new__(NativeExpertBank)
    bank.device, bank.rows, bank.dma_stream = "cuda:0", 2, None
    bank.pinned_params = WeakKeyDictionary()
    bank.consumer_views, bank.strides = (
        dict.fromkeys(COMPONENTS),
        dict.fromkeys(COMPONENTS, 1),
    )
    bank.copy_calls = bank.copy_bytes = 0
    bank.completed = set()
    if failure:
        with pytest.raises(RuntimeError, match="partial DMA"):
            bank._copy_registered({Slab(): ([0], [0])})
        assert bank.copy_bytes == 0 and not bank.completed
    else:
        bank._copy_registered({Slab(): ([0], [0])})
        assert bank.copy_bytes == len(COMPONENTS)
        assert bank.completed == {(0, name) for name in COMPONENTS}
    assert calls == (
        [("copy_wait", 0), ("dma", 123), ("consumer_wait", 123)]
        if handle == 0
        else [("dma", 456)]
    )
