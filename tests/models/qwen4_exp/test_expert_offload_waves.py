# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host control and lease admission; CUDA/math remain a separate gate."""

from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.expert_offload_bank import NativeExpertBank
from vllm.models.qwen4_exp.nvidia.expert_offload_provider import NativeExpertProvider
from vllm.models.qwen4_exp.nvidia.expert_offload_waves import NativeWaveExecutor
from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry
from vllm.v1.core.elastic_expert import NativeExpertBudget


def test_semantic_completion_includes_empty_and_failed_ranks_without_physical_ids():
    provider = NativeExpertProvider.__new__(NativeExpertProvider)
    provider.bank = SimpleNamespace(state="READY", generation=3)
    seen = []

    def exchange(packet):
        seen.append(tuple(packet))
        return np.tile(packet, (3, 1))

    provider.coordinator = SimpleNamespace(
        send=torch.empty(72), exchange_control=exchange
    )
    ids = np.empty((0, 10), np.int32)
    weights = np.empty((0, 10), np.float32)
    provider._complete_wave_layer(7, ids, weights)
    assert seen[0][:5] == (1, 3, 7, 0, 71)
    with pytest.raises(RuntimeError, match="semantic wave completion"):
        provider._complete_wave_layer(7, ids, weights, error=OSError("short read"))
    assert len(seen) == 2 and provider.bank.state == "POISONED"
    provider.bank.state = "READY"
    provider._complete_wave_layer(7, ids, weights)
    assert seen[-1] == seen[0]


def test_stale_rank_completion_is_not_accepted():
    provider = NativeExpertProvider.__new__(NativeExpertProvider)
    provider.bank = SimpleNamespace(state="READY", generation=3)

    def exchange(packet):
        peers = np.tile(packet, (3, 1))
        peers[2, 1] -= 1
        return peers

    provider.coordinator = SimpleNamespace(
        send=torch.empty(72), exchange_control=exchange
    )
    with pytest.raises(RuntimeError, match="semantic wave completion"):
        provider._complete_wave_layer(
            1, np.zeros((1, 1), np.int32), np.ones((1, 1), np.float32)
        )


def test_wave_slot_keeps_source_until_every_reader_drains():
    executor = NativeWaveExecutor.__new__(NativeWaveExecutor)
    events = []
    slot = SimpleNamespace(
        rows=[object()],
        done=SimpleNamespace(synchronize=lambda: events.append("compute")),
        copied=SimpleNamespace(synchronize=lambda: events.append("copy")),
    )
    executor.retire_slot(slot)
    assert events == ["compute", "copy"] and slot.rows == []
    executor.retire_slot(slot)
    assert events == ["compute", "copy"]


def test_failed_drain_does_not_free_source_or_report_a_reusable_slot():
    executor = NativeWaveExecutor.__new__(NativeWaveExecutor)

    def failure():
        raise RuntimeError("device failure")

    slot = SimpleNamespace(
        rows=[object()], done=SimpleNamespace(synchronize=failure), copied=None
    )
    with pytest.raises(RuntimeError):
        executor.retire_slot(slot)
    assert slot.rows and slot.done is not None
    slot.done.synchronize = lambda: None
    executor.retire_slot(slot)
    assert not slot.rows


@pytest.mark.parametrize("failed_capture", [False, True])
def test_every_slot_capture_holds_real_bank_lease_and_recovers(failed_capture):
    class Buffer:
        def fill_(self, value):
            pass

        def zero_(self):
            pass

    executor = NativeWaveExecutor.__new__(NativeWaveExecutor)
    bank = executor.bank = NativeExpertBank.__new__(NativeExpertBank)
    bank.state, bank.lock, bank.leases = "READY", RLock(), set()
    bank.graphs, bank.stable_graphs = set(), set()
    drains = []
    bank.fence = lambda: SimpleNamespace(synchronize=lambda: drains.append(True))
    executor.slots = [
        SimpleNamespace(
            kernels={},
            lane_ids=Buffer(),
            local_ids=Buffer(),
            offset=Buffer(),
            group_rows=Buffer(),
            rows=[],
            done=None,
            copied=None,
        )
        for _ in range(2)
    ]

    def capture(slot, bucket):
        bank.register_graph(object(), stable_views=True)
        if failed_capture:
            raise RuntimeError("injected capture failure")
        slot.kernels.setdefault(bucket, True)

    executor.kernel = capture
    if failed_capture:
        with pytest.raises(RuntimeError, match="injected capture"):
            executor.prepare_kernels(iter([4, 64, 4]))
        assert not bank.leases and drains == [True]
        failed_capture = False
    executor.prepare_kernels(iter([4, 64, 4]))
    assert [set(slot.kernels) for slot in executor.slots] == [{4, 64}, {4, 64}]
    assert not bank.leases
    completed_drains = len(drains)
    executor.prepare_kernels([4, 64])
    assert len(drains) == completed_drains


def test_double_staging_is_charged_before_elastic_hot_grants():
    geometry = NVFP4ExpertGeometry(2560, 640, 3)
    one = NativeExpertBudget(geometry, 48, 512, staging=32)
    two = NativeExpertBudget(geometry, 48, 512, staging=32, wave_slots=2)
    assert two.scratch_rows == 64
    for rank in range(3):
        for hot in (0, 2169, 3280):
            assert two.rank_mapped_bytes(rank, hot) >= one.rank_mapped_bytes(rank, hot)
        assert two.rank_mapped_bytes(rank, 0) > one.rank_mapped_bytes(rank, 0)
    for slots in (0, 3, True):
        with pytest.raises(ValueError):
            NativeExpertBudget(geometry, 48, 512, wave_slots=slots)


def test_source_too_small_for_two_slots_is_rejected_before_cuda_allocation():
    cpu = SimpleNamespace(ready_enabled=True, source_stats=lambda: {"capacity": 64})
    provider = SimpleNamespace(
        bank=SimpleNamespace(staging=64, source=SimpleNamespace(cpu=cpu))
    )
    with pytest.raises(ValueError, match="source capacity"):
        NativeWaveExecutor(provider, 2)


# Whole jobs may move between CPU and GPU, never individual contributions.
def test_mixed_partition_preserves_original_lanes_and_rebuilds_slots():
    import numpy as np

    from vllm.models.qwen4_exp.nvidia.expert_offload_dispatch import (
        gpu_waves,
        grouped_jobs,
    )
    from vllm.models.qwen4_exp.nvidia.expert_offload_plan import plan

    for m in (4, 8, 32, 64, 65):
        ids = (np.arange(m * 10).reshape(m, 10) * 37 % 512).astype(np.int32)
        jobs = grouped_jobs(plan(ids, 512, 32))
        selected = set(sorted(jobs)[::3])
        waves = gpu_waves(jobs, selected, 32)
        cpu_lanes = {int(lane) for key in selected for lane in jobs[key]}
        observed = []
        for wave in waves:
            assert len(wave.experts) <= 32
            for lane, slot in zip(wave.lanes, wave.slots, strict=True):
                assert ids.ravel()[lane] == wave.experts[slot]
                observed.append(int(lane))
        assert len(observed) == len(set(observed))
        assert not cpu_lanes & set(observed)
        assert cpu_lanes | set(observed) == set(range(m * 10))
