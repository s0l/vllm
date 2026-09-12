# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host hints cannot authorize a resident GPU read or escape common voting."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.expert_offload_hot import resident_hint
from vllm.models.qwen4_exp.nvidia.expert_offload_plan import plan


def fixture():
    ids = np.arange(40, dtype=np.int32).reshape(4, 10)
    waves = plan(ids, 512, 32)
    bank = SimpleNamespace(
        source=SimpleNamespace(experts=512),
        tables=SimpleNamespace(pool_rows=64),
        host_hot={(13, e): 63 - e for e in range(40)},
    )
    return bank, waves


def test_complete_hint_and_missing_hint_preserve_logical_to_physical_mapping():
    bank, waves = fixture()
    selected, keys, rows, status = resident_hint(bank, 13, waves)
    np.testing.assert_array_equal(selected, np.arange(40))
    np.testing.assert_array_equal(keys, np.arange(40) + 13 * 512)
    np.testing.assert_array_equal(rows, 63 - np.arange(40))
    assert status == 1 and rows.dtype == np.int32
    del bank.host_hot[13, 3]
    _, _, rows, status = resident_hint(bank, 13, waves)
    assert status == 0 and rows[3] == -1


@pytest.mark.parametrize("bad", [-1, 64, 1 << 50, True, 1.5, "1", None])
def test_corrupt_hint_becomes_common_rejection_without_integer_overflow(bad):
    bank, waves = fixture()
    bank.host_hot[13, 3] = bad
    _, _, rows, status = resident_hint(bank, 13, waves)
    assert status == -1 and rows[3] == -1


@pytest.mark.parametrize("index", [0, 1, 2])
def test_native_boundary_rejects_strided_inputs_before_cuda(index):
    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import (
        NativeMappedBankConsumer,
    )

    bank = SimpleNamespace(
        state="READY",
        staging=32,
        source=SimpleNamespace(experts=512),
        consumer_views={
            "w13_weight": torch.empty(32, 128, 64, dtype=torch.uint8),
            "w2_weight": torch.empty(32, 128, 32, dtype=torch.uint8),
            "w13_weight_scale": torch.empty(32, 128, 8, dtype=torch.uint8),
            "w2_weight_scale": torch.empty(32, 128, 4, dtype=torch.uint8),
        },
    )
    consumer = NativeMappedBankConsumer(bank, torch.zeros(32, dtype=torch.int32))
    inputs = [torch.ones(2, 8), torch.ones(2, 3), torch.zeros(2, 3, dtype=torch.int32)]
    inputs[index] = inputs[index][:1].expand_as(inputs[index])
    assert not inputs[index].is_contiguous()
    with pytest.raises(ValueError, match="must be contiguous"):
        consumer(*inputs)


@pytest.mark.parametrize("peer", ["ready", "missing", "bad_owner", "mapping", "mode"])
def test_joint_vote_rejects_divergent_ownership_before_conditional_execution(
    monkeypatch, peer
):
    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import NativeBankCoordinator

    coordinator = NativeBankCoordinator.__new__(NativeBankCoordinator)
    coordinator.bank = SimpleNamespace(state="READY", generation=4)
    coordinator.group = SimpleNamespace(device_group="control")
    coordinator.ranks = 3
    coordinator.send = torch.empty(72, dtype=torch.int64)
    coordinator.recv = torch.empty(216, dtype=torch.int64)
    calls = []

    def gather(out, incoming, *, group):
        calls.append(group)
        peers = out.view(3, -1)
        peers.copy_(incoming.expand_as(peers))
        if peer == "missing":
            peers[1, 38] = 0
        elif peer == "bad_owner":
            peers[1, 38] = -1
        elif peer == "mapping":
            peers[1, -1] += 1
        elif peer == "mode":
            peers[1, 5] = 0

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather)
    proposal = SimpleNamespace(vote=[1, 64] + [7] * 32, validate=lambda status: None)
    ids = np.arange(10, dtype=np.int32)[None]
    weights = np.full((1, 10), 0.1, dtype=np.float32)
    if peer in ("ready", "missing"):
        assert coordinator.admit_routes(13, ids, weights, resident=proposal) == (
            peer == "ready"
        )
        assert coordinator.bank.state == "READY"
    else:
        with pytest.raises(RuntimeError, match="rank-inconsistent"):
            coordinator.admit_routes(13, ids, weights, resident=proposal)
        assert coordinator.bank.state == "POISONED"
    assert calls == ["control"]


def test_full_bank_recovery_does_not_trust_corrupt_reverse_map():
    from threading import RLock

    from vllm.models.qwen4_exp.nvidia import expert_offload_tables as gp
    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import NativeExpertBank

    bank = NativeExpertBank.__new__(NativeExpertBank)
    bank.lock, bank.state, bank.device = RLock(), "RESIZING", "cpu"
    bank.source = SimpleNamespace(experts=16)
    bank.tables = gp.allocate_global_tables("cpu", 16, [4, 4], 4)
    bank.tables.row_key[0] = -1
    bank.tables.row_key[1] = 1 << 28
    bank.tables.hot_phys[20] = 1 << 28
    bank.tables.error.fill_(1)
    bank.host_hot, bank.host_rows = {(0, 0): 0}, {0: (0, 0)}
    bank.backings = {}
    bank.targets = lambda rows: {}
    bank._views = lambda rows: {}
    bank._initialize_free_rows = lambda start, end: None
    bank.staging, bank.generation = 4, 5
    bank.finish_resize(8, invalidate=True)
    assert bank.state == "READY" and bank.generation == 6
    assert not bank.host_hot and not bank.host_rows
    assert (bank.tables.hot_phys == -1).all()
    assert (bank.tables.row_key == -1).all()
    assert torch.equal(bank.tables.cold_phys, torch.arange(16).repeat(2).int())
    assert not bank.tables.error.any()


def test_step_observer_is_bounded_and_file_failure_cannot_diverge_ranks(tmp_path):
    import json

    from vllm.models.qwen4_exp.nvidia.expert_offload_provider import (
        NativeExpertProvider,
    )

    provider = NativeExpertProvider.__new__(NativeExpertProvider)
    provider.bank = residency_fixture()
    provider.trace_layers, provider.trace_steps = [], 0
    provider.trace_limit, provider.trace_byte_limit = 64, 64 << 20
    provider.trace_bytes, provider.trace_before = 0, None
    provider.trace_path = str(tmp_path / "trace")
    provider.dummy = False
    for step in range(66):
        for layer in range(2):
            provider.last_step = dict(layer=layer, source_hits=step, hot_rows=16)
            provider._trace_last_step()
    records = [
        json.loads(line)
        for line in (tmp_path / "trace-rank1.jsonl").read_text().splitlines()
    ]
    assert len(records) == 64 and records[-1]["step"] == 63
    assert all(len(row["layers"]) == 2 for row in records)
    provider.trace_steps = 0
    provider.trace_path = str(tmp_path / "missing" / "trace")
    provider._trace_last_step()
    assert provider.trace_path == ""


def residency_fixture():
    from threading import RLock

    values = {(0, 1): {"w": np.ones(16, dtype=np.uint8)}}
    return SimpleNamespace(
        lock=RLock(),
        generation=7,
        state="READY",
        staging=2,
        tables=SimpleNamespace(pool_rows=4),
        host_hot={(0, 1): 0, (1, 3): 2},
        strides={"w": 16},
        leases=set(),
        backings={"w": SimpleNamespace(info=SimpleNamespace(committed=96))},
        source=SimpleNamespace(
            lock=RLock(),
            layers=2,
            experts=8,
            rank=1,
            cache=values,
            limit=256,
            used=16,
            pinned_pool=None,
            hits=3,
            misses=2,
            read_bytes=32,
        ),
    )


def test_residency_snapshot_covers_unused_owners_without_retaining_weights():
    import gc
    import weakref

    from vllm.models.qwen4_exp.nvidia.expert_offload_admission import cache_snapshot

    bank = residency_fixture()
    ref = weakref.ref(bank.source.cache[0, 1]["w"])
    snapshot = cache_snapshot(bank)
    assert snapshot["hot_keys"] == [(0, 1, 0), (1, 3, 2)]
    assert snapshot["ram_keys"] == [(0, 1)]
    assert snapshot["gpu_ram_duplicate_payload_bytes"] == 16
    assert snapshot["bank_mapped_bytes"] == 96
    bank.host_hot.clear()
    bank.source.cache.clear()
    gc.collect()
    assert ref() is None and len(snapshot["hot_keys"]) == 2


def test_large_prefill_histogram_matches_independent_counter_and_excludes_padding():
    from collections import Counter

    from vllm.models.qwen4_exp.nvidia.expert_offload_admission import route_histogram

    ids = (np.arange(2048 * 10).reshape(2048, 10) % 512).astype(np.int32)
    weights = np.ones(ids.shape, np.float32)
    ids[1551:] = -1
    weights[1551:] = 0
    counts, tokens = route_histogram(ids, weights, 512)
    oracle = Counter(e for row in ids[:1551].tolist() for e in row)
    assert tokens == 1551 and counts.tolist() == [oracle[e] for e in range(512)]
    assert int(counts.sum()) == 15510
    ids[0, 0] = -1
    with pytest.raises(ValueError):
        route_histogram(ids, weights, 512)
    ids[0, 0] = 512
    with pytest.raises(ValueError):
        route_histogram(ids, weights, 512)
    ids[0, 0] = 0
    weights[0, 0] = np.nan
    with pytest.raises(ValueError):
        route_histogram(ids, weights, 512)


def test_frequency_admission_uses_complete_history_instead_of_last_layers():
    from vllm.models.qwen4_exp.nvidia.expert_offload_admission import (
        ExpertFrequencyAdmission,
    )

    policy = ExpertFrequencyAdmission(4, 16, history_steps=2)
    observed = np.ones((4, 16), np.uint32)
    observed[:, 0] = 8
    policy.observe(observed)
    observed.fill(0)  # Caller mutation must not change stored history.
    proposal = policy.plan(
        [52, 53, 54, 55], list(range(64)), capacity=4, max_promotions=4
    )
    assert proposal.desired == (0, 16, 32, 48)
    assert len(proposal.promotions) == len(proposal.evictions) == 4
    assert policy.validate(proposal, [52, 53, 54, 55]) is proposal
    for _ in range(2):
        changed = np.zeros((4, 16), np.uint32)
        changed[:, 15] = 9
        policy.observe(changed)
    changed = policy.plan(
        proposal.desired, [15, 31, 47, 63], capacity=4, max_promotions=4
    )
    assert changed.desired == (15, 31, 47, 63)
    with pytest.raises(ValueError, match="stale"):
        policy.validate(proposal, [52, 53, 54, 55])


def test_frequency_admission_preserves_ties_leases_and_explicit_copy_budget():
    from vllm.models.qwen4_exp.nvidia.expert_offload_admission import (
        ExpertFrequencyAdmission,
    )

    policy = ExpertFrequencyAdmission(1, 16, history_steps=2)
    counts = np.zeros((1, 16), np.uint32)
    counts[0, :8] = 1
    policy.observe(counts)
    current = (4, 5, 6, 7)
    assert (
        policy.plan(current, list(range(8)), capacity=4, max_promotions=8).promotions
        == ()
    )
    counts[0, :4] = 10
    policy.observe(counts)
    limited = policy.plan(
        current, [0, 1, 2, 3], capacity=4, max_promotions=1, protected=[4]
    )
    assert len(limited.promotions) == 1 and 4 in limited.desired
    assert (
        policy.plan(current, [0, 1, 2, 3], capacity=4, max_promotions=0).desired
        == current
    )
    with pytest.raises(ValueError, match="lease"):
        policy.plan(current, [], capacity=0, max_promotions=0, protected=[4])
    shrink = policy.plan(current, [], capacity=1, max_promotions=0, protected=[6])
    assert shrink.desired == (6,)
    # An available source row with no observed use is not a prediction.
    assert policy.plan([], [15], capacity=8, max_promotions=8).desired == ()
    grown = policy.plan([6], [0, 1, 2, 3], capacity=5, max_promotions=4)
    assert grown.desired == (0, 1, 2, 3, 6)
    with pytest.raises(ValueError, match="stale"):
        policy.validate(grown, [])


def test_histogram_observer_failure_cannot_interrupt_expert_execution(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import expert_offload_provider as module

    provider = module.NativeExpertProvider.__new__(module.NativeExpertProvider)
    provider.bank, provider.trace_path = residency_fixture(), "observer"

    def failed_snapshot(bank):
        raise OSError("observer unavailable")

    monkeypatch.setattr(module, "cache_snapshot", failed_snapshot)
    assert (
        provider._trace_routes(
            0, np.zeros((1, 10), np.int32), np.ones((1, 10), np.float32)
        )
        == {}
    )
    assert provider.trace_path == "" and provider.bank.state == "READY"


@pytest.mark.parametrize("m", [4, 65, 4096])
def test_resident_first_waves_preserve_each_assignment_and_padding(m):
    ids = ((np.arange(m)[:, None] * 13 + np.arange(10) * 37) % 512).astype(np.int32)
    padding = np.zeros(m, np.bool_)
    padding[-1] = True
    ids[-1] = -1
    resident = np.arange(256, 512, dtype=np.int32)
    waves = plan(ids, 512, 32, is_padding=padding, resident_experts=resident)
    received: list[tuple[int, int]] = []
    experts: list[int] = []
    for wave in waves:
        experts.extend(wave.experts)
        for tile in wave.tiles(256):
            received.extend(
                (int(lane), tile.experts[int(slot)])
                for lane, slot in zip(tile.lanes, tile.slots, strict=True)
            )
    expected = [(lane, int(e)) for lane, e in enumerate(ids.flat) if e >= 0]
    assert sorted(received) == expected
    assert len(experts) == len(set(experts))
    assert experts == sorted(experts, key=lambda e: (e not in resident, e))
    with pytest.raises(ValueError, match="resident expert"):
        plan(ids, 512, 32, is_padding=padding, resident_experts=np.array([512]))


def test_stream_scan_policy_is_explicit_and_rejects_truthy_strings():
    from vllm.utils.nvfp4_expert_stream import StreamExpertConfig

    options = dict(
        library="/test/cpu.so",
        sha256="a" * 64,
        cores=[0, 1, 2],
        max_m=4,
        history_steps=16,
        max_promotions=192,
    )
    assert StreamExpertConfig.from_options(options, 3).scan_order is False
    assert StreamExpertConfig.from_options(dict(options, scan_order=True), 3).scan_order
    with pytest.raises(ValueError, match="requires a bool"):
        StreamExpertConfig.from_options(dict(options, scan_order="false"), 3)
