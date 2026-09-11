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
        consumer_views={},
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
    provider.bank = SimpleNamespace(source=SimpleNamespace(layers=2, rank=1))
    provider.trace_layers, provider.trace_steps = [], 0
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
