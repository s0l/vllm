# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-layer residency, free physical rows and resize/refill contracts."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import expert_offload_tables as gp


def fresh(rows=3):
    tables = gp.allocate_global_tables("cpu", 8, [0, 0, 0], 4)
    gp.resize_tables(tables, rows)
    gp.set_gate(tables, True)
    return tables, gp.allocate_step_buffers("cpu", 8, 4)


def run(tables, buffers, layer, ids):
    gp.step(tables, layer, torch.tensor(ids, dtype=torch.int32), buffers)
    gp.check_global_tables(tables)
    n = int(buffers.gather_count[0])
    return list(zip(buffers.gather_src[:n].tolist(), buffers.gather_dst[:n].tolist()))


def test_global_hits_lru_and_no_victim_staging():
    tables, buffers = fresh()
    assert run(tables, buffers, 0, [2, 3]) == [(2, 0), (3, 1)]
    assert run(tables, buffers, 1, [4]) == [(4, 2)]
    assert run(tables, buffers, 0, [2, 2, -1]) == []
    assert buffers.routes.tolist() == [0, 0, -1, -1]
    # The global victim belongs to layer0; there is no per-layer fixed quota.
    assert run(tables, buffers, 2, [5]) == [(5, 1)]
    assert int(tables.hot_phys[3]) == -1
    gp.set_control(tables, protect_recent=10)
    assert run(tables, buffers, 0, [7]) == [(7, 3)]
    assert int(tables.hot_phys[7]) == -1
    assert int(buffers.step_map[7]) == 3


def test_shrink_grow_invalidates_old_rows_and_refills_free_rows():
    tables, buffers = fresh()
    run(tables, buffers, 0, [0, 1, 2])
    assert gp.resize_tables(tables, 1) == (1, 2)
    gp.check_global_tables(tables)
    assert run(tables, buffers, 0, [0]) == []
    gp.set_control(tables, protect_recent=100)
    gp.resize_tables(tables, 3)
    assert tables.row_key.tolist() == [0, -1, -1, -1, -1, -1, -1]
    # Old row addresses returning never imply old expert bytes are valid.
    assert run(tables, buffers, 0, [1, 2]) == [(1, 1), (2, 2)]
    assert gp.resize_tables(tables, 0) == (0, 1, 2)
    assert run(tables, buffers, 1, [2, 3]) == [(2, 0), (3, 1)]
    assert not bool((tables.hot_phys >= 0).any())


def test_freeze_and_deferred_promotion_keep_every_expert_computable():
    tables, buffers = fresh()
    gp.set_gate(tables, False)
    assert run(tables, buffers, 0, [1, 2, 1]) == [(1, 3), (2, 4)]
    assert buffers.routes.tolist() == [3, 4, 3, -1]
    gp.set_control(tables, gate=1, promote_limit=1, promote_min_misses=2)
    assert run(tables, buffers, 0, [1, 2]) == [(1, 3), (2, 4)]
    assert run(tables, buffers, 0, [1, 2]) == [(1, 0), (2, 3)]
    assert run(tables, buffers, 0, [1, 2]) == [(2, 1)]
    assert run(tables, buffers, 0, [1, 2]) == []


def test_invalid_inputs_fail_before_mutation_or_poison_the_epoch():
    tables, buffers = fresh()
    before = tables.hot_phys.clone()
    for layer, ids in [(3, torch.tensor([0])), (0, torch.tensor([0.5]))]:
        with pytest.raises(ValueError):
            gp.step(tables, layer, ids, buffers)
        assert torch.equal(before, tables.hot_phys)
    with pytest.raises(ValueError, match="plan width"):
        run(tables, buffers, 0, [0, 1, 2, 3, 4])
    gp.step(tables, 0, torch.tensor([-2, 8]), buffers)
    with pytest.raises(RuntimeError, match="device error"):
        gp.check_global_tables(tables)
    assert not bool((buffers.routes >= 0).any())
    recovered, scratch = fresh()
    assert run(recovered, scratch, 0, [2]) == [(2, 0)]


@pytest.mark.parametrize("tokens", [0, 1, 4, 32, 4096])
@pytest.mark.parametrize("capacity", [1, 2, 32])
def test_compact_provider_conserves_weighted_lanes_and_last_tile(tokens, capacity):
    import numpy as np

    from vllm.models.qwen4_exp.nvidia.expert_offload_plan import plan

    rng = np.random.default_rng(1907)
    ids = np.argsort(rng.random((tokens, 512)), axis=1)[:, :10].astype(np.int32)
    weights = rng.random((tokens, 10))
    actual = np.full(ids.size, np.nan)
    seen = np.zeros(ids.size, dtype=np.int32)
    for wave in plan(ids, 512, capacity):
        for tile in wave.tiles(4096):
            assert len(tile.experts) <= capacity
            assert len(tile.lanes) <= tile.bucket <= 4096
            experts = np.asarray(tile.experts)[tile.slots]
            assert np.array_equal(experts, ids.flat[tile.lanes])
            seen[tile.lanes] += 1
            actual[tile.lanes] = (experts + 1) * weights.flat[tile.lanes]
    assert np.all(seen == 1)
    assert np.array_equal(actual.reshape(ids.shape), (ids + 1) * weights)


def test_compact_provider_rejects_invalid_routes_and_recovers():
    import numpy as np

    from vllm.models.qwen4_exp.nvidia.expert_offload_plan import plan

    valid = np.arange(10, dtype=np.int32)[None]
    for invalid in (valid - 1, valid + 512, valid.astype(float), valid[0], valid * 0):
        with pytest.raises(ValueError):
            plan(invalid, 512, 32)
        assert len(plan(valid, 512, 32)) == 1
    wave = plan(np.tile(valid, (819, 1)), 512, 32)[0]
    assert [len(t.lanes) for t in wave.tiles(4096)] == [4096, 4094]


@pytest.mark.parametrize("tokens", [0, 1, 4, 33])
@pytest.mark.parametrize("mask_kind", ["none", "tail", "interior", "all"])
def test_compact_provider_conserves_only_explicit_live_lanes(tokens, mask_kind):
    import numpy as np

    from vllm.models.qwen4_exp.nvidia.expert_offload_plan import plan

    rng = np.random.default_rng(6721)
    ids = np.argsort(rng.random((tokens, 512)), axis=1)[:, :10].astype(np.int32)
    padding = np.zeros(tokens, dtype=np.bool_)
    if mask_kind == "tail":
        padding[-1:] = True
    elif mask_kind == "interior":
        padding[::3] = True
    elif mask_kind == "all":
        padding[:] = True
    ids[padding] = -1
    seen = np.zeros(ids.size, dtype=np.int32)
    for wave in plan(ids, 512, 7, is_padding=padding):
        for tile in wave.tiles(16):
            experts = np.asarray(tile.experts)[tile.slots]
            assert np.array_equal(experts, ids.flat[tile.lanes])
            seen[tile.lanes] += 1
    assert np.array_equal(
        seen.reshape(ids.shape), np.broadcast_to(~padding[:, None], ids.shape)
    )


def test_compact_padding_does_not_hide_live_corruption_and_recovers():
    import numpy as np

    from vllm.models.qwen4_exp.nvidia.expert_offload_plan import plan

    valid = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    padding = np.array([False, True, False, True])
    valid[padding] = -1
    for bad_mask in (padding.astype(np.int32), padding[:, None], padding[:-1]):
        with pytest.raises(ValueError, match="padding mask"):
            plan(valid, 512, 32, is_padding=bad_mask)
    for row, value in ((0, -1), (0, 512), (0, 1), (1, 0)):
        bad = valid.copy()
        bad[row, 0] = value
        with pytest.raises(ValueError):
            plan(bad, 512, 32, is_padding=padding)
        assert len(plan(valid, 512, 32, is_padding=padding)) == 1
    with pytest.raises(ValueError, match="expert outside"):
        plan(valid, 512, 32)


@pytest.mark.parametrize("failure", [None, "peer", "local"])
def test_provider_votes_before_variable_wave_schedule(monkeypatch, failure):
    from types import SimpleNamespace

    import numpy as np

    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import NativeBankCoordinator

    coordinator = NativeBankCoordinator.__new__(NativeBankCoordinator)
    coordinator.bank = SimpleNamespace(state="READY", generation=4)
    coordinator.group = SimpleNamespace(device_group="control")
    coordinator.ranks = 3
    coordinator.send = torch.empty(64, dtype=torch.int64)
    coordinator.recv = torch.empty(192, dtype=torch.int64)

    def gather(out, incoming, *, group):
        assert group == "control"
        peers = out.view(3, -1)
        peers.copy_(incoming.expand_as(peers))
        if failure == "peer":
            peers[1, 8] += 1

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather)
    ids = np.arange(10, dtype=np.int32)[None]
    weights = np.full((1, 10), 0.1, dtype=np.float32)
    error = ValueError("invalid shape") if failure == "local" else None
    if failure is None:
        coordinator.admit_routes(3, ids, weights)
        assert coordinator.bank.state == "READY"
    else:
        with pytest.raises(RuntimeError, match="rank-inconsistent native routing"):
            coordinator.admit_routes(3, ids, weights, error=error)
        assert coordinator.bank.state == "POISONED"
