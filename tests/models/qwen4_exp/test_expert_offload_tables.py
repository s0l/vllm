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
