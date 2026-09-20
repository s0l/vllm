# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical byte and live KV admission controls for revocable expert grants."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry
from vllm.v1.core.elastic_expert import ElasticExpertGrant, NativeExpertBudget
from vllm.v1.core.elastic_graph import ElasticAdmissionController, RuntimeGeneration
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


def make_budget(*, tp=3, **options):
    return NativeExpertBudget(NVFP4ExpertGeometry(2560, 640, tp), 48, 512, **options)


def make_config(options, tp=3):
    return SimpleNamespace(
        additional_config={"flashnext_native_experts": options},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=2560,
                moe_intermediate_size=640,
                num_hidden_layers=48,
                num_experts=512,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


@pytest.mark.parametrize("wave_slots", [1, 2])
def test_native_residency_reserves_every_staging_slot_before_attach(wave_slots):
    from vllm.models.qwen4_exp.nvidia.expert_offload_residency import (
        NativeExpertResidency,
    )

    budget = cast(
        NativeExpertBudget,
        make_budget(max_hot_rows=128, staging=32, wave_slots=wave_slots),
    )
    rows = 32 * wave_slots
    sizes = [budget.round(rows * stride) for stride in budget.strides]
    sizes.append(budget.round((128 + rows) * 24))
    targets = {str(i): size for i, size in enumerate(sizes)}
    bank = SimpleNamespace(
        source=SimpleNamespace(
            geometry=budget.geometry, rank=0, layers=48, experts=512
        ),
        tables=SimpleNamespace(pool_rows=0),
        targets=lambda hot: targets,
        max_rows=128 + rows - 1,
        quantum=2 << 20,
        backings={
            name: SimpleNamespace(info=SimpleNamespace(committed=size))
            for name, size in targets.items()
        },
    )
    controller = SimpleNamespace(
        auxiliary_owner=None, backings={}, auxiliary_targets={}
    )
    provider = SimpleNamespace(bank=bank)
    with pytest.raises(ValueError, match="initial residency"):
        NativeExpertResidency(provider, controller, budget)
    assert controller.auxiliary_owner is None and not controller.backings
    bank.max_rows += 1
    owner = NativeExpertResidency(provider, controller, budget)
    assert owner is not None
    assert budget is not None
    assert owner.grant is not None
    assert cast(Any, controller).auxiliary_owner is owner
    assert bank.elastic_controller is controller
    assert owner.grant == ElasticExpertGrant(0, 0)
    assert sum(controller.auxiliary_targets.values()) == budget.rank_mapped_bytes(0, 0)


@pytest.mark.parametrize("staging", [1, 2, 32])
def test_native_quantum_curve_and_maximal_fit(staging):
    budget = make_budget(max_hot_rows=128, staging=staging)
    previous = 0
    for hot in range(129):
        # Independently price the four native TP3 arrays and six FP32 scalars.
        rows, q = hot + staging, 2 << 20
        sizes = (
            rows * 512 * 1280,
            rows * 2560 * 128,
            rows * 512 * 160,
            rows * 2560 * 16,
            (128 + staging) * 24,
        )
        expected = sum(((size + q - 1) // q) * q for size in sizes)
        assert budget.mapped_bytes(hot) == expected >= previous
        previous = expected
        grant = budget.grant(hot)
        fitted = budget.fit(grant.borrowed_bytes)
        assert fitted.hot_rows >= hot and fitted.borrowed_bytes <= grant.borrowed_bytes
        if fitted.hot_rows < 128:
            assert (
                budget.grant(fitted.hot_rows + 1).borrowed_bytes > grant.borrowed_bytes
            )
        if grant.borrowed_bytes:
            assert budget.fit(grant.borrowed_bytes - 1).hot_rows < hot


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 5])
def test_compact_rank_fit_is_maximal_and_accounts_actual_arrays(tp):
    budget = NativeExpertBudget.from_config(
        make_config(
            dict(
                max_hot_rows=128,
                ram_cache_total_bytes=1 << 30,
                pin_ram_cache=True,
                hot_read=True,
                prepared_archive="/prepared",
                stream_experts=dict(
                    library="/executor.so",
                    sha256="a" * 64,
                    cores=list(range(tp * 2)),
                    max_m=4,
                    history_steps=16,
                    max_promotions=192,
                ),
            ),
            tp,
        )
    )
    q = 2 << 20
    for hot in (0, 17, 64, 127):
        expected = []
        for rank in range(tp):
            width = min(budget.geometry.local, 640 - rank * budget.geometry.local)
            # Four physical arrays; scalars are already resident at zero HOT.
            strides = (
                2 * width * 1280,
                2560 * width // 2,
                2 * width * 160,
                2560 * width // 16,
            )
            expected.append(
                sum(
                    ((hot + 32) * s + q - 1) // q * q - (32 * s + q - 1) // q * q
                    for s in strides
                )
            )
        assert budget.rank_borrowed_bytes(hot) == tuple(expected)
        grant = budget.fit_by_rank(expected)
        assert grant.hot_rows >= hot
        assert all(
            a <= b
            for a, b in zip(
                budget.rank_borrowed_bytes(grant.hot_rows), expected, strict=True
            )
        )
        if grant.hot_rows < 128:
            assert any(
                a > b
                for a, b in zip(
                    budget.rank_borrowed_bytes(grant.hot_rows + 1),
                    expected,
                    strict=True,
                )
            )


@pytest.mark.parametrize(
    "options",
    [
        None,
        {"hot_rows": 1},
        {"staging": 3},
        {"staging": True},
        {"max_hot_rows": -1},
        {"ram_cache_bytes": 1.5},
        {"ram_cache_total_bytes": True},
        {"ram_cache_total_bytes": -1},
        {"ram_cache_total_bytes": 1.5},
        {"ram_cache_total_bytes": 32 << 30, "ram_cache_bytes": 1 << 30},
        {"pin_ram_cache": 1},
        {"pin_ram_cache": "true"},
        {"pin_ram_cache": None},
        {"hot_read": 1},
        {"prepared_archive": 1},
        {"prepared_archive": "relative/archive"},
        {"prepared_archive": ""},
        {"unknown": 1},
    ],
)
def test_invalid_budget_rejected_before_source_or_device_allocation(options):
    with pytest.raises(ValueError):
        NativeExpertBudget.from_config(make_config(options))


@pytest.mark.parametrize(
    "tp,local", [(1, 640), (2, 320), (3, 256), (4, 192), (5, 128), (8, 128), (16, 64)]
)
def test_scheduler_budget_consumes_model_tp_configuration(tp, local):
    budget = NativeExpertBudget.from_config(make_config({}, tp))
    assert budget.geometry.local == local
    assert budget.geometry.tp == tp
    assert budget.geometry.physical == local * tp
    assert budget.strides[0] == 2 * local * 1280
    if tp == 3:
        assert budget.strides == (655360, 327680, 81920, 40960)
        assert budget.base_bytes == 38 << 20


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 5, 8, 16])
def test_total_host_cache_budget_splits_by_configured_tp_and_recovers(tp):
    total = 32 << 30
    budget = NativeExpertBudget.from_config(
        make_config(
            {
                "ram_cache_total_bytes": total,
                "pin_ram_cache": True,
                "hot_read": True,
                "prepared_archive": "/archive",
            },
            tp,
        )
    )
    assert budget.pin_ram_cache is True
    assert budget.hot_read is True and budget.prepared_archive == "/archive"
    assert NativeExpertBudget.from_config(make_config({}, tp)).pin_ram_cache is False
    assert budget.ram_cache_bytes * tp <= total
    assert total - budget.ram_cache_bytes * tp < tp
    assert (
        NativeExpertBudget.from_config(
            make_config({"ram_cache_total_bytes": 0}, tp)
        ).ram_cache_bytes
        == 0
    )
    assert (
        NativeExpertBudget.from_config(
            make_config({"ram_cache_bytes": 123}, tp)
        ).ram_cache_bytes
        == 123
    )
    assert (
        NativeExpertBudget.from_config(make_config({}, tp)).ram_cache_bytes == 1 << 30
    )


def make_coordinator():
    q = 2 << 20
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["gdn"],
                MambaSpec(
                    block_size=4,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    separate_pool=True,
                    separate_pool_num_blocks=16,
                ),
            ),
        ],
        elastic_attention_stride=q,
        elastic_gdn_stride=q,
        elastic_mapping_quantum=q,
        elastic_gdn_initial_blocks=4,
        elastic_gdn_blocks_per_request=1,
        elastic_budget_bytes=132 * q,
        elastic_rank_primary_mapped_bytes=(tuple(i * q for i in range(129)),) * 3,
        elastic_rank_gdn_mapped_bytes=(tuple(i * q for i in range(17)),) * 3,
        elastic_rank_budget_bytes=(132 * q,) * 3,
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=512,
        scheduler_block_size=4,
        hash_block_size=4,
    ).coordinator


def test_scheduler_reclaims_experts_before_live_kv_admission_and_recovers():
    coordinator = make_coordinator()
    s = object.__new__(Scheduler)
    s.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
    s._elastic_native_budget = make_budget(max_hot_rows=128)
    s._elastic_native_hot_rows = 0
    s._refresh_elastic_cache_frontier = Mock()
    s._release_elastic_prefix_hits = Mock()
    graph = 16 << 20
    assert coordinator.set_elastic_external_memory(graph)

    def grant():
        return s._plan_elastic_expert_grant(
            has_user_tokens=True, minimum_free_primary_blocks=4
        )

    first = grant()
    assert first.hot_rows == 128
    before = coordinator.take_elastic_transition()
    assert before is not None
    s._schedule_with_prefix_leases = Mock(side_effect=lambda *a, **k: grant())
    assert s.schedule() == first
    assert coordinator.take_elastic_transition() is None
    allocated = []

    def pressure(*args, **kwargs):
        assert coordinator.elastic_expert_memory_bytes == 0
        allocated.extend(coordinator.block_pool.get_new_blocks(96))
        return grant()

    s._schedule_with_prefix_leases.side_effect = pressure
    smaller = s.schedule()
    assert 0 < smaller.hot_rows < first.hot_rows
    assert coordinator.elastic_external_memory_bytes == graph
    assert coordinator.block_pool.active_num_gpu_blocks > max(
        b.block_id for b in allocated
    )
    assert coordinator.block_pool.get_num_free_blocks() >= 4
    receipt = coordinator.elastic_kv_authority_receipt(8)
    assert receipt["graph_external_bytes"] == graph
    assert receipt["expert_borrowed_bytes"] == smaller.borrowed_bytes
    assert not coordinator.set_elastic_expert_memory(first.borrowed_bytes)
    assert coordinator.elastic_expert_memory_bytes == smaller.borrowed_bytes
    coordinator.block_pool.free_blocks(allocated)
    s._schedule_with_prefix_leases.side_effect = lambda *a, **k: grant()
    assert s.schedule() == first


def test_expert_grant_cannot_cross_a_pinned_kv_tail():
    coordinator = make_coordinator()
    pinned = coordinator.block_pool.blocks[120]
    coordinator.block_pool.touch([pinned])
    available = coordinator.max_elastic_external_memory()
    assert available == 7 * (2 << 20)
    assert not coordinator.set_elastic_expert_memory(available + (2 << 20))
    assert coordinator.elastic_expert_memory_bytes == 0 and pinned.ref_cnt == 1
    assert coordinator.set_elastic_expert_memory(available)
    assert coordinator.block_pool.active_num_gpu_blocks == 121


def test_rank_expert_charge_preserves_capacity_and_rejects_pinned_tail():
    coordinator = make_coordinator()
    q = 2 << 20
    # Smaller rank2 budget and shard: a replicated charge needlessly caps KV.
    coordinator.kv_cache_config.elastic_rank_budget_bytes = (132 * q, 132 * q, 116 * q)
    assert coordinator.set_elastic_expert_memory(32 * q)
    assert coordinator.block_pool.active_num_gpu_blocks == 80
    assert coordinator.set_elastic_expert_memory(
        32 * q, rank_requested_bytes=(32 * q, 32 * q, 16 * q)
    )
    assert coordinator.block_pool.active_num_gpu_blocks == 96
    receipt = coordinator.elastic_kv_authority_receipt(8)
    assert receipt["effective_attention_blocks_per_rank"] == (96, 96, 96)
    assert receipt["expert_borrowed_bytes_per_rank"] == (32 * q, 32 * q, 16 * q)
    pinned = coordinator.block_pool.blocks[95]
    coordinator.block_pool.touch([pinned])
    assert not coordinator.set_elastic_expert_memory(
        33 * q, rank_requested_bytes=(33 * q, 33 * q, 17 * q)
    )
    assert coordinator.elastic_expert_rank_memory_bytes == (32 * q, 32 * q, 16 * q)
    assert pinned.ref_cnt == 1
    coordinator.block_pool.free_blocks([pinned])
    assert coordinator.set_elastic_expert_memory(0)
    assert coordinator.elastic_expert_rank_memory_bytes == ()
    assert coordinator.block_pool.active_num_gpu_blocks == 112


@pytest.mark.parametrize("bad", [(1, 2), (1, -1, 1), (1, True, 1), (1, 4 << 20, 1)])
def test_invalid_rank_expert_charge_leaves_authority_unchanged(bad):
    coordinator = make_coordinator()
    with pytest.raises(ValueError):
        coordinator.set_elastic_expert_memory(2, rank_requested_bytes=bad)
    assert coordinator.elastic_expert_memory_bytes == 0
    assert coordinator.elastic_expert_rank_memory_bytes == ()


def test_physical_receipt_distinguishes_kv_tail_from_expert_allocation():
    from vllm.v1.worker.gpu.elastic_gdn import ElasticKVController

    q = 2 << 20
    controller = ElasticKVController(torch.device("cpu"))
    controller.backings = {
        name: SimpleNamespace(info=SimpleNamespace(committed=committed, quantum=q))
        for name, committed in (
            ("elastic-attention-0", 8 * q),
            ("elastic-gdn", 2 * q),
            ("native-w13", 3 * q),
        )
    }
    controller.geometry = {"elastic-attention-0": q, "elastic-gdn": q}
    controller.auxiliary_targets = {"native-w13": 3 * q}
    controller._physical_budget_bytes = 15 * q
    controller._external_memory_bytes = 2 * q
    controller._logical_transition = (3, 2)
    receipt = controller.physical_memory_receipt()
    owners = {row["owner"]: row for row in receipt["backings"]}
    assert owners["elastic-attention-0"]["retained_tail_bytes"] == 5 * q
    assert owners["native-w13"]["retained_tail_bytes"] == 0
    assert sum(row["mapped_bytes"] for row in owners.values()) == 13 * q
    controller._logical_transition = (7, 2)
    assert (
        controller.physical_memory_receipt()["backings"][0]["retained_tail_bytes"] == q
    )
    controller._logical_transition = None
    assert controller.physical_memory_receipt()["backings"][0]["required_bytes"] is None


def test_expert_grant_changes_both_plan_and_replay_epoch_identity():
    controller = ElasticAdmissionController(RuntimeGeneration("fixture"))
    plan = controller.plan(
        "elastic-00000000000000000001", (), request_bytes=0, available_bytes=0
    )
    budget = make_budget(staging=1)
    # A geometry change may consume only already rounded staging slack.
    assert budget.grant(0).borrowed_bytes == budget.grant(1).borrowed_bytes
    a, b = (replace(plan, expert_grant=budget.grant(n)) for n in (0, 1))
    assert a.fingerprint != b.fingerprint
    assert a.execution_epoch_fingerprint != b.execution_epoch_fingerprint


@pytest.mark.parametrize("failure", [None, "bytes", "peer", "stale", "plan", "lease"])
def test_worker_admits_common_grant_before_retiring_or_mutating_bank(
    monkeypatch, failure
):
    from vllm.models.qwen4_exp.nvidia.expert_offload_residency import (
        NativeExpertResidency,
    )

    budget = make_budget(max_hot_rows=128)
    owner = NativeExpertResidency.__new__(NativeExpertResidency)
    owner.budget, owner.grant = budget, budget.grant(0)
    owner.last_sequence = 1 if failure == "stale" else 0
    owner.bank = SimpleNamespace(
        state="READY",
        generation=0,
        tables=SimpleNamespace(pool_rows=0),
        leases={object()} if failure == "lease" else set(),
        targets=lambda hot: {"fixture": budget.mapped_bytes(hot)},
    )
    coordinator = SimpleNamespace(
        identity=[17, 41],
        send=torch.empty(64, dtype=torch.int64),
        recv=torch.empty(3 * 64, dtype=torch.int64),
        ranks=3,
        group=SimpleNamespace(device_group=object()),
    )
    owner.provider = SimpleNamespace(
        active=False, coordinator=coordinator, retire=Mock(), quiesce=Mock()
    )
    owner.controller = SimpleNamespace(
        auxiliary_targets={"native-fixture": budget.base_bytes}
    )
    before = dict(owner.controller.auxiliary_targets)

    def gather(destination, source, *, group):
        assert group is coordinator.group.device_group
        destination.view(3, -1).copy_(source.expand(3, -1))
        if failure == "peer":
            destination.view(3, -1)[1, 4] += 1

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    grant = budget.grant(64)
    if failure == "bytes":
        grant = ElasticExpertGrant(grant.hot_rows, grant.borrowed_bytes + 1)
    plan = None
    if failure == "plan":
        plan = SimpleNamespace(expert_grant=budget.grant(32), fingerprint="fixture")
    output = SimpleNamespace(
        elastic_expert_grant=grant,
        elastic_transaction_id="elastic-00000000000000000001",
        elastic_step_plan=plan,
        elastic_plan_fingerprint="fixture" if plan else None,
        elastic_kv_transition=(16, 4),
        elastic_external_memory_bytes=0,
    )
    if failure:
        with pytest.raises(RuntimeError, match="rejected across ranks"):
            owner.admit(output)
        assert (
            owner.bank.state == "POISONED"
            and owner.controller.auxiliary_targets == before
        )
    else:
        owner.admit(output)
        assert owner.pending and owner.last_sequence == 1
        assert owner.controller.auxiliary_targets[
            "native-fixture"
        ] == budget.mapped_bytes(64)
    assert owner.bank.tables.pool_rows == 0
    owner.provider.retire.assert_not_called()
    owner.provider.quiesce.assert_not_called()
