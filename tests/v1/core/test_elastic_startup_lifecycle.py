# SPDX-License-Identifier: Apache-2.0
"""Cross-layer regression controls for elastic startup and settlement."""

from dataclasses import replace

import pytest

from vllm.platforms import current_platform
from vllm.v1.core.elastic_graph import (
    ElasticAdmissionController,
    ElasticGraphError,
    ElasticPlanKind,
    ElasticResidencyReceipt,
    ElasticRuntimeConfig,
    GraphExecutionPolicy,
    GraphPrice,
    OwnerGraphExecutionPolicy,
    RuntimeGeneration,
)
from vllm.v1.worker import startup_plan

from .elastic_lifecycle_utils import (
    idle_output,
    reclaim_plan,
    residency_receipt,
    scheduler_boundary,
)
from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test


def _execution_policy() -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="lifecycle-test-v1",
        math_contract="lifecycle-test-v1",
        owners=(
            OwnerGraphExecutionPolicy("mtp_decode", (1,), "PIECEWISE"),
            OwnerGraphExecutionPolicy("mtp_prefill", (), "PIECEWISE"),
            OwnerGraphExecutionPolicy("target", (1,), "PIECEWISE"),
        ),
    )


def test_real_elastic_scheduler_constructor_uses_configured_width(monkeypatch):
    monkeypatch.delenv("AG2_VLLM_ELASTIC_REQUIRE_CATALOG", raising=False)
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    # CPU platform normalization disables CUDA Graph mode. Force only the
    # already-unit-tested activation authority so this control executes the
    # real elastic Scheduler.__init__ path without allocating a GPU.
    monkeypatch.setattr(
        ElasticRuntimeConfig,
        "from_vllm_config",
        classmethod(lambda cls, _config: cls(enabled=True)),
    )
    monkeypatch.setattr(startup_plan, "load_elastic_graph_catalog", lambda *_: {})

    scheduler = create_scheduler(
        max_num_seqs=16,
        additional_config={"elastic_gdn_backing": True},
        elastic_graph_execution_policy=_execution_policy().to_payload(),
    )

    assert scheduler.elastic_on_demand_graphs
    assert scheduler.max_num_running_reqs == 16
    assert max(scheduler._elastic_short_decode_inventory) == 16


def test_real_elastic_scheduler_constructor_fails_closed_without_catalog(
    monkeypatch,
):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(
        ElasticRuntimeConfig,
        "from_vllm_config",
        classmethod(lambda cls, _config: cls(enabled=True)),
    )
    monkeypatch.setattr(startup_plan, "load_elastic_graph_catalog", lambda *_: {})
    monkeypatch.setenv("AG2_VLLM_ELASTIC_REQUIRE_CATALOG", "1")

    with pytest.raises(RuntimeError, match="explicit offline catalog tool"):
        create_scheduler(
            max_num_seqs=16,
            additional_config={"elastic_gdn_backing": True},
            elastic_graph_execution_policy=_execution_policy().to_payload(),
        )


def test_idle_pinned_receipt_settles_as_retained_baseline_without_step_loan():
    scheduler, coordinator = scheduler_boundary()
    receipt = residency_receipt(pinned=True)

    scheduler._settle_elastic_graph_loan(
        idle_output(),
        worker_resident_bytes=receipt.resident_bytes,
        worker_floor_bytes=receipt.floor_bytes,
        worker_receipt=receipt,
    )

    assert scheduler._elastic_admission_controller.resident_bytes == 88
    assert scheduler._elastic_admission_controller.pinned_resident_bytes == 88
    assert scheduler._elastic_admission_controller.evictable_resident_bytes == 0
    assert not scheduler._needs_elastic_idle_reclaim()
    assert coordinator.elastic_external_memory_bytes == 88
    assert coordinator.set_calls == [(88, 0)]


def test_zero_token_reclaim_accepts_pinned_residue_during_abort_cleanup():
    scheduler, coordinator = scheduler_boundary(running=True)
    plan = reclaim_plan()
    receipt = residency_receipt(
        pinned=True,
        transaction_id=plan.transaction_id,
    )
    output = idle_output()
    output.elastic_step_plan = plan

    scheduler._settle_elastic_graph_loan(
        output,
        worker_resident_bytes=receipt.resident_bytes,
        worker_floor_bytes=receipt.floor_bytes,
        worker_receipt=receipt,
    )

    assert scheduler.running
    assert scheduler._elastic_admission_controller.resident_bytes == 88
    assert scheduler._elastic_admission_controller.pinned_resident_bytes == 88
    assert scheduler._elastic_admission_controller.evictable_resident_bytes == 0
    assert coordinator.elastic_external_memory_bytes == 88
    assert coordinator.set_calls == [(88, 0)]


def test_pressure_reclaim_requires_pinned_victims_absent():
    scheduler, coordinator = scheduler_boundary(running=True)
    plan = replace(reclaim_plan(), kind=ElasticPlanKind.PRESSURE_RECLAIM)
    receipt = residency_receipt(
        pinned=True,
        transaction_id=plan.transaction_id,
    )
    output = idle_output()
    output.elastic_step_plan = plan

    with pytest.raises(RuntimeError, match="retained an unprotected"):
        scheduler._settle_elastic_graph_loan(
            output,
            worker_resident_bytes=receipt.resident_bytes,
            worker_floor_bytes=receipt.floor_bytes,
            worker_receipt=receipt,
        )

    assert coordinator.elastic_external_memory_bytes == 0
    assert coordinator.set_calls == []
    assert scheduler.running


def test_pressure_reclaim_accepts_empty_receipt_without_changing_active_requests():
    scheduler, coordinator = scheduler_boundary(running=True)
    plan = replace(reclaim_plan(), kind=ElasticPlanKind.PRESSURE_RECLAIM)
    receipt = ElasticResidencyReceipt(
        generation=plan.generation,
        transaction_id=plan.transaction_id,
        resident_bytes=0,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=0,
        cublas_workspace_bytes=0,
        entries=(),
    )
    output = idle_output()
    output.elastic_step_plan = plan

    scheduler._settle_elastic_graph_loan(
        output,
        worker_resident_bytes=0,
        worker_floor_bytes=0,
        worker_receipt=receipt,
    )

    assert scheduler._elastic_admission_controller.resident_bytes == 0
    assert coordinator.elastic_external_memory_bytes == 0
    assert coordinator.set_calls == []
    assert scheduler.running


def test_zero_token_preserve_accepts_authoritative_evictable_receipt():
    scheduler, coordinator = scheduler_boundary(running=True)
    receipt = residency_receipt(pinned=False, reclaimable_bytes=88)
    output = idle_output()
    output.elastic_preserve_graph_residency = True

    scheduler._settle_elastic_graph_loan(
        output,
        worker_resident_bytes=receipt.resident_bytes,
        worker_floor_bytes=receipt.floor_bytes,
        worker_receipt=receipt,
    )

    assert scheduler.running
    assert scheduler._elastic_admission_controller.resident_bytes == 88
    assert scheduler._elastic_admission_controller.pinned_resident_bytes == 0
    assert scheduler._elastic_admission_controller.evictable_resident_bytes == 88
    assert coordinator.elastic_external_memory_bytes == 88
    assert coordinator.set_calls == [(88, 0)]


@pytest.mark.parametrize(
    ("running", "pinned", "reclaimable_bytes"),
    [
        (False, False, 88),
        (True, True, 0),
    ],
)
def test_zero_step_loan_rejects_non_idle_pinned_or_evictable_residency(
    running,
    pinned,
    reclaimable_bytes,
):
    scheduler, coordinator = scheduler_boundary(running=running)
    receipt = residency_receipt(
        pinned=pinned,
        reclaimable_bytes=reclaimable_bytes,
    )

    with pytest.raises(RuntimeError, match="exceeds its Graph loan component"):
        scheduler._settle_elastic_graph_loan(
            idle_output(),
            worker_resident_bytes=receipt.resident_bytes,
            worker_floor_bytes=receipt.floor_bytes,
            worker_receipt=receipt,
        )

    assert coordinator.elastic_external_memory_bytes == 0
    assert coordinator.set_calls == []


def test_settlement_rejects_missing_receipt_without_mutating_kv_accounting():
    scheduler, coordinator = scheduler_boundary()

    with pytest.raises(RuntimeError, match="omitted residency receipt"):
        scheduler._settle_elastic_graph_loan(
            idle_output(),
            worker_resident_bytes=0,
            worker_floor_bytes=0,
        )

    assert coordinator.elastic_external_memory_bytes == 0
    assert coordinator.set_calls == []


def test_settlement_rejects_stale_generation_before_mutating_hot_state():
    scheduler, coordinator = scheduler_boundary()
    accepted = residency_receipt(pinned=True)
    scheduler._sync_elastic_residency_receipt(accepted)
    prior_entries = dict(scheduler._elastic_admission_controller.entries)
    stale_generation = RuntimeGeneration("stale-lifecycle-fixture")
    stale = replace(
        accepted,
        generation=stale_generation,
        entries=(
            replace(
                accepted.entries[0],
                key=replace(accepted.entries[0].key, generation=stale_generation),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="generation differs"):
        scheduler._settle_elastic_graph_loan(
            idle_output(),
            worker_resident_bytes=stale.resident_bytes,
            worker_floor_bytes=stale.floor_bytes,
            worker_receipt=stale,
        )

    assert dict(scheduler._elastic_admission_controller.entries) == prior_entries
    assert coordinator.elastic_external_memory_bytes == 0
    assert coordinator.set_calls == []


def test_hot_synchronization_is_atomic_when_a_leased_entry_is_omitted():
    scheduler, _coordinator = scheduler_boundary()
    cache: ElasticAdmissionController = scheduler._elastic_admission_controller
    first = residency_receipt(pinned=False, reclaimable_bytes=88).entries[0]
    second = replace(
        first,
        key=replace(
            first.key,
            logical=replace(first.key.logical, token_bucket=2),
        ),
    )
    first_price = GraphPrice(88, 88, f"private-pool:{first.key.identity}")
    second_price = GraphPrice(88, 88, f"private-pool:{second.key.identity}")
    cache.synchronize_hot(
        (
            (first.key, first_price, False, 88),
            (second.key, second_price, False, 88),
        )
    )
    plan = cache.plan(
        "leased",
        (second.key,),
        request_bytes=0,
        available_bytes=176,
    )
    cache.commit_user(plan)
    prior_entries = dict(cache.entries)

    with pytest.raises(ElasticGraphError, match="dropped a scheduler-leased graph"):
        cache.synchronize_hot(())

    assert dict(cache.entries) == prior_entries


def test_hot_synchronization_is_atomic_when_reclaim_identity_changes():
    scheduler, _coordinator = scheduler_boundary()
    cache: ElasticAdmissionController = scheduler._elastic_admission_controller
    first = residency_receipt(pinned=False, reclaimable_bytes=88).entries[0]
    second = replace(
        first,
        key=replace(
            first.key,
            logical=replace(first.key.logical, token_bucket=2),
        ),
    )
    first_price = GraphPrice(88, 88, f"private-pool:{first.key.identity}")
    second_price = GraphPrice(88, 88, f"private-pool:{second.key.identity}")
    cache.synchronize_hot(
        (
            (first.key, first_price, False, 88),
            (second.key, second_price, False, 88),
        )
    )
    prior_entries = dict(cache.entries)
    changed_price = replace(second_price, reclaim_group="changed-group")

    with pytest.raises(ElasticGraphError, match="reclaim identity changed"):
        cache.synchronize_hot(((second.key, changed_price, False, 88),))

    assert dict(cache.entries) == prior_entries
