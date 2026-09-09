# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic fixtures for the scheduler/worker elastic lifecycle boundary."""

from types import SimpleNamespace

from vllm.v1.core.elastic_graph import (
    ElasticAdmissionController,
    ElasticPlanKind,
    ElasticResidencyEntry,
    ElasticResidencyReceipt,
    ElasticStepPlan,
    LogicalDispatchKey,
    PhysicalReplayKey,
    RuntimeGeneration,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

GENERATION = RuntimeGeneration("elastic-lifecycle-fixture")


class ElasticCoordinatorFixture:
    def __init__(self, external_bytes: int = 0) -> None:
        self.elastic_external_memory_bytes = external_bytes
        self.set_calls: list[tuple[int, int]] = []

    @staticmethod
    def normalize_elastic_external_memory(value: int) -> int:
        return value

    def set_elastic_external_memory(
        self,
        value: int,
        *,
        minimum_free_primary_blocks: int,
    ) -> bool:
        self.elastic_external_memory_bytes = value
        self.set_calls.append((value, minimum_free_primary_blocks))
        return True


def physical_key(*, mode: str = "FULL") -> PhysicalReplayKey:
    return PhysicalReplayKey(
        logical=LogicalDispatchKey(
            owner="target",
            mode=mode,
            token_bucket=1,
            logical_num_reqs=1,
            uniform_query_len=1,
        ),
        physical_num_reqs=1,
        generation=GENERATION,
    )


def residency_receipt(
    *,
    pinned: bool,
    reclaimable_bytes: int = 0,
    lease_ids: tuple[str, ...] = (),
    resident_bytes: int = 88,
    floor_bytes: int = 0,
    transaction_id: str | None = None,
) -> ElasticResidencyReceipt:
    entry = ElasticResidencyEntry(
        key=physical_key(mode="FULL" if pinned else "PIECEWISE"),
        pinned=pinned,
        resident_bytes=resident_bytes,
        local_pool_bytes=resident_bytes,
        reclaimable_bytes=reclaimable_bytes,
        lease_ids=lease_ids,
    )
    return ElasticResidencyReceipt(
        generation=GENERATION,
        transaction_id=transaction_id,
        resident_bytes=resident_bytes,
        floor_bytes=floor_bytes,
        transition_floor_bytes=floor_bytes,
        peak_bytes=resident_bytes,
        cublas_workspace_bytes=16,
        entries=(entry,),
    )


def reclaim_plan(transaction_id: str = "elastic-fixture-reclaim") -> ElasticStepPlan:
    victim = physical_key(mode="PIECEWISE")
    return ElasticStepPlan(
        transaction_id=transaction_id,
        generation=GENERATION,
        kind=ElasticPlanKind.RECLAIM,
        physical_keys=(),
        hot_hits=(),
        cold_misses=(),
        protected_keys=(),
        victim_keys=(victim,),
        reclaim_groups=(f"private-pool:{victim.identity}",),
        capture_order=(),
        kv_transition=None,
        request_bytes=0,
        available_bytes=0,
        capture_loan_bytes=0,
        reclaim_bytes=88,
    )


def idle_output(*, grant_bytes: int = 0) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.elastic_external_memory_bytes = grant_bytes
    output.elastic_graph_external_memory_bytes = grant_bytes
    output.elastic_successor_primary_headroom = 0
    return output


def scheduler_boundary(
    *,
    grant_bytes: int = 0,
    running: bool = False,
) -> tuple[Scheduler, ElasticCoordinatorFixture]:
    scheduler = object.__new__(Scheduler)
    coordinator = ElasticCoordinatorFixture(grant_bytes)
    scheduler.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller = ElasticAdmissionController(GENERATION)
    scheduler._elastic_admission_controller.reserve_loan(None, grant_bytes)
    scheduler._elastic_admission_controller.step_key = None
    scheduler._elastic_admission_controller.recapture_pending_key = None
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_graph_catalog_coverage = {}
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler._elastic_admission_controller.pinned_resident_bytes = 0
    scheduler._elastic_admission_controller.evictable_resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.transition_floor_bytes = 0
    scheduler._elastic_admission_controller.cublas_workspace_bytes = 0
    scheduler._elastic_admission_controller.last_maintenance_step_key = None
    scheduler._elastic_admission_controller.idle_cleanup_started_at = None
    scheduler._elastic_graph_idle_cleanup_timeout_s = 5.0
    scheduler._elastic_restore_retention_id = None
    scheduler.running = [object()] if running else []
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=4096,
        max_num_seqs=64,
    )
    return scheduler, coordinator
