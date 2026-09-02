# SPDX-License-Identifier: Apache-2.0
"""Saved FIX7 trace oracle for the KV6 Step 4 controller extraction."""

from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from vllm.v1.core.elastic_graph import (
    ElasticAdmissionController,
    ElasticGraphError,
    ElasticResidencyEntry,
    ElasticResidencyReceipt,
    GraphPrice,
    LogicalDispatchKey,
    PhysicalReplayKey,
    RuntimeGeneration,
    require_plan_consensus,
)

TRACE_PATH = Path(__file__).with_name("fixtures") / "elastic_admission_fix7_trace.json"
PINNED_GROWTH_TRACE_PATH = (
    Path(__file__).with_name("fixtures") / "elastic_pinned_idle_drain_trace.json"
)


class _LegacySchedulerBoundary:
    """The pre-extraction Scheduler ownership used by the FIX7 reference."""

    def __init__(self, generation: RuntimeGeneration) -> None:
        self.graph = ElasticAdmissionController(generation)
        self.loans: deque[tuple[tuple[int, ...] | None, int]] = deque()
        self.transaction_seq = 0

    def next_transaction_id(self) -> str:
        self.transaction_seq += 1
        return f"elastic-{self.transaction_seq:020d}"

    def reserve_loan(self, step_key: tuple[int, ...] | None, grant: int) -> None:
        self.loans.append((step_key, grant))

    def settle_next_loan(self) -> tuple[tuple[int, ...] | None, int]:
        return self.loans.popleft()

    def cancel_latest_loan(self) -> tuple[tuple[int, ...] | None, int]:
        return self.loans.pop()

    def accept_receipt(self, receipt: ElasticResidencyReceipt) -> None:
        """Replay the accepted FIX7 raw-tuple publication semantics."""
        parsed = [
            (
                entry.key,
                GraphPrice(
                    entry.resident_bytes,
                    entry.resident_bytes,
                    f"private-pool:{entry.key.identity}",
                ),
                entry.pinned,
                entry.reclaimable_bytes,
            )
            for entry in receipt.entries
        ]
        self.graph.synchronize_hot(parsed)
        self.graph.resident_bytes = receipt.resident_bytes
        self.graph.pinned_resident_bytes = sum(
            entry.resident_bytes for entry in receipt.entries if entry.pinned
        )
        self.graph.evictable_resident_bytes = sum(
            entry.resident_bytes for entry in receipt.entries if not entry.pinned
        )
        self.graph.floor_bytes = receipt.floor_bytes
        self.graph.transition_floor_bytes = receipt.transition_floor_bytes
        self.graph.cublas_workspace_bytes = receipt.cublas_workspace_bytes


def _load_trace() -> dict[str, Any]:
    trace = json.loads(TRACE_PATH.read_text())
    if trace.get("schema") != 2:
        raise AssertionError("FIX7 trace requires explicit destination-set schema v2")
    return trace


def _keys(
    trace: dict[str, Any], generation: RuntimeGeneration
) -> dict[str, PhysicalReplayKey]:
    return {
        name: PhysicalReplayKey(
            logical=LogicalDispatchKey(
                owner=row["owner"],
                mode=row["mode"],
                token_bucket=row["token_bucket"],
                logical_num_reqs=(row["physical_x"] if row["mode"] == "FULL" else None),
                uniform_query_len=None,
            ),
            physical_num_reqs=row["physical_x"],
            generation=generation,
        )
        for name, row in trace["keys"].items()
    }


def _loan_rows(boundary: Any) -> list[list[Any]]:
    rows = (
        boundary.loans
        if isinstance(boundary, _LegacySchedulerBoundary)
        else ((loan.step_key, loan.grant_bytes) for loan in boundary.pending_loans)
    )
    return [[None if key is None else list(key), grant] for key, grant in rows]


def _graph(boundary: Any) -> ElasticAdmissionController:
    return (
        boundary.graph if isinstance(boundary, _LegacySchedulerBoundary) else boundary
    )


def _plan_projection(plan: Any, labels: dict[PhysicalReplayKey, str]) -> dict[str, Any]:
    def names(items: tuple[PhysicalReplayKey, ...]) -> list[str]:
        return [labels[item] for item in items]

    return {
        "kind": plan.kind.value,
        "physical_keys": names(plan.physical_keys),
        "hot_hits": names(plan.hot_hits),
        "cold_misses": names(plan.cold_misses),
        "victim_keys": names(plan.victim_keys),
        "kv_transition": (
            None if plan.kv_transition is None else list(plan.kv_transition)
        ),
        "capture_loan_bytes": plan.capture_loan_bytes,
        "reclaim_bytes": plan.reclaim_bytes,
        "defer_reason": plan.defer_reason,
    }


def _replay(trace: dict[str, Any], boundary: Any) -> dict[str, Any]:
    generation = RuntimeGeneration(trace["generation"])
    keys = _keys(trace, generation)
    labels = {value: name for name, value in keys.items()}
    graph = _graph(boundary)
    transcript: list[Any] = []

    for event in trace["events"]:
        op = event["op"]
        if op == "next_transaction":
            actual = boundary.next_transaction_id()
        elif op == "publish_hot":
            reclaim_group = event["reclaim_group"]
            if reclaim_group == "physical_private_pool":
                reclaim_group = f"private-pool:{keys[event['key']].identity}"
            graph.publish_hot(
                keys[event["key"]],
                GraphPrice(
                    event["resident_bytes"],
                    event["capture_peak_bytes"],
                    reclaim_group,
                ),
                pinned=event["pinned"],
            )
            actual = None
        elif op == "reserve_loan":
            step_key = tuple(event["step_key"])
            if isinstance(boundary, _LegacySchedulerBoundary):
                boundary.reserve_loan(step_key, event["grant_bytes"])
            else:
                boundary.reserve_loan(step_key, event["grant_bytes"])
            actual = _loan_rows(boundary)
            assert actual == event["expected_loans"]
        elif op in {"plan_user_cancel", "plan_defer", "plan_capture_fail_recovery"}:
            plan = graph.plan(
                event["transaction_id"],
                tuple(keys[name] for name in event["keys"]),
                request_bytes=event["request_bytes"],
                available_bytes=event["available_bytes"],
                kv_transition=tuple(event["kv_transition"]),
                destination_capture_endpoint_bytes=event.get(
                    "destination_capture_endpoint_bytes"
                ),
                shared_resident_bytes=event.get("shared_resident_bytes", 0),
            )
            actual = _plan_projection(plan, labels)
            if op == "plan_user_cancel":
                graph.commit_user(plan)
                graph.cancel(plan.transaction_id)
            elif op == "plan_defer":
                graph.observe_defer(plan)
            else:
                graph.begin_maintenance(plan)
                graph.fail_maintenance(plan)
        elif op == "settle_next_loan":
            loan = boundary.settle_next_loan()
            actual = (
                [
                    None if loan.step_key is None else list(loan.step_key),
                    loan.grant_bytes,
                ]
                if not isinstance(boundary, _LegacySchedulerBoundary)
                else [None if loan[0] is None else list(loan[0]), loan[1]]
            )
        elif op == "cancel_latest_loan":
            loan = boundary.cancel_latest_loan()
            actual = (
                [
                    None if loan.step_key is None else list(loan.step_key),
                    loan.grant_bytes,
                ]
                if not isinstance(boundary, _LegacySchedulerBoundary)
                else [None if loan[0] is None else list(loan[0]), loan[1]]
            )
        elif op == "accept_receipt":
            receipt = ElasticResidencyReceipt(
                generation=generation,
                transaction_id=event["transaction_id"],
                resident_bytes=event["resident_bytes"],
                floor_bytes=event["floor_bytes"],
                transition_floor_bytes=event["transition_floor_bytes"],
                peak_bytes=event["peak_bytes"],
                cublas_workspace_bytes=event["cublas_workspace_bytes"],
                entries=tuple(
                    ElasticResidencyEntry(
                        key=keys[row["key"]],
                        resident_bytes=row["resident_bytes"],
                        local_pool_bytes=row["local_pool_bytes"],
                        pinned=row["pinned"],
                        reclaimable_bytes=row["reclaimable_bytes"],
                        lease_ids=tuple(row["lease_ids"]),
                    )
                    for row in event["entries"]
                ),
            )
            if isinstance(boundary, _LegacySchedulerBoundary):
                boundary.accept_receipt(receipt)
            else:
                boundary.accept_residency_receipt(receipt)
            actual = {
                "resident_bytes": graph.resident_bytes,
                "pinned_resident_bytes": graph.pinned_resident_bytes,
                "evictable_resident_bytes": graph.evictable_resident_bytes,
                "floor_bytes": graph.floor_bytes,
                "transition_floor_bytes": graph.transition_floor_bytes,
                "cublas_workspace_bytes": graph.cublas_workspace_bytes,
            }
        else:  # pragma: no cover - fixture schema is fail-closed below
            raise AssertionError(f"unsupported trace operation: {op}")

        if "expected" in event:
            assert actual == event["expected"]
        transcript.append(actual)

    stats = asdict(graph.stats)
    stats["defer_reasons"] = [list(row) for row in stats["defer_reasons"]]
    final = {
        "pending_loans": _loan_rows(boundary),
        "entries": {
            labels[key]: [entry.state.value, sorted(entry.leases)]
            for key, entry in graph.entries.items()
        },
        "accounting": {
            "resident_bytes": graph.resident_bytes,
            "pinned_resident_bytes": graph.pinned_resident_bytes,
            "evictable_resident_bytes": graph.evictable_resident_bytes,
            "floor_bytes": graph.floor_bytes,
            "transition_floor_bytes": graph.transition_floor_bytes,
            "cublas_workspace_bytes": graph.cublas_workspace_bytes,
        },
        "stats": stats,
    }
    assert final == trace["final"]
    return {"transcript": transcript, "final": final}


def test_saved_fix7_trace_is_identical_across_controller_extraction() -> None:
    trace = _load_trace()
    generation = RuntimeGeneration(trace["generation"])

    legacy = _replay(trace, _LegacySchedulerBoundary(generation))
    extracted = _replay(trace, ElasticAdmissionController(generation))

    assert extracted == legacy


def test_saved_trace_oracle_detects_order_mutation() -> None:
    trace = deepcopy(_load_trace())
    trace["events"][5]["expected"] = [[1, 3, 1, 4, 4], 96]

    with pytest.raises(AssertionError):
        _replay(
            trace, ElasticAdmissionController(RuntimeGeneration(trace["generation"]))
        )


def test_stale_receipt_validator_rejects_without_state_mutation() -> None:
    trace = _load_trace()
    generation = RuntimeGeneration(trace["generation"])
    controller = ElasticAdmissionController(generation)
    before = controller.snapshot
    stale = ElasticResidencyReceipt(
        generation=RuntimeGeneration("stale-generation"),
        transaction_id=None,
        resident_bytes=0,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=0,
        cublas_workspace_bytes=0,
        entries=(),
    )

    with pytest.raises(ElasticGraphError, match="generation differs"):
        controller.accept_residency_receipt(stale)

    assert controller.snapshot == before
    assert trace["declared_validator_difference"]["controller"] == (
        "reject before mutation"
    )


def test_saved_pinned_growth_trace_selects_one_rank_consistent_idle_drain() -> None:
    trace = json.loads(PINNED_GROWTH_TRACE_PATH.read_text())
    generation = RuntimeGeneration(trace["generation"])
    plans = []
    controllers = []
    for _ in range(3):
        controller = ElasticAdmissionController(generation)
        pinned = PhysicalReplayKey(
            logical=LogicalDispatchKey(
                owner="target",
                mode="FULL",
                token_bucket=17,
                logical_num_reqs=17,
                uniform_query_len=1,
            ),
            physical_num_reqs=17,
            generation=generation,
        )
        evictable = PhysicalReplayKey(
            logical=LogicalDispatchKey(
                owner="target",
                mode="PIECEWISE",
                token_bucket=68,
                logical_num_reqs=None,
                uniform_query_len=None,
            ),
            physical_num_reqs=17,
            generation=generation,
        )
        controller.publish_hot(
            pinned,
            GraphPrice(
                trace["resident"]["pinned_bytes"],
                trace["resident"]["pinned_bytes"],
                "pinned-full",
            ),
            pinned=True,
        )
        controller.publish_hot(
            evictable,
            GraphPrice(
                trace["resident"]["evictable_bytes"],
                trace["resident"]["evictable_bytes"],
                "evictable-piecewise",
            ),
            pinned=False,
        )
        controller.resident_bytes = trace["resident"]["total_bytes"]
        controller.pinned_resident_bytes = trace["resident"]["pinned_bytes"]
        controller.evictable_resident_bytes = trace["resident"]["evictable_bytes"]
        controller.floor_bytes = trace["resident"]["floor_bytes"]
        before = controller.snapshot
        plan = controller.plan_pressure_reclaim_all(
            "pressure-x39",
            request_bytes=trace["resident"]["total_bytes"],
            available_bytes=0,
        )
        assert controller.snapshot == before
        assert plan.kind.value == "pressure_reclaim"
        assert set(plan.victim_keys) == {pinned, evictable}
        assert plan.capture_loan_bytes == trace["resident"]["floor_bytes"]
        controllers.append(controller)
        plans.append(plan)

    require_plan_consensus(plans)
    for controller, plan in zip(controllers, plans, strict=True):
        controller.begin_reclaim(plan)
        assert not any(entry.hot for entry in controller.entries.values())
