# SPDX-License-Identifier: Apache-2.0
"""Capture-time stream handoff for registered K3 elastic graph plans."""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import TypeVar

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)

T = TypeVar("T")
_TRACE_STATE: dict[int, dict[str, object]] = {}


def _record_handoff_phase(label: str, phase: str) -> None:
    context = get_forward_context()
    thread_id = threading.get_ident()
    previous = _TRACE_STATE.get(thread_id, {})
    invocation = int(previous.get("invocation", 0))
    if phase == "enter":
        invocation += 1
    _TRACE_STATE[thread_id] = {
        "thread_id": thread_id,
        "thread_name": threading.current_thread().name,
        "wave": context.additional_kwargs.get("k3_elastic_wave"),
        "plan": context.additional_kwargs.get("k3_elastic_graph_plan"),
        "label": label,
        "invocation": invocation,
        "phase": phase,
    }


def k3_elastic_handoff_trace_snapshot() -> list[dict[str, object]]:
    return [dict(value) for _, value in sorted(_TRACE_STATE.items())]


def k3_elastic_handoff_active() -> bool:
    if not dbo_enabled() or not is_forward_context_available():
        return False
    return bool(
        get_forward_context().additional_kwargs.get("k3_elastic_graph_plan")
    )


def run_k3_elastic_communication(
    operation: Callable[[], T], *, label: str = "unknown"
) -> T:
    if not k3_elastic_handoff_active():
        return operation()
    _record_handoff_phase(label, "enter")
    _record_handoff_phase(label, "yield_compute_to_comm")
    dbo_yield_and_switch_from_compute_to_comm()
    try:
        _record_handoff_phase(label, "operation")
        result = operation()
        _record_handoff_phase(label, "operation_complete")
        return result
    finally:
        _record_handoff_phase(label, "yield_comm_to_compute")
        dbo_yield_and_switch_from_comm_to_compute()
        _record_handoff_phase(label, "complete")
