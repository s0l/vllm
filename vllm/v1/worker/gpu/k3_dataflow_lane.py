# SPDX-License-Identifier: Apache-2.0
"""Single-thread CUDA event lane for registered K3 dataflow graphs.

This module only constructs a deterministic device graph.  It has no worker
threads, CPU wake/sleep baton, timeout, admission policy or implicit quantum.
The caller owns cohort scheduling and supplies the complete semantic TP ticket
sequence selected by an already validated graph plan.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

import torch


T = TypeVar("T")
_ACTIVE_ROUTE: ContextVar["K3DataflowRoute | None"] = ContextVar(
    "k3_dataflow_route", default=None
)


class K3DataflowLaneError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class K3DataflowTicket:
    ordinal: int
    cohort: int
    stage: int
    phase: str

    def validate(self) -> None:
        if self.ordinal < 0 or self.cohort < 0 or self.stage < 0:
            raise K3DataflowLaneError("ticket coordinates must be nonnegative")
        if not self.phase:
            raise K3DataflowLaneError("ticket phase must be nonempty")


@dataclass(frozen=True, slots=True)
class K3DataflowRoute:
    lane: "K3DataflowLane"
    ticket: K3DataflowTicket
    compute_stream: torch.cuda.Stream
    wait_for_completion: bool = True


@contextmanager
def activate_k3_dataflow_route(
    lane: "K3DataflowLane",
    ticket: K3DataflowTicket,
    compute_stream: torch.cuda.Stream,
    *,
    wait_for_completion: bool = True,
) -> Iterator[None]:
    if _ACTIVE_ROUTE.get() is not None:
        raise K3DataflowLaneError("nested K3 communication route is forbidden")
    token = _ACTIVE_ROUTE.set(
        K3DataflowRoute(
            lane=lane,
            ticket=ticket,
            compute_stream=compute_stream,
            wait_for_completion=wait_for_completion,
        )
    )
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


def run_k3_dataflow_communication(operation: Callable[[], T]) -> T:
    route = _ACTIVE_ROUTE.get()
    if route is None:
        return operation()
    return route.lane.submit(
        route.ticket,
        operation,
        compute_stream=route.compute_stream,
        wait_for_completion=route.wait_for_completion,
    )


class K3DataflowLane:
    """Capture-time constructor for one globally ordered communication lane."""

    def __init__(
        self,
        tickets: Sequence[K3DataflowTicket],
        *,
        device: torch.device,
        communication_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.device = device
        self.tickets = tuple(tickets)
        if not self.tickets:
            raise K3DataflowLaneError("ticket sequence cannot be empty")
        for expected, ticket in enumerate(self.tickets):
            ticket.validate()
            if ticket.ordinal != expected:
                raise K3DataflowLaneError(
                    "ticket ordinals must be contiguous and globally ordered"
                )
        semantic = {
            (ticket.cohort, ticket.stage, ticket.phase) for ticket in self.tickets
        }
        if len(semantic) != len(self.tickets):
            raise K3DataflowLaneError("semantic tickets must be unique")
        self.communication_stream = communication_stream or torch.cuda.Stream(
            device=device
        )
        self._start = torch.cuda.Event()
        self._ready = [torch.cuda.Event() for _ in self.tickets]
        self._done = [torch.cuda.Event() for _ in self.tickets]
        self._cursor = 0
        self._active = False

    @property
    def cursor(self) -> int:
        return self._cursor

    def begin(
        self,
        *,
        origin_stream: torch.cuda.Stream | None = None,
        compute_streams: Sequence[torch.cuda.Stream] = (),
    ) -> None:
        if self._active:
            raise K3DataflowLaneError("ticket generation is already active")
        self._cursor = 0
        self._active = True
        if compute_streams:
            origin = origin_stream or torch.cuda.current_stream(self.device)
            self._start.record(origin)
            self.communication_stream.wait_event(self._start)
            for stream in compute_streams:
                stream.wait_event(self._start)

    def submit(
        self,
        ticket: K3DataflowTicket,
        operation: Callable[[], T],
        *,
        compute_stream: torch.cuda.Stream | None = None,
        wait_for_completion: bool = True,
    ) -> T:
        if not self._active:
            raise K3DataflowLaneError("begin() must precede ticket submission")
        if self._cursor >= len(self.tickets):
            raise K3DataflowLaneError("ticket sequence overflow")
        expected = self.tickets[self._cursor]
        if ticket != expected:
            raise K3DataflowLaneError(
                f"ticket order mismatch: expected={expected} actual={ticket}"
            )
        compute = compute_stream or torch.cuda.current_stream(self.device)
        ready = self._ready[self._cursor]
        done = self._done[self._cursor]
        ready.record(compute)
        self.communication_stream.wait_event(ready)
        with torch.cuda.stream(self.communication_stream):
            result = operation()
            done.record(self.communication_stream)
        # The CUDA allocator cannot infer event-ordered cross-stream use from
        # the Python lifetime alone.  Keep a tensor result owned until its
        # consumer stream has finished with it; otherwise a later graph replay
        # may recycle the storage while the compute continuation still reads it.
        if isinstance(result, torch.Tensor):
            result.record_stream(compute)
        # This is a device dependency only.  The host thread never waits.
        # A scheduler may defer this wait when the next packet belongs to an
        # independent cohort; that cohort owner's continuation must then wait
        # on done_event(ticket) explicitly before consuming the result.
        if wait_for_completion:
            compute.wait_event(done)
        self._cursor += 1
        return result

    def ready_event(self, ticket: K3DataflowTicket) -> torch.cuda.Event:
        self._validate_known(ticket)
        return self._ready[ticket.ordinal]

    def done_event(self, ticket: K3DataflowTicket) -> torch.cuda.Event:
        self._validate_known(ticket)
        return self._done[ticket.ordinal]

    def finish(self) -> None:
        if not self._active:
            raise K3DataflowLaneError("no ticket generation is active")
        if self._cursor != len(self.tickets):
            raise K3DataflowLaneError(
                f"incomplete ticket generation: {self._cursor}/{len(self.tickets)}"
            )
        self._active = False

    def abort(self) -> None:
        """Discard host construction state after a pre-replay failure.

        This does not claim that already enqueued GPU work is cancelled.  The
        caller must destroy the failed capture/process before reusing state.
        """

        self._active = False
        self._cursor = 0

    def _validate_known(self, ticket: K3DataflowTicket) -> None:
        ticket.validate()
        if ticket.ordinal >= len(self.tickets) or self.tickets[ticket.ordinal] != ticket:
            raise K3DataflowLaneError(f"unknown ticket: {ticket}")
