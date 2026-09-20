# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded allocation profiling, distinct from model/request calibration.

An envelope is an empirical memory profile, not an execution receipt for every
logical request. It deliberately gives no shared-owner credit. Runtime physical
admission and overshoot checks remain authoritative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CONTRACT = "request-free-allocation-envelope-v2"


def allocation_profile_shapes(k: int, max_x: int, budget: int):
    if any(type(v) is not int for v in (k, max_x, budget)) or not (
        k >= 0 and 1 <= max_x <= budget and max_x * (k + 1) <= budget
    ):
        raise ValueError("invalid allocation profile domain")
    boundaries = []
    m = 1
    while m < budget:
        boundaries.append(m)
        m *= 2
    boundaries.append(budget)
    # X bounds cover request metadata; M bounds cover token-major allocations.
    # q is semantic and must survive even when target uses PIECEWISE.
    corners = {(0, k, x, m, 0) for m in boundaries for x in (1, min(m, max_x))}
    corners.update((0, k, x, x * q, q) for x in (1, max_x) for q in range(1, k + 2))
    # Serving retains the power-of-two short-decode inventory before READY.
    # Measure each terminal owner set directly instead of pricing those HOT
    # aliases only from the aggregate envelope.
    decode_xs = []
    x = 1
    while x < max_x:
        decode_xs.append(x)
        x *= 2
    decode_xs.append(max_x)
    corners.update((0, k, x, x * q, q) for x in decode_xs for q in range(1, k + 2))
    # Independent odd/interior controls detect shape-dependent algorithm changes.
    interior_x = {x for x in (2, 3, 17, max_x - 1) if 1 < x < max_x}
    holdouts = {(0, k, x, x * q, q) for x in interior_x for q in range(1, k + 2)}
    holdouts.update((0, k, x, budget, 0) for x in interior_x)
    return tuple(sorted(corners)), tuple(sorted(holdouts - corners))


def _nonnegative(value: Any) -> bool:
    return type(value) is int and value >= 0


def allocation_envelope_proof(
    *,
    k: int,
    max_x: int,
    budget: int,
    policy: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    corners, holdouts = allocation_profile_shapes(k, max_x, budget)
    measured = {tuple(sample["step_key"]): dict(sample) for sample in samples}
    if len(measured) != len(samples) or set(measured) != set(corners + holdouts):
        raise ValueError("allocation profile is missing or repeats physical controls")
    for sample in measured.values():
        for name in (
            "capture_peak_bytes",
            "resident_bytes",
            "floor_bytes",
            "replay_extra_bytes",
            "source_reads",
            "stable_replays",
        ):
            if not _nonnegative(sample.get(name)):
                raise ValueError("allocation profile has invalid measurements")
        if sample["source_reads"] or sample["stable_replays"] < 1:
            raise ValueError("allocation profile has I/O or unstable replay")
        if sample["capture_peak_bytes"] < sample["resident_bytes"]:
            raise ValueError("allocation profile understates resident memory")
    # Odd/interior shapes are physical response-surface controls.  A measured
    # non-monotonic shape must widen the conservative envelope rather than make
    # an otherwise complete profile unpublishable. Scratch is independently
    # maximized and added; no inferred owner overlap is subtracted.
    controls = corners + holdouts
    peak = max(measured[key]["capture_peak_bytes"] for key in controls)
    scratch = max(measured[key]["replay_extra_bytes"] for key in controls)
    resident = max(measured[key]["resident_bytes"] for key in controls)
    floor = max(measured[key]["floor_bytes"] for key in controls)
    return dict(
        contract=CONTRACT,
        k=k,
        max_x=max_x,
        budget=budget,
        policy=policy,
        samples=[measured[key] for key in corners + holdouts],
        peak_bytes=peak + scratch,
        # A replay can introduce persistent inner graphs or library state.
        # Treat all extra memory as retained until physical teardown proves
        # otherwise; never lend these bytes twice to Graph and expert owners.
        resident_bytes=resident + scratch,
        floor_bytes=floor + scratch,
        shared_owner_credit_bytes=0,
    )


def allocation_envelope_row(proof: Mapping[str, Any]) -> dict[str, Any]:
    return dict(
        cold_peak_bytes=proof["peak_bytes"],
        hot_peak_bytes=proof["peak_bytes"],
        resident_bytes=proof["resident_bytes"],
        floor_bytes=proof["floor_bytes"],
        cold_observations=0,
        cold_stable_replays=0,
        hot_observations=0,
        hot_stable_replays=0,
        allocation_profile=dict(proof),
    )


def validate_allocation_envelope_row(step_key, row, *, policy=None, budget=None):
    """Validate provenance and arithmetic; never invent executed-row counts."""
    proof = row.get("allocation_profile")
    if not isinstance(proof, dict) or proof.get("contract") != CONTRACT:
        raise ValueError("missing allocation envelope contract")
    rebuilt = allocation_envelope_proof(
        **{name: proof[name] for name in ("k", "max_x", "budget", "policy", "samples")}
    )
    if rebuilt != proof or allocation_envelope_row(proof) != {
        name: row.get(name) for name in allocation_envelope_row(proof)
    }:
        raise ValueError("allocation envelope provenance or price was modified")
    if row.get("resident_key_bytes"):
        raise ValueError("allocation envelope cannot claim shared owner credit")
    if not (
        len(step_key) == 5
        and step_key[1] == proof["k"]
        and 1 <= step_key[2] <= proof["max_x"]
        and step_key[2] <= step_key[3] <= proof["budget"]
    ):
        raise ValueError("step is outside the allocation envelope")
    if policy is not None and proof["policy"] != policy:
        raise ValueError("allocation envelope has stale Graph policy")
    if budget is not None and proof["budget"] != budget:
        raise ValueError("allocation envelope has stale token budget")
