# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure schema and offline lifecycle helpers for elastic Graph catalogs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION = 5


def elastic_graph_catalog_row_complete(
    step_key: tuple[int, ...] | list[int],
    row: dict[str, int],
    *,
    representation: str,
) -> bool:
    """Validate evidence for the reachable replay lifecycle."""
    cold_complete = bool(
        row.get("cold_peak_bytes")
        and row.get("cold_observations", 0) >= 2
        and row.get("cold_stable_replays", 0) >= 1
    )
    if not cold_complete:
        return False
    if representation == "bounded_exact_hotset" and step_key[0] == 0:
        return True
    return bool(
        row.get("hot_peak_bytes")
        and row.get("hot_observations", 0) >= 2
        and row.get("hot_stable_replays", 0) >= 1
    )


def _step_key(value: Any, *, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RuntimeError(f"catalog contains a malformed {label} step key")
    return tuple(value)


def _validate_catalog_structure(payload: dict[str, Any]) -> None:
    coverage = payload.get("coverage")
    shapes = payload.get("shapes")
    if not isinstance(coverage, dict):
        raise RuntimeError("catalog source has no coverage object")
    if not isinstance(shapes, list):
        raise RuntimeError("catalog source has no physical shape rows")
    required_values = coverage.get("required_step_keys")
    if not isinstance(required_values, list):
        raise RuntimeError("catalog source has no required step-key inventory")
    required = tuple(_step_key(value, label="required") for value in required_values)
    if len(set(required)) != len(required):
        raise RuntimeError("catalog source has duplicate required step keys")
    rows: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in shapes:
        if not isinstance(row, dict):
            raise RuntimeError("catalog source contains a malformed shape row")
        key = _step_key(row.get("step_key"), label="shape")
        if key in rows:
            raise RuntimeError("catalog source contains duplicate shape rows")
        rows[key] = row
    if any(key not in rows for key in required):
        raise RuntimeError("catalog source required coverage is incomplete")
    if coverage.get("required_shapes") != len(required):
        raise RuntimeError("catalog source required-shape count is inconsistent")
    if payload.get("complete_shapes") != len(shapes):
        raise RuntimeError("catalog source complete-shape count is inconsistent")
    restore_values = coverage.get("restore_step_keys", [])
    if not isinstance(restore_values, list):
        raise RuntimeError("catalog source restore inventory is malformed")
    restore = tuple(_step_key(value, label="restore") for value in restore_values)
    if any(key not in required for key in restore):
        raise RuntimeError("catalog source restore keys exceed required coverage")


def load_sealed_catalog(path: Path, *, require_migration: bool) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"catalog source is unreadable: {path}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
        or payload.get("sealed") is not True
    ):
        raise RuntimeError("catalog source must be a sealed current-schema object")
    fingerprint = payload.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 16
        or any(char not in "0123456789abcdef" for char in fingerprint)
    ):
        raise RuntimeError("catalog source has an invalid fingerprint")
    migrated = payload.get("migrated_from") is not None
    if require_migration and (not migrated or not payload.get("migration_scope")):
        raise RuntimeError("catalog source has no explicit migration lineage")
    if not require_migration and (migrated or "migration_scope" in payload):
        raise RuntimeError("serving catalog still contains migration lineage")
    _validate_catalog_structure(payload)
    return payload


def finalize_migrated_catalog(source: Path, output_root: Path) -> Path:
    """Finalize an explicit migrated artifact without mutating its source."""
    finalized = load_sealed_catalog(source, require_migration=True)
    finalized.pop("migrated_from", None)
    finalized.pop("migration_scope", None)
    finalized["finalized_offline"] = True
    fingerprint = finalized["fingerprint"]
    destination = (
        output_root
        / "elastic_graph_catalog"
        / f"elastic_graph_catalog_{fingerprint}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite finalized catalog: {destination}")
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(finalized, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def rebind_finalized_catalog(
    source: Path,
    output_root: Path,
    *,
    destination_fingerprint: str,
    reason: str,
) -> Path:
    """Bind accepted physical rows to an explicitly resolved source identity."""
    if (
        len(destination_fingerprint) != 16
        or any(char not in "0123456789abcdef" for char in destination_fingerprint)
    ):
        raise RuntimeError("destination catalog fingerprint is invalid")
    if not reason.strip():
        raise RuntimeError("catalog rebind requires a non-empty reason")
    rebound = load_sealed_catalog(source, require_migration=False)
    source_fingerprint = rebound["fingerprint"]
    if source_fingerprint == destination_fingerprint:
        raise RuntimeError("catalog rebind destination equals its source fingerprint")
    rebound["fingerprint"] = destination_fingerprint
    rebound["offline_rebind"] = {
        "reason": reason,
        "source_fingerprint": source_fingerprint,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    destination = (
        output_root
        / "elastic_graph_catalog"
        / f"elastic_graph_catalog_{destination_fingerprint}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite rebound catalog: {destination}")
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(rebound, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
