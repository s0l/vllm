# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure schema and offline lifecycle helpers for elastic Graph catalogs."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION = 6


def expected_semantic_token_witnesses(
    *,
    configured_k: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    prefill_k: int | None = None,
) -> tuple[tuple[tuple[int, int, int, int, int], int], ...]:
    """Derive the complete schema-6 semantic tail witness inventory."""
    if configured_k <= 0:
        return ()
    phase_k = configured_k if prefill_k is None else prefill_k
    if phase_k < 0:
        raise ValueError("prefill K cannot be negative")
    witnesses: dict[tuple[int, int, int, int, int], int] = {}
    for live_tokens in range(1, min(3, max_num_batched_tokens) + 1):
        physical_tokens = min(
            1 << (live_tokens - 1).bit_length(), max_num_batched_tokens
        )
        witnesses[(0, phase_k, 1, physical_tokens, 0)] = live_tokens
    mixed_live_tokens = phase_k + 2
    if max_num_seqs >= 2 and mixed_live_tokens <= max_num_batched_tokens:
        physical_tokens = min(
            1 << (mixed_live_tokens - 1).bit_length(), max_num_batched_tokens
        )
        witnesses[(0, phase_k, 2, physical_tokens, 0)] = mixed_live_tokens
    return tuple(sorted(witnesses.items()))


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
        not isinstance(value, (list, tuple))
        or len(value) != 5
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RuntimeError(f"catalog contains a malformed {label} step key")
    return tuple(value)


def validate_elastic_catalog_key_inventory(
    required_values: Any,
    restore_values: Any,
    *,
    label: str,
    require_required: bool = True,
    require_restore: bool = False,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Normalize and validate the shared required/restore key contract."""
    if not isinstance(required_values, (list, tuple)):
        raise RuntimeError(f"{label} has no required step-key inventory")
    if not isinstance(restore_values, (list, tuple)):
        raise RuntimeError(f"{label} restore inventory is malformed")
    required = tuple(
        _step_key(value, label=f"{label} required") for value in required_values
    )
    restore = tuple(
        _step_key(value, label=f"{label} restore") for value in restore_values
    )
    if require_required and not required:
        raise RuntimeError(f"{label} has an empty required step-key inventory")
    if require_restore and not restore:
        raise RuntimeError(f"{label} has an empty restore step-key inventory")
    if len(set(required)) != len(required):
        raise RuntimeError(f"{label} has duplicate required step keys")
    if len(set(restore)) != len(restore):
        raise RuntimeError(f"{label} has duplicate restore step keys")
    if any(key not in required for key in restore):
        raise RuntimeError(f"{label} restore keys exceed required coverage")
    return required, restore


def validate_schema6_surface_lineage(
    coverage: dict[str, Any],
    required: tuple[tuple[int, ...], ...],
) -> None:
    """Validate the typed source-to-serving surface carried by new catalogs.

    Older schema-6 artifacts predate this optional extension. Once any typed
    lineage field is present, the complete versioned extension is part of the
    sealed consumer ABI and cannot degrade to best-effort provenance.
    """
    lineage_fields = {
        "surface_lineage_contract",
        "source_surface_schema",
        "source_surface_required_shapes",
        "source_surface_sha256",
        "migrated_source_required_shapes",
        "migrated_source_step_keys",
        "schema6_added_witness_shapes",
        "surface_migration_contract",
        "surface_full_aliases",
        "owner_evidence_step_keys",
    }
    present_fields = lineage_fields.intersection(coverage)
    if not present_fields:
        return
    if present_fields != lineage_fields:
        raise RuntimeError("catalog source has incomplete surface lineage metadata")
    if coverage.get("surface_lineage_contract") != "typed-source-serving-v1":
        raise RuntimeError("catalog source has unsupported surface lineage contract")

    source_schema = coverage.get("source_surface_schema")
    source_count = coverage.get("source_surface_required_shapes")
    source_sha256 = coverage.get("source_surface_sha256")
    migrated_count = coverage.get("migrated_source_required_shapes")
    added_witness_count = coverage.get("schema6_added_witness_shapes")
    if (
        isinstance(source_schema, bool)
        or source_schema not in (5, ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION)
        or isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count < 1
        or not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_sha256)
        or isinstance(migrated_count, bool)
        or not isinstance(migrated_count, int)
        or migrated_count < 1
        or isinstance(added_witness_count, bool)
        or not isinstance(added_witness_count, int)
        or added_witness_count < 0
    ):
        raise RuntimeError("catalog source has invalid surface lineage metadata")

    migrated_values = coverage.get("migrated_source_step_keys")
    if not isinstance(migrated_values, list):
        raise RuntimeError("catalog source has no migrated source inventory")
    migrated = tuple(
        _step_key(value, label="migrated source") for value in migrated_values
    )
    if len(set(migrated)) != len(migrated) or len(migrated) != migrated_count:
        raise RuntimeError("catalog source has invalid migrated source inventory")

    witness_values = coverage.get("semantic_token_witnesses")
    assert isinstance(witness_values, list)
    witness_keys = {
        _step_key(witness.get("step_key"), label="semantic witness")
        for witness in witness_values
        if isinstance(witness, dict)
    }
    if set(required) != set(migrated).union(witness_keys) or added_witness_count != len(
        witness_keys.difference(migrated)
    ):
        raise RuntimeError("catalog source lineage does not cover required shapes")

    aliases_value = coverage.get("surface_full_aliases")
    evidence_values = coverage.get("owner_evidence_step_keys")
    if not isinstance(aliases_value, list) or not isinstance(evidence_values, list):
        raise RuntimeError("catalog source has malformed typed FULL lineage")
    aliases = []
    for alias in aliases_value:
        if not isinstance(alias, dict):
            raise RuntimeError("catalog source has malformed FULL aliases")
        aliases.append(
            (
                _step_key(alias.get("semantic_step_key"), label="alias semantic"),
                _step_key(alias.get("physical_step_key"), label="alias physical"),
            )
        )
    evidence = tuple(
        _step_key(value, label="owner evidence") for value in evidence_values
    )
    alias_sources = {source for source, _physical in aliases}
    if len(alias_sources) != len(aliases) or len(set(evidence)) != len(evidence):
        raise RuntimeError("catalog source has duplicate typed FULL lineage")

    migration_contract = coverage.get("surface_migration_contract")
    if source_schema == 5:
        from vllm.v1.core.elastic_graph import (
            select_short_decode_physical_x,
            short_decode_inventory_xs,
        )

        decode_max_x = coverage.get("decode_max_x")
        if (
            isinstance(decode_max_x, bool)
            or not isinstance(decode_max_x, int)
            or decode_max_x < 1
        ):
            raise RuntimeError("catalog source has invalid lineage decode boundary")
        decode_k = next(iter(alias_sources), (None, None))[1]
        expected_sources = {(1, decode_k, x, x, 1) for x in range(1, decode_max_x + 1)}
        inventory_xs = short_decode_inventory_xs(decode_max_x)
        if (
            migration_contract != "schema5-full-owner-evidence-and-bounded-alias-v1"
            or alias_sources != expected_sources
            or set(evidence) != expected_sources
            or not expected_sources.issubset(migrated)
            or source_count != len(migrated)
            or any(
                physical
                != (
                    1,
                    semantic[1],
                    select_short_decode_physical_x(semantic[2], inventory_xs),
                    select_short_decode_physical_x(semantic[2], inventory_xs),
                    semantic[4],
                )
                or physical not in migrated
                for semantic, physical in aliases
            )
        ):
            raise RuntimeError("catalog source has invalid schema-5 FULL lineage")
    elif (
        migration_contract is not None
        or aliases
        or evidence
        or source_count != len(migrated)
    ):
        raise RuntimeError("catalog source has invalid schema-6 surface lineage")


def _validate_catalog_structure(payload: dict[str, Any]) -> None:
    coverage = payload.get("coverage")
    shapes = payload.get("shapes")
    if not isinstance(coverage, dict):
        raise RuntimeError("catalog source has no coverage object")
    if not isinstance(shapes, list):
        raise RuntimeError("catalog source has no physical shape rows")
    required, _restore = validate_elastic_catalog_key_inventory(
        coverage.get("required_step_keys"),
        coverage.get("restore_step_keys", []),
        label="catalog source",
    )
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
    if payload.get("schema") >= 6:
        witnesses = coverage.get("semantic_token_witnesses")
        if coverage.get(
            "semantic_witness_contract"
        ) != "live-current-to-physical-piecewise-v1" or not isinstance(witnesses, list):
            raise RuntimeError("catalog source has no semantic witness contract")
        witness_keys = []
        for witness in witnesses:
            if not isinstance(witness, dict):
                raise RuntimeError("catalog source has malformed semantic witnesses")
            key = _step_key(witness.get("step_key"), label="semantic witness")
            live_tokens = witness.get("live_num_tokens")
            if (
                isinstance(live_tokens, bool)
                or not isinstance(live_tokens, int)
                or live_tokens <= 0
                or live_tokens < key[2]
                or live_tokens > key[3]
                or key[0] != 0
                or key[4] != 0
                or key not in required
            ):
                raise RuntimeError("catalog source has malformed semantic witnesses")
            witness_keys.append(key)
        if len(set(witness_keys)) != len(witness_keys):
            raise RuntimeError("catalog source has duplicate semantic witnesses")
        validate_schema6_surface_lineage(coverage, required)


def _load_sealed_catalog_bytes(
    source_bytes: bytes,
    *,
    source_label: str,
    require_migration: bool,
    allow_previous_schema: bool = False,
) -> dict[str, Any]:
    try:
        payload: Any = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"catalog source is unreadable: {source_label}") from error
    accepted_schemas = {ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION}
    if allow_previous_schema:
        accepted_schemas.add(ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION - 1)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") not in accepted_schemas
        or payload.get("sealed") is not True
    ):
        expected = "current-or-previous" if allow_previous_schema else "current"
        raise RuntimeError(f"catalog source must be a sealed {expected}-schema object")
    if (allow_previous_schema or not require_migration) and (
        payload.get("finalized_offline") is not True
    ):
        raise RuntimeError("catalog source must be offline-finalized")
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


def load_sealed_catalog(
    path: Path | str,
    *,
    require_migration: bool,
    allow_previous_schema: bool = False,
) -> dict[str, Any]:
    """Load a sealed catalog or, explicitly, its prior shape-surface ABI.

    ``allow_previous_schema`` is only for calibration inventory discovery.
    Callers still receive the old object unchanged; no measurement row is
    rebound to the current runtime identity.
    """
    payload, _digest = load_sealed_catalog_with_digest(
        path,
        require_migration=require_migration,
        allow_previous_schema=allow_previous_schema,
    )
    return payload


def load_sealed_catalog_with_digest(
    path: Path | str,
    *,
    require_migration: bool,
    allow_previous_schema: bool = False,
) -> tuple[dict[str, Any], str]:
    """Load and hash exactly the same immutable source byte sequence."""
    path = Path(path)
    try:
        source_bytes = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"catalog source is unreadable: {path}") from error
    payload = _load_sealed_catalog_bytes(
        source_bytes,
        source_label=str(path),
        require_migration=require_migration,
        allow_previous_schema=allow_previous_schema,
    )
    return payload, hashlib.sha256(source_bytes).hexdigest()


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
    write_json_exclusive(destination, finalized)
    return destination


def rebind_finalized_catalog(
    source: Path,
    output_root: Path,
    *,
    destination_fingerprint: str,
    reason: str,
) -> Path:
    """Bind accepted physical rows to an explicitly resolved source identity."""
    if len(destination_fingerprint) != 16 or any(
        char not in "0123456789abcdef" for char in destination_fingerprint
    ):
        raise RuntimeError("destination catalog fingerprint is invalid")
    if not reason.strip():
        raise RuntimeError("catalog rebind requires a non-empty reason")
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise RuntimeError(f"catalog source is unreadable: {source}") from error
    rebound = _load_sealed_catalog_bytes(
        source_bytes,
        source_label=str(source),
        require_migration=False,
    )
    source_fingerprint = rebound["fingerprint"]
    if "price_identity" in rebound:
        raise RuntimeError(
            "price catalogs require compatibility migration, not fingerprint rebind"
        )
    if source_fingerprint == destination_fingerprint:
        raise RuntimeError("catalog rebind destination equals its source fingerprint")
    rebound["fingerprint"] = destination_fingerprint
    rebound["offline_rebind"] = {
        "reason": reason,
        "source_fingerprint": source_fingerprint,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    destination = (
        output_root
        / "elastic_graph_catalog"
        / f"elastic_graph_catalog_{destination_fingerprint}.json"
    )
    write_json_exclusive(destination, rebound)
    return destination


def write_json_exclusive(destination: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable JSON artifact without a replace race."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable catalog: {destination}"
            ) from error
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
