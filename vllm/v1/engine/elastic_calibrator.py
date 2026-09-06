# SPDX-License-Identifier: Apache-2.0
"""Pre-READY producer for a measured elastic CUDA Graph catalog."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from vllm.logger import init_logger
from vllm.v1.core.elastic_catalog import (
    ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
    elastic_graph_catalog_row_complete,
    expected_semantic_token_witnesses,
    validate_elastic_catalog_key_inventory,
)
from vllm.v1.core.elastic_graph import (
    select_short_decode_physical_x,
    short_decode_inventory_xs,
)

logger = init_logger(__name__)


@contextmanager
def _catalog_producer_lock(output_root: Path, fingerprint: str):
    """Single producer per immutable fingerprint; process death releases it."""
    directory = output_root / "elastic_graph_catalog"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"elastic_graph_catalog_{fingerprint}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"elastic catalog producer already active: {fingerprint}"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class CalibrationSurface:
    required: tuple[tuple[int, int, int, int, int], ...]
    restore: tuple[tuple[int, int, int, int, int], ...]
    decode_max_x: int
    mixed_max_x: int
    full_context_max_x: int
    semantic_token_witnesses: tuple[
        tuple[tuple[int, int, int, int, int], int], ...
    ] = ()
    mixed_query_witnesses: tuple[tuple[tuple[int, int, int, int, int], int], ...] = ()
    source_schema: int = ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
    source_required_shapes: int = 0
    migration_contract: str | None = None
    full_aliases: tuple[
        tuple[
            tuple[int, int, int, int, int],
            tuple[int, int, int, int, int],
        ],
        ...,
    ] = ()
    migrated_source_keys: tuple[tuple[int, int, int, int, int], ...] = ()
    source_sha256: str | None = None
    owner_evidence_keys: tuple[tuple[int, int, int, int, int], ...] = ()

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        policy_fingerprint: str,
        configured_k: int,
        prefill_k: int | None = None,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        source_sha256: str | None = None,
    ) -> CalibrationSurface:
        coverage = payload.get("coverage")
        if not isinstance(coverage, dict):
            raise RuntimeError("calibration surface has no coverage object")
        if coverage.get("representation") != "bounded_exact_hotset":
            raise RuntimeError("calibration requires a bounded exact surface")
        if coverage.get("graph_execution_policy_fingerprint") != policy_fingerprint:
            raise RuntimeError("calibration surface policy does not match runtime")
        source_schema = payload.get("schema")
        if (
            isinstance(source_schema, bool)
            or not isinstance(source_schema, int)
            or source_schema not in (5, ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION)
        ):
            raise RuntimeError("calibration surface has unsupported source schema")
        if source_sha256 is not None and (
            len(source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_sha256)
        ):
            raise RuntimeError("calibration surface SHA256 is malformed")

        def keys(name: str) -> tuple[tuple[int, int, int, int, int], ...]:
            values = coverage.get(name)
            if not isinstance(values, list):
                raise RuntimeError(f"calibration surface has no {name}")
            result = []
            for value in values:
                if (
                    not isinstance(value, list)
                    or len(value) != 5
                    or any(
                        isinstance(item, bool) or not isinstance(item, int)
                        for item in value
                    )
                ):
                    raise RuntimeError(f"calibration surface has malformed {name}")
                result.append(tuple(value))
            return tuple(sorted(result))  # type: ignore[return-value]

        required = keys("required_step_keys")
        restore = keys("restore_step_keys")
        required, restore = validate_elastic_catalog_key_inventory(
            required,
            restore,
            label="calibration surface",
            require_restore=True,
        )
        source_required_shapes = len(required)
        declared_required_shapes = coverage.get("required_shapes")
        if declared_required_shapes is not None and (
            isinstance(declared_required_shapes, bool)
            or not isinstance(declared_required_shapes, int)
            or declared_required_shapes != source_required_shapes
        ):
            raise RuntimeError(
                "calibration surface required-shape count is inconsistent"
            )
        if (
            len(restore) != 2
            or sum(key[0] == 0 for key in restore) != 1
            or sum(key[0] == 1 for key in restore) != 1
        ):
            raise RuntimeError(
                "calibration surface requires one FULL/PIECEWISE restore pair"
            )
        effective_prefill_k = configured_k if prefill_k is None else prefill_k
        if any(
            key[1] != (configured_k if key[0] == 1 else effective_prefill_k)
            or key[2] < 1
            or key[2] > max_num_seqs
            or key[3] < 1
            or key[3] > max_num_batched_tokens
            or key[4] < 0
            for key in required
        ):
            raise RuntimeError("calibration surface exceeds the effective runtime")
        decode_max_x = coverage.get("decode_max_x")
        mixed_max_x = coverage.get("mixed_max_x")
        full_context_max_x = coverage.get("full_context_max_x")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (decode_max_x, mixed_max_x, full_context_max_x)
        ):
            raise RuntimeError("calibration surface has invalid product boundaries")
        if (
            decode_max_x > max_num_seqs
            or mixed_max_x > decode_max_x
            or full_context_max_x > decode_max_x
            or max(key[2] for key in required) > decode_max_x
        ):
            raise RuntimeError("calibration surface boundaries are inconsistent")
        migration_contract = None
        full_aliases = ()
        if source_schema == 5:
            # Schema 5 recorded exact qlen1 rows. In schema 6 those rows have
            # two typed roles: bounded aliases describe the target product
            # route, while the exact rows remain owner evidence for the
            # mtp_decode FULL descriptor reached by mixed current execution.
            inventory_xs = short_decode_inventory_xs(decode_max_x)
            source_full_keys = {key for key in required if key[0] == 1}
            expected_source_full_keys = {
                (1, configured_k, x, x, 1) for x in range(1, decode_max_x + 1)
            }
            if source_full_keys != expected_source_full_keys:
                missing = sorted(expected_source_full_keys - source_full_keys)[:8]
                extra = sorted(source_full_keys - expected_source_full_keys)[:8]
                raise RuntimeError(
                    "schema-5 FULL semantic domain is incomplete or malformed: "
                    f"missing={missing!r} extra={extra!r}"
                )

            def physical_key(
                key: tuple[int, int, int, int, int],
            ) -> tuple[int, int, int, int, int]:
                if key[0] != 1:
                    return key
                if key[4] <= 0 or key[3] != key[2] * key[4]:
                    raise RuntimeError(
                        "schema-5 FULL key is not a semantic X/query class: "
                        f"key={key!r}"
                    )
                physical_x = select_short_decode_physical_x(key[2], inventory_xs)
                return (1, key[1], physical_x, physical_x * key[4], key[4])

            full_aliases = tuple(
                (key, physical_key(key)) for key in required if key[0] == 1
            )
            migration_contract = "schema5-full-owner-evidence-and-bounded-alias-v1"
            owner_evidence_keys = tuple(sorted(source_full_keys))
        else:
            owner_evidence_keys = ()
        migrated_source_keys = required
        semantic_token_witnesses = dict(
            expected_semantic_token_witnesses(
                configured_k=configured_k,
                prefill_k=effective_prefill_k,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
            )
        )
        # Schema-6 execution identity requires direct semantic witnesses for
        # prefix and short-prefill tails. PIECEWISE M is a physical bucket, so
        # live M3 is represented by physical M4 rather than a fictitious M3
        # catalog row. The X2 mixed witness similarly maps live M5 to M8.
        if semantic_token_witnesses:
            required = tuple(sorted(set(required).union(semantic_token_witnesses)))
        declared_witnesses = coverage.get("semantic_token_witnesses")
        if declared_witnesses is not None:
            if not isinstance(declared_witnesses, list):
                raise RuntimeError("calibration surface has malformed witnesses")
            parsed_witnesses = []
            for witness in declared_witnesses:
                if not isinstance(witness, dict):
                    raise RuntimeError("calibration surface has malformed witnesses")
                step_key = witness.get("step_key")
                live_tokens = witness.get("live_num_tokens")
                if (
                    not isinstance(step_key, list)
                    or len(step_key) != 5
                    or any(
                        isinstance(item, bool) or not isinstance(item, int)
                        for item in step_key
                    )
                    or isinstance(live_tokens, bool)
                    or not isinstance(live_tokens, int)
                ):
                    raise RuntimeError("calibration surface has malformed witnesses")
                parsed_witnesses.append((tuple(step_key), live_tokens))
            if tuple(sorted(parsed_witnesses)) != tuple(
                sorted(semantic_token_witnesses.items())
            ):
                raise RuntimeError(
                    "calibration surface semantic witnesses differ from runtime"
                )
        declared_mixed = coverage.get("mixed_query_witnesses")
        if declared_mixed is None:
            mixed_query_witnesses = tuple(
                (key, query_len)
                for key in required
                if key[0] == 0 and key[4] == 0 and key[2] >= 2
                for query_len in (1, 1 + configured_k)
                if semantic_token_witnesses.get(key, key[3]) > (key[2] - 1) * query_len
            )
        else:
            if not isinstance(declared_mixed, list):
                raise RuntimeError("calibration surface has malformed mixed witnesses")
            parsed_mixed = []
            for witness in declared_mixed:
                if not isinstance(witness, dict):
                    raise RuntimeError(
                        "calibration surface has malformed mixed witnesses"
                    )
                step_key = witness.get("step_key")
                query_len = witness.get("query_len")
                if (
                    not isinstance(step_key, list)
                    or len(step_key) != 5
                    or any(
                        isinstance(item, bool) or not isinstance(item, int)
                        for item in step_key
                    )
                    or isinstance(query_len, bool)
                    or not isinstance(query_len, int)
                ):
                    raise RuntimeError(
                        "calibration surface has malformed mixed witnesses"
                    )
                key = tuple(step_key)
                if (
                    key not in required
                    or key[0] != 0
                    or key[4] != 0
                    or key[2] < 2
                    or query_len not in (1, 1 + configured_k)
                    or semantic_token_witnesses.get(key, key[3])
                    <= (key[2] - 1) * query_len
                ):
                    raise RuntimeError(
                        "calibration surface mixed witness is outside its "
                        "declared physical/semantic domain"
                    )
                parsed_mixed.append((key, query_len))
            if len(set(parsed_mixed)) != len(parsed_mixed):
                raise RuntimeError("calibration surface has duplicate mixed witnesses")
            eligible_query_lens = {
                query_len
                for key in required
                if key[0] == 0 and key[4] == 0 and key[2] >= 2
                for query_len in (1, 1 + configured_k)
                if semantic_token_witnesses.get(key, key[3]) > (key[2] - 1) * query_len
            }
            if {query_len for _key, query_len in parsed_mixed} != eligible_query_lens:
                raise RuntimeError(
                    "calibration surface has incomplete mixed query classes"
                )
            mixed_query_witnesses = tuple(sorted(parsed_mixed))
        return cls(
            required=required,
            restore=restore,
            decode_max_x=decode_max_x,
            mixed_max_x=mixed_max_x,
            full_context_max_x=full_context_max_x,
            semantic_token_witnesses=tuple(sorted(semantic_token_witnesses.items())),
            mixed_query_witnesses=mixed_query_witnesses,
            source_schema=source_schema,
            source_required_shapes=source_required_shapes,
            migration_contract=migration_contract,
            full_aliases=full_aliases,
            migrated_source_keys=migrated_source_keys,
            source_sha256=source_sha256,
            owner_evidence_keys=owner_evidence_keys,
        )


@dataclass(frozen=True)
class CalibrationResult:
    destination: Path
    fingerprint: str
    surface: CalibrationSurface
    measured_shapes: int
    wall_seconds: float


@dataclass(frozen=True)
class CalibrationCheckpoint:
    catalog: dict[tuple[int, ...], dict[str, Any]]
    mixed_query_witnesses: frozenset[tuple[tuple[int, ...], int]]
    process_epochs: int = 0
    wall_seconds: float = 0.0


class ElasticCalibrationRestartRequired(RuntimeError):
    """A bounded producer epoch completed and needs a fresh communicator."""


def parse_calibration_checkpoint(
    payload: Any,
    *,
    surface: CalibrationSurface,
    fingerprint: str,
    surface_sha256: str | None,
) -> CalibrationCheckpoint:
    """Validate identity-bound observations from an earlier process epoch."""
    if payload is None:
        return CalibrationCheckpoint({}, frozenset())
    if not isinstance(payload, dict):
        raise RuntimeError("elastic calibration checkpoint is malformed")
    schema = payload.get("schema")
    if schema not in {
        "ag2-elastic-calibration-checkpoint-v1",
        "ag2-elastic-calibration-checkpoint-v2",
    }:
        raise RuntimeError("elastic calibration checkpoint has unsupported schema")
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError("elastic calibration checkpoint fingerprint differs")
    if payload.get("surface_sha256") != surface_sha256:
        raise RuntimeError("elastic calibration checkpoint surface differs")
    process_epochs = payload.get("process_epochs")
    if (
        isinstance(process_epochs, bool)
        or not isinstance(process_epochs, int)
        or process_epochs < 0
    ):
        raise RuntimeError("elastic calibration checkpoint epoch count is invalid")
    wall_seconds = payload.get("calibration_wall_seconds")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or wall_seconds < 0
    ):
        raise RuntimeError("elastic calibration checkpoint wall time is invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("elastic calibration checkpoint has no row list")
    required = set(surface.required)
    catalog: dict[tuple[int, ...], dict[str, Any]] = {}
    numeric_fields = {
        "cold_peak_bytes",
        "hot_peak_bytes",
        "resident_bytes",
        "floor_bytes",
        "cold_observations",
        "cold_stable_replays",
        "hot_observations",
        "hot_stable_replays",
    }
    allowed_fields = numeric_fields | {"resident_key_bytes"}
    for item in rows:
        if not isinstance(item, dict):
            raise RuntimeError("elastic calibration checkpoint row is malformed")
        raw_key = item.get("step_key")
        if (
            not isinstance(raw_key, list)
            or len(raw_key) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_key
            )
        ):
            raise RuntimeError("elastic calibration checkpoint key is malformed")
        key = tuple(raw_key)
        if key not in required or key in catalog:
            raise RuntimeError(
                "elastic calibration checkpoint row inventory is invalid"
            )
        row = item.get("row")
        if not isinstance(row, dict) or not set(row).issubset(allowed_fields):
            raise RuntimeError("elastic calibration checkpoint row data is malformed")
        normalized = dict(row)
        if any(
            isinstance(normalized.get(name), bool)
            or not isinstance(normalized.get(name), int)
            or normalized[name] < 0
            for name in numeric_fields
        ):
            raise RuntimeError("elastic calibration checkpoint counters are invalid")
        resident_key_bytes = normalized.get("resident_key_bytes")
        if resident_key_bytes is not None:
            if not isinstance(resident_key_bytes, list) or any(
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or isinstance(entry[1], bool)
                or not isinstance(entry[1], int)
                or entry[1] < 0
                for entry in resident_key_bytes
            ):
                raise RuntimeError(
                    "elastic calibration checkpoint residency map is invalid"
                )
            normalized_residency = tuple(
                (entry[0], entry[1]) for entry in resident_key_bytes
            )
            if (
                tuple(sorted(normalized_residency)) != normalized_residency
                or len({identity for identity, _value in normalized_residency})
                != len(normalized_residency)
                or sum(value for _identity, value in normalized_residency)
                > normalized["resident_bytes"]
            ):
                raise RuntimeError(
                    "elastic calibration checkpoint residency map is inconsistent"
                )
            normalized["resident_key_bytes"] = normalized_residency
        complete = elastic_graph_catalog_row_complete(
            key,
            normalized,
            representation="bounded_exact_hotset",
        )
        cold_observations = normalized["cold_observations"]
        hot_observations = normalized["hot_observations"]
        if schema == "ag2-elastic-calibration-checkpoint-v1" and not complete:
            raise RuntimeError("elastic calibration checkpoint contains incomplete row")
        if schema == "ag2-elastic-calibration-checkpoint-v2" and (
            cold_observations < 1
            or normalized["cold_peak_bytes"] < 1
            or normalized["resident_bytes"] < 1
            or normalized["floor_bytes"] > normalized["resident_bytes"]
            or normalized["resident_bytes"]
            > max(normalized["cold_peak_bytes"], normalized["hot_peak_bytes"])
            or normalized["cold_stable_replays"] > max(0, cold_observations - 1)
            or normalized["hot_stable_replays"] > max(0, hot_observations - 1)
            or bool(hot_observations) != bool(normalized["hot_peak_bytes"])
        ):
            raise RuntimeError(
                "elastic calibration checkpoint row observations are inconsistent"
            )
        catalog[key] = normalized
    completed_shapes = sum(
        elastic_graph_catalog_row_complete(
            key,
            row,
            representation="bounded_exact_hotset",
        )
        for key, row in catalog.items()
    )
    declared_completed = payload.get("completed_shapes")
    if (
        isinstance(declared_completed, bool)
        or not isinstance(declared_completed, int)
        or declared_completed != completed_shapes
    ):
        raise RuntimeError("elastic calibration checkpoint row count differs")
    if schema == "ag2-elastic-calibration-checkpoint-v2":
        declared_observed = payload.get("observed_shapes")
        if (
            isinstance(declared_observed, bool)
            or not isinstance(declared_observed, int)
            or declared_observed != len(catalog)
        ):
            raise RuntimeError(
                "elastic calibration checkpoint observation count differs"
            )

    raw_witnesses = payload.get("mixed_query_witnesses")
    if not isinstance(raw_witnesses, list):
        raise RuntimeError("elastic calibration checkpoint has no mixed witnesses")
    witnesses: set[tuple[tuple[int, ...], int]] = set()
    declared_witnesses = set(surface.mixed_query_witnesses)
    for item in raw_witnesses:
        if not isinstance(item, dict):
            raise RuntimeError("elastic calibration checkpoint witness is malformed")
        raw_key = item.get("step_key")
        query_len = item.get("query_len")
        if (
            not isinstance(raw_key, list)
            or len(raw_key) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_key
            )
            or isinstance(query_len, bool)
            or not isinstance(query_len, int)
        ):
            raise RuntimeError("elastic calibration checkpoint witness is malformed")
        witness = (tuple(raw_key), query_len)
        if witness not in declared_witnesses or witness in witnesses:
            raise RuntimeError(
                "elastic calibration checkpoint witness inventory is invalid"
            )
        witnesses.add(witness)
    return CalibrationCheckpoint(
        catalog,
        frozenset(witnesses),
        process_epochs,
        float(wall_seconds),
    )


def calibration_checkpoint_payload(
    *,
    catalog: dict[tuple[int, ...], dict[str, Any]],
    mixed_query_witnesses: set[tuple[tuple[int, ...], int]],
    fingerprint: str,
    surface_sha256: str | None,
    process_epochs: int,
    calibration_wall_seconds: float,
) -> dict[str, Any]:
    """Serialize validated observations across clean process epochs."""
    observed = {
        key: dict(row)
        for key, row in catalog.items()
        if row.get("cold_observations", 0) > 0
    }
    completed_shapes = sum(
        elastic_graph_catalog_row_complete(
            key,
            row,
            representation="bounded_exact_hotset",
        )
        for key, row in observed.items()
    )
    return {
        "schema": "ag2-elastic-calibration-checkpoint-v2",
        "fingerprint": fingerprint,
        "surface_sha256": surface_sha256,
        "process_epochs": process_epochs,
        "calibration_wall_seconds": calibration_wall_seconds,
        "completed_shapes": completed_shapes,
        "observed_shapes": len(observed),
        "rows": [
            {"step_key": list(key), "row": observed[key]} for key in sorted(observed)
        ],
        "mixed_query_witnesses": [
            {"step_key": list(key), "query_len": query_len}
            for key, query_len in sorted(mixed_query_witnesses)
        ],
    }


class ElasticCatalogCalibrator:
    """Remeasure an accepted shape surface through production EngineCore."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.scheduler = cast(Any, owner.scheduler)
        self.serial = 0

    def _complete(self, key: tuple[int, ...]) -> bool:
        return elastic_graph_catalog_row_complete(
            key,
            self.scheduler._elastic_graph_catalog.get(key) or {},
            representation="bounded_exact_hotset",
        )

    def _next_serial(self) -> int:
        self.serial += 1
        return self.serial

    def _validate_surface_before_mutation(self, surface: CalibrationSurface) -> None:
        """Prove every declared identity against the effective scheduler."""
        restore_mode = self.scheduler._elastic_restore_mode
        self.scheduler._elastic_restore_mode = False
        try:
            for semantic_key, physical_key in surface.full_aliases:
                actual = self.scheduler._canonical_elastic_graph_step_key(
                    {
                        f"decode-alias-{index}": semantic_key[4]
                        for index in range(semantic_key[2])
                    },
                    semantic_key[1],
                    True,
                )
                if actual != physical_key:
                    raise RuntimeError(
                        "schema-5 FULL alias differs from effective runtime: "
                        f"semantic={semantic_key!r} "
                        f"declared_physical={physical_key!r} derived={actual!r}"
                    )
        finally:
            self.scheduler._elastic_restore_mode = restore_mode
        witnesses = dict(surface.semantic_token_witnesses)
        owner_evidence = set(surface.owner_evidence_keys)
        for key in surface.required:
            physical_keys = self.scheduler._resolve_elastic_step_physical_keys(key)
            if not physical_keys:
                raise RuntimeError(
                    "calibration surface contains a compiled-only shape: "
                    f"step_key={key!r}"
                )
            if key in owner_evidence:
                mixed_key = self.scheduler._canonical_elastic_graph_step_key(
                    {f"mixed-owner-{index}": 1 for index in range(key[2])},
                    key[1],
                    False,
                )
                mixed_physical = self.scheduler._resolve_elastic_step_physical_keys(
                    mixed_key
                )
                evidence_mtp = tuple(
                    item for item in physical_keys if item.logical.owner == "mtp_decode"
                )
                mixed_mtp = tuple(
                    item
                    for item in mixed_physical
                    if item.logical.owner == "mtp_decode"
                )
                if len(evidence_mtp) != 1 or evidence_mtp != mixed_mtp:
                    raise RuntimeError(
                        "schema-5 FULL owner evidence differs from mixed runtime: "
                        f"evidence={key!r} mixed={mixed_key!r}"
                    )
                continue
            if key[4]:
                actual = self.scheduler._canonical_elastic_graph_step_key(
                    {f"decode-{index}": key[4] for index in range(key[2])},
                    key[1],
                    True,
                )
            else:
                live_tokens = witnesses.get(key, key[3])
                if live_tokens < key[2]:
                    raise RuntimeError(
                        "calibration prefill witness cannot give every row a "
                        f"positive length: step_key={key!r} live={live_tokens}"
                    )
                lengths = [
                    live_tokens // key[2] + int(index < live_tokens % key[2])
                    for index in range(key[2])
                ]
                if any(length <= 0 for length in lengths):
                    raise RuntimeError(
                        "calibration prefill witness produced a zero-length row: "
                        f"step_key={key!r} lengths={lengths!r}"
                    )
                actual = self.scheduler._canonical_elastic_graph_step_key(
                    {
                        f"prefill-{index}": length
                        for index, length in enumerate(lengths)
                    },
                    key[1],
                    False,
                )
            if actual != key:
                raise RuntimeError(
                    "calibration surface identity differs from effective runtime: "
                    f"declared={key!r} derived={actual!r}"
                )

        full_key = next(key for key in surface.restore if key[0] == 1)
        expected_restore = set(
            self.owner._elastic_restore_wave_step_keys(
                k=full_key[1],
                x=full_key[2],
                query_len=full_key[4],
            )
        )
        if set(surface.restore) != expected_restore:
            raise RuntimeError(
                "calibration restore pair differs from effective runtime: "
                f"declared={sorted(surface.restore)!r} "
                f"derived={sorted(expected_restore)!r}"
            )

    def _balanced_prefill_pair(
        self, *, k: int, x: int, m: int
    ) -> tuple[tuple[int, ...] | None, int]:
        if not 1 <= x <= m:
            raise ValueError(f"balanced prefill requires 1 <= X <= M, got {x}/{m}")
        serial = self._next_serial()
        lengths = [m // x + int(index < m % x) for index in range(x)]
        expected = self.scheduler._canonical_elastic_graph_step_key(
            {f"prefill-{index}": length for index, length in enumerate(lengths)},
            k,
            False,
        )
        if expected is None:
            raise RuntimeError("balanced prefill has no graph identity")
        self.owner._begin_elastic_restore_physical_epoch((expected,))
        outputs = []
        for repeat in range(2):
            if repeat:
                self.owner._prepare_elastic_restore_capture(expected)
                self.scheduler.assert_elastic_restore_captures_hot((expected,))
            request_ids = []
            for index, prompt_len in enumerate(lengths):
                request_id = (
                    f"_elastic_offline_prefill_{serial}_{k}_{m}_{x}_{repeat}_{index}"
                )
                request_ids.append(request_id)
                self.owner._add_elastic_restore_request(
                    request_id=request_id,
                    prompt_len=prompt_len,
                    k=k,
                    max_tokens=1,
                )
            admitted = self.owner._prepare_elastic_restore_admission(
                request_ids,
                retain_hot_graphs=bool(
                    self.scheduler._elastic_admission_controller.resident_bytes
                ),
            )
            if admitted != x:
                self.owner._rollback_elastic_restore_physical_epoch(
                    request_ids=request_ids,
                    step_keys=(expected,),
                )
                if not 0 < admitted < x:
                    raise RuntimeError(
                        "balanced prefill made no monotone admission progress"
                    )
                return None, admitted
            output = self.owner._run_elastic_restore_step()
            actual = len(output.num_scheduled_tokens)
            if actual != x:
                self.owner._rollback_elastic_restore_physical_epoch(
                    request_ids=request_ids,
                    step_keys=(expected,),
                )
                if not 0 < actual < x:
                    raise RuntimeError("balanced prefill made no scheduler progress")
                return None, actual
            outputs.append(output)
            self.owner.abort_requests(request_ids)
        for output in outputs:
            self.owner._assert_elastic_restore_key(output, expected)
        self.owner._drain_elastic_restore()
        return expected, x

    def _mixed_prefill_pair(
        self, *, k: int, x: int, m: int, query_len: int
    ) -> tuple[tuple[int, ...] | None, int]:
        decode_reqs = x - 1
        prefill_len = m - decode_reqs * query_len
        if decode_reqs <= 0 or prefill_len <= 0:
            return None, min(x, 1)
        serial = self._next_serial()
        base_ids = [
            f"_elastic_offline_base_{serial}_{index}" for index in range(decode_reqs)
        ]
        base_key = self.scheduler._canonical_elastic_graph_step_key(
            dict.fromkeys(base_ids, 2), k, False
        )
        expected = self.scheduler._canonical_elastic_graph_step_key(
            {
                **{f"decode-{index}": query_len for index in range(decode_reqs)},
                "prefill": prefill_len,
            },
            k,
            False,
        )
        if base_key is None or expected is None:
            raise RuntimeError("mixed prefill has no graph identity")
        declared = (base_key, expected)
        self.owner._begin_elastic_restore_physical_epoch(declared)
        retention = self.scheduler.retain_elastic_restore_captures(declared)
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                self.scheduler.release_elastic_restore_retention(retention)
                released = True

        for request_id in base_ids:
            self.owner._add_elastic_restore_request(
                request_id=request_id, prompt_len=2, k=k, max_tokens=16
            )
        admitted = self.owner._prepare_elastic_restore_admission(
            base_ids, retain_hot_graphs=True
        )
        if admitted != decode_reqs:
            release()
            self.owner._rollback_elastic_restore_physical_epoch(
                request_ids=base_ids,
                step_keys=declared,
            )
            return None, admitted
        base_output = self.owner._run_elastic_restore_step()
        if len(base_output.num_scheduled_tokens) != decode_reqs:
            release()
            self.owner._rollback_elastic_restore_physical_epoch(
                request_ids=base_ids, step_keys=declared
            )
            return None, len(base_output.num_scheduled_tokens)

        outputs = []
        for repeat in range(2):
            if k > 0 and query_len == 1:
                for request_id in base_ids:
                    self.scheduler.requests[request_id].spec_token_ids.clear()
            prefill_id = f"_elastic_offline_mixed_{serial}_{repeat}"
            self.owner._add_elastic_restore_request(
                request_id=prefill_id,
                prompt_len=prefill_len,
                k=k,
                max_tokens=1,
            )
            output = None
            for _ in range(4):
                self.scheduler.prepare_elastic_restore_execution(expected)
                candidate = self.owner._run_elastic_restore_step()
                if candidate.num_scheduled_tokens:
                    output = candidate
                    break
            actual = 0 if output is None else len(output.num_scheduled_tokens)
            if actual != x:
                release()
                self.owner._rollback_elastic_restore_physical_epoch(
                    request_ids=[prefill_id, *base_ids], step_keys=declared
                )
                return None, actual
            outputs.append(output)
            self.owner.abort_requests([prefill_id])
        for output in outputs:
            self.owner._assert_elastic_restore_key(output, expected)
        release()
        self.owner.abort_requests(base_ids)
        self.owner._drain_elastic_restore()
        return expected, x

    def calibrate(
        self,
        surface: CalibrationSurface,
        *,
        checkpoint: CalibrationCheckpoint | None = None,
        checkpoint_callback: (
            Callable[
                [
                    dict[tuple[int, ...], dict[str, Any]],
                    set[tuple[tuple[int, ...], int]],
                ],
                None,
            ]
            | None
        ) = None,
        max_new_rows_per_process: int = 0,
        max_producer_epochs_per_process: int = 0,
    ) -> dict[tuple[int, ...], dict[str, Any]]:
        if self.owner.is_pooling_model or self.owner.async_scheduling:
            raise RuntimeError("pre-READY calibration requires synchronous generation")
        if max_new_rows_per_process < 0:
            raise ValueError("calibration process row limit cannot be negative")
        if max_producer_epochs_per_process < 0:
            raise ValueError("calibration producer epoch limit cannot be negative")
        if max_producer_epochs_per_process and checkpoint_callback is None:
            raise ValueError(
                "calibration producer epoch limit requires a checkpoint callback"
            )
        previous_mode = self.scheduler._elastic_restore_mode
        previous_catalog = self.scheduler._elastic_graph_catalog
        working_catalog = {key: dict(row) for key, row in previous_catalog.items()}
        if checkpoint is not None:
            working_catalog.update(
                {key: dict(row) for key, row in checkpoint.catalog.items()}
            )
        previous_coverage = self.scheduler._elastic_graph_catalog_coverage
        previous_short_decode_inventory = getattr(
            self.scheduler, "_elastic_short_decode_inventory", {}
        )
        started = time.monotonic()
        measured: dict[tuple[int, ...], dict[str, Any]] | None = None
        semantic_token_witnesses = dict(surface.semantic_token_witnesses)
        executed_mixed_witnesses = set(
            () if checkpoint is None else checkpoint.mixed_query_witnesses
        )
        initial_complete = {
            key
            for key in surface.required
            if elastic_graph_catalog_row_complete(
                key,
                working_catalog.get(key) or {},
                representation="bounded_exact_hotset",
            )
        }
        producer_epochs = 0

        def checkpoint_progress(
            *, allow_restart: bool, producer_completed: bool = False
        ) -> None:
            nonlocal producer_epochs
            if producer_completed:
                producer_epochs += 1
            complete = {
                key: dict(working_catalog[key])
                for key in surface.required
                if elastic_graph_catalog_row_complete(
                    key,
                    working_catalog.get(key) or {},
                    representation="bounded_exact_hotset",
                )
            }
            if checkpoint_callback is not None:
                observed = {
                    key: dict(working_catalog[key])
                    for key in surface.required
                    if working_catalog.get(key, {}).get("cold_observations", 0) > 0
                }
                checkpoint_callback(observed, executed_mixed_witnesses)
            new_rows = len(set(complete) - initial_complete)
            if (
                allow_restart
                and len(complete) < len(surface.required)
                and (
                    (max_new_rows_per_process and new_rows >= max_new_rows_per_process)
                    or (
                        max_producer_epochs_per_process
                        and producer_epochs >= max_producer_epochs_per_process
                    )
                )
            ):
                raise ElasticCalibrationRestartRequired(
                    "bounded calibration process epoch completed: "
                    f"new_rows={new_rows} total_rows={len(complete)}/"
                    f"{len(surface.required)} producer_epochs={producer_epochs}"
                )

        try:
            # Canonical decode X is selected from the accepted surface, not
            # from the larger raw/effective scheduler cap.  Install this
            # deterministic calibration view before validating any declared
            # key, and restore it on every validation/execution failure.
            self.scheduler._elastic_restore_mode = True
            self.scheduler._elastic_graph_catalog = working_catalog
            self.scheduler._elastic_graph_catalog_coverage = {
                "representation": "bounded_exact_hotset",
                "decode_max_x": surface.decode_max_x,
            }
            rebuild_inventory = getattr(
                self.scheduler, "_rebuild_elastic_short_decode_inventory", None
            )
            if rebuild_inventory is not None:
                rebuild_inventory(surface.decode_max_x)
            self._validate_surface_before_mutation(surface)

            # The restore PIECEWISE row is physical owner evidence for the
            # paired FULL decode lifecycle, not an assertion that the same X
            # can be admitted as a fresh balanced-prefill cohort.  Measure the
            # pair through its actual producer first.  The ordinary required
            # loop will then observe both rows as complete and will not turn
            # the owner-only row into a stricter, unrelated product workload.
            restore_full = next((key for key in surface.restore if key[0] == 1), None)
            if restore_full is not None and any(
                not self._complete(key) for key in surface.restore
            ):
                restore_prefill = next(key for key in surface.restore if key[0] == 0)
                for _ in range(4):
                    if all(self._complete(key) for key in surface.restore):
                        break
                    actual, admitted = self.owner._run_elastic_full_restore_wave(
                        k=restore_full[1],
                        x=restore_full[2],
                        query_len=restore_full[4],
                        serial=self._next_serial(),
                    )
                    if actual != restore_full or admitted != restore_full[2]:
                        raise RuntimeError(
                            "calibration contracted accepted restore boundary: "
                            f"expected={restore_full!r} actual={actual!r} "
                            f"admitted={admitted}"
                        )
                    checkpoint_progress(
                        allow_restart=True,
                        producer_completed=True,
                    )
                    if not self._complete(restore_prefill):
                        prompt_len = self.owner._elastic_restore_prefill_prompt_len()
                        prefill_actual, prefill_admitted = self._balanced_prefill_pair(
                            k=restore_prefill[1],
                            x=restore_prefill[2],
                            m=restore_prefill[2] * prompt_len,
                        )
                        if (
                            prefill_actual != restore_prefill
                            or prefill_admitted != restore_prefill[2]
                        ):
                            raise RuntimeError(
                                "calibration contracted accepted restore prefill: "
                                f"expected={restore_prefill!r} "
                                f"actual={prefill_actual!r} "
                                f"admitted={prefill_admitted}"
                            )
                        checkpoint_progress(
                            allow_restart=True,
                            producer_completed=True,
                        )
                if any(not self._complete(key) for key in surface.restore):
                    raise RuntimeError("calibration restore family did not stabilize")
                checkpoint_progress(allow_restart=True)
                # A process boundary may occur after any successfully drained
                # producer. Partial observations remain identity-bound and the
                # complete pair is still required before later rows execute.
            mixed_by_key: dict[tuple[int, ...], list[int]] = {}
            for key, query_len in surface.mixed_query_witnesses:
                mixed_by_key.setdefault(key, []).append(query_len)
            for key in surface.required:
                # A physical PIECEWISE key does not imply that a balanced-
                # prefill distribution is feasible at the same X/M. Execute a
                # declared semantic producer when its row is reached in the
                # dependency-safe required order. Its stable replays provide
                # physical owner evidence; the generic loop then only fills a
                # row that the specific producer did not complete.
                for query_len in mixed_by_key.get(key, ()):
                    if (key, query_len) in executed_mixed_witnesses:
                        continue
                    live_tokens = semantic_token_witnesses.get(key, key[3])
                    mixed, mixed_admitted = self._mixed_prefill_pair(
                        k=key[1],
                        x=key[2],
                        m=live_tokens,
                        query_len=query_len,
                    )
                    if mixed != key or mixed_admitted != key[2]:
                        raise RuntimeError(
                            "mixed prefill contracted accepted surface: "
                            f"expected={key!r} query_len={query_len} "
                            f"actual={mixed!r} admitted={mixed_admitted}"
                        )
                    executed_mixed_witnesses.add((key, query_len))
                    checkpoint_progress(
                        allow_restart=True,
                        producer_completed=True,
                    )
                for _ in range(4):
                    if self._complete(key):
                        break
                    if key[4]:
                        actual, admitted = self.owner._run_elastic_full_restore_wave(
                            k=key[1],
                            x=key[2],
                            query_len=key[4],
                            serial=self._next_serial(),
                        )
                    else:
                        live_tokens = semantic_token_witnesses.get(key, key[3])
                        actual, admitted = self._balanced_prefill_pair(
                            k=key[1], x=key[2], m=live_tokens
                        )
                        if actual != key or admitted != key[2]:
                            raise RuntimeError(
                                "calibration contracted accepted surface: "
                                f"expected={key!r} actual={actual!r} "
                                f"admitted={admitted}"
                            )
                    if key[4] and (actual != key or admitted != key[2]):
                        raise RuntimeError(
                            "calibration contracted accepted surface: "
                            f"expected={key!r} actual={actual!r} admitted={admitted}"
                        )
                    checkpoint_progress(
                        allow_restart=True,
                        producer_completed=True,
                    )
                if not self._complete(key):
                    raise RuntimeError(
                        f"calibration row did not stabilize: step_key={key!r}"
                    )
                logger.info(
                    "Pre-READY elastic calibration progress: complete=%d/%d key=%s",
                    sum(self._complete(item) for item in surface.required),
                    len(surface.required),
                    key,
                )
                checkpoint_progress(allow_restart=True)
            if executed_mixed_witnesses != set(surface.mixed_query_witnesses):
                raise RuntimeError("calibration left a mixed witness unexecuted")
            measured = {
                key: dict(self.scheduler._elastic_graph_catalog[key])
                for key in surface.required
            }
        finally:
            self.scheduler._elastic_restore_mode = previous_mode
            self.scheduler._elastic_graph_catalog = previous_catalog
            self.scheduler._elastic_graph_catalog_coverage = previous_coverage
            self.scheduler._elastic_short_decode_inventory = (
                previous_short_decode_inventory
            )
        logger.info(
            "Pre-READY elastic calibration measured %d shapes in %.3f s",
            len(surface.required),
            time.monotonic() - started,
        )
        assert measured is not None
        return measured

    def validate_capacity(
        self,
        surface: CalibrationSurface,
        catalog: dict[tuple[int, ...], dict[str, Any]],
    ) -> None:
        graph_bytes = max(
            max(catalog[key]["cold_peak_bytes"], catalog[key]["hot_peak_bytes"])
            for key in surface.restore
        )
        coordinator = self.scheduler.kv_cache_manager.coordinator
        feasible = coordinator.max_elastic_full_context_requests(
            int(self.scheduler._elastic_primary_blocks_per_max_request),
            graph_bytes,
            surface.decode_max_x,
        )
        if feasible < surface.full_context_max_x:
            raise RuntimeError(
                "measured Graph residency violates the accepted full-context "
                f"boundary: required={surface.full_context_max_x} feasible={feasible}"
            )


def calibrate_and_publish_catalog(
    owner: Any,
    surface_payload: dict[str, Any],
    *,
    output_root: Path,
    expected_fingerprint: str | None = None,
    surface_sha256: str | None = None,
    checkpoint_payload: Any = None,
    checkpoint_generation: str | None = None,
    seed_catalog_path: str | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    max_new_rows_per_process: int = 0,
    max_producer_epochs_per_process: int = 0,
) -> CalibrationResult:
    """Measure and atomically publish one exact-runtime catalog."""
    from vllm.v1.worker.elastic_catalog_tool import publish_measured_catalog
    from vllm.v1.worker.startup_plan import (
        compute_elastic_graph_catalog_fingerprint,
    )

    started = time.monotonic()
    scheduler = owner.scheduler
    policy = scheduler._elastic_graph_execution_policy
    if policy is None:
        raise RuntimeError("effective runtime has no elastic Graph policy")
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        owner.vllm_config, scheduler.kv_cache_config
    )
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise RuntimeError(
            "pre-READY calibration runtime identity differs from the expected "
            f"identity: expected={expected_fingerprint} actual={fingerprint}"
        )
    surface = CalibrationSurface.from_payload(
        surface_payload,
        policy_fingerprint=policy.fingerprint,
        configured_k=int(scheduler.num_spec_tokens),
        prefill_k=owner._elastic_restore_prefill_k(int(scheduler.num_spec_tokens)),
        max_num_seqs=int(scheduler.max_num_running_reqs),
        max_num_batched_tokens=(
            owner.vllm_config.scheduler_config.max_num_batched_tokens
        ),
        source_sha256=surface_sha256,
    )
    if checkpoint_payload is None and seed_catalog_path is not None:
        from vllm.v1.worker.startup_plan import (
            load_elastic_graph_catalog,
            load_elastic_graph_catalog_coverage,
        )

        seed_rows = load_elastic_graph_catalog(
            owner.vllm_config, scheduler.kv_cache_config, catalog_path=seed_catalog_path
        )
        seed_coverage = load_elastic_graph_catalog_coverage(
            owner.vllm_config, scheduler.kv_cache_config, catalog_path=seed_catalog_path
        )
        if getattr(seed_rows, "source_sha256", None) != seed_coverage.get(
            "_catalog_source_sha256"
        ):
            raise RuntimeError("calibration seed changed during validation")
        from vllm.v1.core.elastic_price_identity import select_compatible_price_seed

        reusable, witnesses = select_compatible_price_seed(
            seed_rows, seed_coverage, surface
        )
        checkpoint_payload = calibration_checkpoint_payload(
            catalog=reusable,
            mixed_query_witnesses=witnesses,
            fingerprint=fingerprint,
            surface_sha256=surface_sha256,
            process_epochs=0,
            calibration_wall_seconds=0.0,
        )
        checkpoint_generation = scheduler._elastic_admission_controller.generation.value
        logger.info(
            "Elastic calibration seed: reused=%d required=%d source_sha256=%s",
            len(reusable),
            len(surface.required),
            seed_rows.source_sha256,
        )
    checkpoint = parse_calibration_checkpoint(
        checkpoint_payload,
        surface=surface,
        fingerprint=fingerprint,
        surface_sha256=surface_sha256,
    )
    if checkpoint_generation is not None and checkpoint.catalog:
        from dataclasses import replace

        from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes
        from vllm.v1.core.elastic_price_identity import remap_catalog_resident_keys

        # A partial row may name a retained owner measured by a different row.
        # Resolve the complete declared inventory, then retain only observations.
        inventory = {key: checkpoint.catalog.get(key, {}) for key in surface.required}
        rebound = remap_catalog_resident_keys(
            inventory,
            source_generation=checkpoint_generation,
            destination_generation=scheduler._elastic_admission_controller.generation.value,
            max_num_batched_tokens=owner.vllm_config.scheduler_config.max_num_batched_tokens,
            compiled_piecewise_sizes=tuple(
                configured_compiled_piecewise_sizes(owner.vllm_config)
            ),
            policy=policy,
        )
        checkpoint = replace(
            checkpoint, catalog={key: rebound[key] for key in checkpoint.catalog}
        )
    process_epoch = checkpoint.process_epochs + 1

    def publish_checkpoint(
        catalog: dict[tuple[int, ...], dict[str, Any]],
        mixed_query_witnesses: set[tuple[tuple[int, ...], int]],
    ) -> None:
        if checkpoint_callback is None:
            return
        checkpoint_callback(
            calibration_checkpoint_payload(
                catalog=catalog,
                mixed_query_witnesses=mixed_query_witnesses,
                fingerprint=fingerprint,
                surface_sha256=surface_sha256,
                process_epochs=process_epoch,
                calibration_wall_seconds=(
                    checkpoint.wall_seconds + time.monotonic() - started
                ),
            )
        )

    with _catalog_producer_lock(output_root, fingerprint):
        calibrator = ElasticCatalogCalibrator(owner)
        measured = calibrator.calibrate(
            surface,
            checkpoint=checkpoint,
            checkpoint_callback=publish_checkpoint,
            max_new_rows_per_process=max_new_rows_per_process,
            max_producer_epochs_per_process=max_producer_epochs_per_process,
        )
        calibrator.validate_capacity(surface, measured)
        total_wall_seconds = checkpoint.wall_seconds + time.monotonic() - started
        destination = publish_measured_catalog(
            owner.vllm_config,
            scheduler.kv_cache_config,
            measured,
            required=surface.required,
            restore=surface.restore,
            semantic_token_witnesses=surface.semantic_token_witnesses,
            mixed_query_witnesses=surface.mixed_query_witnesses,
            decode_max_x=surface.decode_max_x,
            mixed_max_x=surface.mixed_max_x,
            full_context_max_x=surface.full_context_max_x,
            calibration_wall_seconds=total_wall_seconds,
            output_root=output_root,
            source_surface_schema=surface.source_schema,
            source_surface_required_shapes=surface.source_required_shapes,
            surface_migration_contract=surface.migration_contract,
            surface_full_aliases=surface.full_aliases,
            migrated_source_keys=surface.migrated_source_keys,
            source_surface_sha256=surface.source_sha256,
            owner_evidence_keys=surface.owner_evidence_keys,
        )
    return CalibrationResult(
        destination=destination,
        fingerprint=fingerprint,
        surface=surface,
        measured_shapes=len(measured),
        wall_seconds=total_wall_seconds,
    )
