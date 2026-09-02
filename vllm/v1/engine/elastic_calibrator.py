# SPDX-License-Identifier: Apache-2.0
"""Pre-READY producer for a measured elastic CUDA Graph catalog."""

from __future__ import annotations

import fcntl
import os
import time
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
        return cls(
            required=required,
            restore=restore,
            decode_max_x=decode_max_x,
            mixed_max_x=mixed_max_x,
            full_context_max_x=full_context_max_x,
            semantic_token_witnesses=tuple(sorted(semantic_token_witnesses.items())),
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
                request_ids=base_ids, step_keys=declared
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
        self, surface: CalibrationSurface
    ) -> dict[tuple[int, ...], dict[str, Any]]:
        if self.owner.is_pooling_model or self.owner.async_scheduling:
            raise RuntimeError("pre-READY calibration requires synchronous generation")
        configured_k = int(self.scheduler.num_spec_tokens)
        previous_mode = self.scheduler._elastic_restore_mode
        previous_catalog = self.scheduler._elastic_graph_catalog
        working_catalog = {key: dict(row) for key, row in previous_catalog.items()}
        previous_coverage = self.scheduler._elastic_graph_catalog_coverage
        previous_short_decode_inventory = getattr(
            self.scheduler, "_elastic_short_decode_inventory", {}
        )
        started = time.monotonic()
        measured: dict[tuple[int, ...], dict[str, Any]] | None = None
        semantic_token_witnesses = dict(surface.semantic_token_witnesses)
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
            for key in surface.required:
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
                        for query_len in (1, 1 + configured_k):
                            if key[2] >= 2 and live_tokens > (key[2] - 1) * query_len:
                                mixed, mixed_admitted = self._mixed_prefill_pair(
                                    k=key[1],
                                    x=key[2],
                                    m=live_tokens,
                                    query_len=query_len,
                                )
                                if mixed != key or mixed_admitted != key[2]:
                                    raise RuntimeError(
                                        "mixed prefill contracted accepted surface"
                                    )
                    if key[4] and (actual != key or admitted != key[2]):
                        raise RuntimeError(
                            "calibration contracted accepted surface: "
                            f"expected={key!r} actual={actual!r} admitted={admitted}"
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
    with _catalog_producer_lock(output_root, fingerprint):
        calibrator = ElasticCatalogCalibrator(owner)
        measured = calibrator.calibrate(surface)
        calibrator.validate_capacity(surface, measured)
        destination = publish_measured_catalog(
            owner.vllm_config,
            scheduler.kv_cache_config,
            measured,
            required=surface.required,
            restore=surface.restore,
            semantic_token_witnesses=surface.semantic_token_witnesses,
            decode_max_x=surface.decode_max_x,
            mixed_max_x=surface.mixed_max_x,
            full_context_max_x=surface.full_context_max_x,
            calibration_wall_seconds=time.monotonic() - started,
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
        wall_seconds=time.monotonic() - started,
    )
