# SPDX-License-Identifier: Apache-2.0
"""Pre-READY producer for a measured elastic CUDA Graph catalog."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from vllm.logger import init_logger
from vllm.v1.core.elastic_catalog import elastic_graph_catalog_row_complete

logger = init_logger(__name__)


@dataclass(frozen=True)
class CalibrationSurface:
    required: tuple[tuple[int, int, int, int, int], ...]
    restore: tuple[tuple[int, int, int, int, int], ...]
    decode_max_x: int
    mixed_max_x: int
    full_context_max_x: int

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        policy_fingerprint: str,
        configured_k: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> CalibrationSurface:
        coverage = payload.get("coverage")
        if not isinstance(coverage, dict):
            raise RuntimeError("calibration surface has no coverage object")
        if coverage.get("representation") != "bounded_exact_hotset":
            raise RuntimeError("calibration requires a bounded exact surface")
        if coverage.get("graph_execution_policy_fingerprint") != policy_fingerprint:
            raise RuntimeError("calibration surface policy does not match runtime")

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
            if len(set(result)) != len(result):
                raise RuntimeError(f"calibration surface has duplicate {name}")
            return tuple(sorted(result))  # type: ignore[return-value]

        required = keys("required_step_keys")
        restore = keys("restore_step_keys")
        if not required or not restore or not set(restore).issubset(required):
            raise RuntimeError("calibration surface has invalid restore coverage")
        if (
            len(restore) != 2
            or sum(key[0] == 0 for key in restore) != 1
            or sum(key[0] == 1 for key in restore) != 1
        ):
            raise RuntimeError(
                "calibration surface requires one FULL/PIECEWISE restore pair"
            )
        if any(
            key[1] != configured_k
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
        return cls(
            required=required,
            restore=restore,
            decode_max_x=decode_max_x,
            mixed_max_x=mixed_max_x,
            full_context_max_x=full_context_max_x,
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
    ) -> dict[tuple[int, ...], dict[str, int]]:
        if self.owner.is_pooling_model or self.owner.async_scheduling:
            raise RuntimeError("pre-READY calibration requires synchronous generation")
        configured_k = int(self.scheduler.num_spec_tokens)
        previous_mode = self.scheduler._elastic_restore_mode
        previous_catalog = dict(self.scheduler._elastic_graph_catalog)
        previous_coverage = self.scheduler._elastic_graph_catalog_coverage
        self.scheduler._elastic_restore_mode = True
        self.scheduler._elastic_graph_catalog_coverage = {
            "representation": "bounded_exact_hotset"
        }
        started = time.monotonic()
        measured: dict[tuple[int, ...], dict[str, int]] | None = None
        try:
            for key in surface.required:
                if not self.scheduler._resolve_elastic_step_physical_keys(key):
                    raise RuntimeError(
                        "calibration surface contains a compiled-only shape: "
                        f"step_key={key!r}"
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
                        actual, admitted = self._balanced_prefill_pair(
                            k=key[1], x=key[2], m=key[3]
                        )
                        if actual != key or admitted != key[2]:
                            raise RuntimeError(
                                "calibration contracted accepted surface: "
                                f"expected={key!r} actual={actual!r} "
                                f"admitted={admitted}"
                            )
                        for query_len in (1, 1 + configured_k):
                            if key[2] >= 2 and key[3] > (key[2] - 1) * query_len:
                                mixed, mixed_admitted = self._mixed_prefill_pair(
                                    k=key[1],
                                    x=key[2],
                                    m=key[3],
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
        catalog: dict[tuple[int, ...], dict[str, int]],
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
        max_num_seqs=int(scheduler.max_num_running_reqs),
        max_num_batched_tokens=(
            owner.vllm_config.scheduler_config.max_num_batched_tokens
        ),
    )
    calibrator = ElasticCatalogCalibrator(owner)
    measured = calibrator.calibrate(surface)
    calibrator.validate_capacity(surface, measured)
    destination = publish_measured_catalog(
        owner.vllm_config,
        scheduler.kv_cache_config,
        measured,
        required=surface.required,
        restore=surface.restore,
        decode_max_x=surface.decode_max_x,
        mixed_max_x=surface.mixed_max_x,
        full_context_max_x=surface.full_context_max_x,
        calibration_wall_seconds=time.monotonic() - started,
        output_root=output_root,
    )
    return CalibrationResult(
        destination=destination,
        fingerprint=fingerprint,
        surface=surface,
        measured_shapes=len(measured),
        wall_seconds=time.monotonic() - started,
    )
