# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single owner for elastic KV/scheduler startup ordering.

The callbacks keep model-specific KV initialization and EngineCore lifecycle
setup at their natural owners, while this module makes the order between KV,
policy, scheduler generation, workers, and pre-READY executable restore
explicit rather than documentary.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.core.elastic_graph import GraphExecutionPolicy
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


def _write_calibration_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_measurement_request(fingerprint: str, surface_sha256: str) -> None:
    from vllm.v1.core.elastic_price_identity import validate_calibration_request

    request_path = os.environ.get("AG2_VLLM_ELASTIC_CALIBRATION_REQUEST", "")
    try:
        request = Path(request_path)
        if not request_path or request.is_symlink() or not request.is_file():
            raise ValueError("measurement request must be a regular file")
        request_payload = json.loads(request.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "calibration not authorized by a measurement decision; set "
            "AG2_VLLM_ELASTIC_CALIBRATION_REQUEST after compatibility review"
        ) from error
    validate_calibration_request(
        request_payload, fingerprint=fingerprint, surface_sha256=surface_sha256
    )


def _auto_profile_runtime_catalog(owner: Any) -> None:
    """Build, price, publish and activate the effective pre-READY surface."""
    from vllm import envs
    from vllm.v1.engine.elastic_calibrator import (
        _catalog_producer_lock,
        derive_runtime_calibration_surface,
    )
    from vllm.v1.engine.elastic_memory_profile import profile_allocation_catalog
    from vllm.v1.worker.elastic_catalog_tool import publish_measured_catalog
    from vllm.v1.worker.startup_plan import (
        _elastic_graph_catalog_path,
        compute_elastic_graph_catalog_fingerprint,
        load_elastic_graph_catalog,
        load_elastic_graph_catalog_coverage,
    )

    scheduler = owner.scheduler
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        owner.vllm_config, scheduler.kv_cache_config
    )
    destination = Path(_elastic_graph_catalog_path(fingerprint))
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            "automatic startup calibration refuses an unaccepted canonical "
            f"destination: {destination}"
        )
    payload, surface = derive_runtime_calibration_surface(owner)
    surface_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    surface_sha256 = hashlib.sha256(surface_bytes).hexdigest()
    catalog_dir = Path(envs.VLLM_CACHE_ROOT) / "elastic_graph_catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    surface_path = catalog_dir / f"auto_surface_{fingerprint}.json"
    if surface_path.exists():
        if surface_path.is_symlink() or surface_path.read_bytes() != surface_bytes:
            raise RuntimeError("runtime-derived calibration surface identity differs")
    else:
        temporary = surface_path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_bytes(surface_bytes)
        os.replace(temporary, surface_path)
    receipt_path = Path(
        os.environ.get("AG2_VLLM_ELASTIC_CALIBRATION_RECEIPT")
        or catalog_dir / f"auto_calibration_{fingerprint}.json"
    )
    previous_attempts = 0
    if receipt_path.exists() and not receipt_path.is_symlink():
        try:
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            previous_attempts = int(previous.get("attempt", 0))
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(
                "automatic calibration receipt cannot be resumed"
            ) from error
    receipt: dict[str, Any] = {
        "schema": "ag2-elastic-startup-calibration-v1",
        "stage": "profiling",
        "attempt": previous_attempts + 1,
        "fingerprint": fingerprint,
        "surface": str(surface_path),
        "surface_sha256": surface_sha256,
        "policy_fingerprint": scheduler._elastic_graph_execution_policy.fingerprint,
        "required_shapes": len(surface.required) + len(surface.serving_hotset),
        "physical_samples": [],
    }
    _write_calibration_receipt(receipt_path, receipt)
    started = time.monotonic()

    def progress(samples: list[dict[str, Any]], next_key: Any, wall: float) -> None:
        receipt.update(
            stage="profiling",
            physical_samples=samples,
            next_physical_key=next_key,
            profiling_wall_seconds=wall,
        )
        _write_calibration_receipt(receipt_path, receipt)

    try:
        with _catalog_producer_lock(Path(envs.VLLM_CACHE_ROOT), fingerprint):
            rows = profile_allocation_catalog(owner, surface, progress=progress)
            publish_rows = dict(rows)
            for alias in surface.serving_hotset:
                if alias in publish_rows:
                    raise RuntimeError("serving hotset alias duplicates a measured row")
                publish_rows[alias] = dict(next(iter(rows.values())))
            required = tuple(
                sorted(set(surface.required).union(surface.serving_hotset))
            )
            published = publish_measured_catalog(
                owner.vllm_config,
                scheduler.kv_cache_config,
                publish_rows,
                required=required,
                restore=surface.restore,
                decode_max_x=surface.decode_max_x,
                mixed_max_x=surface.mixed_max_x,
                full_context_max_x=surface.full_context_max_x,
                calibration_wall_seconds=time.monotonic() - started,
                output_root=Path(envs.VLLM_CACHE_ROOT),
                semantic_token_witnesses=surface.semantic_token_witnesses,
                mixed_query_witnesses=surface.mixed_query_witnesses,
                restore_decode=surface.restore_decode,
            )
        if published != destination:
            raise RuntimeError("automatic calibration published a noncanonical catalog")
        catalog = load_elastic_graph_catalog(
            owner.vllm_config, scheduler.kv_cache_config
        )
        coverage = load_elastic_graph_catalog_coverage(
            owner.vllm_config, scheduler.kv_cache_config
        )
        scheduler.activate_elastic_graph_catalog(catalog, coverage)
        receipt.update(
            stage="complete",
            destination=str(published),
            measured_shapes=len(publish_rows),
            wall_seconds=time.monotonic() - started,
        )
        _write_calibration_receipt(receipt_path, receipt)
        logger.warning(
            "Elastic catalog profiled automatically before READY: "
            "fingerprint=%s shapes=%d destination=%s",
            fingerprint,
            len(publish_rows),
            published,
        )
    except BaseException as error:
        receipt.update(
            stage="failed",
            error={"type": type(error).__name__, "message": str(error)},
            wall_seconds=time.monotonic() - started,
        )
        _write_calibration_receipt(receipt_path, receipt)
        raise


def _auto_calibrate_missing_catalog(owner: Any) -> None:
    from vllm import envs
    from vllm.v1.core.elastic_catalog import load_calibration_surface_with_digest
    from vllm.v1.engine.elastic_calibrator import (
        ElasticCalibrationRestartRequired,
        calibrate_and_publish_catalog,
    )
    from vllm.v1.worker.startup_plan import (
        compute_elastic_graph_catalog_fingerprint,
        load_elastic_graph_catalog,
        load_elastic_graph_catalog_coverage,
    )

    surface_value = os.environ.get("AG2_VLLM_ELASTIC_CALIBRATION_SURFACE", "")
    if not surface_value:
        _auto_profile_runtime_catalog(owner)
        return
    surface_path = Path(surface_value)
    surface_payload, surface_sha256 = load_calibration_surface_with_digest(surface_path)
    scheduler = owner.scheduler
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        owner.vllm_config, scheduler.kv_cache_config
    )
    _require_measurement_request(fingerprint, surface_sha256)
    from vllm.v1.worker.startup_plan import _elastic_graph_catalog_path

    canonical_destination = Path(_elastic_graph_catalog_path(fingerprint))
    try:
        canonical_destination.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise RuntimeError(
            "automatic elastic calibration cannot prove the canonical "
            f"destination is absent: {canonical_destination}"
        ) from error
    else:
        raise RuntimeError(
            "automatic elastic calibration refuses an existing canonical "
            "destination; validate or quarantine it before retrying: "
            f"{canonical_destination}"
        )
    receipt_path = Path(
        os.environ.get("AG2_VLLM_ELASTIC_CALIBRATION_RECEIPT")
        or Path(envs.VLLM_CACHE_ROOT)
        / "elastic_graph_catalog"
        / f"auto_calibration_{fingerprint}.json"
    )
    resolved_surface = surface_path.resolve(strict=False)
    resolved_canonical = canonical_destination.resolve(strict=False)
    resolved_receipt = receipt_path.resolve(strict=False)
    if len({resolved_surface, resolved_canonical, resolved_receipt}) != 3:
        raise RuntimeError(
            "elastic calibration surface, canonical catalog and receipt must "
            "resolve to three distinct paths"
        )
    seed_path = os.environ.get("AG2_VLLM_ELASTIC_CALIBRATION_SEED_CATALOG", "")
    if seed_path and Path(seed_path).resolve(strict=False) in {
        resolved_surface,
        resolved_canonical,
        resolved_receipt,
    }:
        raise RuntimeError(
            "calibration seed must be distinct from mutable destinations and surface"
        )
    checkpoint_payload = None
    from vllm.v1.worker.startup_plan import elastic_catalog_owner_generation

    current_owner_generation = elastic_catalog_owner_generation(
        owner.vllm_config, scheduler.kv_cache_config
    )
    checkpoint_generation: str | None = current_owner_generation
    if receipt_path.exists():
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise RuntimeError(
                f"elastic calibration receipt is not a regular file: {receipt_path}"
            )
        try:
            prior_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "elastic calibration receipt cannot be resumed"
            ) from error
        if (
            not isinstance(prior_receipt, dict)
            or prior_receipt.get("schema") != "ag2-elastic-auto-calibration-v1"
            or prior_receipt.get("fingerprint") != fingerprint
            or prior_receipt.get("surface_sha256") != surface_sha256
        ):
            raise RuntimeError("elastic calibration receipt identity differs")
        if prior_receipt.get("stage") not in {"checkpointed", "calibrating"}:
            raise RuntimeError(
                "failed or terminal calibration requires a new reviewed decision"
            )
        checkpoint_payload = prior_receipt.get("checkpoint")
        checkpoint_generation = prior_receipt.get("resident_owner_generation")
        if not isinstance(checkpoint_generation, str) or not checkpoint_generation:
            raise RuntimeError(
                "legacy calibration checkpoint needs explicit owner binding"
            )
    raw_process_limit = os.environ.get(
        "AG2_VLLM_ELASTIC_CALIBRATION_MAX_NEW_ROWS_PER_PROCESS", "0"
    )
    try:
        process_limit = int(raw_process_limit)
    except ValueError as error:
        raise RuntimeError(
            "elastic calibration process row limit is invalid"
        ) from error
    if process_limit < 0:
        raise RuntimeError("elastic calibration process row limit is invalid")
    raw_producer_limit = os.environ.get(
        "AG2_VLLM_ELASTIC_CALIBRATION_MAX_PRODUCER_EPOCHS_PER_PROCESS", "0"
    )
    try:
        producer_limit = int(raw_producer_limit)
    except ValueError as error:
        raise RuntimeError(
            "elastic calibration producer epoch limit is invalid"
        ) from error
    if producer_limit < 0:
        raise RuntimeError("elastic calibration producer epoch limit is invalid")
    receipt: dict[str, Any] = {
        "schema": "ag2-elastic-auto-calibration-v1",
        "stage": "calibrating",
        "fingerprint": fingerprint,
        "surface": str(resolved_surface),
        "surface_sha256": surface_sha256,
        "max_new_rows_per_process": process_limit,
        "max_producer_epochs_per_process": producer_limit,
        "resident_owner_generation": checkpoint_generation,
    }
    if checkpoint_payload is not None:
        receipt["checkpoint"] = checkpoint_payload
    _write_calibration_receipt(receipt_path, receipt)

    def record_checkpoint(payload: dict[str, Any]) -> None:
        receipt["checkpoint"] = payload
        receipt["resident_owner_generation"] = current_owner_generation
        receipt["stage"] = "calibrating"
        _write_calibration_receipt(receipt_path, receipt)

    try:
        result = calibrate_and_publish_catalog(
            owner,
            surface_payload,
            output_root=Path(envs.VLLM_CACHE_ROOT),
            expected_fingerprint=fingerprint,
            surface_sha256=surface_sha256,
            checkpoint_payload=checkpoint_payload,
            checkpoint_generation=checkpoint_generation,
            seed_catalog_path=os.environ.get(
                "AG2_VLLM_ELASTIC_CALIBRATION_SEED_CATALOG"
            )
            or None,
            checkpoint_callback=record_checkpoint,
            max_new_rows_per_process=process_limit,
            max_producer_epochs_per_process=producer_limit,
        )
        catalog = load_elastic_graph_catalog(
            owner.vllm_config, scheduler.kv_cache_config
        )
        coverage = load_elastic_graph_catalog_coverage(
            owner.vllm_config, scheduler.kv_cache_config
        )
        scheduler.activate_elastic_graph_catalog(catalog, coverage)
        receipt.update(
            stage="complete",
            destination=str(result.destination),
            measured_shapes=result.measured_shapes,
            wall_seconds=result.wall_seconds,
        )
        _write_calibration_receipt(receipt_path, receipt)
        logger.warning(
            "Elastic catalog calibrated automatically before READY: "
            "fingerprint=%s shapes=%d destination=%s",
            result.fingerprint,
            result.measured_shapes,
            result.destination,
        )
    except ElasticCalibrationRestartRequired as error:
        receipt.update(
            stage="checkpointed",
            restart_required=True,
            restart_reason=str(error),
        )
        _write_calibration_receipt(receipt_path, receipt)
        raise
    except BaseException as error:
        receipt.update(
            stage="failed",
            error={"type": type(error).__name__, "message": str(error)},
        )
        _write_calibration_receipt(receipt_path, receipt)
        raise


@dataclass(frozen=True)
class ElasticRuntimePreparation:
    kv_cache_config: KVCacheConfig
    structured_output_manager: StructuredOutputManager
    scheduler: SchedulerInterface
    hash_block_size: int
    generation_receipt: dict[str, Any] | None


def resolve_elastic_graph_execution_policy(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    collective_rpc: Callable[..., list[Any]],
) -> str | None:
    additional_config = vllm_config.additional_config
    elastic_enabled = bool(
        isinstance(additional_config, dict)
        and additional_config.get("elastic_gdn_backing", False)
        and not vllm_config.model_config.enforce_eager
        and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    )
    if not elastic_enabled:
        return None
    payloads = collective_rpc("get_elastic_graph_execution_policy")
    if not payloads or any(payload is None for payload in payloads):
        raise RuntimeError(
            "elastic Graph execution policy is absent on one or more ranks"
        )
    policies = [GraphExecutionPolicy.from_payload(payload) for payload in payloads]
    fingerprints = {policy.fingerprint for policy in policies}
    if len(fingerprints) != 1:
        raise RuntimeError(
            "elastic Graph execution policy differs across ranks: "
            f"{sorted(fingerprints)!r}"
        )
    policy = policies[0]
    kv_cache_config.elastic_graph_execution_policy = policy.to_payload()
    logger.info(
        "Elastic Graph execution policy resolved: fingerprint=%s "
        "verifier=%s verifier_configuration=%s math=%s owners=%s",
        policy.fingerprint,
        policy.verifier_contract,
        policy.verifier_configuration,
        policy.math_contract,
        tuple(
            (owner.owner, owner.full_query_lens, owner.piecewise_mode)
            for owner in policy.owners
        ),
    )
    return policy.fingerprint


def synchronize_elastic_runtime_generation(
    scheduler: SchedulerInterface,
    collective_rpc: Callable[..., list[Any]],
) -> dict[str, Any] | None:
    if not getattr(scheduler, "elastic_on_demand_graphs", False):
        return None
    controller = getattr(scheduler, "_elastic_admission_controller", None)
    if controller is None:
        return None
    generation = controller.generation
    worker_generations = collective_rpc(
        "set_elastic_runtime_generation", args=(generation.value,)
    )
    if not worker_generations or set(worker_generations) != {generation.value}:
        raise RuntimeError(
            "elastic workers did not accept the scheduler runtime generation: "
            f"scheduler={generation.value!r} workers={worker_generations!r}"
        )
    logger.info(
        "Elastic runtime generation synchronized after KV initialization: %s",
        generation.value,
    )
    return {"scheduler": generation.value, "workers": worker_generations}


def prepare_elastic_runtime(
    *,
    vllm_config: VllmConfig,
    initialize_kv_cache: Callable[[], KVCacheConfig],
    collective_rpc: Callable[..., list[Any]],
    include_finished_set: bool,
    log_stats: bool,
    structured_output_manager_factory: (
        Callable[[], StructuredOutputManager] | None
    ) = None,
) -> ElasticRuntimePreparation:
    """Prepare the post-KV scheduler epoch in the production order."""
    kv_cache_config = initialize_kv_cache()
    resolve_elastic_graph_execution_policy(vllm_config, kv_cache_config, collective_rpc)
    manager = (
        StructuredOutputManager(vllm_config)
        if structured_output_manager_factory is None
        else structured_output_manager_factory()
    )
    scheduler_cls = vllm_config.scheduler_config.get_scheduler_cls()

    if (
        not kv_cache_config.kv_cache_groups
        and vllm_config.scheduler_config.enable_chunked_prefill
    ):
        logger.warning("Disabling chunked prefill for model without KVCache")
        vllm_config.scheduler_config.enable_chunked_prefill = False

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=manager,
        include_finished_set=include_finished_set,
        log_stats=log_stats,
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )
    generation_receipt = synchronize_elastic_runtime_generation(
        scheduler, collective_rpc
    )
    return ElasticRuntimePreparation(
        kv_cache_config=kv_cache_config,
        structured_output_manager=manager,
        scheduler=scheduler,
        hash_block_size=hash_block_size,
        generation_receipt=generation_receipt,
    )


def replay_allocation_profile_prefix_trace(
    worker: Any, step_key: tuple[int, int, int, int, int]
) -> dict:
    """Replay cumulative physical-owner prefixes and name the first bad edge."""
    from vllm.v1.core import elastic_graph
    from vllm.v1.core.elastic_graph import GraphExecutionPolicy, RuntimeGeneration
    from vllm.v1.engine.elastic_memory_profile import replay_allocation_profile

    runner = worker.model_runner
    managers = {
        manager.dynamic_graph_owner: manager
        for manager in runner._dynamic_graph_working_set().managers
    }
    policy = GraphExecutionPolicy.from_payload(
        worker.get_elastic_graph_execution_policy()
    )
    keys = elastic_graph.resolve_step_physical_keys(
        step_key,
        generation=RuntimeGeneration(next(iter(managers.values())).runtime_generation),
        max_num_batched_tokens=runner.max_num_tokens,
        policy=policy,
    )
    original_resolver = elastic_graph.resolve_step_physical_keys
    final_receipt: dict | None = None
    try:
        for prefix_length in range(1, len(keys) + 1):
            prefix = keys[:prefix_length]
            elastic_graph.resolve_step_physical_keys = (
                lambda *args, _prefix=prefix, **kwargs: _prefix
            )
            try:
                final_receipt = replay_allocation_profile(worker, step_key)
            except Exception as error:
                owners = tuple(key.logical.owner for key in prefix)
                raise RuntimeError(
                    "allocation restore replay failed after physical-owner "
                    f"prefix={owners!r} step_key={step_key!r}"
                ) from error
    finally:
        elastic_graph.resolve_step_physical_keys = original_resolver
    assert final_receipt is not None
    return final_receipt


def restore_profiled_graph_carrier(owner: Any) -> None:
    """Restore measured native owners without manufacturing model requests."""
    from vllm.v1.engine.elastic_memory_profile import replay_allocation_profile

    replay = (
        replay_allocation_profile_prefix_trace
        if os.environ.get("AG2_VLLM_ELASTIC_RESTORE_PREFIX_TRACE", "0") == "1"
        else replay_allocation_profile
    )

    scheduler = owner.scheduler
    coverage = scheduler._elastic_graph_catalog_coverage
    x = coverage["decode_max_x"]
    if (
        coverage.get("serving_carrier_contract")
        != "retained-terminal-mtp-no-cold-serving-v1"
        or coverage.get("serving_carrier_owner") != "mtp_decode"
        or type(x) is not int
        or not 1 <= x <= scheduler.max_num_running_reqs
        or scheduler.has_unfinished_requests()
    ):
        raise RuntimeError("invalid request-free startup carrier contract")
    previous_mode = scheduler._elastic_restore_mode
    scheduler._elastic_restore_mode = True
    try:
        owner._reclaim_elastic_restore_hotset_before_wave()
        for query_len in (1, scheduler.num_spec_tokens + 1):
            # Preserve semantic q for the independent MTP prefill manager,
            # even when the target's catalog key is token-major PIECEWISE.
            key = (0, scheduler.num_spec_tokens, x, x * query_len, query_len)
            owner._prepare_elastic_restore_capture(key)
            scheduler.assert_elastic_restore_captures_hot((key,))
            receipts = owner.collective_rpc(replay, args=(key,))
            if not receipts or any(
                receipt["source_reads"] or receipt["stable_replays"] < 1
                for receipt in receipts
            ):
                raise RuntimeError("request-free carrier replay failed its profile")
            carriers = scheduler.resolve_elastic_serving_carrier_physical_keys((key,))
            retention = scheduler.retain_elastic_restore_captures((key,))
            try:
                # This publishes the serving lease and consumes the temporary
                # restore retention before any request-free reclaim is legal.
                scheduler.promote_elastic_restore_retention_to_serving(
                    retention, carriers
                )
            finally:
                if scheduler._elastic_restore_retention_id == retention:
                    scheduler.release_elastic_restore_retention(retention)
            owner._reclaim_elastic_restore_hotset_before_wave()
        hotset_steps = tuple(
            tuple(key) for key in coverage.get("serving_hotset_step_keys", ())
        )
        configured_hotset = tuple(getattr(scheduler, "_elastic_serving_hotset_xs", ()))
        if configured_hotset and not hotset_steps:
            raise RuntimeError("profiled startup omitted the configured hotset")
        for key in hotset_steps:
            owner._prepare_elastic_restore_capture(key)
            scheduler.assert_elastic_restore_captures_hot((key,))
            receipts = owner.collective_rpc(replay, args=(key,))
            if not receipts or any(
                receipt["source_reads"] or receipt["stable_replays"] < 1
                for receipt in receipts
            ):
                raise RuntimeError("request-free hotset replay failed its profile")
            physical_keys = scheduler._resolve_elastic_step_physical_keys(key)
            if not physical_keys:
                continue
            retention = scheduler.retain_elastic_restore_captures((key,))
            try:
                scheduler.promote_elastic_restore_retention_to_serving(
                    retention, physical_keys
                )
            finally:
                if scheduler._elastic_restore_retention_id == retention:
                    scheduler.release_elastic_restore_retention(retention)
            owner._reclaim_elastic_restore_hotset_before_wave()
        if not scheduler._elastic_serving_carrier_keys or any(
            not scheduler._elastic_admission_controller.entries[key].hot
            for key in scheduler._elastic_serving_carrier_keys
        ):
            raise RuntimeError("request-free startup lost its serving carrier")
        scheduler.max_num_running_reqs = coverage["mixed_max_x"]
        logger.info(
            "Restored profiled MTP carrier without model requests: "
            "X=%d owners=%d hotset_shapes=%d",
            x,
            len(scheduler._elastic_serving_carrier_keys),
            len(hotset_steps),
        )
    finally:
        scheduler._elastic_restore_mode = previous_mode


def preload_full_expert_source(worker: Any) -> list[dict[str, Any]]:
    """Load and lock a full-capacity expert source after executable restore.

    Partial caches retain demand loading. No HOT placement or model state is
    changed, and repeated calls must perform no further archive reads.
    """
    import time

    results = []
    state = worker.model_runner.model_state
    for provider in getattr(state, "_native_providers", ()):
        if provider.stream_path is None:
            continue
        source = provider.bank.source
        cpu = provider.stream_path.cpu
        before = cpu.source_stats()
        total = source.layers * source.experts
        if before["capacity"] < total:
            continue
        if before["capacity"] != total or provider.admission.exclusive_ram:
            raise RuntimeError("full expert source has incompatible ownership")
        registration = provider.admission.promotion.registration
        resident = set(cpu.resident_keys())
        started = time.monotonic()
        for layer in range(source.layers):
            pending = [
                e
                for e in range(source.experts)
                if layer * source.experts + e not in resident
            ]
            for offset in range(0, len(pending), 32):
                cpu.load_rows(layer, pending[offset : offset + 32])
            if layer % 8 == 7 or layer + 1 == source.layers:
                logger.info(
                    "EXPERT_SOURCE_PRELOAD rank=%d layers=%d/%d rows=%d",
                    worker.rank,
                    layer + 1,
                    source.layers,
                    cpu.source_stats()["rows"],
                )
        after = cpu.source_stats()
        if (
            set(cpu.resident_keys()) != set(range(total))
            or after["rows"] != total
            or after["evictions"] != before["evictions"]
        ):
            raise RuntimeError("full expert source did not retain every row")
        registration.ensure([registration.address], [registration.nbytes])
        if registration.registered_bytes != registration.nbytes:
            raise RuntimeError("full expert source is not completely page locked")
        results.append(
            dict(
                rank=worker.rank,
                rows=total,
                registered_bytes=registration.registered_bytes,
                read_bytes=after["read_bytes"] - before["read_bytes"],
                wall_s=time.monotonic() - started,
            )
        )
    return results


def complete_elastic_startup(owner: Any) -> str:
    """Restore an exact sealed elastic runtime and publish its residency."""
    from vllm.v1.engine.elastic_calibrator import ElasticCalibrationRestartRequired

    scheduler = owner.scheduler
    outcome = "not_elastic"
    if getattr(scheduler, "elastic_on_demand_graphs", False) and getattr(
        scheduler, "_elastic_require_catalog", False
    ):
        try:
            if not scheduler._elastic_graph_catalog:
                if not getattr(scheduler, "_elastic_auto_calibrate", False):
                    raise RuntimeError(
                        "required elastic CUDA Graph catalog is absent; serving "
                        "will not calibrate implicitly. Resolve price compatibility "
                        "and catalog location first; only proven producer changes "
                        "or uncovered forms justify a reviewed measurement request"
                    )
                _auto_calibrate_missing_catalog(owner)
            representation = scheduler._elastic_graph_catalog_coverage.get(
                "representation", "pinned_full_family"
            )
            if representation == "bounded_exact_hotset":
                from vllm.v1.core.elastic_memory_profile import CONTRACT

                if (
                    scheduler._elastic_graph_catalog_coverage.get(
                        "memory_evidence_contract"
                    )
                    == CONTRACT
                ):
                    restore_profiled_graph_carrier(owner)
                else:
                    owner._restore_elastic_bounded_hotset()
            else:
                owner._restore_elastic_pinned_full_family()
            vllm_config = getattr(owner, "vllm_config", None)
            additional_config = getattr(vllm_config, "additional_config", None)
            if isinstance(additional_config, dict) and additional_config.get(
                "flashnext_native_experts"
            ):
                source_rows = owner.collective_rpc(preload_full_expert_source)
                logger.info("EXPERT_SOURCE_PRE_READY: %s", source_rows)
        except ElasticCalibrationRestartRequired:
            logger.warning(
                "ELASTIC_CALIBRATION_RESTART_REQUIRED: bounded pre-READY "
                "calibration checkpointed and requires a clean process epoch"
            )
            owner._shutdown_failed_elastic_startup()
            raise
        except Exception:
            logger.critical(
                "ELASTIC_PRE_READY_RESTORE_FAILED: sealed serving startup did "
                "not restore its complete carrier before READY",
                exc_info=True,
            )
            owner._shutdown_failed_elastic_startup()
            raise
        outcome = "restored"

    if getattr(scheduler, "elastic_on_demand_graphs", False):
        owner._synchronize_elastic_startup_residency()
    return outcome
