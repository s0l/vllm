# SPDX-License-Identifier: Apache-2.0
"""Single owner for elastic KV/scheduler startup ordering.

The callbacks keep model-specific KV initialization and EngineCore lifecycle
setup at their natural owners, while this module makes the order between KV,
policy, scheduler generation, workers, and pre-READY executable restore
explicit rather than documentary.
"""

from __future__ import annotations

import json
import os
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


def _auto_calibrate_missing_catalog(owner: Any) -> None:
    from vllm import envs
    from vllm.v1.core.elastic_catalog import load_sealed_catalog_with_digest
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
        raise RuntimeError(
            "automatic elastic calibration requires "
            "AG2_VLLM_ELASTIC_CALIBRATION_SURFACE"
        )
    surface_path = Path(surface_value)
    surface_payload, surface_sha256 = load_sealed_catalog_with_digest(
        surface_path,
        require_migration=False,
        allow_previous_schema=True,
    )
    scheduler = owner.scheduler
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        owner.vllm_config, scheduler.kv_cache_config
    )
    canonical_destination = (
        Path(envs.VLLM_CACHE_ROOT)
        / "elastic_graph_catalog"
        / f"elastic_graph_catalog_{fingerprint}.json"
    )
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
    checkpoint_payload = None
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
        checkpoint_payload = prior_receipt.get("checkpoint")
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
    }
    if checkpoint_payload is not None:
        receipt["checkpoint"] = checkpoint_payload
    _write_calibration_receipt(receipt_path, receipt)

    def record_checkpoint(payload: dict[str, Any]) -> None:
        receipt["checkpoint"] = payload
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
                    raise RuntimeError("required elastic Graph catalog is absent")
                _auto_calibrate_missing_catalog(owner)
            representation = scheduler._elastic_graph_catalog_coverage.get(
                "representation", "pinned_full_family"
            )
            if representation == "bounded_exact_hotset":
                owner._restore_elastic_bounded_hotset()
            else:
                owner._restore_elastic_pinned_full_family()
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
