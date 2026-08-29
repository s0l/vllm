# SPDX-License-Identifier: Apache-2.0
"""Single owner for elastic KV/scheduler startup ordering.

The callbacks keep model-specific KV initialization and EngineCore lifecycle
setup at their natural owners, while this module makes the order between KV,
policy, scheduler generation, workers, and pre-READY executable restore
explicit rather than documentary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    resolve_elastic_graph_execution_policy(
        vllm_config, kv_cache_config, collective_rpc
    )
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
    scheduler = owner.scheduler
    outcome = "not_elastic"
    if (
        getattr(scheduler, "elastic_on_demand_graphs", False)
        and getattr(scheduler, "_elastic_require_catalog", False)
    ):
        try:
            representation = scheduler._elastic_graph_catalog_coverage.get(
                "representation", "pinned_full_family"
            )
            if representation == "bounded_exact_hotset":
                owner._restore_elastic_bounded_hotset()
            else:
                owner._restore_elastic_pinned_full_family()
        except Exception:
            logger.critical(
                "PINNED_FULL_RESTORE_FAILED: sealed serving startup did not "
                "restore the complete FULL family before READY",
                exc_info=True,
            )
            owner._shutdown_failed_elastic_startup()
            raise
        outcome = "restored"

    if getattr(scheduler, "elastic_on_demand_graphs", False):
        owner._synchronize_elastic_startup_residency()
    return outcome
