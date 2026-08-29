# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persist and reuse the memory-profiling result across engine boots.

On startup, vLLM measures how much GPU memory the KV cache can use and
computes the ``--kv-cache-memory`` value that reproduces that allocation.
For a fixed (model, config, hardware, library) combination the result is
deterministic, yet it is re-measured on every boot.

When ``VLLM_ENABLE_STARTUP_PLAN=1``, each worker persists that value under
``{VLLM_CACHE_ROOT}/startup_plan/`` (regenerable derived state, alongside
the torch.compile cache), keyed by a fingerprint of everything the value
depends on, and later boots apply it automatically -- skipping the
memory-profiling measurement and the CUDA-graph memory estimation pass --
if and only if the fingerprint matches and the device has at least as much
free memory as when the plan was recorded. On any mismatch the worker
falls back to full profiling, so a stale plan costs nothing and is never
trusted.
"""

import hashlib
import importlib.util
import json
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

PLAN_SCHEMA_VERSION = 3
GRAPH_RECIPE_SCHEMA_VERSION = 2
ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION = 5
ELASTIC_RUNTIME_GENERATION_SCHEMA_VERSION = 2
ELASTIC_GRAPH_CATALOG_MIGRATION_SCOPE = "compiled-piecewise-bypass-v1"
MTP_PHASE_CONTINUITY_CATALOG_MIGRATION_SCOPE = (
    "mtp-phase-continuity-compiled-carrier-v1"
)
MTP_BATCHED_Q1_CATALOG_MIGRATION_SCOPE = "mtp-batched-q1-verifier-v1"
MTP_BATCHED_Q1_WORKSPACE_CATALOG_MIGRATION_SCOPE = (
    "mtp-batched-q1-workspace-resize-v1"
)
GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE = "graph-execution-policy-v1"
CONTROL_PLANE_ONLY_CATALOG_MIGRATION_SCOPE = "control-plane-only-v1"
ELASTIC_CAPTURE_STATE_ABI = "dynamic-capture-state-v1"
BOUNDED_PIECEWISE_REPLAY_CONTRACT = "cold_capture_bounds_same_key_hot-v1"


def elastic_graph_catalog_row_complete(
    step_key: tuple[int, ...] | list[int],
    row: dict[str, int],
    *,
    representation: str,
) -> bool:
    """Validate evidence according to the reachable replay lifecycle.

    A bounded PIECEWISE carrier is reclaimed before a fresh request cohort can
    cross admission, and a short prefill has only one execution step.  Its
    repeated cold-capture envelope therefore bounds any same-key hot replay;
    requiring a synthetic inter-cohort hot counter would contradict serving.
    FULL carriers execute repeated decode steps and retain the independent
    cold/hot requirement.
    """
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
ELASTIC_GRAPH_CALIBRATION_CHECKPOINT_MIGRATION_SCOPE = (
    "identity-matched-partial-calibration-v1"
)
MTP_PHASE_CONTINUITY_CALIBRATION_MIGRATION_SCOPE = (
    "mtp-phase-continuity-calibration-v1"
)


def _elastic_runtime_source_hashes() -> dict[str, str]:
    """Bind measurements to the Python implementation that owns the DAG."""
    modules = (
        "vllm.distributed.parallel_state",
        "vllm.v1.attention.backends.flashinfer",
        "vllm.v1.core.kv_cache_capacity",
        "vllm.v1.core.kv_cache_coordinator",
        "vllm.v1.core.elastic_graph",
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.engine.core",
        "vllm.v1.sample.ops.topk_topp_sampler",
        "vllm.v1.worker.gpu.cudagraph_utils",
        "vllm.v1.worker.gpu.elastic_gdn",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils",
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
        "vllm.v1.worker.startup_plan",
    )
    result: dict[str, str] = {}
    for module_name in modules:
        spec = importlib.util.find_spec(module_name)
        path = spec.origin if spec is not None else None
        if path is None:
            raise RuntimeError(
                f"cannot resolve elastic runtime source for {module_name}"
            )
        with open(path, "rb") as stream:
            result[module_name] = hashlib.sha256(stream.read()).hexdigest()
    return result


def compute_elastic_runtime_generation(vllm_config: VllmConfig) -> str:
    """Return the content namespace shared by scheduler and Graph owners."""
    from vllm import __version__ as vllm_version

    try:
        device_name = current_platform.get_device_name()
        device_capability = str(current_platform.get_device_capability() or "")
    except NotImplementedError:
        # Pure CPU contract tests run before platform discovery. The sentinel
        # remains a distinct generation and cannot match a live CUDA catalog.
        device_name = "platform-unavailable"
        device_capability = "platform-unavailable"
    config_hash = vllm_config.compute_hash()
    if not isinstance(config_hash, str):
        config_hash = "config-hash-unavailable"
    factors = {
        "schema": ELASTIC_RUNTIME_GENERATION_SCHEMA_VERSION,
        "vllm": vllm_version,
        "vllm_config": config_hash,
        "profile_config": _profile_config_factors(vllm_config),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "device_name": device_name,
        "device_capability": device_capability,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "elastic_graph_hotset_cap_mb": os.environ.get(
            "AG2_VLLM_ELASTIC_GRAPH_HOTSET_CAP_MB", ""
        ),
        "elastic_mm_activation_loan_bytes": os.environ.get(
            "AG2_VLLM_ELASTIC_MM_ACTIVATION_LOAN_BYTES", ""
        ),
        "runtime_source_hashes": _elastic_runtime_source_hashes(),
    }
    return hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()


def _startup_profile_source_hashes() -> dict[str, str]:
    """Bind a persisted KV byte count to its authoritative producers."""
    modules = (
        "vllm.v1.core.kv_cache_utils",
        "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.worker.gpu_worker",
    )
    result: dict[str, str] = {}
    for module_name in modules:
        spec = importlib.util.find_spec(module_name)
        path = spec.origin if spec is not None else None
        if path is None:
            raise RuntimeError(
                f"cannot resolve startup-profile source for {module_name}"
            )
        with open(path, "rb") as stream:
            result[module_name] = hashlib.sha256(stream.read()).hexdigest()
    return result


def _profile_config_factors(vllm_config: VllmConfig) -> dict[str, Any]:
    """Return values that affect profile memory beyond computation-graph ABI.

    ``VllmConfig.compute_hash()`` deliberately follows graph identity. In
    particular, ``SchedulerConfig.compute_hash()`` omits ``max_num_seqs`` and
    async scheduling. Those values still size runner buffers and change memory
    profiling, so a persisted KV budget requires the stronger identity below.
    """
    scheduler = getattr(vllm_config, "scheduler_config", None)
    model = getattr(vllm_config, "model_config", None)
    multimodal = getattr(model, "multimodal_config", None)
    return {
        "scheduler": (
            None
            if scheduler is None
            else {
                "max_num_seqs": scheduler.max_num_seqs,
                "max_num_batched_tokens": scheduler.max_num_batched_tokens,
                "max_num_scheduled_tokens": scheduler.max_num_scheduled_tokens,
                "max_num_encoder_input_tokens": (
                    scheduler.max_num_encoder_input_tokens
                ),
                "encoder_cache_size": scheduler.encoder_cache_size,
                "async_scheduling": scheduler.async_scheduling,
                "enable_chunked_prefill": scheduler.enable_chunked_prefill,
                "disable_chunked_mm_input": scheduler.disable_chunked_mm_input,
            }
        ),
        "multimodal": (
            None
            if multimodal is None
            else {
                "compute_hash": multimodal.compute_hash(),
                "language_model_only": multimodal.language_model_only,
                "skip_mm_profiling": multimodal.skip_mm_profiling,
                "limit_per_prompt": repr(multimodal.limit_per_prompt),
                "mm_processor_kwargs": repr(multimodal.mm_processor_kwargs),
                "mm_encoder_tp_mode": multimodal.mm_encoder_tp_mode,
            }
        ),
    }


def compute_plan_fingerprint(
    vllm_config: VllmConfig, rank: int, world_size: int
) -> str:
    """Hash everything the profiled KV-cache memory value depends on.

    ``VllmConfig.compute_hash()`` covers the vLLM version and the model,
    cache, parallel, and compilation configs, but deliberately contains no
    device identity (``DeviceConfig.compute_hash`` is empty), so device
    name, total memory, compute capability, and the torch/CUDA build are
    added here. The vLLM version is also pinned as an explicit factor so
    version invalidation holds no matter how ``compute_hash`` evolves.
    Rank is included because per-rank memory use differs under TP/PP.
    Driver-only changes are not part of the key; the free-memory gate at
    apply time bounds the residual risk.
    """
    # Imported here (as VllmConfig.compute_hash does) to avoid a cycle with
    # the top-level vllm package.
    from vllm import __version__ as vllm_version

    capability = current_platform.get_device_capability()
    factors = {
        "schema": PLAN_SCHEMA_VERSION,
        "vllm": vllm_version,
        "vllm_config": vllm_config.compute_hash(),
        "profile_config": _profile_config_factors(vllm_config),
        "device_name": current_platform.get_device_name(),
        "device_total_memory": current_platform.get_device_total_memory(),
        "device_capability": str(capability) if capability else "",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "profile_source_hashes": _startup_profile_source_hashes(),
        "profile_env": {
            name: os.environ.get(name, "")
            for name in (
                "AG2_VLLM_DCP_PREFILL_QUERY_SCRATCH",
                "AG2_VLLM_MTP_BF16_GATE_UP_SCRATCH",
                "AG2_VLLM_NVFP4_BATCH_INVARIANT",
                "AG2_VLLM_SHARED_LMHEAD_FP8",
                "AG2_VLLM_TP3_OWNER_PREQUANT",
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_USE_V2_MODEL_RUNNER",
            )
        },
        "rank": rank,
        "world_size": world_size,
    }
    digest = hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _plan_path(fingerprint: str) -> str:
    """Plans are regenerable derived state, so they live under the standard
    vLLM cache root (like the torch.compile cache) and relocate with
    ``VLLM_CACHE_ROOT`` instead of needing a location knob of their own."""
    # VLLM_CACHE_ROOT is already user-expanded by envs.py.
    return os.path.join(
        envs.VLLM_CACHE_ROOT, "startup_plan", f"startup_plan_{fingerprint}.json"
    )


def _graph_recipe_path(fingerprint: str, owner: str) -> str:
    safe_owner = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in owner)
    return os.path.join(
        envs.VLLM_CACHE_ROOT,
        "cudagraph_recipes",
        f"cudagraph_recipe_{fingerprint}_{safe_owner}.json",
    )


def compute_elastic_graph_catalog_fingerprint(
    vllm_config: VllmConfig, kv_cache_config: Any
) -> str:
    """Identify a measured graph/KV coexistence surface.

    Unlike a graph executable, the catalog is rank-safe scheduler evidence.
    Bind it to the complete runtime config and the post-profile, worst-rank KV
    geometry so a model, B, topology, CUDA build, or physical pool change
    cannot silently reuse old measurements.
    """
    from vllm import __version__ as vllm_version

    factors = {
        "schema": ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "vllm": vllm_version,
        "vllm_config": vllm_config.compute_hash(),
        "profile_config": _profile_config_factors(vllm_config),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "device_name": current_platform.get_device_name(),
        "device_total_memory": current_platform.get_device_total_memory(),
        "device_capability": str(current_platform.get_device_capability() or ""),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_dag_env": {
            name: os.environ.get(name, "")
            for name in (
                "AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH",
                "AG2_VLLM_ELASTIC_GRAPH_HOTSET_CAP_MB",
                "AG2_VLLM_ELASTIC_MM_ACTIVATION_LOAN_BYTES",
                "AG2_VLLM_MTP_DCP_PSEUDO_DECODE",
                "AG2_VLLM_MTP_DCP_BATCHED_DECODE",
                "AG2_VLLM_MTP_DCP_BATCHED_FIXED_SPLIT_SIZE",
                "AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB",
                "AG2_VLLM_MTP_DCP_SEQUENTIAL_DECODE",
                "AG2_VLLM_TP3_EMBEDDING_NCCL",
                "AG2_VLLM_TP3_EXACT_OWNER_MIN_ROWS",
                "AG2_VLLM_TP3_OWNER_MIN_ROWS",
                "AG2_VLLM_TP3_OWNER_PREQUANT",
                "AG2_VLLM_TP3_UNIFIED_EXACT_BACKEND",
                "AG2_VLLM_TP3_UNIFIED_EXACT_REDUCE",
                "NCCL_ALGO",
                "NCCL_PROTO",
                "VLLM_TP3_CE_REDUCE",
                "VLLM_USE_FLASHINFER_SAMPLER",
            )
        },
        "runtime_source_hashes": _elastic_runtime_source_hashes(),
        "world_size": vllm_config.parallel_config.world_size,
        "num_blocks": kv_cache_config.num_blocks,
        "attention_stride": kv_cache_config.elastic_attention_stride,
        "gdn_stride": kv_cache_config.elastic_gdn_stride,
        "mapping_quantum": kv_cache_config.elastic_mapping_quantum,
        "gdn_initial_blocks": kv_cache_config.elastic_gdn_initial_blocks,
        "gdn_blocks_per_request": kv_cache_config.elastic_gdn_blocks_per_request,
        "rank_budgets": kv_cache_config.elastic_rank_budget_bytes,
        "rank_primary": kv_cache_config.elastic_rank_primary_mapped_bytes,
        "rank_gdn": kv_cache_config.elastic_rank_gdn_mapped_bytes,
        "graph_execution_policy": getattr(
            kv_cache_config, "elastic_graph_execution_policy", None
        ),
    }
    digest = hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _elastic_graph_catalog_path(fingerprint: str) -> str:
    return os.path.join(
        envs.VLLM_CACHE_ROOT,
        "elastic_graph_catalog",
        f"elastic_graph_catalog_{fingerprint}.json",
    )


def _effective_graph_execution_policy(kv_cache_config: Any):
    from vllm.v1.core.elastic_graph import GraphExecutionPolicy

    payload = getattr(kv_cache_config, "elastic_graph_execution_policy", None)
    if payload is None:
        raise RuntimeError("elastic Graph catalog requires an execution policy")
    return GraphExecutionPolicy.from_payload(payload)


def _catalog_policy_matches(payload: Any, policy: Any) -> bool:
    from vllm.v1.core.elastic_graph import GraphExecutionPolicy

    if not isinstance(payload, dict):
        return False
    try:
        stored = GraphExecutionPolicy.from_payload(payload)
    except (TypeError, ValueError):
        return False
    return stored.fingerprint == policy.fingerprint


def _catalog_policy_row_metadata(
    step_key: tuple[int, ...] | list[int],
    *,
    policy: Any,
    max_num_batched_tokens: int,
    compiled_piecewise_sizes: Any,
) -> dict[str, Any]:
    from vllm.v1.core.elastic_graph import RuntimeGeneration, resolve_step_physical_keys

    keys = resolve_step_physical_keys(
        tuple(step_key),
        RuntimeGeneration("catalog-policy-projection"),
        max_num_batched_tokens,
        compiled_piecewise_sizes,
        policy,
    )
    return {
        "policy_fingerprint": policy.fingerprint,
        "verifier_contract": policy.verifier_contract,
        "verifier_configuration": policy.verifier_configuration,
        "math_contract": policy.math_contract,
        "capture_state_abi": ELASTIC_CAPTURE_STATE_ABI,
        "physical_keys": [
            {
                "owner": key.logical.owner,
                "mode": key.logical.mode,
                "tokens": key.logical.token_bucket,
                "physical_x": key.physical_num_reqs,
                "uniform_query_len": key.logical.uniform_query_len,
            }
            for key in keys
        ],
    }


def _validate_sealed_migration_source(
    payload: dict[str, Any],
    *,
    expected_fingerprint: str,
    vllm_config: VllmConfig,
) -> None:
    """Reject a named source unless its sealed evidence is self-consistent."""
    from vllm.v1.core.elastic_graph import GraphExecutionPolicy

    coverage = payload.get("coverage")
    try:
        policy = GraphExecutionPolicy.from_payload(
            payload.get("graph_execution_policy")
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("migration source has no valid recorded policy") from error
    declared_compiled = (
        coverage.get("compiled_piecewise_sizes", [])
        if isinstance(coverage, dict)
        else None
    )
    if (
        payload.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
        or payload.get("fingerprint") != expected_fingerprint
        or payload.get("sealed") is not True
        or not isinstance(coverage, dict)
        or coverage.get("graph_execution_policy_fingerprint")
        != policy.fingerprint
        or coverage.get("verifier_contract") != policy.verifier_contract
        or coverage.get("verifier_configuration")
        != policy.verifier_configuration
        or coverage.get("math_contract") != policy.math_contract
        or coverage.get("capture_state_abi") != ELASTIC_CAPTURE_STATE_ABI
        or not isinstance(declared_compiled, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in declared_compiled
        )
    ):
        raise RuntimeError("migration source has inconsistent sealed coverage")
    shapes = payload.get("shapes")
    required = coverage.get("required_step_keys")
    representation = coverage.get("representation", "pinned_full_family")
    if (
        not isinstance(shapes, list)
        or not isinstance(required, list)
        or representation not in {"pinned_full_family", "bounded_exact_hotset"}
        or coverage.get("required_shapes") != len(required)
    ):
        raise RuntimeError("migration source lacks shapes or required coverage")
    required_keys = {
        tuple(key)
        for key in required
        if isinstance(key, list) and len(key) == 5
    }
    rows_by_key: dict[tuple[int, ...], dict[str, Any]] = {}
    complete_shape_count = 0
    for row in shapes:
        key = row.get("step_key") if isinstance(row, dict) else None
        if (
            not isinstance(key, list)
            or len(key) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in key
            )
        ):
            raise RuntimeError("migration source contains a malformed row")
        expected = _catalog_policy_row_metadata(
            key,
            policy=policy,
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            compiled_piecewise_sizes=frozenset(declared_compiled),
        )
        if any(row.get(name) != value for name, value in expected.items()):
            raise RuntimeError(
                "migration source row has stale representation lineage: "
                f"step_key={key!r}"
            )
        fields = {
            name: row.get(name)
            for name in (
                "cold_peak_bytes",
                "hot_peak_bytes",
                "resident_bytes",
                "floor_bytes",
                "cold_observations",
                "cold_stable_replays",
                "hot_observations",
                "hot_stable_replays",
            )
        }
        finalized = row.get("finalized_pinned_owner_set", 0)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fields.values()
        ) or (
            isinstance(finalized, bool) or finalized not in {0, 1}
        ):
            raise RuntimeError("migration source contains invalid row counters")
        complete_row = dict(fields)
        if finalized:
            complete_row["finalized_pinned_owner_set"] = finalized
        complete = elastic_graph_catalog_row_complete(
            key, complete_row, representation=representation
        ) or (
            representation == "pinned_full_family"
            and key[0] == 1
            and finalized == 1
            and fields["cold_peak_bytes"]
            and fields["hot_peak_bytes"]
            and fields["cold_observations"] >= 2
            and fields["hot_observations"] >= 2
            and fields["hot_stable_replays"] >= 1
        )
        complete_shape_count += int(bool(complete))
        # A sealed catalog may retain incomplete probes above its accepted
        # product boundary.  They remain useful calibration history, but only
        # required_step_keys are serving evidence and therefore completeness
        # gates.  Structural, policy-lineage and counter validation still
        # applies to every historical row.
        if tuple(key) in required_keys and not complete:
            raise RuntimeError(
                "migration source contains an incomplete required row: "
                f"step_key={key!r}"
            )
        rows_by_key[tuple(key)] = row
    if any(
        not isinstance(key, list) or tuple(key) not in rows_by_key
        for key in required
    ):
        raise RuntimeError("migration source required coverage is incomplete")
    if payload.get("complete_shapes") != complete_shape_count:
        raise RuntimeError("migration source complete-shape count is inconsistent")


def project_elastic_graph_catalog_to_policy(
    payload: dict[str, Any],
    vllm_config: VllmConfig,
    policy: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project measured rows onto one policy without recapturing valid rows."""
    from vllm.v1.core.elastic_graph import (
        ElasticGraphError,
        configured_compiled_piecewise_sizes,
    )

    preproject_rejected: list[dict[str, Any]] = []
    preproject_rejected_bytes = 0
    if policy.verifier_contract == "batched-causal-q1-v1":
        raw_coverage = payload.get("coverage")
        raw_required = (
            raw_coverage.get("required_step_keys", [])
            if isinstance(raw_coverage, dict)
            else []
        )
        incompatible_q4_full = [
            key
            for key in raw_required
            if (
                isinstance(key, list)
                and len(key) == 5
                and key[0] == 1
                and key[1] > 0
                and key[4] == key[1] + 1
            )
        ]
        if incompatible_q4_full:
            shape_by_key = {
                tuple(row.get("step_key", ())): row
                for row in payload.get("shapes", [])
                if isinstance(row, dict)
            }
            preproject_rejected = [
                {
                    "step_key": key,
                    "reason": "q4 FULL forbidden by active Graph execution policy",
                }
                for key in incompatible_q4_full
            ]
            preproject_rejected_bytes = sum(
                int(shape_by_key.get(tuple(key), {}).get("resident_bytes", 0) or 0)
                for key in incompatible_q4_full
            )
            payload = _migrate_mtp_batched_q1_catalog(payload, vllm_config)
        # Older accepted batched-q1 catalogs named every semantic verifier X
        # by its padded PIECEWISE token bucket and left qlen unset.  The active
        # runtime owns only the power-of-two X inventory plus the exact MaxX
        # endpoint.  Canonicalize to that physical inventory before projecting
        # owners: X33..X40 are one terminal X40/M160 executable, not eight
        # exact graphs and not a blind X64/M256 carrier.
        raw_coverage = payload.get("coverage")
        verifier_keys = (
            raw_coverage.get("mtp_verifier_piecewise_step_keys", [])
            if isinstance(raw_coverage, dict)
            else []
        )
        if verifier_keys:
            from vllm.v1.core.elastic_graph import select_short_decode_physical_x

            verifier_set = {
                tuple(key)
                for key in verifier_keys
                if isinstance(key, list) and len(key) == 5
            }
            decode_max_x = raw_coverage.get("decode_max_x")
            if (
                isinstance(decode_max_x, bool)
                or not isinstance(decode_max_x, int)
                or decode_max_x < 1
            ):
                raise RuntimeError("verifier migration lacks DecodeMaxX")
            inventory_xs: list[int] = []
            inventory_x = 1
            while inventory_x <= decode_max_x:
                inventory_xs.append(inventory_x)
                inventory_x <<= 1
            if inventory_xs[-1] != decode_max_x:
                inventory_xs.append(decode_max_x)
            rekeyed: dict[tuple[int, ...], tuple[int, ...]] = {}
            for old in verifier_set:
                if old[0] != 0 or old[1] <= 0 or old[2] <= 0:
                    raise RuntimeError("invalid padded verifier catalog key")
                physical_x = select_short_decode_physical_x(
                    old[2], inventory_xs
                )
                exact_m = physical_x * (old[1] + 1)
                rekeyed[old] = (
                    old[0], old[1], physical_x, exact_m, old[1] + 1
                )

            def map_key(value: Any) -> Any:
                if isinstance(value, list) and tuple(value) in rekeyed:
                    return list(rekeyed[tuple(value)])
                return value

            migrated_payload = dict(payload)
            migrated_coverage = dict(raw_coverage)
            for field in ("required_step_keys", "restore_step_keys"):
                values = migrated_coverage.get(field)
                if isinstance(values, list):
                    mapped = [map_key(value) for value in values]
                    migrated_coverage[field] = [
                        list(value)
                        for value in sorted({tuple(value) for value in mapped})
                    ]
            migrated_coverage["mtp_verifier_piecewise_step_keys"] = [
                list(value) for value in sorted(set(rekeyed.values()))
            ]
            verifier_rows: dict[tuple[int, ...], dict[str, Any]] = {}
            migrated_shapes: list[dict[str, Any]] = []
            for row in payload.get("shapes", []):
                migrated_row = dict(row)
                old_key = tuple(row.get("step_key", ()))
                if old_key in rekeyed:
                    migrated_row["migrated_from_step_key"] = list(old_key)
                    new_key = rekeyed[old_key]
                    migrated_row["step_key"] = list(new_key)
                    # Every declared inventory endpoint exists in the source
                    # family.  It is the only candidate with matching physical
                    # X; owner compatibility below still rejects terminal
                    # M160 when the old M256 carrier was compiled-only.
                    if old_key[2] == new_key[2]:
                        verifier_rows[new_key] = migrated_row
                    continue
                migrated_shapes.append(migrated_row)
            migrated_shapes.extend(verifier_rows.values())
            migrated_payload["coverage"] = migrated_coverage
            migrated_payload["shapes"] = migrated_shapes
            migrated_payload["complete_shapes"] = len(migrated_shapes)
            payload = migrated_payload
    compiled = configured_compiled_piecewise_sizes(vllm_config)
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("policy migration source lacks coverage")
    restore = coverage.get("restore_step_keys", [])
    if compiled and any(
        isinstance(key, list)
        and len(key) == 5
        and key[0] == 0
        and key[1] == 0
        and key[3] in compiled
        for key in restore
    ):
        # The old bounded catalog restored a K0 PIECEWISE witness.  Once that
        # carrier becomes compiled-only, first move restore authority to the
        # already measured K3 witness, then remove the now-empty K0 Graph rows.
        payload = _migrate_mtp_phase_continuity_catalog(payload, vllm_config)
    coverage = payload.get("coverage")
    required = (
        coverage.get("required_step_keys", [])
        if isinstance(coverage, dict)
        else []
    )
    if compiled and any(
        isinstance(key, list)
        and len(key) == 5
        and key[0] == 0
        and key[1] == 0
        and key[3] in compiled
        for key in required
    ):
        payload = _migrate_compiled_piecewise_catalog(payload, vllm_config)
    shapes = payload.get("shapes")
    coverage = payload.get("coverage")
    if not isinstance(shapes, list) or not isinstance(coverage, dict):
        raise RuntimeError("policy migration source lacks shapes or coverage")
    source_contract = coverage.get("compiled_piecewise_contract")
    source_owners = coverage.get("compiled_piecewise_owners")
    if source_contract != ("torch-compile-no-cudagraph-v1" if compiled else None):
        raise RuntimeError(
            "policy migration cannot change the compiled PIECEWISE contract"
        )
    if (source_owners or []) != (["mtp_prefill", "target"] if compiled else []):
        raise RuntimeError(
            "policy migration cannot change compiled PIECEWISE ownership"
        )
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    retained = []
    rejected = list(preproject_rejected)
    retained_resident_bytes = 0
    rejected_resident_bytes = preproject_rejected_bytes
    for row in shapes:
        if not isinstance(row, dict) or not isinstance(row.get("step_key"), list):
            rejected.append({"step_key": None, "reason": "malformed-row"})
            continue
        key = row["step_key"]
        try:
            metadata = _catalog_policy_row_metadata(
                key,
                policy=policy,
                max_num_batched_tokens=max_tokens,
                compiled_piecewise_sizes=compiled,
            )
            source_physical_keys = row.get("physical_keys")
            if source_physical_keys is not None and not isinstance(
                source_physical_keys, list
            ):
                raise ElasticGraphError(
                    "catalog row has malformed physical owner provenance"
                )
            if source_physical_keys is not None:
                remaining = list(source_physical_keys)
                compatible = True
                for new_key in metadata["physical_keys"]:
                    match = next(
                        (
                            old_key
                            for old_key in remaining
                            if old_key.get("owner") == new_key.get("owner")
                            and old_key.get("mode") == new_key.get("mode")
                            and old_key.get("physical_x")
                            == new_key.get("physical_x")
                            and isinstance(old_key.get("tokens"), int)
                            and isinstance(new_key.get("tokens"), int)
                            and old_key["tokens"] >= new_key["tokens"]
                        ),
                        None,
                    )
                    if match is None:
                        compatible = False
                        break
                    remaining.remove(match)
                removable_compiled = all(
                    item.get("owner") in {"target", "mtp_prefill"}
                    and item.get("mode") == "PIECEWISE"
                    and item.get("tokens") in compiled
                    for item in remaining
                )
                if not compatible or not removable_compiled:
                    raise ElasticGraphError(
                        "policy migration adds or enlarges a physical owner"
                    )
        except (ElasticGraphError, ValueError, TypeError) as error:
            rejected.append({"step_key": key, "reason": str(error)})
            rejected_resident_bytes += int(row.get("resident_bytes", 0) or 0)
            continue
        migrated_row = dict(row)
        migrated_row.update(metadata)
        retained.append(migrated_row)
        retained_resident_bytes += int(row.get("resident_bytes", 0) or 0)

    retained_keys = {tuple(row["step_key"]) for row in retained}
    required = coverage.get("required_step_keys", [])
    compatible_required = [
        key for key in required if isinstance(key, list) and tuple(key) in retained_keys
    ]
    missing = [
        key
        for key in required
        if isinstance(key, list) and tuple(key) not in retained_keys
    ]
    migrated_coverage = dict(coverage)
    migrated_coverage.update(
        {
            "required_step_keys": compatible_required,
            "required_shapes": len(compatible_required),
            "graph_execution_policy_fingerprint": policy.fingerprint,
            "verifier_contract": policy.verifier_contract,
            "verifier_configuration": policy.verifier_configuration,
            "math_contract": policy.math_contract,
            "capture_state_abi": ELASTIC_CAPTURE_STATE_ABI,
            # Compiled carriers are part of the effective execution policy,
            # not historical price metadata.  Leaving the source declaration
            # unchanged produced a catalog that passed projection but was
            # correctly rejected by the serving coverage validator.
            "compiled_piecewise_sizes": sorted(compiled),
            "compiled_piecewise_contract": (
                "torch-compile-no-cudagraph-v1" if compiled else None
            ),
            "compiled_piecewise_owners": (
                ["mtp_prefill", "target"] if compiled else []
            ),
        }
    )
    migrated = dict(payload)
    migrated["coverage"] = migrated_coverage
    migrated["shapes"] = retained
    migrated["complete_shapes"] = len(retained)
    migrated["graph_execution_policy"] = policy.to_payload()
    report = {
        "retained_keys": [row["step_key"] for row in retained],
        "rejected": rejected,
        "missing_required_keys": missing,
        "retained_resident_bytes": retained_resident_bytes,
        "rejected_resident_bytes": rejected_resident_bytes,
    }
    migrated["policy_migration_report"] = report
    if missing:
        migrated["sealed"] = False
    return migrated, report


def _migrate_compiled_piecewise_catalog(
    payload: dict[str, Any], vllm_config: VllmConfig
) -> dict[str, Any]:
    """Remove only Graph rows replaced by an exact compiled-only contract."""
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes

    compiled_sizes = configured_compiled_piecewise_sizes(vllm_config)
    if not compiled_sizes:
        raise RuntimeError(
            "compiled PIECEWISE catalog migration requires configured sizes"
        )

    def is_compiled_key(value: Any) -> bool:
        return bool(
            isinstance(value, list)
            and len(value) == 5
            and value[0] == 0
            and value[1] == 0
            and value[3] in compiled_sizes
        )

    coverage = payload.get("coverage")
    shapes = payload.get("shapes")
    if not isinstance(coverage, dict) or not isinstance(shapes, list):
        raise RuntimeError("compiled PIECEWISE migration source lacks coverage")
    required = coverage.get("required_step_keys")
    restore = coverage.get("restore_step_keys")
    if not isinstance(required, list) or not isinstance(restore, list):
        raise RuntimeError("compiled PIECEWISE migration source lacks key sets")
    removed_required = [key for key in required if is_compiled_key(key)]
    removed_shapes = [row for row in shapes if is_compiled_key(row.get("step_key"))]
    if not removed_required or len(removed_required) != len(removed_shapes):
        raise RuntimeError(
            "compiled PIECEWISE migration did not resolve one-to-one catalog rows"
        )
    if any(is_compiled_key(key) for key in restore):
        raise RuntimeError(
            "compiled PIECEWISE migration cannot remove a startup restore key"
        )
    migrated = dict(payload)
    migrated_coverage = dict(coverage)
    migrated_coverage["required_step_keys"] = [
        key for key in required if not is_compiled_key(key)
    ]
    migrated_coverage["required_shapes"] = len(
        migrated_coverage["required_step_keys"]
    )
    migrated_coverage["compiled_piecewise_sizes"] = sorted(compiled_sizes)
    migrated_coverage["compiled_piecewise_contract"] = (
        "torch-compile-no-cudagraph-v1"
    )
    excluded = list(migrated_coverage.get("excluded_graph_step_keys", []))
    for key in removed_required:
        if key not in excluded:
            excluded.append(key)
    migrated_coverage["excluded_graph_step_keys"] = excluded
    migrated["coverage"] = migrated_coverage
    migrated["shapes"] = [
        row for row in shapes if not is_compiled_key(row.get("step_key"))
    ]
    migrated["complete_shapes"] = len(migrated["shapes"])
    return migrated


def _migrate_mtp_phase_continuity_catalog(
    payload: dict[str, Any], vllm_config: VllmConfig
) -> dict[str, Any]:
    """Rebind only the compiled carrier and its minimal restore witness."""
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes

    compiled_sizes = configured_compiled_piecewise_sizes(vllm_config)
    speculative = getattr(vllm_config, "speculative_config", None)
    k = getattr(speculative, "num_speculative_tokens", 0)
    if (
        not compiled_sizes
        or not isinstance(k, int)
        or k <= 0
        or getattr(speculative, "disable_speculation_on_non_decode", True)
    ):
        raise RuntimeError(
            "MTP phase-continuity migration requires active uninterrupted K"
        )
    coverage = payload.get("coverage")
    shapes = payload.get("shapes")
    if (
        not isinstance(coverage, dict)
        or coverage.get("representation") != "bounded_exact_hotset"
        or coverage.get("compiled_piecewise_contract")
        != "torch-compile-no-cudagraph-v1"
        or not isinstance(shapes, list)
    ):
        raise RuntimeError(
            "MTP phase-continuity migration source lacks the sealed compiled "
            "carrier contract"
        )
    required = coverage.get("required_step_keys")
    restore = coverage.get("restore_step_keys")
    if not isinstance(required, list) or not isinstance(restore, list):
        raise RuntimeError("MTP phase-continuity source lacks key sets")
    piecewise_restore = [key for key in restore if key[0] == 0]
    full_restore = [key for key in restore if key[0] == 1]
    if len(piecewise_restore) != 1 or len(full_restore) != 1:
        raise RuntimeError(
            "MTP phase-continuity source requires one PIECEWISE/FULL restore pair"
        )
    old_piecewise = piecewise_restore[0]
    new_piecewise = [old_piecewise[0], k, *old_piecewise[2:]]
    shape_keys = {
        tuple(row.get("step_key", ())) for row in shapes if isinstance(row, dict)
    }
    if new_piecewise not in required or tuple(new_piecewise) not in shape_keys:
        raise RuntimeError(
            "MTP phase-continuity source has no exact K restore witness"
        )

    migrated = dict(payload)
    migrated_coverage = dict(coverage)
    migrated_coverage["restore_step_keys"] = [new_piecewise, full_restore[0]]
    migrated_coverage["compiled_piecewise_owners"] = ["mtp_prefill", "target"]
    excluded = list(migrated_coverage.get("excluded_graph_step_keys", []))
    for key in tuple(excluded):
        if (
            isinstance(key, list)
            and len(key) == 5
            and key[0] == 0
            and key[1] == 0
            and key[3] in compiled_sizes
        ):
            k3_key = [key[0], k, *key[2:]]
            if k3_key not in excluded:
                excluded.append(k3_key)
    migrated_coverage["excluded_graph_step_keys"] = excluded
    migrated["coverage"] = migrated_coverage
    return migrated


def _migrate_mtp_batched_q1_catalog(
    payload: dict[str, Any], vllm_config: VllmConfig
) -> dict[str, Any]:
    """Replace uniform q4 FULL rows with existing PIECEWISE carriers."""
    speculative = getattr(vllm_config, "speculative_config", None)
    k = getattr(speculative, "num_speculative_tokens", 0)
    if (
        not isinstance(k, int)
        or k <= 0
        or os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "0") != "1"
        or os.environ.get("AG2_VLLM_MTP_DCP_PSEUDO_DECODE", "0") == "1"
        or os.environ.get("AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH", "0")
        == "1"
    ):
        raise RuntimeError(
            "batched-q1 catalog migration requires batched decode only and "
            "PIECEWISE qlen>1 attention"
        )
    coverage = payload.get("coverage")
    shapes = payload.get("shapes")
    if (
        not isinstance(coverage, dict)
        or coverage.get("representation") != "bounded_exact_hotset"
        or not isinstance(shapes, list)
    ):
        raise RuntimeError("batched-q1 migration source lacks bounded coverage")
    required = coverage.get("required_step_keys")
    restore = coverage.get("restore_step_keys")
    if not isinstance(required, list) or not isinstance(restore, list):
        raise RuntimeError("batched-q1 migration source lacks key sets")

    def is_uniform_q4_full(key: Any) -> bool:
        return bool(
            isinstance(key, list)
            and len(key) == 5
            and key[0] == 1
            and key[1] == k
            and key[2] > 0
            and key[3] == (k + 1) * key[2]
            and key[4] == k + 1
        )

    removed = [key for key in required if is_uniform_q4_full(key)]
    shape_by_key = {
        tuple(row.get("step_key", ())): row
        for row in shapes
        if isinstance(row, dict)
    }
    if len(removed) != coverage.get("decode_max_x"):
        raise RuntimeError(
            "batched-q1 migration did not resolve one uniform FULL row per X"
        )
    if any(tuple(key) not in shape_by_key for key in removed):
        raise RuntimeError("batched-q1 migration source is missing a FULL q4 row")
    if any(is_uniform_q4_full(key) for key in restore):
        raise RuntimeError("batched-q1 migration cannot remove a restore key")

    alternatives: list[list[int]] = []
    for full_key in removed:
        x = full_key[2]
        candidates = [
            key
            for key in required
            if isinstance(key, list)
            and len(key) == 5
            and key[0] == 0
            and key[1] == k
            and key[2] == x
            and key[3] >= (k + 1) * x
            and key[4] == 0
        ]
        if len(candidates) != 1 or tuple(candidates[0]) not in shape_by_key:
            raise RuntimeError(
                "batched-q1 migration lacks one exact PIECEWISE carrier for "
                f"X={x}: {candidates!r}"
            )
        alternatives.append(candidates[0])

    migrated = dict(payload)
    migrated_coverage = dict(coverage)
    migrated_coverage["required_step_keys"] = [
        key for key in required if not is_uniform_q4_full(key)
    ]
    migrated_coverage["required_shapes"] = len(
        migrated_coverage["required_step_keys"]
    )
    excluded = list(migrated_coverage.get("excluded_graph_step_keys", []))
    excluded.extend(key for key in removed if key not in excluded)
    migrated_coverage["excluded_graph_step_keys"] = excluded
    migrated_coverage["mtp_verifier_contract"] = "batched-causal-q1-v1"
    migrated_coverage["mtp_verifier_workspace_bytes"] = int(
        os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB", "132")
    ) * (1 << 20)
    migrated_coverage["mtp_verifier_piecewise_step_keys"] = alternatives
    migrated["coverage"] = migrated_coverage
    migrated["shapes"] = [
        row
        for row in shapes
        if not is_uniform_q4_full(
            row.get("step_key") if isinstance(row, dict) else None
        )
    ]
    migrated["complete_shapes"] = len(migrated["shapes"])
    return migrated


def _migrate_mtp_batched_q1_workspace_catalog(
    payload: dict[str, Any], vllm_config: VllmConfig
) -> dict[str, Any]:
    """Resize only the proved batched-q1 workspace ledger."""
    del vllm_config
    if (
        os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "0") != "1"
        or os.environ.get("AG2_VLLM_MTP_DCP_PSEUDO_DECODE", "0") == "1"
        or os.environ.get("AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH", "0")
        == "1"
    ):
        raise RuntimeError(
            "batched-q1 workspace migration requires the unchanged batched "
            "PIECEWISE verifier route"
        )
    coverage = payload.get("coverage")
    shapes = payload.get("shapes")
    complete_shapes = payload.get("complete_shapes")
    if (
        not isinstance(coverage, dict)
        or coverage.get("representation") != "bounded_exact_hotset"
        or coverage.get("mtp_verifier_contract") != "batched-causal-q1-v1"
        or not isinstance(shapes, list)
        or complete_shapes != len(shapes)
        or coverage.get("required_shapes") != len(shapes)
    ):
        raise RuntimeError(
            "batched-q1 workspace migration source lacks complete exact coverage"
        )
    old_workspace = coverage.get("mtp_verifier_workspace_bytes")
    new_workspace = int(
        os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB", "0")
    ) * (1 << 20)
    if (
        not isinstance(old_workspace, int)
        or old_workspace <= 0
        or new_workspace <= 0
        or new_workspace >= old_workspace
    ):
        raise RuntimeError(
            "batched-q1 workspace migration requires a strict proved reduction: "
            f"old={old_workspace!r} new={new_workspace!r}"
        )
    migrated = dict(payload)
    migrated_coverage = dict(coverage)
    migrated_coverage["mtp_verifier_workspace_bytes"] = new_workspace
    migrated["coverage"] = migrated_coverage
    return migrated


def load_elastic_graph_catalog(
    vllm_config: VllmConfig, kv_cache_config: Any
) -> dict[tuple[int, ...], dict[str, int]]:
    """Load only identity-matched, complete cold+hot shape measurements."""
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return {}
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes

    path = _elastic_graph_catalog_path(fingerprint)
    migrate_from = os.environ.get(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", ""
    ).strip()
    migration_scope = os.environ.get(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE", ""
    ).strip()
    try:
        with open(path) as f:
            payload = json.load(f)
    except FileNotFoundError:
        if not migrate_from:
            logger.warning(
                "Elastic CUDA Graph catalog is absent for the effective "
                "runtime identity: path=%s fingerprint=%s",
                path,
                fingerprint,
            )
            return {}
        if migration_scope not in {
            ELASTIC_GRAPH_CATALOG_MIGRATION_SCOPE,
            MTP_PHASE_CONTINUITY_CATALOG_MIGRATION_SCOPE,
            MTP_BATCHED_Q1_CATALOG_MIGRATION_SCOPE,
            MTP_BATCHED_Q1_WORKSPACE_CATALOG_MIGRATION_SCOPE,
            GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE,
            CONTROL_PLANE_ONLY_CATALOG_MIGRATION_SCOPE,
        }:
            raise RuntimeError(
                "elastic Graph catalog migration requires the exact reviewed "
                "scope: expected="
                f"{ELASTIC_GRAPH_CATALOG_MIGRATION_SCOPE!r} "
                f"actual={migration_scope!r}"
            ) from None
        if (
            len(migrate_from) != 16
            or any(ch not in "0123456789abcdef" for ch in migrate_from)
            or migrate_from == fingerprint
        ):
            raise RuntimeError(
                "invalid elastic Graph catalog migration source: "
                f"{migrate_from!r}"
            ) from None
        source_path = _elastic_graph_catalog_path(migrate_from)
        try:
            with open(source_path) as f:
                payload = json.load(f)
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                "explicit elastic Graph catalog migration source is unreadable: "
                f"{source_path}"
            ) from e
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
            or payload.get("fingerprint") != migrate_from
            or payload.get("sealed") is not True
        ):
            raise RuntimeError(
                "explicit elastic Graph catalog migration source is not an "
                f"identity-matched sealed catalog: {source_path}"
            ) from None
        _validate_sealed_migration_source(
            payload,
            expected_fingerprint=migrate_from,
            vllm_config=vllm_config,
        )
        if migration_scope == CONTROL_PLANE_ONLY_CATALOG_MIGRATION_SCOPE:
            from vllm.v1.core.elastic_graph import (
                configured_compiled_piecewise_sizes,
            )

            coverage = payload.get("coverage")
            active_policy = _effective_graph_execution_policy(kv_cache_config)
            active_compiled = sorted(
                configured_compiled_piecewise_sizes(vllm_config)
            )
            if (
                not isinstance(coverage, dict)
                or coverage.get("representation") != "bounded_exact_hotset"
                or not _catalog_policy_matches(
                    payload.get("graph_execution_policy"), active_policy
                )
                or coverage.get("compiled_piecewise_sizes") != active_compiled
            ):
                raise RuntimeError(
                    "control-plane-only catalog migration changed the physical "
                    "Graph policy or bounded representation"
                ) from None
            logger.warning(
                "Reusing sealed bounded Graph measurements after an explicit "
                "control-plane-only lineage decision: source=%s",
                source_path,
            )
        elif migration_scope == GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE:
            policy = _effective_graph_execution_policy(kv_cache_config)
            payload, report = project_elastic_graph_catalog_to_policy(
                dict(payload), vllm_config, policy
            )
            logger.warning(
                "Elastic Graph policy migration dry-run result: retained=%d "
                "rejected=%d missing=%d retained_bytes=%d rejected_bytes=%d",
                len(report["retained_keys"]),
                len(report["rejected"]),
                len(report["missing_required_keys"]),
                report["retained_resident_bytes"],
                report["rejected_resident_bytes"],
            )
        elif migration_scope == MTP_PHASE_CONTINUITY_CATALOG_MIGRATION_SCOPE:
            payload = _migrate_mtp_phase_continuity_catalog(
                dict(payload), vllm_config
            )
        elif migration_scope == MTP_BATCHED_Q1_CATALOG_MIGRATION_SCOPE:
            payload = _migrate_mtp_batched_q1_catalog(
                dict(payload), vllm_config
            )
        elif migration_scope == MTP_BATCHED_Q1_WORKSPACE_CATALOG_MIGRATION_SCOPE:
            payload = _migrate_mtp_batched_q1_workspace_catalog(
                dict(payload), vllm_config
            )
        else:
            payload = _migrate_compiled_piecewise_catalog(
                dict(payload), vllm_config
            )
        payload["fingerprint"] = fingerprint
        payload["migrated_from"] = migrate_from
        payload["migration_scope"] = migration_scope
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(tmp, path)
        logger.warning(
            "Migrated sealed elastic CUDA Graph catalog after an explicit "
            "%s lineage decision: source=%s destination=%s",
            migration_scope,
            source_path,
            path,
        )
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable elastic Graph catalog %s: %s", path, e)
        return {}
    migrated_from = payload.get("migrated_from") if isinstance(payload, dict) else None
    if migrated_from is not None and (
        migrate_from != migrated_from
        or migration_scope != payload.get("migration_scope")
        or migration_scope
        not in {
            ELASTIC_GRAPH_CATALOG_MIGRATION_SCOPE,
            MTP_PHASE_CONTINUITY_CATALOG_MIGRATION_SCOPE,
            MTP_BATCHED_Q1_CATALOG_MIGRATION_SCOPE,
            MTP_BATCHED_Q1_WORKSPACE_CATALOG_MIGRATION_SCOPE,
            GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE,
            CONTROL_PLANE_ONLY_CATALOG_MIGRATION_SCOPE,
        }
    ):
        raise RuntimeError(
            "migrated elastic Graph catalog lacks the exact active lineage "
            "authorization: "
            f"catalog_source={migrated_from!r} active_source={migrate_from!r} "
            f"catalog_scope={payload.get('migration_scope')!r} "
            f"active_scope={migration_scope!r}"
        )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
        or payload.get("fingerprint") != fingerprint
        or payload.get("sealed") is not True
        or not _catalog_policy_matches(
            payload.get("graph_execution_policy"),
            _effective_graph_execution_policy(kv_cache_config),
        )
    ):
        logger.warning("Ignoring stale or unsealed elastic Graph catalog %s", path)
        return {}
    coverage = payload.get("coverage")
    policy = _effective_graph_execution_policy(kv_cache_config)
    compiled_piecewise_sizes = configured_compiled_piecewise_sizes(vllm_config)
    representation = (
        coverage.get("representation", "pinned_full_family")
        if isinstance(coverage, dict)
        else "pinned_full_family"
    )
    result: dict[tuple[int, ...], dict[str, int]] = {}
    for row in payload.get("shapes", []):
        if not isinstance(row, dict):
            continue
        key = row.get("step_key")
        if (
            not isinstance(key, list)
            or len(key) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in key
            )
        ):
            continue
        expected_policy_metadata = _catalog_policy_row_metadata(
            key,
            policy=policy,
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            compiled_piecewise_sizes=compiled_piecewise_sizes,
        )
        if any(
            row.get(name) != value
            for name, value in expected_policy_metadata.items()
        ):
            raise RuntimeError(
                "elastic Graph catalog row has stale representation lineage: "
                f"step_key={key!r}"
            )
        fields = {
            name: row.get(name)
            for name in (
                "cold_peak_bytes",
                "hot_peak_bytes",
                "resident_bytes",
                "floor_bytes",
                "cold_observations",
                "cold_stable_replays",
                "hot_observations",
                "hot_stable_replays",
            )
        }
        finalized_pinned_owner_set = row.get(
            "finalized_pinned_owner_set", 0
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fields.values()
        ) or (
            isinstance(finalized_pinned_owner_set, bool)
            or finalized_pinned_owner_set not in {0, 1}
        ):
            continue
        complete_row = dict(fields)
        if finalized_pinned_owner_set:
            complete_row["finalized_pinned_owner_set"] = (
                finalized_pinned_owner_set
            )
        if not elastic_graph_catalog_row_complete(
            key, complete_row, representation=representation
        ) and not (
            representation == "pinned_full_family"
            and key[0] == 1
            and finalized_pinned_owner_set == 1
            and fields["cold_peak_bytes"]
            and fields["hot_peak_bytes"]
            and fields["cold_observations"] >= 2
            and fields["hot_observations"] >= 2
            and fields["hot_stable_replays"] >= 1
        ):
            continue
        if finalized_pinned_owner_set:
            fields["finalized_pinned_owner_set"] = finalized_pinned_owner_set
        result[tuple(key)] = fields  # type: ignore[assignment]
    logger.info(
        "Loaded elastic CUDA Graph catalog %s (%d complete shapes)",
        path,
        len(result),
    )
    return result


def load_elastic_graph_catalog_coverage(
    vllm_config: VllmConfig, kv_cache_config: Any
) -> dict[str, Any]:
    """Load the sealed product boundary paired with the price catalog.

    Shape rows may include probes above the accepted fixed point.  Serving must
    therefore consume the explicitly sealed boundary rather than infer MaxX
    from the largest complete historical row.
    """
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    path = _elastic_graph_catalog_path(fingerprint)
    try:
        with open(path) as f:
            payload = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"sealed elastic Graph catalog coverage is unreadable: {path}"
        ) from error
    coverage = payload.get("coverage") if isinstance(payload, dict) else None
    policy = _effective_graph_execution_policy(kv_cache_config)
    integer_fields = (
        "decode_max_x",
        "mixed_max_x",
        "full_context_max_x",
        "pinned_full_entries",
        "pinned_full_bytes",
    )
    representation = (
        coverage.get("representation", "pinned_full_family")
        if isinstance(coverage, dict)
        else None
    )
    restore_step_keys = (
        coverage.get("restore_step_keys", [])
        if isinstance(coverage, dict)
        else None
    )
    required_step_keys = (
        coverage.get("required_step_keys", [])
        if isinstance(coverage, dict)
        else None
    )
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes

    configured_compiled = configured_compiled_piecewise_sizes(vllm_config)
    declared_compiled = (
        coverage.get("compiled_piecewise_sizes", [])
        if isinstance(coverage, dict)
        else None
    )
    compiled_contract = (
        coverage.get("compiled_piecewise_contract")
        if isinstance(coverage, dict)
        else None
    )
    if (
        payload.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
        or payload.get("fingerprint") != fingerprint
        or payload.get("sealed") is not True
        or not _catalog_policy_matches(
            payload.get("graph_execution_policy"), policy
        )
        or not isinstance(coverage, dict)
        or coverage.get("graph_execution_policy_fingerprint")
        != policy.fingerprint
        or coverage.get("verifier_contract") != policy.verifier_contract
        or coverage.get("verifier_configuration")
        != policy.verifier_configuration
        or coverage.get("math_contract") != policy.math_contract
        or coverage.get("capture_state_abi") != ELASTIC_CAPTURE_STATE_ABI
        or any(
            isinstance(coverage.get(name), bool)
            or not isinstance(coverage.get(name), int)
            or coverage[name] < (1 if name.endswith("max_x") else 0)
            for name in integer_fields
        )
        or coverage["mixed_max_x"] > coverage["decode_max_x"]
        or coverage["full_context_max_x"] > coverage["decode_max_x"]
        or not isinstance(required_step_keys, list)
        or (
            configured_compiled
            and (
                declared_compiled != sorted(configured_compiled)
                or compiled_contract != "torch-compile-no-cudagraph-v1"
                or any(
                    isinstance(key, list)
                    and len(key) == 5
                    and key[0] == 0
                    and key[1] == 0
                    and key[3] in configured_compiled
                    for key in required_step_keys
                )
            )
        )
        or (not configured_compiled and bool(declared_compiled))
        or representation not in {"pinned_full_family", "bounded_exact_hotset"}
        or (
            representation == "pinned_full_family"
            and (
                coverage["pinned_full_entries"] < 1
                or coverage["pinned_full_bytes"] < 1
            )
        )
        or (
            representation == "bounded_exact_hotset"
            and (
                coverage["pinned_full_entries"] != 0
                or coverage["pinned_full_bytes"] != 0
                or not isinstance(restore_step_keys, list)
                or not restore_step_keys
                or any(key not in required_step_keys for key in restore_step_keys)
                or coverage.get("piecewise_replay_contract")
                != BOUNDED_PIECEWISE_REPLAY_CONTRACT
            )
        )
    ):
        raise RuntimeError(
            f"sealed elastic Graph catalog has no valid product boundary: {path}"
        )
    return coverage


def load_elastic_graph_calibration_checkpoint(
    vllm_config: VllmConfig, kv_cache_config: Any
) -> dict[tuple[int, ...], dict[str, int]]:
    """Resume synchronized cold measurements from a failed pre-READY run.

    Serving continues to accept only a sealed catalog through
    :func:`load_elastic_graph_catalog`.  Calibration may reuse partial rows
    because each row was atomically persisted only after all ranks returned a
    synchronized worker receipt.  An explicit lineage migration is required
    when scheduler-only recovery code changed the catalog fingerprint.
    """
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return {}
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    path = _elastic_graph_catalog_path(fingerprint)
    migrate_from = os.environ.get(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", ""
    ).strip()
    migration_scope = os.environ.get(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE", ""
    ).strip()
    try:
        with open(path) as f:
            payload = json.load(f)
    except FileNotFoundError:
        if migration_scope not in {
            ELASTIC_GRAPH_CALIBRATION_CHECKPOINT_MIGRATION_SCOPE,
            MTP_PHASE_CONTINUITY_CALIBRATION_MIGRATION_SCOPE,
        }:
            return {}
        if (
            len(migrate_from) != 16
            or any(ch not in "0123456789abcdef" for ch in migrate_from)
            or migrate_from == fingerprint
        ):
            raise RuntimeError(
                "invalid elastic calibration checkpoint migration source: "
                f"{migrate_from!r}"
            ) from None
        source_path = _elastic_graph_catalog_path(migrate_from)
        try:
            with open(source_path) as f:
                payload = json.load(f)
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                "elastic calibration checkpoint migration source is unreadable: "
                f"{source_path}"
            ) from e
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
            or payload.get("fingerprint") != migrate_from
        ):
            raise RuntimeError(
                "elastic calibration checkpoint migration source is not "
                f"identity-matched: {source_path}"
            ) from None
        if payload.get("sealed") is True:
            _validate_sealed_migration_source(
                payload,
                expected_fingerprint=migrate_from,
                vllm_config=vllm_config,
            )
        payload = dict(payload)
        if migration_scope == MTP_PHASE_CONTINUITY_CALIBRATION_MIGRATION_SCOPE:
            from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes

            coverage = payload.get("coverage")
            if (
                payload.get("sealed") is not True
                or not isinstance(coverage, dict)
                or coverage.get("representation") != "bounded_exact_hotset"
                or coverage.get("compiled_piecewise_contract")
                != "torch-compile-no-cudagraph-v1"
            ):
                raise RuntimeError(
                    "MTP phase-continuity migration requires a sealed bounded "
                    "compiled-PIECEWISE source catalog"
                ) from None
            retained_shapes = []
            reused_k0_piecewise = 0
            active_policy = _effective_graph_execution_policy(kv_cache_config)
            active_compiled = configured_compiled_piecewise_sizes(vllm_config)
            for row in payload.get("shapes", []):
                key = row.get("step_key") if isinstance(row, dict) else None
                if (
                    isinstance(key, list)
                    and len(key) == 5
                    and key[0] == 0
                    and key[1] == 0
                ):
                    # K0 remains a required calibration control and its
                    # physical owner set is unchanged. Only the newly
                    # reachable K3 high-prefill rows lack source evidence.
                    metadata = _catalog_policy_row_metadata(
                        key,
                        policy=active_policy,
                        max_num_batched_tokens=(
                            vllm_config.scheduler_config.max_num_batched_tokens
                        ),
                        compiled_piecewise_sizes=active_compiled,
                    )
                    if row.get("physical_keys") != metadata["physical_keys"]:
                        raise RuntimeError(
                            "MTP phase-continuity K0 row changed physical owners: "
                            f"step_key={key!r}"
                        ) from None
                    reused_k0_piecewise += 1
                    migrated_row = dict(row)
                    migrated_row.update(metadata)
                    retained_shapes.append(migrated_row)
            if not reused_k0_piecewise:
                raise RuntimeError(
                    "MTP phase-continuity migration source has no K0 "
                    "PIECEWISE control rows to reuse"
                ) from None
            payload["shapes"] = retained_shapes
            payload["sealed"] = False
            payload.pop("coverage", None)
            payload["phase_continuity_reused_k0_piecewise_rows"] = (
                reused_k0_piecewise
            )
        payload["fingerprint"] = fingerprint
        # A calibration checkpoint must never become a serving catalog merely
        # because the process died between the atomic copy and its first
        # measurement.  Only the dedicated sealed-catalog migration path above
        # may preserve ``sealed=True``.
        payload["sealed"] = False
        payload["calibration_migrated_from"] = migrate_from
        payload["calibration_migration_scope"] = migration_scope
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(tmp, path)
        logger.warning(
            "Migrated identity-matched partial elastic calibration checkpoint: "
            "source=%s destination=%s",
            source_path,
            path,
        )
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Ignoring unreadable elastic calibration checkpoint %s: %s", path, e
        )
        return {}
    migrated_from = (
        payload.get("calibration_migrated_from")
        if isinstance(payload, dict)
        else None
    )
    if migrated_from is not None and (
        migrate_from != migrated_from
        or migration_scope != payload.get("calibration_migration_scope")
        or migration_scope
        not in {
            ELASTIC_GRAPH_CALIBRATION_CHECKPOINT_MIGRATION_SCOPE,
            MTP_PHASE_CONTINUITY_CALIBRATION_MIGRATION_SCOPE,
        }
    ):
        raise RuntimeError(
            "partial elastic calibration checkpoint lacks active lineage "
            "authorization"
        )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
        or payload.get("fingerprint") != fingerprint
    ):
        logger.warning(
            "Ignoring stale elastic calibration checkpoint %s", path
        )
        return {}
    result: dict[tuple[int, ...], dict[str, int]] = {}
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes

    policy = _effective_graph_execution_policy(kv_cache_config)
    compiled_piecewise_sizes = configured_compiled_piecewise_sizes(vllm_config)
    field_names = (
        "cold_peak_bytes",
        "hot_peak_bytes",
        "resident_bytes",
        "floor_bytes",
        "cold_observations",
        "cold_stable_replays",
        "hot_observations",
        "hot_stable_replays",
    )
    for row in payload.get("shapes", []):
        if not isinstance(row, dict):
            continue
        key = row.get("step_key")
        fields = {name: row.get(name) for name in field_names}
        finalized_pinned_owner_set = row.get(
            "finalized_pinned_owner_set", 0
        )
        if (
            not isinstance(key, list)
            or len(key) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in key
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in fields.values()
            )
            or isinstance(finalized_pinned_owner_set, bool)
            or finalized_pinned_owner_set not in {0, 1}
            or not fields["cold_peak_bytes"]
        ):
            continue
        expected_policy_metadata = _catalog_policy_row_metadata(
            key,
            policy=policy,
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            compiled_piecewise_sizes=compiled_piecewise_sizes,
        )
        if any(
            row.get(name) != value
            for name, value in expected_policy_metadata.items()
        ):
            logger.warning(
                "Dropping partial elastic calibration row with stale lineage: "
                "step_key=%s",
                key,
            )
            continue
        if finalized_pinned_owner_set:
            fields["finalized_pinned_owner_set"] = finalized_pinned_owner_set
        result[tuple(key)] = fields  # type: ignore[assignment]
    logger.warning(
        "Resuming elastic calibration from %d synchronized partial rows: %s",
        len(result),
        path,
    )
    return result


def load_elastic_graph_calibration_boundary(
    vllm_config: VllmConfig, kv_cache_config: Any
) -> dict[str, int]:
    """Load only a conservative boundary from a migrated partial checkpoint.

    A graph-policy migration may retain synchronized prices from a sealed
    source while leaving a small set of newly reachable shapes unmeasured.
    Calibration must not probe above the source's accepted DecodeMaxX: the
    migration can add work, but cannot prove that a larger cohort became
    feasible.  Serving still rejects this unsealed file through the normal
    catalog loader.
    """
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return {}
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    path = _elastic_graph_catalog_path(fingerprint)
    try:
        with open(path) as f:
            payload = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
        or payload.get("fingerprint") != fingerprint
    ):
        return {}
    policy_migration = (
        payload.get("sealed") is False
        and payload.get("migration_scope")
        == GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE
    )
    identity_checkpoint = (
        payload.get("calibration_migration_scope")
        == ELASTIC_GRAPH_CALIBRATION_CHECKPOINT_MIGRATION_SCOPE
    )
    if not (policy_migration or identity_checkpoint):
        return {}
    source_fingerprint = payload.get(
        "migrated_from" if policy_migration else "calibration_migrated_from"
    )
    active_source = os.environ.get(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", ""
    ).strip()
    active_scope = os.environ.get(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE", ""
    ).strip()
    expected_scope = (
        GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE
        if policy_migration
        else ELASTIC_GRAPH_CALIBRATION_CHECKPOINT_MIGRATION_SCOPE
    )
    if (
        not isinstance(source_fingerprint, str)
        or len(source_fingerprint) != 16
        or any(ch not in "0123456789abcdef" for ch in source_fingerprint)
        or active_source != source_fingerprint
        or active_scope != expected_scope
    ):
        return {}
    try:
        with open(_elastic_graph_catalog_path(source_fingerprint)) as f:
            source = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    source_coverage = source.get("coverage") if isinstance(source, dict) else None
    if identity_checkpoint:
        try:
            _validate_sealed_migration_source(
                source,
                expected_fingerprint=source_fingerprint,
                vllm_config=vllm_config,
            )
        except (RuntimeError, TypeError, ValueError):
            return {}
    coverage = source_coverage if identity_checkpoint else payload.get("coverage")
    policy = _effective_graph_execution_policy(kv_cache_config)
    decode_max_x = (
        coverage.get("decode_max_x") if isinstance(coverage, dict) else None
    )
    if (
        not isinstance(source, dict)
        or source.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
        or source.get("fingerprint") != source_fingerprint
        or source.get("sealed") is not True
        or not isinstance(coverage, dict)
        or coverage.get("graph_execution_policy_fingerprint")
        != policy.fingerprint
        or isinstance(decode_max_x, bool)
        or not isinstance(decode_max_x, int)
        or decode_max_x < 1
    ):
        return {}
    return {
        "decode_max_x": decode_max_x,
        "bounded_exact_hotset": int(
            coverage.get("representation") == "bounded_exact_hotset"
        ),
    }


def save_elastic_graph_catalog(
    vllm_config: VllmConfig,
    kv_cache_config: Any,
    catalog: dict[tuple[int, ...], dict[str, int]],
    *,
    sealed: bool = False,
    coverage: dict[str, Any] | None = None,
) -> str | None:
    """Atomically persist rank-safe measurements; incomplete rows stay visible."""
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return None
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes

    fingerprint = compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    path = _elastic_graph_catalog_path(fingerprint)
    policy = _effective_graph_execution_policy(kv_cache_config)
    compiled_piecewise_sizes = configured_compiled_piecewise_sizes(vllm_config)
    prior_payload: dict[str, Any] = {}
    if not sealed and coverage is None:
        try:
            with open(path) as f:
                candidate = json.load(f)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            candidate = None
        if (
            isinstance(candidate, dict)
            and candidate.get("schema") == ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
            and candidate.get("fingerprint") == fingerprint
            and candidate.get("sealed") is False
        ):
            prior_payload = candidate
    prior_coverage = prior_payload.get("coverage")
    coverage_payload = dict(
        coverage
        if coverage is not None
        else (prior_coverage if isinstance(prior_coverage, dict) else {})
    )
    coverage_payload.update(
        {
            "compiled_piecewise_sizes": sorted(compiled_piecewise_sizes),
            "compiled_piecewise_contract": (
                "torch-compile-no-cudagraph-v1"
                if compiled_piecewise_sizes
                else None
            ),
            "compiled_piecewise_owners": (
                ["mtp_prefill", "target"]
                if compiled_piecewise_sizes
                else []
            ),
            "graph_execution_policy_fingerprint": policy.fingerprint,
            "verifier_contract": policy.verifier_contract,
            "verifier_configuration": policy.verifier_configuration,
            "math_contract": policy.math_contract,
            "capture_state_abi": ELASTIC_CAPTURE_STATE_ABI,
        }
    )
    shapes = []
    for key, values in sorted(catalog.items()):
        row = {
            "step_key": list(key),
            "cold_peak_bytes": int(values.get("cold_peak_bytes", 0)),
            "hot_peak_bytes": int(values.get("hot_peak_bytes", 0)),
            "resident_bytes": int(values.get("resident_bytes", 0)),
            "floor_bytes": int(values.get("floor_bytes", 0)),
            "cold_observations": int(values.get("cold_observations", 0)),
            "cold_stable_replays": int(values.get("cold_stable_replays", 0)),
            "hot_observations": int(values.get("hot_observations", 0)),
            "hot_stable_replays": int(values.get("hot_stable_replays", 0)),
            "finalized_pinned_owner_set": int(
                values.get("finalized_pinned_owner_set", 0)
            ),
        }
        row.update(
            _catalog_policy_row_metadata(
                key,
                policy=policy,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
                compiled_piecewise_sizes=compiled_piecewise_sizes,
            )
        )
        shapes.append(row)
    payload = {
        "schema": ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "sealed": sealed,
        "coverage": coverage_payload,
        "graph_execution_policy": policy.to_payload(),
        "complete_shapes": sum(
            elastic_graph_catalog_row_complete(
                row["step_key"],
                row,
                representation=(
                    coverage_payload.get(
                        "representation", "pinned_full_family"
                    )
                ),
            )
            or bool(
                row["step_key"][0] == 1
                and row["finalized_pinned_owner_set"] == 1
                and row["cold_peak_bytes"]
                and row["hot_peak_bytes"]
                and row["cold_observations"] >= 2
                and row["hot_observations"] >= 2
                and row["hot_stable_replays"] >= 1
            )
            for row in shapes
        ),
        "shapes": shapes,
    }
    if not sealed:
        for name in (
            "migrated_from",
            "migration_scope",
            "calibration_migrated_from",
            "calibration_migration_scope",
        ):
            if name in prior_payload:
                payload[name] = prior_payload[name]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, sort_keys=True)
    os.replace(tmp, path)
    return path


def try_reseal_policy_migrated_elastic_graph_catalog(
    vllm_config: VllmConfig,
    kv_cache_config: Any,
    catalog: dict[tuple[int, ...], dict[str, int]],
    *,
    calibration_wall_seconds: float,
) -> dict[str, Any] | None:
    """Seal a completed bounded policy projection without replaying old rows.

    The persisted migration report is deliberately not trusted: incremental
    checkpoint saves need not preserve it.  Recomputing the projection from
    the named sealed source makes the source rows, rejected rows, and newly
    required rows deterministic under the active policy.
    """
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return None
    migrate_from = os.environ.get(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", ""
    ).strip()
    migration_scope = os.environ.get(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE", ""
    ).strip()
    if migration_scope != GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE:
        return None
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    if (
        len(migrate_from) != 16
        or any(ch not in "0123456789abcdef" for ch in migrate_from)
        or migrate_from == fingerprint
    ):
        raise RuntimeError("policy reseal has invalid migration source lineage")

    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or speculative_config.num_speculative_tokens != 3
        or speculative_config.disable_speculation_on_non_decode
    ):
        raise RuntimeError(
            "policy reseal requires the unchanged K3 product-prefill contract"
        )

    path = _elastic_graph_catalog_path(fingerprint)
    source_path = _elastic_graph_catalog_path(migrate_from)
    try:
        with open(path) as stream:
            destination = json.load(stream)
    except FileNotFoundError:
        # A migration source may remain configured across a policy change, but
        # reseal is only an optimization once the destination has checkpointed
        # at least one measured row.  No destination means there is no partial
        # migration to recover: let the normal calibration path create it.
        return None
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("policy reseal destination is unreadable") from error
    try:
        with open(source_path) as stream:
            source = json.load(stream)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError("policy reseal source lineage is unreadable") from error
    if (
        not isinstance(destination, dict)
        or destination.get("schema") != ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
        or destination.get("fingerprint") != fingerprint
        or destination.get("sealed") is not False
        or destination.get("migrated_from") != migrate_from
        or destination.get("migration_scope") != migration_scope
    ):
        raise RuntimeError("policy reseal destination has stale lineage")
    if not isinstance(source, dict):
        raise RuntimeError("policy reseal source is malformed")
    _validate_sealed_migration_source(
        source,
        expected_fingerprint=migrate_from,
        vllm_config=vllm_config,
    )

    policy = _effective_graph_execution_policy(kv_cache_config)
    projected, report = project_elastic_graph_catalog_to_policy(
        dict(source), vllm_config, policy
    )
    coverage = projected.get("coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("representation") != "bounded_exact_hotset"
    ):
        return None
    missing_values = report.get("missing_required_keys")
    retained_values = report.get("retained_keys")
    if not isinstance(missing_values, list) or not isinstance(retained_values, list):
        raise RuntimeError("policy reseal recomputation produced no exact report")

    def parse_keys(values: Any, *, label: str) -> set[tuple[int, ...]]:
        if not isinstance(values, list):
            raise RuntimeError(f"policy reseal has no {label} key list")
        result: set[tuple[int, ...]] = set()
        for value in values:
            if (
                not isinstance(value, list)
                or len(value) != 5
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
            ):
                raise RuntimeError(f"policy reseal has malformed {label} key")
            result.add(tuple(value))
        if len(result) != len(values):
            raise RuntimeError(f"policy reseal has duplicate {label} keys")
        return result

    missing = parse_keys(missing_values, label="missing")
    retained = parse_keys(retained_values, label="retained")
    if not missing or missing & retained:
        raise RuntimeError("policy reseal has an invalid projected key partition")
    projected_rows = {
        tuple(row["step_key"]): row
        for row in projected.get("shapes", [])
        if isinstance(row, dict) and isinstance(row.get("step_key"), list)
    }
    if set(projected_rows) != retained:
        raise RuntimeError("policy reseal retained-row projection is inconsistent")

    destination_rows = destination.get("shapes")
    if not isinstance(destination_rows, list):
        raise RuntimeError("policy reseal destination has no shape rows")
    persisted_rows: dict[tuple[int, ...], dict[str, Any]] = {}
    field_names = (
        "cold_peak_bytes",
        "hot_peak_bytes",
        "resident_bytes",
        "floor_bytes",
        "cold_observations",
        "cold_stable_replays",
        "hot_observations",
        "hot_stable_replays",
        "finalized_pinned_owner_set",
    )
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes

    compiled = configured_compiled_piecewise_sizes(vllm_config)
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    for row in destination_rows:
        key_value = row.get("step_key") if isinstance(row, dict) else None
        if not isinstance(key_value, list) or len(key_value) != 5:
            raise RuntimeError("policy reseal destination contains a malformed row")
        key = tuple(key_value)
        if key in persisted_rows:
            raise RuntimeError("policy reseal destination contains duplicate rows")
        metadata = _catalog_policy_row_metadata(
            key_value,
            policy=policy,
            max_num_batched_tokens=max_tokens,
            compiled_piecewise_sizes=compiled,
        )
        if any(row.get(name) != value for name, value in metadata.items()):
            raise RuntimeError("policy reseal destination row has stale policy lineage")
        persisted_rows[key] = row
    persisted_keys = set(persisted_rows)
    if not retained.issubset(persisted_keys):
        raise RuntimeError("policy reseal destination lost a retained source row")
    if not persisted_keys.issubset(retained | missing):
        raise RuntimeError("policy reseal destination key set changed unexpectedly")
    if set(catalog) != persisted_keys:
        raise RuntimeError("policy reseal in-memory key set differs from checkpoint")
    if not missing.issubset(persisted_keys):
        return None

    for key in retained:
        source_row = projected_rows[key]
        persisted = persisted_rows[key]
        current = catalog[key]
        if any(
            int(persisted.get(name, 0)) != int(current.get(name, 0))
            for name in field_names
        ):
            raise RuntimeError(
                "policy reseal retained measurement differs between memory and "
                "checkpoint: "
                f"step_key={key!r}"
            )
        source_values = {
            name: int(source_row.get(name, 0)) for name in field_names
        }
        current_values = {
            name: int(current.get(name, 0)) for name in field_names
        }
        monotone_names = (
            "cold_peak_bytes",
            "hot_peak_bytes",
            "resident_bytes",
            "floor_bytes",
            "cold_observations",
            "hot_observations",
        )
        if any(
            current_values[name] < source_values[name] for name in monotone_names
        ):
            raise RuntimeError(
                "policy reseal retained measurement differs non-monotonically "
                "from sealed source: "
                f"step_key={key!r}"
            )
        if (
            current_values["finalized_pinned_owner_set"]
            != source_values["finalized_pinned_owner_set"]
        ):
            raise RuntimeError(
                "policy reseal retained finalized-owner invariant changed: "
                f"step_key={key!r}"
            )
        changed = any(
            current_values[name] != source_values[name] for name in field_names
        )
        observations_advanced = (
            current_values["cold_observations"]
            > source_values["cold_observations"]
            or current_values["hot_observations"]
            > source_values["hot_observations"]
        )
        if changed and not observations_advanced:
            raise RuntimeError(
                "policy reseal retained measurement differs without a new "
                f"observation: step_key={key!r}"
            )
        if not elastic_graph_catalog_row_complete(
            key, current, representation="bounded_exact_hotset"
        ):
            raise RuntimeError(
                "policy reseal retained measurement is no longer complete: "
                f"step_key={key!r}"
            )
    for key in missing:
        persisted = persisted_rows[key]
        current = catalog[key]
        if any(
            int(persisted.get(name, 0)) != int(current.get(name, 0))
            for name in field_names
        ):
            raise RuntimeError(
                "policy reseal in-memory measurement differs from checkpoint: "
                f"step_key={key!r}"
            )
        if not elastic_graph_catalog_row_complete(
            key, current, representation="bounded_exact_hotset"
        ):
            return None

    compatible_required = parse_keys(
        coverage.get("required_step_keys", []), label="required"
    )
    if not compatible_required.issubset(retained):
        raise RuntimeError("policy reseal required rows are not source-retained")
    required = compatible_required | missing
    restore = parse_keys(coverage.get("restore_step_keys", []), label="restore")
    if not restore or not restore.issubset(required):
        raise RuntimeError("policy reseal restore subset is invalid")
    boundary_names = (
        "decode_max_x",
        "mixed_max_x",
        "full_context_max_x",
        "pinned_full_entries",
        "pinned_full_bytes",
    )
    boundaries = {name: coverage.get(name) for name in boundary_names}
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in boundaries.values()
    ):
        raise RuntimeError("policy reseal source boundaries are malformed")

    seal_elastic_graph_catalog(
        vllm_config,
        kv_cache_config,
        catalog,
        required_step_keys=required,
        calibration_wall_seconds=calibration_wall_seconds,
        decode_max_x=boundaries["decode_max_x"],
        mixed_max_x=boundaries["mixed_max_x"],
        full_context_max_x=boundaries["full_context_max_x"],
        pinned_full_entries=boundaries["pinned_full_entries"],
        pinned_full_bytes=boundaries["pinned_full_bytes"],
        representation="bounded_exact_hotset",
        restore_step_keys=restore,
        inherited_coverage=coverage,
    )
    return load_elastic_graph_catalog_coverage(vllm_config, kv_cache_config)


def seal_elastic_graph_catalog(
    vllm_config: VllmConfig,
    kv_cache_config: Any,
    catalog: dict[tuple[int, ...], dict[str, int]],
    *,
    required_step_keys: Iterable[tuple[int, ...]],
    calibration_wall_seconds: float,
    decode_max_x: int,
    mixed_max_x: int,
    full_context_max_x: int,
    pinned_full_entries: int,
    pinned_full_bytes: int,
    representation: str = "pinned_full_family",
    restore_step_keys: Iterable[tuple[int, ...]] = (),
    inherited_coverage: dict[str, Any] | None = None,
) -> str:
    """Publish a catalog only after every declared runtime class is complete."""
    required = tuple(sorted(set(required_step_keys)))
    missing = []
    for key in required:
        row = catalog.get(key)
        complete = bool(
            row
            and elastic_graph_catalog_row_complete(
                key, row, representation=representation
            )
        )
        finalized_legacy_full = bool(
            row
            and representation == "pinned_full_family"
            and key[0] == 1
            and row.get("finalized_pinned_owner_set", 0) == 1
            and row.get("cold_peak_bytes")
            and row.get("hot_peak_bytes")
            and row.get("cold_observations", 0) >= 2
            and row.get("hot_observations", 0) >= 2
            and row.get("hot_stable_replays", 0) >= 1
        )
        if not (complete or finalized_legacy_full):
            missing.append(key)
    if missing:
        raise RuntimeError(
            "elastic CUDA Graph catalog coverage is incomplete: "
            f"missing={len(missing)} first={missing[:8]}"
        )
    restore = tuple(sorted(set(restore_step_keys)))
    if representation not in {"pinned_full_family", "bounded_exact_hotset"}:
        raise RuntimeError(
            f"unknown elastic Graph catalog representation: {representation!r}"
        )
    pinned_boundary_invalid = (
        pinned_full_entries < 1 or pinned_full_bytes < 1
        if representation == "pinned_full_family"
        else pinned_full_entries != 0 or pinned_full_bytes != 0
    )
    if (
        decode_max_x < 1
        or mixed_max_x < 1
        or full_context_max_x < 1
        or mixed_max_x > decode_max_x
        or full_context_max_x > decode_max_x
        or pinned_boundary_invalid
        or (
            representation == "bounded_exact_hotset"
            and (not restore or any(key not in required for key in restore))
        )
    ):
        raise RuntimeError(
            "elastic CUDA Graph product boundary is invalid: "
            f"decode={decode_max_x} mixed={mixed_max_x} "
            f"full_context={full_context_max_x} "
            f"pinned_entries={pinned_full_entries} "
            f"pinned_bytes={pinned_full_bytes} "
            f"representation={representation!r} restore={restore!r}"
        )
    coverage = dict(inherited_coverage or {})
    coverage.update(
        {
            "required_step_keys": [list(key) for key in required],
            "required_shapes": len(required),
            "calibration_wall_seconds": calibration_wall_seconds,
            "decode_max_x": decode_max_x,
            "mixed_max_x": mixed_max_x,
            "full_context_max_x": full_context_max_x,
            "pinned_full_entries": pinned_full_entries,
            "pinned_full_bytes": pinned_full_bytes,
        }
    )
    if representation == "bounded_exact_hotset":
        coverage.update(
            representation=representation,
            restore_step_keys=[list(key) for key in restore],
            piecewise_replay_contract=BOUNDED_PIECEWISE_REPLAY_CONTRACT,
        )
    path = save_elastic_graph_catalog(
        vllm_config,
        kv_cache_config,
        catalog,
        sealed=True,
        coverage=coverage,
    )
    if path is None:
        raise RuntimeError("startup-plan persistence is disabled; catalog cannot seal")
    logger.info(
        "Sealed elastic CUDA Graph catalog: path=%s required_shapes=%d "
        "calibration_wall_seconds=%.3f",
        path,
        len(required),
        calibration_wall_seconds,
    )
    return path


def maybe_save_cudagraph_recipe(
    vllm_config: VllmConfig,
    *,
    rank: int,
    world_size: int,
    owner: str,
    descriptors: Iterable[Any],
) -> None:
    """Persist every reachable descriptor without retaining a device graph.

    A ``cudaGraphExec`` owns process-local device pointers and allocator pools
    and cannot be safely restored from disk. This recipe is the persistent
    half of on-demand capture: compiled code is handled by vLLM's AOT cache;
    the exact descriptor catalog is saved here and live graph residency is
    reconstructed only after scheduler admission grants a KV loan.
    """
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return
    serialized = sorted(
        (
            {
                "mode": descriptor.cg_mode.name,
                "num_tokens": descriptor.num_tokens,
                "num_reqs": descriptor.num_reqs,
                "uniform_token_count": descriptor.uniform_token_count,
                "num_active_loras": descriptor.num_active_loras,
                "physical_num_reqs": descriptor.physical_num_reqs,
                "runtime_generation": descriptor.runtime_generation,
            }
            for descriptor in descriptors
        ),
        key=lambda item: (
            item["mode"],
            item["num_tokens"],
            -1 if item["num_reqs"] is None else item["num_reqs"],
            -1 if item["uniform_token_count"] is None else item["uniform_token_count"],
            item["num_active_loras"],
            -1
            if item["physical_num_reqs"] is None
            else item["physical_num_reqs"],
            item["runtime_generation"],
        ),
    )
    try:
        fingerprint = compute_plan_fingerprint(vllm_config, rank, world_size)
        payload = {
            "schema": GRAPH_RECIPE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "owner": owner,
            "descriptors": serialized,
        }
        path = _graph_recipe_path(fingerprint, owner)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path) as f:
                existing = json.load(f)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            existing = None
        if existing == payload:
            logger.info(
                "Reusing CUDA Graph recipe %s (%d descriptors)",
                path,
                len(serialized),
            )
            return
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(tmp, path)
        logger.info(
            "Saved CUDA Graph recipe to %s (%d descriptors)",
            path,
            len(serialized),
        )
    except Exception as e:
        logger.warning("Failed to save CUDA Graph recipe for %s: %s", owner, e)


def load_cudagraph_recipe(
    vllm_config: VllmConfig,
    *,
    rank: int,
    world_size: int,
    owner: str,
) -> list[dict[str, Any]]:
    """Load structurally valid CUDA Graph descriptor metadata.

    Runtime-specific semantic validation is deliberately left to the graph
    manager, which owns the effective FULL/PIECEWISE and shape contracts. Any
    malformed cache entry is regenerable and therefore fails closed to an
    empty catalog rather than making engine startup unavailable.
    """
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return []
    try:
        fingerprint = compute_plan_fingerprint(vllm_config, rank, world_size)
        path = _graph_recipe_path(fingerprint, owner)
        try:
            with open(path) as f:
                payload = json.load(f)
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Ignoring unreadable CUDA Graph recipe %s: %s", path, e)
            return []
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != GRAPH_RECIPE_SCHEMA_VERSION
            or payload.get("fingerprint") != fingerprint
            or payload.get("owner") != owner
        ):
            logger.warning("Ignoring stale CUDA Graph recipe %s", path)
            return []
        descriptors = payload.get("descriptors")
        expected_keys = {
            "mode",
            "num_tokens",
            "num_reqs",
            "uniform_token_count",
            "num_active_loras",
            "physical_num_reqs",
            "runtime_generation",
        }
        if not isinstance(descriptors, list):
            raise ValueError("descriptors must be a list")
        for descriptor in descriptors:
            if not isinstance(descriptor, dict) or set(descriptor) != expected_keys:
                raise ValueError("descriptor keys do not match the recipe schema")
            mode = descriptor["mode"]
            num_tokens = descriptor["num_tokens"]
            num_reqs = descriptor["num_reqs"]
            uniform_token_count = descriptor["uniform_token_count"]
            num_active_loras = descriptor["num_active_loras"]
            physical_num_reqs = descriptor["physical_num_reqs"]
            runtime_generation = descriptor["runtime_generation"]
            if mode not in {"FULL", "PIECEWISE"}:
                raise ValueError(f"unsupported CUDA Graph mode {mode!r}")
            if (
                isinstance(num_tokens, bool)
                or not isinstance(num_tokens, int)
                or num_tokens <= 0
                or (
                    num_reqs is not None
                    and (
                        isinstance(num_reqs, bool)
                        or not isinstance(num_reqs, int)
                        or num_reqs <= 0
                    )
                )
                or (
                    uniform_token_count is not None
                    and (
                        isinstance(uniform_token_count, bool)
                        or not isinstance(uniform_token_count, int)
                        or uniform_token_count <= 0
                    )
                )
                or isinstance(num_active_loras, bool)
                or not isinstance(num_active_loras, int)
                or num_active_loras < 0
                or isinstance(physical_num_reqs, bool)
                or not isinstance(physical_num_reqs, int)
                or physical_num_reqs <= 0
                or not isinstance(runtime_generation, str)
                or not runtime_generation
            ):
                raise ValueError("descriptor contains invalid scalar values")
        logger.info(
            "Loaded CUDA Graph recipe %s (%d WARM descriptors; startup "
            "residency is zero)",
            path,
            len(descriptors),
        )
        return descriptors
    except Exception as e:
        logger.warning("Ignoring invalid CUDA Graph recipe for %s: %s", owner, e)
        return []


def _load_plan(fingerprint: str) -> dict | None:
    """Load a plan for this fingerprint; None if absent or unreadable."""
    path = _plan_path(fingerprint)
    try:
        with open(path) as f:
            plan = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable startup plan %s: %s", path, e)
        return None
    if (
        plan.get("schema") != PLAN_SCHEMA_VERSION
        or plan.get("fingerprint") != fingerprint
    ):
        return None
    return plan


def _applicable_kv_cache_memory_bytes(
    plan: dict, current_free_memory: int
) -> int | None:
    """The apply-time OOM-safety gate.

    The recorded value is only valid if the device has at least as much
    free memory now as when the plan was measured (co-tenants, leaked
    allocations, or MIG changes all reduce it). Outside that envelope,
    return None and let the caller re-profile.
    """
    kv_bytes = plan.get("kv_cache_memory_bytes")
    baseline = plan.get("free_memory_baseline")
    if not isinstance(kv_bytes, int) or not isinstance(baseline, int):
        return None
    if kv_bytes <= 0:
        return None
    if current_free_memory < baseline:
        logger.info(
            "Startup plan not applied: current free memory (%.2f GiB) is "
            "below the recorded baseline (%.2f GiB); falling back to full "
            "memory profiling.",
            current_free_memory / (1 << 30),
            baseline / (1 << 30),
        )
        return None
    return kv_bytes


def maybe_apply_startup_plan(worker: "Worker") -> None:
    """If enabled and ``--kv-cache-memory`` was not set explicitly, apply a
    persisted plan by setting ``worker.cache_config.kv_cache_memory_bytes``.
    No-op unless ``VLLM_ENABLE_STARTUP_PLAN=1``."""
    if (
        not envs.VLLM_ENABLE_STARTUP_PLAN
        or worker.cache_config.kv_cache_memory_bytes is not None
    ):
        return
    fingerprint = compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    plan = _load_plan(fingerprint)
    if plan is None:
        return
    current_free_memory = worker.init_snapshot.free_memory
    kv_bytes = _applicable_kv_cache_memory_bytes(plan, current_free_memory)
    if kv_bytes is None:
        return
    logger.info(
        "Applying persisted startup plan (fingerprint %s): "
        "kv_cache_memory_bytes=%d (%.2f GiB), recorded free-memory "
        "baseline %.2f GiB, current %.2f GiB. Memory profiling will "
        "be skipped.",
        fingerprint,
        kv_bytes,
        kv_bytes / (1 << 30),
        plan["free_memory_baseline"] / (1 << 30),
        current_free_memory / (1 << 30),
    )
    worker.cache_config.kv_cache_memory_bytes = kv_bytes


def maybe_save_startup_plan(worker: "Worker", kv_cache_memory_bytes: int) -> None:
    """Atomically persist this boot's profiling result for future boots.
    No-op unless ``VLLM_ENABLE_STARTUP_PLAN=1``; failures are logged,
    never raised."""
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return
    fingerprint = compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    path = _plan_path(fingerprint)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema": PLAN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "kv_cache_memory_bytes": int(kv_cache_memory_bytes),
            "free_memory_baseline": int(worker.init_snapshot.free_memory),
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        logger.info("Saved startup plan to %s", path)
    except OSError as e:
        logger.warning("Failed to save startup plan to %s: %s", path, e)
