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
from vllm.v1.core.elastic_catalog import (
    ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
    elastic_graph_catalog_row_complete,
)
from vllm.v1.core.elastic_runtime import (
    elastic_profile_config_factors,
    elastic_runtime_source_hashes,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

PLAN_SCHEMA_VERSION = 3
GRAPH_RECIPE_SCHEMA_VERSION = 2
ELASTIC_CAPTURE_STATE_ABI = "dynamic-capture-state-v1"
BOUNDED_PIECEWISE_REPLAY_CONTRACT = "cold_capture_bounds_same_key_hot-v1"


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
        "profile_config": elastic_profile_config_factors(vllm_config),
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
        "profile_config": elastic_profile_config_factors(vllm_config),
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
        "runtime_source_hashes": elastic_runtime_source_hashes(),
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
    component_hashes = {
        name: hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()[:16]
        for name, value in factors.items()
    }
    logger.info(
        "Elastic Graph catalog identity components: %s",
        component_hashes,
    )
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
    try:
        with open(path) as stream:
            payload = json.load(stream)
    except FileNotFoundError:
        logger.warning(
            "Elastic CUDA Graph catalog is absent for the effective runtime "
            "identity: path=%s fingerprint=%s",
            path,
            fingerprint,
        )
        return {}
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Ignoring unreadable elastic Graph catalog %s: %s", path, error)
        return {}
    if isinstance(payload, dict) and (
        "migrated_from" in payload or "migration_scope" in payload
    ):
        raise RuntimeError(
            "serving requires an offline-finalized elastic Graph catalog "
            "without migration lineage fields"
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
