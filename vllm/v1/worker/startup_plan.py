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
    expected_semantic_token_witnesses,
    load_sealed_catalog_with_digest,
    validate_elastic_catalog_key_inventory,
)
from vllm.v1.core.elastic_runtime import (
    elastic_catalog_physical_source_hashes,
    elastic_profile_config_factors,
    elastic_runtime_source_hashes,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class ElasticGraphCatalog(dict[tuple[int, ...], dict[str, Any]]):
    """Validated price rows bound to one exact source byte sequence."""

    def __init__(self, *, source_sha256: str) -> None:
        super().__init__()
        self.source_sha256 = source_sha256


PLAN_SCHEMA_VERSION = 4
GRAPH_RECIPE_SCHEMA_VERSION = 2
ELASTIC_CAPTURE_STATE_ABI = "dynamic-capture-state-v1"
BOUNDED_PIECEWISE_REPLAY_CONTRACT = "cold_capture_bounds_same_key_hot-v1"

# Environment switches below change persistent workspaces, temporary peaks,
# CUDA Graph coverage, or the collective/attention implementation exercised by
# memory profiling.  They are therefore part of the cached KV-budget identity,
# even when VllmConfig.compute_hash() is unchanged.
STARTUP_PROFILE_ENV_NAMES = (
    "AG2_VLLM_TP3_ROW_PROFILE_SHA256",
    "AG2_VLLM_DCP_ABSOLUTE_PREFILL_SEGMENT_SIZE",
    "AG2_VLLM_DCP_ABSOLUTE_SEGMENT_PREFILL",
    "AG2_VLLM_DCP_CANONICAL_PAGED_FIXED_SPLIT_SIZE",
    "AG2_VLLM_DCP_CANONICAL_PAGED_MASK_MAX_BITS",
    "AG2_VLLM_DCP_CANONICAL_PAGED_PREFILL",
    "AG2_VLLM_DCP_PREFILL_MATCH_FP8_NEW_TOKENS",
    "AG2_VLLM_DCP_PREFILL_QUERY_SCRATCH",
    "AG2_VLLM_FLASHINFER_DCP_CONTEXT_FIXED_SPLIT_SIZE",
    "AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH",
    "AG2_VLLM_FLASHINFER_DCP_PREFILL_NO_SPLIT",
    "AG2_VLLM_FLASHINFER_DCP_RAGGED_FIXED_SPLIT_SIZE",
    "AG2_VLLM_FLASHINFER_LOG2_LSE_MERGE",
    "AG2_VLLM_FLASHINFER_Q1_DISABLE_SPLIT_KV",
    "AG2_VLLM_FLASHINFER_Q1_FIXED_SPLIT_SIZE",
    "AG2_VLLM_MTP_BF16_GATE_UP_SCRATCH",
    "AG2_VLLM_MTP_DCP_BATCHED_DECODE",
    "AG2_VLLM_MTP_DCP_BATCHED_DISABLE_SPLIT_KV",
    "AG2_VLLM_MTP_DCP_BATCHED_FIXED_SPLIT_SIZE",
    "AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB",
    "AG2_VLLM_MTP_DCP_MATCH_FP8_NEW_TOKENS",
    "AG2_VLLM_MTP_DCP_PSEUDO_DECODE",
    "AG2_VLLM_MTP_DCP_SEQUENTIAL_DECODE",
    "AG2_VLLM_MTP_DCP_SEQUENTIAL_FIXED_SPLIT_SIZE",
    "AG2_VLLM_MTP_DEVICE_CE",
    "AG2_VLLM_NVFP4_BATCH_INVARIANT",
    "AG2_VLLM_SHARED_LMHEAD_FP8",
    "AG2_VLLM_TP3_EMBEDDING_NCCL",
    "AG2_VLLM_TP3_EXACT_OWNER_MIN_ROWS",
    "AG2_VLLM_TP3_OWNER_MIN_ROWS",
    "AG2_VLLM_TP3_OWNER_PREQUANT",
    "AG2_VLLM_TP3_PIECEWISE_DEVICE_CE",
    "AG2_VLLM_TP3_PIECEWISE_DEVICE_CE_PACKED",
    "AG2_VLLM_TP3_PREFILL_CANONICAL_REDUCE",
    "AG2_VLLM_TP3_UNIFIED_EXACT_BACKEND",
    "AG2_VLLM_TP3_UNIFIED_EXACT_REDUCE",
    "VLLM_DCP_NATIVE_RS_MAX_ROWS",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
    "VLLM_TP3_CE_REDUCE",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_USE_V2_MODEL_RUNNER",
)


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
        # Model and execution-DAG changes can alter temporary and captured
        # memory even when the serialized VllmConfig is unchanged. Reuse the
        # same bounded source inventory as elastic Graph identity so a stale
        # KV/profile plan cannot survive such a change.
        "runtime_source_hashes": elastic_runtime_source_hashes(),
        "profile_env": {
            name: os.environ.get(name, "") for name in STARTUP_PROFILE_ENV_NAMES
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
    """Identify physical prices; current placement is checked at restore."""
    return compute_elastic_graph_price_identity(vllm_config, kv_cache_config)[
        "fingerprint"
    ]


def compute_elastic_graph_price_identity(
    vllm_config: VllmConfig, kv_cache_config: Any
) -> dict[str, Any]:
    from vllm.v1.core.elastic_price_identity import (
        flashinfer_price_tactics,
        native_price_provenance,
        price_identity,
    )

    factors = {
        "schema": ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "vllm_config": vllm_config.compute_hash(include_version=False),
        "profile_config": elastic_profile_config_factors(vllm_config),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "device_name": current_platform.get_device_name(),
        "device_total_memory": current_platform.get_device_total_memory(),
        "device_capability": str(current_platform.get_device_capability() or ""),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_dag_env": {
            name: getattr(envs, name, os.environ.get(name, ""))
            for name in sorted(
                set(STARTUP_PROFILE_ENV_NAMES)
                .difference(
                    {
                        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
                    }
                )
                .union(
                    {
                        "AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH",
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
                        "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
                    }
                )
            )
        },
        "physical_source_hashes": elastic_catalog_physical_source_hashes(),
        "native_provenance": native_price_provenance(),
        "flashinfer_tactics": flashinfer_price_tactics(vllm_config),
        "allocator_config": os.environ.get(
            "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        ),
        "world_size": vllm_config.parallel_config.world_size,
        "attention_stride": kv_cache_config.elastic_attention_stride,
        "gdn_stride": kv_cache_config.elastic_gdn_stride,
        "mapping_quantum": kv_cache_config.elastic_mapping_quantum,
        "gdn_blocks_per_request": kv_cache_config.elastic_gdn_blocks_per_request,
        "graph_execution_policy": getattr(
            kv_cache_config, "elastic_graph_execution_policy", None
        ),
    }
    component_hashes = {
        name: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        for name, value in factors.items()
    }
    logger.info(
        "Elastic Graph price identity components: %s",
        component_hashes,
    )
    return price_identity(factors)


def _elastic_graph_catalog_path(fingerprint: str) -> str:
    selected = os.environ.get("AG2_VLLM_ELASTIC_CATALOG_PATH", "")
    if selected:
        if not os.path.isabs(selected):
            raise RuntimeError("explicit elastic catalog path must be absolute")
        return selected
    return os.path.join(
        envs.VLLM_CACHE_ROOT,
        "elastic_graph_catalog",
        f"elastic_graph_catalog_{fingerprint}.json",
    )


def elastic_catalog_owner_generation(
    vllm_config: VllmConfig, kv_cache_config: Any
) -> str:
    from vllm.v1.core.elastic_graph import bind_runtime_generation_to_policy
    from vllm.v1.core.elastic_runtime import compute_elastic_runtime_generation

    return bind_runtime_generation_to_policy(
        compute_elastic_runtime_generation(vllm_config),
        _effective_graph_execution_policy(kv_cache_config),
    )


def _bind_catalog_price_owners(payload, rows, vllm_config, kv_cache_config):
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes
    from vllm.v1.core.elastic_price_identity import (
        PRICE_OWNER_GENERATION,
        price_identity_differences,
        remap_catalog_resident_keys,
    )

    identity = payload.get("price_identity")
    if identity is None:
        # Legacy catalogs remain readable only in their original namespace.
        # The new price namespace can be entered only by explicit migration.
        raise RuntimeError("legacy catalog requires explicit price-identity migration")
    differences = price_identity_differences(
        identity, compute_elastic_graph_price_identity(vllm_config, kv_cache_config)
    )
    if differences:
        raise RuntimeError(f"elastic price compatibility mismatch: {differences}")
    if payload.get("resident_owner_generation") != PRICE_OWNER_GENERATION:
        raise RuntimeError("elastic catalog has no portable resident-owner identity")
    return remap_catalog_resident_keys(
        rows,
        source_generation=PRICE_OWNER_GENERATION,
        destination_generation=elastic_catalog_owner_generation(
            vllm_config, kv_cache_config
        ),
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        compiled_piecewise_sizes=tuple(
            configured_compiled_piecewise_sizes(vllm_config)
        ),
        policy=_effective_graph_execution_policy(kv_cache_config),
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
        tuple(step_key),  # type: ignore[arg-type]  # validated five-field wire key
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


def _validated_catalog_resident_key_bytes(
    raw: Any,
    *,
    resident_bytes: int,
    cold_peak_bytes: int,
) -> tuple[tuple[str, int], ...] | None:
    """Validate receipt-key byte provenance carried by one measured endpoint."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError("resident-key provenance must be a sequence")
    pairs: list[tuple[str, int]] = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or len(item[0]) != 64
            or any(character not in "0123456789abcdef" for character in item[0])
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 0
        ):
            raise RuntimeError(
                "resident-key provenance requires lowercase SHA256 identities "
                "and non-negative byte counts"
            )
        pairs.append((item[0], item[1]))
    if [identity for identity, _value in pairs] != sorted(
        {identity for identity, _value in pairs}
    ):
        raise RuntimeError("resident-key provenance must be sorted and unique")
    if sum(value for _identity, value in pairs) > min(resident_bytes, cold_peak_bytes):
        raise RuntimeError("resident-key provenance exceeds its measured endpoint")
    return tuple(pairs)


def load_elastic_graph_catalog(
    vllm_config: VllmConfig, kv_cache_config: Any, *, catalog_path: str | None = None
) -> dict[tuple[int, ...], dict[str, Any]]:
    """Load only identity-matched, complete cold+hot shape measurements."""
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return {}
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    from vllm.v1.core.elastic_graph import (
        configured_compiled_piecewise_sizes,
    )

    path = catalog_path or _elastic_graph_catalog_path(fingerprint)
    try:
        os.lstat(path)
    except FileNotFoundError:
        logger.warning(
            "Elastic CUDA Graph price catalog is absent: "
            "reason=compatibility-unresolved path=%s fingerprint=%s; "
            "absence is not a measurement decision",
            path,
            fingerprint,
        )
        return {}
    except OSError as error:
        raise RuntimeError(
            "canonical elastic Graph catalog path cannot be inspected; "
            f"refusing automatic calibration over it: {path}: {error}"
        ) from error
    try:
        payload, source_sha256 = load_sealed_catalog_with_digest(
            path, require_migration=False
        )
    except RuntimeError as error:
        if "migration lineage" in str(error) or "offline-finalized" in str(error):
            raise RuntimeError(
                "serving requires an offline-finalized elastic Graph catalog "
                "without migration lineage fields"
            ) from error
        raise RuntimeError(
            "canonical elastic Graph catalog exists but is unreadable; "
            f"refusing automatic calibration over it: {path}: {error}"
        ) from error
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
        raise RuntimeError(
            "canonical elastic Graph catalog has stale, unsealed, or "
            f"identity-mismatched contents: {path}"
        )
    coverage = payload.get("coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("semantic_witness_contract")
        != "live-current-to-physical-piecewise-v1"
        or not isinstance(coverage.get("semantic_token_witnesses"), list)
    ):
        raise RuntimeError(
            f"canonical elastic Graph catalog has no semantic witness identity: {path}"
        )
    required_inventory, _restore_inventory = validate_elastic_catalog_key_inventory(
        coverage.get("required_step_keys"),
        coverage.get("restore_step_keys", []),
        label="canonical elastic Graph catalog",
    )
    required_step_keys = set(required_inventory)
    decode_k = int(vllm_config.num_speculative_tokens)
    speculative_config = vllm_config.speculative_config
    prefill_k = (
        0
        if speculative_config is not None
        and speculative_config.disable_speculation_on_non_decode
        else decode_k
    )
    if any(
        key[1] != (decode_k if key[0] == 1 else prefill_k) for key in required_inventory
    ):
        raise RuntimeError(
            "canonical elastic Graph catalog differs from phase-specific K"
        )
    declared_witnesses = tuple(
        sorted(
            (
                tuple(witness["step_key"]),
                witness["live_num_tokens"],
            )
            for witness in coverage["semantic_token_witnesses"]
        )
    )
    expected_witnesses = expected_semantic_token_witnesses(
        configured_k=decode_k,
        prefill_k=prefill_k,
        max_num_seqs=coverage.get("decode_max_x", 0),
        max_num_batched_tokens=(vllm_config.scheduler_config.max_num_batched_tokens),
    )
    if declared_witnesses != expected_witnesses:
        raise RuntimeError(
            "canonical elastic Graph catalog semantic witnesses differ "
            "from the effective runtime"
        )
    shapes = payload.get("shapes")
    if (
        not isinstance(shapes, list)
        or payload.get("complete_shapes") != len(shapes)
        or coverage.get("required_shapes") != len(required_step_keys)
    ):
        raise RuntimeError(
            f"sealed elastic Graph catalog shape coverage is inconsistent: {path}"
        )
    policy = _effective_graph_execution_policy(kv_cache_config)
    compiled_piecewise_sizes = configured_compiled_piecewise_sizes(vllm_config)
    representation = (
        coverage.get("representation", "pinned_full_family")
        if isinstance(coverage, dict)
        else "pinned_full_family"
    )
    result = ElasticGraphCatalog(source_sha256=source_sha256)
    for row in shapes:
        if not isinstance(row, dict):
            raise RuntimeError("elastic Graph catalog contains a non-object row")
        key = row.get("step_key")
        if (
            not isinstance(key, list)
            or len(key) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in key
            )
        ):
            raise RuntimeError("elastic Graph catalog contains an invalid step key")
        catalog_key = tuple(key)
        if catalog_key in result:
            raise RuntimeError(
                f"elastic Graph catalog contains duplicate step key: {key!r}"
            )
        expected_policy_metadata = _catalog_policy_row_metadata(
            key,
            policy=policy,
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            compiled_piecewise_sizes=compiled_piecewise_sizes,
        )
        if any(
            row.get(name) != value for name, value in expected_policy_metadata.items()
        ):
            raise RuntimeError(
                "elastic Graph catalog row has stale representation lineage: "
                f"step_key={key!r}"
            )
        fields: dict[str, Any] = {
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
        finalized_pinned_owner_set = row.get("finalized_pinned_owner_set", 0)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fields.values()
        ) or (
            isinstance(finalized_pinned_owner_set, bool)
            or finalized_pinned_owner_set not in {0, 1}
        ):
            raise RuntimeError(
                "elastic Graph catalog row has invalid numeric fields: "
                f"step_key={key!r}"
            )
        try:
            resident_key_bytes = _validated_catalog_resident_key_bytes(
                row.get("resident_key_bytes"),
                resident_bytes=fields["resident_bytes"],
                cold_peak_bytes=fields["cold_peak_bytes"],
            )
        except RuntimeError as error:
            raise RuntimeError(
                "elastic Graph catalog row has invalid resident-key provenance: "
                f"step_key={key!r}"
            ) from error
        complete_row = dict(fields)
        if "allocation_profile" in row:
            from vllm.v1.core.elastic_memory_profile import (
                validate_allocation_envelope_row,
            )

            try:
                validate_allocation_envelope_row(
                    key,
                    row,
                    policy=policy.fingerprint,
                    budget=vllm_config.scheduler_config.max_num_batched_tokens,
                )
            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError("invalid allocation profile catalog row") from error
            fields["allocation_profile"] = row["allocation_profile"]
            complete_row["allocation_profile"] = row["allocation_profile"]
        if finalized_pinned_owner_set:
            complete_row["finalized_pinned_owner_set"] = finalized_pinned_owner_set
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
            raise RuntimeError(
                f"elastic Graph catalog row is incomplete: step_key={key!r}"
            )
        if finalized_pinned_owner_set:
            fields["finalized_pinned_owner_set"] = finalized_pinned_owner_set
        if resident_key_bytes is not None:
            fields["resident_key_bytes"] = resident_key_bytes
        result[catalog_key] = fields  # type: ignore[assignment]
    if set(result) != required_step_keys:
        raise RuntimeError(
            "sealed elastic Graph catalog does not exactly cover required shapes: "
            f"missing={sorted(required_step_keys.difference(result))!r} "
            f"extra={sorted(set(result).difference(required_step_keys))!r}"
        )
    result.update(
        _bind_catalog_price_owners(payload, result, vllm_config, kv_cache_config)
    )
    logger.info(
        "Loaded elastic CUDA Graph catalog %s (%d complete shapes)",
        path,
        len(result),
    )
    return result


def load_elastic_graph_catalog_coverage(
    vllm_config: VllmConfig, kv_cache_config: Any, *, catalog_path: str | None = None
) -> dict[str, Any]:
    """Load the sealed product boundary paired with the price catalog.

    Shape rows may include probes above the accepted fixed point.  Serving must
    therefore consume the explicitly sealed boundary rather than infer MaxX
    from the largest complete historical row.
    """
    fingerprint = compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    from vllm.v1.core.elastic_graph import (
        EXECUTION_MANIFEST_SCHEMA,
        DispatchRepresentation,
    )

    path = catalog_path or _elastic_graph_catalog_path(fingerprint)
    try:
        payload, source_sha256 = load_sealed_catalog_with_digest(
            path, require_migration=False
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"sealed elastic Graph catalog coverage is unreadable: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"sealed elastic Graph catalog coverage is malformed: {path}"
        )
    coverage = payload.get("coverage")
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
        coverage.get("restore_step_keys", []) if isinstance(coverage, dict) else None
    )
    required_step_keys = (
        coverage.get("required_step_keys", []) if isinstance(coverage, dict) else None
    )
    _required_inventory, restore_inventory = validate_elastic_catalog_key_inventory(
        required_step_keys,
        restore_step_keys,
        label="sealed elastic Graph catalog",
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
        or not _catalog_policy_matches(payload.get("graph_execution_policy"), policy)
        or not isinstance(coverage, dict)
        or coverage.get("graph_execution_policy_fingerprint") != policy.fingerprint
        or coverage.get("verifier_contract") != policy.verifier_contract
        or coverage.get("verifier_configuration") != policy.verifier_configuration
        or coverage.get("math_contract") != policy.math_contract
        or coverage.get("capture_state_abi") != ELASTIC_CAPTURE_STATE_ABI
        or coverage.get("execution_manifest_schema") != EXECUTION_MANIFEST_SCHEMA
        or coverage.get("dispatch_representations")
        != [item.value for item in DispatchRepresentation]
        or coverage.get("residency_intent_contract")
        != "current-dispatch-hot-successor-union-v1"
        or coverage.get("speculative_depth_contract")
        != "scheduled-requested-executed-k-v1"
        or coverage.get("semantic_witness_contract")
        != "live-current-to-physical-piecewise-v1"
        or not isinstance(coverage.get("semantic_token_witnesses"), list)
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
                coverage["pinned_full_entries"] < 1 or coverage["pinned_full_bytes"] < 1
            )
        )
        or (
            representation == "bounded_exact_hotset"
            and (
                coverage["pinned_full_entries"] != 0
                or coverage["pinned_full_bytes"] != 0
                or not restore_inventory
                or coverage.get("piecewise_replay_contract")
                != BOUNDED_PIECEWISE_REPLAY_CONTRACT
            )
        )
    ):
        raise RuntimeError(
            f"sealed elastic Graph catalog has no valid product boundary: {path}"
        )
    if "restore_decode" in coverage:
        from vllm.v1.core.elastic_catalog import RestoreDecodeGeometry

        geometry = RestoreDecodeGeometry.from_payload(coverage["restore_decode"])
        budget = vllm_config.scheduler_config.max_num_batched_tokens
        k = int(vllm_config.num_speculative_tokens)
        geometry.validate_runtime(
            configured_k=k,
            max_num_seqs=coverage["decode_max_x"],
            max_num_batched_tokens=budget,
        )
        speculative = vllm_config.speculative_config
        prefill_k = (
            0
            if speculative is not None and speculative.disable_speculation_on_non_decode
            else k
        )
        if set(
            geometry.step_keys(
                policy=policy,
                prefill_k=prefill_k,
                max_num_batched_tokens=budget,
            )
        ) != set(restore_inventory):
            raise RuntimeError("sealed catalog restore pair differs from runtime")
    return dict(coverage) | {"_catalog_source_sha256": source_sha256}


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
            -1 if item["physical_num_reqs"] is None else item["physical_num_reqs"],
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
