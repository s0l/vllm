# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared deterministic identity for the elastic Graph runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.core.elastic_graph import (
    compute_elastic_runtime_generation_from_factors,
)

ELASTIC_RUNTIME_GENERATION_SCHEMA_VERSION = 3

_ELASTIC_RUNTIME_SOURCE_MODULES = (
    "vllm.model_executor.warmup.flashinfer_autotune_cache",
    "vllm.model_executor.warmup.kernel_warmup",
    "vllm.distributed.parallel_state",
    "vllm.distributed.device_communicators.tp3_exact_reduce",
    "vllm.distributed.device_communicators.tp3_ce_all_reduce",
    "vllm.v1.attention.backends.flashinfer",
    "vllm.v1.core.kv_cache_capacity",
    "vllm.v1.core.kv_cache_coordinator",
    "vllm.v1.core.elastic_graph",
    "vllm.v1.core.elastic_memory_profile",
    "vllm.v1.core.elastic_price_identity",
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.engine.core",
    "vllm.v1.engine.elastic_calibrator",
    "vllm.v1.engine.elastic_memory_profile",
    "vllm.v1.engine.elastic_bootstrap",
    "vllm.v1.sample.ops.topk_topp_sampler",
    "vllm.v1.worker.elastic_catalog_tool",
    "vllm.v1.worker.gpu.attn_utils",
    "vllm.v1.worker.gpu.cudagraph_utils",
    "vllm.v1.worker.gpu.elastic_gdn",
    "vllm.v1.worker.gpu.mm.encoder_runner",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu.model_states.default",
    "vllm.v1.worker.gpu.model_states.encoder_decoder",
    "vllm.v1.worker.gpu.model_states.interface",
    "vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils",
    "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
    "vllm.v1.worker.gpu.spec_decode.speculator",
    "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator",
    "vllm.v1.worker.startup_plan",
    # Effective Exp25/Qwen3.8 target and draft graph owners. Hash the model
    # composition, shared attention implementation, MTP implementation, and
    # fused primitives independently: any can change graph memory, launch
    # topology, or draft/target numerics without changing VllmConfig.
    "vllm.model_executor.models.qwen3_5",
    "vllm.model_executor.models.qwen3_next",
    "vllm.model_executor.models.qwen3_next_row",
    "vllm.model_executor.models.qwen3_next_ready",
    "vllm.model_executor.models.qwen3_next_exact_qk",
    "vllm.model_executor.models.qwen3_next_exact_qk_kernels",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_exact_norm",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_exact_norm_kernels",
    "vllm.model_executor.models.qwen3_5_mtp",
    "vllm.model_executor.layers.fused_qk_norm_rope",
    "vllm.model_executor.layers.rotary_embedding",
    # FlashNext uses the new model package, not the legacy Qwen wrappers.
    # Its model-specific allocations, expert residency and breakable graph
    # producers must invalidate both startup budgets and physical prices.
    "vllm.models.qwen4_exp",
    "vllm.models.qwen4_exp.config",
    "vllm.models.qwen4_exp.common.hyperconnection",
    "vllm.models.qwen4_exp.common.ple",
    "vllm.models.qwen4_exp.common.qsa_cache",
    "vllm.models.qwen4_exp.nvidia.model",
    "vllm.models.qwen4_exp.nvidia.model_state",
    "vllm.models.qwen4_exp.nvidia.mtp",
    "vllm.models.qwen4_exp.nvidia.mtp_partition",
    "vllm.models.qwen4_exp.nvidia.hyperconnection",
    "vllm.models.qwen4_exp.nvidia.low_latency_gemm",
    "vllm.models.qwen4_exp.nvidia.ple_layer",
    "vllm.models.qwen4_exp.nvidia.ple_offload",
    "vllm.models.qwen4_exp.nvidia.indexer_qsa",
    "vllm.models.qwen4_exp.nvidia.qsa",
    "vllm.models.qwen4_exp.nvidia.qsa_flashinfer",
    "vllm.models.qwen4_exp.nvidia.qsa_qkv",
    "vllm.models.qwen4_exp.nvidia.ops.hc",
    "vllm.models.qwen4_exp.nvidia.ops.ple",
    "vllm.models.qwen4_exp.nvidia.ops.qsa",
    "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
    "vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer",
    "vllm.models.qwen4_exp.nvidia.expert_offload_bank",
    "vllm.models.qwen4_exp.nvidia.expert_offload_moe",
    "vllm.models.qwen4_exp.nvidia.expert_offload_plan",
    "vllm.models.qwen4_exp.nvidia.expert_offload_provider",
    "vllm.models.qwen4_exp.nvidia.expert_offload_residency",
    "vllm.models.qwen4_exp.nvidia.expert_offload_source",
    "vllm.models.qwen4_exp.nvidia.expert_offload_tables",
    "vllm.third_party.flash_linear_attention.ops.layernorm_guard",
    "vllm.utils.nvfp4_expert_geometry",
    "vllm.compilation.breakable_cudagraph",
    "vllm._custom_ops",
    "vllm.model_executor.layers.fused_moe.experts.cutlass_moe",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner",
    "vllm.model_executor.layers.mamba.abstract",
    "vllm.model_executor.warmup.qwen4_exp_qsa_warmup",
    "vllm.v1.core.elastic_expert",
    "vllm.v1.kv_cache_interface",
    "vllm.v1.worker.gpu.block_table",
)

# A serving/runtime generation protects scheduler-worker protocol and mutable
# state, so it intentionally follows the complete source inventory above.  A
# measured catalog has a narrower consumed contract: physical CUDA Graph/KV
# coexistence for explicit replay keys.  Scheduler admission and residency
# policy may change while those rows remain valid; required-key, policy and
# execution-manifest validation at load time still rejects a semantic surface
# change.  Keep modules that own graph construction, model math, collectives,
# calibration measurement, or physical memory here.
_ELASTIC_PHYSICAL_CATALOG_SOURCE_MODULES = tuple(
    module_name
    for module_name in _ELASTIC_RUNTIME_SOURCE_MODULES
    if module_name
    not in {
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.worker.elastic_catalog_tool",
        "vllm.v1.worker.startup_plan",
        "vllm.v1.core.elastic_price_identity",
        "vllm.v1.engine.elastic_bootstrap",
    }
)


def _source_hashes(module_names: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for module_name in module_names:
        spec = importlib.util.find_spec(module_name)
        path = spec.origin if spec is not None else None
        if path is None:
            raise RuntimeError(
                f"cannot resolve elastic runtime source for {module_name}"
            )
        with open(path, "rb") as stream:
            result[module_name] = hashlib.sha256(stream.read()).hexdigest()
    return result


def elastic_runtime_source_hashes() -> dict[str, str]:
    """Bind runtime state to every Python owner in the elastic protocol."""
    return _source_hashes(_ELASTIC_RUNTIME_SOURCE_MODULES)


def elastic_catalog_physical_source_hashes() -> dict[str, str]:
    """Bind catalog rows only to code that can change their physical cost."""
    from vllm.v1.core.elastic_price_identity import source_semantic_digest

    result = {}
    for module_name in _ELASTIC_PHYSICAL_CATALOG_SOURCE_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve physical price producer: {module_name}")
        with open(spec.origin, encoding="utf-8") as stream:
            result[module_name] = source_semantic_digest(stream.read())
    return result


def elastic_auto_calibration_enabled() -> bool:
    """Admit automatic calibration only in an explicit maintenance job."""
    requested = os.environ.get("AG2_VLLM_ELASTIC_AUTO_CALIBRATE", "0") == "1"
    role = os.environ.get("AG2_VLLM_ELASTIC_CALIBRATION_ROLE", "")
    if requested and role != "maintenance":
        raise RuntimeError(
            "automatic elastic calibration is maintenance-only; set "
            "AG2_VLLM_ELASTIC_CALIBRATION_ROLE=maintenance in an explicit "
            "calibration job, or disable AG2_VLLM_ELASTIC_AUTO_CALIBRATE "
            "for serving"
        )
    return requested


def elastic_profile_config_factors(vllm_config: VllmConfig) -> dict[str, Any]:
    """Return profile-memory factors omitted by graph identity hashing."""
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


def compute_elastic_runtime_generation(vllm_config: VllmConfig) -> str:
    """Return the content namespace shared by scheduler and Graph owners."""
    from vllm import __version__ as vllm_version

    try:
        device_name = current_platform.get_device_name()
        device_capability = str(current_platform.get_device_capability() or "")
    except NotImplementedError:
        device_name = "platform-unavailable"
        device_capability = "platform-unavailable"
    config_hash = vllm_config.compute_hash()
    if not isinstance(config_hash, str):
        config_hash = "config-hash-unavailable"
    factors = {
        "schema": ELASTIC_RUNTIME_GENERATION_SCHEMA_VERSION,
        "vllm": vllm_version,
        "vllm_config": config_hash,
        "profile_config": elastic_profile_config_factors(vllm_config),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "device_name": device_name,
        "device_capability": device_capability,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "elastic_mm_activation_loan_bytes": os.environ.get(
            "AG2_VLLM_ELASTIC_MM_ACTIVATION_LOAN_BYTES", ""
        ),
        "runtime_source_hashes": elastic_runtime_source_hashes(),
        "row_profile_sha256": os.environ.get("AG2_VLLM_TP3_ROW_PROFILE_SHA256", ""),
    }
    return compute_elastic_runtime_generation_from_factors(factors)
