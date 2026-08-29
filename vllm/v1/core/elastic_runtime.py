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

ELASTIC_RUNTIME_GENERATION_SCHEMA_VERSION = 2


def elastic_runtime_source_hashes() -> dict[str, str]:
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
    }
    return compute_elastic_runtime_generation_from_factors(factors)
