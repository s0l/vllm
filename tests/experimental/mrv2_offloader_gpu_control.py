"""Exact-image GPU control for sequential model-layer offloading."""

import gc
import json

import torch

from vllm.config import OffloadConfig
from vllm.config.offload import UVAOffloadConfig
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)

MIB = 1024**2
MODULE_BYTES = 8192 * 8192 * 2


def module_generator(observed_allocated_bytes: list[int]):
    for _ in range(3):
        module = torch.nn.Linear(
            8192,
            8192,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        observed_allocated_bytes.append(torch.cuda.memory_allocated(0))
        yield module


def run_control(cpu_offload_gb: float):
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(0)
    config = OffloadConfig(
        offload_backend="auto",
        uva=UVAOffloadConfig(cpu_offload_gb=cpu_offload_gb),
    )
    set_offloader(create_offloader(config))
    observed: list[int] = []
    modules = get_offloader().wrap_modules(module_generator(observed))
    final = torch.cuda.memory_allocated(0)
    offloaded = int(getattr(get_offloader(), "cpu_offload_bytes", 0))
    result = {
        "offloader": type(get_offloader()).__name__,
        "baseline_bytes": baseline,
        "observed_allocated_bytes": observed,
        "final_allocated_bytes": final,
        "offloaded_bytes": offloaded,
    }
    del modules
    set_offloader(create_offloader(OffloadConfig()))
    gc.collect()
    torch.cuda.empty_cache()
    return result


negative = run_control(0)
positive = run_control(0.25)
checks = {
    "negative_is_noop": negative["offloader"] == "NoopOffloader",
    "negative_retains_three_modules": (
        negative["final_allocated_bytes"] >= 3 * MODULE_BYTES
    ),
    "positive_is_uva": positive["offloader"] == "UVAOffloader",
    "positive_offloads_two_modules": positive["offloaded_bytes"]
    == 2 * MODULE_BYTES,
    "positive_retains_only_one_module": (
        positive["final_allocated_bytes"] <= MODULE_BYTES + 8 * MIB
    ),
    "positive_frees_before_next_construction": max(
        positive["observed_allocated_bytes"]
    )
    <= 2 * MODULE_BYTES + 8 * MIB,
}
artifact = {
    "negative": negative,
    "positive": positive,
    "checks": checks,
    "passed": all(checks.values()),
}
print(json.dumps(artifact, indent=2))
if not artifact["passed"]:
    raise SystemExit(1)
