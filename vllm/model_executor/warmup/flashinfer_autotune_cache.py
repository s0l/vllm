# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    factors = aot_compile_hash_factors(runner.vllm_config)
    return hashlib.sha256(str(factors).encode()).hexdigest()


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def synchronize_flashinfer_autotune_cache(
    *,
    cache_path: Path,
    world: Any,
    tuner: Any,
    save_leader: bool,
) -> bool:
    """Replace rank-local tuning results with rank 0's serialized choices."""
    is_leader = world.rank_in_group == 0
    if is_leader and save_leader:
        tuner.save_configs(str(cache_path))

    tune_results = (
        cache_path.read_bytes() if is_leader and cache_path.exists() else None
    )
    tune_results = world.broadcast_object(tune_results, src=0)
    if tune_results is None:
        return False

    write_flashinfer_autotune_cache(cache_path, tune_results)
    world.barrier()
    tuner.clear_cache()
    if not tuner.load_configs(str(cache_path)):
        raise RuntimeError("Failed to load synchronized FlashInfer autotune configs")
    return True
