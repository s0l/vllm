# SPDX-License-Identifier: Apache-2.0
"""Fail-closed loader for the default-off AG2 K3 ARC variable-K sidecar."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _Sidecar:
    manifest: dict
    by_suffix: dict[str, dict]
    tensors: dict[str, torch.Tensor]


_CACHE: dict[tuple[str, int, str], _Sidecar] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tp_rank() -> int:
    diagnostic = os.environ.get("AG2_VLLM_NVFP4_ARC_TP_RANK")
    if diagnostic is not None:
        rank = int(diagnostic)
    else:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    if rank not in (0, 1, 2):
        raise RuntimeError(f"ARC requires TP rank 0..2, got {rank}")
    return rank


def _load(directory: Path, rank: int, device: torch.device) -> _Sidecar:
    cache_key = (str(directory.resolve()), rank, str(device))
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    manifest_path = directory / f"rank{rank}.manifest.json"
    sidecar_path = directory / f"rank{rank}.safetensors"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_schema = manifest.get("schema")
    tensor_schema = {
        "ag2-k3-arc-k64-sidecar-manifest-v1": "ag2-k3-arc-k64-sidecar-v1",
        "ag2-k3-arc-variable-k-sidecar-manifest-v2": (
            "ag2-k3-arc-variable-k-sidecar-v2"
        ),
        "ag2-k3-arc-hybrid-sidecar-manifest-v3": (
            "ag2-k3-arc-hybrid-sidecar-v3"
        ),
    }.get(manifest_schema)
    record_count = manifest.get("record_count")
    tensor_count = manifest.get("tensor_count")
    expected_records = (
        241
        if manifest_schema == "ag2-k3-arc-hybrid-sidecar-manifest-v3"
        else 128
    )
    runtime_prefix_root = manifest.get("runtime_prefix_root")
    if (
        tensor_schema is None
        or not manifest.get("ok")
        or manifest.get("tp_rank") != rank
        or record_count != expected_records
        or tensor_count != 3 * expected_records
        or (
            manifest_schema == "ag2-k3-arc-hybrid-sidecar-manifest-v3"
            and runtime_prefix_root != "language_model.model."
        )
    ):
        raise RuntimeError(f"invalid ARC sidecar manifest {manifest_path}")
    actual_sha = _sha256(sidecar_path)
    if actual_sha != manifest.get("sidecar_sha256"):
        raise RuntimeError(
            f"ARC sidecar SHA mismatch for rank {rank}: {actual_sha}"
        )
    with safe_open(sidecar_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if (
            metadata.get("schema") != tensor_schema
            or metadata.get("tp_rank") != str(rank)
            or len(list(handle.keys())) != tensor_count
        ):
            raise RuntimeError(f"invalid ARC safetensors metadata {sidecar_path}")
    tensors = load_file(sidecar_path, device=str(device))
    by_suffix = {record["runtime_suffix"]: record for record in manifest["records"]}
    if len(by_suffix) != record_count:
        raise RuntimeError("ARC runtime suffix set is incomplete or duplicated")
    result = _Sidecar(manifest=manifest, by_suffix=by_suffix, tensors=tensors)
    _CACHE[cache_key] = result
    logger.info(
        "Loaded AG2 ARC sidecar rank=%d sha256=%s records=%d bytes=%d schema=%s",
        rank,
        actual_sha,
        len(by_suffix),
        manifest["total_tensor_bytes"],
        manifest_schema,
    )
    return result


def maybe_apply_ag2_nvfp4_arc_sidecar(
    layer: torch.nn.Module, prefix: str | None
) -> bool:
    directory_text = os.environ.get("AG2_VLLM_NVFP4_ARC_SIDECAR_DIR", "").strip()
    if not directory_text:
        return False
    if not prefix:
        raise RuntimeError("ARC sidecar enabled but NVFP4 layer prefix is missing")
    rank = _tp_rank()
    sidecar = _load(Path(directory_text), rank, layer.weight.device)
    runtime_prefix_root = sidecar.manifest.get("runtime_prefix_root")
    if runtime_prefix_root and not prefix.startswith(runtime_prefix_root):
        return False
    matches = [
        (suffix, record)
        for suffix, record in sidecar.by_suffix.items()
        if prefix.endswith(suffix)
    ]
    if not matches:
        return False
    if len(matches) != 1:
        raise RuntimeError(f"ARC prefix {prefix!r} matches multiple sidecar records")
    suffix, record = matches[0]
    key = record["key"]
    selected = sidecar.tensors[f"{key}.selected"]
    extra_weight = sidecar.tensors[f"{key}.weight"]
    extra_scale = sidecar.tensors[f"{key}.weight_scale"]
    selected_count = record["augmented_k"] - record["base_k"]
    if (
        layer.weight.dtype != torch.uint8
        or layer.weight.shape[1] * 2 != record["base_k"]
        or layer.weight_scale.dtype != torch.float8_e4m3fn
        or layer.weight_scale.shape[1] != record["base_k"] // 16
        or layer.weight.shape[0] != record["output_features"]
        or selected_count not in (64, 256, 512)
        or tuple(extra_weight.shape)
        != (record["output_features"], selected_count // 2)
        or tuple(extra_scale.shape)
        != (record["output_features"], selected_count // 16)
        or selected.dtype != torch.int32
        or tuple(selected.shape) != (selected_count,)
        or int(selected.min()) < 0
        or int(selected.max()) >= record["base_k"]
    ):
        raise RuntimeError(f"ARC sidecar ABI mismatch for {prefix}")
    base_weight = layer.weight.data
    base_scale = layer.weight_scale.data
    augmented_weight = torch.cat((base_weight, extra_weight), dim=1).contiguous()
    augmented_scale = torch.cat((base_scale, extra_scale), dim=1).contiguous()
    if not (
        torch.equal(augmented_weight[:, : base_weight.shape[1]], base_weight)
        and torch.equal(augmented_scale[:, : base_scale.shape[1]], base_scale)
    ):
        raise RuntimeError(f"ARC base payload changed while appending {prefix}")
    layer.weight = torch.nn.Parameter(augmented_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(augmented_scale, requires_grad=False)
    layer.register_buffer(
        "_ag2_nvfp4_arc_selected", selected.contiguous(), persistent=False
    )
    layer._ag2_nvfp4_arc_runtime_suffix = suffix
    layer._ag2_nvfp4_arc_sidecar_sha256 = sidecar.manifest["sidecar_sha256"]
    logger.info(
        "AG2 ARC appended prefix=%s rank=%d K=%d->%d",
        prefix,
        rank,
        record["base_k"],
        record["augmented_k"],
    )
    return True
