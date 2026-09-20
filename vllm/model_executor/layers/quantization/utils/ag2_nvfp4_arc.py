# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed loader for the default-off AG2 K3 ARC variable-K sidecar."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
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
    applied_prefixes: set[str] = field(default_factory=set)


_CACHE: dict[tuple[str, int, str], _Sidecar] = {}
_OWNER_WIDTHS = (2048, 2048, 1024)
_OWNER_OFFSETS = (0, 2048, 4096)


def _owner_runtime_active() -> bool:
    return os.environ.get("AG2_VLLM_TP3_OWNER_PREQUANT", "0") == "1"


def _record_applied_prefix(sidecar: _Sidecar, prefix: str, rank: int) -> None:
    """Record exact manifest coverage without emitting one INFO line per layer."""
    if prefix in sidecar.applied_prefixes:
        raise RuntimeError(f"ARC sidecar applied twice to {prefix}")
    sidecar.applied_prefixes.add(prefix)
    applied_count = len(sidecar.applied_prefixes)
    expected_count = len(sidecar.by_suffix)
    if applied_count > expected_count:
        raise RuntimeError(
            "ARC applied-prefix count exceeded manifest: "
            f"rank={rank} applied={applied_count} expected={expected_count}"
        )
    if applied_count == expected_count:
        logger.info(
            "AG2 ARC application complete rank=%d records=%d sha256=%s",
            rank,
            expected_count,
            sidecar.manifest["sidecar_sha256"],
        )


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
        "ag2-k3-arc-hybrid-sidecar-manifest-v3": ("ag2-k3-arc-hybrid-sidecar-v3"),
    }.get(manifest_schema)
    record_count = manifest.get("record_count")
    tensor_count = manifest.get("tensor_count")
    expected_records = (
        241 if manifest_schema == "ag2-k3-arc-hybrid-sidecar-manifest-v3" else 128
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
        raise RuntimeError(f"ARC sidecar SHA mismatch for rank {rank}: {actual_sha}")
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
        or tuple(extra_weight.shape) != (record["output_features"], selected_count // 2)
        or tuple(extra_scale.shape) != (record["output_features"], selected_count // 16)
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
    if _owner_runtime_active():
        selected_cpu: list[torch.Tensor] = []
        for peer_rank in range(3):
            peer_manifest_path = Path(directory_text) / f"rank{peer_rank}.manifest.json"
            peer_sidecar_path = Path(directory_text) / f"rank{peer_rank}.safetensors"
            peer_manifest = json.loads(peer_manifest_path.read_text(encoding="utf-8"))
            peer_matches = [
                item
                for item in peer_manifest.get("records", ())
                if item.get("runtime_suffix") == suffix
            ]
            if (
                peer_manifest.get("tp_rank") != peer_rank
                or len(peer_matches) != 1
                or peer_matches[0].get("selected_count") != selected_count
            ):
                raise RuntimeError(
                    f"ARC peer selected manifest mismatch for {prefix} rank={peer_rank}"
                )
            peer_record = peer_matches[0]
            with safe_open(peer_sidecar_path, framework="pt", device="cpu") as handle:
                peer_selected = handle.get_tensor(
                    f"{peer_record['key']}.selected"
                ).contiguous()
            peer_digest = hashlib.sha256(peer_selected.numpy().tobytes()).hexdigest()
            if (
                peer_selected.dtype != torch.int32
                or tuple(peer_selected.shape) != (selected_count,)
                or peer_digest != peer_record.get("selected_sha256")
            ):
                raise RuntimeError(
                    f"ARC peer selected tensor mismatch for {prefix} rank={peer_rank}"
                )
            selected_cpu.append(peer_selected)
        selected_all_cpu = torch.stack(selected_cpu)
        layer.register_buffer(
            "_ag2_nvfp4_arc_selected_all",
            selected_all_cpu.to(device=layer.weight.device),
            persistent=False,
        )
        route_counts = [[0] * 3 for _ in range(3)]
        destination_positions: list[list[torch.Tensor]] = []
        for destination in range(3):
            per_owner: list[torch.Tensor] = []
            for owner in range(3):
                start = _OWNER_OFFSETS[owner]
                stop = start + _OWNER_WIDTHS[owner]
                positions = torch.nonzero(
                    (selected_all_cpu[destination] >= start)
                    & (selected_all_cpu[destination] < stop),
                    as_tuple=False,
                ).flatten()
                per_owner.append(positions)
                route_counts[owner][destination] = positions.numel()
            if sum(item.numel() for item in per_owner) != selected_count:
                raise RuntimeError(f"ARC selected ownership incomplete for {prefix}")
            destination_positions.append(per_owner)
        for destination in range(3):
            positions = destination_positions[destination][rank]
            local_indices = (
                selected_all_cpu[destination].index_select(0, positions)
                - _OWNER_OFFSETS[rank]
            ).to(torch.int64)
            layer.register_buffer(
                f"_ag2_nvfp4_arc_route_to_{destination}",
                local_indices.to(device=layer.weight.device),
                persistent=False,
            )
        inverse_order = torch.argsort(torch.cat(destination_positions[rank])).to(
            torch.int64
        )
        layer.register_buffer(
            "_ag2_nvfp4_arc_inverse_order",
            inverse_order.to(device=layer.weight.device),
            persistent=False,
        )
        layer._ag2_nvfp4_arc_route_counts = tuple(
            value for owner in route_counts for value in owner
        )
    layer._ag2_nvfp4_arc_runtime_suffix = suffix
    layer._ag2_nvfp4_arc_sidecar_sha256 = sidecar.manifest["sidecar_sha256"]
    layer._ag2_nvfp4_owner_metadata_mode = "model-bound-arc"
    _record_applied_prefix(sidecar, prefix, rank)
    logger.debug(
        "AG2 ARC appended prefix=%s rank=%d K=%d->%d",
        prefix,
        rank,
        record["base_k"],
        record["augmented_k"],
    )
    return True


def ensure_ag2_nvfp4_base_owner_metadata(layer: torch.nn.Module) -> bool:
    """Install the model-independent zero-tail owner ABI when ARC is absent."""
    if not _owner_runtime_active():
        return False
    if hasattr(layer, "_ag2_nvfp4_arc_selected_all"):
        return False
    if not hasattr(layer, "weight") or layer.weight.ndim != 2:
        raise RuntimeError("base owner metadata requires a loaded matrix weight")
    device = layer.weight.device
    world = len(_OWNER_WIDTHS)
    layer.register_buffer(
        "_ag2_nvfp4_arc_selected",
        torch.empty(0, dtype=torch.int32, device=device),
        persistent=False,
    )
    layer.register_buffer(
        "_ag2_nvfp4_arc_selected_all",
        torch.empty((world, 0), dtype=torch.int32, device=device),
        persistent=False,
    )
    for destination in range(world):
        layer.register_buffer(
            f"_ag2_nvfp4_arc_route_to_{destination}",
            torch.empty(0, dtype=torch.int64, device=device),
            persistent=False,
        )
    layer.register_buffer(
        "_ag2_nvfp4_arc_inverse_order",
        torch.empty(0, dtype=torch.int64, device=device),
        persistent=False,
    )
    layer._ag2_nvfp4_arc_route_counts = (0,) * (world * world)
    layer._ag2_nvfp4_owner_metadata_mode = "native-base-only"
    return True
