# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable host-resident E8 expert archive used by FlashNext serving.

The serving layout intentionally stores direct uint16 E8 code indices.  QuIP#'s
packed int64 layout has the same byte count, but is canonicalized by the archive
builder so the fused consumer never materializes or rearranges weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from vllm.utils.e8_expert_config import E8ArchiveConfig

FORMAT = "flashnext-e8-direct-v1"
OWNERS = (205, 205, 102)
FIELDS = {
    "q_gu": ("<u2", (1280, 320)),
    "q_down": ("<u2", (2560, 80)),
    "su_gu": ("<f2", (2560,)),
    "sv_gate": ("<f2", (640,)),
    "sv_up": ("<f2", (640,)),
    "su_down": ("<f2", (640,)),
    "sv_down": ("<f2", (2560,)),
}


def owner_span(rank: int) -> tuple[int, int]:
    if type(rank) is not int or not 0 <= rank < len(OWNERS):
        raise ValueError("invalid E8 owner rank")
    return sum(OWNERS[:rank]), sum(OWNERS[: rank + 1])


def _identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return {"bytes": stat.st_size, "sha256": digest.hexdigest()}


def unpack_quip_e8(packed: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Invert QuIP# E8P12_codebook.maybe_pack_idxs exactly."""
    value = np.asarray(packed)
    if value.dtype != np.int64 or value.shape != (rows, cols // 32):
        raise ValueError("invalid packed QuIP E8 matrix")
    words = value.reshape(rows // 16, cols // 64, 8, 4).transpose(0, 2, 1, 3)
    words = words.reshape(rows // 2, cols // 16)
    unsigned = words.view(np.uint64)
    abs32 = unsigned & np.uint64(0xFFFFFFFF)
    sign32 = unsigned >> np.uint64(32)
    idxs = np.empty((rows // 2, cols // 16, 2, 2), dtype=np.uint16)
    for i in range(4):
        signs = np.zeros(unsigned.shape, dtype=np.uint16)
        for j in range(8):
            signs |= ((sign32 >> np.uint64(4 * j + i)) & 1).astype(np.uint16) << j
        absolute = ((abs32 >> np.uint64(8 * i)) & 255).astype(np.uint16)
        idxs[:, :, i % 2, i // 2] = (absolute << 8) | signs
    return idxs.transpose(0, 2, 1, 3).reshape(rows, cols // 8)


def canonical_expert(record: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(record) == set(FIELDS):
        result = {
            name: np.ascontiguousarray(record[name], dtype=dtype)
            for name, (dtype, _) in FIELDS.items()
        }
        for name, (_, shape) in FIELDS.items():
            if result[name].shape != shape:
                raise ValueError(f"invalid direct E8 field {name}")
        return result
    required = {
        f"{projection}_{field}"
        for projection in ("gate", "up", "down")
        for field in ("Qidxs", "SU", "SV")
    }
    if set(record) != required:
        raise ValueError("incomplete E8 expert record")
    gate = unpack_quip_e8(record["gate_Qidxs"], 640, 2560)
    up = unpack_quip_e8(record["up_Qidxs"], 640, 2560)
    down = unpack_quip_e8(record["down_Qidxs"], 2560, 640)
    su_gate = np.asarray(record["gate_SU"], dtype=np.float16)
    su_up = np.asarray(record["up_SU"], dtype=np.float16)
    if not np.array_equal(su_gate, su_up):
        raise ValueError("E8 gate/up must share SU")
    result = {
        "q_gu": np.ascontiguousarray(np.concatenate((gate, up), axis=0)),
        "q_down": np.ascontiguousarray(down),
        "su_gu": np.ascontiguousarray(su_gate),
        "sv_gate": np.ascontiguousarray(record["gate_SV"], dtype=np.float16),
        "sv_up": np.ascontiguousarray(record["up_SV"], dtype=np.float16),
        "su_down": np.ascontiguousarray(record["down_SU"], dtype=np.float16),
        "sv_down": np.ascontiguousarray(record["down_SV"], dtype=np.float16),
    }
    for name, (dtype, shape) in FIELDS.items():
        if result[name].dtype.str != dtype or result[name].shape != shape:
            raise ValueError(f"invalid canonical E8 field {name}")
    return result


class E8HostArchive:
    """One rank's complete E8 bank copied once into pinned system RAM."""

    def __init__(self, config: E8ArchiveConfig, rank: int):
        root = Path(config.path).resolve(strict=True)
        manifest_path = root / "manifest.json"
        before = _identity(manifest_path)
        meta = json.loads(manifest_path.read_text())
        if (
            meta.get("format") != FORMAT
            or meta.get("status") != "READY"
            or meta.get("layers") != 48
            or meta.get("experts") != 512
            or meta.get("owners") != list(OWNERS)
            or _identity(manifest_path) != before
        ):
            raise ValueError("E8 archive identity or geometry mismatch")
        self.root, self.rank = root, rank
        self.layers, self.owner = 48, OWNERS[rank]
        self.begin, self.end = owner_span(rank)
        self.resident = math.floor(self.owner * config.resident_fraction)
        self.cold = self.owner - self.resident
        self.host: dict[str, torch.Tensor] = {}
        expected_files = {f"rank{rank}:{name}" for name in FIELDS} | {
            "codebook",
            "hadamard20",
        }
        if not expected_files <= set(meta.get("files", {})):
            raise ValueError("incomplete E8 archive inventory")
        for name, (dtype, shape) in FIELDS.items():
            entry = meta["files"][f"rank{rank}:{name}"]
            path = root / entry["file"]
            expected_shape = (self.layers, self.owner, *shape)
            expected_bytes = math.prod(expected_shape) * np.dtype(dtype).itemsize
            if (
                _identity(path) != entry["identity"]
                or path.stat().st_size != expected_bytes
            ):
                raise OSError(f"changed E8 archive field {name}")
            mapped = np.memmap(path, mode="r", dtype=dtype, shape=expected_shape)
            pinned = torch.empty(
                expected_shape,
                dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype,
                device="cpu",
                pin_memory=True,
            )
            np.copyto(pinned.numpy(), mapped, casting="no")
            del mapped
            self.host[name] = pinned
        self.logical_to_physical = self._apply_resident_layout(config)
        self.codebook = self._load_aux(meta, "codebook", (65536, 8))
        self.hadamard20 = self._load_aux(meta, "hadamard20", (20, 20))

    def _apply_resident_layout(self, config: E8ArchiveConfig) -> torch.Tensor | None:
        """Move selected logical experts into the existing resident prefix."""
        if config.resident_layout is None:
            return None
        source = json.loads(Path(config.resident_layout).read_text())
        if (
            source.get("schema") != "flashnext-e8-resident-layout-v1"
            or source.get("layers") != self.layers
            or source.get("owners") != list(OWNERS)
            or source.get("resident_fraction") != config.resident_fraction
        ):
            raise ValueError("E8 resident layout geometry mismatch")
        rows = source.get("selected")
        if not isinstance(rows, list) or len(rows) != self.layers:
            raise ValueError("E8 resident layout has invalid layers")
        inverse = torch.empty(
            (self.layers, self.owner),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        for layer, selected_by_rank in enumerate(rows):
            if not isinstance(selected_by_rank, list) or len(selected_by_rank) != 3:
                raise ValueError("E8 resident layout has invalid ranks")
            selected_global = selected_by_rank[self.rank]
            if (
                not isinstance(selected_global, list)
                or len(selected_global) != self.resident
                or len(set(selected_global)) != self.resident
                or any(not self.begin <= value < self.end for value in selected_global)
            ):
                raise ValueError("E8 resident layout has invalid expert set")
            selected = [value - self.begin for value in selected_global]
            selected_set = set(selected)
            order = selected + [
                expert for expert in range(self.owner) if expert not in selected_set
            ]
            for physical, logical in enumerate(order):
                inverse[layer, logical] = physical
            for value in self.host.values():
                target = value[layer].numpy()
                reordered = np.take(target, order, axis=0)
                np.copyto(target, reordered, casting="no")
        return inverse

    def _load_aux(self, meta, name, shape):
        entry = meta["files"][name]
        path = self.root / entry["file"]
        if (
            _identity(path) != entry["identity"]
            or path.stat().st_size != math.prod(shape) * 2
        ):
            raise OSError(f"changed E8 archive auxiliary {name}")
        value = np.fromfile(path, dtype="<f2").reshape(shape)
        return torch.from_numpy(value.copy()).pin_memory()

    @property
    def pinned_bytes(self) -> int:
        return (
            sum(x.numel() * x.element_size() for x in self.host.values())
            + self.codebook.numel() * 2
            + self.hadamard20.numel() * 2
        )
