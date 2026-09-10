# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable, bounded checkpoint source for native CUTLASS expert rows.

The raw TP transform and backend-finalized schema were admitted against
independent GPU finalization; no model repack or resident full-layer tensor.
"""

import json
import math
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import regex as re

from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry

from .ple_offload import _header, fingerprint

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FIELDS = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def validate(bundle, hidden, width):
    if hidden <= 0 or width <= 0 or hidden % 16 or width % 16:
        raise ValueError("NVFP4 block geometry")
    if set(bundle) != {p + "." + f for p in PROJECTIONS for f in FIELDS}:
        raise ValueError("incomplete expert bundle")
    for p in PROJECTIONS:
        m, k = (hidden, width) if p == "down_proj" else (width, hidden)
        for f, shape in [
            ("weight", (m, k // 2)),
            ("weight_scale", (m, k // 16)),
            ("weight_scale_2", ()),
            ("input_scale", ()),
        ]:
            x = bundle[p + "." + f]
            dtype = np.dtype("float32") if shape == () else np.dtype("uint8")
            if not isinstance(x, np.ndarray) or x.shape != shape or x.dtype != dtype:
                raise ValueError("wrong shape/dtype: " + p + "." + f)
            if shape == () and (not np.isfinite(x) or x <= 0):
                raise ValueError("invalid global scale")
            if f == "weight_scale" and np.any((x & 127) == 127):
                raise ValueError("nonfinite block scale")


def split_bundle(bundle, *, hidden, width, physical, tp, selected_rank=None):
    """Return independent rank bundles; immutable caller source, no requantization."""
    validate(bundle, hidden, width)
    if type(tp) is not int or tp <= 0 or physical < width or physical % tp:
        raise ValueError("invalid physical geometry")
    local = physical // tp
    if local % 16:
        raise ValueError("TP boundaries must align to FP4 scale blocks")
    if selected_rank is not None and (
        type(selected_rank) is not int or not 0 <= selected_rank < tp
    ):
        raise ValueError("invalid selected rank")
    ranks = []
    for rank in range(tp) if selected_rank is None else [selected_rank]:
        start, end = rank * local, min((rank + 1) * local, width)
        valid = max(0, end - start)
        result = {}
        for p in PROJECTIONS:
            for f in FIELDS:
                source = bundle[p + "." + f]
                if f in ("input_scale", "weight_scale_2"):
                    value = source.copy()
                else:
                    unit = 2 if f == "weight" else 16
                    fill = 0 if f == "weight" else 56  # E4M3FN +1, finite padding.
                    if p == "down_proj":
                        value = np.full((hidden, local // unit), fill, dtype=np.uint8)
                        if valid:
                            value[:, : valid // unit] = source[
                                :, start // unit : end // unit
                            ]
                    else:
                        value = np.full((local, hidden // unit), fill, dtype=np.uint8)
                        if valid:
                            value[:valid] = source[start:end]
                result[p + "." + f] = value
        ranks.append(result)
    return ranks


def swizzle(raw):
    m, k = raw.shape
    mp, kp = (m + 127) // 128 * 128, (k + 3) // 4 * 4
    padded = np.zeros((mp, kp), dtype=np.uint8)
    padded[:m, :k] = raw
    return (
        padded.reshape(mp // 128, 4, 32, kp // 4, 4)
        .transpose(0, 3, 2, 1, 4)
        .copy()
        .reshape(mp, kp)
    )


def prepare(bundle):
    """Per-expert CUTLASS schema, including fused alphas and reciprocal scales."""
    if not np.array_equal(
        bundle["gate_proj.weight_scale_2"], bundle["up_proj.weight_scale_2"]
    ):
        raise ValueError("gate/up global scales differ")
    a13 = np.maximum(bundle["gate_proj.input_scale"], bundle["up_proj.input_scale"])
    a2 = bundle["down_proj.input_scale"]
    return {
        "w13_weight": np.concatenate(
            [bundle[p + ".weight"] for p in ["gate_proj", "up_proj"]]
        ),
        "w2_weight": bundle["down_proj.weight"].copy(),
        "w13_weight_scale": swizzle(
            np.concatenate(
                [bundle[p + ".weight_scale"] for p in ["gate_proj", "up_proj"]]
            )
        ),
        "w2_weight_scale": swizzle(bundle["down_proj.weight_scale"]),
        "w13_weight_scale_2": np.asarray(
            bundle["gate_proj.weight_scale_2"] * a13, dtype=np.float32
        ),
        "w2_weight_scale_2": np.asarray(
            bundle["down_proj.weight_scale_2"] * a2, dtype=np.float32
        ),
        "w13_input_scale": np.asarray(a13, dtype=np.float32),
        "w2_input_scale": np.asarray(a2, dtype=np.float32),
        "a1_gscale": np.asarray(np.float32(1) / a13, dtype=np.float32),
        "a2_gscale": np.asarray(np.float32(1) / a2, dtype=np.float32),
    }


class NativeExpertStore:
    """One rank's bounded prepared-row cache; OS source page cache is separate."""

    def __init__(
        self,
        checkpoint,
        rank,
        cache_bytes,
        *,
        layers,
        experts,
        geometry: NVFP4ExpertGeometry,
    ):
        hidden, width, physical, tp = (
            geometry.hidden,
            geometry.width,
            geometry.physical,
            geometry.tp,
        )
        if min(layers, experts, hidden, width, physical, tp) <= 0:
            raise ValueError("invalid expert source geometry")
        if not 0 <= rank < tp or cache_bytes < 0 or physical < width or physical % tp:
            raise ValueError("invalid expert source rank/budget")
        if hidden % 16 or width % 16 or physical // tp % 16:
            raise ValueError("expert source must align to NVFP4 scale blocks")
        self.root = Path(checkpoint).resolve(strict=True)
        self.geometry = geometry
        self.rank, self.tp = rank, tp
        self.hidden, self.width, self.physical = hidden, width, physical
        self.layers, self.experts = layers, experts
        self.limit = cache_bytes
        self.cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] = OrderedDict()
        self.used = self.peak = self.read_bytes = self.hits = self.misses = 0
        self.closed = False
        self.lock = RLock()
        index = self.root / "model.safetensors.index.json"
        self.files = {str(index): fingerprint(index)}
        weight_map = json.loads(index.read_text())["weight_map"]
        pattern = re.compile(
            r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)"
        )
        selected: dict[str, list[tuple[int, int, str, str]]] = {}
        for name, filename in weight_map.items():
            match = pattern.fullmatch(name)
            if match is None:
                continue
            layer, expert, field = int(match[1]), int(match[2]), match[3]
            if not 0 <= layer < layers or not 0 <= expert < experts:
                raise ValueError("expert checkpoint/config identity mismatch")
            if Path(filename).name != filename:
                raise ValueError("expert shard must belong to selected snapshot")
            selected.setdefault(filename, []).append((layer, expert, field, name))
        self.entries: dict[
            tuple[int, int], dict[str, tuple[str, int, int, tuple[int, ...], str]]
        ] = {}
        for filename, fields in selected.items():
            path = self.root / filename
            self.files[str(path)] = fingerprint(path)
            header, base = _header(path)
            for layer, expert, field, name in fields:
                if field not in {p + "." + f for p in PROJECTIONS for f in FIELDS}:
                    raise ValueError(f"unsupported expert tensor {name}")
                projection, leaf = field.split(".")
                m, k = (hidden, width) if projection == "down_proj" else (width, hidden)
                shape = {
                    "weight": [m, k // 2],
                    "weight_scale": [m, k // 16],
                    "weight_scale_2": [],
                    "input_scale": [],
                }[leaf]
                dtype = {
                    "weight": "U8",
                    "weight_scale": "F8_E4M3",
                    "weight_scale_2": "F32",
                    "input_scale": "F32",
                }[leaf]
                meta = header[name]
                start, end = meta["data_offsets"]
                size = math.prod(shape) * (4 if dtype == "F32" else 1)
                if (
                    meta["shape"] != shape
                    or meta["dtype"] != dtype
                    or end - start != size
                    or not 0 <= start <= end <= self.files[str(path)][2] - base
                ):
                    raise ValueError(f"invalid native expert source tensor {name}")
                entries = self.entries.setdefault((layer, expert), {})
                if field in entries:
                    raise ValueError("duplicate source component")
                entries[field] = (str(path), base + start, size, tuple(shape), dtype)
        required = {p + "." + f for p in PROJECTIONS for f in FIELDS}
        if len(self.entries) != layers * experts or any(
            set(e) != required for e in self.entries.values()
        ):
            raise ValueError("incomplete native expert checkpoint inventory")
        self.check_files(self.files)

    def check_files(self, paths):
        if any(fingerprint(p) != self.files[p] for p in paths):
            raise OSError("expert source identity changed; restart required")

    def get(self, layer, expert):
        with self.lock:
            if self.closed:
                raise RuntimeError("native expert source is closed")
            if (layer, expert) not in self.entries:
                raise ValueError("unknown checkpoint expert")
            entries = self.entries[layer, expert]
            files = {e[0] for e in entries.values()} | {
                str(self.root / "model.safetensors.index.json")
            }
            try:
                self.check_files(files)
                key = layer, expert
                if key in self.cache:
                    self.hits += 1
                    self.cache.move_to_end(key)
                    return self.cache[key]
                spans: list[list[Any]] = []
                for name, entry in sorted(
                    entries.items(), key=lambda e: (e[1][0], e[1][1])
                ):
                    path, offset, size, shape, dtype = entry
                    if spans and spans[-1][0] == path and spans[-1][2] == offset:
                        spans[-1][2] += size
                        spans[-1][3].append((name, entry))
                    else:
                        spans.append([path, offset, offset + size, [(name, entry)]])
                raw = {}
                for path, start, end, fields in spans:
                    with open(path, "rb") as f:
                        f.seek(start)
                        blob = f.read(end - start)
                    if len(blob) != end - start:
                        raise OSError("truncated native expert source")
                    self.read_bytes += len(blob)
                    for name, (_, offset, size, shape, dtype) in fields:
                        raw[name] = np.frombuffer(
                            blob,
                            dtype="<f4" if dtype == "F32" else "u1",
                            offset=offset - start,
                            count=math.prod(shape),
                        ).reshape(shape)
                prepared = prepare(
                    split_bundle(
                        raw,
                        hidden=self.hidden,
                        width=self.width,
                        physical=self.physical,
                        tp=self.tp,
                        selected_rank=self.rank,
                    )[0]
                )
                self.check_files(files)
                for array in prepared.values():
                    array.setflags(write=False)
                self.misses += 1
                size = sum(a.nbytes for a in prepared.values())
                if size <= self.limit:
                    while self.used + size > self.limit:
                        _, old = self.cache.popitem(last=False)
                        self.used -= sum(a.nbytes for a in old.values())
                    self.cache[key] = prepared
                    self.used += size
                    self.peak = max(self.peak, self.used)
                return prepared
            except Exception:
                self.close()
                raise

    def close(self):
        with self.lock:
            self.closed = True
            self.cache.clear()
            self.used = 0
