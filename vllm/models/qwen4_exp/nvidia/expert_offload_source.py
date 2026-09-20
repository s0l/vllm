# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable, bounded checkpoint source for native CUTLASS expert rows.

The raw TP transform and backend-finalized schema were admitted against
independent GPU finalization; no model repack or resident full-layer tensor.
"""

import heapq
import json
import math
import os
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import regex as re

from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry

from .ple_offload import _header, fingerprint

if TYPE_CHECKING:
    from .expert_offload_pinned import PinnedExpertPool

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


def split_bundle(
    bundle, *, hidden, width, physical, tp, selected_rank=None, spans=None
):
    """Return independent rank bundles; immutable caller source, no requantization."""
    validate(bundle, hidden, width)
    if type(tp) is not int or tp <= 0 or physical < width or physical % tp:
        raise ValueError("invalid physical geometry")
    local = physical // tp
    if spans is not None and (
        len(spans) != tp
        or spans[0][0] != 0
        or spans[-1][1] != width
        or any(a >= b or a % 64 or b % 64 for a, b in spans)
        or any(a[1] != b[0] for a, b in zip(spans, spans[1:]))
    ):
        raise ValueError("invalid contiguous expert ownership")
    if local % 16:
        raise ValueError("TP boundaries must align to FP4 scale blocks")
    if selected_rank is not None and (
        type(selected_rank) is not int or not 0 <= selected_rank < tp
    ):
        raise ValueError("invalid selected rank")
    ranks = []
    for rank in range(tp) if selected_rank is None else [selected_rank]:
        start, end = rank * local, min((rank + 1) * local, width)
        if spans is not None:
            start, end = spans[rank]
            local = end - start
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
        cache_policy="frequency",
        io_workers=4,
        pin_cache=False,
        archive_path=None,
        partition="legacy",
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
        if (
            cache_policy not in ("lru", "frequency")
            or type(io_workers) is not int
            or not 1 <= io_workers <= 16
            or type(pin_cache) is not bool
        ):
            raise ValueError("invalid expert cache policy or I/O concurrency")
        self.root = Path(checkpoint).resolve(strict=True)
        self.geometry = geometry
        if partition not in ("legacy", "balanced"):
            raise ValueError("unknown expert partition")
        self.partition = partition
        self.spans = (
            tuple(geometry.owner_span(r, balanced=True) for r in range(tp))
            if partition == "balanced"
            else None
        )
        self.rank, self.tp = rank, tp
        self.hidden, self.width, self.physical = hidden, width, physical
        self.layers, self.experts = layers, experts
        self.limit = cache_bytes
        self.cache: OrderedDict[tuple[int, int], dict[str, np.ndarray]] = OrderedDict()
        self.cache_policy, self.io_workers = cache_policy, io_workers
        self.pin_cache = pin_cache
        self.pinned_pool: PinnedExpertPool | None = None
        self.frequency = np.zeros((layers, experts), dtype=np.uint16)
        self.residency_scores = None
        self.accesses = 0
        self.cache_heap: list[tuple[int, tuple[int, int]]] = []
        self.cache_bypasses = self.read_calls = 0
        self._reader: ThreadPoolExecutor | None = None
        self._fds: dict[str, BinaryIO] = {}
        self.used = self.peak = self.read_bytes = self.hits = self.misses = 0
        self.closed = False
        self.archive = None
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
        if archive_path is not None:
            from .expert_offload_archive import PreparedExpertArchive

            self.archive = PreparedExpertArchive(archive_path, self)
            self.files.update(self.archive.files)

    def check_files(self, paths):
        if any(fingerprint(p) != self.files[p] for p in paths):
            raise OSError("expert source identity changed; restart required")

    def borrow_resident(self, keys):
        """Retain immutable RAM rows without starting reads or changing policy."""
        with self.lock:
            if self.closed:
                raise RuntimeError("closed expert source")
            return [self.cache[key] for key in keys]

    def _access(self, key):
        if self.cache_policy == "lru" or self.residency_scores is not None:
            return
        self.accesses += 1
        if self.accesses % max(1, self.layers * self.experts * 2) == 0:
            self.frequency >>= 1
            self._rebuild_heap()
        self.frequency[key] = min(int(self.frequency[key]) + 1, 65535)
        if key in self.cache:
            heapq.heappush(self.cache_heap, (int(self.frequency[key]), key))
        if len(self.cache_heap) > max(16, len(self.cache) * 4):
            self._rebuild_heap()

    def _rebuild_heap(self):
        self.cache_heap = [(self._score(k), k) for k in self.cache]
        heapq.heapify(self.cache_heap)

    def _score(self, key):
        if key in getattr(self, "residency_hot", ()):
            return -1.0
        values = (
            self.frequency if self.residency_scores is None else self.residency_scores
        )
        return float(values[key])

    def set_residency_scores(self, scores, gpu_only_keys, *, gpu_hot_keys=()):
        """Admit residual priorities and drop cache references, never leases.

        The caller must have published all GPU copies and drained CPU readers.
        NumPy bases continue to own any returned pinned rows independently.
        Registered slabs remain reusable within the existing physical budget.
        """
        if (
            not isinstance(scores, np.ndarray)
            or scores.shape != self.frequency.shape
            or not np.isfinite(scores).all()
            or np.any(scores < 0)
        ):
            raise ValueError("invalid residual expert priorities")
        keys, hot_keys = tuple(gpu_only_keys), tuple(gpu_hot_keys)
        if any(
            len(key) != 2
            or any(type(v) is not int for v in key)
            or not 0 <= key[0] < self.layers
            or not 0 <= key[1] < self.experts
            for key in keys + hot_keys
        ):
            raise ValueError("invalid GPU-only expert key")
        with self.lock:
            if self.closed:
                raise RuntimeError("closed expert source")
            self.residency_scores = scores.astype(np.float64, copy=True)
            self.residency_hot = set(hot_keys)
            released = 0
            for key in keys:
                value = self.cache.pop(key, None)
                if value is not None:
                    released += sum(a.nbytes for a in value.values())
            self.used -= released
            self._rebuild_heap()
            return released

    def _retain(self, key, prepared):
        size = sum(a.nbytes for a in prepared.values())
        limit = self.limit
        if self.pin_cache and size <= limit:
            if self.pinned_pool is None:
                import torch

                from .expert_offload_pinned import PinnedExpertPool

                self.pinned_pool = PinnedExpertPool(
                    prepared, limit, device=torch.accelerator.current_device_index()
                )
            limit = self.pinned_pool.capacity * size
        if size > limit:
            return prepared
        while self.used + size > limit:
            if self.cache_policy == "frequency":
                while self.cache_heap and (
                    self.cache_heap[0][1] not in self.cache
                    or self.cache_heap[0][0] != self._score(self.cache_heap[0][1])
                ):
                    heapq.heappop(self.cache_heap)
                score, victim = self.cache_heap[0]
                # Equal-frequency scans must not replace the entire working set.
                if self._score(key) <= score:
                    self.cache_bypasses += 1
                    return prepared
                heapq.heappop(self.cache_heap)
                old = self.cache.pop(victim)
            else:
                _, old = self.cache.popitem(last=False)
            self.used -= sum(a.nbytes for a in old.values())
            del old  # Release an unobserved pinned victim before requesting a slot.
        if self.pinned_pool is not None:
            pinned = self.pinned_pool.retain(prepared)
            if pinned is None:
                self.cache_bypasses += 1
                return prepared
            prepared = pinned
        self.cache[key] = prepared
        self.used += size
        self.peak = max(self.peak, self.used)
        if self.cache_policy == "frequency":
            heapq.heappush(self.cache_heap, (self._score(key), key))
        return prepared

    def _read_raw(self, key):
        if self.archive is not None:
            return self.archive.read(key, self._fds[self.archive.paths[key[0]]])
        entries = self.entries[key]
        spans: list[list[Any]] = []
        for name, entry in sorted(entries.items(), key=lambda e: (e[1][0], e[1][1])):
            path, offset, size, shape, dtype = entry
            if spans and spans[-1][0] == path and spans[-1][2] == offset:
                spans[-1][2] += size
                spans[-1][3].append((name, entry))
            else:
                spans.append([path, offset, offset + size, [(name, entry)]])
        raw, read_bytes = {}, 0
        for path, start, end, fields in spans:
            blob = os.pread(self._fds[path].fileno(), end - start, start)
            if len(blob) != end - start:
                raise OSError("truncated native expert source")
            read_bytes += len(blob)
            for name, (_, offset, size, shape, dtype) in fields:
                raw[name] = np.frombuffer(
                    blob,
                    dtype="<f4" if dtype == "F32" else "u1",
                    offset=offset - start,
                    count=math.prod(shape),
                ).reshape(shape)
        return raw, read_bytes, len(spans)

    def _prepare_raw(self, raw):
        if self.archive is not None:
            return raw
        prepared = prepare(
            split_bundle(
                raw,
                hidden=self.hidden,
                width=self.width,
                physical=self.physical,
                tp=self.tp,
                selected_rank=self.rank,
                spans=self.spans,
            )[0]
        )
        for array in prepared.values():
            array.setflags(write=False)
        return prepared

    def _read_expert(self, key):
        raw, read_bytes, read_calls = self._read_raw(key)
        return self._prepare_raw(raw), read_bytes, read_calls

    def _read_many(self, missing, destinations=None):
        if self._reader is None:
            self._reader = ThreadPoolExecutor(
                max_workers=self.io_workers, thread_name_prefix="expert-io"
            )
        keys = iter(missing)
        pending: deque[Future[Any]] = deque()

        def read(key):
            if destinations is None:
                return self._read_raw(key)
            assert self.archive is not None
            return self.archive.read(
                key, self._fds[self.archive.paths[key[0]]], destinations[key]
            )

        for _ in range(min(len(missing), self.io_workers * 2)):
            pending.append(self._reader.submit(read, next(keys)))
        rows = []
        while pending:
            raw, read_bytes, read_calls = pending.popleft().result()
            key = next(keys, None)
            if key is not None:
                pending.append(self._reader.submit(read, key))
            rows.append((self._prepare_raw(raw), read_bytes, read_calls))
        return rows

    def get(self, layer, expert):
        return self.get_many(layer, (expert,))[0]

    def get_many(self, layer, experts):
        """Fetch one bounded demand batch; readers never mutate cache ownership."""
        return self._get_many(layer, experts)

    def get_many_into(self, layer, experts, arrays, positions):
        """Miss views may borrow caller staging until its next fenced reuse."""
        if self.archive is None or (not self.pin_cache and self.limit):
            raise ValueError("borrowed rows require a pinned or disabled RAM cache")
        if len(positions) != len(experts) or len(set(positions)) != len(positions):
            raise ValueError("invalid prepared destination positions")
        destinations: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        spans = []
        for expert, position in zip(experts, positions):
            if type(position) is not int or position < 0:
                raise ValueError("invalid prepared destination position")
            values = {
                name: array[position : position + 1].reshape(
                    self.archive.fields[name]["shape"]
                )
                for name, array in arrays.items()
            }
            spans.extend(self.archive.validate_buffers(values))
            destinations.setdefault((layer, expert), values)
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("overlapping prepared read destinations")
        return self._get_many(layer, experts, destinations)

    def _get_many(self, layer, experts, destinations=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("native expert source is closed")
            keys = [(layer, expert) for expert in experts]
            if len(keys) > 1024:
                raise ValueError("expert source batch exceeds bounded queue")
            if any(key not in self.entries for key in keys):
                raise ValueError("unknown checkpoint expert")
            files = {e[0] for key in keys for e in self.entries[key].values()} | {
                str(self.root / "model.safetensors.index.json")
            }
            if self.archive is not None:
                files |= {str(self.archive.root / "manifest.json")}
                files |= {self.archive.paths[key[0]] for key in keys}
            try:
                self.check_files(files)
                result, missing = {}, []
                for key in dict.fromkeys(keys):
                    self._access(key)
                    if key in self.cache:
                        self.hits += 1
                        self.cache.move_to_end(key)
                        result[key] = self.cache[key]
                    else:
                        missing.append(key)
                for key in missing:
                    paths = (
                        (self.archive.paths[key[0]],)
                        if self.archive is not None
                        else {entry[0] for entry in self.entries[key].values()}
                    )
                    for path in paths:
                        if path not in self._fds:
                            # Store owns these handles until close; FileIO also
                            # closes them if an unused store is garbage-collected.
                            self._fds[path] = open(path, "rb", buffering=0)  # noqa: SIM115
                if (
                    len(missing) > 1 and self.io_workers > 1
                ) or destinations is not None:
                    # Overlap blocking reads, but prepare on the owning thread.
                    # Concurrent TP ranks already parallelize this CPU transform;
                    # running it in every I/O worker oversubscribes the CPU budget.
                    rows = self._read_many(missing, destinations)
                else:
                    rows = [self._read_expert(key) for key in missing]
                self.check_files(files)
                for key, (prepared, read_bytes, read_calls) in zip(missing, rows):
                    self.misses += 1
                    self.read_bytes += read_bytes
                    self.read_calls += read_calls
                    result[key] = self._retain(key, prepared)
                return [result[key] for key in keys]
            except Exception:
                self.close()
                raise

    def close(self):
        with self.lock:
            self.closed = True
            if self._reader is not None:
                self._reader.shutdown(wait=True, cancel_futures=True)
                self._reader = None
            for handle in self._fds.values():
                handle.close()
            self._fds.clear()
            self.cache.clear()
            self.cache_heap.clear()
            self.used = 0
            if self.pinned_pool is not None:
                self.pinned_pool.close()
