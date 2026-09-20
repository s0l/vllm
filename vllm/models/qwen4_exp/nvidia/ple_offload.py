# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded FP8 PLE rows prepared before the captured model forward.

The mmap gather and runner staging design adapt vllm-project/vllm#54129
(pinned source d3d0337ad502007c40042cc0ec4d3a16cc13e6c1c096d82893a119f30019ac03).
Unlike a resident embedding, neither construction nor loading creates a table
Parameter. Source identity, cache capacity and DMA lifetime remain explicit.
"""

import json
import math
import os
import struct
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import regex as re
import torch
from torch import nn


def fingerprint(path):
    s = os.stat(path)
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _header(path):
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError("truncated PLE safetensors header length")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 100 << 20:
            raise ValueError("PLE safetensors header exceeds 100MiB")
        raw = f.read(length)
        if len(raw) != length:
            raise ValueError("truncated PLE safetensors header")
    return json.loads(raw), 8 + length


class PleRowStore:
    """Immutable checkpoint rows with a bounded LRU; page cache is external."""

    def __init__(
        self, model_path, layer_idx, rows, row_bytes, parts, cache_bytes, max_ids
    ):
        self.lock = threading.RLock()
        self.closed = False
        self.maps = {}
        self.pool = None
        self.cache = OrderedDict()
        self.read_rows = self.calls = self.peak_cache_bytes = 0
        if min(rows, row_bytes, parts, max_ids) <= 0 or cache_bytes < 0:
            raise ValueError("invalid PLE store geometry/budget")
        self.rows, self.row_bytes = rows, row_bytes
        self.max_ids = max_ids
        self.capacity = cache_bytes // row_bytes
        self.shard_size = (rows + parts - 1) // parts
        root = Path(model_path).resolve(strict=True)
        index = root / "model.safetensors.index.json"
        self.files = {str(index): fingerprint(index)}
        weight_map = json.loads(index.read_text())["weight_map"]
        marker = f".layers.{layer_idx}.ple.ple_embedding.ngram_embedding."
        selected = {k: v for k, v in weight_map.items() if marker in k}
        if not selected or len({k.split(marker)[0] for k in selected}) != 1:
            raise ValueError("missing or ambiguous PLE checkpoint table")
        self.entries = {}
        self.shards = {}
        headers = {}
        try:
            for name, filename in selected.items():
                path = root / filename
                if path.parent.resolve() != root:
                    raise ValueError("PLE shard must belong to selected snapshot")
                if str(path) not in self.files:
                    self.files[str(path)] = fingerprint(path)
                    headers[str(path)] = _header(path)
                header, base = headers[str(path)]
                meta = header[name]
                start, end = meta["data_offsets"]
                if not 0 <= start <= end <= self.files[str(path)][2] - base:
                    raise ValueError("PLE tensor range outside source")
                leaf = name.split(marker)[1]
                self.entries[leaf] = (str(path), base + start, end - start, meta)
                match = re.fullmatch(r"shard_(\d+)\.weight", leaf)
                if match:
                    i = int(match[1])
                    expected_rows = min(self.shard_size, rows - i * self.shard_size)
                    if (
                        not 0 <= i < parts
                        or expected_rows <= 0
                        or meta["dtype"] != "F8_E4M3"
                        or meta["shape"] != [expected_rows, row_bytes]
                        or end - start != expected_rows * row_bytes
                    ):
                        raise ValueError(f"invalid NVIDIA FP8 PLE shard {i}")
                    self.shards[i] = (str(path), base + start, expected_rows)
                elif leaf != "weight_scale":
                    raise ValueError(f"unsupported PLE source tensor {leaf}")
            expected_parts = math.ceil(rows / self.shard_size)
            if sorted(self.shards) != list(range(expected_parts)):
                raise ValueError("incomplete PLE source shard coverage")
            path, offset, size, meta = self.entries["weight_scale"]
            dtype = {
                "BF16": torch.bfloat16,
                "F16": torch.float16,
                "F32": torch.float32,
            }.get(meta["dtype"])
            if dtype is None or math.prod(meta["shape"]) != 1:
                raise ValueError("PLE requires one global scale")
            if size != torch.empty((), dtype=dtype).element_size():
                raise ValueError("invalid PLE scale byte width")
            with open(path, "rb") as f:
                f.seek(offset)
                raw = bytearray(f.read(size))
            if len(raw) != size:
                raise ValueError("truncated PLE scale")
            self.scale = torch.frombuffer(raw, dtype=dtype).float().item()
            if not math.isfinite(self.scale) or self.scale <= 0:
                raise ValueError("invalid PLE global scale")
            self.check_files()
            for i, (path, offset, count) in self.shards.items():
                self.maps[i] = np.memmap(
                    path,
                    dtype=np.uint8,
                    mode="r",
                    offset=offset,
                    shape=(count, row_bytes),
                )
            self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ple-rows")
        except Exception:
            self.close()
            raise

    def check_files(self):
        if any(fingerprint(p) != stamp for p, stamp in self.files.items()):
            raise OSError("PLE backing identity changed; restart required")

    def gather_into(self, ids, destination):
        with self.lock:
            if self.closed:
                raise RuntimeError("PLE store is closed")
            if (
                ids.size > self.max_ids
                or ids.dtype != np.int64
                or ids.ndim != 1
                or destination.dtype != np.uint8
                or destination.shape != (len(ids), self.row_bytes)
                or not destination.flags.c_contiguous
                or not destination.flags.writeable
            ):
                raise ValueError("invalid PLE row destination")
            if len(ids) and (ids.min() < 0 or ids.max() >= self.rows):
                raise IndexError("PLE row outside checkpoint table")
            try:
                self.check_files()
                unique, inverse = np.unique(ids, return_inverse=True)
                staged = np.empty((len(unique), self.row_bytes), dtype=np.uint8)
                missing = []
                for j, key in enumerate(unique):
                    key = int(key)
                    if key in self.cache:
                        staged[j] = np.frombuffer(self.cache[key], dtype=np.uint8)
                        self.cache.move_to_end(key)
                    else:
                        missing.append(j)
                # Each batch bounds queue depth and transient fancy-index copies.
                for begin in range(0, len(missing), 512):
                    positions = np.asarray(missing[begin : begin + 512])
                    shard_ids = unique[positions] // self.shard_size
                    tasks = []
                    for shard in np.unique(shard_ids):
                        selected = positions[shard_ids == shard]
                        tasks.append((int(shard), selected))

                    def read(task):
                        shard, selected = task
                        staged[selected] = self.maps[shard][
                            unique[selected] % self.shard_size
                        ]

                    assert self.pool is not None
                    for _ in self.pool.map(read, tasks):
                        pass
                    self.read_rows += len(positions)
                    for pos in positions:
                        row = staged[pos]
                        if np.any((row & 127) == 127):
                            raise ValueError("non-finite FP8 PLE row")
                        if self.capacity:
                            while len(self.cache) >= self.capacity:
                                self.cache.popitem(last=False)
                            self.cache[int(unique[pos])] = row.tobytes()
                    self.peak_cache_bytes = max(
                        self.peak_cache_bytes, len(self.cache) * self.row_bytes
                    )
                self.check_files()
                destination[:] = staged[inverse]
                self.calls += 1
            except Exception:
                self.close()
                raise

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                if self.pool is not None:
                    self.pool.shutdown(wait=True, cancel_futures=True)
                self.maps.clear()
                self.cache.clear()


class MmapPLEEmbedding(nn.Module):
    """A fixed staging buffer, never an embedding-table Parameter."""

    weight_scale: torch.Tensor
    raw: torch.Tensor | None
    pinned: torch.Tensor | None
    event: torch.cuda.Event | None

    def __init__(self, rows, dim, prefix, model_path, parts, cache_bytes, max_ids):
        super().__init__()
        layer = re.search(r"\.layers\.(\d+)\.", "." + prefix)
        if layer is None:
            raise ValueError("PLE layer identity missing from prefix")
        self.org_vocab_size = rows
        self.embedding_dim = dim
        self.store = PleRowStore(
            model_path, int(layer[1]), rows, dim, parts, cache_bytes, max_ids
        )
        self.register_buffer(
            "weight_scale", torch.full((), float("nan")), persistent=False
        )
        self.register_buffer("raw", None, persistent=False)
        self.pinned = None
        self.event = None
        self.loaded = False
        self.seen = set()
        self.lock = threading.RLock()
        self.capacity = 0
        self.prepared = False

    def accept_weight(self, name, weight):
        if self.loaded or name in self.seen:
            raise RuntimeError("duplicate PLE weight load; restart required")
        _, _, _, meta = self.store.entries[name]
        dtype = {
            "F8_E4M3": torch.float8_e4m3fn,
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
        }[meta["dtype"]]
        if list(weight.shape) != meta["shape"] or weight.dtype != dtype:
            raise ValueError(f"PLE loader/source metadata mismatch for {name}")
        if name == "weight_scale":
            if weight.float().item() != self.store.scale:
                raise ValueError("PLE loader/source global scale mismatch")
            self.weight_scale.copy_(weight.reshape(()))
        self.seen.add(name)

    def finish_load(self):
        if self.seen != self.store.entries.keys():
            raise ValueError("PLE loader did not cover every source shard/scale")
        self.store.check_files()
        self.loaded = True

    def initialize_staging(self, capacity, heads, device):
        if (
            self.raw is not None
            or not self.loaded
            or capacity <= 0
            or capacity * heads > self.store.max_ids
        ):
            raise RuntimeError("invalid PLE staging initialization")
        self.capacity = capacity
        self.weight_scale = self.weight_scale.to(device)
        raw = torch.empty(
            capacity, heads, self.embedding_dim, dtype=torch.uint8, device=device
        )
        self.raw = raw
        self.pinned = torch.empty(
            raw.shape, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self.event = torch.cuda.Event()

    def prepare(self, ids, padded_tokens):
        with self.lock:
            if (
                self.raw is None
                or self.pinned is None
                or self.event is None
                or not self.loaded
            ):
                raise RuntimeError("PLE rows not initialized")
            actual = ids.shape[0]
            if not 0 <= actual <= padded_tokens <= self.capacity:
                raise ValueError("PLE staging capacity exceeded")
            if tuple(ids.shape[1:]) != (self.raw.shape[1],):
                raise ValueError("PLE hash head count mismatch")
            self.prepared = False
            try:
                cpu_ids = ids.to(device="cpu", dtype=torch.int64).numpy().reshape(-1)
                self.store.gather_into(
                    cpu_ids,
                    self.pinned[:actual].numpy().reshape(-1, self.embedding_dim),
                )
                self.pinned[actual:padded_tokens].zero_()
                # Current-stream order protects the previous graph reader. Waiting
                # for this DMA also protects the pinned source on its next use.
                self.raw[:padded_tokens].copy_(
                    self.pinned[:padded_tokens], non_blocking=True
                )
                self.event.record()
                self.event.synchronize()
                self.prepared = True
            except Exception:
                torch.cuda.current_stream(self.raw.device).synchronize()
                self.store.close()
                raise

    def prepare_dummy(self, padded_tokens):
        if self.raw is None or not 0 <= padded_tokens <= self.capacity:
            raise ValueError("invalid PLE dummy staging capacity")
        self.raw[:padded_tokens].zero_()
        self.prepared = True

    def forward(self, *_):
        raise RuntimeError("PLE mmap rows must be prepared by ModelRunnerV2")


def reject_mmap_reload(model: nn.Module) -> None:
    """Reject before any parent loader mutates dense or vision weights."""
    if any(isinstance(m, MmapPLEEmbedding) and m.loaded for m in model.modules()):
        raise RuntimeError("Reloading a PLE mmap model requires a new worker")
