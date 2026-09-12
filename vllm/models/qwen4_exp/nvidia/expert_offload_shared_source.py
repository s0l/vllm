# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One bounded RAM source for CPU experts, GPU staging and HOT promotion."""

from collections.abc import Mapping
from threading import RLock

import numpy as np


class ResidentRows(Mapping):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return (
            divmod(key, self.source.experts) for key in self.source.cpu.resident_keys()
        )

    def __len__(self):
        return len(self.source.cpu.resident_keys())

    def __contains__(self, key):
        # Mapping's default probes __getitem__, which leases native weights
        # and rejects a nonresident key. Membership is metadata-only.
        return any(candidate == key for candidate in self)

    def __getitem__(self, key):
        return self.source.borrow_resident([key])[0]


class SharedNativeSource:
    """No second cache: source rows are owned and replaced inside native code."""

    layers: int
    experts: int
    rank: int

    def __init__(self, source, cpu, fields, row_bytes):
        self.root = source.root
        self.geometry = source.geometry
        self.rank = source.rank
        self.tp = source.tp
        self.hidden = source.hidden
        self.width = source.width
        self.physical = source.physical
        self.layers = source.layers
        self.experts = source.experts
        self.archive = source.archive
        self.files = source.files
        self.entries = source.entries
        self.check_files = source.check_files
        self.cpu, self.fields, self.row_bytes = cpu, fields, row_bytes
        self.cache, self.lock = ResidentRows(self), RLock()
        # The legacy bank staging path understands PinnedExpertBundle slabs.
        # This source uses a native region lease and registers it separately.
        self.pin_cache = False
        self.pinned_pool = None
        stats = cpu.source_stats()
        self.limit = stats["capacity"] * stats["stride"]

    def borrow_resident(self, keys):
        return self.cpu.borrow_rows(
            [layer * self.experts + expert for layer, expert in keys],
            self.fields,
            self.row_bytes,
        )

    def register_uploads(self):
        return RegisteredSourceRegion(self.cpu)

    def resident_bitmap(self):
        return self.cpu.resident_bitmap()

    def get_many(self, layer, experts):
        if not experts:
            return []
        with self.lock:
            self.cpu.load_rows(layer, experts)
            return self.borrow_resident([(layer, expert) for expert in experts])

    def get(self, layer, expert):
        return self.get_many(layer, [expert])[0]

    def get_many_into(self, layer, experts, buffers, positions):
        if len(experts) != len(positions) or len(set(positions)) != len(positions):
            raise ValueError("inconsistent native staging positions")
        rows = self.get_many(layer, experts)
        result = []
        for row, position in zip(rows, positions, strict=True):
            views = {}
            for name, value in row.items():
                target = buffers[name][position : position + 1].reshape(value.shape)
                target[...] = value
                views[name] = target
            result.append(views)
        return result

    @property
    def closed(self):
        return self.cpu.ptr is None

    @property
    def used(self):
        return self.cpu.source_stats()["rows"] * self.row_bytes

    @property
    def hits(self):
        return self.cpu.source_stats()["hits"]

    @property
    def misses(self):
        return self.cpu.source_stats()["misses"]

    @property
    def read_bytes(self):
        return self.cpu.source_stats()["read_bytes"]

    def set_residency_scores(self, scores, gpu_only_keys, *, gpu_hot_keys=()):
        if (
            scores.shape != (self.layers, self.experts)
            or not np.isfinite(scores).all()
            or np.any(scores < 0)
        ):
            raise ValueError("invalid native residual priorities")
        # Consumed residual priorities survive scans; READY GPU duplicates
        # remain the first reclaimable tier unless a row lease protects them.
        hot = tuple(gpu_hot_keys) or tuple(gpu_only_keys)
        if not set(gpu_only_keys) <= set(hot):
            raise ValueError("GPU-only source keys are not published HOT")
        self.cpu.source_scores(scores)
        released = self.cpu.source_hot(gpu_only_keys, drop=True) if gpu_only_keys else 0
        self.cpu.source_hot(hot)
        return released


class RegisteredSourceRegion:
    """Pin existing source pages; retain their allocation until DMA drains."""

    def __init__(self, cpu):
        import torch

        self.cpu = cpu
        if not hasattr(cpu, "upload_registrations"):
            cpu.upload_registrations = []
        self.address, self.nbytes, self.lease = cpu.borrow_region()
        self.runtime = torch.cuda.cudart()
        self.registered = False
        try:
            status = self.runtime.cudaHostRegister(self.address, self.nbytes, 0)
            if status.value:
                raise RuntimeError(f"source registration failed: {status}")
            self.registered = True
            # Failed unregister must retain the native allocation lease even
            # if the caller discards its promotion owner after the exception.
            cpu.upload_registrations.append(self)
        except BaseException:
            self.lease.close()
            raise

    def close(self):
        # Caller owns the upload stream and must drain it before unregister.
        if self.registered:
            status = self.runtime.cudaHostUnregister(self.address)
            if status.value:
                raise RuntimeError(f"source unregister failed: {status}")
            self.registered = False
        self.lease.close()
        if self in self.cpu.upload_registrations:
            self.cpu.upload_registrations.remove(self)
