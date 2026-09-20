# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One bounded RAM source for CPU experts, GPU staging and HOT promotion."""

import ctypes as ct
import os
from collections.abc import Mapping
from dataclasses import dataclass
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
        self.partition = getattr(source, "partition", "legacy")
        self.spans = getattr(source, "spans", None)
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
        # The native mmap reserves capacity without touching empty slots.
        # Registering all of it here would materialize the whole RAM budget
        # before compilation. Only DMA consumers need page-locked source.
        return RegisteredSourceRegion(
            self.cpu, chunk_bytes=32 * self.cpu.source_stats()["stride"]
        )

    def resident_bitmap(self):
        return self.cpu.resident_bitmap()

    def get_many(self, layer, experts):
        if not experts:
            return []
        if getattr(self.cpu, "ready_enabled", False):
            return self.cpu.borrow_rows(
                experts, self.fields, self.row_bytes, load_layer=layer
            )
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


@dataclass
class _SourceRegistration:
    address: int
    nbytes: int
    registered: bool = False
    locked: bool = False


class RegisteredSourceRegion:
    """Retain the source allocation; pin demanded chunks through DMA retirement.

    ``chunk_bytes=None`` is the eager whole-region control. A failed demand
    retains earlier pins because their DMA may still be in flight. The caller
    drains all streams before close, including when registration failed.
    """

    def __init__(self, cpu, *, chunk_bytes=None):
        import torch

        self.cpu = cpu
        if not hasattr(cpu, "upload_registrations"):
            cpu.upload_registrations = []
        self.runtime = torch.cuda.cudart()
        self.lock = RLock()
        self.spans = {}
        self.state = "OPEN"
        self.registration_calls = 0
        self.libc = ct.CDLL(None, use_errno=True)
        for name in ("mlock", "munlock"):
            operation = getattr(self.libc, name)
            operation.argtypes = (ct.c_void_p, ct.c_size_t)
            operation.restype = ct.c_int
        self.address, self.nbytes, self.lease = cpu.borrow_region()
        # Keep the owner reachable even if failed cleanup outlives its caller.
        cpu.upload_registrations.append(self)
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            self.chunk_bytes = self.nbytes if chunk_bytes is None else chunk_bytes
            if (
                self.address % page
                or self.nbytes <= 0
                or self.nbytes % page
                or type(self.chunk_bytes) is not int
                or self.chunk_bytes <= 0
                or self.chunk_bytes % page
            ):
                raise ValueError("source registration requires page-aligned chunks")
            if chunk_bytes is None:
                self._register(0)
        except BaseException:
            self.close()
            raise

    @property
    def registered_bytes(self):
        with self.lock:
            return sum(s.nbytes for s in self.spans.values() if s.registered)

    @property
    def registered(self):
        return bool(self.registered_bytes)

    @property
    def locked(self):
        with self.lock:
            return any(s.locked for s in self.spans.values())

    def _register(self, offset):
        span = _SourceRegistration(
            self.address + offset, min(self.chunk_bytes, self.nbytes - offset)
        )
        status = self.runtime.cudaHostRegister(span.address, span.nbytes, 0)
        if status.value:
            raise RuntimeError(f"source registration failed: {status}")
        span.registered = True
        self.spans[offset] = span
        self.registration_calls += 1
        # CUDA pins alone need not mark the VMA unevictable. mlock prevents
        # futile OS reclaim. A failed attempt may already have locked pages.
        span.locked = True
        if self.libc.mlock(span.address, span.nbytes):
            raise OSError(ct.get_errno(), "source mlock failed")

    def ensure(self, addresses, sizes):
        """Validate the entire demand before registering any new source pages."""
        import torch

        with self.lock:
            if self.state != "OPEN":
                raise RuntimeError("source registration owner is not open")
            if len(addresses) != len(sizes):
                raise ValueError("inconsistent source DMA ranges")
            needed: set[int] = set()
            for address, size in zip(addresses, sizes, strict=True):
                if (
                    type(address) is not int
                    or type(size) is not int
                    or size <= 0
                    or address < self.address
                    or address + size > self.address + self.nbytes
                ):
                    raise ValueError("source DMA range outside leased allocation")
                first = (address - self.address) // self.chunk_bytes
                last = (address + size - 1 - self.address) // self.chunk_bytes
                needed.update(i * self.chunk_bytes for i in range(first, last + 1))
            missing = sorted(needed.difference(self.spans))
            if not missing:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("source registration during CUDA capture")
            try:
                for offset in missing:
                    self._register(offset)
            except BaseException:
                self.state = "FAILED"
                raise

    def close(self):
        # Caller owns the upload stream and must drain it before unregister.
        with self.lock:
            if self.state == "CLOSED":
                return
            self.state = "CLOSING"
            for offset in list(reversed(self.spans)):
                span = self.spans[offset]
                if span.registered:
                    status = self.runtime.cudaHostUnregister(span.address)
                    if status.value:
                        raise RuntimeError(f"source unregister failed: {status}")
                    span.registered = False
                if span.locked:
                    if self.libc.munlock(span.address, span.nbytes):
                        raise OSError(ct.get_errno(), "source munlock failed")
                    span.locked = False
                del self.spans[offset]
            self.lease.close()
            self.cpu.upload_registrations.remove(self)
            self.state = "CLOSED"
