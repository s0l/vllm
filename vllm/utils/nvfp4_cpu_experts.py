# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded AVX2 NVFP4 execution borrowing the prepared expert source rows."""

import ctypes as ct
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CpuExpertConfig:
    library: str
    sha256: str
    cores: tuple[int, ...]
    max_m: int = 4

    @classmethod
    def from_options(cls, options, tp, *, max_supported_m=4):
        if options is None:
            return None
        if not isinstance(options, dict) or set(options) != {
            "library",
            "sha256",
            "cores",
            "max_m",
        }:
            raise ValueError("CPU experts require explicit library/SHA/cores/max_m")
        lib, digest, cores, max_m = (
            options[k] for k in ("library", "sha256", "cores", "max_m")
        )
        if (
            type(lib) is not str
            or not Path(lib).is_absolute()
            or type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not isinstance(cores, (list, tuple))
            or any(type(c) is not int or c < 0 for c in cores)
            or len(set(cores)) != len(cores)
            or not tp <= len(cores) <= tp * 32
            or type(max_m) is not int
            or not 1 <= max_m <= max_supported_m
        ):
            raise ValueError("unsupported CPU expert resource/identity configuration")
        return cls(lib, digest, tuple(cores), max_m)

    def rank_cores(self, rank, tp):
        if not 0 <= rank < tp or len(self.cores) < tp:
            raise ValueError("CPU expert TP rank/budget mismatch")
        n, remainder = divmod(len(self.cores), tp)
        start = rank * n + min(rank, remainder)
        return self.cores[start : start + n + (rank < remainder)]


def pointer(array):
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be contiguous")
    return ct.c_void_p(array.ctypes.data)


def library(path, expected_sha):
    if not {"avx2", "fma"} <= set(Path("/proc/cpuinfo").read_text().split()):
        raise ValueError("NVFP4 CPU experts require AVX2 and FMA")
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha:
        raise ValueError("CPU expert native SHA differs from configuration")
    lib: Any = ct.CDLL(str(Path(path).resolve()))
    lib.fn_create_unbound.argtypes = [ct.c_int] * 5
    lib.fn_create_unbound.restype = ct.c_void_p
    lib.fn_destroy.argtypes = [ct.c_void_p]
    lib.fn_destroy.restype = None
    lib.fn_bind_prepared.argtypes = (
        [ct.c_void_p, ct.c_int] + [ct.c_void_p] * 5 + [ct.c_float] * 2
    )
    lib.fn_bind_prepared.restype = ct.c_int
    lib.fn_unbind.argtypes = [ct.c_void_p]
    lib.fn_unbind.restype = ct.c_int
    lib.fn_run.argtypes = [ct.c_void_p] * 4 + [ct.c_int] * 2 + [ct.c_void_p]
    lib.fn_run.restype = ct.c_int
    return lib


def load_cpu_experts(config, geometry, rank, topk):
    """Attach once per worker, before profile and OpenMP expert execution."""
    cores = config.rank_cores(rank, geometry.tp)
    if not set(cores) <= os.sched_getaffinity(0):
        raise ValueError("CPU expert cores are outside the worker affinity")
    lib = library(config.library, config.sha256)
    # New OpenMP expert workers inherit this disjoint per-rank set. The main
    # worker also submits GPU work; API and engine retain their own affinity.
    if os.environ.get("OMP_PROC_BIND", "false").lower() not in ("false", "0"):
        raise ValueError("CPU expert affinity requires OMP_PROC_BIND=false")
    os.sched_setaffinity(0, cores)
    executor = PreparedCpuExperts(
        lib,
        hidden=geometry.hidden,
        local_width=geometry.local,
        capacity=1 << (config.max_m * topk - 1).bit_length(),
        max_m=config.max_m,
        threads=len(cores),
    )
    executor.identity = dict(
        library=config.library,
        sha256=config.sha256,
        isa="avx2,fma",
        cores=cores,
        threads=len(cores),
        total_threads=len(config.cores),
        max_m=config.max_m,
        capacity=executor.e,
        weight_bytes=0,
        scale_workspace_bytes=0,
        scratch_bytes=executor.e
        * executor.max_m
        * (6 * executor.h + 10 * executor.i + 8),
    )
    return executor


class PreparedCpuExperts:
    """Borrow immutable source mappings until synchronous native work drains."""

    def __init__(self, lib, *, hidden, local_width, capacity=64, max_m=4, threads=2):
        self.lib, self.h, self.i = lib, hidden, local_width
        self.e, self.max_m, self.threads = capacity, max_m, threads
        self.ptr = lib.fn_create_unbound(capacity, hidden, local_width, max_m, threads)
        if not self.ptr:
            raise ValueError("CPU scratch geometry admission rejected")
        self.owners: tuple[MappingProxyType, ...] = ()
        self.identity: dict = {}
        self.generation = -1
        self.bound = False
        self.lock = Lock()

    @property
    def weight_bytes(self):
        return 0

    @property
    def scale_workspace_bytes(self):
        return 0

    def bind(self, rows, *, generation):
        if not self.lock.acquire(blocking=False):
            raise ValueError("CPU expert lease busy")
        try:
            if (
                not self.ptr
                or type(generation) is not int
                or generation <= self.generation
            ):
                raise ValueError("stale or closed CPU expert generation")
            if len(rows) > self.e:
                raise ValueError("CPU expert scratch capacity exceeded")
            if self.lib.fn_unbind(self.ptr):
                raise RuntimeError("CPU bindings still in use")
            self.bound, self.owners = False, ()
            self.generation = generation
            for row in rows:
                if hasattr(row, "descriptor"):
                    row.descriptor(row.lease.pool)
            owners = tuple(MappingProxyType(dict(row)) for row in rows)
            for slot, row in enumerate(owners):
                for name, shape in {
                    "w13_weight": (2 * self.i, self.h // 2),
                    "w2_weight": (self.h, self.i // 2),
                    "w13_weight_scale": (2 * self.i, self.h // 16),
                    "w2_weight_scale": (self.h, self.i // 16),
                }.items():
                    a = row[name]
                    if (
                        a.dtype != np.uint8
                        or a.shape != shape
                        or not a.flags.c_contiguous
                    ):
                        raise ValueError("prepared CPU row schema: " + name)
                for name in (
                    "w13_weight_scale_2",
                    "w2_weight_scale_2",
                    "w13_input_scale",
                    "w2_input_scale",
                ):
                    a = row[name]
                    if (
                        a.shape != ()
                        or a.dtype != np.float32
                        or not np.isfinite(a)
                        or a <= 0
                    ):
                        raise ValueError("prepared CPU scalar: " + name)
                code = self.lib.fn_bind_prepared(
                    self.ptr,
                    slot,
                    pointer(row["w13_weight"][: self.i]),
                    pointer(row["w13_weight"][self.i :]),
                    pointer(row["w2_weight"]),
                    pointer(row["w13_weight_scale"]),
                    pointer(row["w2_weight_scale"]),
                    float(
                        np.float32(row["w13_weight_scale_2"] / row["w13_input_scale"])
                    ),
                    float(np.float32(row["w2_weight_scale_2"] / row["w2_input_scale"])),
                )
                if code:
                    raise ValueError(f"direct prepared CPU binding rejected: {code}")
            self.owners, self.bound = owners, True
        except BaseException:
            if self.ptr and self.lib.fn_unbind(self.ptr) == 0:
                self.owners, self.bound = (), False
            raise
        finally:
            self.lock.release()

    def run(self, x, ids, weights, out=None, *, generation):
        if not self.lock.acquire(blocking=False):
            raise ValueError("CPU expert lease busy")
        try:
            if not self.ptr or not self.bound or generation != self.generation:
                raise ValueError("unpublished or stale CPU expert ticket")
            if x.dtype != np.uint16 or x.ndim != 2 or x.shape[1] != self.h:
                raise ValueError("expected BF16 bit storage")
            if ids.dtype != np.int32 or weights.dtype != np.float32:
                raise ValueError("route dtype")
            if ids.ndim != 2 or weights.shape != ids.shape or ids.shape[0] != len(x):
                raise ValueError("route shape")
            if out is None:
                out = np.empty((len(x), self.h), np.float32)
            if out.dtype != np.float32 or out.shape != (len(x), self.h):
                raise ValueError("output shape/dtype")
            code = self.lib.fn_run(
                self.ptr,
                pointer(x),
                pointer(ids),
                pointer(weights),
                len(x),
                ids.shape[1],
                pointer(out),
            )
            if code:
                raise ValueError(f"native CPU expert rejection {code}")
            return out
        finally:
            self.lock.release()

    def release(self, *, generation):
        if not self.lock.acquire(blocking=False):
            raise ValueError("CPU expert lease busy")
        try:
            if generation != self.generation or not self.bound:
                raise ValueError("stale CPU expert release")
            if self.lib.fn_unbind(self.ptr):
                raise RuntimeError("CPU release while native work remains")
            self.bound, self.owners = False, ()
        finally:
            self.lock.release()

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise ValueError("CPU expert lease busy")
        try:
            if self.ptr:
                self.lib.fn_destroy(self.ptr)
                self.ptr = None
            self.owners, self.bound = (), False
        finally:
            self.lock.release()
