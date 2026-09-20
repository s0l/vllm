# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded registered expert rows, with leases retained by NumPy base views.

Registration belongs to the calling thread, outside graph capture. Releasing
a view only returns a slot; it never invokes CUDA from a Python finalizer.
Closed pools with outstanding views are collected explicitly after those
readers drain. Thus eviction and source close cannot recycle a DMA source.
"""

from __future__ import annotations

import mmap
from threading import RLock

import numpy as np
import torch

from vllm.v1.simple_kv_offload import cuda_mem_ops

_retired: set[PinnedExpertPool] = set()
_retired_lock = RLock()


def _register(buffer, device):
    with torch.accelerator.device_index(device):
        cuda_mem_ops.pin_tensor(buffer)


def _unregister(buffer, device):
    from cuda.bindings import driver

    with torch.accelerator.device_index(device):
        (error,) = driver.cuMemHostUnregister(buffer.data_ptr())
        if error != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"expert RAM unregister failed: {error}")


def collect_retired():
    """Call only outside capture, after any possible DMA readers have drained."""
    with _retired_lock:
        pools = tuple(_retired)
    for pool in pools:
        pool.collect()


class _Slab:
    def __init__(self, rows, nbytes, schema, device):
        # mmap makes both registration boundaries page aligned. Independent
        # slabs never accidentally register overlapping allocator pages.
        self.mapping = mmap.mmap(-1, nbytes)
        self.buffer = torch.frombuffer(self.mapping, dtype=torch.uint8)
        self.views = {}
        offset = 0
        for name, (_, _, stride) in schema.items():
            size = rows * stride
            self.views[name] = self.buffer[offset : offset + size].view(rows, stride)
            offset += size
        self.rows, self.nbytes, self.active = rows, nbytes, 0
        self.free = list(reversed(range(rows)))
        _register(self.buffer, device)


class _RowLease:
    def __init__(self, pool, slab, slot):
        self.pool, self.slab, self.slot = pool, slab, slot

    def __del__(self):
        self.pool.release(self.slab, self.slot)


class _ColumnView:
    def __init__(self, lease, name, shape, dtype):
        self.lease = lease
        tensor = lease.slab.views[name]
        self.__array_interface__ = {
            "shape": shape,
            "typestr": dtype.str,
            "data": (tensor.data_ptr() + lease.slot * tensor.stride(0), True),
            "version": 3,
        }


class PinnedExpertBundle(dict):
    """A normal immutable-array bundle with a retained physical row lease."""

    def __init__(self, lease):
        self.lease = lease
        super().__init__(
            (name, np.asarray(_ColumnView(lease, name, shape, dtype)))
            for name, (shape, dtype, _) in lease.pool.schema.items()
        )

    def descriptor(self, pool):
        lease = self.lease
        if lease.pool is not pool or pool.closed or set(self) != set(pool.schema):
            raise RuntimeError("foreign or retired pinned expert row")
        for name, (shape, dtype, stride) in pool.schema.items():
            value = self[name]
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype != dtype
                or value.flags.writeable
                or value.ctypes.data
                != lease.slab.views[name].data_ptr() + lease.slot * stride
            ):
                raise RuntimeError("pinned expert view no longer owns its source")
        return lease.slab, lease.slot


class PinnedExpertPool:
    def __init__(self, sample, limit, *, device, slab_rows=128):
        if type(limit) is not int or limit < 0 or slab_rows < 1:
            raise ValueError("invalid pinned expert pool budget")
        self.schema = {
            name: (value.shape, value.dtype, value.nbytes)
            for name, value in sample.items()
        }
        self.row_bytes = sum(item[2] for item in self.schema.values())
        if not self.row_bytes or any(item[2] <= 0 for item in self.schema.values()):
            raise ValueError("empty pinned expert schema")
        self.limit, self.device = limit, device
        self.planned = []
        remaining = limit
        page = mmap.PAGESIZE
        while remaining >= self.row_bytes:
            rows = min(slab_rows, remaining // self.row_bytes)
            size = (rows * self.row_bytes + page - 1) // page * page
            if size > remaining:
                rows -= 1
                size = (rows * self.row_bytes + page - 1) // page * page
            if not rows:
                break
            self.planned.append((rows, size))
            remaining -= size
        self.capacity = sum(rows for rows, _ in self.planned)
        self.slabs: list[_Slab] = []
        self.allocated = self.peak_allocated = self.bypasses = 0
        self.closed = False
        self.lock = RLock()

    def retain(self, prepared):
        with self.lock:
            if self.closed:
                raise RuntimeError("pinned expert pool is closed")
            if set(prepared) != set(self.schema) or any(
                value.shape != self.schema[name][0]
                or value.dtype != self.schema[name][1]
                for name, value in prepared.items()
            ):
                raise ValueError("incompatible pinned expert schema")
            slab = next((s for s in self.slabs if s.free), None)
            if slab is None and len(self.slabs) < len(self.planned):
                rows, size = self.planned[len(self.slabs)]
                slab = _Slab(rows, size, self.schema, self.device)
                self.slabs.append(slab)
                self.allocated += size
                self.peak_allocated = max(self.peak_allocated, self.allocated)
            if slab is None:
                # Evicted rows can still belong to returned views / DMA. Do
                # not block on the same caller or retain a second heap cache.
                self.bypasses += 1
                return None
            slot = slab.free.pop()
            slab.active += 1
            lease = _RowLease(self, slab, slot)
            for name, value in prepared.items():
                np.copyto(
                    slab.views[name][slot].numpy(),
                    value.reshape(-1).view(np.uint8),
                )
            return PinnedExpertBundle(lease)

    def release(self, slab, slot):
        with self.lock:
            slab.free.append(slot)
            slab.active -= 1

    def close(self):
        with self.lock:
            self.closed = True
            with _retired_lock:
                _retired.add(self)
        self.collect()

    def collect(self):
        with self.lock:
            if not self.closed:
                return
            for slab in tuple(self.slabs):
                if slab.active == 0:
                    # On failure retain the registered allocation and surface
                    # the error; freeing still-registered storage is unsafe.
                    _unregister(slab.buffer, self.device)
                    self.slabs.remove(slab)
                    self.allocated -= slab.nbytes
            if not self.slabs:
                with _retired_lock:
                    _retired.discard(self)
