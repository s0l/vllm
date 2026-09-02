# SPDX-License-Identifier: Apache-2.0
"""Host-memory shared buffers for CustomAllreduce on P2P-less topologies.

vLLM's CustomAllreduce protocol (start/end barriers, per-block per-rank-pair
counters, epoch kept in the shared Signal struct) is memory-agnostic; only
its allocation path is not: `allocate_shared_buffer_and_handle` uses
cudaMalloc + cudaIpcGetMemHandle, which requires GPU P2P. Consumer boards
without P2P therefore fall back to PYNCCL for every collective, paying
~46 us on 30 KB decode rows where a host-staged one-shot exchange measures
~24 us (see evals/ll_pinned_allreduce_s1s2.py).

This module allocates ONE /dev/shm region per node, registers it with
cudaHostRegister on every rank and hands out per-rank offsets of the local
device pointer. All ranks address the same physical memory, so the peer
pointers the kernel expects are just offsets - no kernel change, and the
proven synchronization protocol is reused as-is.

Zero-copy host access is ~3 GB/s effective over PCIe, so this is strictly a
small-message lever: keep max_size small and let NCCL serve large rows.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

SHM_DIR = "/dev/shm"
_registered: dict[str, tuple[mmap.mmap, int]] = {}


def enabled() -> bool:
    return os.environ.get("AG2_VLLM_CUSTOM_AR_HOST", "0") == "1"


def _open_region(name: str, total_bytes: int) -> mmap.mmap:
    path = os.path.join(SHM_DIR, name)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, total_bytes)
            os.pwrite(fd, b"\x00" * min(total_bytes, 1 << 20), 0)
        finally:
            os.close(fd)
    except FileExistsError:
        pass
    deadline = time.time() + 30
    while time.time() < deadline:
        if os.path.exists(path) and os.path.getsize(path) == total_bytes:
            break
        time.sleep(0.02)
    else:
        raise RuntimeError(f"shared region {path} was never sized by any rank")
    fd = os.open(path, os.O_RDWR)
    try:
        return mmap.mmap(fd, total_bytes)
    finally:
        os.close(fd)


def create_host_shared_buffer(
    size_in_bytes: int, world_size: int, rank: int, tag: str
) -> list[int]:
    """Return per-rank device pointers into one shared host region.

    The returned list has the same contract as
    `CustomAllreduce.create_shared_buffer`: entry i is a device-addressable
    pointer to rank i's slice, valid on the calling rank.
    """
    # 2 MiB alignment keeps each rank's slice on its own huge page boundary.
    stride = (size_in_bytes + (2 << 20) - 1) & ~((2 << 20) - 1)
    total = stride * world_size
    name = f"ag2_car_{tag}_{os.environ.get('AG2_CAR_SESSION', 'default')}"
    buf = _open_region(name, total)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    cudart = ctypes.CDLL("libcudart.so")
    rc = cudart.cudaHostRegister(
        ctypes.c_void_p(addr),
        ctypes.c_size_t(total),
        0x02,  # cudaHostRegisterMapped
    )
    if rc not in (0, 712):
        raise RuntimeError(f"cudaHostRegister({total}) failed: {rc}")
    dev = ctypes.c_void_p()
    if (
        cudart.cudaHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(addr), 0)
        != 0
    ):
        raise RuntimeError("cudaHostGetDevicePointer failed")
    _registered[name] = (buf, addr)
    base = dev.value
    logger.info(
        "AG2 host shared buffer '%s': %d ranks x %d bytes (stride %d) at 0x%x",
        name,
        world_size,
        size_in_bytes,
        stride,
        base,
    )
    return [base + i * stride for i in range(world_size)]


def free_host_shared_buffer(tag: str) -> None:
    for name in [k for k in _registered if k.startswith(f"ag2_car_{tag}_")]:
        buf, addr = _registered.pop(name)
        try:
            ctypes.CDLL("libcudart.so").cudaHostUnregister(ctypes.c_void_p(addr))
        finally:
            buf.close()
