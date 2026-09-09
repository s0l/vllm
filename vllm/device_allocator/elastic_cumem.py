# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA VMM backing tensors with stable virtual addresses.

This module is intentionally narrow: one PyTorch MemPool owns one backing
tensor, while the C++ allocator maps a quantum-aligned physical prefix of its
reserved virtual address.  Callers must fence all users before resizing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.utils.system_utils import find_loaded_library

try:
    from vllm.cumem_allocator import (
        configure_elastic,
        elastic_info,
        resize_elastic,
        transfer_elastic,
    )
except ModuleNotFoundError:
    configure_elastic = elastic_info = resize_elastic = transfer_elastic = None


@dataclass(frozen=True)
class ElasticAllocationInfo:
    base: int
    reserved: int
    committed: int
    quantum: int


class ElasticCuMemBacking:
    """Own one stable-VA uint8 tensor and its pluggable allocator wrappers."""

    def __init__(
        self,
        reserved_bytes: int,
        committed_bytes: int,
        quantum_bytes: int,
        device: torch.device,
    ) -> None:
        if configure_elastic is None:
            raise RuntimeError("vLLM cumem allocator extension is unavailable")
        if device.type != "cuda":
            raise ValueError("elastic CuMem backing requires a CUDA device")
        if reserved_bytes <= 0 or quantum_bytes <= 0:
            raise ValueError("reserved_bytes and quantum_bytes must be positive")
        if committed_bytes % quantum_bytes:
            raise ValueError("committed_bytes must be quantum-aligned")

        configure_elastic(committed_bytes, quantum_bytes)
        lib_name = find_loaded_library("cumem_allocator")
        self._allocator = torch.cuda.memory.CUDAPluggableAllocator(
            lib_name, "elastic_malloc", "elastic_free"
        )
        self._pool = torch.cuda.memory.MemPool(self._allocator._allocator)
        with torch.cuda.memory.use_mem_pool(self._pool):
            # Do not zero: touching the reserved but unmapped tail is invalid.
            self.tensor = torch.empty(reserved_bytes, dtype=torch.uint8, device=device)
        info = self.info
        if info.base != self.tensor.data_ptr() or info.committed != committed_bytes:
            raise RuntimeError(
                "elastic allocator geometry mismatch: "
                f"tensor={self.tensor.data_ptr()} info={info}"
            )

    @property
    def info(self) -> ElasticAllocationInfo:
        assert elastic_info is not None
        return ElasticAllocationInfo(*elastic_info(self.tensor.data_ptr()))

    def resize(self, committed_bytes: int, event: torch.cuda.Event | None) -> None:
        """Resize after a caller-provided completion fence.

        An event is mandatory after publication.  Initial controller setup may
        use ``None`` only before any kernel can observe the backing.
        """
        if event is not None:
            event.synchronize()
        elif self.info.committed != 0:
            raise RuntimeError("resizing a published backing requires a CUDA event")
        assert resize_elastic is not None
        resize_elastic(self.tensor.data_ptr(), committed_bytes, True)

    def transfer_to(
        self,
        destination: ElasticCuMemBacking,
        bytes_: int,
        event: torch.cuda.Event,
    ) -> None:
        """Move physical quanta to another stable-VA backing after a fence."""
        if event is None:
            raise RuntimeError("elastic transfer requires a CUDA event")
        event.synchronize()
        assert transfer_elastic is not None
        transfer_elastic(
            self.tensor.data_ptr(),
            destination.tensor.data_ptr(),
            bytes_,
            True,
        )


_live_backings: list[ElasticCuMemBacking] = []


def allocate_elastic_backing(
    reserved_bytes: int,
    committed_bytes: int,
    quantum_bytes: int,
    device: torch.device,
) -> ElasticCuMemBacking:
    backing = ElasticCuMemBacking(
        reserved_bytes, committed_bytes, quantum_bytes, device
    )
    # KV arenas are process-lifetime objects. Keep allocator wrappers alive
    # longer than tensors/graphs, matching CuMemAllocator's lifetime ordering.
    _live_backings.append(backing)
    return backing
