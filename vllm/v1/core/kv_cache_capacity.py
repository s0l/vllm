# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhysicalPoolCapacityPlanner:
    """Byte-accurate capacity model for two elastic physical KV pools.

    Primary tensors have independent CUDA-VMM backings, so each tensor is
    rounded to the mapping quantum separately. The secondary pool has one
    packed backing and is rounded once.
    """

    primary_block_sizes: tuple[int, ...]
    secondary_block_stride: int
    mapping_quantum: int
    budget_bytes: int

    def __post_init__(self) -> None:
        if not self.primary_block_sizes or any(
            size <= 0 for size in self.primary_block_sizes
        ):
            raise ValueError("primary block sizes must be positive")
        if self.secondary_block_stride <= 0:
            raise ValueError("secondary block stride must be positive")
        if self.mapping_quantum <= 0:
            raise ValueError("mapping quantum must be positive")
        if self.budget_bytes <= 0:
            raise ValueError("physical pool budget must be positive")

    def _mapped_bytes(self, logical_bytes: int) -> int:
        if logical_bytes < 0:
            raise ValueError("logical bytes must be non-negative")
        quantum = self.mapping_quantum
        return ((logical_bytes + quantum - 1) // quantum) * quantum

    def primary_mapped_bytes(self, num_blocks: int) -> int:
        if num_blocks < 0:
            raise ValueError("primary block count must be non-negative")
        return sum(
            self._mapped_bytes(num_blocks * block_size)
            for block_size in self.primary_block_sizes
        )

    def secondary_mapped_bytes(self, num_blocks: int) -> int:
        if num_blocks < 0:
            raise ValueError("secondary block count must be non-negative")
        return self._mapped_bytes(num_blocks * self.secondary_block_stride)

    def max_primary_blocks(
        self,
        secondary_blocks: int,
        *,
        upper_bound: int,
    ) -> int:
        """Return the largest primary capacity within the physical budget."""
        if upper_bound < 0:
            raise ValueError("primary block upper bound must be non-negative")
        secondary_bytes = self.secondary_mapped_bytes(secondary_blocks)
        for primary_blocks in range(upper_bound, -1, -1):
            if (
                self.primary_mapped_bytes(primary_blocks) + secondary_bytes
                <= self.budget_bytes
            ):
                return primary_blocks
        return 0
