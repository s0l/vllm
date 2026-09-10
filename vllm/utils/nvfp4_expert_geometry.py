# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared source and memory geometry for uniformly partitioned NVFP4 experts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NVFP4ExpertGeometry:
    hidden: int
    width: int
    tp: int
    local_alignment: int = 64

    def __post_init__(self):
        if any(
            type(v) is not int or v <= 0
            for v in (self.hidden, self.width, self.tp, self.local_alignment)
        ) or any(v % 16 for v in (self.hidden, self.width, self.local_alignment)):
            raise ValueError("invalid NVFP4 expert partition geometry")

    @property
    def local(self):
        quantum = self.tp * self.local_alignment
        return (self.width + quantum - 1) // quantum * self.local_alignment

    @property
    def physical(self):
        return self.tp * self.local

    @property
    def strides(self):
        """Bytes of W13, W2 and their separately swizzled block scales."""
        h, n = self.hidden, self.local
        return (
            2 * n * (h // 2),
            h * (n // 2),
            ((2 * n + 127) // 128 * 128) * ((h // 16 + 3) // 4 * 4),
            ((h + 127) // 128 * 128) * ((n // 16 + 3) // 4 * 4),
        )

    def validate_cutlass(self):
        # Native grouped GEMM addresses scales with unpadded N * (K / 16).
        # Both GEMMs must therefore already satisfy the scale swizzle layout.
        if self.hidden % 128 or self.local % 64:
            raise ValueError("expert partition does not satisfy CUTLASS scale layout")
