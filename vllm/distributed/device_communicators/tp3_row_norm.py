# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit arithmetic ABI for experimental H5120 row continuation.

Launch specifications are immutable data, bound before compilation. No compiler
cache Python or evaluation module is imported by this runtime adapter.
"""

from dataclasses import dataclass

import torch

from . import tp3_row_norm_kernels as kernels


@dataclass(frozen=True)
class RowNormSpec:
    role: str
    rblock: int
    warps: int

    def __post_init__(self):
        if self.role not in ("first", "post", "next", "terminal"):
            raise ValueError("unknown row norm role")
        if (self.rblock, self.warps) not in (
            (512, 8),
            (1024, 8),
            (4096, 16),
            (8192, 16),
        ):
            raise ValueError("unproved row norm launch configuration")

    def apply(self, inputs: tuple[torch.Tensor, ...], gamma: torch.Tensor):
        normalized, carry, _ = self.apply_with_binary(inputs, gamma)
        return normalized, carry

    def apply_with_binary(self, inputs: tuple[torch.Tensor, ...], gamma: torch.Tensor):
        count = {"first": 1, "post": 2, "next": 3, "terminal": 3}[self.role]
        if len(inputs) != count:
            raise ValueError("wrong raw residual operand count")
        shape = inputs[0].shape
        if len(shape) != 2 or shape[1] != 5120 or not 1 <= shape[0] <= 4096:
            raise ValueError("row norm requires M1..4096/H5120")
        if any(
            x.shape != shape
            or x.dtype != torch.bfloat16
            or x.device != inputs[0].device
            or not x.is_contiguous()
            for x in inputs
        ) or (
            gamma.shape != (5120,)
            or gamma.dtype != torch.bfloat16
            or gamma.device != inputs[0].device
            or not gamma.is_contiguous()
        ):
            raise ValueError("row norm requires matching original BF16 operands")
        carry = None
        if self.role == "terminal":
            # The accepted kernel overwrites its first operand. Return a fresh
            # result so the external functional ABI never aliases an input.
            normalized = inputs[0].clone()
            args = (normalized, *inputs[1:], gamma)
            kernel = kernels.norm_6331af9b3b5b
        else:
            normalized = torch.empty_like(inputs[0])
            if self.role == "first":
                kernel = kernels.norm_9dffa1cf3ba0
                args = (*inputs, gamma, normalized)
            elif self.role == "post":
                kernel = kernels.norm_27f5f8cce2f4
                args = (*inputs, gamma, normalized)
            else:
                kernel = kernels.norm_a89f489db113
                carry = torch.empty_like(inputs[0])
                args = (*inputs, gamma, carry, normalized)
        binary = kernel[(shape[0],)](
            *args,
            shape[0],
            5120,
            XBLOCK=1,
            R0_BLOCK=self.rblock,
            num_warps=self.warps,
            num_stages=1,
            enable_fp_fusion=True,
        )
        return normalized, carry, binary
