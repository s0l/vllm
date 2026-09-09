#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compiled exactness control for the MTP final single-output RMS contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compile.backend import TestBackend

import vllm.kernels  # noqa: F401
from vllm import (
    _custom_ops,  # noqa: F401
    ir,
)
from vllm.compilation.passes.ir.clone_elimination import (
    UnsafeCloneEliminationPass,
)
from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
from vllm.config import VllmConfig, set_current_vllm_config

HIDDEN_SIZE = 5120
ROWS = (1, 128, 6656)


class FinalNormControl(nn.Module):
    def __init__(self, *, output_only: bool) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16),
            persistent=False,
        )
        self.output_only = output_only

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() + 1.0
        if self.output_only:
            return ir.ops.fused_add_rms_norm_output_only(x, residual, weight, 1e-6)
        return ir.ops.fused_add_rms_norm(x, residual, weight, 1e-6)[0]


@torch.inference_mode()
def main() -> int:
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    results: list[dict[str, int | bool]] = []

    for rows in ROWS:
        x = torch.randn(rows, HIDDEN_SIZE, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        config = VllmConfig()
        config.scheduler_config.max_num_batched_tokens = max(ROWS)
        with set_current_vllm_config(config):
            baseline_lowering = VllmIRLoweringPass(config)
            baseline_backend = TestBackend(
                baseline_lowering,
                UnsafeCloneEliminationPass(config),
            )
            candidate_lowering = VllmIRLoweringPass(config)
            candidate_backend = TestBackend(
                candidate_lowering,
                UnsafeCloneEliminationPass(config),
            )
            with ir.ops.fused_add_rms_norm.set_priority(["native"]):
                baseline = torch.compile(
                    FinalNormControl(output_only=False),
                    backend=baseline_backend,
                    fullgraph=True,
                )
                expected = baseline(x, residual)
            candidate = torch.compile(
                FinalNormControl(output_only=True),
                backend=candidate_backend,
                fullgraph=True,
            )
            actual = candidate(x, residual)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert set(baseline_lowering.selected_impls["fused_add_rms_norm"].values()) == {
            "native"
        }
        assert set(
            candidate_lowering.selected_impls["fused_add_rms_norm_output_only"].values()
        ) == {"native"}
        results.append(
            {
                "rows": rows,
                "bit_exact": bool(torch.equal(actual, expected)),
                "output_bytes": actual.numel() * actual.element_size(),
            }
        )

    print(json.dumps({"result": "pass", "rows": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
