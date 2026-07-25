"""Deterministic CUDA control for native-Inductor donated RMS buffers."""

from __future__ import annotations

import torch
from torch import nn

import vllm.kernels  # noqa: F401
from tests.compile.backend import TestBackend
from vllm import ir
from vllm.compilation.passes.ir.clone_elimination import (
    UnsafeCloneEliminationPass,
)
from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
from vllm.config import SchedulerConfig, VllmConfig, set_current_vllm_config


class GemmaDonationControl(nn.Module):
    def __init__(self, hidden_size: int, *, donate: bool):
        super().__init__()
        self.register_buffer(
            "weight", torch.zeros(hidden_size, dtype=torch.bfloat16)
        )
        self.donate = donate

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        weight = self.weight.float() + 1.0
        op = (
            ir.ops.fused_add_rms_norm.maybe_inplace
            if self.donate
            else ir.ops.fused_add_rms_norm
        )
        return op(x, residual, weight, 1e-6)


def main() -> None:
    torch.set_default_device("cuda")
    torch.manual_seed(17)
    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=6656,
            max_model_len=6656,
            is_encoder_decoder=False,
        )
    )
    config.kernel_config.ir_op_priority.fused_add_rms_norm = [
        "native_inductor_inplace",
        "native",
    ]

    with set_current_vllm_config(config):
        baseline_lowering = VllmIRLoweringPass(config)
        baseline_cleanup = UnsafeCloneEliminationPass(config)
        baseline_backend = TestBackend(baseline_lowering, baseline_cleanup)
        candidate_lowering = VllmIRLoweringPass(config)
        candidate_cleanup = UnsafeCloneEliminationPass(config)
        candidate_backend = TestBackend(candidate_lowering, candidate_cleanup)
        baseline_model = GemmaDonationControl(hidden_size=5120, donate=False)
        candidate_model = GemmaDonationControl(hidden_size=5120, donate=True)

        with ir.ops.fused_add_rms_norm.set_priority(["native"]):
            baseline = torch.compile(
                baseline_model, backend=baseline_backend, fullgraph=True
            )
        with ir.ops.fused_add_rms_norm.set_priority(
            ["native_inductor_inplace", "native"]
        ):
            candidate = torch.compile(
                candidate_model, backend=candidate_backend, fullgraph=True
            )
            results = []
            for rows in (1, 128, 6656):
                x = torch.randn(rows, 5120, dtype=torch.bfloat16)
                residual = torch.randn_like(x)
                with ir.ops.fused_add_rms_norm.set_priority(["native"]):
                    expected = baseline(x.clone(), residual.clone())
                with ir.ops.fused_add_rms_norm.set_priority(
                    ["native_inductor_inplace", "native"]
                ):
                    output = candidate(x, residual)
                torch.cuda.synchronize()
                exact = [
                    bool(torch.equal(output[i], expected[i])) for i in range(2)
                ]
                aliases = [
                    output[0].data_ptr() == x.data_ptr(),
                    output[1].data_ptr() == residual.data_ptr(),
                ]
                assert exact == [True, True], (rows, exact)
                assert aliases == [True, True], (rows, aliases)
                results.append(
                    {"rows": rows, "bit_exact": exact, "aliases_inputs": aliases}
                )

    baseline_providers = set(
        baseline_lowering.selected_impls["fused_add_rms_norm"].values()
    )
    assert baseline_providers == {"native"}, baseline_providers
    providers = set(
        candidate_lowering.selected_impls["fused_add_rms_norm"].values()
    )
    assert providers == {"native_inductor_inplace"}, providers
    assert (
        candidate_backend.op_count(torch.ops.aten.clone.default, before=False)
        == 0
    )
    print(
        {
            "results": results,
            "baseline_providers": sorted(baseline_providers),
            "providers": sorted(providers),
            "post_lowering_clones": 0,
        }
    )


if __name__ == "__main__":
    main()
