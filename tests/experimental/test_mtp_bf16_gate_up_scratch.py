# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP


class _Linear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.gather_output = False
        self.quant_method = UnquantizedLinearMethod()


class _GateUp(nn.Module):
    def __init__(self, mlp: Qwen2MoeMLP) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp._gate_up(value)


def _make_mlp(device: torch.device) -> Qwen2MoeMLP:
    mlp = Qwen2MoeMLP.__new__(Qwen2MoeMLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = _Linear(
        torch.randn(128, 64, device=device, dtype=torch.bfloat16)
    )
    mlp.register_buffer("_bf16_gate_up_scratch", None, persistent=False)
    return mlp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_gate_up_uses_stable_caller_owned_output() -> None:
    device = torch.device("cuda:0")
    mlp = _make_mlp(device)
    mlp.enable_bf16_gate_up_scratch(256)
    assert mlp._bf16_gate_up_scratch is not None
    pointer = mlp._bf16_gate_up_scratch.data_ptr()
    compiled = torch.compile(_GateUp(mlp), fullgraph=True, dynamic=True)

    for rows in (1, 37, 256):
        value = torch.randn(rows, 64, device=device, dtype=torch.bfloat16)
        expected = torch.mm(value, mlp.gate_up_proj.weight.t())
        actual = compiled(value)
        assert torch.equal(actual, expected)
        assert mlp._bf16_gate_up_scratch.data_ptr() == pointer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gate_up_accepts_shared_runner_owned_workspace() -> None:
    device = torch.device("cuda:0")
    mlp = _make_mlp(device)
    workspace = torch.empty(256, 128, device=device, dtype=torch.bfloat16)
    mlp.enable_bf16_gate_up_scratch(256, scratch=workspace)

    assert mlp._bf16_gate_up_scratch is workspace
    value_a = torch.randn(37, 64, device=device, dtype=torch.bfloat16)
    value_b = torch.randn(19, 64, device=device, dtype=torch.bfloat16)
    expected_a = torch.mm(value_a, mlp.gate_up_proj.weight.t())
    expected_b = torch.mm(value_b, mlp.gate_up_proj.weight.t())
    actual_a = mlp._gate_up(value_a).clone()
    actual_b = mlp._gate_up(value_b).clone()

    assert torch.equal(actual_a, expected_a)
    assert torch.equal(actual_b, expected_b)
    assert mlp._bf16_gate_up_scratch.data_ptr() == workspace.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gate_up_scratch_rejects_runtime_oversize() -> None:
    device = torch.device("cuda:0")
    mlp = _make_mlp(device)
    mlp.enable_bf16_gate_up_scratch(8)
    with pytest.raises(RuntimeError, match="smaller than the runtime token batch"):
        mlp._gate_up(torch.randn(9, 64, device=device, dtype=torch.bfloat16))
