# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import VllmConfig
from vllm.config.device import DeviceConfig
from vllm.forward_context import set_forward_context
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer, _ple_short_conv
from vllm.utils.torch_utils import _encode_layer_name


def test_ple_dispatch_fake_compile_does_not_trace_capture_callback(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fake tracing entered the host capture callback")

    monkeypatch.setattr(
        BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: SimpleNamespace(_capturing=True, add_eager=forbidden)),
    )
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def call(value):
        residual = torch.zeros_like(value)
        torch.ops.vllm.qwen4_exp_ple_short_conv(
            value, residual, _encode_layer_name("unallocated.ple")
        )
        return residual + 1

    compiled = torch.compile(call, fullgraph=True, backend=backend)
    output = compiled(torch.empty(7, 16, device="meta"))
    assert output.device.type == "meta" and len(graphs) == 1
    assert any(
        "qwen4_exp_ple_short_conv" in str(node.target) for node in graphs[0].graph.nodes
    )


def test_ple_dispatch_resolves_current_owner(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    prefix = "test.ple"
    owner = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(owner)
    owner._short_conv = lambda x, y: y.copy_(x + 1)
    config.compilation_config.static_forward_context[prefix] = owner
    value, residual = torch.arange(8).float(), torch.zeros(8)
    with set_forward_context({}, config):
        _ple_short_conv(value, residual, _encode_layer_name(prefix))
    assert torch.equal(residual, value + 1)
    owner._short_conv = lambda x, y: y.copy_(x + 2)
    with set_forward_context({}, config):
        _ple_short_conv(value, residual, _encode_layer_name(prefix))
    assert torch.equal(residual, value + 2)
    before = residual.clone()
    config.compilation_config.static_forward_context[prefix] = object()
    with set_forward_context({}, config), pytest.raises(TypeError, match="owner"):
        _ple_short_conv(value, residual, _encode_layer_name(prefix))
    assert torch.equal(residual, before)
    del config.compilation_config.static_forward_context[prefix]
    with set_forward_context({}, config), pytest.raises(KeyError):
        _ple_short_conv(value, residual, _encode_layer_name(prefix))
    assert torch.equal(residual, before)


@pytest.mark.parametrize("metadata", [[], {"test.ple": object()}])
def test_ple_metadata_rejection_precedes_state_access(monkeypatch, metadata):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    owner = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(owner)
    owner.prefix = "test.ple"
    value, residual = torch.arange(8).float(), torch.full((8,), 9.0)
    before = residual.clone()
    with (
        set_forward_context(metadata, config),
        pytest.raises((TypeError, RuntimeError)),
    ):
        owner._short_conv(value, residual)
    assert torch.equal(residual, before)
    with set_forward_context(None, config):
        owner._short_conv(value, residual)
    assert torch.equal(residual, before)
