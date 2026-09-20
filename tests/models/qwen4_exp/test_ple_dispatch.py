# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.forward_context import set_forward_context
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer


def _owner(prefix: str = "test.ple") -> Qwen4ExpPLELayer:
    owner = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(owner)
    owner.prefix = prefix
    return owner


def test_ple_dispatch_defers_current_inputs_to_capture_callback(monkeypatch):
    monkeypatch.setattr(
        "vllm.compilation.breakable_cudagraph._weak_ref_capture_arg",
        lambda value: value,
    )
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    envs.disable_envs_cache()
    callbacks = []
    capture = SimpleNamespace(_capturing=True, add_eager=callbacks.append)
    monkeypatch.setattr(
        BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: capture),
    )
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    owner = _owner()
    value = torch.arange(8).float()
    residual = torch.zeros(8)
    outer_residual = torch.ones(8)

    raw_short_conv = getattr(owner._short_conv, "__wrapped__", owner._short_conv)
    captured_short_conv = eager_break_during_capture(raw_short_conv)
    with set_forward_context(
        None, config, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
    ):
        captured_short_conv(value, residual, outer_residual)
    assert len(callbacks) == 1
    assert torch.equal(residual, torch.zeros_like(residual))

    with set_forward_context(None, config):
        callbacks[0]()
    assert torch.equal(residual, outer_residual)


@pytest.mark.parametrize("metadata", [[], {"test.ple": object()}])
def test_ple_metadata_rejection_precedes_state_access(metadata):
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    owner = _owner()
    value = torch.arange(8).float()
    residual = torch.full((8,), 9.0)
    outer_residual = torch.ones(8)
    before = residual.clone()
    with (
        set_forward_context(metadata, config),
        pytest.raises((TypeError, RuntimeError)),
    ):
        raw_short_conv = getattr(owner._short_conv, "__wrapped__", owner._short_conv)
        raw_short_conv(value, residual, outer_residual)
    assert torch.equal(residual, before)


def test_ple_profile_path_preserves_outer_residual():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    owner = _owner()
    value = torch.arange(8).float()
    residual = torch.full((8,), 9.0)
    outer_residual = torch.arange(8).float()
    with set_forward_context(None, config):
        raw_short_conv = getattr(owner._short_conv, "__wrapped__", owner._short_conv)
        raw_short_conv(value, residual, outer_residual)
    assert torch.equal(residual, torch.full((8,), 9.0) + outer_residual)
