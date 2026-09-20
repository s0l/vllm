# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine compile admission and owner identity before any model load."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, CompilationConfig, ParallelConfig, VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    _resolve_decode_cudagraph_mode,
)
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


@pytest.mark.parametrize("mode", list(CUDAGraphMode))
@pytest.mark.parametrize("provider", [None, "native_unsafe", "ple_only", "native_safe"])
def test_target_host_provider_capture_capability(monkeypatch, mode, provider):
    from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState

    state = object.__new__(Qwen4ExpModelState)
    state._native_providers = ()
    state._mmap_ple_modules = ()
    if provider == "native_unsafe":
        state._native_providers = (
            SimpleNamespace(full_cudagraph_max_tokens=lambda: 0),
        )
    elif provider == "native_safe":
        state._native_providers = (
            SimpleNamespace(full_cudagraph_max_tokens=lambda: 4),
        )
    elif provider == "ple_only":
        embedding = SimpleNamespace(raw=object(), pinned=object(), event=object())
        state._mmap_ple_modules = (SimpleNamespace(ngram_embedding=embedding),)
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    expected = (
        CUDAGraphMode.PIECEWISE
        if provider == "native_unsafe" and mode != CUDAGraphMode.NONE
        else mode
    )
    assert state.resolve_cudagraph_mode(mode) == expected
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    if provider and mode != CUDAGraphMode.NONE:
        with pytest.raises(RuntimeError, match="require breakable"):
            state.resolve_cudagraph_mode(mode)
    else:
        assert state.resolve_cudagraph_mode(mode) == mode
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    assert state.resolve_cudagraph_mode(mode) == expected
    state._native_providers = ()
    state._mmap_ple_modules = ()
    assert state.resolve_cudagraph_mode(mode) == mode


def test_temporal_e8_capability_is_not_bound_to_an_active_step():
    from vllm.models.qwen4_exp.nvidia.expert_offload_provider import (
        NativeExpertProvider,
    )

    provider = object.__new__(NativeExpertProvider)
    provider.e8_path = SimpleNamespace(
        temporal_cache=True,
        demand_loader=object(),
        deterministic_scatter=True,
        demand_mode=False,
    )
    assert provider.full_cudagraph_max_tokens() == 4


def test_inherited_trace_is_optional_without_legacy_constructor(monkeypatch):
    from vllm.model_executor.models import qwen3_5
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration

    model = object.__new__(Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(model)
    # Disabled tracing must not inspect model inputs or require trace buffers.
    model.maybe_save_ag2_layer0_trace(None, None, 5, "NONE")
    model._ag2_layer0_trace_output = "enabled-diagnostic"
    model._ag2_layer0_trace_saved_query_lens = set()
    model._ag2_layer0_trace_query_lens = {1}
    model._ag2_layer0_trace_token_id = 9
    model._ag2_layer0_trace_position = 2
    calls = []

    def match(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(qwen3_5, "_ag2_layer0_trace_match_row", match)
    model.maybe_save_ag2_layer0_trace(torch.tensor([9]), torch.tensor([2]), 1, "FULL")
    assert len(calls) == 1 and calls[0]["token_ids"] == [9]
    assert calls[0]["output"] == "enabled-diagnostic"
    model._ag2_layer0_trace_output = None
    model.maybe_save_ag2_layer0_trace(None, None, 5, "NONE")
    assert len(calls) == 1


@pytest.mark.parametrize("mode", [None, *CompilationMode])
@pytest.mark.parametrize("enabled", [False, True])
def test_breakable_explicit_stock_and_default(monkeypatch, mode, enabled):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(enabled)))
    config = SimpleNamespace(
        compilation_config=CompilationConfig(mode=mode),
        _uses_breakable_cudagraph_by_default=lambda: False,
    )
    assert VllmConfig._maybe_enable_breakable_cudagraph(config) is enabled
    expected = mode
    if enabled and mode != CompilationMode.STOCK_TORCH_COMPILE:
        expected = CompilationMode.NONE
    assert config.compilation_config.mode == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_stock_v2_admission_requires_breakable(monkeypatch, enabled):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(enabled)))
    config = SimpleNamespace(
        compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE),
        model_config=None,
        speculative_config=None,
        parallel_config=ParallelConfig(),
        cache_config=CacheConfig(),
    )
    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)
    assert ("stock torch.compile" in unsupported) is not enabled


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", list(CUDAGraphMode))
def test_draft_decode_graph_mode(monkeypatch, enabled, mode):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(enabled)))
    actual = _resolve_decode_cudagraph_mode(mode)
    if mode in (
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ):
        assert actual == CUDAGraphMode.FULL_DECODE_ONLY
    elif mode == CUDAGraphMode.PIECEWISE and enabled:
        assert actual == CUDAGraphMode.PIECEWISE
    else:
        assert actual == CUDAGraphMode.NONE


class _Model(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, value):
        return value * self.weight

    def get_mtp_target_hidden_states(self):
        return self.weight


def _runner(compilation):
    runner = object.__new__(GPUModelRunner)
    weight = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    runner.model = _Model(weight)
    runner.speculator = object.__new__(MTPSpeculator)
    runner.speculator.model = _Model(weight)
    runner.vllm_config = SimpleNamespace(compilation_config=compilation)
    return runner


def test_actual_inplace_compile_preserves_shared_owner_and_mutation():
    config = CompilationConfig(
        mode=CompilationMode.STOCK_TORCH_COMPILE, backend="eager"
    )
    runner = _runner(config)
    target, draft = runner.model, runner.get_draft_model()
    original = target.get_mtp_target_hidden_states()
    runner._compile_loaded_models()
    assert runner.model is target and runner.get_draft_model() is draft
    assert target.get_mtp_target_hidden_states() is draft.weight is original
    for value in (2.0, 7.0, 2.0):
        original.fill_(value)
        assert torch.equal(target(torch.tensor([3.0])), torch.tensor([3 * value]))
        assert torch.equal(draft(torch.tensor([3.0])), torch.tensor([3 * value]))


def test_inductor_options_reach_both_finalized_models(monkeypatch):
    options = {"emulate_precision_casts": True, "enable_auto_functionalized_v2": False}
    config = SimpleNamespace(
        mode=CompilationMode.STOCK_TORCH_COMPILE,
        init_backend=lambda _: "inductor",
        inductor_compile_config=options,
    )
    runner = _runner(config)
    calls = []
    monkeypatch.setattr(
        _Model, "compile", lambda owner, **kwargs: calls.append((owner, kwargs))
    )
    runner._compile_loaded_models()
    assert [owner for owner, _ in calls] == [runner.model, runner.get_draft_model()]
    for _, kwargs in calls:
        assert kwargs == dict(fullgraph=True, backend="inductor", options=options)
        assert kwargs["options"] is not options
    config.mode = CompilationMode.NONE
    runner._compile_loaded_models()
    assert len(calls) == 2


def test_partial_breakable_capture_releases_before_propagating_callback_error(
    monkeypatch,
):
    from vllm.compilation import breakable_cudagraph as module
    from vllm.forward_context import BatchDescriptor

    captures = []

    class Capture:
        def __init__(self, pool):
            self.released = False
            captures.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def reset(self):
            self.released = True

    def fail():
        raise RuntimeError("callback failed after first segment")

    monkeypatch.setattr(module, "BreakableCUDAGraphCapture", Capture)
    monkeypatch.setattr(
        module.current_platform, "get_global_graph_pool", lambda: object()
    )
    monkeypatch.setattr(module, "set_graph_pool_id", lambda _: None)
    monkeypatch.setattr(module.gc, "isenabled", lambda: False)
    monkeypatch.setattr(
        module, "get_offloader", lambda: SimpleNamespace(sync_prev_onload=lambda: None)
    )
    wrapper = module.BreakableCUDAGraphWrapper(
        fail, SimpleNamespace(compilation_config=None)
    )
    entry = module._BreakableEntry(batch_descriptor=BatchDescriptor(num_tokens=1))
    wrapper.entries[entry.batch_descriptor] = entry
    with pytest.raises(RuntimeError, match="callback failed"):
        wrapper._capture(entry, (), {})
    assert len(captures) == 1 and captures[0].released
    assert entry.capture is None and entry.output is None
    assert entry.input_addresses is None
    assert not wrapper.entries
