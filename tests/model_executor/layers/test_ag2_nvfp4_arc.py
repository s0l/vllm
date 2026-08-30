# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.kernels.linear.nvfp4 import arc
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4,
)
from vllm.model_executor.layers.quantization.utils import ag2_nvfp4_arc


def _sidecar() -> ag2_nvfp4_arc._Sidecar:
    return ag2_nvfp4_arc._Sidecar(
        manifest={"sidecar_sha256": "a" * 64},
        by_suffix={"layer.0": {}, "layer.1": {}},
        tensors={},
    )


def test_arc_application_logging_is_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    info = Mock()
    monkeypatch.setattr(ag2_nvfp4_arc.logger, "info", info)
    sidecar = _sidecar()

    ag2_nvfp4_arc._record_applied_prefix(sidecar, "model.layer.0", rank=2)
    info.assert_not_called()

    ag2_nvfp4_arc._record_applied_prefix(sidecar, "model.layer.1", rank=2)
    info.assert_called_once_with(
        "AG2 ARC application complete rank=%d records=%d sha256=%s",
        2,
        2,
        "a" * 64,
    )


def test_arc_application_rejects_duplicate_prefix() -> None:
    sidecar = _sidecar()
    ag2_nvfp4_arc._record_applied_prefix(sidecar, "model.layer.0", rank=0)

    with pytest.raises(RuntimeError, match="applied twice"):
        ag2_nvfp4_arc._record_applied_prefix(sidecar, "model.layer.0", rank=0)


def test_base_owner_metadata_installs_empty_capability(monkeypatch) -> None:
    layer = torch.nn.Linear(16, 8, bias=False)
    monkeypatch.setattr(ag2_nvfp4_arc, "_owner_runtime_active", lambda: True)
    assert ag2_nvfp4_arc.ensure_ag2_nvfp4_base_owner_metadata(layer)
    assert layer._ag2_nvfp4_owner_metadata_mode == "native-base-only"
    assert layer._ag2_nvfp4_arc_selected.shape == (0,)
    assert layer._ag2_nvfp4_arc_selected_all.shape == (3, 0)
    assert layer._ag2_nvfp4_arc_route_counts == (0,) * 9
    assert all(
        getattr(layer, f"_ag2_nvfp4_arc_route_to_{rank}").shape == (0,)
        for rank in range(3)
    )


def test_base_owner_metadata_preserves_model_bound_arc(monkeypatch) -> None:
    layer = torch.nn.Linear(16, 8, bias=False)
    layer.register_buffer(
        "_ag2_nvfp4_arc_selected_all", torch.empty((3, 2), dtype=torch.int32)
    )
    layer._ag2_nvfp4_owner_metadata_mode = "model-bound-arc"
    monkeypatch.setattr(ag2_nvfp4_arc, "_owner_runtime_active", lambda: True)

    assert not ag2_nvfp4_arc.ensure_ag2_nvfp4_base_owner_metadata(layer)
    assert layer._ag2_nvfp4_arc_selected_all.shape == (3, 2)
    assert layer._ag2_nvfp4_owner_metadata_mode == "model-bound-arc"


def test_base_owner_metadata_is_inactive_without_owner_runtime(monkeypatch) -> None:
    layer = torch.nn.Linear(16, 8, bias=False)
    monkeypatch.setattr(ag2_nvfp4_arc, "_owner_runtime_active", lambda: False)

    assert not ag2_nvfp4_arc.ensure_ag2_nvfp4_base_owner_metadata(layer)
    assert not hasattr(layer, "_ag2_nvfp4_arc_selected_all")


def test_compressed_tensors_installs_owner_metadata_before_kernel(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_TP3_OWNER_PREQUANT", "1")
    monkeypatch.delenv("AG2_VLLM_NVFP4_ARC_SIDECAR_DIR", raising=False)
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.gate_up_proj"
    layer.weight_packed = torch.nn.Parameter(
        torch.zeros((8, 8), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((8, 1), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_global_scale = torch.nn.Parameter(
        torch.ones(1), requires_grad=False
    )
    layer.input_global_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def assert_owner_metadata(current: torch.nn.Module) -> None:
        assert current._ag2_nvfp4_owner_metadata_mode == "native-base-only"

    kernel = Mock()
    kernel.process_weights_after_loading.side_effect = assert_owner_metadata
    scheme = object.__new__(compressed_tensors_w4a4_nvfp4.CompressedTensorsW4A4Fp4)
    scheme.use_a16 = False
    scheme.kernel = kernel

    scheme.process_weights_after_loading(layer)

    kernel.process_weights_after_loading.assert_called_once_with(layer)


def test_modelopt_installs_owner_metadata_before_kernel(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_TP3_OWNER_PREQUANT", "1")
    monkeypatch.delenv("AG2_VLLM_NVFP4_ARC_SIDECAR_DIR", raising=False)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((8, 8), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((8, 1), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.input_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    layer.weight_scale_2 = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def assert_owner_metadata(current: torch.nn.Module) -> None:
        assert current._ag2_nvfp4_owner_metadata_mode == "native-base-only"

    kernel = Mock()
    kernel.process_weights_after_loading.side_effect = assert_owner_metadata
    method = object.__new__(modelopt.ModelOptNvFp4LinearMethod)
    method.kernel = kernel
    method.layer_prefix = "model.layers.0.mlp.gate_up_proj"

    method.process_weights_after_loading(layer)

    kernel.process_weights_after_loading.assert_called_once_with(layer)


def test_empty_arc_tail_uses_plain_nvfp4_quantization(monkeypatch) -> None:
    x = torch.arange(128, dtype=torch.bfloat16).reshape(2, 64)
    scale = torch.ones(1)
    selected = torch.empty(0, dtype=torch.int32)
    expected_packed = torch.zeros((2, 32), dtype=torch.uint8)
    expected_scales = torch.ones((2, 4), dtype=torch.float8_e4m3fn)
    quantize = Mock(return_value=(expected_packed, expected_scales))
    tail = Mock()
    monkeypatch.setattr(arc, "scaled_fp4_quant", quantize)
    monkeypatch.setattr(arc, "_write_residual_tail_kernel", tail)

    packed, scales = arc._arc_quantize_impl(x, scale, selected)

    assert packed.shape == (2, 32)
    assert packed.data_ptr() == expected_packed.data_ptr()
    assert scales.data_ptr() == expected_scales.data_ptr()
    quantize.assert_called_once()
    args, kwargs = quantize.call_args
    assert args[0].shape == x.shape
    assert args[0].data_ptr() == x.data_ptr()
    assert args[1] is scale
    assert kwargs == {
        "is_sf_swizzled_layout": True,
        "backend": "b12x",
        "padded_n": 64,
    }
    tail.assert_not_called()


def test_arc_quantize_rejects_non_fp16_input() -> None:
    with pytest.raises(ValueError, match=r"BF16/FP16"):
        arc._arc_quantize_impl(
            torch.zeros((2, 64), dtype=torch.float32),
            torch.ones(1),
            torch.empty(0, dtype=torch.int32),
        )
