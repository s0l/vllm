# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig

pytestmark = pytest.mark.cpu_test


def _elastic_config(capture_sizes=None):
    compilation_kwargs = {}
    if capture_sizes is not None:
        compilation_kwargs["cudagraph_capture_sizes"] = capture_sizes
    return SimpleNamespace(
        additional_config={"elastic_gdn_backing": True},
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            compile_sizes=["cudagraph_capture_sizes"],
            **compilation_kwargs,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )


def test_elastic_cudagraph_config_has_runtime_ceiling_without_static_sizes():
    config = _elastic_config()
    VllmConfig._set_cudagraph_sizes(config)

    assert config.compilation_config.cudagraph_capture_sizes == []
    assert config.compilation_config.max_cudagraph_capture_size == 4096
    assert config.compilation_config.compile_sizes == []
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


def test_elastic_cudagraph_config_rejects_explicit_static_sizes():
    config = _elastic_config([1, 2, 4])
    with pytest.raises(ValueError, match="runtime-derived"):
        VllmConfig._set_cudagraph_sizes(config)
