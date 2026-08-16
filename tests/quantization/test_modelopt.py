# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test ModelOpt quantization method setup and weight loading.

Run `pytest tests/quantization/test_modelopt.py`.
"""

import os
from typing import Any, NoReturn
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.model import ModelConfig
from vllm.model_executor.kernels.linear import (
    FlashInferB12xNvFp4LinearKernel,
    HummingNvFp4LinearKernel,
    MarlinNvFp4LinearKernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
    ModelOptNvFp4LinearMethod,
    ModelOptNvFp4W4A16LinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _skip(msg: str) -> NoReturn:
    pytest.skip(msg)
    raise RuntimeError(msg)


def _snapshot_download_or_skip(model_id: str) -> str:
    try:
        from vllm.transformers_utils.repo_utils import hf_api
    except Exception as e:  # pragma: no cover
        _skip(f"huggingface_hub is required to download {model_id}: {e}")

    try:
        return hf_api().snapshot_download(
            repo_id=model_id,
            repo_type="model",
            # These checkpoints are already small; download full repo for simplicity.
            allow_patterns=["*"],
        )
    except Exception as e:
        _skip(f"Failed to download {model_id} from the HF Hub: {e}")


def _mock_lm_head() -> Mock:
    lm_head = Mock(spec=ParallelLMHead)
    lm_head.__class__ = ParallelLMHead
    return lm_head


def _mixed_precision_config(quantized_layers: dict) -> ModelOptMixedPrecisionConfig:
    return ModelOptMixedPrecisionConfig(
        kv_cache_quant_method=None,
        exclude_modules=[],
        quantized_layers=quantized_layers,
        fp8_config=ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        ),
        nvfp4_config=ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        w4a16_nvfp4_config=ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
        mxfp8_config=ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        ),
    )


def test_modelopt_nvfp4_quantizes_parallel_lm_head():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, ModelOptNvFp4LinearMethod)
    assert method.layer_prefix == "lm_head"


def test_modelopt_nvfp4_selective_a16_prefix(monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase

    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    fake_layer = MagicMock(spec=LinearBase)
    monkeypatch.setenv(
        "AG2_VLLM_NVFP4_A16_PREFIXES",
        "*.layers.49.mlp.gate_up_proj,*.layers.60.mlp.gate_up_proj",
    )
    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        selected = config.get_quant_method(
            fake_layer, "language_model.model.layers.60.mlp.gate_up_proj"
        )
        unselected = config.get_quant_method(
            fake_layer, "language_model.model.layers.61.mlp.gate_up_proj"
        )

    assert isinstance(selected, ModelOptNvFp4W4A16LinearMethod)
    assert isinstance(unselected, ModelOptNvFp4LinearMethod)


def test_modelopt_nvfp4_selective_a4_prefix_on_w4a16_base(monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase

    config = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    fake_layer = MagicMock(spec=LinearBase)
    monkeypatch.setenv(
        "AG2_VLLM_NVFP4_A4_PREFIXES",
        ",".join(
            (
                "language_model.model.layers.*.mlp.gate_up_proj",
                "language_model.model.layers.*.linear_attn.in_proj_qkvz",
                "language_model.model.layers.*.self_attn.qkv_proj",
            )
        ),
    )
    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        selected = config.get_quant_method(
            fake_layer, "language_model.model.layers.60.mlp.gate_up_proj"
        )
        selected_gdn = config.get_quant_method(
            fake_layer,
            "language_model.model.layers.60.linear_attn.in_proj_qkvz",
        )
        selected_attention = config.get_quant_method(
            fake_layer,
            "language_model.model.layers.59.self_attn.qkv_proj",
        )
        unselected = config.get_quant_method(
            fake_layer, "language_model.model.layers.60.mlp.down_proj"
        )
        mtp_unselected = config.get_quant_method(
            fake_layer, "mtp.layers.0.mlp.gate_up_proj"
        )

    assert isinstance(selected, ModelOptNvFp4LinearMethod)
    assert selected.layer_prefix.endswith(".mlp.gate_up_proj")
    assert isinstance(selected_gdn, ModelOptNvFp4LinearMethod)
    assert selected_gdn.layer_prefix.endswith(".linear_attn.in_proj_qkvz")
    assert isinstance(selected_attention, ModelOptNvFp4LinearMethod)
    assert selected_attention.layer_prefix.endswith(".self_attn.qkv_proj")
    assert isinstance(unselected, ModelOptNvFp4W4A16LinearMethod)
    assert isinstance(mtp_unselected, ModelOptNvFp4W4A16LinearMethod)


def test_modelopt_nvfp4_selective_a4_requires_w4a16_base(monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase

    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    monkeypatch.setenv(
        "AG2_VLLM_NVFP4_A4_PREFIXES",
        "language_model.model.layers.*.mlp.gate_up_proj",
    )
    with pytest.raises(ValueError, match="requires a W4A16 base"):
        config.get_quant_method(
            MagicMock(spec=LinearBase),
            "language_model.model.layers.60.mlp.gate_up_proj",
        )


def test_modelopt_nvfp4_selective_a4_and_a16_are_mutually_exclusive(monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase

    config = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    monkeypatch.setenv("AG2_VLLM_NVFP4_A4_PREFIXES", "*.mlp.gate_up_proj")
    monkeypatch.setenv("AG2_VLLM_NVFP4_A16_PREFIXES", "*.mlp.down_proj")
    with pytest.raises(ValueError, match="cannot combine A16 and A4"):
        config.get_quant_method(
            MagicMock(spec=LinearBase),
            "language_model.model.layers.60.mlp.gate_up_proj",
        )


def test_nvfp4_batch_invariant_allows_explicit_b12x(monkeypatch):
    import vllm.envs as envs
    import vllm.model_executor.kernels.linear as linear_kernels

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "AG2_VLLM_NVFP4_BATCH_INVARIANT", False)
    monkeypatch.setattr(linear_kernels, "_get_linear_backend", lambda: "flashinfer_b12x")
    monkeypatch.setattr(
        FlashInferB12xNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )

    kernel = init_nvfp4_linear_kernel(use_a16=False)

    assert isinstance(kernel, FlashInferB12xNvFp4LinearKernel)


def test_nvfp4_per_format_b12x_selector(monkeypatch):
    import vllm.envs as envs
    import vllm.model_executor.kernels.linear as linear_kernels

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(envs, "AG2_VLLM_NVFP4_BATCH_INVARIANT", False)
    monkeypatch.setattr(envs, "AG2_VLLM_NVFP4_B12X", True)
    monkeypatch.setattr(linear_kernels, "_get_linear_backend", lambda: "auto")
    monkeypatch.setattr(
        FlashInferB12xNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )

    kernel = init_nvfp4_linear_kernel(use_a16=False)

    assert isinstance(kernel, FlashInferB12xNvFp4LinearKernel)


def test_nvfp4_per_format_b12x_rejects_w4a16(monkeypatch):
    import vllm.envs as envs
    import vllm.model_executor.kernels.linear as linear_kernels

    monkeypatch.setattr(envs, "AG2_VLLM_NVFP4_B12X", True)
    monkeypatch.setattr(linear_kernels, "_get_linear_backend", lambda: "auto")

    with pytest.raises(ValueError, match="supports only W4A4"):
        init_nvfp4_linear_kernel(use_a16=True)


def test_nvfp4_per_format_b12x_fails_when_unsupported(monkeypatch):
    import vllm.envs as envs
    import vllm.model_executor.kernels.linear as linear_kernels

    monkeypatch.setattr(envs, "AG2_VLLM_NVFP4_B12X", True)
    monkeypatch.setattr(linear_kernels, "_get_linear_backend", lambda: "auto")
    monkeypatch.setattr(
        FlashInferB12xNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (False, "test unsupported")),
    )

    with pytest.raises(ValueError, match="test unsupported"):
        init_nvfp4_linear_kernel(use_a16=False)


def test_modelopt_fp8_updates_weight_dims_after_transpose():
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.empty(3, 2), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.ones(1), requires_grad=False)
    )
    layer.register_parameter(
        "input_scale", torch.nn.Parameter(torch.ones(1), requires_grad=False)
    )

    method = ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod)
    method.fp8_linear = Mock()
    method.process_weights_after_loading(layer)

    assert layer.weight.shape == (2, 3)
    assert layer.weight.input_dim == 0
    assert layer.weight.output_dim == 1
    method.fp8_linear.process_weights_after_loading.assert_called_once_with(layer)


def test_modelopt_nvfp4_leaves_excluded_parallel_lm_head_unquantized():
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=["lm_head"],
    )

    method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, UnquantizedLinearMethod)


def test_modelopt_mixed_precision_quantizes_parallel_lm_head():
    config = _mixed_precision_config(
        {"lm_head": {"quant_algo": "NVFP4", "group_size": 16}}
    )

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        method = config.get_quant_method(_mock_lm_head(), prefix="lm_head")

    assert isinstance(method, ModelOptNvFp4LinearMethod)


def test_modelopt_mixed_precision_resolves_declared_packed_projection():
    config = _mixed_precision_config(
        {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "MXFP8"},
            "model.layers.0.self_attn.k_proj": {"quant_algo": "MXFP8"},
            "model.layers.0.self_attn.v_proj": {"quant_algo": "MXFP8"},
        }
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert config._resolve_quant_algo("model.layers.0.self_attn.qkv_proj") == "MXFP8"


def test_modelopt_mixed_precision_does_not_quantize_unlisted_fused_sibling():
    config = _mixed_precision_config(
        {
            "model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
            "model.layers.0.linear_attn.in_proj_z": {"quant_algo": "FP8"},
            "model.layers.0.linear_attn.out_proj": {"quant_algo": "FP8"},
        }
    )
    config.packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    assert (
        config._resolve_quant_algo("model.layers.0.linear_attn.in_proj_qkvz") == "FP8"
    )
    assert config._resolve_quant_algo("model.layers.0.linear_attn.in_proj_ba") is None


def test_modelopt_mixed_precision_composes_gemma4_mappers():
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
    from vllm.model_executor.models.gemma4_mm import (
        Gemma4ForConditionalGeneration,
    )

    config = _mixed_precision_config(
        {
            "model.language_model.layers.0.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
            "model.language_model.layers.1.moe.experts.gate_up_proj": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
        }
    )

    config.apply_vllm_mapper(
        Gemma4ForConditionalGeneration.hf_to_vllm_mapper.get_unstacked_mapper()
    )
    config.apply_vllm_mapper(Gemma4ForCausalLM.hf_to_vllm_mapper.get_unstacked_mapper())

    expected_prefix = "language_model.model.layers.0.moe.experts"
    assert set(config.quantized_layers) == {
        expected_prefix,
        "language_model.model.layers.1.moe.gate_up_proj",
    }
    assert config._resolve_quant_algo(expected_prefix) == "NVFP4"


def test_modelopt_mixed_precision_infers_fused_gate_up_projection():
    from vllm.model_executor.layers.linear import LinearBase

    config = _mixed_precision_config(
        {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"},
            "model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4"},
        }
    )

    fake_layer = MagicMock(spec=LinearBase)
    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        method = config.get_quant_method(fake_layer, "model.layers.0.mlp.gate_up_proj")

    assert isinstance(method, ModelOptNvFp4LinearMethod)


@pytest.mark.parametrize(
    ("quantized_prefix", "missing_prefix"),
    [
        ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj"),
        ("model.layers.0.self_attn.o_proj", "model.layers.0.self_attn.qkv_proj"),
    ],
)
def test_modelopt_mixed_precision_does_not_infer_missing_sibling_linear(
    quantized_prefix, missing_prefix
):
    from vllm.model_executor.layers.linear import LinearBase

    config = _mixed_precision_config(
        {
            quantized_prefix: {"quant_algo": "NVFP4"},
        }
    )

    fake_layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(fake_layer, missing_prefix)

    assert isinstance(method, UnquantizedLinearMethod)


def test_vocab_parallel_embedding_weight_loader_accepts_scalar_scale():
    holder = Mock()
    scale = torch.nn.Parameter(torch.empty(1))
    loaded_scale = torch.tensor(2.0)

    VocabParallelEmbedding.weight_loader(holder, scale, loaded_scale)

    assert torch.equal(scale, loaded_scale.reshape(1))


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_checkpoint_setup(default_vllm_config, vllm_runner):
    """Test ModelOpt FP8 checkpoint loading and structure validation."""
    # TODO: provide a small publicly available test checkpoint
    model_path = (
        "/home/scratch.omniml_data_1/zhiyu/ckpts/test_ckpts/"
        "TinyLlama-1.1B-Chat-v1.0-fp8-0710"
    )

    # Skip test if checkpoint doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(
            f"Test checkpoint not found at {model_path}. "
            "This test requires a local ModelOpt FP8 checkpoint."
        )

    # Set model config as model_config.dtype is required in ModelOptFp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    with vllm_runner(model_path, quantization="modelopt", enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            # Check that ModelOpt quantization method is properly applied
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8LinearMethod,
            )

            assert isinstance(qkv_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(gate_up_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptFp8LinearMethod)

            # Check weight dtype is FP8
            assert qkv_proj.weight.dtype == torch.float8_e4m3fn
            assert o_proj.weight.dtype == torch.float8_e4m3fn
            assert gate_up_proj.weight.dtype == torch.float8_e4m3fn
            assert down_proj.weight.dtype == torch.float8_e4m3fn

            # Check scales are present and have correct dtype
            assert hasattr(qkv_proj, "weight_scale")
            assert hasattr(qkv_proj, "input_scale")
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.input_scale.dtype == torch.float32

            assert hasattr(o_proj, "weight_scale")
            assert hasattr(o_proj, "input_scale")
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.input_scale.dtype == torch.float32

            assert hasattr(gate_up_proj, "weight_scale")
            assert hasattr(gate_up_proj, "input_scale")
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.input_scale.dtype == torch.float32

            assert hasattr(down_proj, "weight_scale")
            assert hasattr(down_proj, "input_scale")
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.input_scale.dtype == torch.float32

        llm.apply_model(check_model)

        # Run a simple generation test to ensure the model works
        output = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        assert output
        print(f"ModelOpt FP8 output: {output}")


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_pc_pt_checkpoint_setup(default_vllm_config, vllm_runner):
    """Test ModelOpt FP8_PER_CHANNEL_PER_TOKEN checkpoint setup."""
    model_id = "CedricHwang/qwen2.5-0.5b-modelopt-fp8-pc-pt"
    model_path = _snapshot_download_or_skip(model_id)

    # Set model config as model_config.dtype is required in ModelOptFp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    with vllm_runner(model_path, quantization="modelopt", enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8PcPtLinearMethod,
            )

            assert isinstance(qkv_proj.quant_method, ModelOptFp8PcPtLinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptFp8PcPtLinearMethod)
            assert isinstance(gate_up_proj.quant_method, ModelOptFp8PcPtLinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptFp8PcPtLinearMethod)

            fp8_dtype = current_platform.fp8_dtype()
            assert qkv_proj.weight.dtype == fp8_dtype
            assert o_proj.weight.dtype == fp8_dtype
            assert gate_up_proj.weight.dtype == fp8_dtype
            assert down_proj.weight.dtype == fp8_dtype

            # Per-channel scales; activations are dynamically scaled per token.
            assert hasattr(qkv_proj, "weight_scale")
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.weight_scale.dim() == 1
            assert not hasattr(qkv_proj, "input_scale")

            assert hasattr(o_proj, "weight_scale")
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.weight_scale.dim() == 1
            assert not hasattr(o_proj, "input_scale")

            assert hasattr(gate_up_proj, "weight_scale")
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.weight_scale.dim() == 1
            assert not hasattr(gate_up_proj, "input_scale")

            assert hasattr(down_proj, "weight_scale")
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.weight_scale.dim() == 1
            assert not hasattr(down_proj, "input_scale")

        llm.apply_model(check_model)

        output = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        assert output
        print(f"ModelOpt FP8_PER_CHANNEL_PER_TOKEN output: {output}")


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_pb_wo_checkpoint_setup(default_vllm_config, vllm_runner):
    """Test ModelOpt FP8_PB_WO checkpoint setup."""
    model_id = "CedricHwang/qwen2.5-0.5b-modelopt-fp8-pb-wo"
    model_path = _snapshot_download_or_skip(model_id)

    # Set model config as model_config.dtype is required in ModelOptFp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    with vllm_runner(model_path, quantization="modelopt", enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8PbWoLinearMethod,
            )

            assert isinstance(qkv_proj.quant_method, ModelOptFp8PbWoLinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptFp8PbWoLinearMethod)
            assert isinstance(gate_up_proj.quant_method, ModelOptFp8PbWoLinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptFp8PbWoLinearMethod)

            assert qkv_proj.weight.dtype == torch.float8_e4m3fn
            assert o_proj.weight.dtype == torch.float8_e4m3fn
            assert gate_up_proj.weight.dtype == torch.float8_e4m3fn
            assert down_proj.weight.dtype == torch.float8_e4m3fn

            # Block scales; should be materialized as a 2D [out_blk, in_blk] tensor.
            assert hasattr(qkv_proj, "weight_scale")
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.weight_scale.dim() == 2

            assert hasattr(o_proj, "weight_scale")
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.weight_scale.dim() == 2

            assert hasattr(gate_up_proj, "weight_scale")
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.weight_scale.dim() == 2

            assert hasattr(down_proj, "weight_scale")
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.weight_scale.dim() == 2

        llm.apply_model(check_model)

        output = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        assert output
        print(f"ModelOpt FP8_PB_WO output: {output}")


def test_modelopt_nvfp4_config_dispatches_w4a4_method():
    """``quant_method="NVFP4"`` (W4A4 default) routes to the existing
    ``ModelOptNvFp4LinearMethod``."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
    )

    config = ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert config.LinearMethodCls is ModelOptNvFp4LinearMethod
    assert config.quant_method == "NVFP4"


def test_modelopt_nvfp4_config_dispatches_w4a16_method():
    """``quant_method="W4A16_NVFP4"`` routes to
    ``ModelOptNvFp4W4A16LinearMethod`` instead of the W4A4 sibling.

    Mirrors the FP8 dispatch precedent (``ModelOptFp8Config`` selects
    one of three FP8 LinearMethods on ``quant_method``); a regression
    here would mean a W4A16 NVFP4 checkpoint silently loaded under the
    W4A4 activation-quantization path.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
        ModelOptNvFp4W4A16LinearMethod,
    )

    config = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert config.LinearMethodCls is ModelOptNvFp4W4A16LinearMethod
    assert config.LinearMethodCls is not ModelOptNvFp4LinearMethod
    assert config.quant_method == "W4A16_NVFP4"


def test_modelopt_nvfp4_env_can_force_w4a4_checkpoint_to_w4a16(monkeypatch):
    """The research override changes only runtime activation arithmetic.

    W4A4-shaped checkpoints retain their packed weights/scales and are loaded
    by the W4A16 method, whose compatibility placeholder consumes and discards
    the otherwise-unused input scale.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4W4A16LinearMethod,
    )

    monkeypatch.setenv("AG2_VLLM_NVFP4_FORCE_W4A16", "1")
    config = ModelOptNvFp4Config._from_config(
        quant_method="NVFP4",
        kv_cache_quant_method=None,
        exclude_modules=[],
        original_config={
            "quantization": {
                "group_size": 16,
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
            }
        },
        group_size=16,
    )

    assert config.quant_method == "W4A16_NVFP4"
    assert config.LinearMethodCls is ModelOptNvFp4W4A16LinearMethod


@pytest.mark.parametrize(
    ("linear_backend", "kernel_cls"),
    [("auto", MarlinNvFp4LinearKernel), ("humming", HummingNvFp4LinearKernel)],
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_modelopt_w4a16_respects_linear_backend(linear_backend, kernel_cls):
    vllm_config = VllmConfig()
    vllm_config.kernel_config.linear_backend = linear_backend
    with set_current_vllm_config(vllm_config):
        method = ModelOptNvFp4W4A16LinearMethod(
            ModelOptNvFp4Config(quant_method="W4A16_NVFP4")
        )
    assert isinstance(method.kernel, kernel_cls)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_modelopt_w4a16_batch_invariant_dispatches_humming(monkeypatch):
    """The narrow W4A16 flag must not alter attention or select W4A4 CUTLASS."""
    from vllm import envs

    envs.disable_envs_cache()
    monkeypatch.setenv("AG2_VLLM_NVFP4_BATCH_INVARIANT", "1")
    monkeypatch.setenv("AG2_VLLM_NVFP4_MARLIN_FIXED_SCHEDULE", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        method = ModelOptNvFp4W4A16LinearMethod(
            ModelOptNvFp4Config(quant_method="W4A16_NVFP4")
        )
    assert isinstance(method.kernel, HummingNvFp4LinearKernel)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_modelopt_w4a16_batch_invariant_dispatches_fixed_marlin(monkeypatch):
    """The fixed-schedule POC keeps W4A16 and selects Marlin explicitly."""
    from vllm import envs

    envs.disable_envs_cache()
    monkeypatch.setenv("AG2_VLLM_NVFP4_BATCH_INVARIANT", "1")
    monkeypatch.setenv("AG2_VLLM_NVFP4_MARLIN_FIXED_SCHEDULE", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        method = ModelOptNvFp4W4A16LinearMethod(
            ModelOptNvFp4Config(quant_method="W4A16_NVFP4")
        )
    assert isinstance(method.kernel, MarlinNvFp4LinearKernel)


@pytest.mark.parametrize(
    "quant_method, expected_use_a16, act_key_is_none",
    [
        ("NVFP4", False, False),  # W4A4 default
        ("W4A16_NVFP4", True, True),  # native W4A16 ckpt
    ],
)
def test_modelopt_nvfp4_moe_dispatches_to_marlin_when_w4a16(
    quant_method, expected_use_a16, act_key_is_none
):
    """``ModelOptNvFp4FusedMoE``: when the ckpt's ``quant_method`` is
    ``W4A16_NVFP4``, the MoE class must pass ``activation_key=None`` to
    ``select_nvfp4_moe_backend``. That filters out every W4A4 backend
    (their ``_supports_quant_scheme`` requires
    ``(kNvfp4Static, kNvfp4Dynamic)`` exactly); Marlin survives because
    it only checks ``weight_key``. A regression here would mean a W4A16
    ckpt silently went to the cutlass W4A4 path.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )

    config = ModelOptNvFp4Config(
        quant_method=quant_method,
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
    )

    mock_select = MagicMock(return_value=(MagicMock(), MagicMock()))
    with (
        patch(
            "vllm.model_executor.layers.quantization.modelopt.select_nvfp4_moe_backend",
            mock_select,
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt."
            "is_global_sf_supported_for_nvfp4_backend",
            return_value=False,
        ),
    ):
        moe = ModelOptNvFp4FusedMoE(config, MagicMock())

    assert moe.use_a16 is expected_use_a16
    _, kwargs = mock_select.call_args
    assert kwargs["weight_key"] is kNvfp4Static
    if act_key_is_none:
        assert kwargs["activation_key"] is None
    else:
        assert kwargs["activation_key"] is kNvfp4Dynamic


@pytest.mark.parametrize(
    "per_layer_algo, expected_linear_cls_name",
    [
        ("NVFP4", "ModelOptNvFp4LinearMethod"),
        ("W4A16_NVFP4", "ModelOptNvFp4W4A16LinearMethod"),
    ],
)
def test_modelopt_mixed_precision_dispatches_w4a16_layer(
    per_layer_algo, expected_linear_cls_name
):
    """``ModelOptMixedPrecisionConfig.get_quant_method`` must route a Linear
    layer to the right LinearMethod based on its per-layer ``quant_algo``
    entry in ``quantized_layers``. Verifies the new ``W4A16_NVFP4`` branch
    coexists with the existing ``NVFP4`` branch without regression. A
    regression here would mean a W4A16 layer in a mixed-precision ckpt
    silently fell through to ``UnquantizedLinearMethod``.

    NOTE: FP8 dispatch (the third branch of get_quant_method) is not
    covered here because ``ModelOptFp8LinearMethod.__init__`` reads
    ``get_current_vllm_config().model_config.dtype``, which requires a
    fully constructed ``ModelConfig`` (real model path). FP8 routing in
    mixed-precision is exercised by the existing integration tests
    above that use the ``vllm_runner`` fixture (e.g.
    ``test_modelopt_fp8_checkpoint_setup``). Our PR doesn't change the
    FP8 branch, so this isn't a coverage gap.
    """
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import modelopt as m

    if (
        expected_linear_cls_name == "ModelOptNvFp4W4A16LinearMethod"
        and current_platform.is_rocm()
    ):
        pytest.skip("ModelOptNvFp4W4A16LinearMethod is not supported with rocm")

    hf_quant_config: dict[str, Any] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "group_size": 16,
            "quantized_layers": {
                "model.layers.0.fake_proj": {"quant_algo": per_layer_algo},
            },
        }
    }
    config = m.ModelOptMixedPrecisionConfig.from_config(hf_quant_config)

    fake_layer = MagicMock(spec=LinearBase)
    method = config.get_quant_method(fake_layer, "model.layers.0.fake_proj")

    expected_cls = getattr(m, expected_linear_cls_name)
    assert isinstance(method, expected_cls), (
        f"Expected {expected_linear_cls_name}, got {type(method).__name__}"
    )


def test_modelopt_mixed_precision_builds_w4a16_sibling_config():
    """Sanity: ``ModelOptMixedPrecisionConfig._from_config`` builds **two**
    NVFP4 sub-configs — one for W4A4 (default) and one tagged
    ``quant_method='W4A16_NVFP4'`` — so per-layer dispatch can hand
    Marlin-bound layers the right config without re-instantiating it on
    every call.
    """
    from vllm.model_executor.layers.quantization import modelopt as m

    hf_quant_config: dict[str, Any] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "group_size": 16,
            "quantized_layers": {
                "model.layers.0.a": {"quant_algo": "NVFP4"},
                "model.layers.0.b": {"quant_algo": "W4A16_NVFP4"},
            },
        }
    }
    config = m.ModelOptMixedPrecisionConfig.from_config(hf_quant_config)

    assert config.nvfp4_config.quant_method == "NVFP4"
    assert config.nvfp4_config.LinearMethodCls is m.ModelOptNvFp4LinearMethod
    assert config.w4a16_nvfp4_config.quant_method == "W4A16_NVFP4"
    assert config.w4a16_nvfp4_config.LinearMethodCls is m.ModelOptNvFp4W4A16LinearMethod
