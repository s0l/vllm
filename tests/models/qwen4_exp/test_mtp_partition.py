# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP source coverage, TP loader tails and isolated draft configuration."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from vllm.model_executor.layers.fused_moe.oracle.fp8 import refine_fp8_moe_block_shape
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import PaddedMergedColumnParallelLinear
from vllm.model_executor.models.qwen3_5_mtp import _mtp_fc_padded_output_size
from vllm.models.qwen4_exp.nvidia.mtp_partition import (
    MTPExpertLoader,
    partition_native_mtp,
)


def loader(tp=3, experts=2):
    return MTPExpertLoader(
        hidden=256, width=640, tp=tp, block=128, layers=1, experts=experts
    )


def weights(experts=2):
    result = []
    for expert in range(experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (256, 640) if projection == "down_proj" else (640, 256)
            value = (
                (torch.arange(shape[0] * shape[1]).reshape(shape) + expert) % 17 - 8
            ).to(torch.float8_e4m3fn)
            scale = (
                torch.arange(shape[0] // 128 * (shape[1] // 128)).reshape(
                    shape[0] // 128, shape[1] // 128
                )
                + 1
            ).bfloat16()
            prefix = f"layers.0.mlp.experts.{expert}.{projection}"
            result += [
                (prefix + ".weight", value),
                (prefix + ".weight_scale_inv", scale),
            ]
    return result


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 5, 8, 16])
def test_fp8_complete_source_padding_survives_real_tp_loader(tp):
    owner = loader(tp)
    source = weights()
    # Independent AutoWeightsLoader visits must accumulate one complete stream.
    transformed = dict(owner.transform(source[:3])) | dict(owner.transform(source[3:]))
    owner.finish()
    routed = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed)
    routed.moe_config = SimpleNamespace(
        is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=tp),
        intermediate_size_per_partition=owner.physical // tp,
        hidden_dim=256,
        tp_size=tp,
    )
    refined = refine_fp8_moe_block_shape(routed.moe_config, [128, 128])
    factor = 128 // refined[0] if refined else 1
    routed.quant_config = None
    routed.quant_method = SimpleNamespace(
        weight_scale_refine=(factor, factor) if refined else None
    )
    routed.expert_map_manager = SimpleNamespace(map_global_to_local=lambda i: i)
    for name, original in source:
        padded = transformed[name]
        scale, down, up = (
            name.endswith("_inv"),
            ".down_proj." in name,
            ".up_proj." in name,
        )
        unit = 128 // factor if scale else 1
        local, hidden = owner.physical // tp // unit, 256 // unit
        gathered = []
        for rank in range(tp):
            routed.moe_config.tp_rank = rank
            shape = (1, hidden, local) if down else (1, 2 * local, hidden)
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=padded.dtype), requires_grad=False
            )
            if scale:
                param.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
            shard_id = "w2" if down else "w3" if up else "w1"
            weight_name = ("w2" if down else "w13") + (
                "_weight_scale_inv" if scale else "_weight"
            )
            routed.weight_loader(param, padded, weight_name, shard_id, 0)
            target = param.data[0]
            if not down:
                target = target[local:] if up else target[:local]
            gathered.append(target)
        rebuilt = torch.cat(gathered, dim=int(down))
        logical = 640 // unit
        actual = rebuilt[:, :logical] if down else rebuilt[:logical]
        tail = rebuilt[:, logical:] if down else rebuilt[logical:]
        expected = (
            original.repeat_interleave(factor, 0).repeat_interleave(factor, 1)
            if scale
            else original
        )
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert (tail.float() == (1 if scale else 0)).all()
    with pytest.raises(RuntimeError, match="new worker"):
        list(owner.transform(source))


@pytest.mark.parametrize(
    "failure",
    ["shape", "dtype", "field", "duplicate", "nan", "scale", "missing", "index"],
)
def test_malformed_mtp_stream_closes_epoch_and_fresh_source_recovers(failure):
    source = weights()
    if failure == "shape":
        source[0] = source[0][0], source[0][1][:-1]
    elif failure == "dtype":
        source[0] = source[0][0], source[0][1].bfloat16()
    elif failure == "field":
        source[0] = source[0][0] + "_unexpected", source[0][1]
    elif failure == "duplicate":
        source.append(source[0])
    elif failure == "nan":
        source[0][1][0, 0] = float("nan")
    elif failure == "scale":
        source[1][1][0, 0] = 0
    elif failure == "missing":
        source.pop()
    elif failure == "index":
        source[0] = source[0][0].replace("layers.0", "layers.1"), source[0][1]
    owner = loader()
    with pytest.raises(ValueError):
        list(owner.transform(source))
        owner.finish()
    assert owner.closed
    with pytest.raises(RuntimeError):
        list(owner.transform(weights()))
    fresh = loader()
    list(fresh.transform(weights()))
    fresh.finish()


def test_cancelled_partial_mtp_stream_cannot_resume_as_another_load():
    owner = loader()
    stream = owner.transform(weights())
    next(stream)
    stream.close()
    assert owner.closed
    with pytest.raises(RuntimeError, match="new worker"):
        list(owner.transform(weights()))
    fresh = loader()
    list(fresh.transform(weights()))
    fresh.finish()


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("algorithm", ["FP8_BLOCK_SCALES", "FP8_PB_WO"])
def test_draft_physical_geometry_does_not_mutate_target_or_source(tp, algorithm):
    text = PretrainedConfig(
        hidden_size=2560, moe_intermediate_size=640, shared_expert_intermediate_size=640
    )
    draft = SimpleNamespace(
        additional_config={"flashnext_native_experts": {}},
        model_config=SimpleNamespace(hf_config=text, hf_text_config=text),
        kernel_config=SimpleNamespace(moe_backend="cutlass"),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        compilation_config=object(),
        quant_config=SimpleNamespace(
            quantized_layers={
                "mtp.layers.48.mlp.experts": {
                    "quant_algo": algorithm,
                    "group_size": 128,
                }
            }
        ),
    )
    text.num_experts = 512
    physical, owner = partition_native_mtp(draft, start_layer=48, layers=1)
    assert physical.model_config.hf_text_config is not text
    assert (
        physical.model_config.hf_config.get_text_config()
        is physical.model_config.hf_text_config
    )
    assert (
        text.moe_intermediate_size == 640
        and draft.kernel_config.moe_backend == "cutlass"
    )
    assert physical.kernel_config.moe_backend == "triton"
    assert physical.model_config.hf_text_config.moe_intermediate_size == owner.physical
    assert physical.model_config.hf_text_config.shared_expert_intermediate_size == 640
    assert physical.compilation_config is draft.compilation_config
    assert owner.physical >= 640 and owner.physical % 128 == 0
    assert owner.physical % (tp * 32) == 0
    assert owner.physical == {1: 640, 2: 640, 3: 768, 4: 640, 5: 640, 8: 768}[tp]
    draft.quant_config.quantized_layers["mtp.layers.48.mlp.experts"]["quant_algo"] = (
        "NVFP4"
    )
    with pytest.raises(ValueError, match="FP8 block128"):
        partition_native_mtp(draft, start_layer=48, layers=1)


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 5, 8])
def test_padded_fc_source_loader_preserves_2d_and_hc_inputs(monkeypatch, tp):
    hidden = 2560
    physical = _mtp_fc_padded_output_size(hidden, tp)
    weight = (torch.arange(hidden * hidden).reshape(hidden, hidden) % 11 - 5).bfloat16()
    x = (torch.arange(2 * 4 * hidden).reshape(2, 4, hidden) % 3 - 1).bfloat16()
    results: list[list[torch.Tensor]] = [[], []]
    for rank in range(tp):
        for module in (
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.parameter",
        ):
            monkeypatch.setattr(
                module + ".get_tensor_model_parallel_world_size", lambda: tp
            )
            monkeypatch.setattr(
                module + ".get_tensor_model_parallel_rank", lambda rank=rank: rank
            )
        fc = PaddedMergedColumnParallelLinear(
            hidden,
            [hidden],
            [physical],
            bias=False,
            return_bias=False,
            params_dtype=torch.bfloat16,
        )
        fc.weight.weight_loader(fc.weight, weight)
        for index, value in enumerate((x[:, 0], x)):
            results[index].append(torch.nn.functional.linear(value, fc.weight))
    for parts, value in zip(results, (x[:, 0], x)):
        result = torch.cat(parts, dim=-1)
        assert not result[..., hidden:].count_nonzero()
        assert torch.equal(
            result[..., :hidden], torch.nn.functional.linear(value, weight)
        )


def test_actual_mtp_forward_removes_fc_padding_before_hc_continuation(monkeypatch):
    from vllm.models.qwen4_exp.nvidia import mtp

    model = mtp.Qwen4ExpMultiTokenPredictor.__new__(mtp.Qwen4ExpMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    model.hc_count, model.hidden_size, model.num_mtp_layers = 4, 16, 1
    model.pre_fc_norm_embedding = model.pre_fc_norm_hidden = torch.nn.Identity()
    model.fc_embedding = model.fc_hidden = lambda x: torch.nn.functional.pad(
        x * 2, (0, 8), value=1234
    )

    def layer(**kwargs):
        assert kwargs["hidden_states"].shape == (2, 64)
        assert kwargs["hidden_states"].is_contiguous()
        assert kwargs["prev_block_output"].shape == (2, 16)
        assert kwargs["prev_block_output"].is_contiguous()
        return kwargs["hidden_states"], kwargs["prev_block_output"], None

    model.layers = [layer]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda h, b, i: (h, b, None)
    )
    monkeypatch.setattr(
        mtp,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    hidden, embedding = torch.arange(128).reshape(2, 64).float(), torch.ones(2, 16)
    sample, multi = model(None, torch.arange(2), hidden, inputs_embeds=embedding)
    assert torch.equal(sample, embedding * 2) and torch.equal(multi, hidden * 2)
