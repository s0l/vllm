# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned Q/gate row loading and global K/V head identity before attention."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.qsa_qkv import QSAOwnedQKVLinear


@pytest.fixture(autouse=True)
def parameter_tp_metadata(monkeypatch):
    # CPU source-loader control only; real rank dispatch is a separate GPU gate.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )


def test_tp3_q_gate_heads_cover_source_once_and_kv_are_replicated():
    hidden, heads, kv_heads, dim = 16, 24, 2, 8
    q = (
        torch.arange(heads * 2 * dim * hidden, dtype=torch.float32)
        .reshape(-1, hidden)
        .bfloat16()
    )
    k = torch.full((kv_heads * dim, hidden), -3, dtype=torch.bfloat16)
    v = torch.full_like(k, 7)
    local_q = []
    for rank in range(3):
        layer = QSAOwnedQKVLinear(hidden, heads, kv_heads, dim, tp_size=3, tp_rank=rank)
        for name, tensor in [("v", v), ("q", q), ("k", k)]:
            layer.weight.weight_loader(layer.weight, tensor, name)
        a, b, c = layer.weight.split(
            [heads // 3 * 2 * dim, kv_heads * dim, kv_heads * dim]
        )
        local_q.append(a)
        assert torch.equal(b, k) and torch.equal(c, v)
        assert (
            a.reshape(8, 2, dim, hidden)[0, 0, 0, 0]
            == q.reshape(heads, 2, dim, hidden)[rank * 8, 0, 0, 0]
        )
    assert torch.equal(torch.cat(local_q), q)


def test_projection_load_rejects_incomplete_duplicate_and_invalid_shards():
    layer = QSAOwnedQKVLinear(16, 24, 2, 8, tp_size=3, tp_rank=1)
    q = torch.ones(384, 16, dtype=torch.bfloat16)
    before = layer.weight.detach().clone()
    for name, value in [(None, q), ("q", q.float()), ("k", q)]:
        with pytest.raises(ValueError):
            layer.weight.weight_loader(layer.weight, value, name)
    # Check the write boundary via exact bytes, including uninitialized NaNs.
    assert torch.equal(layer.weight.view(torch.uint8), before.view(torch.uint8))
    layer.weight.weight_loader(layer.weight, q, "q")
    with pytest.raises(ValueError, match="duplicate"):
        layer.weight.weight_loader(layer.weight, q, "q")
    with pytest.raises(RuntimeError, match="incomplete"):
        layer(torch.ones(1, 16, dtype=torch.bfloat16))
    for name in ("k", "v"):
        layer.weight.weight_loader(
            layer.weight, torch.ones(16, 16, dtype=torch.bfloat16), name
        )
    assert layer.loaded_shards == {"q", "k", "v"}


def test_actual_auto_loader_preserves_qkv_shard_metadata():
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpModel

    parent = torch.nn.Module()
    parent.self_attn = torch.nn.Module()
    layer = QSAOwnedQKVLinear(16, 24, 2, 8, tp_size=3, tp_rank=1)
    parent.self_attn.qkv_proj = layer
    tensors = {
        "q": torch.arange(384 * 16).reshape(384, 16).bfloat16(),
        "k": torch.full((16, 16), -2, dtype=torch.bfloat16),
        "v": torch.full((16, 16), 3, dtype=torch.bfloat16),
    }
    loaded = AutoWeightsLoader(parent).load_weights(
        [(f"self_attn.{name}_proj.weight", value) for name, value in tensors.items()],
        mapper=Qwen4ExpModel.hf_to_vllm_mapper,
    )
    assert loaded == {"self_attn.qkv_proj.weight"}
    assert layer.loaded_shards == {"q", "k", "v"}
    assert torch.equal(
        layer.weight, torch.cat([tensors["q"][128:256], tensors["k"], tensors["v"]])
    )
