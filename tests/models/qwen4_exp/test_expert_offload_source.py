# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source identity, admission and bounded residency before native execution."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm.models.qwen4_exp.nvidia.expert_offload_source import NativeExpertStore


def write_source(path, scale=1.0):
    tensors = {}
    for layer in range(2):
        for expert in range(3):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                m, k = (32, 16) if projection == "down_proj" else (16, 32)
                tensors[f"{prefix}.{projection}.weight"] = torch.full(
                    (m, k // 2), layer * 3 + expert + 1, dtype=torch.uint8
                )
                tensors[f"{prefix}.{projection}.weight_scale"] = torch.full(
                    (m, k // 16), 56, dtype=torch.uint8
                ).view(torch.float8_e4m3fn)
                for name in ("weight_scale_2", "input_scale"):
                    tensors[f"{prefix}.{projection}.{name}"] = torch.tensor(scale)
    save_file(tensors, str(path / "weights.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")})
    )


def store(path, budget=4096, rank=0):
    return NativeExpertStore(
        path,
        rank,
        budget,
        layers=2,
        experts=3,
        hidden=32,
        width=16,
        physical=48,
    )


def test_source_cache_identity_eviction_padding_and_invalid_id_recovery(tmp_path):
    write_source(tmp_path)
    source = store(tmp_path)
    first = source.get(0, 1)
    assert len(first) == 10
    assert np.all(first["w13_weight"] == 2)
    assert all(not a.flags.writeable for a in first.values())
    read = source.read_bytes
    assert source.get(0, 1) is first and source.read_bytes == read
    with pytest.raises(ValueError, match="unknown checkpoint expert"):
        source.get(4, 0)
    assert source.get(0, 1) is first
    for layer in range(2):
        for expert in range(3):
            source.get(layer, expert)
    assert source.used <= source.limit and source.peak <= source.limit
    assert len(source.cache) < 6
    source.get(0, 1)
    assert source.read_bytes > read
    # Ranks outside the logical width retain a finite, exactly zero FP4 tail.
    padded = store(tmp_path, rank=2).get(0, 1)
    assert not np.any(padded["w13_weight"])
    assert not np.any(padded["w2_weight"])
    source.close()
    with pytest.raises(RuntimeError, match="closed"):
        source.get(0, 1)


def test_changed_source_rejects_cached_rows_and_new_epoch_recovers(tmp_path):
    write_source(tmp_path)
    source = store(tmp_path)
    before = source.get(0, 0)["w2_weight_scale_2"].copy()
    write_source(tmp_path, scale=2.0)
    with pytest.raises(OSError, match="source identity changed"):
        source.get(0, 0)
    assert source.closed and not source.cache and source.used == 0
    assert not np.array_equal(store(tmp_path).get(0, 0)["w2_weight_scale_2"], before)


def test_incomplete_inventory_and_nonfinite_scale_fail_closed(tmp_path):
    write_source(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"].pop(next(iter(data["weight_map"])))
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete native expert"):
        store(tmp_path)
    write_source(tmp_path, scale=float("nan"))
    source = store(tmp_path)
    with pytest.raises(ValueError, match="invalid global scale"):
        source.get(0, 0)
    assert source.closed and source.used == 0
