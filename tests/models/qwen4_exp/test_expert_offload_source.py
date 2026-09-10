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


def test_model_expert_loader_accounts_metadata_without_resident_parameters():
    from types import SimpleNamespace

    import torch

    from vllm.models.qwen4_exp.nvidia.expert_offload_moe import NativeOffloadedExperts

    source = SimpleNamespace(
        experts=2,
        entries={
            (3, e): {"gate_proj.weight": ("fixture", 0, 8, (2, 4), "U8")}
            for e in range(2)
        },
        files={"fixture": "pinned"},
        check_files=lambda paths: None,
    )
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    provider = SimpleNamespace(bank=SimpleNamespace(source=source))
    owner = NativeOffloadedExperts(provider, cfg, 3, "model.layers.3.mlp.experts")
    assert not list(owner.parameters()) and not list(owner.buffers())
    with pytest.raises(ValueError, match="missed source"):
        owner.finish_load()
    with pytest.raises(ValueError, match="metadata mismatch"):
        owner.load_weights([("0.gate_proj.weight", torch.empty(2, 4))])
    assert not owner.seen
    for e in range(2):
        owner.load_weights(
            [
                (
                    f"{e}.gate_proj.weight",
                    torch.empty(2, 4, dtype=torch.uint8, device="meta"),
                )
            ]
        )
    owner.finish_load()
    assert owner.loaded
    with pytest.raises(RuntimeError, match="reload requires"):
        owner.finish_load()
    with pytest.raises(RuntimeError, match="reload requires"):
        owner.load_weights(
            [("0.gate_proj.weight", torch.empty(2, 4, dtype=torch.uint8))]
        )


def test_native_target_opt_in_never_selects_mtp_layout():
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia.expert_offload_moe import native_experts_enabled

    cfg = SimpleNamespace(additional_config={"flashnext_native_experts": {}})
    assert native_experts_enabled(cfg, "model.language_model.layers.0.mlp")
    assert not native_experts_enabled(cfg, "mtp.layers.0.mlp")
    assert not native_experts_enabled(cfg, "model.mtp.layers.0.mlp")
    cfg.additional_config = {}
    assert not native_experts_enabled(cfg, "model.language_model.layers.0.mlp")


def test_native_model_state_switches_dummy_and_real_without_ple(
    monkeypatch,
):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    calls = []
    owner = Qwen4ExpModelState.__new__(Qwen4ExpModelState)
    owner.uses_ngram_embedding = False
    owner._native_providers = (
        SimpleNamespace(prepare_execution=lambda *, dummy: calls.append(dummy)),
    )
    monkeypatch.setattr(MambaHybridModelState, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(MambaHybridModelState, "prepare_dummy_inputs", lambda *args: {})
    owner.prepare_dummy_inputs(1, 1)
    owner.prepare_inputs(None, None)
    owner.prepare_runtime_dummy_inputs(None, None)
    owner.prepare_inputs(None, None)
    assert calls == [True, False, True, False]


def test_native_model_does_not_silently_consume_dense_owner_prequant_rows():
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration

    with pytest.raises(ValueError, match="owner-prequant"):
        Qwen4ExpForConditionalGeneration.forward(
            torch.nn.Module(), None, None, ready_rows=torch.tensor([1])
        )
