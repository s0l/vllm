# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source identity, admission and bounded residency before native execution."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm.models.qwen4_exp.nvidia.expert_offload_source import (
    NativeExpertStore,
    prepare,
    split_bundle,
)
from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry


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
        geometry=NVFP4ExpertGeometry(32, 16, tp=3, local_alignment=16),
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


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize("hidden,width", [(2560, 640), (384, 656)])
def test_tp_partition_preserves_source_tail_scales_and_exact_mapped_bytes(
    tp, hidden, width
):
    from vllm.v1.core.elastic_expert import NativeExpertBudget

    geometry = NVFP4ExpertGeometry(hidden, width, tp)
    rng = np.random.default_rng(10277)
    raw = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        m, k = (hidden, width) if projection == "down_proj" else (width, hidden)
        raw[projection + ".weight"] = rng.integers(0, 256, (m, k // 2), dtype=np.uint8)
        raw[projection + ".weight_scale"] = rng.integers(
            32, 64, (m, k // 16), dtype=np.uint8
        )
        raw[projection + ".weight_scale_2"] = np.asarray(0.5, dtype=np.float32)
        raw[projection + ".input_scale"] = np.asarray(1.5, dtype=np.float32)
    pristine = {k: v.copy() for k, v in raw.items()}
    shards = split_bundle(
        raw, hidden=hidden, width=width, physical=geometry.physical, tp=tp
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        axis = 1 if projection == "down_proj" else 0
        for field, packing, pad in (("weight", 2, 0), ("weight_scale", 16, 56)):
            name = projection + "." + field
            joined = np.concatenate([s[name] for s in shards], axis=axis)
            logical = width // packing if axis else width
            source, padding = np.split(joined, [logical], axis=axis)
            np.testing.assert_array_equal(source, raw[name])
            assert np.all(padding == pad)
        for field in ("weight_scale_2", "input_scale"):
            name = projection + "." + field
            assert all(np.array_equal(s[name], raw[name]) for s in shards)
    assert all(np.array_equal(pristine[k], raw[k]) for k in raw)
    # A dropped/shifted final source row cannot pass the reconstruction oracle.
    last = np.concatenate([s["gate_proj.weight"] for s in shards])[:width].copy()
    last[-1, 0] ^= 1
    assert not np.array_equal(last, raw["gate_proj.weight"])
    budget = NativeExpertBudget(geometry, 48, 512, max_hot_rows=128, staging=32)
    for rank, shard in enumerate(shards):
        assert all(
            np.array_equal(a, b)
            for a, b in zip(
                shard.values(),
                split_bundle(
                    raw,
                    hidden=hidden,
                    width=width,
                    physical=geometry.physical,
                    tp=tp,
                    selected_rank=rank,
                )[0].values(),
            )
        )
        prepared = prepare(shard)
        arrays = [
            prepared[k]
            for k in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
        ]
        assert tuple(a.nbytes for a in arrays) == budget.strides
        for hot in (0, 1, 31, 32, 127, 128):
            sizes = [(hot + 32) * a.nbytes for a in arrays] + [(128 + 32) * 24]
            q = 2 << 20
            expected = sum((size + q - 1) // q * q for size in sizes)
            assert budget.mapped_bytes(hot) == expected
    if tp == 16:
        assert all(not s["down_proj.weight"].any() for s in shards[11:])


@pytest.mark.parametrize("tp", [0, -1, True, 1.5])
def test_invalid_tp_rejected_before_checkpoint_access(tmp_path, tp):
    with pytest.raises(ValueError, match="partition geometry"):
        NativeExpertStore(
            tmp_path,
            0,
            0,
            layers=1,
            experts=1,
            geometry=NVFP4ExpertGeometry(2560, 640, tp),
        )


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("mismatch", [None, "group", "rank", "config"])
def test_provider_admits_configured_tp_and_rejects_stale_partition_before_allocation(
    monkeypatch, tp, mismatch
):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import expert_offload_provider as module

    source = SimpleNamespace(
        geometry=NVFP4ExpertGeometry(2560, 640, tp),
        tp=tp,
        rank=0,
        hidden=2560,
        experts=512,
    )
    group = SimpleNamespace(device_group=object())
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp + (mismatch == "config")
        )
    )
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda g: tp + (mismatch == "group")
    )
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda g: int(mismatch == "rank")
    )

    class Admitted(Exception):
        pass

    def allocation(*args):
        raise Admitted

    monkeypatch.setattr(module, "NativeBankCoordinator", allocation)
    with pytest.raises(ValueError if mismatch else Admitted):
        module.NativeExpertProvider(SimpleNamespace(source=source), group, cfg, topk=10)


@pytest.mark.parametrize("default_device", ["cpu", "meta"])
def test_provider_pinned_staging_owns_cpu_under_loader_device(
    monkeypatch, default_device
):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import expert_offload_provider as module

    source = SimpleNamespace(
        geometry=NVFP4ExpertGeometry(2560, 640, 2),
        tp=2,
        rank=0,
        hidden=2560,
        experts=512,
    )
    bank = SimpleNamespace(source=source, device=torch.device("meta"), staging=32)
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2))
    group = SimpleNamespace(device_group=object())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda g: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda g: 0)
    monkeypatch.setattr(module, "NativeBankCoordinator", lambda *args: object())
    original_empty = torch.empty
    pinned = []

    def allocation(*args, **kwargs):
        if kwargs.get("pin_memory"):
            device = torch.device(kwargs.get("device", torch.get_default_device()))
            assert device.type == "cpu", "pinned storage must own CPU"
            pinned.append((args, kwargs["dtype"]))
            # CPU-only control checks the requested pinning contract without
            # needing a CUDA driver. The full loader proves real pinned storage.
            kwargs["pin_memory"] = False
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocation)
    with torch.device(default_device):
        provider = module.NativeExpertProvider(
            bank, group, cfg, topk=10, max_tokens=4, max_lanes=32
        )
        with pytest.raises(AssertionError, match="pinned storage must own CPU"):
            torch.empty(1, device="meta", pin_memory=True)
    assert len(pinned) == 3
    assert provider.x.device.type == "meta"
    for tensor, shape, dtype in (
        (provider.host_lanes, (32,), torch.int64),
        (provider.host_local_ids, (32, 1), torch.int32),
        (provider.host_group_rows, (32,), torch.int32),
    ):
        assert tensor.device.type == "cpu"
        assert tensor.shape == shape and tensor.dtype == dtype


def test_native_callback_defers_unproduced_capture_routes_until_replay(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import expert_offload_provider as module

    calls = []

    def run(layer, hidden, weights, ids, output, *, is_padding=None):
        module.plan(
            ids.numpy(),
            3,
            3,
            is_padding=None if is_padding is None else is_padding.numpy(),
        )
        calls.append(layer)
        output.copy_(hidden)

    owner = SimpleNamespace(provider=SimpleNamespace(run=run), layer_id=2)
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"owner": owner}),
    )
    x, weights = torch.ones(1, 4), torch.ones(1, 2)
    ids, output = torch.zeros(1, 2, dtype=torch.int32), torch.ones_like(x)
    capture_type = module.BreakableCUDAGraphCapture
    # add_eager has closed its segment while retaining capture TLS.
    with monkeypatch.context() as capture:
        capture.setattr(
            capture_type._tls,
            "active",
            SimpleNamespace(_capturing=False),
            raising=False,
        )
        module._native_experts(x, weights, ids, output, "owner")
        assert not calls and not output.count_nonzero()
    with pytest.raises(ValueError, match="duplicate expert"):
        module._native_experts(x, weights, ids, output, "owner")
    assert not calls and not output.count_nonzero()
    ids[0, 1] = 1
    module._native_experts(x, weights, ids, output, "owner")
    assert calls == [2] and torch.equal(output, x)
    padding = torch.ones(1, dtype=torch.bool)
    ids.fill_(-1)
    module._native_experts(x, weights, ids, output, "owner", padding)
    assert calls == [2, 2]
    padding.zero_()
    with pytest.raises(ValueError, match="expert outside"):
        module._native_experts(x, weights, ids, output, "owner", padding)
