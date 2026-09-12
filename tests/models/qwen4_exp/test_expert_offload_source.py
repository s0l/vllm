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


def store(path, budget=4096, rank=0, **kwargs):
    return NativeExpertStore(
        path,
        rank,
        budget,
        layers=2,
        experts=3,
        geometry=NVFP4ExpertGeometry(32, 16, tp=3, local_alignment=16),
        **kwargs,
    )


def test_source_cache_identity_eviction_padding_and_invalid_id_recovery(tmp_path):
    write_source(tmp_path)
    source = store(tmp_path, cache_policy="lru")
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


def test_frequency_admission_survives_scans_and_adapts_to_new_demand(tmp_path):
    write_source(tmp_path)
    probe = store(tmp_path, budget=0)
    size = sum(a.nbytes for a in probe.get(0, 0).values())
    probe.close()
    sources = {
        policy: store(tmp_path, budget=size * 3, cache_policy=policy)
        for policy in ("lru", "frequency")
    }
    keys = [(layer, expert) for layer in range(2) for expert in range(3)]
    for source in sources.values():
        for _ in range(3):
            for key in keys:
                source.get(*key)
        assert source.used == size * 3
    assert sources["lru"].hits == 0
    assert sources["frequency"].hits >= 6
    source = sources["frequency"]
    new_key = next(k for k in keys if k not in source.cache)
    for _ in range(8):
        source.get(*new_key)
    read = source.read_bytes
    assert source.get(*new_key) is source.cache[new_key]
    assert source.read_bytes == read
    for source in sources.values():
        source.close()


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_batched_reads_preserve_order_duplicates_and_live_evicted_values(
    tmp_path, rank
):
    write_source(tmp_path)
    control = store(tmp_path, budget=0, rank=rank, io_workers=1)
    source = store(tmp_path, budget=4096, rank=rank, io_workers=4)
    expected = [control.get(1, expert) for expert in (2, 0, 2, 1)]
    actual = source.get_many(1, (2, 0, 2, 1))
    assert actual[0] is actual[2]
    assert source.misses == 3
    for a, b in zip(actual, expected):
        assert a.keys() == b.keys()
        assert all(np.array_equal(a[k], b[k]) for k in a)
        assert all(not value.flags.writeable for value in a.values())
    source.close()
    control.close()
    assert source.used == 0 and not source._fds
    assert all(np.array_equal(actual[0][k], expected[0][k]) for k in actual[0])


def test_partial_concurrent_read_never_publishes_and_fresh_source_recovers(
    tmp_path, monkeypatch
):
    from vllm.models.qwen4_exp.nvidia import expert_offload_source as module

    write_source(tmp_path)
    source = store(tmp_path)
    original = module.os.pread
    with monkeypatch.context() as patch:
        patch.setattr(
            module.os, "pread", lambda fd, n, offset: original(fd, n, offset)[:-1]
        )
        with pytest.raises(OSError, match="truncated"):
            source.get_many(0, (0, 1, 2))
    assert source.closed and not source.cache and not source._fds
    recovered = store(tmp_path)
    assert len(recovered.get_many(0, (0, 1, 2))) == 3
    recovered.close()


def test_bad_batch_recovers_without_io_and_cached_source_mutation_rejects(tmp_path):
    write_source(tmp_path)
    source = store(tmp_path)
    before = source.get_many(0, (0, 1))
    read = source.read_bytes
    with pytest.raises(ValueError, match="unknown checkpoint expert"):
        source.get_many(0, (0, 1, 4))
    assert source.read_bytes == read and source.get_many(0, (0, 1)) == before
    assert source.get_many(0, ()) == []
    write_source(tmp_path, scale=2.0)
    with pytest.raises(OSError, match="source identity changed"):
        source.get_many(0, (0, 1))
    assert source.closed and not source._fds


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
        SimpleNamespace(
            prepare_execution=lambda *, dummy, **kwargs: calls.append((dummy, kwargs))
        ),
    )
    monkeypatch.setattr(MambaHybridModelState, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(MambaHybridModelState, "prepare_dummy_inputs", lambda *args: {})
    owner.prepare_dummy_inputs(1, 1)
    batch = SimpleNamespace(
        num_tokens_after_padding=4,
        num_tokens=3,
        num_reqs=1,
        req_ids=["request-tail"],
        has_prefill=False,
        query_start_loc_np=np.array([0, 3]),
        num_computed_tokens_np=np.array([19]),
        num_scheduled_tokens=np.array([3]),
    )
    owner.prepare_inputs(batch, None)
    owner.prepare_runtime_dummy_inputs(batch, None)
    owner.prepare_inputs(batch, None)
    assert [dummy for dummy, _ in calls] == [True, False, True, False]
    assert calls[1][1]["num_tokens"] == 4
    assert calls[1][1]["identity"]["req_ids"] == ["request-tail"]
    assert calls[1][1]["identity"]["scheduled"] == [3]


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


def test_provider_retirement_starts_a_fresh_graph_pool_epoch(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.models.qwen4_exp.nvidia import expert_offload_provider as module

    provider = module.NativeExpertProvider.__new__(module.NativeExpertProvider)
    graph = Mock()
    provider.bank = SimpleNamespace(
        graphs={graph}, stable_graphs={graph}, close_prefetch=Mock()
    )
    provider.kernels = {16: (object(), graph)}
    provider.admission = provider.hybrid_path = provider.stream_path = None
    provider.quiesce = Mock()
    provider.graph_pool = object()
    fresh = object()
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: fresh)
    provider.retire()
    provider.quiesce.assert_called_once_with()
    graph.reset.assert_called_once_with()
    assert provider.graph_pool is fresh
    assert not (provider.kernels or provider.bank.graphs or provider.bank.stable_graphs)
    assert provider.consumer is None and provider.consumer_rows is None


def _prefetch_bank(source, sample=None):
    from threading import RLock

    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import NativeExpertBank

    bank = NativeExpertBank.__new__(NativeExpertBank)
    bank.source, bank.lock = source, RLock()
    bank.state, bank.staging, bank.leases = "READY", 4, {object()}
    bank.host_hot = {(1, 0): 0}
    bank.prefetch_reader = bank.prefetch_future = bank.prefetch_layer = None
    bank.prefetch_error = None
    bank.prefetch_ids = ()
    bank.next_demand = None
    bank.prepared = {}
    bank.prepared_native = {}
    bank.prefetch_region = None
    bank.prefetch_experts = bank.prefetch_consumed = 0
    sample = source.get(0, 0) if sample is None else sample
    bank.shapes = {name: value.shape for name, value in sample.items()}

    def buffers():
        arrays = {
            name: np.zeros((4, *a.shape), dtype=a.dtype) for name, a in sample.items()
        }
        return {name: torch.from_numpy(a) for name, a in arrays.items()}, arrays

    bank._allocate_pinned = buffers
    bank.pinned, bank.pinned_numpy = buffers()
    bank.next_pinned = bank.next_pinned_numpy = None
    return bank


def test_registered_prefetch_retains_native_views_and_rejects_stale_region():
    from types import SimpleNamespace

    sample = {"w": np.zeros((2, 2), np.uint8)}
    supplied = []

    def get_many(layer, experts):
        rows = [{"w": np.full((2, 2), layer * 16 + e, np.uint8)} for e in experts]
        for row in rows:
            row["w"].flags.writeable = False
        supplied.append(rows)
        return rows

    source = SimpleNamespace(pin_cache=False, get_many=get_many)
    bank = _prefetch_bank(source, sample)
    region = SimpleNamespace(registered=True)
    bank.registered_source_region = region

    def forbidden(*args):
        raise AssertionError("registered prefetch allocated or copied pinned staging")

    bank._allocate_pinned = bank._fill_pinned = forbidden
    try:
        bank.prefetch(1, [0, 1, 2])
        bank._finish_prefetch(1)
        assert set(bank.prepared_native) == {1, 2} and not bank.prepared
        assert bank.prepared_native[1] is supplied[-1][0]
        assert bank.next_pinned is None
        bank.prefetch(1, [3])
        bank.prefetch_future.result()
        bank.registered_source_region = SimpleNamespace(registered=True)
        with pytest.raises(RuntimeError, match="stale registered"):
            bank._finish_prefetch(1)
        assert not bank.prepared_native and bank.prefetch_future is None
        bank.registered_source_region = region
        bank.prefetch(1, [2])
        bank._finish_prefetch(1)
        assert set(bank.prepared_native) == {2}
    finally:
        bank.close_prefetch()
    assert not bank.prepared_native


def test_exact_prefetch_excludes_hot_and_retains_bypassed_rows(tmp_path):
    write_source(tmp_path)
    source = store(tmp_path, budget=0)
    bank = _prefetch_bank(source)
    misses = source.misses
    try:
        bank.prefetch(1, (0, 1, 2))
        bank._finish_prefetch(1)
        assert set(bank.prepared) == {1, 2}
        assert source.misses - misses == 2 and not source.cache
        assert bank.prefetch_experts == 2
        for expert, position in bank.prepared.items():
            expected = source.get(1, expert)
            for name, value in expected.items():
                np.testing.assert_array_equal(bank.pinned_numpy[name][position], value)
        bank.drain_prefetch()
        assert not bank.prepared and bank.prefetch_future is None
        with pytest.raises(ValueError, match="one staging wave"):
            bank.prefetch(1, (1, 2, 3, 4, 5))
    finally:
        bank.close_prefetch()
        source.close()
    assert bank.prefetch_reader is None


def test_prefetch_error_and_stale_epoch_are_joined_before_recovery(tmp_path):
    write_source(tmp_path)
    source = store(tmp_path, budget=0)
    bank = _prefetch_bank(source)
    try:
        bank.prefetch(1, (1, 2))
        with pytest.raises(RuntimeError, match="stale native prefetch"):
            bank._finish_prefetch(0)
        assert not bank.prepared and bank.prefetch_future is None
        bank.prefetch(1, (1, 2))
        bank._finish_prefetch(1)
        assert set(bank.prepared) == {1, 2}
        bank.prefetch(1, (9,))
        with pytest.raises(ValueError, match="unknown checkpoint"):
            bank._finish_prefetch(1)
        assert not bank.prepared and bank.prefetch_future is None
    finally:
        bank.close_prefetch()
        source.close()


def test_prefetch_cancellation_waits_for_outstanding_source_reader():
    from threading import Event, Thread
    from types import SimpleNamespace

    entered, release, closed = Event(), Event(), Event()
    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import COMPONENTS, SCALARS

    sample = {
        name: np.ones((), dtype=np.float32)
        if name in SCALARS
        else np.ones((1,), dtype=np.uint8)
        for name in COMPONENTS
    }

    def read(layer, experts):
        entered.set()
        assert release.wait(5)
        return [sample for _ in experts]

    bank = _prefetch_bank(SimpleNamespace(get_many=read), sample)
    bank.prefetch(1, (1,))
    assert entered.wait(5)

    def close():
        bank.close_prefetch()
        closed.set()

    thread = Thread(target=close)
    thread.start()
    try:
        assert not closed.wait(0.05)
    finally:
        release.set()
        thread.join(5)
    assert closed.is_set() and not thread.is_alive()
    assert bank.prefetch_future is bank.prefetch_reader is None
    assert not bank.prepared


def test_prefetch_tracks_provisional_victims_and_swaps_pinned_generations(tmp_path):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia.expert_offload_bank import NativeBankPlan

    write_source(tmp_path)
    source = store(tmp_path, budget=0)
    bank = _prefetch_bank(source)
    original = bank.pinned
    bank.state = "LOADING"
    bank.tables = SimpleNamespace(pool_rows=1)
    # Row0 previously held next-wave expert(1,0); this pending copy evicts it.
    bank.plan = NativeBankPlan(1, 0, (0,), ((2, 0),))
    try:
        bank.prefetch(1, (0, 1))
        bank._finish_prefetch(1)
        assert set(bank.prepared) == {0, 1}
        assert bank.pinned is not original and bank.next_pinned is original
        first = {name: a.copy() for name, a in bank.pinned_numpy.items()}
        bank.prefetch(0, (1, 2))
        bank.prefetch_future.result()
        assert all(
            np.array_equal(a, first[name]) for name, a in bank.pinned_numpy.items()
        )
        bank._finish_prefetch(0)
        assert bank.pinned is original
        for expert, position in bank.prepared.items():
            for name, value in source.get(0, expert).items():
                np.testing.assert_array_equal(bank.pinned_numpy[name][position], value)
        # Allocation errors must surface at the common next-stage boundary.
        bank.next_demand = (1, (0, 1, 2, 3, 4))
        bank.start_next()
        with pytest.raises(ValueError, match="one staging wave"):
            bank._finish_prefetch(1)
    finally:
        bank.close_prefetch()
        source.close()


def test_parallel_reads_keep_preparation_on_owner_and_bound_raw_queue():
    from threading import Lock, get_ident

    source = NativeExpertStore.__new__(NativeExpertStore)
    source._reader, source.io_workers = None, 2
    owner = get_ident()
    lock = Lock()
    buffered = peak = 0

    def read(key):
        nonlocal buffered, peak
        assert get_ident() != owner
        with lock:
            buffered += 1
            peak = max(peak, buffered)
        return key, 1, 1

    def prepare(raw):
        nonlocal buffered
        assert get_ident() == owner
        with lock:
            buffered -= 1
        return raw

    source._read_raw, source._prepare_raw = read, prepare
    try:
        assert source._read_many(list(range(32))) == [(key, 1, 1) for key in range(32)]
        assert buffered == 0 and peak <= source.io_workers * 2 + 1
    finally:
        source._reader.shutdown(wait=True, cancel_futures=True)
