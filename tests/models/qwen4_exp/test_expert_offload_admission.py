# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU state controls; physical DMA/Graph acceptance is a separate gate."""

from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia import expert_offload_tables as gp
from vllm.models.qwen4_exp.nvidia.expert_offload_admission import (
    ExpertFrequencyAdmission,
    NativeExpertAdmission,
)
from vllm.models.qwen4_exp.nvidia.expert_offload_bank import (
    COMPONENTS,
    NativeBankCoordinator,
    NativeExpertBank,
)
from vllm.models.qwen4_exp.nvidia.expert_offload_source import NativeExpertStore


def test_manager_closes_capture_inputs_on_success_failure_and_recovery(monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import (
        CudaGraphManager,
        ModelCudaGraphManager,
    )

    manager = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    manager.use_breakable_cg = manager.tp3_owner_prequant = False
    manager.cudagraph_mode = CUDAGraphMode.NONE
    events = []
    fail = False

    def capture(*args, **kwargs):
        events.append("capture")
        if fail:
            raise ValueError("original capture failure")

    monkeypatch.setattr(CudaGraphManager, "capture", capture)
    state = SimpleNamespace(finish_native_capture=lambda: events.append("release"))
    for fail in (False, True, False):
        events.clear()
        if fail:
            with pytest.raises(ValueError, match="original capture failure"):
                manager.capture(None, state, None, None, None, [], None)
        else:
            manager.capture(None, state, None, None, None, [], None)
        assert events == ["capture", "release"]


def test_failed_allocation_rpc_keeps_primary_error_and_restores_core_state(monkeypatch):
    from vllm.v1.core.elastic_graph import ElasticPlanKind
    from vllm.v1.engine import elastic_memory_profile as profile
    from vllm.v1.engine.elastic_calibrator import ElasticCatalogCalibrator

    key = (0, 3, 1, 1, 0)
    previous: dict[tuple[int, ...], dict] = {}
    events: list[str] = []
    scheduler = SimpleNamespace(
        has_unfinished_requests=lambda: False,
        _elastic_graph_execution_policy=SimpleNamespace(fingerprint="policy"),
        _elastic_restore_mode=False,
        _elastic_graph_catalog=previous,
        num_spec_tokens=3,
        prepare_elastic_restore_idle_reclaim=lambda: events.append("reclaim"),
        prepare_elastic_restore_capture=lambda key: True,
        schedule=lambda **kwargs: SimpleNamespace(
            total_num_scheduled_tokens=0,
            elastic_step_plan=SimpleNamespace(kind=ElasticPlanKind.MAINTENANCE),
        ),
        update_from_output=lambda *args: None,
        assert_elastic_restore_captures_hot=lambda keys: None,
    )

    def execute(output):
        scheduler._elastic_graph_catalog[key] = {}
        return SimpleNamespace(req_ids=[])

    def failed_rpc(*args, **kwargs):
        raise ValueError("primary worker failure")

    owner = SimpleNamespace(
        scheduler=scheduler,
        vllm_config=SimpleNamespace(
            additional_config={"flashnext_native_experts": True},
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        ),
        model_executor=SimpleNamespace(
            execute_model=execute, collective_rpc=failed_rpc
        ),
    )
    monkeypatch.setattr(
        profile, "allocation_profile_shapes", lambda *args: ((key,), ())
    )
    monkeypatch.setattr(
        ElasticCatalogCalibrator,
        "_validate_surface_before_mutation",
        lambda *args: None,
    )
    for _ in range(2):
        events.clear()
        with pytest.raises(ValueError, match="primary worker failure"):
            profile.profile_allocation_catalog(owner, SimpleNamespace(decode_max_x=1))
        assert events == ["reclaim"]
        assert scheduler._elastic_graph_catalog is previous
        assert scheduler._elastic_restore_mode is False


def test_fp4_scale_bound_covers_independent_partition_maximum():
    from vllm._custom_ops import _fp4_moe_blockscale_rows

    # Independent dynamic programming over every count partition; no copy of
    # the closed-form allocation expression in this oracle.
    maximum = [0] + [-(10**9)] * 300
    for groups in range(1, 9):
        maximum = [
            max(maximum[n - c] + ((c + 127) // 128) * 128 for c in range(n + 1))
            for n in range(301)
        ]
        assert maximum == [_fp4_moe_blockscale_rows(n, groups) for n in range(301)]
    assert _fp4_moe_blockscale_rows(40, 512) == 5120
    assert _fp4_moe_blockscale_rows(0, 512) == 0
    for lanes, groups in ((-1, 512), (40, 0)):
        with pytest.raises(ValueError, match="geometry"):
            _fp4_moe_blockscale_rows(lanes, groups)


def test_stream_trace_conserves_hot_cold_and_padding_without_weight_access():
    from vllm.models.qwen4_exp.nvidia.expert_offload_admission import stream_trace_layer

    ids = np.tile(np.arange(10, dtype=np.int32), (2, 1))
    weights = np.full(ids.shape, 0.1, np.float32)
    cold = ids.copy()
    cold[:, :5] = -1

    def render():
        return stream_trace_layer(
            3, ids, weights, cold, cpu={}, state="READY", hot_rows=96, num_experts=512
        )

    row = render()
    assert (row["hot_lanes"], row["cold_lanes"], row["useful_lanes"]) == (10, 10, 20)
    assert row["route"]["gpu_hint_experts"] == list(range(5))
    assert row["route"]["cpu_experts"] == list(range(5, 10))
    cold[0, 9] = 111
    with pytest.raises(ValueError, match="identity"):
        render()
    cold[0, 9] = 9
    ids[1] = cold[1] = -1
    weights[1] = 0
    row = render()
    assert row["useful_lanes"] == 10 and row["route"]["useful_tokens"] == 1


def test_residency_snapshot_counts_uniform_bytes_without_borrowing_weights():
    from vllm.models.qwen4_exp.nvidia.expert_offload_admission import cache_snapshot

    class MetadataOnly(dict):
        def __getitem__(self, key):
            raise AssertionError("observer borrowed expert weights")

    source = SimpleNamespace(
        lock=RLock(),
        cache=MetadataOnly({(0, 1): None, (0, 2): None}),
        pinned_pool=None,
        used=32,
        limit=64,
        hits=2,
        misses=2,
        read_bytes=32,
    )
    bank = SimpleNamespace(
        source=source,
        lock=RLock(),
        host_hot={(0, 2): 0},
        strides={"w": 16},
        generation=1,
        state="READY",
        tables=SimpleNamespace(pool_rows=2),
        staging=1,
        backings={"w": SimpleNamespace(info=SimpleNamespace(committed=64))},
        leases=set(),
    )
    receipt = cache_snapshot(bank)
    assert receipt["gpu_ram_duplicate_keys"] == 1
    assert receipt["gpu_ram_duplicate_payload_bytes"] == 16


@pytest.mark.parametrize(
    "tp,widths",
    [(1, [640]), (2, [320, 320]), (3, [256, 256, 128]), (4, [192, 192, 192, 64])],
)
def test_stream_ram_and_vmm_follow_actual_tp_shards(tp, widths):
    from vllm.models.qwen4_exp.nvidia.expert_offload_archive import row_schema
    from vllm.v1.core.elastic_expert import NativeExpertBudget

    options = dict(
        ram_cache_total_bytes=28 << 30,
        prepared_archive="/prepared",
        pin_ram_cache=True,
        hot_read=True,
        stream_experts=dict(
            library="/executor.so",
            sha256="a" * 64,
            cores=list(range(tp * 2)),
            max_m=4,
            history_steps=16,
            max_promotions=192,
        ),
    )
    config = SimpleNamespace(
        additional_config={"flashnext_native_experts": options},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=2560,
                moe_intermediate_size=640,
                num_hidden_layers=48,
                num_experts=512,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )
    budget = NativeExpertBudget.from_config(config)
    capacities, ram = [], []
    for rank, width in enumerate(widths):
        geometry = budget.rank_geometry(rank)
        assert geometry.local == width
        fields, _, stride = row_schema(geometry)
        local_ram = budget.rank_ram_bytes(rank)
        capacities.append(local_ram // stride)
        ram.append(local_ram)
        for hot in (0, 1, 3699, 8192):
            q, rows = 2 << 20, hot + budget.staging
            physical = sum(
                ((rows * fields[name]["nbytes"] + q - 1) // q) * q
                for name in (
                    "w13_weight",
                    "w2_weight",
                    "w13_weight_scale",
                    "w2_weight_scale",
                )
            )
            physical += ((budget.max_hot_rows + budget.staging) * 24 + q - 1) // q * q
            assert (
                physical
                == budget.rank_mapped_bytes(rank, hot)
                <= budget.mapped_bytes(hot)
            )
    assert len(set(capacities)) == 1 and sum(ram) <= 28 << 30
    stream_options = options["stream_experts"]
    assert isinstance(stream_options, dict)
    stream_options["history_steps"] = 0
    with pytest.raises(ValueError, match="placement"):
        NativeExpertBudget.from_config(config)


def test_stream_completion_rejects_missing_layer_before_policy(monkeypatch):
    from unittest.mock import Mock

    from vllm.models.qwen4_exp.nvidia.expert_offload_stream import NativeStreamExperts

    owner = NativeStreamExperts.__new__(NativeStreamExperts)
    bank = SimpleNamespace(state="READY", generation=1, leases={object()})
    owner.bank, owner.m, owner.started = bank, 1, 0
    owner.statuses = torch.tensor([[0], [-1]], dtype=torch.int32)
    owner.routes = torch.ones((2, 1, 2), dtype=torch.int32)
    owner.hw = torch.ones((2, 1, 2), dtype=torch.float32)
    owner.abort_step = lambda: bank.leases.clear()
    coordinator = SimpleNamespace(
        send=torch.empty(40, dtype=torch.int64),
        recv=torch.empty(40, dtype=torch.int64),
        ranks=1,
        group=SimpleNamespace(device_group=None),
    )
    admission = Mock()
    owner.provider = SimpleNamespace(coordinator=coordinator, admission=admission)
    monkeypatch.setattr(
        torch.distributed, "all_gather_single", lambda dst, src, **kw: dst.copy_(src)
    )
    with pytest.raises(RuntimeError, match="completion/route"):
        owner.finish(dummy=False)
    assert bank.state == "POISONED" and not bank.leases
    admission.finish_step.assert_not_called()


def bank_fixture():
    bank = NativeExpertBank.__new__(NativeExpertBank)
    bank.lock, bank.state, bank.device = RLock(), "READY", "cpu"
    bank.tables = gp.allocate_global_tables("cpu", 8, [2, 2], 2)
    bank.source = SimpleNamespace(layers=2, experts=8)
    bank.host_hot = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    bank.host_rows = {v: k for k, v in bank.host_hot.items()}
    bank.leases, bank.completed, bank.prepared = set(), set(), {}
    bank.pending_promotions = set()
    bank.rows, bank.staging, bank.generation = 6, 2, 0
    bank._finish_prefetch = lambda layer: None
    return bank


def test_normalized_prefill_does_not_outvote_decode_by_physical_m():
    policy = ExpertFrequencyAdmission(1, 8, history_steps=2)
    prefill = np.array([[1024, 0, 0, 0, 0, 0, 0, 0]], np.uint32)
    decode = np.array([[0, 4, 0, 0, 0, 0, 0, 0]], np.uint32)
    policy.observe(prefill, useful_tokens=1024)
    policy.observe(decode, useful_tokens=4)
    assert policy.frequency[0, 0] == policy.frequency[0, 1] == 1
    for _ in range(50):
        policy.observe(decode, useful_tokens=3)
    assert policy.frequency.min() == 0
    proposal = policy.plan([0], [1], capacity=1, max_promotions=1)
    assert proposal.desired == (1,)
    before = policy.frequency.copy()
    with pytest.raises(ValueError):
        policy.observe(decode, useful_tokens=0)
    np.testing.assert_array_equal(policy.frequency, before)


def test_complete_model_observation_discards_cancelled_tail():
    owner = NativeExpertAdmission.__new__(NativeExpertAdmission)
    owner.bank = bank_fixture()
    owner.counts = np.zeros((2, 8), np.uint32)
    owner.next_layer, owner.tokens = 0, None
    ids = np.array([[1, 2]], np.int32)
    weights = np.ones((1, 2), np.float32)
    owner.observe_layer(0, ids, weights)
    owner.observe_layer(0, ids + 2, weights)  # New model after cancellation.
    owner.observe_layer(1, ids, weights)
    assert owner.counts[0, 1] == 0 and owner.counts[0, 3] == 1
    with pytest.raises(ValueError, match="reordered"):
        owner.observe_layer(1, ids, weights)


def test_explicit_placement_cannot_publish_partial_components():
    bank = bank_fixture()
    ticket = bank.begin(0, torch.tensor([3], dtype=torch.int32), destinations=[2])
    assert bank.state == "LOADING" and bank.host_hot[1, 0] == 2
    assert (0, 3) not in bank.host_hot  # Device row is still provisional.
    with pytest.raises(RuntimeError, match="not ready"):
        bank.acquire(ticket)
    bank.completed = {(2, name) for name in COMPONENTS[:-1]}
    with pytest.raises(RuntimeError, match="incomplete"):
        bank.publish(ticket, all_ranks_ready=True)
    assert bank.state == "POISONED" and (0, 3) not in bank.host_hot


def test_explicit_placement_preserves_retained_rows_and_publishes_reverse_owner():
    bank = bank_fixture()
    ticket = bank.begin(0, torch.tensor([3], dtype=torch.int32), destinations=[2])
    bank.completed = {(2, name) for name in COMPONENTS}
    bank.publish(ticket, all_ranks_ready=True)
    assert bank.host_hot == {(0, 0): 0, (0, 1): 1, (0, 3): 2, (1, 1): 3}
    assert bank.host_rows[2] == (0, 3)
    assert bank.tables.hot_phys[8] == -1 and bank.tables.cold_phys[8] == 0
    assert bank.tables.hot_phys[3] == 2 and bank.tables.row_key[2] == 3
    lease = bank.acquire(ticket)
    with pytest.raises(RuntimeError, match="unavailable"):
        bank.begin(0, torch.tensor([4], dtype=torch.int32), destinations=[3])
    assert lease in bank.leases and bank.state == "READY"


@pytest.mark.parametrize("case", ["range", "duplicate", "already_hot", "stale_device"])
def test_invalid_assignment_fails_before_any_owner_change(case):
    bank = bank_fixture()
    ids, rows = [3], [2]
    if case == "range":
        rows = [bank.tables.pool_rows]
    elif case == "duplicate":
        ids, rows = [3, 4], [2, 2]
    elif case == "already_hot":
        ids = [1]
    else:
        bank.tables.row_key[2] = -1
    before = bank.tables.hot_phys.clone()
    with pytest.raises((ValueError, RuntimeError)):
        bank.begin(0, torch.tensor(ids, dtype=torch.int32), destinations=rows)
    assert bank.state == "POISONED"
    assert torch.equal(before, bank.tables.hot_phys)


def test_peer_assignment_divergence_rejects_before_first_copy(monkeypatch):
    bank = bank_fixture()
    policy = ExpertFrequencyAdmission(2, 8, history_steps=2)
    counts = np.zeros((2, 8), np.uint32)
    counts[:, 3] = 8
    policy.observe(counts)
    proposal = policy.plan([0, 1, 8, 9], [3, 11], capacity=4, max_promotions=2)
    coordinator = NativeBankCoordinator.__new__(NativeBankCoordinator)
    coordinator.bank, coordinator.ranks = bank, 3
    coordinator.group = SimpleNamespace(device_group="control")
    coordinator.send, coordinator.recv = (
        torch.empty(72, dtype=torch.int64),
        torch.empty(216, dtype=torch.int64),
    )
    calls = []
    coordinator.stage = lambda *a, **kw: calls.append((a, kw))

    def divergent(out, incoming, *, group):
        out.view(3, -1).copy_(incoming.expand(3, -1))
        out.view(3, -1)[1, 3] += 1

    monkeypatch.setattr(torch.distributed, "all_gather_single", divergent)
    before = bank.tables.hot_phys.clone()
    with pytest.raises(RuntimeError, match="rank-inconsistent"):
        coordinator.apply_admission(proposal)
    assert not calls and torch.equal(before, bank.tables.hot_phys)


def test_gpu_ram_release_keeps_borrowed_arrays_and_updates_residual_priority():
    source = NativeExpertStore.__new__(NativeExpertStore)
    source.lock, source.closed = RLock(), False
    source.layers, source.experts, source.used = 2, 8, 32
    source.frequency, source.residency_scores = np.zeros((2, 8), np.uint16), None
    source.cache = {
        (0, 1): {"w": np.arange(16, dtype=np.uint8)},
        (1, 2): {"w": np.arange(16, dtype=np.uint8)},
    }
    borrowed = source.cache[0, 1]["w"]
    scores = np.zeros((2, 8), np.float64)
    scores[1, 2] = 0.75
    assert source.set_residency_scores(scores, [(0, 1)]) == 16
    scores.fill(9)
    assert source.used == 16 and source.cache_heap == [(0.75, (1, 2))]
    np.testing.assert_array_equal(borrowed, np.arange(16, dtype=np.uint8))
    with pytest.raises(ValueError):
        source.set_residency_scores(scores * np.nan, [(1, 2)])
    assert source.used == 16 and (1, 2) in source.cache


def asynchronous_fixture(monkeypatch):
    from concurrent.futures import Future

    from vllm.models.qwen4_exp.nvidia.expert_offload_promotion import (
        NativeExpertPromotion,
    )

    bank = bank_fixture()
    bank.strides = {name: 4 for name in COMPONENTS}
    bank.copy_bytes = bank.copy_calls = 0
    bank.source.lock = RLock()
    bank.source.cache = {
        (layer, 3): {name: np.ones(1, np.float32) for name in COMPONENTS}
        for layer in range(2)
    }
    bank.source.borrow_resident = lambda keys: [bank.source.cache[key] for key in keys]
    coordinator = NativeBankCoordinator.__new__(NativeBankCoordinator)
    coordinator.bank, coordinator.ranks = bank, 3
    coordinator.group = SimpleNamespace(device_group="control")
    coordinator.send, coordinator.recv = (
        torch.empty(72, dtype=torch.int64),
        torch.empty(216, dtype=torch.int64),
    )
    status = {"peer": 1}

    def gather(out, incoming, *, group):
        peers = out.view(3, -1)
        peers.copy_(incoming.expand(3, -1))
        peers[1, 0] = min(int(incoming[0]), status["peer"])

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather)
    upload = NativeExpertPromotion.__new__(NativeExpertPromotion)
    upload.bank, upload.coordinator = bank, coordinator
    upload.future, upload.pending, upload.sequence, upload.last_step = None, None, 0, {}
    upload.staging_bytes = 80
    upload.registration = None
    completion: Future[dict] = Future()
    upload.worker = SimpleNamespace(submit=lambda *args: completion)
    policy = ExpertFrequencyAdmission(2, 8, history_steps=2)
    counts = np.zeros((2, 8), np.uint32)
    counts[:, 3] = 8
    policy.observe(counts)
    proposal = policy.plan([0, 1, 8, 9], [3, 11], capacity=4, max_promotions=2)
    return upload, completion, proposal, status


def test_async_rows_stay_cpu_misses_until_every_rank_completes(monkeypatch):
    upload, completion, proposal, status = asynchronous_fixture(monkeypatch)
    upload.start(proposal)
    bank = upload.bank
    assert bank.state == "READY" and bank.pending_promotions == {2, 3}
    assert bank.host_hot == {(0, 0): 0, (0, 1): 1}
    assert bank.tables.hot_phys[3] == bank.tables.hot_phys[11] == -1
    assert not upload.poll()
    completion.set_result(dict(error=None, copies=1, bytes=80, copy_wall_ms=1))
    status["peer"] = 0
    assert not upload.poll()  # Local readiness cannot publish the TP model.
    assert (0, 3) not in bank.host_hot
    status["peer"] = 1
    assert upload.poll()
    assert set(bank.host_hot) == {(0, 0), (0, 1), (0, 3), (1, 3)}
    assert not bank.pending_promotions and bank.copy_bytes == 80
    assert bank.tables.row_key[2] == 3 and bank.tables.row_key[3] == 11


def test_async_failure_and_resize_cannot_expose_unfinished_rows(monkeypatch):
    upload, completion, proposal, status = asynchronous_fixture(monkeypatch)
    upload.start(proposal)
    bank = upload.bank
    bank.targets = lambda rows: {}
    with pytest.raises(RuntimeError, match="maintenance unavailable"):
        bank.prepare_resize(0)
    completion.set_result(dict(error="short copy", copies=0, bytes=0, copy_wall_ms=1))
    with pytest.raises(RuntimeError, match="asynchronous expert"):
        upload.poll()
    assert bank.state == "POISONED" and not bank.pending_promotions
    assert (0, 3) not in bank.host_hot and bank.tables.hot_phys[3] == -1
