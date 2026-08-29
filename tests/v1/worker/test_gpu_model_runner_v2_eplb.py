#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call

import pytest
import torch

from vllm.config.offload import OffloadConfig
from vllm.v1.core.elastic_graph import ElasticPlanKind
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu import eplb_utils as eplb
from vllm.v1.worker.gpu import model_runner as mrv2


def test_request_free_maintenance_preserves_published_graph_owner_set():
    output = SimpleNamespace(
        elastic_step_plan=SimpleNamespace(kind=ElasticPlanKind.MAINTENANCE),
        elastic_abort_staged_hotset=False,
    )

    assert mrv2._release_idle_graph_cache(output) is False


def test_request_free_idle_and_reclaim_release_graph_owner_set():
    idle = SimpleNamespace(
        elastic_step_plan=None,
        elastic_abort_staged_hotset=False,
    )
    reclaim = SimpleNamespace(
        elastic_step_plan=SimpleNamespace(kind=ElasticPlanKind.RECLAIM),
        elastic_abort_staged_hotset=False,
    )

    assert mrv2._release_idle_graph_cache(idle) is True
    assert mrv2._release_idle_graph_cache(reclaim) is True


def test_dynamic_graph_working_set_includes_target_and_draft_owners():
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1)
    )
    target = MagicMock()
    target.dynamic_resident_bytes = 7
    target.dynamic_active_graph_bytes = 5
    target.has_pending_dynamic_capture.return_value = False
    prefill = MagicMock()
    prefill.dynamic_resident_bytes = 11
    prefill.dynamic_active_graph_bytes = 7
    prefill.has_pending_dynamic_capture.return_value = False
    decode = MagicMock()
    decode.dynamic_resident_bytes = 13
    decode.dynamic_active_graph_bytes = 9
    decode.has_pending_dynamic_capture.return_value = False
    runner.cudagraph_manager = target
    runner.speculator = SimpleNamespace(
        dynamic_cudagraph_managers=lambda: (prefill, decode)
    )
    runner.elastic_kv_controller = MagicMock()
    runner.elastic_kv_controller.reconcile_external_memory.return_value = 37
    runner._trim_dynamic_attention_cudagraph_state = MagicMock(return_value=(3, 17))
    runner._measure_elastic_cublas_workspace_bytes = MagicMock(return_value=0)
    runner._elastic_step_measurement_active = True

    working_set = runner._dynamic_graph_working_set()
    assert working_set.managers == (target, prefill, decode)
    working_set.transition_floor_upper_bound_bytes = MagicMock(return_value=19)
    runner._dynamic_graph_working_set = lambda: working_set
    assert runner._finish_dynamic_graph_step() == (31, 10, 19)
    runner.elastic_kv_controller.reconcile_external_memory.assert_not_called()
    runner._trim_dynamic_attention_cudagraph_state.assert_not_called()
    for manager in (target, prefill, decode):
        manager.finish_dynamic_step.assert_called_once_with()


def test_dynamic_graph_working_set_persists_staged_retirement_until_successor(
    monkeypatch,
):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    target = MagicMock()
    target.dynamic_graph_owner = "target"
    prefill = MagicMock()
    prefill.dynamic_graph_owner = "mtp_prefill"
    decode = MagicMock()
    decode.dynamic_graph_owner = "mtp_decode"
    runner.cudagraph_manager = target
    runner.speculator = SimpleNamespace(
        dynamic_cudagraph_managers=lambda: (prefill, decode)
    )

    capture_boundary = runner._dynamic_graph_working_set()
    target_victim = SimpleNamespace(
        logical=SimpleNamespace(owner="target"), identity="old-target"
    )
    prefill_victim = SimpleNamespace(
        logical=SimpleNamespace(owner="mtp_prefill"), identity="old-prefill"
    )
    plan = SimpleNamespace(
        staged_hotset_replace=True,
        transaction_id="q1-to-q4",
        victim_keys=(target_victim, prefill_victim),
        physical_keys=(MagicMock(),),
    )
    capture_boundary.stage_hotset_victim_commit(plan)

    settlement_boundary = runner._dynamic_graph_working_set()
    assert settlement_boundary is capture_boundary
    assert settlement_boundary.has_pending_staged_hotset_retirement

    # Request-free maintenance publishes but does not consume the successor.
    # Its settlement must retain the physical victims.
    runner._elastic_step_measurement_active = False
    runner._elastic_cached_graph_receipt = (101, 7, 3)
    monkeypatch.setattr(mrv2.torch.cuda, "synchronize", lambda _device: None)
    assert runner._finish_dynamic_graph_step(transaction_id="q1-to-q4") == (101, 7, 3)
    target.evict_physical_key.assert_not_called()
    prefill.evict_physical_key.assert_not_called()
    assert settlement_boundary.has_pending_staged_hotset_retirement

    # The first distinct USER settlement clears the fence, but the predecessor
    # graph objects remain HOT through the active workload epoch. Idle teardown
    # is the first safe physical destruction point for their shared pool state.
    assert runner._finish_dynamic_graph_step(
        transaction_id="q4-first-user",
        step_plan=SimpleNamespace(
            transaction_id="q4-first-user",
            kind=mrv2.ElasticPlanKind.USER,
            physical_keys=plan.physical_keys,
        ),
    ) == (101, 7, 3)
    target.is_physical_key_hot.assert_called_once_with(target_victim)
    prefill.is_physical_key_hot.assert_called_once_with(prefill_victim)
    target.evict_physical_key.assert_not_called()
    prefill.evict_physical_key.assert_not_called()
    decode.evict_physical_key.assert_not_called()
    assert not settlement_boundary.has_pending_staged_hotset_retirement


def test_dynamic_graph_working_set_rejects_runtime_manager_replacement():
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.cudagraph_manager = MagicMock(dynamic_graph_owner="target")
    draft = MagicMock(dynamic_graph_owner="mtp_decode")
    runner.speculator = SimpleNamespace(dynamic_cudagraph_managers=lambda: (draft,))
    runner._dynamic_graph_working_set()

    runner.speculator = SimpleNamespace(
        dynamic_cudagraph_managers=lambda: (
            MagicMock(dynamic_graph_owner="mtp_decode"),
        )
    )
    with pytest.raises(
        RuntimeError,
        match="manager set changed after working-set lifecycle initialization",
    ):
        runner._dynamic_graph_working_set()


def test_idle_dynamic_graph_step_reports_physical_memory_floor(monkeypatch, caplog):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1)
    )
    working_set = MagicMock()
    working_set.resident_bytes = 31
    working_set.active_graph_bytes = 20
    working_set.has_pending_staged_hotset_retirement = False
    runner._dynamic_graph_working_set = lambda: working_set
    runner.elastic_kv_controller = MagicMock()
    runner.elastic_kv_controller.reconcile_external_memory.return_value = 37
    runner._record_elastic_x0_allocation_delta = MagicMock()
    runner._measure_elastic_cublas_workspace_bytes = MagicMock(return_value=0)
    runner._elastic_step_measurement_active = True
    monkeypatch.setattr(mrv2.torch.cuda, "memory_allocated", lambda _device: 101)
    monkeypatch.setattr(mrv2.torch.cuda, "memory_reserved", lambda _device: 202)
    monkeypatch.setattr(mrv2.torch.cuda, "mem_get_info", lambda _device: (303, 404))

    assert runner._finish_dynamic_graph_step(release_idle_cache=True) == (37, 17, 17)
    working_set.finish_idle_step.assert_called_once_with()
    runner.elastic_kv_controller.reconcile_external_memory.assert_called_once_with(20)
    working_set.clear_idle_retention_after_physical_reconcile.assert_called_once_with()
    assert runner._elastic_idle_logged_floor_bytes == 17
    assert "allocator_allocated_bytes=101" in caplog.text
    assert "driver_free_bytes=303" in caplog.text


def test_idle_dynamic_graph_step_reports_zero_floor_after_graph_eviction(
    monkeypatch, caplog
):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1)
    )
    runner._elastic_idle_logged_floor_bytes = 0

    class WorkingSet:
        resident_bytes = 31
        active_graph_bytes = 20
        idle_retention_cleared = False
        has_pending_staged_hotset_retirement = False

        def finish_idle_step(self):
            self.resident_bytes = 0
            self.active_graph_bytes = 0

        def reconcile_retained_cleanup(self, _reclaimed):
            return None

        def clear_idle_retention_after_physical_reconcile(self):
            self.idle_retention_cleared = True

    working_set = WorkingSet()
    runner._dynamic_graph_working_set = lambda: working_set
    runner.elastic_kv_controller = MagicMock()
    runner.elastic_kv_controller.reconcile_external_memory.return_value = 0
    runner._record_elastic_x0_allocation_delta = MagicMock()
    runner._clear_elastic_cublas_workspaces = MagicMock(return_value=0)
    runner._measure_elastic_cublas_workspace_bytes = MagicMock(return_value=0)
    runner._elastic_step_measurement_active = True
    monkeypatch.setattr(mrv2.torch.cuda, "memory_allocated", lambda _device: 101)
    monkeypatch.setattr(mrv2.torch.cuda, "memory_reserved", lambda _device: 202)
    monkeypatch.setattr(mrv2.torch.cuda, "mem_get_info", lambda _device: (303, 404))

    assert runner._finish_dynamic_graph_step(release_idle_cache=True) == (0, 0, 0)
    runner.elastic_kv_controller.reconcile_external_memory.assert_called_once_with(0)
    assert working_set.idle_retention_cleared
    assert "floor_bytes=0 evicted_graph_bytes=20 active_graph_bytes=0" in caplog.text


def test_idle_x0_settles_graphs_and_keeps_pinned_floor_before_kv_return():
    runner = object.__new__(mrv2.GPUModelRunner)
    working_set = MagicMock()
    working_set.resident_bytes = 29
    working_set.active_graph_bytes = 0
    working_set.has_pending_staged_hotset_retirement = False
    runner._trim_dynamic_attention_cudagraph_state = MagicMock(return_value=(3, 41))
    runner._clear_elastic_cublas_workspaces = MagicMock(return_value=17)
    runner._measure_elastic_cublas_workspace_bytes = MagicMock(return_value=13)

    assert runner._prepare_dynamic_graph_idle_kv_return(working_set, "x0-17") == 42

    working_set.finish_idle_step.assert_called_once_with()
    working_set.evict_unpinned_for_idle.assert_called_once_with("x0-17")
    runner._trim_dynamic_attention_cudagraph_state.assert_called_once_with(working_set)
    working_set.reconcile_retained_cleanup.assert_called_once_with(41)
    runner._clear_elastic_cublas_workspaces.assert_called_once_with(working_set)


def test_idle_x0_preserves_global_workspace_while_pinned_graph_is_hot():
    runner = object.__new__(mrv2.GPUModelRunner)
    working_set = MagicMock()
    working_set.resident_bytes = 29
    working_set.active_graph_bytes = 7
    working_set.has_pending_staged_hotset_retirement = False
    runner._trim_dynamic_attention_cudagraph_state = MagicMock(return_value=(3, 41))
    runner._clear_elastic_cublas_workspaces = MagicMock(return_value=17)
    runner._measure_elastic_cublas_workspace_bytes = MagicMock(return_value=13)

    assert runner._prepare_dynamic_graph_idle_kv_return(working_set, "x0-18") == 42

    working_set.finish_idle_step.assert_called_once_with()
    working_set.evict_unpinned_for_idle.assert_called_once_with("x0-18")
    working_set.reconcile_retained_cleanup.assert_called_once_with(41)
    runner._clear_elastic_cublas_workspaces.assert_not_called()


def test_synthetic_warmup_does_not_settle_dynamic_graph_lifecycle():
    runner = object.__new__(mrv2.GPUModelRunner)
    runner._finish_dynamic_graph_step = MagicMock(return_value=(31, 7, 13))
    output = MagicMock(spec=ModelRunnerOutput)

    runner._settle_dynamic_graph_step_after_sampling(output, False, None, None)

    runner._finish_dynamic_graph_step.assert_not_called()


def test_live_step_settles_dynamic_graph_lifecycle_after_sampling():
    runner = object.__new__(mrv2.GPUModelRunner)
    runner._elastic_last_step_peak_external_bytes = 41
    runner._finish_dynamic_graph_step = MagicMock(return_value=(31, 7, 13))
    runner._set_elastic_cublas_workspace_unit = MagicMock()
    working_set = MagicMock()
    working_set.hot_snapshot.return_value = (("target",),)
    runner._dynamic_graph_working_set = lambda: working_set
    output = SimpleNamespace(
        elastic_external_memory_bytes=0,
        elastic_external_memory_floor_bytes=0,
        elastic_external_memory_transition_floor_bytes=0,
        elastic_external_memory_peak_bytes=0,
    )

    runner._settle_dynamic_graph_step_after_sampling(output, True, "txn-16", None)

    assert output.elastic_external_memory_bytes == 31
    assert output.elastic_external_memory_floor_bytes == 7
    assert output.elastic_external_memory_transition_floor_bytes == 13
    assert output.elastic_external_memory_peak_bytes == 41
    assert output.elastic_hot_graphs == (("target",),)
    runner._finish_dynamic_graph_step.assert_called_once_with(
        transaction_id="txn-16",
        step_plan=None,
        staged_hotset_consumed=True,
    )


def test_post_sampling_settlement_retains_same_transaction_consumed_hotset(
    monkeypatch,
):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner._elastic_step_measurement_active = False
    runner._elastic_cached_graph_receipt = (31, 7, 13)
    working_set = MagicMock()
    working_set.resident_bytes = 31
    working_set.has_pending_staged_hotset_retirement = True
    working_set.pending_staged_hotset_transaction_id = "maintenance-16"
    runner._dynamic_graph_working_set = lambda: working_set
    synchronize = MagicMock()
    monkeypatch.setattr(mrv2.torch.cuda, "synchronize", synchronize)

    assert runner._finish_dynamic_graph_step(
        transaction_id="maintenance-16",
        staged_hotset_consumed=True,
    ) == (31, 7, 13)

    assert working_set.method_calls[:3] == [
        call.finish_step(),
        call.release_leases("maintenance-16"),
        call.finish_staged_hotset_after_consumers("maintenance-16"),
    ]
    synchronize.assert_called_once_with(runner.device)


def test_post_sampling_settlement_retains_staged_hotset_after_successor(
    monkeypatch,
):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner._elastic_step_measurement_active = False
    runner._elastic_cached_graph_receipt = (31, 7, 13)
    working_set = MagicMock()
    working_set.resident_bytes = 31
    working_set.has_pending_staged_hotset_retirement = True
    working_set.pending_staged_hotset_transaction_id = "maintenance-16"
    runner._dynamic_graph_working_set = lambda: working_set
    synchronize = MagicMock()
    monkeypatch.setattr(mrv2.torch.cuda, "synchronize", synchronize)

    successor = SimpleNamespace(transaction_id="txn-17")
    assert runner._finish_dynamic_graph_step(
        transaction_id="txn-17", step_plan=successor
    ) == (31, 7, 13)
    assert working_set.method_calls[:3] == [
        call.finish_step(),
        call.release_leases("txn-17"),
        call.finish_staged_hotset_after_successor(successor),
    ]
    synchronize.assert_called_once_with(runner.device)


def test_idle_preserve_receipt_keeps_hot_graph_without_finishing_step():
    runner = object.__new__(mrv2.GPUModelRunner)
    runner._elastic_cached_graph_receipt = (125, 35, 50)
    working_set = MagicMock()
    working_set.resident_bytes = 100
    working_set.active_graph_bytes = 70
    working_set.transition_floor_upper_bound_bytes.return_value = 50
    runner._dynamic_graph_working_set = lambda: working_set
    runner._measure_elastic_cublas_workspace_bytes = MagicMock(return_value=20)

    assert runner._current_dynamic_graph_receipt() == (125, 35, 50)
    working_set.finish_dynamic_step.assert_not_called()


def test_next_elastic_step_trims_only_on_real_downward_transition(monkeypatch):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cuda:2")
    controller = MagicMock()
    controller.external_memory_bytes = 200
    controller.apply_scheduler_step.side_effect = lambda _transition, value: value
    runner.elastic_kv_controller = controller
    synchronize = MagicMock()
    empty_cache = MagicMock()
    collect = MagicMock(return_value=0)
    monkeypatch.setattr(mrv2.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(mrv2.torch.accelerator, "empty_cache", empty_cache)
    monkeypatch.setattr(mrv2.gc, "collect", collect)

    # Equal X/loan and growth do not touch allocator caches.
    assert runner._apply_next_elastic_kv_step(None, 200) == 200
    assert runner._apply_next_elastic_kv_step((8, 25), 300) == 300
    synchronize.assert_not_called()
    empty_cache.assert_not_called()
    collect.assert_not_called()

    # The next known smaller step performs exactly one pre-step trim.
    assert runner._apply_next_elastic_kv_step((10, 4), 100) == 100
    assert synchronize.call_count == 2
    empty_cache.assert_called_once_with()
    collect.assert_called_once_with()
    assert runner._elastic_retained_transition_floor_bytes == 0


def test_next_elastic_step_surfaces_retryable_physical_floor(monkeypatch):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cuda:2")
    controller = MagicMock()
    controller.external_memory_bytes = 200
    controller.apply_scheduler_step.return_value = 140
    runner.elastic_kv_controller = controller
    monkeypatch.setattr(mrv2.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(mrv2.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(mrv2.gc, "collect", lambda: 0)

    assert runner._apply_next_elastic_kv_step(None, 100) == 140
    assert runner._elastic_retained_transition_floor_bytes == 40


def test_elastic_cublas_measurement_excludes_startup_baseline(monkeypatch):
    runner = object.__new__(mrv2.GPUModelRunner)
    workspace = 32 << 20
    baseline = {
        101: (workspace, workspace, "(0, 0)", "model"),
        102: (4096, 4096, "(0, 0)", "other"),
    }
    current = {
        **baseline,
        201: (workspace, workspace, "(0, 0)", "cublas"),
        202: (workspace, workspace, "(0, 0)", "cublas"),
        203: (workspace, workspace, "(7, 0)", "graph-private"),
    }
    runner._elastic_cublas_workspace_baseline = frozenset({101})
    runner._elastic_active_allocator_snapshot = lambda: current
    monkeypatch.setattr(
        mrv2.torch._C,
        "_cuda_getCublasWorkspaceSize",
        lambda: workspace,
    )

    assert runner._measure_elastic_cublas_workspace_bytes() == 2 * workspace


def test_elastic_cublas_measurement_requires_pre_step_baseline(monkeypatch):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner._elastic_active_allocator_snapshot = lambda: {}
    monkeypatch.setattr(
        mrv2.torch._C,
        "_cuda_getCublasWorkspaceSize",
        lambda: 32 << 20,
    )

    try:
        runner._measure_elastic_cublas_workspace_bytes()
    except RuntimeError as error:
        assert "baseline was not captured before" in str(error)
    else:
        raise AssertionError("post-step cuBLAS measurement must fail closed")


def test_elastic_step_captures_cublas_baseline_before_measurement(monkeypatch):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cuda:2")
    runner._elastic_cublas_workspace_addresses = MagicMock(
        return_value=frozenset({101})
    )
    monkeypatch.setattr(mrv2.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(mrv2.torch.cuda, "memory_reserved", lambda _device: 11)
    monkeypatch.setattr(mrv2.torch.cuda, "mem_get_info", lambda _device: (22, 33))
    monkeypatch.setattr(
        mrv2.torch.cuda, "reset_peak_memory_stats", lambda _device: None
    )

    runner._begin_elastic_step_measurement()

    assert runner._elastic_cublas_workspace_baseline == frozenset({101})
    assert runner._elastic_step_baseline_reserved_bytes == 11
    assert runner._elastic_step_baseline_driver_free_bytes == 22
    runner._elastic_cublas_workspace_addresses.assert_called_once_with()


def test_elastic_cublas_clear_rejects_live_graphs(monkeypatch):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cuda:2")
    clear = MagicMock()
    monkeypatch.setattr(mrv2.torch._C, "_cuda_clearCublasWorkspaces", clear)

    try:
        runner._clear_elastic_cublas_workspaces(SimpleNamespace(active_graph_bytes=1))
    except RuntimeError as error:
        assert "while dynamic CUDA Graphs are HOT" in str(error)
    else:
        raise AssertionError("live Graph workspaces must not be cleared")
    clear.assert_not_called()


def test_elastic_cublas_clear_reclaims_only_after_graph_eviction(monkeypatch):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cuda:2")
    runner._elastic_cublas_workspace_baseline = frozenset({101})
    runner._elastic_cublas_workspace_addresses = MagicMock(
        return_value=frozenset({301})
    )
    clear = MagicMock()
    free_bytes = iter((100, 180))
    monkeypatch.setattr(mrv2.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(
        mrv2.torch.accelerator,
        "get_memory_info",
        lambda: (next(free_bytes), 1000),
    )
    monkeypatch.setattr(mrv2.torch._C, "_cuda_clearCublasWorkspaces", clear)
    monkeypatch.setattr(mrv2.gc, "collect", lambda: 0)
    monkeypatch.setattr(mrv2.torch.accelerator, "empty_cache", lambda: None)

    reclaimed = runner._clear_elastic_cublas_workspaces(
        SimpleNamespace(active_graph_bytes=0)
    )

    assert reclaimed == 80
    assert runner._elastic_cublas_workspace_baseline == frozenset({301})
    runner._elastic_cublas_workspace_addresses.assert_called_once_with()
    clear.assert_called_once_with()


def test_dynamic_attention_graph_state_keeps_request_and_token_key_domains(
    monkeypatch,
):
    runner = object.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cuda:2")
    target_builder = MagicMock()
    draft_builder = MagicMock()
    target_group = MagicMock()
    target_group.get_metadata_builder.return_value = target_builder
    draft_group = MagicMock()
    draft_group.get_metadata_builder.return_value = draft_builder
    runner.attn_groups = [[target_group]]
    runner.speculator = SimpleNamespace(attn_groups=[[draft_group]])
    working_set = SimpleNamespace(
        hot_request_counts=frozenset({8, 12}),
        hot_token_counts=frozenset({64, 128}),
    )
    events = []
    target_builder.has_stale_dynamic_cudagraph_wrappers.return_value = True
    draft_builder.has_stale_dynamic_cudagraph_wrappers.return_value = False
    target_builder.trim_dynamic_cudagraph_wrappers.return_value = 2
    draft_builder.trim_dynamic_cudagraph_wrappers.return_value = 1
    target_builder.trim_dynamic_cudagraph_wrappers.side_effect = lambda **_kwargs: (
        events.append("target_trim") or 2
    )
    draft_builder.trim_dynamic_cudagraph_wrappers.side_effect = lambda **_kwargs: (
        events.append("draft_trim") or 1
    )
    monkeypatch.setattr(
        mrv2.torch.cuda,
        "synchronize",
        lambda _device: events.append("synchronize"),
    )
    monkeypatch.setattr(mrv2.torch.accelerator, "empty_cache", lambda: None)
    memory_info = iter(((100, 200), (130, 200)))
    monkeypatch.setattr(
        mrv2.torch.accelerator, "get_memory_info", lambda: next(memory_info)
    )

    assert runner._trim_dynamic_attention_cudagraph_state(working_set) == (3, 30)
    assert events[:3] == ["synchronize", "target_trim", "draft_trim"]
    assert events.count("synchronize") == 2
    target_builder.trim_dynamic_cudagraph_wrappers.assert_called_once_with(
        keep_request_batch_sizes=frozenset({8, 12}),
        keep_token_batch_sizes=frozenset({64, 128}),
    )
    draft_builder.trim_dynamic_cudagraph_wrappers.assert_called_once_with(
        keep_request_batch_sizes=frozenset({8, 12}),
        keep_token_batch_sizes=frozenset({64, 128}),
    )


def test_elastic_active_allocator_snapshot_keeps_only_integer_metadata(monkeypatch):
    runner = object.__new__(mrv2.GPUModelRunner)
    monkeypatch.setattr(
        mrv2.torch.cuda,
        "memory_snapshot",
        lambda: [
            {
                "address": 1000,
                "segment_pool_id": (7, 9),
                "blocks": [
                    {
                        "size": 128,
                        "requested_size": 96,
                        "state": "active_allocated",
                        "frames": [
                            {"filename": "owner.py", "line": 12, "name": "make"}
                        ],
                    },
                    {"size": 64, "state": "inactive"},
                    {
                        "size": 32,
                        "requested_size": 24,
                        "state": "active_allocated",
                    },
                ],
            }
        ],
    )

    assert runner._elastic_active_allocator_snapshot() == {
        1000: (128, 96, "(7, 9)", "owner.py:12:make"),
        1192: (32, 24, "(7, 9)", ""),
    }


def test_empty_output_preserves_measured_external_memory_floor():
    output = ModelRunnerOutput.with_kv_conn_output_only(None, 17, 11)

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert output.elastic_external_memory_bytes == 17
    assert output.elastic_external_memory_floor_bytes == 11


class FakeMemoryProfiler:
    def __enter__(self):
        self.consumed_memory = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEplbState:
    instances: list["FakeEplbState"] = []
    from_mapping_kwargs: dict[str, Any] | None = None

    def __init__(self, parallel_config: Any, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.add_model_calls: list[tuple[Any, Any]] = []
        self.step_calls: list[tuple[bool, bool, bool]] = []
        self.async_started = False
        self.is_async = True
        self.built_from_mapping = False
        FakeEplbState.instances.append(self)

    def add_model(self, model: Any, model_config: Any) -> None:
        self.add_model_calls.append((model, model_config))

    def step(self, is_dummy: bool, is_profile: bool, *, log_stats: bool) -> None:
        self.step_calls.append((is_dummy, is_profile, log_stats))

    def start_async_loop(self) -> None:
        self.async_started = True

    @classmethod
    def from_mapping(cls, **kwargs: Any) -> "FakeEplbState":
        cls.from_mapping_kwargs = kwargs
        state = cls(kwargs["parallel_config"], kwargs["device"])
        state.built_from_mapping = True
        return state


def _make_runner(**overrides: Any) -> Any:
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.model_config = SimpleNamespace(model="test-model")
    runner.load_config = SimpleNamespace(load_format="hf")
    runner.parallel_config = SimpleNamespace(
        enable_eplb=True,
        enable_elastic_ep=False,
        eplb_config=SimpleNamespace(log_balancedness=True),
    )
    runner.vllm_config = SimpleNamespace(
        load_config=runner.load_config,
        model_config=runner.model_config,
        offload_config=OffloadConfig(),
    )
    runner.lora_config = None
    runner.aux_hidden_trace = SimpleNamespace(enabled=False)
    runner.use_aux_hidden_state_outputs = False
    runner.speculative_config = None
    runner.speculator = None
    runner.num_speculative_steps = 0
    runner.encoder_cache = None
    runner.is_pooling_model = False
    runner.is_last_pp_rank = True
    runner.is_first_pp_rank = True
    runner.max_num_reqs = 8
    runner.max_num_tokens = 16
    runner.decode_query_len = 1
    runner.kv_connector = SimpleNamespace(
        set_disabled=lambda *_: None,
        post_forward=lambda *_, **__: None,
    )
    runner.eplb = eplb.EPLBController(runner.parallel_config, runner.device)
    runner.pooling_runner = None
    runner.execute_model_state = None
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def test_v2_load_model_registers_moe_with_eplb(monkeypatch):
    FakeEplbState.instances.clear()
    model = SimpleNamespace(is_moe=True)

    monkeypatch.setattr(mrv2, "DeviceMemoryProfiler", FakeMemoryProfiler)
    monkeypatch.setattr(eplb, "EplbState", FakeEplbState)
    monkeypatch.setattr(
        mrv2,
        "get_model_loader",
        lambda load_config: SimpleNamespace(load_model=lambda **_: model),
    )
    monkeypatch.setattr(
        mrv2,
        "init_model_state",
        lambda *args: SimpleNamespace(num_new_sampled_tokens_per_step=1),
    )
    monkeypatch.setattr(
        eplb,
        "get_mixture_of_experts_model",
        lambda loaded_model: (
            loaded_model if getattr(loaded_model, "is_moe", False) else None
        ),
    )

    runner = _make_runner(is_last_pp_rank=False)
    mrv2.GPUModelRunner.load_model(runner)

    assert runner.model is model
    assert runner.model_state is not None
    assert runner.eplb_state is not None
    assert runner.eplb_state.add_model_calls == [(model, runner.model_config)]
    assert runner.eplb_state.async_started is True


def test_v2_load_model_with_dummy_weights_skips_eplb_registration(monkeypatch):
    FakeEplbState.instances.clear()
    model = SimpleNamespace(is_moe=True)

    monkeypatch.setattr(mrv2, "DeviceMemoryProfiler", FakeMemoryProfiler)
    monkeypatch.setattr(eplb, "EplbState", FakeEplbState)
    monkeypatch.setattr(
        mrv2,
        "get_model_loader",
        lambda load_config: SimpleNamespace(load_model=lambda **_: model),
    )
    monkeypatch.setattr(
        mrv2,
        "init_model_state",
        lambda *args: SimpleNamespace(num_new_sampled_tokens_per_step=1),
    )
    monkeypatch.setattr(eplb, "get_mixture_of_experts_model", lambda model: model)

    runner = _make_runner(is_last_pp_rank=False)
    mrv2.GPUModelRunner.load_model(runner, load_dummy_weights=True)

    assert runner.load_config.load_format == "dummy"
    assert runner.eplb_state is not None
    assert runner.eplb_state.add_model_calls == []
    assert runner.eplb_state.async_started is False


def test_v2_setup_eplb_from_mapping_rebuilds_state(monkeypatch):
    FakeEplbState.instances.clear()
    FakeEplbState.from_mapping_kwargs = None
    monkeypatch.setattr(eplb, "EplbState", FakeEplbState)
    monkeypatch.setattr(eplb, "get_mixture_of_experts_model", lambda model: model)

    runner = _make_runner(model=SimpleNamespace(is_moe=True))
    mapping = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    mrv2.GPUModelRunner.setup_eplb_from_mapping(runner, mapping, 2)

    assert runner.eplb_state is not None
    assert runner.eplb_state.built_from_mapping is True
    assert FakeEplbState.from_mapping_kwargs is not None
    assert FakeEplbState.from_mapping_kwargs["expanded_physical_to_logical"] is mapping
    assert FakeEplbState.from_mapping_kwargs["num_valid_physical_experts"] == 2


def test_v2_sample_tokens_runs_eplb_on_non_last_pp_rank(monkeypatch):
    events = []
    runner = _make_runner(is_last_pp_rank=False, num_speculative_steps=0)
    runner.execute_model_state = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=2, idx_mapping=torch.zeros(2, dtype=torch.int32)
        ),
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        finished_req_ids=set(),
        routed_experts=None,
        num_tokens_across_dp=None,
        num_spec_tokens_to_schedule=0,
        gdn_checkpoint_keys=None,
        elastic_external_memory_bytes=0,
        elastic_external_memory_floor_bytes=0,
        elastic_mm_activation_loan_bytes=0,
        elastic_dynamic_graph_step_started=False,
        elastic_transaction_id=None,
        is_synthetic_warmup=False,
    )
    runner.req_states = SimpleNamespace()

    def fake_receive(*args, **kwargs):
        events.append("receive")
        # all_decode_next=True, so model_state.postprocess_state is skipped.
        return True

    runner.pp_handler = SimpleNamespace(receive=fake_receive)
    runner.postprocess_num_computed_tokens = lambda *args, **kwargs: events.append(
        "postprocess_num_computed_tokens"
    )
    runner.eplb.step = lambda *args, **kwargs: events.append("eplb")

    output = mrv2.GPUModelRunner.sample_tokens(runner, None)
    assert output in (EMPTY_MODEL_RUNNER_OUTPUT, None)
    assert events == ["receive", "postprocess_num_computed_tokens", "eplb"]
