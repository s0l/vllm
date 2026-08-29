# SPDX-License-Identifier: Apache-2.0
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm.v1.core.elastic_graph import (
    ElasticGraphCache,
    ElasticPlanKind,
    GraphExecutionPolicy,
    OwnerGraphExecutionPolicy,
    RuntimeGeneration,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore, _ElasticCalibrationPartialWave
from vllm.v1.worker import startup_plan

pytestmark = pytest.mark.cpu_test


def _current_q4_piecewise_policy() -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="batched-causal-q1-v1",
        math_contract="pending-batched-q1-product-math-v1",
        owners=(
            OwnerGraphExecutionPolicy("mtp_decode", (1,), "PIECEWISE"),
            OwnerGraphExecutionPolicy("mtp_prefill", (), "PIECEWISE"),
            OwnerGraphExecutionPolicy(
                "target", (1,), "PIECEWISE", piecewise_query_len_min_tokens=32
            ),
        ),
    )


def _approved_q4_full_policy() -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="approved-q4-full-control-v1",
        math_contract="approved-q4-full-math-v1",
        owners=(
            OwnerGraphExecutionPolicy("mtp_decode", (1,), "PIECEWISE"),
            OwnerGraphExecutionPolicy("mtp_prefill", (), "PIECEWISE"),
            OwnerGraphExecutionPolicy("target", (1, 4), "PIECEWISE"),
        ),
    )


def test_elastic_piecewise_boundaries_derive_from_runtime_b():
    assert EngineCore._elastic_piecewise_boundaries(1) == (1,)
    assert EngineCore._elastic_piecewise_boundaries(4) == (1, 2, 4)
    assert EngineCore._elastic_piecewise_boundaries(6) == (1, 2, 4, 6)


def test_calibration_physical_form_plan_is_ascending_and_terminal_last():
    inventory = {
        x: (
            SimpleNamespace(
                physical_num_reqs=x,
                logical=SimpleNamespace(
                    owner="parallel_draft",
                    mode="FULL",
                    token_bucket=x * 7,
                    uniform_query_len=7,
                ),
            ),
        )
        for x in (1, 2, 4, 7)
    }

    plan = EngineCore._elastic_calibration_physical_form_plan(inventory, 7)

    assert tuple(x for x, _keys in plan) == (1, 2, 4, 7)
    assert tuple(keys[0].logical.token_bucket for _x, keys in plan) == (
        7,
        14,
        28,
        49,
    )


@pytest.mark.parametrize(
    ("inventory", "boundary_x", "match"),
    [
        ({1: (), 7: (), 4: ()}, 7, "ascending terminal"),
        ({1: (), 2: ()}, 4, "ascending terminal"),
        (
            {1: (SimpleNamespace(physical_num_reqs=2),)},
            1,
            "lost cohort identity",
        ),
    ],
)
def test_calibration_physical_form_plan_rejects_invalid_identity(
    inventory, boundary_x, match
):
    with pytest.raises(RuntimeError, match=match):
        EngineCore._elastic_calibration_physical_form_plan(inventory, boundary_x)


def test_bounded_calibration_reuses_cold_complete_piecewise_row():
    key = (0, 3, 32, 128, 4)
    row = {
        "cold_peak_bytes": 64,
        "hot_peak_bytes": 0,
        "resident_bytes": 32,
        "floor_bytes": 0,
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 0,
        "hot_stable_replays": 0,
    }

    assert EngineCore._elastic_calibration_catalog_row_stable(
        key, row, bounded_exact_hotset=True
    )
    assert not EngineCore._elastic_calibration_catalog_row_stable(
        key, row, bounded_exact_hotset=False
    )


def test_elastic_mixed_prefill_inventory_excludes_impossible_packed_shapes():
    reachable = EngineCore._elastic_mixed_prefill_reachable

    assert reachable(x=2, m=2, query_len=1)
    assert not reachable(x=2, m=2, query_len=4)
    assert not reachable(x=1, m=4096, query_len=1)
    assert reachable(x=18, m=4096, query_len=4)


def test_sealed_serving_restart_restores_every_full_shape_before_ready():
    core = object.__new__(EngineCore)
    required = [
        [1, 3, x, x * query_len, query_len]
        for query_len in (1, 4)
        for x in (1, 2)
    ]
    scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={
            "mixed_max_x": 2,
            "required_step_keys": required,
        },
        num_spec_tokens=3,
        _elastic_calibration_mode=False,
        max_num_running_reqs=64,
    )
    core.scheduler = scheduler
    calls = []

    def restore_wave(*, k, x, query_len, serial):
        calls.append((k, x, query_len, serial))
        return (1, k, x, x * query_len, query_len), x

    core._run_elastic_full_calibration_wave = restore_wave

    core._restore_elastic_pinned_full_family()

    assert calls == [
        (3, 1, 4, 1),
        (3, 1, 1, 2),
        (3, 2, 4, 3),
        (3, 2, 1, 4),
    ]
    assert scheduler.max_num_running_reqs == 2
    assert scheduler._elastic_calibration_mode is False


def test_sealed_bounded_restart_restores_only_minimal_carrier_pair():
    core = object.__new__(EngineCore)
    piecewise = [0, 0, 1, 2, 0]
    full = [1, 3, 1, 1, 1]
    scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={
            "representation": "bounded_exact_hotset",
            "mixed_max_x": 40,
            "required_step_keys": [piecewise, full, [1, 3, 40, 40, 1]],
            "restore_step_keys": [piecewise, full],
        },
        num_spec_tokens=3,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                disable_speculation_on_non_decode=True
            )
        ),
        _elastic_calibration_mode=False,
        _elastic_restore_mode=False,
        max_num_running_reqs=64,
    )
    core.scheduler = scheduler
    core._run_elastic_full_calibration_wave = MagicMock(
        return_value=((1, 3, 1, 1, 1), 1)
    )

    core._restore_elastic_bounded_hotset()

    core._run_elastic_full_calibration_wave.assert_called_once_with(
        k=3, x=1, query_len=1, serial=1
    )
    assert scheduler.max_num_running_reqs == 40
    assert scheduler._elastic_calibration_mode is False
    assert scheduler._elastic_restore_mode is False


def test_sealed_bounded_restart_rejects_wrong_phase_split_carrier_k():
    core = object.__new__(EngineCore)
    piecewise = [0, 3, 1, 2, 0]
    full = [1, 3, 1, 1, 1]
    scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={
            "representation": "bounded_exact_hotset",
            "mixed_max_x": 40,
            "required_step_keys": [piecewise, full],
            "restore_step_keys": [piecewise, full],
        },
        num_spec_tokens=3,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                disable_speculation_on_non_decode=True
            )
        ),
    )
    core.scheduler = scheduler

    with pytest.raises(RuntimeError, match="expected_piecewise_k=0"):
        core._restore_elastic_bounded_hotset()


def test_sealed_bounded_restart_rejects_non_pair_restore_subset():
    core = object.__new__(EngineCore)
    scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={
            "representation": "bounded_exact_hotset",
            "mixed_max_x": 40,
            "required_step_keys": [[1, 3, 1, 1, 1]],
            "restore_step_keys": [[1, 3, 1, 1, 1]],
        },
        num_spec_tokens=3,
    )
    core.scheduler = scheduler

    with pytest.raises(RuntimeError, match="exactly one FULL/PIECEWISE"):
        core._restore_elastic_bounded_hotset()


def test_partial_active_calibration_cohort_is_cancelled_before_model_execution():
    core = object.__new__(EngineCore)
    scheduler_output = SimpleNamespace(
        is_pure_decode_step=True,
        num_scheduled_tokens={"a": 4},
    )
    scheduler = SimpleNamespace(
        has_requests=MagicMock(return_value=True),
        schedule=MagicMock(return_value=scheduler_output),
        cancel_unexecuted_elastic_calibration_step=MagicMock(),
    )
    core.scheduler = scheduler
    core.model_executor = MagicMock()
    core._ag2_step_trace_path = ""
    core._ag2_step_trace_complete = False
    core._elastic_calibration_expected_decode_ids = frozenset(("a", "b"))

    with pytest.raises(_ElasticCalibrationPartialWave) as error:
        core.step()

    assert error.value.admitted_x == 1
    scheduler.cancel_unexecuted_elastic_calibration_step.assert_called_once_with(
        scheduler_output
    )
    core.model_executor.execute_model.assert_not_called()


def test_calibration_request_crosses_normal_preprocess_boundary():
    core = object.__new__(EngineCore)
    scheduled_request = object()
    core.preprocess_add_request = MagicMock(return_value=(scheduled_request, 7))
    core.add_request = MagicMock()
    core.scheduler = SimpleNamespace(num_spec_tokens=3)

    core._add_elastic_calibration_request(
        request_id="calibration-request",
        prompt_len=2,
        k=3,
        max_tokens=16,
    )

    core.preprocess_add_request.assert_called_once()
    engine_request = core.preprocess_add_request.call_args.args[0]
    assert engine_request.request_id == "calibration-request"
    assert len(engine_request.prompt_token_ids) == 2
    assert engine_request.sampling_params.max_tokens == 16
    core.add_request.assert_called_once_with(scheduled_request, 7)


def test_failed_calibration_explicitly_shuts_down_partial_engine():
    core = object.__new__(EngineCore)
    core.shutdown = MagicMock()

    core._shutdown_failed_elastic_calibration()

    core.shutdown.assert_called_once_with()


def test_failed_calibration_shutdown_does_not_mask_original_failure():
    core = object.__new__(EngineCore)
    core.shutdown = MagicMock(side_effect=RuntimeError("teardown failed"))

    core._shutdown_failed_elastic_calibration()

    core.shutdown.assert_called_once_with()


def test_post_kv_scheduler_generation_is_authoritative_on_workers():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        elastic_on_demand_graphs=True,
        _elastic_runtime_generation=SimpleNamespace(value="post-kv-generation"),
    )
    core.collective_rpc = MagicMock(
        return_value=["post-kv-generation", "post-kv-generation"]
    )

    core._synchronize_elastic_runtime_generation()

    core.collective_rpc.assert_called_once_with(
        "set_elastic_runtime_generation", args=("post-kv-generation",)
    )


def test_post_kv_scheduler_generation_rejects_worker_divergence():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        elastic_on_demand_graphs=True,
        _elastic_runtime_generation=SimpleNamespace(value="scheduler-generation"),
    )
    core.collective_rpc = MagicMock(
        return_value=["scheduler-generation", "stale-worker-generation"]
    )

    with pytest.raises(RuntimeError, match="did not accept"):
        core._synchronize_elastic_runtime_generation()


def test_static_runtime_does_not_rebind_worker_generation():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        elastic_on_demand_graphs=False,
        _elastic_runtime_generation=SimpleNamespace(value="unused-generation"),
    )
    core.collective_rpc = MagicMock()

    core._synchronize_elastic_runtime_generation()

    core.collective_rpc.assert_not_called()


def _elastic_hot_row(generation: str, resident_bytes: int = 64):
    return (
        "owner:key",
        "target",
        "FULL",
        1,
        1,
        1,
        (),
        generation,
        True,
        resident_bytes,
        resident_bytes,
        resident_bytes,
        0,
    )


def test_startup_residency_publishes_workspace_before_first_plan():
    core = object.__new__(EngineCore)
    generation = "post-kv-generation"
    snapshots = [
        (_elastic_hot_row(generation, 64),),
        (_elastic_hot_row(generation, 64),),
        (_elastic_hot_row(generation, 64),),
    ]
    workspace_receipts = [
        (generation, 32),
        (generation, 48),
        (generation, 32),
    ]
    scheduler = SimpleNamespace(
        _elastic_runtime_generation=SimpleNamespace(value=generation),
        _elastic_cublas_workspace_unit_bytes=0,
        _sync_elastic_hot_graphs=MagicMock(),
        _publish_elastic_startup_capacity=MagicMock(),
    )
    core.scheduler = scheduler
    core.collective_rpc = MagicMock(
        side_effect=[snapshots, workspace_receipts]
    )

    core._synchronize_elastic_startup_residency()

    assert core.collective_rpc.call_args_list == [
        (("get_elastic_graph_hot_snapshot",),),
        (("get_elastic_graph_workspace_receipt",),),
    ]
    scheduler._sync_elastic_hot_graphs.assert_called_once_with(snapshots[0])
    scheduler._publish_elastic_startup_capacity.assert_called_once_with()
    assert scheduler._elastic_cublas_workspace_unit_bytes == 48


def test_startup_workspace_receipt_changes_saved_physical_cold_grant_once():
    core = object.__new__(EngineCore)
    generation = RuntimeGeneration("post-kv-generation")
    step_key = (1, 3, 1, 4, 4)
    scheduler = object.__new__(Scheduler)
    scheduler._elastic_runtime_generation = generation
    scheduler._elastic_graph_execution_policy = _approved_q4_full_policy()
    scheduler._elastic_calibration_mode = False
    scheduler._elastic_graph_cache = ElasticGraphCache(generation)
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=4096)
    scheduler._elastic_graph_step_key = step_key
    scheduler._elastic_graph_recapture_pending_key = None
    scheduler._elastic_pending_graph_loans = deque()
    scheduler._elastic_graph_capture_envelopes = {
        step_key: (297795584, 0, 0)
    }
    scheduler._elastic_graph_measured_bytes = {}
    scheduler._elastic_graph_floor_bytes = 0
    scheduler._elastic_graph_transition_floor_bytes = 0
    scheduler._elastic_graph_resident_bytes = 244056064
    scheduler._elastic_cublas_workspace_unit_bytes = 0
    scheduler._sync_elastic_hot_graphs = MagicMock()
    scheduler._publish_elastic_startup_capacity = MagicMock()
    core.scheduler = scheduler
    snapshots = [
        (_elastic_hot_row(generation.value),),
        (_elastic_hot_row(generation.value),),
        (_elastic_hot_row(generation.value),),
    ]
    core.collective_rpc = MagicMock(
        side_effect=[
            snapshots,
            [(generation.value, 33554432)] * 3,
        ]
    )

    # This is the exact stale live grant before startup publishes the unit.
    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        297795584,
        True,
    )

    core._synchronize_elastic_startup_residency()

    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        331350016,
        True,
    )
    scheduler._publish_elastic_startup_capacity.assert_called_once_with()


@pytest.mark.parametrize(
    "workspace_receipts,match",
    [
        (
            [("post-kv-generation", 32), ("stale-generation", 32)],
            "generation",
        ),
        (
            [("post-kv-generation", 32), ("post-kv-generation", 0)],
            "positive",
        ),
        (
            [("post-kv-generation", 32), ("post-kv-generation", -1)],
            "positive",
        ),
        (
            [("post-kv-generation", 32), ("malformed",)],
            "malformed",
        ),
    ],
)
def test_startup_workspace_receipt_fails_before_residency_mutation(
    workspace_receipts, match
):
    core = object.__new__(EngineCore)
    generation = "post-kv-generation"
    snapshots = [
        (_elastic_hot_row(generation),),
        (_elastic_hot_row(generation),),
    ]
    scheduler = SimpleNamespace(
        _elastic_runtime_generation=SimpleNamespace(value=generation),
        _elastic_cublas_workspace_unit_bytes=17,
        _sync_elastic_hot_graphs=MagicMock(),
    )
    core.scheduler = scheduler
    core.collective_rpc = MagicMock(
        side_effect=[snapshots, workspace_receipts]
    )

    with pytest.raises(RuntimeError, match=match):
        core._synchronize_elastic_startup_residency()

    scheduler._sync_elastic_hot_graphs.assert_not_called()
    assert scheduler._elastic_cublas_workspace_unit_bytes == 17


def test_calibration_cold_epoch_is_explicit_and_restores_idle_policy():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(_elastic_force_idle_cleanup=False)

    with core._elastic_calibration_cold_epoch():
        assert core.scheduler._elastic_force_idle_cleanup is True

    assert core.scheduler._elastic_force_idle_cleanup is False


def test_full_calibration_probe_downshifts_before_any_model_execution():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        _elastic_graph_resident_bytes=256,
        _canonical_elastic_graph_step_key=MagicMock(
            return_value=(0, 3, 2, 4, 0)
        ),
        assert_elastic_calibration_captures_hot=MagicMock(),
    )
    core._add_elastic_calibration_request = MagicMock()
    core._prepare_elastic_calibration_capture = MagicMock()
    core._run_elastic_calibration_step = MagicMock()
    core._prepare_elastic_calibration_admission = MagicMock(return_value=1)
    core._reclaim_elastic_calibration_hotset_before_wave = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()

    key, feasible_x = core._run_elastic_full_calibration_wave(
        k=3,
        x=2,
        query_len=4,
        serial=1,
        discover_max_x=True,
    )

    assert key is None
    assert feasible_x == 1
    assert [
        invocation.args[0]
        for invocation in core._prepare_elastic_calibration_capture.call_args_list
    ] == [
        (0, 3, 2, 4, 0),
        (1, 3, 2, 8, 4),
    ]
    core._run_elastic_calibration_step.assert_not_called()
    core._prepare_elastic_calibration_admission.assert_called_once_with(
        [
            "_elastic_cal_full_1_3_4_2_0",
            "_elastic_cal_full_1_3_4_2_1",
        ],
        retain_hot_graphs=True,
    )
    core.abort_requests.assert_called_once()
    core._drain_elastic_calibration.assert_called_once_with()
    assert core._reclaim_elastic_calibration_hotset_before_wave.call_count == 2
    assert core._reclaim_elastic_calibration_hotset_before_wave.call_args_list == [
        call(((0, 3, 2, 4, 0), (1, 3, 2, 8, 4))),
        call(((0, 3, 2, 4, 0), (1, 3, 2, 8, 4))),
    ]
    assert core.scheduler._elastic_force_idle_cleanup is False


def test_full_calibration_prepares_generic_prefill_and_decode_before_admission():
    core = object.__new__(EngineCore)
    prefill_key = (0, 5, 3, 8, 0)
    decode_key = (0, 5, 2, 12, 6)
    events = []
    scheduler = SimpleNamespace(
        requests={},
        _elastic_graph_resident_bytes=384,
        _elastic_graph_execution_policy=SimpleNamespace(
            verifier_contract="batched-causal-q1-v1"
        ),
        _canonical_elastic_graph_step_key=MagicMock(return_value=prefill_key),
        assert_elastic_calibration_captures_hot=MagicMock(),
    )
    core.scheduler = scheduler
    core._add_elastic_calibration_request = MagicMock(
        side_effect=lambda **_: events.append("add")
    )
    core._prepare_elastic_calibration_capture = MagicMock(
        side_effect=lambda key: events.append(("capture", key))
    )
    core._prepare_elastic_calibration_admission = MagicMock(
        side_effect=lambda *_args, **_kwargs: events.append("admit") or 2
    )
    prefill = SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 2})
    core._run_elastic_calibration_step = MagicMock(
        side_effect=[prefill, _ElasticCalibrationPartialWave(1, 2)]
    )
    core._reclaim_elastic_calibration_hotset_before_wave = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()
    scheduler.finish_unexecuted_elastic_calibration_step = MagicMock()

    key, feasible_x = core._run_elastic_full_calibration_wave_in_epoch(
        k=5,
        x=2,
        query_len=6,
        serial=7,
        discover_max_x=True,
    )

    assert key is None
    assert feasible_x == 1
    assert events == [
        ("capture", prefill_key),
        ("capture", decode_key),
        "add",
        "add",
        "admit",
    ]


def test_full_calibration_executes_idle_reclaim_before_declared_wave():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        requests={},
        _elastic_graph_resident_bytes=256,
        _canonical_elastic_graph_step_key=MagicMock(
            return_value=(1, 0, 1, 2, 0)
        ),
        assert_elastic_calibration_captures_hot=MagicMock(),
    )
    boundary_events = []
    core._add_elastic_calibration_request = MagicMock(
        side_effect=lambda **_: boundary_events.append("request")
    )
    core._reclaim_elastic_calibration_hotset_before_wave = MagicMock(
        side_effect=lambda _keys=(): boundary_events.append("reclaim")
    )
    core._prepare_elastic_calibration_admission = MagicMock(return_value=1)
    core._prepare_elastic_calibration_capture = MagicMock()
    prefill = SimpleNamespace(num_scheduled_tokens={"a": 2})
    cold = SimpleNamespace(num_scheduled_tokens={"a": 1})
    hot = SimpleNamespace(num_scheduled_tokens={"a": 1})
    core._run_elastic_calibration_step = MagicMock(
        side_effect=[prefill, cold, hot]
    )
    core._assert_elastic_calibration_key = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()

    key, feasible_x = core._run_elastic_full_calibration_wave_in_epoch(
        k=0,
        x=1,
        query_len=1,
        serial=1,
    )

    assert key == (1, 0, 1, 1, 1)
    assert feasible_x == 1
    core._reclaim_elastic_calibration_hotset_before_wave.assert_called_once_with(
        ((1, 0, 1, 2, 0), (1, 0, 1, 1, 1))
    )
    assert boundary_events == ["reclaim", "request"]
    core._prepare_elastic_calibration_admission.assert_called_once_with(
        ["_elastic_cal_full_1_0_1_1_0"],
        retain_hot_graphs=True,
    )
    assert core._run_elastic_calibration_step.call_count == 3


def test_idle_reclaim_completes_before_any_wave_request_is_added():
    core = object.__new__(EngineCore)
    reclaim = SimpleNamespace(kind=ElasticPlanKind.RECLAIM)
    core.scheduler = SimpleNamespace(
        prepare_elastic_calibration_idle_reclaim=MagicMock(return_value=True)
    )
    core._run_elastic_calibration_step = MagicMock(
        return_value=SimpleNamespace(
            total_num_scheduled_tokens=0,
            elastic_step_plan=reclaim,
        )
    )

    core._reclaim_elastic_calibration_hotset_before_wave()

    core.scheduler.prepare_elastic_calibration_idle_reclaim.assert_called_once_with(
        ()
    )
    core._run_elastic_calibration_step.assert_called_once_with()


def test_calibration_cold_capture_is_request_free_and_identity_bound():
    core = object.__new__(EngineCore)
    step_key = (0, 2, 11, 32, 0)
    physical_keys = ("target-piecewise-32", "mtp-prefill-piecewise-32")
    plan = SimpleNamespace(
        kind=ElasticPlanKind.MAINTENANCE,
        physical_keys=physical_keys,
    )
    scheduler = SimpleNamespace(
        prepare_elastic_calibration_capture=MagicMock(return_value=True),
        _elastic_graph_carrier_closure_step_key=MagicMock(return_value=step_key),
        _resolve_elastic_step_physical_keys=MagicMock(return_value=physical_keys),
    )
    core.scheduler = scheduler
    core._run_elastic_calibration_step = MagicMock(
        return_value=SimpleNamespace(
            total_num_scheduled_tokens=0,
            elastic_step_plan=plan,
        )
    )

    core._prepare_elastic_calibration_capture(step_key)

    scheduler.prepare_elastic_calibration_capture.assert_called_once_with(step_key)


def test_calibration_cold_capture_rejects_request_commit():
    core = object.__new__(EngineCore)
    step_key = (1, 4, 3, 15, 5)
    physical_keys = ("target-full-3x5",)
    core.scheduler = SimpleNamespace(
        prepare_elastic_calibration_capture=MagicMock(return_value=True),
        _elastic_graph_carrier_closure_step_key=MagicMock(return_value=step_key),
        _resolve_elastic_step_physical_keys=MagicMock(return_value=physical_keys),
    )
    core._run_elastic_calibration_step = MagicMock(
        return_value=SimpleNamespace(
            total_num_scheduled_tokens=1,
            elastic_step_plan=SimpleNamespace(
                kind=ElasticPlanKind.MAINTENANCE,
                physical_keys=physical_keys,
            ),
        )
    )

    with pytest.raises(RuntimeError, match="crossed a request commit boundary"):
        core._prepare_elastic_calibration_capture(step_key)


def test_calibration_cold_capture_rejects_physical_owner_drift():
    core = object.__new__(EngineCore)
    step_key = (0, 5, 7, 64, 0)
    core.scheduler = SimpleNamespace(
        prepare_elastic_calibration_capture=MagicMock(return_value=True),
        _elastic_graph_carrier_closure_step_key=MagicMock(return_value=step_key),
        _resolve_elastic_step_physical_keys=MagicMock(
            return_value=("expected-owner",)
        ),
    )
    core._run_elastic_calibration_step = MagicMock(
        return_value=SimpleNamespace(
            total_num_scheduled_tokens=0,
            elastic_step_plan=SimpleNamespace(
                kind=ElasticPlanKind.MAINTENANCE,
                physical_keys=("different-owner",),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="changed physical owner identity"):
        core._prepare_elastic_calibration_capture(step_key)


def test_engine_step_work_includes_request_free_elastic_maintenance():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        has_requests=MagicMock(return_value=False),
        has_pending_elastic_maintenance=MagicMock(return_value=True),
    )

    assert core._has_scheduler_step_work() is True

    core.scheduler.has_pending_elastic_maintenance.return_value = False
    assert core._has_scheduler_step_work() is False


def test_full_calibration_probe_downshifts_after_pinned_family_contraction():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        requests={},
        _elastic_graph_resident_bytes=256,
        _canonical_elastic_graph_step_key=MagicMock(
            return_value=(0, 3, 2, 4, 0)
        ),
        assert_elastic_calibration_captures_hot=MagicMock(),
    )
    core._add_elastic_calibration_request = MagicMock()
    core._prepare_elastic_calibration_capture = MagicMock()
    prefill = SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 2})
    cold = SimpleNamespace(num_scheduled_tokens={"a": 4})
    core._run_elastic_calibration_step = MagicMock(side_effect=[prefill, cold])
    core._prepare_elastic_calibration_admission = MagicMock(return_value=2)
    core._reclaim_elastic_calibration_hotset_before_wave = MagicMock()
    core._assert_elastic_calibration_key = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()

    key, feasible_x = core._run_elastic_full_calibration_wave(
        k=3,
        x=2,
        query_len=4,
        serial=1,
    )

    assert key is None
    assert feasible_x == 1
    core._assert_elastic_calibration_key.assert_not_called()
    core.abort_requests.assert_called_once()
    core._drain_elastic_calibration.assert_called_once_with()
    assert core.scheduler._elastic_force_idle_cleanup is False


def test_full_calibration_rejects_scheduler_above_wave_preflight():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        _elastic_graph_resident_bytes=256,
        _canonical_elastic_graph_step_key=MagicMock(
            return_value=(0, 3, 2, 4, 0)
        ),
        assert_elastic_calibration_captures_hot=MagicMock(),
    )
    core._add_elastic_calibration_request = MagicMock()
    core._prepare_elastic_calibration_capture = MagicMock()
    core._prepare_elastic_calibration_admission = MagicMock(return_value=2)
    core._reclaim_elastic_calibration_hotset_before_wave = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()
    core._run_elastic_calibration_step = MagicMock(
        return_value=SimpleNamespace(
            num_scheduled_tokens={"a": 2, "b": 2, "c": 2}
        )
    )

    with pytest.raises(RuntimeError, match="exceeded its exact wave preflight"):
        core._run_elastic_full_calibration_wave(
            k=3,
            x=2,
            query_len=4,
            serial=1,
            discover_max_x=True,
        )

    core.abort_requests.assert_called_once()
    core._drain_elastic_calibration.assert_called_once_with()
    assert core._reclaim_elastic_calibration_hotset_before_wave.call_args_list == [
        call(((0, 3, 2, 4, 0), (1, 3, 2, 8, 4))),
        call(((0, 3, 2, 4, 0), (1, 3, 2, 8, 4))),
    ]


def test_decode_capacity_uses_measured_cold_peak_and_exact_live_blocks():
    core = object.__new__(EngineCore)
    key = (1, 3, 23, 92, 4)
    block_pool = SimpleNamespace(
        active_num_gpu_blocks=47,
        get_num_free_blocks=MagicMock(return_value=23),
    )
    coordinator = SimpleNamespace(
        block_pool=block_pool,
        max_elastic_full_context_requests=MagicMock(return_value=22),
    )
    core.scheduler = SimpleNamespace(
        _elastic_graph_catalog={key: {"cold_peak_bytes": 293_601_280}},
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
    )

    assert core._elastic_calibration_decode_capacity(x=23, step_key=key) == 22
    coordinator.max_elastic_full_context_requests.assert_called_once_with(
        1,
        293_601_280,
        23,
    )


def test_pre_ready_calibration_seals_only_after_complete_plan(monkeypatch):
    core = object.__new__(EngineCore)
    coordinator = MagicMock()
    coordinator.max_elastic_full_context_requests.return_value = 1
    coordinator.elastic_full_context_capacity_receipt.return_value = {
        "attention_blocks": 10,
        "required_attention_blocks": 4,
        "residual_attention_blocks": 6,
        "gdn_blocks": 4,
    }
    scheduler = SimpleNamespace(
        num_spec_tokens=3,
        dynamic_sd_lookup=None,
        max_num_running_reqs=2,
        kv_cache_config=object(),
        _elastic_graph_catalog={},
        _elastic_graph_execution_policy=_current_q4_piecewise_policy(),
        _elastic_short_decode_inventory={},
        _elastic_calibration_mode=True,
        _elastic_require_catalog=False,
        _publish_elastic_startup_capacity=MagicMock(),
        _elastic_primary_blocks_per_max_request=3,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
    )
    def rebuild_inventory(max_x):
        xs = []
        value = 1
        while value <= max_x:
            xs.append(value)
            value <<= 1
        if xs[-1] != max_x:
            xs.append(max_x)
        scheduler._elastic_short_decode_inventory = {x: () for x in xs}

    scheduler._rebuild_elastic_short_decode_inventory = rebuild_inventory
    scheduler._resolve_elastic_step_physical_keys = lambda key: (key,)
    core.scheduler = scheduler
    core.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        model_config=SimpleNamespace(max_model_len=16),
    )
    core.is_pooling_model = False
    core.async_scheduling = False
    core.batch_queue = None

    full_calls = []
    prefill_calls = []
    mixed_calls = []

    def observe(key):
        row = scheduler._elastic_graph_catalog.setdefault(
            key,
            {
                "cold_peak_bytes": 8,
                "hot_peak_bytes": 4,
                "cold_observations": 0,
                "cold_stable_replays": 0,
                "hot_observations": 0,
                "hot_stable_replays": 0,
            },
        )
        row["cold_observations"] += 1
        row["hot_observations"] += 1
        if row["cold_observations"] >= 2:
            row["cold_stable_replays"] = 1
            row["hot_stable_replays"] = 1

    def full_wave(*, k, x, query_len, serial, discover_max_x=False):
        full_calls.append((k, x, query_len, serial))
        key = core._elastic_calibration_decode_key(
            k=k, x=x, query_len=query_len
        )
        observe(key)
        return key, x

    def balanced_pair(*, k, x, m, serial):
        prefill_calls.append((k, x, m, serial))
        key = (0, k, x, m, 0)
        observe(key)
        return key, x

    def mixed_pair(*, k, x, m, query_len, serial):
        mixed_calls.append((k, x, m, query_len, serial))
        key = (0, k, x, m, 0)
        observe(key)
        return key, x

    core._run_elastic_full_calibration_wave = full_wave
    core._run_elastic_balanced_prefill_pair = balanced_pair
    core._run_elastic_mixed_prefill_pair = mixed_pair
    sealed = MagicMock(return_value="/tmp/catalog.json")
    monkeypatch.setattr(startup_plan, "seal_elastic_graph_catalog", sealed)

    core._run_elastic_startup_calibration()

    # Product-K FULL includes both normal K+1 verification and the reachable
    # one-token acceptance-contraction family, X1..X2.
    assert len(full_calls) == 8
    # K0 and product-K max-B worst forms plus compact K0 holdouts.
    assert len(prefill_calls) == 8
    assert {call[0:3] for call in prefill_calls if call[2] == 4} >= {
        (0, 2, 4),
        (3, 2, 4),
    }
    # The product-K pure max-B witness shares its physical row with the mixed
    # witness, so the exact compatible row needs no redundant third replay.
    assert len(mixed_calls) == 2
    assert full_calls[0][0:3] == (3, 1, 4)  # Product K at floor(raw C0).
    assert {(call[0], call[1], call[2]) for call in full_calls} >= {
        (3, 1, 1),
        (3, 2, 1),
        (3, 1, 4),
        (3, 2, 4),
    }
    sealed.assert_called_once()
    required = sealed.call_args.kwargs["required_step_keys"]
    # Full-context X1 is a separate 262144-token KV guarantee. It must not
    # collapse the proven short/mixed product boundary from X2 to X1.
    assert all(key[2] <= 2 for key in required)
    assert (1, 3, 1, 1, 1) in required
    assert (1, 3, 2, 2, 1) in required
    assert sealed.call_args.kwargs["decode_max_x"] == 2
    assert sealed.call_args.kwargs["mixed_max_x"] == 2
    assert sealed.call_args.kwargs["full_context_max_x"] == 1
    assert scheduler.max_num_running_reqs == 2
    assert scheduler._elastic_calibration_mode is False
    assert scheduler._elastic_require_catalog is True
    scheduler._publish_elastic_startup_capacity.assert_called_once_with()


def test_policy_migration_fast_reseal_restores_only_bounded_hotset(monkeypatch):
    core = object.__new__(EngineCore)
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        _elastic_graph_catalog={(0, 3, 2, 8, 0): {}},
        _elastic_graph_catalog_coverage={},
        max_num_running_reqs=44,
        _elastic_calibration_mode=True,
        _elastic_require_catalog=False,
        _publish_elastic_startup_capacity=MagicMock(),
    )
    core.scheduler = scheduler
    core.vllm_config = object()
    core._restore_elastic_bounded_hotset = MagicMock()
    coverage = {
        "required_step_keys": [[1, 3, 1, 1, 1], [0, 3, 2, 8, 0]],
        "decode_max_x": 40,
        "mixed_max_x": 40,
        "full_context_max_x": 1,
    }
    reseal = MagicMock(return_value=coverage)
    monkeypatch.setattr(
        startup_plan,
        "try_reseal_policy_migrated_elastic_graph_catalog",
        reseal,
    )

    assert core._try_fast_reseal_policy_migration(started=0.0)

    reseal.assert_called_once()
    assert scheduler._elastic_graph_catalog_coverage == coverage
    assert scheduler.max_num_running_reqs == 40
    assert scheduler._elastic_calibration_mode is False
    assert scheduler._elastic_require_catalog is True
    core._restore_elastic_bounded_hotset.assert_called_once_with()
    scheduler._publish_elastic_startup_capacity.assert_called_once_with()


def test_pre_ready_calibration_accepts_constant_dynamic_k_schedule(monkeypatch):
    core = object.__new__(EngineCore)
    coordinator = MagicMock()
    coordinator.max_elastic_full_context_requests.return_value = 1
    coordinator.elastic_full_context_capacity_receipt.return_value = {
        "attention_blocks": 10,
        "required_attention_blocks": 4,
        "residual_attention_blocks": 6,
        "gdn_blocks": 4,
    }
    scheduler = SimpleNamespace(
        num_spec_tokens=3,
        dynamic_sd_lookup=[0, 3, 3, 3, 3, 3],
        max_num_running_reqs=5,
        kv_cache_config=object(),
        _elastic_graph_catalog={},
        _elastic_graph_execution_policy=_current_q4_piecewise_policy(),
        _elastic_short_decode_inventory={},
        _elastic_calibration_mode=True,
        _elastic_require_catalog=False,
        _publish_elastic_startup_capacity=MagicMock(),
        _elastic_primary_blocks_per_max_request=3,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
    )
    def rebuild_inventory(max_x):
        xs = []
        value = 1
        while value <= max_x:
            xs.append(value)
            value <<= 1
        if xs[-1] != max_x:
            xs.append(max_x)
        scheduler._elastic_short_decode_inventory = {x: () for x in xs}

    scheduler._rebuild_elastic_short_decode_inventory = rebuild_inventory
    scheduler._resolve_elastic_step_physical_keys = lambda key: (key,)
    core.scheduler = scheduler
    core.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=20),
        model_config=SimpleNamespace(max_model_len=16),
    )
    core.is_pooling_model = False
    core.async_scheduling = False
    core.batch_queue = None

    full_calls = []
    terminal_downshifted = False

    def full_wave(*, k, x, query_len, serial, discover_max_x=False):
        nonlocal terminal_downshifted
        full_calls.append((x, query_len, discover_max_x))
        if discover_max_x and x == 5 and not terminal_downshifted:
            terminal_downshifted = True
            return None, 3
        key = core._elastic_calibration_decode_key(
            k=k, x=x, query_len=query_len
        )
        scheduler._elastic_graph_catalog[key] = {
            "cold_peak_bytes": 8,
            "hot_peak_bytes": 4,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
        }
        return key, x

    core._run_elastic_full_calibration_wave = MagicMock(side_effect=full_wave)
    core._run_elastic_balanced_prefill_pair = MagicMock(
        side_effect=lambda *, k, x, m, serial: (
            scheduler._elastic_graph_catalog.setdefault(
                (0, k, x, m, 0),
                {
                    "cold_peak_bytes": 8,
                    "hot_peak_bytes": 4,
                    "cold_observations": 2,
                    "cold_stable_replays": 1,
                    "hot_observations": 2,
                    "hot_stable_replays": 1,
                },
            )
            and ((0, k, x, m, 0), x)
        )
    )
    core._run_elastic_mixed_prefill_pair = MagicMock(
        side_effect=lambda *, k, x, m, query_len, serial: (
            scheduler._elastic_graph_catalog.setdefault(
                (0, k, x, m, 0),
                {
                    "cold_peak_bytes": 8,
                    "hot_peak_bytes": 4,
                    "cold_observations": 2,
                    "cold_stable_replays": 1,
                    "hot_observations": 2,
                    "hot_stable_replays": 1,
                },
            )
            and ((0, k, x, m, 0), x)
        )
    )
    sealed = MagicMock(return_value="catalog")
    monkeypatch.setattr(startup_plan, "seal_elastic_graph_catalog", sealed)

    core._run_elastic_startup_calibration()

    assert scheduler._elastic_calibration_mode is False
    assert scheduler._elastic_require_catalog is True
    first_terminal_index = next(
        index
        for index, (x, query_len, terminal) in enumerate(full_calls)
        if x == 5 and query_len == 4 and terminal
    )
    assert tuple(
        dict.fromkeys(
            x
            for x, query_len, _terminal in full_calls[:first_terminal_index]
            if query_len == 4
        )
    ) == (1, 2, 4)
    assert first_terminal_index > next(
        index
        for index, (x, query_len, _terminal) in enumerate(full_calls)
        if x == 4 and query_len == 4
    )
    assert next(
        (x, query_len, terminal)
        for x, query_len, terminal in full_calls[first_terminal_index + 1 :]
        if terminal
    ) == (3, 4, True)
    assert sealed.call_args.kwargs["decode_max_x"] == 3
    assert (0, 3, 1, 4, 4) in sealed.call_args.kwargs["required_step_keys"]


def test_pre_ready_calibration_rejects_genuinely_dynamic_k_schedule():
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        num_spec_tokens=3,
        dynamic_sd_lookup=[0, 3, 2],
        max_num_running_reqs=2,
    )
    core.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1)
    )
    core.is_pooling_model = False
    core.async_scheduling = False
    core.batch_queue = None

    with pytest.raises(RuntimeError, match=r"X=1\.\.2 resolves K=\[2, 3\]"):
        core._run_elastic_startup_calibration()


def test_balanced_prefill_cold_hot_witnesses_do_not_mix_decode_tail():
    core = object.__new__(EngineCore)
    key = (0, 0, 2, 8, 0)
    output = SimpleNamespace(
        num_scheduled_tokens={"a": 4, "b": 4},
        num_spec_tokens_to_schedule=0,
        is_pure_decode_step=False,
    )
    scheduler = SimpleNamespace(
        _canonical_elastic_graph_step_key=MagicMock(return_value=key),
        assert_elastic_calibration_captures_hot=MagicMock(),
    )
    core.scheduler = scheduler
    core._begin_elastic_calibration_physical_epoch = MagicMock()
    core._prepare_elastic_calibration_capture = MagicMock()
    core._add_elastic_calibration_request = MagicMock()
    core._prepare_elastic_calibration_admission = MagicMock(return_value=2)
    core._run_elastic_calibration_step = MagicMock(side_effect=[output, output])
    core._assert_elastic_calibration_key = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()

    assert core._run_elastic_balanced_prefill_pair(k=0, x=2, m=8, serial=9) == (
        key,
        2,
    )

    assert core.abort_requests.call_count == 2
    assert all(len(call.args[0]) == 2 for call in core.abort_requests.call_args_list)
    assert core._run_elastic_calibration_step.call_count == 2
    core._prepare_elastic_calibration_capture.assert_called_once_with(key)
    scheduler.assert_elastic_calibration_captures_hot.assert_called_once_with(
        (key,)
    )
    core._drain_elastic_calibration.assert_called_once_with()


def test_balanced_product_prefill_preserves_configured_k_in_catalog_key():
    core = object.__new__(EngineCore)
    key = (0, 3, 1, 8, 0)
    output = SimpleNamespace(
        num_scheduled_tokens={"a": 8},
        num_spec_tokens_to_schedule=3,
        is_pure_decode_step=False,
    )
    scheduler = SimpleNamespace(
        _canonical_elastic_graph_step_key=MagicMock(return_value=key),
        assert_elastic_calibration_captures_hot=MagicMock(),
    )
    core.scheduler = scheduler
    core._begin_elastic_calibration_physical_epoch = MagicMock()
    core._prepare_elastic_calibration_capture = MagicMock()
    core._add_elastic_calibration_request = MagicMock()
    core._prepare_elastic_calibration_admission = MagicMock(return_value=1)
    core._run_elastic_calibration_step = MagicMock(side_effect=[output, output])
    core._assert_elastic_calibration_key = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()

    assert core._run_elastic_balanced_prefill_pair(k=3, x=1, m=8, serial=11) == (
        key,
        1,
    )
    scheduler._canonical_elastic_graph_step_key.assert_called_once_with(
        {"prefill-0": 8}, 3, False
    )
    core._prepare_elastic_calibration_capture.assert_called_once_with(key)
    scheduler.assert_elastic_calibration_captures_hot.assert_called_once_with(
        (key,)
    )


def test_balanced_prefill_probe_returns_monotone_admission_downshift():
    core = object.__new__(EngineCore)
    key = (0, 0, 2, 8, 0)
    output = SimpleNamespace(num_scheduled_tokens={"a": 4})
    core.scheduler = SimpleNamespace(
        _canonical_elastic_graph_step_key=MagicMock(return_value=key)
    )
    core._begin_elastic_calibration_physical_epoch = MagicMock()
    core._prepare_elastic_calibration_capture = MagicMock()
    core._add_elastic_calibration_request = MagicMock()
    core._prepare_elastic_calibration_admission = MagicMock(return_value=2)
    core._run_elastic_calibration_step = MagicMock(return_value=output)
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()
    core._reclaim_elastic_calibration_hotset_before_wave = MagicMock()

    assert core._run_elastic_balanced_prefill_pair(k=0, x=2, m=8, serial=10) == (
        None,
        1,
    )
    core.abort_requests.assert_called_once()
    core._drain_elastic_calibration.assert_called_once_with()
    core._reclaim_elastic_calibration_hotset_before_wave.assert_called_once_with(
        (key,)
    )


def test_mixed_prefill_cold_hot_preserves_only_decode_cohort():
    core = object.__new__(EngineCore)
    key = (0, 0, 3, 16, 0)
    prefill = SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 2})
    mixed = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1, "c": 14})
    scheduler = SimpleNamespace(
        requests={},
        _canonical_elastic_graph_step_key=MagicMock(return_value=key),
    )
    core.scheduler = scheduler
    core._begin_elastic_calibration_physical_epoch = MagicMock()
    core._add_elastic_calibration_request = MagicMock()
    core._prepare_elastic_calibration_admission = MagicMock(return_value=2)
    core._run_elastic_calibration_step = MagicMock(side_effect=[prefill, mixed, mixed])
    core._assert_elastic_calibration_key = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_calibration = MagicMock()

    assert core._run_elastic_mixed_prefill_pair(
        k=0, x=3, m=16, query_len=1, serial=10
    ) == (key, 3)

    aborted = [call.args[0] for call in core.abort_requests.call_args_list]
    assert len(aborted) == 3
    assert all(len(ids) == 1 for ids in aborted[:2])
    assert len(aborted[2]) == 2
    core._drain_elastic_calibration.assert_called_once_with()
