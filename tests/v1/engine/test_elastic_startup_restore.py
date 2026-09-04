# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm.v1.core.elastic_graph import (
    ElasticAdmissionController,
    ElasticResidencyReceipt,
    RuntimeGeneration,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import elastic_bootstrap
from vllm.v1.engine.core import EngineCore


def receipt(
    generation: str = "restore-generation",
    *,
    resident: int = 64,
    workspace: int = 32,
) -> ElasticResidencyReceipt:
    return ElasticResidencyReceipt(
        generation=RuntimeGeneration(generation),
        transaction_id=None,
        resident_bytes=resident,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=resident,
        cublas_workspace_bytes=workspace,
        entries=(),
    )


def test_startup_residency_publishes_one_typed_consensus_before_ready() -> None:
    core = object.__new__(EngineCore)
    accepted = receipt()
    scheduler = SimpleNamespace(
        _elastic_admission_controller=ElasticAdmissionController(
            RuntimeGeneration("restore-generation")
        ),
        _elastic_cublas_workspace_unit_bytes=0,
        _sync_elastic_residency_receipt=MagicMock(),
        _publish_elastic_startup_capacity=MagicMock(),
    )
    core.scheduler = scheduler
    core.collective_rpc = MagicMock(return_value=(accepted, accepted, accepted))

    core._synchronize_elastic_startup_residency()

    core.collective_rpc.assert_called_once_with("get_elastic_graph_residency_receipt")
    scheduler._sync_elastic_residency_receipt.assert_called_once_with(accepted)
    scheduler._publish_elastic_startup_capacity.assert_called_once_with()
    assert scheduler._elastic_cublas_workspace_unit_bytes == 32


@pytest.mark.parametrize(
    "receipts,match",
    [
        ((receipt("stale"),), "generation"),
        ((receipt(), receipt(resident=65)), "differs across worker ranks"),
    ],
)
def test_startup_residency_rejects_stale_or_rank_divergent_receipt(
    receipts, match
) -> None:
    core = object.__new__(EngineCore)
    scheduler = SimpleNamespace(
        _elastic_admission_controller=ElasticAdmissionController(
            RuntimeGeneration("restore-generation")
        ),
        _elastic_cublas_workspace_unit_bytes=17,
        _sync_elastic_residency_receipt=MagicMock(),
        _publish_elastic_startup_capacity=MagicMock(),
    )
    core.scheduler = scheduler
    core.collective_rpc = MagicMock(return_value=receipts)

    with pytest.raises(RuntimeError, match=match):
        core._synchronize_elastic_startup_residency()

    scheduler._sync_elastic_residency_receipt.assert_not_called()
    scheduler._publish_elastic_startup_capacity.assert_not_called()
    assert scheduler._elastic_cublas_workspace_unit_bytes == 17


def test_bounded_restart_retains_complete_terminal_serving_carrier() -> None:
    core = object.__new__(EngineCore)
    piecewise = [0, 3, 1, 2, 0]
    bridge = [1, 3, 1, 1, 1]
    verification = [0, 3, 1, 4, 4]
    carrier_keys = ("prefill", "bridge", "verification", "terminal-mtp")
    controller = SimpleNamespace(
        resident_bytes=123,
        floor_bytes=0,
        entries={key: SimpleNamespace(hot=True) for key in carrier_keys},
    )
    scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={
            "representation": "bounded_exact_hotset",
            "mixed_max_x": 39,
            "required_step_keys": [piecewise, bridge, verification],
            "restore_step_keys": [piecewise, bridge],
            "serving_carrier_step_keys": [piecewise, bridge, verification],
            "serving_carrier_contract": ("retained-terminal-mtp-no-cold-serving-v1"),
            "serving_carrier_owner": "mtp_decode",
        },
        num_spec_tokens=3,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(disable_speculation_on_non_decode=False)
        ),
        _elastic_admission_controller=controller,
        _elastic_serving_carrier_bytes=MagicMock(return_value=23),
        resolve_elastic_serving_carrier_physical_keys=MagicMock(
            return_value=("terminal-mtp",)
        ),
        _resolve_elastic_step_physical_keys=MagicMock(
            side_effect=lambda key: (
                carrier_keys[
                    (tuple(piecewise), tuple(bridge), tuple(verification)).index(key)
                ],
            )
        ),
        _elastic_restore_mode=False,
        max_num_running_reqs=64,
    )
    core.scheduler = scheduler
    core._elastic_restore_execution_step_keys = MagicMock(
        side_effect=lambda *, query_len, **_kwargs: (
            (tuple(piecewise), tuple(bridge))
            if query_len == 1
            else (tuple(piecewise), tuple(verification))
        )
    )
    core._run_elastic_full_restore_wave_in_epoch = MagicMock(
        side_effect=((tuple(bridge), 1), (tuple(verification), 1))
    )

    core._restore_elastic_bounded_hotset()

    assert core._run_elastic_full_restore_wave_in_epoch.call_args_list == [
        call(k=3, x=1, query_len=1, serial=1, preserve_hotset=True),
        call(k=3, x=1, query_len=4, serial=2, preserve_hotset=True),
    ]
    assert scheduler._elastic_serving_carrier_keys == ("terminal-mtp",)
    assert scheduler._elastic_serving_carrier_resident_bytes == 23
    assert scheduler._elastic_restore_mode is False
    assert scheduler.max_num_running_reqs == 39


def test_bounded_restart_rejects_unknown_serving_carrier_contract() -> None:
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={"serving_carrier_contract": "unknown-v1"}
    )

    with pytest.raises(RuntimeError, match="serving carrier contract is unsupported"):
        core._restore_elastic_bounded_hotset()


def test_bounded_restart_second_wave_failure_preserves_pre_ready_boundary() -> None:
    core = object.__new__(EngineCore)
    prefill = (0, 3, 1, 2, 0)
    q1 = (1, 3, 1, 1, 1)
    q4 = (0, 3, 1, 4, 4)
    scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={
            "representation": "bounded_exact_hotset",
            "mixed_max_x": 39,
            "required_step_keys": [list(prefill), list(q1), list(q4)],
            "serving_carrier_step_keys": [list(prefill), list(q1), list(q4)],
            "serving_carrier_contract": ("retained-terminal-mtp-no-cold-serving-v1"),
            "serving_carrier_owner": "mtp_decode",
        },
        num_spec_tokens=3,
        _elastic_restore_mode=False,
        max_num_running_reqs=64,
    )
    core.scheduler = scheduler
    core._elastic_restore_execution_step_keys = MagicMock(
        side_effect=lambda *, query_len, **_kwargs: (
            (prefill, q1) if query_len == 1 else (prefill, q4)
        )
    )
    core._run_elastic_full_restore_wave_in_epoch = MagicMock(
        side_effect=((q1, 1), (None, 0))
    )

    with pytest.raises(RuntimeError, match="terminal carrier"):
        core._restore_elastic_bounded_hotset()

    assert scheduler._elastic_restore_mode is False
    assert scheduler.max_num_running_reqs == 64


def test_bounded_restart_rejects_same_k_x_wrong_m_before_wave() -> None:
    core = object.__new__(EngineCore)
    expected_piecewise = (0, 3, 1, 2, 0)
    wrong_m_piecewise = [0, 3, 1, 4, 0]
    bridge = [1, 3, 1, 1, 1]
    verification = [0, 3, 1, 4, 4]
    scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={
            "representation": "bounded_exact_hotset",
            "mixed_max_x": 39,
            "required_step_keys": [wrong_m_piecewise, bridge, verification],
            "restore_step_keys": [wrong_m_piecewise, bridge],
            "serving_carrier_step_keys": [
                wrong_m_piecewise,
                bridge,
                verification,
            ],
            "serving_carrier_contract": ("retained-terminal-mtp-no-cold-serving-v1"),
            "serving_carrier_owner": "mtp_decode",
        },
        num_spec_tokens=3,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(disable_speculation_on_non_decode=False)
        ),
        _elastic_restore_mode=False,
        max_num_running_reqs=64,
    )
    core.scheduler = scheduler
    core._elastic_restore_execution_step_keys = MagicMock(
        return_value=(expected_piecewise, tuple(bridge), tuple(verification))
    )
    core._run_elastic_full_restore_wave_in_epoch = MagicMock()

    with pytest.raises(RuntimeError, match=r"prefill/q1/q\(K\+1\) identity"):
        core._restore_elastic_bounded_hotset()

    core._run_elastic_full_restore_wave_in_epoch.assert_not_called()
    assert scheduler._elastic_restore_mode is False
    assert scheduler.max_num_running_reqs == 64


@pytest.mark.parametrize("resident_bytes,expected", [(0, False), (64, True)])
def test_restore_preflight_reads_hot_residency_from_controller(
    resident_bytes: int,
    expected: bool,
) -> None:
    class StopAfterObservation(RuntimeError):
        pass

    controller = ElasticAdmissionController(RuntimeGeneration("restore-generation"))
    controller.publish_physical_accounting(
        resident_bytes=resident_bytes,
        floor_bytes=0,
        transition_floor_bytes=0,
        cublas_workspace_bytes=0,
        maintenance_step_key=None,
    )
    scheduler = SimpleNamespace(
        _elastic_admission_controller=controller,
        _canonical_elastic_graph_step_key=MagicMock(return_value=(0, 3, 1, 2, 0)),
        retain_elastic_restore_captures=MagicMock(return_value="retention"),
    )
    core = object.__new__(EngineCore)
    core.scheduler = scheduler
    core._elastic_restore_decode_key = MagicMock(return_value=(1, 3, 1, 4, 4))
    core._begin_elastic_restore_physical_epoch = MagicMock()
    core._add_elastic_restore_request = MagicMock()

    def observe(_request_ids, *, retain_hot_graphs):
        assert retain_hot_graphs is expected
        raise StopAfterObservation

    core._prepare_elastic_restore_admission = MagicMock(side_effect=observe)

    with pytest.raises(StopAfterObservation):
        core._run_elastic_full_restore_wave_in_epoch(
            k=3,
            x=1,
            query_len=1,
            serial=1,
        )


def test_q4_restore_executes_q1_bridge_before_target_decode() -> None:
    core = object.__new__(EngineCore)
    request_ids = [
        "_elastic_restore_full_1_3_4_2_0",
        "_elastic_restore_full_1_3_4_2_1",
    ]
    requests = {
        request_id: SimpleNamespace(spec_token_ids=[]) for request_id in request_ids
    }
    scheduler = SimpleNamespace(
        _elastic_admission_controller=SimpleNamespace(resident_bytes=0),
        requests=requests,
        retain_elastic_restore_captures=MagicMock(return_value="retention"),
        release_elastic_restore_retention=MagicMock(),
        prepare_elastic_restore_execution=MagicMock(),
    )
    core.scheduler = scheduler
    prefill_key = (0, 0, 2, 4, 0)
    bridge_key = (1, 3, 2, 2, 1)
    target_key = (1, 3, 2, 8, 4)
    core._elastic_restore_wave_step_keys = MagicMock(
        return_value=(prefill_key, target_key)
    )
    core._elastic_restore_decode_key = MagicMock(
        side_effect=lambda *, query_len, **_kwargs: (
            bridge_key if query_len == 1 else target_key
        )
    )
    core._elastic_restore_prefill_k = MagicMock(return_value=0)
    core._begin_elastic_restore_physical_epoch = MagicMock()
    core._add_elastic_restore_request = MagicMock()
    core._prepare_elastic_restore_admission = MagicMock(return_value=2)
    core._assert_elastic_restore_key = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_restore = MagicMock()
    core._rollback_elastic_restore_physical_epoch = MagicMock()

    def output(tokens: int, query_len: int):
        return SimpleNamespace(
            num_scheduled_tokens={request_id: tokens for request_id in request_ids},
            num_spec_tokens_to_schedule=3,
            is_pure_decode_step=query_len > 0,
        )

    steps = iter((output(2, 0), output(1, 1), output(4, 4), output(4, 4)))

    def run_step():
        result = next(steps)
        if result is not None and result.num_scheduled_tokens:
            first_tokens = next(iter(result.num_scheduled_tokens.values()))
            if first_tokens == 1:
                for request in requests.values():
                    request.spec_token_ids = [11, 12, 13]
        return result

    core._run_elastic_restore_step = MagicMock(side_effect=run_step)

    actual, admitted = core._run_elastic_full_restore_wave_in_epoch(
        k=3,
        x=2,
        query_len=4,
        serial=1,
    )

    assert (actual, admitted) == (target_key, 2)
    core._begin_elastic_restore_physical_epoch.assert_called_once_with(
        (prefill_key, bridge_key, target_key)
    )
    assert scheduler.prepare_elastic_restore_execution.call_args_list == [
        call(bridge_key),
        call(target_key),
        call(target_key),
    ]
    scheduler.release_elastic_restore_retention.assert_called_once_with("retention")
    assert core._assert_elastic_restore_key.call_args_list == [
        call(output(1, 1), bridge_key),
        call(output(4, 4), target_key),
        call(output(4, 4), target_key),
    ]
    core._rollback_elastic_restore_physical_epoch.assert_not_called()


@pytest.mark.parametrize(
    ("disable_on_non_decode", "expected"),
    ((False, False), (True, True)),
)
def test_q4_bridge_follows_effective_non_decode_speculation_policy(
    disable_on_non_decode: bool,
    expected: bool,
) -> None:
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                disable_speculation_on_non_decode=disable_on_non_decode
            )
        )
    )

    assert core._elastic_restore_needs_decode_bridge(k=3, query_len=4) is expected
    assert not core._elastic_restore_needs_decode_bridge(k=3, query_len=1)
    assert not core._elastic_restore_needs_decode_bridge(k=0, query_len=4)


def test_q4_restore_uses_prefill_drafts_without_artificial_bridge() -> None:
    core = object.__new__(EngineCore)
    request_id = "_elastic_restore_full_1_3_4_1_0"
    request = SimpleNamespace(spec_token_ids=[])
    scheduler = SimpleNamespace(
        _elastic_admission_controller=SimpleNamespace(resident_bytes=0),
        requests={request_id: request},
        retain_elastic_restore_captures=MagicMock(return_value="retention"),
        release_elastic_restore_retention=MagicMock(),
        prepare_elastic_restore_execution=MagicMock(),
    )
    core.scheduler = scheduler
    prefill_key = (0, 3, 1, 2, 0)
    target_key = (1, 3, 1, 4, 4)
    core._elastic_restore_wave_step_keys = MagicMock(
        return_value=(prefill_key, target_key)
    )
    core._elastic_restore_prefill_k = MagicMock(return_value=3)
    core._elastic_restore_decode_key = MagicMock(return_value=target_key)
    core._begin_elastic_restore_physical_epoch = MagicMock()
    core._add_elastic_restore_request = MagicMock()
    core._prepare_elastic_restore_admission = MagicMock(return_value=1)
    core._assert_elastic_restore_key = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_restore = MagicMock()
    core._rollback_elastic_restore_physical_epoch = MagicMock()
    prefill = SimpleNamespace(num_scheduled_tokens={request_id: 2})
    cold = SimpleNamespace(num_scheduled_tokens={request_id: 4})
    hot = SimpleNamespace(num_scheduled_tokens={request_id: 4})

    def run_step():
        if not request.spec_token_ids:
            request.spec_token_ids = [11, 12, 13]
            return prefill
        return cold if core._run_elastic_restore_step.call_count == 2 else hot

    core._run_elastic_restore_step = MagicMock(side_effect=run_step)

    assert core._run_elastic_full_restore_wave_in_epoch(
        k=3,
        x=1,
        query_len=4,
        serial=1,
    ) == (target_key, 1)

    core._begin_elastic_restore_physical_epoch.assert_called_once_with(
        (prefill_key, target_key)
    )
    assert scheduler.prepare_elastic_restore_execution.call_args_list == [
        call(target_key),
        call(target_key),
    ]
    scheduler.release_elastic_restore_retention.assert_called_once_with("retention")
    core._rollback_elastic_restore_physical_epoch.assert_not_called()


def test_preserved_restore_promotes_carrier_before_request_free_drain() -> None:
    core = object.__new__(EngineCore)
    request_id = "_elastic_restore_full_1_3_1_1_0"
    request = SimpleNamespace(spec_token_ids=[])
    events = []
    scheduler = SimpleNamespace(
        _elastic_admission_controller=SimpleNamespace(resident_bytes=0),
        _elastic_restore_mode=True,
        requests={request_id: request},
        retain_elastic_restore_captures=MagicMock(return_value="retention"),
        resolve_elastic_serving_carrier_physical_keys=MagicMock(
            return_value=("terminal-mtp",)
        ),
        promote_elastic_restore_retention_to_serving=lambda value, keys: events.append(
            ("promote", value, keys)
        ),
        release_elastic_restore_retention=lambda value: events.append(
            ("release", value)
        ),
        prepare_elastic_restore_execution=MagicMock(),
    )
    core.scheduler = scheduler
    prefill_key = (0, 3, 1, 2, 0)
    target_key = (1, 3, 1, 1, 1)
    core._elastic_restore_wave_step_keys = MagicMock(
        return_value=(prefill_key, target_key)
    )
    core._elastic_restore_decode_key = MagicMock(return_value=target_key)
    core._elastic_restore_prefill_k = MagicMock(return_value=3)
    core._begin_elastic_restore_physical_epoch = MagicMock()
    core._add_elastic_restore_request = MagicMock()
    core._prepare_elastic_restore_admission = MagicMock(return_value=1)
    core._assert_elastic_restore_key = MagicMock()
    core.abort_requests = lambda request_ids: events.append(("abort", request_ids))
    core._drain_elastic_restore = lambda: events.append(("drain", None))
    core._rollback_elastic_restore_physical_epoch = MagicMock()
    prefill = SimpleNamespace(num_scheduled_tokens={request_id: 2})
    cold = SimpleNamespace(num_scheduled_tokens={request_id: 1})
    hot = SimpleNamespace(num_scheduled_tokens={request_id: 1})
    core._run_elastic_restore_step = MagicMock(side_effect=(prefill, cold, hot))

    assert core._run_elastic_full_restore_wave_in_epoch(
        k=3,
        x=1,
        query_len=1,
        serial=1,
        preserve_hotset=True,
    ) == (target_key, 1)

    assert events == [
        ("promote", "retention", ("terminal-mtp",)),
        ("abort", [request_id]),
        ("drain", None),
    ]
    assert scheduler._elastic_restore_mode is False


def test_restore_retention_becomes_protected_before_lease_release() -> None:
    retained = ("prefill", "terminal-mtp", "q1-target")
    release_observations = []
    scheduler = SimpleNamespace(
        _elastic_restore_mode=True,
        _elastic_restore_retention_id="retention",
        _elastic_restore_retained_physical_keys=retained,
        _elastic_serving_carrier_keys=("existing",),
        _elastic_serving_carrier_resident_bytes=0,
    )

    def release(transaction_id):
        release_observations.append(
            (transaction_id, scheduler._elastic_serving_carrier_keys)
        )

    scheduler._elastic_admission_controller = SimpleNamespace(
        entries={key: SimpleNamespace(hot=True) for key in retained},
        release=release,
    )
    scheduler._elastic_serving_carrier_bytes = lambda: (
        len(scheduler._elastic_serving_carrier_keys) * 10
    )

    Scheduler.promote_elastic_restore_retention_to_serving(
        scheduler, "retention", ("terminal-mtp",)
    )

    assert release_observations == [
        ("retention", ("existing", "terminal-mtp")),
    ]
    assert scheduler._elastic_serving_carrier_resident_bytes == 20
    assert scheduler._elastic_restore_retention_id is None
    assert scheduler._elastic_restore_retained_physical_keys == ()


def test_restore_retention_rejects_non_hot_serving_promotion() -> None:
    scheduler = SimpleNamespace(
        _elastic_restore_mode=True,
        _elastic_restore_retention_id="retention",
        _elastic_restore_retained_physical_keys=("cold",),
        _elastic_serving_carrier_keys=(),
        _elastic_serving_carrier_resident_bytes=0,
        _elastic_admission_controller=SimpleNamespace(
            entries={"cold": SimpleNamespace(hot=False)},
            release=MagicMock(),
        ),
        _elastic_serving_carrier_bytes=MagicMock(),
    )

    with pytest.raises(RuntimeError, match="requires a HOT restore DAG"):
        Scheduler.promote_elastic_restore_retention_to_serving(
            scheduler, "retention", ("cold",)
        )

    scheduler._elastic_admission_controller.release.assert_not_called()


def test_q4_restore_rejects_missing_prefill_drafts_before_target_execution() -> None:
    core = object.__new__(EngineCore)
    request_id = "_elastic_restore_full_1_3_4_1_0"
    scheduler = SimpleNamespace(
        _elastic_admission_controller=SimpleNamespace(resident_bytes=0),
        requests={request_id: SimpleNamespace(spec_token_ids=[])},
        retain_elastic_restore_captures=MagicMock(return_value="retention"),
        prepare_elastic_restore_execution=MagicMock(),
    )
    core.scheduler = scheduler
    prefill_key = (0, 3, 1, 2, 0)
    target_key = (1, 3, 1, 4, 4)
    core._elastic_restore_wave_step_keys = MagicMock(
        return_value=(prefill_key, target_key)
    )
    core._elastic_restore_prefill_k = MagicMock(return_value=3)
    core._begin_elastic_restore_physical_epoch = MagicMock()
    core._add_elastic_restore_request = MagicMock()
    core._prepare_elastic_restore_admission = MagicMock(return_value=1)
    core._run_elastic_restore_step = MagicMock(
        return_value=SimpleNamespace(num_scheduled_tokens={request_id: 2})
    )
    core._rollback_elastic_restore_physical_epoch = MagicMock()

    with pytest.raises(RuntimeError, match="prefill did not produce"):
        core._run_elastic_full_restore_wave_in_epoch(
            k=3,
            x=1,
            query_len=4,
            serial=1,
        )

    scheduler.prepare_elastic_restore_execution.assert_not_called()
    core._rollback_elastic_restore_physical_epoch.assert_called_once_with(
        request_ids=[request_id],
        step_keys=(prefill_key, target_key),
    )


def test_q4_restore_rolls_back_when_q1_bridge_produces_no_drafts() -> None:
    core = object.__new__(EngineCore)
    request_id = "_elastic_restore_full_1_3_4_1_0"
    scheduler = SimpleNamespace(
        _elastic_admission_controller=SimpleNamespace(resident_bytes=0),
        requests={request_id: SimpleNamespace(spec_token_ids=[])},
        _elastic_restore_retention_id="retention",
        retain_elastic_restore_captures=MagicMock(return_value="retention"),
        release_elastic_restore_retention=MagicMock(),
        prepare_elastic_restore_execution=MagicMock(),
    )
    core.scheduler = scheduler
    prefill_key = (0, 0, 1, 2, 0)
    bridge_key = (1, 3, 1, 1, 1)
    target_key = (1, 3, 1, 4, 4)
    core._elastic_restore_wave_step_keys = MagicMock(
        return_value=(prefill_key, target_key)
    )
    core._elastic_restore_decode_key = MagicMock(
        side_effect=lambda *, query_len, **_kwargs: (
            bridge_key if query_len == 1 else target_key
        )
    )
    core._elastic_restore_prefill_k = MagicMock(return_value=0)
    core._begin_elastic_restore_physical_epoch = MagicMock()
    core._add_elastic_restore_request = MagicMock()
    core._prepare_elastic_restore_admission = MagicMock(return_value=1)
    core._assert_elastic_restore_key = MagicMock()
    core._run_elastic_restore_step = MagicMock(
        side_effect=(
            SimpleNamespace(num_scheduled_tokens={request_id: 2}),
            SimpleNamespace(num_scheduled_tokens={request_id: 1}),
        )
    )
    core._rollback_elastic_restore_physical_epoch = MagicMock()
    core.abort_requests = MagicMock()
    core._drain_elastic_restore = MagicMock()

    with pytest.raises(RuntimeError, match="did not produce the target draft width"):
        core._run_elastic_full_restore_wave_in_epoch(
            k=3,
            x=1,
            query_len=4,
            serial=1,
        )

    core._rollback_elastic_restore_physical_epoch.assert_called_once_with(
        request_ids=[request_id],
        step_keys=(prefill_key, bridge_key, target_key),
    )
    core.abort_requests.assert_not_called()


def test_restore_rollback_releases_retention_before_drain() -> None:
    events = []
    scheduler = SimpleNamespace(
        _elastic_restore_retention_id="retention",
        release_elastic_restore_retention=lambda value: events.append(
            ("release", value)
        ),
        reconcile_elastic_restore_rollback=lambda: events.append(("reconcile", None)),
    )
    core = object.__new__(EngineCore)
    core.scheduler = scheduler
    core.abort_requests = lambda request_ids: events.append(("abort", request_ids))
    core._drain_elastic_restore = lambda: events.append(("drain", None))
    core._reclaim_elastic_restore_hotset_before_wave = lambda: events.append(
        ("reclaim", None)
    )

    core._rollback_elastic_restore_physical_epoch(
        request_ids=["request"],
        step_keys=((1, 3, 1, 1, 1),),
    )

    assert events == [
        ("release", "retention"),
        ("abort", ["request"]),
        ("reconcile", None),
        ("drain", None),
        ("reclaim", None),
    ]


def test_restore_drain_converts_idle_cleanup_debt_to_explicit_reclaim() -> None:
    core = object.__new__(EngineCore)
    scheduler = SimpleNamespace(
        has_requests=MagicMock(side_effect=(True, True, False)),
        has_pending_elastic_maintenance=MagicMock(return_value=False),
        _needs_elastic_idle_reclaim=MagicMock(side_effect=(True, True)),
        has_unfinished_requests=MagicMock(side_effect=(False, False)),
        has_finished_requests=MagicMock(side_effect=(True, False)),
        finish_elastic_restore_epoch=MagicMock(),
    )
    core.scheduler = scheduler
    core._run_elastic_restore_step = MagicMock()
    core._reclaim_elastic_restore_hotset_before_wave = MagicMock()

    core._drain_elastic_restore()

    core._run_elastic_restore_step.assert_called_once_with()
    core._reclaim_elastic_restore_hotset_before_wave.assert_called_once_with()
    scheduler.finish_elastic_restore_epoch.assert_called_once_with()


def test_restore_drain_executes_request_free_pending_maintenance() -> None:
    core = object.__new__(EngineCore)
    scheduler = SimpleNamespace(
        has_requests=MagicMock(return_value=False),
        has_pending_elastic_maintenance=MagicMock(side_effect=(True, False)),
        _needs_elastic_idle_reclaim=MagicMock(return_value=False),
        finish_elastic_restore_epoch=MagicMock(),
    )
    core.scheduler = scheduler
    core._run_elastic_restore_step = MagicMock()
    core._reclaim_elastic_restore_hotset_before_wave = MagicMock()

    core._drain_elastic_restore()

    core._run_elastic_restore_step.assert_called_once_with()
    core._reclaim_elastic_restore_hotset_before_wave.assert_not_called()
    scheduler.finish_elastic_restore_epoch.assert_called_once_with()


def test_failed_restore_shuts_down_partial_engine_without_masking_failure() -> None:
    core = object.__new__(EngineCore)
    core.shutdown = MagicMock(side_effect=RuntimeError("teardown failed"))

    core._shutdown_failed_elastic_startup()

    core.shutdown.assert_called_once_with()


def test_prepare_elastic_runtime_preserves_production_barrier_order(
    monkeypatch,
) -> None:
    events = []
    kv_config = SimpleNamespace(kv_cache_groups=(object(),))
    manager = object()
    scheduler = SimpleNamespace(elastic_on_demand_graphs=True)
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            get_scheduler_cls=lambda: scheduler_cls,
        )
    )

    def initialize_kv_cache():
        events.append("kv")
        return kv_config

    def resolve_policy(config, resolved_kv, collective_rpc):
        assert config is vllm_config
        assert resolved_kv is kv_config
        events.append("policy")

    def scheduler_cls(**kwargs):
        assert kwargs["kv_cache_config"] is kv_config
        assert kwargs["structured_output_manager"] is manager
        events.append("scheduler")
        return scheduler

    def synchronize_generation(resolved_scheduler, collective_rpc):
        assert resolved_scheduler is scheduler
        events.append("generation")
        return {"scheduler": "generation", "workers": ["generation"] * 3}

    monkeypatch.setattr(
        elastic_bootstrap,
        "resolve_elastic_graph_execution_policy",
        resolve_policy,
    )
    monkeypatch.setattr(
        elastic_bootstrap,
        "resolve_kv_cache_block_sizes",
        lambda *_: (16, 16),
    )
    monkeypatch.setattr(
        elastic_bootstrap,
        "synchronize_elastic_runtime_generation",
        synchronize_generation,
    )

    prepared = elastic_bootstrap.prepare_elastic_runtime(
        vllm_config=vllm_config,
        initialize_kv_cache=initialize_kv_cache,
        collective_rpc=MagicMock(),
        include_finished_set=False,
        log_stats=True,
        structured_output_manager_factory=lambda: events.append("manager") or manager,
    )

    assert events == ["kv", "policy", "manager", "scheduler", "generation"]
    assert prepared.scheduler is scheduler
    assert prepared.generation_receipt == {
        "scheduler": "generation",
        "workers": ["generation"] * 3,
    }


def test_complete_elastic_startup_restores_then_publishes_before_ready() -> None:
    events = []
    owner = SimpleNamespace(
        scheduler=SimpleNamespace(
            elastic_on_demand_graphs=True,
            _elastic_require_catalog=True,
            _elastic_graph_catalog={(1, 3, 1, 1, 1): {}},
            _elastic_graph_catalog_coverage={"representation": "bounded_exact_hotset"},
        ),
        _restore_elastic_bounded_hotset=lambda: events.append("restore"),
        _restore_elastic_pinned_full_family=lambda: events.append("wrong-restore"),
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    outcome = elastic_bootstrap.complete_elastic_startup(owner)

    assert outcome == "restored"
    assert events == ["restore", "publish"]


def test_complete_elastic_startup_failure_stops_before_publication() -> None:
    events = []

    def fail_restore():
        events.append("restore")
        raise RuntimeError("restore failed")

    owner = SimpleNamespace(
        scheduler=SimpleNamespace(
            elastic_on_demand_graphs=True,
            _elastic_require_catalog=True,
            _elastic_graph_catalog={(1, 3, 1, 1, 1): {}},
            _elastic_graph_catalog_coverage={"representation": "bounded_exact_hotset"},
        ),
        _restore_elastic_bounded_hotset=fail_restore,
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    with pytest.raises(RuntimeError, match="restore failed"):
        elastic_bootstrap.complete_elastic_startup(owner)

    assert events == ["restore", "shutdown"]


def test_complete_elastic_startup_auto_calibrates_miss_before_restore(
    monkeypatch,
) -> None:
    events = []
    scheduler = SimpleNamespace(
        elastic_on_demand_graphs=True,
        _elastic_require_catalog=True,
        _elastic_auto_calibrate=True,
        _elastic_graph_catalog={},
        _elastic_graph_catalog_coverage={},
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        _restore_elastic_bounded_hotset=lambda: events.append("restore"),
        _restore_elastic_pinned_full_family=lambda: events.append("wrong-restore"),
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    def calibrate(resolved_owner):
        assert resolved_owner is owner
        events.append("calibrate")
        scheduler._elastic_graph_catalog = {(1, 3, 1, 1, 1): {}}
        scheduler._elastic_graph_catalog_coverage = {
            "representation": "bounded_exact_hotset"
        }

    monkeypatch.setattr(elastic_bootstrap, "_auto_calibrate_missing_catalog", calibrate)

    assert elastic_bootstrap.complete_elastic_startup(owner) == "restored"
    assert events == ["calibrate", "restore", "publish"]


def test_complete_elastic_startup_catalog_miss_fails_once_and_shuts_down() -> None:
    events = []
    owner = SimpleNamespace(
        scheduler=SimpleNamespace(
            elastic_on_demand_graphs=True,
            _elastic_require_catalog=True,
            _elastic_auto_calibrate=False,
            _elastic_graph_catalog={},
        ),
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    with pytest.raises(RuntimeError, match="will not calibrate implicitly"):
        elastic_bootstrap.complete_elastic_startup(owner)

    assert events == ["shutdown"]


def test_complete_elastic_startup_auto_calibration_failure_shuts_down(
    monkeypatch,
) -> None:
    events = []
    owner = SimpleNamespace(
        scheduler=SimpleNamespace(
            elastic_on_demand_graphs=True,
            _elastic_require_catalog=True,
            _elastic_auto_calibrate=True,
            _elastic_graph_catalog={},
        ),
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    def fail(_owner):
        events.append("calibrate")
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(elastic_bootstrap, "_auto_calibrate_missing_catalog", fail)

    with pytest.raises(RuntimeError, match="calibration failed"):
        elastic_bootstrap.complete_elastic_startup(owner)
    assert events == ["calibrate", "shutdown"]


def test_complete_elastic_startup_classifies_bounded_restart_as_expected(
    monkeypatch,
) -> None:
    from vllm.v1.engine.elastic_calibrator import ElasticCalibrationRestartRequired

    events = []
    owner = SimpleNamespace(
        scheduler=SimpleNamespace(
            elastic_on_demand_graphs=True,
            _elastic_require_catalog=True,
            _elastic_auto_calibrate=True,
            _elastic_graph_catalog={},
        ),
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    def checkpoint(_owner):
        events.append("checkpoint")
        raise ElasticCalibrationRestartRequired("epoch complete")

    warning = MagicMock()
    critical = MagicMock()
    monkeypatch.setattr(
        elastic_bootstrap, "_auto_calibrate_missing_catalog", checkpoint
    )
    monkeypatch.setattr(elastic_bootstrap.logger, "warning", warning)
    monkeypatch.setattr(elastic_bootstrap.logger, "critical", critical)

    with pytest.raises(ElasticCalibrationRestartRequired, match="epoch complete"):
        elastic_bootstrap.complete_elastic_startup(owner)

    assert events == ["checkpoint", "shutdown"]
    warning.assert_called_once()
    assert "ELASTIC_CALIBRATION_RESTART_REQUIRED" in warning.call_args.args[0]
    critical.assert_not_called()


def test_auto_calibration_seals_activates_and_records_receipt(
    tmp_path, monkeypatch
) -> None:
    from vllm import envs
    from vllm.v1.core import elastic_catalog
    from vllm.v1.engine import elastic_calibrator
    from vllm.v1.worker import startup_plan

    surface = tmp_path / "surface.json"
    surface.write_text("{}", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "elastic_graph_catalog" / "catalog.json"
    activated = []
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        activate_elastic_graph_catalog=lambda *args: activated.append(args),
    )
    owner = SimpleNamespace(vllm_config=object(), scheduler=scheduler)
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_SURFACE", str(surface))
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_RECEIPT", str(receipt_path))
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        elastic_catalog,
        "load_sealed_catalog_with_digest",
        lambda *_args, **_kwargs: ({"ok": 1}, "c" * 64),
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_args: "1123456789abcdef",
    )
    monkeypatch.setattr(
        startup_plan, "load_elastic_graph_catalog", lambda *_args: {"row": {}}
    )
    monkeypatch.setattr(
        startup_plan,
        "load_elastic_graph_catalog_coverage",
        lambda *_args: {"decode_max_x": 39},
    )
    monkeypatch.setattr(
        elastic_calibrator,
        "calibrate_and_publish_catalog",
        lambda *_args, **_kwargs: SimpleNamespace(
            destination=destination,
            fingerprint="1123456789abcdef",
            measured_shapes=51,
            wall_seconds=1.0,
        ),
    )

    elastic_bootstrap._auto_calibrate_missing_catalog(owner)

    assert activated == [({"row": {}}, {"decode_max_x": 39})]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["stage"] == "complete"
    assert receipt["measured_shapes"] == 51
    assert receipt["destination"] == str(destination)
    assert receipt["surface_sha256"] == "c" * 64


def test_auto_calibration_failure_receipt_does_not_activate(tmp_path, monkeypatch):
    from vllm import envs
    from vllm.v1.core import elastic_catalog
    from vllm.v1.engine import elastic_calibrator
    from vllm.v1.worker import startup_plan

    surface = tmp_path / "surface.json"
    surface.write_text("{}", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        activate_elastic_graph_catalog=MagicMock(),
    )
    owner = SimpleNamespace(vllm_config=object(), scheduler=scheduler)
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_SURFACE", str(surface))
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_RECEIPT", str(receipt_path))
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        elastic_catalog,
        "load_sealed_catalog_with_digest",
        lambda *_args, **_kwargs: ({"ok": 1}, "c" * 64),
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_args: "1123456789abcdef",
    )
    monkeypatch.setattr(
        elastic_calibrator,
        "calibrate_and_publish_catalog",
        MagicMock(side_effect=RuntimeError("shape contracted")),
    )

    with pytest.raises(RuntimeError, match="shape contracted"):
        elastic_bootstrap._auto_calibrate_missing_catalog(owner)

    scheduler.activate_elastic_graph_catalog.assert_not_called()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["stage"] == "failed"
    assert receipt["error"]["type"] == "RuntimeError"
    assert receipt["surface_sha256"] == "c" * 64


def test_auto_calibration_records_restartable_checkpoint(tmp_path, monkeypatch):
    from vllm import envs
    from vllm.v1.core import elastic_catalog
    from vllm.v1.engine import elastic_calibrator
    from vllm.v1.worker import startup_plan

    surface = tmp_path / "surface.json"
    surface.write_text("{}", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        activate_elastic_graph_catalog=MagicMock(),
    )
    owner = SimpleNamespace(vllm_config=object(), scheduler=scheduler)
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_SURFACE", str(surface))
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_RECEIPT", str(receipt_path))
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_MAX_NEW_ROWS_PER_PROCESS", "8")
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CALIBRATION_MAX_PRODUCER_EPOCHS_PER_PROCESS", "1"
    )
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        elastic_catalog,
        "load_sealed_catalog_with_digest",
        lambda *_args, **_kwargs: ({"ok": 1}, "c" * 64),
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_args: "1123456789abcdef",
    )
    checkpoint = {
        "schema": "ag2-elastic-calibration-checkpoint-v1",
        "fingerprint": "1123456789abcdef",
        "surface_sha256": "c" * 64,
        "process_epochs": 1,
        "calibration_wall_seconds": 1.0,
        "completed_shapes": 0,
        "rows": [],
        "mixed_query_witnesses": [],
    }

    seen_checkpoints = []

    def calibrate(*_args, **kwargs):
        assert kwargs["max_new_rows_per_process"] == 8
        assert kwargs["max_producer_epochs_per_process"] == 1
        seen_checkpoints.append(kwargs["checkpoint_payload"])
        kwargs["checkpoint_callback"](checkpoint)
        raise elastic_calibrator.ElasticCalibrationRestartRequired("epoch complete")

    monkeypatch.setattr(elastic_calibrator, "calibrate_and_publish_catalog", calibrate)

    for _epoch in range(2):
        with pytest.raises(
            elastic_calibrator.ElasticCalibrationRestartRequired,
            match="epoch complete",
        ):
            elastic_bootstrap._auto_calibrate_missing_catalog(owner)

    scheduler.activate_elastic_graph_catalog.assert_not_called()
    assert seen_checkpoints == [None, checkpoint]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["stage"] == "checkpointed"
    assert receipt["restart_required"] is True
    assert receipt["checkpoint"] == checkpoint


def test_auto_calibration_rejects_existing_canonical_before_measurement(
    tmp_path, monkeypatch
) -> None:
    from vllm import envs
    from vllm.v1.core import elastic_catalog
    from vllm.v1.engine import elastic_calibrator
    from vllm.v1.worker import startup_plan

    fingerprint = "1123456789abcdef"
    surface = tmp_path / "surface.json"
    surface.write_text("{}", encoding="utf-8")
    canonical = (
        tmp_path / "elastic_graph_catalog" / f"elastic_graph_catalog_{fingerprint}.json"
    )
    canonical.parent.mkdir()
    canonical.write_text("{broken", encoding="utf-8")
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        activate_elastic_graph_catalog=MagicMock(),
    )
    owner = SimpleNamespace(vllm_config=object(), scheduler=scheduler)
    calibrate = MagicMock()
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_SURFACE", str(surface))
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        elastic_catalog,
        "load_sealed_catalog_with_digest",
        lambda *_args, **_kwargs: ({"ok": 1}, "c" * 64),
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_args: fingerprint,
    )
    monkeypatch.setattr(
        elastic_calibrator,
        "calibrate_and_publish_catalog",
        calibrate,
    )

    with pytest.raises(RuntimeError, match="existing canonical destination"):
        elastic_bootstrap._auto_calibrate_missing_catalog(owner)

    calibrate.assert_not_called()
    scheduler.activate_elastic_graph_catalog.assert_not_called()
