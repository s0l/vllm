# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine import FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.worker_failure import (
    WorkerFailure,
    WorkerFailureCode,
    WorkerRemoteError,
)


def _remote_failure(
    code: WorkerFailureCode, message: str, *, remote_type: str = "remote.Error"
) -> WorkerRemoteError:
    return WorkerRemoteError(WorkerFailure(code, remote_type, message))


def _core_for_results(results: list[object]) -> tuple[SimpleNamespace, Mock]:
    scheduler_output = SimpleNamespace(
        is_pure_decode_step=False,
        num_scheduled_tokens={"request-0": 1},
        total_num_scheduled_tokens=1,
    )
    scheduler = Mock()
    scheduler.has_requests.return_value = True
    scheduler.has_pending_elastic_maintenance.return_value = False
    scheduler.schedule.return_value = scheduler_output
    scheduler.get_grammar_bitmask.return_value = None
    scheduler.take_finished_request_ids.return_value = {0: {"request-0"}}

    model_executor = Mock()

    def execute_model(_scheduler_output, *, non_block):
        assert non_block
        future: Future[object] = Future()
        result = results.pop(0)
        if isinstance(result, BaseException):
            future.set_exception(result)
        else:
            future.set_result(result)
        return future

    model_executor.execute_model.side_effect = execute_model
    core = SimpleNamespace(
        scheduler=scheduler,
        model_executor=model_executor,
        _ag2_step_trace_path="",
        _ag2_step_trace_complete=False,
        _elastic_restore_expected_decode_ids=frozenset(),
        _should_throttle_prefills=lambda: False,
        _has_scheduler_step_work=lambda: True,
        capture_iteration_details=lambda _output: nullcontext(None),
        log_error_detail=lambda _output: nullcontext(),
        _process_aborts_queue=lambda: None,
        _attach_iteration_details=lambda _outputs, _details: None,
    )
    return core, scheduler


def test_engine_core_recovers_manifest_mismatch_and_accepts_next_step() -> None:
    valid_outputs = {0: SimpleNamespace(outputs=[])}
    core, scheduler = _core_for_results(
        [
            _remote_failure(
                WorkerFailureCode.ELASTIC_EXECUTION_PLAN_MISMATCH,
                "ELASTIC_EXECUTION_PLAN_MISMATCH: owner=target",
            ),
            object(),
        ]
    )
    request = SimpleNamespace(
        request_id="request-0",
        client_index=0,
        trace_headers=None,
        take_events=lambda: [],
    )
    scheduler.recover_elastic_execution_plan_mismatch.return_value = [request]
    scheduler.update_from_output.return_value = valid_outputs

    recovered, executed = EngineCore.step(core)

    assert not executed
    assert recovered[0].finished_requests == {"request-0"}
    assert recovered[0].outputs[0].finish_reason == FinishReason.ERROR
    scheduler.update_from_output.assert_not_called()
    scheduler.take_finished_request_ids.assert_called_once_with()

    following, executed = EngineCore.step(core)

    assert executed
    assert following is valid_outputs
    scheduler.update_from_output.assert_called_once()


def test_engine_core_mismatch_drains_concurrent_abort_before_recovery() -> None:
    core, scheduler = _core_for_results(
        [
            _remote_failure(
                WorkerFailureCode.ELASTIC_EXECUTION_PLAN_MISMATCH,
                "ELASTIC_EXECUTION_PLAN_MISMATCH: owner=target",
            )
        ]
    )
    failed = SimpleNamespace(
        request_id="request-0",
        client_index=0,
        trace_headers=None,
        take_events=lambda: [],
    )
    order: list[str] = []

    def process_aborts():
        order.append("abort")

    def recover(_scheduler_output):
        order.append("recover")
        return [failed]

    core._process_aborts_queue = process_aborts
    scheduler.recover_elastic_execution_plan_mismatch.side_effect = recover
    scheduler.take_finished_request_ids.return_value = {
        0: {"request-0"},
        1: {"request-aborted"},
    }

    outputs, executed = EngineCore.step(core)

    assert not executed
    assert order == ["abort", "recover"]
    assert outputs[0].finished_requests == {"request-0"}
    assert [output.request_id for output in outputs[0].outputs] == ["request-0"]
    assert outputs[1].finished_requests == {"request-aborted"}
    assert outputs[1].outputs == []


def test_engine_core_routes_request_free_graph_mismatch_to_recovery() -> None:
    core, scheduler = _core_for_results(
        [
            _remote_failure(
                WorkerFailureCode.ELASTIC_EXECUTION_PLAN_MISMATCH,
                "ELASTIC_EXECUTION_PLAN_MISMATCH: graph-only rank drift",
            )
        ]
    )
    scheduler.schedule.return_value = SimpleNamespace(
        is_pure_decode_step=False,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
    )
    failed = SimpleNamespace(
        request_id="request-0",
        client_index=0,
        trace_headers=None,
        take_events=lambda: [],
    )
    scheduler.recover_elastic_execution_plan_mismatch.return_value = [failed]
    scheduler.take_finished_request_ids.return_value = {0: {"request-0"}}

    outputs, executed = EngineCore.step(core)

    assert not executed
    assert outputs[0].finished_requests == {"request-0"}
    assert outputs[0].outputs[0].finish_reason == FinishReason.ERROR
    scheduler.recover_elastic_execution_plan_mismatch.assert_called_once_with(
        scheduler.schedule.return_value
    )


def test_engine_core_does_not_relabel_unrelated_worker_failure() -> None:
    core, scheduler = _core_for_results(
        [_remote_failure(WorkerFailureCode.GENERIC, "CUDA illegal access")]
    )

    try:
        EngineCore.step(core)
    except RuntimeError as error:
        assert "CUDA illegal access" in str(error)
    else:
        raise AssertionError("unrelated worker failure was swallowed")
    scheduler.recover_elastic_execution_plan_mismatch.assert_not_called()


def test_engine_core_never_rolls_back_post_materialization_failure() -> None:
    core, scheduler = _core_for_results(
        [
            _remote_failure(
                WorkerFailureCode.GENERIC,
                "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH: all-rank input "
                "materialization vote rejected before model collectives",
            )
        ]
    )

    try:
        EngineCore.step(core)
    except RuntimeError as error:
        assert "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH:" in str(error)
    else:
        raise AssertionError("post-materialization failure was swallowed")
    scheduler.recover_elastic_execution_plan_mismatch.assert_not_called()


def test_engine_core_does_not_recover_embedded_pre_mutation_marker() -> None:
    core, scheduler = _core_for_results(
        [
            _remote_failure(
                WorkerFailureCode.GENERIC,
                "outer worker failure caused by "
                "ELASTIC_EXECUTION_PLAN_MISMATCH: stale inner diagnostic",
            )
        ]
    )

    try:
        EngineCore.step(core)
    except RuntimeError as error:
        assert "outer worker failure" in str(error)
    else:
        raise AssertionError("embedded mismatch marker was swallowed")
    scheduler.recover_elastic_execution_plan_mismatch.assert_not_called()
