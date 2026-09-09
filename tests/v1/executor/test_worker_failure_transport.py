# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.executor.worker_failure import WorkerFailureCode, WorkerRemoteError
from vllm.v1.worker.gpu.cudagraph_utils import ElasticExecutionPlanMismatch


def _round_trip_worker_failure(error: Exception) -> tuple[object, object]:
    response_mq = Mock()
    worker = SimpleNamespace(worker_response_mq=response_mq)
    WorkerProc.enqueue_output(worker, error)
    (result,), _kwargs = response_mq.enqueue.call_args
    return pickle.loads(pickle.dumps(result))


def _executor_for_response(response: tuple[object, object]) -> SimpleNamespace:
    response_mq = Mock()
    response_mq.dequeue.return_value = response
    return SimpleNamespace(
        rpc_broadcast_mq=Mock(),
        response_mqs=(response_mq,),
        output_rank=0,
        futures_queue=deque(),
        is_failed=False,
    )


def test_elastic_mismatch_survives_worker_response_transport() -> None:
    response = _round_trip_worker_failure(
        ElasticExecutionPlanMismatch(
            "ELASTIC_EXECUTION_PLAN_MISMATCH: exact owner drift"
        )
    )
    executor = _executor_for_response(response)

    with pytest.raises(WorkerRemoteError) as exc_info:
        MultiprocExecutor.collective_rpc(
            executor,
            "execute_model",
            unique_reply_rank=0,
        )

    assert (
        exc_info.value.failure.code == WorkerFailureCode.ELASTIC_EXECUTION_PLAN_MISMATCH
    )
    assert exc_info.value.failure.message.endswith("exact owner drift")


def test_unrelated_worker_failure_remains_generic_and_fatal() -> None:
    response = _round_trip_worker_failure(RuntimeError("CUDA illegal access"))
    executor = _executor_for_response(response)

    with pytest.raises(WorkerRemoteError) as exc_info:
        MultiprocExecutor.collective_rpc(
            executor,
            "execute_model",
            unique_reply_rank=0,
        )

    assert exc_info.value.failure.code == WorkerFailureCode.GENERIC
    assert exc_info.value.failure.message == "CUDA illegal access"
