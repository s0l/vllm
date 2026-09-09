# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.elastic_graph import RuntimeGeneration
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker

pytestmark = pytest.mark.cpu_test


def test_elastic_graph_residency_receipt_delegates_complete_state(monkeypatch):
    expected = Mock(generation=RuntimeGeneration("generation-a"))
    working_set = Mock()
    working_set.residency_receipt.return_value = expected
    model_runner = SimpleNamespace(
        cudagraph_manager=object(),
        _current_dynamic_graph_receipt=lambda: (64, 16, 24),
        _elastic_last_step_peak_external_bytes=80,
        _dynamic_graph_working_set=lambda: working_set,
    )
    worker = object.__new__(Worker)
    worker.model_runner = model_runner
    monkeypatch.setattr(
        gpu_worker.torch._C,
        "_cuda_getCublasWorkspaceSize",
        lambda: 33554432,
    )

    assert worker.get_elastic_graph_residency_receipt() is expected
    working_set.residency_receipt.assert_called_once_with(
        transaction_id=None,
        resident_bytes=64,
        floor_bytes=16,
        transition_floor_bytes=24,
        peak_bytes=80,
        cublas_workspace_bytes=33554432,
    )


def test_elastic_graph_residency_receipt_requires_graph_manager():
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(cudagraph_manager=None)

    with pytest.raises(RuntimeError, match="requires Graph manager"):
        worker.get_elastic_graph_residency_receipt()
