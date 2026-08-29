# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm.v1.worker import worker_base
from vllm.v1.worker.worker_base import WorkerBase

pytestmark = pytest.mark.cpu_test


def test_elastic_graph_workspace_receipt_binds_generation(monkeypatch):
    worker = object.__new__(WorkerBase)
    worker.model_runner = SimpleNamespace(
        _dynamic_graph_working_set=lambda: SimpleNamespace(
            managers=(
                SimpleNamespace(runtime_generation="generation-a"),
                SimpleNamespace(runtime_generation="generation-a"),
            )
        )
    )
    monkeypatch.setattr(
        worker_base.torch._C,
        "_cuda_getCublasWorkspaceSize",
        lambda: 33554432,
    )

    assert worker.get_elastic_graph_workspace_receipt() == (
        "generation-a",
        33554432,
    )


def test_elastic_graph_workspace_receipt_rejects_manager_generation_split():
    worker = object.__new__(WorkerBase)
    worker.model_runner = SimpleNamespace(
        _dynamic_graph_working_set=lambda: SimpleNamespace(
            managers=(
                SimpleNamespace(runtime_generation="generation-a"),
                SimpleNamespace(runtime_generation="generation-b"),
            )
        )
    )

    with pytest.raises(RuntimeError, match="generation"):
        worker.get_elastic_graph_workspace_receipt()
