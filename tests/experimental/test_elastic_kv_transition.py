# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _Owner:
    def __init__(self, committed: int, quantum: int = 4):
        self.committed = committed
        self.quantum = quantum
        self.resizes: list[int] = []

    @property
    def info(self):
        return SimpleNamespace(committed=self.committed, quantum=self.quantum)

    def resize(self, committed: int, event) -> None:
        self.committed = committed
        self.resizes.append(committed)


class _Event:
    def record(self, stream) -> None:
        pass


class TestElasticKVTransition(unittest.TestCase):
    def _runner(self):
        runner = object.__new__(GPUModelRunner)
        runner.device = torch.device("cpu")
        runner.elastic_kv_backings = {
            "elastic-attention-0": _Owner(16),
            "elastic-gdn": _Owner(8),
        }
        runner.elastic_kv_geometry = {
            "elastic-attention-0": 4,
            "elastic-gdn": 4,
        }
        return runner

    def test_vote_is_scoped_to_tp_group(self):
        runner = self._runner()
        tp_group = object()
        calls = []

        def all_reduce(vote, *, op, group):
            calls.append((op, group))

        with (
            patch.object(torch.cuda, "Event", return_value=_Event()),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_reduce", side_effect=all_reduce),
            patch(
                "vllm.v1.worker.gpu_model_runner.get_tp_group",
                return_value=SimpleNamespace(device_group=tp_group),
            ),
        ):
            runner._apply_elastic_kv_transition((5, 3))

        self.assertEqual(calls, [(torch.distributed.ReduceOp.MIN, tp_group)])

    def test_remote_failure_rolls_back_local_resize(self):
        runner = self._runner()

        def reject(vote, *, op, group):
            vote.zero_()

        with (
            patch.object(torch.cuda, "Event", return_value=_Event()),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_reduce", side_effect=reject),
            patch(
                "vllm.v1.worker.gpu_model_runner.get_tp_group",
                return_value=SimpleNamespace(device_group=object()),
            ),
            self.assertRaisesRegex(RuntimeError, "aborted"),
        ):
            runner._apply_elastic_kv_transition((5, 3))

        self.assertEqual(
            runner.elastic_kv_backings["elastic-attention-0"].committed, 16
        )
        self.assertEqual(runner.elastic_kv_backings["elastic-gdn"].committed, 8)


if __name__ == "__main__":
    unittest.main()
