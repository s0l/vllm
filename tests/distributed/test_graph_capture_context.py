# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.distributed import parallel_state


class _CaptureGroup:
    def __init__(self):
        self.contexts = []

    @contextmanager
    def graph_capture(self, context):
        self.contexts.append(context)
        yield context


def test_graph_capture_reuses_explicit_context_for_every_group():
    context = SimpleNamespace(stream=object())
    tp, pp, dp = _CaptureGroup(), _CaptureGroup(), _CaptureGroup()

    with (
        patch.object(parallel_state, "get_tp_group", return_value=tp),
        patch.object(parallel_state, "get_pp_group", return_value=pp),
        patch.object(parallel_state, "get_dp_group", return_value=dp),
        patch.object(torch.cuda, "Stream", side_effect=AssertionError),
        parallel_state.graph_capture(
            torch.device("cuda:0"), graph_capture_context=context
        ) as observed,
    ):
        assert observed is context

    assert tp.contexts == [context]
    assert pp.contexts == [context]
    assert dp.contexts == [context]
