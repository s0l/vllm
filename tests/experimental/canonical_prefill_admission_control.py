#!/usr/bin/env python3
"""Exact-image control for residual waiting-prefill admission."""

from __future__ import annotations

import os
from unittest.mock import patch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput


def _make_scheduler(enabled: bool):
    value = "1" if enabled else "0"
    with patch.dict(
        os.environ,
        {"AG2_VLLM_CANONICAL_PREFILL_ADMISSION": value},
    ):
        scheduler = create_scheduler(
            max_num_batched_tokens=7056,
            max_model_len=20000,
        )
    requests = create_requests(num_requests=8, num_tokens=1513)
    for request in requests:
        scheduler.add_request(request)
    return scheduler, requests


def main() -> int:
    scheduler, requests = _make_scheduler(enabled=False)
    disabled = scheduler.schedule()
    assert disabled.total_num_scheduled_tokens == 7056
    assert disabled.num_scheduled_tokens[requests[4].request_id] == 1004
    assert list(scheduler.running) == requests[:5]
    assert list(scheduler.waiting) == requests[5:]

    scheduler, requests = _make_scheduler(enabled=True)
    first = scheduler.schedule()
    assert first.total_num_scheduled_tokens == 4 * 1513
    assert first.num_scheduled_tokens == {
        request.request_id: 1513 for request in requests[:4]
    }
    assert list(scheduler.running) == requests[:4]
    assert list(scheduler.waiting) == requests[4:]
    assert scheduler.num_canonical_prefill_deferrals_since_last_stats == 1

    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=[request.request_id for request in requests[:4]],
            req_id_to_index={
                request.request_id: i for i, request in enumerate(requests[:4])
            },
            sampled_token_ids=[[0] for _ in requests[:4]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    second = scheduler.schedule()
    assert second.total_num_scheduled_tokens == 4 + 4 * 1513
    assert all(
        request.request_id in second.num_scheduled_tokens for request in requests
    )
    assert not scheduler.waiting
    assert sum(first.num_scheduled_tokens.values()) + sum(
        second.num_scheduled_tokens[request.request_id] for request in requests[4:]
    ) == 8 * 1513
    print(
        "canonical_prefill_admission_control: PASS "
        "disabled_residual=1004 enabled_steps=6052,6056 token_loss=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
