#!/usr/bin/env python3
"""Exact-image control for residual waiting-prefill admission."""

from __future__ import annotations

import os
from unittest.mock import patch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.platforms import current_platform


def _make_scheduler(enabled: bool):
    value = "1" if enabled else "0"
    with (
        patch.dict(
            os.environ,
            {"AG2_VLLM_CANONICAL_PREFILL_ADMISSION": value},
        ),
        patch.object(current_platform, "device_type", "cpu"),
    ):
        scheduler = create_scheduler(
            max_num_batched_tokens=7056,
            max_model_len=2048,
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

    try:
        _make_scheduler(enabled=True)
    except ValueError as exc:
        assert "is retired" in str(exc)
    else:
        raise AssertionError("retired canonical prefill admission was enabled")
    print(
        "canonical_prefill_admission_control: PASS "
        "residual=1004 retired_serialization_rejected=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
