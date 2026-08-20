from types import SimpleNamespace

import pytest

from vllm.profiler.wrapper import _should_create_profiler_schedule


@pytest.mark.parametrize(
    ("wait", "warmup", "active", "expected"),
    [
        (0, 0, 0, False),
        (0, 0, 5, True),
        (1, 0, 0, True),
        (0, 1, 0, True),
    ],
)
def test_should_create_profiler_schedule(
    wait: int, warmup: int, active: int, expected: bool
) -> None:
    config = SimpleNamespace(
        wait_iterations=wait,
        warmup_iterations=warmup,
        active_iterations=active,
    )

    assert _should_create_profiler_schedule(config) is expected
