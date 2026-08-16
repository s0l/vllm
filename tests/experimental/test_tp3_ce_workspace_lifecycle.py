# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.distributed.device_communicators.tp3_ce_all_reduce import (
    _workspace_layout,
)


def test_workspace_uses_physical_runner_bound_not_context_ceiling(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TP3_CE_QUANT_BLOCK", "8192")
    runtime = _workspace_layout(max_rows=4096, cols=5120, world_size=3)
    context = _workspace_layout(max_rows=12288, cols=5120, world_size=3)

    assert runtime == {
        "quant_block": 8192,
        "num_blocks": 1,
        "max_payload": 20_987_904,
        "host_bytes": 62_967_808,
        "gpu_bytes": 62_963_712,
        "header_bytes": 4096,
    }
    assert context["gpu_bytes"] == 188_891_136
    assert context["gpu_bytes"] - runtime["gpu_bytes"] == 125_927_424


@pytest.mark.parametrize("max_rows,cols,world_size", [(0, 5120, 3), (1, 0, 3), (1, 1, 0)])
def test_workspace_layout_rejects_nonpositive_dimensions(
    max_rows: int, cols: int, world_size: int
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _workspace_layout(max_rows, cols, world_size)


def test_workspace_layout_rejects_nonpositive_quant_block(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TP3_CE_QUANT_BLOCK", "0")
    with pytest.raises(ValueError, match="QUANT_BLOCK"):
        _workspace_layout(4096, 5120, 3)
