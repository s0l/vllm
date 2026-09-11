# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.third_party.flash_linear_attention.ops.layernorm_guard as norm
from vllm.utils.math_utils import cdiv, next_power_of_2


@pytest.mark.parametrize("sm_count", [1, 2, 8, 36, 128, 256])
def test_rows_per_block_preserves_launch_geometry(monkeypatch, sm_count):
    monkeypatch.setattr(norm, "num_compute_units", lambda _device: sm_count)
    device = torch.device("cuda", 0)
    for rows in range(262145):
        expected = min(next_power_of_2(cdiv(rows, 2 * sm_count)), 4)
        assert norm.calc_rows_per_block(rows, device) == expected


@pytest.mark.parametrize("rows_per_token", [2, 15, 18, 64])
def test_rows_per_block_has_bounded_dynamo_regions(monkeypatch, rows_per_token):
    sm_count = 36
    monkeypatch.setattr(norm, "num_compute_units", lambda _device: sm_count)
    graphs = []

    def backend(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    def consume(x):
        return x * norm.calc_rows_per_block(x.shape[0], x.device)

    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(recompile_limit=8):
            compiled = torch.compile(
                consume, backend=backend, dynamic=True, fullgraph=True
            )
            tokens = [
                4096,
                1,
                2,
                3,
                4,
                5,
                7,
                8,
                9,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
            ]
            for count in tokens + [31, 17, 2047, 63, 5]:
                rows = count * rows_per_token
                x = torch.randn(rows, 1)
                expected = min(next_power_of_2(cdiv(rows, 2 * sm_count)), 4)
                assert torch.equal(compiled(x), x * expected)
            assert len(graphs) <= 3
    finally:
        torch._dynamo.reset()
