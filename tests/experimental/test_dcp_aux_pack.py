# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import flashinfer


def _packs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((5, 12), torch.nan),
        torch.full((5, 5), torch.nan),
    )


def test_decode_aux_pack_selects_first_row_per_request(monkeypatch) -> None:
    monkeypatch.setattr(
        flashinfer,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=3),
    )
    layer = SimpleNamespace(num_heads=1, head_size=2)
    output_pack, lse_pack = _packs()
    values = torch.arange(20 * 3 * 2, dtype=torch.float32).reshape(20, 3, 2)

    flashinfer._write_ag2_dcp_aux_pack(
        layer,
        "local_output",
        values,
        output_pack,
        lse_pack,
        request_tail_count=5,
        request_row_stride=4,
    )

    assert torch.equal(output_pack[:, :6], values[::4].reshape(5, 6))
    assert torch.isnan(output_pack[:, 6:]).all()


def test_decode_aux_pack_rejects_two_row_selectors(monkeypatch) -> None:
    monkeypatch.setattr(
        flashinfer,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=3),
    )
    layer = SimpleNamespace(num_heads=1, head_size=2)
    output_pack, lse_pack = _packs()

    with pytest.raises(RuntimeError, match="cannot combine"):
        flashinfer._write_ag2_dcp_aux_pack(
            layer,
            "local_output",
            torch.zeros((20, 3, 2)),
            output_pack,
            lse_pack,
            request_tail_indices=torch.tensor([3, 7, 11, 15, 19]),
            request_tail_count=5,
            request_row_stride=4,
        )
