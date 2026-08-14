# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.model_executor.models.ag2_folded_mtp import (
    FoldedMTPAttentionBridge,
    FoldedMTPGeometry,
    _gemma_rms_norm,
)


def test_folded_mtp_q_ownership_is_exact() -> None:
    geometry = FoldedMTPGeometry()

    assert [geometry.dense_q_range(rank) for rank in (0, 1)] == [
        (0, 12),
        (12, 24),
    ]
    assert [geometry.attention_q_range(rank) for rank in (0, 1, 2)] == [
        (0, 8),
        (8, 16),
        (16, 24),
    ]
    geometry.validate()


def test_folded_mtp_transport_covers_only_cross_owner_q_spans() -> None:
    geometry = FoldedMTPGeometry()

    assert [
        (item.source, item.destination, item.tensor, item.head_start, item.head_end)
        for item in geometry.input_transfers()
    ] == [
        (0, 1, "q", 8, 12),
        (1, 2, "q", 16, 24),
        (0, 1, "kv", 0, 2),
        (1, 0, "kv", 2, 4),
        (0, 2, "kv", 0, 2),
        (1, 2, "kv", 2, 4),
    ]
    assert [item.head_count for item in geometry.output_transfers()] == [4, 8]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_q_heads": 32}, "Q24/KV4"),
        ({"total_kv_heads": 8}, "Q24/KV4"),
        ({"dense_pair_size": 3}, "natural TP2"),
        ({"attention_world_size": 2}, "TP3/DCP3"),
    ],
)
def test_folded_mtp_rejects_unproven_geometry(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FoldedMTPGeometry(**kwargs)


def test_folded_mtp_rejects_invalid_rank() -> None:
    geometry = FoldedMTPGeometry()
    with pytest.raises(ValueError, match="dense-pair"):
        geometry.dense_q_range(2)
    with pytest.raises(ValueError, match="attention owner"):
        geometry.attention_q_range(3)


class _Attention:
    num_heads = 8
    num_kv_heads = 4
    head_size = 256


class _Transport:
    head_dim = 256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_heads", 12, "8 Q heads"),
        ("num_kv_heads", 2, "all 4 KV heads"),
        ("head_size", 128, "head size mismatch"),
    ],
)
def test_folded_mtp_bridge_rejects_wrong_attention_contract(
    field: str, value: int, message: str
) -> None:
    attention = _Attention()
    setattr(attention, field, value)
    with pytest.raises(ValueError, match=message):
        FoldedMTPAttentionBridge(attention, _Transport())  # type: ignore[arg-type]


def test_natural_tp2_uses_gemma_rms_norm_checkpoint_semantics() -> None:
    value = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor(
        [-0.5, 0.0, 0.25, 1.0],
        dtype=torch.bfloat16,
    )
    epsilon = 1e-6
    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    expected = (normalized * (1.0 + weight.float())).to(value.dtype)
    rejected_old_formula = (normalized * weight.float()).to(value.dtype)

    actual = _gemma_rms_norm(value, weight, epsilon)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, rejected_old_formula)
