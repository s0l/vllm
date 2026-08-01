# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.distributed.parallel_state import (
    _should_use_tp3_ce,
    _should_use_tp3_ce_physical,
    _should_use_tp3_piecewise_device_ce,
    _should_use_tp3_sd_canonical_reduce,
    _should_use_tp3_sd_deterministic_reduce,
    _tp3_sd_canonical_reduce,
    _tp3_sd_deterministic_reduce,
    _tp3_sd_deterministic_sum,
)


def test_tp3_device_ce_covers_piecewise_and_intermediate_prefill():
    common = {
        "tensor_dim": 2,
        "rows": 4464,
        "hidden_size": 5120,
        "tp_world_size": 3,
    }
    assert _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("PIECEWISE"),
        **common,
    )
    assert not _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("FULL"),
        **common,
    )
    assert _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("NONE"),
        **(common | {"rows": 845}),
    )
    assert _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("NONE"),
        **(common | {"rows": 1344}),
    )
    assert not _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("NONE"),
        **(common | {"rows": 24}),
    )
    assert not _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("NONE"),
        **(common | {"rows": 4096}),
    )
    assert not _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=None,
        **common,
    )
    assert not _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("PIECEWISE"),
        **(common | {"rows": 0}),
    )
    assert not _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("PIECEWISE"),
        **(common | {"hidden_size": 4096}),
    )
    assert not _should_use_tp3_piecewise_device_ce(
        cudagraph_mode=_Mode("PIECEWISE"),
        **(common | {"tp_world_size": 2}),
    )


def test_tp3_ce_requires_known_large_logical_batch():
    args = (3, 2, 5120, 3)
    assert not _should_use_tp3_ce(None, *args)
    assert not _should_use_tp3_ce(1365, *args)
    assert _should_use_tp3_ce(1366, *args)


def test_tp3_ce_rejects_unsupported_layouts():
    assert not _should_use_tp3_ce(4096, 1, 1, 5120, 3)
    assert not _should_use_tp3_ce(4096, 1, 2, 4096, 3)
    assert not _should_use_tp3_ce(4096, 1, 2, 5120, 2)


def test_tp3_ce_physical_fallback_is_conservative():
    assert not _should_use_tp3_ce_physical(False, 7056, 2, 5120, 3)
    assert not _should_use_tp3_ce_physical(True, 4095, 2, 5120, 3)
    assert _should_use_tp3_ce_physical(True, 4096, 2, 5120, 3)
    assert _should_use_tp3_ce_physical(True, 7056, 2, 5120, 3)
    assert not _should_use_tp3_ce_physical(True, 7056, 1, 5120, 3)
    assert not _should_use_tp3_ce_physical(True, 7056, 2, 4096, 3)
    assert not _should_use_tp3_ce_physical(True, 7056, 2, 5120, 2)


class _Mode:
    def __init__(self, name: str):
        self.name = name


def test_tp3_sd_canonical_reduce_is_full_decode_only():
    args = (True, _Mode("FULL"), 2, 1, 5120, 3)
    assert _should_use_tp3_sd_canonical_reduce(*args)
    assert _should_use_tp3_sd_canonical_reduce(True, _Mode("FULL"), 2, 8, 5120, 3)
    assert not _should_use_tp3_sd_canonical_reduce(
        False, _Mode("FULL"), 2, 1, 5120, 3
    )
    assert not _should_use_tp3_sd_canonical_reduce(
        True, _Mode("PIECEWISE"), 2, 1, 5120, 3
    )
    assert not _should_use_tp3_sd_canonical_reduce(
        True, _Mode("FULL"), 2, 9, 5120, 3
    )
    assert not _should_use_tp3_sd_canonical_reduce(
        True, _Mode("FULL"), 1, 1, 5120, 3
    )
    assert not _should_use_tp3_sd_canonical_reduce(
        True, _Mode("FULL"), 2, 1, 4096, 3
    )
    assert not _should_use_tp3_sd_canonical_reduce(
        True, _Mode("FULL"), 2, 1, 5120, 2
    )


def test_tp3_sd_canonical_reduce_pads_and_slices():
    source = torch.arange(2 * 4, dtype=torch.float32).view(2, 4)
    observed_shape = None

    def reduce_fn(padded: torch.Tensor) -> torch.Tensor:
        nonlocal observed_shape
        observed_shape = tuple(padded.shape)
        return padded * 3

    actual = _tp3_sd_canonical_reduce(source, reduce_fn)
    assert observed_shape == (6, 4)
    torch.testing.assert_close(actual, source * 3, rtol=0, atol=0)


def test_tp3_sd_deterministic_reduce_covers_matched_small_lanes():
    assert _should_use_tp3_sd_deterministic_reduce(True, 2, 1, 5120, 3)
    assert _should_use_tp3_sd_deterministic_reduce(True, 2, 3, 5120, 3)
    assert _should_use_tp3_sd_deterministic_reduce(True, 2, 24, 5120, 3)
    assert not _should_use_tp3_sd_deterministic_reduce(True, 2, 25, 5120, 3)
    assert not _should_use_tp3_sd_deterministic_reduce(
        False, 2, 1, 5120, 3
    )
    assert not _should_use_tp3_sd_deterministic_reduce(True, 1, 1, 5120, 3)
    assert not _should_use_tp3_sd_deterministic_reduce(True, 2, 1, 4096, 3)
    assert not _should_use_tp3_sd_deterministic_reduce(True, 2, 1, 5120, 2)


def test_tp3_sd_deterministic_sum_is_shape_independent_and_fp32_rounded():
    rank_values = torch.tensor(
        [
            [[1.0, 0.333984375, 0.001953125]],
            [[0.00390625, 0.333984375, -0.001953125]],
            [[-1.0, 0.333984375, 0.0009765625]],
        ],
        dtype=torch.bfloat16,
    )
    qlen1 = _tp3_sd_deterministic_sum(rank_values)
    qlen3_values = torch.zeros((3, 3, 3), dtype=torch.bfloat16)
    qlen3_values[:, 0] = rank_values[:, 0]
    qlen3_values[:, 1:] = torch.tensor(
        [[2.0, -3.0, 4.0], [-5.0, 6.0, -7.0]], dtype=torch.bfloat16
    )
    qlen3 = _tp3_sd_deterministic_sum(qlen3_values)
    expected = (
        rank_values[0].float()
        + rank_values[1].float()
        + rank_values[2].float()
    ).bfloat16()
    torch.testing.assert_close(qlen1, expected, rtol=0, atol=0)
    torch.testing.assert_close(qlen3[0], qlen1[0], rtol=0, atol=0)


def test_tp3_sd_deterministic_sum_rejects_wrong_world_size():
    try:
        _tp3_sd_deterministic_sum(torch.zeros((2, 1, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong TP world size was accepted")


def test_tp3_sd_deterministic_reduce_gathers_rank_major_on_dim_zero():
    local = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    rank_major = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.bfloat16
    )

    class _Communicator:
        def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
            assert tensor is local
            assert dim == 0
            return rank_major

    class _Group:
        device_communicator = _Communicator()

    actual = _tp3_sd_deterministic_reduce(local, _Group())
    expected = torch.tensor([[9.0, 12.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
