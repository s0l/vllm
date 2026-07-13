import pytest

from vllm.experimental.dual_view_nvfp4 import (
    qwopus_mlp_row_segments,
    validate_qwopus_mlp_row_plan,
)


def test_qwopus_mlp_row_plan_covers_both_views_exactly():
    validate_qwopus_mlp_row_plan()


def test_qwopus_mlp_row_plan_rejects_other_ranks():
    with pytest.raises(ValueError):
        qwopus_mlp_row_segments(3)


def test_qwopus_prefill_segment_order_matches_contiguous_halves():
    rank0 = qwopus_mlp_row_segments(0)
    rank1 = qwopus_mlp_row_segments(1)
    assert [item.global_start for item in rank0] == [0, 5824]
    assert sum(item.width for item in rank0) == 8704
    assert [item.global_start for item in rank1] == [8704, 11648]
    assert sum(item.width for item in rank1) == 8704
