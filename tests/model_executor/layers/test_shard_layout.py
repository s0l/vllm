# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.shard_layout import ShardLayout, ShardSpan


@pytest.mark.parametrize(
    ("logical_size", "padded_size", "tp_size"),
    [
        (1, 3, 3),
        (16, 18, 3),
        (4304, 4305, 3),
        (8192, 8192, 4),
    ],
)
def test_contiguous_padded_layout_exact_coverage(
    logical_size: int,
    padded_size: int,
    tp_size: int,
):
    layouts = tuple(
        ShardLayout.contiguous_padded(
            logical_size=logical_size,
            padded_size=padded_size,
            tp_size=tp_size,
            tp_rank=rank,
        )
        for rank in range(tp_size)
    )
    ShardLayout.validate_exact_source_partition(layouts)

    reconstructed = []
    for layout in layouts:
        local = [None] * layout.destination_size
        for span in layout.spans:
            for offset in range(span.size):
                local[span.destination_start + offset] = span.source_start + offset
        reconstructed.extend(value for value in local if value is not None)
    assert reconstructed == list(range(logical_size))


def test_explicit_layout_exact_coverage_and_padding():
    ranges = ((0, 6), (6, 5), (11, 5))
    layouts = tuple(
        ShardLayout.explicit(
            logical_size=16,
            destination_start=7,
            destination_size=6,
            source_start=start,
            source_size=size,
        )
        for start, size in ranges
    )
    ShardLayout.validate_exact_source_partition(layouts)
    assert [
        layout.destination_size - sum(s.size for s in layout.spans)
        for layout in layouts
    ] == [
        0,
        1,
        1,
    ]


@pytest.mark.parametrize(
    "layouts",
    [
        (
            ShardLayout(4, 0, 2, (ShardSpan(0, 0, 2),)),
            ShardLayout(4, 0, 2, (ShardSpan(1, 0, 2),)),
        ),
        (
            ShardLayout(4, 0, 2, (ShardSpan(0, 0, 2),)),
            ShardLayout(4, 0, 1, (ShardSpan(3, 0, 1),)),
        ),
    ],
)
def test_exact_partition_rejects_source_overlap_or_gap(layouts):
    with pytest.raises(ValueError, match="gap or overlap"):
        ShardLayout.validate_exact_source_partition(layouts)


def test_layout_rejects_destination_overlap():
    with pytest.raises(ValueError, match="destination spans overlap"):
        ShardLayout(
            logical_size=4,
            destination_start=0,
            destination_size=4,
            spans=(ShardSpan(0, 0, 2), ShardSpan(2, 1, 2)),
        )
