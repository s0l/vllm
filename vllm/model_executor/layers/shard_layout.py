# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShardSpan:
    source_start: int
    destination_start: int
    size: int

    @property
    def source_end(self) -> int:
        return self.source_start + self.size

    @property
    def destination_end(self) -> int:
        return self.destination_start + self.size


@dataclass(frozen=True, slots=True)
class ShardLayout:
    """One rank's logical source spans and padded local destination range."""

    logical_size: int
    destination_start: int
    destination_size: int
    spans: tuple[ShardSpan, ...]

    def __post_init__(self) -> None:
        if self.logical_size < 0:
            raise ValueError("logical size must be non-negative")
        if self.destination_start < 0 or self.destination_size < 0:
            raise ValueError("destination range must be non-negative")
        for span in self.spans:
            if span.size <= 0:
                raise ValueError("shard span size must be positive")
            if span.source_start < 0 or span.source_end > self.logical_size:
                raise ValueError("shard span exceeds the logical source")
            if (
                span.destination_start < self.destination_start
                or span.destination_end > self.destination_start + self.destination_size
            ):
                raise ValueError("shard span exceeds the padded destination")
        ordered = sorted(self.spans, key=lambda span: span.destination_start)
        if any(
            left.destination_end > right.destination_start
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError("shard destination spans overlap")

    @classmethod
    def contiguous_padded(
        cls,
        *,
        logical_size: int,
        padded_size: int,
        tp_size: int,
        tp_rank: int,
        destination_start: int = 0,
    ) -> "ShardLayout":
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError("invalid tensor-parallel rank geometry")
        if padded_size < logical_size or padded_size % tp_size:
            raise ValueError("padded size must cover the logical size and divide by TP")
        destination_size = padded_size // tp_size
        source_start = tp_rank * destination_size
        copy_size = max(0, min(destination_size, logical_size - source_start))
        spans = (
            (
                ShardSpan(
                    source_start=source_start,
                    destination_start=destination_start,
                    size=copy_size,
                ),
            )
            if copy_size
            else ()
        )
        return cls(
            logical_size=logical_size,
            destination_start=destination_start,
            destination_size=destination_size,
            spans=spans,
        )

    @classmethod
    def explicit(
        cls,
        *,
        logical_size: int,
        destination_start: int,
        destination_size: int,
        source_start: int,
        source_size: int,
    ) -> "ShardLayout":
        if (
            source_start < 0
            or source_size < 0
            or source_start + source_size > logical_size
        ):
            raise ValueError("explicit span exceeds the logical source")
        spans = (
            (
                ShardSpan(
                    source_start=source_start,
                    destination_start=destination_start,
                    size=source_size,
                ),
            )
            if source_size
            else ()
        )
        return cls(
            logical_size=logical_size,
            destination_start=destination_start,
            destination_size=destination_size,
            spans=spans,
        )

    @staticmethod
    def validate_exact_source_partition(layouts: tuple["ShardLayout", ...]) -> None:
        if not layouts:
            raise ValueError("source partition must contain at least one layout")
        logical_size = layouts[0].logical_size
        if any(layout.logical_size != logical_size for layout in layouts):
            raise ValueError("source partition logical sizes differ")
        spans = sorted(
            (span for layout in layouts for span in layout.spans),
            key=lambda span: span.source_start,
        )
        expected_start = 0
        for span in spans:
            if span.source_start != expected_start:
                raise ValueError("source partition has a gap or overlap")
            expected_start = span.source_end
        if expected_start != logical_size:
            raise ValueError("source partition does not cover the logical source")
