# SPDX-License-Identifier: Apache-2.0
"""Deterministic no-migration owner policy for Exp11 DCP pages."""

from __future__ import annotations

from fractions import Fraction

POLICY_VERSION = "exp11-884-776-equal-v1"
PAGE_SIZE = 16
_KNOTS = (
    (0, (Fraction(2, 5), Fraction(2, 5), Fraction(1, 5))),
    (4096, (Fraction(2, 5), Fraction(2, 5), Fraction(1, 5))),
    (8192, (Fraction(7, 20), Fraction(7, 20), Fraction(3, 10))),
    (16384, (Fraction(7, 20), Fraction(7, 20), Fraction(3, 10))),
    (32768, (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))),
)


class ElasticPageOwnerPolicy:
    """Lazily materialize one request-independent owner sequence."""

    def __init__(self, *, page_size: int = PAGE_SIZE) -> None:
        if page_size <= 0:
            raise ValueError("Exp11 owner policy requires a positive page size")
        self.page_size = page_size
        self._owners: list[int] = []
        self._counts = [0, 0, 0]

    @staticmethod
    def _shares(tokens: int) -> tuple[Fraction, Fraction, Fraction]:
        if tokens < 0:
            raise ValueError("token position must be non-negative")
        if tokens >= _KNOTS[-1][0]:
            return _KNOTS[-1][1]
        for (left_tokens, left), (right_tokens, right) in zip(
            _KNOTS, _KNOTS[1:], strict=True
        ):
            if left_tokens <= tokens <= right_tokens:
                fraction = Fraction(
                    tokens - left_tokens, right_tokens - left_tokens
                )
                return tuple(
                    left[rank] + fraction * (right[rank] - left[rank])
                    for rank in range(3)
                )  # type: ignore[return-value]
        raise AssertionError("owner policy knots do not cover token position")

    def _extend(self, pages: int) -> None:
        while len(self._owners) < pages:
            page = len(self._owners)
            shares = self._shares((page + 1) * self.page_size)
            desired = [shares[rank] * (page + 1) for rank in range(3)]
            deficits = [desired[rank] - self._counts[rank] for rank in range(3)]
            owner = max(
                range(3),
                key=lambda rank: (deficits[rank], -((rank - page) % 3)),
            )
            self._owners.append(owner)
            self._counts[owner] += 1

    def owners(self, start_page: int, num_pages: int) -> tuple[int, ...]:
        if start_page < 0 or num_pages < 0:
            raise ValueError("page range must be non-negative")
        self._extend(start_page + num_pages)
        return tuple(self._owners[start_page : start_page + num_pages])

    def counts(self, pages: int) -> tuple[int, int, int]:
        if pages < 0:
            raise ValueError("page count must be non-negative")
        self._extend(pages)
        prefix = self._owners[:pages]
        return tuple(prefix.count(rank) for rank in range(3))  # type: ignore[return-value]

    def local_length(self, tokens: int, rank: int) -> int:
        if tokens < 0 or not 0 <= rank < 3:
            raise ValueError("invalid token length or DCP rank")
        full_pages, remainder = divmod(tokens, self.page_size)
        self._extend(full_pages + (remainder > 0))
        local = self._owners[:full_pages].count(rank) * self.page_size
        if remainder and self._owners[full_pages] == rank:
            local += remainder
        return local

    def build_luts(
        self, max_pages: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """Return owner, owner-local ordinal, and prefix-count LUTs."""
        if max_pages <= 0:
            raise ValueError("owner LUT requires at least one page")
        self._extend(max_pages)
        local_counts = [0, 0, 0]
        ordinals = []
        prefix_counts = [[0] for _ in range(3)]
        for owner in self._owners[:max_pages]:
            ordinals.append(local_counts[owner])
            local_counts[owner] += 1
            for rank in range(3):
                prefix_counts[rank].append(local_counts[rank])
        return (
            tuple(self._owners[:max_pages]),
            tuple(ordinals),
            tuple(tuple(values) for values in prefix_counts),
        )
