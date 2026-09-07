"""Physical row ownership and wire layout for experimental TP3 continuation.

This plan does not select a route or infer speed from payload bytes.
Bytes count directed off-rank logical payload, not measured PCIe transactions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RowOwnerPlan:
    rows: tuple[int, int, int]
    hidden: int
    augmented_k: tuple[int, int, int]

    def __post_init__(self):
        if len(self.rows) != 3 or any(type(n) is not int or n <= 0 for n in self.rows):
            raise ValueError("three positive owner row counts are required")
        if type(self.hidden) is not int or self.hidden <= 0 or self.hidden % 64:
            raise ValueError("hidden width must be positive and 64-aligned")
        if len(self.augmented_k) != 3 or any(
            type(k) is not int or k < self.hidden or k % 64 for k in self.augmented_k
        ):
            raise ValueError("three aligned consumer widths must cover hidden")

    @property
    def total_rows(self):
        return sum(self.rows)

    @property
    def offsets(self):
        return (0, self.rows[0], self.rows[0] + self.rows[1])

    def gather_splits(self, rank):
        """BF16 element counts, indexed by destination then source rank."""
        self._rank(rank)
        send = tuple(n * self.hidden for n in self.rows)
        return send, (send[rank],) * 3

    def publish_splits(self, rank, *, packed):
        """Byte counts, including local copies and destination-specific ARC."""
        self._rank(rank)
        widths = self.payload_widths(packed=packed)
        return (
            tuple(self.rows[rank] * width for width in widths),
            tuple(n * widths[rank] for n in self.rows),
        )

    def payload_widths(self, *, packed):
        return (
            tuple(k // 2 + k // 16 for k in self.augmented_k)
            if packed
            else (2 * self.hidden,) * 3
        )

    def wire_splits(self, rank, *, alignment=16):
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("wire alignment must be a positive power of two")
        send, receive = self.publish_splits(rank, packed=True)
        return tuple(
            tuple((n + alignment - 1) // alignment * alignment for n in values)
            for values in (send, receive)
        )

    def ledger(self, *, packed):
        edges = []
        widths = self.payload_widths(packed=packed)
        for source in range(3):
            for destination in range(3):
                if source != destination:
                    edges.append(
                        {
                            "source": source,
                            "destination": destination,
                            "gather_bytes": 2 * self.hidden * self.rows[destination],
                            "publish_bytes": self.rows[source] * widths[destination],
                        }
                    )
        total = sum(e["gather_bytes"] + e["publish_bytes"] for e in edges)
        control = 4 * self.total_rows * self.hidden * 2
        return {
            "rows": self.rows,
            "offsets": self.offsets,
            "augmented_k": self.augmented_k,
            "packed": packed,
            "edges": edges,
            "off_rank_bytes": total,
            "column_owner_control_bytes": control,
            "payload_reduction_fraction": 1 - total / control,
            "collective_phases": 2,
            "statistics_collectives": 0,
            "monolithic_gemm_calls_per_rank": 1,
            "quant_calls_per_rank_initial_implementation": 3 if packed else 1,
            "resident_memory_delta": "UNKNOWN until liveness proof",
            "latency_delta": "UNKNOWN; bytes are not time",
        }

    @staticmethod
    def _rank(rank):
        if type(rank) is not int or rank not in (0, 1, 2):
            raise ValueError("rank must be 0, 1 or 2")


def partition_rows(total, weights=(9, 9, 2)):
    """Largest-remainder allocation; every admitted owner has real rows.

    Small M is deliberately outside this candidate, not silently padded.
    """
    if type(total) is not int or total < 3:
        raise ValueError("row-owner candidate requires at least three rows")
    if len(weights) != 3 or any(type(w) is not int or w <= 0 for w in weights):
        raise ValueError("three positive weights required")
    denominator = sum(weights)
    remaining = total - 3
    counts = [1 + remaining * w // denominator for w in weights]
    order = sorted(range(3), key=lambda r: (-(remaining * weights[r] % denominator), r))
    for rank in order[: total - sum(counts)]:
        counts[rank] += 1
    return tuple(counts)
