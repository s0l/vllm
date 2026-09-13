# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned, acknowledged residency for exact recurrent checkpoints."""

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointPlan:
    sequence: int
    saves: tuple[bytes, ...]
    evictions: tuple[bytes, ...]
    resident: tuple[bytes, ...]


class GDNCheckpointPolicy:
    """GreedyDual aging with prefix tokens as recomputation-cost proxy.

    Each entry has the same model-specific byte charge. Only acknowledged
    entries are lookup candidates. Commands execute FIFO on each worker:
    restore first, then evict, then save. Removing a key here prevents future
    restores; earlier queued restores execute before its eviction command.
    """

    def __init__(self, budget_bytes: int, entry_bytes: int, *, cost_aware=True):
        if entry_bytes <= 0 or budget_bytes < entry_bytes:
            raise ValueError("GDN checkpoint budget must fit one positive entry")
        self.capacity = budget_bytes // entry_bytes
        self.entry_bytes = entry_bytes
        self.budget_bytes = budget_bytes
        self.cost_aware = cost_aware
        self.entries: OrderedDict[bytes, tuple[int, int]] = OrderedDict()
        self.ready: set[bytes] = set()
        self.inflation = 0
        self.sequence = 0
        self.acknowledged = 0

    def touch(self, key: bytes) -> None:
        if key in self.entries:
            _, cost = self.entries[key]
            self.entries[key] = (self.inflation + cost, cost)
            self.entries.move_to_end(key)

    def plan(
        self, candidates: dict[bytes, int], *, protected: set[bytes] | None = None
    ) -> CheckpointPlan:
        if any(
            not isinstance(k, bytes) or not k or v <= 0 for k, v in candidates.items()
        ):
            raise ValueError("Invalid GDN checkpoint candidate")
        before = set(self.entries)
        protected = protected or set()
        if not protected <= before:
            raise RuntimeError("Protected GDN checkpoint is not resident")
        # A single wave may offer more checkpoints than the entire budget.
        # Keep the most valuable offers, without admitting then evicting a
        # checkpoint whose save has not yet executed in this wave.
        offered = sorted(
            (k for k in candidates if k not in protected),
            key=lambda key: candidates[key],
            reverse=True,
        )[: self.capacity - len(protected)]
        selected = set(offered) | protected
        for key in offered:
            cost = candidates[key] if self.cost_aware else 1
            if key not in self.entries:
                while len(self.entries) >= self.capacity:
                    victims = [k for k in self.entries if k not in selected]
                    victim = min(victims, key=lambda k: self.entries[k][0])
                    self.inflation = self.entries.pop(victim)[0]
                    self.ready.discard(victim)
                self.entries[key] = (self.inflation + cost, cost)
            else:
                self.touch(key)
        self.sequence += 1
        return CheckpointPlan(
            self.sequence,
            tuple(k for k in self.entries if k not in before),
            tuple(k for k in before if k not in self.entries),
            tuple(self.entries),
        )

    def acknowledge(self, plan: CheckpointPlan, keys: tuple[bytes, ...]) -> None:
        if plan.sequence != self.acknowledged + 1:
            raise RuntimeError("Out-of-order GDN checkpoint acknowledgement")
        if len(keys) != len(set(keys)) or set(keys) != set(plan.resident):
            raise RuntimeError("Worker GDN checkpoint membership differs from plan")
        self.acknowledged = plan.sequence
        # A newer scheduled wave may already have retired an older snapshot's
        # keys. Its acknowledgement must never resurrect them.
        self.ready = set(keys).intersection(self.entries)

    def visible(self) -> tuple[bytes, ...]:
        return tuple(k for k in self.entries if k in self.ready)
