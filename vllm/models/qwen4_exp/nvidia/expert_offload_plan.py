# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conserve token/top-k lanes while grouping bounded expert demand."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Wave:
    experts: tuple[int, ...]
    lanes: np.ndarray
    slots: np.ndarray
    bucket: int

    def tiles(self, max_lanes):
        if max_lanes <= 0 or max_lanes & (max_lanes - 1):
            raise ValueError("active-lane capacity must be a positive power of two")
        return tuple(
            Wave(
                self.experts,
                self.lanes[start : start + max_lanes],
                self.slots[start : start + max_lanes],
                1 << (min(max_lanes, len(self.lanes) - start) - 1).bit_length(),
            )
            for start in range(0, len(self.lanes), max_lanes)
        )


def plan(ids, num_experts, capacity, *, is_padding=None, resident_experts=None):
    """Preserve each token/top-k lane once; only expert loads are deduplicated."""
    if (
        not isinstance(ids, np.ndarray)
        or ids.ndim != 2
        or ids.dtype not in (np.int32, np.int64)
        or not 1 <= capacity <= num_experts
        or not 1 <= ids.shape[1] <= num_experts
    ):
        raise ValueError("invalid compact route geometry")
    active_ids = ids
    if is_padding is not None:
        if (
            not isinstance(is_padding, np.ndarray)
            or is_padding.dtype != np.bool_
            or is_padding.shape != (ids.shape[0],)
        ):
            raise ValueError("invalid native padding mask")
        if np.any(ids[is_padding] != -1):
            raise ValueError("padding row has a live expert")
        active_ids = ids[~is_padding]
    if active_ids.size and (active_ids.min() < 0 or active_ids.max() >= num_experts):
        raise ValueError("expert outside logical layer")
    ordered = np.sort(active_ids, axis=1)
    if np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("duplicate expert in one token's top-k")
    flat = ids.reshape(-1)
    order = np.argsort(flat, kind="stable")
    if is_padding is not None:
        order = order[flat[order] >= 0]
    demand, starts, counts = np.unique(
        flat[order], return_index=True, return_counts=True
    )
    if resident_experts is not None:
        resident = np.asarray(resident_experts)
        if (
            resident.ndim != 1
            or resident.dtype.kind not in "iu"
            or np.any(resident < 0)
            or np.any(resident >= num_experts)
        ):
            raise ValueError("invalid resident expert identities")
        permutation = np.argsort(~np.isin(demand, resident), kind="stable")
        if len(permutation):
            order = np.concatenate(
                [order[starts[i] : starts[i] + counts[i]] for i in permutation]
            )
            demand, counts = demand[permutation], counts[permutation]
            starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    waves = []
    for first in range(0, len(demand), capacity):
        last = min(first + capacity, len(demand))
        lanes = order[starts[first] : starts[last - 1] + counts[last - 1]].astype(
            np.int64, copy=True
        )
        slots = np.repeat(np.arange(last - first, dtype=np.int32), counts[first:last])
        lanes.setflags(write=False)
        slots.setflags(write=False)
        waves.append(
            Wave(
                tuple(int(e) for e in demand[first:last]),
                lanes,
                slots,
                1 << (len(lanes) - 1).bit_length(),
            )
        )
    return waves
