# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded route statistics and past-only admission for the shared expert bank.

These plans propose ownership; bank leases, copies and rank-common publication
remain the authority. Scores never authorize reading an unpublished row.
"""

from collections import deque
from dataclasses import dataclass
from hashlib import sha256

import numpy as np


def route_histogram(ids, weights, num_experts):
    """Count useful selections, including large prefills but excluding padding."""
    if (
        type(num_experts) is not int
        or num_experts <= 0
        or not isinstance(ids, np.ndarray)
        or not isinstance(weights, np.ndarray)
        or ids.ndim != 2
        or weights.shape != ids.shape
        or ids.dtype.kind not in "iu"
        or weights.dtype.kind != "f"
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("invalid expert histogram inputs")
    active = weights > 0
    if np.any((ids >= num_experts) | (ids < -1) | ((ids == -1) & active)):
        raise ValueError("invalid expert histogram identity")
    chosen = ids[active].astype(np.int64, copy=False)
    counts = np.bincount(chosen, minlength=num_experts)
    if counts.max(initial=0) > np.iinfo(np.uint32).max:
        raise ValueError("expert histogram exceeds bounded counters")
    return counts.astype(np.uint32), int(active.any(axis=1).sum())


def cache_snapshot(bank):
    """Snapshot all metadata owners without retaining any expert tensor."""
    source = bank.source
    with bank.lock, source.lock:
        hot = sorted(
            (layer, expert, row) for (layer, expert), row in bank.host_hot.items()
        )
        ram = sorted(source.cache)
        hot_keys = {(layer, expert) for layer, expert, _ in hot}
        ram_keys = set(ram)
        overlap = hot_keys & ram_keys
        pinned = source.pinned_pool
        controller = getattr(bank, "elastic_controller", None)
        return dict(
            schema="native-expert-residency-v1",
            generation=bank.generation,
            state=bank.state,
            hot_capacity=bank.tables.pool_rows,
            hot_keys=hot,
            hot_map_role="published_host_hint; device ownership checked at consumption",
            ram_keys=ram,
            hot_resident_keys=len(hot),
            hot_unfilled_rows=bank.tables.pool_rows - len(hot),
            elastic_physical_memory=None
            if controller is None
            else controller.physical_memory_receipt(),
            ram_resident_keys=len(ram),
            gpu_ram_duplicate_keys=len(overlap),
            gpu_ram_duplicate_payload_bytes=len(overlap) * sum(bank.strides.values()),
            ram_retained_bytes=source.used,
            ram_budget_bytes=source.limit,
            ram_registered_bytes=(0 if pinned is None else pinned.allocated)
            + getattr(bank, "promotion_registered_source_bytes", 0),
            ram_registered_active_rows=0
            if pinned is None
            else sum(slab.active for slab in pinned.slabs),
            staging_rows=bank.staging,
            row_payload_bytes=sum(bank.strides.values()),
            bank_mapped_bytes=sum(
                owner.info.committed for owner in bank.backings.values()
            ),
            active_bank_leases=len(bank.leases),
            pending_hot_rows=sorted(getattr(bank, "pending_promotions", ())),
            promotion_staging_bytes=getattr(bank, "promotion_staging_bytes", 0),
            source_hits=source.hits,
            source_misses=source.misses,
            source_read_bytes=source.read_bytes,
        )


def stream_trace_layer(
    layer, ids, weights, cold_ids, *, cpu, state, hot_rows, num_experts
):
    """Describe consumed lanes without rereading or borrowing expert weights."""
    counts, tokens = route_histogram(ids, weights, num_experts)
    if cold_ids.shape != ids.shape or np.any((cold_ids != -1) & (cold_ids != ids)):
        raise ValueError("stream trace cold identity differs from consumed routes")
    useful = weights > 0
    cold = useful & (cold_ids >= 0)
    return dict(
        layer=layer,
        tokens=len(ids),
        execution="stream",
        dummy=False,
        state=state,
        hot_rows=hot_rows,
        useful_lanes=int(useful.sum()),
        route=dict(
            ids=ids.tolist(),
            weights=weights.tolist(),
            selected_histogram=counts.tolist(),
            useful_tokens=tokens,
            selected_experts=np.flatnonzero(counts).tolist(),
            gpu_hint_experts=np.unique(ids[useful & ~cold]).tolist(),
            cpu_experts=np.unique(cold_ids[cold]).tolist(),
        ),
        hot_lanes=int((useful & ~cold).sum()),
        cold_lanes=int(cold.sum()),
        cpu=cpu,
    )


@dataclass(frozen=True)
class ExpertAdmissionPlan:
    observation: int
    capacity: int
    resident: tuple[int, ...]
    desired: tuple[int, ...]
    promotions: tuple[int, ...]
    evictions: tuple[int, ...]
    protected: tuple[int, ...]

    def __post_init__(self):
        current, desired = set(self.resident), set(self.desired)
        if (
            type(self.observation) is not int
            or self.observation < 0
            or type(self.capacity) is not int
            or self.capacity < 0
            or len(desired) > self.capacity
            or any(
                len(set(x)) != len(x)
                for x in (
                    self.resident,
                    self.desired,
                    self.promotions,
                    self.evictions,
                    self.protected,
                )
            )
            or set(self.promotions) != desired - current
            or set(self.evictions) != current - desired
            or not set(self.protected) <= current & desired
        ):
            raise ValueError("inconsistent expert admission plan")

    @property
    def digest(self):
        identity = (
            self.observation,
            self.capacity,
            self.resident,
            self.desired,
            self.promotions,
            self.evictions,
            self.protected,
        )
        return sha256(repr(identity).encode()).hexdigest()


class ExpertFrequencyAdmission:
    """Exact bounded history with admission after an observed model step.

    A step is the complete layer set, not one staging wave. This avoids making
    recency depend on the layer's position in the model. The caller supplies
    only available source keys; history alone cannot fetch unseen weights.
    """

    def __init__(self, layers, experts, *, history_steps):
        if any(type(v) is not int or v <= 0 for v in (layers, experts, history_steps)):
            raise ValueError("invalid expert history geometry")
        self.layers, self.experts = layers, experts
        self.history_steps = history_steps
        self.history: deque[np.ndarray] = deque()
        self.frequency = np.zeros((layers, experts), dtype=np.float64)
        self.observation = 0

    def observe(self, counts, *, useful_tokens=1):
        if (
            not isinstance(counts, np.ndarray)
            or counts.shape != self.frequency.shape
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
            or np.any(counts > np.iinfo(np.uint32).max)
            or type(useful_tokens) is not int
            or useful_tokens <= 0
        ):
            raise ValueError("invalid complete expert observation")
        # Each model step contributes its per-token distribution. A large
        # prefill must not outweigh the next several decode steps merely by M.
        owned = counts.astype(np.float64, copy=True) / useful_tokens
        if len(self.history) == self.history_steps:
            self.history.popleft()
        self.history.append(owned)
        # Bounded recomputation avoids negative roundoff from long-running
        # add/subtract drift when a key ages out of normalized observations.
        self.frequency.fill(0)
        for observation in self.history:
            self.frequency += observation
        self.observation += 1

    def _keys(self, values):
        keys = tuple(values)
        if any(
            type(k) is not int or not 0 <= k < self.layers * self.experts for k in keys
        ) or len(set(keys)) != len(keys):
            raise ValueError("invalid or duplicate expert key")
        return set(keys)

    def plan(
        self,
        resident,
        available,
        *,
        capacity,
        max_promotions,
        protected=(),
        fill_free=False,
    ):
        arrays = [
            np.fromiter(self._keys(values), dtype=np.int64)
            for values in (resident, available, protected)
        ]
        return self.plan_arrays(
            *arrays[:2],
            protected=arrays[2],
            capacity=capacity,
            max_promotions=max_promotions,
            fill_free=fill_free,
        )

    def plan_arrays(
        self,
        resident,
        available,
        *,
        capacity,
        max_promotions,
        protected=None,
        fill_free=False,
    ):
        if (
            type(capacity) is not int
            or not 0 <= capacity <= self.layers * self.experts
            or type(max_promotions) is not int
            or max_promotions < 0
            or type(fill_free) is not bool
        ):
            raise ValueError("invalid expert admission budget")
        scores = self.frequency.reshape(-1)

        def mask(values):
            if (
                not isinstance(values, np.ndarray)
                or values.ndim != 1
                or values.dtype != np.int64
                or np.any(values < 0)
                or np.any(values >= scores.size)
            ):
                raise ValueError("invalid typed expert keys")
            result = np.zeros(scores.size, dtype=np.bool_)
            result[values] = True
            if int(result.sum()) != len(values):
                raise ValueError("duplicate expert keys")
            return result

        current, available = mask(resident), mask(available)
        protected = mask(np.empty(0, np.int64) if protected is None else protected)
        protected_count = int(protected.sum())
        if np.any(protected & ~current) or protected_count > capacity:
            raise ValueError("expert shrink would evict an active lease")

        def ranked(selected, *, weakest=False):
            keys = np.flatnonzero(selected)
            order = (
                np.lexsort((-keys, scores[keys]))
                if weakest
                else np.lexsort((keys, -scores[keys]))
            )
            return keys[order]

        desired = protected.copy()
        retained = ranked(current & ~protected)
        desired[retained[: capacity - protected_count]] = True
        size = int(desired.sum())
        limit = max_promotions + (capacity - size if fill_free else 0)
        candidates = ranked(available & ~current & (scores > 0))
        victims = ranked(desired & ~protected, weakest=True)
        victim_index = 0
        promoted: list[int] = []
        for candidate in candidates:
            if len(promoted) == limit:
                break
            if size == capacity:
                if (
                    victim_index == len(victims)
                    or scores[candidate] <= scores[victims[victim_index]]
                ):
                    break
                desired[victims[victim_index]] = False
                victim_index += 1
            else:
                size += 1
            desired[candidate] = True
            promoted.append(int(candidate))
        return ExpertAdmissionPlan(
            self.observation,
            capacity,
            tuple(np.flatnonzero(current).tolist()),
            tuple(np.flatnonzero(desired).tolist()),
            tuple(promoted),
            tuple(np.flatnonzero(current & ~desired).tolist()),
            tuple(np.flatnonzero(protected).tolist()),
        )

    def validate(self, proposal, resident):
        if (
            not isinstance(proposal, ExpertAdmissionPlan)
            or proposal.observation != self.observation
            or set(proposal.resident) != self._keys(resident)
        ):
            raise ValueError("stale expert admission plan")
        self._keys(proposal.desired)
        return proposal


class NativeExpertAdmission:
    """Past-only placement at a completed target-model boundary.

    The shared bank keeps its existing copy/read leases and rank-common
    publication. This owner only proposes rows; it never authorizes GPU reads.
    Explicit opt-in is required until the complete execution gate is accepted.
    """

    def __init__(
        self,
        provider,
        *,
        history_steps,
        max_promotions,
        exclusive_ram,
        asynchronous=True,
        registered_source=False,
        host_control=False,
    ):
        import torch

        bank = provider.bank
        if type(max_promotions) is not int or max_promotions < 0:
            raise ValueError("invalid model-step promotion budget")
        if type(exclusive_ram) is not bool:
            raise ValueError("invalid RAM ownership policy")
        if registered_source and not asynchronous:
            raise ValueError("registered source requires asynchronous promotion")
        if type(host_control) is not bool or (host_control and not asynchronous):
            raise ValueError("host control requires asynchronous promotion")
        provider.coordinator.host_control = host_control
        self.provider, self.bank = provider, bank
        self.policy = ExpertFrequencyAdmission(
            bank.source.layers, bank.source.experts, history_steps=history_steps
        )
        self.max_promotions, self.exclusive_ram = max_promotions, exclusive_ram
        self.counts = np.zeros(self.policy.frequency.shape, np.uint32)
        self.next_layer, self.tokens = 0, None
        self.last_step = {}
        self.promotion = None
        if asynchronous:
            from .expert_offload_promotion import NativeExpertPromotion

            self.promotion = NativeExpertPromotion(
                provider, registered_source=registered_source
            )
            bank.promotion_staging_bytes = self.promotion.staging_bytes
        packed = (self.counts.size + 7) // 8
        self.availability = torch.empty(
            packed + 33,
            dtype=torch.uint8,
            device="cpu" if host_control else bank.device,
        )
        self.peer_availability = torch.empty(
            (packed + 33) * provider.coordinator.ranks,
            dtype=torch.uint8,
            device=self.availability.device,
        )

    def observe_layer(self, layer, ids, weights):
        if layer == 0:
            # Cancellation discards an incomplete observation. It does not
            # commit a partial layer set or age the previous complete history.
            self.counts.fill(0)
            self.next_layer, self.tokens = 0, None
        if layer != self.next_layer:
            raise ValueError("incomplete or reordered expert model observation")
        counts, tokens = route_histogram(ids, weights, self.bank.source.experts)
        if self.tokens is not None and tokens != self.tokens:
            raise ValueError("expert layers disagree on useful token count")
        self.counts[layer] = counts
        self.tokens = tokens
        self.next_layer += 1

    def observe_prefill_layer(self, layer, ids, weights):
        """Retain each request's recent routes while its source rows are present."""
        identity = getattr(self.provider, "execution_identity", None)
        stream = self.provider.stream_path
        horizon = self.policy.history_steps * (stream.max_m if stream else 1)
        selected = None
        if identity is not None:
            offsets = identity["query_start_loc"]
            if (
                len(offsets) != identity["num_reqs"] + 1
                or not offsets
                or offsets[0] != 0
                or offsets[-1] != identity["tokens"]
                or offsets[-1] > len(ids)
                or any(type(v) is not int for v in offsets)
                or any(a > b for a, b in zip(offsets, offsets[1:]))
            ):
                raise ValueError("invalid request ranges for expert retention")
            selected = (
                np.concatenate(
                    [
                        np.arange(max(start, end - horizon), end)
                        for start, end in zip(offsets, offsets[1:])
                    ]
                )
                if identity["num_reqs"]
                else np.array([], dtype=np.int64)
            )
        # Standalone callers without request boundaries retain all supplied
        # rows; guessing a concatenated batch's tail would starve earlier requests.
        self.observe_layer(
            layer,
            ids if selected is None else ids[selected],
            weights if selected is None else weights[selected],
        )
        if layer == 0:
            self.retention_scores = self.policy.frequency.copy()
        if self.tokens:
            self.retention_scores[layer] += self.counts[layer] / self.tokens
        self.bank.source.set_residency_scores(
            self.retention_scores,
            (),
            gpu_hot_keys=self.bank.host_hot,
        )

    def finish_step(self):
        import time

        import torch

        bank, source = self.bank, self.bank.source
        if self.next_layer != source.layers or bank.state != "READY" or bank.leases:
            raise RuntimeError("expert admission requires a completed unleased model")
        started = time.perf_counter()
        phases = {}
        phase_start = started

        def mark(name):
            nonlocal phase_start
            now = time.perf_counter()
            phases[name] = (now - phase_start) * 1000
            phase_start = now

        self.next_layer = 0
        if not self.tokens:
            self.last_step = dict(promotions=0, reason="padding-only model step")
            return
        self.policy.observe(self.counts, useful_tokens=self.tokens)
        mark("history_ms")
        was_pending = self.promotion is not None and self.promotion.pending is not None
        pending = self.promotion is not None and not self.promotion.poll()
        completed_copy = (
            dict(self.promotion.last_step)
            if self.promotion is not None and was_pending and not pending
            else None
        )
        mark("publish_ms")
        # RAM scores describe residual demand after VRAM ownership. Retiring a
        # cache reference cannot recycle a still-leased pinned source row.
        error = None
        available = np.zeros(self.counts.size, np.uint8)
        try:
            source.set_residency_scores(
                self.policy.frequency,
                bank.host_hot if self.exclusive_ram else (),
                gpu_hot_keys=bank.host_hot,
            )
            with source.lock:
                if hasattr(source, "resident_bitmap"):
                    available = source.resident_bitmap()
                    if (
                        available.dtype != np.uint8
                        or available.shape != (self.counts.size,)
                        or np.any(available > 1)
                    ):
                        raise ValueError("invalid source residency bitmap")
                else:
                    for layer, expert in source.cache:
                        available[layer * source.experts + expert] = 1
        except Exception as exc:
            error = exc
        mark("source_scores_and_keys_ms")
        identity = sha256(
            self.policy.frequency.tobytes()
            + repr((self.policy.observation, bank.generation, pending)).encode()
        ).digest()
        packet = np.concatenate(
            (
                np.array([int(error is None)], np.uint8),
                np.frombuffer(identity, np.uint8),
                np.packbits(available),
            )
        )
        self.availability.copy_(torch.from_numpy(packet))
        coordinator = self.provider.coordinator
        torch.distributed.all_gather_single(
            self.peer_availability,
            self.availability,
            group=coordinator.group.cpu_group
            if coordinator.host_control
            else coordinator.group.device_group,
        )
        peers = self.peer_availability.cpu().numpy().reshape(coordinator.ranks, -1)
        mark("availability_consensus_ms")
        if (
            not (peers[:, 0] == 1).all()
            or not (peers[:, 1:33] == peers[:1, 1:33]).all()
        ):
            bank.state = "POISONED"
            if self.promotion is not None:
                self.promotion.cancel()
            raise RuntimeError("rank-inconsistent expert observation/source") from error
        if pending:
            self.last_step = dict(
                promotions=0,
                pending=True,
                completed_copy=completed_copy,
                phases_ms=phases,
                wall_ms=(time.perf_counter() - started) * 1000,
            )
            return
        common = np.unpackbits(np.bitwise_and.reduce(peers[:, 33:], axis=0))[
            : self.counts.size
        ]
        resident = np.fromiter(
            (layer * source.experts + expert for layer, expert in bank.host_hot),
            dtype=np.int64,
        )
        proposal = self.policy.plan_arrays(
            resident,
            np.flatnonzero(common),
            capacity=bank.tables.pool_rows,
            max_promotions=self.max_promotions,
            # Zero disables admission, including bootstrap, for frozen controls.
            fill_free=self.max_promotions > 0,
        )
        # The bank validates actual current ownership again before retirement.
        mark("placement_plan_ms")
        before_copy, before_read = bank.copy_bytes, source.read_bytes
        if self.promotion is None:
            coordinator.apply_admission(proposal)
        else:
            self.promotion.start(proposal)
        mark("upload_reserve_launch_ms")
        released = source.set_residency_scores(
            self.policy.frequency,
            bank.host_hot if self.exclusive_ram else (),
            gpu_hot_keys=bank.host_hot,
        )
        mark("source_after_retire_ms")
        self.last_step = dict(
            observation=self.policy.observation,
            phases_ms=phases,
            promotions=len(proposal.promotions),
            completed_copy=completed_copy,
            evictions=len(proposal.evictions),
            plan_digest=proposal.digest,
            available_keys=int(common.sum()),
            hot_keys=len(bank.host_hot),
            h2d_bytes=bank.copy_bytes - before_copy,
            source_read_bytes=source.read_bytes - before_read,
            ram_cache_references_released_bytes=released,
            wall_ms=(time.perf_counter() - started) * 1000,
            async_copy=None
            if self.promotion is None
            else dict(self.promotion.last_step),
        )

    def quiesce(self):
        if self.promotion is not None:
            if self.bank.state == "POISONED":
                self.promotion.cancel()
            else:
                self.promotion.poll(wait=True)

    def close(self):
        if self.promotion is not None:
            self.promotion.close()
