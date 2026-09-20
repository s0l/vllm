# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed wave slots: next H2D overlaps the current expert computation.

The provider owns the host dispatcher outside capture. Weight/metadata storage
is leased through the slot's final compute event. Graphs consume fixed addresses
and changing descriptor tensors; each slot has its own graph memory pool.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.forward_context import set_forward_context

from .expert_offload_bank import COMPONENTS, NativeMappedBankConsumer


@dataclass
class WaveSlot:
    group_rows: torch.Tensor
    lane_ids: torch.Tensor
    local_ids: torch.Tensor
    offset: torch.Tensor
    host_rows: torch.Tensor
    host_lanes: torch.Tensor
    host_ids: torch.Tensor
    pinned: dict
    pinned_numpy: dict
    error: torch.Tensor
    pool: tuple
    kernels: dict = field(default_factory=dict)
    compiled: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    done: torch.cuda.Event | None = None
    copied: torch.cuda.Event | None = None


class NativeWaveExecutor:
    """One provider invocation; no GPU operations from native host callbacks."""

    def __init__(self, provider, slots):
        bank = provider.bank
        if slots not in (1, 2) or bank.staging % slots:
            raise ValueError("wave slots must partition the admitted staging bank")
        cpu = getattr(bank.source, "cpu", None)
        if cpu is not None:
            required = bank.staging + int(getattr(cpu, "ready_enabled", False))
            if cpu.source_stats()["capacity"] < required:
                raise ValueError(
                    "source capacity cannot admit all wave leases and demand reserve"
                )
        self.provider, self.bank = provider, bank
        self.use_compiled = False
        self.trace_events = False
        self.last_timeline = []
        self.wave_rows = bank.staging // slots
        self.copy_stream = torch.cuda.Stream(device=bank.device)
        self.slots = []
        capacity = provider.max_tokens * provider.topk + provider.max_lanes
        for number in range(slots):
            start, end = number * self.wave_rows, (number + 1) * self.wave_rows
            self.slots.append(
                WaveSlot(
                    group_rows=torch.zeros(
                        self.wave_rows, dtype=torch.int32, device=bank.device
                    ),
                    lane_ids=torch.full(
                        (capacity,), -1, dtype=torch.int64, device=bank.device
                    ),
                    local_ids=torch.zeros(
                        (capacity, 1), dtype=torch.int32, device=bank.device
                    ),
                    offset=torch.zeros((), dtype=torch.int64, device=bank.device),
                    host_rows=torch.zeros(
                        self.wave_rows, dtype=torch.int32, device="cpu", pin_memory=True
                    ),
                    host_lanes=torch.empty(
                        capacity, dtype=torch.int64, device="cpu", pin_memory=True
                    ),
                    host_ids=torch.empty(
                        (capacity, 1), dtype=torch.int32, device="cpu", pin_memory=True
                    ),
                    pinned={
                        name: value[start:end] for name, value in bank.pinned.items()
                    },
                    pinned_numpy={
                        name: value[start:end]
                        for name, value in bank.pinned_numpy.items()
                    },
                    error=torch.zeros((), dtype=torch.int32, device=bank.device),
                    pool=torch.cuda.graph_pool_handle(),
                )
            )
        self.metadata_bytes = sum(
            tensor.numel() * tensor.element_size()
            for slot in self.slots
            for tensor in (
                slot.group_rows,
                slot.lane_ids,
                slot.local_ids,
                slot.offset,
                slot.error,
            )
        )
        self.host_metadata_bytes = sum(
            tensor.numel() * tensor.element_size()
            for slot in self.slots
            for tensor in (slot.host_rows, slot.host_lanes, slot.host_ids)
        )
        self.max_source_lease_bytes = (
            slots * self.wave_rows * sum(bank.strides.values())
        )

    def kernel(self, slot, bucket):
        if bucket in slot.kernels:
            direct_fn, graph = slot.kernels[bucket]
            if graph not in self.bank.graphs:
                raise RuntimeError("retired wave graph")
            return direct_fn, graph
        provider = self.provider
        consumer = NativeMappedBankConsumer(
            self.bank, slot.group_rows, num_groups=self.wave_rows
        )

        def forward():
            positions = torch.arange(bucket, device=self.bank.device) + slot.offset
            lanes = slot.lane_ids.index_select(0, positions)
            ids = slot.local_ids.index_select(0, positions)
            x = provider.x.index_select(0, lanes.clamp_min(0) // provider.topk)
            weights = provider.weights.flatten().index_select(0, lanes.clamp_min(0))
            y = consumer(x, (weights * (lanes >= 0)).unsqueeze(1), ids)
            torch.ops.vllm.flashnext_native_scatter(y, lanes, provider.lanes)

        with set_forward_context(None, provider.config, num_tokens=bucket):
            compiled = torch.compile(forward, fullgraph=True)
            for _ in range(3):
                compiled()
            self.bank.fence().synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=slot.pool):
                compiled()
            self.bank.register_graph(graph, stable_views=True)
            provider.graph_captures += 1

        def direct():
            with set_forward_context(None, provider.config, num_tokens=bucket):
                forward()

        slot.kernels[bucket] = direct, graph
        slot.compiled[bucket] = compiled
        return direct, graph

    def prepare_kernels(self, buckets):
        # Called before measured source/copy work, including runtime profile.
        # All slot data is initialized and no source read occurs during capture.
        buckets = sorted(set(buckets))
        if all(bucket in slot.kernels for slot in self.slots for bucket in buckets):
            return
        bank = self.bank
        lease = object()
        with bank.lock:
            if bank.state != "READY" or bank.leases:
                raise RuntimeError(
                    "wave capture requires exclusive placement authority"
                )
            bank.leases.add(lease)
        try:
            for slot in self.slots:
                if any(bucket not in slot.kernels for bucket in buckets):
                    self.retire_slot(slot)
                    slot.lane_ids.fill_(-1)
                    slot.local_ids.zero_()
                    slot.offset.zero_()
                    slot.group_rows.zero_()
                for bucket in buckets:
                    self.kernel(slot, bucket)
        finally:
            bank.release(lease, bank.fence())

    def retire_slot(self, slot):
        if slot.done is not None:
            slot.done.synchronize()
            slot.done = None
        if slot.copied is not None:
            slot.copied.synchronize()
            slot.copied = None
        slot.rows.clear()

    def drain(self):
        for slot in self.slots:
            self.retire_slot(slot)

    def run(self, layer, waves):
        bank, provider = self.bank, self.provider
        if bank.state != "READY" or bank.leases:
            raise RuntimeError("wave execution requires exclusive placement authority")
        self.prepare_kernels(
            tile.bucket for wave in waves for tile in wave.tiles(provider.max_lanes)
        )
        current = torch.cuda.current_stream(bank.device)
        # Source fields/row maps may be read, but resize/promotion publication
        # cannot change the admitted GPU prefix until the last reader drains.
        lease = object()
        bank.leases.add(lease)
        tiles = useful = 0
        timeline = []
        anchor = None
        try:
            if self.trace_events:
                anchor = torch.cuda.Event(enable_timing=True)
                anchor.record(current)
            for slot in self.slots:
                slot.error.zero_()
            for index, wave in enumerate(waves):
                slot = self.slots[index % len(self.slots)]
                self.retire_slot(slot)
                scratch = (
                    bank.tables.pool_rows + (index % len(self.slots)) * self.wave_rows
                )
                missing: list[int] = []
                destinations: list[int] = []
                physical: list[int] = []
                hot: list[bool] = []
                for expert in wave.experts:
                    row = bank.host_hot.get((layer, expert))
                    if row is None:
                        row = scratch + len(missing)
                        missing.append(expert)
                        destinations.append(row)
                        hot.append(False)
                    else:
                        if not 0 <= row < bank.tables.pool_rows:
                            raise ValueError(
                                "HOT proposal is outside admitted residency"
                            )
                        hot.append(True)
                    physical.append(row)
                if len(physical) > self.wave_rows:
                    raise ValueError("wave exceeds its fixed slot")
                slot.rows = bank.source.get_many(layer, missing) if missing else []
                prepared_tiles = tuple(wave.tiles(provider.max_lanes))
                extent = (
                    sum(len(tile.lanes) for tile in prepared_tiles[:-1])
                    + prepared_tiles[-1].bucket
                )
                slot.host_lanes[:extent].fill_(-1)
                slot.host_ids[:extent].zero_()
                slot.host_lanes[: len(wave.lanes)].copy_(
                    torch.from_numpy(wave.lanes.copy())
                )
                slot.host_ids[: len(wave.slots), 0].copy_(
                    torch.from_numpy(wave.slots.copy())
                )
                slot.host_rows.zero_()
                slot.host_rows[: len(physical)].copy_(
                    torch.tensor(physical, dtype=torch.int32)
                )
                region = getattr(bank, "registered_source_region", None)
                if missing and region is None:
                    bank._fill_pinned(slot.rows, range(len(missing)), slot.pinned_numpy)
                if index == 0:
                    self.copy_stream.wait_stream(current)
                with torch.cuda.stream(self.copy_stream):
                    stamp = None
                    if self.trace_events:
                        stamp = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                        stamp[0].record()
                    if missing:
                        if region is not None:
                            bank._copy_native_rows(slot.rows, destinations, region)
                        else:
                            for name in COMPONENTS:
                                bank.consumer_views[name][
                                    scratch : scratch + len(missing)
                                ].copy_(
                                    slot.pinned[name][: len(missing)], non_blocking=True
                                )
                                bank.copy_calls += 1
                            bank.copy_bytes += len(missing) * sum(bank.strides.values())
                    slot.group_rows.copy_(slot.host_rows, non_blocking=True)
                    slot.lane_ids[:extent].copy_(
                        slot.host_lanes[:extent], non_blocking=True
                    )
                    slot.local_ids[:extent].copy_(
                        slot.host_ids[:extent], non_blocking=True
                    )
                    slot.copied = bank.fence()
                    if stamp is not None:
                        stamp[1].record()
                current.wait_event(slot.copied)
                # Host HOT maps propose placement; GPU ownership is checked
                # before these physical rows can be consumed. Invalid hints
                # address safe row zero and fail the common completion vote.
                if any(hot):
                    positions = torch.tensor(np.flatnonzero(hot), device=bank.device)
                    keys = torch.tensor(
                        [
                            layer * bank.source.experts + expert
                            for expert, is_hot in zip(wave.experts, hot, strict=True)
                            if is_hot
                        ],
                        device=bank.device,
                    )
                    rows = slot.group_rows[positions].long()
                    valid = (bank.tables.hot_phys[keys].long() == rows) & (
                        bank.tables.row_key[rows].long() == keys
                    )
                    slot.error.copy_(torch.maximum(slot.error, (~valid).any().int()))
                    slot.group_rows[positions] = torch.where(valid, rows, 0).int()
                offset = 0
                if stamp is not None:
                    stamp[2].record(current)
                for tile in prepared_tiles:
                    slot.offset.fill_(offset)
                    direct, graph = self.kernel(slot, tile.bucket)
                    if provider.use_graphs:
                        graph.replay()
                    elif self.use_compiled:
                        with set_forward_context(
                            None, provider.config, num_tokens=tile.bucket
                        ):
                            slot.compiled[tile.bucket]()
                    else:
                        direct()
                    offset += len(tile.lanes)
                    tiles += 1
                    useful += len(tile.lanes)
                slot.done = bank.fence()
                if stamp is not None:
                    stamp[3].record(current)
                    timeline.append(stamp)
            self.drain()
            if anchor is not None:
                # Events are observers; normal execution has no timestamp
                # allocation, and neither source nor compute waits on them.
                current.synchronize()
                self.last_timeline = [
                    dict(
                        zip(
                            (
                                "copy_start_ms",
                                "copy_end_ms",
                                "compute_start_ms",
                                "compute_end_ms",
                            ),
                            (anchor.elapsed_time(event) for event in stamp),
                            strict=True,
                        )
                    )
                    for stamp in timeline
                ]
            if any(int(slot.error.item()) for slot in self.slots):
                raise RuntimeError("wave HOT ownership validation failed")
            return tiles, useful
        finally:
            # Includes partial DMA failure. No timeout grants permission to
            # recycle live rows; a poisoned device requires cohort recovery.
            self.copy_stream.synchronize()
            current.synchronize()
            self.drain()
            bank.leases.discard(lease)

    def close(self):
        self.drain()
        for slot in self.slots:
            for _, graph in slot.kernels.values():
                graph.reset()
                self.bank.graphs.discard(graph)
                self.bank.stable_graphs.discard(graph)
            slot.kernels.clear()
            slot.compiled.clear()
            slot.pool = torch.cuda.graph_pool_handle()
