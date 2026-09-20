# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete native expert rows in a global, leased CUDA VMM bank.

Placement is provisional until every copied component and every rank publish.
The caller owns common-plan consensus, elastic budget admission and graph
retirement; no method here can infer that a scheduler transition was admitted.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
import torch

from . import expert_offload_tables as gp

ARRAYS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
SCALARS = (
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
    "a1_gscale",
    "a2_gscale",
)
COMPONENTS = ARRAYS + SCALARS


@dataclass(frozen=True)
class NativeBankPlan:
    generation: int
    layer: int
    routes: tuple[int, ...]
    copies: tuple[tuple[int, int], ...]


class NativeExpertBank:
    """One rank's physical bank; all readers must hold an explicit lease."""

    def __init__(
        self, source, device, *, hot_rows, max_hot_rows, staging=32, quantum=2 << 20
    ):
        from vllm.device_allocator.elastic_cumem import allocate_elastic_backing

        if not 0 <= hot_rows <= max_hot_rows <= source.layers * source.experts:
            raise ValueError("invalid native HOT capacity")
        if staging < 1 or staging & (staging - 1):
            raise ValueError("power-of-two native staging capacity required")
        self.source, self.device = source, device
        self.staging, self.max_rows = staging, max_hot_rows + staging
        self.rows, self.quantum = hot_rows + staging, quantum
        sample = source.get(0, 0)
        if set(sample) != set(COMPONENTS):
            raise ValueError("incomplete native schema")
        self.shapes = {name: value.shape for name, value in sample.items()}
        self.strides = {name: value.nbytes for name, value in sample.items()}
        self.dtypes = {
            name: (
                torch.float32
                if name in SCALARS
                else torch.float8_e4m3fn
                if name.endswith("weight_scale")
                else torch.uint8
            )
            for name in COMPONENTS
        }
        self.backings = {
            name: allocate_elastic_backing(
                self._round(self.max_rows * self.strides[name]),
                self._round(self.rows * self.strides[name]),
                quantum,
                device,
            )
            for name in ARRAYS
        }
        scalar_bytes = self._round(self.max_rows * len(SCALARS) * 4)
        self.scalar_bytes = scalar_bytes
        self.backings["globals"] = allocate_elastic_backing(
            scalar_bytes, scalar_bytes, quantum, device
        )
        self.pinned, self.pinned_numpy = self._allocate_pinned()
        self.next_pinned = self.next_pinned_numpy = None
        self.prepared_native = {}
        self.prefetch_region = None
        self.tables = gp.allocate_global_tables(
            device, source.experts, [0] * source.layers, staging
        )
        gp.resize_tables(self.tables, hot_rows)
        gp.set_gate(self.tables, True)
        self.buffers = gp.allocate_step_buffers(device, source.experts, staging)
        self.state, self.generation = "READY", 0
        self.plan: NativeBankPlan | None = None
        self.completed: set[tuple[int, str]] = set()
        self.leases: set[object] = set()
        self.pending_promotions: set[int] = set()
        self.graphs: set[torch.cuda.CUDAGraph] = set()
        self.stable_graphs: set[torch.cuda.CUDAGraph] = set()
        self.lock = RLock()
        self.copy_bytes = self.copy_calls = 0
        self.pinned_params: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
        self.dma_stream = None
        self.direct_rows = self.staged_rows = 0
        # A publication-only hint avoids reading experts already in HOT. It
        # never authorizes a GPU read: device tables and leases remain authority.
        self.host_hot = {}
        self.host_rows = {}
        self.prefetch_reader = None
        self.prefetch_future = None
        self.prefetch_error = None
        self.prefetch_layer = None
        self.prefetch_ids: tuple[int, ...] = ()
        self.next_demand = None
        self.prepared = {}
        self.prefetch_experts = self.prefetch_consumed = 0
        self.views = self._views(self.rows)
        # These views describe reserved VA, not physical residency. Only mapped
        # rows published by the current plan may be read by a native consumer.
        self.consumer_views = self._views(self.max_rows)
        self._initialize_free_rows(0, self.rows)

    def _allocate_pinned(self):
        pinned = {
            name: torch.empty(
                (self.staging, *self.shapes[name]),
                dtype=self.dtypes[name],
                device="cpu",
                pin_memory=True,
            )
            for name in COMPONENTS
        }
        arrays = {
            name: value.numpy() if name in SCALARS else value.view(torch.uint8).numpy()
            for name, value in pinned.items()
        }
        return pinned, arrays

    def _fill_pinned(self, rows, positions, arrays):
        if len(rows) != len(positions):
            raise ValueError("incomplete native batch")
        for i, values in zip(positions, rows):
            if set(values) != set(COMPONENTS):
                raise ValueError("incomplete native row")
            for name in COMPONENTS:
                value = values[name]
                if value.shape != self.shapes[name] or value.dtype != (
                    np.dtype("float32") if name in SCALARS else np.dtype("uint8")
                ):
                    raise ValueError("incompatible native row")
                destination = arrays[name][i : i + 1].reshape(self.shapes[name])
                if value.ctypes.data != destination.ctypes.data:
                    destination[...] = value

    def _initialize_free_rows(self, start, end):
        for name, value in self.views.items():
            value[start:end].fill_(0 if name.endswith("weight") else 1)

    def _round(self, size):
        return (size + self.quantum - 1) // self.quantum * self.quantum

    def targets(self, hot_rows):
        if not 0 <= hot_rows <= self.max_rows - self.staging:
            raise ValueError("native HOT capacity outside reservation")
        rows = hot_rows + self.staging
        return {name: self._round(rows * self.strides[name]) for name in ARRAYS} | {
            "globals": self.scalar_bytes
        }

    def _views(self, rows):
        values = {
            name: self.backings[name]
            .tensor[: rows * self.strides[name]]
            .view(self.dtypes[name])
            .view(rows, *self.shapes[name])
            for name in ARRAYS
        }
        raw = self.backings["globals"].tensor.view(torch.float32)
        for i, name in enumerate(SCALARS):
            values[name] = raw[i * self.max_rows : i * self.max_rows + rows]
        return values

    def _check(self, plan):
        if plan is None or plan is not self.plan:
            raise RuntimeError("stale or foreign native plan")

    def begin(self, layer, ids, *, destinations=None):
        with self.lock:
            if self.state != "READY" or self.leases:
                raise RuntimeError("native bank unavailable")
            self.state = "LOADING"
            try:
                self._finish_prefetch(layer)
                if destinations is not None:
                    return self._begin_placement(layer, ids, destinations)
                gp.step(self.tables, layer, ids, self.buffers)
                if int(self.tables.error[0]):
                    raise ValueError("invalid native expert route")
                n = int(self.buffers.gather_count[0])
                self.generation += 1
                self.plan = NativeBankPlan(
                    self.generation,
                    layer,
                    tuple(self.buffers.routes[: ids.numel()].tolist()),
                    tuple(
                        zip(
                            self.buffers.gather_src[:n].tolist(),
                            self.buffers.gather_dst[:n].tolist(),
                        )
                    ),
                )
                self.completed.clear()
                return self.plan
            except Exception:
                self.prepared.clear()
                self.state = "POISONED"
                raise

    def _begin_placement(self, layer, ids, destinations):
        """Reserve explicit admission rows under the ordinary LOADING epoch."""
        experts = ids.tolist()
        rows = tuple(destinations)
        if (
            self.pending_promotions
            or not 0 <= layer < self.source.layers
            or ids.dtype != torch.int32
            or ids.ndim != 1
            or not 0 < len(experts) <= self.staging
            or len(rows) != len(experts)
            or len(set(experts)) != len(experts)
            or len(set(rows)) != len(rows)
            or any(not 0 <= e < self.source.experts for e in experts)
            or any(
                type(row) is not int or not 0 <= row < self.tables.pool_rows
                for row in rows
            )
            or any((layer, e) in self.host_hot for e in experts)
        ):
            raise ValueError("invalid explicit expert placement")
        keys = [layer * self.source.experts + e for e in experts]
        old = [self.host_rows.get(row) for row in rows]
        expected = [
            -1 if key is None else key[0] * self.source.experts + key[1] for key in old
        ]
        if (
            self.tables.row_key[list(rows)].tolist() != expected
            or self.tables.hot_phys[keys].tolist() != [-1] * len(keys)
            or int(self.tables.error[0])
        ):
            raise RuntimeError("explicit placement disagrees with device ownership")
        victims = [key for key in expected if key >= 0]
        if victims:
            if self.tables.hot_phys[victims].tolist() != [
                row for row, key in zip(rows, expected) if key >= 0
            ]:
                raise RuntimeError("explicit victim disagrees with reverse ownership")
            self.tables.hot_phys[victims] = -1
            self.tables.cold_phys[victims] = torch.tensor(
                [key % self.source.experts for key in victims],
                dtype=torch.int32,
                device=self.device,
            )
        # Consumers cannot see these provisional owners until all components
        # and ranks publish; a partial failure poisons the ordinary bank epoch.
        self.tables.hot_phys[keys] = torch.tensor(
            rows, dtype=torch.int32, device=self.device
        )
        self.tables.cold_phys[keys] = -1
        self.tables.row_key[list(rows)] = torch.tensor(
            keys, dtype=torch.int32, device=self.device
        )
        self.tables.row_use[list(rows)] = self.tables.clock[0]
        self.tables.miss_count[keys] = 0
        self.generation += 1
        self.plan = NativeBankPlan(
            self.generation, layer, (), tuple(zip(experts, rows))
        )
        self.completed.clear()
        return self.plan

    def prefetch(self, layer, experts):
        """Fill the other pinned generation while current DMA/compute runs.

        No destination is reserved or written here. The next ordinary stage
        consumes the completed pinned generation and votes before publication.
        """
        with self.lock:
            if getattr(self.source, "pin_cache", False):
                raise RuntimeError("pinned cache registration excludes source prefetch")
            active = (self.state == "READY" and self.leases) or (
                self.state == "LOADING" and self.plan is not None
            )
            if not active or self.prefetch_future:
                raise RuntimeError("native prefetch requires one active wave")
            victims = (
                {row for _, row in self.plan.copies if row < self.tables.pool_rows}
                if self.state == "LOADING" and self.plan is not None
                else set()
            )
            missing = tuple(
                e
                for e in experts
                if (layer, e) not in self.host_hot or self.host_hot[layer, e] in victims
            )
            if not missing:
                return
            if len(missing) > self.staging:
                raise ValueError("native prefetch exceeds one staging wave")
            region = getattr(self, "registered_source_region", None)
            if region is None and self.next_pinned is None:
                self.next_pinned, self.next_pinned_numpy = self._allocate_pinned()
            if self.prefetch_reader is None:
                self.prefetch_reader = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="expert-next-wave"
                )
            self.prefetch_layer, self.prefetch_ids = layer, missing
            self.prefetch_region = region

            def fill():
                rows = self.source.get_many(layer, missing)
                if region is not None:
                    return rows
                self._fill_pinned(rows, range(len(missing)), self.next_pinned_numpy)
                return None

            self.prefetch_future = self.prefetch_reader.submit(fill)
            self.prefetch_experts += len(missing)

    def start_next(self):
        if self.next_demand is not None:
            demand, self.next_demand = self.next_demand, None
            try:
                self.prefetch(*demand)
            except Exception as exc:
                # Even allocation/queue failure belongs to the next common
                # stage vote. Current published rows remain valid to consume.
                self.prefetch_error = exc

    def _finish_prefetch(self, layer):
        self.prepared.clear()
        self.prepared_native.clear()
        future, self.prefetch_future = self.prefetch_future, None
        error, self.prefetch_error = self.prefetch_error, None
        try:
            rows = None if future is None else future.result()
            if error is not None:
                raise error
            if future is None:
                return
            if layer != self.prefetch_layer:
                raise RuntimeError("stale native prefetch layer")
            if self.prefetch_region is not None:
                if (
                    self.prefetch_region
                    is not getattr(self, "registered_source_region", None)
                    or rows is None
                    or len(rows) != len(self.prefetch_ids)
                ):
                    raise RuntimeError("stale registered prefetch source")
                self.prepared_native = dict(zip(self.prefetch_ids, rows, strict=True))
                return
            # begin runs only after the preceding copy event and compute lease
            # completed. Neither buffer has an outstanding DMA reader now.
            self.pinned, self.next_pinned = self.next_pinned, self.pinned
            self.pinned_numpy, self.next_pinned_numpy = (
                self.next_pinned_numpy,
                self.pinned_numpy,
            )
            self.prepared = {expert: i for i, expert in enumerate(self.prefetch_ids)}
        finally:
            self.prefetch_layer, self.prefetch_ids = None, ()
            self.prefetch_region = None
            future = rows = None

    def drain_prefetch(self):
        """Cancellation joins CPU readers before releasing their source epoch."""
        with self.lock:
            try:
                self._finish_prefetch(self.prefetch_layer)
            finally:
                self.prepared.clear()
                self.prepared_native.clear()
                self.next_demand = None

    def close_prefetch(self):
        try:
            self.drain_prefetch()
        finally:
            if self.prefetch_reader is not None:
                self.prefetch_reader.shutdown(wait=True, cancel_futures=True)
                self.prefetch_reader = None

    def copy(self, plan):
        """Complete all local copies and drain pinned readers before returning."""
        with self.lock:
            self._check(plan)
            if self.state != "LOADING" or self.completed:
                raise RuntimeError("native copy unavailable")
            rows: list[Any] | None = None
            ordinary: list[Any] | None = None
            row: Any = None
            try:
                pairs = sorted(plan.copies, key=lambda pair: pair[1])
                region = getattr(self, "registered_source_region", None)
                if region is not None:
                    missing = [e for e, _ in pairs if e not in self.prepared_native]
                    self.prefetch_consumed += len(pairs) - len(missing)
                    self.prepared_native.update(
                        zip(
                            missing,
                            self.source.get_many(plan.layer, missing),
                            strict=True,
                        )
                    )
                    rows = [self.prepared_native[e] for e, _ in pairs]
                    self._copy_native_rows(rows, [slot for _, slot in pairs], region)
                    self.direct_rows += len(pairs)
                    self.start_next()
                    return
                missing = [e for e, _ in pairs if e not in self.prepared]
                self.prefetch_consumed += sum(e in self.prepared for e, _ in pairs)
                positions = dict(self.prepared)
                free = [i for i in range(self.staging) if i not in positions.values()]
                # Stale placement hints may leave unused prefetched rows. Only
                # actual demand owns a position in this generation.
                if len(free) < len(missing):
                    positions = {e: i for e, i in positions.items() if e in dict(pairs)}
                    free = [
                        i for i in range(self.staging) if i not in positions.values()
                    ]
                # Keep every returned bundle alive through the finally fence.
                # Its NumPy bases own a source row even if cache eviction occurs.
                read_into = getattr(self.source, "archive", None) is not None and (
                    self.source.pin_cache or not self.source.limit
                )
                rows = (
                    self.source.get_many_into(
                        plan.layer, missing, self.pinned_numpy, free[: len(missing)]
                    )
                    if read_into
                    else self.source.get_many(plan.layer, missing)
                )
                if len(rows) != len(missing):
                    raise ValueError("incomplete native batch")
                registered: dict[Any, tuple[list[int], list[int]]] = {}
                ordinary, ordinary_ids = [], []
                direct = set()
                if getattr(self.source, "pinned_pool", None) is not None:
                    from .expert_offload_pinned import PinnedExpertBundle

                    destinations = dict(pairs)
                    for expert, row in zip(missing, rows):
                        if isinstance(row, PinnedExpertBundle):
                            slab, slot = row.descriptor(self.source.pinned_pool)
                            src, dst = registered.setdefault(slab, ([], []))
                            src.append(slot)
                            dst.append(destinations[expert])
                            direct.add(expert)
                        else:
                            ordinary.append(row)
                            ordinary_ids.append(expert)
                else:
                    ordinary, ordinary_ids = rows, missing
                original_positions = dict(zip(missing, free))
                ordinary_positions = [original_positions[e] for e in ordinary_ids]
                self._fill_pinned(ordinary, ordinary_positions, self.pinned_numpy)
                positions.update(zip(ordinary_ids, ordinary_positions))
                self._copy_registered(registered)
                self.direct_rows += len(direct)
                pairs = [(e, r) for e, r in pairs if e not in direct]
                self.staged_rows += len(pairs)
                # The other CPU buffer starts filling before this generation's
                # DMA, rather than after rank copy-ready synchronization.
                self.start_next()
                start = 0
                while start < len(pairs):
                    end = start + 1
                    while (
                        end < len(pairs)
                        and pairs[end][1] == pairs[end - 1][1] + 1
                        and positions[pairs[end][0]] == positions[pairs[end - 1][0]] + 1
                    ):
                        end += 1
                    dst = pairs[start][1]
                    src = positions[pairs[start][0]]
                    for name in COMPONENTS:
                        self.views[name][dst : dst + end - start].copy_(
                            self.pinned[name][src : src + end - start],
                            non_blocking=True,
                        )
                        self.copy_calls += 1
                        self.copy_bytes += (end - start) * self.strides[name]
                        self.completed.update(
                            (row, name) for _, row in pairs[start:end]
                        )
                    start = end
            except Exception:
                self.state = "POISONED"
                raise
            finally:
                # Includes partial copy: pinned buffers cannot be reused early.
                self.fence().synchronize()
                # An exception traceback can outlive the failed stage. Once
                # DMA has drained, it must not keep evicted source row leases.
                rows = ordinary = row = None
                self.prepared.clear()
                self.prepared_native.clear()

    def _copy_native_rows(self, rows, destinations, region):
        """Copy leased immutable native RAM directly, without a second host copy."""
        if not rows:
            return
        from vllm import _custom_ops as ops

        if len(rows) != len(destinations):
            raise ValueError("unavailable registered source generation")
        sources, targets, sizes = [], [], []
        for row, slot in zip(rows, destinations, strict=True):
            if not 0 <= slot < self.rows:
                raise ValueError("native DMA destination outside committed bank")
            for name in COMPONENTS:
                value = row[name]
                address, nbytes = value.ctypes.data, value.nbytes
                if (
                    not value.flags.c_contiguous
                    or value.flags.writeable
                    or nbytes != self.strides[name]
                    or address < region.address
                    or address + nbytes > region.address + region.nbytes
                ):
                    raise ValueError("native DMA source outside leased layout")
                sources.append(address)
                targets.append(
                    self.consumer_views[name].data_ptr() + slot * self.strides[name]
                )
                sizes.append(nbytes)
        region.ensure(sources, sizes)
        descriptors = [
            torch.from_numpy(np.asarray(values, np.int64))
            for values in (sources, targets, sizes)
        ]
        current = stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream == 0:
            if self.dma_stream is None:
                self.dma_stream = torch.cuda.Stream(device=self.device)
            stream = self.dma_stream
            stream.wait_stream(current)
        try:
            with torch.cuda.stream(stream):
                ops.swap_blocks_batch(
                    descriptors[0],
                    descriptors[1],
                    descriptors[2],
                    is_src_access_order_any=True,
                )
            self.copy_calls += 1
            self.copy_bytes += sum(sizes)
            self.completed.update(
                (slot, name) for slot in destinations for name in COMPONENTS
            )
        finally:
            if stream is not current:
                current.wait_stream(stream)

    def _copy_registered(self, groups):
        if not groups:
            return
        from vllm.v1.simple_kv_offload import cuda_mem_ops

        current = stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream == 0:
            # cuMemcpyBatchAsync rejects the legacy NULL stream. Preserve that
            # caller's dependencies through one reusable explicit copy stream.
            if self.dma_stream is None:
                self.dma_stream = torch.cuda.Stream(device=self.device)
            stream = self.dma_stream
            stream.wait_stream(current)
        try:
            for slab, (sources, destinations) in groups.items():
                if any(not 0 <= row < self.rows for row in destinations):
                    raise ValueError("pinned DMA destination outside committed bank")
                params = self.pinned_params.get(slab)
                if params is None:
                    params = cuda_mem_ops.build_params(
                        {name: slab.views[name] for name in COMPONENTS},
                        {name: self.consumer_views[name] for name in COMPONENTS},
                        stream,
                    )
                    self.pinned_params[slab] = params
                # ps.graph_capture can change the current stream after construction.
                params = params._replace(stream_handle=stream.cuda_stream)
                cuda_mem_ops.copy_blocks(sources, destinations, params)
                self.copy_calls += 1
                self.copy_bytes += len(sources) * sum(self.strides.values())
                self.completed.update(
                    (row, name) for row in destinations for name in COMPONENTS
                )
        finally:
            if stream is not current:
                # Include partial DMA in the caller's eventual completion fence.
                current.wait_stream(stream)

    def local_ready(self, plan):
        self._check(plan)
        return self.state == "LOADING" and self.completed == {
            (row, name) for _, row in plan.copies for name in COMPONENTS
        }

    def publish(self, plan, *, all_ranks_ready):
        with self.lock:
            self._check(plan)
            if not all_ranks_ready or not self.local_ready(plan):
                self.state = "POISONED"
                raise RuntimeError("incomplete native publication")
            for expert, row in plan.copies:
                if row < self.tables.pool_rows:
                    old = self.host_rows.get(row)
                    if old is not None:
                        self.host_hot.pop(old, None)
                    key = (plan.layer, expert)
                    self.host_rows[row] = key
                    self.host_hot[key] = row
            self.prepared.clear()
            self.state = "READY"

    def abort(self, plan):
        with self.lock:
            self._check(plan)
            if self.leases:
                raise RuntimeError("native bank leased")
            self.fence().synchronize()
            self.state = "POISONED"

    def acquire(self, plan):
        with self.lock:
            self._check(plan)
            if self.state != "READY":
                raise RuntimeError("native bank not ready")
            lease = object()
            self.leases.add(lease)
            return lease

    def release(self, lease, event):
        with self.lock:
            if lease not in self.leases or event is None:
                raise RuntimeError("invalid or unfenced native reader")
            event.synchronize()
            self.leases.remove(lease)

    def fence(self):
        # CPU experts participate in the stream DAG. Busy host waits can
        # starve their pinned workers while the GPU waits for those workers.
        event = torch.cuda.Event(blocking=True)
        event.record(torch.cuda.current_stream(self.device))
        return event

    def register_graph(self, graph, *, stable_views=False):
        if self.state != "READY" or not self.leases:
            raise RuntimeError("graph capture requires a native read lease")
        self.graphs.add(graph)
        if stable_views:
            self.stable_graphs.add(graph)

    def prepare_resize(self, hot_rows):
        """Close consumers before an externally admitted VMM transaction."""
        with self.lock:
            targets = self.targets(hot_rows)
            if (
                self.leases
                or self.pending_promotions
                or self.state not in ("READY", "POISONED")
            ):
                raise RuntimeError("native bank maintenance unavailable")
            self.drain_prefetch()
            self.state, self.plan = "RESIZING", None
            event = self.fence()
            event.synchronize()
            # Mapped consumers use fixed reserved views and external row maps.
            # Planner/active-shape captures still depend on resized metadata.
            for graph in self.graphs - self.stable_graphs:
                graph.reset()
            self.graphs.intersection_update(self.stable_graphs)
            return targets, event

    def finish_resize(self, hot_rows, *, invalidate=False):
        """Publish geometry; rollback must invalidate potentially overwritten rows."""
        with self.lock:
            if self.state != "RESIZING":
                raise RuntimeError("native resize was not prepared")
            observed = {
                name: owner.info.committed for name, owner in self.backings.items()
            }
            if observed != self.targets(hot_rows):
                self.state = "POISONED"
                raise RuntimeError("native mapping disagrees with published geometry")
            if invalidate:
                self.host_hot.clear()
                self.host_rows.clear()
                # Recovery cannot trust either side of a rejected ownership map.
                self.tables.hot_phys.fill_(-1)
                self.tables.cold_phys.copy_(
                    torch.arange(
                        self.tables.keys, device=self.device, dtype=torch.int32
                    ).remainder(self.source.experts)
                )
                self.tables.row_key.fill_(-1)
                gp.resize_tables(self.tables, 0)
                self.tables.clock.zero_()
                self.tables.forwards.zero_()
                self.tables.miss_count.zero_()
            old_hot = self.tables.pool_rows
            evicted = gp.resize_tables(self.tables, hot_rows)
            for row in tuple(self.host_rows):
                if row >= hot_rows:
                    self.host_hot.pop(self.host_rows.pop(row), None)
            self.rows = hot_rows + self.staging
            self.views = self._views(self.rows)
            self._initialize_free_rows(min(old_hot, hot_rows), self.rows)
            self.generation += 1
            self.tables.error.zero_()
            self.state = "READY"
            return evicted


def make_native_bank_consumer(bank, prefix):
    """Attach finalized storage without allocating or repacking a second bank.

    The returned factory owns active-shape metadata only. Its graphs must be
    retired with the bank before geometry changes. It is not a checkpoint
    loader target: all ten native components are already finalized by source.
    """
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        make_nvfp4_moe_kernel,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
    )

    if bank.state != "READY":
        raise RuntimeError("native bank is not published")
    with torch.device("meta"):
        layer = FusedMoEFactory(
            num_experts=bank.rows,
            top_k=1,
            hidden_size=bank.source.hidden,
            intermediate_size=bank.source.physical,
            params_dtype=torch.bfloat16,
            renormalize=False,
            reduce_results=False,
            tp_size=bank.source.tp,
            dp_size=1,
            prefix=prefix,
            quant_config=ModelOptNvFp4Config(
                quant_method="NVFP4",
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
            ),
        )
    method = layer._quant_method
    assert isinstance(method, ModelOptNvFp4FusedMoE)
    if method.nvfp4_backend != NvFp4MoeBackend.VLLM_CUTLASS:
        raise RuntimeError("native bank requires the admitted CUTLASS schema")
    for name in ARRAYS + SCALARS[:4]:
        layer.routed_experts.register_parameter(
            name, torch.nn.Parameter(bank.views[name], requires_grad=False)
        )
    method.moe_quant_config = method.get_fused_moe_quant_config(layer.routed_experts)
    assert method.experts_cls is not None
    for name in SCALARS[4:]:
        getattr(method.moe_quant_config, name).data = bank.views[name]
    method.moe_kernel = make_nvfp4_moe_kernel(
        moe_quant_config=method.moe_quant_config,
        moe_config=method.moe,
        experts_cls=method.experts_cls,
        backend=method.nvfp4_backend,
        routing_tables=layer.routed_experts._expert_routing_tables(),
    )
    # CUTLASS's normal finalizer only multiplies alphas by activation scales.
    # Source.prepare already did that; applying it twice corrupts the weights.
    return layer


class NativeMappedBankConsumer:
    """Same CUTLASS MoE DAG with compact groups addressing leased bank rows.

    Routing, publication and TP reduction belong to the provider/outer owner.
    This consumer neither copies expert weights nor creates a model registry
    entry. Stable reserved views survive resize; the provider validates that
    every selected row belongs to the currently committed physical prefix.
    """

    def __init__(self, bank, expert_rows, *, num_groups=None):
        self.num_groups = bank.staging if num_groups is None else num_groups
        if (
            bank.state != "READY"
            or not 1 <= self.num_groups <= bank.source.experts
            or expert_rows.shape != (self.num_groups,)
        ):
            raise RuntimeError("invalid mapped native consumer geometry")
        self.bank = bank
        self.values, self.expert_rows = bank.consumer_views, expert_rows
        w13, w2 = self.values["w13_weight"], self.values["w2_weight"]
        if w13.ndim != 3 or w2.ndim != 3:
            raise ValueError("invalid mapped native weight rank")
        rows, twice_n, half_k = w13.shape
        n, k = twice_n // 2, half_k * 2
        expected = {
            "w2_weight": (rows, k, n // 2),
            "w13_weight_scale": (rows, 2 * n, k // 16),
            "w2_weight_scale": (rows, k, n // 16),
        }
        if (
            twice_n % 2
            or n < 64
            or n % 64
            or k % 128
            or any(self.values[name].shape != shape for name, shape in expected.items())
        ):
            raise ValueError("invalid mapped native column/scale geometry")

    def __call__(self, hidden, weights, ids):
        if not all(value.is_contiguous() for value in (hidden, weights, ids)):
            raise ValueError("mapped native inputs must be contiguous")
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            run_cutlass_moe_fp4,
        )

        m, k = hidden.shape
        if k != self.values["w13_weight"].shape[2] * 2:
            raise ValueError("mapped native activation/weight hidden mismatch")
        lanes = m * ids.shape[1]
        # The admitted row owns its local columns. TP partitions need not
        # retain the uniform splitter's all-zero final column block.
        n = self.values["w13_weight"].shape[1] // 2
        values = self.values
        output = torch.empty_like(hidden)
        workspace13 = torch.empty(
            (lanes, max(2 * n, k)), device=hidden.device, dtype=hidden.dtype
        )
        workspace2 = torch.empty((lanes, n), device=hidden.device, dtype=hidden.dtype)
        run_cutlass_moe_fp4(
            output=output,
            a=hidden,
            a1_gscale=values["a1_gscale"],
            w1_fp4=values["w13_weight"],
            w1_blockscale=values["w13_weight_scale"],
            w1_alphas=values["w13_weight_scale_2"],
            a2_gscale=values["a2_gscale"],
            w2_fp4=values["w2_weight"],
            w2_blockscale=values["w2_weight_scale"],
            w2_alphas=values["w2_weight_scale_2"],
            topk_weights=weights,
            topk_ids=ids,
            activation=MoEActivation.SILU,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=self.num_groups,
            device=hidden.device,
            expert_rows=self.expert_rows,
        )
        return output


class NativeBankCoordinator:
    """Rank-synchronous publication, with no CPU collective on HOT hits.

    All participants call stage in the same order, including local failures.
    Exact selected IDs, physical mapping, source epoch and copy plan are
    compared on the device before any rank enters a conditional copy phase.
    """

    def __init__(self, bank, group):
        self.bank, self.group = bank, group
        self.ranks = torch.distributed.get_world_size(group.device_group)
        source = bank.source
        index = source.root / "model.safetensors.index.json"
        epoch = sha256(index.read_bytes()).digest()
        self.identity = list(epoch) + [
            source.layers,
            source.experts,
            source.hidden,
            source.physical,
            source.tp,
        ]
        self.identity += list(
            sha256(
                repr(
                    (
                        getattr(source, "partition", "legacy"),
                        getattr(source, "spans", None),
                    )
                ).encode()
            ).digest()
        )
        width = max(72, 8 + len(self.identity) + bank.staging * 4)
        self.send = torch.empty(width, dtype=torch.int64, device=bank.device)
        self.recv = torch.empty(
            self.ranks * width, dtype=torch.int64, device=bank.device
        )
        self.copy_vote = torch.empty(1, dtype=torch.int32, device=bank.device)
        self.plan_votes = self.copy_votes = self.hot_steps = 0
        self.host_control = False

    def exchange_control(self, payload):
        """Exchange host-owned metadata; this is not a CUDA completion fence."""
        if self.host_control:
            send = torch.tensor(payload, dtype=torch.int64, device="cpu")
            recv = torch.empty(self.ranks * send.numel(), dtype=send.dtype)
            torch.distributed.all_gather_single(recv, send, group=self.group.cpu_group)
        else:
            self.send.copy_(torch.tensor(payload, dtype=torch.int64))
            torch.distributed.all_gather_single(
                self.recv, self.send, group=self.group.device_group
            )
            recv = self.recv
        return recv.view(self.ranks, -1).cpu().numpy()

    def admit_routes(
        self, layer, ids, weights, *, error=None, dummy=False, resident=None
    ):
        """Agree the entire demand before entering a variable number of waves."""
        digest = sha256()
        if error is None:
            digest.update(ids.tobytes())
            digest.update(weights.tobytes())
        payload = [
            int(error is None and self.bank.state == "READY"),
            self.bank.generation,
            layer,
            ids.shape[0] if error is None else -1,
            ids.shape[1] if error is None else -1,
            int(dummy) + 2 * int(resident is not None),
        ] + list(digest.digest())
        base_width = len(payload)
        if resident is not None:
            payload += resident.vote
        payload += [-1] * (self.send.numel() - len(payload))
        self.send.copy_(torch.tensor(payload, dtype=torch.int64))
        if resident is not None:
            resident.validate(self.send[base_width:])
        torch.distributed.all_gather_single(
            self.recv, self.send, group=self.group.device_group
        )
        # One transfer retires metadata DMA and consumes both decisions. CPU
        # publication hints propose rows; device ownership authorizes reads.
        peers = self.recv.view(self.ranks, -1).cpu().numpy()
        common = (peers[:, :base_width] == peers[:1, :base_width]).all() and (
            peers[:, 0] == 1
        ).all()
        if not common:
            self.bank.state = "POISONED"
            raise RuntimeError("rank-inconsistent native routing") from error
        if resident is None:
            return False
        status = peers[:, base_width]
        all_hot = bool((status == 1).all())
        if (status < 0).any() or (
            all_hot and not (peers[:, base_width:] == peers[:1, base_width:]).all()
        ):
            self.bank.state = "POISONED"
            raise RuntimeError("rank-inconsistent resident expert ownership")
        return all_hot

    def recover(self):
        """Agree a new generation after physical rollback and full invalidation.

        A rejected local plan may have advanced only some ranks' counters.
        Returning to READY therefore requires an explicit shared epoch; source
        errors still require a fresh immutable source, not this state reset.
        """
        bank = self.bank
        valid = (
            bank.state == "READY"
            and not bank.leases
            and not bool((bank.tables.hot_phys >= 0).any())
            and not bank.source.closed
        )
        payload = [int(valid), bank.generation, bank.rows] + self.identity
        payload += [-1] * (self.send.numel() - len(payload))
        self.send.copy_(torch.tensor(payload, dtype=torch.int64))
        torch.distributed.all_gather_into_tensor(
            self.recv, self.send, group=self.group.device_group
        )
        peers = self.recv.view(self.ranks, -1).cpu()
        if not bool((peers[:, 0] == 1).all()) or not torch.equal(
            peers[:, 2:], peers[:1, 2:].expand(self.ranks, -1)
        ):
            bank.state = "POISONED"
            raise RuntimeError("native recovery requires a common empty bank")
        bank.generation = int(peers[:, 1].max()) + 1

    def admission_assignments(self, proposal):
        """Agree the entire assignment before any bounded copy wave begins."""
        from .expert_offload_admission import ExpertAdmissionPlan

        bank, error = self.bank, None
        try:
            current = tuple(
                sorted(
                    layer * bank.source.experts + expert
                    for layer, expert in bank.host_hot
                )
            )
            if (
                not isinstance(proposal, ExpertAdmissionPlan)
                or proposal.resident != current
                or proposal.capacity != bank.tables.pool_rows
                or bank.state != "READY"
                or bank.leases
                or bank.pending_promotions
                or len(current) > proposal.capacity
                or any(
                    type(key) is not int
                    or not 0 <= key < bank.source.layers * bank.source.experts
                    for key in proposal.desired
                )
            ):
                raise ValueError("stale or unavailable expert assignment")
            retained = set(proposal.desired) & set(current)
            occupied = {
                row
                for (layer, expert), row in bank.host_hot.items()
                if layer * bank.source.experts + expert in retained
            }
            free = [row for row in range(proposal.capacity) if row not in occupied]
            if len(free) < len(proposal.promotions):
                raise ValueError("expert assignment exceeds committed rows")
            assignments = sorted(
                (key // bank.source.experts, key % bank.source.experts, row)
                for key, row in zip(proposal.promotions, free)
            )
            digest = sha256((proposal.digest + repr(assignments)).encode()).digest()
        except Exception as exc:
            error, digest = exc, bytes(32)
        payload = [int(error is None), bank.generation] + list(digest)
        payload += [-1] * (self.send.numel() - len(payload))
        peers = self.exchange_control(payload)
        if not (peers[:, 0] == 1).all() or not (peers == peers[:1]).all():
            bank.state = "POISONED"
            raise RuntimeError("rank-inconsistent expert assignment") from error
        return assignments

    def apply_admission(self, proposal):
        """Synchronous reference transaction for the asynchronous control."""
        bank = self.bank
        assignments = self.admission_assignments(proposal)
        offset = 0
        while offset < len(assignments):
            layer = assignments[offset][0]
            end = offset
            while (
                end < len(assignments)
                and assignments[end][0] == layer
                and end - offset < bank.staging
            ):
                end += 1
            batch = assignments[offset:end]
            self.stage(
                layer,
                torch.tensor([e for _, e, _ in batch], dtype=torch.int32),
                destinations=[r for _, _, r in batch],
            )
            offset = end
        observed = tuple(
            sorted(
                layer * bank.source.experts + expert for layer, expert in bank.host_hot
            )
        )
        if observed != proposal.desired:
            bank.state = "POISONED"
            raise RuntimeError("published expert assignment differs from admission")

    def stage(self, layer, ids, *, destinations=None):
        bank = self.bank
        error, plan = None, None
        try:
            plan = bank.begin(layer, ids, destinations=destinations)
            raw_ids = ids.flatten().tolist()
        except Exception as exc:
            error, raw_ids = exc, []
        payload = [
            int(error is None),
            bank.generation,
            layer,
            len(raw_ids),
            len(plan.copies) if plan else 0,
            bank.rows,
            bank.staging,
            1 if destinations is None else 2,
        ] + self.identity
        if plan is not None:
            for values in (
                raw_ids,
                plan.routes,
                [src for src, _ in plan.copies],
                [dst for _, dst in plan.copies],
            ):
                payload += list(values) + [-1] * (bank.staging - len(values))
        else:
            payload += [-1] * (bank.staging * 4)
        payload += [-1] * (self.send.numel() - len(payload))
        self.send.copy_(torch.tensor(payload, dtype=torch.int64))
        torch.distributed.all_gather_into_tensor(
            self.recv, self.send, group=self.group.device_group
        )
        self.plan_votes += 1
        peers = self.recv.view(self.ranks, -1)
        common = bool(((peers == peers[:1]).all() & (peers[:, 0] == 1).all()).item())
        if not common:
            bank.state = "POISONED"
            raise RuntimeError("rank-inconsistent native plan") from error
        assert plan is not None
        if not plan.copies:
            bank.start_next()
            bank.publish(plan, all_ranks_ready=True)
            self.hot_steps += 1
            return plan
        try:
            bank.copy(plan)
        except Exception as exc:
            error = exc
        self.copy_vote.fill_(int(error is None and bank.local_ready(plan)))
        torch.distributed.all_reduce(
            self.copy_vote,
            op=torch.distributed.ReduceOp.MIN,
            group=self.group.device_group,
        )
        self.copy_votes += 1
        ready = bool(self.copy_vote.item())
        try:
            bank.publish(plan, all_ranks_ready=ready)
        except Exception as exc:
            raise RuntimeError("rank-incomplete native copy") from (error or exc)
        return plan
