# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded compact native expert execution at a breakable Graph seam."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr
from vllm.utils.torch_utils import direct_register_custom_op

from .expert_offload_admission import cache_snapshot, route_histogram
from .expert_offload_bank import NativeBankCoordinator, NativeMappedBankConsumer
from .expert_offload_plan import plan


@tr.jit
def _scatter_kernel(Y, IDS, LANES, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    lane = tl.load(IDS + row)
    values = tl.load(Y + row * H + cols, cols < H, 0)
    tl.store(LANES + lane * H + cols, values, (lane >= 0) & (cols < H))


def _scatter(y: torch.Tensor, ids: torch.Tensor, lanes: torch.Tensor) -> None:
    _scatter_kernel[(y.shape[0], tr.cdiv(y.shape[1], 256))](
        y, ids, lanes, y.shape[1], 256
    )


def _scatter_fake(y: torch.Tensor, ids: torch.Tensor, lanes: torch.Tensor) -> None:
    pass


direct_register_custom_op(
    "flashnext_native_scatter",
    _scatter,
    mutates_args=["lanes"],
    fake_impl=_scatter_fake,
)


@eager_break_during_capture
def _native_experts(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    is_padding: torch.Tensor | None = None,
) -> None:
    owner = get_forward_context().no_compile_layers[layer_name]
    if BreakableCUDAGraphCapture.is_active():
        # The captured router has not executed yet. Only replay may read its
        # demand; capture registers this callback without touching residency.
        output.zero_()
        return
    owner.provider.run(
        owner.layer_id, hidden, weights, ids, output, is_padding=is_padding
    )


def _native_experts_fake(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    is_padding: torch.Tensor | None = None,
) -> None:
    pass


direct_register_custom_op(
    "flashnext_native_experts",
    _native_experts,
    mutates_args=["output"],
    fake_impl=_native_experts_fake,
)


class NativeExpertProvider:
    """One serialized worker owns shared staging and a global expert bank.

    Inner Graphs consume stable staging addresses, independent of which outer
    layer or request owns the input/output. Only the scheduler may resize the
    bank; its transaction must quiesce readers before changing physical mappings.
    """

    def __init__(
        self, bank, group, vllm_config, *, topk, max_tokens=4096, max_lanes=4096
    ):
        bank.source.geometry.validate_cutlass()
        if (
            type(topk) is not int
            or not 1 <= topk <= bank.source.experts
            or max_tokens < 1
            or max_lanes < 1
            or max_lanes & (max_lanes - 1)
            or max_lanes > max_tokens * topk
            or bank.source.tp != vllm_config.parallel_config.tensor_parallel_size
            or bank.source.tp != torch.distributed.get_world_size(group.device_group)
            or bank.source.rank != torch.distributed.get_rank(group.device_group)
        ):
            raise ValueError("invalid compact native provider geometry")
        self.bank, self.config = bank, vllm_config
        self.coordinator = NativeBankCoordinator(bank, group)
        self.max_tokens, self.max_lanes = max_tokens, max_lanes
        self.topk, self.hidden = topk, bank.source.hidden
        self.x = torch.empty(
            max_tokens, self.hidden, device=bank.device, dtype=torch.bfloat16
        )
        self.weights = torch.empty(
            max_tokens, self.topk, device=bank.device, dtype=torch.float32
        )
        self.lanes = torch.empty(
            max_tokens, self.topk, self.hidden, device=bank.device, dtype=torch.bfloat16
        )
        self.lane_ids = torch.empty(max_lanes, device=bank.device, dtype=torch.int64)
        self.local_ids = torch.empty(
            max_lanes, 1, device=bank.device, dtype=torch.int32
        )
        self.group_rows = torch.empty(
            bank.staging, device=bank.device, dtype=torch.int32
        )
        self.host_lanes = torch.empty(
            max_lanes, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.host_local_ids = torch.empty(
            max_lanes, 1, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.host_group_rows = torch.empty(
            bank.staging, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self.expert_ids = torch.empty(
            bank.staging, device=bank.device, dtype=torch.int32
        )
        self.consumer = None
        self.consumer_rows = None
        self.kernels = {}
        # No captured intermediate escapes forward: the persistent lane buffer
        # owns the consumed output. These Graphs never execute concurrently.
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.graph_captures = 0
        self.use_graphs = True
        # Standalone opt-in until the full source/copy DAG shows a benefit.
        self.use_prefetch = False
        self.use_hot_path = False
        self.use_scan_order = False
        self.hot_path = None
        self.hybrid_path = None
        self.stream_path = None
        self.admission = None
        self.dummy = False
        self.active = False
        self.forward_count = 0
        self.last_step = {}
        self.trace_path = os.environ.get("AG2_VLLM_EXPERT_TRACE", "")
        self.trace_layers = []
        self.trace_steps = 0
        self.trace_before = None
        self.trace_bytes = 0
        self.trace_byte_limit = 64 << 20
        self.trace_limit = int(os.environ.get("AG2_VLLM_EXPERT_TRACE_STEPS", "64"))
        if not 0 <= self.trace_limit <= 4096:
            raise ValueError("expert trace step budget must be within 0..4096")
        if self.trace_path:
            with open(f"{self.trace_path}-rank{bank.source.rank}.jsonl", "a"):
                pass

    def prepare_execution(self, *, dummy, num_tokens=None, identity=None):
        if self.stream_path is not None:
            self.stream_path.abort_step()
        if self.active or self.bank.leases or self.bank.state != "READY":
            raise RuntimeError("cannot prepare an unavailable expert provider")
        self.dummy = bool(dummy)
        self.execution_identity = identity
        if self.stream_path is not None:
            self.stream_path.prepare(
                dummy=dummy, num_tokens=num_tokens, identity=identity
            )

    def finish_execution(self, *, dummy):
        if self.stream_path is not None:
            self.stream_path.finish(dummy=dummy)

    def enable_cpu_experts(self, executor):
        """Attach an admitted bounded CPU executor before profiling/serving."""
        self.quiesce()
        if self.hybrid_path is not None:
            raise RuntimeError("CPU expert executor already attached")
        from .expert_offload_hot import NativeHotRead
        from .expert_offload_hybrid import NativeHybridRead

        if self.hot_path is None:
            self.hot_path = NativeHotRead(self)
        self.hybrid_path = NativeHybridRead(self, executor)
        self.use_hot_path = True

    def enable_frequency_admission(
        self,
        *,
        history_steps,
        max_promotions,
        exclusive_ram,
        asynchronous=True,
        registered_source=False,
    ):
        """Opt in only after the matched copy/source/execution control passes."""
        from . import expert_offload_tables as gp
        from .expert_offload_admission import NativeExpertAdmission

        self.quiesce()
        if self.admission is not None:
            raise RuntimeError("expert admission owner already attached")
        self.admission = NativeExpertAdmission(
            self,
            history_steps=history_steps,
            max_promotions=max_promotions,
            exclusive_ram=exclusive_ram,
            asynchronous=asynchronous,
            registered_source=registered_source,
        )
        # Demand misses use bounded staging. Only the whole-model admission
        # owner can replace HOT, so layer traversal no longer controls eviction.
        gp.set_gate(self.bank.tables, False)
        if registered_source:
            capacity = self.bank.source.cpu.source_stats()["capacity"]
            self.use_prefetch = capacity >= 2 * self.bank.staging

    def _profile(self, layer, hidden, output):
        # Explicit dummy mode is supplied by ModelState, never inferred from
        # attention metadata. It profiles the maximum native tile without I/O.
        ids = np.full((hidden.shape[0], self.topk), -1, dtype=np.int32)
        weights = np.zeros(ids.shape, dtype=np.float32)
        self.coordinator.admit_routes(layer, ids, weights, dummy=True)
        ticket = self.coordinator.stage(layer, self.expert_ids[:0])
        lease = self.bank.acquire(ticket)
        try:
            self.x.zero_()
            self.weights.zero_()
            self.lane_ids.copy_(torch.arange(self.max_lanes, device=self.bank.device))
            self.local_ids.zero_()
            self.group_rows.zero_()
            _, graph = self._kernel(self.max_lanes)
            graph.replay()
            if self.use_hot_path:
                if self.hot_path is None:
                    from .expert_offload_hot import NativeHotRead

                    self.hot_path = NativeHotRead(self)
                self.hot_path.profile()
            output.zero_()
        finally:
            self.bank.release(lease, self.bank.fence())

    def quiesce(self):
        if self.stream_path is not None:
            self.stream_path.abort_step()
        if self.active or self.bank.leases:
            raise RuntimeError("cannot retire an active expert provider")
        if self.admission is not None:
            self.admission.quiesce()
        self.bank.fence().synchronize()

    def retire(self):
        self.quiesce()
        self.bank.close_prefetch()
        if self.admission is not None:
            self.admission.close()
            self.admission = None
        if self.stream_path is not None:
            self.stream_path.retire()
            self.stream_path.cpu.close()
            self.stream_path = None
        for _, graph in self.kernels.values():
            graph.reset()
            self.bank.graphs.discard(graph)
            self.bank.stable_graphs.discard(graph)
        self.kernels.clear()
        if self.hybrid_path is not None:
            self.hybrid_path.close()
            self.hybrid_path = None
        self.hot_path = None
        # The allocator releases a pool when its last Graph is reset. A later
        # recovery capture needs a fresh pool epoch, even at the same addresses.
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.consumer = None
        self.consumer_rows = None

    def _kernel(self, bucket):
        if bucket in self.kernels:
            if (
                self.kernels[bucket][1] not in self.bank.graphs
                or self.consumer_rows != self.bank.max_rows
            ):
                raise RuntimeError("native provider graphs require retirement")
            return self.kernels[bucket]
        if self.consumer is None:
            self.consumer = NativeMappedBankConsumer(self.bank, self.group_rows)
            self.consumer_rows = self.bank.max_rows
        consumer = self.consumer
        lane_ids, local_ids = self.lane_ids[:bucket], self.local_ids[:bucket]

        def forward():
            compact = self.x.index_select(0, lane_ids.clamp_min(0) // self.topk)
            weights = self.weights.flatten().index_select(0, lane_ids.clamp_min(0))
            y = consumer(compact, (weights * (lane_ids >= 0)).unsqueeze(1), local_ids)
            torch.ops.vllm.flashnext_native_scatter(y, lane_ids, self.lanes)

        with set_forward_context(None, self.config, num_tokens=bucket):
            compiled = torch.compile(forward, fullgraph=True)
            for _ in range(3):
                compiled()
            self.bank.fence().synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                compiled()
            self.bank.register_graph(graph, stable_views=True)
            self.graph_captures += 1

        def direct():
            with set_forward_context(None, self.config, num_tokens=bucket):
                forward()

        self.kernels[bucket] = direct, graph
        return direct, graph

    def run(self, layer, hidden, weights, ids, output, *, is_padding=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("native expert demand requires breakable Graph capture")
        if self.active:
            raise RuntimeError("native provider does not support concurrent forwards")
        self.active = True
        before_bytes = self.bank.copy_bytes
        before_read = self.bank.source.read_bytes
        before_hits, before_misses = self.bank.source.hits, self.bank.source.misses
        before_prefetch = self.bank.prefetch_experts
        before_consumed = self.bank.prefetch_consumed
        started = time.perf_counter()
        steps, tiles, useful, hot_calls = 0, 0, 0, 0
        route_trace = {}
        hybrid_used = False
        scan_source = (
            self.bank.source.cpu
            if self.use_scan_order and not self.dummy and hidden.shape[0] > 64
            else None
        )
        try:
            if scan_source is not None:
                scan_source.source_scan_order(True)
            error = None
            cpu_ids = cpu_weights = cpu_padding = None
            validated_waves = resident = None
            try:
                if (
                    hidden.ndim != 2
                    or hidden.shape[1] != self.hidden
                    or hidden.dtype != torch.bfloat16
                    or hidden.device != self.bank.device
                    or ids.shape != (hidden.shape[0], self.topk)
                    or weights.shape != ids.shape
                    or weights.dtype != torch.float32
                    or output.shape != hidden.shape
                    or output.dtype != hidden.dtype
                    or output.device != hidden.device
                    or not 0 <= layer < self.bank.source.layers
                ):
                    raise ValueError("invalid native provider inputs")
                if is_padding is not None and (
                    is_padding.shape != (hidden.shape[0],)
                    or is_padding.dtype != torch.bool
                    or is_padding.device != self.bank.device
                ):
                    raise ValueError("invalid native padding mask")
                if not self.dummy:
                    cpu_ids = ids.cpu().numpy()
                    cpu_weights = weights.cpu().numpy()
                    if is_padding is not None:
                        cpu_padding = is_padding.cpu().numpy()
                        cpu_weights = cpu_weights.copy()
                        cpu_weights[cpu_padding] = 0
                    if not np.isfinite(cpu_weights).all() or np.any(cpu_weights < 0):
                        raise ValueError("invalid native routing weights")
                    validated_waves = plan(
                        cpu_ids,
                        self.bank.source.experts,
                        self.bank.staging,
                        is_padding=cpu_padding,
                        resident_experts=(
                            np.flatnonzero(
                                scan_source.resident_bitmap().reshape(
                                    self.bank.source.layers, self.bank.source.experts
                                )[layer]
                            )
                            if scan_source is not None
                            else None
                        ),
                    )
                    if (
                        self.use_hot_path
                        and 0 < hidden.shape[0] <= 64
                        and validated_waves
                    ):
                        if self.hot_path is None:
                            from .expert_offload_hot import NativeHotRead

                            self.hot_path = NativeHotRead(self)
                        resident = (
                            self.hybrid_path
                            if self.hybrid_path is not None
                            and hidden.shape[0] <= self.hybrid_path.max_tokens
                            else self.hot_path
                        )
                        resident.propose(layer, validated_waves)
                    if self.admission is not None:
                        self.admission.observe_prefill_layer(
                            layer, cpu_ids, cpu_weights
                        )
            except Exception as exc:
                error = exc
            if self.dummy and error is None:
                self._profile(layer, hidden, output)
                return
            all_hot = self.coordinator.admit_routes(
                layer,
                cpu_ids,
                cpu_weights,
                error=error,
                dummy=self.dummy,
                resident=resident,
            )
            assert cpu_ids is not None
            assert cpu_weights is not None
            if self.trace_path and self.trace_steps < self.trace_limit:
                route_trace = self._trace_routes(layer, cpu_ids, cpu_weights)
            if hidden.shape[0] == 0:
                output.zero_()
                return
            for start in range(0, hidden.shape[0], self.max_tokens):
                count = min(self.max_tokens, hidden.shape[0] - start)
                self.x[:count].copy_(hidden[start : start + count])
                self.weights[:count].copy_(weights[start : start + count])
                padding = (
                    None if cpu_padding is None else cpu_padding[start : start + count]
                )
                if padding is not None and padding.any():
                    padding_rows = torch.as_tensor(
                        np.flatnonzero(padding), device=self.bank.device
                    )
                    self.x[:count].index_fill_(0, padding_rows, 0)
                    self.weights[:count].index_fill_(0, padding_rows, 0)
                    self.lanes[:count].index_fill_(0, padding_rows, 0)
                waves = (
                    validated_waves
                    if hidden.shape[0] <= self.max_tokens
                    else plan(
                        cpu_ids[start : start + count],
                        self.bank.source.experts,
                        self.bank.staging,
                        is_padding=padding,
                    )
                )
                assert waves is not None
                if all_hot:
                    assert resident is not None
                    if self.hybrid_path is not None and resident is self.hybrid_path:
                        resident.run(layer, waves, cpu_ids, cpu_weights)
                        hybrid_used = True
                    else:
                        resident.run(layer, waves, cpu_ids)
                    output[:count].copy_(resident.output[:count])
                    steps += len(waves)
                    tiles += 1
                    useful += sum(len(wave.lanes) for wave in waves)
                    hot_calls += 1
                    continue
                for wave_index, wave in enumerate(waves):
                    self.bank.next_demand = (
                        (layer, waves[wave_index + 1].experts)
                        if self.use_prefetch and wave_index + 1 < len(waves)
                        else None
                    )
                    demand = self.expert_ids[: len(wave.experts)]
                    demand.copy_(torch.tensor(wave.experts, dtype=torch.int32))
                    ticket = self.coordinator.stage(layer, demand)
                    lease = self.bank.acquire(ticket)
                    try:
                        rows = np.asarray(ticket.routes, dtype=np.int32)
                        if rows.shape != (len(wave.experts),) or np.any(
                            (rows < 0) | (rows >= self.bank.rows)
                        ):
                            raise RuntimeError("invalid published native physical rows")
                        self.host_group_rows.zero_()
                        self.host_group_rows[: len(rows)].copy_(torch.from_numpy(rows))
                        self.group_rows.copy_(self.host_group_rows, non_blocking=True)
                        for tile in wave.tiles(self.max_lanes):
                            size, bucket = len(tile.lanes), tile.bucket
                            host_lanes = self.host_lanes[:bucket]
                            host_local_ids = self.host_local_ids[:bucket]
                            host_lanes.fill_(-1)
                            host_local_ids.zero_()
                            host_lanes[:size].copy_(torch.from_numpy(tile.lanes.copy()))
                            host_local_ids[:size, 0].copy_(
                                torch.from_numpy(tile.slots.copy())
                            )
                            self.lane_ids[:bucket].copy_(host_lanes, non_blocking=True)
                            self.local_ids[:bucket].copy_(
                                host_local_ids, non_blocking=True
                            )
                            direct, graph = self._kernel(bucket)
                            if self.use_graphs:
                                graph.replay()
                            else:
                                direct()
                            # Pinned metadata and output lanes cannot be reused
                            # while DMA or their last reader is outstanding.
                            self.bank.fence().synchronize()
                            tiles += 1
                            useful += size
                    finally:
                        self.bank.release(lease, self.bank.fence())
                    steps += 1
                output[start : start + count].copy_(self.lanes[:count].sum(dim=1))
            if self.admission is not None and layer == self.bank.source.layers - 1:
                self.admission.finish_step()
            self.forward_count += 1
        except Exception:
            self.bank.state = "POISONED"
            raise
        finally:
            drain_error = None
            try:
                self.bank.drain_prefetch()
                if scan_source is not None:
                    scan_source.source_scan_order(False)
            except Exception as exc:
                self.bank.state = "POISONED"
                drain_error = exc
            self.active = False
            self.last_step = dict(
                layer=layer,
                waves=steps,
                tiles=tiles,
                useful_lanes=useful,
                h2d_expert_bytes=self.bank.copy_bytes - before_bytes,
                source_read_bytes=self.bank.source.read_bytes - before_read,
                prefetched_experts=self.bank.prefetch_experts - before_prefetch,
                consumed_prefetch=self.bank.prefetch_consumed - before_consumed,
                wall_ms=(time.perf_counter() - started) * 1000,
                state=self.bank.state,
                dummy=self.dummy,
                hot_calls=hot_calls,
                hot_rows=self.bank.tables.pool_rows,
                source_hits=self.bank.source.hits - before_hits,
                source_misses=self.bank.source.misses - before_misses,
                ram_retained_bytes=self.bank.source.used,
                tokens=hidden.shape[0],
                execution="hybrid" if hybrid_used else "gpu",
                hybrid=dict(self.hybrid_path.last_step)
                if hybrid_used and self.hybrid_path is not None
                else {},
                route=route_trace,
                admission=dict(self.admission.last_step)
                if self.admission is not None and layer == self.bank.source.layers - 1
                else {},
            )
            self._trace_last_step()
            if drain_error is not None:
                raise RuntimeError(
                    "native prefetch failed while draining"
                ) from drain_error

    def _disable_trace(self, error):
        from vllm.logger import init_logger

        init_logger(__name__).warning("Expert trace failed: %s", error)
        self.trace_path = ""

    def _trace_routes(self, layer, ids, weights):
        try:
            if layer == 0:
                self.trace_before = cache_snapshot(self.bank)
            counts, useful_tokens = route_histogram(
                ids, weights, self.bank.source.experts
            )
            selected = np.flatnonzero(counts).tolist()
            # One metadata snapshot, without probing or leasing expert rows.
            ram_keys = set(self.bank.source.cache)
            context = get_forward_context()
            descriptor = context.batch_descriptor
            route = dict(
                selected_experts=selected,
                selected_histogram=counts.tolist(),
                useful_tokens=useful_tokens,
                gpu_hint_experts=[
                    e for e in selected if (layer, e) in self.bank.host_hot
                ],
                ram_experts=[e for e in selected if (layer, e) in ram_keys],
                runtime=dict(
                    num_tokens_unpadded=context.num_tokens_unpadded,
                    target_pure_decode=context.tp3_sd_phase_reduce,
                    draft=context.tp3_mtp_device_ce,
                    graph_mode=str(context.cudagraph_runtime_mode),
                    descriptor=None
                    if descriptor is None
                    else dict(
                        num_tokens=descriptor.num_tokens,
                        num_reqs=descriptor.num_reqs,
                        physical_num_reqs=descriptor.physical_num_reqs,
                        owner=descriptor.cudagraph_owner,
                        generation=descriptor.runtime_generation,
                    ),
                ),
            )
            if len(ids) <= 64:
                route.update(ids=ids.tolist(), weights=weights.tolist())
            if self.admission is not None:
                admission = self.admission
                retained = admission.counts[layer]
                if (
                    admission.next_layer != layer + 1
                    or admission.tokens is None
                    or not 0 <= admission.tokens <= useful_tokens
                    or retained.shape != counts.shape
                    or np.any(retained > counts)
                ):
                    raise ValueError("stale or invalid expert retention observation")
                # Full-block totals cannot reconstruct request-local tails.
                # Snapshot the actual policy input without another GPU readback.
                route.update(
                    admission_histogram=retained.tolist(),
                    admission_useful_tokens=admission.tokens,
                )
            return route
        except Exception as error:
            self._disable_trace(error)
            return {}

    def _trace_last_step(self):
        if self.trace_path and not self.dummy and self.trace_steps < self.trace_limit:
            if self.last_step["layer"] == 0:
                self.trace_layers.clear()
            self.trace_layers.append(dict(self.last_step))
            if self.last_step["layer"] == self.bank.source.layers - 1:
                try:
                    row = dict(
                        schema="native-expert-step-v3",
                        rank=self.bank.source.rank,
                        step=self.trace_steps,
                        unix=time.time(),
                        layers=self.trace_layers,
                        residency_before=self.trace_before,
                        residency_after=cache_snapshot(self.bank),
                        identity=getattr(self, "execution_identity", None),
                    )
                    payload = json.dumps(row) + "\n"
                    size = len(payload.encode())
                    if self.trace_bytes + size > self.trace_byte_limit:
                        raise ValueError("expert trace byte budget exhausted")
                    with open(
                        f"{self.trace_path}-rank{self.bank.source.rank}.jsonl", "a"
                    ) as handle:
                        handle.write(payload)
                    self.trace_bytes += size
                except Exception as error:
                    self._disable_trace(error)
                self.trace_steps += 1
                self.trace_layers.clear()
                self.trace_before = None
