# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded compact native expert execution at a breakable Graph seam."""

from __future__ import annotations

import time

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr
from vllm.utils.torch_utils import direct_register_custom_op

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
) -> None:
    owner = get_forward_context().no_compile_layers[layer_name]
    owner.provider.run(owner.layer_id, hidden, weights, ids, output)


def _native_experts_fake(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
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
    bank; its transaction must call retire before changing physical mappings.
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
        self.use_graphs = True
        self.dummy = False
        self.active = False
        self.forward_count = 0
        self.last_step = {}

    def prepare_execution(self, *, dummy):
        if self.active or self.bank.leases or self.bank.state != "READY":
            raise RuntimeError("cannot prepare an unavailable expert provider")
        self.dummy = bool(dummy)

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
            output.zero_()
        finally:
            self.bank.release(lease, self.bank.fence())

    def retire(self):
        if self.active or self.bank.leases:
            raise RuntimeError("cannot retire an active expert provider")
        self.bank.fence().synchronize()
        for _, graph in self.kernels.values():
            graph.reset()
            self.bank.graphs.discard(graph)
        self.kernels.clear()
        self.consumer = None
        self.consumer_rows = None

    def _kernel(self, bucket):
        if bucket in self.kernels:
            if (
                self.kernels[bucket][1] not in self.bank.graphs
                or self.consumer_rows != self.bank.rows
            ):
                raise RuntimeError("native provider graphs require retirement")
            return self.kernels[bucket]
        if self.consumer is None:
            self.consumer = NativeMappedBankConsumer(self.bank, self.group_rows)
            self.consumer_rows = self.bank.rows
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
            with torch.cuda.graph(graph):
                compiled()
            self.bank.register_graph(graph)

        def direct():
            with set_forward_context(None, self.config, num_tokens=bucket):
                forward()

        self.kernels[bucket] = direct, graph
        return direct, graph

    def run(self, layer, hidden, weights, ids, output):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("native expert demand requires breakable Graph capture")
        if self.active:
            raise RuntimeError("native provider does not support concurrent forwards")
        self.active = True
        before_bytes = self.bank.copy_bytes
        before_read = self.bank.source.read_bytes
        started = time.perf_counter()
        steps, tiles, useful = 0, 0, 0
        try:
            error = None
            cpu_ids = cpu_weights = None
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
                if not self.dummy:
                    cpu_ids = ids.cpu().numpy()
                    cpu_weights = weights.cpu().numpy()
                    if not np.isfinite(cpu_weights).all() or np.any(cpu_weights < 0):
                        raise ValueError("invalid native routing weights")
                    # Validate before consensus; actual per-chunk schedules follow.
                    plan(cpu_ids, self.bank.source.experts, self.bank.staging)
            except Exception as exc:
                error = exc
            if self.dummy and error is None:
                self._profile(layer, hidden, output)
                return
            self.coordinator.admit_routes(
                layer, cpu_ids, cpu_weights, error=error, dummy=self.dummy
            )
            assert cpu_ids is not None
            if hidden.shape[0] == 0:
                output.zero_()
                return
            for start in range(0, hidden.shape[0], self.max_tokens):
                count = min(self.max_tokens, hidden.shape[0] - start)
                self.x[:count].copy_(hidden[start : start + count])
                self.weights[:count].copy_(weights[start : start + count])
                waves = plan(
                    cpu_ids[start : start + count],
                    self.bank.source.experts,
                    self.bank.staging,
                )
                for wave in waves:
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
            self.forward_count += 1
        except Exception:
            self.bank.state = "POISONED"
            raise
        finally:
            self.active = False
            self.last_step = dict(
                layer=layer,
                waves=steps,
                tiles=tiles,
                useful_lanes=useful,
                h2d_expert_bytes=self.bank.copy_bytes - before_bytes,
                source_read_bytes=self.bank.source.read_bytes - before_read,
                wall_ms=(time.perf_counter() - started) * 1000,
                state=self.bank.state,
                dummy=self.dummy,
            )
