# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture-safe short-step experts with one host admission per target forward."""

import hashlib
import time

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from .expert_offload_admission import cache_snapshot, stream_trace_layer
from .expert_offload_bank import NativeMappedBankConsumer


def _stream_experts(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    is_padding: torch.Tensor | None = None,
) -> None:
    owner = get_forward_context().no_compile_layers[layer_name]
    owner.provider.stream_path.run(
        owner.layer_id, hidden, weights, ids, output, is_padding=is_padding
    )


def _stream_fake(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    is_padding: torch.Tensor | None = None,
) -> None:
    pass


direct_register_custom_op(
    "flashnext_stream_experts",
    _stream_experts,
    mutates_args=["output"],
    fake_impl=_stream_fake,
)


class NativeStreamExperts:
    """Stable buffers/tasks outlive every graph that captures their addresses.

    Only ModelState preparation and target completion mutate host residency.
    During replay native CUDA host nodes dispatch work to the persistent CPU
    team; no Python callback, route readback or collective is added per layer.
    """

    def __init__(self, provider, cpu, *, max_m=None):
        self.provider, self.bank, self.cpu = provider, provider.bank, cpu
        self.max_m = cpu.max_m if max_m is None else max_m
        if type(self.max_m) is not int or not 1 <= self.max_m <= cpu.max_m:
            raise ValueError("captured stream domain exceeds CPU workspace")
        self.layers, self.experts = cpu.layers, cpu.experts
        self.shared = {}
        self.enabled = torch.zeros((), dtype=torch.bool, device=self.bank.device)
        self.mapping = self.bank.tables.hot_phys.view(self.layers, self.experts)
        with torch.device("cpu"):
            self.hx = torch.zeros(
                (self.layers, self.max_m, cpu.h), dtype=torch.bfloat16, pin_memory=True
            )
            self.hi = torch.full(
                (self.layers, self.max_m, cpu.topk),
                -1,
                dtype=torch.int32,
                pin_memory=True,
            )
            self.routes = torch.empty_like(self.hi, pin_memory=True)
            self.hw = torch.zeros(self.hi.shape, dtype=torch.float32, pin_memory=True)
            self.ho = torch.zeros(self.hx.shape, dtype=torch.float32, pin_memory=True)
            self.epochs = torch.full(
                (self.layers, 1), cpu.epoch, dtype=torch.uint64, pin_memory=True
            )
            self.statuses = torch.full(
                (self.layers, 1), -1, dtype=torch.int32, pin_memory=True
            )
        self.cold_output = torch.empty_like(self.ho, device=self.bank.device)
        self.tasks = {
            (layer, m): cpu.task(
                layer,
                self.hx[layer, :m].view(torch.uint16).numpy(),
                self.hi[layer, :m].numpy(),
                self.hw[layer, :m].numpy(),
                self.ho[layer, :m].numpy(),
                self.epochs[layer].numpy(),
                self.statuses[layer].numpy(),
            )
            for layer in range(self.layers)
            for m in range(1, self.max_m + 1)
        }
        self.stream = torch.cuda.Stream(device=self.bank.device)
        self.lease, self.m, self.identity = None, 0, None
        self.started = 0.0
        self.before_source = self.before_cache = None
        self.compiled_gpu = torch.compile(self.gpu, fullgraph=True)
        self.compiled_join = torch.compile(
            self.join, fullgraph=True, options={"emulate_precision_casts": True}
        )

    def prepare(self, *, dummy, num_tokens, identity):
        # A preceding dummy/capture can be owned by the graph manager rather
        # than execute_model. Drain it without committing a policy observation.
        self.abort_step()
        self.m = (
            num_tokens if num_tokens is not None and 0 < num_tokens <= self.max_m else 0
        )
        self.identity = identity
        self.enabled.fill_(not dummy)
        if not self.m:
            self.provider.admission.quiesce()
            return
        if self.bank.state != "READY":
            raise RuntimeError("stream experts require a published bank")
        if self.provider.target_cpu_only and self.bank.tables.pool_rows != 0:
            raise RuntimeError("target CPU-only has a nonzero GPU expert grant")
        self.statuses.fill_(-1)
        self.lease = object()
        self.bank.leases.add(self.lease)
        self.before_source = self.cpu.source_stats()
        self.before_cache = (
            cache_snapshot(self.bank) if self.provider.trace_path else None
        )
        self.started = time.perf_counter()

    def retain_graph(self, graph):
        self.bank.register_graph(graph, stable_views=True)
        self.cpu.retain_graph(graph)

    def gpu(self, hidden, weights, ids, table):
        if self.provider.target_cpu_only:
            return torch.zeros_like(hidden)
        valid = (ids >= 0) & (ids < self.experts)
        hot = valid & (table[ids.clamp(0, self.experts - 1).long()] >= 0)
        keys = torch.where(hot, ids, -1).int()
        return NativeMappedBankConsumer(
            self.bank, table.clamp_min(0), num_groups=self.experts
        )(hidden, weights, keys)

    @staticmethod
    def join(gpu, cpu, shared):
        return (gpu.float() + cpu).bfloat16() + shared

    def run(self, layer, hidden, weights, ids, output, *, is_padding=None):
        m = hidden.shape[0]
        if (
            not 0 < m <= self.max_m
            or hidden.shape != (m, self.cpu.h)
            or ids.shape != (m, self.cpu.topk)
            or weights.shape != ids.shape
            or layer not in self.shared
            or self.lease is None
            or self.bank.state != "READY"
        ):
            raise RuntimeError("unprepared stream expert invocation")
        capture = BreakableCUDAGraphCapture.current()
        if capture is not None and capture._capturing:
            self.retain_graph(capture._current_graph)
        current = torch.cuda.current_stream(self.bank.device)
        active = self.enabled.expand(m)
        if is_padding is not None:
            active = active & ~is_padding
        route = torch.where(active[:, None], ids, -1).int()
        weights = torch.where(active[:, None], weights, 0).float()
        table = self.mapping[layer]
        valid = (route >= 0) & (route < self.experts)
        hot = valid & (table[route.clamp(0, self.experts - 1).long()] >= 0)
        cold = torch.where(hot, -1, route).int()
        task = self.tasks[layer, m]
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            self.hx[layer, :m].copy_(hidden, non_blocking=True)
            self.hi[layer, :m].copy_(cold, non_blocking=True)
            self.routes[layer, :m].copy_(route, non_blocking=True)
            self.hw[layer, :m].copy_(weights, non_blocking=True)
            task.enqueue(self.stream)
        gpu = self.compiled_gpu(hidden, weights, route, table)
        shared = self.shared[layer](hidden)
        with torch.cuda.stream(self.stream):
            task.wait_on(self.stream)
            self.cold_output[layer, :m].copy_(self.ho[layer, :m], non_blocking=True)
        current.wait_stream(self.stream)
        output.copy_(self.compiled_join(gpu, self.cold_output[layer, :m], shared))

    def abort_step(self):
        if self.lease is not None:
            self.bank.release(self.lease, self.bank.fence())
            self.lease = None
        self.m = 0

    def finish(self, *, dummy):
        if not self.m:
            return
        m, started = self.m, self.started
        self.abort_step()
        statuses = self.statuses.numpy()
        ids, weights = self.routes[:, :m].numpy(), self.hw[:, :m].numpy()
        digest = hashlib.sha256(ids.tobytes() + weights.tobytes()).digest()
        coordinator = self.provider.coordinator
        packet = [int(not np.any(statuses)), self.bank.generation, m] + list(digest)
        packet += [-1] * (coordinator.send.numel() - len(packet))
        coordinator.send.copy_(torch.tensor(packet, dtype=torch.int64))
        torch.distributed.all_gather_single(
            coordinator.recv, coordinator.send, group=coordinator.group.device_group
        )
        peers = coordinator.recv.view(coordinator.ranks, -1).cpu().numpy()
        if not (peers[:, 0] == 1).all() or not (peers == peers[:1]).all():
            self.bank.state = "POISONED"
            raise RuntimeError(
                f"stream expert completion/route mismatch: {statuses.ravel().tolist()}"
            )
        if dummy:
            assert self.before_source is not None
            if (
                self.cpu.source_stats()["read_bytes"]
                != self.before_source["read_bytes"]
            ):
                raise RuntimeError("dummy stream read expert weights")
            return
        admission = self.provider.admission
        for layer in range(self.layers):
            admission.observe_layer(layer, ids[layer], weights[layer])
        after_execution = self.cpu.source_stats()
        execution_ms = (time.perf_counter() - started) * 1000
        admission.finish_step()
        self.provider.last_step = dict(
            execution="stream",
            tokens=m,
            execution_ms=execution_ms,
            wall_ms=(time.perf_counter() - started) * 1000,
            source_before=self.before_source,
            source_after=after_execution,
            admission=dict(admission.last_step),
            identity=self.identity,
            route_sha256=digest.hex(),
            state=self.bank.state,
        )
        if (
            self.provider.trace_path
            and self.provider.trace_steps < self.provider.trace_limit
        ):
            self.provider.trace_before = self.before_cache
            self.provider.trace_layers.clear()
            aggregate = dict(self.provider.last_step)
            for layer in range(self.layers):
                row = stream_trace_layer(
                    layer,
                    ids[layer],
                    weights[layer],
                    self.hi[layer, :m].numpy(),
                    cpu=self.tasks[layer, m].stats(),
                    state=self.bank.state,
                    hot_rows=self.bank.tables.pool_rows,
                    num_experts=self.experts,
                )
                row["model_step"] = aggregate if layer == self.layers - 1 else None
                self.provider.last_step = row
                self.provider._trace_last_step()
            self.provider.last_step = aggregate

    def retire(self):
        self.abort_step()
        self.cpu.quiesce()
        # Resets are idempotent: elastic Graph retirement may already have
        # reset these nodes. Task pointers survive until every node is retired.
        for graph in tuple(self.cpu.graphs):
            self.cpu.retire_graph(graph)
            self.bank.graphs.discard(graph)
            self.bank.stable_graphs.discard(graph)
