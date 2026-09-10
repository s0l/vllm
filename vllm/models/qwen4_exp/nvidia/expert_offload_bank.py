# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete native expert rows in a global, leased CUDA VMM bank.

Placement is provisional until every copied component and every rank publish.
The caller owns common-plan consensus, elastic budget admission and graph
retirement; no method here can infer that a scheduler transition was admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

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
        self.pinned = {
            name: torch.empty(
                (staging, *self.shapes[name]),
                dtype=self.dtypes[name],
                device="cpu",
                pin_memory=True,
            )
            for name in COMPONENTS
        }
        self.pinned_numpy = {
            name: (
                value.numpy() if name in SCALARS else value.view(torch.uint8).numpy()
            )
            for name, value in self.pinned.items()
        }
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
        self.graphs: set[torch.cuda.CUDAGraph] = set()
        self.lock = RLock()
        self.copy_bytes = self.copy_calls = 0
        self.views = self._views(self.rows)
        self._initialize_free_rows(0, self.rows)

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

    def begin(self, layer, ids):
        with self.lock:
            if self.state != "READY" or self.leases:
                raise RuntimeError("native bank unavailable")
            self.state = "LOADING"
            try:
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
                self.state = "POISONED"
                raise

    def copy(self, plan):
        """Complete all local copies and drain pinned readers before returning."""
        with self.lock:
            self._check(plan)
            if self.state != "LOADING" or self.completed:
                raise RuntimeError("native copy unavailable")
            try:
                pairs = sorted(plan.copies, key=lambda pair: pair[1])
                for i, (expert, _) in enumerate(pairs):
                    values = self.source.get(plan.layer, expert)
                    if set(values) != set(COMPONENTS):
                        raise ValueError("incomplete native row")
                    for name in COMPONENTS:
                        value = values[name]
                        if value.shape != self.shapes[name] or value.dtype != (
                            np.dtype("float32")
                            if name in SCALARS
                            else np.dtype("uint8")
                        ):
                            raise ValueError("incompatible native row")
                        self.pinned_numpy[name][i] = value
                start = 0
                while start < len(pairs):
                    end = start + 1
                    while end < len(pairs) and pairs[end][1] == pairs[end - 1][1] + 1:
                        end += 1
                    dst = pairs[start][1]
                    for name in COMPONENTS:
                        self.views[name][dst : dst + end - start].copy_(
                            self.pinned[name][start:end], non_blocking=True
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
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return event

    def register_graph(self, graph):
        if self.state != "READY" or not self.leases:
            raise RuntimeError("graph capture requires a native read lease")
        self.graphs.add(graph)

    def prepare_resize(self, hot_rows):
        """Close consumers before an externally admitted VMM transaction."""
        with self.lock:
            targets = self.targets(hot_rows)
            if self.leases or self.state not in ("READY", "POISONED"):
                raise RuntimeError("native bank maintenance unavailable")
            self.state, self.plan = "RESIZING", None
            event = self.fence()
            event.synchronize()
            for graph in self.graphs:
                graph.reset()
            self.graphs.clear()
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
                gp.resize_tables(self.tables, 0)
                self.tables.clock.zero_()
                self.tables.forwards.zero_()
                self.tables.miss_count.zero_()
            old_hot = self.tables.pool_rows
            evicted = gp.resize_tables(self.tables, hot_rows)
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
    entry. Its views are valid only for the bank geometry at construction.
    """

    def __init__(self, bank, expert_rows):
        if bank.state != "READY" or expert_rows.shape != (bank.staging,):
            raise RuntimeError("invalid mapped native consumer geometry")
        self.bank, self.values, self.expert_rows = bank, bank.views, expert_rows

    def __call__(self, hidden, weights, ids):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            run_cutlass_moe_fp4,
        )

        m, k = hidden.shape
        n = self.bank.source.physical // self.bank.source.tp
        values = self.values
        output = torch.empty_like(hidden)
        workspace13 = torch.empty(
            (m, max(2 * n, k)), device=hidden.device, dtype=hidden.dtype
        )
        workspace2 = torch.empty((m, n), device=hidden.device, dtype=hidden.dtype)
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
            e=self.bank.staging,
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
        width = 8 + len(self.identity) + bank.staging * 4
        self.send = torch.empty(width, dtype=torch.int64, device=bank.device)
        self.recv = torch.empty(
            self.ranks * width, dtype=torch.int64, device=bank.device
        )
        self.copy_vote = torch.empty(1, dtype=torch.int32, device=bank.device)
        self.plan_votes = self.copy_votes = self.hot_steps = 0

    def admit_routes(self, layer, ids, weights, *, error=None, dummy=False):
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
            int(dummy),
        ] + list(digest.digest())
        payload += [-1] * (self.send.numel() - len(payload))
        self.send.copy_(torch.tensor(payload, dtype=torch.int64))
        torch.distributed.all_gather_into_tensor(
            self.recv, self.send, group=self.group.device_group
        )
        peers = self.recv.view(self.ranks, -1)
        common = bool(((peers == peers[:1]).all() & (peers[:, 0] == 1).all()).item())
        if not common:
            self.bank.state = "POISONED"
            raise RuntimeError("rank-inconsistent native routing") from error

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

    def stage(self, layer, ids):
        bank = self.bank
        error, plan = None, None
        try:
            plan = bank.begin(layer, ids)
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
            1,
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
