# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coalesce a fully resident small demand without weakening GPU ownership."""

from hashlib import sha256

import numpy as np
import torch

from vllm.forward_context import set_forward_context
from vllm.triton_utils import tl
from vllm.triton_utils import triton as tr

from .expert_offload_bank import NativeBankPlan, NativeMappedBankConsumer


def resident_hint(bank, layer, waves):
    selected = np.asarray([e for wave in waves for e in wave.experts], dtype=np.int64)
    present, rows, invalid = [], [], False
    for expert in selected:
        key = (layer, int(expert))
        found = key in bank.host_hot
        row = bank.host_hot.get(key, -1)
        valid = type(row) in (int, np.int32, np.int64) and (
            0 <= int(row) < bank.tables.pool_rows
        )
        invalid |= found and not valid
        present.append(found)
        rows.append(int(row) if valid else -1)
    return (
        selected,
        selected + layer * bank.source.experts,
        np.asarray(rows, dtype=np.int32),
        -1 if invalid else int(all(present)),
    )


@tr.jit(do_not_specialize=["N", "POOL"])
def _validate_hot(
    HOT,
    OWNER,
    ERROR,
    KEYS,
    ROWS,
    STATUS,
    N,
    POOL,
    B: tl.constexpr,
):
    i = tl.arange(0, B)
    active = i < N
    key = tl.load(KEYS + i, active, 0)
    row = tl.load(ROWS + i, active, 0)
    bounded = (row >= 0) & (row < POOL)
    observed = tl.load(HOT + key, active, -1)
    owner = tl.load(OWNER + tl.where(bounded, row, 0), active & bounded, -1)
    valid = tl.min(
        tl.where(active, bounded & (observed == row) & (owner == key), True).to(
            tl.int32
        ),
        0,
    )
    valid = (valid != 0) & (tl.load(ERROR) == 0)
    tl.store(STATUS, tl.where(valid, 1, -1))


@tr.jit(do_not_specialize=["N", "WAVES"])
def _touch_hot(
    ROWS,
    USE,
    CLOCK,
    FORWARDS,
    GATE,
    N,
    WAVES,
    WIDTH: tl.constexpr,
    FIRST: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.arange(0, B)
    row = tl.load(ROWS + i, i < N, 0)
    if tl.load(GATE) != 0:
        clock = tl.load(CLOCK)
        tl.store(USE + row, clock + i // WIDTH + 1, i < N)
        tl.store(CLOCK, clock + WAVES)
        if FIRST:
            tl.store(FORWARDS, tl.load(FORWARDS) + WAVES)


class NativeHotRead:
    def __init__(self, provider):
        self.provider, self.bank = provider, provider.bank
        bank = self.bank
        experts = bank.source.experts
        self.keys = torch.empty(experts, dtype=torch.int64, device=bank.device)
        self.rows = torch.empty(experts, dtype=torch.int32, device=bank.device)
        self.ids = torch.empty(
            (64, provider.topk), dtype=torch.int32, device=bank.device
        )
        self.output = torch.empty(
            (64, provider.hidden), dtype=torch.bfloat16, device=bank.device
        )
        self.host_keys = torch.empty_like(self.keys, device="cpu", pin_memory=True)
        self.host_rows = torch.empty_like(self.rows, device="cpu", pin_memory=True)
        self.host_ids = torch.empty_like(self.ids, device="cpu", pin_memory=True)
        self.calls = self.fallbacks = 0

    def profile(self):
        """Fixed geometry family under the caller's explicit dummy bank lease."""
        if not self.provider.dummy or not self.bank.leases:
            raise RuntimeError("resident profiling requires a dummy read lease")
        if ("hot", 64, self.bank.source.experts) in self.provider.kernels:
            return
        self.rows.zero_()
        self.ids.zero_()
        for count in (1, 2, 4, 8, 16, 32, 64):
            self._kernel(count, self.bank.source.experts)

    def _kernel(self, count, groups):
        provider, bank = self.provider, self.bank
        key = ("hot", count, groups)
        if key in provider.kernels:
            if provider.kernels[key][1] not in bank.stable_graphs:
                raise RuntimeError("resident native graph was retired")
            return provider.kernels[key]
        consumer = NativeMappedBankConsumer(bank, self.rows[:groups], num_groups=groups)

        def forward():
            self.output[:count].copy_(
                consumer(provider.x[:count], provider.weights[:count], self.ids[:count])
            )

        with set_forward_context(None, provider.config, num_tokens=count):
            compiled = torch.compile(forward, fullgraph=True)
            for _ in range(3):
                compiled()
            bank.fence().synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=provider.graph_pool):
                compiled()
            bank.register_graph(graph, stable_views=True)
            provider.graph_captures += 1

        def direct():
            with set_forward_context(None, provider.config, num_tokens=count):
                forward()

        provider.kernels[key] = direct, graph
        return direct, graph

    def propose(self, layer, waves):
        """Prepare a bounded proposal for the mandatory route consensus."""
        bank = self.bank
        self.selected, self.cpu_keys, self.cpu_rows, status = resident_hint(
            bank, layer, waves
        )
        self.generation, self.layer = bank.generation, layer
        digest = sha256(self.cpu_keys.tobytes() + self.cpu_rows.tobytes()).digest()
        self.vote = [status, bank.rows] + list(digest)
        if status == 1:
            n = len(self.selected)
            self.host_keys[:n].copy_(torch.from_numpy(self.cpu_keys))
            self.host_rows.zero_()
            self.host_rows[:n].copy_(torch.from_numpy(self.cpu_rows))
            self.keys[:n].copy_(self.host_keys[:n], non_blocking=True)
            self.rows.copy_(self.host_rows, non_blocking=True)

    def validate(self, status):
        if self.vote[0] == 1:
            bank = self.bank
            _validate_hot[(1,)](
                bank.tables.hot_phys,
                bank.tables.row_key,
                bank.tables.error,
                self.keys,
                self.rows,
                status,
                len(self.selected),
                bank.tables.pool_rows,
                tr.next_power_of_2(bank.source.experts),
            )

    def run(self, layer, waves, cpu_ids):
        """Consume an ownership proposal admitted by the shared route vote."""
        bank, provider = self.bank, self.provider
        count = cpu_ids.shape[0]
        with bank.lock:
            if (
                bank.state != "READY"
                or bank.leases
                or (self.generation, self.layer) != (bank.generation, layer)
                or not 0 < count <= 64
                or not waves
            ):
                raise RuntimeError("resident native bank unavailable")
            selected, rows = self.selected, self.cpu_rows
            # A single lease protects all already-published weights. Preserve
            # the old planner's exact recency/forward/generation progression.
            bank.generation += len(waves)
            bank.plan = ticket = NativeBankPlan(
                bank.generation, layer, tuple(int(r) for r in rows), ()
            )
            bank.completed.clear()
            lease = bank.acquire(ticket)
            try:
                n = len(selected)
                _touch_hot[(1,)](
                    self.rows,
                    bank.tables.row_use,
                    bank.tables.clock,
                    bank.tables.forwards,
                    bank.tables.gate,
                    n,
                    len(waves),
                    bank.staging,
                    layer == 0,
                    tr.next_power_of_2(bank.source.experts),
                )
                local_ids = np.searchsorted(selected, cpu_ids).astype(np.int32)
                # Ignored padded lanes may contain arbitrary expert sentinels.
                np.clip(local_ids, 0, len(selected) - 1, out=local_ids)
                bucket = tr.next_power_of_2(count)
                # Seven token buckets, one fixed expert geometry. Expert IDs or
                # their distinct count must never create a calibration axis.
                provider.x[count:bucket].zero_()
                provider.weights[count:bucket].zero_()
                self.host_ids[:bucket].zero_()
                self.host_ids[:count].copy_(torch.from_numpy(local_ids))
                self.ids[:bucket].copy_(self.host_ids[:bucket], non_blocking=True)
                direct, graph = self._kernel(bucket, bank.source.experts)
                if provider.use_graphs:
                    graph.replay()
                else:
                    direct()
                self.calls += 1
                return True
            finally:
                bank.release(lease, bank.fence())
