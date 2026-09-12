# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in mixed HOT GPU / selected CPU execution under the real bank lease."""

import time
from hashlib import sha256

import numpy as np
import torch

from .expert_offload_bank import NativeBankPlan
from .expert_offload_hot import _touch_hot, _validate_hot, resident_hint


class NativeHybridRead:
    def __init__(self, provider, executor):
        bank = provider.bank
        if (
            executor.h != provider.hidden
            or executor.i != bank.source.geometry.local
            or not 0 < executor.max_m <= 64
            or executor.e < executor.max_m * provider.topk
            or executor.weight_bytes != 0
            or executor.scale_workspace_bytes != 0
        ):
            raise ValueError("CPU executor differs from admitted prepared geometry")
        self.provider, self.bank, self.cpu = provider, bank, executor
        self.hot = provider.hot_path
        self.max_tokens = executor.max_m
        self.output = self.hot.output
        shape = (self.max_tokens, provider.hidden)
        self.host_x = torch.empty(
            shape, dtype=torch.bfloat16, device="cpu", pin_memory=True
        )
        self.host_y = torch.empty(
            shape, dtype=torch.float32, device="cpu", pin_memory=True
        )
        self.gpu_y = torch.empty(shape, dtype=torch.float32, device=bank.device)
        self.input_ready = torch.cuda.Event()
        self.cpu_epoch = 0
        self.calls = 0
        self.last_step = {}

    def propose(self, layer, waves):
        selected, keys, rows, status = resident_hint(self.bank, layer, waves)
        present = rows >= 0
        self.selected, self.cold = selected[present], selected[~present]
        self.cpu_rows = rows[present]
        self.generation, self.layer = self.bank.generation, layer
        digest = sha256(keys.tobytes() + rows.tobytes()).digest()
        self.vote = [-1 if status < 0 else 1, self.bank.rows] + list(digest)
        self.all_hot = status == 1
        if status < 0:
            return
        hot, n = self.hot, len(self.selected)
        if self.all_hot:
            hot.selected, hot.cpu_keys, hot.cpu_rows = selected, keys, rows
            hot.generation, hot.layer, hot.vote = self.generation, layer, self.vote
        hot.host_keys[:n].copy_(torch.from_numpy(keys[present]))
        hot.host_rows.zero_()
        hot.host_rows[:n].copy_(torch.from_numpy(self.cpu_rows))
        hot.keys[:n].copy_(hot.host_keys[:n], non_blocking=True)
        hot.rows.copy_(hot.host_rows, non_blocking=True)

    def validate(self, status):
        if self.vote[0] != 1:
            return
        bank = self.bank
        _validate_hot[(1,)](
            bank.tables.hot_phys,
            bank.tables.row_key,
            bank.tables.error,
            self.hot.keys,
            self.hot.rows,
            status,
            len(self.selected),
            bank.tables.pool_rows,
            512,
        )

    def _ready(self, error):
        coordinator, bank = self.provider.coordinator, self.bank
        payload = [int(error is None), bank.generation, self.layer]
        payload += [-1] * (coordinator.send.numel() - len(payload))
        coordinator.send.copy_(torch.tensor(payload, dtype=torch.int64))
        torch.distributed.all_gather_single(
            coordinator.recv, coordinator.send, group=coordinator.group.device_group
        )
        peers = coordinator.recv.view(coordinator.ranks, -1).cpu().numpy()
        if not (peers[:, 0] == 1).all() or not (peers == peers[:1]).all():
            bank.state = "POISONED"
            raise RuntimeError("rank-common CPU expert execution rejected") from error

    def run(self, layer, waves, cpu_ids, cpu_weights):
        bank, provider, hot = self.bank, self.provider, self.hot
        timings = dict(source_ms=0.0, bind_ms=0.0, cpu_ms=0.0, input_wait_ms=0.0)
        if self.all_hot:
            self.last_step = dict(
                hot_keys=len(self.selected),
                cold_keys=0,
                hot_lanes=int((cpu_weights > 0).sum()),
                cold_lanes=0,
                cpu_generation=self.cpu_epoch,
                **timings,
            )
            return hot.run(layer, waves, cpu_ids)
        count = len(cpu_ids)
        with bank.lock:
            if (
                bank.state != "READY"
                or bank.leases
                or (self.generation, self.layer) != (bank.generation, layer)
                or not 0 < count <= self.max_tokens
            ):
                raise RuntimeError("mixed expert bank unavailable")
            bank.generation += len(waves)
            bank.plan = ticket = NativeBankPlan(
                bank.generation, layer, tuple(int(r) for r in self.cpu_rows), ()
            )
            bank.completed.clear()
            lease = bank.acquire(ticket)
            try:
                active = cpu_weights > 0
                hot_mask = np.isin(cpu_ids, self.selected) & active
                cold_mask = np.isin(cpu_ids, self.cold) & active
                if not np.array_equal(hot_mask | cold_mask, active):
                    raise RuntimeError("mixed expert contribution missing")
                if len(self.cold):
                    self.host_x[:count].copy_(provider.x[:count], non_blocking=True)
                    self.input_ready.record()
                # Preserve virtual wave recency for the resident subset.
                offset = 0
                for wave in waves:
                    n = int(np.isin(wave.experts, self.selected).sum())
                    _touch_hot[(1,)](
                        hot.rows[offset:],
                        bank.tables.row_use,
                        bank.tables.clock,
                        bank.tables.forwards,
                        bank.tables.gate,
                        n,
                        1,
                        bank.staging,
                        layer == 0,
                        512,
                    )
                    offset += n
                if len(self.selected):
                    bucket = 1 << (count - 1).bit_length()
                    mapped = np.full(cpu_ids.shape, -1, np.int32)
                    mapped[hot_mask] = np.searchsorted(self.selected, cpu_ids[hot_mask])
                    hot.host_ids[:bucket].fill_(-1)
                    hot.host_ids[:count].copy_(torch.from_numpy(mapped))
                    hot.ids[:bucket].copy_(hot.host_ids[:bucket], non_blocking=True)
                    provider.x[count:bucket].zero_()
                    provider.weights[count:bucket].zero_()
                    direct, graph = hot._kernel(bucket, bank.source.experts)
                    if provider.use_graphs:
                        graph.replay()
                    else:
                        direct()
                else:
                    hot.output[:count].zero_()
                error = None
                rows = None
                if len(self.cold):
                    self.cpu_epoch += 1
                    try:
                        started = time.perf_counter()
                        rows = bank.source.get_many(layer, self.cold.tolist())
                        timings["source_ms"] = (time.perf_counter() - started) * 1000
                        started = time.perf_counter()
                        self.cpu.bind(rows, generation=self.cpu_epoch)
                        timings["bind_ms"] = (time.perf_counter() - started) * 1000
                        started = time.perf_counter()
                        self.input_ready.synchronize()
                        timings["input_wait_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                        mapped = np.full(cpu_ids.shape, -1, np.int32)
                        mapped[cold_mask] = np.searchsorted(
                            self.cold, cpu_ids[cold_mask]
                        )
                        started = time.perf_counter()
                        self.cpu.run(
                            self.host_x[:count].view(torch.uint16).numpy(),
                            mapped,
                            cpu_weights,
                            self.host_y[:count].numpy(),
                            generation=self.cpu_epoch,
                        )
                        timings["cpu_ms"] = (time.perf_counter() - started) * 1000
                    except Exception as exc:
                        error = RuntimeError(f"{type(exc).__name__}: {exc}")
                    finally:
                        if self.cpu.bound:
                            try:
                                self.cpu.release(generation=self.cpu_epoch)
                            except Exception as exc:
                                error = RuntimeError(f"CPU lease release: {exc}")
                        rows = None
                    if error is None:
                        self.gpu_y[:count].copy_(self.host_y[:count], non_blocking=True)
                    self._ready(error)
                    torch.add(
                        hot.output[:count], self.gpu_y[:count], out=self.gpu_y[:count]
                    )
                    hot.output[:count].copy_(self.gpu_y[:count])
                self.calls += 1
                self.last_step = dict(
                    hot_keys=len(self.selected),
                    cold_keys=len(self.cold),
                    hot_lanes=int(hot_mask.sum()),
                    cold_lanes=int(cold_mask.sum()),
                    cpu_generation=self.cpu_epoch,
                    **timings,
                )
            finally:
                bank.release(lease, bank.fence())

    def close(self):
        self.provider.quiesce()
        self.cpu.close()
