# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole CPU jobs overlap bounded GPU waves at the native provider seam.

Opt-in physical executor. The caller supplies an independently admitted policy;
this module never substitutes routes, weights or request identity. CPU token
compaction is a buffer bound, not a whole-batch algorithm threshold.
"""

import numpy as np
import torch

from .expert_offload_plan import Wave


def grouped_jobs(waves):
    jobs = {}
    for wave in waves:
        for slot, expert in enumerate(wave.experts):
            if expert in jobs:
                raise ValueError("duplicate whole expert job")
            jobs[expert] = wave.lanes[wave.slots == slot]
    return jobs


def gpu_waves(jobs, cpu_keys, capacity):
    remaining = [(key, lanes) for key, lanes in jobs.items() if key not in cpu_keys]
    waves = []
    for start in range(0, len(remaining), capacity):
        group = remaining[start : start + capacity]
        lanes = np.concatenate([lanes for _, lanes in group])
        slots = np.concatenate(
            [
                np.full(len(lanes), slot, np.int32)
                for slot, (_, lanes) in enumerate(group)
            ]
        )
        waves.append(
            Wave(
                tuple(key for key, _ in group),
                lanes,
                slots,
                1 << (len(lanes) - 1).bit_length(),
            )
        )
    return waves


class NativeMixedDispatch:
    def __init__(self, provider, cpu, policy):
        if (
            provider.wave_pipeline is None
            or provider.topk != cpu.topk
            or provider.hidden != cpu.h
            or cpu is not provider.bank.source.cpu
            or not callable(policy)
        ):
            raise ValueError(
                "mixed dispatch requires one shared source and CPU/GPU geometry"
            )
        self.provider, self.cpu, self.policy = provider, cpu, policy
        self.max_m = min(cpu.max_m, 64)
        self.stream = torch.cuda.Stream(device=provider.bank.device)
        with torch.device("cpu"):
            self.hx = torch.empty(
                self.max_m, cpu.h, dtype=torch.bfloat16, pin_memory=True
            )
            self.hi = torch.empty(
                self.max_m, cpu.topk, dtype=torch.int32, pin_memory=True
            )
            self.hw = torch.empty_like(self.hi, dtype=torch.float32, pin_memory=True)
            self.ho = torch.empty_like(self.hx, dtype=torch.float32, pin_memory=True)
            self.epoch = torch.full(
                (1,), cpu.epoch, dtype=torch.uint64, pin_memory=True
            )
            self.status = torch.full((1,), -1, dtype=torch.int32, pin_memory=True)
        self.device_output = torch.empty_like(self.ho, device=provider.bank.device)
        self.tasks = {}
        self.last_step = {}
        self.active = False

    def workspace(self):
        return dict(
            host_pinned_bytes=sum(
                t.numel() * t.element_size()
                for t in (self.hx, self.hi, self.hw, self.ho, self.epoch, self.status)
            ),
            gpu_bytes=self.device_output.numel() * self.device_output.element_size(),
        )

    def task(self, layer, m):
        key = layer, m
        if key not in self.tasks:
            self.tasks[key] = self.cpu.task(
                layer,
                self.hx[:m].view(torch.uint16).numpy(),
                self.hi[:m].numpy(),
                self.hw[:m].numpy(),
                self.ho[:m].numpy(),
                self.epoch.numpy(),
                self.status.numpy(),
            )
        return self.tasks[key]

    def run(self, layer, waves, ids, weights, output):
        provider, bank = self.provider, self.provider.bank
        if self.active or bank.leases or bank.state != "READY":
            raise RuntimeError("mixed dispatch requires exclusive placement authority")
        if self.cpu.epoch != int(self.epoch[0]):
            raise RuntimeError("stale mixed CPU source epoch")
        jobs = grouped_jobs(waves)
        hot = {e for e in jobs if (layer, e) in bank.host_hot}
        bitmap = self.cpu.resident_bitmap().reshape(
            bank.source.layers, bank.source.experts
        )
        ready = {e for e in jobs if bitmap[layer, e]}
        if provider.target_cpu_only:
            if hot:
                raise RuntimeError("target CPU-only forbids HOT jobs")
            if len(jobs) + bank.staging + 1 > self.cpu.source_stats()["capacity"]:
                raise RuntimeError("target CPU-only source lease capacity exhausted")
            missing = sorted(set(jobs) - ready)
            if missing:
                # Cold startup still routes entirely through CPU. Load before
                # borrowing rows or submitting work, under placement authority.
                self.cpu.load_rows(layer, missing)
                bitmap = self.cpu.resident_bitmap().reshape(
                    bank.source.layers, bank.source.experts
                )
                ready = {e for e in jobs if bitmap[layer, e]}
            if not set(jobs) <= ready:
                raise RuntimeError("target CPU-only source load did not publish rows")
            decision = dict(cpu=sorted(jobs), status="TARGET_CPU_ONLY")
        else:
            decision = self.policy(layer, jobs, hot, ready, provider.topk)
        selected_values = decision["cpu"]
        if (
            len(set(selected_values)) != len(selected_values)
            or not set(selected_values) <= ready - hot
        ):
            raise ValueError("policy assigned duplicate, HOT or unavailable CPU jobs")
        selected = set(selected_values)
        # Hold all chosen source rows through CPU completion. GPU wave leases
        # and one guaranteed demand reserve still have independent capacity.
        capacity = self.cpu.source_stats()["capacity"]
        if len(selected) + bank.staging + 1 > capacity:
            if provider.target_cpu_only:
                raise RuntimeError("target CPU-only source lease capacity exhausted")
            selected = set()
            decision = dict(decision, fallback="source lease reserve")
        self.last_step = dict(
            decision=decision,
            cpu_experts=sorted(selected),
            cpu_tokens=0,
            cpu_lanes=0,
            cpu_service={},
            source_epoch=self.cpu.epoch,
        )
        if not selected:
            tiles, useful = (
                provider.wave_pipeline.run(layer, waves) if waves else (0, 0)
            )
            output.copy_(provider.lanes[: len(ids)].sum(dim=1))
            return len(waves), tiles, useful
        lanes = np.concatenate([jobs[e] for e in sorted(selected)])
        tokens = np.unique(lanes // provider.topk)
        m = len(tokens)
        if not 0 < m <= self.max_m:
            raise ValueError("whole CPU jobs exceed compact token workspace")
        keep = np.isin(ids[tokens], list(selected))
        self.hi[:m].copy_(
            torch.from_numpy(np.where(keep, ids[tokens], -1).astype(np.int32))
        )
        self.hw[:m].copy_(
            torch.from_numpy(np.where(keep, weights[tokens], 0).astype(np.float32))
        )
        self.status.fill_(-1)
        task = self.task(layer, m)
        gpu = gpu_waves(jobs, selected, provider.wave_pipeline.wave_rows)
        # Captures only use initialized empty descriptors. Prepare them before
        # native work so a compiler error cannot strand a submitted CPU task.
        provider.wave_pipeline.prepare_kernels(
            tile.bucket for wave in gpu for tile in wave.tiles(provider.max_lanes)
        )
        current = torch.cuda.current_stream(bank.device)
        token_ids = torch.as_tensor(tokens, dtype=torch.int64, device=bank.device)
        lane_ids = torch.as_tensor(lanes, dtype=torch.int64, device=bank.device)
        rows, enqueued = [], False
        self.active = True
        try:
            rows = bank.source.borrow_resident([(layer, e) for e in sorted(selected)])
            provider.lanes.view(-1, provider.hidden).index_fill_(0, lane_ids, 0)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                self.hx[:m].copy_(
                    provider.x.index_select(0, token_ids), non_blocking=True
                )
                task.enqueue(self.stream)
                enqueued = True
            tiles, useful = provider.wave_pipeline.run(layer, gpu) if gpu else (0, 0)
            with torch.cuda.stream(self.stream):
                task.wait_on(self.stream)
                self.device_output[:m].copy_(self.ho[:m], non_blocking=True)
            # Check CPU completion before the provider's common semantic vote.
            self.stream.synchronize()
            task.check()
            enqueued = False
            output.copy_(provider.lanes[: len(ids)].sum(dim=1))
            merged = (
                output.index_select(0, token_ids).float() + self.device_output[:m]
            ).bfloat16()
            output.index_copy_(0, token_ids, merged)
            self.last_step.update(
                cpu_tokens=m, cpu_lanes=len(lanes), cpu_service=task.stats()
            )
            return len(gpu), tiles, useful + len(lanes)
        finally:
            if enqueued:
                # A failure still drains the callbacks and native readers.
                # Device faults are handled by the outer process watchdog.
                if task.progress()["active"]:
                    try:
                        task.cancel()
                    except RuntimeError:
                        # Completion can race the best-effort cancel. Inactive
                        # is safe only because we still drain the stream below.
                        if task.progress()["active"]:
                            raise
                task.wait_on(self.stream)
                self.stream.synchronize()
            rows.clear()
            self.active = False

    def retire(self):
        if self.active:
            raise RuntimeError("cannot retire active mixed dispatch")
        self.stream.synchronize()
        for task in self.tasks.values():
            task.close()
        self.tasks.clear()
