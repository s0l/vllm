# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded asynchronous copies into reserved, unpublished expert rows.

The decode graph sees a miss until every rank has finished the upload. Only
the model-boundary thread changes row maps; the copy worker owns immutable
source leases, an explicit stream and separate bounded pinned staging.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import numpy as np
import torch

from .expert_offload_bank import COMPONENTS


class NativeExpertPromotion:
    def __init__(self, provider, *, registered_source=False):
        self.provider, self.bank = provider, provider.bank
        self.coordinator = provider.coordinator
        self.stream = torch.cuda.Stream(device=self.bank.device)
        self.registration = None
        self.pinned, self.pinned_numpy = {}, {}
        if registered_source:
            self.registration = self.bank.source.register_uploads()
        else:
            self.pinned, self.pinned_numpy = self.bank._allocate_pinned()
        self.staging_bytes = sum(value.nbytes for value in self.pinned_numpy.values())
        try:
            self.worker = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="expert-upload"
            )
        except BaseException:
            if self.registration is not None:
                self.registration.close()
            raise
        self.bank.registered_source_region = self.registration
        self.future, self.pending = None, None
        self.sequence = 0
        self.last_step = {}
        self.invalidation_fence = None

    def _vote(self, phase, *, ready=1, error=None):
        bank, coordinator = self.bank, self.coordinator
        digest = sha256(repr((phase, self.sequence, self.pending)).encode()).digest()
        payload = [-1 if error is not None else ready, bank.generation] + list(digest)
        payload += [-1] * (coordinator.send.numel() - len(payload))
        peers = coordinator.exchange_control(payload)
        if (peers[:, 0] < 0).any() or not (peers[:, 1:] == peers[:1, 1:]).all():
            bank.state = "POISONED"
            self.cancel()
            raise RuntimeError(
                "rank-inconsistent asynchronous expert " + phase
            ) from error
        return bool((peers[:, 0] == 1).all())

    def start(self, proposal):
        if self.pending is not None:
            raise RuntimeError("expert upload already pending")
        bank = self.bank
        started = time.perf_counter()
        assignments = self.coordinator.admission_assignments(proposal)
        assigned = time.perf_counter()
        if not assignments:
            self.last_step = dict(promotions=0, pending=False)
            return
        self.sequence += 1
        self.pending = tuple(assignments)
        rows, error = None, None
        try:
            with bank.lock, bank.source.lock:
                if bank.leases or bank.state != "READY":
                    raise RuntimeError(
                        "asynchronous admission requires exclusive placement authority"
                    )
                admitted_bytes = len(assignments) * sum(bank.strides.values())
                if admitted_bytes > self.provider.admission.max_upload_bytes:
                    raise ValueError("expert upload exceeds its admitted byte budget")
                # Borrow all existing cache rows before releasing the lock.
                # CPU/source eviction may drop a cache ref, but not these leases.
                rows = bank.source.borrow_resident(
                    [(layer, expert) for layer, expert, _ in assignments]
                )
                borrowed = time.perf_counter()
                if any(set(row) != set(COMPONENTS) for row in rows):
                    raise ValueError("incomplete upload source bundle")
                slots = [r for _, _, r in assignments]
                keys = [
                    layer * bank.source.experts + expert
                    for layer, expert, _ in assignments
                ]
                old = [bank.host_rows.get(r) for r in slots]
                old_keys = [
                    -1 if key is None else key[0] * bank.source.experts + key[1]
                    for key in old
                ]
                victims = [key for key in old_keys if key >= 0]
                victim_slots = [slot for slot, key in zip(slots, old_keys) if key >= 0]
                observed = (
                    torch.cat(
                        (
                            bank.tables.row_key[slots].long(),
                            bank.tables.hot_phys[keys].long(),
                            bank.tables.error.reshape(-1).long(),
                            bank.tables.gate.reshape(-1).long(),
                            bank.tables.hot_phys[victims].long(),
                        )
                    )
                    .cpu()
                    .tolist()
                )
                expected = old_keys + [-1] * len(keys) + [0, 0] + victim_slots
                if observed != expected or any(
                    key is not None and bank.host_hot.get(key) != slot
                    for key, slot in zip(old, slots)
                ):
                    raise RuntimeError(
                        "upload reservation differs from published ownership"
                    )
                bank.pending_promotions.update(slots)
                bank.generation += 1
                bank.plan = None
                if victims:
                    bank.tables.hot_phys[victims] = -1
                    bank.tables.cold_phys[victims] = torch.tensor(
                        [key % bank.source.experts for key in victims],
                        dtype=torch.int32,
                        device=bank.device,
                    )
                bank.tables.row_key[slots] = -1
                # CPU consensus does not fence device writes. The upload stream
                # explicitly waits for retirement before overwriting any row.
                self.invalidation_fence = bank.fence()
                for key, slot in zip(old, slots):
                    if key is not None:
                        del bank.host_hot[key]
                    bank.host_rows.pop(slot, None)
        except Exception as exc:
            error = exc
        self._vote("reserve", error=error)
        reserved = time.perf_counter()
        try:
            self.future = self.worker.submit(self._copy, rows)
        except Exception as exc:
            error = exc
        self._vote("launch", error=error)
        self.last_step = dict(
            reserve_phases_ms=dict(
                assignment_ms=(assigned - started) * 1000,
                borrow_ms=(borrowed - assigned) * 1000,
                invalidate_and_vote_ms=(reserved - borrowed) * 1000,
                launch_and_vote_ms=(time.perf_counter() - reserved) * 1000,
            ),
            promotions=len(assignments),
            pending=True,
            source_lease_payload_bytes=len(assignments) * sum(bank.strides.values()),
            staging_bytes=self.staging_bytes,
            registered_source_bytes=0
            if self.registration is None
            else self.registration.registered_bytes,
        )

    def _copy(self, rows):
        """No model/table mutations, rank collectives or new weight cache here."""
        from vllm.v1.simple_kv_offload import cuda_mem_ops

        bank, error = self.bank, None
        started, calls, copied = time.perf_counter(), 0, 0
        fill_ms = drain_ms = descriptor_ms = 0.0
        batch = None
        try:
            assert self.pending is not None
            with (
                torch.accelerator.device_index(bank.device.index),
                torch.cuda.stream(self.stream),
            ):
                if self.invalidation_fence is None:
                    raise RuntimeError("upload has no slot retirement fence")
                self.stream.wait_event(self.invalidation_fence)
                if self.registration is not None:
                    from vllm import _custom_ops as ops

                    prepared = time.perf_counter()
                    src, dst, sizes = [], [], []
                    region = self.registration
                    for row, (_, _, slot) in zip(rows, self.pending, strict=True):
                        for name in COMPONENTS:
                            value, target = row[name], bank.consumer_views[name]
                            address, nbytes = value.ctypes.data, value.nbytes
                            if (
                                not value.flags.c_contiguous
                                or value.flags.writeable
                                or nbytes != bank.strides[name]
                                or address < region.address
                                or address + nbytes > region.address + region.nbytes
                            ):
                                raise ValueError(
                                    "upload source outside leased registered layout"
                                )
                            src.append(address)
                            dst.append(target.data_ptr() + slot * bank.strides[name])
                            sizes.append(nbytes)
                    region.ensure(src, sizes)
                    descriptors = [
                        torch.from_numpy(np.asarray(v, np.int64))
                        for v in (src, dst, sizes)
                    ]
                    descriptor_ms = (time.perf_counter() - prepared) * 1000
                    ops.swap_blocks_batch(
                        descriptors[0],
                        descriptors[1],
                        descriptors[2],
                        is_src_access_order_any=True,
                    )
                    calls, copied = 1, sum(sizes)
                else:
                    params = cuda_mem_ops.build_params(
                        self.pinned, bank.consumer_views, self.stream
                    )
                    for start in range(0, len(rows), bank.staging):
                        batch = rows[start : start + bank.staging]
                        positions = list(range(len(batch)))
                        tick = time.perf_counter()
                        bank._fill_pinned(batch, positions, self.pinned_numpy)
                        fill_ms += (time.perf_counter() - tick) * 1000
                        targets = [
                            r for _, _, r in self.pending[start : start + len(batch)]
                        ]
                        cuda_mem_ops.copy_blocks(positions, targets, params)
                        calls += 1
                        copied += len(batch) * sum(bank.strides.values())
                        # Staging cannot be overwritten until its DMA drains.
                        tick = time.perf_counter()
                        self.stream.synchronize()
                        drain_ms += (time.perf_counter() - tick) * 1000
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                tick = time.perf_counter()
                self.stream.synchronize()
                drain_ms += (time.perf_counter() - tick) * 1000
            except Exception as exc:
                error = f"upload completion: {exc}"
            # Don't retain source leases via Future exception tracebacks.
            rows = batch = None
        return dict(
            error=error,
            copies=calls,
            bytes=copied,
            transfer="registered" if self.registration is not None else "staged",
            host_fill_ms=fill_ms,
            descriptor_ms=descriptor_ms,
            drain_ms=drain_ms,
            copy_wall_ms=(time.perf_counter() - started) * 1000,
        )

    def poll(self, *, wait=False):
        if self.pending is None:
            return True
        error: Exception | None = None
        result = None
        if self.future is None:
            error = RuntimeError("upload has no worker")
        elif wait or self.future.done():
            result = self.future.result()
            if result["error"] is not None:
                error = RuntimeError(result["error"])
        if not self._vote("ready", ready=int(result is not None), error=error):
            return False
        assert result is not None
        bank = self.bank
        try:
            with bank.lock:
                slots = [r for _, _, r in self.pending]
                keys = [
                    layer * bank.source.experts + expert
                    for layer, expert, _ in self.pending
                ]
                observed = (
                    torch.cat(
                        (
                            bank.tables.row_key[slots],
                            bank.tables.hot_phys[keys],
                        )
                    )
                    .cpu()
                    .tolist()
                )
                if (
                    bank.state != "READY"
                    or bank.leases
                    or bank.pending_promotions != set(slots)
                    or observed != [-1] * (len(slots) + len(keys))
                ):
                    raise RuntimeError(
                        "upload destination was reused before publication"
                    )
                bank.tables.hot_phys[keys] = torch.tensor(
                    slots, dtype=torch.int32, device=bank.device
                )
                bank.tables.cold_phys[keys] = -1
                bank.tables.row_key[slots] = torch.tensor(
                    keys, dtype=torch.int32, device=bank.device
                )
                bank.tables.row_use[slots] = bank.tables.clock[0]
                bank.tables.miss_count[keys] = 0
                bank.generation += 1
                bank.plan = None
        except Exception as exc:
            error = exc
        self._vote("publish", error=error)
        for layer, expert, row in self.pending:
            bank.host_hot[layer, expert] = row
            bank.host_rows[row] = (layer, expert)
        bank.pending_promotions.clear()
        bank.copy_bytes += result["bytes"]
        bank.copy_calls += result["copies"]
        self.last_step.update(result, pending=False)
        self.future, self.pending = None, None
        return True

    def cancel(self):
        """Drain local DMA before rollback/close; never publish partial rows."""
        if self.future is not None:
            self.future.result()
        if self.pending is not None:
            self.bank.state = "POISONED"
        self.future, self.pending = None, None
        self.bank.pending_promotions.clear()

    def close(self):
        try:
            self.cancel()
        finally:
            self.worker.shutdown(wait=True, cancel_futures=True)
            if self.registration is not None:
                self.registration.close()
                self.bank.registered_source_region = None
