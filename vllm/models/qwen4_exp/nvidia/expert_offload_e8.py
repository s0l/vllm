# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-expert E8 data plane for FlashNext routed experts."""

from __future__ import annotations

import json
import os
import time

import torch

from vllm.utils.e8_expert_config import E8ArchiveConfig

from .expert_offload_e8_archive import E8HostArchive
from .expert_offload_e8_demand import E8DemandLoader
from .expert_offload_e8_kernels import (
    build_tasks,
    compact_routes,
    masked_e8_activation,
    masked_e8_scatter,
    masked_e8_scatter_deterministic,
    masked_hadamard,
    persistent_grouped_e8_mm,
)


class E8ExpertExecutor:
    """Persistent resident bank plus a two-slot cold layer pipeline."""

    def __init__(
        self,
        config: E8ArchiveConfig,
        rank: int,
        device: torch.device,
        *,
        topk: int,
        hidden: int,
    ):
        if topk != 10 or hidden != 2560:
            raise ValueError("unsupported E8 model geometry")
        self.config, self.device = config, device
        self.archive = E8HostArchive(config, rank)
        self.rank, self.topk, self.hidden = rank, topk, hidden
        self.owner, self.resident, self.cold = (
            self.archive.owner,
            self.archive.resident,
            self.archive.cold,
        )
        self.logical_to_physical = (
            None
            if self.archive.logical_to_physical is None
            else self.archive.logical_to_physical.to(device=device)
        )
        self.max_tokens = config.max_tokens
        self.max_lanes = self.max_tokens * topk
        self.scratch_lanes = self._bucket(self.max_lanes)
        self.max_tasks = (self.max_lanes + 7) // 8 + self.owner
        self.copy_stream = torch.cuda.Stream(device=device)
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.consumed = [torch.cuda.Event() for _ in range(2)]
        for event in self.consumed:
            event.record(torch.cuda.current_stream(device))

        host = self.archive.host
        self.q_resident = {
            name: host[name][:, : self.resident].to(device=device, non_blocking=False)
            for name in ("q_gu", "q_down")
        }
        self.q_cold = {
            name: [
                torch.empty_like(host[name][0, self.resident :], device=device)
                for _ in range(2)
            ]
            for name in ("q_gu", "q_down")
        }
        metadata_names = ("su_gu", "sv_gate", "sv_up", "su_down", "sv_down")
        self.metadata = [
            {
                name: torch.empty_like(host[name][0], device=device)
                for name in metadata_names
            }
            for _ in range(2)
        ]
        self.codebook = self.archive.codebook.to(device=device, non_blocking=False)
        self.hadamard20 = self.archive.hadamard20.to(device=device, non_blocking=False)

        self.counts = torch.empty(self.owner, dtype=torch.int32, device=device)
        self.cold_slots = torch.empty(self.owner, dtype=torch.int32, device=device)
        self.cold_slot_count = torch.empty(1, dtype=torch.int32, device=device)
        self.offsets = torch.empty(self.owner + 1, dtype=torch.int32, device=device)
        self.cursor = torch.empty(self.owner, dtype=torch.int32, device=device)
        self.tokens = torch.empty(self.scratch_lanes, dtype=torch.int32, device=device)
        self.route_weights = torch.empty(
            self.scratch_lanes, dtype=torch.float32, device=device
        )
        self.route_experts = torch.empty(
            self.scratch_lanes, dtype=torch.int32, device=device
        )
        self.task_counter = torch.empty(1, dtype=torch.int32, device=device)
        self.task_expert = torch.empty(self.max_tasks, dtype=torch.int32, device=device)
        self.task_begin = torch.empty(self.max_tasks, dtype=torch.int32, device=device)
        self.task_rows = torch.empty(self.max_tasks, dtype=torch.int32, device=device)
        self.x = torch.empty(
            (self.scratch_lanes, 2560), dtype=torch.float16, device=device
        )
        self.gu = torch.empty(
            (self.scratch_lanes, 1280), dtype=torch.float16, device=device
        )
        self.middle = torch.empty(
            (self.scratch_lanes, 640), dtype=torch.float16, device=device
        )
        self.down = torch.empty(
            (self.scratch_lanes, 2560), dtype=torch.float16, device=device
        )
        self.partial = torch.empty(
            (self.max_tokens, 2560), dtype=torch.float32, device=device
        )
        self.transform_tmp_2560 = torch.empty_like(self.down)
        self.transform_tmp_640 = torch.empty_like(self.middle)
        self.gate_rows = torch.empty_like(self.middle)
        self.up_rows = torch.empty_like(self.middle)
        self.middle_pre = torch.empty_like(self.middle)
        self.persistent_workers = 128
        self.expected_layer = 0
        self.prepared = False
        self.dummy = False
        self.warmed = False
        self.last_step: dict[str, object] = {}
        self.trace_path = os.environ.get("AG2_VLLM_E8_TRACE", "")
        self.trace_limit = int(os.environ.get("AG2_VLLM_E8_TRACE_STEPS", "4"))
        self.diagnostic_sync = os.environ.get("AG2_VLLM_E8_DIAGNOSTIC_SYNC", "0") == "1"
        if not 0 <= self.trace_limit <= 64:
            raise ValueError("E8 trace step budget must be within 0..64")
        self.trace_steps = 0
        self.trace_active = False
        self.trace_identity = None
        self.active_tokens: int | None = None
        self.physical_tokens: int | None = None
        self.demand_mode = False
        self.demand_loader = (
            E8DemandLoader(config.demand_library, host, self.owner)
            if config.demand_library is not None
            else None
        )
        self.temporal_cache = bool(
            self.demand_loader is not None
            and os.environ.get("AG2_VLLM_E8_TEMPORAL_CACHE", "0") == "1"
        )
        self.deterministic_scatter = bool(
            self.temporal_cache
            and os.environ.get("AG2_VLLM_E8_DETERMINISTIC_SCATTER", "0") == "1"
        )
        if self.temporal_cache:
            self.temporal_metadata = {
                name: host[name].to(device=device, non_blocking=False)
                for name in metadata_names
            }
            initial_slots = torch.full(
                (self.owner,), -1, dtype=torch.int32, device=device
            )
            initial_slots[: self.resident] = torch.arange(
                self.resident, dtype=torch.int32, device=device
            )
            self.temporal_resident_slots = initial_slots.expand(48, -1).clone()
            self.temporal_slot_experts = (
                torch.arange(self.resident, dtype=torch.int32, device=device)
                .expand(48, -1)
                .clone()
            )
            self.temporal_ages = torch.zeros(
                (48, self.resident), dtype=torch.uint64, device=device
            )
            self.temporal_clock = torch.zeros(48, dtype=torch.uint64, device=device)
            self.temporal_cold_experts = torch.empty(
                self.cold, dtype=torch.int32, device=device
            )
            self.temporal_promotion_slots = torch.empty_like(self.temporal_cold_experts)
        self.trace_starts = (
            [torch.cuda.Event(enable_timing=True) for _ in range(48)]
            if self.trace_path
            else []
        )
        self.trace_ends = (
            [torch.cuda.Event(enable_timing=True) for _ in range(48)]
            if self.trace_path
            else []
        )
        if self.trace_path:
            with open(f"{self.trace_path}-rank{rank}.jsonl", "a"):
                pass

    def _diagnostic_marker(self, layer: int, phase: str, *, sync: bool = True) -> None:
        if not (self.trace_active and self.diagnostic_sync):
            return
        if sync:
            torch.cuda.current_stream(self.device).synchronize()
        row = {
            "schema": "flashnext-e8-diagnostic-marker-v1",
            "rank": self.rank,
            "step": self.trace_steps,
            "layer": layer,
            "phase": phase,
            "unix": time.time(),
            "identity": self.trace_identity,
        }
        with open(f"{self.trace_path}-rank{self.rank}.markers.jsonl", "a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    def _stage(self, layer: int, *, demand: bool = False) -> None:
        if self.temporal_cache and demand:
            return
        slot = layer & 1
        host = self.archive.host
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.consumed[slot])
            if not demand:
                for name in ("q_gu", "q_down"):
                    self.q_cold[name][slot].copy_(
                        host[name][layer, self.resident :], non_blocking=True
                    )
            for name, destination in self.metadata[slot].items():
                destination.copy_(host[name][layer], non_blocking=True)
            self.ready[slot].record()

    def prepare_execution(self, *, dummy: bool, identity=None, num_tokens=None) -> None:
        if self.prepared:
            raise RuntimeError("E8 execution epoch already active")
        self.expected_layer = 0
        self.dummy = bool(dummy)
        identity_req_ids = identity.get("req_ids", ()) if identity is not None else ()
        synthetic_warmup = bool(identity_req_ids) and all(
            str(req_id).startswith("_warmup_") for req_id in identity_req_ids
        )
        self.trace_active = bool(
            self.trace_path
            and identity is not None
            and not self.dummy
            and not synthetic_warmup
            and self.trace_steps < self.trace_limit
        )
        self.trace_identity = identity if self.trace_active else None
        self.active_tokens = None if identity is None else int(identity["tokens"])
        self.physical_tokens = (
            int(identity["physical_tokens"])
            if identity is not None
            else None
            if num_tokens is None
            else int(num_tokens)
        )
        if self.active_tokens is not None and self.active_tokens < 0:
            raise ValueError("invalid E8 logical token count")
        if self.physical_tokens is not None and self.physical_tokens < 1:
            raise ValueError("invalid E8 physical token count")
        if (
            self.active_tokens is not None
            and self.physical_tokens is not None
            and self.active_tokens > self.physical_tokens
        ):
            raise ValueError("E8 logical token count exceeds physical shape")
        self._diagnostic_marker(-1, "execution_begin", sync=False)
        self.demand_mode = bool(
            self.demand_loader is not None
            and self.physical_tokens is not None
            and self.physical_tokens <= self.config.demand_max_tokens
        )
        if not self.dummy:
            self._stage(0, demand=self.demand_mode)
        self.prepared = True

    def finish_execution(self) -> None:
        if self.prepared and self.expected_layer not in (0, 48):
            raise RuntimeError("incomplete E8 layer traversal")
        if self.trace_active and self.expected_layer == 48:
            self.trace_ends[-1].synchronize()
            expert_ms = [
                start.elapsed_time(end)
                for start, end in zip(self.trace_starts, self.trace_ends)
            ]
            between_ms = [
                self.trace_ends[layer].elapsed_time(self.trace_starts[layer + 1])
                for layer in range(47)
            ]
            row = {
                "schema": "flashnext-e8-device-timeline-v1",
                "rank": self.rank,
                "step": self.trace_steps,
                "unix": time.time(),
                "identity": self.trace_identity,
                "expert_ms": expert_ms,
                "between_ms": between_ms,
                "expert_sum_ms": sum(expert_ms),
                "between_sum_ms": sum(between_ms),
                "model_span_ms": self.trace_starts[0].elapsed_time(self.trace_ends[-1]),
            }
            with open(f"{self.trace_path}-rank{self.rank}.jsonl", "a") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.trace_steps += 1
        self.prepared = False
        self.dummy = False
        self.trace_active = False
        self.trace_identity = None
        self.active_tokens = None
        self.physical_tokens = None
        self.demand_mode = False

    def abort_execution(self) -> None:
        """Discard capture-only host state after a failed/incomplete forward."""
        self.prepared = False
        self.dummy = False
        self.trace_active = False
        self.trace_identity = None
        self.active_tokens = None
        self.physical_tokens = None
        self.demand_mode = False
        self.expected_layer = 0

    def warmup(self) -> None:
        """Compile and consume the full layer pipeline before memory profiling."""
        if self.warmed:
            return
        if self.prepared:
            raise RuntimeError("E8 warmup overlaps an execution epoch")
        hidden = torch.full(
            (self.max_tokens, self.hidden),
            1e-3,
            dtype=torch.bfloat16,
            device=self.device,
        )
        weights = torch.full(
            (self.max_tokens, self.topk),
            1.0 / self.topk,
            dtype=torch.float32,
            device=self.device,
        )
        route_pattern = torch.tensor(
            self._warmup_route_pattern(
                self.archive.begin,
                self.archive.end,
                topk=self.topk,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        # fused_topk returns a dense row-major tensor.  An expanded stride-0
        # surrogate compiles a different Triton specialization and leaves the
        # first product request paying for the real route kernels.
        ids = route_pattern.expand(self.max_tokens, -1).contiguous()
        output = torch.empty_like(hidden)
        self.prepare_execution(dummy=False)
        try:
            for layer in range(48):
                self.run(layer, hidden, weights, ids, output)
            torch.cuda.current_stream(self.device).synchronize()
            if not output.isfinite().all().item():
                raise RuntimeError("nonfinite E8 warmup output")
            self.finish_execution()
        except BaseException:
            self.prepared = False
            self.dummy = False
            raise
        finally:
            del hidden, weights, ids, output

        # Synthetic model inputs can collapse onto one owner and therefore do
        # not cover the bucket/row-tile signatures produced by ordinary
        # routing.  Compile the balanced local-lane surface directly on one
        # physical layer; the full 48-layer arm above remains the data-plane
        # correctness and staging control.
        for rows in self._warmup_token_surface(self.max_tokens):
            hidden = torch.full(
                (rows, self.hidden),
                1e-3,
                dtype=torch.bfloat16,
                device=self.device,
            )
            weights = torch.full(
                (rows, self.topk),
                1.0 / self.topk,
                dtype=torch.float32,
                device=self.device,
            )
            ids = route_pattern.expand(rows, -1).contiguous()
            output = torch.empty_like(hidden)
            self.prepare_execution(dummy=False)
            try:
                self._run_chunk(0, hidden, weights, ids, output, 0)
                self.consumed[0].record(torch.cuda.current_stream(self.device))
                torch.cuda.current_stream(self.device).synchronize()
                if not output.isfinite().all().item():
                    raise RuntimeError("nonfinite E8 route-surface warmup output")
            finally:
                self.abort_execution()
                del hidden, weights, ids, output

        # Compile both demand layouts without mutating cache state.  Bulk
        # prefill uses a cold-slot map without a temporal resident map, while
        # short decode supplies both maps.  Triton specializes those boolean
        # contracts independently, so covering only the temporal form leaves
        # the first ordinary prefill paying for compilation after READY.
        # A zero device task count makes every launch a physical no-op.
        if self.temporal_cache:
            self.task_counter.zero_()
            demand_layouts = (
                (self.cold_slots, None),
                (self.cold_slots, self.temporal_resident_slots[0]),
            )
            for block_m in (8, 32, 64, 128):
                for cold_map, resident_map in demand_layouts:
                    persistent_grouped_e8_mm(
                        self.x,
                        self.q_resident["q_gu"],
                        self.q_cold["q_gu"][0],
                        self.codebook,
                        self.gu,
                        self.task_counter,
                        self.task_expert,
                        self.task_begin,
                        self.task_rows,
                        layer=0,
                        resident=self.resident,
                        workers=self.persistent_workers,
                        block_m=block_m,
                        cold_map=cold_map,
                        resident_map=resident_map,
                    )
                    persistent_grouped_e8_mm(
                        self.middle,
                        self.q_resident["q_down"],
                        self.q_cold["q_down"][0],
                        self.codebook,
                        self.down,
                        self.task_counter,
                        self.task_expert,
                        self.task_begin,
                        self.task_rows,
                        layer=0,
                        resident=self.resident,
                        workers=self.persistent_workers,
                        block_m=block_m,
                        cold_map=cold_map,
                        resident_map=resident_map,
                    )
            torch.cuda.current_stream(self.device).synchronize()
        self.warmed = True

    @staticmethod
    def _warmup_route_pattern(
        begin: int,
        end: int,
        *,
        total_experts: int = 512,
        topk: int = 10,
    ) -> tuple[int, ...]:
        """Return a unique route row with topology-proportional local work."""
        if not 0 <= begin < end <= total_experts or not 1 <= topk <= total_experts:
            raise ValueError("invalid E8 warmup route geometry")
        local_count = round(topk * (end - begin) / total_experts)
        local_count = min(topk, max(1, local_count))
        local = list(range(begin, begin + local_count))
        remote = list(range(0, begin)) + list(range(end, total_experts))
        if len(remote) < topk - local_count:
            raise ValueError("insufficient remote E8 warmup routes")
        return tuple(local + remote[: topk - local_count])

    @staticmethod
    def _warmup_token_surface(max_tokens: int) -> tuple[int, ...]:
        """Cover every row-tile boundary and power-of-two route bucket."""
        if max_tokens < 1:
            raise ValueError("invalid E8 warmup token capacity")
        candidates = (1, 2, 4, 8, 16, 31, 32, 64, 127, 128, 256, 511, 512)
        return tuple(sorted({n for n in (*candidates, max_tokens) if n <= max_tokens}))

    @staticmethod
    def _bucket(count: int) -> int:
        return 0 if count == 0 else 1 << (count - 1).bit_length()

    @staticmethod
    def _row_tile(tokens: int) -> int:
        """Reuse decoded E8 weights only when enough routed rows exist."""
        if tokens >= 512:
            return 128
        if tokens >= 128:
            return 64
        return 32 if tokens >= 32 else 8

    def _run_chunk(self, layer, hidden, weights, ids, output, slot):
        self._diagnostic_marker(layer, "chunk_begin", sync=False)
        self._diagnostic_marker(layer, "pre_compact_sync")
        observer = (
            (lambda phase: self._diagnostic_marker(layer, phase))
            if self.trace_active and self.diagnostic_sync
            else None
        )
        compact_routes(
            ids,
            weights,
            begin=self.archive.begin,
            end=self.archive.end,
            counts=self.counts,
            offsets=self.offsets,
            cursor=self.cursor,
            tokens_out=self.tokens,
            weights_out=self.route_weights,
            experts_out=self.route_experts,
            synchronize_count=False,
            observer=observer,
            expert_map=(
                None
                if self.logical_to_physical is None
                else self.logical_to_physical[layer]
            ),
        )
        self._diagnostic_marker(layer, "compact_done")
        bucket = self._bucket(ids.numel())
        slot = layer & 1
        current = torch.cuda.current_stream(self.device)
        if not (self.temporal_cache and self.demand_mode):
            current.wait_event(self.ready[slot])
        self._diagnostic_marker(layer, "stage_ready")
        temporal_active = self.temporal_cache and hidden.shape[0] <= 4
        if self.demand_mode:
            assert self.demand_loader is not None
            if temporal_active:
                self.demand_loader.temporal_gather(
                    layer=layer,
                    counts=self.counts,
                    resident_slots=self.temporal_resident_slots[layer],
                    cold_slots=self.cold_slots,
                    cold_experts=self.temporal_cold_experts,
                    cold_count=self.cold_slot_count,
                    slot_experts=self.temporal_slot_experts[layer],
                    ages=self.temporal_ages[layer],
                    clock=self.temporal_clock[layer : layer + 1],
                    promotion_slots=self.temporal_promotion_slots,
                    resident_bank=self.q_resident,
                )
            else:
                self.demand_loader.gather(
                    layer=layer,
                    resident=self.resident,
                    counts=self.counts,
                    slots=self.cold_slots,
                    slot_count=self.cold_slot_count,
                    destination=self.q_cold,
                    slot=slot,
                )
            self._diagnostic_marker(layer, "demand_ready")
        self.partial[: hidden.shape[0]].zero_()
        if bucket:
            row_tile = self._row_tile(hidden.shape[0])
            task_cap = min(bucket, self.max_tasks)
            build_tasks(
                self.counts,
                self.offsets,
                self.task_counter,
                self.task_expert[:task_cap],
                self.task_begin[:task_cap],
                self.task_rows[:task_cap],
                # Keep arbitrary tails on a finite signature surface while
                # counts retain the exact useful work.
                max_tokens=self._bucket(hidden.shape[0]),
                rows=row_tile,
            )
            self._diagnostic_marker(layer, "tasks_done")
            expert = self.route_experts[:bucket]
            token = self.tokens[:bucket]
            meta = (
                {name: value[layer] for name, value in self.temporal_metadata.items()}
                if temporal_active
                else self.metadata[slot]
            )
            lane_count = self.offsets[self.owner :]
            masked_hadamard(
                hidden,
                self.hadamard20,
                self.x,
                self.transform_tmp_2560,
                lane_count,
                max_lanes=bucket,
                transpose=True,
                scale=meta["su_gu"],
                row_index=token,
                expert_index=expert,
            )
            self._diagnostic_marker(layer, "input_transform_done")
            self.gu[:bucket].zero_()
            persistent_grouped_e8_mm(
                self.x[:bucket],
                self.q_resident["q_gu"],
                self.q_cold["q_gu"][slot],
                self.codebook,
                self.gu[:bucket],
                self.task_counter,
                self.task_expert[:task_cap],
                self.task_begin[:task_cap],
                self.task_rows[:task_cap],
                layer=layer,
                resident=self.resident,
                workers=self.persistent_workers,
                block_m=row_tile,
                cold_map=self.cold_slots if self.demand_mode else None,
                resident_map=(
                    self.temporal_resident_slots[layer]
                    if self.demand_mode and temporal_active
                    else None
                ),
            )
            self._diagnostic_marker(layer, "gu_mm_done")
            masked_hadamard(
                self.gu[:, :640],
                self.hadamard20,
                self.gate_rows,
                self.transform_tmp_640,
                lane_count,
                max_lanes=bucket,
            )
            self._diagnostic_marker(layer, "gate_transform_done")
            masked_hadamard(
                self.gu[:, 640:],
                self.hadamard20,
                self.up_rows,
                self.transform_tmp_640,
                lane_count,
                max_lanes=bucket,
            )
            self._diagnostic_marker(layer, "up_transform_done")
            masked_e8_activation(
                self.gate_rows,
                self.up_rows,
                meta["sv_gate"],
                meta["sv_up"],
                meta["su_down"],
                expert,
                self.middle_pre,
                lane_count,
                max_lanes=bucket,
            )
            self._diagnostic_marker(layer, "activation_done")
            masked_hadamard(
                self.middle_pre,
                self.hadamard20,
                self.middle,
                self.transform_tmp_640,
                lane_count,
                max_lanes=bucket,
                transpose=True,
            )
            self._diagnostic_marker(layer, "middle_transform_done")
            self.down[:bucket].zero_()
            persistent_grouped_e8_mm(
                self.middle[:bucket],
                self.q_resident["q_down"],
                self.q_cold["q_down"][slot],
                self.codebook,
                self.down[:bucket],
                self.task_counter,
                self.task_expert[:task_cap],
                self.task_begin[:task_cap],
                self.task_rows[:task_cap],
                layer=layer,
                resident=self.resident,
                workers=self.persistent_workers,
                block_m=row_tile,
                cold_map=self.cold_slots if self.demand_mode else None,
                resident_map=(
                    self.temporal_resident_slots[layer]
                    if self.demand_mode and temporal_active
                    else None
                ),
            )
            self._diagnostic_marker(layer, "down_mm_done")
            masked_hadamard(
                self.down,
                self.hadamard20,
                self.x,
                self.transform_tmp_2560,
                lane_count,
                max_lanes=bucket,
            )
            self._diagnostic_marker(layer, "output_transform_done")
            scatter = (
                masked_e8_scatter_deterministic
                if self.deterministic_scatter and hidden.shape[0] <= 4
                else masked_e8_scatter
            )
            scatter(
                self.x,
                meta["sv_down"],
                token,
                expert,
                self.route_weights,
                self.partial[: hidden.shape[0]],
                lane_count,
                max_lanes=bucket,
            )
            self._diagnostic_marker(layer, "scatter_done")
        output.copy_(self.partial[: hidden.shape[0]])
        self._diagnostic_marker(layer, "chunk_done")
        return None, bucket

    def run(
        self,
        layer: int,
        hidden: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        output: torch.Tensor,
        *,
        is_padding=None,
    ) -> None:
        if not self.prepared or layer != self.expected_layer:
            raise RuntimeError("stale or out-of-order E8 layer")
        if (
            hidden.ndim != 2
            or hidden.shape[1] != 2560
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.device
            or ids.shape != (hidden.shape[0], 10)
            or weights.shape != ids.shape
            or weights.dtype != torch.float32
            or output.shape != hidden.shape
        ):
            raise ValueError("invalid E8 execution inputs")
        active_tokens = (
            hidden.shape[0] if self.active_tokens is None else self.active_tokens
        )
        if active_tokens > hidden.shape[0]:
            raise ValueError("E8 logical token count exceeds physical shape")
        if is_padding is not None and is_padding.shape != (hidden.shape[0],):
            raise ValueError("invalid E8 padding mask")
        started = time.perf_counter()
        if self.trace_active:
            self.trace_starts[layer].record(torch.cuda.current_stream(self.device))
        if layer + 1 < 48:
            self._stage(layer + 1, demand=self.demand_mode)
        slot = layer & 1
        buckets = []
        for start in range(0, active_tokens, self.max_tokens):
            end = min(active_tokens, start + self.max_tokens)
            _, bucket = self._run_chunk(
                layer,
                hidden[start:end],
                weights[start:end],
                ids[start:end],
                output[start:end],
                slot,
            )
            buckets.append(bucket)
        if active_tokens < hidden.shape[0]:
            output[active_tokens:].zero_()
        self.consumed[slot].record(torch.cuda.current_stream(self.device))
        if self.trace_active:
            self.trace_ends[layer].record(torch.cuda.current_stream(self.device))
        self.expected_layer += 1
        self.last_step = {
            "layer": layer,
            "tokens": active_tokens,
            "physical_tokens": hidden.shape[0],
            "chunks": len(buckets),
            "local_lanes": None,
            "buckets": buckets,
            "resident_experts": self.resident,
            "cold_experts": self.cold,
            "implementation": "device_local_persistent",
            "transfer_mode": "demand_uva" if self.demand_mode else "bulk_copy",
            "wall_ms": (time.perf_counter() - started) * 1000,
        }

    def close(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
