# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler/worker contract for the native FlashNext expert bank."""

from dataclasses import dataclass
from pathlib import Path

from vllm.utils.e8_expert_config import E8ArchiveConfig
from vllm.utils.nvfp4_cpu_experts import CpuExpertConfig
from vllm.utils.nvfp4_expert_dispatch import DispatchConfig
from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry
from vllm.utils.nvfp4_expert_stream import StreamExpertConfig


@dataclass(frozen=True)
class ElasticExpertGrant:
    hot_rows: int
    borrowed_bytes: int

    def __post_init__(self):
        if any(
            type(v) is not int or v < 0 for v in (self.hot_rows, self.borrowed_bytes)
        ):
            raise ValueError("invalid elastic expert grant")


@dataclass(frozen=True)
class NativeExpertBudget:
    geometry: NVFP4ExpertGeometry
    layers: int
    experts: int
    max_hot_rows: int = 8192
    staging: int = 32
    quantum: int = 2 << 20
    ram_cache_bytes: int = 1 << 30
    pin_ram_cache: bool = False
    hot_read: bool = False
    prepared_archive: str | None = None
    cpu_experts: CpuExpertConfig | None = None
    stream_experts: StreamExpertConfig | None = None
    partition: str = "legacy"
    wave_slots: int = 1
    dispatch_profile: DispatchConfig | None = None
    target_cpu_only: bool = False
    e8_archive: E8ArchiveConfig | None = None

    def __post_init__(self):
        self.geometry.validate_cutlass()
        if self.e8_archive is not None and (
            not isinstance(self.e8_archive, E8ArchiveConfig)
            or self.max_hot_rows != 0
            or self.cpu_experts is not None
            or self.stream_experts is not None
            or self.dispatch_profile is not None
            or self.hot_read
            or self.target_cpu_only
            or self.geometry.tp != 3
        ):
            raise ValueError("E8 experts require an exclusive TP3 native backend")
        if type(self.target_cpu_only) is not bool or (
            self.target_cpu_only
            and (
                self.max_hot_rows != 0
                or self.stream_experts is None
                or self.dispatch_profile is None
            )
        ):
            raise ValueError(
                "target CPU-only requires zero HOT and stream/mixed executors"
            )
        if self.dispatch_profile is not None and (
            not isinstance(self.dispatch_profile, DispatchConfig)
            or self.geometry.tp != 3
            or self.partition != "balanced"
            or self.staging != 32
            or self.wave_slots != 2
            or self.stream_experts is None
            or self.stream_experts.cpu.max_m != 64
            or self.stream_experts.capture_max_m != 4
            or not self.stream_experts.ready_pipeline
            or not self.stream_experts.column_jobs
        ):
            raise ValueError("mixed dispatch requires the calibrated physical executor")
        if self.partition not in ("legacy", "balanced") or (
            self.partition == "balanced"
            and self.stream_experts is None
            and self.e8_archive is None
        ):
            raise ValueError("balanced partition requires native stream experts")
        if (
            any(type(v) is not int or v <= 0 for v in (self.layers, self.experts))
            or type(self.max_hot_rows) is not int
            or not 0 <= self.max_hot_rows <= self.layers * self.experts
            or type(self.staging) is not int
            or not 1 <= self.staging <= min(self.experts, 1024)
            or self.staging & (self.staging - 1)
            or type(self.wave_slots) is not int
            or self.wave_slots not in (1, 2)
            or self.staging * self.wave_slots > self.experts
            or type(self.quantum) is not int
            or self.quantum != 2 << 20
            or type(self.ram_cache_bytes) is not int
            or self.ram_cache_bytes < 0
            or type(self.pin_ram_cache) is not bool
            or type(self.hot_read) is not bool
            or (
                self.stream_experts is not None
                and (
                    not isinstance(self.stream_experts, StreamExpertConfig)
                    or self.cpu_experts is not None
                    or not self.hot_read
                    or not self.pin_ram_cache
                    or self.prepared_archive is None
                )
            )
            or (
                self.cpu_experts is not None
                and (
                    not isinstance(self.cpu_experts, CpuExpertConfig)
                    or not self.hot_read
                    or not self.pin_ram_cache
                    or self.prepared_archive is None
                )
            )
            or (
                self.prepared_archive is not None
                and (
                    type(self.prepared_archive) is not str
                    or not Path(self.prepared_archive).is_absolute()
                )
            )
        ):
            raise ValueError("unsupported native expert budget geometry")
        if self.stream_experts is not None:
            for rank in range(self.geometry.tp):
                self.rank_geometry(rank).validate_cutlass()
                if self.rank_ram_bytes(rank) <= 0:
                    raise ValueError("stream source requires a nonempty RAM budget")

    @classmethod
    def from_config(cls, config):
        extra = config.additional_config
        if not isinstance(extra, dict) or "flashnext_native_experts" not in extra:
            return None
        options = extra["flashnext_native_experts"]
        if (
            not isinstance(options, dict)
            or options.keys()
            - {
                "hot_rows",
                "max_hot_rows",
                "staging",
                "ram_cache_bytes",
                "ram_cache_total_bytes",
                "pin_ram_cache",
                "hot_read",
                "prepared_archive",
                "cpu_experts",
                "stream_experts",
                "partition",
                "wave_slots",
                "dispatch_profile",
                "target_cpu_only",
                "e8_archive",
            }
            or any(
                type(v) is not int or v < 0
                for key, v in options.items()
                if key
                not in {
                    "pin_ram_cache",
                    "hot_read",
                    "prepared_archive",
                    "cpu_experts",
                    "stream_experts",
                    "partition",
                    "dispatch_profile",
                    "target_cpu_only",
                    "e8_archive",
                }
            )
            or type(options.get("pin_ram_cache", False)) is not bool
            or type(options.get("hot_read", False)) is not bool
            or options.get("hot_rows", 0) != 0
            or {"ram_cache_bytes", "ram_cache_total_bytes"} <= options.keys()
        ):
            raise ValueError("native elastic experts require zero initial HOT rows")
        text = config.model_config.hf_text_config
        e8_archive = E8ArchiveConfig.from_options(options.get("e8_archive"))
        return cls(
            NVFP4ExpertGeometry(
                text.hidden_size,
                text.moe_intermediate_size,
                config.parallel_config.tensor_parallel_size,
            ),
            text.num_hidden_layers,
            text.num_experts,
            options.get(
                "max_hot_rows",
                0
                if e8_archive is not None
                else min(8192, text.num_hidden_layers * text.num_experts),
            ),
            options.get("staging", 32),
            ram_cache_bytes=(
                options["ram_cache_total_bytes"]
                // config.parallel_config.tensor_parallel_size
                if "ram_cache_total_bytes" in options
                else options.get("ram_cache_bytes", 1 << 30)
            ),
            pin_ram_cache=options.get("pin_ram_cache", False),
            hot_read=options.get("hot_read", False),
            prepared_archive=options.get("prepared_archive"),
            cpu_experts=CpuExpertConfig.from_options(
                options.get("cpu_experts"), config.parallel_config.tensor_parallel_size
            ),
            stream_experts=StreamExpertConfig.from_options(
                options.get("stream_experts"),
                config.parallel_config.tensor_parallel_size,
            ),
            partition=options.get("partition", "legacy"),
            wave_slots=options.get("wave_slots", 1),
            dispatch_profile=DispatchConfig.from_options(
                options.get("dispatch_profile")
            ),
            target_cpu_only=options.get("target_cpu_only", False),
            e8_archive=e8_archive,
        )

    def dispatch_workspace(self):
        """Fixed compact buffers; the general worker profile owns this base.

        CPU native scratch, the captured stream and the transient policy matrix
        are separate owners, not additional borrowed HOT rows.
        """
        if self.dispatch_profile is None:
            return dict(host_pinned_bytes=0, gpu_bytes=0)
        return dict(
            host_pinned_bytes=64 * (6 * self.geometry.hidden + 8 * 10) + 12,
            gpu_bytes=64 * 4 * self.geometry.hidden,
        )

    @property
    def scratch_rows(self):
        return self.staging * self.wave_slots

    def rank_geometry(self, rank):
        if not 0 <= rank < self.geometry.tp:
            raise ValueError("expert rank outside configured TP")
        if self.stream_experts is None and self.e8_archive is None:
            return self.geometry
        start, end = self.geometry.owner_span(
            rank, balanced=self.partition == "balanced"
        )
        width = max(0, end - start)
        if width < 64 or width % 64:
            raise ValueError("unsupported compact expert owner")
        return NVFP4ExpertGeometry(self.geometry.hidden, width, 1)

    def rank_ram_bytes(self, rank):
        if self.stream_experts is None:
            return self.ram_cache_bytes
        strides = [
            ((sum(self.rank_geometry(r).strides) + 24 + 4095) // 4096) * 4096
            for r in range(self.geometry.tp)
        ]
        rows = self.ram_cache_bytes * self.geometry.tp // sum(strides)
        return rows * strides[rank]

    def rank_mapped_bytes(self, rank, hot_rows):
        self.mapped_bytes(hot_rows)
        return self.round((self.max_hot_rows + self.scratch_rows) * 6 * 4) + sum(
            self.round((hot_rows + self.scratch_rows) * stride)
            for stride in self.rank_geometry(rank).strides
        )

    @property
    def strides(self):
        return self.geometry.strides

    def round(self, size):
        return (size + self.quantum - 1) // self.quantum * self.quantum

    def mapped_bytes(self, hot_rows):
        if type(hot_rows) is not int or not 0 <= hot_rows <= self.max_hot_rows:
            raise ValueError("expert grant exceeds reserved rows")
        scalar_bytes = self.round((self.max_hot_rows + self.scratch_rows) * 6 * 4)
        return scalar_bytes + sum(
            self.round((hot_rows + self.scratch_rows) * stride)
            for stride in self.strides
        )

    @property
    def base_bytes(self):
        return self.mapped_bytes(0)

    def grant(self, hot_rows):
        return ElasticExpertGrant(
            hot_rows, self.mapped_bytes(hot_rows) - self.base_bytes
        )

    def rank_borrowed_bytes(self, hot_rows):
        """Physical growth above each rank's already profiled staging bank."""
        return tuple(
            self.rank_mapped_bytes(rank, hot_rows) - self.rank_mapped_bytes(rank, 0)
            for rank in range(self.geometry.tp)
        )

    def fit_by_rank(self, available_bytes, *, max_rows=None):
        if len(available_bytes) != self.geometry.tp or any(
            type(value) is not int or value < 0 for value in available_bytes
        ):
            raise ValueError("invalid rank expert free budgets")
        lo, hi = 0, self.max_hot_rows if max_rows is None else max_rows
        self.mapped_bytes(hi)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if all(
                used <= available
                for used, available in zip(
                    self.rank_borrowed_bytes(mid), available_bytes, strict=True
                )
            ):
                lo = mid
            else:
                hi = mid - 1
        return self.grant(lo)

    def fit(self, available_bytes, *, max_rows=None):
        if type(available_bytes) is not int or available_bytes < 0:
            raise ValueError("invalid expert free budget")
        lo, hi = 0, self.max_hot_rows if max_rows is None else max_rows
        self.mapped_bytes(hi)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.grant(mid).borrowed_bytes <= available_bytes:
                lo = mid
            else:
                hi = mid - 1
        return self.grant(lo)
