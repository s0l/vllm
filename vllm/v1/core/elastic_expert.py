# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler/worker contract for the native FlashNext expert bank."""

from dataclasses import dataclass
from pathlib import Path

from vllm.utils.nvfp4_cpu_experts import CpuExpertConfig
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

    def __post_init__(self):
        self.geometry.validate_cutlass()
        if (
            any(type(v) is not int or v <= 0 for v in (self.layers, self.experts))
            or type(self.max_hot_rows) is not int
            or not 0 <= self.max_hot_rows <= self.layers * self.experts
            or type(self.staging) is not int
            or not 1 <= self.staging <= min(self.experts, 1024)
            or self.staging & (self.staging - 1)
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
                }
            )
            or type(options.get("pin_ram_cache", False)) is not bool
            or type(options.get("hot_read", False)) is not bool
            or options.get("hot_rows", 0) != 0
            or {"ram_cache_bytes", "ram_cache_total_bytes"} <= options.keys()
        ):
            raise ValueError("native elastic experts require zero initial HOT rows")
        text = config.model_config.hf_text_config
        return cls(
            NVFP4ExpertGeometry(
                text.hidden_size,
                text.moe_intermediate_size,
                config.parallel_config.tensor_parallel_size,
            ),
            text.num_hidden_layers,
            text.num_experts,
            options.get(
                "max_hot_rows", min(8192, text.num_hidden_layers * text.num_experts)
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
        )

    def rank_geometry(self, rank):
        if not 0 <= rank < self.geometry.tp:
            raise ValueError("expert rank outside configured TP")
        if self.stream_experts is None:
            return self.geometry
        width = min(
            self.geometry.local,
            max(0, self.geometry.width - rank * self.geometry.local),
        )
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
        return self.round((self.max_hot_rows + self.staging) * 6 * 4) + sum(
            self.round((hot_rows + self.staging) * stride)
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
        scalar_bytes = self.round((self.max_hot_rows + self.staging) * 6 * 4)
        return scalar_bytes + sum(
            self.round((hot_rows + self.staging) * stride) for stride in self.strides
        )

    @property
    def base_bytes(self):
        return self.mapped_bytes(0)

    def grant(self, hot_rows):
        return ElasticExpertGrant(
            hot_rows, self.mapped_bytes(hot_rows) - self.base_bytes
        )

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
