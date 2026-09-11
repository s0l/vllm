# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler/worker contract for the native FlashNext expert bank."""

from dataclasses import dataclass

from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry


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
        ):
            raise ValueError("unsupported native expert budget geometry")

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
            }
            or any(type(v) is not int or v < 0 for v in options.values())
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
