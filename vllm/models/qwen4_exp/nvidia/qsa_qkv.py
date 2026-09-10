# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA projection with owned Q/gate heads and replicated K/V heads.

Retains the normal packed qkv_proj.weight checkpoint interface. Each q_proj
head contains its Q vector followed by its gate; K/V stay in global head order
for the sequence-owned sparse DCP consumer.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from vllm.model_executor.layers.linear import ReplicatedLinear


class QSAOwnedQKVLinear(ReplicatedLinear):
    """BF16 checkpoint projection for TP boundaries crossing a GQA group."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        tp_size: int,
        tp_rank: int,
        params_dtype: torch.dtype = torch.bfloat16,
        prefix: str = "",
    ) -> None:
        if (
            min(hidden_size, num_heads, num_kv_heads, head_dim, tp_size) < 1
            or num_heads % tp_size
            or num_heads % num_kv_heads
            or not 0 <= tp_rank < tp_size
            or params_dtype != torch.bfloat16
        ):
            raise ValueError("invalid owned QSA projection geometry/dtype")
        self.owner_rank, self.owner_size = tp_rank, tp_size
        self.total_q_rows = num_heads * head_dim * 2
        self.local_q_rows = self.total_q_rows // tp_size
        self.kv_rows = num_kv_heads * head_dim
        self.loaded_shards: set[str] = set()
        super().__init__(
            hidden_size,
            self.local_q_rows + self.kv_rows * 2,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
            disable_tp=True,
        )

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | None = None,
    ) -> None:
        if param is not self.weight or shard_id not in ("q", "k", "v"):
            raise ValueError("QSA projection requires named q/k/v source shards")
        if shard_id in self.loaded_shards:
            raise ValueError("duplicate QSA projection source shard")
        rows = self.total_q_rows if shard_id == "q" else self.kv_rows
        if (
            loaded_weight.shape != (rows, self.input_size)
            or loaded_weight.dtype != self.params_dtype
        ):
            raise ValueError("QSA source projection shape/dtype mismatch")
        if shard_id == "q":
            start = self.owner_rank * self.local_q_rows
            value = loaded_weight[start : start + self.local_q_rows]
            offset = 0
        else:
            value = loaded_weight
            offset = self.local_q_rows + (self.kv_rows if shard_id == "v" else 0)
        param.data[offset : offset + value.shape[0]].copy_(value)
        self.loaded_shards.add(shard_id)

    def forward(self, x: torch.Tensor):
        if self.loaded_shards != {"q", "k", "v"}:
            raise RuntimeError("QSA projection has incomplete source coverage")
        return super().forward(x)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, weight in weights:
            if name != "weight":
                raise ValueError("unsupported QSA projection source parameter")
            self.weight_loader(self.weight, weight, getattr(weight, "shard_id", None))
            loaded.add(name)
        return loaded
