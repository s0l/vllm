# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-owned FP8 QSA pages and explicit FlashInfer scratch ownership."""

from __future__ import annotations

import copy
import math

import torch

from vllm.distributed.parallel_state import get_dcp_group, get_tp_group
from vllm.utils.torch_utils import direct_register_custom_op

_plans: dict[int, SparseQSAPlan] = {}


def canonical_dcp_block_indices(
    block_indices: torch.Tensor, visible_blocks: torch.Tensor
) -> torch.Tensor:
    """One rank owns tied top-k membership; token order is stable on every rank."""
    sentinel = torch.iinfo(torch.int32).max
    columns = torch.arange(block_indices.shape[1], device=block_indices.device)
    valid = columns < visible_blocks.unsqueeze(1)
    local = torch.where(valid, block_indices, sentinel)
    shared = get_dcp_group().all_gather(local, dim=0)[: block_indices.shape[0]]
    ordered = shared.sort(dim=1).values
    return torch.where(ordered == sentinel, -1, ordered)


def page_one_view(cache: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, ...]:
    """Alias logical [B,H,N,2D] in layer-compact NHD storage; never copy KV."""
    if (
        cache.ndim != 4
        or cache.shape[-1] != 2 * head_dim
        or cache.dtype != torch.float8_e4m3fn
        or not cache.transpose(1, 2).is_contiguous()
    ):
        raise ValueError("sparse QSA requires layer-compact NHD FP8 KV storage")
    physical = cache.transpose(1, 2)
    return physical.view(-1, 1, cache.shape[1], 2 * head_dim).split(head_dim, -1)


def map_sparse_pages(
    packed: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    *,
    block_size: int,
    cp_size: int,
    cp_rank: int,
    interleave: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume indexer columns plus count using the BlockTables CP contract."""
    columns = torch.arange(width, device=packed.device)
    indices = packed[:, :-1]
    indices = torch.nn.functional.pad(indices, (0, width - indices.shape[1]), value=-1)
    count = packed[:, -1:]
    virtual_span = block_size * cp_size
    offset = indices.clamp_min(0) % virtual_span
    local = (
        indices.clamp_min(0) // virtual_span * block_size
        + offset // (interleave * cp_size) * interleave
        + offset % interleave
    )
    blocks = local // block_size
    requests = token_to_req.long().unsqueeze(1)
    valid = (
        (indices >= 0)
        & (columns < count)
        & (count <= packed.shape[1] - 1)
        & (offset // interleave % cp_size == cp_rank)
        & (blocks < block_table.shape[1])
        & (requests >= 0)
        & (requests < block_table.shape[0])
    )
    physical = block_table[
        requests.clamp(0, block_table.shape[0] - 1),
        blocks.clamp(0, block_table.shape[1] - 1).long(),
    ]
    valid = valid & (physical >= 0)
    slots = physical * block_size + local % block_size
    return torch.where(valid, slots, 0).int(), valid


def _sparse_run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: list[torch.Tensor],
    workspaces: list[torch.Tensor],
    output: torch.Tensor,
    lse: torch.Tensor,
    k_scale: float,
    v_scale: float,
    plan_id: int,
) -> None:
    wrapper = copy.copy(_plans[plan_id].wrapper)
    names = (
        "_paged_kv_indices_buf",
        "_custom_mask_buf",
        "_qo_indptr_buf",
        "_paged_kv_indptr_buf",
        "_paged_kv_last_page_len_buf",
        "_mask_indptr_buf",
    )
    if len(metadata) != len(names) or len(workspaces) != 2:
        raise ValueError("incomplete sparse QSA plan bindings")
    for name, tensor in zip(names, metadata):
        setattr(wrapper, name, tensor)
    wrapper._float_workspace_buffer, wrapper._int_workspace_buffer = workspaces
    wrapper.run(
        query,
        (key, value),
        k_scale=k_scale,
        v_scale=v_scale,
        out=output,
        lse=lse,
        return_lse=True,
    )


def _sparse_run_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: list[torch.Tensor],
    workspaces: list[torch.Tensor],
    output: torch.Tensor,
    lse: torch.Tensor,
    k_scale: float,
    v_scale: float,
    plan_id: int,
) -> None:
    pass


direct_register_custom_op(
    "qsa_flashinfer_sparse_run",
    _sparse_run,
    mutates_args=["workspaces", "output", "lse"],
    fake_impl=_sparse_run_fake,
)


class SparseQSAPlan:
    """Fixed tile geometry. The caller owns plan lifetime and scratch leases.

    Plans must be constructed before capture. Layers may share a plan only
    when serialized through the same explicit mutable scratch arguments.
    Scales are immutable capture-time values; changing them retires the graph.
    """

    def __init__(
        self,
        rows: int,
        selection_width: int,
        *,
        num_heads: int = 24,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        workspace: torch.Tensor | None = None,
    ) -> None:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        if (
            rows < 1
            or selection_width < 1
            or num_heads % num_kv_heads
            or not all(math.isfinite(s) and s > 0 for s in (k_scale, v_scale))
        ):
            raise ValueError("invalid sparse QSA plan geometry/scales")
        self.rows, self.head_dim = rows, head_dim
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.width = (selection_width + 7) // 8 * 8
        self.k_scale, self.v_scale = k_scale, v_scale
        self.indices = torch.zeros(rows * self.width, device="cuda", dtype=torch.int32)
        self.mask = torch.zeros(
            rows * (self.width // 8), device="cuda", dtype=torch.uint8
        )
        self.bits = 1 << torch.arange(8, device="cuda", dtype=torch.int32)
        self.output = torch.empty(
            rows, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        self.lse = torch.empty(rows, num_heads, device="cuda", dtype=torch.float32)
        if workspace is None:
            workspace = torch.empty(32 << 20, device="cuda", dtype=torch.uint8)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf=torch.empty(rows + 1, device="cuda", dtype=torch.int32),
            paged_kv_indptr_buf=torch.empty(rows + 1, device="cuda", dtype=torch.int32),
            paged_kv_indices_buf=self.indices,
            paged_kv_last_page_len_buf=torch.empty(
                rows, device="cuda", dtype=torch.int32
            ),
            custom_mask_buf=self.mask,
            mask_indptr_buf=torch.empty(rows + 1, device="cuda", dtype=torch.int32),
            backend="fa2",
        )
        self.wrapper.plan(
            torch.arange(rows + 1, dtype=torch.int32),
            torch.arange(rows + 1, dtype=torch.int32) * self.width,
            torch.zeros(rows * self.width, dtype=torch.int32),
            torch.ones(rows, dtype=torch.int32),
            num_heads,
            num_kv_heads,
            head_dim,
            1,
            causal=False,
            custom_mask=torch.zeros(rows * self.width, device="cuda", dtype=torch.bool),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
        )
        expected = torch.arange(rows + 1, dtype=torch.int32) * (self.width // 8)
        if not torch.equal(self.wrapper._mask_indptr_buf.cpu(), expected):
            raise RuntimeError("FlashInfer sparse mask offsets are not byte offsets")
        self.plan_id = len(_plans)
        _plans[self.plan_id] = self
        self.metadata = [
            self.indices,
            self.mask,
            self.wrapper._qo_indptr_buf,
            self.wrapper._paged_kv_indptr_buf,
            self.wrapper._paged_kv_last_page_len_buf,
            self.wrapper._mask_indptr_buf,
        ]
        self.workspaces = [
            self.wrapper._float_workspace_buffer,
            self.wrapper._int_workspace_buffer,
        ]

    def __call__(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        packed: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        *,
        block_size: int,
        cp_size: int,
        cp_rank: int,
        interleave: int,
    ) -> torch.Tensor:
        if (
            query.shape[0] != self.rows
            or packed.shape[0] != self.rows
            or block_size % interleave
            or not 0 <= cp_rank < cp_size
        ):
            raise ValueError("sparse QSA query/ownership does not match its plan")
        slots, valid = map_sparse_pages(
            packed,
            block_table,
            token_to_req,
            block_size=block_size,
            cp_size=cp_size,
            cp_rank=cp_rank,
            interleave=interleave,
            width=self.width,
        )
        key, value = page_one_view(cache, self.head_dim)
        valid = valid & (slots < key.shape[0])
        self.indices.copy_(torch.where(valid, slots, 0).flatten())
        self.mask.copy_(
            (valid.reshape(-1, 8).int() * self.bits).sum(-1).to(torch.uint8)
        )
        all_q = get_tp_group().all_gather(query, dim=1)
        torch.ops.vllm.qsa_flashinfer_sparse_run(
            all_q,
            key,
            value,
            self.metadata,
            self.workspaces,
            self.output,
            self.lse,
            self.k_scale,
            self.v_scale,
            self.plan_id,
        )
        present = valid.any(dim=1, keepdim=True)
        output = torch.where(present.unsqueeze(-1), self.output, 0)
        lse = torch.where(present, self.lse, -torch.inf)
        dcp = get_dcp_group()
        outputs = dcp.all_gather(output, dim=0).reshape(
            cp_size, self.rows, self.num_heads, self.head_dim
        )
        lses = dcp.all_gather(lse, dim=0).reshape(cp_size, self.rows, self.num_heads)
        maximum = lses.amax(0)
        mass = torch.exp2(lses - torch.where(torch.isfinite(maximum), maximum, 0))
        total = mass.sum(0).clamp_min(torch.finfo(torch.float32).tiny)
        merged = (outputs.float() * (mass / total).unsqueeze(-1)).sum(0)
        tp = get_tp_group()
        owned_heads = self.num_heads // tp.world_size
        return merged[
            :, tp.rank_in_group * owned_heads : (tp.rank_in_group + 1) * owned_heads
        ].to(torch.bfloat16)
