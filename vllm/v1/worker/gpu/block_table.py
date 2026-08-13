# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.rank_projected_owner import ElasticPageOwnerPolicy
from vllm.v1.worker.gpu.buffer_utils import (
    FusedStagedWriter,
    StagedWriteTensor,
    UvaBackedTensor,
    _load_ptr,
)


class BlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int],
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
        rank_projected_groups: list[bool] | None = None,
        full_history_groups: list[bool] | None = None,
    ):
        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave
        self.rank_projected_groups = rank_projected_groups
        self.full_history_groups = full_history_groups

        if rank_projected_groups is not None and full_history_groups is not None:
            raise ValueError(
                "rank-projected and replicated full-history groups are mutually "
                "exclusive"
            )

        if any(bs % kbs for bs, kbs in zip(block_sizes, kernel_block_sizes)):
            raise ValueError("KV block size must be divisible by kernel page size")
        self.blocks_per_kv_block = [
            bs // kbs for bs, kbs in zip(block_sizes, kernel_block_sizes)
        ]

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
        if full_history_groups is not None:
            if cp_size != 3 or cp_interleave != 1:
                raise ValueError("Exp11 9/9/6 full history requires DCP3/interleave1")
            if len(full_history_groups) != self.num_kv_cache_groups:
                raise ValueError("full-history KV group mask length mismatch")
            if not any(full_history_groups):
                raise ValueError("full-history KV requires an attention group")
            self.full_history_group_mask = torch.tensor(
                full_history_groups, dtype=torch.int8, device=device
            )
        else:
            self.full_history_group_mask = None
        if rank_projected_groups is not None:
            if cp_size != 3 or cp_interleave != 1:
                raise ValueError("Exp11 rank-projected KV requires DCP3/interleave1")
            if len(rank_projected_groups) != self.num_kv_cache_groups:
                raise ValueError("rank-projected KV group mask length mismatch")
            if not any(rank_projected_groups):
                raise ValueError("rank-projected KV requires an attention group")
            projected_page_sizes = {
                kernel_page_size
                for kernel_page_size, projected in zip(
                    kernel_block_sizes, rank_projected_groups, strict=True
                )
                if projected
            }
            if len(projected_page_sizes) != 1:
                raise ValueError(
                    "rank-projected attention groups require one kernel page size"
                )
            self.rank_projected_page_size = projected_page_sizes.pop()
            max_global_pages = max(
                value * pages_per_block * cp_size
                for value, pages_per_block, projected in zip(
                    max_num_blocks_per_group,
                    self.blocks_per_kv_block,
                    rank_projected_groups,
                    strict=True,
                )
                if projected
            )
            owners, ordinals, prefix_counts = ElasticPageOwnerPolicy(
                page_size=self.rank_projected_page_size
            ).build_luts(max_global_pages)
            self.rank_projected_owner_lut_host = owners
            self.rank_projected_owner_lut = torch.tensor(
                owners, dtype=torch.uint8, device=device
            )
            self.rank_projected_ordinal_lut = torch.tensor(
                ordinals, dtype=torch.int32, device=device
            )
            self.rank_projected_prefix_counts = torch.tensor(
                prefix_counts, dtype=torch.int32, device=device
            )
            self.rank_projected_group_mask = torch.tensor(
                rank_projected_groups, dtype=torch.int8, device=device
            )
        else:
            self.rank_projected_page_size = None
            self.rank_projected_owner_lut_host = None
            self.rank_projected_owner_lut = None
            self.rank_projected_ordinal_lut = None
            self.rank_projected_prefix_counts = None
            self.rank_projected_group_mask = None

        if rank_projected_groups is not None:
            for blocks_per_kv_block, projected in zip(
                self.blocks_per_kv_block,
                rank_projected_groups,
                strict=True,
            ):
                if projected and blocks_per_kv_block <= 0:
                    raise ValueError(
                        "rank-projected attention requires at least one kernel "
                        "page per worker KV block"
                    )

        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        self.host_block_tables: list[np.ndarray] = []
        for i in range(self.num_kv_cache_groups):
            physical_blocks_per_logical = (
                2
                if rank_projected_groups is not None
                and rank_projected_groups[i]
                else 1
            )
            max_num_blocks = (
                max_num_blocks_per_group[i]
                * self.blocks_per_kv_block[i]
                * physical_blocks_per_logical
            )
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks), dtype=torch.int32, device=device
            )
            self.block_tables.append(block_table)
            # A small authoritative host mirror used by fail-closed diagnostics.
            # Reading the GPU table back immediately before graph replay would
            # insert a device synchronization and perturb distributed execution.
            self.host_block_tables.append(
                np.zeros((self.max_num_reqs, max_num_blocks), dtype=np.int32)
            )

        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )
        self.logical_num_blocks = np.zeros(
            (self.num_kv_cache_groups, self.max_num_reqs), dtype=np.int32
        )
        self.fused_writer: FusedStagedWriter | None = None
        if self.num_kv_cache_groups > 1:
            # Only the multi-group path uses the fused writer.
            self.fused_writer = FusedStagedWriter(
                self.device, self.num_kv_cache_groups * self.max_num_reqs
            )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]

        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        # Called at init and after a CuMem kv_cache wake-up. The ptr tensors
        # cache raw data_ptr() values that go stale once the underlying tensors
        # are reallocated on wake; block_sizes_tensor needs re-populating
        # because its storage lives under the kv_cache pool tag and comes back
        # with undefined contents.
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            block_ids = new_block_ids[i]
            bpk = self.blocks_per_kv_block[i]
            rank_projected = bool(
                self.rank_projected_groups is not None
                and self.rank_projected_groups[i]
            )
            if rank_projected:
                global_pages_per_block = bpk * self.cp_size
                logical_start = (
                    0 if overwrite else int(self.logical_num_blocks[i, req_index])
                )
                owners = self.rank_projected_owner_lut_host
                assert owners is not None
                expanded: list[int] = []
                for logical_offset, block_id in enumerate(block_ids):
                    global_page_start = (
                        logical_start + logical_offset
                    ) * global_pages_per_block
                    local_slot = 0
                    for page in range(
                        global_page_start,
                        global_page_start + global_pages_per_block,
                    ):
                        if owners[page] == self.cp_rank:
                            expanded.append(block_id * 2 * bpk + local_slot)
                            local_slot += 1
                    if local_slot > 2 * bpk:
                        raise ValueError(
                            "rank-projected packed superblock exceeds two worker "
                            "KV blocks"
                        )
                block_ids = expanded
                self.logical_num_blocks[i, req_index] = logical_start + len(
                    new_block_ids[i]
                )
            elif bpk > 1:
                block_ids = [b * bpk + k for b in block_ids for k in range(bpk)]
                self.logical_num_blocks[i, req_index] = (
                    0 if overwrite else self.logical_num_blocks[i, req_index]
                ) + len(new_block_ids[i])
            else:
                self.logical_num_blocks[i, req_index] = (
                    0 if overwrite else self.logical_num_blocks[i, req_index]
                ) + len(new_block_ids[i])
            self.block_tables[i].stage_write(req_index, start, block_ids)
            end = start + len(block_ids)
            self.host_block_tables[i][req_index, start:end] = block_ids
            self.num_blocks.np[i, req_index] = end

    def apply_staged_writes(self) -> None:
        if self.num_kv_cache_groups == 0:
            return
        if self.num_kv_cache_groups == 1:
            # Single group: write directly, skipping the per-write group lookup.
            self.block_tables[0].apply_write()
        elif self.num_kv_cache_groups > 1:
            # Multiple groups: apply all block tables with one fused kernel.
            assert self.fused_writer is not None
            self.fused_writer.apply(
                self.block_tables, self.block_table_ptrs, self.block_table_strides
            )
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs
        else:
            assert out_ptrs is not None
            assert len(out) == self.num_kv_cache_groups
        num_reqs = idx_mapping.shape[0]
        # Launch kernel with num_reqs_padded to fuse zeroing of padded rows.
        _gather_block_tables_kernel[(self.num_kv_cache_groups, num_reqs_padded)](
            idx_mapping,
            self.block_table_ptrs,
            out_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            num_reqs,
            BLOCK_SIZE=1024,  # type: ignore
        )
        return tuple(bt[:num_reqs_padded] for bt in out)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        #
        # Zero the rows so dummy runs write mamba state to the reserved null
        # block rather than through the previous real step's (stale) block
        # ids, which may point at blocks since freed and reallocated.
        return tuple(
            block_table[:num_reqs].zero_() for block_table in self.input_block_tables
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_kv_cache_groups == 0:
            return (self.slot_mappings if out is None else out)[:, :num_tokens_padded]
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        slot_mappings = self.slot_mappings if out is None else out
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            slot_mappings.shape[1],
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            slot_mappings,
            slot_mappings.stride(0),
            self.cp_rank,
            self.rank_projected_owner_lut
            if self.rank_projected_owner_lut is not None
            else positions,
            self.rank_projected_ordinal_lut
            if self.rank_projected_ordinal_lut is not None
            else positions,
            self.rank_projected_group_mask
            if self.rank_projected_group_mask is not None
            else positions,
            self.full_history_group_mask
            if self.full_history_group_mask is not None
            else positions,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            RANK_PROJECTED=self.rank_projected_groups is not None,
            FULL_HISTORY=self.full_history_groups is not None,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        return slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    max_num_blocks = stride  # stride equals max_num_blocks for this group.
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # Zero out padded rows.
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    rank_projected_owner_lut,
    rank_projected_ordinal_lut,
    rank_projected_group_mask,
    full_history_group_mask,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    RANK_PROJECTED: tl.constexpr,
    FULL_HISTORY: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

            if RANK_PROJECTED:
                use_rank_projection = (
                    tl.load(rank_projected_group_mask + group_id) != 0
                )
                global_pages = positions // block_size
                page_offsets = positions % block_size
                owners = tl.load(rank_projected_owner_lut + global_pages)
                local_ordinals = tl.load(
                    rank_projected_ordinal_lut + global_pages
                )
                projected_block_numbers = tl.load(
                    block_table_ptr
                    + req_state_idx * block_table_stride
                    + local_ordinals
                )
                projected_slots = projected_block_numbers * block_size + page_offsets
                projected_slots = tl.where(
                    owners == cp_rank, projected_slots, PAD_ID
                )
                slot_ids = tl.where(use_rank_projection, projected_slots, slot_ids)

            if FULL_HISTORY:
                use_full_history = tl.load(full_history_group_mask + group_id) != 0
                full_block_indices = positions // block_size
                full_offsets = positions % block_size
                full_block_numbers = tl.load(
                    block_table_ptr
                    + req_state_idx * block_table_stride
                    + full_block_indices
                )
                full_slots = full_block_numbers * block_size + full_offsets
                slot_ids = tl.where(use_full_history, full_slots, slot_ids)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
