# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


def prepare_dcp_local_seq_lens(
    dcp_local_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
    rank_projected_owner_lut: torch.Tensor | None = None,
    rank_projected_prefix_counts: torch.Tensor | None = None,
    rank_projected_page_size: int | None = None,
    full_history: bool = False,
) -> None:
    """Populate the persistent DCP local seq_lens buffer (CUDA graph safe)."""
    if dcp_size == 1:
        return

    max_num_reqs = dcp_local_seq_lens.shape[0]
    rank_projected = rank_projected_owner_lut is not None
    if rank_projected and full_history:
        raise ValueError(
            "rank-projected and replicated full-history lengths are mutually "
            "exclusive"
        )
    if rank_projected != (rank_projected_prefix_counts is not None):
        raise ValueError(
            "rank-projected DCP local-length LUTs must be provided together"
        )
    if rank_projected:
        assert rank_projected_owner_lut is not None
        assert rank_projected_prefix_counts is not None
        if dcp_size != 3 or cp_interleave != 1:
            raise ValueError("Exp11 rank-projected DCP requires size 3/interleave 1")
        if rank_projected_prefix_counts.shape[0] != dcp_size:
            raise ValueError("rank-projected prefix-count world mismatch")
        if rank_projected_page_size is None or rank_projected_page_size <= 0:
            raise ValueError("rank-projected DCP requires a positive page size")
    BLOCK_SIZE = 128
    num_blocks = triton.cdiv(max_num_reqs, BLOCK_SIZE)
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        dcp_local_seq_lens,
        seq_lens,
        dcp_size,
        dcp_rank,
        cp_interleave,
        num_reqs,
        max_num_reqs,
        rank_projected_owner_lut
        if rank_projected_owner_lut is not None
        else seq_lens,
        rank_projected_prefix_counts
        if rank_projected_prefix_counts is not None
        else seq_lens,
        (
            rank_projected_prefix_counts.stride(0)
            if rank_projected_prefix_counts is not None
            else 0
        ),
        rank_projected_page_size or 1,
        RANK_PROJECTED=rank_projected,
        FULL_HISTORY=full_history,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _dcp_local_seq_lens_kernel(
    out_ptr,
    seq_lens_ptr,
    dcp_size,
    dcp_rank,
    cp_interleave,
    num_reqs,
    max_num_reqs,
    owner_lut_ptr,
    prefix_counts_ptr,
    prefix_counts_stride,
    rank_projected_page_size,
    RANK_PROJECTED: tl.constexpr,
    FULL_HISTORY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    seq_lens = tl.load(seq_lens_ptr + block, mask=block < num_reqs)

    if FULL_HISTORY:
        local_seq_lens = seq_lens
    elif RANK_PROJECTED:
        full_pages = seq_lens // rank_projected_page_size
        partial = seq_lens % rank_projected_page_size
        full_local_pages = tl.load(
            prefix_counts_ptr + dcp_rank * prefix_counts_stride + full_pages
        )
        partial_owner = tl.load(
            owner_lut_ptr + full_pages,
            mask=partial > 0,
            other=-1,
        )
        local_seq_lens = full_local_pages * rank_projected_page_size + tl.where(
            partial_owner == dcp_rank, partial, 0
        )
    else:
        # Distribute KV cache among different ranks, in a round-robin manner.
        rounds = seq_lens // (dcp_size * cp_interleave)
        remainder = seq_lens % (dcp_size * cp_interleave)

        remainder = tl.maximum(remainder - dcp_rank * cp_interleave, 0)
        remainder = tl.minimum(remainder, cp_interleave)
        local_seq_lens = rounds * cp_interleave + remainder

    # For [num_reqs, max_num_reqs), pad with 0
    local_seq_lens = tl.where(block < num_reqs, local_seq_lens, 0)
    tl.store(out_ptr + block, local_seq_lens, mask=block < max_num_reqs)
