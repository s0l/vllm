# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
)

DEVICE = current_platform.device_type


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    (
        "query_start_loc",
        "seq_lens",
        "prefill_lens",
        "cu_num_logits",
        "draft_tokens",
        "expected_indices",
        "expected_input_ids",
    ),
    [
        ([0, 1], [1], [1], [0, 1], [[0, 0, 0]], [0], [7]),
        ([0, 1], [2], [1], [0, 1], [[0, 0, 0]], [0], [91]),
        ([0, 4], [5], [1], [0, 4], [[11, 12, 13]], [0, 1, 2, 3], [91, 11, 12, 13]),
        ([0, 1, 4], [2, 3], [1, 3], [0, 1, 2], [[0], [0]], [0, 3], [91, 7, 7, 7]),
    ],
)
def test_combine_sampled_and_draft_tokens_writes_indices_before_prefill_exit(
    query_start_loc: list[int],
    seq_lens: list[int],
    prefill_lens: list[int],
    cu_num_logits: list[int],
    draft_tokens: list[list[int]],
    expected_indices: list[int],
    expected_input_ids: list[int],
):
    """Prefill rows must not bypass the independent logits-index write."""
    device = torch.device(DEVICE)
    num_reqs = len(seq_lens)
    input_ids = torch.full((query_start_loc[-1],), 7, dtype=torch.int32, device=device)
    indices = combine_sampled_and_draft_tokens(
        input_ids=input_ids,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64, device=device),
        last_sampled_tokens=torch.full(
            (num_reqs,), 91, dtype=torch.int32, device=device
        ),
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        prefill_len=torch.tensor(prefill_lens, dtype=torch.int32, device=device),
        draft_tokens=torch.tensor(draft_tokens, dtype=torch.int32, device=device),
        cu_num_logits=torch.tensor(cu_num_logits, dtype=torch.int32, device=device),
        num_logits=len(expected_indices),
    )

    assert indices.cpu().tolist() == expected_indices
    assert input_ids.cpu().tolist() == expected_input_ids


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_combine_sampled_and_draft_tokens_uses_persistent_output_buffer():
    """Sampling indices must outlive model execution in their startup storage."""
    device = torch.device(DEVICE)
    buffers = InputBuffers(max_num_reqs=2, max_num_tokens=8, device=device)
    indices = combine_sampled_and_draft_tokens(
        input_ids=buffers.input_ids[:2],
        idx_mapping=torch.arange(2, dtype=torch.int64, device=device),
        last_sampled_tokens=torch.zeros(2, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        seq_lens=torch.ones(2, dtype=torch.int32, device=device),
        prefill_len=torch.ones(2, dtype=torch.int32, device=device),
        draft_tokens=torch.zeros((2, 3), dtype=torch.int32, device=device),
        cu_num_logits=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        num_logits=2,
        logits_indices_buffer=buffers.logits_indices,
    )

    assert indices.data_ptr() == buffers.logits_indices.data_ptr()
    assert indices.cpu().tolist() == [0, 1]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_combine_sampled_and_draft_tokens_rejects_small_persistent_buffer():
    device = torch.device(DEVICE)
    with pytest.raises(ValueError, match="too small"):
        combine_sampled_and_draft_tokens(
            input_ids=torch.zeros(2, dtype=torch.int32, device=device),
            idx_mapping=torch.arange(2, dtype=torch.int64, device=device),
            last_sampled_tokens=torch.zeros(2, dtype=torch.int32, device=device),
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
            seq_lens=torch.ones(2, dtype=torch.int32, device=device),
            prefill_len=torch.ones(2, dtype=torch.int32, device=device),
            draft_tokens=torch.zeros((2, 3), dtype=torch.int32, device=device),
            cu_num_logits=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
            num_logits=2,
            logits_indices_buffer=torch.empty(1, dtype=torch.int64, device=device),
        )


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )
