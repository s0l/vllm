# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fused vocab-parallel embedding kernel
(``vllm._custom_ops.vocab_parallel_embedding``).

The guarantee: the kernel is a drop-in for the TP > 1 embedding path, i.e. it
returns bit-exactly what mask + shift + gather + ``masked_fill_`` returns on
each rank, and summing the per-rank outputs (what the all-reduce does)
reconstructs a full-table lookup.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers import vocab_parallel_embedding as embedding_module
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
    pad_vocab_size,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the fused vocab-parallel embedding kernel is CUDA-only",
)


def test_fused_forward_uses_embedding_collective(monkeypatch):
    """The fused lookup is still only a rank-local TP partial."""
    layer = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(layer)
    layer.tp_size = 3
    layer.use_fused_embedding = True
    layer.embedding_dim = 4
    layer.weight = torch.nn.Parameter(torch.zeros(2, 4))
    layer.shard_indices = VocabParallelEmbedding._get_indices(
        vocab_size_padded=6,
        org_vocab_size_padded=6,
        vocab_size=6,
        org_vocab_size=6,
        tp_rank=0,
        tp_size=3,
    )

    partial = torch.full((2, 4), 3.0)
    reduced = torch.full((2, 4), 9.0)
    calls = []

    monkeypatch.setattr(ops, "vocab_parallel_embedding", lambda *args: partial)

    def reduce_rank_partials(value):
        calls.append(value)
        return reduced

    monkeypatch.setattr(
        embedding_module,
        "tensor_model_parallel_embedding_all_reduce",
        reduce_rank_partials,
    )

    output = layer(torch.tensor([0, 1], dtype=torch.int32))

    assert calls == [partial]
    assert output is reduced


def _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size, pad=64):
    return VocabParallelEmbedding._get_indices(
        pad_vocab_size(vocab_size, pad),
        pad_vocab_size(org_vocab_size, pad),
        vocab_size,
        org_vocab_size,
        tp_rank,
        tp_size,
    )


def _reference(input_ids, weight, si):
    """The path the kernel replaces."""
    masked_input, input_mask = get_masked_input_and_mask(
        input_ids,
        si.org_vocab_start_index,
        si.org_vocab_end_index,
        si.num_org_vocab_padding,
        si.added_vocab_start_index,
        si.added_vocab_end_index,
    )
    out = torch.nn.functional.embedding(masked_input.long(), weight)
    return out.masked_fill_(input_mask.unsqueeze(-1), 0)


def _fused(input_ids, weight, si):
    return ops.vocab_parallel_embedding(
        input_ids,
        weight,
        si.org_vocab_start_index,
        si.org_vocab_end_index,
        si.num_org_vocab_padding,
        si.added_vocab_start_index,
        si.added_vocab_end_index,
    )


def _shard(full_table, si, rows_per_rank, dtype):
    """Lay a rank's rows out the way the weight loader does."""
    weight = torch.zeros(rows_per_rank, full_table.shape[1], dtype=dtype, device="cuda")
    num_org = si.org_vocab_end_index - si.org_vocab_start_index
    weight[:num_org] = full_table[si.org_vocab_start_index : si.org_vocab_end_index]
    added_at = num_org + si.num_org_vocab_padding
    num_added = si.added_vocab_end_index - si.added_vocab_start_index
    weight[added_at : added_at + num_added] = full_table[
        si.added_vocab_start_index : si.added_vocab_end_index
    ]
    return weight


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize(
    "vocab_size,org_vocab_size", [(128256, 128256), (32256, 32000)]
)
@pytest.mark.parametrize("hidden", [4096, 3584])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_matches_masked_gather(tp_size, vocab_size, org_vocab_size, hidden, id_dtype):
    """Every rank's partial output is bit-exact with the eager path, including
    the added-vocab (LoRA) rows and the ids owned by another rank."""
    set_random_seed(7)
    dtype = torch.bfloat16
    rows = pad_vocab_size(vocab_size, 64) // tp_size
    full_table = torch.randn(pad_vocab_size(vocab_size, 64), hidden, dtype=dtype).cuda()
    ids = torch.randint(0, vocab_size, (257,), dtype=id_dtype, device="cuda")

    for tp_rank in range(tp_size):
        si = _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size)
        weight = _shard(full_table, si, rows, dtype)
        torch.testing.assert_close(
            _fused(ids, weight, si), _reference(ids, weight, si), atol=0.0, rtol=0.0
        )


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("tp_size", [2, 8])
def test_rank_sum_is_full_lookup(tp_size):
    """What the all-reduce sees: the partials add up to a full-table lookup."""
    set_random_seed(11)
    vocab_size, org_vocab_size, hidden = 32256, 32000, 2048
    dtype = torch.bfloat16
    rows = pad_vocab_size(vocab_size, 64) // tp_size
    full_table = torch.randn(pad_vocab_size(vocab_size, 64), hidden, dtype=dtype).cuda()
    # Ids from both the base and the added (LoRA) range.
    ids = torch.randint(0, vocab_size, (129,), dtype=torch.int32, device="cuda")
    ids[:16] = torch.randint(
        org_vocab_size, vocab_size, (16,), dtype=torch.int32, device="cuda"
    )

    total = torch.zeros(ids.shape[0], hidden, dtype=dtype, device="cuda")
    for tp_rank in range(tp_size):
        si = _shard_indices(vocab_size, org_vocab_size, tp_rank, tp_size)
        total += _fused(ids, _shard(full_table, si, rows, dtype), si)

    torch.testing.assert_close(total, full_table[ids.long()], atol=0.0, rtol=0.0)
