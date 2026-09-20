# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata


def make_metadata(**kwargs) -> CommonAttentionMetadata:
    values = dict(
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=7,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int64),
    )
    values.update(kwargs)
    return CommonAttentionMetadata(**values)


def test_deprecated_cpu_caches_are_lazy_and_reused() -> None:
    metadata = make_metadata()

    with pytest.warns(DeprecationWarning):
        seq_lens_cpu = metadata.seq_lens_cpu
    with pytest.warns(DeprecationWarning):
        num_computed_tokens_cpu = metadata.num_computed_tokens_cpu

    assert seq_lens_cpu.tolist() == [5, 7]
    assert num_computed_tokens_cpu.tolist() == [3, 6]
    assert metadata._seq_lens_cpu is seq_lens_cpu
    assert metadata._num_computed_tokens_cpu is num_computed_tokens_cpu


def test_unpadded_preserves_only_materialized_cpu_cache_prefix() -> None:
    metadata = make_metadata()
    cold = metadata.unpadded(num_actual_tokens=2, num_actual_reqs=1)
    assert cold._seq_lens_cpu is None
    assert cold._num_computed_tokens_cpu is None

    with pytest.warns(DeprecationWarning):
        _ = metadata.num_computed_tokens_cpu
    warm = metadata.unpadded(num_actual_tokens=2, num_actual_reqs=1)
    assert warm._seq_lens_cpu.tolist() == [5]
    assert warm._num_computed_tokens_cpu.tolist() == [3]


def test_precomputed_cpu_caches_avoid_recomputation() -> None:
    seq_lens_cpu = torch.tensor([50, 70], dtype=torch.int32)
    computed_cpu = torch.tensor([30, 60], dtype=torch.int32)
    metadata = make_metadata(
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=computed_cpu,
    )

    with pytest.warns(DeprecationWarning):
        assert metadata.seq_lens_cpu is seq_lens_cpu
    with pytest.warns(DeprecationWarning):
        assert metadata.num_computed_tokens_cpu is computed_cpu
