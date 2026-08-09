# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.tp3_conveyor import (
    make_tp3_decode_conveyor_slices,
    slice_input_batch,
)


def _k3_decode_batch(num_reqs: int) -> InputBatch:
    num_tokens = num_reqs * 4
    buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    scheduled = np.full(num_reqs, 4, dtype=np.int32)
    drafts = np.full(num_reqs, 3, dtype=np.int32)
    query_start = np.arange(0, num_tokens + 1, 4, dtype=np.int32)
    return replace(
        batch,
        num_scheduled_tokens=scheduled,
        num_draft_tokens=int(drafts.sum()),
        num_draft_tokens_per_req=drafts,
        query_start_loc=torch.from_numpy(query_start),
        query_start_loc_np=query_start,
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        is_padding=torch.zeros(num_tokens, dtype=torch.bool),
    )


def _desc(batch: InputBatch, mode: CUDAGraphMode) -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=mode,
        num_tokens=batch.num_tokens,
        num_reqs=None,
        uniform_token_count=4,
    )


@pytest.mark.parametrize(
    ("num_reqs", "expected_request_counts", "expected_token_counts"),
    [(38, [19, 19], [76, 76])],
)
def test_large_uniform_k3_is_split_on_request_boundary(
    num_reqs: int,
    expected_request_counts: list[int],
    expected_token_counts: list[int],
) -> None:
    batch = _k3_decode_batch(num_reqs)
    waves = make_tp3_decode_conveyor_slices(
        batch, _desc(batch, CUDAGraphMode.NONE), decode_query_len=4
    )
    assert waves is not None
    assert [w.request_slice.stop - w.request_slice.start for w in waves] == (
        expected_request_counts
    )
    assert [w.num_tokens for w in waves] == expected_token_counts

    second = slice_input_batch(batch, waves[1])
    assert second.req_ids == batch.req_ids[waves[1].request_slice]
    assert second.query_start_loc_np.tolist() == list(
        range(0, expected_token_counts[1] + 1, 4)
    )
    assert second.num_tokens == expected_token_counts[1]
    assert second.num_reqs == expected_request_counts[1]
    assert second.logits_indices.min().item() >= 0
    assert second.logits_indices.max().item() < second.num_tokens


@pytest.mark.parametrize("num_reqs", [1, 8, 32])
def test_small_batches_retain_monolithic_path(num_reqs: int) -> None:
    batch = _k3_decode_batch(num_reqs)
    assert (
        make_tp3_decode_conveyor_slices(
            batch, _desc(batch, CUDAGraphMode.NONE), decode_query_len=4
        )
        is None
    )


def test_odd_large_batch_retains_monolithic_path() -> None:
    batch = _k3_decode_batch(37)
    assert (
        make_tp3_decode_conveyor_slices(
            batch, _desc(batch, CUDAGraphMode.NONE), decode_query_len=4
        )
        is None
    )


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE])
def test_captured_batches_retain_monolithic_path(mode: CUDAGraphMode) -> None:
    batch = _k3_decode_batch(38)
    assert make_tp3_decode_conveyor_slices(batch, _desc(batch, mode), 4) is None


def test_prefill_or_nonuniform_geometry_retain_monolithic_path() -> None:
    batch = _k3_decode_batch(38)
    prefill = replace(
        batch,
        is_prefilling_np=np.array([True] + [False] * 37, dtype=np.bool_),
    )
    assert (
        make_tp3_decode_conveyor_slices(
            prefill, _desc(prefill, CUDAGraphMode.NONE), 4
        )
        is None
    )

    nonuniform = replace(
        batch,
        num_scheduled_tokens=np.array([3] + [4] * 37, dtype=np.int32),
    )
    assert (
        make_tp3_decode_conveyor_slices(
            nonuniform, _desc(nonuniform, CUDAGraphMode.NONE), 4
        )
        is None
    )
