# SPDX-License-Identifier: Apache-2.0
"""Default-off TP3 decode conveyor helpers for Model Runner V2."""

from dataclasses import replace

import numpy as np

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.ubatch_utils import UBatchSlice


def make_tp3_decode_conveyor_slices(
    input_batch: InputBatch,
    batch_desc: BatchExecutionDescriptor,
    decode_query_len: int,
) -> list[UBatchSlice] | None:
    """Return two request-aligned waves for an eligible large K3 decode.

    The first POC is deliberately narrow: only an uncaptured, pure uniform
    target-verification step above the current M128 graph family is split.
    Prefill, mixed lifecycle, padding, small/tail batches and malformed draft
    geometry retain the existing monolithic path at unchanged concurrency.
    """
    if batch_desc.cg_mode != CUDAGraphMode.NONE:
        return None
    if decode_query_len <= 1 or input_batch.num_reqs < 2:
        return None
    if input_batch.num_tokens <= 128:
        return None
    # The first proof keeps both FlashInfer wrapper families shape-identical.
    # Uneven M72/M76 tail overlap is a separate experiment after M76/M76.
    if input_batch.num_reqs % 2:
        return None
    if input_batch.num_tokens_after_padding != input_batch.num_tokens:
        return None
    if input_batch.num_tokens != input_batch.num_reqs * decode_query_len:
        return None
    if np.any(input_batch.num_scheduled_tokens != decode_query_len):
        return None
    if np.any(input_batch.is_prefilling_np):
        return None
    draft_counts = input_batch.num_draft_tokens_per_req
    if draft_counts is None or np.any(draft_counts != decode_query_len - 1):
        return None

    first_reqs = input_batch.num_reqs // 2
    split_token = first_reqs * decode_query_len
    return [
        UBatchSlice(slice(0, first_reqs), slice(0, split_token)),
        UBatchSlice(
            slice(first_reqs, input_batch.num_reqs),
            slice(split_token, input_batch.num_tokens),
        ),
    ]


def slice_input_batch(
    input_batch: InputBatch,
    ubatch_slice: UBatchSlice,
) -> InputBatch:
    """Create the metadata view consumed by one request-aligned wave."""
    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice
    request_start = request_slice.start
    request_stop = request_slice.stop
    token_start = token_slice.start
    token_stop = token_slice.stop
    assert request_start is not None and request_stop is not None
    assert token_start is not None and token_stop is not None
    num_reqs = request_stop - request_start
    num_tokens = token_stop - token_start
    assert num_reqs > 0 and num_tokens > 0

    query_start_loc_np = (
        input_batch.query_start_loc_np[request_start : request_stop + 1]
        - token_start
    ).copy()
    query_start_loc = (
        input_batch.query_start_loc[request_start : request_stop + 1]
        - token_start
    )
    cu_num_logits_np = (
        input_batch.cu_num_logits_np[request_start : request_stop + 1]
        - input_batch.cu_num_logits_np[request_start]
    ).copy()
    cu_num_logits = (
        input_batch.cu_num_logits[request_start : request_stop + 1]
        - input_batch.cu_num_logits[request_start]
    )
    logits_start = int(input_batch.cu_num_logits_np[request_start])
    logits_stop = int(input_batch.cu_num_logits_np[request_stop])
    num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
    sliced_drafts = (
        None
        if num_draft_tokens_per_req is None
        else num_draft_tokens_per_req[request_slice]
    )

    return replace(
        input_batch,
        req_ids=input_batch.req_ids[request_slice],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=input_batch.idx_mapping[request_slice],
        idx_mapping_np=input_batch.idx_mapping_np[request_slice],
        expanded_idx_mapping=input_batch.expanded_idx_mapping[
            logits_start:logits_stop
        ],
        expanded_local_pos=input_batch.expanded_local_pos[
            logits_start:logits_stop
        ],
        num_scheduled_tokens=input_batch.num_scheduled_tokens[request_slice],
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=(0 if sliced_drafts is None else int(sliced_drafts.sum())),
        num_draft_tokens_per_req=sliced_drafts,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        seq_lens=input_batch.seq_lens[request_slice],
        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[request_slice],
        dcp_local_seq_lens=(
            None
            if input_batch.dcp_local_seq_lens is None
            else input_batch.dcp_local_seq_lens[request_slice]
        ),
        num_computed_tokens_np=input_batch.num_computed_tokens_np[request_slice],
        prefill_len_np=input_batch.prefill_len_np[request_slice],
        num_computed_prefill_tokens_np=(
            input_batch.num_computed_prefill_tokens_np[request_slice]
        ),
        is_prefilling_np=input_batch.is_prefilling_np[request_slice],
        max_seq_len_np=(
            None
            if input_batch.max_seq_len_np is None
            else input_batch.max_seq_len_np[request_slice]
        ),
        input_ids=input_batch.input_ids[token_slice],
        positions=input_batch.positions[token_slice],
        is_padding=input_batch.is_padding[token_slice],
        logits_indices=input_batch.logits_indices[logits_start:logits_stop]
        - token_start,
        cu_num_logits=cu_num_logits,
        cu_num_logits_np=cu_num_logits_np,
        prompt_lens=(
            None
            if input_batch.prompt_lens is None
            else input_batch.prompt_lens[request_slice]
        ),
    )
