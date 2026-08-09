# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch ordering in Model Runner V2 (vllm.v1.worker.gpu).

split_decodes_and_prefills assumes decode -> short_extend -> prefill request
ordering. With spec decode (decode_query_len > 1), a shorter chunked-prefill
tail sorted in front of the uniform decodes misclassifies every decode as a
prefill.
"""

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flashinfer import _ag2_semantic_prefill_split
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids


def _make_common_attn_metadata(query_lens: list[int]) -> CommonAttentionMetadata:
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    torch.cumsum(
        torch.tensor(query_lens, dtype=torch.int32), 0, out=query_start_loc[1:]
    )
    seq_lens = torch.tensor([1000 + q for q in query_lens], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        max_seq_len=int(seq_lens.max()),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        block_table_tensor=torch.zeros(num_reqs, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
    )


def test_sort_batch_req_ids_no_spec():
    # decode_query_len == 1: plain ascending order (decodes first).
    num_tokens_per_req = {"p1": 100, "d1": 1, "p2": 7, "d2": 1}
    assert sort_batch_req_ids(num_tokens_per_req, 1) == ["d1", "d2", "p2", "p1"]


def test_sort_batch_req_ids_spec_decode():
    # decode_query_len == 2 (MTP k=1): uniform decodes lead, then the 1-token
    # chunked-prefill tail, then longer prefills.
    num_tokens_per_req = {"tail": 1, "d1": 2, "p1": 100, "d2": 2}
    assert sort_batch_req_ids(num_tokens_per_req, 2) == ["d1", "d2", "tail", "p1"]


def test_spec_decodes_lead_short_prefill_tail():
    # With the fixed ordering, split_decodes_and_prefills classifies the
    # uniform 2-token decodes as decodes even when a 1-token prefill tail is
    # in the batch (indexer-style: require_uniform, threshold=1+k).
    num_tokens_per_req = {"tail": 1, **{f"d{i}": 2 for i in range(8)}}
    req_ids = sort_batch_req_ids(num_tokens_per_req, 2)
    query_lens = [num_tokens_per_req[r] for r in req_ids]
    assert query_lens == [2] * 8 + [1]

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(
            _make_common_attn_metadata(query_lens),
            decode_threshold=2,
            require_uniform=True,
        )
    )
    assert (num_decodes, num_prefills) == (8, 1)
    assert (num_decode_tokens, num_prefill_tokens) == (16, 1)


def test_lifecycle_order_prevents_short_prefill_between_decode_rows():
    # Regression fixture from the fatal mixed Document Review wave. Shape-only
    # sorting put the 4-token prompt chunk between 4-token target decode rows.
    num_tokens_per_req = {
        "decode-a": 4,
        "prefill-10": 10,
        "prefill-183": 183,
        "prefill-4": 4,
        "prefill-186": 186,
        "prefill-81": 81,
        "prefill-180": 180,
        "decode-b": 4,
    }
    lifecycle = {
        req_id: req_id.startswith("prefill-") for req_id in num_tokens_per_req
    }

    req_ids = sort_batch_req_ids(
        num_tokens_per_req,
        4,
        is_prefilling_by_req=lifecycle,
    )

    assert req_ids == [
        "decode-a",
        "decode-b",
        "prefill-4",
        "prefill-10",
        "prefill-81",
        "prefill-180",
        "prefill-183",
        "prefill-186",
    ]
    assert [lifecycle[req_id] for req_id in req_ids] == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert set(req_ids) == set(num_tokens_per_req)

    common = _make_common_attn_metadata(
        [num_tokens_per_req[req_id] for req_id in req_ids]
    )
    common.is_prefilling = torch.tensor(
        [lifecycle[req_id] for req_id in req_ids], dtype=torch.bool
    )
    generic = split_decodes_and_prefills(
        common,
        decode_threshold=4,
        require_uniform=True,
    )
    assert _ag2_semantic_prefill_split(common, generic) == (2, 6, 8, 644)

    # The original shape-only order is the negative control: the short prompt
    # row interrupts the decode cohort and must still fail closed in FlashInfer.
    bad_req_ids = sort_batch_req_ids(num_tokens_per_req, 4)
    bad_common = _make_common_attn_metadata(
        [num_tokens_per_req[req_id] for req_id in bad_req_ids]
    )
    bad_common.is_prefilling = torch.tensor(
        [lifecycle[req_id] for req_id in bad_req_ids], dtype=torch.bool
    )
    bad_generic = split_decodes_and_prefills(
        bad_common,
        decode_threshold=4,
        require_uniform=True,
    )
    try:
        _ag2_semantic_prefill_split(bad_common, bad_generic)
    except RuntimeError as exc:
        assert "not a contiguous suffix" in str(exc)
    else:
        raise AssertionError("Crash-shaped non-contiguous lifecycle was accepted")


def test_lifecycle_order_rejects_request_identity_drift():
    num_tokens_per_req = {"decode": 4, "prefill": 64}

    try:
        sort_batch_req_ids(
            num_tokens_per_req,
            4,
            is_prefilling_by_req={"decode": False, "stale": True},
        )
    except ValueError as exc:
        assert "missing=['prefill']" in str(exc)
        assert "extra=['stale']" in str(exc)
    else:
        raise AssertionError("lifecycle/request identity drift was accepted")
