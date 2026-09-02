# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends import flashinfer


def test_dcp_kv_history_pack_follows_pages_and_trims_tail():
    # Four physical pages, two tokens per page, one head with width three.
    key_cache = torch.arange(24, dtype=torch.uint8).view(4, 2, 1, 3)
    value_cache = key_cache + 40
    # Request zero consumes physical pages 2,0; request one consumes 1,3.
    # Rows 1 and 3 are speculative siblings and deliberately have no pages.
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=torch.tensor([0, 2, 2, 4, 4], dtype=torch.int32),
        _paged_kv_indices_buf=torch.tensor([2, 0, 1, 3], dtype=torch.int32),
        _paged_kv_last_page_len_buf=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
    )
    decode = SimpleNamespace(wrapper=wrapper)
    history = torch.full((2, 2, 4, 1, 3), 255, dtype=torch.uint8)
    meta = torch.empty((2, 4), dtype=torch.int32)
    pages = torch.empty((2, 2), dtype=torch.int32)
    observer = torch.empty((2, 24), dtype=torch.int32)

    flashinfer._write_ag2_dcp_kv_history_pack(
        SimpleNamespace(),
        (key_cache, value_cache),
        decode,
        "NHD",
        history,
        meta,
        pages,
        observer,
        request_count=2,
        request_row_stride=2,
    )

    expected_key_0 = torch.cat((key_cache[2], key_cache[0, :1]), dim=0)
    expected_key_1 = torch.cat((key_cache[1], key_cache[3, :1]), dim=0)
    expected_value_0 = torch.cat((value_cache[2], value_cache[0, :1]), dim=0)
    expected_value_1 = torch.cat((value_cache[1], value_cache[3, :1]), dim=0)
    torch.testing.assert_close(history[0, 0, :3], expected_key_0)
    torch.testing.assert_close(history[1, 0, :3], expected_key_1)
    torch.testing.assert_close(history[0, 1, :3], expected_value_0)
    torch.testing.assert_close(history[1, 1, :3], expected_value_1)
    torch.testing.assert_close(history[:, :, 3], torch.zeros_like(history[:, :, 3]))
    torch.testing.assert_close(
        meta,
        torch.tensor([[0, 3, 2, 1], [2, 3, 2, 1]], dtype=torch.int32),
    )


def test_dcp_kv_history_pack_publishes_invalid_sentinel_for_empty_pages():
    wrapper = SimpleNamespace(
        _paged_kv_indptr_buf=torch.tensor([0, 0], dtype=torch.int32),
        _paged_kv_indices_buf=torch.empty(0, dtype=torch.int32),
        _paged_kv_last_page_len_buf=torch.tensor([0], dtype=torch.int32),
    )
    history = torch.full((1, 2, 1, 1, 1), 17, dtype=torch.uint8)
    meta = torch.full((1, 4), 19, dtype=torch.int32)
    pages = torch.full((1, 1), 19, dtype=torch.int32)
    observer = torch.full((1, 24), 19, dtype=torch.int32)
    flashinfer._write_ag2_dcp_kv_history_pack(
        SimpleNamespace(),
        (torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8)),
        SimpleNamespace(wrapper=wrapper),
        "NHD",
        history,
        meta,
        pages,
        observer,
        request_count=1,
        request_row_stride=1,
    )
    torch.testing.assert_close(history, torch.zeros_like(history))
    torch.testing.assert_close(meta, torch.full_like(meta, -1))
    torch.testing.assert_close(pages, torch.full_like(pages, -1))
    torch.testing.assert_close(observer, torch.full_like(observer, -1))


def test_dcp_request_trace_indices_select_first_or_tail():
    qo_indptr = torch.tensor([0, 4, 10], dtype=torch.int32)
    torch.testing.assert_close(
        flashinfer._ag2_dcp_request_trace_indices(qo_indptr, "first"),
        torch.tensor([0, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        flashinfer._ag2_dcp_request_trace_indices(qo_indptr, "tail"),
        torch.tensor([3, 9], dtype=torch.int32),
    )
    try:
        flashinfer._ag2_dcp_request_trace_indices(qo_indptr, "middle")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid request-row mode must fail closed")


def test_dcp_prefill_kv_history_selects_only_qlen4_requests():
    key_cache = torch.arange(36, dtype=torch.uint8).view(6, 2, 1, 3)
    value_cache = key_cache + 80
    # Request 0 is K3 target verification (qlen4); request 1 is an unrelated
    # six-token prefill. Only request 0 may populate the observer pack.
    qo_indptr = torch.tensor([0, 4, 10], dtype=torch.int32)
    paged_indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    page_indices = torch.tensor([4, 1, 5, 2], dtype=torch.int32)
    last_page_len = torch.tensor([1, 2], dtype=torch.int32)
    history = torch.full((2, 2, 4, 1, 3), 255, dtype=torch.uint8)
    meta = torch.empty((2, 4), dtype=torch.int32)
    pages = torch.empty((2, 4), dtype=torch.int32)
    observer = torch.empty((2, 24), dtype=torch.int32)

    flashinfer._write_ag2_dcp_prefill_kv_history_pack(
        (key_cache, value_cache),
        kv_layout="NHD",
        qo_indptr_cpu=qo_indptr,
        paged_kv_indptr_cpu=paged_indptr,
        paged_kv_indices=page_indices,
        paged_kv_last_page_len_cpu=last_page_len,
        history_pack=history,
        history_meta=meta,
        page_indices_pack=pages,
        observer_meta=observer,
        use_cuda_graph=False,
        request_row="first",
    )

    expected_key = torch.cat((key_cache[4], key_cache[1, :1]), dim=0)
    expected_value = torch.cat((value_cache[4], value_cache[1, :1]), dim=0)
    torch.testing.assert_close(history[0, 0, :3], expected_key)
    torch.testing.assert_close(history[0, 1, :3], expected_value)
    torch.testing.assert_close(history[0, :, 3], torch.zeros_like(history[0, :, 3]))
    torch.testing.assert_close(history[1], torch.zeros_like(history[1]))
    torch.testing.assert_close(
        meta,
        torch.tensor([[0, 3, 2, 1], [-1, -1, -1, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(pages[0, :2], torch.tensor([4, 1], dtype=torch.int32))
    torch.testing.assert_close(pages[0, 2:], torch.full((2,), -1, dtype=torch.int32))
    assert observer[0, 0].item() == flashinfer.AG2_DCP_OBSERVER_SCHEMA_VERSION
    assert observer[0, 1].item() == (
        flashinfer.AG2_DCP_OBSERVER_PLAN_BIT | flashinfer.AG2_DCP_OBSERVER_HISTORY_BIT
    )
    assert observer[0, 2].item() == 1
    assert observer[0, 3].item() == 0
    assert observer[0, 5].item() == 4
    assert observer[0, 11].item() == 10
    assert observer[1, 2].item() == 0


def test_dcp_prefill_kv_history_keeps_all_qlen4_request_rows():
    key_cache = torch.arange(24, dtype=torch.uint8).view(4, 2, 1, 3)
    value_cache = key_cache + 40
    history = torch.empty((2, 2, 4, 1, 3), dtype=torch.uint8)
    meta = torch.empty((2, 4), dtype=torch.int32)
    pages = torch.empty((2, 4), dtype=torch.int32)
    observer = torch.empty((2, 24), dtype=torch.int32)

    flashinfer._write_ag2_dcp_prefill_kv_history_pack(
        (key_cache, value_cache),
        kv_layout="NHD",
        qo_indptr_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        paged_kv_indptr_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        paged_kv_indices=torch.tensor([2, 0, 1, 3], dtype=torch.int32),
        paged_kv_last_page_len_cpu=torch.tensor([1, 1], dtype=torch.int32),
        history_pack=history,
        history_meta=meta,
        page_indices_pack=pages,
        observer_meta=observer,
        use_cuda_graph=False,
        request_row="first",
    )

    torch.testing.assert_close(
        meta,
        torch.tensor([[0, 3, 2, 1], [4, 3, 2, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        history[1, 0, :3], torch.cat((key_cache[1], key_cache[3, :1]), dim=0)
    )


def test_dcp_prefill_observer_fails_closed_above_narrow_capacity():
    request_count = 6
    observer_rows = 5
    history = torch.full((observer_rows, 2, 4, 1, 3), 255, dtype=torch.uint8)
    meta = torch.zeros((observer_rows, 4), dtype=torch.int32)
    pages = torch.zeros((observer_rows, 4), dtype=torch.int32)
    observer = torch.zeros((observer_rows, 24), dtype=torch.int32)

    flashinfer._write_ag2_dcp_prefill_kv_history_pack(
        (
            torch.empty((1, 2, 1, 3), dtype=torch.uint8),
            torch.empty((1, 2, 1, 3), dtype=torch.uint8),
        ),
        kv_layout="NHD",
        qo_indptr_cpu=torch.arange(0, 4 * request_count + 1, 4, dtype=torch.int32),
        paged_kv_indptr_cpu=torch.zeros(request_count + 1, dtype=torch.int32),
        paged_kv_indices=torch.empty(0, dtype=torch.int32),
        paged_kv_last_page_len_cpu=torch.zeros(request_count, dtype=torch.int32),
        history_pack=history,
        history_meta=meta,
        page_indices_pack=pages,
        observer_meta=observer,
        use_cuda_graph=False,
        request_row="first",
    )

    assert torch.count_nonzero(history).item() == 0
    torch.testing.assert_close(meta, torch.full_like(meta, -1))
    torch.testing.assert_close(pages, torch.full_like(pages, -1))
    torch.testing.assert_close(observer, torch.full_like(observer, -1))


def test_dcp_observer_stage_completeness(monkeypatch):
    monkeypatch.setattr(
        flashinfer,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    layer = SimpleNamespace(num_heads=1, head_size=3)
    indices = torch.tensor([0, 4], dtype=torch.int32)
    output_pack = torch.full((2, 12), torch.nan, dtype=torch.float32)
    lse_pack = torch.full((2, 3), torch.nan, dtype=torch.float32)
    observer = torch.zeros((2, 24), dtype=torch.int32)
    observer[:, 1] = (
        flashinfer.AG2_DCP_OBSERVER_PLAN_BIT
        | flashinfer.AG2_DCP_OBSERVER_HISTORY_BIT
        | flashinfer.AG2_DCP_OBSERVER_CURRENT_KV_BIT
        | flashinfer.AG2_DCP_OBSERVER_SCALES_BIT
    )

    for stage in (
        "local_output",
        "combined_output",
        "query_output",
        "merged_output",
    ):
        flashinfer._write_ag2_dcp_aux_pack(
            layer,
            stage,
            torch.arange(24, dtype=torch.float32).view(8, 1, 3),
            output_pack,
            lse_pack,
            request_tail_indices=indices,
            request_tail_count=2,
            dcp_observer_meta=observer,
        )
    for stage in ("local_lse", "combined_lse", "query_lse"):
        flashinfer._write_ag2_dcp_aux_pack(
            layer,
            stage,
            torch.arange(8, dtype=torch.float32).view(8, 1),
            output_pack,
            lse_pack,
            request_tail_indices=indices,
            request_tail_count=2,
            dcp_observer_meta=observer,
        )

    torch.testing.assert_close(
        observer[:, 1],
        torch.full(
            (2,),
            flashinfer.AG2_DCP_OBSERVER_COMPLETE_MASK,
            dtype=torch.int32,
        ),
    )
    assert torch.isfinite(output_pack).all()
    assert torch.isfinite(lse_pack).all()
