# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flashinfer import (
    _ag2_resolve_canonical_prefill_route,
    _ag2_semantic_prefill_split,
    _dcp_causal_paged_custom_mask,
    _flashinfer_seq_lens_and_blocks_for_paged_kv,
    _physicalize_empty_dcp_decode_rows,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.dcp import mask_dcp_empty_shards_


def test_empty_dcp_decode_rows_receive_one_nonsemantic_page() -> None:
    seq_lens = np.array([7, 0, 0, 9], dtype=np.int32)
    num_blocks = np.array([1, 0, 0, 1], dtype=np.int32)

    physical_lens, physical_blocks, empty_rows = _physicalize_empty_dcp_decode_rows(
        seq_lens, num_blocks, num_decodes=3
    )

    assert physical_lens.tolist() == [7, 1, 1, 9]
    assert physical_blocks.tolist() == [1, 1, 1, 1]
    assert empty_rows.tolist() == [False, True, True]
    assert seq_lens.tolist() == [7, 0, 0, 9]
    assert num_blocks.tolist() == [1, 0, 0, 1]


def test_empty_dcp_decode_physicalization_does_not_touch_prefill_rows() -> None:
    seq_lens = np.array([5, 0], dtype=np.int32)
    num_blocks = np.array([1, 0], dtype=np.int32)

    physical_lens, physical_blocks, empty_rows = _physicalize_empty_dcp_decode_rows(
        seq_lens, num_blocks, num_decodes=1
    )

    assert physical_lens is seq_lens
    assert physical_blocks is num_blocks
    assert empty_rows.tolist() == [False]


def test_empty_dcp_decode_dummy_page_is_removed_from_consumed_lse() -> None:
    lse = torch.tensor([[2.0, 3.0], [5.0, 7.0]])

    mask_dcp_empty_shards_(
        lse,
        seq_lens=torch.tensor([7, 0], dtype=torch.int32),
        query_start_loc=None,
    )

    assert torch.equal(lse[0], torch.tensor([2.0, 3.0]))
    assert torch.isneginf(lse[1]).all()


def _common_metadata(
    query_lens: list[int], is_prefilling: list[bool]
) -> CommonAttentionMetadata:
    starts = [0]
    for query_len in query_lens:
        starts.append(starts[-1] + query_len)
    num_reqs = len(query_lens)
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        seq_lens=torch.tensor(query_lens, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=starts[-1],
        max_query_len=max(query_lens),
        max_seq_len=max(query_lens),
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(starts[-1], dtype=torch.int64),
        is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool),
    )


def _owned_global_positions(
    seq_len: int, rank: int, world: int, interleave: int
) -> list[int]:
    return [
        position
        for position in range(seq_len)
        if (position // interleave) % world == rank
    ]


@pytest.mark.parametrize("world,interleave", [(3, 1), (3, 2), (4, 4)])
def test_canonical_paged_mask_matches_global_causal_ownership(
    world: int, interleave: int
) -> None:
    seq_lens = torch.tensor([17, 64, 1344], dtype=torch.int32)
    query_lens = [7, 31, 896]
    qo_indptr = torch.tensor(
        [0, query_lens[0], sum(query_lens[:2]), sum(query_lens)],
        dtype=torch.int32,
    )

    for rank in range(world):
        flat_mask, local_seq_lens = _dcp_causal_paged_custom_mask(
            seq_lens, qo_indptr, world, rank, interleave
        )
        offset = 0
        for seq_len, query_len, local_seq_len in zip(
            seq_lens.tolist(), query_lens, local_seq_lens.tolist(), strict=True
        ):
            owned = _owned_global_positions(seq_len, rank, world, interleave)
            actual = flat_mask[offset : offset + query_len * local_seq_len].reshape(
                query_len, local_seq_len
            )
            query_positions = range(seq_len - query_len, seq_len)
            expected = torch.tensor(
                [
                    [key_position <= query_position for key_position in owned]
                    for query_position in query_positions
                ],
                dtype=torch.bool,
            )
            assert torch.equal(actual, expected)
            offset += query_len * local_seq_len
        assert offset == flat_mask.numel()


@pytest.mark.parametrize(
    "boundary",
    [0, 1, 3, 7, 15, 31, 63, 127, 319, 447, 511, 575, 767, 831, 895, 1215, 1279, 1343],
)
def test_canonical_paged_mask_never_exposes_future_key(boundary: int) -> None:
    seq_len = 1344
    query_len = seq_len - boundary
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    qo_indptr = torch.tensor([0, query_len], dtype=torch.int32)

    for rank in range(3):
        flat_mask, local_seq_lens = _dcp_causal_paged_custom_mask(
            seq_lens, qo_indptr, 3, rank, 1
        )
        mask = flat_mask.reshape(query_len, int(local_seq_lens[0]))
        owned = _owned_global_positions(seq_len, rank, 3, 1)
        for row, query_position in enumerate(range(boundary, seq_len)):
            visible = [owned[index] for index in mask[row].nonzero().flatten().tolist()]
            assert all(position <= query_position for position in visible)
            assert visible == [
                position for position in owned if position <= query_position
            ]


@pytest.mark.parametrize(
    "seq_lens,qo_indptr,error",
    [
        ([4], [0, 0], "empty query"),
        ([3], [0, 4], "shorter than"),
        ([4, 5], [0, 4], "one query span"),
    ],
)
def test_canonical_paged_mask_rejects_invalid_metadata(
    seq_lens: list[int], qo_indptr: list[int], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _dcp_causal_paged_custom_mask(
            torch.tensor(seq_lens, dtype=torch.int32),
            torch.tensor(qo_indptr, dtype=torch.int32),
            3,
            0,
            1,
        )


@pytest.mark.parametrize("query_len", [1344, 512, 128])
def test_canonical_paged_metadata_includes_just_written_query(
    query_len: int,
) -> None:
    seq_lens = torch.tensor([1344], dtype=torch.int32)
    qo_indptr = torch.tensor([0, query_len], dtype=torch.int32)

    full_local, full_np, full_blocks = _flashinfer_seq_lens_and_blocks_for_paged_kv(
        seq_lens,
        qo_indptr,
        0,
        1,
        16,
        use_dcp=True,
        dcp_world_size=3,
        dcp_rank=0,
        dcp_kv_cache_interleave_size=1,
        include_prefill_query=True,
    )
    context_local, _, _ = _flashinfer_seq_lens_and_blocks_for_paged_kv(
        seq_lens,
        qo_indptr,
        0,
        1,
        16,
        use_dcp=True,
        dcp_world_size=3,
        dcp_rank=0,
        dcp_kv_cache_interleave_size=1,
        include_prefill_query=False,
    )

    assert int(full_local[0]) == 448
    assert int(full_np[0]) == 448
    assert int(full_blocks[0]) == 28
    assert int(context_local[0]) == (1344 - query_len + 2) // 3


def test_semantic_split_recovers_uniform_short_prompt_extends() -> None:
    metadata = _common_metadata([128, 128, 128], [True, True, True])
    generic = split_decodes_and_prefills(
        metadata, decode_threshold=128, require_uniform=True
    )
    assert generic == (3, 0, 384, 0)
    assert _ag2_semantic_prefill_split(metadata, generic) == (0, 3, 0, 384)


def test_semantic_split_preserves_target_verification_decode() -> None:
    metadata = _common_metadata([4, 4, 4], [False, False, False])
    generic = split_decodes_and_prefills(
        metadata, decode_threshold=4, require_uniform=True
    )
    assert _ag2_semantic_prefill_split(metadata, generic) == generic


def test_semantic_split_preserves_mixed_decode_prefill_order() -> None:
    metadata = _common_metadata([4, 128, 128], [False, True, True])
    generic = split_decodes_and_prefills(
        metadata, decode_threshold=1, require_uniform=True
    )
    assert generic == (0, 3, 0, 260)
    assert _ag2_semantic_prefill_split(metadata, generic) == generic


def test_semantic_split_matches_real_qlen4_plus_new_prompt_wave() -> None:
    metadata = _common_metadata(
        [4, 4, 4, 4, 4, 79], [False, False, False, False, False, True]
    )
    generic = split_decodes_and_prefills(
        metadata, decode_threshold=1, require_uniform=True
    )
    assert generic == (0, 6, 0, 99)
    assert _ag2_semantic_prefill_split(metadata, generic) == generic


def test_semantic_split_rejects_noncontiguous_lifecycle() -> None:
    metadata = _common_metadata([4, 128, 4], [False, True, False])
    generic = split_decodes_and_prefills(
        metadata, decode_threshold=128, require_uniform=True
    )
    with pytest.raises(RuntimeError, match="contiguous suffix"):
        _ag2_semantic_prefill_split(metadata, generic)


def test_canonical_route_overrides_uniform_short_extend_and_cascade() -> None:
    metadata = _common_metadata([128, 128], [True, True])
    generic = split_decodes_and_prefills(
        metadata, decode_threshold=128, require_uniform=True
    )
    semantic, canonical_start, cascade = _ag2_resolve_canonical_prefill_route(
        common_attn_metadata=metadata,
        generic_split=generic,
        common_prefix_len=64,
        enabled=True,
        causal=True,
        use_dcp=True,
        use_dcp_pseudo_decode=False,
    )
    assert semantic == (0, 2, 0, 256)
    assert canonical_start == 0
    assert cascade is False


def test_canonical_route_never_selects_target_verification() -> None:
    metadata = _common_metadata([4, 4], [False, False])
    generic = split_decodes_and_prefills(
        metadata, decode_threshold=4, require_uniform=True
    )
    semantic, canonical_start, _ = _ag2_resolve_canonical_prefill_route(
        common_attn_metadata=metadata,
        generic_split=generic,
        common_prefix_len=0,
        enabled=True,
        causal=True,
        use_dcp=True,
        use_dcp_pseudo_decode=False,
    )
    assert semantic == generic
    assert canonical_start is None


def test_canonical_route_preserves_draft_without_prompt_lifecycle() -> None:
    metadata = _common_metadata([4, 4], [False, False])
    metadata.is_prefilling = None
    generic = (2, 0, 8, 0)
    semantic, canonical_start, cascade = _ag2_resolve_canonical_prefill_route(
        common_attn_metadata=metadata,
        generic_split=generic,
        common_prefix_len=0,
        enabled=True,
        causal=True,
        use_dcp=True,
        use_dcp_pseudo_decode=False,
    )
    assert semantic == generic
    assert canonical_start is None
    assert cascade is False


def test_canonical_route_preserves_computational_prefill_in_real_mixed_wave() -> None:
    metadata = _common_metadata(
        [4, 4, 4, 4, 4, 79], [False, False, False, False, False, True]
    )
    generic = split_decodes_and_prefills(
        metadata, decode_threshold=1, require_uniform=True
    )
    semantic, canonical_start, cascade = _ag2_resolve_canonical_prefill_route(
        common_attn_metadata=metadata,
        generic_split=generic,
        common_prefix_len=0,
        enabled=True,
        causal=True,
        use_dcp=True,
        use_dcp_pseudo_decode=False,
    )
    assert generic == (0, 6, 0, 99)
    assert semantic == generic
    assert canonical_start == 5
    assert cascade is False
