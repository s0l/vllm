# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded-page handling in create_kv_cache_views.

Guards that a page_size_padded spec strides the block dimension by the padded page
while keeping per-block content compact, so padding bytes at the end of each page are
never addressed by the logical view.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    compute_layout_strides,
)
from vllm.v1.worker.gpu.attn_utils import (
    _allocate_kv_cache,
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
)


class _FakeMetadataBuilder:
    def __init__(self, support: AttentionCGSupport):
        self.support = support

    def get_cudagraph_support(self, *_args):
        return self.support


class _TargetBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


class _DraftBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


def test_exp22_allocator_preserves_upstream_shared_layout() -> None:
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    tensor_size = num_blocks * spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, layers=["a"]),
            KVCacheTensor(size=tensor_size, layers=["b"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], spec),
            KVCacheGroupSpec(["b"], spec),
        ],
    )

    caches = _allocate_kv_cache(
        config, {}, torch.device("cpu"), KVCacheLayout.LBHNC, [2, 2]
    )

    assert (
        caches["a"].untyped_storage().data_ptr()
        == caches["b"].untyped_storage().data_ptr()
    )
    assert caches["a"].shape == (2 * num_blocks, 1, 2, 4)


def test_exp22_legacy_shared_by_names_aliases_not_layout_layers() -> None:
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * spec.page_size_bytes,
                shared_by=["a", "b"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], spec),
            KVCacheGroupSpec(["b"], spec),
        ],
    )

    caches = _allocate_kv_cache(
        config, {}, torch.device("cpu"), KVCacheLayout.LBNHC, [2, 2]
    )

    assert caches["a"].shape == (2 * num_blocks, 1, 2, 4)
    assert caches["a"].stride() == caches["b"].stride()
    assert caches["a"].storage_offset() == caches["b"].storage_offset()
    assert (
        caches["a"].untyped_storage().data_ptr()
        == caches["b"].untyped_storage().data_ptr()
    )


def test_exp22_profile_shape_splits_manager_block_into_39_kernel_blocks() -> None:
    spec = FullAttentionSpec(
        block_size=2496,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        state_content_bytes=1,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=2 * spec.page_size_bytes, shared_by=["attn"])
        ],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], spec)],
    )

    caches = _allocate_kv_cache(
        config, {}, torch.device("cpu"), KVCacheLayout.LBNHC, [64]
    )

    assert caches["attn"].shape == (78, 1, 64, 1)
    assert caches["attn"].stride(0) == 64


def test_exp22_allocator_keeps_separate_gdn_pool_independent() -> None:
    num_blocks = 3
    attention = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    mamba = MambaSpec(
        block_size=1,
        shapes=((4,),),
        dtypes=(torch.uint8,),
        separate_pool=True,
        separate_pool_num_blocks=5,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * attention.page_size_bytes,
                layers=["attn"],
            ),
            KVCacheTensor(size=5 * mamba.page_size_bytes, layers=["gdn"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["gdn"], mamba),
        ],
    )

    caches = _allocate_kv_cache(
        config, {}, torch.device("cpu"), KVCacheLayout.LBHNC, [2, 1]
    )

    assert caches["gdn"].shape == (5, 1, 1, 4)
    assert (
        caches["attn"].untyped_storage().data_ptr()
        != caches["gdn"].untyped_storage().data_ptr()
    )


def test_exp22_profile_mamba_view_uses_64_allocated_not_193_spec_blocks() -> None:
    mamba = MambaSpec(
        block_size=1,
        shapes=((4,),),
        dtypes=(torch.uint8,),
        separate_pool=True,
        separate_pool_num_blocks=193,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=64 * mamba.page_size_bytes, shared_by=["gdn"])
        ],
        kv_cache_groups=[KVCacheGroupSpec(["gdn"], mamba)],
    )

    caches = _allocate_kv_cache(
        config, {}, torch.device("cpu"), KVCacheLayout.LBNHC, [1]
    )

    assert caches["gdn"].shape == (64, 1, 1, 4)


def test_exp22_profile_mamba_view_rejects_partial_physical_block() -> None:
    mamba = MambaSpec(
        block_size=1,
        shapes=((4,),),
        dtypes=(torch.uint8,),
        separate_pool=True,
        separate_pool_num_blocks=193,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(size=257, shared_by=["gdn"])],
        kv_cache_groups=[KVCacheGroupSpec(["gdn"], mamba)],
    )

    with pytest.raises(ValueError, match="not an integer number"):
        _allocate_kv_cache(config, {}, torch.device("cpu"), KVCacheLayout.LBNHC, [1])


def test_exp22_allocator_views_shared_gdn_backing_by_page() -> None:
    mamba = MambaSpec(
        block_size=1,
        shapes=((4,),),
        dtypes=(torch.uint8,),
        separate_pool=True,
        separate_pool_num_blocks=5,
    )
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=40,
                layers=["gdn.0"],
                offset=0,
                block_stride=8,
                backing_id="gdn",
                num_blocks=5,
                logical_block_size=8,
            ),
            KVCacheTensor(
                size=40,
                layers=["gdn.1"],
                offset=4,
                block_stride=8,
                backing_id="gdn",
                num_blocks=5,
                logical_block_size=8,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["gdn.0"], mamba),
            KVCacheGroupSpec(["gdn.1"], mamba),
        ],
    )

    caches = _allocate_kv_cache(
        config, {}, torch.device("cpu"), KVCacheLayout.LBHNC, [1, 1]
    )

    assert (
        caches["gdn.0"].untyped_storage().data_ptr()
        == caches["gdn.1"].untyped_storage().data_ptr()
    )

    assert caches["gdn.0"].stride(0) == 8
    assert caches["gdn.1"].storage_offset() == 4


def test_exp22_allocator_reuses_elastic_owner_and_records_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import elastic_cumem

    calls: list[tuple[int, int, int, torch.device]] = []

    def fake_allocate(
        *,
        reserved_bytes: int,
        committed_bytes: int,
        quantum_bytes: int,
        device: torch.device,
    ) -> SimpleNamespace:
        calls.append((reserved_bytes, committed_bytes, quantum_bytes, device))
        return SimpleNamespace(tensor=torch.zeros(reserved_bytes, dtype=torch.int8))

    monkeypatch.setattr(elastic_cumem, "allocate_elastic_backing", fake_allocate)
    mamba = MambaSpec(
        block_size=1,
        shapes=((4,),),
        dtypes=(torch.uint8,),
        separate_pool=True,
        separate_pool_num_blocks=5,
    )
    tensors = [
        KVCacheTensor(
            size=40,
            layers=[layer],
            offset=offset,
            block_stride=8,
            backing_id="elastic-gdn",
            committed_size=16,
            mapping_quantum=8,
            num_blocks=5,
            logical_block_size=8,
        )
        for layer, offset in (("gdn.0", 0), ("gdn.1", 4))
    ]
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            KVCacheGroupSpec(["gdn.0"], mamba),
            KVCacheGroupSpec(["gdn.1"], mamba),
        ],
    )
    owners: dict[str, SimpleNamespace] = {}
    geometry: dict[str, int] = {}

    caches = _allocate_kv_cache(
        config,
        {},
        torch.device("cpu"),
        KVCacheLayout.LBHNC,
        [1, 1],
        owners,
        geometry,
    )

    assert calls == [(40, 16, 8, torch.device("cpu"))]
    assert set(owners) == {"elastic-gdn"}
    assert geometry == {"elastic-gdn": 8}
    assert (
        caches["gdn.0"].untyped_storage().data_ptr()
        == caches["gdn.1"].untyped_storage().data_ptr()
    )

    local_caches = _allocate_kv_cache(
        config, {}, torch.device("cpu"), KVCacheLayout.LBHNC, [1, 1]
    )
    assert len(calls) == 2
    assert (
        local_caches["gdn.0"].untyped_storage().data_ptr()
        == local_caches["gdn.1"].untyped_storage().data_ptr()
    )


def test_exp22_elastic_owners_allocate_largest_commit_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import elastic_cumem

    calls: list[int] = []

    def fake_allocate(
        *,
        reserved_bytes: int,
        committed_bytes: int,
        quantum_bytes: int,
        device: torch.device,
    ) -> SimpleNamespace:
        calls.append(committed_bytes)
        return SimpleNamespace(tensor=torch.zeros(reserved_bytes, dtype=torch.int8))

    monkeypatch.setattr(elastic_cumem, "allocate_elastic_backing", fake_allocate)
    spec = FullAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
    )
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[
            KVCacheTensor(
                size=16,
                layers=["small"],
                backing_id="small",
                committed_size=8,
                mapping_quantum=8,
                num_blocks=1,
                logical_block_size=1,
            ),
            KVCacheTensor(
                size=32,
                layers=["large"],
                backing_id="large",
                committed_size=24,
                mapping_quantum=8,
                num_blocks=1,
                logical_block_size=1,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["small"], spec),
            KVCacheGroupSpec(["large"], spec),
        ],
    )

    _allocate_kv_cache(config, {}, torch.device("cpu"), KVCacheLayout.LBHNC, [1, 1])

    assert calls == [24, 8]


def test_exp22_elastic_owner_rejects_duplicate_geometry() -> None:
    spec = FullAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
    )
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[
            KVCacheTensor(
                size=size,
                layers=[layer],
                backing_id="shared",
                committed_size=size,
                mapping_quantum=8,
                num_blocks=1,
                logical_block_size=1,
            )
            for size, layer in ((16, "a"), (24, "b"))
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], spec),
            KVCacheGroupSpec(["b"], spec),
        ],
    )

    with pytest.raises(ValueError, match="inconsistent elastic geometry"):
        _allocate_kv_cache(config, {}, torch.device("cpu"), KVCacheLayout.LBHNC, [1, 1])


def test_exp22_allocator_rejects_inconsistent_shared_backing_sizes() -> None:
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=64, layers=["a"], backing_id="shared"),
            KVCacheTensor(size=128, layers=["b"], backing_id="shared"),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], spec),
            KVCacheGroupSpec(["b"], spec),
        ],
    )

    with pytest.raises(ValueError, match="inconsistent sizes"):
        _allocate_kv_cache(config, {}, torch.device("cpu"), KVCacheLayout.LBHNC, [2, 2])


def test_attention_checks_preserve_global_and_target_scoped_support():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    target_group = AttentionGroup(
        _TargetBackend,
        ["target"],
        spec,
        0,  # type: ignore[arg-type]
    )
    target_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.ALWAYS)  # type: ignore[list-item]
    ]
    draft_group = AttentionGroup(
        _DraftBackend,
        ["draft"],
        spec,
        0,  # type: ignore[arg-type]
    )
    draft_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.UNIFORM_BATCH)  # type: ignore[list-item]
    ]
    groups = [[target_group, draft_group]]

    # The runner-wide execution mode must still honor the drafter's limit.
    unfiltered = get_attn_cg_support(groups, None)  # type: ignore[arg-type]
    assert unfiltered.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert unfiltered.min_cg_attn_backend == "_DraftBackend"

    # Adaptive verification validates only the target's varlen graphs.
    target_only = get_attn_cg_support(
        groups,
        None,  # type: ignore[arg-type]
        checked_layer_names={"target"},
    )
    assert target_only.min_cg_support == AttentionCGSupport.ALWAYS
    assert target_only.min_cg_attn_backend is None
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        is None
    )

    # Shared target/draft groups still participate in target-scoped checks.
    draft_group.layer_names.append("target")
    target_with_shared_group = get_attn_cg_support(
        groups,
        None,  # type: ignore[arg-type]
        checked_layer_names={"target"},
    )
    assert target_with_shared_group.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        == "_DraftBackend"
    )


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    (
        "kernel_block_sizes",
        "storage_block_size",
        "expected_num_blocks",
        "expected_num_states",
    ),
    [
        (None, None, 4, 64),
        ([256], None, 4, 64),
        ([64], None, 16, 16),
        ([64], 256, 4, 64),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    storage_block_size: int | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        storage_block_size=storage_block_size,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                layers=["layer.0"],
                layer_stride=num_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )

    assert caches["layer.0"].shape == (expected_num_blocks, 1, expected_num_states, 128)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    copy_kv_cache_blocks_inplace(caches, num_blocks, copies)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


def test_fixed_block_stride_propagates_outward_in_lhbnc():
    num_blocks = 3
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    natural = compute_layout_strides(spec, num_blocks, num_layers, KVCacheLayout.LHBNC)
    block_stride = natural[1] + 8

    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        KVCacheLayout.LHBNC,
        fixed_strides=(None, block_stride, None, None, None),
    )

    assert strides[1] == block_stride
    assert strides[2] == block_stride * num_blocks
    assert strides[0] == strides[2] * spec.num_heads


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC stores each head group separately, so a block's bytes are scattered
    # across L*H regions.
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 4,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize(
    "layout,num_layers",
    [
        (KVCacheLayout.LBHNC, 2),
        # Splitting needs a manager block to be one dense page, which a
        # block-outermost layout only gives when the block holds one layer.
        (KVCacheLayout.BLHNC, 1),
    ],
)
def test_copy_kv_cache_blocks_with_virtual_block_splitting(
    layout: KVCacheLayout, num_layers: int
):
    num_blocks = 4
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=spec.block_size // physical_per_logical,
    )

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(cache.shape[0]):
            cache[block_idx].fill_(100 * layer_idx + block_idx)
    expected = [[cache[i].clone() for i in range(cache.shape[0])] for cache in caches]

    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    dst_start = 2 * physical_per_logical
    for layer_idx, cache in enumerate(caches):
        for physical_idx in range(physical_per_logical):
            torch.testing.assert_close(
                cache[dst_start + physical_idx], expected[layer_idx][physical_idx]
            )
