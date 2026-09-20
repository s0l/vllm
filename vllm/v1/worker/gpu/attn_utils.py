# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.ubatch_utils import get_num_ubatches
from vllm.v1.worker.utils import (
    AttentionGroup,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    prepare_kernel_block_sizes,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class AttentionCGSupportInfo:
    min_cg_support: AttentionCGSupport = AttentionCGSupport.ALWAYS
    min_cg_attn_backend: str | None = None

    def narrow(
        self, support: AttentionCGSupport, backend: str | None
    ) -> "AttentionCGSupportInfo":
        """Return an info tightened by ``support`` if it is more restrictive.

        Lets attention groups built outside ``init_attn_backend`` (e.g.
        encoder-only layers) contribute to the runner's cudagraph decision.
        """
        if support.value < self.min_cg_support.value:
            return AttentionCGSupportInfo(support, backend)
        return self


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = cast(type[Any], AttentionLayerBase)
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)
    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            # This layer will use KV cache of the sharing target layer.
            continue
        # Skip modules that don't need KV cache (eg encoder-only attention)
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            if isinstance(spec, AttentionSpec):
                spec = attn_module.get_attn_backend().customize_spec(spec)
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec


def get_shared_kv_cache_layers(vllm_config: VllmConfig):
    attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
    return {
        layer_name: kv_tgt_layer
        for layer_name, attn_module in attn_layers.items()
        if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name)
    }


def add_kv_sharing_layers_to_config(
    kv_cache_config: KVCacheConfig, vllm_config: VllmConfig
) -> None:
    add_kv_sharing_layers_to_kv_cache_groups(
        get_shared_kv_cache_layers(vllm_config), kv_cache_config.kv_cache_groups
    )


def init_attn_backend(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
    active_layer_names: set[str] | None = None,
) -> tuple[list[list[AttentionGroup]], AttentionCGSupportInfo, list[int]]:
    # Phase 1: discover attention groups for each kv cache group.
    attn_groups: list[list[AttentionGroup]] = []

    # Add KV-sharing layers to their target's kv cache group so they are
    # discovered alongside the target layer in Phase 1 below.
    add_kv_sharing_layers_to_config(kv_cache_config, vllm_config)

    # Phase 1: discover attention groups for each kv cache group.
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
        kv_cache_config.kv_cache_groups
    ):
        layer_names = kv_cache_group_spec.layer_names
        if active_layer_names is not None:
            layer_names = list(active_layer_names.intersection(layer_names))

        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)

        group_map: dict[tuple[tuple[str, str], KVCacheSpec, int], AttentionGroup] = {}
        group_order: list[tuple[tuple[str, str], KVCacheSpec, int]] = []

        for layer_name in layer_names:
            attn_backend = attn_layers[layer_name].get_attn_backend()

            layer_kv_cache_spec: KVCacheSpec = kv_cache_group_spec.kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]

            # Split on per-rank num_heads_q so layers with different Q-head
            # counts (e.g. a spec-decode draft head and its target) get separate
            # metadata builders.
            num_heads_q = getattr(attn_layers[layer_name], "num_heads", 0)
            key = (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q)
            if key not in group_map:
                group_map[key] = AttentionGroup(
                    attn_backend, [layer_name], layer_kv_cache_spec, kv_cache_group_id
                )
                group_order.append(key)
            else:
                group_map[key].layer_names.append(layer_name)

        attn_groups.append([group_map[key] for key in group_order])

    # Phase 2: pick a kernel block size per kv cache group that is supported
    # by all backends within that group.
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)

    # Phase 3: create metadata builders and determine cudagraph support.
    attn_backend_workspace = None
    for kv_cache_group_id, groups in enumerate(attn_groups):
        kernel_block_size = None
        if kv_cache_group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[kv_cache_group_id]
        for group in groups:
            group.create_metadata_builders(
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=kernel_block_size,
                # Microbatches build attention metadata concurrently, and some
                # builders keep the prepared metadata on themselves (MLA stores
                # it on the prefill backend), so each ubatch needs its own.
                num_metadata_builders=get_num_ubatches(vllm_config.parallel_config),
            )
            # The microbatches' builders share the workspace: they all issue
            # attention on the one compute stream the threads hand off, so the
            # buffer is written serially, as it already is across steps.
            for builder in group.metadata_builders:
                if hasattr(builder, "share_persistent_kernel_scratch_from"):
                    continue
                if attn_backend_workspace is None:
                    if hasattr(builder, "_get_workspace_buffer"):
                        attn_backend_workspace = builder._get_workspace_buffer()
                elif hasattr(builder, "set_workspace_buffer"):
                    builder.set_workspace_buffer(attn_backend_workspace)

    # Attention groups execute sequentially in V2. Select the largest
    # compatible owner before binding scratch so group order cannot produce an
    # undersized shared workspace. Metadata buffers remain group-private.
    shareable_builders: list[Any] = [
        builder
        for groups in attn_groups
        for group in groups
        for builder in group.metadata_builders
        if hasattr(
            builder,
            "share_persistent_kernel_scratch_from",
        )
    ]
    if shareable_builders:
        owner = max(
            shareable_builders,
            key=lambda builder: builder.get_workspace_buffer_size(),
        )
        owner._get_workspace_buffer()
        for builder in shareable_builders:
            if builder is not owner:
                builder.share_persistent_kernel_scratch_from(owner)
        gc.collect()
        torch.accelerator.empty_cache()
    attn_cg_support_info = get_attn_cg_support(attn_groups, vllm_config)
    return attn_groups, attn_cg_support_info, kernel_block_sizes


def get_attn_cg_support(
    attn_groups: list[list[AttentionGroup]],
    vllm_config: VllmConfig,
    checked_layer_names: set[str] | None = None,
) -> AttentionCGSupportInfo:
    """Return the weakest CUDA graph support among the checked layers."""
    min_cg_support = AttentionCGSupport.ALWAYS
    min_cg_attn_backend = None
    for groups in attn_groups:
        for group in groups:
            if checked_layer_names is not None and checked_layer_names.isdisjoint(
                group.layer_names
            ):
                continue
            builder = group.get_metadata_builder(0)
            cg_support = builder.get_cudagraph_support(
                vllm_config,
                group.kv_cache_spec,
            )
            if cg_support.value < min_cg_support.value:
                min_cg_support = cg_support
                min_cg_attn_backend = group.backend.__name__
    return AttentionCGSupportInfo(
        min_cg_support=min_cg_support,
        min_cg_attn_backend=min_cg_attn_backend,
    )


def get_query_lens_mismatch_unsupported_backend(
    attn_groups: list[list[AttentionGroup]],
    checked_layer_names: set[str] | None = None,
) -> str | None:
    """Name the first backend needing the CPU query lengths to be exact, if any.

    The attention selector already excludes these when adaptive verification is
    enabled, but models that hard-wire their backend never consult it. See
    AttentionBackend.supports_device_cpu_query_lens_mismatch().
    """
    for groups in attn_groups:
        for group in groups:
            if checked_layer_names is not None and checked_layer_names.isdisjoint(
                group.layer_names
            ):
                continue
            if not group.backend.supports_device_cpu_query_lens_mismatch():
                return group.backend.__name__
    return None


def _allocate_kv_cache(
    kv_cache_config: KVCacheConfig,
    shared_layers: dict[str, str],
    device: torch.device,
    layout: KVCacheLayout,
    kernel_block_sizes: list[int],
    elastic_backings: dict[str, Any] | None = None,
    elastic_geometry: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Allocate Exp22 backings and view them through the current layout API.

    ``KVCacheLayout`` and ``create_kv_cache_views`` remain authoritative for
    physical strides. Exp22 extends only backing ownership with elastic VMM
    and a separate GDN pool; it must not revive removed backend shape APIs.
    """
    kv_caches: dict[str, torch.Tensor] = {}
    packed_backings: dict[str, torch.Tensor] = {}
    owners = elastic_backings if elastic_backings is not None else {}
    has_separate_pool = any(
        isinstance(group.kv_cache_spec, MambaSpec) and group.kv_cache_spec.separate_pool
        for group in kv_cache_config.kv_cache_groups
    )

    def resolve_backing_id(tensor_index: int) -> str:
        tensor = kv_cache_config.kv_cache_tensors[tensor_index]
        if tensor.backing_id:
            return tensor.backing_id
        if has_separate_pool:
            return f"separate-{tensor_index}"
        return "upstream-shared"

    # CUDA VMM owners use independent custom MemPools and cannot consume
    # cached blocks held by PyTorch's default allocator. Allocate independent
    # owners largest-physical-commit first so a large arena is never stranded
    # behind smaller mappings. View/binding order below remains unchanged.
    elastic_plan: dict[str, tuple[int, int, int]] = {}
    for tensor_index, tensor in enumerate(kv_cache_config.kv_cache_tensors):
        if not tensor.mapping_quantum:
            continue
        backing_id = resolve_backing_id(tensor_index)
        geometry = (tensor.size, tensor.committed_size, tensor.mapping_quantum)
        previous = elastic_plan.setdefault(backing_id, geometry)
        if previous != geometry:
            raise ValueError(
                f"KV backing {backing_id!r} has inconsistent elastic geometry: "
                f"{previous} and {geometry}"
            )

    if elastic_plan:
        from vllm.device_allocator.elastic_cumem import allocate_elastic_backing

        if device.type == "cuda":
            torch.accelerator.synchronize()
            gc.collect()
            torch.accelerator.empty_cache()
        for backing_id, (reserved, committed, quantum) in sorted(
            elastic_plan.items(),
            key=lambda item: (-item[1][1], -item[1][0], item[0]),
        ):
            if backing_id not in owners:
                owners[backing_id] = allocate_elastic_backing(
                    reserved_bytes=reserved,
                    committed_bytes=committed,
                    quantum_bytes=quantum,
                    device=device,
                )

    for tensor_index, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
        backing_id = resolve_backing_id(tensor_index)

        if kv_cache_tensor.mapping_quantum:
            owner = owners[backing_id]
            raw = owner.tensor.view(torch.int8)
            if raw.numel() != kv_cache_tensor.size:
                raise ValueError(
                    f"KV backing {backing_id!r} has inconsistent reserved size: "
                    f"{raw.numel()} and {kv_cache_tensor.size}"
                )
            if elastic_geometry is not None:
                block_geometry = elastic_geometry.setdefault(
                    backing_id, kv_cache_tensor.logical_block_size
                )
                if block_geometry != kv_cache_tensor.logical_block_size:
                    raise ValueError(
                        f"KV backing {backing_id!r} has inconsistent geometry: "
                        f"{block_geometry} and {kv_cache_tensor.logical_block_size}"
                    )
            previous = packed_backings.setdefault(backing_id, raw)
            if previous.data_ptr() != raw.data_ptr():
                raise ValueError(f"KV backing id {backing_id!r} is not unique")
        else:
            raw = packed_backings.get(backing_id)
            if raw is None:
                raw = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
                packed_backings[backing_id] = raw
            elif raw.numel() != kv_cache_tensor.size:
                raise ValueError(
                    f"KV backing {backing_id!r} has inconsistent sizes: "
                    f"{raw.numel()} and {kv_cache_tensor.size}"
                )

        # A zero layer stride identifies the Exp22 descriptor. Its
        # ``shared_by`` list names cache groups overlaid on one physical slot;
        # current ``layers`` instead names distinct L-axis slices. Materialize
        # each legacy alias as an independent one-layer view of the same bytes.
        legacy_aliases = kv_cache_tensor.layer_stride == 0
        view_layer_sets = (
            ([layer_name] for layer_name in kv_cache_tensor.layers)
            if legacy_aliases
            else (kv_cache_tensor.layers,)
        )
        for view_layers in view_layer_sets:
            layer_name = view_layers[0]
            group_id, group = next(
                (group_id, group)
                for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
                if layer_name in group.layer_names
            )
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[layer_name]

            num_blocks = kv_cache_tensor.num_blocks or kv_cache_config.num_blocks
            if isinstance(spec, MambaSpec) and spec.separate_pool:
                # Profiling allocates a transient fixed pool while the spec
                # retains the production ceiling. The descriptor's physical
                # extent is authoritative for this view; elastic production
                # descriptors publish an explicit logical block count.
                physical_block_stride = (
                    kv_cache_tensor.block_stride or spec.page_size_bytes
                )
                if kv_cache_tensor.num_blocks:
                    num_blocks = kv_cache_tensor.num_blocks
                else:
                    num_blocks, remainder = divmod(
                        kv_cache_tensor.size, physical_block_stride
                    )
                    if remainder:
                        raise ValueError(
                            "separate Mamba backing is not an integer number of "
                            f"blocks: size={kv_cache_tensor.size} "
                            f"stride={physical_block_stride}"
                        )
            kernel_block_size = (
                kernel_block_sizes[group_id]
                if group_id < len(kernel_block_sizes)
                else None
            )
            view_layer_stride = cast(
                Any, None if legacy_aliases else kv_cache_tensor.layer_stride
            )
            view_config = replace(
                kv_cache_tensor,
                layers=view_layers,
                shared_by=view_layers,
                layer_stride=view_layer_stride,
                block_stride=(
                    kv_cache_tensor.block_stride or spec.page_size_bytes
                    if legacy_aliases
                    else kv_cache_tensor.block_stride
                ),
            )
            views = create_kv_cache_views(
                raw,
                spec,
                num_blocks,
                layout,
                view_config,
                kernel_block_size=kernel_block_size,
            )
            kv_caches.update(zip(view_layers, views, strict=True))

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == (kv_caches.keys() | shared_layers.keys()), (
        "Some layers are not correctly initialized"
    )
    return kv_caches


def init_kv_cache(
    runner_kv_caches: list[torch.Tensor | list[torch.Tensor]],
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list[AttentionGroup]],
    device: torch.device,
    cache_dtype: str,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
    elastic_backings: dict[str, Any] | None = None,
    elastic_geometry: dict[str, int] | None = None,
    kv_cache_allocation_context: AbstractContextManager | None = None,
) -> dict[str, Any]:
    shared_kv_cache_layers = get_shared_kv_cache_layers(vllm_config)
    allocation_context = kv_cache_allocation_context or nullcontext()
    with allocation_context:
        kv_caches = _allocate_kv_cache(
            kv_cache_config,
            shared_kv_cache_layers,
            device,
            vllm_config.cache_config.get_resolved_kv_cache_layout(),
            kernel_block_sizes,
            elastic_backings,
            elastic_geometry,
        )
    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]
    # Dual-attention models (e.g. LongCat-Flash) put two Attention modules per
    # decoder layer, so a layer name carries two integers (layer + module index).
    num_attn_module = (
        2
        if vllm_config.model_config.hf_config.model_type
        in ("longcat_flash", "longcat_flash_ngram")
        else 1
    )
    bind_kv_cache(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module,
        kv_cache_groups=kv_cache_config.kv_cache_groups,
    )
    return kv_caches


def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    slot_mappings_by_layer: dict[str, torch.Tensor] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mapping
    return slot_mappings_by_layer


def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    dcp_local_seq_lens: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    is_prefilling: torch.Tensor | None = None,
    request_ids: tuple[str | None, ...] | None = None,
    num_scheduled_tokens_cpu: torch.Tensor | None = None,
    num_computed_tokens_provenance_cpu: torch.Tensor | None = None,
    num_prompt_tokens_cpu: torch.Tensor | None = None,
    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
    full_cudagraph: bool = False,
    causal: bool | torch.Tensor | Mapping[int, bool] = True,
    rswa_prefix_lens: torch.Tensor | None = None,
    ubatch_idx: int = 0,
) -> dict[str, Any]:
    seq_lens = seq_lens[:num_reqs]
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]
    if seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound[:num_reqs]

    attn_metadata: dict[str, Any] = {}
    # V2 hybrid layouts can have one KV-cache group per layer.  Building the
    # same backend/spec metadata for every group repeats CPU classification,
    # pinned allocations and H2D copies.  Reuse the first build when a backend
    # explicitly supplies the block-table rebinding contract.
    cached_attn_metadata: dict[tuple[KVCacheSpec, type[Any]], Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]
        # Per-group causal for hybrid drafters (mixed SWA/full attention).
        group_causal = (
            causal if isinstance(causal, (bool, torch.Tensor)) else causal.get(i, True)
        )

        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        # Model-specific metadata (e.g. Mamba hybrid) may supply its own
        # padding-aware is_prefilling, which takes precedence over the default.
        group_is_prefilling = common_attn_metadata_extra_kwargs.pop(
            "is_prefilling", is_prefilling
        )
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            full_cudagraph=full_cudagraph,
            causal=group_causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=positions,
            is_prefilling=group_is_prefilling,
            request_ids=request_ids,
            num_scheduled_tokens_cpu=num_scheduled_tokens_cpu,
            num_computed_tokens_provenance_cpu=num_computed_tokens_provenance_cpu,
            num_prompt_tokens_cpu=num_prompt_tokens_cpu,
            mm_req_doc_ranges=mm_req_doc_ranges,
            rswa_prefix_lens=rswa_prefix_lens,
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(ubatch_idx)
            kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(attn_metadata_builder))
            if for_cudagraph_capture:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and attn_metadata_builder.supports_update_block_table
                and getattr(
                    attn_metadata_builder,
                    "can_update_block_table",
                    lambda metadata: True,
                )(cached_attn_metadata[cache_key])
            ):
                metadata = attn_metadata_builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder,
                        num_reqs,
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
                if attn_metadata_builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = metadata
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata


def compute_mm_prefix_ranges(
    req_ids: list[str],
    mm_features: dict[str, list[MultiModalFeatureSpec]],
    sliding_window: int | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """Compute PrefixLM bidirectional ranges for multimodal tokens.

    Ranges exceeding sliding_window are skipped to prevent early tokens
    from attending across the entire image span.
    """
    req_doc_ranges: dict[int, list[tuple[int, int]]] = {}
    for req_idx, req_id in enumerate(req_ids):
        image_doc_ranges = []
        for mm_feature in mm_features.get(req_id, ()):
            if mm_feature.modality not in ("image", "video"):
                continue
            for r in mm_feature.mm_position.extract_embeds_range():
                if sliding_window is not None and (r[1] - r[0] + 1) > sliding_window:
                    continue
                image_doc_ranges.append(r)
        req_doc_ranges[req_idx] = image_doc_ranges
    return req_doc_ranges
