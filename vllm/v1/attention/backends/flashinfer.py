# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from flashinfer.decode import fast_decode_plan, trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor
from typing_extensions import override

from vllm import _custom_ops as custom_ops
from vllm import envs
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    force_use_trtllm_attention,
    supports_trtllm_attention,
    use_trtllm_attention,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
    is_strictly_contiguous,
    nvfp4_kv_cache_full_dim,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    get_num_attention_heads_from_layers,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024
FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM = 16

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

_ag2_canonical_paged_logged_shapes: set[tuple[int, int, int, int]] = set()
_ag2_canonical_route_provenance_keys: set[tuple[object, ...]] = set()

logger = init_logger(__name__)

trtllm_workspace_buffer = None
_ag2_dcp_prefill_query_scratch: torch.Tensor | None = None


def get_query_scale_for_flashinfer(
    q_scale: float,
    query_dtype: torch.dtype,
) -> float | None:
    """Pass q_scale only when FlashInfer receives an FP8 query.

    FlashInfer folds a supplied q_scale into the attention softmax scale.
    Model-dtype queries were not divided by q_scale, so applying a calibrated
    non-unit scale to BF16/FP16 queries changes the attention logits.
    """
    if query_dtype in (FP8_DTYPE, torch.float8_e5m2):
        return q_scale
    return None


def set_ag2_dcp_prefill_query_scratch(scratch: torch.Tensor | None) -> None:
    global _ag2_dcp_prefill_query_scratch
    if scratch is not None and (scratch.dtype != torch.bfloat16 or not scratch.is_cuda):
        raise ValueError("DCP prefill query scratch must be a CUDA BF16 tensor")
    _ag2_dcp_prefill_query_scratch = scratch


def _resolve_decode_split_plan(
    *,
    num_decode_tokens: int,
    num_decodes: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    spec_target_only: bool,
    qlen1_fixed_split_size: int = -1,
    qlen1_disable_split_kv: bool = False,
) -> tuple[int, bool, bool]:
    """Resolve native-decode split policy and graph-wrapper eligibility.

    A speculative target verifies more than one query token per request.  Its
    vLLM lane is already PIECEWISE, so a fixed split can use FlashInfer's
    ordinary replannable wrapper.  MTP draft decode has one query token and
    retains the existing FULL CUDA-graph wrapper.

    Returns ``(fixed_split_size, disable_split_kv, force_non_graph_wrapper)``.
    """
    if num_decodes <= 0 or num_decode_tokens % num_decodes:
        raise ValueError(
            "FlashInfer native decode requires a positive request count and "
            "uniform query length: "
            f"{num_decode_tokens=} {num_decodes=}"
        )
    query_len = num_decode_tokens // num_decodes
    if query_len == 1 and qlen1_fixed_split_size > 0:
        return qlen1_fixed_split_size, qlen1_disable_split_kv, False
    if spec_target_only and query_len == 1:
        return -1, False, False
    force_non_graph_wrapper = spec_target_only and fixed_split_size > 0
    return fixed_split_size, disable_split_kv, force_non_graph_wrapper


def _flashinfer_seq_lens_and_blocks_for_paged_kv(
    seq_lens_cpu: torch.Tensor,
    qo_indptr_cpu: torch.Tensor,
    num_decodes: int,
    num_prefills: int,
    page_size: int,
    *,
    use_dcp: bool,
    dcp_world_size: int,
    dcp_rank: int,
    dcp_kv_cache_interleave_size: int,
    include_prefill_query: bool = False,
    include_prefill_query_from: int | None = None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Return FlashInfer paged-KV lengths after context/DCP localization.

    ``CommonAttentionMetadata.seq_lens`` includes the scheduled query tokens.
    Native FlashInfer DCP prefill has two phases: attend over the previously
    cached context with paged KV, then attend over the new ragged KV tokens.
    Therefore the paged-KV metadata must be computed from context lengths, and
    when DCP is enabled those context lengths must be localized before deriving
    page counts and last-page lengths.
    """
    localized_seq_lens_cpu = seq_lens_cpu.clone()
    if use_dcp:
        if include_prefill_query and include_prefill_query_from is not None:
            raise ValueError(
                "Specify either include_prefill_query or "
                "include_prefill_query_from, not both"
            )
        if include_prefill_query:
            include_prefill_query_from = num_decodes
        if include_prefill_query_from is None:
            include_prefill_query_from = num_decodes + num_prefills
        if not num_decodes <= include_prefill_query_from <= num_decodes + num_prefills:
            raise ValueError(
                "Prefill-query inclusion boundary is outside the prefill cohort: "
                f"{include_prefill_query_from=} {num_decodes=} {num_prefills=}"
            )
        if num_prefills > 0 and include_prefill_query_from > num_decodes:
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[num_decodes:] - qo_indptr_cpu[num_decodes]
            )
            query_lens_prefill_cpu = (
                qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
            )
            standard_prefill_count = include_prefill_query_from - num_decodes
            localized_seq_lens_cpu[
                num_decodes:include_prefill_query_from
            ] -= query_lens_prefill_cpu[:standard_prefill_count]

        localized_seq_lens_cpu = get_dcp_local_seq_lens(
            localized_seq_lens_cpu,
            dcp_world_size,
            dcp_rank,
            dcp_kv_cache_interleave_size,
        )

    seq_lens_np = localized_seq_lens_cpu.numpy()
    num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
    return localized_seq_lens_cpu, seq_lens_np, num_blocks_np


def _dcp_pseudo_decode_rows(
    seq_lens_cpu: torch.Tensor,
    qo_indptr_cpu: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    dcp_kv_cache_interleave_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand qlen>1 requests into causal qlen=1 DCP decode rows.

    The source KV for every row is the original request's block table. The
    inclusive global length advances once per query token, then is localized
    independently for the current DCP rank. This prevents earlier verification
    rows from seeing later speculative K/V that has already been written to the
    paged cache.
    """
    query_lens = qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]
    if query_lens.numel() != seq_lens_cpu.numel():
        raise ValueError(
            "Pseudo-decode requires one query length per sequence length: "
            f"{query_lens.numel()=} versus {seq_lens_cpu.numel()=}."
        )
    if bool(torch.any(query_lens <= 0)):
        raise ValueError("Pseudo-decode does not support empty query rows.")

    row_to_req = torch.repeat_interleave(
        torch.arange(query_lens.numel(), dtype=torch.int64),
        query_lens.to(torch.int64),
    )
    row_offsets = torch.cat(
        [
            torch.arange(1, int(query_len) + 1, dtype=seq_lens_cpu.dtype)
            for query_len in query_lens
        ]
    )
    query_lens_per_row = torch.repeat_interleave(query_lens, query_lens)
    final_seq_lens_per_row = seq_lens_cpu[row_to_req]
    global_row_lens = final_seq_lens_per_row - query_lens_per_row + row_offsets
    local_row_lens = get_dcp_local_seq_lens(
        global_row_lens,
        dcp_world_size,
        dcp_rank,
        dcp_kv_cache_interleave_size,
    )
    return row_to_req, local_row_lens


def _dcp_causal_paged_custom_mask(
    seq_lens_cpu: torch.Tensor,
    qo_indptr_cpu: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    dcp_kv_cache_interleave_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a causal mask over full DCP-local paged KV for each prefill.

    ``seq_lens_cpu`` includes the newly scheduled query tokens, which have
    already been written to paged KV before the attention backend runs.  A
    query at global position ``p`` may consume the first DCP-local
    ``get_dcp_local_seq_lens(p + 1)`` keys.  FlashInfer expects custom masks
    flattened request by request with ``q_len * local_kv_len`` elements.

    This helper only defines and validates the metadata contract.  It is not
    selected by the runtime until a separate default-off canonical-paged POC
    passes saved-boundary and cost gates.
    """
    if seq_lens_cpu.device.type != "cpu" or qo_indptr_cpu.device.type != "cpu":
        raise ValueError("DCP causal paged mask metadata must be built on CPU")
    if seq_lens_cpu.ndim != 1 or qo_indptr_cpu.ndim != 1:
        raise ValueError("DCP causal paged mask expects one-dimensional metadata")
    if qo_indptr_cpu.numel() != seq_lens_cpu.numel() + 1:
        raise ValueError(
            "DCP causal paged mask requires one query span per sequence: "
            f"{qo_indptr_cpu.numel()=} {seq_lens_cpu.numel()=}"
        )
    if dcp_world_size < 1 or not 0 <= dcp_rank < dcp_world_size:
        raise ValueError(
            f"Invalid DCP geometry: {dcp_world_size=} {dcp_rank=}"
        )
    if dcp_kv_cache_interleave_size < 1:
        raise ValueError("DCP KV cache interleave size must be positive")

    query_lens = qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]
    if bool(torch.any(query_lens <= 0)):
        raise ValueError("DCP causal paged mask does not support empty query spans")
    if bool(torch.any(seq_lens_cpu < query_lens)):
        raise ValueError("DCP sequence length cannot be shorter than its query span")

    local_seq_lens = get_dcp_local_seq_lens(
        seq_lens_cpu,
        dcp_world_size,
        dcp_rank,
        dcp_kv_cache_interleave_size,
    )
    request_masks: list[torch.Tensor] = []
    for seq_len_t, query_len_t, local_seq_len_t in zip(
        seq_lens_cpu, query_lens, local_seq_lens, strict=True
    ):
        seq_len = int(seq_len_t)
        query_len = int(query_len_t)
        local_seq_len = int(local_seq_len_t)
        first_query_position = seq_len - query_len
        inclusive_global_bounds = torch.arange(
            first_query_position + 1,
            seq_len + 1,
            dtype=torch.int32,
        )
        visible_local_lens = get_dcp_local_seq_lens(
            inclusive_global_bounds,
            dcp_world_size,
            dcp_rank,
            dcp_kv_cache_interleave_size,
        )
        local_positions = torch.arange(local_seq_len, dtype=torch.int32)
        request_masks.append(
            (local_positions.unsqueeze(0) < visible_local_lens.unsqueeze(1)).reshape(
                -1
            )
        )
    return torch.cat(request_masks), local_seq_lens


def _dcp_pseudo_block_table_capacity(
    max_model_len: int,
    max_num_batched_tokens: int,
    page_size: int,
    dcp_world_size: int,
) -> int:
    """Bound the worker's hybrid block-table width before graph capture."""
    if min(max_model_len, max_num_batched_tokens, page_size, dcp_world_size) <= 0:
        raise ValueError("pseudo-decode capacity inputs must be positive")
    # History pages are DCP-local, but the scheduler tail is already expressed
    # in kernel pages and must not be divided by DCP.  Keep one world-sized
    # boundary/speculation guard, then mirror the worker's page alignment.
    base_pages = cdiv(max_model_len, page_size * dcp_world_size)
    scheduler_tail_pages = cdiv(max_num_batched_tokens, page_size)
    capacity = base_pages + scheduler_tail_pages + dcp_world_size
    if page_size <= 128:
        alignment = 128 // page_size
        capacity = cdiv(capacity, alignment) * alignment
    return capacity


@dataclass(frozen=True)
class _AbsolutePrefillSegments:
    """CPU metadata for a scheduler-history-independent prompt segmentation."""

    qo_indptr_cpu: torch.Tensor
    request_indices_cpu: torch.Tensor
    global_starts_cpu: torch.Tensor
    global_ends_cpu: torch.Tensor
    context_query_indices_cpu: torch.Tensor
    context_segment_indices_cpu: torch.Tensor


def _ag2_absolute_prefill_segments(
    *,
    global_seq_lens_cpu: torch.Tensor,
    qo_indptr_cpu: torch.Tensor,
    num_prompt_tokens_cpu: torch.Tensor,
    segment_size: int,
) -> _AbsolutePrefillSegments:
    """Split actual prompt rows at stable absolute token boundaries.

    The caller supplies only a contiguous actual-prefill cohort. Artificial
    scheduler boundaries must already satisfy the model's recurrent-state
    alignment. A non-aligned final endpoint is valid only when it is the
    semantic end of the prompt. The flattened query-token order is preserved;
    this function changes only the attention batch-item boundaries.
    """
    tensors = (global_seq_lens_cpu, qo_indptr_cpu, num_prompt_tokens_cpu)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("Absolute prefill segment metadata must be built on CPU")
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("Absolute prefill segment metadata must be one-dimensional")
    num_requests = int(global_seq_lens_cpu.numel())
    if qo_indptr_cpu.numel() != num_requests + 1:
        raise ValueError("Absolute prefill requires one query span per request")
    if num_prompt_tokens_cpu.numel() != num_requests:
        raise ValueError("Absolute prefill requires one prompt length per request")
    if segment_size < 1:
        raise ValueError("Absolute prefill segment size must be positive")
    if int(qo_indptr_cpu[0]) != 0 or bool(
        torch.any(qo_indptr_cpu[1:] <= qo_indptr_cpu[:-1])
    ):
        raise ValueError("Absolute prefill query spans must be positive and monotonic")

    segment_lengths: list[int] = []
    request_indices: list[int] = []
    global_starts: list[int] = []
    global_ends: list[int] = []
    context_query_indices: list[int] = []
    context_segment_indices: list[int] = []
    for request_index in range(num_requests):
        query_begin = int(qo_indptr_cpu[request_index])
        query_end = int(qo_indptr_cpu[request_index + 1])
        query_len = query_end - query_begin
        global_end = int(global_seq_lens_cpu[request_index])
        prompt_len = int(num_prompt_tokens_cpu[request_index])
        global_start = global_end - query_len
        if global_start < 0 or global_end > prompt_len:
            raise ValueError(
                "Absolute prefill span is outside the semantic prompt: "
                f"request={request_index} start={global_start} end={global_end} "
                f"prompt={prompt_len}"
            )
        if global_start % segment_size:
            raise ValueError(
                "Absolute prefill cannot resume from an unaligned boundary: "
                f"request={request_index} start={global_start} "
                f"segment_size={segment_size}"
            )
        if global_end < prompt_len and global_end % segment_size:
            raise ValueError(
                "Absolute prefill cannot commit an unaligned non-final boundary: "
                f"request={request_index} end={global_end} prompt={prompt_len} "
                f"segment_size={segment_size}"
            )

        segment_start = global_start
        while segment_start < global_end:
            segment_end = min(segment_start + segment_size, global_end)
            segment_index = len(segment_lengths)
            segment_length = segment_end - segment_start
            flat_query_start = query_begin + segment_start - global_start
            segment_lengths.append(segment_length)
            request_indices.append(request_index)
            global_starts.append(segment_start)
            global_ends.append(segment_end)
            if segment_start > 0:
                context_segment_indices.append(segment_index)
                context_query_indices.extend(
                    range(flat_query_start, flat_query_start + segment_length)
                )
            segment_start = segment_end

    cumulative = [0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
    if cumulative[-1] != int(qo_indptr_cpu[-1]):
        raise AssertionError("Absolute segments did not preserve flattened query size")
    return _AbsolutePrefillSegments(
        qo_indptr_cpu=torch.tensor(cumulative, dtype=torch.int32),
        request_indices_cpu=torch.tensor(request_indices, dtype=torch.int32),
        global_starts_cpu=torch.tensor(global_starts, dtype=torch.int32),
        global_ends_cpu=torch.tensor(global_ends, dtype=torch.int32),
        context_query_indices_cpu=torch.tensor(
            context_query_indices, dtype=torch.int64
        ),
        context_segment_indices_cpu=torch.tensor(
            context_segment_indices, dtype=torch.int32
        ),
    )


def _ag2_is_actual_prefill_cohort(
    is_prefilling: torch.Tensor | np.ndarray | list[bool],
    num_decodes: int,
    num_prefills: int,
) -> bool:
    """Distinguish real prompt prefill from qlen>1 target verification."""
    if num_decodes < 0 or num_prefills < 1:
        raise ValueError("Actual-prefill selector requires non-negative decode count")
    prefill_start = num_decodes
    prefill_end = prefill_start + num_prefills
    if len(is_prefilling) < prefill_end:
        raise ValueError(
            "Actual-prefill selector metadata is shorter than the selected cohort"
        )
    if isinstance(is_prefilling, torch.Tensor):
        return bool(torch.all(is_prefilling[prefill_start:prefill_end]).item())
    return bool(np.all(np.asarray(is_prefilling)[prefill_start:prefill_end]))


def _ag2_semantic_prefill_split(
    common_attn_metadata: CommonAttentionMetadata,
    generic_split: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Split real prompt-prefill rows by lifecycle, including short extends."""
    is_prefilling = common_attn_metadata.is_prefilling
    if is_prefilling is None:
        raise RuntimeError(
            "Canonical paged DCP prefill requires request lifecycle metadata"
        )
    lifecycle = (
        is_prefilling[: common_attn_metadata.num_reqs].cpu().numpy()
        if isinstance(is_prefilling, torch.Tensor)
        else np.asarray(is_prefilling[: common_attn_metadata.num_reqs])
    ).astype(np.bool_, copy=False)
    if not bool(np.any(lifecycle)):
        # Preserve the existing target/decode routing exactly. In particular,
        # qlen4 speculative verification is not a prompt-prefill merely because
        # its physical query length exceeds one.
        return generic_split
    first_actual_prefill = int(np.argmax(lifecycle))
    if bool(np.any(~lifecycle[first_actual_prefill:])):
        raise RuntimeError(
            "Canonical paged DCP prefill requires decode rows before prompt-prefill "
            "rows; lifecycle metadata is not a contiguous suffix"
        )
    # ``generic_split`` is a computational dispatch contract, not a request
    # lifecycle classification. In particular, qlen>1 target verification is
    # intentionally routed through FlashInfer's prefill kernel. Never turn an
    # existing computational-prefill row back into native decode merely
    # because its request is no longer in prompt-prefill lifecycle. Expand the
    # prefill suffix only when a short actual prompt was classified as decode.
    num_decodes = min(generic_split[0], first_actual_prefill)
    num_prefills = common_attn_metadata.num_reqs - num_decodes
    num_decode_tokens = int(
        common_attn_metadata.query_start_loc_cpu[num_decodes].item()
    )
    num_prefill_tokens = common_attn_metadata.num_actual_tokens - num_decode_tokens
    return num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens


def _ag2_resolve_canonical_prefill_route(
    *,
    common_attn_metadata: CommonAttentionMetadata,
    generic_split: tuple[int, int, int, int],
    common_prefix_len: int,
    enabled: bool,
    causal: bool,
    use_dcp: bool,
    use_dcp_pseudo_decode: bool,
) -> tuple[tuple[int, int, int, int], int | None, bool]:
    """Resolve semantic split, canonical arm and cascade as one contract."""
    semantic_split = generic_split
    canonical_paged_start_req: int | None = None
    if enabled and causal and use_dcp and not use_dcp_pseudo_decode:
        is_prefilling = common_attn_metadata.is_prefilling
        # Draft/speculator metadata is a separate non-prompt producer and does
        # not carry main-runner prompt lifecycle. It is explicitly outside this
        # candidate: preserve its existing generic route and never select the
        # canonical prompt arm. Main prompt metadata supplies lifecycle and is
        # proven separately by the bounded route artifact.
        if is_prefilling is not None:
            semantic_split = _ag2_semantic_prefill_split(
                common_attn_metadata,
                generic_split,
            )
            num_decodes, num_prefills, _, _ = semantic_split
            lifecycle = (
                is_prefilling[: common_attn_metadata.num_reqs].cpu().numpy()
                if isinstance(is_prefilling, torch.Tensor)
                else np.asarray(is_prefilling[: common_attn_metadata.num_reqs])
            ).astype(np.bool_, copy=False)
            if bool(np.any(lifecycle)):
                canonical_paged_start_req = int(np.argmax(lifecycle))
                if not num_decodes <= canonical_paged_start_req < (
                    num_decodes + num_prefills
                ):
                    raise RuntimeError(
                        "Actual prompt-prefill boundary is outside the resolved "
                        "computational-prefill cohort"
                    )
        else:
            num_decodes, num_prefills = generic_split[:2]
    # Candidate A consumes the complete block table through one full-paged
    # custom-mask call. Keeping cascade enabled for that cohort would silently
    # route a common prefix through a second execution tree.
    use_cascade = common_prefix_len > 0 and canonical_paged_start_req is None
    return semantic_split, canonical_paged_start_req, use_cascade


def _ag2_record_canonical_route_provenance(
    *,
    common_attn_metadata: CommonAttentionMetadata,
    common_prefix_len: int,
    generic_split: tuple[int, int, int, int],
    semantic_split: tuple[int, int, int, int],
    canonical_paged_prefill: bool,
    canonical_mode: str | None,
    canonical_paged_start_req: int | None,
    use_cascade: bool,
    dcp_rank: int,
    for_cudagraph_capture: bool,
) -> None:
    """Persist one bounded record per distinct runtime route signature."""
    output = os.environ.get("AG2_VLLM_DCP_CANONICAL_ROUTE_OUTPUT", "")
    if not output or dcp_rank != 0:
        return
    num_reqs = common_attn_metadata.num_reqs
    request_ids = common_attn_metadata.request_ids
    scheduled = common_attn_metadata.num_scheduled_tokens_cpu
    computed = common_attn_metadata.num_computed_tokens_provenance_cpu
    lifecycle = common_attn_metadata.is_prefilling
    prompt_tokens = common_attn_metadata.num_prompt_tokens_cpu
    if (
        request_ids is None
        or scheduled is None
        or computed is None
        or lifecycle is None
    ):
        raise RuntimeError(
            "Canonical route provenance requires request ids, computed/scheduled "
            "counts and lifecycle metadata"
        )
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1 : num_reqs + 1]
        - common_attn_metadata.query_start_loc_cpu[:num_reqs]
    ).tolist()
    record = {
        "schema": 1,
        "request_ids": list(request_ids[:num_reqs]),
        "num_computed_tokens": computed[:num_reqs].tolist(),
        "num_scheduled_tokens": scheduled[:num_reqs].tolist(),
        "is_prefilling": lifecycle[:num_reqs].tolist(),
        "query_lens": query_lens,
        "common_prefix_len": int(common_prefix_len),
        "generic_split": list(generic_split),
        "semantic_split": list(semantic_split),
        "canonical_paged_prefill": bool(canonical_paged_prefill),
        "canonical_mode": (
            canonical_mode
            if canonical_mode is not None
            else ("dense_paged" if canonical_paged_prefill else "off")
        ),
        "canonical_paged_start_req": canonical_paged_start_req,
        "use_cascade": bool(use_cascade),
        "for_cudagraph_capture": bool(for_cudagraph_capture),
    }
    key = (
        tuple(record["request_ids"]),
        tuple(record["num_computed_tokens"]),
        tuple(record["num_scheduled_tokens"]),
        tuple(record["is_prefilling"]),
        int(common_prefix_len),
        tuple(generic_split),
        tuple(semantic_split),
        bool(canonical_paged_prefill),
        record["canonical_mode"],
        canonical_paged_start_req,
        bool(use_cascade),
        bool(for_cudagraph_capture),
    )
    if prompt_tokens is not None:
        record["num_prompt_tokens"] = prompt_tokens[:num_reqs].tolist()
    if key in _ag2_canonical_route_provenance_keys:
        return
    max_records = int(
        os.environ.get("AG2_VLLM_DCP_CANONICAL_ROUTE_MAX_RECORDS", "4096")
    )
    if max_records < 1 or len(_ag2_canonical_route_provenance_keys) >= max_records:
        raise RuntimeError(
            "Canonical route provenance capacity exhausted before recording a new "
            f"signature: capacity={max_records}"
        )
    _ag2_canonical_route_provenance_keys.add(key)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _get_trtllm_workspace_buffer():
    global trtllm_workspace_buffer
    if trtllm_workspace_buffer is None:
        trtllm_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
        )
    return trtllm_workspace_buffer


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    src_stride_page,
    src_stride_kv,
    src_stride_head,
    src_stride_block,
    src_stride_head_size,
    DST_K_CACHE_STRIDE: tl.constexpr,
    DST_KV_CACHE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    mock_page_idx = batch_idx * block_table_stride + mock_block_table_idx + 1
    logical_offsets = tl.arange(0, HEAD_STRIDE)
    block_offsets = logical_offsets // HEAD_SIZE
    head_size_offsets = logical_offsets % HEAD_SIZE

    for h in range(NUM_KV_HEADS):
        h_off = tl.cast(h, tl.int64)

        # Read K from source (supports non-contiguous page/kv/head strides)
        src_k = (
            orig_page_num * src_stride_page
            + h_off * src_stride_head
            + block_offsets * src_stride_block
            + head_size_offsets * src_stride_head_size
        )
        fp8_k = tl.load(kv_cache_ptr + src_k)
        dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)

        # Write K to contiguous mock cache
        dst_k = mock_page_idx * DST_KV_CACHE_STRIDE + h * HEAD_STRIDE + logical_offsets
        tl.store(mock_kv_cache_ptr + dst_k, dequant_k)

        # Read V from source (offset by src_stride_kv for the V half)
        src_v = (
            orig_page_num * src_stride_page
            + src_stride_kv
            + h_off * src_stride_head
            + block_offsets * src_stride_block
            + head_size_offsets * src_stride_head_size
        )
        fp8_v = tl.load(kv_cache_ptr + src_v)
        dequant_v = (fp8_v.to(tl.float32) * v_scale_val).to(dequant_dtype)

        # Write V to contiguous mock cache
        dst_v = (
            mock_page_idx * DST_KV_CACHE_STRIDE
            + DST_K_CACHE_STRIDE
            + h * HEAD_STRIDE
            + logical_offsets
        )
        tl.store(mock_kv_cache_ptr + dst_v, dequant_v)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    # Logical source layout is (B, 2, H, N, D). The tensor may be a
    # non-contiguous view, so the Triton kernel indexes it with actual strides.
    strides = kv_cache.stride()
    num_kv_heads, block_size, head_size = s[2], s[3], s[4]
    head_stride = block_size * head_size
    k_cache_stride = num_kv_heads * head_stride
    kv_cache_stride = k_cache_stride * s[1]

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    # we simply sequentially index the pages needed by this prefill
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        k_cache_stride,
        kv_cache_stride,
        head_stride,
        head_size,
        num_kv_heads,
    )
    return mock_kv_cache, mock_block_table


def _copy_ag2_dcp_trace(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    """Copy an attention intermediate into an optional graph-safe trace buffer."""
    if not getattr(layer, "_ag2_dcp_trace_enabled", False):
        return
    buffer = getattr(layer, f"_ag2_dcp_trace_{name}")
    trace = value[: buffer.shape[0]]
    buffer[: trace.shape[0]].copy_(trace)


AG2_DCP_OBSERVER_SCHEMA_VERSION = 1
AG2_DCP_OBSERVER_PLAN_BIT = 1 << 0
AG2_DCP_OBSERVER_HISTORY_BIT = 1 << 1
AG2_DCP_OBSERVER_CURRENT_KV_BIT = 1 << 2
AG2_DCP_OBSERVER_SCALES_BIT = 1 << 3
AG2_DCP_OBSERVER_STAGE_BITS = {
    "local_output": 1 << 4,
    "local_lse": 1 << 5,
    "combined_output": 1 << 6,
    "combined_lse": 1 << 7,
    "query_output": 1 << 8,
    "query_lse": 1 << 9,
    "merged_output": 1 << 10,
}
AG2_DCP_OBSERVER_COMPLETE_MASK = (
    AG2_DCP_OBSERVER_PLAN_BIT
    | AG2_DCP_OBSERVER_HISTORY_BIT
    | AG2_DCP_OBSERVER_CURRENT_KV_BIT
    | AG2_DCP_OBSERVER_SCALES_BIT
    | sum(AG2_DCP_OBSERVER_STAGE_BITS.values())
)


def _write_ag2_dcp_aux_pack(
    layer: torch.nn.Module,
    stage: str,
    value: torch.Tensor,
    dcp_output_pack: torch.Tensor | None,
    dcp_lse_pack: torch.Tensor | None,
    request_tail_indices: torch.Tensor | None = None,
    request_tail_count: int = 0,
    request_row_stride: int = 0,
    dcp_observer_meta: torch.Tensor | None = None,
) -> None:
    """Write per-request-tail DCP state into explicit custom-op outputs.

    These packs differ from the historical registered trace buffers: they are
    explicit mutated arguments of the attention custom op and are returned as
    model auxiliary outputs. This makes their CUDA Graph dataflow
    authoritative.
    """
    if dcp_output_pack is None and dcp_lse_pack is None:
        return
    if dcp_output_pack is None or dcp_lse_pack is None:
        raise RuntimeError("DCP auxiliary output and LSE packs must be paired")
    if request_tail_indices is not None and request_row_stride:
        raise RuntimeError(
            "DCP auxiliary pack cannot combine explicit indices and row stride"
        )
    if request_row_stride < 0:
        raise RuntimeError("DCP auxiliary pack row stride must be nonnegative")
    strided_request_rows = request_row_stride > 0
    legacy_row_zero = request_tail_indices is None and not strided_request_rows
    if legacy_row_zero:
        request_tail_count = 1
    if request_tail_count > dcp_output_pack.shape[0]:
        # Non-target CUDA Graph capture shapes can be wider than the bounded
        # observer. Preserve an explicit incomplete sentinel; the analyzer
        # must reject it if such a shape reaches the target packet.
        dcp_output_pack.fill_(torch.nan)
        dcp_lse_pack.fill_(torch.nan)
        if dcp_observer_meta is not None:
            dcp_observer_meta.fill_(-1)
        return
    local_heads = layer.num_heads
    total_heads = local_heads * get_dcp_group().world_size
    head_dim = value.shape[-1] if value.ndim == 3 else layer.head_size
    output_layout = {
        "local_output": (0, total_heads * head_dim),
        "combined_output": (
            total_heads * head_dim,
            local_heads * head_dim,
        ),
        "query_output": (
            (total_heads + local_heads) * head_dim,
            local_heads * head_dim,
        ),
        "merged_output": (
            (total_heads + 2 * local_heads) * head_dim,
            local_heads * head_dim,
        ),
    }
    lse_layout = {
        "local_lse": (0, total_heads),
        "combined_lse": (total_heads, local_heads),
        "query_lse": (total_heads + local_heads, local_heads),
    }
    if stage in output_layout:
        offset, expected = output_layout[stage]
        if legacy_row_zero:
            selected = value[:1]
        elif strided_request_rows:
            selected = value[
                : request_tail_count * request_row_stride : request_row_stride
            ]
        else:
            selected = torch.index_select(
                value,
                0,
                request_tail_indices[:request_tail_count],
            )
        selected = selected.reshape(request_tail_count, -1)
        if selected.shape[1] != expected:
            raise RuntimeError(
                f"DCP auxiliary {stage} output has {selected.shape[1]} values, "
                f"expected {expected}"
            )
        dcp_output_pack[:request_tail_count, offset : offset + expected].copy_(selected)
        if (
            dcp_observer_meta is not None
            and request_tail_count <= dcp_observer_meta.shape[0]
        ):
            dcp_observer_meta[:request_tail_count, 1].bitwise_or_(
                AG2_DCP_OBSERVER_STAGE_BITS[stage]
            )
        return
    if stage in lse_layout:
        offset, expected = lse_layout[stage]
        if legacy_row_zero:
            selected = value[:1]
        elif strided_request_rows:
            selected = value[
                : request_tail_count * request_row_stride : request_row_stride
            ]
        else:
            selected = torch.index_select(
                value,
                0,
                request_tail_indices[:request_tail_count],
            )
        selected = selected.reshape(request_tail_count, -1)
        if selected.shape[1] != expected:
            raise RuntimeError(
                f"DCP auxiliary {stage} LSE has {selected.shape[1]} values, "
                f"expected {expected}"
            )
        dcp_lse_pack[:request_tail_count, offset : offset + expected].copy_(selected)
        if (
            dcp_observer_meta is not None
            and request_tail_count <= dcp_observer_meta.shape[0]
        ):
            dcp_observer_meta[:request_tail_count, 1].bitwise_or_(
                AG2_DCP_OBSERVER_STAGE_BITS[stage]
            )
        return
    raise ValueError(f"Unknown DCP auxiliary pack stage: {stage}")


def _write_ag2_dcp_kv_history_pack(
    layer: torch.nn.Module,
    kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
    decode: "FIDecode",
    history_pack: torch.Tensor | None,
    history_meta: torch.Tensor | None,
    page_indices_pack: torch.Tensor | None,
    observer_meta: torch.Tensor | None,
    *,
    request_count: int,
    request_row_stride: int,
) -> None:
    """Copy each request's first decode row's logical paged KV into a trace.

    The trace contains only consumed tokens, in logical sequence order. Physical
    page ids and unused last-page tails are excluded, so equal histories compare
    byte-for-byte even when requests occupy different cache blocks.
    """
    del layer
    outputs = (history_pack, history_meta, page_indices_pack, observer_meta)
    if all(output is None for output in outputs):
        return
    if any(output is None for output in outputs):
        raise RuntimeError(
            "DCP KV history data, pages and observer metadata must be paired"
        )
    assert history_pack is not None
    assert history_meta is not None
    assert page_indices_pack is not None
    assert observer_meta is not None
    if request_row_stride < 1:
        raise RuntimeError("DCP KV history requires a positive request row stride")
    if request_count > history_pack.shape[0]:
        history_pack.zero_()
        history_meta.fill_(-1)
        page_indices_pack.fill_(-1)
        observer_meta.fill_(-1)
        return
    wrapper = decode.wrapper
    indptr = wrapper._paged_kv_indptr_buf
    indices = wrapper._paged_kv_indices_buf
    last_page_len = wrapper._paged_kv_last_page_len_buf
    if indptr is None or indices is None or last_page_len is None:
        raise RuntimeError("FlashInfer decode wrapper did not retain paged KV metadata")
    if indices.numel() == 0:
        # Dummy CUDA Graph inputs can have valid zero-page indptr metadata and
        # an empty page table. There is no consumed history to observe.
        history_pack.zero_()
        history_meta.fill_(-1)
        page_indices_pack.fill_(-1)
        observer_meta.fill_(-1)
        return

    layout = get_kv_cache_layout()
    if layout == "HND":
        page_size = kv_cache_tuple[0].shape[2]
    elif layout == "NHD":
        page_size = kv_cache_tuple[0].shape[1]
    else:
        raise RuntimeError(f"Unsupported KV cache layout for history trace: {layout}")
    capacity = history_pack.shape[2]
    if any(cache.element_size() != 1 for cache in kv_cache_tuple):
        raise RuntimeError("DCP KV history byte trace currently requires FP8 KV")

    if request_row_stride == 1:
        # Draft qlen=1 runs inside FULL CUDA Graphs. Capture a bounded logical
        # tail using only device tensor operations: scalar CPU reads would
        # either break capture or turn the record into a stale side effect.
        # The wrapper buffers are the exact metadata consumed by FlashInfer.
        request_rows = torch.arange(
            request_count,
            dtype=torch.long,
            device=indptr.device,
        )
        page_starts = indptr[:request_count].to(torch.long)
        page_ends = indptr[1 : request_count + 1].to(torch.long)
        total_pages = page_ends - page_starts
        max_trace_pages = (history_pack.shape[2] + page_size - 1) // page_size
        if max_trace_pages > page_indices_pack.shape[1]:
            raise RuntimeError("DCP page provenance pack is narrower than KV history")
        trace_starts = torch.maximum(page_starts, page_ends - max_trace_pages)
        trace_pages = page_ends - trace_starts
        page_offsets = torch.arange(
            max_trace_pages,
            dtype=torch.long,
            device=indptr.device,
        )
        flat_offsets = trace_starts[:, None] + page_offsets[None, :]
        valid_pages = page_offsets[None, :] < trace_pages[:, None]
        safe_offsets = flat_offsets.clamp(0, max(indices.numel() - 1, 0))
        selected_page_ids = indices.to(torch.long)[safe_offsets]
        selected_page_ids = torch.where(
            valid_pages,
            selected_page_ids,
            torch.zeros_like(selected_page_ids),
        )

        final_page_lens = last_page_len[:request_count].to(torch.long)
        logical_tokens = torch.where(
            total_pages > 0,
            (total_pages - 1) * page_size + final_page_lens,
            torch.zeros_like(total_pages),
        )
        retained_tokens = torch.where(
            trace_pages > 0,
            (trace_pages - 1) * page_size + final_page_lens,
            torch.zeros_like(trace_pages),
        ).clamp_max(history_pack.shape[2])

        history_pack.zero_()
        history_meta.fill_(-1)
        page_indices_pack.fill_(-1)
        page_indices_pack[:request_count, :max_trace_pages].copy_(
            torch.where(
                valid_pages,
                selected_page_ids.to(page_indices_pack.dtype),
                torch.full_like(selected_page_ids, -1).to(page_indices_pack.dtype),
            )
        )
        for kv_index, cache in enumerate(kv_cache_tuple):
            pages = torch.index_select(cache, 0, selected_page_ids.reshape(-1))
            pages = pages.reshape(request_count, max_trace_pages, *cache.shape[1:])
            if layout == "HND":
                pages = pages.permute(0, 1, 3, 2, 4)
            logical = pages.reshape(
                request_count,
                max_trace_pages * page_size,
                pages.shape[-2],
                pages.shape[-1],
            )[:, : history_pack.shape[2]]
            token_offsets = torch.arange(
                history_pack.shape[2],
                dtype=torch.long,
                device=logical.device,
            )
            valid_tokens = token_offsets[None, :] < retained_tokens[:, None]
            logical = torch.where(
                valid_tokens[:, :, None, None],
                logical,
                torch.zeros_like(logical),
            )
            history_pack[:request_count, kv_index].copy_(logical.view(torch.uint8))

        history_meta[:request_count].copy_(
            torch.stack(
                [request_rows, retained_tokens, trace_pages, final_page_lens], dim=1
            ).to(history_meta.dtype)
        )
        if observer_meta is not None:
            if observer_meta.shape[1] < 24:
                raise RuntimeError("DCP causal observer metadata requires 24 columns")
            observer_meta.fill_(-1)
            observer_meta[:request_count, 0] = AG2_DCP_OBSERVER_SCHEMA_VERSION
            observer_meta[:request_count, 1] = (
                AG2_DCP_OBSERVER_PLAN_BIT | AG2_DCP_OBSERVER_HISTORY_BIT
            )
            observer_meta[:request_count, 2] = 1
            observer_meta[:request_count, 3] = request_rows.to(observer_meta.dtype)
            observer_meta[:request_count, 4] = request_rows.to(observer_meta.dtype)
            observer_meta[:request_count, 5] = request_row_stride
            observer_meta[:request_count, 6] = logical_tokens.to(observer_meta.dtype)
            observer_meta[:request_count, 7] = total_pages.to(observer_meta.dtype)
            observer_meta[:request_count, 8] = final_page_lens.to(observer_meta.dtype)
            observer_meta[:request_count, 9] = page_starts.to(observer_meta.dtype)
            observer_meta[:request_count, 10] = page_ends.to(observer_meta.dtype)
            observer_meta[:request_count, 11] = request_count
            observer_meta[:request_count, 12] = int(
                getattr(wrapper, "_use_cuda_graph", True)
            )
            observer_meta[:request_count, 13] = 3  # decode request row
            observer_meta[:request_count, 14] = page_size
            observer_meta[:request_count, 15] = 1 if layout == "NHD" else 2
        return
    history_pack.zero_()
    history_meta.fill_(-1)
    for request_index in range(request_count):
        source_row = request_index * request_row_stride
        page_start = int(indptr[source_row].item())
        page_end = int(indptr[source_row + 1].item())
        num_pages = page_end - page_start
        final_page_len = int(last_page_len[source_row].item())
        logical_tokens = (
            (num_pages - 1) * page_size + final_page_len if num_pages else 0
        )
        if logical_tokens > capacity:
            raise RuntimeError(
                "DCP KV history exceeds trace capacity: "
                f"{logical_tokens=} > {capacity=}"
            )
        page_ids = indices[page_start:page_end].to(torch.long)
        for kv_index, cache in enumerate(kv_cache_tuple):
            pages = torch.index_select(cache, 0, page_ids)
            if layout == "HND":
                pages = pages.permute(0, 2, 1, 3)
            logical = pages.reshape(-1, pages.shape[-2], pages.shape[-1])[
                :logical_tokens
            ]
            history_pack[request_index, kv_index, :logical_tokens].copy_(
                logical.view(torch.uint8)
            )
        history_meta[request_index, 0] = source_row
        history_meta[request_index, 1] = logical_tokens
        history_meta[request_index, 2] = num_pages
        history_meta[request_index, 3] = final_page_len


def _ag2_dcp_request_trace_indices(
    qo_indptr_cpu: torch.Tensor,
    row: str,
) -> torch.Tensor:
    """Return one flattened query-row index for each DCP prefill request."""
    if row == "first":
        return qo_indptr_cpu[:-1]
    if row == "tail":
        return qo_indptr_cpu[1:] - 1
    raise ValueError(
        f"AG2 DCP request-row trace must select 'first' or 'tail', got {row!r}"
    )


def _write_ag2_dcp_prefill_kv_history_pack(
    kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
    *,
    qo_indptr_cpu: torch.Tensor | None,
    paged_kv_indptr_cpu: torch.Tensor | None,
    paged_kv_indices: torch.Tensor | None,
    paged_kv_last_page_len_cpu: torch.Tensor | None,
    history_pack: torch.Tensor | None,
    history_meta: torch.Tensor | None,
    page_indices_pack: torch.Tensor | None,
    observer_meta: torch.Tensor | None,
    use_cuda_graph: bool,
    request_row: str,
    selected_query_len: int = 4,
) -> None:
    """Capture logical historical KV for selected DCP-prefill requests.

    Pack row ``i`` always describes request ``i``. Only requests whose query
    length equals ``selected_query_len`` are populated; other rows retain the
    fail-closed ``zero/-1`` sentinel. This excludes unrelated long prefills in
    a mixed batch while preserving request-index alignment with auxiliary QKV
    and attention-output packs.
    """
    observer_outputs = (
        history_pack,
        history_meta,
        page_indices_pack,
        observer_meta,
    )
    if all(value is None for value in observer_outputs):
        return
    if any(value is None for value in observer_outputs):
        raise RuntimeError(
            "DCP causal observer history, pages and metadata must be provided together"
        )
    assert history_pack is not None
    assert history_meta is not None
    assert page_indices_pack is not None
    assert observer_meta is not None
    if any(
        value is None
        for value in (
            qo_indptr_cpu,
            paged_kv_indptr_cpu,
            paged_kv_indices,
            paged_kv_last_page_len_cpu,
        )
    ):
        # Compile/profile and graph warmup can execute the compile-time trace
        # contract before a real paged plan has populated wrapper metadata.
        # Publish an explicit invalid sentinel; a real persisted qlen1 packet
        # must later satisfy the complete observer bitmap in the analyzer.
        history_pack.zero_()
        history_meta.fill_(-1)
        page_indices_pack.fill_(-1)
        observer_meta.fill_(-1)
        return
    assert qo_indptr_cpu is not None
    assert paged_kv_indptr_cpu is not None
    assert paged_kv_indices is not None
    assert paged_kv_last_page_len_cpu is not None

    request_count = int(qo_indptr_cpu.numel()) - 1
    if request_count > history_pack.shape[0]:
        # The five-row observer is intentionally narrower than the 44-request
        # compile envelope. Leave a fail-closed sentinel for unrelated warmup
        # or larger batches; the target x5 must fit and become complete.
        history_pack.zero_()
        history_meta.fill_(-1)
        page_indices_pack.fill_(-1)
        observer_meta.fill_(-1)
        return
    if any(
        value.shape[0] < request_count
        for value in (history_meta, page_indices_pack, observer_meta)
    ):
        raise RuntimeError("DCP causal observer outputs have inconsistent row capacity")
    if observer_meta.shape[1] < 24:
        raise RuntimeError("DCP causal observer metadata requires 24 columns")
    if int(paged_kv_indptr_cpu.numel()) != request_count + 1:
        raise RuntimeError("DCP prefill KV indptr does not match request count")
    if int(paged_kv_last_page_len_cpu.numel()) < request_count:
        raise RuntimeError("DCP prefill last-page lengths do not match requests")

    layout = get_kv_cache_layout()
    if layout == "HND":
        page_size = kv_cache_tuple[0].shape[2]
    elif layout == "NHD":
        page_size = kv_cache_tuple[0].shape[1]
    else:
        raise RuntimeError(f"Unsupported KV cache layout for history trace: {layout}")
    if any(cache.element_size() != 1 for cache in kv_cache_tuple):
        raise RuntimeError("DCP KV history byte trace currently requires FP8 KV")

    capacity = history_pack.shape[2]
    history_pack.zero_()
    history_meta.fill_(-1)
    page_indices_pack.fill_(-1)
    observer_meta.fill_(-1)
    query_lens = qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]
    for request_index in range(request_count):
        request_query_len = int(query_lens[request_index].item())
        trace_row = int(
            _ag2_dcp_request_trace_indices(qo_indptr_cpu, request_row)[
                request_index
            ].item()
        )
        selected = request_query_len == selected_query_len
        # Scalar device fills are CUDA-Graph-capturable. Constructing a CUDA
        # tensor from this CPU metadata performs a host-to-device copy during
        # capture and is rejected by PyTorch (or can freeze capture-time
        # values into later replays).
        observer_meta[request_index, 0] = AG2_DCP_OBSERVER_SCHEMA_VERSION
        observer_meta[request_index, 1] = AG2_DCP_OBSERVER_PLAN_BIT
        observer_meta[request_index, 2] = int(selected)
        observer_meta[request_index, 3] = trace_row
        observer_meta[request_index, 4] = request_index
        observer_meta[request_index, 5] = request_query_len
        observer_meta[request_index, 11] = int(qo_indptr_cpu[-1].item())
        observer_meta[request_index, 12] = int(use_cuda_graph)
        observer_meta[request_index, 13] = 1 if request_row == "first" else 2
        observer_meta[request_index, 14] = page_size
        observer_meta[request_index, 15] = 1 if layout == "NHD" else 2
        if not selected:
            continue
        page_start = int(paged_kv_indptr_cpu[request_index].item())
        page_end = int(paged_kv_indptr_cpu[request_index + 1].item())
        num_pages = page_end - page_start
        final_page_len = int(paged_kv_last_page_len_cpu[request_index].item())
        logical_tokens = (
            (num_pages - 1) * page_size + final_page_len if num_pages else 0
        )
        if logical_tokens > capacity:
            raise RuntimeError(
                "DCP KV history exceeds trace capacity: "
                f"{logical_tokens=} > {capacity=}"
            )
        if num_pages > page_indices_pack.shape[1]:
            raise RuntimeError(
                "DCP KV page provenance exceeds trace capacity: "
                f"{num_pages=} > {page_indices_pack.shape[1]}"
            )
        page_ids = paged_kv_indices[page_start:page_end].to(torch.long)
        page_indices_pack[request_index, :num_pages].copy_(
            page_ids.to(page_indices_pack.dtype)
        )
        for kv_index, cache in enumerate(kv_cache_tuple):
            pages = torch.index_select(cache, 0, page_ids)
            if layout == "HND":
                pages = pages.permute(0, 2, 1, 3)
            logical = pages.reshape(-1, pages.shape[-2], pages.shape[-1])[
                :logical_tokens
            ]
            history_pack[request_index, kv_index, :logical_tokens].copy_(
                logical.view(torch.uint8)
            )
        history_meta[request_index, 0] = trace_row
        history_meta[request_index, 1] = logical_tokens
        history_meta[request_index, 2] = num_pages
        history_meta[request_index, 3] = final_page_len
        observer_meta[request_index, 1].bitwise_or_(AG2_DCP_OBSERVER_HISTORY_BIT)
        observer_meta[request_index, 6] = logical_tokens
        observer_meta[request_index, 7] = num_pages
        observer_meta[request_index, 8] = final_page_len
        observer_meta[request_index, 9] = page_start
        observer_meta[request_index, 10] = page_end


def _write_ag2_dcp_current_kv_and_scales(
    layer: torch.nn.Module,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    request_indices: torch.Tensor | None,
    request_count: int,
    current_kv_pack: torch.Tensor | None,
    scale_pack: torch.Tensor | None,
    observer_meta: torch.Tensor | None,
    match_fp8_new_tokens: bool,
    sm_scale: float,
) -> None:
    outputs = (current_kv_pack, scale_pack, observer_meta)
    if all(output is None for output in outputs):
        return
    if any(output is None for output in outputs):
        raise RuntimeError("DCP current-KV, scale and observer packs must be paired")
    assert current_kv_pack is not None
    assert scale_pack is not None
    assert observer_meta is not None
    if request_indices is None:
        if request_count == 0:
            # Profile/compile can execute before semantic request rows are
            # attached. Keep the complete pack invalid until a real plan
            # supplies row ownership.
            current_kv_pack.zero_()
            scale_pack.fill_(torch.nan)
            observer_meta.fill_(-1)
            return
        raise RuntimeError("DCP current-KV observer requires request row indices")
    if request_count > current_kv_pack.shape[0]:
        current_kv_pack.zero_()
        scale_pack.fill_(torch.nan)
        observer_meta.fill_(-1)
        return
    selected_key = torch.index_select(key, 0, request_indices[:request_count])
    selected_value = torch.index_select(value, 0, request_indices[:request_count])
    if current_kv_pack.shape[2:] != selected_key.shape[1:]:
        raise RuntimeError(
            "DCP current-KV observer geometry differs from consumed K/V: "
            f"{current_kv_pack.shape[2:]} != {selected_key.shape[1:]}"
        )
    current_kv_pack.zero_()
    current_kv_pack[:request_count, 0].copy_(selected_key)
    current_kv_pack[:request_count, 1].copy_(selected_value)
    scale_pack.fill_(torch.nan)
    # Avoid torch.tensor([...], device="cuda") here. This function executes
    # inside the qlen=1 FULL CUDA Graph and such construction is a CPU->CUDA
    # copy during capture. Device scalar fills are recorded by the graph and
    # these scales are immutable for the lifetime of the loaded layer.
    scale_pack[:request_count, 0].fill_(sm_scale)
    scale_pack[:request_count, 1].fill_(float(layer._q_scale_float))
    scale_pack[:request_count, 2].fill_(float(layer._k_scale_float))
    scale_pack[:request_count, 3].fill_(float(layer._v_scale_float))
    observer_meta[:request_count, 1].bitwise_or_(
        AG2_DCP_OBSERVER_CURRENT_KV_BIT | AG2_DCP_OBSERVER_SCALES_BIT
    )
    observer_meta[:request_count, 18] = int(match_fp8_new_tokens)
    observer_meta[:request_count, 20] = selected_key.shape[1]


def _match_fp8_cache_representation(
    value: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Round-trip live K/V through the exact static FP8 cache representation."""
    quantized = (value / scale).to(FP8_DTYPE)
    return quantized.to(value.dtype) * scale


class BatchDCPPrefillWrapper:
    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        dcp_a2a: bool = False,
        *,
        use_cuda_graph: bool = False,
        qo_indptr_buffer: torch.Tensor | None = None,
        paged_kv_indptr_buffer: torch.Tensor | None = None,
        paged_kv_indices_buffer: torch.Tensor | None = None,
        paged_kv_last_page_len_buffer: torch.Tensor | None = None,
        context_int_workspace_buffer: torch.Tensor | None = None,
        new_tokens_int_workspace_buffer: torch.Tensor | None = None,
    ):
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._convert_log2_lse_for_merge = (
            os.environ.get("AG2_VLLM_FLASHINFER_LOG2_LSE_MERGE", "0") == "1"
        )
        self._use_cuda_graph = use_cuda_graph
        self._disable_split_kv_for_cuda_graph = (
            use_cuda_graph
            and os.environ.get("AG2_VLLM_FLASHINFER_DCP_PREFILL_NO_SPLIT", "0") == "1"
        )
        self._ragged_fixed_split_size = int(
            os.environ.get(
                "AG2_VLLM_FLASHINFER_DCP_RAGGED_FIXED_SPLIT_SIZE",
                "-1",
            )
        )
        self._context_fixed_split_size = int(
            os.environ.get(
                "AG2_VLLM_FLASHINFER_DCP_CONTEXT_FIXED_SPLIT_SIZE",
                "-1",
            )
        )
        if self._context_fixed_split_size == 0 or self._context_fixed_split_size < -1:
            raise ValueError("DCP context fixed split size must be -1 or positive")
        self._context = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            get_kv_cache_layout(),
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buffer,
            paged_kv_indptr_buf=paged_kv_indptr_buffer,
            paged_kv_indices_buf=paged_kv_indices_buffer,
            paged_kv_last_page_len_buf=paged_kv_last_page_len_buffer,
        )
        # Candidate A may coexist with ordinary qlen>1 target verification in
        # one computational-prefill cohort. Its full-paged plan must not
        # overwrite the ordinary context plan used by the prefix of that
        # cohort, so keep a second PIECEWISE wrapper. It intentionally has no
        # CUDA-graph buffers: canonical-paged execution is rejected in graph
        # mode below.
        self._canonical_context = (
            None
            if use_cuda_graph
            else BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer,
                get_kv_cache_layout(),
            )
        )
        self._absolute_segment_context = (
            None
            if use_cuda_graph
            else BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer,
                get_kv_cache_layout(),
            )
        )
        self._absolute_segment_current = (
            None
            if use_cuda_graph
            else BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)
        )
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer,
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buffer,
            kv_indptr_buf=qo_indptr_buffer,
        )
        if use_cuda_graph:
            if (
                context_int_workspace_buffer is None
                or new_tokens_int_workspace_buffer is None
            ):
                raise ValueError(
                    "DCP prefill CUDA graph mode requires persistent context "
                    "and new-token integer workspaces."
                )
            self._context.reset_workspace_buffer(
                workspace_buffer, context_int_workspace_buffer
            )
            self._new_tokens.reset_workspace_buffer(
                workspace_buffer, new_tokens_int_workspace_buffer
            )
        self._local_kv_head_index_tensors: dict[
            tuple[tuple[int, ...], torch.device], torch.Tensor
        ] = {}
        self._ag2_request_tail_indices: torch.Tensor | None = None
        self._ag2_request_tail_count = 0
        self._ag2_history_qo_indptr_cpu: torch.Tensor | None = None
        self._ag2_history_paged_kv_indptr_cpu: torch.Tensor | None = None
        self._ag2_history_paged_kv_indices: torch.Tensor | None = None
        self._ag2_history_last_page_len_cpu: torch.Tensor | None = None
        self._ag2_request_trace_row = "tail"
        self._ag2_prefill_fixed_split_size = -1
        self._ag2_prefill_disable_split_kv = False
        self._ag2_window_left = -1
        self._ag2_dcp_world_size = 1
        self._ag2_sm_scale = 1.0
        self._ag2_logits_soft_cap = 0.0
        self._ag2_plan_ready = False
        self._canonical_paged = False
        self._canonical_paged_start_req = 0
        self._canonical_paged_start_token = 0
        self._canonical_paged_mask_bits = 0
        self._absolute_segmented = False
        self._absolute_segment_start_token = 0
        self._absolute_segment_context_query_indices: torch.Tensor | None = None

    def _get_local_kv_head_index_tensor(
        self,
        local_kv_head_indices: list[int] | tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        key = (tuple(local_kv_head_indices), device)
        tensor = self._local_kv_head_index_tensors.get(key)
        if tensor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DCP KV-head index tensor was not initialized before "
                    "CUDA graph capture"
                )
            tensor = torch.tensor(
                local_kv_head_indices,
                device=device,
                dtype=torch.long,
            )
            self._local_kv_head_index_tensors[key] = tensor
        return tensor

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
        canonical_paged: bool = False,
        absolute_segmented: bool = False,
        canonical_paged_start_req: int = 0,
        global_seq_lens_cpu: torch.Tensor | None = None,
        num_prompt_tokens_cpu: torch.Tensor | None = None,
        dcp_rank: int | None = None,
        dcp_kv_cache_interleave_size: int | None = None,
    ):
        """Plan the prefill operation with given parameters."""
        self._ag2_plan_ready = False
        self._canonical_paged = canonical_paged
        self._absolute_segmented = absolute_segmented
        self._canonical_paged_start_req = 0
        self._canonical_paged_start_token = 0
        self._absolute_segment_start_token = 0
        self._absolute_segment_context_query_indices = None
        if canonical_paged and absolute_segmented:
            raise ValueError(
                "Dense canonical paged and absolute-segment prefill are "
                "mutually exclusive"
            )
        context_fixed_split_size = (
            self._context_fixed_split_size
            if self._context_fixed_split_size > 0
            else prefill_fixed_split_size
        )
        self._ag2_prefill_fixed_split_size = context_fixed_split_size
        self._ag2_prefill_disable_split_kv = disable_split_kv
        self._ag2_window_left = window_left
        self._ag2_dcp_world_size = dcp_world_size
        self._ag2_sm_scale = sm_scale
        self._ag2_logits_soft_cap = logits_soft_cap or 0.0
        trace_tail_rows = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS",
                "0",
            )
        )
        if trace_tail_rows:
            request_tail_count = int(qo_indptr_cpu.numel()) - 1
            if request_tail_count > trace_tail_rows:
                raise RuntimeError(
                    "DCP request-tail trace capacity is smaller than the "
                    f"planned request count: {trace_tail_rows} < "
                    f"{request_tail_count}"
                )
            if (
                self._ag2_request_tail_indices is None
                or self._ag2_request_tail_indices.shape[0] != trace_tail_rows
                or self._ag2_request_tail_indices.device != paged_kv_indices.device
            ):
                self._ag2_request_tail_indices = torch.zeros(
                    trace_tail_rows,
                    dtype=qo_indptr_cpu.dtype,
                    device=paged_kv_indices.device,
                )
            trace_row = os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_ROW",
                "tail",
            )
            self._ag2_request_trace_row = trace_row
            tail_indices = _ag2_dcp_request_trace_indices(
                qo_indptr_cpu,
                trace_row,
            ).to(
                device=paged_kv_indices.device,
                non_blocking=True,
            )
            self._ag2_request_tail_indices[:request_tail_count].copy_(tail_indices)
            self._ag2_request_tail_count = request_tail_count
        if int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_KV_HISTORY_TOKENS",
                "0",
            )
        ):
            # The ordinary qlen4 DCP-prefill lane is PIECEWISE, so planning is
            # outside graph replay. Clone the small CPU metadata because the
            # builder reuses its staging tensors for later batches.
            if self._use_cuda_graph:
                raise RuntimeError(
                    "DCP prefill KV history observer is not CUDA-graph compatible"
                )
            self._ag2_history_qo_indptr_cpu = qo_indptr_cpu.clone()
            self._ag2_history_paged_kv_indptr_cpu = paged_kv_indptr_cpu.clone()
            self._ag2_history_paged_kv_indices = paged_kv_indices
            self._ag2_history_last_page_len_cpu = paged_kv_last_page_len_cpu.clone()
        if absolute_segmented:
            if self._use_cuda_graph:
                raise RuntimeError(
                    "Absolute-segment DCP prefill POC is PIECEWISE-only"
                )
            if window_left != -1:
                raise ValueError(
                    "Absolute-segment DCP prefill is not proven for sliding-window "
                    f"attention: {window_left=}"
                )
            if global_seq_lens_cpu is None or num_prompt_tokens_cpu is None:
                raise ValueError(
                    "Absolute-segment DCP prefill requires sequence and prompt "
                    "lengths"
                )
            if dcp_rank is None or dcp_kv_cache_interleave_size is None:
                raise ValueError(
                    "Absolute-segment DCP prefill requires explicit DCP geometry"
                )
            if (
                self._absolute_segment_context is None
                or self._absolute_segment_current is None
            ):
                raise RuntimeError(
                    "Absolute-segment DCP prefill has no PIECEWISE wrappers"
                )
            num_requests = int(qo_indptr_cpu.numel()) - 1
            if not 0 <= canonical_paged_start_req < num_requests:
                raise ValueError(
                    "Absolute-segment start request is outside the planned "
                    f"prefill cohort: {canonical_paged_start_req=} {num_requests=}"
                )
            self._canonical_paged_start_req = canonical_paged_start_req
            self._absolute_segment_start_token = int(
                qo_indptr_cpu[canonical_paged_start_req]
            )
            segment_size = int(
                os.environ.get(
                    "AG2_VLLM_DCP_ABSOLUTE_PREFILL_SEGMENT_SIZE",
                    "64",
                )
            )
            canonical_qo_indptr_cpu = (
                qo_indptr_cpu[canonical_paged_start_req:]
                - qo_indptr_cpu[canonical_paged_start_req]
            )
            canonical_global_seq_lens_cpu = global_seq_lens_cpu[
                canonical_paged_start_req:
            ]
            canonical_prompt_lens_cpu = num_prompt_tokens_cpu[
                canonical_paged_start_req:
            ]
            segments = _ag2_absolute_prefill_segments(
                global_seq_lens_cpu=canonical_global_seq_lens_cpu,
                qo_indptr_cpu=canonical_qo_indptr_cpu,
                num_prompt_tokens_cpu=canonical_prompt_lens_cpu,
                segment_size=segment_size,
            )

            canonical_page_offset = int(
                paged_kv_indptr_cpu[canonical_paged_start_req]
            )
            canonical_request_page_indptr = (
                paged_kv_indptr_cpu[canonical_paged_start_req:]
                - canonical_page_offset
            )
            canonical_page_indices = paged_kv_indices[canonical_page_offset:]
            context_page_parts: list[torch.Tensor] = []
            context_page_counts: list[int] = []
            context_last_page_lens: list[int] = []
            for segment_index_t in segments.context_segment_indices_cpu:
                segment_index = int(segment_index_t)
                request_index = int(segments.request_indices_cpu[segment_index])
                global_start = int(segments.global_starts_cpu[segment_index])
                local_len = int(
                    get_dcp_local_seq_lens(
                        torch.tensor([global_start], dtype=torch.int32),
                        dcp_world_size,
                        dcp_rank,
                        dcp_kv_cache_interleave_size,
                    )[0]
                )
                page_count = cdiv(local_len, page_size)
                request_page_start = int(
                    canonical_request_page_indptr[request_index]
                )
                request_page_end = int(
                    canonical_request_page_indptr[request_index + 1]
                )
                if page_count < 1 or request_page_start + page_count > request_page_end:
                    raise RuntimeError(
                        "Absolute-segment context exceeds the request page table: "
                        f"request={request_index} start={global_start} "
                        f"pages={page_count} available="
                        f"{request_page_end - request_page_start}"
                    )
                context_page_parts.append(
                    canonical_page_indices[
                        request_page_start : request_page_start + page_count
                    ]
                )
                context_page_counts.append(page_count)
                context_last_page_lens.append(local_len % page_size or page_size)

            metadata_device = paged_kv_indices.device
            context_lengths = [
                int(segments.qo_indptr_cpu[index + 1])
                - int(segments.qo_indptr_cpu[index])
                for index in segments.context_segment_indices_cpu.tolist()
            ]
            context_qo_indptr = [0]
            context_page_indptr = [0]
            for length in context_lengths:
                context_qo_indptr.append(context_qo_indptr[-1] + length)
            for count in context_page_counts:
                context_page_indptr.append(context_page_indptr[-1] + count)
            if context_lengths:
                self._absolute_segment_context.plan(
                    qo_indptr=torch.tensor(
                        context_qo_indptr,
                        dtype=torch.int32,
                        device=metadata_device,
                    ),
                    paged_kv_indptr=torch.tensor(
                        context_page_indptr,
                        dtype=torch.int32,
                        device=metadata_device,
                    ),
                    paged_kv_indices=torch.cat(context_page_parts),
                    paged_kv_last_page_len=torch.tensor(
                        context_last_page_lens,
                        dtype=torch.int32,
                        device=metadata_device,
                    ),
                    num_qo_heads=num_qo_heads * dcp_world_size,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    page_size=page_size,
                    causal=False,
                    sm_scale=sm_scale,
                    window_left=window_left,
                    logits_soft_cap=logits_soft_cap,
                    q_data_type=q_data_type,
                    kv_data_type=kv_cache_dtype,
                    fixed_split_size=context_fixed_split_size,
                    disable_split_kv=disable_split_kv,
                )
            self._absolute_segment_current.plan(
                qo_indptr=segments.qo_indptr_cpu.to(metadata_device),
                kv_indptr=segments.qo_indptr_cpu.to(metadata_device),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
                causal=True,
                sm_scale=sm_scale,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
                q_data_type=q_data_type,
                fixed_split_size=(
                    self._ragged_fixed_split_size
                    if self._ragged_fixed_split_size > 0
                    else prefill_fixed_split_size
                ),
                disable_split_kv=disable_split_kv,
            )
            self._absolute_segment_context_query_indices = (
                segments.context_query_indices_cpu.to(metadata_device)
            )
            if dcp_rank == 0:
                logger.info_once(
                    "Absolute-segment DCP prefill route: requests=%d "
                    "segments=%d context_segments=%d query_tokens=%d "
                    "page_references=%d segment_size=%d",
                    int(canonical_global_seq_lens_cpu.numel()),
                    int(segments.request_indices_cpu.numel()),
                    len(context_lengths),
                    int(segments.qo_indptr_cpu[-1]),
                    sum(context_page_counts),
                    segment_size,
                )

            if canonical_paged_start_req == 0:
                self._ag2_request_tail_count = 0
                self._ag2_history_qo_indptr_cpu = None
                self._ag2_history_paged_kv_indptr_cpu = None
                self._ag2_history_paged_kv_indices = None
                self._ag2_history_last_page_len_cpu = None
                self._ag2_plan_ready = True
                return

            qo_indptr_cpu = qo_indptr_cpu[: canonical_paged_start_req + 1]
            paged_kv_indptr_cpu = paged_kv_indptr_cpu[
                : canonical_paged_start_req + 1
            ]
            paged_kv_last_page_len_cpu = paged_kv_last_page_len_cpu[
                :canonical_paged_start_req
            ]
            if self._ag2_request_tail_indices is not None:
                tail_indices = _ag2_dcp_request_trace_indices(
                    qo_indptr_cpu,
                    self._ag2_request_trace_row,
                ).to(device=paged_kv_indices.device, non_blocking=True)
                self._ag2_request_tail_indices[:canonical_paged_start_req].copy_(
                    tail_indices
                )
            self._ag2_request_tail_count = canonical_paged_start_req
            if self._ag2_history_qo_indptr_cpu is not None:
                self._ag2_history_qo_indptr_cpu = qo_indptr_cpu.clone()
                self._ag2_history_paged_kv_indptr_cpu = (
                    paged_kv_indptr_cpu.clone()
                )
                self._ag2_history_last_page_len_cpu = (
                    paged_kv_last_page_len_cpu.clone()
                )

        if canonical_paged:
            if self._use_cuda_graph:
                raise RuntimeError(
                    "Canonical paged DCP prefill POC is PIECEWISE-only"
                )
            if global_seq_lens_cpu is None:
                raise ValueError(
                    "Canonical paged DCP prefill requires global sequence lengths"
                )
            if dcp_rank is None or dcp_kv_cache_interleave_size is None:
                raise ValueError(
                    "Canonical paged DCP prefill requires explicit DCP rank "
                    "and KV-cache interleave metadata"
                )
            num_requests = int(qo_indptr_cpu.numel()) - 1
            if not 0 <= canonical_paged_start_req < num_requests:
                raise ValueError(
                    "Canonical paged DCP start request is outside the planned "
                    f"prefill cohort: {canonical_paged_start_req=} "
                    f"{num_requests=}"
                )
            self._canonical_paged_start_req = canonical_paged_start_req
            self._canonical_paged_start_token = int(
                qo_indptr_cpu[canonical_paged_start_req].item()
            )
            canonical_qo_indptr_cpu = (
                qo_indptr_cpu[canonical_paged_start_req:]
                - qo_indptr_cpu[canonical_paged_start_req]
            )
            canonical_page_offset = int(
                paged_kv_indptr_cpu[canonical_paged_start_req].item()
            )
            canonical_paged_kv_indptr_cpu = (
                paged_kv_indptr_cpu[canonical_paged_start_req:]
                - canonical_page_offset
            )
            canonical_paged_kv_indices = paged_kv_indices[canonical_page_offset:]
            canonical_last_page_len_cpu = paged_kv_last_page_len_cpu[
                canonical_paged_start_req:
            ]
            canonical_global_seq_lens_cpu = global_seq_lens_cpu[
                canonical_paged_start_req:
            ]
            custom_mask, local_seq_lens = _dcp_causal_paged_custom_mask(
                canonical_global_seq_lens_cpu,
                canonical_qo_indptr_cpu,
                dcp_world_size,
                dcp_rank,
                dcp_kv_cache_interleave_size,
            )
            expected_local_seq_lens = (
                (
                    canonical_paged_kv_indptr_cpu[1:]
                    - canonical_paged_kv_indptr_cpu[:-1]
                    - 1
                )
                * page_size
                + canonical_last_page_len_cpu
            )
            if not torch.equal(
                local_seq_lens.to(expected_local_seq_lens.dtype),
                expected_local_seq_lens,
            ):
                raise RuntimeError(
                    "Canonical paged DCP mask and paged metadata disagree: "
                    f"mask={local_seq_lens.tolist()} "
                    f"paged={expected_local_seq_lens.tolist()}"
                )
            self._canonical_paged_mask_bits = custom_mask.numel()
            route_shape = (
                int(custom_mask.numel()),
                int(canonical_qo_indptr_cpu.numel() - 1),
                int(canonical_qo_indptr_cpu[-1]),
                int(local_seq_lens.max()),
            )
            if dcp_rank == 0 and route_shape not in _ag2_canonical_paged_logged_shapes:
                _ag2_canonical_paged_logged_shapes.add(route_shape)
                logger.info(
                    "Canonical paged DCP prefill runtime route: "
                    "mask_bits=%d requests=%d query_tokens=%d "
                    "max_local_kv_tokens=%d",
                    *route_shape,
                )
            max_mask_bits = int(
                os.environ.get(
                    "AG2_VLLM_DCP_CANONICAL_PAGED_MASK_MAX_BITS",
                    str(32 * 1024 * 1024),
                )
            )
            if max_mask_bits < 1 or custom_mask.numel() > max_mask_bits:
                raise RuntimeError(
                    "Canonical paged DCP bool-mask budget exceeded: "
                    f"required_bits={custom_mask.numel()} "
                    f"max_bits={max_mask_bits}. This bounded POC must not "
                    "silently allocate a production-scale dense mask."
                )
            canonical_fixed_split_size = int(
                os.environ.get(
                    "AG2_VLLM_DCP_CANONICAL_PAGED_FIXED_SPLIT_SIZE",
                    "4096",
                )
            )
            if canonical_fixed_split_size < 1:
                raise ValueError(
                    "Canonical paged DCP fixed split size must be positive"
                )
            metadata_device = paged_kv_indices.device
            if self._canonical_context is None:
                raise RuntimeError(
                    "Canonical paged DCP prefill has no PIECEWISE wrapper"
                )
            self._canonical_context.plan(
                qo_indptr=canonical_qo_indptr_cpu.to(metadata_device),
                paged_kv_indptr=canonical_paged_kv_indptr_cpu.to(metadata_device),
                paged_kv_indices=canonical_paged_kv_indices,
                paged_kv_last_page_len=canonical_last_page_len_cpu.to(
                    metadata_device
                ),
                num_qo_heads=num_qo_heads * dcp_world_size,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                custom_mask=custom_mask.to(metadata_device),
                causal=False,
                sm_scale=sm_scale,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
                q_data_type=q_data_type,
                kv_data_type=kv_cache_dtype,
                fixed_split_size=canonical_fixed_split_size,
                disable_split_kv=False,
            )
            if canonical_paged_start_req == 0:
                self._ag2_request_tail_count = 0
                self._ag2_history_qo_indptr_cpu = None
                self._ag2_history_paged_kv_indptr_cpu = None
                self._ag2_history_paged_kv_indices = None
                self._ag2_history_last_page_len_cpu = None
                self._ag2_plan_ready = True
                return

            # The prefix remains on the established paged-context + ragged-
            # query path. Rebind the local planning views so its metadata is
            # independent of the canonical suffix plan above.
            qo_indptr_cpu = qo_indptr_cpu[: canonical_paged_start_req + 1]
            paged_kv_indptr_cpu = paged_kv_indptr_cpu[
                : canonical_paged_start_req + 1
            ]
            paged_kv_last_page_len_cpu = paged_kv_last_page_len_cpu[
                :canonical_paged_start_req
            ]
            if self._ag2_request_tail_indices is not None:
                tail_indices = _ag2_dcp_request_trace_indices(
                    qo_indptr_cpu,
                    self._ag2_request_trace_row,
                ).to(device=paged_kv_indices.device, non_blocking=True)
                self._ag2_request_tail_indices[:canonical_paged_start_req].copy_(
                    tail_indices
                )
            self._ag2_request_tail_count = canonical_paged_start_req
            if self._ag2_history_qo_indptr_cpu is not None:
                self._ag2_history_qo_indptr_cpu = qo_indptr_cpu.clone()
                self._ag2_history_paged_kv_indptr_cpu = (
                    paged_kv_indptr_cpu.clone()
                )
                self._ag2_history_last_page_len_cpu = (
                    paged_kv_last_page_len_cpu.clone()
                )

        self._context.plan(
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=False,  # This is context run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            fixed_split_size=context_fixed_split_size,
            # Research draft: forcing launch-static split-K did not make M6
            # replay correct. Keep it available only for explicit reproduction,
            # not as part of the native DCP prefill CUDA Graph POC.
            disable_split_kv=(
                disable_split_kv or self._disable_split_kv_for_cuda_graph
            ),
        )
        self._new_tokens.plan(
            qo_indptr=qo_indptr_cpu,
            kv_indptr=qo_indptr_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=True,  # This is newtokens run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            fixed_split_size=(
                self._ragged_fixed_split_size
                if self._ragged_fixed_split_size > 0
                else prefill_fixed_split_size
            ),
            disable_split_kv=(
                disable_split_kv or self._disable_split_kv_for_cuda_graph
            ),
        )
        self._ag2_plan_ready = True

    def assert_plan_contract(
        self,
        *,
        window_left: int,
        logits_soft_cap: float,
        sm_scale: float,
    ) -> None:
        """Validate caller-owned plan state without FlashInfer private fields."""
        assert self._ag2_plan_ready
        assert self._ag2_window_left == window_left
        assert self._ag2_logits_soft_cap == logits_soft_cap
        assert self._ag2_sm_scale == sm_scale

    def run(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        dcp_output_pack: torch.Tensor | None = None,
        dcp_lse_pack: torch.Tensor | None = None,
        dcp_kv_history_pack: torch.Tensor | None = None,
        dcp_kv_history_meta: torch.Tensor | None = None,
        dcp_kv_page_indices_pack: torch.Tensor | None = None,
        dcp_observer_meta: torch.Tensor | None = None,
        dcp_current_kv_pack: torch.Tensor | None = None,
        dcp_scale_pack: torch.Tensor | None = None,
        match_fp8_new_tokens: bool = False,
    ):
        _write_ag2_dcp_prefill_kv_history_pack(
            kv_cache_tuple,
            qo_indptr_cpu=self._ag2_history_qo_indptr_cpu,
            paged_kv_indptr_cpu=self._ag2_history_paged_kv_indptr_cpu,
            paged_kv_indices=self._ag2_history_paged_kv_indices,
            paged_kv_last_page_len_cpu=self._ag2_history_last_page_len_cpu,
            history_pack=dcp_kv_history_pack,
            history_meta=dcp_kv_history_meta,
            page_indices_pack=dcp_kv_page_indices_pack,
            observer_meta=dcp_observer_meta,
            use_cuda_graph=self._use_cuda_graph,
            request_row=self._ag2_request_trace_row,
        )
        if dcp_observer_meta is not None:
            request_count = self._ag2_request_tail_count
            dcp_observer_meta[:request_count, 16] = self._ag2_prefill_fixed_split_size
            dcp_observer_meta[:request_count, 17] = int(
                self._ag2_prefill_disable_split_kv
            )
            dcp_observer_meta[:request_count, 19] = self._ag2_dcp_world_size
            dcp_observer_meta[:request_count, 21] = self._ag2_window_left
        if envs.AG2_VLLM_DCP_PREFILL_QUERY_SCRATCH:
            query = prefill_query.contiguous()
            dcp_group = get_dcp_group()
            if dcp_group.world_size != 3 or query.ndim != 3:
                raise RuntimeError(
                    "AG2 DCP query scratch requires DCP3 and a rank-3 query, got "
                    f"world_size={dcp_group.world_size} shape={tuple(query.shape)}"
                )
            scratch = _ag2_dcp_prefill_query_scratch
            output_elements = dcp_group.world_size * query.numel()
            if scratch is None or scratch.numel() < 2 * output_elements:
                raise RuntimeError(
                    "AG2 DCP query scratch is absent or too small: "
                    f"need={2 * output_elements} available="
                    f"{0 if scratch is None else scratch.numel()}"
                )
            flat_scratch = scratch.view(-1)
            raw_output = flat_scratch[:output_elements].view(
                dcp_group.world_size * query.shape[0],
                query.shape[1],
                query.shape[2],
            )
            prefill_query_across_dcp = flat_scratch[
                output_elements : 2 * output_elements
            ].view(
                query.shape[0],
                dcp_group.world_size * query.shape[1],
                query.shape[2],
            )
            dcp_group.all_gather_into(
                query, raw_output, prefill_query_across_dcp, dim=1
            )
        else:
            prefill_query_across_dcp = get_dcp_group().all_gather(
                prefill_query.contiguous(), dim=1
            )
        if self._absolute_segmented:
            if getattr(layer, "_ag2_dcp_trace_enabled", False):
                raise RuntimeError(
                    "Absolute-segment DCP prefill requires its dedicated "
                    "phase-tree artifact; the legacy request-tail trace is "
                    "not a compatible schema"
                )
            if self._absolute_segment_current is None:
                raise RuntimeError(
                    "Absolute-segment DCP prefill has no current-segment wrapper"
                )
            canonical_start = self._absolute_segment_start_token
            current_query = prefill_query[canonical_start:]
            current_key = key[canonical_start:]
            current_value = value[canonical_start:]
            if getattr(layer, "dcp_full_kv_attention_heads", False):
                local_kv_head_indices = getattr(
                    layer, "dcp_local_kv_head_indices", None
                )
                if local_kv_head_indices is None:
                    raise ValueError(
                        "DCP full-KV attention requires local KV head indices."
                    )
                local_kv_head_indices_tensor = (
                    self._get_local_kv_head_index_tensor(
                        local_kv_head_indices,
                        current_key.device,
                    )
                )
                current_key = torch.index_select(
                    current_key,
                    dim=1,
                    index=local_kv_head_indices_tensor,
                )
                current_value = torch.index_select(
                    current_value,
                    dim=1,
                    index=local_kv_head_indices_tensor,
                )
            current_output, current_lse = self._absolute_segment_current.run(
                current_query,
                current_key,
                current_value,
                q_scale=get_query_scale_for_flashinfer(
                    layer._q_scale_float,
                    current_query.dtype,
                ),
                return_lse=True,
            )
            segment_out = out[canonical_start:]
            segment_out.copy_(current_output)
            context_indices = self._absolute_segment_context_query_indices
            if context_indices is None:
                raise RuntimeError("Absolute-segment context row metadata is absent")
            if context_indices.numel():
                if self._absolute_segment_context is None:
                    raise RuntimeError(
                        "Absolute-segment DCP prefill has no context wrapper"
                    )
                context_query = torch.index_select(
                    prefill_query_across_dcp[canonical_start:],
                    0,
                    context_indices,
                )
                context_output_tmp, context_lse_tmp = (
                    self._absolute_segment_context.run(
                        context_query,
                        kv_cache_tuple,
                        q_scale=get_query_scale_for_flashinfer(
                            layer._q_scale_float,
                            context_query.dtype,
                        ),
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        return_lse=True,
                    )
                )
                context_output, context_lse = self._dcp_combine(
                    context_output_tmp,
                    context_lse_tmp,
                    get_dcp_group(),
                    return_lse=True,
                )
                selected_current_output = torch.index_select(
                    current_output,
                    0,
                    context_indices,
                )
                selected_current_lse = torch.index_select(
                    current_lse,
                    0,
                    context_indices,
                )
                merged_output = torch.empty_like(context_output)
                context_lse = context_lse.transpose(0, 1).contiguous()
                selected_current_lse = (
                    selected_current_lse.transpose(0, 1).contiguous()
                )
                if self._convert_log2_lse_for_merge:
                    context_lse = context_lse * 0.6931471805599453
                    selected_current_lse = (
                        selected_current_lse * 0.6931471805599453
                    )
                merge_attn_states(
                    merged_output,
                    context_output,
                    context_lse,
                    selected_current_output,
                    selected_current_lse,
                )
                segment_out.index_copy_(0, context_indices, merged_output)
            if canonical_start == 0:
                return out
            # A mixed computational-prefill wave may have qlen4 target
            # verification before the actual prompt suffix. Keep that prefix
            # on the established wrappers and route only the prompt segments.
            prefill_query = prefill_query[:canonical_start]
            prefill_query_across_dcp = prefill_query_across_dcp[:canonical_start]
            key = key[:canonical_start]
            value = value[:canonical_start]
            out = out[:canonical_start]

        if self._canonical_paged:
            if getattr(layer, "_ag2_dcp_trace_enabled", False):
                raise RuntimeError(
                    "Canonical paged DCP prefill cannot use the legacy "
                    "paged+ragged+merge trace schema. Reuse the validated "
                    "canonical saved-stage artifact or add a dedicated "
                    "single-paged-call trace schema before enabling it."
                )
            if self._canonical_context is None:
                raise RuntimeError(
                    "Canonical paged DCP prefill has no PIECEWISE wrapper"
                )
            canonical_start = self._canonical_paged_start_token
            canonical_output_tmp, canonical_lse_tmp = self._canonical_context.run(
                prefill_query_across_dcp[canonical_start:],
                kv_cache_tuple,
                q_scale=get_query_scale_for_flashinfer(
                    layer._q_scale_float,
                    prefill_query_across_dcp.dtype,
                ),
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                return_lse=True,
            )
            out[canonical_start:].copy_(
                self._dcp_combine(
                    canonical_output_tmp,
                    canonical_lse_tmp,
                    get_dcp_group(),
                )
            )
            if canonical_start == 0:
                return out
            # Preserve the established target-verification computation for the
            # prefix of a mixed wave. Only the actual prompt suffix uses the
            # canonical full-paged call.
            prefill_query = prefill_query[:canonical_start]
            prefill_query_across_dcp = prefill_query_across_dcp[:canonical_start]
            key = key[:canonical_start]
            value = value[:canonical_start]
            out = out[:canonical_start]

        output_context_tmp, lse_context_tmp = self._context.run(
            prefill_query_across_dcp,
            kv_cache_tuple,
            q_scale=get_query_scale_for_flashinfer(
                layer._q_scale_float,
                prefill_query_across_dcp.dtype,
            ),
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        _copy_ag2_dcp_trace(layer, "prefill_context_local_output", output_context_tmp)
        _copy_ag2_dcp_trace(layer, "prefill_context_local_lse", lse_context_tmp)
        _write_ag2_dcp_aux_pack(
            layer,
            "local_output",
            output_context_tmp,
            dcp_output_pack,
            dcp_lse_pack,
            request_tail_indices=self._ag2_request_tail_indices,
            request_tail_count=self._ag2_request_tail_count,
            dcp_observer_meta=dcp_observer_meta,
        )
        _write_ag2_dcp_aux_pack(
            layer,
            "local_lse",
            lse_context_tmp,
            dcp_output_pack,
            dcp_lse_pack,
            request_tail_indices=self._ag2_request_tail_indices,
            request_tail_count=self._ag2_request_tail_count,
            dcp_observer_meta=dcp_observer_meta,
        )
        del prefill_query_across_dcp
        output_context, lse_context = self._dcp_combine(
            output_context_tmp,
            lse_context_tmp,
            get_dcp_group(),
            return_lse=True,
        )
        _copy_ag2_dcp_trace(layer, "prefill_context_combined_output", output_context)
        _copy_ag2_dcp_trace(layer, "prefill_context_combined_lse", lse_context)
        _write_ag2_dcp_aux_pack(
            layer,
            "combined_output",
            output_context,
            dcp_output_pack,
            dcp_lse_pack,
            request_tail_indices=self._ag2_request_tail_indices,
            request_tail_count=self._ag2_request_tail_count,
            dcp_observer_meta=dcp_observer_meta,
        )
        _write_ag2_dcp_aux_pack(
            layer,
            "combined_lse",
            lse_context,
            dcp_output_pack,
            dcp_lse_pack,
            request_tail_indices=self._ag2_request_tail_indices,
            request_tail_count=self._ag2_request_tail_count,
            dcp_observer_meta=dcp_observer_meta,
        )
        del output_context_tmp, lse_context_tmp
        lse_context = lse_context.transpose(0, 1).contiguous()

        if getattr(layer, "dcp_full_kv_attention_heads", False):
            local_kv_head_indices = getattr(layer, "dcp_local_kv_head_indices", None)
            if local_kv_head_indices is None:
                raise ValueError(
                    "DCP full-KV attention requires local KV head indices."
                )
            local_kv_head_indices_tensor = self._get_local_kv_head_index_tensor(
                local_kv_head_indices, key.device
            )
            key = torch.index_select(key, dim=1, index=local_kv_head_indices_tensor)
            value = torch.index_select(value, dim=1, index=local_kv_head_indices_tensor)

        if match_fp8_new_tokens:
            key = _match_fp8_cache_representation(key, layer._k_scale)
            value = _match_fp8_cache_representation(value, layer._v_scale)

        _write_ag2_dcp_current_kv_and_scales(
            layer,
            key,
            value,
            request_indices=self._ag2_request_tail_indices,
            request_count=self._ag2_request_tail_count,
            current_kv_pack=dcp_current_kv_pack,
            scale_pack=dcp_scale_pack,
            observer_meta=dcp_observer_meta,
            match_fp8_new_tokens=match_fp8_new_tokens,
            sm_scale=self._ag2_sm_scale,
        )

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            q_scale=get_query_scale_for_flashinfer(
                layer._q_scale_float,
                prefill_query.dtype,
            ),
            return_lse=True,
        )
        lse_query = lse_query.transpose(0, 1).contiguous()
        _copy_ag2_dcp_trace(layer, "prefill_query_output", output_query)
        _copy_ag2_dcp_trace(layer, "prefill_query_lse", lse_query.transpose(0, 1))
        _write_ag2_dcp_aux_pack(
            layer,
            "query_output",
            output_query,
            dcp_output_pack,
            dcp_lse_pack,
            request_tail_indices=self._ag2_request_tail_indices,
            request_tail_count=self._ag2_request_tail_count,
            dcp_observer_meta=dcp_observer_meta,
        )
        _write_ag2_dcp_aux_pack(
            layer,
            "query_lse",
            lse_query.transpose(0, 1),
            dcp_output_pack,
            dcp_lse_pack,
            request_tail_indices=self._ag2_request_tail_indices,
            request_tail_count=self._ag2_request_tail_count,
            dcp_observer_meta=dcp_observer_meta,
        )

        # FlashInfer returns base-2 LSE, while vLLM's generic
        # merge_attn_states kernel consumes natural-log LSE.
        if self._convert_log2_lse_for_merge:
            lse_context = lse_context * 0.6931471805599453
            lse_query = lse_query * 0.6931471805599453

        merge_attn_states(
            out,
            output_context,
            lse_context,
            output_query,
            lse_query,
        )
        _copy_ag2_dcp_trace(layer, "prefill_merged_output", out)
        _write_ag2_dcp_aux_pack(
            layer,
            "merged_output",
            out,
            dcp_output_pack,
            dcp_lse_pack,
            request_tail_indices=self._ag2_request_tail_indices,
            request_tail_count=self._ag2_request_tail_count,
            dcp_observer_meta=dcp_observer_meta,
        )
        return out


class BatchDCPPseudoPrefillWrapper:
    """Read causal-clipped speculative rows from paged KV in one DCP phase."""

    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        dcp_a2a: bool = False,
    ):
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._paged = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            get_kv_cache_layout(),
        )
        self._ag2_window_left = -1
        self._ag2_logits_soft_cap = 0.0
        self._ag2_sm_scale = 1.0
        self._ag2_plan_ready = False

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ) -> None:
        self._ag2_plan_ready = False
        self._paged.plan(
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            # Every row's paged length is already clipped to its inclusive
            # causal boundary, so no query-relative mask is needed.
            causal=False,
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            fixed_split_size=prefill_fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self._ag2_window_left = window_left
        self._ag2_logits_soft_cap = logits_soft_cap or 0.0
        self._ag2_sm_scale = sm_scale
        self._ag2_plan_ready = True

    def assert_plan_contract(
        self,
        *,
        window_left: int,
        logits_soft_cap: float,
        sm_scale: float,
    ) -> None:
        """Validate caller-owned plan state without FlashInfer private fields."""
        assert self._ag2_plan_ready
        assert self._ag2_window_left == window_left
        assert self._ag2_logits_soft_cap == logits_soft_cap
        assert self._ag2_sm_scale == sm_scale

    def run(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor,
    ) -> torch.Tensor:
        query_across_dcp = get_dcp_group().all_gather(
            query.contiguous(),
            dim=1,
        )
        output_tmp, lse = self._paged.run(
            query_across_dcp,
            kv_cache_tuple,
            q_scale=get_query_scale_for_flashinfer(
                layer._q_scale_float,
                query_across_dcp.dtype,
            ),
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        del query_across_dcp
        out.copy_(
            self._dcp_combine(
                output_tmp,
                lse,
                get_dcp_group(),
            )
        )
        return out


class BatchDCPBatchedDecodeWrapper:
    """Run every speculative target row in one fixed-split qlen1 batch."""

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        fixed_split_size: int,
        dcp_a2a: bool = False,
    ):
        if fixed_split_size < 1:
            raise ValueError("batched DCP decode requires fixed_split_size >= 1")
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._fixed_split_size = fixed_split_size
        self._decode = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            get_kv_cache_layout(),
            use_tensor_cores=True,
            backend="auto",
        )
        self._num_rows = 0

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ) -> None:
        del qo_indptr_cpu, prefill_fixed_split_size
        self._num_rows = paged_kv_last_page_len_cpu.shape[0]
        if paged_kv_indptr_cpu.shape[0] != self._num_rows + 1:
            raise ValueError(
                "batched DCP decode paged indptr must contain one entry per "
                f"row plus the terminator, got {paged_kv_indptr_cpu.shape[0]} "
                f"for {self._num_rows} rows"
            )
        self._decode.plan(
            paged_kv_indptr_cpu,
            paged_kv_indices,
            paged_kv_last_page_len_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            o_data_type=q_data_type,
            sm_scale=sm_scale,
            fixed_split_size=self._fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def run(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape[0] != self._num_rows:
            raise ValueError(
                f"batched DCP decode expected {self._num_rows} rows, "
                f"received {query.shape[0]}"
            )
        query_across_dcp = get_dcp_group().all_gather(
            query.contiguous(),
            dim=1,
        )
        output_tmp, lse = self._decode.run(
            query_across_dcp,
            kv_cache_tuple,
            q_scale=get_query_scale_for_flashinfer(
                layer._q_scale_float,
                query_across_dcp.dtype,
            ),
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        del query_across_dcp
        out.copy_(
            self._dcp_combine(
                output_tmp,
                lse,
                get_dcp_group(),
            )
        )
        return out


class BatchDCPSequentialDecodeWrapper:
    """Run speculative target rows as phase-batched native qlen1 decode."""

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_phases: int,
        fixed_split_size: int,
        dcp_a2a: bool = False,
    ):
        if num_phases < 2:
            raise ValueError("sequential DCP decode requires at least two phases")
        if fixed_split_size < 1:
            raise ValueError("sequential DCP decode requires fixed_split_size >= 1")
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._num_phases = num_phases
        self._fixed_split_size = fixed_split_size
        phase_workspace_size = workspace_buffer.numel() // num_phases
        minimum_phase_workspace = 128 * 1024 * 1024
        if phase_workspace_size < minimum_phase_workspace:
            raise ValueError(
                "Sequential DCP decode requires at least 128 MiB of FlashInfer "
                f"float workspace per phase; received {workspace_buffer.numel()} "
                f"bytes for {num_phases} phases."
            )
        # FlashInfer stores split-K planning state in the float workspace.
        # Give every phase a disjoint slice of the already-reserved global
        # buffer; sharing it makes earlier plans deterministically wrong.
        phase_workspaces = [
            workspace_buffer.narrow(
                0,
                phase * phase_workspace_size,
                phase_workspace_size,
            )
            for phase in range(num_phases)
        ]
        self._phases = [
            BatchDecodeWithPagedKVCacheWrapper(
                phase_workspace,
                get_kv_cache_layout(),
                use_tensor_cores=True,
                backend="auto",
            )
            for phase_workspace in phase_workspaces
        ]
        self._num_requests = 0

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ) -> None:
        del qo_indptr_cpu, prefill_fixed_split_size
        num_rows = paged_kv_last_page_len_cpu.shape[0]
        if num_rows % self._num_phases:
            raise ValueError(
                f"phase-major row count {num_rows} is not divisible by "
                f"{self._num_phases}"
            )
        self._num_requests = num_rows // self._num_phases
        for phase, wrapper in enumerate(self._phases):
            request_start = phase * self._num_requests
            request_end = request_start + self._num_requests
            page_start = int(paged_kv_indptr_cpu[request_start])
            page_end = int(paged_kv_indptr_cpu[request_end])
            phase_indptr = (
                paged_kv_indptr_cpu[request_start : request_end + 1] - page_start
            )
            wrapper.plan(
                phase_indptr,
                paged_kv_indices[page_start:page_end],
                paged_kv_last_page_len_cpu[request_start:request_end],
                num_qo_heads=num_qo_heads * dcp_world_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                window_left=window_left,
                logits_soft_cap=logits_soft_cap,
                q_data_type=q_data_type,
                kv_data_type=kv_cache_dtype,
                o_data_type=q_data_type,
                sm_scale=sm_scale,
                fixed_split_size=self._fixed_split_size,
                disable_split_kv=disable_split_kv,
            )

    def run(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache_tuple: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor,
    ) -> torch.Tensor:
        expected_rows = self._num_phases * self._num_requests
        if query.shape[0] != expected_rows:
            raise ValueError(
                f"sequential DCP decode expected {expected_rows} rows, "
                f"received {query.shape[0]}"
            )
        # Scheduler order is request-major. Metadata and kernel execution are
        # phase-major so each wrapper sees one qlen1 row per request.
        phase_major_query = (
            query.view(self._num_requests, self._num_phases, *query.shape[1:])
            .transpose(0, 1)
            .contiguous()
            .view(expected_rows, *query.shape[1:])
        )
        query_across_dcp = get_dcp_group().all_gather(
            phase_major_query,
            dim=1,
        )
        phase_outputs = []
        phase_lses = []
        for phase, wrapper in enumerate(self._phases):
            row_start = phase * self._num_requests
            row_end = row_start + self._num_requests
            output_tmp, lse_tmp = wrapper.run(
                query_across_dcp[row_start:row_end],
                kv_cache_tuple,
                q_scale=get_query_scale_for_flashinfer(
                    layer._q_scale_float,
                    query_across_dcp.dtype,
                ),
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                return_lse=True,
            )
            phase_outputs.append(output_tmp)
            phase_lses.append(lse_tmp)
        del query_across_dcp
        phase_major_output = self._dcp_combine(
            torch.cat(phase_outputs),
            torch.cat(phase_lses),
            get_dcp_group(),
        )
        out.copy_(
            phase_major_output.view(
                self._num_phases,
                self._num_requests,
                *phase_major_output.shape[1:],
            )
            .transpose(0, 1)
            .reshape_as(out)
        )
        return out


class FlashInferBackend(AttentionBackend):
    supports_dcp_full_kv_attention_heads: ClassVar[bool] = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Page sizes >= 128 only run on the trtllm-gen dynamic kernel (GQA/MQA
        # on Blackwell); advertise them only when usable so selection never
        # picks a large kernel block we cannot serve.
        use_large_pages = False
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            pc = vllm_config.parallel_config
            mc = vllm_config.model_config
            num_qo_heads = mc.get_num_attention_heads(pc)
            num_kv_heads = mc.get_num_kv_heads(pc)
            use_large_pages = (
                num_kv_heads > 0
                and num_qo_heads // num_kv_heads > 1
                and current_platform.is_device_capability_family(100)
                and can_use_trtllm_attention(num_qo_heads, num_kv_heads)
            )
        if not use_large_pages:
            return [16, 32, 64]
        return [16, 32, 64, 128, 256, 512, 1024]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "nvfp4":
            full_dim = nvfp4_kv_cache_full_dim(head_size)
            return (num_blocks, 2 * num_kv_heads, block_size, full_dim)
        # Pack K and V in the content dim (B, H, N, 2*hs).
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` (logical (B, H, N, 2*hs)) to the actual memory
        # layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, block_size, num_kv_heads, 2*head_size)
            return (1, 0, 3, 2, 4)
        elif cache_layout == "NHD":
            # (num_blocks, block_size, num_kv_heads, 2*head_size)
            stride_order = (0, 2, 1, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, block_size, 2*head_size)
            return (1, 2, 0, 3, 4)
        elif cache_layout == "HND":
            # (num_blocks, num_kv_heads, block_size, 2*head_size)
            stride_order = (0, 1, 2, 3)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        elif kv_cache_dtype == "nvfp4":
            return torch.uint8
        else:
            raise ValueError(f"Unrecognized dtype: {kv_cache_dtype}")

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype == "nvfp4":
            return (
                current_platform.is_device_capability_family(100)
                and supports_trtllm_attention(is_prefill=True)
                and supports_trtllm_attention(is_prefill=False)
            )
        return super().supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256, 512]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FlashInfer supports SM75+, but is currently broken on SM75 (Turing):
        # https://github.com/flashinfer-ai/flashinfer/issues/3620 (fix:
        # https://github.com/flashinfer-ai/flashinfer/pull/3621). Temporarily
        # raise the floor to SM80 so it is not auto-selected on SM75 until
        # that fix lands; revert to DeviceCapability(7, 5) once it does.
        return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """FlashInfer supports sinks only on the SM100 trtllm-gen path."""
        from vllm.utils.flashinfer import (
            force_use_trtllm_attention,
        )

        # Respect explicit disable flag (e.g.,
        # --attention-config.use_trtllm_attention=0)
        if force_use_trtllm_attention() is False:
            return False

        if not current_platform.is_device_capability_family(100):
            return False

        # Check if TRTLLM is supported on this platform
        return supports_trtllm_attention(
            is_prefill=False
        ) and supports_trtllm_attention(is_prefill=True)

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            return "HND"
        return None

    forward_includes_kv_cache_update: bool = False


@dataclass
class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: (
        BatchPrefillWithPagedKVCacheWrapper
        | BatchDCPPrefillWrapper
        | BatchDCPPseudoPrefillWrapper
        | BatchDCPBatchedDecodeWrapper
        | BatchDCPSequentialDecodeWrapper
    )
    match_fp8_new_tokens: bool = False


@dataclass
class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper


class FlashInferDecodeKernel(Enum):
    """Decode kernels selected inside the FlashInfer backend."""

    XQA = "xqa"
    TRTLLM_GEN = "trtllm-gen"


@dataclass
class TRTLLMPrefill:
    """Metadata for the TRTLLM prefill pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to prefill requests.
    Shape: [num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to prefill requests.
    Shape: [num_prefills]
    """

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor

    max_q_len: int
    """
    The maximum query length *among prefill requests*.
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class FlashInferTrtllmAPIDecode:
    """Metadata for decode paths using FlashInfer's TRTLLM decode API.

    FlashInfer exposes both XQA (SM90) and trtllm-gen (SM100) through
    ``trtllm_batch_decode_with_kv_cache``.  Keep them as distinct vLLM
    decode kernels because their dtype/layout/output constraints differ.
    """

    kernel: FlashInferDecodeKernel

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to decode requests.
    Shape: [num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to decode requests.
    Shape: [num_decodes]
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class FlashInferMetadata:
    num_actual_tokens: int
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    # The data types of the query for prefill and decode.
    # On SM90, these two data types may be different.
    q_data_type_prefill: torch.dtype
    q_data_type_decode: torch.dtype

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    causal: bool

    prefill: FIPrefill | TRTLLMPrefill | None
    """
    Holds the metadata for the prefill portion of the batch.
    Will be `None` if `num_prefill_tokens == 0`.
    """

    decode: FIDecode | FlashInferTrtllmAPIDecode | None
    """
    Holds the metadata for the decode portion of the batch.
    Will be `None` if `num_decode_tokens == 0`.
    """

    # --- Special Case: Cascade Attention ---

    use_cascade: bool
    """
    If True, the entire batch is a cascade attention call, and the
    `prefill` and `decode` fields will both be None.
    """

    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    kv_cache_spec: AttentionSpec
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config
        self._workspace_buffer = None
        self._prefill_wrapper: (
            BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper | None
        ) = None  # Wrapper for prefill/append
        self._dcp_prefill_cudagraph_enabled = (
            os.environ.get("AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH", "0") == "1"
        )
        self._dcp_prefill_wrappers_cudagraph: dict[int, BatchDCPPrefillWrapper] = {}
        self._dcp_prefill_captured_wrappers: dict[int, BatchDCPPrefillWrapper] = {}
        self._dcp_prefill_qo_indptr_buffer: torch.Tensor | None = None
        self._dcp_prefill_context_int_workspace: torch.Tensor | None = None
        self._dcp_prefill_new_tokens_int_workspace: torch.Tensor | None = None
        self._dcp_pseudo_prefill_wrapper: BatchDCPPseudoPrefillWrapper | None = None
        self._dcp_batched_decode_wrapper: BatchDCPBatchedDecodeWrapper | None = None
        self._dcp_batched_decode_workspace: torch.Tensor | None = None
        self._dcp_sequential_decode_wrapper: BatchDCPSequentialDecodeWrapper | None = (
            None
        )
        self._noncausal_prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = (
            None  # Wrapper for non-causal prefill (DFlash)
        )
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        prefill_batch_invariant = envs.VLLM_BATCH_INVARIANT or (
            os.environ.get(
                "AG2_VLLM_FLASHINFER_PREFILL_BATCH_INVARIANT",
                "0",
            )
            == "1"
        )
        if envs.VLLM_BATCH_INVARIANT:
            self.decode_fixed_split_size = 2048
            self.decode_disable_split_kv = True
            self.decode_fixed_split_spec_target_only = False
        else:
            self.qlen1_fixed_split_size = int(
                os.environ.get(
                    "AG2_VLLM_FLASHINFER_Q1_FIXED_SPLIT_SIZE",
                    "-1",
                )
            )
            self.qlen1_disable_split_kv = (
                os.environ.get(
                    "AG2_VLLM_FLASHINFER_Q1_DISABLE_SPLIT_KV",
                    "0",
                )
                == "1"
            )
            if self.qlen1_fixed_split_size == 0 or self.qlen1_fixed_split_size < -1:
                raise ValueError(
                    "FlashInfer qlen1 fixed split size must be -1 or positive"
                )
            if self.qlen1_fixed_split_size > 0:
                logger.warning_once(
                    "Research POC: FlashInfer qlen1 fixed-split planning is "
                    "enabled (fixed_split_size=%d, disable_split_kv=%s). "
                    "FULL CUDA Graphs remain enabled; long-context graph "
                    "portability is not yet accepted.",
                    self.qlen1_fixed_split_size,
                    self.qlen1_disable_split_kv,
                )
            self.decode_fixed_split_size = int(
                os.environ.get(
                    "AG2_VLLM_FLASHINFER_SPEC_TARGET_FIXED_SPLIT_SIZE",
                    "-1",
                )
            )
            self.decode_disable_split_kv = (
                os.environ.get(
                    "AG2_VLLM_FLASHINFER_SPEC_TARGET_DISABLE_SPLIT_KV",
                    "0",
                )
                == "1"
            )
            self.decode_fixed_split_spec_target_only = True
            if self.decode_fixed_split_size == 0 or self.decode_fixed_split_size < -1:
                raise ValueError(
                    "FlashInfer speculative-target fixed split size must be "
                    "-1 or positive"
                )
            if self.decode_fixed_split_size > 0:
                logger.warning_once(
                    "Research POC: FlashInfer fixed-split planning is enabled "
                    "only for multi-token speculative target verification "
                    "(fixed_split_size=%d, disable_split_kv=%s); qlen=1 "
                    "decode and its FULL CUDA graphs are unchanged.",
                    self.decode_fixed_split_size,
                    self.decode_disable_split_kv,
                )
        if envs.VLLM_BATCH_INVARIANT:
            self.qlen1_fixed_split_size = -1
            self.qlen1_disable_split_kv = False
        if prefill_batch_invariant:
            self.prefill_fixed_split_size = 4096
            self.prefill_disable_split_kv = True
            if not envs.VLLM_BATCH_INVARIANT:
                logger.warning_once(
                    "Research POC: FlashInfer prefill-only batch-invariant "
                    "planning is enabled (fixed_split_size=4096, "
                    "disable_split_kv=True); decode and GDN are unchanged."
                )
        else:
            self.prefill_fixed_split_size = -1
            self.prefill_disable_split_kv = False

        self.compilation_config = vllm_config.compilation_config
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len, self.kv_cache_spec.block_size
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        speculative_config = vllm_config.speculative_config
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        self.enable_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        if self.enable_cuda_graph:
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[
                int, BatchDecodeWithPagedKVCacheWrapper
            ] = {}
            self._decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs
            if self.compilation_config.max_cudagraph_capture_size is not None:
                self._decode_cudagraph_max_bs = min(
                    self._decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.dcp_kv_cache_interleave_size = (
                vllm_config.parallel_config.dcp_kv_cache_interleave_size
            )
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.dcp_kv_cache_interleave_size = 1
        self.use_dcp = self.dcp_world_size > 1
        self._dcp_canonical_paged_prefill = (
            os.environ.get("AG2_VLLM_DCP_CANONICAL_PAGED_PREFILL", "0") == "1"
        )
        self._dcp_absolute_segment_prefill = (
            os.environ.get("AG2_VLLM_DCP_ABSOLUTE_SEGMENT_PREFILL", "0") == "1"
        )
        if (
            self._dcp_canonical_paged_prefill
            and self._dcp_absolute_segment_prefill
        ):
            raise ValueError(
                "Dense canonical paged and absolute-segment DCP prefill POCs "
                "are mutually exclusive"
            )
        if self._dcp_canonical_paged_prefill:
            if not self.use_dcp:
                raise ValueError(
                    "AG2_VLLM_DCP_CANONICAL_PAGED_PREFILL requires DCP"
                )
            logger.warning_once(
                "Research-only bounded canonical paged DCP prompt-prefill is "
                "enabled. Actual causal prompt prefills with no cascade will "
                "read the full just-updated paged KV through one custom-mask "
                "FlashInfer call; target/MTP decode and non-causal paths are "
                "unchanged."
            )
        if self._dcp_absolute_segment_prefill:
            if not self.use_dcp:
                raise ValueError(
                    "AG2_VLLM_DCP_ABSOLUTE_SEGMENT_PREFILL requires DCP"
                )
            logger.warning_once(
                "Research-only absolute-segment DCP prompt-prefill is enabled. "
                "Actual prompt rows use stable paged-history + ragged-current "
                "phase trees; target/MTP decode and non-causal paths are unchanged."
            )
        if self._dcp_prefill_cudagraph_enabled:
            if not self.use_dcp:
                raise ValueError(
                    "AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH requires DCP."
                )
            if not self.enable_cuda_graph:
                raise ValueError(
                    "AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH requires FULL "
                    "CUDA graph mode."
                )
            self._dcp_prefill_qo_indptr_buffer = torch.empty(
                max_num_reqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # FlashInfer allocates 8 MiB per prefill wrapper by default. The
            # plan/run lifecycle is sequential across graph shapes, so retain
            # one persistent workspace per phase instead of one per shape.
            self._dcp_prefill_context_int_workspace = torch.empty(
                8 * 1024 * 1024,
                dtype=torch.uint8,
                device=self.device,
            )
            self._dcp_prefill_new_tokens_int_workspace = torch.empty(
                8 * 1024 * 1024,
                dtype=torch.uint8,
                device=self.device,
            )
            logger.warning_once(
                "Research-only FlashInfer DCP prefill CUDA Graph support is "
                "enabled: qlen=%d, max_reqs=%d, persistent integer workspace="
                "16 MiB per attention metadata builder.",
                1 + int(num_spec_tokens or 0),
                max_num_reqs,
            )
        self._dcp_pseudo_decode_enabled = (
            os.environ.get("AG2_VLLM_MTP_DCP_PSEUDO_DECODE", "0") == "1"
        )
        self._dcp_sequential_decode_enabled = (
            os.environ.get("AG2_VLLM_MTP_DCP_SEQUENTIAL_DECODE", "0") == "1"
        )
        self._dcp_batched_decode_enabled = (
            os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "0") == "1"
        )
        self._dcp_sequential_fixed_split_size = int(
            os.environ.get(
                "AG2_VLLM_MTP_DCP_SEQUENTIAL_FIXED_SPLIT_SIZE",
                "4",
            )
        )
        self._dcp_batched_workspace_mib = int(
            os.environ.get(
                "AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB",
                "132",
            )
        )
        self._dcp_batched_fixed_split_size = int(
            os.environ.get(
                "AG2_VLLM_MTP_DCP_BATCHED_FIXED_SPLIT_SIZE",
                "4",
            )
        )
        self._dcp_match_fp8_new_tokens = (
            os.environ.get("AG2_VLLM_MTP_DCP_MATCH_FP8_NEW_TOKENS", "0") == "1"
        )
        self._dcp_prefill_match_fp8_new_tokens = (
            os.environ.get("AG2_VLLM_DCP_PREFILL_MATCH_FP8_NEW_TOKENS", "0") == "1"
        )
        enabled_special_paths = sum(
            (
                self._dcp_pseudo_decode_enabled,
                self._dcp_sequential_decode_enabled,
                self._dcp_batched_decode_enabled,
            )
        )
        if enabled_special_paths > 1:
            raise ValueError(
                "DCP pseudo-prefill, sequential-decode and batched-decode "
                "research paths are mutually exclusive."
            )
        self._dcp_special_decode_enabled = (
            self._dcp_pseudo_decode_enabled
            or self._dcp_sequential_decode_enabled
            or self._dcp_batched_decode_enabled
        )
        self._dcp_pseudo_decode_query_len = 1 + int(num_spec_tokens or 0)
        self._dcp_pseudo_decode_max_rows = (
            1 + int(num_spec_tokens or 0)
        ) * max_num_reqs
        self._dcp_pseudo_decode_block_tables: torch.Tensor | None = None
        self._dcp_pseudo_decode_row_to_req: torch.Tensor | None = None
        if self._dcp_special_decode_enabled and not self.use_dcp:
            raise ValueError(
                "DCP speculative attention research paths require DCP world size > 1."
            )
        self.dcp_a2a = (
            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        # Compatible with models with non-uniform per-layer head counts.
        self.num_qo_heads = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(self.vllm_config.parallel_config)

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size
        if self._dcp_special_decode_enabled:
            # Worker-side hybrid block tables may reserve entries beyond the
            # nominal max sequence for the currently scheduled token batch.
            # The metadata builder receives only the kernel-sized spec, not
            # the original hybrid storage geometry, so include one complete
            # scheduler batch as a conservative pre-capture bound.
            max_local_blocks = _dcp_pseudo_block_table_capacity(
                self.model_config.max_model_len,
                self.max_num_batched_tokens,
                self.page_size,
                self.dcp_world_size,
            )
            self._dcp_pseudo_decode_block_tables = torch.empty(
                (self._dcp_pseudo_decode_max_rows, max_local_blocks),
                dtype=torch.int32,
                device=self.device,
            )
            self._dcp_pseudo_decode_row_to_req = torch.empty(
                self._dcp_pseudo_decode_max_rows,
                dtype=torch.int64,
                device=self.device,
            )

        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            self.cache_dtype = self.cache_config.cache_dtype
            # Cannot use self.kv_cache_spec.dtype here because kv_cache_spec
            # storage dtype may not be the same as the op dtype (uint8 vs fp8_e4m3)
            self.is_kvcache_nvfp4 = self.cache_dtype == "nvfp4"
            if self.is_kvcache_nvfp4:
                if (
                    force_use_trtllm_attention() is False
                    or not supports_trtllm_attention(is_prefill=True)
                    or not supports_trtllm_attention(is_prefill=False)
                ):
                    raise ValueError(
                        "--kv-cache-dtype nvfp4 requires the SM100 trtllm-gen "
                        "FlashInfer path."
                    )
                # For NVFP4, kv_cache_dtype stays as the string "nvfp4"
                # which is passed to FlashInferImpl
                self.kv_cache_dtype = self.cache_dtype
            else:
                self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(
                    self.cache_dtype
                )
        else:
            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        if (
            self._dcp_match_fp8_new_tokens or self._dcp_prefill_match_fp8_new_tokens
        ) and (not self.use_dcp or not self.cache_dtype.startswith("fp8")):
            raise ValueError(
                "Matching live new-token K/V to the FP8 cache "
                "requires DCP and an FP8 KV cache."
            )
        if self._dcp_match_fp8_new_tokens:
            logger.warning_once(
                "Research-only DCP speculative new-token FP8 matching is enabled. "
                "Only uniform target decode queries with is_prefilling=false "
                "will round-trip live K/V through the cache representation."
            )
        if self._dcp_prefill_match_fp8_new_tokens:
            logger.warning_once(
                "Research-only causal DCP prefill new-token FP8 matching is enabled. "
                "Every native causal DCP prefill will round-trip live K/V through "
                "the cache representation to remove scheduler-dependent mixed "
                "BF16/FP8 attention inputs."
            )

        # Compute per-phase Q dtype.  On SM90 (XQA decode), the prefill and
        # decode phases require different Q dtypes when the KV cache is FP8
        # (FP8-Q for the FI native prefill, BF16/FP16-Q for XQA decode),
        # so both values must be tracked independently.
        self.q_data_type_prefill = self.get_q_data_type(is_prefill=True)
        self.q_data_type_decode = self.get_q_data_type(is_prefill=False)

        # Prefer TRTLLM/XQA for decoding whenever supported. The decode kernel
        # must be selected statically for FULL cudagraph capture.
        can_use_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            self.num_qo_heads, self.num_kv_heads, is_prefill=False
        )
        # Page sizes >= 128 require the trtllm-gen GQA/MQA path (guaranteed by
        # get_supported_kernel_block_sizes).
        assert self.page_size <= 64 or (
            current_platform.is_device_capability_family(100)
            and can_use_xqa_or_trtllm_gen_decode
            and self.num_qo_heads // self.num_kv_heads > 1
        ), f"Unexpected FlashInfer page size {self.page_size} without trtllm-gen GQA"
        self.use_trtllm_decode_attention = can_use_xqa_or_trtllm_gen_decode
        self.flashinfer_trtllm_api_decode_kernel: FlashInferDecodeKernel | None = (
            self._get_flashinfer_trtllm_api_decode_kernel()
            if can_use_xqa_or_trtllm_gen_decode
            else None
        )
        if (
            self.use_dcp
            and self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
        ):
            logger.warning_once(
                "FlashInfer XQA decode does not support returning LSE and "
                "therefore does not support DCP, reverting to native FlashInfer "
                "decode."
            )
            self.use_trtllm_decode_attention = False
            self.flashinfer_trtllm_api_decode_kernel = None
        supports_spec_as_decode = (
            self.flashinfer_trtllm_api_decode_kernel
            == FlashInferDecodeKernel.TRTLLM_GEN
        )
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=supports_spec_as_decode,
            # trtllm-gen decode receives no cp_rank/global-seq-len information,
            # so its end-aligned causal mask is wrong for q_len > 1 over the
            # DCP-interleaved local KV shard (spec token i misses up to
            # (dcp_world_size - 1) * (q_len - 1 - i) KV entries, including its
            # own). Keep the threshold at 1 under DCP so spec queries take the
            # DCP-aware prefill path, until the kernel is CP-aware (compare
            # flash_attn_varlen_func's cp_world_size/cp_rank/cp_tot_seqused_k).
            supports_dcp_with_varlen=False,
        )

        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        # TODO: discard this for trtllm-gen backend
        per_layer_parameters = get_per_layer_parameters(
            vllm_config, layer_names, FlashInferImpl
        )
        if current_platform.is_device_capability(90) and any(
            params.window_left != -1 for params in per_layer_parameters.values()
        ):
            # FlashInfer SM90 sliding-window prefill is not reliable with FP8-Q:
            # https://github.com/flashinfer-ai/flashinfer/issues/3578
            raise NotImplementedError(
                "FlashInfer backend on SM90 currently crashes with "
                "sliding-window attention layers. Use the default attention "
                "backend."
            )
        self.global_hyperparameters = infer_global_hyperparameters(per_layer_parameters)
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and not FlashInferBackend.supports_sink():
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs."
            )
        capability = current_platform.get_device_capability()
        arch = f"sm{capability.major}{capability.minor}" if capability else "unknown"
        decode_backend = (
            self.flashinfer_trtllm_api_decode_kernel.value
            if self.flashinfer_trtllm_api_decode_kernel is not None
            else "flashinfer-native"
        )
        logger.info_once(
            "FlashInfer resolved query dtypes: prefill=%s, decode=%s, "
            "decode_backend=%s, kv_cache_dtype=%s, arch=%s",
            self.q_data_type_prefill,
            self.q_data_type_decode,
            decode_backend,
            self.kv_cache_dtype,
            arch,
        )
        if self._dcp_pseudo_decode_enabled:
            logger.warning_once(
                "Research-only DCP pseudo-decode is enabled: target qlen>1 "
                "decode rows will alias the request block table and run as "
                "independent causal-clipped qlen=1 rows through one paged-"
                "prefill kernel and one DCP combine."
            )
        if self._dcp_sequential_decode_enabled:
            if self.q_data_type_prefill != self.q_data_type_decode:
                raise ValueError(
                    "Sequential DCP decode currently requires identical "
                    "prefill and decode query dtypes."
                )
            logger.warning_once(
                "Research-only DCP sequential decode is enabled: target "
                "qlen>1 rows will run as phase-batched native qlen=1 "
                "FlashInfer decode operations with one DCP gather/combine."
            )
        # Preparing persistent buffers
        # Since we do not have explicit synchronization in ModelRunnerV2, we do not pin
        # reused CPU buffers to avoid a race condition between step N async copies to
        # GPU and step N+1 buffer updates.
        self.pin_memory = not vllm_config.use_v2_model_runner and PIN_MEMORY
        metadata_max_reqs = max_num_reqs
        if self._dcp_special_decode_enabled:
            metadata_max_reqs = self._dcp_pseudo_decode_max_rows
            max_local_pages_per_row = cdiv(
                self.model_config.max_model_len,
                self.page_size * self.dcp_world_size,
            )
            max_num_pages = max(
                max_num_pages,
                metadata_max_reqs * max_local_pages_per_row,
            )
        self.paged_kv_indptr = self._make_buffer(metadata_max_reqs + 1)
        self.paged_kv_indptr_cpu_buffer = torch.zeros_like(
            self.paged_kv_indptr.cpu, pin_memory=self.pin_memory
        )  # Extra buffer for mutable paged_kv_indptr.cpu in cuda graph mode
        self.paged_kv_indices = self._make_buffer(max_num_pages)
        self.paged_kv_last_page_len = self._make_buffer(metadata_max_reqs)

    # Keep SM90 prefill/decode Q dtype selection in one place.
    def get_q_data_type(self, is_prefill: bool) -> torch.dtype:
        # The user sets --attention-config.disable_flashinfer_q_quantization
        # to 1 explicitly, use model dtype for query.
        if self.vllm_config.attention_config.disable_flashinfer_q_quantization:
            return self.model_config.dtype

        # self.cache_dtype is resolved per KV-cache group: it is "auto" when
        # this group is unquantized (e.g. --kv-cache-dtype-skip-layers), even
        # if cache_config requests a quantized dtype globally.
        cache_dtype = self.cache_dtype

        # On SM90, XQA decode requires BF16/FP16-Q even with FP8 KV cache.
        # FI native prefill on SM90 still uses FP8-Q in that case.
        if (
            current_platform.is_device_capability(90)
            and not is_prefill
            and force_use_trtllm_attention() is not False
            and cache_dtype.startswith("fp8")
        ):
            return self.model_config.dtype

        # Otherwise, match Q dtype to the KV cache dtype.
        if cache_dtype.startswith("fp8"):
            # FP8-Q requires an fp8 tensor-core attention path
            # (FI native fa3 on SM90, trtllm-gen/XQA on SM100).
            # Architectures with only fa2 (e.g. SM89, SM120) cannot
            # consume FP8 queries, so keep the model dtype for Q there.
            if current_platform.is_device_capability(
                90
            ) or current_platform.is_device_capability_family(100):
                return FlashInferBackend.get_dtype_for_flashinfer(cache_dtype)
            return self.model_config.dtype
        if cache_dtype == "nvfp4":
            return FlashInferBackend.get_dtype_for_flashinfer("fp8_e4m3")
        return self.kv_cache_spec.dtype

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype = torch.int32
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=True,
        )

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        """Get the cudagraph support level for FlashInfer attention.

        The SM90 XQA integration only enables single-token decode today. Keep
        specdec CUDA graphs limited to trtllm-gen until vLLM wires the XQA
        specdec mask.
        """
        if current_platform.is_device_capability(90):
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        if (
            os.environ.get("AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH", "0") == "1"
            and vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            return AttentionCGSupport.UNIFORM_BATCH

        # For UniformTypeKVCacheSpecs, check all contained specs
        kv_specs = (
            kv_cache_spec.kv_cache_specs.values()
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
            else [kv_cache_spec]
        )
        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        has_trtllm_support: bool = len(kv_specs) > 0
        for spec in kv_specs:
            if not isinstance(spec, AttentionSpec):
                # FlashInfer only applies to attention, so we don't consider other types
                # of KV spec (e.g. Mamba) here. This is mostly for type checking.
                continue
            if not can_use_trtllm_attention(
                num_qo_heads=num_qo_heads,
                num_kv_heads=spec.num_kv_heads,
                is_prefill=False,
            ):
                has_trtllm_support = False
                break

        # trtllm-gen only supports causal attention.
        if has_trtllm_support and not vllm_config.attention_config.use_non_causal:
            return AttentionCGSupport.UNIFORM_BATCH
        else:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
            if envs.VLLM_BATCH_INVARIANT:
                buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
            else:
                # FlashInfer prefill temp buffers (batch_prefill_tmp_v, ...)
                # scale with the prefill chunk and query-head footprint, NOT
                # context length. The fixed ~394 MiB default is too small for
                # wide-head models at the default 8192-token chunk on some
                # archs (e.g. sm_120), where FlashInfer hard-errors instead of
                # growing. Size to the batch's head footprint; never shrink
                # below the configured default.
                est = (
                    self.max_num_batched_tokens
                    * self.num_qo_heads
                    * self.head_dim
                    * FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM
                )
                buffer_size = max(buffer_size, est)
            self._workspace_buffer = torch.zeros(
                buffer_size, dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        self._workspace_buffer = workspace_buffer

    @staticmethod
    def _get_flashinfer_trtllm_api_decode_kernel() -> FlashInferDecodeKernel:
        if current_platform.is_device_capability(90):
            return FlashInferDecodeKernel.XQA
        assert current_platform.is_device_capability_family(100)
        return FlashInferDecodeKernel.TRTLLM_GEN

    def _get_prefill_wrapper(
        self,
        causal: bool = True,
        *,
        cudagraph_batch_size: int | None = None,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if cudagraph_batch_size is not None:
            if not causal or not self.use_dcp:
                raise ValueError(
                    "DCP prefill CUDA graph wrappers require causal DCP attention."
                )
            wrapper = self._dcp_prefill_wrappers_cudagraph.get(cudagraph_batch_size)
            if wrapper is None:
                qo_indptr = self._dcp_prefill_qo_indptr_buffer
                context_int_workspace = self._dcp_prefill_context_int_workspace
                new_tokens_int_workspace = self._dcp_prefill_new_tokens_int_workspace
                assert qo_indptr is not None
                assert context_int_workspace is not None
                assert new_tokens_int_workspace is not None
                wrapper = BatchDCPPrefillWrapper(
                    workspace_buffer=self._get_workspace_buffer(),
                    dcp_a2a=self.dcp_a2a,
                    use_cuda_graph=True,
                    qo_indptr_buffer=qo_indptr[: cudagraph_batch_size + 1],
                    paged_kv_indptr_buffer=self.paged_kv_indptr.gpu[
                        : cudagraph_batch_size + 1
                    ],
                    paged_kv_indices_buffer=self.paged_kv_indices.gpu,
                    paged_kv_last_page_len_buffer=self.paged_kv_last_page_len.gpu[
                        :cudagraph_batch_size
                    ],
                    context_int_workspace_buffer=context_int_workspace,
                    new_tokens_int_workspace_buffer=new_tokens_int_workspace,
                )
                self._dcp_prefill_wrappers_cudagraph[cudagraph_batch_size] = wrapper
            return wrapper

        if not causal:
            if self.use_dcp:
                raise NotImplementedError(
                    "FlashInfer non-causal prefill is not supported with DCP yet."
                )
            if self.is_kvcache_nvfp4:
                raise NotImplementedError(
                    "FlashInfer non-causal attention is not supported with "
                    "NVFP4 KV cache."
                )
            if self._noncausal_prefill_wrapper is None:
                self._noncausal_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(),
                    get_kv_cache_layout(),
                    backend="auto",
                )
            return self._noncausal_prefill_wrapper

        if self._prefill_wrapper is None:
            if self.use_dcp:
                self._prefill_wrapper = BatchDCPPrefillWrapper(
                    workspace_buffer=self._get_workspace_buffer(),
                    dcp_a2a=self.dcp_a2a,
                )
            else:
                # NVFP4 KV cache requires the trtllm-gen backend inside
                # the wrapper; fa2/fa3 do not support nvfp4.
                backend = "trtllm-gen" if self.is_kvcache_nvfp4 else "auto"
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(),
                    get_kv_cache_layout(),
                    backend=backend,
                )
        assert self._prefill_wrapper is not None
        return self._prefill_wrapper

    def _dcp_prefill_cudagraph_batch_size(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_decodes: int,
        num_prefills: int,
        *,
        for_cudagraph_capture: bool = False,
    ) -> int | None:
        """Return the fixed graph batch size for a uniform MTP verification."""
        if (
            not self._dcp_prefill_cudagraph_enabled
            or not self.use_dcp
            or common_attn_metadata.causal is not True
            or num_decodes != 0
            or num_prefills <= 0
        ):
            return None

        query_lens = (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        active_rows = query_lens != 0
        nonzero_query_lens = query_lens[active_rows]
        if nonzero_query_lens.numel() == 0 or bool(
            torch.any(nonzero_query_lens != self._dcp_pseudo_decode_query_len)
        ):
            return None

        diagnostic_allow_unverified = (
            os.environ.get(
                "AG2_VLLM_DIAGNOSTIC_ALLOW_UNVERIFIED_DCP_PREFILL_GRAPH",
                "0",
            )
            == "1"
        )
        if not for_cudagraph_capture and not diagnostic_allow_unverified:
            num_decode_draft_tokens_cpu = (
                common_attn_metadata.num_decode_draft_tokens_cpu
            )
            if num_decode_draft_tokens_cpu is None:
                return None
            if num_decode_draft_tokens_cpu.numel() != query_lens.numel():
                return None
            active_draft_counts = num_decode_draft_tokens_cpu[active_rows]
            if bool(
                torch.any(active_draft_counts != self._dcp_pseudo_decode_query_len - 1)
            ):
                return None

        is_prefilling = common_attn_metadata.is_prefilling
        if is_prefilling is not None:
            if isinstance(is_prefilling, torch.Tensor):
                if bool(torch.any(is_prefilling[active_rows]).item()):
                    return None
            elif bool(np.any(np.asarray(is_prefilling)[active_rows.numpy()])):
                return None
        return num_prefills

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> FlashInferMetadata:
        """Build metadata with the same persistent wrapper replay will plan.

        Capture batches are synthetic and therefore have no authoritative
        per-request draft markers.  The capture role may select the qlen3
        wrapper from physical shape, while ordinary runtime builds remain
        fail-closed on ``num_decode_draft_tokens_cpu``.
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            for_cudagraph_capture=True,
        )

    def _get_dcp_pseudo_prefill_wrapper(self) -> BatchDCPPseudoPrefillWrapper:
        if self._dcp_pseudo_prefill_wrapper is None:
            self._dcp_pseudo_prefill_wrapper = BatchDCPPseudoPrefillWrapper(
                workspace_buffer=self._get_workspace_buffer(),
                dcp_a2a=self.dcp_a2a,
            )
        return self._dcp_pseudo_prefill_wrapper

    def _verify_dcp_prefill_cudagraph_wrapper_identity(
        self,
        batch_size: int,
        wrapper: BatchDCPPrefillWrapper,
        *,
        for_cudagraph_capture: bool,
    ) -> None:
        """Require capture and replay planning to target one wrapper object."""
        if for_cudagraph_capture:
            captured = self._dcp_prefill_captured_wrappers.setdefault(
                batch_size, wrapper
            )
            if captured is not wrapper:
                raise RuntimeError(
                    "DCP prefill CUDA graph capture changed wrapper identity "
                    f"for batch_size={batch_size}"
                )
            return

        captured = self._dcp_prefill_captured_wrappers.get(batch_size)
        if captured is None:
            raise RuntimeError(
                "DCP prefill CUDA graph runtime selected an uncaptured wrapper "
                f"for batch_size={batch_size}"
            )
        if captured is not wrapper:
            raise RuntimeError(
                "DCP prefill CUDA graph capture/replay wrapper mismatch for "
                f"batch_size={batch_size}"
            )

    def _get_dcp_batched_decode_wrapper(
        self,
    ) -> BatchDCPBatchedDecodeWrapper:
        if self._dcp_batched_decode_wrapper is None:
            if self._dcp_batched_workspace_mib < 1:
                raise ValueError(
                    "AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB must be positive"
                )
            self._dcp_batched_decode_workspace = torch.zeros(
                self._dcp_batched_workspace_mib * 1024 * 1024,
                dtype=torch.uint8,
                device=self.device,
            )
            self._dcp_batched_decode_wrapper = BatchDCPBatchedDecodeWrapper(
                workspace_buffer=self._dcp_batched_decode_workspace,
                fixed_split_size=self._dcp_batched_fixed_split_size,
                dcp_a2a=self.dcp_a2a,
            )
        return self._dcp_batched_decode_wrapper

    def _get_dcp_sequential_decode_wrapper(
        self,
    ) -> BatchDCPSequentialDecodeWrapper:
        if self._dcp_sequential_decode_wrapper is None:
            self._dcp_sequential_decode_wrapper = BatchDCPSequentialDecodeWrapper(
                workspace_buffer=self._get_workspace_buffer(),
                num_phases=self._dcp_pseudo_decode_query_len,
                fixed_split_size=self._dcp_sequential_fixed_split_size,
                dcp_a2a=self.dcp_a2a,
            )
        return self._dcp_sequential_decode_wrapper

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None
            # NVFP4 KV cache requires the trtllm-gen backend inside
            # the wrapper; fa2/fa3 do not support nvfp4.
            backend = "trtllm-gen" if self.is_kvcache_nvfp4 else "auto"
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
                use_tensor_cores=True,
                backend=backend,
            )

            # save the decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    def _get_cascade_wrapper(self):
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), get_kv_cache_layout()
            )
        return self._cascade_wrapper

    def _is_uniform_dcp_target_decode(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> bool:
        if (
            not self.use_dcp
            or common_prefix_len != 0
            or common_attn_metadata.causal is not True
        ):
            return False

        query_lens = (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        if query_lens.numel() == 0 or bool(
            torch.any(query_lens != self._dcp_pseudo_decode_query_len)
        ):
            return False
        if int(query_lens.sum()) != common_attn_metadata.num_actual_tokens:
            return False

        is_prefilling = common_attn_metadata.is_prefilling
        if is_prefilling is None:
            return False
        if isinstance(is_prefilling, torch.Tensor):
            if bool(torch.any(is_prefilling[: query_lens.numel()]).item()):
                return False
        elif bool(np.any(np.asarray(is_prefilling)[: query_lens.numel()])):
            return False

        return True

    def _should_match_fp8_new_tokens(
        self,
        *,
        causal: bool,
        num_prefills: int,
        use_dcp_pseudo_decode: bool,
        uniform_target_decode: bool,
    ) -> bool:
        if self._dcp_match_fp8_new_tokens and uniform_target_decode:
            return True
        return (
            self._dcp_prefill_match_fp8_new_tokens
            and self.use_dcp
            and causal
            and num_prefills > 0
            and not use_dcp_pseudo_decode
        )

    def _can_use_dcp_pseudo_decode(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> bool:
        if not self._dcp_special_decode_enabled or not (
            self._is_uniform_dcp_target_decode(
                common_prefix_len,
                common_attn_metadata,
            )
        ):
            return False
        if common_attn_metadata.num_actual_tokens > self._dcp_pseudo_decode_max_rows:
            raise ValueError(
                "DCP pseudo-decode row capacity is too small: "
                f"{common_attn_metadata.num_actual_tokens=} > "
                f"{self._dcp_pseudo_decode_max_rows=}."
            )
        return True

    def _prepare_dcp_pseudo_decode(
        self,
        seq_lens_cpu: torch.Tensor,
        qo_indptr_cpu: torch.Tensor,
        block_table_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_to_req_cpu, local_row_lens_cpu = _dcp_pseudo_decode_rows(
            seq_lens_cpu,
            qo_indptr_cpu,
            self.dcp_world_size,
            self.dcp_rank,
            self.dcp_kv_cache_interleave_size,
        )
        if self._dcp_sequential_decode_enabled:
            num_rows = row_to_req_cpu.numel()
            if num_rows % self._dcp_pseudo_decode_query_len:
                raise ValueError(
                    f"sequential DCP row count {num_rows} is not divisible by "
                    f"{self._dcp_pseudo_decode_query_len}"
                )
            num_requests = num_rows // self._dcp_pseudo_decode_query_len
            phase_major_order = (
                torch.arange(num_rows)
                .view(num_requests, self._dcp_pseudo_decode_query_len)
                .transpose(0, 1)
                .reshape(-1)
            )
            row_to_req_cpu = row_to_req_cpu[phase_major_order]
            local_row_lens_cpu = local_row_lens_cpu[phase_major_order]
        num_rows = row_to_req_cpu.numel()
        block_table_width = block_table_tensor.shape[1]
        pseudo_block_tables = self._dcp_pseudo_decode_block_tables
        pseudo_row_to_req = self._dcp_pseudo_decode_row_to_req
        assert pseudo_block_tables is not None
        assert pseudo_row_to_req is not None
        if block_table_width > pseudo_block_tables.shape[1]:
            raise ValueError(
                "DCP pseudo-decode block-table capacity is too small: "
                f"{block_table_width=} > "
                f"{pseudo_block_tables.shape[1]=}."
            )

        row_to_req = pseudo_row_to_req[:num_rows]
        row_to_req.copy_(row_to_req_cpu, non_blocking=True)
        selected_block_tables = pseudo_block_tables[:num_rows, :block_table_width]
        torch.index_select(
            block_table_tensor,
            0,
            row_to_req,
            out=selected_block_tables,
        )
        return selected_block_tables, local_row_lens_cpu

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        """
        Compute paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len for FlashInfer
        attention.

        Results are stored in self.paged_kv_indptr,
        self.paged_kv_indices, self.paged_kv_last_page_len buffers.

        Returns paged_kv_indices, a GPU tensor with shape [num_actual_pages].
        """
        max_num_blocks = int(num_blocks_np.max(initial=0))
        if max_num_blocks > block_table_tensor.shape[1]:
            raise ValueError(
                "FlashInfer paged-KV metadata requires more pages than the "
                "block table provides: "
                f"max_num_blocks={max_num_blocks}, "
                f"block_table_width={block_table_tensor.shape[1]}, "
                f"page_size={page_size}, num_reqs={num_reqs}."
            )
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],
        )
        # NOTE(woosuk): Because self.paged_kv_indptr_cpu can be modified
        # after this line (e.g., for cuda graphs), we need to copy the data to
        # self.paged_kv_indptr_buffer to avoid race condition.
        self.paged_kv_indptr_cpu_buffer[: num_reqs + 1] = self.paged_kv_indptr.cpu[
            : num_reqs + 1
        ]
        paged_kv_indptr = self.paged_kv_indptr.gpu[: num_reqs + 1]
        paged_kv_indptr.copy_(
            self.paged_kv_indptr_cpu_buffer[: num_reqs + 1], non_blocking=True
        )

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_reqs]
        paged_kv_indices = self.paged_kv_indices.gpu[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(
            self.paged_kv_last_page_len.cpu[:num_reqs], non_blocking=True
        )
        return paged_kv_indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        for_cudagraph_capture: bool = False,
    ) -> FlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        causal = common_attn_metadata.causal
        use_dcp_pseudo_decode = self._can_use_dcp_pseudo_decode(
            common_prefix_len,
            common_attn_metadata,
        )
        uniform_target_decode = self._is_uniform_dcp_target_decode(
            common_prefix_len,
            common_attn_metadata,
        )
        if causal:
            generic_split = split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        else:
            # FlashInfer decode/TRTLLM paths cannot express non-causal
            # query-query attention, so DFlash runs as native prefill.
            num_decodes = 0
            num_prefills = num_reqs
            num_decode_tokens = 0
            num_prefill_tokens = num_actual_tokens
            generic_split = (
                num_decodes,
                num_prefills,
                num_decode_tokens,
                num_prefill_tokens,
            )
            semantic_split = generic_split
        semantic_split, canonical_paged_start_req, use_cascade = (
            _ag2_resolve_canonical_prefill_route(
                common_attn_metadata=common_attn_metadata,
                generic_split=generic_split,
                common_prefix_len=common_prefix_len,
                enabled=(
                    self._dcp_canonical_paged_prefill
                    or self._dcp_absolute_segment_prefill
                ),
                causal=bool(causal),
                use_dcp=self.use_dcp,
                use_dcp_pseudo_decode=use_dcp_pseudo_decode,
            )
        )
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            semantic_split
        )
        canonical_paged_prefill = canonical_paged_start_req is not None
        match_fp8_new_tokens = self._should_match_fp8_new_tokens(
            causal=causal,
            num_prefills=num_prefills,
            use_dcp_pseudo_decode=use_dcp_pseudo_decode,
            uniform_target_decode=uniform_target_decode,
        )
        if use_dcp_pseudo_decode:
            # Every flattened verification token becomes an independent
            # single-query paged-prefill row. The output remains in the
            # original flattened token order, so no permutation is required.
            num_reqs = num_actual_tokens
            num_decodes = 0
            num_prefills = num_actual_tokens
            num_decode_tokens = 0
            num_prefill_tokens = num_actual_tokens

        page_size = self.page_size
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu
        pseudo_seq_lens_cpu: torch.Tensor | None = None
        if use_dcp_pseudo_decode:
            original_seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            (
                block_table_tensor,
                pseudo_seq_lens_cpu,
            ) = self._prepare_dcp_pseudo_decode(
                original_seq_lens_cpu,
                qo_indptr_cpu,
                block_table_tensor,
            )
            qo_indptr_cpu = torch.arange(
                num_actual_tokens + 1,
                dtype=torch.int32,
                device="cpu",
            )

        # Step 1: Decide which dispatch modes to use:
        # - Cascade attention (distinct mode)
        # - Prefill (FI native or TRTLLM)
        # - Decode (FI native, XQA, or trtllm-gen)
        if (
            self._dcp_canonical_paged_prefill
            or self._dcp_absolute_segment_prefill
        ) and causal and self.use_dcp:
            is_prefilling = common_attn_metadata.is_prefilling
            has_actual_prefill = False
            if is_prefilling is not None:
                active_lifecycle = is_prefilling[: common_attn_metadata.num_reqs]
                has_actual_prefill = (
                    bool(torch.any(active_lifecycle).item())
                    if isinstance(active_lifecycle, torch.Tensor)
                    else bool(np.any(active_lifecycle))
                )
            if has_actual_prefill:
                _ag2_record_canonical_route_provenance(
                    common_attn_metadata=common_attn_metadata,
                    common_prefix_len=common_prefix_len,
                    generic_split=generic_split,
                    semantic_split=semantic_split,
                    canonical_paged_prefill=canonical_paged_prefill,
                    canonical_mode=(
                        "absolute_segment"
                        if self._dcp_absolute_segment_prefill
                        and canonical_paged_prefill
                        else (
                            "dense_paged"
                            if self._dcp_canonical_paged_prefill
                            and canonical_paged_prefill
                            else "off"
                        )
                    ),
                    canonical_paged_start_req=canonical_paged_start_req,
                    use_cascade=use_cascade,
                    dcp_rank=self.dcp_rank,
                    for_cudagraph_capture=for_cudagraph_capture,
                )
        uses_spec_reorder = self.reorder_batch_threshold > 1
        # Page sizes >= 128 must use trtllm-gen; force it for prefill too.
        prefill_force_trtllm = (
            True if page_size >= 128 else self.attention_config.use_trtllm_attention
        )
        prefill_use_trtllm = not use_dcp_pseudo_decode and (
            causal
            and use_trtllm_attention(
                self.num_qo_heads,
                self.num_kv_heads,
                num_prefill_tokens,
                max_seq_len,
                self.dcp_world_size,
                self.cache_dtype,
                self.q_data_type_prefill,
                is_prefill=True,
                force_use_trtllm=prefill_force_trtllm,
                has_sinks=self.has_sinks,
                has_spec=uses_spec_reorder,
            )
        )
        decode_with_flashinfer_trtllm_api = causal and self.use_trtllm_decode_attention

        if not causal and self.use_dcp:
            raise NotImplementedError(
                "FlashInfer non-causal prefill is not supported with DCP yet."
            )
        if not causal and self.use_trtllm_decode_attention:
            logger.warning_once(
                "Using FlashInfer for draft model non-causal attention; TRTLLM "
                "can still be used for target model causal attention."
            )
        all_uses_trtllm = causal and (
            (num_prefills == 0 or prefill_use_trtllm)
            and (num_decodes == 0 or decode_with_flashinfer_trtllm_api)
        )

        if not all_uses_trtllm:
            if self.has_sinks:
                raise NotImplementedError(
                    "FlashInfer backend currently does not support attention "
                    "sinks, please use trtllm on blackwell or flash attention "
                    "on earlier GPUs."
                )

            if not self.global_hyperparameters.has_same_window_lefts:
                raise ValueError(
                    "Window left is not the same for all layers. "
                    "One potential fix is to set disable_sliding_window=True"
                )

            assert self.global_hyperparameters.has_same_all_params, (
                "FlashInfer backend currently only supports models in which "
                "all layers share the same values for the following "
                "hyperparameters: `window_left`, `logits_soft_cap`, "
                "`sm_scale`."
            )

        # Step 2: Initialize the output metadata
        # Leave prefill/decode/cascade_wrapper empty, to be populated
        # case by case depending on the batch contents and backend selection.
        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type_prefill=self.q_data_type_prefill,
            q_data_type_decode=self.q_data_type_decode,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            causal=causal,
            use_cascade=use_cascade,
            prefill=None,
            decode=None,
            cascade_wrapper=None,
        )

        # Guard access to seq_lens_cpu, which may not always be needed
        # and can be expensive to retrieve in async mode.
        # When all attention (both prefill and decode) uses TRTLLM,
        # seq_lens_cpu is not needed since TRTLLM paths use GPU tensors
        # (block_tables, seq_lens) directly.
        needs_seq_lens_cpu = self.use_dcp or use_cascade or not all_uses_trtllm
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
        if use_dcp_pseudo_decode:
            assert pseudo_seq_lens_cpu is not None
            seq_lens_cpu = pseudo_seq_lens_cpu
            seq_lens_np = seq_lens_cpu.numpy()
            num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
        elif seq_lens_cpu is not None:
            (
                seq_lens_cpu,
                seq_lens_np,
                num_blocks_np,
            ) = _flashinfer_seq_lens_and_blocks_for_paged_kv(
                seq_lens_cpu,
                qo_indptr_cpu,
                num_decodes,
                num_prefills,
                page_size,
                use_dcp=self.use_dcp,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                dcp_kv_cache_interleave_size=self.dcp_kv_cache_interleave_size,
                include_prefill_query_from=canonical_paged_start_req,
            )
        else:
            seq_lens_np = None
            num_blocks_np = None

        # Adjust num_block_np for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size
            num_blocks_np -= num_common_kv_blocks

        # Compute paged_kv_indices if necessary
        # paged_kv_indices is only needed for FlashInfer native paths;
        # XQA/trtllm-gen paths use block_tables directly on GPU.
        needs_native_paged_prefill = num_prefills > 0 and not prefill_use_trtllm
        needs_native_paged_decode = (
            num_decodes > 0 and not decode_with_flashinfer_trtllm_api
        )
        needs_paged_kv_indices = (
            use_cascade or needs_native_paged_prefill or needs_native_paged_decode
        )
        if needs_paged_kv_indices:
            assert num_blocks_np is not None
            assert seq_lens_np is not None
            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )
        else:
            paged_kv_indices = None

        # Early-out for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            # Grab the blocks of the shared prefix from the first request.
            num_common_kv_blocks = common_prefix_len // page_size

            # Create CPU versions directly for cascade (no GPU versions needed)
            shared_qo_indptr_cpu = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indptr_cpu = torch.tensor(
                [0, num_common_kv_blocks], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indices_cpu = block_table_tensor[0, :num_common_kv_blocks]
            shared_kv_last_page_len_cpu = torch.tensor(
                [page_size], dtype=torch.int32, device="cpu"
            )

            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]
            num_blocks_np -= num_common_kv_blocks

            assert paged_kv_indices is not None
            paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: 1 + num_reqs]
            paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]

            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            # Cascade attention must use the same q dtype for prefill and decode
            # because it does not support FP8 kv-cache or FP8 query yet.
            assert self.q_data_type_prefill == self.q_data_type_decode
            attn_metadata.cascade_wrapper.plan(
                qo_indptr_arr=[shared_qo_indptr_cpu, qo_indptr_cpu],
                paged_kv_indptr_arr=[shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],
                paged_kv_indices_arr=[shared_kv_page_indices_cpu, paged_kv_indices],
                paged_kv_last_page_len=[
                    shared_kv_last_page_len_cpu,
                    paged_kv_last_page_len_cpu,
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_type_prefill,
                kv_data_type=self.kv_cache_dtype,
            )
            return attn_metadata

        # Step 3: Handle prefill and decode pathways case by case
        ## PREFILL PATHWAY
        if num_prefills > 0:
            # Slices for shared prefill metadata
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1

            if prefill_use_trtllm:
                # TRTLLM prefill has no cross-rank combine for DCP-sharded KV;
                # use_trtllm_attention never selects it when DCP is enabled.
                assert not self.use_dcp
                # Create GPU versions
                qo_indptr_prefill_gpu = (
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]
                )
                # Compute cum_seq_lens_kv on GPU to avoid CPU sync.
                # This is the cumulative sum of the number of KV cache
                # blocks per prefill request.
                prefill_seq_lens = seq_lens[prefill_start:]
                num_blocks_per_req = (prefill_seq_lens + page_size - 1) // page_size
                paged_kv_indptr_prefill_gpu = self.paged_kv_indptr.gpu[
                    prefill_start : num_reqs + 1
                ]
                # Assign to slice to avoid cpu sync.
                paged_kv_indptr_prefill_gpu[:1] = 0
                torch.cumsum(
                    num_blocks_per_req,
                    dim=0,
                    out=paged_kv_indptr_prefill_gpu[1:],
                )
                # Compute max_q_len for prefill requests
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                max_q_len_prefill = int(query_lens_prefill_cpu.max().item())
                attn_metadata.prefill = TRTLLMPrefill(
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=prefill_seq_lens,
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                    max_q_len=max_q_len_prefill,
                    max_seq_len=max_seq_len,
                )
            else:
                dcp_prefill_cudagraph_batch_size = (
                    self._dcp_prefill_cudagraph_batch_size(
                        common_attn_metadata,
                        num_decodes,
                        num_prefills,
                        for_cudagraph_capture=for_cudagraph_capture,
                    )
                )
                prefill_wrapper = (
                    (
                        self._get_dcp_batched_decode_wrapper()
                        if self._dcp_batched_decode_enabled
                        else (
                            self._get_dcp_sequential_decode_wrapper()
                            if self._dcp_sequential_decode_enabled
                            else self._get_dcp_pseudo_prefill_wrapper()
                        )
                    )
                    if use_dcp_pseudo_decode
                    else self._get_prefill_wrapper(
                        causal=attn_metadata.causal,
                        cudagraph_batch_size=dcp_prefill_cudagraph_batch_size,
                    )
                )
                if dcp_prefill_cudagraph_batch_size is not None:
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    self._verify_dcp_prefill_cudagraph_wrapper_identity(
                        dcp_prefill_cudagraph_batch_size,
                        prefill_wrapper,
                        for_cudagraph_capture=for_cudagraph_capture,
                    )
                # Slicing CPU buffers that are only needed for FI native prefills
                paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                    prefill_start:num_reqs
                ]
                assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
                paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                    prefill_start : num_reqs + 1
                ]
                assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1
                if self.use_dcp:
                    assert isinstance(
                        prefill_wrapper,
                        BatchDCPPrefillWrapper
                        | BatchDCPPseudoPrefillWrapper
                        | BatchDCPBatchedDecodeWrapper
                        | BatchDCPSequentialDecodeWrapper,
                    )
                    prefill_wrapper.plan(
                        qo_indptr_cpu=qo_indptr_prefill_cpu,
                        paged_kv_indptr_cpu=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len_cpu=paged_kv_last_page_len_prefill_cpu,
                        page_size=self.page_size,
                        num_qo_heads=self.num_qo_heads,
                        dcp_world_size=self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type_prefill,
                        kv_cache_dtype=self.kv_cache_dtype,
                        prefill_fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.prefill_disable_split_kv,
                        **(
                            {
                                "canonical_paged": (
                                    self._dcp_canonical_paged_prefill
                                ),
                                "absolute_segmented": (
                                    self._dcp_absolute_segment_prefill
                                ),
                                "canonical_paged_start_req": (
                                    canonical_paged_start_req - prefill_start
                                ),
                                "global_seq_lens_cpu": (
                                    common_attn_metadata.seq_lens_cpu[
                                        prefill_start:num_reqs
                                    ]
                                ),
                                "num_prompt_tokens_cpu": (
                                    common_attn_metadata.num_prompt_tokens_cpu[
                                        prefill_start:num_reqs
                                    ]
                                    if common_attn_metadata.num_prompt_tokens_cpu
                                    is not None
                                    else None
                                ),
                                "dcp_rank": self.dcp_rank,
                                "dcp_kv_cache_interleave_size": (
                                    self.dcp_kv_cache_interleave_size
                                ),
                            }
                            if canonical_paged_prefill
                            and isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                            else {}
                        ),
                    )
                else:
                    assert isinstance(
                        prefill_wrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                    )
                    # NVFP4 trtllm kernel only supports FP8 output;
                    # use FP8 o_data_type so the wrapper matches the
                    # FP8 output buffer allocated in forward().
                    o_dtype = (
                        FP8_DTYPE if self.is_kvcache_nvfp4 else self.model_config.dtype
                    )
                    prefill_wrapper.plan(
                        qo_indptr=qo_indptr_prefill_cpu,
                        paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,
                        num_qo_heads=self.num_qo_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim_qk=self.head_dim,
                        page_size=self.page_size,
                        causal=attn_metadata.causal,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type_prefill,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=o_dtype,
                        fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.prefill_disable_split_kv,
                    )
                attn_metadata.prefill = FIPrefill(
                    wrapper=prefill_wrapper,
                    match_fp8_new_tokens=match_fp8_new_tokens,
                )

        ## DECODE PATHWAY
        if num_decodes > 0:
            if decode_with_flashinfer_trtllm_api:
                assert num_decode_tokens % num_decodes == 0, (
                    "XQA/trtllm-gen decode requires uniform query lengths per request. "
                    f"Got {num_decode_tokens=} and {num_decodes=}."
                )
                assert self.flashinfer_trtllm_api_decode_kernel is not None
                seq_lens_decode = seq_lens[:num_decodes]
                if self.use_dcp:
                    assert common_attn_metadata.dcp_local_seq_lens is not None
                    seq_lens_decode = common_attn_metadata.dcp_local_seq_lens[
                        :num_decodes
                    ]
                attn_metadata.decode = FlashInferTrtllmAPIDecode(
                    kernel=self.flashinfer_trtllm_api_decode_kernel,
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens_decode,
                    max_seq_len=max_seq_len,
                )
            else:
                assert seq_lens_cpu is not None
                (
                    decode_fixed_split_size,
                    decode_disable_split_kv,
                    force_non_graph_wrapper,
                ) = _resolve_decode_split_plan(
                    num_decode_tokens=num_decode_tokens,
                    num_decodes=num_decodes,
                    fixed_split_size=self.decode_fixed_split_size,
                    disable_split_kv=self.decode_disable_split_kv,
                    spec_target_only=self.decode_fixed_split_spec_target_only,
                    qlen1_fixed_split_size=self.qlen1_fixed_split_size,
                    qlen1_disable_split_kv=self.qlen1_disable_split_kv,
                )
                pure_decode = num_prefills == 0
                use_cudagraph = (
                    self.enable_cuda_graph
                    and pure_decode
                    and num_decode_tokens <= self._decode_cudagraph_max_bs
                )
                if force_non_graph_wrapper:
                    # FlashInfer explicitly does not guarantee fixed split-K
                    # compatibility with CUDA graphs because changing KV
                    # lengths changes the internal number of CTAs.  K>0 target
                    # verification already uses vLLM's PIECEWISE lane, so use
                    # the ordinary replannable wrapper for this attention op
                    # without touching qlen=1 FULL graphs.
                    use_cudagraph = False
                num_input_tokens = num_decode_tokens

                decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph
                )
                # Use the persistent buffer with padding length,
                # instead of the same address but chunked version
                # in atten_metadata when using cudagraph.
                # NVFP4 trtllm kernel only supports FP8 output;
                # use FP8 o_data_type so the wrapper matches the
                # FP8 output buffer allocated in forward().
                o_dtype = (
                    FP8_DTYPE if self.is_kvcache_nvfp4 else self.model_config.dtype
                )
                fast_plan_decode(
                    decode_wrapper,
                    indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],
                    indices=paged_kv_indices,
                    last_page_len_cpu=self.paged_kv_last_page_len.cpu[
                        :num_input_tokens
                    ],
                    num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    # Disable flashinfer's pos encoding and use vllm's rope.
                    pos_encoding_mode="NONE",
                    sm_scale=self.sm_scale,
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap,
                    q_data_type=self.q_data_type_decode,
                    kv_data_type=self.kv_cache_dtype,
                    o_data_type=o_dtype,
                    fixed_split_size=decode_fixed_split_size,
                    disable_split_kv=decode_disable_split_kv,
                )
                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        # TODO: Cascade attention doesn't work, disable it for now
        # return use_cascade_attention(*args, **kwargs)
        return False


class FlashInferImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.is_kvcache_nvfp4 = kv_cache_dtype == "nvfp4"
        self.fp4_data_dim = head_size // 2 if self.is_kvcache_nvfp4 else 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

        self.sinks: torch.Tensor | None = None
        # Keep the source so RL weight updates can refresh the runtime tensor.
        self._sinks_source = sinks
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks

        self.supports_xqa_or_trtllm_gen_decode = can_use_trtllm_attention(
            num_heads, num_kv_heads, is_prefill=False
        )
        vllm_config = get_current_vllm_config_or_none()
        # Query pre-quantization needs a single dtype for the whole query tensor.
        # SM90 XQA needs BF16/FP16-Q for decode and FP8 for prefill,
        # so only enable this for SM100 trtllm-gen where both use FP8-Q.
        self.supports_quant_query_input = (
            self.supports_xqa_or_trtllm_gen_decode
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and current_platform.is_device_capability_family(100)
            and vllm_config is not None
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        )
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        self.o_sf_scale: float | None = None

        # Pre-allocated FP8 output buffer for NVFP4 without fused output quant.
        if self.is_kvcache_nvfp4 and vllm_config is not None:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._nvfp4_fp8_out = torch.empty(
                (max_num_tokens, num_heads, head_size),
                dtype=FP8_DTYPE,
                device="cuda",
            )
        else:
            self._nvfp4_fp8_out = None

        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if dcp_a2a:
            self.dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self.dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        # XQA does not support FP8/NVFP4 output, so require trtllm-gen
        # (SM100+) here.  Without that we cannot fuse the output quant.
        return (
            self.supports_xqa_or_trtllm_gen_decode
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and current_platform.is_device_capability_family(100)
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )

    # FlashInfer requires attention sinks to be float32
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        source_sinks = self._sinks_source
        if source_sinks is None:
            return
        if source_sinks.dtype == torch.float32:
            self.sinks = source_sinks
        elif self.sinks is None or self.sinks.dtype != torch.float32:
            self.sinks = source_sinks.to(torch.float32)
        else:
            self.sinks.copy_(source_sinks)

    def get_xqa_bmm1_scale(self, layer: torch.nn.Module, q_data_type: torch.dtype):
        bmm1_scale = self.scale
        if is_quantized_kv_cache(self.kv_cache_dtype):
            q_scale = get_query_scale_for_flashinfer(
                layer._q_scale_float,
                q_data_type,
            )
            if q_scale is not None:
                bmm1_scale *= q_scale
            bmm1_scale *= layer._k_scale_float
        return bmm1_scale

    # SM90 may need FP8-Q for native prefill and BF16/FP16-Q for XQA decode,
    # so quantize only the slice whose target dtype differs.
    def maybe_quant_query(
        self,
        query: torch.Tensor,
        q_data_type: torch.dtype,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if query.dtype != q_data_type:
            assert query.dtype in [torch.float16, torch.bfloat16]
            assert q_data_type in [torch.float8_e4m3fn, torch.float8_e5m2]
            assert query.dim() == 3
            num_tokens = query.shape[0]
            num_heads = query.shape[1]
            head_size = query.shape[2]
            assert query.stride(2) == 1 and query.stride(1) == head_size
            query_quantized, _ = custom_ops.scaled_fp8_quant(
                query.view(num_tokens, num_heads * head_size), scale=scale
            )
            return query_quantized.view(num_tokens, num_heads, head_size)

        return query

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        dcp_output_pack: torch.Tensor | None = None,
        dcp_lse_pack: torch.Tensor | None = None,
        dcp_kv_history_pack: torch.Tensor | None = None,
        dcp_kv_history_meta: torch.Tensor | None = None,
        dcp_kv_page_indices_pack: torch.Tensor | None = None,
        dcp_observer_meta: torch.Tensor | None = None,
        dcp_current_kv_pack: torch.Tensor | None = None,
        dcp_scale_pack: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tensor with different possible shapes:
                - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
                - HND: [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._v_scale_float

        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)
        decode_kernel = (
            attn_metadata.decode.kernel
            if isinstance(attn_metadata.decode, FlashInferTrtllmAPIDecode)
            else None
        )
        decode_with_xqa = decode_kernel == FlashInferDecodeKernel.XQA
        decode_with_trtllm_gen = decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
        decode_with_flashinfer_trtllm_api = decode_with_xqa or decode_with_trtllm_gen

        # The attn+quant fusion happens when output_scale is provided.
        if output_scale is None:
            assert output_block_scale is None, (
                "output_block_scale is not supported when fusion has not happened"
            )
        else:
            assert attn_metadata.q_data_type_prefill == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened for prefill."
            )
            assert attn_metadata.q_data_type_decode == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened for decode."
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (
                attn_metadata.num_decodes == 0 or decode_with_trtllm_gen
            ), "Output quant fusion requires TRTLLM prefill/trtllm-gen decode"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, (
                    "output_block_scale should not be provided for fp8 output"
                )
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, (
                    "output_block_scale is required for nvfp4 output"
                )
            else:
                raise ValueError(f"Unsupported output dtype: {output.dtype}")

            # TRTLLM attn kernel requires to scale to pass as a host scalar,
            # store the o scale as a host scalar in warmup run with cuda graph
            # not enabled
            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # FlashInfer treats uint8 KV cache as NVFP4. vLLM stores FP8 KV cache
        # as uint8 bytes, so pass FP8 caches with their logical dtype.
        if not self.is_kvcache_nvfp4 and kv_cache.dtype == torch.uint8:
            fp8_view_dtype = None
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3", torch.float8_e4m3fn):
                fp8_view_dtype = torch.float8_e4m3fn
            elif self.kv_cache_dtype in ("fp8_e5m2", torch.float8_e5m2):
                fp8_view_dtype = torch.float8_e5m2
            if fp8_view_dtype is not None:
                kv_cache = kv_cache.view(fp8_view_dtype)

        # Inputs and outputs may be padded for CUDA graphs
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        if attn_metadata.use_cascade:
            # Cascade attention (rare case).
            assert attn_metadata.cascade_wrapper is not None
            stride_order = FlashInferBackend.get_kv_cache_stride_order()
            if self.is_kvcache_nvfp4:
                kv_cache_views = tuple(
                    cache.permute(*stride_order)
                    for cache in kv_cache.split(self.num_kv_heads, dim=1)
                )
            else:
                kv_perm = kv_cache.permute(*stride_order)
                kv_cache_views = kv_perm.split(self.head_size, dim=-1)
            kv_tuple = tuple(
                canonicalize_singleton_dim_strides(cache) for cache in kv_cache_views
            )
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_tuple))
            return output

        # When using spec decoding, num_decodes can be < num_decode_tokens
        # because some decode requests may have more than one query token.
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)  # HND and contiguous
        # Fix degenerate strides on any size-1 dimension (e.g. num_kv_heads=1
        # with TP=8).  PyTorch permits non-canonical strides on size-1 dims;
        # CUDA TMA requires ≥16-byte alignment on all non-outermost strides.
        # canonicalize_singleton_dim_strides patches metadata via as_strided —
        # zero-copy.  See vllm.utils.torch_utils.
        fixed = canonicalize_singleton_dim_strides(kv_cache_permute)
        if fixed is not kv_cache_permute:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashInfer): "
                "shape=%s, strides before=%s, strides after=%s",
                kv_cache_permute.shape,
                kv_cache_permute.stride(),
                fixed.stride(),
            )
        kv_cache_permute = fixed

        # Split K/V — zero-copy views. NVFP4 stores K/V as separate head
        # groups; other dtypes pack K/V in the content dim.
        hs = self.head_size
        nvfp4_kv_data = None
        nvfp4_kv_block_scales = None
        if self.is_kvcache_nvfp4:
            k_cache, v_cache = kv_cache.split(self.num_kv_heads, dim=1)
            kv_cache_tuple = (
                canonicalize_singleton_dim_strides(k_cache.permute(*stride_order)),
                canonicalize_singleton_dim_strides(v_cache.permute(*stride_order)),
            )
            k_data, k_sf = nvfp4_split_data_scale(kv_cache_tuple[0])
            v_data, v_sf = nvfp4_split_data_scale(kv_cache_tuple[1])
            nvfp4_kv_data = (k_data, v_data)
            nvfp4_kv_block_scales = (k_sf, v_sf)
        else:
            kv_cache_tuple = kv_cache_permute.split(hs, dim=-1)

        use_dcp = self.dcp_world_size > 1

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back.
        if num_prefill_tokens > 0:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            # Convert query to the expected dtype for prefill if needed.
            prefill_query = self.maybe_quant_query(
                prefill_query,
                attn_metadata.q_data_type_prefill,
                layer._q_scale,
            )

            if not prefill_use_trtllm:
                assert isinstance(attn_metadata.prefill, FIPrefill)
                prefill_wrapper = attn_metadata.prefill.wrapper
                assert prefill_wrapper is not None
                if use_dcp:
                    if isinstance(
                        prefill_wrapper,
                        BatchDCPBatchedDecodeWrapper | BatchDCPSequentialDecodeWrapper,
                    ):
                        prefill_wrapper.run(
                            layer,
                            prefill_query,
                            kv_cache_tuple,
                            out=output[num_decode_tokens:],
                        )
                    elif isinstance(prefill_wrapper, BatchDCPPseudoPrefillWrapper):
                        prefill_wrapper.assert_plan_contract(
                            window_left=self.window_left,
                            logits_soft_cap=self.logits_soft_cap or 0.0,
                            sm_scale=self.scale,
                        )
                        prefill_wrapper.run(
                            layer,
                            prefill_query,
                            kv_cache_tuple,
                            out=output[num_decode_tokens:],
                        )
                    else:
                        assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                        prefill_wrapper.assert_plan_contract(
                            window_left=self.window_left,
                            logits_soft_cap=self.logits_soft_cap or 0.0,
                            sm_scale=self.scale,
                        )

                        prefill_wrapper.run(
                            layer,
                            prefill_query,
                            kv_cache_tuple,
                            key[num_decode_tokens:],
                            value[num_decode_tokens:],
                            out=output[num_decode_tokens:],
                            dcp_output_pack=dcp_output_pack,
                            dcp_lse_pack=dcp_lse_pack,
                            dcp_kv_history_pack=dcp_kv_history_pack,
                            dcp_kv_history_meta=dcp_kv_history_meta,
                            dcp_kv_page_indices_pack=dcp_kv_page_indices_pack,
                            dcp_observer_meta=dcp_observer_meta,
                            dcp_current_kv_pack=dcp_current_kv_pack,
                            dcp_scale_pack=dcp_scale_pack,
                            match_fp8_new_tokens=(
                                attn_metadata.prefill.match_fp8_new_tokens
                            ),
                        )
                else:
                    assert isinstance(
                        prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper
                    )
                    assert prefill_wrapper._window_left == self.window_left
                    assert prefill_wrapper._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._sm_scale == self.scale
                    assert prefill_wrapper._causal == attn_metadata.causal

                    if self.is_kvcache_nvfp4:
                        kv_cache_for_fi = nvfp4_kv_data
                    else:
                        kv_cache_for_fi = kv_cache_tuple
                    kv_cache_sf = (
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    )

                    # NVFP4 trtllm kernel only supports FP8 output.
                    # Use a pre-allocated FP8 buffer and dequantize
                    # afterwards.
                    needs_fp8_out_prefill = (
                        self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                    )
                    if needs_fp8_out_prefill:
                        out_prefill = self._nvfp4_fp8_out[:num_prefill_tokens]
                    else:
                        out_prefill = output[num_decode_tokens:]

                    prefill_wrapper.run(
                        prefill_query,
                        kv_cache_for_fi,
                        q_scale=get_query_scale_for_flashinfer(
                            layer._q_scale_float,
                            prefill_query.dtype,
                        ),
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_prefill,
                        kv_cache_sf=kv_cache_sf,
                    )

                    if needs_fp8_out_prefill:
                        output[
                            num_decode_tokens : num_decode_tokens + num_prefill_tokens
                        ].copy_(out_prefill.to(output.dtype))
            else:
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
                # prefill_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                prefill_query = prefill_query.contiguous()
                prefill_query = canonicalize_singleton_dim_strides(prefill_query)
                workspace_buffer = _get_trtllm_workspace_buffer()
                block_tables_prefill = attn_metadata.prefill.block_tables
                seq_lens_prefill = attn_metadata.prefill.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(prefill_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_prefill)
                assert is_strictly_contiguous(seq_lens_prefill)

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[num_decode_tokens:],
                        scale=output_block_scale,
                        scale_start_index=num_decode_tokens,
                        original_shape=prefill_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[num_decode_tokens:]

                # NVFP4 trtllm kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_prefill_tokens]

                prefill_kv_block_scales = None
                if self.is_kvcache_nvfp4:
                    # NVFP4 trtllm-gen kernel requires FP8 query.
                    assert attn_metadata.q_data_type_prefill == FP8_DTYPE, (
                        "NVFP4 KV cache requires FP8 quantized queries for "
                        "trtllm-gen prefill. Set "
                        "disable_flashinfer_q_quantization=False."
                    )
                    mock_kv_cache = nvfp4_kv_data
                    mock_block_table = block_tables_prefill
                    prefill_kv_block_scales = nvfp4_kv_block_scales
                elif (
                    attn_metadata.q_data_type_prefill != FP8_DTYPE
                    and self.kv_cache_dtype.startswith("fp8")
                ):
                    # TRTLLM prefill attention does not support BF16 Q
                    # and fp8 kv cache. So to enable prefill attention
                    # with fp8 kv cache, we can construct a mock block
                    # and mock kv cache with BF16 KV involved in the prefill.
                    kv_cache_permute = canonicalize_singleton_dim_strides(
                        kv_cache_permute
                    )
                    kv_strides = kv_cache_permute.stride()
                    assert (
                        kv_strides[-1] == 1
                        and kv_strides[-2] == kv_cache_permute.shape[-1]
                    ), (
                        "KV cache inner dims (block_size, head_size) must be "
                        f"contiguous, got strides {kv_strides}"
                    )
                    # fp8 uses (B, H, N, 2*hs); reshape to (B, 2, H, N, hs)
                    # for the dequant kernel — zero-copy view. The dequant
                    # kernel handles the interleaved K/V block stride.
                    B_kv, H_kv, N_kv = kv_cache_permute.shape[:3]
                    kv_cache_5d = kv_cache_permute.view(B_kv, H_kv, N_kv, 2, hs)
                    kv_cache_5d = kv_cache_5d.permute(0, 3, 1, 2, 4)
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                        kv_cache_5d,
                        block_tables_prefill,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.q_data_type_prefill,
                    )
                else:
                    mock_kv_cache = kv_cache_tuple
                    mock_block_table = block_tables_prefill

                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_cache_sf=prefill_kv_block_scales,
                )

                if needs_fp8_out:
                    output[
                        num_decode_tokens : num_decode_tokens + num_prefill_tokens
                    ].copy_(out[:num_prefill_tokens].to(output.dtype))

        if num_decode_tokens > 0:
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens
            if num_decode_tokens % attn_metadata.num_decodes != 0:
                raise RuntimeError(
                    "FlashInfer decode tokens must divide evenly across requests: "
                    f"{num_decode_tokens=} {attn_metadata.num_decodes=}"
                )
            decode_query_len_per_req = num_decode_tokens // attn_metadata.num_decodes

            # Convert query to the expected dtype for decode if needed.
            decode_query = self.maybe_quant_query(
                decode_query,
                attn_metadata.q_data_type_decode,
                layer._q_scale,
            )

            if not decode_with_flashinfer_trtllm_api:
                assert isinstance(attn_metadata.decode, FIDecode)
                decode_wrapper = attn_metadata.decode.wrapper
                assert decode_wrapper is not None
                assert decode_wrapper._window_left == self.window_left
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
                assert decode_wrapper._sm_scale == self.scale

                if self.is_kvcache_nvfp4:
                    kv_cache_for_fi = nvfp4_kv_data
                else:
                    kv_cache_for_fi = kv_cache_tuple
                kv_cache_sf = nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None

                # NVFP4 kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out_decode = self._nvfp4_fp8_out[:num_decode_tokens]
                else:
                    out_decode = output[:num_decode_tokens]

                if use_dcp:
                    _write_ag2_dcp_kv_history_pack(
                        layer,
                        kv_cache_tuple,
                        attn_metadata.decode,
                        dcp_kv_history_pack,
                        dcp_kv_history_meta,
                        dcp_kv_page_indices_pack,
                        dcp_observer_meta,
                        request_count=attn_metadata.num_decodes,
                        request_row_stride=decode_query_len_per_req,
                    )
                    decode_request_rows = torch.arange(
                        0,
                        num_decode_tokens,
                        decode_query_len_per_req,
                        dtype=torch.long,
                        device=key.device,
                    )
                    _write_ag2_dcp_current_kv_and_scales(
                        layer,
                        key,
                        value,
                        request_indices=decode_request_rows,
                        request_count=attn_metadata.num_decodes,
                        current_kv_pack=dcp_current_kv_pack,
                        scale_pack=dcp_scale_pack,
                        observer_meta=dcp_observer_meta,
                        match_fp8_new_tokens=False,
                        sm_scale=self.scale,
                    )
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    output_tmp = torch.empty_like(decode_query)
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_for_fi,
                        q_scale=get_query_scale_for_flashinfer(
                            layer._q_scale_float,
                            decode_query.dtype,
                        ),
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output_tmp,
                        lse=lse,
                        return_lse=True,
                        kv_cache_sf=kv_cache_sf,
                    )
                    _copy_ag2_dcp_trace(layer, "decode_local_output", output_tmp)
                    _copy_ag2_dcp_trace(layer, "decode_local_lse", lse)
                    _write_ag2_dcp_aux_pack(
                        layer,
                        "local_output",
                        output_tmp,
                        dcp_output_pack,
                        dcp_lse_pack,
                        request_tail_count=attn_metadata.num_decodes,
                        request_row_stride=decode_query_len_per_req,
                        dcp_observer_meta=dcp_observer_meta,
                    )
                    _write_ag2_dcp_aux_pack(
                        layer,
                        "local_lse",
                        lse,
                        dcp_output_pack,
                        dcp_lse_pack,
                        request_tail_count=attn_metadata.num_decodes,
                        request_row_stride=decode_query_len_per_req,
                        dcp_observer_meta=dcp_observer_meta,
                    )
                    if getattr(layer, "_ag2_dcp_trace_enabled", False):
                        combined_output, combined_lse = self.dcp_combine(
                            output_tmp,
                            lse,
                            get_dcp_group(),
                            return_lse=True,
                        )
                        _copy_ag2_dcp_trace(
                            layer, "decode_combined_output", combined_output
                        )
                        _copy_ag2_dcp_trace(layer, "decode_combined_lse", combined_lse)
                        _write_ag2_dcp_aux_pack(
                            layer,
                            "combined_output",
                            combined_output,
                            dcp_output_pack,
                            dcp_lse_pack,
                            request_tail_count=attn_metadata.num_decodes,
                            request_row_stride=decode_query_len_per_req,
                            dcp_observer_meta=dcp_observer_meta,
                        )
                        _write_ag2_dcp_aux_pack(
                            layer,
                            "combined_lse",
                            combined_lse,
                            dcp_output_pack,
                            dcp_lse_pack,
                            request_tail_count=attn_metadata.num_decodes,
                            request_row_stride=decode_query_len_per_req,
                            dcp_observer_meta=dcp_observer_meta,
                        )
                        output[:num_decode_tokens] = combined_output
                    else:
                        if dcp_output_pack is not None:
                            combined_output, combined_lse = self.dcp_combine(
                                output_tmp,
                                lse,
                                get_dcp_group(),
                                return_lse=True,
                            )
                            _write_ag2_dcp_aux_pack(
                                layer,
                                "combined_output",
                                combined_output,
                                dcp_output_pack,
                                dcp_lse_pack,
                                request_tail_count=attn_metadata.num_decodes,
                                request_row_stride=decode_query_len_per_req,
                                dcp_observer_meta=dcp_observer_meta,
                            )
                            _write_ag2_dcp_aux_pack(
                                layer,
                                "combined_lse",
                                combined_lse,
                                dcp_output_pack,
                                dcp_lse_pack,
                                request_tail_count=attn_metadata.num_decodes,
                                request_row_stride=decode_query_len_per_req,
                                dcp_observer_meta=dcp_observer_meta,
                            )
                            output[:num_decode_tokens] = combined_output
                        else:
                            output[:num_decode_tokens] = self.dcp_combine(
                                output_tmp,
                                lse,
                                get_dcp_group(),
                            )
                else:
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_for_fi,
                        q_scale=get_query_scale_for_flashinfer(
                            layer._q_scale_float,
                            decode_query.dtype,
                        ),
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_decode,
                        kv_cache_sf=kv_cache_sf,
                    )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out_decode.to(output.dtype))
            else:
                assert isinstance(attn_metadata.decode, FlashInferTrtllmAPIDecode)
                # decode_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                decode_query = decode_query.contiguous()
                decode_query = canonicalize_singleton_dim_strides(decode_query)
                workspace_buffer = _get_trtllm_workspace_buffer()
                block_tables_decode = attn_metadata.decode.block_tables
                seq_lens_decode = attn_metadata.decode.seq_lens

                # trtllm-gen needs HND layout on SM100. XQA is selected
                # separately on SM90 and does not use this SM100 layout gate.
                if decode_with_trtllm_gen:
                    assert get_kv_cache_layout() == "HND"
                else:
                    assert decode_with_xqa
                assert is_strictly_contiguous(decode_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_decode)
                assert is_strictly_contiguous(seq_lens_decode)
                kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)
                kv_strides = kv_cache_permute.stride()
                assert (
                    kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1]
                ), (
                    "KV cache inner dims (block_size, head_size) must be "
                    f"contiguous, got strides {kv_strides}"
                )

                if use_dcp:
                    assert decode_with_trtllm_gen
                    if output.dtype == FP4_DTYPE:
                        raise NotImplementedError(
                            "DCP decode with FlashInfer trtllm-gen does not support "
                            "FP4 attention output yet."
                        )
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    decode_query = canonicalize_singleton_dim_strides(decode_query)

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[:num_decode_tokens],
                        scale=output_block_scale,
                        scale_start_index=0,
                        original_shape=decode_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[:num_decode_tokens]

                # NVFP4 trtllm kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_decode_tokens]

                if num_decode_tokens % attn_metadata.num_decodes != 0:
                    # This gets triggered when the dummy_run forces
                    # attention to be initialized with q_len = 0
                    q_len_per_req = 1
                else:
                    q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

                if decode_with_xqa and q_len_per_req > 1:
                    raise NotImplementedError(
                        "FlashInfer XQA speculative decode is not wired in vLLM yet."
                    )

                # XQA decode can use model-dtype Q with FP8 KV, so only include
                # q_scale when the decode query is actually FP8.
                bmm1_scale = (
                    self.get_xqa_bmm1_scale(layer, attn_metadata.q_data_type_decode)
                    if decode_with_xqa
                    else self.bmm1_scale
                )

                lse = None
                if use_dcp:
                    out = torch.empty(
                        decode_query.shape,
                        dtype=output.dtype,
                        device=decode_query.device,
                    )
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )

                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=(
                        nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_tuple
                    ),
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables_decode,
                    seq_lens=seq_lens_decode,
                    max_seq_len=attn_metadata.decode.max_seq_len,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_layout=get_kv_cache_layout(),
                    backend=attn_metadata.decode.kernel.value,
                    q_len_per_req=q_len_per_req,
                    kv_cache_sf=(
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    ),
                    lse=lse,
                    return_lse=self.need_to_return_lse_for_decode,
                )

                if use_dcp:
                    assert isinstance(out, torch.Tensor)
                    assert lse is not None
                    _copy_ag2_dcp_trace(layer, "decode_local_output", out)
                    _copy_ag2_dcp_trace(layer, "decode_local_lse", lse)
                    _write_ag2_dcp_aux_pack(
                        layer,
                        "local_output",
                        out,
                        dcp_output_pack,
                        dcp_lse_pack,
                        request_tail_count=attn_metadata.num_decodes,
                        request_row_stride=decode_query_len_per_req,
                    )
                    _write_ag2_dcp_aux_pack(
                        layer,
                        "local_lse",
                        lse,
                        dcp_output_pack,
                        dcp_lse_pack,
                        request_tail_count=attn_metadata.num_decodes,
                        request_row_stride=decode_query_len_per_req,
                    )
                    if getattr(layer, "_ag2_dcp_trace_enabled", False):
                        combined_output, combined_lse = self.dcp_combine(
                            out,
                            lse,
                            get_dcp_group(),
                            return_lse=True,
                        )
                        _copy_ag2_dcp_trace(
                            layer, "decode_combined_output", combined_output
                        )
                        _copy_ag2_dcp_trace(layer, "decode_combined_lse", combined_lse)
                        _write_ag2_dcp_aux_pack(
                            layer,
                            "combined_output",
                            combined_output,
                            dcp_output_pack,
                            dcp_lse_pack,
                            request_tail_count=attn_metadata.num_decodes,
                            request_row_stride=decode_query_len_per_req,
                        )
                        _write_ag2_dcp_aux_pack(
                            layer,
                            "combined_lse",
                            combined_lse,
                            dcp_output_pack,
                            dcp_lse_pack,
                            request_tail_count=attn_metadata.num_decodes,
                            request_row_stride=decode_query_len_per_req,
                        )
                        output[:num_decode_tokens] = combined_output
                    else:
                        if dcp_output_pack is not None:
                            combined_output, combined_lse = self.dcp_combine(
                                out,
                                lse,
                                get_dcp_group(),
                                return_lse=True,
                            )
                            _write_ag2_dcp_aux_pack(
                                layer,
                                "combined_output",
                                combined_output,
                                dcp_output_pack,
                                dcp_lse_pack,
                                request_tail_count=attn_metadata.num_decodes,
                                request_row_stride=decode_query_len_per_req,
                            )
                            _write_ag2_dcp_aux_pack(
                                layer,
                                "combined_lse",
                                combined_lse,
                                dcp_output_pack,
                                dcp_lse_pack,
                                request_tail_count=attn_metadata.num_decodes,
                                request_row_stride=decode_query_len_per_req,
                            )
                            output[:num_decode_tokens] = combined_output
                        else:
                            output[:num_decode_tokens] = self.dcp_combine(
                                out,
                                lse,
                                get_dcp_group(),
                            )
                elif needs_fp8_out:
                    output[:num_decode_tokens].copy_(out.to(output.dtype))
        return output_padded

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            if self.is_kvcache_nvfp4:
                # (B, 2*H, N, full_dim) -> ((B, N, H, full_dim),
                #                            (B, N, H, full_dim));
                # K heads first, then V heads.
                k_cache, v_cache = kv_cache.transpose(1, 2).split(
                    self.num_kv_heads, dim=-2
                )
            else:
                # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
                k_cache, v_cache = kv_cache.transpose(1, 2).split(
                    self.head_size, dim=-1
                )
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )


def fast_plan_decode(
    self,  # decode wrapper
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: str | torch.dtype | None = "float16",
    kv_data_type: str | torch.dtype | None = None,
    o_data_type: str | torch.dtype | None = None,
    data_type: str | torch.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for
    cudagraph capture/replay, while the no cudagraph version turns back
    to the original plan.
    using original plan after passing host-side buffers:
    - only host-to-device copy of indptr and last_page_len buffers
    Modifications for cudagraph:
    - only host-to-device copy of indptr and last_page_len buffers.
    - avoid device-to-device copy of indices buffer.

    Part of the code get inspiration from the original plan from FlashInfer repo
    and the implementation of fast_decode_plan for FlashInfer in SGlang repo.
    """
    # Warm up with the original plan if it is first call, and always run the
    # original plan if we run for dynamic shape. For fixed shape (cudagraph),
    # this warm up is to generate the _cached_module for the decode wrapper.
    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):
        self.plan(
            indptr=indptr_cpu,
            indices=indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=None,
            seq_lens=None,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self.vllm_first_call = False
        return

    assert self.is_cuda_graph_enabled, "Should be cudagraph only here"

    fast_decode_plan(
        self,
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
