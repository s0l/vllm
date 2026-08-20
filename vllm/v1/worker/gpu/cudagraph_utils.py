# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import os
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, NamedTuple, Protocol

import torch
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def copy_aux_hidden_state(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    num_tokens: int,
    token_major: bool,
) -> None:
    """Copy either a token-major or a fixed-shape diagnostic graph output."""
    if token_major:
        destination[:num_tokens].copy_(source)
    else:
        if destination.shape != source.shape:
            raise RuntimeError(
                "Fixed-shape auxiliary CUDA Graph output changed shape: "
                f"{tuple(destination.shape)} != {tuple(source.shape)}"
            )
        destination.copy_(source)


def view_aux_hidden_state(
    tensor: torch.Tensor,
    *,
    num_tokens: int,
    token_major: bool,
) -> torch.Tensor:
    """Return the active token slice or the complete fixed-shape output."""
    return tensor[:num_tokens] if token_major else tensor


class AttentionState(NamedTuple):
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Describes the shape of the batch and CG mode to run; this is used to make shape
    matches between the capture and runtime."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None
    num_active_loras: int = 0


class DynamicGraphResidency(str, Enum):
    WARM = "warm"
    QUEUED = "queued"
    HOT = "hot"
    COOLDOWN = "cooldown"


@dataclass
class DynamicGraphEntry:
    descriptor: BatchExecutionDescriptor
    pinned: bool = False
    state: DynamicGraphResidency = DynamicGraphResidency.WARM
    hits: int = 0
    last_used_epoch: int = 0
    cooldown_until_epoch: int = 0
    charged_bytes: int = 0
    graph_segments: int = 0


class CreateForwardFn(Protocol):
    """Factory that prepares inputs (OUTSIDE the graph) and returns a
    forward_fn. Called with warmup=True for the warmup pass and warmup=False
    for the captured pass."""

    def __call__(
        self,
        desc: BatchExecutionDescriptor,
        warmup: bool,
    ) -> Callable[[CUDAGraphMode], None]: ...


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    num_active_loras: int,
) -> bool:
    # desc.uniform_token_count=None (PIECEWISE) can handle any uniform_token_count
    # desc.num_reqs=None means no request padding needed (PIECEWISE)
    exact_multitoken_full_requests = not (
        desc.cg_mode == CUDAGraphMode.FULL
        and desc.uniform_token_count is not None
        and desc.uniform_token_count > 1
    ) or (desc.num_reqs == num_reqs and desc.num_tokens == num_tokens)
    return (
        (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
        and desc.num_active_loras == num_active_loras
        and exact_multitoken_full_requests
    )


def get_uniform_token_count(
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
) -> int | None:
    """
    Return the uniform token count if batch is uniform, else None.
    A batch is uniform if all requests have the same number of tokens.
    """
    if (max_query_len == num_tokens // num_reqs) and (
        num_tokens == max_query_len * num_reqs
    ):
        return max_query_len
    return None


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        full_decode_query_lens: set[int] | None = None,
        full_decode_cap_query_lens: set[int] | None = None,
        expand_dynamic_decode_query_lens: bool = True,
        max_uniform_decode_reqs: int | None = None,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_uniform_decode_reqs = (
            self.max_num_reqs
            if max_uniform_decode_reqs is None
            else min(max_uniform_decode_reqs, self.max_num_reqs)
        )
        if self.max_uniform_decode_reqs < 1:
            raise ValueError("max_uniform_decode_reqs must be positive")
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len
        self.full_decode_query_lens = full_decode_query_lens
        self.full_decode_cap_query_lens = full_decode_cap_query_lens
        self.expand_dynamic_decode_query_lens = expand_dynamic_decode_query_lens

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank
        self.lora_capture_cases = lora_capture_cases or [0]
        additional_config = vllm_config.additional_config
        self._p3_crash_diagnostic = bool(
            isinstance(additional_config, dict)
            and additional_config.get("p3_crash_diagnostic", False)
        )
        self._p3_graph_debug = bool(
            isinstance(additional_config, dict)
            and additional_config.get("p3_graph_debug", False)
        )
        self._p3_global_sync = bool(
            isinstance(additional_config, dict)
            and additional_config.get("p3_global_sync", False)
        )
        p3_node_cutoff = (
            additional_config.get("p3_node_cutoff")
            if isinstance(additional_config, dict)
            else None
        )
        if p3_node_cutoff is not None:
            if not isinstance(p3_node_cutoff, int) or p3_node_cutoff < 0:
                raise ValueError("p3_node_cutoff must be a non-negative integer")
        p3_node_prefix_counts = (
            additional_config.get("p3_node_prefix_counts")
            if isinstance(additional_config, dict)
            else None
        )
        if p3_node_prefix_counts is None:
            p3_node_prefix_counts = []
        if (
            not isinstance(p3_node_prefix_counts, list)
            or any(
                not isinstance(count, int) or count <= 0
                for count in p3_node_prefix_counts
            )
            or p3_node_prefix_counts != sorted(set(p3_node_prefix_counts))
        ):
            raise ValueError(
                "p3_node_prefix_counts must be a strictly increasing list "
                "of positive integers"
            )
        if p3_node_cutoff is not None and p3_node_prefix_counts:
            raise ValueError(
                "p3_node_cutoff and p3_node_prefix_counts are mutually exclusive"
            )
        self._p3_node_cutoff: int | None = p3_node_cutoff
        self._p3_node_prefix_counts = tuple(p3_node_prefix_counts)
        self._p3_prefix_diagnostic_descs: set[BatchExecutionDescriptor] = set()
        self._p3_prefix_diagnostic_state: dict[
            BatchExecutionDescriptor,
            tuple[Any, list[tuple[int, Any]], int],
        ] = {}
        dynamic_sizes = (
            additional_config.get("dynamic_cudagraph_capture_sizes", [])
            if isinstance(additional_config, dict)
            else []
        )
        if not isinstance(dynamic_sizes, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in dynamic_sizes
        ):
            raise ValueError(
                "dynamic_cudagraph_capture_sizes must be a list of positive integers"
            )
        startup_sizes = set(self.compilation_config.cudagraph_capture_sizes or [])
        self.dynamic_capture_sizes = tuple(
            sorted(set(dynamic_sizes).difference(startup_sizes))
        )
        dynamic_budget_mb = (
            additional_config.get("dynamic_cudagraph_budget_mb", 0)
            if isinstance(additional_config, dict)
            else 0
        )
        if isinstance(dynamic_budget_mb, bool) or not isinstance(
            dynamic_budget_mb, int
        ):
            raise ValueError("dynamic_cudagraph_budget_mb must be an integer")
        if bool(self.dynamic_capture_sizes) != (dynamic_budget_mb > 0):
            raise ValueError(
                "dynamic CUDA graphs require both capture sizes and a positive "
                "dynamic_cudagraph_budget_mb"
            )
        self.dynamic_graph_budget_bytes = dynamic_budget_mb * 1024 * 1024
        max_entry_mb = (
            additional_config.get(
                "dynamic_cudagraph_max_entry_mb", dynamic_budget_mb
            )
            if isinstance(additional_config, dict)
            else 0
        )
        guard_mb = (
            additional_config.get("dynamic_cudagraph_guard_mb", 150)
            if isinstance(additional_config, dict)
            else 150
        )
        self.dynamic_graph_min_hits = (
            additional_config.get("dynamic_cudagraph_min_hits", 2)
            if isinstance(additional_config, dict)
            else 2
        )
        self.dynamic_graph_cooldown_steps = (
            additional_config.get("dynamic_cudagraph_cooldown_steps", 128)
            if isinstance(additional_config, dict)
            else 128
        )
        for name, value, minimum in (
            ("dynamic_cudagraph_max_entry_mb", max_entry_mb, 1),
            ("dynamic_cudagraph_guard_mb", guard_mb, 0),
            ("dynamic_cudagraph_min_hits", self.dynamic_graph_min_hits, 1),
            (
                "dynamic_cudagraph_cooldown_steps",
                self.dynamic_graph_cooldown_steps,
                1,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or (self.dynamic_capture_sizes and value < minimum)
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.dynamic_graph_max_entry_bytes = max_entry_mb * 1024 * 1024
        self.dynamic_graph_guard_bytes = guard_mb * 1024 * 1024
        pinned_sizes = (
            additional_config.get("dynamic_cudagraph_pinned_sizes", [])
            if isinstance(additional_config, dict)
            else []
        )
        if not isinstance(pinned_sizes, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in pinned_sizes
        ):
            raise ValueError(
                "dynamic_cudagraph_pinned_sizes must be a list of positive integers"
            )
        if not set(pinned_sizes).issubset(self.dynamic_capture_sizes):
            raise ValueError(
                "dynamic_cudagraph_pinned_sizes must be dynamic capture sizes"
            )
        self.dynamic_graph_pinned_sizes = frozenset(pinned_sizes)
        self._dynamic_graph_entries: dict[
            BatchExecutionDescriptor, DynamicGraphEntry
        ] = {}
        self._dynamic_pending: BatchExecutionDescriptor | None = None
        self._dynamic_epoch = 0
        self._dynamic_resident_bytes = 0
        # Precompute actual num_active_loras -> captured case mapping so that
        # dispatch() is a plain dict lookup instead of a per-call bisect.
        self._lora_dispatch_map, self._max_lora_case = self._build_lora_dispatch_map()

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None

        self._graphs_captured = False

        self._candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}
        self.max_capture_tokens = 0

        # Breakable CUDA graph (PW CUDA graph without torch.compile)
        self.use_breakable_cg = (
            is_breakable_cudagraph_enabled()
            and self.cudagraph_mode.has_piecewise_cudagraphs()
        )
        if self.dynamic_capture_sizes and self.use_breakable_cg:
            raise ValueError(
                "dynamic CUDA graph residency does not yet support breakable graphs"
            )
        self.breakable_cg_runner: BreakableCUDAGraphWrapper | None = None

        self._init_candidates()

    def _build_lora_dispatch_map(self) -> tuple[dict[int, int], int]:
        """Precompute actual num_active_loras -> effective captured case.

        Mirrors the num_tokens candidate expansion in ``_init_candidates``:
        every possible active-LoRA count is mapped ahead of time to the
        smallest captured case that can serve it, so ``dispatch`` is a plain
        dict lookup instead of a per-call bisect.
        """
        captured_with_lora = sorted(c for c in self.lora_capture_cases if c > 0)
        if not captured_with_lora:
            return {}, 0
        dispatch_map: dict[int, int] = {}
        case_idx = 0
        for n in range(1, captured_with_lora[-1] + 1):
            while captured_with_lora[case_idx] < n:
                case_idx += 1
            dispatch_map[n] = captured_with_lora[case_idx]
        return dispatch_map, captured_with_lora[-1]

    def _resolve_effective_loras(self, num_active_loras: int) -> int:
        """Map an actual active-LoRA count to its captured graph case."""
        if num_active_loras <= 0 or not self._lora_dispatch_map:
            return num_active_loras
        # Counts above the largest captured case clamp to it.
        return self._lora_dispatch_map.get(num_active_loras, self._max_lora_case)

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
        registered_mixed_sizes = sorted(
            set(capture_sizes).union(self.dynamic_capture_sizes)
        )
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()
        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size

        descs_by_token_lora: dict[tuple[int, int], list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )
        descs_by_mode: defaultdict[CUDAGraphMode, list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )

        # When using Dynamic SD, num_speculative_tokens is the max number of
        # draft tokens. The scheduler might use a smaller number so we need
        # to capture graphs for all possible values during decode.
        speculative_config = self.vllm_config.speculative_config
        if (
            self.expand_dynamic_decode_query_lens
            and
            speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            num_spec_per_batch_size = (
                speculative_config.num_speculative_tokens_per_batch_size
            )
            # uses_dynamic_speculative_decoding() guarantees this is set.
            assert num_spec_per_batch_size is not None
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens
            # _per_step. Recover num_new_sampled_tokens_per_step
            # from the values the manager already has.
            num_new_sampled_tokens_per_step = (
                self.decode_query_len - self.vllm_config.num_speculative_tokens
            )
            # Each entry is (range_start, range_end, num_speculative_tokens).
            decode_query_lens = [
                x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
            ]
        else:
            decode_query_lens = [self.decode_query_len]
        if any(query_len <= 0 for query_len in decode_query_lens):
            raise ValueError(
                "CUDA graph decode query lengths must be positive, got "
                f"{decode_query_lens}. A draft-model graph must disable "
                "dynamic target decode-query expansion."
            )
        if self.full_decode_query_lens is not None:
            # The runtime phase policy can select a non-speculative lane that
            # is absent from the static batch-size schedule. Capture the
            # explicitly requested FULL lanes directly instead of intersecting
            # them with that incomplete schedule.
            decode_query_lens = sorted(self.full_decode_query_lens)
        max_decode_tokens = self.max_uniform_decode_reqs * max(decode_query_lens)

        # FULL uniform decode has a request-count contract that is independent
        # from PIECEWISE token padding. When elastic KV provides a post-profile
        # cap, capture every reachable request count without expanding the
        # mixed/PIECEWISE family.
        if separate_decode_routine and decode_mode:
            for decode_query_len in decode_query_lens:
                query_capture_sizes = set(capture_sizes)
                query_max_decode_tokens = (
                    self.max_uniform_decode_reqs * decode_query_len
                )
                capture_cap_boundary = decode_query_len in (
                    getattr(self, "full_decode_cap_query_lens", None) or set()
                )
                if capture_cap_boundary:
                    query_capture_sizes.add(query_max_decode_tokens)
                if self.max_uniform_decode_reqs < self.max_num_reqs:
                    if decode_query_len > 1:
                        query_capture_sizes.update(
                            decode_query_len * num_reqs
                            for num_reqs in range(
                                1, self.max_uniform_decode_reqs + 1
                            )
                        )
                    else:
                        # Single-token FULL decode can pad request rows.
                        # Preserve its sparse family and add only the exact
                        # cap boundary.
                        query_capture_sizes.add(self.max_uniform_decode_reqs)
                for num_tokens, num_active_loras in product(
                    sorted(query_capture_sizes), self.lora_capture_cases
                ):
                    rounded_num_tokens = round_up(num_tokens, decode_query_len)
                    rounded_num_reqs = rounded_num_tokens // decode_query_len

                    if (
                        rounded_num_tokens > max_decode_tokens
                        or (
                            rounded_num_tokens > max_cg_capture_size
                            and not (
                                capture_cap_boundary
                                and rounded_num_tokens == query_max_decode_tokens
                            )
                        )
                        or rounded_num_reqs > self.max_uniform_decode_reqs
                    ):
                        continue

                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=rounded_num_tokens,
                        num_reqs=rounded_num_reqs,
                        uniform_token_count=decode_query_len,
                        num_active_loras=num_active_loras,
                    )

                    # avoid duplicate graphs
                    if desc not in descs_by_mode[decode_mode]:
                        descs_by_mode[decode_mode].append(desc)
                        descs_by_token_lora[
                            (rounded_num_tokens, num_active_loras)
                        ].append(desc)

        for num_tokens, num_active_loras in product(
            registered_mixed_sizes, self.lora_capture_cases
        ):
            if mixed_mode:
                # for PIECEWISE graphs there is no limit on requests when replaying
                # i.e. no request padding is needed, so we leave it as None.
                # For breakable PW graphs, break-point kernels read the real batch
                # from the forward context; in-graph kernels handle the token padding
                # themselves from the padded slot_mapping (rows with slot == -1).
                num_reqs = None
                if mixed_mode == CUDAGraphMode.FULL:
                    num_reqs = min(num_tokens, self.max_num_reqs)
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)
                if num_tokens in self.dynamic_capture_sizes:
                    self._dynamic_graph_entries[desc] = DynamicGraphEntry(
                        descriptor=desc,
                        pinned=num_tokens in self.dynamic_graph_pinned_sizes,
                    )

        if not descs_by_token_lora:
            return

        all_token_counts = sorted({k[0] for k in descs_by_token_lora})
        current_range_start = 0
        for token_cg_size in all_token_counts:
            for i in range(current_range_start, token_cg_size + 1):
                for num_active_loras in self.lora_capture_cases:
                    staging_key = (token_cg_size, num_active_loras)
                    if staging_key in descs_by_token_lora:
                        self._candidates[(i, num_active_loras)] = descs_by_token_lora[
                            staging_key
                        ]
            current_range_start = token_cg_size + 1

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            startup_descs = [
                desc for desc in descs if desc not in self._dynamic_graph_entries
            ]
            if startup_descs:
                self._capture_descs[mode] = startup_descs
        self.max_capture_tokens = max(
            (
                desc.num_tokens
                for descs in self._capture_descs.values()
                for desc in descs
            ),
            default=0,
        )

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    def has_pending_dynamic_capture(self) -> bool:
        return self._dynamic_pending is not None

    def _observe_dynamic_descriptor(self, desc: BatchExecutionDescriptor) -> bool:
        entry = self._dynamic_graph_entries[desc]
        self._dynamic_epoch += 1
        entry.last_used_epoch = self._dynamic_epoch
        if entry.state == DynamicGraphResidency.COOLDOWN:
            if self._dynamic_epoch < entry.cooldown_until_epoch:
                return False
            entry.state = DynamicGraphResidency.WARM
            entry.hits = 0
        if entry.state == DynamicGraphResidency.HOT:
            return True
        entry.hits += 1
        if (
            entry.state == DynamicGraphResidency.WARM
            and entry.hits >= self.dynamic_graph_min_hits
            and self._dynamic_pending is None
        ):
            entry.state = DynamicGraphResidency.QUEUED
            self._dynamic_pending = desc
            logger.info(
                "Dynamic CUDA graph queued: descriptor=%s hits=%d epoch=%d",
                desc,
                entry.hits,
                self._dynamic_epoch,
            )
        return False

    @staticmethod
    def _runtime_batch_descriptor(desc: BatchExecutionDescriptor) -> BatchDescriptor:
        return BatchDescriptor(
            num_tokens=desc.num_tokens,
            has_lora=desc.num_active_loras > 0,
            num_active_loras=desc.num_active_loras,
        )

    def _cooldown_dynamic_entry(self, entry: DynamicGraphEntry) -> None:
        entry.state = DynamicGraphResidency.COOLDOWN
        entry.hits = 0
        entry.cooldown_until_epoch = (
            self._dynamic_epoch + self.dynamic_graph_cooldown_steps
        )

    def _evict_dynamic_entry(self, entry: DynamicGraphEntry) -> None:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        if entry.state == DynamicGraphResidency.HOT:
            torch.cuda.synchronize(self.device)
        evicted = CUDAGraphWrapper.evict_batch_descriptor(
            self._runtime_batch_descriptor(entry.descriptor)
        )
        if entry.graph_segments and evicted != entry.graph_segments:
            raise RuntimeError(
                "Dynamic CUDA graph eviction was incomplete: "
                f"descriptor={entry.descriptor}, expected={entry.graph_segments}, "
                f"evicted={evicted}"
            )
        self._dynamic_resident_bytes -= entry.charged_bytes
        entry.charged_bytes = 0
        entry.graph_segments = 0
        entry.state = DynamicGraphResidency.WARM
        entry.hits = 0
        logger.info("Dynamic CUDA graph evicted: descriptor=%s", entry.descriptor)

    def _reserve_dynamic_capture(self, candidate: DynamicGraphEntry) -> bool:
        required = self.dynamic_graph_max_entry_bytes
        while self._dynamic_resident_bytes + required > self.dynamic_graph_budget_bytes:
            victims = sorted(
                (
                    entry
                    for entry in self._dynamic_graph_entries.values()
                    if entry.state == DynamicGraphResidency.HOT
                    and not entry.pinned
                    and entry is not candidate
                ),
                key=lambda entry: entry.last_used_epoch,
            )
            if not victims:
                return False
            self._evict_dynamic_entry(victims[0])
        local_free = torch.accelerator.get_memory_info()[0]
        free_tensor = torch.tensor(local_free, dtype=torch.int64, device=self.device)
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                free_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
        return int(free_tensor.item()) >= required + self.dynamic_graph_guard_bytes

    def _consensus_dynamic_candidate(
        self, desc: BatchExecutionDescriptor
    ) -> bool:
        group = get_tp_group()
        identity = (
            int(desc.cg_mode.value),
            desc.num_tokens,
            desc.num_reqs,
            desc.uniform_token_count,
            desc.num_active_loras,
        )
        expected = group.broadcast_object(
            identity if group.is_first_rank else None,
            src=0,
        )
        matches = torch.tensor(
            int(identity == expected), dtype=torch.int32, device=self.device
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                matches,
                op=torch.distributed.ReduceOp.MIN,
                group=group.device_group,
            )
        return bool(matches.item())

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
        capture_descs: dict[
            CUDAGraphMode, list[BatchExecutionDescriptor]
        ] | None = None,
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a forward_fn. For FULL and breakable PIECEWISE modes,
                it is invoked once with warmup=True and again with warmup=False
                because attention backends may mutate or lazily initialize
                metadata during warmup.
        """
        with graph_capture(device=self.device):
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            selected_descs = self._capture_descs if capture_descs is None else capture_descs
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in selected_descs:
                    continue

                descs = selected_descs[mode]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc, warmup=True)

                    # Warmup
                    forward_fn(CUDAGraphMode.NONE)

                    # Capture
                    logger.debug(
                        "CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc
                    )
                    if (
                        desc.cg_mode == CUDAGraphMode.PIECEWISE
                        and not self.use_breakable_cg
                    ):
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        # Capture with fresh attention state.
                        forward_fn = create_forward_fn(desc, warmup=False)
                        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                            forward_fn(CUDAGraphMode.PIECEWISE)
                            continue
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        prefix_diagnostic_graph = (
                            (
                                self._p3_node_cutoff is not None
                                or bool(self._p3_node_prefix_counts)
                            )
                            and progress_bar_desc == "Capturing CUDA graphs"
                            and desc.uniform_token_count == 3
                            and desc.num_tokens == 6
                        )
                        debug_p3_graph = (
                            (
                                self._p3_graph_debug
                                or prefix_diagnostic_graph
                            )
                            and desc.uniform_token_count == 3
                            and desc.num_tokens == 6
                        )
                        graph = torch.cuda.CUDAGraph(keep_graph=debug_p3_graph)
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        if self.pool is not None:
                            set_graph_pool_id(self.pool)
                        else:
                            set_graph_pool_id(current_platform.graph_pool_handle())
                        with torch.cuda.graph(graph, self.pool):
                            forward_fn(CUDAGraphMode.NONE)
                            # Join offloader's copy stream after forward to avoid
                            # unjoined stream error. The last layer's start_prefetch
                            # forks copy_stream, but wait_prefetch only happens in
                            # the next forward pass.
                            get_offloader().join_after_forward()
                        self.graphs[desc] = graph
                        if debug_p3_graph:
                            graph_role = (
                                progress_bar_desc.lower()
                                .replace(" ", "-")
                                .replace("/", "-")
                            )
                            dump_path = (
                                "/workspace/eval-results/"
                                f"p3-cuda-graph-{graph_role}-pid{os.getpid()}-"
                                f"tokens{desc.num_tokens}-reqs{desc.num_reqs}.dot"
                            )
                            try:
                                cudart = ctypes.CDLL("libcudart.so.13")
                                dot_print = cudart.cudaGraphDebugDotPrint
                                dot_print.argtypes = [
                                    ctypes.c_void_p,
                                    ctypes.c_char_p,
                                    ctypes.c_uint,
                                ]
                                dot_print.restype = ctypes.c_int
                                result = dot_print(
                                    ctypes.c_void_p(graph.raw_cuda_graph()),
                                    dump_path.encode(),
                                    ctypes.c_uint(1),  # Verbose.
                                )
                                if result != 0:
                                    raise RuntimeError(
                                        "cudaGraphDebugDotPrint failed with "
                                        f"cudaError_t={result}"
                                    )
                                logger.warning(
                                    "P3 CUDA graph debug dump: desc=%s path=%s",
                                    desc,
                                    dump_path,
                                )
                                if prefix_diagnostic_graph:
                                    from cuda.bindings import runtime as cuda_runtime

                                    with open(
                                        dump_path,
                                        encoding="utf-8",
                                    ) as dot_file:
                                        dot_contents = dot_file.read()
                                    topo_by_local_id = {
                                        int(local_id): int(topo_id)
                                        for local_id, topo_id in re.findall(
                                            r"(?<![A-Za-z])(\d+) "
                                            r"\(topoId: (\d+)\)",
                                            dot_contents,
                                        )
                                    }
                                    graph.instantiate()
                                    raw_graph = cuda_runtime.cudaGraph_t(
                                        graph.raw_cuda_graph()
                                    )
                                    raw_graph_exec = cuda_runtime.cudaGraphExec_t(
                                        graph.raw_cuda_graph_exec()
                                    )
                                    error, _, num_nodes = (
                                        cuda_runtime.cudaGraphGetNodes(raw_graph)
                                    )
                                    if int(error) != 0:
                                        raise RuntimeError(
                                            "cudaGraphGetNodes(size) failed: "
                                            f"{error}"
                                        )
                                    error, nodes, actual_nodes = (
                                        cuda_runtime.cudaGraphGetNodes(
                                            raw_graph, num_nodes
                                        )
                                    )
                                    if int(error) != 0 or actual_nodes != num_nodes:
                                        raise RuntimeError(
                                            "cudaGraphGetNodes failed: "
                                            f"error={error}, expected={num_nodes}, "
                                            f"actual={actual_nodes}"
                                        )
                                    supported_types = {
                                        cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel,
                                        cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy,
                                        cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset,
                                    }
                                    diagnostic_nodes: list[tuple[int, Any]] = []
                                    if self._p3_node_prefix_counts:
                                        if self._p3_node_prefix_counts[-1] > actual_nodes:
                                            raise ValueError(
                                                "p3_node_prefix_counts exceeds "
                                                f"target graph size {actual_nodes}: "
                                                f"{self._p3_node_prefix_counts}"
                                            )
                                        active_cutoff = (
                                            actual_nodes
                                            - self._p3_node_prefix_counts[0]
                                        )
                                    else:
                                        assert self._p3_node_cutoff is not None
                                        if self._p3_node_cutoff > actual_nodes:
                                            raise ValueError(
                                                "p3_node_cutoff exceeds target "
                                                f"graph size {actual_nodes}: "
                                                f"{self._p3_node_cutoff}"
                                            )
                                        active_cutoff = self._p3_node_cutoff
                                    disabled = 0
                                    for node in nodes[:actual_nodes]:
                                        error, local_id = (
                                            cuda_runtime.cudaGraphNodeGetLocalId(node)
                                        )
                                        if int(error) != 0:
                                            raise RuntimeError(
                                                "cudaGraphNodeGetLocalId failed: "
                                                f"{error}"
                                            )
                                        error, node_type = (
                                            cuda_runtime.cudaGraphNodeGetType(node)
                                        )
                                        if int(error) != 0:
                                            raise RuntimeError(
                                                "cudaGraphNodeGetType failed: "
                                                f"{error}"
                                            )
                                        topo_id = topo_by_local_id.get(int(local_id))
                                        if topo_id is None:
                                            raise RuntimeError(
                                                "DOT is missing CUDA graph node "
                                                f"local_id={int(local_id)}"
                                            )
                                        if node_type in supported_types:
                                            diagnostic_nodes.append((topo_id, node))
                                        if (
                                            topo_id < active_cutoff
                                            and node_type in supported_types
                                        ):
                                            result = (
                                                cuda_runtime.cudaGraphNodeSetEnabled(
                                                    raw_graph_exec, node, 0
                                                )
                                            )
                                            if int(result[0]) != 0:
                                                raise RuntimeError(
                                                    "cudaGraphNodeSetEnabled failed: "
                                                    f"local_id={int(local_id)}, "
                                                    f"topo_id={topo_id}, "
                                                    f"error={result[0]}"
                                                )
                                            disabled += 1
                                    logger.warning(
                                        "P3 CUDA graph prefix diagnostic armed: "
                                        "desc=%s cutoff=%d prefix_counts=%s "
                                        "disabled=%d/%d",
                                        desc,
                                        active_cutoff,
                                        self._p3_node_prefix_counts or None,
                                        disabled,
                                        actual_nodes,
                                    )
                                    self._p3_prefix_diagnostic_descs.add(desc)
                                    self._p3_prefix_diagnostic_state[desc] = (
                                        raw_graph_exec,
                                        diagnostic_nodes,
                                        actual_nodes,
                                    )
                            except Exception:
                                logger.exception(
                                    "P3 CUDA graph debug dump failed: desc=%s "
                                    "path=%s",
                                    desc,
                                    dump_path,
                                )
                                if prefix_diagnostic_graph:
                                    raise
                        compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""

        effective_loras = self._resolve_effective_loras(num_active_loras)
        key = (num_tokens, effective_loras)
        if self._graphs_captured and num_tokens > 0 and key in self._candidates:
            for desc in self._candidates[key]:
                if _is_compatible(
                    desc,
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    effective_loras,
                ):
                    if desc in self._dynamic_graph_entries:
                        if not self._observe_dynamic_descriptor(desc):
                            continue
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=effective_loras,
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """Replay a captured FULL cudagraph."""
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        diagnose_p3_graph = (
            self._p3_crash_diagnostic and desc.uniform_token_count == 3
        )
        sync_p3_graph = self._p3_global_sync and desc.uniform_token_count == 3
        if sync_p3_graph:
            torch.cuda.synchronize(self.device)
        if diagnose_p3_graph:
            logger.warning("P3 CUDA graph replay begin: desc=%s", desc)
        prefix_diagnostic = (
            desc in self._p3_prefix_diagnostic_descs
        )
        if prefix_diagnostic:
            if self._p3_node_prefix_counts:
                from cuda.bindings import runtime as cuda_runtime

                raw_graph_exec, diagnostic_nodes, total_nodes = (
                    self._p3_prefix_diagnostic_state[desc]
                )
                for prefix_count in self._p3_node_prefix_counts:
                    active_cutoff = total_nodes - prefix_count
                    for topo_id, node in diagnostic_nodes:
                        result = cuda_runtime.cudaGraphNodeSetEnabled(
                            raw_graph_exec,
                            node,
                            int(topo_id >= active_cutoff),
                        )
                        if int(result[0]) != 0:
                            raise RuntimeError(
                                "cudaGraphNodeSetEnabled failed during "
                                "progressive replay: "
                                f"topo_id={topo_id}, cutoff={active_cutoff}, "
                                f"error={result[0]}"
                            )
                    logger.warning(
                        "P3 CUDA graph progressive prefix begin: "
                        "desc=%s prefix_nodes=%d cutoff=%d",
                        desc,
                        prefix_count,
                        active_cutoff,
                    )
                    self.graphs[desc].replay()
                    torch.cuda.synchronize(self.device)
                    logger.warning(
                        "P3 CUDA graph progressive prefix passed: "
                        "desc=%s prefix_nodes=%d cutoff=%d",
                        desc,
                        prefix_count,
                        active_cutoff,
                    )
            else:
                self.graphs[desc].replay()
                torch.cuda.synchronize(self.device)
            raise RuntimeError(
                "P3 CUDA graph prefix diagnostic completed without a CUDA "
                "fault: "
                f"cutoff={self._p3_node_cutoff}, "
                f"prefix_counts={self._p3_node_prefix_counts or None}, "
                f"desc={desc}"
            )
        self.graphs[desc].replay()
        if sync_p3_graph:
            torch.cuda.synchronize(self.device)
        if diagnose_p3_graph:
            logger.warning("P3 CUDA graph replay end: desc=%s", desc)

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableCUDAGraphWrapper(
                model, self.vllm_config
            )

    def run_pw_graph(self, model: nn.Module, model_inputs: dict[str, Any]) -> Any:
        if not self.use_breakable_cg:
            # Default: Use torch-compiled piecewise cudagraph.
            return model(**model_inputs)
        assert self.breakable_cg_runner is not None
        return self.breakable_cg_runner(**model_inputs)


class ModelCudaGraphManager(CudaGraphManager):
    """CudaGraphManager with model-specific capture and hidden state management."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        full_decode_query_lens: set[int] | None = None,
        full_decode_cap_query_lens: set[int] | None = None,
        tp3_sd_phase_reduce: bool = False,
        tp3_owner_prequant: bool = False,
        max_uniform_decode_reqs: int | None = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            full_decode_query_lens=full_decode_query_lens,
            full_decode_cap_query_lens=full_decode_cap_query_lens,
            max_uniform_decode_reqs=max_uniform_decode_reqs,
        )
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.aux_hidden_states_token_major: list[bool] = []
        self.use_aux_hidden_state_outputs = False
        self.intermediate_tensors: IntermediateTensors | None = None
        self.tp3_sd_phase_reduce = tp3_sd_phase_reduce
        self.tp3_owner_prequant = tp3_owner_prequant
        self._tp3_owner_replay_receipts: set[
            tuple[int, int | None, int | None]
        ] = set()

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        capture_descs: dict[
            CUDAGraphMode, list[BatchExecutionDescriptor]
        ] | None = None,
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        owner_capture_descs: set[tuple[int, int | None, int | None]] = set()
        if self.use_breakable_cg:
            self.init_breakable_cg_runner(model)

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)

            # Set LoRA state before capture so kernels see correct adapters.
            if lora_capture_hook is not None:
                lora_capture_hook(desc.num_active_loras, num_reqs, num_tokens)

            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )

            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None
                assert intermediate_tensors is not None
                model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]

            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
            )

            if envs.AG2_VLLM_NVFP4_MARLIN_ISOLATE_PREFILL:
                layout = input_buffers.marlin_request_layout_cpu
                if desc.cg_mode == CUDAGraphMode.FULL:
                    # FULL graphs are replayed only for uniform decode lanes.
                    # Capture the original single batched Marlin launch.
                    layout[0] = num_reqs
                    layout[1] = num_reqs
                    layout[2 : num_reqs + 3].zero_()
                else:
                    # The op is cudagraph-unsafe in PIECEWISE mode and is
                    # executed dynamically. This valid one-request layout is
                    # only for compile/capture warmup.
                    layout[0] = 1
                    layout[1] = 0
                    layout[2] = 0
                    layout[3] = num_tokens

            # Capture with dummy rows marked as padding.
            input_buffers.is_padding.fill_(True)

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                # FULL graph capture calls this closure with cg_mode NONE.
                # Use the target manager's static role plus the descriptor,
                # never graph mode, to mark the qlen1 target-decode lane.
                tp3_sd_phase_reduce = (
                    self.tp3_sd_phase_reduce
                    and desc.cg_mode == CUDAGraphMode.FULL
                    and desc.uniform_token_count == 1
                )
                tp3_owner_prequant_decode = (
                    self.tp3_owner_prequant
                    and desc.cg_mode == CUDAGraphMode.FULL
                    and desc.uniform_token_count == self.decode_query_len
                    and num_tokens >= envs.AG2_VLLM_TP3_OWNER_MIN_ROWS
                )
                if tp3_owner_prequant_decode:
                    owner_capture_descs.add(
                        (desc.num_tokens, desc.num_reqs, desc.uniform_token_count)
                    )
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                        tp3_sd_phase_reduce=tp3_sd_phase_reduce,
                    )
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                    is_padding=input_buffers.is_padding[:num_tokens],
                    tp3_sd_phase_reduce=tp3_sd_phase_reduce,
                    tp3_owner_prequant_decode=tp3_owner_prequant_decode,
                    marlin_request_layout_cpu=(
                        input_buffers.marlin_request_layout_cpu
                        if envs.AG2_VLLM_NVFP4_MARLIN_ISOLATE_PREFILL
                        else None
                    ),
                ):
                    if cg_mode == CUDAGraphMode.PIECEWISE:
                        # PIECEWISE graph (compiled PW or breakable, chosen inside
                        # run_pw_graph).
                        model_output = self.run_pw_graph(model, model_inputs)
                    else:
                        model_output = model(**model_inputs)

                if cg_mode == CUDAGraphMode.PIECEWISE:
                    # PW CUDA graph (compiled or breakable) internally handles the
                    # model outputs. No need to keep track of the hidden states.
                    return None

                if self.is_last_pp_rank:
                    # Last PP rank (common case).
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        aux_hidden_states = []
                    if self.hidden_states is None:
                        self.hidden_states = torch.empty(
                            (self.max_capture_tokens, *hidden_states.shape[1:]),
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                    self.hidden_states[:num_tokens] = hidden_states
                    if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            (
                                torch.empty(
                                    (self.max_capture_tokens, *x.shape[1:]),
                                    dtype=x.dtype,
                                    device=x.device,
                                )
                                if x.ndim > 0 and x.shape[0] == num_tokens
                                else torch.empty_like(x)
                            )
                            for x in aux_hidden_states
                        ]
                        self.aux_hidden_states_token_major = [
                            aux.ndim > 0 and aux.shape[0] == num_tokens
                            for aux in aux_hidden_states
                        ]
                    for destination, aux, token_major in zip(
                        self.aux_hidden_states,
                        aux_hidden_states,
                        self.aux_hidden_states_token_major,
                        strict=True,
                    ):
                        copy_aux_hidden_state(
                            destination,
                            aux,
                            num_tokens=num_tokens,
                            token_major=token_major,
                        )
                else:
                    # Non-last PP rank.
                    assert isinstance(model_output, IntermediateTensors)
                    intermediate_tensors = model_output
                    if self.intermediate_tensors is None:
                        self.intermediate_tensors = IntermediateTensors.empty_like(
                            intermediate_tensors
                        )
                    for k, v in intermediate_tensors.tensors.items():
                        self.intermediate_tensors[k][:num_tokens] = v

            return forward_fn

        super().capture(create_forward_fn, progress_bar_desc, capture_descs)
        if self.tp3_owner_prequant:
            logger.warning(
                "TP3 owner prequant CUDA Graph capture completed: descriptors=%s",
                sorted(owner_capture_descs),
            )

    def capture_next_dynamic(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
    ) -> bool:
        desc = self._dynamic_pending
        if desc is None:
            return False
        entry = self._dynamic_graph_entries[desc]
        if not self._consensus_dynamic_candidate(desc):
            self._dynamic_pending = None
            self._cooldown_dynamic_entry(entry)
            logger.error(
                "Dynamic CUDA graph rank descriptor mismatch: descriptor=%s", desc
            )
            return False
        if not self._reserve_dynamic_capture(entry):
            self._dynamic_pending = None
            self._cooldown_dynamic_entry(entry)
            logger.warning(
                "Dynamic CUDA graph budget or guard refused capture: "
                "descriptor=%s resident_bytes=%d budget_bytes=%d",
                desc,
                self._dynamic_resident_bytes,
                self.dynamic_graph_budget_bytes,
            )
            return False

        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        runtime_desc = self._runtime_batch_descriptor(desc)
        start_free = torch.accelerator.get_memory_info()[0]
        try:
            self.capture(
                model,
                model_state,
                input_buffers,
                intermediate_tensors,
                block_tables,
                attn_groups,
                kv_cache_config,
                has_lora=has_lora,
                use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                lora_capture_hook=lora_capture_hook,
                progress_bar_desc="Promoting dynamic CUDA graph",
                capture_descs={CUDAGraphMode.PIECEWISE: [desc]},
            )
            torch.cuda.synchronize(self.device)
        except Exception:
            CUDAGraphWrapper.evict_batch_descriptor(runtime_desc)
            self._dynamic_pending = None
            self._cooldown_dynamic_entry(entry)
            logger.exception(
                "Dynamic CUDA graph capture failed: descriptor=%s", desc
            )
            raise

        graph_segments = CUDAGraphWrapper.count_batch_descriptor(runtime_desc)
        local_charge = max(0, start_free - torch.accelerator.get_memory_info()[0])
        charge = torch.tensor(local_charge, dtype=torch.int64, device=self.device)
        segment_count = torch.tensor(
            graph_segments, dtype=torch.int32, device=self.device
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                charge,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
            torch.distributed.all_reduce(
                segment_count,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
        charged_bytes = int(charge.item())
        graph_segments = int(segment_count.item())
        if (
            graph_segments <= 0
            or charged_bytes > self.dynamic_graph_max_entry_bytes
            or self._dynamic_resident_bytes + charged_bytes
            > self.dynamic_graph_budget_bytes
        ):
            CUDAGraphWrapper.evict_batch_descriptor(runtime_desc)
            self._dynamic_pending = None
            self._cooldown_dynamic_entry(entry)
            logger.error(
                "Dynamic CUDA graph publication rejected: descriptor=%s "
                "segments=%d charged_bytes=%d resident_bytes=%d budget_bytes=%d",
                desc,
                graph_segments,
                charged_bytes,
                self._dynamic_resident_bytes,
                self.dynamic_graph_budget_bytes,
            )
            return False
        get_tp_group().barrier()
        entry.state = DynamicGraphResidency.HOT
        entry.charged_bytes = charged_bytes
        entry.graph_segments = graph_segments
        self._dynamic_resident_bytes += charged_bytes
        self._dynamic_pending = None
        logger.info(
            "Dynamic CUDA graph HOT: descriptor=%s segments=%d charged_bytes=%d "
            "resident_bytes=%d budget_bytes=%d",
            desc,
            graph_segments,
            charged_bytes,
            self._dynamic_resident_bytes,
            self.dynamic_graph_budget_bytes,
        )
        return True

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Replay a captured FULL cudagraph and return hidden states."""
        super().run_fullgraph(desc)
        receipt = (desc.num_tokens, desc.num_reqs, desc.uniform_token_count)
        if (
            self.tp3_owner_prequant
            and desc.uniform_token_count == self.decode_query_len
            and desc.num_tokens >= envs.AG2_VLLM_TP3_OWNER_MIN_ROWS
            and receipt not in self._tp3_owner_replay_receipts
        ):
            self._tp3_owner_replay_receipts.add(receipt)
            logger.warning(
                "TP3 owner prequant CUDA Graph replay observed: descriptor=%s",
                receipt,
            )
        if not self.is_last_pp_rank:
            assert self.intermediate_tensors is not None
            return self.intermediate_tensors[: desc.num_tokens]

        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [
            view_aux_hidden_state(
                tensor,
                num_tokens=desc.num_tokens,
                token_major=token_major,
            )
            for tensor, token_major in zip(
                self.aux_hidden_states,
                self.aux_hidden_states_token_major,
                strict=True,
            )
        ]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    # NOTE(woosuk): Attention metadata is required not just by standard attention
    # kernels, but also by specialized attention-like operations (e.g., Inkling's sconv,
    # DSV4 compressor), which maintain their own states and require special metadata
    # such as block tables.
    # During CUDA graph capture:
    # - For FULL CUDA graphs: We set for_capture=True so that both attention and
    #   attention-like ops produce capturable metadata compatible with CUDA graphs.
    # - For PIECEWISE CUDA graphs: We still build attention metadata, but set
    #   for_capture=False. This is because:
    #     * Attention-like ops (such as sconv or DSV4 compressor) may not be used as
    #       breakpoints in PIECEWISE CUDA graphs, so we must generate their attention
    #       metadata so they can execute and be captured during graph capture.
    #     * Standard attention ops that are treated as breakpoints will be executed
    #       eagerly at capture time (not included in the graph itself), and for these,
    #       setting for_capture=False is essential. Some attention backends
    #       (like linear attention) cannot generate capturable metadata for prefill,
    #       so for_capture=False ensures they execute without issue.
    #     * We assume that attention-like operations intended for capture will still
    #       produce capturable metadata, even when for_capture=False. While this
    #       assumption is brittle, it currently works in practice.
    # In summary: We always generate attention metadata for both FULL and PIECEWISE
    # CUDA graphs, setting for_capture=True for FULL graphs, and for_capture=False
    # for PIECEWISE graphs, to ensure correct execution and capture.
    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return AttentionState(attn_metadata, slot_mappings_by_layer)
