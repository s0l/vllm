# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import os
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, NamedTuple, Protocol

import torch
import torch.nn as nn
from tqdm import tqdm

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
    return (
        (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
        and desc.num_active_loras == num_active_loras
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
        # Precompute actual num_active_loras -> captured case mapping so that
        # dispatch() is a plain dict lookup instead of a per-call bisect.
        self._lora_dispatch_map, self._max_lora_case = self._build_lora_dispatch_map()

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None

        self._graphs_captured = False

        self._candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}

        # Breakable CUDA graph (PW CUDA graph without torch.compile)
        self.use_breakable_cg = (
            is_breakable_cudagraph_enabled()
            and self.cudagraph_mode.has_piecewise_cudagraphs()
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
                        or rounded_num_tokens > max_cg_capture_size
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
            capture_sizes, self.lora_capture_cases
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
            self._capture_descs[mode] = descs

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
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
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
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
        tp3_sd_phase_reduce: bool = False,
        max_uniform_decode_reqs: int | None = None,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            full_decode_query_lens=full_decode_query_lens,
            max_uniform_decode_reqs=max_uniform_decode_reqs,
        )
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False
        self.intermediate_tensors: IntermediateTensors | None = None
        self.tp3_sd_phase_reduce = tp3_sd_phase_reduce

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
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
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
                skip_attn=(
                    desc.cg_mode == CUDAGraphMode.PIECEWISE
                    and not self.use_breakable_cg
                ),
            )

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
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    assert (attn_metadata is not None) == self.use_breakable_cg
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
                        self.hidden_states = torch.empty_like(hidden_states)
                    self.hidden_states[:num_tokens] = hidden_states
                    if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            torch.empty_like(x) for x in aux_hidden_states
                        ]
                    for i, aux in enumerate(aux_hidden_states):
                        self.aux_hidden_states[i][:num_tokens] = aux
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

        super().capture(create_forward_fn, progress_bar_desc)

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Replay a captured FULL cudagraph and return hidden states."""
        super().run_fullgraph(desc)
        if not self.is_last_pp_rank:
            assert self.intermediate_tensors is not None
            return self.intermediate_tensors[: desc.num_tokens]

        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    skip_attn: bool = False,
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

    attn_metadata = None
    if not skip_attn:
        attn_metadata = model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            input_block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=True,
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)
