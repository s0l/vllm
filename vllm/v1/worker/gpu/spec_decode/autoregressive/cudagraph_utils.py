# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class SpeculatorCudaGraphManager(CudaGraphManager):
    """CudaGraphManager for draft prefill and decode.

    Builds fresh dummy inputs and attention metadata for every warmup and
    capture pass so that the contents of the shared persistent buffers
    (e.g. query_start_loc, seq_lens, FA3 scheduler metadata) always match
    the batch descriptor being captured. Reusing metadata built during an
    earlier capture would execute kernels with stale buffer contents.
    """

    def _store_capture_output(
        self,
        desc: BatchExecutionDescriptor,
        output: Any,
    ) -> None:
        required = getattr(self, "_ag2_required_capture_output_fields", None)
        if required is not None:
            if not isinstance(output, dict):
                raise RuntimeError(
                    "Required speculator CUDA Graph output was not published "
                    f"during capture for {desc}"
                )
            missing = required.difference(output)
            if missing:
                raise RuntimeError(
                    "Speculator CUDA Graph output is incomplete during capture "
                    f"for {desc}: missing={sorted(missing)}"
                )
        if not hasattr(self, "_ag2_capture_outputs"):
            self._ag2_capture_outputs: dict[BatchExecutionDescriptor, Any] = {}
        self._ag2_capture_outputs[desc] = output

    def require_capture_output(self, fields: frozenset[str]) -> None:
        if not fields:
            raise ValueError("required CUDA Graph output fields cannot be empty")
        self._ag2_required_capture_output_fields = fields

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
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
                layout[0] = num_reqs
                layout[1] = num_reqs
                layout[2 : num_reqs + 3].zero_()

            def captured_forward(cg_mode: CUDAGraphMode) -> Any:
                output = forward_fn(
                    num_reqs,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cg_mode,
                )
                if not warmup:
                    self._store_capture_output(desc, output)
                return output

            return captured_forward

        super().capture(create_forward_fn, progress_bar_desc)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> Any:
        super().run_fullgraph(desc)
        outputs = getattr(self, "_ag2_capture_outputs", {})
        return outputs.get(desc)
