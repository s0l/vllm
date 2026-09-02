# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.ag2_draft_capture import (
    Ag2DraftCapture,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.ag2_flight_recorder import (
    Ag2FlightRecorder,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.ag2_mtp_layer_capture import (
    MTP_LAYER_TRACE_FIELDS,
    Ag2MtpLayerCapture,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup, get_uniform_decode_token_count

logger = init_logger(__name__)


def _resolve_prefill_cudagraph_mode(
    configured_mode: CUDAGraphMode,
    attention_support: AttentionCGSupport,
    query_len: int,
) -> CUDAGraphMode:
    """Select a safe graph mode for the first autoregressive draft pass."""
    if (
        query_len > 1
        and attention_support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    ):
        if configured_mode.has_piecewise_cudagraphs():
            return CUDAGraphMode.PIECEWISE
        return CUDAGraphMode.NONE
    return configured_mode


class AutoRegressiveSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._ag2_mtp_boundary_saved = False
        self._ag2_mtp_boundary_path: Path | None = None
        self._ag2_draft_capture = Ag2DraftCapture.from_env()
        self._ag2_mtp_layer_capture = Ag2MtpLayerCapture.from_env(
            get_tensor_model_parallel_rank()
        )
        self._ag2_current_mtp_trace: dict[str, torch.Tensor] | None = None
        self._ag2_flight_recorder = (
            Ag2FlightRecorder.from_env(num_drafts=self.num_speculative_steps)
            if get_tensor_model_parallel_rank() == 0
            else None
        )

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.sample_src_positions = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        self.inputs_embeds: torch.Tensor | None = None

        if self._ag2_draft_capture is not None:
            steps = self.num_speculative_steps
            shape = (steps, self.max_num_reqs)
            hidden_shape = (*shape, self.hidden_size)
            self._ag2_current_input_hidden = torch.zeros(
                self.max_num_reqs,
                self.hidden_size,
                dtype=self.dtype,
                device=device,
            )
            self._ag2_current_sample_hidden = torch.zeros_like(
                self._ag2_current_input_hidden
            )
            self._ag2_current_input_ids = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int64, device=device
            )
            self._ag2_current_positions = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int64, device=device
            )
            self._ag2_step_input_hidden = torch.zeros(
                hidden_shape, dtype=self.dtype, device=device
            )
            self._ag2_step_sample_hidden = torch.zeros_like(self._ag2_step_input_hidden)
            self._ag2_step_input_ids = torch.full(
                shape, -1, dtype=torch.int64, device=device
            )
            self._ag2_step_positions = torch.full(
                shape, -1, dtype=torch.int64, device=device
            )
            self._ag2_step_top_pairs = torch.zeros(
                *shape,
                self.vllm_config.parallel_config.tensor_parallel_size,
                2,
                dtype=torch.float32,
                device=device,
            )
            self._ag2_step_top_tokens = torch.full(
                shape, -1, dtype=torch.int64, device=device
            )

        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.use_fused_multi_step_decode = False

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if self.supports_mm_inputs:
            self.inputs_embeds = torch.zeros(
                self.max_num_tokens,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

        if self._ag2_draft_capture is None:
            return
        if not hasattr(self.model, "ag2_enable_top_token_trace"):
            raise RuntimeError(
                "AG2 draft capture requires local-argmax provenance support"
            )
        self.model.ag2_enable_top_token_trace(self.max_num_reqs)
        if self._ag2_mtp_layer_capture is not None:
            if not hasattr(self.model, "ag2_enable_mtp_layer_trace"):
                raise RuntimeError("AG2 MTP layer capture requires Qwen trace support")
            # x5 dispatches through the eight-row FULL CUDA Graph. Wider
            # startup capture shapes retain fail-closed sentinels and are not
            # persisted by the bounded five-request diagnostic.
            self.model.ag2_enable_mtp_layer_trace(rows=8, history_tokens=640)

    # Lifecycle hooks for model-specific optimizations. Subclasses override
    # the ones they need. These fire in both `capture` and `propose` so that
    # any state they toggle (e.g. attention flags baked into a CUDA graph) is
    # identical at capture time and replay time.
    def on_prefill_begin(self, num_reqs: int) -> None: ...

    def on_prefill_end(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_begin(self, num_reqs: int) -> None: ...

    def on_multi_step_decode_end(self, num_reqs: int) -> None: ...

    @property
    def advance_draft_positions(self) -> bool:
        """
        Whether to increment positions and seq_lens between draft steps.

        True for Eagle/standard MTP (each step produces new KV).
        False for Gemma4 MTP (Q-only, shares target KV, constant positions).
        """
        return True

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._configure_fused_multi_step_decode()

    def _configure_fused_multi_step_decode(self) -> None:
        if self.num_speculative_steps == 1:
            self.use_fused_multi_step_decode = False
            return

        if not self.advance_draft_positions:
            self.use_fused_multi_step_decode = True
            return

        unsupported_backends = sorted(
            {
                attn_group.backend.get_name()
                for attn_groups in self.attn_groups
                for attn_group in attn_groups
                if not attn_group.supports_draft_decode_metadata_update
            }
        )
        self.use_fused_multi_step_decode = not unsupported_backends
        if unsupported_backends:
            logger.info_once(
                "Fused multi-step draft decode is not supported by attention "
                "backend(s) %s; falling back to rebuilding attention metadata "
                "between draft steps.",
                ", ".join(unsupported_backends),
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        prefill_query_len = self.num_speculative_steps + 1
        prefill_cudagraph_mode = _resolve_prefill_cudagraph_mode(
            cudagraph_mode,
            self.attn_cg_support.min_cg_support,
            prefill_query_len,
        )
        if prefill_cudagraph_mode != cudagraph_mode:
            logger.info(
                "Draft prefill query_len=%d will use %s instead of %s "
                "because %s only supports full CUDA graphs for single-token "
                "decode.",
                prefill_query_len,
                prefill_cudagraph_mode,
                cudagraph_mode,
                self.attn_cg_support.min_cg_attn_backend,
            )

        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            prefill_cudagraph_mode,
            prefill_query_len,
            expand_dynamic_decode_query_lens=False,
            max_uniform_decode_reqs=(
                self.kv_cache_config.effective_max_resident_seqs or self.max_num_reqs
            ),
            owner="mtp_prefill",
            elastic_graph_activation="speculative",
            elastic_graph_token_source="step",
        )

        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        # Initialize cudagraph manager for draft decodes (draft positions > 0).
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
            expand_dynamic_decode_query_lens=False,
            max_uniform_decode_reqs=(
                self.kv_cache_config.effective_max_resident_seqs or self.max_num_reqs
            ),
            owner="mtp_decode",
            elastic_graph_activation="speculative",
            elastic_graph_token_source="requests",
        )
        if self._ag2_mtp_layer_capture is not None:
            self.decode_cudagraph_manager.require_capture_output(MTP_LAYER_TRACE_FIELDS)

    def dynamic_cudagraph_managers(self) -> tuple[SpeculatorCudaGraphManager, ...]:
        return tuple(
            manager
            for manager in (
                self.prefill_cudagraph_manager,
                self.decode_cudagraph_manager,
            )
            if manager is not None
        )

    def capture_next_dynamic(self, manager: SpeculatorCudaGraphManager) -> bool:
        if manager is self.prefill_cudagraph_manager:
            forward_fn = self._prefill
            progress_bar_desc = "Promoting dynamic prefill CUDA graph"
            # Draft prefill consumes the target model's attention metadata and
            # slot mappings at runtime. Match the accepted startup-capture
            # contract instead of constructing incompatible metadata from the
            # drafter's post-prefill decode buffers.
            capture_input_buffers = self.target_input_buffers
            capture_attn_groups = self.target_attn_groups
            capture_begin = self.on_prefill_begin
            capture_end = self.on_prefill_end
        elif manager is self.decode_cudagraph_manager:
            forward_fn = self._decode_capture_fn()
            progress_bar_desc = "Promoting dynamic decode CUDA graph"
            capture_input_buffers = self.input_buffers
            capture_attn_groups = self.attn_groups
            capture_begin = self.on_multi_step_decode_begin
            capture_end = self.on_multi_step_decode_end
        else:
            raise ValueError("unknown speculator CUDA Graph manager")

        # Dynamic promotion happens before live request state is copied into
        # the drafter for this step. Clear indices left by earlier warmups so
        # the dummy capture cannot gather beyond its exact runtime descriptor.
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()

        def capture_override(capture_descs, capture_complete_hook) -> None:
            capture_begin(self.max_num_reqs)
            try:
                manager.capture(
                    forward_fn,
                    self.model_state,
                    capture_input_buffers,
                    self.block_tables,
                    capture_attn_groups,
                    self.kv_cache_config,
                    progress_bar_desc=progress_bar_desc,
                    capture_descs=capture_descs,
                    capture_complete_hook=capture_complete_hook,
                )
            finally:
                capture_end(self.max_num_reqs)

        return manager.capture_next_dynamic(
            self.model,
            self.model_state,
            self.input_buffers,
            None,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            capture_override=capture_override,
        )

    def _decode_capture_fn(self) -> Callable[..., Any]:
        """Return the same decode body that one runtime replay consumes."""
        return (
            self._generate_fused_drafts
            if self.use_fused_multi_step_decode
            else self._generate_draft
        )

    def capture(self) -> None:
        logger.info("Capturing model for speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()

        # Capture the prefill routine (model forward + compute_logits +
        # sample).
        # For FULL graphs, the entire routine is recorded as one graph.
        # For PIECEWISE, only the model's compiled regions are captured
        # and the rest (compute_logits, gumbel_sample) runs eagerly.
        # Draft prefill reuses the target model's attention metadata at
        # runtime, so capture builds its dummy metadata through the target
        # model runner's builders and buffers.
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)

        self.on_prefill_begin(self.max_num_reqs)
        self.prefill_cudagraph_manager.capture(
            self._prefill,
            self.model_state,
            self.target_input_buffers,
            self.block_tables,
            self.target_attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing prefill CUDA graphs",
        )
        self.on_prefill_end(self.max_num_reqs)

        if self.num_speculative_steps == 1:
            return

        self.on_multi_step_decode_begin(self.max_num_reqs)
        # Capture either the fused decode loop or one decode step per graph.
        assert self.decode_cudagraph_manager is not None
        self.decode_cudagraph_manager.capture(
            self._decode_capture_fn(),
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing decode CUDA graphs",
        )
        self.on_multi_step_decode_end(self.max_num_reqs)

    @torch.inference_mode()
    def capture_target_lm_head_inputs(
        self,
        target_lm_head_hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        if self._ag2_draft_capture is not None:
            self._ag2_draft_capture.stage_target_lm_head_inputs(
                rank=get_tensor_model_parallel_rank(),
                hidden_states=target_lm_head_hidden_states,
                input_batch=input_batch,
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_query_len = input_batch.num_scheduled_tokens.max()
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )

        # NOTE(woosuk): To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions. By doing so,
        # we can also reuse the attention metadata (e.g., query_start_loc,
        # seq_lens) of the target model.
        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_tokens_padded].copy_(hidden_states)

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )
        if self._ag2_mtp_layer_capture is not None and not dummy_run:
            self._ag2_mtp_layer_capture.begin(
                req_ids=input_batch.req_ids,
                idx_mapping=input_batch.idx_mapping,
                num_reqs=num_reqs,
            )

        # Get the input ids and last token indices for the speculator.
        prepare_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )
        self._maybe_save_ag2_mtp_boundary(
            input_batch=input_batch,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            dummy_run=dummy_run,
        )

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay.
        uniform_token_count = get_uniform_decode_token_count(
            num_reqs,
            # Use the actual number of tokens without padding added by
            # the target model during FULL cudagraph.
            num_tokens,
            max_query_len,
            input_batch.has_prefill,
        )
        prefill_batch_desc, prefill_batch_sync = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            # The target carrier is padded for stable buffers, but the
            # prefill CUDA Graph safety decision must use live rows. Passing
            # num_tokens_after_padding here hid underfilled B64 replay from
            # the manager even after the target correctly selected NONE.
            input_batch.num_tokens,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            # Target compile/profile dummy runs propagate into both drafter
            # phases without scheduler admission. Compile the direct path;
            # only live admitted work may enter elastic Graph dispatch.
            need_eager=dummy_run or is_profile,
            dp_sync=dp_sync,
        )
        num_tokens_across_dp = (
            prefill_batch_sync.num_tokens_across_dp
            if prefill_batch_sync is not None
            else None
        )

        self._prepare_eplb_forward(num_tokens)

        # Target and MTP owners can select different physical modes for the
        # same live input (for example target FULL M3 and MTP PIECEWISE M4).
        # The target InputBatch mask is consequently only target-sized. Build
        # the MTP mask in its own stable buffer and mark its physical tail;
        # slicing the shorter target mask silently returned three elements for
        # a four-token MTP replay.
        self.input_buffers.is_padding[: input_batch.num_tokens].copy_(
            input_batch.is_padding[: input_batch.num_tokens]
        )
        if prefill_batch_desc.num_tokens > input_batch.num_tokens:
            self.input_buffers.is_padding[
                input_batch.num_tokens : prefill_batch_desc.num_tokens
            ].fill_(True)
        prefill_is_padding = self.input_buffers.is_padding[
            : prefill_batch_desc.num_tokens
        ]

        prefill_receipt_name = (
            "ag2.mtp_prefill_graph_receipt"
            f"|mode={prefill_batch_desc.cg_mode.name}"
            f"|tokens_unpadded={input_batch.num_tokens}"
            f"|tokens_padded={prefill_batch_desc.num_tokens}"
            f"|requests={num_reqs}"
            f"|uniform_token_count={uniform_token_count}"
            f"|descriptor={prefill_batch_desc}"
        )
        prefill_default_scope = (
            "ag2.mtp_prefill.full"
            if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL
            else (
                "ag2.mtp_prefill.piecewise"
                if prefill_batch_desc.cg_mode == CUDAGraphMode.PIECEWISE
                else "ag2.mtp_prefill.eager"
            )
        )
        prefill_scope = (
            torch.profiler.record_function(prefill_receipt_name)
            if os.getenv("AG2_VLLM_GRAPH_MODE_RECEIPT") == "1" and not dummy_run
            else record_function_or_nullcontext(prefill_default_scope)
        )

        with prefill_scope:
            if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
                # Replay the full graph for draft prefill.
                assert self.prefill_cudagraph_manager is not None
                self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
            else:
                # The target model's attention metadata and slot mappings
                # can directly be used for draft prefill, because of the
                # identical batch shape and KV cache layout.
                self._prefill(
                    num_reqs,
                    prefill_batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                    mm_inputs=mm_inputs,
                    physical_num_reqs=prefill_batch_desc.physical_num_reqs,
                    runtime_generation=prefill_batch_desc.runtime_generation,
                    num_tokens_unpadded=input_batch.num_tokens,
                    is_padding=prefill_is_padding,
                )

        self.on_prefill_end(num_reqs)
        self._ag2_snapshot_proposal_step(
            0,
            num_reqs,
            trace_row_indices=self.last_token_indices[:num_reqs],
        )

        if self.num_speculative_steps == 1:
            # Early exit.
            self._maybe_append_ag2_mtp_drafts(num_reqs)
            return self.draft_tokens[:num_reqs, :1]

        # Prepare the inputs for the decode steps.
        with record_function_or_nullcontext("ag2.mtp_prepare_decode"):
            prepare_decode_inputs(
                self.draft_tokens[:num_reqs, 0],
                input_batch.seq_lens,
                num_rejected,
                self.input_buffers,
                self.sample_src_positions,
                self.max_model_len,
                self.max_num_reqs,
                advance_draft_positions=self.advance_draft_positions,
            )

        # Each request produces exactly 1 token per draft generation step,
        # enabling FULL graph replay.
        decode_batch_desc, decode_batch_sync = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=dummy_run or is_profile,
        )
        num_tokens_across_dp = (
            decode_batch_sync.num_tokens_across_dp
            if decode_batch_sync is not None
            else None
        )
        self.input_buffers.is_padding[:num_reqs].fill_(False)
        if decode_batch_desc.num_tokens > num_reqs:
            self.input_buffers.is_padding[
                num_reqs : decode_batch_desc.num_tokens
            ].fill_(True)

        self.on_multi_step_decode_begin(num_reqs)
        # Generate the remaining num_speculative_steps - 1 draft tokens.
        decode_receipt_name = (
            "ag2.mtp_decode_graph_receipt"
            f"|mode={decode_batch_desc.cg_mode.name}"
            f"|tokens_unpadded={num_reqs}"
            f"|tokens_padded={decode_batch_desc.num_tokens}"
            f"|requests={num_reqs}"
            "|uniform_token_count=1"
            f"|descriptor={decode_batch_desc}"
        )
        decode_default_scope = (
            "ag2.mtp_decode.full"
            if decode_batch_desc.cg_mode == CUDAGraphMode.FULL
            else "ag2.mtp_decode.eager"
        )
        decode_scope = (
            torch.profiler.record_function(decode_receipt_name)
            if os.getenv("AG2_VLLM_GRAPH_MODE_RECEIPT") == "1" and not dummy_run
            else record_function_or_nullcontext(decode_default_scope)
        )
        with decode_scope:
            decode_fn = (
                self._fused_multi_step_decode
                if self.use_fused_multi_step_decode
                else self._multi_step_decode
            )
            decode_fn(
                num_reqs,
                dummy_run and skip_attn_for_dummy_run,
                decode_batch_desc,
                num_tokens_across_dp,
                input_batch.seq_lens_cpu_upper_bound,
            )

        self._maybe_append_ag2_mtp_drafts(num_reqs)
        if self._ag2_flight_recorder is not None and not dummy_run:
            self._ag2_flight_recorder.record(
                num_reqs=num_reqs,
                req_ids=input_batch.req_ids,
                draft_tokens=self.draft_tokens,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                idx_mapping=self.idx_mapping,
            )
        if self._ag2_draft_capture is not None and not dummy_run:
            self._ag2_draft_capture.collect(
                rank=get_tensor_model_parallel_rank(),
                num_reqs=num_reqs,
                draft_logits=self.draft_logits,
                draft_tokens=self.draft_tokens,
                hidden_states=self.hidden_states,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                idx_mapping=self.idx_mapping,
                temperature=self.temperature,
                seeds=self.seeds,
                step_input_hidden=self._ag2_step_input_hidden,
                step_sample_hidden=self._ag2_step_sample_hidden,
                step_input_ids=self._ag2_step_input_ids,
                step_positions=self._ag2_step_positions,
                step_top_pairs=self._ag2_step_top_pairs,
                step_top_tokens=self._ag2_step_top_tokens,
            )
        if self._ag2_mtp_layer_capture is not None and not dummy_run:
            self._ag2_mtp_layer_capture.finalize(
                draft_tokens=self.draft_tokens,
                num_reqs=num_reqs,
            )
        self.on_multi_step_decode_end(num_reqs)
        return self.draft_tokens[:num_reqs]

    def _maybe_save_ag2_mtp_boundary(
        self,
        *,
        input_batch: InputBatch,
        num_tokens: int,
        num_reqs: int,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        dummy_run: bool,
    ) -> None:
        """Save the exact first MTP inputs once for offline reference replay."""
        output = os.environ.get("AG2_VLLM_MTP_BOUNDARY_OUTPUT")
        if not output or dummy_run or self._ag2_mtp_boundary_saved:
            return

        rank = get_tensor_model_parallel_rank()
        path = Path(f"{output}.rank{rank}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "rank": rank,
            "num_speculative_steps": self.num_speculative_steps,
            "num_tokens": num_tokens,
            "num_reqs": num_reqs,
            "target_input_ids": input_batch.input_ids[:num_tokens].detach().cpu(),
            "target_positions": input_batch.positions[:num_tokens].detach().cpu(),
            "target_query_start_loc": input_batch.query_start_loc[: num_reqs + 1]
            .detach()
            .cpu(),
            "target_seq_lens": input_batch.seq_lens[:num_reqs].detach().cpu(),
            "target_hidden_states": self.hidden_states[:num_tokens].detach().cpu(),
            "mtp_input_ids": self.input_buffers.input_ids[:num_tokens].detach().cpu(),
            "mtp_positions": self.input_buffers.positions[:num_tokens].detach().cpu(),
            "mtp_query_start_loc": self.input_buffers.query_start_loc[: num_reqs + 1]
            .detach()
            .cpu(),
            "mtp_seq_lens": self.input_buffers.seq_lens[:num_reqs].detach().cpu(),
            "last_token_indices": self.last_token_indices[:num_reqs].detach().cpu(),
            "num_sampled": num_sampled[:num_reqs].detach().cpu(),
            "num_rejected": num_rejected[:num_reqs].detach().cpu(),
        }
        torch.save(payload, path)
        self._ag2_mtp_boundary_saved = True
        self._ag2_mtp_boundary_path = path
        logger.warning(
            "Saved one-shot MTP consumed-input boundary rank=%d path=%s "
            "tokens=%d requests=%d K=%d",
            rank,
            path,
            num_tokens,
            num_reqs,
            self.num_speculative_steps,
        )

    def _maybe_append_ag2_mtp_drafts(self, num_reqs: int) -> None:
        path = self._ag2_mtp_boundary_path
        if path is None:
            return
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["runtime_draft_tokens"] = (
            self.draft_tokens[:num_reqs, : self.num_speculative_steps].detach().cpu()
        )
        torch.save(payload, path)
        self._ag2_mtp_boundary_path = None

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        return_ag2_mtp_trace: bool = False,
        cudagraph_owner: str = "mtp_prefill",
        physical_num_reqs: int | None = None,
        runtime_generation: str = "static",
        num_tokens_unpadded: int | None = None,
        is_padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        if num_tokens_unpadded is None:
            num_tokens_unpadded = num_tokens
        if not 0 < num_tokens_unpadded <= num_tokens:
            raise RuntimeError("invalid MTP padded/live token contract")
        if is_padding is None:
            is_padding = self.input_buffers.is_padding[:num_tokens]
        if is_padding.numel() != num_tokens:
            raise RuntimeError("MTP padding mask does not match physical tokens")
        batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens,
            cudagraph_owner=cudagraph_owner,
            physical_num_reqs=physical_num_reqs,
            runtime_generation=runtime_generation,
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
            is_padding=is_padding,
            num_tokens_unpadded=num_tokens_unpadded,
            # Every call in this model is owned by the MTP drafter. Keep the
            # identity explicit and graph-static so FULL capture cannot alias
            # it with an equal-shaped target decode or short prefill.
            tp3_mtp_device_ce=(os.environ.get("AG2_VLLM_MTP_DEVICE_CE", "0") == "1"),
        ):
            inputs_embeds = None
            if self.supports_mm_inputs:
                assert self.inputs_embeds is not None
                # Merge multimodal embeddings with input ids.
                mm_embeds, is_mm_embed = mm_inputs or (None, None)
                num_input_tokens = (
                    is_mm_embed.shape[0] if is_mm_embed is not None else num_tokens
                )
                self.inputs_embeds[:num_input_tokens] = self.model.embed_input_ids(
                    self.input_buffers.input_ids[:num_input_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                inputs_embeds = self.inputs_embeds[:num_tokens]

            model_inputs = dict(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self._ag2_mtp_layer_capture is not None:
                # This model is compiled with TorchCompileWithNoGuards. A
                # Python flag observed as False by the first profile call
                # cannot be switched to True later for decode graph capture.
                # Keep the diagnostic return contract invariant from the
                # first compiled invocation; persistence remains position-
                # gated outside the graph.
                model_inputs["return_ag2_mtp_trace"] = True
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                # Draft prefill with PIECEWISE cudagraph (compiled PW or breakable),
                # chosen inside run_pw_graph.
                assert self.prefill_cudagraph_manager is not None
                ret_hidden_states = self.prefill_cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                # Eager (NONE): call the raw model directly.
                ret_hidden_states = self.model(**model_inputs)
        # Some MTP models declare a single-tensor contract but return
        # (logits_hidden, feedback_hidden) for final-norm correctness.
        mtp_trace = None
        if isinstance(ret_hidden_states, tuple) and len(ret_hidden_states) == 3:
            last_hidden_states, hidden_states, mtp_trace = ret_hidden_states
        elif isinstance(ret_hidden_states, tuple):
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states, mtp_trace

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        physical_num_reqs: int | None = None,
        runtime_generation: str = "static",
        num_tokens_unpadded: int | None = None,
        is_padding: torch.Tensor | None = None,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        # The output hidden state at position P (= positions) and the token id
        # at P+1 are used to draft the token at P+2. Sampling keys a draw by the
        # position before the sampled token, so the net adjustment is +1.
        sample_src_positions = positions + 1
        idx_mapping = self.idx_mapping[:num_reqs]

        if self._ag2_draft_capture is not None:
            self._ag2_current_input_hidden[:num_reqs].copy_(
                self.hidden_states[last_token_indices]
            )
            self._ag2_current_input_ids[:num_reqs].copy_(
                self.input_buffers.input_ids[last_token_indices]
            )
            self._ag2_current_positions[:num_reqs].copy_(positions)

        last_hidden_states, hidden_states, mtp_trace = self._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            mm_inputs=mm_inputs,
            cudagraph_owner="mtp_prefill",
            physical_num_reqs=physical_num_reqs,
            runtime_generation=runtime_generation,
            num_tokens_unpadded=num_tokens_unpadded,
            is_padding=is_padding,
        )
        self._ag2_current_mtp_trace = mtp_trace
        sample_hidden_states = last_hidden_states[last_token_indices]
        if self._ag2_draft_capture is not None:
            self._ag2_current_sample_hidden[:num_reqs].copy_(sample_hidden_states)

        sample_hidden_states = last_hidden_states[last_token_indices]
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            sample_hidden_states,
            sample_src_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        if last_hidden_states is hidden_states:
            self.hidden_states[:num_reqs] = sample_hidden_states
        else:
            self.hidden_states[:num_reqs] = hidden_states[last_token_indices]
        self.input_buffers.positions[:num_reqs] = positions
        self.sample_src_positions[:num_reqs] = sample_src_positions

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, self.num_speculative_steps):
            # Rebuild every step when positions advance, or just once
            # on the first step when positions are constant (Gemma4 MTP).
            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
                attn_metadata = self._build_draft_attn_metadata(
                    num_reqs=num_reqs,
                    num_reqs_padded=batch_desc.num_reqs or num_reqs,
                    num_tokens_padded=batch_desc.num_tokens,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=step,
                )

            self.current_draft_step.fill_(step)

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                graph_output = self.decode_cudagraph_manager.run_fullgraph(batch_desc)
                if self._ag2_mtp_layer_capture is not None:
                    if not isinstance(graph_output, dict):
                        raise RuntimeError(
                            "AG2 MTP FULL graph did not publish its trace outputs"
                        )
                    self._ag2_current_mtp_trace = graph_output
            else:
                self._generate_draft(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=batch_desc.cg_mode,
                    physical_num_reqs=batch_desc.physical_num_reqs,
                    runtime_generation=batch_desc.runtime_generation,
                )
            self._ag2_snapshot_proposal_step(step, num_reqs)

    def _fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        if not skip_attn:
            slot_mappings = self.block_tables.compute_slot_mappings(
                idx_mapping,
                query_start_loc,
                positions,
                batch_desc.num_tokens,
            )
            if batch_desc.cg_mode != CUDAGraphMode.FULL:
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=batch_desc.num_reqs or num_reqs,
                num_tokens_padded=batch_desc.num_tokens,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=1,
            )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return

        self._generate_fused_drafts(
            num_reqs,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
            batch_desc.cg_mode,
        )

    def _generate_fused_drafts(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        physical_num_reqs: int | None = None,
        runtime_generation: str = "static",
    ) -> None:
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        attn_groups = (
            [group for groups in self.attn_groups for group in groups]
            if attn_metadata is not None
            else []
        )

        for step in range(1, self.num_speculative_steps):
            self.current_draft_step.fill_(step)
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                physical_num_reqs=physical_num_reqs,
                runtime_generation=runtime_generation,
            )
            if (
                step < self.num_speculative_steps - 1
                and attn_metadata is not None
                and self.advance_draft_positions
            ):
                self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    num_tokens_padded,
                )
                for attn_group in attn_groups:
                    attn_group.update_draft_decode_metadata(attn_metadata)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        physical_num_reqs: int | None = None,
        runtime_generation: str = "static",
    ) -> dict[str, torch.Tensor] | None:
        self._prepare_eplb_forward(num_reqs)

        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        if self._ag2_draft_capture is not None:
            self._ag2_current_input_hidden[:num_reqs].copy_(
                self.hidden_states[:num_reqs]
            )
            self._ag2_current_input_ids[:num_reqs].copy_(
                self.input_buffers.input_ids[:num_reqs]
            )
            self._ag2_current_positions[:num_reqs].copy_(positions)
        # Run the draft model forward pass.
        last_hidden_states, hidden_states, mtp_trace = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            return_ag2_mtp_trace=self._ag2_mtp_layer_capture is not None,
            cudagraph_owner="mtp_decode",
            physical_num_reqs=physical_num_reqs,
            runtime_generation=runtime_generation,
            num_tokens_unpadded=num_reqs,
            is_padding=self.input_buffers.is_padding[:num_tokens_padded],
        )
        self._ag2_current_mtp_trace = mtp_trace
        last_hidden_states = last_hidden_states[:num_reqs]
        if self._ag2_draft_capture is not None:
            self._ag2_current_sample_hidden[:num_reqs].copy_(last_hidden_states)

        # Sample the draft tokens.
        sample_hidden_states = last_hidden_states[:num_reqs]
        sample_src_positions = self.sample_src_positions[:num_reqs]
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            sample_src_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

        # Update the inputs for the next step.
        update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            self.sample_src_positions,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )
        return mtp_trace

    def _ag2_snapshot_proposal_step(
        self,
        step: int,
        num_reqs: int,
        *,
        trace_row_indices: torch.Tensor | None = None,
    ) -> None:
        """Snapshot one completed proposal step outside graph dispatch."""
        if self._ag2_draft_capture is None and self._ag2_mtp_layer_capture is None:
            return
        if not 0 <= step < self.num_speculative_steps:
            raise RuntimeError(f"AG2 proposal step is out of bounds: {step}")
        if self._ag2_draft_capture is not None:
            self._ag2_step_input_hidden[step, :num_reqs].copy_(
                self._ag2_current_input_hidden[:num_reqs]
            )
            self._ag2_step_sample_hidden[step, :num_reqs].copy_(
                self._ag2_current_sample_hidden[:num_reqs]
            )
            self._ag2_step_input_ids[step, :num_reqs].copy_(
                self._ag2_current_input_ids[:num_reqs]
            )
            self._ag2_step_positions[step, :num_reqs].copy_(
                self._ag2_current_positions[:num_reqs]
            )
            top_pairs, top_tokens = self.model.ag2_get_top_token_trace()
            self._ag2_step_top_pairs[step, :num_reqs].copy_(top_pairs[:num_reqs])
            self._ag2_step_top_tokens[step, :num_reqs].copy_(top_tokens[:num_reqs])
        if self._ag2_mtp_layer_capture is not None:
            self._ag2_mtp_layer_capture.stage(
                proposal_step=step,
                num_reqs=num_reqs,
                trace=self._ag2_current_mtp_trace,
                dispatch_positions=self._ag2_current_positions,
                trace_row_indices=trace_row_indices,
            )


@triton.jit
def _prepare_prefill_inputs_kernel(
    last_token_indices_ptr,
    draft_current_step_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_query_start_loc_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)

    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling.
        # Get the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)

    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)

    # Copy query start locations.
    tl.store(draft_query_start_loc_ptr + req_idx, query_start)
    # Copy sequence lengths.
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)
    if req_idx == (num_reqs - 1):
        # Reset the current draft step to 0.
        tl.store(draft_current_step_ptr, 0)
        # Pad query_start_loc for CUDA graphs.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(draft_query_start_loc_ptr + block, query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_prefill_inputs(
    # [num_reqs]
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    max_num_reqs,
) -> torch.Tensor:
    num_reqs = input_batch.num_reqs
    _prepare_prefill_inputs_kernel[(num_reqs,)](
        last_token_indices,
        current_draft_step,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        max_num_reqs,
        BLOCK_SIZE=1024,
    )
    return last_token_indices


@triton.jit
def _prepare_decode_inputs_kernel(
    draft_tokens_ptr,
    draft_tokens_stride,
    target_seq_lens_ptr,
    num_rejected_ptr,
    input_ids_ptr,
    positions_ptr,
    sample_src_positions_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        # Compute query_start_loc. Pad it with the last query_start_loc
        # for CUDA graphs.
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            q = tl.where(block < num_reqs, block, num_reqs)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, q, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(req_idx, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    # draft token -> input id.
    draft_token = tl.load(draft_tokens_ptr + req_idx * draft_tokens_stride)
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Advance the draft sampling key.
    sample_position = tl.load(sample_src_positions_ptr + req_idx)
    tl.store(sample_src_positions_ptr + req_idx, sample_position + 1)

    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    seq_len = target_seq_len - num_rejected
    if ADVANCE_DRAFT_POSITIONS:
        # Compute position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(seq_lens_ptr + req_idx, seq_len)


def prepare_decode_inputs(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    sample_src_positions: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool = True,
):
    num_reqs = draft_tokens.shape[0]
    _prepare_decode_inputs_kernel[(num_reqs + 1,)](
        draft_tokens,
        draft_tokens.stride(0),
        target_seq_lens,
        num_rejected,
        input_buffers.input_ids,
        input_buffers.positions,
        sample_src_positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        max_model_len,
        max_num_reqs,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
    )


@triton.jit
def _update_draft_inputs_kernel(
    output_draft_tokens_ptr,
    output_draft_tokens_stride,
    next_input_hidden_states_ptr,
    next_input_hidden_states_stride,
    input_ids_ptr,
    positions_ptr,
    sample_src_positions_ptr,
    seq_lens_ptr,
    draft_tokens_ptr,
    current_draft_step_ptr,
    hidden_states_ptr,
    hidden_states_stride,
    hidden_size,
    max_model_len,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    ADVANCE_DRAFT_POSITIONS: tl.constexpr,
):
    req_idx = tl.program_id(0)

    # Write the sampled draft token into self.draft_tokens[req_idx, step].
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    step = tl.load(current_draft_step_ptr)
    tl.store(
        output_draft_tokens_ptr + req_idx * output_draft_tokens_stride + step,
        draft_token,
    )

    if step >= num_speculative_steps - 1:
        # This is the final step. Skip updating draft forward inputs.
        return

    # Advance the draft sampling key.
    sample_position = tl.load(sample_src_positions_ptr + req_idx)
    tl.store(sample_src_positions_ptr + req_idx, sample_position + 1)

    # Write the sampled draft token into the input ids tensor for the next
    # forward pass.
    tl.store(input_ids_ptr + req_idx, draft_token)

    # Copy hidden states into the input hidden states tensor for the next
    # forward pass.
    for i in range(0, hidden_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < hidden_size
        hidden_states = tl.load(
            hidden_states_ptr + req_idx * hidden_states_stride + block,
            mask=mask,
        )
        tl.store(
            next_input_hidden_states_ptr
            + req_idx * next_input_hidden_states_stride
            + block,
            hidden_states,
            mask=mask,
        )

    if ADVANCE_DRAFT_POSITIONS:
        # Increment position and seq_lens.
        # NOTE(woosuk): To prevent out-of-range access, we clamp these values
        # if they reach the max model length.
        position = tl.load(positions_ptr + req_idx)
        position = tl.minimum(position + 1, max_model_len - 1)
        tl.store(positions_ptr + req_idx, position)

        seq_len = tl.load(seq_lens_ptr + req_idx)
        seq_len = tl.minimum(seq_len + 1, max_model_len)
        tl.store(seq_lens_ptr + req_idx, seq_len)


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    hidden_states: torch.Tensor,
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    sample_src_positions: torch.Tensor,
    num_reqs: int,
    max_model_len: int,
    num_speculative_steps: int,
    advance_draft_positions: bool = True,
):
    _, hidden_size = hidden_states.shape
    _update_draft_inputs_kernel[(num_reqs,)](
        output_draft_tokens,
        output_draft_tokens.stride(0),
        next_input_hidden_states,
        next_input_hidden_states.stride(0),
        input_buffers.input_ids,
        input_buffers.positions,
        sample_src_positions,
        input_buffers.seq_lens,
        draft_tokens,
        current_draft_step,
        hidden_states,
        hidden_states.stride(0),
        hidden_size,
        max_model_len,
        num_speculative_steps,
        BLOCK_SIZE=1024,
        ADVANCE_DRAFT_POSITIONS=advance_draft_positions,
    )
