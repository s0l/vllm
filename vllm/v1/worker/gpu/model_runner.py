# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NOTE: Coding style guide for this file:
This model runner is shared by all models: text and multimodal, generative
and embedding, public and private. As a result, this file must only contain
code that is common to every model. Model-specific behavior belongs in the
appropriate model-specific files.

In other words:
* Be paranoid about changing this file. It should remain stable.
* Be even more paranoid about adding new lines. It should remain minimal.

Even for shared features (for example, different parallelism modes), keep the
complexity out of this path. The less common the feature, the more it should be
hidden. Prefer utility functions defined elsewhere and call them from here,
instead of embedding feature-specific logic directly.
"""

import functools
import gc
import hashlib
import json
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from copy import copy, deepcopy
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.interfaces import requires_raw_input_tokens
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import (
    MultiModalBudget,
    get_dummy_encoder_profile_inputs,
)
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.gc_utils import freeze_gc_for_cudagraph_capture
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.elastic_graph import (
    DispatchRepresentation,
    ElasticPlanKind,
    ElasticStepPlan,
    ExecutionManifest,
    canonical_execution_request_order,
    execution_manifest_phase_from_step_key,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    DraftTokenIds,
    ECConnectorOutput,
    ModelRunnerOutput,
    RoutedExpertsTensors,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu import pcp_manager as pcp
from vllm.v1.worker.gpu.async_utils import (
    AsyncOutput,
    AsyncPoolingOutput,
    StepTimingCollector,
)
from vllm.v1.worker.gpu.attn_utils import (
    add_kv_sharing_layers_to_config,
    build_slot_mappings_by_layer,
    get_attn_cg_support,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.aux_hidden_trace import AuxHiddenTrace
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import (
    async_copy_to_gpu,
    set_default_max_concurrency,
)
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    DynamicGraphWorkingSet,
    ElasticExecutionPlanMismatch,
    ModelCudaGraphManager,
    make_cudagraph_stats,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    profile_cudagraph_memory as _profile_cudagraph_memory,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.ec_connector import NO_OP_EC_CONNECTOR, get_ec_connector
from vllm.v1.worker.gpu.elastic_gdn import (
    ElasticKVController,
    V2GDNCheckpointManager,
    validate_elastic_attention_block_tables,
)
from vllm.v1.worker.gpu.eplb_utils import EPLBController, step_eplb_after
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
    set_dummy_context,
)
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import (
    LoraState,
    create_lora_capture_hook,
    get_lora_capture_cases,
    get_num_active_loras_for_dispatch,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.sample.batch_shard import (
    BatchSharder,
    all_to_all_logits,
    gather_sampler_output,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.shutdown import free_before_shutdown
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    maybe_create_adaptive_verification_manager,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
    verify_supports_aux_hidden_states_over_pp,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    get_max_chunk_logits,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.gpu.target_boundary_capture import TargetBoundaryCapture
from vllm.v1.worker.gpu.ubatch_utils import (
    UBatchRunner,
    UBatchState,
    maybe_build_ubatch_runner,
)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    clear_layer_kv_caches,
    copy_kv_cache_blocks_inplace,
    get_kv_caches_for_block_copy,
    get_uniform_decode_token_count,
    prepare_kernel_block_sizes,
)
from vllm.v1.worker.workspace import lock_workspace, use_workspace_lane

logger = init_logger(__name__)

_AG2_GRAPH_MODE_RECEIPT = os.environ.get("AG2_VLLM_GRAPH_MODE_RECEIPT") == "1"
_AG2_GRAPH_MODE_LOGGED_RECEIPTS: set[tuple[str, str, int, int, int]] = set()


def _elastic_new_request_prompt_len(request: Any) -> int:
    """Return the prompt boundary used by both Scheduler and ReqState.

    V2 ``prefill_token_ids`` is the complete current token stream and may
    already include output tokens on a cached/repeated request. It is not the
    semantic prompt boundary used by ``Request.num_prompt_tokens``.
    """
    try:
        return length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids,
            getattr(request, "prompt_embeds", None),
        )
    except ValueError as error:
        raise ElasticExecutionPlanMismatch(
            "ELASTIC_EXECUTION_PLAN_MISMATCH: invalid new-request prompt "
            f"identity: {error}"
        ) from error


def _elastic_new_request_execution_prefill_len(request: Any) -> int:
    """Validate and return the scheduler-owned execution phase boundary.

    The semantic prompt length is not sufficient after preemption: emitted
    output tokens can become part of the stream whose KV/state must be
    reconstructed.  Elastic manifest validation therefore consumes the
    explicit scheduler boundary that ``RequestState.prefill_len`` will use,
    while retaining prompt and token-stream bounds as transport checks.
    """
    prompt_len = _elastic_new_request_prompt_len(request)
    execution_prefill_len = getattr(request, "execution_prefill_len", None)
    if type(execution_prefill_len) is not int:
        raise ElasticExecutionPlanMismatch(
            "ELASTIC_EXECUTION_PLAN_MISMATCH: new request omitted a valid "
            "integer execution_prefill_len"
        )
    if execution_prefill_len < prompt_len:
        raise ElasticExecutionPlanMismatch(
            "ELASTIC_EXECUTION_PLAN_MISMATCH: execution_prefill_len "
            f"{execution_prefill_len} is smaller than prompt length {prompt_len}"
        )
    prefill_token_ids = getattr(request, "prefill_token_ids", None)
    if prefill_token_ids is None:
        raise ElasticExecutionPlanMismatch(
            "ELASTIC_EXECUTION_PLAN_MISMATCH: new request omitted the "
            "transported prefill token stream"
        )
    if execution_prefill_len > len(prefill_token_ids):
        raise ElasticExecutionPlanMismatch(
            "ELASTIC_EXECUTION_PLAN_MISMATCH: execution_prefill_len "
            f"{execution_prefill_len} exceeds transported token stream "
            f"{len(prefill_token_ids)}"
        )
    return execution_prefill_len


def _validate_elastic_materialized_input_batch(
    manifest: ExecutionManifest,
    input_batch: InputBatch,
) -> None:
    """Read back the exact row lifecycle materialized by ``prepare_inputs``."""
    observed_request_ids = tuple(input_batch.req_ids)
    observed_query_lens = tuple(
        int(value) for value in input_batch.num_scheduled_tokens
    )
    observed_is_prefilling = tuple(
        bool(value) for value in input_batch.is_prefilling_np
    )
    observed_draft_rows = (
        (0,) * input_batch.num_reqs
        if input_batch.num_draft_tokens_per_req is None
        else tuple(int(value) for value in input_batch.num_draft_tokens_per_req)
    )
    if (
        observed_request_ids != manifest.request_ids
        or observed_query_lens != manifest.per_request_query_lens
        or observed_is_prefilling != manifest.per_request_is_prefilling
        or observed_draft_rows != manifest.scheduled_draft_rows
    ):
        raise RuntimeError(
            "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH: predicted and "
            "materialized InputBatch identities differ: "
            f"planned_ids={manifest.request_ids!r} "
            f"observed_ids={observed_request_ids!r} "
            f"planned_qlens={manifest.per_request_query_lens!r} "
            f"observed_qlens={observed_query_lens!r} "
            f"planned_prefill={manifest.per_request_is_prefilling!r} "
            f"observed_prefill={observed_is_prefilling!r} "
            f"planned_drafts={manifest.scheduled_draft_rows!r} "
            f"observed_drafts={observed_draft_rows!r}"
        )


def _prepare_elastic_local_staging_with_consensus(
    prepare_local_staging: Callable[[], Any],
    plan: ElasticStepPlan,
    working_set: Any,
    *,
    consensus_phase: str = "post_materialization",
    observer_fingerprint: Callable[[Any], str] | None = None,
) -> Any:
    """Finish local staging, then make every rank vote before collectives.

    Local request/input copies, attention table construction, Mamba state
    preprocessing, and LoRA activation can fail asymmetrically after the
    admitted plan has mutated worker state. Convert every such failure into
    the same post-mutation vote; raising locally before that vote can strand
    peers at the first model collective.
    """
    staged = None
    staging_error = None
    staged_fingerprint = None
    try:
        staged = prepare_local_staging()
        if observer_fingerprint is not None:
            staged_fingerprint = observer_fingerprint(staged)
    except Exception as error:
        staging_error = (
            "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH: local input "
            f"staging failed: {type(error).__name__}: {error}"
        )
    working_set.require_post_materialization_consensus(
        plan,
        validation_error=staging_error,
        observer_fingerprint=staged_fingerprint,
        phase=consensus_phase,
    )
    if staged is None:
        raise RuntimeError(
            "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH: all-rank vote accepted "
            "failed local input staging"
        )
    return staged


def _elastic_mm_staging_fingerprint(staged: Any) -> str:
    """Hash the ordered local MM tensor contract before TP embedding."""
    staged_embeddings, _prepared_inputs = staged
    if staged_embeddings is None:
        encoder_outputs = _prepared_inputs.get("encoder_outputs", [])
        payload = {
            "encoder_decoder_outputs": [
                {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for tensor in encoder_outputs
            ]
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    mm_embeddings, is_mm_embed = staged_embeddings
    payload = {
        "embeddings": [
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for tensor in mm_embeddings
        ],
        "mask_shape": list(is_mm_embed.shape),
        "mask_count": int(is_mm_embed.sum().item()),
        "mask_indices": torch.nonzero(is_mm_embed, as_tuple=False).flatten().tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _elastic_input_staging_fingerprint(staged: Any) -> str:
    """Bind branch choice and scheduled encoder identity into the first vote."""

    def schema(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {
                str(key): schema(item)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [schema(item) for item in value]
        return type(value).__qualname__

    staged_encoder = staged[7]
    encoder_signature: list[tuple[str, str]] = []
    encoder_batch_schema: list[Any] = []
    if staged_encoder is not None:
        hashes, batches = staged_encoder
        modalities = [
            modality
            for modality, num_items, _kwargs in batches
            for _ in range(num_items)
        ]
        encoder_signature = list(zip(hashes, modalities, strict=True))
        encoder_batch_schema = [
            {
                "modality": modality,
                "num_items": num_items,
                "kwargs": schema(kwargs),
            }
            for modality, num_items, kwargs in batches
        ]
    payload = {
        "mm_finalize_after_encoder": bool(staged[6]),
        "encoder_signature": encoder_signature,
        "encoder_batch_schema": encoder_batch_schema,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _stage_elastic_mm_inputs_with_consensus(
    model_state: Any,
    completed_encoder: Any,
    input_batch: InputBatch,
    req_states: Any,
    plan: ElasticStepPlan,
    working_set: Any,
) -> tuple[Any, dict[str, Any]]:
    """Gather local post-encoder state before the TP embedding collective."""

    def stage_mm_inputs():
        model_state.commit_staged_mm_encoder(completed_encoder)
        staged_embeddings = model_state.stage_mm_embeddings(input_batch, req_states)
        return staged_embeddings, model_state.prepare_inputs(input_batch, req_states)

    return _prepare_elastic_local_staging_with_consensus(
        stage_mm_inputs,
        plan,
        working_set,
        consensus_phase="post_mm_materialization",
        observer_fingerprint=_elastic_mm_staging_fingerprint,
    )


def _release_idle_graph_cache(scheduler_output: SchedulerOutput) -> bool:
    """Return whether a zero-token step is a real idle/reclaim boundary.

    Request-free MAINTENANCE is executable work: it publishes the exact graph
    owner set for the deferred user step. Treating it as X0 destroys that set
    before the scheduler can consume it and makes worker/scheduler residency
    diverge. Ordinary request-free steps, including RECLAIM, remain X0.
    """
    plan = scheduler_output.elastic_step_plan
    return not (plan is not None and plan.kind == ElasticPlanKind.MAINTENANCE)


class GPUModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self._draft_workspace_lane = int(
            self.speculative_config is not None and self.speculative_config.use_dspark()
        )
        self.observability_config = vllm_config.observability_config
        self.jit_warmup_registry = JitWarmupRegistry(vllm_config)

        self.device = device
        self.elastic_kv_controller = ElasticKVController(device)
        self._prepared_attn_groups: list[list[AttentionGroup]] | None = None
        self._prepared_attn_config_signature: tuple[Any, ...] | None = None
        self._prepared_kernel_block_sizes: tuple[int, ...] | None = None
        self.gdn_checkpoint_manager: V2GDNCheckpointManager | None = None
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        # Lazily initialized in _init_kv_zero_meta() when the KV cache needs
        # zeroing (e.g. hybrid models with fp8 KV cache).
        self.kv_block_zeroer: KVBlockZeroer | None = None

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.is_encoder_only = vllm_config.is_mm_encoder_only
        self.is_encoder_decoder = self.model_config.is_encoder_decoder

        self.output_copy_stream = torch.cuda.Stream(self.device)
        self.target_boundary_capture = TargetBoundaryCapture.from_env()

        # Pipeline parallelism.
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank

        # Size the UVA buffer pools to the max number of concurrent in-flight
        # steps. Must run before any pooled buffer is constructed
        set_default_max_concurrency(vllm_config.max_concurrent_batches)

        # PP broadcast/recv helper. Runs the collective on a side stream.
        self.pp_handler: PPHandler | None = None

        # Persistent buffer for intermediate tensors (non-first PP ranks).
        self.intermediate_tensors: IntermediateTensors | None = None

        # Data parallelism.
        self.dp_size = self.parallel_config.data_parallel_size
        self.dp_rank = self.parallel_config.data_parallel_rank

        # Dual batch overlap. Created in initialize_kv_cache(), once everything
        # it runs the microbatched forward with exists.
        self.ubatch_runner: UBatchRunner | None = None

        # Detect EP all2all peer faults to prevent emitting corrupted output.
        # Only meaningful for MoE + DP with an FT-capable all2all backend.
        self.check_ep_fault = False
        if self.dp_size > 1 and self.model_config.is_moe:
            self.check_ep_fault = get_ep_all2all_manager().support_fault_tolerance

        # Decode context parallelism.
        self.dcp_size = self.parallel_config.decode_context_parallel_size
        self.use_dcp = self.dcp_size > 1
        self.dcp_rank = get_dcp_group().rank_in_group if self.use_dcp else 0
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size

        # Multimodal
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        self.uses_inputs_embeds = (
            self.supports_mm_inputs or self.model_config.enable_prompt_embeds
        )
        self.encoder_cache = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            self.encoder_cache = EncoderCache()
        self.ec_connector = get_ec_connector(vllm_config, self.encoder_cache)

        # Speculative decoding.
        self.speculator = None
        self.use_aux_hidden_state_outputs = False
        self.aux_hidden_trace = AuxHiddenTrace.from_env()
        self.num_speculative_steps = vllm_config.num_speculative_tokens
        if self.speculative_config is not None:
            if self.is_last_pp_rank:
                self.speculator = init_speculator(self.vllm_config, self.device)

            if self.speculative_config.method in (
                "eagle3",
                "dflash",
                "dspark",
                "extract_hidden_states",
            ):
                # Drafting may require auxiliary hidden states from target model outputs
                self.use_aux_hidden_state_outputs = True
        if self.aux_hidden_trace.enabled:
            if self.use_aux_hidden_state_outputs:
                raise ValueError(
                    "Aux hidden trace cannot share auxiliary outputs with a drafter"
                )
            if self.use_pp:
                raise ValueError("Aux hidden trace does not support pipeline parallel")
            self.use_aux_hidden_state_outputs = True

        # Draft tokens propagation - for spec-dec + struct outputs.
        self.draft_tokens_handler = DraftTokensHandler(self.device)

        self.pcp_manager: pcp.PCPManager | None = None

        # Pooling models.
        self.is_pooling_model = self.model_config.runner_type == "pooling"
        self.pooling_runner: PoolingRunner | None = None

        # Multi-module MTP feeds its modules the next num_speculative_steps prefill
        # tokens during chunked prefill. Other speculators only read the immediate
        # next one.
        num_prefill_lookahead = (
            self.num_speculative_steps
            if self.speculative_config is not None
            and self.speculative_config.use_multi_module_mtp()
            else 1
        )

        self.step_timing = StepTimingCollector()

        # General request states.
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.adaptive_verification: AdaptiveVerificationManager | None = None
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        self.marlin_gate_up_scratch: torch.Tensor | None = None
        if (
            envs.AG2_VLLM_NVFP4_MARLIN_GATE_UP_SCRATCH
            or envs.AG2_VLLM_DCP_PREFILL_QUERY_SCRATCH
            or envs.AG2_VLLM_TP3_OWNER_PREQUANT
        ):
            # Qwen3.5/3.6 TP3 gate+up physical width: 2 * 5824.  Allocate the
            # largest destination before model/KV profiling so its ownership
            # and HBM cost are explicit instead of depending on runtime
            # allocator contiguity after smaller outputs split the segment.
            gate_up_elements = self.max_num_tokens * 11648
            workspace_elements = gate_up_elements
            if envs.AG2_VLLM_DCP_PREFILL_QUERY_SCRATCH:
                if self.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"DCP prefill query scratch requires BF16, got {self.dtype}"
                    )
                dcp_world_size = self.parallel_config.decode_context_parallel_size
                if (
                    self.parallel_config.tensor_parallel_size != 3
                    or dcp_world_size != 3
                ):
                    raise RuntimeError("DCP prefill query scratch requires TP3/DCP3")
                local_q_heads = self.model_config.get_num_attention_heads(
                    self.parallel_config
                )
                head_dim = self.model_config.get_head_size()
                dcp_elements = (
                    2 * self.max_num_tokens * dcp_world_size * local_q_heads * head_dim
                )
                workspace_elements = max(workspace_elements, dcp_elements)
            workspace_rows = cdiv(workspace_elements, 11648)
            self.marlin_gate_up_scratch = torch.empty(
                (workspace_rows, 11648),
                dtype=self.dtype,
                device=self.device,
            )
            from vllm.model_executor.kernels.linear.nvfp4.marlin import (
                set_nvfp4_marlin_gate_up_scratch,
            )

            # The historical accessor is also the early, model-runner-owned
            # workspace handoff to MTP.  DCP prefill needs that lifetime even
            # when the selected target linear backend is not Marlin.
            set_nvfp4_marlin_gate_up_scratch(self.marlin_gate_up_scratch)
            if envs.AG2_VLLM_TP3_OWNER_PREQUANT:
                from vllm.distributed.device_communicators.tp3_owner_prequant import (
                    set_tp3_owner_prequant_workspace,
                )

                set_tp3_owner_prequant_workspace(self.marlin_gate_up_scratch)
            if envs.AG2_VLLM_DCP_PREFILL_QUERY_SCRATCH:
                from vllm.v1.attention.backends.flashinfer import (
                    set_ag2_dcp_prefill_query_scratch,
                )

                set_ag2_dcp_prefill_query_scratch(self.marlin_gate_up_scratch)
            logger.info(
                "Configured model-runner-owned shared DCP/gate-up scratch: "
                "shape=%s bytes=%d dcp_prefill=%s",
                tuple(self.marlin_gate_up_scratch.shape),
                self.marlin_gate_up_scratch.numel()
                * self.marlin_gate_up_scratch.element_size(),
                envs.AG2_VLLM_DCP_PREFILL_QUERY_SCRATCH,
            )
        if self.use_pp:
            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
            )

        # Samplers and decode_query_len created in load_model() after
        # model_state exists (num_new_sampled_tokens_per_step from ModelState).
        self.sampler: Sampler | None = None
        self.rejection_sampler: RejectionSampler | None = None
        self.batch_sharder: BatchSharder | None = None
        self.prompt_logprobs_worker: PromptLogprobsWorker | None = None
        self.structured_outputs_worker: StructuredOutputsWorker | None = None
        self.cudagraph_manager: ModelCudaGraphManager | None = None

        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        self.lora_capture_cases = [0]
        if self.lora_config:
            self.lora_capture_cases = get_lora_capture_cases(
                self.lora_config, self.compilation_config
            )

        # KV Connector if configured.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        # For transferring state from execute_model to subsequent sample_tokens call.
        self.execute_model_state: ExecuteModelState | None = None

        # Expert parallelism load balancer.
        self.eplb = EPLBController(self.parallel_config, self.device)
        self.routed_experts_capturer: RoutedExpertsCapturer | None = None

        set_offloader(create_offloader(self.vllm_config.offload_config))

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    def init_routed_experts_capturer(self) -> None:
        """Initialize target-model capture on every participating worker."""
        self.routed_experts_capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=self.max_num_tokens,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
        )
        bind_routed_experts_capturer(self.model, self.routed_experts_capturer)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            tasks.extend(self.model_state.get_supported_generation_tasks())
        if self.is_pooling_model:
            # Do not rely on pooling_runner here, since this information is needed
            # on the first PP rank, while pooling_runner is only initialized
            # on the last PP rank.
            tasks.extend(PoolingRunner.get_supported_tasks(self.model))
        return tuple(tasks)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        # Model Runner V2 constructs layers through the shared model loader,
        # whose make_layers() consults the process-global offloader. Initialize
        # it before the first layer is constructed, matching the V1 lifecycle.
        set_offloader(create_offloader(self.vllm_config.offload_config))
        time_before_load = time.perf_counter()
        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        self.eplb.prepare_load()
        eplb_models_added = False
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info_once("Loading model from scratch...")

            # Capture warmup providers selected while constructing the model.
            with self.jit_warmup_registry.activate():
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.vllm_config.model_config,
                )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )

            if self.aux_hidden_trace.enabled:
                self.aux_hidden_trace.configure_model(self.model)
            elif self.use_aux_hidden_state_outputs:
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
                if self.use_pp:
                    assert self.speculative_config.method is not None
                    verify_supports_aux_hidden_states_over_pp(
                        self.model, self.speculative_config.method
                    )
                    assert self.pp_handler is not None
                    self.pp_handler.configure_aux_hidden_state_relay(self.model)
            if isinstance(self.speculator, DraftModelSpeculator):
                with use_workspace_lane(self._draft_workspace_lane):
                    self.speculator.load_model(self.model)
                    eplb_models_added = self.eplb.maybe_register_speculator(
                        self.speculator, self.speculative_config, load_dummy_weights
                    )
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        # Initialize the components that require the model.
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )

        self.decode_query_len = (
            self.num_speculative_steps
            + self.model_state.num_new_sampled_tokens_per_step
        )

        if self.parallel_config.enable_batch_sharded_sampling:
            if hasattr(self.model, "compute_logits_local"):
                self.batch_sharder = BatchSharder(
                    max_num_reqs=self.max_num_reqs,
                    max_num_logits_per_req=self.decode_query_len,
                    device=self.device,
                )
                logger.info("Batch-sharded sampling enabled.")
            else:
                logger.warning_once(
                    "Disabling batch-sharded sampling: %s does not implement "
                    "compute_logits_local",
                    type(self.model).__name__,
                )

        # Initialize samplers. Model states may override via custom_sampler().
        if self.is_last_pp_rank and not self.is_pooling_model:
            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
                enable_trace_replay=self.model_config.enable_trace_replay,
                reasoning_config=self.vllm_config.reasoning_config,
                return_sampling_mask=self.model_config.return_sampling_mask,
            )
            custom = self.model_state.custom_sampler(self.sampler)

            if custom:
                self.sampler, self.rejection_sampler = custom
            elif self.speculative_config is not None:
                self.rejection_sampler = RejectionSampler(
                    self.sampler,
                    self.speculative_config,
                    self.device,
                )
            self.prompt_logprobs_worker = PromptLogprobsWorker(
                self.max_num_reqs,
                logprobs_mode=self.model_config.logprobs_mode,
            )
            self.structured_outputs_worker = StructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * self.decode_query_len,
                vocab_size=self.vocab_size,
                device=self.device,
                mask_stride=self.decode_query_len,
                num_bonus_tokens=self.model_state.num_new_sampled_tokens_per_step,
            )

        if self.is_pooling_model and self.is_last_pp_rank:
            self.pooling_runner = PoolingRunner(self.model, self.vllm_config)
        eplb_models_added |= self.eplb.maybe_register_model(
            self.model,
            self.model_config,
            load_dummy_weights,
        )
        self.eplb.maybe_start_async_loop(eplb_models_added)

        if not self.is_first_pp_rank:
            # For non-first PP ranks, create intermediate tensors sized
            # for the max capture size so they can be sliced per batch.
            # Save as persistent member so runtime can copy received data
            # into the same addresses that the CUDA graphs captured.
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

        # Finalize offloaded storage only after model weights and any
        # post-loading transformations are complete.
        get_offloader().post_init()

    def get_model(self) -> nn.Module:
        return self.model

    def get_draft_model(self) -> nn.Module | None:
        speculator = self.speculator
        if not isinstance(speculator, DraftModelSpeculator):
            return None
        return speculator.model

    def reload_weights(self, *args, **kwargs) -> None:
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.reload_weights(self, *args, **kwargs)  # type: ignore[arg-type]

    def update_config(self, *args, **kwargs) -> None:
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.update_config(self, *args, **kwargs)  # type: ignore[arg-type]

        # v2 reads config via self.vllm_config (e.g. in load_model), so keep it
        # in sync with the attributes the v1 helper just replaced.
        self.vllm_config.model_config = self.model_config
        self.vllm_config.load_config = self.load_config

    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream:
        # Cache the default CUDA stream to avoid lookup overhead.
        return torch.cuda.current_stream(self.device)

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        encoder_runner = getattr(self.model_state, "encoder_runner", None)
        if encoder_runner is None:
            return {}
        return encoder_runner.get_encoder_timing_stats()

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.vllm_config)

    @staticmethod
    def _attn_config_signature(kv_cache_config: KVCacheConfig) -> tuple[Any, ...]:
        return tuple(
            (tuple(group.layer_names), group.kv_cache_spec)
            for group in kv_cache_config.kv_cache_groups
        )

    @torch.inference_mode()
    def prepare_static_attn_owners_for_kv_sizing(self) -> int:
        """Retain V2 attention builders before elastic KV publication."""
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
        )

        if getattr(self, "attn_groups", None):
            raise RuntimeError("profiling attention state was not released")
        kv_cache_groups = get_kv_cache_groups(
            self.vllm_config, self.get_kv_cache_spec()
        )
        saved_override = self.cache_config.num_gpu_blocks_override
        try:
            self.cache_config.num_gpu_blocks_override = 1
            sizing_config = get_kv_cache_config_from_groups(
                self.vllm_config, kv_cache_groups, available_memory=0
            )
        finally:
            self.cache_config.num_gpu_blocks_override = saved_override

        config_signature = self._attn_config_signature(sizing_config)
        allocated_before = torch.accelerator.memory_allocated()
        with set_current_vllm_config(self.vllm_config):
            groups, _, kernel_block_sizes = init_attn_backend(
                sizing_config,
                self.vllm_config,
                self.device,
            )
        self.attn_groups = groups
        self._prepared_attn_groups = groups
        self._prepared_attn_config_signature = config_signature
        self._prepared_kernel_block_sizes = tuple(kernel_block_sizes)
        torch.accelerator.synchronize()
        gc.collect()
        torch.accelerator.empty_cache()
        allocated_bytes = max(
            torch.accelerator.memory_allocated() - allocated_before, 0
        )
        logger.info(
            "Prepared V2 production attention owners before KV sizing: "
            "%d groups, %.2f MiB newly allocated",
            sum(map(len, groups)),
            allocated_bytes / (1 << 20),
        )
        return allocated_bytes

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        # GPUWorker finalizes the PD interleave before KV cache initialization.
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            # Cross-attention block tables need to index encoder tokens, which
            # can exceed the decoder's max_model_len.
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes = []
        max_num_blocks_per_group = []
        slot_mapping_enabled = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            layer_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            slot_mapping_enabled.append(not isinstance(layer_spec, CircularBufferSpec))
            # Let each cache type account for CP. Attention KV is DCP-sharded,
            # while Mamba/GDN recurrent state is replicated across DCP ranks.
            max_num_blocks = spec.max_num_blocks_per_req(
                self.vllm_config, block_table_max_model_len
            )
            # Preserve each cache type's alignment requirements after applying
            # its topology-aware block-table width.
            if isinstance(layer_spec, (MambaSpec, CircularBufferSpec)):
                max_num_blocks = get_block_table_width(
                    max_num_blocks, spec.block_size, token_alignment=None
                )
            else:
                max_num_blocks = get_block_table_width(max_num_blocks, spec.block_size)
            max_num_blocks_per_group.append(max_num_blocks)

        target_attn_layer_names = None
        if isinstance(self.speculator, DraftModelSpeculator):
            # Adaptive verification validates target attention separately.
            target_attn_layer_names = {
                layer_name
                for group in self.kv_cache_config.kv_cache_groups
                for layer_name in group.layer_names
            } - self.speculator.draft_attn_layer_names
        reuse_prepared_attn = (
            not is_profiling and self._prepared_attn_groups is not None
        )
        if reuse_prepared_attn:
            if (
                self._attn_config_signature(self.kv_cache_config)
                != self._prepared_attn_config_signature
            ):
                raise RuntimeError(
                    "KV configuration changed V2 attention ownership after "
                    "capacity was published; refusing an unaccounted rebuild"
                )
            self.attn_groups = self._prepared_attn_groups
            assert self._prepared_kernel_block_sizes is not None
            add_kv_sharing_layers_to_config(self.kv_cache_config, self.vllm_config)
            self.kernel_block_sizes = prepare_kernel_block_sizes(
                self.kv_cache_config, self.attn_groups
            )
            if tuple(self.kernel_block_sizes) != self._prepared_kernel_block_sizes:
                raise RuntimeError(
                    "KV configuration changed V2 kernel block geometry after "
                    "capacity was published; refusing an unaccounted rebuild"
                )
            attn_cg_support = get_attn_cg_support(self.attn_groups, self.vllm_config)
            self._prepared_attn_groups = None
            self._prepared_attn_config_signature = None
            self._prepared_kernel_block_sizes = None
        else:
            (
                self.attn_groups,
                attn_cg_support,
                self.kernel_block_sizes,
            ) = init_attn_backend(
                self.kv_cache_config,
                self.vllm_config,
                self.device,
            )
        additional_attn_cg_support = self.model_state.get_additional_cg_support()
        attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)
        # The speculator clears the flag at load time when the checkpoint has
        # no confidence head, so it holds the effective value.
        self.adaptive_verification = maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=getattr(
                self.speculator, "enable_adaptive_verification", False
            ),
            attn_groups=self.attn_groups,
            attn_cg_support=attn_cg_support,
            req_states=self.req_states,
            query_start_loc=self.input_buffers.query_start_loc,
            num_bonus_tokens=self.model_state.num_new_sampled_tokens_per_step,
            max_total_logits=get_max_chunk_logits(self.vocab_size),
            vllm_config=self.vllm_config,
            target_layer_names=target_attn_layer_names,
            additional_attn_cg_support=additional_attn_cg_support,
        )

        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            slot_mapping_enabled=slot_mapping_enabled,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        self.pcp_manager = pcp.maybe_build_pcp_manager(
            self.vllm_config,
            self.device,
            self.supports_mm_inputs,
            self.req_states,
            self.block_tables,
            cls=self.pcp_manager_cls,
        )
        self.ubatch_runner = maybe_build_ubatch_runner(
            self.vllm_config,
            self.device,
            self.model_state,
            self.attn_groups,
            self.kv_cache_config,
            self.max_num_reqs,
        )
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config,
            self.kv_cache_config,
            use_replayssm=self.vllm_config.cache_config.use_replayssm,
        )
        if self.adaptive_verification is not None:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        full_decode_query_lens: set[int] | None = None
        cg_validation_query_len = self.decode_query_len
        speculative_config = self.vllm_config.speculative_config
        if (
            speculative_config is not None
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            schedule = speculative_config.num_speculative_tokens_per_batch_size
            assert schedule is not None
            num_new_sampled_tokens = (
                self.decode_query_len - self.vllm_config.num_speculative_tokens
            )
            scheduled_query_lens = {
                num_spec_tokens + num_new_sampled_tokens
                for _, _, num_spec_tokens in schedule
            }
            if (
                attn_cg_support.min_cg_support
                == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
            ):
                # Always preserve the baseline non-speculative q_len=1 FULL
                # lane; phase policy can select K=0 even when the configured
                # batch-size schedule contains only K>0 entries.
                full_decode_query_lens = {1}
                cg_validation_query_len = 1
                logger.info(
                    "Dynamic speculative decoding will use FULL CUDA graphs "
                    "for query_len=1 and PIECEWISE for query lengths %s due "
                    "to %s support.",
                    sorted(scheduled_query_lens - {1}),
                    attn_cg_support.min_cg_attn_backend,
                )
            elif (
                os.environ.get("AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH", "0") == "1"
            ):
                full_decode_query_lens = {1, *scheduled_query_lens}
                cg_validation_query_len = max(full_decode_query_lens)
                logger.warning(
                    "Research-only FlashInfer DCP prefill CUDA graphs are "
                    "enabled for dynamic decode query lengths %s.",
                    sorted(full_decode_query_lens),
                )
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            cg_validation_query_len,
            use_v2_model_runner=True,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=(
                self.kv_cache_config.effective_max_resident_seqs or self.max_num_reqs
            ),
        )
        self.cudagraph_manager = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
            lora_capture_cases=self.lora_capture_cases,
            varlen_decode=self.adaptive_verification is not None,
            full_decode_query_lens=full_decode_query_lens,
            full_decode_cap_query_lens=(
                {self.decode_query_len} if envs.AG2_VLLM_TP3_OWNER_PREQUANT else None
            ),
            tp3_sd_phase_reduce=envs.VLLM_TP3_SD_PHASE_REDUCE,
            tp3_owner_prequant=envs.AG2_VLLM_TP3_OWNER_PREQUANT,
            max_uniform_decode_reqs=(
                self.kv_cache_config.effective_max_resident_seqs or self.max_num_reqs
            ),
            owner="target",
        )
        check_attention_cp_compatibility(self.vllm_config)
        if isinstance(self.speculator, DraftModelSpeculator):
            # HACK(woosuk)
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            # After set_attn, so the speculator can size its cudagraph mode
            # to its own attention support.
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        # Profiling left equivalent block-table/input owners alive, and final
        # initialization may replace them. Drop superseded cached blocks before
        # the large KV backings are allocated so allocator history is not part
        # of the capacity contract.
        torch.accelerator.synchronize()
        gc.collect()
        torch.accelerator.empty_cache()
        logger.info(
            "V2 pre-KV physical free after final static initialization: %.2f MiB",
            torch.accelerator.get_memory_info()[0] / (1 << 20),
        )

        self.kv_caches: list[torch.Tensor] = []
        kv_caches_dict = init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_groups,
            self.device,
            self.cache_config.cache_dtype,
            self.kernel_block_sizes,
            self.vllm_config,
            self.elastic_kv_controller.backings,
            self.elastic_kv_controller.geometry,
        )
        self.kv_caches_for_block_copy = get_kv_caches_for_block_copy(
            self.kv_caches,
            kv_caches_dict,
            self.kv_cache_config,
        )
        if self.kv_caches_for_block_copy is not self.kv_caches:
            logger.info(
                "KV block CoW is isolated from separate Mamba/GDN backing "
                "(%d non-Mamba layer views).",
                len(self.kv_caches_for_block_copy),
            )
        if self.kv_cache_config.elastic_mapping_quantum:
            self.elastic_kv_controller.configure_physical_budget(
                self.kv_cache_config.elastic_budget_bytes,
                self.kv_cache_config.elastic_mapping_quantum,
                (
                    self.kv_cache_config.num_blocks,
                    self.kv_cache_config.elastic_gdn_initial_blocks,
                ),
            )
        if any(
            isinstance(group.kv_cache_spec, MambaSpec)
            and group.kv_cache_spec.separate_pool
            for group in self.kv_cache_config.kv_cache_groups
        ):
            self.gdn_checkpoint_manager = V2GDNCheckpointManager()
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            static_forward_context=self.compilation_config.static_forward_context,
            num_blocks=self.kv_cache_config.num_blocks,
        )

    @torch.inference_mode()
    @step_eplb_after(is_dummy=True)
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        context_len: int = 0,
        skip_eplb: bool = False,
        is_profile: bool = False,
        valid_dummy_state_slots: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if skip_attn and not is_profile:
            raise ValueError(
                "skip_attn must only be True for initial memory profiling."
            )

        # Create a dummy scheduler output.
        num_reqs = min(num_tokens, self.max_num_reqs)
        if uniform_decode:
            # HACK(lucas): for now since the worker is shared between MRV1 and MRV2,
            # and for spec-decode with MTP we want to make sure the dummy runs use
            # 1+num_speculative_tokens we use max here, this will likely be eventually
            # changed in the worker: https://github.com/vllm-project/vllm/pull/35243
            num_tokens = max(num_tokens, self.decode_query_len)
            num_reqs = num_tokens // self.decode_query_len
            assert num_tokens % self.decode_query_len == 0
        # Distribute the remainder evenly so no dummy request exceeds
        # ceil(num_tokens / num_reqs) <= max_model_len tokens.
        num_tokens_per_request = [
            num_tokens // num_reqs + (i >= num_reqs - num_tokens % num_reqs)
            for i in range(num_reqs)
        ]

        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens
        dummy_scheduler_output.num_spec_tokens_to_schedule = self.num_speculative_steps

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Get the intermediate tensors for the dummy run.
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            assert self.intermediate_tensors is not None
            intermediate_tensors = self.intermediate_tensors[:num_tokens]

        max_loras = self.lora_config.max_loras if self.lora_config is not None else 0
        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens=np.array(num_tokens_per_request, dtype=np.int32),
            num_sampled_tokens=None,
            remove_lora=True,
            num_active_loras=max_loras,
        ):
            # Execute the model.
            self.execute_model(
                dummy_scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                is_profile=is_profile,
                context_len=context_len,
                valid_dummy_state_slots=valid_dummy_state_slots,
            )
        self.kv_connector.set_disabled(False)

        # Non-last PP ranks don't produce output for sampling.
        if not self.is_last_pp_rank:
            return None, None

        assert self.execute_model_state is not None
        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        dp_sync = self.execute_model_state.dp_sync
        self.execute_model_state = None

        self.step_timing.forward_end()

        # dummy run the eagle speculator's propose to ensure DP/EP sync.
        if self.speculator is not None:
            assert self.sampler is not None
            self.step_timing.drafter_start()
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
            if self.speculator.supports_mm_inputs:
                mm_inputs = (
                    [],
                    torch.zeros(
                        input_batch.num_tokens,
                        dtype=torch.bool,
                        device="cpu",
                    ),
                )

            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            with use_workspace_lane(self._draft_workspace_lane):
                self.speculator.propose(
                    input_batch=input_batch,
                    attn_metadata=attn_metadata,
                    slot_mappings=slot_mappings_by_layer,
                    last_hidden_states=spec_hidden_states,
                    aux_hidden_states=aux_hidden_states,
                    num_sampled=torch.ones(
                        input_batch.num_reqs, dtype=torch.int32, device=self.device
                    ),
                    num_rejected=torch.zeros(
                        input_batch.num_reqs, dtype=torch.int32, device=self.device
                    ),
                    last_sampled=self.req_states.last_sampled_tokens,
                    next_prefill_tokens=self.req_states.next_prefill_tokens,
                    temperature=self.sampler.sampling_states.temperature.gpu,
                    seeds=self.sampler.sampling_states.seeds.gpu,
                    dp_sync=dp_sync,
                    dummy_run=True,
                    skip_attn_for_dummy_run=skip_attn,
                    mm_inputs=mm_inputs,
                    is_profile=is_profile,
                )
            self.step_timing.drafter_end()

        assert hidden_states is not None  # Last PP rank always has hidden_states
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        dummy_input_batch = InputBatch.make_dummy(
            num_reqs, num_reqs, self.input_buffers
        )

        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        assert self.sampler is not None
        self.sampler(logits, dummy_input_batch)

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        assert self.pooling_runner is not None
        self.pooling_runner.dummy_pooler_run(hidden_states)

    @torch.inference_mode()
    def profile_run(self) -> None:
        if self.supports_mm_inputs and self.is_first_pp_rank:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and not mm_config.skip_mm_profiling:
                mm_budget = MultiModalBudget(
                    self.vllm_config,
                    self.mm_registry,
                    enable_cache=False,
                )
                dummy_mm_inputs = get_dummy_encoder_profile_inputs(
                    self.mm_registry,
                    mm_budget,
                )
                self.model_state.encoder_runner.profile_encoder_cache(
                    dummy_mm_inputs, mm_budget
                )

        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, is_profile=True
        )

        # Only run sampler/pooler on last PP rank (non-last ranks return None).
        if self.is_last_pp_rank:
            assert sample_hidden_states is not None
            if self.pooling_runner is None:
                self._dummy_sampler_run(sample_hidden_states)
            else:
                self._dummy_pooler_run(hidden_states)

        torch.accelerator.synchronize()
        del hidden_states, sample_hidden_states
        self.reset_encoder_cache()
        gc.collect()

    def reset_mm_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()
        if self.pooling_runner is not None:
            self.pooling_runner.clear()

    @torch.inference_mode()
    def profile_cudagraph_memory(self) -> int:
        """Estimate the GPU memory required to capture CUDA graphs."""
        return _profile_cudagraph_memory(self)

    @torch.inference_mode()
    def capture_model(self, *, profile_only: bool = False) -> int:
        assert self.cudagraph_manager is not None
        capture_encoder = (
            self.model_state.supports_mm_inputs
            and self.model_state.encoder_runner.has_cudagraph()
        )
        capture_decoder = self.cudagraph_manager.needs_capture()
        if not capture_encoder and not capture_decoder:
            if self.cudagraph_manager.defer_startup_graphs:
                logger.info(
                    "Startup CUDA Graph capture is deferred; exact runtime "
                    "descriptors will capture under same-step KV loans"
                )
            else:
                logger.warning(
                    "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                    "ensure `cudagraph_mode` was not manually set to `NONE`"
                )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        with freeze_gc_for_cudagraph_capture():
            torch.accelerator.empty_cache()
            start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

            with self.maybe_setup_dummy_loras(self.lora_config):
                if capture_encoder:
                    self.model_state.encoder_runner.capture()

                if capture_decoder:
                    input_buffers = self.input_buffers
                    if self.pcp_manager is not None:
                        input_buffers = self.pcp_manager.input_buffers
                    self.cudagraph_manager.capture(
                        self.model,
                        self.model_state,
                        input_buffers,
                        self.intermediate_tensors,
                        self.block_tables,
                        self.attn_groups,
                        self.kv_cache_config,
                        pcp_manager=self.pcp_manager,
                        has_lora=self.lora_config is not None,
                        use_aux_hidden_state_outputs=self.use_aux_hidden_state_outputs,
                        lora_capture_hook=create_lora_capture_hook(
                            self.lora_config, self
                        ),
                    )
                    if self.speculator is not None:
                        with use_workspace_lane(self._draft_workspace_lane):
                            self.speculator.capture()
                    if self.adaptive_verification is not None:
                        with self.step_timing.collect() as timings:
                            for batch in self.adaptive_verification.batches_to_profile(
                                self.cudagraph_manager.captured_token_counts()
                            ):
                                self._dummy_run(**batch)
                        self.adaptive_verification.set_initial_cost_curves(timings)

            end_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        if not profile_only:
            # Lock workspace to prevent resizing during execution. A resize after
            # capture frees the static cuda graph buffer.
            lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def _remove_request(self, req_id: str) -> bool:
        # Call model_state.remove_request *before* req_states.remove_request
        # so the model_state can still look up the slot index.
        self.model_state.remove_request(req_id)
        req_idx = self.req_states.remove_request(req_id)
        if req_idx is None:
            return False
        if self.pooling_runner is not None:
            self.pooling_runner.remove_request(req_idx)
        if self.pp_handler is not None:
            self.pp_handler.on_req_idx_freed(req_idx)
        if self.encoder_cache is not None:
            self.encoder_cache.remove_request(req_id)
        if self.prompt_logprobs_worker is not None:
            self.prompt_logprobs_worker.remove_request(req_id)
        self.lora_state.remove_request(req_id)
        return True

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        if self.pooling_runner is not None:
            # Preempted docs keep their query-use reservation until rescheduled.
            self.pooling_runner.on_requests_finished(finished_req_ids)
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        # Sorted so every TP rank frees request slots in the same order.
        # Features like batch-sharded sampling derive rank request ownership
        # from the slot index.
        for req_id in sorted(finished_req_ids):
            self._remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.free_encoder_cache(mm_hash)

    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs(
                self.req_states.draft_tokens
            )
            if outputs is not None:
                self.postprocess_sampled(**outputs)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            # Streaming input update: request already exists from a prior
            # chunk. Remove old state so it can be cleanly re-added below
            # with the updated prompt_token_ids and mm_features.
            self._remove_request(req_id)

            prompt_len = length_from_prompt_token_ids_or_embeds(
                new_req_data.prompt_token_ids,
                new_req_data.prompt_embeds,
            )
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
                execution_prefill_len=new_req_data.execution_prefill_len,
            )
            req_index = self.req_states.req_id_to_index[req_id]
            if self.adaptive_verification is not None:
                self.adaptive_verification.add_request(req_index)

            if self.pooling_runner is not None:
                assert new_req_data.pooling_params is not None
                assert new_req_data.prompt_token_ids is not None
                self.pooling_runner.add_request(
                    req_id,
                    req_index,
                    new_req_data.pooling_params,
                    new_req_data.prompt_token_ids,
                )

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
                assert self.sampler is not None
                self.sampler.add_request(
                    req_index,
                    prompt_len,
                    new_req_data.sampling_params,
                )
                assert self.prompt_logprobs_worker is not None
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.model_state.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # Update CPU num_computed_prefill_tokens.
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Apply copy-on-write block copies for partial prefix-cache hits, after
        # zeroing new blocks and before the forward pass reads them.
        if scheduler_output.kv_cache_block_copies:
            copy_kv_cache_blocks_inplace(
                self.kv_caches_for_block_copy,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

    def gather_batch_req_state(
        self, scheduler_output: SchedulerOutput, dummy_run: bool
    ) -> tuple["BatchReqState | None", int | None]:
        """Gather CPU request state in the execution order selected by Exp22."""
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())

        if dummy_run:
            # Dummy batches are uniform by construction.
            return None, get_uniform_decode_token_count(
                num_reqs, num_toks, max_query_len, has_prefill=False
            )

        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        # batch_idx -> req_id. Shape alone cannot distinguish a short chunked
        # prefill from target decode/spec verification. Keep every request in
        # its existing stable shape order, but place completed-prefill rows
        # before prompt-prefill rows so attention backends receive the semantic
        # decode -> prefill lifecycle layout their split metadata describes.
        computed_prefill_tokens = self.req_states.num_computed_prefill_tokens
        prefill_lengths = self.req_states.prefill_len.np
        is_prefilling_by_req = {
            req_id: bool(
                computed_prefill_tokens[self.req_states.req_id_to_index[req_id]]
                < prefill_lengths[self.req_states.req_id_to_index[req_id]]
            )
            for req_id in num_tokens_per_req
        }
        req_ids = sort_batch_req_ids(
            num_tokens_per_req,
            self.decode_query_len,
            is_prefilling_by_req=is_prefilling_by_req,
        )
        numtoks_iter = map(num_tokens_per_req.__getitem__, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.__getitem__, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.intp, count=num_reqs)
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[
            idx_mapping_np
        ]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np

        if self.adaptive_verification is not None and draft_tokens:
            num_toks = self.adaptive_verification.get_num_tokens(
                num_tokens_per_req, draft_tokens
            )

        batch_state = BatchReqState(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_toks,
            idx_mapping_np=idx_mapping_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            is_prefilling_np=is_prefilling_np,
            has_prefill=bool(is_prefilling_np.any()),
        )
        return batch_state, get_uniform_decode_token_count(
            num_reqs, num_toks, max_query_len, batch_state.has_prefill
        )

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: "BatchReqState",
        batch_desc: BatchExecutionDescriptor,
    ) -> InputBatch:
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
        assert num_tokens > 0
        # Padding is part of the consumed correctness contract, not only a MoE
        # optimization hint. Every captured consumer sees the same live/tail
        # mask; kernels that support skip-padding may additionally avoid work.
        is_padding = self.input_buffers.is_padding
        is_padding[:num_tokens].fill_(False)
        is_padding[num_tokens:num_tokens_after_padding].fill_(True)

        req_ids = batch_req_state.req_ids
        num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
        num_reqs = len(req_ids)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
            num_logits = num_draft_tokens_per_req + num_bonus_tokens
            # combine_sampled_and_draft_tokens places a request's logits rows
            # at [query_end - num_logits, query_end). Fewer query rows than
            # that would silently select the preceding request's hidden states.
            assert (num_scheduled_tokens_np >= num_logits).all()
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

        adaptive_verification = (
            self.adaptive_verification if num_draft_tokens_per_req is not None else None
        )
        num_scheduled_tokens_upper_bound = num_scheduled_tokens_np
        if adaptive_verification is not None:
            # num_scheduled_tokens represents the draft budget evenly distributed across
            # all verification requests, `reallocate_drafts` will unevenly assign the
            # draft budget to requests on the GPU side only.
            num_scheduled_tokens_np, cu_num_logits_np = (
                adaptive_verification.compact_batch(
                    num_draft_tokens_per_req,
                    num_scheduled_tokens_np,
                    cu_num_logits_np,
                )
            )

        # Get query_start_loc.
        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)
        # PIECEWISE descriptors omit logical num_reqs, but elastic decode
        # executables capture metadata at a physical request bucket. Repeating
        # the terminal query offset creates zero-token dummy request slots.
        bounded_short_decode = bool(
            scheduler_output.is_pure_decode_step
            and scheduler_output.num_spec_tokens_to_schedule > 0
            and num_tokens == num_reqs * self.decode_query_len
        )
        num_reqs_padded = (
            batch_desc.num_reqs
            or (batch_desc.physical_num_reqs if bounded_short_decode else None)
            or num_reqs
        )
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        query_start_loc = self.input_buffers.query_start_loc
        async_copy_to_gpu(query_start_loc_np, out=query_start_loc)
        if adaptive_verification is not None:
            cu_num_logits, query_start_loc, total_num_draft_tokens = (
                adaptive_verification.reallocate_drafts(req_ids, idx_mapping)
            )
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
        if draft_tokens:
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, self.decode_query_len
            )
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = query_start_loc[: num_reqs_padded + 1]
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens
        num_computed_prefill_tokens_np = computed_prefill_tokens_np[idx_mapping_np]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np
        if envs.AG2_VLLM_NVFP4_MARLIN_ISOLATE_PREFILL:
            prefill_indices = np.flatnonzero(is_prefilling_np)
            num_decodes = int(prefill_indices[0]) if prefill_indices.size else num_reqs
            if np.any(is_prefilling_np[:num_decodes]) or np.any(
                ~is_prefilling_np[num_decodes:]
            ):
                raise RuntimeError(
                    "NVFP4 Marlin prefill isolation requires decode requests "
                    "before prefill requests"
                )
            layout = self.input_buffers.marlin_request_layout_cpu
            layout[0] = num_reqs
            layout[1] = num_decodes
            layout[2 : num_reqs + 3].copy_(
                torch.from_numpy(query_start_loc_np[: num_reqs + 1])
            )

        # Get prefill tokens if any.
        if batch_req_state.has_prefill:
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

        dcp_local_seq_lens = None
        if self.use_dcp:
            # Prepare dcp local seq_lens.
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
            self.input_buffers.logits_indices,
        )

        # CPU upper bound on seq_lens; padded entries left at zero.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens_upper_bound,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            # prompt_lens is only used in R-SWA case.
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        input_batch = InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens_upper_bound,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            marlin_request_layout_cpu=self.input_buffers.marlin_request_layout_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
            max_query_len=(
                int(num_scheduled_tokens_upper_bound.max())
                if adaptive_verification is not None
                else None
            ),
        )
        return pcp.maybe_partition_pcp_batch(
            self.pcp_manager,
            input_batch,
            padded_num_tokens=batch_desc.num_tokens,
        )

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if self.pcp_manager is not None:
            return self.pcp_manager.prepare_attn(input_batch)

        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].
        # Kernel pads beyond num_tokens with PAD_SLOT_ID.
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch, valid_state_slots: bool = False
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        if valid_state_slots:
            state_slots = torch.arange(
                1,
                input_batch.num_reqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            for block_table in block_tables:
                block_table[:, 0].copy_(state_slots)
        slot_mappings = pcp.maybe_get_pcp_dummy_slot_mappings(
            self.pcp_manager, self.block_tables, input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
        slot_mappings_by_layer: dict[str, torch.Tensor] | None = None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        global_input_batch = input_batch
        global_sample_hidden_states = hidden_states[input_batch.logits_indices]
        if self.speculator is not None and hasattr(
            self.speculator, "capture_target_lm_head_inputs"
        ):
            self.speculator.capture_target_lm_head_inputs(
                global_sample_hidden_states, input_batch
            )

        shard_metadata = None
        if self.batch_sharder is not None:
            input_batch, sorted_logits_indices, grammar_output, shard_metadata = (
                self.batch_sharder.shard_sampler_inputs(input_batch, grammar_output)
            )
            sample_hidden_states = hidden_states[sorted_logits_indices]
            local_logits = self.model.compute_logits_local(sample_hidden_states)
            logits = all_to_all_logits(local_logits, shard_metadata)
            logits = logits[:, : self.vocab_size]
            use_sparse_target_topk = False
        else:
            sample_hidden_states = global_sample_hidden_states
            use_sparse_target_topk = (
                input_batch.num_draft_tokens > 0
                and self.rejection_sampler is not None
                and self.target_boundary_capture is None
                and hasattr(self.model, "compute_local_logits")
                and self.rejection_sampler.can_use_sparse_target_topk(input_batch)
            )
            if use_sparse_target_topk:
                local_logits, vocab_start = self.model.compute_local_logits(
                    sample_hidden_states
                )
                if grammar_output is not None:
                    self.structured_outputs_worker.apply_grammar_bitmask(
                        local_logits,
                        input_batch,
                        grammar_output.structured_output_request_ids,
                        grammar_output.grammar_bitmask,
                        vocab_start=vocab_start,
                    )
                assert self.speculator is not None
                sampler_output = self.rejection_sampler.sample_sparse_target_topk(
                    local_logits,
                    vocab_start,
                    input_batch,
                    self.speculator.draft_logits,
                )
            else:
                logits = self.model.compute_logits(sample_hidden_states)
                if self.target_boundary_capture is not None:
                    self.target_boundary_capture.capture(
                        sample_hidden_states,
                        logits,
                        input_batch,
                        slot_mappings_by_layer,
                    )

        if grammar_output is not None and not use_sparse_target_topk:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        sampler_output: SamplerOutput | None
        if use_sparse_target_topk:
            pass
        elif input_batch.num_reqs == 0:
            # A sharded TP rank with no owned rows contributes padding to the
            # gather below.
            sampler_output = None
        elif input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        if shard_metadata is not None:
            # Gather the sharded sampler outputs from the TP ranks into a single
            # sampler output.
            assert self.sampler is not None
            sampler_output = gather_sampler_output(
                sampler_output,
                shard_metadata,
                device=self.device,
                global_batch=global_input_batch,
                local_batch=input_batch,
                gather_num_nans=self.sampler.compute_nans,
                logprobs_dims=self.sampler.get_logprobs_dims(
                    global_input_batch.idx_mapping_np,
                    # Rejection sampler does not return logprob token ids.
                    include_token_ids=(
                        global_input_batch.num_draft_tokens == 0
                        or self.rejection_sampler is None
                    ),
                ),
            )

        assert sampler_output is not None
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,  # May include -1 for masked entries
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        # Update the number of computed tokens.
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        else:
            output_bin_counts = None
        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

        self.model_state.postprocess_state(
            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu
        )

    def _merge_ec_connector_no_forward(
        self, scheduler_output: SchedulerOutput, output: ModelRunnerOutput
    ) -> ModelRunnerOutput:
        """Let the EC connector send/recv on a step with no work to run."""
        return ModelRunnerOutput.with_ec_conn_output(
            output,
            self.ec_connector.no_forward(scheduler_output).ec_connector_output,
        )

    def _commit_staged_mm_embeddings(
        self, embeddings: Any, input_batch: InputBatch
    ) -> torch.Tensor | None:
        """Commit staged embeddings and preserve upstream prompt embeds."""
        inputs_embeds = self.model_state.commit_staged_mm_embeddings(
            embeddings, input_batch
        )
        apply_prompt_embeddings = getattr(
            self.model_state, "apply_prompt_embeddings", None
        )
        if callable(apply_prompt_embeddings):
            assert inputs_embeds is not None
            inputs_embeds = apply_prompt_embeddings(
                input_batch, self.req_states, inputs_embeds
            )
        return inputs_embeds

    @torch.inference_mode()
    def _dynamic_graph_working_set(self) -> DynamicGraphWorkingSet:
        assert self.cudagraph_manager is not None
        managers = [self.cudagraph_manager]
        dynamic_managers = getattr(self.speculator, "dynamic_cudagraph_managers", None)
        if dynamic_managers is not None:
            managers.extend(dynamic_managers())
        manager_tuple = tuple(managers)
        working_set = getattr(self, "_elastic_dynamic_graph_working_set", None)
        if working_set is None:
            # The working set owns transaction state in addition to aggregating
            # managers.  In particular, staged hotset victims are armed before
            # model execution and retired only after sampling has consumed the
            # preceding executables.  Reconstructing this object at settlement
            # silently drops that pending retirement and lets worker residency
            # diverge from scheduler ownership.
            working_set = DynamicGraphWorkingSet(manager_tuple)
            self._elastic_dynamic_graph_working_set = working_set
        elif working_set.managers != manager_tuple:
            raise RuntimeError(
                "dynamic CUDA Graph manager set changed after working-set "
                "lifecycle initialization"
            )
        return working_set

    def _trim_dynamic_attention_cudagraph_state(
        self, working_set: DynamicGraphWorkingSet
    ) -> tuple[int, int]:
        """Release FlashInfer shape wrappers with no remaining HOT Graph."""
        attn_group_sets = [getattr(self, "attn_groups", ())]
        speculator_attn_groups = getattr(
            getattr(self, "speculator", None), "attn_groups", None
        )
        if speculator_attn_groups is not None:
            attn_group_sets.append(speculator_attn_groups)

        builders: list[Any] = []
        seen: set[int] = set()
        for attn_groups in attn_group_sets:
            for groups in attn_groups:
                for group in groups:
                    builder = group.get_metadata_builder(0)
                    trim = getattr(builder, "trim_dynamic_cudagraph_wrappers", None)
                    if trim is None or id(builder) in seen:
                        continue
                    seen.add(id(builder))
                    builders.append(builder)
        if not builders:
            return 0, 0

        keep_request_batch_sizes = working_set.hot_request_counts
        keep_token_batch_sizes = working_set.hot_token_counts
        # Graph managers have already removed future consumers. Complete the
        # last possible replay before dropping FlashInfer wrapper generations;
        # their native metadata addresses are embedded in captured kernels.
        torch.cuda.synchronize(self.device)
        removed = sum(
            builder.trim_dynamic_cudagraph_wrappers(
                keep_request_batch_sizes=keep_request_batch_sizes,
                keep_token_batch_sizes=keep_token_batch_sizes,
            )
            for builder in builders
        )
        reclaimed = 0
        if removed:
            free_before = torch.accelerator.get_memory_info()[0]
            gc.collect()
            torch.accelerator.empty_cache()
            torch.cuda.synchronize(self.device)
            reclaimed = max(0, torch.accelerator.get_memory_info()[0] - free_before)
            logger.info(
                "Dynamic FlashInfer CUDA Graph metadata trimmed: "
                "removed_wrappers=%d keep_request_batch_sizes=%s "
                "keep_token_batch_sizes=%s "
                "local_reclaimed_bytes=%d",
                removed,
                sorted(keep_request_batch_sizes),
                sorted(keep_token_batch_sizes),
                reclaimed,
            )
        return removed, reclaimed

    def _elastic_active_allocator_snapshot(
        self,
    ) -> dict[int, tuple[int, int, str, str]]:
        """Describe active CUDA blocks without retaining allocator objects."""
        active: dict[int, tuple[int, int, str, str]] = {}
        for segment in torch.cuda.memory_snapshot():
            address = int(segment["address"])
            offset = 0
            pool = repr(segment.get("segment_pool_id"))
            for block in segment["blocks"]:
                size = int(block["size"])
                block_address = int(block.get("address", address + offset))
                offset += size
                if block.get("state") != "active_allocated":
                    continue
                frames = block.get("frames") or ()
                frame = ""
                if frames:
                    top = frames[0]
                    frame = (
                        f"{top.get('filename', '')}:"
                        f"{top.get('line', '')}:"
                        f"{top.get('name', '')}"
                    )
                active[block_address] = (
                    size,
                    int(block.get("requested_size", size)),
                    pool,
                    frame,
                )
        return active

    def _elastic_python_cuda_storage_snapshot(
        self,
    ) -> dict[int, tuple[int, str, tuple[int, ...]]]:
        """Describe Python-visible CUDA storages without keeping tensor refs."""
        storages: dict[int, tuple[int, str, tuple[int, ...]]] = {}
        device_index = self.device.index
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor) or obj.device.type != "cuda":
                    continue
                if device_index is not None and obj.device.index != device_index:
                    continue
                storage = obj.untyped_storage()
                address = int(storage.data_ptr())
                if not address:
                    continue
                nbytes = int(storage.nbytes())
                current = storages.get(address)
                if current is None or nbytes > current[0]:
                    storages[address] = (
                        nbytes,
                        str(obj.dtype),
                        tuple(int(dim) for dim in obj.shape),
                    )
            except Exception:
                # GC can expose partially destructed tensor subclasses. They
                # are not safe diagnostic owners and must not break X0.
                continue
        return storages

    def _elastic_cublas_workspace_addresses(
        self,
        active: dict[int, tuple[int, int, str, str]] | None = None,
    ) -> frozenset[int]:
        """Return active classic cuBLAS workspace allocation addresses."""
        workspace_size = int(torch._C._cuda_getCublasWorkspaceSize())
        if active is None:
            active = self._elastic_active_allocator_snapshot()
        return frozenset(
            address
            for address, (_, requested, pool, _) in active.items()
            if requested == workspace_size and pool == "(0, 0)"
        )

    def _measure_elastic_cublas_workspace_bytes(self) -> int:
        """Measure classic cuBLAS workspaces created after the X0 baseline."""
        baseline = getattr(self, "_elastic_cublas_workspace_baseline", None)
        if baseline is None:
            raise RuntimeError(
                "elastic cuBLAS workspace baseline was not captured before "
                "the dynamic CUDA Graph step"
            )
        active = self._elastic_active_allocator_snapshot()
        addresses = self._elastic_cublas_workspace_addresses(active)
        return sum(
            active[address][0] for address in addresses if address not in baseline
        )

    @staticmethod
    def _set_elastic_cublas_workspace_unit(
        model_runner_output: ModelRunnerOutput,
    ) -> None:
        model_runner_output.elastic_cublas_workspace_unit_bytes = int(
            torch._C._cuda_getCublasWorkspaceSize()
        )

    def _publish_elastic_residency_receipt(
        self,
        model_runner_output: ModelRunnerOutput,
        transaction_id: str | None,
    ) -> None:
        model_runner_output.elastic_residency_receipt = (
            self._dynamic_graph_working_set().residency_receipt(
                transaction_id=transaction_id,
                resident_bytes=model_runner_output.elastic_external_memory_bytes,
                floor_bytes=model_runner_output.elastic_external_memory_floor_bytes,
                transition_floor_bytes=(
                    model_runner_output.elastic_external_memory_transition_floor_bytes
                ),
                peak_bytes=model_runner_output.elastic_external_memory_peak_bytes,
                cublas_workspace_bytes=(
                    model_runner_output.elastic_cublas_workspace_unit_bytes
                ),
            )
        )

    def _clear_elastic_cublas_workspaces(
        self, working_set: DynamicGraphWorkingSet
    ) -> int:
        """Release stream workspaces only after the old Graph set is empty."""
        if working_set.active_graph_bytes:
            raise RuntimeError(
                "cannot clear cuBLAS workspaces while dynamic CUDA Graphs are HOT"
            )
        torch.cuda.synchronize(self.device)
        free_before = torch.accelerator.get_memory_info()[0]
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.accelerator.empty_cache()
        torch.cuda.synchronize(self.device)
        # All classic cuBLAS workspaces were explicitly released.  Rebuild the
        # baseline from surviving allocations: a non-cuBLAS block may happen
        # to have the same requested size, while freed workspace addresses are
        # absent and will still be charged if the allocator later reuses them.
        self._elastic_cublas_workspace_baseline = (
            self._elastic_cublas_workspace_addresses()
        )
        reclaimed = max(0, torch.accelerator.get_memory_info()[0] - free_before)
        if reclaimed:
            logger.info(
                "Dynamic cuBLAS workspaces cleared after Graph eviction: "
                "local_reclaimed_bytes=%d",
                reclaimed,
            )
        return reclaimed

    def _record_elastic_x0_allocation_delta(self, *, dynamic_wave: bool) -> None:
        active = self._elastic_active_allocator_snapshot()
        storages = self._elastic_python_cuda_storage_snapshot()
        baseline_active = getattr(self, "_elastic_x0_active_baseline", None)
        baseline_storages = getattr(self, "_elastic_x0_storage_baseline", None)
        if baseline_active is None or baseline_storages is None:
            self._elastic_x0_active_baseline = frozenset(active)
            self._elastic_x0_storage_baseline = frozenset(storages)
            return
        if not dynamic_wave:
            return

        new_active = [
            (address, *metadata)
            for address, metadata in active.items()
            if address not in baseline_active
        ]
        new_storages = [
            (address, *metadata)
            for address, metadata in storages.items()
            if address not in baseline_storages
        ]
        new_active.sort(key=lambda item: item[1], reverse=True)
        new_storages.sort(key=lambda item: item[1], reverse=True)
        logger.warning(
            "Elastic X0 allocation-owner receipt: new_active_bytes=%d "
            "new_active_blocks=%d top_active=%s new_python_storage_bytes=%d "
            "new_python_storages=%d top_python=%s",
            sum(item[1] for item in new_active),
            len(new_active),
            new_active[:16],
            sum(item[1] for item in new_storages),
            len(new_storages),
            new_storages[:16],
        )

    def _finish_dynamic_graph_step(
        self,
        *,
        release_idle_cache: bool = False,
        transaction_id: str | None = None,
        step_plan: ElasticStepPlan | None = None,
    ) -> tuple[int, int, int]:
        working_set = self._dynamic_graph_working_set()
        if not getattr(self, "_elastic_step_measurement_active", False):
            working_set.finish_step()
            if transaction_id is not None:
                working_set.release_leases(transaction_id)
            return getattr(
                self,
                "_elastic_cached_graph_receipt",
                (working_set.resident_bytes, 0, 0),
            )
        active_graph_bytes_before = working_set.active_graph_bytes
        if release_idle_cache:
            working_set.finish_idle_step()
        else:
            working_set.finish_step()
        if transaction_id is not None:
            working_set.release_leases(transaction_id)
        # FlashInfer CUDA-Graph wrappers own address-stable planning tensors
        # that are shared across repeated replays of the live cohort.  A
        # successor's first synchronized replay is not a wrapper-lifetime
        # boundary: X32 survived that replay and faulted only after the X16
        # wrapper was trimmed.  Retain stale wrappers while any product wave
        # is active; the explicit X0 path evicts unpinned graphs first and is
        # the proven safe reclamation boundary.
        trimmed_wrappers = 0
        wrapper_reclaimed = 0
        if release_idle_cache:
            trimmed_wrappers, wrapper_reclaimed = (
                self._trim_dynamic_attention_cudagraph_state(working_set)
            )
        working_set.reconcile_retained_cleanup(wrapper_reclaimed)
        if working_set.active_graph_bytes == 0:
            self._clear_elastic_cublas_workspaces(working_set)
        cublas_workspace_bytes = self._measure_elastic_cublas_workspace_bytes()
        if active_graph_bytes_before > 0 or trimmed_wrappers > 0:
            self._elastic_dynamic_wave_observed = True
        measured_external = working_set.resident_bytes + cublas_workspace_bytes
        # A non-idle completion only measures what survived the step. Returning
        # the transient portion here makes the allocator oscillate even when
        # the next descriptor is identical. X0 is different: it is an explicit
        # next-step decision and may return everything immediately.
        retained_transition_floor = getattr(
            self, "_elastic_retained_transition_floor_bytes", 0
        )
        if release_idle_cache:
            retained_transition_floor = 0
        if release_idle_cache:
            # X0 has no live Graph/cuBLAS owner. The retention ledger is a
            # conservative teardown estimate, not a new allocation request.
            # Ask VMM for zero and let its rank-safe driver-free preflight
            # return the actual physical floor. Keeping the ledger in
            # ``measured_external`` here caused a 0 <-> one-quantum loop.
            effective_external = self.elastic_kv_controller.reconcile_external_memory(
                working_set.active_graph_bytes + cublas_workspace_bytes
            )
            working_set.clear_idle_retention_after_physical_reconcile()
        else:
            effective_external = measured_external + retained_transition_floor
        physical_floor = max(
            retained_transition_floor,
            effective_external
            - working_set.active_graph_bytes
            - cublas_workspace_bytes,
            0,
        )
        transition_floor_upper_bound = max(
            physical_floor,
            0
            if release_idle_cache
            else working_set.transition_floor_upper_bound_bytes(),
        )
        baseline_reserved = getattr(self, "_elastic_step_baseline_reserved_bytes", None)
        baseline_driver_free = getattr(
            self, "_elastic_step_baseline_driver_free_bytes", None
        )
        if baseline_reserved is None or baseline_driver_free is None:
            local_peak_external = effective_external
        else:
            transient_peak = max(
                0,
                torch.cuda.max_memory_reserved(self.device) - baseline_reserved,
                baseline_driver_free - torch.cuda.mem_get_info(self.device)[0],
            )
            mm_overlap_peak = getattr(self, "_elastic_step_mm_overlap_peak_bytes", 0)
            local_peak_external = max(
                effective_external,
                transient_peak,
                mm_overlap_peak,
            )
        peak = torch.tensor(
            local_peak_external,
            dtype=torch.int64,
            device=self.device,
        )
        if self.vllm_config.parallel_config.tensor_parallel_size > 1:
            torch.distributed.all_reduce(
                peak,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
        self._elastic_last_step_peak_external_bytes = int(peak.item())
        self._elastic_step_baseline_reserved_bytes = None
        self._elastic_step_baseline_driver_free_bytes = None
        self._elastic_step_mm_overlap_peak_bytes = 0
        self._elastic_step_baseline_external_bytes = None
        if release_idle_cache:
            dynamic_wave = getattr(self, "_elastic_dynamic_wave_observed", False)
            previous_floor = getattr(self, "_elastic_idle_logged_floor_bytes", None)
            if (
                previous_floor != physical_floor
                or active_graph_bytes_before > 0
                or dynamic_wave
            ):
                allocated = torch.cuda.memory_allocated(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                driver_free, driver_total = torch.cuda.mem_get_info(self.device)
                logger.warning(
                    "Elastic X0 physical memory receipt: floor_bytes=%d "
                    "evicted_graph_bytes=%d "
                    "active_graph_bytes=%d cublas_workspace_bytes=%d "
                    "allocator_allocated_bytes=%d "
                    "allocator_reserved_bytes=%d driver_free_bytes=%d "
                    "driver_total_bytes=%d",
                    physical_floor,
                    max(
                        0,
                        active_graph_bytes_before - working_set.active_graph_bytes,
                    )
                    + getattr(self, "_elastic_pre_idle_evicted_graph_bytes", 0),
                    working_set.active_graph_bytes,
                    cublas_workspace_bytes,
                    allocated,
                    reserved,
                    driver_free,
                    driver_total,
                )
                self._elastic_idle_logged_floor_bytes = physical_floor
            self._record_elastic_x0_allocation_delta(dynamic_wave=dynamic_wave)
            self._elastic_dynamic_wave_observed = False
        receipt = (
            effective_external,
            physical_floor,
            transition_floor_upper_bound,
        )
        self._elastic_cached_graph_receipt = receipt
        self._elastic_step_measurement_active = False
        return receipt

    def _current_dynamic_graph_receipt(self) -> tuple[int, int, int]:
        """Return the last boundary measurement without observing HOT replay."""
        working_set = self._dynamic_graph_working_set()
        return getattr(
            self,
            "_elastic_cached_graph_receipt",
            (working_set.resident_bytes, 0, 0),
        )

    def _settle_dynamic_graph_step_after_sampling(
        self,
        model_runner_output: ModelRunnerOutput,
        dynamic_graph_step_started: bool,
        transaction_id: str | None,
        step_plan: ElasticStepPlan | None,
    ) -> None:
        """Settle only a live step that began the elastic Graph lifecycle.

        Profile and synthetic startup warmups deliberately bypass same-step
        graph planning.  They may still build FlashInfer wrapper state used by
        later warmups, so treating them as a completed dynamic step would trim
        live startup metadata and can surface as an asynchronous illegal
        memory access on the following CUDA operation.
        """
        if not dynamic_graph_step_started:
            return
        (
            model_runner_output.elastic_external_memory_bytes,
            model_runner_output.elastic_external_memory_floor_bytes,
            model_runner_output.elastic_external_memory_transition_floor_bytes,
        ) = self._finish_dynamic_graph_step(
            transaction_id=transaction_id,
            step_plan=step_plan,
        )
        model_runner_output.elastic_external_memory_peak_bytes = getattr(
            self, "_elastic_last_step_peak_external_bytes", 0
        )
        self._set_elastic_cublas_workspace_unit(model_runner_output)
        self._publish_elastic_residency_receipt(model_runner_output, transaction_id)

    def _begin_elastic_step_measurement(self) -> None:
        """Start the physical high-water window after KV has been shrunk."""
        self._elastic_step_measurement_active = True
        torch.cuda.synchronize(self.device)
        if getattr(self, "_elastic_cublas_workspace_baseline", None) is None:
            # Capture the immutable startup floor before this step can create
            # CUDA Graph or cuBLAS workspaces.  Initializing this lazily in the
            # post-step measurement hid the entire first-step cuBLAS cost.
            self._elastic_cublas_workspace_baseline = (
                self._elastic_cublas_workspace_addresses()
            )
        self._elastic_step_baseline_reserved_bytes = torch.cuda.memory_reserved(
            self.device
        )
        self._elastic_step_baseline_driver_free_bytes = torch.cuda.mem_get_info(
            self.device
        )[0]
        working_set = self._dynamic_graph_working_set()
        self._elastic_step_baseline_external_bytes = (
            working_set.resident_bytes
            + self._measure_elastic_cublas_workspace_bytes()
            + getattr(self, "_elastic_retained_transition_floor_bytes", 0)
        )
        torch.cuda.reset_peak_memory_stats(self.device)

    @staticmethod
    def _elastic_mm_overlap_peak(
        baseline_external_bytes: int, observed_transient_delta_bytes: int
    ) -> int:
        return baseline_external_bytes + observed_transient_delta_bytes

    def _record_elastic_mm_transient_peak(self) -> None:
        """Record transient MM bytes separately from the HOT Graph endpoint."""
        baseline_reserved = getattr(self, "_elastic_step_baseline_reserved_bytes", None)
        baseline_driver_free = getattr(
            self, "_elastic_step_baseline_driver_free_bytes", None
        )
        baseline_external = getattr(self, "_elastic_step_baseline_external_bytes", None)
        if (
            baseline_reserved is None
            or baseline_driver_free is None
            or baseline_external is None
        ):
            raise RuntimeError("MM activation loan has no active measurement window")
        observed = max(
            0,
            torch.cuda.max_memory_reserved(self.device) - baseline_reserved,
            baseline_driver_free - torch.cuda.mem_get_info(self.device)[0],
        )
        overlap = self._elastic_mm_overlap_peak(baseline_external, observed)
        self._elastic_step_mm_overlap_peak_bytes = max(
            getattr(self, "_elastic_step_mm_overlap_peak_bytes", 0),
            overlap,
        )

    def _prepare_dynamic_graph_idle_kv_return(
        self,
        working_set: DynamicGraphWorkingSet,
        transaction_id: str,
    ) -> int:
        """Settle graph owners before an administrative X0 expands KV.

        A zero-token successor cannot consume an evictable executable or its
        retired attention wrapper.  If KV is expanded first, those CUDA
        allocations become an unaccounted physical floor on top of the FULL
        logical KV target.  Settle the idle descriptor and tear down retired
        wrappers while the old external loan is still mapped.  Pinned HOT
        entries remain resident and are returned as the minimum external loan.
        """
        working_set.prepare_idle_reclaim_before_post_consensus()
        self._elastic_pre_idle_evicted_graph_bytes = (
            working_set.evict_unpinned_for_idle(transaction_id)
        )
        _trimmed_wrappers, wrapper_reclaimed = (
            self._trim_dynamic_attention_cudagraph_state(working_set)
        )
        working_set.reconcile_retained_cleanup(wrapper_reclaimed)
        if working_set.active_graph_bytes == 0:
            self._clear_elastic_cublas_workspaces(working_set)
        return (
            working_set.resident_bytes + self._measure_elastic_cublas_workspace_bytes()
        )

    def _apply_next_elastic_kv_step(
        self,
        transition: tuple[int, int] | None,
        requested_external: int,
    ) -> int:
        """Apply the next known loan after incompatible owners were released."""
        if requested_external < self.elastic_kv_controller.external_memory_bytes:
            # Sampling/transient allocations from the completed step may
            # remain cached even after incompatible Graph owners are gone.
            # Trim only on a real downward transition; equal plateaus avoid
            # synchronization and allocator churn entirely.
            torch.cuda.synchronize(self.device)
            gc.collect()
            torch.accelerator.empty_cache()
            torch.cuda.synchronize(self.device)
        effective_external = self.elastic_kv_controller.apply_scheduler_step(
            transition,
            requested_external,
        )
        if effective_external != requested_external:
            logger.warning(
                "Elastic next-step loan retained a measured physical floor: "
                "requested_bytes=%d effective_bytes=%d",
                requested_external,
                effective_external,
            )
        self._elastic_retained_transition_floor_bytes = max(
            0, effective_external - requested_external
        )
        return effective_external

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        elastic_transition_applied = False
        elastic_dynamic_graph_step_started = False
        elastic_transaction_id = scheduler_output.elastic_transaction_id
        # Dummy/profile execution bypasses the dynamic residency transaction,
        # but ExecuteModelState still carries the immutable scheduler plan.
        # Resolve it before the branch so startup profiling cannot observe an
        # unbound local.
        elastic_plan = scheduler_output.elastic_step_plan
        elastic_equal_kv_noop_validated = False
        is_synthetic_warmup = scheduler_output.is_synthetic_warmup
        if (
            not dummy_run
            and not is_profile
            and not is_synthetic_warmup
            and self.cudagraph_manager is not None
            and not scheduler_output.elastic_preserve_graph_residency
        ):
            working_set = self._dynamic_graph_working_set()
            early_num_reqs = 0
            early_num_toks = 0
            early_uniform_tok_count = None
            early_geometry_error = None
            speculative_active = False
            semantic_short_decode = False
            try:
                if (
                    self.lora_config is None
                    and scheduler_output.total_num_scheduled_tokens > 0
                ):
                    early_num_reqs = len(scheduler_output.num_scheduled_tokens)
                    early_num_toks = scheduler_output.total_num_scheduled_tokens
                    early_max_query_len = max(
                        scheduler_output.num_scheduled_tokens.values()
                    )
                    early_uniform_tok_count = get_uniform_decode_token_count(
                        early_num_reqs,
                        early_num_toks,
                        early_max_query_len,
                        has_prefill=not scheduler_output.is_pure_decode_step,
                    )
                speculative_active = scheduler_output.num_spec_tokens_to_schedule > 0
                semantic_short_decode = bool(
                    scheduler_output.is_pure_decode_step
                    and speculative_active
                    and early_uniform_tok_count == self.decode_query_len
                    and early_num_toks == early_num_reqs * self.decode_query_len
                )
            except Exception as error:
                early_geometry_error = (
                    "ELASTIC_EXECUTION_PLAN_MISMATCH: local pre-mutation batch "
                    f"geometry failed: {type(error).__name__}: {error}"
                )
            compiled_dispatch_owners: frozenset[str] | None = None
            if elastic_plan is not None:
                validation_error = early_geometry_error
                try:
                    if self.dp_size != 1:
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic on-demand "
                            "graphs require DP=1 until cross-DP failure consensus "
                            "is implemented"
                        )
                    if self.parallel_config.pipeline_parallel_size != 1:
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic on-demand "
                            "graphs require PP=1 until intermediate-tensor "
                            "failure consensus is implemented"
                        )
                    if self.lora_config is not None:
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic on-demand "
                            "graphs do not yet bind active LoRA identity"
                        )
                    if self.kv_connector is not NO_OP_KV_CONNECTOR:
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic on-demand "
                            "graphs do not yet bind active KV connector state"
                        )
                    if (
                        self.supports_mm_inputs
                        and self.is_first_pp_rank
                        and self.ec_connector is not NO_OP_EC_CONNECTOR
                    ):
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic MM "
                            "execution does not yet support active encoder-cache "
                            "transfer"
                        )
                    if (
                        self.supports_mm_inputs
                        and self.is_first_pp_rank
                        and not self.is_encoder_only
                        and not self.is_encoder_decoder
                    ):
                        self.model_state.validate_elastic_mm_embedding_split()
                    if elastic_transaction_id != elastic_plan.transaction_id:
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic transaction "
                            "id differs from immutable plan"
                        )
                    if (
                        scheduler_output.elastic_plan_fingerprint
                        != elastic_plan.fingerprint
                    ):
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic plan "
                            "fingerprint changed in scheduler transport"
                        )
                    if (
                        elastic_plan.kv_transition
                        != scheduler_output.elastic_kv_transition
                    ):
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic KV transition "
                            "differs from immutable plan"
                        )
                    if (
                        elastic_plan.capture_loan_bytes
                        != scheduler_output.elastic_external_memory_bytes
                    ):
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic capture loan "
                            "differs from immutable plan"
                        )
                    new_requests = {
                        request.req_id: request
                        for request in scheduler_output.scheduled_new_reqs
                    }
                    cached_computed_tokens = dict(
                        zip(
                            scheduler_output.scheduled_cached_reqs.req_ids,
                            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
                            strict=True,
                        )
                    )
                    is_prefilling_by_request: dict[str, bool] = {}
                    for request_id in scheduler_output.num_scheduled_tokens:
                        new_request = new_requests.get(request_id)
                        if new_request is not None:
                            computed_tokens = new_request.num_computed_tokens
                            prefill_len = _elastic_new_request_execution_prefill_len(
                                new_request
                            )
                        else:
                            request_index = self.req_states.req_id_to_index.get(
                                request_id
                            )
                            if request_index is None:
                                raise ElasticExecutionPlanMismatch(
                                    "ELASTIC_EXECUTION_PLAN_MISMATCH: worker omitted "
                                    f"request state for {request_id!r}"
                                )
                            computed_tokens = cached_computed_tokens.get(
                                request_id,
                                int(
                                    self.req_states.num_computed_tokens_np[
                                        request_index
                                    ]
                                ),
                            )
                            prefill_len = int(
                                self.req_states.prefill_len.np[request_index]
                            )
                        is_prefilling_by_request[request_id] = (
                            computed_tokens < prefill_len
                        )
                    ordered_execution_ids = canonical_execution_request_order(
                        scheduler_output.num_scheduled_tokens,
                        is_prefilling_by_request=is_prefilling_by_request,
                        decode_query_len=self.decode_query_len,
                    )
                    working_set.validate_execution_manifest(
                        elastic_plan,
                        step_key=scheduler_output.elastic_graph_step_key,
                        request_ids=ordered_execution_ids,
                        per_request_query_lens=tuple(
                            scheduler_output.num_scheduled_tokens[request_id]
                            for request_id in ordered_execution_ids
                        ),
                        per_request_is_prefilling=tuple(
                            is_prefilling_by_request[request_id]
                            for request_id in ordered_execution_ids
                        ),
                        scheduled_draft_rows=tuple(
                            len(
                                scheduler_output.scheduled_spec_decode_tokens.get(
                                    request_id, ()
                                )
                            )
                            for request_id in ordered_execution_ids
                        ),
                        scheduled_encoder_inputs=(
                            scheduler_output.scheduled_encoder_inputs
                        ),
                        requested_output_k=(
                            scheduler_output.num_spec_tokens_to_schedule
                        ),
                        executed_drafter_k=(
                            self.num_speculative_steps
                            if scheduler_output.num_spec_tokens_to_schedule > 0
                            else 0
                        ),
                        phase=execution_manifest_phase_from_step_key(
                            scheduler_output.elastic_graph_step_key
                        ),
                        max_num_batched_tokens=self.max_num_tokens,
                    )
                    compiled_dispatch_owners = frozenset(
                        dispatch.invocation.owner
                        for dispatch in elastic_plan.current_dispatch
                        if dispatch.representation
                        == DispatchRepresentation.COMPILED_ONLY
                    )
                    managers = {
                        manager.dynamic_graph_owner: manager
                        for manager in working_set.managers
                    }
                    if not compiled_dispatch_owners.issubset(managers):
                        raise ElasticExecutionPlanMismatch(
                            "ELASTIC_EXECUTION_PLAN_MISMATCH: explicit compiled "
                            "dispatch references an unknown owner"
                        )
                    for owner in compiled_dispatch_owners:
                        manager = managers[owner]
                        if not manager.is_compiled_piecewise_shape(
                            early_num_toks,
                            num_reqs=early_num_reqs,
                            semantic_decode=semantic_short_decode,
                        ):
                            raise ElasticExecutionPlanMismatch(
                                "ELASTIC_EXECUTION_PLAN_MISMATCH: explicit compiled "
                                f"route is not executable: owner={owner!r}"
                            )
                    elastic_equal_kv_noop_validated = (
                        self.elastic_kv_controller.validate_equal_scheduler_step(
                            scheduler_output.elastic_kv_transition,
                            scheduler_output.elastic_external_memory_bytes,
                        )
                    )
                except Exception as error:
                    # Every deterministic identity/read-back failure in this
                    # pre-mutation block must enter the same CPU-group vote.
                    # Letting one rank escape early would strand its peers in
                    # the collective or allow a partial Graph/KV transition.
                    local_validation_error = (
                        "ELASTIC_EXECUTION_PLAN_MISMATCH: local pre-mutation "
                        f"validation failed: {type(error).__name__}: {error}"
                    )
                    validation_error = (
                        local_validation_error
                        if validation_error is None
                        else f"{validation_error}; {local_validation_error}"
                    )
                    compiled_dispatch_owners = frozenset()
                working_set.begin_admitted_step(
                    elastic_plan,
                    validation_error=validation_error,
                )
            else:
                working_set.begin_step()
            elastic_dynamic_graph_step_started = True
            active_graph_bytes_before_shape_release = working_set.active_graph_bytes
            if elastic_plan is None:
                managers_to_queue = working_set.managers
            else:
                # A configured compiled-only target or MTP-prefill carrier has
                # no physical CUDA Graph key in the immutable plan. It still
                # needs the exact same-step descriptor so dispatch can enter
                # the compiled PIECEWISE path; no other missing owner is allowed.
                assert compiled_dispatch_owners is not None
                managers_to_queue = tuple(
                    manager
                    for manager in working_set.managers
                    if manager.dynamic_graph_owner in compiled_dispatch_owners
                    and not manager._dynamic_step_planned
                )
            if managers_to_queue:
                for manager in managers_to_queue:
                    if (
                        manager.elastic_graph_activation == "speculative"
                        and not speculative_active
                    ):
                        num_reqs, num_tokens, uniform_token_count = (0, 0, None)
                    elif manager.elastic_graph_token_source == "step":
                        num_reqs, num_tokens, uniform_token_count = (
                            early_num_reqs,
                            early_num_toks,
                            early_uniform_tok_count,
                        )
                    elif manager.elastic_graph_token_source == "requests":
                        num_reqs, num_tokens, uniform_token_count = (
                            early_num_reqs,
                            early_num_reqs,
                            1,
                        )
                    elif manager.elastic_graph_token_source == "fixed_query":
                        query_len = manager.elastic_graph_fixed_query_len
                        if query_len is None:
                            raise RuntimeError(
                                "fixed-query Graph manager omitted its query length"
                            )
                        num_reqs, num_tokens, uniform_token_count = (
                            early_num_reqs,
                            early_num_reqs * query_len,
                            query_len,
                        )
                    else:
                        raise RuntimeError(
                            "Graph manager omitted its runtime token-source contract: "
                            f"owner={manager.dynamic_graph_owner!r} "
                            f"source={manager.elastic_graph_token_source!r}"
                        )
                    allow_full = bool(
                        manager.elastic_graph_token_source
                        in {"requests", "fixed_query"}
                        or scheduler_output.is_pure_decode_step
                    )
                    manager.queue_runtime_descriptor(
                        num_reqs,
                        num_tokens,
                        uniform_token_count,
                        0,
                        allow_full=allow_full,
                        semantic_decode=semantic_short_decode,
                    )
            idle_external_floor = None
            if (
                scheduler_output.total_num_scheduled_tokens == 0
                and _release_idle_graph_cache(scheduler_output)
            ):
                # Request-free MAINTENANCE is a capture/publication phase, not
                # X0. Its staged victims must remain HOT until the newly
                # published set and every downstream consumer have completed;
                # pre-idle eviction here destroyed them before the deferred
                # post-consumer commit could validate transaction ownership.
                idle_external_floor = self._prepare_dynamic_graph_idle_kv_return(
                    working_set,
                    elastic_transaction_id or "",
                )
                self._elastic_dynamic_wave_observed = True
            elif (
                active_graph_bytes_before_shape_release > 0
                and working_set.active_graph_bytes == 0
            ):
                # FlashInfer/TVM wrapper teardown is not safe in the narrow
                # lifetime gap between evicting one descriptor and planning
                # the successor: PIECEWISE64 -> FULL18 reproduced a native
                # use-after-lifetime on the successor's first warmup.  Keep
                # the retired wrapper through successor capture/replay.  The
                # ordinary post-step _finish_dynamic_graph_step() trims it
                # against the newly HOT working set, so this is one-transition
                # deferred reclamation rather than permanent residency.
                self._clear_elastic_cublas_workspaces(working_set)
                self._elastic_dynamic_wave_observed = True
            requested_external = scheduler_output.elastic_external_memory_bytes
            transition = scheduler_output.elastic_kv_transition
            if (
                idle_external_floor is not None
                and idle_external_floor > requested_external
            ):
                # Scheduler X0 describes the logical no-work target. Pinned
                # graphs and physically retained teardown pages still consume
                # part of the invariant budget, so keep the prior minimum KV
                # geometry and fill only the actually available remainder.
                requested_external = idle_external_floor
                transition = None
            capture_managers = working_set.pending_managers()
            measurement_required = (
                bool(capture_managers)
                or requested_external
                != self.elastic_kv_controller.external_memory_bytes
                or scheduler_output.elastic_mm_activation_loan_bytes > 0
            )
            with record_function_or_nullcontext("ag2.elastic_kv_transition"):
                if (
                    elastic_equal_kv_noop_validated
                    and transition is None
                    and requested_external
                    == scheduler_output.elastic_external_memory_bytes
                ):
                    # The local backing layout joined the already-required
                    # pre-mutation all-rank vote above. Re-entering apply()
                    # would perform two device collectives and synchronizing
                    # scalar reads only to rediscover that nothing changed.
                    self._elastic_retained_transition_floor_bytes = 0
                else:
                    self._apply_next_elastic_kv_step(
                        transition,
                        requested_external,
                    )
            elastic_transition_applied = True
            if measurement_required:
                self._begin_elastic_step_measurement()
                self._elastic_step_mm_overlap_peak_bytes = 0
            if capture_managers:
                with self.maybe_setup_dummy_loras(self.lora_config):
                    try:
                        for capture_manager in capture_managers:
                            if not working_set.prepare_manager_capture(
                                capture_manager,
                                scheduler_output.elastic_external_memory_bytes,
                            ):
                                raise RuntimeError(
                                    "scheduler loan is insufficient for same-step "
                                    f"CUDA Graph capture: owner="
                                    f"{capture_manager.dynamic_graph_owner}"
                                )
                            if capture_manager is self.cudagraph_manager:
                                captured = capture_manager.capture_next_dynamic(
                                    self.model,
                                    self.model_state,
                                    self.input_buffers,
                                    self.intermediate_tensors,
                                    self.block_tables,
                                    self.attn_groups,
                                    self.kv_cache_config,
                                    has_lora=self.lora_config is not None,
                                    use_aux_hidden_state_outputs=(
                                        self.use_aux_hidden_state_outputs
                                    ),
                                    lora_capture_hook=create_lora_capture_hook(
                                        self.lora_config, self
                                    ),
                                )
                            else:
                                capture_speculator = getattr(
                                    self.speculator, "capture_next_dynamic", None
                                )
                                if capture_speculator is None:
                                    raise RuntimeError(
                                        "draft CUDA Graph manager has no dynamic "
                                        "capture lifecycle"
                                    )
                                captured = capture_speculator(capture_manager)
                            if not captured:
                                rejection = getattr(
                                    capture_manager,
                                    "last_dynamic_capture_rejection",
                                    None,
                                )
                                raise RuntimeError(
                                    "same-step CUDA Graph capture failed closed: "
                                    f"owner={capture_manager.dynamic_graph_owner} "
                                    f"discriminator={rejection!r}"
                                )
                    except Exception:
                        if elastic_plan is not None:
                            working_set.discard_failed_capture(elastic_plan)
                        # A failed capture has no downstream consumer, so
                        # reclaim its loan immediately before propagating the
                        # fail-closed error. Successful capture/replay keeps
                        # the same-step loan through target execution and the
                        # separate sampling RPC; _finish_dynamic_graph_step
                        # returns it only after those consumers complete.
                        self.elastic_kv_controller.reconcile_external_memory(
                            working_set.resident_bytes
                        )
                        raise
            if elastic_transaction_id is None:
                raise RuntimeError(
                    "elastic CUDA Graph execution requires scheduler transaction id"
                )
            working_set.acquire_leases(elastic_transaction_id)
        if not dummy_run:

            def apply_scheduler_state() -> bool:
                if not elastic_transition_applied:
                    with record_function_or_nullcontext("ag2.elastic_kv_transition"):
                        self.elastic_kv_controller.apply_scheduler_step(
                            scheduler_output.elastic_kv_transition,
                            scheduler_output.elastic_external_memory_bytes,
                        )
                if self.gdn_checkpoint_manager is not None:
                    self.gdn_checkpoint_manager.update_request_blocks(scheduler_output)
                self.update_pp_decode_requests()
                self.finish_requests(scheduler_output)
                self.free_states(scheduler_output)
                self.add_requests(scheduler_output)
                self.update_requests(scheduler_output)
                self.block_tables.apply_staged_writes()
                additional_config = self.vllm_config.additional_config
                if isinstance(additional_config, dict) and additional_config.get(
                    "p3_block_range_diagnostic", False
                ):
                    validate_elastic_attention_block_tables(
                        self.block_tables,
                        self.kv_cache_config,
                        self.req_states.req_id_to_index,
                        scheduler_output.num_scheduled_tokens,
                        self.elastic_kv_controller.mapped_attention_block_capacity(),
                    )
                if self.gdn_checkpoint_manager is not None:
                    num_computed_tokens = {
                        req_id: int(
                            self.req_states.num_computed_tokens_np[
                                self.req_states.req_id_to_index[req_id]
                            ]
                        )
                        for req_id in scheduler_output.num_scheduled_tokens
                    }
                    self.gdn_checkpoint_manager.prepare(
                        scheduler_output,
                        self.kv_cache_config,
                        num_computed_tokens,
                        self.compilation_config.static_forward_context,
                    )
                return True

            state_application_error: Exception | None = None
            try:
                apply_scheduler_state()
            except Exception as error:
                state_application_error = error
            if scheduler_output.total_num_scheduled_tokens == 0:
                if elastic_plan is not None:
                    assert working_set is not None
                    working_set.require_post_materialization_consensus(
                        elastic_plan,
                        validation_error=(
                            None
                            if state_application_error is None
                            else (
                                "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH: local "
                                "scheduler-state application failed: "
                                f"{type(state_application_error).__name__}: "
                                f"{state_application_error}"
                            )
                        ),
                        phase="post_state_application",
                    )
                elif state_application_error is not None:
                    raise state_application_error
                # No need to run the model.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                if isinstance(empty_output, ModelRunnerOutput):
                    empty_output = copy(empty_output)
                if elastic_dynamic_graph_step_started:
                    (
                        external_request,
                        external_floor,
                        transition_floor,
                    ) = self._finish_dynamic_graph_step(
                        release_idle_cache=_release_idle_graph_cache(scheduler_output),
                        transaction_id=elastic_transaction_id,
                        step_plan=elastic_plan,
                    )
                    if isinstance(empty_output, ModelRunnerOutput):
                        empty_output.elastic_external_memory_bytes = external_request
                        empty_output.elastic_external_memory_floor_bytes = (
                            external_floor
                        )
                        empty_output.elastic_external_memory_transition_floor_bytes = (
                            transition_floor
                        )
                        empty_output.elastic_external_memory_peak_bytes = getattr(
                            self, "_elastic_last_step_peak_external_bytes", 0
                        )
                        self._set_elastic_cublas_workspace_unit(empty_output)
                        self._publish_elastic_residency_receipt(
                            empty_output, elastic_transaction_id
                        )
                elif scheduler_output.elastic_preserve_graph_residency and isinstance(
                    empty_output, ModelRunnerOutput
                ):
                    (
                        empty_output.elastic_external_memory_bytes,
                        empty_output.elastic_external_memory_floor_bytes,
                        empty_output.elastic_external_memory_transition_floor_bytes,
                    ) = self._current_dynamic_graph_receipt()
                    empty_output.elastic_external_memory_peak_bytes = (
                        empty_output.elastic_external_memory_bytes
                    )
                    self._set_elastic_cublas_workspace_unit(empty_output)
                    self._publish_elastic_residency_receipt(
                        empty_output, elastic_transaction_id
                    )
                if isinstance(empty_output, ModelRunnerOutput):
                    empty_output.elastic_mm_activation_loan_bytes = (
                        scheduler_output.elastic_mm_activation_loan_bytes
                    )
                return self._merge_ec_connector_no_forward(
                    scheduler_output, empty_output
                )

            if state_application_error is not None and elastic_plan is None:
                raise state_application_error

        # Get batch descriptor and sync across DP ranks.
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        batch_req_state, uniform_tok_count = self.gather_batch_req_state(
            scheduler_output, dummy_run
        )
        if batch_req_state is not None:
            num_toks = batch_req_state.num_tokens
            if self.pcp_manager is not None:
                num_toks = self.pcp_manager.get_num_tokens_for_dispatch(
                    batch_req_state.num_scheduled_tokens,
                    batch_req_state.is_prefilling_np,
                )

        num_active_loras = 0
        if self.lora_config:
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            num_active_loras = get_num_active_loras_for_dispatch(
                self.lora_config, self.lora_state, req_ids, dummy_run
            )

        skip_compiled = False
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Encoder-decoder models such as Whisper should run eager/non-compiled
            # when encoder inputs are scheduled, because this step updates
            # cross-attention cache with dynamic encoder outputs.
            skip_compiled = True

        batch_desc, dp_sync = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            max_query_len=max_query_len,
            # Compile/profile dummy runs have no scheduler admission and must
            # execute the compiled direct path, never an elastic Graph. Live
            # execution remains fail-closed on same-step planning below.
            need_eager=(
                dummy_run or is_profile or is_synthetic_warmup or skip_compiled
            ),
            num_active_loras=num_active_loras,
            parallel_config=self.parallel_config,
            allow_ubatching=(
                self.ubatch_runner is not None and not skip_attn_for_dummy_run
            ),
            uniform_decode=uniform_tok_count == self.decode_query_len,
        )

        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            if isinstance(empty_output, ModelRunnerOutput):
                empty_output = copy(empty_output)
            if elastic_dynamic_graph_step_started:
                (
                    external_request,
                    external_floor,
                    transition_floor,
                ) = self._finish_dynamic_graph_step(
                    release_idle_cache=_release_idle_graph_cache(scheduler_output),
                    transaction_id=elastic_transaction_id,
                    step_plan=elastic_plan,
                )
                if isinstance(empty_output, ModelRunnerOutput):
                    empty_output.elastic_external_memory_bytes = external_request
                    empty_output.elastic_external_memory_floor_bytes = external_floor
                    empty_output.elastic_external_memory_transition_floor_bytes = (
                        transition_floor
                    )
                    self._set_elastic_cublas_workspace_unit(empty_output)
                    self._publish_elastic_residency_receipt(
                        empty_output, elastic_transaction_id
                    )
            if isinstance(empty_output, ModelRunnerOutput):
                empty_output.elastic_mm_activation_loan_bytes = (
                    scheduler_output.elastic_mm_activation_loan_bytes
                )
            return self._merge_ec_connector_no_forward(scheduler_output, empty_output)

        ubatch_state: UBatchState | None = None
        cudagraph_stats = None
        if not dummy_run and self.observability_config.cudagraph_metrics:
            cudagraph_stats = make_cudagraph_stats(batch_desc, num_toks)
        graph_receipt = None
        if _AG2_GRAPH_MODE_RECEIPT and not dummy_run and not is_synthetic_warmup:
            graph_receipt_key = (
                self.cudagraph_manager.dynamic_graph_owner,
                batch_desc.cg_mode.name,
                num_toks,
                batch_desc.num_tokens,
                num_reqs,
            )
            if graph_receipt_key not in _AG2_GRAPH_MODE_LOGGED_RECEIPTS:
                _AG2_GRAPH_MODE_LOGGED_RECEIPTS.add(graph_receipt_key)
                logger.info(
                    "AG2 CUDA Graph execution receipt: owner=%s mode=%s "
                    "tokens_unpadded=%d tokens_padded=%d requests=%d "
                    "descriptor=%r",
                    self.cudagraph_manager.dynamic_graph_owner,
                    batch_desc.cg_mode.name,
                    num_toks,
                    batch_desc.num_tokens,
                    num_reqs,
                    batch_desc,
                )
            graph_receipt = (
                "ag2.graph_receipt"
                f"|mode={batch_desc.cg_mode.name}"
                f"|tokens_unpadded={num_toks}"
                f"|tokens_padded={batch_desc.num_tokens}"
                f"|requests={num_reqs}"
                f"|descriptor={batch_desc!r}"
            )

        if not dummy_run:
            assert batch_req_state is not None

            def prepare_real_input_staging():
                nonlocal ubatch_state
                # Finish every rank-local fallible stage before the final
                # all-rank vote. The next operation after this closure is the
                # common model-state attention phase.
                if state_application_error is not None:
                    raise RuntimeError(
                        "local scheduler-state application failed: "
                        f"{type(state_application_error).__name__}: "
                        f"{state_application_error}"
                    ) from state_application_error
                with record_function_or_nullcontext("ag2.target_prepare_inputs"):
                    input_batch = self.prepare_inputs(
                        scheduler_output, batch_req_state, batch_desc
                    )
                if elastic_plan is not None:
                    assert elastic_plan.execution_manifest is not None
                    _validate_elastic_materialized_input_batch(
                        elastic_plan.execution_manifest, input_batch
                    )
                self._validate_elastic_calibration_sampling_indices(
                    input_batch,
                    hidden_rows=input_batch.num_tokens_after_padding,
                    phase="pre_forward",
                    only_single_token_prefill=True,
                )
                with record_function_or_nullcontext("ag2.target_prepare_attention"):
                    block_tables, slot_mappings = self.prepare_attn(input_batch)
                # Mamba "align" pre-copy migrates recurrent state before
                # attention metadata observes accepted-token state.
                with record_function_or_nullcontext("ag2.target_preprocess_state"):
                    self.model_state.preprocess_state(
                        input_batch,
                        block_tables,
                        self.kv_cache_config,
                        self.req_states.num_computed_tokens.gpu,
                    )
                if self.lora_config:
                    lora_inputs = self.lora_state.make_lora_inputs(
                        input_batch.req_ids,
                        input_batch.idx_mapping_np,
                        input_batch.num_scheduled_tokens,
                    )
                    self._set_active_loras(*lora_inputs)
                if batch_desc.num_ubatches > 1:
                    assert self.ubatch_runner is not None
                    ubatch_state = self.ubatch_runner.prepare(
                        input_batch, block_tables, slot_mappings
                    )
                    slot_mappings_by_layer = None
                    attn_metadata = None
                else:
                    slot_mappings_by_layer = build_slot_mappings_by_layer(
                        slot_mappings, self.kv_cache_config
                    )
                    with record_function_or_nullcontext(
                        "ag2.target_prepare_attention_metadata"
                    ):
                        attn_metadata = self.model_state.prepare_attn(
                            input_batch,
                            batch_desc.cg_mode,
                            block_tables,
                            slot_mappings,
                            self.attn_groups,
                            self.kv_cache_config,
                            for_capture=False,
                        )
                if (
                    self.supports_mm_inputs
                    and self.is_first_pp_rank
                    and self.lora_config is not None
                ):
                    set_active_mm_loras(
                        model=self.model,
                        lora_manager=self.lora_manager,
                        encoder_cache=self.encoder_cache,
                        req_id_to_index=self.req_states.req_id_to_index,
                        lora_state=self.lora_state,
                        scheduled_encoder_inputs=(
                            scheduler_output.scheduled_encoder_inputs
                        ),
                    )
                # MM model states finalize positions/encoder outputs only
                # after prepare_inputs_embeds. Non-MM inputs are complete here.
                mm_finalize_after_encoder = bool(
                    self.supports_mm_inputs
                    and self.is_first_pp_rank
                    and self.encoder_cache is not None
                    and any(
                        self.encoder_cache.mm_features.get(request_id)
                        for request_id in input_batch.req_ids
                    )
                )
                if (
                    elastic_plan is not None
                    and self.supports_mm_inputs
                    and self.is_first_pp_rank
                    and not self.is_encoder_only
                    and not self.is_encoder_decoder
                ):
                    self.model_state.validate_elastic_mm_embedding_split(input_batch)
                if mm_finalize_after_encoder and elastic_plan is not None:
                    self.model_state.validate_mm_cache_readiness(
                        scheduler_output.scheduled_encoder_inputs,
                        input_batch,
                    )
                staged_mm_encoder = (
                    self.model_state.stage_mm_encoder(
                        scheduler_output.scheduled_encoder_inputs,
                        input_batch.req_ids,
                    )
                    if mm_finalize_after_encoder and elastic_plan is not None
                    else None
                )
                if mm_finalize_after_encoder and elastic_plan is not None:
                    self.model_state.validate_staged_mm_encoder(
                        scheduler_output.scheduled_encoder_inputs,
                        staged_mm_encoder,
                        input_batch.req_ids,
                    )
                staged_mm_embeddings = (
                    self.model_state.stage_mm_embeddings(input_batch, self.req_states)
                    if (
                        self.uses_inputs_embeds
                        and self.is_first_pp_rank
                        and not self.is_encoder_only
                        and not mm_finalize_after_encoder
                        and elastic_plan is not None
                    )
                    else None
                )
                prepared_model_inputs = (
                    None
                    if mm_finalize_after_encoder
                    else self.model_state.prepare_inputs(input_batch, self.req_states)
                )
                if ubatch_state is None:
                    self.eplb.prepare_forward(self.model_config, input_batch.num_tokens)
                else:
                    self.eplb.prepare_forward(
                        self.model_config, input_batch.num_tokens, ubatch_state.slices
                    )
                return (
                    input_batch,
                    block_tables,
                    slot_mappings,
                    slot_mappings_by_layer,
                    attn_metadata,
                    prepared_model_inputs,
                    mm_finalize_after_encoder,
                    staged_mm_encoder,
                    staged_mm_embeddings,
                )

            if elastic_plan is not None:
                assert working_set is not None
                (
                    input_batch,
                    block_tables,
                    slot_mappings,
                    slot_mappings_by_layer,
                    attn_metadata,
                    prepared_model_inputs,
                    mm_finalize_after_encoder,
                    staged_mm_encoder,
                    staged_mm_embeddings,
                ) = _prepare_elastic_local_staging_with_consensus(
                    prepare_real_input_staging,
                    elastic_plan,
                    working_set,
                    observer_fingerprint=_elastic_input_staging_fingerprint,
                )
            else:
                (
                    input_batch,
                    block_tables,
                    slot_mappings,
                    slot_mappings_by_layer,
                    attn_metadata,
                    prepared_model_inputs,
                    mm_finalize_after_encoder,
                    staged_mm_encoder,
                    staged_mm_embeddings,
                ) = prepare_real_input_staging()
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            dummy_num_reqs = batch_desc.num_reqs or num_reqs
            input_batch = InputBatch.make_dummy(
                dummy_num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
                max_query_len=batch_desc.max_query_len,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(
                    input_batch, valid_dummy_state_slots
                )
                if context_len:
                    set_dummy_context(
                        input_batch,
                        self.block_tables,
                        context_len,
                        self.kv_cache_config.num_blocks,
                        self.max_model_len,
                    )
            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None

        if dummy_run:
            attn_metadata = None
            slot_mappings_by_layer = None
        if dummy_run and batch_desc.num_ubatches > 1:
            assert self.ubatch_runner is not None
            assert block_tables is not None and slot_mappings is not None
            ubatch_state = self.ubatch_runner.prepare(
                input_batch, block_tables, slot_mappings
            )
        elif batch_desc.num_ubatches == 1 and not (
            dummy_run and skip_attn_for_dummy_run
        ):
            assert slot_mappings is not None
            if dummy_run:
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
            if dummy_run:
                assert block_tables is not None
                with record_function_or_nullcontext(
                    "ag2.target_prepare_attention_metadata"
                ):
                    attn_metadata = self.model_state.prepare_attn(
                        input_batch,
                        batch_desc.cg_mode,
                        block_tables,
                        slot_mappings,
                        self.attn_groups,
                        self.kv_cache_config,
                        # FULL replay re-stages capture-time metadata buffers.
                        for_capture=batch_desc.cg_mode == CUDAGraphMode.FULL,
                    )

        input_ids = input_batch.input_ids
        inputs_embeds = None
        ec_connector_output = None
        if self.uses_inputs_embeds and self.is_first_pp_rank:
            # Prepare inputs_embeds (MM encoder outputs and/or prompt_embeds
            # overlay). Only first PP rank prepares them.
            if dummy_run:
                # Obtain embeddings of correct shape for compiled model.
                inputs_embeds = self.model_state.dummy_inputs_embeds(
                    input_batch.num_tokens_after_padding
                )
            else:
                scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
                with self.ec_connector.maybe_get_output(
                    scheduler_output
                ) as ec_connector_output:
                    completed_encoder = None
                    if mm_finalize_after_encoder and elastic_plan is not None:
                        # Readiness was converged before entering the encoder
                        # phase. Every rank executes it in the same order.
                        completed_encoder = (
                            self.model_state.execute_staged_mm_encoder_collective(
                                staged_mm_encoder
                            )
                        )
                    if self.is_encoder_only:
                        # Encode and publish, nothing else: this instance runs no
                        # language model, so the gather inside prepare_inputs_embeds
                        # would build an inputs_embeds nobody reads -- and it
                        # raises "Encoder cache miss" for any scheduled item this
                        # instance did not encode, taking the engine down with it.
                        if mm_finalize_after_encoder and elastic_plan is None:
                            self.model_state.execute_mm_encoder(
                                scheduled_encoder_inputs
                            )
                        elif mm_finalize_after_encoder:
                            assert working_set is not None
                            _prepare_elastic_local_staging_with_consensus(
                                lambda: (
                                    self.model_state.commit_staged_mm_encoder(
                                        completed_encoder
                                    ),
                                ),
                                elastic_plan,
                                working_set,
                                consensus_phase="post_mm_encoder_commit",
                            )
                            if scheduler_output.elastic_mm_activation_loan_bytes:
                                self._record_elastic_mm_transient_peak()
                    elif mm_finalize_after_encoder:
                        if elastic_plan is not None:
                            assert working_set is not None
                            (
                                staged_mm_embeddings,
                                prepared_model_inputs,
                            ) = _stage_elastic_mm_inputs_with_consensus(
                                self.model_state,
                                completed_encoder,
                                input_batch,
                                self.req_states,
                                elastic_plan,
                                working_set,
                            )
                            computed_embeddings = (
                                self.model_state.execute_staged_mm_embeddings(
                                    staged_mm_embeddings, input_batch
                                )
                            )
                            (inputs_embeds,) = (
                                _prepare_elastic_local_staging_with_consensus(
                                    lambda: (
                                        self._commit_staged_mm_embeddings(
                                            computed_embeddings, input_batch
                                        ),
                                    ),
                                    elastic_plan,
                                    working_set,
                                    consensus_phase="post_mm_embedding",
                                )
                            )
                            if scheduler_output.elastic_mm_activation_loan_bytes:
                                self._record_elastic_mm_transient_peak()
                        else:
                            inputs_embeds = self.model_state.prepare_inputs_embeds(
                                scheduled_encoder_inputs,
                                input_batch,
                                self.req_states,
                            )
                            prepared_model_inputs = self.model_state.prepare_inputs(
                                input_batch, self.req_states
                            )
                    elif not self.is_encoder_only:
                        if elastic_plan is not None:
                            computed_embeddings = (
                                self.model_state.execute_staged_mm_embeddings(
                                    staged_mm_embeddings, input_batch
                                )
                            )
                            # With no media state, manifest validation proves a
                            # uniform token-only merge shape. Avoid a third CPU
                            # consensus on the steady text path.
                            inputs_embeds = self._commit_staged_mm_embeddings(
                                computed_embeddings, input_batch
                            )
                        else:
                            inputs_embeds = self.model_state.prepare_inputs_embeds(
                                scheduled_encoder_inputs,
                                input_batch,
                                self.req_states,
                            )
            if inputs_embeds is not None and not requires_raw_input_tokens(self.model):
                input_ids = None

        if elastic_plan is not None and not dummy_run:
            assert working_set is not None
            working_set.validate_post_materialization_completion()

        if self.is_encoder_only:
            output = make_empty_encoder_model_runner_output(scheduler_output)
            output.ec_connector_output = ec_connector_output
            return output

        if dummy_run:
            prepared_model_inputs = self.model_state.prepare_runtime_dummy_inputs(
                input_batch, self.req_states
            )
        model_inputs = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": None,
            # NOTE: Values returned by `prepare_inputs` will override the default
            # values above.
            **prepared_model_inputs,
        }
        from vllm.model_executor.models.qwen3_next_ready import (
            bind_ready_rows,
            ready_compaction_enabled,
        )

        model_inputs = bind_ready_rows(
            model_inputs,
            enabled=ready_compaction_enabled(self.vllm_config),
            actual_rows=input_batch.num_tokens,
            physical_rows=input_batch.num_tokens_after_padding,
            full_graph=batch_desc.cg_mode == CUDAGraphMode.FULL,
        )
        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None

            # Prepare the intermediate tensors.
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            n = input_batch.num_tokens_after_padding
            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
            }
            model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
            del intermediate_tensors

        ubatch_slices = ubatch_state.slices if ubatch_state is not None else None
        # Real staging already updates EPLB before its all-rank vote.
        if dummy_run:
            self.eplb.prepare_forward(
                self.model_config, input_batch.num_tokens, ubatch_slices
            )

        self.step_timing.record_batch(
            input_batch, batch_desc.cg_mode == CUDAGraphMode.FULL
        )
        self.step_timing.forward_start()

        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            additional_config = self.vllm_config.additional_config
            if (
                isinstance(additional_config, dict)
                and additional_config.get("p3_crash_diagnostic", False)
                and batch_desc.uniform_token_count == 3
            ):
                logger.warning(
                    "P3 FULL dispatch: desc=%s scheduled=%s spec_lengths=%s "
                    "pure_decode=%s structured=%s pending_structured=%s",
                    batch_desc,
                    scheduler_output.num_scheduled_tokens,
                    {
                        req_id: len(token_ids)
                        for req_id, token_ids in (
                            scheduler_output.scheduled_spec_decode_tokens.items()
                        )
                    },
                    scheduler_output.is_pure_decode_step,
                    scheduler_output.has_structured_output_requests,
                    scheduler_output.pending_structured_output_tokens,
                )
            forward_scope = (
                torch.profiler.record_function(graph_receipt)
                if graph_receipt is not None
                else record_function_or_nullcontext("ag2.target_forward.full")
            )
            with forward_scope:
                model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            # For piecewise and eager mode, just call model().
            tp3_sd_phase_reduce = (
                envs.VLLM_TP3_SD_PHASE_REDUCE and scheduler_output.is_pure_decode_step
            )
            tp3_owner_prequant_decode = bool(
                self.cudagraph_manager is not None
                and self.cudagraph_manager.uses_tp3_owner_prequant_decode(batch_desc)
            )
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
                tp3_sd_phase_reduce=tp3_sd_phase_reduce,
                tp3_owner_prequant_decode=tp3_owner_prequant_decode,
                # Memory profiling runs before the CUDA Graph manager exists.
                # That eager path cannot publish a graph, but it still needs a
                # stable target identity. Live Graph execution always uses the
                # manager's explicit owner.
                cudagraph_owner=(
                    self.cudagraph_manager.dynamic_graph_owner
                    if self.cudagraph_manager is not None
                    else "target"
                ),
                physical_num_reqs=batch_desc.physical_num_reqs,
                runtime_generation=batch_desc.runtime_generation,
            )

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=(
                    dp_sync.num_tokens_across_dp if dp_sync is not None else None
                ),
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=input_batch.is_padding,
                num_tokens_unpadded=input_batch.num_tokens,
                tp3_sd_phase_reduce=tp3_sd_phase_reduce,
                tp3_owner_prequant_decode=tp3_owner_prequant_decode,
                marlin_request_layout_cpu=(
                    self.input_buffers.marlin_request_layout_cpu
                    if envs.AG2_VLLM_NVFP4_MARLIN_ISOLATE_PREFILL
                    else None
                ),
            ):
                self.kv_connector.pre_forward(scheduler_output)
                if ubatch_state is not None:
                    assert self.ubatch_runner is not None
                    model_output = self.ubatch_runner.run(
                        self.model, model_inputs, ubatch_state
                    )
                elif batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    # Run the PIECEWISE graph (compiled PW cudagraph or breakable
                    # cudagraph, chosen inside run_pw_graph). cg_mode is only
                    # PIECEWISE after the cudagraph manager exists.
                    assert self.cudagraph_manager is not None
                    forward_scope = (
                        torch.profiler.record_function(graph_receipt)
                        if graph_receipt is not None
                        else record_function_or_nullcontext(
                            "ag2.target_forward.piecewise"
                        )
                    )
                    with forward_scope:
                        model_output = self.cudagraph_manager.run_pw_graph(
                            self.model, model_inputs
                        )
                else:
                    compiled_no_cudagraph = bool(
                        self.cudagraph_manager is not None
                        and self.cudagraph_manager.defer_startup_graphs
                        and input_batch.num_tokens_after_padding
                        in self.cudagraph_manager.compiled_piecewise_sizes
                    )
                    if compiled_no_cudagraph and skip_compiled:
                        raise RuntimeError(
                            "compiled-only PIECEWISE carrier attempted to skip "
                            "torch.compile"
                        )
                    forward_scope = (
                        torch.profiler.record_function(graph_receipt)
                        if graph_receipt is not None
                        else record_function_or_nullcontext(
                            "ag2.target_forward.compiled_no_cudagraph"
                            if compiled_no_cudagraph
                            else "ag2.target_forward.eager"
                        )
                    )
                    with forward_scope:
                        model_output = self.model(**model_inputs)

        if not dummy_run:
            save_layer0_trace = getattr(self.model, "maybe_save_ag2_layer0_trace", None)
            if save_layer0_trace is not None:
                save_layer0_trace(
                    input_ids=input_batch.input_ids,
                    positions=input_batch.positions,
                    query_len=scheduler_output.total_num_scheduled_tokens,
                    cudagraph_mode=batch_desc.cg_mode.name,
                )

        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
                if self.aux_hidden_trace.enabled:
                    released_raw_refs = self.aux_hidden_trace.release_raw_model_refs(
                        self.model
                    )
                    if dummy_run and released_raw_refs:
                        logger.info_once(
                            "Released %d raw auxiliary trace references before "
                            "MTP proposer lifecycle",
                            released_raw_refs,
                            scope="local",
                        )
                    if not dummy_run:
                        self.aux_hidden_trace.maybe_save(
                            input_ids=input_batch.input_ids,
                            positions=input_batch.positions,
                            query_len=scheduler_output.total_num_scheduled_tokens,
                            cudagraph_mode=batch_desc.cg_mode.name,
                            aux_hidden_states=aux_hidden_states,
                            req_ids=list(input_batch.req_ids),
                            query_start_loc=[
                                int(value)
                                for value in input_batch.query_start_loc_np[
                                    : input_batch.num_reqs + 1
                                ]
                            ],
                            num_scheduled_tokens=[
                                int(value) for value in input_batch.num_scheduled_tokens
                            ],
                            num_computed_tokens=[
                                int(value)
                                for value in input_batch.num_computed_tokens_np
                            ],
                            slot_mappings_by_layer=slot_mappings_by_layer,
                        )
                    # The trace is a graph-output observer, not a drafter input.
                    aux_hidden_states = None
            else:
                assert isinstance(model_output, torch.Tensor), (
                    "Target model must return a Tensor when auxiliary hidden "
                    "states are disabled; got "
                    f"{type(model_output).__name__}"
                    + (
                        f" with length {len(model_output)}"
                        if isinstance(model_output, (tuple, list))
                        else ""
                    )
                )
                hidden_states = model_output
                aux_hidden_states = None
            output_intermediate_tensors = None
        else:
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            aux_hidden_states = None
            output_intermediate_tensors = model_output

        routed_experts = None
        if not dummy_run and (capturer := self.routed_experts_capturer) is not None:
            assert slot_mappings is not None
            routed_experts = capturer.get_routed_experts(slot_mappings, num_toks)

        finished_req_ids = scheduler_output.finished_req_ids
        gdn_checkpoint_keys = (
            self.gdn_checkpoint_manager.save(
                scheduler_output,
                self.kv_cache_config,
                self.compilation_config.static_forward_context,
            )
            if not dummy_run and self.gdn_checkpoint_manager is not None
            else None
        )
        elastic_external_memory_bytes = 0
        elastic_external_memory_floor_bytes = 0
        # Generation finishes the elastic measurement only after sampling (and
        # MTP proposal, when enabled). Ending it here omits the sampler
        # high-water from the next-step feasibility receipt.
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            dp_sync=dp_sync,
            finished_req_ids=finished_req_ids,
            ec_connector_output=ec_connector_output,
            routed_experts=routed_experts,
            cudagraph_stats=cudagraph_stats,
            num_spec_tokens_to_schedule=(scheduler_output.num_spec_tokens_to_schedule),
            gdn_checkpoint_keys=gdn_checkpoint_keys,
            elastic_external_memory_bytes=elastic_external_memory_bytes,
            elastic_external_memory_floor_bytes=(elastic_external_memory_floor_bytes),
            elastic_mm_activation_loan_bytes=(
                scheduler_output.elastic_mm_activation_loan_bytes
            ),
            elastic_dynamic_graph_step_started=elastic_dynamic_graph_step_started,
            elastic_transaction_id=elastic_transaction_id,
            elastic_step_plan=elastic_plan,
            is_synthetic_warmup=is_synthetic_warmup,
        )

        if not self.is_last_pp_rank:
            # Non-last PP rank: return IntermediateTensors for sending.
            assert output_intermediate_tensors is not None
            assert self.pp_handler is not None
            return self.pp_handler.relay_aux_hidden_states(
                model_inputs["intermediate_tensors"], output_intermediate_tensors
            )
        return None

    @staticmethod
    def _expected_sampling_indices(input_batch: InputBatch) -> np.ndarray:
        """Reconstruct sampling rows from the authoritative CPU schedule."""
        pieces: list[np.ndarray] = []
        for req_idx in range(input_batch.num_reqs):
            logit_start = int(input_batch.cu_num_logits_np[req_idx])
            logit_end = int(input_batch.cu_num_logits_np[req_idx + 1])
            num_logits = logit_end - logit_start
            query_start = int(input_batch.query_start_loc_np[req_idx])
            query_end = int(input_batch.query_start_loc_np[req_idx + 1])
            row_start = query_end - num_logits
            if num_logits < 0 or row_start < query_start or row_start > query_end:
                raise RuntimeError(
                    "invalid CPU sampling-index contract: "
                    f"request={req_idx} query=[{query_start},{query_end}) "
                    f"logits=[{logit_start},{logit_end})"
                )
            pieces.append(np.arange(row_start, query_end, dtype=np.int64))
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(pieces)

    def _validate_elastic_calibration_sampling_indices(
        self,
        input_batch: InputBatch,
        *,
        hidden_rows: int,
        phase: str,
        only_single_token_prefill: bool = False,
    ) -> None:
        """Fail before unsafe advanced indexing during pre-READY calibration.

        A CUDA device-side bounds assertion poisons every rank and leaves the
        GPUs requiring an owner reset. Calibration requests can afford one
        synchronization here: compare the device-produced rows with the
        authoritative CPU schedule before launching ``IndexKernel``. The
        check is deliberately absent from the serving critical path.
        """
        if not input_batch.req_ids or not all(
            req_id.startswith("_elastic_cal_") for req_id in input_batch.req_ids
        ):
            return
        if only_single_token_prefill and not (
            input_batch.num_reqs == 1
            and input_batch.query_start_loc_np.tolist() == [0, 1]
            and input_batch.cu_num_logits_np.tolist() == [0, 1]
            and input_batch.req_ids[0].startswith("_elastic_cal_prefill_")
        ):
            return
        expected = self._expected_sampling_indices(input_batch)
        actual = input_batch.logits_indices.detach().cpu().numpy()
        device_query_start_loc = input_batch.query_start_loc.detach().cpu().tolist()
        device_cu_num_logits = input_batch.cu_num_logits.detach().cpu().tolist()
        if (
            actual.shape != expected.shape
            or not np.array_equal(actual, expected)
            or (
                actual.size
                and (int(actual.min()) < 0 or int(actual.max()) >= hidden_rows)
            )
        ):
            raise RuntimeError(
                "elastic calibration sampling indices failed closed before "
                "CUDA advanced indexing: "
                f"phase={phase} requests={input_batch.req_ids} "
                f"hidden_rows={hidden_rows} "
                f"query_start_loc={input_batch.query_start_loc_np.tolist()} "
                f"device_query_start_loc={device_query_start_loc} "
                f"cu_num_logits={input_batch.cu_num_logits_np.tolist()} "
                f"device_cu_num_logits={device_cu_num_logits} "
                f"idx_mapping_shape={tuple(input_batch.idx_mapping.shape)} "
                f"expected={expected.tolist()} actual={actual.tolist()}"
            )
        if only_single_token_prefill:
            logger.warning(
                "Elastic single-token prefill sampling boundary passed: "
                "phase=%s expected=%s actual=%s device_query_start_loc=%s "
                "device_cu_num_logits=%s idx_mapping_shape=%s",
                phase,
                expected.tolist(),
                actual.tolist(),
                device_query_start_loc,
                device_cu_num_logits,
                tuple(input_batch.idx_mapping.shape),
            )

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        dp_sync = self.execute_model_state.dp_sync
        finished_req_ids = self.execute_model_state.finished_req_ids
        ec_connector_output = self.execute_model_state.ec_connector_output
        routed_experts = self.execute_model_state.routed_experts
        cudagraph_stats = self.execute_model_state.cudagraph_stats
        num_spec_tokens_to_schedule = (
            self.execute_model_state.num_spec_tokens_to_schedule
        )
        gdn_checkpoint_keys = self.execute_model_state.gdn_checkpoint_keys
        elastic_external_memory_bytes = (
            self.execute_model_state.elastic_external_memory_bytes
        )
        elastic_external_memory_floor_bytes = (
            self.execute_model_state.elastic_external_memory_floor_bytes
        )
        elastic_mm_activation_loan_bytes = (
            self.execute_model_state.elastic_mm_activation_loan_bytes
        )
        is_synthetic_warmup = self.execute_model_state.is_synthetic_warmup
        dynamic_graph_step_started = (
            self.execute_model_state.elastic_dynamic_graph_step_started
        )
        elastic_transaction_id = self.execute_model_state.elastic_transaction_id
        elastic_step_plan = getattr(self.execute_model_state, "elastic_step_plan", None)
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
            # Will be adjusted for rejections if necessary in update_requests.
            self.postprocess_num_computed_tokens(input_batch)
            if not all_decode_next:
                # Might contain non-final prefill chunks, which will be scheduled
                # in the immediate next step (rather than in pp_size steps).
                self.model_state.postprocess_state(input_batch.idx_mapping, 0)

            # Post-step KV connector related operations.
            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            output = ModelRunnerOutput.with_kv_conn_output_only(
                kv_connector_output,
                elastic_external_memory_bytes,
                elastic_external_memory_floor_bytes,
            )
            output.elastic_mm_activation_loan_bytes = elastic_mm_activation_loan_bytes
            return ModelRunnerOutput.with_ec_conn_output(output, ec_connector_output)

        # Last rank: sample tokens
        hidden_states, input_batch = pcp.maybe_restore_pcp_for_sampling(
            self.pcp_manager, hidden_states, input_batch
        )
        self._validate_elastic_calibration_sampling_indices(
            input_batch,
            hidden_rows=int(hidden_states.shape[0]),
            phase="post_forward",
        )

        with record_function_or_nullcontext("ag2.target_sample_or_reject"):
            sampler_output, num_sampled, num_rejected = self.sample(
                hidden_states,
                input_batch,
                grammar_output,
                slot_mappings_by_layer,
            )

        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            cudagraph_stats=cudagraph_stats,
            gdn_checkpoint_keys=gdn_checkpoint_keys,
            elastic_external_memory_bytes=elastic_external_memory_bytes,
            elastic_external_memory_floor_bytes=(elastic_external_memory_floor_bytes),
            elastic_mm_activation_loan_bytes=elastic_mm_activation_loan_bytes,
        )
        # Start async output copy here so that it can overlap with speculator proposal.
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            check_ep_fault=self.check_ep_fault,
            routed_experts=routed_experts,
        )

        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        if self.speculator is not None and self.speculator.supports_mm_inputs:
            # Get cached multimodal embeddings for draft forward.
            # NOTE: This is done here because postprocess updates
            # num_computed_prefill_tokens.
            # The EAGLE/MTP drafter reads one position ahead of the target.
            # TODO(TheEpicDolphin): Gather MM embeddings for all speculative
            # steps during multi-module MTP.
            mm_inputs = self.model_state.gather_mm_embeddings(
                input_batch, draft_lookahead=1
            )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        with record_function_or_nullcontext("ag2.target_postprocess_sampled"):
            self.postprocess_sampled(
                input_batch.idx_mapping,
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch.query_start_loc,
            )

        draft_tokens_for_output: torch.Tensor | None = None
        if self.speculator is not None and num_spec_tokens_to_schedule > 0:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            with (
                use_workspace_lane(self._draft_workspace_lane),
                record_function_or_nullcontext("ag2.mtp_propose"),
            ):
                draft_tokens = self.speculator.propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings_by_layer,
                    spec_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    self.req_states.last_sampled_tokens,
                    self.req_states.next_prefill_tokens,
                    self.sampler.sampling_states.temperature.gpu,
                    self.sampler.sampling_states.seeds.gpu,
                    dp_sync=dp_sync,
                    mm_inputs=mm_inputs,
                    is_profile=is_synthetic_warmup,
                )
            if num_spec_tokens_to_schedule < draft_tokens.shape[1]:
                draft_tokens = draft_tokens[:, :num_spec_tokens_to_schedule]
            self.req_states.draft_tokens[
                input_batch.idx_mapping, : draft_tokens.shape[1]
            ] = draft_tokens
            if self.adaptive_verification is not None:
                self.adaptive_verification.record_confidences(
                    self.speculator.draft_token_confidence_probs, input_batch
                )
            draft_tokens_for_output = draft_tokens
        elif self.speculator is not None:
            # K=0 must avoid both drafter execution and stale draft publication.
            draft_tokens_for_output = self.req_states.draft_tokens[
                input_batch.idx_mapping, :0
            ]

        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            if draft_tokens_for_output is None:
                draft_tokens_for_output = self.req_states.draft_tokens[
                    input_batch.idx_mapping
                ]
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                draft_tokens_for_output,
            )
            if self.pp_handler is not None:
                self.pp_handler.broadcast_drafts(
                    self.req_states.draft_tokens, input_batch
                )

        self._settle_dynamic_graph_step_after_sampling(
            model_runner_output,
            dynamic_graph_step_started,
            elastic_transaction_id,
            elastic_step_plan,
        )

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.kv_connector_output = kv_connector_output
        model_runner_output.ec_connector_output = ec_connector_output

        return async_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.draft_tokens_handler.get_draft_tokens()

    @torch.inference_mode()
    @step_eplb_after()
    def pool(self) -> AsyncPoolingOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        ec_connector_output = self.execute_model_state.ec_connector_output
        elastic_external_memory_bytes = (
            self.execute_model_state.elastic_external_memory_bytes
        )
        elastic_external_memory_floor_bytes = (
            self.execute_model_state.elastic_external_memory_floor_bytes
        )
        elastic_mm_activation_loan_bytes = (
            self.execute_model_state.elastic_mm_activation_loan_bytes
        )
        self.execute_model_state = None

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)

        if not self.is_last_pp_rank:
            self.postprocess_num_computed_tokens(input_batch)
            output = ModelRunnerOutput.with_kv_conn_output_only(
                kv_connector_output,
                elastic_external_memory_bytes,
                elastic_external_memory_floor_bytes,
            )
            output.elastic_mm_activation_loan_bytes = elastic_mm_activation_loan_bytes
            return ModelRunnerOutput.with_ec_conn_output(output, ec_connector_output)

        assert self.pooling_runner is not None
        pooler_output, finished_mask = self.pooling_runner.pool(
            hidden_states, input_batch, self.req_states
        )

        # Build the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            kv_connector_output=kv_connector_output,
            ec_connector_output=ec_connector_output,
            elastic_external_memory_bytes=elastic_external_memory_bytes,
            elastic_external_memory_floor_bytes=(elastic_external_memory_floor_bytes),
            elastic_mm_activation_loan_bytes=elastic_mm_activation_loan_bytes,
        )
        async_output = AsyncPoolingOutput(
            model_runner_output=model_runner_output,
            pooler_output=pooler_output,
            finished_mask=finished_mask,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        self.postprocess_num_computed_tokens(input_batch)
        return async_output

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        # Update the number of computed tokens.
        post_update_num_computed_tokens(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

    def shutdown(self) -> None:
        """Release GPU tensors (model weights, KV caches, workspace) so that
        memory is reclaimable when running in the same process."""
        torch.accelerator.synchronize()
        self.cudagraph_manager = None
        if hasattr(self, "kv_caches"):
            self.kv_caches.clear()
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        self._prepared_attn_groups = None
        self._prepared_attn_config_signature = None
        self._prepared_kernel_block_sizes = None
        if hasattr(self, "kv_cache_config"):
            del self.kv_cache_config
        if hasattr(self, "model_state") and self.model_state.supports_mm_inputs:
            self.model_state.encoder_runner.clear()
        free_before_shutdown(self.vllm_config)
        if hasattr(self, "model_state"):
            del self.model_state
        # Detach the layer-level KV/state cache tensors before dropping the
        # models; the model objects can outlive this runner.
        speculator = getattr(self, "speculator", None)
        if speculator is not None:
            if draft_model := getattr(speculator, "model", None):
                clear_layer_kv_caches(draft_model.modules())
            self.speculator = None
        if hasattr(self, "model"):
            clear_layer_kv_caches(self.model.modules())
            del self.model

        gc.collect()
        torch.accelerator.empty_cache()
        logger.debug("Cleaned up model weights, KV caches, and workspace")

    ########### EPLB methods start ###########
    @property
    def eplb_state(self):
        return self.eplb.state

    @eplb_state.setter
    def eplb_state(self, state) -> None:
        self.eplb.state = state

    @property
    def eep_eplb_suppressed(self) -> bool:
        return self.eplb.suppressed

    @eep_eplb_suppressed.setter
    def eep_eplb_suppressed(self, suppressed: bool) -> None:
        self.eplb.suppressed = suppressed

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
    ) -> None:
        self.eplb.setup_from_mapping(
            self.model,
            self.model_config,
            expanded_physical_to_logical,
        )

    ########### EPLB methods end ###########

    # Out-of-tree hardware runners can select a PCP manager class.
    @property
    def pcp_manager_cls(self) -> type[pcp.PCPManager]:
        return pcp.PCPManager


class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    dp_sync: DPSyncState | None
    finished_req_ids: set[str]
    ec_connector_output: ECConnectorOutput | None
    routed_experts: RoutedExpertsTensors | None
    cudagraph_stats: CUDAGraphStat | None
    num_spec_tokens_to_schedule: int
    gdn_checkpoint_keys: tuple[bytes, ...] | None
    elastic_external_memory_bytes: int
    elastic_external_memory_floor_bytes: int
    elastic_mm_activation_loan_bytes: int
    elastic_dynamic_graph_step_started: bool
    elastic_transaction_id: str | None
    elastic_step_plan: ElasticStepPlan | None
    is_synthetic_warmup: bool


class BatchReqState(NamedTuple):
    """CPU request state for a scheduled batch, in batch (sorted) order."""

    req_ids: list[str]
    num_scheduled_tokens: np.ndarray  # [num_reqs]
    # May be less than scheduler_output.total_num_scheduled_tokens:
    # adaptive verification trims the draft budget before running.
    num_tokens: int
    idx_mapping_np: np.ndarray  # [num_reqs]
    prefill_len_np: np.ndarray  # [num_reqs]
    num_computed_prefill_tokens_np: np.ndarray  # [num_reqs]
    is_prefilling_np: np.ndarray  # [num_reqs]
    has_prefill: bool


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int],
    decode_query_len: int,
    *,
    is_prefilling_by_req: dict[str, bool] | None = None,
) -> list[str]:
    # Order completed-prefill requests before prompt-prefill requests, then
    # retain the established shape order inside each lifecycle cohort.
    # split_decodes_and_prefills relies on uniform target decodes
    # (query_len == decode_query_len) leading, while semantic DCP prefill needs
    # actual prompt rows to form one suffix even when a chunk is decode-sized.
    if is_prefilling_by_req is not None:
        missing = num_tokens_per_req.keys() - is_prefilling_by_req.keys()
        extra = is_prefilling_by_req.keys() - num_tokens_per_req.keys()
        if missing or extra:
            raise ValueError(
                "Batch lifecycle identity does not match scheduled requests: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    def key(req_id: str) -> tuple[bool, bool, int]:
        num_tokens = num_tokens_per_req[req_id]
        return (
            False if is_prefilling_by_req is None else is_prefilling_by_req[req_id],
            num_tokens != decode_query_len,
            num_tokens,
        )

    return sorted(num_tokens_per_req, key=key)
