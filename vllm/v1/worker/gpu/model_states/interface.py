# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.interfaces import (
    SupportsEncoderCudaGraph,
    supports_encoder_cudagraph,
)
from vllm.tasks import GenerationTask
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class ModelSpecificAttnMetadata:
    """Base class for model-specific attention metadata."""

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {}


class ModelState(ABC):
    supports_prompt_embeds: ClassVar[bool] = False
    """Whether this state implements user-provided prompt embeddings."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device

        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
        self.dtype = self.model_config.dtype

        self.supports_mm_inputs = encoder_cache is not None
        if encoder_cache is not None:
            enable_encoder_cuda_graph = (
                not self.model_config.enforce_eager
                and vllm_config.compilation_config.cudagraph_mm_encoder
                and supports_encoder_cudagraph(model)
            )
            cudagraph_manager = (
                EncoderCudaGraphManager(
                    vllm_config=vllm_config,
                    device=device,
                    dtype=self.dtype,
                    model=cast(SupportsEncoderCudaGraph, model),
                )
                if enable_encoder_cuda_graph
                else None
            )

            self.encoder_cache = encoder_cache
            observability_config = vllm_config.observability_config
            self.encoder_runner = EncoderRunner(
                model=self.model,
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                encoder_cache=encoder_cache,
                dtype=self.dtype,
                device=self.device,
                cudagraph_manager=cudagraph_manager,
                enable_timing=bool(
                    observability_config
                    and observability_config.enable_mm_processor_stats
                ),
            )

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        from vllm.model_executor.models.interfaces import (
            supports_realtime,
            supports_transcription,
        )
        from vllm.model_executor.models.interfaces_base import is_text_generation_model

        supported_tasks = list[GenerationTask]()
        if is_text_generation_model(self.model):
            supported_tasks.append("generate")
        if supports_transcription(self.model):
            if self.model.supports_transcription_only:
                return ("transcription",)
            supported_tasks.append("transcription")
        if supports_realtime(self.model):
            supported_tasks.append("realtime")
        return tuple(supported_tasks)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        return None

    def remove_request(self, req_id: str) -> None:
        return None

    def apply_staged_writes(self) -> None:
        return None

    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
        """Cudagraph support of attention groups this ModelState builds outside
        ``init_attn_backend`` (e.g. encoder-only layers).

        Returns the minimum support level and its backend name. The default of
        ``ALWAYS`` imposes no extra constraint on the runner's cudagraph mode.
        """
        return AttentionCGSupport.ALWAYS, None

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Hook run on real batches before the forward pass (after block tables
        are gathered). Used by mamba "align" prefix caching to pre-copy state
        across block boundaries. No-op by default."""
        return None

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        return None

    @abstractmethod
    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        """Prepare the ``inputs_embeds`` tensor for the current forward pass."""
        raise NotImplementedError

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor | None:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        return None

    def execute_mm_encoder(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> None:
        """Run the multi-modal encoder and cache its outputs by `mm_hash`.

        The encode half of `get_mm_embeddings`, without the gather, for callers
        that run no language model.
        """
        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            with self.encoder_runner.timed_encoder_operation(
                scheduled_encoder_inputs.keys()
            ):
                encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

    def stage_mm_encoder(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        req_ids: list[str] | None = None,
    ) -> tuple[list[str], list[Any]]:
        """Build rank-local encoder inputs before entering model collectives."""
        return self.encoder_runner.stage_mm_encoder_batches(scheduled_encoder_inputs)

    def execute_staged_mm_encoder(self, staged: tuple[list[str], list[Any]]) -> None:
        """Execute an all-rank-converged encoder stage exactly once."""
        completed = self.execute_staged_mm_encoder_collective(staged)
        self.commit_staged_mm_encoder(completed)

    def execute_staged_mm_encoder_collective(
        self, staged: tuple[list[str], list[Any]]
    ) -> tuple[list[str], list[tuple[int, object]]]:
        """Run only the common encoder collective/model phase."""
        mm_hashes, batches = staged
        grouped_outputs = (
            self.encoder_runner.execute_staged_mm_encoder_batches(batches)
            if batches
            else []
        )
        return mm_hashes, grouped_outputs

    def commit_staged_mm_encoder(
        self, completed: tuple[list[str], list[tuple[int, object]]]
    ) -> None:
        """Commit rank-local encoder outputs after the common model phase."""
        mm_hashes, grouped_outputs = completed
        encoder_outputs = self.encoder_runner.finalize_staged_mm_encoder_outputs(
            grouped_outputs
        )
        self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

    def validate_staged_mm_encoder(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        staged: tuple[list[str], list[Any]],
        req_ids: list[str] | None = None,
    ) -> None:
        """Bind rank-local cache state to scheduler-declared encoder work."""
        staged_hashes, staged_batches = staged
        expected: list[tuple[str, str]] = []
        for req_id in req_ids or scheduled_encoder_inputs:
            input_ids = scheduled_encoder_inputs.get(req_id, [])
            if not input_ids:
                continue
            features = self.encoder_cache.mm_features[req_id]
            for input_id in input_ids:
                feature = features[input_id]
                if feature.data is not None:
                    expected.append((feature.identifier, feature.modality))
        staged_modalities = [
            modality
            for modality, num_items, _kwargs in staged_batches
            for _ in range(num_items)
        ]
        observed = list(zip(staged_hashes, staged_modalities, strict=True))
        if observed != expected:
            raise RuntimeError(
                "elastic MM encoder staging differs from scheduler-declared "
                f"work: expected={expected!r} observed={observed!r}"
            )

    def stage_mm_embeddings(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> Any:
        """Perform local post-encoder gathering before TP token embedding."""
        return None

    def execute_staged_mm_embeddings(
        self, staged: Any, input_batch: InputBatch
    ) -> torch.Tensor | None:
        """Run the common token-embedding phase for a staged MM batch."""
        return None

    def commit_staged_mm_embeddings(
        self, embeddings: Any, input_batch: InputBatch
    ) -> torch.Tensor | None:
        """Commit post-collective embeddings to graph-stable input storage."""
        return embeddings

    def validate_mm_cache_readiness(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> None:
        """Fail locally before any rank enters the MM encoder phase."""
        self.encoder_runner.validate_cache_readiness(
            scheduled_encoder_inputs,
            input_batch.req_ids,
            input_batch.num_scheduled_tokens,
            input_batch.prefill_len_np,
            input_batch.num_computed_tokens_np,
        )

    def validate_elastic_mm_embedding_split(
        self, input_batch: InputBatch | None = None
    ) -> None:
        """Require an explicit collective/local-tail boundary for elastic MM."""
        if not callable(
            getattr(self.model, "embed_text_input_ids_for_elastic", None)
        ) or not callable(
            getattr(self.model, "merge_multimodal_embeddings_for_elastic", None)
        ):
            raise RuntimeError(
                "elastic MM execution requires model-specific split text "
                "embedding and multimodal merge hooks"
            )
        if input_batch is not None and (
            input_batch.num_tokens_after_padding
            > self.encoder_runner.inputs_embeds.shape[0]
        ):
            raise RuntimeError(
                "elastic MM graph input exceeds the persistent embedding buffer"
            )

    def gather_mm_embeddings(
        self, input_batch: InputBatch, draft_lookahead: int = 0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Gather cached multimodal embeddings."""
        return self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            input_batch.prefill_len_np,
            input_batch.num_computed_tokens_np,
            draft_lookahead=draft_lookahead,
        )

    @abstractmethod
    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_runtime_dummy_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        """Prepare a profile/empty runtime batch, without real request-only I/O."""
        return self.prepare_inputs(input_batch, req_states)

    @abstractmethod
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        """Wrap or replace the default sampler.

        Called after model loading with the already-constructed base
        ``Sampler``.  Return ``None`` to keep the defaults, or
        ``(sampler, rejection_sampler | None)`` to override.
        """
        return None

    num_new_sampled_tokens_per_step: int = 1
    """New tokens sampled on each decode step 
    (excluding accepted draft tokens, a.k.a num bonus tokens)."""
