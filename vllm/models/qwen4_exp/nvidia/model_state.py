# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

from .expert_offload_moe import NativeOffloadedExperts
from .ple_layer import Qwen4ExpNGramEmbedding
from .ple_offload import MmapPLEEmbedding


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._initialize_native_providers(vllm_config, model)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        self._mmap_ple_modules: tuple[Qwen4ExpNGramEmbedding, ...] = ()
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self._initialize_mmap_staging(vllm_config, model)

    def _initialize_native_providers(self, vllm_config, model):
        owners = tuple(
            m for m in model.modules() if isinstance(m, NativeOffloadedExperts)
        )
        declared = tuple(
            m
            for m in vllm_config.compilation_config.static_forward_context.values()
            if isinstance(m, NativeOffloadedExperts)
        )
        if {id(m) for m in owners} != {id(m) for m in declared} or any(
            not m.loaded for m in owners
        ):
            raise RuntimeError(
                "native expert model/forward-context inventories disagree"
            )
        self._native_providers = tuple(
            {id(m.provider): m.provider for m in owners}.values()
        )

    def _prepare_native_experts(self, *, dummy, num_tokens=None, input_batch=None):
        identity = (
            None
            if input_batch is None
            else dict(
                req_ids=list(input_batch.req_ids),
                num_reqs=input_batch.num_reqs,
                tokens=input_batch.num_tokens,
                physical_tokens=input_batch.num_tokens_after_padding,
                phase="prefill_or_mixed" if input_batch.has_prefill else "decode",
                query_start_loc=input_batch.query_start_loc_np[
                    : input_batch.num_reqs + 1
                ].tolist(),
                computed=input_batch.num_computed_tokens_np.tolist(),
                scheduled=input_batch.num_scheduled_tokens.tolist(),
            )
        )
        for provider in getattr(self, "_native_providers", ()):
            provider.prepare_execution(
                dummy=dummy, num_tokens=num_tokens, identity=identity
            )

    def finish_native_experts(self, *, dummy):
        for provider in self._native_providers:
            provider.finish_execution(dummy=dummy)

    def resolve_cudagraph_mode(self, mode: CUDAGraphMode) -> CUDAGraphMode:
        if mode == CUDAGraphMode.NONE or not (
            getattr(self, "_native_providers", ())
            or getattr(self, "_mmap_ple_modules", ())
        ):
            return super().resolve_cudagraph_mode(mode)
        from vllm.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled,
        )

        if not is_breakable_cudagraph_enabled():
            raise RuntimeError(
                "FlashNext host weight providers require breakable Graphs"
            )
        # Host demand resolution and staging are ordered eager callbacks between
        # GPU graph segments. Attention's FULL support cannot cover these edges.
        return CUDAGraphMode.PIECEWISE

    def _initialize_mmap_staging(self, vllm_config, model):
        modules = tuple(
            m
            for m in model.modules()
            if isinstance(m, Qwen4ExpNGramEmbedding)
            and isinstance(m.ngram_embedding, MmapPLEEmbedding)
        )
        declared = tuple(
            getattr(m, "ple_embedding", None)
            for m in vllm_config.compilation_config.static_forward_context.values()
        )
        declared_ids = {
            id(m)
            for m in declared
            if isinstance(m, Qwen4ExpNGramEmbedding)
            and isinstance(m.ngram_embedding, MmapPLEEmbedding)
        }
        if {id(m) for m in modules} != declared_ids:
            raise RuntimeError(
                "PLE staging module/forward-context inventories disagree"
            )
        for module in modules:
            module.ngram_embedding.initialize_staging(
                self.max_num_tokens, module.ngram_heads, self.device
            )
        self._mmap_ple_modules = modules

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        self._prepare_native_experts(
            dummy=False,
            num_tokens=input_batch.num_tokens_after_padding,
            input_batch=input_batch,
        )
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        context = self._prepare_ngram_context(input_batch, req_states)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=context,
        )
        for module in self._mmap_ple_modules:
            module.prepare_mmap_rows(
                input_batch.input_ids[: input_batch.num_tokens],
                query_start_loc[: input_batch.num_reqs + 1],
                context[: input_batch.num_reqs],
                input_batch.num_tokens_after_padding,
            )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        self._prepare_native_experts(dummy=True, num_tokens=num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        model_inputs.update(self._prepare_dummy_ple(num_reqs, num_tokens))
        return model_inputs

    def _prepare_dummy_ple(self, num_reqs, num_tokens):
        if (
            not 0 <= num_reqs <= self.max_num_reqs
            or not 0 <= num_tokens <= self.max_num_tokens
            or (num_reqs == 0 and num_tokens != 0)
        ):
            raise ValueError("invalid PLE dummy batch dimensions")
        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, max(1, num_reqs))
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        for module in self._mmap_ple_modules:
            module.ngram_embedding.prepare_dummy(num_tokens)
        return dict(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )

    def prepare_runtime_dummy_inputs(self, input_batch, req_states):
        model_inputs = super().prepare_inputs(input_batch, req_states)
        self._prepare_native_experts(
            dummy=True, num_tokens=input_batch.num_tokens_after_padding
        )
        if self.uses_ngram_embedding:
            model_inputs.update(
                self._prepare_dummy_ple(
                    input_batch.num_reqs_after_padding,
                    input_batch.num_tokens_after_padding,
                )
            )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
