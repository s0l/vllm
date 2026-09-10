# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

from typing import ClassVar, cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from .indexer_qsa import QSAIndexer


@eager_break_during_capture
def _qsa_fp8_owner(
    projected_qk: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    caches: list[torch.Tensor],
    selection: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._run_fp8_qsa(
        projected_qk, positions, query, key, value, caches, selection, output
    )


def _qsa_fp8_owner_fake(
    projected_qk: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    caches: list[torch.Tensor],
    selection: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    pass


direct_register_custom_op(
    "qsa_fp8_owner",
    _qsa_fp8_owner,
    mutates_args=["caches", "selection", "output"],
    fake_impl=_qsa_fp8_owner_fake,
)


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
        )
        return output


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        extra = vllm_config.additional_config
        self.fp8_dcp = isinstance(extra, dict) and bool(extra.get("flashnext_qsa_dcp"))
        self.supports_dcp = self.fp8_dcp
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        allowed_cache = ("fp8", "fp8_e4m3") if self.fp8_dcp else ("auto", "bfloat16")
        if cache_config.cache_dtype not in allowed_cache:
            raise NotImplementedError(
                f"Qwen4Exp QSA requires cache dtype {allowed_cache}"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if not self.fp8_dcp and (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        if self.fp8_dcp and (
            tp_size != parallel_config.tensor_parallel_size
            or parallel_config.decode_context_parallel_size != tp_size
            or parallel_config.prefill_context_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.enable_dbo
            or self.hidden_size != 2560
            or (config.num_attention_heads, config.num_key_value_heads, config.head_dim)
            != (24, 2, 256)
            or (config.indexer_budget, config.indexer_compress_ratio) != (2048, 4)
            or vllm_config.attention_config.resolve_indexer_kv_dtype("bf16") != "bf16"
        ):
            raise NotImplementedError(
                "FlashNext FP8 QSA requires matching TP/DCP groups and model geometry"
            )
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.fp8_dcp:
            self.num_kv_heads = self.total_num_kv_heads
        elif self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        if not self.fp8_dcp:
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        if self.fp8_dcp:
            from .qsa_qkv import QSAOwnedQKVLinear

            self.qkv_proj = QSAOwnedQKVLinear(
                self.hidden_size,
                self.total_num_heads,
                self.total_num_kv_heads,
                self.head_dim,
                tp_size=tp_size,
                tp_rank=get_tensor_model_parallel_rank(),
                prefix=f"{prefix}.qkv_proj",
            )
        else:
            from .model import without_modelopt_fp4

            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.total_num_heads * (1 + self.attn_output_gate),
                self.total_num_kv_heads,
                bias=False,
                quant_config=without_modelopt_fp4(quant_config),
                prefix=f"{prefix}.qkv_proj",
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.fp8_dcp:
            # The generic helper returns byte storage for FP8; FI consumes
            # typed e4m3 views over those same bytes.
            self.kv_cache_torch_dtype = torch.float8_e4m3fn
        if not self.fp8_dcp and self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 cache storage")
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend: type[AttentionBackend]
        if self.fp8_dcp:
            from .qsa_flashinfer import QSAFlashInferBackend, QSAFlashInferImpl

            self.attn_backend = QSAFlashInferBackend
            self.impl = QSAFlashInferImpl(
                vllm_config,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
            )
        else:
            self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
            self.impl = Qwen4ExpQSAFlashAttentionImpl(
                self.num_heads,
                self.head_dim,
                self.scaling,
                self.num_kv_heads,
                None,
                None,
                self.kv_cache_dtype,
                None,
                AttentionType.DECODER,
                None,
            )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        if self.fp8_dcp:
            from .qsa_flashinfer import page_one_view

            page_one_view(kv_cache, self.head_dim)
            if kv_cache.shape[1] != self.num_kv_heads:
                raise ValueError("QSA cache must contain both global KV heads")
        self.kv_cache = kv_cache

    def _run_fp8_qsa(
        self, projected_qk, positions, query, key, value, caches, selection, output
    ) -> None:
        from vllm import _custom_ops as ops

        from .qsa_flashinfer import (
            QSAFlashInferImpl,
            QSAFlashInferMetadata,
            page_one_view,
        )

        impl = cast(QSAFlashInferImpl, self.impl)
        impl.pool.prepare()
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        output.zero_()
        if not isinstance(metadata, dict):
            return
        if len(caches) != 3 or any(cache.numel() == 0 for cache in caches):
            raise RuntimeError("QSA main and side caches must be bound")
        page_one_view(caches[0], self.head_dim)
        main = cast(QSAFlashInferMetadata, metadata[self.layer_name])
        side = cast(QSAForwardMetadata, metadata[self.indexer.raw_key_cache.prefix])
        count = main.num_actual_tokens
        if count < 0 or count != side.num_actual_tokens or count > query.shape[0]:
            raise RuntimeError("QSA main and side token counts disagree")
        if selection.shape != (query.shape[0], self.indexer.packed_output_width):
            raise ValueError("QSA selection has an invalid shape")
        if count == 0:
            return
        # Bind the actual custom-op operands, which AOT may functionalize.
        # Restore owner pointers even if the indexer rejects its metadata.
        raw, compressed = self.indexer.raw_key_cache, self.indexer.compressed_key_cache
        originals = raw.kv_cache, compressed.kv_cache
        try:
            raw.bind_kv_cache(caches[1].transpose(1, 2))
            compressed.bind_kv_cache(caches[2].transpose(1, 2))
            self.indexer(projected_qk, positions, selection[:count])
            key_cache, value_cache = caches[0].transpose(1, 2).split(self.head_dim, -1)
            ops.reshape_and_cache_flash(
                key[:count],
                value[:count],
                key_cache,
                value_cache,
                main.slot_mapping[:count],
                self.kv_cache_dtype,
                self._k_scale,
                self._v_scale,
            )
            impl.forward_qsa(
                self,
                query[:count],
                caches[0],
                main,
                output[:count],
                selection[:count],
                side.token_to_req[:count],
            )
        finally:
            raw.bind_kv_cache(originals[0].transpose(1, 2))
            compressed.bind_kv_cache(originals[1].transpose(1, 2))

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    @eager_break_during_capture
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        return_ag2_mtp_trace: bool = False,
        *,
        return_tp_partial: bool = False,
    ) -> torch.Tensor:
        if return_ag2_mtp_trace or return_tp_partial:
            raise ValueError("QSA owner does not expose dense attention trace carriers")
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        # Keep the index projection outside the eager break.
        projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        if self.fp8_dcp:
            torch.ops.vllm.qsa_fp8_owner(
                projected_qk,
                positions,
                query,
                key,
                value,
                [
                    self.kv_cache,
                    self.indexer.raw_key_cache.kv_cache,
                    self.indexer.compressed_key_cache.kv_cache,
                ],
                self.topk_indices_buffer[:num_tokens],
                attn_output,
                self.layer_name,
            )
        else:
            self._run_qsa(
                projected_qk,
                positions,
                query,
                key,
                value,
                attn_output,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
]
