# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

import os
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_owner_residual_arc_prequant,
    tensor_model_parallel_owner_materialize_aux,
    tensor_model_parallel_owner_terminal_norm,
    tensor_model_parallel_reduce_scatter,
    tensor_model_parallel_unified_exact_all_reduce,
    tensor_model_parallel_v1_block5_fused_add_rms_norm,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.head_partition import (
    make_attention_head_partition,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.fusion.quant_activation import (
    GDNQuantizedActivations,
    QuantizedActivation,
)
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    QKVParallelLinearOverlappingGQA,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


def _ag2_tp3_owner_prequant_enabled(
    vllm_config: VllmConfig,
    prefix: str,
    *,
    is_moe_layer: bool,
) -> bool:
    """Resolve and fail-close the target-only owner-prequant experiment."""
    if os.environ.get("AG2_VLLM_TP3_OWNER_PREQUANT", "0") != "1":
        return False
    # The current ARC sidecar describes target layers only. MTP deliberately
    # stays on its accepted exact-reduce path until it has its own proof.
    if ".mtp.layers." in f".{prefix}" or prefix.startswith("mtp.layers."):
        return False
    parallel = vllm_config.parallel_config
    if (
        parallel.tensor_parallel_size != 3
        or parallel.pipeline_parallel_size != 1
        or parallel.use_sequence_parallel_moe
        or is_moe_layer
    ):
        raise RuntimeError(
            "TP3 owner prequant requires dense TP=3, PP=1 without sequence parallelism"
        )
    if os.environ.get("VLLM_TP3_CE_REDUCE", "0") != "1":
        raise RuntimeError(
            "TP3 owner prequant requires CE (compressed all-reduce) enabled"
        )
    if not os.environ.get("AG2_VLLM_NVFP4_ARC_SIDECAR_DIR", "").strip():
        raise RuntimeError("TP3 owner prequant requires the ARC sidecar")
    boolean_trace_switches = (
        "AG2_VLLM_AUX_HIDDEN_TRACE_ALL_INTERNAL_BOUNDARIES",
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ALL_BOUNDARIES",
        "AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_ATTENTION_BOUNDARY",
        "AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_GDN_BOUNDARIES",
        "AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_BOUNDARIES",
        "AG2_VLLM_AUX_HIDDEN_TRACE_GDN_REPLAY_ONLY",
    )
    payload_trace_switches = (
        "AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT",
        "AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS",
        "AG2_VLLM_AUX_HIDDEN_TRACE_INTERNAL_BOUNDARY_LAYERS",
        "AG2_VLLM_AUX_HIDDEN_TRACE_SEQUENCE_BOUNDARY_LAYERS",
        "AG2_VLLM_LAYER0_TRACE_OUTPUT",
    )
    boundary_trace_switches = (
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_LAYER",
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_GDN_BOUNDARY_LAYER",
    )
    incompatible = sorted(
        [name for name in boolean_trace_switches if os.environ.get(name, "0") == "1"]
        + [name for name in payload_trace_switches if os.environ.get(name, "")]
        + [
            name
            for name in boundary_trace_switches
            if int(os.environ.get(name, "-1")) >= 0
        ]
    )
    if incompatible:
        raise RuntimeError(
            "TP3 owner prequant is incompatible with diagnostic traces: "
            + ", ".join(incompatible)
        )
    return True


def _ag2_tp3_mtp_block5_enabled(
    vllm_config: VllmConfig,
    prefix: str,
) -> bool:
    if os.environ.get("AG2_VLLM_TP3_MTP_BLOCK5_RMS", "0") != "1":
        return False
    is_mtp = ".mtp.layers." in f".{prefix}" or prefix.startswith("mtp.layers.")
    if not is_mtp:
        return False
    if os.environ.get("AG2_VLLM_TP3_OWNER_PREQUANT", "0") != "1":
        raise RuntimeError("MTP block5 RMS requires target owner-prequant")
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 3 or parallel.pipeline_parallel_size != 1:
        raise RuntimeError("MTP block5 RMS requires TP=3 and PP=1")
    if os.environ.get("VLLM_TP3_CE_REDUCE", "0") != "1":
        raise RuntimeError("MTP block5 RMS requires CE (compressed all-reduce)")
    return True


KVCache = tuple[torch.Tensor, torch.Tensor]

AG2_COMPACT_FULL_ATTENTION_STAGE_ORDER = (
    "qkv",
    "q",
    "k",
    "v",
    "gate",
    "dcp_output_pack",
    "dcp_lse_pack",
    "core",
    "gated",
    "output_parallel",
    "output",
)
AG2_COMPACT_GDN_STAGE_ORDER = (
    "qkvz",
    "ba",
    "core",
    "gated_norm",
    "output_parallel",
    "output",
)
AG2_COMPACT_MLP_STAGE_ORDER = (
    "gate_up",
    "activation",
    "down_parallel",
    "output",
)


def _ag2_selected_stages(env_name: str, stages: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(env_name, "")
    if not raw:
        return stages
    requested = {value for value in raw.split(",") if value}
    unknown = requested.difference(stages)
    if unknown:
        raise ValueError(f"Unknown {env_name} stages: {sorted(unknown)}")
    return tuple(stage for stage in stages if stage in requested)


def _ag2_compact_full_attention_stages() -> tuple[str, ...]:
    raw = os.environ.get(
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_FULL_ATTENTION_STAGES",
        "",
    )
    if not raw:
        return AG2_COMPACT_FULL_ATTENTION_STAGE_ORDER
    requested = {value for value in raw.split(",") if value}
    unknown = requested.difference(AG2_COMPACT_FULL_ATTENTION_STAGE_ORDER)
    if unknown:
        raise ValueError(
            f"Unknown compact full-attention trace stages: {sorted(unknown)}"
        )
    return tuple(
        stage for stage in AG2_COMPACT_FULL_ATTENTION_STAGE_ORDER if stage in requested
    )


def _ag2_internal_trace_layer_enabled(layer_idx: int) -> bool:
    if (
        os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_ALL_INTERNAL_BOUNDARIES",
            "0",
        )
        == "1"
    ):
        return True
    return layer_idx in {
        int(value)
        for value in os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_INTERNAL_BOUNDARY_LAYERS",
            "",
        ).split(",")
        if value
    }


def _is_shared_expert_fse_compatible(quant_config) -> bool:
    """Check if shared expert can be fused with routed experts.

    FSE requires that shared and routed expert weights use the same
    quantization format. Returns False when the shared expert is
    excluded from quantization (e.g. float32 shared in an MXFP4 model)
    or has a different quant spec than routed experts.
    """
    if quant_config is None:
        return True
    # Quark stores its full config dict in quant_config.quant_config
    raw_config = getattr(quant_config, "quant_config", None)
    if not isinstance(raw_config, dict):
        return True
    exclude = raw_config.get("exclude", [])
    if not exclude:
        return True
    return not any("shared_expert." in str(e) for e in exclude)


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        _fse_requested = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        _fse_enabled = _fse_requested and _is_shared_expert_fse_compatible(quant_config)
        if _fse_requested and not _fse_enabled:
            logger.warning(
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
                "shared expert has a different quantization spec than routed "
                "experts. Falling back to non-fused shared expert path."
            )
        if _fse_enabled or config.shared_expert_intermediate_size <= 0:
            self.shared_expert = None
        else:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the MoERunner class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.attn_head_partition = None
        use_overlapping_gqa = (
            self.total_num_kv_heads % tp_size != 0
            and tp_size % self.total_num_kv_heads != 0
        )
        vllm_config = get_current_vllm_config_or_none()
        dcp_size = (
            vllm_config.parallel_config.decode_context_parallel_size
            if vllm_config is not None
            else 1
        )
        self.dcp_full_kv_attention_heads = use_overlapping_gqa and dcp_size > 1
        if self.dcp_full_kv_attention_heads:
            self.attn_head_partition = make_attention_head_partition(
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            self.num_kv_heads = self.total_num_kv_heads
        elif use_overlapping_gqa:
            self.attn_head_partition = make_attention_head_partition(
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            self.num_kv_heads = self.attn_head_partition.num_kv_heads
        elif self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        logger.info_once(
            "Qwen full-attention layout: tp=%d dcp=%d q_heads=%d "
            "kv_heads=%d overlapping_gqa=%s full_kv=%s local_kv_heads=%d",
            tp_size,
            dcp_size,
            self.total_num_heads,
            self.total_num_kv_heads,
            use_overlapping_gqa,
            self.dcp_full_kv_attention_heads,
            self.num_kv_heads,
        )
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        self._ag2_tp3_unified_exact_reduce = (
            os.environ.get("AG2_VLLM_TP3_UNIFIED_EXACT_REDUCE", "0") == "1"
            and reduce_results
            and tp_size == 3
        )

        qkv_proj_cls = (
            QKVParallelLinearOverlappingGQA
            if use_overlapping_gqa or self.dcp_full_kv_attention_heads
            else QKVParallelLinear
        )
        qkv_kwargs = {}
        if self.dcp_full_kv_attention_heads:
            qkv_kwargs["kv_head_indices"] = tuple(range(self.total_num_kv_heads))
        elif self.attn_head_partition is not None:
            qkv_kwargs["kv_head_indices"] = self.attn_head_partition.kv_head_indices
        self.qkv_proj = qkv_proj_cls(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            **qkv_kwargs,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=(reduce_results and not self._ag2_tp3_unified_exact_reduce),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        # Late-interaction retrieval models (e.g. ColQwen3.5) run BIDIRECTIONAL
        # attention on the full_attention layers; they set config.is_causal=False
        # via a VerifyAndUpdateConfig handler. Generation models leave is_causal
        # unset (-> causal/DECODER), so this is a no-op for them. Mirrors qwen3.py.
        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )
        if (
            self.dcp_full_kv_attention_heads
            and not self.attn.attn_backend.supports_dcp_full_kv_attention_heads
        ):
            raise NotImplementedError(
                "The selected attention backend does not support DCP with "
                "full global KV heads. Use a backend that explicitly "
                "implements rank-local KV-head selection."
            )
        self.attn.dcp_full_kv_attention_heads = self.dcp_full_kv_attention_heads
        if self.dcp_full_kv_attention_heads:
            assert self.attn_head_partition is not None
            self.attn.dcp_local_kv_head_indices = (
                self.attn_head_partition.kv_head_indices
            )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Fuse the gated split + QK-RMSNorm + (partial) NeoX RoPE + gate copy.
        # TODO: support MRoPE
        mm_config = model_config.multimodal_config if model_config else None
        text_only = mm_config is None or mm_config.language_model_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )
        layer_idx = extract_layer_index(prefix)
        trace_layer = int(os.environ.get("AG2_VLLM_LAYER_TRACE_LAYER", "0"))
        self._ag2_full_attn_trace_enabled = (
            bool(os.environ.get("AG2_VLLM_LAYER0_TRACE_OUTPUT"))
            and layer_idx == trace_layer
        )
        self._ag2_aux_full_boundaries_enabled = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_BOUNDARIES", "0"
        ) == "1" and layer_idx == int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "0")
        )
        full_attention_stages_raw = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_STAGES", ""
        )
        self._ag2_aux_full_stages = tuple(
            stage
            for stage in (
                "qkv",
                "q",
                "k",
                "v",
                "dcp_output_pack",
                "dcp_lse_pack",
                "dcp_kv_history_pack",
                "dcp_kv_history_meta",
                "dcp_kv_page_indices_pack",
                "dcp_observer_meta",
                "dcp_current_kv_pack",
                "dcp_scale_pack",
                "core",
                "gated",
                "output_parallel",
                "output",
            )
            if (
                stage in full_attention_stages_raw.split(",")
                if full_attention_stages_raw
                else stage
                not in {
                    "dcp_kv_history_pack",
                    "dcp_kv_history_meta",
                    "dcp_kv_page_indices_pack",
                    "dcp_observer_meta",
                    "dcp_current_kv_pack",
                    "dcp_scale_pack",
                }
            )
        )
        self.o_proj._ag2_aux_output_parallel_enabled = (
            self._ag2_aux_full_boundaries_enabled
            and "output_parallel" in self._ag2_aux_full_stages
        )
        self.attn._ag2_aux_dcp_pack_enabled = (
            self._ag2_aux_full_boundaries_enabled
            and (
                "dcp_output_pack" in self._ag2_aux_full_stages
                or "dcp_lse_pack" in self._ag2_aux_full_stages
            )
        )
        self.attn._ag2_aux_dcp_kv_history_enabled = (
            self._ag2_aux_full_boundaries_enabled
            and any(
                stage in self._ag2_aux_full_stages
                for stage in (
                    "dcp_kv_history_pack",
                    "dcp_kv_history_meta",
                    "dcp_kv_page_indices_pack",
                    "dcp_observer_meta",
                    "dcp_current_kv_pack",
                    "dcp_scale_pack",
                )
            )
        )
        self.attn._ag2_aux_dcp_kv_history_tokens = int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_DCP_KV_HISTORY_TOKENS", "0")
        )
        self.attn._ag2_aux_dcp_kv_history_rows = int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_DCP_KV_HISTORY_ROWS", "0")
        )
        self.attn._ag2_aux_dcp_pack_rows = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS",
                "0",
            )
        )
        if self.attn._ag2_aux_dcp_pack_enabled and not {
            "dcp_output_pack",
            "dcp_lse_pack",
        }.issubset(self._ag2_aux_full_stages):
            raise ValueError(
                "DCP auxiliary output and LSE packs must be requested together"
            )
        if self.attn._ag2_aux_dcp_kv_history_enabled and not {
            "dcp_kv_history_pack",
            "dcp_kv_history_meta",
            "dcp_kv_page_indices_pack",
            "dcp_observer_meta",
            "dcp_current_kv_pack",
            "dcp_scale_pack",
        }.issubset(self._ag2_aux_full_stages):
            raise ValueError(
                "DCP causal observer history, pages and metadata must be "
                "requested together"
            )
        if self.attn._ag2_aux_dcp_kv_history_enabled and (
            self.attn._ag2_aux_dcp_kv_history_tokens < 1
            or self.attn._ag2_aux_dcp_kv_history_rows < 1
        ):
            raise ValueError(
                "DCP KV history trace requires positive row and token capacities"
            )
        if self._ag2_full_attn_trace_enabled:
            trace_rows = 3
            trace_prefix_tokens = 256
            qkv_size = self.q_size * (2 if self.attn_output_gate else 1)
            qkv_size += 2 * self.kv_size
            trace_buffers = {
                "qkv": (trace_rows, qkv_size),
                "q": (trace_rows, self.q_size),
                "k": (trace_rows, self.kv_size),
                "v": (trace_rows, self.kv_size),
                "gate": (trace_rows, self.q_size),
                "core": (trace_rows, self.q_size),
                "gated_output": (trace_rows, self.q_size),
                "attention_output": (trace_rows, self.hidden_size),
            }
            for name, shape in trace_buffers.items():
                self.register_buffer(
                    f"_ag2_trace_{name}",
                    torch.zeros(shape, dtype=model_config.dtype),
                    persistent=False,
                )
            self.o_proj.register_buffer(
                "_ag2_trace_output_parallel",
                torch.zeros(
                    (trace_rows, self.hidden_size),
                    dtype=model_config.dtype,
                ),
                persistent=False,
            )

            self.register_buffer(
                "_ag2_trace_prefix_positions",
                torch.full((trace_prefix_tokens,), -1, dtype=torch.long),
                persistent=False,
            )
            for name in ("k", "v"):
                self.register_buffer(
                    f"_ag2_trace_prefix_{name}",
                    torch.zeros(
                        (trace_prefix_tokens, self.kv_size),
                        dtype=model_config.dtype,
                    ),
                    persistent=False,
                )
            self.attn._ag2_dcp_trace_enabled = True
            dcp_trace_buffers = {
                "decode_local_output": (
                    trace_rows,
                    self.total_num_heads,
                    self.head_dim,
                ),
                "decode_local_lse": (trace_rows, self.total_num_heads),
                "decode_combined_output": (
                    trace_rows,
                    self.num_heads,
                    self.head_dim,
                ),
                "decode_combined_lse": (trace_rows, self.num_heads),
                "prefill_context_local_output": (
                    trace_rows,
                    self.total_num_heads,
                    self.head_dim,
                ),
                "prefill_context_local_lse": (
                    trace_rows,
                    self.total_num_heads,
                ),
                "prefill_context_combined_output": (
                    trace_rows,
                    self.num_heads,
                    self.head_dim,
                ),
                "prefill_context_combined_lse": (
                    trace_rows,
                    self.num_heads,
                ),
                "prefill_query_output": (
                    trace_rows,
                    self.num_heads,
                    self.head_dim,
                ),
                "prefill_query_lse": (trace_rows, self.num_heads),
                "prefill_merged_output": (
                    trace_rows,
                    self.num_heads,
                    self.head_dim,
                ),
            }
            for name, shape in dcp_trace_buffers.items():
                dtype = torch.float32 if name.endswith("_lse") else model_config.dtype
                self.attn.register_buffer(
                    f"_ag2_dcp_trace_{name}",
                    torch.zeros(shape, dtype=dtype),
                    persistent=False,
                )

    def ag2_enable_compact_full_trace(self) -> None:
        """Enable the bounded all-layer attention packet before compilation."""
        stages = _ag2_compact_full_attention_stages()
        if not stages:
            raise ValueError("Compact full-attention trace requires stages")
        if bool({"dcp_output_pack", "dcp_lse_pack"}.intersection(stages)) and not {
            "dcp_output_pack",
            "dcp_lse_pack",
        }.issubset(stages):
            raise ValueError(
                "Compact DCP output and LSE packs must be requested together"
            )
        self._ag2_aux_compact_full_stages = stages
        # The ordinary full-boundary producer shares this forward path. Keep
        # its consumer registry atomic with the compact registry; `gate` is
        # compact-only and comes from the returned trace mapping.
        self._ag2_aux_full_stages = tuple(stage for stage in stages if stage != "gate")
        self._ag2_aux_full_boundaries_enabled = True
        # return_ag2_mtp_trace publishes a shared attention mapping whose
        # output_parallel entry is constructed before the compact consumer
        # selects its requested stages. Keep that coupled producer initialized
        # even for a narrow stage set such as ("gated",).
        self.o_proj._ag2_aux_output_parallel_enabled = True
        self.attn._ag2_aux_dcp_pack_enabled = bool(
            {"dcp_output_pack", "dcp_lse_pack"}.intersection(stages)
        )
        self.attn._ag2_aux_dcp_pack_rows = int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS", "0")
        )
        if self.attn._ag2_aux_dcp_pack_enabled and self.attn._ag2_aux_dcp_pack_rows < 1:
            raise ValueError(
                "Full internal attention trace requires positive "
                "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS"
            )

    def ag2_enable_projection_calibration_capture(
        self, capacity: int, dtype: torch.dtype
    ) -> None:
        if capacity < 1:
            raise ValueError("projection capture capacity must be positive")
        self._ag2_projection_capture_enabled = True
        self.register_buffer(
            "_ag2_projection_capture_gated",
            torch.full((capacity, self.q_size), torch.nan, dtype=dtype),
            persistent=False,
        )

    def ag2_enable_sequence_full_trace(self) -> None:
        """Enable fixed, sequence-wide full-attention fingerprint producers."""
        self._ag2_aux_compact_full_stages = (
            "qkv",
            "q",
            "k",
            "v",
            "gate",
            "core",
            "gated",
            "output_parallel",
            "output",
        )
        self._ag2_aux_full_stages = tuple(
            stage for stage in self._ag2_aux_compact_full_stages if stage != "gate"
        )
        self._ag2_aux_full_boundaries_enabled = True
        self.o_proj._ag2_aux_output_parallel_enabled = True

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return post-norm, post-RoPE (q, k, v) and the pre-sigmoid gate.

        Dispatches between the fused Triton kernel and the eager
        split + QK-RMSNorm + RoPE path. ``gate`` is ``None`` when output
        gating is disabled.
        """
        if self.use_fused_qk_norm_rope_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            # mRoPE passes positions as (3, n_tokens) for T/H/W. Fusion is only
            # enabled text-only, where the three rows are identical, so taking
            # the T row is exact. (1D positions pass through.)
            pos = positions[0] if positions.ndim == 2 else positions
            q, k, gate = fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
                self.rotary_emb.cos_sin_cache,
                pos,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
            return q, k, v, gate

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        return_ag2_mtp_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mtp_trace: dict[str, torch.Tensor] = {}
        qkv, _ = self.qkv_proj(hidden_states)
        if return_ag2_mtp_trace:
            mtp_trace["qkv"] = qkv
        if self._ag2_aux_full_boundaries_enabled and "qkv" in self._ag2_aux_full_stages:
            self._ag2_aux_qkv = qkv
        if self._ag2_full_attn_trace_enabled:
            trace = qkv[:3]
            self._ag2_trace_qkv[: trace.shape[0]].copy_(trace)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        if return_ag2_mtp_trace:
            mtp_trace.update({"q": q, "k": k, "v": v})
            if gate is not None:
                mtp_trace["gate"] = gate
        if self._ag2_aux_full_boundaries_enabled:
            if "q" in self._ag2_aux_full_stages:
                self._ag2_aux_q = q
            if "k" in self._ag2_aux_full_stages:
                self._ag2_aux_k = k
            if "v" in self._ag2_aux_full_stages:
                self._ag2_aux_v = v
        if self._ag2_full_attn_trace_enabled:
            for name, value in (("q", q), ("k", k), ("v", v)):
                trace = value[:3]
                getattr(self, f"_ag2_trace_{name}")[: trace.shape[0]].copy_(trace)
            if gate is not None:
                trace = gate[:3]
                self._ag2_trace_gate[: trace.shape[0]].copy_(trace)
            position_values = positions[0] if positions.ndim == 2 else positions
            prefix_rows = min(k.shape[0], self._ag2_trace_prefix_k.shape[0])
            prefix_positions = position_values[:prefix_rows].clamp(
                0, self._ag2_trace_prefix_k.shape[0] - 1
            )
            self._ag2_trace_prefix_positions.index_copy_(
                0, prefix_positions, position_values[:prefix_rows]
            )
            self._ag2_trace_prefix_k.index_copy_(0, prefix_positions, k[:prefix_rows])
            self._ag2_trace_prefix_v.index_copy_(0, prefix_positions, v[:prefix_rows])
        attn_output = self.attn(
            q,
            k,
            v,
            ag2_dcp_trace=(
                return_ag2_mtp_trace or self._ag2_aux_full_boundaries_enabled
            ),
        )
        if return_ag2_mtp_trace:
            mtp_trace["core"] = attn_output
            dcp_trace_names: list[str] = []
            if self.attn._ag2_aux_dcp_pack_enabled:
                dcp_trace_names.extend(("output_pack", "lse_pack"))
            if self.attn._ag2_aux_dcp_kv_history_enabled:
                dcp_trace_names.extend(
                    (
                        "kv_history_pack",
                        "kv_history_meta",
                        "kv_page_indices_pack",
                        "observer_meta",
                        "current_kv_pack",
                        "scale_pack",
                    )
                )
            for name in dcp_trace_names:
                mtp_trace[f"dcp_{name}"] = getattr(self.attn, f"_ag2_aux_dcp_{name}")
        if self._ag2_aux_full_boundaries_enabled:
            if "dcp_output_pack" in self._ag2_aux_full_stages:
                self._ag2_aux_dcp_output_pack = self.attn._ag2_aux_dcp_output_pack
            if "dcp_lse_pack" in self._ag2_aux_full_stages:
                self._ag2_aux_dcp_lse_pack = self.attn._ag2_aux_dcp_lse_pack
            if "dcp_kv_history_pack" in self._ag2_aux_full_stages:
                self._ag2_aux_dcp_kv_history_pack = (
                    self.attn._ag2_aux_dcp_kv_history_pack
                )
            if "dcp_kv_history_meta" in self._ag2_aux_full_stages:
                self._ag2_aux_dcp_kv_history_meta = (
                    self.attn._ag2_aux_dcp_kv_history_meta
                )
            if "dcp_kv_page_indices_pack" in self._ag2_aux_full_stages:
                self._ag2_aux_dcp_kv_page_indices_pack = (
                    self.attn._ag2_aux_dcp_kv_page_indices_pack
                )
            if "dcp_observer_meta" in self._ag2_aux_full_stages:
                self._ag2_aux_dcp_observer_meta = self.attn._ag2_aux_dcp_observer_meta
            if "dcp_current_kv_pack" in self._ag2_aux_full_stages:
                self._ag2_aux_dcp_current_kv_pack = (
                    self.attn._ag2_aux_dcp_current_kv_pack
                )
            if "dcp_scale_pack" in self._ag2_aux_full_stages:
                self._ag2_aux_dcp_scale_pack = self.attn._ag2_aux_dcp_scale_pack
        if (
            self._ag2_aux_full_boundaries_enabled
            and "core" in self._ag2_aux_full_stages
        ):
            self._ag2_aux_core = attn_output
        if self._ag2_full_attn_trace_enabled:
            trace = attn_output[:3]
            self._ag2_trace_core[: trace.shape[0]].copy_(trace)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        if getattr(self, "_ag2_projection_capture_enabled", False):
            selected = _ag2_compact_select(
                attn_output,
                self._ag2_projection_capture_row_indices,
            )
            self._ag2_projection_capture_gated.copy_(selected)
        if return_ag2_mtp_trace:
            mtp_trace["gated"] = attn_output
        if (
            self._ag2_aux_full_boundaries_enabled
            and "gated" in self._ag2_aux_full_stages
        ):
            self._ag2_aux_gated = attn_output
        if self._ag2_full_attn_trace_enabled:
            trace = attn_output[:3]
            self._ag2_trace_gated_output[: trace.shape[0]].copy_(trace)
        output, _ = self.o_proj(attn_output)
        if self._ag2_tp3_unified_exact_reduce:
            output = tensor_model_parallel_unified_exact_all_reduce(output)
        if return_ag2_mtp_trace:
            mtp_trace["output_parallel"] = self.o_proj._ag2_aux_output_parallel
            mtp_trace["output"] = output
        if self._ag2_aux_full_boundaries_enabled:
            if "output_parallel" in self._ag2_aux_full_stages:
                self._ag2_aux_output_parallel = self.o_proj._ag2_aux_output_parallel
            if "output" in self._ag2_aux_full_stages:
                self._ag2_aux_output = output
        if self._ag2_full_attn_trace_enabled:
            trace = output[:3]
            self._ag2_trace_attention_output[: trace.shape[0]].copy_(trace)
        if return_ag2_mtp_trace:
            return output, mtp_trace
        return output


def _ag2_compact_row_indices(
    positions: torch.Tensor,
    target_position: int | tuple[int, ...],
    capacity: int,
) -> torch.Tensor:
    flat_positions = positions[0] if positions.ndim == 2 else positions
    target_positions = (
        target_position if isinstance(target_position, tuple) else (target_position,)
    )
    matches = flat_positions.eq(target_positions[0])
    for value in target_positions[1:]:
        matches = torch.logical_or(matches, flat_positions.eq(value))
    match_scores = torch.nn.functional.pad(
        matches.to(torch.int32),
        (0, capacity),
        value=-1,
    )
    raw_indices = torch.topk(
        match_scores,
        k=capacity,
        largest=True,
        sorted=False,
    ).indices
    return torch.where(
        raw_indices < flat_positions.numel(),
        raw_indices,
        torch.full_like(raw_indices, -1),
    )


def _ag2_compact_select(
    value: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    valid_rows = row_indices >= 0
    safe_row_indices = torch.where(
        valid_rows,
        row_indices,
        torch.zeros_like(row_indices),
    )
    selected = torch.index_select(value, 0, safe_row_indices)
    return torch.where(
        valid_rows.unsqueeze(-1),
        selected,
        torch.zeros_like(selected),
    )


@triton.jit
def _ag2_trace_bit_hash_kernel(
    value_ptr,
    packet_ptr,
    width: tl.constexpr,
    element_bits: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < width
    raw = tl.load(value_ptr + row * width + offsets, mask=mask, other=0.0)
    if element_bits == 16:
        bits = raw.to(tl.int16, bitcast=True).to(tl.int32)
    else:
        bits = raw.to(tl.int32, bitcast=True)
    indices = offsets.to(tl.int32)
    weight_b = (indices * 1103515245 + 12345) & 0x7FFFFFFF
    hash_sum = tl.sum(bits.to(tl.int64), axis=0)
    hash_a = tl.sum((bits * (indices + 1)).to(tl.int64), axis=0)
    hash_b = tl.sum((bits * weight_b).to(tl.int64), axis=0)
    packet_row = packet_ptr + row * 9
    tl.store(packet_row, width * (element_bits // 8))
    tl.store(packet_row + 1, hash_sum)
    tl.store(packet_row + 2, hash_a)
    tl.store(packet_row + 3, hash_b)


@triton.jit
def _ag2_trace_numeric_kernel(
    value_ptr,
    packet_ptr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < width
    numeric = tl.load(value_ptr + row * width + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    positive_inf = float("inf")
    negative_inf = -float("inf")
    minimum_values = tl.where(mask, numeric, positive_inf)
    maximum_values = tl.where(mask, numeric, negative_inf)
    summary_0 = tl.sum(numeric, axis=0)
    summary_1 = tl.sum(tl.abs(numeric), axis=0)
    summary_2 = tl.sum(numeric * numeric, axis=0)
    summary_3 = tl.min(minimum_values, axis=0)
    summary_4 = tl.max(maximum_values, axis=0)
    packet_row = packet_ptr + row * 9
    tl.store(packet_row + 4, summary_0.to(tl.int32, bitcast=True).to(tl.int64))
    tl.store(packet_row + 5, summary_1.to(tl.int32, bitcast=True).to(tl.int64))
    tl.store(packet_row + 6, summary_2.to(tl.int32, bitcast=True).to(tl.int64))
    tl.store(packet_row + 7, summary_3.to(tl.int32, bitcast=True).to(tl.int64))
    tl.store(packet_row + 8, summary_4.to(tl.int32, bitcast=True).to(tl.int64))


def _ag2_trace_packet_cuda(value: torch.Tensor) -> torch.Tensor:
    rows = value.shape[0]
    contiguous = value.contiguous()
    flat_width = contiguous[0].numel()
    element_bits = contiguous.element_size() * 8
    if element_bits not in (16, 32) or not contiguous.dtype.is_floating_point:
        return _ag2_trace_packet_torch(contiguous)
    packet = torch.empty((rows, 9), dtype=torch.int64, device=value.device)
    block = triton.next_power_of_2(flat_width)
    grid = (rows,)
    _ag2_trace_bit_hash_kernel[grid](
        contiguous,
        packet,
        width=flat_width,
        element_bits=element_bits,
        block=block,
        num_warps=8,
    )
    _ag2_trace_numeric_kernel[grid](
        contiguous,
        packet,
        width=flat_width,
        block=block,
        num_warps=8,
    )
    return packet


def _ag2_trace_packet_torch(value: torch.Tensor) -> torch.Tensor:
    """Portable control implementation; production CUDA uses no temporaries."""
    rows = value.shape[0]
    contiguous = value.contiguous()
    bit_rows = contiguous.view(torch.int16).reshape(rows, -1).to(torch.int32)
    words = bit_rows.shape[1]
    width = words * 2
    index = torch.arange(words, dtype=torch.int32, device=value.device)
    weight_a = index + 1
    weight_b = torch.bitwise_and(index * 1103515245 + 12345, 0x7FFFFFFF)
    hash_sum = bit_rows.sum(dim=1, dtype=torch.int64)
    hash_a = (bit_rows * weight_a).sum(dim=1, dtype=torch.int64)
    hash_b = (bit_rows * weight_b).sum(dim=1, dtype=torch.int64)

    numeric = contiguous.reshape(rows, -1).to(torch.float32)
    summaries = torch.stack(
        (
            numeric.sum(dim=1),
            numeric.abs().sum(dim=1),
            numeric.square().sum(dim=1),
            numeric.amin(dim=1),
            numeric.amax(dim=1),
        ),
        dim=1,
    )
    summary_bits = summaries.contiguous().view(torch.int32).to(torch.int64)
    widths = torch.full_like(hash_sum, width)
    return torch.cat(
        (
            widths.unsqueeze(1),
            hash_sum.unsqueeze(1),
            hash_a.unsqueeze(1),
            hash_b.unsqueeze(1),
            summary_bits,
        ),
        dim=1,
    )


def _ag2_trace_packet_impl(value: torch.Tensor) -> torch.Tensor:
    """Compute the exact-bit/numeric packet used by the broad trace."""
    if value.is_cuda:
        return _ag2_trace_packet_cuda(value)
    return _ag2_trace_packet_torch(value)


# Keep the 1,300 broad-trace reductions opaque to Inductor.  If they are
# inlined into the target model graph, every checkpoint becomes an independent
# graph output dependency and its aggregate compile-time workspace starves the
# accepted batch-invariant FC kernel of the temporary BF16 lm_head allocation
# it needs.  The custom op remains CUDA-graph capturable: its body is composed
# only of ordinary CUDA tensor operations and has a fixed [rows, 9] result.
@torch.library.custom_op("vllm::ag2_trace_packet", mutates_args=())
def _ag2_trace_packet_op(value: torch.Tensor) -> torch.Tensor:
    return _ag2_trace_packet_impl(value)


@_ag2_trace_packet_op.register_fake
def _ag2_trace_packet_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty((value.shape[0], 9), dtype=torch.int64, device=value.device)


def _ag2_trace_packet(value: torch.Tensor) -> torch.Tensor:
    """Return a bounded exact-bit/numeric packet for broad trace outputs.

    Keeping hundreds of selected hidden vectors as graph outputs prevents
    Inductor from releasing their storage and makes the observer itself
    inadmissible.  The packet keeps two independent exact-bit word hashes,
    exact float32 summary bit patterns and the row width.  Targeted raw replay
    remains available after this packet identifies the first divergent stage.
    """
    if (
        os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "0") != "1"
        or not value.dtype.is_floating_point
        or value.ndim < 2
    ):
        return value
    return _ag2_trace_packet_op(value)


_AG2_TRANSIENT_TRACE_ATTRS = (
    "_ag2_aux_qkv",
    "_ag2_aux_q",
    "_ag2_aux_k",
    "_ag2_aux_v",
    "_ag2_aux_qkvz",
    "_ag2_aux_ba",
    "_ag2_aux_core",
    "_ag2_aux_gated",
    "_ag2_aux_gated_norm",
    "_ag2_aux_output_parallel",
    "_ag2_aux_output",
    "_ag2_aux_dcp_output_pack",
    "_ag2_aux_dcp_lse_pack",
    "_ag2_aux_dcp_kv_history_pack",
    "_ag2_aux_dcp_kv_history_meta",
    "_ag2_aux_dcp_kv_page_indices_pack",
    "_ag2_aux_dcp_observer_meta",
    "_ag2_aux_dcp_current_kv_pack",
    "_ag2_aux_dcp_scale_pack",
    "_ag2_aux_replay_float",
    "_ag2_aux_replay_state",
    "_ag2_aux_replay_meta",
    "_ag2_aux_gdn_input_norm",
    "_ag2_aux_gdn_boundaries",
    "_ag2_aux_full_boundaries",
    "_ag2_aux_post_attention_norm",
    "_ag2_aux_post_attention_residual",
    "_ag2_aux_compact_row_indices",
    "_ag2_aux_compact_input_norm",
    "_ag2_aux_compact_attention_output",
    "_ag2_aux_compact_post_attention_norm",
    "_ag2_aux_compact_post_attention_residual",
    "_ag2_aux_compact_full_boundaries",
    "_ag2_aux_compact_gdn_boundaries",
    "_ag2_aux_compact_mlp_boundaries",
    "_ag2_aux_compact_qkvz",
    "_ag2_aux_compact_ba",
    "_ag2_aux_compact_core",
    "_ag2_aux_compact_gated_norm",
    "_ag2_aux_compact_output_parallel",
    "_ag2_aux_compact_output",
    "_ag2_aux_compact_boundaries",
    "_ag2_aux_compact_gate_up",
    "_ag2_aux_compact_activation",
    "_ag2_aux_compact_down_parallel",
    "_ag2_aux_full_gate_up",
    "_ag2_aux_full_activation",
    "_ag2_aux_full_down_parallel",
    "_ag2_aux_full_output",
    "_ag2_aux_sequence_operator_boundaries",
    "_ag2_aux_sequence_mlp_boundaries",
)


def _ag2_release_layer_trace_refs(layer: nn.Module) -> None:
    """End raw trace ownership inside the same compiled forward.

    Dynamo represents observable module-attribute mutations as hidden graph
    outputs.  Clearing them only in the runner is therefore too late: every
    compiled replay would still materialize and return all raw intermediates.
    Make ``None`` the final attribute state before model return so the graph
    exposes only the independent fingerprint packets.
    """
    if os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "0") != "1":
        return
    attention = getattr(layer, "self_attn", None)
    linear_attention = getattr(layer, "linear_attn", None)
    mlp = getattr(layer, "mlp", None)
    modules = (
        layer,
        attention,
        getattr(attention, "attn", None),
        getattr(attention, "o_proj", None),
        linear_attention,
        getattr(linear_attention, "out_proj", None),
        mlp,
        getattr(mlp, "down_proj", None),
    )
    for module in modules:
        if module is None:
            continue
        for name in _AG2_TRANSIENT_TRACE_ATTRS:
            if hasattr(module, name):
                setattr(module, name, None)


def _ag2_packetize_and_clear_trace_attrs(
    module: nn.Module,
    names: tuple[str, ...],
) -> tuple[torch.Tensor, ...]:
    """Packetize semantic boundaries now, then end their raw ownership."""
    packets = tuple(_ag2_trace_packet(getattr(module, name)) for name in names)
    for name in names:
        setattr(module, name, None)
    return packets


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        sequence_boundary_layers = {
            int(value)
            for value in os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_SEQUENCE_BOUNDARY_LAYERS",
                "",
            ).split(",")
            if value
        }
        self._ag2_aux_sequence_boundary_enabled = (
            self.layer_idx in sequence_boundary_layers
        )
        self._ag2_aux_all_internal_boundaries = _ag2_internal_trace_layer_enabled(
            self.layer_idx
        )
        self._ag2_aux_compact_gdn_stages = _ag2_selected_stages(
            "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_GDN_STAGES",
            AG2_COMPACT_GDN_STAGE_ORDER,
        )
        self._ag2_aux_compact_mlp_stages = _ag2_selected_stages(
            "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_MLP_STAGES",
            AG2_COMPACT_MLP_STAGE_ORDER,
        )
        self._ag2_aux_attention_boundary_enabled = (
            layer_type == "full_attention"
            and os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_ATTENTION_BOUNDARY", "0"
            )
            == "1"
            and self.layer_idx
            == int(
                os.environ.get(
                    "AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "0"
                )
            )
        )
        self._ag2_aux_gdn_boundaries_enabled = layer_type == "linear_attention" and (
            self._ag2_aux_sequence_boundary_enabled
            or (
                os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_GDN_BOUNDARIES", "0")
                == "1"
                and self.layer_idx
                == int(
                    os.environ.get(
                        "AG2_VLLM_AUX_HIDDEN_TRACE_GDN_BOUNDARY_LAYER",
                        "0",
                    )
                )
            )
        )
        self._ag2_aux_compact_boundary_enabled = (
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ALL_BOUNDARIES",
                "0",
            )
            == "1"
            and self.layer_idx
            <= int(
                os.environ.get(
                    "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_MAX_LAYER",
                    "2147483647",
                )
            )
            or self.layer_idx
            == int(
                os.environ.get(
                    "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_LAYER",
                    "-1",
                )
            )
            or self.layer_idx
            == int(
                os.environ.get(
                    "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_GDN_BOUNDARY_LAYER",
                    "-1",
                )
            )
        )
        self._ag2_aux_compact_gdn_boundaries_enabled = (
            layer_type == "linear_attention"
            and (
                self._ag2_aux_all_internal_boundaries
                or self.layer_idx
                == int(
                    os.environ.get(
                        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_GDN_BOUNDARY_LAYER",
                        "-1",
                    )
                )
            )
        )
        self._ag2_aux_gdn_replay_enabled = (
            layer_type == "linear_attention"
            and os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_GDN_REPLAY_ONLY", "0") == "1"
            and self.layer_idx
            == int(
                os.environ.get(
                    "AG2_VLLM_AUX_HIDDEN_TRACE_GDN_BOUNDARY_LAYER",
                    "0",
                )
            )
        )

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        is_moe_layer = (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        )
        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )
        self._ag2_tp3_owner_prequant = _ag2_tp3_owner_prequant_enabled(
            vllm_config,
            prefix,
            is_moe_layer=is_moe_layer,
        )
        self._ag2_tp3_mtp_block5 = _ag2_tp3_mtp_block5_enabled(vllm_config, prefix)
        reduce_results = not (
            self.use_attn_reduce_scatter_for_moe or self._ag2_tp3_owner_prequant
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
                reduce_results=reduce_results,
            )
            if self._ag2_aux_all_internal_boundaries:
                self.linear_attn.ag2_enable_compact_trace()
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if is_moe_layer:
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=not self._ag2_tp3_owner_prequant,
                prefix=f"{prefix}.mlp",
            )
        if self._ag2_aux_all_internal_boundaries:
            if not isinstance(self.mlp, Qwen3NextMLP):
                raise NotImplementedError(
                    "Full internal trace currently requires the dense Qwen MLP"
                )
            self.mlp._ag2_aux_compact_trace_enabled = True
            self.mlp.down_proj._ag2_aux_output_parallel_enabled = True
            if self.layer_type == "full_attention":
                self.self_attn.ag2_enable_compact_full_trace()
        if self._ag2_aux_sequence_boundary_enabled:
            if not isinstance(self.mlp, Qwen3NextMLP):
                raise NotImplementedError(
                    "Sequence boundary trace currently requires the dense Qwen MLP"
                )
            self.mlp._ag2_aux_full_trace_enabled = True
            self.mlp.down_proj._ag2_aux_output_parallel_enabled = True
            if self.layer_type == "full_attention":
                self.self_attn.ag2_enable_sequence_full_trace()

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self._ag2_enable_projection_calibration_capture(vllm_config.model_config.dtype)

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    def _ag2_enable_projection_calibration_capture(self, dtype: torch.dtype) -> None:
        output = os.environ.get("AG2_VLLM_PROJECTION_CALIBRATION_OUTPUT", "")
        self._ag2_projection_capture_enabled = bool(output)
        if not self._ag2_projection_capture_enabled:
            return
        positions_raw = os.environ.get("AG2_VLLM_PROJECTION_CALIBRATION_POSITIONS", "")
        positions = tuple(
            dict.fromkeys(int(value) for value in positions_raw.split(",") if value)
        )
        capacity = int(os.environ.get("AG2_VLLM_PROJECTION_CALIBRATION_CAPACITY", "4"))
        if not positions or any(position < 0 for position in positions):
            raise ValueError("projection calibration requires nonnegative positions")
        if capacity < len(positions):
            raise ValueError(
                "projection calibration capacity is smaller than positions"
            )
        self._ag2_projection_capture_positions = positions
        self._ag2_projection_capture_capacity = capacity
        self.register_buffer(
            "_ag2_projection_capture_rows",
            torch.full((capacity,), -1, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_ag2_projection_capture_attention_input",
            torch.full(
                (capacity, self.input_layernorm.weight.shape[0]),
                torch.nan,
                dtype=dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_ag2_projection_capture_mlp_input",
            torch.full(
                (capacity, self.post_attention_layernorm.weight.shape[0]),
                torch.nan,
                dtype=dtype,
            ),
            persistent=False,
        )
        if self.layer_type == "full_attention":
            self.self_attn.ag2_enable_projection_calibration_capture(capacity, dtype)
        else:
            self.linear_attn.ag2_enable_projection_calibration_capture(capacity, dtype)
        enable_mlp_capture = getattr(
            self.mlp, "ag2_enable_projection_calibration_capture", None
        )
        if enable_mlp_capture is None:
            raise NotImplementedError(
                "projection calibration currently requires the dense Qwen MLP"
            )
        enable_mlp_capture(capacity, dtype)

    def _ag2_owner_prequantize(
        self,
        contribution: torch.Tensor,
        residual: torch.Tensor,
        norm: Qwen3NextRMSNorm,
        consumer: nn.Module,
        *,
        gdn_pair: bool = False,
    ) -> tuple[
        QuantizedActivation | GDNQuantizedActivations,
        torch.Tensor,
    ]:
        if contribution.ndim != 2 or contribution.shape[1] != 5120:
            raise RuntimeError(
                "TP3 owner prequant requires a two-dimensional [M,5120] input"
            )
        if residual.shape != contribution.shape:
            raise RuntimeError(
                "TP3 owner prequant requires a full [M,5120] residual carrier"
            )
        required = (
            "input_global_scale_inv",
            "_ag2_nvfp4_arc_selected_all",
            "_ag2_nvfp4_arc_route_to_0",
            "_ag2_nvfp4_arc_route_to_1",
            "_ag2_nvfp4_arc_route_to_2",
            "_ag2_nvfp4_arc_inverse_order",
            "_ag2_nvfp4_arc_route_counts",
        )
        missing = [name for name in required if not hasattr(consumer, name)]
        if missing:
            raise RuntimeError(
                "TP3 owner prequant consumer is missing ARC ABI: " + ", ".join(missing)
            )
        arc_q, arc_sf, base_q, base_sf, residual = (
            tensor_model_parallel_owner_residual_arc_prequant(
                contribution,
                residual,
                norm.weight.float() + 1.0,
                consumer.input_global_scale_inv,
                consumer._ag2_nvfp4_arc_selected_all,
                consumer._ag2_nvfp4_arc_route_to_0,
                consumer._ag2_nvfp4_arc_route_to_1,
                consumer._ag2_nvfp4_arc_route_to_2,
                consumer._ag2_nvfp4_arc_inverse_order,
                list(consumer._ag2_nvfp4_arc_route_counts),
                norm.variance_epsilon,
            )
        )
        arc = QuantizedActivation(
            data=arc_q,
            scale=arc_sf,
            orig_dtype=contribution.dtype,
            orig_shape=contribution.shape,
            quant_key=kNvfp4Dynamic,
        )
        if not gdn_pair:
            return arc, residual
        canonical = QuantizedActivation(
            data=base_q,
            scale=base_sf,
            orig_dtype=contribution.dtype,
            orig_shape=contribution.shape,
            quant_key=kNvfp4Dynamic,
        )
        return GDNQuantizedActivations(qkvz=arc, ba=canonical), residual

    def _ag2_owner_attention_prequantize(
        self,
        contribution: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[
        QuantizedActivation | GDNQuantizedActivations,
        torch.Tensor,
    ]:
        if self.layer_type == "linear_attention":
            return self._ag2_owner_prequantize(
                contribution,
                residual,
                self.input_layernorm,
                self.linear_attn.in_proj_qkvz,
                gdn_pair=True,
            )
        return self._ag2_owner_prequantize(
            contribution,
            residual,
            self.input_layernorm,
            self.self_attn.qkv_proj,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        return_ag2_mtp_trace: bool = False,
        **kwargs: object,
    ):
        mtp_trace: dict[str, torch.Tensor] = {}
        if self._ag2_tp3_owner_prequant and return_ag2_mtp_trace:
            raise RuntimeError(
                "TP3 owner prequant does not expose full-BF16 diagnostic traces"
            )
        full_num_tokens = positions.shape[-1]
        input_is_sequence_parallel = (
            self.use_attn_reduce_scatter_for_moe
            and residual is not None
            and hidden_states.shape[0] != full_num_tokens
        )

        if getattr(self, "_ag2_layer0_trace_enabled", False):
            trace = hidden_states[:3]
            self._ag2_trace_input[: trace.shape[0]].copy_(trace)
        if return_ag2_mtp_trace:
            mtp_trace["layer_input"] = hidden_states
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self._ag2_tp3_owner_prequant:
            hidden_states, residual = self._ag2_owner_attention_prequantize(
                hidden_states,
                residual,
            )
        elif self._ag2_tp3_mtp_block5:
            hidden_states, residual = (
                tensor_model_parallel_v1_block5_fused_add_rms_norm(
                    hidden_states,
                    residual,
                    self.input_layernorm.weight.float() + 1.0,
                    self.input_layernorm.variance_epsilon,
                )
            )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if return_ag2_mtp_trace:
            mtp_trace["input_norm"] = hidden_states
        if self._ag2_projection_capture_enabled:
            capture_rows = _ag2_compact_row_indices(
                positions,
                self._ag2_projection_capture_positions,
                self._ag2_projection_capture_capacity,
            )
            self._ag2_projection_capture_rows.copy_(capture_rows)
            self._ag2_projection_capture_attention_input.copy_(
                _ag2_compact_select(hidden_states, capture_rows)
            )
            self.mlp._ag2_projection_capture_row_indices = capture_rows
            if self.layer_type == "full_attention":
                self.self_attn._ag2_projection_capture_row_indices = capture_rows
            else:
                self.linear_attn._ag2_projection_capture_row_indices = capture_rows
        if self._ag2_aux_compact_boundary_enabled:
            configured_positions = tuple(
                int(value)
                for value in os.environ.get(
                    "AG2_VLLM_AUX_HIDDEN_TRACE_POSITIONS",
                    os.environ["AG2_VLLM_AUX_HIDDEN_TRACE_POSITION"],
                ).split(",")
                if value
            )
            row_indices = _ag2_compact_row_indices(
                positions,
                configured_positions,
                int(
                    os.environ.get(
                        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_CAPACITY",
                        "64",
                    )
                ),
            )
            self._ag2_aux_compact_row_indices = row_indices
            self._ag2_aux_compact_input_norm = _ag2_compact_select(
                hidden_states,
                row_indices,
            )
        if (
            getattr(self, "_ag2_aux_gdn_boundaries_enabled", False)
            or self._ag2_aux_sequence_boundary_enabled
        ):
            self._ag2_aux_gdn_input_norm = (
                _ag2_trace_packet(hidden_states)
                if self._ag2_aux_sequence_boundary_enabled
                else hidden_states
            )
        if getattr(self, "_ag2_layer0_trace_enabled", False):
            position_trace = positions.reshape(-1)[:3]
            self._ag2_trace_positions[: position_trace.shape[0]].copy_(position_trace)
            trace = hidden_states[:3]
            self._ag2_trace_input_norm[: trace.shape[0]].copy_(trace)

        if input_is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.layer_type == "linear_attention":
            if self._ag2_aux_compact_gdn_boundaries_enabled:
                self.linear_attn._ag2_aux_compact_row_indices = (
                    self._ag2_aux_compact_row_indices
                )
            hidden_states = self.linear_attn(hidden_states=hidden_states)
            if getattr(self, "_ag2_aux_gdn_boundaries_enabled", False):
                operator_boundaries = (
                    self._ag2_aux_gdn_input_norm,
                    self.linear_attn._ag2_aux_qkvz,
                    self.linear_attn._ag2_aux_ba,
                    self.linear_attn._ag2_aux_core,
                    self.linear_attn._ag2_aux_gated_norm,
                    self.linear_attn._ag2_aux_output_parallel,
                    hidden_states,
                )
                if self._ag2_aux_sequence_boundary_enabled:
                    internal_names = (
                        "_ag2_aux_qkvz",
                        "_ag2_aux_ba",
                        "_ag2_aux_core",
                        "_ag2_aux_gated_norm",
                        "_ag2_aux_output_parallel",
                    )
                    internal_packets = _ag2_packetize_and_clear_trace_attrs(
                        self.linear_attn,
                        internal_names,
                    )
                    self._ag2_aux_sequence_operator_boundaries = (
                        _ag2_trace_packet(operator_boundaries[0]),
                        *internal_packets,
                        _ag2_trace_packet(operator_boundaries[-1]),
                    )
                else:
                    self._ag2_aux_gdn_boundaries = (
                        operator_boundaries[:3]
                        + (
                            self.linear_attn._ag2_aux_replay_float,
                            self.linear_attn._ag2_aux_replay_state,
                            self.linear_attn._ag2_aux_replay_meta,
                        )
                        + operator_boundaries[3:]
                    )
        elif self.layer_type == "full_attention":
            capture_internal = (
                self._ag2_aux_all_internal_boundaries
                or self._ag2_aux_sequence_boundary_enabled
            )
            attn_result = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                return_ag2_mtp_trace=return_ag2_mtp_trace or capture_internal,
            )
            if return_ag2_mtp_trace or capture_internal:
                hidden_states, attention_trace = attn_result
                if return_ag2_mtp_trace:
                    mtp_trace.update(
                        {
                            f"attention_{name}": value
                            for name, value in attention_trace.items()
                        }
                    )
                if capture_internal:
                    if self._ag2_aux_sequence_boundary_enabled:
                        self._ag2_aux_sequence_operator_boundaries = (
                            self._ag2_aux_gdn_input_norm,
                        ) + tuple(
                            _ag2_trace_packet(attention_trace[name])
                            for name in self.self_attn._ag2_aux_compact_full_stages
                            if not name.startswith("dcp_")
                        )
                    else:
                        self._ag2_aux_compact_full_boundaries = tuple(
                            (
                                attention_trace[name]
                                if name.startswith("dcp_")
                                else _ag2_compact_select(
                                    attention_trace[name],
                                    self._ag2_aux_compact_row_indices,
                                )
                            )
                            for name in self.self_attn._ag2_aux_compact_full_stages
                        )
            else:
                hidden_states = attn_result
            if getattr(self.self_attn, "_ag2_aux_full_boundaries_enabled", False):
                self._ag2_aux_full_boundaries = tuple(
                    getattr(self.self_attn, f"_ag2_aux_{stage}")
                    for stage in self.self_attn._ag2_aux_full_stages
                )
        else:
            raise ValueError("Invalid layer_type")
        if getattr(self, "_ag2_layer0_trace_enabled", False):
            trace = hidden_states[:3]
            self._ag2_trace_attention_output[: trace.shape[0]].copy_(trace)
        if return_ag2_mtp_trace:
            mtp_trace["attention_output"] = hidden_states
        if self._ag2_aux_compact_boundary_enabled:
            self._ag2_aux_compact_attention_output = _ag2_compact_select(
                hidden_states,
                self._ag2_aux_compact_row_indices,
            )

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            # small trick using minus, eg. -17 % 8 = 7
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            # pad if not divisible by world size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if not input_is_sequence_parallel:
                residual = sequence_parallel_chunk(residual)

        # Fully Connected
        if self._ag2_tp3_owner_prequant:
            hidden_states, residual = self._ag2_owner_prequantize(
                hidden_states,
                residual,
                self.post_attention_layernorm,
                self.mlp.gate_up_proj,
            )
        elif self._ag2_tp3_mtp_block5:
            hidden_states, residual = (
                tensor_model_parallel_v1_block5_fused_add_rms_norm(
                    hidden_states,
                    residual,
                    self.post_attention_layernorm.weight.float() + 1.0,
                    self.post_attention_layernorm.variance_epsilon,
                )
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        if self._ag2_projection_capture_enabled:
            self._ag2_projection_capture_mlp_input.copy_(
                _ag2_compact_select(
                    hidden_states,
                    self._ag2_projection_capture_rows,
                )
            )
        if return_ag2_mtp_trace:
            mtp_trace["post_attention_norm"] = hidden_states
            mtp_trace["post_attention_residual"] = residual
        if (
            getattr(self, "_ag2_aux_attention_boundary_enabled", False)
            or self._ag2_aux_sequence_boundary_enabled
        ):
            if self._ag2_aux_sequence_boundary_enabled:
                self._ag2_aux_post_attention_norm = _ag2_trace_packet(hidden_states)
                self._ag2_aux_post_attention_residual = _ag2_trace_packet(residual)
            else:
                self._ag2_aux_post_attention_norm = hidden_states
                self._ag2_aux_post_attention_residual = residual
        if self._ag2_aux_compact_boundary_enabled:
            self._ag2_aux_compact_post_attention_norm = _ag2_compact_select(
                hidden_states,
                self._ag2_aux_compact_row_indices,
            )
            self._ag2_aux_compact_post_attention_residual = _ag2_compact_select(
                residual,
                self._ag2_aux_compact_row_indices,
            )
        if getattr(self, "_ag2_layer0_trace_enabled", False):
            hidden_trace = hidden_states[:3]
            residual_trace = residual[:3]
            self._ag2_trace_post_attention_norm[: hidden_trace.shape[0]].copy_(
                hidden_trace
            )
            self._ag2_trace_post_attention_residual[: residual_trace.shape[0]].copy_(
                residual_trace
            )
        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = self.mlp(
                hidden_states,
                already_sequence_parallel=True,
            )
        else:
            if self._ag2_aux_all_internal_boundaries:
                self.mlp._ag2_aux_compact_row_indices = (
                    self._ag2_aux_compact_row_indices
                )
            hidden_states = self.mlp(hidden_states)
        if self._ag2_aux_all_internal_boundaries:
            mlp_boundaries = (
                ("gate_up", self.mlp._ag2_aux_compact_gate_up),
                ("activation", self.mlp._ag2_aux_compact_activation),
                ("down_parallel", self.mlp._ag2_aux_compact_down_parallel),
                ("output", self.mlp._ag2_aux_compact_output),
            )
            self._ag2_aux_compact_mlp_boundaries = tuple(
                value for _stage, value in mlp_boundaries
            )
        if self._ag2_aux_sequence_boundary_enabled:
            self._ag2_aux_sequence_mlp_boundaries = (
                _ag2_packetize_and_clear_trace_attrs(
                    self.mlp,
                    (
                        "_ag2_aux_full_gate_up",
                        "_ag2_aux_full_activation",
                        "_ag2_aux_full_down_parallel",
                        "_ag2_aux_full_output",
                    ),
                )
            )
        if return_ag2_mtp_trace:
            mtp_trace["mlp_output"] = hidden_states
        if getattr(self, "_ag2_layer0_trace_enabled", False):
            trace = hidden_states[:3]
            self._ag2_trace_mlp_output[: trace.shape[0]].copy_(trace)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        if return_ag2_mtp_trace:
            return hidden_states, residual, mtp_trace
        return hidden_states, residual


def _all_gather_hidden_and_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    full_num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if residual is None:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]
        return hidden_states, None

    combined_states = torch.cat([hidden_states, residual], dim=-1)
    combined_states = tensor_model_parallel_all_gather(combined_states, 0)
    combined_states = combined_states[:full_num_tokens]
    hidden_states, residual = combined_states.split([hidden_size, hidden_size], dim=-1)
    return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self._ag2_tp3_owner_prequant = any(
            getattr(layer, "_ag2_tp3_owner_prequant", False)
            for layer in self.layers
            if not isinstance(layer, PPMissingLayer)
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def _maybe_add_ag2_aux_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> list[torch.Tensor]:
        if layer_idx not in self.aux_hidden_state_layers:
            return aux_hidden_states
        compact_position = getattr(
            self,
            "_ag2_aux_trace_compact_position",
            None,
        )
        if compact_position is None:
            start = len(aux_hidden_states)
            aux_hidden_states = self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_idx,
                hidden_states,
                residual,
            )
            if getattr(self, "_ag2_aux_trace_request_chunks", False):
                aux_hidden_states[start:] = [
                    _ag2_trace_packet(value) for value in aux_hidden_states[start:]
                ]
            return aux_hidden_states

        # Text mRoPE supplies identical T/H/W rows with shape (3, num_tokens),
        # while hidden_states remains row-major over num_tokens.  Use the same
        # authoritative T row as the fused RoPE path; flattening would create
        # indices up to 3 * num_tokens and corrupt the hidden-state selection.
        flat_positions = positions[0] if positions.ndim == 2 else positions
        capacity = int(self._ag2_aux_trace_compact_capacity)
        num_rows = flat_positions.numel()
        compact_positions = getattr(
            self,
            "_ag2_aux_trace_compact_positions",
            (compact_position,),
        )
        matches = flat_positions.eq(compact_positions[0])
        for value in compact_positions[1:]:
            matches = torch.logical_or(matches, flat_positions.eq(value))
        match_scores = matches.to(torch.int32)
        # TorchCompileWithNoGuards reuses one graph over every shape in the
        # compile range.  Always append a fixed dummy tail: a Python branch on
        # num_rows would be specialized at the initial large shape and then be
        # invalid for a smaller CUDA Graph descriptor.
        match_scores = torch.nn.functional.pad(
            match_scores,
            (0, capacity),
            value=-1,
        )
        row_indices = torch.topk(
            match_scores,
            k=capacity,
            largest=True,
            sorted=False,
        ).indices
        valid_rows = row_indices < num_rows
        safe_row_indices = torch.where(
            valid_rows,
            row_indices,
            torch.zeros_like(row_indices),
        )
        selected_hidden = torch.index_select(hidden_states, 0, safe_row_indices)
        selected_residual = (
            torch.zeros_like(selected_hidden)
            if residual is None
            else torch.index_select(residual, 0, safe_row_indices)
        )
        selected_combined = selected_hidden + selected_residual
        row_indices = torch.where(
            valid_rows,
            row_indices,
            torch.full_like(row_indices, -1),
        )
        selected_hidden = torch.where(
            valid_rows.unsqueeze(-1),
            selected_hidden,
            torch.zeros_like(selected_hidden),
        )
        selected_residual = torch.where(
            valid_rows.unsqueeze(-1),
            selected_residual,
            torch.zeros_like(selected_residual),
        )
        selected_combined = torch.where(
            valid_rows.unsqueeze(-1),
            selected_combined,
            torch.zeros_like(selected_combined),
        )
        aux_hidden_states.extend(
            (
                row_indices,
                _ag2_trace_packet(selected_hidden),
                _ag2_trace_packet(selected_residual),
                _ag2_trace_packet(selected_combined),
            )
        )
        return aux_hidden_states

    def _maybe_add_ag2_compact_layer_boundary(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        layer: nn.Module,
    ) -> list[torch.Tensor]:
        all_boundaries_enabled = getattr(
            self,
            "_ag2_aux_trace_compact_all_boundaries",
            False,
        ) and layer_idx <= getattr(
            self,
            "_ag2_aux_trace_compact_boundary_max_layer",
            -1,
        )
        if not all_boundaries_enabled and (
            layer_idx
            != getattr(
                self,
                "_ag2_aux_trace_compact_boundary_layer",
                -1,
            )
        ):
            return aux_hidden_states

        aux_hidden_states.extend(
            (
                layer._ag2_aux_compact_row_indices,
                _ag2_trace_packet(layer._ag2_aux_compact_input_norm),
                _ag2_trace_packet(layer._ag2_aux_compact_attention_output),
                _ag2_trace_packet(layer._ag2_aux_compact_post_attention_norm),
                _ag2_trace_packet(layer._ag2_aux_compact_post_attention_residual),
            )
        )
        return aux_hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        full_num_tokens = positions.shape[-1]
        aux_hidden_states = self._maybe_add_ag2_aux_hidden_state(
            [], 0, hidden_states, residual, positions
        )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if (
                hidden_states.shape[0] != full_num_tokens
                and not layer.use_attn_reduce_scatter_for_moe
            ):
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            all_boundaries_enabled = getattr(
                self,
                "_ag2_aux_trace_compact_all_boundaries",
                False,
            ) and layer_idx <= getattr(
                self,
                "_ag2_aux_trace_compact_boundary_max_layer",
                -1,
            )
            if all_boundaries_enabled or layer_idx == getattr(
                self,
                "_ag2_aux_trace_compact_boundary_layer",
                -1,
            ):
                self._maybe_add_ag2_compact_layer_boundary(
                    aux_hidden_states,
                    layer_idx,
                    layer,
                )
            if getattr(layer, "_ag2_aux_all_internal_boundaries", False):
                if layer.layer_type == "full_attention":
                    aux_hidden_states.extend(
                        _ag2_trace_packet(value)
                        for value in layer._ag2_aux_compact_full_boundaries
                    )
                else:
                    aux_hidden_states.extend(
                        _ag2_trace_packet(value)
                        for stage, value in zip(
                            AG2_COMPACT_GDN_STAGE_ORDER,
                            layer.linear_attn._ag2_aux_compact_boundaries,
                            strict=True,
                        )
                        if stage in layer._ag2_aux_compact_gdn_stages
                    )
                aux_hidden_states.extend(
                    _ag2_trace_packet(value)
                    for stage, value in zip(
                        AG2_COMPACT_MLP_STAGE_ORDER,
                        layer._ag2_aux_compact_mlp_boundaries,
                        strict=True,
                    )
                    if stage in layer._ag2_aux_compact_mlp_stages
                )
            if getattr(layer, "_ag2_aux_sequence_boundary_enabled", False):
                aux_hidden_states.extend(
                    _ag2_trace_packet(value)
                    for value in layer._ag2_aux_sequence_operator_boundaries
                )
                aux_hidden_states.extend(
                    (
                        _ag2_trace_packet(layer._ag2_aux_post_attention_norm),
                        _ag2_trace_packet(layer._ag2_aux_post_attention_residual),
                    )
                )
                aux_hidden_states.extend(
                    _ag2_trace_packet(value)
                    for value in layer._ag2_aux_sequence_mlp_boundaries
                )
            if getattr(layer, "_ag2_aux_attention_boundary_enabled", False):
                if getattr(layer, "_ag2_aux_full_boundaries", None) is not None:
                    aux_hidden_states.extend(layer._ag2_aux_full_boundaries)
                if not getattr(self, "_ag2_aux_trace_compact_rows", False):
                    aux_hidden_states.extend(
                        (
                            layer._ag2_aux_post_attention_norm,
                            layer._ag2_aux_post_attention_residual,
                        )
                    )
            if getattr(layer, "_ag2_aux_gdn_boundaries_enabled", False) and not getattr(
                layer, "_ag2_aux_sequence_boundary_enabled", False
            ):
                aux_hidden_states.extend(
                    _ag2_trace_packet(value) for value in layer._ag2_aux_gdn_boundaries
                )
            if getattr(layer, "_ag2_aux_gdn_replay_enabled", False):
                aux_hidden_states.extend(
                    (
                        layer.linear_attn._ag2_aux_replay_float,
                        layer.linear_attn._ag2_aux_replay_state,
                        layer.linear_attn._ag2_aux_replay_meta,
                    )
                )
            if not getattr(
                layer, "_ag2_aux_all_internal_boundaries", False
            ) and getattr(layer, "_ag2_aux_compact_gdn_boundaries_enabled", False):
                aux_hidden_states.extend(layer.linear_attn._ag2_aux_compact_boundaries)
            if (layer_idx + 1) in self.aux_hidden_state_layers and hidden_states.shape[
                0
            ] != full_num_tokens:
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            if (
                self._ag2_tp3_owner_prequant
                and (layer_idx + 1) in self.aux_hidden_state_layers
            ):
                if residual is None:
                    raise RuntimeError(
                        "TP3 owner MTP boundary requires a residual carrier"
                    )
                aux_hidden_states.append(
                    tensor_model_parallel_owner_materialize_aux(
                        hidden_states,
                        residual,
                    )
                )
            else:
                self._maybe_add_ag2_aux_hidden_state(
                    aux_hidden_states,
                    layer_idx + 1,
                    hidden_states,
                    residual,
                    positions,
                )
            _ag2_release_layer_trace_refs(layer)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if hidden_states.shape[0] != full_num_tokens:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        if self._ag2_tp3_owner_prequant:
            if residual is None:
                raise RuntimeError("TP3 owner terminal norm requires a residual")
            hidden_states = tensor_model_parallel_owner_terminal_norm(
                hidden_states,
                residual,
                self.norm.weight.float() + 1.0,
                self.norm.variance_epsilon,
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def _ag2_validate_tp3_owner_prequant_weights(self) -> None:
        """Validate every consumer ABI once, after checkpoint loading."""
        for layer in self.layers:
            if not getattr(layer, "_ag2_tp3_owner_prequant", False):
                continue
            consumers = [layer.mlp.gate_up_proj]
            if layer.layer_type == "linear_attention":
                qkvz = layer.linear_attn.in_proj_qkvz
                ba = layer.linear_attn.in_proj_ba
                if not torch.equal(
                    qkvz.input_global_scale_inv,
                    ba.input_global_scale_inv,
                ):
                    raise RuntimeError(
                        f"layer {layer.layer_idx} GDN qkvz/ba input scales differ"
                    )
                consumers.append(qkvz)
            else:
                consumers.append(layer.self_attn.qkv_proj)
            for consumer in consumers:
                if getattr(consumer, "input_quant_key", None) != kNvfp4Dynamic:
                    raise RuntimeError(
                        f"layer {layer.layer_idx} consumer cannot accept NVFP4 QA"
                    )
                if not hasattr(consumer, "_ag2_nvfp4_arc_selected_all"):
                    raise RuntimeError(
                        f"layer {layer.layer_idx} consumer is missing ARC owner metadata"
                    )
        logger.warning(
            "Validated target-only TP3 owner residual + ARC prequant ABI for %d layers",
            sum(
                getattr(layer, "_ag2_tp3_owner_prequant", False)
                for layer in self.layers
            ),
        )


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        if (
            hf_config.linear_num_key_heads % tp_size != 0
            or hf_config.linear_num_value_heads % tp_size != 0
        ):
            tp_size = 1
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights)
