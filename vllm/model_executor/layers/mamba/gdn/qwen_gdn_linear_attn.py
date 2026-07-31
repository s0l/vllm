# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import os
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_gdn_all_reduce
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ExplicitPaddedMergedColumnParallelLinear,
    ExplicitPaddedRowParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.gdn.head_partition import (
    explicit_gdn_conv_weight_loader,
    explicit_vector_weight_loader,
    make_gdn_head_partition,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# Optional ROCm AITER Triton kernels for the GDN decode path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = rocm_aiter_ops.are_gdn_triton_kernels_available()

if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)

_AG2_GDN_REPLAY_MAX_TOKENS = 128
_AG2_GDN_REPLAY_MAX_SEQS = 8
_AG2_GDN_REPLAY_META_ELEMENTS = 512


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:
    """Resolve GDN prefill backend.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) — no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
    """
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    supports_flashinfer = False
    supports_cutedsl = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )
    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s).",
        chosen,
        requested_backend,
        head_k_dim,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend

        if backend in ("flashinfer", "cutedsl") and active_backend != backend:
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)

        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
            # Native FLA has finished reading the original value tensor before
            # its recurrent kernel produces v_new. Reuse that dead storage for
            # the equally shaped result. This is explicit and inference-only;
            # the FlashInfer and CuteDSL backends remain unchanged.
            value_out=v if core_attn_out is not None else None,
        )

    def forward_cutedsl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            chunk_gated_delta_rule_cutedsl,
        )

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        assert cu_seqlens is not None
        assert chunk_indices is not None
        assert chunk_offsets is not None

        o, final_state = chunk_gated_delta_rule_cutedsl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        if not output_final_state:
            final_state = None
        return o, final_state


@PluggableLayer.register("qwen_gated_delta_net_attention")
class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if getattr(self, "gdn_explicit_partition", False):
            conv_state_shape = MambaStateShapeCalculator._orient_conv_shape(
                self.padded_local_conv_dim,
                self.conv_kernel_size - 1 + self.num_spec,
            )
            temporal_state_shape = (
                self.padded_local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
            )
            return (
                conv_state_shape,
                temporal_state_shape,
                conv_state_shape,
                temporal_state_shape,
            )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        gqa_interleaved_layout=False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gqa_interleaved_layout = gqa_interleaved_layout
        incompatible_gdn_tp = (
            self.num_k_heads % self.tp_size != 0 or self.num_v_heads % self.tp_size != 0
        )
        self.gdn_explicit_partition = incompatible_gdn_tp and (
            not self.gqa_interleaved_layout
        )
        self.disable_tp_for_gdn = incompatible_gdn_tp and (
            not self.gdn_explicit_partition
        )
        if self.disable_tp_for_gdn:
            self.tp_size = 1
            self.tp_rank = 0
        self.gdn_partition = make_gdn_head_partition(
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.local_num_k_heads = self.gdn_partition.k_count
        self.local_num_v_heads = self.gdn_partition.v_count
        self.padded_local_num_k_heads = self.gdn_partition.max_k_count
        self.padded_local_num_v_heads = self.gdn_partition.max_v_count
        self.local_key_dim = self.gdn_partition.local_key_dim
        self.local_value_dim = self.gdn_partition.local_value_dim
        self.padded_local_key_dim = self.gdn_partition.padded_key_dim
        self.padded_local_value_dim = self.gdn_partition.padded_value_dim
        self.local_conv_dim = self.gdn_partition.local_conv_dim
        self.padded_local_conv_dim = self.gdn_partition.padded_conv_dim
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        else:
            self._forward_method = self.forward_cuda

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        conv_output_size = (
            self.padded_local_conv_dim * self.tp_size
            if self.gdn_explicit_partition
            else self.conv_dim
        )
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=conv_output_size,
            bias=False,
            prefix=f"{prefix}.conv1d",
            disable_tp=self.disable_tp_for_gdn,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
            disable_tp=self.disable_tp_for_gdn,
        )

        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_ba",
            disable_tp=self.disable_tp_for_gdn,
        )
        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        if self.gdn_explicit_partition:
            self.conv1d.weight.weight_loader = explicit_gdn_conv_weight_loader(
                self.gdn_partition
            )
        else:
            self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
                [
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ],
                self.tp_size,
                self.tp_rank,
            )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.padded_local_num_v_heads),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                self.padded_local_num_v_heads,
                dtype=torch.float32,
            )
        )

        if self.gdn_explicit_partition:
            set_weight_attrs(
                self.A_log,
                {
                    "weight_loader": explicit_vector_weight_loader(
                        self.gdn_partition.v_start, self.local_num_v_heads
                    )
                },
            )
            set_weight_attrs(
                self.dt_bias,
                {
                    "weight_loader": explicit_vector_weight_loader(
                        self.gdn_partition.v_start, self.local_num_v_heads
                    )
                },
            )
        elif not self.disable_tp_for_gdn:
            set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
            set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )
        self._ag2_aux_boundaries_enabled = (
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_GDN_BOUNDARIES", "0")
            == "1"
            and prefix.endswith(
                ".layers."
                + os.environ.get(
                    "AG2_VLLM_AUX_HIDDEN_TRACE_GDN_BOUNDARY_LAYER",
                    "0",
                )
                + ".linear_attn"
            )
        )
        if self._ag2_aux_boundaries_enabled:
            per_token_elements = (
                2 * self.local_key_dim
                + self.local_value_dim
                + 2 * self.local_num_v_heads
            )
            state_elements = (
                _AG2_GDN_REPLAY_MAX_SEQS
                * self.padded_local_num_v_heads
                * self.head_v_dim
                * self.head_k_dim
            )
            self.register_buffer(
                "_ag2_replay_float_buffer",
                torch.zeros(
                    _AG2_GDN_REPLAY_MAX_TOKENS * per_token_elements,
                    dtype=torch.float32,
                    device=current_platform.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                "_ag2_replay_state_buffer",
                torch.zeros(
                    state_elements,
                    dtype=torch.float32,
                    device=current_platform.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                "_ag2_replay_meta_buffer",
                torch.full(
                    (_AG2_GDN_REPLAY_META_ELEMENTS,),
                    -1,
                    dtype=torch.int64,
                    device=current_platform.current_device(),
                ),
                persistent=False,
            )
            self._ag2_aux_replay_float = self._ag2_replay_float_buffer
            self._ag2_aux_replay_state = self._ag2_replay_state_buffer
            self._ag2_aux_replay_meta = self._ag2_replay_meta_buffer
        self._ag2_gdn_prefill_batch_invariant_reduce = (
            os.environ.get(
                "AG2_VLLM_GDN_PREFILL_BATCH_INVARIANT_REDUCE",
                "0",
            )
            == "1"
            and reduce_results
            and self.tp_size == 3
        )
        if (
            self._ag2_gdn_prefill_batch_invariant_reduce
            and prefix.endswith(".layers.0.linear_attn")
        ):
            logger.warning(
                "Enabled experimental TP3 batch-invariant reduction for "
                "short GDN prefills"
            )
        out_proj_reduce_results = (
            reduce_results and not self._ag2_gdn_prefill_batch_invariant_reduce
        )

        if self.gdn_explicit_partition:
            self.out_proj = ExplicitPaddedRowParallelLinear(
                input_size=self.value_dim,
                padded_input_size=self.padded_local_value_dim * self.tp_size,
                local_start=self.gdn_partition.v_start * self.head_v_dim,
                local_size=self.local_value_dim,
                output_size=self.hidden_size,
                bias=False,
                input_is_parallel=True,
                reduce_results=out_proj_reduce_results,
                quant_config=self.quant_config,
                prefix=f"{prefix}.out_proj",
            )
        else:
            self.out_proj = RowParallelLinear(
                self.value_dim,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                reduce_results=out_proj_reduce_results,
                quant_config=self.quant_config,
                prefix=f"{prefix}.out_proj",
                disable_tp=self.disable_tp_for_gdn,
            )
        self.out_proj._ag2_aux_output_parallel_enabled = (
            self._ag2_aux_boundaries_enabled
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.gdn_prefill_backend = self.chunk_gated_delta_rule.gdn_prefill_backend
        self._prefill_kernels_warmed_up = False
        self._ag2_layer0_trace_enabled = bool(
            os.environ.get("AG2_VLLM_LAYER0_TRACE_OUTPUT")
        ) and prefix.endswith(".layers.0.linear_attn")
        self._ag2_mtp_gdn_row0_reference = (
            os.environ.get("AG2_VLLM_MTP_GDN_ROW0_REFERENCE", "0") == "1"
        )
        self._ag2_mtp_replay_commit = bool(
            vllm_config.additional_config.get("gdn_mtp_replay_commit", False)
        )
        if self._ag2_mtp_replay_commit:
            if self.num_spec < 1:
                raise ValueError("gdn_mtp_replay_commit requires speculative decoding")
            max_reqs = vllm_config.scheduler_config.max_num_seqs
            max_window = self.num_spec + 1
            max_tokens = max_reqs * max_window
            activation_dtype = vllm_config.model_config.dtype
            self.register_buffer(
                "_ag2_mtp_journal_k",
                torch.empty(
                    max_tokens,
                    self.local_num_k_heads,
                    self.head_k_dim,
                    dtype=activation_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_ag2_mtp_journal_v",
                torch.empty(
                    max_tokens,
                    self.local_num_v_heads,
                    self.head_v_dim,
                    dtype=activation_dtype,
                ),
                persistent=False,
            )
            for name in ("a", "b"):
                self.register_buffer(
                    f"_ag2_mtp_journal_{name}",
                    torch.empty(
                        max_tokens,
                        self.local_num_v_heads,
                        dtype=activation_dtype,
                    ),
                    persistent=False,
                )
            self.register_buffer(
                "_ag2_mtp_journal_query_start",
                torch.empty(max_reqs + 1, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "_ag2_mtp_journal_state_ids",
                torch.empty(max_reqs, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "_ag2_mtp_journal_batch_indices",
                torch.empty(max_reqs, dtype=torch.int32),
                persistent=False,
            )
        if self._ag2_layer0_trace_enabled:
            trace_rows = 3
            self.out_proj.register_buffer(
                "_ag2_trace_output_parallel",
                torch.zeros(
                    (trace_rows, self.hidden_size),
                    dtype=vllm_config.model_config.dtype,
                ),
                persistent=False,
            )
            trace_buffers = {
                "qkvz": (
                    trace_rows,
                    2 * self.padded_local_key_dim + 2 * self.padded_local_value_dim,
                ),
                "ba": (trace_rows, 2 * self.padded_local_num_v_heads),
                "post_conv_qkv": (trace_rows, self.local_conv_dim),
                "core": (trace_rows, self.padded_local_value_dim),
                "z": (trace_rows, self.padded_local_value_dim),
                "gated_norm": (trace_rows, self.padded_local_value_dim),
                "attention_output": (trace_rows, self.hidden_size),
            }
            for name, shape in trace_buffers.items():
                self.register_buffer(
                    f"_ag2_trace_{name}",
                    torch.zeros(shape, dtype=vllm_config.model_config.dtype),
                    persistent=False,
                )
            self.register_buffer(
                "_ag2_trace_ssm_state",
                torch.zeros(
                    (
                        trace_rows,
                        self.local_num_v_heads,
                        self.head_v_dim,
                        self.head_k_dim,
                    ),
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_ag2_trace_conv_state",
                torch.zeros(
                    (
                        trace_rows,
                        self.padded_local_conv_dim,
                        self.conv_kernel_size - 1 + self.num_spec,
                    ),
                    dtype=vllm_config.model_config.dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_ag2_trace_state_indices",
                torch.full((trace_rows,), -1, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "_ag2_trace_num_accepted_tokens",
                torch.full((trace_rows,), -1, dtype=torch.int32),
                persistent=False,
            )
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
            and not self.gdn_explicit_partition
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _record_mtp_replay_journal(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> None:
        """Capture raw recurrence inputs while leaving live FP32 state frozen."""
        num_reqs = attn_metadata.num_spec_decodes
        query_start = attn_metadata.spec_query_start_loc
        state_ids = attn_metadata.spec_state_indices_tensor
        batch_indices = attn_metadata.spec_batch_indices
        assert num_reqs > 0
        assert query_start is not None
        assert state_ids is not None
        assert batch_indices is not None
        num_tokens = key.shape[1]
        self._ag2_mtp_journal_k[:num_tokens].copy_(key.squeeze(0))
        self._ag2_mtp_journal_v[:num_tokens].copy_(value.squeeze(0))
        # Match the exact rows consumed by the current recurrent verifier.
        self._ag2_mtp_journal_a[:num_tokens].copy_(
            a[:num_tokens, : self.local_num_v_heads]
        )
        self._ag2_mtp_journal_b[:num_tokens].copy_(
            b[:num_tokens, : self.local_num_v_heads]
        )
        self._ag2_mtp_journal_query_start[: num_reqs + 1].copy_(
            query_start[: num_reqs + 1]
        )
        self._ag2_mtp_journal_state_ids[:num_reqs].copy_(state_ids[:num_reqs, 0])
        self._ag2_mtp_journal_batch_indices[:num_reqs].copy_(
            batch_indices[:num_reqs]
        )

    def commit_mtp_replay_journal(
        self,
        accepted: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Replay exactly the accepted raw-input prefix into the live state."""
        if not self._ag2_mtp_replay_commit:
            return
        if num_reqs == 0:
            return
        source_cu = self._ag2_mtp_journal_query_start[: num_reqs + 1]
        replay_ids = self._ag2_mtp_journal_state_ids[:num_reqs, None].expand(
            -1, self.num_spec + 1
        )
        journal_k = self._ag2_mtp_journal_k.unsqueeze(0)
        fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log[: self.local_num_v_heads],
            a=self._ag2_mtp_journal_a,
            b=self._ag2_mtp_journal_b,
            dt_bias=self.dt_bias[: self.local_num_v_heads],
            q=journal_k,
            k=journal_k,
            v=self._ag2_mtp_journal_v.unsqueeze(0),
            initial_state=self.kv_cache[1],
            inplace_final_state=True,
            store_output=False,
            cu_seqlens=source_cu,
            ssm_state_indices=replay_ids,
            sequence_lengths=accepted[:num_reqs],
            use_qk_l2norm_in_kernel=True,
        )

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        disable_tp: bool = False,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        if self.gdn_explicit_partition and not self.gqa_interleaved_layout:
            partition = self.gdn_partition
            return ExplicitPaddedMergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=output_sizes,
                padded_output_sizes=partition.padded_qkvz_output_sizes,
                local_starts=[
                    partition.k_start * self.head_k_dim,
                    partition.k_start * self.head_k_dim,
                    partition.v_start * self.head_v_dim,
                    partition.v_start * self.head_v_dim,
                ],
                local_sizes=[
                    partition.local_key_dim,
                    partition.local_key_dim,
                    partition.local_value_dim,
                    partition.local_value_dim,
                ],
                bias=False,
                quant_config=quant_config,
                prefix=prefix,
            )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=disable_tp,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        disable_tp: bool = False,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        if self.gdn_explicit_partition and not self.gqa_interleaved_layout:
            partition = self.gdn_partition
            return ExplicitPaddedMergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=output_sizes,
                padded_output_sizes=partition.padded_ba_output_sizes,
                local_starts=[partition.v_start, partition.v_start],
                local_sizes=[partition.v_count, partition.v_count],
                bias=False,
                quant_config=quant_config,
                prefix=prefix,
                disable_tp=self.maybe_disable_tp(quant_config),
            )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=disable_tp or self.maybe_disable_tp(quant_config),
        )

    def maybe_disable_tp(self, quant_config: QuantizationConfig | None) -> bool:
        """Whether to replicate ba_proj instead of TP-sharding it.

        Marlin requires output_size_per_partition >= MIN_THREAD_N=64, which
        the Qwen3.5 non-interleaved [num_v_heads]*2 layout violates at TP>=2
        (e.g. num_v_heads=64, TP=4 -> 16). Replicating the projection keeps
        each rank above the Marlin threshold; forward() then slices b/a to
        the local TP partition. Qwen3-Next's interleaved [num_v_heads*2]
        layout is unaffected and stays TP-sharded.

        See https://github.com/vllm-project/vllm/issues/35924
        """
        return (
            current_platform.is_cuda()
            and not self.gqa_interleaved_layout
            and isinstance(quant_config, (AutoAWQConfig, AutoGPTQConfig, INCConfig))
        )

    def split_ba(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ba.chunk(2, dim=-1)
        if self.gdn_explicit_partition:
            b = b[:, : self.local_num_v_heads]
            a = a[:, : self.local_num_v_heads]
            return b, a
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            # ba_proj is replicated for Marlin; slice b/a to local TP rank.
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a

    def _split_non_interleaved_qkvz(
        self, mixed_qkvz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.gdn_explicit_partition:
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            return mixed_qkv, z.reshape(z.size(0), -1, self.head_v_dim)

        qkv_size = self.padded_local_key_dim * 2 + self.padded_local_value_dim
        z_size = self.padded_local_value_dim
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z[:, : self.local_value_dim].reshape(
            z.size(0), self.local_num_v_heads, self.head_v_dim
        )
        return mixed_qkv, z

    def _strip_padded_mixed_qkv(self, mixed_qkv: torch.Tensor | None):
        if mixed_qkv is None or not self.gdn_explicit_partition:
            return mixed_qkv
        q, k, v = mixed_qkv.split(
            [
                self.padded_local_key_dim,
                self.padded_local_key_dim,
                self.padded_local_value_dim,
            ],
            dim=-1,
        )
        return torch.cat(
            [
                q[:, : self.local_key_dim],
                k[:, : self.local_key_dim],
                v[:, : self.local_value_dim],
            ],
            dim=-1,
        )

    def _pad_local_value_flat(self, value: torch.Tensor) -> torch.Tensor:
        if not self.gdn_explicit_partition:
            return value
        if self.local_value_dim == self.padded_local_value_dim:
            return value
        return torch.nn.functional.pad(
            value,
            (0, self.padded_local_value_dim - self.local_value_dim),
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """
        Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            mixed_qkv, z_out = self._split_non_interleaved_qkvz(mixed_qkvz)
            b, a = self.split_ba(mixed_ba)
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        new_tensor_shape_qkvz = base_shape_qkvz + (
            ng,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = base_shape_ba + (
            ng,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.local_key_dim
        k_dim = self.local_key_dim
        v_dim = self.local_value_dim

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_method(hidden_states)

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        core_attn_out = self._pad_local_value_flat(core_attn_out)
        if self._ag2_aux_boundaries_enabled:
            self._ag2_aux_gated_norm = core_attn_out
        if self._ag2_layer0_trace_enabled:
            trace = core_attn_out[:3]
            self._ag2_trace_gated_norm[: trace.shape[0]].copy_(trace)
        output, _ = self.out_proj(core_attn_out)
        if self._ag2_gdn_prefill_batch_invariant_reduce:
            output = tensor_model_parallel_gdn_all_reduce(output)
        if self._ag2_aux_boundaries_enabled:
            self._ag2_aux_output_parallel = self.out_proj._ag2_aux_output_parallel
        if self._ag2_layer0_trace_enabled:
            trace = output[:3]
            self._ag2_trace_attention_output[: trace.shape[0]].copy_(trace)
        return output

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path."""
        if GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.local_num_v_heads, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.local_num_v_heads, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )

            torch.ops.vllm.qwen_gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
                use_aiter=True,
            )

            return self._output_projection(core_attn_out, z)
        else:
            return self.forward_cuda(hidden_states)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        if self._ag2_aux_boundaries_enabled:
            self._ag2_aux_qkvz = mixed_qkvz
            self._ag2_aux_ba = ba
        if self._ag2_layer0_trace_enabled:
            qkvz_trace = mixed_qkvz[:3]
            ba_trace = ba[:3]
            self._ag2_trace_qkvz[: qkvz_trace.shape[0]].copy_(qkvz_trace)
            self._ag2_trace_ba[: ba_trace.shape[0]].copy_(ba_trace)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            mixed_qkv, z = self._split_non_interleaved_qkvz(mixed_qkvz)
            b, a = self.split_ba(ba)
            b = b.contiguous()
            a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.local_num_v_heads, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        replay_float_out = None
        replay_state_out = None
        replay_meta_out = None
        if self._ag2_aux_boundaries_enabled:
            replay_float_out = self._ag2_replay_float_buffer
            replay_state_out = self._ag2_replay_state_buffer
            replay_meta_out = self._ag2_replay_meta_buffer

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
            replay_float_out=replay_float_out,
            replay_state_out=replay_state_out,
            replay_meta_out=replay_meta_out,
        )
        if self._ag2_aux_boundaries_enabled:
            assert replay_float_out is not None
            assert replay_state_out is not None
            assert replay_meta_out is not None
            self._ag2_aux_replay_float = replay_float_out
            self._ag2_aux_replay_state = replay_state_out
            self._ag2_aux_replay_meta = replay_meta_out
            self._ag2_aux_core = self._pad_local_value_flat(
                core_attn_out.flatten(-2)
            )
        if self._ag2_layer0_trace_enabled:
            core_flat = self._pad_local_value_flat(core_attn_out.flatten(-2))[:3]
            z_flat = self._pad_local_value_flat(z.flatten(-2))[:3]
            self._ag2_trace_core[: core_flat.shape[0]].copy_(core_flat)
            self._ag2_trace_z[: z_flat.shape[0]].copy_(z_flat)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        return self._output_projection(core_attn_out, z)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.local_num_v_heads, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            mixed_qkv, z = self._split_non_interleaved_qkvz(mixed_qkvz)
            b, a = self.split_ba(ba)

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.local_num_v_heads, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        core_attn_out = self._pad_local_value_flat(core_attn_out)
        out, _ = self.out_proj(core_attn_out)
        return out

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.local_num_k_heads
        num_v_heads = self.local_num_v_heads
        _, state_dtype = self.get_state_dtype()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
        dummy_mixed_qkv = torch.randn(
            T,
            self.local_conv_dim
            if self.gdn_explicit_partition
            else qkv_or_qkvz.shape[-1] - v_dim,
            device=device,
            dtype=dtype,
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self.A_log[:num_v_heads],
            dt_bias=self.dt_bias[:num_v_heads],
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

        # CuteDSL kernels require metadata
        chunk_indices = None
        chunk_offsets = None
        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, T)

        try:
            self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
        finally:
            del (
                dummy_mixed_qkv,
                q,
                k,
                v,
                dummy_a,
                dummy_b,
                g,
                beta,
                state,
                cu_seqlens,
                chunk_indices,
                chunk_offsets,
            )

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill) interleaved-GQA layouts,
        dispatches directly to ``_forward_core_decode_aiter``. Otherwise unpacks
        the packed layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # The AITER fused reshape/conv kernel expects Qwen3-Next's interleaved
        # GQA layout. Qwen3.5 uses a non-interleaved q/k/v/z layout and must use
        # the generic path below to split/rearrange inputs correctly.
        if (
            self.gqa_interleaved_layout
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_aiter(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        core_attn_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

    @staticmethod
    def _capture_gdn_replay_inputs(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        chunk_indices: torch.Tensor,
        chunk_offsets: torch.Tensor,
        replay_float_out: torch.Tensor | None,
        replay_state_out: torch.Tensor | None,
        replay_meta_out: torch.Tensor | None,
    ) -> None:
        """Copy one short packed FLA call into explicit custom-op outputs.

        Python attribute mutation inside a custom op is invisible to
        torch.compile/CUDA Graph replay.  These buffers are instead declared as
        mutable custom-op arguments, so the captured copies are authoritative.
        Metadata layout:

        - [0]: status (1=captured, -2=capacity exceeded)
        - [1:8]: float lengths for q/k/v/g/beta, state length, sequence count
        - [8:28]: q/k/v/g/beta shapes, four dimensions each
        - [28:32]: initial-state shape
        - [32:36]: cu/chunk-index/chunk-offset lengths and chunk-index columns
        - [64...]: concatenated integer addressing metadata
        """
        if (
            replay_float_out is None
            or replay_state_out is None
            or replay_meta_out is None
        ):
            return

        tensors = (q, k, v, g, beta)
        lengths = tuple(tensor.numel() for tensor in tensors)
        float_required = sum(lengths)
        state_required = initial_state.numel()
        integer_required = (
            cu_seqlens.numel() + chunk_indices.numel() + chunk_offsets.numel()
        )
        fits = (
            q.shape[1] <= _AG2_GDN_REPLAY_MAX_TOKENS
            and float_required <= replay_float_out.numel()
            and state_required <= replay_state_out.numel()
            and 64 + integer_required <= replay_meta_out.numel()
        )
        replay_meta_out.fill_(-1)
        replay_meta_out[0] = 1 if fits else -2
        if not fits:
            return

        offset = 0
        for tensor, length in zip(tensors, lengths, strict=True):
            replay_float_out[offset : offset + length].copy_(
                tensor.reshape(-1).float()
            )
            offset += length
        replay_state_out[:state_required].copy_(initial_state.reshape(-1).float())

        replay_meta_out[1:8].copy_(
            torch.tensor(
                (*lengths, state_required, initial_state.shape[0]),
                dtype=torch.int64,
                device=replay_meta_out.device,
            )
        )
        for index, tensor in enumerate(tensors):
            shape = (*tensor.shape, 1, 1, 1, 1)[:4]
            replay_meta_out[8 + 4 * index : 12 + 4 * index].copy_(
                torch.tensor(
                    shape,
                    dtype=torch.int64,
                    device=replay_meta_out.device,
                )
            )
        state_shape = (*initial_state.shape, 1, 1, 1, 1)[:4]
        replay_meta_out[28:32].copy_(
            torch.tensor(
                state_shape,
                dtype=torch.int64,
                device=replay_meta_out.device,
            )
        )
        chunk_index_columns = chunk_indices.shape[-1] if chunk_indices.ndim > 1 else 1
        replay_meta_out[32:36].copy_(
            torch.tensor(
                (
                    cu_seqlens.numel(),
                    chunk_indices.numel(),
                    chunk_offsets.numel(),
                    chunk_index_columns,
                ),
                dtype=torch.int64,
                device=replay_meta_out.device,
            )
        )
        dtype_codes = {
            torch.bfloat16: 1,
            torch.float32: 2,
            torch.float16: 3,
        }
        replay_meta_out[36:38].copy_(
            torch.tensor(
                (
                    dtype_codes.get(initial_state.dtype, 0),
                    dtype_codes.get(q.dtype, 0),
                ),
                dtype=torch.int64,
                device=replay_meta_out.device,
            )
        )
        integer_offset = 64
        for tensor in (cu_seqlens, chunk_indices, chunk_offsets):
            length = tensor.numel()
            replay_meta_out[integer_offset : integer_offset + length].copy_(
                tensor.reshape(-1).to(torch.int64)
            )
            integer_offset += length

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        replay_float_out: torch.Tensor | None = None,
        replay_state_out: torch.Tensor | None = None,
        replay_meta_out: torch.Tensor | None = None,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        ssm_state = (
            ssm_state[:, : self.local_num_v_heads, :, :]
            if self.gdn_explicit_partition
            else ssm_state
        )
        if self._ag2_layer0_trace_enabled:
            traced_indices = (
                spec_state_indices_tensor.reshape(-1)
                if spec_sequence_masks is not None
                else non_spec_state_indices_tensor.reshape(-1)
            )[:3]
            traced_conv_state = conv_state.index_select(
                0, traced_indices.to(dtype=torch.long)
            )
            self._ag2_trace_conv_state[: traced_conv_state.shape[0]].copy_(
                traced_conv_state
            )
        A_log = self.A_log[: self.local_num_v_heads]
        dt_bias = self.dt_bias[: self.local_num_v_heads]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        row0_reference_conv = None
        row0_reference_ssm = None
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            if self._ag2_mtp_gdn_row0_reference and forward_context.tp3_sd_phase_reduce:
                if attn_metadata.num_spec_decodes != 1:
                    raise RuntimeError("MTP GDN row-zero reference POC admits x1 only")
                committed_index = spec_state_indices_tensor[:1, 0].long()
                reference_conv_state = conv_state.index_select(
                    0, committed_index
                ).clone()
                reference_state_index = torch.zeros(
                    1,
                    dtype=torch.int32,
                    device=mixed_qkv_spec.device,
                )
                row0_reference_conv = causal_conv1d_update(
                    mixed_qkv_spec[:1].clone(),
                    reference_conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=reference_state_index,
                    validate_data=True,
                )
                row0_reference_ssm = ssm_state.index_select(0, committed_index).clone()
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        # The packed projection input has been consumed by the convolution
        # paths above. Drop this reference before recurrent prefill; the
        # convolution outputs are separate tensors (or retain the storage
        # through their own alias in the spec-only path).
        del mixed_qkv
        # Spec/decode tensors are tiny and retain the existing compact layout.
        # Prefill goes directly from padded TP3 conv output into
        # fused_post_conv_prep, whose input offsets distinguish padded input
        # heads from real output heads. Avoid materializing a 42-50 MiB
        # compacting torch.cat at the 6656-token production chunk boundary.
        mixed_qkv_spec = self._strip_padded_mixed_qkv(mixed_qkv_spec)
        if attn_metadata.num_prefills == 0:
            mixed_qkv_non_spec = self._strip_padded_mixed_qkv(mixed_qkv_non_spec)
        if self._ag2_layer0_trace_enabled:
            traced_conv = (
                mixed_qkv_spec
                if spec_sequence_masks is not None
                else mixed_qkv_non_spec
            )
            assert traced_conv is not None
            traced_conv = self._strip_padded_mixed_qkv(traced_conv[:3])
            self._ag2_trace_post_conv_qkv[: traced_conv.shape[0]].copy_(traced_conv)

            traced_indices = (
                spec_state_indices_tensor.reshape(-1)
                if spec_sequence_masks is not None
                else non_spec_state_indices_tensor.reshape(-1)
            )[:3]
            self._ag2_trace_state_indices.fill_(-1)
            self._ag2_trace_state_indices[: traced_indices.shape[0]].copy_(
                traced_indices
            )
            traced_state = ssm_state.index_select(
                0, traced_indices.to(dtype=torch.long)
            )
            self._ag2_trace_ssm_state[: traced_state.shape[0]].copy_(traced_state)
            self._ag2_trace_num_accepted_tokens.fill_(-1)
            if num_accepted_tokens is not None:
                traced_accepted = num_accepted_tokens.reshape(-1)[:3]
                self._ag2_trace_num_accepted_tokens[: traced_accepted.shape[0]].copy_(
                    traced_accepted
                )
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        # Split mixed non-spec-decode+prefill to process independently
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            mixed_qkv_decode = None
            if split_non_spec:
                # Only this tiny decode prefix is read after prefill prep.
                # Copy it so the full packed convolution output can be released
                # before the large recurrent-state allocation.
                mixed_qkv_decode = self._strip_padded_mixed_qkv(
                    mixed_qkv_non_spec[:num_decode_tokens].clone()
                )
                conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a_non_spec[num_decode_tokens:]
                b_prefill = b_non_spec[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv_non_spec
                a_prefill = a_non_spec
                b_prefill = b_non_spec

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=conv_output_prefill,
                a=a_prefill,
                b=b_prefill,
                A_log=A_log,
                dt_bias=dt_bias,
                num_k_heads=self.local_num_k_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
                input_num_k_heads=(
                    self.padded_local_num_k_heads
                    if self.gdn_explicit_partition
                    else self.local_num_k_heads
                ),
                input_num_v_heads=(
                    self.padded_local_num_v_heads
                    if self.gdn_explicit_partition
                    else self.local_num_v_heads
                ),
            )
            del conv_output_prefill, mixed_qkv_non_spec
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            if self._ag2_mtp_replay_commit:
                self._record_mtp_replay_journal(
                    key=key_spec,
                    value=value_spec,
                    a=a,
                    b=b,
                    attn_metadata=attn_metadata,
                )
            row0_reference_core = None
            if row0_reference_conv is not None:
                assert row0_reference_ssm is not None
                reference_q, reference_k, reference_v = self.rearrange_mixed_qkv(
                    self._strip_padded_mixed_qkv(row0_reference_conv)
                )
                reference_state_index = torch.zeros(
                    1,
                    dtype=torch.int32,
                    device=reference_q.device,
                )
                reference_cu = torch.tensor(
                    [0, 1],
                    dtype=torch.int32,
                    device=reference_q.device,
                )
                reference_token_index = spec_token_indx[:1]
                row0_reference_core, _ = fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    a=a.index_select(0, reference_token_index),
                    b=b.index_select(0, reference_token_index),
                    dt_bias=dt_bias,
                    q=reference_q,
                    k=reference_k,
                    v=reference_v,
                    initial_state=row0_reference_ssm,
                    inplace_final_state=True,
                    cu_seqlens=reference_cu,
                    ssm_state_indices=reference_state_index,
                    use_qk_l2norm_in_kernel=True,
                )
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    a=a,
                    b=b,
                    dt_bias=dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    store_final_state=not self._ag2_mtp_replay_commit,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
            if row0_reference_core is not None:
                core_attn_out_spec[:, :1].copy_(row0_reference_core)
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part
        if split_non_spec:
            assert mixed_qkv_decode is not None
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_decode
            )
            core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                dt_bias=dt_bias,
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
        if attn_metadata.num_prefills > 0:
            # State indices, initial-state mask and cu_seqlens for the chunk
            # kernel are precomputed by the metadata builder (the prefill tail
            # when decodes are peeled off, else the full non-spec batch), so they
            # don't need to be re-derived per layer.
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            initial_state = ssm_state[prefill_state_indices]
            initial_state[~prefill_has_initial_state, ...] = 0
            # The outer custom op already owns the final core-attention output.
            # Write the prefill result directly into its decode-first slice
            # instead of allocating another value-sized tensor in chunk_fwd_o.
            # This is especially important at the 8K-token prefill boundary,
            # where the redundant BF16 output is 32 MiB per rank.
            prefill_output_start = num_decode_tokens if split_non_spec else 0
            prefill_num_tokens = query_non_spec.shape[1]
            prefill_output = core_attn_out[
                prefill_output_start : prefill_output_start + prefill_num_tokens
            ].unsqueeze(0)
            self._capture_gdn_replay_inputs(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                replay_float_out=replay_float_out,
                replay_state_out=replay_state_out,
                replay_meta_out=replay_meta_out,
            )
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                core_attn_out=prefill_output,
            )
            # Init cache
            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)

            if split_non_spec:
                # The prefill tail is already in its final destination. Fill
                # only the small peeled-decode prefix; avoid a second full
                # output allocation from torch.cat.
                core_attn_out[:num_decode_tokens].copy_(core_attn_out_decode.squeeze(0))
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    a=a,
                    b=b,
                    dt_bias=dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        elif attn_metadata.num_prefills > 0:
            # Pure/mixed non-spec prefill was produced directly in
            # core_attn_out above.
            pass
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

    def _forward_core_decode_aiter(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        return


def qwen_gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
    replay_float_out: torch.Tensor | None = None,
    replay_state_out: torch.Tensor | None = None,
    replay_meta_out: torch.Tensor | None = None,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``use_aiter=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``use_aiter=True`` (AITER Triton path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if use_aiter:
        self._forward_core_rocm(
            qkvz=qkv_or_qkvz,
            ba=b_or_ba,
            z_out=a_or_z_out,
            core_attn_out=core_attn_out,
        )
    else:
        self._forward_core(
            mixed_qkv=qkv_or_qkvz,
            b=b_or_ba,
            a=a_or_z_out,
            core_attn_out=core_attn_out,
            replay_float_out=replay_float_out,
            replay_state_out=replay_state_out,
            replay_meta_out=replay_meta_out,
        )


def gdn_attention_core_fake(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
    replay_float_out: torch.Tensor | None = None,
    replay_state_out: torch.Tensor | None = None,
    replay_meta_out: torch.Tensor | None = None,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="qwen_gdn_attention_core",
    op_func=qwen_gdn_attention_core,
    mutates_args=[
        "a_or_z_out",
        "core_attn_out",
        "replay_float_out",
        "replay_state_out",
        "replay_meta_out",
    ],
    fake_impl=gdn_attention_core_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
