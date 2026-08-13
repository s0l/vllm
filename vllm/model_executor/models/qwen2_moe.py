# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

import os

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2MoeConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_unified_exact_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    PaddedMergedColumnParallelLinear,
    PaddedRowParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _dense_mlp_padded_intermediate_multiple(tp_size: int) -> int:
    # Keep the rank-local K dimension aligned for grouped W4A16/AWQ kernels.
    return tp_size * 32


def _ag2_bf16_mm_out_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Keep a caller-owned BF16 GEMM destination opaque to functionalization."""
    torch.mm(input_, weight.t(), out=output)


def _ag2_bf16_mm_out_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del input_, weight, output


direct_register_custom_op(
    op_name="ag2_bf16_mm_out",
    op_func=_ag2_bf16_mm_out_impl,
    mutates_args=["output"],
    fake_impl=_ag2_bf16_mm_out_fake,
    tags=(torch.Tag.cudagraph_unsafe,),
)


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self._ag2_tp3_unified_exact_reduce = (
            os.environ.get("AG2_VLLM_TP3_UNIFIED_EXACT_REDUCE", "0") == "1"
            and reduce_results
            and not is_sequence_parallel
            and tp_size == 3
        )
        down_reduce_results = (
            reduce_results and not self._ag2_tp3_unified_exact_reduce
        )
        needs_padding = not is_sequence_parallel and intermediate_size % tp_size != 0
        if needs_padding:
            padded_intermediate_size = _ceil_to_multiple(
                intermediate_size,
                _dense_mlp_padded_intermediate_multiple(tp_size),
            )
            self.gate_up_proj = PaddedMergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                [padded_intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = PaddedRowParallelLinear(
                intermediate_size,
                padded_intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=down_reduce_results,
                prefix=f"{prefix}.down_proj",
            )
        else:
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                disable_tp=is_sequence_parallel,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=down_reduce_results,
                disable_tp=is_sequence_parallel,
                prefix=f"{prefix}.down_proj",
            )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate
        self.register_buffer(
            "_bf16_gate_up_scratch",
            None,
            persistent=False,
        )

    def ag2_enable_projection_calibration_capture(
        self, capacity: int, dtype: torch.dtype
    ) -> None:
        if capacity < 1:
            raise ValueError("projection capture capacity must be positive")
        self._ag2_projection_capture_enabled = True
        self.register_buffer(
            "_ag2_projection_capture_activation",
            torch.full(
                (capacity, self.down_proj.input_size_per_partition),
                torch.nan,
                dtype=dtype,
            ),
            persistent=False,
        )

    def enable_bf16_gate_up_scratch(
        self,
        max_num_tokens: int,
        scratch: torch.Tensor | None = None,
    ) -> None:
        """Give an unquantized dense MLP a compiler-visible output scratch.

        Ownership lives on the MLP because the caller must provide the buffer
        to ``torch.mm(..., out=...)`` before Inductor plans the gate/up GEMM.
        Selecting a buffer inside a lower linear kernel is too late: Inductor
        has already emitted a fresh output allocation at that point.
        """
        if max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be positive")
        if not isinstance(
            self.gate_up_proj.quant_method,
            UnquantizedLinearMethod,
        ):
            raise ValueError("BF16 gate/up scratch requires an unquantized MLP")
        if self.gate_up_proj.bias is not None:
            raise ValueError("BF16 gate/up scratch does not support bias")
        if self.gate_up_proj.gather_output:
            raise ValueError("BF16 gate/up scratch requires rank-local output")
        weight = self.gate_up_proj.weight
        if scratch is None:
            scratch = torch.empty(
                (max_num_tokens, weight.shape[0]),
                dtype=weight.dtype,
                device=weight.device,
            )
        elif (
            scratch.ndim != 2
            or scratch.shape[0] < max_num_tokens
            or scratch.shape[1] != weight.shape[0]
            or scratch.dtype != weight.dtype
            or scratch.device != weight.device
            or not scratch.is_contiguous()
        ):
            raise ValueError(
                "Shared BF16 gate/up scratch does not match the MTP layer: "
                f"scratch={tuple(scratch.shape)} {scratch.dtype} "
                f"{scratch.device}; required=({max_num_tokens}, "
                f"{weight.shape[0]}) {weight.dtype} {weight.device}"
            )
        self._bf16_gate_up_scratch = scratch

    def _gate_up(self, x: torch.Tensor) -> torch.Tensor:
        scratch = self._bf16_gate_up_scratch
        if scratch is None:
            gate_up, _ = self.gate_up_proj(x)
            return gate_up
        if x.ndim != 2:
            raise RuntimeError(
                "BF16 gate/up scratch currently requires a two-dimensional input"
            )
        if x.shape[0] > scratch.shape[0]:
            raise RuntimeError(
                "BF16 gate/up scratch is smaller than the runtime token batch: "
                f"rows={x.shape[0]} capacity={scratch.shape[0]}"
            )
        output = scratch[: x.shape[0]]
        torch.ops.vllm.ag2_bf16_mm_out(x, self.gate_up_proj.weight, output)
        return output

    def forward(self, x):
        gate_up = self._gate_up(x)
        out = self.act_fn(gate_up)
        if getattr(self, "_ag2_aux_full_trace_enabled", False):
            self._ag2_aux_full_gate_up = gate_up
            self._ag2_aux_full_activation = out
        if getattr(self, "_ag2_aux_compact_trace_enabled", False):
            row_indices = self._ag2_aux_compact_row_indices

            def compact(value: torch.Tensor) -> torch.Tensor:
                valid = row_indices >= 0
                safe = torch.where(valid, row_indices, torch.zeros_like(row_indices))
                selected = torch.index_select(value, 0, safe)
                return torch.where(
                    valid.unsqueeze(-1), selected, torch.zeros_like(selected)
                )

            self._ag2_aux_compact_gate_up = compact(gate_up)
            self._ag2_aux_compact_activation = compact(out)
        if getattr(self, "_ag2_projection_capture_enabled", False):
            row_indices = self._ag2_projection_capture_row_indices
            valid = row_indices >= 0
            safe = torch.where(valid, row_indices, torch.zeros_like(row_indices))
            selected = torch.index_select(out, 0, safe)
            selected = torch.where(
                valid.unsqueeze(-1), selected, torch.zeros_like(selected)
            )
            self._ag2_projection_capture_activation.copy_(selected)
        out, _ = self.down_proj(out)
        if self._ag2_tp3_unified_exact_reduce:
            out = tensor_model_parallel_unified_exact_all_reduce(out)

        if getattr(self, "_ag2_aux_full_trace_enabled", False):
            self._ag2_aux_full_down_parallel = (
                self.down_proj._ag2_aux_output_parallel
            )

        if getattr(self, "_ag2_aux_compact_trace_enabled", False):
            self._ag2_aux_compact_down_parallel = compact(
                self.down_proj._ag2_aux_output_parallel
            )
            self._ag2_aux_compact_output = compact(out)

        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out

        if getattr(self, "_ag2_aux_full_trace_enabled", False):
            self._ag2_aux_full_output = out

        return out


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
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

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_expert,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(orig_shape)


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen2MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
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
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    fall_back_to_pt_during_load = False
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            # .experts.gate_up_proj must be handled by MoERunner.load_weights for EP
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen2MoeMLP exists
        if (
            getattr(config, "mlp_only_layers", [])
            or config.shared_expert_intermediate_size > 0
        ):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen2MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _maybe_reshape(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                # GGUF: make sure that shared_expert_gate is a 2D tensor.
                if "mlp.shared_expert_gate" in name and loaded_weight.dim() == 1:
                    loaded_weight = loaded_weight[None, :]
                yield name, loaded_weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            _maybe_reshape(weights), mapper=self.hf_to_vllm_mapper
        )
