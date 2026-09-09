# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

import os
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.head_partition import (
    make_gdn_head_partition,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    AG2_COMPACT_GDN_STAGE_ORDER,
    AG2_COMPACT_MLP_STAGE_ORDER,
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
    _ag2_internal_trace_layer_enabled,
    _ag2_selected_stages,
    _ag2_tp3_mtp_block5_enabled,
    _ag2_tp3_owner_prequant_enabled,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


def _ag2_layer0_trace_match_row(
    *,
    output: str | None,
    saved_query_lens: set[int],
    query_len: int,
    expected_query_lens: set[int],
    token_ids: list[int],
    expected_token_id: int,
    positions: list[int],
    expected_position: int,
) -> int | None:
    if (
        not output
        or query_len in saved_query_lens
        or query_len not in expected_query_lens
    ):
        return None
    matches = [
        row
        for row, (token_id, position) in enumerate(
            zip(token_ids[:query_len], positions[:query_len])
        )
        if token_id == expected_token_id and position == expected_position
    ]
    return matches[0] if len(matches) == 1 else None


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        # transformers 5.x renames the top-level Qwen3.5-MoE config class to
        # Qwen3_5MoeTextConfig for text-only models, while transformers ≤4.x
        # returns Qwen3_5MoeConfig (the multimodal wrapper).  Accept both so
        # that vLLM works regardless of which transformers version is installed.
        return self.ctx.get_hf_config((Qwen3_5MoeConfig, Qwen3_5MoeTextConfig))


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

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
        # Qwen3_5 intentionally bypasses Qwen3NextDecoderLayer.__init__, while
        # inheriting its forward. Keep diagnostic forward attributes explicit.
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
        is_moe_layer = config.model_type == "qwen3_5_moe_text"
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
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
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
                prefix=f"{prefix}.self_attn",
                reduce_results=reduce_results,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=not self._ag2_tp3_owner_prequant,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")
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

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
        trace_layer = int(os.environ.get("AG2_VLLM_LAYER_TRACE_LAYER", "0"))
        self._ag2_layer0_trace_enabled = (
            bool(os.environ.get("AG2_VLLM_LAYER0_TRACE_OUTPUT"))
            and self.layer_idx == trace_layer
        )
        self._ag2_aux_attention_boundary_enabled = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_ATTENTION_BOUNDARY", "0"
        ) == "1" and self.layer_idx == int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "0")
        )
        self._ag2_aux_gdn_boundaries_enabled = (
            self.layer_type == "linear_attention"
            and (
                self._ag2_aux_sequence_boundary_enabled
                or (
                    os.environ.get(
                        "AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_GDN_BOUNDARIES", "0"
                    )
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
        )
        self._ag2_aux_gdn_replay_enabled = (
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_GDN_REPLAY_ONLY", "0") == "1"
            and self.layer_idx
            == int(
                os.environ.get(
                    "AG2_VLLM_AUX_HIDDEN_TRACE_GDN_BOUNDARY_LAYER",
                    "0",
                )
            )
            and self.layer_type == "linear_attention"
        )
        if self._ag2_layer0_trace_enabled:
            self.register_buffer(
                "_ag2_trace_positions",
                torch.full((3,), -1, dtype=torch.long),
                persistent=False,
            )
            for name in (
                "input",
                "input_norm",
                "attention_output",
                "post_attention_norm",
                "post_attention_residual",
                "mlp_output",
            ):
                self.register_buffer(
                    f"_ag2_trace_{name}",
                    torch.zeros(
                        (3, config.hidden_size),
                        dtype=vllm_config.model_config.dtype,
                    ),
                    persistent=False,
                )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
)
class Qwen3_5Model(Qwen3NextModel):
    # Qwen3.5 ships the GDN in_proj checkpoints separately (qwen3-next
    # pre-fuses them); fuse them on top of the qwen3-next QKV/gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
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
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # FSE must match construction (Qwen3NextSparseMoeBlock): reroute the
        # shared expert into the extra fused slot only when AITER FSE is both
        # requested and compatible with the quant spec.
        if "moe" in self.config.model_type:
            weights = maybe_fuse_shared_experts(
                weights,
                enabled=self.is_fused_shared_expert_enabled,
                n_routed_experts=self.config.num_experts,
                n_shared_experts=1,
                ckpt_prefix="mlp.shared_expert",
            )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    # Maps PEFT embed/lm_head LoRA targets onto vLLM embedding wrappers.
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Some community text-only checkpoints keep the extraneous
    # `model.language_model.` prefix inherited from the VL training stack.
    # Strip it so both prefixed and clean checkpoints load correctly.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model.", "mtp.": None},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def process_weights_after_loading(self) -> None:
        if self.model._ag2_tp3_owner_prequant:
            self.model._ag2_validate_tp3_owner_prequant_weights()
        from vllm.model_executor.models.qwen3_next_row import prepare_row_continuation

        prepare_row_continuation(
            self.model, self.model_config.model, vllm_config=self.vllm_config
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        ready_rows: torch.Tensor | None = None,
        **kwargs: object,
    ):
        binding = getattr(self.model, "_ag2_row_continuation", None)
        ready = binding is not None and binding.ready_compaction
        if ready != (ready_rows is not None):
            raise ValueError("ready target invocation must match its prepared recipe")
        physical_rows = None
        if ready_rows is not None:
            from vllm.model_executor.models.qwen3_next_ready import compact_inputs

            input_ids, positions, inputs_embeds, physical_rows = compact_inputs(
                input_ids, positions, inputs_embeds, ready_rows
            )
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
        if physical_rows is not None:
            from vllm.model_executor.models.qwen3_next_ready import physical_output

            hidden_states = physical_output(hidden_states, physical_rows)
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
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def compute_local_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        return self.logits_processor.get_local_logits(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()


########################################################
# Qwen3_5-Dense
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    supports_multimodal_pruning = True

    hf_to_vllm_mapper = (
        Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(orig_to_new_prefix={"mtp.": None})
    )

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def compute_local_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        return self.language_model.compute_local_logits(hidden_states)

    def process_weights_after_loading(self) -> None:
        self.language_model.process_weights_after_loading()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # attributes needed by EVS-related functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if multimodal_config.language_model_only:
            self.visual = None
        else:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self._ag2_layer0_trace_output = os.environ.get("AG2_VLLM_LAYER0_TRACE_OUTPUT")
        self._ag2_layer0_trace_layer = int(
            os.environ.get("AG2_VLLM_LAYER_TRACE_LAYER", "0")
        )
        query_lens_raw = os.environ.get(
            "AG2_VLLM_LAYER0_TRACE_QUERY_LENS"
        ) or os.environ.get("AG2_VLLM_LAYER0_TRACE_QUERY_LEN", "")
        self._ag2_layer0_trace_query_lens = {
            int(value) for value in query_lens_raw.split(",") if value
        }
        self._ag2_layer0_trace_token_id = int(
            os.environ.get("AG2_VLLM_LAYER0_TRACE_TOKEN_ID", "-1")
        )
        self._ag2_layer0_trace_position = int(
            os.environ.get("AG2_VLLM_LAYER0_TRACE_POSITION", "-1")
        )
        self._ag2_layer0_trace_saved_query_lens: set[int] = set()
        if self._ag2_layer0_trace_output and (
            not self._ag2_layer0_trace_query_lens
            or not self._ag2_layer0_trace_query_lens.issubset({1, 3})
            or self._ag2_layer0_trace_token_id < 0
            or self._ag2_layer0_trace_position < 0
        ):
            raise ValueError(
                "Layer trace requires query lengths in {1,3}, "
                "a token id and an absolute position"
            )

    def maybe_save_ag2_layer0_trace(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_len: int,
        cudagraph_mode: str,
    ) -> None:
        output = self._ag2_layer0_trace_output
        if not output:
            return
        position_values = positions[0] if positions.ndim == 2 else positions
        token_ids = input_ids[:query_len].detach().cpu().tolist()
        position_ids = position_values[:query_len].detach().cpu().tolist()
        match_row = _ag2_layer0_trace_match_row(
            output=output,
            saved_query_lens=self._ag2_layer0_trace_saved_query_lens,
            query_len=query_len,
            expected_query_lens=self._ag2_layer0_trace_query_lens,
            token_ids=token_ids,
            expected_token_id=self._ag2_layer0_trace_token_id,
            positions=position_ids,
            expected_position=self._ag2_layer0_trace_position,
        )
        if match_row is None:
            return

        layer_idx = self._ag2_layer0_trace_layer
        layer = self.language_model.model.layers[layer_idx]
        rank = get_tp_group().rank_in_group
        path = Path(f"{output}.q{query_len}.rank{rank}.pt")
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite layer trace: {path}")

        graph_positions = layer._ag2_trace_positions[:query_len]
        expected_positions = position_values[:query_len]
        # CUDA Graph replay is asynchronous with respect to this host-side
        # diagnostic. Synchronize before reading graph-written buffers.
        torch.cuda.current_stream().synchronize()
        graph_marker_matches = torch.equal(graph_positions, expected_positions)
        if not graph_marker_matches:
            logger.warning(
                "Layer trace graph position marker was not replayed: "
                "graph=%s request=%s. Saving the diagnostic with "
                "graph_marker_matches=false; host input ids/positions remain "
                "the trace identity.",
                graph_positions.detach().cpu().tolist(),
                expected_positions.detach().cpu().tolist(),
            )

        stages = {}
        for name in (
            "input",
            "input_norm",
            "attention_output",
            "post_attention_norm",
            "post_attention_residual",
            "mlp_output",
        ):
            stages[f"decoder.{name}"] = (
                getattr(layer, f"_ag2_trace_{name}")[:query_len].detach().cpu().clone()
            )
        trace_metadata = {"layer_idx": layer_idx, "layer_type": layer.layer_type}
        if layer.layer_type == "linear_attention":
            linear_attn = layer.linear_attn
            for name in (
                "qkvz",
                "ba",
                "post_conv_qkv",
                "core",
                "z",
                "gated_norm",
                "attention_output",
            ):
                stages[f"gdn.{name}"] = (
                    getattr(linear_attn, f"_ag2_trace_{name}")[:query_len]
                    .detach()
                    .cpu()
                    .clone()
                )
            stages["gdn.local_attention_output"] = (
                linear_attn.out_proj._ag2_trace_output_parallel[:query_len]
                .detach()
                .cpu()
                .clone()
            )
            stages["gdn.ssm_state"] = (
                linear_attn._ag2_trace_ssm_state.detach().cpu().clone()
            )
            stages["gdn.conv_state"] = (
                linear_attn._ag2_trace_conv_state.detach().cpu().clone()
            )
            stages["gdn.state_indices"] = (
                linear_attn._ag2_trace_state_indices.detach().cpu().clone()
            )
            stages["gdn.num_accepted_tokens"] = (
                linear_attn._ag2_trace_num_accepted_tokens.detach().cpu().clone()
            )
            trace_metadata.update(
                {
                    "local_num_k_heads": linear_attn.local_num_k_heads,
                    "local_num_v_heads": linear_attn.local_num_v_heads,
                    "padded_local_num_k_heads": (linear_attn.padded_local_num_k_heads),
                    "padded_local_num_v_heads": (linear_attn.padded_local_num_v_heads),
                }
            )
        elif layer.layer_type == "full_attention":
            full_attn = layer.self_attn
            for name in (
                "qkv",
                "q",
                "k",
                "v",
                "gate",
                "core",
                "gated_output",
                "attention_output",
            ):
                stages[f"full.{name}"] = (
                    getattr(full_attn, f"_ag2_trace_{name}")[:query_len]
                    .detach()
                    .cpu()
                    .clone()
                )
            stages["full.local_attention_output"] = (
                full_attn.o_proj._ag2_trace_output_parallel[:query_len]
                .detach()
                .cpu()
                .clone()
            )
            dcp_trace_names = (
                (
                    "decode_local_output",
                    "decode_local_lse",
                    "decode_combined_output",
                    "decode_combined_lse",
                )
                if query_len == 1
                else (
                    "prefill_context_local_output",
                    "prefill_context_local_lse",
                    "prefill_context_combined_output",
                    "prefill_context_combined_lse",
                    "prefill_query_output",
                    "prefill_query_lse",
                    "prefill_merged_output",
                )
            )
            for name in dcp_trace_names:
                stages[f"full.dcp.{name}"] = (
                    getattr(full_attn.attn, f"_ag2_dcp_trace_{name}")[:query_len]
                    .detach()
                    .cpu()
                    .clone()
                )
            prefix_end = self._ag2_layer0_trace_position + 1
            prefix_positions = full_attn._ag2_trace_prefix_positions[:prefix_end]
            expected_prefix_positions = torch.arange(
                prefix_end,
                device=prefix_positions.device,
                dtype=prefix_positions.dtype,
            )
            if not torch.equal(prefix_positions, expected_prefix_positions):
                raise RuntimeError(
                    "Full-attention prefix trace is incomplete: "
                    f"observed={prefix_positions.detach().cpu().tolist()}"
                )
            stages["full.prefix_k"] = (
                full_attn._ag2_trace_prefix_k[:prefix_end].detach().cpu().clone()
            )
            stages["full.prefix_v"] = (
                full_attn._ag2_trace_prefix_v[:prefix_end].detach().cpu().clone()
            )
            trace_metadata.update(
                {
                    "local_num_heads": full_attn.num_heads,
                    "local_num_kv_heads": full_attn.num_kv_heads,
                    "total_num_heads": full_attn.total_num_heads,
                    "total_num_kv_heads": full_attn.total_num_kv_heads,
                    "head_dim": full_attn.head_dim,
                    "scale": full_attn.scaling,
                    "local_kv_head_indices": tuple(
                        full_attn.attn_head_partition.kv_head_indices
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported trace layer type {layer.layer_type}")

        torch.save(
            {
                "rank": rank,
                "query_len": query_len,
                "cudagraph_mode": cudagraph_mode,
                "input_ids": input_ids[:query_len].detach().cpu().clone(),
                "positions": position_values[:query_len].detach().cpu().clone(),
                "graph_positions": graph_positions.detach().cpu().clone(),
                "graph_marker_matches": graph_marker_matches,
                "match_row": match_row,
                "stages": stages,
                **trace_metadata,
            },
            path,
        )
        self._ag2_layer0_trace_saved_query_lens.add(query_len)
        logger.warning(
            "Saved matched layer-%d phase trace rank=%d qlen=%d row=%d "
            "token=%d position=%d mode=%s path=%s",
            layer_idx,
            rank,
            query_len,
            match_row,
            token_ids[match_row],
            position_ids[match_row],
            cudagraph_mode,
            path,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def embed_text_input_ids_for_elastic(
        self,
        input_ids: torch.Tensor,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run only the TP text-embedding phase for elastic MM execution."""
        return self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

    def merge_multimodal_embeddings_for_elastic(
        self,
        text_embeddings: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        *,
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        """Run the rank-local MM merge after the TP embedding collective."""
        return _merge_multimodal_embeddings(
            inputs_embeds=text_embeddings,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        ready_rows: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # The language-model wrapper owns the exterior P/M invocation adapter.
        # Calling its compiled backbone directly would bypass that contract.
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            ready_rows=ready_rows,
        )

        return hidden_states

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["mtp."]
        if self.multimodal_config.language_model_only:
            skip_prefixes.append("visual.")
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=skip_prefixes,
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

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
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        if (
            hf_config.linear_num_key_heads % tp_size != 0
            or hf_config.linear_num_value_heads % tp_size != 0
        ):
            partition = make_gdn_head_partition(
                num_k_heads=hf_config.linear_num_key_heads,
                num_v_heads=hf_config.linear_num_value_heads,
                head_k_dim=hf_config.linear_key_head_dim,
                head_v_dim=hf_config.linear_value_head_dim,
                tp_size=tp_size,
                tp_rank=0,
            )
            conv_state_shape = MambaStateShapeCalculator._orient_conv_shape(
                partition.padded_conv_dim,
                hf_config.linear_conv_kernel_dim - 1 + num_spec,
            )
            temporal_state_shape = (
                partition.max_v_count,
                hf_config.linear_value_head_dim,
                hf_config.linear_key_head_dim,
            )
            return (
                conv_state_shape,
                temporal_state_shape,
                conv_state_shape,
                temporal_state_shape,
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


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # attributes needed by EVS-related functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
