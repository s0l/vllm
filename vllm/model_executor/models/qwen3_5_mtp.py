# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3_5 MTP model."""

from collections.abc import Iterable
import os

import torch
from torch import nn

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    get_nvfp4_marlin_gate_up_scratch,
)
from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.linear import PaddedMergedColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.ag2_fp8_draft_head import (
    Fp8DraftHead,
    install_shared_fp8_lm_head,
    shared_fp8_head_enabled,
)
from vllm.model_executor.models.ag2_folded_mtp import (
    FoldedMTPNaturalTP2Predictor,
)
from vllm.model_executor.models.interfaces import LocalArgmaxMixin
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_next import (
    QwenNextMixtureOfExperts,
    _all_gather_hidden_and_residual,
    _is_shared_expert_fse_compatible,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)

logger = init_logger(__name__)


def _mtp_fc_padded_output_size(
    hidden_size: int, tp_size: int, local_alignment: int = 64
) -> int:
    """Pad the MTP projection so every TP rank owns an aligned output shard."""
    global_alignment = tp_size * local_alignment
    return (
        (hidden_size + global_alignment - 1) // global_alignment
    ) * global_alignment


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = model_config.hf_text_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.natural_tp2 = os.environ.get("AG2_VLLM_MTP_NATURAL_TP2", "0") == "1"
        self.natural_tp2_predictor: FoldedMTPNaturalTP2Predictor | None = None
        if self.natural_tp2:
            if not (get_pp_group().is_first_rank and get_pp_group().is_last_rank):
                raise NotImplementedError("natural TP2 MTP currently requires PP1")
            self.natural_tp2_predictor = FoldedMTPNaturalTP2Predictor(
                vllm_config=vllm_config,
                prefix=prefix,
            )
            self.fc_padded_output_size = self.config.hidden_size
            self.layers = torch.nn.ModuleList()
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(
                    ["hidden_states", "residual"], config.hidden_size
                )
            )
            logger.warning_once(
                "Enabled experimental natural-TP2 MTP trunk on ranks 0/1 "
                "with rank 2 as DCP3 Attention satellite."
            )
            return

        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
        # missing from hf_quant_config.json exclude_modules. Force unquantized.
        # Ref: https://github.com/vllm-project/vllm/pull/38650
        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124
        fc_quant = (
            None
            if (quant_config and quant_config.get_name() == "modelopt_fp4")
            else quant_config
        )
        # The following draft layer needs the full hidden vector. Pad the
        # projection output to aligned TP shards, all-gather it, then discard
        # the zero-padded tail. This avoids replicating the 100 MiB BF16 matrix
        # when hidden_size is not divisible by TP.
        tp_size = get_tensor_model_parallel_world_size()
        self.fc_padded_output_size = _mtp_fc_padded_output_size(
            self.config.hidden_size, tp_size
        )
        if self.fc_padded_output_size != self.config.hidden_size:
            logger.warning_once(
                "Padding Qwen3.5 MTP fc output from %d to %d for "
                "tensor_parallel_size=%d.",
                self.config.hidden_size,
                self.fc_padded_output_size,
                tp_size,
            )
        self.fc = PaddedMergedColumnParallelLinear(
            self.config.hidden_size * 2,
            output_sizes=[self.config.hidden_size],
            padded_output_sizes=[self.fc_padded_output_size],
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=fc_quant,
            prefix=f"{prefix}.fc",
        )
        if envs.AG2_VLLM_MTP_FC_BATCH_INVARIANT:
            logger.warning_once(
                "Using the experimental row-invariant BF16 kernel only for "
                "Qwen3.5 MTP fc; global batch-invariant mode remains disabled."
            )

        # GPTQ: quantized checkpoints may exclude MTP from quantization via
        # quantization_config.dynamic with "-:pattern" entries. When detected,
        # disable quantization for MTP layers so they use unquantized params.
        original_quant = vllm_config.quant_config
        if quant_config and quant_config.get_name() not in ("modelopt_fp4",):
            hf_qc = getattr(model_config.hf_config, "quantization_config", None)
            if isinstance(hf_qc, dict):
                dynamic = hf_qc.get("dynamic", {})
                if any(k.startswith("-:") and "mtp" in k for k in dynamic):
                    vllm_config.quant_config = None
        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
        if envs.AG2_VLLM_MTP_BF16_GATE_UP_SCRATCH:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            shared_scratch = get_nvfp4_marlin_gate_up_scratch()
            if envs.AG2_VLLM_NVFP4_MARLIN_GATE_UP_SCRATCH and shared_scratch is None:
                raise RuntimeError(
                    "Shared target/MTP gate/up workspace was requested but the "
                    "model runner has not installed the NVFP4 Marlin scratch"
                )
            for layer in self.layers:
                layer.mlp.enable_bf16_gate_up_scratch(
                    max_num_tokens,
                    scratch=shared_scratch,
                )
            scratch = self.layers[0].mlp._bf16_gate_up_scratch
            assert scratch is not None
            logger.warning_once(
                "Configured MTP-owned BF16 gate/up scratch: shape=%s bytes=%d "
                "shared_with_nvfp4_marlin=%s",
                tuple(scratch.shape),
                scratch.numel() * scratch.element_size(),
                shared_scratch is not None,
            )
        vllm_config.quant_config = original_quant
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        return_ag2_mtp_trace: bool = False,
    ):
        if self.natural_tp2:
            if return_ag2_mtp_trace:
                raise NotImplementedError("natural TP2 trace is not implemented")
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            assert self.natural_tp2_predictor is not None
            return self.natural_tp2_predictor(
                positions,
                hidden_states,
                inputs_embeds,
            )
        mtp_trace: dict[str, torch.Tensor] = {}
        if return_ag2_mtp_trace:
            mtp_trace["input_ids"] = input_ids
            mtp_trace["positions"] = positions
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            if return_ag2_mtp_trace:
                mtp_trace["embedding"] = inputs_embeds
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            if return_ag2_mtp_trace:
                # Preserve the consumed target-feedback boundary separately
                # from its RMSNorm result. This distinguishes inherited target
                # noise from an MTP normalization/graph-row defect.
                mtp_trace["feedback_input"] = hidden_states
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            if return_ag2_mtp_trace:
                mtp_trace["embedding_norm"] = inputs_embeds
                mtp_trace["feedback_norm"] = hidden_states
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            if envs.AG2_VLLM_MTP_FC_BATCH_INVARIANT:
                output_parallel = linear_batch_invariant(
                    hidden_states,
                    self.fc.weight,
                )
                hidden_states = tensor_model_parallel_all_gather(output_parallel)
            else:
                hidden_states = self.fc(hidden_states)
            hidden_states = hidden_states[..., : self.config.hidden_size]
            if return_ag2_mtp_trace:
                mtp_trace["fc_output"] = hidden_states
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        layer_result = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            return_ag2_mtp_trace=return_ag2_mtp_trace,
        )
        if return_ag2_mtp_trace:
            hidden_states, residual, layer_trace = layer_result
            mtp_trace.update(layer_trace)
        else:
            hidden_states, residual = layer_result

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                positions.shape[-1],
                self.config.hidden_size,
            )

        # The predictor returns only normalized hidden states. Avoid the
        # two-output fused op here: its residual result is dead but otherwise
        # materializes a full BF16 [tokens, hidden] tensor in compiled prefill.
        assert residual is not None
        output = self.norm.forward_native_output_only(hidden_states, residual)
        if return_ag2_mtp_trace:
            mtp_trace["final_norm"] = output
            return output, output, mtp_trace
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.natural_tp2:
            assert self.natural_tp2_predictor is not None
            loaded: set[str] = set()
            passthrough: list[tuple[str, torch.Tensor]] = []
            for name, weight in weights:
                mapped = name.removeprefix("mtp.")
                if self.natural_tp2_predictor.load_weight(mapped, weight):
                    loaded.add(name)
                else:
                    passthrough.append((name, weight))
            self.natural_tp2_predictor.validate_loaded()
            loader = AutoWeightsLoader(self)
            return loaded | loader.load_weights(
                passthrough, mapper=self.hf_to_vllm_mapper
            )
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
            and _is_shared_expert_fse_compatible(
                get_current_vllm_config().quant_config
            ),
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_5MTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._ag2_fp8_head: Fp8DraftHead | None = None

    def ag2_init_fp8_head(self) -> None:
        """Build the FP8 draft-head copy after lm_head sharing, before capture."""
        if shared_fp8_head_enabled():
            if Fp8DraftHead.enabled():
                raise ValueError(
                    "AG2 shared and additive FP8 lm_head modes are mutually exclusive"
                )
            if self.config.tie_word_embeddings:
                raise ValueError(
                    "AG2 shared FP8 lm_head cannot replace tied input embeddings"
                )
            source_bytes, installed_bytes = install_shared_fp8_lm_head(self.lm_head)
            if source_bytes:
                logger.warning(
                    "AG2 shared target/draft FP8 lm_head POC enabled: shard %s, "
                    "%.1f -> %.1f MiB",
                    tuple(self.lm_head.weight.shape),
                    source_bytes / 1024**2,
                    installed_bytes / 1024**2,
                )
            return
        if not Fp8DraftHead.enabled() or self._ag2_fp8_head is not None:
            return
        self._ag2_fp8_head = Fp8DraftHead(self.lm_head.weight)
        logger.info(
            "AG2 FP8 draft lm_head enabled: shard %s, +%.0f MiB",
            tuple(self.lm_head.weight.shape),
            self._ag2_fp8_head.w8.nbytes / 1024**2,
        )

    def ag2_enable_top_token_trace(self, max_num_reqs: int) -> None:
        self.logits_processor.ag2_enable_top_token_trace(
            max_num_reqs,
            get_tensor_model_parallel_world_size(),
            self.lm_head.weight.device,
        )

    def ag2_get_top_token_trace(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.logits_processor.ag2_top_token_gathered_pairs,
            self.logits_processor.ag2_top_token_selected,
        )

    def ag2_enable_mtp_layer_trace(
        self, *, rows: int = 5, history_tokens: int = 640
    ) -> None:
        layer = self.model.layers[0]
        attention = layer.self_attn
        attention.o_proj._ag2_aux_output_parallel_enabled = True
        attention.attn._ag2_aux_dcp_pack_enabled = True
        attention.attn._ag2_aux_dcp_pack_rows = rows
        attention.attn._ag2_aux_dcp_kv_history_enabled = True
        attention.attn._ag2_aux_dcp_kv_history_rows = rows
        attention.attn._ag2_aux_dcp_kv_history_tokens = history_tokens

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        head = self._ag2_fp8_head
        # The FP8 GEMV kernel re-reads the weight shard per row of M, so it
        # only wins for latency-bound small batches; BF16 cuBLAS reads the
        # shard once for the whole batch.
        if head is None or hidden_states.shape[0] > 2:
            return super().get_top_tokens(hidden_states)
        lp = self.logits_processor
        assert lp.scale == 1.0 and getattr(lp, "soft_cap", None) is None
        logits = head.logits(hidden_states)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        local_max_vals, local_max_indices = logits.max(dim=-1)
        global_indices = (
            local_max_indices + self.lm_head.shard_indices.org_vocab_start_index
        )
        tp_size = get_tensor_model_parallel_world_size()
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        if tp_size == 1:
            gathered = local_pair.unsqueeze(1)
            top = global_indices
        else:
            gathered = tensor_model_parallel_all_gather(local_pair, dim=-1).view(
                hidden_states.shape[0], tp_size, 2
            )
            max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
            top = (
                gathered[:, :, 1]
                .gather(dim=-1, index=max_rank_idx)
                .squeeze(-1)
                .to(torch.int64)
            )
        d2t = getattr(self, "draft_id_to_target_id", None)
        if d2t is not None:
            top = top + d2t[top]
        trace_pairs = getattr(
            self.logits_processor, "ag2_top_token_gathered_pairs", None
        )
        trace_selected = getattr(
            self.logits_processor, "ag2_top_token_selected", None
        )
        if trace_pairs is not None and trace_selected is not None:
            rows = hidden_states.shape[0]
            trace_pairs[:rows].copy_(gathered)
            trace_selected[:rows].copy_(top)
        return top

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        return_ag2_mtp_trace = bool(kwargs.pop("return_ag2_mtp_trace", False))
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            return_ag2_mtp_trace=return_ag2_mtp_trace,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model.natural_tp2:
            predictor = self.model.natural_tp2_predictor
            assert predictor is not None
            loaded: set[str] = set()
            passthrough: list[tuple[str, torch.Tensor]] = []
            for name, weight in weights:
                if name.startswith("mtp.") and predictor.load_weight(
                    name.removeprefix("mtp."), weight
                ):
                    loaded.add(name)
                    continue
                if "embed_tokens" in name:
                    passthrough.append(
                        (name.replace("language_model.", ""), weight)
                    )
                elif "lm_head" in name:
                    passthrough.append((name, weight))
            predictor.validate_loaded()
            loader = AutoWeightsLoader(self)
            return loaded | loader.load_weights(passthrough)

        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(remap_weight_names(weights))


class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.set_moe_parameters()
