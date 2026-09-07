# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in target-only row continuation, configured after weight loading."""

import os

import torch
from torch import nn

from vllm.distributed.device_communicators import tp3_row_ops  # noqa: F401
from vllm.distributed.device_communicators.tp3_row_profile import (
    read_peer_selections,
    read_profile,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferB12xNvFp4LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    GDNQuantizedActivations,
    QuantizedActivation,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic


def quantized(q, sf, rows):
    return QuantizedActivation(
        q, sf, torch.bfloat16, torch.Size((rows, 5120)), kNvfp4Dynamic
    )


class Boundary(nn.Module):
    def __init__(self, config, selections, device):
        super().__init__()
        self.config = list(config)
        for rank, value in enumerate(selections):
            self.register_buffer(f"selected_{rank}", value.to(device), persistent=False)

    def selections(self):
        return [self.selected_0, self.selected_1, self.selected_2]


class RowContinuation(nn.Module):
    def __init__(self, profile, peers, layers, device, profile_sha):
        super().__init__()
        self.profile_sha = profile_sha
        self.first_config = profile["norms"]["0.input"]
        self.terminal_config = profile["norms"]["terminal"]
        self.group_name = get_tp_group().unique_name
        self.post = nn.ModuleList()
        self.inputs = nn.ModuleList()
        for i, layer in enumerate(layers):
            suffix = layer.mlp.gate_up_proj._ag2_nvfp4_arc_runtime_suffix
            self.post.append(
                Boundary(
                    profile["norms"][f"{i}.post"], [p[suffix] for p in peers], device
                )
            )
            if i:
                projection = (
                    layer.linear_attn.in_proj_qkvz
                    if layer.layer_type == "linear_attention"
                    else layer.self_attn.qkv_proj
                )
                suffix = projection._ag2_nvfp4_arc_runtime_suffix
                self.inputs.append(
                    Boundary(
                        profile["norms"][f"{i}.input"],
                        [p[suffix] for p in peers],
                        device,
                    )
                )


def prepare_row_continuation(model, model_root):
    expected_sha = os.environ.get("AG2_VLLM_TP3_ROW_PROFILE_SHA256", "")
    path = os.environ.get("AG2_VLLM_TP3_ROW_PROFILE", "")
    if not expected_sha and not path:
        return
    profile = read_profile(path, expected_sha, model_root)
    old = getattr(model, "_ag2_row_continuation", None)
    if old is not None:
        if old.profile_sha != expected_sha:
            raise ValueError("cannot rebind a compiled row model to another profile")
        return
    group = get_tp_group()
    if (
        group.world_size != 3
        or get_pp_group().world_size != 1
        or model.start_layer != 0
        or model.end_layer != 64
        or len(model.layers) != 64
        or model.aux_hidden_state_layers
        or model.use_sequence_parallel
        or model._ag2_tp3_owner_prequant
    ):
        raise ValueError("row continuation requires untraced dense TP3/PP1 full target")
    peers, identities = read_peer_selections(
        os.environ["AG2_VLLM_NVFP4_ARC_SIDECAR_DIR"]
    )
    if identities != profile["sidecar_sha256"]:
        raise ValueError("row profile ARC sidecar identity mismatch")
    consumers = []
    for i, layer in enumerate(model.layers):
        if layer.layer_type not in ("linear_attention", "full_attention"):
            raise ValueError("unsupported row attention type")
        if (
            layer.layer_scale
            or layer.use_attn_reduce_scatter_for_moe
            or layer._ag2_tp3_mtp_block5
            or layer._ag2_tp3_owner_prequant
            or layer.mlp.expert_gate is not None
            or not layer.mlp._ag2_tp3_unified_exact_reduce
            or layer.mlp.down_proj.reduce_results
        ):
            raise ValueError(f"unsupported row layer/reducer contract at layer{i}")
        attention = (
            layer.linear_attn
            if layer.layer_type == "linear_attention"
            else layer.self_attn
        )
        for module in (model, layer, layer.mlp, attention):
            if any(
                (name.startswith("_ag2_aux") or name == "_ag2_layer0_trace_enabled")
                and isinstance(value, bool)
                and value
                for name, value in vars(module).items()
            ):
                raise ValueError("row continuation cannot expose noncanonical traces")
        attention = (
            layer.linear_attn
            if layer.layer_type == "linear_attention"
            else layer.self_attn
        )
        projection = (
            attention.in_proj_qkvz
            if layer.layer_type == "linear_attention"
            else attention.qkv_proj
        )
        output = (
            attention.out_proj
            if layer.layer_type == "linear_attention"
            else attention.o_proj
        )
        if not attention._ag2_tp3_unified_exact_reduce or output.reduce_results:
            raise ValueError("row attention must return a rank-local partial")
        selected_consumers = [layer.mlp.gate_up_proj] + ([projection] if i else [])
        for consumer in selected_consumers:
            kernel = consumer.scheme.kernel
            suffix = getattr(consumer, "_ag2_nvfp4_arc_runtime_suffix", None)
            if (
                not isinstance(kernel, FlashInferB12xNvFp4LinearKernel)
                or any(suffix not in peer for peer in peers)
                or consumer._ag2_nvfp4_arc_sidecar_sha256
                != identities[group.rank_in_group]
                or not torch.equal(
                    consumer._ag2_nvfp4_arc_selected.cpu(),
                    peers[group.rank_in_group][suffix],
                )
                or consumer.input_global_scale_inv.numel() != 1
                or consumer.input_global_scale_inv.dtype != torch.float32
            ):
                raise ValueError("row consumer ARC/B12x ABI mismatch")
            consumers.append(consumer)
        if layer.layer_type == "linear_attention" and i:
            ba = attention.in_proj_ba
            if not isinstance(
                ba.scheme.kernel, FlashInferB12xNvFp4LinearKernel
            ) or not torch.equal(
                ba.input_global_scale_inv, projection.input_global_scale_inv
            ):
                raise ValueError("row GDN base/ARC multiplier mismatch")
            consumers.append(ba)
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            if (
                norm.variance_epsilon != 1e-6
                or norm.weight.shape != (5120,)
                or norm.weight.dtype != torch.bfloat16
                or not norm.weight.is_contiguous()
                or norm.weight.device != model.norm.weight.device
            ):
                raise ValueError("row norm checkpoint ABI mismatch")
    if (
        model.norm.variance_epsilon != 1e-6
        or model.norm.weight.dtype != torch.bfloat16
        or model.norm.weight.shape != (5120,)
        or not model.norm.weight.is_contiguous()
    ):
        raise ValueError("row terminal norm ABI mismatch")
    binding = RowContinuation(
        profile, peers, model.layers, model.norm.weight.device, expected_sha
    )
    # Startup failure aborts the model instance. No live route is enabled until
    # all allocations and consumers have been admitted.
    for consumer in consumers:
        consumer.scheme.kernel.bind_row_prequant_input(consumer)
    model._ag2_row_continuation = binding


def row_model_forward(
    model, binding, input_ids, positions, intermediate_tensors, inputs_embeds
):
    if intermediate_tensors is not None or model.aux_hidden_state_layers:
        raise ValueError("row target does not admit PP or auxiliary hidden consumers")
    residual = (
        inputs_embeds if inputs_embeds is not None else model.embed_input_ids(input_ids)
    )
    if residual.shape[0] != positions.shape[-1]:
        raise ValueError("row target requires full physical invocation rows")
    hidden = torch.ops.vllm.tp3_row_first(
        residual,
        model.layers[0].input_layernorm.weight,
        binding.first_config,
        binding.group_name,
    )
    contribution = attention = None
    for i, layer in enumerate(model.layers):
        if i:
            assert contribution is not None and attention is not None
            boundary = binding.inputs[i - 1]
            linear = layer.layer_type == "linear_attention"
            projection = (
                layer.linear_attn.in_proj_qkvz if linear else layer.self_attn.qkv_proj
            )
            q, sf, bq, bsf, residual = torch.ops.vllm.tp3_row_next(
                contribution,
                attention,
                residual,
                layer.input_layernorm.weight,
                projection.input_global_scale_inv,
                boundary.selections(),
                boundary.config,
                linear,
                binding.group_name,
            )
            hidden = quantized(q, sf, residual.shape[0])
            if linear:
                hidden = GDNQuantizedActivations(
                    hidden, quantized(bq, bsf, residual.shape[0])
                )
        if layer.layer_type == "linear_attention":
            partial = layer.linear_attn(hidden_states=hidden, return_tp_partial=True)
        else:
            partial = layer.self_attn(
                positions=positions, hidden_states=hidden, return_tp_partial=True
            )
        boundary = binding.post[i]
        q, sf, attention = torch.ops.vllm.tp3_row_post(
            partial,
            residual,
            layer.post_attention_layernorm.weight,
            layer.mlp.gate_up_proj.input_global_scale_inv,
            boundary.selections(),
            boundary.config,
            binding.group_name,
        )
        contribution = layer.mlp(
            quantized(q, sf, residual.shape[0]), return_tp_partial=True
        )
    assert contribution is not None and attention is not None
    return torch.ops.vllm.tp3_row_terminal(
        contribution,
        attention,
        residual,
        model.norm.weight,
        binding.terminal_config,
        binding.group_name,
    )
