# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashNext model adapter for one worker's bounded native expert provider."""

from __future__ import annotations

import weakref

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.elastic_expert import NativeExpertBudget

from .expert_offload_bank import NativeExpertBank
from .expert_offload_provider import NativeExpertProvider
from .expert_offload_source import NativeExpertStore

_providers: weakref.WeakValueDictionary[int, NativeExpertProvider] = (
    weakref.WeakValueDictionary()
)


def native_experts_enabled(vllm_config, prefix):
    extra = vllm_config.additional_config
    return (
        isinstance(extra, dict)
        and "flashnext_native_experts" in extra
        and ".mtp." not in "." + prefix
    )


def get_native_provider(vllm_config):
    budget = NativeExpertBudget.from_config(vllm_config)
    assert budget is not None
    options = vllm_config.additional_config["flashnext_native_experts"]
    fields = {"hot_rows", "max_hot_rows", "staging", "ram_cache_bytes"}
    if not isinstance(options, dict) or options.keys() - fields:
        raise ValueError("invalid FlashNext native expert options")
    parallel = vllm_config.parallel_config
    text = vllm_config.model_config.hf_text_config
    if (
        parallel.tensor_parallel_size != 3
        or parallel.decode_context_parallel_size != 3
        or parallel.prefill_context_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_expert_parallel
        or parallel.enable_eplb
        or parallel.enable_dbo
        or parallel.use_sequence_parallel_moe
        or get_tensor_model_parallel_world_size() != 3
        or (text.hidden_size, text.num_experts, text.num_experts_per_tok)
        != (2560, 512, 10)
        or (text.moe_intermediate_size, text.num_hidden_layers) != (640, 48)
        or vllm_config.kernel_config.moe_backend != "cutlass"
        or vllm_config.model_config.dtype != torch.bfloat16
    ):
        raise ValueError(
            "native FlashNext experts require admitted TP3 CUTLASS geometry"
        )
    key = id(vllm_config.compilation_config.static_forward_context)
    provider = _providers.get(key)
    if provider is None:
        source = NativeExpertStore(
            vllm_config.model_config.model,
            get_tensor_model_parallel_rank(),
            options.get("ram_cache_bytes", 1 << 30),
            layers=48,
            experts=512,
        )
        device = get_tp_group().device
        bank = NativeExpertBank(
            source,
            device,
            hot_rows=0,
            max_hot_rows=budget.max_hot_rows,
            staging=budget.staging,
        )
        if (
            sum(bank.targets(0).values()) != budget.base_bytes
            or tuple(
                bank.strides[name]
                for name in (
                    "w13_weight",
                    "w2_weight",
                    "w13_weight_scale",
                    "w2_weight_scale",
                )
            )
            != budget.strides
        ):
            raise RuntimeError("native bank differs from scheduler byte geometry")
        provider = NativeExpertProvider(bank, get_tp_group(), vllm_config)
        _providers[key] = provider
    return provider


class NativeOffloadedExperts(nn.Module):
    """Account for every source field without materializing resident experts."""

    def __init__(self, provider, vllm_config, layer_id, prefix):
        super().__init__()
        self.provider, self.layer_id = provider, layer_id
        self.layer_name = prefix
        self.seen: set[str] = set()
        self.loaded = False
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError("duplicate native expert owner")
        context[prefix] = self

    def load_weights(self, weights):
        source = self.provider.bank.source
        accepted = set()
        for name, tensor in weights:
            if self.loaded or name in self.seen:
                raise RuntimeError("native expert reload requires a new worker")
            try:
                expert, field = name.split(".", 1)
                _, _, _, shape, dtype = source.entries[self.layer_id, int(expert)][
                    field
                ]
            except (ValueError, KeyError) as exc:
                raise ValueError("unknown native expert source field") from exc
            expected_dtype = {
                "U8": torch.uint8,
                "F8_E4M3": torch.float8_e4m3fn,
                "F32": torch.float32,
            }[dtype]
            if tuple(tensor.shape) != shape or tensor.dtype != expected_dtype:
                raise ValueError("native expert loader/source metadata mismatch")
            self.seen.add(name)
            accepted.add(name)
        return accepted

    def finish_load(self):
        if self.loaded:
            raise RuntimeError("native expert reload requires a new worker")
        source = self.provider.bank.source
        expected = {
            f"{expert}.{field}"
            for expert in range(source.experts)
            for field in source.entries[self.layer_id, expert]
        }
        if self.seen != expected:
            raise ValueError("native expert loader missed source fields")
        source.check_files(source.files)
        self.loaded = True

    def forward(self, hidden, weights, ids):
        if not self.loaded:
            raise RuntimeError("native expert weights are not admitted")
        output = torch.empty_like(hidden)
        torch.ops.vllm.flashnext_native_experts(
            hidden, weights, ids, output, self.layer_name
        )
        return output


class FlashNextNativeMoeBlock(Qwen3NextSparseMoeBlock):
    def __init__(self, vllm_config, prefix="", *, provider=None):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        if provider is None:
            provider = get_native_provider(vllm_config)
        if config.shared_expert_intermediate_size != 640 or config.hidden_act != "silu":
            raise ValueError("unsupported native shared expert geometry")
        self.tp_size, self.ep_size = 3, 1
        self.is_sequence_parallel = False
        self.enable_eplb = self.is_fused_shared_expert_enabled = False
        self.replicate_shared_expert = False
        self.n_routed_experts = self.n_logical_experts = self.n_physical_experts = 512
        self.n_local_physical_experts = 512
        self.n_shared_experts, self.n_redundant_experts = 1, 0
        self.renormalize = bool(getattr(config, "norm_topk_prob", True))
        self.gate = ReplicatedLinear(2560, 512, bias=False, prefix=f"{prefix}.gate")
        self.shared_expert_gate = ReplicatedLinear(
            2560, 1, bias=False, prefix=f"{prefix}.shared_expert_gate"
        )
        self.shared_expert = Qwen3NextMLP(
            2560,
            640,
            "silu",
            reduce_results=False,
            expert_gate=self.shared_expert_gate,
            prefix=f"{prefix}.shared_expert",
        )
        self.experts = NativeOffloadedExperts(
            provider, vllm_config, extract_layer_index(prefix), f"{prefix}.experts"
        )

    def forward(self, hidden_states, already_sequence_parallel=False):
        if already_sequence_parallel:
            raise ValueError("native experts require replicated token inputs")
        if hidden_states.shape[0] == 0:
            return hidden_states
        logits = self.gate(hidden_states)[0]
        weights, ids, _ = fused_topk(
            hidden_states, logits.float(), 10, self.renormalize
        )
        routed = self.experts(hidden_states, weights, ids)
        assert self.shared_expert is not None
        shared = self.shared_expert(hidden_states)
        return tensor_model_parallel_all_reduce(routed + shared)


def reject_native_expert_reload(model):
    if any(isinstance(m, NativeOffloadedExperts) and m.loaded for m in model.modules()):
        raise RuntimeError("native expert reload requires a new worker")


def finish_native_expert_load(model):
    for module in model.modules():
        if isinstance(module, NativeOffloadedExperts):
            module.finish_load()
