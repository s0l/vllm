# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static FlashNext GDN allocation agrees with its actual padded consumer."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM


@pytest.mark.parametrize("tp", [1, 2, 3, 4])
@pytest.mark.parametrize("num_spec", [0, 3])
def test_flashnext_gdn_static_pages_match_non_interleaved_instance(tp, num_spec):
    text = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec),
    )
    padded_k = (16 + tp - 1) // tp
    consumer = SimpleNamespace(
        gdn_explicit_partition=bool(16 % tp or 48 % tp),
        padded_local_conv_dim=padded_k * 5 * 128,
        padded_local_num_v_heads=padded_k * 3,
        conv_kernel_size=4,
        num_spec=num_spec,
        head_v_dim=128,
        head_k_dim=128,
        tp_size=tp,
        num_k_heads=16,
        num_v_heads=48,
    )
    actual = Qwen4ExpForCausalLM.get_gdn_mamba_state_shape_from_config(cfg)
    assert actual == QwenGatedDeltaNetAttention.get_state_shape(consumer)
    # Only the two dtype-owned states are physically consumed by MambaBase.
    assert torch.tensor(actual[0]).prod() == padded_k * 640 * (3 + num_spec)
    assert actual[1] == (padded_k * 3, 128, 128)
