# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("language_model_only", "expected"),
    [(False, False), (True, True)],
)
def test_qwen3_next_fused_qk_dispatch_preserves_multimodal_mrope_contract(
    monkeypatch: pytest.MonkeyPatch,
    language_model_only: bool,
    expected: bool,
) -> None:
    from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
    from vllm.model_executor.models import qwen3_next

    class FakeAttention(torch.nn.Module):
        supports_dcp_full_kv_attention_heads = False

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    text_config = SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        num_hidden_layers=64,
        max_position_embeddings=262144,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
        rms_norm_eps=1e-6,
        attn_output_gate=True,
    )
    model_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(language_model_only=language_model_only)
    )
    rotary_emb = object.__new__(MRotaryEmbedding)
    for name, value in (
        ("is_neox_style", True),
        ("dtype", torch.bfloat16),
        ("rotary_dim", 64),
        ("mrope_section", [11, 11, 10]),
        ("mrope_interleaved", True),
    ):
        object.__setattr__(rotary_emb, name, value)

    monkeypatch.setattr(qwen3_next, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_next, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        qwen3_next, "QKVParallelLinear", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        qwen3_next, "RowParallelLinear", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(qwen3_next, "get_rope", lambda *args, **kwargs: rotary_emb)
    monkeypatch.setattr(
        qwen3_next, "Qwen3NextRMSNorm", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(qwen3_next, "Attention", FakeAttention)
    monkeypatch.setattr(qwen3_next.current_platform, "is_cuda", lambda: True)

    attention = qwen3_next.Qwen3NextAttention(
        text_config,
        model_config=model_config,
        prefix="model.layers.3.self_attn",
    )

    assert attention.use_fused_qk_norm_rope_gate is expected


def test_elastic_runtime_identity_covers_qwen3_5_graph_owners():
    from vllm.v1.core import elastic_runtime

    owners = set(elastic_runtime._ELASTIC_RUNTIME_SOURCE_MODULES)
    assert {
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.models.qwen3_next",
        "vllm.model_executor.models.qwen3_5_mtp",
        "vllm.model_executor.layers.fused_qk_norm_rope",
        "vllm.model_executor.layers.rotary_embedding",
        "vllm.v1.worker.gpu.attn_utils",
        "vllm.v1.worker.gpu.spec_decode.speculator",
        "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator",
    } <= owners

    assert {
        "vllm.v1.worker.gpu.attn_utils",
        "vllm.v1.worker.gpu.spec_decode.speculator",
        "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator",
    } <= set(elastic_runtime._ELASTIC_PHYSICAL_CATALOG_SOURCE_MODULES)

    hashes = elastic_runtime.elastic_runtime_source_hashes()
    assert owners == hashes.keys()
    assert all(len(source_hash) == 64 for source_hash in hashes.values())
