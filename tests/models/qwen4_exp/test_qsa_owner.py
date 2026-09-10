# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner admission and real cache descriptors before the CUDA consumer gate."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig
from vllm.models.qwen4_exp.common.qsa_cache import (
    QSACompressedKeyCache,
    QSAKeyStateCache,
    QSAStateBackend,
)
from vllm.models.qwen4_exp.nvidia.qsa_flashinfer import (
    QSAFlashInferBackend,
    QSAFlashInferMetadataBuilder,
    SparseQSAPool,
    page_one_view,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_cache_layout import KVCacheLayout


def config():
    cfg = VllmConfig(device_config=DeviceConfig("cpu"))
    cfg.cache_config.block_size = 64
    cfg.cache_config.kv_cache_layout = "LBNHC"
    cfg.scheduler_config.max_num_batched_tokens = 128
    cfg.scheduler_config.max_num_seqs = 4
    cfg.scheduler_config.disable_hybrid_kv_cache_manager = False
    cfg.parallel_config.tensor_parallel_size = 3
    cfg.parallel_config.decode_context_parallel_size = 3
    cfg.speculative_config = SimpleNamespace(
        num_speculative_tokens=3, parallel_drafting=False
    )
    cfg.model_config = SimpleNamespace(max_model_len=8192, dtype=torch.bfloat16)
    cfg.additional_config = {"gdn_separate_pool": True, "gdn_pool_blocks": 4}
    cfg.additional_config["flashnext_qsa_dcp"] = True
    return cfg


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 6, 8])
def test_engine_config_admits_qsa_sequence_ownership_without_relaxing_gqa(tp):
    from vllm.config import ModelConfig

    cfg = config()
    cfg.parallel_config.tensor_parallel_size = tp
    cfg.parallel_config.decode_context_parallel_size = tp
    cfg.cache_config.cache_dtype = "fp8"
    cfg.model_config.hf_text_config = SimpleNamespace(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    cfg.model_config.model_arch_config = SimpleNamespace(
        model_type="qwen4_exp_text", total_num_attention_heads=24
    )
    cfg.model_config.get_total_num_kv_heads = lambda: 2
    cfg.model_config.use_mla = False
    cfg.model_config.multimodal_config = None
    assert cfg._uses_sequence_sharded_qsa()
    ModelConfig.verify_with_parallel_config(
        cfg.model_config, cfg.parallel_config, sequence_sharded_kv=True
    )
    if tp > 1:
        with pytest.raises(ValueError, match="parallel"):
            ModelConfig.verify_with_parallel_config(
                cfg.model_config, cfg.parallel_config
            )
    cfg.additional_config.clear()
    assert not cfg._uses_sequence_sharded_qsa()
    cfg.additional_config["flashnext_qsa_dcp"] = True
    cfg.model_config.hf_text_config.model_type = "other_model"
    with pytest.raises(ValueError, match="FlashNext FP8 QSA"):
        cfg._uses_sequence_sharded_qsa()
    cfg.model_config.hf_text_config.model_type = "qwen4_exp_mtp"
    assert cfg._uses_sequence_sharded_qsa()


@pytest.mark.parametrize("invalid", ["dcp", "kv", "heads", "budget", "flag", "dbo"])
def test_engine_qsa_sequence_ownership_rejects_invalid_profile(invalid):
    cfg = config()
    cfg.cache_config.cache_dtype = "fp8"
    text = SimpleNamespace(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    cfg.model_config.hf_text_config = text
    if invalid == "dcp":
        cfg.parallel_config.decode_context_parallel_size = 1
    elif invalid == "kv":
        cfg.cache_config.cache_dtype = "bfloat16"
    elif invalid == "heads":
        text.num_key_value_heads = 4
    elif invalid == "budget":
        text.indexer_budget = 1024
    elif invalid == "flag":
        cfg.additional_config["flashnext_qsa_dcp"] = "true"
    elif invalid == "dbo":
        cfg.parallel_config.enable_dbo = True
    with pytest.raises(ValueError, match="FlashNext FP8 QSA"):
        cfg._uses_sequence_sharded_qsa()


def test_actual_separate_allocator_preserves_qsa_page_one_alias_and_side_state():
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )
    from vllm.v1.worker.gpu.attn_utils import _allocate_kv_cache

    cfg = config()
    raw = QSAKeyStateCache(
        head_size=128,
        dtype=torch.bfloat16,
        cache_config=cfg.cache_config,
        vllm_config=cfg,
        prefix="raw",
        compress_ratio=4,
        cache_rope_positions=True,
    )
    compressed = QSACompressedKeyCache(
        head_size=128,
        dtype=torch.bfloat16,
        cache_config=cfg.cache_config,
        vllm_config=cfg,
        prefix="compressed",
        compress_ratio=4,
    )
    main = FullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.float8_e4m3fn,
    )
    specs = {
        "main": main,
        "main2": main,
        "raw": raw.get_kv_cache_spec(cfg),
        "compressed": compressed.get_kv_cache_spec(cfg),
        "gdn": MambaSpec(
            block_size=64,
            shapes=((6, 3840), (18, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
        ),
    }
    # Preserve two repeated motifs, as in the target model. Different cache
    # groups may intentionally alias physical slots with disjoint block IDs.
    for name in ("raw", "compressed", "gdn"):
        specs[name + "2"] = specs[name]
    groups = get_kv_cache_groups(cfg, specs)
    allocation = get_kv_cache_config_from_groups(cfg, groups, 16 << 20)
    views = _allocate_kv_cache(
        allocation,
        {},
        torch.device("cpu"),
        KVCacheLayout.LBNHC,
        [g.kv_cache_spec.block_size for g in groups],
    )
    for name in ("main", "main2"):
        k, v = page_one_view(views[name], 256)
        assert k.data_ptr() == views[name].data_ptr()
        assert v.data_ptr() - k.data_ptr() == 256
        assert k.stride() == (1024, 1024, 512, 1)
    # Independent dense owners must not alias each other or the side states.
    addresses = [v.untyped_storage().data_ptr() for v in views.values()]
    assert len(set(addresses)) == len(addresses)
    raw.bind_kv_cache(views["raw"])
    compressed.bind_kv_cache(views["compressed"])
    raw.key_cache[1, 3, 0].fill_(2)
    raw.rope_position_cache[1, 3, 0].copy_(torch.tensor([9, 4, 2]))
    assert torch.equal(views["raw"][1, 0, 3, :128], torch.full((128,), 2))
    assert raw.rope_position_cache[1, 3, 0].tolist() == [9, 4, 2]
    assert views["main"].view(torch.uint8).count_nonzero() == 0
    assert views["compressed"].count_nonzero() == 0
    assert raw.get_attn_backend().supported_kv_cache_layouts() == (KVCacheLayout.LBNHC,)
    assert KVCacheLayout.LBNHC not in QSAStateBackend.supported_kv_cache_layouts()
    assert QSAFlashInferBackend.supported_kv_cache_layouts() == (KVCacheLayout.LBNHC,)
    with pytest.raises(ValueError, match="layer-compact"):
        page_one_view(views["main"].contiguous(), 256)


def test_qsa_metadata_keeps_global_lengths_and_rank_local_slots():
    from vllm.v1.attention.backend import CommonAttentionMetadata

    cfg = config()
    spec = FullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.float8_e4m3fn,
    )
    builder = QSAFlashInferMetadataBuilder(spec, ["main"], cfg, torch.device("cpu"))
    assert builder.reorder_batch_threshold == 4
    starts = torch.tensor([0, 4, 8], dtype=torch.int32)
    table = torch.tensor([[3, 9], [2, 1]], dtype=torch.int32)
    slots = torch.tensor([-1, 64, -1, -1, 192, -1, -1, 193])
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor([4099, 9]),
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=4,
        max_seq_len=4099,
        block_table_tensor=table,
        slot_mapping=slots,
    )
    md = builder.build_for_cudagraph_capture(common)
    assert md.num_actual_tokens == 8 and md.max_query_len == 4
    assert md.block_table is table and md.slot_mapping is slots


def test_qsa_shared_pool_follows_static_context_without_allocating_in_constructor():
    cfg = config()
    first = SparseQSAPool.for_config(cfg)
    draft = SimpleNamespace(compilation_config=cfg.compilation_config)
    assert SparseQSAPool.for_config(draft) is first
    assert not first.plans and first.workspace is None
    assert SparseQSAPool.for_config(config()) is not first


@pytest.mark.parametrize("invalid", ["dcp", "kv_dtype", "dbo", "heads", "budget"])
def test_qsa_owner_rejects_unproven_geometry_before_weight_allocation(
    monkeypatch, invalid
):
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.qsa.get_tensor_model_parallel_world_size",
        lambda: 3,
    )
    cfg = config()
    cfg.additional_config["flashnext_qsa_dcp"] = True
    cfg.cache_config.cache_dtype = "fp8_e4m3"
    text = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )
    if invalid == "dcp":
        cfg.parallel_config.decode_context_parallel_size = 1
    elif invalid == "kv_dtype":
        cfg.cache_config.cache_dtype = "bfloat16"
    elif invalid == "dbo":
        cfg.parallel_config.enable_dbo = True
    elif invalid == "heads":
        text.num_key_value_heads = 8
    elif invalid == "budget":
        text.indexer_budget = 1024
    with pytest.raises(NotImplementedError, match="requires"):
        Qwen4ExpQSAAttention(vllm_config=cfg, config=text, layer_id=3)
    assert not cfg.compilation_config.static_forward_context


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 6, 8])
def test_actual_owner_exposes_fp8_spec_before_allocator_binding(monkeypatch, tp):
    from vllm.config import set_current_vllm_config
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention

    for module in (
        "vllm.models.qwen4_exp.nvidia.qsa",
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            module + ".get_tensor_model_parallel_world_size", lambda: tp
        )
        monkeypatch.setattr(module + ".get_tensor_model_parallel_rank", lambda: 0)
    cfg = config()
    cfg.parallel_config.tensor_parallel_size = tp
    cfg.parallel_config.decode_context_parallel_size = tp
    cfg.additional_config["flashnext_qsa_dcp"] = True
    cfg.cache_config.cache_dtype = "fp8_e4m3"
    cfg.model_config.multimodal_config = None
    cfg.model_config.uses_mrope = True
    text = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    with set_current_vllm_config(cfg):
        owner = Qwen4ExpQSAAttention(
            vllm_config=cfg, config=text, layer_id=3, prefix="model.layers.3.self_attn"
        )
    spec = owner.get_kv_cache_spec(cfg)
    assert owner.num_heads == owner.impl.num_heads == 24 // tp
    assert owner.q_size == 24 // tp * 256
    assert spec.dtype == torch.float8_e4m3fn
    assert spec.num_kv_heads == 2 and spec.page_size_bytes == 65536
    good = (
        torch.zeros(2, 64, 2, 512, dtype=torch.uint8).view(spec.dtype).transpose(1, 2)
    )
    owner.bind_kv_cache(good)
    for invalid in (good.contiguous(), good.view(torch.uint8), good[:, :1]):
        with pytest.raises(ValueError):
            owner.bind_kv_cache(invalid)
        assert owner.kv_cache is good
    assert not owner.impl.pool.plans
    for option in ("return_ag2_mtp_trace", "return_tp_partial"):
        with pytest.raises(ValueError, match="trace carriers"):
            owner.forward(None, None, **{option: True})
