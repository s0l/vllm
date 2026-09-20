# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import _finalize_separate_gdn_pool_specs
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def make_ple_spec_owner(tp, k, *, replay_commit=True, separate_pool=True):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=64,
            mamba_page_size_padded=None,
            mamba_cache_mode="align",
            mamba_cache_dtype="auto",
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=SimpleNamespace(num_speculative_tokens=k) if k else None,
        additional_config=dict(
            gdn_mtp_replay_commit=replay_commit, gdn_separate_pool=separate_pool
        ),
    )
    # Only the spec methods are under test; no checkpoint/giant embedding or
    # model constructor is needed to validate scheduling and cache geometry.
    owner = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(owner)
    owner.model_config = config.model_config
    owner.cache_config = config.cache_config
    owner.hc_hidden_size = 10240
    owner.conv_state_len = 9
    owner.num_spec_tokens = k
    return config, owner


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_ple_replay_commit_has_one_extended_replicated_state(tp, k):
    config, owner = make_ple_spec_owner(tp, k)
    spec = owner.get_kv_cache_spec(config)
    assert spec.mamba_type == MambaAttentionBackendEnum.SHORT_CONV
    assert spec.tp_replicated and spec.num_speculative_blocks == 0
    assert spec.dtypes == (torch.bfloat16,)
    assert sorted(spec.shapes[0]) == [9 + k, 10240]
    assert spec.page_size_bytes == (9 + k) * 10240 * 2

    # Use the real separate-pool finalization and model-state scheduling
    # compatibility check; different state shapes/types must remain legal.
    gdn = replace(
        spec,
        shapes=((6, 32), (2, 8, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        tp_replicated=False,
    )
    groups = _finalize_separate_gdn_pool_specs(
        config,
        [KVCacheGroupSpec(["gdn"], gdn), KVCacheGroupSpec(["ple"], spec)],
    )
    cache = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups)
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state._mamba_spec = None
    ids, representative = state._get_mamba_group_info(cache)
    assert ids == [0, 1]
    assert representative.separate_pool
    assert representative.num_speculative_blocks == 0


def test_ple_keeps_ordinary_spec_policy_without_replay_commit():
    config, owner = make_ple_spec_owner(2, 3, replay_commit=False)
    assert owner.get_kv_cache_spec(config).num_speculative_blocks == 3


def test_ple_replay_commit_requires_separate_pool():
    config, owner = make_ple_spec_owner(2, 3, separate_pool=False)
    with pytest.raises(ValueError, match="requires gdn_separate_pool"):
        owner.get_kv_cache_spec(config)


def test_undeclared_conv_owner_cannot_enter_one_state_policy():
    config, owner = make_ple_spec_owner(2, 3)
    assert not MambaBase.supports_conv_only_spec_commit
    owner.supports_conv_only_spec_commit = False
    with pytest.raises(ValueError, match="explicitly supported conv-only"):
        owner.get_kv_cache_spec(config)
    owner.supports_conv_only_spec_commit = True
    assert owner.get_kv_cache_spec(config).num_speculative_blocks == 0


def test_conv_capability_cannot_hide_a_temporal_state(monkeypatch):
    config, owner = make_ple_spec_owner(2, 3)
    monkeypatch.setattr(
        owner, "get_state_dtype", lambda: (torch.bfloat16, torch.float32)
    )
    with pytest.raises(ValueError, match="explicitly supported conv-only"):
        owner.get_kv_cache_spec(config)
