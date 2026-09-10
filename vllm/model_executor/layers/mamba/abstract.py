# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Iterable
from math import prod

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class MambaBase(AttentionLayerBase):
    """
    Base class for Mamba-like layers which support the v1 engine.
    Inherit from this class if you implement a custom layer.
    """

    # Contains the KV cache (mamba state) for the layer
    # in the shape specified by `self.get_state_shape`.
    kv_cache: tuple[torch.Tensor, ...]
    supports_dcp: bool = False
    # Opt in only when speculative candidates live in one extended conv
    # window, with no temporal state. Separate-pool postprocess shifts that
    # window and resets acceptance before the next forward.
    supports_conv_only_spec_commit: bool = False

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Unpack a raw ``[B, 1, 1, C]`` int8 page view into per-state views.

        Each block's ``C`` bytes hold the layer's states (e.g. conv, ssm)
        packed contiguously; slice them out and reinterpret per dtype/shape.
        """
        pages = kv_cache.squeeze(dim=(1, 2))
        states: list[torch.Tensor] = []
        offset = 0
        for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype()):
            nbytes = prod(shape) * get_dtype_size(dtype)
            state = pages[:, offset : offset + nbytes].view(dtype)
            states.append(state.view(-1, *shape))
            offset += nbytes
        self.kv_cache = tuple(states)

    @abstractmethod
    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        """
        Defines the shape of the state.
        For mamba layers this is usually a (conv_state, ssm_state) tuple.
        In this case, returns (conv_state_shape, ssm_state_shape).
        """
        pass

    @property
    @abstractmethod
    def mamba_type(self) -> MambaAttentionBackendEnum:
        pass

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return False

    @abstractmethod
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        pass

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        assert mamba_block_size is not None
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        additional = (
            vllm_config.additional_config
            if isinstance(vllm_config.additional_config, dict)
            else {}
        )
        replay_commit = bool(additional.get("gdn_mtp_replay_commit", False))
        num_speculative_blocks = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        if replay_commit:
            conv_only = (
                self.mamba_type == MambaAttentionBackendEnum.SHORT_CONV
                and self.supports_conv_only_spec_commit
                and len(self.get_state_dtype()) == 1
            )
            if self.mamba_type != MambaAttentionBackendEnum.GDN_ATTN and not conv_only:
                raise ValueError(
                    "gdn_mtp_replay_commit requires GDN or an explicitly "
                    "supported conv-only speculative state"
                )
            if not additional.get("gdn_separate_pool", False):
                raise ValueError("gdn_mtp_replay_commit requires gdn_separate_pool")
            # The verifier journals compact recurrence inputs and commits the
            # accepted prefix into the one live state after sampling.
            # Conv-only owners use the same postprocess window shift, without
            # a recurrence journal or extra speculative blocks.
            num_speculative_blocks = 0
        return MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            tp_replicated=self.is_kv_cache_tp_replicated,
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            num_speculative_blocks=num_speculative_blocks,
            state_update_chunk_alignment=(
                self.get_attn_backend().get_state_update_chunk_alignment()
            ),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)
