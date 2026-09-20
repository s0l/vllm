# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass, replace
from typing import Literal

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import MambaSpec


def _gdn_decode_cudagraph_capacity(
    max_num_seqs: int,
    num_spec: int,
    max_capture_size: int | None,
    owner_full_query_len: bool,
) -> int:
    """Return token capacity for the admitted FULL decode family.

    The generic capture ceiling is token-shaped.  The owner path declares one
    additional FULL boundary at ``max_num_seqs * (num_spec + 1)``; accepting
    that descriptor requires the GDN metadata buffers to use the same bound.
    This only grows capacity and does not create any additional graph shape.
    """
    full_decode_tokens = max_num_seqs * (num_spec + 1)
    if owner_full_query_len or max_capture_size is None:
        return full_decode_tokens
    return min(full_decode_tokens, max_capture_size)


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True

    @classmethod
    def get_state_update_chunk_alignment(cls) -> int:
        # The FLA/CuTeDSL GDN prefill scan groups recurrence updates in
        # 64-token chunks. A resumed prefill is numerically identical to the
        # uninterrupted scan only when its artificial split is on this grid.
        return 64


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_batch_indices: torch.Tensor | None = None  # compact spec row -> batch row
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    kv_cache_spec: MambaSpec
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    # Hybrid models commonly place every GDN layer in its own KV-cache group.
    # Decode geometry is identical across those groups; only the state block
    # table changes.  Let the runner build the expensive common metadata once
    # and rebind the state indices for each remaining group.
    supports_update_block_table = True

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        additional_config = vllm_config.additional_config
        self.use_mtp_replay_commit = bool(
            additional_config.get("gdn_mtp_replay_commit", False)
            if isinstance(additional_config, dict)
            else False
        )
        if self.use_mtp_replay_commit:
            assert self.use_spec_decode
            assert self.kv_cache_spec.separate_pool
            assert self.kv_cache_spec.num_speculative_blocks == 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs = _gdn_decode_cudagraph_capacity(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.num_spec,
            self.compilation_config.max_cudagraph_capture_size,
            envs.AG2_VLLM_TP3_OWNER_PREQUANT,
        )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_batch_indices: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            assert prefill_query_start_loc is not None
            assert prefill_query_start_loc_cpu is not None
            total_tokens = int(prefill_query_start_loc_cpu[-1].item())
            return prepare_metadata_cutedsl(
                prefill_query_start_loc,
                total_tokens,
                FLA_CHUNK_SIZE,
            )

        # Only prefill batches use FLA chunk ops.
        # Pre-compute on CPU and async-copy to GPU to avoid
        # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        assert prefill_query_start_loc_cpu is not None
        return (
            async_tensor_h2d(
                prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
            async_tensor_h2d(
                prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                device=device,
            ),
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        if self.use_mtp_replay_commit:
            # The physical topology has one committed state per request. Keep
            # the verifier's existing [request, token] metadata contract by
            # presenting repeated aliases; its recurrent path is no-store.
            block_table_tensor = block_table_tensor[:, :1].expand(-1, self.num_spec + 1)

        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_batch_indices: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if (
                num_spec_decodes == 0
                or num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item()
                == 0
            ):
                num_spec_decodes = 0
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
                )
                spec_batch_indices = async_tensor_h2d(
                    torch.nonzero(spec_sequence_masks_cpu, as_tuple=False)
                    .flatten()
                    .to(torch.int32),
                    device=query_start_loc.device,
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[non_spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    non_spec_sequence_masks_cpu, 0
                ]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[non_spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            chunk_indices, chunk_offsets = self._build_chunk_metadata(
                prefill_query_start_loc,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )

        if num_prefills > 0:
            context_lens_tensor = m.compute_num_computed_tokens()
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert spec_batch_indices is not None
            self.spec_batch_indices[:num_spec_decodes].copy_(
                spec_batch_indices, non_blocking=True
            )
            spec_batch_indices = self.spec_batch_indices[:batch_size]
            spec_batch_indices[num_spec_decodes:].fill_(0)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_batch_indices=spec_batch_indices,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu).sub_(1)
        assert num_decode_draft_tokens_cpu.shape == num_accepted_tokens.shape

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)

    def can_update_block_table(self, metadata: GDNAttentionMetadata) -> bool:
        """Whether ``update_block_table`` covers this metadata shape.

        The first bounded optimization is deliberately decode-only.  Prefill
        and mixed batches keep the existing full builder until their chunk and
        convolution metadata have a separate equivalence proof.
        """
        return (
            self.kv_cache_spec.separate_pool
            and metadata.num_prefills == 0
            and (metadata.num_decodes == 0 or metadata.num_spec_decodes == 0)
        )

    def update_block_table(
        self,
        metadata: GDNAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> GDNAttentionMetadata:
        """Rebind decode state indices while sharing group-invariant metadata."""
        del slot_mapping
        if not self.can_update_block_table(metadata):
            raise ValueError("GDN block-table reuse only supports pure decode batches")

        block_table_tensor = mamba_get_block_table_tensor(
            blk_table,
            # Align/separate-pool decode selects physical state columns from the
            # block table.  ``seq_lens`` is unused for separate-pool mode; the
            # cached metadata therefore need not retain another step tensor.
            torch.empty(0, dtype=torch.int32, device=blk_table.device),
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        if self.use_mtp_replay_commit:
            block_table_tensor = block_table_tensor[:, :1].expand(-1, self.num_spec + 1)

        if metadata.num_spec_decodes:
            assert metadata.spec_batch_indices is not None
            rows = metadata.spec_batch_indices[: metadata.num_spec_decodes].long()
            state_indices = block_table_tensor.index_select(0, rows)[
                :, : self.num_spec + 1
            ]
            if self.use_full_cuda_graph:
                assert metadata.spec_sequence_masks is not None
                self.spec_state_indices_tensor[: metadata.num_spec_decodes].copy_(
                    state_indices, non_blocking=True
                )
                state_indices = self.spec_state_indices_tensor[
                    : metadata.spec_sequence_masks.shape[0]
                ]
                state_indices[metadata.num_spec_decodes :].fill_(NULL_BLOCK_ID)
            return replace(metadata, spec_state_indices_tensor=state_indices)

        state_indices = block_table_tensor[: metadata.num_decodes, 0]
        if self.use_full_cuda_graph:
            assert metadata.non_spec_query_start_loc is not None
            self.non_spec_state_indices_tensor[: metadata.num_decodes].copy_(
                state_indices, non_blocking=True
            )
            state_indices = self.non_spec_state_indices_tensor[
                : metadata.non_spec_query_start_loc.shape[0] - 1
            ]
            state_indices[metadata.num_decodes :].fill_(NULL_BLOCK_ID)
        return replace(metadata, non_spec_state_indices_tensor=state_indices)
