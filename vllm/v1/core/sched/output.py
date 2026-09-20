# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
from vllm.multimodal.utils import strip_covered_mm_data
from vllm.v1.core.elastic_expert import ElasticExpertGrant
from vllm.v1.core.elastic_graph import ElasticStepPlan

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    KVCacheBlockCopy = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None
    # DeepSeek-V4.1 only: SWA bounded replay; see Request.replay_start.
    replay_start: int = 0

    # Scheduler-owned boundary of the token stream that must execute as
    # prefill.  For a resumed request this can extend past the user prompt over
    # output tokens whose KV/state is being reconstructed.
    execution_prefill_len: int | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
        uses_mrope: bool = False,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=strip_covered_mm_data(
                request.mm_features,
                request.num_computed_tokens,
                uses_mrope=uses_mrope,
            ),
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
            replay_start=request.replay_start,
            execution_prefill_len=request.execution_prefill_len,
        )

    @property
    def prompt_len(self) -> int:
        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        if self.prompt_embeds is not None:
            return self.prompt_embeds.shape[0]
        return 0

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"execution_prefill_len={self.execution_prefill_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"execution_prefill_len={self.execution_prefill_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # MRV1-only: For requests not scheduled in the last step, propagate the token ids
    # to the connector. Won't contain requests scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # Version of dataclass repr with token IDs obfuscated.
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class ScheduledEncoderInputStats:
    """Stats for encoder inputs scheduled in one iteration."""

    num_inputs: int = 0
    output_tokens: int = 0


@dataclass
class KVConnectorBlockState:
    """Scheduler-local block state offered to a producer-side KV connector."""

    # Requests scheduled this step and requests with a boundary-state hand-off.
    req_ids: set[str]
    # Resolve on access to avoid copying tables the connector never reads.
    resolve_block_ids: Callable[[str], tuple[list[int], ...]]
    # Exact Mamba "align" boundary-state hand-offs.
    boundary_state_offloads: dict[str, list[tuple[int, int, int]]]

    def get_block_ids(self, req_id: str) -> tuple[list[int], ...] | None:
        if req_id not in self.req_ids:
            return None
        return self.resolve_block_ids(req_id)


@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    scheduled_encoder_input_stats: ScheduledEncoderInputStats | None = None

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # Whether any scheduled request consumes KV that the connector loads
    # synchronously during this step (load_async=False).
    has_sync_kv_loads: bool = False

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None
    # EC Cache Manager metadata
    ec_manager_metadata: EncoderCacheManagerMetadata | None = None
    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    new_block_ids_to_zero: list[int] | None = None

    # CoW copies to apply after zeroing new blocks and before forward.
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None

    # Complete block-table rows that replace incrementally appended block IDs.
    block_table_updates: dict[str, tuple[list[int], ...]] | None = None

    # Scheduler-local; always None by the time this reaches a worker.
    kv_connector_block_state: KVConnectorBlockState | None = None

    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    num_spec_tokens_to_schedule: int = 0

    # Explicit phase contract for target-model collectives. True only when all
    # scheduled rows are ordinary autoregressive decode rows (including prior
    # draft positions), never for prefill, replay or mixed batches.
    is_pure_decode_step: bool = False

    # Experimental separate-pool GDN prefix checkpoint commands. Keys are the
    # exact chained content hashes of scheduler-block boundaries. Workers save
    # after a successful forward and restore before preprocess_mamba.
    gdn_checkpoint_save: dict[str, bytes] | None = None
    gdn_checkpoint_restore: dict[str, bytes] | None = None
    gdn_checkpoint_plan: tuple[int, tuple[bytes, ...], tuple[bytes, ...]] | None = None
    gdn_checkpoint_budget: tuple[int, int] | None = None

    # Physical mapped-prefix sizes for (attention, GDN) stable-VA arenas.
    elastic_kv_transition: tuple[int, int] | None = None

    # Step-scoped physical bytes loaned from the elastic KV arena to dynamic
    # CUDA Graph pools and transient consumers. Zero restores the X1 KV baseline.
    elastic_external_memory_bytes: int = 0

    # Revocable native expert growth above the bank paid during model profiling.
    # This is separate from Graph/MM external allocations.
    elastic_expert_grant: ElasticExpertGrant | None = None

    # Immutable breakdown of the aggregate external loan. The Graph endpoint
    # is absolute; the MM activation loan is incremental and step-scoped.
    elastic_graph_external_memory_bytes: int = 0
    elastic_mm_activation_loan_bytes: int = 0

    # Exact KV primary-block delta reserved for the sampled successor of this
    # step. It is computed once during scheduling and carried through
    # settlement so mutable request state cannot create a second authority.
    elastic_successor_primary_headroom: int = 0

    # Immutable transaction identity shared by scheduler and every worker rank.
    elastic_transaction_id: str | None = None

    # Canonical physical Graph shape committed when this output was scheduled.
    # Request cancellation may change the scheduler's live carrier before the
    # worker result settles, so settlement must not reconstruct this identity
    # from mutable scheduler state.
    elastic_graph_step_key: tuple[int, ...] | None = None

    # Complete physical plan selected before worker-side CUDA/KV mutation.
    # Workers validate its runtime generation and rank fingerprint and never
    # reconstruct owner identity or eviction policy from request metadata.
    elastic_step_plan: ElasticStepPlan | None = None
    elastic_plan_fingerprint: str | None = None

    # Request-free barrier emitted immediately after a staged hotset
    # publication. Workers retire only the hidden victim fences, keep the new
    # candidate resident, and publish a synchronized physical receipt before
    # the scheduler may observe a changed request/cache shape.

    # A zero-token service tick has no next execution shape and therefore must
    # not evict the last HOT graph merely because the engine is idle. Explicit
    # teardown/recovery leaves this false and requests X0 instead.
    elastic_preserve_graph_residency: bool = False

    # True only for scheduler-bypassing startup kernel warmup. These batches
    # exercise compiled kernels and KV mappings, but are not live admissions
    # and therefore cannot authorize an on-demand CUDA Graph/KV loan.
    is_synthetic_warmup: bool = False

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
