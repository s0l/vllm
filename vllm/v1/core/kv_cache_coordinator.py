# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence
from typing import NamedTuple

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_capacity import PhysicalPoolCapacityPlanner
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    KVCacheBlock,
    dcp_world_size_for_kv_cache_spec,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


def _validate_prefix_cache_retention_interval(
    retention_interval: int | None,
    scheduler_block_size: int,
    kv_cache_config: KVCacheConfig,
) -> None:
    if retention_interval is None:
        return

    # Retention sparsifies sliding-window and Mamba (linear-attention)
    # checkpoints; full-attention and chunked-local groups cache densely and
    # ignore it (their hit granularity must stay fine).
    if not any(
        isinstance(g.kv_cache_spec, (SlidingWindowSpec, MambaSpec))
        for g in kv_cache_config.kv_cache_groups
    ):
        if retention_interval == 0:
            return
        raise ValueError(
            "prefix_cache_retention_interval is set but this model has "
            "no sliding-window or Mamba KV cache group, so retention has no "
            "effect. Set it to 0 (it only applies to sliding-window and Mamba "
            "attention)."
        )

    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:
        raise ValueError(
            f"prefix_cache_retention_interval ({retention_interval}) "
            "must be non-negative and a multiple of scheduler_block_size "
            f"({scheduler_block_size})."
        )


class KVCacheBlockPoolRequirements(NamedTuple):
    primary: int = 0
    mamba: int = 0

    @property
    def total(self) -> int:
        return self.primary + self.mamba

    def __add__(  # type: ignore[override]  # Component sums, not tuple concatenation.
        self, other: "KVCacheBlockPoolRequirements"
    ) -> "KVCacheBlockPoolRequirements":
        return KVCacheBlockPoolRequirements(
            self.primary + other.primary,
            self.mamba + other.mamba,
        )


class ElasticAdmissionPlan(NamedTuple):
    max_requests: int
    requirements: KVCacheBlockPoolRequirements


class KVCacheCoordinator(ABC):
    """Coordinate the KV cache of different KV cache groups."""

    enable_partial_hash_hits = False

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.hash_block_size = hash_block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        # The scheduling granularity (LCM of all group block sizes), must be a multiple
        # of the hash_block_size and the block size of each group.
        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
        self.scheduler_block_size = scheduler_block_size
        self.num_reprefillable_tokens = max(0, num_prefill_lookahead - 1)

        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
            prefer_low_id_allocations=bool(kv_cache_config.elastic_mapping_quantum),
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        # During chunked prefill with EAGLE, the single next prefill lookahead
        # token past the chunk boundary is combined with the final hidden state
        # and written to the KV cache. Therefore, the final chunk token must be
        # excluded from prefix cache hits to prevent requests from acquiring the
        # KV cache slot polluted with the next prefill token, which may or may not
        # be present after the matching prefix. The last-block drop handles this
        # edge case. During multi-module MTP, the issue generalizes to a prefill
        # lookahead of num_speculative_tokens, so the dropped tail must be large
        # enough to contain them. Hits land on scheduler-block boundaries (see
        # `_cache_hit_alignment_tokens`), so the excluded tail is
        # scheduler_block_size, not the group's own block size.
        if (
            enable_caching
            and self.eagle_group_ids
            and scheduler_block_size < num_prefill_lookahead
        ):
            raise ValueError(
                f"Multi-module MTP with prefix caching requires scheduler_block_size"
                f" (={scheduler_block_size}) >= num_speculative_tokens"
                f" (={num_prefill_lookahead})."
            )

        separate_mamba_specs = [
            group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
            and group.kv_cache_spec.separate_pool
        ]
        separate_gdn_pool = bool(separate_mamba_specs)
        self.mamba_block_pool: BlockPool | None = None
        self.gdn_checkpoint_keys: OrderedDict[BlockHash, None] | None = None
        self.gdn_checkpoint_limit = 8
        self._elastic_transition: tuple[int, int] | None = None
        self._last_emitted_elastic_transition: tuple[int, int] | None = None
        self._physical_pool_planner: PhysicalPoolCapacityPlanner | None = None
        self.elastic_external_memory_bytes = 0
        self.elastic_expert_memory_bytes = 0
        self.elastic_expert_rank_memory_bytes: tuple[int, ...] = ()
        self.last_elastic_rejection: dict[str, object] | None = None
        if separate_gdn_pool:
            pool_sizes = {
                spec.separate_pool_num_blocks for spec in separate_mamba_specs
            }
            if len(pool_sizes) != 1 or 0 in pool_sizes:
                raise ValueError(
                    "Separate GDN cache groups must use one positive pool size"
                )
            self.mamba_block_pool = BlockPool(
                num_gpu_blocks=pool_sizes.pop(),
                enable_caching=enable_caching,
                hash_block_size=hash_block_size,
                enable_kv_cache_events=enable_kv_cache_events,
                metrics_collector=metrics_collector,
                active_num_gpu_blocks=(
                    kv_cache_config.elastic_gdn_initial_blocks or None
                ),
            )
            # This mirrors the pinned-host LRU in every worker. It contains
            # content identities only; recurrent bytes never live in EngineCore.
            self.gdn_checkpoint_keys = OrderedDict()

        if kv_cache_config.elastic_mapping_quantum and self.mamba_block_pool:
            attention_block_sizes = tuple(
                tensor.logical_block_size
                for tensor in kv_cache_config.kv_cache_tensors
                if tensor.backing_id.startswith("elastic-attention-")
            ) or (kv_cache_config.elastic_attention_stride,)
            self._physical_pool_planner = PhysicalPoolCapacityPlanner(
                primary_block_sizes=attention_block_sizes,
                secondary_block_stride=kv_cache_config.elastic_gdn_stride,
                mapping_quantum=kv_cache_config.elastic_mapping_quantum,
                budget_bytes=kv_cache_config.elastic_budget_bytes,
            )
            self._last_emitted_elastic_transition = (
                self.block_pool.active_num_gpu_blocks,
                self.mamba_block_pool.active_num_gpu_blocks,
            )

        managers: list[SingleTypeKVCacheManager] = []
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            manager_pool = (
                self.mamba_block_pool
                if self.mamba_block_pool is not None
                and isinstance(kv_cache_group.kv_cache_spec, MambaSpec)
                else self.block_pool
            )
            manager = get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                role=kv_cache_group.role,
                block_pool=manager_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size_for_kv_cache_spec(
                    kv_cache_group.kv_cache_spec, dcp_world_size
                ),
                pcp_world_size=pcp_world_size,
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
            )
            managers.append(manager)
        self.single_type_managers = tuple(managers)
        # Match Mamba checkpoints to Eagle's attention replay boundary.
        if use_eagle:
            for manager in self.single_type_managers:
                if isinstance(manager, MambaManager):
                    manager.drop_eagle_checkpoint_block = True
        self.group_block_sizes = tuple(
            manager.block_size for manager in self.single_type_managers
        )

        # A positive retention interval must be a multiple of the base hit granularity
        # (``scheduler_block_size``) to land on real cache-hit boundaries.
        # 0 = keep only the latest replay boundary; None = dense;
        self.retention_interval = kv_cache_config.prefix_cache_retention_interval
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )

    def _elastic_attention_capacity(
        self,
        gdn_blocks: int,
        external_memory_bytes: int | None = None,
        expert_memory_bytes: int | None = None,
    ) -> int:
        if external_memory_bytes is None:
            external_memory_bytes = self.elastic_external_memory_bytes
        expert_bytes = (
            self.elastic_expert_memory_bytes
            if expert_memory_bytes is None
            else expert_memory_bytes
        )
        primary_surfaces = self.kv_cache_config.elastic_rank_primary_mapped_bytes
        gdn_surfaces = self.kv_cache_config.elastic_rank_gdn_mapped_bytes
        rank_budgets = self.kv_cache_config.elastic_rank_budget_bytes
        if primary_surfaces or gdn_surfaces or rank_budgets:
            if not (len(primary_surfaces) == len(gdn_surfaces) == len(rank_budgets)):
                raise RuntimeError("elastic rank capacity surfaces are incomplete")
            quantum = self.kv_cache_config.elastic_mapping_quantum
            rank_experts = (
                self.elastic_expert_rank_memory_bytes
                if expert_memory_bytes is None
                else ()
            ) or (expert_bytes,) * len(rank_budgets)
            capacities = []
            for primary_bytes, gdn_bytes, budget, expert in zip(
                primary_surfaces, gdn_surfaces, rank_budgets, rank_experts, strict=True
            ):
                external_mapped = (
                    cdiv(external_memory_bytes + expert, quantum) * quantum
                )
                if not 0 <= gdn_blocks < len(gdn_bytes):
                    return 0
                available_primary = budget - gdn_bytes[gdn_blocks] - external_mapped
                capacities.append(bisect_right(primary_bytes, available_primary) - 1)
            return max(min(capacities), 0)

        rank_safe_capacity = (
            self.kv_cache_config.elastic_attention_capacity_by_gdn_blocks
        )
        if rank_safe_capacity:
            if not 0 <= gdn_blocks < len(rank_safe_capacity):
                return 0
            return rank_safe_capacity[gdn_blocks]
        assert self._physical_pool_planner is not None
        return self._physical_pool_planner.max_primary_blocks(
            gdn_blocks,
            upper_bound=self.block_pool.num_gpu_blocks,
            external_bytes=external_memory_bytes + expert_bytes,
        )

    def max_elastic_full_context_requests(
        self,
        primary_blocks_per_request: int,
        external_memory_bytes: int,
        max_requests: int,
    ) -> int:
        """Exact known-footprint capacity for max-length request residency.

        The result includes the permanent null block in both pools, GDN growth
        per request, rank-asymmetric mapped-byte surfaces and the supplied
        executable loan. It is exact to the configured mapping quantum; it is
        meaningful only when ``external_memory_bytes`` is measured or comes
        from a previously proven capture envelope.
        """
        if primary_blocks_per_request <= 0 or max_requests < 0:
            raise ValueError("elastic full-context capacity inputs are invalid")
        config = self.kv_cache_config
        gdn_per_request = config.elastic_gdn_blocks_per_request
        if gdn_per_request <= 0:
            return 0
        executable_x = 0
        for num_reqs in range(1, max_requests + 1):
            gdn_blocks = max(
                config.elastic_gdn_initial_blocks,
                1 + num_reqs * gdn_per_request,
            )
            attention_blocks = self._elastic_attention_capacity(
                gdn_blocks,
                external_memory_bytes,
            )
            if attention_blocks < 1 + num_reqs * primary_blocks_per_request:
                break
            executable_x = num_reqs
        return executable_x

    def elastic_full_context_capacity_receipt(
        self,
        primary_blocks_per_request: int,
        external_memory_bytes: int,
        num_reqs: int,
    ) -> dict[str, int]:
        """Return the exact mapped-block ledger for one proven runtime shape."""
        if primary_blocks_per_request <= 0 or num_reqs < 0:
            raise ValueError("elastic full-context receipt inputs are invalid")
        config = self.kv_cache_config
        gdn_blocks = max(
            config.elastic_gdn_initial_blocks,
            1 + num_reqs * config.elastic_gdn_blocks_per_request,
        )
        attention_blocks = self._elastic_attention_capacity(
            gdn_blocks,
            external_memory_bytes,
        )
        required_attention_blocks = 1 + num_reqs * primary_blocks_per_request
        return {
            "num_reqs": num_reqs,
            "external_memory_bytes": external_memory_bytes,
            "gdn_blocks": gdn_blocks,
            "attention_blocks": attention_blocks,
            "required_attention_blocks": required_attention_blocks,
            "residual_attention_blocks": (attention_blocks - required_attention_blocks),
        }

    def elastic_kv_authority_receipt(
        self,
        primary_blocks_per_max_request: int,
        external_memory_bytes: int | None = None,
    ) -> dict[str, object]:
        """Publish raw and executable KV capacity without conflating MaxX.

        ``attention_token_equivalent`` is deliberately expressed in
        max-model-length-equivalent scheduler tokens. Hybrid groups consume a
        different number of physical pages, so a single raw "token count" is
        otherwise ambiguous. The block ledger remains the physical authority.
        """
        if primary_blocks_per_max_request <= 0:
            raise ValueError("KV authority requires positive max-request blocks")
        config = self.kv_cache_config
        external = (
            self.elastic_external_memory_bytes
            if external_memory_bytes is None
            else external_memory_bytes
        )
        if external < 0:
            raise ValueError("KV authority external bytes cannot be negative")
        gdn_blocks = (
            self.mamba_block_pool.active_num_gpu_blocks
            if self.mamba_block_pool is not None
            else 0
        )

        def rank_capacities(
            gdn: int,
            graph_bytes: int,
            expert_bytes: int = 0,
            rank_experts: tuple[int, ...] = (),
        ) -> tuple[int, ...]:
            primary = config.elastic_rank_primary_mapped_bytes
            secondary = config.elastic_rank_gdn_mapped_bytes
            budgets = config.elastic_rank_budget_bytes
            if primary or secondary or budgets:
                if not (len(primary) == len(secondary) == len(budgets)):
                    raise RuntimeError("elastic rank capacity surfaces are incomplete")
                quantum = config.elastic_mapping_quantum
                capacities = []
                for primary_bytes, gdn_bytes, budget, expert in zip(
                    primary,
                    secondary,
                    budgets,
                    rank_experts or (expert_bytes,) * len(budgets),
                    strict=True,
                ):
                    mapped_graph = cdiv(graph_bytes + expert, quantum) * quantum
                    if not 0 <= gdn < len(gdn_bytes):
                        capacities.append(0)
                        continue
                    available = budget - gdn_bytes[gdn] - mapped_graph
                    capacities.append(
                        max(bisect_right(primary_bytes, available) - 1, 0)
                    )
                return tuple(capacities)
            return (self._elastic_attention_capacity(gdn, graph_bytes, expert_bytes),)

        raw_gdn = config.elastic_gdn_initial_blocks if self.mamba_block_pool else 0
        raw_blocks = rank_capacities(raw_gdn, 0)
        effective_blocks = rank_capacities(
            gdn_blocks,
            external,
            self.elastic_expert_memory_bytes,
            self.elastic_expert_rank_memory_bytes,
        )

        def mapped_attention_bytes(blocks_by_rank: tuple[int, ...]) -> tuple[int, ...]:
            primary = config.elastic_rank_primary_mapped_bytes
            if primary:
                if len(primary) != len(blocks_by_rank):
                    raise RuntimeError("elastic rank attention surfaces are incomplete")
                result = []
                for blocks, surface in zip(blocks_by_rank, primary, strict=True):
                    if not 0 <= blocks < len(surface):
                        raise RuntimeError(
                            "elastic attention block receipt exceeds a rank surface"
                        )
                    result.append(int(surface[blocks]))
                return tuple(result)
            stride = int(config.elastic_attention_stride)
            return tuple(blocks * stride for blocks in blocks_by_rank)

        def token_equivalent(blocks: int) -> int:
            usable_blocks = max(blocks - 1, 0)  # permanent null block
            return usable_blocks * self.max_model_len // primary_blocks_per_max_request

        return {
            "max_model_len": self.max_model_len,
            "scheduler_block_size": self.scheduler_block_size,
            "primary_blocks_per_max_request": primary_blocks_per_max_request,
            "raw_attention_blocks_per_rank": raw_blocks,
            "raw_attention_bytes_per_rank": mapped_attention_bytes(raw_blocks),
            "raw_attention_token_equivalent_per_rank": tuple(
                token_equivalent(blocks) for blocks in raw_blocks
            ),
            "effective_attention_blocks_per_rank": effective_blocks,
            "effective_attention_bytes_per_rank": mapped_attention_bytes(
                effective_blocks
            ),
            "effective_attention_token_equivalent_per_rank": tuple(
                token_equivalent(blocks) for blocks in effective_blocks
            ),
            "active_gdn_blocks": gdn_blocks,
            "graph_external_bytes": external,
            "expert_borrowed_bytes": self.elastic_expert_memory_bytes,
            "expert_borrowed_bytes_per_rank": self.elastic_expert_rank_memory_bytes,
            "rank_budget_bytes": tuple(config.elastic_rank_budget_bytes),
        }

    def set_elastic_expert_memory(
        self,
        requested_bytes: int,
        minimum_free_primary_blocks: int = 0,
        *,
        rank_requested_bytes: tuple[int, ...] = (),
    ) -> bool:
        """Change a revocable expert grant without relabeling it as Graph memory."""
        requested = self.normalize_elastic_external_memory(requested_bytes)
        if rank_requested_bytes and (
            len(rank_requested_bytes)
            != len(self.kv_cache_config.elastic_rank_budget_bytes)
            or any(
                type(v) is not int or v < 0 or v > requested
                for v in rank_requested_bytes
            )
        ):
            raise ValueError("invalid per-rank expert memory")
        if requested and not self.kv_cache_config.elastic_mapping_quantum:
            return False
        previous = self.elastic_expert_memory_bytes
        previous_ranks = self.elastic_expert_rank_memory_bytes
        self.elastic_expert_memory_bytes = requested
        self.elastic_expert_rank_memory_bytes = tuple(
            self.normalize_elastic_external_memory(v) for v in rank_requested_bytes
        )
        if self.set_elastic_external_memory(
            self.elastic_external_memory_bytes, minimum_free_primary_blocks
        ):
            return True
        self.elastic_expert_memory_bytes = previous
        self.elastic_expert_rank_memory_bytes = previous_ranks
        return False

    def set_elastic_external_memory(
        self,
        requested_bytes: int,
        minimum_free_primary_blocks: int = 0,
    ) -> bool:
        """Loan free attention-tail quanta to a step-scoped external owner."""
        config = self.kv_cache_config
        pool = self.mamba_block_pool
        if requested_bytes < 0:
            raise ValueError("elastic external memory cannot be negative")
        if minimum_free_primary_blocks < 0:
            raise ValueError("elastic primary headroom cannot be negative")
        if not config.elastic_mapping_quantum or pool is None:
            return requested_bytes == 0

        normalized = self.normalize_elastic_external_memory(requested_bytes)
        previous = self.elastic_external_memory_bytes
        self.elastic_external_memory_bytes = normalized
        desired_attention = self._elastic_attention_capacity(pool.active_num_gpu_blocks)
        if desired_attention < 1:
            self.elastic_external_memory_bytes = previous
            return self._reject_elastic_capacity(
                "external_physical_budget",
                KVCacheBlockPoolRequirements(),
                desired_gdn_blocks=pool.active_num_gpu_blocks,
                desired_attention_blocks=desired_attention,
            )

        used_primary = (
            self.block_pool.active_num_gpu_blocks
            - self.block_pool.get_num_free_blocks()
        )
        if desired_attention < used_primary + minimum_free_primary_blocks:
            self.elastic_external_memory_bytes = previous
            return self._reject_elastic_capacity(
                "external_attention_headroom",
                KVCacheBlockPoolRequirements(primary=minimum_free_primary_blocks),
                desired_gdn_blocks=pool.active_num_gpu_blocks,
                desired_attention_blocks=desired_attention,
            )

        current_attention = self.block_pool.active_num_gpu_blocks
        if desired_attention < current_attention:
            if not self.block_pool.deactivate_tail_blocks(desired_attention):
                pinned_tail = tuple(
                    (block.block_id, block.ref_cnt)
                    for block in self.block_pool.blocks[
                        desired_attention:current_attention
                    ]
                    if block.ref_cnt != 0
                )
                self.elastic_external_memory_bytes = previous
                return self._reject_elastic_capacity(
                    "external_attention_tail_pinned",
                    KVCacheBlockPoolRequirements(),
                    desired_gdn_blocks=pool.active_num_gpu_blocks,
                    desired_attention_blocks=desired_attention,
                    pinned_attention_tail=pinned_tail,
                )
        elif desired_attention > current_attention:
            self.block_pool.activate_tail_blocks(desired_attention)

        if desired_attention != current_attention:
            self._record_elastic_transition()
        self.last_elastic_rejection = None
        return True

    def normalize_elastic_external_memory(self, requested_bytes: int) -> int:
        """Return the physical mapping-quantum representation of a loan."""
        if requested_bytes < 0:
            raise ValueError("elastic external memory cannot be negative")
        quantum = self.kv_cache_config.elastic_mapping_quantum
        if not quantum:
            return requested_bytes
        return cdiv(requested_bytes, quantum) * quantum

    def max_elastic_external_memory(
        self,
        minimum_free_primary_blocks: int = 0,
        minimum_attention_blocks: int = 0,
        gdn_blocks: int | None = None,
    ) -> int:
        return min(
            self.max_elastic_external_memory_by_rank(
                minimum_free_primary_blocks, minimum_attention_blocks, gdn_blocks
            )
        )

    def max_elastic_external_memory_by_rank(
        self,
        minimum_free_primary_blocks: int = 0,
        minimum_attention_blocks: int = 0,
        gdn_blocks: int | None = None,
    ) -> tuple[int, ...]:
        """Return the largest step-scoped loan the current KV tail can fund."""
        config = self.kv_cache_config
        pool = self.mamba_block_pool
        quantum = config.elastic_mapping_quantum
        if not quantum or pool is None:
            return (0,) * max(len(config.elastic_rank_budget_bytes), 1)
        if minimum_free_primary_blocks < 0:
            raise ValueError("elastic primary headroom cannot be negative")
        if minimum_attention_blocks < 0:
            raise ValueError("elastic minimum attention capacity cannot be negative")
        if gdn_blocks is None:
            gdn_blocks = pool.active_num_gpu_blocks
        if not 0 <= gdn_blocks <= pool.num_gpu_blocks:
            raise ValueError("elastic GDN capacity is outside the virtual pool")

        current_attention = self.block_pool.active_num_gpu_blocks
        used_primary = current_attention - self.block_pool.get_num_free_blocks()
        minimum_attention = max(
            used_primary + minimum_free_primary_blocks,
            minimum_attention_blocks,
            1,
        )
        for block in self.block_pool.blocks[:current_attention]:
            if block.ref_cnt:
                minimum_attention = max(minimum_attention, block.block_id + 1)
        # This is a prospective query: the next admitted request can require
        # more primary blocks than are mapped under the current external loan.
        # Pricing only the currently active prefix overstates the loan that can
        # coexist with that request and moves the rejection past KV allocation.
        if minimum_attention > self.block_pool.num_gpu_blocks:
            return (0,) * max(len(config.elastic_rank_budget_bytes), 1)

        rank_budgets = config.elastic_rank_budget_bytes
        rank_primary = config.elastic_rank_primary_mapped_bytes
        rank_gdn = config.elastic_rank_gdn_mapped_bytes
        if rank_budgets and rank_primary and rank_gdn:
            available = tuple(
                budget
                - primary[min(minimum_attention, len(primary) - 1)]
                - gdn[min(gdn_blocks, len(gdn) - 1)]
                for budget, primary, gdn in zip(
                    rank_budgets, rank_primary, rank_gdn, strict=True
                )
            )
        else:
            assert self._physical_pool_planner is not None
            available = (
                self._physical_pool_planner.budget_bytes
                - self._physical_pool_planner.primary_mapped_bytes(minimum_attention)
                - self._physical_pool_planner.secondary_mapped_bytes(gdn_blocks),
            )
        experts = self.elastic_expert_rank_memory_bytes or (
            self.elastic_expert_memory_bytes,
        ) * len(available)
        return tuple(
            max((value - expert) // quantum * quantum, 0)
            for value, expert in zip(available, experts, strict=True)
        )

    def elastic_gdn_blocks_after_allocation(
        self,
        required_blocks: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...] = (),
    ) -> int:
        """Return post-allocation GDN residency with unused lookahead removed."""
        pool = self.mamba_block_pool
        if pool is None:
            return 0
        if required_blocks < 0:
            raise ValueError("elastic GDN allocation cannot be negative")
        current_free_blocks = pool.get_num_free_blocks()
        missing_blocks = max(required_blocks - current_free_blocks, 0)
        future_active_blocks = min(
            pool.active_num_gpu_blocks + missing_blocks,
            pool.num_gpu_blocks,
        )
        highest_used = max(
            (block.block_id for block in pool.blocks if block.ref_cnt),
            default=0,
        )
        prospective_blocks = pool.free_block_queue.peek_left_n(
            min(required_blocks, current_free_blocks)
        )
        highest_prospective = max(
            (block.block_id for block in prospective_blocks),
            default=0,
        )
        highest_cached = 0
        for blocks in new_computed_blocks:
            for block in blocks:
                block_id = block.block_id
                if (
                    0 <= block_id < pool.num_gpu_blocks
                    and block is pool.blocks[block_id]
                ):
                    highest_cached = max(highest_cached, block_id)
        return max(
            self.kv_cache_config.elastic_gdn_initial_blocks,
            future_active_blocks,
            highest_used + 1,
            highest_prospective + 1,
            highest_cached + 1,
        )

    def _record_elastic_transition(self) -> None:
        assert self.mamba_block_pool is not None
        self._elastic_transition = (
            self.block_pool.active_num_gpu_blocks,
            self.mamba_block_pool.active_num_gpu_blocks,
        )

    def _reject_elastic_capacity(
        self,
        reason: str,
        requirements: "KVCacheBlockPoolRequirements",
        *,
        desired_gdn_blocks: int,
        desired_attention_blocks: int | None = None,
        pinned_attention_tail: tuple[tuple[int, int], ...] = (),
    ) -> bool:
        assert self.mamba_block_pool is not None
        rejection: dict[str, object] = {
            "reason": reason,
            "requirements": requirements,
            "attention_active": self.block_pool.active_num_gpu_blocks,
            "attention_free": self.block_pool.get_num_free_blocks(),
            "gdn_active": self.mamba_block_pool.active_num_gpu_blocks,
            "gdn_free": self.mamba_block_pool.get_num_free_blocks(),
            "desired_gdn": desired_gdn_blocks,
            "desired_attention": desired_attention_blocks,
            "pinned_attention_tail": pinned_attention_tail,
        }
        if rejection != self.last_elastic_rejection:
            logger.warning("Elastic KV admission rejected: %s", rejection)
        self.last_elastic_rejection = rejection
        return False

    def ensure_elastic_capacity(
        self, requirements: "KVCacheBlockPoolRequirements"
    ) -> bool:
        """Prepare enough mapped logical capacity before block allocation."""
        config = self.kv_cache_config
        pool = self.mamba_block_pool
        if not config.elastic_mapping_quantum or pool is None:
            return True
        missing = max(requirements.mamba - pool.get_num_free_blocks(), 0)
        if missing == 0:
            self.last_elastic_rejection = None
            return True
        new_gdn_blocks = pool.active_num_gpu_blocks + missing
        if new_gdn_blocks > pool.num_gpu_blocks:
            return self._reject_elastic_capacity(
                "virtual_gdn_limit",
                requirements,
                desired_gdn_blocks=new_gdn_blocks,
            )
        new_attention_blocks = self._elastic_attention_capacity(new_gdn_blocks)
        if new_attention_blocks < 1:
            return self._reject_elastic_capacity(
                "physical_budget",
                requirements,
                desired_gdn_blocks=new_gdn_blocks,
                desired_attention_blocks=new_attention_blocks,
            )
        if not self.block_pool.deactivate_tail_blocks(new_attention_blocks):
            pinned_tail = tuple(
                (block.block_id, block.ref_cnt)
                for block in self.block_pool.blocks[
                    new_attention_blocks : self.block_pool.active_num_gpu_blocks
                ]
                if block.ref_cnt != 0
            )
            return self._reject_elastic_capacity(
                "attention_tail_pinned",
                requirements,
                desired_gdn_blocks=new_gdn_blocks,
                desired_attention_blocks=new_attention_blocks,
                pinned_attention_tail=pinned_tail,
            )
        pool.activate_tail_blocks(new_gdn_blocks)
        self._record_elastic_transition()
        self.last_elastic_rejection = None
        return True

    def plan_elastic_admission_wave(
        self,
        primary_blocks_per_request: Sequence[int],
        *,
        external_memory_bytes: int | None = None,
    ) -> ElasticAdmissionPlan | None:
        """Compute MaxX for the next admission wave without mutating pools.

        Incremental GDN growth can otherwise let early requests take logical
        attention IDs from a tail that a later request needs to deactivate.
        """
        pool = self.mamba_block_pool
        blocks_per_request = self.kv_cache_config.elastic_gdn_blocks_per_request
        if (
            not primary_blocks_per_request
            or not self.kv_cache_config.elastic_mapping_quantum
            or pool is None
            or blocks_per_request <= 0
        ):
            return None
        if any(blocks < 0 for blocks in primary_blocks_per_request):
            raise ValueError("primary admission requirements cannot be negative")
        if external_memory_bytes is None:
            external_memory_bytes = self.elastic_external_memory_bytes
        if external_memory_bytes < 0:
            raise ValueError("external admission requirement cannot be negative")

        max_virtual_requests = (pool.num_gpu_blocks - 1) // blocks_per_request
        candidates = min(len(primary_blocks_per_request), max_virtual_requests)
        # active - free includes the null block and every referenced primary
        # block. Cached tail blocks have ref_cnt=0 and may be evicted.
        used_primary = (
            self.block_pool.active_num_gpu_blocks
            - self.block_pool.get_num_free_blocks()
        )
        for num_requests in range(candidates, 0, -1):
            required_primary = sum(primary_blocks_per_request[:num_requests])
            required_free_gdn = num_requests * blocks_per_request
            missing_gdn = max(required_free_gdn - pool.get_num_free_blocks(), 0)
            desired_gdn = pool.active_num_gpu_blocks + missing_gdn
            if desired_gdn > pool.num_gpu_blocks:
                continue

            # A prospective loan return can grow attention capacity even when
            # the GDN pool already covers the next wave.
            desired_attention = self._elastic_attention_capacity(
                desired_gdn, external_memory_bytes
            )
            # Preserve every candidate's conservative full-sequence primary
            # requirement in addition to null/currently referenced blocks.
            if desired_attention < used_primary + required_primary:
                continue
            if desired_attention < self.block_pool.active_num_gpu_blocks and any(
                block.ref_cnt != 0
                for block in self.block_pool.blocks[
                    desired_attention : self.block_pool.active_num_gpu_blocks
                ]
            ):
                continue

            return ElasticAdmissionPlan(
                max_requests=num_requests,
                requirements=KVCacheBlockPoolRequirements(
                    primary=required_primary,
                    mamba=required_free_gdn,
                ),
            )
        return None

    def apply_elastic_admission_wave(
        self,
        primary_blocks_per_request: Sequence[int],
        *,
        external_memory_bytes: int | None = None,
    ) -> int:
        """Compute and apply the feasible MaxX layout before allocation."""
        explicit_external = external_memory_bytes is not None
        plan = self.plan_elastic_admission_wave(
            primary_blocks_per_request,
            external_memory_bytes=external_memory_bytes,
        )
        if plan is None:
            return 0
        previous_external = self.elastic_external_memory_bytes
        if explicit_external:
            assert external_memory_bytes is not None
            if not self.set_elastic_external_memory(
                external_memory_bytes,
                minimum_free_primary_blocks=plan.requirements.primary,
            ):
                return 0
        if not self.ensure_elastic_capacity(plan.requirements):
            if explicit_external:
                self.rebalance_elastic_capacity()
                if not self.set_elastic_external_memory(previous_external):
                    raise RuntimeError(
                        "elastic admission failure could not restore external memory"
                    )
            return 0
        return plan.max_requests

    def rebalance_elastic_capacity(self) -> None:
        """Return unused GDN tail mappings to the attention arena."""
        config = self.kv_cache_config
        pool = self.mamba_block_pool
        if not config.elastic_mapping_quantum or pool is None:
            return
        highest_used = max(
            (block.block_id for block in pool.blocks if block.ref_cnt > 0),
            default=0,
        )
        desired_gdn = max(config.elastic_gdn_initial_blocks, highest_used + 1)
        changed = desired_gdn != pool.active_num_gpu_blocks
        if desired_gdn < pool.active_num_gpu_blocks:
            assert pool.deactivate_tail_blocks(desired_gdn)
        desired_attention = self._elastic_attention_capacity(desired_gdn)
        if desired_attention > self.block_pool.active_num_gpu_blocks:
            self.block_pool.activate_tail_blocks(desired_attention)
            changed = True
        if changed:
            self._record_elastic_transition()

    def take_elastic_transition(self) -> tuple[int, int] | None:
        transition = self._elastic_transition
        self._elastic_transition = None
        if transition == self._last_emitted_elastic_transition:
            return None
        if transition is not None:
            self._last_emitted_elastic_transition = transition
        return transition

    def register_gdn_checkpoint(self, key: bytes) -> None:
        if self.gdn_checkpoint_keys is None:
            return
        block_hash = BlockHash(key)
        self.gdn_checkpoint_keys[block_hash] = None
        self.gdn_checkpoint_keys.move_to_end(block_hash)
        while len(self.gdn_checkpoint_keys) > self.gdn_checkpoint_limit:
            self.gdn_checkpoint_keys.popitem(last=False)

    def sync_gdn_checkpoints(self, keys: tuple[bytes, ...]) -> None:
        """Replace scheduler membership with the authoritative worker LRU."""
        if self.gdn_checkpoint_keys is None:
            return
        if len(keys) > self.gdn_checkpoint_limit or len(set(keys)) != len(keys):
            raise RuntimeError("Invalid worker GDN checkpoint snapshot")
        self.gdn_checkpoint_keys = OrderedDict((BlockHash(key), None) for key in keys)

    def has_gdn_checkpoint(self, key: BlockHash, *, touch: bool = True) -> bool:
        if self.gdn_checkpoint_keys is None or key not in self.gdn_checkpoint_keys:
            return False
        if touch:
            self.gdn_checkpoint_keys.move_to_end(key)
        return True

    def find_gdn_checkpoint_boundary(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
        spec: MambaSpec,
    ) -> int:
        """Return the newest exact recurrent checkpoint boundary in tokens."""
        checkpoint_block_size = (
            self.hash_block_size
            if getattr(self, "enable_dcp_fine_prefix", False)
            else spec.block_size * self.dcp_world_size * self.pcp_world_size
        )
        spec_hashes = BlockHashListWithBlockSize(
            block_hashes,
            self.hash_block_size,
            checkpoint_block_size,
        )
        num_blocks = min(max_length // checkpoint_block_size, len(spec_hashes))
        while num_blocks > 0 and not self.has_gdn_checkpoint(
            spec_hashes[num_blocks - 1], touch=False
        ):
            num_blocks -= 1
        # Lookup can run for a waiting request that is not admitted in this
        # scheduler step. Touching the scheduler-side LRU here would then have
        # no matching restore in the workers and can make their eviction order
        # diverge. The scheduler touches the key only when it emits the actual
        # restore command.
        return num_blocks * checkpoint_block_size

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """Get the number of device blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.
            total_computed_tokens: Include both local and external tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, apply the recycling-aware
                per-request admission cap (SWA / chunked-local). Set only by
                the full-sequence admission gate; per-step allocation must
                leave it False so the predictor matches `allocate_new_blocks`.

        Returns:
            The number of blocks to allocate.

        """
        requirements = self.get_block_pool_requirements(
            request_id=request_id,
            num_tokens=num_tokens,
            new_computed_blocks=new_computed_blocks,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
        return requirements.total

    def get_block_pool_requirements(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> "KVCacheBlockPoolRequirements":
        primary_blocks = 0
        mamba_blocks = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                manager_blocks = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                manager_blocks = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
            if manager.block_pool is self.mamba_block_pool:
                mamba_blocks += manager_blocks
            else:
                primary_blocks += manager_blocks
        return KVCacheBlockPoolRequirements(primary_blocks, mamba_blocks)

    def can_allocate(
        self,
        requirements: "KVCacheBlockPoolRequirements",
        reserved: "KVCacheBlockPoolRequirements | None" = None,
        primary_watermark_blocks: int = 0,
    ) -> bool:
        reserved = reserved or KVCacheBlockPoolRequirements()
        if (
            requirements.primary + reserved.primary + primary_watermark_blocks
            > self.block_pool.get_num_free_blocks()
        ):
            return False
        mamba_required = requirements.mamba + reserved.mamba
        if mamba_required == 0:
            return True
        return (
            self.mamba_block_pool is not None
            and mamba_required <= self.mamba_block_pool.get_num_free_blocks()
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Add the new computed blocks to the request. Optionally allocate new
            blocks for external computed tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.

        """
        # A running request is already tracked in num_cached_block and won't
        # have new prefix-cache hits, so this is a no-op for it.
        if any(
            request_id in manager.num_cached_block
            for manager in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # Two-phase allocation (issue #33775): first touch every group's local
        # cache-hit blocks, then allocate external blocks for every group. This
        # ensures an earlier group's external `get_new_blocks` cannot evict a
        # later group's not-yet-touched cache-hit blocks.
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens > 0:
            for manager in self.single_type_managers:
                manager.allocate_external_computed_blocks(
                    request_id,
                    num_local_computed_tokens,
                    num_external_computed_tokens,
                )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.

        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    def get_replay_boundaries(self, request: Request) -> tuple[int, ...]:
        """Positions a later request replaying this prompt can resume at.

        A hit is the shortest across all groups, so every group retains state
        at each position; EAGLE groups also keep the block above, which they
        match and drop back from (see ``reachable_block_mask``).

        Two positions are reachable: a resend of the identical prompt is capped
        at ``num_tokens - 1`` (its last token is recomputed for logits), a
        longer sibling matches the final aligned block. They differ only on a
        block-aligned prompt, where retaining just the higher one collapses the
        resend's hit to 0. The alignment is the scheduler block size, not the
        finer hash granularity, which would over-estimate the reach.
        """
        if not self.eagle_group_ids:
            return (request.num_prompt_tokens - 1,)
        block = self.scheduler_block_size
        resend = (request.num_prompt_tokens - 1) // block * block
        extension = request.num_prompt_tokens // block * block
        return tuple(sorted({max(resend - block, 0), max(extension - block, 0)}))

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).

        """
        boundaries = self.get_replay_boundaries(request)
        for manager in self.single_type_managers:
            if not manager.enable_caching:
                continue
            # Only cache tokens with finalized KV. The last num_reprefillable_tokens
            # tokens can be re-prefilled during multi-module MTP.
            num_tokens_to_cache = max(
                0, num_computed_tokens - self.num_reprefillable_tokens
            )
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
                replay_boundaries=boundaries,
            )

    def emit_cached_block_events(
        self, request: Request, computed_blocks: tuple[list[KVCacheBlock], ...]
    ) -> None:
        for group_idx, group_blocks in enumerate(computed_blocks):
            if group_blocks:
                manager = self.single_type_managers[group_idx]
                group = self.kv_cache_config.kv_cache_groups[group_idx]
                manager.block_pool.emit_cached_block_events(
                    request,
                    len(group_blocks),
                    group.kv_cache_spec.block_size,
                    group_idx,
                )

    def reset_prefix_cache(self) -> bool:
        """Reset each manager's pool once."""
        pools = dict.fromkeys(
            manager.block_pool for manager in self.single_type_managers
        )
        return all([pool.reset_prefix_cache() for pool in pools])

    def free(self, request_id: str) -> None:
        """Free the blocks for the request.

        Args:
            request_id: The request ID.

        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping from all single-type managers and
        return its blocks without returning them to the block pool. The
        caller must eventually pass the returned blocks to
        `block_pool.free_blocks`, freeing them in reverse order (so that
        tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.

        """
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Get the number of common prefix blocks for all requests with allocated
        KV cache for each kv cache group.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache group.

        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length. R-SWA managers use this to
                free gap blocks between the prefill tail and decode window; other
                manager types ignore it.

        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """Get the blocks for the request."""
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """Returns the per-group hit blocks, the hit length, and the number of
        ``num_uncached_common_prefix_tokens`` (a shared prefix that a
        sparse-retention group has not cached yet; 0 unless hybrid)."""
        pass

    def new_step_starts(self) -> None:
        """Notify each manager that a new step is starting."""
        for manager in self.single_type_managers:
            manager.new_step_starts()


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.dcp_world_size = self.single_type_managers[0].dcp_world_size
        self.pcp_world_size = pcp_world_size
        self.block_size = self.single_type_managers[0].block_size
        # For models using only Mamba, block_size is set to max_model_len when
        # prefix caching is disabled, and hash_block_size validation is skipped.
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )
        # Single group; useless but just set ``use_eagle`` for consistency regardless.
        self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        # Single group: nothing "uncached common" -- no other group to lag it.
        return hit_blocks, hit_length, 0


class SpecGroup(NamedTuple):
    """KV cache groups that share one spec, batched together for a single
    cache-hit lookup.

    ``use_eagle`` is True iff any member group is an EAGLE/MTP group. Members
    sharing a spec are cached and looked up jointly, so the EAGLE last-block drop
    is necessarily decided for the whole spec group.
    """

    spec: KVCacheSpec
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]
    use_eagle: bool


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
        allow_partial_hash_hits: bool = True,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        # Only groups that participate in prefix caching must satisfy the
        # divisibility constraint; groups that opt out (e.g. GLM-5.3-Flash kpool
        # tail, block_size=kpool) are scratch buffers and excluded.
        cacheable_block_sizes = [
            manager.block_size
            for manager, group in zip(
                self.single_type_managers, kv_cache_config.kv_cache_groups
            )
            if group.kv_cache_spec.prefix_cacheable
        ]
        assert all(
            block_size % hash_block_size == 0 for block_size in cacheable_block_sizes
        ), (
            "Each KV cache group's real block_size must be divisible by "
            f"hash_block_size. block_sizes={cacheable_block_sizes}, "
            f"hash_block_size={hash_block_size}"
        )
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        if dcp_world_size > 1:
            # DCP shards full-attention KV; Mamba and explicitly replicated
            # circular state keep one complete owner on each rank.
            for g in kv_cache_config.kv_cache_groups:
                spec = g.kv_cache_spec
                assert isinstance(spec, (FullAttentionSpec, MambaSpec)) or (
                    isinstance(spec, CircularBufferSpec) and spec.dcp_replicated
                ), (
                    "DCP with hybrid KV cache layouts only supports "
                    "full-attention, Mamba and replicated circular groups, got: "
                    f"{type(g.kv_cache_spec).__name__}."
                )
        # Fine-grained hash hits require Mamba "align" and compatible cache
        # managers in every group. TP needs hashing finer than the Mamba block;
        # DCP accepts equality because it scales the effective full-attention
        # block instead.
        has_partial_mamba_group = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            and g.kv_cache_spec.mamba_cache_mode == "align"
            and (
                (dcp_world_size == 1 and g.kv_cache_spec.block_size > hash_block_size)
                or (
                    dcp_world_size > 1 and g.kv_cache_spec.block_size >= hash_block_size
                )
            )
            for g in kv_cache_config.kv_cache_groups
        )
        self.enable_partial_hash_hits = (
            allow_partial_hash_hits and has_partial_mamba_group
        )
        self.enable_dcp_fine_prefix = (
            os.environ.get("AG2_VLLM_DCP_FINE_PREFIX", "0") == "1"
        )
        if self.enable_dcp_fine_prefix:
            if dcp_world_size <= 1:
                raise ValueError("AG2_VLLM_DCP_FINE_PREFIX requires DCP world size > 1")
            if self.gdn_checkpoint_keys is None:
                raise ValueError(
                    "AG2_VLLM_DCP_FINE_PREFIX requires a separate GDN pool"
                )
            if hash_block_size % dcp_world_size != 0:
                raise ValueError(
                    "DCP fine-prefix match unit must be divisible by DCP world size"
                )
            if not all(
                not g.kv_cache_spec.prefix_cacheable
                or isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec))
                for g in kv_cache_config.kv_cache_groups
            ):
                raise ValueError(
                    "DCP fine-prefix only supports prefix-cacheable "
                    "full-attention + Mamba groups"
                )
        # Generic DCP partial hits use upstream per-group geometry. Only the
        # separate host-state adapter needs its own activation gate.
        self.enable_partial_hash_hits = (
            self.gdn_checkpoint_keys is None
            or dcp_world_size == 1
            or self.enable_dcp_fine_prefix
        ) and has_partial_mamba_group
        if self.enable_partial_hash_hits:
            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager, group in zip(
                    self.single_type_managers, kv_cache_config.kv_cache_groups
                )
                if group.kv_cache_spec.prefix_cacheable
                and not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
            if unsupported_partial_hit_managers:
                self.enable_partial_hash_hits = False
                logger.warning_once(
                    "Disabling fine-grained prefix-cache hits because these KV "
                    "cache managers require block-aligned lookups: %s.",
                    ", ".join(sorted(unsupported_partial_hit_managers)),
                )
        cache_hit_alignment_tokens = self._cache_hit_alignment_tokens
        for manager in self.single_type_managers:
            manager.cache_hit_alignment_tokens = cache_hit_alignment_tokens
        self.verify_and_split_kv_cache_groups()

    @property
    def _cache_hit_alignment_tokens(self) -> int:
        # Fine-grained partial hits may return hash-block-aligned lengths;
        # otherwise it must stay scheduler-block-aligned.
        return (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.scheduler_block_size
        )

    def verify_and_split_kv_cache_groups(self) -> None:
        """Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        self.attention_groups: list[SpecGroup] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            # Skip groups that opt out of prefix caching (e.g. GLM-5.3-Flash
            # kpool tail): their blocks are per-request scratch, never
            # shareable, so they must not participate in hit lookup (their
            # manager-level hooks already no-op). Their slot in the per-group
            # hit tuple stays empty.
            if not g.kv_cache_spec.prefix_cacheable:
                continue
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec
            use_eagle = i in self.eagle_group_ids

            # Try to find an existing group with the same spec
            for idx, group in enumerate(self.attention_groups):
                if (
                    group.spec == spec
                    and group.manager_cls is manager_cls
                    and self.single_type_managers[group.group_ids[0]].block_pool
                    is self.single_type_managers[i].block_pool
                ):
                    group.group_ids.append(i)
                    if use_eagle and not group.use_eagle:
                        self.attention_groups[idx] = group._replace(use_eagle=True)
                    break
            else:
                self.attention_groups.append(
                    SpecGroup(spec, [i], manager_cls, use_eagle)
                )

        assert self.attention_groups, (
            "HybridKVCacheCoordinator requires at least one cacheable group."
        )

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups.sort(
            key=lambda g: not isinstance(g.spec, FullAttentionSpec)
        )

        # Dense reference group for per-group lookups (None when the model
        # has no full-attention layers): full attention is downward-closed,
        # so any group reporting a longer per-group hit implies the union of
        # per-group hits is not consistent at a single boundary (#46453).
        first = self.attention_groups[0]
        self.full_attention_group_id: int | None = (
            first.group_ids[0] if isinstance(first.spec, FullAttentionSpec) else None
        )

        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True

    def _align_cacheable(self, num_tokens: int) -> int:
        """Largest prefix of ``num_tokens`` a future cache hit could match.

        Hits are ``scheduler_block_size``-aligned (see
        ``find_longest_cache_hit``) unless fine-grained partial hash hits are
        enabled, in which case no rounding applies -- rounding even to
        ``hash_block_size`` would re-register a privatized Mamba tail.
        """
        if self.enable_partial_hash_hits:
            return num_tokens
        return round_down(num_tokens, self.scheduler_block_size)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        cached_num_computed_tokens = self._align_cacheable(num_computed_tokens)
        boundaries = self.get_replay_boundaries(request)
        for manager in self.single_type_managers:
            if not manager.enable_caching:
                continue
            num_tokens_to_cache = cached_num_computed_tokens
            # EAGLE groups match one block past each aligned boundary and drop
            # it, so make that lookahead block eligible to be cached.
            if manager.use_eagle and cached_num_computed_tokens > 0:
                # Only cache tokens with finalized KV. The last
                # num_reprefillable_tokens tokens can be re-prefilled during
                # multi-module MTP.
                num_finalized_computed_tokens = max(
                    0, num_computed_tokens - self.num_reprefillable_tokens
                )
                cached_num_finalized_computed_tokens = self._align_cacheable(
                    num_finalized_computed_tokens
                )
                num_tokens_to_cache = min(
                    num_finalized_computed_tokens,
                    cached_num_finalized_computed_tokens + manager.block_size,
                )
            # The manager already knows the fine hit granularity
            # (``scheduler_block_size``); retention is passed separately so it
            # can keep both the coarse segment tails and the fine replay
            # boundary (which needs the fine value).
            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
                replay_boundaries=boundaries,
            )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the reconciled (combined) cache hit.
                - ``num_uncached_common_prefix_tokens``: a shared prefix that a
                  sparse-retention group has not cached yet (0 unless hybrid).

        """
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
                self.attention_groups
            ):
                first_group_id = group_ids[0]
                # DCP/PCP shard each block's KV across ranks, so the manager's
                # effective block size may exceed the spec's.
                group_block_size = self.single_type_managers[first_group_id].block_size
                cached_blocks = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified

                _max_length = curr_hit_length
                # Eagle matches one extra drop unit (one hash unit for
                # fine-grained managers, else one cache block) and then drops
                # it, landing back at the candidate length. No margin for
                # mamba: its finder never drops (draft models have no mamba
                # layers), so the hit would grow past the candidate.
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and group_block_size > self.hash_block_size
                        else group_block_size
                    )
                    _max_length = min(
                        curr_hit_length + eagle_margin, max_cache_hit_length
                    )
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.single_type_managers[first_group_id].block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=self.single_type_managers[
                        first_group_id
                    ].dcp_world_size,
                    pcp_world_size=self.single_type_managers[
                        first_group_id
                    ].pcp_world_size,
                )
                if isinstance(spec, MambaSpec) and spec.separate_pool:
                    # The exact recurrent bytes live in the worker host LRU,
                    # not in a GPU KV block. Search exact DCP/PCP-aligned
                    # boundaries newest-first and represent a verified hit with
                    # positional nulls. Allocation later creates one live slot.
                    checkpoint_block_size = (
                        self.hash_block_size
                        if self.enable_dcp_fine_prefix
                        else spec.block_size * self.dcp_world_size * self.pcp_world_size
                    )
                    _new_hit_length = self.find_gdn_checkpoint_boundary(
                        block_hashes, _max_length, spec
                    )
                    num_blocks = _new_hit_length // checkpoint_block_size
                    hit_blocks = tuple(
                        [self.block_pool.null_block] * num_blocks for _ in group_ids
                    )
                else:
                    hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                        block_hashes=block_hashes,
                        max_length=_max_length,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        drop_eagle_block=drop_eagle_block,
                        alignment_tokens=self._cache_hit_alignment_tokens,
                        dcp_world_size=self.single_type_managers[
                            first_group_id
                        ].dcp_world_size,
                        pcp_world_size=self.single_type_managers[
                            first_group_id
                        ].pcp_world_size,
                    )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate every full-attention group (target and draft) blocks
        # to final hit_length.
        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            manager = self.single_type_managers[group.group_ids[0]]
            if manager.retains_longer_hit:
                continue
            group_block_size = manager.block_size
            num_blocks = cdiv(hit_length, group_block_size)
            for group_id in group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        # Uncached shared prefix detection: if any attn. group cached a longer
        # prefix than the reconciled hit, it is an uncached common prefix across
        # requests that a sparse-retention group hasn't cached yet.
        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        )
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens

    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
        """Like find_longest_cache_hit but evaluates each group independently.

        Returns:
            (blocks_per_group, hit_lengths_per_group)

        """
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        hit_lengths: list[int] = [0] * num_groups

        for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
            if isinstance(spec, MambaSpec) and spec.separate_pool:
                group_hit = self.find_gdn_checkpoint_boundary(
                    block_hashes, max_cache_hit_length, spec
                )
                unit = (
                    self.hash_block_size
                    if self.enable_dcp_fine_prefix
                    else spec.block_size * self.dcp_world_size * self.pcp_world_size
                )
                for gid in group_ids:
                    hit_blocks[gid] = [self.block_pool.null_block] * (group_hit // unit)
                    hit_lengths[gid] = group_hit
                continue
            blocks, group_hit = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.single_type_managers[group_ids[0]].block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=use_eagle,
                alignment_tokens=self._cache_hit_alignment_tokens,
                dcp_world_size=self.single_type_managers[group_ids[0]].dcp_world_size,
                pcp_world_size=self.single_type_managers[group_ids[0]].pcp_world_size,
            )
            for gid, blks in zip(group_ids, blocks):
                hit_blocks[gid] = blks
                hit_lengths[gid] = group_hit

        return tuple(hit_blocks), tuple(hit_lengths)


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
    num_prefill_lookahead: int = 0,
    allow_partial_hash_hits: bool = True,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
        num_prefill_lookahead=num_prefill_lookahead,
        allow_partial_hash_hits=allow_partial_hash_hits,
    )
