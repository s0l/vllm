# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import gc
import hashlib
import itertools
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

import regex as re
import torch
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import (
    GraphCaptureContext,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.v1.core.elastic_graph import (
    DispatchRepresentation,
    ElasticGraphError,
    ElasticPlanKind,
    ElasticResidencyEntry,
    ElasticResidencyReceipt,
    ElasticRuntimeConfig,
    ElasticStepPlan,
    ExecutionManifest,
    GraphExecutionPolicy,
    LogicalDispatchKey,
    OwnerGraphExecutionPolicy,
    PhysicalReplayKey,
    RuntimeGeneration,
    build_execution_manifest,
    configured_compiled_piecewise_sizes,
)
from vllm.v1.executor.worker_failure import WorkerFailureCode
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    elastic_piecewise_token_boundary,
)
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup, clear_layer_kv_caches

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

logger = init_logger(__name__)


class ElasticExecutionPlanMismatch(RuntimeError):
    """Current owner geometry differs before any Graph or KV mutation."""

    worker_failure_code = WorkerFailureCode.ELASTIC_EXECUTION_PLAN_MISMATCH


def _execution_manifest_difference(
    planned: ExecutionManifest, observed: ExecutionManifest
) -> str:
    """Describe identity-only differences without logging prompt contents."""
    differences = []
    for name in (
        "generation",
        "invocations",
        "request_ids",
        "per_request_query_lens",
        "per_request_is_prefilling",
        "scheduled_draft_rows",
        "scheduled_encoder_inputs",
        "active_lora_ids",
        "requested_output_k",
        "executed_drafter_k",
        "schema",
    ):
        planned_value = getattr(planned, name)
        observed_value = getattr(observed, name)
        if planned_value == observed_value:
            continue
        if isinstance(planned_value, tuple) and isinstance(observed_value, tuple):
            common = min(len(planned_value), len(observed_value))
            first_index = next(
                (
                    index
                    for index in range(common)
                    if planned_value[index] != observed_value[index]
                ),
                common,
            )
            planned_item = (
                planned_value[first_index]
                if first_index < len(planned_value)
                else "<missing>"
            )
            observed_item = (
                observed_value[first_index]
                if first_index < len(observed_value)
                else "<missing>"
            )
            differences.append(
                f"{name}[{first_index}]: planned={planned_item!r} "
                f"observed={observed_item!r} "
                f"lengths={len(planned_value)}/{len(observed_value)}"
            )
        else:
            differences.append(
                f"{name}: planned={planned_value!r} observed={observed_value!r}"
            )
    return "; ".join(differences) or "no field difference"


def _effective_mtp_verifier_contract(
    owners: tuple[OwnerGraphExecutionPolicy, ...] | None = None,
) -> tuple[str, str, str]:
    speculative = tuple(
        owner for owner in (owners or ()) if owner.activation == "speculative"
    )
    if owners is not None and not speculative:
        return (
            "native-backend-v1",
            "native-backend-v1:no-draft-owners",
            "native-target-baseline-v1",
        )
    fixed_query = tuple(
        owner for owner in speculative if owner.token_source == "fixed_query"
    )
    if owners is not None and fixed_query:
        if len(fixed_query) != len(speculative):
            raise RuntimeError(
                "parallel-draft Graph policy mixes incompatible owner formulas"
            )
        configuration = ",".join(
            f"{owner.owner}:q{owner.fixed_query_len}"
            for owner in sorted(fixed_query, key=lambda item: item.execution_order or 0)
        )
        return (
            "parallel-draft-query-v1",
            f"parallel-draft-query-v1:{configuration}",
            "pending-parallel-draft-product-math-v1",
        )
    token_sources = {owner.token_source for owner in speculative}
    if owners is not None and token_sources != {"step", "requests"}:
        raise RuntimeError(
            "speculative Graph owners publish an unknown execution topology: "
            f"sources={sorted(token_sources)!r}"
        )
    enabled = {
        "pseudo-prefill-v1": os.environ.get("AG2_VLLM_MTP_DCP_PSEUDO_DECODE", "0")
        == "1",
        "batched-causal-q1-v1": os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "0")
        == "1",
        "sequential-causal-q1-v1": os.environ.get(
            "AG2_VLLM_MTP_DCP_SEQUENTIAL_DECODE", "0"
        )
        == "1",
    }
    selected = [name for name, active in enabled.items() if active]
    if len(selected) > 1:
        raise RuntimeError(
            "multiple MTP verifier implementations are simultaneously enabled"
        )
    verifier = selected[0] if selected else "native-backend-v1"
    verifier_configuration = verifier
    math_contract = {
        "pseudo-prefill-v1": "rejected-pseudo-prefill-common-history-v1",
        "batched-causal-q1-v1": "pending-batched-q1-product-math-v1",
        "sequential-causal-q1-v1": "pending-sequential-q1-product-math-v1",
        "native-backend-v1": "native-target-baseline-v1",
    }[verifier]
    if verifier == "batched-causal-q1-v1":
        fixed_split = os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_FIXED_SPLIT_SIZE", "")
        disable_split = os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_DISABLE_SPLIT_KV", "0")
        workspace_mib = os.environ.get("AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB", "")
        verifier_configuration = (
            f"{verifier}:fixed_split={fixed_split}:"
            f"disable_split={disable_split}:workspace_mib={workspace_mib}"
        )
        if (fixed_split, disable_split, workspace_mib) == ("2048", "1", "96"):
            math_contract = "accepted-batched-q1-split2048-nosplit-forced-prefix-v1"
    return verifier, verifier_configuration, math_contract


def graph_execution_policy_from_managers(
    managers: tuple["CudaGraphManager", ...],
) -> GraphExecutionPolicy:
    """Build the shared policy from managers after backend initialization."""
    if not managers:
        raise RuntimeError("graph execution policy requires at least one manager")
    owners = []
    execution_order = {id(manager): index for index, manager in enumerate(managers)}
    for manager in sorted(managers, key=lambda item: item.dynamic_graph_owner):
        decode_mode = manager.cudagraph_mode.decode_mode()
        full_query_lens = (
            tuple(sorted(manager._runtime_decode_query_lens()))
            if decode_mode == CUDAGraphMode.FULL
            else ()
        )
        owners.append(
            OwnerGraphExecutionPolicy(
                owner=manager.dynamic_graph_owner,
                full_query_lens=full_query_lens,
                piecewise_mode=manager.cudagraph_mode.mixed_mode().name,
                full_max_tokens=manager.full_decode_max_tokens,
                compiled_piecewise_sizes=tuple(
                    sorted(manager.compiled_piecewise_sizes)
                ),
                capability_contract=(
                    "effective-cudagraph-manager-v2:math-lane-identity-v1:"
                    f"{manager.cudagraph_mode.name}:"
                    f"decode={decode_mode.name}:"
                    f"mixed={manager.cudagraph_mode.mixed_mode().name}:"
                    f"full_max_tokens={manager.full_decode_max_tokens}"
                ),
                piecewise_padding_contract=(
                    "exact-batched-decode-m-v1"
                    if manager.elastic_graph_token_source == "step"
                    else "power-of-two-terminal-exact-v1"
                ),
                piecewise_query_len_min_tokens=(
                    envs.AG2_VLLM_TP3_OWNER_MIN_ROWS
                    if getattr(manager, "tp3_owner_prequant", False)
                    else None
                ),
                activation=manager.elastic_graph_activation,
                token_source=manager.elastic_graph_token_source,
                fixed_query_len=manager.elastic_graph_fixed_query_len,
                execution_order=execution_order[id(manager)],
            )
        )
    (
        verifier_contract,
        verifier_configuration,
        math_contract,
    ) = _effective_mtp_verifier_contract(tuple(owners))
    return GraphExecutionPolicy(
        verifier_contract=verifier_contract,
        verifier_configuration=verifier_configuration,
        math_contract=math_contract,
        owners=tuple(owners),
    )


def uses_elastic_on_demand_graphs(vllm_config: VllmConfig) -> bool:
    return ElasticRuntimeConfig.from_vllm_config(vllm_config).enabled


def copy_aux_hidden_state(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    num_tokens: int,
    token_major: bool,
) -> None:
    """Copy either a token-major or a fixed-shape diagnostic graph output."""
    if token_major:
        destination[:num_tokens].copy_(source)
    else:
        if destination.shape != source.shape:
            raise RuntimeError(
                "Fixed-shape auxiliary CUDA Graph output changed shape: "
                f"{tuple(destination.shape)} != {tuple(source.shape)}"
            )
        destination.copy_(source)


def view_aux_hidden_state(
    tensor: torch.Tensor,
    *,
    num_tokens: int,
    token_major: bool,
) -> torch.Tensor:
    """Return the active token slice or the complete fixed-shape output."""
    return tensor[:num_tokens] if token_major else tensor


def dynamic_capture_publication_allowed(
    graph_segments: int,
    charged_bytes: int,
    granted_bytes: int,
) -> bool:
    """Validate graph existence and its incremental elastic-KV charge."""
    return graph_segments > 0 and 0 <= charged_bytes <= granted_bytes


def dynamic_capture_physical_charge(
    capture_delta_bytes: int,
    private_pool_bytes: int,
) -> int:
    """Charge both newly allocated bytes and an allocator-reused graph pool."""
    if capture_delta_bytes < 0 or private_pool_bytes < 0:
        raise ValueError("dynamic capture byte counts must be non-negative")
    return max(capture_delta_bytes, private_pool_bytes)


class AttentionState(NamedTuple):
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Describes the shape of the batch and CG mode to run; this is used to make shape
    matches between the capture and runtime."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None
    # Upper bound on per-request query length. Varlen decode graphs leave
    # uniform_token_count unset, so this is what keeps a prefill batch out of one.
    max_query_len: int | None = None
    num_active_loras: int = 0
    # Number of microbatches the batch is split into (DBO). 1 means no splitting.
    num_ubatches: int = 1
    physical_num_reqs: int | None = None
    runtime_generation: str = "static"
    semantic_decode: bool = False

    def logical_dispatch_key(self, owner: str) -> LogicalDispatchKey:
        return LogicalDispatchKey(
            owner=owner,
            mode=self.cg_mode.name,
            token_bucket=self.num_tokens,
            logical_num_reqs=self.num_reqs,
            uniform_query_len=self.uniform_token_count,
            active_loras=self.num_active_loras,
        )

    def physical_replay_key(self, owner: str) -> PhysicalReplayKey:
        physical_x = self.physical_num_reqs or self.num_reqs
        if physical_x is None:
            raise RuntimeError("physical CUDA Graph identity requires request count")
        return PhysicalReplayKey(
            logical=self.logical_dispatch_key(owner),
            physical_num_reqs=physical_x,
            generation=RuntimeGeneration(self.runtime_generation),
        )


class DynamicGraphResidency(str, Enum):
    WARM = "warm"
    QUEUED = "queued"
    HOT = "hot"
    COOLDOWN = "cooldown"


@dataclass
class DynamicGraphRetentionLedger:
    """Device-wide graph bytes not yet returned by CUDA teardown."""

    local_bytes: int = 0
    rank_safe_bytes: int = 0
    step_start_local_bytes: int = 0
    step_released_capture_state_bytes: int = 0


@dataclass
class DynamicGraphCaptureState:
    """One lazily-created capture stream shared by all dynamic graph owners."""

    context: GraphCaptureContext | None = None


@dataclass
class DynamicModelCaptureBundle:
    """Descriptor-private address-stable target output state."""

    hidden_states: torch.Tensor | None = None
    aux_hidden_states: list[torch.Tensor] = field(default_factory=list)
    aux_hidden_states_token_major: list[bool] = field(default_factory=list)
    intermediate_tensors: IntermediateTensors | None = None


@dataclass
class DynamicGraphEntry:
    descriptor: BatchExecutionDescriptor
    # PIECEWISE descriptors intentionally omit num_reqs because the compiled
    # bucket is token-sized.  FlashInfer graph metadata is still request-count
    # specific, though, so retain the physical runtime X separately for the
    # lifetime of the captured graph.
    runtime_num_reqs: int | None = None
    pinned: bool = False
    min_hits: int = 1
    state: DynamicGraphResidency = DynamicGraphResidency.WARM
    hits: int = 0
    last_used_epoch: int = 0
    cooldown_until_epoch: int = 0
    estimated_bytes: int = 0
    charged_bytes: int = 0
    local_charged_bytes: int = 0
    local_capture_state_bytes: int = 0
    local_pool_bytes: int = 0
    # Rank-min physical bytes whose private pool passed the eviction oracle.
    reclaimable_bytes: int = 0
    graph_segments: int = 0
    graph_pool: Any | None = None
    # CUDA Graphs replay the exact device addresses used during capture.
    # Keep the closure that owns attention metadata, slot mappings and model
    # inputs alive for exactly the executable's residency lifetime.
    capture_state: Callable[[CUDAGraphMode], None] | None = None
    lifetime_bundle: DynamicModelCaptureBundle | None = None
    leases: set[str] = field(default_factory=set)
    deferred_free: bool = False


def make_cudagraph_stats(
    batch_desc: BatchExecutionDescriptor, num_tokens: int
) -> CUDAGraphStat:
    return CUDAGraphStat(
        num_unpadded_tokens=num_tokens,
        num_padded_tokens=batch_desc.num_tokens,
        num_paddings=batch_desc.num_tokens - num_tokens,
        runtime_mode=str(batch_desc.cg_mode),
    )


class CreateForwardFn(Protocol):
    """Factory that prepares inputs (OUTSIDE the graph) and returns a
    forward_fn. Called with warmup=True for the warmup pass and warmup=False
    for the captured pass."""

    def __call__(
        self,
        desc: BatchExecutionDescriptor,
        warmup: bool,
    ) -> Callable[[CUDAGraphMode], None]: ...


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    num_active_loras: int,
    max_query_len: int | None = None,
    num_ubatches: int = 1,
) -> bool:
    # desc.uniform_token_count=None (PIECEWISE) can handle any uniform_token_count
    # desc.num_reqs=None means no request padding needed (PIECEWISE)
    # desc.max_query_len=None means the graph does not constrain query length; a
    # caller that does not track max_query_len must not match one that does
    # A graph captured for N microbatches can only serve a batch split N ways.
    exact_multitoken_full_requests = not (
        desc.cg_mode == CUDAGraphMode.FULL
        and desc.uniform_token_count is not None
        and desc.uniform_token_count > 1
    ) or (desc.num_reqs == num_reqs and desc.num_tokens == num_tokens)
    return (
        (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (
            desc.max_query_len is None
            or (max_query_len is not None and desc.max_query_len >= max_query_len)
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
        and desc.num_active_loras == num_active_loras
        and exact_multitoken_full_requests
        and desc.num_ubatches == num_ubatches
    )


def has_compiled_submodule(model: nn.Module) -> bool:
    """Whether any submodule is an active @support_torch_compile module."""
    return any(
        isinstance(m, TorchCompileWithNoGuardsWrapper)
        and not getattr(m, "do_not_compile", True)
        for m in model.modules()
    )


def get_uniform_token_count(
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
) -> int | None:
    """
    Return the uniform token count if batch is uniform, else None.
    A batch is uniform if all requests have the same number of tokens.
    """
    if (max_query_len == num_tokens // num_reqs) and (
        num_tokens == max_query_len * num_reqs
    ):
        return max_query_len
    return None


class DynamicGraphWorkingSet:
    """Aggregate one step-scoped KV loan across graph owners.

    Managers keep independent descriptors and private graph pools, while the
    elastic KV arena exposes one external-memory counter. Missing owners are
    captured sequentially before execution from the same scheduler grant;
    already-HOT owners remain charged throughout the transaction.
    """

    def __init__(self, managers: tuple["CudaGraphManager", ...]):
        self.managers = managers
        self._planned_capture_order: tuple[PhysicalReplayKey, ...] = ()
        self._rank_vote_local: torch.Tensor | None = None
        self._rank_vote_gathered: torch.Tensor | None = None
        self._rank_consensus_calls: dict[str, int] = {}
        self._rank_consensus_ns: dict[str, int] = {}
        self._rank_consensus_reuses: dict[str, int] = {}
        self._accepted_decode_epoch: str | None = None
        self._accepted_decode_observers: tuple[tuple[str, str | None], ...] | None = (
            None
        )
        self._active_step_plan_fingerprint: str | None = None
        self._active_step_decode_epoch: str | None = None
        self._active_step_decode_observers: tuple[tuple[str, str | None], ...] = ()
        self._active_step_reuses_consensus = False
        if managers and all(
            isinstance(manager, CudaGraphManager) for manager in managers
        ):
            ledger = managers[0]._dynamic_retention_ledger
            for index, manager in enumerate(managers):
                manager_ledger = manager._dynamic_retention_ledger
                if manager_ledger is not ledger and (
                    manager_ledger.local_bytes or manager_ledger.rank_safe_bytes
                ):
                    raise RuntimeError(
                        "CUDA Graph managers acquired independent retained-memory "
                        "state before working-set registration"
                    )
                manager._dynamic_retention_ledger = ledger
                manager._owns_dynamic_retention_ledger = index == 0

            capture_states: list[DynamicGraphCaptureState] = []
            for manager in managers:
                state = getattr(manager, "_dynamic_capture_state", None)
                if state is None:
                    state = DynamicGraphCaptureState()
                    manager._dynamic_capture_state = state
                if all(state is not known for known in capture_states):
                    capture_states.append(state)
            initialized_states = [
                state for state in capture_states if state.context is not None
            ]
            if initialized_states:
                capture_state = initialized_states[0]
                if any(
                    state.context is not capture_state.context
                    for state in initialized_states[1:]
                ):
                    raise RuntimeError(
                        "CUDA Graph managers acquired independent capture streams "
                        "before working-set registration"
                    )
            else:
                capture_state = capture_states[0]
            for manager in managers:
                manager._dynamic_capture_state = capture_state

    @property
    def resident_bytes(self) -> int:
        return sum(manager.dynamic_resident_bytes for manager in self.managers)

    @property
    def active_graph_bytes(self) -> int:
        """Return reclaimable HOT graph bytes, excluding physical floors."""
        return sum(manager.dynamic_active_graph_bytes for manager in self.managers)

    def transition_floor_upper_bound_bytes(self) -> int:
        """Bound the floor left by evicting every currently HOT entry.

        Dynamic eviction is accepted only when driver-free bytes recover at
        least ``local_pool_bytes - 1 MiB``.  Therefore the unreclaimed part of
        each entry can be bounded before eviction from its already measured
        charge and private-pool size.  Publishing the TP maximum lets the
        scheduler price the next changed-key capture without a guessed reserve
        or the overly conservative sum of both complete graph residencies.
        """
        if not self.managers:
            return 0
        ledger = self.managers[0]._dynamic_retention_ledger
        local_upper_bound = ledger.local_bytes
        reclaim_tolerance = 1 << 20
        for manager in self.managers:
            for entry in manager._dynamic_graph_entries.values():
                if entry.state != DynamicGraphResidency.HOT:
                    continue
                guaranteed_reclaim = max(
                    0,
                    entry.local_pool_bytes - reclaim_tolerance,
                )
                local_upper_bound += max(
                    0,
                    entry.local_charged_bytes - guaranteed_reclaim,
                )
        upper_bound = torch.tensor(
            local_upper_bound,
            dtype=torch.int64,
            device=self.managers[0].device,
        )
        if self.managers[0].tp_size > 1:
            torch.distributed.all_reduce(
                upper_bound,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
        return int(upper_bound.item())

    @property
    def hot_request_counts(self) -> frozenset[int]:
        """Request counts whose FlashInfer graph metadata is still live."""
        return frozenset(
            runtime_num_reqs
            for manager in self.managers
            for entry in manager._dynamic_graph_entries.values()
            if entry.state == DynamicGraphResidency.HOT
            and (
                runtime_num_reqs := (
                    entry.runtime_num_reqs or entry.descriptor.num_reqs
                )
            )
            is not None
        )

    @property
    def hot_token_counts(self) -> frozenset[int]:
        """Token carriers whose FlashInfer decode metadata is still live."""
        return frozenset(
            entry.descriptor.num_tokens
            for manager in self.managers
            for entry in manager._dynamic_graph_entries.values()
            if entry.state == DynamicGraphResidency.HOT
        )

    def begin_step(self) -> None:
        self._planned_capture_order = ()
        if self.managers:
            ledger = self.managers[0]._dynamic_retention_ledger
            ledger.step_start_local_bytes = ledger.local_bytes
            ledger.step_released_capture_state_bytes = 0
        for manager in self.managers:
            manager.begin_dynamic_step()

    def reconcile_retained_cleanup(
        self,
        local_reclaimed_bytes: int,
        *,
        force_rank_consensus: bool = False,
    ) -> int:
        """Retire only this step's retention with graph-owned cleanup."""
        if not self.managers:
            return 0
        return self.managers[0]._reconcile_dynamic_retained_cleanup(
            local_reclaimed_bytes,
            force_rank_consensus=force_rank_consensus,
        )

    def clear_idle_retention_after_physical_reconcile(self) -> None:
        """Retire teardown estimates after X0 measured the real VMM floor.

        Pinned HOT graph residency is accounted independently in
        ``active_graph_bytes`` and may legally survive an administrative X0.
        The retention ledger contains only teardown estimates, so clearing it
        after rank-safe physical reconciliation does not release or hide those
        live executables.
        """
        if not self.managers:
            return
        ledger = self.managers[0]._dynamic_retention_ledger
        ledger.local_bytes = 0
        ledger.rank_safe_bytes = 0
        ledger.step_start_local_bytes = 0
        ledger.step_released_capture_state_bytes = 0

    def pending_managers(self) -> tuple["CudaGraphManager", ...]:
        pending = tuple(
            manager
            for manager in self.managers
            if manager.has_pending_dynamic_capture()
        )
        if not self._planned_capture_order:
            return pending
        by_key = {
            manager._dynamic_pending.physical_replay_key(
                manager.dynamic_graph_owner
            ): manager
            for manager in pending
            if manager._dynamic_pending is not None
        }
        if set(by_key) != set(self._planned_capture_order):
            raise RuntimeError(
                "worker pending CUDA Graph set differs from immutable capture "
                f"order: worker={tuple(by_key)!r} "
                f"plan={self._planned_capture_order!r}"
            )
        return tuple(by_key[key] for key in self._planned_capture_order)

    def validate_plan(self, plan: ElasticStepPlan) -> None:
        """Validate worker-local residency without changing Graph state.

        This method deliberately resolves every owner and every operation that
        can fail synchronously before a lease, eviction, queue transition or KV
        change.  The caller publishes any resulting error in the all-rank vote
        before entering :meth:`begin_step`.
        """
        if plan.kind == ElasticPlanKind.DEFER:
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: a deferred elastic plan "
                "cannot reach a worker"
            )
        owner_names = tuple(manager.dynamic_graph_owner for manager in self.managers)
        if len(set(owner_names)) != len(owner_names):
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: worker Graph owners are not "
                f"unique: owners={owner_names!r}"
            )
        managers = dict(zip(owner_names, self.managers, strict=True))

        key_groups = {
            "physical": plan.physical_keys,
            "HOT": plan.hot_hits,
            "COLD": plan.cold_misses,
            "protected": plan.protected_keys,
            "victim": plan.victim_keys,
            "capture-order": plan.capture_order,
        }
        for label, keys in key_groups.items():
            if len(set(keys)) != len(keys):
                raise ElasticExecutionPlanMismatch(
                    "ELASTIC_EXECUTION_PLAN_MISMATCH: worker plan contains "
                    f"duplicate {label} keys"
                )
            for key in keys:
                if key.generation != plan.generation:
                    raise ElasticExecutionPlanMismatch(
                        "ELASTIC_EXECUTION_PLAN_MISMATCH: worker plan contains "
                        f"a stale {label} key: key={key.identity}"
                    )
                if key.logical.owner not in managers:
                    raise ElasticExecutionPlanMismatch(
                        "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic plan references "
                        f"unknown {label} owner {key.logical.owner!r}"
                    )

        classified_keys = set(plan.hot_hits).union(plan.cold_misses)
        if classified_keys != set(plan.physical_keys):
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: physical keys are not exactly "
                "partitioned into HOT hits and COLD misses"
            )
        if set(plan.capture_order) != set(plan.cold_misses):
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: immutable capture order differs "
                f"from COLD misses: capture={plan.capture_order!r} "
                f"cold={plan.cold_misses!r}"
            )
        if set(plan.victim_keys).intersection(plan.physical_keys):
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: current physical key was also "
                "selected as an eviction victim"
            )
        if set(plan.victim_keys).intersection(plan.protected_keys):
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: protected physical key was also "
                "selected as an eviction victim"
            )

        # Protected keys belong to a scheduler-owned multi-step physical epoch.
        # Validate the future lease without making them dispatch candidates.
        for key in plan.protected_keys:
            managers[key.logical.owner].validate_physical_key_lease(
                key, plan.transaction_id
            )
        for key in plan.victim_keys:
            managers[key.logical.owner].validate_physical_key_eviction(
                key,
                transaction_id=plan.transaction_id,
                reason="elastic_admission_plan",
                administrative=plan.kind
                in {
                    ElasticPlanKind.RECLAIM,
                    ElasticPlanKind.PRESSURE_RECLAIM,
                },
            )
        for key in plan.physical_keys:
            manager = managers[key.logical.owner]
            hot = manager.is_physical_key_hot(key)
            if key in plan.hot_hits and not hot:
                raise ElasticExecutionPlanMismatch(
                    "ELASTIC_EXECUTION_PLAN_MISMATCH: scheduler planned a HOT "
                    "hit absent from worker residency: "
                    f"owner={key.logical.owner} key={key.identity}"
                )
            if key in plan.cold_misses and hot:
                raise ElasticExecutionPlanMismatch(
                    "ELASTIC_EXECUTION_PLAN_MISMATCH: scheduler planned a COLD "
                    "miss already HOT on worker: "
                    f"owner={key.logical.owner} key={key.identity}"
                )
            if plan.kind == ElasticPlanKind.USER and not hot:
                raise ElasticExecutionPlanMismatch(
                    "ELASTIC_EXECUTION_PLAN_MISMATCH: USER elastic plan cannot "
                    "trigger capture"
                )
        dispatch_keys = tuple(
            dispatch.physical_key
            for dispatch in getattr(plan, "current_dispatch", ())
            if dispatch.representation == DispatchRepresentation.HOT_GRAPH
        )
        keys_to_queue = (
            dispatch_keys
            if getattr(plan, "execution_manifest", None) is not None
            else plan.physical_keys
        )
        pending_by_owner: dict[str, PhysicalReplayKey] = {}
        for key in keys_to_queue:
            if key is None:
                raise ElasticExecutionPlanMismatch(
                    "ELASTIC_EXECUTION_PLAN_MISMATCH: HOT Graph dispatch omitted "
                    "its physical key"
                )
            manager = managers[key.logical.owner]
            manager.validate_physical_key_queue(key)
            if not manager.is_physical_key_hot(key):
                prior = pending_by_owner.get(key.logical.owner)
                if prior is not None and prior != key:
                    raise ElasticExecutionPlanMismatch(
                        "ELASTIC_EXECUTION_PLAN_MISMATCH: one Graph owner received "
                        "multiple simultaneous COLD keys"
                    )
                pending_by_owner[key.logical.owner] = key

        predicted_pending = tuple(pending_by_owner.values())
        if set(predicted_pending) != set(plan.capture_order):
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: worker capture set differs "
                f"from immutable elastic plan: worker={predicted_pending!r} "
                f"plan={plan.capture_order!r}"
            )

    def _apply_validated_plan(self, plan: ElasticStepPlan) -> None:
        """Apply a plan whose complete local preconditions already passed."""
        managers = {manager.dynamic_graph_owner: manager for manager in self.managers}
        for key in plan.protected_keys:
            managers[key.logical.owner].acquire_physical_key_lease(
                key, plan.transaction_id
            )
        local_eviction_error: Exception | None = None
        for key in plan.victim_keys:
            if local_eviction_error is not None:
                break
            try:
                managers[key.logical.owner].evict_physical_key(
                    key,
                    transaction_id=plan.transaction_id,
                    reason="elastic_admission_plan",
                    administrative=plan.kind
                    in {
                        ElasticPlanKind.RECLAIM,
                        ElasticPlanKind.PRESSURE_RECLAIM,
                    },
                    defer_rank_consensus=True,
                )
            except Exception as error:
                local_eviction_error = error
        if plan.victim_keys and self.managers:
            # A common scheduler key does not imply identical local CUDA graph
            # segments or teardown timing. Publish once after the whole local
            # victim set instead of joining the TP device communicator from
            # inside each physical eviction.
            self.managers[0]._publish_deferred_eviction_consensus(local_eviction_error)
        dispatch_keys = tuple(
            dispatch.physical_key
            for dispatch in getattr(plan, "current_dispatch", ())
            if dispatch.representation == DispatchRepresentation.HOT_GRAPH
        )
        keys_to_queue = (
            dispatch_keys
            if getattr(plan, "execution_manifest", None) is not None
            else plan.physical_keys
        )
        for key in keys_to_queue:
            assert key is not None
            managers[key.logical.owner].queue_physical_key(key)
        self._planned_capture_order = plan.capture_order

    def apply_plan(self, plan: ElasticStepPlan) -> None:
        """Validate then execute a plan for non-distributed/direct callers."""
        self.validate_plan(plan)
        self._apply_validated_plan(plan)

    def validate_execution_manifest(
        self,
        plan: ElasticStepPlan,
        *,
        step_key: tuple[int, int, int, int, int] | None,
        request_ids: tuple[str, ...],
        per_request_query_lens: tuple[int, ...],
        per_request_is_prefilling: tuple[bool, ...],
        scheduled_draft_rows: tuple[int, ...],
        scheduled_encoder_inputs: Mapping[str, Sequence[int]] | None = None,
        requested_output_k: int,
        executed_drafter_k: int,
        phase: str | None,
        max_num_batched_tokens: int,
        active_loras: int = 0,
    ) -> None:
        """Validate exact current work before leases, eviction, capture or KV."""
        if plan.execution_manifest is None:
            if per_request_query_lens:
                raise ElasticExecutionPlanMismatch(
                    "ELASTIC_EXECUTION_PLAN_MISMATCH: USER plan omitted manifest"
                )
            return
        if step_key is None:
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: manifest omitted step key"
            )
        if phase is None:
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: manifest omitted execution phase"
            )
        policy = graph_execution_policy_from_managers(self.managers)
        planned_physical_keys = tuple(
            dispatch.physical_key
            for dispatch in plan.current_dispatch
            if dispatch.representation == DispatchRepresentation.HOT_GRAPH
            and dispatch.physical_key is not None
        )
        try:
            observed_manifest, observed_dispatch = build_execution_manifest(
                step_key=step_key,
                request_ids=request_ids,
                per_request_query_lens=per_request_query_lens,
                per_request_is_prefilling=per_request_is_prefilling,
                scheduled_draft_rows=scheduled_draft_rows,
                scheduled_encoder_inputs=scheduled_encoder_inputs,
                requested_output_k=requested_output_k,
                executed_drafter_k=executed_drafter_k,
                phase=phase,
                generation=plan.generation,
                policy=policy,
                max_num_batched_tokens=max_num_batched_tokens,
                active_loras=active_loras,
                physical_keys=planned_physical_keys,
            )
        except (ElasticGraphError, ValueError) as exc:
            raise ElasticExecutionPlanMismatch(
                f"ELASTIC_EXECUTION_PLAN_MISMATCH: {exc}"
            ) from exc
        if observed_manifest != plan.execution_manifest:
            field_difference = _execution_manifest_difference(
                plan.execution_manifest, observed_manifest
            )
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: scheduler and worker manifests "
                f"differ: planned={plan.execution_manifest.fingerprint} "
                f"observed={observed_manifest.fingerprint} "
                f"fields={field_difference}"
            )
        if observed_dispatch != plan.current_dispatch:
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: scheduler and worker dispatch "
                f"differ: planned={plan.current_dispatch!r} "
                f"observed={observed_dispatch!r}"
            )

    def require_rank_consensus(
        self,
        plan: ElasticStepPlan,
        *,
        validation_error: str | None = None,
        observer_fingerprint: str | None = None,
        phase: str = "pre_mutation",
    ) -> str:
        """Run and account one fixed-size all-rank plan vote."""
        started_ns = time.perf_counter_ns()
        try:
            self._require_rank_consensus_impl(
                plan,
                validation_error=validation_error,
                observer_fingerprint=f"{phase}:{observer_fingerprint or ''}",
            )
            return plan.fingerprint
        finally:
            elapsed_ns = time.perf_counter_ns() - started_ns
            calls = self._rank_consensus_calls.get(phase, 0) + 1
            total_ns = self._rank_consensus_ns.get(phase, 0) + elapsed_ns
            self._rank_consensus_calls[phase] = calls
            self._rank_consensus_ns[phase] = total_ns
            if calls == 1 or calls % 64 == 0:
                logger.info(
                    "AG2 elastic rank-consensus timing: phase=%s calls=%d "
                    "total_ms=%.3f mean_us=%.3f last_us=%.3f",
                    phase,
                    calls,
                    total_ns / 1e6,
                    total_ns / calls / 1e3,
                    elapsed_ns / 1e3,
                )

    def rank_consensus_timing_snapshot(self) -> dict[str, tuple[int, int]]:
        """Return phase -> (calls, total_ns) for matched perf receipts."""
        return {
            phase: (calls, self._rank_consensus_ns.get(phase, 0))
            for phase, calls in self._rank_consensus_calls.items()
        }

    def rank_consensus_reuse_snapshot(self) -> dict[str, int]:
        """Return phase -> skipped collective count for perf receipts."""
        return dict(self._rank_consensus_reuses)

    @staticmethod
    def _reusable_decode_epoch(plan: ElasticStepPlan) -> str | None:
        """Return a stable epoch only for mutation-free text decode."""
        if not plan.reusable_decode_consensus_epoch:
            return None
        return plan.execution_epoch_fingerprint

    def _invalidate_decode_consensus_epoch(self) -> None:
        self._accepted_decode_epoch = None
        self._accepted_decode_observers = None
        self._active_step_decode_observers = ()

    def _record_consensus_reuse(self, phase: str) -> None:
        reuses = self._rank_consensus_reuses.get(phase, 0) + 1
        self._rank_consensus_reuses[phase] = reuses
        if reuses == 1 or reuses % 64 == 0:
            logger.info(
                "AG2 elastic rank-consensus reuse: phase=%s skips=%d",
                phase,
                reuses,
            )

    def _raise_reused_epoch_drift(self, detail: str) -> None:
        self._invalidate_decode_consensus_epoch()
        raise RuntimeError(
            "ELASTIC_STEADY_EPOCH_LOCAL_DRIFT: an already unanimous decode "
            f"epoch diverged locally; runtime restart is required: {detail}"
        )

    def _raise_active_step_plan_drift(self, detail: str) -> None:
        self._invalidate_decode_consensus_epoch()
        raise RuntimeError(
            "ELASTIC_ACTIVE_STEP_PLAN_DRIFT: the post-mutation boundary no "
            f"longer matches its admitted plan; runtime restart is required: {detail}"
        )

    def _require_rank_consensus_impl(
        self,
        plan: ElasticStepPlan,
        *,
        validation_error: str | None = None,
        observer_fingerprint: str | None = None,
    ) -> str:
        """Make every rank accept or reject before any CUDA/KV mutation.

        Callers decide whether a previously unanimous steady-decode epoch can
        reuse its boundary. Whenever this method is entered, the compact vote
        remains mandatory and converges every rank-local error.
        """
        local_error = validation_error
        try:
            generations = {
                RuntimeGeneration(manager.runtime_generation)
                for manager in self.managers
            }
            if self.managers and generations != {plan.generation}:
                generation_error = (
                    "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic plan generation "
                    "differs from worker runtime"
                )
                local_error = (
                    generation_error
                    if local_error is None
                    else f"{local_error}; {generation_error}"
                )
        except Exception as error:
            generation_error = (
                "ELASTIC_EXECUTION_PLAN_MISMATCH: invalid worker runtime "
                f"generation: {type(error).__name__}: {error}"
            )
            local_error = (
                generation_error
                if local_error is None
                else f"{local_error}; {generation_error}"
            )
        try:
            local_fingerprint = plan.fingerprint
            if observer_fingerprint is not None:
                local_fingerprint = hashlib.sha256(
                    f"{local_fingerprint}:{observer_fingerprint}".encode()
                ).hexdigest()
        except Exception as error:
            fingerprint_error = (
                "ELASTIC_EXECUTION_PLAN_MISMATCH: invalid local plan "
                f"fingerprint: {type(error).__name__}: {error}"
            )
            local_error = (
                fingerprint_error
                if local_error is None
                else f"{local_error}; {fingerprint_error}"
            )
            local_fingerprint = f"invalid:{type(error).__name__}"

        if not self.managers or self.managers[0].tp_size == 1:
            if local_error is not None:
                raise ElasticExecutionPlanMismatch(local_error)
            return local_fingerprint

        try:
            fingerprint_bytes = bytes.fromhex(local_fingerprint)
            if len(fingerprint_bytes) != 32:
                raise ValueError("plan fingerprint is not a SHA-256 digest")
        except (TypeError, ValueError) as error:
            fingerprint_error = (
                "ELASTIC_EXECUTION_PLAN_MISMATCH: invalid local plan fingerprint "
                f"encoding: {type(error).__name__}: {error}"
            )
            local_error = (
                fingerprint_error
                if local_error is None
                else f"{local_error}; {fingerprint_error}"
            )
            fingerprint_bytes = bytes(32)

        tp_size = self.managers[0].tp_size
        vote_width = 5
        if self._rank_vote_local is None:
            self._rank_vote_local = torch.empty(
                vote_width, dtype=torch.int64, device="cpu"
            )
        if (
            self._rank_vote_gathered is None
            or self._rank_vote_gathered.numel() != tp_size * vote_width
        ):
            self._rank_vote_gathered = torch.empty(
                tp_size * vote_width, dtype=torch.int64, device="cpu"
            )
        local_vote = self._rank_vote_local
        gathered_vote = self._rank_vote_gathered
        for index in range(4):
            local_vote[index] = int.from_bytes(
                fingerprint_bytes[index * 8 : (index + 1) * 8],
                byteorder="big",
                signed=True,
            )
        local_vote[4] = int(local_error is not None)
        torch.distributed.all_gather_single(
            gathered_vote,
            local_vote,
            group=get_tp_group().cpu_group,
        )
        rank_votes = gathered_vote.view(tp_size, vote_width)
        fingerprint_mismatch = any(
            not torch.equal(rank_votes[0, :4], rank_votes[rank, :4])
            for rank in range(1, tp_size)
        )
        rank_rejected = any(bool(rank_votes[rank, 4]) for rank in range(tp_size))
        if not fingerprint_mismatch and not rank_rejected:
            return local_fingerprint

        # The fixed-size vote keeps the accepted decode path free of Python
        # serialization.  All ranks reach this diagnostic gather only after the
        # shared tensor result says the step must be rejected.
        gathered_diagnostics: list[tuple[str, str | None] | None] = [
            None for _ in range(tp_size)
        ]
        torch.distributed.all_gather_object(
            gathered_diagnostics,
            (local_fingerprint, local_error),
            group=get_tp_group().cpu_group,
        )
        if any(item is None for item in gathered_diagnostics):
            raise RuntimeError("elastic rank consensus returned empty diagnostics")
        rank_results = tuple(item for item in gathered_diagnostics if item is not None)
        fingerprints = {item[0] for item in rank_results}
        if len(fingerprints) != 1:
            raise ElasticExecutionPlanMismatch(
                "ELASTIC_EXECUTION_PLAN_MISMATCH: elastic graph plan "
                "fingerprint differs by rank"
            )
        errors = tuple(
            (rank, item[1])
            for rank, item in enumerate(rank_results)
            if item[1] is not None
        )
        raise ElasticExecutionPlanMismatch(
            "ELASTIC_EXECUTION_PLAN_MISMATCH: rank validation rejected "
            f"before mutation: errors={errors!r}"
        )

    def begin_admitted_step(
        self,
        plan: ElasticStepPlan,
        *,
        validation_error: str | None = None,
    ) -> None:
        """Validate, vote and only then cross the first mutation boundary."""
        local_error = validation_error
        try:
            self.validate_plan(plan)
        except Exception as error:
            plan_error = (
                "ELASTIC_EXECUTION_PLAN_MISMATCH: local worker plan validation "
                f"failed: {type(error).__name__}: {error}"
            )
            local_error = (
                plan_error if local_error is None else f"{local_error}; {plan_error}"
            )
        decode_epoch = self._reusable_decode_epoch(plan)
        reuse_consensus = plan.reuse_rank_consensus
        if reuse_consensus:
            if decode_epoch != self._accepted_decode_epoch:
                self._raise_reused_epoch_drift(
                    "scheduler requested reuse without the matching locally "
                    f"accepted epoch: scheduler={decode_epoch!r} "
                    f"worker={self._accepted_decode_epoch!r}"
                )
            if local_error is not None:
                self._raise_reused_epoch_drift(local_error)
            self._record_consensus_reuse("pre_mutation")
        else:
            # Crossing any non-identical or non-decode boundary invalidates the
            # old proof before the new all-rank vote.
            self._invalidate_decode_consensus_epoch()
            self.require_rank_consensus(plan, validation_error=local_error)
        self._active_step_plan_fingerprint = plan.fingerprint
        self._active_step_decode_epoch = decode_epoch
        self._active_step_decode_observers = ()
        self._active_step_reuses_consensus = reuse_consensus
        self.begin_step()
        self._apply_validated_plan(plan)

    def require_post_materialization_consensus(
        self,
        plan: ElasticStepPlan,
        *,
        validation_error: str | None = None,
        observer_fingerprint: str | None = None,
        phase: str = "post_materialization",
    ) -> str:
        """Converge rank-local input observers before model collectives.

        This vote is intentionally independent of the pre-mutation vote. Input
        materialization can expose rank-local request-state drift only after
        the admitted plan has already crossed its mutation boundary.
        """
        if plan.fingerprint != self._active_step_plan_fingerprint:
            self._raise_active_step_plan_drift(
                "post-materialization plan differs from the admitted step"
            )
        observer = (phase, observer_fingerprint)
        if self._active_step_reuses_consensus:
            if validation_error is not None:
                self._raise_reused_epoch_drift(validation_error)
            accepted = self._accepted_decode_observers or ()
            index = len(self._active_step_decode_observers)
            if index >= len(accepted) or observer != accepted[index]:
                self._raise_reused_epoch_drift(
                    "post-materialization observer differs from the accepted "
                    f"boundary: index={index} accepted={accepted!r} "
                    f"observed={observer!r}"
                )
            self._active_step_decode_observers += (observer,)
            self._record_consensus_reuse(phase)
            return plan.fingerprint
        try:
            result = self.require_rank_consensus(
                plan,
                validation_error=validation_error,
                observer_fingerprint=observer_fingerprint,
                phase=phase,
            )
            if self._active_step_decode_epoch is not None:
                # Publish only after model execution reaches finish_step().
                self._active_step_decode_observers += (observer,)
            return result
        except ElasticExecutionPlanMismatch as error:
            self._invalidate_decode_consensus_epoch()
            raise RuntimeError(
                "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH: all-rank input "
                "materialization vote rejected before model collectives; "
                "rank diagnostics are attached as the chained cause"
            ) from error

    def validate_post_materialization_completion(self) -> None:
        """Reject an incomplete reused phase sequence before model forward."""
        if (
            self._active_step_reuses_consensus
            and self._accepted_decode_epoch is not None
            and self._active_step_decode_observers != self._accepted_decode_observers
        ):
            self._raise_reused_epoch_drift(
                "post-materialization observer sequence is incomplete: "
                f"accepted={self._accepted_decode_observers!r} "
                f"observed={self._active_step_decode_observers!r}"
            )

    def discard_failed_capture(self, plan: ElasticStepPlan) -> None:
        """Remove only destinations created by a failed capture transaction."""
        managers = {manager.dynamic_graph_owner: manager for manager in self.managers}
        for key in reversed(plan.cold_misses):
            manager = managers.get(key.logical.owner)
            if manager is not None:
                manager.discard_failed_physical_key(
                    key, transaction_id=plan.transaction_id
                )

    def prepare_manager_capture(
        self,
        manager: "CudaGraphManager",
        granted_bytes: int,
    ) -> bool:
        other_resident = self.resident_bytes - manager.dynamic_resident_bytes
        return (
            granted_bytes >= other_resident
            and manager.prepare_pending_dynamic_capture(granted_bytes - other_resident)
        )

    def finish_step(self) -> int:
        self.validate_post_materialization_completion()
        for manager in self.managers:
            manager.finish_dynamic_step()
        if (
            not self._active_step_reuses_consensus
            and self._active_step_decode_epoch is not None
            and self._active_step_decode_observers
        ):
            self._accepted_decode_epoch = self._active_step_decode_epoch
            self._accepted_decode_observers = self._active_step_decode_observers
        self._planned_capture_order = ()
        self._active_step_plan_fingerprint = None
        self._active_step_decode_epoch = None
        self._active_step_decode_observers = ()
        self._active_step_reuses_consensus = False
        return self.resident_bytes

    def acquire_leases(self, transaction_id: str) -> None:
        for manager in self.managers:
            manager.acquire_step_leases(transaction_id)

    def release_leases(self, transaction_id: str) -> None:
        for manager in self.managers:
            manager.release_transaction_leases(transaction_id)

    def residency_receipt(
        self,
        *,
        transaction_id: str | None,
        resident_bytes: int,
        floor_bytes: int,
        transition_floor_bytes: int,
        peak_bytes: int,
        cublas_workspace_bytes: int,
    ) -> ElasticResidencyReceipt:
        entries = []
        for manager in self.managers:
            for entry in manager._dynamic_graph_entries.values():
                if entry.state != DynamicGraphResidency.HOT:
                    continue
                key = entry.descriptor.physical_replay_key(manager.dynamic_graph_owner)
                # Victims remain physically resident only as a transition
                # lifetime fence.  Publishing them as selectable HOT would
                # let scheduler policy reuse a logically retired carrier.
                entries.append(
                    ElasticResidencyEntry(
                        key=key,
                        pinned=entry.pinned,
                        resident_bytes=entry.charged_bytes,
                        local_pool_bytes=entry.local_pool_bytes,
                        # Physical pool proof remains manager-owned for a
                        # later administrative teardown. Pinning prevents the
                        # scheduler from consuming the proof until an explicit
                        # idle rebuild; an active lease cannot cross receipt
                        # publication and therefore suppresses the proof.
                        reclaimable_bytes=(
                            0 if entry.leases else entry.reclaimable_bytes
                        ),
                        lease_ids=tuple(sorted(entry.leases)),
                    )
                )
        generations = {manager.runtime_generation for manager in self.managers}
        if len(generations) != 1:
            raise RuntimeError("elastic residency publication mixed generations")
        return ElasticResidencyReceipt(
            generation=RuntimeGeneration(next(iter(generations))),
            transaction_id=transaction_id,
            resident_bytes=resident_bytes,
            floor_bytes=floor_bytes,
            transition_floor_bytes=transition_floor_bytes,
            peak_bytes=max(peak_bytes, resident_bytes, transition_floor_bytes),
            cublas_workspace_bytes=cublas_workspace_bytes,
            entries=tuple(sorted(entries, key=lambda item: item.key.identity)),
        )

    def rebind_runtime_generation(self, generation: RuntimeGeneration) -> None:
        """Bind deferred graph metadata to the scheduler's post-KV epoch.

        Workers construct graph managers before KV profiling mutates the shared
        config.  The scheduler is created afterwards, so its generation is the
        authoritative identity for every subsequently admitted transaction.
        Rebinding is deliberately legal only while residency is still empty.
        """
        self._invalidate_decode_consensus_epoch()
        for manager in self.managers:
            manager.rebind_runtime_generation(generation.value)

    def finish_idle_step(self) -> int:
        """Finish an X0 step and return transient allocator cache to KV.

        Sampling runs after model execution and can leave free allocator
        segments behind after the final graph has already been evicted.  An
        idle step is the only safe point where no CUDA producer can still
        consume those segments, so release them before measuring the physical
        external-memory floor.
        """
        self.finish_step()
        self._invalidate_decode_consensus_epoch()
        return self._release_idle_allocator_cache()

    def prepare_idle_reclaim_before_post_consensus(self) -> int:
        """Settle request-free owners without ending the admitted transaction.

        Administrative X0 must finish and evict Graph owners before expanding
        KV, but its post-state all-rank vote still belongs to the same admitted
        plan. Keep that active plan identity until the ordinary final
        ``finish_step`` while invalidating any previously reusable decode
        boundary.
        """
        if (
            self._active_step_plan_fingerprint is None
            or self._active_step_decode_epoch is not None
            or self._active_step_reuses_consensus
        ):
            raise RuntimeError(
                "pre-consensus idle reclaim requires an active non-decode step"
            )
        for manager in self.managers:
            manager.finish_dynamic_step()
        self._invalidate_decode_consensus_epoch()
        return self._release_idle_allocator_cache()

    def _release_idle_allocator_cache(self) -> int:
        """Release allocator cache after all current Graph owners are settled."""
        if not self.managers:
            return 0
        device = self.managers[0].device
        torch.accelerator.synchronize(device)
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize(device)
        return self.resident_bytes

    def evict_unpinned_for_idle(self, transaction_id: str) -> int:
        """Administratively destroy every evictable HOT entry for X0.

        Ordinary descriptor completion intentionally preserves the cache.
        Calibration cold epochs need an explicit idle boundary that removes
        only non-pinned executables. Reuse the fail-closed physical reclaim
        primitive in one stable rank order.
        """
        if not transaction_id:
            raise RuntimeError("administrative X0 eviction requires transaction id")
        victims: list[tuple[CudaGraphManager, DynamicGraphEntry]] = []
        for manager in self.managers:
            victims.extend(
                (manager, entry)
                for entry in manager._dynamic_graph_entries.values()
                if entry.state == DynamicGraphResidency.HOT and not entry.pinned
            )
        victims.sort(
            key=lambda item: repr(
                item[1]
                .descriptor.physical_replay_key(item[0].dynamic_graph_owner)
                .identity
            )
        )
        charged = sum(entry.charged_bytes for _manager, entry in victims)
        local_error: Exception | None = None
        for manager, entry in victims:
            if local_error is not None:
                break
            try:
                manager._evict_dynamic_entry(
                    entry,
                    transaction_id=transaction_id,
                    reason="administrative_idle_x0",
                    defer_rank_consensus=True,
                )
            except Exception as error:
                local_error = error
        # Victim sets are physical and may legitimately differ by rank. Never
        # put a TP collective inside that rank-local loop: X32 left rank0 with
        # one extra victim and collided its eviction vote with peers' next KV
        # vote. All managers share one device retention ledger, so publish it
        # exactly once after every local victim has been processed.
        if self.managers:
            self.managers[0]._publish_deferred_eviction_consensus(local_error)
        return charged

    def prune_pinned_full_above(
        self, max_x: int, transaction_id: str
    ) -> tuple[int, int, int]:
        """Calibration-only teardown of unreachable FULL probe classes."""
        if max_x < 0 or not transaction_id:
            raise ValueError("pinned FULL pruning requires max_x>=0 and provenance")
        victims: list[tuple[CudaGraphManager, DynamicGraphEntry]] = []
        for manager in self.managers:
            victims.extend(
                (manager, entry)
                for entry in manager._dynamic_graph_entries.values()
                if entry.state == DynamicGraphResidency.HOT
                and entry.pinned
                and entry.descriptor.cg_mode == CUDAGraphMode.FULL
                and (entry.descriptor.physical_num_reqs or 0) > max_x
            )
        victims.sort(
            key=lambda item: repr(
                item[1]
                .descriptor.physical_replay_key(item[0].dynamic_graph_owner)
                .identity
            )
        )
        charged = sum(entry.charged_bytes for _manager, entry in victims)
        local_error: Exception | None = None
        for manager, entry in victims:
            if local_error is not None:
                break
            try:
                manager._evict_dynamic_entry(
                    entry,
                    transaction_id=transaction_id,
                    reason="calibration_unreachable_full_prune",
                    administrative=True,
                    defer_rank_consensus=True,
                )
            except Exception as error:
                local_error = error
        if self.managers:
            self.managers[0]._publish_deferred_eviction_consensus(local_error)
        return len(victims), charged, self.resident_bytes


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        full_decode_query_lens: set[int] | None = None,
        full_decode_cap_query_lens: set[int] | None = None,
        full_decode_max_tokens: int | None = None,
        expand_dynamic_decode_query_lens: bool = True,
        max_uniform_decode_reqs: int | None = None,
        owner: str = "unknown",
        elastic_graph_activation: str = "always",
        elastic_graph_token_source: str = "step",
        elastic_graph_fixed_query_len: int | None = None,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_uniform_decode_reqs = (
            self.max_num_reqs
            if max_uniform_decode_reqs is None
            else min(max_uniform_decode_reqs, self.max_num_reqs)
        )
        if self.max_uniform_decode_reqs < 1:
            raise ValueError("max_uniform_decode_reqs must be positive")
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len
        self.varlen_decode = varlen_decode
        self.full_decode_query_lens = full_decode_query_lens
        self.full_decode_cap_query_lens = full_decode_cap_query_lens
        if full_decode_max_tokens is not None and full_decode_max_tokens <= 0:
            raise ValueError("full_decode_max_tokens must be positive")
        self.full_decode_max_tokens = full_decode_max_tokens
        self.expand_dynamic_decode_query_lens = expand_dynamic_decode_query_lens

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank
        self.lora_capture_cases = lora_capture_cases or [0]
        additional_config = vllm_config.additional_config
        self._p3_crash_diagnostic = bool(
            isinstance(additional_config, dict)
            and additional_config.get("p3_crash_diagnostic", False)
        )
        self._p3_graph_debug = bool(
            isinstance(additional_config, dict)
            and additional_config.get("p3_graph_debug", False)
        )
        self._p3_global_sync = bool(
            isinstance(additional_config, dict)
            and additional_config.get("p3_global_sync", False)
        )
        p3_node_cutoff = (
            additional_config.get("p3_node_cutoff")
            if isinstance(additional_config, dict)
            else None
        )
        if p3_node_cutoff is not None and (
            not isinstance(p3_node_cutoff, int) or p3_node_cutoff < 0
        ):
            raise ValueError("p3_node_cutoff must be a non-negative integer")
        p3_node_prefix_counts = (
            additional_config.get("p3_node_prefix_counts")
            if isinstance(additional_config, dict)
            else None
        )
        if p3_node_prefix_counts is None:
            p3_node_prefix_counts = []
        if (
            not isinstance(p3_node_prefix_counts, list)
            or any(
                not isinstance(count, int) or count <= 0
                for count in p3_node_prefix_counts
            )
            or p3_node_prefix_counts != sorted(set(p3_node_prefix_counts))
        ):
            raise ValueError(
                "p3_node_prefix_counts must be a strictly increasing list "
                "of positive integers"
            )
        if p3_node_cutoff is not None and p3_node_prefix_counts:
            raise ValueError(
                "p3_node_cutoff and p3_node_prefix_counts are mutually exclusive"
            )
        self._p3_node_cutoff: int | None = p3_node_cutoff
        self._p3_node_prefix_counts = tuple(p3_node_prefix_counts)
        self._p3_prefix_diagnostic_descs: set[BatchExecutionDescriptor] = set()
        self._p3_prefix_diagnostic_state: dict[
            BatchExecutionDescriptor,
            tuple[Any, list[tuple[int, Any]], int],
        ] = {}
        self.dynamic_graph_owner = owner
        self.elastic_graph_activation = elastic_graph_activation
        self.elastic_graph_token_source = elastic_graph_token_source
        self.elastic_graph_fixed_query_len = elastic_graph_fixed_query_len
        self.compiled_piecewise_sizes = configured_compiled_piecewise_sizes(vllm_config)
        self._compiled_piecewise_dispatch_logged: set[int] = set()
        self.defer_startup_graphs = uses_elastic_on_demand_graphs(vllm_config)
        if self.defer_startup_graphs:
            from vllm.v1.core.elastic_runtime import (
                compute_elastic_runtime_generation,
            )

            self.runtime_generation = compute_elastic_runtime_generation(vllm_config)
        else:
            self.runtime_generation = "static"
        legacy_elastic_graph_keys = {
            "dynamic_cudagraph_capture_sizes",
            "dynamic_cudagraph_full_capture_sizes",
            "dynamic_cudagraph_piecewise_capture_range",
            "dynamic_cudagraph_piecewise_coverage_sizes",
            "dynamic_cudagraph_piecewise_safety_sizes",
            "dynamic_cudagraph_budget_mb",
            "dynamic_cudagraph_max_entry_mb",
            "dynamic_cudagraph_guard_mb",
            "dynamic_cudagraph_min_hits",
            "dynamic_cudagraph_full_min_hits",
            "dynamic_cudagraph_piecewise_min_hits",
            "dynamic_cudagraph_piecewise_min_padding_pct",
            "dynamic_cudagraph_cooldown_steps",
            "dynamic_cudagraph_pinned_sizes",
        }
        if self.defer_startup_graphs:
            assert isinstance(additional_config, dict)
            configured_legacy_keys = sorted(
                legacy_elastic_graph_keys.intersection(additional_config)
            )
            if configured_legacy_keys:
                raise ValueError(
                    "elastic CUDA Graph shapes and memory are runtime-derived; "
                    "remove legacy additional-config keys: "
                    f"{configured_legacy_keys}"
                )
        dynamic_sizes = (
            additional_config.get("dynamic_cudagraph_capture_sizes", [])
            if isinstance(additional_config, dict)
            else []
        )
        if not isinstance(dynamic_sizes, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in dynamic_sizes
        ):
            raise ValueError(
                "dynamic_cudagraph_capture_sizes must be a list of positive integers"
            )
        dynamic_full_sizes = (
            additional_config.get("dynamic_cudagraph_full_capture_sizes", [])
            if isinstance(additional_config, dict)
            else []
        )
        coverage_sizes = (
            additional_config.get("dynamic_cudagraph_piecewise_coverage_sizes", [])
            if isinstance(additional_config, dict)
            else []
        )
        safety_sizes = (
            additional_config.get("dynamic_cudagraph_piecewise_safety_sizes", [])
            if isinstance(additional_config, dict)
            else []
        )
        piecewise_range = (
            additional_config.get("dynamic_cudagraph_piecewise_capture_range", [])
            if isinstance(additional_config, dict)
            else []
        )
        for name, sizes in (
            ("dynamic_cudagraph_full_capture_sizes", dynamic_full_sizes),
            ("dynamic_cudagraph_piecewise_coverage_sizes", coverage_sizes),
            ("dynamic_cudagraph_piecewise_safety_sizes", safety_sizes),
        ):
            if not isinstance(sizes, list) or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in sizes
            ):
                raise ValueError(f"{name} must be a list of positive integers")
        if (
            not isinstance(piecewise_range, list)
            or len(piecewise_range) not in (0, 2)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in piecewise_range
            )
            or (piecewise_range and piecewise_range[0] > piecewise_range[1])
        ):
            raise ValueError(
                "dynamic_cudagraph_piecewise_capture_range must be an empty "
                "list or [min_tokens, max_tokens]"
            )
        startup_sizes = set(self.compilation_config.cudagraph_capture_sizes or [])
        # An elastic KV arena cannot publish its full X1 capacity and then
        # eagerly materialize every configured CUDA Graph: that recreates a
        # hidden startup reserve and can OOM after KV has been committed.  The
        # target runner therefore treats the ordinary startup shapes as
        # on-demand candidates when a dynamic graph ceiling is configured.
        # Their memory is borrowed only after scheduler admission and is
        # physically returned by the normal dynamic eviction path.
        dynamic_piecewise_sizes = set(dynamic_sizes)
        dynamic_full_capture_sizes = set(dynamic_full_sizes)
        if self.defer_startup_graphs:
            dynamic_piecewise_sizes.clear()
            dynamic_full_capture_sizes.clear()
            coverage_sizes = []
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            safety_sizes = []
            boundary = 1
            while boundary < max_tokens:
                safety_sizes.append(boundary)
                boundary <<= 1
            safety_sizes.append(max_tokens)
            piecewise_range = []
        else:
            dynamic_piecewise_sizes.difference_update(startup_sizes)
            dynamic_full_capture_sizes.difference_update(startup_sizes)
        self.dynamic_piecewise_capture_sizes = tuple(sorted(dynamic_piecewise_sizes))
        self.dynamic_full_capture_sizes = tuple(sorted(dynamic_full_capture_sizes))
        self.dynamic_piecewise_coverage_sizes = tuple(
            sorted(set(coverage_sizes).difference(startup_sizes))
        )
        safety_size_set = set(safety_sizes)
        if not self.defer_startup_graphs:
            safety_size_set.difference_update(startup_sizes)
        self.dynamic_piecewise_safety_sizes = tuple(sorted(safety_size_set))
        self.dynamic_piecewise_capture_range = (
            (piecewise_range[0], piecewise_range[1]) if piecewise_range else None
        )
        piecewise_static_sizes = set(coverage_sizes).union(safety_sizes)
        if not self.defer_startup_graphs:
            piecewise_static_sizes.update(startup_sizes)
        self.dynamic_piecewise_static_sizes = frozenset(piecewise_static_sizes)
        self.dynamic_capture_sizes = tuple(
            sorted(
                set(self.dynamic_piecewise_capture_sizes)
                .union(self.dynamic_full_capture_sizes)
                .union(self.dynamic_piecewise_safety_sizes)
            )
        )
        self.has_dynamic_capture_candidates = self.defer_startup_graphs or bool(
            self.dynamic_capture_sizes or self.dynamic_piecewise_capture_range
        )
        if not self.defer_startup_graphs:
            dynamic_budget_mb = (
                additional_config.get("dynamic_cudagraph_budget_mb", 0)
                if isinstance(additional_config, dict)
                else 0
            )
            if isinstance(dynamic_budget_mb, bool) or not isinstance(
                dynamic_budget_mb, int
            ):
                raise ValueError("dynamic_cudagraph_budget_mb must be an integer")
            if self.has_dynamic_capture_candidates != (dynamic_budget_mb > 0):
                raise ValueError(
                    "dynamic CUDA graphs require both capture sizes and a positive "
                    "dynamic_cudagraph_budget_mb"
                )
            self.dynamic_graph_budget_bytes = dynamic_budget_mb * 1024 * 1024
            max_entry_mb = (
                additional_config.get(
                    "dynamic_cudagraph_max_entry_mb", dynamic_budget_mb
                )
                if isinstance(additional_config, dict)
                else 0
            )
            guard_mb = (
                additional_config.get("dynamic_cudagraph_guard_mb", 150)
                if isinstance(additional_config, dict)
                else 150
            )
            self.dynamic_graph_min_hits = (
                additional_config.get("dynamic_cudagraph_min_hits", 2)
                if isinstance(additional_config, dict)
                else 2
            )
            self.dynamic_graph_full_min_hits = (
                additional_config.get(
                    "dynamic_cudagraph_full_min_hits", self.dynamic_graph_min_hits
                )
                if isinstance(additional_config, dict)
                else self.dynamic_graph_min_hits
            )
            self.dynamic_graph_piecewise_min_hits = (
                additional_config.get(
                    "dynamic_cudagraph_piecewise_min_hits",
                    self.dynamic_graph_min_hits,
                )
                if isinstance(additional_config, dict)
                else self.dynamic_graph_min_hits
            )
            self.dynamic_graph_piecewise_min_padding_pct = (
                additional_config.get("dynamic_cudagraph_piecewise_min_padding_pct", 25)
                if isinstance(additional_config, dict)
                else 25
            )
            self.dynamic_graph_cooldown_steps = (
                additional_config.get("dynamic_cudagraph_cooldown_steps", 128)
                if isinstance(additional_config, dict)
                else 128
            )
            for name, value, minimum in (
                ("dynamic_cudagraph_max_entry_mb", max_entry_mb, 1),
                ("dynamic_cudagraph_guard_mb", guard_mb, 0),
                ("dynamic_cudagraph_min_hits", self.dynamic_graph_min_hits, 1),
                (
                    "dynamic_cudagraph_full_min_hits",
                    self.dynamic_graph_full_min_hits,
                    1,
                ),
                (
                    "dynamic_cudagraph_piecewise_min_hits",
                    self.dynamic_graph_piecewise_min_hits,
                    1,
                ),
                (
                    "dynamic_cudagraph_piecewise_min_padding_pct",
                    self.dynamic_graph_piecewise_min_padding_pct,
                    0,
                ),
                (
                    "dynamic_cudagraph_cooldown_steps",
                    self.dynamic_graph_cooldown_steps,
                    1,
                ),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or (self.dynamic_capture_sizes and value < minimum)
                ):
                    raise ValueError(f"{name} must be an integer >= {minimum}")
            self.dynamic_graph_max_entry_bytes = max_entry_mb * 1024 * 1024
            self.dynamic_graph_guard_bytes = guard_mb * 1024 * 1024
            if self.dynamic_graph_max_entry_bytes > self.dynamic_graph_budget_bytes:
                raise ValueError(
                    "dynamic_cudagraph_max_entry_mb cannot exceed the dynamic "
                    "CUDA Graph budget"
                )
            pinned_sizes = (
                additional_config.get("dynamic_cudagraph_pinned_sizes", [])
                if isinstance(additional_config, dict)
                else []
            )
            if not isinstance(pinned_sizes, list) or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in pinned_sizes
            ):
                raise ValueError(
                    "dynamic_cudagraph_pinned_sizes must be a list of positive integers"
                )
            if not set(pinned_sizes).issubset(self.dynamic_capture_sizes):
                raise ValueError(
                    "dynamic_cudagraph_pinned_sizes must be dynamic capture sizes"
                )
            self.dynamic_graph_pinned_sizes = frozenset(pinned_sizes)
        self._dynamic_graph_entries: dict[
            BatchExecutionDescriptor, DynamicGraphEntry
        ] = {}
        self._dynamic_pending: BatchExecutionDescriptor | None = None
        self._dynamic_epoch = 0
        self._dynamic_resident_bytes = 0
        # CUDA may retain graph-exec/driver metadata after every reachable
        # graph and its private allocator pool have been destroyed. This is
        # measured post-teardown residency, not a speculative reserve. Keep it
        # visible to the elastic KV arena so KV never maps over physical bytes
        # that the lower layer did not return.
        self._dynamic_retention_ledger = DynamicGraphRetentionLedger()
        self._owns_dynamic_retention_ledger = True
        # Dynamic target/MTP managers are registered into one working set
        # before capture.  The working set aliases this holder across owners,
        # so recaptures reuse one stream and cannot accumulate a new
        # stream-associated cuBLAS workspace for every owner/transition.
        self._dynamic_capture_state = DynamicGraphCaptureState()
        self._dynamic_capture_granted_bytes = 0
        # Persist the exact fail-closed discriminator across the bool-returning
        # capture API. MultiprocessExecutor otherwise preserves only
        # ``captured=False`` and drops the rank-local admission/publication
        # evidence before the parent can write its experiment receipt.
        self.last_dynamic_capture_rejection: str | None = None
        self._last_dynamic_dispatch: BatchExecutionDescriptor | None = None
        self._dynamic_step_candidates: set[BatchExecutionDescriptor] = set()
        self._dynamic_step_planned = False
        # Precompute actual num_active_loras -> captured case mapping so that
        # dispatch() is a plain dict lookup instead of a per-call bisect.
        self._lora_dispatch_map, self._max_lora_case = self._build_lora_dispatch_map()

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None

        self._graphs_captured = False

        self._candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}
        self.max_capture_tokens = 0

        # Breakable piecewise Graphs can wrap an explicitly compiled model.
        self.use_breakable_cg = (
            is_breakable_cudagraph_enabled()
            and self.cudagraph_mode.has_piecewise_cudagraphs()
        )
        self.breakable_cg_runner: BreakableCUDAGraphWrapper | None = None

        self._init_candidates()
        if self.defer_startup_graphs:
            self._load_dynamic_recipe_entries()
        if self.defer_startup_graphs and self._dynamic_graph_entries:
            from vllm.v1.worker.startup_plan import maybe_save_cudagraph_recipe

            maybe_save_cudagraph_recipe(
                self.vllm_config,
                rank=self.vllm_config.parallel_config.rank,
                world_size=self.vllm_config.parallel_config.world_size,
                owner=self.dynamic_graph_owner,
                descriptors=self._dynamic_graph_entries,
            )
        if (
            self.defer_startup_graphs
            and self._dynamic_graph_entries
            and not self._capture_descs
        ):
            # dispatch() uses this bit to enable candidate observation.  No
            # graph is resident here; every entry is still WARM and must pass
            # the normal hit, MaxX, loan and capture gates before use.
            self._graphs_captured = True
            logger.debug(
                "Deferred %d %s CUDA Graph descriptors to elastic on-demand "
                "capture; startup residency is zero",
                len(self._dynamic_graph_entries),
                self.dynamic_graph_owner,
            )

    def _load_dynamic_recipe_entries(self) -> None:
        """Restore exact recipe keys as zero-residency WARM metadata."""
        from vllm.v1.worker.startup_plan import load_cudagraph_recipe

        rows = load_cudagraph_recipe(
            self.vllm_config,
            rank=self.vllm_config.parallel_config.rank,
            world_size=self.vllm_config.parallel_config.world_size,
            owner=self.dynamic_graph_owner,
        )
        restored = 0
        for row in rows:
            mode = CUDAGraphMode[row["mode"]]
            num_reqs = row["num_reqs"]
            runtime_num_reqs = row["physical_num_reqs"]
            uniform_token_count = row["uniform_token_count"]
            if row["runtime_generation"] != self.runtime_generation:
                logger.warning(
                    "Ignoring stale CUDA Graph runtime generation for %s: %s",
                    self.dynamic_graph_owner,
                    row["runtime_generation"],
                )
                continue
            if mode == CUDAGraphMode.PIECEWISE:
                if num_reqs is not None or uniform_token_count is not None:
                    logger.warning(
                        "Ignoring semantically invalid PIECEWISE CUDA Graph "
                        "recipe row for %s: %s",
                        self.dynamic_graph_owner,
                        row,
                    )
                    continue
                allow_full = False
            else:
                if num_reqs is None:
                    logger.warning(
                        "Ignoring FULL CUDA Graph recipe row without num_reqs "
                        "for %s: %s",
                        self.dynamic_graph_owner,
                        row,
                    )
                    continue
                allow_full = uniform_token_count is not None
            expected = self.runtime_descriptor(
                runtime_num_reqs,
                row["num_tokens"],
                uniform_token_count,
                row["num_active_loras"],
                allow_full=allow_full,
                semantic_decode=bool(
                    runtime_num_reqs > 0
                    and row["num_tokens"] == runtime_num_reqs * self.decode_query_len
                    and self.dynamic_graph_owner
                    in {"target", "mtp_prefill", "mtp_decode"}
                ),
            )
            if expected is None:
                logger.warning(
                    "Ignoring unreachable CUDA Graph recipe row for %s: %s",
                    self.dynamic_graph_owner,
                    row,
                )
                continue
            descriptor = BatchExecutionDescriptor(
                cg_mode=mode,
                num_tokens=row["num_tokens"],
                num_reqs=num_reqs,
                uniform_token_count=uniform_token_count,
                num_active_loras=row["num_active_loras"],
                physical_num_reqs=runtime_num_reqs,
                runtime_generation=self.runtime_generation,
                semantic_decode=expected.semantic_decode,
            )
            if expected != descriptor:
                logger.warning(
                    "Ignoring unreachable CUDA Graph recipe row for %s: %s",
                    self.dynamic_graph_owner,
                    row,
                )
                continue
            if descriptor not in self._dynamic_graph_entries:
                self._dynamic_graph_entries[descriptor] = DynamicGraphEntry(
                    descriptor=descriptor
                )
                restored += 1
        if restored:
            logger.debug(
                "Restored %d %s CUDA Graph descriptors as WARM metadata; "
                "startup graph residency is zero",
                restored,
                self.dynamic_graph_owner,
            )

    def rebind_runtime_generation(self, generation: str) -> None:
        """Rekey zero-residency metadata to the post-KV scheduler epoch."""
        if generation == self.runtime_generation:
            return
        if not self.defer_startup_graphs:
            raise RuntimeError("static CUDA Graph metadata cannot change generation")
        if (
            self.graphs
            or self._dynamic_pending is not None
            or self._last_dynamic_dispatch is not None
            or self._dynamic_step_candidates
            or self._dynamic_step_planned
            or self._p3_prefix_diagnostic_descs
            or self._p3_prefix_diagnostic_state
        ):
            raise RuntimeError(
                "cannot change elastic runtime generation after graph activity"
            )
        if any(
            entry.state != DynamicGraphResidency.WARM
            or entry.charged_bytes
            or entry.local_pool_bytes
            or entry.reclaimable_bytes
            or entry.leases
            for entry in self._dynamic_graph_entries.values()
        ):
            raise RuntimeError(
                "cannot change elastic runtime generation with resident state"
            )

        def rebound(
            descriptor: BatchExecutionDescriptor,
        ) -> BatchExecutionDescriptor:
            return replace(descriptor, runtime_generation=generation)

        entries: dict[BatchExecutionDescriptor, DynamicGraphEntry] = {}
        for descriptor, entry in self._dynamic_graph_entries.items():
            descriptor = rebound(descriptor)
            if descriptor in entries:
                raise RuntimeError("elastic generation rebind produced a collision")
            entry.descriptor = descriptor
            entries[descriptor] = entry
        self._dynamic_graph_entries = entries
        self._capture_descs = {
            mode: [rebound(descriptor) for descriptor in descriptors]
            for mode, descriptors in self._capture_descs.items()
        }
        self._candidates = {
            key: [rebound(descriptor) for descriptor in descriptors]
            for key, descriptors in self._candidates.items()
        }
        self.runtime_generation = generation

    def _build_lora_dispatch_map(self) -> tuple[dict[int, int], int]:
        """Precompute actual num_active_loras -> effective captured case.

        Mirrors the num_tokens candidate expansion in ``_init_candidates``:
        every possible active-LoRA count is mapped ahead of time to the
        smallest captured case that can serve it, so ``dispatch`` is a plain
        dict lookup instead of a per-call bisect.
        """
        captured_with_lora = sorted(c for c in self.lora_capture_cases if c > 0)
        if not captured_with_lora:
            return {}, 0
        dispatch_map: dict[int, int] = {}
        case_idx = 0
        for n in range(1, captured_with_lora[-1] + 1):
            while captured_with_lora[case_idx] < n:
                case_idx += 1
            dispatch_map[n] = captured_with_lora[case_idx]
        return dispatch_map, captured_with_lora[-1]

    def _resolve_effective_loras(self, num_active_loras: int) -> int:
        """Map an actual active-LoRA count to its captured graph case."""
        if num_active_loras <= 0 or not self._lora_dispatch_map:
            return num_active_loras
        # Counts above the largest captured case clamp to it.
        return self._lora_dispatch_map.get(num_active_loras, self._max_lora_case)

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        if self.defer_startup_graphs:
            # Elastic mode derives the exact descriptor from the admitted
            # runtime shape. Configured/default capture-size lists are neither
            # a coverage contract nor a reason to withhold startup memory.
            self.max_capture_tokens = (
                self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self._graphs_captured = True
            return
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
        registered_mixed_sizes = sorted(
            set(capture_sizes)
            .union(self.dynamic_piecewise_coverage_sizes)
            .union(self.dynamic_piecewise_capture_sizes)
            .union(self.dynamic_piecewise_safety_sizes)
        )
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()
        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size

        descs_by_mode: defaultdict[CUDAGraphMode, list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )
        descs_by_token_lora: defaultdict[
            tuple[int, int], list[BatchExecutionDescriptor]
        ] = defaultdict(list)

        # When using Dynamic SD, num_speculative_tokens is the max number of
        # draft tokens. The scheduler might use a smaller number so we need
        # to capture graphs for all possible values during decode.
        speculative_config = self.vllm_config.speculative_config
        if (
            self.expand_dynamic_decode_query_lens
            and speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens
            # _per_step. Recover num_new_sampled_tokens_per_step
            # from the values the manager already has.
            num_new_sampled_tokens_per_step = (
                self.decode_query_len - self.vllm_config.num_speculative_tokens
            )
            dense_schedule = build_dynamic_sd_schedule_lookup(
                speculative_config.num_speculative_tokens_per_batch_size,
                vllm_max_batch_size=self.max_num_reqs,
                vllm_num_speculative_tokens=self.vllm_config.num_speculative_tokens,
            )
            decode_query_lens = sorted(
                {
                    num_spec + num_new_sampled_tokens_per_step
                    for num_spec in dense_schedule[1:]
                }
            )
        else:
            decode_query_lens = [self.decode_query_len]
        if any(query_len <= 0 for query_len in decode_query_lens):
            raise ValueError(
                "CUDA graph decode query lengths must be positive, got "
                f"{decode_query_lens}. A draft-model graph must disable "
                "dynamic target decode-query expansion."
            )
        if self.full_decode_query_lens is not None:
            # The runtime phase policy can select a non-speculative lane that
            # is absent from the static batch-size schedule. Capture the
            # explicitly requested FULL lanes directly instead of intersecting
            # them with that incomplete schedule.
            decode_query_lens = sorted(self.full_decode_query_lens)
        max_decode_tokens = self.max_uniform_decode_reqs * max(decode_query_lens)

        capture_varlen_decode = (
            separate_decode_routine and bool(decode_mode) and self.varlen_decode
        )
        if capture_varlen_decode:
            varlen_cases = product(capture_sizes, self.lora_capture_cases)
            for num_tokens, num_active_loras in varlen_cases:
                # Varlen decode graphs take any mix of 1..decode_query_len tokens per
                # request, worst case 1 token per request (or max_num_reqs)
                if num_tokens > max_decode_tokens:
                    continue
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=num_tokens,
                    num_reqs=min(num_tokens, self.max_num_reqs),
                    max_query_len=self.decode_query_len,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[decode_mode].append(desc)
                descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)
        # FULL uniform decode has a request-count contract that is independent
        # from PIECEWISE token padding. When elastic KV provides a post-profile
        # cap, capture every reachable request count without expanding the
        # mixed/PIECEWISE family.
        elif separate_decode_routine and decode_mode:
            for decode_query_len in decode_query_lens:
                query_capture_sizes = set(capture_sizes).union(
                    self.dynamic_full_capture_sizes
                )
                query_max_decode_tokens = (
                    self.max_uniform_decode_reqs * decode_query_len
                )
                capture_cap_boundary = decode_query_len in (
                    getattr(self, "full_decode_cap_query_lens", None) or set()
                )
                if capture_cap_boundary:
                    query_capture_sizes.add(query_max_decode_tokens)
                if decode_query_len > 1:
                    # Multi-token FULL replay cannot pad request rows. Register
                    # every physically reachable X; otherwise gaps in a sparse
                    # startup-size list silently dispatch to NONE.
                    query_capture_sizes.update(
                        decode_query_len * num_reqs
                        for num_reqs in range(1, self.max_uniform_decode_reqs + 1)
                    )
                elif self.max_uniform_decode_reqs < self.max_num_reqs:
                    # Single-token FULL decode can pad request rows. Preserve
                    # its sparse family and add only the exact cap boundary.
                    query_capture_sizes.add(self.max_uniform_decode_reqs)
                for num_tokens, num_active_loras in product(
                    sorted(query_capture_sizes), self.lora_capture_cases
                ):
                    rounded_num_tokens = round_up(num_tokens, decode_query_len)
                    rounded_num_reqs = rounded_num_tokens // decode_query_len

                    if (
                        rounded_num_tokens > max_decode_tokens
                        or (
                            self.full_decode_max_tokens is not None
                            and rounded_num_tokens > self.full_decode_max_tokens
                        )
                        or (
                            rounded_num_tokens > max_cg_capture_size
                            and not (
                                capture_cap_boundary
                                and rounded_num_tokens == query_max_decode_tokens
                            )
                            and not self.defer_startup_graphs
                            and rounded_num_tokens
                            not in self.dynamic_full_capture_sizes
                        )
                        or rounded_num_reqs > self.max_uniform_decode_reqs
                    ):
                        continue

                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=rounded_num_tokens,
                        num_reqs=rounded_num_reqs,
                        uniform_token_count=decode_query_len,
                        num_active_loras=num_active_loras,
                    )

                    # avoid duplicate graphs
                    if desc not in descs_by_mode[decode_mode]:
                        descs_by_mode[decode_mode].append(desc)
                        descs_by_token_lora[
                            (rounded_num_tokens, num_active_loras)
                        ].append(desc)
                        if (
                            self.defer_startup_graphs
                            or rounded_num_tokens in self.dynamic_full_capture_sizes
                        ):
                            self._dynamic_graph_entries[desc] = DynamicGraphEntry(
                                descriptor=desc,
                                pinned=(
                                    rounded_num_tokens
                                    in self.dynamic_graph_pinned_sizes
                                ),
                                min_hits=self.dynamic_graph_full_min_hits,
                            )

        for num_tokens, num_active_loras in product(
            registered_mixed_sizes, self.lora_capture_cases
        ):
            # recoverSSM cannot capture a dummy query wider than its workspace.
            if mixed_mode and (
                not self.vllm_config.cache_config.use_kda_recoverssm
                or num_tokens <= max_decode_tokens
            ):
                # for PIECEWISE graphs there is no limit on requests when replaying
                # i.e. no request padding is needed, so we leave it as None.
                # For breakable PW graphs, break-point kernels read the real batch
                # from the forward context; in-graph kernels handle the token padding
                # themselves from the padded slot_mapping (rows with slot == -1).
                num_reqs = None
                if mixed_mode == CUDAGraphMode.FULL:
                    num_reqs = min(num_tokens, self.max_num_reqs)
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)
                if num_tokens in self.dynamic_piecewise_capture_sizes or (
                    num_tokens in self.dynamic_piecewise_safety_sizes
                ):
                    self._dynamic_graph_entries[desc] = DynamicGraphEntry(
                        descriptor=desc,
                        pinned=num_tokens in self.dynamic_graph_pinned_sizes,
                        min_hits=(
                            1
                            if num_tokens in self.dynamic_piecewise_safety_sizes
                            else self.dynamic_graph_piecewise_min_hits
                        ),
                    )

        if not descs_by_token_lora:
            return

        all_token_counts = sorted({k[0] for k in descs_by_token_lora})
        for num_tokens in range(all_token_counts[-1] + 1):
            for num_active_loras in self.lora_capture_cases:
                candidates: list[BatchExecutionDescriptor] = []
                for token_cg_size in all_token_counts:
                    if token_cg_size < num_tokens:
                        continue
                    candidates.extend(
                        descs_by_token_lora.get((token_cg_size, num_active_loras), [])
                    )
                if candidates:
                    self._candidates[(num_tokens, num_active_loras)] = candidates

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            startup_descs = [
                desc for desc in descs if desc not in self._dynamic_graph_entries
            ]
            if startup_descs:
                self._capture_descs[mode] = startup_descs
        self.max_capture_tokens = max(
            (desc.num_tokens for descs in descs_by_mode.values() for desc in descs),
            default=0,
        )

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    def has_pending_dynamic_capture(self) -> bool:
        return self._dynamic_pending is not None

    @property
    def dynamic_resident_bytes(self) -> int:
        retained = (
            self._dynamic_retention_ledger.rank_safe_bytes
            if self._owns_dynamic_retention_ledger
            else 0
        )
        return self._dynamic_resident_bytes + retained

    @property
    def dynamic_active_graph_bytes(self) -> int:
        """Return bytes charged to currently HOT graph executables only."""
        return self._dynamic_resident_bytes

    def begin_dynamic_step(self) -> None:
        self._last_dynamic_dispatch = None
        self._dynamic_step_candidates.clear()
        self._dynamic_step_planned = False

    def _is_exact_batched_decode_shape(
        self,
        num_reqs: int,
        num_tokens: int,
        *,
        semantic_decode: bool,
    ) -> bool:
        """Return whether this is the exact assembled multi-token decode M.

        The target and MTP-prefill owners consume the K+1 token wave as one
        physical shape even though the verifier contract is represented as
        PIECEWISE.  Until padded-tail mutation is proven for the complete DAG,
        exact M is the only Graph-backed production representation.
        """
        return bool(
            semantic_decode
            and self.dynamic_graph_owner in {"target", "mtp_prefill"}
            and num_reqs > 0
            and num_tokens == num_reqs * self.decode_query_len
        )

    def is_compiled_piecewise_shape(
        self,
        num_tokens: int,
        num_reqs: int | None = None,
        *,
        semantic_decode: bool = False,
    ) -> bool:
        """Return whether this PIECEWISE invocation must bypass CUDA Graph.

        A PIECEWISE descriptor is captured at its power-of-two token bucket.
        Replaying that executable with fewer live tokens is not accepted by the
        elastic runtime contract: the padded tail has not been proven inert for
        every captured producer/consumer.  Keep exact buckets graph-backed, but
        route every underfilled tail through the outer compiled model without
        an inner CUDA Graph.  Explicitly configured buckets remain compiled-only
        even when they are exact.
        """
        if self.dynamic_graph_owner not in {"target", "mtp_prefill"} or num_tokens <= 0:
            return False
        if num_reqs is not None and self._is_exact_batched_decode_shape(
            num_reqs, num_tokens, semantic_decode=semantic_decode
        ):
            return False
        token_bucket = elastic_piecewise_token_boundary(
            num_tokens,
            self.vllm_config.scheduler_config.max_num_batched_tokens,
        )
        return (
            num_tokens != token_bucket or token_bucket in self.compiled_piecewise_sizes
        )

    def _runtime_decode_query_lens(self) -> set[int]:
        if self.full_decode_query_lens is not None:
            return set(self.full_decode_query_lens)
        speculative_config = self.vllm_config.speculative_config
        if (
            self.expand_dynamic_decode_query_lens
            and speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            schedule = speculative_config.num_speculative_tokens_per_batch_size
            assert schedule is not None
            sampled = self.decode_query_len - self.vllm_config.num_speculative_tokens
            return {num_spec + sampled for _, _, num_spec in schedule}
        return {self.decode_query_len}

    def runtime_descriptor(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        *,
        allow_full: bool,
        semantic_decode: bool = False,
    ) -> BatchExecutionDescriptor | None:
        """Derive the exact graph key from the admitted runtime shape."""
        if not self.defer_startup_graphs or num_tokens <= 0:
            return None
        if num_tokens > self.vllm_config.scheduler_config.max_num_batched_tokens:
            raise RuntimeError(
                "admitted CUDA Graph shape exceeds max_num_batched_tokens: "
                f"owner={self.dynamic_graph_owner} num_tokens={num_tokens}"
            )
        effective_loras = self._resolve_effective_loras(num_active_loras)
        decode_mode = self.cudagraph_mode.decode_mode()
        if (
            allow_full
            and decode_mode == CUDAGraphMode.FULL
            and uniform_token_count in self._runtime_decode_query_lens()
            and (
                self.full_decode_max_tokens is None
                or num_tokens <= self.full_decode_max_tokens
            )
            and num_reqs <= self.max_uniform_decode_reqs
            and num_tokens == num_reqs * uniform_token_count
        ):
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform_token_count=uniform_token_count,
                num_active_loras=effective_loras,
                physical_num_reqs=num_reqs,
                runtime_generation=self.runtime_generation,
                semantic_decode=semantic_decode,
            )
        mixed_mode = self.cudagraph_mode.mixed_mode()
        if mixed_mode == CUDAGraphMode.NONE:
            return None
        exact_batched_decode = self._is_exact_batched_decode_shape(
            num_reqs, num_tokens, semantic_decode=semantic_decode
        )
        graph_tokens = (
            num_tokens
            if exact_batched_decode
            else elastic_piecewise_token_boundary(
                num_tokens,
                self.vllm_config.scheduler_config.max_num_batched_tokens,
            )
        )
        return BatchExecutionDescriptor(
            cg_mode=mixed_mode,
            num_tokens=graph_tokens,
            num_reqs=num_reqs if mixed_mode == CUDAGraphMode.FULL else None,
            uniform_token_count=(
                uniform_token_count
                if exact_batched_decode and uniform_token_count == self.decode_query_len
                else None
            ),
            num_active_loras=effective_loras,
            physical_num_reqs=num_reqs,
            runtime_generation=self.runtime_generation,
            semantic_decode=semantic_decode,
        )

    def _capture_num_reqs(self, desc: BatchExecutionDescriptor) -> int:
        """Resolve the physical request cardinality used to build capture inputs.

        PIECEWISE descriptors deliberately key executable segments by their token
        bucket, so ``desc.num_reqs`` is ``None``.  Dynamic attention metadata is
        nevertheless request-cardinality dependent.  Once runtime admission has
        queued a physical shape, capture must use that exact X rather than the old
        startup fallback of treating every padded token as a separate request.
        """
        if desc.physical_num_reqs is not None:
            entry = self._dynamic_graph_entries.get(desc)
            if (
                entry is not None
                and entry.runtime_num_reqs is not None
                and entry.runtime_num_reqs != desc.physical_num_reqs
            ):
                raise RuntimeError(
                    "invalid physical request cardinality for CUDA Graph capture: "
                    f"owner={self.dynamic_graph_owner} descriptor={desc} "
                    f"entry_runtime_num_reqs={entry.runtime_num_reqs}"
                )
            return desc.physical_num_reqs
        if desc.num_reqs is not None:
            return desc.num_reqs
        entry = self._dynamic_graph_entries.get(desc)
        runtime_num_reqs = entry.runtime_num_reqs if entry is not None else None
        if runtime_num_reqs is None:
            return min(desc.num_tokens, self.max_num_reqs)
        upper_bound = min(desc.num_tokens, self.max_num_reqs)
        if not 1 <= runtime_num_reqs <= upper_bound:
            raise RuntimeError(
                "invalid physical request cardinality for CUDA Graph capture: "
                f"owner={self.dynamic_graph_owner} descriptor={desc} "
                f"runtime_num_reqs={runtime_num_reqs} upper_bound={upper_bound}"
            )
        return runtime_num_reqs

    def queue_runtime_descriptor(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        *,
        allow_full: bool,
        semantic_decode: bool = False,
    ) -> BatchExecutionDescriptor | None:
        self._dynamic_step_planned = True
        self.last_dynamic_capture_rejection = None
        desc = self.runtime_descriptor(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            allow_full=allow_full,
            semantic_decode=semantic_decode,
        )
        if desc is None:
            return None
        if (
            self.defer_startup_graphs
            and self.dynamic_graph_owner in {"target", "mtp_prefill"}
            and desc.cg_mode == CUDAGraphMode.PIECEWISE
            and self.is_compiled_piecewise_shape(
                num_tokens,
                num_reqs=num_reqs,
                semantic_decode=semantic_decode,
            )
        ):
            # Same-step planning is still mandatory, but this configured
            # carrier deliberately uses NONE at the inner CUDA Graph wrapper
            # boundary.  The outer model remains torch-compiled.
            return replace(desc, cg_mode=CUDAGraphMode.NONE)
        self._dynamic_step_candidates.add(desc)
        self._dynamic_epoch += 1
        entry = self._dynamic_graph_entries.get(desc)
        if entry is None:
            entry = DynamicGraphEntry(
                descriptor=desc,
                pinned=desc.cg_mode == CUDAGraphMode.FULL,
            )
            self._dynamic_graph_entries[desc] = entry
            from vllm.v1.worker.startup_plan import maybe_save_cudagraph_recipe

            maybe_save_cudagraph_recipe(
                self.vllm_config,
                rank=self.vllm_config.parallel_config.rank,
                world_size=self.vllm_config.parallel_config.world_size,
                owner=self.dynamic_graph_owner,
                descriptors=self._dynamic_graph_entries,
            )
        entry.last_used_epoch = self._dynamic_epoch
        if entry.state == DynamicGraphResidency.HOT:
            return desc
        entry.runtime_num_reqs = num_reqs
        if entry.state == DynamicGraphResidency.COOLDOWN:
            raise RuntimeError(
                "required runtime CUDA Graph is cooling down after a physical "
                f"capture failure: owner={self.dynamic_graph_owner} desc={desc}"
            )
        if self._dynamic_pending is not None and self._dynamic_pending != desc:
            self.cancel_pending_dynamic_capture()
        entry.state = DynamicGraphResidency.QUEUED
        entry.hits = 1
        self._dynamic_pending = desc
        logger.debug(
            "Runtime CUDA Graph queued for same-step capture: owner=%s desc=%s",
            self.dynamic_graph_owner,
            desc,
        )
        return desc

    def _descriptor_for_physical_key(
        self, key: PhysicalReplayKey
    ) -> BatchExecutionDescriptor:
        """Resolve one scheduler key without changing dynamic Graph state."""
        if key.logical.owner != self.dynamic_graph_owner:
            raise RuntimeError(
                "elastic graph owner mismatch: "
                f"manager={self.dynamic_graph_owner!r} key={key.logical.owner!r}"
            )
        if key.generation.value != self.runtime_generation:
            raise RuntimeError(
                "stale elastic graph runtime generation: "
                f"manager={self.runtime_generation!r} key={key.generation.value!r}"
            )
        try:
            mode = CUDAGraphMode[key.logical.mode]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown elastic CUDA Graph mode {key.logical.mode!r}"
            ) from exc
        decode_mode = self.cudagraph_mode.decode_mode()
        elastic_token_source = getattr(self, "elastic_graph_token_source", "legacy-v1")
        owner_policy = OwnerGraphExecutionPolicy(
            owner=self.dynamic_graph_owner,
            full_query_lens=(
                tuple(sorted(self._runtime_decode_query_lens()))
                if decode_mode == CUDAGraphMode.FULL
                else ()
            ),
            piecewise_mode=self.cudagraph_mode.mixed_mode().name,
            compiled_piecewise_sizes=tuple(sorted(self.compiled_piecewise_sizes)),
            capability_contract=(
                "effective-cudagraph-manager-v2:math-lane-identity-v1:"
                f"{self.cudagraph_mode.name}:"
                f"decode={decode_mode.name}:"
                f"mixed={self.cudagraph_mode.mixed_mode().name}"
            ),
            piecewise_padding_contract=(
                "exact-batched-decode-m-v1"
                if elastic_token_source == "step"
                else "power-of-two-terminal-exact-v1"
            ),
            piecewise_query_len_min_tokens=(
                envs.AG2_VLLM_TP3_OWNER_MIN_ROWS
                if getattr(self, "tp3_owner_prequant", False)
                else None
            ),
            activation=getattr(self, "elastic_graph_activation", "legacy-v1"),
            token_source=elastic_token_source,
            fixed_query_len=getattr(self, "elastic_graph_fixed_query_len", None),
        )
        owner_policy.validate_physical_key(key)
        desc = BatchExecutionDescriptor(
            cg_mode=mode,
            num_tokens=key.logical.token_bucket,
            num_reqs=key.logical.logical_num_reqs,
            uniform_token_count=key.logical.uniform_query_len,
            num_active_loras=key.logical.active_loras,
            physical_num_reqs=key.physical_num_reqs,
            runtime_generation=key.generation.value,
            semantic_decode=bool(
                key.physical_num_reqs > 0
                and key.logical.token_bucket
                == key.physical_num_reqs * self.decode_query_len
                and elastic_token_source
                in {"step", "requests", "fixed_query", "legacy-v1"}
            ),
        )
        if desc.physical_replay_key(self.dynamic_graph_owner) != key:
            raise RuntimeError("elastic physical key did not round-trip")
        return desc

    def validate_physical_key_queue(
        self, key: PhysicalReplayKey
    ) -> BatchExecutionDescriptor:
        """Validate queue preconditions without creating or touching an entry."""
        desc = self._descriptor_for_physical_key(key)
        entry = self._dynamic_graph_entries.get(desc)
        if entry is not None and entry.state == DynamicGraphResidency.COOLDOWN:
            raise RuntimeError(
                "required physical CUDA Graph is cooling down after capture failure: "
                f"owner={self.dynamic_graph_owner} descriptor={desc}"
            )
        # A retained HOT carrier adds no capture. Queueing it after the one
        # COLD destination must be equivalent to the reverse order; only a
        # second distinct non-HOT destination conflicts with pending capture.
        if (
            self._dynamic_pending is not None
            and self._dynamic_pending != desc
            and (entry is None or entry.state != DynamicGraphResidency.HOT)
        ):
            raise RuntimeError(
                "one manager received multiple simultaneous cold physical keys"
            )
        return desc

    def queue_physical_key(self, key: PhysicalReplayKey) -> BatchExecutionDescriptor:
        """Queue exactly the physical key resolved by the scheduler plan."""
        desc = self.validate_physical_key_queue(key)
        mode = desc.cg_mode
        self._dynamic_step_planned = True
        self.last_dynamic_capture_rejection = None
        self._dynamic_step_candidates.add(desc)
        self._dynamic_epoch += 1
        entry = self._dynamic_graph_entries.get(desc)
        if entry is None:
            entry = DynamicGraphEntry(
                descriptor=desc,
                runtime_num_reqs=key.physical_num_reqs,
                pinned=mode == CUDAGraphMode.FULL,
            )
            self._dynamic_graph_entries[desc] = entry
        entry.last_used_epoch = self._dynamic_epoch
        if entry.state == DynamicGraphResidency.HOT:
            return desc
        entry.runtime_num_reqs = key.physical_num_reqs
        entry.state = DynamicGraphResidency.QUEUED
        entry.hits = max(1, entry.hits)
        self._dynamic_pending = desc
        return desc

    def is_physical_key_hot(self, key: PhysicalReplayKey) -> bool:
        if key.logical.owner != self.dynamic_graph_owner:
            return False
        return any(
            entry.state == DynamicGraphResidency.HOT
            and desc.physical_replay_key(self.dynamic_graph_owner) == key
            for desc, entry in self._dynamic_graph_entries.items()
        )

    def _entries_for_physical_key(
        self, key: PhysicalReplayKey
    ) -> tuple[DynamicGraphEntry, ...]:
        if key.logical.owner != self.dynamic_graph_owner:
            raise RuntimeError(
                "elastic graph owner mismatch: "
                f"manager={self.dynamic_graph_owner!r} key={key.logical.owner!r}"
            )
        return tuple(
            entry
            for desc, entry in self._dynamic_graph_entries.items()
            if desc.physical_replay_key(self.dynamic_graph_owner) == key
        )

    def validate_physical_key_lease(
        self, key: PhysicalReplayKey, transaction_id: str
    ) -> None:
        """Validate a protected-key lease without touching its lease set."""
        if not transaction_id:
            raise RuntimeError("dynamic CUDA Graph lease requires transaction id")
        matching = self._entries_for_physical_key(key)
        if len(matching) != 1 or matching[0].state != DynamicGraphResidency.HOT:
            raise RuntimeError(
                "scheduler-protected CUDA Graph is not uniquely HOT: "
                f"owner={self.dynamic_graph_owner} key={key.identity}"
            )

    def validate_physical_key_eviction(
        self,
        key: PhysicalReplayKey,
        *,
        transaction_id: str,
        reason: str,
        administrative: bool = False,
    ) -> None:
        """Validate an eviction victim without synchronizing or freeing CUDA."""
        matching = self._entries_for_physical_key(key)
        if len(matching) != 1:
            raise RuntimeError(
                "elastic eviction victim is absent or ambiguous: "
                f"owner={self.dynamic_graph_owner} key={key.identity}"
            )
        self._validate_dynamic_entry_eviction(
            matching[0],
            transaction_id=transaction_id,
            reason=reason,
            administrative=administrative,
        )

    def evict_physical_key(
        self,
        key: PhysicalReplayKey,
        *,
        transaction_id: str,
        reason: str,
        administrative: bool = False,
        defer_rank_consensus: bool = False,
    ) -> None:
        """Destroy one scheduler-selected executable, with exact provenance."""
        matching = self._entries_for_physical_key(key)
        self.validate_physical_key_eviction(
            key,
            transaction_id=transaction_id,
            reason=reason,
            administrative=administrative,
        )
        self._evict_dynamic_entry(
            matching[0],
            transaction_id=transaction_id,
            reason=reason,
            administrative=administrative,
            defer_rank_consensus=defer_rank_consensus,
        )

    def discard_failed_physical_key(
        self,
        key: PhysicalReplayKey,
        *,
        transaction_id: str,
    ) -> None:
        """Remove a failed destination without touching unrelated HOT entries."""
        matching = [
            entry
            for desc, entry in self._dynamic_graph_entries.items()
            if desc.physical_replay_key(self.dynamic_graph_owner) == key
        ]
        if len(matching) != 1:
            raise RuntimeError("failed capture destination is absent or ambiguous")
        entry = matching[0]
        if entry.state == DynamicGraphResidency.HOT:
            self._evict_dynamic_entry(
                entry,
                transaction_id=transaction_id,
                reason="failed_capture_rollback",
                administrative=True,
            )
        elif self._dynamic_pending == entry.descriptor:
            self.cancel_pending_dynamic_capture()
        elif entry.graph_segments:
            raise RuntimeError("non-HOT failed capture retained executable segments")
        entry.state = DynamicGraphResidency.WARM
        entry.hits = 0
        entry.runtime_num_reqs = None

    def release_unused_dynamic_residency(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        planned_descriptor: BatchExecutionDescriptor | None = None,
    ) -> None:
        """Compatibility shim; ordinary shape changes never evict residency."""
        return None

    def _dynamic_descriptor_matches_step(
        self,
        desc: BatchExecutionDescriptor,
        *,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
    ) -> bool:
        if not _is_compatible(
            desc,
            num_reqs,
            num_tokens,
            uniform_token_count,
            self._resolve_effective_loras(num_active_loras),
            max_query_len,
        ):
            return False
        if desc.cg_mode == CUDAGraphMode.FULL:
            if desc.uniform_token_count == 1:
                return (
                    desc.num_reqs is not None
                    and desc.num_reqs >= num_reqs
                    and desc.num_tokens == desc.num_reqs
                    and num_tokens == num_reqs
                )
            return desc.num_tokens == num_tokens and desc.num_reqs == num_reqs
        if (
            self.dynamic_graph_owner in {"target", "mtp_prefill"}
            and uniform_token_count is not None
            and uniform_token_count > 0
            and num_tokens == num_reqs * uniform_token_count
        ):
            physical_x = desc.physical_num_reqs
            return bool(
                physical_x is not None
                and physical_x >= num_reqs
                and desc.num_tokens == physical_x * uniform_token_count
            )
        if desc.num_tokens not in self.dynamic_piecewise_safety_sizes:
            return desc.num_tokens == num_tokens
        return True

    def cancel_pending_dynamic_capture(self) -> None:
        desc = self._dynamic_pending
        if desc is None:
            return
        entry = self._dynamic_graph_entries[desc]
        self._dynamic_pending = None
        # The grant authorizes exactly this pending descriptor in the current
        # scheduler transaction.  A shape change/cancellation must revoke it;
        # otherwise a later retry could reach capture with stale physical
        # authority if its own prepare phase is skipped or fails early.
        self._dynamic_capture_granted_bytes = 0
        # A scheduler shape change is not a capture failure. Returning this
        # descriptor to WARM lets a later recurrence request a fresh loan;
        # COOLDOWN is reserved for failed/rejected physical capture.
        entry.state = DynamicGraphResidency.WARM
        entry.hits = 0
        entry.runtime_num_reqs = None
        entry.cooldown_until_epoch = 0

    def dynamic_memory_request_bytes(self) -> int:
        """Return current residency plus any known pending footprint."""
        if self._dynamic_pending is None:
            return self.dynamic_resident_bytes
        entry = self._dynamic_graph_entries[self._dynamic_pending]
        return self.dynamic_resident_bytes + entry.estimated_bytes

    def prepare_pending_dynamic_capture(self, granted_bytes: int) -> bool:
        """Validate a preplanned capture loan without choosing local victims."""
        desc = self._dynamic_pending
        if desc is None:
            return False
        entry = self._dynamic_graph_entries[desc]
        required = self._dynamic_capture_loan_bytes(entry)
        if granted_bytes < required:
            logger.warning(
                "Dynamic CUDA graph promotion denied by elastic KV: owner=%s "
                "descriptor=%s required_bytes=%d granted_bytes=%d",
                self.dynamic_graph_owner,
                desc,
                required,
                granted_bytes,
            )
            return False
        self._dynamic_capture_granted_bytes = granted_bytes
        return True

    def capture_next_dynamic(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        capture_override: Callable[
            [
                dict[CUDAGraphMode, list[BatchExecutionDescriptor]],
                Callable[
                    [BatchExecutionDescriptor, Callable[[CUDAGraphMode], None]], None
                ],
            ],
            None,
        ]
        | None = None,
    ) -> bool:
        """Promote one graph for a manager with a custom capture frontend.

        The target manager owns the physical promotion implementation today.
        Draft managers share that lifecycle but provide ``capture_override``
        because their input/output preparation differs from the target model.
        Keeping the entrypoint on the common manager prevents that owner-
        independent lifecycle from disappearing on the MTP path.
        """
        return ModelCudaGraphManager.capture_next_dynamic(
            self,  # type: ignore[arg-type]
            model,
            model_state,
            input_buffers,
            intermediate_tensors,
            block_tables,
            attn_groups,
            kv_cache_config,
            has_lora=has_lora,
            use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
            lora_capture_hook=lora_capture_hook,
            capture_override=capture_override,
        )

    def finish_dynamic_step(self) -> int:
        """Finish accounting without changing physical graph residency."""
        pending = self._dynamic_pending
        if pending is not None and pending not in self._dynamic_step_candidates:
            self.cancel_pending_dynamic_capture()
        self._last_dynamic_dispatch = None
        return self.dynamic_memory_request_bytes()

    def acquire_step_leases(self, transaction_id: str) -> None:
        if not transaction_id:
            raise RuntimeError("dynamic CUDA Graph lease requires transaction id")
        for desc in self._dynamic_step_candidates:
            entry = self._dynamic_graph_entries[desc]
            if entry.state != DynamicGraphResidency.HOT:
                raise RuntimeError(
                    "cannot lease a non-HOT CUDA Graph entry: "
                    f"owner={self.dynamic_graph_owner} descriptor={desc}"
                )
            entry.leases.add(transaction_id)

    def acquire_physical_key_lease(
        self, key: PhysicalReplayKey, transaction_id: str
    ) -> None:
        """Lease one scheduler-protected HOT key without selecting dispatch."""
        self.validate_physical_key_lease(key, transaction_id)
        matching = self._entries_for_physical_key(key)
        matching[0].leases.add(transaction_id)

    def release_transaction_leases(self, transaction_id: str) -> None:
        for entry in self._dynamic_graph_entries.values():
            entry.leases.discard(transaction_id)

    def _observe_dynamic_descriptor(self, desc: BatchExecutionDescriptor) -> bool:
        self._dynamic_step_candidates.add(desc)
        entry = self._dynamic_graph_entries[desc]
        self._dynamic_epoch += 1
        entry.last_used_epoch = self._dynamic_epoch
        if entry.state == DynamicGraphResidency.COOLDOWN:
            if self._dynamic_epoch < entry.cooldown_until_epoch:
                return False
            entry.state = DynamicGraphResidency.WARM
            entry.hits = 0
        if entry.state == DynamicGraphResidency.HOT:
            return True
        entry.hits += 1
        if (
            entry.state == DynamicGraphResidency.WARM
            and entry.hits >= entry.min_hits
            and self._dynamic_pending is None
        ):
            entry.state = DynamicGraphResidency.QUEUED
            self._dynamic_pending = desc
            logger.debug(
                "Dynamic CUDA graph queued: descriptor=%s hits=%d epoch=%d",
                desc,
                entry.hits,
                self._dynamic_epoch,
            )
        return False

    def _runtime_batch_descriptor(
        self, desc: BatchExecutionDescriptor
    ) -> BatchDescriptor:
        return BatchDescriptor(
            num_tokens=desc.num_tokens,
            has_lora=desc.num_active_loras > 0,
            num_active_loras=desc.num_active_loras,
            tp3_owner_prequant_decode=self.uses_tp3_owner_prequant_decode(desc),
            cudagraph_owner=self.dynamic_graph_owner,
            physical_num_reqs=desc.physical_num_reqs,
            runtime_generation=desc.runtime_generation,
        )

    def uses_tp3_owner_prequant_decode(self, _desc: BatchExecutionDescriptor) -> bool:
        """Base Graph owners do not consume the target owner-prequant lane."""
        return False

    def _cooldown_dynamic_entry(self, entry: DynamicGraphEntry) -> None:
        if self.defer_startup_graphs:
            entry.state = DynamicGraphResidency.WARM
            entry.hits = 0
            entry.cooldown_until_epoch = 0
            return
        entry.state = DynamicGraphResidency.COOLDOWN
        entry.hits = 0
        entry.cooldown_until_epoch = (
            self._dynamic_epoch + self.dynamic_graph_cooldown_steps
        )

    def _dynamic_capture_state_bytes(self, _desc: BatchExecutionDescriptor) -> int:
        """Return live manager state whose addresses belong to a Graph."""
        return 0

    def _release_dynamic_capture_state(self, _desc: BatchExecutionDescriptor) -> None:
        """Drop live manager state owned by one evicted Graph descriptor."""

    def _destroy_dynamic_graphs(self, entry: DynamicGraphEntry) -> int:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        evicted = CUDAGraphWrapper.evict_batch_descriptor(
            self._runtime_batch_descriptor(entry.descriptor), entry.graph_pool
        )
        evicted += BreakableCUDAGraphWrapper.evict_batch_descriptor(
            self._runtime_batch_descriptor(entry.descriptor), entry.graph_pool
        )
        graph = self.graphs.pop(entry.descriptor, None)
        if graph is not None:
            physical_segments = int(getattr(graph, "num_graphs", 1))
            graph.reset()
            evicted += physical_segments
        # Reset the executable before releasing its external input owners.
        entry.capture_state = None
        self._release_dynamic_capture_state(entry.descriptor)
        # Drop the manager's last private-pool handle before the caller runs
        # gc/empty_cache and measures physical reclamation. Keeping this handle
        # on the entry makes the reclaim oracle retain part of the pool it is
        # trying to prove was released.
        entry.graph_pool = None
        return evicted

    @contextmanager
    def _use_dynamic_graph_pool(self, graph_pool: object):
        """Route every graph-producing layer to one descriptor-private pool."""
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        wrappers: list[CUDAGraphWrapper | BreakableCUDAGraphWrapper] = [
            *list(CUDAGraphWrapper._all_instances),
            *list(BreakableCUDAGraphWrapper._all_instances),
        ]
        original_wrapper_pools = {
            id(wrapper): wrapper.graph_pool for wrapper in wrappers
        }
        original_manager_pool = self.pool
        try:
            for wrapper in wrappers:
                wrapper.graph_pool = graph_pool
            self.pool = graph_pool
            yield
        finally:
            self.pool = original_manager_pool
            for wrapper in wrappers:
                wrapper.graph_pool = original_wrapper_pools[id(wrapper)]

    def _trim_cuda_graph_memory(self) -> None:
        """Return CUDA's unused graph-allocation cache after graph teardown."""
        _trim_device_graph_memory(self.device)

    @staticmethod
    def _dynamic_graph_pool_bytes(graph_pool: object | None) -> int:
        if graph_pool is None:
            return 0
        return sum(
            int(segment["total_size"])
            for segment in torch.cuda.memory_snapshot()
            if segment.get("segment_pool_id") == graph_pool
        )

    def _validate_dynamic_entry_eviction(
        self,
        entry: DynamicGraphEntry,
        *,
        transaction_id: str,
        reason: str,
        administrative: bool = False,
    ) -> None:
        if not transaction_id or not reason:
            raise RuntimeError("CUDA Graph eviction requires transaction provenance")
        if entry.pinned and not administrative:
            raise RuntimeError(
                "ordinary admission cannot evict a pinned FULL CUDA Graph: "
                f"owner={self.dynamic_graph_owner} descriptor={entry.descriptor}"
            )
        if entry.leases:
            raise RuntimeError(
                "cannot evict a leased CUDA Graph executable: "
                f"owner={self.dynamic_graph_owner} descriptor={entry.descriptor} "
                f"leases={sorted(entry.leases)}"
            )
        if entry.deferred_free:
            raise RuntimeError(
                "cannot evict a CUDA Graph with unfinished deferred work: "
                f"owner={self.dynamic_graph_owner} descriptor={entry.descriptor}"
            )
        if entry.state != DynamicGraphResidency.HOT:
            raise RuntimeError(
                "elastic eviction victim is not HOT: "
                f"owner={self.dynamic_graph_owner} descriptor={entry.descriptor}"
            )

    def _evict_dynamic_entry(
        self,
        entry: DynamicGraphEntry,
        *,
        transaction_id: str,
        reason: str,
        administrative: bool = False,
        defer_rank_consensus: bool = False,
    ) -> None:
        self._validate_dynamic_entry_eviction(
            entry,
            transaction_id=transaction_id,
            reason=reason,
            administrative=administrative,
        )
        if entry.state == DynamicGraphResidency.HOT:
            torch.accelerator.synchronize(self.device)
        released_capture_state = entry.local_capture_state_bytes
        free_before = torch.accelerator.get_memory_info()[0]
        evicted = self._destroy_dynamic_graphs(entry)
        if entry.graph_segments and evicted != entry.graph_segments:
            raise RuntimeError(
                "Dynamic CUDA graph eviction was incomplete: "
                f"descriptor={entry.descriptor}, expected={entry.graph_segments}, "
                f"evicted={evicted}"
            )
        self._trim_cuda_graph_memory()
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize(self.device)
        local_reclaimed = max(0, torch.accelerator.get_memory_info()[0] - free_before)
        local_pool_reclaimed = bool(
            local_reclaimed + (1 << 20) >= entry.local_pool_bytes
        )
        if not defer_rank_consensus:
            pool_reclaimed = torch.tensor(
                int(local_pool_reclaimed),
                dtype=torch.int32,
                device=self.device,
            )
            if self.tp_size > 1:
                torch.distributed.all_reduce(
                    pool_reclaimed,
                    op=torch.distributed.ReduceOp.MIN,
                    group=get_tp_group().device_group,
                )
            local_pool_reclaimed = bool(pool_reclaimed.item())
        if not local_pool_reclaimed:
            raise RuntimeError(
                "Dynamic CUDA graph eviction did not physically reclaim its "
                f"pool: owner={self.dynamic_graph_owner} "
                f"descriptor={entry.descriptor} local_charge="
                f"{entry.local_charged_bytes} local_pool={entry.local_pool_bytes} "
                f"local_reclaimed={local_reclaimed}"
            )
        retained_increment = self._account_dynamic_retained_bytes(
            entry.local_charged_bytes,
            local_reclaimed,
            publish=not defer_rank_consensus,
        )
        self._dynamic_retention_ledger.step_released_capture_state_bytes += (
            released_capture_state
        )
        self._dynamic_resident_bytes -= entry.charged_bytes
        entry.charged_bytes = 0
        entry.local_charged_bytes = 0
        entry.local_capture_state_bytes = 0
        entry.local_pool_bytes = 0
        entry.reclaimable_bytes = 0
        entry.graph_segments = 0
        entry.state = DynamicGraphResidency.WARM
        entry.hits = 0
        entry.runtime_num_reqs = None
        logger.debug(
            "Dynamic CUDA graph evicted: owner=%s descriptor=%s "
            "local_reclaimed_bytes=%d retained_cuda_bytes_delta=%d "
            "retained_cuda_bytes_total=%d resident_bytes=%d",
            self.dynamic_graph_owner,
            entry.descriptor,
            local_reclaimed,
            retained_increment,
            self._dynamic_retention_ledger.rank_safe_bytes,
            self.dynamic_resident_bytes,
        )

    def _account_dynamic_retained_bytes(
        self,
        local_charge: int,
        local_reclaimed: int,
        *,
        publish: bool = True,
    ) -> int:
        """Publish CUDA bytes that graph teardown could not return.

        The private graph pool has a separate fail-closed reclaim oracle. Any
        remainder here is a physically measured lower-layer residency delta.
        Accumulating it per rank and publishing the TP maximum keeps the KV
        boundary safe without inventing a configured reserve.
        """
        ledger = self._dynamic_retention_ledger
        ledger.local_bytes = max(
            0,
            ledger.local_bytes + local_charge - local_reclaimed,
        )
        if not publish:
            return ledger.local_bytes - ledger.rank_safe_bytes
        retained = torch.tensor(
            ledger.local_bytes,
            dtype=torch.int64,
            device=self.device,
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                retained,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
        previous = ledger.rank_safe_bytes
        ledger.rank_safe_bytes = int(retained.item())
        return ledger.rank_safe_bytes - previous

    def _publish_deferred_eviction_consensus(
        self, local_error: Exception | None
    ) -> None:
        """Publish one fixed-order result after rank-local idle evictions."""
        accepted = torch.tensor(int(local_error is None), dtype=torch.int32)
        retained = torch.tensor(
            self._dynamic_retention_ledger.local_bytes,
            dtype=torch.int64,
        )
        if self.tp_size > 1:
            cpu_group = get_tp_group().cpu_group
            torch.distributed.all_reduce(
                accepted,
                op=torch.distributed.ReduceOp.MIN,
                group=cpu_group,
            )
            torch.distributed.all_reduce(
                retained,
                op=torch.distributed.ReduceOp.MAX,
                group=cpu_group,
            )
        self._dynamic_retention_ledger.rank_safe_bytes = int(retained.item())
        if not bool(accepted.item()):
            raise RuntimeError(
                "administrative CUDA Graph eviction failed on at least one rank"
            ) from local_error

    def _reconcile_dynamic_retained_cleanup(
        self,
        local_reclaimed_bytes: int,
        *,
        force_rank_consensus: bool = False,
    ) -> int:
        """Publish delayed cleanup without consuming older retained memory.

        Graph eviction runs before attention wrappers are trimmed. Capture state
        is also explicitly dropped during eviction, but a small allocation in a
        shared allocator segment need not increase driver-free bytes. Both are
        graph-owned releases. Limit their credit to retention added since the
        current working-set step began, so unrelated or historical floors stay
        fail-closed.
        """
        if local_reclaimed_bytes < 0:
            raise ValueError("dynamic cleanup reclaim cannot be negative")
        ledger = self._dynamic_retention_ledger
        step_increase = max(0, ledger.local_bytes - ledger.step_start_local_bytes)
        if (
            step_increase == 0
            and local_reclaimed_bytes == 0
            and ledger.step_released_capture_state_bytes == 0
            and not force_rank_consensus
        ):
            # Steady HOT replay did not change the physical retention ledger.
            # Its already-published rank-safe value remains valid, so do not
            # put a control-plane collective (and host join) on every token.
            return 0
        cleanup_credit = min(
            step_increase,
            local_reclaimed_bytes + ledger.step_released_capture_state_bytes,
        )
        ledger.local_bytes -= cleanup_credit
        if not force_rank_consensus and self.tp_size > 1 and self.device.type == "cuda":
            # Captured model graphs may contain NCCL on their own streams.  A
            # retention collective submitted before those streams finish can
            # invert the global communicator order across ranks.  Retention
            # changes only on maintenance/capture/eviction, so establish the
            # consumer-complete boundary here, off the steady HOT path.
            torch.accelerator.synchronize(self.device)
        retained = torch.tensor(
            ledger.local_bytes,
            dtype=torch.int64,
            device="cpu" if force_rank_consensus else self.device,
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                retained,
                op=torch.distributed.ReduceOp.MAX,
                group=(
                    get_tp_group().cpu_group
                    if force_rank_consensus
                    else get_tp_group().device_group
                ),
            )
        previous = ledger.rank_safe_bytes
        ledger.rank_safe_bytes = int(retained.item())
        ledger.step_start_local_bytes = ledger.local_bytes
        ledger.step_released_capture_state_bytes = 0
        return ledger.rank_safe_bytes - previous

    def _reclaim_unpublished_capture(
        self,
        entry: DynamicGraphEntry,
        graph_segments: int,
        local_charge: int,
        local_pool_charge: int,
    ) -> None:
        free_before = torch.accelerator.get_memory_info()[0]
        evicted = self._destroy_dynamic_graphs(entry)
        if graph_segments and evicted != graph_segments:
            raise RuntimeError(
                "Unpublished dynamic CUDA graph cleanup was incomplete: "
                f"descriptor={entry.descriptor}, expected={graph_segments}, "
                f"evicted={evicted}"
            )
        self._trim_cuda_graph_memory()
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize(self.device)
        local_reclaimed = max(0, torch.accelerator.get_memory_info()[0] - free_before)
        pool_reclaimed = torch.tensor(
            int(local_reclaimed + (1 << 20) >= local_pool_charge),
            dtype=torch.int32,
            device=self.device,
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                pool_reclaimed,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
        if not bool(pool_reclaimed.item()):
            raise RuntimeError(
                "Unpublished dynamic CUDA graph did not physically reclaim "
                f"its pool: owner={self.dynamic_graph_owner} "
                f"descriptor={entry.descriptor} local_charge={local_charge} "
                f"local_pool={local_pool_charge} "
                f"local_reclaimed={local_reclaimed}"
            )
        retained_increment = self._account_dynamic_retained_bytes(
            local_charge,
            local_reclaimed,
        )
        logger.debug(
            "Unpublished dynamic CUDA graph reclaimed: owner=%s descriptor=%s "
            "local_charge=%d local_pool=%d local_reclaimed=%d "
            "retained_cuda_bytes_delta=%d retained_cuda_bytes_total=%d",
            self.dynamic_graph_owner,
            entry.descriptor,
            local_charge,
            local_pool_charge,
            local_reclaimed,
            retained_increment,
            self._dynamic_retention_ledger.rank_safe_bytes,
        )

    def _dynamic_capture_loan_bytes(self, entry: DynamicGraphEntry) -> int:
        resident_bytes = max(entry.estimated_bytes, 1)
        # The scheduler grant is the calibrated full-step cold envelope. Do not
        # add a guessed capture workspace here; that would recreate a reserve
        # and double-count the measured peak.
        return resident_bytes

    def _get_dynamic_graph_capture_context(self) -> GraphCaptureContext:
        context = self._dynamic_capture_state.context
        if context is None:
            context = GraphCaptureContext(torch.cuda.Stream(device=self.device))
            self._dynamic_capture_state.context = context
        return context

    def _admit_dynamic_capture(self, candidate: DynamicGraphEntry) -> bool:
        self.last_dynamic_capture_rejection = None
        required = self._dynamic_capture_loan_bytes(candidate)
        if required > self._dynamic_capture_granted_bytes:
            self.last_dynamic_capture_rejection = (
                "phase=loan required_bytes="
                f"{required} granted_bytes={self._dynamic_capture_granted_bytes}"
            )
            return False
        # The scheduler loan is physical capacity already unmapped from KV,
        # but free segments retained by PyTorch's caching allocator do not
        # appear in the driver-free counter below.  A falling wave can leave
        # exactly that state after evicting a larger graph family: the loan is
        # sufficient, yet admission falsely rejects a smaller recapture before
        # capture gets a chance to run.  Purge only free allocator segments;
        # live tensors and live graph pools remain allocated.
        torch.accelerator.synchronize(self.device)
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize(self.device)
        local_free = torch.accelerator.get_memory_info()[0]
        free_tensor = torch.tensor(local_free, dtype=torch.int64, device=self.device)
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                free_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
        rank_safe_free = int(free_tensor.item())
        if rank_safe_free < required:
            self.last_dynamic_capture_rejection = (
                "phase=physical_free required_bytes="
                f"{required} granted_bytes={self._dynamic_capture_granted_bytes} "
                f"rank_safe_free_bytes={rank_safe_free}"
            )
            return False
        return True

    def _piecewise_safety_size(self, num_tokens: int) -> int | None:
        return next(
            (
                size
                for size in self.dynamic_piecewise_safety_sizes
                if size >= num_tokens
            ),
            None,
        )

    def ensure_piecewise_safety_for_first_use(
        self,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
    ) -> bool:
        if (
            not self._graphs_captured
            or not self.dynamic_piecewise_safety_sizes
            or self.dynamic_piecewise_capture_range is None
            or num_tokens < self.dynamic_piecewise_capture_range[0]
        ):
            return False
        safety_size = self._piecewise_safety_size(num_tokens)
        if safety_size is None:
            raise RuntimeError(
                "PIECEWISE execution shape exceeds the dynamic safety boundary: "
                f"num_tokens={num_tokens} max_safety_size="
                f"{self.dynamic_piecewise_safety_sizes[-1]}"
            )
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.PIECEWISE,
            num_tokens=safety_size,
            num_reqs=None,
            num_active_loras=self._resolve_effective_loras(num_active_loras),
        )
        entry = self._dynamic_graph_entries[desc]
        if entry.state == DynamicGraphResidency.HOT:
            return False
        if entry.state == DynamicGraphResidency.COOLDOWN:
            raise RuntimeError(
                f"Required PIECEWISE safety graph is cooling down: descriptor={desc}"
            )
        if self._dynamic_pending is not None and self._dynamic_pending != desc:
            displaced = self._dynamic_graph_entries[self._dynamic_pending]
            displaced.state = DynamicGraphResidency.WARM
            displaced.hits = 0
        entry.state = DynamicGraphResidency.QUEUED
        entry.hits = max(entry.hits, 1)
        self._dynamic_pending = desc
        logger.debug(
            "Dynamic PIECEWISE safety graph queued before first use: descriptor=%s",
            desc,
        )
        return True

    def _consensus_dynamic_candidate(self, desc: BatchExecutionDescriptor) -> bool:
        group = get_tp_group()
        identity = (
            self.dynamic_graph_owner,
            int(desc.cg_mode.value),
            desc.num_tokens,
            desc.num_reqs,
            desc.uniform_token_count,
            desc.num_active_loras,
        )
        expected = group.broadcast_object(
            identity if group.is_first_rank else None,
            src=0,
        )
        matches = torch.tensor(
            int(identity == expected), dtype=torch.int32, device=self.device
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                matches,
                op=torch.distributed.ReduceOp.MIN,
                group=group.device_group,
            )
        return bool(matches.item())

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
        capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]]
        | None = None,
        capture_begin_hook: Callable[[BatchExecutionDescriptor], None] | None = None,
        capture_complete_hook: Callable[
            [BatchExecutionDescriptor, Callable[[CUDAGraphMode], None]], None
        ]
        | None = None,
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a forward_fn. For FULL and breakable PIECEWISE modes,
                it is invoked once with warmup=True and again with warmup=False
                because attention backends may mutate or lazily initialize
                metadata during warmup.
        """
        capture_context = (
            self._get_dynamic_graph_capture_context()
            if self.defer_startup_graphs
            else None
        )
        capture_scope = (
            graph_capture(device=self.device)
            if capture_context is None
            else graph_capture(
                device=self.device,
                graph_capture_context=capture_context,
            )
        )
        with capture_scope:
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            selected_descs = (
                self._capture_descs if capture_descs is None else capture_descs
            )
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in selected_descs:
                    continue

                descs = selected_descs[mode]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    logger.debug(
                        "CUDA Graph capture phase: owner=%s rank=%d "
                        "phase=prepare_warmup descriptor=%s physical_num_reqs=%d",
                        self.dynamic_graph_owner,
                        self.vllm_config.parallel_config.rank,
                        desc,
                        self._capture_num_reqs(desc),
                    )
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc, warmup=True)

                    # Warmup
                    logger.debug(
                        "CUDA Graph capture phase: owner=%s rank=%d "
                        "phase=warmup_forward_begin descriptor=%s",
                        self.dynamic_graph_owner,
                        self.vllm_config.parallel_config.rank,
                        desc,
                    )
                    forward_fn(CUDAGraphMode.NONE)
                    torch.accelerator.synchronize(self.device)
                    logger.debug(
                        "CUDA Graph capture phase: owner=%s rank=%d "
                        "phase=warmup_forward_end descriptor=%s",
                        self.dynamic_graph_owner,
                        self.vllm_config.parallel_config.rank,
                        desc,
                    )

                    # Capture
                    logger.debug(
                        "CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc
                    )
                    if (
                        desc.cg_mode == CUDAGraphMode.PIECEWISE
                        and not self.use_breakable_cg
                    ):
                        if capture_begin_hook is not None:
                            capture_begin_hook(desc)
                        forward_fn(CUDAGraphMode.PIECEWISE)
                        if capture_complete_hook is not None:
                            capture_complete_hook(desc, forward_fn)
                    else:
                        # Capture with fresh attention state.
                        logger.debug(
                            "CUDA Graph capture phase: owner=%s rank=%d "
                            "phase=prepare_fresh descriptor=%s",
                            self.dynamic_graph_owner,
                            self.vllm_config.parallel_config.rank,
                            desc,
                        )
                        forward_fn = create_forward_fn(desc, warmup=False)
                        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                            if capture_begin_hook is not None:
                                capture_begin_hook(desc)
                            forward_fn(CUDAGraphMode.PIECEWISE)
                            if capture_complete_hook is not None:
                                capture_complete_hook(desc, forward_fn)
                            continue
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        prefix_diagnostic_graph = (
                            (
                                self._p3_node_cutoff is not None
                                or bool(self._p3_node_prefix_counts)
                            )
                            and progress_bar_desc == "Capturing CUDA graphs"
                            and desc.uniform_token_count == 3
                            and desc.num_tokens == 6
                        )
                        debug_p3_graph = (
                            (self._p3_graph_debug or prefix_diagnostic_graph)
                            and desc.uniform_token_count == 3
                            and desc.num_tokens == 6
                        )
                        native_config = self.vllm_config.additional_config
                        breakable_native_full = bool(
                            self.use_breakable_cg
                            and self.dynamic_graph_owner == "target"
                            and isinstance(native_config, dict)
                            and native_config.get("flashnext_native_experts")
                        )
                        if breakable_native_full:
                            import os

                            if (
                                os.environ.get("AG2_FLASHNEXT_QSA_COMMAND_CAPTURE")
                                == "1"
                            ):
                                from vllm.models.qwen4_exp.nvidia import (
                                    qsa_command_capture,
                                )

                                capture_type: Any = (
                                    qsa_command_capture.QSACommandCapture
                                )
                            else:
                                from vllm.compilation.breakable_cudagraph import (
                                    BreakableCUDAGraphCapture,
                                )

                                capture_type = BreakableCUDAGraphCapture
                            graph = capture_type(pool=self.pool)
                        else:
                            graph = torch.cuda.CUDAGraph(
                                keep_graph=(
                                    debug_p3_graph
                                    or bool(
                                        getattr(
                                            self,
                                            "_standalone_keep_full_graph",
                                            False,
                                        )
                                    )
                                )
                            )
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        if self.pool is not None:
                            set_graph_pool_id(self.pool)
                        else:
                            set_graph_pool_id(current_platform.graph_pool_handle())
                        if capture_begin_hook is not None:
                            capture_begin_hook(desc)
                        logger.debug(
                            "CUDA Graph capture phase: owner=%s rank=%d "
                            "phase=capture_begin descriptor=%s",
                            self.dynamic_graph_owner,
                            self.vllm_config.parallel_config.rank,
                            desc,
                        )
                        # Native CPU/GPU experts contain host work between GPU
                        # segments. A monolithic CUDA Graph freezes that work at
                        # capture time, so native target FULL retains callbacks.
                        active_capture_scope: Any = (
                            graph
                            if breakable_native_full
                            else torch.cuda.graph(
                                graph,
                                self.pool,
                                stream=torch.cuda.current_stream(),
                            )
                        )
                        with active_capture_scope:
                            forward_fn(CUDAGraphMode.NONE)
                            # Join offloader's copy stream after forward to avoid
                            # unjoined stream error. The last layer's start_prefetch
                            # forks copy_stream, but wait_prefetch only happens in
                            # the next forward pass.
                            get_offloader().join_after_forward()
                        torch.accelerator.synchronize(self.device)
                        logger.debug(
                            "CUDA Graph capture phase: owner=%s rank=%d "
                            "phase=capture_end descriptor=%s",
                            self.dynamic_graph_owner,
                            self.vllm_config.parallel_config.rank,
                            desc,
                        )
                        self.graphs[desc] = graph
                        if capture_complete_hook is not None:
                            capture_complete_hook(desc, forward_fn)
                        if debug_p3_graph:
                            graph_role = (
                                progress_bar_desc.lower()
                                .replace(" ", "-")
                                .replace("/", "-")
                            )
                            dump_path = (
                                "/workspace/eval-results/"
                                f"p3-cuda-graph-{graph_role}-pid{os.getpid()}-"
                                f"tokens{desc.num_tokens}-reqs{desc.num_reqs}.dot"
                            )
                            try:
                                cudart = ctypes.CDLL("libcudart.so.13")
                                dot_print = cudart.cudaGraphDebugDotPrint
                                dot_print.argtypes = [
                                    ctypes.c_void_p,
                                    ctypes.c_char_p,
                                    ctypes.c_uint,
                                ]
                                dot_print.restype = ctypes.c_int
                                result = dot_print(
                                    ctypes.c_void_p(graph.raw_cuda_graph()),
                                    dump_path.encode(),
                                    ctypes.c_uint(1),  # Verbose.
                                )
                                if result != 0:
                                    raise RuntimeError(
                                        "cudaGraphDebugDotPrint failed with "
                                        f"cudaError_t={result}"
                                    )
                                logger.warning(
                                    "P3 CUDA graph debug dump: desc=%s path=%s",
                                    desc,
                                    dump_path,
                                )
                                if prefix_diagnostic_graph:
                                    from cuda.bindings import runtime as cuda_runtime

                                    with open(
                                        dump_path,
                                        encoding="utf-8",
                                    ) as dot_file:
                                        dot_contents = dot_file.read()
                                    topo_by_local_id = {
                                        int(local_id): int(topo_id)
                                        for local_id, topo_id in re.findall(
                                            r"(?<![A-Za-z])(\d+) "
                                            r"\(topoId: (\d+)\)",
                                            dot_contents,
                                        )
                                    }
                                    graph.instantiate()
                                    raw_graph = cuda_runtime.cudaGraph_t(
                                        graph.raw_cuda_graph()
                                    )
                                    raw_graph_exec = cuda_runtime.cudaGraphExec_t(
                                        graph.raw_cuda_graph_exec()
                                    )
                                    error, _, num_nodes = (
                                        cuda_runtime.cudaGraphGetNodes(raw_graph)
                                    )
                                    if int(error) != 0:
                                        raise RuntimeError(
                                            f"cudaGraphGetNodes(size) failed: {error}"
                                        )
                                    error, nodes, actual_nodes = (
                                        cuda_runtime.cudaGraphGetNodes(
                                            raw_graph, num_nodes
                                        )
                                    )
                                    if int(error) != 0 or actual_nodes != num_nodes:
                                        raise RuntimeError(
                                            "cudaGraphGetNodes failed: "
                                            f"error={error}, expected={num_nodes}, "
                                            f"actual={actual_nodes}"
                                        )
                                    supported_types = {
                                        cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel,
                                        cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy,
                                        cuda_runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset,
                                    }
                                    diagnostic_nodes: list[tuple[int, Any]] = []
                                    if self._p3_node_prefix_counts:
                                        if (
                                            self._p3_node_prefix_counts[-1]
                                            > actual_nodes
                                        ):
                                            raise ValueError(
                                                "p3_node_prefix_counts exceeds "
                                                f"target graph size {actual_nodes}: "
                                                f"{self._p3_node_prefix_counts}"
                                            )
                                        active_cutoff = (
                                            actual_nodes
                                            - self._p3_node_prefix_counts[0]
                                        )
                                    else:
                                        assert self._p3_node_cutoff is not None
                                        if self._p3_node_cutoff > actual_nodes:
                                            raise ValueError(
                                                "p3_node_cutoff exceeds target "
                                                f"graph size {actual_nodes}: "
                                                f"{self._p3_node_cutoff}"
                                            )
                                        active_cutoff = self._p3_node_cutoff
                                    disabled = 0
                                    for node in nodes[:actual_nodes]:
                                        error, local_id = (
                                            cuda_runtime.cudaGraphNodeGetLocalId(node)
                                        )
                                        if int(error) != 0:
                                            raise RuntimeError(
                                                "cudaGraphNodeGetLocalId failed: "
                                                f"{error}"
                                            )
                                        error, node_type = (
                                            cuda_runtime.cudaGraphNodeGetType(node)
                                        )
                                        if int(error) != 0:
                                            raise RuntimeError(
                                                f"cudaGraphNodeGetType failed: {error}"
                                            )
                                        topo_id = topo_by_local_id.get(int(local_id))
                                        if topo_id is None:
                                            raise RuntimeError(
                                                "DOT is missing CUDA graph node "
                                                f"local_id={int(local_id)}"
                                            )
                                        if node_type in supported_types:
                                            diagnostic_nodes.append((topo_id, node))
                                        if (
                                            topo_id < active_cutoff
                                            and node_type in supported_types
                                        ):
                                            result = (
                                                cuda_runtime.cudaGraphNodeSetEnabled(
                                                    raw_graph_exec, node, 0
                                                )
                                            )
                                            if int(result[0]) != 0:
                                                raise RuntimeError(
                                                    "cudaGraphNodeSetEnabled failed: "
                                                    f"local_id={int(local_id)}, "
                                                    f"topo_id={topo_id}, "
                                                    f"error={result[0]}"
                                                )
                                            disabled += 1
                                    logger.warning(
                                        "P3 CUDA graph prefix diagnostic armed: "
                                        "desc=%s cutoff=%d prefix_counts=%s "
                                        "disabled=%d/%d",
                                        desc,
                                        active_cutoff,
                                        self._p3_node_prefix_counts or None,
                                        disabled,
                                        actual_nodes,
                                    )
                                    self._p3_prefix_diagnostic_descs.add(desc)
                                    self._p3_prefix_diagnostic_state[desc] = (
                                        raw_graph_exec,
                                        diagnostic_nodes,
                                        actual_nodes,
                                    )
                            except Exception:
                                logger.exception(
                                    "P3 CUDA graph debug dump failed: desc=%s path=%s",
                                    desc,
                                    dump_path,
                                )
                                if prefix_diagnostic_graph:
                                    raise
                        compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True

    def captured_token_counts(self) -> list[int]:
        """Sorted token counts with a captured graph, ignoring LoRA variants."""
        return sorted(
            {desc.num_tokens for desc in self.graphs if desc.num_active_loras == 0}
        )

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""

        effective_loras = self._resolve_effective_loras(num_active_loras)
        if self.defer_startup_graphs:
            if not self._dynamic_step_planned:
                raise RuntimeError(
                    "elastic CUDA Graph dispatch reached execution before "
                    f"same-step planning: owner={self.dynamic_graph_owner}"
                )
            bounded_decode_candidate = any(
                desc.semantic_decode
                and desc.physical_num_reqs is not None
                and desc.physical_num_reqs >= num_reqs
                and desc.num_tokens == desc.physical_num_reqs * self.decode_query_len
                and uniform_token_count == self.decode_query_len
                and num_tokens == num_reqs * self.decode_query_len
                for desc in self._dynamic_step_candidates
            )
            compiled_piecewise_dispatch = self.is_compiled_piecewise_shape(
                num_tokens,
                num_reqs=num_reqs,
                semantic_decode=bounded_decode_candidate,
            ) and (
                not self._dynamic_step_candidates
                or all(
                    desc.cg_mode == CUDAGraphMode.PIECEWISE
                    for desc in self._dynamic_step_candidates
                )
            )
            if compiled_piecewise_dispatch:
                # A scheduler-owned maintenance step may have captured and
                # queued the canonical exact bucket before the following user
                # step reveals its actual token tail. Runtime dispatch is the
                # final authority: never consume that PIECEWISE candidate for
                # an underfilled invocation. FULL candidates are deliberately
                # excluded above and retain their exact-X contract.
                token_bucket = elastic_piecewise_token_boundary(
                    num_tokens,
                    self.vllm_config.scheduler_config.max_num_batched_tokens,
                )
                if token_bucket not in self._compiled_piecewise_dispatch_logged:
                    logger.debug(
                        "Elastic %s compiled-only PIECEWISE dispatch: "
                        "tokens=%d token_bucket=%d physical_reqs=%d "
                        "outer_compile=required cuda_graph=disabled",
                        self.dynamic_graph_owner,
                        num_tokens,
                        token_bucket,
                        num_reqs,
                    )
                    self._compiled_piecewise_dispatch_logged.add(token_bucket)
                desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=token_bucket,
                    num_reqs=None,
                    uniform_token_count=(
                        uniform_token_count if bounded_decode_candidate else None
                    ),
                    num_active_loras=effective_loras,
                    physical_num_reqs=num_reqs,
                    runtime_generation=self.runtime_generation,
                    semantic_decode=bounded_decode_candidate,
                )
                self._last_dynamic_dispatch = desc
                return desc
            matches = [
                desc
                for desc in self._dynamic_step_candidates
                if self._dynamic_descriptor_matches_step(
                    desc,
                    num_reqs=num_reqs,
                    num_tokens=num_tokens,
                    uniform_token_count=uniform_token_count,
                    num_active_loras=num_active_loras,
                    max_query_len=max_query_len,
                )
            ]
            if not matches:
                raise RuntimeError(
                    "elastic product dispatch has no exact HOT CUDA Graph; "
                    "NONE/eager fallback is forbidden: "
                    f"owner={self.dynamic_graph_owner} requests={num_reqs} "
                    f"tokens={num_tokens} uniform={uniform_token_count}"
                )
            if len(matches) != 1:
                raise RuntimeError(
                    "runtime CUDA Graph planning produced ambiguous descriptors: "
                    f"owner={self.dynamic_graph_owner} matches={matches}"
                )
            desc = matches[0]
            entry = self._dynamic_graph_entries[desc]
            if entry.state != DynamicGraphResidency.HOT:
                raise RuntimeError(
                    "runtime CUDA Graph was not captured before execution: "
                    f"owner={self.dynamic_graph_owner} desc={desc} "
                    f"state={entry.state.value}"
                )
            self._last_dynamic_dispatch = desc
            return desc

        safety_size = self._piecewise_safety_size(num_tokens)
        if (
            self._graphs_captured
            and uniform_token_count is None
            and self.dynamic_piecewise_capture_range is not None
            and self.dynamic_piecewise_capture_range[0]
            <= num_tokens
            <= self.dynamic_piecewise_capture_range[1]
            and num_tokens not in self.dynamic_piecewise_static_sizes
            and safety_size is not None
            and (safety_size - num_tokens) * 100
            >= num_tokens * self.dynamic_graph_piecewise_min_padding_pct
        ):
            dynamic_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.PIECEWISE,
                num_tokens=num_tokens,
                num_reqs=None,
                num_active_loras=effective_loras,
            )
            self._dynamic_graph_entries.setdefault(
                dynamic_desc,
                DynamicGraphEntry(
                    descriptor=dynamic_desc,
                    min_hits=self.dynamic_graph_piecewise_min_hits,
                ),
            )
            if self._observe_dynamic_descriptor(dynamic_desc):
                self._last_dynamic_dispatch = dynamic_desc
                return dynamic_desc
        key = (num_tokens, effective_loras)
        if self._graphs_captured and num_tokens > 0 and key in self._candidates:
            for desc in self._candidates[key]:
                if _is_compatible(
                    desc,
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    effective_loras,
                    max_query_len,
                    num_ubatches,
                ):
                    if desc in self._dynamic_graph_entries:
                        if not self._dynamic_descriptor_matches_step(
                            desc,
                            num_reqs=num_reqs,
                            num_tokens=num_tokens,
                            uniform_token_count=uniform_token_count,
                            num_active_loras=num_active_loras,
                            max_query_len=max_query_len,
                        ):
                            continue
                        if not self._observe_dynamic_descriptor(desc):
                            continue
                        self._last_dynamic_dispatch = desc
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=effective_loras,
            num_ubatches=num_ubatches,
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """Replay a captured FULL cudagraph."""
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        diagnose_p3_graph = self._p3_crash_diagnostic and desc.uniform_token_count == 3
        sync_p3_graph = self._p3_global_sync and desc.uniform_token_count == 3
        if sync_p3_graph:
            torch.accelerator.synchronize(self.device)
        if diagnose_p3_graph:
            logger.warning("P3 CUDA graph replay begin: desc=%s", desc)
        prefix_diagnostic = desc in self._p3_prefix_diagnostic_descs
        if prefix_diagnostic:
            if self._p3_node_prefix_counts:
                from cuda.bindings import runtime as cuda_runtime

                raw_graph_exec, diagnostic_nodes, total_nodes = (
                    self._p3_prefix_diagnostic_state[desc]
                )
                for prefix_count in self._p3_node_prefix_counts:
                    active_cutoff = total_nodes - prefix_count
                    for topo_id, node in diagnostic_nodes:
                        result = cuda_runtime.cudaGraphNodeSetEnabled(
                            raw_graph_exec,
                            node,
                            int(topo_id >= active_cutoff),
                        )
                        if int(result[0]) != 0:
                            raise RuntimeError(
                                "cudaGraphNodeSetEnabled failed during "
                                "progressive replay: "
                                f"topo_id={topo_id}, cutoff={active_cutoff}, "
                                f"error={result[0]}"
                            )
                    logger.warning(
                        "P3 CUDA graph progressive prefix begin: "
                        "desc=%s prefix_nodes=%d cutoff=%d",
                        desc,
                        prefix_count,
                        active_cutoff,
                    )
                    self.graphs[desc].replay()
                    torch.accelerator.synchronize(self.device)
                    logger.warning(
                        "P3 CUDA graph progressive prefix passed: "
                        "desc=%s prefix_nodes=%d cutoff=%d",
                        desc,
                        prefix_count,
                        active_cutoff,
                    )
            else:
                self.graphs[desc].replay()
                torch.accelerator.synchronize(self.device)
            raise RuntimeError(
                "P3 CUDA graph prefix diagnostic completed without a CUDA "
                "fault: "
                f"cutoff={self._p3_node_cutoff}, "
                f"prefix_counts={self._p3_node_prefix_counts or None}, "
                f"desc={desc}"
            )
        self.graphs[desc].replay()
        if sync_p3_graph:
            torch.accelerator.synchronize(self.device)
        if diagnose_p3_graph:
            logger.warning("P3 CUDA graph replay end: desc=%s", desc)

    def run_dynamic_capture_state_direct_control(
        self, desc: BatchExecutionDescriptor
    ) -> Any:
        """Execute a HOT entry's retained static closure without replay.

        This is a standalone economics control, not a product fallback.
        Callers must first resolve the original production descriptor; this
        method neither dispatches nor manufactures a NONE descriptor.
        """
        entry = self._dynamic_capture_state_direct_entry(desc)
        assert entry.capture_state is not None
        return entry.capture_state(CUDAGraphMode.NONE)

    def _dynamic_capture_state_direct_entry(
        self, desc: BatchExecutionDescriptor
    ) -> DynamicGraphEntry:
        """Validate and return the retained entry for the direct control."""
        entry = self._dynamic_graph_entries.get(desc)
        if entry is None:
            raise RuntimeError(
                "direct capture-state control has no dynamic entry: "
                f"owner={self.dynamic_graph_owner} descriptor={desc}"
            )
        if entry.state != DynamicGraphResidency.HOT:
            raise RuntimeError(
                "direct capture-state control requires a HOT entry: "
                f"owner={self.dynamic_graph_owner} descriptor={desc} "
                f"state={entry.state.value}"
            )
        if entry.capture_state is None:
            raise RuntimeError(
                "direct capture-state control requires a retained closure: "
                f"owner={self.dynamic_graph_owner} descriptor={desc}"
            )
        return entry

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableCUDAGraphWrapper(
                model, self.vllm_config
            )

    def run_pw_graph(self, model: nn.Module, model_inputs: dict[str, Any]) -> Any:
        if not self.use_breakable_cg:
            # Default: Use torch-compiled piecewise cudagraph.
            return model(**model_inputs)
        assert self.breakable_cg_runner is not None
        return self.breakable_cg_runner(**model_inputs)


class ModelCudaGraphManager(CudaGraphManager):
    """CudaGraphManager with model-specific capture and hidden state management."""

    _max_full_descs_to_capture: int
    _capture_mem_samples: list[int]

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        full_decode_query_lens: set[int] | None = None,
        full_decode_cap_query_lens: set[int] | None = None,
        full_decode_max_tokens: int | None = None,
        tp3_sd_phase_reduce: bool = False,
        tp3_owner_prequant: bool = False,
        max_uniform_decode_reqs: int | None = None,
        owner: str = "target",
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            full_decode_query_lens=full_decode_query_lens,
            full_decode_cap_query_lens=full_decode_cap_query_lens,
            full_decode_max_tokens=full_decode_max_tokens,
            max_uniform_decode_reqs=max_uniform_decode_reqs,
            owner=owner,
        )
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.aux_hidden_states_token_major: list[bool] = []
        self.use_aux_hidden_state_outputs = False
        self.intermediate_tensors: IntermediateTensors | None = None
        self.tp3_sd_phase_reduce = tp3_sd_phase_reduce
        self.tp3_owner_prequant = tp3_owner_prequant
        self._tp3_owner_replay_receipts: set[tuple[int, int | None, int | None]] = set()

    def uses_tp3_owner_prequant_decode(self, desc: BatchExecutionDescriptor) -> bool:
        """Resolve the graph-captured owner-prequant lane.

        The compiled-only carrier restores the previously accepted reducer
        math.  Owner-prequant is a separate captured representation and may
        not leak into that path merely because semantic decode geometry is the
        same.
        """
        return bool(
            self.tp3_owner_prequant
            and desc.cg_mode != CUDAGraphMode.NONE
            and desc.semantic_decode
            and desc.uniform_token_count == self.decode_query_len
            and desc.num_tokens >= envs.AG2_VLLM_TP3_OWNER_MIN_ROWS
        )

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _dynamic_capture_state_bytes(self, desc: BatchExecutionDescriptor) -> int:
        entry = getattr(self, "_dynamic_graph_entries", {}).get(desc)
        bundle = entry.lifetime_bundle if entry is not None else None
        state = bundle or self
        total = 0
        if state.hidden_states is not None:
            total += self._tensor_bytes(state.hidden_states)
        total += sum(self._tensor_bytes(tensor) for tensor in state.aux_hidden_states)
        if state.intermediate_tensors is not None:
            total += sum(
                self._tensor_bytes(tensor)
                for tensor in state.intermediate_tensors.tensors.values()
            )
        return total

    def _release_dynamic_capture_state(self, desc: BatchExecutionDescriptor) -> None:
        entry = getattr(self, "_dynamic_graph_entries", {}).get(desc)
        if entry is not None:
            entry.lifetime_bundle = None
            return
        self.hidden_states = None
        self.aux_hidden_states.clear()
        self.aux_hidden_states_token_major.clear()
        self.intermediate_tensors = None

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        pcp_manager: "PCPManager | None" = None,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]]
        | None = None,
        capture_begin_hook: Callable[[BatchExecutionDescriptor], None] | None = None,
        capture_complete_hook: Callable[
            [BatchExecutionDescriptor, Callable[[CUDAGraphMode], None]], None
        ]
        | None = None,
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        owner_capture_descs: set[tuple[int, int | None, int | None]] = set()
        if self.use_breakable_cg:
            self.init_breakable_cg_runner(model)

        if self.cudagraph_mode.has_piecewise_cudagraphs() and not (
            self.use_breakable_cg or has_compiled_submodule(model)
        ):
            raise RuntimeError(
                f"{type(model).__name__}: piecewise CUDA graphs "
                f"(cudagraph_mode={self.cudagraph_mode.name}) unavailable, "
                "model is not torch-compiled and breakable CUDA graph is off. "
                "Set VLLM_USE_BREAKABLE_CUDAGRAPH=1 or cudagraph_mode=NONE/FULL."
            )

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = self._capture_num_reqs(desc)
            capture_outputs: DynamicModelCaptureBundle | ModelCudaGraphManager = self
            if self.defer_startup_graphs:
                entry = self._dynamic_graph_entries[desc]
                if entry.lifetime_bundle is None:
                    entry.lifetime_bundle = DynamicModelCaptureBundle()
                capture_outputs = entry.lifetime_bundle

            # Set LoRA state before capture so kernels see correct adapters.
            if lora_capture_hook is not None:
                lora_capture_hook(desc.num_active_loras, num_reqs, num_tokens)

            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )

            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            from vllm.model_executor.models.qwen3_next_ready import (
                bind_ready_rows,
                ready_compaction_enabled,
            )

            model_inputs = bind_ready_rows(
                model_inputs,
                enabled=ready_compaction_enabled(self.vllm_config),
                actual_rows=num_tokens,
                physical_rows=num_tokens,
                full_graph=True,
            )
            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None
                assert intermediate_tensors is not None
                model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]

            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
                max_query_len=desc.max_query_len,
                pcp_manager=pcp_manager,
            )

            if envs.AG2_VLLM_NVFP4_MARLIN_ISOLATE_PREFILL:
                layout = input_buffers.marlin_request_layout_cpu
                if desc.cg_mode == CUDAGraphMode.FULL:
                    # FULL graphs are replayed only for uniform decode lanes.
                    # Capture the original single batched Marlin launch.
                    layout[0] = num_reqs
                    layout[1] = num_reqs
                    layout[2 : num_reqs + 3].zero_()
                else:
                    # The op is cudagraph-unsafe in PIECEWISE mode and is
                    # executed dynamically. This valid one-request layout is
                    # only for compile/capture warmup.
                    layout[0] = 1
                    layout[1] = 0
                    layout[2] = 0
                    layout[3] = num_tokens

            # Capture with dummy rows marked as padding.
            input_buffers.is_padding.fill_(True)

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                # FULL graph capture calls this closure with cg_mode NONE.
                # Use the target manager's static role plus the descriptor,
                # never graph mode, to mark the qlen1 target-decode lane.
                tp3_sd_phase_reduce = (
                    self.tp3_sd_phase_reduce
                    and desc.cg_mode == CUDAGraphMode.FULL
                    and desc.uniform_token_count == 1
                )
                tp3_owner_prequant_decode = self.uses_tp3_owner_prequant_decode(desc)
                if tp3_owner_prequant_decode:
                    owner_capture_descs.add(
                        (desc.num_tokens, desc.num_reqs, desc.uniform_token_count)
                    )
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                        tp3_sd_phase_reduce=tp3_sd_phase_reduce,
                        tp3_owner_prequant_decode=tp3_owner_prequant_decode,
                        cudagraph_owner=self.dynamic_graph_owner,
                        physical_num_reqs=desc.physical_num_reqs,
                        runtime_generation=desc.runtime_generation,
                    )
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                    is_padding=input_buffers.is_padding[:num_tokens],
                    tp3_sd_phase_reduce=tp3_sd_phase_reduce,
                    tp3_owner_prequant_decode=tp3_owner_prequant_decode,
                    marlin_request_layout_cpu=(
                        input_buffers.marlin_request_layout_cpu
                        if envs.AG2_VLLM_NVFP4_MARLIN_ISOLATE_PREFILL
                        else None
                    ),
                ):
                    if cg_mode == CUDAGraphMode.PIECEWISE:
                        # PIECEWISE graph (compiled PW or breakable, chosen inside
                        # run_pw_graph).
                        model_output = self.run_pw_graph(model, model_inputs)
                    else:
                        model_output = model(**model_inputs)

                # Capture invokes the retained closure outside execute_model's
                # normal completion seam.  E8 host sequencing belongs to each
                # invocation (warmup and capture), not to the whole capture call.
                finish_capture_forward = getattr(
                    model_state, "finish_native_capture_forward", None
                )
                if finish_capture_forward is not None:
                    finish_capture_forward()

                if cg_mode == CUDAGraphMode.PIECEWISE:
                    # PW CUDA graph (compiled or breakable) internally handles the
                    # model outputs. No need to keep track of the hidden states.
                    return None

                if self.is_last_pp_rank:
                    # Last PP rank (common case).
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        aux_hidden_states = []
                    if capture_outputs.hidden_states is None:
                        capture_tokens = (
                            num_tokens
                            if self.defer_startup_graphs
                            else self.max_capture_tokens
                        )
                        capture_outputs.hidden_states = torch.empty(
                            (capture_tokens, *hidden_states.shape[1:]),
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                    capture_outputs.hidden_states[:num_tokens] = hidden_states
                    if (
                        self.use_aux_hidden_state_outputs
                        and not capture_outputs.aux_hidden_states
                    ):
                        capture_tokens = (
                            num_tokens
                            if self.defer_startup_graphs
                            else self.max_capture_tokens
                        )
                        capture_outputs.aux_hidden_states = [
                            (
                                torch.empty(
                                    (capture_tokens, *x.shape[1:]),
                                    dtype=x.dtype,
                                    device=x.device,
                                )
                                if x.ndim > 0 and x.shape[0] == num_tokens
                                else torch.empty_like(x)
                            )
                            for x in aux_hidden_states
                        ]
                        capture_outputs.aux_hidden_states_token_major = [
                            aux.ndim > 0 and aux.shape[0] == num_tokens
                            for aux in aux_hidden_states
                        ]
                    for destination, aux, token_major in zip(
                        capture_outputs.aux_hidden_states,
                        aux_hidden_states,
                        capture_outputs.aux_hidden_states_token_major,
                        strict=True,
                    ):
                        copy_aux_hidden_state(
                            destination,
                            aux,
                            num_tokens=num_tokens,
                            token_major=token_major,
                        )
                else:
                    # Non-last PP rank.
                    assert isinstance(model_output, IntermediateTensors)
                    intermediate_tensors = model_output
                    if capture_outputs.intermediate_tensors is None:
                        capture_outputs.intermediate_tensors = (
                            IntermediateTensors.empty_like(intermediate_tensors)
                        )
                    for k, v in intermediate_tensors.tensors.items():
                        capture_outputs.intermediate_tensors[k][:num_tokens] = v

            def allocation_profile_replay_context():
                # Allocation profiling replays a retained FULL descriptor
                # outside execute_model. Recreate the capture-time context so
                # every eager boundary (QSA, GDN and future callbacks) sees the
                # same complete contract rather than adding callback-specific
                # no-context fallbacks.
                tp3_sd_phase_reduce = (
                    self.tp3_sd_phase_reduce
                    and desc.cg_mode == CUDAGraphMode.FULL
                    and desc.uniform_token_count == 1
                )
                tp3_owner_prequant_decode = self.uses_tp3_owner_prequant_decode(desc)
                return set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=None,
                    is_padding=input_buffers.is_padding[:num_tokens],
                    tp3_sd_phase_reduce=tp3_sd_phase_reduce,
                    tp3_owner_prequant_decode=tp3_owner_prequant_decode,
                    marlin_request_layout_cpu=(
                        input_buffers.marlin_request_layout_cpu
                        if envs.AG2_VLLM_NVFP4_MARLIN_ISOLATE_PREFILL
                        else None
                    ),
                )

            cast(
                Any, forward_fn
            ).allocation_profile_replay_context = allocation_profile_replay_context

            return forward_fn

        try:
            super().capture(
                create_forward_fn,
                progress_bar_desc,
                capture_descs,
                capture_begin_hook=capture_begin_hook,
                capture_complete_hook=capture_complete_hook,
            )
        finally:
            # These closures bypass execute_model's normal completion seam.
            # Their transient read lease must not outlive capture/rollback.
            finish_native_capture = getattr(model_state, "finish_native_capture", None)
            if finish_native_capture is not None:
                finish_native_capture()
        if self.tp3_owner_prequant:
            logger.warning(
                "TP3 owner prequant CUDA Graph capture completed: descriptors=%s",
                sorted(owner_capture_descs),
            )

    def capture_next_dynamic(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        capture_override: Callable[
            [
                dict[CUDAGraphMode, list[BatchExecutionDescriptor]],
                Callable[
                    [BatchExecutionDescriptor, Callable[[CUDAGraphMode], None]], None
                ],
            ],
            None,
        ]
        | None = None,
    ) -> bool:
        desc = self._dynamic_pending
        if desc is None:
            return False
        entry = self._dynamic_graph_entries[desc]
        logger.debug(
            "Dynamic CUDA graph promotion phase: owner=%s rank=%d "
            "phase=consensus_begin descriptor=%s",
            self.dynamic_graph_owner,
            self.vllm_config.parallel_config.rank,
            desc,
        )
        if not self._consensus_dynamic_candidate(desc):
            self.last_dynamic_capture_rejection = (
                f"phase=rank_consensus descriptor={desc!r}"
            )
            self._dynamic_pending = None
            self._dynamic_capture_granted_bytes = 0
            self._cooldown_dynamic_entry(entry)
            logger.error(
                "Dynamic CUDA graph rank descriptor mismatch: descriptor=%s", desc
            )
            return False
        logger.debug(
            "Dynamic CUDA graph promotion phase: owner=%s rank=%d "
            "phase=consensus_end descriptor=%s",
            self.dynamic_graph_owner,
            self.vllm_config.parallel_config.rank,
            desc,
        )
        logger.debug(
            "Dynamic CUDA graph promotion phase: owner=%s rank=%d "
            "phase=admission_begin descriptor=%s granted_bytes=%d",
            self.dynamic_graph_owner,
            self.vllm_config.parallel_config.rank,
            desc,
            self._dynamic_capture_granted_bytes,
        )
        if not self._admit_dynamic_capture(entry):
            self._dynamic_pending = None
            granted_bytes = self._dynamic_capture_granted_bytes
            self._dynamic_capture_granted_bytes = 0
            self._cooldown_dynamic_entry(entry)
            logger.warning(
                "Dynamic CUDA graph same-step loan refused capture: "
                "descriptor=%s resident_bytes=%d granted_bytes=%d",
                desc,
                self._dynamic_resident_bytes,
                granted_bytes,
            )
            return False
        logger.debug(
            "Dynamic CUDA graph promotion phase: owner=%s rank=%d "
            "phase=admission_end descriptor=%s granted_bytes=%d",
            self.dynamic_graph_owner,
            self.vllm_config.parallel_config.rank,
            desc,
            self._dynamic_capture_granted_bytes,
        )

        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        # The private-pool scope snapshots existing wrappers. A fresh manager
        # must register its wrapper before that snapshot, including when this
        # entrypoint is used by the drafter's capture override.
        if self.use_breakable_cg:
            self.init_breakable_cg_runner(model)
        runtime_desc = self._runtime_batch_descriptor(desc)
        entry.graph_pool = current_platform.graph_pool_handle()
        start_free: int | None = None
        capture_state_bytes = 0

        def record_capture_baseline(
            capture_desc: BatchExecutionDescriptor,
        ) -> None:
            nonlocal start_free
            if capture_desc != desc:
                raise RuntimeError(
                    "Dynamic CUDA graph capture baseline descriptor mismatch: "
                    f"expected={desc}, actual={capture_desc}"
                )
            # Measure before building the closure: its attention metadata and
            # slot mappings are address-stable Graph residency, not baseline.
            # Keep the first baseline across warmup + fresh preparation: the
            # end-to-end physical delta then includes every retained byte and
            # excludes disposable state that has died by publication.
            torch.accelerator.synchronize(self.device)
            gc.collect()
            torch.accelerator.empty_cache()
            torch.accelerator.synchronize(self.device)
            start_free = torch.accelerator.get_memory_info()[0]

        def retain_capture_state(
            capture_desc: BatchExecutionDescriptor,
            forward_fn: Callable[[CUDAGraphMode], None],
        ) -> None:
            if capture_desc != desc:
                raise RuntimeError(
                    "Dynamic CUDA graph capture-state descriptor mismatch: "
                    f"expected={desc}, actual={capture_desc}"
                )
            entry.capture_state = forward_fn

        try:
            record_capture_baseline(desc)
            with self._use_dynamic_graph_pool(entry.graph_pool):
                capture_descs = {desc.cg_mode: [desc]}
                if capture_override is None:
                    self.capture(
                        model,
                        model_state,
                        input_buffers,
                        intermediate_tensors,
                        block_tables,
                        attn_groups,
                        kv_cache_config,
                        has_lora=has_lora,
                        use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                        lora_capture_hook=lora_capture_hook,
                        progress_bar_desc="Promoting dynamic CUDA graph",
                        capture_descs=capture_descs,
                        capture_complete_hook=retain_capture_state,
                    )
                else:
                    capture_override(capture_descs, retain_capture_state)
                torch.accelerator.synchronize(self.device)
        except Exception:
            self._destroy_dynamic_graphs(entry)
            gc.collect()
            torch.accelerator.empty_cache()
            torch.accelerator.synchronize(self.device)
            self._dynamic_pending = None
            self._dynamic_capture_granted_bytes = 0
            self._cooldown_dynamic_entry(entry)
            logger.exception("Dynamic CUDA graph capture failed: descriptor=%s", desc)
            raise

        if start_free is None:
            raise RuntimeError(
                f"Dynamic CUDA graph capture baseline was not recorded: {desc}"
            )
        if entry.capture_state is None:
            raise RuntimeError(
                f"Dynamic CUDA graph capture state was not retained: {desc}"
            )
        capture_state_bytes = self._dynamic_capture_state_bytes(desc)
        logger.debug(
            "Dynamic CUDA graph promotion phase: owner=%s rank=%d "
            "phase=publication_begin descriptor=%s",
            self.dynamic_graph_owner,
            self.vllm_config.parallel_config.rank,
            desc,
        )
        graph_segments = (
            CUDAGraphWrapper.count_batch_descriptor(runtime_desc, entry.graph_pool)
            + BreakableCUDAGraphWrapper.count_batch_descriptor(
                runtime_desc, entry.graph_pool
            )
            + (
                int(getattr(self.graphs[desc], "num_graphs", 1))
                if desc in self.graphs
                else 0
            )
        )
        end_free = torch.accelerator.get_memory_info()[0]
        local_capture_delta = max(0, start_free - end_free)
        local_pool_charge = self._dynamic_graph_pool_bytes(entry.graph_pool)
        # The elastic contract charges every physical byte created by capture,
        # not only the private pool. If a default-pool allocation cannot be
        # reclaimed with the graph, X1 KV restoration must fail closed rather
        # than turn that allocation into an implicit permanent reserve.
        # Conversely, allocator reuse can make the driver-free delta smaller
        # than the exact private pool now owned by this executable. The pool is
        # still unavailable to KV and therefore remains a physical charge.
        # The baseline precedes closure construction, so capture_state_bytes
        # is already included in the physical delta. Keep it separately only
        # for release observability; adding it again would double-charge KV.
        local_charge = dynamic_capture_physical_charge(
            local_capture_delta,
            local_pool_charge,
        )
        local_non_pool_delta = max(0, local_charge - local_pool_charge)
        charge = torch.tensor(local_charge, dtype=torch.int64, device=self.device)
        reclaimable = torch.tensor(
            max(0, local_pool_charge - (1 << 20)),
            dtype=torch.int64,
            device=self.device,
        )
        segment_count = torch.tensor(
            graph_segments, dtype=torch.int32, device=self.device
        )
        if self.tp_size > 1:
            torch.distributed.all_reduce(
                charge,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
            torch.distributed.all_reduce(
                reclaimable,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
            torch.distributed.all_reduce(
                segment_count,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
        charged_bytes = int(charge.item())
        reclaimable_bytes = int(reclaimable.item())
        graph_segments = int(segment_count.item())
        if not dynamic_capture_publication_allowed(
            graph_segments,
            charged_bytes,
            self._dynamic_capture_granted_bytes,
        ):
            self._reclaim_unpublished_capture(
                entry,
                graph_segments,
                local_charge,
                local_pool_charge,
            )
            self._dynamic_pending = None
            granted_bytes = self._dynamic_capture_granted_bytes
            self.last_dynamic_capture_rejection = (
                "phase=publication "
                f"segments={graph_segments} charged_bytes={charged_bytes} "
                f"granted_bytes={granted_bytes}"
            )
            self._dynamic_capture_granted_bytes = 0
            self._cooldown_dynamic_entry(entry)
            logger.error(
                "Dynamic CUDA graph publication rejected: descriptor=%s "
                "segments=%d charged_bytes=%d granted_bytes=%d",
                desc,
                graph_segments,
                charged_bytes,
                granted_bytes,
            )
            return False
        get_tp_group().barrier()
        entry.state = DynamicGraphResidency.HOT
        entry.estimated_bytes = charged_bytes
        entry.charged_bytes = charged_bytes
        entry.local_charged_bytes = local_charge
        entry.local_capture_state_bytes = capture_state_bytes
        entry.local_pool_bytes = local_pool_charge
        entry.reclaimable_bytes = reclaimable_bytes
        entry.graph_segments = graph_segments
        self._dynamic_resident_bytes += charged_bytes
        self.last_dynamic_capture_rejection = None
        self._dynamic_pending = None
        granted_bytes = self._dynamic_capture_granted_bytes
        self._dynamic_capture_granted_bytes = 0
        logger.debug(
            "Dynamic CUDA graph HOT: owner=%s descriptor=%s segments=%d "
            "charged_bytes=%d capture_delta_bytes=%d capture_state_bytes=%d "
            "non_pool_delta_bytes=%d "
            "resident_bytes=%d granted_bytes=%d",
            self.dynamic_graph_owner,
            desc,
            graph_segments,
            charged_bytes,
            local_capture_delta,
            capture_state_bytes,
            local_non_pool_delta,
            self.dynamic_resident_bytes,
            granted_bytes,
        )
        return True

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Replay a captured FULL cudagraph and return hidden states."""
        super().run_fullgraph(desc)
        receipt = (desc.num_tokens, desc.num_reqs, desc.uniform_token_count)
        if (
            self.uses_tp3_owner_prequant_decode(desc)
            and receipt not in self._tp3_owner_replay_receipts
        ):
            self._tp3_owner_replay_receipts.add(receipt)
            logger.warning(
                "TP3 owner prequant CUDA Graph replay observed: descriptor=%s",
                receipt,
            )
        return self._dynamic_capture_output(desc)

    def allocation_profile_replay_context(self, desc: BatchExecutionDescriptor):
        """Restore the complete capture-time context for request-free replay."""
        entry = self._dynamic_capture_state_direct_entry(desc)
        if entry.capture_state is None:
            raise RuntimeError("allocation profile lacks retained capture state")
        factory = getattr(
            entry.capture_state, "allocation_profile_replay_context", None
        )
        if factory is None:
            raise RuntimeError("allocation profile capture omitted its replay context")
        return factory()

    def run_dynamic_capture_state_direct_control(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Execute a HOT entry's retained static closure without Graph replay.

        This is a standalone economics control, not a product fallback. It
        preserves the capture-time model inputs, attention metadata, slot
        mappings and output bundle while changing only the inner execution to
        direct torch-compiled ``NONE``. Product dispatch must still resolve the
        original FULL/PIECEWISE descriptor before a caller may use it.
        """
        entry = self._dynamic_capture_state_direct_entry(desc)
        if entry.lifetime_bundle is None:
            raise RuntimeError(
                "model direct capture-state control requires its output bundle: "
                f"owner={self.dynamic_graph_owner} descriptor={desc}"
            )
        assert entry.capture_state is not None
        entry.capture_state(CUDAGraphMode.NONE)
        return self._dynamic_capture_output(desc)

    def _dynamic_capture_output(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Return the address-stable output bundle owned by ``desc``."""
        entry = self._dynamic_graph_entries.get(desc)
        capture_outputs = (
            entry.lifetime_bundle
            if entry is not None and entry.lifetime_bundle is not None
            else self
        )
        if not self.is_last_pp_rank:
            assert capture_outputs.intermediate_tensors is not None
            return capture_outputs.intermediate_tensors[: desc.num_tokens]

        assert capture_outputs.hidden_states is not None
        hidden_states = capture_outputs.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [
            view_aux_hidden_state(
                tensor,
                num_tokens=desc.num_tokens,
                token_major=token_major,
            )
            for tensor, token_major in zip(
                capture_outputs.aux_hidden_states,
                capture_outputs.aux_hidden_states_token_major,
                strict=True,
            )
        ]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
    max_query_len: int | None = None,
    pcp_manager: "PCPManager | None" = None,
) -> AttentionState:
    input_batch = InputBatch.make_dummy(
        num_reqs, num_tokens, input_buffers, max_query_len=max_query_len
    )
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mapping_provider: BlockTables | PCPManager = block_tables
    if pcp_manager is not None:
        slot_mapping_provider = pcp_manager
    slot_mappings = slot_mapping_provider.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    # NOTE(woosuk): Attention metadata is required not just by standard attention
    # kernels, but also by specialized attention-like operations (e.g., Inkling's sconv,
    # DSV4 compressor), which maintain their own states and require special metadata
    # such as block tables.
    # During CUDA graph capture:
    # - For FULL CUDA graphs: We set for_capture=True so that both attention and
    #   attention-like ops produce capturable metadata compatible with CUDA graphs.
    # - For PIECEWISE CUDA graphs: We still build attention metadata, but set
    #   for_capture=False. This is because:
    #     * Attention-like ops (such as sconv or DSV4 compressor) may not be used as
    #       breakpoints in PIECEWISE CUDA graphs, so we must generate their attention
    #       metadata so they can execute and be captured during graph capture.
    #     * Standard attention ops that are treated as breakpoints will be executed
    #       eagerly at capture time (not included in the graph itself), and for these,
    #       setting for_capture=False is essential. Some attention backends
    #       (like linear attention) cannot generate capturable metadata for prefill,
    #       so for_capture=False ensures they execute without issue.
    #     * We assume that attention-like operations intended for capture will still
    #       produce capturable metadata, even when for_capture=False. While this
    #       assumption is brittle, it currently works in practice.
    # In summary: We always generate attention metadata for both FULL and PIECEWISE
    # CUDA graphs, setting for_capture=True for FULL graphs, and for_capture=False
    # for PIECEWISE graphs, to ensure correct execution and capture.
    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.FULL if full_cudagraph else CUDAGraphMode.PIECEWISE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


# ---------------------------------------------------------------------------
# CUDA graph memory profiling
# ---------------------------------------------------------------------------


def _trim_device_graph_memory(device: torch.device) -> None:
    """Release driver graph memory after every executable/pool owner is gone."""
    from cuda.bindings import runtime as cuda_runtime

    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    result = cuda_runtime.cudaDeviceGraphMemTrim(device_index)
    if int(result[0]) != 0:
        raise RuntimeError(
            "cudaDeviceGraphMemTrim failed after CUDA Graph teardown: "
            f"device={device_index} error={result[0]}"
        )


# Number of FULL graphs captured during profiling; the total FULL capture
# cost is extrapolated from this sample to avoid a second full capture.
_FULL_GRAPH_PROFILING_SAMPLES = 2
# Floor for the extrapolated per-graph cost (driver overhead per graph).
_MIN_PER_GRAPH_BYTES = 1 << 20


@torch.inference_mode()
def profile_cudagraph_memory(runner: "GPUModelRunner") -> int:
    """Estimate the GPU memory needed for CUDA graph capture.

    Called during memory profiling, *before* the real KV cache is allocated,
    so that ``Worker.determine_available_memory`` can reserve headroom for
    graph capture. Bootstraps a minimal KV cache, runs ``capture_model()``
    once, then releases everything so the real init/capture path starts clean.

    FULL graphs bake in KV cache pointers, so only the largest few are
    captured (into a throwaway pool) and their total cost is extrapolated.
    PIECEWISE, encoder and speculator graphs are measured in full. All
    profiling captures are discarded afterwards: replaying graphs recorded
    against the throwaway profiling state is unsafe (e.g. inductor graph
    partition reclaims the storages of earlier cudagraph recordings once the
    real capture records new ones, leading to use-after-free crashes).
    """
    if runner.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        return 0

    gc.collect()
    torch.accelerator.empty_cache()

    # Run the whole profiling phase against a throwaway CUDA graph pool by
    # pointing the global graph pool singleton at it: objects that bind the
    # pool lazily during profiling (speculator cudagraph managers, breakable
    # runners created mid-capture) then land on the throwaway pool too. Pools
    # bound before profiling (piecewise wrappers) are swapped explicitly in
    # the inner block. Profiling graphs captured into the persistent global
    # pool and then discarded would drop its use_count to 0, tripping the c10
    # allocator's create_or_incref_pool assert when the real capture reuses
    # that pool ("use_count > 0 INTERNAL ASSERT FAILED").
    platform_cls = type(current_platform)
    saved_global_pool = platform_cls._global_graph_pool
    throwaway_pool = current_platform.graph_pool_handle()
    platform_cls._global_graph_pool = throwaway_pool

    try:
        with set_current_vllm_config(runner.vllm_config):
            _init_minimal_kv_cache_for_profiling(runner)

        manager = runner.cudagraph_manager
        assert manager is not None

        # Don't count profiling captures; the real capture_model() runs later.
        saved_num_cudagraph_captured = compilation_counter.num_cudagraph_captured
        saved_capture_triggers = compilation_counter.num_gpu_runner_capture_triggers
        all_wrappers: list[Any] = []
        original_pools: dict[int, Any] = {}
        speculator = getattr(runner, "speculator", None)
        spec_manager_names: list[str] = []
        try:
            if not manager.needs_capture():
                return 0
            manager.pool = throwaway_pool
            if manager.use_breakable_cg:
                # Create the breakable runner before the wrapper pool swap so
                # its pool is covered as well.
                manager.init_breakable_cg_runner(runner.model)
            all_wrappers = list(CUDAGraphWrapper._all_instances) + list(
                BreakableCUDAGraphWrapper._all_instances
            )
            for wrapper in all_wrappers:
                original_pools[id(wrapper)] = wrapper.graph_pool
                wrapper.graph_pool = throwaway_pool
            if speculator is not None:
                spec_manager_names = [
                    name
                    for name, value in vars(speculator).items()
                    if isinstance(value, CudaGraphManager)
                ]
            manager._max_full_descs_to_capture = _FULL_GRAPH_PROFILING_SAMPLES
            mem_samples: list[int] = []
            manager._capture_mem_samples = mem_samples

            measured = int(runner.capture_model(profile_only=True))

            # The measured delta covers PIECEWISE, encoder and speculator graphs
            # plus the sampled FULL graphs; swap the sampled FULL cost for the
            # extrapolated total. FULL and PIECEWISE share one pool here just as
            # they share the global pool at runtime, so the overlap is not
            # double-counted.
            num_full_graphs = len(manager._capture_descs.get(CUDAGraphMode.FULL, []))
            full_estimate = _extrapolate_full_graph_memory(mem_samples, num_full_graphs)
            return max(measured - sum(mem_samples) + full_estimate, 0)
        finally:
            compilation_counter.num_cudagraph_captured = saved_num_cudagraph_captured
            compilation_counter.num_gpu_runner_capture_triggers = saved_capture_triggers
            CUDAGraphWrapper.clear_all_graphs()
            BreakableCUDAGraphWrapper.clear_all_graphs()
            for wrapper in all_wrappers:
                if id(wrapper) in original_pools:
                    wrapper.graph_pool = original_pools[id(wrapper)]
            # Drop the speculator's cudagraph managers; the real
            # initialize_kv_cache re-creates them. Their profiling graphs
            # release the throwaway pool here rather than after the real init.
            for name in spec_manager_names:
                setattr(speculator, name, None)
            # Drop local references before teardown detaches the runner's
            # manager and flushes the allocator.
            del manager
            _teardown_profiling_state(runner)
    finally:
        platform_cls._global_graph_pool = saved_global_pool
        # CUDA can retain physical pages in its device graph cache even after
        # PyTorch graph objects and allocator segments have been released.
        # Drop the final throwaway-pool handle before trimming that cache; the
        # elastic KV custom MemPools cannot borrow those pages otherwise.
        throwaway_pool = None
        gc.collect()
        torch.accelerator.empty_cache()
        free_before_trim = torch.accelerator.get_memory_info()[0]
        _trim_device_graph_memory(runner.device)
        free_after_trim = torch.accelerator.get_memory_info()[0]
        logger.info(
            "Released throwaway CUDA Graph epoch before KV sizing: %.2f MiB",
            max(free_after_trim - free_before_trim, 0) / (1 << 20),
        )


def _extrapolate_full_graph_memory(mem_samples: list[int], total_graphs: int) -> int:
    """Extrapolate the total FULL capture cost from samples of the largest
    graphs. The first capture allocates the pool baseline; later graphs mostly
    reuse it, so the second sample is taken as the per-graph cost."""
    if not mem_samples:
        return 0
    first_capture = mem_samples[0]
    per_graph = max(mem_samples[1], _MIN_PER_GRAPH_BYTES) if len(mem_samples) > 1 else 0
    return first_capture + (total_graphs - 1) * per_graph


def _init_minimal_kv_cache_for_profiling(runner: "GPUModelRunner") -> None:
    """Allocate the smallest KV cache that still lets every graph be captured."""
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    kv_cache_spec = runner.get_kv_cache_spec()
    kv_cache_groups = get_kv_cache_groups(runner.vllm_config, kv_cache_spec)
    # At least one block per sequence is required to capture the graphs.
    min_blocks = (
        min(runner.max_num_reqs, runner.compilation_config.max_cudagraph_capture_size)
        or 1
    )
    saved_override = runner.cache_config.num_gpu_blocks_override
    runner.cache_config.num_gpu_blocks_override = min_blocks
    try:
        minimal_config = get_kv_cache_config_from_groups(
            runner.vllm_config, kv_cache_groups, available_memory=0
        )
    finally:
        runner.cache_config.num_gpu_blocks_override = saved_override

    runner.initialize_kv_cache(minimal_config, is_profiling=True)
    runner.cache_config.num_gpu_blocks = minimal_config.num_blocks


def _teardown_profiling_state(runner: "GPUModelRunner") -> None:
    """Release the profiling KV cache and captured graphs while keeping model
    weights, so the real ``initialize_kv_cache`` starts from a clean slate."""
    torch.accelerator.synchronize()
    if hasattr(runner.model_state, "_mamba_ctx"):
        runner.model_state._mamba_ctx = None
    # Invalidate the align-mode Mamba group metadata cached from the
    # profiling KVCacheConfig: the real (e.g. PP-projected) config may
    # place Mamba layers into a different group layout, so it must be
    # re-derived from the real config.
    if hasattr(runner.model_state, "_mamba_group_ids"):
        runner.model_state._mamba_group_ids = []
    if hasattr(runner.model_state, "_mamba_spec"):
        runner.model_state._mamba_spec = None
    # Separate-pool GDN selects attention-only views for block CoW.  That list
    # is a second owner of the profiling KV tensors, not an alias of
    # ``kv_caches``.  Drop it before clearing the primary list or the complete
    # temporary attention cache survives into production KV construction.
    if hasattr(runner, "kv_caches_for_block_copy"):
        if runner.kv_caches_for_block_copy is not getattr(runner, "kv_caches", None):
            runner.kv_caches_for_block_copy.clear()
        del runner.kv_caches_for_block_copy
    if hasattr(runner, "kv_caches"):
        runner.kv_caches.clear()
    if hasattr(runner, "attn_groups"):
        runner.attn_groups.clear()
    # Draft builders are independently constructed by set_attn. Keeping them
    # here pins the profiling scratch until final KV initialization, after the
    # production KV budget has already been published.
    speculator = getattr(runner, "speculator", None)
    if speculator is not None and hasattr(speculator, "attn_groups"):
        speculator.attn_groups.clear()
    if hasattr(runner, "kv_cache_config"):
        del runner.kv_cache_config
    # Dropping the manager releases the profiling graphs and throwaway pool.
    runner.cudagraph_manager = None
    # Release encoder graphs captured during profiling; the real
    # capture_model() re-captures them.
    if runner.model_state.supports_mm_inputs:
        runner.model_state.encoder_runner.clear()
    # Detach profiling KV tensors held by attention layers. The layers live
    # in the static forward context for compiled models.
    layers: Iterable[Any] = runner.compilation_config.static_forward_context.values()
    if (model := getattr(runner, "model", None)) is not None:
        layers = itertools.chain(layers, model.modules())
    clear_layer_kv_caches(layers)
    runner.cache_config.num_gpu_blocks = None
    runner.maybe_remove_all_loras(runner.lora_config)
    gc.collect()
    torch.accelerator.empty_cache()
