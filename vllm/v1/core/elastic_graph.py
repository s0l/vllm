# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure admission model for elastic CUDA Graph residency.

This module intentionally has no torch or CUDA dependency.  Scheduler and
worker code exchange the immutable objects defined here instead of deriving
physical graph identity or eviction policy independently.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from functools import cached_property
from typing import Any


class ElasticGraphError(RuntimeError):
    """An elastic graph state transition violated the admission contract."""


class GraphResidency(str, Enum):
    COLD = "cold"
    ADMISSION_PENDING = "admission_pending"
    CAPTURING = "capturing"
    HOT_EVICTABLE = "hot_evictable"
    HOT_PINNED = "hot_pinned"
    COOLDOWN = "cooldown"


class ElasticPlanKind(str, Enum):
    USER = "user"
    MAINTENANCE = "maintenance"
    RECLAIM = "reclaim"
    DEFER = "defer"


@dataclass(frozen=True, order=True)
class RuntimeGeneration:
    value: str

    def __post_init__(self) -> None:
        if not self.value or any(char.isspace() for char in self.value):
            raise ValueError("runtime generation must be a non-empty token")


@dataclass(frozen=True, order=True)
class SemanticGraphStep:
    """Model invocation geometry before any CUDA Graph representation choice."""

    num_spec_tokens: int
    num_reqs: int
    num_tokens: int
    uniform_query_len: int | None
    phase: str
    active_owners: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.num_spec_tokens < 0 or self.num_reqs <= 0 or self.num_tokens <= 0:
            raise ValueError("semantic Graph step has invalid K/X/M")
        if self.uniform_query_len is not None and self.uniform_query_len <= 0:
            raise ValueError("semantic uniform query length must be positive")
        if self.phase not in {"decode", "mixed"}:
            raise ValueError("semantic Graph phase must be decode or mixed")
        if not self.active_owners or self.active_owners[0] != "target":
            raise ValueError("semantic Graph step requires target as first owner")
        if len(set(self.active_owners)) != len(self.active_owners):
            raise ValueError("semantic Graph owners must be unique")
        if self.phase == "decode" and self.uniform_query_len is None:
            raise ValueError("uniform decode must retain its semantic query length")


GRAPH_EXECUTION_POLICY_SCHEMA = 1


@dataclass(frozen=True, order=True)
class OwnerGraphExecutionPolicy:
    """Effective CUDA Graph representations for one runtime owner."""

    owner: str
    full_query_lens: tuple[int, ...]
    piecewise_mode: str
    compiled_piecewise_sizes: tuple[int, ...] = ()
    capability_contract: str = "cuda-graph-manager-v1"
    full_exact_x: bool = True
    piecewise_padding_contract: str = "exact-batched-decode-m-v1"
    piecewise_query_len_min_tokens: int | None = None
    # Runtime-published semantic-to-physical shape contract.  ``legacy-v1``
    # exists only so old sealed/test payloads can still be inspected; newly
    # constructed worker policies must publish an explicit contract.
    activation: str = "legacy-v1"
    token_source: str = "legacy-v1"
    fixed_query_len: int | None = None
    execution_order: int | None = None

    def __post_init__(self) -> None:
        if not self.owner:
            raise ValueError("graph execution policy owner is required")
        if not self.capability_contract or not self.piecewise_padding_contract:
            raise ValueError("graph owner policy requires capability contracts")
        if self.piecewise_padding_contract not in {
            "exact-batched-decode-m-v1",
            "power-of-two-terminal-exact-v1",
        }:
            raise ValueError("unknown PIECEWISE padding contract")
        if self.full_query_lens and not self.full_exact_x:
            raise ValueError("FULL Graph policy must preserve exact X")
        if self.piecewise_mode not in {"PIECEWISE", "NONE"}:
            raise ValueError("piecewise mode must be PIECEWISE or NONE")
        if self.activation not in {"legacy-v1", "always", "speculative"}:
            raise ValueError("unknown Graph owner activation contract")
        if self.token_source not in {
            "legacy-v1",
            "step",
            "requests",
            "fixed_query",
        }:
            raise ValueError("unknown Graph owner token-source contract")
        if self.token_source == "fixed_query":
            if (
                isinstance(self.fixed_query_len, bool)
                or not isinstance(self.fixed_query_len, int)
                or self.fixed_query_len <= 0
            ):
                raise ValueError(
                    "fixed-query Graph owner requires a positive query length"
                )
        elif self.fixed_query_len is not None:
            raise ValueError(
                "only a fixed-query Graph owner may publish fixed_query_len"
            )
        if self.execution_order is not None and (
            isinstance(self.execution_order, bool)
            or not isinstance(self.execution_order, int)
            or self.execution_order < 0
        ):
            raise ValueError("Graph owner execution order must be non-negative")
        if (
            self.piecewise_query_len_min_tokens is not None
            and (
                isinstance(self.piecewise_query_len_min_tokens, bool)
                or not isinstance(self.piecewise_query_len_min_tokens, int)
                or self.piecewise_query_len_min_tokens <= 0
            )
        ):
            raise ValueError(
                "PIECEWISE query-length threshold must be a positive integer"
            )
        if (
            tuple(sorted(set(self.full_query_lens))) != self.full_query_lens
            or any(query_len <= 0 for query_len in self.full_query_lens)
        ):
            raise ValueError("FULL query lengths must be sorted unique positives")
        if (
            tuple(sorted(set(self.compiled_piecewise_sizes)))
            != self.compiled_piecewise_sizes
            or any(size <= 0 for size in self.compiled_piecewise_sizes)
        ):
            raise ValueError(
                "compiled PIECEWISE sizes must be sorted unique positives"
            )

    def mode_for(self, uniform_query_len: int | None) -> str:
        if uniform_query_len in self.full_query_lens:
            return "FULL"
        return self.piecewise_mode

    def validate_physical_key(self, key: PhysicalReplayKey) -> None:
        if key.logical.owner != self.owner:
            raise ElasticGraphError(
                "physical key owner differs from Graph execution policy"
            )
        mode = key.logical.mode
        if mode == "FULL":
            query_len = key.logical.uniform_query_len
            if query_len not in self.full_query_lens:
                raise ElasticGraphError(
                    "physical FULL key violates graph execution policy: "
                    f"owner={self.owner} query_len={query_len} "
                    f"allowed={self.full_query_lens}"
                )
            if key.logical.logical_num_reqs != key.physical_num_reqs:
                raise ElasticGraphError("FULL key lost exact request cardinality")
            expected_tokens = key.physical_num_reqs * query_len
            if key.logical.token_bucket != expected_tokens:
                raise ElasticGraphError("FULL key lost exact token cardinality")
            return
        if mode != self.piecewise_mode or mode != "PIECEWISE":
            raise ElasticGraphError(
                "physical key violates owner Graph representation policy: "
                f"owner={self.owner} mode={mode} "
                f"piecewise={self.piecewise_mode}"
            )
        if key.logical.logical_num_reqs is not None:
            raise ElasticGraphError("PIECEWISE key cannot claim logical requests")
        if key.logical.uniform_query_len is not None and (
            self.piecewise_query_len_min_tokens is None
            or key.logical.token_bucket < self.piecewise_query_len_min_tokens
        ):
            raise ElasticGraphError(
                "PIECEWISE key carries an inactive semantic query length: "
                f"owner={self.owner} tokens={key.logical.token_bucket} "
                f"minimum={self.piecewise_query_len_min_tokens}"
            )


@dataclass(frozen=True)
class GraphExecutionPolicy:
    """All-rank authority for scheduler and worker Graph representation."""

    verifier_contract: str
    math_contract: str
    owners: tuple[OwnerGraphExecutionPolicy, ...]
    verifier_configuration: str = "unspecified-v1"
    schema: int = GRAPH_EXECUTION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GRAPH_EXECUTION_POLICY_SCHEMA:
            raise ValueError("unsupported graph execution policy schema")
        if (
            not self.verifier_contract
            or not self.math_contract
            or not self.verifier_configuration
        ):
            raise ValueError("graph policy requires verifier and math contracts")
        owner_names = tuple(policy.owner for policy in self.owners)
        if tuple(sorted(owner_names)) != owner_names or len(set(owner_names)) != len(
            owner_names
        ):
            raise ValueError("graph execution policy owners must be sorted unique")
        if "target" not in owner_names:
            raise ValueError("graph execution policy requires a target owner")
        runtime_owners = tuple(
            owner for owner in self.owners if owner.activation != "legacy-v1"
        )
        if runtime_owners:
            orders = tuple(owner.execution_order for owner in runtime_owners)
            if any(order is None for order in orders) or len(set(orders)) != len(
                orders
            ):
                raise ValueError(
                    "runtime Graph owners require unique execution order"
                )

    @cached_property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    def owner_policy(self, owner: str) -> OwnerGraphExecutionPolicy:
        for policy in self.owners:
            if policy.owner == owner:
                return policy
        raise ElasticGraphError(f"graph execution policy has no owner {owner!r}")

    def mode_for(self, owner: str, uniform_query_len: int | None) -> str:
        return self.owner_policy(owner).mode_for(uniform_query_len)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GraphExecutionPolicy:
        if not isinstance(payload, Mapping):
            raise ValueError("graph execution policy payload must be a mapping")

        def required_string(source: Mapping[str, Any], name: str) -> str:
            value = source.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"graph execution policy {name} must be a non-empty string"
                )
            return value

        def integer_sequence(source: Mapping[str, Any], name: str) -> tuple[int, ...]:
            value = source.get(name)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(
                    f"graph execution policy {name} must be an integer sequence"
                )
            if any(
                isinstance(item, bool) or not isinstance(item, int) for item in value
            ):
                raise ValueError(
                    f"graph execution policy {name} must contain only integers"
                )
            return tuple(value)

        schema = payload.get("schema")
        if isinstance(schema, bool) or not isinstance(schema, int):
            raise ValueError("graph execution policy schema must be an integer")
        raw_owners = payload.get("owners")
        if not isinstance(raw_owners, Sequence) or isinstance(
            raw_owners, (str, bytes)
        ):
            raise ValueError("graph execution policy owners must be a sequence")
        owners = []
        for raw in raw_owners:
            if not isinstance(raw, Mapping):
                raise ValueError("graph execution policy owner must be a mapping")
            full_exact_x = raw.get("full_exact_x")
            if not isinstance(full_exact_x, bool):
                raise ValueError(
                    "graph execution policy full_exact_x must be a boolean"
                )
            query_len_min_tokens = raw.get("piecewise_query_len_min_tokens")
            if (
                query_len_min_tokens is not None
                and (
                    isinstance(query_len_min_tokens, bool)
                    or not isinstance(query_len_min_tokens, int)
                )
            ):
                raise ValueError(
                    "graph execution policy piecewise_query_len_min_tokens "
                    "must be an integer or null"
                )
            owners.append(
                OwnerGraphExecutionPolicy(
                    owner=required_string(raw, "owner"),
                    full_query_lens=integer_sequence(raw, "full_query_lens"),
                    piecewise_mode=required_string(raw, "piecewise_mode"),
                    compiled_piecewise_sizes=integer_sequence(
                        raw, "compiled_piecewise_sizes"
                    ),
                    capability_contract=required_string(
                        raw, "capability_contract"
                    ),
                    full_exact_x=full_exact_x,
                    piecewise_padding_contract=required_string(
                        raw, "piecewise_padding_contract"
                    ),
                    piecewise_query_len_min_tokens=query_len_min_tokens,
                    activation=raw.get("activation", "legacy-v1"),
                    token_source=raw.get("token_source", "legacy-v1"),
                    fixed_query_len=raw.get("fixed_query_len"),
                    execution_order=raw.get("execution_order"),
                )
            )
        policy = cls(
            schema=schema,
            verifier_contract=required_string(payload, "verifier_contract"),
            math_contract=required_string(payload, "math_contract"),
            owners=tuple(owners),
            verifier_configuration=required_string(
                payload, "verifier_configuration"
            ),
        )
        claimed = payload.get("fingerprint")
        if claimed is not None and not isinstance(claimed, str):
            raise ValueError("graph execution policy fingerprint must be a string")
        if claimed is not None and claimed != policy.fingerprint:
            raise ValueError("graph execution policy fingerprint mismatch")
        return policy

    def validate_physical_key(self, key: PhysicalReplayKey) -> None:
        owner_policy = self.owner_policy(key.logical.owner)
        owner_policy.validate_physical_key(key)


def bind_runtime_generation_to_policy(
    generation: str, policy: GraphExecutionPolicy
) -> str:
    """Bind an existing source/runtime generation to representation policy."""
    return _fingerprint(
        {
            "runtime_generation": generation,
            "graph_execution_policy": policy.fingerprint,
        }
    )


@dataclass(frozen=True, order=True)
class LogicalDispatchKey:
    owner: str
    mode: str
    token_bucket: int
    logical_num_reqs: int | None
    uniform_query_len: int | None
    active_loras: int = 0

    def __post_init__(self) -> None:
        if not self.owner or not self.mode:
            raise ValueError("logical graph owner and mode are required")
        if self.token_bucket <= 0 or self.active_loras < 0:
            raise ValueError("invalid logical graph geometry")
        if self.logical_num_reqs is not None and self.logical_num_reqs <= 0:
            raise ValueError("logical_num_reqs must be positive when present")
        if self.uniform_query_len is not None and self.uniform_query_len <= 0:
            raise ValueError("uniform_query_len must be positive when present")


@dataclass(frozen=True, order=True)
class PhysicalReplayKey:
    logical: LogicalDispatchKey
    physical_num_reqs: int
    generation: RuntimeGeneration

    def __post_init__(self) -> None:
        if self.physical_num_reqs <= 0:
            raise ValueError("physical_num_reqs must be positive")
        if self.physical_num_reqs > self.logical.token_bucket:
            raise ValueError("physical request count exceeds the token carrier")
        logical_x = self.logical.logical_num_reqs
        if logical_x is not None and logical_x != self.physical_num_reqs:
            raise ValueError("FULL logical and physical request counts diverge")

    @property
    def identity(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True)
class GraphPrice:
    resident_bytes: int
    capture_peak_bytes: int
    reclaim_group: str

    def __post_init__(self) -> None:
        if self.resident_bytes < 0 or self.capture_peak_bytes < 0:
            raise ValueError("graph prices cannot be negative")
        if self.capture_peak_bytes < self.resident_bytes:
            raise ValueError("capture peak cannot be below resident bytes")
        if not self.reclaim_group:
            raise ValueError("a graph price requires a reclaim group")


@dataclass(frozen=True)
class ReclaimGroup:
    group_id: str
    keys: tuple[PhysicalReplayKey, ...]
    reclaimable_bytes: int
    retained_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.group_id or not self.keys:
            raise ValueError("a reclaim group requires an id and entries")
        if self.reclaimable_bytes < 0 or self.retained_bytes < 0:
            raise ValueError("reclaim accounting cannot be negative")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("a reclaim group contains duplicate keys")


@dataclass(frozen=True)
class ElasticGraphEntry:
    key: PhysicalReplayKey
    state: GraphResidency = GraphResidency.COLD
    price: GraphPrice | None = None
    leases: frozenset[str] = frozenset()
    last_used_epoch: int = 0
    deferred_free: bool = False

    @property
    def hot(self) -> bool:
        return self.state in {
            GraphResidency.HOT_EVICTABLE,
            GraphResidency.HOT_PINNED,
        }

    @property
    def pinned(self) -> bool:
        return self.state == GraphResidency.HOT_PINNED

    @property
    def reclaimable(self) -> bool:
        return (
            self.hot and not self.pinned and not self.leases and not self.deferred_free
        )


@dataclass(frozen=True)
class ElasticGraphStats:
    hot_hits: int
    cold_misses: int
    promotions: int
    evictions: int
    evicted_bytes: int
    deferrals: int
    defer_reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ElasticStepPlan:
    transaction_id: str
    generation: RuntimeGeneration
    kind: ElasticPlanKind
    physical_keys: tuple[PhysicalReplayKey, ...]
    hot_hits: tuple[PhysicalReplayKey, ...]
    cold_misses: tuple[PhysicalReplayKey, ...]
    protected_keys: tuple[PhysicalReplayKey, ...]
    victim_keys: tuple[PhysicalReplayKey, ...]
    reclaim_groups: tuple[str, ...]
    capture_order: tuple[PhysicalReplayKey, ...]
    kv_transition: tuple[int, int] | None
    request_bytes: int
    available_bytes: int
    capture_loan_bytes: int
    reclaim_bytes: int
    defer_reason: str | None = None
    staged_hotset_replace: bool = False
    residency_cap_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise ValueError("transaction_id is required")
        for key in self.physical_keys:
            if key.generation != self.generation:
                raise ValueError("plan contains a stale runtime generation")
        for name in (
            "request_bytes",
            "available_bytes",
            "capture_loan_bytes",
            "reclaim_bytes",
            "residency_cap_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.kind == ElasticPlanKind.USER and self.cold_misses:
            raise ValueError("a user plan cannot contain cold graph misses")
        if self.kind == ElasticPlanKind.MAINTENANCE and not self.cold_misses:
            raise ValueError("maintenance requires at least one cold miss")
        if self.kind == ElasticPlanKind.RECLAIM and (
            self.cold_misses or not self.victim_keys
        ):
            raise ValueError("reclaim requires victims and cannot capture")
        if self.kind == ElasticPlanKind.DEFER and not self.defer_reason:
            raise ValueError("a deferred plan requires an explicit reason")
        if self.kind != ElasticPlanKind.DEFER and self.defer_reason is not None:
            raise ValueError("only deferred plans may carry a defer reason")
        if set(self.hot_hits).intersection(self.cold_misses):
            raise ValueError("one physical key cannot be both HOT and COLD")
        if not set(self.hot_hits).union(self.cold_misses).issubset(self.physical_keys):
            raise ValueError("plan hit/miss keys are outside its owner set")
        if set(self.victim_keys).intersection(self.protected_keys):
            raise ValueError("a protected key cannot be selected as a victim")
        if self.staged_hotset_replace and self.kind != ElasticPlanKind.MAINTENANCE:
            raise ValueError("staged hotset publication requires maintenance")
        if self.residency_cap_bytes and self.kind == ElasticPlanKind.DEFER:
            raise ValueError("deferred plans cannot claim an active residency cap")

    @cached_property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.value if isinstance(item, Enum) else asdict(item),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def require_plan_consensus(plans: Sequence[ElasticStepPlan]) -> str:
    """Return the common fingerprint or reject rank divergence."""
    if not plans:
        raise ElasticGraphError("rank consensus requires at least one plan")
    fingerprints = {plan.fingerprint for plan in plans}
    if len(fingerprints) != 1:
        raise ElasticGraphError("elastic graph plan fingerprint differs by rank")
    return next(iter(fingerprints))


def configured_compiled_piecewise_sizes(vllm_config: Any) -> frozenset[int]:
    """Return token carriers that stay compiled but own no CUDA Graph.

    The exemption is intentionally exact and configuration-bound.  It cannot
    silently turn an arbitrary missing graph into an eager fallback.
    """
    additional_config = getattr(vllm_config, "additional_config", None)
    raw_sizes = (
        additional_config.get("elastic_compiled_piecewise_sizes", [])
        if isinstance(additional_config, dict)
        else []
    )
    if not isinstance(raw_sizes, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in raw_sizes
    ):
        raise ElasticGraphError(
            "elastic_compiled_piecewise_sizes must be a list of positive integers"
        )
    sizes = frozenset(raw_sizes)
    if len(sizes) != len(raw_sizes):
        raise ElasticGraphError(
            "elastic_compiled_piecewise_sizes must not contain duplicates"
        )
    max_tokens = getattr(
        getattr(vllm_config, "scheduler_config", None),
        "max_num_batched_tokens",
        None,
    )
    if sizes and (not isinstance(max_tokens, int) or max_tokens <= 0):
        raise ElasticGraphError(
            "compiled PIECEWISE sizes require max_num_batched_tokens"
        )
    if any(size > max_tokens for size in sizes):
        raise ElasticGraphError(
            "elastic_compiled_piecewise_sizes exceeds max_num_batched_tokens"
        )
    return sizes


def resolve_step_physical_keys(
    step_key: tuple[int, int, int, int, int] | None,
    generation: RuntimeGeneration,
    max_num_batched_tokens: int | None = None,
    compiled_piecewise_sizes: Iterable[int] = (),
    policy: GraphExecutionPolicy | None = None,
) -> tuple[PhysicalReplayKey, ...]:
    """Resolve the target/MTP owner set from the canonical scheduler shape."""
    if step_key is None:
        return ()
    full, num_spec_tokens, num_reqs, num_tokens, uniform_query_len = step_key
    if num_reqs <= 0 or num_tokens <= 0:
        raise ElasticGraphError("non-empty graph step has invalid geometry")
    target_mode = "FULL" if full else "PIECEWISE"

    def piecewise_boundary(tokens: int) -> int:
        boundary = 1 << (tokens - 1).bit_length()
        if max_num_batched_tokens is None:
            return boundary
        if max_num_batched_tokens <= 0 or tokens > max_num_batched_tokens:
            raise ElasticGraphError(
                "PIECEWISE graph tokens exceed max_num_batched_tokens"
            )
        return min(boundary, max_num_batched_tokens)

    def make(
        owner: str,
        mode: str,
        tokens: int,
        physical_x: int,
        uniform: int | None,
    ) -> PhysicalReplayKey:
        return PhysicalReplayKey(
            logical=LogicalDispatchKey(
                owner=owner,
                mode=mode,
                token_bucket=tokens,
                logical_num_reqs=physical_x if mode == "FULL" else None,
                # Exact PIECEWISE decode may still carry semantic qlen. Graph
                # mode controls executable coverage, not model math.
                uniform_query_len=uniform,
            ),
            physical_num_reqs=physical_x,
            generation=generation,
        )

    compiled_sizes = frozenset(compiled_piecewise_sizes)
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in compiled_sizes
    ):
        raise ElasticGraphError(
            "compiled PIECEWISE sizes must contain positive integers"
        )
    if policy is not None and all(
        owner.activation != "legacy-v1" and owner.token_source != "legacy-v1"
        for owner in policy.owners
    ):
        # New runtimes publish the semantic-to-physical transform for every
        # owner.  No scheduler-side knowledge of MTP, DFlash, DSpark, or a
        # particular K belongs here: the observed step supplies X/M/qlen and
        # each manager policy supplies the rows it actually consumes.
        ordered_owners = tuple(
            sorted(policy.owners, key=lambda owner: owner.execution_order or 0)
        )
        keys: list[PhysicalReplayKey] = []
        semantic_uniform = uniform_query_len or None
        uniform_step = bool(
            semantic_uniform is not None
            and num_tokens == num_reqs * semantic_uniform
        )
        for owner_policy in ordered_owners:
            if (
                owner_policy.activation == "speculative"
                and num_spec_tokens <= 0
            ):
                continue
            if owner_policy.token_source == "step":
                owner_tokens = num_tokens
                owner_uniform = semantic_uniform
                exact_step = uniform_step
            elif owner_policy.token_source == "requests":
                owner_tokens = num_reqs
                owner_uniform = 1
                exact_step = True
            elif owner_policy.token_source == "fixed_query":
                assert owner_policy.fixed_query_len is not None
                owner_uniform = owner_policy.fixed_query_len
                owner_tokens = num_reqs * owner_uniform
                exact_step = True
            else:  # guarded above; retain a fail-closed error for corrupt policy
                raise ElasticGraphError(
                    "runtime Graph owner has no physical shape contract"
                )

            mode = owner_policy.mode_for(owner_uniform)
            if mode == "NONE":
                continue
            if mode == "FULL":
                key = make(
                    owner_policy.owner,
                    mode,
                    owner_tokens,
                    num_reqs,
                    owner_uniform,
                )
            else:
                preserve_exact_m = bool(
                    exact_step
                    and owner_policy.piecewise_padding_contract
                    == "exact-batched-decode-m-v1"
                )
                physical_tokens = (
                    owner_tokens
                    if preserve_exact_m
                    else piecewise_boundary(owner_tokens)
                )
                owner_compiled_sizes = frozenset(
                    owner_policy.compiled_piecewise_sizes
                )
                if not preserve_exact_m and physical_tokens in owner_compiled_sizes:
                    continue
                minimum = owner_policy.piecewise_query_len_min_tokens
                physical_uniform = (
                    owner_uniform
                    if minimum is not None and physical_tokens >= minimum
                    else None
                )
                key = make(
                    owner_policy.owner,
                    mode,
                    physical_tokens,
                    num_reqs,
                    physical_uniform,
                )
            owner_policy.validate_physical_key(key)
            keys.append(key)
        return tuple(keys)

    target_uniform_query_len = uniform_query_len or None
    # Geometry alone is insufficient to identify the short speculative decode
    # lane: a mixed/prefill step may coincidentally have M=(K+1)X.  The
    # canonical semantic key additionally carries qlen=K+1 for exact batched
    # verification.  Resolve this before applying compiled-carrier exemptions
    # so terminal X1/M4 is protected just like X8/M32 and X40/M160.
    exact_batched_decode = (
        num_spec_tokens > 0
        and uniform_query_len == num_spec_tokens + 1
        and num_tokens == num_reqs * (num_spec_tokens + 1)
    )
    if target_mode == "PIECEWISE" and policy is not None:
        target_policy = policy.owner_policy("target")
        minimum = target_policy.piecewise_query_len_min_tokens
        if minimum is None or num_tokens < minimum:
            target_uniform_query_len = None
    # A configured compiled carrier is an owner+token execution contract, not
    # a semantic-phase exception.  The outer torch.compile executable consumes
    # exact short decode as well as mixed/prefill work; only the inner CUDA
    # Graph wrapper is absent.  Reintroducing a Graph merely because qlen=K+1
    # changes the physical DAG without catalog evidence and makes the scheduler
    # price an executable that the sealed catalog never measured.
    compiled_target = (
        target_mode == "PIECEWISE"
        and not exact_batched_decode
        and target_uniform_query_len is None
        and num_tokens in compiled_sizes
    )
    keys = []
    if not compiled_target:
        keys.append(
            make(
                "target",
                target_mode,
                num_tokens,
                num_reqs,
                target_uniform_query_len,
            )
        )
    if num_spec_tokens > 0:
        # The first K3 draft pass has its own qlen=K+1. FlashInfer exposes
        # FULL only for uniform single-token decode, so the worker always
        # resolves this owner to a token-bucketed PIECEWISE descriptor while
        # retaining exact physical X. Do not inherit the target's mode.
        # With the batched-q1 verifier, an assembled K-step decode wave has
        # exact physical M=(K+1)X.  Scheduler and both PIECEWISE workers must
        # name the same executable.  Generic prefill shapes retain the
        # power-of-two boundary until padded-tail mutation is proven; the
        # short decode family stays exact and Graph-backed instead of silently
        # falling through to NONE.
        # Target and MTP-prefill do not share a CUDA Graph mode: an exact
        # uniform K+1 verification wave may select FULL for the target while
        # FlashInfer still exposes only PIECEWISE for MTP-prefill.  Exact-M is
        # nevertheless the common physical carrier.  Conditioning this test
        # on the target mode made FULL target M160 resolve MTP-prefill as the
        # padded M256 class; a configured compiled-only M256 then removed the
        # required M160 owner from the atomic scheduler plan while the worker
        # correctly dispatched exact M160 and failed closed.
        mtp_prefill_tokens = (
            num_tokens if exact_batched_decode else piecewise_boundary(num_tokens)
        )
        # The same configured compiled carrier contract applies to the exact
        # K+1 verification wave.  MTP decode remains a distinct FULL Graph.
        if exact_batched_decode or mtp_prefill_tokens not in compiled_sizes:
            keys.append(
                make(
                    "mtp_prefill",
                    "PIECEWISE",
                    mtp_prefill_tokens,
                    num_reqs,
                    None,
                )
            )
        keys.append(make("mtp_decode", "FULL", num_reqs, num_reqs, 1))
    resolved = tuple(keys)
    if policy is not None:
        for key in resolved:
            policy.validate_physical_key(key)
    return resolved


def derive_short_decode_graph_inventory(
    *,
    max_x: int,
    num_spec_tokens: int,
    generation: RuntimeGeneration,
    max_num_batched_tokens: int,
    compiled_piecewise_sizes: Iterable[int] = (),
    policy: GraphExecutionPolicy | None = None,
) -> Mapping[int, tuple[PhysicalReplayKey, ...]]:
    """Derive the complete K-step short-decode inventory from MaxX.

    Power-of-two cohort sizes are the reusable core and a non-power-of-two
    configured endpoint is retained exactly. DCP is intentionally absent:
    it partitions attention context, not the assembled request/M axis.
    """
    if max_x <= 0:
        raise ValueError("max_x must be positive")
    if num_spec_tokens < 0:
        raise ValueError("short decode requires non-negative K")
    if max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")
    endpoint_m = max_x * (num_spec_tokens + 1)
    if endpoint_m > max_num_batched_tokens:
        raise ElasticGraphError(
            "short-decode MaxX exceeds the configured token carrier: "
            f"X={max_x} K={num_spec_tokens} M={endpoint_m} "
            f"max_num_batched_tokens={max_num_batched_tokens}"
        )

    xs: list[int] = []
    x = 1
    while x <= max_x:
        xs.append(x)
        x <<= 1
    if xs[-1] != max_x:
        xs.append(max_x)
    return {
        x: resolve_step_physical_keys(
            (
                0,
                num_spec_tokens,
                x,
                x * (num_spec_tokens + 1),
                num_spec_tokens + 1,
            ),
            generation,
            max_num_batched_tokens=max_num_batched_tokens,
            compiled_piecewise_sizes=compiled_piecewise_sizes,
            policy=policy,
        )
        for x in xs
    }


def select_short_decode_physical_x(
    actual_x: int,
    inventory_xs: Iterable[int],
) -> int:
    """Return the smallest declared physical cohort covering ``actual_x``."""
    if isinstance(actual_x, bool) or not isinstance(actual_x, int) or actual_x <= 0:
        raise ValueError("actual short-decode X must be a positive integer")
    xs = tuple(inventory_xs)
    if (
        not xs
        or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in xs)
        or tuple(sorted(set(xs))) != xs
    ):
        raise ValueError("short-decode inventory X values must be sorted and unique")
    for physical_x in xs:
        if physical_x >= actual_x:
            return physical_x
    raise ElasticGraphError(
        "semantic short-decode X exceeds the declared physical inventory: "
        f"actual_x={actual_x} max_x={xs[-1]}"
    )


class ElasticGraphCache:
    """Deterministic multi-entry cache and admission planner.

    CUDA teardown/publication is performed by the caller.  This object owns
    only policy state and therefore can be replayed in scheduler tests.
    """

    def __init__(self, generation: RuntimeGeneration):
        self.generation = generation
        self._entries: dict[PhysicalReplayKey, ElasticGraphEntry] = {}
        self._groups: dict[str, ReclaimGroup] = {}
        self._epoch = 0
        self._hot_hits = 0
        self._cold_misses = 0
        self._promotions = 0
        self._evictions = 0
        self._evicted_bytes = 0
        self._deferrals = 0
        self._defer_reasons: Counter[str] = Counter()
        self._trace: deque[tuple[str, str, str]] = deque(maxlen=128)

    @property
    def entries(self) -> Mapping[PhysicalReplayKey, ElasticGraphEntry]:
        return self._entries

    @property
    def stats(self) -> ElasticGraphStats:
        return ElasticGraphStats(
            hot_hits=self._hot_hits,
            cold_misses=self._cold_misses,
            promotions=self._promotions,
            evictions=self._evictions,
            evicted_bytes=self._evicted_bytes,
            deferrals=self._deferrals,
            defer_reasons=tuple(sorted(self._defer_reasons.items())),
        )

    @property
    def trace(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(self._trace)

    def register(
        self,
        key: PhysicalReplayKey,
        *,
        price: GraphPrice | None = None,
    ) -> None:
        self._require_generation(key)
        existing = self._entries.get(key)
        if existing is None:
            self._entries[key] = ElasticGraphEntry(key=key, price=price)
            return
        if price is None:
            return
        if existing.price is None:
            self._entries[key] = replace(existing, price=price)
            return
        if existing.price.reclaim_group != price.reclaim_group:
            raise ElasticGraphError(
                "graph reclaim identity changed inside one generation"
            )
        # CUDA capture cost is history-dependent: a recapture can reuse lower-
        # layer allocations and become cheaper than the first cold capture.
        # Preserve the componentwise maximum as the future admission envelope;
        # synchronize_hot separately rebuilds current physical residency and
        # reclaim groups from the worker's latest receipt.
        envelope = GraphPrice(
            resident_bytes=max(existing.price.resident_bytes, price.resident_bytes),
            capture_peak_bytes=max(
                existing.price.capture_peak_bytes, price.capture_peak_bytes
            ),
            reclaim_group=price.reclaim_group,
        )
        self._entries[key] = replace(existing, price=envelope)

    def publish_hot(
        self,
        key: PhysicalReplayKey,
        price: GraphPrice,
        *,
        pinned: bool,
    ) -> None:
        self.register(key, price=price)
        entry = self._entries[key]
        if entry.leases:
            raise ElasticGraphError("cannot replace a leased graph executable")
        state = GraphResidency.HOT_PINNED if pinned else GraphResidency.HOT_EVICTABLE
        self._epoch += 1
        self._entries[key] = replace(
            entry,
            state=state,
            last_used_epoch=self._epoch,
            deferred_free=False,
        )

    def retain_hot(
        self,
        transaction_id: str,
        keys: Iterable[PhysicalReplayKey],
    ) -> None:
        """Lease a complete already-HOT physical set without partial commit."""
        if not transaction_id:
            raise ValueError("retention transaction id is required")
        retained = tuple(dict.fromkeys(keys))
        if not retained:
            raise ValueError("retention requires at least one physical key")
        for key in retained:
            self._require_generation(key)
            entry = self._entries.get(key)
            if entry is None or not entry.hot or entry.deferred_free:
                raise ElasticGraphError(
                    "retention requires every physical key HOT and stable"
                )
        for key in retained:
            entry = self._entries[key]
            self._entries[key] = replace(
                entry, leases=entry.leases.union((transaction_id,))
            )

    def unpin_idle(
        self,
        keys: Iterable[PhysicalReplayKey],
    ) -> tuple[PhysicalReplayKey, ...]:
        """Make selected idle pinned entries reclaimable for a rebuild."""
        selected = set(keys)
        pinned = tuple(
            entry
            for entry in self._entries.values()
            if entry.pinned and entry.key in selected
        )
        if any(entry.leases or entry.deferred_free for entry in pinned):
            raise ElasticGraphError("cannot unpin an active graph executable")
        for entry in pinned:
            self._entries[entry.key] = replace(
                entry,
                state=GraphResidency.HOT_EVICTABLE,
            )
        return tuple(entry.key for entry in pinned)

    def install_reclaim_group(self, group: ReclaimGroup) -> None:
        for key in group.keys:
            self._require_generation(key)
            entry = self._entries.get(key)
            if entry is None or not entry.hot:
                raise ElasticGraphError("reclaim group references a non-HOT entry")
            if entry.price is None or entry.price.reclaim_group != group.group_id:
                raise ElasticGraphError("reclaim group identity disagrees with price")
        self._groups[group.group_id] = group

    def synchronize_hot(
        self,
        rows: Iterable[tuple[PhysicalReplayKey, GraphPrice, bool, int]],
    ) -> None:
        """Replace scheduler residency with a completed worker receipt.

        The receipt is authoritative only after execution/capture publication
        and lease release.  Catalog prices survive a COLD transition, while
        HOT membership and reclaim groups are rebuilt from physical read-back.
        """
        observed: dict[PhysicalReplayKey, tuple[GraphPrice, bool, int]] = {}
        for key, price, pinned, reclaimable_bytes in rows:
            self._require_generation(key)
            if key in observed:
                raise ElasticGraphError("worker HOT receipt contains a duplicate key")
            if reclaimable_bytes < 0:
                raise ElasticGraphError("worker reclaim proof cannot be negative")
            observed[key] = (price, pinned, reclaimable_bytes)
        for key, entry in tuple(self._entries.items()):
            if entry.hot and key not in observed:
                if entry.leases:
                    observed_ids = tuple(
                        sorted(item.identity for item in observed)
                    )
                    raise ElasticGraphError(
                        "worker receipt dropped a scheduler-leased graph: "
                        f"missing={key.identity!r} "
                        f"leases={tuple(sorted(entry.leases))!r} "
                        f"observed={observed_ids!r}"
                    )
                self._entries[key] = replace(
                    entry,
                    state=GraphResidency.COLD,
                    deferred_free=False,
                )
        self._groups.clear()
        for key, (price, pinned, reclaimable_bytes) in observed.items():
            prior = self._entries.get(key)
            was_capturing = (
                prior is not None and prior.state == GraphResidency.CAPTURING
            )
            if prior is not None and prior.leases:
                # A worker receipt is authoritative physical read-back, not a
                # replacement of an executable that remained resident. Keep
                # the scheduler lease until its owning multi-step epoch ends.
                self.register(key, price=price)
                retained = self._entries[key]
                self._epoch += 1
                self._entries[key] = replace(
                    retained,
                    state=(
                        GraphResidency.HOT_PINNED
                        if pinned
                        else GraphResidency.HOT_EVICTABLE
                    ),
                    last_used_epoch=self._epoch,
                    deferred_free=False,
                )
            else:
                self.publish_hot(key, price, pinned=pinned)
            if was_capturing:
                self._promotions += 1
            if not pinned and reclaimable_bytes:
                self.install_reclaim_group(
                    ReclaimGroup(
                        group_id=price.reclaim_group,
                        keys=(key,),
                        reclaimable_bytes=reclaimable_bytes,
                        retained_bytes=max(0, price.resident_bytes - reclaimable_bytes),
                    )
                )

    @staticmethod
    def _ordered_capture_misses(
        misses: Iterable[PhysicalReplayKey],
    ) -> tuple[PhysicalReplayKey, ...]:
        """Return one global graph-safe order across serial owners.

        The lower-level manager captures PIECEWISE before FULL because the
        former initializes mutable compiler/attention wrapper state and the
        latter must retain the final address-stable replay state.  A
        multi-owner transaction is one physical capture DAG and must preserve
        the same ordering across manager boundaries.  Python's stable sort
        retains the scheduler owner order within each mode.
        """
        ordered = tuple(misses)
        unknown_modes = {
            key.logical.mode
            for key in ordered
            if key.logical.mode not in {"PIECEWISE", "FULL"}
        }
        if unknown_modes:
            raise ElasticGraphError(
                f"unsupported CUDA Graph capture mode(s): {sorted(unknown_modes)}"
            )
        return tuple(
            sorted(
                ordered,
                key=lambda key: 0 if key.logical.mode == "PIECEWISE" else 1,
            )
        )

    def plan(
        self,
        transaction_id: str,
        physical_keys: Iterable[PhysicalReplayKey],
        *,
        request_bytes: int,
        available_bytes: int,
        kv_transition: tuple[int, int] | None = None,
        protected_keys: Iterable[PhysicalReplayKey] = (),
        class_envelopes: Mapping[LogicalDispatchKey, GraphPrice] | None = None,
        owner_set_capture_envelope_bytes: int | None = None,
        retained_transition_overlap_bytes: int = 0,
        shared_resident_bytes: int = 0,
        replace_unleased_on_miss: bool = False,
        residency_cap_bytes: int = 0,
    ) -> ElasticStepPlan:
        if request_bytes < 0 or available_bytes < 0:
            raise ValueError("admission byte counts cannot be negative")
        if residency_cap_bytes < 0:
            raise ValueError("residency cap cannot be negative")
        if retained_transition_overlap_bytes < 0:
            raise ValueError("retained transition overlap cannot be negative")
        if shared_resident_bytes < 0 or shared_resident_bytes > request_bytes:
            raise ValueError(
                "shared resident bytes must lie inside current request bytes"
            )
        if replace_unleased_on_miss and residency_cap_bytes <= 0:
            raise ValueError("bounded hotset replacement requires a positive cap")
        keys = tuple(dict.fromkeys(physical_keys))
        protected = tuple(dict.fromkeys(protected_keys))
        for key in (*keys, *protected):
            self._require_generation(key)
        hits: list[PhysicalReplayKey] = []
        misses: list[PhysicalReplayKey] = []
        prices: dict[PhysicalReplayKey, GraphPrice] = {}
        envelopes = class_envelopes or {}
        for key in keys:
            entry = self._entries.get(key)
            if entry is not None and entry.hot:
                hits.append(key)
                continue
            misses.append(key)
            price = entry.price if entry is not None else None
            price = price or envelopes.get(key.logical)
            if price is None:
                if owner_set_capture_envelope_bytes is not None:
                    continue
                return self._defer(
                    transaction_id,
                    keys,
                    tuple(hits),
                    tuple(misses),
                    protected,
                    request_bytes,
                    available_bytes,
                    kv_transition,
                    "missing_exact_price_or_class_envelope",
                )
            prices[key] = price

        if owner_set_capture_envelope_bytes is not None:
            if owner_set_capture_envelope_bytes < 0:
                raise ValueError("owner-set capture envelope cannot be negative")
            capture_loan = owner_set_capture_envelope_bytes if misses else 0
            required_bytes = max(request_bytes, capture_loan)
        else:
            capture_loan = sum(price.capture_peak_bytes for price in prices.values())
            required_bytes = request_bytes + capture_loan
        if replace_unleased_on_miss:
            if not misses:
                if request_bytes > residency_cap_bytes:
                    return self._defer(
                        transaction_id,
                        keys,
                        tuple(hits),
                        tuple(misses),
                        protected,
                        request_bytes,
                        available_bytes,
                        kv_transition,
                        "hotset_residency_over_cap",
                    )
            else:
                if owner_set_capture_envelope_bytes is None:
                    return self._defer(
                        transaction_id,
                        keys,
                        tuple(hits),
                        tuple(misses),
                        protected,
                        request_bytes,
                        available_bytes,
                        kv_transition,
                        "hotset_owner_set_envelope_required",
                    )
                # Keep a multi-entry working set while its predicted settled
                # residency remains under the declared cap.
                # The previous implementation replaced every unleased entry
                # on every miss, so a repeated product wave recaptured the
                # same small set indefinitely even with hundreds of MiB of
                # unused hotset capacity. For an unknown exact key the
                # synchronized owner-set envelope is a conservative upper
                # bound for the entire new set, so current+envelope below the
                # cap also proves that eviction is unnecessary. Worker
                # read-back remains the final cap oracle after capture.
                priced_retention = (
                    len(prices) == len(misses)
                    and request_bytes
                    + sum(price.resident_bytes for price in prices.values())
                    <= residency_cap_bytes
                )
                conservative_unknown_retention = (
                    len(prices) != len(misses)
                    and request_bytes
                    + max(
                        0,
                        owner_set_capture_envelope_bytes - shared_resident_bytes,
                    )
                    <= residency_cap_bytes
                )
                retain_existing = (
                    priced_retention or conservative_unknown_retention
                )
                destination_increment = (
                    sum(price.resident_bytes for price in prices.values())
                    if len(prices) == len(misses)
                    else max(
                        0,
                        owner_set_capture_envelope_bytes - shared_resident_bytes,
                    )
                )
                settled_deficit = max(
                    0,
                    request_bytes
                    + destination_increment
                    - residency_cap_bytes,
                )
                replacement: tuple[
                    tuple[PhysicalReplayKey, ...], tuple[str, ...], int
                ] | None
                if retain_existing:
                    replacement = ((), (), 0)
                elif len(prices) == len(misses):
                    replacement = self._select_victims(
                        settled_deficit,
                        set(keys).union(protected),
                    )
                else:
                    # An unpriced owner-set envelope is an aggregate endpoint,
                    # not a marginal resident price. It can prove all-old -> B
                    # replacement, but cannot rank a partial subset by settled
                    # bytes. Keep the older fail-closed whole-set fallback only
                    # for this UNKNOWN case; measured classes use the minimal
                    # deficit selector above.
                    replacement = self._select_complete_hotset_victims(
                        set(keys).union(protected)
                    )
                if replacement is None:
                    return self._defer(
                        transaction_id,
                        keys,
                        tuple(hits),
                        tuple(misses),
                        protected,
                        request_bytes,
                        available_bytes,
                        kv_transition,
                        "hotset_victim_not_reclaimable",
                    )
                victims, groups, reclaimed = replacement
                if (
                    len(prices) == len(misses)
                    and reclaimed < settled_deficit
                ):
                    return self._defer(
                        transaction_id,
                        keys,
                        tuple(hits),
                        tuple(misses),
                        protected,
                        request_bytes,
                        available_bytes,
                        kv_transition,
                        "hotset_victim_not_reclaimable",
                    )
                # A replacement endpoint was calibrated while the prior set
                # remained alive until post-consumer publication, so max(old,
                # destination) is its proven atomic bound. Retention is a
                # different physical DAG: unrelated entries intentionally stay
                # resident after publication. Their current endpoint and each
                # exact miss capture peak must coexist. Treating the destination
                # envelope as if it already contained those retained entries
                # underfunded a K0 B64 -> FULL/q1 transition by 1.4 MiB.
                capture_loan = (
                    max(
                        owner_set_capture_envelope_bytes,
                        request_bytes
                        + retained_transition_overlap_bytes
                        + (
                            sum(
                                price.capture_peak_bytes
                                for price in prices.values()
                            )
                            if len(prices) == len(misses)
                            else max(
                                0,
                                owner_set_capture_envelope_bytes
                                - shared_resident_bytes,
                            )
                        ),
                    )
                    if retain_existing
                    else max(
                        request_bytes,
                        owner_set_capture_envelope_bytes,
                    )
                )
                if capture_loan > available_bytes:
                    return self._defer(
                        transaction_id,
                        keys,
                        tuple(hits),
                        tuple(misses),
                        protected,
                        request_bytes,
                        available_bytes,
                        kv_transition,
                        "insufficient_atomic_hotset_transition_bytes",
                        capture_loan=capture_loan,
                    )
                return ElasticStepPlan(
                    transaction_id=transaction_id,
                    generation=self.generation,
                    kind=ElasticPlanKind.MAINTENANCE,
                    physical_keys=keys,
                    hot_hits=tuple(hits),
                    cold_misses=tuple(misses),
                    protected_keys=protected,
                    victim_keys=victims,
                    reclaim_groups=groups,
                    capture_order=self._ordered_capture_misses(misses),
                    kv_transition=kv_transition,
                    request_bytes=request_bytes,
                    available_bytes=available_bytes,
                    capture_loan_bytes=capture_loan,
                    reclaim_bytes=reclaimed,
                    # Empty->first-set also needs worker actual-charge cap
                    # validation before it becomes scheduler authority.
                    staged_hotset_replace=True,
                    residency_cap_bytes=residency_cap_bytes,
                )
        deficit = max(0, required_bytes - available_bytes)
        victims, groups, reclaimed = self._select_victims(
            deficit, set(keys).union(protected)
        )
        if reclaimed < deficit:
            return self._defer(
                transaction_id,
                keys,
                tuple(hits),
                tuple(misses),
                protected,
                request_bytes,
                available_bytes,
                kv_transition,
                "insufficient_reclaimable_graph_and_kv_bytes",
                victims=victims,
                groups=groups,
                reclaimed=reclaimed,
                capture_loan=capture_loan,
            )
        if owner_set_capture_envelope_bytes is not None and misses and reclaimed:
            # The owner-set envelope is an endpoint bound for the resident set
            # before the planned victims are destroyed.  A reclaim proof is the
            # only evidence that can reduce that endpoint before capture; carry
            # the reduced bound to the worker instead of an impossible
            # pre-reclaim loan larger than ``available_bytes``.
            capture_loan = max(0, capture_loan - reclaimed)
        kind = ElasticPlanKind.MAINTENANCE if misses else ElasticPlanKind.USER
        return ElasticStepPlan(
            transaction_id=transaction_id,
            generation=self.generation,
            kind=kind,
            physical_keys=keys,
            hot_hits=tuple(hits),
            cold_misses=tuple(misses),
            protected_keys=protected,
            victim_keys=victims,
            reclaim_groups=groups,
            capture_order=self._ordered_capture_misses(misses),
            kv_transition=kv_transition,
            request_bytes=request_bytes,
            available_bytes=available_bytes,
            capture_loan_bytes=capture_loan,
            reclaim_bytes=reclaimed,
        )

    def commit_user(self, plan: ElasticStepPlan) -> None:
        self._validate_plan(plan, ElasticPlanKind.USER)
        for key in plan.physical_keys:
            entry = self._entries.get(key)
            if entry is None or not entry.hot:
                raise ElasticGraphError("user commit requires every key HOT")
        self._epoch += 1
        for key in plan.physical_keys:
            entry = self._entries[key]
            self._entries[key] = replace(
                entry,
                leases=entry.leases.union((plan.transaction_id,)),
                last_used_epoch=self._epoch,
            )
        self._hot_hits += len(plan.hot_hits)
        self._trace.append((plan.transaction_id, "user", plan.fingerprint))

    def plan_reclaim_all(
        self,
        transaction_id: str,
        *,
        request_bytes: int,
        available_bytes: int,
        protected_keys: Iterable[PhysicalReplayKey] = (),
    ) -> ElasticStepPlan:
        protected = tuple(dict.fromkeys(protected_keys))
        victims, groups, reclaimed = self._select_victims(1 << 62, set(protected))
        if not victims:
            return self._defer(
                transaction_id,
                (),
                (),
                (),
                protected,
                request_bytes,
                available_bytes,
                None,
                "no_reclaimable_piecewise_graphs",
            )
        return ElasticStepPlan(
            transaction_id=transaction_id,
            generation=self.generation,
            kind=ElasticPlanKind.RECLAIM,
            physical_keys=(),
            hot_hits=(),
            cold_misses=(),
            protected_keys=protected,
            victim_keys=victims,
            reclaim_groups=groups,
            capture_order=(),
            kv_transition=None,
            request_bytes=request_bytes,
            available_bytes=available_bytes,
            capture_loan_bytes=max(0, request_bytes - reclaimed),
            reclaim_bytes=reclaimed,
        )

    def begin_reclaim(self, plan: ElasticStepPlan) -> None:
        self._validate_plan(plan, ElasticPlanKind.RECLAIM)
        for key in plan.victim_keys:
            entry = self._entries.get(key)
            if entry is None or not entry.reclaimable:
                raise ElasticGraphError("reclaim victim is no longer reclaimable")
            self._entries[key] = replace(entry, state=GraphResidency.COLD)
        self._evictions += len(plan.victim_keys)
        self._evicted_bytes += plan.reclaim_bytes
        self._trace.append((plan.transaction_id, "reclaim", plan.fingerprint))

    def begin_maintenance(self, plan: ElasticStepPlan) -> None:
        self._validate_plan(plan, ElasticPlanKind.MAINTENANCE)
        for key in plan.victim_keys:
            entry = self._entries.get(key)
            if entry is None or not entry.reclaimable:
                raise ElasticGraphError("maintenance victim is no longer reclaimable")
            if plan.staged_hotset_replace:
                self._entries[key] = replace(
                    entry,
                    leases=entry.leases.union((plan.transaction_id,)),
                )
            else:
                self._entries[key] = replace(entry, state=GraphResidency.COLD)
        for key in plan.cold_misses:
            entry = self._entries.get(key) or ElasticGraphEntry(key=key)
            if entry.hot or entry.leases:
                raise ElasticGraphError("maintenance miss changed before capture")
            self._entries[key] = replace(entry, state=GraphResidency.CAPTURING)
        self._cold_misses += len(plan.cold_misses)
        if not plan.staged_hotset_replace:
            self._evictions += len(plan.victim_keys)
            self._evicted_bytes += plan.reclaim_bytes
        self._trace.append((plan.transaction_id, "maintenance", plan.fingerprint))

    def finish_maintenance(
        self,
        plan: ElasticStepPlan,
        prices: Mapping[PhysicalReplayKey, GraphPrice],
        *,
        pinned_keys: Iterable[PhysicalReplayKey] = (),
    ) -> None:
        self._validate_plan(plan, ElasticPlanKind.MAINTENANCE)
        pinned = set(pinned_keys)
        for key in plan.cold_misses:
            entry = self._entries.get(key)
            if entry is None or entry.state != GraphResidency.CAPTURING:
                raise ElasticGraphError("maintenance publication is stale")
            price = prices.get(key)
            if price is None:
                raise ElasticGraphError("publication omitted a measured graph price")
            self.publish_hot(key, price, pinned=key in pinned)
        self.commit_staged_hotset_replace(plan)
        self._promotions += len(plan.cold_misses)

    def commit_staged_hotset_replace(self, plan: ElasticStepPlan) -> None:
        """Commit victim retirement only after every new owner published."""
        self._validate_plan(plan, ElasticPlanKind.MAINTENANCE)
        if not plan.staged_hotset_replace:
            return
        for key in plan.cold_misses:
            entry = self._entries.get(key)
            if entry is None or not entry.hot:
                raise ElasticGraphError(
                    "staged hotset commit requires every new owner HOT"
                )
        for key in plan.victim_keys:
            entry = self._entries.get(key)
            if entry is None or entry.leases != frozenset((plan.transaction_id,)):
                raise ElasticGraphError("staged victim lease changed before commit")
            self._entries[key] = replace(
                entry,
                state=GraphResidency.COLD,
                leases=frozenset(),
            )
        self._evictions += len(plan.victim_keys)
        self._evicted_bytes += plan.reclaim_bytes

    def abort_staged_hotset_replace(self, plan: ElasticStepPlan) -> None:
        """Roll back an unused published candidate to the preceding HOT set."""
        self._validate_plan(plan, ElasticPlanKind.MAINTENANCE)
        if not plan.staged_hotset_replace:
            raise ElasticGraphError("only a staged hotset replacement can be aborted")
        for key in plan.cold_misses:
            entry = self._entries.get(key)
            if entry is None or not entry.hot or entry.leases:
                raise ElasticGraphError(
                    "staged hotset abort requires an unleased published candidate"
                )
            self._entries[key] = replace(entry, state=GraphResidency.COLD)
        for key in plan.victim_keys:
            entry = self._entries.get(key)
            if entry is None or entry.hot or entry.leases:
                raise ElasticGraphError(
                    "staged hotset abort requires an unleased retired victim"
                )
            self._entries[key] = replace(entry, state=GraphResidency.HOT_EVICTABLE)
        self._promotions -= len(plan.cold_misses)
        self._evictions -= len(plan.victim_keys)
        self._evicted_bytes -= plan.reclaim_bytes
        if self._promotions < 0 or self._evictions < 0 or self._evicted_bytes < 0:
            raise ElasticGraphError("staged hotset abort underflowed cache counters")
        self._trace.append((plan.transaction_id, "abort_staged", plan.fingerprint))

    def fail_maintenance(self, plan: ElasticStepPlan) -> None:
        self._validate_plan(plan, ElasticPlanKind.MAINTENANCE)
        for key in plan.cold_misses:
            entry = self._entries.get(key)
            if entry is not None and entry.state == GraphResidency.CAPTURING:
                self._entries[key] = replace(
                    entry, state=GraphResidency.COOLDOWN, leases=frozenset()
                )
        if plan.staged_hotset_replace:
            for key in plan.victim_keys:
                entry = self._entries.get(key)
                if entry is not None and plan.transaction_id in entry.leases:
                    self._entries[key] = replace(
                        entry,
                        leases=entry.leases.difference((plan.transaction_id,)),
                    )

    def release(self, transaction_id: str) -> None:
        for key, entry in tuple(self._entries.items()):
            if transaction_id in entry.leases:
                self._entries[key] = replace(
                    entry, leases=entry.leases.difference((transaction_id,))
                )

    cancel = release

    def observe_defer(self, plan: ElasticStepPlan) -> None:
        self._validate_plan(plan, ElasticPlanKind.DEFER)
        assert plan.defer_reason is not None
        self._deferrals += 1
        self._defer_reasons[plan.defer_reason] += 1
        self._trace.append((plan.transaction_id, "defer", plan.fingerprint))

    def _select_victims(
        self,
        deficit: int,
        protected: set[PhysicalReplayKey],
    ) -> tuple[tuple[PhysicalReplayKey, ...], tuple[str, ...], int]:
        if deficit <= 0:
            return (), (), 0
        candidates: list[tuple[int, int, str, ReclaimGroup]] = []
        for group_id, group in self._groups.items():
            entries = [self._entries[key] for key in group.keys]
            if any(key in protected for key in group.keys):
                continue
            if not all(entry.reclaimable for entry in entries):
                continue
            oldest = min(entry.last_used_epoch for entry in entries)
            enough = int(group.reclaimable_bytes < deficit)
            candidates.append((oldest, enough, group_id, group))
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        victims: list[PhysicalReplayKey] = []
        groups: list[str] = []
        reclaimed = 0
        for _oldest, _enough, group_id, group in candidates:
            groups.append(group_id)
            victims.extend(group.keys)
            reclaimed += group.reclaimable_bytes
            if reclaimed >= deficit:
                break
        return tuple(victims), tuple(groups), reclaimed

    def _select_complete_hotset_victims(
        self,
        protected: set[PhysicalReplayKey],
    ) -> tuple[
        tuple[PhysicalReplayKey, ...], tuple[str, ...], int
    ] | None:
        """Return every old HOT reclaim group, or fail if any cannot retire."""
        outside = {
            key
            for key, entry in self._entries.items()
            if entry.hot and key not in protected
        }
        if not outside:
            return (), (), 0
        groups = []
        covered: set[PhysicalReplayKey] = set()
        for group_id, group in sorted(self._groups.items()):
            group_keys = set(group.keys)
            if not group_keys.intersection(outside):
                continue
            if not group_keys.issubset(outside):
                return None
            entries = [self._entries[key] for key in group.keys]
            if not all(entry.reclaimable for entry in entries):
                return None
            groups.append(group)
            covered.update(group.keys)
        if covered != outside:
            return None
        return (
            tuple(key for group in groups for key in group.keys),
            tuple(group.group_id for group in groups),
            sum(group.reclaimable_bytes for group in groups),
        )

    def _defer(
        self,
        transaction_id: str,
        keys: tuple[PhysicalReplayKey, ...],
        hits: tuple[PhysicalReplayKey, ...],
        misses: tuple[PhysicalReplayKey, ...],
        protected: tuple[PhysicalReplayKey, ...],
        request_bytes: int,
        available_bytes: int,
        kv_transition: tuple[int, int] | None,
        reason: str,
        *,
        victims: tuple[PhysicalReplayKey, ...] = (),
        groups: tuple[str, ...] = (),
        reclaimed: int = 0,
        capture_loan: int = 0,
    ) -> ElasticStepPlan:
        return ElasticStepPlan(
            transaction_id=transaction_id,
            generation=self.generation,
            kind=ElasticPlanKind.DEFER,
            physical_keys=keys,
            hot_hits=hits,
            cold_misses=misses,
            protected_keys=protected,
            victim_keys=victims,
            reclaim_groups=groups,
            capture_order=(),
            kv_transition=kv_transition,
            request_bytes=request_bytes,
            available_bytes=available_bytes,
            capture_loan_bytes=capture_loan,
            reclaim_bytes=reclaimed,
            defer_reason=reason,
        )

    def _validate_plan(self, plan: ElasticStepPlan, kind: ElasticPlanKind) -> None:
        if plan.generation != self.generation:
            raise ElasticGraphError("elastic graph plan belongs to a stale generation")
        if plan.kind != kind:
            raise ElasticGraphError(f"expected a {kind.value} plan")

    def _require_generation(self, key: PhysicalReplayKey) -> None:
        if key.generation != self.generation:
            raise ElasticGraphError("physical graph key belongs to a stale generation")
