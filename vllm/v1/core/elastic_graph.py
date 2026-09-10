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

from vllm.v1.core.elastic_expert import ElasticExpertGrant


class ElasticGraphError(RuntimeError):
    """An elastic graph state transition violated the admission contract."""


@dataclass(frozen=True)
class ElasticRuntimeConfig:
    """Single activation authority shared by scheduler and Graph workers."""

    enabled: bool

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> ElasticRuntimeConfig:
        additional = getattr(vllm_config, "additional_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
        mode = getattr(compilation_config, "cudagraph_mode", None)
        mode_name = getattr(mode, "name", str(mode))
        return cls(
            enabled=bool(
                isinstance(additional, dict)
                and additional.get("elastic_gdn_backing", False)
                and not getattr(model_config, "enforce_eager", False)
                and mode_name != "NONE"
            )
        )


class GraphResidency(str, Enum):
    COLD = "cold"
    CAPTURING = "capturing"
    HOT_EVICTABLE = "hot_evictable"
    HOT_PINNED = "hot_pinned"


class ElasticPlanKind(str, Enum):
    USER = "user"
    MAINTENANCE = "maintenance"
    RECLAIM = "reclaim"
    PRESSURE_RECLAIM = "pressure_reclaim"
    DEFER = "defer"


class ElasticMaintenanceExecution(str, Enum):
    """Physical execution boundary for a COLD graph promotion."""

    COUPLED_USER = "coupled_user"
    GRAPH_ONLY = "graph_only"


class DispatchRepresentation(str, Enum):
    HOT_GRAPH = "hot_graph"
    COMPILED_ONLY = "compiled_only"
    INACTIVE = "inactive"
    FORBIDDEN = "forbidden"


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
        if self.piecewise_query_len_min_tokens is not None and (
            isinstance(self.piecewise_query_len_min_tokens, bool)
            or not isinstance(self.piecewise_query_len_min_tokens, int)
            or self.piecewise_query_len_min_tokens <= 0
        ):
            raise ValueError(
                "PIECEWISE query-length threshold must be a positive integer"
            )
        if tuple(sorted(set(self.full_query_lens))) != self.full_query_lens or any(
            query_len <= 0 for query_len in self.full_query_lens
        ):
            raise ValueError("FULL query lengths must be sorted unique positives")
        if tuple(
            sorted(set(self.compiled_piecewise_sizes))
        ) != self.compiled_piecewise_sizes or any(
            size <= 0 for size in self.compiled_piecewise_sizes
        ):
            raise ValueError("compiled PIECEWISE sizes must be sorted unique positives")

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
                raise ValueError("runtime Graph owners require unique execution order")

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
        if not isinstance(raw_owners, Sequence) or isinstance(raw_owners, (str, bytes)):
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
            if query_len_min_tokens is not None and (
                isinstance(query_len_min_tokens, bool)
                or not isinstance(query_len_min_tokens, int)
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
                    capability_contract=required_string(raw, "capability_contract"),
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
            verifier_configuration=required_string(payload, "verifier_configuration"),
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


def compute_elastic_runtime_generation_from_factors(
    factors: Mapping[str, Any],
) -> str:
    """Canonical generation computation shared by all runtime participants."""
    return hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()


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


EXECUTION_MANIFEST_SCHEMA = 2


@dataclass(frozen=True, order=True)
class OwnerInvocation:
    """Exact current invocation for one ordered runtime owner."""

    owner: str
    activation: str
    phase: str
    semantic_num_reqs: int
    physical_num_reqs: int
    live_num_tokens: int
    physical_num_tokens: int
    uniform_query_len: int | None
    active_loras: int
    requested_output_k: int
    executed_drafter_k: int
    execution_order: int
    generation: RuntimeGeneration

    def __post_init__(self) -> None:
        if (
            not self.owner
            or self.activation not in {"always", "speculative"}
            or self.phase not in {"decode", "mixed"}
        ):
            raise ValueError("owner invocation has invalid identity")
        for name in (
            "semantic_num_reqs",
            "physical_num_reqs",
            "live_num_tokens",
            "physical_num_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.physical_num_reqs < self.semantic_num_reqs:
            raise ValueError("physical request carrier underfills semantic requests")
        if self.physical_num_tokens < self.live_num_tokens:
            raise ValueError("physical token carrier underfills live tokens")
        if self.uniform_query_len is not None and (
            isinstance(self.uniform_query_len, bool)
            or not isinstance(self.uniform_query_len, int)
            or self.uniform_query_len <= 0
        ):
            raise ValueError("uniform query length must be a positive integer")
        for name in (
            "active_loras",
            "requested_output_k",
            "executed_drafter_k",
            "execution_order",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ExecutionManifest:
    """Immutable current work identity, independent of future residency."""

    generation: RuntimeGeneration
    invocations: tuple[OwnerInvocation, ...]
    request_ids: tuple[str, ...]
    per_request_query_lens: tuple[int, ...]
    per_request_is_prefilling: tuple[bool, ...]
    scheduled_draft_rows: tuple[int, ...]
    scheduled_encoder_inputs: tuple[tuple[str, tuple[int, ...]], ...]
    active_lora_ids: tuple[int, ...]
    requested_output_k: int
    executed_drafter_k: int
    schema: int = EXECUTION_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_MANIFEST_SCHEMA:
            raise ValueError("unsupported execution manifest schema")
        if not self.invocations or self.invocations[0].owner != "target":
            raise ValueError("execution manifest requires target first")
        owners = tuple(invocation.owner for invocation in self.invocations)
        if len(set(owners)) != len(owners):
            raise ValueError("execution manifest owners must be unique")
        orders = tuple(invocation.execution_order for invocation in self.invocations)
        if tuple(sorted(orders)) != orders or len(set(orders)) != len(orders):
            raise ValueError("execution manifest order must be sorted and unique")
        if any(
            invocation.generation != self.generation for invocation in self.invocations
        ):
            raise ValueError("execution manifest mixes runtime generations")
        if (
            not self.request_ids
            or len(set(self.request_ids)) != len(self.request_ids)
            or any(
                not isinstance(request_id, str) or not request_id
                for request_id in self.request_ids
            )
        ):
            raise ValueError("manifest request IDs must be non-empty and unique")
        if len(self.request_ids) != len(self.per_request_query_lens):
            raise ValueError("manifest request IDs and query lengths differ")
        if not self.per_request_query_lens or any(
            isinstance(length, bool) or not isinstance(length, int) or length <= 0
            for length in self.per_request_query_lens
        ):
            raise ValueError("manifest query lengths must be positive integers")
        if len(self.per_request_is_prefilling) != len(self.request_ids) or any(
            type(value) is not bool for value in self.per_request_is_prefilling
        ):
            raise ValueError("manifest prefill phases must align with requests")
        if len(self.scheduled_draft_rows) != len(self.per_request_query_lens) or any(
            isinstance(rows, bool) or not isinstance(rows, int) or rows < 0
            for rows in self.scheduled_draft_rows
        ):
            raise ValueError("manifest draft rows must align with requests")
        encoder_request_ids = tuple(
            request_id for request_id, _indices in self.scheduled_encoder_inputs
        )
        if (
            tuple(sorted(encoder_request_ids)) != encoder_request_ids
            or len(set(encoder_request_ids)) != len(encoder_request_ids)
            or not set(encoder_request_ids).issubset(self.request_ids)
            or any(
                not indices
                or tuple(sorted(set(indices))) != indices
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in indices
                )
                for _request_id, indices in self.scheduled_encoder_inputs
            )
        ):
            raise ValueError("manifest encoder schedule is not canonical")
        if tuple(sorted(set(self.active_lora_ids))) != self.active_lora_ids or any(
            isinstance(lora_id, bool) or not isinstance(lora_id, int) or lora_id <= 0
            for lora_id in self.active_lora_ids
        ):
            raise ValueError("manifest active LoRA IDs must be sorted unique positives")
        if any(
            invocation.active_loras != len(self.active_lora_ids)
            for invocation in self.invocations
        ):
            raise ValueError("manifest LoRA identity differs from Graph capture case")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.requested_output_k, self.executed_drafter_k)
        ):
            raise ValueError(
                "manifest speculative K values must be non-negative integers"
            )
        if any(
            invocation.requested_output_k != self.requested_output_k
            or invocation.executed_drafter_k != self.executed_drafter_k
            for invocation in self.invocations
        ):
            raise ValueError("manifest speculative K differs from owner invocation")
        if self.requested_output_k == 0 and self.executed_drafter_k != 0:
            raise ValueError("inactive speculation cannot execute drafter steps")
        if (
            self.requested_output_k > 0
            and self.executed_drafter_k < self.requested_output_k
        ):
            raise ValueError("physical drafter depth underfills requested output K")
        if any(rows > self.executed_drafter_k for rows in self.scheduled_draft_rows):
            raise ValueError("scheduled draft rows exceed physical drafter depth")
        semantic_num_reqs = len(self.request_ids)
        if any(
            invocation.semantic_num_reqs != semantic_num_reqs
            for invocation in self.invocations
        ):
            raise ValueError("manifest owner request cardinality differs")
        if self.invocations[0].live_num_tokens != sum(self.per_request_query_lens):
            raise ValueError("target invocation live tokens differ from request work")

    @cached_property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True)
class OwnerDispatch:
    invocation: OwnerInvocation
    representation: DispatchRepresentation
    physical_key: PhysicalReplayKey | None = None

    def __post_init__(self) -> None:
        if self.representation == DispatchRepresentation.HOT_GRAPH:
            if self.physical_key is None:
                raise ValueError("HOT_GRAPH dispatch requires a physical key")
            if self.physical_key.logical.owner != self.invocation.owner:
                raise ValueError("dispatch key owner differs from invocation")
            if self.physical_key.generation != self.invocation.generation:
                raise ValueError("dispatch key generation differs from invocation")
        elif self.physical_key is not None:
            raise ValueError("only HOT_GRAPH dispatch may carry a physical key")
        if self.representation in {
            DispatchRepresentation.INACTIVE,
            DispatchRepresentation.FORBIDDEN,
        }:
            raise ValueError("inactive or forbidden owners cannot enter dispatch")


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


ELASTIC_RESIDENCY_RECEIPT_SCHEMA = 1
ELASTIC_RESIDENCY_RECEIPT_SCHEMA_FINGERPRINT = hashlib.sha256(
    json.dumps(
        {
            "schema": ELASTIC_RESIDENCY_RECEIPT_SCHEMA,
            "receipt": (
                "generation",
                "transaction_id",
                "resident_bytes",
                "floor_bytes",
                "transition_floor_bytes",
                "peak_bytes",
                "cublas_workspace_bytes",
                "entries",
                "complete",
            ),
            "entry": (
                "key",
                "pinned",
                "resident_bytes",
                "local_pool_bytes",
                "reclaimable_bytes",
                "lease_ids",
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


@dataclass(frozen=True)
class ElasticResidencyEntry:
    """One physical executable in a worker residency publication."""

    key: PhysicalReplayKey
    pinned: bool
    resident_bytes: int
    local_pool_bytes: int
    reclaimable_bytes: int
    lease_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("resident_bytes", "local_pool_bytes", "reclaimable_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.reclaimable_bytes > self.resident_bytes:
            raise ValueError("reclaim proof exceeds resident bytes")
        if self.local_pool_bytes > self.resident_bytes:
            raise ValueError("local pool exceeds rank-safe resident bytes")
        if self.reclaimable_bytes > self.local_pool_bytes:
            raise ValueError("reclaim proof exceeds the local pool")
        # Pinning is current policy; reclaimable_bytes is the physical proof
        # available after a future explicit idle unpin. A lease, unlike a pin,
        # crosses an active ownership boundary and must still suppress proof.
        if tuple(sorted(set(self.lease_ids))) != self.lease_ids:
            raise ValueError("lease ids must be sorted and unique")
        if self.lease_ids and self.reclaimable_bytes:
            raise ValueError("a leased Graph entry cannot be reclaimable")


@dataclass(frozen=True)
class ElasticResidencyReceipt:
    """Complete generation-bound worker publication; never a positional ABI."""

    generation: RuntimeGeneration
    transaction_id: str | None
    resident_bytes: int
    floor_bytes: int
    transition_floor_bytes: int
    peak_bytes: int
    cublas_workspace_bytes: int
    entries: tuple[ElasticResidencyEntry, ...]
    complete: bool = True
    schema: int = ELASTIC_RESIDENCY_RECEIPT_SCHEMA
    schema_fingerprint: str = ELASTIC_RESIDENCY_RECEIPT_SCHEMA_FINGERPRINT

    def __post_init__(self) -> None:
        if self.schema != ELASTIC_RESIDENCY_RECEIPT_SCHEMA:
            raise ValueError("unsupported elastic residency receipt schema")
        if self.schema_fingerprint != ELASTIC_RESIDENCY_RECEIPT_SCHEMA_FINGERPRINT:
            raise ValueError("elastic residency receipt schema fingerprint mismatch")
        if not self.complete:
            raise ValueError("partial elastic residency receipts are forbidden")
        for name in (
            "resident_bytes",
            "floor_bytes",
            "transition_floor_bytes",
            "peak_bytes",
            "cublas_workspace_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.floor_bytes > self.resident_bytes:
            raise ValueError("physical floor exceeds aggregate resident bytes")
        if self.transition_floor_bytes > self.peak_bytes:
            raise ValueError("transition floor exceeds measured peak")
        if self.peak_bytes < self.resident_bytes:
            raise ValueError("measured peak is below aggregate resident bytes")
        keys = tuple(entry.key for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ValueError("elastic residency receipt contains duplicate keys")
        if any(entry.key.generation != self.generation for entry in self.entries):
            raise ValueError("elastic residency receipt mixes runtime generations")
        if tuple(sorted(keys, key=lambda key: key.identity)) != keys:
            raise ValueError("elastic residency entries must be identity-sorted")
        if sum(entry.resident_bytes for entry in self.entries) > self.resident_bytes:
            raise ValueError(
                "elastic residency entries exceed aggregate resident bytes"
            )

    @cached_property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True)
class ElasticAdmissionLoan:
    """One FIFO scheduler-to-worker external-memory reservation."""

    step_key: tuple[int, ...] | None
    grant_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.grant_bytes, bool)
            or not isinstance(self.grant_bytes, int)
            or self.grant_bytes < 0
        ):
            raise ValueError("elastic admission loan must be non-negative")


@dataclass(frozen=True)
class ElasticAdmissionSnapshot:
    """Stable controller state used by differential trace replay."""

    generation: RuntimeGeneration
    pending_loans: tuple[ElasticAdmissionLoan, ...]
    pending_maintenance: tuple[str, str, tuple[int, ...] | None] | None
    step_key: tuple[int, ...] | None
    recapture_pending_key: tuple[int, ...] | None
    resident_bytes: int
    pinned_resident_bytes: int
    evictable_resident_bytes: int
    floor_bytes: int
    transition_floor_bytes: int
    cublas_workspace_bytes: int
    entry_states: tuple[tuple[str, str, tuple[str, ...]], ...]
    stats: ElasticGraphStats


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
class _AdmissionMutationSnapshot:
    entries: dict[PhysicalReplayKey, ElasticGraphEntry | None]
    epoch: int
    hot_hits: int
    cold_misses: int
    promotions: int
    evictions: int
    evicted_bytes: int
    trace: tuple[tuple[str, str, str], ...]


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
    execution_manifest: ExecutionManifest | None = None
    current_dispatch: tuple[OwnerDispatch, ...] = ()
    successor_keys: tuple[PhysicalReplayKey, ...] = ()
    maintenance_execution: ElasticMaintenanceExecution | None = None
    reuse_rank_consensus: bool = False
    expert_grant: ElasticExpertGrant | None = None

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
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.kind == ElasticPlanKind.USER and self.cold_misses:
            raise ValueError("a user plan cannot contain cold graph misses")
        if self.kind == ElasticPlanKind.MAINTENANCE and not self.cold_misses:
            raise ValueError("maintenance requires at least one cold miss")
        if self.kind == ElasticPlanKind.MAINTENANCE:
            if self.maintenance_execution is None:
                raise ValueError("maintenance requires an execution boundary")
        elif self.maintenance_execution is not None:
            raise ValueError("only maintenance may carry an execution boundary")
        if self.kind in {
            ElasticPlanKind.RECLAIM,
            ElasticPlanKind.PRESSURE_RECLAIM,
        } and (self.cold_misses or not self.victim_keys):
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
        if set(self.victim_keys).intersection(self.physical_keys):
            raise ValueError("a current physical key cannot be selected as a victim")
        if self.kind == ElasticPlanKind.USER and self.victim_keys:
            raise ValueError("a user plan cannot evict graph residency")
        if self.execution_manifest is None:
            if self.current_dispatch:
                raise ValueError("current dispatch requires an execution manifest")
        else:
            if self.execution_manifest.generation != self.generation:
                raise ValueError("execution manifest generation differs from plan")
            invocation_owners = tuple(
                invocation.owner for invocation in self.execution_manifest.invocations
            )
            dispatch_owners = tuple(
                dispatch.invocation.owner for dispatch in self.current_dispatch
            )
            if dispatch_owners != invocation_owners:
                raise ValueError("current dispatch does not cover exact owner order")
            if (
                tuple(dispatch.invocation for dispatch in self.current_dispatch)
                != self.execution_manifest.invocations
            ):
                raise ValueError("dispatch invocation differs from manifest")
            current_keys = tuple(
                dispatch.physical_key
                for dispatch in self.current_dispatch
                if dispatch.representation == DispatchRepresentation.HOT_GRAPH
            )
            if len(set(current_keys)) != len(current_keys):
                raise ValueError("current dispatch contains duplicate physical keys")
            if not set(current_keys).issubset(self.physical_keys):
                raise ValueError("current dispatch key is outside plan residency")
            if self.kind == ElasticPlanKind.USER and not set(current_keys).issubset(
                self.hot_hits
            ):
                raise ValueError("every USER Graph dispatch key must be HOT")
            allowed_residency = set(current_keys).union(self.successor_keys)
            if not set(self.physical_keys).issubset(allowed_residency):
                raise ValueError(
                    "execution residency contains neither current nor successor keys"
                )
            protected_successors = set(self.physical_keys).intersection(
                self.successor_keys
            ) - set(current_keys)
            if not protected_successors.issubset(self.protected_keys):
                raise ValueError("resident successor keys must be explicitly protected")
        if any(key.generation != self.generation for key in self.successor_keys):
            raise ValueError("successor residency mixes runtime generations")
        if len(set(self.successor_keys)) != len(self.successor_keys):
            raise ValueError("successor residency contains duplicate keys")
        if self.reuse_rank_consensus and not self.reusable_decode_consensus_epoch:
            raise ValueError(
                "rank consensus reuse requires mutation-free HOT text decode"
            )

    @property
    def reusable_decode_consensus_epoch(self) -> bool:
        """Whether this plan can reuse a scheduler-declared rank boundary."""
        manifest = self.execution_manifest
        return bool(
            self.kind == ElasticPlanKind.USER
            and manifest is not None
            and not any(manifest.per_request_is_prefilling)
            and not manifest.scheduled_encoder_inputs
            and not manifest.active_lora_ids
            and self.kv_transition is None
            and not self.cold_misses
            and not self.victim_keys
            and not self.capture_order
            and all(
                dispatch.representation == DispatchRepresentation.HOT_GRAPH
                for dispatch in self.current_dispatch
            )
        )

    @cached_property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    @cached_property
    def execution_epoch_fingerprint(self) -> str:
        """Stable execution identity shared by consecutive decode steps.

        A transaction id identifies one scheduler/worker mutation and lease
        lifetime; it intentionally changes every step. Admission byte ledgers
        are scheduler evidence and are not consumed by a HOT USER worker step.
        The epoch therefore binds every worker-consumed state and execution
        field while leaving transaction and advisory byte accounting outside.
        """
        payload = {
            "generation": self.generation,
            "kind": self.kind,
            "physical_keys": self.physical_keys,
            "hot_hits": self.hot_hits,
            "cold_misses": self.cold_misses,
            "protected_keys": self.protected_keys,
            "victim_keys": self.victim_keys,
            "reclaim_groups": self.reclaim_groups,
            "capture_order": self.capture_order,
            "kv_transition": self.kv_transition,
            "defer_reason": self.defer_reason,
            "execution_manifest": self.execution_manifest,
            "current_dispatch": self.current_dispatch,
            "successor_keys": self.successor_keys,
            "maintenance_execution": self.maintenance_execution,
            "expert_grant": self.expert_grant,
        }
        return _fingerprint(payload)


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
            semantic_uniform is not None and num_tokens == num_reqs * semantic_uniform
        )
        for owner_policy in ordered_owners:
            if owner_policy.activation == "speculative" and num_spec_tokens <= 0:
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
            if (
                owner_policy.token_source == "step"
                and semantic_uniform is None
                and mode == "FULL"
            ):
                # FULL replay for token-major owners is legal only for a
                # scheduler-declared uniform decode. A mixed/prefill q1 tail
                # can share the same numeric qlen while requiring PIECEWISE.
                mode = owner_policy.piecewise_mode
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
                owner_compiled_sizes = frozenset(owner_policy.compiled_piecewise_sizes)
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


def build_execution_manifest(
    *,
    step_key: tuple[int, int, int, int, int],
    request_ids: Sequence[str],
    per_request_query_lens: Sequence[int],
    per_request_is_prefilling: Sequence[bool],
    scheduled_draft_rows: Sequence[int],
    scheduled_encoder_inputs: Mapping[str, Sequence[int]] | None = None,
    active_lora_ids: Sequence[int] = (),
    requested_output_k: int,
    executed_drafter_k: int,
    phase: str,
    generation: RuntimeGeneration,
    policy: GraphExecutionPolicy,
    max_num_batched_tokens: int,
    active_loras: int = 0,
    physical_keys: Sequence[PhysicalReplayKey] | None = None,
) -> tuple[ExecutionManifest, tuple[OwnerDispatch, ...]]:
    """Resolve current owner work without consulting successor residency."""
    query_lens = tuple(per_request_query_lens)
    is_prefilling = tuple(per_request_is_prefilling)
    draft_rows = tuple(scheduled_draft_rows)
    ordered_request_ids = tuple(request_ids)
    encoder_inputs = tuple(
        sorted(
            (
                request_id,
                tuple(sorted(set(indices))),
            )
            for request_id, indices in (scheduled_encoder_inputs or {}).items()
        )
    )
    lora_ids = tuple(sorted(set(active_lora_ids)))
    if not query_lens:
        raise ElasticGraphError("current execution requires at least one request")
    if len(ordered_request_ids) != len(query_lens):
        raise ElasticGraphError("request IDs do not align with current execution")
    if len(is_prefilling) != len(query_lens) or any(
        type(value) is not bool for value in is_prefilling
    ):
        raise ElasticGraphError("request prefill phases do not align with execution")
    if len(draft_rows) != len(query_lens):
        raise ElasticGraphError("scheduled draft rows do not align with requests")
    if active_loras != len(lora_ids):
        raise ElasticGraphError(
            "active LoRA identity differs from the exact Graph capture case"
        )
    full, step_k, physical_x, step_tokens, step_uniform = step_key
    if step_k != requested_output_k:
        raise ElasticGraphError(
            "execution step K differs from requested draft output K"
        )
    if requested_output_k == 0 and executed_drafter_k != 0:
        raise ElasticGraphError("inactive speculation cannot execute drafter steps")
    if requested_output_k > 0 and executed_drafter_k < requested_output_k:
        raise ElasticGraphError(
            "physical drafter depth cannot underfill requested draft output K"
        )
    if any(rows > executed_drafter_k for rows in draft_rows):
        raise ElasticGraphError(
            "scheduled target draft rows exceed physical drafter depth"
        )
    semantic_x = len(query_lens)
    if physical_x < semantic_x:
        raise ElasticGraphError("execution carrier underfills semantic requests")
    uniform_step = query_lens[0] if len(set(query_lens)) == 1 else None
    exact_decode = phase == "decode" and uniform_step is not None
    if phase == "mixed":
        if physical_x != semantic_x or step_uniform != 0:
            raise ElasticGraphError(
                "mixed current execution cannot inherit a successor carrier"
            )
        live_step_tokens = sum(query_lens)
        if live_step_tokens > max_num_batched_tokens:
            raise ElasticGraphError("current execution exceeds max_num_batched_tokens")
        expected_step_tokens = 1 << (live_step_tokens - 1).bit_length()
    elif exact_decode:
        if step_uniform != uniform_step:
            raise ElasticGraphError("decode execution lost exact query length")
        expected_step_tokens = physical_x * uniform_step
    else:
        raise ElasticGraphError("decode execution has inconsistent query geometry")
    if step_tokens != expected_step_tokens:
        raise ElasticGraphError(
            "current execution token carrier differs from observed work: "
            f"observed={sum(query_lens)} carrier={step_tokens} "
            f"expected={expected_step_tokens}"
        )
    if full and not exact_decode:
        raise ElasticGraphError("FULL current execution requires exact decode")
    exact_keys = resolve_step_physical_keys(
        step_key,
        generation,
        max_num_batched_tokens=max_num_batched_tokens,
        policy=policy,
    )
    current_keys = exact_keys if physical_keys is None else tuple(physical_keys)
    exact_by_owner = {key.logical.owner: key for key in exact_keys}
    if {key.logical.owner for key in current_keys} != set(exact_by_owner):
        raise ElasticGraphError("current physical owner set differs from active owners")
    for key in current_keys:
        if key.generation != generation:
            raise ElasticGraphError("current physical owner set has stale generation")
        policy.owner_policy(key.logical.owner).validate_physical_key(key)
        exact_key = exact_by_owner[key.logical.owner]
        if key != exact_key:
            raise ElasticGraphError(
                "current physical key differs from the execution step: "
                f"owner={key.logical.owner} expected={exact_key.identity} "
                f"actual={key.identity}"
            )
    keys_by_owner = {key.logical.owner: key for key in current_keys}
    if len(keys_by_owner) != len(current_keys):
        raise ElasticGraphError("current physical owner set contains duplicates")

    invocations: list[OwnerInvocation] = []
    dispatches: list[OwnerDispatch] = []
    ordered_policies = tuple(
        sorted(policy.owners, key=lambda owner: owner.execution_order or 0)
    )
    for owner_policy in ordered_policies:
        if owner_policy.activation == "speculative" and requested_output_k <= 0:
            continue
        if owner_policy.token_source == "step":
            live_tokens = sum(query_lens)
            owner_uniform = uniform_step
        elif owner_policy.token_source == "requests":
            live_tokens = semantic_x
            owner_uniform = 1
        elif owner_policy.token_source == "fixed_query":
            assert owner_policy.fixed_query_len is not None
            owner_uniform = owner_policy.fixed_query_len
            live_tokens = semantic_x * owner_uniform
        else:
            raise ElasticGraphError(
                "execution manifest requires an explicit owner token source"
            )

        key = keys_by_owner.pop(owner_policy.owner, None)
        if key is not None:
            physical_tokens = key.logical.token_bucket
            representation = DispatchRepresentation.HOT_GRAPH
        else:
            mode = owner_policy.mode_for(owner_uniform)
            if (
                owner_policy.token_source == "step"
                and phase == "mixed"
                and mode == "FULL"
            ):
                mode = owner_policy.piecewise_mode
            if mode == "NONE":
                physical_tokens = live_tokens
            elif mode == "PIECEWISE":
                physical_tokens = min(
                    1 << (live_tokens - 1).bit_length(), max_num_batched_tokens
                )
                if physical_tokens not in owner_policy.compiled_piecewise_sizes:
                    raise ElasticGraphError(
                        "active owner has neither a Graph nor a compiled-only route: "
                        f"owner={owner_policy.owner} tokens={live_tokens}"
                    )
            else:
                raise ElasticGraphError(
                    "active FULL owner has no current physical Graph key: "
                    f"owner={owner_policy.owner}"
                )
            representation = DispatchRepresentation.COMPILED_ONLY

        owner_physical_x = physical_x if key is None else key.physical_num_reqs
        invocation = OwnerInvocation(
            owner=owner_policy.owner,
            activation=owner_policy.activation,
            phase=phase,
            semantic_num_reqs=semantic_x,
            physical_num_reqs=owner_physical_x,
            live_num_tokens=live_tokens,
            physical_num_tokens=physical_tokens,
            uniform_query_len=owner_uniform,
            active_loras=active_loras,
            requested_output_k=requested_output_k,
            executed_drafter_k=executed_drafter_k,
            execution_order=owner_policy.execution_order or 0,
            generation=generation,
        )
        dispatches.append(
            OwnerDispatch(
                invocation=invocation,
                representation=representation,
                physical_key=key,
            )
        )
        invocations.append(invocation)
    if keys_by_owner:
        raise ElasticGraphError(
            "current physical owner set contains inactive owners: "
            f"owners={tuple(sorted(keys_by_owner))}"
        )
    manifest = ExecutionManifest(
        generation=generation,
        invocations=tuple(invocations),
        request_ids=ordered_request_ids,
        per_request_query_lens=query_lens,
        per_request_is_prefilling=is_prefilling,
        scheduled_draft_rows=draft_rows,
        scheduled_encoder_inputs=encoder_inputs,
        active_lora_ids=lora_ids,
        requested_output_k=requested_output_k,
        executed_drafter_k=executed_drafter_k,
    )
    return manifest, tuple(dispatches)


def execution_manifest_phase_from_step_key(
    step_key: tuple[int, int, int, int, int] | None,
) -> str | None:
    """Return the physical execution lane encoded by a canonical step key.

    A lifecycle-pure decode with variable accepted draft rows uses the same
    PIECEWISE lane as mixed work. Only an exact uniform-query key carries a
    non-zero query-length marker and therefore enters the decode lane.
    """
    if step_key is None:
        return None
    return "decode" if step_key[4] > 0 else "mixed"


def canonical_execution_request_order(
    num_tokens_per_request: Mapping[str, int],
    *,
    is_prefilling_by_request: Mapping[str, bool],
    decode_query_len: int,
) -> tuple[str, ...]:
    """Mirror the target InputBatch lifecycle ordering without mutation."""
    if decode_query_len <= 0:
        raise ValueError("decode query length must be positive")
    request_ids = num_tokens_per_request.keys()
    missing = request_ids - is_prefilling_by_request.keys()
    extra = is_prefilling_by_request.keys() - request_ids
    if missing or extra:
        raise ValueError(
            "execution lifecycle identity differs from scheduled requests: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    return tuple(
        sorted(
            num_tokens_per_request,
            key=lambda request_id: (
                is_prefilling_by_request[request_id],
                num_tokens_per_request[request_id] != decode_query_len,
                num_tokens_per_request[request_id],
            ),
        )
    )


def short_decode_inventory_xs(max_x: int) -> tuple[int, ...]:
    """Return the bounded physical cohort classes through ``max_x``."""
    if isinstance(max_x, bool) or not isinstance(max_x, int) or max_x <= 0:
        raise ValueError("max_x must be a positive integer")
    xs: list[int] = []
    x = 1
    while x <= max_x:
        xs.append(x)
        x <<= 1
    if xs[-1] != max_x:
        xs.append(max_x)
    return tuple(xs)


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
    xs = short_decode_inventory_xs(max_x)
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


class ElasticAdmissionController:
    """Scheduler-owned admission, loan and residency state machine.

    CUDA teardown/publication is performed by the caller.  This object owns
    policy and FIFO loan state and can therefore be replayed without Scheduler.
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
        self._pending_loans: deque[ElasticAdmissionLoan] = deque()
        self._pending_maintenance_plan: ElasticStepPlan | None = None
        self._pending_maintenance_step_key: tuple[int, ...] | None = None
        self._transaction_seq = 0
        self._last_receipt: ElasticResidencyReceipt | None = None
        self._pre_mutation_snapshots: dict[str, _AdmissionMutationSnapshot] = {}
        # Settled execution shape and the one cold shape awaiting its first
        # authoritative worker measurement.
        self.step_key: tuple[int, ...] | None = None
        self.recapture_pending_key: tuple[int, ...] | None = None
        # Runtime measurements replace provisional discovery loans; they are
        # evidence, never a fixed reserve.
        self.measured_bytes: dict[tuple[int, ...], int] = {}
        self.capture_envelopes: dict[tuple[int, ...], tuple[int, int, int]] = {}
        self._capture_envelope_provenance: dict[
            tuple[int, ...],
            tuple[
                tuple[int, int, int],
                tuple[tuple[str, int], ...] | None,
            ],
        ] = {}
        self.resident_bytes = 0
        self.pinned_resident_bytes = 0
        self.evictable_resident_bytes = 0
        # Actual and prospective non-reclaimable allocator floors.
        self.floor_bytes = 0
        self.transition_floor_bytes = 0
        self.cublas_workspace_bytes = 0
        self.last_maintenance_step_key: tuple[int, ...] | None = None
        self.idle_cleanup_started_at: float | None = None
        self.last_capacity_receipt: tuple[object, ...] | None = None

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

    @property
    def pending_loans(self) -> tuple[ElasticAdmissionLoan, ...]:
        return tuple(self._pending_loans)

    @property
    def latest_loan(self) -> ElasticAdmissionLoan | None:
        return self._pending_loans[-1] if self._pending_loans else None

    @property
    def pending_maintenance_plan(self) -> ElasticStepPlan | None:
        return self._pending_maintenance_plan

    @property
    def pending_maintenance_step_key(self) -> tuple[int, ...] | None:
        return self._pending_maintenance_step_key

    @property
    def last_receipt(self) -> ElasticResidencyReceipt | None:
        return self._last_receipt

    @property
    def snapshot(self) -> ElasticAdmissionSnapshot:
        return ElasticAdmissionSnapshot(
            generation=self.generation,
            pending_loans=tuple(self._pending_loans),
            pending_maintenance=(
                None
                if self._pending_maintenance_plan is None
                else (
                    self._pending_maintenance_plan.kind.value,
                    self._pending_maintenance_plan.transaction_id,
                    self._pending_maintenance_step_key,
                )
            ),
            step_key=self.step_key,
            recapture_pending_key=self.recapture_pending_key,
            resident_bytes=self.resident_bytes,
            pinned_resident_bytes=self.pinned_resident_bytes,
            evictable_resident_bytes=self.evictable_resident_bytes,
            floor_bytes=self.floor_bytes,
            transition_floor_bytes=self.transition_floor_bytes,
            cublas_workspace_bytes=self.cublas_workspace_bytes,
            entry_states=tuple(
                sorted(
                    (
                        key.identity,
                        entry.state.value,
                        tuple(sorted(entry.leases)),
                    )
                    for key, entry in self._entries.items()
                )
            ),
            stats=self.stats,
        )

    def next_transaction_id(self) -> str:
        self._transaction_seq += 1
        return f"elastic-{self._transaction_seq:020d}"

    def reserve_loan(
        self,
        step_key: tuple[int, ...] | None,
        grant_bytes: int,
    ) -> ElasticAdmissionLoan:
        loan = ElasticAdmissionLoan(step_key, grant_bytes)
        self._pending_loans.append(loan)
        return loan

    def replace_latest_loan(
        self,
        step_key: tuple[int, ...] | None,
        grant_bytes: int,
    ) -> ElasticAdmissionLoan:
        if not self._pending_loans:
            raise ElasticGraphError("cannot replace a missing elastic loan")
        loan = ElasticAdmissionLoan(step_key, grant_bytes)
        self._pending_loans[-1] = loan
        return loan

    def settle_next_loan(self) -> ElasticAdmissionLoan:
        if not self._pending_loans:
            raise ElasticGraphError("cannot settle a missing elastic loan")
        return self._pending_loans.popleft()

    def cancel_latest_loan(self) -> ElasticAdmissionLoan:
        if not self._pending_loans:
            raise ElasticGraphError("cannot cancel a missing elastic loan")
        return self._pending_loans.pop()

    def clear_loans(self) -> None:
        self._pending_loans.clear()

    def arm_maintenance(
        self,
        plan: ElasticStepPlan,
        step_key: tuple[int, ...] | None,
    ) -> None:
        if plan.generation != self.generation:
            raise ElasticGraphError("pending maintenance has a stale generation")
        if plan.kind == ElasticPlanKind.MAINTENANCE and step_key is None:
            raise ElasticGraphError("capture maintenance requires a step key")
        if (
            plan.kind
            in {
                ElasticPlanKind.RECLAIM,
                ElasticPlanKind.PRESSURE_RECLAIM,
            }
            and step_key is not None
        ):
            raise ElasticGraphError("pressure reclaim cannot carry a step key")
        if plan.kind not in {
            ElasticPlanKind.MAINTENANCE,
            ElasticPlanKind.RECLAIM,
            ElasticPlanKind.PRESSURE_RECLAIM,
        }:
            raise ElasticGraphError("only physical maintenance can be pending")
        self._pending_maintenance_plan = plan
        self._pending_maintenance_step_key = step_key

    def clear_maintenance(self) -> None:
        self._pending_maintenance_plan = None
        self._pending_maintenance_step_key = None

    def require_armed_maintenance_discardable(
        self, expected_transaction_id: str
    ) -> ElasticStepPlan:
        """Validate that an exact serving proposal has not begun mutation."""
        plan = self._pending_maintenance_plan
        if plan is None or plan.transaction_id != expected_transaction_id:
            raise ElasticGraphError("armed maintenance transaction changed")
        if plan.kind != ElasticPlanKind.MAINTENANCE:
            raise ElasticGraphError("exclusive maintenance cannot be discarded")
        if plan.maintenance_execution != ElasticMaintenanceExecution.COUPLED_USER:
            raise ElasticGraphError("graph-only maintenance cannot be discarded")
        capturing = any(
            self._entries.get(key) is not None
            and self._entries[key].state == GraphResidency.CAPTURING
            for key in plan.cold_misses
        )
        if expected_transaction_id in self._pre_mutation_snapshots or capturing:
            raise ElasticGraphError(
                "armed maintenance cannot be discarded after physical mutation"
            )
        return plan

    def discard_armed_maintenance(self, expected_transaction_id: str) -> None:
        """Discard one exact, validated pre-mutation serving proposal."""
        self.require_armed_maintenance_discardable(expected_transaction_id)
        self.clear_maintenance()

    def max_pending_grant(self) -> int:
        return max((loan.grant_bytes for loan in self._pending_loans), default=0)

    def record_measurement(
        self,
        step_key: tuple[int, ...],
        measured_bytes: int,
        *,
        keep_max: bool = False,
    ) -> None:
        if measured_bytes < 0:
            raise ElasticGraphError("elastic measurement cannot be negative")
        if keep_max:
            measured_bytes = max(
                measured_bytes,
                self.measured_bytes.get(step_key, 0),
            )
        self.measured_bytes[step_key] = measured_bytes

    def record_capture_envelope(
        self,
        owner_key: tuple[int, ...],
        envelope: tuple[int, int, int],
        *,
        merge_max: bool = False,
        resident_key_bytes: Mapping[str, int] | Iterable[tuple[str, int]] | None = None,
    ) -> None:
        if any(value < 0 for value in envelope):
            raise ElasticGraphError("elastic capture envelope cannot be negative")
        if resident_key_bytes is None:
            provenance = None
        else:
            provenance_map = dict(resident_key_bytes)
            if any(
                not isinstance(identity, str)
                or not identity
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for identity, value in provenance_map.items()
            ):
                raise ElasticGraphError(
                    "elastic capture provenance requires key identities and byte counts"
                )
            if sum(provenance_map.values()) > envelope[0]:
                raise ElasticGraphError(
                    "elastic capture provenance exceeds the destination endpoint"
                )
            provenance = tuple(sorted(provenance_map.items()))
        if merge_max:
            has_prior = owner_key in self.capture_envelopes
            prior = self.capture_envelopes.get(owner_key, (0, 0, 0))
            prior_evidence = self._capture_envelope_provenance.get(owner_key)
            prior_provenance = (
                prior_evidence[1]
                if prior_evidence is not None and prior_evidence[0] == prior
                else None
            )
            if not has_prior:
                selected_provenance = provenance
            else:
                prior_provenance_map = dict(prior_provenance or ())
                selected_provenance = (
                    None
                    if provenance is None or prior_provenance is None
                    else tuple(
                        sorted(
                            (
                                identity,
                                min(value, prior_provenance_map[identity]),
                            )
                            for identity, value in provenance
                            if identity in prior_provenance_map
                        )
                    )
                )
            envelope = tuple(max(prior[index], envelope[index]) for index in range(3))
            provenance = selected_provenance
        self.capture_envelopes[owner_key] = envelope
        self._capture_envelope_provenance[owner_key] = envelope, provenance

    def capture_envelope_resident_key_bytes(
        self, owner_key: tuple[int, ...]
    ) -> tuple[tuple[str, int], ...] | None:
        """Return source receipt key bytes for the still-selected endpoint."""
        envelope = self.capture_envelopes.get(owner_key)
        evidence = self._capture_envelope_provenance.get(owner_key)
        if envelope is None or evidence is None or evidence[0] != envelope:
            return None
        return evidence[1]

    def mark_recapture(self, step_key: tuple[int, ...]) -> None:
        self.recapture_pending_key = step_key

    def finish_recapture(self, step_key: tuple[int, ...]) -> None:
        if self.recapture_pending_key != step_key:
            raise ElasticGraphError("settled recapture key differs from pending key")
        self.recapture_pending_key = None

    def publish_step_key(self, step_key: tuple[int, ...] | None) -> None:
        self.step_key = step_key

    def capacity_receipt_changed(self, receipt: tuple[object, ...]) -> bool:
        return receipt != self.last_capacity_receipt

    def remember_capacity_receipt(self, receipt: tuple[object, ...]) -> None:
        self.last_capacity_receipt = receipt

    def publish_physical_accounting(
        self,
        *,
        resident_bytes: int,
        floor_bytes: int,
        transition_floor_bytes: int,
        cublas_workspace_bytes: int,
        maintenance_step_key: tuple[int, ...] | None,
    ) -> None:
        if not 0 <= floor_bytes <= transition_floor_bytes <= resident_bytes:
            raise ElasticGraphError("elastic physical accounting is inconsistent")
        if cublas_workspace_bytes < 0:
            raise ElasticGraphError("elastic workspace cannot be negative")
        self.resident_bytes = resident_bytes
        self.floor_bytes = floor_bytes
        self.transition_floor_bytes = transition_floor_bytes
        if cublas_workspace_bytes:
            self.cublas_workspace_bytes = cublas_workspace_bytes
        self.last_maintenance_step_key = maintenance_step_key

    def idle_cleanup_expired(
        self,
        *,
        required: bool,
        now: float,
        timeout_s: float,
    ) -> bool:
        if not required:
            self.idle_cleanup_started_at = None
            return False
        if self.idle_cleanup_started_at is None:
            self.idle_cleanup_started_at = now
            return False
        return now - self.idle_cleanup_started_at >= timeout_s

    def accept_residency_receipt(
        self,
        receipt: ElasticResidencyReceipt,
        *,
        expected_transaction_id: str | None = None,
    ) -> tuple[int, int]:
        """Validate and atomically publish one complete physical receipt.

        Returns pinned and evictable resident bytes.  The complete receipt is
        retained as controller authority only after Graph state synchronization
        succeeds.
        """
        self.validate_residency_receipt(
            receipt,
            expected_transaction_id=expected_transaction_id,
        )
        parsed = self._residency_receipt_rows(receipt)
        self.synchronize_hot(parsed)
        self._last_receipt = receipt
        pinned_bytes = sum(
            price.resident_bytes for _key, price, pinned, _proof in parsed if pinned
        )
        evictable_bytes = sum(
            price.resident_bytes for _key, price, pinned, _proof in parsed if not pinned
        )
        self.resident_bytes = receipt.resident_bytes
        self.pinned_resident_bytes = pinned_bytes
        self.evictable_resident_bytes = evictable_bytes
        self.floor_bytes = receipt.floor_bytes
        self.transition_floor_bytes = receipt.transition_floor_bytes
        if receipt.cublas_workspace_bytes:
            self.cublas_workspace_bytes = receipt.cublas_workspace_bytes
        if receipt.transaction_id is not None:
            self._pre_mutation_snapshots.pop(receipt.transaction_id, None)
        return pinned_bytes, evictable_bytes

    @staticmethod
    def _residency_receipt_rows(
        receipt: ElasticResidencyReceipt,
    ) -> list[tuple[PhysicalReplayKey, GraphPrice, bool, int]]:
        return [
            (
                entry.key,
                GraphPrice(
                    resident_bytes=entry.resident_bytes,
                    capture_peak_bytes=entry.resident_bytes,
                    reclaim_group=f"private-pool:{entry.key.identity}",
                ),
                entry.pinned or entry.reclaimable_bytes == 0,
                entry.reclaimable_bytes,
            )
            for entry in receipt.entries
        ]

    def validate_residency_publication(
        self,
        receipt: ElasticResidencyReceipt,
        *,
        expected_transaction_id: str | None = None,
        required_hot_keys: Iterable[PhysicalReplayKey] = (),
        releasing_transaction_id: str | None = None,
    ) -> None:
        """Validate the receipt's post-release HOT replacement read-only."""
        self.validate_residency_receipt(
            receipt,
            expected_transaction_id=expected_transaction_id,
            required_hot_keys=required_hot_keys,
        )
        rows = self._residency_receipt_rows(receipt)
        observed = {key: price for key, price, _pinned, _proof in rows}
        observed_ids = tuple(sorted(key.identity for key in observed))
        for key, entry in self._entries.items():
            remaining_leases = entry.leases.difference(
                () if releasing_transaction_id is None else (releasing_transaction_id,)
            )
            if entry.hot and key not in observed and remaining_leases:
                raise ElasticGraphError(
                    "worker receipt dropped a scheduler-leased graph: "
                    f"missing={key.identity!r} "
                    f"leases={tuple(sorted(remaining_leases))!r} "
                    f"observed={observed_ids!r}"
                )
        for key, price in observed.items():
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.price is not None
                and entry.price.reclaim_group != price.reclaim_group
            ):
                raise ElasticGraphError(
                    "graph reclaim identity changed inside one generation"
                )

    def validate_residency_receipt(
        self,
        receipt: ElasticResidencyReceipt,
        *,
        expected_transaction_id: str | None = None,
        required_hot_keys: Iterable[PhysicalReplayKey] = (),
    ) -> None:
        """Validate a complete worker receipt without changing controller state."""
        if receipt.generation != self.generation:
            raise ElasticGraphError(
                "worker residency generation differs from controller"
            )
        if (
            expected_transaction_id is not None
            and receipt.transaction_id != expected_transaction_id
        ):
            raise ElasticGraphError(
                "worker residency transaction differs from controller"
            )
        for entry in receipt.entries:
            if entry.lease_ids:
                raise ElasticGraphError(
                    "worker published a HOT receipt before releasing leases"
                )
        required = set(required_hot_keys)
        for key in required:
            self._require_generation(key)
        missing = required.difference(entry.key for entry in receipt.entries)
        if missing:
            raise ElasticGraphError(
                "worker residency receipt omitted required HOT keys: "
                f"{tuple(sorted(key.identity for key in missing))!r}"
            )

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

    def unpin_idle(self) -> tuple[PhysicalReplayKey, ...]:
        """Make every proof-backed idle pin reclaimable for calibration."""
        reclaimable_keys = {
            key for group in self._groups.values() for key in group.keys
        }
        pinned = tuple(
            entry
            for entry in self._entries.values()
            if entry.pinned and entry.key in reclaimable_keys
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
        # Validate the complete publication before changing any cache entry.
        # A rejected physical read-back must leave the last accepted scheduler
        # view intact; otherwise a later retry starts from a partially COLD
        # state that no worker ever published.
        observed_ids = tuple(sorted(item.identity for item in observed))
        for key, entry in self._entries.items():
            if entry.hot and key not in observed and entry.leases:
                raise ElasticGraphError(
                    "worker receipt dropped a scheduler-leased graph: "
                    f"missing={key.identity!r} "
                    f"leases={tuple(sorted(entry.leases))!r} "
                    f"observed={observed_ids!r}"
                )
        for key, (price, _pinned, _reclaimable_bytes) in observed.items():
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.price is not None
                and entry.price.reclaim_group != price.reclaim_group
            ):
                raise ElasticGraphError(
                    "graph reclaim identity changed inside one generation"
                )
        for key, entry in tuple(self._entries.items()):
            if entry.hot and key not in observed:
                self._entries[key] = replace(
                    entry,
                    state=GraphResidency.COLD,
                    deferred_free=False,
                )
        self._groups.clear()
        for key, (price, pinned, reclaimable_bytes) in observed.items():
            # An executable without a physical reclaim proof may be reusable,
            # but deleting it cannot fund another capture. Represent that
            # retained pool residency as pinned instead of creating the
            # impossible HOT_EVICTABLE-without-a-group state.
            pinned = pinned or reclaimable_bytes == 0
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
            if reclaimable_bytes:
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

    @staticmethod
    def compose_destination_capture_loan(
        *,
        current_residency_bytes: int,
        destination_capture_endpoint_bytes: int,
        retained_transition_overlap_bytes: int = 0,
        shared_resident_bytes: int = 0,
    ) -> int:
        """Compose one cold transaction without duplicating physical owners."""
        if current_residency_bytes < 0:
            raise ValueError("current residency cannot be negative")
        if destination_capture_endpoint_bytes < 0:
            raise ValueError("destination capture endpoint cannot be negative")
        if retained_transition_overlap_bytes < 0:
            raise ValueError("retained transition overlap cannot be negative")
        if (
            shared_resident_bytes < 0
            or shared_resident_bytes > current_residency_bytes
            or shared_resident_bytes > destination_capture_endpoint_bytes
        ):
            raise ValueError(
                "shared resident bytes must lie inside current and destination sets"
            )
        return current_residency_bytes + max(
            0,
            destination_capture_endpoint_bytes
            + retained_transition_overlap_bytes
            - shared_resident_bytes,
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
        destination_capture_endpoint_bytes: int | None = None,
        retained_transition_overlap_bytes: int = 0,
        shared_resident_bytes: int = 0,
        post_transition_endpoint_bytes: int | None = None,
        post_transition_shared_resident_bytes: int | None = None,
        post_transition_extra_bytes: int = 0,
        maintenance_execution: ElasticMaintenanceExecution = (
            ElasticMaintenanceExecution.COUPLED_USER
        ),
    ) -> ElasticStepPlan:
        if request_bytes < 0 or available_bytes < 0:
            raise ValueError("admission byte counts cannot be negative")
        if destination_capture_endpoint_bytes is not None:
            # Validate the complete endpoint contract before inspecting keys;
            # malformed accounting cannot become a HOT-path no-op.
            self.compose_destination_capture_loan(
                current_residency_bytes=request_bytes,
                destination_capture_endpoint_bytes=(destination_capture_endpoint_bytes),
                retained_transition_overlap_bytes=(retained_transition_overlap_bytes),
                shared_resident_bytes=shared_resident_bytes,
            )
        elif retained_transition_overlap_bytes < 0:
            raise ValueError("retained transition overlap cannot be negative")
        elif shared_resident_bytes < 0 or shared_resident_bytes > request_bytes:
            raise ValueError(
                "shared resident bytes must lie inside current request bytes"
            )
        if post_transition_extra_bytes < 0:
            raise ValueError("post-transition extra bytes cannot be negative")
        if destination_capture_endpoint_bytes is None and (
            post_transition_endpoint_bytes is not None
            or post_transition_shared_resident_bytes is not None
            or post_transition_extra_bytes
        ):
            raise ValueError(
                "post-transition accounting requires a destination capture endpoint"
            )
        if post_transition_endpoint_bytes is not None:
            if post_transition_endpoint_bytes < 0:
                raise ValueError("post-transition endpoint bytes cannot be negative")
            post_shared = (
                shared_resident_bytes
                if post_transition_shared_resident_bytes is None
                else post_transition_shared_resident_bytes
            )
            self.compose_destination_capture_loan(
                current_residency_bytes=request_bytes,
                destination_capture_endpoint_bytes=post_transition_endpoint_bytes,
                shared_resident_bytes=post_shared,
            )
        elif post_transition_shared_resident_bytes is not None:
            raise ValueError(
                "post-transition shared bytes require an explicit endpoint"
            )
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
                if destination_capture_endpoint_bytes is not None:
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

        if destination_capture_endpoint_bytes is not None:
            # The destination endpoint is measured independently of the
            # current set. Both coexist until capture settlement; subtract only
            # the intersection published by the worker receipt. This temporary
            # loan is returned to KV after settlement.
            capture_loan = (
                self.compose_destination_capture_loan(
                    current_residency_bytes=request_bytes,
                    destination_capture_endpoint_bytes=(
                        destination_capture_endpoint_bytes
                    ),
                    retained_transition_overlap_bytes=(
                        retained_transition_overlap_bytes
                    ),
                    shared_resident_bytes=shared_resident_bytes,
                )
                if misses
                else 0
            )
            # Capture and the later consumer can have different peaks.  Select
            # one victim set that proves both phases without charging the
            # consumer-only allocation during capture.
            post_transition_bytes = (
                self.compose_destination_capture_loan(
                    current_residency_bytes=request_bytes,
                    destination_capture_endpoint_bytes=(
                        destination_capture_endpoint_bytes
                        if post_transition_endpoint_bytes is None
                        else post_transition_endpoint_bytes
                    ),
                    shared_resident_bytes=(
                        shared_resident_bytes
                        if post_transition_shared_resident_bytes is None
                        else post_transition_shared_resident_bytes
                    ),
                )
                + post_transition_extra_bytes
            )
            required_bytes = max(
                request_bytes,
                capture_loan,
                post_transition_bytes,
            )
        else:
            capture_loan = (
                sum(price.capture_peak_bytes for price in prices.values())
                + retained_transition_overlap_bytes
            )
            required_bytes = request_bytes + capture_loan
        deficit = max(0, required_bytes - available_bytes)
        if not misses and deficit:
            # A USER step may acquire leases but cannot retire residency: the
            # scheduler and worker otherwise apply different physical state.
            # Reclaim must be an explicit maintenance transaction followed by
            # a newly priced USER step.
            return self._defer(
                transaction_id,
                keys,
                tuple(hits),
                (),
                protected,
                request_bytes,
                available_bytes,
                kv_transition,
                "hot_endpoint_requires_explicit_reclaim",
                capture_loan=capture_loan,
            )
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
        if destination_capture_endpoint_bytes is not None and misses and reclaimed:
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
            maintenance_execution=(
                maintenance_execution if kind == ElasticPlanKind.MAINTENANCE else None
            ),
        )

    def commit_user(self, plan: ElasticStepPlan) -> None:
        self._validate_plan(plan, ElasticPlanKind.USER)
        for key in plan.physical_keys:
            entry = self._entries.get(key)
            if entry is None or not entry.hot:
                raise ElasticGraphError("user commit requires every key HOT")
        self._snapshot_pre_mutation(plan.transaction_id, plan.physical_keys)
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

    def plan_pressure_reclaim_all(
        self,
        transaction_id: str,
        *,
        request_bytes: int,
        available_bytes: int,
        protected_keys: Iterable[PhysicalReplayKey] = (),
    ) -> ElasticStepPlan:
        """Plan all-or-nothing teardown at a physical-quiescent boundary."""
        protected = tuple(dict.fromkeys(protected_keys))
        for key in protected:
            self._require_generation(key)
        protected_set = set(protected)
        candidates = tuple(
            sorted(
                (
                    entry
                    for key, entry in self._entries.items()
                    if entry.hot and key not in protected_set
                ),
                key=lambda entry: entry.key.identity,
            )
        )
        if not candidates:
            return self._defer(
                transaction_id,
                (),
                (),
                (),
                protected,
                request_bytes,
                available_bytes,
                None,
                "no_graphs_to_pressure_reclaim",
            )
        if any(entry.leases or entry.deferred_free for entry in candidates):
            return self._defer(
                transaction_id,
                (),
                (),
                (),
                protected,
                request_bytes,
                available_bytes,
                None,
                "pressure_reclaim_has_active_graph_lease",
            )
        return ElasticStepPlan(
            transaction_id=transaction_id,
            generation=self.generation,
            kind=ElasticPlanKind.PRESSURE_RECLAIM,
            physical_keys=(),
            hot_hits=(),
            cold_misses=(),
            protected_keys=protected,
            victim_keys=tuple(entry.key for entry in candidates),
            # Administrative destruction is proved by the complete worker
            # receipt, not by ordinary unpinned reclaim-group accounting.
            reclaim_groups=(),
            capture_order=(),
            kv_transition=None,
            request_bytes=request_bytes,
            available_bytes=available_bytes,
            capture_loan_bytes=self.floor_bytes,
            reclaim_bytes=max(0, request_bytes - self.floor_bytes),
        )

    def begin_reclaim(self, plan: ElasticStepPlan) -> None:
        if plan.kind not in {
            ElasticPlanKind.RECLAIM,
            ElasticPlanKind.PRESSURE_RECLAIM,
        }:
            raise ElasticGraphError("unexpected elastic plan kind")
        self._validate_plan(plan, plan.kind)
        self._snapshot_pre_mutation(plan.transaction_id, plan.victim_keys)
        for key in plan.victim_keys:
            entry = self._entries.get(key)
            admissible = bool(
                entry is not None
                and (
                    entry.reclaimable
                    or (
                        plan.kind == ElasticPlanKind.PRESSURE_RECLAIM
                        and entry.hot
                        and not entry.leases
                        and not entry.deferred_free
                    )
                )
            )
            if not admissible:
                raise ElasticGraphError("reclaim victim is no longer reclaimable")
            assert entry is not None
            self._entries[key] = replace(entry, state=GraphResidency.COLD)
        self._evictions += len(plan.victim_keys)
        self._evicted_bytes += plan.reclaim_bytes
        self._trace.append((plan.transaction_id, "reclaim", plan.fingerprint))

    def begin_maintenance(self, plan: ElasticStepPlan) -> None:
        self._validate_plan(plan, ElasticPlanKind.MAINTENANCE)
        touched_keys = tuple(dict.fromkeys((*plan.victim_keys, *plan.cold_misses)))
        for key in plan.victim_keys:
            entry = self._entries.get(key)
            if entry is None or not entry.reclaimable:
                raise ElasticGraphError("maintenance victim is no longer reclaimable")
        for key in plan.cold_misses:
            entry = self._entries.get(key)
            if entry is not None and (entry.hot or entry.leases):
                raise ElasticGraphError("maintenance miss changed before capture")
        self._snapshot_pre_mutation(plan.transaction_id, touched_keys)
        for key in plan.victim_keys:
            entry = self._entries[key]
            self._entries[key] = replace(entry, state=GraphResidency.COLD)
        for key in plan.cold_misses:
            entry = self._entries.get(key) or ElasticGraphEntry(key=key)
            self._entries[key] = replace(entry, state=GraphResidency.CAPTURING)
        self._cold_misses += len(plan.cold_misses)
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
        self._promotions += len(plan.cold_misses)

    def fail_maintenance(self, plan: ElasticStepPlan) -> None:
        self._validate_plan(plan, ElasticPlanKind.MAINTENANCE)
        for key in plan.cold_misses:
            entry = self._entries.get(key)
            if entry is not None and entry.state == GraphResidency.CAPTURING:
                self._entries[key] = replace(
                    entry, state=GraphResidency.COLD, leases=frozenset()
                )
        self._pre_mutation_snapshots.pop(plan.transaction_id, None)

    def _snapshot_pre_mutation(
        self,
        transaction_id: str,
        keys: Iterable[PhysicalReplayKey],
    ) -> None:
        if transaction_id in self._pre_mutation_snapshots:
            raise ElasticGraphError("transaction already has a mutation snapshot")
        self._pre_mutation_snapshots[transaction_id] = _AdmissionMutationSnapshot(
            entries={key: self._entries.get(key) for key in keys},
            epoch=self._epoch,
            hot_hits=self._hot_hits,
            cold_misses=self._cold_misses,
            promotions=self._promotions,
            evictions=self._evictions,
            evicted_bytes=self._evicted_bytes,
            trace=tuple(self._trace),
        )

    def rollback_pre_mutation(self, plan: ElasticStepPlan) -> None:
        """Restore scheduler policy when workers rejected before mutation."""
        if plan.kind not in {
            ElasticPlanKind.USER,
            ElasticPlanKind.MAINTENANCE,
            ElasticPlanKind.RECLAIM,
            ElasticPlanKind.PRESSURE_RECLAIM,
        }:
            raise ElasticGraphError("execution kind cannot roll back pre-mutation")
        snapshot = self._pre_mutation_snapshots.pop(plan.transaction_id, None)
        if snapshot is None:
            raise ElasticGraphError("execution rollback lost its state snapshot")
        for key, entry in snapshot.entries.items():
            if entry is None:
                self._entries.pop(key, None)
            else:
                self._entries[key] = entry
        self._epoch = snapshot.epoch
        self._hot_hits = snapshot.hot_hits
        self._cold_misses = snapshot.cold_misses
        self._promotions = snapshot.promotions
        self._evictions = snapshot.evictions
        self._evicted_bytes = snapshot.evicted_bytes
        self._trace = deque(snapshot.trace, maxlen=128)

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

    def defer_admission(
        self,
        transaction_id: str,
        *,
        request_bytes: int,
        available_bytes: int,
        reason: str,
    ) -> ElasticStepPlan:
        """Publish a typed no-mutation scheduler admission deferral."""
        plan = self._defer(
            transaction_id,
            (),
            (),
            (),
            (),
            request_bytes,
            available_bytes,
            None,
            reason,
        )
        self.observe_defer(plan)
        return plan

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
    ) -> tuple[tuple[PhysicalReplayKey, ...], tuple[str, ...], int] | None:
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
