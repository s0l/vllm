# SPDX-License-Identifier: Apache-2.0
"""Fail-closed registry contracts for captured K3 target graph plans."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.ubatch_utils import UBatchSlice


class K3ElasticGraphPlanError(ValueError):
    pass


@dataclass(frozen=True)
class K3ElasticWorkloadKey:
    runtime_identity: str
    x: int
    rows_per_request: int
    mode: str
    context_bucket: str
    kv_pressure_bucket: str
    prefill_bucket: str
    graph_family: str


@dataclass(frozen=True)
class K3ElasticGraphPlan:
    name: str
    runtime_identity: str
    partition_x: tuple[int, ...]
    rows_per_request: frozenset[int]
    modes: frozenset[str]
    context_buckets: frozenset[str]
    kv_pressure_buckets: frozenset[str]
    prefill_buckets: frozenset[str]
    graph_families: frozenset[str]
    proof_sha256: str
    implementation_sha256: str
    state_bytes: int
    priority: int = 100

    @property
    def total_x(self) -> int:
        return sum(self.partition_x)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> K3ElasticGraphPlan:
        required = {
            "name",
            "runtime_identity",
            "partition_x",
            "rows_per_request",
            "modes",
            "context_buckets",
            "kv_pressure_buckets",
            "prefill_buckets",
            "graph_families",
            "proof_sha256",
            "implementation_sha256",
            "state_bytes",
        }
        missing = sorted(required - value.keys())
        unknown = sorted(value.keys() - required - {"priority"})
        if missing or unknown:
            raise K3ElasticGraphPlanError(
                f"invalid plan fields: missing={missing} unknown={unknown}"
            )
        plan = cls(
            name=str(value["name"]),
            runtime_identity=str(value["runtime_identity"]),
            partition_x=tuple(int(item) for item in value["partition_x"]),
            rows_per_request=frozenset(
                int(item) for item in value["rows_per_request"]
            ),
            modes=frozenset(str(item) for item in value["modes"]),
            context_buckets=frozenset(
                str(item) for item in value["context_buckets"]
            ),
            kv_pressure_buckets=frozenset(
                str(item) for item in value["kv_pressure_buckets"]
            ),
            prefill_buckets=frozenset(
                str(item) for item in value["prefill_buckets"]
            ),
            graph_families=frozenset(
                str(item) for item in value["graph_families"]
            ),
            proof_sha256=str(value["proof_sha256"]),
            implementation_sha256=str(value["implementation_sha256"]),
            state_bytes=int(value["state_bytes"]),
            priority=int(value.get("priority", 100)),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if not self.name or not self.runtime_identity:
            raise K3ElasticGraphPlanError("plan identity is incomplete")
        if (
            len(self.partition_x) < 2
            or any(item <= 0 for item in self.partition_x)
            or tuple(sorted(self.partition_x, reverse=True)) != self.partition_x
        ):
            raise K3ElasticGraphPlanError(
                "partition must contain at least two positive ordered waves"
            )
        envelopes = (
            self.rows_per_request,
            self.modes,
            self.context_buckets,
            self.kv_pressure_buckets,
            self.prefill_buckets,
            self.graph_families,
        )
        if any(not values for values in envelopes):
            raise K3ElasticGraphPlanError("plan execution envelope is incomplete")
        if any(item <= 0 for item in self.rows_per_request):
            raise K3ElasticGraphPlanError("rows_per_request must be positive")
        if len(self.proof_sha256) != 64 or len(self.implementation_sha256) != 64:
            raise K3ElasticGraphPlanError("plan SHA-256 is invalid")
        if self.state_bytes < 0:
            raise K3ElasticGraphPlanError("plan state_bytes must be nonnegative")

    def matches(self, key: K3ElasticWorkloadKey, state_bytes: int) -> bool:
        return (
            self.runtime_identity == key.runtime_identity
            and self.total_x == key.x
            and key.rows_per_request in self.rows_per_request
            and key.mode in self.modes
            and key.context_bucket in self.context_buckets
            and key.kv_pressure_bucket in self.kv_pressure_buckets
            and key.prefill_bucket in self.prefill_buckets
            and key.graph_family in self.graph_families
            and self.state_bytes <= state_bytes
        )

    def request_slices(self) -> tuple[slice, ...]:
        start = 0
        result = []
        for wave_x in self.partition_x:
            result.append(slice(start, start + wave_x))
            start += wave_x
        return tuple(result)

    def token_slices(self, rows_per_request: int) -> tuple[slice, ...]:
        if rows_per_request not in self.rows_per_request:
            raise K3ElasticGraphPlanError("unsupported rows_per_request")
        return tuple(
            slice(item.start * rows_per_request, item.stop * rows_per_request)
            for item in self.request_slices()
        )


class K3ElasticGraphRegistry:
    def __init__(self, plans: list[K3ElasticGraphPlan]):
        if not plans:
            raise K3ElasticGraphPlanError("registry cannot be empty")
        names = [plan.name for plan in plans]
        if len(names) != len(set(names)):
            raise K3ElasticGraphPlanError("plan names must be unique")
        for plan in plans:
            plan.validate()
        self._plans = tuple(plans)

    @classmethod
    def from_config(cls, value: list[dict[str, Any]]) -> K3ElasticGraphRegistry:
        return cls([K3ElasticGraphPlan.from_dict(item) for item in value])

    @classmethod
    def from_json(cls, value: str) -> K3ElasticGraphRegistry:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise K3ElasticGraphPlanError("graph registry JSON is invalid") from exc
        if not isinstance(decoded, list):
            raise K3ElasticGraphPlanError("graph registry must be a JSON list")
        if not all(isinstance(item, dict) for item in decoded):
            raise K3ElasticGraphPlanError("every graph plan must be an object")
        return cls.from_config(decoded)

    @property
    def max_waves(self) -> int:
        return max(len(plan.partition_x) for plan in self._plans)

    @property
    def plans(self) -> tuple[K3ElasticGraphPlan, ...]:
        return self._plans

    def candidates(
        self, key: K3ElasticWorkloadKey, state_bytes: int
    ) -> tuple[K3ElasticGraphPlan, ...]:
        if key.x <= 0 or key.rows_per_request <= 0 or state_bytes < 0:
            raise K3ElasticGraphPlanError("invalid workload/state geometry")
        matches = [plan for plan in self._plans if plan.matches(key, state_bytes)]
        return tuple(sorted(matches, key=lambda plan: (plan.priority, plan.name)))


def context_bucket(max_context_tokens: int) -> str:
    if max_context_tokens < 0:
        raise K3ElasticGraphPlanError("context length must be nonnegative")
    if max_context_tokens <= 65536:
        return "0-64k"
    if max_context_tokens <= 131072:
        return "64k-128k"
    if max_context_tokens <= 262144:
        return "128k-256k"
    return ">256k"


def kv_pressure_bucket(kv_pressure: float) -> str:
    if not 0.0 <= kv_pressure <= 1.0:
        raise K3ElasticGraphPlanError("KV pressure must be within [0, 1]")
    if kv_pressure < 0.5:
        return "0-50pct"
    if kv_pressure < 0.8:
        return "50-80pct"
    if kv_pressure < 0.95:
        return "80-95pct"
    return "95-100pct"


def prefill_bucket(prefill_tokens: int) -> str:
    if prefill_tokens < 0:
        raise K3ElasticGraphPlanError("prefill tokens must be nonnegative")
    if prefill_tokens == 0:
        return "none"
    if prefill_tokens <= 4096:
        return "1-4k"
    if prefill_tokens <= 32768:
        return "4k-32k"
    if prefill_tokens <= 131072:
        return "32k-128k"
    return ">128k"


def make_decode_slices(
    plan: K3ElasticGraphPlan,
    input_batch: InputBatch,
    batch_desc: BatchExecutionDescriptor,
    decode_query_len: int,
) -> tuple[UBatchSlice, ...] | None:
    if decode_query_len not in plan.rows_per_request:
        return None
    if input_batch.num_reqs != plan.total_x:
        return None
    if input_batch.num_tokens_after_padding != input_batch.num_tokens:
        return None
    if batch_desc.num_tokens != input_batch.num_tokens:
        return None
    if batch_desc.num_reqs is not None and batch_desc.num_reqs != input_batch.num_reqs:
        return None
    if input_batch.num_tokens != input_batch.num_reqs * decode_query_len:
        return None
    if np.any(input_batch.num_scheduled_tokens != decode_query_len):
        return None
    if np.any(input_batch.is_prefilling_np):
        return None
    draft_counts = input_batch.num_draft_tokens_per_req
    if draft_counts is None or np.any(draft_counts != decode_query_len - 1):
        return None
    return tuple(
        UBatchSlice(request_slice, token_slice)
        for request_slice, token_slice in zip(
            plan.request_slices(),
            plan.token_slices(decode_query_len),
            strict=True,
        )
    )


def slice_input_batch(
    input_batch: InputBatch,
    ubatch_slice: UBatchSlice,
) -> InputBatch:
    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice
    request_start = request_slice.start
    request_stop = request_slice.stop
    token_start = token_slice.start
    token_stop = token_slice.stop
    if None in (request_start, request_stop, token_start, token_stop):
        raise K3ElasticGraphPlanError("wave slices must be bounded")
    assert request_start is not None and request_stop is not None
    assert token_start is not None and token_stop is not None
    num_reqs = request_stop - request_start
    num_tokens = token_stop - token_start
    if num_reqs <= 0 or num_tokens <= 0:
        raise K3ElasticGraphPlanError("wave slices must be nonempty")

    query_start_loc_np = (
        input_batch.query_start_loc_np[request_start : request_stop + 1]
        - token_start
    ).copy()
    query_start_loc = (
        input_batch.query_start_loc[request_start : request_stop + 1]
        - token_start
    )
    logits_start = int(input_batch.cu_num_logits_np[request_start])
    logits_stop = int(input_batch.cu_num_logits_np[request_stop])
    cu_num_logits_np = (
        input_batch.cu_num_logits_np[request_start : request_stop + 1]
        - logits_start
    ).copy()
    cu_num_logits = (
        input_batch.cu_num_logits[request_start : request_stop + 1]
        - logits_start
    )
    draft_counts = input_batch.num_draft_tokens_per_req
    sliced_drafts = None if draft_counts is None else draft_counts[request_slice]

    return replace(
        input_batch,
        req_ids=input_batch.req_ids[request_slice],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=input_batch.idx_mapping[request_slice],
        idx_mapping_np=input_batch.idx_mapping_np[request_slice],
        expanded_idx_mapping=input_batch.expanded_idx_mapping[
            logits_start:logits_stop
        ],
        expanded_local_pos=input_batch.expanded_local_pos[logits_start:logits_stop],
        num_scheduled_tokens=input_batch.num_scheduled_tokens[request_slice],
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=(0 if sliced_drafts is None else int(sliced_drafts.sum())),
        num_draft_tokens_per_req=sliced_drafts,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        marlin_request_layout_cpu=_slice_marlin_request_layout(
            input_batch.marlin_request_layout_cpu,
            num_reqs,
            query_start_loc_np,
        ),
        seq_lens=input_batch.seq_lens[request_slice],
        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[request_slice],
        dcp_local_seq_lens=(
            None
            if input_batch.dcp_local_seq_lens is None
            else input_batch.dcp_local_seq_lens[request_slice]
        ),
        num_computed_tokens_np=input_batch.num_computed_tokens_np[request_slice],
        prefill_len_np=input_batch.prefill_len_np[request_slice],
        num_computed_prefill_tokens_np=(
            input_batch.num_computed_prefill_tokens_np[request_slice]
        ),
        is_prefilling_np=input_batch.is_prefilling_np[request_slice],
        max_seq_len_np=(
            None
            if input_batch.max_seq_len_np is None
            else input_batch.max_seq_len_np[request_slice]
        ),
        input_ids=input_batch.input_ids[token_slice],
        positions=input_batch.positions[token_slice],
        is_padding=input_batch.is_padding[token_slice],
        logits_indices=(
            input_batch.logits_indices[logits_start:logits_stop] - token_start
        ),
        cu_num_logits=cu_num_logits,
        cu_num_logits_np=cu_num_logits_np,
        prompt_lens=(
            None
            if input_batch.prompt_lens is None
            else input_batch.prompt_lens[request_slice]
        ),
    )


def _slice_marlin_request_layout(
    source: Any,
    num_reqs: int,
    query_start_loc_np: np.ndarray,
):
    layout = source.new_zeros(num_reqs + 3)
    layout[0] = num_reqs
    layout[1] = num_reqs
    layout[2 : num_reqs + 3].copy_(
        source.new_tensor(query_start_loc_np)
    )
    return layout
