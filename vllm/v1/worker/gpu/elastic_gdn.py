# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 runner adapters for elastic KV and separate-pool GDN checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.worker.mamba_utils import GDNPrefixCheckpointStore

if TYPE_CHECKING:
    from vllm.device_allocator.elastic_cumem import ElasticCuMemBacking

logger = init_logger(__name__)


class ElasticKVController:
    """Own and resize the stable-VA elastic KV backings used by MRV2."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.backings: dict[str, ElasticCuMemBacking] = {}
        self.geometry: dict[str, int] = {}
        # Non-KV owners publish exact mapped byte targets at a fenced boundary.
        # They must never absorb unused KV budget merely by lexical key order.
        self.auxiliary_targets: dict[str, int] = {}
        self.auxiliary_owner: Any = None
        self._physical_budget_bytes: int | None = None
        self._mapping_quantum: int | None = None
        self._logical_transition: tuple[int, int] | None = None
        self._external_memory_bytes = 0
        self._last_logged_external_floor_bytes = 0

    @property
    def external_memory_bytes(self) -> int:
        """Return the worker-applied, mapping-quantized external loan."""
        return self._external_memory_bytes

    def physical_memory_receipt(self) -> dict[str, Any]:
        """Report mapped owners and unrequested KV without querying the GPU."""
        rows = []
        for key, owner in sorted(self.backings.items()):
            info = owner.info
            required = None
            if key in self.auxiliary_targets:
                required = self.auxiliary_targets[key]
            elif self._logical_transition is not None:
                blocks = self._logical_transition[key == "elastic-gdn"]
                size = blocks * self.geometry[key]
                required = (size + info.quantum - 1) // info.quantum * info.quantum
            rows.append(
                dict(
                    owner=key,
                    mapped_bytes=info.committed,
                    required_bytes=required,
                    retained_tail_bytes=None
                    if required is None
                    else max(info.committed - required, 0),
                )
            )
        return dict(
            physical_budget_bytes=self._physical_budget_bytes,
            external_memory_bytes=self._external_memory_bytes,
            logical_transition=self._logical_transition,
            backings=rows,
        )

    def configure_physical_budget(
        self,
        budget_bytes: int,
        quantum: int,
        initial_transition: tuple[int, int],
    ) -> None:
        """Pin the largest representable planner budget across transitions."""
        if budget_bytes <= 0 or quantum <= 0:
            raise ValueError("elastic physical budget and quantum must be positive")
        physical_budget = budget_bytes // quantum * quantum
        committed = sum(owner.info.committed for owner in self.backings.values())
        reserved = sum(owner.info.reserved for owner in self.backings.values())
        if not committed <= physical_budget <= reserved:
            raise ValueError(
                "elastic physical budget is outside backing bounds: "
                f"committed={committed}, budget={physical_budget}, "
                f"reserved={reserved}"
            )
        self._physical_budget_bytes = physical_budget
        self._mapping_quantum = quantum
        self._logical_transition = initial_transition

    def _preserve_physical_budget(
        self,
        targets: dict[str, int],
        external_memory_bytes: int,
    ) -> dict[str, int]:
        if self._physical_budget_bytes is None:
            self._physical_budget_bytes = sum(
                owner.info.committed for owner in self.backings.values()
            )
        if external_memory_bytes < 0:
            raise ValueError("elastic external memory cannot be negative")
        quantum = self._mapping_quantum
        if quantum is None:
            quantum = next(iter(self.backings.values())).info.quantum
        external_mapped = (external_memory_bytes + quantum - 1) // quantum * quantum
        if external_mapped > self._physical_budget_bytes:
            raise RuntimeError(
                "elastic external target exceeds the physical budget: "
                f"external={external_memory_bytes} mapped={external_mapped} "
                f"budget={self._physical_budget_bytes} quantum={quantum}"
            )
        retained_budget = self._physical_budget_bytes - external_mapped
        normalized = dict(targets)
        slack = retained_budget - sum(normalized.values())
        if slack < 0:
            raise RuntimeError(
                "elastic KV and external targets exceed the physical budget by "
                f"{-slack} bytes: external={external_memory_bytes} "
                f"mapped={external_mapped} budget={self._physical_budget_bytes} "
                f"target_sum={sum(normalized.values())} targets={normalized!r}"
            )
        for key in sorted(normalized):
            if key in self.auxiliary_targets:
                continue
            if slack == 0:
                break
            info = self.backings[key].info
            available = info.reserved - normalized[key]
            retained = min(slack, available)
            normalized[key] += retained
            slack -= retained
        if slack:
            raise RuntimeError(
                f"elastic KV stable VA cannot retain {slack} budget bytes"
            )
        return normalized

    def _apply_targets(self, targets: dict[str, int], fence: torch.cuda.Event) -> None:
        current = {key: owner.info.committed for key, owner in self.backings.items()}
        donors: list[list[Any]] = [
            [key, current[key] - targets[key]] for key in sorted(current)
        ]
        donors = [entry for entry in donors if entry[1] > 0]
        receivers: list[list[Any]] = [
            [key, targets[key] - current[key]] for key in sorted(current)
        ]
        receivers = [entry for entry in receivers if entry[1] > 0]

        donor_idx = receiver_idx = 0
        while donor_idx < len(donors) and receiver_idx < len(receivers):
            donor_key, available = donors[donor_idx]
            receiver_key, required = receivers[receiver_idx]
            moved = min(available, required)
            self.backings[donor_key].transfer_to(
                self.backings[receiver_key], moved, fence
            )
            donors[donor_idx][1] -= moved
            receivers[receiver_idx][1] -= moved
            if donors[donor_idx][1] == 0:
                donor_idx += 1
            if receivers[receiver_idx][1] == 0:
                receiver_idx += 1

        # Only a real decrease/increase in total committed bytes reaches these
        # calls. Transfers above handle a constant-budget rebalance without
        # releasing and recreating physical handles.
        for key, unused in donors[donor_idx:]:
            if unused:
                self.backings[key].resize(targets[key], fence)
        for key, missing in receivers[receiver_idx:]:
            if missing:
                self.backings[key].resize(targets[key], fence)

    def _target_sizes(
        self,
        transition: tuple[int, int],
        external_memory_bytes: int,
    ) -> dict[str, int]:
        """Derive the exact backing layout for one logical transition."""
        attention_blocks, gdn_blocks = transition
        if self.auxiliary_targets.keys() - self.backings.keys():
            raise ValueError("auxiliary elastic target without a backing")
        targets: dict[str, int] = {}
        for key, owner in self.backings.items():
            if key in self.auxiliary_targets:
                size = self.auxiliary_targets[key]
                info = owner.info
                if (
                    key in self.geometry
                    or type(size) is not int
                    or size < 0
                    or size > info.reserved
                    or size % info.quantum
                ):
                    raise ValueError(f"invalid auxiliary elastic target: {key}")
                targets[key] = size
                continue
            block_bytes = self.geometry[key]
            blocks = gdn_blocks if key == "elastic-gdn" else attention_blocks
            info = owner.info
            if (
                (
                    self._mapping_quantum is not None
                    and info.quantum != self._mapping_quantum
                )
                or info.committed < 0
                or info.committed > info.reserved
                or info.reserved % info.quantum
            ):
                raise RuntimeError(
                    f"invalid elastic backing geometry for {key}: {info!r}"
                )
            targets[key] = (
                (blocks * block_bytes + info.quantum - 1) // info.quantum * info.quantum
            )
            if targets[key] > info.reserved:
                raise RuntimeError(
                    f"elastic target exceeds reserved backing for {key}: "
                    f"blocks={blocks} block_bytes={block_bytes} target={targets[key]} "
                    f"reserved={info.reserved}"
                )
        return self._preserve_physical_budget(targets, external_memory_bytes)

    def validate_equal_scheduler_step(
        self,
        transition: tuple[int, int] | None,
        external_memory_bytes: int,
    ) -> bool:
        """Validate a common-plan no-op before the existing all-rank vote.

        Returning ``True`` authorizes the caller to skip ``apply()`` only
        after that vote succeeds. A local backing drift raises here and is
        therefore propagated to every rank by the plan-consensus boundary.
        """
        if (
            transition is not None
            or (self.auxiliary_owner is not None and self.auxiliary_owner.pending)
            or not self.backings
            or self._physical_budget_bytes is None
            or self._mapping_quantum is None
        ):
            return False
        target_transition = self._logical_transition
        if target_transition is None:
            return False
        if (
            external_memory_bytes < 0
            or external_memory_bytes != self._external_memory_bytes
        ):
            return False
        quantum = self._mapping_quantum
        requested = (external_memory_bytes + quantum - 1) // quantum * quantum
        current = (self._external_memory_bytes + quantum - 1) // quantum * quantum
        if requested != current:
            return False
        expected = self._target_sizes(target_transition, requested)
        observed = {key: owner.info.committed for key, owner in self.backings.items()}
        if observed != expected:
            raise RuntimeError(
                "equal elastic scheduler step has divergent backing geometry: "
                f"expected={expected!r} observed={observed!r}"
            )
        return True

    def apply(
        self,
        transition: tuple[int, int] | None,
        external_memory_bytes: int = 0,
    ) -> None:
        owner = self.auxiliary_owner
        pending = owner is not None and owner.pending
        if pending:
            owner.prepare()
        try:
            self._apply(transition, external_memory_bytes)
        except Exception:
            if pending:
                owner.finish(success=False)
            raise
        else:
            if pending:
                owner.finish(success=True)

    def _apply(
        self,
        transition: tuple[int, int] | None,
        external_memory_bytes: int = 0,
    ) -> None:
        target_transition = transition or self._logical_transition
        if target_transition is None:
            return
        if not self.backings:
            raise RuntimeError("elastic KV transition without elastic backings")

        attention_ids = sorted(
            key for key in self.backings if key.startswith("elastic-attention-")
        )
        old_sizes = {key: owner.info.committed for key, owner in self.backings.items()}
        targets: dict[str, int] = {}
        preflight_error: Exception | None = None
        try:
            targets = self._target_sizes(target_transition, external_memory_bytes)
        except Exception as exc:
            preflight_error = exc

        preflight_vote = torch.tensor(
            0 if preflight_error else 1,
            dtype=torch.int32,
            device=self.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                preflight_vote,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
        if not bool(preflight_vote.item()):
            raise RuntimeError(
                "elastic KV transition rejected by rank-synchronous preflight"
            ) from preflight_error

        any_change = torch.tensor(
            int(targets != old_sizes),
            dtype=torch.int32,
            device=self.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                any_change,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
        if not bool(any_change.item()):
            self._logical_transition = target_transition
            self._external_memory_bytes = external_memory_bytes
            return

        fence = torch.cuda.Event()
        fence.record(torch.cuda.current_stream())
        apply_error: Exception | None = None
        try:
            self._apply_targets(targets, fence)
        except Exception as exc:
            apply_error = exc

        apply_vote = torch.tensor(
            0 if apply_error else 1,
            dtype=torch.int32,
            device=self.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                apply_vote,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().device_group,
            )
        if not bool(apply_vote.item()):
            rollback = torch.cuda.Event()
            rollback.record(torch.cuda.current_stream())
            rollback_error: Exception | None = None
            try:
                self._apply_targets(old_sizes, rollback)
            except Exception as exc:
                rollback_error = exc
            rollback_vote = torch.tensor(
                0 if rollback_error else 1,
                dtype=torch.int32,
                device=self.device,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    rollback_vote,
                    op=torch.distributed.ReduceOp.MIN,
                    group=get_tp_group().device_group,
                )
            if not bool(rollback_vote.item()):
                raise RuntimeError(
                    "elastic KV transition rollback failed on at least one rank"
                ) from rollback_error
            raise RuntimeError(
                "elastic KV transition aborted on at least one rank"
            ) from apply_error

        logger.debug(
            "Elastic KV transition committed: attention=%.3f GiB GDN=%.3f GiB "
            "external=%.3f GiB",
            sum(self.backings[key].info.committed for key in attention_ids) / 1024**3,
            self.backings["elastic-gdn"].info.committed / 1024**3,
            external_memory_bytes / 1024**3,
        )
        self._logical_transition = target_transition
        self._external_memory_bytes = external_memory_bytes

    def reconcile_external_memory(self, external_memory_bytes: int) -> int:
        """Return as much external memory as can be mapped back into KV.

        CUDA may keep graph-pool or allocator mappings alive after the graph
        ledger has released them.  Query rank-safe driver-free physical memory
        before changing any VMM mapping and return only whole quanta that are
        already available.  This is a retryable observed floor, not a reserve
        and not the old mutate/fail/rollback binary search.
        """
        if not self.backings:
            self.apply(None, external_memory_bytes)
            return external_memory_bytes

        quantum = self._mapping_quantum
        if quantum is None:
            quantum = next(iter(self.backings.values())).info.quantum

        requested = (external_memory_bytes + quantum - 1) // quantum * quantum
        current = (self._external_memory_bytes + quantum - 1) // quantum * quantum

        local_free, local_total = torch.accelerator.get_memory_info(self.device)
        if not 0 <= local_free <= local_total:
            raise RuntimeError(
                "CUDA free-memory oracle returned an invalid local value: "
                f"free={local_free} total={local_total}"
            )
        # This decision must be rank synchronous. Rank-specific allocator floors
        # can put one rank on the grow/no-op side and another on the return side.
        # A conditional device collective here previously collided with the next
        # preflight collective at X32; making the oracle CPU-side merely exposed
        # the same bug as a hang. Every rank now enters the same two control-plane
        # collectives and then applies one common worst-rank floor.
        control = torch.tensor((requested, -requested, local_free), dtype=torch.int64)
        cpu_group = None
        if torch.distributed.is_initialized():
            cpu_group = get_tp_group().cpu_group
            torch.distributed.all_reduce(
                control,
                op=torch.distributed.ReduceOp.MIN,
                group=cpu_group,
            )
        requested_min = int(control[0].item())
        requested_max = -int(control[1].item())
        rank_safe_free = int(control[2].item())
        if requested_min != requested_max or requested != requested_min:
            raise RuntimeError(
                "elastic external target differs across ranks: "
                f"local={requested} min={requested_min} max={requested_max}"
            )
        if not 0 <= rank_safe_free <= local_total:
            raise RuntimeError(
                "rank-safe CUDA free-memory oracle returned an invalid value: "
                f"free={rank_safe_free} local_total={local_total}"
            )
        returnable = min(
            max(current - requested, 0), rank_safe_free // quantum * quantum
        )
        local_effective = max(requested, current - returnable)
        effective_tensor = torch.tensor(local_effective, dtype=torch.int64)
        if cpu_group is not None:
            torch.distributed.all_reduce(
                effective_tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=cpu_group,
            )
        effective = int(effective_tensor.item())
        physical_budget_bytes = self._physical_budget_bytes
        if physical_budget_bytes is None:
            raise RuntimeError("elastic physical memory budget is unavailable")
        if not requested <= effective <= physical_budget_bytes:
            raise RuntimeError(
                "rank-safe elastic external floor is outside its budget: "
                f"requested={requested} effective={effective} "
                f"budget={physical_budget_bytes}"
            )
        # All distributed ranks must enter apply(), including ranks for which
        # the common floor is locally unchanged, because apply has its own
        # rank-synchronous preflight and publication votes.
        if (
            cpu_group is not None
            or effective != current
            or (self.auxiliary_owner is not None and self.auxiliary_owner.pending)
        ):
            self.apply(None, effective)

        retained = effective - requested
        if retained != getattr(self, "_last_logged_external_floor_bytes", 0):
            logger.warning(
                "Elastic KV measured a temporary physical external-memory floor: "
                "requested=%.3f GiB effective=%.3f GiB retained=%.3f GiB "
                "rank_safe_driver_free=%.3f GiB; "
                "future steps will retry the return",
                requested / 1024**3,
                effective / 1024**3,
                retained / 1024**3,
                rank_safe_free / 1024**3,
            )
            self._last_logged_external_floor_bytes = retained
        elif retained == 0:
            self._last_logged_external_floor_bytes = 0
        return effective

    def apply_scheduler_step(
        self,
        transition: tuple[int, int] | None,
        external_memory_bytes: int,
    ) -> int:
        """Apply a worker step without mistaking a physical floor for OOM.

        A missing transition means attention/GDN geometry is unchanged and
        only the step-scoped external loan is moving. That return must use the
        measured floor path: allocator or CUDA Graph teardown may make a small
        part temporarily unavailable to KV even though the scheduler's logical
        request is correct. Real geometry changes remain exact and fail closed.
        """
        if transition is None:
            return self.reconcile_external_memory(external_memory_bytes)
        self.apply(transition, external_memory_bytes)
        return external_memory_bytes

    def mapped_attention_block_capacity(self) -> int | None:
        """Return the fully mapped logical attention-block prefix."""
        attention_ids = [
            key for key in self.backings if key.startswith("elastic-attention-")
        ]
        if not attention_ids:
            return None
        return min(
            self.backings[key].info.committed // self.geometry[key]
            for key in attention_ids
        )


def validate_elastic_attention_block_tables(
    block_tables: Any,
    kv_cache_config: KVCacheConfig,
    req_id_to_index: dict[str, int],
    scheduled_req_ids: Any,
    mapped_attention_blocks: int | None,
) -> None:
    """Fail before CUDA graph replay if a live table points past mapped VA."""
    if mapped_attention_blocks is None:
        return
    attention_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if isinstance(group.kv_cache_spec, AttentionSpec)
    ]
    for req_id in scheduled_req_ids:
        req_index = req_id_to_index[req_id]
        for group_id in attention_group_ids:
            count = int(block_tables.num_blocks.np[group_id, req_index])
            if count == 0:
                continue
            blocks_per_kv_block = block_tables.blocks_per_kv_block[group_id]
            mapped_kernel_blocks = mapped_attention_blocks * blocks_per_kv_block
            # The CPU mirror is authoritative for the staged block-table
            # writes and lets this diagnostic remain replay-neutral. Calling
            # .item() on the GPU mirror here inserts a rank-local device
            # synchronization immediately before CUDA Graph replay, which can
            # turn a distributed warmup into a diagnostic-induced hang.
            block_ids = block_tables.host_block_tables[group_id][req_index, :count]
            invalid = (block_ids < 0) | (block_ids >= mapped_kernel_blocks)
            if bool(invalid.any()):
                invalid_ids = block_ids[invalid].tolist()
                raise RuntimeError(
                    "elastic attention block table references unmapped VA: "
                    f"request={req_id} group={group_id} "
                    f"mapped_logical_blocks={mapped_attention_blocks} "
                    f"blocks_per_kv_block={blocks_per_kv_block} "
                    f"invalid_kernel_block_ids={invalid_ids}"
                )


class V2GDNCheckpointManager:
    """Translate MRV2 request storage into the existing exact GDN store."""

    def __init__(self) -> None:
        self.store = GDNPrefixCheckpointStore(limit=32, advertised_limit=8)
        self._block_ids: dict[str, tuple[list[int], ...]] = {}

    @staticmethod
    def _is_synthetic_request(req_id: str) -> bool:
        # MRV2 kernel warmup builds scheduler-shaped requests whose block
        # tables exercise kernels but do not represent cacheable request
        # state. They must not enter the exact-prefix checkpoint lifecycle.
        return req_id.startswith(("_warmup_", "_dummy_req_"))

    @staticmethod
    def _copy_block_ids(
        block_ids: tuple[list[int], ...],
    ) -> tuple[list[int], ...]:
        return tuple(list(group) for group in block_ids)

    def update_request_blocks(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in scheduler_output.finished_req_ids:
            self._block_ids.pop(req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            if self._is_synthetic_request(request.req_id):
                continue
            self._block_ids[request.req_id] = self._copy_block_ids(request.block_ids)

        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_block_ids in zip(cached.req_ids, cached.new_block_ids):
            if self._is_synthetic_request(req_id):
                continue
            if new_block_ids is None:
                continue
            if req_id in cached.resumed_req_ids:
                self._block_ids[req_id] = self._copy_block_ids(new_block_ids)
                continue
            current = self._block_ids.get(req_id)
            if current is None:
                raise RuntimeError(
                    f"missing MRV2 GDN block state for cached request {req_id}"
                )
            if len(current) != len(new_block_ids):
                raise RuntimeError("MRV2 GDN block group count changed")
            self._block_ids[req_id] = tuple(
                old + list(new) for old, new in zip(current, new_block_ids)
            )

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: dict[str, int],
        forward_context: dict[str, Any],
    ) -> None:
        restored = set((scheduler_output.gdn_checkpoint_restore or {}).keys())
        for req_id in scheduler_output.num_scheduled_tokens:
            if self._is_synthetic_request(req_id):
                continue
            block_ids = self._block_ids[req_id]
            if req_id not in restored and num_computed_tokens[req_id] == 0:
                self.store.zero_blocks(block_ids, kv_cache_config, forward_context)

        for req_id, key in (scheduler_output.gdn_checkpoint_restore or {}).items():
            self.store.restore_blocks(
                key, self._block_ids[req_id], kv_cache_config, forward_context
            )

    def save(
        self,
        scheduler_output: SchedulerOutput,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
    ) -> tuple[bytes, ...]:
        managed = getattr(scheduler_output, "gdn_checkpoint_plan", None) is not None
        if managed:
            self.store.begin_checkpoint_plan(scheduler_output)
        for req_id, key in (scheduler_output.gdn_checkpoint_save or {}).items():
            self.store.save_blocks(
                key, self._block_ids[req_id], kv_cache_config, forward_context
            )
        if managed:
            self.store.finish_checkpoint_plan(scheduler_output)
        return self.store.snapshot_keys()
