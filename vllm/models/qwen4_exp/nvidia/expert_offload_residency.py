# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fenced native-bank participant in the existing elastic VMM transaction."""

import hashlib
import json

import torch

from vllm.v1.core.elastic_expert import ElasticExpertGrant


class NativeExpertResidency:
    def __init__(self, provider, controller, budget):
        self.provider, self.bank = provider, provider.bank
        self.controller, self.budget = controller, budget
        self.grant = budget.grant(0)
        self.previous = self.grant
        self.last_sequence = 0
        self.resizes = self.rollbacks = 0
        if (
            controller.auxiliary_owner is not None
            or self.bank.source.geometry != budget.geometry
            or (self.bank.source.layers, self.bank.source.experts)
            != (budget.layers, budget.experts)
            or self.bank.tables.pool_rows != 0
            or sum(self.bank.targets(0).values())
            != budget.rank_mapped_bytes(self.bank.source.rank, 0)
            or self.bank.max_rows != budget.max_hot_rows + budget.staging
            or self.bank.quantum != budget.quantum
        ):
            raise ValueError("native initial residency differs from profiling budget")
        for key, backing in self.bank.backings.items():
            name = "native-" + key
            if name in controller.backings:
                raise ValueError("duplicate elastic native backing")
            controller.backings[name] = backing
            controller.auxiliary_targets[name] = backing.info.committed
        controller.auxiliary_owner = self
        self.bank.elastic_controller = controller

    @property
    def pending(self):
        return self.bank.tables.pool_rows != self.grant.hot_rows

    def _vote(self, phase, payload, error=None):
        coordinator = self.provider.coordinator
        try:
            digest = hashlib.sha256(
                json.dumps(
                    [phase, coordinator.identity, payload], sort_keys=True, default=vars
                ).encode()
            ).digest()
        except Exception as exc:
            digest, error = bytes(32), error or exc
        values = [int(error is None), self.bank.generation] + list(digest)
        values += [-1] * (coordinator.send.numel() - len(values))
        coordinator.send.copy_(torch.tensor(values, dtype=torch.int64))
        torch.distributed.all_gather_into_tensor(
            coordinator.recv, coordinator.send, group=coordinator.group.device_group
        )
        peers = coordinator.recv.view(coordinator.ranks, -1)
        if not bool(((peers == peers[:1]).all() & (peers[:, 0] == 1).all()).item()):
            self.bank.state = "POISONED"
            raise RuntimeError(
                f"native residency {phase} rejected across ranks"
            ) from error

    def admit(self, output):
        grant = output.elastic_expert_grant
        error = None
        sequence = 0
        try:
            transaction = output.elastic_transaction_id
            if not isinstance(transaction, str) or not transaction.startswith(
                "elastic-"
            ):
                raise ValueError("missing expert transaction identity")
            sequence = int(transaction.removeprefix("elastic-"))
            if sequence <= self.last_sequence or self.pending:
                raise RuntimeError("stale or unconsumed native expert grant")
            if type(grant) is not ElasticExpertGrant or grant != self.budget.grant(
                grant.hot_rows
            ):
                raise ValueError("native grant differs from exact mapped-byte curve")
            if self.provider.active or self.bank.leases or self.bank.state != "READY":
                raise RuntimeError("native bank is not quiescent")
            plan = output.elastic_step_plan
            if plan is not None and (
                plan.expert_grant != grant
                or plan.fingerprint != output.elastic_plan_fingerprint
            ):
                raise ValueError("native grant differs from immutable step plan")
        except Exception as exc:
            error = exc
        self._vote(
            "admission",
            [
                output.elastic_transaction_id,
                output.elastic_plan_fingerprint,
                grant,
                output.elastic_kv_transition,
                output.elastic_external_memory_bytes,
                self.bank.tables.pool_rows,
            ],
            error,
        )
        self.previous = self.grant
        self.grant = grant
        self.last_sequence = sequence
        self.controller.auxiliary_targets.update(
            {"native-" + k: v for k, v in self.bank.targets(grant.hot_rows).items()}
        )

    def prepare(self):
        error = None
        try:
            self.provider.quiesce()
            self.bank.prepare_resize(self.grant.hot_rows)
        except Exception as exc:
            error = exc
        self._vote("prepare", [self.previous, self.grant], error)

    def finish(self, *, success):
        target = self.grant if success else self.previous
        error = None
        try:
            self.bank.finish_resize(target.hot_rows, invalidate=not success)
        except Exception as exc:
            error = exc
        self._vote("publication", [target, success], error)
        if not success:
            self.provider.coordinator.recover()
            self.rollbacks += 1
        else:
            self.resizes += 1
        self.grant = target
        self.controller.auxiliary_targets.update(
            {"native-" + k: v for k, v in self.bank.targets(target.hot_rows).items()}
        )
