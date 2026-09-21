# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-free allocation profiling through the real Graph lifecycle."""

from __future__ import annotations

import time
from typing import Any

from vllm.v1.core.elastic_memory_profile import (
    allocation_envelope_proof,
    allocation_envelope_row,
    allocation_profile_shapes,
)


def _validate_native_expert_profile_contract(
    config: Any, providers: tuple[Any, ...]
) -> bool:
    additional = config.additional_config or {}
    native_experts = bool(additional.get("flashnext_native_experts"))
    if native_experts != bool(providers):
        raise RuntimeError(
            "allocation profile model path disagrees with native expert providers"
        )
    return native_experts


def replay_allocation_profile(
    worker: Any, step_key: tuple[int, int, int, int, int]
) -> dict:
    """Replay retained dummy closures under the already committed KV loan.

    No user request, expert identity or sampler result is consumed. The RPC is
    legal only after request-free MAINTENANCE, with no concurrent execution.
    """
    import torch

    with torch.inference_mode():
        return _replay_allocation_profile(worker, step_key)


def _replay_allocation_profile(
    worker: Any, step_key: tuple[int, int, int, int, int]
) -> dict:
    import torch

    from vllm.config import CUDAGraphMode
    from vllm.v1.core.elastic_graph import RuntimeGeneration, resolve_step_physical_keys

    runner = worker.model_runner
    if runner.execute_model_state is not None:
        raise RuntimeError("allocation profiling cannot interrupt a model step")
    working_set = runner._dynamic_graph_working_set()
    managers = {m.dynamic_graph_owner: m for m in working_set.managers}
    policy = worker.get_elastic_graph_execution_policy()
    from vllm.v1.core.elastic_graph import GraphExecutionPolicy

    keys = resolve_step_physical_keys(
        step_key,
        generation=RuntimeGeneration(next(iter(managers.values())).runtime_generation),
        max_num_batched_tokens=runner.max_num_tokens,
        policy=GraphExecutionPolicy.from_payload(policy),
    )
    providers = tuple(
        {
            id(provider): provider
            for provider in getattr(runner.model_state, "_native_providers", ())
        }.values()
    )
    _validate_native_expert_profile_contract(runner.vllm_config, providers)
    if any(p.active for p in providers):
        raise RuntimeError("allocation profiling requires idle providers")
    sources = tuple({id(p.bank.source): p.bank.source for p in providers}.values())
    before_reads = sum(source.misses for source in sources)
    for provider in providers:
        provider.quiesce()
    transaction = "allocation-profile-replay"
    try:
        for key in keys:
            managers[key.logical.owner].acquire_physical_key_lease(key, transaction)
        torch.accelerator.synchronize()
        baseline = torch.accelerator.memory_reserved()
        baseline_free = torch.accelerator.get_memory_info()[0]
        if getattr(runner, "_elastic_step_measurement_active", False):
            raise RuntimeError("allocation profile overlaps unfinished measurement")
        torch.accelerator.reset_peak_memory_stats()
        # Runtime waves use every power-of-two lane bucket. Materialize this
        # finite inner family once, without choosing or reading real experts.
        # Merely profiling the largest kernel would omit retained small graphs.
        import numpy as np

        for provider in providers:
            if provider.e8_path is not None:
                continue
            if all(
                (1 << exponent) in provider.kernels
                for exponent in range(provider.max_lanes.bit_length())
            ):
                continue
            ids = np.full((1, provider.topk), -1, dtype=np.int32)
            provider.coordinator.admit_routes(
                0, ids, np.zeros(ids.shape, dtype=np.float32), dummy=True
            )
            ticket = provider.coordinator.stage(0, provider.expert_ids[:0])
            lease = provider.bank.acquire(ticket)
            try:
                bucket = 1
                while bucket <= provider.max_lanes:
                    provider._kernel(bucket)
                    bucket *= 2
            finally:
                provider.bank.release(lease, provider.bank.fence())
        retained = []
        for _ in range(2):
            for key in keys:
                manager = managers[key.logical.owner]
                desc = manager._descriptor_for_physical_key(key)
                entry = manager._dynamic_capture_state_direct_entry(desc)
                if key.logical.owner == "target":
                    # A physical step can begin with an MTP-only Graph.  In
                    # that case the target providers have not entered a dummy
                    # epoch yet.  Establish it at the actual target boundary
                    # instead of requiring stale state from an earlier key.
                    for provider in providers:
                        provider.prepare_execution(
                            dummy=True, num_tokens=desc.num_tokens
                        )
                if desc.cg_mode == CUDAGraphMode.FULL:
                    startup_replay = getattr(
                        manager, "allocation_profile_replay_context", None
                    )
                    if startup_replay is None:
                        manager.run_fullgraph(desc)
                    else:
                        with startup_replay(desc):
                            manager.run_fullgraph(desc)
                else:
                    entry.capture_state(CUDAGraphMode.PIECEWISE)
                if key.logical.owner == "target":
                    for provider in providers:
                        provider.finish_execution(dummy=True)
            # Sampling has a separate allocation lifetime after target. Its
            # maximum row count is X; values are intentionally synthetic.
            sample = torch.zeros(
                (step_key[2], runner.model_config.get_hidden_size()),
                dtype=torch.bfloat16,
                device=runner.device,
            )
            runner._dummy_sampler_run(sample)
            del sample
            torch.accelerator.synchronize()
            retained.append(torch.accelerator.memory_allocated())
        extra = max(
            0,
            torch.accelerator.max_memory_reserved() - baseline,
            baseline_free - torch.accelerator.get_memory_info()[0],
        )
        if retained[1] > retained[0]:
            raise RuntimeError(f"allocation replay retained new state: {retained}")
        return dict(
            replay_extra_bytes=extra,
            stable_replays=1,
            source_reads=sum(s.misses for s in sources) - before_reads,
            retained_bytes=retained,
        )
    finally:
        for manager in managers.values():
            manager.release_transaction_leases(transaction)


def profile_allocation_catalog(owner: Any, surface: Any, *, progress=None):
    """Measure physical allocation controls once; derive logical row prices."""
    from vllm.v1.core.elastic_graph import ElasticPlanKind
    from vllm.v1.engine.elastic_calibrator import ElasticCatalogCalibrator

    scheduler = owner.scheduler
    if scheduler.has_unfinished_requests():
        raise RuntimeError("allocation profile requires a request-free scheduler")
    config = owner.vllm_config
    policy = scheduler._elastic_graph_execution_policy
    corners, holdouts = allocation_profile_shapes(
        scheduler.num_spec_tokens,
        surface.decode_max_x,
        config.scheduler_config.max_num_batched_tokens,
    )
    controls = tuple(
        key
        for key in corners + holdouts
        if scheduler._resolve_elastic_step_physical_keys(key)
    )
    if not controls:
        raise RuntimeError("allocation profile has no Graph-backed physical controls")
    ElasticCatalogCalibrator(owner)._validate_surface_before_mutation(surface)
    previous_mode, previous_catalog = (
        scheduler._elastic_restore_mode,
        scheduler._elastic_graph_catalog,
    )
    scheduler._elastic_restore_mode = True
    scheduler._elastic_graph_catalog = {}
    samples: list[dict[str, Any]] = []
    started = time.monotonic()

    def execute(kind):
        scheduled = scheduler.schedule(physical_quiescent=True)
        if (
            scheduled.total_num_scheduled_tokens
            or scheduled.elastic_step_plan.kind != kind
        ):
            raise RuntimeError("allocation profile crossed a request boundary")
        output = owner.model_executor.execute_model(scheduled)
        if output is None or output.req_ids:
            raise RuntimeError("allocation maintenance returned model output")
        scheduler.update_from_output(scheduled, output)

    def reclaim():
        if scheduler.prepare_elastic_restore_idle_reclaim():
            execute(ElasticPlanKind.RECLAIM)

    completed = False
    try:
        reclaim()
        for key in controls:
            if progress is not None:
                progress(samples, key, time.monotonic() - started)
            if not scheduler.prepare_elastic_restore_capture(key):
                raise RuntimeError("allocation profile expected a cold owner set")
            execute(ElasticPlanKind.MAINTENANCE)
            scheduler.assert_elastic_restore_captures_hot((key,))
            observed = scheduler._elastic_graph_catalog[key]
            replays = owner.model_executor.collective_rpc(
                replay_allocation_profile, args=(key,)
            )
            sample = dict(
                step_key=list(key),
                capture_peak_bytes=observed["cold_peak_bytes"],
                resident_bytes=observed["resident_bytes"],
                floor_bytes=observed["floor_bytes"],
                replay_extra_bytes=max(r["replay_extra_bytes"] for r in replays),
                stable_replays=min(r["stable_replays"] for r in replays),
                source_reads=sum(r["source_reads"] for r in replays),
            )
            samples.append(sample)
            reclaim()
        proof = allocation_envelope_proof(
            k=scheduler.num_spec_tokens,
            max_x=surface.decode_max_x,
            budget=config.scheduler_config.max_num_batched_tokens,
            policy=policy.fingerprint,
            samples=samples,
            controls=controls,
        )
        rows = {key: allocation_envelope_row(proof) for key in surface.required}
        ElasticCatalogCalibrator(owner).validate_capacity(surface, rows)
        if progress is not None:
            progress(samples, None, time.monotonic() - started)
        completed = True
        return rows
    finally:
        try:
            # A failed RPC can leave a poisoned worker owner. Reclaiming through
            # it masks the original failure and is not a recovery protocol.
            # The contained caller must terminate that worker generation.
            if completed:
                reclaim()
        finally:
            scheduler._elastic_restore_mode = previous_mode
            scheduler._elastic_graph_catalog = previous_catalog
