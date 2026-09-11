# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real ownership/lease composition; allocation and worker RPC are controls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.elastic_graph import (
    ElasticAdmissionController,
    GraphExecutionPolicy,
    GraphPrice,
    OwnerGraphExecutionPolicy,
    RuntimeGeneration,
)
from vllm.v1.core.elastic_memory_profile import CONTRACT
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.elastic_bootstrap import complete_elastic_startup


def startup_control():
    scheduler = object.__new__(Scheduler)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_require_catalog = True
    scheduler._elastic_graph_catalog = {(0, 3, 64, 256, 0): {}}
    scheduler._elastic_graph_catalog_coverage = {
        "representation": "bounded_exact_hotset",
        "memory_evidence_contract": CONTRACT,
        "serving_carrier_contract": "retained-terminal-mtp-no-cold-serving-v1",
        "serving_carrier_owner": "mtp_decode",
        "decode_max_x": 64,
        "mixed_max_x": 64,
    }
    scheduler._elastic_graph_execution_policy = GraphExecutionPolicy(
        verifier_contract="native-backend-v1",
        math_contract="native-target-baseline-v1",
        owners=(
            OwnerGraphExecutionPolicy(
                "mtp_decode",
                (1,),
                "NONE",
                activation="speculative",
                token_source="requests",
                execution_order=2,
            ),
            OwnerGraphExecutionPolicy(
                "mtp_prefill",
                (4,),
                "PIECEWISE",
                activation="speculative",
                execution_order=1,
            ),
            OwnerGraphExecutionPolicy("target", (), "PIECEWISE"),
        ),
    )
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=4096)
    scheduler.max_num_running_reqs = 64
    scheduler.num_spec_tokens = 3
    scheduler._elastic_restore_mode = False
    scheduler._elastic_restore_retention_id = None
    scheduler._elastic_restore_retained_physical_keys = ()
    scheduler._elastic_serving_carrier_keys = ()
    scheduler._elastic_admission_controller = ElasticAdmissionController(
        RuntimeGeneration("profiled-startup-control")
    )
    scheduler.has_unfinished_requests = lambda: False
    controller = scheduler._elastic_admission_controller
    events: list[tuple] = []

    def capture(key):
        assert scheduler._elastic_restore_mode
        events.append(("capture", key))
        for physical in scheduler._resolve_elastic_step_physical_keys(key):
            if (
                physical not in controller.entries
                or not controller.entries[physical].hot
            ):
                controller.publish_hot(
                    physical, GraphPrice(10, 20, physical.identity), pinned=False
                )

    def reclaim():
        # Worker allocation is simulated; authoritative lease and serving-owner
        # selection are real. Synchronization rejects losing a retained lease.
        keep = set(scheduler._elastic_serving_carrier_keys)
        events.append(("reclaim", tuple(keep)))
        controller.synchronize_hot(
            (key, entry.price, False, 0)
            for key, entry in controller.entries.items()
            if key in keep and entry.hot
        )

    def replay(_function, *, args):
        scheduler.assert_elastic_restore_captures_hot((args[0],))
        events.append(("replay", args[0]))
        return [{"source_reads": 0, "stable_replays": 1}] * 3

    owner = SimpleNamespace(
        scheduler=scheduler,
        _prepare_elastic_restore_capture=MagicMock(side_effect=capture),
        _reclaim_elastic_restore_hotset_before_wave=MagicMock(side_effect=reclaim),
        collective_rpc=MagicMock(side_effect=replay),
        _restore_elastic_bounded_hotset=MagicMock(),
        _synchronize_elastic_startup_residency=MagicMock(
            side_effect=lambda: events.append(("publish",))
        ),
        _shutdown_failed_elastic_startup=MagicMock(),
    )
    return owner, events


def test_profiled_startup_reclaims_target_but_keeps_terminal_mtp_before_ready():
    owner, events = startup_control()
    assert complete_elastic_startup(owner) == "restored"
    assert [e[1] for e in events if e[0] == "capture"] == [
        (0, 3, 64, 64, 1),
        (0, 3, 64, 256, 4),
    ]
    scheduler = owner.scheduler
    hot = {
        key
        for key, entry in scheduler._elastic_admission_controller.entries.items()
        if entry.hot
    }
    assert hot == set(scheduler._elastic_serving_carrier_keys)
    assert len(hot) == 1
    assert next(iter(hot)).logical.owner == "mtp_decode"
    assert all(
        not e.leases for e in scheduler._elastic_admission_controller.entries.values()
    )
    assert scheduler._elastic_restore_retention_id is None
    assert scheduler._elastic_restore_mode is False
    assert events[-1] == ("publish",)
    owner._restore_elastic_bounded_hotset.assert_not_called()
    owner._shutdown_failed_elastic_startup.assert_not_called()


@pytest.mark.parametrize("failure", ["io", "unstable", "empty", "capture", "promotion"])
def test_profiled_startup_failure_never_publishes_and_fresh_epoch_recovers(failure):
    owner, _ = startup_control()
    if failure in {"io", "unstable", "empty"}:
        owner.collective_rpc.side_effect = None
        owner.collective_rpc.return_value = (
            []
            if failure == "empty"
            else [
                {
                    "source_reads": int(failure == "io"),
                    "stable_replays": int(failure != "unstable"),
                }
            ]
            * 3
        )
    elif failure == "capture":
        owner._prepare_elastic_restore_capture.side_effect = RuntimeError("capture")
    else:
        owner.scheduler.promote_elastic_restore_retention_to_serving = MagicMock(
            side_effect=RuntimeError("promotion")
        )
    with pytest.raises(RuntimeError):
        complete_elastic_startup(owner)
    owner._synchronize_elastic_startup_residency.assert_not_called()
    owner._shutdown_failed_elastic_startup.assert_called_once_with()
    assert owner.scheduler._elastic_restore_mode is False
    assert owner.scheduler._elastic_restore_retention_id is None
    assert all(
        not e.leases
        for e in owner.scheduler._elastic_admission_controller.entries.values()
    )
    recovery, _ = startup_control()
    assert complete_elastic_startup(recovery) == "restored"


@pytest.mark.parametrize("value", [0, 65, True])
def test_profiled_startup_rejects_bad_domain_before_capture(value):
    owner, _ = startup_control()
    owner.scheduler._elastic_graph_catalog_coverage["decode_max_x"] = value
    with pytest.raises(RuntimeError, match="carrier contract"):
        complete_elastic_startup(owner)
    owner._prepare_elastic_restore_capture.assert_not_called()
    owner._synchronize_elastic_startup_residency.assert_not_called()


def test_profiled_startup_rejects_request_admission_before_capture():
    owner, _ = startup_control()
    owner.scheduler.has_unfinished_requests = lambda: True
    with pytest.raises(RuntimeError, match="carrier contract"):
        complete_elastic_startup(owner)
    owner._prepare_elastic_restore_capture.assert_not_called()


@pytest.mark.parametrize("valid", [True, False])
def test_profiled_capacity_uses_policy_decode_phase_not_full_mode(valid, caplog):
    from vllm.v1.core.elastic_memory_profile import (
        allocation_envelope_proof,
        allocation_envelope_row,
        allocation_profile_shapes,
    )

    owner, _ = startup_control()
    scheduler = owner.scheduler
    shapes = allocation_profile_shapes(3, 64, 4096)
    samples = [
        {
            "step_key": list(key),
            "role": role,
            "capture_peak_bytes": 100,
            "resident_bytes": 80,
            "floor_bytes": 20,
            "replay_extra_bytes": 10,
            "stable_replays": 1,
            "source_reads": 0,
        }
        for role, keys in enumerate(shapes)
        for key in keys
    ]
    proof = allocation_envelope_proof(
        k=3,
        max_x=64,
        budget=4096,
        policy=scheduler._elastic_graph_execution_policy.fingerprint,
        samples=samples,
    )
    row = allocation_envelope_row(proof)
    scheduler._elastic_graph_catalog = {(0, 3, 64, 4096, 0): row}
    if valid:
        scheduler._elastic_graph_catalog[(0, 3, 64, 256, 0)] = row
    scheduler._elastic_primary_blocks_per_max_request = 1
    scheduler._elastic_graph_catalog_coverage["full_context_max_x"] = 0
    coordinator = MagicMock()
    coordinator.max_elastic_full_context_requests.return_value = 0
    coordinator.elastic_full_context_capacity_receipt.return_value = dict(
        residual_attention_blocks=100,
        attention_blocks=100,
        required_attention_blocks=0,
        gdn_blocks=0,
    )
    scheduler.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
    caplog.set_level("INFO")
    if valid:
        scheduler._publish_elastic_startup_capacity()
        assert "DecodeMaxX[K3]=64" in caplog.text
        assert "GuaranteedMaxX[K3,B4096]=64" in caplog.text
        assert row["cold_observations"] == row["hot_observations"] == 0
    else:
        with pytest.raises(RuntimeError, match="complete decode row"):
            scheduler._publish_elastic_startup_capacity()
