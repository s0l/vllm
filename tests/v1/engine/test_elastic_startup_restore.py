# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.elastic_graph import (
    ElasticAdmissionController,
    ElasticResidencyReceipt,
    RuntimeGeneration,
)
from vllm.v1.engine import elastic_bootstrap
from vllm.v1.engine.core import EngineCore


def receipt(
    generation: str = "restore-generation",
    *,
    resident: int = 64,
    workspace: int = 32,
) -> ElasticResidencyReceipt:
    return ElasticResidencyReceipt(
        generation=RuntimeGeneration(generation),
        transaction_id=None,
        resident_bytes=resident,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=resident,
        cublas_workspace_bytes=workspace,
        entries=(),
    )


def test_startup_residency_publishes_one_typed_consensus_before_ready() -> None:
    core = object.__new__(EngineCore)
    accepted = receipt()
    scheduler = SimpleNamespace(
        _elastic_admission_controller=ElasticAdmissionController(
            RuntimeGeneration("restore-generation")
        ),
        _elastic_cublas_workspace_unit_bytes=0,
        _sync_elastic_residency_receipt=MagicMock(),
        _publish_elastic_startup_capacity=MagicMock(),
    )
    core.scheduler = scheduler
    core.collective_rpc = MagicMock(return_value=(accepted, accepted, accepted))

    core._synchronize_elastic_startup_residency()

    core.collective_rpc.assert_called_once_with("get_elastic_graph_residency_receipt")
    scheduler._sync_elastic_residency_receipt.assert_called_once_with(accepted)
    scheduler._publish_elastic_startup_capacity.assert_called_once_with()
    assert scheduler._elastic_cublas_workspace_unit_bytes == 32


@pytest.mark.parametrize(
    "receipts,match",
    [
        ((receipt("stale"),), "generation"),
        ((receipt(), receipt(resident=65)), "differs across worker ranks"),
    ],
)
def test_startup_residency_rejects_stale_or_rank_divergent_receipt(
    receipts, match
) -> None:
    core = object.__new__(EngineCore)
    scheduler = SimpleNamespace(
        _elastic_admission_controller=ElasticAdmissionController(
            RuntimeGeneration("restore-generation")
        ),
        _elastic_cublas_workspace_unit_bytes=17,
        _sync_elastic_residency_receipt=MagicMock(),
        _publish_elastic_startup_capacity=MagicMock(),
    )
    core.scheduler = scheduler
    core.collective_rpc = MagicMock(return_value=receipts)

    with pytest.raises(RuntimeError, match=match):
        core._synchronize_elastic_startup_residency()

    scheduler._sync_elastic_residency_receipt.assert_not_called()
    scheduler._publish_elastic_startup_capacity.assert_not_called()
    assert scheduler._elastic_cublas_workspace_unit_bytes == 17


def test_bounded_restart_restores_only_declared_full_carrier() -> None:
    core = object.__new__(EngineCore)
    piecewise = [0, 0, 1, 2, 0]
    full = [1, 3, 1, 1, 1]
    scheduler = SimpleNamespace(
        _elastic_graph_catalog_coverage={
            "representation": "bounded_exact_hotset",
            "mixed_max_x": 39,
            "required_step_keys": [piecewise, full],
            "restore_step_keys": [piecewise, full],
        },
        num_spec_tokens=3,
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(disable_speculation_on_non_decode=True)
        ),
        _elastic_restore_mode=False,
        max_num_running_reqs=64,
    )
    core.scheduler = scheduler
    core._run_elastic_full_restore_wave_in_epoch = MagicMock(
        return_value=((1, 3, 1, 1, 1), 1)
    )

    core._restore_elastic_bounded_hotset()

    core._run_elastic_full_restore_wave_in_epoch.assert_called_once_with(
        k=3, x=1, query_len=1, serial=1
    )
    assert scheduler._elastic_restore_mode is False
    assert scheduler.max_num_running_reqs == 39


@pytest.mark.parametrize("resident_bytes,expected", [(0, False), (64, True)])
def test_restore_preflight_reads_hot_residency_from_controller(
    resident_bytes: int,
    expected: bool,
) -> None:
    class StopAfterObservation(RuntimeError):
        pass

    controller = ElasticAdmissionController(RuntimeGeneration("restore-generation"))
    controller.publish_physical_accounting(
        resident_bytes=resident_bytes,
        floor_bytes=0,
        transition_floor_bytes=0,
        cublas_workspace_bytes=0,
        maintenance_step_key=None,
    )
    scheduler = SimpleNamespace(
        _elastic_admission_controller=controller,
        _canonical_elastic_graph_step_key=MagicMock(return_value=(0, 3, 1, 2, 0)),
    )
    core = object.__new__(EngineCore)
    core.scheduler = scheduler
    core._elastic_restore_decode_key = MagicMock(return_value=(1, 3, 1, 4, 4))
    core._begin_elastic_restore_physical_epoch = MagicMock()
    core._add_elastic_restore_request = MagicMock()

    def observe(_request_ids, *, retain_hot_graphs):
        assert retain_hot_graphs is expected
        raise StopAfterObservation

    core._prepare_elastic_restore_admission = MagicMock(side_effect=observe)

    with pytest.raises(StopAfterObservation):
        core._run_elastic_full_restore_wave_in_epoch(
            k=3,
            x=1,
            query_len=1,
            serial=1,
        )


def test_restore_drain_converts_idle_cleanup_debt_to_explicit_reclaim() -> None:
    core = object.__new__(EngineCore)
    scheduler = SimpleNamespace(
        has_requests=MagicMock(side_effect=(True, True, False)),
        _needs_elastic_idle_reclaim=MagicMock(side_effect=(True, True)),
        has_unfinished_requests=MagicMock(side_effect=(False, False)),
        has_finished_requests=MagicMock(side_effect=(True, False)),
    )
    core.scheduler = scheduler
    core._run_elastic_restore_step = MagicMock()
    core._reclaim_elastic_restore_hotset_before_wave = MagicMock()

    core._drain_elastic_restore()

    core._run_elastic_restore_step.assert_called_once_with()
    core._reclaim_elastic_restore_hotset_before_wave.assert_called_once_with()


def test_failed_restore_shuts_down_partial_engine_without_masking_failure() -> None:
    core = object.__new__(EngineCore)
    core.shutdown = MagicMock(side_effect=RuntimeError("teardown failed"))

    core._shutdown_failed_elastic_startup()

    core.shutdown.assert_called_once_with()


def test_prepare_elastic_runtime_preserves_production_barrier_order(
    monkeypatch,
) -> None:
    events = []
    kv_config = SimpleNamespace(kv_cache_groups=(object(),))
    manager = object()
    scheduler = SimpleNamespace(elastic_on_demand_graphs=True)
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            get_scheduler_cls=lambda: scheduler_cls,
        )
    )

    def initialize_kv_cache():
        events.append("kv")
        return kv_config

    def resolve_policy(config, resolved_kv, collective_rpc):
        assert config is vllm_config
        assert resolved_kv is kv_config
        events.append("policy")

    def scheduler_cls(**kwargs):
        assert kwargs["kv_cache_config"] is kv_config
        assert kwargs["structured_output_manager"] is manager
        events.append("scheduler")
        return scheduler

    def synchronize_generation(resolved_scheduler, collective_rpc):
        assert resolved_scheduler is scheduler
        events.append("generation")
        return {"scheduler": "generation", "workers": ["generation"] * 3}

    monkeypatch.setattr(
        elastic_bootstrap,
        "resolve_elastic_graph_execution_policy",
        resolve_policy,
    )
    monkeypatch.setattr(
        elastic_bootstrap,
        "resolve_kv_cache_block_sizes",
        lambda *_: (16, 16),
    )
    monkeypatch.setattr(
        elastic_bootstrap,
        "synchronize_elastic_runtime_generation",
        synchronize_generation,
    )

    prepared = elastic_bootstrap.prepare_elastic_runtime(
        vllm_config=vllm_config,
        initialize_kv_cache=initialize_kv_cache,
        collective_rpc=MagicMock(),
        include_finished_set=False,
        log_stats=True,
        structured_output_manager_factory=lambda: events.append("manager") or manager,
    )

    assert events == ["kv", "policy", "manager", "scheduler", "generation"]
    assert prepared.scheduler is scheduler
    assert prepared.generation_receipt == {
        "scheduler": "generation",
        "workers": ["generation"] * 3,
    }


def test_complete_elastic_startup_restores_then_publishes_before_ready() -> None:
    events = []
    owner = SimpleNamespace(
        scheduler=SimpleNamespace(
            elastic_on_demand_graphs=True,
            _elastic_require_catalog=True,
            _elastic_graph_catalog={(1, 3, 1, 1, 1): {}},
            _elastic_graph_catalog_coverage={"representation": "bounded_exact_hotset"},
        ),
        _restore_elastic_bounded_hotset=lambda: events.append("restore"),
        _restore_elastic_pinned_full_family=lambda: events.append("wrong-restore"),
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    outcome = elastic_bootstrap.complete_elastic_startup(owner)

    assert outcome == "restored"
    assert events == ["restore", "publish"]


def test_complete_elastic_startup_failure_stops_before_publication() -> None:
    events = []

    def fail_restore():
        events.append("restore")
        raise RuntimeError("restore failed")

    owner = SimpleNamespace(
        scheduler=SimpleNamespace(
            elastic_on_demand_graphs=True,
            _elastic_require_catalog=True,
            _elastic_graph_catalog={(1, 3, 1, 1, 1): {}},
            _elastic_graph_catalog_coverage={"representation": "bounded_exact_hotset"},
        ),
        _restore_elastic_bounded_hotset=fail_restore,
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    with pytest.raises(RuntimeError, match="restore failed"):
        elastic_bootstrap.complete_elastic_startup(owner)

    assert events == ["restore", "shutdown"]


def test_complete_elastic_startup_auto_calibrates_miss_before_restore(
    monkeypatch,
) -> None:
    events = []
    scheduler = SimpleNamespace(
        elastic_on_demand_graphs=True,
        _elastic_require_catalog=True,
        _elastic_auto_calibrate=True,
        _elastic_graph_catalog={},
        _elastic_graph_catalog_coverage={},
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        _restore_elastic_bounded_hotset=lambda: events.append("restore"),
        _restore_elastic_pinned_full_family=lambda: events.append("wrong-restore"),
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    def calibrate(resolved_owner):
        assert resolved_owner is owner
        events.append("calibrate")
        scheduler._elastic_graph_catalog = {(1, 3, 1, 1, 1): {}}
        scheduler._elastic_graph_catalog_coverage = {
            "representation": "bounded_exact_hotset"
        }

    monkeypatch.setattr(elastic_bootstrap, "_auto_calibrate_missing_catalog", calibrate)

    assert elastic_bootstrap.complete_elastic_startup(owner) == "restored"
    assert events == ["calibrate", "restore", "publish"]


def test_complete_elastic_startup_auto_calibration_failure_shuts_down(
    monkeypatch,
) -> None:
    events = []
    owner = SimpleNamespace(
        scheduler=SimpleNamespace(
            elastic_on_demand_graphs=True,
            _elastic_require_catalog=True,
            _elastic_auto_calibrate=True,
            _elastic_graph_catalog={},
        ),
        _synchronize_elastic_startup_residency=lambda: events.append("publish"),
        _shutdown_failed_elastic_startup=lambda: events.append("shutdown"),
    )

    def fail(_owner):
        events.append("calibrate")
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(elastic_bootstrap, "_auto_calibrate_missing_catalog", fail)

    with pytest.raises(RuntimeError, match="calibration failed"):
        elastic_bootstrap.complete_elastic_startup(owner)
    assert events == ["calibrate", "shutdown"]


def test_auto_calibration_seals_activates_and_records_receipt(
    tmp_path, monkeypatch
) -> None:
    from vllm import envs
    from vllm.v1.core import elastic_catalog
    from vllm.v1.engine import elastic_calibrator
    from vllm.v1.worker import startup_plan

    surface = tmp_path / "surface.json"
    surface.write_text("{}", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "elastic_graph_catalog" / "catalog.json"
    activated = []
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        activate_elastic_graph_catalog=lambda *args: activated.append(args),
    )
    owner = SimpleNamespace(vllm_config=object(), scheduler=scheduler)
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_SURFACE", str(surface))
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_RECEIPT", str(receipt_path))
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        elastic_catalog, "load_sealed_catalog", lambda *_args, **_kwargs: {"ok": 1}
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_args: "1123456789abcdef",
    )
    monkeypatch.setattr(
        startup_plan, "load_elastic_graph_catalog", lambda *_args: {"row": {}}
    )
    monkeypatch.setattr(
        startup_plan,
        "load_elastic_graph_catalog_coverage",
        lambda *_args: {"decode_max_x": 39},
    )
    monkeypatch.setattr(
        elastic_calibrator,
        "calibrate_and_publish_catalog",
        lambda *_args, **_kwargs: SimpleNamespace(
            destination=destination,
            fingerprint="1123456789abcdef",
            measured_shapes=51,
            wall_seconds=1.0,
        ),
    )

    elastic_bootstrap._auto_calibrate_missing_catalog(owner)

    assert activated == [({"row": {}}, {"decode_max_x": 39})]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["stage"] == "complete"
    assert receipt["measured_shapes"] == 51
    assert receipt["destination"] == str(destination)


def test_auto_calibration_failure_receipt_does_not_activate(tmp_path, monkeypatch):
    from vllm import envs
    from vllm.v1.core import elastic_catalog
    from vllm.v1.engine import elastic_calibrator
    from vllm.v1.worker import startup_plan

    surface = tmp_path / "surface.json"
    surface.write_text("{}", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    scheduler = SimpleNamespace(
        kv_cache_config=object(),
        activate_elastic_graph_catalog=MagicMock(),
    )
    owner = SimpleNamespace(vllm_config=object(), scheduler=scheduler)
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_SURFACE", str(surface))
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_RECEIPT", str(receipt_path))
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        elastic_catalog, "load_sealed_catalog", lambda *_args, **_kwargs: {"ok": 1}
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_args: "1123456789abcdef",
    )
    monkeypatch.setattr(
        elastic_calibrator,
        "calibrate_and_publish_catalog",
        MagicMock(side_effect=RuntimeError("shape contracted")),
    )

    with pytest.raises(RuntimeError, match="shape contracted"):
        elastic_bootstrap._auto_calibrate_missing_catalog(owner)

    scheduler.activate_elastic_graph_catalog.assert_not_called()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["stage"] == "failed"
    assert receipt["error"]["type"] == "RuntimeError"
