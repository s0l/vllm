# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from vllm.v1.core.elastic_catalog import (
    ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
    load_sealed_catalog,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.elastic_calibrator import (
    CalibrationSurface,
    ElasticCatalogCalibrator,
)
from vllm.v1.worker.elastic_catalog_tool import (
    finalize_catalog,
    publish_measured_catalog,
    rebind_catalog,
)


def _source(tmp_path, **updates):
    payload = {
        "schema": ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "fingerprint": "0123456789abcdef",
        "sealed": True,
        "migrated_from": "fedcba9876543210",
        "migration_scope": "graph-execution-policy-v1",
        "complete_shapes": 1,
        "coverage": {
            "required_shapes": 1,
            "required_step_keys": [[1, 3, 1, 1, 1]],
            "restore_step_keys": [[1, 3, 1, 1, 1]],
        },
        "shapes": [{"step_key": [1, 3, 1, 1, 1]}],
    }
    payload.update(updates)
    path = tmp_path / "source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_finalize_catalog_strips_runtime_lineage_without_mutating_source(tmp_path):
    source = _source(tmp_path)

    destination = finalize_catalog(source, tmp_path / "candidate")

    original = json.loads(source.read_text(encoding="utf-8"))
    finalized = json.loads(destination.read_text(encoding="utf-8"))
    assert original["migrated_from"] == "fedcba9876543210"
    assert "migrated_from" not in finalized
    assert "migration_scope" not in finalized
    assert finalized["finalized_offline"] is True
    assert finalized["coverage"] == original["coverage"]
    assert finalized["shapes"] == original["shapes"]


def test_finalize_catalog_rejects_partial_or_unmigrated_input(tmp_path):
    with pytest.raises(RuntimeError, match="sealed"):
        finalize_catalog(_source(tmp_path, sealed=False), tmp_path / "candidate")

    with pytest.raises(RuntimeError, match="no explicit migration lineage"):
        finalize_catalog(
            _source(tmp_path, migrated_from=None, migration_scope=None),
            tmp_path / "candidate-2",
        )


def test_finalize_catalog_never_overwrites(tmp_path):
    source = _source(tmp_path)
    root = tmp_path / "candidate"
    finalize_catalog(source, root)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_catalog(source, root)


def test_finalize_catalog_rejects_incomplete_required_coverage(tmp_path):
    source = _source(
        tmp_path,
        coverage={
            "required_shapes": 2,
            "required_step_keys": [[1, 3, 1, 1, 1], [1, 3, 2, 2, 1]],
            "restore_step_keys": [[1, 3, 1, 1, 1]],
        },
    )

    with pytest.raises(RuntimeError, match="required coverage is incomplete"):
        finalize_catalog(source, tmp_path / "candidate")


def test_rebind_catalog_records_source_identity_without_mutating_source(tmp_path):
    migrated = _source(tmp_path)
    finalized = finalize_catalog(migrated, tmp_path / "finalized")

    destination = rebind_catalog(
        finalized,
        tmp_path / "candidate",
        "1123456789abcdef",
        "KV6 source-only cleanup",
    )

    original = json.loads(finalized.read_text(encoding="utf-8"))
    rebound = json.loads(destination.read_text(encoding="utf-8"))
    assert original["fingerprint"] == "0123456789abcdef"
    assert rebound["fingerprint"] == "1123456789abcdef"
    assert rebound["offline_rebind"]["source_fingerprint"] == original["fingerprint"]
    assert len(rebound["offline_rebind"]["source_sha256"]) == 64
    assert rebound["coverage"] == original["coverage"]
    assert rebound["shapes"] == original["shapes"]


def test_rebind_catalog_rejects_invalid_identity_and_overwrite(tmp_path):
    migrated = _source(tmp_path)
    finalized = finalize_catalog(migrated, tmp_path / "finalized")
    root = tmp_path / "candidate"

    with pytest.raises(RuntimeError, match="fingerprint is invalid"):
        rebind_catalog(finalized, root, "not-a-fingerprint", "KV6")
    with pytest.raises(RuntimeError, match="non-empty reason"):
        rebind_catalog(finalized, root, "1123456789abcdef", "")
    with pytest.raises(RuntimeError, match="equals its source"):
        rebind_catalog(finalized, root, "0123456789abcdef", "KV6")

    rebind_catalog(finalized, root, "1123456789abcdef", "KV6")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rebind_catalog(finalized, root, "1123456789abcdef", "KV6")


def test_scheduler_activates_measured_catalog_only_before_requests():
    measurements = []
    envelopes = []
    key = (0, 3, 2, 8, 4)
    row = _complete_row()
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_graph_catalog_coverage={},
        max_num_running_reqs=47,
        has_requests=lambda: False,
        _record_elastic_capture_envelope=lambda *args: envelopes.append(args),
        _elastic_admission_controller=SimpleNamespace(
            record_measurement=lambda *args: measurements.append(args)
        ),
    )
    coverage = {
        "representation": "bounded_exact_hotset",
        "decode_max_x": 39,
        "mixed_max_x": 39,
        "full_context_max_x": 1,
    }

    Scheduler.activate_elastic_graph_catalog(scheduler, {key: row}, coverage)

    assert scheduler.max_num_running_reqs == 39
    assert scheduler._elastic_graph_catalog == {key: row}
    assert envelopes and measurements == [(key, 1)]

    with pytest.raises(RuntimeError, match="previously active catalog"):
        Scheduler.activate_elastic_graph_catalog(scheduler, {key: row}, coverage)


def _surface_payload(**coverage_updates):
    coverage = {
        "representation": "bounded_exact_hotset",
        "graph_execution_policy_fingerprint": "policy",
        "required_step_keys": [[0, 3, 4, 16, 4], [1, 3, 4, 4, 1]],
        "restore_step_keys": [[0, 3, 4, 16, 4], [1, 3, 4, 4, 1]],
        "decode_max_x": 4,
        "mixed_max_x": 4,
        "full_context_max_x": 1,
    }
    coverage.update(coverage_updates)
    return {"fingerprint": "0123456789abcdef", "coverage": coverage}


def test_calibration_surface_accepts_exact_runtime_contract():
    surface = CalibrationSurface.from_payload(
        _surface_payload(),
        policy_fingerprint="policy",
        configured_k=3,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )

    assert surface.decode_max_x == 4
    assert surface.required == ((0, 3, 4, 16, 4), (1, 3, 4, 4, 1))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"graph_execution_policy_fingerprint": "stale"}, "policy"),
        ({"restore_step_keys": [[1, 3, 3, 3, 1]]}, "restore"),
        ({"decode_max_x": 3}, "boundaries"),
    ],
)
def test_calibration_surface_rejects_stale_or_incomplete_contract(updates, message):
    with pytest.raises(RuntimeError, match=message):
        CalibrationSurface.from_payload(
            _surface_payload(**updates),
            policy_fingerprint="policy",
            configured_k=3,
            max_num_seqs=4,
            max_num_batched_tokens=16,
        )


def test_publish_measured_catalog_is_atomic_and_complete(tmp_path, monkeypatch):
    from vllm.v1.core import elastic_graph
    from vllm.v1.worker import startup_plan

    policy = SimpleNamespace(
        fingerprint="policy",
        verifier_contract="verifier",
        verifier_configuration="verifier-config",
        math_contract="math",
        to_payload=lambda: {"fingerprint": "policy"},
    )
    monkeypatch.setattr(
        startup_plan, "_effective_graph_execution_policy", lambda _config: policy
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda _config, _kv: "1123456789abcdef",
    )
    monkeypatch.setattr(
        startup_plan,
        "_catalog_policy_row_metadata",
        lambda *_args, **_kwargs: {"policy_fingerprint": "policy"},
    )
    monkeypatch.setattr(
        elastic_graph, "configured_compiled_piecewise_sizes", lambda _config: set()
    )
    key = (1, 3, 1, 1, 1)
    row = {
        "cold_peak_bytes": 1,
        "hot_peak_bytes": 1,
        "resident_bytes": 1,
        "floor_bytes": 0,
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 2,
        "hot_stable_replays": 1,
    }
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16)
    )

    destination = publish_measured_catalog(
        config,
        object(),
        {key: row},
        required=(key,),
        restore=(key,),
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
        calibration_wall_seconds=1.0,
        output_root=tmp_path,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert load_sealed_catalog(destination, require_migration=False) == payload
    assert payload["sealed"] is True
    assert payload["finalized_offline"] is True
    assert payload["complete_shapes"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_measured_catalog(
            config,
            object(),
            {key: row},
            required=(key,),
            restore=(key,),
            decode_max_x=1,
            mixed_max_x=1,
            full_context_max_x=1,
            calibration_wall_seconds=1.0,
            output_root=tmp_path,
        )


def test_publish_measured_catalog_rejects_partial_row(tmp_path):
    key = (1, 3, 1, 1, 1)
    with pytest.raises(RuntimeError, match="incomplete"):
        publish_measured_catalog(
            object(),
            object(),
            {key: {"cold_peak_bytes": 1}},
            required=(key,),
            restore=(key,),
            decode_max_x=1,
            mixed_max_x=1,
            full_context_max_x=1,
            calibration_wall_seconds=1.0,
            output_root=tmp_path,
        )


def _complete_row():
    return {
        "cold_peak_bytes": 1,
        "hot_peak_bytes": 1,
        "resident_bytes": 1,
        "floor_bytes": 0,
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 2,
        "hot_stable_replays": 1,
    }


def test_calibrator_measures_every_surface_family(monkeypatch):
    original_key = (1, 0, 1, 1, 1)
    original_row = _complete_row()
    catalog = {original_key: original_row}
    original_coverage = {"representation": "original"}
    scheduler = SimpleNamespace(
        _elastic_graph_catalog=catalog,
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage=original_coverage,
        _resolve_elastic_step_physical_keys=lambda _key: ("physical",),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
    )
    calls = []

    def full(**kwargs):
        key = (
            1 if kwargs["query_len"] == 1 else 0,
            3,
            kwargs["x"],
            kwargs["x"] * kwargs["query_len"],
            kwargs["query_len"],
        )
        catalog[key] = _complete_row()
        calls.append(("full", key))
        return key, kwargs["x"]

    owner._run_elastic_full_restore_wave = full
    calibrator = ElasticCatalogCalibrator(owner)

    def balanced(*, k, x, m):
        key = (0, k, x, m, 0)
        catalog[key] = _complete_row()
        calls.append(("balanced", key))
        return key, x

    def mixed(*, k, x, m, query_len):
        key = (0, k, x, m, 0)
        calls.append((f"mixed-q{query_len}", key))
        return key, x

    monkeypatch.setattr(calibrator, "_balanced_prefill_pair", balanced)
    monkeypatch.setattr(calibrator, "_mixed_prefill_pair", mixed)
    surface = CalibrationSurface(
        required=((0, 3, 2, 8, 4), (0, 3, 4, 16, 0), (1, 3, 4, 4, 1)),
        restore=((0, 3, 2, 8, 4), (1, 3, 4, 4, 1)),
        decode_max_x=4,
        mixed_max_x=4,
        full_context_max_x=1,
    )

    measured = calibrator.calibrate(surface)

    assert set(measured) == set(surface.required)
    assert ("full", (0, 3, 2, 8, 4)) in calls
    assert ("full", (1, 3, 4, 4, 1)) in calls
    assert ("balanced", (0, 3, 4, 16, 0)) in calls
    assert ("mixed-q1", (0, 3, 4, 16, 0)) in calls
    assert ("mixed-q4", (0, 3, 4, 16, 0)) in calls
    assert scheduler._elastic_restore_mode is False
    assert scheduler._elastic_graph_catalog == {original_key: original_row}
    assert scheduler._elastic_graph_catalog is not catalog
    assert scheduler._elastic_graph_catalog_coverage is original_coverage


def test_calibrator_rejects_surface_contraction(monkeypatch):
    original_key = (1, 0, 1, 1, 1)
    original_catalog = {original_key: _complete_row()}
    original_coverage = {"representation": "original"}
    scheduler = SimpleNamespace(
        _elastic_graph_catalog=original_catalog,
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage=original_coverage,
        _resolve_elastic_step_physical_keys=lambda _key: ("physical",),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
    )
    calibrator = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(
        calibrator,
        "_balanced_prefill_pair",
        lambda **_kwargs: (None, 3),
    )
    surface = CalibrationSurface(
        required=((0, 3, 4, 16, 0),),
        restore=((0, 3, 4, 16, 0),),
        decode_max_x=4,
        mixed_max_x=4,
        full_context_max_x=1,
    )

    with pytest.raises(RuntimeError, match="contracted accepted surface"):
        calibrator.calibrate(surface)
    assert scheduler._elastic_restore_mode is False
    assert scheduler._elastic_graph_catalog == original_catalog
    assert scheduler._elastic_graph_catalog is not original_catalog
    assert scheduler._elastic_graph_catalog_coverage is original_coverage


def test_calibrator_rejects_compiled_only_required_shape():
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
        _resolve_elastic_step_physical_keys=lambda _key: (),
        num_spec_tokens=3,
    )
    calibrator = ElasticCatalogCalibrator(
        SimpleNamespace(
            scheduler=scheduler,
            is_pooling_model=False,
            async_scheduling=False,
        )
    )
    surface = CalibrationSurface(
        required=((0, 3, 4, 16, 0),),
        restore=((0, 3, 4, 16, 0),),
        decode_max_x=4,
        mixed_max_x=4,
        full_context_max_x=1,
    )

    with pytest.raises(RuntimeError, match="compiled-only"):
        calibrator.calibrate(surface)


def test_calibrator_validates_full_context_against_measured_restore():
    coordinator = SimpleNamespace(max_elastic_full_context_requests=lambda *_args: 0)
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
        _elastic_primary_blocks_per_max_request=10,
    )
    calibrator = ElasticCatalogCalibrator(SimpleNamespace(scheduler=scheduler))
    full = (1, 3, 1, 1, 1)
    piecewise = (0, 3, 1, 4, 4)
    surface = CalibrationSurface(
        required=(piecewise, full),
        restore=(piecewise, full),
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
    )
    catalog = {
        full: {"cold_peak_bytes": 1, "hot_peak_bytes": 2},
        piecewise: {"cold_peak_bytes": 2, "hot_peak_bytes": 3},
    }

    with pytest.raises(RuntimeError, match="full-context"):
        calibrator.validate_capacity(surface, catalog)
