# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.elastic_catalog import (
    ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
    load_sealed_catalog,
    write_json_exclusive,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.elastic_calibrator import (
    CalibrationCheckpoint,
    CalibrationSurface,
    ElasticCalibrationRestartRequired,
    ElasticCatalogCalibrator,
    calibration_checkpoint_payload,
    parse_calibration_checkpoint,
)
from vllm.v1.worker.elastic_catalog_tool import (
    finalize_catalog,
    publish_measured_catalog,
    rebind_catalog,
)
from vllm.v1.worker.startup_plan import ElasticGraphCatalog


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
            "semantic_token_witnesses": [],
            "semantic_witness_contract": "live-current-to-physical-piecewise-v1",
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


def test_calibration_surface_loader_accepts_only_finalized_previous_schema(tmp_path):
    previous = _source(
        tmp_path,
        schema=ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION - 1,
        finalized_offline=True,
    )
    previous_payload = json.loads(previous.read_text(encoding="utf-8"))
    previous_payload.pop("migrated_from")
    previous_payload.pop("migration_scope")
    previous.write_text(json.dumps(previous_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="sealed current-schema"):
        load_sealed_catalog(previous, require_migration=False)
    loaded = load_sealed_catalog(
        previous,
        require_migration=False,
        allow_previous_schema=True,
    )
    assert loaded["schema"] == ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION - 1

    unfinalized = _source(
        tmp_path,
        schema=ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION - 1,
        finalized_offline=False,
    )
    unfinalized_payload = json.loads(unfinalized.read_text(encoding="utf-8"))
    unfinalized_payload.pop("migrated_from")
    unfinalized_payload.pop("migration_scope")
    unfinalized.write_text(json.dumps(unfinalized_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="offline-finalized"):
        load_sealed_catalog(
            unfinalized,
            require_migration=False,
            allow_previous_schema=True,
        )


def test_current_catalog_requires_semantic_witness_contract(tmp_path):
    source = _source(tmp_path, finalized_offline=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.pop("migrated_from")
    payload.pop("migration_scope")
    payload["coverage"].pop("semantic_witness_contract")
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="semantic witness contract"):
        load_sealed_catalog(source, require_migration=False)


def test_finalize_catalog_never_overwrites(tmp_path):
    source = _source(tmp_path)
    root = tmp_path / "candidate"
    finalize_catalog(source, root)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_catalog(source, root)


def test_exclusive_catalog_publish_has_one_concurrency_winner(tmp_path):
    destination = tmp_path / "catalog.json"

    def publish(value: int):
        try:
            write_json_exclusive(destination, {"winner": value})
            return "published"
        except FileExistsError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))

    assert sorted(outcomes) == ["lost", "published"]
    assert json.loads(destination.read_text(encoding="utf-8"))["winner"] in {1, 2}


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


@pytest.mark.parametrize("inventory", ["required_step_keys", "restore_step_keys"])
def test_sealed_catalog_rejects_duplicate_key_inventory(tmp_path, inventory):
    source = _source(tmp_path, finalized_offline=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.pop("migrated_from")
    payload.pop("migration_scope")
    payload["coverage"][inventory] *= 2
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=f"duplicate {inventory.split('_')[0]}"):
        load_sealed_catalog(source, require_migration=False)


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
        _elastic_short_decode_inventory={47: ("raw",)},
        max_num_running_reqs=47,
        has_requests=lambda: False,
        _rebuild_elastic_short_decode_inventory=Mock(),
        _elastic_admission_controller=SimpleNamespace(
            record_measurement=lambda *args: measurements.append(args),
            record_capture_envelope=lambda *args, **kwargs: envelopes.append(
                (args, kwargs)
            ),
        ),
    )
    coverage = {
        "representation": "bounded_exact_hotset",
        "decode_max_x": 39,
        "mixed_max_x": 39,
        "full_context_max_x": 1,
        "required_step_keys": [list(key)],
        "restore_step_keys": [],
    }

    catalog = ElasticGraphCatalog(source_sha256="a" * 64)
    catalog[key] = row
    coverage["_catalog_source_sha256"] = "a" * 64
    Scheduler.activate_elastic_graph_catalog(scheduler, catalog, coverage)

    assert scheduler.max_num_running_reqs == 39
    assert scheduler._elastic_graph_catalog == {key: row}
    assert envelopes and measurements == [(key, 1)]
    scheduler._rebuild_elastic_short_decode_inventory.assert_called_once_with(39)

    with pytest.raises(RuntimeError, match="previously active catalog"):
        Scheduler.activate_elastic_graph_catalog(scheduler, catalog, coverage)


def test_scheduler_rejects_catalog_rows_and_coverage_from_different_bytes():
    key = (0, 3, 2, 8, 4)
    catalog = ElasticGraphCatalog(source_sha256="a" * 64)
    catalog[key] = _complete_row()
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        max_num_running_reqs=47,
        _elastic_admission_controller=SimpleNamespace(
            pending_maintenance_plan=None,
        ),
    )
    coverage = {
        "representation": "bounded_exact_hotset",
        "decode_max_x": 39,
        "mixed_max_x": 39,
        "full_context_max_x": 1,
        "required_step_keys": [list(key)],
        "restore_step_keys": [],
        "_catalog_source_sha256": "b" * 64,
    }

    with pytest.raises(RuntimeError, match="differ at source bytes"):
        Scheduler.activate_elastic_graph_catalog(scheduler, catalog, coverage)


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
    return {"schema": 6, "fingerprint": "0123456789abcdef", "coverage": coverage}


def test_calibration_surface_accepts_exact_runtime_contract():
    surface = CalibrationSurface.from_payload(
        _surface_payload(),
        policy_fingerprint="policy",
        configured_k=3,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )

    assert surface.decode_max_x == 4
    assert surface.required == (
        (0, 3, 1, 1, 0),
        (0, 3, 1, 2, 0),
        (0, 3, 1, 4, 0),
        (0, 3, 2, 8, 0),
        (0, 3, 4, 16, 4),
        (1, 3, 4, 4, 1),
    )
    assert surface.semantic_token_witnesses == (
        ((0, 3, 1, 1, 0), 1),
        ((0, 3, 1, 2, 0), 2),
        ((0, 3, 1, 4, 0), 3),
        ((0, 3, 2, 8, 0), 5),
    )
    assert surface.mixed_query_witnesses == (
        ((0, 3, 2, 8, 0), 1),
        ((0, 3, 2, 8, 0), 4),
    )


def test_calibration_surface_accepts_explicit_shape_specific_mixed_witnesses():
    payload = _surface_payload(
        required_step_keys=[
            [0, 3, 3, 16, 0],
            [0, 3, 4, 16, 0],
            [0, 3, 4, 16, 4],
            [1, 3, 4, 4, 1],
        ],
        restore_step_keys=[[0, 3, 4, 16, 4], [1, 3, 4, 4, 1]],
        mixed_query_witnesses=[
            {"step_key": [0, 3, 4, 16, 0], "query_len": 1},
            {"step_key": [0, 3, 3, 16, 0], "query_len": 4},
        ],
    )

    surface = CalibrationSurface.from_payload(
        payload,
        policy_fingerprint="policy",
        configured_k=3,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )

    assert surface.mixed_query_witnesses == (
        ((0, 3, 3, 16, 0), 4),
        ((0, 3, 4, 16, 0), 1),
    )


def test_calibration_surface_rejects_mixed_witness_outside_physical_domain():
    with pytest.raises(RuntimeError, match="outside its declared"):
        CalibrationSurface.from_payload(
            _surface_payload(
                mixed_query_witnesses=[{"step_key": [0, 3, 3, 16, 0], "query_len": 4}]
            ),
            policy_fingerprint="policy",
            configured_k=3,
            max_num_seqs=4,
            max_num_batched_tokens=16,
        )


def test_calibration_surface_rejects_missing_mixed_query_class():
    with pytest.raises(RuntimeError, match="incomplete mixed query classes"):
        CalibrationSurface.from_payload(
            _surface_payload(
                required_step_keys=[
                    [0, 3, 2, 8, 0],
                    [0, 3, 4, 16, 4],
                    [1, 3, 4, 4, 1],
                ],
                mixed_query_witnesses=[{"step_key": [0, 3, 2, 8, 0], "query_len": 1}],
            ),
            policy_fingerprint="policy",
            configured_k=3,
            max_num_seqs=4,
            max_num_batched_tokens=16,
        )


def test_schema5_surface_compiles_semantic_full_x_to_bounded_physical_rows():
    payload = _surface_payload(
        required_step_keys=[
            [0, 3, 5, 20, 4],
            *[[1, 3, x, x, 1] for x in range(1, 6)],
        ],
        restore_step_keys=[[0, 3, 5, 20, 4], [1, 3, 5, 5, 1]],
        required_shapes=6,
        decode_max_x=5,
        mixed_max_x=5,
    )
    payload["schema"] = 5

    surface = CalibrationSurface.from_payload(
        payload,
        policy_fingerprint="policy",
        configured_k=3,
        max_num_seqs=5,
        max_num_batched_tokens=32,
    )

    full = tuple(key for key in surface.required if key[0] == 1)
    assert full == (
        (1, 3, 1, 1, 1),
        (1, 3, 2, 2, 1),
        (1, 3, 3, 3, 1),
        (1, 3, 4, 4, 1),
        (1, 3, 5, 5, 1),
    )
    assert surface.restore == ((0, 3, 5, 20, 4), (1, 3, 5, 5, 1))
    assert surface.source_schema == 5
    assert surface.source_required_shapes == 6
    assert (
        surface.migration_contract == "schema5-full-owner-evidence-and-bounded-alias-v1"
    )
    assert len(surface.full_aliases) == 5
    assert ((1, 3, 3, 3, 1), (1, 3, 4, 4, 1)) in surface.full_aliases
    assert surface.owner_evidence_keys == full


def test_schema5_surface_rejects_nonsemantic_full_geometry():
    payload = _surface_payload(
        required_step_keys=[[0, 3, 4, 16, 4], [1, 3, 3, 4, 1]],
        restore_step_keys=[[0, 3, 4, 16, 4], [1, 3, 3, 4, 1]],
    )
    payload["schema"] = 5

    with pytest.raises(RuntimeError, match="FULL semantic domain"):
        CalibrationSurface.from_payload(
            payload,
            policy_fingerprint="policy",
            configured_k=3,
            max_num_seqs=4,
            max_num_batched_tokens=16,
        )


def test_schema5_surface_rejects_missing_full_alias_domain_member():
    payload = _surface_payload(
        required_step_keys=[
            [0, 3, 4, 16, 4],
            [1, 3, 1, 1, 1],
            [1, 3, 2, 2, 1],
            [1, 3, 4, 4, 1],
        ],
        restore_step_keys=[[0, 3, 4, 16, 4], [1, 3, 4, 4, 1]],
    )
    payload["schema"] = 5

    with pytest.raises(RuntimeError, match="FULL semantic domain"):
        CalibrationSurface.from_payload(
            payload,
            policy_fingerprint="policy",
            configured_k=3,
            max_num_seqs=4,
            max_num_batched_tokens=16,
        )


def test_schema6_surface_does_not_silently_rewrite_full_x():
    payload = _surface_payload(
        required_step_keys=[[0, 3, 4, 16, 4], [1, 3, 3, 3, 1]],
        restore_step_keys=[[0, 3, 4, 16, 4], [1, 3, 3, 3, 1]],
    )
    payload["schema"] = 6

    surface = CalibrationSurface.from_payload(
        payload,
        policy_fingerprint="policy",
        configured_k=3,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )

    assert (1, 3, 3, 3, 1) in surface.required
    assert surface.migration_contract is None


def test_calibration_surface_accepts_k3_decode_k0_prefill_contract():
    payload = _surface_payload(
        required_step_keys=[[0, 0, 4, 16, 0], [1, 3, 4, 4, 1]],
        restore_step_keys=[[0, 0, 4, 16, 0], [1, 3, 4, 4, 1]],
    )

    surface = CalibrationSurface.from_payload(
        payload,
        policy_fingerprint="policy",
        configured_k=3,
        prefill_k=0,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )

    assert all(key[1] == 0 for key in surface.required if key[0] == 0)
    assert all(key[1] == 3 for key in surface.required if key[0] == 1)
    assert surface.semantic_token_witnesses == (
        ((0, 0, 1, 1, 0), 1),
        ((0, 0, 1, 2, 0), 2),
        ((0, 0, 1, 4, 0), 3),
        ((0, 0, 2, 2, 0), 2),
    )


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


def test_calibration_surface_rejects_stale_semantic_witnesses():
    with pytest.raises(RuntimeError, match="semantic witnesses differ"):
        CalibrationSurface.from_payload(
            _surface_payload(
                semantic_token_witnesses=[
                    {"step_key": [0, 3, 1, 4, 0], "live_num_tokens": 4}
                ]
            ),
            policy_fingerprint="policy",
            configured_k=3,
            max_num_seqs=4,
            max_num_batched_tokens=16,
        )


@pytest.mark.parametrize("live_tokens", [1, 5])
def test_current_catalog_rejects_noncanonical_semantic_witness(
    tmp_path, live_tokens: int
) -> None:
    path = _source(tmp_path, finalized_offline=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("migrated_from")
    payload.pop("migration_scope")
    payload["coverage"]["required_step_keys"] = [[0, 3, 2, 4, 0]]
    payload["coverage"]["restore_step_keys"] = [[0, 3, 2, 4, 0]]
    payload["coverage"]["semantic_witness_contract"] = (
        "live-current-to-physical-piecewise-v1"
    )
    payload["coverage"]["semantic_token_witnesses"] = [
        {"step_key": [0, 3, 2, 4, 0], "live_num_tokens": live_tokens}
    ]
    payload["shapes"] = [{"step_key": [0, 3, 2, 4, 0]}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="malformed semantic witnesses"):
        load_sealed_catalog(path, require_migration=False)


def test_current_catalog_structure_accepts_capped_non_power_of_two_boundary():
    from vllm.v1.core.elastic_catalog import _validate_catalog_structure

    key = [0, 3, 1, 3, 0]
    payload = {
        "schema": 6,
        "complete_shapes": 1,
        "coverage": {
            "required_step_keys": [key],
            "restore_step_keys": [key],
            "required_shapes": 1,
            "semantic_witness_contract": "live-current-to-physical-piecewise-v1",
            "semantic_token_witnesses": [{"step_key": key, "live_num_tokens": 3}],
        },
        "shapes": [{"step_key": key}],
    }

    _validate_catalog_structure(payload)


def test_publish_measured_catalog_is_atomic_and_complete(tmp_path, monkeypatch):
    from vllm.v1.core import elastic_graph, elastic_price_identity
    from vllm.v1.worker import startup_plan

    # This fixture tests atomic publication/coverage with a synthetic policy;
    # descriptor binding is exercised through the real loader and identity tests.
    monkeypatch.setattr(
        startup_plan, "elastic_catalog_owner_generation", lambda *_: "source"
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_price_identity",
        lambda *_: elastic_price_identity.price_identity({"fixture": "atomic"}),
    )
    monkeypatch.setattr(
        elastic_price_identity,
        "remap_catalog_resident_keys",
        lambda rows, **_kwargs: rows,
    )

    policy = SimpleNamespace(
        fingerprint="policy",
        verifier_contract="verifier",
        verifier_configuration="verifier-config",
        math_contract="math",
        mode_for=lambda _owner, _query_len: "PIECEWISE",
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
    key = (0, 3, 1, 1, 0)
    full_key = (1, 3, 1, 1, 1)
    verification_key = (0, 3, 1, 4, 4)
    row = {
        "cold_peak_bytes": 1,
        "hot_peak_bytes": 1,
        "resident_bytes": 1,
        "floor_bytes": 0,
        "resident_key_bytes": (("a" * 64, 1),),
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 2,
        "hot_stable_replays": 1,
    }
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
        num_speculative_tokens=3,
        speculative_config=None,
    )

    destination = publish_measured_catalog(
        config,
        object(),
        {key: row, full_key: row, verification_key: row},
        required=(key, full_key, verification_key),
        restore=(key, full_key),
        semantic_token_witnesses=((key, 1),),
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
        calibration_wall_seconds=1.0,
        output_root=tmp_path,
        source_surface_schema=5,
        source_surface_required_shapes=3,
        surface_migration_contract="schema5-full-owner-evidence-and-bounded-alias-v1",
        surface_full_aliases=((full_key, full_key),),
        migrated_source_keys=(key, full_key, verification_key),
        source_surface_sha256="b" * 64,
        owner_evidence_keys=(full_key,),
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert load_sealed_catalog(destination, require_migration=False) == payload
    assert payload["sealed"] is True
    assert payload["finalized_offline"] is True
    assert payload["complete_shapes"] == 3
    assert payload["coverage"]["serving_carrier_step_keys"] == [
        list(key),
        list(verification_key),
        list(full_key),
    ]
    assert payload["coverage"]["serving_carrier_contract"] == (
        "retained-terminal-mtp-no-cold-serving-v1"
    )
    assert payload["coverage"]["serving_carrier_owner"] == "mtp_decode"
    assert all(
        shape["resident_key_bytes"] == [["a" * 64, 1]] for shape in payload["shapes"]
    )
    assert payload["coverage"]["semantic_token_witnesses"] == [
        {"step_key": [0, 3, 1, 1, 0], "live_num_tokens": 1}
    ]
    assert payload["coverage"]["mixed_query_witnesses"] == []
    assert (
        payload["coverage"]["mixed_query_witness_contract"]
        == "shape-specific-query-distribution-v1"
    )
    assert payload["coverage"]["source_surface_schema"] == 5
    assert payload["coverage"]["surface_lineage_contract"] == "typed-source-serving-v1"
    assert payload["coverage"]["source_surface_required_shapes"] == 3
    assert payload["coverage"]["source_surface_sha256"] == "b" * 64
    assert payload["coverage"]["migrated_source_required_shapes"] == 3
    assert payload["coverage"]["migrated_source_step_keys"] == [
        list(key),
        list(full_key),
        list(verification_key),
    ]
    assert payload["coverage"]["schema6_added_witness_shapes"] == 0
    assert (
        payload["coverage"]["surface_migration_contract"]
        == "schema5-full-owner-evidence-and-bounded-alias-v1"
    )
    assert payload["coverage"]["surface_full_aliases"] == [
        {
            "semantic_step_key": [1, 3, 1, 1, 1],
            "physical_step_key": [1, 3, 1, 1, 1],
        }
    ]
    assert payload["coverage"]["owner_evidence_step_keys"] == [list(full_key)]
    assert (
        payload["coverage"]["semantic_witness_contract"]
        == "live-current-to-physical-piecewise-v1"
    )
    assert (
        payload["coverage"]["execution_manifest_schema"]
        == elastic_graph.EXECUTION_MANIFEST_SCHEMA
    )
    assert payload["coverage"]["dispatch_representations"] == [
        "hot_graph",
        "compiled_only",
        "inactive",
        "forbidden",
    ]
    assert (
        payload["coverage"]["residency_intent_contract"]
        == "current-dispatch-hot-successor-union-v1"
    )
    invalid_lineage_mutations = {
        "missing-inventory": lambda coverage: coverage.pop("migrated_source_step_keys"),
        "wrong-alias": lambda coverage: coverage["surface_full_aliases"][0].update(
            {"physical_step_key": [1, 3, 2, 2, 1]}
        ),
        "missing-evidence": lambda coverage: coverage.update(
            {"owner_evidence_step_keys": []}
        ),
        "malformed-sha": lambda coverage: coverage.update(
            {"source_surface_sha256": "B" * 64}
        ),
    }
    for name, mutate in invalid_lineage_mutations.items():
        invalid_payload = json.loads(json.dumps(payload))
        mutate(invalid_payload["coverage"])
        invalid_path = tmp_path / f"invalid-lineage-{name}.json"
        invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="lineage|source inventory"):
            load_sealed_catalog(invalid_path, require_migration=False)
    invalid_provenance = dict(row)
    invalid_provenance["resident_key_bytes"] = (("a" * 64, 2),)
    with pytest.raises(RuntimeError, match="exceeds its measured endpoint"):
        publish_measured_catalog(
            config,
            object(),
            {key: invalid_provenance},
            required=(key,),
            restore=(key,),
            semantic_token_witnesses=((key, 1),),
            decode_max_x=1,
            mixed_max_x=1,
            full_context_max_x=1,
            calibration_wall_seconds=1.0,
            output_root=tmp_path,
        )
    for invalid_identity in ("A" * 64, "a" * 63, "g" * 64):
        invalid_provenance = dict(row)
        invalid_provenance["resident_key_bytes"] = ((invalid_identity, 1),)
        with pytest.raises(RuntimeError, match="lowercase SHA256"):
            publish_measured_catalog(
                config,
                object(),
                {key: invalid_provenance},
                required=(key,),
                restore=(key,),
                semantic_token_witnesses=((key, 1),),
                decode_max_x=1,
                mixed_max_x=1,
                full_context_max_x=1,
                calibration_wall_seconds=1.0,
                output_root=tmp_path,
            )
    for required, restore in (
        ((key, key), (key,)),
        ((key,), (key, key)),
    ):
        with pytest.raises(RuntimeError, match="duplicate"):
            publish_measured_catalog(
                config,
                object(),
                {key: row},
                required=required,
                restore=restore,
                semantic_token_witnesses=((key, 1),),
                decode_max_x=1,
                mixed_max_x=1,
                full_context_max_x=1,
                calibration_wall_seconds=1.0,
                output_root=tmp_path,
            )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_measured_catalog(
            config,
            object(),
            {key: row, full_key: row, verification_key: row},
            required=(key, full_key, verification_key),
            restore=(key, full_key),
            semantic_token_witnesses=((key, 1),),
            decode_max_x=1,
            mixed_max_x=1,
            full_context_max_x=1,
            calibration_wall_seconds=1.0,
            output_root=tmp_path,
        )


def test_publish_measured_catalog_rejects_incomplete_witnesses_before_write(
    tmp_path, monkeypatch
):
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
        lambda _config, _kv: "2123456789abcdef",
    )
    monkeypatch.setattr(
        elastic_graph, "configured_compiled_piecewise_sizes", lambda _config: set()
    )
    key = (0, 3, 1, 1, 0)
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
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
        num_speculative_tokens=3,
        speculative_config=None,
    )
    destination = (
        tmp_path
        / "elastic_graph_catalog"
        / "elastic_graph_catalog_2123456789abcdef.json"
    )

    with pytest.raises(RuntimeError, match="witnesses differ"):
        publish_measured_catalog(
            config,
            object(),
            {key: row},
            required=(key,),
            restore=(key,),
            semantic_token_witnesses=(),
            decode_max_x=1,
            mixed_max_x=1,
            full_context_max_x=1,
            calibration_wall_seconds=1.0,
            output_root=tmp_path,
        )

    assert not destination.exists()


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


def test_calibration_checkpoint_round_trip_preserves_complete_evidence():
    key = (0, 3, 2, 8, 0)
    witness = (key, 4)
    surface = CalibrationSurface(
        required=(key,),
        restore=(key,),
        decode_max_x=2,
        mixed_max_x=2,
        full_context_max_x=1,
        mixed_query_witnesses=(witness,),
    )
    row = {**_complete_row(), "resident_key_bytes": (("target", 1),)}
    payload = calibration_checkpoint_payload(
        catalog={key: row},
        mixed_query_witnesses={witness},
        fingerprint="0123456789abcdef",
        surface_sha256="a" * 64,
        process_epochs=2,
        calibration_wall_seconds=12.5,
    )
    parsed = parse_calibration_checkpoint(
        json.loads(json.dumps(payload)),
        surface=surface,
        fingerprint="0123456789abcdef",
        surface_sha256="a" * 64,
    )

    assert parsed.catalog == {key: row}
    assert parsed.mixed_query_witnesses == frozenset({witness})
    assert parsed.process_epochs == 2
    assert parsed.wall_seconds == 12.5


def test_calibration_checkpoint_preserves_consistent_partial_row():
    key = (0, 3, 1, 2, 0)
    surface = CalibrationSurface(
        required=(key,),
        restore=(key,),
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
    )
    row = {
        **_complete_row(),
        "cold_observations": 1,
        "cold_stable_replays": 0,
        "hot_observations": 0,
        "hot_stable_replays": 0,
        "hot_peak_bytes": 0,
    }
    payload = calibration_checkpoint_payload(
        catalog={key: row},
        mixed_query_witnesses=set(),
        fingerprint="0123456789abcdef",
        surface_sha256="a" * 64,
        process_epochs=1,
        calibration_wall_seconds=1.0,
    )

    parsed = parse_calibration_checkpoint(
        payload,
        surface=surface,
        fingerprint="0123456789abcdef",
        surface_sha256="a" * 64,
    )

    assert parsed.catalog == {key: row}
    assert payload["completed_shapes"] == 0
    assert payload["observed_shapes"] == 1


def test_calibration_checkpoint_rejects_inconsistent_partial_row():
    key = (0, 3, 1, 2, 0)
    surface = CalibrationSurface(
        required=(key,),
        restore=(key,),
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
    )
    payload = calibration_checkpoint_payload(
        catalog={key: {**_complete_row(), "cold_observations": 1}},
        mixed_query_witnesses=set(),
        fingerprint="0123456789abcdef",
        surface_sha256="a" * 64,
        process_epochs=1,
        calibration_wall_seconds=1.0,
    )

    with pytest.raises(RuntimeError, match="observations are inconsistent"):
        parse_calibration_checkpoint(
            payload,
            surface=surface,
            fingerprint="0123456789abcdef",
            surface_sha256="a" * 64,
        )


def test_calibrator_resumes_only_missing_rows_after_bounded_process_epoch(
    monkeypatch,
):
    keys = ((0, 3, 1, 2, 0), (0, 3, 2, 4, 0))
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
        _resolve_elastic_step_physical_keys=lambda _key: ("physical",),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
    )
    calls = []

    def balanced(*, k, x, m):
        key = (0, k, x, m, 0)
        scheduler._elastic_graph_catalog[key] = _complete_row()
        calls.append(key)
        return key, x

    surface = CalibrationSurface(
        required=keys,
        restore=keys,
        decode_max_x=2,
        mixed_max_x=2,
        full_context_max_x=1,
    )
    saved = {}
    first = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(first, "_validate_surface_before_mutation", lambda _s: None)
    monkeypatch.setattr(first, "_balanced_prefill_pair", balanced)

    with pytest.raises(ElasticCalibrationRestartRequired):
        first.calibrate(
            surface,
            checkpoint_callback=lambda catalog, witnesses: saved.update(
                catalog=catalog, witnesses=set(witnesses)
            ),
            max_new_rows_per_process=1,
        )

    assert calls == [keys[0]]
    checkpoint = CalibrationCheckpoint(saved["catalog"], frozenset())
    second = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(second, "_validate_surface_before_mutation", lambda _s: None)
    monkeypatch.setattr(second, "_balanced_prefill_pair", balanced)

    assert set(second.calibrate(surface, checkpoint=checkpoint)) == set(keys)
    assert calls == [keys[0], keys[1]]


def test_calibrator_resumes_partial_row_in_fresh_producer_epoch(monkeypatch):
    key = (0, 3, 1, 2, 0)
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
        _resolve_elastic_step_physical_keys=lambda _key: ("physical",),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
    )
    calls = []

    def balanced(*, k, x, m):
        calls.append((k, x, m))
        scheduler._elastic_graph_catalog[key] = (
            {
                **_complete_row(),
                "cold_observations": 1,
                "cold_stable_replays": 0,
                "hot_observations": 0,
                "hot_stable_replays": 0,
                "hot_peak_bytes": 0,
            }
            if len(calls) == 1
            else _complete_row()
        )
        return key, x

    surface = CalibrationSurface(
        required=(key,),
        restore=(key,),
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
    )
    saved = {}
    first = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(first, "_validate_surface_before_mutation", lambda _s: None)
    monkeypatch.setattr(first, "_balanced_prefill_pair", balanced)

    with pytest.raises(ElasticCalibrationRestartRequired, match="producer_epochs=1"):
        first.calibrate(
            surface,
            checkpoint_callback=lambda catalog, witnesses: saved.update(
                catalog=catalog, witnesses=set(witnesses)
            ),
            max_producer_epochs_per_process=1,
        )

    assert saved["catalog"][key]["cold_observations"] == 1
    second = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(second, "_validate_surface_before_mutation", lambda _s: None)
    monkeypatch.setattr(second, "_balanced_prefill_pair", balanced)
    measured = second.calibrate(
        surface,
        checkpoint=CalibrationCheckpoint(saved["catalog"], frozenset()),
        checkpoint_callback=lambda _catalog, _witnesses: None,
        max_producer_epochs_per_process=1,
    )

    assert measured[key] == _complete_row()
    assert calls == [(3, 1, 2), (3, 1, 2)]


def test_calibrator_does_not_checkpoint_contracted_producer(monkeypatch):
    key = (0, 3, 2, 4, 0)
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
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
        calibrator, "_validate_surface_before_mutation", lambda _s: None
    )

    def contracted(**_kwargs):
        scheduler._elastic_graph_catalog[key] = {
            **_complete_row(),
            "cold_observations": 1,
            "cold_stable_replays": 0,
        }
        return None, 1

    monkeypatch.setattr(calibrator, "_balanced_prefill_pair", contracted)
    checkpoint = Mock()
    surface = CalibrationSurface(
        required=(key,),
        restore=(key,),
        decode_max_x=2,
        mixed_max_x=2,
        full_context_max_x=1,
    )

    with pytest.raises(RuntimeError, match="contracted accepted surface"):
        calibrator.calibrate(
            surface,
            checkpoint_callback=checkpoint,
            max_producer_epochs_per_process=1,
        )

    checkpoint.assert_not_called()


def test_calibrator_rejects_producer_limit_without_checkpoint_sink():
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
    )
    calibrator = ElasticCatalogCalibrator(
        SimpleNamespace(
            scheduler=scheduler,
            is_pooling_model=False,
            async_scheduling=False,
        )
    )
    surface = CalibrationSurface(
        required=((0, 3, 1, 2, 0),),
        restore=((0, 3, 1, 2, 0),),
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
    )

    with pytest.raises(ValueError, match="requires a checkpoint callback"):
        calibrator.calibrate(surface, max_producer_epochs_per_process=1)


def test_calibrator_resume_does_not_repeat_mixed_semantic_witness(monkeypatch):
    key = (0, 3, 4, 16, 0)
    witness = (key, 4)
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
        _resolve_elastic_step_physical_keys=lambda _key: ("physical",),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
    )
    surface = CalibrationSurface(
        required=(key,),
        restore=(key,),
        decode_max_x=4,
        mixed_max_x=4,
        full_context_max_x=1,
        mixed_query_witnesses=(witness,),
    )
    calls = []
    saved_witnesses = set()

    def mixed(**kwargs):
        calls.append(kwargs["query_len"])
        return key, 4

    first = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(first, "_validate_surface_before_mutation", lambda _s: None)
    monkeypatch.setattr(first, "_mixed_prefill_pair", mixed)
    checkpoint = CalibrationCheckpoint({key: _complete_row()}, frozenset())

    first.calibrate(
        surface,
        checkpoint=checkpoint,
        checkpoint_callback=lambda _catalog, witnesses: saved_witnesses.update(
            witnesses
        ),
    )
    assert calls == [4]
    assert saved_witnesses == {witness}

    second = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(second, "_validate_surface_before_mutation", lambda _s: None)
    monkeypatch.setattr(second, "_mixed_prefill_pair", mixed)
    second.calibrate(
        surface,
        checkpoint=CalibrationCheckpoint(
            {key: _complete_row()}, frozenset(saved_witnesses)
        ),
    )
    assert calls == [4]


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
        _elastic_restore_prefill_prompt_len=lambda: 2,
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
        scheduler._elastic_graph_catalog[key] = _complete_row()
        if kwargs["x"] == 4 and kwargs["query_len"] == 1:
            scheduler._elastic_graph_catalog[(0, 3, 2, 8, 4)] = _complete_row()
        calls.append(("full", key))
        return key, kwargs["x"]

    owner._run_elastic_full_restore_wave = full
    calibrator = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(
        calibrator, "_validate_surface_before_mutation", lambda _s: None
    )

    def balanced(*, k, x, m):
        physical_m = 1 << (m - 1).bit_length()
        key = (0, k, x, physical_m, 0)
        scheduler._elastic_graph_catalog[key] = _complete_row()
        calls.append(("balanced", key))
        return key, x

    def mixed(*, k, x, m, query_len):
        physical_m = 1 << (m - 1).bit_length()
        key = (0, k, x, physical_m, 0)
        scheduler._elastic_graph_catalog[key] = _complete_row()
        calls.append((f"mixed-q{query_len}", key))
        return key, x

    monkeypatch.setattr(calibrator, "_balanced_prefill_pair", balanced)
    monkeypatch.setattr(calibrator, "_mixed_prefill_pair", mixed)
    surface = CalibrationSurface(
        required=(
            (0, 3, 1, 2, 0),
            (0, 3, 2, 8, 4),
            (0, 3, 4, 16, 0),
            (1, 3, 4, 4, 1),
        ),
        restore=((0, 3, 2, 8, 4), (1, 3, 4, 4, 1)),
        decode_max_x=4,
        mixed_max_x=4,
        full_context_max_x=1,
        semantic_token_witnesses=(((0, 3, 4, 16, 0), 15),),
        mixed_query_witnesses=(
            ((0, 3, 4, 16, 0), 1),
            ((0, 3, 4, 16, 0), 4),
        ),
    )

    measured = calibrator.calibrate(surface)

    assert set(measured) == set(surface.required)
    assert ("full", (1, 3, 4, 4, 1)) in calls
    assert ("balanced", (0, 3, 1, 2, 0)) in calls
    assert ("balanced", (0, 3, 4, 16, 0)) not in calls
    assert ("mixed-q1", (0, 3, 4, 16, 0)) in calls
    assert ("mixed-q4", (0, 3, 4, 16, 0)) in calls
    assert calls.index(("mixed-q1", (0, 3, 4, 16, 0))) < calls.index(
        ("mixed-q4", (0, 3, 4, 16, 0))
    )
    assert calls.index(("balanced", (0, 3, 1, 2, 0))) < calls.index(
        ("mixed-q1", (0, 3, 4, 16, 0))
    )
    assert scheduler._elastic_restore_mode is False
    assert scheduler._elastic_graph_catalog == {original_key: original_row}
    assert scheduler._elastic_graph_catalog is catalog
    assert scheduler._elastic_graph_catalog_coverage is original_coverage


def test_calibrator_measures_restore_owner_only_through_full_lifecycle(monkeypatch):
    piecewise = (0, 3, 38, 128, 0)
    full = (1, 3, 38, 38, 1)
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
        _resolve_elastic_step_physical_keys=lambda _key: ("physical",),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
        _elastic_restore_prefill_prompt_len=lambda: 2,
    )
    calibrator = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(
        calibrator, "_validate_surface_before_mutation", lambda _surface: None
    )

    def complete_prefill(*, k, x, m):
        assert (k, x, m) == (3, 38, 76)
        scheduler._elastic_graph_catalog[piecewise] = _complete_row()
        return piecewise, 38

    balanced = Mock(side_effect=complete_prefill)
    monkeypatch.setattr(calibrator, "_balanced_prefill_pair", balanced)

    restore_calls = 0

    def restore(**kwargs):
        nonlocal restore_calls
        restore_calls += 1
        assert kwargs["x"] == 38
        assert kwargs["query_len"] == 1
        scheduler._elastic_graph_catalog[full] = (
            _complete_row()
            if restore_calls == 2
            else {
                "cold_peak_bytes": 1,
                "cold_observations": 1,
                "cold_stable_replays": 0,
            }
        )
        return full, 38

    owner._run_elastic_full_restore_wave = Mock(side_effect=restore)
    surface = CalibrationSurface(
        required=(piecewise, full),
        restore=(piecewise, full),
        decode_max_x=38,
        mixed_max_x=38,
        full_context_max_x=1,
    )

    assert set(calibrator.calibrate(surface)) == {piecewise, full}
    assert owner._run_elastic_full_restore_wave.call_count == 2
    balanced.assert_called_once_with(k=3, x=38, m=76)


def test_calibrator_checkpoints_partial_restore_family_before_restart(monkeypatch):
    piecewise = (0, 3, 38, 128, 0)
    full = (1, 3, 38, 38, 1)
    later = (0, 3, 1, 2, 0)
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
        _resolve_elastic_step_physical_keys=lambda _key: ("physical",),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
        _elastic_restore_prefill_prompt_len=lambda: 2,
    )
    calibrator = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(
        calibrator, "_validate_surface_before_mutation", lambda _surface: None
    )

    def balanced(*, k, x, m):
        key = piecewise if (k, x, m) == (3, 38, 76) else (0, k, x, m, 0)
        scheduler._elastic_graph_catalog[key] = _complete_row()
        return key, x

    balanced_mock = Mock(side_effect=balanced)
    monkeypatch.setattr(calibrator, "_balanced_prefill_pair", balanced_mock)

    restore_calls = 0

    def restore(**_kwargs):
        nonlocal restore_calls
        restore_calls += 1
        scheduler._elastic_graph_catalog[full] = (
            _complete_row()
            if restore_calls == 2
            else {
                **_complete_row(),
                "cold_observations": 1,
                "cold_stable_replays": 0,
                "hot_observations": 0,
                "hot_stable_replays": 0,
                "hot_peak_bytes": 0,
            }
        )
        return full, 38

    owner._run_elastic_full_restore_wave = Mock(side_effect=restore)
    surface = CalibrationSurface(
        required=(piecewise, full, later),
        restore=(piecewise, full),
        decode_max_x=38,
        mixed_max_x=38,
        full_context_max_x=1,
    )
    saved = {}

    with pytest.raises(ElasticCalibrationRestartRequired):
        calibrator.calibrate(
            surface,
            checkpoint_callback=lambda catalog, witnesses: saved.update(
                catalog=catalog, witnesses=set(witnesses)
            ),
            max_new_rows_per_process=1,
        )

    assert set(saved["catalog"]) == {piecewise, full}
    assert saved["catalog"][full]["cold_observations"] == 1
    assert owner._run_elastic_full_restore_wave.call_count == 1
    balanced_mock.assert_called_once_with(k=3, x=38, m=76)

    resumed = ElasticCatalogCalibrator(owner)
    monkeypatch.setattr(
        resumed, "_validate_surface_before_mutation", lambda _surface: None
    )
    monkeypatch.setattr(resumed, "_balanced_prefill_pair", balanced_mock)
    resumed_saved = {}
    with pytest.raises(ElasticCalibrationRestartRequired):
        resumed.calibrate(
            surface,
            checkpoint=CalibrationCheckpoint(saved["catalog"], frozenset()),
            checkpoint_callback=lambda catalog, witnesses: resumed_saved.update(
                catalog=catalog, witnesses=set(witnesses)
            ),
            max_new_rows_per_process=1,
        )

    assert set(resumed_saved["catalog"]) == {piecewise, full}
    assert resumed_saved["catalog"][full] == _complete_row()
    assert owner._run_elastic_full_restore_wave.call_count == 2
    balanced_mock.assert_called_once_with(k=3, x=38, m=76)


def test_calibrator_executes_live_m3_against_physical_m4(monkeypatch):
    key = (0, 3, 1, 4, 0)
    catalog = {}
    scheduler = SimpleNamespace(
        _elastic_graph_catalog=catalog,
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage={},
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
        calibrator, "_validate_surface_before_mutation", lambda _s: None
    )
    observed = []

    def balanced(*, k, x, m):
        observed.append((k, x, m))
        scheduler._elastic_graph_catalog[key] = _complete_row()
        return key, x

    monkeypatch.setattr(calibrator, "_balanced_prefill_pair", balanced)
    surface = CalibrationSurface(
        required=(key,),
        restore=(key,),
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
        semantic_token_witnesses=((key, 3),),
    )

    assert set(calibrator.calibrate(surface)) == {key}
    assert observed == [(3, 1, 3)]


def test_calibrator_installs_surface_decode_boundary_before_validation(monkeypatch):
    key = (0, 3, 39, 156, 4)
    original_inventory = {47: ("raw-cap",)}
    original_coverage = {"representation": "original", "decode_max_x": 47}
    scheduler = SimpleNamespace(
        _elastic_graph_catalog={key: _complete_row()},
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage=original_coverage,
        _elastic_short_decode_inventory=original_inventory,
        _resolve_elastic_step_physical_keys=lambda _key: ("physical",),
        num_spec_tokens=3,
    )

    def rebuild(max_x):
        scheduler._elastic_short_decode_inventory = {max_x: ("surface",)}

    scheduler._rebuild_elastic_short_decode_inventory = rebuild
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
    )
    calibrator = ElasticCatalogCalibrator(owner)

    def validate(surface):
        assert scheduler._elastic_restore_mode is True
        assert scheduler._elastic_graph_catalog_coverage["decode_max_x"] == 39
        assert scheduler._elastic_short_decode_inventory == {39: ("surface",)}
        assert surface.decode_max_x == 39

    monkeypatch.setattr(calibrator, "_validate_surface_before_mutation", validate)
    surface = CalibrationSurface(
        required=(key,),
        restore=(key,),
        decode_max_x=39,
        mixed_max_x=39,
        full_context_max_x=1,
    )

    assert set(calibrator.calibrate(surface)) == {key}
    assert scheduler._elastic_restore_mode is False
    assert scheduler._elastic_graph_catalog_coverage is original_coverage
    assert scheduler._elastic_short_decode_inventory is original_inventory


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
        calibrator, "_validate_surface_before_mutation", lambda _s: None
    )
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
    assert scheduler._elastic_graph_catalog is original_catalog
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


def test_calibrator_rejects_wrong_decode_identity_before_runtime_mutation():
    declared_full = (1, 3, 2, 4, 2)
    declared_prefill = (0, 3, 2, 4, 0)
    original_catalog = {("sentinel",): {"value": 1}}
    original_coverage = {"representation": "original"}
    scheduler = SimpleNamespace(
        _elastic_graph_catalog=original_catalog,
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage=original_coverage,
        _resolve_elastic_step_physical_keys=Mock(return_value=("physical",)),
        _canonical_elastic_graph_step_key=Mock(return_value=(1, 3, 2, 8, 4)),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
        _elastic_restore_wave_step_keys=Mock(),
        _begin_elastic_restore_physical_epoch=Mock(),
    )
    surface = CalibrationSurface(
        required=(declared_full, declared_prefill),
        restore=(declared_full, declared_prefill),
        decode_max_x=2,
        mixed_max_x=2,
        full_context_max_x=1,
    )

    with pytest.raises(RuntimeError, match="identity differs"):
        ElasticCatalogCalibrator(owner).calibrate(surface)

    assert scheduler._elastic_restore_mode is False
    assert scheduler._elastic_graph_catalog is original_catalog
    assert scheduler._elastic_graph_catalog_coverage is original_coverage
    owner._elastic_restore_wave_step_keys.assert_not_called()
    owner._begin_elastic_restore_physical_epoch.assert_not_called()


def test_calibrator_proves_each_schema5_semantic_full_alias_before_mutation():
    physical_full = (1, 3, 4, 4, 1)
    piecewise = (0, 3, 4, 16, 4)
    semantic_full = (1, 3, 3, 3, 1)
    original_catalog = {("sentinel",): {"value": 1}}
    original_coverage = {"representation": "original", "decode_max_x": 47}
    original_inventory = {47: ("raw",)}
    scheduler = SimpleNamespace(
        _elastic_graph_catalog=original_catalog,
        _elastic_restore_mode=False,
        _elastic_graph_catalog_coverage=original_coverage,
        _elastic_short_decode_inventory=original_inventory,
        _rebuild_elastic_short_decode_inventory=lambda _max_x: None,
        _resolve_elastic_step_physical_keys=Mock(return_value=("physical",)),
        _canonical_elastic_graph_step_key=Mock(return_value=(1, 3, 8, 8, 1)),
        num_spec_tokens=3,
    )
    owner = SimpleNamespace(
        scheduler=scheduler,
        is_pooling_model=False,
        async_scheduling=False,
        _elastic_restore_wave_step_keys=Mock(),
        _begin_elastic_restore_physical_epoch=Mock(),
    )
    surface = CalibrationSurface(
        required=(piecewise, physical_full),
        restore=(piecewise, physical_full),
        decode_max_x=4,
        mixed_max_x=4,
        full_context_max_x=1,
        source_schema=5,
        source_required_shapes=3,
        migration_contract="schema5-full-owner-evidence-and-bounded-alias-v1",
        full_aliases=((semantic_full, physical_full),),
    )

    with pytest.raises(RuntimeError, match="FULL alias differs"):
        ElasticCatalogCalibrator(owner).calibrate(surface)

    assert scheduler._elastic_graph_catalog is original_catalog
    assert scheduler._elastic_graph_catalog_coverage is original_coverage
    assert scheduler._elastic_short_decode_inventory is original_inventory
    owner._elastic_restore_wave_step_keys.assert_not_called()
    owner._begin_elastic_restore_physical_epoch.assert_not_called()


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
