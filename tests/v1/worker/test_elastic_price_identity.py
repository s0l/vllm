# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.v1.core.elastic_graph import RuntimeGeneration, resolve_step_physical_keys
from vllm.v1.core.elastic_price_identity import (
    PLACEMENT_FACTORS,
    PRICE_OWNER_GENERATION,
    flashinfer_price_tactics,
    price_identity,
    price_identity_differences,
    remap_catalog_resident_keys,
    select_compatible_price_seed,
    source_semantic_digest,
    validate_calibration_request,
    validate_price_identity,
)


def test_flashinfer_tactics_bind_contents_not_location(tmp_path, monkeypatch):
    from vllm.model_executor.warmup import flashinfer_autotune_cache as cache

    config = SimpleNamespace(
        kernel_config=SimpleNamespace(enable_flashinfer_autotune=True)
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    selected = [first]

    def resolve(runner, *, create_parent):
        assert runner.vllm_config is config
        assert create_parent is False
        return selected[0]

    monkeypatch.setattr(cache, "resolve_flashinfer_autotune_file", resolve)
    missing = flashinfer_price_tactics(config)
    first.write_text('{"a": 1, "b": 2}')
    measured = flashinfer_price_tactics(config)
    assert measured != missing
    second.write_text('{"b": 2, "a": 1}')
    selected[0] = second
    assert flashinfer_price_tactics(config) == measured
    second.write_text('{"b": 3, "a": 1}')
    assert flashinfer_price_tactics(config) != measured
    second.write_text("corrupt")
    with pytest.raises(ValueError):
        flashinfer_price_tactics(config)
    config.kernel_config.enable_flashinfer_autotune = False
    assert flashinfer_price_tactics(config) == {"mode": "default"}


def test_native_price_binds_flashinfer_packaged_kernel_record(tmp_path, monkeypatch):
    from vllm.v1.core import elastic_price_identity as identity

    native = tmp_path / "native.so"
    native.write_bytes(b"fixed native control")
    records = {"flashinfer-jit-cache": "module.so,sha256=before,123"}

    def distribution(name):
        return SimpleNamespace(
            version="1.0",
            files=[Path("vllm/_C.so")] if name == "vllm" else [],
            locate_file=lambda entry: native,
            read_text=lambda entry: records.get(name, "unchanged"),
        )

    monkeypatch.setattr(identity.metadata, "distribution", distribution)
    monkeypatch.setattr(
        identity.util, "find_spec", lambda name: SimpleNamespace(origin=str(native))
    )
    identity.native_price_provenance.cache_clear()
    try:
        before = identity.native_price_provenance()
        assert "flashinfer-jit-cache" in before
        identity.native_price_provenance.cache_clear()
        assert identity.native_price_provenance() == before
        records["flashinfer-jit-cache"] = "module.so,sha256=after,123"
        identity.native_price_provenance.cache_clear()
        after = identity.native_price_provenance()
        assert after["flashinfer-jit-cache"] != before["flashinfer-jit-cache"]
        assert {key for key in before if before[key] != after[key]} == {
            "flashinfer-jit-cache"
        }
    finally:
        identity.native_price_provenance.cache_clear()


def test_price_identity_survives_json_policy_roundtrip():
    factors = {"graph_execution_policy": {"owners": ({"full_query_lens": (1,)},)}}
    live = price_identity(factors)
    stored = json.loads(json.dumps(live))
    assert live["fingerprint"] == stored["fingerprint"]
    assert price_identity_differences(stored, live) == ()
    assert live == stored
    changed = deepcopy(factors)
    changed["graph_execution_policy"]["owners"][0]["full_query_lens"] = (1, 4)
    assert price_identity_differences(live, price_identity(changed)) == (
        "graph_execution_policy",
    )


def test_price_identity_does_not_alias_mutable_input():
    factors = {"graph_execution_policy": {"owners": ["target"]}}
    identity = price_identity(factors)
    factors["graph_execution_policy"]["owners"].append("mtp_decode")
    assert identity["factors"]["graph_execution_policy"]["owners"] == ["target"]


def test_seed_preserves_only_matching_producers_and_witnesses():
    same, changed, removed = (1, 3, 1, 1, 1), (1, 3, 3, 3, 1), (1, 3, 4, 4, 1)
    rows = {key: {"cold_peak_bytes": 99} for key in (same, changed, removed)}
    coverage = {
        "surface_full_aliases": [
            {"semantic_step_key": list(changed), "physical_step_key": list(removed)},
        ]
    }
    surface = SimpleNamespace(
        required=(same, changed),
        full_aliases=(),
        semantic_token_witnesses=(),
        mixed_query_witnesses=(),
    )
    reused, witnesses = select_compatible_price_seed(rows, coverage, surface)
    assert reused == {same: {"cold_peak_bytes": 99}}
    assert not witnesses
    assert len(rows) == 3


def test_seed_does_not_inherit_changed_live_token_witness():
    key = (0, 3, 2, 8, 0)
    rows = {key: {"cold_peak_bytes": 99}}
    coverage = {
        "semantic_token_witnesses": [{"step_key": list(key), "live_num_tokens": 5}],
        "mixed_query_witnesses": [{"step_key": list(key), "query_len": 1}],
    }
    surface = SimpleNamespace(
        required=(key,),
        full_aliases=(),
        semantic_token_witnesses=((key, 6),),
        mixed_query_witnesses=((key, 1),),
    )
    assert select_compatible_price_seed(rows, coverage, surface) == ({}, set())
    surface.semantic_token_witnesses = ((key, 5),)
    assert select_compatible_price_seed(rows, coverage, surface) == (rows, {(key, 1)})


def test_maintenance_flag_does_not_replace_measurement_request(tmp_path, monkeypatch):
    from vllm.v1.engine.elastic_bootstrap import _require_measurement_request

    monkeypatch.setenv("AG2_VLLM_ELASTIC_AUTO_CALIBRATE", "1")
    monkeypatch.delenv("AG2_VLLM_ELASTIC_CALIBRATION_REQUEST", raising=False)
    with pytest.raises(RuntimeError, match="not authorized"):
        _require_measurement_request("a" * 16, "b" * 64)


def _migration_fixture(tmp_path):
    from tests.v1.worker.test_startup_plan import _write_catalog

    config, kv, _key, _row, source = _write_catalog(tmp_path)
    payload = json.loads(source.read_text())
    del payload["price_identity"]
    del payload["resident_owner_generation"]
    source.write_text(json.dumps(payload))
    evidence = tmp_path / "review.md"
    evidence.write_text("Synthetic same-envelope control; no live compatibility claim.")
    identity = price_identity({"fixture": "loader"})
    review = {
        "schema": "elastic-price-migration-review-v1",
        "contract": "same-physical-price-envelope-v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_generation": PRICE_OWNER_GENERATION,
        "destination_price_identity": identity,
        "max_num_batched_tokens": 4096,
        "evidence": {str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()},
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    return source, review_path, evidence, identity, config, kv


def test_reviewed_legacy_migration_loads_real_owners_without_remeasurement(
    tmp_path, monkeypatch
):
    import vllm.envs as envs
    from vllm.v1.worker import startup_plan
    from vllm.v1.worker.elastic_catalog_tool import migrate_price_catalog

    source, review, _evidence, identity, config, kv = _migration_fixture(tmp_path)
    original = source.read_bytes()
    destination = tmp_path / "version-2.json"
    migrate_price_catalog(source, destination, review)
    assert source.read_bytes() == original
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: identity["fingerprint"],
    )
    monkeypatch.setattr(
        startup_plan, "compute_elastic_graph_price_identity", lambda *_: identity
    )
    monkeypatch.setattr(
        startup_plan, "elastic_catalog_owner_generation", lambda *_: "new-process"
    )
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CATALOG_PATH", str(destination))
    loaded = startup_plan.load_elastic_graph_catalog(config, kv)
    assert loaded
    from tests.v1.worker.test_startup_plan import _resident_owners

    assert loaded[(1, 3, 1, 1, 1)]["resident_key_bytes"] == _resident_owners(
        "new-process"
    )
    assert loaded[(1, 3, 1, 1, 1)]["cold_peak_bytes"] == 320
    with pytest.raises(FileExistsError):
        migrate_price_catalog(source, destination, review)
    assert source.read_bytes() == original


@pytest.mark.parametrize("mutation", ["source", "evidence", "generation", "manifest"])
def test_migration_rejects_unproven_or_stale_inputs(tmp_path, mutation):
    from vllm.v1.worker.elastic_catalog_tool import migrate_price_catalog

    source, review, evidence, _identity, _config, _kv = _migration_fixture(tmp_path)
    payload = json.loads(review.read_text())
    if mutation == "source":
        payload["source_sha256"] = "0" * 64
    elif mutation == "evidence":
        evidence.write_text("changed")
    elif mutation == "generation":
        payload["source_generation"] = "foreign"
    else:
        payload["destination_price_identity"]["fingerprint"] = "0" * 16
    review.write_text(json.dumps(payload))
    destination = tmp_path / "must-not-exist.json"
    with pytest.raises(RuntimeError):
        migrate_price_catalog(source, destination, review)
    assert not destination.exists()


def test_measurement_request_file_recovery(tmp_path, monkeypatch):
    from vllm.v1.engine.elastic_bootstrap import _require_measurement_request

    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "schema": "elastic-measurement-request-v1",
                "fingerprint": "a" * 16,
                "surface_sha256": "b" * 64,
                "reason": "missing-coverage",
                "evidence": "bounded inventory review",
            }
        )
    )
    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_REQUEST", str(path))
    _require_measurement_request("a" * 16, "b" * 64)
    with pytest.raises(RuntimeError, match="identity-bound"):
        _require_measurement_request("c" * 16, "b" * 64)
    path.write_text("{broken")
    with pytest.raises(RuntimeError, match="not authorized"):
        _require_measurement_request("a" * 16, "b" * 64)


@pytest.mark.parametrize("name", sorted(PLACEMENT_FACTORS))
def test_placement_mutations_reuse_prices(name):
    assert price_identity({"producer": "a", name: 1}) == price_identity(
        {"producer": "a", name: 999}
    )


@pytest.mark.parametrize("name", ["kernel", "layout", "stride", "allocator", "policy"])
def test_physical_mutations_invalidate_only_named_component(name):
    before = price_identity({name: "a", "other": 12})
    after = price_identity({name: "b", "other": 12})
    assert before["fingerprint"] != after["fingerprint"]
    assert price_identity_differences(before, after) == (name,)


def test_comments_and_formatting_are_not_producer_changes():
    assert source_semantic_digest("x = f(1)\n") == source_semantic_digest(
        "# new comment\nx=f( 1 ) # another\n"
    )
    assert source_semantic_digest("x=f(1)") != source_semantic_digest("x=f(2)")
    assert source_semantic_digest("logger.info(f())") != source_semantic_digest(
        "logger.info(g())"
    )


@pytest.mark.parametrize("mutation", ["digest", "schema", "budget", "missing"])
def test_corrupt_manifest_cannot_authorize_reuse(mutation):
    payload = price_identity({"producer": "a"})
    if mutation == "digest":
        payload["fingerprint"] = "0" * 16
    elif mutation == "schema":
        payload["schema"] = "future"
    elif mutation == "budget":
        payload["factors"]["num_blocks"] = 1
    else:
        del payload["factors"]
    with pytest.raises(RuntimeError, match="invalid elastic price"):
        validate_price_identity(payload)


def _rows(generation="old"):
    key = (1, 3, 2, 2, 1)
    owners = resolve_step_physical_keys(key, RuntimeGeneration(generation), 4096)
    return {
        key: {
            "resident_bytes": 600,
            "resident_key_bytes": [
                (owner.identity, 100 * (index + 1))
                for index, owner in enumerate(owners)
            ],
        }
    }


def _remap(rows, source, destination):
    return remap_catalog_resident_keys(
        rows,
        source_generation=source,
        destination_generation=destination,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(),
        policy=None,
    )


def test_portable_prices_bind_new_epoch_without_losing_credit():
    original = _rows()
    saved = deepcopy(original)
    portable = _remap(original, "old", PRICE_OWNER_GENERATION)
    loaded = _remap(portable, PRICE_OWNER_GENERATION, "new")
    assert original == saved
    key = next(iter(loaded))
    assert dict(loaded[key]["resident_key_bytes"]) == dict(
        _rows("new")[key]["resident_key_bytes"]
    )
    assert loaded[key]["resident_bytes"] == 600
    recovered = _remap(loaded, "new", "old")
    assert dict(recovered[key]["resident_key_bytes"]) == dict(
        original[key]["resident_key_bytes"]
    )


@pytest.mark.parametrize(
    "mutation", ["unknown", "duplicate", "negative", "bool", "stale"]
)
def test_invalid_owner_provenance_fails_closed(mutation):
    rows = _rows()
    entries = next(iter(rows.values()))["resident_key_bytes"]
    identity, _ = entries[0]
    if mutation == "unknown":
        entries[0] = ("f" * 64, 100)
    elif mutation == "duplicate":
        entries.append(entries[0])
    elif mutation == "negative":
        entries[0] = (identity, -1)
    elif mutation == "bool":
        entries[0] = (identity, True)
    with pytest.raises(RuntimeError, match="unknown or invalid"):
        _remap(rows, "stale" if mutation == "stale" else "old", "new")


@pytest.mark.parametrize(
    "reason", ["initial-measurement", "changed-producer", "missing-coverage"]
)
def test_measurement_request_binds_exact_decision(reason):
    payload = {
        "schema": "elastic-measurement-request-v1",
        "fingerprint": "a" * 16,
        "surface_sha256": "b" * 64,
        "reason": reason,
        "evidence": "reviewed receipt",
    }
    validate_calibration_request(payload, fingerprint="a" * 16, surface_sha256="b" * 64)
    for key, value in [
        ("reason", "unknown-compatibility"),
        ("fingerprint", "c" * 16),
        ("surface_sha256", "d" * 64),
        ("evidence", ""),
    ]:
        with pytest.raises(RuntimeError, match="measurement request"):
            validate_calibration_request(
                {**payload, key: value}, fingerprint="a" * 16, surface_sha256="b" * 64
            )
