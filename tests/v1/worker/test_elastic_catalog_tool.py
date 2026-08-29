# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from vllm.v1.core.elastic_catalog import ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION
from vllm.v1.worker.elastic_catalog_tool import finalize_catalog, rebind_catalog


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
