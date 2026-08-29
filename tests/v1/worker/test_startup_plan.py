# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.v1.core.elastic_graph import GraphExecutionPolicy, OwnerGraphExecutionPolicy
from vllm.v1.worker import startup_plan

pytestmark = pytest.mark.cpu_test


def _catalog_policy() -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="approved-q4-full-control-v1",
        math_contract="approved-q4-full-math-v1",
        owners=(
            OwnerGraphExecutionPolicy("mtp_decode", (1,), "PIECEWISE"),
            OwnerGraphExecutionPolicy("mtp_prefill", (), "PIECEWISE"),
            OwnerGraphExecutionPolicy("target", (1, 4), "PIECEWISE"),
        ),
    )


def _catalog_config() -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        additional_config={"elastic_compiled_piecewise_sizes": []},
    )


def _catalog_kv_config() -> SimpleNamespace:
    return SimpleNamespace(
        elastic_graph_execution_policy=_catalog_policy().to_payload()
    )


def test_cudagraph_recipe_persists_all_reachable_descriptors(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(startup_plan, "compute_plan_fingerprint", lambda *_: "fp")
    monkeypatch.setattr(
        startup_plan, "compute_elastic_runtime_generation", lambda *_: "generation"
    )

    full = SimpleNamespace(
        cg_mode=SimpleNamespace(name="FULL"),
        num_tokens=4,
        num_reqs=1,
        uniform_token_count=4,
        num_active_loras=0,
        physical_num_reqs=1,
        runtime_generation="generation",
    )
    piecewise = SimpleNamespace(
        cg_mode=SimpleNamespace(name="PIECEWISE"),
        num_tokens=8,
        num_reqs=None,
        uniform_token_count=None,
        num_active_loras=0,
        physical_num_reqs=8,
        runtime_generation="generation",
    )
    startup_plan.maybe_save_cudagraph_recipe(
        SimpleNamespace(),
        rank=0,
        world_size=3,
        owner="mtp/decode",
        descriptors=(piecewise, full),
    )

    path = tmp_path / "cudagraph_recipes" / "cudagraph_recipe_fp_mtp_decode.json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == startup_plan.GRAPH_RECIPE_SCHEMA_VERSION
    assert payload["fingerprint"] == "fp"
    assert payload["owner"] == "mtp/decode"
    assert payload["descriptors"] == [
        {
            "mode": "FULL",
            "num_tokens": 4,
            "num_reqs": 1,
            "uniform_token_count": 4,
            "num_active_loras": 0,
            "physical_num_reqs": 1,
            "runtime_generation": "generation",
        },
        {
            "mode": "PIECEWISE",
            "num_tokens": 8,
            "num_reqs": None,
            "uniform_token_count": None,
            "num_active_loras": 0,
            "physical_num_reqs": 8,
            "runtime_generation": "generation",
        },
    ]

    before = path.stat().st_mtime_ns
    startup_plan.maybe_save_cudagraph_recipe(
        SimpleNamespace(),
        rank=0,
        world_size=3,
        owner="mtp/decode",
        descriptors=(piecewise, full),
    )
    assert path.stat().st_mtime_ns == before

    loaded = startup_plan.load_cudagraph_recipe(
        SimpleNamespace(), rank=0, world_size=3, owner="mtp/decode"
    )
    assert loaded == payload["descriptors"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(schema=999),
        lambda payload: payload.update(fingerprint="stale"),
        lambda payload: payload.update(owner="other"),
        lambda payload: payload.update(descriptors={}),
        lambda payload: payload["descriptors"][0].update(mode="NONE"),
        lambda payload: payload["descriptors"][0].update(num_tokens=True),
        lambda payload: payload["descriptors"][0].update(num_reqs=0),
        lambda payload: payload["descriptors"][0].update(uniform_token_count=-1),
        lambda payload: payload["descriptors"][0].update(num_active_loras=-1),
        lambda payload: payload["descriptors"][0].update(extra=1),
    ],
)
def test_cudagraph_recipe_invalid_cache_fails_closed(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(startup_plan, "compute_plan_fingerprint", lambda *_: "fp")
    path = tmp_path / "cudagraph_recipes" / "cudagraph_recipe_fp_target.json"
    path.parent.mkdir()
    payload = {
        "schema": startup_plan.GRAPH_RECIPE_SCHEMA_VERSION,
        "fingerprint": "fp",
        "owner": "target",
        "descriptors": [
            {
                "mode": "FULL",
                "num_tokens": 4,
                "num_reqs": 1,
                "uniform_token_count": 4,
                "num_active_loras": 0,
            }
        ],
    }
    mutation(payload)
    path.write_text(json.dumps(payload))

    assert (
        startup_plan.load_cudagraph_recipe(
            SimpleNamespace(), rank=0, world_size=3, owner="target"
        )
        == []
    )


def test_cudagraph_recipe_missing_corrupt_and_disabled_are_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(startup_plan, "compute_plan_fingerprint", lambda *_: "fp")
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    assert (
        startup_plan.load_cudagraph_recipe(
            SimpleNamespace(), rank=0, world_size=3, owner="target"
        )
        == []
    )

    path = tmp_path / "cudagraph_recipes" / "cudagraph_recipe_fp_target.json"
    path.parent.mkdir()
    path.write_text("{")
    assert (
        startup_plan.load_cudagraph_recipe(
            SimpleNamespace(), rank=0, world_size=3, owner="target"
        )
        == []
    )

    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", False)
    assert (
        startup_plan.load_cudagraph_recipe(
            SimpleNamespace(), rank=0, world_size=3, owner="target"
        )
        == []
    )


def test_elastic_graph_catalog_loads_only_complete_cold_hot_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    complete = (1, 3, 12, 48, 4)
    incomplete = (0, 3, 12, 4096, 0)
    path = startup_plan.save_elastic_graph_catalog(
        config,
        kv_config,
        {
            complete: {
                "cold_peak_bytes": 320,
                "hot_peak_bytes": 192,
                "resident_bytes": 160,
                "floor_bytes": 16,
                "cold_observations": 2,
                "cold_stable_replays": 1,
                "hot_observations": 2,
                "hot_stable_replays": 1,
            },
            incomplete: {
                "cold_peak_bytes": 512,
                "hot_peak_bytes": 0,
                "resident_bytes": 256,
                "floor_bytes": 32,
                "cold_observations": 1,
                "cold_stable_replays": 0,
                "hot_observations": 0,
                "hot_stable_replays": 0,
            },
        },
        sealed=True,
    )

    assert path is not None
    payload = json.loads(
        (
            tmp_path
            / "elastic_graph_catalog"
            / ("elastic_graph_catalog_elastic-fp.json")
        ).read_text()
    )
    assert payload["complete_shapes"] == 1
    assert payload["sealed"] is True
    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {
        complete: {
            "cold_peak_bytes": 320,
            "hot_peak_bytes": 192,
            "resident_bytes": 160,
            "floor_bytes": 16,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
        }
    }


def test_elastic_graph_catalog_migration_requires_exact_reviewed_scope(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    source = tmp_path / "elastic_graph_catalog" / (
        "elastic_graph_catalog_fedcba9876543210.json"
    )
    source.parent.mkdir()
    source_policy = _catalog_policy()
    source_row = {
        "step_key": [0, 0, 40, 4096, 0],
        "cold_peak_bytes": 64,
        "hot_peak_bytes": 32,
        "resident_bytes": 32,
        "floor_bytes": 0,
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 2,
        "hot_stable_replays": 1,
    }
    source_row.update(
        startup_plan._catalog_policy_row_metadata(
            source_row["step_key"],
            policy=source_policy,
            max_num_batched_tokens=4096,
            compiled_piecewise_sizes=frozenset(),
        )
    )
    source.write_text(
        json.dumps(
            {
                "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
                "fingerprint": "fedcba9876543210",
                "sealed": True,
                "complete_shapes": 1,
                "graph_execution_policy": source_policy.to_payload(),
                "coverage": {
                    "representation": "bounded_exact_hotset",
                    "required_step_keys": [[0, 0, 40, 4096, 0]],
                    "required_shapes": 1,
                    "restore_step_keys": [],
                    "compiled_piecewise_sizes": [],
                    "graph_execution_policy_fingerprint": source_policy.fingerprint,
                    "verifier_contract": source_policy.verifier_contract,
                    "verifier_configuration": source_policy.verifier_configuration,
                    "math_contract": source_policy.math_contract,
                    "capture_state_abi": startup_plan.ELASTIC_CAPTURE_STATE_ABI,
                },
                "shapes": [source_row],
            }
        )
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", "fedcba9876543210"
    )

    vllm_config = SimpleNamespace(
        additional_config={"elastic_compiled_piecewise_sizes": [4096]},
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )
    with pytest.raises(RuntimeError, match="exact reviewed scope"):
        startup_plan.load_elastic_graph_catalog(
            vllm_config, _catalog_kv_config()
        )

    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE",
        startup_plan.ELASTIC_GRAPH_CATALOG_MIGRATION_SCOPE,
    )
    assert startup_plan.load_elastic_graph_catalog(
        vllm_config, _catalog_kv_config()
    ) == {}
    migrated = json.loads(
        (
            tmp_path
            / "elastic_graph_catalog"
            / "elastic_graph_catalog_0123456789abcdef.json"
        ).read_text()
    )
    assert migrated["migrated_from"] == "fedcba9876543210"
    assert (
        migrated["migration_scope"]
        == startup_plan.ELASTIC_GRAPH_CATALOG_MIGRATION_SCOPE
    )
    assert migrated["coverage"]["required_step_keys"] == []
    assert migrated["coverage"]["compiled_piecewise_sizes"] == [4096]
    assert migrated["shapes"] == []

    monkeypatch.delenv("AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE")
    with pytest.raises(RuntimeError, match="lacks the exact active lineage"):
        startup_plan.load_elastic_graph_catalog(
            vllm_config, _catalog_kv_config()
        )


def test_compiled_piecewise_migration_preserves_unrelated_rows_and_restore_pair():
    compiled = [0, 0, 40, 4096, 0]
    piecewise_restore = [0, 0, 1, 2, 0]
    full_restore = [1, 3, 1, 1, 1]
    other = [1, 3, 40, 40, 1]
    payload = {
        "coverage": {
            "required_step_keys": [piecewise_restore, compiled, full_restore, other],
            "required_shapes": 4,
            "restore_step_keys": [piecewise_restore, full_restore],
        },
        "shapes": [
            {"step_key": piecewise_restore, "marker": "piecewise-restore"},
            {"step_key": compiled, "marker": "removed"},
            {"step_key": full_restore, "marker": "full-restore"},
            {"step_key": other, "marker": "decode"},
        ],
        "complete_shapes": 4,
    }
    config = SimpleNamespace(
        additional_config={"elastic_compiled_piecewise_sizes": [4096]},
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )

    migrated = startup_plan._migrate_compiled_piecewise_catalog(payload, config)

    assert migrated["coverage"]["required_step_keys"] == [
        piecewise_restore,
        full_restore,
        other,
    ]
    assert migrated["coverage"]["restore_step_keys"] == [
        piecewise_restore,
        full_restore,
    ]
    assert migrated["coverage"]["excluded_graph_step_keys"] == [compiled]
    assert [row["marker"] for row in migrated["shapes"]] == [
        "piecewise-restore",
        "full-restore",
        "decode",
    ]
    assert migrated["coverage"]["required_shapes"] == 3
    assert migrated["complete_shapes"] == 3


def test_mtp_phase_continuity_catalog_migration_rebinds_restore_only():
    k0_restore = [0, 0, 1, 2, 0]
    k3_restore = [0, 3, 1, 2, 0]
    full_restore = [1, 3, 1, 1, 1]
    excluded_k0 = [0, 0, 40, 4096, 0]
    payload = {
        "coverage": {
            "representation": "bounded_exact_hotset",
            "compiled_piecewise_contract": "torch-compile-no-cudagraph-v1",
            "required_step_keys": [k0_restore, k3_restore, full_restore],
            "restore_step_keys": [k0_restore, full_restore],
            "excluded_graph_step_keys": [excluded_k0],
        },
        "shapes": [
            {"step_key": k0_restore},
            {"step_key": k3_restore},
            {"step_key": full_restore},
        ],
    }
    config = SimpleNamespace(
        additional_config={"elastic_compiled_piecewise_sizes": [4096]},
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            disable_speculation_on_non_decode=False,
        ),
    )

    migrated = startup_plan._migrate_mtp_phase_continuity_catalog(
        payload, config
    )

    assert migrated["coverage"]["restore_step_keys"] == [
        k3_restore,
        full_restore,
    ]
    assert migrated["coverage"]["compiled_piecewise_owners"] == [
        "mtp_prefill",
        "target",
    ]
    assert [0, 3, 40, 4096, 0] in migrated["coverage"][
        "excluded_graph_step_keys"
    ]
    assert migrated["shapes"] == payload["shapes"]


def test_mtp_batched_q1_migration_reuses_only_existing_piecewise_carriers(
    monkeypatch,
):
    q1 = [1, 3, 1, 1, 1]
    q4_x1 = [1, 3, 1, 4, 4]
    q4_x2 = [1, 3, 2, 8, 4]
    piecewise_x1 = [0, 3, 1, 4, 0]
    piecewise_x2 = [0, 3, 2, 8, 0]
    payload = {
        "coverage": {
            "representation": "bounded_exact_hotset",
            "decode_max_x": 2,
            "required_step_keys": [
                q1,
                q4_x1,
                q4_x2,
                piecewise_x1,
                piecewise_x2,
            ],
            "required_shapes": 5,
            "restore_step_keys": [q1],
            "excluded_graph_step_keys": [],
        },
        "shapes": [
            {"step_key": key, "marker": index}
            for index, key in enumerate(
                [q1, q4_x1, q4_x2, piecewise_x1, piecewise_x2]
            )
        ],
        "complete_shapes": 5,
    }
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3)
    )
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "1")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_PSEUDO_DECODE", "0")
    monkeypatch.setenv("AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH", "0")

    migrated = startup_plan._migrate_mtp_batched_q1_catalog(payload, config)

    assert migrated["coverage"]["required_step_keys"] == [
        q1,
        piecewise_x1,
        piecewise_x2,
    ]
    assert migrated["coverage"]["restore_step_keys"] == [q1]
    assert migrated["coverage"]["mtp_verifier_piecewise_step_keys"] == [
        piecewise_x1,
        piecewise_x2,
    ]
    assert migrated["coverage"]["mtp_verifier_workspace_bytes"] == 138412032
    assert migrated["coverage"]["excluded_graph_step_keys"] == [q4_x1, q4_x2]
    assert migrated["complete_shapes"] == 3


def test_policy_projection_reuses_piecewise_and_reports_rejected_full(
    monkeypatch,
):
    q4_full = [1, 3, 1, 4, 4]
    q4_piecewise = [0, 3, 1, 4, 0]
    canonical_q4_piecewise = [0, 3, 1, 4, 4]
    payload = {
        "sealed": True,
        "coverage": {
            "representation": "bounded_exact_hotset",
            "decode_max_x": 1,
            "required_step_keys": [q4_full, q4_piecewise],
            "required_shapes": 2,
            "restore_step_keys": [],
        },
        "shapes": [
            {"step_key": q4_full, "resident_bytes": 100},
            {"step_key": q4_piecewise, "resident_bytes": 20},
        ],
    }
    policy = GraphExecutionPolicy(
        verifier_contract="batched-causal-q1-v1",
        math_contract="pending-batched-q1-product-math-v1",
        owners=(
            OwnerGraphExecutionPolicy("mtp_decode", (1,), "PIECEWISE"),
            OwnerGraphExecutionPolicy("mtp_prefill", (), "PIECEWISE"),
            OwnerGraphExecutionPolicy("target", (1,), "PIECEWISE"),
        ),
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        additional_config={"elastic_compiled_piecewise_sizes": []},
    )
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "1")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_PSEUDO_DECODE", "0")
    monkeypatch.setenv("AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH", "0")

    migrated, report = startup_plan.project_elastic_graph_catalog_to_policy(
        payload, config, policy
    )

    assert [row["step_key"] for row in migrated["shapes"]] == [
        canonical_q4_piecewise
    ]
    assert report["retained_resident_bytes"] == 20
    assert report["rejected_resident_bytes"] == 100
    assert report["missing_required_keys"] == []
    assert report["rejected"] == [
        {
            "step_key": q4_full,
            "reason": "q4 FULL forbidden by active Graph execution policy",
        }
    ]
    assert migrated["sealed"] is True
    assert migrated["shapes"][0]["policy_fingerprint"] == policy.fingerprint


def test_policy_projection_rebinds_compiled_carrier_boundary(tmp_path, monkeypatch):
    source_key = [0, 3, 1, 64, 0]
    payload = {
        "sealed": True,
        "coverage": {
            "representation": "bounded_exact_hotset",
            "decode_max_x": 1,
            "mixed_max_x": 1,
            "full_context_max_x": 1,
            "pinned_full_entries": 0,
            "pinned_full_bytes": 0,
            "piecewise_replay_contract": "cold_capture_bounds_same_key_hot-v1",
            "restore_step_keys": [source_key],
            "required_step_keys": [source_key],
            "required_shapes": 1,
            "compiled_piecewise_sizes": [256, 512, 1024, 2048, 4096],
            "compiled_piecewise_contract": "torch-compile-no-cudagraph-v1",
            "compiled_piecewise_owners": ["mtp_prefill", "target"],
        },
        "shapes": [{"step_key": source_key, "resident_bytes": 20}],
    }
    compiled = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    policy = GraphExecutionPolicy(
        verifier_contract="batched-causal-q1-v1",
        math_contract="accepted-v1",
        owners=(
            OwnerGraphExecutionPolicy(
                "mtp_decode", (1,), "NONE", tuple(compiled)
            ),
            OwnerGraphExecutionPolicy(
                "mtp_prefill", (), "PIECEWISE", tuple(compiled)
            ),
            OwnerGraphExecutionPolicy(
                "target", (1,), "PIECEWISE", tuple(compiled)
            ),
        ),
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            disable_speculation_on_non_decode=False,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        additional_config={"elastic_compiled_piecewise_sizes": compiled},
    )

    migrated, _report = startup_plan.project_elastic_graph_catalog_to_policy(
        payload, config, policy
    )

    coverage = migrated["coverage"]
    assert coverage["compiled_piecewise_sizes"] == compiled
    assert coverage["compiled_piecewise_contract"] == (
        "torch-compile-no-cudagraph-v1"
    )
    assert coverage["compiled_piecewise_owners"] == ["mtp_prefill", "target"]

    # The migration report is not the acceptance oracle. Re-read the projected
    # artifact through the exact serving coverage validator so a stale policy
    # field cannot survive until an expensive EngineCore startup.
    migrated.update(
        {
            "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
            "fingerprint": "projected-fp",
            "sealed": True,
        }
    )
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "projected-fp",
    )
    catalog_dir = tmp_path / "elastic_graph_catalog"
    catalog_dir.mkdir()
    (catalog_dir / "elastic_graph_catalog_projected-fp.json").write_text(
        json.dumps(migrated)
    )
    loaded = startup_plan.load_elastic_graph_catalog_coverage(
        config,
        SimpleNamespace(elastic_graph_execution_policy=policy.to_payload()),
    )
    assert loaded["compiled_piecewise_sizes"] == compiled


def test_mtp_batched_q1_workspace_migration_changes_only_ledger(monkeypatch):
    shapes = [{"step_key": [0, 3, 1, 4, 0], "marker": 1}]
    payload = {
        "coverage": {
            "representation": "bounded_exact_hotset",
            "mtp_verifier_contract": "batched-causal-q1-v1",
            "mtp_verifier_workspace_bytes": 132 << 20,
            "required_shapes": 1,
            "restore_step_keys": [[0, 3, 1, 2, 0]],
        },
        "shapes": shapes,
        "complete_shapes": 1,
    }
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "1")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_PSEUDO_DECODE", "0")
    monkeypatch.setenv("AG2_VLLM_FLASHINFER_DCP_PREFILL_CUDAGRAPH", "0")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB", "64")

    migrated = startup_plan._migrate_mtp_batched_q1_workspace_catalog(
        payload, SimpleNamespace()
    )

    assert migrated["coverage"]["mtp_verifier_workspace_bytes"] == 64 << 20
    assert migrated["coverage"]["restore_step_keys"] == [[0, 3, 1, 2, 0]]
    assert migrated["shapes"] == shapes
    assert migrated["complete_shapes"] == 1


def test_missing_elastic_graph_catalog_reports_effective_identity(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    monkeypatch.delenv("AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", raising=False)
    monkeypatch.delenv("AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE", raising=False)

    assert startup_plan.load_elastic_graph_catalog(
        SimpleNamespace(), SimpleNamespace()
    ) == {}
    assert "fingerprint=0123456789abcdef" in caplog.text
    assert (
        "elastic_graph_catalog_0123456789abcdef.json" in caplog.text
    )


def test_elastic_graph_catalog_rejects_unsealed_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    startup_plan.save_elastic_graph_catalog(
        config,
        kv_config,
        {
            (1, 0, 1, 1, 1): {
                "cold_peak_bytes": 64,
                "hot_peak_bytes": 32,
                "resident_bytes": 32,
                "floor_bytes": 0,
                "cold_observations": 1,
                "cold_stable_replays": 0,
                "hot_observations": 1,
                "hot_stable_replays": 0,
            }
        },
    )

    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {}


def test_elastic_calibration_resumes_identity_matched_partial_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    key = (0, 3, 23, 64, 0)
    startup_plan.save_elastic_graph_catalog(
        config,
        kv_config,
        {
            key: {
                "cold_peak_bytes": 251658240,
                "hot_peak_bytes": 0,
                "resident_bytes": 251658240,
                "floor_bytes": 0,
                "cold_observations": 1,
                "cold_stable_replays": 0,
                "hot_observations": 0,
                "hot_stable_replays": 0,
            }
        },
    )

    assert startup_plan.load_elastic_graph_calibration_checkpoint(
        config, kv_config
    ) == {
        key: {
            "cold_peak_bytes": 251658240,
            "hot_peak_bytes": 0,
            "resident_bytes": 251658240,
            "floor_bytes": 0,
            "cold_observations": 1,
            "cold_stable_replays": 0,
            "hot_observations": 0,
            "hot_stable_replays": 0,
        }
    }


def test_elastic_calibration_partial_migration_requires_exact_scope(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    source = tmp_path / "elastic_graph_catalog" / (
        "elastic_graph_catalog_fedcba9876543210.json"
    )
    source.parent.mkdir()
    partial_row = {
        "step_key": [0, 3, 23, 64, 0],
        "cold_peak_bytes": 251658240,
        "hot_peak_bytes": 0,
        "resident_bytes": 251658240,
        "floor_bytes": 0,
        "cold_observations": 1,
        "cold_stable_replays": 0,
        "hot_observations": 0,
        "hot_stable_replays": 0,
    }
    partial_row.update(
        startup_plan._catalog_policy_row_metadata(
            partial_row["step_key"],
            policy=_catalog_policy(),
            max_num_batched_tokens=4096,
            compiled_piecewise_sizes=frozenset(),
        )
    )
    source.write_text(
        json.dumps(
            {
                "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
                "fingerprint": "fedcba9876543210",
                "sealed": False,
                "shapes": [partial_row],
            }
        )
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", "fedcba9876543210"
    )
    assert (
        startup_plan.load_elastic_graph_calibration_checkpoint(config, kv_config)
        == {}
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE",
        startup_plan.ELASTIC_GRAPH_CALIBRATION_CHECKPOINT_MIGRATION_SCOPE,
    )

    resumed = startup_plan.load_elastic_graph_calibration_checkpoint(
        config, kv_config
    )

    assert resumed[(0, 3, 23, 64, 0)]["cold_peak_bytes"] == 251658240
    destination = json.loads(
        (
            tmp_path
            / "elastic_graph_catalog"
            / "elastic_graph_catalog_0123456789abcdef.json"
        ).read_text()
    )
    assert destination["calibration_migrated_from"] == "fedcba9876543210"
    assert destination["sealed"] is False


def test_calibration_boundary_reuses_only_sealed_policy_migration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    policy = startup_plan._effective_graph_execution_policy(kv_config)
    catalog_dir = tmp_path / "elastic_graph_catalog"
    catalog_dir.mkdir()
    source_fingerprint = "fedcba9876543210"
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", source_fingerprint
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE",
        startup_plan.GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE,
    )
    (catalog_dir / f"elastic_graph_catalog_{source_fingerprint}.json").write_text(
        json.dumps(
            {
                "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
                "fingerprint": source_fingerprint,
                "sealed": True,
            }
        )
    )
    destination = catalog_dir / "elastic_graph_catalog_0123456789abcdef.json"
    destination.write_text(
        json.dumps(
            {
                "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
                "fingerprint": "0123456789abcdef",
                "sealed": False,
                "migrated_from": source_fingerprint,
                "migration_scope": (
                    startup_plan.GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE
                ),
                "coverage": {
                    "decode_max_x": 40,
                    "graph_execution_policy_fingerprint": policy.fingerprint,
                },
            }
        )
    )

    assert startup_plan.load_elastic_graph_calibration_boundary(
        config, kv_config
    ) == {"decode_max_x": 40, "bounded_exact_hotset": 0}

    source = json.loads(
        (catalog_dir / f"elastic_graph_catalog_{source_fingerprint}.json").read_text()
    )
    source["sealed"] = False
    (catalog_dir / f"elastic_graph_catalog_{source_fingerprint}.json").write_text(
        json.dumps(source)
    )
    assert startup_plan.load_elastic_graph_calibration_boundary(
        config, kv_config
    ) == {}


def test_calibration_boundary_preserves_sealed_identity_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    policy = startup_plan._effective_graph_execution_policy(kv_config)
    catalog_dir = tmp_path / "elastic_graph_catalog"
    catalog_dir.mkdir()
    source_fingerprint = "fedcba9876543210"
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", source_fingerprint
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE",
        startup_plan.ELASTIC_GRAPH_CALIBRATION_CHECKPOINT_MIGRATION_SCOPE,
    )
    monkeypatch.setattr(
        startup_plan, "_validate_sealed_migration_source", lambda *args, **kwargs: None
    )
    source = {
        "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "fingerprint": source_fingerprint,
        "sealed": True,
        "coverage": {
            "decode_max_x": 40,
            "representation": "bounded_exact_hotset",
            "graph_execution_policy_fingerprint": policy.fingerprint,
        },
    }
    (catalog_dir / f"elastic_graph_catalog_{source_fingerprint}.json").write_text(
        json.dumps(source)
    )
    destination = {
        "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "fingerprint": "0123456789abcdef",
        "sealed": True,
        "calibration_migrated_from": source_fingerprint,
        "calibration_migration_scope": (
            startup_plan.ELASTIC_GRAPH_CALIBRATION_CHECKPOINT_MIGRATION_SCOPE
        ),
        "coverage": {
            "decode_max_x": 40,
            "representation": "bounded_exact_hotset",
            "graph_execution_policy_fingerprint": policy.fingerprint,
        },
    }
    (
        catalog_dir / "elastic_graph_catalog_0123456789abcdef.json"
    ).write_text(json.dumps(destination))

    assert startup_plan.load_elastic_graph_calibration_boundary(
        config, kv_config
    ) == {"decode_max_x": 40, "bounded_exact_hotset": 1}


def _policy_reseal_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", "fedcba9876543210"
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE",
        startup_plan.GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE,
    )
    policy = _catalog_policy()
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            disable_speculation_on_non_decode=False,
        ),
        additional_config={"elastic_compiled_piecewise_sizes": []},
    )
    kv_config = SimpleNamespace(
        elastic_graph_execution_policy=policy.to_payload()
    )
    retained_keys = ((1, 3, 1, 1, 1), (0, 3, 1, 4, 0))
    missing_key = (0, 3, 2, 8, 0)

    def complete_row(key, *, cold=64):
        row = {
            "step_key": list(key),
            "cold_peak_bytes": cold,
            "hot_peak_bytes": 32,
            "resident_bytes": 32,
            "floor_bytes": 0,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
            "finalized_pinned_owner_set": 0,
        }
        row.update(
            startup_plan._catalog_policy_row_metadata(
                row["step_key"],
                policy=policy,
                max_num_batched_tokens=4096,
                compiled_piecewise_sizes=frozenset(),
            )
        )
        return row

    source_rows = [complete_row(key) for key in retained_keys]
    coverage = {
        "representation": "bounded_exact_hotset",
        "decode_max_x": 2,
        "mixed_max_x": 2,
        "full_context_max_x": 1,
        "pinned_full_entries": 0,
        "pinned_full_bytes": 0,
        "required_step_keys": [list(key) for key in retained_keys],
        "required_shapes": len(retained_keys),
        "restore_step_keys": [list(key) for key in retained_keys],
        "piecewise_replay_contract": (
            startup_plan.BOUNDED_PIECEWISE_REPLAY_CONTRACT
        ),
        "compiled_piecewise_sizes": [],
        "compiled_piecewise_contract": None,
        "compiled_piecewise_owners": [],
        "graph_execution_policy_fingerprint": policy.fingerprint,
        "verifier_contract": policy.verifier_contract,
        "verifier_configuration": policy.verifier_configuration,
        "math_contract": policy.math_contract,
        "capture_state_abi": startup_plan.ELASTIC_CAPTURE_STATE_ABI,
    }
    source = {
        "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "fingerprint": "fedcba9876543210",
        "sealed": True,
        "coverage": coverage,
        "graph_execution_policy": policy.to_payload(),
        "complete_shapes": len(source_rows),
        "shapes": source_rows,
    }
    destination_rows = [*source_rows, complete_row(missing_key, cold=96)]
    destination = {
        "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "fingerprint": "0123456789abcdef",
        "sealed": False,
        "migrated_from": "fedcba9876543210",
        "migration_scope": startup_plan.GRAPH_EXECUTION_POLICY_CATALOG_MIGRATION_SCOPE,
        "coverage": coverage,
        "graph_execution_policy": policy.to_payload(),
        "complete_shapes": len(destination_rows),
        "shapes": destination_rows,
    }
    catalog_dir = tmp_path / "elastic_graph_catalog"
    catalog_dir.mkdir()
    source_path = catalog_dir / "elastic_graph_catalog_fedcba9876543210.json"
    destination_path = catalog_dir / "elastic_graph_catalog_0123456789abcdef.json"
    source_path.write_text(json.dumps(source))
    destination_path.write_text(json.dumps(destination))
    catalog = {
        tuple(row["step_key"]): {
            name: row[name]
            for name in (
                "cold_peak_bytes",
                "hot_peak_bytes",
                "resident_bytes",
                "floor_bytes",
                "cold_observations",
                "cold_stable_replays",
                "hot_observations",
                "hot_stable_replays",
                "finalized_pinned_owner_set",
            )
        }
        for row in destination_rows
    }
    report = {
        "retained_keys": [list(key) for key in retained_keys],
        "missing_required_keys": [list(missing_key)],
    }

    def project(payload, *_):
        projected = dict(payload)
        projected["coverage"] = dict(payload["coverage"])
        projected["coverage"]["required_step_keys"] = [
            list(key) for key in retained_keys
        ]
        return projected, report

    monkeypatch.setattr(
        startup_plan,
        "project_elastic_graph_catalog_to_policy",
        project,
    )
    return config, kv_config, catalog, source_path, destination_path, missing_key


def test_policy_migration_fast_reseal_recomputes_dropped_report(
    tmp_path, monkeypatch
):
    config, kv_config, catalog, _source, destination, missing = (
        _policy_reseal_fixture(tmp_path, monkeypatch)
    )

    coverage = startup_plan.try_reseal_policy_migrated_elastic_graph_catalog(
        config,
        kv_config,
        catalog,
        calibration_wall_seconds=0.25,
    )

    assert coverage is not None
    assert coverage["decode_max_x"] == 2
    assert coverage["mixed_max_x"] == 2
    assert coverage["full_context_max_x"] == 1
    assert list(missing) in coverage["required_step_keys"]
    payload = json.loads(destination.read_text())
    assert payload["sealed"] is True
    assert payload["complete_shapes"] == 3


def test_policy_migration_fast_reseal_is_miss_before_destination_checkpoint(
    tmp_path, monkeypatch
):
    config, kv_config, catalog, _source, destination, _missing = (
        _policy_reseal_fixture(tmp_path, monkeypatch)
    )
    destination.unlink()

    assert (
        startup_plan.try_reseal_policy_migrated_elastic_graph_catalog(
            config,
            kv_config,
            catalog,
            calibration_wall_seconds=0.0,
        )
        is None
    )


@pytest.mark.parametrize("mutate_source", [False, True])
def test_policy_migration_fast_reseal_rejects_retained_row_mutation(
    tmp_path, monkeypatch, mutate_source
):
    config, kv_config, catalog, source, destination, _missing = (
        _policy_reseal_fixture(tmp_path, monkeypatch)
    )
    path = source if mutate_source else destination
    payload = json.loads(path.read_text())
    payload["shapes"][0]["cold_peak_bytes"] += 1
    path.write_text(json.dumps(payload))
    if not mutate_source:
        catalog[tuple(payload["shapes"][0]["step_key"])][
            "cold_peak_bytes"
        ] += 1

    with pytest.raises(RuntimeError, match="retained measurement differs"):
        startup_plan.try_reseal_policy_migrated_elastic_graph_catalog(
            config,
            kv_config,
            catalog,
            calibration_wall_seconds=0.25,
        )


def test_policy_migration_fast_reseal_accepts_monotone_retained_refresh(
    tmp_path, monkeypatch
):
    config, kv_config, catalog, source, destination, _missing = (
        _policy_reseal_fixture(tmp_path, monkeypatch)
    )
    retained = (1, 3, 1, 1, 1)
    source_payload = json.loads(source.read_text())
    for row in source_payload["shapes"]:
        if row["step_key"] == list(retained):
            row["cold_stable_replays"] = 2
    source.write_text(json.dumps(source_payload))
    catalog[retained]["cold_peak_bytes"] += 32
    catalog[retained]["resident_bytes"] += 16
    catalog[retained]["cold_observations"] += 1
    payload = json.loads(destination.read_text())
    for row in payload["shapes"]:
        if row["step_key"] == list(retained):
            row["cold_peak_bytes"] += 32
            row["resident_bytes"] += 16
            row["cold_observations"] += 1
    destination.write_text(json.dumps(payload))

    coverage = startup_plan.try_reseal_policy_migrated_elastic_graph_catalog(
        config,
        kv_config,
        catalog,
        calibration_wall_seconds=0.25,
    )

    assert coverage is not None
    sealed = json.loads(destination.read_text())
    refreshed = next(
        row for row in sealed["shapes"] if row["step_key"] == list(retained)
    )
    assert refreshed["cold_peak_bytes"] == 96
    assert refreshed["cold_observations"] == 3
    assert refreshed["cold_stable_replays"] == 1


def test_policy_migration_fast_reseal_waits_for_new_row_stability(
    tmp_path, monkeypatch
):
    config, kv_config, catalog, _source, destination, missing = (
        _policy_reseal_fixture(tmp_path, monkeypatch)
    )
    catalog[missing]["cold_stable_replays"] = 0
    payload = json.loads(destination.read_text())
    for row in payload["shapes"]:
        if row["step_key"] == list(missing):
            row["cold_stable_replays"] = 0
    destination.write_text(json.dumps(payload))

    assert (
        startup_plan.try_reseal_policy_migrated_elastic_graph_catalog(
            config,
            kv_config,
            catalog,
            calibration_wall_seconds=0.25,
        )
        is None
    )
    assert json.loads(destination.read_text())["sealed"] is False


def test_policy_migration_fast_reseal_waits_for_absent_new_row(
    tmp_path, monkeypatch
):
    config, kv_config, catalog, _source, destination, missing = (
        _policy_reseal_fixture(tmp_path, monkeypatch)
    )
    catalog.pop(missing)
    payload = json.loads(destination.read_text())
    payload["shapes"] = [
        row for row in payload["shapes"] if row["step_key"] != list(missing)
    ]
    payload["complete_shapes"] = len(payload["shapes"])
    destination.write_text(json.dumps(payload))

    assert (
        startup_plan.try_reseal_policy_migrated_elastic_graph_catalog(
            config,
            kv_config,
            catalog,
            calibration_wall_seconds=0.25,
        )
        is None
    )
    assert json.loads(destination.read_text())["sealed"] is False


def test_mtp_phase_continuity_migration_reuses_unchanged_k0_control(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    source = tmp_path / "elastic_graph_catalog" / (
        "elastic_graph_catalog_fedcba9876543210.json"
    )
    source.parent.mkdir()

    def row(key):
        value = {
            "step_key": list(key),
            "cold_peak_bytes": 64,
            "hot_peak_bytes": 32,
            "resident_bytes": 32,
            "floor_bytes": 0,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
        }
        value.update(
            startup_plan._catalog_policy_row_metadata(
                value["step_key"],
                policy=_catalog_policy(),
                max_num_batched_tokens=4096,
                compiled_piecewise_sizes=frozenset(),
            )
        )
        return value

    source_rows = [
        row((0, 0, 40, 4096, 0)),
        row((0, 3, 40, 256, 0)),
        row((1, 3, 40, 40, 1)),
    ]
    source_policy = _catalog_policy()

    source.write_text(
        json.dumps(
            {
                "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
                "fingerprint": "fedcba9876543210",
                "sealed": True,
                "complete_shapes": len(source_rows),
                "graph_execution_policy": source_policy.to_payload(),
                "coverage": {
                    "representation": "bounded_exact_hotset",
                    "graph_execution_policy_fingerprint": source_policy.fingerprint,
                    "verifier_contract": source_policy.verifier_contract,
                    "verifier_configuration": source_policy.verifier_configuration,
                    "math_contract": source_policy.math_contract,
                    "capture_state_abi": startup_plan.ELASTIC_CAPTURE_STATE_ABI,
                    "compiled_piecewise_sizes": [],
                    "compiled_piecewise_contract": (
                        "torch-compile-no-cudagraph-v1"
                    ),
                    "required_step_keys": [
                        row["step_key"] for row in source_rows
                    ],
                    "required_shapes": len(source_rows),
                },
                "shapes": source_rows,
            }
        )
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", "fedcba9876543210"
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE",
        startup_plan.MTP_PHASE_CONTINUITY_CALIBRATION_MIGRATION_SCOPE,
    )

    resumed = startup_plan.load_elastic_graph_calibration_checkpoint(
        config, kv_config
    )

    assert (0, 0, 40, 4096, 0) in resumed
    assert (0, 3, 40, 256, 0) not in resumed
    assert (1, 3, 40, 40, 1) not in resumed
    destination = json.loads(
        (
            tmp_path
            / "elastic_graph_catalog"
            / "elastic_graph_catalog_0123456789abcdef.json"
        ).read_text()
    )
    assert destination["sealed"] is False
    assert "coverage" not in destination
    assert destination["phase_continuity_reused_k0_piecewise_rows"] == 1


def test_seal_elastic_graph_catalog_requires_every_declared_shape(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    complete = (1, 0, 1, 1, 1)
    catalog = {
        complete: {
            "cold_peak_bytes": 64,
            "hot_peak_bytes": 32,
            "resident_bytes": 32,
            "floor_bytes": 0,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
        }
    }

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        startup_plan.seal_elastic_graph_catalog(
            config,
            kv_config,
            catalog,
            required_step_keys=(complete, (1, 0, 2, 2, 1)),
            calibration_wall_seconds=1.0,
            decode_max_x=1,
            mixed_max_x=1,
            full_context_max_x=1,
            pinned_full_entries=1,
            pinned_full_bytes=32,
        )

    path = startup_plan.seal_elastic_graph_catalog(
        config,
        kv_config,
        catalog,
        required_step_keys=(complete,),
        calibration_wall_seconds=1.0,
        decode_max_x=1,
        mixed_max_x=1,
        full_context_max_x=1,
        pinned_full_entries=1,
        pinned_full_bytes=32,
    )
    assert path.endswith("elastic_graph_catalog_elastic-fp.json")
    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == catalog
    assert startup_plan.load_elastic_graph_catalog_coverage(config, kv_config) == {
        "required_step_keys": [list(complete)],
        "required_shapes": 1,
        "calibration_wall_seconds": 1.0,
        "decode_max_x": 1,
        "mixed_max_x": 1,
        "full_context_max_x": 1,
            "pinned_full_entries": 1,
            "pinned_full_bytes": 32,
            "compiled_piecewise_sizes": [],
            "graph_execution_policy_fingerprint": _catalog_policy().fingerprint,
        "verifier_contract": "approved-q4-full-control-v1",
        "verifier_configuration": "unspecified-v1",
        "math_contract": "approved-q4-full-math-v1",
        "capture_state_abi": startup_plan.ELASTIC_CAPTURE_STATE_ABI,
    }


def test_seal_accepts_finalized_pinned_full_without_repeat_cold_capture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    full = (1, 3, 9, 9, 1)
    row = {
        "cold_peak_bytes": 1447034880,
        "hot_peak_bytes": 799014912,
        "resident_bytes": 1447034880,
        "floor_bytes": 0,
        "cold_observations": 3,
        "cold_stable_replays": 0,
        "hot_observations": 2,
        "hot_stable_replays": 1,
        "finalized_pinned_owner_set": 1,
    }

    path = startup_plan.seal_elastic_graph_catalog(
        config,
        kv_config,
        {full: row},
        required_step_keys=(full,),
        calibration_wall_seconds=1.0,
        decode_max_x=9,
        mixed_max_x=9,
        full_context_max_x=1,
        pinned_full_entries=1,
        pinned_full_bytes=32,
    )
    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {full: row}
    assert path.endswith("elastic_graph_catalog_elastic-fp.json")

    piecewise = (0, 3, 9, 4096, 0)
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        startup_plan.seal_elastic_graph_catalog(
            config,
            kv_config,
            {piecewise: row},
            required_step_keys=(piecewise,),
            calibration_wall_seconds=1.0,
            decode_max_x=9,
            mixed_max_x=9,
            full_context_max_x=1,
            pinned_full_entries=1,
            pinned_full_bytes=32,
        )


def test_seal_and_load_bounded_exact_hotset_without_pinned_family(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    piecewise = (0, 3, 1, 2, 0)
    full = (1, 3, 1, 1, 1)
    full_row = {
        "cold_peak_bytes": 64,
        "hot_peak_bytes": 32,
        "resident_bytes": 32,
        "floor_bytes": 0,
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 2,
        "hot_stable_replays": 1,
    }
    piecewise_row = {
        **full_row,
        "hot_peak_bytes": 0,
        "hot_observations": 0,
        "hot_stable_replays": 0,
    }

    path = startup_plan.seal_elastic_graph_catalog(
        config,
        kv_config,
        {piecewise: piecewise_row, full: full_row},
        required_step_keys=(piecewise, full),
        calibration_wall_seconds=4.0,
        decode_max_x=40,
        mixed_max_x=40,
        full_context_max_x=1,
        pinned_full_entries=0,
        pinned_full_bytes=0,
        representation="bounded_exact_hotset",
        restore_step_keys=(piecewise, full),
    )

    assert path.endswith("elastic_graph_catalog_elastic-fp.json")
    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {
        piecewise: piecewise_row,
        full: full_row,
    }
    coverage = startup_plan.load_elastic_graph_catalog_coverage(config, kv_config)
    assert coverage["representation"] == "bounded_exact_hotset"
    assert coverage["restore_step_keys"] == [list(piecewise), list(full)]
    assert coverage["pinned_full_entries"] == 0
    assert coverage["pinned_full_bytes"] == 0
    assert coverage["piecewise_replay_contract"] == (
        startup_plan.BOUNDED_PIECEWISE_REPLAY_CONTRACT
    )


def test_control_plane_only_migration_preserves_sealed_bounded_catalog(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    fingerprint = {"value": "fedcba9876543210"}
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: fingerprint["value"],
    )
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    piecewise = (0, 3, 1, 2, 0)
    full = (1, 3, 1, 1, 1)
    full_row = {
        "cold_peak_bytes": 64,
        "hot_peak_bytes": 32,
        "resident_bytes": 32,
        "floor_bytes": 0,
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 2,
        "hot_stable_replays": 1,
    }
    piecewise_row = {
        **full_row,
        "hot_peak_bytes": 0,
        "hot_observations": 0,
        "hot_stable_replays": 0,
    }
    startup_plan.seal_elastic_graph_catalog(
        config,
        kv_config,
        {piecewise: piecewise_row, full: full_row},
        required_step_keys=(piecewise, full),
        calibration_wall_seconds=1.0,
        decode_max_x=40,
        mixed_max_x=40,
        full_context_max_x=1,
        pinned_full_entries=0,
        pinned_full_bytes=0,
        representation="bounded_exact_hotset",
        restore_step_keys=(piecewise, full),
    )

    fingerprint["value"] = "0123456789abcdef"
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATE_FROM", "fedcba9876543210"
    )
    monkeypatch.setenv(
        "AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE",
        startup_plan.CONTROL_PLANE_ONLY_CATALOG_MIGRATION_SCOPE,
    )

    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {
        piecewise: piecewise_row,
        full: full_row,
    }
    destination = json.loads(
        (
            tmp_path
            / "elastic_graph_catalog"
            / "elastic_graph_catalog_0123456789abcdef.json"
        ).read_text()
    )
    assert destination["sealed"] is True
    assert destination["coverage"]["representation"] == "bounded_exact_hotset"
    assert destination["coverage"]["decode_max_x"] == 40
    assert destination["migrated_from"] == "fedcba9876543210"

    monkeypatch.delenv("AG2_VLLM_ELASTIC_CATALOG_MIGRATION_SCOPE")
    with pytest.raises(RuntimeError, match="exact active lineage"):
        startup_plan.load_elastic_graph_catalog(config, kv_config)


def test_bounded_piecewise_still_requires_repeated_stable_cold_capture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    piecewise = (0, 3, 1, 2, 0)
    incomplete = {
        "cold_peak_bytes": 64,
        "hot_peak_bytes": 0,
        "resident_bytes": 32,
        "floor_bytes": 0,
        "cold_observations": 1,
        "cold_stable_replays": 0,
        "hot_observations": 0,
        "hot_stable_replays": 0,
    }

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        startup_plan.seal_elastic_graph_catalog(
            SimpleNamespace(),
            SimpleNamespace(),
            {piecewise: incomplete},
            required_step_keys=(piecewise,),
            calibration_wall_seconds=1.0,
            decode_max_x=1,
            mixed_max_x=1,
            full_context_max_x=1,
            pinned_full_entries=0,
            pinned_full_bytes=0,
            representation="bounded_exact_hotset",
            restore_step_keys=(piecewise,),
        )


def test_bounded_catalog_rejects_pinned_or_uncovered_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    key = (1, 3, 1, 1, 1)
    row = {
        "cold_peak_bytes": 64,
        "hot_peak_bytes": 32,
        "resident_bytes": 32,
        "floor_bytes": 0,
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 2,
        "hot_stable_replays": 1,
    }
    kwargs = dict(
        required_step_keys=(key,),
        calibration_wall_seconds=1.0,
        decode_max_x=40,
        mixed_max_x=40,
        full_context_max_x=1,
        pinned_full_entries=0,
        pinned_full_bytes=0,
        representation="bounded_exact_hotset",
        restore_step_keys=((0, 3, 1, 2, 0),),
    )

    with pytest.raises(RuntimeError, match="product boundary is invalid"):
        startup_plan.seal_elastic_graph_catalog(
            SimpleNamespace(), SimpleNamespace(), {key: row}, **kwargs
        )
