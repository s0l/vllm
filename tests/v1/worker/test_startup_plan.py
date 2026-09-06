# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.v1.core.elastic_graph import (
    EXECUTION_MANIFEST_SCHEMA,
    GraphExecutionPolicy,
    OwnerGraphExecutionPolicy,
    RuntimeGeneration,
    resolve_step_physical_keys,
)
from vllm.v1.core.elastic_price_identity import PRICE_OWNER_GENERATION, price_identity
from vllm.v1.worker import startup_plan

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def price_platform_fixture(monkeypatch):
    # These loader tests use synthetic configs; native/config fingerprint
    # sensitivity is exercised separately by test_gpu_worker.
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_price_identity",
        lambda *_: price_identity({"fixture": "loader"}),
    )
    monkeypatch.setattr(
        startup_plan, "elastic_catalog_owner_generation", lambda *_: "loader-runtime"
    )


def _resident_owners(generation):
    owners = resolve_step_physical_keys(
        (1, 3, 1, 1, 1), RuntimeGeneration(generation), 4096, (), _catalog_policy()
    )
    return tuple(sorted(((owners[0].identity, 96), (owners[1].identity, 48))))


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
        num_speculative_tokens=3,
        speculative_config=None,
    )


def _catalog_kv_config() -> SimpleNamespace:
    return SimpleNamespace(
        elastic_graph_execution_policy=_catalog_policy().to_payload()
    )


def test_cudagraph_recipe_persists_all_reachable_descriptors(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(startup_plan, "compute_plan_fingerprint", lambda *_: "fp")

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


def _write_catalog(tmp_path, *, sealed=True, fingerprint="0123456789abcdef"):
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    key = (1, 3, 1, 1, 1)
    row = {
        "step_key": list(key),
        "cold_peak_bytes": 320,
        "hot_peak_bytes": 192,
        "resident_bytes": 160,
        "floor_bytes": 16,
        "resident_key_bytes": _resident_owners(PRICE_OWNER_GENERATION),
        "cold_observations": 2,
        "cold_stable_replays": 1,
        "hot_observations": 2,
        "hot_stable_replays": 1,
        **startup_plan._catalog_policy_row_metadata(
            key,
            policy=_catalog_policy(),
            max_num_batched_tokens=4096,
            compiled_piecewise_sizes=frozenset(),
        ),
    }
    semantic_witnesses = startup_plan.expected_semantic_token_witnesses(
        configured_k=3,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
    )
    shapes = [row]
    for witness_key, _live_tokens in semantic_witnesses:
        witness_row = dict(row)
        witness_row["step_key"] = list(witness_key)
        witness_row.update(
            startup_plan._catalog_policy_row_metadata(
                witness_key,
                policy=_catalog_policy(),
                max_num_batched_tokens=4096,
                compiled_piecewise_sizes=frozenset(),
            )
        )
        shapes.append(witness_row)
    required_keys = [key, *(witness_key for witness_key, _ in semantic_witnesses)]
    coverage = {
        "decode_max_x": 1,
        "mixed_max_x": 1,
        "full_context_max_x": 1,
        "pinned_full_entries": 1,
        "pinned_full_bytes": 160,
        "required_shapes": len(required_keys),
        "required_step_keys": [list(required_key) for required_key in required_keys],
        "semantic_token_witnesses": [
            {"step_key": list(witness_key), "live_num_tokens": live_tokens}
            for witness_key, live_tokens in semantic_witnesses
        ],
        "semantic_witness_contract": "live-current-to-physical-piecewise-v1",
        "compiled_piecewise_sizes": [],
        "graph_execution_policy_fingerprint": _catalog_policy().fingerprint,
        "verifier_contract": _catalog_policy().verifier_contract,
        "verifier_configuration": _catalog_policy().verifier_configuration,
        "math_contract": _catalog_policy().math_contract,
        "capture_state_abi": startup_plan.ELASTIC_CAPTURE_STATE_ABI,
        "execution_manifest_schema": EXECUTION_MANIFEST_SCHEMA,
        "dispatch_representations": [
            "hot_graph",
            "compiled_only",
            "inactive",
            "forbidden",
        ],
        "residency_intent_contract": "current-dispatch-hot-successor-union-v1",
        "speculative_depth_contract": "scheduled-requested-executed-k-v1",
    }
    payload = {
        "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "price_identity": price_identity({"fixture": "loader"}),
        "resident_owner_generation": PRICE_OWNER_GENERATION,
        "fingerprint": fingerprint,
        "sealed": sealed,
        "finalized_offline": True,
        "complete_shapes": len(shapes),
        "coverage": coverage,
        "graph_execution_policy": _catalog_policy().to_payload(),
        "shapes": shapes,
    }
    path = (
        tmp_path
        / "elastic_graph_catalog"
        / "elastic_graph_catalog_0123456789abcdef.json"
    )
    path.parent.mkdir()
    path.write_text(json.dumps(payload))
    return config, kv_config, key, row, path


def test_exact_elastic_catalog_loads_complete_identity_matched_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config, kv_config, key, row, _path = _write_catalog(tmp_path)

    loaded = startup_plan.load_elastic_graph_catalog(config, kv_config)
    assert loaded[key] == (
        {
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
            )
        }
        | {"resident_key_bytes": _resident_owners("loader-runtime")}
    )
    assert len(loaded) == 1 + len(
        startup_plan.expected_semantic_token_witnesses(
            configured_k=3,
            max_num_seqs=1,
            max_num_batched_tokens=4096,
        )
    )
    coverage = startup_plan.load_elastic_graph_catalog_coverage(config, kv_config)
    assert coverage["decode_max_x"] == 1
    assert coverage["required_step_keys"] == [
        list(key),
        *(
            list(witness_key)
            for witness_key, _live_tokens in (
                startup_plan.expected_semantic_token_witnesses(
                    configured_k=3,
                    max_num_seqs=1,
                    max_num_batched_tokens=4096,
                )
            )
        ),
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "unsealed",
        "stale",
        "not_finalized",
        "lineage",
        "migration",
        "witness",
        "provenance",
        "provenance_overflow",
        "provenance_uppercase",
        "provenance_short",
        "provenance_nonhex",
    ],
)
def test_exact_elastic_catalog_rejects_invalid_serving_artifact(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config, kv_config, _key, _row, path = _write_catalog(tmp_path)
    payload = json.loads(path.read_text())
    if mutation == "unsealed":
        payload["sealed"] = False
    elif mutation == "stale":
        payload["fingerprint"] = "stale-fingerprint"
    elif mutation == "not_finalized":
        payload["finalized_offline"] = False
    elif mutation == "lineage":
        payload["graph_execution_policy"]["fingerprint"] = "0" * 64
    elif mutation == "witness":
        payload["coverage"]["semantic_token_witnesses"] = None
    elif mutation == "provenance":
        payload["shapes"][0]["resident_key_bytes"] = [["a" * 64, -1]]
    elif mutation == "provenance_overflow":
        payload["shapes"][0]["resident_key_bytes"] = [["a" * 64, 161]]
    elif mutation == "provenance_uppercase":
        payload["shapes"][0]["resident_key_bytes"] = [["A" * 64, 1]]
    elif mutation == "provenance_short":
        payload["shapes"][0]["resident_key_bytes"] = [["a" * 63, 1]]
    elif mutation == "provenance_nonhex":
        payload["shapes"][0]["resident_key_bytes"] = [["g" * 64, 1]]
    else:
        payload["migrated_from"] = "0123456789abcdef"
        payload["migration_scope"] = "removed"
    path.write_text(json.dumps(payload))

    if mutation in {"migration", "not_finalized"}:
        with pytest.raises(RuntimeError, match="offline-finalized"):
            startup_plan.load_elastic_graph_catalog(config, kv_config)
    elif mutation.startswith("provenance"):
        with pytest.raises(RuntimeError, match="resident-key provenance"):
            startup_plan.load_elastic_graph_catalog(config, kv_config)
    else:
        with pytest.raises(RuntimeError, match="canonical elastic Graph catalog"):
            startup_plan.load_elastic_graph_catalog(config, kv_config)


@pytest.mark.parametrize("inventory", ["required_step_keys", "restore_step_keys"])
def test_elastic_catalog_rejects_duplicate_key_inventory(
    tmp_path, monkeypatch, inventory
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config, kv_config, key, _row, path = _write_catalog(tmp_path)
    payload = json.loads(path.read_text())
    payload["coverage"][inventory] = [list(key), list(key)]
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=f"duplicate {inventory.split('_')[0]}"):
        startup_plan.load_elastic_graph_catalog(config, kv_config)


def test_existing_corrupt_canonical_catalog_is_not_a_calibration_miss(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config, kv_config, _key, _row, path = _write_catalog(tmp_path)
    path.write_text("{")

    with pytest.raises(RuntimeError, match="refusing automatic calibration"):
        startup_plan.load_elastic_graph_catalog(config, kv_config)


def test_broken_symlink_canonical_catalog_is_not_a_calibration_miss(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config, kv_config, _key, _row, path = _write_catalog(tmp_path)
    path.unlink()
    path.symlink_to(path.with_name("missing-catalog.json"))

    with pytest.raises(RuntimeError, match="refusing automatic calibration"):
        startup_plan.load_elastic_graph_catalog(config, kv_config)


def test_uninspectable_canonical_catalog_is_not_a_calibration_miss(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config, kv_config, _key, _row, path = _write_catalog(tmp_path)
    real_lstat = startup_plan.os.lstat

    def reject_catalog_lstat(candidate):
        if candidate == str(path):
            raise PermissionError("catalog access denied")
        return real_lstat(candidate)

    monkeypatch.setattr(startup_plan.os, "lstat", reject_catalog_lstat)
    with pytest.raises(RuntimeError, match="cannot be inspected"):
        startup_plan.load_elastic_graph_catalog(config, kv_config)


def test_elastic_catalog_rejects_partial_required_shape_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "0123456789abcdef",
    )
    config, kv_config, _key, _row, path = _write_catalog(tmp_path)
    payload = json.loads(path.read_text())
    missing_key = [1, 3, 2, 2, 1]
    payload["coverage"]["required_step_keys"].append(missing_key)
    payload["coverage"]["required_shapes"] += 1
    broken_row = dict(payload["shapes"][0])
    broken_row["step_key"] = missing_key
    broken_row.update(
        startup_plan._catalog_policy_row_metadata(
            missing_key,
            policy=_catalog_policy(),
            max_num_batched_tokens=4096,
            compiled_piecewise_sizes=frozenset(),
        )
    )
    broken_row["cold_peak_bytes"] = -1
    payload["shapes"].append(broken_row)
    payload["complete_shapes"] += 1
    path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="invalid numeric fields"):
        startup_plan.load_elastic_graph_catalog(config, kv_config)


def test_missing_exact_elastic_catalog_is_empty_and_coverage_fails(
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

    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {}
    with pytest.raises(RuntimeError, match="coverage is unreadable"):
        startup_plan.load_elastic_graph_catalog_coverage(config, kv_config)
