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



def _write_catalog(tmp_path, *, sealed=True, fingerprint="elastic-fp"):
    config = _catalog_config()
    kv_config = _catalog_kv_config()
    key = (1, 3, 1, 1, 1)
    row = {
        "step_key": list(key),
        "cold_peak_bytes": 320,
        "hot_peak_bytes": 192,
        "resident_bytes": 160,
        "floor_bytes": 16,
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
    coverage = {
        "decode_max_x": 1,
        "mixed_max_x": 1,
        "full_context_max_x": 1,
        "pinned_full_entries": 1,
        "pinned_full_bytes": 160,
        "required_shapes": 1,
        "required_step_keys": [list(key)],
        "compiled_piecewise_sizes": [],
        "graph_execution_policy_fingerprint": _catalog_policy().fingerprint,
        "verifier_contract": _catalog_policy().verifier_contract,
        "verifier_configuration": _catalog_policy().verifier_configuration,
        "math_contract": _catalog_policy().math_contract,
        "capture_state_abi": startup_plan.ELASTIC_CAPTURE_STATE_ABI,
    }
    payload = {
        "schema": startup_plan.ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "sealed": sealed,
        "complete_shapes": 1,
        "coverage": coverage,
        "graph_execution_policy": _catalog_policy().to_payload(),
        "shapes": [row],
    }
    path = (
        tmp_path
        / "elastic_graph_catalog"
        / "elastic_graph_catalog_elastic-fp.json"
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
        lambda *_: "elastic-fp",
    )
    config, kv_config, key, row, _path = _write_catalog(tmp_path)

    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {
        key: {
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
    }
    coverage = startup_plan.load_elastic_graph_catalog_coverage(config, kv_config)
    assert coverage["decode_max_x"] == 1
    assert coverage["required_step_keys"] == [list(key)]


@pytest.mark.parametrize("mutation", ["unsealed", "stale", "lineage", "migration"])
def test_exact_elastic_catalog_rejects_invalid_serving_artifact(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "elastic-fp",
    )
    config, kv_config, _key, _row, path = _write_catalog(tmp_path)
    payload = json.loads(path.read_text())
    if mutation == "unsealed":
        payload["sealed"] = False
    elif mutation == "stale":
        payload["fingerprint"] = "stale-fingerprint"
    elif mutation == "lineage":
        payload["graph_execution_policy"]["fingerprint"] = "0" * 64
    else:
        payload["migrated_from"] = "0123456789abcdef"
        payload["migration_scope"] = "removed"
    path.write_text(json.dumps(payload))

    if mutation == "migration":
        with pytest.raises(RuntimeError, match="offline-finalized"):
            startup_plan.load_elastic_graph_catalog(config, kv_config)
    else:
        assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {}


def test_missing_exact_elastic_catalog_is_empty_and_coverage_fails(
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

    assert startup_plan.load_elastic_graph_catalog(config, kv_config) == {}
    with pytest.raises(RuntimeError, match="coverage is unreadable"):
        startup_plan.load_elastic_graph_catalog_coverage(config, kv_config)
