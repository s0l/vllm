# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.gpu_worker import (
    _reconcile_pre_kv_physical_budget,
    _startup_kv_memory_deductions,
    _startup_plan_kv_budget,
    maybe_rocm_profiling_fallback,
)
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)


@pytest.mark.parametrize(
    ("profiled", "profile_free", "prepared_free", "expected"),
    [
        (6_000, 8_000, 7_200, 5_200),
        (6_000, 8_000, 8_000, 6_000),
        (6_000, 8_000, 8_500, 6_000),
        (500, 8_000, 7_300, 0),
        (-100, 8_000, 8_000, 0),
    ],
)
def test_pre_kv_physical_budget_charges_only_endpoint_consumption(
    profiled: int,
    profile_free: int,
    prepared_free: int,
    expected: int,
):
    assert (
        _reconcile_pre_kv_physical_budget(
            profiled,
            profile_free,
            prepared_free,
        )
        == expected
    )


@pytest.mark.parametrize("endpoints", [(-1, 0), (0, -1)])
def test_pre_kv_budget_rejects_invalid_physical_endpoint(endpoints):
    with pytest.raises(ValueError, match="nonnegative"):
        _reconcile_pre_kv_physical_budget(100, *endpoints)


@pytest.mark.parametrize("cached_reuse", [0, 128, 512])
def test_pre_kv_budget_conserves_physical_pages_when_owners_reuse_cache(cached_reuse):
    total, excluded, resident, cached, transient = 16_000, 1_000, 7_000, 512, 400
    owner = 512
    profile_free = total - resident - cached
    profiled = profile_free - excluded - transient
    prepared_free = profile_free - (owner - cached_reuse)
    budget = _reconcile_pre_kv_physical_budget(profiled, profile_free, prepared_free)
    # Independent final-state accounting: reused pages move from cached to live,
    # rather than appearing as a second allocation outside the physical budget.
    final_live = resident + owner
    final_cached = cached - cached_reuse
    assert budget + final_live + final_cached + excluded + transient == total


def test_startup_plan_persists_the_consumed_kv_budget_contract():
    assert (
        _startup_plan_kv_budget(
            elastic_dynamic_kv=True,
            final_available_bytes=5_200,
            requested_limit_bytes=6_000,
        )
        == 5_200
    )
    assert (
        _startup_plan_kv_budget(
            elastic_dynamic_kv=False,
            final_available_bytes=5_200,
            requested_limit_bytes=6_000,
        )
        == 6_000
    )


# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(
    config_hash="abc123",
    free_memory=78 * GiB_bytes,
    kv_bytes=None,
    *,
    max_num_seqs=64,
    async_scheduling=False,
    skip_mm_profiling=False,
):
    """The minimal Worker surface the startup-plan entry points touch."""
    scheduler_config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=4096,
        max_num_scheduled_tokens=4096,
        max_num_encoder_input_tokens=4096,
        encoder_cache_size=4096,
        async_scheduling=async_scheduling,
        enable_chunked_prefill=True,
        disable_chunked_mm_input=False,
    )
    multimodal_config = SimpleNamespace(
        compute_hash=lambda: "mm-graph-hash",
        language_model_only=False,
        skip_mm_profiling=skip_mm_profiling,
        limit_per_prompt={"image": 1},
        mm_processor_kwargs=None,
        mm_encoder_tp_mode="weights",
    )
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            compute_hash=lambda **_kwargs: config_hash,
            scheduler_config=scheduler_config,
            model_config=SimpleNamespace(multimodal_config=multimodal_config),
            parallel_config=SimpleNamespace(world_size=3),
        ),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    from vllm.v1.core import elastic_price_identity

    monkeypatch.setattr(
        elastic_price_identity,
        "native_price_provenance",
        lambda: {"fixture-native": "unchanged"},
    )
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(envs, "VLLM_ENABLE_STARTUP_PLAN", True)
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker(max_num_seqs=40).vllm_config, 0, 1)
    assert base != fp(_plan_worker(async_scheduling=True).vllm_config, 0, 1)
    assert base != fp(_plan_worker(skip_mm_profiling=True).vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch.object(
        startup_plan,
        "_startup_profile_source_hashes",
        return_value={"changed": "source"},
    ):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch.object(
        startup_plan,
        "elastic_runtime_source_hashes",
        return_value={"changed": "runtime-source"},
    ):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


@pytest.mark.parametrize(
    "name",
    startup_plan.STARTUP_PROFILE_ENV_NAMES,
)
def test_startup_plan_fingerprint_tracks_physical_dag_env(plan_env, monkeypatch, name):
    fp = startup_plan.compute_plan_fingerprint
    monkeypatch.setenv(name, "before")
    before = fp(_plan_worker().vllm_config, 0, 1)
    monkeypatch.setenv(name, "after")
    assert before != fp(_plan_worker().vllm_config, 0, 1)


def test_row_profile_content_binds_price_startup_and_runtime(plan_env, monkeypatch):
    from vllm.v1.core import elastic_runtime

    config = _plan_worker().vllm_config
    kv = SimpleNamespace(
        elastic_attention_stride=128,
        elastic_gdn_stride=64,
        elastic_mapping_quantum=32,
        elastic_gdn_blocks_per_request=3,
    )

    def identities():
        return (
            startup_plan.compute_plan_fingerprint(config, 0, 3),
            startup_plan.compute_elastic_graph_catalog_fingerprint(config, kv),
            elastic_runtime.compute_elastic_runtime_generation(config),
        )

    monkeypatch.setenv("AG2_VLLM_TP3_ROW_PROFILE_SHA256", "a" * 64)
    monkeypatch.setenv("AG2_VLLM_TP3_ROW_PROFILE", "/first/profile")
    original = identities()
    monkeypatch.setenv("AG2_VLLM_TP3_ROW_PROFILE", "/moved/profile")
    assert identities() == original
    monkeypatch.setenv("AG2_VLLM_TP3_ROW_PROFILE_SHA256", "b" * 64)
    assert all(a != b for a, b in zip(original, identities(), strict=True))


def test_elastic_identities_include_profile_config(plan_env):
    from vllm.v1.core import elastic_runtime

    base = _plan_worker().vllm_config
    maxseq40 = _plan_worker(max_num_seqs=40).vllm_config
    kv = SimpleNamespace(
        num_blocks=64,
        elastic_attention_stride=86_900_736,
        elastic_gdn_stride=19_611_648,
        elastic_mapping_quantum=2 << 20,
        elastic_gdn_initial_blocks=4,
        elastic_gdn_blocks_per_request=3,
        elastic_rank_budget_bytes=(1, 2, 3),
        elastic_rank_primary_mapped_bytes=(4, 5, 6),
        elastic_rank_gdn_mapped_bytes=(7, 8, 9),
        elastic_graph_execution_policy=None,
    )
    with patch.object(
        elastic_runtime,
        "elastic_runtime_source_hashes",
        return_value={"source": "same"},
    ):
        assert elastic_runtime.compute_elastic_runtime_generation(
            base
        ) != elastic_runtime.compute_elastic_runtime_generation(maxseq40)
        assert startup_plan.compute_elastic_graph_catalog_fingerprint(
            base, kv
        ) != startup_plan.compute_elastic_graph_catalog_fingerprint(maxseq40, kv)

        policy_catalog = startup_plan.compute_elastic_graph_catalog_fingerprint(
            base,
            SimpleNamespace(
                **{
                    **vars(kv),
                    "elastic_graph_execution_policy": {"fingerprint": "policy-a"},
                }
            ),
        )
        assert policy_catalog != startup_plan.compute_elastic_graph_catalog_fingerprint(
            base, kv
        )

        with patch.dict(
            "os.environ",
            {"AG2_VLLM_ELASTIC_MM_ACTIVATION_LOAN_BYTES": "1096810496"},
            clear=False,
        ):
            assert base.compute_hash() == "abc123"
            loan_generation = elastic_runtime.compute_elastic_runtime_generation(base)
            loan_catalog = startup_plan.compute_elastic_graph_catalog_fingerprint(
                base, kv
            )
        assert loan_generation != elastic_runtime.compute_elastic_runtime_generation(
            base
        )
        assert loan_catalog == startup_plan.compute_elastic_graph_catalog_fingerprint(
            base, kv
        )


def test_catalog_identity_ignores_serving_load_policy(plan_env):
    from vllm.v1.core import elastic_runtime

    config = _plan_worker().vllm_config
    kv = SimpleNamespace(
        num_blocks=64,
        elastic_attention_stride=86_900_736,
        elastic_gdn_stride=19_611_648,
        elastic_mapping_quantum=2 << 20,
        elastic_gdn_initial_blocks=4,
        elastic_gdn_blocks_per_request=3,
        elastic_rank_budget_bytes=(1, 2, 3),
        elastic_rank_primary_mapped_bytes=(4, 5, 6),
        elastic_rank_gdn_mapped_bytes=(7, 8, 9),
        elastic_graph_execution_policy=None,
    )
    with patch.object(
        elastic_runtime,
        "elastic_runtime_source_hashes",
        return_value={"source": "same"},
    ):
        with patch.dict(
            "os.environ", {"AG2_VLLM_ELASTIC_REQUIRE_CATALOG": "0"}, clear=False
        ):
            discovery = startup_plan.compute_elastic_graph_catalog_fingerprint(
                config, kv
            )
        with patch.dict(
            "os.environ", {"AG2_VLLM_ELASTIC_REQUIRE_CATALOG": "1"}, clear=False
        ):
            serving = startup_plan.compute_elastic_graph_catalog_fingerprint(config, kv)
    assert discovery == serving


def test_catalog_identity_separates_runtime_policy_from_physical_sources(plan_env):
    from vllm.v1.core import elastic_runtime

    config = _plan_worker().vllm_config
    kv = SimpleNamespace(
        num_blocks=64,
        elastic_attention_stride=86_900_736,
        elastic_gdn_stride=19_611_648,
        elastic_mapping_quantum=2 << 20,
        elastic_gdn_initial_blocks=4,
        elastic_gdn_blocks_per_request=3,
        elastic_rank_primary_mapped_bytes=(4, 5, 6),
        elastic_rank_gdn_mapped_bytes=(7, 8, 9),
        elastic_graph_execution_policy=None,
    )
    excluded = {
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.worker.elastic_catalog_tool",
        "vllm.v1.worker.startup_plan",
    }
    assert excluded.isdisjoint(elastic_runtime._ELASTIC_PHYSICAL_CATALOG_SOURCE_MODULES)
    assert excluded.issubset(elastic_runtime._ELASTIC_RUNTIME_SOURCE_MODULES)

    with patch.object(
        startup_plan,
        "elastic_catalog_physical_source_hashes",
        return_value={"physical": "stable"},
    ):
        catalog = startup_plan.compute_elastic_graph_catalog_fingerprint(config, kv)
        with patch.object(
            startup_plan,
            "elastic_runtime_source_hashes",
            return_value={"runtime-policy": "changed"},
        ):
            assert (
                startup_plan.compute_elastic_graph_catalog_fingerprint(config, kv)
                == catalog
            )

    with patch.object(
        startup_plan,
        "elastic_catalog_physical_source_hashes",
        return_value={"physical": "changed"},
    ):
        assert (
            startup_plan.compute_elastic_graph_catalog_fingerprint(config, kv)
            != catalog
        )


def test_automatic_catalog_calibration_requires_explicit_maintenance_role(
    monkeypatch,
):
    from vllm.v1.core.elastic_runtime import elastic_auto_calibration_enabled

    monkeypatch.delenv("AG2_VLLM_ELASTIC_AUTO_CALIBRATE", raising=False)
    monkeypatch.delenv("AG2_VLLM_ELASTIC_CALIBRATION_ROLE", raising=False)
    assert elastic_auto_calibration_enabled() is False

    monkeypatch.setenv("AG2_VLLM_ELASTIC_AUTO_CALIBRATE", "1")
    with pytest.raises(RuntimeError, match="maintenance-only"):
        elastic_auto_calibration_enabled()

    monkeypatch.setenv("AG2_VLLM_ELASTIC_CALIBRATION_ROLE", "maintenance")
    assert elastic_auto_calibration_enabled() is True


def test_catalog_identity_ignores_unmapped_budget_slack(plan_env):
    config = _plan_worker().vllm_config
    kv = SimpleNamespace(
        num_blocks=64,
        elastic_attention_stride=86_900_736,
        elastic_gdn_stride=19_611_648,
        elastic_mapping_quantum=2 << 20,
        elastic_gdn_initial_blocks=4,
        elastic_gdn_blocks_per_request=3,
        elastic_rank_budget_bytes=(100, 200, 300),
        elastic_rank_primary_mapped_bytes=(4, 5, 6),
        elastic_rank_gdn_mapped_bytes=(7, 8, 9),
        elastic_graph_execution_policy=None,
    )
    base = startup_plan.compute_elastic_graph_catalog_fingerprint(config, kv)
    slack_only = SimpleNamespace(
        **{
            **vars(kv),
            "elastic_rank_budget_bytes": tuple(
                value + (2 << 20) for value in kv.elastic_rank_budget_bytes
            ),
        }
    )
    mapped_change = SimpleNamespace(
        **{
            **vars(kv),
            "elastic_rank_primary_mapped_bytes": (4, 5, 7),
        }
    )

    assert (
        startup_plan.compute_elastic_graph_catalog_fingerprint(config, slack_only)
        == base
    )
    assert (
        startup_plan.compute_elastic_graph_catalog_fingerprint(config, mapped_change)
        == base
    )


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


def test_elastic_startup_does_not_withhold_graph_guess_or_generic_buffer():
    graph_guess = 700 << 20
    assert _startup_kv_memory_deductions(
        elastic_dynamic_kv=True,
        cudagraph_memory_estimate=graph_guess,
        estimate_cudagraphs=True,
    ) == (0, 0)
    assert _startup_kv_memory_deductions(
        elastic_dynamic_kv=False,
        cudagraph_memory_estimate=graph_guess,
        estimate_cudagraphs=True,
    ) == (graph_guess, 150 << 20)


# The fallback reads only the sign of the measured drop and this process's torch
# reservation; free memory is only logged, so no amount here is a device size.
ANY_FREE_MEMORY = 8 * GiB_bytes
MEASURED_DROP = 4 * GiB_bytes
TORCH_RESERVED = 3 * GiB_bytes
RELEASED_BY_OTHERS = 2 * GiB_bytes


def _snapshot(free_memory, torch_memory=0):
    return SimpleNamespace(free_memory=free_memory, torch_memory=torch_memory)


def _profile_result(consumed, reserved_before=0, reserved_after=0):
    """A result whose free-memory readings agree with `consumed`, which
    `memory_profiling` derives as the drop in free memory, negative when it grew."""
    return SimpleNamespace(
        total_consumed=consumed,
        transient_peak_headroom=0,
        before_create=_snapshot(ANY_FREE_MEMORY, reserved_before),
        after_profile=_snapshot(ANY_FREE_MEMORY - consumed, reserved_after),
    )


@pytest.fixture
def rocm(request):
    with patch.object(
        gpu_worker, "current_platform", SimpleNamespace(is_rocm=lambda: request.param)
    ):
        yield request.param


@pytest.mark.parametrize("rocm", [True, False], indirect=True)
def test_profiling_fallback_declines_when_free_memory_dropped(rocm):
    """The profiling measurement is kept as-is whenever free memory dropped."""
    result = _profile_result(consumed=MEASURED_DROP)

    assert maybe_rocm_profiling_fallback(result) is None


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_replaces_a_released_measurement(rocm):
    """A negative measurement describes the rest of the device, so it is replaced
    by this process's reservation, which the rest of the device cannot move."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_after=TORCH_RESERVED,
    )

    assert maybe_rocm_profiling_fallback(result) == TORCH_RESERVED


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_never_returns_a_negative_amount(rocm):
    """A reservation that shrank across the run cannot become negative usage."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_before=TORCH_RESERVED,
        reserved_after=0,
    )

    assert maybe_rocm_profiling_fallback(result) == 0


@pytest.mark.parametrize("rocm", [False], indirect=True)
def test_profiling_fallback_declines_off_rocm(rocm):
    """Platforms that account frees eagerly keep reporting the error, so the
    caller's assertion stays reachable there."""
    result = _profile_result(consumed=-RELEASED_BY_OTHERS)

    assert maybe_rocm_profiling_fallback(result) is None
