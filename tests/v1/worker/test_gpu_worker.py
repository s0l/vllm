# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu_worker import _post_warmup_elastic_kv_budget
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
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
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_fingerprint_ignores_superseded_bootstrap_reserve(plan_env):
    class Config:
        def __init__(self, reserve: int, backend: str = "flashinfer") -> None:
            self.additional_config = {"elastic_runtime_reserve_mb": reserve}
            self.backend = backend
            self.not_copyable = object()

        def compute_hash(self) -> str:
            return f"{self.backend}:{self.additional_config}"

    fp = startup_plan.compute_plan_fingerprint
    first = Config(1024)
    assert fp(first, 0, 1) == fp(Config(2048), 0, 1)
    assert first.additional_config == {"elastic_runtime_reserve_mb": 1024}
    assert fp(Config(1024), 0, 1) != fp(Config(1024, "other"), 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes
    assert not applied._startup_plan_includes_runtime_headroom

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes
    assert explicit._startup_plan_includes_runtime_headroom


def test_startup_plan_save_uses_pre_planning_identity(plan_env):
    worker = _plan_worker(config_hash="before")
    maybe_apply_startup_plan(worker)
    assert worker._startup_plan_fingerprint

    worker.vllm_config = SimpleNamespace(compute_hash=lambda: "after")
    maybe_save_startup_plan(worker, 50 * GiB_bytes)

    replay = _plan_worker(config_hash="before")
    maybe_apply_startup_plan(replay)
    assert replay.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes


def test_startup_plan_promotes_headroom_only_after_full_warmup(plan_env):
    worker = _plan_worker()
    maybe_apply_startup_plan(worker)
    maybe_save_startup_plan(
        worker,
        45 * GiB_bytes,
        post_warmup_complete=True,
    )
    replay = _plan_worker()
    maybe_apply_startup_plan(replay)
    assert replay.cache_config.kv_cache_memory_bytes == 45 * GiB_bytes
    assert replay._startup_plan_includes_runtime_headroom


def test_post_warmup_elastic_budget_accounts_for_full_runtime_headroom():
    mib = 1 << 20
    assert _post_warmup_elastic_kv_budget(
        physical_budget=4000 * mib,
        current_free=1200 * mib,
        initial_free=15000 * mib,
        requested_memory=14500 * mib,
        mapping_quantum=2 * mib,
    ) == 4550 * mib
