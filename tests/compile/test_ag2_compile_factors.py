# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.compilation.caching import _ag2_downstream_compile_factors


@pytest.mark.parametrize(
    "suffix",
    [
        "REQUIRE_CATALOG",
        "AUTO_CALIBRATE",
        "CATALOG_PATH",
        "CALIBRATION_ROLE",
        "CALIBRATION_SURFACE",
        "CALIBRATION_RECEIPT",
        "CALIBRATION_REQUEST",
        "CALIBRATION_SEED_CATALOG",
        "CALIBRATION_MAX_NEW_ROWS_PER_PROCESS",
        "CALIBRATION_MAX_PRODUCER_EPOCHS_PER_PROCESS",
    ],
)
def test_catalog_control_plane_does_not_recompile_model(monkeypatch, suffix):
    name = "AG2_VLLM_ELASTIC_" + suffix
    monkeypatch.delenv(name, raising=False)
    original = _ag2_downstream_compile_factors()
    monkeypatch.setenv(name, "one")
    assert _ag2_downstream_compile_factors() == original
    monkeypatch.setenv(name, "two")
    assert _ag2_downstream_compile_factors() == original


def test_unknown_downstream_flag_still_invalidates_compile(monkeypatch):
    monkeypatch.delenv("AG2_VLLM_ELASTIC_NEW_KERNEL", raising=False)
    original = _ag2_downstream_compile_factors()
    monkeypatch.setenv("AG2_VLLM_ELASTIC_NEW_KERNEL", "1")
    assert _ag2_downstream_compile_factors() != original


def test_row_profile_content_not_location_identifies_compile(monkeypatch):
    monkeypatch.setenv("AG2_VLLM_TP3_ROW_PROFILE_SHA256", "a" * 64)
    monkeypatch.setenv("AG2_VLLM_TP3_ROW_PROFILE", "/first/profile.json")
    original = _ag2_downstream_compile_factors()
    monkeypatch.setenv("AG2_VLLM_TP3_ROW_PROFILE", "/moved/profile.json")
    assert _ag2_downstream_compile_factors() == original
    monkeypatch.setenv("AG2_VLLM_TP3_ROW_PROFILE_SHA256", "b" * 64)
    assert _ag2_downstream_compile_factors() != original


def test_admission_policy_does_not_change_model_compile_identity(monkeypatch):
    monkeypatch.setenv("AG2_VLLM_TP3_OWNER_PREQUANT", "1")
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "1")
    monkeypatch.setenv("AG2_VLLM_PREFILL_ADMISSION_DELAY_MS", "0")
    monkeypatch.setenv("AG2_VLLM_PREFILL_ADMISSION_MAX_DELAY_MS", "")
    before = _ag2_downstream_compile_factors()

    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "64")
    monkeypatch.setenv("AG2_VLLM_PREFILL_ADMISSION_DELAY_MS", "2")
    monkeypatch.setenv("AG2_VLLM_PREFILL_ADMISSION_MAX_DELAY_MS", "20")
    after = _ag2_downstream_compile_factors()

    assert after == before
    assert after["AG2_VLLM_TP3_OWNER_PREQUANT"] == "1"


def test_graph_receipt_observer_does_not_change_model_compile_identity(monkeypatch):
    monkeypatch.setenv("AG2_VLLM_GRAPH_MODE_RECEIPT", "1")
    before = _ag2_downstream_compile_factors()

    monkeypatch.setenv("AG2_VLLM_GRAPH_MODE_RECEIPT", "0")
    after = _ag2_downstream_compile_factors()

    assert after == before
    assert after["AG2_VLLM_GRAPH_MODE_RECEIPT"] == "1"
