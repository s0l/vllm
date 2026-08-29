# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.compilation.caching import _ag2_downstream_compile_factors


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
