# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from types import SimpleNamespace

import pytest

from vllm.v1.core.elastic_memory_profile import (
    allocation_envelope_proof,
    allocation_envelope_row,
    allocation_profile_shapes,
    validate_allocation_envelope_row,
)


def samples():
    corners, holdouts = allocation_profile_shapes(3, 64, 4096)
    return [
        dict(
            step_key=list(key),
            capture_peak_bytes=10000,
            resident_bytes=9000,
            floor_bytes=0,
            replay_extra_bytes=1000,
            source_reads=0,
            stable_replays=1,
        )
        for key in corners + holdouts
    ]


@pytest.mark.parametrize(
    ("native_experts", "provider_count", "expected"),
    [(False, 0, False), (True, 1, True)],
)
def test_allocation_profile_accepts_dense_and_native_expert_paths(
    native_experts, provider_count, expected
):
    from vllm.v1.engine.elastic_memory_profile import (
        _validate_native_expert_profile_contract,
    )

    config = SimpleNamespace(
        additional_config={"flashnext_native_experts": native_experts}
    )
    providers = tuple(object() for _ in range(provider_count))

    assert _validate_native_expert_profile_contract(config, providers) is expected


@pytest.mark.parametrize(("native_experts", "provider_count"), [(False, 1), (True, 0)])
def test_allocation_profile_rejects_model_provider_mismatch(
    native_experts, provider_count
):
    from vllm.v1.engine.elastic_memory_profile import (
        _validate_native_expert_profile_contract,
    )

    config = SimpleNamespace(
        additional_config={"flashnext_native_experts": native_experts}
    )
    providers = tuple(object() for _ in range(provider_count))

    with pytest.raises(RuntimeError, match="disagrees with native expert providers"):
        _validate_native_expert_profile_contract(config, providers)


def test_worker_rpc_enters_inference_mode_and_restores_caller(monkeypatch):
    import torch

    from vllm.v1.engine import elastic_memory_profile as producer

    with torch.inference_mode():
        state = torch.zeros(1)
    with pytest.raises(RuntimeError, match="InferenceMode"):
        state.add_(1)

    def replay(worker, step_key):
        assert torch.is_inference_mode_enabled()
        state.add_(1)
        if worker == "fail":
            raise RuntimeError("diagnostic failure")
        return {"value": state.item()}

    monkeypatch.setattr(producer, "_replay_allocation_profile", replay)
    assert not torch.is_inference_mode_enabled()
    assert producer.replay_allocation_profile("ok", (0, 3, 1, 1, 0))["value"] > 0
    assert not torch.is_inference_mode_enabled()
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        producer.replay_allocation_profile("fail", (0, 3, 1, 1, 0))
    assert not torch.is_inference_mode_enabled()
    producer.replay_allocation_profile("ok", (0, 3, 1, 1, 0))


def test_full_domain_profile_does_not_execute_logical_matrix():
    corners, holdouts = allocation_profile_shapes(3, 64, 4096)
    assert len(corners + holdouts) < 64
    assert (0, 3, 64, 4096, 0) in corners
    assert (0, 3, 1, 4096, 0) in corners
    assert (0, 3, 17, 51, 3) in holdouts
    proof = allocation_envelope_proof(
        k=3, max_x=64, budget=4096, policy="fixture", samples=samples()
    )
    row = allocation_envelope_row(proof)
    for x in range(1, 65):
        validate_allocation_envelope_row(
            (0, 3, x, 4096, 0), row, policy="fixture", budget=4096
        )
    assert row["cold_observations"] == row["hot_observations"] == 0


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "unstable", "io", "bool", "negative"]
)
def test_invalid_or_falsified_profile_rejects_and_recovers(change):
    original = samples()
    bad = copy.deepcopy(original)
    if change == "missing":
        bad.pop()
    elif change == "duplicate":
        bad.append(bad[0])
    elif change == "unstable":
        bad[0]["stable_replays"] = 0
    elif change == "io":
        bad[0]["source_reads"] = 1
    elif change == "bool":
        bad[0]["resident_bytes"] = True
    else:
        bad[0]["replay_extra_bytes"] = -1
    with pytest.raises(ValueError):
        allocation_envelope_proof(
            k=3, max_x=64, budget=4096, policy="fixture", samples=bad
        )
    allocation_envelope_proof(
        k=3, max_x=64, budget=4096, policy="fixture", samples=original
    )


def test_non_monotonic_holdout_widens_conservative_envelope():
    measured = samples()
    measured[-1]["capture_peak_bytes"] = 12000
    measured[-1]["resident_bytes"] = 11000
    proof = allocation_envelope_proof(
        k=3, max_x=64, budget=4096, policy="fixture", samples=measured
    )
    assert proof["peak_bytes"] == 13000
    assert proof["resident_bytes"] == 12000


def test_policy_filtered_physical_controls_are_sealed_and_validated():
    measured = samples()[:2]
    controls = tuple(tuple(sample["step_key"]) for sample in measured)
    proof = allocation_envelope_proof(
        k=3,
        max_x=64,
        budget=4096,
        policy="fixture",
        samples=measured,
        controls=controls,
    )
    row = allocation_envelope_row(proof)

    assert proof["controls"] == [list(key) for key in controls]
    validate_allocation_envelope_row(
        (0, 3, 1, 1, 0), row, policy="fixture", budget=4096
    )

    missing = copy.deepcopy(proof)
    missing["controls"].pop()
    with pytest.raises(ValueError, match="missing or repeats"):
        validate_allocation_envelope_row(
            (0, 3, 1, 1, 0),
            allocation_envelope_row(missing),
            policy="fixture",
            budget=4096,
        )
