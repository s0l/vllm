# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU producer/publisher/consumer proof; synthetic bytes are never GPU evidence."""

import copy
import json
from types import SimpleNamespace

import pytest

from vllm.v1.core.elastic_graph import (
    GraphExecutionPolicy,
    OwnerGraphExecutionPolicy,
    RuntimeGeneration,
    SemanticGraphStep,
    canonical_graph_step_key,
    short_decode_inventory_xs,
)
from vllm.v1.core.elastic_price_identity import price_identity
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.elastic_calibrator import (
    CalibrationSurface,
    ElasticCatalogCalibrator,
)
from vllm.v1.worker import startup_plan
from vllm.v1.worker.elastic_catalog_tool import publish_measured_catalog


def policy(k=3, *, full=False, batched=False):
    return GraphExecutionPolicy(
        verifier_contract="batched-causal-q1-v1" if batched else "native-backend-v1",
        math_contract="native-target-baseline-v1",
        owners=(
            OwnerGraphExecutionPolicy(
                "mtp_decode",
                (1,),
                "NONE",
                activation="speculative",
                token_source="requests",
                execution_order=2,
            ),
            OwnerGraphExecutionPolicy(
                "mtp_prefill",
                (k + 1,),
                "PIECEWISE",
                activation="speculative",
                token_source="step",
                execution_order=1,
            ),
            OwnerGraphExecutionPolicy(
                "target",
                (1, k + 1) if full else (),
                "PIECEWISE",
                activation="always",
                token_source="step",
                execution_order=0,
            ),
        ),
    )


@pytest.mark.parametrize(
    "full,batched,phase,qlen,budget,expected",
    [
        (False, False, "decode", 1, 32, (0, 3, 3, 4, 0)),
        (False, False, "decode", 4, 32, (0, 3, 3, 16, 0)),
        (False, False, "decode", 4, 13, (0, 3, 3, 13, 0)),
        (False, False, "decode", 4, None, (0, 3, 3, 12, 0)),
        (True, False, "decode", 1, 32, (1, 3, 4, 4, 1)),
        (True, False, "decode", 4, 32, (1, 3, 4, 16, 4)),
        (False, True, "decode", 4, 32, (0, 3, 4, 16, 4)),
        (True, False, "mixed", 4, 32, (0, 3, 3, 16, 0)),
    ],
)
def test_canonical_key_preserves_existing_representation_contract(
    full,
    batched,
    phase,
    qlen,
    budget,
    expected,
):
    semantic = SemanticGraphStep(
        3,
        3,
        3 * qlen,
        qlen if phase == "decode" else None,
        phase,
        ("target", "mtp_prefill", "mtp_decode"),
    )
    assert (
        canonical_graph_step_key(
            semantic,
            policy(full=full, batched=batched),
            physical_num_reqs=4,
            max_num_batched_tokens=budget,
        )
        == expected
    )


class SimulatedExecution:
    """Keep real semantic producers; replace allocation, admission and model work.

    Requests originate in EngineCore._add_elastic_restore_request. The step
    simulator consumes their prompt/draft state and the real scheduler computes
    the key. Physical owner identities and sealed row validation remain real.
    """

    def __init__(self, k, x, *, full=False):
        self.calls = []
        self.retentions: set[int] = set()
        self.fail_admission = False
        self.k = k
        self.config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
            additional_config={},
            num_speculative_tokens=k,
            speculative_config=None,
        )
        scheduler = self.scheduler = object.__new__(Scheduler)
        scheduler.scheduler_config = self.config.scheduler_config
        scheduler.vllm_config = self.config
        scheduler.num_spec_tokens = k
        scheduler.max_num_running_reqs = x
        scheduler.elastic_on_demand_graphs = True
        scheduler._elastic_restore_mode = True
        scheduler._elastic_graph_execution_policy = policy(k, full=full)
        scheduler._elastic_graph_catalog = {}
        scheduler._elastic_graph_catalog_coverage = {}
        scheduler._elastic_admission_controller = SimpleNamespace(
            generation=RuntimeGeneration("cpu-catalog"),
            resident_bytes=0,
            entries={},
        )
        scheduler.requests = {}
        scheduler._rebuild_elastic_short_decode_inventory = None
        scheduler.prepare_elastic_restore_execution = self.assert_hot
        scheduler.assert_elastic_restore_captures_hot = self.assert_hot
        scheduler.retain_elastic_restore_captures = self.retain
        scheduler.release_elastic_restore_retention = self.retentions.remove
        scheduler.promote_elastic_restore_retention_to_serving = self.promote
        core = self.core = object.__new__(EngineCore)
        core.scheduler = scheduler
        core.vllm_config = self.config
        core.is_pooling_model = False
        core.async_scheduling = False
        core.preprocess_add_request = self.preprocess
        core.add_request = self.add
        core.abort_requests = self.abort
        core._begin_elastic_restore_physical_epoch = self.capture
        core._prepare_elastic_restore_capture = lambda key: self.capture((key,))
        core._prepare_elastic_restore_admission = self.admit
        core._rollback_elastic_restore_physical_epoch = self.rollback
        core._run_elastic_restore_step = self.step
        core._drain_elastic_restore = self.drain

    def preprocess(self, request):
        assert request.sampling_params.temperature == 0.6
        assert request.cache_salt == request.request_id
        return SimpleNamespace(
            request_id=request.request_id,
            prompt_len=len(request.prompt_token_ids),
            prefilled=False,
            spec_token_ids=[],
        ), 0

    def add(self, request, _wave):
        self.scheduler.requests[request.request_id] = request

    def abort(self, ids):
        for request_id in ids:
            self.scheduler.requests.pop(request_id, None)

    def retain(self, keys):
        self.assert_hot(keys)
        token = len(self.calls)
        self.retentions.add(token)
        return token

    def promote(self, token, keys):
        assert len(keys) == 1 and keys[0].logical.owner == "mtp_decode"
        self.retentions.remove(token)

    def capture(self, keys):
        self.calls.append(("capture", tuple(keys)))
        entries = self.scheduler._elastic_admission_controller.entries
        for key in keys:
            for physical in self.scheduler._resolve_elastic_step_physical_keys(key):
                entries[physical] = SimpleNamespace(
                    hot=True,
                    price=SimpleNamespace(resident_bytes=1),
                )

    def assert_hot(self, keys):
        if isinstance(keys[0], int):
            keys = (keys,)
        for key in keys:
            assert all(
                self.scheduler._elastic_admission_controller.entries[p].hot
                for p in self.scheduler._resolve_elastic_step_physical_keys(key)
            )

    def admit(self, ids, **_kwargs):
        return 0 if self.fail_admission else len(ids)

    def rollback(self, *, request_ids, step_keys):
        self.calls.append(("rollback", step_keys))
        self.abort(request_ids)
        self.retentions.clear()

    def step(self):
        requests = self.scheduler.requests
        decode = all(request.prefilled for request in requests.values())
        lengths = {
            name: 1 + len(request.spec_token_ids)
            if request.prefilled
            else request.prompt_len
            for name, request in requests.items()
        }
        key = self.scheduler._canonical_elastic_graph_step_key(lengths, self.k, decode)
        self.assert_hot(key)
        self.calls.append(
            ("decode" if decode else "mixed", key, tuple(lengths.values()))
        )
        owners = self.scheduler._resolve_elastic_step_physical_keys(key)
        row = self.scheduler._elastic_graph_catalog.setdefault(key, {})
        observations = row.get("cold_observations", 0) + 1
        row.update(
            cold_peak_bytes=len(owners) + 1,
            hot_peak_bytes=len(owners) + 1,
            resident_bytes=len(owners),
            floor_bytes=0,
            resident_key_bytes=tuple(sorted((p.identity, 1) for p in owners)),
            cold_observations=observations,
            cold_stable_replays=observations - 1,
            hot_observations=observations,
            hot_stable_replays=observations - 1,
        )
        for request in requests.values():
            request.prefilled = True
            request.spec_token_ids[:] = [1] * self.k
        return SimpleNamespace(
            num_scheduled_tokens=lengths,
            num_spec_tokens_to_schedule=self.k,
            is_pure_decode_step=decode,
        )

    def drain(self):
        assert not self.scheduler.requests and not self.retentions

    def surface(self):
        x = self.scheduler.max_num_running_reqs
        canonical = self.scheduler._canonical_elastic_graph_step_key
        required = set()
        for n in range(1, x + 1):
            for m in (1, 2, 4, 8, 16, 32):
                if m >= n:
                    lengths = {str(i): m // n + int(i < m % n) for i in range(n)}
                    required.add(canonical(lengths, self.k, False))
            if n in short_decode_inventory_xs(x):
                for q in (1, self.k + 1):
                    required.add(
                        canonical(dict.fromkeys(map(str, range(n)), q), self.k, True)
                    )
        restore = self.core._elastic_restore_wave_step_keys(k=self.k, x=x, query_len=1)
        return dict(
            schema=6,
            coverage=dict(
                representation="bounded_exact_hotset",
                graph_execution_policy_fingerprint=self.scheduler._elastic_graph_execution_policy.fingerprint,
                required_step_keys=[list(key) for key in sorted(required)],
                restore_step_keys=[list(key) for key in sorted(set(restore))],
                restore_decode=dict(k=self.k, x=x, query_len=1),
                decode_max_x=x,
                mixed_max_x=x,
                full_context_max_x=1,
            ),
        )

    def parse(self, payload):
        return CalibrationSurface.from_payload(
            payload,
            policy_fingerprint=self.scheduler._elastic_graph_execution_policy.fingerprint,
            configured_k=self.k,
            max_num_seqs=self.scheduler.max_num_running_reqs,
            max_num_batched_tokens=32,
        )


@pytest.mark.parametrize(
    "k,x,full", [(3, 1, False), (3, 3, False), (1, 3, False), (3, 5, True)]
)
def test_actual_producers_publish_load_and_restore(k, x, full, tmp_path, monkeypatch):
    sim = SimulatedExecution(k, x, full=full)
    source = sim.surface()
    if full:
        del source["coverage"]["restore_decode"]  # Legacy FULL source still works.
    surface = sim.parse(source)
    before = sim.scheduler._elastic_graph_catalog
    catalog = ElasticCatalogCalibrator(sim.core).calibrate(surface)
    assert sim.scheduler._elastic_graph_catalog is before and before == {}
    assert not sim.scheduler.requests and not sim.retentions
    assert any(call[0] == "decode" for call in sim.calls)
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_catalog_fingerprint",
        lambda *_: "1123456789abcdef",
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_graph_price_identity",
        lambda *_: price_identity({"fixture": "semantic-contract"}),
    )
    monkeypatch.setattr(
        startup_plan, "elastic_catalog_owner_generation", lambda *_: "cpu-catalog"
    )
    kv = SimpleNamespace(
        elastic_graph_execution_policy=sim.scheduler._elastic_graph_execution_policy.to_payload()
    )
    destination = publish_measured_catalog(
        sim.config,
        kv,
        catalog,
        required=surface.required,
        restore=surface.restore,
        restore_decode=surface.restore_decode,
        semantic_token_witnesses=surface.semantic_token_witnesses,
        mixed_query_witnesses=surface.mixed_query_witnesses,
        decode_max_x=x,
        mixed_max_x=x,
        full_context_max_x=1,
        calibration_wall_seconds=0,
        output_root=tmp_path,
    )
    loaded = startup_plan.load_elastic_graph_catalog(
        sim.config, kv, catalog_path=str(destination)
    )
    coverage = startup_plan.load_elastic_graph_catalog_coverage(
        sim.config, kv, catalog_path=str(destination)
    )
    assert set(loaded) == set(surface.required)
    assert coverage["restore_decode"] == dict(k=k, x=x, query_len=1)
    sim.scheduler._elastic_graph_catalog = loaded
    sim.scheduler._elastic_graph_catalog_coverage = coverage
    sim.scheduler._elastic_restore_mode = False
    for change in ("missing", "extra", "wrong_k", "bad_x"):
        bad_coverage = copy.deepcopy(coverage)
        if change == "missing":
            bad_coverage["serving_carrier_step_keys"].pop()
        elif change == "extra":
            extra = next(
                key
                for key in bad_coverage["required_step_keys"]
                if key not in bad_coverage["serving_carrier_step_keys"]
            )
            bad_coverage["serving_carrier_step_keys"].append(extra)
        elif change == "wrong_k":
            bad_coverage["serving_carrier_step_keys"][0][1] += 1
        else:
            bad_coverage["decode_max_x"] = True
        sim.scheduler._elastic_graph_catalog_coverage = bad_coverage
        before_calls = len(sim.calls)
        with pytest.raises(RuntimeError):
            sim.core._restore_elastic_bounded_hotset()
        assert len(sim.calls) == before_calls
        assert sim.scheduler._elastic_restore_mode is False
    sim.scheduler._elastic_graph_catalog_coverage = coverage
    sim.core._restore_elastic_bounded_hotset()
    assert sim.scheduler._elastic_restore_mode is False
    assert len(sim.scheduler._elastic_serving_carrier_keys) == 1
    assert sim.scheduler._elastic_serving_carrier_resident_bytes == 1
    assert not sim.scheduler.requests and not sim.retentions
    if k == 1 and not full:
        assert len(coverage["serving_carrier_step_keys"]) == 2

    bad = json.loads(destination.read_text())
    bad["coverage"]["restore_decode"]["query_len"] = k + 1
    malformed = tmp_path / "wrong-role.json"
    malformed.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="restore pair"):
        startup_plan.load_elastic_graph_catalog_coverage(
            sim.config, kv, catalog_path=str(malformed)
        )
    startup_plan.load_elastic_graph_catalog_coverage(
        sim.config, kv, catalog_path=str(destination)
    )


@pytest.mark.parametrize(
    "mutation", ["missing", "null", "bool", "k", "x", "qlen", "extra", "wrong_pair"]
)
def test_piecewise_role_negatives_leave_producer_untouched(mutation):
    sim = SimulatedExecution(3, 3)
    valid = sim.surface()
    bad = copy.deepcopy(valid)
    geometry = bad["coverage"]["restore_decode"]
    if mutation == "missing":
        del bad["coverage"]["restore_decode"]
    elif mutation == "null":
        bad["coverage"]["restore_decode"] = None
    elif mutation == "extra":
        geometry["phase"] = "decode"
    elif mutation == "wrong_pair":
        geometry["query_len"] = 4
    else:
        geometry[{"bool": "x", "k": "k", "x": "x", "qlen": "query_len"}[mutation]] = {
            "bool": True,
            "k": 2,
            "x": 4,
            "qlen": 5,
        }[mutation]
    validator = ElasticCatalogCalibrator(sim.core)
    with pytest.raises((ValueError, RuntimeError)):
        validator._validate_surface_before_mutation(sim.parse(bad))
    assert not sim.calls and sim.scheduler._elastic_graph_catalog == {}
    validator._validate_surface_before_mutation(sim.parse(valid))


def test_decode_admission_failure_rolls_back_and_recovers():
    sim = SimulatedExecution(3, 3)
    surface = sim.parse(sim.surface())
    sim.fail_admission = True
    with pytest.raises(RuntimeError, match="preflight admitted no requests"):
        ElasticCatalogCalibrator(sim.core).calibrate(surface)
    assert not sim.scheduler.requests and not sim.retentions
    assert sim.scheduler._elastic_graph_catalog == {}
    assert sim.scheduler._elastic_graph_catalog_coverage == {}
    sim.fail_admission = False
    assert set(ElasticCatalogCalibrator(sim.core).calibrate(surface)) == set(
        surface.required
    )
