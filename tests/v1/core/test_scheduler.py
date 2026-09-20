# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import dataclasses
import time
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
import torch

import vllm.envs as envs
from vllm.config import (
    CacheConfig,
    ECTransferConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.elastic_graph import (
    DispatchRepresentation,
    ElasticAdmissionController,
    ElasticGraphError,
    ElasticMaintenanceExecution,
    ElasticPlanKind,
    ElasticResidencyEntry,
    ElasticResidencyReceipt,
    GraphExecutionPolicy,
    GraphPrice,
    OwnerGraphExecutionPolicy,
    PhysicalReplayKey,
    ReclaimGroup,
    RuntimeGeneration,
    SemanticGraphStep,
    build_execution_manifest,
    resolve_step_physical_keys,
)
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheBlockPoolRequirements,
)
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import (
    EncoderWaveOverlay,
    Scheduler,
    _elastic_catalog_cold_residency_envelope,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import (
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
    SamplingMaskLists,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputGrammar, StructuredOutputManager

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler, mock_kv


@pytest.mark.parametrize("mm_bytes,expected", [(0, 3), (8, 3), (100, 1)])
def test_prefix_cache_frontier_prices_rank_gdn_graph_and_vision(mm_bytes, expected):
    """The weakest rank and the two phase peaks define a non-reserved frontier."""
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.worker.startup_plan import ElasticGraphCatalog

    coordinator = object.__new__(HybridKVCacheCoordinator)
    coordinator.kv_cache_config = SimpleNamespace(
        elastic_mapping_quantum=2,
        elastic_gdn_initial_blocks=4,
        elastic_gdn_blocks_per_request=3,
        elastic_rank_budget_bytes=(100, 90),
        elastic_rank_primary_mapped_bytes=(tuple(range(0, 90, 10)),) * 2,
        elastic_rank_gdn_mapped_bytes=(tuple(range(0, 30, 3)),) * 2,
    )
    coordinator.block_pool = BlockPool(8, True, 4, prefer_low_id_allocations=True)
    coordinator.mamba_block_pool = BlockPool(10, False, 4)
    scheduler = object.__new__(Scheduler)
    scheduler.elastic_on_demand_graphs = True
    scheduler.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
    scheduler.max_num_running_reqs = 2
    scheduler._elastic_mm_activation_loan_bytes = mm_bytes
    scheduler._elastic_graph_catalog = ElasticGraphCatalog(source_sha256="a" * 64)
    scheduler._elastic_graph_catalog[(0, 3, 2, 8, 4)] = {
        "cold_peak_bytes": 30,
        "hot_peak_bytes": 20,
    }
    scheduler._elastic_graph_catalog_coverage = {"_catalog_source_sha256": "a" * 64}
    scheduler._elastic_pressure_floor_external_bytes = lambda: 0
    scheduler._refresh_elastic_cache_frontier()
    assert coordinator.block_pool.cache_preservation_num_blocks == expected
    assert coordinator.block_pool.active_num_gpu_blocks == 8
    assert coordinator.block_pool.get_num_free_blocks() == 7
    # Foreign/unsealed coverage cannot activate speculative cache placement.
    scheduler._elastic_graph_catalog_coverage.clear()
    scheduler._refresh_elastic_cache_frontier()
    assert coordinator.block_pool.cache_preservation_num_blocks == 1


def _elastic_controller(scheduler: Scheduler) -> ElasticAdmissionController:
    if not hasattr(scheduler, "scheduler_config"):
        scheduler.scheduler_config = SimpleNamespace(
            max_num_batched_tokens=4096,
            max_num_seqs=64,
        )
    if not hasattr(scheduler, "_elastic_graph_execution_policy"):
        scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    if not hasattr(scheduler, "_elastic_compiled_piecewise_sizes"):
        scheduler._elastic_compiled_piecewise_sizes = frozenset()
    if not hasattr(scheduler, "_elastic_admission_controller"):
        scheduler._elastic_admission_controller = ElasticAdmissionController(
            RuntimeGeneration("scheduler-fixture")
        )
    return scheduler._elastic_admission_controller


def _new_elastic_scheduler(
    generation: RuntimeGeneration | None = None,
) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    if generation is not None:
        scheduler._elastic_admission_controller = ElasticAdmissionController(generation)
    _elastic_controller(scheduler)
    return scheduler


def _catalog_activation_scheduler(
    *, unfinished: bool = False, finished: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        _elastic_graph_catalog={},
        _elastic_graph_catalog_coverage={"decode_max_x": 47},
        requests={"completed-calibration-tombstone": object()},
        has_unfinished_requests=Mock(return_value=unfinished),
        has_finished_requests=Mock(return_value=finished),
        num_waiting_for_streaming_input=0,
        _elastic_restore_retention_id=None,
        _elastic_admission_controller=SimpleNamespace(
            pending_maintenance_plan=None,
            record_measurement=Mock(),
            record_capture_envelope=Mock(),
        ),
        _elastic_short_decode_inventory={47: ("raw",)},
        _rebuild_elastic_short_decode_inventory=Mock(),
        max_num_running_reqs=64,
        _record_elastic_capture_envelope=Mock(),
    )


def _activate_catalog(scheduler: SimpleNamespace) -> None:
    from vllm.v1.worker.startup_plan import ElasticGraphCatalog

    catalog = ElasticGraphCatalog(source_sha256="a" * 64)
    catalog[(1, 3, 1, 1, 1)] = {
        "cold_peak_bytes": 10,
        "floor_bytes": 0,
        "hot_peak_bytes": 9,
    }
    Scheduler.activate_elastic_graph_catalog(
        scheduler,
        catalog,
        {
            "representation": "bounded_exact_hotset",
            "decode_max_x": 39,
            "mixed_max_x": 39,
            "full_context_max_x": 39,
            "required_step_keys": [[1, 3, 1, 1, 1]],
            "restore_step_keys": [],
            "_catalog_source_sha256": "a" * 64,
        },
    )


def test_catalog_activation_accepts_drained_calibration_tombstones() -> None:
    scheduler = _catalog_activation_scheduler()

    _activate_catalog(scheduler)

    assert scheduler.max_num_running_reqs == 39
    assert scheduler._elastic_graph_catalog
    scheduler._rebuild_elastic_short_decode_inventory.assert_called_once_with(39)


@pytest.mark.parametrize("reverse", [False, True])
def test_catalog_activation_prices_physical_sets_before_order_free_m_class(reverse):
    """A one-Graph mixed row cannot overwrite a three-Graph decode price."""
    from vllm.v1.worker.startup_plan import ElasticGraphCatalog

    scheduler = _new_elastic_scheduler()
    scheduler._elastic_compiled_piecewise_sizes = frozenset((128,))
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy((128,))
    scheduler.max_num_running_reqs = 64
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_graph_catalog_coverage = {}
    decode, mixed = (0, 3, 32, 128, 4), (0, 3, 39, 128, 0)
    decode_keys = scheduler._resolve_elastic_step_physical_keys(decode)
    mixed_keys = scheduler._resolve_elastic_step_physical_keys(mixed)
    assert len(decode_keys) == 3 and len(mixed_keys) == 1
    rows = (
        (
            decode,
            dict(
                cold_peak_bytes=168,
                resident_bytes=200,
                hot_peak_bytes=200,
                floor_bytes=16,
                resident_key_bytes=[(key.identity, 40) for key in decode_keys],
            ),
        ),
        (
            mixed,
            dict(
                cold_peak_bytes=202,
                resident_bytes=202,
                hot_peak_bytes=180,
                floor_bytes=0,
                resident_key_bytes=[(mixed_keys[0].identity, 44)],
            ),
        ),
    )
    catalog = ElasticGraphCatalog(source_sha256="a" * 64)
    catalog.update(reversed(rows) if reverse else rows)
    scheduler.activate_elastic_graph_catalog(
        catalog,
        dict(
            representation="bounded_exact_hotset",
            decode_max_x=39,
            mixed_max_x=39,
            full_context_max_x=39,
            required_step_keys=[list(decode), list(mixed)],
            restore_step_keys=[],
            _catalog_source_sha256="a" * 64,
        ),
    )
    controller = scheduler._elastic_admission_controller
    for class_key in ((0, 3, 0, 128, 0), (0, 3, 0, 0, 0)):
        assert controller.capture_envelopes[class_key] == (202, 16, 0)
        assert controller.capture_envelope_resident_key_bytes(class_key) == ()
    assert scheduler._elastic_capture_envelope_with_provenance(decode) == (
        (200, 16, 0),
        tuple(sorted((key.identity, 40) for key in decode_keys)),
    )
    assert scheduler._elastic_capture_envelope_with_provenance(mixed) == (
        (202, 0, 0),
        ((mixed_keys[0].identity, 44),),
    )


@pytest.mark.parametrize("source_floor", [0, 16, 31, 174])
def test_elastic_cold_destination_does_not_reprice_the_source_floor(source_floor):
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    controller = scheduler._elastic_admission_controller
    key = (0, 3, 32, 128, 4)
    controller.resident_bytes = 174
    controller.transition_floor_bytes = source_floor
    controller.record_capture_envelope(
        scheduler._elastic_graph_owner_key(key),
        (200, 16, 0),
        resident_key_bytes=(),
    )
    assert scheduler._estimate_elastic_graph_step_bytes(key) == (200, True)
    assert scheduler._elastic_capture_shared_evidence(key, 200) == ()
    assert controller.transition_floor_bytes == source_floor


def test_catalog_activation_rejects_stale_restore_carrier_without_mutation() -> None:
    scheduler = _catalog_activation_scheduler()
    stale_carrier = (0, 3, 47, 188, 4)
    scheduler._elastic_graph_carrier_step_key = stale_carrier
    previous_coverage = scheduler._elastic_graph_catalog_coverage

    with pytest.raises(RuntimeError, match="quiescent pre-READY lifecycle"):
        _activate_catalog(scheduler)

    assert scheduler._elastic_graph_catalog == {}
    assert scheduler._elastic_graph_catalog_coverage is previous_coverage
    assert scheduler._elastic_graph_carrier_step_key == stale_carrier
    assert scheduler.max_num_running_reqs == 64
    scheduler._rebuild_elastic_short_decode_inventory.assert_not_called()


def test_catalog_activation_rebuilds_consumed_decode_boundary() -> None:
    """The accepted surface, not the raw scheduler cap, owns canonical X."""
    from vllm.v1.worker.startup_plan import ElasticGraphCatalog

    scheduler = _new_elastic_scheduler(RuntimeGeneration("accepted-x39"))
    scheduler.elastic_on_demand_graphs = True
    scheduler.num_spec_tokens = 3
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=47)
    scheduler._elastic_compiled_piecewise_sizes = frozenset(
        (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    )
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy(
        tuple(sorted(scheduler._elastic_compiled_piecewise_sizes))
    )
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 47}
    scheduler._rebuild_elastic_short_decode_inventory(47)
    assert max(scheduler._elastic_short_decode_inventory) == 47
    scheduler.requests = {"completed-calibration-tombstone": object()}
    scheduler.has_unfinished_requests = Mock(return_value=False)
    scheduler.has_finished_requests = Mock(return_value=False)
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._elastic_restore_retention_id = None
    scheduler._record_elastic_capture_envelope = Mock()
    scheduler.max_num_running_reqs = 47

    catalog = ElasticGraphCatalog(source_sha256="a" * 64)
    catalog[(1, 3, 39, 39, 1)] = {
        "cold_peak_bytes": 10,
        "floor_bytes": 0,
        "hot_peak_bytes": 9,
    }
    Scheduler.activate_elastic_graph_catalog(
        scheduler,
        catalog,
        {
            "representation": "bounded_exact_hotset",
            "decode_max_x": 39,
            "mixed_max_x": 39,
            "full_context_max_x": 39,
            "required_step_keys": [[1, 3, 39, 39, 1]],
            "restore_step_keys": [],
            "_catalog_source_sha256": "a" * 64,
        },
    )

    assert max(scheduler._elastic_short_decode_inventory) == 39
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(39)}, 3, True
    ) == (0, 3, 39, 156, 4)


@pytest.mark.parametrize("unfinished,finished", [(True, False), (False, True)])
def test_catalog_activation_rejects_live_scheduler_lifecycle(
    unfinished: bool, finished: bool
) -> None:
    scheduler = _catalog_activation_scheduler(unfinished=unfinished, finished=finished)

    with pytest.raises(RuntimeError, match="quiescent pre-READY lifecycle"):
        _activate_catalog(scheduler)


def _reset_elastic_loans(
    scheduler: Scheduler,
    *loans: tuple[tuple[int, ...] | None, int],
) -> None:
    controller = _elastic_controller(scheduler)
    controller.clear_loans()
    for step_key, grant_bytes in loans:
        controller.reserve_loan(step_key, grant_bytes)


def _clear_elastic_maintenance(scheduler: Scheduler) -> None:
    _elastic_controller(scheduler).clear_maintenance()


def _arm_elastic_maintenance(
    scheduler: Scheduler,
    plan: Any,
    step_key: tuple[int, ...] | None,
) -> None:
    _elastic_controller(scheduler).arm_maintenance(plan, step_key)


def _arm_fixture_capture(
    scheduler: Scheduler,
    step_key: tuple[int, ...],
    capture_loan_bytes: int = 96,
    maintenance_execution: ElasticMaintenanceExecution = (
        ElasticMaintenanceExecution.COUPLED_USER
    ),
) -> Any:
    controller = _elastic_controller(scheduler)
    physical_keys = scheduler._resolve_elastic_step_physical_keys(step_key)
    for physical_key in physical_keys:
        controller.register(
            physical_key,
            price=GraphPrice(
                1,
                1,
                f"fixture:{physical_key.identity}",
            ),
        )
    plan = controller.plan(
        controller.next_transaction_id(),
        physical_keys,
        request_bytes=0,
        available_bytes=capture_loan_bytes,
        destination_capture_endpoint_bytes=capture_loan_bytes,
        maintenance_execution=maintenance_execution,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    controller.arm_maintenance(plan, step_key)
    return plan


def _pending_elastic_maintenance(scheduler: Scheduler) -> Any:
    return _elastic_controller(scheduler).pending_maintenance_plan


def _publish_elastic_step_hot(
    scheduler: Scheduler,
    step_key: tuple[int, ...],
) -> None:
    for physical_key in scheduler._resolve_elastic_step_physical_keys(step_key):
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(
                resident_bytes=1,
                capture_peak_bytes=1,
                reclaim_group=f"private-pool:{physical_key.identity}",
            ),
            pinned=physical_key.logical.mode == "FULL",
        )


def _settle_elastic_with_receipt(
    scheduler: Scheduler,
    output: SchedulerOutput,
    resident_bytes: int,
    floor_bytes: int,
    peak_bytes: int = 0,
    *,
    publish_step_key: tuple[int, ...] | None = None,
) -> None:
    plan = output.elastic_step_plan
    transaction_id = (
        plan.transaction_id if plan is not None else output.elastic_transaction_id
    )
    physical_keys = (
        ()
        if publish_step_key is None
        else tuple(
            sorted(
                scheduler._resolve_elastic_step_physical_keys(publish_step_key),
                key=lambda key: key.identity,
            )
        )
    )
    scheduler._settle_elastic_graph_loan(
        output,
        resident_bytes,
        floor_bytes,
        peak_bytes,
        worker_receipt=ElasticResidencyReceipt(
            generation=_elastic_controller(scheduler).generation,
            transaction_id=transaction_id,
            resident_bytes=resident_bytes,
            floor_bytes=floor_bytes,
            transition_floor_bytes=floor_bytes,
            peak_bytes=max(peak_bytes, resident_bytes),
            cublas_workspace_bytes=0,
            entries=tuple(
                ElasticResidencyEntry(
                    key=physical_key,
                    resident_bytes=1,
                    local_pool_bytes=1,
                    pinned=physical_key.logical.mode == "FULL",
                    reclaimable_bytes=(0 if physical_key.logical.mode == "FULL" else 1),
                )
                for physical_key in physical_keys
            ),
        ),
    )


def _pending_elastic_loans(
    scheduler: Scheduler,
) -> tuple[tuple[tuple[int, ...] | None, int], ...]:
    return tuple(
        (loan.step_key, loan.grant_bytes)
        for loan in scheduler._elastic_admission_controller.pending_loans
    )


pytestmark = pytest.mark.cpu_test


def _current_q4_piecewise_policy(
    compiled_piecewise_sizes: tuple[int, ...] = (),
) -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="batched-causal-q1-v1",
        math_contract="pending-batched-q1-product-math-v1",
        owners=(
            OwnerGraphExecutionPolicy(
                "mtp_decode",
                (1,),
                "PIECEWISE",
                activation="speculative",
                token_source="requests",
                execution_order=2,
            ),
            OwnerGraphExecutionPolicy(
                "mtp_prefill",
                (),
                "PIECEWISE",
                compiled_piecewise_sizes=compiled_piecewise_sizes,
                activation="speculative",
                token_source="step",
                execution_order=1,
            ),
            OwnerGraphExecutionPolicy(
                "target",
                (1,),
                "PIECEWISE",
                compiled_piecewise_sizes=compiled_piecewise_sizes,
                piecewise_query_len_min_tokens=32,
                activation="always",
                token_source="step",
                execution_order=0,
            ),
        ),
    )


def _approved_q4_full_policy() -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="approved-q4-full-control-v1",
        math_contract="approved-q4-full-math-v1",
        owners=(
            OwnerGraphExecutionPolicy(
                "mtp_decode",
                (1,),
                "PIECEWISE",
                activation="speculative",
                token_source="requests",
                execution_order=2,
            ),
            OwnerGraphExecutionPolicy(
                "mtp_prefill",
                (),
                "PIECEWISE",
                activation="speculative",
                token_source="step",
                execution_order=1,
            ),
            OwnerGraphExecutionPolicy(
                "target",
                (1, 4),
                "PIECEWISE",
                activation="always",
                token_source="step",
                execution_order=0,
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _bind_test_graph_execution_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind legacy FULL-q4 fixtures to their explicit historical policy."""
    monkeypatch.setattr(
        Scheduler,
        "_elastic_graph_execution_policy",
        _approved_q4_full_policy(),
        raising=False,
    )


def test_make_scheduled_encoder_input_stats_output_embeddings():
    scheduler = create_scheduler()
    mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="image",
            identifier="image-0",
            mm_position=PlaceholderRange(offset=0, length=196),
        ),
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier="video-0",
            mm_position=PlaceholderRange(offset=200, length=196),
        ),
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="audio",
            identifier="audio-0",
            mm_position=PlaceholderRange(offset=400, length=49),
        ),
    ]
    scheduler.requests["req"] = Mock(mm_features=mm_features)

    stats = scheduler._make_scheduled_encoder_input_stats({"req": [0, 1, 2]})

    assert stats is not None
    assert stats.num_inputs == 3
    assert stats.output_tokens == 441


def test_scheduled_encoder_input_stats_disabled_without_iteration_logging(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = create_scheduler()
    make_stats = Mock(side_effect=AssertionError("stats should not be computed"))
    monkeypatch.setattr(scheduler, "_make_scheduled_encoder_input_stats", make_stats)

    scheduler_output = scheduler.schedule()

    make_stats.assert_not_called()
    assert scheduler_output.scheduled_encoder_input_stats is None


def test_scheduled_encoder_input_stats_disabled_without_log_stats(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = create_scheduler()
    scheduler.log_stats = False
    scheduler.observability_config.enable_logging_iteration_details = True
    make_stats = Mock(side_effect=AssertionError("stats should not be computed"))
    monkeypatch.setattr(scheduler, "_make_scheduled_encoder_input_stats", make_stats)

    scheduler_output = scheduler.schedule()

    make_stats.assert_not_called()
    assert scheduler_output.scheduled_encoder_input_stats is None


def test_add_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)

    for i, request in enumerate(requests):
        scheduler.add_request(request)
        assert request.request_id in scheduler.requests
        assert len(scheduler.waiting) == i + 1


def test_finish_request():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i


def test_get_num_unfinished_requests():
    scheduler = create_scheduler()
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (False, None),
        (True, 5),
    ],
)
def test_schedule(enable_prefix_caching: bool, prompt_logprobs: int | None):
    """Test scheduling.
    Two cases: default APC/no prompt logprobs; APC=True + prompt logprobs
    """
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching)
    requests = create_requests(num_requests=10, prompt_logprobs=prompt_logprobs)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    # Verify all requests are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request


def test_scheduler_stats_route_to_existing_output_client():
    scheduler = create_scheduler()
    request = create_requests(num_requests=1)[0]
    request.client_index = 1
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[1000]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert 0 not in engine_core_outputs
    assert engine_core_outputs[1].scheduler_stats is not None
    assert len(engine_core_outputs[1].outputs) == 1


def test_schedule_multimodal_requests():
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    mm_positions = [[PlaceholderRange(offset=i, length=100)] for i in range(10)]
    requests = create_requests(
        num_requests=10,
        num_tokens=200,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)
    assert len(output.scheduled_encoder_inputs) == 10
    for req_id, encoder_input in output.scheduled_encoder_inputs.items():
        assert len(encoder_input) == 1


def test_async_scheduling_pp_allows_rescheduling_with_output_placeholders():
    """Async scheduling + PP: allow multi-step in-flight scheduling per request"""
    scheduler = create_scheduler(async_scheduling=True, pipeline_parallel_size=2)
    (req,) = create_requests(num_requests=1, num_tokens=8)
    scheduler.add_request(req)

    _ = scheduler.schedule()
    assert req.num_output_placeholders > 0

    # before any update_from_output, we still expect the request can be
    # scheduled again (multi-step in-flight).
    output = scheduler.schedule()
    assert req.request_id in output.num_scheduled_tokens


def test_cached_request_data_resumed_all_token_ids_mrv1_only():
    """all_token_ids carries a resumed request's token ids to the connector
    for the V1 model runner, but is skipped entirely for the V2 model runner.
    """
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

    scheduler = create_scheduler(use_v2_model_runner=False)
    (req,) = create_requests(num_requests=1, num_tokens=8)
    req.append_output_token_ids([101, 102, 103])

    # A resumed request was not scheduled in the previous step.
    assert req.request_id not in scheduler.prev_step_scheduled_req_ids

    empty_blocks = KVCacheBlocks(blocks=((),))

    def make_cached():
        return scheduler._make_cached_request_data(
            running_reqs=[],
            resumed_reqs=[req],
            num_scheduled_tokens={req.request_id: 1},
            spec_decode_tokens={},
            req_to_new_blocks={req.request_id: empty_blocks},
        )

    # V1 model runner: the full token id list is propagated.
    assert not scheduler.use_v2_model_runner
    cached = make_cached()
    assert req.request_id in cached.resumed_req_ids
    assert cached.all_token_ids[req.request_id] == list(req.all_token_ids)

    # V2 model runner: all_token_ids is skipped entirely.
    scheduler.use_v2_model_runner = True
    cached = make_cached()
    assert req.request_id in cached.resumed_req_ids
    assert cached.all_token_ids == {}


def test_schedule_partial_requests():
    """Test scheduling behavior with partial requests.

    This test verifies that:
    1. The scheduler can handle multiple partial requests in a single step when
       constrained by encoder budget.
    2. A request in RUNNING state may be unscheduled in subsequent steps if
       there is insufficient encoder budget.
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
    )
    mm_positions = [[PlaceholderRange(offset=100, length=600)] for _ in range(3)]
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    assert scheduler.max_num_encoder_input_tokens == 1024
    # The first request is scheduled fully.
    assert output.num_scheduled_tokens[requests[0].request_id] == 800
    # The second request is scheduled partially.
    # The <img> tokens are not scheduled because of the encoder budget.
    assert output.num_scheduled_tokens[requests[1].request_id] == 100
    # The third request is also scheduled partially.
    # The <img> tokens are not scheduled because of the encoder budget.
    assert output.num_scheduled_tokens[requests[2].request_id] == 100
    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        # Only the first request has a sampled token id because
        # the rest requests are still being prefilled.
        sampled_token_ids=[[0], [], []],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step.
    # Only the first and second requests are scheduled.
    # The third request is in the RUNNING state but not scheduled in this step
    # because of the encoder budget.
    output = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 2
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 1
    assert output.num_scheduled_tokens[requests[1].request_id] == 700
    assert requests[2].request_id not in output.num_scheduled_tokens


@pytest.mark.parametrize("has_running", [True, False])
def test_schedule_prefills_gating(has_running: bool):
    """DP prefill-balancing gate: when `throttle_prefills` is True, a new
    WAITING (prefill) request is deferred ONLY if this rank has running work to
    protect. With no running requests, the prefill is admitted regardless (so a
    throttled step is never wasted as a dummy), and running/decode requests are
    unaffected. Once the cadence allows prefills again, the request is admitted.
    """
    scheduler = create_scheduler(max_num_seqs=16, max_num_batched_tokens=8192)

    if has_running:
        # Establish a running (decode) request via a prefill + output step.
        (running_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["run0"])
        scheduler.add_request(running_req)
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=["run0"],
                req_id_to_index={"run0": 0},
                sampled_token_ids=[[0]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        assert len(scheduler.running) == 1

    # Add a new WAITING (prefill) request, with prefills gated off.
    (new_req,) = create_requests(num_requests=1, num_tokens=8, req_ids=["new0"])
    scheduler.add_request(new_req)
    output = scheduler.schedule(throttle_prefills=True)

    if has_running:
        # There is running work to protect, so the new prefill is deferred...
        assert "new0" not in output.num_scheduled_tokens
        assert new_req.status == RequestStatus.WAITING
        # ...while the running/decode request keeps being scheduled.
        assert "run0" in output.num_scheduled_tokens
        # When the cadence allows prefills again, the request is admitted.
        output = scheduler.schedule()

    # No running work to protect (or cadence now open): the prefill is admitted.
    assert "new0" in output.num_scheduled_tokens
    assert any(r.req_id == "new0" for r in output.scheduled_new_reqs)


def _setup_remote_kv_resume(num_prompt_tokens: int, matched_tokens: int):
    """Drive a remote-KV request `r2` to the resume point (async load complete)
    while another request `r1` is already decoding, so the step is throttle-
    eligible. Returns the scheduler. The connector matches `matched_tokens` of
    `r2`'s prompt; the rest (if any) is local prefill.
    """
    from tests.v1.kv_connector.unit.utils import create_model_runner_output

    BLOCK_SIZE = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=matched_tokens, is_async=True),
        block_size=BLOCK_SIZE,
    )
    # Distinct prompts so r2 gets no local prefix cache hit from r1, only the
    # connector's external async load.
    r1, r2 = create_requests(
        num_requests=2,
        num_tokens=num_prompt_tokens,
        max_tokens=20,
        block_size=BLOCK_SIZE,
        req_ids=["r1", "r2"],
    )

    # r1: drive through its async KV load into the running (decode) state, so
    # self.running is non-empty (which makes the next step throttle-eligible).
    scheduler.add_request(r1)
    _step_until_kv_transfer_finished(scheduler, ["r1"])
    output = scheduler.schedule()  # promote + schedule r1
    assert "r1" in output.num_scheduled_tokens
    scheduler.update_from_output(
        output, create_model_runner_output([r1], token_id=1000)
    )
    assert scheduler.running  # r1 now decoding

    # r2: a second remote-KV request; complete its async load while r1 decodes.
    scheduler.add_request(r2)
    output = scheduler.schedule()  # r1 decodes; r2 -> WAITING_FOR_REMOTE_KVS
    assert r2.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(
        output, create_model_runner_output([r1], finished_recving={"r2"})
    )
    assert "r2" in scheduler.finished_recving_kv_req_ids
    return scheduler


def test_throttle_prefills_excludes_fully_transferred_remote_kv():
    """A remote-KV resume whose whole prompt was transferred (no local prefill
    left, e.g. the decode side of P/D disaggregation) must NOT be throttled by
    the DP prefill cadence -- its single-token step has no prefill compute to
    defer, so delaying it would be pointless.
    """
    block_size = 16
    num_prompt = block_size * 2
    # Fully matched: the whole prompt is loaded remotely.
    scheduler = _setup_remote_kv_resume(num_prompt, matched_tokens=num_prompt)

    output = scheduler.schedule(throttle_prefills=True)
    assert "r2" in output.num_scheduled_tokens
    assert "r1" in output.num_scheduled_tokens


def test_throttle_prefills_defers_remote_kv_resume_with_local_prefill():
    """A remote-KV resume with local prefill still to compute (the connector
    only matched part of the prompt) IS throttled by the DP prefill cadence,
    like any other request doing local prefill compute this step.
    """
    block_size = 16
    num_prompt = block_size * 4
    # Half matched: the remaining half is local prefill compute.
    scheduler = _setup_remote_kv_resume(num_prompt, matched_tokens=num_prompt // 2)

    output = scheduler.schedule(throttle_prefills=True)
    assert "r2" not in output.num_scheduled_tokens  # deferred (has local prefill)
    assert "r1" in output.num_scheduled_tokens


def test_throttle_defers_inflight_prefill_chunk():
    """DP prefill balancing throttles ALL prefill compute on a throttled step,
    not just new admissions: an in-progress (chunked) prefill already in the
    running queue is also deferred, so the step runs decode-only, while a
    separate decode keeps being scheduled."""
    scheduler = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=50, enable_chunked_prefill=True
    )

    # A short request that finishes prefill in one step -> a running decode.
    (decode_req,) = create_requests(num_requests=1, num_tokens=4, req_ids=["dec0"])
    scheduler.add_request(decode_req)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["dec0"],
            req_id_to_index={"dec0": 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert decode_req in scheduler.running and not decode_req.is_prefill_chunk

    # A long request (80 tokens, budget 50) -> prefilled in chunks.
    (chunk_req,) = create_requests(num_requests=1, num_tokens=80, req_ids=["chk0"])
    scheduler.add_request(chunk_req)
    output = scheduler.schedule()  # first chunk of chk0 + decode of dec0
    assert output.num_scheduled_tokens["chk0"] > 0
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["dec0", "chk0"],
            req_id_to_index={"dec0": 0, "chk0": 1},
            sampled_token_ids=[[0], []],  # no token sampled for partial prefill
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert chunk_req.is_prefill_chunk  # still mid-prefill, in running

    # Throttled step: the in-flight prefill chunk is deferred, the decode runs.
    output = scheduler.schedule(throttle_prefills=True)
    assert "chk0" not in output.num_scheduled_tokens
    assert "dec0" in output.num_scheduled_tokens

    # When the cadence opens again, the prefill chunk resumes.
    output = scheduler.schedule()
    assert "chk0" in output.num_scheduled_tokens


def test_throttle_capacity_bound_guard_admits():
    """Saturation guard: if a cadence-aligned release step cannot drain the
    waiting prefill queue (it ran out of token budget), the throttle backs off on
    the next step so the backlog cannot grow into a TTFT avalanche -- prefills are
    admitted even though throttle_prefills is set."""
    scheduler = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=200, enable_chunked_prefill=True
    )
    a, b = create_requests(num_requests=2, num_tokens=200, req_ids=["a", "b"])
    scheduler.add_request(a)
    scheduler.add_request(b)

    # Release step (throttle off): `a` fills the 200-token budget; `b` cannot be
    # reached, so the waiting queue is not drained -> capacity-bound.
    output = scheduler.schedule()
    assert "a" in output.num_scheduled_tokens
    assert "b" not in output.num_scheduled_tokens
    assert scheduler.prefill_capacity_bound

    # Throttle. Because the previous release was capacity-bound, the guard backs
    # off and `b` is admitted rather than stalling the backlog.
    output = scheduler.schedule(throttle_prefills=True)
    assert "b" in output.num_scheduled_tokens


def test_no_mm_input_chunking():
    # Disable multimodal input chunking.
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        disable_chunked_mm_input=True,
        max_model_len=2048,
    )
    mm_positions = [[PlaceholderRange(offset=400, length=800)]]
    requests = create_requests(
        num_requests=1, num_tokens=1200, mm_positions=mm_positions
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    # We want to only see the 400 text tokens at the start scheduled
    assert output.num_scheduled_tokens[requests[0].request_id] == 400

    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(output.finished_req_ids) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 800

    # Test that we fail if we disable chunked mm input and use too small
    # of a max_num_batched_tokens for the mm input.
    with pytest.raises(ValueError):
        _ = create_scheduler(
            model="llava-hf/llava-1.5-7b-hf",
            max_num_batched_tokens=100,
            disable_chunked_mm_input=True,
        )


@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_schedule_concurrent_partial_requests(enable_prefix_caching: bool):
    """Test scheduling behavior with concurrent partial requests.

    This test verifies that: there are multiple long prefill requests in the
    RUNNING state, and we can schedule them together.

    """
    scheduler = create_scheduler(
        model="facebook/opt-125m",
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=3,
        num_tokens=800,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 3
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    # The first request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[0].request_id] == 400
    # The second request is scheduled partially - 400.
    assert output.num_scheduled_tokens[requests[1].request_id] == 400
    # The third request is also scheduled partially - 1024 - 400 - 400 = 224.
    assert output.num_scheduled_tokens[requests[2].request_id] == 224
    req_to_index = {request.request_id: i for i, request in enumerate(requests)}
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Schedule the next step. All three requests are running.
    # Processed the remaining prefills of the first and second requests.
    output1 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output1.scheduled_new_reqs) == 0
    assert output1.scheduled_cached_reqs.num_reqs == 3
    assert len(output1.finished_req_ids) == 0
    assert output1.num_scheduled_tokens[requests[0].request_id] == 400
    assert output1.num_scheduled_tokens[requests[1].request_id] == 400
    assert output1.num_scheduled_tokens[requests[2].request_id] == 224

    # Schedule the third step. All three requests are running.
    # First and second requests are in the decode stage.
    # All the remaining tokens in the third request are processed.
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id for request in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0], [0]] + [[] for _ in range(len(requests) - 2)],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output1, model_runner_output)
    output2 = scheduler.schedule()
    assert len(scheduler.running) == 3
    assert len(output2.scheduled_new_reqs) == 0
    assert output2.scheduled_cached_reqs.num_reqs == 3
    assert len(output2.finished_req_ids) == 0
    assert output2.num_scheduled_tokens[requests[0].request_id] == 1
    assert output2.num_scheduled_tokens[requests[1].request_id] == 1
    assert output2.num_scheduled_tokens[requests[2].request_id] == 800 - 224 - 224


def test_update_from_output_routes_sampling_masks_by_request():
    """Each request receives the sampler row at its own batch index."""
    scheduler = create_scheduler()
    scheduler.return_sampling_mask = True
    requests = create_requests(num_requests=3, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: 1 for req in requests},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[1], [3], [4]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        sampling_masks=SamplingMaskLists(
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int32), np.array([0, 2, 3, 6])
        ),
    )

    outputs = scheduler.update_from_output(scheduler_output, model_output)[0].outputs

    assert [out.request_id for out in outputs] == [req.request_id for req in requests]
    assert [out.new_sampling_mask.token_ids.tolist() for out in outputs] == [
        [1, 2],
        [3],
        [4, 5, 6],
    ]
    assert all(out.new_sampling_mask.offsets is None for out in outputs)


def test_ag2_concurrent_partial_prefill_shares_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "8")
    scheduler = create_scheduler(
        max_num_batched_tokens=6656,
        max_model_len=20000,
    )
    scheduler.elastic_on_demand_graphs = False
    scheduler.max_model_len = 20000
    requests = create_requests(num_requests=3, num_tokens=10000)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 6654
    assert output.num_scheduled_tokens == {
        request.request_id: 2218 for request in requests
    }
    assert list(scheduler.running) == requests
    assert not scheduler.waiting


def test_ag2_concurrent_partial_prefill_preserves_single_request_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "8")
    scheduler = create_scheduler(
        max_num_batched_tokens=6656,
        max_model_len=20000,
    )
    scheduler.elastic_on_demand_graphs = False
    scheduler.max_model_len = 20000
    request = create_requests(num_requests=1, num_tokens=10000)[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 6656}


def test_ag2_concurrent_partial_prefill_preserves_decode_priority(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "8")
    scheduler = create_scheduler(
        max_num_batched_tokens=6656,
        max_model_len=20000,
    )
    scheduler.elastic_on_demand_graphs = False
    scheduler.max_model_len = 20000
    decode = create_requests(
        num_requests=1,
        num_tokens=10,
        req_ids=["decode"],
    )[0]
    scheduler.add_request(decode)
    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        ModelRunnerOutput(
            req_ids=[decode.request_id],
            req_id_to_index={decode.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    prefills = create_requests(
        num_requests=3,
        num_tokens=10000,
        req_ids=["prefill-0", "prefill-1", "prefill-2"],
    )
    for request in prefills:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[decode.request_id] == 1
    assert {
        request.request_id: output.num_scheduled_tokens[request.request_id]
        for request in prefills
    } == {request.request_id: 2218 for request in prefills}
    assert output.total_num_scheduled_tokens == 6655


def test_ag2_concurrent_partial_prefill_keeps_requests_in_lockstep(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "8")
    scheduler = create_scheduler(
        max_num_batched_tokens=6656,
        max_model_len=20000,
    )
    scheduler.max_model_len = 20000
    requests = create_requests(num_requests=3, num_tokens=10000)
    for request in requests:
        scheduler.add_request(request)

    for _ in range(4):
        output = scheduler.schedule()
        assert set(output.num_scheduled_tokens) == {
            request.request_id for request in requests
        }
        assert len(set(output.num_scheduled_tokens.values())) == 1
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id for request in requests],
                req_id_to_index={
                    request.request_id: i for i, request in enumerate(requests)
                },
                sampled_token_ids=[
                    [0] if request.num_computed_tokens >= request.num_tokens else []
                    for request in requests
                ],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

    assert all(request.num_preemptions == 0 for request in requests)
    assert {request.num_computed_tokens for request in requests} == {8872}


def test_ag2_concurrent_partial_prefill_schedules_equal_tails(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "8")
    scheduler = create_scheduler(
        max_num_batched_tokens=6656,
        max_model_len=20000,
    )
    scheduler.max_model_len = 20000
    requests = create_requests(num_requests=3, num_tokens=5000)
    for request in requests:
        scheduler.add_request(request)

    for _ in range(2):
        output = scheduler.schedule()
        assert set(output.num_scheduled_tokens.values()) == {2218}
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id for request in requests],
                req_id_to_index={
                    request.request_id: i for i, request in enumerate(requests)
                },
                sampled_token_ids=[[] for _ in requests],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

    tail = scheduler.schedule()
    assert tail.num_scheduled_tokens == {
        request.request_id: 564 for request in requests
    }
    assert all(request.num_preemptions == 0 for request in requests)


def test_ag2_concurrent_partial_prefill_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", raising=False)
    scheduler = create_scheduler(
        max_num_batched_tokens=6656,
        max_model_len=20000,
    )
    scheduler.max_model_len = 20000
    requests = create_requests(num_requests=3, num_tokens=10000)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {requests[0].request_id: 6656}
    assert list(scheduler.running) == [requests[0]]
    assert list(scheduler.waiting) == requests[1:]


def test_ag2_concurrent_partial_prefill_reserves_full_isl_across_admissions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "2")
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=1024,
        num_blocks=100,
        block_size=16,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=60 * 16,
        max_tokens=1,
        block_size=16,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert list(output.num_scheduled_tokens) == [requests[0].request_id]
    assert list(scheduler.running) == [requests[0]]
    assert list(scheduler.waiting) == [requests[1]]
    assert requests[0].num_preemptions == requests[1].num_preemptions == 0


def test_ag2_concurrent_partial_prefill_admits_combined_full_isl_when_it_fits(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "2")
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=1024,
        num_blocks=100,
        block_size=16,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=40 * 16,
        max_tokens=1,
        block_size=16,
    )
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {
        request.request_id: 16 for request in requests
    }
    assert list(scheduler.running) == requests
    assert not scheduler.waiting


def test_ag2_canonical_prefill_admission_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_CANONICAL_PREFILL_ADMISSION", "1")
    with pytest.raises(ValueError, match="is retired"):
        create_scheduler(
            max_num_batched_tokens=7056,
            max_model_len=20000,
        )


def test_ag2_canonical_prefill_admission_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("AG2_VLLM_CANONICAL_PREFILL_ADMISSION", raising=False)
    scheduler = create_scheduler(
        max_num_batched_tokens=7056,
        max_model_len=20000,
    )
    requests = create_requests(num_requests=8, num_tokens=1513)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 7056
    assert output.num_scheduled_tokens[requests[4].request_id] == 1004
    assert list(scheduler.running) == requests[:5]
    assert list(scheduler.waiting) == requests[5:]


def test_ag2_concurrent_partial_prefill_admission_delay_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = object.__new__(Scheduler)
    scheduler._elastic_restore_mode = False
    scheduler.prefill_admission_delay_s = 0.05
    scheduler.prefill_admission_max_delay_s = 0.2
    scheduler.max_concurrent_partial_prefills = 8
    scheduler.max_num_running_reqs = 8
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.waiting = [
        Mock(
            num_computed_tokens=0,
            num_prompt_tokens=10000,
            execution_prefill_len=10000,
            arrival_time=100.0,
        )
    ]

    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.time", lambda: 100.1)
    assert scheduler._should_delay_waiting_prefill_admission() is True

    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.time", lambda: 100.201)
    assert scheduler._should_delay_waiting_prefill_admission() is False


def test_ag2_elastic_burst_delay_releases_complete_visible_wave(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = False
    scheduler.prefill_admission_delay_s = 0.002
    scheduler.prefill_admission_max_delay_s = 0.020
    scheduler.max_concurrent_partial_prefills = 64
    scheduler.max_num_running_reqs = 64
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.waiting = [
        Mock(
            num_computed_tokens=0,
            num_prompt_tokens=1513,
            execution_prefill_len=1513,
            arrival_time=100.000,
        ),
        Mock(
            num_computed_tokens=0,
            num_prompt_tokens=1513,
            execution_prefill_len=1513,
            arrival_time=100.001,
        ),
    ]

    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.time", lambda: 100.002)
    assert scheduler._should_delay_waiting_prefill_admission() is True

    # A partial burst remains open until the hard 20 ms bound, then the
    # existing final-wave preflight prices every request visible at that point
    # atomically rather than admitting an X1 prefix first.
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.time", lambda: 100.019)
    assert scheduler._should_delay_waiting_prefill_admission() is True
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.time", lambda: 100.021)
    assert scheduler._should_delay_waiting_prefill_admission() is False

    scheduler._elastic_restore_mode = True
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.time", lambda: 100.002)
    assert scheduler._should_delay_waiting_prefill_admission() is False

    scheduler._elastic_restore_mode = False
    scheduler.waiting = [scheduler.waiting[0]]
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.time", lambda: 100.019)
    assert scheduler._should_delay_waiting_prefill_admission() is True
    monkeypatch.setattr("vllm.v1.core.sched.scheduler.time.time", lambda: 100.021)
    assert scheduler._should_delay_waiting_prefill_admission() is False


def test_ag2_kv_tail_handoff_retains_kv_while_peers_progress(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "8")
    monkeypatch.setenv("AG2_VLLM_KV_TAIL_HANDOFF", "1")
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        max_model_len=512,
    )
    requests = create_requests(
        num_requests=3,
        num_tokens=160,
        max_tokens=1,
        req_ids=["blocked", "peer-1", "peer-2"],
    )
    for request in requests:
        scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=[request.request_id for request in requests],
            req_id_to_index={
                request.request_id: i for i, request in enumerate(requests)
            },
            sampled_token_ids=[[] for _ in requests],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    blocked = requests[0]
    blocked_tokens = blocked.num_computed_tokens
    blocked_blocks = scheduler.kv_cache_manager.get_blocks(
        blocked.request_id
    ).get_block_ids()

    original_allocate_slots = scheduler.kv_cache_manager.allocate_slots

    def reject_blocked_once(request, *args, **kwargs):
        if request is blocked:
            return None
        return original_allocate_slots(request, *args, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = reject_blocked_once
    handoff = scheduler.schedule()

    assert blocked.request_id not in handoff.num_scheduled_tokens
    assert set(handoff.num_scheduled_tokens) == {"peer-1", "peer-2"}
    assert blocked.status == RequestStatus.RUNNING
    assert blocked.num_computed_tokens == blocked_tokens
    assert (
        scheduler.kv_cache_manager.get_blocks(blocked.request_id).get_block_ids()
        == blocked_blocks
    )
    assert all(request.num_preemptions == 0 for request in requests)
    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.num_kv_tail_deferrals == 1

    scheduler.update_from_output(
        handoff,
        ModelRunnerOutput(
            req_ids=["peer-1", "peer-2"],
            req_id_to_index={"peer-1": 0, "peer-2": 1},
            sampled_token_ids=[[], []],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.kv_cache_manager.allocate_slots = original_allocate_slots
    recovered = scheduler.schedule()
    assert blocked.request_id in recovered.num_scheduled_tokens
    assert blocked.num_preemptions == 0


def test_ag2_kv_tail_handoff_falls_back_when_every_peer_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AG2_VLLM_MAX_CONCURRENT_PARTIAL_PREFILLS", "8")
    monkeypatch.setenv("AG2_VLLM_KV_TAIL_HANDOFF", "1")
    scheduler = create_scheduler(
        max_num_batched_tokens=96,
        max_model_len=512,
    )
    requests = create_requests(
        num_requests=3,
        num_tokens=160,
        max_tokens=1,
        req_ids=["first", "second", "victim"],
    )
    for request in requests:
        scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=[request.request_id for request in requests],
            req_id_to_index={
                request.request_id: i for i, request in enumerate(requests)
            },
            sampled_token_ids=[[] for _ in requests],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    original_allocate_slots = scheduler.kv_cache_manager.allocate_slots
    scheduler.kv_cache_manager.allocate_slots = lambda *args, **kwargs: None
    blocked = scheduler.schedule()

    assert not blocked.num_scheduled_tokens
    assert requests[0].num_preemptions == 0
    assert requests[1].num_preemptions == 0
    assert requests[2].num_preemptions == 1
    assert requests[2].status == RequestStatus.PREEMPTED
    assert requests[2].request_id in blocked.preempted_req_ids
    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.num_kv_tail_deferrals == 2

    scheduler.kv_cache_manager.allocate_slots = original_allocate_slots
    recovered = scheduler.schedule()
    assert recovered.total_num_scheduled_tokens > 0


def test_stop_via_update_from_output():
    """Test stopping behavior through update_from_output"""
    scheduler = create_scheduler(num_speculative_tokens=1)

    # Test case 1: Stop on EOS token
    requests = create_requests(num_requests=2, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 1, requests[1].request_id: 2},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [],
            requests[1].request_id: [10],
        },
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[
            [EOS_TOKEN_ID],
            [10, 11],
        ],  # First request hits EOS, second continues
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped, second continues
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID]
    assert list(requests[1].output_token_ids) == [10, 11]

    # Test case 2: Stop on custom stop token
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=10, stop_token_ids=[42, 43])
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3, requests[1].request_id: 2},
        total_num_scheduled_tokens=5,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [10, 42],
            requests[1].request_id: [13],
        },
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[10, 42, 12], [13, 14]],  # First request hits stop token
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped on custom token
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].stop_reason == 42
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 42]
    assert list(requests[1].output_token_ids) == [13, 14]

    # Test case 3: Stop on max tokens
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=2)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)
        req.status = RequestStatus.RUNNING

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3, requests[1].request_id: 1},
        total_num_scheduled_tokens=4,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [10, 11],
            requests[1].request_id: [],
        },
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[10, 11, 12], [13]],  # First request exceeds max_tokens
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped due to length
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 11]  # Truncated to max_tokens
    assert list(requests[1].output_token_ids) == [13]

    # Test case 4: Ignore EOS flag
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=1, max_tokens=10, ignore_eos=True)
    requests[0].num_computed_tokens = requests[0].num_tokens
    scheduler.requests[requests[0].request_id] = requests[0]
    scheduler.running.append(requests[0])

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={requests[0].request_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={requests[0].request_id: [EOS_TOKEN_ID, 10]},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID, 10, 11]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify request continues past EOS
    assert len(scheduler.running) == 1
    assert not requests[0].is_finished()
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID, 10, 11]


def test_check_stop_min_tokens():
    """Test that sampled stop tokens still finish under min_tokens."""
    from vllm.v1.core.sched.utils import check_stop

    # Test case 1: regular tokens below min_tokens should keep running.
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=5,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.append_output_token_ids([10, 11, 12])

    result = check_stop(request, max_model_len=100)
    assert result is False, "Should not stop when num_output_tokens<min_tokens"

    # Test case 2: if EOS is sampled, it should stop even below min_tokens.
    request_eos = Request(
        request_id="eos",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request_eos.append_output_token_ids([10, 11, EOS_TOKEN_ID])
    result = check_stop(request_eos, max_model_len=100)
    assert result is True, "Sampled EOS should stop even below min_tokens"
    assert request_eos.status == RequestStatus.FINISHED_STOPPED

    # Test case 3: ignore_eos should still prevent EOS-based stopping.
    sampling_params_ignore_eos = SamplingParams(
        ignore_eos=True,
        max_tokens=20,
        min_tokens=5,
    )
    sampling_params_ignore_eos.update_from_generation_config({}, EOS_TOKEN_ID)
    request_ignore_eos = Request(
        request_id="ignore-eos",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_ignore_eos,
        pooling_params=None,
    )
    request_ignore_eos.append_output_token_ids([EOS_TOKEN_ID])
    result = check_stop(request_ignore_eos, max_model_len=100)
    assert result is False, "ignore_eos should still suppress EOS stopping"

    # Test case 4: min_tokens = 0 should follow normal stopping logic.
    sampling_params_no_min = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=0,
    )
    sampling_params_no_min.update_from_generation_config({}, EOS_TOKEN_ID)
    request_no_min = Request(
        request_id="1",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_no_min,
        pooling_params=None,
    )
    request_no_min.append_output_token_ids([10, EOS_TOKEN_ID])

    result = check_stop(request_no_min, max_model_len=100)
    assert result is True, "Should stop on EOS when min_tokens=0"
    assert request_no_min.status == RequestStatus.FINISHED_STOPPED

    # Test case 5: if a stop token is sampled, it should stop immediately.
    sampling_params_stop = SamplingParams(
        ignore_eos=False,
        max_tokens=20,
        min_tokens=5,
        stop_token_ids=[42],
    )
    sampling_params_stop.update_from_generation_config({}, EOS_TOKEN_ID)
    request_stop = Request(
        request_id="2",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_stop,
        pooling_params=None,
    )
    request_stop.append_output_token_ids([10, 11, 42])
    result = check_stop(request_stop, max_model_len=100)
    assert result is True, "Sampled stop tokens should stop even below min_tokens"
    assert request_stop.status == RequestStatus.FINISHED_STOPPED
    assert request_stop.stop_reason == 42


def test_check_stop_length_cap_below_min_tokens():
    """Length caps must finish a request even below min_tokens.

    Guards against requests hanging forever when min_tokens cannot be
    satisfied: min_tokens only defers EOS/stop tokens, so a request that
    reaches max_model_len (or a max_tokens clamped below min_tokens) must
    still finish as FINISHED_LENGTH_CAPPED instead of spinning in the
    scheduler with zero new tokens to schedule.
    """
    from vllm.v1.core.sched.utils import check_stop

    # Reaching max_model_len below min_tokens must stop the request.
    sampling_params = SamplingParams(
        ignore_eos=True,
        max_tokens=500,
        min_tokens=500,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    # 7 output tokens: num_tokens == 10 == max_model_len, min_tokens unmet.
    request.append_output_token_ids([10, 11, 12, 13, 14, 15, 16])

    result = check_stop(request, max_model_len=10)
    assert result is True, "Should stop at max_model_len even below min_tokens"
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    # Reaching max_tokens below min_tokens must stop the request. This can
    # happen when max_tokens is clamped to the context window after
    # SamplingParams validation (e.g. max_tokens=None on a long prompt).
    sampling_params_capped = SamplingParams(
        ignore_eos=True,
        max_tokens=500,
        min_tokens=500,
    )
    sampling_params_capped.max_tokens = 3
    sampling_params_capped.update_from_generation_config({}, EOS_TOKEN_ID)
    request_capped = Request(
        request_id="1",
        prompt_token_ids=[0, 1, 2],
        sampling_params=sampling_params_capped,
        pooling_params=None,
    )
    request_capped.append_output_token_ids([10, 11, 12])

    result = check_stop(request_capped, max_model_len=100)
    assert result is True, "Should stop at max_tokens even below min_tokens"
    assert request_capped.status == RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.parametrize(
    "enable_prefix_caching, prompt_logprobs",
    [
        (False, None),
        (True, 5),
    ],
)
def test_schedule_concurrent_batches(
    enable_prefix_caching: bool, prompt_logprobs: int | None
):
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=512,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert scheduler_output0.num_scheduled_tokens[requests[0].request_id] == 512

    # The first request is still running, so only schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert scheduler_output1.num_scheduled_tokens[requests[1].request_id] == 512

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Schedule the next step.
    # The first request can be scheduled again while the second
    # request is still running.
    scheduler_output2 = scheduler.schedule()
    assert scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1

    # Model output of the second request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output)


@pytest.mark.parametrize("enable_chunked_prefill", [True, False])
def test_schedule_order(enable_chunked_prefill: bool):
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=3,
        enable_chunked_prefill=enable_chunked_prefill,
    )

    # long requests
    requests = create_requests(num_requests=2, num_tokens=800, req_ids=["1", "2"])
    # short requests
    requests += create_requests(num_requests=2, num_tokens=10, req_ids=["3", "4"])

    for request in requests:
        scheduler.add_request(request)

    scheduler_output1 = scheduler.schedule()

    if enable_chunked_prefill:
        # When enable chunked prefill, long requests will be chunked.
        assert len(scheduler_output1.scheduled_new_reqs) == 2
    else:
        # When disable chunked prefill, should not skip the long requests,
        # and scheduling subsequent short requests in advance,
        # even though there is still token budgets remaining.
        assert len(scheduler_output1.scheduled_new_reqs) == 1


def test_preempt_during_execution():
    # NOTE(woosuk): The actual number of available blocks is 10 instead of 11
    # because block 0 is reserved as the null block.
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,
        enable_prefix_caching=False,
    )
    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.num_scheduled_tokens) == 1
    assert len(scheduler_output0.scheduled_new_reqs[0].block_ids[0]) == 5

    # Schedule the second request while the first request is still running.
    # This scenario can occur in certain cases, when max_concurrent_batches > 1
    # (e.g., when pipeline parallelism is used).
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.num_scheduled_tokens) == 1
    assert len(scheduler_output1.scheduled_new_reqs[0].block_ids[0]) == 5

    # Get the output of the first request.
    model_runner_output0 = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output0)

    # Schedule the first request again. This will cause the preemption
    # of the second request because the KV cache is full.
    _ = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert scheduler.running[0] == requests[0]
    assert requests[1].status == RequestStatus.PREEMPTED

    model_runner_output1 = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output1)

    # The second request (that is preempted) should be updated with the
    # sampled token id.
    assert len(requests[1].output_token_ids) == 1
    assert requests[1].output_token_ids[0] == 42


def test_prefix_cache_query_not_inflated_by_connector_defer():
    """The GPU prefix-cache query is recorded at admission, so a request the
    connector defers several times is counted once, not once per retry."""
    num_defers_before_matching = 3
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=0,
            is_async=False,
            num_defers_before_matching=num_defers_before_matching,
        ),
    )
    request = create_requests(num_requests=1, num_tokens=32, block_size=16)[0]
    scheduler.add_request(request)

    # Each deferred step re-runs the lookup but records nothing.
    for _ in range(num_defers_before_matching):
        assert not scheduler.schedule().scheduled_new_reqs

    output = scheduler.schedule()
    assert any(r.req_id == request.request_id for r in output.scheduled_new_reqs)

    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert stats.requests == 1
    assert stats.queries == request.num_tokens


def test_preemption_re_records_prefix_cache_query():
    """A preempted request re-enters the lookup on resume, so its recomputation
    is counted again into the preempted stats."""
    scheduler = create_scheduler(enable_prefix_caching=True)
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.preempted_requests) == (1, 0)

    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    assert request.status == RequestStatus.PREEMPTED

    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert request.num_stale_output_tokens == 0
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None

    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert stats.preempted_requests == 1


def test_prefix_cache_stats_not_recorded_when_caching_disabled():
    """With prefix caching off there is no local lookup, so admitting a request
    records no phantom miss."""
    scheduler = create_scheduler(enable_prefix_caching=False)
    for request in create_requests(num_requests=2):
        scheduler.add_request(request)

    scheduler.schedule()

    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)


def test_prefix_cache_stats_counted_once_for_retried_then_scheduled_request():
    """A real cache hit rejected once by allocate_slots and admitted on the next
    step is counted exactly once, hits included."""
    block_size = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        block_size=block_size,
    )

    # Seed the cache so the next request with the same prompt hits it.
    seed = create_requests(
        num_requests=1,
        num_tokens=block_size * 2,
        max_tokens=2,
        same_prompt=True,
        block_size=block_size,
        req_ids=["seed"],
    )[0]
    scheduler.add_request(seed)
    _step_until_done(
        scheduler,
        scheduler.schedule(),
        ModelRunnerOutput(
            req_ids=["seed"],
            req_id_to_index={"seed": 0},
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    # The seeding step swapped in a fresh accumulator, so re-read it; the
    # retried request must be the only thing recorded from here on.
    stats = scheduler.kv_cache_manager.prefix_cache_stats
    assert stats is not None
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)

    retried = create_requests(
        num_requests=1,
        num_tokens=block_size * 3,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["retried"],
    )[0]
    scheduler.add_request(retried)

    # Reject the first allocation attempt, then delegate to the real one.
    orig_allocate_slots = scheduler.kv_cache_manager.allocate_slots
    allocate_results: list = []

    def spy_allocate_slots(*args, **kwargs):
        result = None if not allocate_results else orig_allocate_slots(*args, **kwargs)
        allocate_results.append(result)
        return result

    scheduler.kv_cache_manager.allocate_slots = spy_allocate_slots

    assert not scheduler.schedule().scheduled_new_reqs
    assert (stats.requests, stats.queries, stats.hits) == (0, 0, 0)

    assert "retried" in scheduler.schedule().num_scheduled_tokens
    assert allocate_results[0] is None and allocate_results[1] is not None
    assert (stats.requests, stats.queries, stats.hits) == (
        1,
        retried.num_tokens,
        block_size * 2,
    )


def test_scheduler_reset_prefix_cache():
    scheduler = create_scheduler(enable_prefix_caching=True)
    requests = create_requests(num_requests=10)
    for request in requests:
        scheduler.add_request(request)

    # Initial scheduling, requests should be at the running state now
    _ = scheduler.schedule()

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert scheduler.running[i] == request

    # Reset prefix cache should fail since there are still running requests
    # and they are taking KV cache
    assert not scheduler.reset_prefix_cache()

    # Reset prefix cache with reset_running_requests=True. All running requests
    # Should be pushed back to the waiting queue and kv cache should be freed
    assert scheduler.reset_prefix_cache(reset_running_requests=True)

    # Verify requests moved from running to waiting
    assert len(scheduler.waiting) == len(requests)
    assert len(scheduler.running) == 0

    for i, request in enumerate(requests):
        assert scheduler.waiting[i] == request


def test_reset_connector_cache_no_connector_is_no_op_success():
    """``reset_connector_cache`` must return True when no connector is
    configured.

    Without this, ``reset_prefix_cache(reset_connector=True)`` returns
    ``False`` on every engine that doesn't have a KV connector configured —
    even when the local prefix cache reset succeeded — and any caller that
    interprets the return value as "did the reset I asked for succeed?"
    sees a spurious failure.
    """
    scheduler = create_scheduler(enable_prefix_caching=True)
    assert scheduler.connector is None

    # No-connector reset is treated as success.
    assert scheduler.reset_connector_cache() is True

    # End-to-end: reset_prefix_cache(reset_connector=True) on an idle
    # scheduler succeeds with or without a connector.
    assert scheduler.reset_prefix_cache(reset_connector=True) is True


def test_draft_slots_budgeted_per_scheduled_request(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    (tmp_path / "config.json").write_text(
        '{"architectures": ["OPTForCausalLM"], "model_type": "opt"}'
    )
    scheduler = create_scheduler(
        model=str(tmp_path),
        max_num_seqs=16,
        max_num_batched_tokens=20,
        num_speculative_tokens=4,
        parallel_drafting=True,
        skip_tokenizer_init=True,
    )
    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    assert scheduler.max_num_scheduled_tokens == 20
    assert speculative_config.max_num_new_slots_for_drafting == 3

    for request in create_requests(num_requests=2, num_tokens=10):
        scheduler.add_request(request)

    assert scheduler.schedule().num_scheduled_tokens == {"0": 10, "1": 4}


# Note - these test cases mirror some of those in test_rejection_sampler.py
@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected,expected_per_req",
    [
        ([[1, 2, 3]], [[1, 2, 3, 4]], (1, 3, 3, [1, 1, 1]), [[0, 0, 0, 1]]),  # perfect
        ([[1, 2, 3]], [[1, 5]], (1, 3, 1, [1, 0, 0]), [[0, 1, 0, 0]]),  # early mismatch
        (
            [[1, 2], [3]],
            [[1, 2, 5], [3, 4]],
            (2, 3, 3, [2, 1]),
            [[0, 0, 1], [0, 1, 0]],
        ),  # multiple sequences
        ([[1]], [[1, 2]], (1, 1, 1, [1]), [[0, 1]]),  # single token sequence
        ([[]], [[5]], (0, 0, 0, [0]), [[0, 0]]),  # empty sequence -> empty accumulator
        (
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 7], [4, 8]],
            (2, 6, 3, [2, 1, 0]),
            [[0, 0, 1, 0], [0, 1, 0, 0]],
        ),  # multiple mismatches
    ],
)
def test_schedule_spec_decoding_stats(
    spec_tokens, output_tokens, expected, expected_per_req
):
    """Test scheduling behavior with speculative decoding.

    This test verifies that:
    1. Speculated tokens get scheduled correctly
    2. The aggregate SpecDecodingStats count draft and accepted tokens
    3. The per-request accumulator (enabled via per_request_spec_decode_metrics)
       buckets the same acceptance by accepted draft count (j)
    """
    num_spec_tokens = max(1, max(len(t) for t in spec_tokens))
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec_tokens,
        per_request_spec_decode_metrics="summary",
    )
    requests = create_requests(num_requests=len(spec_tokens), num_tokens=1)
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    # Schedule a decode, which will also draft speculative tokens
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.total_num_scheduled_tokens == len(requests)
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1
        assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0] for _ in range(len(requests))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)
    draft_token_ids = DraftTokenIds(req_ids, spec_tokens)
    scheduler.update_draft_token_ids(draft_token_ids)

    for i in range(len(requests)):
        running_req = scheduler.running[i]
        # The prompt token
        assert running_req.num_computed_tokens == 1
        # The prompt token and the sampled token
        assert running_req.num_tokens == 2
        # The prompt token, the sampled token, and the speculated tokens
        assert running_req.num_tokens_with_spec == 2 + len(spec_tokens[i])

    # No draft or accepted tokens counted yet
    assert not engine_core_outputs or (
        engine_core_outputs[0].scheduler_stats.spec_decoding_stats is None
    )

    # Schedule the speculated tokens for validation
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    # The sampled token and speculated tokens
    assert output.total_num_scheduled_tokens == len(requests) + sum(
        len(ids) for ids in spec_tokens
    )
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1 + len(spec_tokens[i])
        if spec_tokens[i]:
            assert len(output.scheduled_spec_decode_tokens[req_id]) == len(
                spec_tokens[i]
            )
        else:
            assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=output_tokens,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    scheduler_stats = (
        engine_core_outputs[0].scheduler_stats if engine_core_outputs else None
    )
    if expected[0] == 0:
        assert scheduler_stats is not None
        assert scheduler_stats.spec_decoding_stats is None
    else:
        assert scheduler_stats is not None
        assert scheduler_stats.spec_decoding_stats is not None
        stats = scheduler_stats.spec_decoding_stats
        assert stats.num_drafts == expected[0]
        assert stats.num_draft_tokens == expected[1]
        assert stats.num_accepted_tokens == expected[2]
        assert stats.num_accepted_tokens_per_pos == expected[3]

    # Per-request accumulator: the same acceptance, bucketed by accepted draft
    # count (j) on each request rather than summed across the batch. The
    # accumulator is created eagerly on add_request, so every request has one
    # (an empty histogram when it drafted nothing).
    for i, req_id in enumerate(req_ids):
        payload = scheduler.requests[req_id].spec_decode_metrics.to_dict()
        assert payload["acceptance_histogram"] == expected_per_req[i]
        assert payload["num_draft_tokens"] == len(spec_tokens[i])
        assert "per_step_accepted" not in payload  # summary level


def _run_spec_verify_steps(scheduler, rounds, num_invalid_per_round=None):
    """Drive prefill + one draft/verify step per round for a single request.

    ``rounds`` is a list of ``(spec_token_ids, output_token_ids)`` -- one verify
    step each. ``num_invalid_per_round`` optionally injects the grammar-invalid
    draft count structured-output decoding would set on the verify output.
    Returns the request.
    """
    [req] = create_requests(num_requests=1, num_tokens=1)
    scheduler.add_request(req)
    rid = req.request_id
    req_to_index = {rid: 0}

    def _mk_output(sampled):
        return ModelRunnerOutput(
            req_ids=[rid],
            req_id_to_index=req_to_index,
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    # Prefill: sample one token.
    scheduler.update_from_output(scheduler.schedule(), _mk_output([[0]]))
    for i, (spec, out) in enumerate(rounds):
        scheduler.update_draft_token_ids(DraftTokenIds([rid], [spec]))
        output = scheduler.schedule()
        if num_invalid_per_round is not None and num_invalid_per_round[i]:
            output.num_invalid_spec_tokens = {rid: num_invalid_per_round[i]}
        scheduler.update_from_output(output, _mk_output([out]))
    return req


def test_per_request_spec_decode_detailed_records_per_step():
    scheduler = create_scheduler(
        num_speculative_tokens=3,
        per_request_spec_decode_metrics="detailed",
    )
    # Three verify steps for one request; num_accepted = len(output) - 1, so the
    # outputs below accept 3, 1, then 0 drafts across the steps.
    req = _run_spec_verify_steps(
        scheduler,
        [
            ([1, 2, 3], [1, 2, 3, 4]),  # accept 3
            ([5, 6, 7], [5, 8]),  # accept 1
            ([9, 10, 11], [12]),  # accept 0
        ],
    )
    payload = scheduler.requests[req.request_id].spec_decode_metrics.to_dict()
    assert payload["per_step_accepted"] == [3, 1, 0]
    assert payload["per_step_drafted"] == [3, 3, 3]
    assert payload["num_spec_steps"] == 3


def test_per_request_spec_decode_subtracts_invalid_drafts():
    # Grammar-invalidated drafts (num_invalid_spec_tokens, set by structured
    # output) are excluded from the proposed count, mirroring the aggregate.
    scheduler = create_scheduler(
        num_speculative_tokens=3,
        per_request_spec_decode_metrics="summary",
    )
    # One verify step: 3 drafted, 1 grammar-invalid, output accepts 2.
    req = _run_spec_verify_steps(
        scheduler,
        [([1, 2, 3], [1, 2, 5])],
        num_invalid_per_round=[1],
    )
    payload = scheduler.requests[req.request_id].spec_decode_metrics.to_dict()
    assert payload["num_draft_tokens"] == 2  # 3 drafted - 1 invalid
    assert payload["num_accepted_draft_tokens"] == 2  # len([1,2,5]) - 1
    assert payload["acceptance_histogram"] == [0, 0, 1, 0]


def test_per_request_spec_decode_acceptance_disabled_by_default():
    scheduler = create_scheduler(num_speculative_tokens=3)
    assert scheduler.spec_decode_metrics_level == "none"
    req = _run_spec_verify_steps(scheduler, [([1, 2, 3], [1, 2, 3, 4])])
    assert scheduler.requests[req.request_id].spec_decode_metrics is None


def test_spec_decoding_stats_empty_output():
    """Test that spec decoding stats handle empty output tokens gracefully.

    This is a regression test for a bug where empty sampled_token_ids
    would cause num_accepted = len([]) - 1 = -1, leading to a
    ValueError when incrementing a Prometheus counter with a negative value.
    """
    num_spec_tokens = 3
    scheduler = create_scheduler(num_speculative_tokens=num_spec_tokens)
    requests = create_requests(num_requests=1, num_tokens=1)
    request = requests[0]
    req_id = request.request_id

    scheduler.add_request(request)

    # Initial schedule (prefill)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1

    # Complete the prefill with a sampled token
    model_runner_output = ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Add draft tokens for speculation
    draft_token_ids = DraftTokenIds([req_id], [[1, 2, 3]])
    scheduler.update_draft_token_ids(draft_token_ids)

    # Schedule the speculated tokens for validation
    output = scheduler.schedule()
    assert req_id in output.scheduled_spec_decode_tokens
    assert len(output.scheduled_spec_decode_tokens[req_id]) == 3

    # Simulate empty output tokens (e.g., due to request abortion or error)
    # This would previously cause num_accepted = -1 and crash
    model_runner_output = ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[]],  # Empty output tokens
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # This should not raise an error
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    # Spec decoding stats should be None since no tokens were generated
    scheduler_stats = (
        engine_core_outputs[0].scheduler_stats if engine_core_outputs else None
    )
    assert scheduler_stats is None or scheduler_stats.spec_decoding_stats is None


def test_no_spec_tokens_scheduled_for_prefill_chunks():
    """Test that draft tokens are ignored for prefill chunk requests.

    When a request is being prefilled in chunks (chunked prefill), draft tokens
    from `update_draft_token_ids` should be ignored until the prefill is complete.

    The bug manifests when:
    - A prefill chunk is scheduled
    - Draft tokens are provided via update_draft_token_ids
    - The next schedule has enough budget to include spec tokens

    Without the fix, spec tokens would incorrectly be scheduled with the
    remaining prefill tokens. With the fix, draft tokens are ignored for
    prefill chunks.
    """
    num_spec_tokens = 3
    # Use budget of 50, with 80 token prompt:
    # - First chunk: 50 tokens
    # - Second chunk: 30 remaining + potentially 3 spec tokens = 33
    # Without fix: num_scheduled_spec_tokens = 33 + 50 - 80 = 3 (BUG!)
    # With fix: spec_token_ids cleared, so no spec tokens scheduled
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec_tokens,
        max_num_batched_tokens=50,
        enable_chunked_prefill=True,
    )
    requests = create_requests(num_requests=1, num_tokens=80)
    req = requests[0]
    scheduler.add_request(req)

    # First schedule - prefill chunk (50 of 80 tokens)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.num_scheduled_tokens[req.request_id] == 50

    # Update from output (no sampled token since still prefilling)
    req_to_index = {req.request_id: 0}
    model_runner_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Provide draft tokens while request is still in prefill.
    # The fix ensures these are ignored for prefill chunks.
    draft_token_ids = DraftTokenIds([req.request_id], [[1, 2, 3]])
    scheduler.update_draft_token_ids(draft_token_ids)

    # Second schedule - remaining 30 tokens of prefill
    output = scheduler.schedule()
    # KEY ASSERTION: Should schedule exactly the remaining 30 prefill tokens,
    # NOT 33 (30 + 3 spec). Without the fix, this would be 33.
    assert output.num_scheduled_tokens[req.request_id] == 30, (
        f"Expected 30 tokens (remaining prefill only), "
        f"got {output.num_scheduled_tokens[req.request_id]}. "
        "Spec tokens should not be scheduled with prefill chunks."
    )
    # No spec tokens should be in the output
    assert req.request_id not in output.scheduled_spec_decode_tokens, (
        "Spec tokens should not be scheduled with prefill chunks"
    )

    # Update from output with a sampled token (prefill complete)
    model_runner_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index=req_to_index,
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_runner_output)

    # Now provide draft tokens - should be accepted since prefill is complete
    draft_token_ids = DraftTokenIds([req.request_id], [[1, 2, 3]])
    scheduler.update_draft_token_ids(draft_token_ids)

    # spec_token_ids SHOULD be set after prefill is complete
    assert req.spec_token_ids == [1, 2, 3], (
        f"spec_token_ids should be set after prefill, got {req.spec_token_ids}"
    )

    # Third schedule - decode phase with spec tokens
    output = scheduler.schedule()
    # 1 new token + 3 spec tokens = 4
    assert output.num_scheduled_tokens[req.request_id] == 4
    assert req.request_id in output.scheduled_spec_decode_tokens
    assert len(output.scheduled_spec_decode_tokens[req.request_id]) == num_spec_tokens


def _model_output(scheduler, output, sampled):
    """Feed `sampled` (per-request list) back to the scheduler."""
    req_ids = list(output.num_scheduled_tokens.keys())
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_spec_decode_padding_first_decode_step():
    """A request taking its first decode step (whole prompt already computed via
    a prefix-cache hit) is padded with placeholder (-1) spec tokens so it enters
    the worker with the same 1 + num_spec_tokens shape as the other speculative
    decodes, keeping the batch uniform.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
    )
    # Two identical 33-token prompts: 2 full blocks (32 tokens) get cached, so a
    # second identical request hits num_computed == num_prompt_tokens - 1.
    r1, r2 = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=16
    )

    # Drive r1 through prefill so its prompt blocks are cached, then give it real
    # drafts so it is a running speculative decode (1 + num_spec shape).
    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, out, [[100]])
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # r2 arrives; its whole prompt is a prefix-cache hit -> first decode step.
    scheduler.add_request(r2)
    out = scheduler.schedule()

    # r1 verifies its real drafts.
    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    # r2 is padded to the 1 + num_spec shape with placeholder (-1) drafts.
    assert out.num_scheduled_tokens[r2.request_id] == 1 + num_spec
    assert out.scheduled_spec_decode_tokens[r2.request_id] == [-1] * num_spec


def test_spec_decode_padding_resumed_request_without_running_requests():
    """Pad a synchronously resumed request even without local running work."""
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        use_kv_connector=mock_kv(matched_tokens=32, is_async=False),
    )
    (resumed,) = create_requests(num_requests=1, num_tokens=33, max_tokens=16)

    scheduler.add_request(resumed)
    out = scheduler.schedule()

    assert out.num_scheduled_tokens[resumed.request_id] == 1 + num_spec
    assert out.scheduled_spec_decode_tokens[resumed.request_id] == [-1] * num_spec


def test_spec_decode_padding_skipped_for_diffusion():
    """Diffusion spec tokens are the fixed-size denoising canvas, not
    rejectable drafts: a first-decode-step request must keep its 1-token span
    instead of being padded to 1 + num_spec_tokens, which would overflow the
    canvas.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
    )
    # Diffusion schedulers initialize this to 0 (model_config.is_diffusion).
    scheduler.num_sampled_tokens_per_step = 0
    r1, r2 = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=16
    )

    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, out, [[100]])
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # r2 arrives; its whole prompt is a prefix-cache hit -> needs exactly
    # 1 token while r1 is a running speculative decode.
    scheduler.add_request(r2)
    out = scheduler.schedule()

    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    # r2 keeps its true 1-token span; no placeholder drafts are attached.
    assert out.num_scheduled_tokens[r2.request_id] == 1
    assert r2.request_id not in out.scheduled_spec_decode_tokens


@pytest.mark.parametrize("aligned_tokens", [1, 2, 3])
def test_spec_decode_padding_dropped_when_recurrent_alignment_clips(
    aligned_tokens: int,
):
    """A clipped padded request must drop its padding, not shorten it.

    The sampler derives a request's row window from its draft count, so a
    padded request that keeps fewer than 1 + num_spec query rows would select
    the preceding request's hidden states.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
    )
    r1, r2 = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=16
    )

    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r1.request_id] == 33
    _model_output(scheduler, out, [[100]])
    scheduler.update_draft_token_ids(DraftTokenIds([r1.request_id], [[1, 2, 3]]))

    # Model the recurrent align split that clips a padded 1+3 prompt tail at
    # the next cache boundary. The real QSA block size produces all three cases
    # depending on prompt length modulo four.
    scheduler.need_mamba_block_aligned_split = True
    original_split = scheduler._mamba_block_aligned_split

    def align_prompt_tail(request, num_new_tokens, *args):
        if request is r2:
            return aligned_tokens
        return original_split(request, num_new_tokens, *args)

    scheduler._mamba_block_aligned_split = align_prompt_tail
    scheduler.add_request(r2)
    out = scheduler.schedule()

    assert out.scheduled_spec_decode_tokens[r1.request_id] == [1, 2, 3]
    assert out.num_scheduled_tokens[r2.request_id] == 1
    assert r2.request_id not in out.scheduled_spec_decode_tokens


def test_spec_decode_padding_skipped_with_prefill_in_batch():
    """Padding is skipped when the batch contains a prefill chunk: the batch is
    already mixed/non-uniform, so padding a new decode request buys nothing.
    """
    num_spec = 3
    scheduler = create_scheduler(
        num_speculative_tokens=num_spec,
        enable_prefix_caching=True,
        block_size=16,
        max_num_batched_tokens=64,
    )
    # r_warm + r_candidate share a prompt so r_candidate gets a full prefix hit.
    r_warm, r_candidate = create_requests(
        num_requests=2, num_tokens=33, same_prompt=True, max_tokens=1
    )
    # r_long has a different, long prompt that prefills over multiple chunks.
    (r_long,) = create_requests(num_requests=1, num_tokens=100, max_tokens=16)

    # Warm the prefix cache with r_warm's prompt (it finishes; blocks stay cached).
    scheduler.add_request(r_warm)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[r_warm.request_id] == 33
    _model_output(scheduler, out, [[100]])
    assert r_warm.request_id in scheduler.finished_req_ids

    # Start r_long; after one chunk it remains a prefill chunk in the running queue.
    scheduler.add_request(r_long)
    out = scheduler.schedule()
    _model_output(scheduler, out, [[]])  # still prefilling, no sampled token
    assert r_long.is_prefill_chunk

    # r_candidate arrives (prefix-cache hit -> first decode step) alongside the
    # in-flight prefill chunk.
    scheduler.add_request(r_candidate)
    out = scheduler.schedule()

    # The batch has a prefill chunk, so r_candidate is NOT padded.
    assert r_long.request_id in out.num_scheduled_tokens
    assert out.num_scheduled_tokens[r_candidate.request_id] == 1
    assert r_candidate.request_id not in out.scheduled_spec_decode_tokens


def test_scheduler_stats_waiting_queues():
    """Test that scheduler stats correctly report waiting and skipped_waiting queues."""
    # Create scheduler with limited capacity so we can have waiting requests
    scheduler = create_scheduler(max_num_batched_tokens=100)

    # Create requests: some will be scheduled, some will wait on capacity,
    # and some will be blocked by constraints
    all_requests = create_requests(num_requests=5, num_tokens=50)

    # Add 3 requests - only 2 can be scheduled (2 * 50 = 100 tokens)
    # The 3rd will remain in waiting queue (capacity constraint)
    for request in all_requests[:3]:
        scheduler.add_request(request)

    # Manually add 2 more to skipped_waiting to simulate constraint-blocked
    for request in all_requests[3:]:
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler.skipped_waiting.add_request(request)

    # Schedule - this will schedule 2 requests, leaving 1 in waiting
    output = scheduler.schedule()

    # Verify: 2 scheduled, 1 still waiting on capacity, 2 blocked by constraints
    assert len(output.scheduled_new_reqs) == 2
    assert len(scheduler.waiting) == 1
    assert len(scheduler.skipped_waiting) == 2

    # Call update_from_output() to get frontend-facing stat
    scheduled_req_ids = list(output.num_scheduled_tokens.keys())
    model_runner_output = ModelRunnerOutput(
        req_ids=scheduled_req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(scheduled_req_ids)},
        sampled_token_ids=[[1]] * len(scheduled_req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)
    assert engine_core_outputs and len(engine_core_outputs) > 0
    stats = engine_core_outputs[0].scheduler_stats
    assert stats is not None

    # Verify stats match queue lengths after scheduling
    assert stats.num_running_reqs == 2  # 2 were scheduled
    assert stats.num_waiting_reqs == 1  # 1 waiting on capacity
    assert stats.num_skipped_waiting_reqs == 2  # 2 blocked by constraints


def _assert_right_scheduler_output(
    output: SchedulerOutput,
    num_requests: int,
    expected_num_scheduled_tokens: int,
):
    """Check if SchedulerOutput is correct after remote KV cache hit."""

    # We should inject the kv_connector_metadata.
    assert len(output.kv_connector_metadata.requests) == num_requests

    # Only num_tokens - matched_num_new_tokens should be scheduled.
    for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
        assert num_scheduled_tokens == expected_num_scheduled_tokens


def _assert_right_kv_cache_manager(
    scheduler: Scheduler,
    requests: list[Request],
    num_tokens: int,
    block_size: int,
    num_requests: int,
    num_total_blocks: int,
):
    """Check whether KVCacheManager is correct after allocate."""

    # Make sure the request stats are right.
    EXPECTED_TOTAL_BLOCKS = num_tokens // block_size
    for req in requests:
        blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
            0
        ].req_to_blocks[req.request_id]
        hashes = req.block_hashes
        assert (
            scheduler.kv_cache_manager.coordinator.single_type_managers[
                0
            ].num_cached_block[req.request_id]
            == EXPECTED_TOTAL_BLOCKS
        )
        assert len(blocks) == EXPECTED_TOTAL_BLOCKS
        assert len(hashes) == EXPECTED_TOTAL_BLOCKS

    # Make sure we actually touched all the blocks.
    BLOCKS_PER_REQ = num_tokens / block_size
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        == num_total_blocks - num_requests * BLOCKS_PER_REQ
    )


def _step_until_done(
    scheduler: Scheduler,
    output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
):
    """Loop over schedule(), update_from_output() until finished."""

    all_finished = False
    _ = scheduler.update_from_output(output, model_runner_output)
    while not all_finished:
        # Schedule + a few iterations until stopping.
        output = scheduler.schedule()
        assert len(scheduler.running)
        for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
            # We should be in the decode phase now.
            assert num_scheduled_tokens == 1
        if scheduler.connector is not None:
            assert len(output.kv_connector_metadata.requests) == 0
        if scheduler.ec_connector is not None:
            assert len(output.ec_connector_metadata.mm_datas) == 0
        ecos = scheduler.update_from_output(output, model_runner_output)[0]
        all_done = True
        for eco in ecos.outputs:
            if eco.finish_reason is None:
                all_done = False
        all_finished = all_done


def _num_waiting_requests(scheduler: Scheduler) -> int:
    return len(scheduler.waiting) + len(scheduler.skipped_waiting)


def _step_until_kv_transfer_finished(scheduler: Scheduler, req_ids: list[str]):
    """Cycle requests through a KV transfer cycle."""

    # Requests should first transition to WAITING_FOR_REMOTE_KVS
    output = scheduler.schedule()
    assert _num_waiting_requests(scheduler) == len(req_ids)
    assert len(scheduler.running) == 0
    assert len(output.scheduled_new_reqs) == 0
    for req in scheduler.requests.values():
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # No model execution yet
    EMPTY_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    initial_ecos = scheduler.update_from_output(output, EMPTY_OUTPUT)

    # Simulate KV transfer completion using KVConnectorOutput.finished_recving
    output = scheduler.schedule()
    assert _num_waiting_requests(scheduler) == len(req_ids)
    assert len(scheduler.running) == 0

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=KVConnectorOutput(finished_recving=req_ids),
    )
    scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    for req_id in req_ids:
        assert req_id in scheduler.finished_recving_kv_req_ids

    return initial_ecos


@pytest.mark.parametrize(
    ("load_modes", "expected_has_sync_loads"),
    [
        ((True,), False),
        ((False,), True),
        ((True, False), True),
    ],
)
@pytest.mark.skip_global_cleanup
def test_has_sync_kv_loads(
    load_modes: tuple[bool, ...],
    expected_has_sync_loads: bool,
    tmp_path,
):
    (tmp_path / "config.json").write_text(
        '{"architectures": ["OPTForCausalLM"], "model_type": "opt"}'
    )
    block_size = 16
    scheduler = create_scheduler(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        use_kv_connector=mock_kv(matched_tokens=block_size, is_async=False),
        block_size=block_size,
    )
    requests = create_requests(
        num_requests=len(load_modes),
        num_tokens=block_size * 2,
        block_size=block_size,
    )
    for request in requests:
        scheduler.add_request(request)

    scheduler.connector.get_num_new_matched_tokens = Mock(
        side_effect=[(block_size, is_async) for is_async in load_modes]
    )

    output = scheduler.schedule()

    assert output.has_sync_kv_loads is expected_has_sync_loads


@pytest.mark.parametrize("is_async", [False, True])
def test_kv_connector_basic(is_async: bool):
    """
    Test whether Scheduler with KVConnector schedules tokens, allocates
    memory, and cleans up requests as expected under normal operation.
    """

    # Setup Scheduler.
    BLOCK_SIZE = 16
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
        block_size=BLOCK_SIZE,
    )
    NUM_TOTAL_BLOCKS = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    ######################################################
    # FIRST SET OF REQUESTS - External Hit Only
    NUM_REQUESTS = 2
    NUM_TOKENS = NUM_MATCHED_NEW_TOKENS * 2
    MAX_TOKENS = 3
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    if is_async:
        _step_until_kv_transfer_finished(scheduler, req_ids)

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Ensure ScheduleOutput is correct.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output=output,
        num_requests=NUM_REQUESTS,
        # Just the incremental tokens should be scheduled.
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )

    # Ensure KVCacheManager is correct.
    _assert_right_kv_cache_manager(
        scheduler, requests, NUM_TOKENS, BLOCK_SIZE, NUM_REQUESTS, NUM_TOTAL_BLOCKS
    )

    # Continue Generation until done.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    _ = scheduler.schedule()
    # Confirm we clean up the memory properly.
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_TOTAL_BLOCKS
    )

    ######################################################
    # SECOND SET OF REQUESTS - Local And External Hit
    NUM_TOKENS_PREFIX = NUM_TOKENS
    # We will get a local prefix cache hit for the first
    # NUM_TOKENS_PREFIX tokens since they are used above.
    NUM_TOKENS = NUM_TOKENS_PREFIX * 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    if is_async:
        _step_until_kv_transfer_finished(scheduler, req_ids)

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # We should get a local cache hit of NUM_TOKENS_PREFIX and
    # a remote KV cache hit of NUM_MATCHED_NEW_TOKENS.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output=output,
        num_requests=NUM_REQUESTS,
        # Just the incremental tokens after local + remote cache hit.
        expected_num_scheduled_tokens=(
            NUM_TOKENS - NUM_TOKENS_PREFIX - NUM_MATCHED_NEW_TOKENS
        ),
    )

    # Ensure KVCacheManager is correct.
    _assert_right_kv_cache_manager(
        scheduler, requests, NUM_TOKENS, BLOCK_SIZE, NUM_REQUESTS, NUM_TOTAL_BLOCKS
    )

    # Continue Generation until done.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    _ = scheduler.schedule()
    # Confirm we clean up the memory properly.
    assert (
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_TOTAL_BLOCKS
    )


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("local_cache_hits", [False, True])
def test_external_prefix_cache_metrics(is_async: bool, local_cache_hits: bool):
    """
    Verify connector prefix cache metrics are updated
    correctly when the scheduler processes requests with KV connector hits.
    """

    BLOCK_SIZE = 16
    if local_cache_hits:
        NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2  # 32 tokens
        NUM_LOCAL_HITS = NUM_MATCHED_NEW_TOKENS * 2  # 64 tokens
        NUM_REQUESTS = 1
        NUM_TOKENS = NUM_LOCAL_HITS * 2  # 128 tokens
    else:
        NUM_MATCHED_NEW_TOKENS = 4
        NUM_LOCAL_HITS = 0
        NUM_REQUESTS = 2
        NUM_TOKENS = 8  # 8 tokens

    # Setup Scheduler.
    scheduler = create_scheduler(
        enable_prefix_caching=local_cache_hits,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
        block_size=BLOCK_SIZE,
    )

    if local_cache_hits:
        # First, establish local cache by running a request to completion
        requests = create_requests(
            num_requests=1,
            num_tokens=NUM_LOCAL_HITS,
            max_tokens=2,
            block_size=BLOCK_SIZE,
        )
        req_ids = []
        req_to_index = {}
        for i, request in enumerate(requests):
            scheduler.add_request(request)
            req_ids.append(request.request_id)
            req_to_index[request.request_id] = i

        if is_async:
            _step_until_kv_transfer_finished(scheduler, req_ids)

        # Run first request to completion to establish local cache
        output = scheduler.schedule()
        MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_to_index,
            sampled_token_ids=[[1000]] * len(req_ids),
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
        _ = scheduler.schedule()

    # --- Prepare test requests ---
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    initial_ecos = None
    if is_async:
        initial_ecos = _step_until_kv_transfer_finished(scheduler, req_ids)

    # --- Trigger scheduling and simulate model output ---
    output = scheduler.schedule()
    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=[r.request_id for r in requests],
        req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
        sampled_token_ids=[[1000]] * NUM_REQUESTS,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Update scheduler stats
    ecos = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # --- Assertions ---
    assert ecos is not None and len(ecos) > 0
    assert ecos[0].scheduler_stats is not None

    if local_cache_hits:
        # For async, local cache stats come from the first step
        if initial_ecos:
            local_stats = initial_ecos[0].scheduler_stats.prefix_cache_stats
        else:
            local_stats = ecos[0].scheduler_stats.prefix_cache_stats
        assert local_stats is not None
        assert local_stats.queries == NUM_TOKENS * NUM_REQUESTS
        assert local_stats.hits == NUM_LOCAL_HITS * NUM_REQUESTS

    if initial_ecos:
        external_stats = initial_ecos[0].scheduler_stats.connector_prefix_cache_stats
    else:
        external_stats = ecos[0].scheduler_stats.connector_prefix_cache_stats
    assert external_stats is not None

    assert external_stats.queries == (NUM_TOKENS - NUM_LOCAL_HITS) * NUM_REQUESTS
    assert external_stats.hits == NUM_MATCHED_NEW_TOKENS * NUM_REQUESTS
    assert external_stats.requests == NUM_REQUESTS
    assert external_stats.preempted_requests == 0


@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_kv_connector_unable_to_allocate(use_ec_connector, ec_role):
    """
    Test whether scheduler with KVConnector is able to handle
    unable to allocate (run out of blocks in allocate_slots().
    """

    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 4
    NUM_BLOCKS = 10
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE * 2
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=False),
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
    )

    # Create two requests. The second request will not be able to
    # allocate slots because it will not have enough blocks.
    NUM_REQUESTS = 2
    NUM_TOKENS = (NUM_BLOCKS // 2 + 1) * BLOCK_SIZE
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Just one request should be running.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        num_requests=1,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # All memory should be freed, with one request waiting.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1

    # Just one request should be running.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        num_requests=1,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # All memory should be freed, with no requests waiting / running.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_kv_connector_handles_preemption(
    is_async, use_ec_connector, ec_role, use_v2_model_runner
):
    """
    Test whether scheduler with KVConnector is able to handle
    unable to allocate (run out of blocks in allocate_slots().
    """

    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 2
    # NOTE: there is 1 null block, so this is 6 blocks.
    NUM_BLOCKS = 7
    NUM_MATCHED_NEW_TOKENS = BLOCK_SIZE
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(
            matched_tokens=NUM_MATCHED_NEW_TOKENS, is_async=is_async
        ),
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
        use_v2_model_runner=use_v2_model_runner,
    )

    # Create two requests.
    # Both can be scheduled at first, but the second request
    # will be preempted and re-scheduled.
    NUM_REQUESTS = 2
    NUM_TOKENS = BLOCK_SIZE * 2 + 1
    MAX_TOKENS = BLOCK_SIZE * 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # All can be scheduled - 1st token.
    output = scheduler.schedule()
    if is_async:
        assert _num_waiting_requests(scheduler) == 2
        assert scheduler.running == []
        _step_until_kv_transfer_finished(scheduler, req_ids)
        output = scheduler.schedule()

    _assert_right_scheduler_output(
        output,
        # 2 remote kv cache hits.
        num_requests=2,
        expected_num_scheduled_tokens=NUM_TOKENS - NUM_MATCHED_NEW_TOKENS,
    )
    assert len(scheduler.running) == 2
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # All can be scheduled - 2nd token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 2
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)

    # This will generate a new block and cause a preemption - 3rd token.
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Only 1 can be scheduled - 4th (and last token).
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 0
    # All memory should be freed since nothing is running.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1

    # Restarts the preempted request - generate 3rd token.
    # This will have a local and remote cache hit.
    output = scheduler.schedule()
    if is_async:
        waiting_req_ids = [
            req.request_id
            for req in scheduler.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        ]
        assert len(waiting_req_ids) == 1
        _step_until_kv_transfer_finished(scheduler, waiting_req_ids)
        output = scheduler.schedule()

    _assert_right_scheduler_output(
        output,
        # 1 remote kv_cache hit!
        num_requests=1,
        # Only 1 block was preempted and there is a single
        # remote hit. So only single new token is scheduled.
        expected_num_scheduled_tokens=1,
    )
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0
    if use_v2_model_runner:
        # V2 emits a resumed (previously preempted) request as a
        # NewRequestData rather than a cached request.
        assert output.scheduled_cached_reqs.num_reqs == 0
        assert len(output.scheduled_new_reqs) == 1
    else:
        assert output.scheduled_cached_reqs.num_reqs == 1
        assert output.scheduled_new_reqs == []
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # Only 1 can be scheduled - 4th (and last token).
    output = scheduler.schedule()
    _assert_right_scheduler_output(
        output,
        # no connector_metadata
        num_requests=0,
        expected_num_scheduled_tokens=1,
    )
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_new_reqs == []
    assert len(scheduler.running) == 1
    _ = scheduler.update_from_output(output, MODEL_RUNNER_OUTPUT)
    assert len(scheduler.running) == 0
    # All memory should be freed since nothing is running.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1


def make_output(scheduler: Scheduler):
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in scheduler.running],
        req_id_to_index={req.request_id: i for i, req in enumerate(scheduler.running)},
        sampled_token_ids=[[1000]] * len(scheduler.running),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def assert_scheduler_empty(scheduler: Scheduler):
    """Confirm the scheduler is "empty" - i.e. no leaks."""
    # Scheduler Metadata.
    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.finished_req_ids) == 0

    # EncoderCacheManager.
    assert len(scheduler.encoder_cache_manager.freed) == 0
    assert len(scheduler.encoder_cache_manager.cached) == 0

    # KVCache Manager.
    assert (
        len(
            scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
        )
        == 0
    )
    assert (
        len(
            scheduler.kv_cache_manager.coordinator.single_type_managers[
                0
            ].num_cached_block
        )
        == 0
    )
    num_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    assert num_free_blocks == (scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

    # NOTE(rob): just the ref count on blocks will be 0. The hash
    # value, etc will remain since we lazily evict for prefix cache.
    for block in scheduler.kv_cache_manager.block_pool.blocks:
        assert block.ref_cnt == 0
        # assert block._block_hash is None
    # assert (
    #     len(scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block
    #           ) == 0)


def test_memory_leak():
    """Test that we do not have a memory leak."""

    scheduler = create_scheduler(enable_prefix_caching=True)

    NUM_REQUESTS = 5
    NUM_TOKENS = 10
    MAX_TOKENS = 10
    requests = create_requests(
        num_requests=NUM_REQUESTS, num_tokens=NUM_TOKENS, max_tokens=MAX_TOKENS
    )

    # Add each request.
    for request in requests:
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Iterate until done.
    while True:
        scheduler_output = scheduler.schedule()
        if len(scheduler.running) == 0:
            break
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm no memory leak.
    assert_scheduler_empty(scheduler)


def create_scheduler_with_priority(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: bool = False,
    long_prefill_token_threshold: int = 0,
    disable_chunked_mm_input: bool = False,
    use_kv_connector: bool = False,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: int | None = None,
    num_speculative_tokens: int | None = None,
    use_ec_connector: bool = False,
    ec_role: str | None = None,
    use_v2_model_runner: bool | None = None,
) -> Scheduler:
    """Create scheduler with priority policy enabled.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (False)

    Returns:
      {class}`Scheduler` instance with priority scheduling
    """
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy="priority",  # Enable priority scheduling
        # Ensure admission/preemption mechanics are deterministic
        watermark=0.0,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    kv_transfer_config = (
        KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        )
        if use_kv_connector
        else None
    )

    speculative_config: SpeculativeConfig | None = None
    if num_speculative_tokens is not None:
        speculative_config = SpeculativeConfig(
            model="ngram", num_speculative_tokens=num_speculative_tokens
        )

    ec_transfer_config = (
        ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role=ec_role,
            ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test"},
        )
        if use_ec_connector
        else None
    )

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        speculative_config=speculative_config,
        ec_transfer_config=ec_transfer_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
    )
    if use_v2_model_runner is None:
        use_v2_model_runner = bool(envs.VLLM_USE_V2_MODEL_RUNNER)
    scheduler.use_v2_model_runner = use_v2_model_runner
    return scheduler


_none_hash_initialized = False


def create_requests_with_priority(
    num_requests: int,
    priorities: list[int],
    arrival_times: list[float] | None = None,
    num_tokens: int = 10,
    mm_hashes_list: list[list[str]] | None = None,
    mm_positions: list[list[PlaceholderRange]] | None = None,
    max_tokens: int = 16,
    stop_token_ids: list[int] | None = None,
    prompt_logprobs: int | None = None,
    starting_idx: int = 0,
    same_prompt: bool = False,
    block_size: int = 16,
    req_ids: list[str] | None = None,
):
    """Create requests with specified priorities and arrival times."""
    assert len(priorities) == num_requests
    if arrival_times is not None:
        assert len(arrival_times) == num_requests
    else:
        arrival_times = [float(i) for i in range(num_requests)]

    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        prompt_logprobs=prompt_logprobs,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    requests = []

    if mm_hashes_list is not None:
        # NOTE: allow manual input; some mm items can have the same identifier
        # no. of mm_hashes and mm_positions for each request should be identical
        assert mm_positions is not None, (
            "mm_positions must be provided when mm_hashes_list is provided"
        )
        assert len(mm_hashes_list) == len(mm_positions) == num_requests
        assert [len(h) for h in mm_hashes_list] == [len(p) for p in mm_positions]

        # Since same identifier would imply they are identical encoder output
        # Verify mm items with identical identifier are having mm_position.length
        seen_hashes: dict[str, int] = {}

    if req_ids:
        assert len(req_ids) == num_requests
    else:
        req_ids = [f"{i + starting_idx}" for i in range(num_requests)]

    for i in range(num_requests):
        mm_features = []

        for j, position in enumerate(
            mm_positions[i] if mm_positions is not None else []
        ):
            if mm_hashes_list is not None:
                identifier = mm_hashes_list[i][j]

                # Verify if position length is identical
                position_length = position.length
                if identifier in seen_hashes:
                    assert seen_hashes[identifier] == position_length, (
                        f"mm_hash '{identifier}' has inconsistent position lengths: "
                        f"previously {seen_hashes[identifier]}, now {position_length} "
                        f"at request {i}, position {j}"
                    )
                else:
                    seen_hashes[identifier] = position_length
            else:
                # Unique dummy hash for each mm item
                identifier = f"hash{i}_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

        prompt_token_ids = (
            [starting_idx] * num_tokens
            if same_prompt
            else [i + starting_idx] * num_tokens
        )
        request = Request(
            request_id=req_ids[i],
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=mm_features if mm_features else None,
            arrival_time=arrival_times[i],
            priority=priorities[i],
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests


def test_priority_scheduling_basic_ordering():
    """Test that requests are scheduled in priority order
    (lower value = higher priority)."""
    scheduler = create_scheduler_with_priority()

    # Create requests with different priorities
    # Priority 0 (highest), 1, 2 (lowest)
    priorities = [2, 0, 1]  # Add in non-priority order
    arrival_times = [1.0, 2.0, 3.0]  # All different arrival times
    requests = create_requests_with_priority(
        num_requests=3, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests in non-priority order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify priority order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in priority order:
    # req_1 (priority 0), req_2 (priority 1), req_0 (priority 2)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_priority_scheduling_arrival_time_tiebreaker():
    """Test that arrival time is used
    as tiebreaker when priorities are equal."""
    scheduler = create_scheduler_with_priority()

    # Create requests with same priority but different arrival times
    priorities = [1, 1, 1]  # All same priority
    arrival_times = [3.0, 1.0, 2.0]  # Different arrival times
    requests = create_requests_with_priority(
        num_requests=3, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests in non-arrival order
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify arrival time order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 3

    # Verify they are scheduled in arrival time order:
    # req_1 (1.0), req_2 (2.0), req_0 (3.0)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["1", "2", "0"]


def test_priority_scheduling_mixed_priority_and_arrival():
    """Test priority scheduling with mixed priorities and arrival times."""
    scheduler = create_scheduler_with_priority()

    # Create requests with mixed priorities and arrival times
    priorities = [2, 1, 1, 0]  # Mixed priorities
    arrival_times = [1.0, 3.0, 2.0, 4.0]  # Mixed arrival times
    requests = create_requests_with_priority(
        num_requests=4, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule and verify order
    output = scheduler.schedule()

    # Should schedule all requests since they fit in budget
    assert len(output.scheduled_new_reqs) == 4

    # Expected order:
    # 1. req_3 (priority 0, arrival 4.0)
    # 2. req_2 (priority 1, arrival 2.0) - earlier arrival than req_1
    # 3. req_1 (priority 1, arrival 3.0)
    # 4. req_0 (priority 2, arrival 1.0)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_req_ids == ["3", "2", "1", "0"]


def test_priority_scheduling_preemption():
    """Test that under KV block pressure the scheduler preempts the
    lowest-priority *running* request, not the highest-priority one.

    A low-priority request starts running first. Then a high-priority
    request arrives and is admitted to running.  When block pressure
    builds, the scheduler preempts the low-priority running request
    while keeping the high-priority one.

    Block math
    ----------
    block_size = 16, num_blocks = 6 (1 null → 5 usable).

    Phase 1: lo1 (priority 5, 32 tokens) → 2 blocks.  3 free.
             Decode → lo1 has 33 tokens (needs 3rd block on next schedule).
    Phase 2: hi1 (priority 0, 32 tokens) arrives.
             schedule() allocates lo1's 3rd block (3 used) and admits
             hi1 (2 blocks) → 5 used, 0 free. Both running.
             Decode → lo1 34 tokens, hi1 33 tokens.
    Phase 3: schedule() → hi1 needs 3rd block, 0 free → preemption.
             lo1 (priority 5) is preempted, hi1 (priority 0) survives.
    """
    block_size = 16
    num_blocks = 6  # 1 null block → 5 usable
    num_tokens = block_size * 2  # 32 tokens = exactly 2 blocks

    scheduler = create_scheduler_with_priority(
        max_num_seqs=3,
        max_num_batched_tokens=200,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    # --- Phase 1: low-priority request starts running ---
    lo1 = create_requests_with_priority(
        num_requests=1,
        priorities=[5],
        arrival_times=[1.0],
        num_tokens=num_tokens,
        req_ids=["lo1"],
    )[0]
    scheduler.add_request(lo1)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1

    # Decode: lo1 now has 33 tokens (crosses 32-token boundary).
    model_output = ModelRunnerOutput(
        req_ids=["lo1"],
        req_id_to_index={"lo1": 0},
        sampled_token_ids=[[100]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # --- Phase 2: high-priority request arrives AFTER lo1 is running ---
    hi1 = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[2.0],
        num_tokens=num_tokens,
        req_ids=["hi1"],
    )[0]
    scheduler.add_request(hi1)

    # schedule(): lo1 gets its 3rd block (3 used), hi1 admitted (5 used,
    # 0 free).  Both are now running.
    output = scheduler.schedule()
    assert any(r.req_id == "hi1" for r in output.scheduled_new_reqs)
    assert len(scheduler.running) == 2

    # Decode: lo1 → 34 tokens, hi1 → 33 tokens.
    model_output = ModelRunnerOutput(
        req_ids=["lo1", "hi1"],
        req_id_to_index={"lo1": 0, "hi1": 1},
        sampled_token_ids=[[101], [100]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # --- Phase 3: preemption with mixed-priority running requests ---
    # hi1 needs a 3rd block but 0 are free.  The scheduler picks the
    # lowest-priority running request to preempt:
    #   max(running, key=(priority, arrival_time)) → lo1 (5 > 0).
    output = scheduler.schedule()

    lo1_req = scheduler.requests["lo1"]
    assert lo1_req.status == RequestStatus.PREEMPTED, (
        "Expected low-priority 'lo1' to be preempted"
    )
    assert any(req.request_id == "hi1" for req in scheduler.running), (
        "High-priority 'hi1' should still be running"
    )


def test_priority_scheduling_no_preemption_when_space_available():
    """Test that preemption doesn't happen
    when there's space for new requests."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=3,  # Allow 3 concurrent requests
        max_num_batched_tokens=200,  # Sufficient token budget
    )

    # Add two low-priority running requests
    low_priority_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=30,
        req_ids=["lo1", "lo2"],
    )

    for request in low_priority_requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in low_priority_requests],
        req_id_to_index={
            req.request_id: i for i, req in enumerate(low_priority_requests)
        },
        sampled_token_ids=[[100] for _ in low_priority_requests],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Add high-priority request
    high_priority_request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=30,
        req_ids=["hi1"],
    )[0]

    scheduler.add_request(high_priority_request)

    # Schedule - should not preempt since there's space
    output = scheduler.schedule()

    # Should schedule the new request without preemption
    assert len(output.scheduled_new_reqs) == 1
    assert len(scheduler.running) == 3  # All three requests running
    assert len(scheduler.waiting) == 0  # No requests waiting


def test_priority_scheduling_preemption_victim_selection():
    """Test that the correct victim is selected for
    preemption based on priority and arrival time."""
    # This test verifies the priority-based victim selection logic
    # by checking the waiting queue order after adding requests with different
    # priorities
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Force sequential processing to test priority order
    )

    # Create requests with different priorities
    requests = create_requests_with_priority(
        num_requests=3,
        priorities=[3, 2, 0],  # Different priorities: low, medium, high
        arrival_times=[1.0, 2.0, 3.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the highest priority request
    # (req_2, priority 0)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "2"  # Highest priority

    # Verify the waiting queue has the remaining requests in priority order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify priority order
    waiting_requests = list(scheduler.waiting)

    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_1 (priority 2) then req_0 (priority 3)
    assert waiting_priorities == [2, 3]
    assert waiting_req_ids == ["1", "0"]


def test_priority_scheduling_equal_priority_preemption():
    """Test arrival time tiebreaker when requests have equal priority."""
    # This test verifies that arrival time is used as a tiebreaker for equal
    # priorities
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Force sequential processing
    )

    # Create requests with same priority but different arrival times
    requests = create_requests_with_priority(
        num_requests=3,
        priorities=[2, 2, 2],  # Same priority
        arrival_times=[3.0, 1.0, 2.0],  # Different arrival times
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should schedule the request with earliest arrival time
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "1"  # Earliest arrival (1.0)

    # Verify the waiting queue has remaining requests in arrival time order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify arrival time order
    waiting_requests = list(scheduler.waiting)

    waiting_arrival_times = [req.arrival_time for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_2 (arrival 2.0) then req_0 (arrival 3.0)
    assert waiting_arrival_times == [2.0, 3.0]
    assert waiting_req_ids == ["2", "0"]


def test_priority_scheduling_waiting_queue_order():
    """Test that the waiting queue maintains priority order."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Only one request can run at a time
    )

    # Create multiple requests with different priorities
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 1, 2, 0],  # Mixed priorities
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the highest priority request
    # (req_3, priority 0)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_new_reqs[0].req_id == "3"

    # Verify waiting queue has remaining requests in priority order
    assert len(scheduler.waiting) == 3

    # Extract requests from waiting queue
    # (it's a heap, so we need to pop to see order)
    waiting_requests = list(scheduler.waiting)

    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be ordered by priority: req_1 (1), req_2 (2), req_0 (3)
    assert waiting_req_ids == ["1", "2", "0"]
    assert waiting_priorities == [1, 2, 3]


def test_priority_scheduling_fcfs_fallback():
    """Test that FCFS behavior is maintained when all
    requests have same priority."""
    scheduler = create_scheduler_with_priority()

    # Create requests with same priority but different arrival times
    priorities = [1, 1, 1, 1]  # All same priority
    arrival_times = [4.0, 1.0, 3.0, 2.0]  # Different arrival times
    requests = create_requests_with_priority(
        num_requests=4, priorities=priorities, arrival_times=arrival_times
    )

    # Add requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule
    output = scheduler.schedule()

    # Should schedule all requests in arrival time order
    assert len(output.scheduled_new_reqs) == 4
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]

    # Expected order by arrival time:
    # req_1 (1.0), req_3 (2.0), req_2 (3.0), req_0 (4.0)
    assert scheduled_req_ids == ["1", "3", "2", "0"]


def test_priority_scheduling_with_limited_slots():
    """Test priority scheduling when max_num_seqs limits concurrent requests."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=2,  # Only allow 2 concurrent requests
        max_num_batched_tokens=1000,  # Plenty of token budget
    )

    # Create requests with different priorities
    requests = create_requests_with_priority(
        num_requests=4,
        priorities=[3, 1, 2, 0],  # Mixed priorities
        arrival_times=[1.0, 2.0, 3.0, 4.0],
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule - should only schedule the 2 highest priority requests
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 2

    # Should schedule req_3 (priority 0) and req_1 (priority 1)
    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert "3" in scheduled_req_ids  # Priority 0
    assert "1" in scheduled_req_ids  # Priority 1

    # Remaining requests should be in waiting queue in priority order
    assert len(scheduler.waiting) == 2

    # Extract waiting requests and verify order
    waiting_requests = list(scheduler.waiting)
    waiting_priorities = [req.priority for req in waiting_requests]
    waiting_req_ids = [req.request_id for req in waiting_requests]

    # Should be req_2 (priority 2) then req_0 (priority 3)
    assert waiting_priorities == [2, 3]
    assert waiting_req_ids == ["2", "0"]


def test_priority_scheduling_heap_property():
    """Test that the waiting queue maintains heap
    property for priority scheduling."""
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,  # Only one request can run at a time
    )

    # Add requests in random priority order
    priorities = [5, 1, 8, 3, 2, 7, 4, 6]
    arrival_times = [float(i) for i in range(len(priorities))]
    requests = create_requests_with_priority(
        num_requests=len(priorities),
        priorities=priorities,
        arrival_times=arrival_times,
        num_tokens=10,
    )

    # Add all requests
    for request in requests:
        scheduler.add_request(request)

    # Schedule one request at a time and verify priority order
    scheduled_priorities = []

    while scheduler.waiting:
        output = scheduler.schedule()
        if output.scheduled_new_reqs:
            req = output.scheduled_new_reqs[0]
            scheduled_priorities.append(requests[int(req.req_id)].priority)

            # Simulate completion to make room for next request
            model_output = ModelRunnerOutput(
                req_ids=[req.req_id],
                req_id_to_index={req.req_id: 0},
                sampled_token_ids=[[100]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
            scheduler.update_from_output(output, model_output)

            # Finish the request to make room for the next one
            scheduler.finish_requests(req.req_id, RequestStatus.FINISHED_STOPPED)

    # Verify requests were scheduled in priority order (lowest value first)
    expected_priorities = sorted(priorities)
    assert scheduled_priorities == expected_priorities


def test_schedule_skip_tokenizer_init():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    requests = create_requests(num_requests=5)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)


def test_schedule_skip_tokenizer_init_structured_output_request():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    structured_outputs_params = StructuredOutputsParams(regex="[0-9]+")
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=16,
        structured_outputs=structured_outputs_params,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 1


@pytest.mark.parametrize("async_grammar", [True, False])
def test_grammar_compile_error_finishes_only_request(async_grammar: bool):
    scheduler = create_scheduler()
    manager = scheduler.structured_output_manager
    manager.backend = Mock()
    manager.backend.compile_grammar.side_effect = RuntimeError(
        "forced FSM compilation error"
    )
    manager._use_async_grammar_compilation = async_grammar

    sampling_params = SamplingParams(
        max_tokens=16,
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="grammar-error",
        prompt_token_ids=[0, 1],
        sampling_params=sampling_params,
        pooling_params=None,
    )

    manager.grammar_init(request)
    assert request.structured_output_request is not None
    grammar_future = request.structured_output_request._grammar
    assert isinstance(grammar_future, Future)
    assert isinstance(grammar_future.exception(timeout=5), RuntimeError)

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    assert not scheduler_output.num_scheduled_tokens

    engine_core_outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    output = engine_core_outputs[0].outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason == FinishReason.ERROR
    assert output.stop_reason is None

    healthy_request = create_requests(num_requests=1, req_ids=["healthy-request"])[0]
    scheduler.add_request(healthy_request)
    next_output = scheduler.schedule()
    assert [req.req_id for req in next_output.scheduled_new_reqs] == [
        healthy_request.request_id
    ]


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(Scheduler)
    sampling_params = SamplingParams(ignore_eos=True, max_tokens=4)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    request = Request(
        request_id="0",
        prompt_token_ids=[0, 1],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
    request.structured_output_request.grammar.accept_tokens.return_value = False
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True
    scheduler.structured_output_manager.trim_reasoning_for_advance.side_effect = (
        lambda request, new_token_ids: new_token_ids
    )
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.grammar_compile_error_reqs = set()
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.enable_return_routed_experts = False
    scheduler.return_sampling_mask = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req: Request, delay_free_blocks: bool = False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None, None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_core_outputs = scheduler.update_from_output(output, model_runner_output)

    request.structured_output_request.grammar.accept_tokens.assert_called_once_with(
        request.request_id, [123]
    )
    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running
    scheduler._free_request.assert_called_once_with(request)
    assert len(engine_core_outputs[0].outputs) == 1
    engine_core_output = engine_core_outputs[0].outputs[0]
    assert engine_core_output.request_id == request.request_id
    assert engine_core_output.new_token_ids == [123]
    assert engine_core_output.finish_reason == FinishReason.ERROR


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
@pytest.mark.parametrize(
    "use_ec_connector, ec_role", [(False, None), (True, "ec_consumer")]
)
def test_priority_scheduling_preemption_and_resumption_when_out_of_kv(
    use_ec_connector, ec_role, use_v2_model_runner
):
    """Test that priority scheduling preempts lower priority requests
    when out of KV cache space."""
    # Create scheduler with very limited memory to force preemption
    scheduler = create_scheduler_with_priority(
        max_num_seqs=2,  # Allow multiple requests
        max_num_batched_tokens=200,
        num_blocks=5,  # Can hold 64 tokens (first block is null)
        block_size=16,  # Standard block size
        use_kv_connector=True,
        # encoder connector should not affect test results
        use_ec_connector=use_ec_connector,
        ec_role=ec_role,
        use_v2_model_runner=use_v2_model_runner,
    )

    # Create a request and schedule it
    request_low = create_requests_with_priority(
        num_requests=1,
        priorities=[1],
        arrival_times=[0.0],
        num_tokens=30,
        starting_idx=0,
    )[0]
    scheduler.add_request(request_low)
    # 1st schedule
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    # Simulate model execution - 1st decode
    model_output = ModelRunnerOutput(
        req_ids=[request_low.request_id],
        req_id_to_index={request_low.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Create a high priority request and schedule it
    request_high = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[1.0],
        num_tokens=32,
        starting_idx=1,
    )[0]
    scheduler.add_request(request_high)
    # 2nd schedule
    output = scheduler.schedule()
    # KV cache should be full at this point
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 2

    # Simulate model execution - 2nd decode
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100] for _ in requests],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # 3rd schedule - this should trigger preemption
    # req_low needs 32 tokens = 2 blocks
    # req_high needs 33 tokens = 3 blocks
    # so doesn't fit in 4 blocks.
    output = scheduler.schedule()

    # Should have preempted req_low
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_cached_reqs.req_ids[0] == request_high.request_id
    assert scheduler.requests[request_low.request_id].status == RequestStatus.PREEMPTED
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1

    # Simulate model execution - 3rd decode
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[], [100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    # Finish the requests to make room for the preempted requests to resume
    scheduler.update_from_output(output, model_output)
    scheduler.finish_requests(request_high.request_id, RequestStatus.FINISHED_STOPPED)

    # 4th Schedule - this should trigger the resumption
    output = scheduler.schedule()
    scheduled_cached_reqs = output.scheduled_cached_reqs

    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    if use_v2_model_runner:
        # V2 emits the resumed request as a NewRequestData, carrying its full
        # token ids in prefill_token_ids (instead of cached all_token_ids).
        assert scheduled_cached_reqs.num_reqs == 0
        assert len(output.scheduled_new_reqs) == 1
        new_req = output.scheduled_new_reqs[0]
        assert new_req.req_id == request_low.request_id
        # Resumed tokens include 30 prompt tokens and 2 decoded tokens.
        assert len(new_req.prefill_token_ids) == 32
        assert new_req.prefill_token_ids[31] == 100
    else:
        assert len(output.scheduled_new_reqs) == 0
        assert scheduled_cached_reqs.num_reqs == 1
        # Preempted request resumed in scheduled_cached_reqs
        assert len(scheduled_cached_reqs.resumed_req_ids) == 1
        assert len(scheduled_cached_reqs.all_token_ids) == 1
        assert scheduled_cached_reqs.req_ids[0] == request_low.request_id
        assert request_low.request_id in scheduled_cached_reqs.resumed_req_ids
        assert request_low.request_id in scheduled_cached_reqs.all_token_ids
        # Resumed tokens include 30 prompt tokens and 2 decoded tokens
        assert len(scheduled_cached_reqs.all_token_ids[request_low.request_id]) == 32
        assert scheduled_cached_reqs.all_token_ids[request_low.request_id][31] == 100


def test_preempted_output_replay_stays_prefill_until_frozen_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    """A resumed output-bearing request crosses one scheduler phase boundary.

    The first two chunks reconstruct the exact stream frozen at preemption and
    therefore remain manifest-prefill rows.  The first step whose starting
    frontier equals that boundary is ordinary autoregressive decode.
    """
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler_with_priority(
        max_num_seqs=2,
        max_num_batched_tokens=200,
        num_blocks=5,
        block_size=16,
        use_v2_model_runner=True,
    )

    def model_output(
        requests: list[Request], sampled_token_ids: list[list[int]]
    ) -> ModelRunnerOutput:
        request_ids = [request.request_id for request in requests]
        return ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={
                request_id: index for index, request_id in enumerate(request_ids)
            },
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    request_low = create_requests_with_priority(
        num_requests=1,
        priorities=[1],
        arrival_times=[0.0],
        num_tokens=30,
    )[0]
    scheduler.add_request(request_low)
    first = scheduler.schedule()
    scheduler.update_from_output(first, model_output([request_low], [[100]]))

    request_high = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[1.0],
        num_tokens=32,
        starting_idx=1,
    )[0]
    scheduler.add_request(request_high)
    second = scheduler.schedule()
    scheduler.update_from_output(
        second,
        model_output([request_low, request_high], [[100], [100]]),
    )

    preempting = scheduler.schedule()
    assert request_low.status == RequestStatus.PREEMPTED
    assert request_low.num_prompt_tokens == 30
    assert len(request_low.all_token_ids) == 32
    assert request_low.execution_prefill_len == 32
    scheduler.update_from_output(
        preempting,
        model_output([request_low, request_high], [[], [100]]),
    )
    scheduler.finish_requests(request_high.request_id, RequestStatus.FINISHED_STOPPED)

    # Isolate lifecycle identity from graph admission: both physical routes are
    # already HOT and zero-priced, so each output reaches the ordinary manifest
    # builder without capture, eviction, a nonzero resource loan, or GPU work.
    scheduler.max_num_scheduled_tokens = 16
    scheduler._elastic_graph_execution_policy = _approved_q4_full_policy()
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 2}
    for step_key in ((0, 0, 1, 16, 0), (1, 0, 1, 1, 1)):
        for physical_key in scheduler._resolve_elastic_step_physical_keys(step_key):
            scheduler._elastic_admission_controller.publish_hot(
                physical_key,
                GraphPrice(
                    resident_bytes=0,
                    capture_peak_bytes=0,
                    reclaim_group=f"lifecycle:{physical_key.identity}",
                ),
                pinned=physical_key.logical.mode == "FULL",
            )

    assert request_low.num_computed_tokens == 0
    replay_one = scheduler.schedule()
    assert replay_one.num_scheduled_tokens == {request_low.request_id: 16}
    assert replay_one.is_pure_decode_step is False
    assert replay_one.elastic_step_plan is not None
    assert replay_one.elastic_step_plan.execution_manifest is not None
    assert (
        replay_one.elastic_step_plan.execution_manifest.per_request_is_prefilling
        == (True,)
    )
    assert replay_one.scheduled_new_reqs[0].execution_prefill_len == 32
    scheduler.update_from_output(replay_one, model_output([request_low], [[]]))

    assert request_low.num_computed_tokens == 16
    replay_two = scheduler.schedule()
    assert replay_two.num_scheduled_tokens == {request_low.request_id: 16}
    assert replay_two.is_pure_decode_step is False
    assert replay_two.elastic_step_plan is not None
    assert replay_two.elastic_step_plan.execution_manifest is not None
    assert (
        replay_two.elastic_step_plan.execution_manifest.per_request_is_prefilling
        == (True,)
    )
    # The final replay may emit the next token, but that must not move the
    # scheduler-owned reconstruction boundary.
    scheduler.update_from_output(replay_two, model_output([request_low], [[101]]))
    assert request_low.num_computed_tokens == 32
    assert request_low.execution_prefill_len == 32
    assert len(request_low.all_token_ids) == 33

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {request_low.request_id: 1}
    assert decode.is_pure_decode_step is True
    assert decode.elastic_step_plan is not None
    assert decode.elastic_step_plan.execution_manifest is not None
    assert decode.elastic_step_plan.execution_manifest.per_request_is_prefilling == (
        False,
    )


@pytest.mark.parametrize(
    ("enable_chunked_prefill", "is_encoder_decoder", "expect_enabled"),
    [
        (True, False, True),
        (False, False, False),
        # Encoder-decoder models should always have it disabled
        (False, True, False),
        (True, True, False),
    ],
)
def test_chunked_prefill_disabled_for_encoder_decoder(
    enable_chunked_prefill: bool, is_encoder_decoder: bool, expect_enabled: bool
) -> None:
    """Validate that chunked prefill is appropriately disabled for
    encoder-decoder models."""
    scheduler_config = SchedulerConfig(
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=is_encoder_decoder,
        # Must <= max_num_batched_tokens if chunked prefill is disabled
        max_model_len=SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )

    # `is_encoder_decoder` should only be used during construction
    # of the config, and otherwise stored in the model config.
    assert "is_encoder_decoder" not in vars(scheduler_config)
    assert "is_encoder_decoder" not in [
        f.name for f in dataclasses.fields(scheduler_config)
    ]
    _validate_chunked_prefill_settings_for_encoder_decoder(
        scheduler_config, is_encoder_decoder, expect_enabled
    )

    # Ensure it is retained in VllmConfig, even after its post-init.
    vllm_config = VllmConfig(scheduler_config=scheduler_config)
    _validate_chunked_prefill_settings_for_encoder_decoder(
        vllm_config.scheduler_config, is_encoder_decoder, expect_enabled
    )


def _validate_chunked_prefill_settings_for_encoder_decoder(
    scheduler_config: SchedulerConfig, is_encoder_decoder: bool, expect_enabled: bool
) -> None:
    """Validate chunked prefill settings in the scheduler config for
    encoder-decoder models."""
    assert scheduler_config.enable_chunked_prefill is expect_enabled
    if is_encoder_decoder:
        # Encoder-decoder models should automatically disable chunked multimodal
        # inputs as well
        assert scheduler_config.disable_chunked_mm_input is not expect_enabled
    if is_encoder_decoder and not expect_enabled:
        assert scheduler_config.long_prefill_token_threshold == 0


# ==============================================================================
# EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests start
# NOTE: In E->P->D disagg case, both KV and EC Connector works in P instance
# Unless specify, the existence of KV Connector should not affect any test results
# ==============================================================================


def _assert_right_encoder_cache_allocated(
    scheduler: Scheduler,
    hashes_to_check: list[str] | None = None,
    requests: list[Request] | None = None,
    expected_total_allocated: int | None = None,
):
    """Check whether encoder cache is allocated correctly."""
    encoder_cache_manager = scheduler.encoder_cache_manager

    # Verify encoder cache manager exists
    assert encoder_cache_manager is not None, "Encoder cache manager should exist"

    # Verify number of cache
    if expected_total_allocated is not None:
        assert len(encoder_cache_manager.cached) == expected_total_allocated
        if expected_total_allocated == 0:
            return

    # Verify each request with MM data is in cache
    cached_hashes = set(encoder_cache_manager.cached.keys())

    if hashes_to_check:
        missed_hashes = set(hashes_to_check) - cached_hashes
        assert not missed_hashes, (
            f"Miss hashes: {missed_hashes} "
            f"Existing encoder cache: {encoder_cache_manager.cached}"
        )

    for req in requests if requests is not None else []:
        if req.mm_features:
            mm_hashes = [f.identifier for f in req.mm_features]
            req_hashes = set(mm_hashes)  # unique hashes set
            missed_hashes = req_hashes - cached_hashes
            assert not missed_hashes, (
                f"Miss hashes in cache for request {req.request_id}: {missed_hashes} "
                f"Existing encoder cache: {encoder_cache_manager.cached}"
            )


def _assert_right_ec_connector_metadata(
    output: SchedulerOutput,
    mm_features_list: list[MultiModalFeatureSpec],
):
    """Verify that ECConnector metadata EXACTLY matches the input MM data"""
    # Get the connector metadata
    metadata = output.ec_connector_metadata

    # Create lookup dictionaries for efficient access
    metadata_dict = {mm_data.mm_hash: mm_data for mm_data in metadata.mm_datas}

    # Check all required identifiers exist in metadata; and no extra
    # In ECExampleConnector format
    # NOTE: even having same identifier, the mm_features can be different
    # since their mm_position can be in different offsets, etc
    identifiers_dict = {f.identifier for f in mm_features_list}
    assert set(metadata_dict.keys()) == identifiers_dict

    # Verify the info matches
    for i, mm_feature in enumerate(mm_features_list):
        identifier = mm_feature.identifier
        assert metadata_dict[identifier].mm_hash == identifier
        assert metadata_dict[identifier].num_token == mm_feature.mm_position.length


def _assert_right_encoder_inputs(
    output: SchedulerOutput,
    check_exist: bool | None = True,
    requests: list[Request] | None = None,
    expected_encoder_inputs: list[list[int]] | None = None,
    expected_total_reqs: int | None = None,
):
    """Verify that requests/mm_hashes should (not) in scheduled encoder input
    If check_exist is False, this function returns True
    if requests are NOT in encoder inputs"""

    # Get the scheduled encoder inputs
    # NOTE: scheduled_encoder_inputs is a dictionary with request id as key
    scheduled_encoder_inputs = output.scheduled_encoder_inputs

    # Check if scheduled_encoder_inputs is empty as expected
    if expected_total_reqs is not None:
        assert len(scheduled_encoder_inputs) == expected_total_reqs
        if expected_total_reqs == 0:
            return

    # Number of expected encoder inputs should match number of requests
    if expected_encoder_inputs:
        assert check_exist and requests is not None  # only support expect input exist
        assert len(requests) == len(expected_encoder_inputs)

    # Check request (not) exist as expected
    for i, request in enumerate(requests if requests is not None else []):
        assert (request.request_id in scheduled_encoder_inputs) is check_exist, (
            f"Request {request.id} presence mismatch: expected {check_exist}, "
            f"got {request.id in scheduled_encoder_inputs}"
        )
        if expected_encoder_inputs:
            scheduled_encoder_input = scheduled_encoder_inputs[request.request_id]
            assert scheduled_encoder_input == expected_encoder_inputs[i]


def test_scheduler_no_ec_connector_by_default():
    """Test scheduler doesn't have EC connector by default."""
    scheduler = create_scheduler()
    assert scheduler.ec_connector is None


def test_mamba_align_encoder_cache_cap_makes_progress():
    """Two individually cacheable images must not deadlock Mamba alignment."""
    block_size = 768
    encoder_cache_size = 600
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        block_size=block_size,
        enable_prefix_caching=True,
    )
    scheduler.need_mamba_block_aligned_split = True
    scheduler.max_num_encoder_input_tokens = encoder_cache_size
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

    first_image_end = 510
    request = create_requests(
        num_requests=1,
        num_tokens=1010,
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=500),
                PlaceholderRange(offset=first_image_end, length=500),
            ]
        ],
        max_tokens=1,
        block_size=block_size,
        req_ids=["req"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == first_image_end
    assert output.scheduled_encoder_inputs[request.request_id] == [0]

    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    output = scheduler.schedule()
    next_block_boundary = block_size
    assert output.num_scheduled_tokens[request.request_id] == (
        next_block_boundary - first_image_end
    )
    assert output.scheduled_encoder_inputs[request.request_id] == [1]


def test_mamba_align_eagle_schedules_encoder_at_boundary():
    """EAGLE lookahead at an aligned MM boundary requires encoder cache."""
    block_size = 512
    scheduler = create_scheduler(
        max_num_batched_tokens=700,
        max_model_len=2048,
        block_size=block_size,
        enable_prefix_caching=True,
    )
    scheduler.need_mamba_block_aligned_split = True
    scheduler.use_eagle = True
    scheduler.use_eagle_block_drop = True
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_prefill_checkpoint_alignment = None
    scheduler.mamba_fine_grained_prefix_cache = False
    scheduler.num_prefill_lookahead = 1
    scheduler.max_num_encoder_input_tokens = 2048
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=2048)

    request = create_requests(
        num_requests=1,
        num_tokens=1200,
        mm_positions=[[PlaceholderRange(offset=block_size, length=100)]],
        max_tokens=1,
        block_size=block_size,
        req_ids=["req"],
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == block_size
    assert output.scheduled_encoder_inputs[request.request_id] == [0]


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_text_only_request(use_kv_connector):
    """Test text-only requests don't allocate encoder cache."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    NUM_PROMPT_TOKENS = 100

    # Create text-only request (no mm_positions)
    requests = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
    )
    assert not requests[0].mm_features  # No MM data

    scheduler.add_request(requests[0])
    output = scheduler.schedule()

    # Should schedule
    assert len(output.scheduled_new_reqs) == 1

    # Scheduled tokens should equal prompt tokens exactly
    scheduled = output.num_scheduled_tokens[requests[0].request_id]
    assert scheduled == NUM_PROMPT_TOKENS, (
        f"Text-only should schedule {NUM_PROMPT_TOKENS}, got {scheduled}"
    )

    # Encoder cache should be empty
    _assert_right_encoder_cache_allocated(scheduler, expected_total_allocated=0)

    # ECConnector should carry no metadata
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_cache_hit_external_load(use_kv_connector):
    """Test ec_consumer loads from external cache when hit.
    A normal basic operation for EPD disaggrgation"""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        # kv connector should not effect test results
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Create MM request
    NUM_TOKENS = 200  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS = 100
    mm_hashes_list = [["hash_test1"]]
    mm_positions = [[PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS)]]

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=mm_hashes_list,
        mm_positions=mm_positions,
    )[0]

    # Mock cache hit - encoder cache has_exists externally
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request)
    output = scheduler.schedule()
    # Should schedule prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[request.request_id]
    assert scheduled_tokens == NUM_TOKENS

    # Should called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request, 0)

    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request])

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(output, mm_features_list=request.mm_features)

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_cache_miss_computes_locally(use_kv_connector):
    """Test consumer can compute encoder locally when cache miss (fallback)."""
    # encoder cache itself if it doesn't receive it from external storage

    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Verify consumer role
    assert scheduler.ec_connector is not None
    assert not scheduler.ec_connector.is_producer

    # Create MM request
    request_mm_missed = create_requests(
        num_requests=1,
        num_tokens=200,  # Total (including 100 MM)
        mm_positions=[[PlaceholderRange(offset=0, length=100)]],  # 100 MM tokens
    )[0]

    # Mock cache miss - encoder cache doesn't exist externally
    scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    scheduler.add_request(request_mm_missed)
    output = scheduler.schedule()

    # SCHEDULER should decide to compute encoder locally (fallback)
    assert len(output.scheduled_new_reqs) == 1

    # Should schedule full prompt tokens
    scheduled_tokens = output.num_scheduled_tokens[request_mm_missed.request_id]
    assert scheduled_tokens == 200, (
        f"Expected 200 tokens on cache miss, got {scheduled_tokens}"
    )

    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_mm_missed])

    # ECConnector should carry no metadata (missed cache)
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input contain mm for request_mm_missed
    _assert_right_encoder_inputs(
        output,
        requests=[request_mm_missed],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item
        expected_total_reqs=1,
    )

    # Then MODEL_RUNNER will execute the encoder and cache the result


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_with_partial_cache_hit_multi_round(use_kv_connector):
    """Test consumer with partial cache hit (local & connector) with 2 requests."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Create MM request
    NUM_TOKENS_1 = 300  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS_1 = 50
    mm_hashes_list_1 = [["hash1_A", "hash1_B", "hash1_A", "hash1_F"]]
    mm_positions_1 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=100, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=200, length=NUM_ENCODER_TOKENS_1),
            PlaceholderRange(offset=250, length=NUM_ENCODER_TOKENS_1),
        ]
    ]
    has_cache_item_result_map_1 = {"hash1_A": False, "hash1_B": True, "hash1_F": True}
    # Create request with 4 MM items, with 2 identical items
    request1 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_1,
        mm_hashes_list=mm_hashes_list_1,
        mm_positions=mm_positions_1,
        max_tokens=1,  # For simplicity
    )[0]

    # Mock partial cache hit: 1st and 3rd missing, 2nd and 4th exist
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_val: has_cache_item_result_map_1[hash_val]
    )
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request1)
    output = scheduler.schedule()

    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request1.request_id]
    assert scheduled_tokens == NUM_TOKENS_1

    # Encoder cache should contain all mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request1])

    # Should have called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called()
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for 2nd and 4th mm item
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request1.mm_features[1], request1.mm_features[3]]
    )

    # Should schedule ONLY 1 encoder input (index 0), no repeat for identical items
    _assert_right_encoder_inputs(
        output,
        requests=[request1],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item ONLY
        expected_total_reqs=1,
    )

    # Simulate model execution 1 step
    model_output = ModelRunnerOutput(
        req_ids=[request1.request_id],
        req_id_to_index={request1.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # request1 is finished after outputting 1 token
    # Finish request
    scheduler.finish_requests(request1.request_id, RequestStatus.FINISHED_LENGTH_CAPPED)

    # Create another request with 4 MM items
    NUM_TOKENS_2 = 400
    NUM_ENCODER_TOKENS_2 = 50
    mm_hashes_list_2 = [["hash1_C", "hash1_D", "hash1_E", "hash1_A"]]
    mm_positions_2 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=100, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=200, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=250, length=NUM_ENCODER_TOKENS_2),
        ]
    ]
    has_cache_item_result_map_2 = {
        "hash1_C": True,
        "hash1_D": False,
        "hash1_E": False,
        "hash1_A": True,
    }
    request2 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_2,
        mm_hashes_list=mm_hashes_list_2,
        mm_positions=mm_positions_2,
        max_tokens=1,  # For simplicity
    )[0]

    # Mock partial cache hit: only hash1_A and hash1_C exist in connector
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_val: has_cache_item_result_map_2[hash_val]
    )

    scheduler.add_request(request2)
    output = scheduler.schedule()

    # Check
    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request2.request_id]
    assert scheduled_tokens == 400

    # Encoder cache should contain all mm items from request2
    _assert_right_encoder_cache_allocated(scheduler, requests=[request2])

    # hash1_A should not be loaded from connector
    # since it's computed in last request & exist in local cache
    # Order of getting encoder cache should be: local cache -> connector-> compute
    # update_state_after_alloc is called for all paths:
    #   index 0 (hash1_C): connector hit → queued for load
    #   index 1 (hash1_D): cache miss → no-op inside connector
    #   index 2 (hash1_E): cache miss → no-op inside connector
    scheduler.ec_connector.update_state_after_alloc.assert_any_call(request2, 0)
    scheduler.ec_connector.update_state_after_alloc.assert_any_call(request2, 1)
    scheduler.ec_connector.update_state_after_alloc.assert_any_call(request2, 2)

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for hash1_C only (index 0)
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[0]]
    )

    # Should schedule 2 encoder input hash1_D and hash1_E (index 1, 2)
    _assert_right_encoder_inputs(
        output,
        requests=[request2],
        expected_encoder_inputs=[[1, 2]],
        expected_total_reqs=1,
    )


@pytest.mark.parametrize("cache_exist", ["local", "connector_only", "no_where"])
@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_schedule_multiple_requests(cache_exist, use_kv_connector):
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_seqs=10,  # allow multiple requests
        max_num_batched_tokens=2048,
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )
    mm_hashes_list = [[f"hash_{i}"] for i in range(10)]
    mm_positions = [[PlaceholderRange(offset=i, length=100)] for i in range(10)]
    requests = create_requests(
        num_requests=10,
        num_tokens=200,
        mm_hashes_list=mm_hashes_list,
        mm_positions=mm_positions,
    )
    for request in requests:
        scheduler.add_request(request)

    # Set up to test different encoder cache existence scenario after preemption
    # Order of getting encoder cache should be: local cache -> connector-> compute
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    if cache_exist == "local":
        # Allocate cache to cache manager manually to mimic
        for req in requests:
            scheduler.encoder_cache_manager.allocate(req, 0)
    else:
        # Make sure local encoder cache empty
        scheduler.encoder_cache_manager.cached = {}

    if cache_exist == "connector_only":
        # Cache exist in ec_connector
        scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    elif cache_exist == "no_where":
        scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    ## Encoder-cache-specific checks:
    # mm_hashes of requests exist in cache after scheduling for all scenario
    _assert_right_encoder_cache_allocated(scheduler, requests=requests)

    if cache_exist == "connector_only":
        scheduler.ec_connector.update_state_after_alloc.assert_called_with(
            requests[-1], 0
        )

        # Concat mm_features for the 10 requests together
        mm_features_list = [feature for req in requests for feature in req.mm_features]

        # Check metadata should contain mm data for all 10 requests
        _assert_right_ec_connector_metadata(output, mm_features_list=mm_features_list)
    elif cache_exist == "local":
        # Local cache hit: items never reach update_state_after_alloc
        scheduler.ec_connector.update_state_after_alloc.assert_not_called()
        _assert_right_ec_connector_metadata(output, mm_features_list=[])
    else:
        # no_where: called from encoder_inputs_to_schedule but no-op
        # inside connector (has_cache_item returns False)
        assert cache_exist == "no_where"
        scheduler.ec_connector.update_state_after_alloc.assert_called()
        _assert_right_ec_connector_metadata(output, mm_features_list=[])

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # Should only schedule encoder input when cache is not found anywhere
    if cache_exist == "no_where":
        _assert_right_encoder_inputs(
            output,
            requests=requests,
            expected_encoder_inputs=[[0] for _ in range(10)],
            expected_total_reqs=10,
        )
    else:
        _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_unable_to_allocate(use_kv_connector):
    """
    Test whether scheduler with ECConnector is able to handle
    unable to allocate (run out of blocks).
    """

    # Setup Scheduler With Mock External Cache Hit.
    BLOCK_SIZE = 4
    NUM_BLOCKS = 10
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Mock ec_connector load external cache behavior
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    # Create two requests. The second request will not be able to
    # allocate slots because it will not have enough blocks.
    NUM_REQUESTS = 2
    NUM_TOKENS = (NUM_BLOCKS // 2 + 1) * BLOCK_SIZE
    MAX_TOKENS = 2
    requests = create_requests(
        num_requests=NUM_REQUESTS,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=[["hash_1"], ["hash_2"]],
        mm_positions=[
            [PlaceholderRange(offset=1, length=10)] for _ in range(NUM_REQUESTS)
        ],
        max_tokens=MAX_TOKENS,
        block_size=BLOCK_SIZE,
    )
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    # Setup MODEL_RUNNER_OUTPUT to be run in _step_until_done later
    MODEL_RUNNER_OUTPUT = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[1000]] * len(req_ids),
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    # Just one request should be running.
    output = scheduler.schedule()
    scheduled_tokens = output.num_scheduled_tokens[scheduler.running[0].request_id]
    assert scheduled_tokens == NUM_TOKENS
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1

    # Should have called update_state_after_alloc for external load
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(
        scheduler.running[0], 0
    )
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # All memory should be freed, with one request waiting.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 1

    # Just one request should be running.
    output = scheduler.schedule()
    scheduled_tokens = output.num_scheduled_tokens[scheduler.running[0].request_id]
    assert scheduled_tokens == NUM_TOKENS
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 0

    # update_state_after_alloc should be called for loading external cache
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(
        scheduler.running[0], 0
    )
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # All memory should be freed, with no requests waiting / running.
    _step_until_done(scheduler, output, MODEL_RUNNER_OUTPUT)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 0


@pytest.mark.parametrize("cache_exist", ["local", "connector_only", "no_where"])
@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_priority_scheduling_ec_connector_preemption_and_resumption(
    cache_exist, use_kv_connector
):
    """Test that priority scheduling preempts lower priority requests
    when out of KV cache space."""
    # Create scheduler with very limited memory to force preemption
    scheduler = create_scheduler_with_priority(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        max_num_seqs=2,  # allow multiple requests
        # kv connector should not effect test results
        use_kv_connector=use_kv_connector,
        num_blocks=15,  # can hold 244 tokens with 14 blocks (first block is null)
        block_size=16,  # standard block size
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Mock cache hit: Both cache exist in connector (at E->PD initially)
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    # Create a request and schedule it (and to be preempted)
    request_low = create_requests_with_priority(
        num_requests=1,
        priorities=[1],
        arrival_times=[0.0],
        num_tokens=94,
        mm_hashes_list=[["hash_low"]],
        # NOTE: this test only preempt the last block.
        # Setting mm_position at the last block can force to recompute encoding
        mm_positions=[[PlaceholderRange(offset=82, length=10)]],
        starting_idx=0,
    )[0]
    scheduler.add_request(request_low)
    # 1st schedule
    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1
    scheduled_tokens = output.num_scheduled_tokens[request_low.request_id]
    assert scheduled_tokens == 94
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    ## Encoder-cache-specific checks:
    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_low])

    # Verify update_state_after_alloc called (external load)
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request_low, 0)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(
        output, mm_features_list=request_low.mm_features
    )

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 1st decode
    model_output = ModelRunnerOutput(
        req_ids=[request_low.request_id],
        req_id_to_index={request_low.request_id: 0},
        sampled_token_ids=[[100]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # Create a high priority request and schedule it
    request_high = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[1.0],
        num_tokens=128,
        mm_hashes_list=[["hash_high"]],
        mm_positions=[[PlaceholderRange(offset=1, length=10)]],
        max_tokens=2,
        starting_idx=1,
    )[0]
    scheduler.add_request(request_high)
    # 2nd schedule
    output = scheduler.schedule()

    # KV cache should be full at this point
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    assert len(output.scheduled_new_reqs) == 1
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 2

    ## Encoder-cache-specific checks:
    # Encoder cache should contain mm items from request
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_high])

    # Verify update_state_after_alloc called (external load)
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request_high, 0)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of request
    _assert_right_ec_connector_metadata(
        output, mm_features_list=request_high.mm_features
    )

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 2nd decode
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100] for _ in requests],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # 3rd schedule - - this should trigger preemption
    # req_low needs 96 tokens = 6 blocks
    # req_high needs 129 tokens = 9 blocks
    # so doesn't fit in 14 blocks.
    output = scheduler.schedule()

    # Should have preempted req_low
    assert len(output.scheduled_new_reqs) == 0
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert output.scheduled_cached_reqs.req_ids[0] == request_high.request_id
    assert scheduler.requests[request_low.request_id].status == RequestStatus.PREEMPTED
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1

    ## Encoder-cache-specific checks:
    # request_high is in decode phase now
    # ECConnector should carry no metadata
    _assert_right_ec_connector_metadata(output, mm_features_list=[])

    # Scheduled encoder input should be empty; no mm to compute
    _assert_right_encoder_inputs(output, expected_total_reqs=0)

    # Simulate model execution - 3rd decode, after req_low was preempted
    requests = [request_low, request_high]
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={req.request_id: i for i, req in enumerate(requests)},
        sampled_token_ids=[[100], [100, 200]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    # Finish the requests to make room for the preempted requests to resume
    # req_high is finished after outputting 2 tokens
    scheduler.update_from_output(output, model_output)
    scheduler.finish_requests(
        request_high.request_id, RequestStatus.FINISHED_LENGTH_CAPPED
    )

    # Set up to test different encoder cache existence scenario after preemption
    # Order of getting encoder cache should be: local cache -> connector-> compute
    # By default, the cache should still exist in local in this test case
    if cache_exist != "local":
        # Make local encoder cache empty
        scheduler.encoder_cache_manager.cached = {}

    if cache_exist == "connector_only":
        # Cache exist in ec_connector
        scheduler.ec_connector.has_cache_item = Mock(return_value=True)
    elif cache_exist == "no_where":
        scheduler.ec_connector.has_cache_item = Mock(return_value=False)

    # 4th Schedule - this should trigger req_low resumption from waiting
    output = scheduler.schedule()
    scheduled_cached_reqs = output.scheduled_cached_reqs

    assert len(output.scheduled_new_reqs) == 0
    assert scheduled_cached_reqs.num_reqs == 1
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1

    # Preempted request resumed in scheduled_cached_reqs
    assert len(scheduled_cached_reqs.resumed_req_ids) == 1
    assert len(scheduled_cached_reqs.all_token_ids) == 1
    assert scheduled_cached_reqs.req_ids[0] == request_low.request_id
    assert request_low.request_id in scheduled_cached_reqs.resumed_req_ids
    assert request_low.request_id in scheduled_cached_reqs.all_token_ids
    ## Resumed tokens include 94 prompt tokens and 2 decoded tokens
    assert len(scheduled_cached_reqs.all_token_ids[request_low.request_id]) == 96
    assert scheduled_cached_reqs.all_token_ids[request_low.request_id][95] == 100
    assert scheduler.running[0].request_id == request_low.request_id
    assert request_high.request_id in output.finished_req_ids

    ## Encoder-cache-specific checks:
    # mm_hash of request_low exists in cache after scheduling for all scenario
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_low])

    if cache_exist == "connector_only":
        scheduler.ec_connector.update_state_after_alloc.assert_called_with(
            request_low, 0
        )
        _assert_right_ec_connector_metadata(
            output, mm_features_list=request_low.mm_features
        )
    elif cache_exist == "local":
        scheduler.ec_connector.update_state_after_alloc.assert_not_called()
        _assert_right_ec_connector_metadata(output, mm_features_list=[])
    else:
        assert cache_exist == "no_where"
        scheduler.ec_connector.update_state_after_alloc.assert_called_with(
            request_low, 0
        )
        _assert_right_ec_connector_metadata(output, mm_features_list=[])

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # Should only schedule encoder input when cache is not found anywhere
    if cache_exist == "no_where":
        _assert_right_encoder_inputs(
            output,
            requests=[request_low],
            expected_encoder_inputs=[[0]],
            expected_total_reqs=1,
        )
    else:
        _assert_right_encoder_inputs(output, expected_total_reqs=0)


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_allocate_encoder_tokens_with_external_load(use_kv_connector):
    """
    Scenario:
      - Encoder cache size: 32
      - Request A: 1 feature (12 tokens) → NOT cached remotely.
      - Request B: 3 features (3 x 10 tokens) → ALL cached remotely.

    Steps:
      1. Schedule Request A (locally uses 12 tokens).
      2. Schedule Request B (remote cache) - only schedule 1st and 2nd
      3. Free A's cache, then schedule B again (continuation) - schedule 3rd image
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        block_size=16,
        num_blocks=11,  # Can hold 160 tokens (first block is null)
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    # Limit the number of available slots of EncoderCacheManager
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=32)

    # Create MM request1
    NUM_TOKENS_1 = 50  # NOTE: includes mm tokens
    NUM_ENCODER_TOKENS_1 = 12
    mm_hashes_list_1 = [["hash1_1"]]
    mm_positions_1 = [[PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_1)]]

    request1 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_1,
        mm_hashes_list=mm_hashes_list_1,
        mm_positions=mm_positions_1,
        max_tokens=1,  # For simplicity
        req_ids=["req1"],
    )[0]

    # Create MM request1 with 3 MM items
    NUM_TOKENS_2 = 40
    NUM_ENCODER_TOKENS_2 = 10
    mm_hashes_list_2 = [["hash2_1", "hash2_2", "hash2_3"]]
    mm_positions_2 = [
        [
            PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=12, length=NUM_ENCODER_TOKENS_2),
            PlaceholderRange(offset=24, length=NUM_ENCODER_TOKENS_2),
        ]
    ]

    request2 = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS_2,
        mm_hashes_list=mm_hashes_list_2,
        mm_positions=mm_positions_2,
        max_tokens=10,
        req_ids=["req2"],
    )[0]

    # Mock cache hit: MM of request1 NOT cached remotely, request2 cached remotely
    scheduler.ec_connector.has_cache_item = Mock(
        side_effect=lambda hash_value: hash_value in mm_hashes_list_2[0]
    )
    scheduler.ec_connector.update_state_after_alloc = Mock(
        wraps=scheduler.ec_connector.update_state_after_alloc
    )

    scheduler.add_request(request1)
    scheduler.add_request(request2)
    output = scheduler.schedule()

    # Now, since encoder cache manager can only store 32 tokens
    # It should allocated mm item hash1_1, hash2_1 and hash2_2
    scheduled_tokens = output.num_scheduled_tokens[request1.request_id]
    assert scheduled_tokens == NUM_TOKENS_1
    assert scheduler.get_num_unfinished_requests() == 2

    # Encoder cache should contain mm item from request1
    _assert_right_encoder_cache_allocated(
        scheduler, hashes_to_check=["hash1_1", "hash2_1", "hash2_2"]
    )

    # request2's 2nd mm item is the last call of update_state_after_alloc
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request2, 1)
    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata of hash2_1 and hash2_2 ONLY
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[0], request2.mm_features[1]]
    )

    # Should schedule ONLY 1 encoder input
    _assert_right_encoder_inputs(
        output,
        requests=[request1],
        expected_encoder_inputs=[[0]],  # index 0 of the mm item of request1
        expected_total_reqs=1,
    )

    # Simulate model execution 1 step
    model_output = ModelRunnerOutput(
        req_ids=[request1.request_id, request2.request_id],
        req_id_to_index={request1.request_id: 0, request2.request_id: 1},
        sampled_token_ids=[[100], [121]],
        # spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)

    # request1 is finished after outputting 1 token
    # Finish request
    scheduler.finish_requests(request1.request_id, RequestStatus.FINISHED_LENGTH_CAPPED)
    assert scheduler.get_num_unfinished_requests() == 1

    # Schedule again; Now request1's encoder cache should be freed
    # -> hash2_3 can be scheduled and allocated
    output = scheduler.schedule()

    # Check
    # Should schedule all tokens
    scheduled_tokens = output.num_scheduled_tokens[request2.request_id]
    print(f"Hero: scheduled_tokens for req2: {scheduled_tokens}")
    print(f"hero: num_scheduled_tokens 2: {output.num_scheduled_tokens}")

    # Encoder cache should contain all mm items from request2
    _assert_right_encoder_cache_allocated(scheduler, requests=[request2])

    # request2's 3rd mm item is the ONLY call of update_state_after_alloc
    scheduler.ec_connector.update_state_after_alloc.assert_called_with(request2, 2)
    scheduler.ec_connector.update_state_after_alloc.assert_called_once()

    scheduler.ec_connector.update_state_after_alloc.reset_mock()

    # ECConnector should carry metadata for hash2_3 ONLY
    _assert_right_ec_connector_metadata(
        output, mm_features_list=[request2.mm_features[2]]
    )

    # Should schedule no encoder input
    _assert_right_encoder_inputs(
        output,
        expected_total_reqs=0,
    )


# ==============================================================================
# EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests end
# ==============================================================================


def test_prepend_skipped_requests_order():
    scheduler = create_scheduler(max_num_seqs=1, use_kv_connector=True)
    requests = create_requests(num_requests=4)
    for request in requests:
        scheduler.add_request(request)

    # 4 requests waiting, capture their order
    expected_waiting_reqs = list(scheduler.waiting)

    # simulate first 2 waiting requests are waiting for remote KVs
    for req in expected_waiting_reqs[:2]:
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.waiting.remove_requests(expected_waiting_reqs[:2])
    for req in expected_waiting_reqs[:2]:
        scheduler.skipped_waiting.add_request(req)

    # schedule step
    # expect the first 2 waiting to be skipped, the third running,
    # and the fourth waiting
    scheduler.schedule()

    # pop the third request which is expected to be running
    expected_waiting_reqs.pop(2)

    # verify waiting order is preserved
    waiting_reqs = list(scheduler.skipped_waiting) + list(scheduler.waiting)
    assert waiting_reqs == expected_waiting_reqs


def test_remote_kv_promotion_keeps_fcfs_with_grammar_prefix():
    scheduler = create_scheduler(max_num_seqs=1)
    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.return_value = (0, False)

    requests = create_requests(num_requests=4)
    for request in requests:
        scheduler.add_request(request)

    req_grammar_1, req_grammar_2, req_remote, req_tail = list(scheduler.waiting)

    # simulate two structured-output grammar requests at the waiting head
    # that become ready now.
    req_grammar_1.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar_1.structured_output_request = Mock(grammar=object())
    req_grammar_2.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar_2.structured_output_request = Mock(grammar=object())

    # simulate a remote-KV request that is ready to be promoted now.
    req_remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.waiting.remove_requests([req_grammar_1, req_grammar_2, req_remote])
    scheduler.skipped_waiting.add_request(req_grammar_1)
    scheduler.skipped_waiting.add_request(req_grammar_2)
    scheduler.skipped_waiting.add_request(req_remote)
    scheduler.finished_recving_kv_req_ids.add(req_remote.request_id)
    scheduler._update_waiting_for_remote_kv = Mock()

    output = scheduler.schedule()

    assert output.scheduled_new_reqs
    assert output.scheduled_new_reqs[0].req_id == req_grammar_1.request_id
    waiting_req_ids = [
        req.request_id
        for req in list(scheduler.skipped_waiting) + list(scheduler.waiting)
    ]
    assert waiting_req_ids == [
        req_grammar_2.request_id,
        req_remote.request_id,
        req_tail.request_id,
    ]


def test_fcfs_mixed_skipped_waiting_types_keep_order():
    scheduler = create_scheduler(max_num_batched_tokens=20)
    scheduler._update_waiting_for_remote_kv = Mock()

    mk_req = lambda req_id, num_tokens=1: create_requests(  # noqa: E731
        num_requests=1, num_tokens=num_tokens, req_ids=[req_id]
    )[0]
    req_grammar, req_remote, req_stream = (
        mk_req("grammar"),
        mk_req("remote"),
        mk_req("stream"),
    )
    req_regular, req_tail = mk_req("regular", 20), mk_req("tail")
    req_grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    req_grammar.structured_output_request = Mock(grammar=None)
    req_remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    req_stream.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    for req in (req_grammar, req_remote, req_stream, req_regular, req_tail):
        scheduler.add_request(req)
    scheduler.schedule()
    assert list(scheduler.skipped_waiting) == [req_grammar, req_remote, req_stream]

    scheduler.finish_requests(req_regular.request_id, RequestStatus.FINISHED_ABORTED)
    assert not scheduler.running

    req_grammar.structured_output_request = Mock(grammar=object())
    scheduler.finished_recving_kv_req_ids.add(req_remote.request_id)
    req_stream.status = RequestStatus.WAITING

    second_output = scheduler.schedule()
    expected_order = [
        req_grammar.request_id,
        req_remote.request_id,
        req_stream.request_id,
        req_tail.request_id,
    ]
    assert [req.req_id for req in second_output.scheduled_new_reqs] == expected_order
    assert [req.request_id for req in scheduler.running] == expected_order
    scheduler._update_waiting_for_remote_kv.assert_called_once_with(req_remote)


def test_abort_request_waiting_for_remote_kvs():
    scheduler = create_scheduler(use_kv_connector=True)

    # add a single request
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    # set request to waiting for remote KVs, and abort it
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.finish_requests((request.request_id,), RequestStatus.FINISHED_ABORTED)
    assert request.status == RequestStatus.FINISHED_ABORTED

    # verify request is not deleted
    assert request.request_id in scheduler.requests

    # finish recving request
    scheduler_output = scheduler.schedule()
    model_runner_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_connector_output=KVConnectorOutput(finished_recving={request.request_id}),
    )
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # assert request is deleted
    assert request.request_id not in scheduler.requests
    assert not scheduler.finished_recving_kv_req_ids


def test_abort_request_finished_recving():
    scheduler = create_scheduler(use_kv_connector=True)

    # add a single request
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    # set request to waiting for remote KVs, finished but not yet updated
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.finished_recving_kv_req_ids.add(request.request_id)

    # abort request
    scheduler.finish_requests((request.request_id,), RequestStatus.FINISHED_ABORTED)
    assert request.status == RequestStatus.FINISHED_ABORTED

    # verify request is deleted
    assert request.request_id not in scheduler.requests
    assert not scheduler.finished_recving_kv_req_ids


def test_delayed_kv_connector_free_keeps_scheduler_active():
    scheduler = create_scheduler(use_kv_connector=True)
    queued_request, request = create_requests(
        num_requests=2, req_ids=["queued", "finished"]
    )
    scheduler.add_request(queued_request)

    assert not scheduler.has_finished_requests()

    request.status = RequestStatus.FINISHED_STOPPED
    scheduler.requests[request.request_id] = request
    scheduler.finished_req_ids = set()

    assert scheduler.has_finished_requests()
    assert scheduler.has_requests()

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_connector_output=KVConnectorOutput(finished_sending={request.request_id}),
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert request.request_id not in scheduler.requests
    assert not scheduler.has_finished_requests()


def test_scheduler_kv_connector_stats():
    """Test worker-side, scheduler-side, and combined KV connector stats."""

    class GenericKVConnectorStats(KVConnectorStats):
        def reset(self):
            self.data = {}

        def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
            self.data.update(other.data)
            return self

        def reduce(self) -> dict[str, int | float]:
            return {}

        def is_empty(self) -> bool:
            return not self.data

    test_cases = (
        ({"worker": 1}, None, {"worker": 1}),
        (None, {"scheduler": 2}, {"scheduler": 2}),
        ({"worker": 1}, {"scheduler": 2}, {"worker": 1, "scheduler": 2}),
    )

    for worker_data, scheduler_data, expected_data in test_cases:
        scheduler = create_scheduler()
        worker_stats = (
            GenericKVConnectorStats(data=worker_data) if worker_data else None
        )
        scheduler_stats = (
            GenericKVConnectorStats(data=scheduler_data) if scheduler_data else None
        )
        scheduler.connector = Mock()
        scheduler.connector.get_kv_connector_stats.return_value = (
            scheduler_stats if worker_stats is None else None
        )
        scheduler.connector.take_events.return_value = []

        def update_connector_output(
            kv_connector_output: KVConnectorOutput,
            scheduler=scheduler,
            scheduler_stats=scheduler_stats,
        ):
            scheduler.connector.get_kv_connector_stats.return_value = scheduler_stats

        scheduler.connector.update_connector_output.side_effect = (
            update_connector_output
        )

        model_output = ModelRunnerOutput(
            req_ids=["req_0"],
            req_id_to_index={"req_0": 0},
            sampled_token_ids=[[123]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None],
            kv_connector_output=KVConnectorOutput(kv_connector_stats=worker_stats)
            if worker_stats
            else None,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=None,
            num_scheduled_tokens={"req_0": 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

        engine_core_outputs = scheduler.update_from_output(
            scheduler_output, model_output
        )

        final_stats = next(
            iter(engine_core_outputs.values())
        ).scheduler_stats.kv_connector_stats
        assert final_stats == expected_data


def test_ec_connector_update_connector_output_called():
    """Test that worker-side EC connector output is forwarded to the
    EC connector's update_connector_output hook."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        use_ec_connector=True,
        ec_role="ec_consumer",
    )
    scheduler.ec_connector.update_connector_output = Mock()

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    ec_connector_output = ECConnectorOutput(finished_sending={"hash_test1"})
    model_runner_output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        ec_connector_output=ec_connector_output,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    scheduler.ec_connector.update_connector_output.assert_called_once_with(
        ec_connector_output
    )


# ==============================================================================
# Variable-length encoder cross-attention block allocation tests
# ==============================================================================


def _create_encoder_decoder_scheduler(
    block_size: int = 16,
    num_blocks: int = 10000,
    max_num_batched_tokens: int = 8192,
    max_num_seqs: int = 16,
) -> Scheduler:
    """Create a scheduler configured for encoder-decoder cross-attention
    block allocation testing.

    Constructs a scheduler with both FullAttentionSpec (self-attention) and
    CrossAttentionSpec (cross-attention) KV cache groups, then patches it
    to behave as an encoder-decoder model.
    """
    from vllm.v1.core.encoder_cache_manager import EncoderDecoderCacheManager
    from vllm.v1.kv_cache_interface import CrossAttentionSpec

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        # is_encoder_decoder disables chunked prefill and prefix caching
        is_encoder_decoder=True,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    cache_config.num_gpu_blocks = num_blocks

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )

    # KV cache config with both self-attention and cross-attention groups,
    # mirroring an encoder-decoder model like Whisper.
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["self_attn_layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["cross_attn_layer"],
                CrossAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
        ],
    )

    # Construct the scheduler. Since opt-125m is not truly encoder-decoder,
    # the __init__ won't set up encoder-decoder internals. We patch them
    # after construction.
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        hash_block_size=block_size,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )

    # Patch to enable encoder-decoder behavior in the scheduling loop.
    scheduler.is_encoder_decoder = True
    scheduler.max_num_encoder_input_tokens = max_num_batched_tokens
    scheduler.encoder_cache_manager = EncoderDecoderCacheManager(
        cache_size=max_num_batched_tokens
    )

    return scheduler


def _get_num_cross_attn_blocks(scheduler: Scheduler, request_id: str) -> int:
    """Get the number of cross-attention blocks allocated for a request."""
    from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager

    coordinator = scheduler.kv_cache_manager.coordinator
    for manager in coordinator.single_type_managers:
        if isinstance(manager, CrossAttentionManager):
            blocks = manager.req_to_blocks.get(request_id, [])
            return len(blocks)
    raise AssertionError("No CrossAttentionManager found in coordinator")


def test_variable_length_cross_attn_block_allocation():
    """Test that cross-attention blocks are allocated per-request based on
    actual encoder input length, not a fixed maximum.

    Fixed max-encoder-length allocation would assign
    `ceil(max_encoder_tokens / block_size)` blocks to
    every request whereas with dynamic allocation, exactly
    `ceil(actual_encoder_tokens / block_size)` blocks are assigned
    to each request.
    """
    block_size = 16
    scheduler = _create_encoder_decoder_scheduler(block_size=block_size)

    # Create requests with distinctly different encoder input lengths,
    # simulating variable-length audio inputs to a model like Whisper.
    encoder_lengths = [500, 1000, 200]
    num_prompt_tokens = 100  # Decoder prompt tokens

    requests = []
    for i, enc_len in enumerate(encoder_lengths):
        req = create_requests(
            num_requests=1,
            num_tokens=num_prompt_tokens,
            mm_hashes_list=[[f"enc_hash_{i}"]],
            mm_positions=[[PlaceholderRange(offset=0, length=enc_len)]],
            req_ids=[f"req_{i}"],
        )[0]
        requests.append(req)

    # Add and schedule all requests.
    for req in requests:
        scheduler.add_request(req)

    output = scheduler.schedule()

    # All requests should be scheduled.
    assert len(output.scheduled_new_reqs) == len(requests)

    # Verify cross-attention blocks per request match the actual encoder length.
    from math import ceil

    for req, enc_len in zip(requests, encoder_lengths):
        expected_blocks = ceil(enc_len / block_size)
        actual_blocks = _get_num_cross_attn_blocks(scheduler, req.request_id)

        assert actual_blocks == expected_blocks, (
            f"Request {req.request_id} with {enc_len} encoder tokens: "
            f"expected {expected_blocks} cross-attn blocks, "
            f"got {actual_blocks}"
        )

    # Verify that different encoder lengths produce different block counts,
    # confirming variable-length (not fixed-max) allocation.
    block_counts = [
        _get_num_cross_attn_blocks(scheduler, req.request_id) for req in requests
    ]
    assert len(set(block_counts)) > 1, (
        "All requests have the same number of cross-attn blocks, "
        "suggesting static max-based allocation instead of per-request"
    )


def test_cross_attn_blocks_not_over_allocated():
    """Test that cross-attention blocks are not over-allocated compared to
    what each request actually needs."""
    from math import ceil

    block_size = 16
    max_encoder_tokens = 1500  # e.g., Whisper's max mel-spectrogram length
    scheduler = _create_encoder_decoder_scheduler(block_size=block_size)

    # Request with a small encoder input (much less than the max).
    small_enc_len = 200
    request = create_requests(
        num_requests=1,
        num_tokens=100,
        mm_hashes_list=[["enc_small"]],
        mm_positions=[[PlaceholderRange(offset=0, length=small_enc_len)]],
        req_ids=["req_small"],
    )[0]

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1

    actual_blocks = _get_num_cross_attn_blocks(scheduler, request.request_id)
    expected_blocks = ceil(small_enc_len / block_size)
    max_blocks = ceil(max_encoder_tokens / block_size)

    # Blocks should match the actual encoder length.
    assert actual_blocks == expected_blocks, (
        f"Expected {expected_blocks} blocks for {small_enc_len} encoder tokens, "
        f"got {actual_blocks}"
    )

    # Blocks should be strictly less than what max-based allocation would give.
    assert actual_blocks < max_blocks, (
        f"Cross-attn blocks ({actual_blocks}) should be less than max "
        f"({max_blocks}), indicating no over-allocation"
    )


def test_cross_attn_blocks_not_under_allocated():
    """Test that cross-attention blocks are sufficient for each request's
    actual encoder input length. Every encoder token must have a slot.

    Tests various edge cases including exact block boundaries, off-by-one,
    and the minimum/maximum encoder input sizes.
    """
    from math import ceil

    block_size = 16

    # Test various encoder lengths including edge cases around block boundaries.
    test_cases = [
        1,  # Minimum: single encoder token
        block_size - 1,  # Just under one full block
        block_size,  # Exactly one full block
        block_size + 1,  # Just over one block (needs 2 blocks)
        block_size * 10,  # Exact multiple of block size
        block_size * 10 + 1,  # One over exact multiple
        1500,  # Whisper's typical max
    ]

    for enc_len in test_cases:
        scheduler = _create_encoder_decoder_scheduler(block_size=block_size)

        request = create_requests(
            num_requests=1,
            num_tokens=100,
            mm_hashes_list=[[f"enc_{enc_len}"]],
            mm_positions=[[PlaceholderRange(offset=0, length=enc_len)]],
            req_ids=[f"req_{enc_len}"],
        )[0]

        scheduler.add_request(request)
        output = scheduler.schedule()

        assert len(output.scheduled_new_reqs) == 1

        actual_blocks = _get_num_cross_attn_blocks(scheduler, request.request_id)
        expected_blocks = ceil(enc_len / block_size)

        # Number of blocks must be exactly ceil(enc_len / block_size).
        assert actual_blocks == expected_blocks, (
            f"Encoder length {enc_len}: expected {expected_blocks} blocks, "
            f"got {actual_blocks}"
        )

        # Total available slots must be >= encoder tokens (no under-allocation).
        total_slots = actual_blocks * block_size
        assert total_slots >= enc_len, (
            f"Encoder length {enc_len}: total slots {total_slots} < "
            f"needed {enc_len} (under-allocation)"
        )


def test_cross_attn_zero_blocks_without_encoder_inputs():
    """Test that requests without encoder inputs get zero cross-attention
    blocks, even when the scheduler is configured for encoder-decoder."""
    block_size = 16
    scheduler = _create_encoder_decoder_scheduler(block_size=block_size)

    # Create a text-only request (no mm_features).
    request = create_requests(
        num_requests=1,
        num_tokens=100,
        req_ids=["req_text_only"],
    )[0]

    # Text-only request has no encoder inputs.
    assert not request.has_encoder_inputs

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 1

    # No cross-attention blocks should be allocated.
    actual_blocks = _get_num_cross_attn_blocks(scheduler, request.request_id)
    assert actual_blocks == 0, (
        f"Text-only request should have 0 cross-attn blocks, got {actual_blocks}"
    )


def test_eagle3_mm_encoder_cache_with_shift(monkeypatch: pytest.MonkeyPatch):
    """Test EAGLE3 encoder scheduling accounts for shift_computed_tokens.

    Regression test for issue #32469: When EAGLE3 is enabled with
    disable_chunked_mm_input=True, ensure encoder inputs are scheduled
    when tokens overlap the MM range, properly accounting for
    shift_computed_tokens in the boundary calculation.

    Without the fix, the scheduler would fail to schedule encoder inputs
    at the boundary, causing "Encoder cache miss" errors.
    """
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        disable_chunked_mm_input=True,
        max_model_len=2048,
        num_speculative_tokens=4,  # This enables EAGLE with shift=1
    )
    # This upstream regression isolates EAGLE's MM boundary, not the elastic
    # Graph maintenance protocol exercised by dedicated tests below.
    scheduler.elastic_on_demand_graphs = False
    monkeypatch.setattr(
        scheduler, "_canonical_elastic_graph_step_key", Mock(return_value=None)
    )

    mm_start_pos = 100
    mm_length = 576

    mm_positions = [
        [PlaceholderRange(offset=mm_start_pos, length=mm_length)],
    ]

    requests = create_requests(
        num_requests=1,
        num_tokens=mm_start_pos + mm_length + 100,
        mm_positions=mm_positions,
    )

    # Start with some tokens already computed to simulate decoding
    request = requests[0]
    request.num_computed_tokens = 0

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output is not None
    shift_computed_tokens = 1
    req_id = request.request_id

    assert req_id in output.num_scheduled_tokens
    num_scheduled = output.num_scheduled_tokens[req_id]

    mm_feature = request.mm_features[0]
    start_pos = mm_feature.mm_position.offset
    tokens_end = request.num_computed_tokens + num_scheduled
    scheduled_end_with_shift = tokens_end + shift_computed_tokens

    # Assert that we scheduled into the MM range (test setup verification)
    assert scheduled_end_with_shift > start_pos, (
        f"Test setup error: expected to schedule into MM range. "
        f"scheduled_end_with_shift={scheduled_end_with_shift}, "
        f"start_pos={start_pos}"
    )

    # The key assertion: when scheduled tokens overlap MM range
    # (accounting for EAGLE's shift), encoder MUST be scheduled.
    # Without the fix, this would fail at the boundary case.
    assert req_id in output.scheduled_encoder_inputs, (
        f"Encoder input missing: scheduled {num_scheduled} tokens "
        f"(computed={request.num_computed_tokens}, end={tokens_end}, "
        f"shifted_end={scheduled_end_with_shift}) overlapping MM at "
        f"{start_pos}. The fix must schedule encoder inputs."
    )


def test_free_encoder_inputs_respects_unconfirmed_placeholders():
    """Regression test for issue #38551 (rollback path): under async
    scheduling with speculative decoding, num_computed_tokens is advanced
    optimistically and can be rolled back when in-flight draft tokens are
    rejected. Freeing an encoder input as soon as num_computed_tokens passes
    the end of its placeholder range allows a later rollback to rewind back
    into the range, after which the worker's MM-embedding gather reads an
    evicted entry and crashes the engine with "Encoder cache miss". The
    scheduler must retain the input until the *confirmed* progress
    (num_computed_tokens - num_output_placeholders) passes the range end, so
    that no pending rejection can rewind into the range."""
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        num_speculative_tokens=3,
    )
    mm_start_pos = 50
    mm_length = 100
    mm_positions = [
        [PlaceholderRange(offset=mm_start_pos, length=mm_length)],
    ]
    request = create_requests(
        num_requests=1,
        num_tokens=mm_start_pos + mm_length + 100,
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)
    mm_end = mm_start_pos + mm_length

    # One optimistically-scheduled in-flight step advanced num_computed_tokens
    # by 1 sampled + 3 draft tokens; none are confirmed yet, so all 4 are
    # still output placeholders that a rejection could rewind.
    request.num_output_placeholders = 4

    # Optimistic progress reaches the end of the MM range, but the confirmed
    # position (mm_end + 1 - 4) is still inside it: a rejection could rewind
    # back into the range, so the entry must be retained.
    request.num_computed_tokens = mm_end + 1
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == {0}

    # Confirmed position still inside the range.
    request.num_computed_tokens = mm_end + 3
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == {0}

    # Confirmed position (mm_end + 4 - 4) now reaches the range end: even if
    # every unconfirmed token is rejected, progress cannot rewind into the
    # range, so the entry is freed.
    request.num_computed_tokens = mm_end + 4
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == set()


def test_free_encoder_inputs_notifies_the_ec_connector():
    """The connector learns the item is consumed, not just that the request ended.

    Connectors hold per-item transfer state (remote buffers, reservations);
    waiting for `request_finished` pins it for the whole generation.
    """
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    mm_start_pos, mm_length = 50, 100
    request = create_requests(
        num_requests=1,
        num_tokens=mm_start_pos + mm_length + 10,
        mm_positions=[[PlaceholderRange(offset=mm_start_pos, length=mm_length)]],
    )[0]
    scheduler.encoder_cache_manager.allocate(request, 0)
    scheduler.ec_connector = Mock()

    request.num_computed_tokens = mm_start_pos + mm_length - 1
    scheduler._free_encoder_inputs(request)
    scheduler.ec_connector.update_state_after_free.assert_not_called()

    request.num_computed_tokens = mm_start_pos + mm_length
    scheduler._free_encoder_inputs(request)
    scheduler.ec_connector.update_state_after_free.assert_called_once_with(request, 0)


def test_unavailable_encoder_input_fails_the_request_as_retryable():
    """An encoder input the connector gave up on must end the request.

    `FinishReason.ERROR` is the retryable channel the KV connector already uses
    for load failures, so the caller can re-issue; deferring instead parked the
    request until the client timed out.
    """
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    request = create_requests(
        num_requests=1,
        num_tokens=160,
        mm_positions=[[PlaceholderRange(offset=50, length=100)]],
    )[0]
    scheduler.add_request(request)
    scheduler.ec_connector = Mock()
    scheduler.ec_connector.take_unavailable_requests.return_value = {request.request_id}
    # `request_finished` reports (delay_free, params) for the request teardown.
    scheduler.ec_connector.request_finished.return_value = (False, None)

    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    engine_outputs = outputs[0].outputs
    assert [o.finish_reason for o in engine_outputs] == [FinishReason.ERROR]


def test_free_encoder_inputs_defers_for_eagle_lookahead():
    """With EAGLE speculative decoding, the encoder input is retained one extra
    position so the drafter's +1 look-ahead mm-embedding gather (which reads one
    position past the target's computed range) still finds it cached. This is
    the primary mechanism that prevents the drafter "Encoder cache miss"; the
    worker-side token-embedding fallback is only a backstop."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    # create_scheduler only builds ngram spec configs; force the eagle path that
    # _free_encoder_inputs keys off (its read-ahead deferral).
    scheduler.use_eagle = True
    scheduler.use_eagle_block_drop = True
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_prefill_checkpoint_alignment = None
    scheduler.mamba_fine_grained_prefix_cache = False
    scheduler.num_prefill_lookahead = 1
    mm_positions = [[PlaceholderRange(offset=50, length=100)]]
    request = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)
    mm_end = 150  # offset + length

    # Confirmed progress reaches the range end: without spec decode this frees
    # (see test below), but the drafter's +1 look-ahead still needs it.
    request.num_computed_tokens = mm_end
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == {0}

    # One position past the range end: the +1 look-ahead has now passed it.
    request.num_computed_tokens = mm_end + 1
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == set()


def test_free_encoder_inputs_unchanged_without_spec_decode():
    """Without speculative decoding, encoder inputs are freed as soon as
    num_computed_tokens passes the placeholder range, as before."""
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    mm_positions = [[PlaceholderRange(offset=50, length=100)]]
    request = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    manager.allocate(request, 0)

    request.num_computed_tokens = 149
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == {0}

    request.num_computed_tokens = 150
    scheduler._free_encoder_inputs(request)
    assert manager.get_cached_input_ids(request) == set()


def test_encoder_cache_retained_across_preemption_and_resume(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression guard for issue #38551 (preemption path).

    A request preempted under KV pressure resets num_computed_tokens to 0
    and drops its encoder references (scheduler._preempt_request calls
    encoder_cache_manager.free). Because that only moves the entry into
    `freeable` (it is not evicted), the worker still holds it: the scheduler
    must NOT report the mm_hash as freed. On resume, re-requesting the
    encoder input must pull the still-cached entry back out of `freeable`
    without scheduling a recompute, keeping the scheduler and worker
    consistent. The spec-rollback retention margin does not gate this path,
    so it is covered separately here."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        num_speculative_tokens=3,
    )
    mm_positions = [[PlaceholderRange(offset=50, length=100)]]
    request = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_hashes_list=[["img_a"]],
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    mm_hash = request.mm_features[0].identifier

    # Prefill scheduled and computed the encoder input; it is pinned.
    manager.allocate(request, 0)
    assert manager.get_cached_input_ids(request) == {0}

    # Preemption drops the request's encoder references (scheduler.py:
    # _preempt_request -> encoder_cache_manager.free) and resets progress.
    manager.free(request)
    request.num_computed_tokens = 0
    # The entry is now ref-free but only `freeable` (not evicted): the
    # worker still holds it, so nothing must be reported as freed.
    assert mm_hash in manager.cached
    assert mm_hash in manager.freeable
    assert manager.get_freed_mm_hashes() == []

    # Resume re-requests the encoder output. The still-cached entry is pulled
    # back out of `freeable` with no recompute and no worker-side free.
    assert manager.check_and_update_cache(request, 0) is True
    assert mm_hash not in manager.freeable
    assert manager.get_cached_input_ids(request) == {0}
    assert manager.get_freed_mm_hashes() == []


def test_encoder_cache_recomputed_when_evicted_during_preemption(
    monkeypatch: pytest.MonkeyPatch,
):
    """Companion to the retention case (issue #38551, preemption path).

    If a preempted request's retained encoder entry IS evicted under memory
    pressure before it resumes, the scheduler reports the mm_hash as freed
    (so the worker drops it) and a resume must schedule a recompute rather
    than assume the worker still holds it. check_and_update_cache must
    return False so the encoder input is re-scheduled."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        num_speculative_tokens=3,
    )
    mm_positions = [[PlaceholderRange(offset=50, length=100)]]
    request = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_hashes_list=[["img_a"]],
        mm_positions=mm_positions,
    )[0]
    manager = scheduler.encoder_cache_manager
    mm_hash = request.mm_features[0].identifier

    manager.allocate(request, 0)
    # Preemption drops references; the entry becomes freeable.
    manager.free(request)
    request.num_computed_tokens = 0
    assert mm_hash in manager.freeable

    # A new request with a different image hits memory pressure and evicts
    # the freeable entry to make room.
    other = create_requests(
        num_requests=1,
        num_tokens=250,
        mm_hashes_list=[["img_b"]],
        mm_positions=mm_positions,
        req_ids=["1"],
    )[0]
    manager.num_free_slots = 50  # force eviction of the freeable entry
    assert manager.can_allocate(
        other, 0, encoder_compute_budget=10_000, num_embeds_to_schedule=0
    )

    # The evicted entry is reported to the worker, which drops it.
    assert mm_hash not in manager.cached
    assert manager.get_freed_mm_hashes() == [mm_hash]

    # On resume the original request must recompute (cache miss is correct).
    assert manager.check_and_update_cache(request, 0) is False


@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_ec_connector_ensure_cache_available_defers_request(use_kv_connector):
    """Test that ensure_cache_available() returning False defers the request.

    When the EC connector signals a prefetch is in progress (returns False),
    the scheduler should:
    1. Not schedule the request (no KV cache or encoder cache allocated)
    2. Still schedule other requests behind the deferred one
    3. Schedule the deferred request on the next step when ensure_cache_available
       returns True and has_cache_item returns True
    """
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        enable_prefix_caching=True,
        use_kv_connector=use_kv_connector,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )

    NUM_TOKENS = 200
    NUM_ENCODER_TOKENS = 100

    request_deferred = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
        mm_positions=[[PlaceholderRange(offset=0, length=NUM_ENCODER_TOKENS)]],
        req_ids=["deferred"],
    )[0]

    request_behind = create_requests(
        num_requests=1,
        num_tokens=20,
        req_ids=["behind"],
    )[0]

    # --- Step 1: ensure_cache_available returns False → request deferred ---
    scheduler.ec_connector.ensure_cache_available = Mock(return_value=False)

    scheduler.add_request(request_deferred)
    scheduler.add_request(request_behind)
    output = scheduler.schedule()

    # ensure_cache_available must have been called with (request, num_computed_tokens=0)
    # for a brand-new request that has no cached tokens yet.
    ensure_call = scheduler.ec_connector.ensure_cache_available.call_args
    assert ensure_call.args == (request_deferred, 0)
    # Deferred request must NOT be scheduled
    assert request_deferred.request_id not in output.num_scheduled_tokens
    _assert_right_encoder_cache_allocated(scheduler, expected_total_allocated=0)
    # No KV blocks allocated for the deferred request
    for mgr in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert request_deferred.request_id not in mgr.req_to_blocks

    # The text-only request behind the deferred one MUST still be scheduled
    assert request_behind.request_id in output.num_scheduled_tokens
    assert output.num_scheduled_tokens[request_behind.request_id] == 20

    # --- Step 2: prefetch done, cache exists → request scheduled ---
    # has_cache_item is called inside _try_schedule_encoder_inputs (not during
    # deferral), so it is only relevant here in step 2.
    scheduler.ec_connector.ensure_cache_available = Mock(return_value=True)
    scheduler.ec_connector.has_cache_item = Mock(return_value=True)

    output = scheduler.schedule()

    # Now the deferred request should be scheduled
    assert request_deferred.request_id in output.num_scheduled_tokens
    assert output.num_scheduled_tokens[request_deferred.request_id] == NUM_TOKENS
    _assert_right_encoder_cache_allocated(scheduler, requests=[request_deferred])
    # EC connector metadata should carry the deferred request's MM data
    _assert_right_ec_connector_metadata(
        output, mm_features_list=request_deferred.mm_features
    )
    # No local encoder compute — all loaded externally
    _assert_right_encoder_inputs(output, expected_total_reqs=0)


def test_ec_connector_defers_running_request_for_async_reload():
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=32,
        use_ec_connector=True,
        ec_role="ec_consumer",
    )
    request = create_requests(
        num_requests=1,
        num_tokens=128,
        mm_positions=[[PlaceholderRange(offset=48, length=32)]],
        req_ids=["request"],
    )[0]
    scheduler.ec_connector.ensure_cache_available = Mock(side_effect=[True, False])

    scheduler.add_request(request)
    first_output = scheduler.schedule()
    assert first_output.num_scheduled_tokens[request.request_id] == 32

    second_output = scheduler.schedule()
    assert request.request_id not in second_output.num_scheduled_tokens
    ensure_call = scheduler.ec_connector.ensure_cache_available.call_args
    assert ensure_call.args[:2] == (request, 32)


def test_ec_connector_pending_prefetch_only_checks_future_mm_features():
    """Test that future mm feature filtering only yields features beyond
    the computed token frontier.

    Features already within num_computed_tokens (past/boundary) must be
    filtered out; only features that extend beyond the frontier (future) should
    be yielded so that connector implementations know which items to prefetch.

    Filter cases:
      "past":     end = 0  + 16 = 16 <  32 → filtered OUT
      "boundary": end = 16 + 16 = 32 == 32 → filtered OUT (condition is >, not >=)
      "future":   end = 48 + 32 = 80 >  32 → yielded
    """
    BLOCK_SIZE = 16
    NUM_COMPUTED_TOKENS = BLOCK_SIZE * 2  # 32
    NUM_TOKENS = BLOCK_SIZE * 8  # 128

    HASH_PAST = "hash_past"
    HASH_BOUNDARY = "hash_boundary"
    HASH_FUTURE = "hash_future"

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_TOKENS,
        mm_hashes_list=[[HASH_PAST, HASH_BOUNDARY, HASH_FUTURE]],
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=BLOCK_SIZE),  # end=16 (past)
                PlaceholderRange(offset=16, length=BLOCK_SIZE),  # end=32 (boundary)
                PlaceholderRange(offset=48, length=BLOCK_SIZE * 2),  # end=80 (future)
            ]
        ],
        block_size=BLOCK_SIZE,
    )[0]

    future_hashes = [
        f.identifier
        for f in request.mm_features
        if f.mm_position.offset + f.mm_position.length > NUM_COMPUTED_TOKENS
    ]

    assert future_hashes == [HASH_FUTURE], (
        f"Expected only {HASH_FUTURE!r} from future mm feature filtering, "
        f"got {future_hashes!r}. Past/boundary features must be filtered out."
    )


def test_async_load_reservation_prevents_wedge_e2e():
    """Same wedge scenario as PR #40968's lateral-preemption e2e test, but
    resolved by reservation-based admission control instead of preemption.

    A (8 blocks) and B (5 blocks) both want an async KV load, sharing a 4-block
    prefix, in a 10-block pool (9 usable). Admitting both loads would wedge:
    once their recvs finish neither can complete its local prefill (8+5 > 9).

    Here the reservation gate refuses to admit B's load while A's full sequence
    is still reserved, so B never holds blocks and A is free to complete - no
    deadlock, and (unlike lateral preemption) B is never preempted.
    """
    BLOCK_SIZE = 16
    A_TOKENS = BLOCK_SIZE * 8  # bigger request
    B_TOKENS = BLOCK_SIZE * 5  # smaller request
    MATCHED_TOKENS = BLOCK_SIZE * 4  # 4-block prefix loaded for both
    NUM_BLOCKS = 10  # 9 usable; both prefixes fit, but not both full sequences

    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        max_num_seqs=4,
        max_num_batched_tokens=A_TOKENS * 2,
        use_kv_connector=mock_kv(matched_tokens=MATCHED_TOKENS, is_async=True),
    )

    [a] = create_requests(
        num_requests=1, num_tokens=A_TOKENS, block_size=BLOCK_SIZE, req_ids=["a"]
    )
    [b] = create_requests(
        num_requests=1, num_tokens=B_TOKENS, block_size=BLOCK_SIZE, req_ids=["b"]
    )
    scheduler.add_request(a)
    scheduler.add_request(b)

    EMPTY_OUTPUT = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    req_to_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks

    # Step 1: A's load is admitted; B's is held back by the reservation (B never
    # holds blocks, so the wedge precondition - both holding prefixes - is gone).
    out1 = scheduler.schedule()
    assert a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert a.num_computed_tokens == MATCHED_TOKENS
    assert b.status == RequestStatus.WAITING
    assert b.request_id not in req_to_blocks
    assert len(scheduler.running) == 0
    scheduler.update_from_output(out1, EMPTY_OUTPUT)

    # Step 2: nothing changes until A's recv lands.
    out2 = scheduler.schedule()
    assert len(scheduler.running) == 0
    a_finished = dataclasses.replace(
        EMPTY_OUTPUT,
        kv_connector_output=KVConnectorOutput(finished_recving=[a.request_id]),
    )
    scheduler.update_from_output(out2, a_finished)

    # Step 3: A makes forward progress straight to RUNNING - no preemption was
    # needed because B never wedged it.
    out3 = scheduler.schedule()
    assert a.status == RequestStatus.RUNNING
    assert a in scheduler.running
    assert a.request_id in {req.req_id for req in out3.scheduled_new_reqs}
    assert b.status == RequestStatus.WAITING
    assert b.num_preemptions == 0
    assert b.request_id not in req_to_blocks


def _create_hybrid_mamba_connector_scheduler(
    matched_tokens: int,
    block_size: int = 16,
    num_blocks: int = 100,
    supports_divergent_hits: bool = True,
) -> Scheduler:
    """FA + Mamba ("all" cache mode) scheduler with a MockKVConnector."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="all",
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MockKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "matched_tokens": matched_tokens,
                "is_async": False,
                "supports_divergent_local_hybrid_hits": supports_divergent_hits,
            },
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="all",
                ),
            ),
        ],
    )
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
        log_stats=True,
    )


@pytest.mark.parametrize(
    "matched_tokens,expected_num_computed",
    [
        # No external hit: resume on the deepest locally-consistent boundary
        # (block 0's state survives for both groups).
        (0, 16),
        # One external block on top of the reconciled local boundary.
        (16, 32),
    ],
)
def test_hybrid_per_group_hit_divergence_with_connector(
    matched_tokens: int, expected_num_computed: int
):
    """Per-group prefix hits can diverge for hybrid models with a connector
    (#46453): under block pressure the FA prefix tail is evicted while a
    deeper Mamba state block survives. The scheduler must not report the
    deeper hit as locally computed (evicted FA blocks are not resident ->
    engine crash / dirty KV); it falls back to the reconciled boundary that
    every group is consistent at.
    """
    block_size = 16
    scheduler = _create_hybrid_mamba_connector_scheduler(matched_tokens)
    manager = scheduler.kv_cache_manager
    assert isinstance(manager.coordinator, HybridKVCacheCoordinator)

    # Seed a 4-block prefix so both groups cache all four boundaries
    # (mamba cache mode "all" caches every block's state densely).
    [fill] = create_requests(
        num_requests=1,
        num_tokens=4 * block_size,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["fill"],
    )
    computed_blocks, num_computed, _ = manager.get_computed_blocks(fill)
    blocks = manager.allocate_slots(
        fill, fill.num_tokens, num_computed, computed_blocks
    )
    fa_ids = [b.block_id for b in blocks.blocks[0]]
    mamba_ids = [b.block_id for b in blocks.blocks[1]]
    manager.free(fill)

    # Evict the FA tail and the middle mamba states; block 0 (both groups)
    # and the deep mamba state at block 3 survive.
    manager.block_pool.evict_blocks({fa_ids[2], fa_ids[3], mamba_ids[1], mamba_ids[2]})

    # A replay of the prefix plus one extra block now sees diverged
    # per-group hits: FA stops at the evicted tail, while the mamba lookup
    # finds the deeper surviving state.
    [replay] = create_requests(
        num_requests=1,
        num_tokens=5 * block_size,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["replay"],
    )
    _, per_group_hits = manager.coordinator.find_longest_cache_hit_per_group(
        replay.block_hashes, replay.num_tokens - 1
    )
    assert per_group_hits == (2 * block_size, 4 * block_size)  # diverged

    scheduler.add_request(replay)
    output = scheduler.schedule()
    num_scheduled = output.num_scheduled_tokens[replay.request_id]
    assert replay.num_tokens - num_scheduled == expected_num_computed


@pytest.mark.parametrize(
    (
        "supports_divergent_local_hybrid_hits",
        "matched_tokens",
        "replay_blocks",
        "expected_num_computed",
    ),
    [
        (True, 0, 5, 16),
        (True, 16, 6, 80),
        (False, 16, 6, 32),
    ],
)
def test_hybrid_fa_deeper_hit_respects_connector_lookup_policy(
    supports_divergent_local_hybrid_hits: bool,
    matched_tokens: int,
    replay_blocks: int,
    expected_num_computed: int,
):
    """A capable connector may restore missing Mamba state at the deeper FA
    boundary. An external miss or an incapable connector uses a locally
    consistent boundary instead.
    """
    block_size = 16
    scheduler = _create_hybrid_mamba_connector_scheduler(
        matched_tokens,
        supports_divergent_hits=supports_divergent_local_hybrid_hits,
    )
    manager = scheduler.kv_cache_manager
    assert isinstance(manager.coordinator, HybridKVCacheCoordinator)

    # Seed a 4-block prefix in both groups.
    [fill] = create_requests(
        num_requests=1,
        num_tokens=4 * block_size,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["fill"],
    )
    computed_blocks, num_computed, _ = manager.get_computed_blocks(fill)
    blocks = manager.allocate_slots(
        fill, fill.num_tokens, num_computed, computed_blocks
    )
    mamba_ids = [b.block_id for b in blocks.blocks[1]]
    manager.free(fill)

    # Keep all FA blocks; evict every mamba state but block 0. FA reaches 4
    # blocks, the mamba hit only reaches 1 -> diverged (FA > Mamba).
    manager.block_pool.evict_blocks({mamba_ids[1], mamba_ids[2], mamba_ids[3]})

    [replay] = create_requests(
        num_requests=1,
        num_tokens=replay_blocks * block_size,
        max_tokens=1,
        same_prompt=True,
        block_size=block_size,
        req_ids=["replay"],
    )
    _, per_group_hits = manager.coordinator.find_longest_cache_hit_per_group(
        replay.block_hashes, replay.num_tokens - 1
    )
    assert per_group_hits == (4 * block_size, 1 * block_size)  # FA deeper

    scheduler.add_request(replay)
    output = scheduler.schedule()
    num_scheduled = output.num_scheduled_tokens[replay.request_id]
    assert replay.num_tokens - num_scheduled == expected_num_computed


def _make_encoder_instance_request(scheduler, text_prefix=8, image_tokens=16):
    """One request whose image sits after some text, as an EPD proxy sends it.

    `create_scheduler` builds a text-only model, so `compute_mm_encoder_budget`
    leaves the encoder cache at zero capacity; give it room so the admitted path
    is reachable.
    """
    ecm = scheduler.encoder_cache_manager
    ecm.cache_size = ecm.num_free_slots = ecm.num_freeable_slots = 8 * image_tokens
    scheduler.max_num_encoder_input_tokens = 8 * image_tokens

    (request,) = create_requests(
        num_requests=1,
        num_tokens=text_prefix + image_tokens + 4,
        # Completion must hinge on the prompt being encoded, not on max_tokens.
        max_tokens=16,
        mm_hashes_list=[["img-hash-0"]],
        mm_positions=[[PlaceholderRange(offset=text_prefix, length=image_tokens)]],
    )
    scheduler.add_request(request)
    return request


def test_encoder_instance_defers_stop_until_prompt_is_consumed(
    monkeypatch: pytest.MonkeyPatch,
):
    """An encoder instance must not finish a request that has not encoded yet.

    When the encoder cache cannot admit the multi-modal item, the scheduler
    schedules only the tokens before it. Finishing that step would hand the
    client a successful completion for a request whose image was never
    encoded, so no embedding is ever published to the EC connector.
    """
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        use_ec_connector=True,
        ec_role="ec_producer",
    )
    request = _make_encoder_instance_request(scheduler)
    req_id = request.request_id

    # Encoder budget exhausted: the item cannot be admitted this step.
    original_can_allocate = EncoderCacheManager.can_allocate
    monkeypatch.setattr(EncoderCacheManager, "can_allocate", lambda *a, **k: False)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens[req_id] > 0
    assert not output.scheduled_encoder_inputs.get(req_id)
    assert output.num_scheduled_tokens[req_id] < request.num_prompt_tokens

    scheduler.update_from_output(
        output,
        make_empty_encoder_model_runner_output(output),
    )

    assert not request.is_finished(), (
        "request finished before its image was encoded, so no embedding was "
        "published to the EC connector"
    )
    assert request.num_output_tokens == 0

    # With the budget restored the item is encoded and only then does the
    # request finish.
    monkeypatch.setattr(EncoderCacheManager, "can_allocate", original_can_allocate)

    output = scheduler.schedule()
    assert output.scheduled_encoder_inputs.get(req_id) == [0]

    scheduler.update_from_output(
        output,
        make_empty_encoder_model_runner_output(output),
    )
    assert request.is_finished()


def test_encoder_instance_finishes_request_once_prompt_is_consumed():
    """The unblocked path completes in a single step, without sampling."""
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        use_ec_connector=True,
        ec_role="ec_producer",
    )
    request = _make_encoder_instance_request(scheduler)
    req_id = request.request_id

    output = scheduler.schedule()
    assert output.scheduled_encoder_inputs.get(req_id) == [0]
    assert output.num_scheduled_tokens[req_id] == request.num_prompt_tokens

    scheduler.update_from_output(
        output,
        make_empty_encoder_model_runner_output(output),
    )
    assert request.status == RequestStatus.FINISHED_STOPPED
    # The encoder instance publishes an embedding, not tokens.
    assert request.num_output_tokens == 0


@pytest.mark.parametrize("ec_role", ["ec_producer", "ec_consumer"])
def test_encoder_input_skipped_when_connector_already_has_the_item(ec_role: str):
    """Neither role re-encodes what the connector already holds.

    For a consumer the item is loaded; for a producer there is nothing left to
    do at all -- it published that embedding earlier (or a sibling encoder did),
    so a second ViT pass would be pure waste. Reached on any repeat: a second
    chat turn re-sending its image, a sibling encoder behind the proxy's
    round-robin, or a restart that kept the shared storage.

    Pinned because the obvious "fix" for the encoder-instance crash this used to
    cause is to make the producer encode anyway; the crash belongs to the worker
    (an encoder instance must not gather embeddings it never needed), and paying
    for it here would cost every deployment a redundant encode.
    """
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        use_ec_connector=True,
        ec_role=ec_role,
    )
    request = _make_encoder_instance_request(scheduler)
    req_id = request.request_id
    scheduler.ec_connector.has_cache_item = lambda *a, **k: True

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[req_id] > 0
    assert not output.scheduled_encoder_inputs.get(req_id)


def test_elastic_graph_step_key_tracks_runtime_shape_not_static_buckets():
    key_x1 = Scheduler._make_elastic_graph_step_key({"a": 4}, 3, True)
    key_x1_mixed = Scheduler._make_elastic_graph_step_key({"a": 37}, 3, False)
    key_x8 = Scheduler._make_elastic_graph_step_key(
        {str(i): 4 for i in range(8)}, 3, True
    )
    key_x8_long = Scheduler._make_elastic_graph_step_key(
        {str(i): 512 for i in range(8)}, 3, False
    )

    assert len({key_x1, key_x1_mixed, key_x8, key_x8_long}) == 4
    assert key_x1 == SemanticGraphStep(
        3, 1, 4, 4, "decode", ("target", "mtp_prefill", "mtp_decode")
    )
    assert key_x1_mixed == SemanticGraphStep(
        3, 1, 37, None, "mixed", ("target", "mtp_prefill", "mtp_decode")
    )
    assert key_x8 == SemanticGraphStep(
        3, 8, 32, 4, "decode", ("target", "mtp_prefill", "mtp_decode")
    )
    assert key_x8_long == SemanticGraphStep(
        3, 8, 4096, None, "mixed", ("target", "mtp_prefill", "mtp_decode")
    )
    assert Scheduler._make_elastic_graph_step_key({}, 3, False) is None


def test_elastic_graph_step_key_coalesces_piecewise_chunk_distributions():
    live_x12_m4036_a = {str(index): 336 for index in range(11)}
    live_x12_m4036_a["11"] = 340
    live_x12_m4036_b = {str(index): 336 for index in range(10)}
    live_x12_m4036_b.update({"10": 338, "11": 338})

    key_a = Scheduler._make_elastic_graph_step_key(live_x12_m4036_a, 3, False)
    key_b = Scheduler._make_elastic_graph_step_key(live_x12_m4036_b, 3, False)

    assert (
        key_a
        == key_b
        == SemanticGraphStep(
            3, 12, 4036, None, "mixed", ("target", "mtp_prefill", "mtp_decode")
        )
    )
    # A nominal decode flag cannot produce FULL when per-request query lengths
    # differ; the target/MTP-prefill owners use the PIECEWISE identity.
    assert Scheduler._make_elastic_graph_step_key({"a": 4, "b": 3}, 3, True) == (
        SemanticGraphStep(
            3,
            2,
            7,
            None,
            "mixed",
            ("target", "mtp_prefill", "mtp_decode"),
        )
    )
    assert Scheduler._elastic_graph_owner_key((0, 3, 12, 4036, 0)) == (
        0,
        3,
        0,
        4036,
        0,
    )
    assert Scheduler._elastic_graph_owner_key((0, 0, 12, 4036, 0)) == (
        0,
        0,
        0,
        4036,
        0,
    )


def test_elastic_piecewise_step_key_uses_runtime_derived_boundary():
    scheduler = _new_elastic_scheduler()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=64)
    scheduler.max_num_running_reqs = 64
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 40}

    assert scheduler._canonical_elastic_graph_step_key({"a": 37}, 3, False) == (
        0,
        3,
        1,
        64,
        0,
    )
    assert scheduler._canonical_elastic_graph_step_key({"a": 4000}, 3, False) == (
        0,
        3,
        1,
        4096,
        0,
    )
    assert scheduler._canonical_elastic_graph_step_key({"a": 4}, 3, True) == (
        0,
        3,
        1,
        4,
        4,
    )
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(2)}, 3, True
    ) == (0, 3, 2, 8, 4)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(3)}, 3, True
    ) == (0, 3, 4, 16, 4)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(5)}, 3, True
    ) == (0, 3, 8, 32, 4)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(16)}, 3, True
    ) == (0, 3, 16, 64, 4)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(32)}, 3, True
    ) == (0, 3, 32, 128, 4)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(40)}, 3, True
    ) == (0, 3, 40, 160, 4)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(33)}, 3, True
    ) == (0, 3, 40, 160, 4)
    with pytest.raises(ElasticGraphError, match="exceeds the declared"):
        scheduler._canonical_elastic_graph_step_key(
            {str(index): 4 for index in range(41)}, 3, True
        )

    generation = RuntimeGeneration("k3-derived-inventory")
    physical = resolve_step_physical_keys((0, 3, 40, 160, 4), generation, 4096)
    assert [(key.logical.owner, key.logical.token_bucket) for key in physical] == [
        ("target", 160),
        ("mtp_prefill", 160),
        ("mtp_decode", 40),
    ]
    assert physical[0].logical.uniform_query_len == 4
    assert scheduler._canonical_elastic_graph_step_key({"a": 2, "b": 2}, 3, True) == (
        0,
        3,
        2,
        4,
        0,
    )
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 1 for index in range(12)}, 0, False
    ) == (0, 0, 12, 16, 0)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 1 for index in range(7)}, 0, False
    ) == (0, 0, 7, 8, 0)


def test_elastic_unknown_shape_requires_hot_terminal_carrier_after_ready():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_terminal_decode_carrier_x = 39
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"serving_carrier_owner": "mtp_decode"}
    scheduler._elastic_admission_controller.step_key = None
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler.running = [object()]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 1024
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    with pytest.raises(RuntimeError, match="serving MTP decode cold capture"):
        scheduler._can_fund_elastic_graph_step(
            (1, 3, 1, 4, 4), minimum_free_primary_blocks=1
        )
    assert coordinator.elastic_external_memory_bytes == 0


def test_elastic_unknown_shape_can_defer_with_hot_terminal_carrier_after_ready():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_terminal_decode_carrier_x = 39
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"serving_carrier_owner": "mtp_decode"}
    scheduler._elastic_admission_controller.step_key = None
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler.running = [object()]
    carrier = scheduler.resolve_elastic_serving_carrier_physical_keys(
        ((1, 3, 39, 39, 1), (0, 3, 39, 156, 4))
    )
    scheduler._elastic_admission_controller.publish_hot(
        carrier[0], GraphPrice(94, 94, "terminal-mtp"), pinned=True
    )
    scheduler._elastic_serving_carrier_keys = carrier
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 1024
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    fits, required, available = scheduler._can_fund_elastic_graph_step(
        (1, 3, 1, 4, 4), minimum_free_primary_blocks=1
    )

    assert fits is False
    assert required == 0
    assert available == 1024
    assert coordinator.elastic_external_memory_bytes == 0


def test_elastic_external_loan_normalizes_to_physical_quantum():
    coordinator = object.__new__(HybridKVCacheCoordinator)
    coordinator.kv_cache_config = Mock(elastic_mapping_quantum=16)

    assert coordinator.normalize_elastic_external_memory(0) == 0
    assert coordinator.normalize_elastic_external_memory(32) == 32
    assert coordinator.normalize_elastic_external_memory(33) == 48
    with pytest.raises(ValueError, match="cannot be negative"):
        coordinator.normalize_elastic_external_memory(-1)


def test_elastic_catalog_cold_loan_covers_published_residency():
    # Real sealed X1/q4 evidence: calibration's KV-priced cold component was
    # 120 MiB, while the published owner set stabilized at 152 MiB.  A fresh
    # X0 -> X1 rebuild must fund the latter rather than fail after capture.
    row = {
        "cold_peak_bytes": 125829120,
        "hot_peak_bytes": 159383552,
        "resident_bytes": 159383552,
        "floor_bytes": 0,
    }

    assert _elastic_catalog_cold_residency_envelope(row) == 159383552

    legacy_floor_row = {
        "cold_peak_bytes": 100,
        "hot_peak_bytes": 90,
        "floor_bytes": 120,
    }
    assert _elastic_catalog_cold_residency_envelope(legacy_floor_row) == 120


def test_elastic_piecewise_worst_form_envelope_covers_unseen_m():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_admission_controller.capture_envelopes = {}
    worst = (0, 3, 18, 4096, 0)
    holdout = (0, 3, 8, 64, 0)

    scheduler._record_elastic_capture_envelope(worst, (768, 32, 16))

    assert scheduler._elastic_capture_envelope(holdout) == (768, 32, 16)
    scheduler._record_elastic_capture_envelope(holdout, (512, 24, 8))
    assert scheduler._elastic_capture_envelope(holdout) == (512, 24, 8)
    assert scheduler._elastic_admission_controller.capture_envelopes[
        (0, 3, 0, 0, 0)
    ] == (
        768,
        32,
        16,
    )
    assert scheduler._elastic_capture_envelope((1, 3, 8, 32, 4)) is None


def test_bounded_k3_decode_canonical_owner_matrix_covers_every_tail_x():
    scheduler = _new_elastic_scheduler()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=40)
    scheduler.max_num_running_reqs = 40
    compiled = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy(compiled)
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 40}
    generation = RuntimeGeneration("exact-k3-tail-matrix")

    for x in range(1, 41):
        step_key = scheduler._canonical_elastic_graph_step_key(
            {str(index): 4 for index in range(x)}, 3, True
        )
        expected_x = next(bucket for bucket in (1, 2, 4, 8, 16, 32, 40) if bucket >= x)
        assert step_key == (0, 3, expected_x, 4 * expected_x, 4)
        physical = resolve_step_physical_keys(
            step_key,
            generation,
            4096,
            compiled,
            scheduler._elastic_graph_execution_policy,
        )
        assert [key.logical.owner for key in physical] == [
            "target",
            "mtp_prefill",
            "mtp_decode",
        ]
        assert [key.logical.token_bucket for key in physical] == [
            4 * expected_x,
            4 * expected_x,
            expected_x,
        ]
        assert all(key.physical_num_reqs == expected_x for key in physical)

    mixed = scheduler._canonical_elastic_graph_step_key({"tail": 12}, 3, False)
    assert mixed == (0, 3, 1, 16, 0)
    assert [
        key.logical.owner
        for key in resolve_step_physical_keys(
            mixed,
            generation,
            4096,
            compiled,
            scheduler._elastic_graph_execution_policy,
        )
    ] == ["mtp_decode"]

    # Per-request speculative acceptance may differ even though the lifecycle
    # remains pure decode. The physical execution lane is then PIECEWISE mixed:
    # it preserves exact semantic X and may not inherit the uniform X16
    # successor carrier. Target/MTP-prefill use the M32 compiled boundary while
    # MTP-decode uses its already-declared exact X12 FULL descriptor.
    variable_k = scheduler._canonical_elastic_graph_step_key(
        {str(index): token_count for index, token_count in enumerate((4, 3, 2, 1) * 3)},
        3,
        True,
    )
    assert variable_k == (0, 3, 12, 32, 0)
    variable_physical = resolve_step_physical_keys(
        variable_k,
        generation,
        4096,
        compiled,
        scheduler._elastic_graph_execution_policy,
    )
    assert [key.logical.owner for key in variable_physical] == ["mtp_decode"]
    assert variable_physical[0].physical_num_reqs == 12
    exact_x12 = resolve_step_physical_keys(
        (0, 3, 12, 48, 4),
        generation,
        4096,
        compiled,
        scheduler._elastic_graph_execution_policy,
    )
    assert variable_physical[0] == exact_x12[-1]

    # A qlen1 K3 tail uses the same bounded FULL carrier.  FULL replay already
    # supports a semantic request count below its physical request count.
    qlen1_tail = scheduler._canonical_elastic_graph_step_key(
        {str(index): 1 for index in range(11)}, 3, True
    )
    assert qlen1_tail == (1, 3, 16, 16, 1)


def test_live_k3_cohort_retains_carrier_without_oversizing_tail_execution():
    scheduler = _new_elastic_scheduler()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=40)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 40}
    scheduler._elastic_admission_controller.step_key = (0, 3, 7, 16, 0)
    scheduler._elastic_graph_carrier_step_key = (0, 3, 16, 64, 4)
    _publish_elastic_step_hot(scheduler, (0, 3, 16, 64, 4))
    scheduler.running = [object() for _ in range(7)]
    scheduler.waiting = deque()
    scheduler.skipped_waiting = deque()

    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(7)}, 3, True
    ) == (0, 3, 8, 32, 4)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): token_count for index, token_count in enumerate((4, 3, 2, 1))},
        3,
        True,
    ) == (0, 3, 4, 16, 0)
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 1 for index in range(3)}, 3, True
    ) == (1, 3, 4, 4, 1)

    execution_key = (0, 3, 8, 32, 4)
    current, successor, protected = scheduler._elastic_step_residency_intent(
        execution_key
    )
    retained = scheduler._resolve_elastic_step_physical_keys((0, 3, 16, 64, 4))
    assert all(key.physical_num_reqs == 8 for key in current)
    assert set(retained).issubset(successor)
    assert set(retained).issubset(protected)
    scheduler._commit_elastic_graph_carrier_step_key(execution_key)
    scheduler._commit_elastic_graph_carrier_step_key((1, 3, 4, 4, 1))
    assert scheduler._elastic_graph_carrier_step_key == (0, 3, 16, 64, 4)

    # A waiting-only handoff retains the old HOT owner set without dispatching
    # the new semantic request through the oversized carrier.
    scheduler.running = []
    scheduler.waiting = deque((object(),))
    assert scheduler._canonical_elastic_graph_step_key({"handoff": 4}, 3, True) == (
        0,
        3,
        1,
        4,
        4,
    )

    # A new wave after a real request-free drain pays only for its own smallest
    # declared carrier.
    scheduler.waiting.clear()
    assert scheduler._canonical_elastic_graph_step_key({"new": 4}, 3, True) == (
        0,
        3,
        1,
        4,
        4,
    )


@pytest.mark.parametrize(
    ("semantic_x", "physical_x", "retained_x"),
    [(4, 4, 32), (3, 4, 16), (1, 1, 8), (11, 16, 4), (25, 32, 8)],
)
def test_form_plan_dispatches_current_bucket_while_leasing_retained_carrier(
    semantic_x, physical_x, retained_x
):
    generation = RuntimeGeneration("tail-execution-residency-split")
    scheduler = _new_elastic_scheduler(generation)
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=40)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 40}
    current_key = (0, 3, physical_x, physical_x * 4, 4)
    carrier_key = (0, 3, retained_x, retained_x * 4, 4)
    scheduler._elastic_graph_carrier_step_key = carrier_key
    _publish_elastic_step_hot(scheduler, current_key)
    _publish_elastic_step_hot(scheduler, carrier_key)

    current, successor, physical = scheduler._elastic_step_residency_intent(current_key)
    manifest, dispatch = build_execution_manifest(
        step_key=current_key,
        request_ids=tuple(f"request-{index}" for index in range(semantic_x)),
        per_request_query_lens=(4,) * semantic_x,
        per_request_is_prefilling=(False,) * semantic_x,
        scheduled_draft_rows=(3,) * semantic_x,
        requested_output_k=3,
        executed_drafter_k=3,
        phase="decode",
        generation=generation,
        policy=scheduler._elastic_graph_execution_policy,
        max_num_batched_tokens=4096,
        physical_keys=current,
    )
    plan = scheduler._elastic_admission_controller.plan(
        "tail-user-plan",
        physical,
        request_bytes=0,
        available_bytes=4096,
    )
    plan = dataclasses.replace(
        plan,
        execution_manifest=manifest,
        current_dispatch=dispatch,
        successor_keys=successor,
        protected_keys=tuple(key for key in physical if key not in set(current)),
    )
    retained = set(scheduler._resolve_elastic_step_physical_keys(carrier_key))

    assert plan.kind == ElasticPlanKind.USER
    assert all(
        item.invocation.semantic_num_reqs == semantic_x
        and item.invocation.physical_num_reqs == physical_x
        for item in plan.current_dispatch
    )
    assert {item.physical_key for item in plan.current_dispatch} == set(current)
    assert retained.issubset(plan.physical_keys)
    assert retained.issubset(plan.protected_keys)


@pytest.mark.parametrize(
    ("execution_key", "expected_closure"),
    [
        ((0, 3, 32, 4096, 0), (0, 3, 32, 128, 4)),
        ((0, 3, 40, 4096, 0), (0, 3, 40, 160, 4)),
        ((0, 1, 7, 64, 0), (0, 1, 8, 16, 2)),
    ],
)
def test_elastic_carrier_closure_derives_m_from_x_times_k_plus_one(
    execution_key: tuple[int, ...],
    expected_closure: tuple[int, ...],
):
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_seqs=40)
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 40}

    assert (
        scheduler._elastic_graph_carrier_closure_step_key(execution_key)
        == expected_closure
    )


def test_mixed_carrier_closure_and_qlen1_tail_share_bounded_x_inventory():
    scheduler = _new_elastic_scheduler()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=39)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 39}
    inventory = (1, 2, 4, 8, 16, 32, 39)
    scheduler._elastic_short_decode_inventory = dict.fromkeys(inventory, ())
    scheduler.waiting = deque()
    scheduler.skipped_waiting = deque()

    for semantic_x in range(1, 40):
        physical_x = next(x for x in inventory if x >= semantic_x)
        scheduler._elastic_graph_carrier_step_key = None
        scheduler.running = [object() for _ in range(semantic_x)]
        evidence_keys = scheduler._resolve_elastic_step_physical_keys(
            (1, 3, semantic_x, semantic_x, 1)
        )
        mixed_key = scheduler._canonical_elastic_graph_step_key(
            {str(index): 1 for index in range(semantic_x)}, 3, False
        )
        mixed_keys = scheduler._resolve_elastic_step_physical_keys(mixed_key)
        assert tuple(
            key for key in evidence_keys if key.logical.owner == "mtp_decode"
        ) == tuple(key for key in mixed_keys if key.logical.owner == "mtp_decode")
        scheduler._commit_elastic_graph_carrier_step_key((0, 3, semantic_x, 4096, 0))
        assert scheduler._elastic_graph_carrier_step_key == (
            0,
            3,
            physical_x,
            physical_x * 4,
            4,
        )

        # Cancellation and waiting-only handoff execute through the smallest
        # bounded bucket. Residency retention is tested independently above.
        tail_x = max(1, semantic_x - 1)
        tail_physical_x = next(x for x in inventory if x >= tail_x)
        scheduler.running = [object() for _ in range(tail_x)]
        assert scheduler._canonical_elastic_graph_step_key(
            {str(index): 1 for index in range(tail_x)}, 3, True
        ) == (1, 3, tail_physical_x, tail_physical_x, 1)
        scheduler.running = []
        scheduler.waiting = deque((object(),))
        assert scheduler._canonical_elastic_graph_step_key({"handoff": 1}, 3, True) == (
            1,
            3,
            1,
            1,
            1,
        )
        scheduler.waiting.clear()

    # Recovery from a pre-normalization exact carrier is also compiled into
    # the bounded inventory before it can select a FULL tail.
    scheduler._elastic_graph_carrier_step_key = (0, 3, 7, 28, 4)
    scheduler.running = [object() for _ in range(3)]
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 1 for index in range(3)}, 3, True
    ) == (1, 3, 4, 4, 1)


def test_mixed_request_owner_executes_through_bounded_hot_x_inventory():
    generation = RuntimeGeneration("mixed-bounded-request-owner")
    scheduler = _new_elastic_scheduler(generation)
    compiled = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=39)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy(compiled)
    scheduler._elastic_compiled_piecewise_sizes = frozenset(compiled)
    scheduler._elastic_graph_catalog_coverage = {
        "decode_max_x": 39,
        "serving_carrier_owner": "mtp_decode",
    }
    scheduler._elastic_short_decode_inventory = dict.fromkeys(
        (1, 2, 4, 8, 16, 32, 39), ()
    )

    for physical_x in scheduler._elastic_short_decode_inventory:
        _publish_elastic_step_hot(scheduler, (0, 3, physical_x, physical_x * 4, 4))

    for semantic_x in range(1, 40):
        physical_x = next(
            value
            for value in scheduler._elastic_short_decode_inventory
            if value >= semantic_x
        )
        current, _successor, _protected = scheduler._elastic_step_residency_intent(
            (0, 3, semantic_x, 4096, 0)
        )
        mtp_decode = next(key for key in current if key.logical.owner == "mtp_decode")
        assert mtp_decode.physical_num_reqs == physical_x

        query_lens = (2,) + (1,) * (semantic_x - 1)
        step_tokens = 1 << (sum(query_lens) - 1).bit_length()
        step_key = (0, 3, semantic_x, step_tokens, 0)
        manifest_keys = scheduler._elastic_step_residency_intent(step_key)[0]
        manifest, dispatch = build_execution_manifest(
            step_key=step_key,
            request_ids=tuple(f"request-{index}" for index in range(semantic_x)),
            per_request_query_lens=query_lens,
            per_request_is_prefilling=(True,) + (False,) * (semantic_x - 1),
            scheduled_draft_rows=(0,) * semantic_x,
            requested_output_k=3,
            executed_drafter_k=3,
            phase="mixed",
            generation=generation,
            policy=scheduler._elastic_graph_execution_policy,
            max_num_batched_tokens=4096,
            physical_keys=manifest_keys,
        )
        invocation = next(
            item for item in manifest.invocations if item.owner == "mtp_decode"
        )
        assert invocation.semantic_num_reqs == semantic_x
        assert invocation.physical_num_reqs == physical_x
        assert (
            next(
                item for item in dispatch if item.invocation.owner == "mtp_decode"
            ).representation
            == DispatchRepresentation.HOT_GRAPH
        )


def test_sealed_runtime_executes_smallest_bucket_and_retains_terminal_carrier() -> None:
    scheduler = _new_elastic_scheduler(RuntimeGeneration("terminal-carrier"))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=39)
    compiled = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy(compiled)
    scheduler._elastic_compiled_piecewise_sizes = frozenset(compiled)
    scheduler._elastic_terminal_decode_carrier_x = 39
    scheduler._elastic_restore_mode = False

    mixed = scheduler._resolve_elastic_step_physical_keys((0, 3, 12, 1024, 0))
    assert [key.logical.owner for key in mixed] == ["mtp_decode"]
    assert mixed[0].physical_num_reqs == 12

    q1 = scheduler._canonical_elastic_graph_step_key(
        {str(index): 1 for index in range(12)}, 3, True
    )
    assert q1 == (1, 3, 16, 16, 1)
    assert all(
        key.physical_num_reqs == 16
        for key in scheduler._resolve_elastic_step_physical_keys(q1)
    )

    q4 = scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(12)}, 3, True
    )
    assert q4 == (0, 3, 16, 64, 4)
    assert all(
        key.physical_num_reqs == 16
        for key in scheduler._resolve_elastic_step_physical_keys(q4)
    )

    scheduler._elastic_restore_mode = True
    exact = scheduler._resolve_elastic_step_physical_keys((0, 3, 12, 1024, 0))
    assert exact[-1].logical.owner == "mtp_decode"
    assert exact[-1].physical_num_reqs == 12


def test_restore_mode_preserves_declared_mixed_execution_key():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=39)
    scheduler._elastic_short_decode_inventory = {1: (), 2: (), 4: (), 8: (), 39: ()}

    assert scheduler._elastic_graph_carrier_closure_step_key((0, 3, 7, 4096, 0)) == (
        0,
        3,
        7,
        4096,
        0,
    )
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 1 for index in range(3)}, 3, True
    ) == (1, 3, 3, 3, 1)


@pytest.mark.parametrize(
    "stale_carrier",
    [(0, 3, 39, 156, 4), (0, 3, 47, 188, 4)],
)
def test_restore_epoch_clears_logical_carrier_but_retains_hot_residency(
    stale_carrier: tuple[int, ...],
):
    scheduler = _new_elastic_scheduler(RuntimeGeneration("restore-to-product"))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=39)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 39}
    scheduler._elastic_short_decode_inventory = {
        1: (),
        2: (),
        4: (),
        8: (),
        16: (),
        32: (),
        39: (),
    }
    scheduler.running = []
    scheduler.waiting = deque()
    scheduler.skipped_waiting = deque()
    scheduler._elastic_graph_carrier_step_key = stale_carrier
    hot_key = scheduler._resolve_elastic_step_physical_keys((1, 3, 39, 39, 1))[0]
    scheduler._elastic_admission_controller.publish_hot(
        hot_key,
        GraphPrice(
            resident_bytes=1,
            capture_peak_bytes=1,
            reclaim_group="restore-hot",
        ),
        pinned=True,
    )

    scheduler.finish_elastic_restore_epoch()

    assert scheduler._elastic_graph_carrier_step_key is None
    assert scheduler._elastic_admission_controller.entries[hot_key].hot
    assert scheduler._canonical_elastic_graph_step_key(
        {"first-product": 1}, 3, True
    ) == (
        1,
        3,
        1,
        1,
        1,
    )


def test_restore_epoch_rejects_live_synthetic_queue_before_clearing_carrier():
    scheduler = _new_elastic_scheduler()
    scheduler.running = [object()]
    scheduler.waiting = deque()
    scheduler.skipped_waiting = deque()
    scheduler._elastic_graph_carrier_step_key = (0, 3, 39, 156, 4)

    with pytest.raises(RuntimeError, match="live synthetic request cohort"):
        scheduler.finish_elastic_restore_epoch()

    assert scheduler._elastic_graph_carrier_step_key == (0, 3, 39, 156, 4)


@pytest.mark.parametrize(
    ("pending_state", "message"),
    [
        ("maintenance", "pending maintenance"),
        ("loan", "pending loans"),
        ("retention", "active retention"),
    ],
)
def test_restore_epoch_rejects_unsettled_physical_state_before_clearing_carrier(
    pending_state: str,
    message: str,
):
    scheduler = _new_elastic_scheduler()
    scheduler.running = []
    scheduler.waiting = deque()
    scheduler.skipped_waiting = deque()
    scheduler._elastic_graph_carrier_step_key = (0, 3, 39, 156, 4)
    if pending_state == "maintenance":
        scheduler._elastic_admission_controller._pending_maintenance_plan = Mock()
    elif pending_state == "loan":
        scheduler._elastic_admission_controller.reserve_loan((0, 3, 1, 4, 4), 1)
    else:
        scheduler._elastic_restore_retention_id = "active-retention"

    with pytest.raises(RuntimeError, match=message):
        scheduler.finish_elastic_restore_epoch()

    assert scheduler._elastic_graph_carrier_step_key == (0, 3, 39, 156, 4)


def test_elastic_carrier_closure_preserves_pure_decode_tail_phase():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("pure-tail"))
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_compiled_piecewise_sizes = frozenset({4096})
    pure_tail = (1, 3, 12, 12, 1)

    closure = scheduler._elastic_graph_carrier_closure_step_key(pure_tail)
    owners = scheduler._resolve_elastic_step_physical_keys(closure)

    assert closure == pure_tail
    assert [(key.logical.owner, key.logical.mode) for key in owners] == [
        ("target", "FULL"),
        ("mtp_prefill", "PIECEWISE"),
        ("mtp_decode", "FULL"),
    ]


def test_elastic_carrier_growth_retains_residency_but_shrinks_tail_execution():
    scheduler = _new_elastic_scheduler()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=40)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 40}
    scheduler._elastic_graph_carrier_step_key = (0, 3, 32, 128, 4)
    scheduler.running = [object() for _ in range(33)]
    scheduler.waiting = deque()
    scheduler.skipped_waiting = deque()

    grown_execution = scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(33)}, 3, True
    )
    assert grown_execution == (0, 3, 40, 160, 4)
    scheduler._commit_elastic_graph_carrier_step_key(grown_execution)

    scheduler._commit_elastic_graph_carrier_step_key((1, 0, 40, 40, 1))
    assert scheduler._elastic_graph_carrier_step_key == (0, 3, 40, 160, 4)
    scheduler._commit_elastic_graph_carrier_step_key((1, 3, 40, 40, 1))
    assert scheduler._elastic_graph_carrier_step_key == (0, 3, 40, 160, 4)

    scheduler.running = [object() for _ in range(5)]
    assert scheduler._canonical_elastic_graph_step_key(
        {str(index): 4 for index in range(5)}, 3, True
    ) == (0, 3, 8, 32, 4)
    scheduler._commit_elastic_graph_carrier_step_key((0, 3, 8, 32, 4))
    assert scheduler._elastic_graph_carrier_step_key == (0, 3, 40, 160, 4)

    scheduler.running = []
    assert scheduler._canonical_elastic_graph_step_key({"new": 4}, 3, True) == (
        0,
        3,
        1,
        4,
        4,
    )


def test_form_transitions_follow_work_at_fixed_request_count():
    scheduler = _new_elastic_scheduler()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=40)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"decode_max_x": 40}
    scheduler._elastic_graph_carrier_step_key = (0, 3, 32, 128, 4)
    for counts, pure_decode, expected in [
        ((1024, 4), False, (0, 3, 2, 2048, 0)),
        ((4, 4), True, (0, 3, 2, 8, 4)),
        ((1, 1), True, (1, 3, 2, 2, 1)),
        ((4, 1024), False, (0, 3, 2, 2048, 0)),
        ((4, 4), True, (0, 3, 2, 8, 4)),
    ]:
        actual = scheduler._canonical_elastic_graph_step_key(
            dict(zip(("a", "b"), counts)), 3, pure_decode
        )
        assert actual == expected
        scheduler._commit_elastic_graph_carrier_step_key(actual)
        assert scheduler._elastic_graph_carrier_step_key == (0, 3, 32, 128, 4)


def test_mixed_execution_resolves_complete_stable_carrier_owner_set():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("carrier-closure"))
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy((4096,))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_compiled_piecewise_sizes = frozenset({4096})
    execution_key = (0, 3, 32, 4096, 0)

    execution_owners = scheduler._resolve_elastic_step_physical_keys(execution_key)
    closure_key = scheduler._elastic_graph_carrier_closure_step_key(execution_key)
    closure_owners = scheduler._resolve_elastic_step_physical_keys(closure_key)

    assert [key.logical.owner for key in execution_owners] == ["mtp_decode"]
    assert closure_key == (0, 3, 32, 128, 4)
    assert [key.logical.owner for key in closure_owners] == [
        "target",
        "mtp_prefill",
        "mtp_decode",
    ]


def test_cold_mixed_preflight_prices_closure_but_captures_current_owner_set():
    generation = RuntimeGeneration("cold-carrier-closure")
    scheduler = _new_elastic_scheduler(generation)
    execution_key = (0, 3, 32, 4096, 0)
    closure_key = (0, 3, 32, 128, 4)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy((4096,))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_compiled_piecewise_sizes = frozenset({4096})
    scheduler._elastic_admission_controller.capture_envelopes = {
        Scheduler._elastic_graph_owner_key(closure_key): (900, 90, 0)
    }
    scheduler._elastic_graph_catalog = {
        closure_key: {"cold_peak_bytes": 900, "floor_bytes": 90}
    }
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.step_key = None
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.transition_floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 0
    _clear_elastic_maintenance(scheduler)
    closure_owners = scheduler._resolve_elastic_step_physical_keys(closure_key)
    for owner in closure_owners:
        scheduler._elastic_admission_controller.register(
            owner, price=GraphPrice(100, 200, owner.logical.owner)
        )
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 5_000
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    fits, required, available = scheduler._can_fund_elastic_graph_step(
        execution_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=True,
    )

    assert fits is False
    assert required == 900
    assert available == 5_000
    assert (
        scheduler._elastic_admission_controller.pending_maintenance_step_key
        == execution_key
    )
    plan = _pending_elastic_maintenance(scheduler)
    assert plan is not None
    current_owners = scheduler._resolve_elastic_step_physical_keys(execution_key)
    assert [key.logical.owner for key in current_owners] == ["mtp_decode"]
    assert plan.physical_keys == current_owners
    assert plan.capture_order == current_owners
    assert scheduler._elastic_step_residency_intent(execution_key)[1] == closure_owners


def test_unknown_calibration_shape_borrows_full_tail_in_maintenance():
    generation = RuntimeGeneration("calibration-maintenance-holdout")
    scheduler = _new_elastic_scheduler(generation)
    step_key = (1, 0, 7, 7, 1)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_compiled_piecewise_sizes = frozenset()
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_graph_catalog_coverage = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.step_key = None
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.transition_floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 640
    scheduler._elastic_admission_controller.cublas_workspace_bytes = 128
    _clear_elastic_maintenance(scheduler)
    physical_keys = scheduler._resolve_elastic_step_physical_keys(step_key)
    assert physical_keys
    for physical_key in physical_keys:
        scheduler._elastic_admission_controller.register(
            physical_key,
            price=GraphPrice(100, 200, physical_key.logical.owner),
        )
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 640
    coordinator.max_elastic_external_memory.return_value = 5_000
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    fits, required, available = scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=0,
        allow_maintenance=True,
    )

    assert fits is False
    assert required == 5_000
    assert available == 5_000
    plan = _pending_elastic_maintenance(scheduler)
    assert plan is not None
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.capture_loan_bytes == 5_000


def test_calibration_hot_replay_does_not_reopen_capture_state():
    generation = RuntimeGeneration("calibration-hot-replay")
    scheduler = _new_elastic_scheduler(generation)
    step_key = (1, 0, 5, 5, 1)
    scheduler._elastic_restore_mode = True
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.step_key = None
    scheduler._elastic_admission_controller.resident_bytes = 640
    scheduler._elastic_admission_controller.measured_bytes = {step_key: 512}
    physical_keys = scheduler._resolve_elastic_step_physical_keys(step_key)
    assert physical_keys
    for physical_key in physical_keys:
        scheduler._elastic_admission_controller.register(
            physical_key,
            price=GraphPrice(128, 192, physical_key.logical.owner),
        )
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(128, 192, physical_key.logical.owner),
            pinned=False,
        )

    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        640,
        False,
    )


def test_compiled_piecewise_carrier_keeps_hot_endpoint_without_capture_envelope():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("compiled-b4096"))
    scheduler._elastic_restore_mode = False
    scheduler._elastic_compiled_piecewise_sizes = frozenset({4096})
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy((4096,))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_admission_controller.step_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.resident_bytes = 192
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.transition_floor_bytes = 0
    scheduler._elastic_piecewise_physical_floor_bytes = Mock(return_value=0)
    scheduler._elastic_admission_controller.capture_envelopes = {
        (0, 0, 40, 4096, 0): (700, 0, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {(0, 0, 40, 4096, 0): 700}

    assert scheduler._resolve_elastic_step_physical_keys((0, 0, 40, 4096, 0)) == ()
    assert scheduler._estimate_elastic_graph_step_bytes((0, 0, 40, 4096, 0)) == (
        192,
        False,
    )

    scheduler._elastic_compiled_piecewise_sizes = frozenset(
        {256, 512, 1024, 2048, 4096}
    )
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy(
        (256, 512, 1024, 2048, 4096)
    )
    assert scheduler._estimate_elastic_graph_step_bytes((0, 0, 1, 1024, 0)) == (
        192,
        False,
    )

    # Calibration and serving consume the same compiled-only physical DAG.
    # A stale logical/global envelope must not become a fictitious Graph floor
    # after the finite-wave preflight has allocated request KV.
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.capture_envelopes = {
        (0, 0, 0, 0, 0): (5_496_635_392, 0, 0)
    }
    scheduler._elastic_admission_controller.resident_bytes = 0
    assert scheduler._estimate_elastic_graph_step_bytes((0, 0, 42, 4096, 0)) == (
        0,
        False,
    )

    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.resident_bytes = 192

    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 5_000
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    scheduler.elastic_on_demand_graphs = True
    assert scheduler._can_fund_elastic_graph_step(
        (0, 0, 1, 1024, 0), minimum_free_primary_blocks=1
    ) == (True, 192, 5_000)


def test_compiled_k3_carrier_prices_only_its_full_physical_subset():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("compiled-k3-b4096"))
    scheduler._elastic_restore_mode = False
    scheduler._elastic_compiled_piecewise_sizes = frozenset({4, 4096})
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy((4, 4096))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_admission_controller.capture_envelopes = {
        (0, 3, 0, 0, 0): (416 << 20, 22 << 20, 0)
    }
    scheduler._elastic_graph_catalog = {
        (0, 3, 2, 4, 0): {
            "cold_peak_bytes": 64 << 20,
            "floor_bytes": 22 << 20,
        }
    }

    step_key = (0, 3, 2, 4096, 0)
    physical_keys = scheduler._resolve_elastic_step_physical_keys(step_key)
    assert [key.logical.owner for key in physical_keys] == ["mtp_decode"]
    assert [key.logical.mode for key in physical_keys] == ["FULL"]
    scheduler._elastic_admission_controller.register(
        physical_keys[0],
        price=GraphPrice(22 << 20, 24 << 20, "mtp-decode-x2"),
    )

    assert scheduler._elastic_capture_envelope(step_key) == (
        64 << 20,
        22 << 20,
        0,
    )
    assert scheduler._elastic_piecewise_physical_floor_bytes(step_key) == 0


def test_elastic_sealed_physical_superset_prices_k0_full_subset():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("subset-test"))
    scheduler._elastic_restore_mode = False
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_graph_catalog = {
        (1, 3, 1, 1, 1): {
            "cold_peak_bytes": 900,
            "floor_bytes": 90,
        },
        (1, 3, 1, 4, 4): {
            "cold_peak_bytes": 1200,
            "floor_bytes": 80,
        },
    }

    # K0 uses only the target FULL/q1 descriptor. The K3/q1 row contains that
    # exact physical owner; q4 is a different target descriptor and must not
    # be composed into this envelope.
    assert scheduler._elastic_capture_envelope((1, 0, 1, 1, 1)) == (
        900,
        90,
        0,
    )


def test_elastic_sealed_superset_does_not_price_unrelated_full_geometry():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("subset-test"))
    scheduler._elastic_restore_mode = False
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_graph_catalog = {
        (1, 3, 1, 1, 1): {
            "cold_peak_bytes": 900,
            "floor_bytes": 90,
        }
    }

    assert scheduler._elastic_capture_envelope((1, 0, 2, 2, 1)) is None


def test_elastic_product_cold_non_mtp_shape_requires_final_shape_authority():
    generation = RuntimeGeneration("subset-test")
    scheduler = _new_elastic_scheduler(generation)
    step_key = (1, 0, 1, 1, 1)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_graph_catalog = {
        (1, 3, 1, 1, 1): {
            "cold_peak_bytes": 900,
            "floor_bytes": 90,
        }
    }
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller = ElasticAdmissionController(generation)
    scheduler._elastic_admission_controller.step_key = None
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.transition_floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 0
    _clear_elastic_maintenance(scheduler)
    physical_key = resolve_step_physical_keys(step_key, generation, 4096)[0]
    scheduler._elastic_admission_controller.register(
        physical_key,
        price=GraphPrice(100, 200, "target"),
    )
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 5_000
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        # A user prefix has already been selected. This candidate has no
        # immutable final-wave authority and cannot prepare maintenance.
        allow_maintenance=False,
    ) == (False, 900, 5_000)
    assert _pending_elastic_maintenance(scheduler) is None
    assert scheduler._elastic_last_defer_reason == "final_shape_maintenance_required"

    # The same shape, when named by the complete-wave preflight, may prepare
    # one immutable capture plan. Serving binds it to the first exact USER;
    # only calibration/restore may execute it request-free.
    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=True,
    ) == (False, 900, 5_000)
    assert _pending_elastic_maintenance(scheduler) is not None
    assert _pending_elastic_maintenance(scheduler).kind == ElasticPlanKind.MAINTENANCE

    plan = _pending_elastic_maintenance(scheduler)
    scheduler._elastic_admission_controller.begin_maintenance(plan)
    scheduler._elastic_admission_controller.finish_maintenance(
        plan,
        {physical_key: GraphPrice(100, 200, "target")},
    )
    scheduler._elastic_admission_controller.resident_bytes = 100
    scheduler._elastic_admission_controller.measured_bytes[step_key] = 100
    _clear_elastic_maintenance(scheduler)

    # Once the exact owner is HOT, the identical candidate advances without
    # another maintenance plan.
    hot_fits, _required, available = scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=False,
    )
    assert hot_fits is True
    assert available == 5_000
    assert _pending_elastic_maintenance(scheduler) is None


def _make_elastic_running_text_wave_scheduler(num_reqs: int = 8) -> Scheduler:
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    _clear_elastic_maintenance(scheduler)
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler.waiting = ()
    scheduler.skipped_waiting = ()
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.current_step = 7
    scheduler.max_model_len = 262144
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.num_prefill_lookahead = 1
    scheduler.need_mamba_block_aligned_split = False
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.use_eagle = False
    scheduler.use_eagle_block_drop = False
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_prefill_checkpoint_alignment = None
    scheduler.mamba_fine_grained_prefix_cache = False
    scheduler.scheduler_config = Mock(
        long_prefill_token_threshold=0,
        max_num_batched_tokens=4096,
    )
    scheduler._elastic_graph_catalog_coverage = {
        "mtp_verifier_contract": "batched-causal-q1-v1",
        "decode_max_x": 40,
    }
    scheduler.vllm_config = Mock(
        speculative_config=Mock(disable_speculation_on_non_decode=False)
    )
    scheduler.running = []
    scheduler.requests = {}
    scheduler._elastic_deferred_mm_wave = None
    scheduler._elastic_maintenance_started = {}
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.lora_config = None
    scheduler.canonical_prefill_admission = False
    scheduler.max_num_running_reqs = 40
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.policy = SchedulingPolicy.FCFS
    for index in range(num_reqs):
        request = Mock(
            request_id=f"decode-{index}",
            status=RequestStatus.RUNNING,
            num_output_placeholders=0,
            num_computed_tokens=100,
            num_prompt_tokens=100,
            execution_prefill_len=100,
            max_tokens=2048,
            next_decode_eligible_step=0,
            is_prefill_chunk=False,
            has_encoder_inputs=False,
            num_tokens_with_spec=104,
            num_tokens=101,
            spec_token_ids=[11, 12, 13],
            force_non_speculative=False,
        )
        request.is_finished.return_value = False
        scheduler.running.append(request)
        scheduler.requests[request.request_id] = request
    scheduler._elastic_successor_primary_headroom = Mock(return_value=0)
    scheduler._elastic_remaining_resource_requirements = Mock(
        return_value=KVCacheBlockPoolRequirements()
    )
    scheduler._reserve_elastic_admission = Mock(return_value=True)
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.enable_kv_cache_events = False
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.elastic_gdn_blocks_after_allocation.return_value = 0
    return scheduler


def test_running_text_wave_preflights_only_final_assembled_shape():
    scheduler = _make_elastic_running_text_wave_scheduler()
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 8, 32, 4)
    assert prepared_maintenance is False
    scheduler._can_fund_elastic_graph_step.assert_called_once_with(
        key,
        minimum_free_primary_blocks=0,
        gdn_blocks=0,
        allow_maintenance=True,
    )


def test_running_text_wave_prepares_one_final_shape_maintenance():
    scheduler = _make_elastic_running_text_wave_scheduler()

    def prepare_final(key, **_kwargs):
        _arm_fixture_capture(scheduler, key)
        return False, 96, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=prepare_final)

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 8, 32, 4)
    assert prepared_maintenance is True
    scheduler._can_fund_elastic_graph_step.assert_called_once()


@pytest.mark.parametrize(("source_x", "destination_x"), [(32, 16), (4, 8)])
@pytest.mark.parametrize("with_waiting", [False, True])
@pytest.mark.parametrize(
    "blocker", [None, "inflight", "capacity", "unknown", "lease", "loan"]
)
def test_running_cold_form_pressure_reclaim_makes_progress(
    source_x, destination_x, blocker, with_waiting
):
    """A quiescent wave must not wait forever for optional pinned residency."""
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=destination_x)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    controller = scheduler._elastic_admission_controller
    source = (0, 3, source_x, source_x * 4, 4)
    destination = (0, 3, destination_x, destination_x * 4, 4)
    scheduler._elastic_graph_carrier_step_key = source
    terminal = scheduler._resolve_elastic_step_physical_keys((1, 3, 40, 40, 1))[-1]
    scheduler._elastic_serving_carrier_keys = (terminal,)
    controller.publish_hot(terminal, GraphPrice(20, 20, "terminal"), pinned=True)
    for key in scheduler._resolve_elastic_step_physical_keys(source):
        controller.publish_hot(key, GraphPrice(30, 30, key.identity), pinned=True)
    controller.resident_bytes = 120
    controller.floor_bytes = 10
    controller.transition_floor_bytes = 10
    controller.record_capture_envelope(
        scheduler._elastic_graph_owner_key(destination), (100, 10, 0)
    )
    scheduler._elastic_graph_catalog = {}
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.elastic_external_memory_bytes = 120
    coordinator.max_elastic_external_memory.return_value = 150
    coordinator.normalize_elastic_external_memory.side_effect = lambda n: n
    if with_waiting:
        request = Mock(
            request_id="optional-waiting",
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            num_stale_output_tokens=0,
            has_encoder_inputs=False,
            force_non_speculative=False,
            num_tokens=4,
            num_prompt_tokens=4,
            execution_prefill_len=4,
        )
        scheduler.waiting = [request]
        scheduler.requests[request.request_id] = request
        scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
        scheduler.kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = (  # noqa: E501
            KVCacheBlockPoolRequirements(primary=7)
        )
        # The optional expansion cannot be admitted by reclaim; it must not
        # remove the independently fundable RUNNING cold-form recovery path.
        scheduler._prepare_elastic_waiting_deficit_reclaim = Mock(return_value=False)
        can_fund = scheduler._can_fund_elastic_graph_step

        def fund_running_only(key, **kwargs):
            if key != destination:
                return False, 1000, 150
            return can_fund(key, **kwargs)

        scheduler._can_fund_elastic_graph_step = fund_running_only
    if blocker == "capacity":
        coordinator.max_elastic_external_memory.return_value = 129
    elif blocker == "unknown":
        controller.capture_envelopes.clear()
    elif blocker == "lease":
        user = controller.plan(
            "inflight",
            (terminal, *controller.entries),
            request_bytes=0,
            available_bytes=1000,
        )
        controller.commit_user(user)
    elif blocker == "loan":
        controller.reserve_loan(source, 120)
    entries = dict(controller.entries)

    # Neither a pinned graph nor a retained high-water graph is an ordinary
    # LRU victim. The old implementation returned (None, False) forever.
    key, maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
        physical_quiescent=blocker != "inflight",
    )

    if blocker is not None:
        assert key is None and not maintenance
        assert controller.pending_maintenance_plan is None
        assert dict(controller.entries) == entries
        coordinator.set_elastic_external_memory.assert_not_called()
        return
    assert key is None and maintenance
    reclaim = controller.pending_maintenance_plan
    assert reclaim is not None and reclaim.kind == ElasticPlanKind.PRESSURE_RECLAIM
    assert reclaim.protected_keys == (terminal,)
    assert set(reclaim.victim_keys) == set(
        scheduler._resolve_elastic_step_physical_keys(source)
    )
    assert dict(controller.entries) == entries
    coordinator.set_elastic_external_memory.assert_not_called()
    scheduler._reserve_elastic_admission.assert_not_called()
    # Existing worker receipt synchronization supplies this boundary. Reclaim
    # has no USER consumer, then the unchanged RUNNING cohort can prepare its
    # exact cold destination instead of inheriting the source execution M.
    controller.begin_reclaim(reclaim)
    controller.synchronize_hot(((terminal, entries[terminal].price, True, 0),))
    controller.clear_maintenance()
    controller.resident_bytes = 30
    coordinator.elastic_external_memory_bytes = 30
    key, maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
        physical_quiescent=True,
    )
    assert key == destination and maintenance
    capture = controller.pending_maintenance_plan
    assert capture is not None and capture.kind == ElasticPlanKind.MAINTENANCE
    assert set(capture.cold_misses) == set(
        scheduler._resolve_elastic_step_physical_keys(destination)
    )
    controller.clear_maintenance()
    for physical_key in capture.cold_misses:
        controller.publish_hot(
            physical_key, GraphPrice(30, 30, physical_key.identity), pinned=True
        )
    controller.resident_bytes = 120
    coordinator.elastic_external_memory_bytes = 120
    key, maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
        physical_quiescent=True,
    )
    assert key == destination and not maintenance
    assert controller.pending_maintenance_plan is None


def test_running_tail_reclaim_uses_independent_destination_and_new_receipt():
    """Saved byte boundary: no shape fallback, then reprice real survivors."""
    mib = 1 << 20
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=32)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    controller = scheduler._elastic_admission_controller
    source, destination = (0, 3, 39, 156, 4), (0, 3, 32, 128, 4)
    scheduler._elastic_graph_carrier_step_key = source
    scheduler._elastic_terminal_decode_carrier_x = 39
    source_keys = scheduler._resolve_elastic_step_physical_keys(source)
    destination_keys = scheduler._resolve_elastic_step_physical_keys(destination)
    terminal = next(key for key in source_keys if key.logical.owner == "mtp_decode")
    scheduler._elastic_serving_carrier_keys = (terminal,)
    prices = {"target": 26, "mtp_prefill": 46, "mtp_decode": 70}
    entries = tuple(
        sorted(
            (
                ElasticResidencyEntry(
                    key=key,
                    pinned=True,
                    resident_bytes=prices[key.logical.owner] * mib,
                    local_pool_bytes=0,
                    reclaimable_bytes=0,
                )
                for key in source_keys
            ),
            key=lambda entry: entry.key.identity,
        )
    )
    controller.accept_residency_receipt(
        ElasticResidencyReceipt(
            generation=controller.generation,
            transaction_id=None,
            resident_bytes=174 * mib,
            floor_bytes=0,
            transition_floor_bytes=31 * mib,
            peak_bytes=174 * mib,
            cublas_workspace_bytes=32 * mib,
            entries=entries,
        )
    )
    scheduler._elastic_graph_catalog = {
        destination: dict(
            cold_peak_bytes=168 * mib,
            hot_peak_bytes=200 * mib,
            resident_bytes=200 * mib,
            floor_bytes=0,
            resident_key_bytes=[(key.identity, 40 * mib) for key in destination_keys],
        )
    }
    # Conflicting same-M fallback is deliberately present. Only identical
    # physical DAG evidence may select the 200 MiB destination endpoint.
    controller.record_capture_envelope(
        scheduler._elastic_graph_owner_key(destination),
        (202 * mib, 0, 0),
        resident_key_bytes=((terminal.identity, 44 * mib),),
    )
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.elastic_external_memory_bytes = 174 * mib
    coordinator.max_elastic_external_memory.return_value = 296 * mib
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    assert scheduler._can_fund_elastic_graph_step(
        destination,
        minimum_free_primary_blocks=0,
        allow_maintenance=True,
        preview_only=True,
    ) == (False, 374 * mib, 296 * mib)
    assert scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
        physical_quiescent=True,
    ) == (None, True)
    reclaim = controller.pending_maintenance_plan
    assert reclaim is not None and reclaim.kind == ElasticPlanKind.PRESSURE_RECLAIM
    assert reclaim.protected_keys == (terminal,)
    assert set(reclaim.victim_keys) == set(source_keys) - {terminal}
    coordinator.set_elastic_external_memory.assert_not_called()
    scheduler._reserve_elastic_admission.assert_not_called()
    controller.begin_reclaim(reclaim)
    controller.accept_residency_receipt(
        ElasticResidencyReceipt(
            generation=controller.generation,
            transaction_id=reclaim.transaction_id,
            resident_bytes=70 * mib,
            floor_bytes=0,
            transition_floor_bytes=0,
            peak_bytes=174 * mib,
            cublas_workspace_bytes=0,
            entries=tuple(entry for entry in entries if entry.key == terminal),
        ),
        expected_transaction_id=reclaim.transaction_id,
    )
    controller.clear_maintenance()
    coordinator.elastic_external_memory_bytes = 70 * mib
    assert scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
        physical_quiescent=True,
    ) == (destination, True)
    capture = controller.pending_maintenance_plan
    assert capture is not None and capture.kind == ElasticPlanKind.MAINTENANCE
    assert capture.capture_loan_bytes == 270 * mib
    assert set(capture.cold_misses) == set(destination_keys)
    # After physical reclaim, the same wave reserves its exact capture grant.
    # The stub contains allocation; no real request/KV state is consumed here.
    coordinator.set_elastic_external_memory.assert_not_called()
    scheduler._reserve_elastic_admission.assert_called_once_with(
        destination,
        external_memory_bytes=270 * mib,
        minimum_free_primary_blocks=0,
        requirements=KVCacheBlockPoolRequirements(),
    )


@pytest.mark.parametrize("evidence_state", ["valid", "missing", "stale", "victim_only"])
def test_cold_reclaim_sharing_is_receipt_backed_and_survives_teardown(evidence_state):
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=8)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    controller = scheduler._elastic_admission_controller
    destination = (0, 3, 8, 32, 4)
    keys = scheduler._resolve_elastic_step_physical_keys(destination)
    terminal, optional = keys[-1], keys[0]
    scheduler._elastic_serving_carrier_keys = (terminal,)
    entries = tuple(
        sorted(
            (
                ElasticResidencyEntry(
                    key=key,
                    pinned=True,
                    resident_bytes=40,
                    local_pool_bytes=0,
                    reclaimable_bytes=0,
                )
                for key in (terminal, optional)
            ),
            key=lambda entry: entry.key.identity,
        )
    )
    controller.accept_residency_receipt(
        ElasticResidencyReceipt(
            generation=controller.generation,
            transaction_id=None,
            resident_bytes=80,
            floor_bytes=0,
            transition_floor_bytes=0,
            peak_bytes=80,
            cublas_workspace_bytes=0,
            entries=entries,
        )
    )
    evidence: tuple[tuple[str, int], ...] | None = ((terminal.identity, 20),)
    if evidence_state == "missing":
        evidence = None
    elif evidence_state == "stale":
        evidence = ((terminal.identity + "-stale", 20),)
    elif evidence_state == "victim_only":
        evidence = ((optional.identity, 20),)
    controller.record_capture_envelope(
        scheduler._elastic_graph_owner_key(destination),
        (100, 0, 0),
        resident_key_bytes=evidence,
    )
    scheduler._elastic_graph_catalog = {}
    assert scheduler._prepare_elastic_cold_form_reclaim(
        destination,
        required_external=180,
        available_external=120,
        physical_quiescent=True,
    ) is (evidence_state == "valid")
    if evidence_state != "valid":
        assert controller.pending_maintenance_plan is None


def test_hot_preflight_does_not_resolve_cold_catalog_evidence():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=8)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    step = (0, 3, 8, 32, 4)
    controller = scheduler._elastic_admission_controller
    _publish_elastic_step_hot(scheduler, step)
    controller.resident_bytes = 120
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.elastic_external_memory_bytes = 120
    coordinator.max_elastic_external_memory.return_value = 150
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler._elastic_capture_envelope_with_provenance = Mock(
        side_effect=AssertionError("HOT replay consulted COLD evidence")
    )
    assert scheduler._can_fund_elastic_graph_step(
        step,
        minimum_free_primary_blocks=0,
    ) == (True, 120, 150)


def test_running_wave_defer_observer_deduplicates_but_reports_new_budget():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=1)
    scheduler._elastic_last_defer_reason = "insufficient_reclaimable_graph_and_kv_bytes"
    key = (0, 3, 1, 4, 4)
    with patch("vllm.v1.core.sched.scheduler.logger.warning") as warning:
        for available in (100, 100, 120):
            scheduler._observe_elastic_running_wave_defer(
                key,
                required_external=200,
                available_external=available,
                physical_quiescent=True,
            )
        assert warning.call_count == 2
        assert warning.call_args_list[0].args[1:5] == (
            key,
            scheduler._elastic_last_defer_reason,
            200,
            100,
        )
        assert warning.call_args_list[1].args[4] == 120


def test_running_text_wave_prices_full_future_kv_and_gdn_before_allocation():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=2)
    scheduler._elastic_successor_primary_headroom.return_value = 1
    scheduler._elastic_remaining_resource_requirements.return_value = (
        KVCacheBlockPoolRequirements(primary=72, mamba=6)
    )
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.elastic_gdn_blocks_after_allocation.return_value = 13
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 2, 8, 4)
    assert prepared_maintenance is False
    scheduler._can_fund_elastic_graph_step.assert_called_once_with(
        key,
        minimum_free_primary_blocks=73,
        gdn_blocks=13,
        allow_maintenance=True,
    )
    scheduler._reserve_elastic_admission.assert_called_once_with(
        key,
        external_memory_bytes=64,
        minimum_free_primary_blocks=73,
        requirements=KVCacheBlockPoolRequirements(primary=72, mamba=6),
    )


def test_elastic_joint_reservation_rolls_back_when_gdn_mapping_fails():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_preflight_admission_grant = None
    scheduler._resolve_elastic_step_physical_keys = Mock(return_value=())
    coordinator = Mock(elastic_external_memory_bytes=10)

    def set_external(requested: int, **_kwargs) -> bool:
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = set_external
    coordinator.ensure_elastic_capacity.return_value = False
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert not scheduler._reserve_elastic_admission(
        (0, 3, 2, 4096, 0),
        external_memory_bytes=100,
        minimum_free_primary_blocks=72,
        requirements=KVCacheBlockPoolRequirements(primary=72, mamba=6),
    )
    assert coordinator.elastic_external_memory_bytes == 10
    coordinator.rebalance_elastic_capacity.assert_called_once_with()
    assert coordinator.set_elastic_external_memory.call_args_list == [
        call(100, minimum_free_primary_blocks=72),
        call(10),
    ]
    assert scheduler._elastic_preflight_admission_grant is None


def test_running_text_wave_prices_budget_limited_final_shape_only():
    scheduler = _make_elastic_running_text_wave_scheduler()
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=12,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 4, 16, 4)
    assert prepared_maintenance is False
    scheduler._can_fund_elastic_graph_step.assert_called_once_with(
        key,
        minimum_free_primary_blocks=0,
        gdn_blocks=0,
        allow_maintenance=True,
    )


def test_running_text_wave_excludes_temporarily_ineligible_request():
    scheduler = _make_elastic_running_text_wave_scheduler()
    scheduler.running[0].next_decode_eligible_step = scheduler.current_step + 1
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 8, 32, 4)
    assert prepared_maintenance is False
    scheduler._can_fund_elastic_graph_step.assert_called_once_with(
        key,
        minimum_free_primary_blocks=0,
        gdn_blocks=0,
        allow_maintenance=True,
    )


def test_running_text_wave_preserves_liveness_with_waiting_work():
    scheduler = _make_elastic_running_text_wave_scheduler()
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    waiting = [
        Mock(
            request_id=f"prefill-{index}",
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            num_stale_output_tokens=0,
            has_encoder_inputs=False,
            force_non_speculative=False,
            num_tokens=128,
            num_prompt_tokens=128,
            execution_prefill_len=128,
        )
        for index in range(24)
    ]
    scheduler.waiting = waiting
    scheduler.requests.update({request.request_id: request for request in waiting})
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler.kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = (  # noqa: E501
        KVCacheBlockPoolRequirements(primary=7)
    )
    running_key = (0, 3, 8, 32, 4)
    joint_key = (0, 3, 32, 4096, 0)
    scheduler._can_fund_elastic_graph_step = Mock(
        side_effect=lambda key, **_kwargs: (
            (True, 918_552_576, 1_000_000_000)
            if key == running_key
            else (False, 1_503_657_984, 1_000_000_000)
        )
    )
    scheduler._prepare_elastic_waiting_deficit_reclaim = Mock(return_value=False)
    waiting_before = tuple(scheduler.waiting)

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == running_key
    assert prepared_maintenance is False
    attempted = scheduler._can_fund_elastic_graph_step.call_args_list
    assert attempted[0].args[0] == joint_key
    assert attempted[-1].args[0] == running_key
    assert all(item.kwargs.get("preview_only") for item in attempted[:-1])
    scheduler._prepare_elastic_waiting_deficit_reclaim.assert_called_once()
    scheduler._reserve_elastic_admission.assert_called_once()
    assert scheduler._reserve_elastic_admission.call_args.args[0] == running_key
    assert scheduler._elastic_preflight_joint_waiting_request_ids == ()
    assert scheduler._elastic_preflight_waiting_ignore_prefix_request_ids == ()
    assert tuple(scheduler.waiting) == waiting_before


def test_running_x8_waiting_x24_schedule_progresses_when_joint_wave_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    """Replay the saved post-first-token X8/Waiting24 liveness boundary."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=40,
        max_num_batched_tokens=4096,
        num_speculative_tokens=3,
    )
    scheduler.elastic_on_demand_graphs = False
    scheduler._elastic_graph_execution_policy = None
    requests = create_requests(
        num_requests=32,
        num_tokens=128,
        max_tokens=2048,
        ignore_eos=True,
    )
    running = requests[:8]
    waiting = requests[8:]
    for request in running:
        scheduler.add_request(request)

    prefill = scheduler.schedule()
    running_ids = [request.request_id for request in running]
    scheduler.update_from_output(
        prefill,
        ModelRunnerOutput(
            req_ids=running_ids,
            req_id_to_index={
                request_id: index for index, request_id in enumerate(running_ids)
            },
            sampled_token_ids=[[0] for _request in running],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.update_draft_token_ids(
        DraftTokenIds(running_ids, [[11, 12, 13] for _request in running])
    )
    for request in waiting:
        scheduler.add_request(request)

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    running_key = (0, 3, 8, 32, 4)
    for physical_key in scheduler._resolve_elastic_step_physical_keys(running_key):
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(1, 1, f"test:{physical_key.identity}"),
            pinned=False,
        )

    def price_wave(key, **_kwargs):
        if key == running_key:
            return True, 64, 128
        return False, 192, 128

    can_fund = Mock(side_effect=price_wave)
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)
    monkeypatch.setattr(
        scheduler, "_reserve_elastic_admission", Mock(return_value=True)
    )
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 4 for request in running}
    assert output.elastic_graph_step_key == running_key
    assert can_fund.call_args_list[0].args[0] == (0, 3, 32, 4096, 0)
    assert can_fund.call_args_list[-1].args[0] == running_key
    assert all(item.kwargs.get("preview_only") for item in can_fund.call_args_list[:-1])
    assert tuple(scheduler.waiting) == tuple(waiting)
    assert all(request.status == RequestStatus.RUNNING for request in running)
    assert all(request.status == RequestStatus.WAITING for request in waiting)
    assert scheduler._elastic_preflight_joint_waiting_request_ids == ()
    assert scheduler._elastic_preflight_waiting_ignore_prefix_request_ids == ()


def test_running_x8_waiting_x24_selects_largest_feasible_fcfs_prefix():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=8)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    waiting = [
        Mock(
            request_id=f"prefill-{index}",
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            num_stale_output_tokens=0,
            has_encoder_inputs=False,
            force_non_speculative=False,
            num_tokens=128,
            num_prompt_tokens=128,
            execution_prefill_len=128,
        )
        for index in range(24)
    ]
    scheduler.waiting = waiting
    scheduler.requests.update({request.request_id: request for request in waiting})
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)

    def price_wave(key, **kwargs):
        # X32..X21 do not fit. X20 is feasible and must be selected; X19 and
        # smaller must never be probed after the descending first fit.
        if kwargs.get("preview_only"):
            return key[2] <= 20, 96, 128
        assert key[2] == 20
        return True, 96, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=price_wave)
    scheduler._reserve_elastic_admission.return_value = True

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key is not None and key[2] == 20
    assert prepared_maintenance is False
    assert scheduler._elastic_preflight_joint_waiting_request_ids == tuple(
        request.request_id for request in waiting[:12]
    )
    preview_x = [
        call.args[0][2]
        for call in scheduler._can_fund_elastic_graph_step.call_args_list
        if call.kwargs.get("preview_only")
    ]
    assert preview_x == list(range(32, 19, -1))
    assert scheduler._can_fund_elastic_graph_step.call_args_list[-1].args[0][2] == 20


def test_running_x8_waiting_x24_schedule_commits_largest_feasible_fcfs_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    """The public scheduler consumes X20 and leaves the exact FCFS suffix."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=40,
        max_num_batched_tokens=4096,
        num_speculative_tokens=3,
    )
    scheduler.elastic_on_demand_graphs = False
    scheduler._elastic_graph_execution_policy = None
    requests = create_requests(
        num_requests=32,
        num_tokens=128,
        max_tokens=2048,
        ignore_eos=True,
    )
    running = requests[:8]
    waiting = requests[8:]
    for request in running:
        scheduler.add_request(request)
    initial = scheduler.schedule()
    running_ids = [request.request_id for request in running]
    scheduler.update_from_output(
        initial,
        ModelRunnerOutput(
            req_ids=running_ids,
            req_id_to_index={request_id: i for i, request_id in enumerate(running_ids)},
            sampled_token_ids=[[0] for _request in running],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.update_draft_token_ids(
        DraftTokenIds(running_ids, [[11, 12, 13] for _request in running])
    )
    for request in waiting:
        scheduler.add_request(request)

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()

    def price_wave(key, **kwargs):
        feasible = key[2] <= 20
        if key[2] == 20 and not kwargs.get("preview_only"):
            closure = scheduler._elastic_graph_carrier_closure_step_key(key)
            physical_keys = dict.fromkeys(
                (
                    *scheduler._resolve_elastic_step_physical_keys(key),
                    *scheduler._resolve_elastic_step_physical_keys(closure),
                )
            )
            for physical_key in physical_keys:
                scheduler._elastic_admission_controller.publish_hot(
                    physical_key,
                    GraphPrice(
                        0,
                        0,
                        f"private-pool:{physical_key.identity}",
                    ),
                    pinned=False,
                )
        return feasible, 64 if feasible else 192, 128

    can_fund = Mock(side_effect=price_wave)
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)

    def reserve_final(
        key,
        *,
        external_memory_bytes,
        minimum_free_primary_blocks,
        requirements,
    ):
        scheduler._elastic_preflight_admission_grant = SimpleNamespace(
            step_key=key,
            physical_keys=scheduler._elastic_step_residency_intent(key)[2],
            maintenance_transaction_id=None,
            external_memory_bytes=external_memory_bytes,
            previous_external_memory_bytes=0,
            minimum_free_primary_blocks=minimum_free_primary_blocks,
            requirements=requirements,
        )
        return True

    monkeypatch.setattr(scheduler, "_reserve_elastic_admission", reserve_final)

    def reserve_zero_loan(key, **_kwargs):
        scheduler._elastic_admission_controller.reserve_loan(key, 0)
        scheduler._elastic_admission_controller.publish_step_key(key)
        return 0

    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", reserve_zero_loan)

    output = scheduler.schedule()

    admitted_waiting = waiting[:12]
    suffix = waiting[12:]
    assert set(output.num_scheduled_tokens) == {
        *(request.request_id for request in running),
        *(request.request_id for request in admitted_waiting),
    }
    assert output.elastic_graph_step_key is not None
    assert output.elastic_graph_step_key[2] == 20
    assert tuple(scheduler.waiting) == tuple(suffix)
    assert all(request.status == RequestStatus.RUNNING for request in admitted_waiting)
    assert all(request.status == RequestStatus.WAITING for request in suffix)
    assert all(request.num_computed_tokens == 0 for request in suffix)
    preview_x = [
        call.args[0][2]
        for call in can_fund.call_args_list
        if call.kwargs.get("preview_only")
    ]
    assert preview_x == list(range(32, 19, -1))

    # Consume the public output through the same settlement -> carrier commit
    # order as EngineCore.  X20 is deliberately not a bounded carrier: its
    # exact mtp_decode owner is modeled as already HOT by the schema-5 evidence
    # fixture, while the successor/tail affinity must commit as bounded X32.
    current_physical = scheduler._resolve_elastic_step_physical_keys(
        output.elastic_graph_step_key
    )
    exact_mtp = tuple(
        key for key in current_physical if key.logical.owner == "mtp_decode"
    )
    assert len(exact_mtp) == 1 and exact_mtp[0].physical_num_reqs == 20
    assert output.elastic_step_plan is not None
    assert output.elastic_step_plan.kind == ElasticPlanKind.USER
    resident_keys = tuple(
        sorted(
            (
                key
                for key, entry in (
                    scheduler._elastic_admission_controller.entries.items()
                )
                if entry.hot
            ),
            key=lambda key: key.identity,
        )
    )
    scheduled_ids = tuple(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=list(scheduled_ids),
            req_id_to_index={
                request_id: i for i, request_id in enumerate(scheduled_ids)
            },
            sampled_token_ids=[[0] for _request_id in scheduled_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            elastic_external_memory_bytes=output.elastic_external_memory_bytes,
            elastic_residency_receipt=ElasticResidencyReceipt(
                generation=scheduler._elastic_admission_controller.generation,
                transaction_id=output.elastic_step_plan.transaction_id,
                resident_bytes=0,
                floor_bytes=0,
                transition_floor_bytes=0,
                peak_bytes=0,
                cublas_workspace_bytes=0,
                entries=tuple(
                    ElasticResidencyEntry(
                        key=key,
                        pinned=True,
                        resident_bytes=0,
                        local_pool_bytes=0,
                        reclaimable_bytes=0,
                    )
                    for key in resident_keys
                ),
            ),
        ),
    )

    assert not scheduler._elastic_admission_controller.pending_loans
    assert scheduler._elastic_graph_carrier_step_key == (0, 3, 32, 128, 4)


def test_elastic_graph_prefix_preview_is_strictly_read_only():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("prefix-preview"))
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    key = (0, 3, 20, 2048, 0)
    scheduler._estimate_elastic_graph_step_bytes = Mock(return_value=(128, True))
    scheduler._elastic_admission_controller.measured_bytes[key] = 128
    for physical_key in scheduler._resolve_elastic_step_physical_keys(key):
        scheduler._elastic_admission_controller.register(
            physical_key,
            price=GraphPrice(32, 32, f"test:{physical_key.identity}"),
        )
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 256
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    controller = scheduler._elastic_admission_controller
    before = (
        controller.snapshot,
        controller.trace,
        controller._transaction_seq,
        getattr(scheduler, "_elastic_last_defer_reason", None),
    )

    assert scheduler._can_fund_elastic_graph_step(
        key,
        minimum_free_primary_blocks=20,
        gdn_blocks=0,
        allow_maintenance=True,
        preview_only=True,
    ) == (True, 256, 256)

    assert (
        controller.snapshot,
        controller.trace,
        controller._transaction_seq,
        getattr(scheduler, "_elastic_last_defer_reason", None),
    ) == before


def test_encoder_wave_overlay_deduplicates_shared_mm_across_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_seqs=2,
        max_num_batched_tokens=128,
    )
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=50)
    requests = create_requests(
        num_requests=2,
        num_tokens=50,
        mm_hashes_list=[["shared-image"], ["shared-image"]],
        mm_positions=[
            [PlaceholderRange(offset=0, length=50)],
            [PlaceholderRange(offset=0, length=50)],
        ],
    )
    overlay = EncoderWaveOverlay(scheduler.encoder_cache_manager.clone_for_preview())
    budget = 50
    scheduled: list[list[int]] = []

    for request in requests:
        encoder_inputs, num_tokens, budget, _external, _cached = (
            scheduler._try_schedule_encoder_inputs(
                request,
                0,
                50,
                budget,
                encoder_wave_overlay=overlay,
            )
        )
        assert num_tokens == 50
        scheduled.append(encoder_inputs)

    assert scheduled == [[0], []]
    assert budget == 0
    assert overlay.scheduled_identifiers == {"shared-image"}
    assert overlay.cache_manager.cached == {
        "shared-image": {request.request_id for request in requests}
    }
    assert scheduler.encoder_cache_manager.cached == {}
    assert scheduler.encoder_cache_manager.num_free_slots == 50


def test_encoder_wave_overlay_replays_lru_eviction_before_later_claim(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_seqs=3,
        max_num_batched_tokens=128,
    )
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=50)
    seed, new_request, later_old = create_requests(
        num_requests=3,
        num_tokens=50,
        mm_hashes_list=[["old-image"], ["new-image"], ["old-image"]],
        mm_positions=[
            [PlaceholderRange(offset=0, length=50)],
            [PlaceholderRange(offset=0, length=50)],
            [PlaceholderRange(offset=0, length=50)],
        ],
    )
    scheduler.encoder_cache_manager.allocate(seed, 0)
    scheduler.encoder_cache_manager.free(seed)
    overlay = EncoderWaveOverlay(scheduler.encoder_cache_manager.clone_for_preview())

    first = scheduler._try_schedule_encoder_inputs(
        new_request,
        0,
        50,
        100,
        encoder_wave_overlay=overlay,
    )
    second = scheduler._try_schedule_encoder_inputs(
        later_old,
        0,
        50,
        first[2],
        encoder_wave_overlay=overlay,
    )

    assert first[:3] == ([0], 50, 50)
    assert second[:3] == ([], 0, 50)
    assert set(overlay.cache_manager.cached) == {"new-image"}
    assert set(scheduler.encoder_cache_manager.cached) == {"old-image"}
    assert tuple(scheduler.encoder_cache_manager.freeable) == ("old-image",)


@pytest.mark.parametrize(
    "feature_hashes, expected_computed, expected_cached, expected_live_hash",
    [
        (["old-image", "new-image"], [], [0], "old-image"),
        (["new-image", "old-image"], [0], [], "new-image"),
    ],
)
def test_encoder_plan_commit_preserves_feature_order_across_lru_eviction(
    monkeypatch: pytest.MonkeyPatch,
    feature_hashes: list[str],
    expected_computed: list[int],
    expected_cached: list[int],
    expected_live_hash: str,
):
    """Read-only planning and commit must execute identical LRU operations."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_seqs=2,
        max_num_batched_tokens=256,
    )
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=50)
    seed = create_requests(
        num_requests=1,
        num_tokens=50,
        mm_hashes_list=[["old-image"]],
        mm_positions=[[PlaceholderRange(offset=0, length=50)]],
        req_ids=["seed"],
    )[0]
    request = create_requests(
        num_requests=1,
        num_tokens=100,
        mm_hashes_list=[feature_hashes],
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=50),
                PlaceholderRange(offset=50, length=50),
            ]
        ],
        req_ids=["candidate"],
    )[0]
    manager = scheduler.encoder_cache_manager
    manager.allocate(seed, 0)
    manager.free(seed)
    before = (
        {key: set(value) for key, value in manager.cached.items()},
        manager.freeable.copy(),
        manager.num_free_slots,
        manager.num_freeable_slots,
    )

    computed, num_tokens, _budget, external, cached = (
        scheduler._try_schedule_encoder_inputs(request, 0, 100, 100)
    )

    assert computed == expected_computed
    assert cached == expected_cached
    assert external == []
    assert num_tokens == 50
    assert (
        {key: set(value) for key, value in manager.cached.items()},
        manager.freeable.copy(),
        manager.num_free_slots,
        manager.num_freeable_slots,
    ) == before

    scheduler._commit_encoder_cache_plan(request, cached, computed, external)

    assert set(manager.cached) == {expected_live_hash}
    assert manager.cached[expected_live_hash] == {request.request_id}
    assert manager.num_free_slots == 0
    assert manager.num_freeable_slots == 0


def test_running_text_wave_preserves_liveness_with_skipped_waiting_work():
    scheduler = _make_elastic_running_text_wave_scheduler()
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.skipped_waiting = [Mock()]
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 8, 32, 4)
    assert prepared_maintenance is False
    scheduler._can_fund_elastic_graph_step.assert_called_once()


def test_running_waiting_joint_wave_prepares_one_cold_mixed_shape():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=31)
    waiting = Mock(
        request_id="replacement",
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        force_non_speculative=False,
        num_tokens=4,
        num_prompt_tokens=4,
        execution_prefill_len=4,
    )
    scheduler.waiting = [waiting]
    scheduler.requests[waiting.request_id] = waiting
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)

    def prepare_joint(key, **_kwargs):
        if _kwargs.get("preview_only"):
            return True, 96, 128
        _arm_fixture_capture(scheduler, key)
        return False, 96, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=prepare_joint)

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (0, 3, 32, 128, 0)
    assert prepared_maintenance is True
    assert scheduler._elastic_preflight_joint_waiting_request_ids == (
        waiting.request_id,
    )
    assert scheduler._can_fund_elastic_graph_step.call_count == 2
    assert (
        scheduler._can_fund_elastic_graph_step.call_args.kwargs["allow_maintenance"]
        is True
    )


def test_failed_joint_maintenance_reservation_retries_running_wave():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=8)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    waiting = Mock(
        request_id="cold-prefill",
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        force_non_speculative=False,
        num_tokens=128,
        num_prompt_tokens=128,
        execution_prefill_len=128,
    )
    scheduler.waiting = [waiting]
    scheduler.requests[waiting.request_id] = waiting
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler.kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = (  # noqa: E501
        KVCacheBlockPoolRequirements(primary=7)
    )
    running_key = (0, 3, 8, 32, 4)

    def price_wave(key, **_kwargs):
        if _kwargs.get("preview_only"):
            return True, 96, 128
        if key != running_key:
            _arm_fixture_capture(scheduler, key)
            return False, 96, 128
        return True, 64, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=price_wave)
    scheduler._reserve_elastic_admission.side_effect = (False, True)
    scheduler._prepare_elastic_waiting_deficit_reclaim = Mock(return_value=False)

    assert scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    ) == (running_key, False)
    assert scheduler._reserve_elastic_admission.call_count == 2
    assert scheduler._reserve_elastic_admission.call_args.args[0] == running_key
    assert _pending_elastic_maintenance(scheduler) is None
    assert scheduler._elastic_preflight_joint_waiting_request_ids == ()


def test_failed_prefix_joint_variants_retry_running_without_stale_identity():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=8)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    waiting = Mock(
        request_id="prefix-prefill",
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        force_non_speculative=False,
        num_tokens=128,
        num_prompt_tokens=128,
        execution_prefill_len=128,
    )
    scheduler.waiting = [waiting]
    scheduler.requests[waiting.request_id] = waiting
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 64, 0)
    scheduler.kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = (  # noqa: E501
        KVCacheBlockPoolRequirements(primary=7)
    )
    running_key = (0, 3, 8, 32, 4)
    attempted_keys: list[tuple[int, ...]] = []

    def price_wave(key, **_kwargs):
        attempted_keys.append(key)
        if key != running_key:
            return False, 96, 128
        return True, 64, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=price_wave)
    scheduler._reserve_elastic_admission.return_value = True
    scheduler._prepare_elastic_waiting_deficit_reclaim = Mock(return_value=False)

    assert scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    ) == (running_key, False)
    assert attempted_keys == [
        (0, 3, 9, 128, 0),
        (0, 3, 9, 256, 0),  # One cold retry before deferring the arrival.
        running_key,
    ]
    scheduler.kv_cache_manager.get_computed_blocks.assert_called_once_with(waiting)
    scheduler.kv_cache_manager.lease_computed_blocks.return_value.release.assert_called_once()
    assert scheduler._elastic_prefix_hits == {}
    assert scheduler._elastic_preflight_joint_waiting_request_ids == ()
    assert scheduler._elastic_preflight_waiting_ignore_prefix_request_ids == ()
    assert _pending_elastic_maintenance(scheduler) is None
    assert scheduler.waiting == [waiting]


def test_running_waiting_joint_wave_excludes_full_isl_overcapacity():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=1)
    scheduler.scheduler_reserve_full_isl = True
    waiting = Mock(
        request_id="over-cap",
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        force_non_speculative=False,
        num_tokens=4,
        num_prompt_tokens=4,
        execution_prefill_len=4,
    )
    scheduler.waiting = [waiting]
    scheduler.requests[waiting.request_id] = waiting
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler.kv_cache_manager.watermark_blocks = 0
    scheduler._elastic_remaining_resource_requirements.side_effect = (
        lambda request_ids: KVCacheBlockPoolRequirements(
            primary=40 if len(request_ids) == 1 else 120
        )
    )
    scheduler.kv_cache_manager.coordinator.can_allocate.side_effect = (
        lambda requirements, **_kwargs: requirements.primary <= 100
    )
    scheduler.kv_cache_manager.coordinator.max_elastic_external_memory.return_value = 99
    scheduler._elastic_graph_catalog_coverage["pinned_full_bytes"] = 100
    scheduler._elastic_pressure_floor_external_bytes = Mock(return_value=100)
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 1, 4, 4)
    assert prepared_maintenance is False
    assert scheduler._elastic_preflight_joint_waiting_request_ids == ()
    scheduler._can_fund_elastic_graph_step.assert_called_once()


@pytest.mark.parametrize("optional_pinned_bytes", [50, 150])
def test_running_waiting_joint_wave_reaches_graph_plan_after_idle_reclaim(
    optional_pinned_bytes,
):
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=1)
    scheduler.scheduler_reserve_full_isl = True
    waiting = Mock(
        request_id="reclaim-fit",
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        force_non_speculative=False,
        num_tokens=4,
        num_prompt_tokens=4,
        execution_prefill_len=4,
    )
    scheduler.waiting = [waiting]
    scheduler.requests[waiting.request_id] = waiting
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler.kv_cache_manager.watermark_blocks = 0
    scheduler._elastic_remaining_resource_requirements.return_value = (
        KVCacheBlockPoolRequirements(primary=120)
    )
    scheduler.kv_cache_manager.coordinator.can_allocate.return_value = False
    scheduler.kv_cache_manager.coordinator.max_elastic_external_memory.return_value = (
        100
    )
    scheduler._elastic_graph_catalog_coverage["pinned_full_bytes"] = (
        optional_pinned_bytes
    )
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (0, 3, 2, 8, 0)
    assert prepared_maintenance is False
    assert scheduler._elastic_preflight_joint_waiting_request_ids == (
        waiting.request_id,
    )
    assert scheduler._can_fund_elastic_graph_step.call_count == 2
    assert (
        scheduler._can_fund_elastic_graph_step.call_args_list[0].kwargs["preview_only"]
        is True
    )


@pytest.mark.parametrize("running_fits", [False, True])
def test_running_x1_waiting_x38_preflight_requests_one_pressure_reclaim(running_fits):
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=1)
    scheduler.max_num_running_reqs = 39
    waiting = [
        Mock(
            request_id=f"tail-{index}",
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            num_stale_output_tokens=0,
            has_encoder_inputs=False,
            force_non_speculative=False,
            num_tokens=4,
            num_prompt_tokens=4,
            execution_prefill_len=4,
        )
        for index in range(38)
    ]
    scheduler.waiting = waiting
    scheduler.requests.update({request.request_id: request for request in waiting})
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler.kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = (  # noqa: E501
        KVCacheBlockPoolRequirements(primary=7)
    )
    running_key = scheduler._canonical_elastic_graph_step_key({"decode-0": 4}, 3, True)
    scheduler._can_fund_elastic_graph_step = Mock(
        side_effect=lambda key, **_kwargs: (
            (True, 64, 128)
            if running_fits and key == running_key
            else (False, 200, 128)
        )
    )
    scheduler._prepare_elastic_waiting_deficit_reclaim = Mock(return_value=True)

    assert scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
        physical_quiescent=True,
    ) == (None, True)
    scheduler._prepare_elastic_waiting_deficit_reclaim.assert_called_once_with(
        (7,) * 38,
        physical_quiescent=True,
    )


@pytest.mark.parametrize(
    "blocker", [None, "inflight", "capacity", "unknown", "lease", "loan"]
)
def test_running_hot_waiting_cold_graph_reclaim_without_kv_deficit(blocker):
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=1)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    controller = scheduler._elastic_admission_controller
    source = (0, 3, 1, 4, 4)
    destination = (0, 3, 2, 8, 0)
    scheduler._elastic_graph_carrier_step_key = source
    terminal = scheduler._resolve_elastic_step_physical_keys((1, 3, 40, 40, 1))[-1]
    scheduler._elastic_serving_carrier_keys = (terminal,)
    controller.publish_hot(terminal, GraphPrice(20, 20, "terminal"), pinned=True)
    for key in scheduler._resolve_elastic_step_physical_keys(source):
        controller.publish_hot(key, GraphPrice(30, 30, key.identity), pinned=True)
    controller.resident_bytes = 120
    controller.floor_bytes = 10
    controller.transition_floor_bytes = 10
    controller.record_capture_envelope(
        scheduler._elastic_graph_owner_key(destination), (100, 10, 0)
    )
    scheduler._elastic_graph_catalog = {}
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.elastic_external_memory_bytes = 120
    coordinator.max_elastic_external_memory.return_value = 150
    coordinator.normalize_elastic_external_memory.side_effect = lambda n: n
    request = Mock(
        request_id="new-work",
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        force_non_speculative=False,
        num_tokens=4,
        num_prompt_tokens=4,
        execution_prefill_len=4,
    )
    scheduler.waiting = [request]
    scheduler.requests[request.request_id] = request
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler.kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = KVCacheBlockPoolRequirements(  # noqa: E501
        primary=7
    )
    scheduler._prepare_elastic_waiting_deficit_reclaim = Mock(return_value=False)
    # Preview must not depend on a previous wave's mutable diagnostic reason.
    scheduler._elastic_last_defer_reason = None
    if blocker == "capacity":
        coordinator.max_elastic_external_memory.return_value = 129
    elif blocker == "unknown":
        controller.capture_envelopes.clear()
    elif blocker == "lease":
        user = controller.plan(
            "inflight", tuple(controller.entries), request_bytes=0, available_bytes=1000
        )
        controller.commit_user(user)
    elif blocker == "loan":
        controller.reserve_loan(source, 120)
    entries = dict(controller.entries)
    key, maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
        physical_quiescent=blocker != "inflight",
    )
    assert dict(controller.entries) == entries
    coordinator.set_elastic_external_memory.assert_not_called()
    if blocker is not None:
        assert key == source and not maintenance
        assert controller.pending_maintenance_plan is None
        return
    assert key is None and maintenance
    plan = controller.pending_maintenance_plan
    assert plan is not None and plan.kind == ElasticPlanKind.PRESSURE_RECLAIM
    assert plan.protected_keys == (terminal,)
    scheduler._reserve_elastic_admission.assert_not_called()


def test_waiting_preflight_selects_budget_prefix_before_considering_reclaim():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=0)
    waiting = [
        Mock(
            request_id=f"waiting-{index}",
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            num_stale_output_tokens=0,
            has_encoder_inputs=False,
            num_tokens=4,
            num_prompt_tokens=4,
            execution_prefill_len=4,
        )
        for index in range(3)
    ]
    scheduler.waiting = waiting
    scheduler.requests.update({request.request_id: request for request in waiting})
    scheduler.kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = (  # noqa: E501
        KVCacheBlockPoolRequirements(primary=7)
    )
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))
    scheduler._reserve_elastic_admission.return_value = True
    scheduler._prepare_elastic_waiting_deficit_reclaim = Mock(return_value=True)

    key, prepared, request_ids = scheduler._preflight_elastic_waiting_text_wave(
        token_budget=8,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )
    assert key is not None and key[2] == 2
    assert prepared is False
    assert request_ids == ("waiting-0", "waiting-1")
    scheduler._prepare_elastic_waiting_deficit_reclaim.assert_not_called()


def test_waiting_preflight_does_not_chunk_when_chunked_prefill_is_disabled():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=0)
    scheduler.scheduler_config.enable_chunked_prefill = False
    request = Mock(
        request_id="oversized",
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        num_tokens=16,
        num_prompt_tokens=16,
        execution_prefill_len=16,
    )
    scheduler.waiting = [request]
    scheduler.requests[request.request_id] = request
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler._can_fund_elastic_graph_step = Mock()

    assert scheduler._preflight_elastic_waiting_text_wave(
        token_budget=8,
        prefill_chunk_cap=0,
        defer_prefills=False,
    ) == (None, False, ())
    scheduler._can_fund_elastic_graph_step.assert_not_called()


def test_waiting_preflight_rejects_zero_aligned_candidate_before_x0_pricing():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=0)
    scheduler.need_mamba_block_aligned_split = True
    request = Mock(
        request_id="unaligned",
        prefix_cache_hint_tokens=0,
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        num_tokens=4,
        num_prompt_tokens=4,
        execution_prefill_len=4,
    )
    scheduler.waiting = [request]
    scheduler.requests[request.request_id] = request
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler._mamba_block_aligned_split = Mock(return_value=0)
    scheduler._can_fund_elastic_graph_step = Mock()

    assert scheduler._preflight_elastic_waiting_text_wave(
        token_budget=8,
        prefill_chunk_cap=0,
        defer_prefills=False,
    ) == (None, False, ())
    scheduler._can_fund_elastic_graph_step.assert_not_called()


def test_waiting_reclaim_cannot_invalidate_deferred_mm_binding():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=0)
    scheduler._elastic_admission_controller.resident_bytes = 128
    scheduler._elastic_deferred_mm_wave = Mock()
    scheduler.kv_cache_manager.coordinator.plan_elastic_admission_wave = Mock()

    assert not scheduler._prepare_elastic_waiting_deficit_reclaim((7, 7, 7))
    scheduler.kv_cache_manager.coordinator.plan_elastic_admission_wave.assert_not_called()


@pytest.mark.parametrize("queue_name", ["waiting", "skipped_waiting"])
def test_running_x31_and_waiting_x1_commit_one_hot_joint_wave(
    monkeypatch: pytest.MonkeyPatch,
    queue_name: str,
):
    """A queued retry fills fixed X32 residency without a prefix transition."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=40,
        max_num_batched_tokens=4096,
        num_speculative_tokens=3,
    )
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    original_plan = scheduler._elastic_admission_controller.plan

    def keep_fixture_owners_hot(transaction_id, physical_keys, **kwargs):
        for physical_key in physical_keys:
            entry = scheduler._elastic_admission_controller.entries.get(physical_key)
            if entry is None or not entry.hot:
                scheduler._elastic_admission_controller.publish_hot(
                    physical_key,
                    GraphPrice(1, 1, f"test:{physical_key.identity}"),
                    pinned=False,
                )
        return original_plan(transaction_id, physical_keys, **kwargs)

    monkeypatch.setattr(
        scheduler._elastic_admission_controller,
        "plan",
        keep_fixture_owners_hot,
    )
    scheduler.elastic_on_demand_graphs = False
    requests = create_requests(
        num_requests=32,
        num_tokens=4,
        max_tokens=1280,
        ignore_eos=True,
    )
    running, queued = requests[:31], requests[31]
    for request in running:
        scheduler.add_request(request)

    prefill = scheduler.schedule()
    req_ids = [request.request_id for request in running]
    scheduler.update_from_output(
        prefill,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[[0] for _ in running],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.update_draft_token_ids(
        DraftTokenIds(req_ids, [[11, 12, 13] for _ in running])
    )
    scheduler.add_request(queued)
    if queue_name == "skipped_waiting":
        scheduler.waiting.remove_requests([queued])
        scheduler.skipped_waiting.add_request(queued)

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    # The 31 q4 decode rows contribute 124 tokens and the newly admitted
    # four-token prompt contributes the final four.  The joint wave is mixed,
    # so it has no uniform q4 marker even though it exactly fills M128/X32.
    expected_key = (0, 3, 32, 128, 0)
    physical_keys = scheduler._resolve_elastic_step_physical_keys(expected_key)
    for physical_key in physical_keys:
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(1, 1, f"test:{physical_key.identity}"),
            pinned=False,
        )

    can_fund = Mock(return_value=(True, 1, 1_000_000))
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)
    monkeypatch.setattr(
        scheduler, "_reserve_elastic_admission", Mock(return_value=True)
    )
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens == 31 * 4 + 4
    assert output.num_scheduled_tokens == {
        request.request_id: 4 for request in requests
    }
    assert output.is_pure_decode_step is False
    assert output.num_spec_tokens_to_schedule == 3
    assert can_fund.call_count == 2
    assert (
        sum(bool(call.kwargs.get("preview_only")) for call in can_fund.call_args_list)
        == 1
    )
    assert all(call.args[0] == expected_key for call in can_fund.call_args_list)
    assert can_fund.call_args_list[0].kwargs["preview_only"] is True
    assert "preview_only" not in can_fund.call_args_list[1].kwargs
    assert all(
        scheduler._elastic_admission_controller.entries[key].hot
        for key in physical_keys
    )
    assert queued.request_id in {
        request_data.req_id for request_data in output.scheduled_new_reqs
    }
    assert queued.status == RequestStatus.RUNNING
    assert queued in scheduler.running
    assert not scheduler.waiting
    assert not scheduler.skipped_waiting


def test_running_text_wave_preflights_partial_prefills_as_one_final_shape():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=2)
    for request in scheduler.running:
        request.is_prefill_chunk = True
        request.num_computed_tokens = 128
        request.num_prompt_tokens = 1024
        request.execution_prefill_len = 1024
        request.num_tokens = 1024
        request.num_tokens_with_spec = 384
        request.spec_token_ids = []
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (0, 3, 2, 512, 0)
    assert prepared_maintenance is False
    scheduler._can_fund_elastic_graph_step.assert_called_once_with(
        key,
        minimum_free_primary_blocks=0,
        gdn_blocks=0,
        allow_maintenance=True,
    )


def test_running_text_wave_preflights_mixed_rows_as_one_final_shape():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=2)
    partial = scheduler.running[1]
    partial.is_prefill_chunk = True
    partial.num_computed_tokens = 128
    partial.num_prompt_tokens = 1024
    partial.execution_prefill_len = 1024
    partial.num_tokens = 1024
    partial.num_tokens_with_spec = 384
    partial.spec_token_ids = []
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (0, 3, 2, 512, 0)
    assert prepared_maintenance is False
    scheduler._can_fund_elastic_graph_step.assert_called_once_with(
        key,
        minimum_free_primary_blocks=0,
        gdn_blocks=0,
        allow_maintenance=True,
    )


def test_running_text_wave_skips_deferred_partial_prefills():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=2)
    for request in scheduler.running:
        request.is_prefill_chunk = True
    scheduler._can_fund_elastic_graph_step = Mock()

    assert scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=True,
    ) == (None, False)
    scheduler._can_fund_elastic_graph_step.assert_not_called()


def test_running_wave_prices_encoder_work_without_mutation():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=2)
    scheduler.running[0].has_encoder_inputs = True
    scheduler.max_num_encoder_input_tokens = 128
    scheduler._elastic_mm_activation_loan_bytes = 32
    scheduler._try_schedule_encoder_inputs = Mock(return_value=([0], 4, 64, [], []))
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 2, 8, 4)
    assert prepared_maintenance is False
    scheduler._try_schedule_encoder_inputs.assert_called_once()
    assert (
        scheduler._can_fund_elastic_graph_step.call_args.kwargs[
            "mm_activation_loan_bytes"
        ]
        == 32
    )


def test_running_wave_commits_encoder_intent_only_after_row_survives_alignment():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=2)
    scheduler.max_num_encoder_input_tokens = 128
    scheduler._elastic_mm_activation_loan_bytes = 32
    scheduler.need_mamba_block_aligned_split = True
    for request in scheduler.running:
        request.has_encoder_inputs = True

    observed_encoder_budgets: list[int] = []

    def plan_encoder(request, _start, num_tokens, budget, **_kwargs):
        observed_encoder_budgets.append(budget)
        input_index = int(request.request_id.rsplit("-", 1)[1])
        return [input_index], num_tokens, budget - 16, [], []

    scheduler._try_schedule_encoder_inputs = Mock(side_effect=plan_encoder)
    scheduler._mamba_block_aligned_split = Mock(
        side_effect=lambda request, num_tokens, *_args: (
            0 if request.request_id == "decode-0" else num_tokens
        )
    )
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (1, 3, 1, 4, 4)
    assert prepared_maintenance is False
    assert observed_encoder_budgets == [128, 128]
    scheduler._can_fund_elastic_graph_step.assert_called_once()
    assert (
        scheduler._can_fund_elastic_graph_step.call_args.kwargs[
            "mm_activation_loan_bytes"
        ]
        == 32
    )


def test_mixed_running_waiting_mm_wave_arms_one_exact_graph_only_capture():
    scheduler = _make_elastic_running_text_wave_scheduler(num_reqs=2)
    scheduler.max_num_encoder_input_tokens = 128
    scheduler._elastic_mm_activation_loan_bytes = 32
    waiting = []
    for index in range(2):
        request = Mock(
            request_id=f"waiting-{index}",
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            num_stale_output_tokens=0,
            has_encoder_inputs=index == 0,
            num_tokens=2,
            num_prompt_tokens=2,
            execution_prefill_len=2,
            is_prefill_chunk=False,
            force_non_speculative=False,
        )
        request.is_finished.return_value = False
        waiting.append(request)
        scheduler.requests[request.request_id] = request
    scheduler.waiting = waiting
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)

    def plan_encoder(request, _start, num_tokens, budget, **_kwargs):
        encoder_inputs = [0] if request.has_encoder_inputs else []
        return encoder_inputs, num_tokens, budget - len(encoder_inputs), [], []

    scheduler._try_schedule_encoder_inputs = Mock(side_effect=plan_encoder)

    def prepare_graph_only(step_key, **_kwargs):
        if _kwargs.get("preview_only"):
            return True, 96, 128
        _arm_fixture_capture(
            scheduler,
            step_key,
            maintenance_execution=ElasticMaintenanceExecution.GRAPH_ONLY,
        )
        return False, 96, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=prepare_graph_only)

    key, prepared_maintenance = scheduler._preflight_elastic_running_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (0, 3, 4, 16, 0)
    assert prepared_maintenance is True
    wave = scheduler._elastic_deferred_mm_wave
    assert wave is not None
    assert wave.running_request_ids == ("decode-0", "decode-1")
    assert wave.waiting_request_ids == ("waiting-0", "waiting-1")
    assert wave.scheduled_encoder_inputs == (("waiting-0", (0,)),)
    assert scheduler._elastic_preflight_joint_waiting_request_ids == (
        "waiting-0",
        "waiting-1",
    )
    assert (
        scheduler._can_fund_elastic_graph_step.call_args.kwargs[
            "mm_activation_loan_bytes"
        ]
        == 32
    )


def _make_elastic_waiting_text_wave_scheduler(num_reqs: int = 8) -> Scheduler:
    scheduler = _new_elastic_scheduler()
    scheduler.num_prefill_lookahead = 1
    scheduler.elastic_on_demand_graphs = True
    _clear_elastic_maintenance(scheduler)
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler.running = []
    scheduler.skipped_waiting = ()
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.lora_config = None
    scheduler.max_num_running_reqs = 40
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.need_mamba_block_aligned_split = False
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.use_eagle = False
    scheduler.use_eagle_block_drop = False
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_prefill_checkpoint_alignment = None
    scheduler.mamba_fine_grained_prefix_cache = False
    scheduler.scheduler_config = Mock(
        long_prefill_token_threshold=0,
        max_num_batched_tokens=4096,
    )
    scheduler._elastic_graph_catalog_coverage = {
        "mtp_verifier_contract": "batched-causal-q1-v1",
        "decode_max_x": 40,
    }
    scheduler.vllm_config = Mock(
        speculative_config=Mock(disable_speculation_on_non_decode=False)
    )
    scheduler.kv_cache_manager = Mock(enable_kv_cache_events=False)
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    scheduler._elastic_remaining_resource_requirements = Mock(
        return_value=KVCacheBlockPoolRequirements()
    )
    scheduler._reserve_elastic_admission = Mock(return_value=True)
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.elastic_gdn_blocks_after_allocation.return_value = 0
    scheduler.waiting = []
    scheduler.requests = {}
    scheduler._elastic_deferred_mm_wave = None
    for index in range(num_reqs):
        request = Mock(
            request_id=f"prefill-{index}",
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            num_stale_output_tokens=0,
            has_encoder_inputs=False,
            num_tokens=2,
            num_prompt_tokens=2,
            execution_prefill_len=2,
            is_prefill_chunk=False,
            force_non_speculative=False,
        )
        scheduler.waiting.append(request)
        scheduler.requests[request.request_id] = request
    scheduler._elastic_successor_primary_headroom = Mock(return_value=8)
    return scheduler


def test_waiting_text_wave_preflights_only_final_assembled_shape():
    scheduler = _make_elastic_waiting_text_wave_scheduler()
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance, target = scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert key == (0, 3, 8, 16, 0)
    assert prepared_maintenance is False
    assert target == tuple(f"prefill-{index}" for index in range(8))
    assert scheduler._can_fund_elastic_graph_step.call_count == 2
    assert scheduler._can_fund_elastic_graph_step.call_args_list[0].kwargs[
        "preview_only"
    ]
    assert scheduler._can_fund_elastic_graph_step.call_args_list[1].args == (key,)
    assert "preview_only" not in (
        scheduler._can_fund_elastic_graph_step.call_args_list[1].kwargs
    )


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("cold", [False, True])
def test_elastic_prefix_lease_binds_hot_and_cold_wave(mixed, cold):
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, PrefixCacheLease

    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=2)
    if mixed:
        running = _make_elastic_running_text_wave_scheduler(num_reqs=1)
        scheduler.running = running.running
        scheduler.requests.update(running.requests)
        scheduler.num_sampled_tokens_per_step = 1
        scheduler.current_step = 7
        scheduler.max_model_len = 262144
        scheduler.canonical_prefill_admission = False
    pool = BlockPool(5, True, 16)
    blocks = pool.get_new_blocks(1)
    pool.free_blocks(blocks)
    hit = KVCacheBlocks((tuple(blocks),))
    for request in scheduler.waiting:
        request.num_tokens = request.num_prompt_tokens = 128
        request.execution_prefill_len = 128
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (hit, 64, 0)
    scheduler.kv_cache_manager.lease_computed_blocks.side_effect = lambda value: (
        PrefixCacheLease([(pool, value.blocks[0])])
    )

    def fund(key, **kwargs):
        if cold and not kwargs.get("preview_only"):
            _arm_fixture_capture(scheduler, key)
            return False, 96, 128
        return True, 64, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=fund)
    preflight = (
        scheduler._preflight_elastic_running_text_wave
        if mixed
        else scheduler._preflight_elastic_waiting_text_wave
    )
    result = preflight(token_budget=4096, prefill_chunk_cap=0, defer_prefills=False)
    assert result[0][3] == (256 if mixed else 128)
    assert result[1] is cold
    assert scheduler._elastic_preflight_waiting_ignore_prefix_request_ids == ()
    assert blocks[0].ref_cnt == 2
    # Commit must consume exactly the held result even if the lookup source changes.
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    for request in scheduler.waiting:
        assert scheduler._get_local_prefix_cache_hit(request) == (hit, 64, 0, False)
    assert scheduler.kv_cache_manager.get_computed_blocks.call_count == 2
    scheduler._release_elastic_prefix_hits()
    assert blocks[0].ref_cnt == 0


def test_elastic_prefix_schedule_exception_releases_all_leases():
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, PrefixCacheLease

    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=1)
    pool = BlockPool(4, True, 16)
    blocks = pool.get_new_blocks(1)
    pool.free_blocks(blocks)
    hit = KVCacheBlocks((tuple(blocks),))
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (hit, 1, 0)
    scheduler.kv_cache_manager.lease_computed_blocks.side_effect = lambda value: (
        PrefixCacheLease([(pool, value.blocks[0])])
    )

    def fail(*args, **kwargs):
        scheduler._lease_local_prefix_cache_hit(scheduler.waiting[0])
        assert blocks[0].ref_cnt == 1
        raise RuntimeError("failed allocation")

    scheduler._schedule_with_prefix_leases = fail
    with pytest.raises(RuntimeError, match="failed allocation"):
        scheduler.schedule()
    assert blocks[0].ref_cnt == 0
    assert scheduler._elastic_prefix_hits == {}


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "num_cached,shared,hint", [(0, 384, 0), (192, 576, 0), (0, 0, 384), (0, 0, 0)]
)
def test_elastic_prefix_preflight_uses_commit_checkpoint_boundaries(
    mixed, num_cached, shared, hint
):
    """A detected junction is relevant even when no KV prefix is reusable."""
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=1)
    if mixed:
        running = _make_elastic_running_text_wave_scheduler(num_reqs=1)
        scheduler.running = running.running
        scheduler.requests.update(running.requests)
        scheduler.num_sampled_tokens_per_step = 1
        scheduler.current_step = 7
        scheduler.max_model_len = 262144
        scheduler.canonical_prefill_admission = False
    scheduler.need_mamba_block_aligned_split = True
    scheduler.block_size = 7488
    scheduler.hash_block_size = 192
    scheduler.use_eagle = True
    scheduler.use_eagle_block_drop = True
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_prefill_checkpoint_alignment = None
    scheduler.mamba_fine_grained_prefix_cache = False
    scheduler.mamba_partial_cache_hit = True
    scheduler.mamba_state_update_alignment = 64
    scheduler.max_num_scheduled_tokens = 4096
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.__class__ = HybridKVCacheCoordinator
    coordinator.enable_dcp_fine_prefix = True
    coordinator.gdn_checkpoint_keys = {}
    request = scheduler.waiting[0]
    request.num_tokens = request.num_prompt_tokens = request.execution_prefill_len = (
        1200
    )
    request.shared_prefix_boundary = 960  # stale state must not decide this wave
    request.prefix_cache_hint_tokens = hint
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (
        Mock(),
        num_cached,
        shared,
    )
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))
    preflight = (
        scheduler._preflight_elastic_running_text_wave
        if mixed
        else scheduler._preflight_elastic_waiting_text_wave
    )
    result = preflight(token_budget=4096, prefill_chunk_cap=1024, defer_prefills=False)
    key, maintenance = result[:2]
    assert not maintenance
    assert request.shared_prefix_boundary == 960  # preview must not mutate request
    request.shared_prefix_boundary = max(shared, hint)
    committed = scheduler._mamba_block_aligned_split(request, 1024, num_cached, 0)
    expected_rows = committed + (4 if mixed else 0)
    assert key[3] == 1 << (expected_rows - 1).bit_length()
    scheduler._release_elastic_prefix_hits()
    assert scheduler._elastic_prefix_hits == {}

    # A later cold retry must discard the former shared junction without
    # mutating the still-stale request during preflight.
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (Mock(), 0, 0)
    request.prefix_cache_hint_tokens = 0
    stale_boundary = request.shared_prefix_boundary
    cold = preflight(token_budget=4096, prefill_chunk_cap=1024, defer_prefills=False)
    # MTP's prompt-tail checkpoint clips 1024 to 960; +4 decode rows
    # still fit the same M1024 bucket.
    assert cold[0][3] == 1024
    assert request.shared_prefix_boundary == stale_boundary
    scheduler._release_elastic_prefix_hits()


def test_elastic_prefix_boundary_rejects_unaligned_hint_without_mutation():
    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=1)
    scheduler.hash_block_size = 192
    request = scheduler.waiting[0]
    request.prefix_cache_hint_tokens = 193
    request.shared_prefix_boundary = 384
    with pytest.raises(RuntimeError, match="hint is not hash aligned"):
        scheduler._resolved_shared_prefix_boundary(request, 576)
    assert request.shared_prefix_boundary == 384


def test_elastic_prefix_smaller_wave_releases_excluded_waiting_before_pricing():
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, PrefixCacheLease

    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=2)
    # Isolate suffix ownership; cold pressure replanning has its own control.
    scheduler._drop_elastic_prefix_hits_for_pressure = Mock(return_value=False)
    pool = BlockPool(5, True, 16)
    blocks = pool.get_new_blocks(2)
    pool.free_blocks(blocks)
    hits = [KVCacheBlocks(((block,),)) for block in blocks]
    for request in scheduler.waiting:
        request.num_tokens = request.num_prompt_tokens = 128
        request.execution_prefill_len = 128
    scheduler.kv_cache_manager.get_computed_blocks.side_effect = [
        (hit, 64, 0) for hit in hits
    ]
    scheduler.kv_cache_manager.lease_computed_blocks.side_effect = lambda value: (
        PrefixCacheLease([(pool, value.blocks[0])])
    )

    def fund(key, **kwargs):
        assert [block.ref_cnt for block in blocks] == (
            [1, 1] if key[2] == 2 else [1, 0]
        )
        return key[2] == 1, 64, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=fund)
    key, maintenance, selected = scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )
    assert key[2] == 1 and not maintenance
    assert selected == (scheduler.waiting[0].request_id,)
    assert len(scheduler.waiting) == 2
    # Full-sequence demand counts pinned prefix storage once, not as new blocks.
    scheduler.max_model_len = 262144
    request = scheduler.waiting[0]
    Scheduler._request_remaining_blocks(scheduler, request)
    coordinator = scheduler.kv_cache_manager.coordinator
    priced = coordinator.get_block_pool_requirements.call_args.kwargs
    assert priced["new_computed_blocks"] == hits[0].blocks
    assert priced["num_local_computed_tokens"] == 64
    assert request.num_computed_tokens == 0
    scheduler._release_elastic_prefix_hits()
    assert pool.get_num_free_blocks() == 4


@pytest.mark.parametrize("mixed", [False, True])
def test_elastic_prefix_pressure_replans_once_without_reducing_wave(mixed):
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, PrefixCacheLease

    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=2)
    if mixed:
        running = _make_elastic_running_text_wave_scheduler(num_reqs=1)
        scheduler.running = running.running
        scheduler.requests.update(running.requests)
        scheduler.num_sampled_tokens_per_step = 1
        scheduler.current_step = 7
        scheduler.max_model_len = 262144
        scheduler.canonical_prefill_admission = False
    pool = BlockPool(5, True, 16)
    allocated = pool.get_new_blocks(4)
    tail = allocated[-1:]
    pool.free_blocks(allocated)
    hit = KVCacheBlocks((tuple(tail),))
    for request in scheduler.waiting:
        request.num_tokens = request.num_prompt_tokens = 128
        request.execution_prefill_len = 128
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (hit, 64, 0)
    scheduler.kv_cache_manager.lease_computed_blocks.side_effect = lambda value: (
        PrefixCacheLease([(pool, value.blocks[0])])
    )
    refs = []

    def fund(key, **kwargs):
        refs.append(tail[0].ref_cnt)
        # A Graph loan needs the tail page; referenced cache cannot be evicted.
        return tail[0].ref_cnt == 0, 64, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=fund)
    preflight = (
        scheduler._preflight_elastic_running_text_wave
        if mixed
        else scheduler._preflight_elastic_waiting_text_wave
    )
    result = preflight(token_budget=4096, prefill_chunk_cap=0, defer_prefills=False)
    assert result[0] is not None and not result[1]
    assert result[0][2] == (3 if mixed else 2)
    assert refs[0] == 2 and all(ref == 0 for ref in refs[1:])
    assert scheduler.kv_cache_manager.get_computed_blocks.call_count == 2
    assert not scheduler._drop_elastic_prefix_hits_for_pressure()
    for request in scheduler.waiting:
        assert scheduler._get_local_prefix_cache_hit(request)[1] == 0
    scheduler._release_elastic_prefix_hits()
    assert pool.deactivate_tail_blocks(4)


@pytest.mark.parametrize("prompt_tokens,checkpoints", [(7265, []), (16065, [7488])])
def test_coarse_dcp_prefix_requires_effective_block_checkpoint(
    prompt_tokens, checkpoints
):
    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=1)
    scheduler.block_size = 2496 * 3
    scheduler.hash_block_size = 192
    scheduler.use_eagle = True
    scheduler.use_eagle_block_drop = True
    scheduler.mamba_has_prefill_checkpoint_blocks = False
    scheduler.mamba_prefill_checkpoint_alignment = None
    scheduler.mamba_fine_grained_prefix_cache = False
    scheduler.mamba_partial_cache_hit = False
    scheduler.mamba_state_update_alignment = 64
    scheduler.max_num_scheduled_tokens = 4096
    request = scheduler.waiting[0]
    request.execution_prefill_len = request.num_prompt_tokens = prompt_tokens
    request.num_tokens = prompt_tokens
    request.shared_prefix_boundary = 0
    request.prefix_cache_hint_tokens = 0
    boundaries = []
    while request.num_computed_tokens < prompt_tokens:
        chunk = Scheduler._mamba_block_aligned_split(
            scheduler, request, min(4096, prompt_tokens - request.num_computed_tokens)
        )
        assert chunk > 0
        request.num_computed_tokens += chunk
        if request.num_computed_tokens % scheduler.block_size == 0:
            boundaries.append(request.num_computed_tokens)
    assert boundaries == checkpoints


def test_waiting_text_wave_prepares_one_final_maintenance_not_prefixes():
    scheduler = _make_elastic_waiting_text_wave_scheduler()

    def prepare_final(key, **_kwargs):
        if _kwargs.get("preview_only"):
            return True, 96, 128
        _arm_fixture_capture(scheduler, key)
        return False, 96, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=prepare_final)
    key, prepared_maintenance, target = scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert (key, prepared_maintenance, target) == (
        (0, 3, 8, 16, 0),
        True,
        tuple(f"prefill-{index}" for index in range(8)),
    )
    assert scheduler._can_fund_elastic_graph_step.call_count == 2


def _make_deferred_mm_waiting_wave(
    *, num_reqs: int = 2
) -> tuple[Scheduler, dict[str, bool]]:
    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=num_reqs)
    scheduler.max_num_encoder_input_tokens = 128
    scheduler._elastic_mm_activation_loan_bytes = 32
    scheduler.waiting[0].has_encoder_inputs = True
    for request in scheduler.waiting:
        request.is_finished.return_value = False

    encoder_cache_state = {scheduler.waiting[0].request_id: False}

    def plan_encoder(request, _start, num_tokens, budget, **_kwargs):
        encoder_inputs = (
            []
            if not request.has_encoder_inputs
            or encoder_cache_state.get(request.request_id, False)
            else [0]
        )
        return encoder_inputs, num_tokens, budget - len(encoder_inputs), [], []

    scheduler._try_schedule_encoder_inputs = Mock(side_effect=plan_encoder)
    live_fund_calls = 0

    def fund_wave(step_key, **_kwargs):
        nonlocal live_fund_calls
        if _kwargs.get("preview_only"):
            return True, 96, 128
        live_fund_calls += 1
        if live_fund_calls == 1:
            _arm_fixture_capture(
                scheduler,
                step_key,
                maintenance_execution=ElasticMaintenanceExecution.GRAPH_ONLY,
            )
            return False, 96, 128
        return True, 96, 128

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=fund_wave)
    return scheduler, encoder_cache_state


def _settle_deferred_mm_fixture_capture(scheduler: Scheduler) -> tuple[Any, ...]:
    controller = scheduler._elastic_admission_controller
    plan = controller.pending_maintenance_plan
    assert plan is not None
    assert plan.maintenance_execution == ElasticMaintenanceExecution.GRAPH_ONLY
    controller.clear_maintenance()
    for key in plan.cold_misses:
        price = controller.entries[key].price
        assert price is not None
        controller.publish_hot(
            key,
            price,
            pinned=False,
        )
    return plan.cold_misses


def test_deferred_mm_wave_tick_b_ignores_new_arrival_and_preserves_exact_ids():
    scheduler, _encoder_cache_state = _make_deferred_mm_waiting_wave()

    first = scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )
    expected_ids = tuple(request.request_id for request in scheduler.waiting)
    assert first == ((0, 3, 2, 4, 0), True, expected_ids)
    assert scheduler._elastic_deferred_mm_wave is not None
    _settle_deferred_mm_fixture_capture(scheduler)

    arrival = Mock(
        request_id="arrival-after-capture",
        status=RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        has_encoder_inputs=False,
        num_tokens=2,
        num_prompt_tokens=2,
        execution_prefill_len=2,
        is_prefill_chunk=False,
        force_non_speculative=False,
    )
    arrival.is_finished.return_value = False
    scheduler.waiting.append(arrival)
    scheduler.requests[arrival.request_id] = arrival

    second = scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert second == ((0, 3, 2, 4, 0), False, expected_ids)
    assert scheduler._elastic_deferred_mm_wave is not None
    assert scheduler._can_fund_elastic_graph_step.call_count == 3


def test_deferred_mm_wave_cancellation_keeps_hot_capture_and_replans_survivor():
    scheduler, _encoder_cache_state = _make_deferred_mm_waiting_wave()
    scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )
    hot_keys = _settle_deferred_mm_fixture_capture(scheduler)
    cancelled = scheduler.waiting.pop(0)
    scheduler.requests.pop(cancelled.request_id)

    second = scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert second == ((0, 3, 1, 2, 0), False, ("prefill-1",))
    assert scheduler._elastic_deferred_mm_wave is None
    assert all(
        scheduler._elastic_admission_controller.entries[key].hot for key in hot_keys
    )
    assert scheduler._can_fund_elastic_graph_step.call_count == 4


def test_deferred_mm_wave_encoder_cache_drift_replans_before_graph_admission():
    scheduler, encoder_cache_state = _make_deferred_mm_waiting_wave()
    scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )
    _settle_deferred_mm_fixture_capture(scheduler)
    encoder_cache_state["prefill-0"] = True

    second = scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    )

    assert second == (
        (0, 3, 2, 4, 0),
        False,
        ("prefill-0", "prefill-1"),
    )
    assert scheduler._elastic_deferred_mm_wave is None
    # The stale tick-B intent is rejected before `_can_fund`; only the original
    # capture and the recursively replanned HOT/no-MM candidate reach admission.
    assert scheduler._can_fund_elastic_graph_step.call_count == 4
    assert "mm_activation_loan_bytes" not in (
        scheduler._can_fund_elastic_graph_step.call_args.kwargs
    )


def test_schedule_mixed_mm_cold_capture_then_exact_hot_user_wave(
    monkeypatch: pytest.MonkeyPatch,
):
    """A cold mixed MM shape captures before its exact USER wave mutates."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_seqs=5,
        max_num_batched_tokens=64,
    )
    graph_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_execution_policy = None
    scheduler.elastic_on_demand_graphs = False

    running = create_requests(
        num_requests=2,
        num_tokens=4,
        max_tokens=16,
        req_ids=["running-0", "running-1"],
    )
    for request in running:
        scheduler.add_request(request)
    initial_prefill = scheduler.schedule()
    running_ids = [request.request_id for request in running]
    scheduler.update_from_output(
        initial_prefill,
        ModelRunnerOutput(
            req_ids=running_ids,
            req_id_to_index={
                request_id: index for index, request_id in enumerate(running_ids)
            },
            sampled_token_ids=[[41] for _ in running],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler._elastic_graph_execution_policy = graph_policy

    mm_request = create_requests(
        num_requests=1,
        num_tokens=4,
        mm_hashes_list=[["mixed-mm-image"]],
        mm_positions=[[PlaceholderRange(offset=0, length=2)]],
        req_ids=["waiting-mm"],
    )[0]
    text_request = create_requests(
        num_requests=1,
        num_tokens=4,
        req_ids=["waiting-text"],
    )[0]
    waiting = [mm_request, text_request]
    for request in waiting:
        scheduler.add_request(request)

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_mm_activation_loan_bytes = 32
    controller = scheduler._elastic_admission_controller
    coordinator = scheduler.kv_cache_manager.coordinator

    fund_calls = 0

    def fund_exact_wave(step_key, **kwargs):
        nonlocal fund_calls
        if kwargs.get("preview_only"):
            return True, 128, 4096
        fund_calls += 1
        assert kwargs["mm_activation_loan_bytes"] == 32
        if fund_calls == 1:
            physical_keys = scheduler._resolve_elastic_step_physical_keys(step_key)
            for physical_key in physical_keys:
                controller.register(
                    physical_key,
                    price=GraphPrice(
                        1,
                        1,
                        f"private-pool:{physical_key.identity}",
                    ),
                )
            plan = controller.plan(
                controller.next_transaction_id(),
                physical_keys,
                request_bytes=0,
                available_bytes=96,
                destination_capture_endpoint_bytes=96,
                maintenance_execution=ElasticMaintenanceExecution.GRAPH_ONLY,
            )
            assert plan.kind == ElasticPlanKind.MAINTENANCE
            controller.arm_maintenance(plan, step_key)
            return False, 128, 4096
        return True, 128, 4096

    can_fund = Mock(side_effect=fund_exact_wave)
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)
    reserve_admission = Mock(return_value=True)
    monkeypatch.setattr(scheduler, "_reserve_elastic_admission", reserve_admission)

    def set_external_memory(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    monkeypatch.setattr(
        coordinator,
        "set_elastic_external_memory",
        Mock(side_effect=set_external_memory),
    )
    monkeypatch.setattr(
        coordinator,
        "max_elastic_external_memory",
        Mock(return_value=4096),
    )

    def plan_loan(step_key, *, mm_activation_loan_bytes=0, **_kwargs):
        grant = 96 + mm_activation_loan_bytes
        assert coordinator.set_elastic_external_memory(grant)
        controller.reserve_loan(step_key, grant)
        if mm_activation_loan_bytes == 0:
            controller.mark_recapture(step_key)
        return grant

    plan_graph_loan = Mock(side_effect=plan_loan)
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", plan_graph_loan)

    requests = [*running, *waiting]

    def request_state():
        return tuple(
            (
                request.request_id,
                request.status,
                request.num_computed_tokens,
                tuple(request.all_token_ids),
                tuple(request.spec_token_ids),
            )
            for request in requests
        )

    def kv_state():
        return tuple(
            (
                request.request_id,
                tuple(
                    tuple(group_ids)
                    for group_ids in scheduler.kv_cache_manager.get_block_ids(
                        request.request_id
                    )
                ),
            )
            for request in requests
        )

    encoder_manager = scheduler.encoder_cache_manager

    def encoder_state():
        return (
            {
                key: set(request_ids)
                for key, request_ids in encoder_manager.cached.items()
            },
            {
                request_id: set(input_ids)
                for request_id, input_ids in (
                    encoder_manager.request_cached_ids.items()
                )
            },
            tuple(encoder_manager.freeable.items()),
            tuple(encoder_manager.freed),
            encoder_manager.num_free_slots,
            encoder_manager.num_freeable_slots,
        )

    requests_before = request_state()
    kv_before = kv_state()
    encoder_before = encoder_state()
    running_before = tuple(request.request_id for request in scheduler.running)
    waiting_before = tuple(request.request_id for request in scheduler.waiting)

    capture_tick = scheduler.schedule()

    assert capture_tick.num_scheduled_tokens == {}
    assert capture_tick.scheduled_encoder_inputs == {}
    assert capture_tick.scheduled_new_reqs == []
    assert capture_tick.scheduled_cached_reqs.num_reqs == 0
    assert capture_tick.elastic_step_plan is not None
    assert capture_tick.elastic_step_plan.kind == ElasticPlanKind.MAINTENANCE
    assert (
        capture_tick.elastic_step_plan.maintenance_execution
        == ElasticMaintenanceExecution.GRAPH_ONLY
    )
    assert capture_tick.elastic_mm_activation_loan_bytes == 0
    assert capture_tick.elastic_graph_external_memory_bytes == 96
    assert capture_tick.elastic_external_memory_bytes == 96
    assert request_state() == requests_before
    assert kv_state() == kv_before
    assert encoder_state() == encoder_before
    assert tuple(request.request_id for request in scheduler.running) == running_before
    assert tuple(request.request_id for request in scheduler.waiting) == waiting_before

    frozen_wave = scheduler._elastic_deferred_mm_wave
    assert frozen_wave is not None
    capture_key = capture_tick.elastic_graph_step_key
    assert capture_key == frozen_wave.step_key
    _settle_elastic_with_receipt(
        scheduler,
        capture_tick,
        resident_bytes=1,
        floor_bytes=0,
        peak_bytes=1,
        publish_step_key=capture_key,
    )
    assert all(
        controller.entries[key].hot
        for key in scheduler._resolve_elastic_step_physical_keys(capture_key)
    )

    arrival = create_requests(
        num_requests=1,
        num_tokens=4,
        req_ids=["arrival-after-capture"],
    )[0]
    scheduler.add_request(arrival)

    user_tick = scheduler.schedule()

    frozen_tokens = dict(frozen_wave.scheduled_tokens)
    assert user_tick.elastic_graph_step_key == capture_key
    assert user_tick.num_scheduled_tokens == frozen_tokens
    assert set(frozen_tokens) == {request.request_id for request in requests}
    assert arrival.request_id not in user_tick.num_scheduled_tokens
    assert user_tick.scheduled_encoder_inputs == {mm_request.request_id: [0]}
    assert {request_data.req_id for request_data in user_tick.scheduled_new_reqs} == {
        request.request_id for request in waiting
    }
    assert all(
        request.request_id in user_tick.num_scheduled_tokens for request in running
    )
    assert user_tick.elastic_step_plan is not None
    assert user_tick.elastic_step_plan.kind == ElasticPlanKind.USER
    assert not user_tick.elastic_step_plan.cold_misses
    assert user_tick.elastic_mm_activation_loan_bytes == 32
    assert user_tick.elastic_graph_external_memory_bytes == 96
    assert user_tick.elastic_external_memory_bytes == 128
    assert scheduler._elastic_deferred_mm_wave is None
    assert arrival in scheduler.waiting
    assert can_fund.call_count == 3
    assert (
        sum(bool(call.kwargs.get("preview_only")) for call in can_fund.call_args_list)
        == 1
    )
    assert plan_graph_loan.call_count == 2

    _settle_elastic_with_receipt(
        scheduler,
        user_tick,
        resident_bytes=1,
        floor_bytes=0,
        peak_bytes=128,
        publish_step_key=capture_key,
    )
    assert not controller.pending_loans
    assert all(
        user_tick.elastic_transaction_id not in entry.leases
        for entry in controller.entries.values()
    )
    assert scheduler._elastic_deferred_mm_wave is None
    assert arrival in scheduler.waiting


def test_waiting_text_wave_includes_ready_structured_skipped_requests():
    scheduler = _make_elastic_waiting_text_wave_scheduler(num_reqs=6)
    scheduler.grammar_compile_error_reqs = set()
    scheduler.skipped_waiting = tuple(scheduler.waiting)
    scheduler.waiting = []
    for request in scheduler.skipped_waiting:
        request.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        request.structured_output_request = Mock(grammar=object())
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 64, 128))

    key, prepared_maintenance, request_ids = (
        scheduler._preflight_elastic_waiting_text_wave(
            token_budget=4096,
            prefill_chunk_cap=0,
            defer_prefills=False,
        )
    )

    assert key == (0, 3, 6, 16, 0)
    assert prepared_maintenance is False
    assert request_ids == tuple(f"prefill-{index}" for index in range(6))
    assert scheduler._can_fund_elastic_graph_step.call_count == 2
    assert scheduler._can_fund_elastic_graph_step.call_args_list[0].kwargs[
        "preview_only"
    ]
    assert scheduler._can_fund_elastic_graph_step.call_args_list[1].args == (key,)
    for request in scheduler.skipped_waiting:
        assert scheduler._try_promote_blocked_waiting_request(request)
        assert request.status == RequestStatus.WAITING


def test_waiting_text_wave_partial_or_stale_state_fails_before_maintenance():
    scheduler = _make_elastic_waiting_text_wave_scheduler()
    scheduler.waiting[3].num_stale_output_tokens = 1
    scheduler._can_fund_elastic_graph_step = Mock()

    assert scheduler._preflight_elastic_waiting_text_wave(
        token_budget=4096,
        prefill_chunk_cap=0,
        defer_prefills=False,
    ) == (None, False, ())
    scheduler._can_fund_elastic_graph_step.assert_not_called()


def test_waiting_text_wave_prices_final_shape_with_read_only_prefix_preview(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform
    from vllm.v1.core.sched import scheduler as scheduler_module

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    requests = create_requests(num_requests=3, num_tokens=4, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)

    def fund_final(key, **kwargs):
        if kwargs.get("preview_only"):
            return True, 1, 100
        for physical_key in scheduler._resolve_elastic_step_physical_keys(key):
            scheduler._elastic_admission_controller.publish_hot(
                physical_key,
                GraphPrice(1, 1, f"test:{physical_key.identity}"),
                pinned=False,
            )
        return True, 1, 100

    can_fund = Mock(side_effect=fund_final)
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))
    decode_query_lens = []
    canonical_order = scheduler_module.canonical_execution_request_order

    def record_canonical_order(*args, decode_query_len, **kwargs):
        decode_query_lens.append(decode_query_len)
        return canonical_order(*args, decode_query_len=decode_query_len, **kwargs)

    monkeypatch.setattr(
        scheduler_module,
        "canonical_execution_request_order",
        record_canonical_order,
    )

    def reserve_final(
        key,
        *,
        external_memory_bytes,
        minimum_free_primary_blocks,
        requirements,
    ):
        scheduler._elastic_preflight_admission_grant = SimpleNamespace(
            step_key=key,
            physical_keys=scheduler._resolve_elastic_step_physical_keys(key),
            maintenance_transaction_id=None,
            external_memory_bytes=external_memory_bytes,
            previous_external_memory_bytes=0,
            minimum_free_primary_blocks=minimum_free_primary_blocks,
            requirements=requirements,
        )
        return True

    monkeypatch.setattr(scheduler, "_reserve_elastic_admission", reserve_final)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {
        request.request_id: 4 for request in requests
    }
    assert can_fund.call_count == 2
    assert can_fund.call_args_list[0].args[0] == (0, 3, 3, 16, 0)
    assert can_fund.call_args_list[0].kwargs["preview_only"] is True
    assert can_fund.call_args_list[1].args[0] == (0, 3, 3, 16, 0)
    assert output.elastic_step_plan is not None
    assert {
        (key.logical.owner, key.logical.token_bucket)
        for key in output.elastic_step_plan.physical_keys
    } == {("target", 16), ("mtp_prefill", 16), ("mtp_decode", 3)}
    assert {
        (key.logical.owner, key.logical.token_bucket)
        for key in output.elastic_step_plan.successor_keys
    } == {("target", 12), ("mtp_prefill", 12), ("mtp_decode", 3)}
    assert output.elastic_step_plan.execution_manifest is not None
    assert decode_query_lens == [4]
    assert output.elastic_step_plan.execution_manifest.per_request_query_lens == (
        4,
        4,
        4,
    )
    assert output.elastic_step_plan.execution_manifest.per_request_is_prefilling == (
        True,
        True,
        True,
    )


def test_idle_waiting_wave_runs_maximal_feasible_fcfs_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    requests = create_requests(num_requests=8, num_tokens=4, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)

    preview_x: list[int] = []

    def fund_prefix(key, **kwargs):
        if kwargs.get("preview_only"):
            preview_x.append(key[2])
            return key[2] <= 4, key[2], 100
        assert key[2] == 4
        for physical_key in scheduler._resolve_elastic_step_physical_keys(key):
            scheduler._elastic_admission_controller.publish_hot(
                physical_key,
                GraphPrice(1, 1, f"selected:{physical_key.identity}"),
                pinned=False,
            )
        return True, 4, 100

    monkeypatch.setattr(
        scheduler, "_can_fund_elastic_graph_step", Mock(side_effect=fund_prefix)
    )
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    def reserve_prefix(key, **kwargs):
        scheduler._elastic_preflight_admission_grant = SimpleNamespace(
            step_key=key,
            physical_keys=scheduler._elastic_step_residency_intent(key)[2],
            maintenance_transaction_id=None,
            external_memory_bytes=kwargs["external_memory_bytes"],
            previous_external_memory_bytes=0,
            minimum_free_primary_blocks=kwargs["minimum_free_primary_blocks"],
            requirements=kwargs["requirements"],
        )
        return True

    monkeypatch.setattr(scheduler, "_reserve_elastic_admission", reserve_prefix)

    output = scheduler.schedule()

    assert preview_x == [8, 7, 6, 5, 4]
    assert tuple(output.num_scheduled_tokens) == tuple(
        request.request_id for request in requests[:4]
    )
    assert tuple(request.request_id for request in scheduler.waiting) == tuple(
        request.request_id for request in requests[4:]
    )
    assert output.elastic_graph_step_key == (0, 3, 4, 16, 0)


def test_q1_prefill_dispatch_does_not_use_q4_successor(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    request = create_requests(num_requests=1, num_tokens=1, max_tokens=1)[0]
    scheduler.add_request(request)

    def fund_current(key, **_kwargs):
        if _kwargs.get("preview_only"):
            return True, 1, 100
        for physical_key in scheduler._resolve_elastic_step_physical_keys(key):
            scheduler._elastic_admission_controller.publish_hot(
                physical_key,
                GraphPrice(1, 1, f"test:{physical_key.identity}"),
                pinned=False,
            )
        successor = scheduler._elastic_graph_carrier_closure_step_key(key)
        for physical_key in scheduler._resolve_elastic_step_physical_keys(successor):
            entry = scheduler._elastic_admission_controller.entries.get(physical_key)
            if entry is None or not entry.hot:
                scheduler._elastic_admission_controller.publish_hot(
                    physical_key,
                    GraphPrice(1, 1, f"test:{physical_key.identity}"),
                    pinned=False,
                )
        return True, 1, 100

    monkeypatch.setattr(
        scheduler, "_can_fund_elastic_graph_step", Mock(side_effect=fund_current)
    )
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    def reserve_current(key, **kwargs):
        scheduler._elastic_preflight_admission_grant = SimpleNamespace(
            step_key=key,
            physical_keys=scheduler._elastic_step_residency_intent(key)[2],
            maintenance_transaction_id=None,
            external_memory_bytes=kwargs["external_memory_bytes"],
            previous_external_memory_bytes=0,
            minimum_free_primary_blocks=kwargs["minimum_free_primary_blocks"],
            requirements=kwargs["requirements"],
        )
        return True

    monkeypatch.setattr(scheduler, "_reserve_elastic_admission", reserve_current)

    output = scheduler.schedule()

    assert output.elastic_graph_step_key == (0, 3, 1, 1, 0)
    assert output.elastic_step_plan is not None
    assert {
        (key.logical.owner, key.logical.token_bucket)
        for key in output.elastic_step_plan.physical_keys
    } == {
        ("target", 1),
        ("mtp_prefill", 1),
        ("mtp_decode", 1),
        ("target", 4),
        ("mtp_prefill", 4),
    }
    assert {
        (key.logical.owner, key.logical.token_bucket)
        for key in output.elastic_step_plan.successor_keys
    } == {("target", 4), ("mtp_prefill", 4), ("mtp_decode", 1)}
    manifest = output.elastic_step_plan.execution_manifest
    assert manifest is not None
    assert manifest.per_request_query_lens == (1,)
    assert manifest.per_request_is_prefilling == (True,)
    assert manifest.scheduled_draft_rows == (0,)
    assert {
        (key.logical.owner, key.logical.token_bucket)
        for key in output.elastic_step_plan.protected_keys
    } == {("target", 4), ("mtp_prefill", 4)}
    assert {
        (dispatch.invocation.owner, dispatch.invocation.physical_num_tokens)
        for dispatch in output.elastic_step_plan.current_dispatch
    } == {("target", 1), ("mtp_prefill", 1), ("mtp_decode", 1)}
    assert scheduler._elastic_graph_carrier_step_key is None

    scheduler._elastic_admission_controller.reserve_loan(
        output.elastic_graph_step_key, 0
    )
    coordinator = Mock()
    coordinator.set_elastic_external_memory.return_value = True
    scheduler.kv_cache_manager.coordinator = coordinator

    failed = scheduler.recover_elastic_execution_plan_mismatch(output)

    assert [item.request_id for item in failed] == [request.request_id]
    assert failed[0].status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert scheduler._elastic_graph_carrier_step_key is None
    assert not scheduler._elastic_admission_controller.pending_loans
    assert all(
        not entry.leases
        for entry in scheduler._elastic_admission_controller.entries.values()
    )
    coordinator.rebalance_elastic_capacity.assert_called_once_with()
    coordinator.set_elastic_external_memory.assert_called_once_with(
        scheduler._elastic_admission_controller.resident_bytes
    )


def test_finished_request_notifications_are_consumed_exactly_once():
    scheduler = _new_elastic_scheduler()
    scheduler.finished_req_ids_dict = {
        0: {"request-0"},
        1: {"request-1", "request-2"},
    }

    assert scheduler.take_finished_request_ids() == {
        0: {"request-0"},
        1: {"request-1", "request-2"},
    }
    assert scheduler.take_finished_request_ids() == {}
    assert scheduler.finished_req_ids_dict == {}


@pytest.mark.parametrize(
    ("live_tokens", "physical_tokens"),
    ((1, 1), (2, 2), (3, 4), (4, 4), (5, 8)),
)
def test_mixed_tail_reachability_uses_physical_piecewise_buckets(
    monkeypatch: pytest.MonkeyPatch,
    live_tokens: int,
    physical_tokens: int,
) -> None:
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()

    assert scheduler._canonical_elastic_graph_step_key(
        {"request-0": live_tokens}, 3, False
    ) == (0, 3, 1, physical_tokens, 0)


def test_cold_maintenance_does_not_infer_a_third_workspace_overlap():
    """Aggregate source and destination endpoints already include one W each."""
    generation = RuntimeGeneration("retained-overlap")
    scheduler = _new_elastic_scheduler(generation)
    step_key = (0, 3, 1, 4, 0)
    current_endpoint = 185_335_808
    old_grant = 297_795_584
    workspace_unit = 33_554_432
    miss_peaks = (23_855_104, 44_040_192, 44_564_480)

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    residency_step_key = scheduler._elastic_graph_carrier_closure_step_key(step_key)
    owner_key = scheduler._elastic_graph_owner_key(residency_step_key)
    assert owner_key is not None
    scheduler._elastic_admission_controller.capture_envelopes = {
        owner_key: (old_grant, 0, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.step_key = (0, 3, 2, 128, 0)
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    _clear_elastic_maintenance(scheduler)
    scheduler._elastic_admission_controller.resident_bytes = current_endpoint
    scheduler._elastic_admission_controller.cublas_workspace_bytes = workspace_unit
    scheduler._elastic_last_defer_reason = None
    scheduler._estimate_elastic_graph_step_bytes = Mock(return_value=(old_grant, True))

    physical_keys = scheduler._resolve_elastic_step_physical_keys(residency_step_key)
    for graph_key, peak in zip(physical_keys, miss_peaks, strict=True):
        scheduler._elastic_admission_controller.register(
            graph_key,
            price=GraphPrice(peak, peak, f"test:{graph_key.identity}"),
        )
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = current_endpoint
    coordinator.max_elastic_external_memory.return_value = 1 << 30
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        old_grant,
        True,
    )
    assert scheduler._elastic_capture_envelope(residency_step_key) == (
        old_grant,
        0,
        0,
    )

    fits, required, available = scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=True,
    )

    assert fits is False
    assert available == 1 << 30
    assert _pending_elastic_maintenance(scheduler) is not None
    expected_atomic_loan = current_endpoint + old_grant
    assert required == expected_atomic_loan
    assert (
        _pending_elastic_maintenance(scheduler).capture_loan_bytes
        == expected_atomic_loan
    )


def test_cold_x32_plan_deducts_exact_hot_destination_intersection_once():
    """Receipt-backed HOT destination owners are shared, not duplicated."""
    generation = RuntimeGeneration("x32-single-owner-ledger")
    scheduler = _new_elastic_scheduler(generation)
    step_key = (0, 3, 32, 4096, 0)
    current_external = 652_214_272
    pinned_entries = 304_087_040
    evictable_entries = 281_018_368
    shared_pool = 67_108_864
    destination_endpoint = 884_998_144
    transition_workspace = 33_554_432
    expected_loan = 952_107_008

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    controller = scheduler._elastic_admission_controller
    controller.step_key = (1, 3, 8, 32, 4)
    controller.recapture_pending_key = None
    owner_key = scheduler._elastic_graph_owner_key(step_key)
    assert owner_key is not None
    controller.capture_envelopes = {}
    controller.measured_bytes = {}
    controller.floor_bytes = 0
    controller.transition_floor_bytes = 0
    _reset_elastic_loans(scheduler)
    _clear_elastic_maintenance(scheduler)

    destination_current_keys, _successor_keys, physical_keys = (
        scheduler._elastic_step_residency_intent(step_key)
    )
    assert len(destination_current_keys) == 3
    receipt_entries = tuple(
        sorted(
            (
                ElasticResidencyEntry(
                    key=destination_current_keys[0],
                    pinned=True,
                    resident_bytes=pinned_entries,
                    local_pool_bytes=pinned_entries,
                    reclaimable_bytes=0,
                ),
                ElasticResidencyEntry(
                    key=destination_current_keys[1],
                    pinned=False,
                    resident_bytes=evictable_entries,
                    local_pool_bytes=evictable_entries,
                    reclaimable_bytes=evictable_entries,
                ),
            ),
            key=lambda entry: entry.key.identity,
        )
    )
    controller.accept_residency_receipt(
        ElasticResidencyReceipt(
            generation=generation,
            transaction_id=None,
            resident_bytes=current_external,
            floor_bytes=0,
            transition_floor_bytes=0,
            peak_bytes=current_external,
            cublas_workspace_bytes=transition_workspace,
            entries=receipt_entries,
        )
    )
    controller.register(
        destination_current_keys[2],
        price=GraphPrice(1, 1, "x32-cold-miss"),
    )
    assert physical_keys == destination_current_keys
    assert current_external - pinned_entries - evictable_entries == shared_pool
    scheduler._record_elastic_capture_envelope(
        step_key,
        (destination_endpoint, 0, 0),
        publish_global=False,
        resident_key_bytes=tuple(
            (entry.key.identity, entry.resident_bytes) for entry in receipt_entries
        ),
    )

    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = current_external
    coordinator.max_elastic_external_memory.return_value = expected_loan
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        destination_endpoint,
        True,
    )
    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=0,
        allow_maintenance=True,
    ) == (False, expected_loan, expected_loan)
    plan = _pending_elastic_maintenance(scheduler)
    assert plan is not None
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.capture_loan_bytes == expected_loan

    _clear_elastic_maintenance(scheduler)
    coordinator.max_elastic_external_memory.return_value = expected_loan - 1
    entries_before = dict(controller.entries)
    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=0,
        allow_maintenance=True,
    ) == (False, expected_loan, expected_loan - 1)
    assert _pending_elastic_maintenance(scheduler) is None
    assert controller.entries == entries_before


def test_cold_x2_does_not_deduct_untyped_residual_pool():
    """A numeric residual is not allocation-identity proof across endpoints."""
    generation = RuntimeGeneration("x2-transition-segment")
    scheduler = _new_elastic_scheduler(generation)
    current_step_key = (0, 3, 8, 32, 0)
    destination_step_key = (1, 3, 2, 2, 1)
    current_external = 612_368_384
    destination_endpoint = 180_355_072
    transition_segment = 20_971_520
    current_entry_bytes = current_external - destination_endpoint
    expected_loan = 792_723_456

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    controller = scheduler._elastic_admission_controller
    controller.step_key = current_step_key
    controller.recapture_pending_key = None
    controller.capture_envelopes = {}
    controller.measured_bytes = {}
    current_key = scheduler._resolve_elastic_step_physical_keys(current_step_key)[0]
    destination_keys = scheduler._resolve_elastic_step_physical_keys(
        destination_step_key
    )
    assert current_key not in destination_keys
    controller.accept_residency_receipt(
        ElasticResidencyReceipt(
            generation=generation,
            transaction_id=None,
            resident_bytes=current_external,
            floor_bytes=0,
            transition_floor_bytes=0,
            peak_bytes=current_external,
            cublas_workspace_bytes=transition_segment,
            entries=(
                ElasticResidencyEntry(
                    key=current_key,
                    pinned=True,
                    resident_bytes=current_entry_bytes,
                    local_pool_bytes=current_entry_bytes,
                    reclaimable_bytes=0,
                ),
            ),
        )
    )
    for destination_key in destination_keys:
        controller.register(
            destination_key,
            price=GraphPrice(1, 1, f"x2:{destination_key.identity}"),
        )
    scheduler._record_elastic_capture_envelope(
        destination_step_key,
        (destination_endpoint, 0, 0),
        publish_global=False,
        resident_key_bytes=(),
    )
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = current_external
    coordinator.max_elastic_external_memory.return_value = expected_loan
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._can_fund_elastic_graph_step(
        destination_step_key,
        minimum_free_primary_blocks=0,
        allow_maintenance=True,
    ) == (False, expected_loan, expected_loan)
    plan = _pending_elastic_maintenance(scheduler)
    assert plan is not None
    assert plan.capture_loan_bytes == expected_loan


def test_cold_plan_does_not_deduct_protected_successor_from_destination():
    """A retained successor is not a proved part of the measured endpoint."""
    generation = RuntimeGeneration("successor-not-shared")
    scheduler = _new_elastic_scheduler(generation)
    step_key = (0, 3, 1, 1, 0)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    controller = scheduler._elastic_admission_controller
    controller.step_key = (1, 3, 1, 4, 4)
    controller.recapture_pending_key = None
    controller.capture_envelopes = {}
    controller.measured_bytes = {}
    controller.floor_bytes = 0
    controller.transition_floor_bytes = 0
    controller.cublas_workspace_bytes = 0

    current_keys, successor_keys, physical_keys = (
        scheduler._elastic_step_residency_intent(step_key)
    )
    successor_only = next(key for key in successor_keys if key not in current_keys)
    controller.accept_residency_receipt(
        ElasticResidencyReceipt(
            generation=generation,
            transaction_id=None,
            resident_bytes=200,
            floor_bytes=0,
            transition_floor_bytes=0,
            peak_bytes=200,
            cublas_workspace_bytes=0,
            entries=tuple(
                sorted(
                    (
                        ElasticResidencyEntry(
                            key=current_keys[0],
                            pinned=True,
                            resident_bytes=100,
                            local_pool_bytes=100,
                            reclaimable_bytes=0,
                        ),
                        ElasticResidencyEntry(
                            key=successor_only,
                            pinned=False,
                            resident_bytes=80,
                            local_pool_bytes=80,
                            reclaimable_bytes=80,
                        ),
                    ),
                    key=lambda entry: entry.key.identity,
                )
            ),
        )
    )
    for cold_key in current_keys[1:]:
        controller.register(cold_key, price=GraphPrice(1, 1, "destination-miss"))
    for cold_key in physical_keys:
        if cold_key not in controller.entries:
            controller.register(cold_key, price=GraphPrice(1, 1, "destination-miss"))
    scheduler._record_elastic_capture_envelope(
        step_key,
        (250, 0, 0),
        publish_global=False,
        # The same identity contributed only 40 bytes to the saved destination
        # endpoint, so the current 100-byte entry may deduct at most 40.
        resident_key_bytes=((current_keys[0].identity, 40),),
    )
    _reset_elastic_loans(scheduler)
    _clear_elastic_maintenance(scheduler)

    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 200
    coordinator.max_elastic_external_memory.return_value = 410
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=0,
        allow_maintenance=True,
    ) == (False, 410, 410)
    plan = _pending_elastic_maintenance(scheduler)
    assert plan is not None
    assert successor_only in plan.hot_hits
    assert plan.capture_loan_bytes == 410

    # Once the selected destination receipt explicitly names the retained
    # successor, its exact shared bytes are deductible just like current keys.
    _clear_elastic_maintenance(scheduler)
    scheduler._record_elastic_capture_envelope(
        step_key,
        (250, 0, 0),
        publish_global=False,
        resident_key_bytes=(
            (current_keys[0].identity, 40),
            (successor_only.identity, 80),
        ),
    )
    coordinator.max_elastic_external_memory.return_value = 330
    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=0,
        allow_maintenance=True,
    ) == (False, 330, 330)
    plan = _pending_elastic_maintenance(scheduler)
    assert plan is not None
    assert plan.capture_loan_bytes == 330


def test_stale_serving_maintenance_replans_after_fresh_resource_snapshot():
    generation = RuntimeGeneration("stale-serving-replan")
    scheduler = _new_elastic_scheduler(generation)
    old_key = (0, 3, 8, 32, 4)
    new_key = (0, 3, 16, 64, 4)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_maintenance_started = {}
    old_plan = _arm_fixture_capture(scheduler, old_key, 96)
    scheduler._record_elastic_capture_envelope(
        new_key,
        (950, 0, 0),
        publish_global=False,
    )
    stale_grant = SimpleNamespace(
        step_key=old_key,
        physical_keys=old_plan.physical_keys,
        maintenance_transaction_id=old_plan.transaction_id,
        previous_external_memory_bytes=11,
    )
    scheduler._elastic_preflight_admission_grant = stale_grant

    observations: list[tuple[str, int]] = []
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 96

    def set_external(value, **_kwargs):
        coordinator.elastic_external_memory_bytes = value
        return True

    def max_external(**_kwargs):
        observations.append(("max", coordinator.elastic_external_memory_bytes))
        return 1_000 - coordinator.elastic_external_memory_bytes

    coordinator.set_elastic_external_memory.side_effect = set_external
    coordinator.max_elastic_external_memory.side_effect = max_external
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    original_estimator = scheduler._estimate_elastic_graph_step_bytes

    def observed_estimator(step_key):
        observations.append(("estimate", coordinator.elastic_external_memory_bytes))
        return original_estimator(step_key)

    scheduler._estimate_elastic_graph_step_bytes = observed_estimator

    assert scheduler._can_fund_elastic_graph_step(
        new_key,
        minimum_free_primary_blocks=0,
        allow_maintenance=True,
    ) == (False, 961, 989)
    assert observations == [("estimate", 11), ("max", 11)]
    assert coordinator.elastic_external_memory_bytes == 11
    assert scheduler._elastic_preflight_admission_grant is None
    replacement = _pending_elastic_maintenance(scheduler)
    assert replacement is not None
    assert replacement.transaction_id != old_plan.transaction_id
    assert scheduler._elastic_admission_controller.pending_maintenance_step_key == (
        new_key
    )


def test_same_shape_maintenance_rebinds_changed_kv_requirements_atomically():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("same-shape-rebind"))
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_maintenance_started = {}
    step_key = (0, 3, 8, 32, 4)
    plan = _arm_fixture_capture(scheduler, step_key, 96)
    old_requirements = KVCacheBlockPoolRequirements(primary=2, mamba=1)
    new_requirements = KVCacheBlockPoolRequirements(primary=5, mamba=3)
    scheduler._elastic_preflight_admission_grant = SimpleNamespace(
        step_key=step_key,
        physical_keys=plan.physical_keys,
        maintenance_transaction_id=plan.transaction_id,
        external_memory_bytes=96,
        previous_external_memory_bytes=11,
        minimum_free_primary_blocks=3,
        requirements=old_requirements,
    )
    coordinator = Mock(elastic_external_memory_bytes=96)

    def set_external(value, **_kwargs):
        coordinator.elastic_external_memory_bytes = value
        return True

    coordinator.set_elastic_external_memory.side_effect = set_external
    coordinator.ensure_elastic_capacity.return_value = True
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._reserve_pending_elastic_maintenance_admission(
        step_key,
        minimum_free_primary_blocks=8,
        requirements=new_requirements,
    )

    assert _pending_elastic_maintenance(scheduler) is plan
    grant = scheduler._elastic_preflight_admission_grant
    assert grant.maintenance_transaction_id == plan.transaction_id
    assert grant.previous_external_memory_bytes == 11
    assert grant.minimum_free_primary_blocks == 8
    assert grant.requirements == new_requirements
    assert coordinator.elastic_external_memory_bytes == 96
    assert coordinator.set_elastic_external_memory.call_args_list == [
        call(11),
        call(96, minimum_free_primary_blocks=8),
    ]


def test_serving_discard_preserves_exclusive_pressure_reclaim():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("pressure-preserved"))
    scheduler._elastic_maintenance_started = {}
    controller = scheduler._elastic_admission_controller
    hot_key = scheduler._resolve_elastic_step_physical_keys((1, 3, 1, 1, 1))[0]
    controller.publish_hot(hot_key, GraphPrice(8, 8, "pressure"), pinned=True)
    plan = controller.plan_pressure_reclaim_all(
        "pressure-tx", request_bytes=8, available_bytes=0
    )
    assert plan.kind == ElasticPlanKind.PRESSURE_RECLAIM
    controller.arm_maintenance(plan, None)
    before = controller.snapshot

    with pytest.raises(RuntimeError, match="exclusive elastic maintenance"):
        scheduler._clear_pre_mutation_serving_maintenance(reason="must-not-apply")

    assert controller.snapshot == before
    assert controller.pending_maintenance_plan is plan


def test_serving_discard_rejects_same_shape_grant_from_different_transaction():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("grant-tx-mismatch"))
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_maintenance_started = {}
    step_key = (0, 3, 8, 32, 4)
    plan = _arm_fixture_capture(scheduler, step_key, 96)
    grant = SimpleNamespace(
        step_key=step_key,
        physical_keys=plan.physical_keys,
        maintenance_transaction_id="different-tx",
        previous_external_memory_bytes=11,
    )
    scheduler._elastic_preflight_admission_grant = grant
    coordinator = Mock(elastic_external_memory_bytes=96)
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    before = scheduler._elastic_admission_controller.snapshot

    with pytest.raises(RuntimeError, match="not bound to pending maintenance"):
        scheduler._clear_pre_mutation_serving_maintenance(reason="must-not-apply")

    assert scheduler._elastic_admission_controller.snapshot == before
    assert scheduler._elastic_preflight_admission_grant is grant
    coordinator.rebalance_elastic_capacity.assert_not_called()
    coordinator.set_elastic_external_memory.assert_not_called()


@pytest.mark.parametrize("mutation", ("started", "capturing"))
def test_stale_serving_maintenance_rejects_post_mutation_discard(mutation: str):
    scheduler = _new_elastic_scheduler(RuntimeGeneration(f"stale-{mutation}"))
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_maintenance_started = {}
    plan = _arm_fixture_capture(scheduler, (0, 3, 8, 32, 4), 96)
    grant = SimpleNamespace(
        step_key=(0, 3, 8, 32, 4),
        physical_keys=plan.physical_keys,
        maintenance_transaction_id=plan.transaction_id,
        previous_external_memory_bytes=11,
    )
    scheduler._elastic_preflight_admission_grant = grant
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 96
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    scheduler._elastic_last_defer_reason = "sentinel"
    if mutation == "started":
        scheduler._elastic_maintenance_started[plan.transaction_id] = object()
    else:
        scheduler._elastic_admission_controller.begin_maintenance(plan)
    before = scheduler._elastic_admission_controller.snapshot

    with pytest.raises(RuntimeError, match="after physical mutation"):
        scheduler._clear_pre_mutation_serving_maintenance(reason="must-not-apply")

    assert scheduler._elastic_admission_controller.snapshot == before
    assert scheduler._elastic_preflight_admission_grant is grant
    assert coordinator.elastic_external_memory_bytes == 96
    assert scheduler._elastic_last_defer_reason == "sentinel"
    coordinator.rebalance_elastic_capacity.assert_not_called()
    coordinator.set_elastic_external_memory.assert_not_called()


def test_pending_serving_maintenance_commits_with_exact_user_wave(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    request = create_requests(num_requests=1, num_tokens=4)[0]
    scheduler.add_request(request)

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_graph_execution_policy = _approved_q4_full_policy()
    generation = _elastic_controller(scheduler).generation
    prospective_tokens = {request.request_id: 4}
    execution_step_key = scheduler._canonical_elastic_graph_step_key(
        prospective_tokens,
        scheduler._num_spec_tokens_for_step(prospective_tokens, False),
        False,
    )
    assert execution_step_key is not None
    step_key = execution_step_key
    physical_keys = resolve_step_physical_keys(
        step_key,
        generation,
        scheduler.scheduler_config.max_num_batched_tokens,
    )
    for physical_key in physical_keys:
        scheduler._elastic_admission_controller.register(
            physical_key,
            price=GraphPrice(100, 200, physical_key.logical.owner),
        )
    plan = scheduler._elastic_admission_controller.plan(
        "rising-x2",
        physical_keys,
        request_bytes=0,
        available_bytes=1_000,
        destination_capture_endpoint_bytes=900,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    _arm_elastic_maintenance(scheduler, plan, step_key)
    scheduler._elastic_graph_catalog = {
        step_key: {"cold_peak_bytes": 900, "floor_bytes": 0}
    }
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler._elastic_admission_controller.pinned_resident_bytes = 0
    scheduler._elastic_admission_controller.evictable_resident_bytes = 0
    monkeypatch.setattr(
        scheduler.kv_cache_manager.coordinator,
        "max_elastic_external_memory",
        Mock(return_value=1_000),
    )
    fund_calls = []
    original_can_fund = scheduler._can_fund_elastic_graph_step

    def observe_can_fund(*args, **kwargs):
        result = original_can_fund(*args, **kwargs)
        fund_calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", observe_can_fund)

    def reserve_same_wave(
        admitted_step_key: tuple[int, ...],
        *,
        external_memory_bytes: int,
        minimum_free_primary_blocks: int,
        requirements: KVCacheBlockPoolRequirements,
    ) -> bool:
        scheduler._elastic_preflight_admission_grant = SimpleNamespace(
            step_key=admitted_step_key,
            physical_keys=physical_keys,
            maintenance_transaction_id=plan.transaction_id,
            external_memory_bytes=external_memory_bytes,
            previous_external_memory_bytes=0,
            minimum_free_primary_blocks=minimum_free_primary_blocks,
            requirements=requirements,
        )
        return True

    monkeypatch.setattr(scheduler, "_reserve_elastic_admission", reserve_same_wave)
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=900))

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 4}
    assert output.total_num_scheduled_tokens == 4
    assert output.elastic_step_plan is not None
    assert output.elastic_step_plan.kind == ElasticPlanKind.MAINTENANCE
    assert output.elastic_transaction_id == plan.transaction_id
    assert list(scheduler.waiting) == []
    assert scheduler.running == [request]
    assert request.num_computed_tokens == 4
    assert len(fund_calls) == 2
    assert fund_calls[0][1]["preview_only"] is True
    assert "preview_only" not in fund_calls[1][1]


def test_provisional_piecewise_loan_does_not_pollute_global_envelope():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_admission_controller.capture_envelopes = {}
    key = (0, 3, 1, 2, 0)

    scheduler._record_elastic_capture_envelope(key, (5_000, 0, 0), publish_global=False)

    owner_key = scheduler._elastic_graph_owner_key(key)
    assert owner_key is not None
    assert scheduler._elastic_admission_controller.capture_envelopes[owner_key] == (
        5_000,
        0,
        0,
    )
    assert (
        0,
        3,
        0,
        0,
        0,
    ) not in scheduler._elastic_admission_controller.capture_envelopes

    scheduler._record_elastic_capture_envelope(key, (900, 0, 0))
    assert scheduler._elastic_admission_controller.capture_envelopes[
        (0, 3, 0, 0, 0)
    ] == (
        900,
        0,
        0,
    )


@pytest.mark.parametrize("invalid_loan", [True, -1, 1.5, "2097152"])
def test_elastic_mm_loan_config_rejects_non_byte_integer(invalid_loan):
    with pytest.raises(ValueError, match="must be an integer >= 0"):
        Scheduler._resolve_elastic_mm_activation_loan(
            {"elastic_mm_activation_loan_bytes": invalid_loan},
            elastic_on_demand_graphs=True,
            multimodal_config=Mock(skip_mm_profiling=True),
            elastic_mapping_quantum=2 << 20,
        )


def test_elastic_mm_loan_config_is_fail_closed_and_quantum_aligned():
    mm_skip = Mock(skip_mm_profiling=True)
    config = {"elastic_mm_activation_loan_bytes": 4 << 20}

    assert (
        Scheduler._resolve_elastic_mm_activation_loan(
            config,
            elastic_on_demand_graphs=True,
            multimodal_config=mm_skip,
            elastic_mapping_quantum=2 << 20,
        )
        == 4 << 20
    )
    with pytest.raises(ValueError, match="requires elastic KV/Graph"):
        Scheduler._resolve_elastic_mm_activation_loan(
            config,
            elastic_on_demand_graphs=False,
            multimodal_config=mm_skip,
            elastic_mapping_quantum=2 << 20,
        )
    with pytest.raises(ValueError, match="model with skip_mm_profiling"):
        Scheduler._resolve_elastic_mm_activation_loan(
            config,
            elastic_on_demand_graphs=True,
            multimodal_config=Mock(skip_mm_profiling=False),
            elastic_mapping_quantum=2 << 20,
        )
    with pytest.raises(ValueError, match="must align"):
        Scheduler._resolve_elastic_mm_activation_loan(
            {"elastic_mm_activation_loan_bytes": (4 << 20) + 1},
            elastic_on_demand_graphs=True,
            multimodal_config=mm_skip,
            elastic_mapping_quantum=2 << 20,
        )
    with pytest.raises(ValueError, match="requires a nonzero scheduler-visible"):
        Scheduler._resolve_elastic_mm_activation_loan(
            {},
            elastic_on_demand_graphs=True,
            multimodal_config=mm_skip,
            elastic_mapping_quantum=2 << 20,
        )

    assert (
        Scheduler._resolve_elastic_mm_activation_loan(
            {},
            env_value=str(4 << 20),
            elastic_on_demand_graphs=True,
            multimodal_config=mm_skip,
            elastic_mapping_quantum=2 << 20,
        )
        == 4 << 20
    )
    with pytest.raises(ValueError, match="config/env disagree"):
        Scheduler._resolve_elastic_mm_activation_loan(
            config,
            env_value=str(6 << 20),
            elastic_on_demand_graphs=True,
            multimodal_config=mm_skip,
            elastic_mapping_quantum=2 << 20,
        )
    with pytest.raises(ValueError, match="must be an integer"):
        Scheduler._resolve_elastic_mm_activation_loan(
            {},
            env_value="1GiB",
            elastic_on_demand_graphs=True,
            multimodal_config=mm_skip,
            elastic_mapping_quantum=2 << 20,
        )


def _make_hot_elastic_mm_scheduler(
    *, available_external: int = 96
) -> tuple[Scheduler, Mock, tuple[int, ...]]:
    generation = RuntimeGeneration("hot-mm-loan")
    scheduler = _new_elastic_scheduler(generation)
    step_key = (1, 3, 1, 4, 4)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.step_key = step_key
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    _clear_elastic_maintenance(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {step_key: 64}
    scheduler._elastic_admission_controller.resident_bytes = 64
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.transition_floor_bytes = 0
    scheduler._elastic_graph_catalog = {}
    scheduler.running = [object()]
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=64)
    for key in resolve_step_physical_keys(step_key, generation, 4096):
        scheduler._elastic_admission_controller.publish_hot(
            key,
            GraphPrice(1, 1, f"test:{key.identity}"),
            pinned=True,
        )

    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 64
    coordinator.max_elastic_external_memory.return_value = available_external
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    return scheduler, coordinator, step_key


def test_elastic_hot_graph_and_mm_loan_commit_one_aggregate_transaction():
    scheduler, coordinator, step_key = _make_hot_elastic_mm_scheduler()

    assert (
        scheduler._plan_elastic_graph_loan(
            step_key, minimum_free_primary_blocks=1, mm_activation_loan_bytes=32
        )
        == 96
    )
    assert _pending_elastic_loans(scheduler) == ((step_key, 96),)
    coordinator.max_elastic_external_memory.assert_called_once_with(
        minimum_free_primary_blocks=1
    )
    coordinator.set_elastic_external_memory.assert_called_once_with(
        96, minimum_free_primary_blocks=1
    )


def test_elastic_compiled_only_mm_preflight_adds_live_graph_residency():
    scheduler, coordinator, _step_key = _make_hot_elastic_mm_scheduler()
    compiled_key = (0, 0, 1, 128, 0)
    scheduler._elastic_compiled_piecewise_sizes = frozenset({128})
    policy = scheduler._elastic_graph_execution_policy
    scheduler._elastic_graph_execution_policy = dataclasses.replace(
        policy,
        owners=tuple(
            dataclasses.replace(owner, compiled_piecewise_sizes=(128,))
            if owner.owner == "target"
            else owner
            for owner in policy.owners
        ),
    )
    scheduler._elastic_admission_controller.resident_bytes = 32
    scheduler._elastic_admission_controller.measured_bytes = {compiled_key: 0}
    scheduler._elastic_admission_controller.capture_envelopes = {
        compiled_key: (0, 0, 0)
    }
    coordinator.elastic_external_memory_bytes = 32
    coordinator.max_elastic_external_memory.return_value = 64

    assert scheduler._can_fund_elastic_graph_step(
        compiled_key,
        minimum_free_primary_blocks=1,
        mm_activation_loan_bytes=32,
    ) == (True, 64, 64)


def test_elastic_mm_commit_does_not_relabel_aggregate_as_graph_residency():
    scheduler, coordinator, step_key = _make_hot_elastic_mm_scheduler()
    # Candidate preflight has already published the aggregate Graph+MM grant.
    # Commit must rebuild its typed components from the worker Graph receipt,
    # not treat the coordinator total as a Graph-only lower bound.
    coordinator.elastic_external_memory_bytes = 96

    assert (
        scheduler._plan_elastic_graph_loan(
            step_key, minimum_free_primary_blocks=1, mm_activation_loan_bytes=32
        )
        == 96
    )
    assert _pending_elastic_loans(scheduler) == ((step_key, 96),)
    coordinator.set_elastic_external_memory.assert_called_once_with(
        96, minimum_free_primary_blocks=1
    )


def test_elastic_hot_graph_and_mm_one_byte_short_rejects_before_commit():
    scheduler, coordinator, step_key = _make_hot_elastic_mm_scheduler(
        available_external=95
    )

    with pytest.raises(RuntimeError, match=r"Graph\+MM external loan exceeds"):
        scheduler._plan_elastic_graph_loan(
            step_key, minimum_free_primary_blocks=1, mm_activation_loan_bytes=32
        )

    assert not scheduler._elastic_admission_controller.pending_loans
    assert scheduler._elastic_admission_controller.step_key == step_key
    coordinator.set_elastic_external_memory.assert_not_called()


def test_worker_graph_mm_overlap_prices_resident_plus_transient():
    Scheduler._validate_elastic_graph_mm_overlap(
        worker_peak_bytes=104,
        granted_bytes=104,
        graph_granted_bytes=64,
        mm_activation_loan_bytes=40,
    )

    with pytest.raises(RuntimeError, match=r"Graph\+MM physical overlap"):
        Scheduler._validate_elastic_graph_mm_overlap(
            worker_peak_bytes=104,
            granted_bytes=103,
            graph_granted_bytes=64,
            mm_activation_loan_bytes=39,
        )


def test_elastic_mm_loan_rejects_cold_graph_without_state_mutation():
    scheduler, coordinator, step_key = _make_hot_elastic_mm_scheduler()
    scheduler._elastic_admission_controller = ElasticAdmissionController(
        _elastic_controller(scheduler).generation
    )
    entries_before = dict(scheduler._elastic_admission_controller.entries)

    with pytest.raises(RuntimeError, match="cannot overlap a cold Graph"):
        scheduler._plan_elastic_graph_loan(
            step_key, minimum_free_primary_blocks=1, mm_activation_loan_bytes=32
        )

    assert scheduler._elastic_admission_controller.entries == entries_before
    assert not scheduler._elastic_admission_controller.pending_loans
    coordinator.set_elastic_external_memory.assert_not_called()


def test_elastic_mm_cold_preflight_separates_capture_from_hot_mm_replay():
    scheduler, coordinator, step_key = _make_hot_elastic_mm_scheduler()
    scheduler._elastic_restore_mode = True
    generation = _elastic_controller(scheduler).generation
    scheduler._elastic_admission_controller = ElasticAdmissionController(generation)
    controller = scheduler._elastic_admission_controller
    controller.step_key = None
    for key in scheduler._resolve_elastic_step_physical_keys(step_key):
        controller.register(key, price=GraphPrice(1, 1, f"cold-mm:{key.identity}"))
    scheduler._record_elastic_capture_envelope(
        step_key,
        (64, 0, 0),
        publish_global=False,
    )
    controller.measured_bytes = {step_key: 64}
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 96
    entries_before = dict(controller.entries)

    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=True,
        mm_activation_loan_bytes=32,
    ) == (False, 96, 96)
    assert scheduler._elastic_last_defer_reason == "mm_graph_capture_pending"
    plan = controller.pending_maintenance_plan
    assert plan is not None
    assert plan.maintenance_execution == ElasticMaintenanceExecution.GRAPH_ONLY
    assert plan.capture_loan_bytes == 64
    assert getattr(scheduler, "_elastic_preflight_admission_grant", None) is None
    assert controller.entries == entries_before
    coordinator.set_elastic_external_memory.assert_not_called()

    # Model the successful tick-A settlement. Tick B now prices one HOT graph
    # endpoint plus the MM activation, rather than re-entering COLD capture.
    controller.clear_maintenance()
    controller.begin_maintenance(plan)
    controller.finish_maintenance(
        plan,
        {
            key: GraphPrice(64, 64, f"cold-mm:{key.identity}")
            for key in plan.cold_misses
        },
    )
    controller.resident_bytes = 64
    controller.step_key = step_key
    coordinator.elastic_external_memory_bytes = 64
    scheduler._elastic_restore_mode = False

    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=True,
        mm_activation_loan_bytes=32,
    ) == (True, 96, 96)


def test_elastic_mm_cold_preflight_uses_distinct_capture_and_hot_endpoints():
    scheduler, coordinator, step_key = _make_hot_elastic_mm_scheduler()
    scheduler._elastic_restore_mode = True
    generation = _elastic_controller(scheduler).generation
    scheduler._elastic_admission_controller = ElasticAdmissionController(generation)
    controller = scheduler._elastic_admission_controller
    old_key = resolve_step_physical_keys((0, 3, 8, 32, 0), generation, 4096)[0]
    old_price = GraphPrice(60, 60, "old-asymmetric")
    controller.publish_hot(old_key, old_price, pinned=False)
    controller.install_reclaim_group(ReclaimGroup("old-asymmetric", (old_key,), 60))
    for key in scheduler._resolve_elastic_step_physical_keys(step_key):
        controller.register(key, price=GraphPrice(1, 1, f"cold-mm:{key.identity}"))
    scheduler._record_elastic_capture_envelope(
        step_key,
        (140, 0, 0),
        publish_global=False,
    )
    controller.measured_bytes = {step_key: 100}
    coordinator.elastic_external_memory_bytes = 100
    coordinator.max_elastic_external_memory.return_value = 190

    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=True,
        mm_activation_loan_bytes=50,
    ) == (False, 190, 190)
    plan = controller.pending_maintenance_plan
    assert plan is not None
    assert plan.victim_keys == (old_key,)
    assert plan.reclaim_bytes == 60
    assert plan.capture_loan_bytes == 180

    controller.clear_maintenance()
    controller.begin_maintenance(plan)
    controller.finish_maintenance(
        plan,
        {key: GraphPrice(1, 1, f"cold-mm:{key.identity}") for key in plan.cold_misses},
    )
    controller.step_key = step_key
    coordinator.elastic_external_memory_bytes = 100
    scheduler._elastic_restore_mode = False

    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=True,
        mm_activation_loan_bytes=50,
    ) == (True, 150, 190)


def test_elastic_mm_cold_defer_reports_post_dominated_required_peak():
    scheduler, coordinator, step_key = _make_hot_elastic_mm_scheduler()
    scheduler._elastic_restore_mode = True
    generation = _elastic_controller(scheduler).generation
    scheduler._elastic_admission_controller = ElasticAdmissionController(generation)
    controller = scheduler._elastic_admission_controller
    old_key = resolve_step_physical_keys((0, 3, 8, 32, 0), generation, 4096)[0]
    controller.publish_hot(
        old_key,
        GraphPrice(59, 59, "old-one-byte-short"),
        pinned=False,
    )
    controller.install_reclaim_group(ReclaimGroup("old-one-byte-short", (old_key,), 59))
    for key in scheduler._resolve_elastic_step_physical_keys(step_key):
        controller.register(key, price=GraphPrice(1, 1, f"cold-mm:{key.identity}"))
    scheduler._record_elastic_capture_envelope(
        step_key,
        (140, 0, 0),
        publish_global=False,
    )
    controller.measured_bytes = {step_key: 100}
    coordinator.elastic_external_memory_bytes = 100
    coordinator.max_elastic_external_memory.return_value = 190

    assert scheduler._can_fund_elastic_graph_step(
        step_key,
        minimum_free_primary_blocks=1,
        allow_maintenance=True,
        mm_activation_loan_bytes=50,
    ) == (False, 250, 190)
    assert controller.pending_maintenance_plan is None


def test_worker_mm_loan_echo_mismatch_fails_before_graph_settlement():
    scheduler = _new_elastic_scheduler()
    scheduler.kv_cache_manager = Mock(coordinator=Mock())
    scheduler.defer_block_free = False
    scheduler._settle_elastic_graph_loan = Mock()
    output = SchedulerOutput.make_empty()
    output.elastic_mm_activation_loan_bytes = 32
    worker_output = ModelRunnerOutput(req_ids=[], req_id_to_index={})

    with pytest.raises(RuntimeError, match="MM activation loan echo differs"):
        scheduler.update_from_output(output, worker_output)

    scheduler._settle_elastic_graph_loan.assert_not_called()


def test_elastic_graph_async_loans_remain_borrowed_until_fifo_settlement():
    with pytest.raises(RuntimeError, match="requires one physical batch"):
        Scheduler._validate_elastic_batch_lifecycle(True, 2)
    return

    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_admission_controller.step_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_graph_discovery_base_bytes = 128
    scheduler._elastic_graph_discovery_bytes_per_token = 2
    scheduler._elastic_graph_recapture_workspace_bytes = 32
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler.running = [object(), object()]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 608
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: (
        ((value + 15) // 16) * 16
    )

    def commit_external(requested, **_kwargs):
        # Production commits in elastic_mapping_quantum units. Use a small
        # quantum here so the FIFO identity control covers normalization.
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    key_a = (1, 3, 1, 4, 4)
    key_b = (0, 3, 1, 37, 0)
    assert (
        scheduler._plan_elastic_graph_loan(key_a, minimum_free_primary_blocks=1) == 608
    )
    assert (
        scheduler._plan_elastic_graph_loan(key_b, minimum_free_primary_blocks=1) == 608
    )

    output_a = SchedulerOutput.make_empty()
    output_a.num_scheduled_tokens = {"a": 4}
    output_a.total_num_scheduled_tokens = 4
    output_a.num_spec_tokens_to_schedule = 3
    output_a.is_pure_decode_step = True
    output_a.elastic_external_memory_bytes = 608
    scheduler._settle_elastic_graph_loan(output_a, 48, 16)
    assert scheduler._elastic_admission_controller.capture_envelopes[(1, 1, 4, 4)] == (
        48,
        16,
        0,
    )
    coordinator.set_elastic_external_memory.assert_called_with(
        608, minimum_free_primary_blocks=2
    )
    assert scheduler._elastic_admission_controller.step_key == key_b

    # key_a was measured before, but key_b will evict it before this FIFO work
    # executes. Its output carries the measured recapture envelope; the
    # coordinator still holds 608 for the older unsettled key_b output.
    assert (
        scheduler._plan_elastic_graph_loan(key_a, minimum_free_primary_blocks=1) == 48
    )
    assert coordinator.elastic_external_memory_bytes == 608

    output_b = SchedulerOutput.make_empty()
    output_b.num_scheduled_tokens = {"b": 37}
    output_b.total_num_scheduled_tokens = 37
    output_b.num_spec_tokens_to_schedule = 3
    output_b.elastic_external_memory_bytes = 608
    scheduler._settle_elastic_graph_loan(output_b, 32, 16)
    # Settlement is measurement-only. The global scheduler state remains at
    # the FIFO-safe 608 bytes until the next step is actually planned.
    assert coordinator.elastic_external_memory_bytes == 608

    output_a_replay = SchedulerOutput.make_empty()
    output_a_replay.num_scheduled_tokens = {"a": 4}
    output_a_replay.total_num_scheduled_tokens = 4
    output_a_replay.num_spec_tokens_to_schedule = 3
    output_a_replay.is_pure_decode_step = True
    output_a_replay.elastic_external_memory_bytes = 48
    scheduler._settle_elastic_graph_loan(output_a_replay, 48, 16)
    assert not scheduler._elastic_admission_controller.pending_loans
    assert scheduler._elastic_admission_controller.measured_bytes == {
        key_a: 48,
        key_b: 32,
    }

    # The next known step is now the only resize point.
    assert (
        scheduler._plan_elastic_graph_loan(key_a, minimum_free_primary_blocks=1) == 48
    )
    assert coordinator.elastic_external_memory_bytes == 48


def test_settlement_rejects_incomplete_receipt_before_releasing_user_lease():
    generation = RuntimeGeneration("incomplete-receipt")
    scheduler = _new_elastic_scheduler(generation)
    scheduler.elastic_on_demand_graphs = True
    current = scheduler._resolve_elastic_step_physical_keys((0, 3, 1, 4, 0))[0]
    protected = scheduler._resolve_elastic_step_physical_keys((1, 3, 1, 4, 4))[0]
    controller = scheduler._elastic_admission_controller
    controller.publish_hot(current, GraphPrice(1, 1, "current"), pinned=False)
    controller.publish_hot(protected, GraphPrice(1, 1, "protected"), pinned=False)
    plan = controller.plan(
        "incomplete-user-receipt",
        (current,),
        protected_keys=(protected,),
        request_bytes=2,
        available_bytes=2,
    )
    assert plan.kind == ElasticPlanKind.USER
    controller.commit_user(plan)
    output = SchedulerOutput.make_empty()
    output.elastic_transaction_id = plan.transaction_id
    output.elastic_step_plan = plan
    scheduler.kv_cache_manager = Mock(coordinator=Mock())
    receipt = ElasticResidencyReceipt(
        generation=generation,
        transaction_id=plan.transaction_id,
        resident_bytes=1,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=1,
        cublas_workspace_bytes=0,
        entries=(
            ElasticResidencyEntry(
                key=protected,
                resident_bytes=1,
                local_pool_bytes=1,
                pinned=False,
                reclaimable_bytes=1,
            ),
        ),
    )
    before = controller.snapshot

    with pytest.raises(ElasticGraphError, match="omitted required HOT keys"):
        scheduler._settle_elastic_graph_loan(
            output,
            worker_resident_bytes=1,
            worker_floor_bytes=0,
            worker_peak_bytes=1,
            worker_receipt=receipt,
        )

    assert controller.snapshot == before
    assert controller.entries[current].leases == frozenset({plan.transaction_id})


def test_no_plan_settlement_rejects_crossed_receipt_before_state_mutation():
    generation = RuntimeGeneration("no-plan-receipt-identity")
    scheduler = _new_elastic_scheduler(generation)
    scheduler.elastic_on_demand_graphs = True
    controller = scheduler._elastic_admission_controller
    controller.reserve_loan(None, 0)
    output = SchedulerOutput.make_empty()
    output.elastic_transaction_id = "expected-no-plan-transaction"
    output.elastic_preserve_graph_residency = True
    scheduler.kv_cache_manager = Mock(coordinator=Mock())
    receipt = ElasticResidencyReceipt(
        generation=generation,
        transaction_id="stale-no-plan-transaction",
        resident_bytes=0,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=0,
        cublas_workspace_bytes=0,
        entries=(),
    )
    before = controller.snapshot

    with pytest.raises(ElasticGraphError, match="transaction"):
        scheduler._settle_elastic_graph_loan(
            output,
            worker_resident_bytes=0,
            worker_floor_bytes=0,
            worker_receipt=receipt,
        )

    assert controller.snapshot == before


@pytest.mark.parametrize(
    "corruption",
    [
        "transition_below_floor",
        "workspace_changed",
        "wrong_step_key",
        "wrong_grant",
        "wrong_breakdown",
        "drops_other_lease",
        "reclaim_group_changed",
    ],
)
def test_settlement_rejects_accounting_identity_before_any_commit(corruption: str):
    generation = RuntimeGeneration(f"atomic-settlement-{corruption}")
    scheduler = _new_elastic_scheduler(generation)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_retention_id = None
    scheduler.running = [object()]
    controller = scheduler._elastic_admission_controller
    step_key = (0, 3, 1, 4, 0)
    current = scheduler._resolve_elastic_step_physical_keys(step_key)[0]
    controller.publish_hot(
        current,
        GraphPrice(
            1,
            1,
            (
                "legacy-group"
                if corruption == "reclaim_group_changed"
                else f"private-pool:{current.identity}"
            ),
        ),
        pinned=False,
    )
    if corruption == "drops_other_lease":
        other = scheduler._resolve_elastic_step_physical_keys((0, 3, 2, 8, 0))[0]
        controller.publish_hot(
            other,
            GraphPrice(1, 1, f"private-pool:{other.identity}"),
            pinned=False,
        )
        other_plan = controller.plan(
            "other-active-user",
            (other,),
            request_bytes=1,
            available_bytes=8,
        )
        controller.commit_user(other_plan)
    controller.cublas_workspace_bytes = 4
    plan = controller.plan(
        "atomic-user",
        (current,),
        request_bytes=4,
        available_bytes=8,
    )
    assert plan.kind == ElasticPlanKind.USER
    controller.commit_user(plan)
    controller.reserve_loan(step_key, plan.capture_loan_bytes)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"request-0": 4}
    output.total_num_scheduled_tokens = 4
    output.num_spec_tokens_to_schedule = 3
    output.is_pure_decode_step = False
    output.elastic_graph_step_key = step_key
    output.elastic_transaction_id = plan.transaction_id
    output.elastic_step_plan = plan
    output.elastic_plan_fingerprint = plan.fingerprint
    output.elastic_external_memory_bytes = plan.capture_loan_bytes
    output.elastic_graph_external_memory_bytes = plan.capture_loan_bytes

    floor_bytes = 1
    transition_floor_bytes = 1
    workspace_bytes = 4
    if corruption == "transition_below_floor":
        floor_bytes = 2
        transition_floor_bytes = 1
    elif corruption == "workspace_changed":
        workspace_bytes = 8
    elif corruption == "wrong_step_key":
        output.elastic_graph_step_key = (0, 3, 2, 8, 0)
    elif corruption == "wrong_grant":
        output.elastic_external_memory_bytes += 1
        output.elastic_graph_external_memory_bytes += 1
    elif corruption == "wrong_breakdown":
        output.elastic_mm_activation_loan_bytes = 1

    receipt = ElasticResidencyReceipt(
        generation=generation,
        transaction_id=plan.transaction_id,
        resident_bytes=4,
        floor_bytes=floor_bytes,
        transition_floor_bytes=transition_floor_bytes,
        peak_bytes=max(4, floor_bytes, transition_floor_bytes),
        cublas_workspace_bytes=workspace_bytes,
        entries=(
            ElasticResidencyEntry(
                key=current,
                resident_bytes=4,
                local_pool_bytes=4,
                pinned=False,
                reclaimable_bytes=4,
            ),
        ),
    )
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = plan.capture_loan_bytes
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    before = controller.snapshot
    before_receipt = controller.last_receipt

    with pytest.raises((RuntimeError, ElasticGraphError)):
        scheduler._settle_elastic_graph_loan(
            output,
            worker_resident_bytes=receipt.resident_bytes,
            worker_floor_bytes=receipt.floor_bytes,
            worker_receipt=receipt,
        )

    assert controller.snapshot == before
    assert controller.last_receipt == before_receipt
    coordinator.set_elastic_external_memory.assert_not_called()


@pytest.mark.parametrize("calibration_mode", [False, True])
def test_elastic_worker_receipt_above_committed_loan_is_always_fatal(
    calibration_mode: bool,
):
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    key = (1, 3, 2, 8, 4)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = calibration_mode
    scheduler._elastic_admission_controller.recapture_pending_key = key
    _reset_elastic_loans(scheduler, (key, 160))
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler.running = [object(), object()]
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=64)
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 160
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    coordinator.set_elastic_external_memory.return_value = False
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"a": 4, "b": 4}
    output.total_num_scheduled_tokens = 8
    output.num_spec_tokens_to_schedule = 3
    output.is_pure_decode_step = True
    output.elastic_external_memory_bytes = 160

    with pytest.raises(RuntimeError, match="exceeds its Graph loan component"):
        _settle_elastic_with_receipt(scheduler, output, 224, 64, 240)

    assert scheduler._elastic_graph_catalog == {}
    assert scheduler._elastic_admission_controller.capture_envelopes == {}


def test_elastic_restore_capture_prepares_request_free_maintenance():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    _clear_elastic_maintenance(scheduler)

    def prepare_plan(step_key, **kwargs):
        assert kwargs["minimum_free_primary_blocks"] == 0
        assert kwargs["allow_maintenance"] is True
        _arm_fixture_capture(scheduler, step_key, 320)
        return False, 320, 1024

    scheduler._can_fund_elastic_graph_step = Mock(side_effect=prepare_plan)
    step_key = (0, 5, 7, 64, 0)

    assert scheduler.prepare_elastic_restore_capture(step_key) is True


def test_elastic_restore_capture_reuses_hot_owner_set():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    _clear_elastic_maintenance(scheduler)
    scheduler._can_fund_elastic_graph_step = Mock(return_value=(True, 384, 1024))

    assert scheduler.prepare_elastic_restore_capture((1, 1, 9, 18, 2)) is False


def test_elastic_restore_requires_complete_declared_hot_sequence():
    scheduler = _new_elastic_scheduler()
    prefill_owner = Mock(identity="prefill-owner")
    decode_owner = Mock(identity="decode-owner")
    scheduler._elastic_graph_carrier_closure_step_key = Mock(
        side_effect=lambda key: key
    )
    scheduler._resolve_elastic_step_physical_keys = Mock(
        side_effect=[(prefill_owner,), (decode_owner,)]
    )
    scheduler._elastic_admission_controller = SimpleNamespace(
        entries={
            prefill_owner: SimpleNamespace(hot=True),
            decode_owner: SimpleNamespace(hot=False),
        }
    )

    with pytest.raises(RuntimeError, match="did not remain HOT"):
        scheduler.assert_elastic_restore_captures_hot(
            ((0, 1, 9, 32, 0), (1, 1, 9, 18, 2))
        )


def test_elastic_restore_does_not_double_charge_non_kv_capture_peak():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    key = (0, 0, 40, 4096, 0)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.recapture_pending_key = key
    _reset_elastic_loans(scheduler, (key, 160))
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_graph_catalog_coverage = {}
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler.running = [object() for _ in range(40)]
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=64)
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 160
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {
        f"prefill-{index}": 103 if index < 16 else 102 for index in range(40)
    }
    output.total_num_scheduled_tokens = 4096
    output.num_spec_tokens_to_schedule = 0
    output.is_pure_decode_step = False
    output.elastic_external_memory_bytes = 160

    _settle_elastic_with_receipt(
        scheduler,
        output,
        144,
        16,
        168,
    )

    assert scheduler._elastic_graph_catalog[key]["cold_peak_bytes"] == 160
    assert scheduler._elastic_admission_controller.capture_envelopes[
        (0, 0, 0, 4096, 0)
    ] == (
        160,
        16,
        0,
    )


def test_elastic_runtime_cold_replay_cannot_shrink_sealed_capture_envelope():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    key = (1, 3, 1, 4, 4)
    sealed_envelope = 239075328
    runtime_measurement = 174063616
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.recapture_pending_key = key
    _reset_elastic_loans(scheduler, (key, sealed_envelope))
    scheduler._elastic_graph_catalog = {
        key: {
            "cold_peak_bytes": sealed_envelope,
            "hot_peak_bytes": sealed_envelope,
            "resident_bytes": 172752896,
            "floor_bytes": 40632320,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
        }
    }
    scheduler._elastic_admission_controller.capture_envelopes = {
        key: (sealed_envelope, 40632320, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {key: sealed_envelope}
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.idle_cleanup_started_at = None
    scheduler.running = [object()]
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=64)
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = sealed_envelope
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"decode": 4}
    output.total_num_scheduled_tokens = 4
    output.num_spec_tokens_to_schedule = 3
    output.is_pure_decode_step = True
    output.elastic_external_memory_bytes = sealed_envelope

    _settle_elastic_with_receipt(
        scheduler,
        output,
        172752896,
        40632320,
        runtime_measurement,
    )

    assert scheduler._elastic_admission_controller.capture_envelopes[key] == (
        sealed_envelope,
        40632320,
        0,
    )
    assert scheduler._elastic_graph_catalog[key]["cold_peak_bytes"] == (sealed_envelope)


def test_elastic_graph_async_same_shape_inherits_pending_recapture_grant():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_admission_controller.step_key = (1, 3, 1, 4, 4)
    scheduler._elastic_admission_controller.recapture_pending_key = (1, 3, 1, 4, 4)
    _reset_elastic_loans(
        scheduler,
        ((0, 3, 1, 196, 0), 512),
        ((1, 3, 1, 4, 4), 160),
    )
    scheduler._elastic_graph_discovery_base_bytes = 128
    scheduler._elastic_graph_discovery_bytes_per_token = 2
    # This is a valid historical recipe measurement but predates the pending
    # recapture after the incompatible M196 mixed step.
    scheduler._elastic_admission_controller.measured_bytes = {(1, 3, 1, 4, 4): 48}
    scheduler._elastic_admission_controller.resident_bytes = 112
    scheduler._elastic_admission_controller.floor_bytes = 44
    scheduler.running = [object()]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 512
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert (
        scheduler._plan_elastic_graph_loan(
            (1, 3, 1, 4, 4), minimum_free_primary_blocks=1
        )
        == 160
    )
    coordinator.max_elastic_external_memory.assert_not_called()
    coordinator.set_elastic_external_memory.assert_called_once_with(
        512, minimum_free_primary_blocks=1
    )
    assert _pending_elastic_loans(scheduler)[-1] == (
        (1, 3, 1, 4, 4),
        160,
    )


def test_elastic_capacity_receipt_distinguishes_unknown_from_proven(caplog):
    caplog.set_level("DEBUG", logger="vllm.v1.core.sched.scheduler")
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_primary_blocks_per_max_request = 36
    scheduler._elastic_admission_controller.last_capacity_receipt = None
    scheduler.scheduler_config = Mock(max_num_seqs=64)
    scheduler.kv_cache_config = Mock(num_blocks=74)
    scheduler.max_model_len = 262144
    coordinator = Mock()
    coordinator.max_elastic_full_context_requests.return_value = 1
    coordinator.elastic_kv_authority_receipt.return_value = {
        "raw_attention_blocks_per_rank": (74, 74, 74),
        "raw_attention_bytes_per_rank": (740, 740, 740),
        "raw_attention_token_equivalent_per_rank": (262144, 262144, 262144),
        "effective_attention_blocks_per_rank": (37, 37, 37),
        "effective_attention_bytes_per_rank": (370, 370, 370),
        "effective_attention_token_equivalent_per_rank": (131072, 131072, 131072),
        "active_gdn_blocks": 1,
        "primary_blocks_per_max_request": 36,
    }
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    step_key = (1, 3, 2, 8, 4)

    scheduler._log_elastic_executable_capacity(step_key, 830 << 20, known=False)
    assert "calibration pending for cold step" in caplog.text
    coordinator.max_elastic_full_context_requests.assert_not_called()

    caplog.clear()
    scheduler._log_elastic_executable_capacity(step_key, 195 << 20, known=True)
    coordinator.max_elastic_full_context_requests.assert_called_once_with(
        36,
        195 << 20,
        74,
    )
    assert "max_full_context_x=1" in caplog.text
    assert "max_model_len=262144" in caplog.text


def test_elastic_startup_capacity_publishes_only_proven_effective_geometry(caplog):
    caplog.set_level("INFO")
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_primary_blocks_per_max_request = 36
    scheduler.max_num_running_reqs = 64
    scheduler.scheduler_config = Mock(max_num_batched_tokens=16384)
    scheduler.kv_cache_manager = Mock()
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.max_elastic_full_context_requests.return_value = 2
    coordinator.elastic_full_context_capacity_receipt.return_value = {
        "num_reqs": 2,
        "external_memory_bytes": 240,
        "gdn_blocks": 7,
        "attention_blocks": 74,
        "required_attention_blocks": 73,
        "residual_attention_blocks": 1,
    }
    scheduler._elastic_graph_catalog = {
        key: {
            "cold_peak_bytes": cold,
            "hot_peak_bytes": cold,
            "cold_observations": 2,
            "hot_observations": 2,
            "cold_stable_replays": 1,
            "hot_stable_replays": 1,
        }
        for key, cold in {
            (1, 3, 1, 1, 1): 100,
            (1, 3, 2, 2, 1): 200,
            (0, 3, 1, 16384, 0): 120,
            (0, 3, 2, 16384, 0): 240,
        }.items()
    }

    scheduler._publish_elastic_startup_capacity()

    assert "DecodeMaxX[K3]=2" in caplog.text
    assert "GuaranteedMaxX[K3,B16384]=2" in caplog.text
    assert "GuaranteedFullContextX[K3]=2" in caplog.text
    assert "decode_cold_envelope_bytes=200" in caplog.text
    assert "max_b_cold_envelope_bytes=240" in caplog.text
    assert "full_context_cold_envelope_bytes=200" in caplog.text
    assert "residual_attention_blocks=1" in caplog.text
    assert "raw_x0" not in caplog.text
    coordinator.elastic_full_context_capacity_receipt.assert_called_once_with(
        36, 200, 2
    )


def test_elastic_startup_capacity_does_not_confuse_decode_with_full_context(
    caplog,
):
    caplog.set_level("INFO")
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_primary_blocks_per_max_request = 36
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.kv_cache_manager = Mock()
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.max_elastic_full_context_requests.side_effect = lambda _, __, x: (
        1 if x > 1 else x
    )
    coordinator.elastic_full_context_capacity_receipt.return_value = {
        "num_reqs": 1,
        "external_memory_bytes": 100,
        "gdn_blocks": 4,
        "attention_blocks": 64,
        "required_attention_blocks": 37,
        "residual_attention_blocks": 27,
    }

    def complete(cold):
        return {
            "cold_peak_bytes": cold,
            "hot_peak_bytes": cold,
            "cold_observations": 2,
            "hot_observations": 2,
            "cold_stable_replays": 1,
            "hot_stable_replays": 1,
        }

    scheduler._elastic_graph_catalog = {
        **{(1, 3, x, 4 * x, 4): complete(100 + x) for x in range(1, 23)},
        **{(0, 3, x, 4096, 0): complete(200 + x) for x in range(1, 19)},
        # A larger but incomplete probe must not enter any published claim.
        (1, 3, 38, 152, 4): {
            **complete(999),
            "hot_stable_replays": 0,
        },
    }

    scheduler._publish_elastic_startup_capacity()

    assert "DecodeMaxX[K3]=22" in caplog.text
    assert "GuaranteedMaxX[K3,B4096]=18" in caplog.text
    assert "GuaranteedFullContextX[K3]=1" in caplog.text
    assert "DecodeMaxX[K3]=38" not in caplog.text
    coordinator.elastic_full_context_capacity_receipt.assert_called_once_with(
        36, 101, 1
    )


def test_elastic_startup_capacity_uses_sealed_boundary_not_larger_probe_rows(
    caplog,
):
    caplog.set_level("INFO")
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_primary_blocks_per_max_request = 36
    scheduler.num_spec_tokens = 3
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.kv_cache_manager = Mock()
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.max_elastic_full_context_requests.side_effect = lambda _, __, x: (
        1 if x > 1 else x
    )
    coordinator.elastic_full_context_capacity_receipt.return_value = {
        "num_reqs": 1,
        "external_memory_bytes": 100,
        "gdn_blocks": 4,
        "attention_blocks": 64,
        "required_attention_blocks": 37,
        "residual_attention_blocks": 27,
    }
    scheduler._elastic_graph_catalog_coverage = {
        "decode_max_x": 14,
        "mixed_max_x": 14,
        "full_context_max_x": 1,
    }

    def complete(cold):
        return {
            "cold_peak_bytes": cold,
            "hot_peak_bytes": cold,
            "cold_observations": 2,
            "hot_observations": 2,
            "cold_stable_replays": 1,
            "hot_stable_replays": 1,
        }

    scheduler._elastic_graph_catalog = {
        **{(1, 3, x, 4 * x, 4): complete(100 + x) for x in range(1, 23)},
        **{(0, 3, x, 4096, 0): complete(200 + x) for x in range(1, 19)},
    }

    scheduler._publish_elastic_startup_capacity()

    assert "DecodeMaxX[K3]=14" in caplog.text
    assert "GuaranteedMaxX[K3,B4096]=14" in caplog.text
    assert "GuaranteedFullContextX[K3]=1" in caplog.text
    assert "DecodeMaxX[K3]=22" not in caplog.text


def test_bounded_startup_capacity_accepts_stable_piecewise_cold_bound(caplog):
    caplog.set_level("INFO")
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_primary_blocks_per_max_request = 36
    scheduler.num_spec_tokens = 3
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.kv_cache_manager = Mock()
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.max_elastic_full_context_requests.return_value = 1
    coordinator.elastic_full_context_capacity_receipt.return_value = {
        "num_reqs": 1,
        "external_memory_bytes": 100,
        "gdn_blocks": 4,
        "attention_blocks": 64,
        "required_attention_blocks": 37,
        "residual_attention_blocks": 27,
    }
    scheduler._elastic_graph_catalog_coverage = {
        "representation": "bounded_exact_hotset",
        "decode_max_x": 40,
        "mixed_max_x": 40,
        "full_context_max_x": 1,
    }
    scheduler._elastic_graph_catalog = {
        (1, 3, 1, 1, 1): {
            "cold_peak_bytes": 100,
            "hot_peak_bytes": 90,
            "cold_observations": 2,
            "hot_observations": 2,
            "cold_stable_replays": 1,
            "hot_stable_replays": 1,
        },
        (1, 3, 40, 40, 1): {
            "cold_peak_bytes": 300,
            "hot_peak_bytes": 280,
            "cold_observations": 2,
            "hot_observations": 2,
            "cold_stable_replays": 1,
            "hot_stable_replays": 1,
        },
        (0, 3, 40, 4096, 0): {
            "cold_peak_bytes": 400,
            "hot_peak_bytes": 0,
            "cold_observations": 2,
            "hot_observations": 0,
            "cold_stable_replays": 1,
            "hot_stable_replays": 0,
        },
    }

    scheduler._publish_elastic_startup_capacity()

    assert "DecodeMaxX[K3]=40" in caplog.text
    assert "GuaranteedMaxX[K3,B4096]=40" in caplog.text
    assert "GuaranteedFullContextX[K3]=1" in caplog.text


def test_bounded_startup_capacity_accepts_phase_split_nondecode_k0(caplog):
    caplog.set_level("INFO")
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_primary_blocks_per_max_request = 36
    scheduler.num_spec_tokens = 3
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.speculative_config.disable_speculation_on_non_decode = True
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.kv_cache_manager = Mock()
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.max_elastic_full_context_requests.return_value = 1
    coordinator.elastic_full_context_capacity_receipt.return_value = {
        "num_reqs": 1,
        "external_memory_bytes": 100,
        "gdn_blocks": 4,
        "attention_blocks": 64,
        "required_attention_blocks": 37,
        "residual_attention_blocks": 27,
    }
    scheduler._elastic_graph_catalog_coverage = {
        "representation": "bounded_exact_hotset",
        "decode_max_x": 40,
        "mixed_max_x": 40,
        "full_context_max_x": 1,
    }
    scheduler._elastic_admission_controller.pinned_resident_bytes = 0
    full = {
        "cold_peak_bytes": 300,
        "hot_peak_bytes": 280,
        "cold_observations": 2,
        "hot_observations": 2,
        "cold_stable_replays": 1,
        "hot_stable_replays": 1,
    }
    piecewise = {
        "cold_peak_bytes": 620,
        "hot_peak_bytes": 0,
        "cold_observations": 2,
        "hot_observations": 0,
        "cold_stable_replays": 1,
        "hot_stable_replays": 0,
    }
    scheduler._elastic_graph_catalog = {
        (1, 3, 1, 1, 1): full,
        (1, 3, 40, 40, 1): full,
        (0, 0, 40, 4096, 0): piecewise,
    }

    scheduler._publish_elastic_startup_capacity()

    assert "DecodeMaxX[K3]=40" in caplog.text
    assert "GuaranteedMaxX[decodeK3,prefillK0,B4096]=40" in caplog.text


def test_bounded_startup_capacity_accepts_compiled_only_max_b_carrier(caplog):
    caplog.set_level("INFO")
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_primary_blocks_per_max_request = 36
    scheduler.num_spec_tokens = 3
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.speculative_config.disable_speculation_on_non_decode = True
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_compiled_piecewise_sizes = frozenset({4096})
    scheduler.kv_cache_manager = Mock()
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.max_elastic_full_context_requests.return_value = 1
    coordinator.elastic_full_context_capacity_receipt.return_value = {
        "num_reqs": 1,
        "external_memory_bytes": 100,
        "gdn_blocks": 4,
        "attention_blocks": 64,
        "required_attention_blocks": 37,
        "residual_attention_blocks": 27,
    }
    scheduler._elastic_graph_catalog_coverage = {
        "representation": "bounded_exact_hotset",
        "decode_max_x": 40,
        "mixed_max_x": 40,
        "full_context_max_x": 1,
        "compiled_piecewise_sizes": [4096],
        "compiled_piecewise_contract": "torch-compile-no-cudagraph-v1",
    }
    scheduler._elastic_admission_controller.pinned_resident_bytes = 0
    full = {
        "cold_peak_bytes": 300,
        "hot_peak_bytes": 280,
        "cold_observations": 2,
        "hot_observations": 2,
        "cold_stable_replays": 1,
        "hot_stable_replays": 1,
    }
    scheduler._elastic_graph_catalog = {
        (1, 3, 1, 1, 1): full,
        (1, 3, 40, 40, 1): full,
    }

    scheduler._publish_elastic_startup_capacity()

    assert "DecodeMaxX[K3]=40" in caplog.text
    assert "GuaranteedMaxX[decodeK3,prefillK0,B4096]=40" in caplog.text
    assert "max_b_cold_envelope_bytes=0" in caplog.text


def test_product_phase_policy_keeps_k3_decode_and_uses_k0_prefill():
    scheduler = _new_elastic_scheduler()
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = {1: 3}
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.speculative_config.disable_speculation_on_non_decode = True
    scheduler.requests = {
        "r0": Mock(
            force_non_speculative=False,
            num_computed_tokens=0,
            num_prompt_tokens=4096,
        )
    }

    assert scheduler._num_spec_tokens_for_step({"r0": 4096}, False) == 0
    assert scheduler._num_spec_tokens_for_step({"r0": 1}, True) == 3


def test_product_phase_policy_uses_k0_for_nondecode_tail():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.speculative_config.disable_speculation_on_non_decode = True
    scheduler.requests = {
        "r0": Mock(
            force_non_speculative=False,
            num_computed_tokens=4096,
            num_prompt_tokens=4096,
        )
    }

    assert scheduler._num_spec_tokens_for_step({"r0": 2}, False) == 0
    assert scheduler._canonical_elastic_graph_step_key(
        {"r0": 2},
        scheduler._num_spec_tokens_for_step({"r0": 2}, False),
        False,
    ) == (0, 0, 1, 2, 0)
    assert scheduler._num_spec_tokens_for_step({"r0": 1}, True) == 3


def test_product_phase_policy_uses_k0_for_mixed_decode_and_prefill():
    scheduler = _new_elastic_scheduler()
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.speculative_config.disable_speculation_on_non_decode = True
    scheduler.requests = {
        "decode": Mock(
            force_non_speculative=False,
            num_computed_tokens=100,
            num_prompt_tokens=100,
        ),
        "prefill": Mock(
            force_non_speculative=False,
            num_computed_tokens=0,
            num_prompt_tokens=4096,
        ),
    }

    assert (
        scheduler._num_spec_tokens_for_step({"decode": 1, "prefill": 4095}, False) == 0
    )


def test_product_phase_policy_request_override_still_forces_mixed_k0():
    scheduler = _new_elastic_scheduler()
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.speculative_config.disable_speculation_on_non_decode = True
    scheduler.requests = {
        "decode": Mock(
            force_non_speculative=True,
            num_computed_tokens=100,
            num_prompt_tokens=100,
        ),
        "prefill": Mock(
            force_non_speculative=False,
            num_computed_tokens=0,
            num_prompt_tokens=4096,
        ),
    }

    assert (
        scheduler._num_spec_tokens_for_step({"decode": 1, "prefill": 4095}, False) == 0
    )


def test_elastic_graph_first_async_copy_inherits_unmeasured_same_shape_grant():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    live_key = (0, 3, 12, 4036, 0)
    scheduler._elastic_admission_controller.step_key = live_key
    carrier_key = scheduler._elastic_graph_carrier_closure_step_key(live_key)
    # FIFO settlement keeps the mixed execution identity, while provisional
    # residency lookup projects it onto this stable carrier identity.
    scheduler._elastic_admission_controller.recapture_pending_key = carrier_key
    _reset_elastic_loans(scheduler, (live_key, 1929379840))
    scheduler._elastic_graph_discovery_base_bytes = 112 << 20
    scheduler._elastic_graph_discovery_bytes_per_token = 256 << 10
    scheduler._elastic_graph_capture_workspace_bytes = 672 << 20
    scheduler._elastic_sampling_workspace_bytes_per_row = 248320 * 4
    scheduler._elastic_sampling_sort_workspace_bytes_per_row = 248320 * 45
    scheduler._elastic_sampling_sort_workspace_min_bytes = 22 << 20
    scheduler._elastic_sampling_triton_min_rows = 8
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.resident_bytes = 0
    # Exact v30 value: recomputing a cold envelope added this newly measured
    # floor and requested 1,961,588,736 bytes from a 1,960,837,120-byte cap.
    scheduler._elastic_admission_controller.floor_bytes = 33814528
    scheduler.running = [object() for _ in range(12)]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 1929379840
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert (
        scheduler._plan_elastic_graph_loan(
            live_key, minimum_free_primary_blocks=len(scheduler.running)
        )
        == 1929379840
    )
    coordinator.max_elastic_external_memory.assert_not_called()
    coordinator.set_elastic_external_memory.assert_called_once_with(
        1929379840, minimum_free_primary_blocks=12
    )


def test_elastic_graph_measured_x12_mixed_rejects_successor_only_envelope():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    live_key = (0, 3, 12, 4036, 0)
    carrier_key = scheduler._elastic_graph_carrier_closure_step_key(live_key)
    # The mixed execution key and its resident q4 closure are distinct. An
    # intervening execution may leave the sealed closure envelope as the
    # authoritative price for restoring the carrier.
    scheduler._elastic_admission_controller.step_key = (0, 3, 1, 109, 0)
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_graph_discovery_base_bytes = 112 << 20
    scheduler._elastic_graph_discovery_bytes_per_token = 256 << 10
    scheduler._elastic_graph_capture_workspace_bytes = 672 << 20
    scheduler._elastic_graph_recapture_workspace_bytes = 32 << 20
    scheduler._elastic_sampling_workspace_bytes_per_row = 248320 * 4
    scheduler._elastic_sampling_sort_workspace_bytes_per_row = 248320 * 45
    scheduler._elastic_sampling_sort_workspace_min_bytes = 22 << 20
    scheduler._elastic_sampling_triton_min_rows = 8
    # Exact v31 HOT owners plus the maximum twenty 32-MiB cuBLAS workspaces and
    # the worker-reported 33,814,528-byte physical floor.
    measured_external = 569810944 + 423624704 + 44040192 + 20 * (32 << 20) + 33814528
    scheduler._elastic_admission_controller.measured_bytes = {
        carrier_key: measured_external
    }
    scheduler._elastic_admission_controller.capture_envelopes = {
        scheduler._elastic_graph_owner_key(carrier_key): (
            1929379840,
            33814528,
            0,
        )
    }
    scheduler._elastic_admission_controller.resident_bytes = measured_external
    scheduler._elastic_admission_controller.floor_bytes = 33814528
    scheduler.running = [object() for _ in range(12)]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = measured_external
    coordinator.max_elastic_external_memory.return_value = 1960837120
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    # A sealed successor envelope cannot price the distinct current mixed
    # execution. The M4036 owner set needs its own measured row.
    with pytest.raises(RuntimeError, match="unpriced elastic CUDA Graph"):
        scheduler._plan_elastic_graph_loan(
            live_key, minimum_free_primary_blocks=len(scheduler.running)
        )
    assert _pending_elastic_loans(scheduler) == ()
    coordinator.max_elastic_external_memory.assert_called_once_with(
        minimum_free_primary_blocks=12
    )
    coordinator.set_elastic_external_memory.assert_not_called()
    assert scheduler._elastic_admission_controller.recapture_pending_key is None


def test_elastic_restore_retained_owner_reborrows_exact_tail():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.step_key = (0, 3, 1, 64, 0)
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    _publish_elastic_step_hot(scheduler, (1, 3, 1, 4, 4))
    scheduler._elastic_admission_controller.capture_envelopes = {
        (1, 3, 1, 4, 4): (160, 0, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {(1, 3, 1, 4, 4): 128}
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler.running = [object()]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.max_elastic_external_memory.return_value = 608
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    key = (1, 3, 1, 4, 4)
    assert scheduler._can_fund_elastic_graph_step(
        key, minimum_free_primary_blocks=1
    ) == (True, 608, 608)
    assert (
        scheduler._plan_elastic_graph_loan(
            key, minimum_free_primary_blocks=len(scheduler.running)
        )
        == 608
    )
    assert scheduler._elastic_admission_controller.capture_envelopes[key] == (160, 0, 0)


def test_elastic_restore_hot_witness_reborrows_tail_for_coalesced_descriptor():
    scheduler = _new_elastic_scheduler()
    key = (1, 3, 1, 4, 4)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.step_key = key
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    _publish_elastic_step_hot(scheduler, key)
    scheduler._elastic_admission_controller.capture_envelopes = {key: (151, 0, 0)}
    scheduler._elastic_admission_controller.measured_bytes = {key: 126}
    scheduler._elastic_graph_catalog = {key: {"cold_peak_bytes": 151}}
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 149
    scheduler.running = [object()]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 608
    coordinator.max_elastic_external_memory.return_value = 608
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._estimate_elastic_graph_step_bytes(key) == (149, False)
    assert (
        scheduler._plan_elastic_graph_loan(
            key, minimum_free_primary_blocks=len(scheduler.running)
        )
        == 608
    )
    coordinator.max_elastic_external_memory.assert_called_once_with(
        minimum_free_primary_blocks=1
    )


@pytest.mark.parametrize(
    "peak, resident, expected_loan",
    [(300, 174, 300), (344, 174, 332), (350, 350, None)],
)
def test_elastic_hot_price_records_kv_loan_not_total_cuda_peak(
    peak, resident, expected_loan
):
    scheduler = _new_elastic_scheduler(RuntimeGeneration("hot-loan-unit"))
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    key = (0, 3, 39, 1024, 0)
    controller = scheduler._elastic_admission_controller
    controller.step_key = key
    controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler, (key, 332))
    scheduler._elastic_graph_catalog = {
        key: {
            "cold_peak_bytes": 174,
            "hot_peak_bytes": 174,
            "resident_bytes": 174,
            "floor_bytes": 0,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
        }
    }
    controller.capture_envelopes = {key: (174, 0, 0)}
    controller.measured_bytes = {key: 174}
    controller.resident_bytes = 174
    scheduler.running = [object()] * 39
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 332
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    coordinator.set_elastic_external_memory.return_value = True
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {
        **{f"decode-{index}": 4 for index in range(38)},
        "prefill": 872,
    }
    output.total_num_scheduled_tokens = 1024
    output.num_spec_tokens_to_schedule = 3
    output.elastic_external_memory_bytes = 332
    if expected_loan is None:
        with pytest.raises(RuntimeError, match="residency exceeds its Graph loan"):
            _settle_elastic_with_receipt(scheduler, output, resident, 0, peak)
        assert controller.measured_bytes[key] == 174
        assert controller.max_pending_grant() == 332
        return
    _settle_elastic_with_receipt(scheduler, output, resident, 0, peak)
    assert controller.measured_bytes[key] == expected_loan
    assert scheduler._elastic_graph_catalog[key]["hot_peak_bytes"] == expected_loan
    assert controller.last_maintenance_step_key is None
    assert controller.resident_bytes == 174


def test_elastic_restore_hot_growth_expands_cold_recapture_envelope():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    key = (0, 3, 18, 64, 0)
    scheduler._elastic_admission_controller.step_key = key
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler, (key, 608))
    scheduler._elastic_graph_catalog = {
        key: {
            "cold_peak_bytes": 300,
            "hot_peak_bytes": 280,
            "resident_bytes": 280,
            "floor_bytes": 0,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
        }
    }
    owner_key = scheduler._elastic_graph_owner_key(key)
    scheduler._elastic_admission_controller.capture_envelopes = {owner_key: (300, 0, 0)}
    scheduler._elastic_admission_controller.measured_bytes = {key: 280}
    scheduler._elastic_admission_controller.resident_bytes = 280
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=64)
    scheduler.max_num_running_reqs = 18
    scheduler.running = [object() for _ in range(18)]

    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 608
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {
        **{f"decode-{index}": 1 for index in range(17)},
        "prefill": 47,
    }
    output.total_num_scheduled_tokens = 64
    output.num_spec_tokens_to_schedule = 3
    output.elastic_external_memory_bytes = 608
    _settle_elastic_with_receipt(scheduler, output, 320, 0, 320)

    row = scheduler._elastic_graph_catalog[key]
    assert row["hot_peak_bytes"] == 320
    assert row["cold_peak_bytes"] == 340
    assert scheduler._elastic_admission_controller.capture_envelopes[owner_key] == (
        340,
        0,
        0,
    )
    assert coordinator.elastic_external_memory_bytes == 320


def test_elastic_graph_same_owner_k_growth_adds_only_sampling_delta():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    k0_key = (0, 0, 12, 4036, 0)
    k3_key = (0, 3, 12, 4036, 0)
    owner_key = k0_key
    scheduler._elastic_admission_controller.step_key = (0, 3, 1, 109, 0)
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {
        owner_key: (1880000000, 32000000, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {k0_key: 1700000000}
    scheduler._elastic_admission_controller.floor_bytes = 32000000
    scheduler._elastic_admission_controller.resident_bytes = 1700000000
    scheduler._elastic_sampling_workspace_bytes_per_row = 248320 * 4
    scheduler._elastic_sampling_sort_workspace_bytes_per_row = 248320 * 45
    scheduler._elastic_sampling_sort_workspace_min_bytes = 22 << 20
    scheduler._elastic_sampling_triton_min_rows = 8
    scheduler.running = [object() for _ in range(12)]
    scheduler._elastic_restore_mode = False
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 1880000000
    coordinator.max_elastic_external_memory.return_value = 1960837120
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    # K changes the mandatory owner set. K0 evidence cannot authorize an
    # optimistic K3 product capture.
    with pytest.raises(RuntimeError, match="unpriced elastic CUDA Graph"):
        scheduler._plan_elastic_graph_loan(
            k3_key, minimum_free_primary_blocks=len(scheduler.running)
        )
    assert coordinator.elastic_external_memory_bytes == 1880000000

    # The reverse K3->K0 transition must return the same sampling delta before
    # the next step instead of treating a previous larger envelope as sticky.
    scheduler._elastic_admission_controller.clear_loans()
    scheduler._elastic_admission_controller.step_key = (0, 3, 1, 109, 0)
    coordinator.elastic_external_memory_bytes = 1880000000
    assert (
        scheduler._plan_elastic_graph_loan(
            k0_key, minimum_free_primary_blocks=len(scheduler.running)
        )
        == 1880000000
    )
    assert coordinator.elastic_external_memory_bytes == 1880000000


def test_elastic_graph_cold_unknown_uses_exact_prospective_tail():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("test"))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_admission_controller.step_key = (0, 3, 12, 4036, 0)
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.resident_bytes = 1880619008
    scheduler._elastic_admission_controller.floor_bytes = 980992
    scheduler._elastic_restore_mode = True
    scheduler._elastic_graph_discovery_base_bytes = 112 << 20
    scheduler._elastic_graph_discovery_bytes_per_token = 256 << 10
    scheduler._elastic_graph_capture_workspace_bytes = 672 << 20
    scheduler._elastic_sampling_workspace_bytes_per_row = 248320 * 4
    scheduler._elastic_sampling_sort_workspace_bytes_per_row = 248320 * 45
    scheduler._elastic_sampling_sort_workspace_min_bytes = 22 << 20
    scheduler._elastic_sampling_triton_min_rows = 8
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 1929379840
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    coordinator.max_elastic_external_memory.side_effect = lambda **kwargs: (
        1960837120 if kwargs["gdn_blocks"] == 100 else 2050000000
    )
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    fits_with_x32_reserve, required_with_x32, available_with_x32 = (
        scheduler._can_fund_elastic_graph_step(
            (0, 3, 20, 4039, 0),
            minimum_free_primary_blocks=21,
            minimum_attention_blocks=82,
            gdn_blocks=100,
        )
    )
    fits_x20, required_x20, available_x20 = scheduler._can_fund_elastic_graph_step(
        (0, 3, 20, 4039, 0),
        minimum_free_primary_blocks=21,
        minimum_attention_blocks=82,
        # gdn_initial=4 plus 20 requests * 3 blocks/request.
        gdn_blocks=64,
    )

    assert not fits_with_x32_reserve
    assert required_with_x32 == 1929379840
    assert required_x20 == 1929379840
    assert available_with_x32 == 1960837120
    assert not fits_x20
    assert available_x20 == 2050000000
    assert coordinator.max_elastic_external_memory.call_args_list == [
        call(
            minimum_free_primary_blocks=21,
            minimum_attention_blocks=82,
            gdn_blocks=100,
        ),
        call(
            minimum_free_primary_blocks=21,
            minimum_attention_blocks=82,
            gdn_blocks=64,
        ),
    ]


def test_elastic_mixed_step_requires_current_price_not_other_x_or_successor():
    """A mixed step cannot execute from its q=K+1 successor carrier.

    A catalog row for another X and the measured successor are both irrelevant
    to current M2 dispatch.
    """
    scheduler = _new_elastic_scheduler()
    source_key = (1, 3, 1, 4, 4)
    destination_key = (0, 3, 1, 2, 0)
    other_x_key = (0, 3, 2, 2, 0)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.step_key = source_key
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.recapture_pending_key = None
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {source_key: 266338304}
    _publish_elastic_step_hot(scheduler, source_key)
    scheduler._elastic_graph_catalog = {other_x_key: {"cold_peak_bytes": 176160768}}
    scheduler._elastic_admission_controller.floor_bytes = 19660800
    scheduler._elastic_admission_controller.resident_bytes = 265027584
    scheduler.running = [object()]

    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 266338304
    coordinator.max_elastic_external_memory.return_value = 5345640448
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert (
        scheduler._elastic_graph_carrier_closure_step_key(destination_key) == source_key
    )
    with pytest.raises(RuntimeError, match="unpriced elastic CUDA Graph"):
        scheduler._plan_elastic_graph_loan(
            destination_key,
            minimum_free_primary_blocks=len(scheduler.running),
        )
    assert coordinator.elastic_external_memory_bytes == 266338304
    coordinator.max_elastic_external_memory.assert_called_once_with(
        minimum_free_primary_blocks=1
    )


def test_elastic_changed_key_carries_current_residency_without_floor_subtraction():
    scheduler = _new_elastic_scheduler()
    source_key = (0, 3, 1, 128, 0)
    destination_key = (1, 3, 1, 4, 4)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.step_key = source_key
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    _publish_elastic_step_hot(scheduler, destination_key)
    scheduler._elastic_admission_controller.capture_envelopes = {
        destination_key: (239075328, 27242496, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {
        destination_key: 239075328
    }
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 335544320
    scheduler.running = [object()]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 335544320
    coordinator.max_elastic_external_memory.return_value = 5345640448
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert Scheduler._adjust_elastic_capture_envelope(
        (239075328, 27242496, 0), 0, 0
    ) == (239075328, 0, 0)
    assert scheduler._can_fund_elastic_graph_step(
        destination_key,
        minimum_free_primary_blocks=2,
        minimum_attention_blocks=1,
        gdn_blocks=7,
    ) == (True, 335544320, 5345640448)
    coordinator.max_elastic_external_memory.return_value = 300000000
    assert scheduler._can_fund_elastic_graph_step(
        destination_key,
        minimum_free_primary_blocks=2,
        minimum_attention_blocks=1,
        gdn_blocks=7,
    ) == (False, 335544320, 300000000)
    coordinator.max_elastic_external_memory.return_value = 5345640448
    assert (
        scheduler._plan_elastic_graph_loan(
            destination_key,
            minimum_free_primary_blocks=len(scheduler.running),
        )
        == 335544320
    )
    assert coordinator.elastic_external_memory_bytes == 335544320


def test_elastic_cold_piecewise_estimator_returns_raw_destination_endpoint():
    scheduler = _new_elastic_scheduler()
    destination_key = (0, 3, 1, 128, 0)
    scheduler._elastic_admission_controller.step_key = (1, 3, 1, 1, 1)
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {
        (0, 3, 0, 0, 0): (2447376384, 0, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.transition_floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 1463812096
    scheduler._elastic_graph_catalog_coverage = {"pinned_full_bytes": 1411383296}

    # The estimator owns only the independently measured destination endpoint.
    # Current residency and proved sharing are composed once by the admission
    # controller, regardless of catalog baseline coverage.
    assert scheduler._estimate_elastic_graph_step_bytes(destination_key) == (
        2447376384,
        True,
    )

    # Calibration already replaces a known estimate with the exact prospective
    # tail. Composing current residency before that replacement would make the
    # intermediate proven floor self-tax the MaxX search.
    scheduler._elastic_restore_mode = True
    assert scheduler._estimate_elastic_graph_step_bytes(destination_key) == (
        2447376384,
        True,
    )

    # Missing coverage cannot change endpoint ownership or silently pre-add the
    # live source set.
    scheduler._elastic_restore_mode = False
    scheduler._elastic_graph_catalog_coverage = {}
    assert scheduler._estimate_elastic_graph_step_bytes(destination_key) == (
        2447376384,
        True,
    )

    # A small exact row remains a raw endpoint. The planner later composes it
    # with current residency and the one transition-workspace unit.
    scheduler._elastic_graph_catalog_coverage = {"pinned_full_bytes": 1411383296}
    scheduler._elastic_admission_controller.capture_envelopes = {
        (0, 3, 0, 4, 0): (195035136, 0, 0)
    }
    assert scheduler._estimate_elastic_graph_step_bytes((0, 3, 1, 4, 0)) == (
        195035136,
        True,
    )


def test_elastic_fully_hot_owner_set_uses_exact_hot_replay_envelope():
    generation = RuntimeGeneration("hot-fast-path")
    scheduler = _new_elastic_scheduler(generation)
    step_key = (1, 3, 1, 4, 4)
    scheduler._elastic_restore_mode = False
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler._elastic_admission_controller.step_key = (1, 3, 1, 1, 1)
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {
        step_key: (3795845120, 0, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {step_key: 3795845120}
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.transition_floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 1612709888
    scheduler._elastic_admission_controller.cublas_workspace_bytes = 33554432
    scheduler._elastic_admission_controller.last_maintenance_step_key = step_key
    for key in resolve_step_physical_keys(step_key, generation):
        scheduler._elastic_admission_controller.publish_hot(
            key,
            GraphPrice(1, 1, f"test:{key.identity}"),
            pinned=True,
        )

    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        3795845120,
        False,
    )

    # The production failure had a post-maintenance owner set larger than the
    # historical graph component. Its first real replay may create one exact
    # CUDA-runtime cuBLAS workspace and must compose that incremental cost.
    scheduler._elastic_admission_controller.measured_bytes = {step_key: 192937984}
    scheduler._elastic_admission_controller.resident_bytes = 176140288
    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        209694720,
        False,
    )

    # A subsequent HOT replay already settled the workspace. It must not add
    # another unit or grow the loan on every token.
    scheduler._elastic_admission_controller.resident_bytes = 209694720
    scheduler._elastic_admission_controller.last_maintenance_step_key = None
    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        209694720,
        False,
    )

    # Without an exact replay measurement the physical residency remains the
    # only proven price; the HOT fast path still must not reopen capture.
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.resident_bytes = 1612709888
    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        1612709888,
        False,
    )

    # Calibration measures a HOT witness with a broad execution loan, but the
    # already-HOT physical owner set must not be relabelled as a new capture.
    # Its finite-wave preflight still includes the one measured workspace unit
    # needed by the first real replay after request-free maintenance.
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.last_maintenance_step_key = step_key
    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (
        1646264320,
        False,
    )


@pytest.mark.parametrize("sealed", [False, True])
def test_elastic_x2_to_x1_tail_composes_live_transition_floor_upper_bound(sealed):
    scheduler = _new_elastic_scheduler()
    source_key = (1, 3, 2, 8, 4)
    destination_key = (1, 3, 1, 4, 4)
    destination_envelope = 239075328
    current_residency = 219545600
    actual_failed_transition_peak = 240517120
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.step_key = source_key
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {
        destination_key: (destination_envelope, 48214016, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {
        destination_key: destination_envelope
    }
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_admission_controller.floor_bytes = 41287680
    scheduler._elastic_admission_controller.transition_floor_bytes = 110100480
    scheduler._elastic_admission_controller.resident_bytes = current_residency
    scheduler.running = [object()]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = current_residency
    coordinator.max_elastic_external_memory.return_value = 5 << 30
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: (
        ((value + (2 << 20) - 1) // (2 << 20)) * (2 << 20)
    )

    def set_external_memory(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = set_external_memory
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    scheduler._elastic_graph_catalog_coverage = (
        {"_catalog_source_sha256": "a" * 64} if sealed else {}
    )
    prices_before = copy.deepcopy(
        scheduler._elastic_admission_controller.capture_envelopes
    )
    grant = scheduler._plan_elastic_graph_loan(
        destination_key,
        minimum_free_primary_blocks=len(scheduler.running),
    )

    assert grant == 301989888
    assert grant >= actual_failed_transition_peak
    if sealed:
        assert (
            scheduler._elastic_admission_controller.capture_envelopes == prices_before
        )
    assert (
        scheduler._elastic_admission_controller.recapture_pending_key == destination_key
    )


def test_elastic_changed_key_estimator_excludes_transition_workspace_unit():
    scheduler = _new_elastic_scheduler()
    source_key = (1, 3, 2, 8, 4)
    destination_key = (1, 3, 1, 4, 4)
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.step_key = source_key
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {
        destination_key: (100, 10, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.floor_bytes = 10
    scheduler._elastic_admission_controller.transition_floor_bytes = 10
    scheduler._elastic_admission_controller.resident_bytes = 80
    scheduler._elastic_admission_controller.cublas_workspace_bytes = 32

    assert scheduler._estimate_elastic_graph_step_bytes(destination_key) == (
        100,
        True,
    )

    # A same-key replay already owns its workspace and must not grow forever.
    scheduler._elastic_admission_controller.step_key = destination_key
    scheduler._elastic_admission_controller.measured_bytes[destination_key] = 100
    _publish_elastic_step_hot(scheduler, destination_key)
    assert scheduler._estimate_elastic_graph_step_bytes(destination_key) == (
        100,
        False,
    )


def test_elastic_step5_cold_same_logical_owner_requires_recapture():
    """Physical COLD, not logical-key equality, must control recapture."""
    destination_key = (1, 3, 1, 4, 4)
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.step_key = destination_key
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {
        destination_key: (100, 10, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {destination_key: 100}
    scheduler._elastic_admission_controller.floor_bytes = 10
    scheduler._elastic_admission_controller.transition_floor_bytes = 10
    scheduler._elastic_admission_controller.resident_bytes = 80
    scheduler._elastic_admission_controller.cublas_workspace_bytes = 32

    assert scheduler._estimate_elastic_graph_step_bytes(destination_key) == (
        100,
        True,
    )


@pytest.mark.parametrize("price_role", ["unsealed", "serving", "calibration"])
def test_elastic_cold_growth_funds_immediate_same_key_hot_replay(price_role):
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    key = (1, 3, 1, 4, 4)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_admission_controller.step_key = key
    scheduler._elastic_admission_controller.recapture_pending_key = key
    _reset_elastic_loans(scheduler, (key, 337641472))
    scheduler._elastic_admission_controller.capture_envelopes = {
        key: (239075328, 27242496, 0)
    }
    scheduler._elastic_admission_controller.measured_bytes = {key: 239075328}
    scheduler._elastic_graph_catalog = {
        key: {
            "cold_peak_bytes": 239075328,
            "hot_peak_bytes": 239075328,
            "resident_bytes": 239054848,
            "floor_bytes": 27242496,
            "cold_observations": 2,
            "cold_stable_replays": 1,
            "hot_observations": 2,
            "hot_stable_replays": 1,
        }
    }
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 337641472
    scheduler.running = [object()]
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=64)
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 337641472
    coordinator.max_elastic_external_memory.return_value = 1 << 30
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: (
        ((value + (2 << 20) - 1) // (2 << 20)) * (2 << 20)
    )

    def set_external_memory(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = set_external_memory
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"request": 4}
    output.total_num_scheduled_tokens = 4
    output.num_spec_tokens_to_schedule = 3
    output.is_pure_decode_step = True
    output.elastic_external_memory_bytes = 337641472
    scheduler._elastic_graph_catalog_coverage = (
        {} if price_role == "unsealed" else {"_catalog_source_sha256": "a" * 64}
    )
    scheduler._elastic_restore_mode = price_role == "calibration"
    catalog_before = copy.deepcopy(scheduler._elastic_graph_catalog)
    prices_before = copy.deepcopy(
        scheduler._elastic_admission_controller.capture_envelopes
    )
    _settle_elastic_with_receipt(
        scheduler,
        output,
        267124736,
        19660800,
        267124736,
        publish_step_key=key,
    )

    assert scheduler._elastic_admission_controller.measured_bytes[key] == (
        239075328 if price_role == "serving" else 268435456
    )
    if price_role == "serving":
        assert scheduler._elastic_graph_catalog == catalog_before
        assert (
            scheduler._elastic_admission_controller.capture_envelopes == prices_before
        )
    assert scheduler._elastic_admission_controller.recapture_pending_key is None
    # The next assertion exercises serving replay, not calibration's explicit
    # full-tail measurement loan.
    scheduler._elastic_restore_mode = False
    assert (
        scheduler._plan_elastic_graph_loan(
            key, minimum_free_primary_blocks=len(scheduler.running)
        )
        == 268435456
    )


def test_elastic_piecewise_cross_x_floor_rejects_insufficient_calibration_tail():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.step_key = (0, 0, 7, 4096, 0)
    scheduler._elastic_admission_controller.recapture_pending_key = None
    _reset_elastic_loans(scheduler)
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_graph_catalog = {(0, 0, 7, 4096, 0): {"cold_peak_bytes": 600}}
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 0
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    coordinator.max_elastic_external_memory.return_value = 500
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._can_fund_elastic_graph_step(
        (0, 0, 22, 4096, 0),
        minimum_free_primary_blocks=22,
        gdn_blocks=67,
    ) == (False, 600, 500)


def test_restore_admission_failure_snapshot_is_read_only(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(max_num_seqs=3)
    manager = scheduler.kv_cache_manager
    before = manager.block_pool.get_num_free_blocks()
    rejection = ((0, 3, 3, 1024, 0), 201, 200)
    scheduler._elastic_last_graph_admission_rejection = rejection
    tokens = {"base": 4}
    result = scheduler._elastic_restore_admission_diagnostic(
        "graph_funding_rejected", 4092, tokens
    )
    assert result["graph_rejection"] == rejection
    assert result["allocation_rejection"] is None
    assert result["stop_reason"] == "graph_funding_rejected"
    assert result["gdn"] is None
    assert result["primary"]["free"] == before
    assert result["scheduled_tokens"] == tokens
    result["scheduled_tokens"].clear()
    assert tokens == {"base": 4}
    assert manager.block_pool.get_num_free_blocks() == before
    scheduler._elastic_last_graph_admission_rejection = None
    recovered = scheduler._elastic_restore_admission_diagnostic(None, 4096, {})
    assert recovered["graph_rejection"] is None
    assert result["graph_rejection"] == rejection


def test_elastic_graph_maxx_defers_waiting_request_before_kv_allocation(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_restore_mode = True
    requests = create_requests(num_requests=3, num_tokens=4, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)

    scheduler.elastic_on_demand_graphs = True

    def can_fund_impl(key, **_kwargs):
        if key is not None and key[2] <= 2:
            for physical_key in resolve_step_physical_keys(
                key, _elastic_controller(scheduler).generation
            ):
                scheduler._elastic_admission_controller.publish_hot(
                    physical_key,
                    GraphPrice(1, 1, f"test:{physical_key.identity}"),
                    pinned=physical_key.logical.mode == "FULL",
                )
            return True, 100, 200
        return False, 201, 200

    can_fund = Mock(side_effect=can_fund_impl)
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 2
    assert len(scheduler.running) == 2
    assert list(scheduler.waiting) == [requests[2]]
    assert [call.args[0][2] for call in can_fund.call_args_list] == [1, 2, 3]


def test_elastic_mm_rejection_preserves_encoder_cache_before_kv_allocation(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_seqs=1,
        max_num_batched_tokens=256,
    )
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_mm_activation_loan_bytes = 32
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=200)
    old_request = create_requests(
        num_requests=1,
        num_tokens=70,
        mm_hashes_list=[["old-image"]],
        mm_positions=[[PlaceholderRange(offset=0, length=50)]],
        req_ids=["old"],
    )[0]
    request = create_requests(
        num_requests=1,
        num_tokens=170,
        mm_hashes_list=[["cached-image", "new-image"]],
        mm_positions=[
            [
                PlaceholderRange(offset=0, length=50),
                PlaceholderRange(offset=50, length=100),
            ]
        ],
        req_ids=["new"],
    )[0]
    scheduler.encoder_cache_manager.allocate(request, 0)
    scheduler.encoder_cache_manager.allocate(old_request, 0)
    scheduler.encoder_cache_manager.free(old_request)
    scheduler.add_request(request)
    cached_before = {
        key: set(value) for key, value in scheduler.encoder_cache_manager.cached.items()
    }
    freeable_before = scheduler.encoder_cache_manager.freeable.copy()
    can_fund = Mock(return_value=(False, 201, 200))
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)
    allocate_slots = Mock(side_effect=AssertionError("KV allocation crossed preflight"))
    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", allocate_slots)

    output = scheduler.schedule()

    assert not output.num_scheduled_tokens
    assert scheduler.encoder_cache_manager.cached == cached_before
    assert scheduler.encoder_cache_manager.freeable == freeable_before
    assert scheduler.encoder_cache_manager.get_freed_mm_hashes() == []
    assert scheduler.encoder_cache_manager.get_cached_input_ids(request) == {0}
    assert request in scheduler.waiting
    assert can_fund.call_args.kwargs["mm_activation_loan_bytes"] == 32
    allocate_slots.assert_not_called()


def test_restore_wave_consumes_only_its_prepared_execution_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_restore_wave_target = 3
    requests = create_requests(num_requests=3, num_tokens=4, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    prepared_key = (0, 3, 3, 16, 0)
    scheduler._elastic_restore_wave_step_key = prepared_key
    for physical_key in scheduler._resolve_elastic_step_physical_keys(
        scheduler._elastic_graph_carrier_closure_step_key(prepared_key)
    ):
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(1, 1, f"test:{physical_key.identity}"),
            pinned=physical_key.logical.mode == "FULL",
        )
    can_fund = Mock()
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 3
    can_fund.assert_not_called()
    assert scheduler._elastic_restore_wave_target == 0


def test_restore_wave_cannot_expand_beyond_prepared_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    scheduler._elastic_restore_mode = True
    scheduler._elastic_restore_wave_target = 2
    requests = create_requests(num_requests=3, num_tokens=4, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)

    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    prepared_key = (0, 3, 2, 8, 0)
    scheduler._elastic_restore_wave_step_key = prepared_key
    for physical_key in scheduler._resolve_elastic_step_physical_keys(
        scheduler._elastic_graph_carrier_closure_step_key(prepared_key)
    ):
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(1, 1, f"test:{physical_key.identity}"),
            pinned=physical_key.logical.mode == "FULL",
        )
    can_fund = Mock()
    monkeypatch.setattr(scheduler, "_can_fund_elastic_graph_step", can_fund)
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 2
    assert list(scheduler.waiting) == [requests[2]]
    can_fund.assert_not_called()
    assert scheduler._elastic_restore_wave_target == 0


def test_elastic_candidate_lease_stops_without_immediate_lookahead(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(max_num_seqs=3, max_num_batched_tokens=64)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_restore_mode = True
    requests = create_requests(num_requests=3, num_tokens=4, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)

    scheduler.kv_cache_manager.kv_cache_config.elastic_mapping_quantum = 1
    candidate_lease = Mock(side_effect=[2, 1])
    monkeypatch.setattr(
        scheduler,
        "_apply_elastic_waiting_candidate",
        candidate_lease,
    )

    final_key = scheduler._canonical_elastic_graph_step_key(
        {request.request_id: 4 for request in requests[:2]},
        0,
        False,
    )
    assert final_key is not None
    for physical_key in resolve_step_physical_keys(
        final_key, _elastic_controller(scheduler).generation
    ):
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(1, 1, f"test:{physical_key.identity}"),
            pinned=physical_key.logical.mode == "FULL",
        )
    monkeypatch.setattr(
        scheduler,
        "_can_fund_elastic_graph_step",
        Mock(return_value=(True, 100, 200)),
    )
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 2
    assert len(scheduler.running) == 2
    assert list(scheduler.waiting) == [requests[2]]
    assert candidate_lease.call_count == 2


def test_elastic_restore_wave_does_not_replace_atomic_admission(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(max_num_seqs=2, max_num_batched_tokens=64)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_restore_wave_target = 2
    requests = create_requests(num_requests=2, num_tokens=4, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)

    scheduler.kv_cache_manager.kv_cache_config.elastic_mapping_quantum = 1
    candidate_lease = Mock(
        side_effect=AssertionError("atomic calibration admission must not be replaced")
    )
    monkeypatch.setattr(
        scheduler,
        "_apply_elastic_waiting_candidate",
        candidate_lease,
    )

    final_key = scheduler._canonical_elastic_graph_step_key(
        {request.request_id: 4 for request in requests},
        scheduler.num_spec_tokens,
        False,
    )
    assert final_key is not None
    for physical_key in resolve_step_physical_keys(
        final_key,
        _elastic_controller(scheduler).generation,
        scheduler.scheduler_config.max_num_batched_tokens,
        scheduler._elastic_compiled_piecewise_sizes,
        scheduler._elastic_graph_execution_policy,
    ):
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(1, 1, f"test:{physical_key.identity}"),
            pinned=physical_key.logical.mode == "FULL",
        )
    monkeypatch.setattr(
        scheduler,
        "_can_fund_elastic_graph_step",
        Mock(return_value=(True, 100, 200)),
    )
    monkeypatch.setattr(
        scheduler,
        "_plan_elastic_graph_loan",
        Mock(return_value=0),
    )

    output = scheduler.schedule()

    assert len(output.scheduled_new_reqs) == 2
    assert len(scheduler.running) == 2
    candidate_lease.assert_not_called()


def test_calibration_cold_owner_cannot_cross_request_commit_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(max_num_seqs=1, max_num_batched_tokens=64)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_restore_mode = True
    scheduler.add_request(create_requests(num_requests=1, num_tokens=4)[0])

    # Simulate a broken preflight that claims the request fits without first
    # publishing its physical owner set. The final commit barrier must catch
    # the COLD plan in every runtime generation, including CALIBRATING.
    monkeypatch.setattr(
        scheduler,
        "_can_fund_elastic_graph_step",
        Mock(return_value=(True, 100, 200)),
    )
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    with pytest.raises(RuntimeError, match="crossed the scheduler commit boundary"):
        scheduler.schedule()


def test_tight_hot_commit_fallback_only_acquires_leases(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(max_num_seqs=1, max_num_batched_tokens=64)
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler.add_request(create_requests(num_requests=1, num_tokens=4)[0])

    final_key = scheduler._canonical_elastic_graph_step_key({"0": 4}, 0, False)
    assert final_key is not None
    current_keys = scheduler._resolve_elastic_step_physical_keys(final_key)
    assert current_keys
    for physical_key in current_keys:
        scheduler._elastic_admission_controller.publish_hot(
            physical_key,
            GraphPrice(1, 1, f"current:{physical_key.identity}"),
            pinned=False,
        )
    unrelated = PhysicalReplayKey(
        logical=dataclasses.replace(
            current_keys[0].logical,
            token_bucket=current_keys[0].logical.token_bucket + 1,
        ),
        physical_num_reqs=current_keys[0].physical_num_reqs,
        generation=current_keys[0].generation,
    )
    scheduler._elastic_admission_controller.publish_hot(
        unrelated, GraphPrice(1, 1, "unrelated"), pinned=False
    )
    scheduler._elastic_admission_controller.install_reclaim_group(
        ReclaimGroup("unrelated", (unrelated,), 1)
    )
    scheduler._elastic_admission_controller.resident_bytes = 100
    monkeypatch.setattr(
        scheduler,
        "_can_fund_elastic_graph_step",
        Mock(return_value=(True, 100, 100)),
    )
    monkeypatch.setattr(
        scheduler,
        "_estimate_elastic_graph_step_bytes",
        Mock(return_value=(80, False)),
    )
    monkeypatch.setattr(
        scheduler.kv_cache_manager.coordinator,
        "max_elastic_external_memory",
        Mock(return_value=100),
    )
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=100))

    output = scheduler.schedule()

    assert output.elastic_step_plan is not None
    assert output.elastic_step_plan.kind == ElasticPlanKind.USER
    assert output.elastic_step_plan.victim_keys == ()
    assert scheduler._elastic_admission_controller.entries[unrelated].hot


def test_new_pressure_reclaim_preflight_forces_same_output_to_zero_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    """A plan armed during preflight owns this tick before request mutation."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(max_num_seqs=2, max_num_batched_tokens=64)
    request = create_requests(num_requests=1, num_tokens=4, max_tokens=1)[0]
    scheduler.add_request(request)
    scheduler.elastic_on_demand_graphs = True
    monkeypatch.setattr(
        scheduler,
        "_preflight_elastic_running_text_wave",
        Mock(return_value=(None, False)),
    )
    monkeypatch.setattr(
        scheduler,
        "_preflight_elastic_waiting_text_wave",
        Mock(return_value=(None, True, ())),
    )
    candidate = Mock(side_effect=AssertionError("request mutation crossed X0"))
    monkeypatch.setattr(scheduler, "_apply_elastic_waiting_candidate", candidate)

    output = scheduler.schedule(physical_quiescent=True)

    assert output.total_num_scheduled_tokens == 0
    assert request in scheduler.waiting
    assert not scheduler.running
    candidate.assert_not_called()


def test_elastic_candidate_maps_complete_available_waiting_cohort():
    scheduler = _new_elastic_scheduler()
    scheduler.running = [object() for _ in range(12)]
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 32
    requests = [Mock() for _ in range(20)]
    request = requests[0]
    scheduler._elastic_schedulable_waiting_snapshot = Mock(return_value=tuple(requests))
    scheduler._elastic_graph_catalog_coverage = {}
    scheduler._elastic_admission_controller.pinned_resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.resident_bytes = 0
    _clear_elastic_maintenance(scheduler)
    coordinator = Mock()
    coordinator.apply_elastic_admission_wave.return_value = 20
    kv_cache_manager = Mock()
    kv_cache_manager.kv_cache_config.elastic_mapping_quantum = 2 << 20
    kv_cache_manager.watermark_blocks = 1
    kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = Mock(
        primary=7
    )
    kv_cache_manager.coordinator = coordinator
    scheduler.kv_cache_manager = kv_cache_manager

    assert (
        scheduler._apply_elastic_waiting_candidate(
            request,
            token_budget=4096,
            waiting_count=64,
        )
        == 20
    )
    expected = (8, *(7 for _ in range(19)))
    coordinator.plan_elastic_admission_wave.assert_not_called()
    coordinator.apply_elastic_admission_wave.assert_called_once_with(expected)


def test_elastic_running_tail_arms_pressure_reclaim_at_physical_quiescence():
    """A late admissible tail is atomic even after one request became RUNNING.

    This is the deterministic form of the rejected product X39 trace: the
    first request crossed the commit boundary before the other 38 requests.
    Retained optional Graph bytes make the current layout expose only one new
    slot, while a pressure reclaim fits the complete tail. Admitting that one
    slot would serialize a workload whose sealed capacity is X39. Logical
    RUNNING membership is not a physical replay/lease proof.
    """
    scheduler = _new_elastic_scheduler()
    scheduler.running = [object()]
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 39
    requests = [Mock() for _ in range(38)]
    request = requests[0]
    scheduler._elastic_schedulable_waiting_snapshot = Mock(return_value=tuple(requests))
    scheduler._elastic_graph_catalog_coverage = {"pinned_full_bytes": 10}
    _clear_elastic_maintenance(scheduler)
    scheduler.elastic_on_demand_graphs = True
    scheduler._next_elastic_transaction_id = Mock(return_value="late-tail")
    controller = scheduler._elastic_admission_controller
    physical_keys = scheduler._resolve_elastic_step_physical_keys((1, 3, 17, 17, 1))
    for physical_key in physical_keys:
        controller.publish_hot(
            physical_key,
            GraphPrice(8, 8, f"retained:{physical_key.identity}"),
            pinned=physical_key.logical.mode == "FULL",
        )
    controller.pinned_resident_bytes = 8
    controller.floor_bytes = 2
    controller.resident_bytes = 100
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=1),
        SimpleNamespace(max_requests=1),
        SimpleNamespace(max_requests=38),
    ]
    kv_cache_manager = Mock()
    kv_cache_manager.kv_cache_config.elastic_mapping_quantum = 2 << 20
    kv_cache_manager.watermark_blocks = 1
    kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = Mock(
        primary=7
    )
    kv_cache_manager.coordinator = coordinator
    scheduler.kv_cache_manager = kv_cache_manager

    assert (
        scheduler._apply_elastic_waiting_candidate(
            request,
            token_budget=4096,
            waiting_count=38,
            physical_quiescent=True,
        )
        == 0
    )
    expected = (8, *(7 for _ in range(37)))
    coordinator.plan_elastic_admission_wave.assert_has_calls(
        [
            call(expected),
            call(expected, external_memory_bytes=12),
            call(expected, external_memory_bytes=2),
        ]
    )
    coordinator.apply_elastic_admission_wave.assert_not_called()
    reclaim = _pending_elastic_maintenance(scheduler)
    assert reclaim is not None
    assert reclaim.kind == ElasticPlanKind.PRESSURE_RECLAIM
    assert set(reclaim.victim_keys) == set(physical_keys)
    assert controller.stats.deferrals == 0


def test_elastic_running_intrinsic_overcap_keeps_bounded_partial_progress():
    """Do not turn a true full-context X1 capacity into eternal deferral."""
    scheduler = _new_elastic_scheduler()
    scheduler.running = [object()]
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 2
    request = Mock()
    scheduler._elastic_schedulable_waiting_snapshot = Mock(return_value=(request,))
    scheduler._elastic_graph_catalog_coverage = {"pinned_full_bytes": 10}
    scheduler._elastic_admission_controller.pinned_resident_bytes = 8
    scheduler._elastic_admission_controller.floor_bytes = 2
    scheduler._elastic_admission_controller.resident_bytes = 100
    _clear_elastic_maintenance(scheduler)
    scheduler.elastic_on_demand_graphs = True
    controller = scheduler._elastic_admission_controller
    controller.plan_reclaim_all = Mock(wraps=controller.plan_reclaim_all)
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=1),
        SimpleNamespace(max_requests=1),
        SimpleNamespace(max_requests=1),
    ]
    coordinator.apply_elastic_admission_wave.return_value = 1
    kv_cache_manager = Mock()
    kv_cache_manager.kv_cache_config.elastic_mapping_quantum = 2 << 20
    kv_cache_manager.watermark_blocks = 1
    kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = Mock(
        primary=36
    )
    kv_cache_manager.coordinator = coordinator
    scheduler.kv_cache_manager = kv_cache_manager

    assert (
        scheduler._apply_elastic_waiting_candidate(
            request,
            token_budget=4096,
            waiting_count=1,
        )
        == 1
    )
    coordinator.plan_elastic_admission_wave.assert_has_calls(
        [
            call((37,)),
            call((37,), external_memory_bytes=12),
            call((37,), external_memory_bytes=2),
        ]
    )
    coordinator.apply_elastic_admission_wave.assert_called_once_with((37,))
    controller.plan_reclaim_all.assert_not_called()


def test_elastic_idle_candidate_reclaims_when_it_increases_wave_maxx():
    scheduler = _new_elastic_scheduler()
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 40
    requests = [Mock() for _ in range(3)]
    request = requests[0]
    scheduler._elastic_schedulable_waiting_snapshot = Mock(return_value=tuple(requests))
    scheduler._elastic_graph_catalog_coverage = {"pinned_full_bytes": 10}
    scheduler._elastic_admission_controller.pinned_resident_bytes = 8
    scheduler._elastic_admission_controller.floor_bytes = 2
    scheduler._elastic_admission_controller.resident_bytes = 100
    _clear_elastic_maintenance(scheduler)
    scheduler.elastic_on_demand_graphs = True
    controller = scheduler._elastic_admission_controller
    for physical_key in scheduler._resolve_elastic_step_physical_keys((0, 3, 1, 4, 0)):
        group_id = f"idle-wave:{physical_key.identity}"
        controller.publish_hot(
            physical_key,
            GraphPrice(1, 1, group_id),
            pinned=physical_key.logical.mode == "FULL",
        )
        if physical_key.logical.mode == "PIECEWISE":
            controller.install_reclaim_group(
                ReclaimGroup(group_id, (physical_key,), reclaimable_bytes=1)
            )
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=1),
        SimpleNamespace(max_requests=3),
        SimpleNamespace(max_requests=3),
    ]
    kv_cache_manager = Mock()
    kv_cache_manager.kv_cache_config.elastic_mapping_quantum = 2 << 20
    kv_cache_manager.watermark_blocks = 0
    kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = Mock(
        primary=7
    )
    kv_cache_manager.coordinator = coordinator
    scheduler.kv_cache_manager = kv_cache_manager

    assert (
        scheduler._apply_elastic_waiting_candidate(
            request,
            token_budget=4096,
            waiting_count=3,
        )
        == 0
    )
    coordinator.plan_elastic_admission_wave.assert_has_calls(
        [call((7, 7, 7)), call((7, 7, 7), external_memory_bytes=12)]
    )
    coordinator.apply_elastic_admission_wave.assert_not_called()
    reclaim = _pending_elastic_maintenance(scheduler)
    assert reclaim is not None and reclaim.kind == ElasticPlanKind.RECLAIM


def test_elastic_idle_prefix_reclaims_when_retained_state_reduces_declared_maxx():
    scheduler = _new_elastic_scheduler()
    scheduler.running = []
    controller = scheduler._elastic_admission_controller
    physical_key = scheduler._resolve_elastic_step_physical_keys((0, 3, 1, 4, 0))[0]
    group_id = f"declared-wave:{physical_key.identity}"
    controller.publish_hot(
        physical_key,
        GraphPrice(40, 40, group_id),
        pinned=False,
    )
    controller.install_reclaim_group(
        ReclaimGroup(group_id, (physical_key,), reclaimable_bytes=40)
    )
    controller.resident_bytes = 100
    scheduler._elastic_graph_catalog_coverage = {"pinned_full_bytes": 60}
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=34),
        SimpleNamespace(max_requests=35),
        SimpleNamespace(max_requests=35),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._prepare_elastic_waiting_deficit_reclaim(
        (7,) * 34, declared_wave_size=39
    )
    coordinator.plan_elastic_admission_wave.assert_has_calls(
        [
            call((7,) * 35),
            call((7,) * 35, external_memory_bytes=60),
            call((7,) * 35, external_memory_bytes=0),
        ]
    )
    reclaim = _pending_elastic_maintenance(scheduler)
    assert reclaim is not None and reclaim.kind == ElasticPlanKind.RECLAIM


def test_elastic_idle_prefix_keeps_hotset_when_declared_maxx_already_fits():
    scheduler = _new_elastic_scheduler()
    scheduler.running = []
    scheduler._elastic_admission_controller.resident_bytes = 100
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=2),
        SimpleNamespace(max_requests=2),
        SimpleNamespace(max_requests=2),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert not scheduler._prepare_elastic_waiting_deficit_reclaim(
        (7,), declared_wave_size=39, physical_quiescent=False
    )
    coordinator.plan_elastic_admission_wave.assert_has_calls(
        [
            call((7, 7)),
            call((7, 7), external_memory_bytes=0),
            call((7, 7), external_memory_bytes=0),
        ]
    )
    assert _pending_elastic_maintenance(scheduler) is None
    assert scheduler._elastic_admission_controller.stats.deferrals == 0


def test_elastic_idle_prefix_administratively_drains_pinned_growth_for_maxx():
    scheduler = _new_elastic_scheduler()
    scheduler.running = []
    controller = scheduler._elastic_admission_controller
    physical_keys = scheduler._resolve_elastic_step_physical_keys((1, 3, 17, 17, 1))
    for physical_key in physical_keys:
        group_id = f"retained-x17:{physical_key.identity}"
        controller.publish_hot(
            physical_key,
            GraphPrice(40, 40, group_id),
            pinned=physical_key.logical.mode == "FULL",
        )
        if physical_key.logical.mode != "FULL":
            controller.install_reclaim_group(
                ReclaimGroup(group_id, (physical_key,), reclaimable_bytes=40)
            )
    controller.resident_bytes = 100
    controller.floor_bytes = 4
    scheduler._elastic_graph_catalog_coverage = {"pinned_full_bytes": 60}
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=39),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._prepare_elastic_waiting_deficit_reclaim(
        (7,) * 38, declared_wave_size=39
    )

    expected = (7,) * 39
    coordinator.plan_elastic_admission_wave.assert_has_calls(
        [
            call(expected),
            call(expected, external_memory_bytes=64),
            call(expected, external_memory_bytes=4),
        ]
    )
    reclaim = _pending_elastic_maintenance(scheduler)
    assert reclaim is not None
    assert reclaim.kind == ElasticPlanKind.PRESSURE_RECLAIM
    assert set(reclaim.victim_keys) == set(physical_keys)
    assert all(controller.entries[key].hot for key in physical_keys)


def test_elastic_pressure_reclaim_preserves_and_prices_serving_carrier():
    scheduler = _new_elastic_scheduler()
    scheduler.running = []
    controller = scheduler._elastic_admission_controller
    physical_keys = scheduler._resolve_elastic_step_physical_keys((1, 3, 1, 4, 4))
    carrier_key, *legacy_keys = physical_keys
    for physical_key in physical_keys:
        controller.publish_hot(
            physical_key,
            GraphPrice(20, 20, f"pressure:{physical_key.identity}"),
            pinned=True,
        )
    controller.resident_bytes = 100
    controller.floor_bytes = 4
    scheduler._elastic_serving_carrier_keys = (carrier_key,)
    scheduler._elastic_serving_carrier_resident_bytes = 40
    scheduler._elastic_graph_catalog_coverage = {"pinned_full_bytes": 60}
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=39),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._prepare_elastic_waiting_deficit_reclaim(
        (7,) * 38, declared_wave_size=39
    )

    expected = (7,) * 39
    coordinator.plan_elastic_admission_wave.assert_has_calls(
        [
            call(expected),
            call(expected, external_memory_bytes=64),
            call(expected, external_memory_bytes=24),
        ]
    )
    reclaim = _pending_elastic_maintenance(scheduler)
    assert reclaim is not None
    assert reclaim.kind == ElasticPlanKind.PRESSURE_RECLAIM
    assert reclaim.protected_keys == (carrier_key,)
    assert set(reclaim.victim_keys) == set(legacy_keys)


def test_elastic_short_decode_minimum_reuses_x2_for_semantic_x1():
    scheduler = object.__new__(Scheduler)
    scheduler._elastic_restore_mode = False
    scheduler._elastic_terminal_decode_carrier_x = 2
    scheduler._elastic_short_decode_inventory = {1: (), 2: ()}
    scheduler._elastic_short_decode_min_x = 2

    assert scheduler._elastic_short_decode_physical_x(1) == 2
    assert scheduler._elastic_short_decode_physical_x(2) == 2


def test_elastic_serving_carrier_is_only_shared_terminal_mtp_owner():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_graph_catalog_coverage = {"serving_carrier_owner": "mtp_decode"}

    carrier = scheduler.resolve_elastic_serving_carrier_physical_keys(
        ((1, 3, 37, 37, 1), (0, 3, 37, 148, 4))
    )

    assert len(carrier) == 1
    assert carrier[0].logical.owner == "mtp_decode"
    assert carrier[0].physical_num_reqs == 37


def test_elastic_serving_carrier_rejects_non_shared_mtp_geometry():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_graph_catalog_coverage = {"serving_carrier_owner": "mtp_decode"}

    with pytest.raises(RuntimeError, match="do not share exactly one"):
        scheduler.resolve_elastic_serving_carrier_physical_keys(
            ((1, 3, 36, 36, 1), (0, 3, 37, 148, 4))
        )


@pytest.mark.parametrize(
    "step_key",
    ((1, 3, 1, 1, 1), (0, 3, 37, 148, 4)),
)
def test_elastic_product_intent_separates_execution_from_mtp_recovery_carrier(
    step_key,
):
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = False
    scheduler._elastic_terminal_decode_carrier_x = 37
    scheduler._elastic_graph_catalog_coverage = {"serving_carrier_owner": "mtp_decode"}
    carrier = scheduler.resolve_elastic_serving_carrier_physical_keys(
        ((1, 3, 37, 37, 1), (0, 3, 37, 148, 4))
    )
    controller = scheduler._elastic_admission_controller
    controller.publish_hot(carrier[0], GraphPrice(94, 94, "terminal-mtp"), pinned=True)
    scheduler._elastic_serving_carrier_keys = carrier

    current, _successor, protected = scheduler._elastic_step_residency_intent(step_key)

    assert sum(key.logical.owner == "mtp_decode" for key in current) == 1
    assert carrier[0] in protected
    current_mtp = next(key for key in current if key.logical.owner == "mtp_decode")
    if step_key[2] < 37:
        assert current_mtp.physical_num_reqs == 1
        assert current_mtp != carrier[0]
        assert sum(key.logical.owner == "mtp_decode" for key in protected) == 2
    else:
        assert current_mtp == carrier[0]
        assert sum(key.logical.owner == "mtp_decode" for key in protected) == 1


def test_elastic_k0_execution_protects_terminal_mtp_as_successor_intent():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = False
    scheduler._elastic_terminal_decode_carrier_x = 37
    scheduler._elastic_graph_catalog_coverage = {"serving_carrier_owner": "mtp_decode"}
    carrier = scheduler.resolve_elastic_serving_carrier_physical_keys(
        ((1, 3, 37, 37, 1), (0, 3, 37, 148, 4))
    )
    scheduler._elastic_admission_controller.publish_hot(
        carrier[0], GraphPrice(94, 94, "terminal-mtp"), pinned=True
    )
    scheduler._elastic_serving_carrier_keys = carrier

    current, successor, protected = scheduler._elastic_step_residency_intent(
        (1, 0, 1, 1, 1)
    )

    assert [key.logical.owner for key in current] == ["target"]
    assert set(successor) == set((*current, *carrier))
    assert set(protected) == set((*current, *carrier))


def test_force_non_speculative_schedule_keeps_terminal_mtp_out_of_current_dispatch(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=64,
        num_speculative_tokens=3,
    )
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler._elastic_terminal_decode_carrier_x = 37
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    scheduler._elastic_graph_catalog_coverage = {"serving_carrier_owner": "mtp_decode"}
    carrier = scheduler.resolve_elastic_serving_carrier_physical_keys(
        ((1, 3, 37, 37, 1), (0, 3, 37, 148, 4))
    )
    scheduler._elastic_admission_controller.publish_hot(
        carrier[0], GraphPrice(94, 94, "terminal-mtp"), pinned=True
    )
    scheduler._elastic_serving_carrier_keys = carrier
    request = create_requests(num_requests=1, num_tokens=1, max_tokens=1)[0]
    request.force_non_speculative = True
    scheduler.add_request(request)

    def fund_current(key, **kwargs):
        if not kwargs.get("preview_only"):
            for physical_key in scheduler._resolve_elastic_step_physical_keys(key):
                scheduler._elastic_admission_controller.publish_hot(
                    physical_key,
                    GraphPrice(1, 1, f"test:{physical_key.identity}"),
                    pinned=False,
                )
        return True, 1, 100

    monkeypatch.setattr(
        scheduler, "_can_fund_elastic_graph_step", Mock(side_effect=fund_current)
    )
    monkeypatch.setattr(scheduler, "_plan_elastic_graph_loan", Mock(return_value=0))

    def reserve_current(key, **kwargs):
        scheduler._elastic_preflight_admission_grant = SimpleNamespace(
            step_key=key,
            physical_keys=scheduler._elastic_step_residency_intent(key)[2],
            maintenance_transaction_id=None,
            external_memory_bytes=kwargs["external_memory_bytes"],
            previous_external_memory_bytes=0,
            minimum_free_primary_blocks=kwargs["minimum_free_primary_blocks"],
            requirements=kwargs["requirements"],
        )
        return True

    monkeypatch.setattr(scheduler, "_reserve_elastic_admission", reserve_current)

    output = scheduler.schedule()

    assert output.elastic_graph_step_key is not None
    assert output.elastic_graph_step_key[1] == 0
    assert output.elastic_step_plan is not None
    assert [
        dispatch.invocation.owner
        for dispatch in output.elastic_step_plan.current_dispatch
    ] == ["target"]
    assert carrier[0] in output.elastic_step_plan.successor_keys
    assert carrier[0] in output.elastic_step_plan.protected_keys


def test_elastic_idle_drain_requires_administrative_capacity_to_fit_wave():
    scheduler = _new_elastic_scheduler()
    scheduler.running = []
    controller = scheduler._elastic_admission_controller
    physical_key = scheduler._resolve_elastic_step_physical_keys((1, 3, 8, 8, 1))[-1]
    controller.publish_hot(
        physical_key,
        GraphPrice(40, 40, f"holdout:{physical_key.identity}"),
        pinned=True,
    )
    controller.resident_bytes = 40
    controller.floor_bytes = 2
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=19),
        SimpleNamespace(max_requests=20),
        SimpleNamespace(max_requests=23),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert not scheduler._prepare_elastic_waiting_deficit_reclaim(
        (5,) * 23, declared_wave_size=24
    )

    assert _pending_elastic_maintenance(scheduler) is None
    assert controller.entries[physical_key].pinned


def test_elastic_active_wave_arms_pressure_reclaim_when_physically_quiescent():
    scheduler = _new_elastic_scheduler()
    scheduler.running = [object()]
    controller = scheduler._elastic_admission_controller
    controller.resident_bytes = 100
    controller.pinned_resident_bytes = 60
    controller.floor_bytes = 4
    scheduler._next_elastic_transaction_id = Mock(return_value="active-x39")
    physical_key = scheduler._resolve_elastic_step_physical_keys((1, 3, 17, 17, 1))[0]
    controller.publish_hot(
        physical_key,
        GraphPrice(100, 100, f"active:{physical_key.identity}"),
        pinned=True,
    )
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=39),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._prepare_elastic_waiting_deficit_reclaim(
        (7,) * 39, physical_quiescent=True
    )

    reclaim = _pending_elastic_maintenance(scheduler)
    assert reclaim is not None
    assert reclaim.kind == ElasticPlanKind.PRESSURE_RECLAIM
    assert reclaim.victim_keys == (physical_key,)


def test_elastic_active_wave_defers_pressure_reclaim_with_worker_inflight():
    scheduler = _new_elastic_scheduler()
    scheduler.running = [object()]
    controller = scheduler._elastic_admission_controller
    controller.resident_bytes = 100
    controller.pinned_resident_bytes = 60
    controller.floor_bytes = 4
    scheduler._next_elastic_transaction_id = Mock(return_value="inflight-x39")
    physical_key = scheduler._resolve_elastic_step_physical_keys((1, 3, 17, 17, 1))[0]
    controller.publish_hot(
        physical_key,
        GraphPrice(100, 100, f"inflight:{physical_key.identity}"),
        pinned=True,
    )
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=39),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._prepare_elastic_waiting_deficit_reclaim(
        (7,) * 39, physical_quiescent=False
    )

    assert _pending_elastic_maintenance(scheduler) is None
    assert controller.stats.defer_reasons == (
        ("pressure_reclaim_worker_step_inflight", 1),
    )


def test_elastic_active_wave_defers_pressure_reclaim_with_outstanding_loan():
    scheduler = _new_elastic_scheduler()
    scheduler.running = [object()]
    controller = scheduler._elastic_admission_controller
    controller.resident_bytes = 100
    controller.pinned_resident_bytes = 60
    controller.floor_bytes = 4
    controller.reserve_loan((1, 3, 1, 4, 1), 20)
    scheduler._next_elastic_transaction_id = Mock(return_value="loan-x39")
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=39),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._prepare_elastic_waiting_deficit_reclaim(
        (7,) * 39, physical_quiescent=True
    )

    assert _pending_elastic_maintenance(scheduler) is None
    assert controller.stats.defer_reasons == (
        ("pressure_reclaim_has_outstanding_loan", 1),
    )


def test_elastic_active_wave_defers_pressure_reclaim_with_graph_lease():
    scheduler = _new_elastic_scheduler()
    scheduler.running = [object()]
    controller = scheduler._elastic_admission_controller
    physical_key = scheduler._resolve_elastic_step_physical_keys((1, 3, 17, 17, 1))[0]
    controller.publish_hot(
        physical_key,
        GraphPrice(100, 100, f"leased:{physical_key.identity}"),
        pinned=True,
    )
    controller.retain_hot("active-lease", (physical_key,))
    controller.resident_bytes = 100
    controller.pinned_resident_bytes = 60
    controller.floor_bytes = 4
    scheduler._next_elastic_transaction_id = Mock(return_value="leased-x39")
    coordinator = Mock()
    coordinator.plan_elastic_admission_wave.side_effect = [
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=32),
        SimpleNamespace(max_requests=39),
    ]
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    assert scheduler._prepare_elastic_waiting_deficit_reclaim(
        (7,) * 39, physical_quiescent=True
    )

    assert _pending_elastic_maintenance(scheduler) is None
    assert controller.entries[physical_key].hot
    assert controller.entries[physical_key].leases == frozenset({"active-lease"})
    assert controller.stats.defer_reasons == (
        ("pressure_reclaim_has_active_graph_lease", 1),
    )


def test_elastic_restore_prepares_the_complete_declared_wave():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.resident_bytes = 0
    _clear_elastic_maintenance(scheduler)
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.vllm_config = Mock(
        speculative_config=Mock(disable_speculation_on_non_decode=False)
    )
    scheduler.block_size = 2496
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    requests = [
        Mock(
            status=RequestStatus.WAITING,
            num_tokens=2,
            force_non_speculative=False,
        )
        for _ in range(3)
    ]
    scheduler.requests = {
        f"cal-{index}": request for index, request in enumerate(requests)
    }
    coordinator = Mock()
    coordinator.apply_elastic_admission_wave.return_value = 2
    kv_cache_manager = Mock()
    kv_cache_manager.estimate_uncached_full_sequence_requirements.side_effect = [
        Mock(primary=8),
        Mock(primary=5),
        Mock(primary=3),
    ]
    kv_cache_manager.watermark_blocks = 2
    kv_cache_manager.coordinator = coordinator
    scheduler.kv_cache_manager = kv_cache_manager
    scheduler._canonical_elastic_graph_step_key = Mock(return_value=(0, 3, 3, 16, 0))
    scheduler._elastic_capture_envelope = Mock(return_value=(144, 0, 0))
    scheduler._estimate_elastic_graph_step_bytes = Mock(return_value=(144, False))

    admitted = scheduler.prepare_elastic_restore_admission(["cal-0", "cal-1", "cal-2"])

    assert admitted == 2
    assert scheduler._elastic_restore_wave_target == 2
    coordinator.apply_elastic_admission_wave.assert_called_once_with(
        (10, 5, 3), external_memory_bytes=144
    )


def test_cancel_unexecuted_restore_admission_restores_idle_physical_layout():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_restore_wave_target = 2
    scheduler._elastic_restore_wave_step_key = (0, 3, 3, 16, 0)
    scheduler._elastic_restore_execution_step_key = (0, 3, 3, 16, 0)
    scheduler._elastic_admission_controller.resident_bytes = 144
    _reset_elastic_loans(scheduler, ((0, 3, 1, 4, 0), 192))
    coordinator = Mock()
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    scheduler.reconcile_elastic_restore_rollback()

    assert scheduler._elastic_restore_wave_target == 0
    assert scheduler._elastic_restore_wave_step_key is None
    assert scheduler._elastic_restore_execution_step_key is None
    coordinator.rebalance_elastic_capacity.assert_called_once_with()
    coordinator.set_elastic_external_memory.assert_called_once_with(192)


def test_cancel_scheduled_restore_output_reconciles_after_wave_identity_clears():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_restore_wave_target = 0
    scheduler._elastic_restore_wave_step_key = None
    scheduler._elastic_restore_execution_step_key = None
    scheduler._elastic_admission_controller.resident_bytes = 144
    _reset_elastic_loans(scheduler, ((0, 3, 1, 4, 0), 192))
    coordinator = Mock()
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    scheduler.reconcile_elastic_restore_rollback()

    coordinator.rebalance_elastic_capacity.assert_called_once_with()
    coordinator.set_elastic_external_memory.assert_called_once_with(192)


def test_cancel_unexecuted_restore_admission_rejects_incomplete_wave_identity():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_restore_wave_target = 2
    scheduler._elastic_restore_wave_step_key = None

    with pytest.raises(RuntimeError, match="incomplete wave identity"):
        scheduler.reconcile_elastic_restore_rollback()


def test_elastic_restore_retains_hot_graph_charge_across_prompt_cohorts():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.resident_bytes = 256
    _clear_elastic_maintenance(scheduler)
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.vllm_config = Mock(
        speculative_config=Mock(disable_speculation_on_non_decode=False)
    )
    scheduler.block_size = 2496
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    requests = [
        Mock(
            status=RequestStatus.WAITING,
            num_tokens=2,
            force_non_speculative=False,
        )
        for _ in range(2)
    ]
    scheduler.requests = {"retained-0": requests[0], "retained-1": requests[1]}
    coordinator = Mock()
    coordinator.apply_elastic_admission_wave.return_value = 2
    kv_cache_manager = Mock()
    kv_cache_manager.estimate_uncached_full_sequence_requirements.side_effect = [
        Mock(primary=4),
        Mock(primary=4),
    ]
    kv_cache_manager.watermark_blocks = 1
    kv_cache_manager.coordinator = coordinator
    scheduler.kv_cache_manager = kv_cache_manager
    scheduler._canonical_elastic_graph_step_key = Mock(return_value=(0, 3, 2, 8, 0))
    scheduler._elastic_capture_envelope = Mock(return_value=(128, 0, 0))
    scheduler._estimate_elastic_graph_step_bytes = Mock(return_value=(128, False))

    with pytest.raises(RuntimeError, match="retained a preceding HOT set"):
        scheduler.prepare_elastic_restore_admission(["retained-0", "retained-1"])

    admitted = scheduler.prepare_elastic_restore_admission(
        ["retained-0", "retained-1"],
        retain_hot_graphs=True,
    )

    assert admitted == 2
    assert scheduler._elastic_restore_wave_target == 2
    coordinator.apply_elastic_admission_wave.assert_called_once_with(
        (5, 4), external_memory_bytes=256
    )


def test_elastic_restore_does_not_recharge_cold_peak_for_retained_hot_graphs():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller.resident_bytes = 256
    _clear_elastic_maintenance(scheduler)
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.vllm_config = Mock(
        speculative_config=Mock(disable_speculation_on_non_decode=False)
    )
    scheduler.block_size = 2496
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    request = Mock(
        status=RequestStatus.WAITING,
        num_tokens=2,
        force_non_speculative=False,
    )
    scheduler.requests = {"retained": request}
    coordinator = Mock()
    coordinator.apply_elastic_admission_wave.return_value = 1
    kv_cache_manager = Mock()
    kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = Mock(
        primary=4
    )
    kv_cache_manager.watermark_blocks = 1
    kv_cache_manager.coordinator = coordinator
    scheduler.kv_cache_manager = kv_cache_manager
    scheduler._canonical_elastic_graph_step_key = Mock(return_value=(0, 3, 1, 4, 0))
    scheduler._elastic_capture_envelope = Mock(return_value=(3_200, 0, 0))
    scheduler._estimate_elastic_graph_step_bytes = Mock(return_value=(384, False))

    assert (
        scheduler.prepare_elastic_restore_admission(
            ["retained"], retain_hot_graphs=True
        )
        == 1
    )
    coordinator.apply_elastic_admission_wave.assert_called_once_with(
        (5,), external_memory_bytes=384
    )


def test_elastic_restore_first_hot_replay_replaces_cold_measurement():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    step_key = (0, 3, 2, 8, 0)
    controller = scheduler._elastic_admission_controller
    for key in scheduler._resolve_elastic_step_physical_keys(step_key):
        controller.publish_hot(
            key,
            GraphPrice(100, 100, f"first-hot:{key.identity}"),
            pinned=False,
        )
    controller.resident_bytes = 200
    controller.cublas_workspace_bytes = 40
    controller.last_maintenance_step_key = step_key
    controller.measured_bytes[step_key] = 3_200

    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (240, False)

    controller.last_maintenance_step_key = None
    assert scheduler._estimate_elastic_graph_step_bytes(step_key) == (3_200, False)


def test_elastic_restore_charges_successor_at_real_block_boundary():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler.num_spec_tokens = 3
    scheduler.dynamic_sd_lookup = None
    scheduler.vllm_config = Mock(
        speculative_config=Mock(disable_speculation_on_non_decode=False)
    )
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler.block_size = 2496
    scheduler.watermark_blocks = 0
    request = Mock(
        status=RequestStatus.WAITING,
        num_tokens=2496,
        force_non_speculative=False,
    )
    scheduler.requests = {"cal": request}
    coordinator = Mock()
    coordinator.apply_elastic_admission_wave.return_value = 1
    kv_cache_manager = Mock()
    kv_cache_manager.estimate_uncached_full_sequence_requirements.return_value = Mock(
        primary=1
    )
    kv_cache_manager.watermark_blocks = 0
    kv_cache_manager.coordinator = coordinator
    scheduler.kv_cache_manager = kv_cache_manager
    scheduler._canonical_elastic_graph_step_key = Mock(return_value=(0, 3, 1, 2496, 0))
    scheduler._elastic_capture_envelope = Mock(return_value=None)
    scheduler._estimate_elastic_graph_step_bytes = Mock(return_value=(0, False))

    assert scheduler.prepare_elastic_restore_admission(["cal"]) == 1
    coordinator.apply_elastic_admission_wave.assert_called_once_with(
        (2,), external_memory_bytes=0
    )


def test_elastic_successor_headroom_charges_only_real_block_crossings():
    scheduler = _new_elastic_scheduler()
    scheduler.block_size = 2496
    scheduler.max_model_len = 262144
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.requests = {
        "inside": Mock(num_computed_tokens=0),
        "crossing": Mock(num_computed_tokens=2495),
        "second_inside": Mock(num_computed_tokens=2496),
    }

    assert scheduler._elastic_successor_primary_headroom({"inside": 2}) == 0
    assert (
        scheduler._elastic_successor_primary_headroom(
            {"inside": 2, "crossing": 1, "second_inside": 1}
        )
        == 1
    )


def test_elastic_successor_headroom_uses_prospective_override_and_model_cap():
    scheduler = _new_elastic_scheduler()
    scheduler.block_size = 2496
    scheduler.max_model_len = 262144
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.requests = {
        "waiting": Mock(num_computed_tokens=0),
        "at_cap": Mock(num_computed_tokens=262143),
    }

    assert (
        scheduler._elastic_successor_primary_headroom(
            {"waiting": 1}, computed_token_overrides={"waiting": 2495}
        )
        == 1
    )
    assert scheduler._elastic_successor_primary_headroom({"at_cap": 1}) == 0


def test_elastic_restore_prepares_idle_hotset_reclaim_before_requests():
    generation = RuntimeGeneration("restore-idle-reclaim")
    scheduler = _new_elastic_scheduler(generation)
    scheduler._elastic_restore_mode = True
    scheduler.num_spec_tokens = 3
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._elastic_admission_controller.resident_bytes = 297_795_584
    controller = _elastic_controller(scheduler)
    evictable = next(
        physical_key
        for physical_key in scheduler._resolve_elastic_step_physical_keys(
            (0, 3, 1, 4, 0)
        )
        if physical_key.logical.mode == "PIECEWISE"
    )
    controller.publish_hot(
        evictable,
        GraphPrice(297_795_584, 297_795_584, "restore-idle"),
        pinned=False,
    )
    controller.install_reclaim_group(
        ReclaimGroup(
            "restore-idle",
            (evictable,),
            reclaimable_bytes=297_795_584,
        )
    )
    scheduler._elastic_restore_wave_target = 40
    scheduler.requests = {}
    scheduler.has_unfinished_requests = Mock(return_value=False)
    scheduler.has_finished_requests = Mock(return_value=False)
    kv_cache_manager = Mock()
    scheduler.kv_cache_manager = kv_cache_manager

    prepared = scheduler.prepare_elastic_restore_idle_reclaim()

    assert prepared is True
    reclaim = controller.pending_maintenance_plan
    assert reclaim is not None and reclaim.kind == ElasticPlanKind.RECLAIM
    assert controller.pending_maintenance_step_key is None
    assert scheduler._elastic_restore_wave_target == 0
    kv_cache_manager.estimate_uncached_full_sequence_requirements.assert_not_called()


def test_elastic_restore_reclaims_previous_pinned_family_before_next_shape():
    generation = RuntimeGeneration("calibration-isolated-rebuild")
    scheduler = _new_elastic_scheduler(generation)
    retained_key = (1, 0, 3, 3, 1)
    scheduler._elastic_restore_mode = True
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.requests = {}
    scheduler.has_unfinished_requests = Mock(return_value=False)
    scheduler.has_finished_requests = Mock(return_value=False)
    scheduler._elastic_admission_controller.resident_bytes = 128
    scheduler._elastic_admission_controller.floor_bytes = 0
    retained_physical_keys = scheduler._resolve_elastic_step_physical_keys(retained_key)
    for physical_key in retained_physical_keys:
        controller = scheduler._elastic_admission_controller
        controller.publish_hot(
            physical_key,
            GraphPrice(128, 160, physical_key.logical.owner),
            pinned=True,
        )
        controller.install_reclaim_group(
            ReclaimGroup(
                physical_key.logical.owner,
                (physical_key,),
                reclaimable_bytes=128,
            )
        )

    assert scheduler.prepare_elastic_restore_idle_reclaim() is True
    reclaim = scheduler._elastic_admission_controller.pending_maintenance_plan
    assert reclaim is not None and reclaim.kind == ElasticPlanKind.RECLAIM
    assert set(reclaim.victim_keys) == set(retained_physical_keys)
    assert all(
        not entry.pinned
        for entry in scheduler._elastic_admission_controller.entries.values()
    )


def test_elastic_restore_accepts_zero_reclaim_proof_as_terminal_tail():
    generation = RuntimeGeneration("restore-zero-proof-tail")
    scheduler = _new_elastic_scheduler(generation)
    graph_key = next(
        iter(scheduler._resolve_elastic_step_physical_keys((0, 3, 1, 4, 0)))
    )
    scheduler._elastic_restore_mode = True
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.requests = {}
    scheduler.has_unfinished_requests = Mock(return_value=False)
    scheduler.has_finished_requests = Mock(return_value=False)
    controller = _elastic_controller(scheduler)
    controller.resident_bytes = 94
    controller.synchronize_hot(
        ((graph_key, GraphPrice(94, 94, "retained-pool"), False, 0),)
    )

    assert scheduler.prepare_elastic_restore_idle_reclaim() is False
    assert controller.entries[graph_key].pinned


def test_elastic_restore_idle_reclaim_preserves_protected_carrier_only():
    generation = RuntimeGeneration("restore-protected-carrier")
    scheduler = _new_elastic_scheduler(generation)
    graph_key = next(
        iter(scheduler._resolve_elastic_step_physical_keys((1, 3, 1, 1, 1)))
    )
    scheduler._elastic_restore_mode = True
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.requests = {}
    scheduler.has_unfinished_requests = Mock(return_value=False)
    scheduler.has_finished_requests = Mock(return_value=False)
    controller = _elastic_controller(scheduler)
    controller.resident_bytes = 94
    controller.publish_hot(
        graph_key,
        GraphPrice(94, 94, "protected-carrier"),
        pinned=False,
    )
    scheduler._elastic_serving_carrier_keys = (graph_key,)

    assert scheduler.prepare_elastic_restore_idle_reclaim() is False
    assert controller.pending_maintenance_plan is None
    assert controller.entries[graph_key].hot


def test_elastic_restore_idle_reclaim_rejects_unknown_hot_beside_carrier():
    generation = RuntimeGeneration("restore-protected-plus-unknown")
    scheduler = _new_elastic_scheduler(generation)
    carrier = next(iter(scheduler._resolve_elastic_step_physical_keys((1, 3, 1, 1, 1))))
    unknown = next(iter(scheduler._resolve_elastic_step_physical_keys((0, 3, 1, 2, 0))))
    scheduler._elastic_restore_mode = True
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.requests = {}
    scheduler.has_unfinished_requests = Mock(return_value=False)
    scheduler.has_finished_requests = Mock(return_value=False)
    controller = _elastic_controller(scheduler)
    controller.resident_bytes = 188
    controller.publish_hot(carrier, GraphPrice(94, 94, "carrier"), pinned=False)
    controller.publish_hot(unknown, GraphPrice(94, 94, "unknown"), pinned=False)
    scheduler._elastic_serving_carrier_keys = (carrier,)

    with pytest.raises(RuntimeError, match="could not reclaim the preceding HOT set"):
        scheduler.prepare_elastic_restore_idle_reclaim()

    assert controller.pending_maintenance_plan is None


def test_elastic_restore_idle_reclaim_rejects_waiting_request_boundary():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.has_unfinished_requests = Mock(return_value=True)
    scheduler.has_finished_requests = Mock(return_value=False)

    with pytest.raises(RuntimeError, match="precede every request commit"):
        scheduler.prepare_elastic_restore_idle_reclaim()


def test_pending_elastic_maintenance_is_explicit_scheduler_work():
    scheduler = _new_elastic_scheduler()
    _clear_elastic_maintenance(scheduler)
    assert scheduler.has_pending_elastic_maintenance() is False

    _arm_fixture_capture(scheduler, (1, 3, 1, 4, 4))
    assert scheduler.has_pending_elastic_maintenance() is True


def test_zero_token_preserve_does_not_rebalance_or_publish_kv_transition():
    scheduler = _new_elastic_scheduler()
    coordinator = Mock()
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    resource_commit = scheduler._rebalance_elastic_capacity_before_commit(
        has_user_tokens=False,
        maintenance_plan=None,
    )
    transition = coordinator.take_elastic_transition() if resource_commit else None

    assert resource_commit is False
    assert transition is None
    coordinator.rebalance_elastic_capacity.assert_not_called()
    coordinator.take_elastic_transition.assert_not_called()


@pytest.mark.parametrize("has_user_tokens", [False, True])
def test_real_user_or_maintenance_commit_publishes_kv_transition(
    has_user_tokens,
):
    scheduler = _new_elastic_scheduler()
    coordinator = Mock()
    coordinator.take_elastic_transition.return_value = (73, 4)
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    maintenance_plan = None if has_user_tokens else Mock()

    resource_commit = scheduler._rebalance_elastic_capacity_before_commit(
        has_user_tokens=has_user_tokens,
        maintenance_plan=maintenance_plan,
    )
    transition = coordinator.take_elastic_transition() if resource_commit else None

    assert resource_commit is True
    assert transition == (73, 4)
    coordinator.rebalance_elastic_capacity.assert_called_once_with()
    coordinator.take_elastic_transition.assert_called_once_with()


def test_cancelled_unexecuted_maintenance_closes_transaction_and_metrics():
    generation = RuntimeGeneration("cancelled-maintenance")
    step_key = (1, 3, 8, 32, 4)
    physical_keys = resolve_step_physical_keys(step_key, generation, 4096)
    graph_cache = ElasticAdmissionController(generation)
    for key in physical_keys:
        graph_cache.register(key, price=GraphPrice(10, 12, "cancel-test"))
    plan = graph_cache.plan(
        "cancel-tx",
        physical_keys,
        request_bytes=0,
        available_bytes=1_000,
        destination_capture_endpoint_bytes=100,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    graph_cache.begin_maintenance(plan)

    scheduler = _new_elastic_scheduler()
    scheduler._elastic_restore_mode = True
    scheduler._elastic_admission_controller = graph_cache
    scheduler._elastic_maintenance_started = {
        plan.transaction_id: (time.monotonic(), graph_cache.stats, 0)
    }
    _reset_elastic_loans(scheduler, (step_key, 100))
    scheduler._elastic_admission_controller.step_key = step_key
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0

    output = SchedulerOutput.make_empty()
    output.elastic_step_plan = plan
    output.elastic_transaction_id = plan.transaction_id
    output.elastic_external_memory_bytes = 100

    scheduler.cancel_unexecuted_elastic_restore_step(output)

    assert scheduler._elastic_maintenance_started == {}
    assert not scheduler._elastic_admission_controller.pending_loans
    assert scheduler._elastic_admission_controller.step_key is None
    assert scheduler._elastic_maintenance_transactions_total == 1
    for key in physical_keys:
        assert not graph_cache._entries[key].hot
        assert not graph_cache._entries[key].leases
        outcome = (
            f"{key.logical.owner}|{key.logical.mode}|"
            f"{key.logical.token_bucket}|CANCELLED"
        )
        assert scheduler._elastic_graph_key_outcomes[outcome] == 1


def test_elastic_graph_recapture_envelope_decays_after_first_measurement():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.elastic_on_demand_graphs = True
    key = (1, 3, 1, 4, 4)
    scheduler._elastic_admission_controller.step_key = key
    scheduler._elastic_admission_controller.recapture_pending_key = key
    _reset_elastic_loans(scheduler, (key, 160))
    scheduler._elastic_graph_discovery_base_bytes = 128
    scheduler._elastic_graph_discovery_bytes_per_token = 2
    scheduler._elastic_sampling_workspace_bytes_per_row = 4
    scheduler._elastic_admission_controller.measured_bytes = {key: 48}
    scheduler._elastic_admission_controller.resident_bytes = 48
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler.running = [object()]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 160
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    # Before the first recapture settles, the async copy inherits 160.
    assert (
        scheduler._plan_elastic_graph_loan(
            key, minimum_free_primary_blocks=len(scheduler.running)
        )
        == 160
    )
    first = SchedulerOutput.make_empty()
    first.num_scheduled_tokens = {"a": 4}
    first.total_num_scheduled_tokens = 4
    first.num_spec_tokens_to_schedule = 3
    first.is_pure_decode_step = True
    first.elastic_external_memory_bytes = 160
    first.elastic_successor_primary_headroom = 1
    _settle_elastic_with_receipt(
        scheduler,
        first,
        48,
        0,
        publish_step_key=key,
    )
    assert scheduler._elastic_admission_controller.recapture_pending_key is None

    # The outstanding old grant remains globally safe, but the next per-step
    # grant decays to measured graph bytes plus 4 rows of sampling workspace.
    assert (
        scheduler._plan_elastic_graph_loan(
            key, minimum_free_primary_blocks=len(scheduler.running)
        )
        == 48
    )
    assert coordinator.elastic_external_memory_bytes == 160


def test_elastic_settlement_uses_immutable_scheduled_key_after_tail_abort():
    """Client aborts cannot relabel an already executed physical cohort."""
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    scheduler.elastic_on_demand_graphs = True
    scheduled_key = (0, 3, 39, 156, 4)
    _reset_elastic_loans(scheduler, (scheduled_key, 160))
    scheduler._elastic_admission_controller.resident_bytes = 48
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler._elastic_admission_controller.step_key = scheduled_key
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096, max_num_seqs=39)
    # The X39 forward is complete, but 35 clients abort before settlement.
    # Reconstructing against current logical state would mislabel it as X4.
    scheduler.running = [object() for _ in range(4)]
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 160
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    coordinator.set_elastic_external_memory.return_value = True
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {f"request-{index}": 4 for index in range(4)}
    output.total_num_scheduled_tokens = 16
    output.num_spec_tokens_to_schedule = 3
    output.is_pure_decode_step = True
    output.elastic_external_memory_bytes = 160
    output.elastic_graph_step_key = scheduled_key

    _settle_elastic_with_receipt(
        scheduler,
        output,
        48,
        0,
        publish_step_key=scheduled_key,
    )

    assert not scheduler._elastic_admission_controller.pending_loans
    assert scheduler._elastic_admission_controller.step_key == scheduled_key

    _reset_elastic_loans(scheduler, (scheduled_key, 160))
    mismatched = SchedulerOutput.make_empty()
    mismatched.num_scheduled_tokens = {"request-0": 4}
    mismatched.total_num_scheduled_tokens = 4
    mismatched.num_spec_tokens_to_schedule = 3
    mismatched.is_pure_decode_step = True
    mismatched.elastic_external_memory_bytes = 160
    mismatched.elastic_graph_step_key = (0, 3, 4, 16, 4)

    with pytest.raises(RuntimeError, match="settled out of FIFO order"):
        _settle_elastic_with_receipt(
            scheduler,
            mismatched,
            48,
            0,
            publish_step_key=scheduled_key,
        )


def test_elastic_settlement_returns_unused_cold_tail_before_next_admission():
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    key = (0, 3, 1, 2, 0)
    scheduler._elastic_admission_controller.step_key = key
    scheduler._elastic_admission_controller.recapture_pending_key = key
    _reset_elastic_loans(scheduler, (key, 608))
    scheduler._elastic_graph_catalog = {}
    scheduler._elastic_admission_controller.capture_envelopes = {}
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0
    scheduler.running = [object()]

    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 608
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    coordinator.set_elastic_external_memory.side_effect = commit_external
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)

    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"decode": 2}
    output.total_num_scheduled_tokens = 2
    output.num_spec_tokens_to_schedule = 3
    output.elastic_external_memory_bytes = 608
    _settle_elastic_with_receipt(scheduler, output, 48, 16, 48)

    assert coordinator.elastic_external_memory_bytes == 48
    coordinator.set_elastic_external_memory.assert_called_once_with(
        48, minimum_free_primary_blocks=0
    )


def test_elastic_graph_idle_floor_fails_closed_after_deadline(monkeypatch):
    scheduler = _new_elastic_scheduler(RuntimeGeneration("scheduler-unit"))
    scheduler.scheduler_config = Mock(max_num_batched_tokens=4096)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    stale_zero_token_key = (0, 0, 1, 0, 0)
    _reset_elastic_loans(scheduler, (stale_zero_token_key, 88))
    scheduler._elastic_admission_controller.measured_bytes = {}
    scheduler._elastic_admission_controller.recapture_pending_key = None
    scheduler._elastic_admission_controller.resident_bytes = 88
    scheduler._elastic_admission_controller.floor_bytes = 88
    scheduler._elastic_admission_controller.idle_cleanup_started_at = 10.0
    scheduler._elastic_graph_idle_cleanup_timeout_s = 5.0
    scheduler.running = []
    coordinator = Mock()
    coordinator.elastic_external_memory_bytes = 88
    coordinator.normalize_elastic_external_memory.side_effect = lambda value: value
    coordinator.set_elastic_external_memory.return_value = True
    scheduler.kv_cache_manager = Mock(coordinator=coordinator)
    monkeypatch.setattr(time, "monotonic", lambda: 15.0)

    retained = SchedulerOutput.make_empty()
    # Async scheduling may preserve a finished request key with zero scheduled
    # tokens. Logical scheduler ownership, not this stale dictionary, defines
    # whether the physical floor is idle.
    retained.num_scheduled_tokens = {"finished": 0}
    retained.elastic_external_memory_bytes = 88
    with pytest.raises(RuntimeError, match="physical floor did not return to KV"):
        _settle_elastic_with_receipt(scheduler, retained, 88, 88)


def test_elastic_idle_cleanup_stops_when_only_pinned_residency_remains():
    scheduler = _new_elastic_scheduler()
    scheduler._elastic_admission_controller.resident_bytes = 88
    scheduler._elastic_admission_controller.evictable_resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 0

    assert not scheduler._elastic_idle_cleanup_outstanding()

    scheduler._elastic_admission_controller.evictable_resident_bytes = 17
    assert scheduler._elastic_idle_cleanup_outstanding()

    scheduler._elastic_admission_controller.evictable_resident_bytes = 0
    scheduler._elastic_admission_controller.floor_bytes = 13
    assert scheduler._elastic_idle_cleanup_outstanding()


def test_elastic_serving_idle_retains_current_hotset_without_engine_spin():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = False
    scheduler.running = []
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.has_unfinished_requests = Mock(return_value=False)
    scheduler.has_finished_requests = Mock(return_value=False)
    scheduler.connector = None
    scheduler.ec_connector = None
    controller = scheduler._elastic_admission_controller
    graph_key = scheduler._resolve_elastic_step_physical_keys((0, 3, 1, 4, 4))[0]
    group_id = f"idle:{graph_key.identity}"
    controller.publish_hot(graph_key, GraphPrice(88, 88, group_id), pinned=False)
    controller.install_reclaim_group(
        ReclaimGroup(group_id, (graph_key,), reclaimable_bytes=88)
    )
    controller.resident_bytes = 88
    controller.evictable_resident_bytes = 88

    assert scheduler._elastic_idle_cleanup_outstanding()
    assert not scheduler._needs_elastic_idle_reclaim()
    assert not scheduler.has_requests()
    assert controller.entries[graph_key].hot
    assert controller.pending_maintenance_plan is None


def test_elastic_restore_idle_exposes_reclaim_as_scheduler_work():
    scheduler = _new_elastic_scheduler()
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = True
    scheduler.running = []
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.has_unfinished_requests = Mock(return_value=False)
    scheduler.has_finished_requests = Mock(return_value=False)
    scheduler.connector = None
    scheduler.ec_connector = None
    controller = scheduler._elastic_admission_controller
    controller.resident_bytes = 88
    controller.evictable_resident_bytes = 88

    assert scheduler._needs_elastic_idle_reclaim()
    assert scheduler.has_requests()


@pytest.mark.parametrize(
    "restore,resident,pending,preserve,grant",
    [
        (True, 88, False, True, 88),
        (False, 88, False, True, 88),
        (True, 88, True, True, 88),
        (True, 0, False, True, 0),
    ],
)
def test_elastic_finished_drain_keeps_loan_and_graph_intent_consistent(
    monkeypatch: pytest.MonkeyPatch,
    restore: bool,
    resident: int,
    pending: bool,
    preserve: bool,
    grant: int,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    scheduler = create_scheduler(max_num_seqs=1, max_num_batched_tokens=64)
    scheduler.elastic_on_demand_graphs = True
    scheduler._elastic_restore_mode = restore
    scheduler.finished_req_ids.add("finished-prefill")
    controller = scheduler._elastic_admission_controller
    controller.resident_bytes = resident
    controller.evictable_resident_bytes = resident
    scheduler._elastic_graph_execution_policy = _current_q4_piecewise_policy()
    if resident:
        key = scheduler._resolve_elastic_step_physical_keys((0, 3, 1, 4, 0))[0]
        controller.publish_hot(
            key, GraphPrice(resident, resident, "drain"), pinned=False
        )
        controller.install_reclaim_group(ReclaimGroup("drain", (key,), resident))
    if pending:
        controller.reserve_loan(None, resident)
    coordinator = scheduler.kv_cache_manager.coordinator
    coordinator.elastic_external_memory_bytes = resident
    monkeypatch.setattr(
        coordinator, "normalize_elastic_external_memory", lambda value: value
    )

    def commit_external(requested, **_kwargs):
        coordinator.elastic_external_memory_bytes = requested
        return True

    monkeypatch.setattr(coordinator, "set_elastic_external_memory", commit_external)
    output = scheduler.schedule()

    assert output.finished_req_ids == {"finished-prefill"}
    assert output.total_num_scheduled_tokens == 0
    assert output.elastic_step_plan is None
    assert output.elastic_graph_step_key is None
    assert output.elastic_external_memory_bytes == grant
    assert controller.latest_loan.grant_bytes == grant
    assert output.elastic_preserve_graph_residency is preserve
    while controller.pending_loans:
        controller.settle_next_loan()
    if restore and resident:
        # EngineCore can now arm the typed reclaim transaction. Finished IDs
        # and every preceding FIFO consumer have already reached the worker.
        assert not scheduler.has_finished_requests()
        assert scheduler.prepare_elastic_restore_idle_reclaim()
        recovered = scheduler.schedule()
        assert recovered.elastic_external_memory_bytes == 0
        assert not recovered.elastic_preserve_graph_residency
        assert recovered.elastic_step_plan is not None
        assert recovered.elastic_step_plan.kind == ElasticPlanKind.RECLAIM
        assert recovered.elastic_step_plan.victim_keys == (key,)
        from vllm.v1.worker.gpu.cudagraph_utils import DynamicGraphWorkingSet

        manager = Mock(
            dynamic_graph_owner=key.logical.owner,
            runtime_generation=key.generation.value,
            tp_size=1,
            dynamic_resident_bytes=resident,
            _dynamic_pending=None,
        )
        working_set = DynamicGraphWorkingSet((manager,))
        monkeypatch.setattr(
            working_set, "_release_idle_allocator_cache", Mock(return_value=0)
        )
        working_set.begin_step()
        # Reproduce continuous-v2 with the real lifecycle controller, then
        # recover through the actual scheduler-produced maintenance plan.
        with pytest.raises(RuntimeError, match="active non-decode step"):
            working_set.prepare_idle_reclaim_before_post_consensus()
        plan = recovered.elastic_step_plan
        working_set.begin_admitted_step(plan)
        assert working_set.prepare_idle_reclaim_before_post_consensus() == 0
        assert working_set._active_step_plan_fingerprint == plan.fingerprint
        working_set.require_post_materialization_consensus(
            plan, phase="post_state_application"
        )
        working_set.finish_step()
        assert working_set._active_step_plan_fingerprint is None
        manager.evict_physical_key.assert_called_once_with(
            key,
            transaction_id=plan.transaction_id,
            reason="elastic_admission_plan",
            administrative=True,
            defer_rank_consensus=True,
        )
        manager._publish_deferred_eviction_consensus.assert_called_once_with(None)
