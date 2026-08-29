# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.device_communicators import pynccl_allocator
from vllm.forward_context import BatchDescriptor
from vllm.v1.core.elastic_graph import (
    ElasticGraphCache,
    ElasticGraphError,
    ElasticPlanKind,
    GraphPrice,
    ReclaimGroup,
    RuntimeGeneration,
    resolve_step_physical_keys,
)
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_graph_pool_id():
    pynccl_allocator._graph_pool_id = None
    yield
    pynccl_allocator._graph_pool_id = None


def _create_vllm_config(
    additional_config: dict | None = None,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL,
) -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode=cudagraph_mode,
        cudagraph_capture_sizes=[4],
    )
    compilation_config.max_cudagraph_capture_size = 4
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=4)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.model_config.multimodal_config = None
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0
    vllm_config.additional_config = additional_config
    return vllm_config


def test_worker_policy_publishes_backend_shape_contract_and_order() -> None:
    target = MagicMock()
    target.dynamic_graph_owner = "target"
    target.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    target._runtime_decode_query_lens.return_value = {1}
    target.compiled_piecewise_sizes = frozenset()
    target.tp3_owner_prequant = False
    target.elastic_graph_activation = "always"
    target.elastic_graph_token_source = "step"
    target.elastic_graph_fixed_query_len = None

    dflash = MagicMock()
    dflash.dynamic_graph_owner = "dflash_query"
    dflash.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    dflash._runtime_decode_query_lens.return_value = {8}
    dflash.compiled_piecewise_sizes = frozenset()
    dflash.tp3_owner_prequant = False
    dflash.elastic_graph_activation = "speculative"
    dflash.elastic_graph_token_source = "fixed_query"
    dflash.elastic_graph_fixed_query_len = 8

    policy = gpu_cudagraph_utils.graph_execution_policy_from_managers(
        (target, dflash)
    )
    by_owner = {owner.owner: owner for owner in policy.owners}

    assert by_owner["target"].execution_order == 0
    assert by_owner["target"].token_source == "step"
    assert by_owner["dflash_query"].execution_order == 1
    assert by_owner["dflash_query"].token_source == "fixed_query"
    assert by_owner["dflash_query"].fixed_query_len == 8
    assert policy.verifier_contract == "parallel-draft-query-v1"
    assert policy.math_contract == "pending-parallel-draft-product-math-v1"


def test_dflash_publishes_and_captures_its_dynamic_graph_owner() -> None:
    speculator = object.__new__(DFlashSpeculator)
    manager = MagicMock()
    speculator.query_cudagraph_manager = manager
    speculator.sample_indices = MagicMock()
    speculator.sample_pos = MagicMock()
    speculator.sample_idx_mapping = MagicMock()
    speculator._generate_draft = MagicMock()
    speculator.input_buffers = MagicMock()
    speculator.block_tables = MagicMock()
    speculator.attn_groups = MagicMock()
    speculator.kv_cache_config = MagicMock()
    speculator.max_model_len = 4096
    speculator._group_causal = True
    speculator.model = MagicMock()
    speculator.model_state = MagicMock()

    def capture_next_dynamic(*args, capture_override, **kwargs):
        capture_override({CUDAGraphMode.FULL: []}, MagicMock())
        return True

    manager.capture_next_dynamic.side_effect = capture_next_dynamic

    assert speculator.dynamic_cudagraph_managers() == (manager,)
    assert speculator.capture_next_dynamic(manager) is True
    manager.capture.assert_called_once()
    speculator.sample_indices.zero_.assert_called_once_with()
    speculator.sample_pos.zero_.assert_called_once_with()
    speculator.sample_idx_mapping.zero_.assert_called_once_with()


def test_worker_rejects_forbidden_full_before_state_and_recovers() -> None:
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.dynamic_graph_owner = "target"
    manager.runtime_generation = "policy-control"
    manager.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    manager.full_decode_query_lens = {1}
    manager.decode_query_len = 4
    manager.compiled_piecewise_sizes = frozenset()
    manager.tp3_owner_prequant = True
    manager._dynamic_step_planned = False
    manager.last_dynamic_capture_rejection = "sentinel"
    manager._dynamic_step_candidates = set()
    manager._dynamic_epoch = 0
    manager._dynamic_graph_entries = {}
    manager.dynamic_graph_hotset_cap_bytes = 1
    manager._dynamic_pending = None

    generation = RuntimeGeneration("policy-control")
    forbidden = resolve_step_physical_keys((1, 3, 40, 160, 4), generation, 4096)[
        0
    ]
    before = (
        manager._dynamic_step_planned,
        manager.last_dynamic_capture_rejection,
        set(manager._dynamic_step_candidates),
        manager._dynamic_epoch,
        dict(manager._dynamic_graph_entries),
        manager._dynamic_pending,
    )
    with pytest.raises(ElasticGraphError, match="FULL key violates"):
        manager.queue_physical_key(forbidden)
    after = (
        manager._dynamic_step_planned,
        manager.last_dynamic_capture_rejection,
        set(manager._dynamic_step_candidates),
        manager._dynamic_epoch,
        dict(manager._dynamic_graph_entries),
        manager._dynamic_pending,
    )
    assert after == before

    allowed = resolve_step_physical_keys((0, 3, 40, 160, 4), generation, 4096)[0]
    descriptor = manager.queue_physical_key(allowed)
    assert descriptor.cg_mode == CUDAGraphMode.PIECEWISE
    assert manager._dynamic_pending == descriptor


def test_bounded_decode_descriptor_matches_semantic_tail_only() -> None:
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.dynamic_graph_owner = "target"
    manager.decode_query_len = 4
    manager.dynamic_piecewise_safety_sizes = frozenset()
    manager._lora_dispatch_map = {}

    x8 = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=32,
        num_reqs=None,
        uniform_token_count=4,
        physical_num_reqs=8,
    )
    assert manager._dynamic_descriptor_matches_step(
        x8,
        num_reqs=7,
        num_tokens=28,
        uniform_token_count=4,
        num_active_loras=0,
    )
    assert not manager._dynamic_descriptor_matches_step(
        x8,
        num_reqs=7,
        num_tokens=27,
        uniform_token_count=None,
        num_active_loras=0,
    )

    mtp_decode = replace(
        x8,
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
    )
    manager.dynamic_graph_owner = "mtp_decode"
    manager.decode_query_len = 1
    assert manager._dynamic_descriptor_matches_step(
        mtp_decode,
        num_reqs=7,
        num_tokens=7,
        uniform_token_count=1,
        num_active_loras=0,
    )


def test_effective_batched_verifier_binds_accepted_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_DECODE", "1")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_PSEUDO_DECODE", "0")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_SEQUENTIAL_DECODE", "0")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_FIXED_SPLIT_SIZE", "2048")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_DISABLE_SPLIT_KV", "1")
    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB", "96")

    verifier, configuration, math = (
        gpu_cudagraph_utils._effective_mtp_verifier_contract()
    )
    assert verifier == "batched-causal-q1-v1"
    assert configuration.endswith(
        "fixed_split=2048:disable_split=1:workspace_mib=96"
    )
    assert math == "accepted-batched-q1-split2048-nosplit-forced-prefix-v1"

    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB", "64")
    assert gpu_cudagraph_utils._effective_mtp_verifier_contract()[2] == (
        "pending-batched-q1-product-math-v1"
    )


def test_owner_prequant_semantics_require_graph_backed_carrier(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils.envs, "AG2_VLLM_TP3_OWNER_MIN_ROWS", 128
    )
    manager = object.__new__(gpu_cudagraph_utils.ModelCudaGraphManager)
    manager.tp3_owner_prequant = True
    manager.decode_query_len = 4

    full = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=160,
        num_reqs=40,
        uniform_token_count=4,
        semantic_decode=True,
    )
    compiled = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=160,
        num_reqs=None,
        uniform_token_count=4,
        physical_num_reqs=40,
        semantic_decode=True,
    )
    piecewise = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=160,
        num_reqs=None,
        uniform_token_count=4,
        physical_num_reqs=40,
        semantic_decode=True,
    )
    wrong_query_len = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=160,
        num_reqs=None,
        uniform_token_count=1,
        physical_num_reqs=40,
    )
    too_small = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=64,
        num_reqs=None,
        uniform_token_count=4,
        physical_num_reqs=16,
    )

    assert manager.uses_tp3_owner_prequant_decode(full)
    assert not manager.uses_tp3_owner_prequant_decode(compiled)
    assert manager.uses_tp3_owner_prequant_decode(piecewise)
    assert not manager.uses_tp3_owner_prequant_decode(wrong_query_len)
    assert not manager.uses_tp3_owner_prequant_decode(too_small)

    manager.tp3_owner_prequant = False
    assert not manager.uses_tp3_owner_prequant_decode(full)


def test_capture_baseline_hook_runs_after_warmup(monkeypatch):
    graph_pool = object()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "is_global_first_rank",
        lambda: False,
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: graph_pool,
    )

    @contextmanager
    def fake_graph_capture(device):
        del device
        yield None

    monkeypatch.setattr(gpu_cudagraph_utils, "graph_capture", fake_graph_capture)
    monkeypatch.setattr(gpu_cudagraph_utils.torch.cuda, "synchronize", lambda *_: None)
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        decode_query_len=1,
    )
    manager.use_breakable_cg = False
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4,
        num_reqs=None,
    )
    events = []

    def create_forward_fn(capture_desc, warmup):
        events.append(("prepare", capture_desc, warmup))

        def forward_fn(mode):
            events.append(("forward", mode))

        return forward_fn

    manager.capture(
        create_forward_fn,
        capture_descs={CUDAGraphMode.PIECEWISE: [desc]},
        capture_begin_hook=lambda capture_desc: events.append(
            ("baseline", capture_desc)
        ),
    )

    assert events == [
        ("prepare", desc, True),
        ("forward", CUDAGraphMode.NONE),
        ("baseline", desc),
        ("forward", CUDAGraphMode.PIECEWISE),
    ]


def test_dynamic_capture_owner_lives_until_graph_reset(monkeypatch):
    graph_pool = object()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(gpu_cudagraph_utils, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: graph_pool,
    )

    @contextmanager
    def fake_graph_capture(device):
        del device
        yield None

    monkeypatch.setattr(gpu_cudagraph_utils, "graph_capture", fake_graph_capture)
    monkeypatch.setattr(gpu_cudagraph_utils.torch.cuda, "synchronize", lambda *_: None)
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        decode_query_len=1,
    )
    manager.use_breakable_cg = False
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4,
        num_reqs=None,
    )
    entry = gpu_cudagraph_utils.DynamicGraphEntry(descriptor=desc)
    events: list[str] = []
    owner_ref: weakref.ReferenceType | None = None

    class CaptureOwner:
        def __call__(self, mode):
            del mode

        def __del__(self):
            events.append("owner_released")

    def create_forward_fn(capture_desc, warmup):
        nonlocal owner_ref
        assert capture_desc == desc
        assert warmup
        owner = CaptureOwner()
        owner_ref = weakref.ref(owner)
        return owner

    manager.capture(
        create_forward_fn,
        capture_descs={CUDAGraphMode.PIECEWISE: [desc]},
        capture_complete_hook=lambda capture_desc, state: setattr(
            entry, "capture_state", state
        ),
    )

    assert owner_ref is not None and owner_ref() is not None
    graph = MagicMock()
    graph.reset.side_effect = lambda: events.append("graph_reset")
    manager.graphs[desc] = graph
    manager._destroy_dynamic_graphs(entry)

    assert owner_ref() is None
    assert events == ["graph_reset", "owner_released"]


def test_full_capture_sets_graph_pool_id_before_cuda_graph(monkeypatch):
    """FULL capture must set graph_pool_id before entering torch.cuda.graph().

    NCCL symmetric memory checks this global during graph capture; without
    it, capture fails with:
    AssertionError: graph_pool_id is not set under graph capture
    """
    graph_pool = object()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: graph_pool,
    )

    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config(),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=1,
    )

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    manager._capture_descs[CUDAGraphMode.FULL] = [desc]

    def create_forward_fn(desc, warmup):
        return lambda _mode: None

    @contextmanager
    def fake_graph_capture(*args, **kwargs):
        yield SimpleNamespace(stream=MagicMock())

    fake_offloader = MagicMock()
    monkeypatch.setattr(gpu_cudagraph_utils.torch.cuda, "synchronize", lambda *_: None)

    def cuda_graph_enter(*args, **kwargs):
        assert pynccl_allocator._graph_pool_id is graph_pool

    mock_cuda_graph_ctx = MagicMock()
    mock_cuda_graph_ctx.__enter__ = cuda_graph_enter
    mock_cuda_graph_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch.object(gpu_cudagraph_utils, "graph_capture", fake_graph_capture),
        patch.object(gpu_cudagraph_utils, "get_offloader", lambda: fake_offloader),
        patch.object(gpu_cudagraph_utils.torch.cuda, "CUDAGraph"),
        patch.object(
            gpu_cudagraph_utils.torch.cuda,
            "graph",
            return_value=mock_cuda_graph_ctx,
        ) as mock_cuda_graph,
    ):
        manager.capture(create_forward_fn)

    mock_cuda_graph.assert_called_once()


def test_full_multitoken_compatibility_requires_exact_request_shape():
    check = gpu_cudagraph_utils._is_compatible

    exact = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=12,
        num_reqs=3,
        uniform_token_count=4,
    )
    assert check(exact, 3, 12, 4, 0)

    padded_multitoken = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=16,
        num_reqs=4,
        uniform_token_count=4,
    )
    assert not check(padded_multitoken, 3, 12, 4, 0)

    padded_qlen1 = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    assert check(padded_qlen1, 3, 3, 1, 0)

    padded_piecewise = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=16,
        num_reqs=None,
        uniform_token_count=None,
    )
    assert check(padded_piecewise, 3, 12, 4, 0)


def test_dynamic_piecewise_descriptor_queues_before_hot_dispatch(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "dynamic_cudagraph_capture_sizes": [8],
            "dynamic_cudagraph_budget_mb": 64,
            "dynamic_cudagraph_min_hits": 2,
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
    )
    manager._graphs_captured = True

    first = manager.dispatch(2, 8, None, 0)
    second = manager.dispatch(2, 8, None, 0)

    assert first.cg_mode == CUDAGraphMode.NONE
    assert second.cg_mode == CUDAGraphMode.NONE
    assert manager.has_pending_dynamic_capture()
    assert manager._dynamic_pending is not None
    entry = manager._dynamic_graph_entries[manager._dynamic_pending]
    assert entry.state == gpu_cudagraph_utils.DynamicGraphResidency.QUEUED

    entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    manager._dynamic_pending = None
    hot = manager.dispatch(2, 8, None, 0)
    assert hot.cg_mode == CUDAGraphMode.PIECEWISE


def test_elastic_target_captures_runtime_descriptor_before_first_dispatch(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )

    assert manager.defer_startup_graphs
    assert not manager.needs_capture()
    assert manager._graphs_captured
    assert not manager._dynamic_graph_entries

    manager.begin_dynamic_step()
    desc = manager.queue_runtime_descriptor(4, 4, 1, 0, allow_full=True)
    assert desc == BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
        physical_num_reqs=4,
        runtime_generation=manager.runtime_generation,
    )
    assert manager.has_pending_dynamic_capture()
    with pytest.raises(RuntimeError, match="not captured before execution"):
        manager.dispatch(4, 4, 1, 0)

    entry = manager._dynamic_graph_entries[desc]
    entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    manager._dynamic_pending = None
    assert manager.dispatch(4, 4, 1, 0) == desc


@pytest.mark.parametrize("owner", ["target", "mtp_prefill"])
def test_elastic_owner_dispatches_configured_compiled_piecewise_without_capture(
    monkeypatch, caplog, owner,
):
    caplog.set_level("DEBUG")
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "elastic_gdn_backing": True,
            "elastic_compiled_piecewise_sizes": [
                2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
            ],
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.scheduler_config.max_num_batched_tokens = 4096
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner=owner,
    )

    manager.begin_dynamic_step()
    planned = manager.queue_runtime_descriptor(40, 4096, None, 0, allow_full=False)
    assert planned is not None and planned.cg_mode == CUDAGraphMode.NONE
    assert planned.physical_num_reqs == 40
    assert not manager._dynamic_step_candidates
    assert not manager._dynamic_graph_entries
    assert not manager.has_pending_dynamic_capture()
    assert manager.dispatch(40, 4096, None, 0) == planned
    assert manager._compiled_piecewise_dispatch_logged == {4096}

    manager.begin_dynamic_step()
    manager.queue_runtime_descriptor(40, 4096, None, 0, allow_full=False)
    assert manager.dispatch(40, 4096, None, 0) == planned
    assert manager._compiled_piecewise_dispatch_logged == {4096}

    manager.begin_dynamic_step()
    planned_1024 = manager.queue_runtime_descriptor(
        1, 1024, None, 0, allow_full=False
    )
    assert planned_1024 is not None and planned_1024.cg_mode == CUDAGraphMode.NONE
    assert manager.dispatch(1, 1024, None, 0) == planned_1024
    assert not manager._dynamic_step_candidates
    assert not manager._dynamic_graph_entries

    manager.begin_dynamic_step()
    assert manager.is_compiled_piecewise_shape(1152)
    planned_1152 = manager.queue_runtime_descriptor(
        2, 1152, None, 0, allow_full=False
    )
    assert planned_1152 is not None
    assert planned_1152.cg_mode == CUDAGraphMode.NONE
    assert planned_1152.num_tokens == 2048
    assert manager.dispatch(2, 1152, None, 0) == planned_1152
    assert not manager._dynamic_step_candidates
    assert not manager._dynamic_graph_entries

    manager.begin_dynamic_step()
    with pytest.raises(RuntimeError, match="same-step planning"):
        manager.dispatch(40, 4096, None, 0)


@pytest.mark.parametrize("owner", ["target", "mtp_prefill"])
def test_elastic_owner_routes_underfilled_tail_compiled_and_recovers_exact_graph(
    monkeypatch, owner,
):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.scheduler_config.max_num_batched_tokens = 4096
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner=owner,
    )

    # The exact B64 carrier remains graph-backed.
    manager.begin_dynamic_step()
    exact = manager.queue_runtime_descriptor(12, 64, None, 0, allow_full=False)
    assert exact is not None and exact.cg_mode == CUDAGraphMode.PIECEWISE
    exact_entry = manager._dynamic_graph_entries[exact]
    exact_entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    manager._dynamic_pending = None
    assert manager.dispatch(12, 64, None, 0) == exact

    # Reproduce the physical scheduler sequence: maintenance has already
    # queued the canonical B64 candidate before the user step reveals 48 live
    # tokens. Runtime dispatch must not consume that executable.
    manager.begin_dynamic_step()
    queued_exact = manager.queue_runtime_descriptor(
        1, 64, None, 0, allow_full=False
    )
    assert queued_exact is not None
    queued_entry = manager._dynamic_graph_entries[queued_exact]
    queued_entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    manager._dynamic_pending = None
    partial = manager.dispatch(1, 48, None, 0)
    assert partial is not None and partial.cg_mode == CUDAGraphMode.NONE
    assert partial.num_tokens == 64
    assert partial.physical_num_reqs == 1
    assert manager._dynamic_step_candidates == {queued_exact}
    assert exact_entry.state == gpu_cudagraph_utils.DynamicGraphResidency.HOT
    assert queued_entry.state == gpu_cudagraph_utils.DynamicGraphResidency.HOT

    # Recovery is identity-preserving: the same exact executable is reused.
    manager.begin_dynamic_step()
    recovered = manager.queue_runtime_descriptor(
        12, 64, None, 0, allow_full=False
    )
    assert recovered == exact
    assert manager.dispatch(12, 64, None, 0) == exact

def test_elastic_mtp_owner_defers_startup_capture(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="mtp_decode",
    )

    assert manager.defer_startup_graphs
    assert not manager.needs_capture()
    assert manager._graphs_captured
    assert manager.dynamic_resident_bytes == 0
    # Underfill is a target/mtp-prefill PIECEWISE rule. The independent FULL
    # q1 decode owner is never classified as compiled-only.
    assert not manager.is_compiled_piecewise_shape(48)


def test_elastic_recipe_restores_only_reachable_warm_metadata(monkeypatch):
    from vllm.v1.worker import startup_plan

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_runtime_generation",
        lambda _config: "test-generation",
    )
    monkeypatch.setattr(
        startup_plan,
        "load_cudagraph_recipe",
        lambda *args, **kwargs: [
            {
                "mode": "FULL",
                "num_tokens": 4,
                "num_reqs": 4,
                "uniform_token_count": 1,
                "num_active_loras": 0,
                "physical_num_reqs": 4,
                "runtime_generation": "test-generation",
            },
            {
                "mode": "PIECEWISE",
                "num_tokens": 3,
                "num_reqs": None,
                "uniform_token_count": None,
                "num_active_loras": 0,
                "physical_num_reqs": 3,
                "runtime_generation": "test-generation",
            },
            {
                "mode": "FULL",
                "num_tokens": 3,
                "num_reqs": 4,
                "uniform_token_count": 1,
                "num_active_loras": 0,
                "physical_num_reqs": 4,
                "runtime_generation": "test-generation",
            },
        ],
    )
    monkeypatch.setattr(
        startup_plan, "maybe_save_cudagraph_recipe", lambda *a, **k: None
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )

    assert set(manager._dynamic_graph_entries) == {
            BatchExecutionDescriptor(
            CUDAGraphMode.FULL,
            4,
            4,
            1,
            0,
            physical_num_reqs=4,
                runtime_generation="test-generation",
                semantic_decode=True,
        ),
            BatchExecutionDescriptor(
            CUDAGraphMode.PIECEWISE,
            3,
            None,
            None,
            0,
            physical_num_reqs=3,
                runtime_generation="test-generation",
                semantic_decode=True,
        ),
    }
    assert all(
        entry.state == gpu_cudagraph_utils.DynamicGraphResidency.WARM
        for entry in manager._dynamic_graph_entries.values()
    )
    assert manager.dynamic_resident_bytes == 0
    assert not manager.graphs


def test_elastic_recipe_generation_rebinds_before_residency(monkeypatch):
    from vllm.v1.core.elastic_graph import RuntimeGeneration
    from vllm.v1.worker import startup_plan

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    monkeypatch.setattr(
        startup_plan,
        "compute_elastic_runtime_generation",
        lambda _config: "pre-kv-generation",
    )
    monkeypatch.setattr(startup_plan, "load_cudagraph_recipe", lambda *a, **k: [])
    monkeypatch.setattr(
        startup_plan, "maybe_save_cudagraph_recipe", lambda *a, **k: None
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )
    descriptor = manager.runtime_descriptor(4, 4, 1, 0, allow_full=True)
    manager._dynamic_graph_entries[descriptor] = (
        gpu_cudagraph_utils.DynamicGraphEntry(descriptor=descriptor)
    )

    gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,)).rebind_runtime_generation(
        RuntimeGeneration("post-kv-generation")
    )

    assert manager.runtime_generation == "post-kv-generation"
    assert {item.runtime_generation for item in manager._dynamic_graph_entries} == {
        "post-kv-generation"
    }
    entry = next(iter(manager._dynamic_graph_entries.values()))
    entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    with pytest.raises(RuntimeError, match="resident state"):
        manager.rebind_runtime_generation("later-generation")


def test_elastic_runtime_rejects_static_graph_shape_and_budget_knobs(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    config = _create_vllm_config(
        additional_config={
            "elastic_gdn_backing": True,
            "dynamic_cudagraph_capture_sizes": [8],
            "dynamic_cudagraph_budget_mb": 64,
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    with pytest.raises(ValueError, match="runtime-derived"):
        gpu_cudagraph_utils.CudaGraphManager(
            vllm_config=config,
            device=torch.device("cpu"),
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            decode_query_len=1,
            owner="target",
        )


def test_dynamic_working_set_uses_one_aggregate_loan():
    target = MagicMock()
    target.dynamic_resident_bytes = 10
    target.has_pending_dynamic_capture.return_value = True
    target.dynamic_memory_request_bytes.return_value = 100
    target.prepare_pending_dynamic_capture.return_value = True

    mtp_prefill = MagicMock()
    mtp_prefill.dynamic_resident_bytes = 20
    mtp_prefill.has_pending_dynamic_capture.return_value = True
    mtp_prefill.dynamic_memory_request_bytes.return_value = 200

    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet((target, mtp_prefill))
    assert working_set.resident_bytes == 30

    # Each owner receives the aggregate grant minus residency already owned
    # by every other manager.
    assert working_set.prepare_manager_capture(target, 120)
    target.prepare_pending_dynamic_capture.assert_called_once_with(100)
    mtp_prefill.prepare_pending_dynamic_capture.assert_not_called()

    assert working_set.pending_managers() == (target, mtp_prefill)
    assert working_set.prepare_manager_capture(mtp_prefill, 230)
    mtp_prefill.prepare_pending_dynamic_capture.assert_called_once_with(220)

    # The scheduler receives only actual residency after the same-step
    # captures, never another speculative next-step request.
    assert working_set.finish_step() == 30
    target.finish_dynamic_step.assert_called_once_with()
    mtp_prefill.finish_dynamic_step.assert_called_once_with()

    target.has_pending_dynamic_capture.return_value = False
    assert working_set.finish_step() == 30


def test_dynamic_working_set_executes_immutable_piecewise_first_capture_order():
    generation = RuntimeGeneration("worker-m160-order")
    desired = resolve_step_physical_keys(
        (1, 3, 40, 160, 4),
        generation,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(256, 512, 1024, 2048, 4096),
    )
    cache = ElasticGraphCache(generation)
    plan = cache.plan(
        "worker-m160-order",
        desired,
        request_bytes=0,
        available_bytes=512,
        owner_set_capture_envelope_bytes=200,
        replace_unleased_on_miss=True,
        residency_cap_bytes=256,
    )

    managers = []
    for graph_key in desired:
        manager = MagicMock()
        manager.dynamic_graph_owner = graph_key.logical.owner
        manager.is_physical_key_hot.return_value = False
        manager.has_pending_dynamic_capture.return_value = True
        descriptor = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode[graph_key.logical.mode],
            num_tokens=graph_key.logical.token_bucket,
            num_reqs=graph_key.logical.logical_num_reqs,
            uniform_token_count=graph_key.logical.uniform_query_len,
            physical_num_reqs=graph_key.physical_num_reqs,
            runtime_generation=generation.value,
        )

        def queue_physical_key(
            queued_key,
            *,
            expected_key=graph_key,
            queued_descriptor=descriptor,
            queued_manager=manager,
        ):
            assert queued_key == expected_key
            queued_manager._dynamic_pending = queued_descriptor
            return queued_descriptor

        manager.queue_physical_key.side_effect = queue_physical_key
        managers.append(manager)

    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))
    working_set.begin_step()
    working_set.apply_plan(plan)

    pending_owners = [
        manager.dynamic_graph_owner for manager in working_set.pending_managers()
    ]
    assert pending_owners == ["mtp_prefill", "target", "mtp_decode"]


def test_elastic_target_defers_startup_family_without_manual_budget(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    capture_sizes = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
    config.compilation_config.cudagraph_capture_sizes = capture_sizes
    config.compilation_config.max_cudagraph_capture_size = 64
    config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=32,
        max_num_batched_tokens=4096,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=4,
        owner="target",
    )

    assert manager.defer_startup_graphs
    assert manager.dynamic_graph_budget_bytes == 0
    assert manager.dynamic_graph_max_entry_bytes == 0
    assert not manager.needs_capture()
    assert manager._graphs_captured
    assert manager.dynamic_resident_bytes == 0
    assert manager.dynamic_piecewise_capture_sizes == ()
    assert manager.runtime_descriptor(1, 37, 37, 0, allow_full=False) == (
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.PIECEWISE,
            num_tokens=64,
            num_reqs=None,
            physical_num_reqs=1,
            runtime_generation=manager.runtime_generation,
        )
    )
    assert manager.dynamic_piecewise_safety_sizes == (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    )
    manager.begin_dynamic_step()
    desc = manager.queue_runtime_descriptor(8, 4096, None, 0, allow_full=False)
    assert desc == BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4096,
        num_reqs=None,
        physical_num_reqs=8,
        runtime_generation=manager.runtime_generation,
    )


def test_elastic_multitoken_full_registers_every_reachable_x(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    # The legacy sparse list covers only X1 and X2 for qlen4. The elastic
    # server can reach X1..X5, including M12/M16/M20 above the static max.
    config.compilation_config.cudagraph_capture_sizes = [4, 8]
    config.compilation_config.max_cudagraph_capture_size = 8
    config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=5)
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=4,
        owner="target",
    )

    full_x = {
        manager.runtime_descriptor(x, 4 * x, 4, 0, allow_full=True).num_reqs
        for x in range(1, 6)
    }
    assert full_x == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("owner", ["target", "mtp_prefill"])
@pytest.mark.parametrize("x", [1, 2, 4, 8, 16, 32, 40])
def test_elastic_batched_q1_decode_uses_bounded_piecewise_m(
    monkeypatch, owner, x
):
    """K3 final decode M=4X stays Graph-backed, including M160 terminal."""
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "elastic_gdn_backing": True,
            "elastic_compiled_piecewise_sizes": [
                2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
            ],
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.scheduler_config.max_num_batched_tokens = 4096
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=4,
        max_uniform_decode_reqs=40,
        owner=owner,
    )

    m = 4 * x
    desc = manager.runtime_descriptor(
        x, m, 4, 0, allow_full=False, semantic_decode=True
    )

    assert desc is not None
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.num_tokens == m
    assert desc.physical_num_reqs == x
    assert desc.uniform_token_count == 4
    assert desc.semantic_decode
    assert not manager.is_compiled_piecewise_shape(
        m, num_reqs=x, semantic_decode=True
    )


def test_uniform_q4_prompt_prefill_cannot_alias_decode_lane(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "elastic_gdn_backing": True,
            "elastic_compiled_piecewise_sizes": [32],
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.scheduler_config.max_num_batched_tokens = 4096
    manager = gpu_cudagraph_utils.ModelCudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=4,
        max_uniform_decode_reqs=40,
        owner="target",
        tp3_owner_prequant=True,
    )

    generic = manager.runtime_descriptor(
        8, 32, 4, 0, allow_full=False, semantic_decode=False
    )
    assert generic is not None
    assert not generic.semantic_decode
    assert manager.is_compiled_piecewise_shape(
        32, num_reqs=8, semantic_decode=False
    )
    assert not manager.uses_tp3_owner_prequant_decode(generic)

    decode = manager.runtime_descriptor(
        8, 32, 4, 0, allow_full=False, semantic_decode=True
    )
    assert decode is not None and decode.semantic_decode
    assert not manager.is_compiled_piecewise_shape(
        32, num_reqs=8, semantic_decode=True
    )
    assert manager.uses_tp3_owner_prequant_decode(decode)


def test_elastic_nonexact_piecewise_does_not_claim_uniform_decode_semantics(
    monkeypatch,
):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=4,
        max_uniform_decode_reqs=40,
        owner="target",
    )

    padded = manager.runtime_descriptor(39, 160, 4, 0, allow_full=False)
    missing_semantics = manager.runtime_descriptor(
        40, 160, None, 0, allow_full=False
    )

    assert padded.uniform_token_count is None
    assert missing_semantics.uniform_token_count is None


def test_elastic_runtime_sequence_graphs_exact_buckets_and_compiles_tails(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.compilation_config.cudagraph_capture_sizes = [1, 8]
    config.compilation_config.max_cudagraph_capture_size = 8
    config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=8,
        max_num_batched_tokens=4096,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=4,
        owner="target",
    )

    def fake_evict(entry):
        manager._dynamic_resident_bytes -= entry.charged_bytes
        entry.charged_bytes = 0
        entry.state = gpu_cudagraph_utils.DynamicGraphResidency.WARM

    shapes = (
        (1, 4, 4, True, CUDAGraphMode.FULL),
        (1, 37, 37, False, CUDAGraphMode.NONE),
        (8, 32, 4, True, CUDAGraphMode.FULL),
        (8, 4096, None, False, CUDAGraphMode.PIECEWISE),
    )
    with monkeypatch.context() as context:
        context.setattr(manager, "_evict_dynamic_entry", fake_evict)
        for num_reqs, num_tokens, uniform, allow_full, expected_mode in shapes:
            manager.begin_dynamic_step()
            planned = manager.runtime_descriptor(
                num_reqs,
                num_tokens,
                uniform,
                0,
                allow_full=allow_full,
            )
            manager.release_unused_dynamic_residency(
                num_reqs,
                num_tokens,
                uniform,
                0,
                planned_descriptor=planned,
            )
            desc = manager.queue_runtime_descriptor(
                num_reqs,
                num_tokens,
                uniform,
                0,
                allow_full=allow_full,
            )
            assert desc is not None
            assert desc.cg_mode == expected_mode
            if expected_mode == CUDAGraphMode.NONE:
                assert not manager._dynamic_step_candidates
                assert manager.dispatch(num_reqs, num_tokens, uniform, 0) == desc
                # The compiled tail does not allocate a new executable and
                # does not retire an unrelated HOT exact carrier.
                assert manager.finish_dynamic_step() == 16
                continue
            entry = manager._dynamic_graph_entries[desc]
            entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
            entry.charged_bytes = 16
            manager._dynamic_resident_bytes = 16
            manager._dynamic_pending = None
            assert manager.dispatch(num_reqs, num_tokens, uniform, 0) == desc
            assert manager.finish_dynamic_step() == 16


def test_dynamic_piecewise_range_uses_sparse_safety_until_hot(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "dynamic_cudagraph_piecewise_capture_range": [65, 4096],
            "dynamic_cudagraph_piecewise_safety_sizes": [
                128,
                256,
                512,
                1024,
                2048,
                4096,
            ],
            "dynamic_cudagraph_budget_mb": 64,
            "dynamic_cudagraph_min_hits": 2,
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
    )
    manager._graphs_captured = True

    assert manager.ensure_piecewise_safety_for_first_use(65, None, 0)
    safety_entry = manager._dynamic_graph_entries[manager._dynamic_pending]
    safety_entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    manager._dynamic_pending = None
    first = manager.dispatch(8, 65, None, 0)
    second = manager.dispatch(8, 65, None, 0)
    assert first.cg_mode == CUDAGraphMode.PIECEWISE
    assert first.num_tokens == 128
    assert second.cg_mode == CUDAGraphMode.PIECEWISE
    assert second.num_tokens == 128
    assert manager._dynamic_pending is not None

    entry = manager._dynamic_graph_entries[manager._dynamic_pending]
    entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    manager._dynamic_pending = None
    hot = manager.dispatch(8, 65, None, 0)
    assert hot == BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=65,
        num_reqs=None,
        uniform_token_count=None,
    )
    near_bucket = manager.dispatch(8, 127, None, 0)
    assert near_bucket.num_tokens == 128
    assert all(
        entry.descriptor.num_tokens != 127
        for entry in manager._dynamic_graph_entries.values()
    )


def test_dynamic_full_uses_piecewise_safety_until_hot(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "dynamic_cudagraph_full_capture_sizes": [3],
            "dynamic_cudagraph_budget_mb": 64,
            "dynamic_cudagraph_min_hits": 2,
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )
    manager._graphs_captured = True

    first = manager.dispatch(3, 3, 1, 0)
    second = manager.dispatch(3, 3, 1, 0)
    assert first.cg_mode == CUDAGraphMode.FULL
    assert first.num_tokens == 4
    assert second.cg_mode == CUDAGraphMode.FULL
    assert second.num_tokens == 4
    assert manager._dynamic_pending is not None

    entry = manager._dynamic_graph_entries[manager._dynamic_pending]
    entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    manager._dynamic_pending = None
    hot = manager.dispatch(3, 3, 1, 0)
    assert hot == BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=3,
        num_reqs=3,
        uniform_token_count=1,
    )


def test_dynamic_graph_working_set_retains_hot_entries_on_x_drop(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "dynamic_cudagraph_full_capture_sizes": [2, 3, 4],
            "dynamic_cudagraph_budget_mb": 64,
            "dynamic_cudagraph_guard_mb": 0,
            "dynamic_cudagraph_min_hits": 1,
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )
    manager._graphs_captured = True
    entries = {
        entry.descriptor.num_tokens: entry
        for entry in manager._dynamic_graph_entries.values()
        if entry.descriptor.cg_mode == CUDAGraphMode.FULL
    }
    old_entry = entries[2]
    next_entry = entries[3]
    old_entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    old_entry.charged_bytes = 16
    manager._dynamic_resident_bytes = 16
    next_entry.state = gpu_cudagraph_utils.DynamicGraphResidency.QUEUED
    manager._dynamic_pending = next_entry.descriptor

    next_entry.estimated_bytes = 32
    assert not manager.prepare_pending_dynamic_capture(31)
    assert manager._dynamic_pending == next_entry.descriptor
    assert old_entry.state == gpu_cudagraph_utils.DynamicGraphResidency.HOT

    assert manager.prepare_pending_dynamic_capture(32)
    assert old_entry.state == gpu_cudagraph_utils.DynamicGraphResidency.HOT
    assert manager.dynamic_resident_bytes == 16

    next_entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    next_entry.charged_bytes = 20
    manager._dynamic_resident_bytes = 36
    manager._dynamic_pending = None
    manager.begin_dynamic_step()
    assert manager.dispatch(3, 3, 1, 0) == next_entry.descriptor
    assert manager.finish_dynamic_step() == 36

    manager.begin_dynamic_step()
    manager.release_unused_dynamic_residency(1, 1, 1, 0)
    assert manager.dynamic_resident_bytes == 36
    x1 = manager.dispatch(1, 1, 1, 0)
    assert x1.num_tokens >= 1
    assert manager.finish_dynamic_step() == 36
    assert manager.dynamic_resident_bytes == 36


def test_dynamic_piecewise_retains_exact_runtime_x_and_flashinfer_wrapper(
    monkeypatch,
):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "elastic_gdn_backing": True,
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=64,
        max_num_batched_tokens=64,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )
    manager._graphs_captured = True
    manager.begin_dynamic_step()
    desc = manager.queue_runtime_descriptor(12, 64, None, 0, allow_full=False)
    assert desc is not None
    entry = manager._dynamic_graph_entries[desc]
    entry.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    manager._dynamic_pending = None
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,))

    assert entry.runtime_num_reqs == 12
    assert working_set.hot_request_counts == frozenset({12})
    assert working_set.hot_token_counts == frozenset({64})
    assert manager._capture_num_reqs(desc) == 12

    next_desc = manager.runtime_descriptor(32, 64, None, 0, allow_full=False)
    assert next_desc != desc
    manager.release_unused_dynamic_residency(
        32,
        64,
        None,
        0,
        planned_descriptor=next_desc,
    )
    assert entry.state == gpu_cudagraph_utils.DynamicGraphResidency.HOT

    manager.queue_runtime_descriptor(32, 64, None, 0, allow_full=False)
    next_entry = manager._dynamic_graph_entries[next_desc]
    assert next_entry.runtime_num_reqs == 32
    assert next_entry.state == gpu_cudagraph_utils.DynamicGraphResidency.QUEUED
    assert manager._capture_num_reqs(next_desc) == 32
    assert manager._capture_num_reqs(desc) == 12


def test_dynamic_piecewise_capture_uses_physical_x_not_token_bucket(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=64,
        max_num_batched_tokens=64,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )
    manager._graphs_captured = True

    desc = manager.queue_runtime_descriptor(16, 32, None, 0, allow_full=False)

    assert desc is not None
    assert desc.num_reqs is None
    assert manager._capture_num_reqs(desc) == 16


def test_piecewise_capture_rejects_impossible_physical_x(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=64,
        max_num_batched_tokens=64,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )
    manager._graphs_captured = True
    desc = manager.queue_runtime_descriptor(16, 32, None, 0, allow_full=False)
    assert desc is not None
    manager._dynamic_graph_entries[desc].runtime_num_reqs = 33

    with pytest.raises(RuntimeError, match="invalid physical request cardinality"):
        manager._capture_num_reqs(desc)


def test_common_dynamic_capture_entrypoint_covers_speculator_manager():
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        SpeculatorCudaGraphManager,
    )

    assert (
        SpeculatorCudaGraphManager.capture_next_dynamic
        is gpu_cudagraph_utils.CudaGraphManager.capture_next_dynamic
    )


def test_dynamic_capture_publication_accepts_zero_incremental_pool_growth():
    assert gpu_cudagraph_utils.dynamic_capture_publication_allowed(3, 0, 64)


@pytest.mark.parametrize(
    ("graph_segments", "charged_bytes", "granted_bytes"),
    [
        (0, 0, 64),
        (3, -1, 64),
        (3, 65, 64),
    ],
)
def test_dynamic_capture_publication_rejects_invalid_physical_contract(
    graph_segments: int,
    charged_bytes: int,
    granted_bytes: int,
):
    assert not gpu_cudagraph_utils.dynamic_capture_publication_allowed(
        graph_segments,
        charged_bytes,
        granted_bytes,
    )


def test_model_dynamic_capture_state_is_charged_and_released():
    manager = object.__new__(gpu_cudagraph_utils.ModelCudaGraphManager)
    manager.hidden_states = torch.empty((3, 5), dtype=torch.bfloat16)
    manager.aux_hidden_states = [torch.empty((2, 7), dtype=torch.float32)]
    manager.aux_hidden_states_token_major = [True]
    manager.intermediate_tensors = None
    desc = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=3,
        num_reqs=3,
        uniform_token_count=1,
    )

    assert manager._dynamic_capture_state_bytes(desc) == 86
    manager._release_dynamic_capture_state(desc)
    assert manager._dynamic_capture_state_bytes(desc) == 0
    assert manager.hidden_states is None
    assert manager.aux_hidden_states == []
    assert manager.aux_hidden_states_token_major == []


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE])
def test_model_dynamic_capture_state_direct_control_uses_retained_closure(mode):
    manager = object.__new__(gpu_cudagraph_utils.ModelCudaGraphManager)
    manager.dynamic_graph_owner = "target"
    manager.is_last_pp_rank = True
    manager.use_aux_hidden_state_outputs = False
    desc = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=mode,
        num_tokens=3,
        num_reqs=3 if mode == CUDAGraphMode.FULL else None,
        uniform_token_count=1,
        physical_num_reqs=3,
    )
    bundle = gpu_cudagraph_utils.DynamicModelCaptureBundle(
        hidden_states=torch.zeros((3, 5), dtype=torch.bfloat16)
    )
    observed_modes = []

    def direct_static_closure(execution_mode):
        observed_modes.append(execution_mode)
        bundle.hidden_states.fill_(7)

    manager._dynamic_graph_entries = {
        desc: gpu_cudagraph_utils.DynamicGraphEntry(
            descriptor=desc,
            state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
            capture_state=direct_static_closure,
            lifetime_bundle=bundle,
        )
    }

    output = manager.run_dynamic_capture_state_direct_control(desc)

    assert observed_modes == [CUDAGraphMode.NONE]
    assert output.data_ptr() == bundle.hidden_states.data_ptr()
    assert torch.equal(output, torch.full_like(output, 7))


def test_model_dynamic_capture_state_direct_control_fails_closed_without_hot_state():
    manager = object.__new__(gpu_cudagraph_utils.ModelCudaGraphManager)
    manager.dynamic_graph_owner = "target"
    desc = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=3,
        num_reqs=3,
        uniform_token_count=1,
    )
    manager._dynamic_graph_entries = {
        desc: gpu_cudagraph_utils.DynamicGraphEntry(descriptor=desc)
    }

    with pytest.raises(RuntimeError, match="requires a HOT entry"):
        manager.run_dynamic_capture_state_direct_control(desc)


def test_base_dynamic_capture_state_direct_control_returns_speculator_output():
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.dynamic_graph_owner = "mtp_prefill"
    desc = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=160,
        num_reqs=None,
        physical_num_reqs=40,
    )
    observed_modes = []
    expected = {"draft": torch.arange(3)}

    def direct_static_closure(execution_mode):
        observed_modes.append(execution_mode)
        return expected

    manager._dynamic_graph_entries = {
        desc: gpu_cudagraph_utils.DynamicGraphEntry(
            descriptor=desc,
            state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
            capture_state=direct_static_closure,
        )
    }

    output = manager.run_dynamic_capture_state_direct_control(desc)

    assert observed_modes == [CUDAGraphMode.NONE]
    assert output is expected


def test_speculator_dynamic_capture_output_is_released_with_descriptor():
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        SpeculatorCudaGraphManager,
    )

    manager = object.__new__(SpeculatorCudaGraphManager)
    desc = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=3,
        num_reqs=3,
        uniform_token_count=1,
    )
    other = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    manager._ag2_capture_outputs = {
        desc: {"draft": torch.empty(9)},
        other: {"draft": torch.empty(12)},
    }

    manager._release_dynamic_capture_state(desc)

    assert desc not in manager._ag2_capture_outputs
    assert other in manager._ag2_capture_outputs


def test_speculator_eviction_drops_descriptor_output(monkeypatch):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        SpeculatorCudaGraphManager,
    )

    manager = object.__new__(SpeculatorCudaGraphManager)
    manager.dynamic_graph_owner = "mtp_prefill"
    manager.graphs = {}
    desc = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=3,
        num_reqs=3,
        uniform_token_count=1,
    )
    output = torch.empty((3, 11))
    manager._ag2_capture_outputs = {desc: {"hidden": output}}
    entry = gpu_cudagraph_utils.DynamicGraphEntry(
        descriptor=desc,
        graph_pool=object(),
    )
    evict = MagicMock(return_value=1)
    evict_breakable = MagicMock(return_value=2)
    monkeypatch.setattr(CUDAGraphWrapper, "evict_batch_descriptor", evict)
    monkeypatch.setattr(
        BreakableCUDAGraphWrapper,
        "evict_batch_descriptor",
        evict_breakable,
    )

    assert manager._destroy_dynamic_graphs(entry) == 3
    assert manager._ag2_capture_outputs == {}
    assert entry.graph_pool is None


def test_compiled_piecewise_eviction_detaches_then_unwinds_in_reverse(monkeypatch):
    from vllm.compilation.cuda_graph import CUDAGraphEntry, CUDAGraphWrapper

    descriptor = BatchDescriptor(
        num_tokens=1,
        has_lora=False,
        num_active_loras=0,
        cudagraph_owner="target",
    )
    private_pool = object()
    graphs = [MagicMock(), MagicMock()]
    entries = [
        CUDAGraphEntry(
            descriptor,
            cudagraph=graphs[0],
            output=object(),
            graph_pool=private_pool,
            capture_order=10,
        ),
        CUDAGraphEntry(
            descriptor,
            cudagraph=graphs[1],
            output=object(),
            graph_pool=private_pool,
            capture_order=11,
        ),
    ]

    class Wrapper:
        def __init__(self, entry):
            self.concrete_cudagraph_entries = {descriptor: entry}

    wrappers = [Wrapper(entry) for entry in entries]
    reset_order = []

    def reset(order):
        assert all(
            descriptor not in wrapper.concrete_cudagraph_entries
            for wrapper in wrappers
        )
        assert all(entry.output is None for entry in entries)
        reset_order.append(order)

    graphs[0].reset.side_effect = lambda: reset(10)
    graphs[1].reset.side_effect = lambda: reset(11)
    monkeypatch.setattr(CUDAGraphWrapper, "_all_instances", wrappers)

    assert (
        CUDAGraphWrapper.evict_batch_descriptor(descriptor, private_pool) == 2
    )
    assert reset_order == [11, 10]


def test_dynamic_private_pool_routes_and_restores_all_wrapper_types(monkeypatch):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    class Wrapper:
        def __init__(self, graph_pool):
            self.graph_pool = graph_pool

    full_original = object()
    breakable_original = object()
    manager_original = object()
    private_pool = object()
    full = Wrapper(full_original)
    breakable = Wrapper(breakable_original)
    monkeypatch.setattr(CUDAGraphWrapper, "_all_instances", weakref.WeakSet([full]))
    monkeypatch.setattr(
        BreakableCUDAGraphWrapper,
        "_all_instances",
        weakref.WeakSet([breakable]),
    )
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.pool = manager_original

    with (
        pytest.raises(RuntimeError, match="rollback"),
        manager._use_dynamic_graph_pool(private_pool),
    ):
        assert manager.pool is private_pool
        assert full.graph_pool is private_pool
        assert breakable.graph_pool is private_pool
        raise RuntimeError("rollback")

    assert manager.pool is manager_original
    assert full.graph_pool is full_original
    assert breakable.graph_pool is breakable_original


def test_dynamic_graph_trim_targets_manager_device(monkeypatch):
    from cuda.bindings import runtime as cuda_runtime

    trim = MagicMock(return_value=(cuda_runtime.cudaError_t.cudaSuccess,))
    monkeypatch.setattr(cuda_runtime, "cudaDeviceGraphMemTrim", trim)
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.device = torch.device("cuda:2")

    manager._trim_cuda_graph_memory()

    trim.assert_called_once_with(2)


def test_dynamic_capture_admission_purges_free_allocator_cache(monkeypatch):
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.device = torch.device("cpu")
    manager.tp_size = 1
    manager._dynamic_capture_granted_bytes = 64
    candidate = SimpleNamespace(estimated_bytes=32)
    events = []

    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.cuda,
        "synchronize",
        lambda device: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.gc, "collect", lambda: events.append("collect")
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.accelerator,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.accelerator,
        "get_memory_info",
        lambda: (48, 128),
    )

    assert manager._admit_dynamic_capture(candidate)
    assert events == [
        ("synchronize", torch.device("cpu")),
        "collect",
        "empty_cache",
        ("synchronize", torch.device("cpu")),
    ]

    events.clear()
    manager._dynamic_capture_granted_bytes = 31
    assert not manager._admit_dynamic_capture(candidate)
    assert manager.last_dynamic_capture_rejection == (
        "phase=loan required_bytes=32 granted_bytes=31"
    )
    assert events == []


def test_elastic_capture_loan_uses_scheduler_measured_envelope(monkeypatch):
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.device = torch.device("cpu")
    manager.tp_size = 1
    manager.defer_startup_graphs = True
    candidate = SimpleNamespace(estimated_bytes=44 << 20)
    required = candidate.estimated_bytes
    manager._dynamic_capture_granted_bytes = required

    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 2, 2, 1, 0)
    entry = gpu_cudagraph_utils.DynamicGraphEntry(desc)
    entry.state = gpu_cudagraph_utils.DynamicGraphResidency.QUEUED
    entry.estimated_bytes = candidate.estimated_bytes
    manager.dynamic_graph_owner = "mtp_decode"
    manager._dynamic_pending = desc
    manager._dynamic_graph_entries = {desc: entry}

    assert not manager.prepare_pending_dynamic_capture(required - 1)
    assert manager.prepare_pending_dynamic_capture(required)

    monkeypatch.setattr(gpu_cudagraph_utils.torch.cuda, "synchronize", lambda _d: None)
    monkeypatch.setattr(gpu_cudagraph_utils.gc, "collect", lambda: None)
    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.accelerator, "empty_cache", lambda: None
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.accelerator,
        "get_memory_info",
        lambda: (required - 1, required * 2),
    )

    assert manager._dynamic_capture_loan_bytes(candidate) == required
    assert not manager._admit_dynamic_capture(candidate)
    assert manager.last_dynamic_capture_rejection == (
        f"phase=physical_free required_bytes={required} "
        f"granted_bytes={required} rank_safe_free_bytes={required - 1}"
    )

    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.accelerator,
        "get_memory_info",
        lambda: (required, required * 2),
    )
    assert manager._admit_dynamic_capture(candidate)
    assert manager.last_dynamic_capture_rejection is None


def test_runtime_batch_descriptor_separates_physical_graph_owners():
    target = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    target.dynamic_graph_owner = "target"
    mtp_prefill = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    mtp_prefill.dynamic_graph_owner = "mtp_prefill"
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=97,
        num_reqs=None,
    )

    target_key = target._runtime_batch_descriptor(desc)
    mtp_key = mtp_prefill._runtime_batch_descriptor(desc)

    assert target_key == BatchDescriptor(num_tokens=97, cudagraph_owner="target")
    assert mtp_key == BatchDescriptor(num_tokens=97, cudagraph_owner="mtp_prefill")
    assert target_key != mtp_key
    assert len({target_key: "target", mtp_key: "mtp"}) == 2


def test_runtime_batch_descriptor_separates_equal_shape_math_lanes(monkeypatch):
    manager = object.__new__(gpu_cudagraph_utils.ModelCudaGraphManager)
    manager.dynamic_graph_owner = "target"
    manager.tp3_owner_prequant = True
    manager.decode_query_len = 4
    monkeypatch.setattr(gpu_cudagraph_utils.envs, "AG2_VLLM_TP3_OWNER_MIN_ROWS", 1)

    mixed = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4,
        num_reqs=None,
        uniform_token_count=None,
        physical_num_reqs=1,
        runtime_generation="same-generation",
    )
    exact_decode = replace(mixed, uniform_token_count=4, semantic_decode=True)

    mixed_key = manager._runtime_batch_descriptor(mixed)
    decode_key = manager._runtime_batch_descriptor(exact_decode)

    assert mixed_key.tp3_owner_prequant_decode is False
    assert decode_key.tp3_owner_prequant_decode is True
    assert mixed_key != decode_key
    assert len({mixed_key: "mixed", decode_key: "decode"}) == 2


def test_measured_cuda_retention_stays_visible_to_elastic_kv():
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.device = torch.device("cpu")
    manager.tp_size = 1
    manager._dynamic_resident_bytes = 11
    manager._dynamic_retention_ledger = (
        gpu_cudagraph_utils.DynamicGraphRetentionLedger()
    )
    manager._owns_dynamic_retention_ledger = True

    assert manager._account_dynamic_retained_bytes(58, 35) == 23
    assert manager.dynamic_resident_bytes == 34
    assert manager._account_dynamic_retained_bytes(31, 71) == -23
    assert manager.dynamic_resident_bytes == 11


def test_delayed_graph_cleanup_retires_only_current_step_retention():
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.device = torch.device("cpu")
    manager.tp_size = 1
    manager._dynamic_resident_bytes = 0
    manager._dynamic_retention_ledger = gpu_cudagraph_utils.DynamicGraphRetentionLedger(
        local_bytes=17,
        # The 60-byte step increase was already published by eviction.
        rank_safe_bytes=77,
        step_start_local_bytes=17,
        step_released_capture_state_bytes=40,
    )
    manager._owns_dynamic_retention_ledger = True

    # Eviction added 60 bytes. Later wrapper cleanup plus explicitly released
    # capture state can retire those 60, but must not consume the older 17.
    manager._dynamic_retention_ledger.local_bytes = 77
    assert manager._reconcile_dynamic_retained_cleanup(30) == -60
    assert manager.dynamic_resident_bytes == 17
    assert manager._dynamic_retention_ledger.local_bytes == 17
    assert manager._dynamic_retention_ledger.step_start_local_bytes == 17
    assert manager._dynamic_retention_ledger.step_released_capture_state_bytes == 0


def test_hot_replay_retention_reconcile_has_no_collective(monkeypatch):
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.device = torch.device("cuda")
    manager.tp_size = 3
    manager._dynamic_retention_ledger = gpu_cudagraph_utils.DynamicGraphRetentionLedger(
        local_bytes=17,
        rank_safe_bytes=23,
        step_start_local_bytes=17,
    )
    all_reduce = MagicMock()
    synchronize = MagicMock()
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)

    assert manager._reconcile_dynamic_retained_cleanup(0) == 0
    all_reduce.assert_not_called()
    synchronize.assert_not_called()
    assert manager._dynamic_retention_ledger.rank_safe_bytes == 23


def test_retention_collective_waits_for_captured_consumers(monkeypatch):
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.device = torch.device("cuda")
    manager.tp_size = 3
    manager._dynamic_retention_ledger = gpu_cudagraph_utils.DynamicGraphRetentionLedger(
        local_bytes=41,
        rank_safe_bytes=17,
        step_start_local_bytes=17,
    )
    events = []
    retained = MagicMock()
    retained.item.return_value = 41
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: retained)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: events.append(("all_reduce", args[0])),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(device_group="tp"),
    )

    assert manager._reconcile_dynamic_retained_cleanup(0) == 24
    assert events == [("synchronize", torch.device("cuda")), ("all_reduce", retained)]


def test_working_set_counts_device_retention_once_across_owners():
    managers = []
    for owner in ("target", "mtp_prefill", "mtp_decode"):
        manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
        manager.dynamic_graph_owner = owner
        manager._dynamic_resident_bytes = 10
        manager._dynamic_retention_ledger = (
            gpu_cudagraph_utils.DynamicGraphRetentionLedger()
        )
        manager._owns_dynamic_retention_ledger = True
        managers.append(manager)

    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))
    working_set.managers[0]._dynamic_retention_ledger.rank_safe_bytes = 23

    assert working_set.resident_bytes == 53
    assert [manager.dynamic_resident_bytes for manager in managers] == [33, 10, 10]
    assert len({id(manager._dynamic_retention_ledger) for manager in managers}) == 1


def test_working_set_stages_hotset_victims_until_candidate_is_under_cap():
    generation = RuntimeGeneration("worker-hotset")
    old_keys = resolve_step_physical_keys((1, 3, 1, 1, 1), generation, 4096)
    new_keys = resolve_step_physical_keys((1, 3, 16, 16, 1), generation, 4096)
    cache = ElasticGraphCache(generation)
    for graph_key in old_keys:
        graph_price = GraphPrice(10, 10, f"old:{graph_key.identity}")
        cache.publish_hot(graph_key, graph_price, pinned=False)
        cache.install_reclaim_group(
            ReclaimGroup(graph_price.reclaim_group, (graph_key,), 10)
        )
    plan = cache.plan(
        "worker-replace",
        new_keys,
        request_bytes=30,
        available_bytes=256,
        owner_set_capture_envelope_bytes=60,
        replace_unleased_on_miss=True,
        residency_cap_bytes=80,
    )

    managers = []
    for old_key, graph_key in zip(old_keys, new_keys, strict=True):
        manager = MagicMock()
        manager.dynamic_graph_owner = graph_key.logical.owner
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode[graph_key.logical.mode],
            num_tokens=graph_key.logical.token_bucket,
            num_reqs=graph_key.logical.logical_num_reqs,
            uniform_token_count=graph_key.logical.uniform_query_len,
            physical_num_reqs=graph_key.physical_num_reqs,
            runtime_generation=generation.value,
        )
        old_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode[old_key.logical.mode],
            num_tokens=old_key.logical.token_bucket,
            num_reqs=old_key.logical.logical_num_reqs,
            uniform_token_count=old_key.logical.uniform_query_len,
            physical_num_reqs=old_key.physical_num_reqs,
            runtime_generation=generation.value,
        )
        manager._dynamic_graph_entries = {
            desc: gpu_cudagraph_utils.DynamicGraphEntry(
                descriptor=desc,
                state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
                charged_bytes=20,
            ),
            old_desc: gpu_cudagraph_utils.DynamicGraphEntry(
                descriptor=old_desc,
                state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
                charged_bytes=10,
            ),
        }
        manager.dynamic_resident_bytes = 30
        managers.append(manager)
    managers[0]._dynamic_retention_ledger = (
        gpu_cudagraph_utils.DynamicGraphRetentionLedger(rank_safe_bytes=5)
    )
    managers[0].dynamic_resident_bytes = 35
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))

    assert working_set.validate_staged_hotset_candidate(
        plan, external_overhead_bytes=7
    ) == 72
    working_set.stage_hotset_victim_commit(plan)
    assert {
        row[0] for row in working_set.hot_snapshot()
    } == {key.identity for key in new_keys}
    assert working_set.resident_bytes == 95
    for manager in managers:
        manager.evict_physical_key.assert_not_called()

    with pytest.raises(RuntimeError, match="transaction mismatch"):
        working_set.finish_staged_hotset_after_consumers("wrong-transaction")
    for manager in managers:
        manager.evict_physical_key.assert_not_called()

    assert working_set.finish_staged_hotset_after_consumers("worker-replace") == 3
    assert not working_set.has_pending_staged_hotset_retirement
    for manager in managers:
        manager.evict_physical_key.assert_not_called()


def test_working_set_defers_victims_until_distinct_successor_transaction():
    plan = MagicMock()
    plan.staged_hotset_replace = True
    plan.transaction_id = "maintenance"
    candidate = MagicMock()
    candidate.identity = "candidate"
    plan.physical_keys = (candidate,)
    victim = MagicMock()
    victim.logical.owner = "target"
    plan.victim_keys = (victim,)
    manager = MagicMock(dynamic_graph_owner="target")
    manager._dynamic_graph_entries = {}
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,))

    working_set.stage_hotset_victim_commit(plan)
    assert working_set.pending_staged_hotset_transaction_id == "maintenance"
    same = MagicMock(
        transaction_id="maintenance",
        kind=ElasticPlanKind.USER,
        physical_keys=plan.physical_keys,
    )
    with pytest.raises(RuntimeError, match="must differ"):
        working_set.finish_staged_hotset_after_successor(same)
    manager.evict_physical_key.assert_not_called()

    wrong = MagicMock(
        transaction_id="wrong-user",
        kind=ElasticPlanKind.USER,
        physical_keys=(MagicMock(identity="wrong-candidate"),),
    )
    with pytest.raises(RuntimeError, match="owner set differs"):
        working_set.apply_plan(wrong)
    manager.queue_physical_key.assert_not_called()
    manager.evict_physical_key.assert_not_called()

    successor = MagicMock(
        transaction_id="first-user",
        kind=ElasticPlanKind.USER,
        physical_keys=plan.physical_keys,
    )
    assert working_set.finish_staged_hotset_after_successor(successor) == 1
    assert not working_set.has_pending_staged_hotset_retirement
    manager.evict_physical_key.assert_not_called()


def test_working_set_does_not_arm_probation_without_physical_victims():
    first = MagicMock(
        staged_hotset_replace=True,
        transaction_id="empty-to-first",
        victim_keys=(),
    )
    second = MagicMock(
        staged_hotset_replace=True,
        transaction_id="additive-restore",
        victim_keys=(),
    )
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(())

    working_set.stage_hotset_victim_commit(first)
    working_set.stage_hotset_victim_commit(second)

    assert not working_set.has_pending_staged_hotset_retirement
    assert working_set.pending_staged_hotset_transaction_id is None


def test_working_set_hotset_cap_failure_aborts_only_new_candidates():
    generation = RuntimeGeneration("worker-hotset-abort")
    old_keys = resolve_step_physical_keys((1, 3, 1, 1, 1), generation, 4096)
    new_keys = resolve_step_physical_keys((1, 3, 16, 16, 1), generation, 4096)
    cache = ElasticGraphCache(generation)
    for graph_key in old_keys:
        graph_price = GraphPrice(10, 10, f"old:{graph_key.identity}")
        cache.publish_hot(graph_key, graph_price, pinned=False)
        cache.install_reclaim_group(
            ReclaimGroup(graph_price.reclaim_group, (graph_key,), 10)
        )
    plan = cache.plan(
        "worker-abort",
        new_keys,
        request_bytes=30,
        available_bytes=256,
        owner_set_capture_envelope_bytes=60,
        replace_unleased_on_miss=True,
        residency_cap_bytes=80,
    )
    managers = []
    for graph_key in new_keys:
        manager = MagicMock()
        manager.dynamic_graph_owner = graph_key.logical.owner
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode[graph_key.logical.mode],
            num_tokens=graph_key.logical.token_bucket,
            num_reqs=graph_key.logical.logical_num_reqs,
            uniform_token_count=graph_key.logical.uniform_query_len,
            physical_num_reqs=graph_key.physical_num_reqs,
            runtime_generation=generation.value,
        )
        manager._dynamic_graph_entries = {
            desc: gpu_cudagraph_utils.DynamicGraphEntry(
                descriptor=desc,
                state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
                charged_bytes=30,
            )
        }
        managers.append(manager)
    managers[0]._dynamic_retention_ledger = (
        gpu_cudagraph_utils.DynamicGraphRetentionLedger(rank_safe_bytes=1)
    )
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))

    with pytest.raises(RuntimeError, match="exceeds rank-safe residency cap"):
        working_set.validate_staged_hotset_candidate(
            plan, external_overhead_bytes=20
        )
    working_set.abort_staged_hotset_candidate(plan)
    for manager, new_key in zip(managers, new_keys, strict=True):
        manager.discard_staged_physical_key.assert_called_once_with(
            new_key,
            transaction_id="worker-abort",
        )
        manager.evict_physical_key.assert_not_called()


def test_working_set_validates_retained_entries_with_new_candidate():
    generation = RuntimeGeneration("worker-hotset-retain")
    old_key = resolve_step_physical_keys((1, 0, 1, 1, 1), generation, 4096)[0]
    new_key = resolve_step_physical_keys((1, 0, 2, 2, 1), generation, 4096)[0]
    cache = ElasticGraphCache(generation)
    publish_price = GraphPrice(10, 10, f"old:{old_key.identity}")
    cache.publish_hot(old_key, publish_price, pinned=False)
    cache.install_reclaim_group(
        ReclaimGroup(publish_price.reclaim_group, (old_key,), 10)
    )
    cache.register(new_key, price=GraphPrice(20, 24, f"new:{new_key.identity}"))
    plan = cache.plan(
        "worker-retain",
        (new_key,),
        request_bytes=10,
        available_bytes=64,
        owner_set_capture_envelope_bytes=24,
        replace_unleased_on_miss=True,
        residency_cap_bytes=40,
    )
    assert not plan.victim_keys

    manager = MagicMock()
    manager.dynamic_graph_owner = "target"
    entries = {}
    for graph_key, charged in ((old_key, 10), (new_key, 20)):
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode[graph_key.logical.mode],
            num_tokens=graph_key.logical.token_bucket,
            num_reqs=graph_key.logical.logical_num_reqs,
            uniform_token_count=graph_key.logical.uniform_query_len,
            physical_num_reqs=graph_key.physical_num_reqs,
            runtime_generation=generation.value,
        )
        entries[desc] = gpu_cudagraph_utils.DynamicGraphEntry(
            descriptor=desc,
            state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
            charged_bytes=charged,
        )
    manager._dynamic_graph_entries = entries
    manager._dynamic_retention_ledger = (
        gpu_cudagraph_utils.DynamicGraphRetentionLedger(rank_safe_bytes=5)
    )
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,))

    assert working_set.validate_staged_hotset_candidate(plan) == 35


def test_working_set_abort_clears_deferred_hotset_retirement():
    generation = RuntimeGeneration("worker-hotset-abort-deferred")
    old_keys = resolve_step_physical_keys((1, 3, 1, 1, 1), generation, 4096)
    new_keys = resolve_step_physical_keys((1, 3, 16, 16, 1), generation, 4096)
    cache = ElasticGraphCache(generation)
    for graph_key in old_keys:
        graph_price = GraphPrice(10, 10, f"old:{graph_key.identity}")
        cache.publish_hot(graph_key, graph_price, pinned=False)
        cache.install_reclaim_group(
            ReclaimGroup(graph_price.reclaim_group, (graph_key,), 10)
        )
    plan = cache.plan(
        "worker-abort-deferred",
        new_keys,
        request_bytes=30,
        available_bytes=256,
        owner_set_capture_envelope_bytes=60,
        replace_unleased_on_miss=True,
        residency_cap_bytes=80,
    )
    managers = []
    for graph_key in new_keys:
        manager = MagicMock()
        manager.dynamic_graph_owner = graph_key.logical.owner
        manager._dynamic_graph_entries = {}
        managers.append(manager)
    managers[0]._dynamic_retention_ledger = (
        gpu_cudagraph_utils.DynamicGraphRetentionLedger()
    )
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))

    working_set.stage_hotset_victim_commit(plan)
    working_set.abort_staged_hotset_candidate(plan)

    assert working_set.finish_staged_hotset_after_consumers(
        "worker-abort-deferred"
    ) == 0
    for manager, new_key in zip(managers, new_keys, strict=True):
        manager.discard_staged_physical_key.assert_called_once_with(
            new_key,
            transaction_id="worker-abort-deferred",
        )
        manager.evict_physical_key.assert_not_called()


def test_working_set_prices_future_eviction_floor_from_physical_pool_ledger():
    mib = 1 << 20
    managers = []
    for owner, charged_mib, pool_mib in (
        ("target", 44, 44),
        ("mtp_prefill", 44, 2),
        ("mtp_decode", 65, 46),
    ):
        manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
        manager.dynamic_graph_owner = owner
        manager.device = torch.device("cpu")
        manager.tp_size = 1
        manager._dynamic_resident_bytes = charged_mib * mib
        manager._dynamic_retention_ledger = (
            gpu_cudagraph_utils.DynamicGraphRetentionLedger()
        )
        manager._owns_dynamic_retention_ledger = True
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=4,
            num_reqs=1,
            uniform_token_count=4,
        )
        manager._dynamic_graph_entries = {
            desc: gpu_cudagraph_utils.DynamicGraphEntry(
                descriptor=desc,
                state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
                charged_bytes=charged_mib * mib,
                local_charged_bytes=charged_mib * mib,
                local_pool_bytes=pool_mib * mib,
            )
        }
        managers.append(manager)

    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))
    ledger = working_set.managers[0]._dynamic_retention_ledger
    ledger.local_bytes = 41 * mib
    ledger.rank_safe_bytes = 41 * mib

    # Eviction guarantees pool_bytes - 1 MiB reclaim per HOT entry. The
    # remaining upper bound is 41 + 1 + 43 + 20 = 105 MiB.
    assert working_set.transition_floor_upper_bound_bytes() == 105 * mib
    assert working_set.resident_bytes == (41 + 44 + 44 + 65) * mib


def test_working_set_reuses_one_lazy_capture_stream_across_owners(monkeypatch):
    managers = []
    for owner in ("target", "mtp_prefill", "mtp_decode"):
        manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
        manager.dynamic_graph_owner = owner
        manager.device = torch.device("cuda:2")
        manager._dynamic_resident_bytes = 0
        manager._dynamic_retention_ledger = (
            gpu_cudagraph_utils.DynamicGraphRetentionLedger()
        )
        manager._owns_dynamic_retention_ledger = True
        manager._dynamic_capture_state = (
            gpu_cudagraph_utils.DynamicGraphCaptureState()
        )
        managers.append(manager)

    stream = object()
    stream_calls = []
    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.cuda,
        "Stream",
        lambda *, device: stream_calls.append(device) or stream,
    )

    gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))
    contexts = [
        manager._get_dynamic_graph_capture_context() for manager in managers
    ]

    assert stream_calls == [torch.device("cuda:2")]
    assert len({id(manager._dynamic_capture_state) for manager in managers}) == 1
    assert contexts[0] is contexts[1] is contexts[2]
    assert contexts[0].stream is stream


def test_working_set_rejects_independent_live_capture_streams():
    managers = []
    for owner in ("target", "mtp_prefill"):
        manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
        manager.dynamic_graph_owner = owner
        manager._dynamic_resident_bytes = 0
        manager._dynamic_retention_ledger = (
            gpu_cudagraph_utils.DynamicGraphRetentionLedger()
        )
        manager._owns_dynamic_retention_ledger = True
        manager._dynamic_capture_state = gpu_cudagraph_utils.DynamicGraphCaptureState(
            context=object()
        )
        managers.append(manager)

    with pytest.raises(RuntimeError, match="independent capture streams"):
        gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))


def test_working_set_x0_retires_stale_retention_after_vmm_reconcile():
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.dynamic_graph_owner = "target"
    manager._dynamic_resident_bytes = 0
    manager._dynamic_retention_ledger = gpu_cudagraph_utils.DynamicGraphRetentionLedger(
        local_bytes=4_235_264,
        rank_safe_bytes=4_235_264,
        step_start_local_bytes=4_235_264,
        step_released_capture_state_bytes=2_097_152,
    )
    manager._owns_dynamic_retention_ledger = True
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,))

    assert working_set.resident_bytes == 4_235_264
    assert working_set.active_graph_bytes == 0
    working_set.clear_idle_retention_after_physical_reconcile()

    assert working_set.resident_bytes == 0
    ledger = manager._dynamic_retention_ledger
    assert ledger.local_bytes == 0
    assert ledger.rank_safe_bytes == 0
    assert ledger.step_start_local_bytes == 0
    assert ledger.step_released_capture_state_bytes == 0


def test_working_set_x0_retires_only_ledger_with_pinned_graph_active():
    manager = object.__new__(gpu_cudagraph_utils.CudaGraphManager)
    manager.dynamic_graph_owner = "target"
    manager._dynamic_resident_bytes = 7_340_032
    manager._dynamic_retention_ledger = gpu_cudagraph_utils.DynamicGraphRetentionLedger(
        local_bytes=4_235_264,
        rank_safe_bytes=4_235_264,
        step_start_local_bytes=4_235_264,
        step_released_capture_state_bytes=2_097_152,
    )
    manager._owns_dynamic_retention_ledger = True
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,))

    working_set.clear_idle_retention_after_physical_reconcile()

    assert working_set.active_graph_bytes == 7_340_032
    assert working_set.resident_bytes == 7_340_032


def test_working_set_administrative_x0_evicts_only_unpinned_hot_entries():
    manager = MagicMock()
    manager.dynamic_graph_owner = "target"
    cold = SimpleNamespace(
        state=gpu_cudagraph_utils.DynamicGraphResidency.WARM,
        pinned=False,
        charged_bytes=11,
    )
    pinned = SimpleNamespace(
        state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
        pinned=True,
        charged_bytes=13,
    )
    evictable = SimpleNamespace(
        state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
        pinned=False,
        charged_bytes=17,
        descriptor=MagicMock(),
    )
    evictable.descriptor.physical_replay_key.return_value.identity = ("target", 4)
    manager._dynamic_graph_entries = {
        "cold": cold,
        "pinned": pinned,
        "evictable": evictable,
    }
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,))

    assert working_set.evict_unpinned_for_idle("x0-19") == 17
    manager._evict_dynamic_entry.assert_called_once_with(
        evictable,
        transaction_id="x0-19",
        reason="administrative_idle_x0",
    )


def test_working_set_idle_finish_releases_allocator_before_measurement(monkeypatch):
    events = []
    manager = MagicMock()
    manager.device = torch.device("cuda:2")
    manager.dynamic_resident_bytes = 0
    manager.finish_dynamic_step.side_effect = lambda: events.append("finish")
    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.cuda,
        "synchronize",
        lambda device: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.gc, "collect", lambda: events.append("collect")
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.accelerator,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )

    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,))

    assert working_set.finish_idle_step() == 0
    assert events == [
        "finish",
        ("synchronize", torch.device("cuda:2")),
        "collect",
        "empty_cache",
        ("synchronize", torch.device("cuda:2")),
    ]


def test_dynamic_graph_retains_hot_shape_across_falling_wave_and_x1(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={
            "dynamic_cudagraph_full_capture_sizes": [1, 2, 3],
            "dynamic_cudagraph_budget_mb": 64,
            "dynamic_cudagraph_guard_mb": 0,
            "dynamic_cudagraph_min_hits": 2,
        },
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )
    manager._graphs_captured = True

    # The current wave shape may become hot after repeated use.
    manager.begin_dynamic_step()
    manager.release_unused_dynamic_residency(3, 3, 1, 0)
    manager.dispatch(3, 3, 1, 0)
    manager.dispatch(3, 3, 1, 0)
    assert manager._dynamic_pending is not None
    peak = manager._dynamic_graph_entries[manager._dynamic_pending]
    peak.state = gpu_cudagraph_utils.DynamicGraphResidency.HOT
    peak.charged_bytes = 16
    manager._dynamic_resident_bytes = 16
    manager._dynamic_pending = None

    # Falling X retains the peak and learns the exact next shape independently.
    manager.begin_dynamic_step()
    manager.release_unused_dynamic_residency(2, 2, 1, 0)
    assert manager.dynamic_resident_bytes == 16
    manager.dispatch(2, 2, 1, 0)
    manager.dispatch(2, 2, 1, 0)
    assert manager._dynamic_pending is not None
    assert manager._dynamic_pending.num_reqs == 2
    manager.cancel_pending_dynamic_capture()

    # X1 also gets its exact graph without destroying the retained peak.
    manager.begin_dynamic_step()
    manager.release_unused_dynamic_residency(1, 1, 1, 0)
    manager.dispatch(1, 1, 1, 0)
    manager.dispatch(1, 1, 1, 0)
    assert manager._dynamic_pending is not None
    assert manager._dynamic_pending.num_reqs == 1
    manager.cancel_pending_dynamic_capture()

    # A later rise requests only that new exact shape.
    manager.begin_dynamic_step()
    manager.release_unused_dynamic_residency(2, 2, 1, 0)
    manager.dispatch(2, 2, 1, 0)
    manager.dispatch(2, 2, 1, 0)
    assert manager._dynamic_pending is not None


def test_cancelled_dynamic_capture_revokes_grant_before_retry(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    config = _create_vllm_config(
        additional_config={"elastic_gdn_backing": True},
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=1,
        owner="target",
    )

    first = manager.queue_runtime_descriptor(4, 4, 1, 0, allow_full=True)
    assert first is not None
    manager._dynamic_graph_entries[first].estimated_bytes = 64
    assert manager.prepare_pending_dynamic_capture(64)
    assert manager._dynamic_capture_granted_bytes == 64

    manager.cancel_pending_dynamic_capture()

    assert manager._dynamic_pending is None
    assert manager._dynamic_capture_granted_bytes == 0
    assert manager._dynamic_graph_entries[first].state == (
        gpu_cudagraph_utils.DynamicGraphResidency.WARM
    )

    retry = manager.queue_runtime_descriptor(3, 3, 1, 0, allow_full=True)
    assert retry is not None and retry != first
    manager._dynamic_graph_entries[retry].estimated_bytes = 32
    assert not manager.prepare_pending_dynamic_capture(31)
    assert manager._dynamic_capture_granted_bytes == 0
