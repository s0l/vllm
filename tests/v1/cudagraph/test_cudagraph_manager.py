# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import weakref
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
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
    ElasticAdmissionController,
    ElasticGraphError,
    ElasticPlanKind,
    GraphPrice,
    RuntimeGeneration,
    build_execution_manifest,
    resolve_step_physical_keys,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager
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
    vllm_config.model_config.enforce_eager = False
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

    policy = gpu_cudagraph_utils.graph_execution_policy_from_managers((target, dflash))
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
    forbidden = resolve_step_physical_keys((1, 3, 40, 160, 4), generation, 4096)[0]
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

    # Structured-output masking can invalidate the scheduled draft tokens.
    # The first MTP pass then consumes a qlen=1 semantic tail even though the
    # request cohort still owns its larger physical carrier.  This is the same
    # bounded padding contract as qlen=K+1, not an arbitrary PIECEWISE tail.
    mtp_prefill_q1 = replace(
        x8,
        num_tokens=8,
        uniform_token_count=None,
    )
    manager.dynamic_graph_owner = "mtp_prefill"
    manager.decode_query_len = 4
    assert manager._dynamic_descriptor_matches_step(
        mtp_prefill_q1,
        num_reqs=1,
        num_tokens=1,
        uniform_token_count=1,
        num_active_loras=0,
    )
    assert not manager._dynamic_descriptor_matches_step(
        mtp_prefill_q1,
        num_reqs=1,
        num_tokens=2,
        uniform_token_count=1,
        num_active_loras=0,
    )
    assert not manager._dynamic_descriptor_matches_step(
        mtp_prefill_q1,
        num_reqs=1,
        num_tokens=1,
        uniform_token_count=None,
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
    assert configuration.endswith("fixed_split=2048:disable_split=1:workspace_mib=96")
    assert math == "accepted-batched-q1-split2048-nosplit-forced-prefix-v1"

    monkeypatch.setenv("AG2_VLLM_MTP_DCP_BATCHED_WORKSPACE_MIB", "64")
    assert gpu_cudagraph_utils._effective_mtp_verifier_contract()[2] == (
        "pending-batched-q1-product-math-v1"
    )


def test_owner_prequant_semantics_require_graph_backed_carrier(monkeypatch):
    monkeypatch.setattr(gpu_cudagraph_utils.envs, "AG2_VLLM_TP3_OWNER_MIN_ROWS", 128)
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
    monkeypatch,
    caplog,
    owner,
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
    planned_1024 = manager.queue_runtime_descriptor(1, 1024, None, 0, allow_full=False)
    assert planned_1024 is not None and planned_1024.cg_mode == CUDAGraphMode.NONE
    assert manager.dispatch(1, 1024, None, 0) == planned_1024
    assert not manager._dynamic_step_candidates
    assert not manager._dynamic_graph_entries

    manager.begin_dynamic_step()
    assert manager.is_compiled_piecewise_shape(1152)
    planned_1152 = manager.queue_runtime_descriptor(2, 1152, None, 0, allow_full=False)
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
    monkeypatch,
    owner,
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
    queued_exact = manager.queue_runtime_descriptor(1, 64, None, 0, allow_full=False)
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
    recovered = manager.queue_runtime_descriptor(12, 64, None, 0, allow_full=False)
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
    from vllm.v1.core import elastic_runtime
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
        elastic_runtime,
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
            num_active_loras=0,
            physical_num_reqs=4,
            runtime_generation="test-generation",
            semantic_decode=True,
        ),
        BatchExecutionDescriptor(
            CUDAGraphMode.PIECEWISE,
            3,
            None,
            None,
            num_active_loras=0,
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
    from vllm.v1.core import elastic_runtime
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
        elastic_runtime,
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
    manager._dynamic_graph_entries[descriptor] = gpu_cudagraph_utils.DynamicGraphEntry(
        descriptor=descriptor
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


def test_elastic_runtime_rejects_malformed_legacy_knob_before_validation(
    monkeypatch,
):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    config = _create_vllm_config(
        additional_config={
            "elastic_gdn_backing": True,
            "dynamic_cudagraph_capture_sizes": "not-a-list",
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


def test_worker_validates_q1_manifest_before_any_graph_mutation() -> None:
    generation = RuntimeGeneration("worker-current-q1")

    def manager(
        owner: str,
        *,
        activation: str,
        token_source: str,
        decode_query_len: int,
    ) -> MagicMock:
        result = MagicMock()
        result.dynamic_graph_owner = owner
        result.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        result._runtime_decode_query_lens.return_value = {decode_query_len}
        result.compiled_piecewise_sizes = frozenset()
        result.tp3_owner_prequant = False
        result.elastic_graph_activation = activation
        result.elastic_graph_token_source = token_source
        result.elastic_graph_fixed_query_len = None
        return result

    target = manager(
        "target", activation="always", token_source="step", decode_query_len=1
    )
    mtp_prefill = manager(
        "mtp_prefill",
        activation="speculative",
        token_source="step",
        decode_query_len=1,
    )
    mtp_decode = manager(
        "mtp_decode",
        activation="speculative",
        token_source="requests",
        decode_query_len=1,
    )
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(
        (target, mtp_prefill, mtp_decode)
    )
    policy = gpu_cudagraph_utils.graph_execution_policy_from_managers(
        working_set.managers
    )
    step_key = (0, 3, 1, 1, 0)
    manifest, dispatch = build_execution_manifest(
        step_key=step_key,
        request_ids=("request-0",),
        per_request_query_lens=(1,),
        per_request_is_prefilling=(True,),
        scheduled_draft_rows=(0,),
        requested_output_k=3,
        executed_drafter_k=3,
        phase="mixed",
        generation=generation,
        policy=policy,
        max_num_batched_tokens=4096,
    )
    current_keys = tuple(
        item.physical_key for item in dispatch if item.physical_key is not None
    )
    successor_keys = resolve_step_physical_keys(
        (0, 3, 1, 4, 4),
        generation,
        max_num_batched_tokens=4096,
        policy=policy,
    )
    successor_only = tuple(key for key in successor_keys if key not in current_keys)
    residency_union = tuple((*current_keys, *successor_only))
    controller = ElasticAdmissionController(generation)
    for graph_key in residency_union:
        controller.publish_hot(
            graph_key,
            GraphPrice(1, 1, f"test:{graph_key.identity}"),
            pinned=False,
        )
    plan = replace(
        controller.plan(
            "q1-user",
            residency_union,
            request_bytes=3,
            available_bytes=10,
            protected_keys=successor_only,
        ),
        execution_manifest=manifest,
        current_dispatch=dispatch,
        successor_keys=successor_keys,
    )

    working_set.validate_execution_manifest(
        replace(plan, execution_manifest=None, current_dispatch=()),
        step_key=None,
        request_ids=(),
        per_request_query_lens=(),
        per_request_is_prefilling=(),
        scheduled_draft_rows=(),
        requested_output_k=0,
        executed_drafter_k=0,
        phase=None,
        max_num_batched_tokens=4096,
    )

    working_set.validate_execution_manifest(
        plan,
        step_key=step_key,
        request_ids=("request-0",),
        per_request_query_lens=(1,),
        per_request_is_prefilling=(True,),
        scheduled_draft_rows=(0,),
        requested_output_k=3,
        executed_drafter_k=3,
        phase="mixed",
        max_num_batched_tokens=4096,
    )

    with pytest.raises(
        gpu_cudagraph_utils.ElasticExecutionPlanMismatch,
        match=(
            r"fields=request_ids\[0\]: planned='request-0' "
            r"observed='request-1' lengths=1/1"
        ),
    ):
        working_set.validate_execution_manifest(
            plan,
            step_key=step_key,
            request_ids=("request-1",),
            per_request_query_lens=(1,),
            per_request_is_prefilling=(True,),
            scheduled_draft_rows=(0,),
            requested_output_k=3,
            executed_drafter_k=3,
            phase="mixed",
            max_num_batched_tokens=4096,
        )

    with pytest.raises(
        gpu_cudagraph_utils.ElasticExecutionPlanMismatch,
        match="cannot inherit a successor carrier",
    ):
        working_set.validate_execution_manifest(
            plan,
            step_key=(0, 3, 1, 4, 4),
            request_ids=("request-0",),
            per_request_query_lens=(1,),
            per_request_is_prefilling=(True,),
            scheduled_draft_rows=(0,),
            requested_output_k=3,
            executed_drafter_k=3,
            phase="mixed",
            max_num_batched_tokens=4096,
        )
    with pytest.raises(
        gpu_cudagraph_utils.ElasticExecutionPlanMismatch,
        match=(
            r"fields=per_request_is_prefilling\[0\]: planned=True "
            r"observed=False lengths=1/1"
        ),
    ):
        working_set.validate_execution_manifest(
            plan,
            step_key=step_key,
            request_ids=("request-0",),
            per_request_query_lens=(1,),
            per_request_is_prefilling=(False,),
            scheduled_draft_rows=(0,),
            requested_output_k=3,
            executed_drafter_k=3,
            phase="mixed",
            max_num_batched_tokens=4096,
        )
    for owner in working_set.managers:
        owner.evict_physical_key.assert_not_called()
        owner.queue_physical_key.assert_not_called()
        owner.finish_dynamic_step.assert_not_called()

    for owner in working_set.managers:
        owner.is_physical_key_hot.return_value = True
        owner._dynamic_pending = None
    working_set.apply_plan(plan)
    current_by_owner = {key.logical.owner: key for key in current_keys}
    for owner in working_set.managers:
        owner.queue_physical_key.assert_called_once_with(
            current_by_owner[owner.dynamic_graph_owner]
        )
    assert successor_only
    assert all(
        call.args[0] not in successor_only
        for owner in working_set.managers
        for call in owner.queue_physical_key.call_args_list
    )
    for key in successor_only:
        next(
            owner
            for owner in working_set.managers
            if owner.dynamic_graph_owner == key.logical.owner
        ).acquire_physical_key_lease.assert_any_call(key, plan.transaction_id)


def _worker_consensus_fixture():
    generation = RuntimeGeneration("worker-consensus")
    managers = tuple(MagicMock() for _ in range(3))
    for owner, manager in zip(
        ("target", "unused-1", "unused-2"), managers, strict=True
    ):
        manager.dynamic_graph_owner = owner
        manager.runtime_generation = generation.value
        manager.tp_size = 3
        manager.is_physical_key_hot.return_value = owner == "target"
        manager._dynamic_pending = None
    working_set = gpu_cudagraph_utils.DynamicGraphWorkingSet(managers)
    policy = gpu_cudagraph_utils.GraphExecutionPolicy(
        verifier_contract="consensus-v1",
        math_contract="consensus-math-v1",
        owners=(
            gpu_cudagraph_utils.OwnerGraphExecutionPolicy(
                "target",
                (1,),
                "PIECEWISE",
                activation="always",
                token_source="step",
                execution_order=0,
            ),
        ),
    )
    step_key = (1, 0, 1, 1, 1)
    manifest, dispatch = build_execution_manifest(
        step_key=step_key,
        request_ids=("request-0",),
        per_request_query_lens=(1,),
        per_request_is_prefilling=(False,),
        scheduled_draft_rows=(0,),
        requested_output_k=0,
        executed_drafter_k=0,
        phase="decode",
        generation=generation,
        policy=policy,
        max_num_batched_tokens=4096,
    )
    graph_key = dispatch[0].physical_key
    assert graph_key is not None
    controller = ElasticAdmissionController(generation)
    controller.publish_hot(graph_key, GraphPrice(1, 1, "consensus"), pinned=False)
    plan = replace(
        controller.plan(
            "consensus-user",
            (graph_key,),
            request_bytes=1,
            available_bytes=2,
        ),
        execution_manifest=manifest,
        current_dispatch=dispatch,
    )
    return working_set, managers, plan


def _assert_no_worker_plan_mutation(managers) -> None:
    for manager in managers:
        manager.begin_dynamic_step.assert_not_called()
        manager.acquire_physical_key_lease.assert_not_called()
        manager.evict_physical_key.assert_not_called()
        manager.queue_physical_key.assert_not_called()


def test_repeated_manifest_rank_error_votes_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_set, managers, plan = _worker_consensus_fixture()

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )

    vote_count = 0
    diagnostic_count = 0

    def gather_vote(output, value, **_kwargs):
        nonlocal vote_count
        vote_count += 1
        output.view(3, 5)[:] = value
        if vote_count == 2:
            output.view(3, 5)[1, 4] = 1

    def gather_diagnostics(outputs, value, **_kwargs):
        nonlocal diagnostic_count
        diagnostic_count += 1
        outputs[:] = [
            value,
            (value[0], "rank-1 rejected repeated manifest"),
            value,
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather_vote)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_diagnostics)
    # Establish one accepted vote, then replay the exact same fingerprint.
    # The repeat must still enter a collective because a peer can have a
    # rank-local error that is invisible here.
    working_set.require_rank_consensus(plan)
    with pytest.raises(
        gpu_cudagraph_utils.ElasticExecutionPlanMismatch,
        match="rank-1 rejected repeated manifest",
    ):
        working_set.begin_admitted_step(plan)
    # A rejected status must not poison the reusable fixed-size vote buffers.
    assert working_set.require_rank_consensus(plan) == plan.fingerprint
    assert vote_count == 3
    assert diagnostic_count == 1
    _assert_no_worker_plan_mutation(managers)


@pytest.mark.parametrize("local_error", [None, "rank-0 materialization drift"])
def test_post_materialization_vote_converges_rank_local_rejection(
    monkeypatch: pytest.MonkeyPatch,
    local_error: str | None,
) -> None:
    working_set, _managers, plan = _worker_consensus_fixture()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )
    votes = 0

    def gather_vote(output, value, **_kwargs):
        nonlocal votes
        votes += 1
        output.view(3, 5)[:] = value
        # Regardless of which local rank this unit instance represents, the
        # shared vote reports one rank-local materialization rejection.
        output.view(3, 5)[1, 4] = 1

    def gather_diagnostics(outputs, value, **_kwargs):
        outputs[:] = [
            (value[0], local_error),
            (value[0], "rank-1 materialization drift"),
            (value[0], None),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather_vote)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_diagnostics)

    with pytest.raises(
        RuntimeError,
        match="all-rank input materialization vote rejected before model collectives",
    ):
        working_set.require_post_materialization_consensus(
            plan,
            validation_error=local_error,
        )

    assert votes == 1
    calls, total_ns = working_set.rank_consensus_timing_snapshot()[
        "post_materialization"
    ]
    assert calls == 1
    assert total_ns > 0


def test_post_materialization_vote_rejects_observer_fingerprint_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_set, _managers, plan = _worker_consensus_fixture()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )

    def gather_vote(output, value, **_kwargs):
        output.view(3, 5)[:] = value
        output.view(3, 5)[1, 0] ^= 1

    def gather_diagnostics(outputs, value, **_kwargs):
        outputs[:] = [value, ("f" * 64, None), value]

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather_vote)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_diagnostics)

    with pytest.raises(RuntimeError, match="vote rejected before model collectives"):
        working_set.require_post_materialization_consensus(
            plan,
            observer_fingerprint="a" * 64,
            phase="post_mm_materialization",
        )


def test_one_rank_residency_drift_rejects_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_set, managers, plan = _worker_consensus_fixture()
    # Simulate rank 1 losing the scheduler-declared HOT executable while ranks
    # 0 and 2 still have the same immutable plan.
    managers[0].is_physical_key_hot.return_value = False
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )

    def gather_vote(output, value, **_kwargs):
        assert int(value[4]) == 1
        output.view(3, 5)[:] = value
        output.view(3, 5)[0, 4] = 0
        output.view(3, 5)[2, 4] = 0

    def gather_diagnostics(outputs, value, **_kwargs):
        assert value[1] is not None
        assert "HOT hit absent from worker residency" in value[1]
        outputs[:] = [(value[0], None), value, (value[0], None)]

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather_vote)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_diagnostics)
    with pytest.raises(
        gpu_cudagraph_utils.ElasticExecutionPlanMismatch,
        match="HOT hit absent from worker residency",
    ):
        working_set.begin_admitted_step(plan)
    _assert_no_worker_plan_mutation(managers)


def test_victim_current_overlap_rejects_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_set, managers, plan = _worker_consensus_fixture()
    graph_key = plan.physical_keys[0]
    with pytest.raises(ValueError, match="current physical key"):
        replace(plan, victim_keys=(graph_key,))

    # Bypass the frozen dataclass constructor to simulate corrupt transport and
    # prove that the independent worker validator still rejects it.
    corrupted_plan = copy(plan)
    object.__setattr__(corrupted_plan, "victim_keys", (graph_key,))
    corrupted_plan.__dict__.pop("fingerprint", None)
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )

    def gather_vote(output, value, **_kwargs):
        assert int(value[4]) == 1
        output.view(3, 5)[:] = value
        output.view(3, 5)[0, 4] = 0
        output.view(3, 5)[2, 4] = 0

    def gather_diagnostics(outputs, value, **_kwargs):
        assert "selected as an eviction victim" in value[1]
        outputs[:] = [(value[0], None), value, (value[0], None)]

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather_vote)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_diagnostics)
    with pytest.raises(
        gpu_cudagraph_utils.ElasticExecutionPlanMismatch,
        match="selected as an eviction victim",
    ):
        working_set.begin_admitted_step(corrupted_plan)
    _assert_no_worker_plan_mutation(managers)


def test_victim_protected_overlap_rejects_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_set, managers, plan = _worker_consensus_fixture()
    graph_key = plan.physical_keys[0]
    protected_key = replace(
        graph_key,
        logical=replace(graph_key.logical, token_bucket=2),
    )
    with pytest.raises(ValueError, match="protected key"):
        replace(
            plan,
            protected_keys=(protected_key,),
            victim_keys=(protected_key,),
        )

    # Simulate a corrupt decoded object, bypassing the constructor just as a
    # transport/runtime type boundary could. Worker validation must remain a
    # complete guard because applying this plan would acquire the lease before
    # discovering that the same entry cannot be evicted.
    corrupted_plan = copy(plan)
    object.__setattr__(corrupted_plan, "protected_keys", (protected_key,))
    object.__setattr__(corrupted_plan, "victim_keys", (protected_key,))
    corrupted_plan.__dict__.pop("fingerprint", None)
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )

    def gather_vote(output, value, **_kwargs):
        assert int(value[4]) == 1
        output.view(3, 5)[:] = value
        output.view(3, 5)[0, 4] = 0
        output.view(3, 5)[2, 4] = 0

    def gather_diagnostics(outputs, value, **_kwargs):
        assert "protected physical key" in value[1]
        outputs[:] = [(value[0], None), value, (value[0], None)]

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather_vote)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_diagnostics)
    with pytest.raises(
        gpu_cudagraph_utils.ElasticExecutionPlanMismatch,
        match="protected physical key",
    ):
        working_set.begin_admitted_step(corrupted_plan)
    _assert_no_worker_plan_mutation(managers)


def test_rank_plan_hash_drift_rejects_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_set, managers, plan = _worker_consensus_fixture()
    divergent_plan = replace(plan, transaction_id="rank-2-divergent")
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )

    def gather_vote(output, value, **_kwargs):
        output.view(3, 5)[:] = value
        output.view(3, 5)[2, 0] ^= 1

    def gather_diagnostics(outputs, value, **_kwargs):
        outputs[:] = [value, value, (divergent_plan.fingerprint, None)]

    monkeypatch.setattr(torch.distributed, "all_gather_single", gather_vote)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_diagnostics)
    with pytest.raises(
        gpu_cudagraph_utils.ElasticExecutionPlanMismatch,
        match="fingerprint differs by rank",
    ):
        working_set.begin_admitted_step(plan)
    _assert_no_worker_plan_mutation(managers)


def test_rank_vote_fast_path_is_fixed_size_and_skips_object_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_set, _managers, plan = _worker_consensus_fixture()
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )
    vote_count = 0

    def gather_vote(output, value, **_kwargs):
        nonlocal vote_count
        vote_count += 1
        assert value.shape == (5,)
        assert value.dtype == torch.int64
        assert value.device.type == "cpu"
        output.view(3, 5)[:] = value

    object_gather = MagicMock()
    monkeypatch.setattr(torch.distributed, "all_gather_single", gather_vote)
    monkeypatch.setattr(torch.distributed, "all_gather_object", object_gather)

    iterations = 128
    started = time.perf_counter()
    for _ in range(iterations):
        assert working_set.require_rank_consensus(plan) == plan.fingerprint
    elapsed = time.perf_counter() - started

    assert vote_count == iterations
    object_gather.assert_not_called()
    # This bounds only Python/tensor bookkeeping with a mocked collective; it
    # is a regression sentinel, not a claim about real TP/Gloo latency.
    assert elapsed < 1.0


def test_dynamic_working_set_executes_immutable_piecewise_first_capture_order():
    generation = RuntimeGeneration("worker-m160-order")
    desired = resolve_step_physical_keys(
        (1, 3, 40, 160, 4),
        generation,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(256, 512, 1024, 2048, 4096),
    )
    cache = ElasticAdmissionController(generation)
    plan = cache.plan(
        "worker-m160-order",
        desired,
        request_bytes=0,
        available_bytes=512,
        destination_capture_endpoint_bytes=200,
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


@pytest.mark.parametrize(
    ("kind", "administrative"),
    [
        (ElasticPlanKind.MAINTENANCE, False),
        (ElasticPlanKind.RECLAIM, True),
        (ElasticPlanKind.PRESSURE_RECLAIM, True),
    ],
)
def test_dynamic_working_set_forwards_reclaim_authority(kind, administrative):
    generation = RuntimeGeneration("worker-idle-reclaim")
    graph_key = resolve_step_physical_keys(
        (1, 3, 1, 1, 1),
        generation,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(),
    )[0]
    manager = MagicMock()
    manager.dynamic_graph_owner = graph_key.logical.owner
    manager._dynamic_pending = None
    plan = SimpleNamespace(
        kind=kind,
        transaction_id="idle-x0",
        generation=generation,
        protected_keys=(),
        victim_keys=(graph_key,),
        physical_keys=(),
        hot_hits=(),
        cold_misses=(),
        capture_order=(),
    )

    gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,)).apply_plan(plan)

    manager.evict_physical_key.assert_called_once_with(
        graph_key,
        transaction_id="idle-x0",
        reason="elastic_admission_plan",
        administrative=administrative,
    )


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
    legacy_authorities = (
        "dynamic_graph_budget_bytes",
        "dynamic_graph_max_entry_bytes",
        "dynamic_graph_guard_bytes",
        "dynamic_graph_min_hits",
        "dynamic_graph_full_min_hits",
        "dynamic_graph_piecewise_min_hits",
        "dynamic_graph_piecewise_min_padding_pct",
        "dynamic_graph_cooldown_steps",
        "dynamic_graph_pinned_sizes",
    )
    assert not any(hasattr(manager, name) for name in legacy_authorities)
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
    entry = manager._dynamic_graph_entries[desc]
    manager._cooldown_dynamic_entry(entry)
    assert entry.state == gpu_cudagraph_utils.DynamicGraphResidency.WARM
    assert entry.cooldown_until_epoch == 0


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
def test_elastic_batched_q1_decode_uses_bounded_piecewise_m(monkeypatch, owner, x):
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
    assert not manager.is_compiled_piecewise_shape(m, num_reqs=x, semantic_decode=True)


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
    assert manager.is_compiled_piecewise_shape(32, num_reqs=8, semantic_decode=False)
    assert not manager.uses_tp3_owner_prequant_decode(generic)

    decode = manager.runtime_descriptor(
        8, 32, 4, 0, allow_full=False, semantic_decode=True
    )
    assert decode is not None and decode.semantic_decode
    assert not manager.is_compiled_piecewise_shape(32, num_reqs=8, semantic_decode=True)
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
    missing_semantics = manager.runtime_descriptor(40, 160, None, 0, allow_full=False)

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
    ("capture_delta_bytes", "private_pool_bytes", "expected_charge"),
    [
        (64, 48, 64),
        (32, 48, 48),
        (0, 48, 48),
    ],
)
def test_dynamic_capture_charge_includes_allocator_reused_private_pool(
    capture_delta_bytes: int,
    private_pool_bytes: int,
    expected_charge: int,
):
    assert (
        gpu_cudagraph_utils.dynamic_capture_physical_charge(
            capture_delta_bytes,
            private_pool_bytes,
        )
        == expected_charge
    )


def test_dynamic_capture_charge_rejects_negative_measurement():
    with pytest.raises(ValueError, match="must be non-negative"):
        gpu_cudagraph_utils.dynamic_capture_physical_charge(-1, 0)


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
            descriptor not in wrapper.concrete_cudagraph_entries for wrapper in wrappers
        )
        assert all(entry.output is None for entry in entries)
        reset_order.append(order)

    graphs[0].reset.side_effect = lambda: reset(10)
    graphs[1].reset.side_effect = lambda: reset(11)
    monkeypatch.setattr(CUDAGraphWrapper, "_all_instances", wrappers)

    assert CUDAGraphWrapper.evict_batch_descriptor(descriptor, private_pool) == 2
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

    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 2, 2, 1, num_active_loras=0)
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
        manager._dynamic_capture_state = gpu_cudagraph_utils.DynamicGraphCaptureState()
        managers.append(manager)

    stream = object()
    stream_calls = []
    monkeypatch.setattr(
        gpu_cudagraph_utils.torch.cuda,
        "Stream",
        lambda *, device: stream_calls.append(device) or stream,
    )

    gpu_cudagraph_utils.DynamicGraphWorkingSet(tuple(managers))
    contexts = [manager._get_dynamic_graph_capture_context() for manager in managers]

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


def test_residency_receipt_preserves_pinned_proof_but_not_leased_proof():
    pinned_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=1,
        num_reqs=1,
        uniform_token_count=1,
        runtime_generation="receipt-generation",
    )
    leased_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4,
        num_reqs=1,
        uniform_token_count=4,
        runtime_generation="receipt-generation",
    )
    evictable_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=8,
        num_reqs=2,
        uniform_token_count=4,
        runtime_generation="receipt-generation",
    )
    entries = {
        pinned_desc: gpu_cudagraph_utils.DynamicGraphEntry(
            descriptor=pinned_desc,
            pinned=True,
            state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
            charged_bytes=64,
            local_pool_bytes=48,
            reclaimable_bytes=32,
        ),
        leased_desc: gpu_cudagraph_utils.DynamicGraphEntry(
            descriptor=leased_desc,
            state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
            charged_bytes=64,
            local_pool_bytes=48,
            reclaimable_bytes=32,
            leases={"txn"},
        ),
        evictable_desc: gpu_cudagraph_utils.DynamicGraphEntry(
            descriptor=evictable_desc,
            state=gpu_cudagraph_utils.DynamicGraphResidency.HOT,
            charged_bytes=64,
            local_pool_bytes=48,
            reclaimable_bytes=32,
        ),
    }
    manager = SimpleNamespace(
        dynamic_graph_owner="target",
        runtime_generation="receipt-generation",
        _dynamic_graph_entries=entries,
    )

    receipt = gpu_cudagraph_utils.DynamicGraphWorkingSet((manager,)).residency_receipt(
        transaction_id=None,
        resident_bytes=192,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=192,
        cublas_workspace_bytes=0,
    )

    by_key = {entry.key: entry for entry in receipt.entries}
    assert by_key[pinned_desc.physical_replay_key("target")].reclaimable_bytes == 32
    assert by_key[leased_desc.physical_replay_key("target")].reclaimable_bytes == 0
    assert by_key[evictable_desc.physical_replay_key("target")].reclaimable_bytes == 32


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


def test_piecewise_capture_uses_pcp_dummy_slot_mappings():
    num_reqs = 32
    num_tokens = 56
    pcp_world_size = 2
    input_buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))

    pcp_block_tables = SimpleNamespace(
        num_kv_cache_groups=1,
        input_block_tables=(torch.zeros(num_reqs * 2, 1, dtype=torch.int32),),
    )
    pcp_manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=num_reqs,
        max_num_tokens=num_tokens,
        block_tables=pcp_block_tables,
    )

    block_tables = MagicMock()
    block_tables.cp_size = 1
    block_tables.get_dummy_block_tables.return_value = ()
    block_tables.get_dummy_slot_mappings.return_value = torch.zeros(
        1, num_tokens, dtype=torch.int64
    )
    model_state = MagicMock()
    model_state.prepare_attn.return_value = {}
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    gpu_cudagraph_utils.prepare_inputs_to_capture(
        num_reqs,
        num_tokens,
        model_state,
        input_buffers,
        block_tables,
        [],
        kv_cache_config,
        full_cudagraph=False,
        pcp_manager=pcp_manager,
    )

    slot_mappings = model_state.prepare_attn.call_args.args[3]
    assert slot_mappings.shape == (1, num_tokens * pcp_world_size)
    block_tables.get_dummy_slot_mappings.assert_not_called()


def _create_decode_vllm_config(
    capture_sizes: list[int],
    num_speculative_tokens: int = 0,
    dynamic_spec_schedule: list[tuple[int, int, int]] | None = None,
) -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=capture_sizes,
    )
    compilation_config.max_cudagraph_capture_size = capture_sizes[-1]
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=8)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.num_speculative_tokens = num_speculative_tokens
    if dynamic_spec_schedule is None:
        vllm_config.speculative_config = None
    else:
        speculative_config = MagicMock()
        speculative_config.uses_dynamic_speculative_decoding.return_value = True
        speculative_config.num_speculative_tokens_per_batch_size = dynamic_spec_schedule
        vllm_config.speculative_config = speculative_config
    return vllm_config


_DECODE_QUERY_LEN = 3


def _make_spec_decode_manager(
    monkeypatch,
    decode_query_len: int = _DECODE_QUERY_LEN,
    capture_sizes: list[int] | None = None,
    num_speculative_tokens: int = 0,
    dynamic_spec_schedule: list[tuple[int, int, int]] | None = None,
) -> gpu_cudagraph_utils.CudaGraphManager:
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
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_decode_vllm_config(
            capture_sizes or [1, 2, 4, 8, 16, 24],
            num_speculative_tokens=num_speculative_tokens,
            dynamic_spec_schedule=dynamic_spec_schedule,
        ),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=decode_query_len,
    )
    manager._graphs_captured = True
    return manager


def test_uniform_multitoken_decode_keeps_exact_full_graph_shape(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=4,
        num_tokens=12,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count == _DECODE_QUERY_LEN
    # Multi-token FULL replay is request-shape exact: padding it to another
    # request geometry can change MTP state ownership and collective order.
    assert desc.num_tokens == 12
    assert desc.num_reqs == 4


def test_uniform_decode_exact_match_is_not_over_padded(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=3,
        num_tokens=9,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_tokens == 9
    assert desc.num_reqs == 3


def test_mixed_batch_never_selects_a_uniform_decode_graph(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=2,
        num_tokens=12,
        uniform_token_count=None,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.uniform_token_count is None
    assert desc.num_tokens == 16


def test_mixed_batch_at_decode_only_token_count_still_gets_a_graph(monkeypatch):
    """A mixed batch must not fall to eager where only decode graphs are staged.

    ``round_up(size, decode_query_len)`` lands FULL decode graphs on token counts
    the PIECEWISE ladder never uses -- with capture sizes [1, 2, 4, 8, 16, 24]
    and query length 3, FULL covers 3, 6, 9 and 18 while PIECEWISE has only
    1, 2, 4, 8, 16, 24. Building each mode's candidate ranges independently keeps
    a PIECEWISE graph reachable there. Deriving the ranges from the staged sizes
    instead offers a mixed batch nothing but decode descriptors, whose
    ``uniform_token_count`` it can never match, so it runs fully eager.
    """
    manager = _make_spec_decode_manager(monkeypatch)

    decode_only_token_counts = sorted(
        {desc.num_tokens for desc in manager._capture_descs[CUDAGraphMode.FULL]}
        - {desc.num_tokens for desc in manager._capture_descs[CUDAGraphMode.PIECEWISE]}
    )
    # Guard the premise: with no such token counts this test would cover nothing.
    assert decode_only_token_counts

    for num_tokens in decode_only_token_counts:
        desc = manager.dispatch(
            num_reqs=1,
            num_tokens=num_tokens,
            uniform_token_count=None,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.PIECEWISE, num_tokens
        assert desc.uniform_token_count is None, num_tokens
        assert desc.num_tokens >= num_tokens, num_tokens


def test_uniform_decode_beyond_capture_ladder_falls_back(monkeypatch):
    manager = _make_spec_decode_manager(monkeypatch)

    desc = manager.dispatch(
        num_reqs=9,
        num_tokens=27,
        uniform_token_count=_DECODE_QUERY_LEN,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.NONE


def test_dynamic_spec_decode_shared_token_count_stays_reachable(monkeypatch):
    """Every captured decode graph must remain reachable from dispatch().

    Under dynamic speculative decoding ``decode_query_lens`` is a list, and the
    staging loop rounds each capture size up to *every* query length. Several
    decode graphs therefore land on the same ``num_tokens`` -- e.g. capture size
    2 stages both ``round_up(2, 1) == 2`` and ``round_up(2, 2) == 2``. Building
    the candidate ranges one descriptor at a time would give all but the first
    of those an empty range, so they would never enter a candidate list and
    their batches would silently fall through to PIECEWISE.
    """
    manager = _make_spec_decode_manager(
        monkeypatch,
        decode_query_len=4,
        capture_sizes=[2, 4, 6, 8],
        num_speculative_tokens=3,
        # max_num_seqs is 8, and K narrows as the batch grows, so the schedule
        # covers K 3, 2, 1 and 0, giving decode_query_lens [1, 2, 3, 4].
        dynamic_spec_schedule=[(1, 2, 3), (3, 4, 2), (5, 6, 1), (7, 8, 0)],
    )

    full_descs = manager._capture_descs[CUDAGraphMode.FULL]
    by_num_tokens: dict[int, list] = defaultdict(list)
    for desc in full_descs:
        by_num_tokens[desc.num_tokens].append(desc)
    # Guard the premise: without collisions this test would not cover the bug.
    assert any(len(descs) > 1 for descs in by_num_tokens.values())

    for desc in full_descs:
        assert desc in manager._candidates[(desc.num_tokens, 0)], desc
        assert (
            manager.dispatch(
                num_reqs=desc.num_reqs,
                num_tokens=desc.num_tokens,
                uniform_token_count=desc.uniform_token_count,
                num_active_loras=0,
            )
            == desc
        ), desc


@pytest.mark.parametrize(
    "decode_query_len,capture_sizes",
    [(1, [1, 2, 4, 8]), (2, [2, 4, 8, 16]), (8, [8, 16, 32, 64])],
)
def test_divisor_query_len_dispatch_preserves_exact_request_shape(
    monkeypatch, decode_query_len, capture_sizes
):
    manager = _make_spec_decode_manager(
        monkeypatch,
        decode_query_len=decode_query_len,
        capture_sizes=capture_sizes,
    )

    max_reqs = capture_sizes[-1] // decode_query_len
    for num_reqs in range(1, max_reqs + 1):
        num_tokens = num_reqs * decode_query_len
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            uniform_token_count=decode_query_len,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL, num_tokens
        if decode_query_len == 1:
            expected = min(s for s in capture_sizes if s >= num_tokens)
            assert desc.num_tokens == expected
            assert desc.num_reqs == expected
        else:
            assert desc.num_tokens == num_tokens
            assert desc.num_reqs == num_reqs
