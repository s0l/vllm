from types import SimpleNamespace

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.k3_elastic_graph import (
    K3ElasticGraphPlan,
    K3ElasticGraphPlanError,
    K3ElasticGraphRegistry,
    K3ElasticWorkloadKey,
    context_bucket,
    kv_pressure_bucket,
    make_decode_slices,
    prefill_bucket,
)
from vllm.v1.worker.gpu.k3_elastic_runtime import K3ElasticRuntime
import vllm.v1.worker.gpu.k3_elastic_runtime as runtime_module
from vllm.v1.worker.ubatch_utils import UBatchSlice


def plan_config(**updates):
    value = {
        "name": "x38-m128-m24",
        "runtime_identity": "runtime-a",
        "partition_x": [32, 6],
        "rows_per_request": [4],
        "modes": ["decode"],
        "context_buckets": ["0-64k", "64k-128k"],
        "kv_pressure_buckets": ["0-50pct"],
        "prefill_buckets": ["none"],
        "graph_families": ["captured-k3-target"],
        "proof_sha256": "a" * 64,
        "implementation_sha256": "b" * 64,
        "state_bytes": 4096,
        "priority": 1,
    }
    value.update(updates)
    return value


def workload_key(**updates):
    value = {
        "runtime_identity": "runtime-a",
        "x": 38,
        "rows_per_request": 4,
        "mode": "decode",
        "context_bucket": "64k-128k",
        "kv_pressure_bucket": "0-50pct",
        "prefill_bucket": "none",
        "graph_family": "captured-k3-target",
    }
    value.update(updates)
    return K3ElasticWorkloadKey(**value)


def test_registry_matches_complete_envelope_and_builds_unequal_slices():
    registry = K3ElasticGraphRegistry.from_config([plan_config()])
    selected = registry.candidates(workload_key(), state_bytes=4096)

    assert len(selected) == 1
    assert selected[0].request_slices() == (slice(0, 32), slice(32, 38))
    assert selected[0].token_slices(4) == (slice(0, 128), slice(128, 152))


def test_registry_fails_closed_across_every_workload_dimension():
    registry = K3ElasticGraphRegistry.from_config([plan_config()])
    mutations = (
        {"runtime_identity": "runtime-b"},
        {"x": 37},
        {"rows_per_request": 3},
        {"mode": "mixed"},
        {"context_bucket": "128k-256k"},
        {"kv_pressure_bucket": "50-100pct"},
        {"prefill_bucket": "1k-4k"},
        {"graph_family": "piecewise-k3"},
    )
    for mutation in mutations:
        assert not registry.candidates(workload_key(**mutation), state_bytes=4096)


def test_registry_rejects_malformed_or_unproved_plans():
    malformed = (
        plan_config(partition_x=[6, 32]),
        plan_config(partition_x=[38]),
        plan_config(proof_sha256="short"),
        plan_config(rows_per_request=[]),
        plan_config(state_bytes=-1),
    )
    for value in malformed:
        try:
            K3ElasticGraphPlan.from_dict(value)
        except K3ElasticGraphPlanError:
            pass
        else:
            raise AssertionError(f"malformed plan passed: {value}")


def test_registry_orders_candidates_without_hardcoding_a_quantum():
    registry = K3ElasticGraphRegistry.from_config(
        [
            plan_config(name="x38-m128-m24", partition_x=[32, 6], priority=2),
            plan_config(name="x38-m96-m56", partition_x=[24, 14], priority=1),
        ]
    )
    assert [
        plan.name for plan in registry.candidates(workload_key(), state_bytes=4096)
    ] == ["x38-m96-m56", "x38-m128-m24"]


def test_decode_slices_require_exact_unpadded_k3_shape_not_dispatch_mode():
    plan = K3ElasticGraphPlan.from_dict(plan_config())
    input_batch = SimpleNamespace(
        num_reqs=38,
        num_tokens=152,
        num_tokens_after_padding=152,
        num_scheduled_tokens=np.full(38, 4, dtype=np.int32),
        is_prefilling_np=np.zeros(38, dtype=np.bool_),
        num_draft_tokens_per_req=np.full(38, 3, dtype=np.int32),
    )
    full_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=152,
        num_reqs=38,
    )

    slices = make_decode_slices(plan, input_batch, full_desc, decode_query_len=4)
    assert slices is not None
    assert [(item.request_slice, item.token_slice) for item in slices] == [
        (slice(0, 32), slice(0, 128)),
        (slice(32, 38), slice(128, 152)),
    ]

    eager_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=152,
        num_reqs=None,
    )
    assert make_decode_slices(plan, input_batch, eager_desc, 4) == slices

    input_batch.num_tokens_after_padding = 160
    assert make_decode_slices(plan, input_batch, full_desc, 4) is None


def test_registry_json_and_runtime_buckets_are_bounded():
    registry = K3ElasticGraphRegistry.from_json(
        __import__("json").dumps([plan_config()])
    )
    assert registry.max_waves == 2
    assert context_bucket(100_000) == "64k-128k"
    assert kv_pressure_bucket(0.49) == "0-50pct"
    assert kv_pressure_bucket(0.50) == "50-80pct"
    assert prefill_bucket(0) == "none"
    assert prefill_bucket(100_001) == "32k-128k"


def test_v2_runtime_selects_only_complete_exact_envelope():
    runtime = K3ElasticRuntime.__new__(K3ElasticRuntime)
    runtime.registry = K3ElasticGraphRegistry.from_config(
        [plan_config(state_bytes=0)]
    )
    runtime.runtime_identity = "runtime-a"
    runtime.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    scheduler_output = SimpleNamespace(
        is_pure_decode_step=True,
        kv_cache_usage=0.4,
    )
    input_batch = SimpleNamespace(
        num_reqs=38,
        num_tokens=152,
        num_tokens_after_padding=152,
        num_scheduled_tokens=np.full(38, 4, dtype=np.int32),
        is_prefilling_np=np.zeros(38, dtype=np.bool_),
        num_draft_tokens_per_req=np.full(38, 3, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.full((38,), 100_000),
    )
    batch_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=152,
        num_reqs=None,
    )

    selected = runtime.select(scheduler_output, input_batch, batch_desc, 4)
    assert selected is not None
    assert selected[0].name == "x38-m128-m24"

    scheduler_output.kv_cache_usage = 0.8
    assert runtime.select(scheduler_output, input_batch, batch_desc, 4) is None
    scheduler_output.kv_cache_usage = 0.4
    scheduler_output.is_pure_decode_step = False
    assert runtime.select(scheduler_output, input_batch, batch_desc, 4) is None
    scheduler_output.is_pure_decode_step = True
    input_batch.num_tokens_after_padding = 160
    assert runtime.select(scheduler_output, input_batch, batch_desc, 4) is None


def make_real_input_batch() -> InputBatch:
    num_reqs = 38
    num_tokens = 152
    query_start_np = np.arange(num_reqs + 1, dtype=np.int32) * 4
    query_start = torch.from_numpy(query_start_np.copy())
    req_indices = torch.arange(num_reqs, dtype=torch.int64)
    marlin_layout = torch.zeros(num_reqs + 3, dtype=torch.int32)
    marlin_layout[0] = num_reqs
    marlin_layout[1] = num_reqs
    marlin_layout[2:] = query_start
    return InputBatch(
        req_ids=[f"req-{idx}" for idx in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=req_indices,
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        expanded_idx_mapping=req_indices,
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32),
        num_scheduled_tokens=np.full(num_reqs, 4, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=num_reqs * 3,
        num_draft_tokens_per_req=np.full(num_reqs, 3, dtype=np.int32),
        query_start_loc=query_start,
        query_start_loc_np=query_start_np,
        marlin_request_layout_cpu=marlin_layout,
        seq_lens=torch.full((num_reqs,), 100_000, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.full(
            (num_reqs,), 100_000, dtype=torch.int32
        ),
        dcp_local_seq_lens=torch.full((num_reqs,), 33_334, dtype=torch.int32),
        num_computed_tokens_np=np.full(num_reqs, 99_996, dtype=np.int32),
        prefill_len_np=np.full(num_reqs, 10_000, dtype=np.int32),
        num_computed_prefill_tokens_np=np.full(
            num_reqs, 10_000, dtype=np.int32
        ),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        max_seq_len_np=None,
        input_ids=torch.arange(num_tokens, dtype=torch.int32),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        is_padding=torch.zeros(num_tokens, dtype=torch.bool),
        logits_indices=torch.arange(num_reqs, dtype=torch.int64) * 4 + 3,
        cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32),
        cu_num_logits_np=np.arange(num_reqs + 1, dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


def test_v2_prepare_preserves_wave_offsets_and_joins_once():
    input_batch = make_real_input_batch()
    plan = K3ElasticGraphPlan.from_dict(plan_config(state_bytes=0))
    slices = tuple(
        UBatchSlice(req_slice, tok_slice)
        for req_slice, tok_slice in zip(
            plan.request_slices(), plan.token_slices(4), strict=True
        )
    )

    class FakeModelState:
        def __init__(self):
            self.calls = []
            self.joined = None

        def prepare_attn(
            self,
            wave_batch,
            mode,
            wave_tables,
            wave_slots,
            _groups,
            _kv_config,
            metadata_builder_idx,
        ):
            self.calls.append(
                (
                    list(wave_batch.req_ids),
                    mode,
                    tuple(table.shape for table in wave_tables),
                    wave_slots.shape,
                    metadata_builder_idx,
                    wave_batch.query_start_loc_np.copy(),
                    wave_batch.marlin_request_layout_cpu.clone(),
                )
            )
            return {"wave": metadata_builder_idx}

        def begin_joined_mtp_replay_step(self, num_reqs):
            self.joined = num_reqs

    state = FakeModelState()
    runtime = K3ElasticRuntime.__new__(K3ElasticRuntime)
    original_builder = runtime_module.build_slot_mappings_by_layer
    runtime_module.build_slot_mappings_by_layer = (
        lambda slots, _config: {"layer": slots}
    )
    try:
        prepared = runtime.prepare(
            plan,
            slices,
            input_batch,
            (torch.zeros((38, 8), dtype=torch.int32),),
            torch.zeros((2, 152), dtype=torch.int64),
            state,
            [],
            SimpleNamespace(),
        )
    finally:
        runtime_module.build_slot_mappings_by_layer = original_builder

    assert state.joined == 38
    assert [call[4] for call in state.calls] == [0, 1]
    assert [len(call[0]) for call in state.calls] == [32, 6]
    assert [call[3] for call in state.calls] == [
        torch.Size([2, 128]),
        torch.Size([2, 24]),
    ]
    assert state.calls[1][5].tolist() == [0, 4, 8, 12, 16, 20, 24]
    assert state.calls[1][6][:2].tolist() == [6, 6]
    assert [value["wave"] for value in prepared.attn_metadata] == [0, 1]
    assert [batch.num_reqs for batch in prepared.wave_batches] == [32, 6]


def test_v2_primes_every_wave_dcp_capture_state_once():
    class FakePrefillWrapper:
        def __init__(self):
            self.calls = []

        def prime_dcp_local_kv_head_indices(self, indices, device):
            self.calls.append((tuple(indices), device))

    wrappers = [FakePrefillWrapper(), FakePrefillWrapper()]
    prepared = SimpleNamespace(
        slices=(object(), object()),
        attn_metadata=[
            {
                "layer-0": SimpleNamespace(
                    prefill=SimpleNamespace(wrapper=wrappers[wave])
                ),
                "layer-1": SimpleNamespace(
                    prefill=SimpleNamespace(wrapper=wrappers[wave])
                ),
            }
            for wave in range(2)
        ],
    )
    layer = SimpleNamespace(
        dcp_full_kv_attention_heads=True,
        dcp_local_kv_head_indices=(1, 3),
    )
    model = SimpleNamespace(modules=lambda: [layer])
    runtime = K3ElasticRuntime.__new__(K3ElasticRuntime)
    runtime.device = torch.device("cpu")
    runtime._dcp_primed_plans = set()
    plan = SimpleNamespace(name="x38-m128-m24")

    runtime._prime_dcp_capture_state(plan, prepared, model)
    runtime._prime_dcp_capture_state(plan, prepared, model)

    assert [wrapper.calls for wrapper in wrappers] == [
        [((1, 3), torch.device("cpu"))],
        [((1, 3), torch.device("cpu"))],
    ]
