# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import (
    cudagraph_utils as spec_cg_module,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.test_draft_model


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


class _MultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _TextOnlyDraftModel(torch.nn.Module):
    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


@pytest.mark.parametrize(
    ("use_fused", "expected_decode_fn"),
    [(False, "_generate_draft"), (True, "_generate_fused_drafts")],
)
def test_dynamic_capture_uses_the_selected_draft_owner(use_fused, expected_decode_fn):
    speculator = object.__new__(_TestSpeculator)
    prefill_manager = MagicMock()
    decode_manager = MagicMock()
    speculator.prefill_cudagraph_manager = prefill_manager
    speculator.decode_cudagraph_manager = decode_manager
    speculator.model = MagicMock()
    speculator.model_state = MagicMock()
    speculator.input_buffers = MagicMock()
    speculator.target_input_buffers = MagicMock()
    speculator.block_tables = MagicMock()
    speculator.attn_groups = []
    speculator.target_attn_groups = [MagicMock()]
    speculator.kv_cache_config = MagicMock()
    speculator.last_token_indices = MagicMock()
    speculator.idx_mapping = MagicMock()
    speculator.max_num_reqs = 64
    speculator.use_fused_multi_step_decode = use_fused
    speculator.on_prefill_begin = MagicMock()
    speculator.on_prefill_end = MagicMock()
    speculator.on_multi_step_decode_begin = MagicMock()
    speculator.on_multi_step_decode_end = MagicMock()

    descriptor = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)
    complete_hook = MagicMock()

    def capture_next_dynamic(*args, capture_override, **kwargs):
        capture_override({CUDAGraphMode.FULL: [descriptor]}, complete_hook)
        return True

    prefill_manager.capture_next_dynamic.side_effect = capture_next_dynamic
    decode_manager.capture_next_dynamic.side_effect = capture_next_dynamic
    assert speculator.dynamic_cudagraph_managers() == (
        prefill_manager,
        decode_manager,
    )
    assert speculator.capture_next_dynamic(prefill_manager)
    speculator.last_token_indices.zero_.assert_called_once_with()
    speculator.idx_mapping.zero_.assert_called_once_with()
    speculator.on_prefill_begin.assert_called_once_with(64)
    speculator.on_prefill_end.assert_called_once_with(64)
    prefill_manager.capture.assert_called_once()
    capture_args = prefill_manager.capture.call_args
    assert capture_args.args[0] == speculator._prefill
    assert capture_args.args[2] is speculator.target_input_buffers
    assert capture_args.args[4] is speculator.target_attn_groups
    assert capture_args.kwargs["capture_descs"] == {CUDAGraphMode.FULL: [descriptor]}
    assert capture_args.kwargs["capture_complete_hook"] is complete_hook

    assert speculator.capture_next_dynamic(decode_manager)
    assert speculator.last_token_indices.zero_.call_count == 2
    assert speculator.idx_mapping.zero_.call_count == 2
    speculator.on_multi_step_decode_begin.assert_called_once_with(64)
    speculator.on_multi_step_decode_end.assert_called_once_with(64)
    decode_args = decode_manager.capture.call_args
    assert decode_args.args[0] == getattr(speculator, expected_decode_fn)
    assert decode_args.args[2] is speculator.input_buffers
    assert decode_args.args[4] is speculator.attn_groups


def test_dynamic_decode_capture_restores_lifecycle_after_failure():
    speculator = object.__new__(_TestSpeculator)
    manager = MagicMock()
    speculator.prefill_cudagraph_manager = MagicMock()
    speculator.decode_cudagraph_manager = manager
    speculator.model = MagicMock()
    speculator.model_state = MagicMock()
    speculator.input_buffers = MagicMock()
    speculator.block_tables = MagicMock()
    speculator.attn_groups = []
    speculator.kv_cache_config = MagicMock()
    speculator.last_token_indices = MagicMock()
    speculator.idx_mapping = MagicMock()
    speculator.max_num_reqs = 8
    speculator.use_fused_multi_step_decode = True
    speculator.on_multi_step_decode_begin = MagicMock()
    speculator.on_multi_step_decode_end = MagicMock()

    def capture_next_dynamic(*args, capture_override, **kwargs):
        capture_override({CUDAGraphMode.FULL: []}, MagicMock())

    manager.capture_next_dynamic.side_effect = capture_next_dynamic
    manager.capture.side_effect = RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        speculator.capture_next_dynamic(manager)

    speculator.on_multi_step_decode_begin.assert_called_once_with(8)
    speculator.on_multi_step_decode_end.assert_called_once_with(8)


def test_fused_dynamic_capture_forwards_physical_identity_to_every_draft():
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 3
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    speculator._generate_draft = MagicMock()

    speculator._generate_fused_drafts(
        num_reqs=2,
        num_tokens_padded=2,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        physical_num_reqs=2,
        runtime_generation="dynamic-generation",
    )

    assert speculator._generate_draft.call_count == 2
    for call in speculator._generate_draft.call_args_list:
        assert call.kwargs == {
            "physical_num_reqs": 2,
            "runtime_generation": "dynamic-generation",
        }


def test_dynamic_speculator_capture_preserves_physical_replay_identity(monkeypatch):
    manager = object.__new__(SpeculatorCudaGraphManager)
    manager.dp_size = 1
    manager._capture_num_reqs = MagicMock(return_value=2)
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=4,
        num_reqs=None,
        physical_num_reqs=2,
        runtime_generation="post-kv-generation",
    )
    monkeypatch.setattr(
        spec_cg_module,
        "prepare_inputs_to_capture",
        lambda *args, **kwargs: ({}, {}),
    )

    def capture_base(self, create_forward_fn, *args, **kwargs):
        create_forward_fn(descriptor, warmup=False)(CUDAGraphMode.PIECEWISE)

    monkeypatch.setattr(CudaGraphManager, "capture", capture_base)
    forward = MagicMock(return_value=None)

    manager.capture(
        forward,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        [],
        MagicMock(),
        capture_descs={CUDAGraphMode.PIECEWISE: [descriptor]},
    )

    assert forward.call_args.kwargs == {
        "physical_num_reqs": 2,
        "runtime_generation": "post-kv-generation",
    }


def test_mtp_prefill_dispatch_uses_live_rows_but_prefill_keeps_padded_descriptor(
    monkeypatch,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 1
    speculator.max_model_len = 262144
    speculator.method = "mtp"
    speculator.hidden_states = torch.empty((64, 8))
    speculator._copy_request_inputs = MagicMock()
    speculator._ag2_mtp_layer_capture = None
    speculator._ag2_draft_capture = None
    speculator._maybe_save_ag2_mtp_boundary = MagicMock()
    speculator._prepare_eplb_forward = MagicMock()
    speculator._prefill = MagicMock()
    speculator._ag2_snapshot_proposal_step = MagicMock()
    speculator._maybe_append_ag2_mtp_drafts = MagicMock()
    speculator.draft_tokens = torch.zeros((3, 1), dtype=torch.long)
    speculator.last_token_indices = torch.zeros(3, dtype=torch.long)
    speculator.current_draft_step = torch.zeros(1, dtype=torch.long)
    speculator.input_buffers = MagicMock()
    speculator.max_num_reqs = 8
    speculator.prefill_cudagraph_manager = MagicMock()
    speculator.dp_size = 1
    speculator.dp_rank = 0

    input_batch = SimpleNamespace(
        num_tokens_after_padding=64,
        num_tokens=46,
        num_reqs=3,
        num_scheduled_tokens=torch.tensor([16, 15, 15]),
        seq_lens_cpu_upper_bound=torch.tensor([100, 200, 300]),
        idx_mapping=torch.arange(3),
        req_ids=["a", "b", "c"],
        has_prefill=True,
        is_padding=torch.zeros(64, dtype=torch.bool),
    )
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=64,
        num_reqs=None,
    )
    dispatch = MagicMock(return_value=(descriptor, None))
    monkeypatch.setattr(spec_module, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(spec_module, "prepare_prefill_inputs", MagicMock())
    monkeypatch.setattr(
        spec_module, "get_uniform_decode_token_count", lambda *args: None
    )

    result = speculator.propose(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={},
        # The target carrier itself is padded to the stable B64 buffer shape.
        last_hidden_states=torch.ones((64, 8)),
        aux_hidden_states=None,
        num_sampled=torch.zeros(3, dtype=torch.long),
        num_rejected=torch.zeros(3, dtype=torch.long),
        last_sampled=torch.zeros(8, dtype=torch.long),
        next_prefill_tokens=torch.zeros(8, dtype=torch.long),
        temperature=torch.ones(8),
        seeds=torch.zeros(8, dtype=torch.long),
    )

    assert result.shape == (3, 1)
    assert dispatch.call_args.args[2] == 46
    assert speculator._prefill.call_args.args[1] == 64


def _mock_base_model_load(monkeypatch):
    monkeypatch.setattr(
        base_spec_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        DraftModelSpeculator,
        "_validate_local_argmax_reduction",
        lambda self: None,
    )


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
        is_padding=torch.zeros(4, dtype=torch.bool),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    speculator._ag2_draft_capture = None
    speculator._ag2_mtp_layer_capture = None
    return speculator


@pytest.mark.parametrize(("hc_mult", "expected"), [(None, 64), (4, 256)])
def test_speculator_uses_draft_model_hidden_size(monkeypatch, hc_mult, expected):
    # Qwen4Exp targets expose multi-stream HC residuals to the drafter.
    monkeypatch.setattr(base_spec_module, "_target_feeds_hc_residual", lambda _: True)
    monkeypatch.setattr(spec_module, "get_tensor_model_parallel_rank", lambda: 0)
    hf_config = SimpleNamespace()
    if hc_mult is not None:
        hf_config.hc_mult = hc_mult
    draft_model_config = SimpleNamespace(
        hf_config=hf_config,
        get_hidden_size=lambda: 64,
        get_vocab_size=lambda: 32,
    )
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=3,
        draft_model_config=draft_model_config,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=32,
            dtype=torch.float32,
            use_fp64_gumbel=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
        ),
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.hidden_size == expected


def test_mm_support_configured_after_model_load(monkeypatch):
    target_model_config = object()
    draft_model_config = object()
    vllm_config = SimpleNamespace(model_config=target_model_config)
    draft_model = _MultimodalDraftModel()

    def init_base(speculator, vllm_config, device):
        speculator.vllm_config = vllm_config
        speculator.device = device
        speculator.max_num_tokens = 4
        speculator.max_num_reqs = 2
        speculator.hidden_size = 3
        speculator.dtype = torch.float32
        speculator.draft_model_config = draft_model_config
        speculator.supports_mm_inputs = False
        speculator.num_speculative_steps = 0

    checked_configs = []

    def supports_multimodal_inputs(model_config):
        checked_configs.append(model_config)
        return True

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    monkeypatch.setattr(spec_module, "get_tensor_model_parallel_rank", lambda: 0)
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        supports_multimodal_inputs,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert checked_configs == []
    assert not speculator.supports_mm_inputs
    assert speculator.inputs_embeds is None

    speculator.test_draft_model = draft_model
    speculator.load_model(torch.nn.Module())

    assert checked_configs == [target_model_config]
    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)


def test_load_model_keeps_mm_support_for_capable_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    speculator._ag2_draft_capture = None
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator._ag2_draft_capture = None
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        base_spec_module.logger,
        "warning_once",
        lambda message, *args: warning_messages.append(message % args),
    )

    speculator.load_model(torch.nn.Module())

    assert not speculator.supports_mm_inputs
    assert warning_messages == [
        (
            "Draft model _TextOnlyDraftModel does not support external multimodal "
            "embeddings. Embeddings from the target model will not be passed to the "
            "drafter; using text-only draft inputs instead."
        )
    ]


def test_multi_module_mm_support_configured_after_model_load(monkeypatch):
    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        MultiModuleMTPSpeculator,
        "load_draft_model",
        lambda self, target_model, target_attn_layer_names: draft_model,
    )
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden, mtp_trace = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden
    assert mtp_trace is None


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden, mtp_trace = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden
    assert mtp_trace is None


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 0, 3),
        ("_fused_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_fused_multi_step_decode", CUDAGraphMode.FULL, 0, 1),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    method_name,
    cg_mode,
    expected_eager_calls,
    expected_graph_replays,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    speculator._ag2_draft_capture = None
    speculator._ag2_mtp_layer_capture = None
    generate_draft = Mock()
    speculator._generate_draft = generate_draft
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=2,
        num_reqs=2,
    )

    getattr(speculator, method_name)(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        seq_lens_cpu_upper_bound=None,
        num_tokens_across_dp=None,
    )

    assert generate_draft.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


def test_update_draft_decode_metadata_updates_fa3_scheduler_metadata(
    monkeypatch,
):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(8, dtype=torch.int32)
    builder.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    builder.kv_cache_dtype = torch.bfloat16
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 128
    builder.block_size = 16
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.aot_sliding_window = None

    expected = torch.tensor([7, 8, 9], dtype=torch.int32)

    def fake_get_scheduler_metadata(**kwargs):
        return expected

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
        prefix_scheduler_metadata=None,
        max_num_splits=4,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(metadata.scheduler_metadata, expected)
    assert torch.equal(builder.scheduler_metadata[:3], expected)


def test_update_draft_decode_metadata_skips_without_scheduler_metadata(monkeypatch):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

    called = False

    def fake_get_scheduler_metadata(**kwargs):
        nonlocal called
        called = True
        return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert not called
    assert metadata.scheduler_metadata is None
