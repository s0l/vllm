# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_dynamic_capture_uses_the_selected_draft_owner():
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
    prefill_manager.capture.assert_called_once()
    capture_args = prefill_manager.capture.call_args
    assert capture_args.args[0] == speculator._prefill
    assert capture_args.args[2] is speculator.target_input_buffers
    assert capture_args.args[4] is speculator.target_attn_groups
    assert capture_args.kwargs["capture_descs"] == {
        CUDAGraphMode.FULL: [descriptor]
    }
    assert capture_args.kwargs["capture_complete_hook"] is complete_hook

    assert speculator.capture_next_dynamic(decode_manager)
    assert speculator.last_token_indices.zero_.call_count == 2
    decode_args = decode_manager.capture.call_args
    assert decode_args.args[0] == speculator._generate_draft
    assert decode_args.args[2] is speculator.input_buffers
    assert decode_args.args[4] is speculator.attn_groups


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
    )
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=64,
        num_reqs=None,
    )
    dispatch = MagicMock(return_value=(descriptor, None))
    monkeypatch.setattr(spec_module, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(spec_module, "prepare_prefill_inputs", MagicMock())
    monkeypatch.setattr(spec_module, "get_uniform_token_count", lambda *args: None)

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
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    speculator._ag2_mtp_layer_capture = None
    return speculator


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
    monkeypatch.setattr(
        spec_module, "get_tensor_model_parallel_rank", lambda: 0
    )
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
        "Draft model _TextOnlyDraftModel does not support external multimodal "
        "embeddings. Embeddings from the target model will not be passed to the "
        "drafter; using text-only draft inputs instead."
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
