# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode, DeviceConfig, VllmConfig
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_full_graph_replay_publishes_padded_request_context() -> None:
    """x3 -> physical x4 FULL callbacks consume current, not capture state."""
    runner = object.__new__(GPUModelRunner)
    runner.vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    runner.lora_config = None
    runner.input_buffers = SimpleNamespace(marlin_request_layout_cpu=None)
    runner.cudagraph_manager = SimpleNamespace(
        dynamic_graph_owner="target",
        uses_tp3_owner_prequant_decode=lambda _desc: False,
    )
    metadata = {"runtime": object()}
    input_batch = SimpleNamespace(
        num_tokens_after_padding=8,
        num_tokens=6,
        is_padding=torch.tensor([False, False, False, False, False, False, True, True]),
    )
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=3,
        uniform_token_count=2,
        physical_num_reqs=4,
        runtime_generation="test-generation",
        semantic_decode=True,
    )
    scheduler_output = SimpleNamespace(is_pure_decode_step=True)

    assert not is_forward_context_available()
    with runner._full_graph_replay_context(
        attn_metadata=metadata,
        input_batch=input_batch,
        batch_desc=desc,
        scheduler_output=scheduler_output,
        dp_sync=None,
        ubatch_slices=None,
        slot_mappings_by_layer=None,
        skip_compiled=False,
    ):
        context = get_forward_context()
        assert context.attn_metadata is metadata
        assert context.num_tokens_unpadded == 6
        assert context.batch_descriptor.num_tokens == 8
        assert context.batch_descriptor.physical_num_reqs == 4
        assert context.batch_descriptor.runtime_generation == "test-generation"
    assert not is_forward_context_available()
