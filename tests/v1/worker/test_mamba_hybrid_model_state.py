# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def test_add_request_resets_reused_separate_pool_slot() -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full((4,), 4, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.full((4,), 7, dtype=torch.int32)
    state._align_mode = True
    state._separate_mamba_pool = True

    with patch.object(DefaultModelState, "add_request") as parent_add:
        request = SimpleNamespace(num_computed_tokens=128)
        state.add_request(2, request)

    parent_add.assert_called_once_with(2, request)
    torch.testing.assert_close(
        state.num_accepted_tokens_gpu,
        torch.tensor([4, 4, 1, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        state._mamba_state_idx_gpu,
        torch.tensor([7, 7, 0, 7], dtype=torch.int32),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)
