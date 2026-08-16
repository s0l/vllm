# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.structured_outputs import (
    _apply_grammar_bitmask_kernel,
    _apply_local_grammar_bitmask_kernel,
)

if not current_platform.is_cuda():
    pytest.skip("grammar bitmask kernels require CUDA", allow_module_level=True)


def _inputs(global_vocab_size: int = 64):
    logits = torch.zeros((3, 32), dtype=torch.float32, device="cuda")
    # Request 0 owns logit rows [0, 2), request 1 owns [2, 3).
    cu_num_logits = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    # Select request 0 position 1 and request 1 position 0 at stride 4.
    mapping = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
    bitmask = torch.full(
        (2, (global_vocab_size + 31) // 32),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    return logits, cu_num_logits, mapping, bitmask


def test_global_grammar_kernel_uses_request_position_mapping() -> None:
    logits, cu_num_logits, mapping, bitmask = _inputs(32)
    bitmask[:, 0] &= ~(1 << 5)
    _apply_grammar_bitmask_kernel[(2, 1)](
        logits,
        logits.stride(0),
        mapping,
        cu_num_logits,
        bitmask,
        bitmask.stride(0),
        32,
        MASK_STRIDE=4,
        BLOCK_SIZE=32,
    )
    assert torch.isneginf(logits[1, 5])
    assert torch.isneginf(logits[2, 5])
    assert torch.count_nonzero(torch.isneginf(logits)) == 2


def test_local_grammar_kernel_maps_global_vocab_and_request_position() -> None:
    logits, cu_num_logits, mapping, bitmask = _inputs(64)
    bitmask[:, 1] &= ~(1 << 5)  # Global token 37, local token 5.
    _apply_local_grammar_bitmask_kernel[(2, 1)](
        logits,
        logits.stride(0),
        mapping,
        cu_num_logits,
        bitmask,
        bitmask.stride(0),
        32,
        32,
        MASK_STRIDE=4,
        BLOCK_SIZE=32,
    )
    assert torch.isneginf(logits[1, 5])
    assert torch.isneginf(logits[2, 5])
    assert torch.count_nonzero(torch.isneginf(logits)) == 2
