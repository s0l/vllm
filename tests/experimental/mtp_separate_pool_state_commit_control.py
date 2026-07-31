#!/usr/bin/env python3
"""Bit-exact x1/MAX control for speculative separate-pool GDN commit."""

from __future__ import annotations

import json

import torch

from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

MAX_REQUESTS = 21
CONV_WIDTH = 5
CONV_DIM = 3840
TEMPORAL_SHAPE = (18, 128, 128)


def _context(
    conv: torch.Tensor,
    temporal: torch.Tensor,
    block_table: torch.Tensor,
) -> MambaSpecDecodeGPUContext:
    device = conv.device
    return MambaSpecDecodeGPUContext(
        state_base_addrs=torch.tensor(
            [conv.data_ptr(), temporal.data_ptr()],
            dtype=torch.int64,
            device=device,
        ),
        state_block_strides=torch.tensor(
            [
                conv.stride(0) * conv.element_size(),
                temporal.stride(0) * temporal.element_size(),
            ],
            dtype=torch.int64,
            device=device,
        ),
        state_elem_sizes=torch.tensor(
            [conv.element_size(), temporal.element_size()],
            dtype=torch.int32,
            device=device,
        ),
        state_inner_sizes=torch.tensor(
            [conv.stride(1), temporal[0].numel()],
            dtype=torch.int64,
            device=device,
        ),
        state_conv_widths=torch.tensor(
            [CONV_WIDTH, 0],
            dtype=torch.int32,
            device=device,
        ),
        state_group_indices=torch.zeros(2, dtype=torch.int32, device=device),
        state_dim_row_count=torch.zeros(2, dtype=torch.int32, device=device),
        state_dim_row_stride=torch.zeros(2, dtype=torch.int64, device=device),
        block_size=2352,
        num_layers=1,
        num_state_types=2,
        mamba_group_ids=[0],
        num_groups=1,
        num_accepted_tokens_out=torch.zeros(
            MAX_REQUESTS,
            dtype=torch.int32,
            device=device,
        ),
        block_table_ptrs=torch.tensor(
            [block_table.data_ptr()],
            dtype=torch.int64,
            device=device,
        ),
        block_table_stride_req=block_table.stride(0),
        is_initialized=True,
    )


@torch.inference_mode()
def run_case(num_requests: int) -> dict[str, int | bool]:
    if num_requests not in (1, MAX_REQUESTS):
        raise ValueError("control admits only x1 and MAXx")
    torch.manual_seed(20260725 + num_requests)
    device = torch.device("cuda:0")
    num_blocks = 1 + num_requests * 3
    block_table = (
        torch.arange(1, num_blocks, dtype=torch.int32, device=device)
        .view(num_requests, 3)
        .contiguous()
    )
    conv = torch.randn(
        num_blocks,
        CONV_WIDTH,
        CONV_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    temporal = torch.randn(
        num_blocks,
        *TEMPORAL_SHAPE,
        dtype=torch.float32,
        device=device,
    )
    expected_conv = conv.clone()
    expected_temporal = temporal.clone()

    # Batch rows and persistent request slots deliberately differ at MAXx.
    idx_mapping = torch.arange(
        num_requests - 1,
        -1,
        -1,
        dtype=torch.int32,
        device=device,
    )
    # Slot zero must exercise the longest accepted K=2 path in the x1
    # control. MAXx then cycles through accepted counts 3, 1 and 2 while the
    # batch-to-slot mapping is reversed.
    accepted_by_slot = (
        torch.arange(MAX_REQUESTS, dtype=torch.int32, device=device) + 2
    ) % 3 + 1
    accepted = accepted_by_slot.clone()
    original_conv = conv.clone()
    original_temporal = temporal.clone()
    for batch_index in range(num_requests):
        slot = int(idx_mapping[batch_index].item())
        accept_bias = int(accepted_by_slot[slot].item()) - 1
        if accept_bias == 0:
            continue
        base_block = int(block_table[batch_index, 0].item())
        accepted_block = int(block_table[batch_index, accept_bias].item())
        expected_conv[base_block, : CONV_WIDTH - accept_bias] = original_conv[
            base_block, accept_bias:
        ]
        expected_temporal[base_block] = original_temporal[accepted_block]

    context = _context(conv, temporal, block_table)
    context.run_fused_postprocess_separate(
        num_requests,
        accepted,
        idx_mapping,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(temporal, expected_temporal, rtol=0, atol=0)
    torch.testing.assert_close(
        accepted[:num_requests],
        torch.ones(num_requests, dtype=torch.int32, device=device),
        rtol=0,
        atol=0,
    )
    return {
        "requests": num_requests,
        "conv_bit_exact": bool(torch.equal(conv, expected_conv)),
        "temporal_bit_exact": bool(torch.equal(temporal, expected_temporal)),
        "accepted_reset_exact": bool(
            torch.equal(
                accepted[:num_requests],
                torch.ones(num_requests, dtype=torch.int32, device=device),
            )
        ),
    }


@torch.inference_mode()
def run_replay_conv_case(num_requests: int) -> dict[str, int | bool]:
    """Replay topology: one physical block, shift conv only, keep SSM intact."""
    torch.manual_seed(20260730 + num_requests)
    device = torch.device("cuda:0")
    num_blocks = 1 + num_requests
    block_table = torch.arange(
        1, num_blocks, dtype=torch.int32, device=device
    ).view(num_requests, 1)
    conv = torch.randn(
        num_blocks,
        CONV_WIDTH,
        CONV_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    temporal = torch.randn(
        num_blocks,
        *TEMPORAL_SHAPE,
        dtype=torch.float32,
        device=device,
    )
    original_conv = conv.clone()
    original_temporal = temporal.clone()
    expected_conv = conv.clone()
    idx_mapping = torch.arange(
        num_requests - 1, -1, -1, dtype=torch.int32, device=device
    )
    accepted = (
        torch.arange(MAX_REQUESTS, dtype=torch.int32, device=device) + 2
    ) % 3 + 1
    for batch_index in range(num_requests):
        slot = int(idx_mapping[batch_index].item())
        accept_bias = int(accepted[slot].item()) - 1
        if accept_bias == 0:
            continue
        block_id = int(block_table[batch_index, 0].item())
        expected_conv[block_id, : CONV_WIDTH - accept_bias] = original_conv[
            block_id, accept_bias:
        ]

    context = _context(conv, temporal, block_table)
    context.run_fused_postprocess_separate(
        num_requests,
        accepted,
        idx_mapping,
        conv_only=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(temporal, original_temporal, rtol=0, atol=0)
    torch.testing.assert_close(
        accepted[:num_requests],
        torch.ones(num_requests, dtype=torch.int32, device=device),
        rtol=0,
        atol=0,
    )
    return {
        "requests": num_requests,
        "conv_bit_exact": bool(torch.equal(conv, expected_conv)),
        "temporal_untouched": bool(torch.equal(temporal, original_temporal)),
        "accepted_reset_exact": bool(
            torch.equal(
                accepted[:num_requests],
                torch.ones(num_requests, dtype=torch.int32, device=device),
            )
        ),
    }


def main() -> None:
    result = {
        "legacy_x1": run_case(1),
        "legacy_max_x": run_case(MAX_REQUESTS),
        "replay_x1": run_replay_conv_case(1),
        "replay_max_x": run_replay_conv_case(MAX_REQUESTS),
    }
    print(json.dumps({"result": "pass", **result}))


if __name__ == "__main__":
    main()
