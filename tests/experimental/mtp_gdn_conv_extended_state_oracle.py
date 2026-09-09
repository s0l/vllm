#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prove the real speculative GDN conv-state contract against sequential decode.

The Qwen TP3 path stores convolution state in one SD-layout block with
``width - 1 + K`` entries.  Temporal state uses separate speculative blocks,
but convolution state does not.  After verification, acceptance compacts the
extended convolution window inside its committed block.
"""

from __future__ import annotations

import json

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)

MAX_REQUESTS = 21
DIM = 3840
KERNEL_WIDTH = 4
QUERY_LEN = 3
STATE_LEN = KERNEL_WIDTH - 1 + QUERY_LEN - 1


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def _sequential(
    inputs: torch.Tensor,
    initial_history: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run qlen=1 repeatedly with the normal three-entry logical state."""
    num_requests, query_len, _ = inputs.shape
    raw_state = torch.zeros(
        num_requests + 1,
        KERNEL_WIDTH - 1,
        DIM,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    raw_state[1:] = initial_history
    state = raw_state.transpose(1, 2)
    state_indices = torch.arange(
        1, num_requests + 1, dtype=torch.int32, device=inputs.device
    )
    outputs = []
    histories = []
    for token_index in range(query_len):
        output = causal_conv1d_update(
            inputs[:, token_index].clone(),
            state,
            weight,
            bias,
            "silu",
            conv_state_indices=state_indices,
            validate_data=True,
        )
        outputs.append(output)
        histories.append(raw_state[1:].clone())
    return torch.stack(outputs, dim=1), histories


@torch.inference_mode()
def run_case(num_requests: int) -> dict[str, object]:
    if num_requests not in (1, MAX_REQUESTS):
        raise ValueError("oracle admits only x1 and MAXx")
    torch.manual_seed(20260725 + num_requests)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    initial_history = (
        torch.randn(
            num_requests,
            KERNEL_WIDTH - 1,
            DIM,
            dtype=dtype,
            device=device,
        )
        * 0.05
    )
    inputs = (
        torch.randn(
            num_requests,
            QUERY_LEN,
            DIM,
            dtype=dtype,
            device=device,
        )
        * 0.05
    )
    continuation = (
        torch.randn(
            num_requests,
            QUERY_LEN,
            DIM,
            dtype=dtype,
            device=device,
        )
        * 0.05
    )
    weight = torch.randn(DIM, KERNEL_WIDTH, dtype=dtype, device=device) * 0.05
    bias = torch.randn(DIM, dtype=dtype, device=device) * 0.05

    # This is the real physical SD cache layout.  The Qwen layer transposes it
    # to [block, dim, state_len] before calling causal_conv1d_update.
    raw_extended = torch.randn(
        num_requests + 1,
        STATE_LEN,
        DIM,
        dtype=dtype,
        device=device,
    )
    raw_extended[1:, : KERNEL_WIDTH - 1] = initial_history
    extended = raw_extended.transpose(1, 2)
    state_indices = torch.arange(1, num_requests + 1, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(
        0,
        (num_requests + 1) * QUERY_LEN,
        QUERY_LEN,
        dtype=torch.int32,
        device=device,
    )
    accepted_reset = torch.ones(num_requests, dtype=torch.int32, device=device)

    multi_output = causal_conv1d_update(
        inputs.reshape(num_requests * QUERY_LEN, DIM).clone(),
        extended,
        weight,
        bias,
        "silu",
        conv_state_indices=state_indices,
        num_accepted_tokens=accepted_reset,
        query_start_loc=query_start_loc,
        max_query_len=QUERY_LEN,
        validate_data=False,
    ).reshape(num_requests, QUERY_LEN, DIM)
    sequential_output, history_after_token = _sequential(
        inputs, initial_history, weight, bias
    )

    expected_extended = torch.cat([initial_history[:, 1:], inputs], dim=1)
    forward_state_exact = torch.equal(raw_extended[1:], expected_extended)

    accepted = (
        torch.arange(num_requests, dtype=torch.int64, device=device) + 2
    ) % 3 + 1
    committed = raw_extended[1:].clone()
    for request_index, accepted_count in enumerate(accepted.tolist()):
        token_bias = accepted_count - 1
        if token_bias:
            committed[request_index, : STATE_LEN - token_bias] = committed[
                request_index, token_bias:
            ].clone()
    raw_extended[1:] = committed

    expected_committed = torch.stack(
        [
            history_after_token[accepted_count - 1][request_index]
            for request_index, accepted_count in enumerate(accepted.tolist())
        ]
    )
    # Snapshot before the continuation call mutates the same physical cache.
    committed_history = raw_extended[1:, : KERNEL_WIDTH - 1].clone()

    continuation_output = causal_conv1d_update(
        continuation.reshape(num_requests * QUERY_LEN, DIM).clone(),
        extended,
        weight,
        bias,
        "silu",
        conv_state_indices=state_indices,
        num_accepted_tokens=accepted_reset,
        query_start_loc=query_start_loc,
        max_query_len=QUERY_LEN,
        validate_data=False,
    ).reshape(num_requests, QUERY_LEN, DIM)
    sequential_continuation, _ = _sequential(
        continuation, expected_committed, weight, bias
    )

    result = {
        "requests": num_requests,
        "physical_layout": [num_requests + 1, STATE_LEN, DIM],
        "accepted_counts": sorted(set(accepted.tolist())),
        "verification_output_error": _error(multi_output, sequential_output),
        "verification_output_bit_exact": torch.equal(multi_output, sequential_output),
        "extended_forward_state_bit_exact": forward_state_exact,
        "committed_history_error": _error(committed_history, expected_committed),
        "committed_history_bit_exact": torch.equal(
            committed_history, expected_committed
        ),
        "continuation_output_error": _error(
            continuation_output, sequential_continuation
        ),
        "continuation_output_bit_exact": torch.equal(
            continuation_output, sequential_continuation
        ),
    }
    if not all(
        (
            result["verification_output_bit_exact"],
            result["extended_forward_state_bit_exact"],
            result["committed_history_bit_exact"],
            result["continuation_output_bit_exact"],
        )
    ):
        print(json.dumps({"result": "fail", **result}, indent=2))
        raise SystemExit(1)
    return result


def main() -> None:
    result = {"x1": run_case(1), "max_x": run_case(MAX_REQUESTS)}
    print(json.dumps({"result": "pass", **result}, indent=2))


if __name__ == "__main__":
    main()
