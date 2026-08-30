"""Prove separate-pool speculative convolution matches sequential decode."""

from __future__ import annotations

import argparse
import json

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)


def tensor_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - reference.float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-reqs", type=int, choices=(1, 21), default=1)
    args = parser.parse_args()
    torch.manual_seed(212)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_len = 3
    num_reqs = args.num_reqs
    dim = 3840
    width = 4

    inputs = (
        torch.randn(num_reqs, query_len, dim, device=device, dtype=dtype)
        * 0.05
    )
    weight = torch.randn(dim, width, device=device, dtype=dtype) * 0.05
    bias = torch.randn(dim, device=device, dtype=dtype) * 0.05
    initial = (
        torch.randn(
            num_reqs, dim, width - 1, device=device, dtype=dtype
        )
        * 0.05
    )

    # Separate-pool topology: one exact-width block for the committed state,
    # followed by one scratch block per additional speculative position.
    block_ids = torch.arange(
        1,
        num_reqs * query_len + 1,
        device=device,
        dtype=torch.int32,
    ).view(num_reqs, query_len)
    multi_state = torch.zeros(
        num_reqs * query_len + 1,
        dim,
        width - 1,
        device=device,
        dtype=dtype,
    )
    multi_state[block_ids[:, 0].long()] = initial
    dirty_scratch_state = multi_state.clone()
    dirty_scratch_state[block_ids[:, 1:].long().flatten()] = (
        torch.randn_like(dirty_scratch_state[block_ids[:, 1:].long().flatten()])
        * 7
    )
    multi_out = causal_conv1d_update(
        inputs.reshape(num_reqs * query_len, dim).clone(),
        multi_state,
        weight,
        bias,
        "silu",
        conv_state_indices=block_ids,
        num_accepted_tokens=torch.ones(
            num_reqs, device=device, dtype=torch.int32
        ),
        query_start_loc=torch.arange(
            0,
            (num_reqs + 1) * query_len,
            query_len,
            device=device,
            dtype=torch.int32,
        ),
        max_query_len=query_len,
        separate_pool=True,
        validate_data=False,
    )
    dirty_scratch_out = causal_conv1d_update(
        inputs.reshape(num_reqs * query_len, dim).clone(),
        dirty_scratch_state,
        weight,
        bias,
        "silu",
        conv_state_indices=block_ids,
        num_accepted_tokens=torch.ones(
            num_reqs, device=device, dtype=torch.int32
        ),
        query_start_loc=torch.arange(
            0,
            (num_reqs + 1) * query_len,
            query_len,
            device=device,
            dtype=torch.int32,
        ),
        max_query_len=query_len,
        separate_pool=True,
        validate_data=False,
    )

    sequential_state = torch.zeros_like(multi_state)
    committed = block_ids[:, 0]
    sequential_state[committed.long()] = initial
    sequential_outputs = []
    sequential_states = []
    for token_idx in range(query_len):
        output = causal_conv1d_update(
            inputs[:, token_idx].clone(),
            sequential_state,
            weight,
            bias,
            "silu",
            conv_state_indices=committed,
            validate_data=True,
        )
        sequential_outputs.append(output)
        sequential_states.append(sequential_state[committed.long()].clone())
    sequential_out = torch.stack(sequential_outputs, dim=1).reshape(
        num_reqs * query_len, dim
    )
    sequential_state_history = torch.stack(
        sequential_states, dim=1
    )
    multi_state_history = multi_state[
        block_ids.long().flatten()
    ].reshape(num_reqs, query_len, dim, width - 1)
    dirty_scratch_history = dirty_scratch_state[
        block_ids.long().flatten()
    ].reshape(num_reqs, query_len, dim, width - 1)

    result = {
        "shape": {
            "query_len": query_len,
            "num_reqs": num_reqs,
            "dim": dim,
            "width": width,
            "state_dtype": str(dtype),
        },
        "output_error": tensor_error(multi_out, sequential_out),
        "state_history_error": tensor_error(
            multi_state_history, sequential_state_history
        ),
        "scratch_zero_blocks": sum(
            int(torch.count_nonzero(multi_state[block_id]).item()) == 0
            for block_id in block_ids[:, 1:].flatten().tolist()
        ),
        "bit_exact_output": torch.equal(multi_out, sequential_out),
        "bit_exact_state_history": torch.equal(
            multi_state_history, sequential_state_history
        ),
        "dirty_scratch_output_exact": torch.equal(
            dirty_scratch_out, sequential_out
        ),
        "dirty_scratch_state_history_exact": torch.equal(
            dirty_scratch_history, sequential_state_history
        ),
    }
    print(json.dumps(result, indent=2))
    if (
        not result["bit_exact_output"]
        or not result["bit_exact_state_history"]
        or not result["dirty_scratch_output_exact"]
        or not result["dirty_scratch_state_history_exact"]
        or result["scratch_zero_blocks"] != 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
