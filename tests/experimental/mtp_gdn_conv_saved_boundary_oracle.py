#!/usr/bin/env python3
"""Replay the real layer-0 convolution boundary at qlen=1 and qlen=3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)

WEIGHT_NAME = "model.language_model.layers.0.linear_attn.conv1d.weight"
PARTITIONS = ((0, 6), (6, 5), (11, 5))
HEAD_DIM = 128
VALUE_HEADS_PER_KEY_HEAD = 3
GLOBAL_KEY_DIM = 2048
PADDED_KEY_DIM = 768
PADDED_VALUE_DIM = 2304
PADDED_CONV_DIM = PADDED_KEY_DIM * 2 + PADDED_VALUE_DIM


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "differing": int(torch.count_nonzero(delta).item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def _load_global_weight(index_path: Path) -> torch.Tensor:
    index = json.loads(index_path.read_text())
    shard_path = index_path.parent / index["weight_map"][WEIGHT_NAME]
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(WEIGHT_NAME).squeeze(1)


def _local_weight(
    global_weight: torch.Tensor,
    key_start: int,
    key_count: int,
) -> torch.Tensor:
    value_start = key_start * VALUE_HEADS_PER_KEY_HEAD
    value_count = key_count * VALUE_HEADS_PER_KEY_HEAD
    local = torch.zeros(
        PADDED_CONV_DIM,
        global_weight.shape[-1],
        dtype=global_weight.dtype,
    )
    local_key_dim = key_count * HEAD_DIM
    local_value_dim = value_count * HEAD_DIM
    local[:local_key_dim] = global_weight[
        key_start * HEAD_DIM : (key_start + key_count) * HEAD_DIM
    ]
    local[PADDED_KEY_DIM : PADDED_KEY_DIM + local_key_dim] = global_weight[
        GLOBAL_KEY_DIM + key_start * HEAD_DIM : GLOBAL_KEY_DIM
        + (key_start + key_count) * HEAD_DIM
    ]
    value_offset = GLOBAL_KEY_DIM * 2
    local[PADDED_KEY_DIM * 2 : PADDED_KEY_DIM * 2 + local_value_dim] = global_weight[
        value_offset + value_start * HEAD_DIM : value_offset
        + (value_start + value_count) * HEAD_DIM
    ]
    return local


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-prefix", type=Path, required=True)
    parser.add_argument("--model-index", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(212)
    device = torch.device("cuda:0")
    global_weight = _load_global_weight(args.model_index)
    results = {}

    for rank, (key_start, key_count) in enumerate(PARTITIONS):
        trace = torch.load(
            f"{args.trace_prefix}.q3.rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        stages = trace["stages"]
        first_input = stages["gdn.qkvz"][0, :PADDED_CONV_DIM].clone()
        physical_state = stages["gdn.conv_state"][0].clone()

        # Match the live qlen=3 contiguous index_select result. Future rows
        # cannot affect row zero and are intentionally deterministic fixtures.
        multi_input = torch.randn(3, PADDED_CONV_DIM, dtype=torch.bfloat16)
        multi_input[0].copy_(first_input)

        # Match the live qlen=1 split view: feature-contiguous but with the
        # original qkvz row stride, rather than silently normalizing layout.
        single_backing = torch.zeros(1, PADDED_CONV_DIM * 2, dtype=torch.bfloat16)
        single_backing[0, :PADDED_CONV_DIM].copy_(first_input)
        single_input = single_backing[:, :PADDED_CONV_DIM]

        weight = _local_weight(global_weight, key_start, key_count).to(device)
        multi_state = torch.zeros(
            2,
            PADDED_CONV_DIM,
            physical_state.shape[-1],
            dtype=torch.bfloat16,
            device=device,
        )
        multi_state[1].copy_(physical_state.to(device))
        single_state = multi_state.clone()
        state_index = torch.ones(1, dtype=torch.int32, device=device)

        multi_output = causal_conv1d_update(
            multi_input.to(device),
            multi_state,
            weight,
            None,
            "silu",
            conv_state_indices=state_index,
            num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
            max_query_len=3,
            validate_data=False,
        )
        single_output = causal_conv1d_update(
            single_input.to(device),
            single_state,
            weight,
            None,
            "silu",
            conv_state_indices=state_index,
            validate_data=True,
        )

        comparison = _error(multi_output[0], single_output[0])
        comparison["bit_exact"] = torch.equal(multi_output[0], single_output[0])
        results[str(rank)] = comparison

    passed = all(result["bit_exact"] for result in results.values())
    print(json.dumps({"passed": passed, "ranks": results}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
