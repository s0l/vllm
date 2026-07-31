# SPDX-License-Identifier: Apache-2.0
"""Replay saved rank-local rows through TP3 CE and NCCL.

Run inside the exact target container with one process per GPU.  This is a
causal control for the invariant that equal rows on every rank must remain
equal after a TP all-reduce, independently of their position in the batch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.tp3_ce_all_reduce import (
    _dequant_sum_i8_block,
    _quantize_i8_block,
    tp3_ce_all_reduce,
)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _device_ce_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Graph-capturable compressed reduction using NCCL only as transport."""
    rows, cols = tensor.shape
    quant_block = 8192
    num_blocks = (cols + quant_block - 1) // quant_block
    q_local = torch.empty((rows, cols), dtype=torch.uint8, device=tensor.device)
    scale_local = torch.empty(
        (rows, num_blocks), dtype=torch.float32, device=tensor.device
    )
    _quantize_i8_block[(rows, num_blocks)](
        tensor, q_local, scale_local, cols, quant_block
    )
    q_gathered = torch.empty(
        (dist.get_world_size(), rows, cols),
        dtype=torch.uint8,
        device=tensor.device,
    )
    scale_gathered = torch.empty(
        (dist.get_world_size(), rows, num_blocks),
        dtype=torch.float32,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(q_gathered, q_local)
    dist.all_gather_into_tensor(scale_gathered, scale_local)
    output = torch.empty_like(tensor)
    _dequant_sum_i8_block[(rows, num_blocks)](
        q_gathered[0],
        q_gathered[1],
        q_gathered[2],
        scale_gathered[0],
        scale_gathered[1],
        scale_gathered[2],
        output,
        cols,
        quant_block,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=75)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    saved = torch.load(
        args.artifact_dir / f"trace.q75.occ0.rank{rank}.pt",
        map_location="cpu",
        weights_only=False,
    )
    row = saved["layers"]["gdn_output_parallel.0"]
    tensor = row.repeat(args.rows, 1).to(device=f"cuda:{local_rank}")

    # Exercise workspace generations as well as a single call.
    ce_outputs = [tp3_ce_all_reduce(tensor, dist.group.WORLD) for _ in range(2)]
    device_ce_outputs = [_device_ce_all_reduce(tensor) for _ in range(2)]
    reference = tensor.clone()
    dist.all_reduce(reference)
    torch.cuda.synchronize()

    probes = (0, args.rows // 3 - 1, 2 * args.rows // 3 - 1, args.rows - 1)
    result = {
        "rank": rank,
        "probes": probes,
        "ce_invariance": [
            max(_max_abs(output[0], output[index]) for index in probes)
            for output in ce_outputs
        ],
        "ce_repeat": _max_abs(ce_outputs[0], ce_outputs[1]),
        "device_ce_invariance": [
            max(_max_abs(output[0], output[index]) for index in probes)
            for output in device_ce_outputs
        ],
        "device_ce_repeat": _max_abs(device_ce_outputs[0], device_ce_outputs[1]),
        "device_ce_vs_host_ce": [
            _max_abs(output, ce_outputs[0]) for output in device_ce_outputs
        ],
        "nccl_invariance": max(
            _max_abs(reference[0], reference[index]) for index in probes
        ),
        "ce_vs_nccl": [_max_abs(output, reference) for output in ce_outputs],
    }
    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, result)
    if rank == 0:
        print(gathered, flush=True)
        assert all(item is not None for item in gathered)
        assert all(
            item["ce_invariance"] == [0.0, 0.0]  # type: ignore[index]
            for item in gathered
        )
        assert all(item["ce_repeat"] == 0.0 for item in gathered)  # type: ignore[index]
        assert all(
            item["device_ce_invariance"] == [0.0, 0.0]  # type: ignore[index]
            for item in gathered
        )
        assert all(
            item["device_ce_repeat"] == 0.0  # type: ignore[index]
            for item in gathered
        )
        assert all(
            item["device_ce_vs_host_ce"] == [0.0, 0.0]  # type: ignore[index]
            for item in gathered
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
