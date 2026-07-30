"""Replay TP3 BF16 all-reduce with saved real GDN rank-local rows.

This is a diagnostic control, not an inference or model-quality test.  It
places the exact saved rank-local output-projection row at every row position
observed in a real wave, pads the rest of the message with zeros, and calls the
same PyNCCL communicator used by the model runtime.

Run this script in a fresh three-rank process for each NCCL_ALGO/NCCL_PROTO
combination.  NCCL reads those variables when the communicator is created.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

LOCAL_STAGE = "gdn_output_parallel.0"
REDUCED_STAGE = "gdn_output.0"
QUERY_LENGTHS = (36, 56, 649, 721)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-prefix", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_trace(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def paths_for(prefix: str, query_len: int, rank: int) -> list[str]:
    pattern = f"{prefix}.q{query_len}.occ*.rank{rank}.pt"
    paths = glob.glob(pattern)
    return sorted(paths, key=lambda path: load_trace(path)["occurrence"])


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    expected_f = expected.float()
    delta = actual_f - expected_f
    denominator = torch.linalg.vector_norm(expected_f)
    relative_l2 = torch.linalg.vector_norm(delta) / denominator
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(delta.square()))),
        "relative_l2": float(relative_l2),
    }


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy()).hexdigest()


def main() -> None:
    args = parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 3:
        raise RuntimeError(f"expected TP3, got world_size={world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    communicator = PyNcclCommunicator(dist.group.WORLD, device=device)
    if not communicator.available:
        raise RuntimeError("PyNCCL communicator is unavailable")

    # A global Tree-only override is invalid for AllGather. Exercise the other
    # collective used by the real MTP startup before accepting an AllReduce
    # algorithm result.
    all_gather_input = torch.full(
        (8,),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    all_gather_output = torch.empty(
        (world_size * all_gather_input.numel(),),
        dtype=all_gather_input.dtype,
        device=device,
    )
    communicator.all_gather(all_gather_output, all_gather_input)
    torch.cuda.synchronize(device)
    all_gather_expected = torch.cat(
        [
            torch.full_like(all_gather_input.cpu(), source_rank + 1)
            for source_rank in range(world_size)
        ]
    )
    all_gather_exact = torch.equal(
        all_gather_output.cpu(),
        all_gather_expected,
    )
    if not all_gather_exact:
        raise RuntimeError("PyNCCL AllGather positive control failed")

    per_query: dict[str, Any] = {}
    for query_len in QUERY_LENGTHS:
        local_paths = paths_for(args.trace_prefix, query_len, rank)
        if not local_paths:
            raise RuntimeError(
                f"no saved traces for qlen={query_len}, rank={rank}"
            )

        local_traces = [load_trace(path) for path in local_paths]
        local_reference = local_traces[0]["layers"][LOCAL_STAGE].contiguous()
        if any(
            not torch.equal(
                trace["layers"][LOCAL_STAGE],
                local_reference,
            )
            for trace in local_traces[1:]
        ):
            raise RuntimeError(
                f"saved local rows differ within qlen={query_len}, rank={rank}"
            )

        # The authoritative V5 comparison established equality across query
        # shapes. Recheck it here so a mismatched artifact cannot become input
        # evidence for the collective.
        q36_reference = load_trace(
            paths_for(args.trace_prefix, 36, rank)[0]
        )["layers"][LOCAL_STAGE]
        if not torch.equal(local_reference, q36_reference):
            raise RuntimeError(
                f"saved local row differs from qlen=36 at rank={rank}, "
                f"qlen={query_len}"
            )

        width = local_reference.numel()
        collective_input = torch.zeros(
            (query_len, width),
            dtype=local_reference.dtype,
            device=device,
        )
        rows: list[int] = []
        occurrences: list[int] = []
        for trace in local_traces:
            row = int(trace["row"])
            if not 0 <= row < query_len:
                raise RuntimeError(
                    f"invalid row={row} for qlen={query_len}"
                )
            collective_input[row].copy_(local_reference.to(device))
            rows.append(row)
            occurrences.append(int(trace["occurrence"]))

        collective_output = communicator.all_reduce(collective_input)
        torch.cuda.synchronize(device)
        output_cpu = collective_output.cpu()

        row_results: list[dict[str, Any]] = []
        for trace, row, occurrence in zip(
            local_traces, rows, occurrences, strict=True
        ):
            actual = output_cpu[row]
            saved_reduced = trace["layers"][REDUCED_STAGE]

            rank_inputs = [
                load_trace(
                    paths_for(args.trace_prefix, query_len, source_rank)[
                        occurrence
                    ]
                )["layers"][LOCAL_STAGE]
                for source_rank in range(world_size)
            ]
            fixed_oracle = (
                rank_inputs[0].float()
                + rank_inputs[1].float()
                + rank_inputs[2].float()
            ).to(torch.bfloat16)
            row_results.append(
                {
                    "occurrence": occurrence,
                    "row": row,
                    "vs_saved_runtime": metrics(actual, saved_reduced),
                    "vs_fixed_fp32_oracle": metrics(actual, fixed_oracle),
                    "output_sha256": tensor_digest(actual),
                }
            )

        per_query[str(query_len)] = {
            "message_elements": collective_input.numel(),
            "message_bytes": collective_input.numel()
            * collective_input.element_size(),
            "rows": row_results,
        }

    gathered: list[dict[str, Any] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    rank_result = {
        "rank": rank,
        "device": str(device),
        "all_gather_exact": all_gather_exact,
        "queries": per_query,
    }
    dist.gather_object(rank_result, gathered, dst=0)

    if rank == 0:
        assert gathered is not None
        cross_rank_exact = True
        for query_len in QUERY_LENGTHS:
            query_key = str(query_len)
            rank0_rows = gathered[0]["queries"][query_key]["rows"]
            for source_rank in range(1, world_size):
                other_rows = gathered[source_rank]["queries"][query_key]["rows"]
                if [
                    row["output_sha256"] for row in other_rows
                ] != [row["output_sha256"] for row in rank0_rows]:
                    cross_rank_exact = False

        saved_runtime_exact = sum(
            row["vs_saved_runtime"]["exact"]
            for query in per_query.values()
            for row in query["rows"]
        )
        total_rows = sum(
            len(query["rows"]) for query in per_query.values()
        )
        result = {
            "schema": 1,
            "method": "PyNcclCommunicator all_reduce over saved real BF16 rows",
            "trace_prefix": args.trace_prefix,
            "environment": {
                "NCCL_ALGO": os.environ.get("NCCL_ALGO"),
                "NCCL_PROTO": os.environ.get("NCCL_PROTO"),
                "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE"),
                "CUDA_VISIBLE_DEVICES": os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "nccl_version": communicator.nccl.ncclGetVersion(),
            },
            "controls": {
                "world_size": world_size,
                "cross_rank_exact": cross_rank_exact,
                "all_gather_exact_all_ranks": all(
                    item["all_gather_exact"] for item in gathered
                ),
                "saved_runtime_exact_rows": saved_runtime_exact,
                "total_rows": total_rows,
                "default_reproduced_saved_runtime": (
                    saved_runtime_exact == total_rows
                ),
            },
            "ranks": gathered,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result["environment"], sort_keys=True))
        print(json.dumps(result["controls"], sort_keys=True))

    communicator.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
