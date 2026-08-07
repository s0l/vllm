"""Three-GPU CUDA Graph control for explicit MTP-only device CE routing."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import vllm.distributed.parallel_state as parallel_state
from vllm.config import CUDAGraphMode
from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)
from vllm.forward_context import override_forward_context


def _context(mtp: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        num_tokens_unpadded=None,
        tp3_ce_reduce=False,
        tp3_mtp_device_ce=mtp,
        tp3_sd_phase_reduce=False,
    )


def _capture(
    tensor: torch.Tensor,
    *,
    group_name: str,
    mtp: bool,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream), override_forward_context(_context(mtp)):
        for _ in range(3):
            parallel_state.all_reduce(tensor, group_name)
    torch.cuda.current_stream().wait_stream(side_stream)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with override_forward_context(_context(mtp)), torch.cuda.graph(graph):
        output = parallel_state.all_reduce(tensor, group_name)
    return graph, output


def _row_classes(tensor: torch.Tensor) -> int:
    """Count byte-distinct rows without reducing floating-point values."""
    rows = tensor.detach().cpu().contiguous().view(torch.uint8)
    return len({bytes(row.tolist()) for row in rows})


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 3
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    communicator = CudaCommunicator(
        dist.group.WORLD,
        device=device,
        unique_name="tp3-mtp-device-ce-control",
    )
    group_name = "tp3-mtp-device-ce-control"
    group = SimpleNamespace(
        world_size=world_size,
        device_communicator=communicator,
        _all_reduce_out_place=communicator.all_reduce,
        _all_gather_out_place=communicator.all_gather,
    )
    parallel_state._groups[group_name] = lambda: group
    os.environ.update(
        {
            "AG2_VLLM_MTP_DEVICE_CE": "1",
            "AG2_VLLM_TP3_PIECEWISE_DEVICE_CE": "0",
            "AG2_VLLM_TP3_PIECEWISE_DEVICE_CE_PACKED": "0",
            "VLLM_TP3_CE_REDUCE": "0",
            "VLLM_TP3_SD_PHASE_REDUCE": "0",
            "VLLM_TP3_SD_DETERMINISTIC_REDUCE": "0",
            "VLLM_TP3_SD_CANONICAL_REDUCE": "0",
        }
    )

    results = []
    for rows in (3, 31):
        generator = torch.Generator(device=device)
        generator.manual_seed(2026080300 + rows * 10 + rank)
        rank_row = torch.randn(
            (1, 5120),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        # Every semantic row receives the identical rank-local contribution.
        # A row-wise deterministic reducer must therefore emit one byte class,
        # regardless of physical row offset or captured graph shape.
        source = rank_row.expand(rows, -1).clone()
        ce_oracle = parallel_state._tp3_device_ce_reduce(source, group)
        default_oracle = communicator.all_reduce(source.clone())

        mtp_graph, mtp_output = _capture(
            source,
            group_name=group_name,
            mtp=True,
        )
        target_graph, target_output = _capture(
            source,
            group_name=group_name,
            mtp=False,
        )
        mtp_graph.replay()
        target_graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(mtp_output, ce_oracle, rtol=0, atol=0)
        torch.testing.assert_close(target_output, default_oracle, rtol=0, atol=0)
        mtp_row_classes = _row_classes(mtp_output)
        target_row_classes = _row_classes(target_output)
        assert mtp_row_classes == 1

        # Mutation control: the row-class detector must reject a known split.
        mutated = mtp_output.clone()
        mutated[-1, 0] = torch.nextafter(
            mutated[-1, 0],
            torch.full_like(mutated[-1, 0], float("inf")),
        )
        assert _row_classes(mutated) == 2
        results.append(
            {
                "rows": rows,
                "mtp_graph_exact_to_device_ce": True,
                "target_graph_exact_to_default": True,
                "mtp_duplicate_row_classes": mtp_row_classes,
                "target_duplicate_row_classes": target_row_classes,
                "row_class_mutation_control": True,
                "mtp_vs_target_differing_elements": int(
                    torch.count_nonzero(mtp_output != target_output).item()
                ),
            }
        )

    if rank == 0:
        print({"schema": "ag2-tp3-mtp-device-ce-graph-v1", "results": results})

    parallel_state._groups.pop(group_name, None)
    communicator.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
