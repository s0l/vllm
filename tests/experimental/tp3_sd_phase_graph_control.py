"""Three-GPU CUDA Graph control for the explicit TP3 decode phase lane."""

from __future__ import annotations

import gc
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


def _context(phase_reduce: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        num_tokens_unpadded=None,
        tp3_ce_reduce=False,
        tp3_mtp_device_ce=False,
        tp3_sd_phase_reduce=phase_reduce,
    )


def _capture(
    tensor: torch.Tensor,
    *,
    group_name: str,
    phase_reduce: bool,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.cuda.stream(side_stream),
        override_forward_context(_context(phase_reduce)),
    ):
        for _ in range(3):
            parallel_state.all_reduce(tensor, group_name)
    torch.cuda.current_stream().wait_stream(side_stream)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with (
        override_forward_context(_context(phase_reduce)),
        torch.cuda.graph(graph),
    ):
        output = parallel_state.all_reduce(tensor, group_name)
    return graph, output


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
        unique_name="tp3-sd-phase-control",
    )
    group_name = "tp3-sd-phase-control"
    group = SimpleNamespace(
        world_size=world_size,
        device_communicator=communicator,
        _all_reduce_out_place=communicator.all_reduce,
    )
    parallel_state._groups[group_name] = lambda: group
    os.environ.update(
        {
            "VLLM_TP3_CE_REDUCE": "0",
            "VLLM_TP3_SD_PHASE_REDUCE": "1",
            "VLLM_TP3_SD_DETERMINISTIC_REDUCE": "0",
            "VLLM_TP3_SD_CANONICAL_REDUCE": "0",
        }
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(8800 + rank)
    source = torch.randn(
        (3, 5120),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    gathered = communicator.all_gather(source, dim=0).view(world_size, *source.shape)
    oracle = parallel_state._tp3_sd_deterministic_sum(gathered)
    default = communicator.all_reduce(source.clone())

    phase_graph, phase_output = _capture(
        source,
        group_name=group_name,
        phase_reduce=True,
    )
    default_graph, default_output = _capture(
        source,
        group_name=group_name,
        phase_reduce=False,
    )
    phase_graph.replay()
    default_graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(phase_output, oracle, rtol=0, atol=0)
    torch.testing.assert_close(default_output, default, rtol=0, atol=0)
    if rank == 0:
        print(
            {
                "phase_graph_exact_to_fixed_sum": True,
                "false_lane_exact_to_default": True,
                "phase_vs_default_differing_elements": int(
                    torch.count_nonzero(phase_output != default_output).item()
                ),
                "shape": tuple(source.shape),
            }
        )

    # Graph-captured NCCL work owns communicator resources. Release every
    # graph on every rank before the collective communicator teardown; rank 0
    # also performs host-visible result extraction above and must not lag the
    # other ranks into destroy().
    del phase_graph, default_graph, phase_output, default_output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    parallel_state._groups.pop(group_name, None)
    communicator.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
