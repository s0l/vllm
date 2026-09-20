# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Three-GPU numerical and peak-memory control for DCP chunked reduction."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)


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
        unique_name="dcp-control",
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(7300 + rank)
    source = torch.randn(
        (8192, 48, 128),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    torch.cuda.synchronize()

    reference_base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    reference = communicator.reduce_scatter(source, dim=1)
    torch.cuda.synchronize()
    reference_peak = torch.cuda.max_memory_allocated(device) - reference_base

    chunked_base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    actual = communicator.reduce_scatter_chunked(source, dim=1)
    torch.cuda.synchronize()
    chunked_peak = torch.cuda.max_memory_allocated(device) - chunked_base

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    if rank == 0:
        print(
            {
                "reference_shape": tuple(reference.shape),
                "exact": True,
                "reference_peak_mib": reference_peak / 1024**2,
                "chunked_peak_mib": chunked_peak / 1024**2,
                "pid": os.getpid(),
            }
        )
    communicator.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
