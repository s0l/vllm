"""Three-rank exact-output/liveness oracle for conveyor-owned device CE."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import (
    _tp3_device_ce_reduce,
    _tp3_device_ce_reduce_impl,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import ForwardContext
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper, UbatchMetadata
from vllm.v1.worker.ubatching import make_ubatch_contexts


class _Group:
    world_size = 3

    @staticmethod
    def _all_gather_out_place(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        output = torch.empty(
            (dist.get_world_size() * tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(output, tensor)
        return output


@torch.inference_mode()
def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    group = _Group()
    generators = []
    for wave in range(2):
        generator = torch.Generator(device=device)
        generator.manual_seed(7000 + rank * 10 + wave)
        generators.append(generator)
    inputs = [
        torch.randn(76, 5120, device=device, dtype=torch.bfloat16, generator=g)
        for g in generators
    ]

    expected = [_tp3_device_ce_reduce_impl(value, group) for value in inputs]
    torch.cuda.synchronize()
    dist.barrier()

    parallel_config = SimpleNamespace(
        num_ubatches=1,
        enable_expert_parallel=False,
        enable_dbo=False,
        all2all_backend="naive",
    )
    config = SimpleNamespace(
        parallel_config=parallel_config,
        compilation_config=None,
    )
    wrapper = UBatchWrapper(
        lambda **kwargs: kwargs["input_ids"],
        config,
        CUDAGraphMode.NONE,
        device,
        num_ubatches_override=2,
    )
    contexts = make_ubatch_contexts(
        2,
        torch.cuda.current_stream(),
        wrapper.comm_stream,
        [ForwardContext({}, {}, {}), ForwardContext({}, {}, {})],
        wrapper.ready_barrier,
    )
    metadata = [
        UbatchMetadata(
            context=contexts[wave],
            input_ids=inputs[wave],
            positions=torch.empty(0, device=device),
            inputs_embeds=None,
            intermediate_tensors=None,
            num_tokens=inputs[wave].shape[0],
        )
        for wave in range(2)
    ]

    def run_wave(**kwargs) -> torch.Tensor:
        # Real compute on the shared stream before and after CE proves event
        # ownership, while the CE wrapper owns both yield points.
        value = kwargs["input_ids"]
        scratch = value.float().square().mean(dim=1)
        reduced = _tp3_device_ce_reduce(value, group)
        return reduced + scratch[:, None].to(reduced.dtype) * 0

    combined = wrapper._run_ubatches(metadata, run_wave)
    for wave, actual in enumerate(combined.chunk(2)):
        if not torch.equal(actual, expected[wave]):
            raise RuntimeError(f"wave {wave} changed device-CE output")
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(
            "PASS TP3 conveyor CE schedule: two waves, exact output, no deadlock",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
