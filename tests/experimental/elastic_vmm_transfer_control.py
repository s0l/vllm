"""Live CUDA control for transactional elastic VMM handle transfer."""

from __future__ import annotations

import torch

from vllm.device_allocator.elastic_cumem import ElasticCuMemBacking


def main() -> None:
    device = torch.device("cuda:0")
    quantum = 2 * 1024**2
    source = ElasticCuMemBacking(8 * quantum, 4 * quantum, quantum, device)
    destination = ElasticCuMemBacking(8 * quantum, 2 * quantum, quantum, device)

    source.tensor[3 * quantum : 4 * quantum].fill_(123)
    fence = torch.cuda.Event()
    fence.record()
    source.transfer_to(destination, quantum, fence)
    assert source.info.committed == 3 * quantum
    assert destination.info.committed == 3 * quantum
    assert bool(torch.all(destination.tensor[2 * quantum : 3 * quantum] == 123).item())

    reverse_fence = torch.cuda.Event()
    reverse_fence.record()
    destination.transfer_to(source, quantum, reverse_fence)
    assert source.info.committed == 4 * quantum
    assert destination.info.committed == 2 * quantum
    assert bool(torch.all(source.tensor[3 * quantum : 4 * quantum] == 123).item())
    print(
        {
            "forward": "pass",
            "reverse": "pass",
            "physical_contents_preserved": True,
            "quantum_mib": quantum / 1024**2,
        }
    )


if __name__ == "__main__":
    main()
