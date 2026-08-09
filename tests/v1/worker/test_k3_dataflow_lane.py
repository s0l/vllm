from __future__ import annotations

import pytest
import torch

from vllm.v1.worker.gpu.k3_dataflow_lane import (
    K3DataflowLane,
    K3DataflowLaneError,
    K3DataflowTicket,
)


def tickets() -> tuple[K3DataflowTicket, ...]:
    return (
        K3DataflowTicket(0, 0, 0, "attention"),
        K3DataflowTicket(1, 1, 0, "attention"),
        K3DataflowTicket(2, 0, 0, "mlp"),
        K3DataflowTicket(3, 1, 0, "mlp"),
    )


def test_ticket_plan_rejects_order_and_identity_mutations(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: object())
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    device = torch.device("cuda", 0)
    with pytest.raises(K3DataflowLaneError, match="contiguous"):
        K3DataflowLane(
            (
                K3DataflowTicket(0, 0, 0, "attention"),
                K3DataflowTicket(2, 1, 0, "attention"),
            ),
            device=device,
        )
    with pytest.raises(K3DataflowLaneError, match="unique"):
        K3DataflowLane(
            (
                K3DataflowTicket(0, 0, 0, "attention"),
                K3DataflowTicket(1, 0, 0, "attention"),
            ),
            device=device,
        )


def test_lifecycle_fails_closed_before_gpu_submission(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: object())
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    lane = K3DataflowLane(tickets(), device=torch.device("cuda", 0))
    with pytest.raises(K3DataflowLaneError, match="begin"):
        lane.submit(tickets()[0], lambda: None)
    lane.begin()
    with pytest.raises(K3DataflowLaneError, match="already active"):
        lane.begin()
    with pytest.raises(K3DataflowLaneError, match="order mismatch"):
        lane.submit(tickets()[1], lambda: None)
    with pytest.raises(K3DataflowLaneError, match="incomplete"):
        lane.finish()
    lane.abort()
    assert lane.cursor == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_replay_is_exact_and_host_thread_never_waits():
    device = torch.device("cuda", 0)
    compute = torch.cuda.Stream(device=device)
    lane = K3DataflowLane(tickets(), device=device)
    baselines = [
        torch.arange(64, device=device, dtype=torch.float32) + index
        for index in range(4)
    ]
    values = [value.clone() for value in baselines]
    outputs = [torch.empty_like(value) for value in values]
    tail = torch.cuda.Event()

    def run() -> None:
        origin = torch.cuda.current_stream(device)
        lane.begin(origin_stream=origin, compute_streams=(compute,))
        for ticket, baseline, value, output in zip(
            tickets(), baselines, values, outputs, strict=True
        ):
            with torch.cuda.stream(compute):
                value.copy_(baseline)
                value.add_(1.0)
            lane.submit(
                ticket,
                lambda value=value, output=output: torch.mul(value, 2.0, out=output),
                compute_stream=compute,
            )
            with torch.cuda.stream(compute):
                output.add_(3.0)
        lane.finish()
        tail.record(compute)
        origin.wait_event(tail)

    for _ in range(2):
        run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize(device)
    graph.replay()
    torch.cuda.synchronize(device)
    first = [output.clone() for output in outputs]
    graph.replay()
    torch.cuda.synchronize(device)
    second = [output.clone() for output in outputs]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert lane.cursor == len(tickets())
