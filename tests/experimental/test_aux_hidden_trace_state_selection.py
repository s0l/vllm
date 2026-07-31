# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu.aux_hidden_trace as trace_module
from vllm.v1.worker.gpu.aux_hidden_trace import AuxHiddenTrace


class _FakeStream:
    def synchronize(self) -> None:
        pass


def test_per_request_gdn_boundaries_use_occurrence_not_token_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        trace_module,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeStream())

    trace = AuxHiddenTrace(
        output=str(tmp_path / "trace"),
        layers=(1, 2),
        token_id=198,
        position=1512,
        first_gdn_boundaries=True,
        gdn_boundary_layer=1,
    )
    labels = trace.output_labels()
    full_replay = {label for label in labels if label.startswith("gdn_replay_")}
    hidden_states = [
        (
            torch.arange(6).reshape(2, 3) + index * 100
            if label in full_replay
            else torch.arange(50).unsqueeze(1) + index * 1000
        )
        for index, label in enumerate(labels)
    ]
    input_ids = torch.zeros(50, dtype=torch.long)
    positions = torch.zeros(50, dtype=torch.long)
    input_ids[[24, 49]] = 198
    positions[[24, 49]] = 1512

    trace.maybe_save(
        input_ids=input_ids,
        positions=positions,
        query_len=50,
        cudagraph_mode="PIECEWISE",
        aux_hidden_states=hidden_states,
    )

    for occurrence, token_row in enumerate((24, 49)):
        saved = torch.load(
            tmp_path / f"trace.q50.occ{occurrence}.rank0.pt",
            weights_only=False,
        )
        for index, label in enumerate(labels):
            if label in full_replay:
                assert torch.equal(
                    saved["layers"][label],
                    torch.arange(6).reshape(2, 3) + index * 100,
                )
                continue
            expected = token_row + index * 1000
            assert saved["layers"][label].item() == expected
