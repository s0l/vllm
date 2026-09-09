# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.aux_hidden_trace as trace_module
from vllm.v1.worker.gpu.aux_hidden_trace import AuxHiddenTrace


class _FakeStream:
    def synchronize(self) -> None:
        pass


def _trace(tmp_path) -> AuxHiddenTrace:
    trace = AuxHiddenTrace(
        output=str(tmp_path / "trace"),
        layers=(0, 1, 2, 3, 4),
        position=3,
        max_matches_per_query_len=4,
        fingerprint_outputs=True,
        request_prefix="upstream-",
        request_chunks=True,
        sequence_boundary_layers=(0, 1, 2, 3),
    )
    trace._layer_types = {
        0: "linear_attention",
        1: "linear_attention",
        2: "linear_attention",
        3: "full_attention",
    }
    return trace


def test_sequence_boundary_schema_is_ordered_and_bounded(tmp_path) -> None:
    trace = _trace(tmp_path)
    labels = trace.output_labels()

    assert len(labels) == 60
    assert labels[:4] == (
        "layer.0",
        "sequence_input_norm.0",
        "sequence_gdn_qkvz.0",
        "sequence_gdn_ba.0",
    )
    assert "sequence_full_gate.3" in labels
    assert labels[-1] == "layer.4"
    assert not any("replay" in label for label in labels)
    assert trace.output_scopes()["sequence_gdn_qkvz.0"] == "rank_local"
    assert trace.output_scopes()["sequence_gdn_output.0"] == "global"

    max_query_len = 6656
    packet_bytes = len(labels) * max_query_len * 9 * torch.int64.itemsize
    assert packet_bytes == 28_753_920


def test_sequence_boundary_request_packet_uses_dedicated_schema(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        trace_module,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeStream())
    trace = _trace(tmp_path)
    labels = trace.output_labels()
    packets = [torch.arange(36, dtype=torch.int64).reshape(4, 9) for _ in labels]

    trace.maybe_save(
        input_ids=torch.tensor([10, 11, 12, 13]),
        positions=torch.tensor([0, 1, 2, 3]),
        query_len=4,
        cudagraph_mode="FULL",
        aux_hidden_states=packets,
        req_ids=["upstream-0"],
        query_start_loc=[0, 4],
        num_scheduled_tokens=[4],
        num_computed_tokens=[0],
        slot_mappings_by_layer={"layer0": torch.tensor([1, 2, 3, 4])},
    )

    artifact = torch.load(
        tmp_path / "trace.inv000000.rank0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert artifact["schema"] == "ag2-upstream-sequence-fingerprint-v1"
    assert list(artifact["layers"]) == list(labels)
    assert all(value.shape == (4, 9) for value in artifact["layers"].values())


def test_sequence_boundaries_require_complete_checkpoint_contract(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "0,1,2,3")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "3")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_PREFIX", "upstream-")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_CHUNKS", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_SEQUENCE_BOUNDARY_LAYERS", "0,1,2,3")

    with pytest.raises(ValueError, match="following checkpoint"):
        AuxHiddenTrace.from_env()
