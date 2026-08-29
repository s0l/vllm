import json

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_next import (
    _ag2_packetize_and_clear_trace_attrs,
    _ag2_release_layer_trace_refs,
    _ag2_trace_packet,
    _reject_removed_projection_calibration,
)
from vllm.v1.worker.gpu.aux_hidden_trace import AuxHiddenTrace


def test_removed_projection_calibration_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_PROJECTION_CALIBRATION_OUTPUT", "/tmp/rejected")

    with pytest.raises(RuntimeError, match="observer is not neutral"):
        _reject_removed_projection_calibration()


def test_request_chunk_manifest_reports_saved_invocations(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "trace"
    trace = AuxHiddenTrace(
        output=str(output),
        layers=(0,),
        request_chunks=True,
    )
    trace._saved_request_invocations = 3

    trace._write_manifest(rank=2, state="active")

    manifest = json.loads(
        (tmp_path / "trace.rank2.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "active"
    assert manifest["saved_request_invocations"] == 3


def test_fingerprint_detects_bit_permutation_and_magnitude(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")
    torch.manual_seed(7)
    value = torch.randn(31, 5120, dtype=torch.bfloat16)
    expected = _ag2_trace_packet(value)

    assert expected.shape == (31, 9)
    assert expected.dtype == torch.int64
    assert torch.equal(expected, _ag2_trace_packet(value.clone()))

    bit_mutation = value.clone()
    bit_mutation.view(torch.int16)[3, 17] ^= 1
    assert not torch.equal(expected[3], _ag2_trace_packet(bit_mutation)[3])

    permutation = value.clone()
    permutation[5, [10, 11]] = permutation[5, [11, 10]]
    assert not torch.equal(expected[5], _ag2_trace_packet(permutation)[5])

    magnitude = value.clone()
    magnitude[9, 23] += torch.tensor(0.5, dtype=torch.bfloat16)
    assert not torch.equal(expected[9], _ag2_trace_packet(magnitude)[9])


def test_fingerprint_width_remains_bytes(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")
    value = torch.randn(3, 64, dtype=torch.bfloat16)

    packet = _ag2_trace_packet(value)

    assert torch.equal(packet[:, 0], torch.full((3,), 128, dtype=torch.int64))


def test_fingerprint_is_neighbor_invariant(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")
    row = torch.randn(1, 5120, dtype=torch.bfloat16)
    batch = torch.randn(31, 5120, dtype=torch.bfloat16)
    batch[19] = row[0]

    assert torch.equal(_ag2_trace_packet(row)[0], _ag2_trace_packet(batch)[19])


def test_fingerprint_stays_opaque_to_torch_compile(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")
    value = torch.randn(3, 64, dtype=torch.bfloat16)

    @torch.compile(fullgraph=True, backend="eager")
    def compiled(candidate: torch.Tensor) -> torch.Tensor:
        return _ag2_trace_packet(candidate)

    expected = _ag2_trace_packet(value)
    assert torch.equal(expected, compiled(value))

    explanation = torch._dynamo.explain(compiled)(value)
    nodes = [node for graph in explanation.graphs for node in graph.graph.nodes]
    assert any("ag2_trace_packet" in str(node) for node in nodes)


def test_release_raw_refs_preserves_packets_and_configuration(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")

    class DiagnosticLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._ag2_aux_q = torch.randn(3, 8)
            self._ag2_aux_boundaries = (
                torch.randn(3, 8),
                torch.randn(3, 8),
            )
            self._ag2_aux_enabled = True
            self.register_buffer(
                "_ag2_aux_registered", torch.ones(1), persistent=False
            )

    model = nn.Sequential(DiagnosticLayer())
    packet = _ag2_trace_packet(model[0]._ag2_aux_q)
    trace = AuxHiddenTrace(output="unused", layers=(0,), fingerprint_outputs=True)

    assert trace.release_raw_model_refs(model) == 2
    assert model[0]._ag2_aux_q is None
    assert model[0]._ag2_aux_boundaries is None
    assert model[0]._ag2_aux_enabled is True
    assert torch.equal(model[0]._ag2_aux_registered, torch.ones(1))
    assert packet.shape == (3, 9)


def test_semantic_boundary_packetization_releases_raw_attrs(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")
    owner = nn.Module()
    owner.first = torch.randn(3, 64, dtype=torch.bfloat16)
    owner.second = torch.randn(3, 32, dtype=torch.bfloat16)
    expected = (_ag2_trace_packet(owner.first), _ag2_trace_packet(owner.second))

    packets = _ag2_packetize_and_clear_trace_attrs(
        owner,
        ("first", "second"),
    )

    assert all(torch.equal(actual, wanted) for actual, wanted in zip(packets, expected))
    assert owner.first is None
    assert owner.second is None


def test_compiled_layer_release_has_no_raw_side_effect_output(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")

    class ObservedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.mlp = nn.Module()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self._ag2_aux_compact_input_norm = value.square() + value
            self.self_attn._ag2_aux_q = self._ag2_aux_compact_input_norm + 1
            self.mlp._ag2_aux_compact_gate_up = self.self_attn._ag2_aux_q + 1
            packets = torch.stack(
                (
                    _ag2_trace_packet(self._ag2_aux_compact_input_norm),
                    _ag2_trace_packet(self.self_attn._ag2_aux_q),
                    _ag2_trace_packet(self.mlp._ag2_aux_compact_gate_up),
                )
            )
            _ag2_release_layer_trace_refs(self)
            return packets

    layer = ObservedLayer()
    compiled = torch.compile(layer, fullgraph=True, backend="eager")
    value = torch.randn(3, 64, dtype=torch.bfloat16)
    expected = compiled(value)

    assert torch.equal(expected, compiled(value))
    assert layer._ag2_aux_compact_input_norm is None
    assert layer.self_attn._ag2_aux_q is None
    assert layer.mlp._ag2_aux_compact_gate_up is None
