# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.aux_hidden_trace as trace_module
from vllm.model_executor.models.qwen3_next import Qwen3NextModel
from vllm.v1.worker.gpu.aux_hidden_trace import AuxHiddenTrace
from vllm.v1.worker.gpu.cudagraph_utils import copy_aux_hidden_state


class _FakeStream:
    def synchronize(self) -> None:
        pass


def test_compact_checkpoint_trace_configuration(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "4,8,16")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "198")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1512")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ROWS", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_CAPACITY", "44")

    trace = AuxHiddenTrace.from_env()

    assert trace.compact_rows is True
    assert trace.compact_capacity == 44
    assert trace.output_labels() == (
        "layer_row_indices.4",
        "layer_hidden.4",
        "layer_residual.4",
        "layer.4",
        "layer_row_indices.8",
        "layer_hidden.8",
        "layer_residual.8",
        "layer.8",
        "layer_row_indices.16",
        "layer_hidden.16",
        "layer_residual.16",
        "layer.16",
    )


def test_compact_checkpoint_trace_accepts_layer_zero(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "0,4")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "198")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1512")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ROWS", "1")

    trace = AuxHiddenTrace.from_env()

    assert trace.layers == (0, 4)


def test_request_chunk_trace_requires_complete_causal_contract(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "4,5")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "-1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1512")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_PREFIX", "causal-")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_CHUNKS", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_GDN_BOUNDARIES", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_GDN_BOUNDARY_LAYER", "4")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")

    trace = AuxHiddenTrace.from_env()

    assert trace.request_chunks is True
    assert trace.token_id == -1
    assert trace.position == 1512
    assert trace.first_gdn_boundaries is True


def test_request_chunk_trace_accepts_complete_layer_checkpoints(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "0,1,2")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "-1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1512")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_PREFIX", "coarse-")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_CHUNKS", "1")
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1"
    )

    trace = AuxHiddenTrace.from_env()

    assert trace.request_chunks is True
    assert trace.output_labels() == ("layer.0", "layer.1", "layer.2")


def test_request_chunk_trace_saves_invocation_and_trims_replay_buffers(
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
        layers=(4, 5),
        token_id=-1,
        position=2,
        max_matches_per_query_len=4,
        first_gdn_boundaries=True,
        gdn_boundary_layer=4,
        fingerprint_outputs=True,
        request_prefix="causal-",
        request_chunks=True,
    )
    labels = trace.output_labels()
    meta = torch.full((512,), -1, dtype=torch.int64)
    meta[0] = 1
    meta[1:6] = torch.tensor([2, 3, 4, 5, 6])
    meta[6] = 4
    meta[38:42] = torch.tensor([4, 4, 1, 0])
    meta[42] = 3
    meta[43] = 20
    meta[44:48] = torch.tensor([1, 3, 1, 1])
    meta[48] = 1
    hidden_states = []
    for label in labels:
        if label == "gdn_replay_float.4":
            hidden_states.append(torch.arange(64, dtype=torch.float32))
        elif label == "gdn_replay_state.4":
            hidden_states.append(torch.arange(32, dtype=torch.float32))
        elif label == "gdn_replay_meta.4":
            hidden_states.append(meta)
        else:
            hidden_states.append(torch.arange(36, dtype=torch.int64).reshape(4, 9))

    trace.maybe_save(
        input_ids=torch.tensor([10, 11, 10, 11]),
        positions=torch.tensor([0, 1, 2, 3]),
        query_len=4,
        cudagraph_mode="FULL",
        aux_hidden_states=hidden_states,
        req_ids=["causal-00", "other-00"],
        query_start_loc=[0, 2, 4],
        num_scheduled_tokens=[2, 2],
        num_computed_tokens=[0, 0],
        slot_mappings_by_layer={"layer4": torch.tensor([9, 10, 11, 12])},
    )

    artifact = torch.load(
        tmp_path / "trace.inv000000.rank0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert artifact["schema"] == "ag2-gdn-sequence-causal-packet-v1"
    assert artifact["selected_request_indices"] == [0]
    assert artifact["requests"][0]["slot_mappings"]["layer4"] == [9, 10]
    assert artifact["layers"]["gdn_replay_float.4"].numel() == 20
    assert artifact["layers"]["gdn_replay_state.4"].numel() == 18
    assert int(artifact["layers"]["gdn_replay_meta.4"][43]) == 12


def test_full_internal_trace_declares_every_dense_qwen_boundary(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS",
        ",".join(str(layer) for layer in range(65)),
    )
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "198")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1512")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ROWS", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ALL_BOUNDARIES", "1")
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_MAX_LAYER", "63"
    )
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_ALL_INTERNAL_BOUNDARIES", "1"
    )
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_PREFIX", "chatcmpl-ag2-fr-v52-"
    )
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS", "1")

    trace = AuxHiddenTrace.from_env()
    trace._layer_types = {
        layer: "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        for layer in range(64)
    }
    labels = trace.output_labels()

    # 580 coarse layer/checkpoint fields + 720 internal fields:
    # 48 GDN*6 + 16 full-attention*11 + 64 MLP*4. The six large raw DCP/KV
    # history tensors remain a targeted replay facility; broad tracing keeps
    # exact DCP local/combined outputs and LSE without changing residency.
    assert len(labels) == 1300
    assert "compact_gdn_qkvz.0" in labels
    assert "compact_full_qkv.3" in labels
    assert "compact_mlp_down_parallel.63" in labels
    assert trace.request_prefix == "chatcmpl-ag2-fr-v52-"
    assert trace.fingerprint_outputs is True


def test_full_internal_trace_can_omit_unavailable_canonical_dcp_packs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS",
        ",".join(str(layer) for layer in range(65)),
    )
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "1833")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1343")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ROWS", "1")
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ALL_BOUNDARIES", "1"
    )
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_MAX_LAYER", "63"
    )
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_ALL_INTERNAL_BOUNDARIES", "1"
    )
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_FULL_ATTENTION_STAGES",
        "qkv,q,k,v,gate,core,gated,output_parallel,output",
    )

    trace = AuxHiddenTrace.from_env()
    trace._layer_types = {
        layer: "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        for layer in range(64)
    }
    labels = trace.output_labels()

    assert len(labels) == 1268
    assert "compact_full_core.3" in labels
    assert not any("dcp_output_pack" in label for label in labels)
    assert not any("dcp_lse_pack" in label for label in labels)


def test_compact_full_attention_stage_selection_rejects_invalid_sets(
    monkeypatch,
) -> None:
    base = {
        "AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT": "/tmp/trace",
        "AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS": "0,1",
        "AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID": "1833",
        "AG2_VLLM_AUX_HIDDEN_TRACE_POSITION": "1343",
    }
    for stages, message in (
        ("qkv,unknown", "Unknown compact"),
        ("qkv,dcp_output_pack", "must be requested together"),
    ):
        for key, value in base.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv(
            "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_FULL_ATTENTION_STAGES",
            stages,
        )
        with pytest.raises(ValueError, match=message):
            AuxHiddenTrace.from_env()


def test_selected_internal_trace_covers_early_layers_without_full_model_capture(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "0,1,2,3,4,5")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "1833")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1343")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ROWS", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ALL_BOUNDARIES", "1")
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_MAX_LAYER", "4"
    )
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_INTERNAL_BOUNDARY_LAYERS", "0,1,2,3,4"
    )

    trace = AuxHiddenTrace.from_env()
    trace._layer_types = {
        layer: "full_attention" if layer == 3 else "linear_attention"
        for layer in range(5)
    }
    labels = trace.output_labels()

    # 49 coarse checkpoint/boundary fields, 5*4 MLP fields, four GDN
    # layers*6 fields and one full-attention layer*11 fields.
    assert len(labels) == 104
    assert trace.internal_boundary_layers == (0, 1, 2, 3, 4)
    assert "compact_gdn_qkvz.0" in labels
    assert "compact_full_dcp_lse_pack.3" in labels
    assert "compact_mlp_output.4" in labels
    assert "compact_mlp_output.5" not in labels


def test_selected_internal_trace_rejects_layer_without_following_checkpoint(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "0,1,2,3,4,5")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "1833")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1343")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ROWS", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ALL_BOUNDARIES", "1")
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_MAX_LAYER", "4"
    )
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_INTERNAL_BOUNDARY_LAYERS", "5"
    )

    with pytest.raises(ValueError, match="following auxiliary checkpoint"):
        AuxHiddenTrace.from_env()


def test_compact_checkpoint_trace_rejects_internal_boundaries(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "3,4")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "198")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1512")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ROWS", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_ATTENTION_BOUNDARY", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "3")

    with pytest.raises(ValueError, match="global checkpoints only"):
        AuxHiddenTrace.from_env()


def test_compact_checkpoint_selects_matching_physical_rows(
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
        layers=(0, 2),
        token_id=198,
        position=1512,
        compact_rows=True,
        compact_capacity=4,
    )
    row_indices_0 = torch.tensor([49, 7, 24, 2])
    row_indices_2 = torch.tensor([24, 49, 8, 3])
    hidden_states = [
        row_indices_0,
        row_indices_0.unsqueeze(1) + 100,
        row_indices_0.unsqueeze(1) + 200,
        row_indices_0.unsqueeze(1) + 1000,
        row_indices_2,
        row_indices_2.unsqueeze(1) + 300,
        row_indices_2.unsqueeze(1) + 400,
        row_indices_2.unsqueeze(1) + 2000,
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
        assert saved["layers"]["layer.0"].item() == token_row + 1000
        assert saved["layers"]["layer.2"].item() == token_row + 2000
        assert torch.equal(
            saved["layers"]["layer_row_indices.0"],
            row_indices_0,
        )


def test_qwen3_next_compact_checkpoint_avoids_full_hidden_copy() -> None:
    model = SimpleNamespace(
        aux_hidden_state_layers=(4,),
        _ag2_aux_trace_compact_position=12,
        _ag2_aux_trace_compact_capacity=4,
    )
    positions = torch.tensor([10, 12, 11, 12, 13, 14])
    hidden = torch.arange(18).reshape(6, 3)
    residual = hidden + 100

    outputs = Qwen3NextModel._maybe_add_ag2_aux_hidden_state(
        model,
        [],
        4,
        hidden,
        residual,
        positions,
    )

    assert len(outputs) == 4
    selected_indices, selected_hidden, selected_residual, selected_combined = outputs
    assert selected_indices.shape == (4,)
    assert selected_hidden.shape == (4, 3)
    assert selected_residual.shape == (4, 3)
    assert selected_combined.shape == (4, 3)
    for output_index, row_index in enumerate(selected_indices.tolist()):
        assert torch.equal(selected_hidden[output_index], hidden[row_index])
        assert torch.equal(selected_residual[output_index], residual[row_index])
        assert torch.equal(
            selected_combined[output_index],
            hidden[row_index] + residual[row_index],
        )
    assert {1, 3}.issubset(set(selected_indices.tolist()))


def test_qwen3_next_compact_checkpoint_uses_text_mrope_time_row() -> None:
    model = SimpleNamespace(
        aux_hidden_state_layers=(0,),
        _ag2_aux_trace_compact_position=12,
        _ag2_aux_trace_compact_capacity=4,
    )
    text_positions = torch.tensor([10, 12, 11, 12, 13, 14])
    positions = text_positions.repeat(3, 1)
    hidden = torch.arange(18).reshape(6, 3)

    outputs = Qwen3NextModel._maybe_add_ag2_aux_hidden_state(
        model,
        [],
        0,
        hidden,
        None,
        positions,
    )

    selected_indices, selected_hidden, selected_residual, selected_combined = outputs
    assert selected_indices.shape == (4,)
    assert selected_hidden.shape == (4, 3)
    assert {1, 3}.issubset(set(selected_indices.tolist()))
    for row_index, row in zip(selected_indices.tolist(), selected_hidden, strict=True):
        assert torch.equal(row, hidden[row_index])
    assert torch.count_nonzero(selected_residual) == 0
    assert torch.equal(selected_combined, selected_hidden)


def test_qwen3_next_compact_checkpoint_has_fixed_graph_capacity() -> None:
    model = SimpleNamespace(
        aux_hidden_state_layers=(0,),
        _ag2_aux_trace_compact_position=12,
        _ag2_aux_trace_compact_capacity=64,
    )
    destinations = None

    for rows in (352, 64, 56, 35, 1):
        positions = torch.arange(rows).repeat(3, 1)
        if rows <= 12:
            positions[:, -1] = 12
        hidden = torch.arange(rows * 3).reshape(rows, 3)
        outputs = Qwen3NextModel._maybe_add_ag2_aux_hidden_state(
            model,
            [],
            0,
            hidden,
            None,
            positions,
        )
        row_indices, selected_hidden, selected_residual, selected_combined = outputs
        assert row_indices.shape == (64,)
        assert selected_hidden.shape == (64, 3)
        assert selected_residual.shape == (64, 3)
        assert selected_combined.shape == (64, 3)
        if rows < 64:
            assert (row_indices == -1).sum().item() == 64 - rows
            assert torch.count_nonzero(selected_hidden[row_indices == -1]) == 0
            assert torch.count_nonzero(selected_residual[row_indices == -1]) == 0
            assert torch.count_nonzero(selected_combined[row_indices == -1]) == 0
        expected_row = rows - 1 if rows <= 12 else 12
        assert expected_row in row_indices.tolist()

        if destinations is None:
            destinations = [torch.empty_like(output) for output in outputs]
        for destination, source in zip(destinations, outputs, strict=True):
            copy_aux_hidden_state(
                destination,
                source,
                num_tokens=rows,
                token_major=False,
            )


def test_global_checkpoint_trace_does_not_require_dcp_pack_capacity(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "4,8")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "198")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1512")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_BOUNDARIES", "0")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS", "0")

    trace = AuxHiddenTrace.from_env()

    assert trace.layers == (4, 8)
    assert trace.full_attention_boundaries is False


def test_full_dcp_boundary_trace_requires_pack_capacity(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "3,4")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "198")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "1512")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_ATTENTION_BOUNDARY", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "3")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_BOUNDARIES", "1")
    monkeypatch.setenv(
        "AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_STAGES",
        "dcp_output_pack,dcp_lse_pack",
    )
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS", "0")

    with pytest.raises(ValueError, match="positive DCP_REQUEST_TAIL_ROWS"):
        AuxHiddenTrace.from_env()


def test_single_full_boundary_position_allows_wildcard_token(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "3,4")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "-1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "384")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_ATTENTION_BOUNDARY", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "3")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_BOUNDARIES", "1")

    trace = AuxHiddenTrace.from_env()

    assert trace.token_id == -1
    assert trace.positions == (384,)


def test_multiple_full_boundary_positions_reject_wildcard_token(monkeypatch) -> None:
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT", "/tmp/trace")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "3,4")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "-1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_POSITIONS", "383,384")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_ATTENTION_BOUNDARY", "1")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "3")
    monkeypatch.setenv("AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_BOUNDARIES", "1")

    with pytest.raises(ValueError, match="Wildcard TOKEN_ID"):
        AuxHiddenTrace.from_env()


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
        req_ids=["request-a", "request-b"],
        query_start_loc=[0, 25, 50],
        num_scheduled_tokens=[25, 25],
        num_computed_tokens=[0, 1488],
        slot_mappings_by_layer={
            "model.layers.0.self_attn.attn": torch.arange(100, 150),
        },
    )

    for occurrence, token_row in enumerate((24, 49)):
        saved = torch.load(
            tmp_path / f"trace.q50.occ{occurrence}.rank0.pt",
            weights_only=False,
        )
        provenance = saved["request_provenance"]
        assert provenance == {
            "req_id": f"request-{'a' if occurrence == 0 else 'b'}",
            "request_index": occurrence,
            "request_row_offset": 24,
            "query_start": occurrence * 25,
            "query_end": (occurrence + 1) * 25,
            "num_scheduled_tokens": 25,
            "num_computed_tokens": 0 if occurrence == 0 else 1488,
            "slot_mappings": {
                "model.layers.0.self_attn.attn": 124 + occurrence * 25,
            },
        }
        for index, label in enumerate(labels):
            if label in full_replay:
                assert torch.equal(
                    saved["layers"][label],
                    torch.arange(6).reshape(2, 3) + index * 100,
                )
                continue
            expected = token_row + index * 1000
            assert saved["layers"][label].item() == expected


def test_dcp_pack_uses_request_index_not_token_row(
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
        layers=(3, 4),
        token_id=198,
        position=1512,
        first_attention_boundary=True,
        attention_boundary_layer=3,
        full_attention_boundaries=True,
        full_attention_stages=("q", "dcp_output_pack", "dcp_lse_pack", "core"),
        dcp_request_tail_rows=4,
    )
    labels = trace.output_labels()
    hidden_states = []
    for index, label in enumerate(labels):
        if "dcp_" in label:
            hidden_states.append(torch.arange(4).unsqueeze(1) + index * 100)
        else:
            hidden_states.append(torch.arange(50).unsqueeze(1) + index * 1000)
    input_ids = torch.zeros(50, dtype=torch.long)
    positions = torch.zeros(50, dtype=torch.long)
    input_ids[[24, 49]] = 198
    positions[[24, 49]] = 1512

    trace.maybe_save(
        input_ids=input_ids,
        positions=positions,
        query_len=50,
        cudagraph_mode="NONE",
        aux_hidden_states=hidden_states,
        req_ids=["request-a", "request-b"],
        query_start_loc=[0, 25, 50],
    )

    for occurrence, token_row in enumerate((24, 49)):
        saved = torch.load(
            tmp_path / f"trace.q50.occ{occurrence}.rank0.pt",
            weights_only=False,
        )
        for index, label in enumerate(labels):
            if label.startswith("full_q."):
                assert torch.equal(
                    saved["layers"][label],
                    torch.arange(50)[occurrence * 25 : (occurrence + 1) * 25].unsqueeze(
                        1
                    )
                    + index * 1000,
                )
                continue
            expected = (
                occurrence + index * 100
                if "dcp_" in label
                else token_row + index * 1000
            )
            assert saved["layers"][label].item() == expected
