# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.expert_offload_e8_archive import (
    canonical_expert,
    owner_span,
    unpack_quip_e8,
)
from vllm.utils.e8_expert_config import E8ArchiveConfig
from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry
from vllm.v1.core.elastic_expert import NativeExpertBudget


def _quip_pack(indices: torch.Tensor) -> torch.Tensor:
    m, n = indices.shape
    indices = indices.view(m // 2, 2, (n * 8) // 16, 2).transpose(1, 2).contiguous()
    abs32 = (
        (indices[:, :, 0, 0] >> 8)
        + ((indices[:, :, 1, 0] >> 8) << 8)
        + ((indices[:, :, 0, 1] >> 8) << 16)
        + ((indices[:, :, 1, 1] >> 8) << 24)
    )
    sign32 = torch.zeros_like(abs32)
    for i in range(4):
        value = indices[:, :, i % 2, i // 2]
        for j in range(8):
            sign32 += ((value >> j) & 1) << (4 * j + i)
    return (
        ((sign32 << 32) + abs32)
        .reshape(m // 16, 8, n // 8, 4)
        .transpose(1, 2)
        .contiguous()
        .view(m, n // 4)
    )


@pytest.mark.parametrize("shape", [(640, 320), (2560, 80)])
def test_quip_canonicalization_is_lossless(shape):
    generator = torch.Generator().manual_seed(20260917)
    direct = torch.randint(0, 65536, shape, dtype=torch.int64, generator=generator)
    packed = _quip_pack(direct)
    recovered = unpack_quip_e8(packed.numpy(), shape[0], shape[1] * 8)
    np.testing.assert_array_equal(recovered, direct.numpy().astype(np.uint16))


def test_direct_record_preserves_shared_transform_and_exact_shapes():
    record = {
        "q_gu": np.zeros((1280, 320), np.uint16),
        "q_down": np.zeros((2560, 80), np.uint16),
        "su_gu": np.ones(2560, np.float16),
        "sv_gate": np.ones(640, np.float16),
        "sv_up": np.ones(640, np.float16),
        "su_down": np.ones(640, np.float16),
        "sv_down": np.ones(2560, np.float16),
    }
    result = canonical_expert(record)
    assert sum(value.nbytes for value in result.values()) == 1_242_880
    with pytest.raises(ValueError, match="invalid direct"):
        canonical_expert({**record, "q_down": np.zeros((1, 1), np.uint16)})


def test_owner_topology_and_config_are_fail_closed(tmp_path):
    assert [owner_span(rank) for rank in range(3)] == [(0, 205), (205, 410), (410, 512)]
    config = E8ArchiveConfig(str(tmp_path.resolve()), 0.35, 1024)
    assert config.max_tokens == 1024
    for fraction in (-0.1, 1.0):
        with pytest.raises(ValueError, match="invalid E8"):
            E8ArchiveConfig(str(tmp_path.resolve()), fraction, 1024)


def test_demand_library_identity_and_threshold_are_fail_closed(tmp_path):
    library = tmp_path / "libdemand.so"
    library.write_bytes(b"demand-loader")
    import hashlib

    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    config = E8ArchiveConfig(
        str(tmp_path.resolve()),
        demand_library=str(library.resolve()),
        demand_library_sha256=digest,
        demand_max_tokens=32,
    )
    assert config.demand_max_tokens == 32
    with pytest.raises(ValueError, match="paired"):
        E8ArchiveConfig(str(tmp_path.resolve()), demand_library=str(library.resolve()))
    with pytest.raises(ValueError, match="identity"):
        E8ArchiveConfig(
            str(tmp_path.resolve()),
            demand_library=str(library.resolve()),
            demand_library_sha256="0" * 64,
        )


def test_resident_layout_identity_is_fail_closed(tmp_path):
    import hashlib

    layout = tmp_path / "layout.json"
    layout.write_text("{}")
    digest = hashlib.sha256(layout.read_bytes()).hexdigest()
    config = E8ArchiveConfig(
        str(tmp_path.resolve()),
        resident_layout=str(layout.resolve()),
        resident_layout_sha256=digest,
    )
    assert config.resident_layout == str(layout.resolve())
    with pytest.raises(ValueError, match="paired"):
        E8ArchiveConfig(str(tmp_path.resolve()), resident_layout=str(layout.resolve()))
    with pytest.raises(ValueError, match="identity"):
        E8ArchiveConfig(
            str(tmp_path.resolve()),
            resident_layout=str(layout.resolve()),
            resident_layout_sha256="0" * 64,
        )


def test_balanced_partition_is_admitted_only_for_exclusive_e8(tmp_path):
    geometry = NVFP4ExpertGeometry(2560, 640, 3)
    common = dict(
        geometry=geometry,
        layers=48,
        experts=512,
        max_hot_rows=0,
        staging=32,
        ram_cache_bytes=0,
        prepared_archive=str(tmp_path.resolve()),
        partition="balanced",
    )
    budget = NativeExpertBudget(
        **common,
        e8_archive=E8ArchiveConfig(str(tmp_path.resolve()), 0.35, 1024),
    )
    assert budget.partition == "balanced"
    assert [budget.rank_geometry(rank).width for rank in range(3)] == [256, 192, 192]
    assert [budget.rank_geometry(rank).tp for rank in range(3)] == [1, 1, 1]
    with pytest.raises(ValueError, match="balanced partition"):
        NativeExpertBudget(**common)


def test_capture_epochs_close_between_forwards_and_abort_incomplete_state(tmp_path):
    from vllm.models.qwen4_exp.nvidia.expert_offload_e8 import E8ExpertExecutor

    executor = E8ExpertExecutor.__new__(E8ExpertExecutor)
    executor.prepared = False
    executor.expected_layer = 0
    executor.trace_path = ""
    executor.trace_active = False
    executor.trace_identity = None
    executor.trace_steps = 0
    executor.active_tokens = None
    executor.physical_tokens = None
    executor.demand_loader = None
    executor.config = E8ArchiveConfig(str(tmp_path.resolve()))
    executor.demand_mode = False

    for _ in range(2):
        executor.prepare_execution(dummy=True)
        executor.expected_layer = 48
        executor.finish_execution()
        assert not executor.prepared

    executor.prepare_execution(dummy=True)
    executor.expected_layer = 7
    executor.abort_execution()
    assert not executor.prepared
    assert executor.expected_layer == 0


def test_demand_mode_is_bound_to_physical_graph_shape(tmp_path):
    from vllm.models.qwen4_exp.nvidia.expert_offload_e8 import E8ExpertExecutor

    executor = E8ExpertExecutor.__new__(E8ExpertExecutor)
    executor.prepared = False
    executor.expected_layer = 0
    executor.trace_path = ""
    executor.trace_active = False
    executor.trace_identity = None
    executor.trace_steps = 0
    executor.active_tokens = None
    executor.physical_tokens = None
    executor.demand_loader = object()
    executor.config = E8ArchiveConfig(str(tmp_path.resolve()), demand_max_tokens=32)
    executor.demand_mode = False
    executor._stage = lambda layer, demand=False: None

    executor.prepare_execution(dummy=True, num_tokens=32)
    assert executor.demand_mode
    executor.abort_execution()
    executor.prepare_execution(
        dummy=False, identity={"tokens": 1, "physical_tokens": 64}
    )
    assert not executor.demand_mode


def test_row_tile_preserves_decode_and_specializes_large_prefill():
    from vllm.models.qwen4_exp.nvidia.expert_offload_e8 import E8ExpertExecutor

    assert [E8ExpertExecutor._row_tile(tokens) for tokens in (1, 16, 31)] == [
        8,
        8,
        8,
    ]
    assert [E8ExpertExecutor._row_tile(tokens) for tokens in (32, 64, 127)] == [
        32,
        32,
        32,
    ]
    assert [E8ExpertExecutor._row_tile(tokens) for tokens in (128, 323, 511)] == [
        64,
        64,
        64,
    ]
    assert [E8ExpertExecutor._row_tile(tokens) for tokens in (512, 646, 1024)] == [
        128,
        128,
        128,
    ]


def test_warmup_routes_match_weighted_tp3_ownership_and_are_dense():
    from vllm.models.qwen4_exp.nvidia.expert_offload_e8 import E8ExpertExecutor

    spans = [(0, 205), (205, 410), (410, 512)]
    patterns = [
        E8ExpertExecutor._warmup_route_pattern(begin, end) for begin, end in spans
    ]
    assert [
        sum(begin <= expert < end for expert in pattern)
        for pattern, (begin, end) in zip(patterns, spans, strict=True)
    ] == [4, 4, 2]
    assert all(len(pattern) == len(set(pattern)) == 10 for pattern in patterns)


def test_warmup_surface_covers_bucket_and_row_tile_boundaries():
    from vllm.models.qwen4_exp.nvidia.expert_offload_e8 import E8ExpertExecutor

    surface = E8ExpertExecutor._warmup_token_surface(1024)
    assert surface == (1, 2, 4, 8, 16, 31, 32, 64, 127, 128, 256, 511, 512, 1024)
    assert E8ExpertExecutor._warmup_token_surface(4) == (1, 2, 4)
    with pytest.raises(ValueError, match="token capacity"):
        E8ExpertExecutor._warmup_token_surface(0)


def test_only_temporal_m4_is_admitted_during_stream_capture(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.models.qwen4_exp.nvidia.expert_offload_provider import (
        NativeExpertProvider,
    )

    provider = NativeExpertProvider.__new__(NativeExpertProvider)
    provider.e8_path = SimpleNamespace(
        temporal_cache=True, demand_mode=True, run=Mock(), last_step={}
    )
    provider.active = provider.dummy = False
    provider.bank = SimpleNamespace(state="READY")
    provider.last_step = {}
    provider.forward_count = 0
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def inputs(tokens):
        hidden = torch.empty(tokens, 2560)
        return (
            hidden,
            torch.empty(tokens, 10),
            torch.empty(tokens, 10, dtype=torch.int32),
            torch.empty_like(hidden),
        )

    provider.run(0, *inputs(4))
    provider.e8_path.run.assert_called_once()
    assert provider.forward_count == 1 and not provider.active
    with pytest.raises(RuntimeError, match="breakable Graph"):
        provider.run(0, *inputs(8))
    provider.e8_path.temporal_cache = False
    with pytest.raises(RuntimeError, match="breakable Graph"):
        provider.run(0, *inputs(4))
