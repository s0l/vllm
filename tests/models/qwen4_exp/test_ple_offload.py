# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source lifetime and allocation bounds, before any model/GPU construction."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm.models.qwen4_exp.nvidia.ple_offload import PleRowStore

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."


def make_source(path, *, omit=False, bad_scale=False):
    raw = torch.arange(32 * 4, dtype=torch.uint8).reshape(32, 4) % 100
    tensors = {
        PREFIX + "shard_0.weight": raw[:16].view(torch.float8_e4m3fn),
        PREFIX + "weight_scale": torch.tensor(float("nan") if bad_scale else 0.125),
    }
    if not omit:
        tensors[PREFIX + "shard_1.weight"] = raw[16:].view(torch.float8_e4m3fn)
    save_file(tensors, str(path / "table.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "table.safetensors" for k in tensors}})
    )
    return raw.numpy()


def store(path):
    return PleRowStore(path, 1, 32, 4, 2, cache_bytes=32, max_ids=64)


def test_row_order_duplicates_bounds_and_recovery(tmp_path):
    raw = make_source(tmp_path)
    table = store(tmp_path)
    ids = np.asarray([31, 0, 15, 16, 0, 31, 8], dtype=np.int64)
    output = np.empty((len(ids), 4), dtype=np.uint8)
    table.gather_into(ids, output)
    assert np.array_equal(output, raw[ids])
    reads = table.read_rows
    table.gather_into(ids, output)
    assert table.read_rows == reads
    assert table.peak_cache_bytes <= 32
    for invalid in [np.asarray([-1]), np.asarray([32]), np.arange(65) % 32]:
        destination = np.full((len(invalid), 4), 231, dtype=np.uint8)
        with pytest.raises((ValueError, IndexError)):
            table.gather_into(invalid, destination)
        assert (destination == 231).all()
    table.gather_into(ids, output)
    assert np.array_equal(output, raw[ids])
    table.close()
    table.close()
    with pytest.raises(RuntimeError):
        table.gather_into(ids, output)


def test_cached_rows_cannot_hide_replaced_source(tmp_path):
    make_source(tmp_path)
    table = store(tmp_path)
    ids = np.asarray([0, 31], dtype=np.int64)
    output = np.zeros((2, 4), dtype=np.uint8)
    table.gather_into(ids, output)
    # Replace only the selected index; existing cached row bytes still match.
    index = tmp_path / "model.safetensors.index.json"
    saved = index.read_text()
    replacement = tmp_path / "replacement"
    replacement.write_text(saved)
    replacement.replace(index)
    output.fill(231)
    with pytest.raises(OSError, match="identity changed"):
        table.gather_into(ids, output)
    assert (output == 231).all() and table.closed
    fresh = store(tmp_path)
    fresh.gather_into(ids, output)
    assert not (output == 231).all()
    fresh.close()


@pytest.mark.parametrize("kwargs", [{"omit": True}, {"bad_scale": True}])
def test_incomplete_or_invalid_source_never_opens_table(tmp_path, kwargs):
    make_source(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        store(tmp_path)


def test_parent_reload_rejects_before_changing_any_model_weight():
    from vllm.models.qwen4_exp.nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpModel,
    )
    from vllm.models.qwen4_exp.nvidia.ple_offload import MmapPLEEmbedding

    embedding = MmapPLEEmbedding.__new__(MmapPLEEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.loaded = True
    model = torch.nn.Sequential(embedding)
    consumed = []

    def incoming():
        consumed.append(True)
        yield "would_mutate_dense_weight", torch.tensor(1)

    for model_type in [
        Qwen4ExpModel,
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    ]:
        with pytest.raises(RuntimeError, match="new worker"):
            model_type.load_weights(model, incoming())
        assert not consumed
