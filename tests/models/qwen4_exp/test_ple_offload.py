# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source lifetime and allocation bounds, before any model/GPU construction."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

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


def make_loader_owner(path):
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
    from vllm.models.qwen4_exp.nvidia.ple_offload import MmapPLEEmbedding

    owner = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(owner)
    owner._mmap_seen = set()
    for name in (
        "layer_multipliers",
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
    ):
        owner.register_buffer(name, torch.zeros(2, dtype=torch.int64))
    owner.ngram_embedding = MmapPLEEmbedding(
        32, 4, PREFIX.removesuffix(".ngram_embedding."), path, 2, 32, 64
    )
    hashes = [(name, torch.tensor([7, 13])) for name, _ in owner.named_buffers()]
    hashes = [(name, tensor) for name, tensor in hashes if "." not in name]
    rows = [
        ("ngram_embedding." + name.removeprefix(PREFIX), tensor)
        for name, tensor in load_file(str(path / "table.safetensors")).items()
    ]
    return owner, hashes, rows


def test_whole_model_hook_seals_ple_after_interleaved_shard_visits(tmp_path):
    from vllm.models.qwen4_exp.nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )

    make_source(tmp_path)
    owner, hashes, rows = make_loader_owner(tmp_path)
    target = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    torch.nn.Module.__init__(target)
    target.model = torch.nn.Module()
    target.model.ple = owner
    target.model.register_parameter("marker", torch.nn.Parameter(torch.zeros(1)))
    # AutoWeightsLoader groups consecutive prefixes, not all tensors of a child.
    stream = [("model.ple." + name, tensor) for name, tensor in hashes]
    stream += [("model.marker", torch.tensor([9.0]))]
    stream += [("model.ple." + name, tensor) for name, tensor in rows]
    loaded = target.load_weights(iter(stream))
    assert len(loaded) == len(stream)
    assert target.model.marker.item() == 9.0
    assert not owner.ngram_embedding.loaded
    wrapper = torch.nn.Module()
    wrapper.language_model = target
    Qwen4ExpForConditionalGeneration.process_weights_after_loading(wrapper)
    assert owner.ngram_embedding.loaded and owner.ngram_embedding.store.read_rows == 0
    assert torch.equal(owner.layer_multipliers, torch.tensor([7, 13]))
    with pytest.raises(RuntimeError, match="new worker"):
        target.load_weights([("model.marker", torch.tensor([11.0]))])
    assert target.model.marker.item() == 9.0
    owner.ngram_embedding.store.close()


@pytest.mark.parametrize("failure", ["hash", "shard", "duplicate", "source"])
def test_partial_ple_load_cannot_be_promoted_and_new_owner_recovers(tmp_path, failure):
    make_source(tmp_path)
    owner, hashes, rows = make_loader_owner(tmp_path)
    if failure == "hash":
        hashes = hashes[:-1]
    if failure == "shard":
        rows = rows[:-1]
    owner.load_weights(hashes)
    owner.load_weights(rows)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        if failure == "duplicate":
            owner.load_weights(hashes[:1])
        else:
            if failure == "source":
                index = tmp_path / "model.safetensors.index.json"
                replacement = tmp_path / "replacement"
                replacement.write_bytes(index.read_bytes())
                replacement.replace(index)
            owner.finish_mmap_load()
    assert not owner.ngram_embedding.loaded and owner.ngram_embedding.store.closed
    recovered, hashes, rows = make_loader_owner(tmp_path)
    recovered.load_weights(hashes)
    recovered.load_weights(rows)
    recovered.finish_mmap_load()
    assert recovered.ngram_embedding.loaded
    recovered.ngram_embedding.store.close()
