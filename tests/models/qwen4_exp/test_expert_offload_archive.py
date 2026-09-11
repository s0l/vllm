# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless disk rows, atomic publication and borrowed-staging lifetime."""

import json
import os

import numpy as np
import pytest

from vllm.models.qwen4_exp.nvidia.expert_offload_archive import build_archive
from vllm.models.qwen4_exp.nvidia.expert_offload_source import NativeExpertStore
from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry

from .test_expert_offload_source import write_source


def source(root, tp=3, rank=0, **kwargs):
    return NativeExpertStore(
        root,
        rank,
        0,
        layers=2,
        experts=3,
        geometry=NVFP4ExpertGeometry(32, 16, tp=tp, local_alignment=16),
        **kwargs,
    )


@pytest.mark.parametrize("tp", [1, 2, 3, 5])
def test_archive_exact_all_ranks_read_into_and_immutable_heap_views(tmp_path, tp):
    write_source(tmp_path)
    raw = source(tmp_path, tp)
    archive = tmp_path / "prepared"
    build_archive(raw, archive, min_free_bytes=0)
    raw.close()
    for rank in range(tp):
        raw = source(tmp_path, tp, rank)
        prepared = source(tmp_path, tp, rank, archive_path=archive)
        expected = raw.get_many(1, (2, 0, 1))
        actual = prepared.get_many(1, (2, 0, 1))
        assert all(
            np.array_equal(a[n], b[n]) for a, b in zip(actual, expected) for n in a
        )
        assert all(not a.flags.writeable for row in actual for a in row.values())
        arrays = {
            name: np.zeros((3, *a.shape), dtype=a.dtype)
            for name, a in expected[0].items()
        }
        into = prepared.get_many_into(1, (2, 0, 1), arrays, (2, 0, 1))
        for row, ref, position in zip(into, expected, (2, 0, 1)):
            for name in row:
                assert np.array_equal(row[name], ref[name])
                assert (
                    row[name].ctypes.data
                    == arrays[name][position : position + 1].ctypes.data
                )
                assert not row[name].flags.writeable
        arrays["w13_weight"].fill(17)
        assert all((row["w13_weight"] == 17).all() for row in into)
        assert all(
            np.array_equal(a[n], b[n]) for a, b in zip(actual, expected) for n in a
        )
        with pytest.raises(ValueError, match="positions"):
            prepared.get_many_into(1, (0, 1), arrays, (0, 0))
        assert not prepared.closed
        raw.close()
        prepared.close()


def test_interrupted_build_has_no_serving_manifest_and_resumes_finished_layers(
    tmp_path,
):
    write_source(tmp_path)
    raw = source(tmp_path)
    archive = tmp_path / "prepared"

    def cancel(progress):
        raise InterruptedError("owner cancellation")

    with pytest.raises(InterruptedError):
        build_archive(raw, archive, min_free_bytes=0, progress=cancel)
    assert not (archive / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        source(tmp_path, archive_path=archive)
    old = {p.name: p.stat().st_mtime_ns for p in archive.glob("*.bin")}
    progress: list[dict] = []
    build_archive(raw, archive, min_free_bytes=0, progress=progress.append)
    assert progress[-1]["reused"] == 1
    assert all((archive / n).stat().st_mtime_ns == stamp for n, stamp in old.items())
    manifest_before = (archive / "manifest.json").stat().st_mtime_ns
    build_archive(raw, archive, min_free_bytes=0)
    assert (archive / "manifest.json").stat().st_mtime_ns == manifest_before
    raw.close()


@pytest.mark.parametrize(
    "failure", ["short", "partial_into", "file", "manifest", "source"]
)
def test_bad_archive_read_poison_and_fresh_source_recovery(
    tmp_path, monkeypatch, failure
):
    write_source(tmp_path)
    raw = source(tmp_path)
    archive = tmp_path / "prepared"
    build_archive(raw, archive, min_free_bytes=0)
    raw.close()
    prepared = source(tmp_path, archive_path=archive)
    sample = prepared.get(0, 0)
    if failure == "short":
        original = os.pread
        monkeypatch.setattr(
            os, "pread", lambda fd, size, offset: original(fd, size - 1, offset)
        )
    elif failure == "partial_into":
        original_vector = os.preadv
        monkeypatch.setattr(
            os,
            "preadv",
            lambda fd, buffers, offset: original_vector(fd, buffers[:1], offset),
        )
    else:
        path = (
            archive / "layer-000-rank-000.bin"
            if failure == "file"
            else archive / "manifest.json"
            if failure == "manifest"
            else tmp_path / "weights.safetensors"
        )
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
    with pytest.raises(OSError):
        if failure == "partial_into":
            arrays = {
                name: np.empty((3, *a.shape), dtype=a.dtype)
                for name, a in sample.items()
            }
            prepared.get_many_into(0, (0, 1, 2), arrays, (0, 1, 2))
        else:
            prepared.get(0, 0)
    assert prepared.closed and not prepared._fds and prepared._reader is None
    monkeypatch.undo()
    if failure in ("short", "partial_into"):
        recovered = source(tmp_path, archive_path=archive)
        assert all(np.array_equal(a, recovered.get(0, 0)[n]) for n, a in sample.items())
        recovered.close()


def test_schema_and_source_mismatch_rejected_before_data_read(tmp_path):
    write_source(tmp_path)
    raw = source(tmp_path)
    archive = tmp_path / "prepared"
    build_archive(raw, archive, min_free_bytes=0)
    raw.close()
    with pytest.raises(ValueError, match="geometry mismatch"):
        source(tmp_path, tp=2, archive_path=archive)
    path = archive / "manifest.json"
    meta = json.loads(path.read_text())
    meta["identity"]["schema"]["a2_gscale"]["offset"] += 4
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="geometry mismatch"):
        source(tmp_path, archive_path=archive)
