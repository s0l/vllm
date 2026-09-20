# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable, resumable lossless CUTLASS rows; never a replacement checkpoint."""

import fcntl
import json
import math
import os
import shutil
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from vllm.utils.nvfp4_expert_geometry import NVFP4ExpertGeometry

from .ple_offload import fingerprint

FORMAT = "nvfp4-cutlass-expert-row-v1"


def row_schema(geometry):
    h, n = geometry.hidden, geometry.local
    shapes = {
        "w13_weight": (2 * n, h // 2),
        "w2_weight": (h, n // 2),
        "w13_weight_scale": ((2 * n + 127) // 128 * 128, (h // 16 + 3) // 4 * 4),
        "w2_weight_scale": ((h + 127) // 128 * 128, (n // 16 + 3) // 4 * 4),
        "w13_weight_scale_2": (),
        "w2_weight_scale_2": (),
        "w13_input_scale": (),
        "w2_input_scale": (),
        "a1_gscale": (),
        "a2_gscale": (),
    }
    fields, offset = {}, 0
    for name, shape in shapes.items():
        dtype = "<f4" if not shape else "|u1"
        size = math.prod(shape) * np.dtype(dtype).itemsize
        fields[name] = dict(shape=list(shape), dtype=dtype, offset=offset, nbytes=size)
        offset += size
    return fields, offset, (offset + 4095) // 4096 * 4096


def source_identity(source):
    fields, size, stride = row_schema(source.geometry)
    identity = dict(
        format=FORMAT,
        geometry=asdict(source.geometry),
        layers=source.layers,
        experts=source.experts,
        source_files={p: list(v) for p, v in source.files.items()},
        schema=fields,
        row_bytes=size,
        stride=stride,
    )
    if getattr(source, "partition", "legacy") == "balanced":
        identity.update(
            format="nvfp4-cutlass-expert-row-v2-balanced",
            spans=[list(span) for span in source.spans],
            rank_schemas=[
                dict(
                    zip(
                        ("schema", "row_bytes", "stride"),
                        row_schema(NVFP4ExpertGeometry(source.hidden, end - start, 1)),
                    )
                )
                for start, end in source.spans
            ],
        )
    return identity


def rank_schema(identity, rank):
    return identity["rank_schemas"][rank] if "rank_schemas" in identity else identity


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked_file(root, entry, expected_size):
    name = entry["file"]
    if type(name) is not str or Path(name).name != name:
        raise ValueError("prepared archive file must belong to its directory")
    path = root / name
    actual = fingerprint(path)
    if list(actual) != entry["identity"] or actual[2] != expected_size:
        raise OSError("prepared archive file identity changed")
    return str(path), actual


class PreparedExpertArchive:
    def __init__(self, root, source):
        self.root = Path(root).resolve(strict=True)
        manifest = self.root / "manifest.json"
        before = fingerprint(manifest)
        meta = json.loads(manifest.read_text())
        if (
            meta.get("identity") != source_identity(source)
            or meta.get("status") != "READY"
            or fingerprint(manifest) != before
        ):
            raise ValueError("prepared archive source/format/geometry mismatch")
        schema = rank_schema(meta["identity"], source.rank)
        self.fields = schema["schema"]
        self.row_bytes, self.stride = schema["row_bytes"], schema["stride"]
        expected_keys = {
            f"{layer}:{rank}"
            for layer in range(source.layers)
            for rank in range(source.tp)
        }
        if set(meta["files"]) != expected_keys:
            raise ValueError("incomplete prepared archive inventory")
        self.paths, self.files = {}, {str(manifest): before}
        for layer in range(source.layers):
            path, identity = checked_file(
                self.root,
                meta["files"][f"{layer}:{source.rank}"],
                source.experts * self.stride,
            )
            self.paths[layer] = path
            self.files[path] = identity

    def read(self, key, handle, buffers=None):
        _, expert = key
        if buffers is None:
            blob = os.pread(handle.fileno(), self.row_bytes, expert * self.stride)
            if len(blob) != self.row_bytes:
                raise OSError("truncated prepared expert row")
            values = {
                name: np.frombuffer(
                    blob,
                    dtype=field["dtype"],
                    count=math.prod(field["shape"]),
                    offset=field["offset"],
                ).reshape(field["shape"])
                for name, field in self.fields.items()
            }
        else:
            vectors = [memoryview(buffers[name]).cast("B") for name in self.fields]
            count = os.preadv(handle.fileno(), vectors, expert * self.stride)
            if count != self.row_bytes:
                raise OSError("partial prepared read into staging")
            values = {name: value.view() for name, value in buffers.items()}
            for value in values.values():
                value.setflags(write=False)
        return values, self.row_bytes, 1

    def validate_buffers(self, buffers):
        if set(buffers) != set(self.fields):
            raise ValueError("incomplete prepared destination")
        spans = []
        for name, field in self.fields.items():
            value = buffers[name]
            if (
                not isinstance(value, np.ndarray)
                or value.shape != tuple(field["shape"])
                or value.dtype.str != field["dtype"]
                or not value.flags.c_contiguous
                or not value.flags.writeable
            ):
                raise ValueError("invalid prepared destination layout")
            spans.append((value.ctypes.data, value.ctypes.data + value.nbytes))
        return spans


def build_archive(source, root, *, min_free_bytes=20 << 30, progress=None):
    """Bounded CPU writer. A complete manifest is the only serving admission."""
    from .expert_offload_source import prepare, split_bundle

    if source.closed or source.archive is not None:
        raise ValueError("archive builder requires the original immutable source")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    expected = source_identity(source)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / "build.json"
        if marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise ValueError("archive build identity changed")
        else:
            if any(p.name != ".lock" for p in root.iterdir()):
                raise ValueError("archive directory has no matching build identity")
            atomic_json(marker, expected)
        complete, reused = {}, 0
        for layer in range(source.layers):
            source.check_files(source.files)
            receipt = root / f"layer-{layer:03d}.json"
            if receipt.exists():
                prior = json.loads(receipt.read_text())
                if prior["identity"] != expected or prior["layer"] != layer:
                    raise ValueError("prepared layer receipt identity changed")
                entries = prior["files"]
                if len(entries) != source.tp:
                    raise ValueError("incomplete prepared layer")
                for rank, entry in enumerate(entries):
                    checked_file(
                        root,
                        entry,
                        source.experts * rank_schema(expected, rank)["stride"],
                    )
                reused += 1
            else:
                remaining = (
                    (source.layers - layer)
                    * source.experts
                    * sum(
                        rank_schema(expected, rank)["stride"]
                        for rank in range(source.tp)
                    )
                )
                if shutil.disk_usage(root).free < remaining + min_free_bytes:
                    raise OSError("prepared archive would violate free disk reserve")
                paths = [
                    root / f"layer-{layer:03d}-rank-{r:03d}.bin"
                    for r in range(source.tp)
                ]
                hashes = [sha256() for _ in paths]
                with ExitStack() as stack:
                    handles = [
                        stack.enter_context(p.with_suffix(".partial").open("wb"))
                        for p in paths
                    ]
                    for expert in range(source.experts):
                        for entry in source.entries[layer, expert].values():
                            path = entry[0]
                            if path not in source._fds:
                                source._fds[path] = open(path, "rb", buffering=0)  # noqa: SIM115
                        raw, _, _ = source._read_raw((layer, expert))
                        split = split_bundle(
                            raw,
                            hidden=source.hidden,
                            width=source.width,
                            physical=source.physical,
                            tp=source.tp,
                            spans=getattr(source, "spans", None),
                        )
                        for rank, (handle, digest, bundle) in enumerate(
                            zip(handles, hashes, split)
                        ):
                            schema = rank_schema(expected, rank)
                            prepared = prepare(bundle)
                            row = b"".join(
                                prepared[name].tobytes() for name in schema["schema"]
                            )
                            if len(row) != schema["row_bytes"]:
                                raise ValueError("prepared writer schema mismatch")
                            row += bytes(schema["stride"] - len(row))
                            handle.write(row)
                            digest.update(row)
                    source.check_files(source.files)
                    for handle in handles:
                        handle.flush()
                        os.fsync(handle.fileno())
                entries = []
                for path, digest in zip(paths, hashes):
                    os.replace(path.with_suffix(".partial"), path)
                    entries.append(
                        dict(
                            file=path.name,
                            identity=list(fingerprint(path)),
                            sha256=digest.hexdigest(),
                        )
                    )
                atomic_json(
                    receipt, dict(identity=expected, layer=layer, files=entries)
                )
            complete.update(
                {f"{layer}:{rank}": entry for rank, entry in enumerate(entries)}
            )
            if progress is not None:
                progress(dict(layer=layer, layers=source.layers, reused=reused))
        source.check_files(source.files)
        manifest = dict(status="READY", identity=expected, files=complete)
        final = root / "manifest.json"
        if final.exists():
            if json.loads(final.read_text()) != manifest:
                raise ValueError("published prepared archive differs")
        else:
            atomic_json(final, manifest)
        return manifest
