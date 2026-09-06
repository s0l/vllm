# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical price identity is independent of placement and executable epochs."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import metadata, util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PRICE_IDENTITY_SCHEMA = "elastic-price-v1"
PRICE_OWNER_GENERATION = "elastic-price-owner-v1"
PLACEMENT_FACTORS = frozenset(
    {"num_blocks", "rank_primary", "rank_gdn", "gdn_initial_blocks"}
)


def flashinfer_price_tactics(vllm_config: Any) -> dict[str, Any]:
    """Bind selected tactics by contents, never by the cache's location."""
    if not getattr(
        getattr(vllm_config, "kernel_config", None),
        "enable_flashinfer_autotune",
        False,
    ):
        return {"mode": "default"}
    from vllm.model_executor.warmup.flashinfer_autotune_cache import (
        resolve_flashinfer_autotune_file,
    )

    path = resolve_flashinfer_autotune_file(
        SimpleNamespace(vllm_config=vllm_config), create_parent=False
    )
    try:
        payload = json.loads(path.read_bytes())
    except FileNotFoundError:
        return {"mode": "autotune-cache-absent"}
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    return {"mode": "autotune-cache", "sha256": hashlib.sha256(encoded).hexdigest()}


@lru_cache(maxsize=1)
def native_price_provenance() -> dict[str, Any]:
    """Bind installed native dependencies once, outside the serving hot path."""
    result: dict[str, Any] = {}
    for module in ("torch._C",):
        spec = util.find_spec(module)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"missing native price dependency: {module}")
        with open(spec.origin, "rb") as stream:
            result[module] = hashlib.file_digest(stream, "sha256").hexdigest()
    vllm_dist = metadata.distribution("vllm")
    native_files = [
        entry
        for entry in vllm_dist.files or ()
        if str(entry).startswith("vllm/")
        and str(entry).endswith(".so")
        and not Path(entry).name.startswith(("_rust_tool_parser.", "fs_io_C."))
    ]
    if not native_files:
        raise RuntimeError("missing vLLM native artifact inventory")
    for entry in native_files:
        with open(vllm_dist.locate_file(entry), "rb") as stream:
            result[str(entry)] = hashlib.file_digest(stream, "sha256").hexdigest()
    # RECORD is dependency provenance, not a claim to hash JIT-selected cubins.
    # Selected producer source/config remain separate price dependencies.
    for package in (
        "torch",
        "triton",
        "flashinfer-python",
        "flashinfer-jit-cache",
        "nvidia-nccl-cu13",
        "nvidia-nccl-cu12",
    ):
        try:
            dist = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            result[package] = None
            continue
        record = dist.read_text("RECORD")
        result[package] = {
            "version": dist.version,
            "record": None
            if record is None
            else hashlib.sha256(record.encode()).hexdigest(),
        }
        # These libraries can change allocation/collective capture behavior
        # without changing the Python extension or package version.
        for entry in dist.files or ():
            if Path(entry).name in {
                "libtorch_cuda.so",
                "libc10_cuda.so",
                "libnccl.so.2",
            }:
                with open(dist.locate_file(entry), "rb") as stream:
                    result[f"{package}/{entry}"] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
    # PyTorch may link system CUDA/NCCL instead of nvidia-* wheels. Inspect
    # loaded objects, not a guessed installation directory or package version.
    loaded = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split()) >= 6
        and line.split()[-1].startswith("/")
        and Path(line.split()[-1]).name.startswith(
            ("libnccl.so", "libcudart.so", "libcublas.so", "libcublasLt.so")
        )
    }
    for path in sorted(loaded):
        with open(path, "rb") as stream:
            result[f"loaded/{Path(path).name}"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    driver = Path("/proc/driver/nvidia/version")
    result["driver"] = driver.read_text() if driver.is_file() else None
    result["pci_topology"] = sorted(
        str(device.resolve())
        for device in Path("/sys/bus/pci/devices").glob("*")
        if (device / "vendor").read_text().strip() == "0x10de"
        and (device / "class").read_text().strip().startswith("0x03")
    )
    return result


def source_semantic_digest(source: str) -> str:
    """Ignore formatting/comments, never infer executable-code equivalence."""
    tree = ast.parse(source)
    # Retain docstrings: Python code can consume __doc__. Ignoring arbitrary
    # logging expressions would likewise hide argument evaluation side effects.
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def price_identity(factors: Mapping[str, Any]) -> dict[str, Any]:
    physical = {
        key: value for key, value in factors.items() if key not in PLACEMENT_FACTORS
    }
    payload = {"schema": PRICE_IDENTITY_SCHEMA, "factors": physical}
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    # Compare the same JSON representation we hash and persist. Dataclass
    # policies contain tuples, which become lists on disk; retaining raw
    # factors also lets caller mutation invalidate an already-computed digest.
    canonical_payload = json.loads(encoded)
    return {
        **canonical_payload,
        "fingerprint": hashlib.sha256(encoded).hexdigest()[:16],
    }


def validate_price_identity(payload: Any) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "factors", "fingerprint"}
        or payload["schema"] != PRICE_IDENTITY_SCHEMA
        or not isinstance(payload["factors"], dict)
        or PLACEMENT_FACTORS.intersection(payload["factors"])
        or payload != price_identity(payload["factors"])
    ):
        raise RuntimeError("invalid elastic price identity manifest")
    return payload


def price_identity_differences(source: Any, destination: Any) -> tuple[str, ...]:
    left = validate_price_identity(source)["factors"]
    right = validate_price_identity(destination)["factors"]
    return tuple(
        key
        for key in sorted(left.keys() | right.keys())
        if key not in left or key not in right or left[key] != right[key]
    )


def select_compatible_price_seed(rows, coverage, surface):
    """Select already validated rows without widening their semantic evidence."""
    old_aliases = {
        tuple(item["semantic_step_key"]): tuple(item["physical_step_key"])
        for item in coverage.get("surface_full_aliases", ())
    }
    new_aliases = dict(surface.full_aliases)
    old_tokens = {
        tuple(item["step_key"]): item["live_num_tokens"]
        for item in coverage.get("semantic_token_witnesses", ())
    }
    new_tokens = dict(surface.semantic_token_witnesses)
    reusable = {
        key: dict(row)
        for key, row in rows.items()
        if key in surface.required
        and old_aliases.get(key, key) == new_aliases.get(key, key)
        and old_tokens.get(key) == new_tokens.get(key)
    }
    witnesses = {
        (tuple(item["step_key"]), item["query_len"])
        for item in coverage.get("mixed_query_witnesses", ())
        if tuple(item["step_key"]) in reusable
    }.intersection(surface.mixed_query_witnesses)
    return reusable, witnesses


def validate_calibration_request(
    payload: Any, *, fingerprint: str, surface_sha256: str
) -> None:
    """A missing filename or maintenance flag is not a measurement decision."""
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"schema", "fingerprint", "surface_sha256", "reason", "evidence"}
        or payload.get("schema") != "elastic-measurement-request-v1"
        or payload.get("fingerprint") != fingerprint
        or payload.get("surface_sha256") != surface_sha256
        or payload.get("reason")
        not in {"initial-measurement", "changed-producer", "missing-coverage"}
        or not isinstance(payload.get("evidence"), str)
        or not payload["evidence"].strip()
    ):
        raise RuntimeError(
            "calibration requires an explicit identity-bound measurement request; "
            "unknown compatibility is not a calibration reason"
        )


def remap_catalog_resident_keys(
    rows: Mapping[tuple[int, ...], Mapping[str, Any]],
    *,
    source_generation: str,
    destination_generation: str,
    max_num_batched_tokens: int,
    compiled_piecewise_sizes: tuple[int, ...],
    policy: Any,
) -> dict[tuple[int, ...], dict[str, Any]]:
    """Translate every recorded owner by descriptor, never by position/bytes."""
    from vllm.v1.core.elastic_graph import (
        RuntimeGeneration,
        resolve_step_physical_keys,
    )

    if not all(
        isinstance(value, str) and value
        for value in (source_generation, destination_generation)
    ):
        raise RuntimeError("catalog owner binding requires both generations")
    mapping: dict[str, str] = {}
    for step_key in rows:
        left = resolve_step_physical_keys(
            step_key,
            RuntimeGeneration(source_generation),
            max_num_batched_tokens,
            compiled_piecewise_sizes,
            policy,
        )
        right = resolve_step_physical_keys(
            step_key,
            RuntimeGeneration(destination_generation),
            max_num_batched_tokens,
            compiled_piecewise_sizes,
            policy,
        )
        if len(left) != len(right):
            raise RuntimeError("catalog owner binding changed physical inventory")
        for before, after in zip(left, right):
            prior = mapping.setdefault(before.identity, after.identity)
            if prior != after.identity:
                raise RuntimeError("ambiguous catalog owner binding")
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("catalog owner binding is not one-to-one")
    result = {}
    for step_key, row in rows.items():
        copied = dict(row)
        raw = row.get("resident_key_bytes")
        if raw is not None:
            translated = {}
            for identity, size in raw:
                if (
                    identity not in mapping
                    or identity in translated
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                ):
                    raise RuntimeError("unknown or invalid catalog resident owner")
                translated[identity] = size
            copied["resident_key_bytes"] = tuple(
                sorted(
                    (mapping[identity], size) for identity, size in translated.items()
                )
            )
        result[step_key] = copied
    return result
