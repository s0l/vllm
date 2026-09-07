# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content-addressed, data-only profile for the experimental row ABI."""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import torch
from safetensors import safe_open

from .tp3_row_norm import RowNormSpec

RUNTIME_SOURCES = (
    "distributed/device_communicators/tp3_row_profile.py",
    "distributed/device_communicators/tp3_row_ops.py",
    "distributed/device_communicators/tp3_row_norm.py",
    "distributed/device_communicators/tp3_row_norm_kernels.py",
    "distributed/device_communicators/tp3_row_plan.py",
    "distributed/device_communicators/tp3_row_transport.py",
    "distributed/device_communicators/tp3_exact_reduce.py",
    "model_executor/models/qwen3_next.py",
    "model_executor/models/qwen3_5.py",
    "model_executor/models/qwen2_moe.py",
    "model_executor/models/qwen3_next_row.py",
    "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "model_executor/kernels/linear/nvfp4/flashinfer.py",
    "model_executor/kernels/linear/nvfp4/arc.py",
    "model_executor/layers/quantization/utils/nvfp4_utils.py",
    "model_executor/layers/quantization/utils/nvfp4_emulation_utils.py",
    "model_executor/layers/quantization/utils/ag2_nvfp4_arc.py",
    "v1/core/elastic_price_identity.py",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def runtime_source_identity(path):
    from vllm.v1.core.elastic_price_identity import source_semantic_digest

    return source_semantic_digest(Path(path).read_text())


def native_math_identity():
    """Stable installed math artifacts, not process-dependent loaded mappings.

    The separate elastic price identity owns NCCL, driver and topology. Do not
    prime its cached snapshot early during model initialization.
    """
    result = {}
    for package in (
        "torch",
        "triton",
        "flashinfer-python",
        "flashinfer-cubin",
        "flashinfer-jit-cache",
    ):
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            result[package] = None
            continue
        record = distribution.read_text("RECORD")
        result[package] = {
            "version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest()
            if record
            else None,
        }
    distribution = metadata.distribution("vllm")
    files = [
        entry
        for entry in distribution.files or ()
        if str(entry).startswith("vllm/_C") and str(entry).endswith(".so")
    ]
    if not files:
        raise ValueError("row profile requires vLLM native math artifacts")
    result["vllm_native"] = {
        str(entry): sha256(distribution.locate_file(entry)) for entry in files
    }
    return result


def read_profile(path, expected_sha, model_root):
    data = Path(path).read_bytes() if expected_sha else b""
    if not expected_sha or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("row profile content identity mismatch")
    profile = json.loads(data)
    if profile.get("schema") != "ag2-tp3-row-profile-v1":
        raise ValueError("unsupported row profile schema")
    if profile.get("runtime") != {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }:
        raise ValueError("row profile native runtime mismatch")
    if profile.get("native_provenance") != native_math_identity():
        raise ValueError("row profile native artifact mismatch")
    root = Path(model_root)
    for name in ("config.json", "model.safetensors.index.json"):
        if sha256(root / name) != profile["model_files"].get(name):
            raise ValueError("row profile checkpoint identity mismatch")
    package = Path(__file__).parents[2]
    if set(profile["runtime_sources"]) != set(RUNTIME_SOURCES):
        raise ValueError("row profile runtime source coverage mismatch")
    for name, digest in profile["runtime_sources"].items():
        file = (package / name).resolve()
        if (
            not file.is_relative_to(package.resolve())
            or runtime_source_identity(file) != digest
        ):
            raise ValueError("row profile runtime source mismatch")
    required = {"terminal"} | {
        f"{i}.{kind}" for i in range(64) for kind in ("input", "post")
    }
    if set(profile["norms"]) != required:
        raise ValueError("row profile requires all first/layer/terminal boundaries")
    for key, configs in profile["norms"].items():
        role = (
            "terminal"
            if key == "terminal"
            else "first"
            if key == "0.input"
            else "next"
            if key.endswith(".input")
            else "post"
        )
        if len(configs) != 6:
            raise ValueError("row profile requires three destination configurations")
        for r in range(3):
            RowNormSpec(role, configs[2 * r], configs[2 * r + 1])
    return profile


def read_peer_selections(directory):
    """Read selected columns only; never replicate peer expert/ARC weights."""
    result, identities = [], []
    for rank in range(3):
        manifest_path = Path(directory) / f"rank{rank}.manifest.json"
        tensor_path = Path(directory) / f"rank{rank}.safetensors"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("tp_rank") != rank or not manifest.get("ok"):
            raise ValueError("row ARC peer manifest rank/status mismatch")
        if sha256(tensor_path) != manifest.get("sidecar_sha256"):
            raise ValueError("row ARC peer sidecar content mismatch")
        records = {}
        with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
            for record in manifest["records"]:
                suffix = record["runtime_suffix"]
                if suffix in records:
                    raise ValueError("duplicate row ARC peer consumer")
                value = tensors.get_tensor(record["key"] + ".selected")
                count = record["augmented_k"] - record["base_k"]
                if (
                    record["base_k"] != 5120
                    or count not in (64, 256, 512)
                    or value.dtype != torch.int32
                    or value.shape != (count,)
                    or int(value.min()) < 0
                    or int(value.max()) >= 5120
                    or hashlib.sha256(value.numpy().tobytes()).hexdigest()
                    != record["selected_sha256"]
                ):
                    raise ValueError("row ARC peer selection ABI/content mismatch")
                records[suffix] = value.contiguous()
        result.append(records)
        identities.append(manifest["sidecar_sha256"])
    if not (result[0].keys() == result[1].keys() == result[2].keys()):
        raise ValueError("row ARC peers cover different consumers")
    return result, identities
