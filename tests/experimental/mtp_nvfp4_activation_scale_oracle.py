#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sweep ModelOpt NVFP4 activation scales on saved real hidden states.

This is a CPU-only diagnostic.  It replays the exact E2M1 + FP8 block-scale
quantize/dequantize arithmetic used by the W4A4 path without starting model
inference or allocating a CUDA context.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

E2M1_MAX = 6.0
E4M3_MAX = 448.0
BLOCK_SIZE = 16
DEFAULT_FACTORS = (
    0.125,
    0.176776695,
    0.25,
    0.353553391,
    0.5,
    0.707106781,
    1.0,
    1.414213562,
    2.0,
    2.828427125,
    4.0,
    5.656854249,
    8.0,
)


def cast_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    magnitude = torch.abs(x)
    result = torch.zeros_like(magnitude)
    result = torch.where((magnitude > 0.25) & (magnitude < 0.75), 0.5, result)
    result = torch.where((magnitude >= 0.75) & (magnitude <= 1.25), 1.0, result)
    result = torch.where((magnitude > 1.25) & (magnitude < 1.75), 1.5, result)
    result = torch.where((magnitude >= 1.75) & (magnitude <= 2.5), 2.0, result)
    result = torch.where((magnitude > 2.5) & (magnitude < 3.5), 3.0, result)
    result = torch.where((magnitude >= 3.5) & (magnitude <= 5.0), 4.0, result)
    result = torch.where(magnitude > 5.0, 6.0, result)
    return result * sign


def quant_dequant(
    x: torch.Tensor, input_scale: float
) -> tuple[torch.Tensor, dict[str, float]]:
    rows = x.float().reshape(-1, x.shape[-1] // BLOCK_SIZE, BLOCK_SIZE)
    block_amax = rows.abs().amax(dim=-1, keepdim=True)
    input_scale_inv = 1.0 / input_scale
    raw_block_scale = input_scale_inv * block_amax / E2M1_MAX
    saturation = raw_block_scale > E4M3_MAX
    block_scale = raw_block_scale.clamp(-E4M3_MAX, E4M3_MAX)
    block_scale = block_scale.to(torch.float8_e4m3fn).float()
    output_scale = torch.where(
        block_scale == 0.0,
        torch.zeros_like(block_scale),
        input_scale_inv / block_scale,
    )
    scaled = (rows * output_scale).clamp(-E2M1_MAX, E2M1_MAX)
    clipped = scaled.abs() >= E2M1_MAX
    quantized = cast_to_e2m1(scaled)
    dequantized = (quantized * block_scale / input_scale_inv).reshape_as(x)
    stats = {
        "saturated_block_fraction": float(saturation.float().mean()),
        "clipped_value_fraction": float(clipped.float().mean()),
        "zero_value_fraction": float((quantized == 0).float().mean()),
        "raw_block_scale_max": float(raw_block_scale.max()),
        "encoded_block_scale_min": float(block_scale.min()),
        "encoded_block_scale_max": float(block_scale.max()),
    }
    return dequantized, stats


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - reference.float()
    reference_f = reference.float()
    denominator = torch.linalg.vector_norm(reference_f).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return {
        "relative_l2": float(torch.linalg.vector_norm(delta) / denominator),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rms_abs": float(delta.square().mean().sqrt()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual.float().flatten(), reference_f.flatten(), dim=0
            )
        ),
    }


class Checkpoint:
    def __init__(self, root: Path) -> None:
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def tensor(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        with safe_open(self.root / filename, framework="pt", device="cpu") as sf:
            return sf.get_tensor(key)

    def keys(self, prefix: str) -> list[str]:
        return sorted(key for key in self.weight_map if key.startswith(prefix))


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + 1e-6)
    return normalized * (1.0 + weight.float())


def layer_input_scale(checkpoint: Checkpoint, layer: int) -> tuple[float, list[str]]:
    prefix = f"model.language_model.layers.{layer}."
    candidates = [
        key
        for key in checkpoint.keys(prefix)
        if key.endswith(".input_scale")
        and (
            ".linear_attn.in_proj_" in key
            or any(
                f".self_attn.{name}.input_scale" in key
                for name in ("q_proj", "k_proj", "v_proj")
            )
        )
    ]
    if not candidates:
        raise KeyError(f"no attention-input scales found for layer {layer}")
    values = [float(checkpoint.tensor(key).float().reshape(())) for key in candidates]
    if len(set(values)) != 1:
        raise ValueError(f"layer {layer} has non-shared input scales: {values}")
    return values[0], candidates


def trace_records(
    checkpoint: Checkpoint,
    trace_path: Path,
    factors: tuple[float, ...],
) -> dict[str, Any]:
    trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    records: list[dict[str, Any]] = []
    for layer in trace["layer_indices"]:
        hidden = trace["layers"][layer].float().reshape(1, -1)
        norm_weight = checkpoint.tensor(
            f"model.language_model.layers.{layer}.input_layernorm.weight"
        )
        normalized = gemma_rms_norm(hidden, norm_weight)
        checkpoint_scale, scale_keys = layer_input_scale(checkpoint, layer)
        sweeps = []
        for factor in factors:
            effective_scale = checkpoint_scale * factor
            dequantized, quant_stats = quant_dequant(normalized, effective_scale)
            sweeps.append(
                {
                    "factor": factor,
                    "effective_input_scale": effective_scale,
                    **quant_stats,
                    **error_metrics(dequantized, normalized),
                }
            )
        best_l2 = min(sweeps, key=lambda item: item["relative_l2"])
        best_max = min(sweeps, key=lambda item: item["max_abs"])
        records.append(
            {
                "layer": layer,
                "scale_keys": scale_keys,
                "checkpoint_input_scale": checkpoint_scale,
                "checkpoint_calibrated_amax": checkpoint_scale * E2M1_MAX * E4M3_MAX,
                "observed_abs_max": float(normalized.abs().max()),
                "observed_to_calibrated_amax": float(
                    normalized.abs().max() / (checkpoint_scale * E2M1_MAX * E4M3_MAX)
                ),
                "best_relative_l2_factor": best_l2["factor"],
                "best_relative_l2": best_l2["relative_l2"],
                "best_max_abs_factor": best_max["factor"],
                "best_max_abs": best_max["max_abs"],
                "sweep": sweeps,
            }
        )
    return {
        "trace": str(trace_path),
        "query_len": int(trace["query_len"]),
        "input_ids": trace["input_ids"].tolist(),
        "positions": trace["positions"].tolist(),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=DEFAULT_FACTORS,
    )
    args = parser.parse_args()
    factors = tuple(args.factors)
    if any(not math.isfinite(value) or value <= 0 for value in factors):
        parser.error("all scale factors must be finite and positive")

    checkpoint = Checkpoint(args.model_root)
    result = {
        "contract": "CPU replay of W4A4 activation quantization only",
        "model_root": str(args.model_root),
        "factors": factors,
        "traces": [
            trace_records(checkpoint, trace_path, factors) for trace_path in args.trace
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
