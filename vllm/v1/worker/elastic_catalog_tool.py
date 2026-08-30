# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline lifecycle operations for sealed elastic CUDA Graph catalogs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from vllm.v1.core.elastic_catalog import (
    finalize_migrated_catalog,
    rebind_finalized_catalog,
)


def finalize_catalog(source: Path, output_root: Path) -> Path:
    """Create a serving-ready catalog without mutating the migration source."""
    return finalize_migrated_catalog(source, output_root)


def rebind_catalog(
    source: Path,
    output_root: Path,
    destination_fingerprint: str,
    reason: str,
) -> Path:
    """Create an auditable source-identity rebind for accepted physical rows."""
    return rebind_finalized_catalog(
        source,
        output_root,
        destination_fingerprint=destination_fingerprint,
        reason=reason,
    )


def publish_measured_catalog(
    vllm_config: Any,
    kv_cache_config: Any,
    catalog: dict[tuple[int, ...], dict[str, int]],
    *,
    required: tuple[tuple[int, ...], ...],
    restore: tuple[tuple[int, ...], ...],
    decode_max_x: int,
    mixed_max_x: int,
    full_context_max_x: int,
    calibration_wall_seconds: float,
    output_root: Path,
) -> Path:
    """Atomically publish complete measurements from the offline producer."""
    from vllm.v1.core.elastic_catalog import (
        ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        elastic_graph_catalog_row_complete,
    )
    from vllm.v1.core.elastic_graph import configured_compiled_piecewise_sizes
    from vllm.v1.worker import startup_plan

    if not required or not restore or not set(restore).issubset(required):
        raise RuntimeError("measured catalog has invalid required/restore coverage")
    if (
        decode_max_x < 1
        or mixed_max_x < 1
        or full_context_max_x < 1
        or mixed_max_x > decode_max_x
        or full_context_max_x > decode_max_x
    ):
        raise RuntimeError("measured catalog has invalid product boundaries")
    missing = [
        key
        for key in required
        if not elastic_graph_catalog_row_complete(
            key,
            catalog.get(key) or {},
            representation="bounded_exact_hotset",
        )
    ]
    if missing:
        raise RuntimeError(
            "measured catalog coverage is incomplete: "
            f"missing={len(missing)} first={missing[:8]}"
        )

    policy = startup_plan._effective_graph_execution_policy(kv_cache_config)
    compiled = configured_compiled_piecewise_sizes(vllm_config)
    shapes = []
    for key in sorted(required):
        row = {"step_key": list(key), **catalog[key]}
        row.update(
            startup_plan._catalog_policy_row_metadata(
                key,
                policy=policy,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
                compiled_piecewise_sizes=compiled,
            )
        )
        shapes.append(row)
    coverage = {
        "required_step_keys": [list(key) for key in sorted(required)],
        "required_shapes": len(required),
        "restore_step_keys": [list(key) for key in sorted(restore)],
        "calibration_wall_seconds": calibration_wall_seconds,
        "decode_max_x": decode_max_x,
        "mixed_max_x": mixed_max_x,
        "full_context_max_x": full_context_max_x,
        "pinned_full_entries": 0,
        "pinned_full_bytes": 0,
        "representation": "bounded_exact_hotset",
        "piecewise_replay_contract": (startup_plan.BOUNDED_PIECEWISE_REPLAY_CONTRACT),
        "compiled_piecewise_sizes": sorted(compiled),
        "compiled_piecewise_contract": (
            "torch-compile-no-cudagraph-v1" if compiled else None
        ),
        "compiled_piecewise_owners": (["mtp_prefill", "target"] if compiled else []),
        "graph_execution_policy_fingerprint": policy.fingerprint,
        "verifier_contract": policy.verifier_contract,
        "verifier_configuration": policy.verifier_configuration,
        "math_contract": policy.math_contract,
        "capture_state_abi": startup_plan.ELASTIC_CAPTURE_STATE_ABI,
    }
    fingerprint = startup_plan.compute_elastic_graph_catalog_fingerprint(
        vllm_config, kv_cache_config
    )
    payload = {
        "schema": ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "sealed": True,
        "finalized_offline": True,
        "coverage": coverage,
        "graph_execution_policy": policy.to_payload(),
        "complete_shapes": len(shapes),
        "shapes": shapes,
    }
    destination = (
        output_root
        / "elastic_graph_catalog"
        / f"elastic_graph_catalog_{fingerprint}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite measured catalog: {destination}")
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination-fingerprint")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.destination_fingerprint:
        if args.reason is None:
            parser.error("--reason is required with --destination-fingerprint")
        result = rebind_catalog(
            args.source,
            args.output_root,
            args.destination_fingerprint,
            args.reason,
        )
    else:
        if args.reason is not None:
            parser.error("--reason requires --destination-fingerprint")
        result = finalize_catalog(args.source, args.output_root)
    print(result)


if __name__ == "__main__":
    main()
