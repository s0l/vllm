# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline lifecycle operations for sealed elastic CUDA Graph catalogs."""

from __future__ import annotations

import argparse
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
    catalog: dict[tuple[int, ...], dict[str, Any]],
    *,
    required: tuple[tuple[int, ...], ...],
    restore: tuple[tuple[int, ...], ...],
    decode_max_x: int,
    mixed_max_x: int,
    full_context_max_x: int,
    calibration_wall_seconds: float,
    output_root: Path,
    semantic_token_witnesses: tuple[
        tuple[tuple[int, int, int, int, int], int], ...
    ] = (),
    mixed_query_witnesses: tuple[tuple[tuple[int, int, int, int, int], int], ...] = (),
    source_surface_schema: int | None = None,
    source_surface_required_shapes: int | None = None,
    surface_migration_contract: str | None = None,
    surface_full_aliases: tuple[
        tuple[
            tuple[int, int, int, int, int],
            tuple[int, int, int, int, int],
        ],
        ...,
    ] = (),
    migrated_source_keys: tuple[tuple[int, int, int, int, int], ...] = (),
    source_surface_sha256: str | None = None,
    owner_evidence_keys: tuple[tuple[int, int, int, int, int], ...] = (),
) -> Path:
    """Atomically publish complete measurements from the offline producer."""
    from vllm.v1.core.elastic_catalog import (
        ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION,
        elastic_graph_catalog_row_complete,
        expected_semantic_token_witnesses,
        validate_elastic_catalog_key_inventory,
        validate_schema6_surface_lineage,
        write_json_exclusive,
    )
    from vllm.v1.core.elastic_graph import (
        EXECUTION_MANIFEST_SCHEMA,
        DispatchRepresentation,
        configured_compiled_piecewise_sizes,
        select_short_decode_physical_x,
        short_decode_inventory_xs,
    )
    from vllm.v1.kv_cache_interface import elastic_piecewise_token_boundary
    from vllm.v1.worker import startup_plan

    required, restore = validate_elastic_catalog_key_inventory(
        required,
        restore,
        label="measured catalog",
        require_restore=True,
    )
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
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    decode_k = int(vllm_config.num_speculative_tokens)
    speculative_config = vllm_config.speculative_config
    prefill_k = (
        0
        if speculative_config is not None
        and speculative_config.disable_speculation_on_non_decode
        else decode_k
    )
    if any(key[1] != (decode_k if key[0] == 1 else prefill_k) for key in required):
        raise RuntimeError("measured catalog differs from phase-specific K")
    expected_witnesses = expected_semantic_token_witnesses(
        configured_k=decode_k,
        prefill_k=prefill_k,
        max_num_seqs=decode_max_x,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    if semantic_token_witnesses != expected_witnesses:
        raise RuntimeError(
            "measured catalog semantic witnesses differ from the effective runtime"
        )
    witness_keys = tuple(key for key, _live_tokens in semantic_token_witnesses)
    if not set(witness_keys).issubset(required) or any(
        key[0] != 0
        or key[4] != 0
        or live_tokens < key[2]
        or live_tokens > key[3]
        or elastic_piecewise_token_boundary(live_tokens, max_num_batched_tokens)
        != key[3]
        for key, live_tokens in semantic_token_witnesses
    ):
        raise RuntimeError("measured catalog has invalid semantic witnesses")
    semantic_live_tokens = dict(semantic_token_witnesses)
    if len(set(mixed_query_witnesses)) != len(mixed_query_witnesses) or any(
        key not in required
        or key[0] != 0
        or key[4] != 0
        or key[2] < 2
        or query_len not in (1, 1 + decode_k)
        or semantic_live_tokens.get(key, key[3]) <= (key[2] - 1) * query_len
        for key, query_len in mixed_query_witnesses
    ):
        raise RuntimeError("measured catalog has invalid mixed query witnesses")

    policy = startup_plan._effective_graph_execution_policy(kv_cache_config)
    compiled = configured_compiled_piecewise_sizes(vllm_config)
    terminal_query_len = decode_k + 1
    terminal_verification = (
        int(policy.mode_for("target", terminal_query_len) == "FULL"),
        decode_k,
        decode_max_x,
        decode_max_x * terminal_query_len,
        terminal_query_len,
    )
    shapes = []
    for key in sorted(required):
        catalog_row = dict(catalog[key])
        resident_key_bytes = startup_plan._validated_catalog_resident_key_bytes(
            catalog_row.get("resident_key_bytes"),
            resident_bytes=catalog_row["resident_bytes"],
            cold_peak_bytes=catalog_row["cold_peak_bytes"],
        )
        if resident_key_bytes is not None:
            catalog_row["resident_key_bytes"] = resident_key_bytes
        row = {"step_key": list(key), **catalog_row}
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
    if terminal_verification not in required:
        raise RuntimeError(
            "measured catalog omits the terminal verification carrier: "
            f"key={terminal_verification!r}"
        )
    serving_carrier = tuple(sorted(set((*restore, terminal_verification))))
    coverage = {
        "required_step_keys": [list(key) for key in sorted(required)],
        "required_shapes": len(required),
        "semantic_token_witnesses": [
            {"step_key": list(key), "live_num_tokens": live_tokens}
            for key, live_tokens in semantic_token_witnesses
        ],
        "semantic_witness_contract": "live-current-to-physical-piecewise-v1",
        "mixed_query_witnesses": [
            {"step_key": list(key), "query_len": query_len}
            for key, query_len in mixed_query_witnesses
        ],
        "mixed_query_witness_contract": "shape-specific-query-distribution-v1",
        "restore_step_keys": [list(key) for key in sorted(restore)],
        "serving_carrier_step_keys": [list(key) for key in serving_carrier],
        "serving_carrier_contract": "retained-terminal-mtp-no-cold-serving-v1",
        "serving_carrier_owner": "mtp_decode",
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
        "execution_manifest_schema": EXECUTION_MANIFEST_SCHEMA,
        "dispatch_representations": [item.value for item in DispatchRepresentation],
        "residency_intent_contract": "current-dispatch-hot-successor-union-v1",
        "speculative_depth_contract": "scheduled-requested-executed-k-v1",
    }
    if source_surface_schema is not None:
        migrated_source_set = set(migrated_source_keys)
        witness_key_set = set(witness_keys)
        alias_sources = {source for source, _destination in surface_full_aliases}
        alias_destinations = {
            destination for _source, destination in surface_full_aliases
        }
        expected_alias_sources = {
            (1, decode_k, x, x, 1) for x in range(1, decode_max_x + 1)
        }
        inventory_xs = short_decode_inventory_xs(decode_max_x)

        def expected_alias(
            source: tuple[int, int, int, int, int],
        ) -> tuple[int, int, int, int, int]:
            physical_x = select_short_decode_physical_x(source[2], inventory_xs)
            return (1, source[1], physical_x, physical_x * source[4], source[4])

        if (
            isinstance(source_surface_schema, bool)
            or source_surface_schema not in (5, ELASTIC_GRAPH_CATALOG_SCHEMA_VERSION)
            or isinstance(source_surface_required_shapes, bool)
            or not isinstance(source_surface_required_shapes, int)
            or not isinstance(source_surface_sha256, str)
            or len(source_surface_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_surface_sha256)
            or not migrated_source_keys
            or len(migrated_source_set) != len(migrated_source_keys)
            or set(required) != migrated_source_set.union(witness_key_set)
            or (
                source_surface_schema == 5
                and surface_migration_contract
                != "schema5-full-owner-evidence-and-bounded-alias-v1"
            )
            or (
                source_surface_schema == 5
                and (
                    not surface_full_aliases
                    or len(alias_sources) != len(surface_full_aliases)
                    or alias_sources != expected_alias_sources
                    or set(owner_evidence_keys) != alias_sources
                    or not set(owner_evidence_keys).issubset(migrated_source_set)
                    or source_surface_required_shapes
                    != sum(key[0] == 0 for key in migrated_source_keys)
                    + len(alias_sources)
                    or not alias_destinations.issubset(migrated_source_set)
                    or any(
                        source[0] != 1
                        or destination[0] != 1
                        or source[4] <= 0
                        or source[3] != source[2] * source[4]
                        or expected_alias(source) != destination
                        for source, destination in surface_full_aliases
                    )
                )
            )
            or (
                source_surface_schema != 5
                and (
                    surface_migration_contract is not None
                    or bool(surface_full_aliases)
                    or bool(owner_evidence_keys)
                    or source_surface_required_shapes != len(migrated_source_keys)
                )
            )
        ):
            raise RuntimeError("measured catalog has invalid surface migration lineage")
        coverage.update(
            {
                "surface_lineage_contract": "typed-source-serving-v1",
                "source_surface_schema": source_surface_schema,
                "source_surface_required_shapes": source_surface_required_shapes,
                "surface_migration_contract": surface_migration_contract,
                "source_surface_sha256": source_surface_sha256,
                "migrated_source_required_shapes": len(migrated_source_keys),
                "migrated_source_step_keys": [
                    list(key) for key in migrated_source_keys
                ],
                "schema6_added_witness_shapes": len(
                    witness_key_set.difference(migrated_source_set)
                ),
                "surface_full_aliases": [
                    {
                        "semantic_step_key": list(source),
                        "physical_step_key": list(destination),
                    }
                    for source, destination in surface_full_aliases
                ],
                "owner_evidence_step_keys": [list(key) for key in owner_evidence_keys],
            }
        )
    validate_schema6_surface_lineage(coverage, required)
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
    write_json_exclusive(destination, payload)
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
