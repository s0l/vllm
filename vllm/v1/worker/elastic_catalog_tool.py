# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline lifecycle operations for sealed elastic CUDA Graph catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from vllm.v1.core.elastic_catalog import (
    RestoreDecodeGeometry,
    finalize_migrated_catalog,
    rebind_finalized_catalog,
    resolve_restore_decode_geometry,
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


def migrate_price_catalog(source: Path, destination: Path, review_path: Path) -> Path:
    """Migrate a reviewed legacy envelope without changing any measured byte."""
    from vllm.v1.core.elastic_catalog import (
        load_sealed_catalog_with_digest,
        write_json_exclusive,
    )
    from vllm.v1.core.elastic_graph import GraphExecutionPolicy
    from vllm.v1.core.elastic_price_identity import (
        PRICE_OWNER_GENERATION,
        remap_catalog_resident_keys,
        validate_price_identity,
    )

    payload, source_sha = load_sealed_catalog_with_digest(
        source, require_migration=False
    )
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)
    if (
        not isinstance(review, dict)
        or review.get("schema") != "elastic-price-migration-review-v1"
        or review.get("source_sha256") != source_sha
        or review.get("contract") != "same-physical-price-envelope-v1"
        or not isinstance(review.get("source_generation"), str)
        or not review["source_generation"]
        or not isinstance(review.get("evidence"), dict)
        or not review["evidence"]
    ):
        raise RuntimeError(
            "price migration requires an exact-source compatibility review"
        )
    for evidence_path, digest in review["evidence"].items():
        if hashlib.sha256(Path(evidence_path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("price migration review evidence changed")
    identity = validate_price_identity(review.get("destination_price_identity"))
    if (
        source.resolve() == destination.resolve()
        or review_path.resolve() == destination.resolve()
    ):
        raise RuntimeError("price migration must not overwrite its source or review")
    rows = {tuple(row["step_key"]): dict(row) for row in payload["shapes"]}
    rebound = remap_catalog_resident_keys(
        rows,
        source_generation=review["source_generation"],
        destination_generation=PRICE_OWNER_GENERATION,
        max_num_batched_tokens=review["max_num_batched_tokens"],
        compiled_piecewise_sizes=tuple(payload["coverage"]["compiled_piecewise_sizes"]),
        policy=GraphExecutionPolicy.from_payload(payload["graph_execution_policy"]),
    )
    payload["shapes"] = [rebound[tuple(row["step_key"])] for row in payload["shapes"]]
    payload["fingerprint"] = identity["fingerprint"]
    payload["price_identity"] = identity
    payload["resident_owner_generation"] = PRICE_OWNER_GENERATION
    payload["price_migration"] = {
        "source_sha256": source_sha,
        "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "contract": review["contract"],
        "evidence": review["evidence"],
    }
    write_json_exclusive(destination, payload)
    return destination


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
    restore_decode: RestoreDecodeGeometry | None = None,
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
    from vllm.v1.core.elastic_price_identity import (
        PRICE_OWNER_GENERATION,
        remap_catalog_resident_keys,
    )

    catalog = remap_catalog_resident_keys(
        catalog,
        source_generation=startup_plan.elastic_catalog_owner_generation(
            vllm_config, kv_cache_config
        ),
        destination_generation=PRICE_OWNER_GENERATION,
        max_num_batched_tokens=max_num_batched_tokens,
        compiled_piecewise_sizes=tuple(compiled),
        policy=policy,
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
    geometry = restore_decode or resolve_restore_decode_geometry(restore)
    geometry.validate_runtime(
        configured_k=decode_k,
        max_num_seqs=decode_max_x,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    key_options = dict(
        policy=policy,
        prefill_k=prefill_k,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    if set(geometry.step_keys(**key_options)) != set(restore):
        raise RuntimeError("measured catalog restore pair differs from runtime")
    q1 = RestoreDecodeGeometry(k=decode_k, x=decode_max_x, query_len=1)
    verification = RestoreDecodeGeometry(
        k=decode_k,
        x=decode_max_x,
        query_len=decode_k + 1,
    )
    serving_carrier = tuple(
        sorted(
            set(
                (
                    *q1.step_keys(**key_options),
                    *verification.step_keys(**key_options),
                )
            )
        )
    )
    if not set(serving_carrier).issubset(required):
        raise RuntimeError(
            "measured catalog omits the terminal prefill/q1/q(K+1) carrier: "
            f"missing={sorted(set(serving_carrier) - set(required))!r}"
        )
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
        "restore_decode": geometry.to_payload(),
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
        "price_identity": startup_plan.compute_elastic_graph_price_identity(
            vllm_config, kv_cache_config
        ),
        "resident_owner_generation": PRICE_OWNER_GENERATION,
        "sealed": True,
        "finalized_offline": True,
        "coverage": coverage,
        "graph_execution_policy": policy.to_payload(),
        "complete_shapes": len(shapes),
        "shapes": shapes,
    }
    selected_path = os.environ.get("AG2_VLLM_ELASTIC_CATALOG_PATH")
    if selected_path and not Path(selected_path).is_absolute():
        raise RuntimeError("explicit elastic catalog path must be absolute")
    destination = Path(
        selected_path
        or output_root
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
    parser.add_argument("--price-migration-review", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.price_migration_review:
        if args.destination is None or args.destination_fingerprint or args.reason:
            parser.error(
                "price migration requires --destination, without legacy rebind options"
            )
        result = migrate_price_catalog(
            args.source, args.destination, args.price_migration_review
        )
    elif args.destination is not None:
        parser.error("--destination requires --price-migration-review")
    elif args.destination_fingerprint:
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
