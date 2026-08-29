# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline lifecycle operations for sealed elastic CUDA Graph catalogs."""

from __future__ import annotations

import argparse
from pathlib import Path

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
