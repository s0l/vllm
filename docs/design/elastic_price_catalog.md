# Elastic price catalogs

An executable epoch, a physical price envelope, and an admission budget have
different lifetimes. A missing cache filename is not a calibration decision.

## Identities

* Runtime generation remains bound to the complete runtime source inventory.
  CUDA executable pointers and mutable scheduler/worker state cannot cross it.
* Price identity binds physical producer source, model/compile configuration,
  native artifacts, effective kernel switches, layout strides and mapping
  quantum. Source AST hashing ignores comments/formatting but deliberately does
  not assert that arbitrary executable edits or logging calls are equivalent.
* Current primary/GDN mapped bytes, initial mapped GDN blocks and KV block count
  are placement, not prices. Admission and pre-READY restore must validate the
  current budget. A valid price catalog does not promise that its MaxX fits.
* Coverage is versioned independently. `AG2_VLLM_ELASTIC_CATALOG_PATH` selects
  an immutable absolute-path artifact. A larger requested surface must not
  overwrite the accepted smaller catalog merely because their prices share
  one identity. The default path remains the price-fingerprint cache path.

`price_identity` stores components, not only a digest. Unknown components,
producer changes or absent legacy provenance are not automatic equivalence.
The config hash excludes the vLLM version string; actual native artifacts and
producer code remain bound. Native provenance also covers system-linked
CUDA/NCCL libraries, driver version and the NVIDIA PCI device ancestry.
Package RECORD hashes are packaging evidence, not a claim to identify every
JIT cubin. Physical source/config selectors remain independently bound.
Selected FlashInfer autotune JSON is bound by canonical contents, not by path
or formatting; missing, changed and malformed caches are distinct states.
The native inventory excludes known non-Graph I/O and tool-parser extensions;
unresolved native dependencies remain conservative invalidation factors.

## Resident-owner accounting

Published prices use generation `elastic-price-owner-v1`. Loading resolves
the same physical descriptors into the current runtime generation and retains
each measured byte value. Unknown, duplicate, non-bijective or stale IDs fail
closed. Never match owners by equal byte counts or row order. The ordinary
runtime keys and in-memory accounting stay generation-bound.

Legacy catalogs lack portable owner identities. Migration requires an explicit
`elastic-price-migration-review-v1` receipt containing the exact source SHA,
source generation, destination price identity, max batched tokens and hashes
of compatibility evidence. `elastic_catalog_tool --price-migration-review
REVIEW --source SOURCE --destination DEST --output-root ROOT` writes exclusively
without changing the source. This tool validates lineage and mechanical
translation; it does not prove a human's physical-compatibility claim. Missing
source generation or unsupported prices require investigation, not invented
IDs, discarded credit, or relabelling all old rows as freshly measured.

## Calibration decision and resume

Serving keeps `AG2_VLLM_ELASTIC_AUTO_CALIBRATE=0`. An explicit maintenance job
requires both the maintenance role and `AG2_VLLM_ELASTIC_CALIBRATION_REQUEST`,
an identity-bound JSON object with exactly:

```json
{
  "schema": "elastic-measurement-request-v1",
  "fingerprint": "<destination price fingerprint>",
  "surface_sha256": "<exact requested surface SHA256>",
  "reason": "missing-coverage",
  "evidence": "<reviewed decision / evidence reference>"
}
```

Reasons are `initial-measurement`, `changed-producer` or `missing-coverage`.
Unknown compatibility is not a reason. The flag alone cannot start producers.
Failed/terminal receipts cannot be silently resumed; preserve them and create
a separately reviewed recovery decision.

`AG2_VLLM_ELASTIC_CALIBRATION_SEED_CATALOG` selects a compatible sealed source.
Its price identity, schema, policy, row/coverage source SHA, physical aliases
and semantic token witnesses are validated before reuse. Only missing rows and
unexecuted mixed witnesses run; evidence thresholds are unchanged. Seed,
surface, destination and mutable receipt must have distinct paths.

Partial receipts retain their original resident generation until a new
checkpoint is actually written. Resume translates validated observations into
the current generation. Cancellation cannot relabel old observations as new.
Catalog/maintenance paths and flags have no compiled-model consumer and do not
participate in the AOT key; unknown future flags remain conservative factors.

## Acceptance boundary

CPU controls prove identity, immutable migration, real loader translation,
seed selection, stale/malformed rejection and checkpoint recovery. They do not
accept CUDA restoration, math, product quality, MAXx or throughput. Reusing
prices never bypasses current physical admission, restore or product gates.
