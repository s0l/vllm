# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure, fail-closed model/topology plan compiler for AG2 distributed paths.

This module deliberately has no CUDA or torch dependency.  It turns explicit
model/backend capabilities and a measured physical topology into one immutable
receipt.  Runtime code may consume a receipt, but must not independently
reconstruct owner widths, offsets, rank placement, or arithmetic order.

The accepted TP3 numerical contract is logical ``(r0 + r1) + r2``.  Physical
GPUs may be mapped to logical ranks before execution; the logical sum order is
never changed by topology selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "ag2-distributed-runtime-plan-v1"
EXACT_GEOMETRY_RUNTIME_SELECTION_SCHEMA = "ag2.tp3-exact-geometry-runtime-selection.v1"
EXACT_TP3_SUM_ORDER = ((0, 1), 2)


class PlanError(ValueError):
    """The supplied capability, topology, or calibration is not admissible."""


_ACTIVE_RUNTIME_PLAN: RuntimePlan | None = None


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_identity(value: Any) -> str:
    """Return the stable identity used by model/backend/runtime receipts."""
    return _canonical_sha256(value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class ModelGeometry:
    model_identity: str
    hidden_size: int
    quant_group_size: int
    rms_root_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    speculative_tokens: int
    tensor_parallel_size: int

    def validate(self) -> None:
        if not _is_sha256(self.model_identity):
            raise PlanError("model_identity must be a lowercase SHA-256")
        positive = (
            self.hidden_size,
            self.quant_group_size,
            self.rms_root_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.tensor_parallel_size,
        )
        if any(value <= 0 for value in positive) or self.speculative_tokens < 0:
            raise PlanError("model geometry values must be positive")
        if self.tensor_parallel_size != 3:
            raise PlanError("the current exact arithmetic contract requires TP=3")
        if self.hidden_size % self.quant_group_size:
            raise PlanError("hidden_size must be divisible by quant_group_size")
        if self.hidden_size % self.rms_root_size:
            raise PlanError("hidden_size must be divisible by rms_root_size")
        if self.num_attention_heads % self.num_key_value_heads:
            raise PlanError("attention heads must form complete GQA groups")
        if self.num_attention_heads * self.head_dim <= 0:
            raise PlanError("attention projection geometry is empty")


@dataclass(frozen=True)
class BackendCapabilities:
    backend_identity: str
    owner_alignment: int
    preserves_root_local_math: bool

    def validate(self, model: ModelGeometry) -> None:
        if not _is_sha256(self.backend_identity):
            raise PlanError("backend_identity must be a lowercase SHA-256")
        if self.owner_alignment <= 0:
            raise PlanError("owner_alignment must be positive")
        if model.hidden_size % self.owner_alignment:
            raise PlanError("hidden_size must be divisible by owner_alignment")
        if (
            self.preserves_root_local_math
            and model.rms_root_size % self.owner_alignment
        ):
            raise PlanError("RMS root must be divisible by owner_alignment")


@dataclass(frozen=True)
class GpuEndpoint:
    physical_id: str
    pci_bus_id: str
    compute_score: float

    def validate(self) -> None:
        if not self.physical_id or not self.pci_bus_id:
            raise PlanError("GPU identity and PCI BDF are required")
        if not math.isfinite(self.compute_score) or self.compute_score <= 0:
            raise PlanError("GPU compute_score must be finite and positive")


@dataclass(frozen=True)
class PairLink:
    left: str
    right: str
    bandwidth_gbps: float
    latency_us: float

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.left, self.right)))

    def validate(self) -> None:
        if self.left == self.right:
            raise PlanError("a pair link requires two different GPUs")
        if (
            not math.isfinite(self.bandwidth_gbps)
            or self.bandwidth_gbps <= 0
            or not math.isfinite(self.latency_us)
            or self.latency_us < 0
        ):
            raise PlanError("pair-link service must be finite and nonnegative")


@dataclass(frozen=True)
class PhysicalTopology:
    topology_identity: str
    gpus: tuple[GpuEndpoint, GpuEndpoint, GpuEndpoint]
    links: tuple[PairLink, PairLink, PairLink]

    def validate(self) -> None:
        if not _is_sha256(self.topology_identity):
            raise PlanError("topology_identity must be a lowercase SHA-256")
        for gpu in self.gpus:
            gpu.validate()
        physical_ids = tuple(gpu.physical_id for gpu in self.gpus)
        if len(set(physical_ids)) != 3:
            raise PlanError("topology requires three unique physical GPUs")
        for link in self.links:
            link.validate()
        expected = {
            tuple(sorted(pair))
            for pair in (
                (physical_ids[0], physical_ids[1]),
                (physical_ids[0], physical_ids[2]),
                (physical_ids[1], physical_ids[2]),
            )
        }
        actual = {link.key for link in self.links}
        if actual != expected or len(actual) != len(self.links):
            raise PlanError(
                "topology must contain every physical GPU pair exactly once"
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PhysicalTopology:
        try:
            return cls(
                topology_identity=value["topology_identity"],
                gpus=tuple(GpuEndpoint(**item) for item in value["gpus"]),
                links=tuple(PairLink(**item) for item in value["links"]),
            )  # type: ignore[arg-type]
        except (KeyError, TypeError) as error:
            raise PlanError("invalid physical topology receipt") from error


@dataclass(frozen=True)
class MeasuredPlan:
    rows: int
    owner_widths: tuple[int, int, int]
    logical_to_physical: tuple[str, str, str]
    critical_ms: float
    exact: bool

    def validate(self, model: ModelGeometry, backend: BackendCapabilities) -> None:
        if (
            self.rows <= 0
            or not math.isfinite(self.critical_ms)
            or self.critical_ms <= 0
        ):
            raise PlanError("measured plan rows/time must be finite and positive")
        _validate_widths(self.owner_widths, model, backend)
        if len(set(self.logical_to_physical)) != 3:
            raise PlanError("measured logical-to-physical mapping must be bijective")


@dataclass(frozen=True)
class CalibrationSurface:
    model_identity: str
    backend_identity: str
    topology_identity: str
    transport_identity: str
    complete: bool
    entries: tuple[MeasuredPlan, ...]

    def validate(
        self,
        model: ModelGeometry,
        backend: BackendCapabilities,
        topology: PhysicalTopology,
    ) -> None:
        identities = (
            self.model_identity,
            self.backend_identity,
            self.topology_identity,
            self.transport_identity,
        )
        if not all(_is_sha256(value) for value in identities):
            raise PlanError("calibration identities must be lowercase SHA-256 values")
        if (
            self.model_identity != model.model_identity
            or self.backend_identity != backend.backend_identity
            or self.topology_identity != topology.topology_identity
        ):
            raise PlanError("calibration identity is stale for this runtime")
        physical_ids = {gpu.physical_id for gpu in topology.gpus}
        seen: set[tuple[int, tuple[int, int, int], tuple[str, str, str]]] = set()
        for entry in self.entries:
            entry.validate(model, backend)
            if set(entry.logical_to_physical) != physical_ids:
                raise PlanError("calibration references a different physical GPU set")
            key = (entry.rows, entry.owner_widths, entry.logical_to_physical)
            if key in seen:
                raise PlanError("calibration contains a duplicate operating point")
            seen.add(key)


@dataclass(frozen=True)
class RuntimePlan:
    schema: str
    model_identity: str
    backend_identity: str
    topology_identity: str
    rows: int
    owner_widths: tuple[int, int, int]
    owner_offsets: tuple[int, int, int]
    logical_to_physical: tuple[str, str, str]
    exact_sum_order: tuple[tuple[int, int], int]
    source: str
    predicted_critical_ms: float | None
    plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_widths(
    widths: tuple[int, int, int],
    model: ModelGeometry,
    backend: BackendCapabilities,
) -> None:
    if len(widths) != model.tensor_parallel_size or min(widths) <= 0:
        raise PlanError("owner widths must have one positive entry per TP rank")
    if sum(widths) != model.hidden_size:
        raise PlanError("owner widths must conserve hidden_size")
    alignment = (
        model.rms_root_size
        if backend.preserves_root_local_math
        else backend.owner_alignment
    )
    if any(width % alignment for width in widths):
        raise PlanError(f"owner widths must be aligned to {alignment}")


def _apportion_units(
    total: int, weights: tuple[float, float, float]
) -> tuple[int, int, int]:
    if total < len(weights):
        raise PlanError("not enough indivisible owner units for every TP rank")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise PlanError("apportionment weights must be finite and positive")
    remaining = total - len(weights)
    weight_sum = sum(weights)
    quotas = tuple(remaining * weight / weight_sum for weight in weights)
    extras = [math.floor(quota) for quota in quotas]
    left = remaining - sum(extras)
    order = sorted(
        range(len(weights)),
        key=lambda index: (quotas[index] - extras[index], weights[index], -index),
        reverse=True,
    )
    for index in order[:left]:
        extras[index] += 1
    return tuple(value + 1 for value in extras)  # type: ignore[return-value]


def _best_fast_pair_mapping(topology: PhysicalTopology) -> tuple[str, str, str]:
    """Map the strongest physical edge to logical r0/r1.

    Bandwidth owns the primary ordering; latency and endpoint compute service
    are deterministic tie breakers.  A measured complete surface supersedes
    this structural fallback.
    """
    gpu_by_id = {gpu.physical_id: gpu for gpu in topology.gpus}
    best = max(
        topology.links,
        key=lambda link: (
            link.bandwidth_gbps,
            -link.latency_us,
            gpu_by_id[link.left].compute_score + gpu_by_id[link.right].compute_score,
            tuple(reversed(link.key)),
        ),
    )
    pair = sorted(
        best.key,
        key=lambda physical_id: (
            -gpu_by_id[physical_id].compute_score,
            physical_id,
        ),
    )
    leaf = next(physical_id for physical_id in gpu_by_id if physical_id not in pair)
    return pair[0], pair[1], leaf


def _fallback_plan(
    model: ModelGeometry,
    backend: BackendCapabilities,
    topology: PhysicalTopology,
) -> tuple[tuple[int, int, int], tuple[str, str, str]]:
    mapping = _best_fast_pair_mapping(topology)
    gpu_by_id = {gpu.physical_id: gpu for gpu in topology.gpus}
    unit = (
        model.rms_root_size
        if backend.preserves_root_local_math
        else backend.owner_alignment
    )
    unit_count = model.hidden_size // unit
    weights = tuple(gpu_by_id[physical_id].compute_score for physical_id in mapping)
    counts = _apportion_units(unit_count, weights)  # type: ignore[arg-type]
    widths = tuple(count * unit for count in counts)
    _validate_widths(widths, model, backend)
    return widths, mapping


def _offsets(widths: tuple[int, int, int]) -> tuple[int, int, int]:
    return 0, widths[0], widths[0] + widths[1]


def compile_runtime_plan(
    *,
    model: ModelGeometry,
    backend: BackendCapabilities,
    topology: PhysicalTopology,
    rows: int,
    calibration: CalibrationSurface | None = None,
) -> RuntimePlan:
    """Compile one immutable exact plan for an already-known physical shape."""
    model.validate()
    backend.validate(model)
    topology.validate()
    if rows <= 0:
        raise PlanError("rows must be positive")

    selected: MeasuredPlan | None = None
    source = "derived-exact-safe-fallback"
    if calibration is not None:
        calibration.validate(model, backend, topology)
        if calibration.complete:
            compatible = [
                entry
                for entry in calibration.entries
                if entry.rows == rows and entry.exact
            ]
            if compatible:
                selected = min(
                    compatible,
                    key=lambda entry: (
                        entry.critical_ms,
                        entry.owner_widths,
                        entry.logical_to_physical,
                    ),
                )
                source = "measured-complete-surface"

    if selected is None:
        widths, mapping = _fallback_plan(model, backend, topology)
        predicted = None
    else:
        widths = selected.owner_widths
        mapping = selected.logical_to_physical
        predicted = selected.critical_ms

    payload = {
        "schema": PLAN_SCHEMA,
        "model_identity": model.model_identity,
        "backend_identity": backend.backend_identity,
        "topology_identity": topology.topology_identity,
        "rows": rows,
        "owner_widths": widths,
        "owner_offsets": _offsets(widths),
        "logical_to_physical": mapping,
        "exact_sum_order": EXACT_TP3_SUM_ORDER,
        "source": source,
        "predicted_critical_ms": predicted,
    }
    return RuntimePlan(**payload, plan_sha256=_canonical_sha256(payload))


def enumerate_logical_mappings(
    topology: PhysicalTopology,
) -> Iterable[tuple[str, str, str]]:
    """Expose the complete mapping space for offline response-surface builders."""
    topology.validate()
    physical_ids = tuple(gpu.physical_id for gpu in topology.gpus)
    return permutations(physical_ids)


def install_runtime_plan(plan: RuntimePlan) -> None:
    """Install one immutable process-local receipt before model loading.

    Re-installing the same receipt is harmless (target and MTP constructors
    share a process).  A different receipt in the same process is rejected:
    compiled graphs and loader metadata may already depend on the first one.
    """
    global _ACTIVE_RUNTIME_PLAN
    if _ACTIVE_RUNTIME_PLAN is not None and plan != _ACTIVE_RUNTIME_PLAN:
        raise PlanError(
            "runtime plan is immutable after installation: "
            f"existing={_ACTIVE_RUNTIME_PLAN.plan_sha256} "
            f"new={plan.plan_sha256}"
        )
    _ACTIVE_RUNTIME_PLAN = plan


def get_runtime_plan(*, required: bool = True) -> RuntimePlan | None:
    if _ACTIVE_RUNTIME_PLAN is None and required:
        raise PlanError("AG2 runtime plan has not been installed")
    return _ACTIVE_RUNTIME_PLAN


_PCI_SPEED_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s+GT/s")
_PCI_WIDTH_RE = re.compile(r"(?:PCIe\s+)?([0-9]+)$")


def _read_pci_capacity_gbps(bus_id: str) -> float:
    """Return the endpoint's negotiated maximum one-way payload ceiling.

    This is a structural ceiling, not a fabricated pair benchmark.  A
    complete measured calibration surface is still required before using a
    throughput prediction.
    """
    short_bus_id = bus_id.lower().removeprefix("00000000:")
    if short_bus_id.count(":") == 1:
        short_bus_id = f"0000:{short_bus_id}"
    device = Path("/sys/bus/pci/devices") / short_bus_id
    speed_text = (device / "max_link_speed").read_text().strip()
    width_text = (device / "max_link_width").read_text().strip()
    speed_match = _PCI_SPEED_RE.match(speed_text)
    width_match = _PCI_WIDTH_RE.search(width_text)
    if speed_match is None or width_match is None:
        raise PlanError(
            f"cannot parse PCIe capability for {bus_id}: "
            f"speed={speed_text!r} width={width_text!r}"
        )
    transfers = float(speed_match.group(1))
    width = int(width_match.group(1))
    # PCIe 3.0+ uses 128b/130b encoding.  Every currently supported endpoint
    # here is Gen3 or newer; fail closed instead of inventing a Gen1/2 model.
    if transfers < 8.0:
        raise PlanError(f"unsupported pre-Gen3 PCIe endpoint {bus_id}: {speed_text}")
    return transfers * width * (128.0 / 130.0) / 8.0


def _pci_root(bus_id: str) -> str:
    short_bus_id = bus_id.lower().removeprefix("00000000:")
    if short_bus_id.count(":") == 1:
        short_bus_id = f"0000:{short_bus_id}"
    path = (Path("/sys/bus/pci/devices") / short_bus_id).resolve()
    roots = [part for part in path.parts if part.startswith("pci") and ":" in part]
    if not roots:
        raise PlanError(f"cannot resolve PCI root for {bus_id}: {path}")
    return roots[0]


def discover_physical_topology(
    visible_devices: str | None = None,
) -> tuple[PhysicalTopology, tuple[str, str, str]]:
    """Discover the three visible GPUs without importing torch/CUDA.

    GPU service is deliberately neutral here.  Model-specific compute service
    must come from a measured, identity-bound calibration surface; PCIe path
    structure alone must not pretend to predict GEMM throughput.
    """
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records: list[dict[str, str]] = []
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise PlanError(f"invalid nvidia-smi GPU row: {line!r}")
        index, uuid, bus_id, name = fields
        records.append({"index": index, "uuid": uuid, "bus_id": bus_id, "name": name})
    by_token = {
        token: record
        for record in records
        for token in (record["index"], record["uuid"])
    }
    visible_text = (
        visible_devices
        if visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    tokens = tuple(token.strip() for token in visible_text.split(",") if token.strip())
    if not tokens:
        tokens = tuple(record["index"] for record in records)
    if len(tokens) != 3 or len(set(tokens)) != 3:
        raise PlanError(f"TP3 topology requires exactly three visible GPUs: {tokens}")
    try:
        selected = tuple(by_token[token] for token in tokens)
    except KeyError as error:
        raise PlanError(
            f"unknown CUDA_VISIBLE_DEVICES entry: {error.args[0]}"
        ) from error

    endpoint_caps = {
        record["uuid"]: _read_pci_capacity_gbps(record["bus_id"]) for record in selected
    }
    roots = {record["uuid"]: _pci_root(record["bus_id"]) for record in selected}
    gpus = tuple(
        GpuEndpoint(
            physical_id=record["uuid"],
            pci_bus_id=record["bus_id"],
            compute_score=1.0,
        )
        for record in selected
    )
    links: list[PairLink] = []
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left = selected[left_index]["uuid"]
        right = selected[right_index]["uuid"]
        links.append(
            PairLink(
                left=left,
                right=right,
                bandwidth_gbps=min(endpoint_caps[left], endpoint_caps[right]),
                # Structural ordinal only: same PCI root is preferred.  This
                # value is not reported as a measured latency.
                latency_us=0.0 if roots[left] == roots[right] else 1.0,
            )
        )
    identity_payload = {
        "gpus": [asdict(gpu) for gpu in gpus],
        "links": [asdict(link) for link in links],
        "pci_roots": roots,
    }
    topology = PhysicalTopology(
        topology_identity=_canonical_sha256(identity_payload),
        gpus=gpus,  # type: ignore[arg-type]
        links=tuple(links),  # type: ignore[arg-type]
    )
    topology.validate()
    return topology, tokens  # type: ignore[return-value]


def topology_bootstrap_receipt(visible_devices: str | None = None) -> dict[str, Any]:
    topology, visible_tokens = discover_physical_topology(visible_devices)
    mapping = _best_fast_pair_mapping(topology)
    token_by_uuid = {
        gpu.physical_id: token
        for gpu, token in zip(topology.gpus, visible_tokens, strict=True)
    }
    ordered_visible = tuple(token_by_uuid[physical_id] for physical_id in mapping)
    payload = {
        "schema": "ag2-topology-bootstrap-v1",
        "topology": asdict(topology),
        "original_visible_devices": visible_tokens,
        "ordered_visible_devices": ordered_visible,
        "exact_sum_order": EXACT_TP3_SUM_ORDER,
        "service_source": "pci-structure-only; compute-neutral",
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload


def _admitted_exact_geometry_receipts(
    cache_root: Path,
    topology_identity: str,
) -> list[dict[str, Any]]:
    """Resolve one explicitly admitted exact-geometry calibration receipt.

    Calibration artifacts are durable evidence, not runtime authority.  A
    structurally valid benchmark receipt therefore remains inert until the
    fixed runtime-selection record content-binds it to independent acceptance
    evidence.  Missing, corrupt, stale and rejected selections all recover to
    the ordinary exact backend by returning no candidate.
    """
    selection_path = cache_root / "runtime-selection.json"
    if not selection_path.is_file():
        return []
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        unsigned_selection = {
            key: value
            for key, value in selection.items()
            if key != "selection_receipt_id"
        }
        selected_sha256 = selection.get("selected_receipt_sha256")
        if (
            selection.get("schema") != EXACT_GEOMETRY_RUNTIME_SELECTION_SCHEMA
            or selection.get("topology_identity") != topology_identity
            or selection.get("status") != "ADMITTED"
            or not isinstance(selected_sha256, str)
            or not _is_sha256(selected_sha256)
            or not _is_sha256(str(selection.get("evidence_sha256", "")))
            or selection.get("selection_receipt_id")
            != _canonical_sha256(unsigned_selection)
        ):
            return []
        provider_path = cache_root / f"{selected_sha256}.json"
        provider = json.loads(provider_path.read_text(encoding="utf-8"))
        unsigned_provider = {
            key: value for key, value in provider.items() if key != "receipt_sha256"
        }
        if (
            provider.get("schema") != "ag2.tp3-exact-geometry-receipt.v1"
            or provider.get("topology_identity") != topology_identity
            or provider.get("receipt_sha256") != selected_sha256
            or selected_sha256 != _canonical_sha256(unsigned_provider)
        ):
            return []
        return [provider]
    except (OSError, ValueError, TypeError):
        return []


def runtime_bootstrap_receipt(visible_devices: str | None = None) -> dict[str, Any]:
    """Attach topology-matched cached provider receipts to the bootstrap.

    The cache is optional. Missing or malformed entries do not invent a
    schedule: model-side compilation retains a fail-safe W1 path. Matching
    candidates are revalidated against imported source and model capability
    before installation.
    """
    receipt = topology_bootstrap_receipt(visible_devices)
    topology_identity = receipt["topology"]["topology_identity"]
    cache_root = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "ag2"
        / "topology-autotune"
        / "tp3-prequant-wave"
        / topology_identity
    )
    candidates = []
    if cache_root.is_dir():
        for path in sorted(cache_root.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                claimed = value.get("receipt_sha256")
                unsigned = {
                    key: item for key, item in value.items() if key != "receipt_sha256"
                }
                if (
                    value.get("schema") == "ag2.tp3-prequant-wave-receipt.v1"
                    and value.get("topology_identity") == topology_identity
                    and claimed == _canonical_sha256(unsigned)
                ):
                    candidates.append(value)
            except (OSError, ValueError, TypeError):
                continue
    exact_geometry_root = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "ag2"
        / "topology-autotune"
        / "tp3-exact-geometry"
        / topology_identity
    )
    exact_geometry_candidates = _admitted_exact_geometry_receipts(
        exact_geometry_root,
        topology_identity,
    )
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    unsigned_receipt["prequant_wave_receipts"] = candidates
    unsigned_receipt["exact_geometry_receipts"] = exact_geometry_candidates
    unsigned_receipt["receipt_sha256"] = _canonical_sha256(unsigned_receipt)
    return unsigned_receipt
