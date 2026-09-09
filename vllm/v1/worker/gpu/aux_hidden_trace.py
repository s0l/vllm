# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authoritative auxiliary-hidden-state tracing for narrow diagnostics."""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from vllm.distributed import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)

FULL_ATTENTION_STAGE_ORDER = (
    "qkv",
    "q",
    "k",
    "v",
    "dcp_output_pack",
    "dcp_lse_pack",
    "dcp_kv_history_pack",
    "dcp_kv_history_meta",
    "dcp_kv_page_indices_pack",
    "dcp_observer_meta",
    "dcp_current_kv_pack",
    "dcp_scale_pack",
    "core",
    "gated",
    "output_parallel",
    "output",
)
DEFAULT_FULL_ATTENTION_STAGES = tuple(
    stage
    for stage in FULL_ATTENTION_STAGE_ORDER
    if stage
    not in {
        "dcp_kv_history_pack",
        "dcp_kv_history_meta",
        "dcp_kv_page_indices_pack",
        "dcp_observer_meta",
        "dcp_current_kv_pack",
        "dcp_scale_pack",
    }
)

COMPACT_FULL_ATTENTION_STAGES = (
    "qkv",
    "q",
    "k",
    "v",
    "gate",
    "dcp_output_pack",
    "dcp_lse_pack",
    "core",
    "gated",
    "output_parallel",
    "output",
)
COMPACT_GDN_STAGES = (
    "qkvz",
    "ba",
    "core",
    "gated_norm",
    "output_parallel",
    "output",
)
COMPACT_MLP_STAGES = (
    "gate_up",
    "activation",
    "down_parallel",
    "output",
)

AG2_AUX_TRACE_SCHEMA_VERSION = 2


@dataclass
class AuxHiddenTrace:
    output: str | None = None
    layers: tuple[int, ...] = ()
    token_id: int = -1
    position: int = -1
    positions: tuple[int, ...] = ()
    max_matches_per_query_len: int = 64
    first_attention_boundary: bool = False
    attention_boundary_layer: int = 0
    full_attention_boundaries: bool = False
    full_attention_stages: tuple[str, ...] = DEFAULT_FULL_ATTENTION_STAGES
    first_gdn_boundaries: bool = False
    gdn_replay_only: bool = False
    gdn_boundary_layer: int = 0
    packed_compare_prefix_tokens: int = 0
    packed_compare_sequence_tokens: int = 0
    packed_compare_sequences: int = 0
    dcp_request_tail_rows: int = 0
    dcp_request_row: str = "tail"
    compact_rows: bool = False
    compact_capacity: int = 64
    compact_boundary_layer: int = -1
    compact_all_boundaries: bool = False
    compact_boundary_max_layer: int = -1
    compact_gdn_boundary_layer: int = -1
    all_internal_boundaries: bool = False
    internal_boundary_layers: tuple[int, ...] = ()
    compact_full_attention_stages: tuple[str, ...] = COMPACT_FULL_ATTENTION_STAGES
    compact_gdn_stages: tuple[str, ...] = COMPACT_GDN_STAGES
    compact_mlp_stages: tuple[str, ...] = COMPACT_MLP_STAGES
    fingerprint_outputs: bool = False
    request_prefix: str = ""
    request_chunks: bool = False
    sequence_boundary_layers: tuple[int, ...] = ()
    _layer_types: dict[int, str] = field(default_factory=dict)
    _saved_matches: dict[int, int] = field(default_factory=dict)
    _invocation: int = 0
    _saved_request_invocations: int = 0

    def __post_init__(self) -> None:
        # Keep direct construction (used by isolated lifecycle controls) on
        # the same single-position contract as from_env().
        if not self.positions and self.position >= 0:
            self.positions = (self.position,)

    @classmethod
    def from_env(cls) -> AuxHiddenTrace:
        output = os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_OUTPUT") or None
        layers_raw = os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_LAYERS", "")
        if not output and not layers_raw:
            return cls()
        if not output or not layers_raw:
            raise ValueError("Aux hidden trace requires both OUTPUT and LAYERS")
        layers = tuple(dict.fromkeys(int(value) for value in layers_raw.split(",")))
        token_id = int(os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_TOKEN_ID", "-1"))
        position = int(os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_POSITION", "-1"))
        positions_raw = os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_POSITIONS", "")
        positions = tuple(
            dict.fromkeys(int(value) for value in positions_raw.split(",") if value)
        ) or ((position,) if position >= 0 else ())
        max_matches = int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_MAX_MATCHES_PER_QUERY_LEN", "64")
        )
        first_attention_boundary = (
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_ATTENTION_BOUNDARY", "0")
            == "1"
        )
        attention_boundary_layer = int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "0")
        )
        full_attention_boundaries = (
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_BOUNDARIES", "0")
            == "1"
        )
        full_attention_stages_raw = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_STAGES", ""
        )
        requested_full_attention_stages = {
            value for value in full_attention_stages_raw.split(",") if value
        }
        unknown_full_attention_stages = requested_full_attention_stages.difference(
            FULL_ATTENTION_STAGE_ORDER
        )
        if unknown_full_attention_stages:
            raise ValueError(
                "Unknown full-attention trace stages: "
                f"{sorted(unknown_full_attention_stages)}"
            )
        full_attention_stages = tuple(
            stage
            for stage in FULL_ATTENTION_STAGE_ORDER
            if (
                stage in requested_full_attention_stages
                if requested_full_attention_stages
                else stage in DEFAULT_FULL_ATTENTION_STAGES
            )
        )
        first_gdn_boundaries = (
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_GDN_BOUNDARIES", "0") == "1"
        )
        gdn_replay_only = (
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_GDN_REPLAY_ONLY", "0") == "1"
        )
        gdn_boundary_layer = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_GDN_BOUNDARY_LAYER",
                "0",
            )
        )
        packed_compare_prefix_tokens = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_PACKED_COMPARE_PREFIX_TOKENS", "0"
            )
        )
        packed_compare_sequence_tokens = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_PACKED_COMPARE_SEQUENCE_TOKENS", "0"
            )
        )
        packed_compare_sequences = int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_PACKED_COMPARE_SEQUENCES", "0")
        )
        dcp_request_tail_rows = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_TAIL_ROWS",
                "0",
            )
        )
        dcp_request_row = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_DCP_REQUEST_ROW",
            "tail",
        )
        compact_rows = (
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ROWS", "0") == "1"
        )
        compact_capacity = int(
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_CAPACITY", "64")
        )
        compact_boundary_layer = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_LAYER",
                "-1",
            )
        )
        compact_all_boundaries = (
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_ALL_BOUNDARIES",
                "0",
            )
            == "1"
        )
        compact_boundary_max_layer = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_BOUNDARY_MAX_LAYER",
                "-1",
            )
        )
        compact_gdn_boundary_layer = int(
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_GDN_BOUNDARY_LAYER",
                "-1",
            )
        )
        all_internal_boundaries = (
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_ALL_INTERNAL_BOUNDARIES",
                "0",
            )
            == "1"
        )
        internal_boundary_layers_raw = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_INTERNAL_BOUNDARY_LAYERS",
            "",
        )
        internal_boundary_layers = tuple(
            dict.fromkeys(
                int(value) for value in internal_boundary_layers_raw.split(",") if value
            )
        )
        compact_full_stages_raw = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_FULL_ATTENTION_STAGES",
            "",
        )
        requested_compact_full_stages = {
            value for value in compact_full_stages_raw.split(",") if value
        }
        unknown_compact_full_stages = requested_compact_full_stages.difference(
            COMPACT_FULL_ATTENTION_STAGES
        )
        if unknown_compact_full_stages:
            raise ValueError(
                "Unknown compact full-attention trace stages: "
                f"{sorted(unknown_compact_full_stages)}"
            )
        compact_full_attention_stages = tuple(
            stage
            for stage in COMPACT_FULL_ATTENTION_STAGES
            if (
                stage in requested_compact_full_stages
                if requested_compact_full_stages
                else True
            )
        )
        if not compact_full_attention_stages:
            raise ValueError("Compact full-attention trace requires stages")
        compact_gdn_stages_raw = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_GDN_STAGES", ""
        )
        requested_compact_gdn_stages = {
            value for value in compact_gdn_stages_raw.split(",") if value
        }
        unknown_compact_gdn_stages = requested_compact_gdn_stages.difference(
            COMPACT_GDN_STAGES
        )
        if unknown_compact_gdn_stages:
            raise ValueError(
                "Unknown compact GDN trace stages: "
                f"{sorted(unknown_compact_gdn_stages)}"
            )
        compact_gdn_stages = tuple(
            stage
            for stage in COMPACT_GDN_STAGES
            if not requested_compact_gdn_stages or stage in requested_compact_gdn_stages
        )
        compact_mlp_stages_raw = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_COMPACT_MLP_STAGES", ""
        )
        requested_compact_mlp_stages = {
            value for value in compact_mlp_stages_raw.split(",") if value
        }
        unknown_compact_mlp_stages = requested_compact_mlp_stages.difference(
            COMPACT_MLP_STAGES
        )
        if unknown_compact_mlp_stages:
            raise ValueError(
                "Unknown compact MLP trace stages: "
                f"{sorted(unknown_compact_mlp_stages)}"
            )
        compact_mlp_stages = tuple(
            stage
            for stage in COMPACT_MLP_STAGES
            if not requested_compact_mlp_stages or stage in requested_compact_mlp_stages
        )
        if bool(
            {"dcp_output_pack", "dcp_lse_pack"}.intersection(
                compact_full_attention_stages
            )
        ) and not {"dcp_output_pack", "dcp_lse_pack"}.issubset(
            compact_full_attention_stages
        ):
            raise ValueError(
                "Compact DCP output and LSE packs must be requested together"
            )
        fingerprint_outputs = (
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_FINGERPRINT_OUTPUTS",
                "0",
            )
            == "1"
        )
        request_prefix = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_PREFIX",
            "",
        )
        request_chunks = (
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_REQUEST_CHUNKS",
                "0",
            )
            == "1"
        )
        sequence_boundary_layers_raw = os.environ.get(
            "AG2_VLLM_AUX_HIDDEN_TRACE_SEQUENCE_BOUNDARY_LAYERS",
            "",
        )
        sequence_boundary_layers = tuple(
            dict.fromkeys(
                int(value) for value in sequence_boundary_layers_raw.split(",") if value
            )
        )
        if not layers or any(layer < 0 for layer in layers):
            raise ValueError("Aux hidden trace layers must be nonnegative")
        if (
            not positions or any(value < 0 for value in positions)
        ) and not request_chunks:
            raise ValueError(
                "Aux hidden trace requires nonnegative POSITION or POSITIONS"
            )
        if (
            token_id < 0
            and not request_chunks
            and not (request_prefix and compact_rows)
        ):
            raise ValueError(
                "Wildcard TOKEN_ID requires request-filtered compact-row tracing"
            )
        if request_chunks and position < 0:
            raise ValueError(
                "Request-chunk trace requires nonnegative POSITION as its "
                "logical capture ceiling"
            )
        if request_chunks and not request_prefix:
            raise ValueError("Request-chunk trace requires REQUEST_PREFIX")
        if request_chunks and not fingerprint_outputs:
            raise ValueError("Request-chunk trace requires fingerprint outputs")
        if request_chunks and compact_rows:
            raise ValueError("Request-chunk trace and compact-row trace are exclusive")
        if any(layer < 0 for layer in sequence_boundary_layers):
            raise ValueError("Sequence boundary layers must be nonnegative")
        if sequence_boundary_layers and not request_chunks:
            raise ValueError("Sequence boundaries require request-chunk tracing")
        if sequence_boundary_layers and not set(sequence_boundary_layers).issubset(
            range(max(layers))
        ):
            raise ValueError(
                "Sequence boundary layers require their layer and following "
                "checkpoint in LAYERS"
            )
        if sequence_boundary_layers and not {
            layer
            for boundary in sequence_boundary_layers
            for layer in (boundary, boundary + 1)
        }.issubset(layers):
            raise ValueError(
                "Sequence boundaries require checkpoints at every selected layer "
                "and its successor"
            )
        if max_matches < 1:
            raise ValueError("Aux hidden trace max matches must be positive")
        if compact_capacity < 1:
            raise ValueError("Aux hidden trace compact capacity must be positive")
        if compact_boundary_layer >= 0 and not compact_rows:
            raise ValueError("Compact layer boundaries require compact-row tracing")
        if compact_all_boundaries and not compact_rows:
            raise ValueError("All compact boundaries require compact-row tracing")
        if compact_all_boundaries and compact_boundary_max_layer < 0:
            compact_boundary_max_layer = max(layers) - 1
        if compact_all_boundaries and compact_boundary_max_layer >= max(layers):
            raise ValueError(
                "Compact boundary max layer requires the following auxiliary "
                "checkpoint: max boundary must be less than max(LAYERS)"
            )
        if compact_gdn_boundary_layer >= 0 and not compact_rows:
            raise ValueError("Compact GDN boundaries require compact-row tracing")
        if all_internal_boundaries and not (compact_rows and compact_all_boundaries):
            raise ValueError(
                "All internal boundaries require compact-row tracing at all layers"
            )
        if internal_boundary_layers and not (compact_rows and compact_all_boundaries):
            raise ValueError(
                "Selected internal boundaries require compact-row tracing at all "
                "selected layers"
            )
        if any(
            layer < 0 or layer >= max(layers) or layer > compact_boundary_max_layer
            for layer in internal_boundary_layers
        ):
            raise ValueError(
                "Selected internal boundary layers require their layer boundary "
                "and following auxiliary checkpoint"
            )
        if compact_gdn_boundary_layer >= 0 and not {
            compact_gdn_boundary_layer,
            compact_gdn_boundary_layer + 1,
        }.issubset(layers):
            raise ValueError(
                "Compact GDN boundaries require auxiliary checkpoints at "
                "the selected layer and the following layer"
            )
        if compact_boundary_layer >= 0 and not {
            compact_boundary_layer,
            compact_boundary_layer + 1,
        }.issubset(layers):
            raise ValueError(
                "Compact layer boundaries require auxiliary checkpoints at "
                "the selected layer and the following layer"
            )
        if attention_boundary_layer < 0:
            raise ValueError("Attention boundary layer must be nonnegative")
        if first_attention_boundary and not {
            attention_boundary_layer,
            attention_boundary_layer + 1,
        }.issubset(layers):
            raise ValueError(
                "Attention boundary trace requires auxiliary checkpoints at "
                "the selected layer and the following layer"
            )
        if full_attention_boundaries and not first_attention_boundary:
            raise ValueError(
                "Full-attention boundaries require the attention boundary trace"
            )
        if full_attention_boundaries and not full_attention_stages:
            raise ValueError("Full-attention boundary stage set must not be empty")
        has_dcp_pack = full_attention_boundaries and any(
            "dcp_" in stage for stage in full_attention_stages
        )
        if has_dcp_pack and dcp_request_tail_rows < 1:
            raise ValueError(
                "DCP auxiliary pack requires positive DCP_REQUEST_TAIL_ROWS capacity"
            )
        if has_dcp_pack and dcp_request_row not in {"first", "tail"}:
            raise ValueError("DCP request-row trace must select first or tail")
        if first_attention_boundary and first_gdn_boundaries:
            raise ValueError(
                "First-attention and first-GDN boundary traces are mutually exclusive"
            )
        if gdn_boundary_layer < 0:
            raise ValueError("GDN boundary layer must be nonnegative")
        packed_compare = (
            packed_compare_prefix_tokens,
            packed_compare_sequence_tokens,
            packed_compare_sequences,
        )
        if any(packed_compare) and not (
            packed_compare_prefix_tokens >= 0
            and packed_compare_sequence_tokens > 0
            and packed_compare_sequences > 1
        ):
            raise ValueError(
                "Packed comparison requires a nonnegative prefix, positive "
                "sequence length, and at least two sequences"
            )
        compact_dcp_attention = (
            compact_rows
            and first_attention_boundary
            and full_attention_boundaries
            and set(full_attention_stages) == {"dcp_output_pack", "dcp_lse_pack"}
            and (
                (
                    compact_all_boundaries
                    and attention_boundary_layer <= compact_boundary_max_layer
                )
                or compact_boundary_layer == attention_boundary_layer
            )
        )
        if compact_rows and (
            (first_attention_boundary and not compact_dcp_attention)
            or first_gdn_boundaries
            or any(packed_compare)
        ):
            raise ValueError("Compact-row tracing supports global checkpoints only")
        if first_gdn_boundaries and not {
            gdn_boundary_layer,
            gdn_boundary_layer + 1,
        }.issubset(layers):
            raise ValueError(
                "GDN boundary trace requires auxiliary checkpoints at "
                "the selected layer and the following layer"
            )
        if gdn_replay_only and first_gdn_boundaries:
            raise ValueError("GDN replay-only and full GDN boundaries are exclusive")
        if gdn_replay_only and not {
            gdn_boundary_layer,
            gdn_boundary_layer + 1,
        }.issubset(layers):
            raise ValueError(
                "GDN replay-only trace requires auxiliary checkpoints at "
                "the selected layer and the following layer"
            )
        return cls(
            output=output,
            layers=layers,
            token_id=token_id,
            position=position,
            positions=positions,
            max_matches_per_query_len=max_matches,
            first_attention_boundary=first_attention_boundary,
            attention_boundary_layer=attention_boundary_layer,
            full_attention_boundaries=full_attention_boundaries,
            full_attention_stages=full_attention_stages,
            first_gdn_boundaries=first_gdn_boundaries,
            gdn_replay_only=gdn_replay_only,
            gdn_boundary_layer=gdn_boundary_layer,
            packed_compare_prefix_tokens=packed_compare_prefix_tokens,
            packed_compare_sequence_tokens=packed_compare_sequence_tokens,
            packed_compare_sequences=packed_compare_sequences,
            dcp_request_tail_rows=dcp_request_tail_rows,
            dcp_request_row=dcp_request_row,
            compact_rows=compact_rows,
            compact_capacity=compact_capacity,
            compact_boundary_layer=compact_boundary_layer,
            compact_all_boundaries=compact_all_boundaries,
            compact_boundary_max_layer=compact_boundary_max_layer,
            compact_gdn_boundary_layer=compact_gdn_boundary_layer,
            all_internal_boundaries=all_internal_boundaries,
            internal_boundary_layers=internal_boundary_layers,
            compact_full_attention_stages=compact_full_attention_stages,
            compact_gdn_stages=compact_gdn_stages,
            compact_mlp_stages=compact_mlp_stages,
            fingerprint_outputs=fingerprint_outputs,
            request_prefix=request_prefix,
            request_chunks=request_chunks,
            sequence_boundary_layers=sequence_boundary_layers,
        )

    def _packed_sequence_diffs(
        self,
        *,
        query_len: int,
        labels: tuple[str, ...],
        aux_hidden_states: list[torch.Tensor],
    ) -> dict[str, list[dict[str, int | float | bool]]] | None:
        sequence_tokens = self.packed_compare_sequence_tokens
        sequence_count = self.packed_compare_sequences
        if sequence_tokens == 0 or sequence_count == 0:
            return None
        prefix_tokens = self.packed_compare_prefix_tokens
        required_tokens = prefix_tokens + sequence_tokens * sequence_count
        if query_len != required_tokens:
            return None

        summaries: dict[str, list[dict[str, int | float | bool]]] = {}
        for label, hidden in zip(labels, aux_hidden_states, strict=True):
            if label.startswith("gdn_replay_") or hidden.shape[0] < required_tokens:
                continue
            reference = hidden[prefix_tokens : prefix_tokens + sequence_tokens]
            comparisons: list[dict[str, int | float | bool]] = []
            for sequence_index in range(1, sequence_count):
                start = prefix_tokens + sequence_index * sequence_tokens
                candidate = hidden[start : start + sequence_tokens]
                unequal = reference.ne(candidate)
                per_token = unequal.reshape(sequence_tokens, -1).any(dim=1)
                differing_tokens = int(per_token.sum().item())
                first_differing_token = (
                    int(per_token.nonzero(as_tuple=False)[0].item())
                    if differing_tokens
                    else -1
                )
                max_abs = (
                    float((reference.float() - candidate.float()).abs().max().item())
                    if differing_tokens
                    else 0.0
                )
                comparisons.append(
                    {
                        "sequence": sequence_index,
                        "exact": differing_tokens == 0,
                        "differing_tokens": differing_tokens,
                        "first_differing_token": first_differing_token,
                        "max_abs": max_abs,
                    }
                )
            summaries[label] = comparisons
        return summaries

    @property
    def enabled(self) -> bool:
        return self.output is not None

    def release_raw_model_refs(self, model: nn.Module) -> int:
        """Release diagnostic intermediates after their packets are returned.

        Broad tracing stores layer-local tensors on ``_ag2_aux_*`` attributes
        long enough for the model to assemble graph outputs.  Keeping those
        Python references past the target forward pins the raw Q/K/V/GDN/MLP
        tensors while the MTP proposer is compiled or executed.  Fingerprint
        outputs are independent tensors at this point, so the raw references
        are no longer part of the observer contract.

        Configuration flags and registered parameters/buffers are never
        touched.  Eager forwards recreate each transient attribute before it
        is consumed; compiled/CUDA-Graph forwards return the packet outputs
        directly and do not depend on Python-side attribute state.
        """
        if not (self.enabled and self.fingerprint_outputs):
            return 0
        released = 0
        for module in model.modules():
            parameters = module._parameters
            buffers = module._buffers
            for name, value in tuple(vars(module).items()):
                if (
                    not name.startswith("_ag2_aux_")
                    or name in parameters
                    or name in buffers
                ):
                    continue
                contains_tensor = isinstance(value, torch.Tensor) or (
                    isinstance(value, (tuple, list))
                    and any(isinstance(item, torch.Tensor) for item in value)
                )
                if contains_tensor:
                    setattr(module, name, None)
                    released += 1
        return released

    def configure_model(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        setter = getattr(model, "set_aux_hidden_state_layers", None)
        if setter is None:
            raise TypeError(
                f"{type(model).__name__} does not expose auxiliary hidden states"
            )
        setter(self.layers)
        if self.request_chunks:
            configured_request_models = []
            for module in model.modules():
                if getattr(module, "aux_hidden_state_layers", None) == self.layers:
                    module._ag2_aux_trace_request_chunks = True
                    configured_request_models.append(type(module).__name__)
            if not configured_request_models:
                raise RuntimeError(
                    "Request-chunk trace could not locate the configured model"
                )
        decoder_layers = {
            int(module.layer_idx): str(module.layer_type)
            for module in model.modules()
            if hasattr(module, "layer_idx")
            and hasattr(module, "layer_type")
            and str(module.layer_type) in {"linear_attention", "full_attention"}
        }
        if (
            self.all_internal_boundaries
            or self.internal_boundary_layers
            or self.sequence_boundary_layers
        ):
            expected = set(range(max(self.layers)))
            required = (
                expected
                if self.all_internal_boundaries
                else set(self.internal_boundary_layers).union(
                    self.sequence_boundary_layers
                )
            )
            if not required.issubset(decoder_layers):
                raise RuntimeError(
                    "Auxiliary internal trace could not resolve required decoder "
                    f"layers: required={sorted(required)} "
                    f"got={sorted(decoder_layers)}"
                )
            self._layer_types = {
                layer: decoder_layers[layer] for layer in sorted(required)
            }
        if self.compact_rows:
            configured_modules = []
            for module in model.modules():
                if getattr(module, "aux_hidden_state_layers", None) == self.layers:
                    module._ag2_aux_trace_compact_position = self.position
                    module._ag2_aux_trace_compact_positions = self.positions
                    module._ag2_aux_trace_compact_capacity = self.compact_capacity
                    module._ag2_aux_trace_compact_boundary_layer = (
                        self.compact_boundary_layer
                    )
                    module._ag2_aux_trace_compact_all_boundaries = (
                        self.compact_all_boundaries
                    )
                    module._ag2_aux_trace_compact_boundary_max_layer = (
                        self.compact_boundary_max_layer
                    )
                    module._ag2_aux_trace_compact_gdn_boundary_layer = (
                        self.compact_gdn_boundary_layer
                    )
                    module._ag2_aux_trace_compact_rows = True
                    configured_modules.append(type(module).__name__)
            if not configured_modules:
                raise RuntimeError(
                    "Compact auxiliary trace could not locate the configured "
                    "EAGLE model"
                )
        logger.warning(
            "Enabled authoritative auxiliary hidden trace for layers=%s "
            "first_attention_boundary=%s attention_boundary_layer=%d "
            "full_attention_boundaries=%s full_attention_stages=%s "
            "first_gdn_boundaries=%s gdn_boundary_layer=%d "
            "gdn_replay_only=%s "
            "compact_rows=%s compact_capacity=%d compact_gdn_layer=%d "
            "request_chunks=%s",
            self.layers,
            self.first_attention_boundary,
            self.attention_boundary_layer,
            self.full_attention_boundaries,
            self.full_attention_stages,
            self.first_gdn_boundaries,
            self.gdn_boundary_layer,
            self.gdn_replay_only,
            self.compact_rows,
            self.compact_capacity,
            self.compact_gdn_boundary_layer,
            self.request_chunks,
        )
        self._write_manifest(get_tp_group().rank_in_group, "initialized")

    def _write_manifest(self, rank: int, state: str) -> None:
        if not self.enabled:
            return
        path = Path(f"{self.output}.rank{rank}.manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "ag2-aux-hidden-trace-manifest-v1",
                    "state": state,
                    "rank": rank,
                    "token_id": self.token_id,
                    "position": self.position,
                    "positions": list(self.positions),
                    "request_prefix": self.request_prefix,
                    "request_chunks": self.request_chunks,
                    "sequence_boundary_layers": list(self.sequence_boundary_layers),
                    "internal_boundary_layers": list(self.internal_boundary_layers),
                    "compact_full_attention_stages": list(
                        self.compact_full_attention_stages
                    ),
                    "compact_gdn_stages": list(self.compact_gdn_stages),
                    "compact_mlp_stages": list(self.compact_mlp_stages),
                    "representation": (
                        "exact_bit_fingerprint_v1"
                        if self.fingerprint_outputs
                        else "raw"
                    ),
                    "labels": list(self.output_labels()),
                    "scopes": self.output_scopes(),
                    "saved_matches": {
                        str(query_len): count
                        for query_len, count in sorted(self._saved_matches.items())
                    },
                    "saved_request_invocations": self._saved_request_invocations,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def output_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        compact_boundary_max_layer = self.compact_boundary_max_layer
        if self.compact_all_boundaries and compact_boundary_max_layer < 0:
            compact_boundary_max_layer = max(self.layers) - 1
        if 0 in self.layers:
            if self.compact_rows:
                labels.extend(
                    (
                        "layer_row_indices.0",
                        "layer_hidden.0",
                        "layer_residual.0",
                    )
                )
            labels.append("layer.0")
        for decoder_layer in range(max(self.layers)):
            if decoder_layer in self.sequence_boundary_layers:
                layer_type = self._layer_types.get(decoder_layer)
                labels.append(f"sequence_input_norm.{decoder_layer}")
                if layer_type == "full_attention":
                    labels.extend(
                        f"sequence_full_{stage}.{decoder_layer}"
                        for stage in COMPACT_FULL_ATTENTION_STAGES
                        if not stage.startswith("dcp_")
                    )
                elif layer_type == "linear_attention":
                    labels.extend(
                        f"sequence_gdn_{stage}.{decoder_layer}"
                        for stage in COMPACT_GDN_STAGES
                    )
                else:
                    raise RuntimeError(
                        f"Unknown sequence boundary layer type at "
                        f"{decoder_layer}: {layer_type}"
                    )
                labels.extend(
                    (
                        f"sequence_post_attention_norm.{decoder_layer}",
                        f"sequence_post_attention_residual.{decoder_layer}",
                    )
                )
                labels.extend(
                    f"sequence_mlp_{stage}.{decoder_layer}"
                    for stage in COMPACT_MLP_STAGES
                )
            if (
                self.compact_all_boundaries
                and decoder_layer <= compact_boundary_max_layer
            ) or decoder_layer == self.compact_boundary_layer:
                labels.extend(
                    (
                        f"compact_boundary_row_indices.{decoder_layer}",
                        f"input_norm.{decoder_layer}",
                        f"attention_output.{decoder_layer}",
                        f"post_attention_norm.{decoder_layer}",
                        f"post_attention_residual.{decoder_layer}",
                    )
                )
            if (
                self.all_internal_boundaries
                or decoder_layer in self.internal_boundary_layers
            ):
                layer_type = self._layer_types.get(decoder_layer)
                if layer_type == "full_attention":
                    labels.extend(
                        f"compact_full_{stage}.{decoder_layer}"
                        for stage in self.compact_full_attention_stages
                    )
                elif layer_type == "linear_attention":
                    labels.extend(
                        f"compact_gdn_{stage}.{decoder_layer}"
                        for stage in self.compact_gdn_stages
                    )
                else:
                    raise RuntimeError(
                        f"Unknown decoder layer type at {decoder_layer}: {layer_type}"
                    )
                labels.extend(
                    f"compact_mlp_{stage}.{decoder_layer}"
                    for stage in self.compact_mlp_stages
                )
            if (
                self.first_attention_boundary
                and decoder_layer == self.attention_boundary_layer
            ):
                if self.full_attention_boundaries:
                    labels.extend(
                        f"full_{stage}.{decoder_layer}"
                        for stage in self.full_attention_stages
                    )
                if not self.compact_rows:
                    labels.extend(
                        (
                            f"post_attention_norm.{decoder_layer}",
                            f"post_attention_residual.{decoder_layer}",
                        )
                    )
            if self.first_gdn_boundaries and decoder_layer == self.gdn_boundary_layer:
                labels.extend(
                    (
                        f"gdn_input_norm.{decoder_layer}",
                        f"gdn_qkvz.{decoder_layer}",
                        f"gdn_ba.{decoder_layer}",
                        f"gdn_replay_float.{decoder_layer}",
                        f"gdn_replay_state.{decoder_layer}",
                        f"gdn_replay_meta.{decoder_layer}",
                        f"gdn_core.{decoder_layer}",
                        f"gdn_gated_norm.{decoder_layer}",
                        f"gdn_output_parallel.{decoder_layer}",
                        f"gdn_output.{decoder_layer}",
                    )
                )
            if self.gdn_replay_only and decoder_layer == self.gdn_boundary_layer:
                labels.extend(
                    (
                        f"gdn_replay_float.{decoder_layer}",
                        f"gdn_replay_state.{decoder_layer}",
                        f"gdn_replay_meta.{decoder_layer}",
                    )
                )
            if (
                not (
                    self.all_internal_boundaries
                    or decoder_layer in self.internal_boundary_layers
                )
                and decoder_layer == self.compact_gdn_boundary_layer
            ):
                labels.extend(
                    (
                        f"compact_gdn_qkvz.{decoder_layer}",
                        f"compact_gdn_ba.{decoder_layer}",
                        f"compact_gdn_core.{decoder_layer}",
                        f"compact_gdn_gated_norm.{decoder_layer}",
                        f"compact_gdn_output_parallel.{decoder_layer}",
                        f"compact_gdn_output.{decoder_layer}",
                    )
                )
            checkpoint = decoder_layer + 1
            if checkpoint in self.layers:
                if self.compact_rows:
                    labels.extend(
                        (
                            f"layer_row_indices.{checkpoint}",
                            f"layer_hidden.{checkpoint}",
                            f"layer_residual.{checkpoint}",
                        )
                    )
                labels.append(f"layer.{checkpoint}")
        return tuple(labels)

    def output_scopes(self) -> dict[str, str]:
        return {
            label: (
                "rank_local"
                if (label.startswith("full_") and not label.startswith("full_output."))
                or label.startswith(
                    (
                        "gdn_qkvz.",
                        "gdn_ba.",
                        "gdn_replay_",
                        "gdn_core.",
                        "gdn_gated_norm.",
                        "gdn_output_parallel.",
                        "compact_gdn_qkvz.",
                        "compact_gdn_ba.",
                        "compact_gdn_core.",
                        "compact_gdn_gated_norm.",
                        "compact_gdn_output_parallel.",
                        "compact_full_qkv.",
                        "compact_full_q.",
                        "compact_full_k.",
                        "compact_full_v.",
                        "compact_full_gate.",
                        "compact_full_core.",
                        "compact_full_gated.",
                        "compact_full_output_parallel.",
                        "compact_mlp_gate_up.",
                        "compact_mlp_activation.",
                        "compact_mlp_down_parallel.",
                        "sequence_gdn_qkvz.",
                        "sequence_gdn_ba.",
                        "sequence_gdn_core.",
                        "sequence_gdn_gated_norm.",
                        "sequence_gdn_output_parallel.",
                        "sequence_full_qkv.",
                        "sequence_full_q.",
                        "sequence_full_k.",
                        "sequence_full_v.",
                        "sequence_full_gate.",
                        "sequence_full_core.",
                        "sequence_full_gated.",
                        "sequence_full_output_parallel.",
                        "sequence_mlp_gate_up.",
                        "sequence_mlp_activation.",
                        "sequence_mlp_down_parallel.",
                    )
                )
                else "global"
            )
            for label in self.output_labels()
        }

    def maybe_save(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_len: int,
        cudagraph_mode: str,
        aux_hidden_states: list[torch.Tensor],
        req_ids: list[str] | None = None,
        query_start_loc: list[int] | None = None,
        num_scheduled_tokens: list[int] | None = None,
        num_computed_tokens: list[int] | None = None,
        slot_mappings_by_layer: dict[str, torch.Tensor] | None = None,
    ) -> None:
        invocation = self._invocation
        self._invocation += 1
        if self.request_chunks:
            self._maybe_save_request_invocation(
                invocation=invocation,
                input_ids=input_ids,
                positions=positions,
                query_len=query_len,
                cudagraph_mode=cudagraph_mode,
                aux_hidden_states=aux_hidden_states,
                req_ids=req_ids,
                query_start_loc=query_start_loc,
                num_scheduled_tokens=num_scheduled_tokens,
                num_computed_tokens=num_computed_tokens,
                slot_mappings_by_layer=slot_mappings_by_layer,
            )
            return
        saved = self._saved_matches.get(query_len, 0)
        if not self.enabled or saved >= self.max_matches_per_query_len:
            return
        labels = self.output_labels()
        if len(aux_hidden_states) != len(labels):
            raise RuntimeError(
                "Aux hidden trace output count does not match configured outputs: "
                f"{len(aux_hidden_states)} != {len(labels)}"
            )
        has_dcp_pack = any("dcp_output_pack" in label for label in labels)
        hidden_by_label = dict(zip(labels, aux_hidden_states, strict=True))

        flat_positions = positions[0] if positions.ndim == 2 else positions
        token_values = input_ids[:query_len].detach().cpu().tolist()
        position_values = flat_positions[:query_len].detach().cpu().tolist()
        capture_positions = set(self.positions)
        matches = [
            row
            for row, (token_id, position) in enumerate(
                zip(token_values, position_values)
            )
            if (self.token_id < 0 or token_id == self.token_id)
            and position in capture_positions
        ]
        if self.request_prefix:
            if req_ids is None or query_start_loc is None:
                raise RuntimeError(
                    "Request-filtered auxiliary trace requires request provenance"
                )
            matches = [
                row
                for row in matches
                if req_ids[bisect_right(query_start_loc, row) - 1].startswith(
                    self.request_prefix
                )
            ]
        if not matches:
            return

        torch.cuda.current_stream().synchronize()
        rank = get_tp_group().rank_in_group
        selected_matches = matches[: self.max_matches_per_query_len - saved]
        packed_sequence_diffs = self._packed_sequence_diffs(
            query_len=query_len,
            labels=labels,
            aux_hidden_states=aux_hidden_states,
        )
        for compact_index, row in enumerate(selected_matches):
            occurrence = self._saved_matches.get(query_len, 0)
            path = Path(f"{self.output}.q{query_len}.occ{occurrence}.rank{rank}.pt")
            if path.exists():
                raise RuntimeError(f"Refusing to overwrite aux hidden trace: {path}")
            request_provenance = None
            if req_ids is not None or query_start_loc is not None:
                if req_ids is None or query_start_loc is None:
                    raise RuntimeError(
                        "Aux hidden trace request provenance requires both "
                        "req_ids and query_start_loc"
                    )
                if len(query_start_loc) != len(req_ids) + 1:
                    raise RuntimeError(
                        "Aux hidden trace request provenance has inconsistent "
                        "request boundaries"
                    )
                request_index = bisect_right(query_start_loc, row) - 1
                if not 0 <= request_index < len(req_ids):
                    raise RuntimeError(
                        f"Matched row {row} is outside request boundaries "
                        f"{query_start_loc}"
                    )
                request_provenance = {
                    "req_id": req_ids[request_index],
                    "request_index": request_index,
                    "request_row_offset": row - query_start_loc[request_index],
                    "query_start": query_start_loc[request_index],
                    "query_end": query_start_loc[request_index + 1],
                    "num_scheduled_tokens": (
                        num_scheduled_tokens[request_index]
                        if num_scheduled_tokens is not None
                        else None
                    ),
                    "num_computed_tokens": (
                        num_computed_tokens[request_index]
                        if num_computed_tokens is not None
                        else None
                    ),
                    "slot_mappings": {
                        layer_name: int(slot_mapping[row].item())
                        for layer_name, slot_mapping in (
                            slot_mappings_by_layer or {}
                        ).items()
                    },
                }

            def select_layer(
                label: str,
                hidden: torch.Tensor,
                *,
                token_row: int,
                provenance: dict[str, object] | None,
            ) -> torch.Tensor:
                if label.startswith(
                    ("layer_row_indices.", "compact_boundary_row_indices.")
                ):
                    return hidden
                if (
                    self.compact_rows
                    and "dcp_" not in label
                    and label.startswith(
                        (
                            "layer.",
                            "layer_hidden.",
                            "layer_residual.",
                            "input_norm.",
                            "attention_output.",
                            "post_attention_norm.",
                            "post_attention_residual.",
                            "compact_gdn_",
                            "compact_full_",
                            "compact_mlp_",
                        )
                    )
                ):
                    checkpoint = label.rsplit(".", 1)[1]
                    row_label = (
                        f"compact_boundary_row_indices.{checkpoint}"
                        if label.startswith(
                            ("compact_gdn_", "compact_full_", "compact_mlp_")
                        )
                        or label.startswith(
                            (
                                "input_norm.",
                                "attention_output.",
                                "post_attention_norm.",
                                "post_attention_residual.",
                            )
                        )
                        else f"layer_row_indices.{checkpoint}"
                    )
                    row_indices = hidden_by_label[row_label]
                    selected = row_indices.eq(token_row).nonzero(as_tuple=False)
                    if selected.numel() != 1:
                        raise RuntimeError(
                            "Compact auxiliary trace did not retain matched row "
                            f"{token_row} at checkpoint {checkpoint}: "
                            f"indices={row_indices.detach().cpu().tolist()}"
                        )
                    return hidden[int(selected[0].item())]
                if label.startswith("gdn_replay_"):
                    return hidden
                if any(
                    name in label
                    for name in (
                        "dcp_output_pack",
                        "dcp_lse_pack",
                        "dcp_kv_history_pack",
                        "dcp_kv_history_meta",
                        "dcp_kv_page_indices_pack",
                        "dcp_observer_meta",
                        "dcp_current_kv_pack",
                        "dcp_scale_pack",
                    )
                ):
                    if provenance is None:
                        raise RuntimeError(
                            "DCP request-tail pack requires request provenance"
                        )
                    request_index = int(provenance["request_index"])
                    if request_index >= hidden.shape[0]:
                        raise RuntimeError(
                            "DCP request-tail pack does not contain request "
                            f"index {request_index}"
                        )
                    return hidden[request_index]
                if has_dcp_pack and label.startswith(("full_q.", "full_k.", "full_v.")):
                    if provenance is None:
                        raise RuntimeError(
                            "DCP request-slice input requires request provenance"
                        )
                    query_start = int(provenance["query_start"])
                    query_end = int(provenance["query_end"])
                    return hidden[query_start:query_end]
                return hidden[token_row]

            torch.save(
                {
                    "schema_version": AG2_AUX_TRACE_SCHEMA_VERSION,
                    "rank": rank,
                    "query_len": query_len,
                    "occurrence": occurrence,
                    "row": row,
                    "cudagraph_mode": cudagraph_mode,
                    "token_id": int(token_values[row]),
                    "capture_token_id": self.token_id,
                    "position": int(position_values[row]),
                    "capture_positions": list(self.positions),
                    "input_ids": input_ids[:query_len].detach().cpu().clone(),
                    "positions": flat_positions[:query_len].detach().cpu().clone(),
                    "request_provenance": request_provenance,
                    "observer_provenance": {
                        "dcp_request_row": self.dcp_request_row,
                        "full_attention_stages": list(self.full_attention_stages),
                    },
                    "layers": {
                        label: select_layer(
                            label,
                            hidden,
                            token_row=row,
                            provenance=request_provenance,
                        )
                        .detach()
                        .cpu()
                        .clone()
                        for label, hidden in zip(labels, aux_hidden_states, strict=True)
                    },
                    "scopes": self.output_scopes(),
                    "packed_sequence_diffs": packed_sequence_diffs,
                },
                path,
            )
            self._saved_matches[query_len] = occurrence + 1
            self._write_manifest(rank, "active")
            logger.warning(
                "Saved authoritative aux hidden trace rank=%d qlen=%d "
                "occurrence=%d row=%d token=%d position=%d mode=%s path=%s",
                rank,
                query_len,
                occurrence,
                row,
                int(token_values[row]),
                int(position_values[row]),
                cudagraph_mode,
                path,
            )

    def _maybe_save_request_invocation(
        self,
        *,
        invocation: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_len: int,
        cudagraph_mode: str,
        aux_hidden_states: list[torch.Tensor],
        req_ids: list[str] | None,
        query_start_loc: list[int] | None,
        num_scheduled_tokens: list[int] | None,
        num_computed_tokens: list[int] | None,
        slot_mappings_by_layer: dict[str, torch.Tensor] | None,
    ) -> None:
        """Persist one complete target invocation for a selected request family.

        The sequence-wide diagnostic needs the earlier chunks that created a
        resumed GDN state. Saving only the invocation containing one selected
        token would retain the symptom and discard its causal history.
        """
        if self._saved_request_invocations >= self.max_matches_per_query_len:
            return
        if req_ids is None or query_start_loc is None:
            raise RuntimeError("Request-chunk trace requires request provenance")
        if len(query_start_loc) != len(req_ids) + 1:
            raise RuntimeError("Request-chunk trace has inconsistent boundaries")
        if num_scheduled_tokens is None or num_computed_tokens is None:
            raise RuntimeError("Request-chunk trace requires scheduler provenance")

        flat_positions = positions[0] if positions.ndim == 2 else positions
        selected_request_indices: list[int] = []
        for request_index, req_id in enumerate(req_ids):
            if not req_id.startswith(self.request_prefix):
                continue
            start = query_start_loc[request_index]
            end = query_start_loc[request_index + 1]
            if start >= end:
                continue
            request_positions = flat_positions[start:end]
            if bool(request_positions.le(self.position).any().item()):
                selected_request_indices.append(request_index)
        if not selected_request_indices:
            return

        labels = self.output_labels()
        if len(aux_hidden_states) != len(labels):
            raise RuntimeError(
                "Request-chunk trace output count does not match schema: "
                f"{len(aux_hidden_states)} != {len(labels)}"
            )
        torch.cuda.current_stream().synchronize()
        rank = get_tp_group().rank_in_group
        path = Path(f"{self.output}.inv{invocation:06d}.rank{rank}.pt")
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite request-chunk trace: {path}")

        requests: list[dict[str, object]] = []
        for request_index, req_id in enumerate(req_ids):
            start = query_start_loc[request_index]
            end = query_start_loc[request_index + 1]
            requests.append(
                {
                    "req_id": req_id,
                    "request_index": request_index,
                    "selected": request_index in selected_request_indices,
                    "query_start": start,
                    "query_end": end,
                    "num_scheduled_tokens": int(num_scheduled_tokens[request_index]),
                    "num_computed_tokens": int(num_computed_tokens[request_index]),
                    "slot_mappings": {
                        layer_name: [
                            int(value)
                            for value in slot_mapping[start:end].detach().cpu().tolist()
                        ]
                        for layer_name, slot_mapping in (
                            slot_mappings_by_layer or {}
                        ).items()
                    },
                }
            )

        layers_cpu = {
            label: hidden.detach().cpu().clone()
            for label, hidden in zip(labels, aux_hidden_states, strict=True)
        }
        # Replay buffers are fixed-capacity graph outputs. Persist only the
        # authoritative live prefix and rewrite the conv offset to the compact
        # artifact layout; otherwise every small scheduler wave would copy the
        # full MaxX reserve to disk.
        for label in labels:
            if not label.startswith("gdn_replay_meta."):
                continue
            layer = label.rsplit(".", 1)[1]
            meta = layers_cpu[label].to(torch.int64).clone()
            status = int(meta[0])
            float_label = f"gdn_replay_float.{layer}"
            state_label = f"gdn_replay_state.{layer}"
            if status == 1:
                float_length = sum(int(value) for value in meta[1:6])
                layers_cpu[float_label] = layers_cpu[float_label][:float_length].clone()
            if status in (1, 2):
                state = layers_cpu[state_label]
                initial_length = max(0, int(meta[6]))
                computed_length = max(0, int(meta[38]))
                readback_length = max(0, int(meta[39]))
                predecode_length = max(0, int(meta[41]))
                state_segments = [state[:initial_length]]
                if computed_length:
                    state_segments.append(
                        state[initial_length : initial_length + computed_length]
                    )
                if readback_length:
                    start = initial_length + computed_length
                    state_segments.append(state[start : start + readback_length])
                if predecode_length:
                    start = 3 * predecode_length
                    state_segments.append(state[start : start + predecode_length])
                conv_length = max(0, int(meta[42]))
                old_conv_offset = max(0, int(meta[43]))
                new_conv_offset = sum(segment.numel() for segment in state_segments)
                if conv_length:
                    state_segments.extend(
                        (
                            state[old_conv_offset : old_conv_offset + conv_length],
                            state[
                                old_conv_offset + conv_length : old_conv_offset
                                + 2 * conv_length
                            ],
                        )
                    )
                layers_cpu[state_label] = torch.cat(state_segments).clone()
                meta[43] = new_conv_offset
                layers_cpu[label] = meta

        torch.save(
            {
                "schema_version": AG2_AUX_TRACE_SCHEMA_VERSION,
                "schema": (
                    "ag2-upstream-sequence-fingerprint-v1"
                    if self.sequence_boundary_layers
                    else "ag2-gdn-sequence-causal-packet-v1"
                ),
                "rank": rank,
                "invocation": invocation,
                "query_len": query_len,
                "cudagraph_mode": cudagraph_mode,
                "capture_position_ceiling": self.position,
                "selected_request_indices": selected_request_indices,
                "input_ids": input_ids[:query_len].detach().cpu().clone(),
                "positions": flat_positions[:query_len].detach().cpu().clone(),
                "requests": requests,
                "layers": layers_cpu,
                "scopes": self.output_scopes(),
            },
            path,
        )
        self._saved_request_invocations += 1
        self._write_manifest(rank, "active")
        logger.warning(
            "Saved request sequence packet rank=%d invocation=%d qlen=%d "
            "selected_requests=%s mode=%s path=%s",
            rank,
            invocation,
            query_len,
            selected_request_indices,
            cudagraph_mode,
            path,
        )
