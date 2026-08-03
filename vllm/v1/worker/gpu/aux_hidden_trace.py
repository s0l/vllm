# SPDX-License-Identifier: Apache-2.0
"""Authoritative auxiliary-hidden-state tracing for narrow diagnostics."""

from __future__ import annotations

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

AG2_AUX_TRACE_SCHEMA_VERSION = 2


@dataclass
class AuxHiddenTrace:
    output: str | None = None
    layers: tuple[int, ...] = ()
    token_id: int = -1
    position: int = -1
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
    _saved_matches: dict[int, int] = field(default_factory=dict)

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
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_GDN_REPLAY_ONLY", "0")
            == "1"
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
        if not layers or any(layer < 0 for layer in layers):
            raise ValueError("Aux hidden trace layers must be nonnegative")
        if token_id < 0 or position < 0:
            raise ValueError(
                "Aux hidden trace requires nonnegative TOKEN_ID and POSITION"
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
        if (
            compact_all_boundaries
            and compact_boundary_max_layer >= max(layers)
        ):
            raise ValueError(
                "Compact boundary max layer requires the following auxiliary "
                "checkpoint: max boundary must be less than max(LAYERS)"
            )
        if compact_gdn_boundary_layer >= 0 and not compact_rows:
            raise ValueError("Compact GDN boundaries require compact-row tracing")
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
            and set(full_attention_stages)
            == {"dcp_output_pack", "dcp_lse_pack"}
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

    def configure_model(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        setter = getattr(model, "set_aux_hidden_state_layers", None)
        if setter is None:
            raise TypeError(
                f"{type(model).__name__} does not expose auxiliary hidden states"
            )
        setter(self.layers)
        if self.compact_rows:
            configured_modules = []
            for module in model.modules():
                if getattr(module, "aux_hidden_state_layers", None) == self.layers:
                    module._ag2_aux_trace_compact_position = self.position
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
            "compact_rows=%s compact_capacity=%d compact_gdn_layer=%d",
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
        )

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
            if decoder_layer == self.compact_gdn_boundary_layer:
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
        matches = [
            row
            for row, (token_id, position) in enumerate(
                zip(token_values, position_values)
            )
            if token_id == self.token_id and position == self.position
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
                if self.compact_rows and label.startswith(
                    (
                        "layer.",
                        "layer_hidden.",
                        "layer_residual.",
                        "input_norm.",
                        "attention_output.",
                        "post_attention_norm.",
                        "post_attention_residual.",
                        "compact_gdn_",
                    )
                ):
                    checkpoint = label.rsplit(".", 1)[1]
                    row_label = (
                        f"compact_boundary_row_indices.{checkpoint}"
                        if label.startswith("compact_gdn_")
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
                    "token_id": self.token_id,
                    "position": self.position,
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
            logger.warning(
                "Saved authoritative aux hidden trace rank=%d qlen=%d "
                "occurrence=%d row=%d token=%d position=%d mode=%s path=%s",
                rank,
                query_len,
                occurrence,
                row,
                self.token_id,
                self.position,
                cudagraph_mode,
                path,
            )
