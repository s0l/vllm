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
    "core",
    "gated",
    "output_parallel",
    "output",
)


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
    full_attention_stages: tuple[str, ...] = FULL_ATTENTION_STAGE_ORDER
    first_gdn_boundaries: bool = False
    gdn_boundary_layer: int = 0
    packed_compare_prefix_tokens: int = 0
    packed_compare_sequence_tokens: int = 0
    packed_compare_sequences: int = 0
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
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_ATTENTION_BOUNDARY_LAYER", "0"
            )
        )
        full_attention_boundaries = (
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_FULL_ATTENTION_BOUNDARIES", "0"
            )
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
            if not requested_full_attention_stages
            or stage in requested_full_attention_stages
        )
        first_gdn_boundaries = (
            os.environ.get("AG2_VLLM_AUX_HIDDEN_TRACE_FIRST_GDN_BOUNDARIES", "0")
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
            os.environ.get(
                "AG2_VLLM_AUX_HIDDEN_TRACE_PACKED_COMPARE_SEQUENCES", "0"
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
        if first_gdn_boundaries and not {
            gdn_boundary_layer,
            gdn_boundary_layer + 1,
        }.issubset(layers):
            raise ValueError(
                "GDN boundary trace requires auxiliary checkpoints at "
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
            gdn_boundary_layer=gdn_boundary_layer,
            packed_compare_prefix_tokens=packed_compare_prefix_tokens,
            packed_compare_sequence_tokens=packed_compare_sequence_tokens,
            packed_compare_sequences=packed_compare_sequences,
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
        logger.warning(
            "Enabled authoritative auxiliary hidden trace for layers=%s "
            "first_attention_boundary=%s attention_boundary_layer=%d "
            "full_attention_boundaries=%s full_attention_stages=%s "
            "first_gdn_boundaries=%s gdn_boundary_layer=%d",
            self.layers,
            self.first_attention_boundary,
            self.attention_boundary_layer,
            self.full_attention_boundaries,
            self.full_attention_stages,
            self.first_gdn_boundaries,
            self.gdn_boundary_layer,
        )

    def output_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        if 0 in self.layers:
            labels.append("layer.0")
        for decoder_layer in range(max(self.layers)):
            if (
                self.first_attention_boundary
                and decoder_layer == self.attention_boundary_layer
            ):
                if self.full_attention_boundaries:
                    labels.extend(
                        f"full_{stage}.{decoder_layer}"
                        for stage in self.full_attention_stages
                    )
                labels.extend(
                    (
                        f"post_attention_norm.{decoder_layer}",
                        f"post_attention_residual.{decoder_layer}",
                    )
                )
            if (
                self.first_gdn_boundaries
                and decoder_layer == self.gdn_boundary_layer
            ):
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
            checkpoint = decoder_layer + 1
            if checkpoint in self.layers:
                labels.append(f"layer.{checkpoint}")
        return tuple(labels)

    def output_scopes(self) -> dict[str, str]:
        return {
            label: (
                "rank_local"
                if (
                    label.startswith("full_")
                    and not label.startswith("full_output.")
                )
                or label.startswith(
                    (
                        "gdn_qkvz.",
                        "gdn_ba.",
                        "gdn_replay_",
                        "gdn_core.",
                        "gdn_gated_norm.",
                        "gdn_output_parallel.",
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

        has_dcp_pack = any("dcp_output_pack" in label for label in labels)
        if has_dcp_pack and query_len not in (1, 3):
            # Prompt prefill can contain the same token/position at an
            # arbitrary row, but the packed diagnostic is defined only for
            # the matched qlen1 decode and qlen3 verification lanes.
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
            if has_dcp_pack and row != 0:
                raise RuntimeError(
                    "Authoritative DCP pack is intentionally row-zero only; "
                    f"matched token was at row {row}. Run the isolated replay "
                    "instead of widening the graph output."
                )
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
            torch.save(
                {
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
                    "layers": {
                        label: (
                            hidden
                            if label.startswith("gdn_replay_")
                            else hidden[row]
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
