# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, default-off consumed MTP-layer capture for causal diagnosis."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

_SCHEMA = "ag2-mtp-layer-capture-v1"

MTP_LAYER_TRACE_FIELDS = frozenset(
    {
        "input_ids",
        "positions",
        "embedding",
        "embedding_norm",
        "feedback_input",
        "feedback_norm",
        "fc_output",
        "layer_input",
        "input_norm",
        "attention_qkv",
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_gate",
        "attention_core",
        "attention_gated",
        "attention_output_parallel",
        "attention_output",
        "attention_dcp_output_pack",
        "attention_dcp_lse_pack",
        "attention_dcp_kv_history_pack",
        "attention_dcp_kv_history_meta",
        "attention_dcp_kv_page_indices_pack",
        "attention_dcp_observer_meta",
        "attention_dcp_current_kv_pack",
        "attention_dcp_scale_pack",
        "post_attention_norm",
        "post_attention_residual",
        "mlp_output",
        "final_norm",
    }
)


class Ag2MtpLayerCapture:
    def __init__(
        self,
        output: str,
        rank: int,
        pos_min: int,
        pos_max: int,
        max_records: int,
    ) -> None:
        self.output = output
        self.rank = rank
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.max_records = max_records
        self.records_written = 0
        self.pending: list[dict] = []
        self.req_ids: list[str] = []
        self.idx_mapping = torch.empty(0, dtype=torch.int32)
        self.active = False
        if rank == 0:
            manifest = Path(f"{output}.manifest.json")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema": _SCHEMA,
                        "state": "initialized",
                        "pos_min": pos_min,
                        "pos_max": pos_max,
                        "max_records": max_records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    @classmethod
    def from_env(cls, rank: int) -> Ag2MtpLayerCapture | None:
        output = os.environ.get("AG2_VLLM_MTP_LAYER_CAPTURE_OUTPUT")
        if not output:
            return None
        pos_min = int(os.environ.get("AG2_VLLM_MTP_LAYER_CAPTURE_POS_MIN", "-1"))
        pos_max = int(os.environ.get("AG2_VLLM_MTP_LAYER_CAPTURE_POS_MAX", "-1"))
        max_records = int(os.environ.get("AG2_VLLM_MTP_LAYER_CAPTURE_RECORDS", "64"))
        if pos_min < 0 or pos_max < pos_min or max_records <= 0:
            raise ValueError("invalid AG2 MTP layer capture bounds")
        return cls(output, rank, pos_min, pos_max, max_records)

    def begin(
        self,
        *,
        req_ids: list[str],
        idx_mapping: torch.Tensor,
        num_reqs: int,
    ) -> None:
        if self.active:
            raise RuntimeError("AG2 MTP layer capture begin called before finalize")
        self.active = True
        self.pending = []
        self.req_ids = list(req_ids[:num_reqs])
        self.idx_mapping = idx_mapping[:num_reqs].detach().cpu().clone()

    def stage(
        self,
        *,
        proposal_step: int,
        num_reqs: int,
        trace: dict[str, torch.Tensor] | None,
        dispatch_positions: torch.Tensor,
        trace_row_indices: torch.Tensor | None = None,
    ) -> None:
        if not self.active:
            return
        if self.records_written >= self.max_records:
            return
        if trace is None:
            raise RuntimeError("AG2 MTP layer trace was not published by graph replay")
        if "positions" not in trace:
            raise RuntimeError("AG2 MTP layer trace is missing positions")
        positions = dispatch_positions[:num_reqs].detach().cpu().clone()
        if not bool(((positions >= self.pos_min) & (positions <= self.pos_max)).any()):
            return
        captured: dict[str, torch.Tensor | int] = {
            "proposal_step": proposal_step,
            "dispatch_positions": positions,
            "positions": positions,
        }
        for name, value in trace.items():
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"AG2 MTP trace field {name} is not a tensor")
            # DCP packs are request-major. Dense qlen=1 tensors are also
            # request-major, while qlen4 prefill tensors are token-major and
            # must select the exact last-token rows consumed by sampling.
            if trace_row_indices is not None and not name.startswith("attention_dcp_"):
                selected = torch.index_select(
                    value,
                    0,
                    trace_row_indices[:num_reqs].to(value.device, dtype=torch.long),
                )
            else:
                selected = value[:num_reqs]
            captured["graph_returned_positions" if name == "positions" else name] = (
                selected.detach().cpu().clone()
            )
        self.pending.append(captured)

    def finalize(self, *, draft_tokens: torch.Tensor, num_reqs: int) -> None:
        if not self.active:
            raise RuntimeError("AG2 MTP layer capture finalize called without begin")
        self.active = False
        if not self.pending or self.records_written >= self.max_records:
            self.pending = []
            return
        in_window = False
        for step in self.pending:
            positions = step["dispatch_positions"]
            assert isinstance(positions, torch.Tensor)
            in_window = in_window or bool(
                ((positions >= self.pos_min) & (positions <= self.pos_max)).any()
            )
        if not in_window:
            self.pending = []
            return
        path = Path(
            f"{self.output}.rank{self.rank}.record{self.records_written:04d}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": _SCHEMA,
                "rank": self.rank,
                "record": self.records_written,
                "req_ids": self.req_ids,
                "idx_mapping": self.idx_mapping,
                "draft_tokens": draft_tokens[:num_reqs].detach().cpu().clone(),
                "steps": self.pending,
            },
            path,
        )
        self.records_written += 1
        self.pending = []
