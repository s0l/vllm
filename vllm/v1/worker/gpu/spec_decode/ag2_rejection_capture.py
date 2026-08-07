# SPDX-License-Identifier: Apache-2.0
"""Bounded exact-boundary capture for stochastic MTP rejection sampling.

This observer is diagnostic-only. It synchronously saves the complete processed
target and active draft logits only when a target row falls inside an explicit
absolute-position window. The artifact is therefore sufficient for exact
offline replay without paying full-vocabulary capture cost on every decode
step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SCHEMA = "ag2-rejection-capture-v2"


class Ag2RejectionCapture:
    def __init__(
        self,
        output: str,
        pos_min: int,
        pos_max: int,
        max_records: int,
    ) -> None:
        if pos_min < 0 or pos_max < pos_min or max_records <= 0:
            raise ValueError(
                "AG2 rejection capture requires 0 <= POS_MIN <= POS_MAX and "
                "RECORDS > 0"
            )
        self.output = output
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.max_records = max_records
        self.records_written = 0
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
    def from_env(cls) -> Ag2RejectionCapture | None:
        output = os.environ.get("AG2_VLLM_REJECTION_CAPTURE_OUTPUT")
        if not output:
            return None
        return cls(
            output=output,
            pos_min=int(os.environ.get("AG2_VLLM_REJECTION_CAPTURE_POS_MIN", "-1")),
            pos_max=int(os.environ.get("AG2_VLLM_REJECTION_CAPTURE_POS_MAX", "-1")),
            max_records=int(
                os.environ.get("AG2_VLLM_REJECTION_CAPTURE_RECORDS", "8")
            ),
        )

    def capture(
        self,
        *,
        rank: int,
        req_ids: list[str],
        processed_target_logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        positions: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        use_fp64: bool,
        use_block_verification: bool,
        sampling_stages: dict[str, torch.Tensor] | None = None,
        sampling_state: dict[str, Any] | None = None,
    ) -> None:
        if rank != 0 or self.records_written >= self.max_records:
            return
        pos_cpu = positions.detach().cpu()
        selected = (pos_cpu >= self.pos_min) & (pos_cpu <= self.pos_max)
        if not bool(selected.any()):
            return
        num_reqs = len(req_ids)
        if cu_num_logits.numel() != num_reqs + 1:
            raise RuntimeError("AG2 rejection capture request/logit boundary mismatch")
        state_idx = idx_mapping[:num_reqs].long()
        if bool((state_idx < 0).any()):
            raise RuntimeError("AG2 rejection capture saw a negative active state")
        active_draft_logits = None
        if draft_logits is not None:
            active_draft_logits = draft_logits[state_idx].detach().cpu()
        payload: dict[str, Any] = {
            "schema": _SCHEMA,
            "record": self.records_written,
            "window": [self.pos_min, self.pos_max],
            "req_ids": list(req_ids),
            "processed_target_logits": processed_target_logits.detach().cpu(),
            "active_draft_logits": active_draft_logits,
            "draft_sampled": draft_sampled.detach().cpu(),
            "sampled": sampled.detach().cpu(),
            "num_sampled": num_sampled.detach().cpu(),
            "positions": pos_cpu,
            "selected_rows": selected,
            "cu_num_logits": cu_num_logits.detach().cpu(),
            "idx_mapping": state_idx.detach().cpu(),
            "idx_mapping_np": np.asarray(idx_mapping_np).copy(),
            "expanded_idx_mapping": expanded_idx_mapping.detach().cpu(),
            "expanded_local_pos": expanded_local_pos.detach().cpu(),
            "temperature": temperature.detach().cpu(),
            "seeds": seeds.detach().cpu(),
            "use_fp64": use_fp64,
            "use_block_verification": use_block_verification,
            "sampling_stages": sampling_stages,
            "sampling_state": sampling_state,
        }
        path = Path(f"{self.output}.rank{rank}.record{self.records_written:04d}.pt")
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        self.records_written += 1

    def should_capture(self, *, rank: int, positions: torch.Tensor) -> bool:
        if rank != 0 or self.records_written >= self.max_records:
            return False
        return bool(
            ((positions >= self.pos_min) & (positions <= self.pos_max)).any().item()
        )
