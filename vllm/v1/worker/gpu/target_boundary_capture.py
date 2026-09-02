# SPDX-License-Identifier: Apache-2.0
"""Bounded first-output capture around the target LM head.

This diagnostic is deliberately request-filtered and disabled by default.  It
records the exact hidden row entering ``compute_logits`` and the corresponding
raw full-vocabulary output for the first sampling invocation of each matching
request.  The capture sits outside model CUDA graphs and is not performance
evidence: device-to-host copies synchronize the production stream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

_SCHEMA = "ag2-target-boundary-capture-v1"


class TargetBoundaryCapture:
    def __init__(self, output: str, request_prefix: str, max_requests: int):
        self.output = Path(output)
        self.request_prefix = request_prefix
        self.max_requests = max_requests
        self.seen: set[str] = set()
        self.invocation = 0
        self.parts_written = 0
        self._write_manifest("initialized")

    @classmethod
    def from_env(cls) -> TargetBoundaryCapture | None:
        output = os.environ.get("AG2_VLLM_TARGET_BOUNDARY_CAPTURE_OUTPUT", "")
        if not output:
            return None
        # Every TP worker constructs its model runner independently.  Resolve
        # ownership before creating any shared artifact so non-owner ranks do
        # not race rank 0 during initialization.
        if get_tensor_model_parallel_rank() != 0:
            return None
        request_prefix = os.environ.get(
            "AG2_VLLM_TARGET_BOUNDARY_CAPTURE_REQUEST_PREFIX", ""
        )
        if not request_prefix:
            raise ValueError(
                "AG2_VLLM_TARGET_BOUNDARY_CAPTURE_REQUEST_PREFIX is required"
            )
        max_requests = int(
            os.environ.get("AG2_VLLM_TARGET_BOUNDARY_CAPTURE_REQUESTS", "64")
        )
        if max_requests <= 0:
            raise ValueError(
                "AG2_VLLM_TARGET_BOUNDARY_CAPTURE_REQUESTS must be positive"
            )
        return cls(output, request_prefix, max_requests)

    def _write_manifest(self, state: str) -> None:
        manifest = Path(f"{self.output}.manifest.json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(manifest.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": _SCHEMA,
                    "state": state,
                    "request_prefix": self.request_prefix,
                    "max_requests": self.max_requests,
                    "seen_requests": len(self.seen),
                    "parts_written": self.parts_written,
                    "invocations": self.invocation,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest)

    def capture(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        input_batch: Any,
        slot_mappings_by_layer: dict[str, torch.Tensor] | None = None,
    ) -> None:
        invocation = self.invocation
        self.invocation += 1
        if get_tensor_model_parallel_rank() != 0:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("target boundary capture cannot run inside a CUDA graph")
        if len(self.seen) >= self.max_requests:
            self._write_manifest("complete")
            return

        num_reqs = int(input_batch.num_reqs)
        req_ids = [str(value) for value in input_batch.req_ids[:num_reqs]]
        cu_num_logits = [
            int(value) for value in input_batch.cu_num_logits_np[: num_reqs + 1]
        ]
        if (
            cu_num_logits[-1] != hidden_states.shape[0]
            or hidden_states.shape[0] != logits.shape[0]
        ):
            raise RuntimeError(
                "target boundary rows disagree: "
                f"cu={cu_num_logits[-1]} hidden={hidden_states.shape[0]} "
                f"logits={logits.shape[0]}"
            )

        logits_indices = input_batch.logits_indices[: cu_num_logits[-1]]
        positions = input_batch.positions[logits_indices]
        input_ids = input_batch.input_ids[logits_indices]
        prefill_lens = [int(value) for value in input_batch.prefill_len_np[:num_reqs]]

        selected_requests: list[dict[str, Any]] = []
        selected_rows: list[int] = []
        for req_idx, req_id in enumerate(req_ids):
            start, end = cu_num_logits[req_idx : req_idx + 2]
            reached_emission_boundary = end > start and (
                prefill_lens[req_idx] <= 0
                or int(positions[end - 1]) >= prefill_lens[req_idx] - 1
            )
            if (
                not req_id.startswith(self.request_prefix)
                or req_id in self.seen
                or not reached_emission_boundary
                or len(self.seen) >= self.max_requests
            ):
                continue
            rows = list(range(start, end))
            selected_requests.append(
                {
                    "req_id": req_id,
                    "req_index": req_idx,
                    "row_start": len(selected_rows),
                    "row_end": len(selected_rows) + len(rows),
                    "source_row_start": start,
                    "source_row_end": end,
                    "prefill_len": prefill_lens[req_idx],
                }
            )
            selected_rows.extend(rows)
            self.seen.add(req_id)

        if not selected_rows:
            self._write_manifest(
                "complete"
                if len(self.seen) >= self.max_requests
                else ("active" if self.seen else "initialized")
            )
            return

        row_index = torch.tensor(selected_rows, dtype=torch.long, device=logits.device)
        payload = {
            "schema": _SCHEMA,
            "invocation": invocation,
            "num_reqs": num_reqs,
            "num_logits": cu_num_logits[-1],
            "requests": selected_requests,
            "source_rows": torch.tensor(selected_rows, dtype=torch.int64),
            "hidden_states": hidden_states.index_select(0, row_index)
            .detach()
            .to("cpu", torch.float32),
            "raw_logits": logits.index_select(0, row_index).detach().to("cpu"),
            "expanded_idx_mapping": input_batch.expanded_idx_mapping[
                : cu_num_logits[-1]
            ]
            .index_select(0, row_index)
            .detach()
            .cpu(),
            "expanded_local_pos": input_batch.expanded_local_pos[: cu_num_logits[-1]]
            .index_select(0, row_index)
            .detach()
            .cpu(),
            "positions": positions.index_select(0, row_index).detach().cpu(),
            "input_ids": input_ids.index_select(0, row_index).detach().cpu(),
            "idx_mapping": input_batch.idx_mapping[:num_reqs].detach().cpu(),
            "cu_num_logits": torch.tensor(cu_num_logits, dtype=torch.int64),
            "req_ids": req_ids,
            "query_start_loc": input_batch.query_start_loc[: num_reqs + 1]
            .detach()
            .cpu(),
            "num_scheduled_tokens": torch.from_numpy(
                input_batch.num_scheduled_tokens[:num_reqs].copy()
            ),
            "num_computed_tokens": torch.from_numpy(
                input_batch.num_computed_tokens_np[:num_reqs].copy()
            ),
            "seq_lens": input_batch.seq_lens[:num_reqs].detach().cpu(),
            "prompt_lens": (
                input_batch.prompt_lens[:num_reqs].detach().cpu()
                if input_batch.prompt_lens is not None
                else None
            ),
            "prefill_lens": torch.tensor(prefill_lens, dtype=torch.int64),
            "num_tokens": int(input_batch.num_tokens),
            "num_tokens_after_padding": int(input_batch.num_tokens_after_padding),
            "num_draft_tokens": int(input_batch.num_draft_tokens),
            "num_draft_tokens_per_req": (
                torch.from_numpy(input_batch.num_draft_tokens_per_req[:num_reqs].copy())
                if input_batch.num_draft_tokens_per_req is not None
                else None
            ),
            "is_prefilling": torch.from_numpy(
                input_batch.is_prefilling_np[:num_reqs].copy()
            ),
            "logits_indices": logits_indices.detach().cpu(),
            "slot_mappings": {
                layer_name: mapping[: input_batch.num_tokens_after_padding]
                .index_select(0, logits_indices)
                .index_select(0, row_index)
                .detach()
                .cpu()
                for layer_name, mapping in (slot_mappings_by_layer or {}).items()
            },
        }
        path = Path(f"{self.output}.part{self.parts_written:04d}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{path}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self.parts_written += 1
        self._write_manifest(
            "complete" if len(self.seen) >= self.max_requests else "active"
        )
