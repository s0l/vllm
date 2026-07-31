# SPDX-License-Identifier: Apache-2.0
"""Bounded draft-side statistics capture for offline speculation analysis.

Enabled only when ``AG2_VLLM_DRAFT_CAPTURE_OUTPUT`` is set. Requires the
runtime to be configured with ``draft_sample_method="probabilistic"`` so the
persistent ``draft_logits`` buffer is populated; at temperature 0 the gumbel
kernel skips both temperature scaling and noise, so the stored logits are raw
and draft selection remains pure argmax.

Captured per engine step (post-propose, eager, read-only; TP rank 0 saves):

- draft top-K logit values/ids per speculative position (from draft_logits)
- draft_tokens
- exact sampled hidden-state rows consumed by the target lm_head for this step
- draft hidden states buffer (the tensor the speculator retains between
  draft steps; for Qwen3.5 MTP this is the pre-lm_head hidden of the last
  draft step -- the schema records it as ``retained_hidden`` without a
  stronger claim)
- num_sampled / num_rejected (verdicts of the PREVIOUS engine step's drafts)
- last_sampled / next_prefill_tokens (target-chosen tokens, join keys)
- idx_mapping (request-slot join key)

Capture runs are NOT valid performance evidence: collection adds per-step
GPU->CPU synchronization.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_SCHEMA = "ag2-draft-capture-v3"


class Ag2DraftCapture:
    def __init__(self, output: str, max_steps: int, topk: int, flush_every: int):
        self.output = output
        self.max_steps = max_steps
        self.topk = topk
        self.flush_every = flush_every
        self.records: list[dict] = []
        self.steps_captured = 0
        self.parts_written = 0
        self.warned_target_only = False
        self.pending_target_lm_head_hidden_states: torch.Tensor | None = None
        self.done = False

    @classmethod
    def from_env(cls) -> "Ag2DraftCapture | None":
        output = os.environ.get("AG2_VLLM_DRAFT_CAPTURE_OUTPUT")
        if not output:
            return None
        max_steps = int(os.environ.get("AG2_VLLM_DRAFT_CAPTURE_STEPS", "512"))
        topk = int(os.environ.get("AG2_VLLM_DRAFT_CAPTURE_TOPK", "16"))
        flush_every = int(os.environ.get("AG2_VLLM_DRAFT_CAPTURE_FLUSH", "64"))
        if max_steps <= 0 or topk <= 0 or flush_every <= 0:
            raise ValueError(
                "AG2_VLLM_DRAFT_CAPTURE_{STEPS,TOPK,FLUSH} must be positive"
            )
        return cls(output, max_steps, topk, flush_every)

    def stage_target_lm_head_inputs(
        self,
        *,
        rank: int,
        hidden_states: torch.Tensor,
    ) -> None:
        """Stage the exact rows immediately before target ``compute_logits``."""
        if self.done or rank != 0:
            return
        self.pending_target_lm_head_hidden_states = (
            hidden_states.detach().to("cpu", torch.float32)
        )

    def collect(
        self,
        *,
        rank: int,
        num_reqs: int,
        draft_logits: torch.Tensor | None,
        draft_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> None:
        if self.done or rank != 0 or num_reqs <= 0:
            return
        target_lm_head_hidden_states = self.pending_target_lm_head_hidden_states
        self.pending_target_lm_head_hidden_states = None
        if target_lm_head_hidden_states is None:
            raise RuntimeError(
                "AG2 draft capture is missing the exact target lm_head input; "
                "stage it immediately before target compute_logits"
            )
        if draft_logits is None and not self.warned_target_only:
            self.warned_target_only = True
            import logging

            logging.getLogger(__name__).warning(
                "AG2 draft capture has no draft logits; capturing target "
                "hidden states and token/verdict metadata only."
            )
        if draft_logits is None:
            topk_vals = None
            topk_ids = None
        else:
            # draft_logits rows are written by request-state index
            # (idx_mapping), not by batch slot.
            state_idx = idx_mapping[:num_reqs].long().clamp(min=0)
            logits = draft_logits[state_idx]
            k = min(self.topk, logits.shape[-1])
            topk_vals, topk_ids = torch.topk(logits, k, dim=-1)
        state_idx = idx_mapping[:num_reqs].long().clamp(min=0)
        self.records.append(
            {
                "step": self.steps_captured,
                "num_reqs": num_reqs,
                "topk_vals": (
                    topk_vals.detach().to("cpu", torch.float32)
                    if topk_vals is not None
                    else None
                ),
                "topk_ids": (
                    topk_ids.detach().cpu() if topk_ids is not None else None
                ),
                "draft_tokens": draft_tokens[:num_reqs].detach().cpu(),
                "target_lm_head_hidden_states": target_lm_head_hidden_states,
                "retained_hidden": hidden_states[:num_reqs]
                .detach()
                .to("cpu", torch.float32),
                "num_sampled": num_sampled[:num_reqs].detach().cpu(),
                "num_rejected": num_rejected[:num_reqs].detach().cpu(),
                # last_sampled / next_prefill_tokens are indexed by request
                # state (idx_mapping), like draft_logits.
                "last_sampled": last_sampled[state_idx].detach().cpu(),
                "next_prefill_tokens": next_prefill_tokens[state_idx]
                .detach()
                .cpu(),
                "idx_mapping": idx_mapping[:num_reqs].detach().cpu(),
            }
        )
        self.steps_captured += 1
        if len(self.records) >= self.flush_every:
            self._flush(rank)
        if self.steps_captured >= self.max_steps:
            self._flush(rank)
            self.done = True

    def _flush(self, rank: int) -> None:
        if not self.records:
            return
        path = Path(f"{self.output}.rank{rank}.part{self.parts_written:04d}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": _SCHEMA,
                "rank": rank,
                "topk": self.topk,
                "records": self.records,
            },
            path,
        )
        self.parts_written += 1
        self.records = []
