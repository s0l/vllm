# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import json
import os
from pathlib import Path

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch

_SCHEMA = "ag2-draft-capture-v6"


class Ag2DraftCapture:
    def __init__(
        self,
        output: str,
        max_steps: int,
        topk: int,
        flush_every: int,
        request_prefix: str = "",
    ):
        self.output = output
        self.max_steps = max_steps
        self.topk = topk
        self.flush_every = flush_every
        self.request_prefix = request_prefix
        self.records: list[dict] = []
        self.steps_captured = 0
        self.parts_written = 0
        self.warned_target_only = False
        self.pending_target: dict | None = None
        self.done = False
        manifest = Path(f"{self.output}.manifest.json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema": _SCHEMA,
                    "state": "initialized",
                    "max_steps": max_steps,
                    "topk": topk,
                    "flush_every": flush_every,
                    "request_prefix": request_prefix,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_env(cls) -> Ag2DraftCapture | None:
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
        request_prefix = os.environ.get("AG2_VLLM_DRAFT_CAPTURE_REQUEST_PREFIX", "")
        return cls(output, max_steps, topk, flush_every, request_prefix)

    def stage_target_lm_head_inputs(
        self,
        *,
        rank: int,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        """Stage the exact rows immediately before target ``compute_logits``."""
        if self.done or rank != 0:
            return
        num_reqs = input_batch.num_reqs
        if self.request_prefix and not any(
            str(req_id).startswith(self.request_prefix)
            for req_id in input_batch.req_ids[:num_reqs]
        ):
            self.pending_target = {"skip": True}
            return
        num_logits = int(input_batch.cu_num_logits_np[num_reqs])
        if hidden_states.shape[0] != num_logits:
            raise RuntimeError(
                "AG2 target lm_head rows do not match cu_num_logits: "
                f"{hidden_states.shape[0]} != {num_logits}"
            )
        logits_indices = input_batch.logits_indices[:num_logits].detach().cpu()
        positions = input_batch.positions.detach().cpu()
        input_ids = input_batch.input_ids.detach().cpu()
        if logits_indices.numel() and (
            int(logits_indices.min()) < 0
            or int(logits_indices.max()) >= positions.shape[0]
            or int(logits_indices.max()) >= input_ids.shape[0]
        ):
            raise RuntimeError(
                "AG2 target lm_head logits_indices exceed captured input buffers"
            )
        self.pending_target = {
            "hidden_states": hidden_states.detach().to("cpu", torch.float32),
            "req_ids": list(input_batch.req_ids[:num_reqs]),
            "idx_mapping": input_batch.idx_mapping[:num_reqs].detach().cpu(),
            "expanded_idx_mapping": input_batch.expanded_idx_mapping[:num_logits]
            .detach()
            .cpu(),
            "expanded_local_pos": input_batch.expanded_local_pos[:num_logits]
            .detach()
            .cpu(),
            "cu_num_logits": input_batch.cu_num_logits[: num_reqs + 1].detach().cpu(),
            "positions": positions[logits_indices],
            "input_ids": input_ids[logits_indices],
        }

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
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        step_input_hidden: torch.Tensor,
        step_sample_hidden: torch.Tensor,
        step_input_ids: torch.Tensor,
        step_positions: torch.Tensor,
        step_top_pairs: torch.Tensor,
        step_top_tokens: torch.Tensor,
    ) -> None:
        if self.done or rank != 0 or num_reqs <= 0:
            return
        target = self.pending_target
        self.pending_target = None
        if target is None:
            raise RuntimeError(
                "AG2 draft capture is missing the exact target lm_head input; "
                "stage it immediately before target compute_logits"
            )
        if target.get("skip"):
            return
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
            # Validate request-state provenance on CPU before any diagnostic
            # GPU indexing. The observer must never introduce an asynchronous
            # IndexKernel into the production stream.
            state_idx_cpu = idx_mapping[:num_reqs].detach().cpu().long()
            if bool((state_idx_cpu < 0).any()) or bool(
                (state_idx_cpu >= draft_logits.shape[0]).any()
            ):
                raise RuntimeError("AG2 draft capture request state is out of bounds")
            logits = draft_logits.detach().cpu()[state_idx_cpu]
            k = min(self.topk, logits.shape[-1])
            topk_vals, topk_ids = torch.topk(logits, k, dim=-1)
        state_idx_cpu = idx_mapping[:num_reqs].detach().cpu().long()
        if bool((state_idx_cpu < 0).any()):
            raise RuntimeError("AG2 draft capture saw a negative active state")
        last_sampled_cpu = last_sampled.detach().cpu()
        next_prefill_cpu = next_prefill_tokens.detach().cpu()
        if next_prefill_cpu.ndim != 2 or next_prefill_cpu.shape[0] < 1:
            raise RuntimeError(
                "AG2 draft capture expected next_prefill_tokens with shape "
                "[lookahead, request_state]"
            )
        temperature_cpu = temperature.detach().to("cpu", torch.float32)
        seeds_cpu = seeds.detach().cpu()
        state_capacity = min(
            last_sampled_cpu.shape[0],
            next_prefill_cpu.shape[1],
            temperature_cpu.shape[0],
            seeds_cpu.shape[0],
        )
        if state_idx_cpu.numel() and int(state_idx_cpu.max()) >= state_capacity:
            raise RuntimeError(
                "AG2 draft capture active state exceeds request-state buffers"
            )
        self.records.append(
            {
                "step": self.steps_captured,
                "num_reqs": num_reqs,
                "req_ids": target["req_ids"],
                "topk_vals": (
                    topk_vals.detach().to("cpu", torch.float32)
                    if topk_vals is not None
                    else None
                ),
                "topk_ids": (topk_ids.detach().cpu() if topk_ids is not None else None),
                "draft_tokens": draft_tokens[:num_reqs].detach().cpu(),
                "target_lm_head_hidden_states": target["hidden_states"],
                "target_idx_mapping": target["idx_mapping"],
                "target_expanded_idx_mapping": target["expanded_idx_mapping"],
                "target_expanded_local_pos": target["expanded_local_pos"],
                "target_cu_num_logits": target["cu_num_logits"],
                "target_positions": target["positions"],
                "target_input_ids": target["input_ids"],
                "retained_hidden": hidden_states[:num_reqs]
                .detach()
                .to("cpu", torch.float32),
                "num_sampled": num_sampled[:num_reqs].detach().cpu(),
                "num_rejected": num_rejected[:num_reqs].detach().cpu(),
                # last_sampled / next_prefill_tokens are indexed by request
                # state (idx_mapping), like draft_logits.
                "last_sampled": last_sampled_cpu[state_idx_cpu],
                # The autoregressive prepare-prefill kernel consumes the first
                # lookahead row and indexes request state along dimension 1.
                "next_prefill_tokens": next_prefill_cpu[0, state_idx_cpu],
                "idx_mapping": state_idx_cpu,
                "temperature": temperature_cpu[state_idx_cpu],
                "seeds": seeds_cpu[state_idx_cpu],
                # Proposal-major tensors preserve all K autoregressive steps;
                # active request rows are compact within each step.
                "proposal_input_hidden": step_input_hidden[:, :num_reqs]
                .detach()
                .to("cpu", torch.float32),
                "proposal_sample_hidden": step_sample_hidden[:, :num_reqs]
                .detach()
                .to("cpu", torch.float32),
                "proposal_input_ids": step_input_ids[:, :num_reqs].detach().cpu(),
                "proposal_positions": step_positions[:, :num_reqs].detach().cpu(),
                "proposal_top_pairs": step_top_pairs[:, :num_reqs].detach().cpu(),
                "proposal_top_tokens": step_top_tokens[:, :num_reqs].detach().cpu(),
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
