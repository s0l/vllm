# SPDX-License-Identifier: Apache-2.0
"""Always-on crash-proof flight recorder for decode post-mortems.

Motivation: the stochastic non-terminating decode (1 request per ~8-16
reaching the 840 s timeout) cannot be classified after a restart. This ring
records the last N engine steps (token ids + accept verdicts per slot) into
an mmap-backed file that survives crashes and restarts.

Design constraints: zero hot-path synchronization. Per step the GPU tensors
are copied non_blocking into a pinned staging pair; the PREVIOUS step's
staged values (guaranteed complete: one step >> copy time) are written into
the numpy memmap ring. Post-mortem lag of one step is irrelevant.

Env: AG2_VLLM_FLIGHT_RECORDER=<path prefix> enables (rank 0 records),
AG2_VLLM_FLIGHT_RECORDER_STEPS=<N> ring size (default 4096).
Reader: evals/flight_recorder_dump.py.
"""

from __future__ import annotations

import hashlib
import os
import string
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

MAGIC = 0xA62F11A9
MAX_SLOTS = 64
HEADER = 5  # int64s: magic, ring_size, write_seq, max_slots, num_drafts


def external_request_id(req_id: str) -> str:
    """Undo vLLM's documented ``external_id-<8 random hex>`` suffix."""
    prefix, separator, suffix = req_id.rpartition("-")
    if (
        separator
        and len(suffix) == 8
        and all(char in string.hexdigits for char in suffix)
    ):
        return prefix
    return req_id


def stable_request_hash(req_id: str) -> int:
    """Return a reproducible signed int64 join key for an external request."""
    external_id = external_request_id(req_id)
    digest = hashlib.sha256(external_id.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, byteorder="little", signed=True)


class Ag2FlightRecorder:
    def __init__(self, output: str, steps: int, num_drafts: int):
        if num_drafts <= 0:
            raise ValueError("flight recorder requires at least one draft token")
        self.ring_size = steps
        self.num_drafts = num_drafts
        path = Path(f"{output}.rank0.fr.npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        # record: step, t_ns, num_reqs, draft_width, then per-slot
        # [state, external_request_hash, sampled, rejected, last_sampled,
        # drafts...]
        self.slot_len = 5 + num_drafts
        self.rec_len = 4 + MAX_SLOTS * self.slot_len
        self.mm = np.memmap(
            path,
            dtype=np.int64,
            mode="w+",
            shape=(HEADER + steps * self.rec_len,),
        )
        self.mm[0] = MAGIC
        self.mm[1] = steps
        self.mm[2] = 0
        self.mm[3] = MAX_SLOTS
        self.mm[4] = num_drafts
        self.staged: list[dict | None] = [None, None]
        self.seq = 0
        self._request_hashes: dict[str, int] = {}

    @classmethod
    def from_env(cls, *, num_drafts: int) -> Ag2FlightRecorder | None:
        out = os.environ.get("AG2_VLLM_FLIGHT_RECORDER")
        if not out:
            return None
        steps = int(os.environ.get("AG2_VLLM_FLIGHT_RECORDER_STEPS", "4096"))
        return cls(out, steps, num_drafts)

    def _hash_request_ids(self, req_ids: Sequence[str], n: int) -> np.ndarray:
        hashes = np.empty(n, dtype=np.int64)
        for i, req_id in enumerate(req_ids[:n]):
            value = self._request_hashes.get(req_id)
            if value is None:
                value = stable_request_hash(req_id)
                self._request_hashes[req_id] = value
            hashes[i] = value
        return hashes

    def record(
        self,
        *,
        num_reqs: int,
        req_ids: Sequence[str],
        draft_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> None:
        n = min(num_reqs, MAX_SLOTS)
        if len(req_ids) < n:
            raise ValueError(f"only {len(req_ids)} request ids for {n} slots")
        if draft_tokens.ndim != 2 or draft_tokens.shape[1] != self.num_drafts:
            raise ValueError(
                "flight recorder draft width changed: "
                f"expected {self.num_drafts}, got {tuple(draft_tokens.shape)}"
            )
        state_idx = idx_mapping[:n].long().clamp(min=0)
        cur = self.seq % 2
        stage = self.staged[cur]
        need = {
            "state": state_idx,
            "dt": draft_tokens[:n, : self.num_drafts],
            "ns": num_sampled[:n],
            "nr": num_rejected[:n],
            "ls": last_sampled[state_idx].reshape(n, -1)[:, 0],
        }
        if stage is None or stage["n_cap"] < n:
            stage = {
                "n_cap": max(n, 8),
                **{
                    k: torch.empty(
                        (max(n, 8),) + tuple(v.shape[1:]),
                        dtype=torch.int64,
                        pin_memory=True,
                    )
                    for k, v in need.items()
                },
            }
            self.staged[cur] = stage
        stage["req_hash"] = self._hash_request_ids(req_ids, n)
        for k, v in need.items():
            stage[k][:n].copy_(v, non_blocking=True)
        stage["n"] = n
        stage["t_ns"] = time.time_ns()
        stage["seq"] = self.seq

        # flush the PREVIOUS step's staged copy (complete by now)
        prev = self.staged[1 - cur]
        if prev is not None and "n" in prev:
            self._flush(prev)
        self.seq += 1

    def _flush(self, stage: dict) -> None:
        slot = stage["seq"] % self.ring_size
        base = HEADER + slot * self.rec_len
        n = stage["n"]
        rec = np.zeros(self.rec_len, dtype=np.int64)
        rec[0] = stage["seq"]
        rec[1] = stage["t_ns"]
        rec[2] = n
        rec[3] = self.num_drafts
        body = rec[4:].reshape(MAX_SLOTS, self.slot_len)
        body[:n, 0] = stage["state"][:n].numpy()
        body[:n, 1] = stage["req_hash"][:n]
        body[:n, 2] = stage["ns"][:n].numpy()
        body[:n, 3] = stage["nr"][:n].numpy()
        body[:n, 4] = stage["ls"][:n].numpy()
        body[:n, 5:] = stage["dt"][:n].numpy().reshape(n, -1)
        self.mm[base : base + self.rec_len] = rec
        self.mm[2] = stage["seq"] + 1
