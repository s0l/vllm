"""Pure policy helpers for the experimental phase-switched tensor layout."""

from __future__ import annotations

import os
from dataclasses import dataclass


_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DualViewGdnMove:
    source_rank: int
    target_rank: int
    key_head_start: int
    key_head_count: int


def qwopus_tp2_to_tp3_gdn_moves() -> tuple[DualViewGdnMove, ...]:
    """Minimal state movement for physical TP3 order GPU0,GPU2,GPU1.

    TP2 owns key-head ranges [0,8) and [8,16) on GPU0/GPU1.  Decode owns
    [0,6), [6,11), [11,16) on GPU0/GPU2/GPU1, respectively.  Therefore both
    fast-pair ranks retain their outer ranges and only GPU2 receives data.
    """
    return (
        DualViewGdnMove(0, 2, 6, 2),
        DualViewGdnMove(1, 2, 8, 3),
    )


def dual_view_enabled() -> bool:
    return os.getenv(
        "VLLM_EXPERIMENTAL_DUAL_VIEW_PHASE_SEPARATION", "0"
    ).lower() in _TRUE


def choose_dual_view_prefill_phase(
    *,
    current_step: int,
    cadence: int,
    has_running_prefill: bool,
    has_running_decode: bool,
    has_waiting_work: bool,
) -> bool | None:
    """Choose one phase for a forward, never a mixture.

    ``True`` is prefill, ``False`` is decode, and ``None`` means no work. When
    both phases are pending, prefill receives one step per ``cadence`` steps.
    """
    if cadence < 1:
        raise ValueError("dual-view prefill cadence must be positive")
    has_prefill = has_running_prefill or has_waiting_work
    if has_prefill and has_running_decode:
        return current_step % cadence == 0
    if has_prefill:
        return True
    if has_running_decode:
        return False
    return None
