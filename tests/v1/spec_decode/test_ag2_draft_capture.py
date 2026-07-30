# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU controls for Ag2DraftCapture (no vLLM imports needed)."""

import os
import sys
import tempfile
from pathlib import Path

import torch

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from vllm.v1.worker.gpu.spec_decode.autoregressive.ag2_draft_capture import (
        Ag2DraftCapture,
    )
except Exception:  # standalone run outside the vllm tree
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "ag2_draft_capture", os.environ["AG2_CAPTURE_MODULE"]
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    Ag2DraftCapture = _mod.Ag2DraftCapture


def make_inputs(num_reqs=3, k=2, vocab=100, hidden=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "draft_logits": torch.randn(num_reqs + 3, k, vocab, generator=g),
        "draft_tokens": torch.randint(0, vocab, (num_reqs, k), generator=g),
        "target_lm_head_hidden_states": torch.randn(
            num_reqs * (k + 1), hidden, generator=g
        ),
        "hidden_states": torch.randn(num_reqs, hidden, generator=g),
        "num_sampled": torch.randint(1, k + 2, (num_reqs,), generator=g),
        "num_rejected": torch.randint(0, k + 1, (num_reqs,), generator=g),
        "last_sampled": torch.randint(0, vocab, (num_reqs + 3,), generator=g),
        "next_prefill_tokens": torch.randint(
            0, vocab, (num_reqs + 3,), generator=g
        ),
        "idx_mapping": torch.arange(num_reqs, dtype=torch.int32) + 2,
    }


def test_disabled_env():
    os.environ.pop("AG2_VLLM_DRAFT_CAPTURE_OUTPUT", None)
    assert Ag2DraftCapture.from_env() is None
    print("PASS disabled_env")


def test_missing_draft_logits_captures_target_only(tmp):
    cap = Ag2DraftCapture(f"{tmp}/x", max_steps=4, topk=4, flush_every=2)
    inp = make_inputs()
    inp["draft_logits"] = None
    target_hidden = inp.pop("target_lm_head_hidden_states")
    cap.stage_target_lm_head_inputs(rank=0, hidden_states=target_hidden)
    cap.collect(rank=0, num_reqs=3, **inp)
    cap._flush(rank=0)
    parts = list(Path(tmp).glob("*.pt"))
    assert cap.steps_captured == 1 and len(parts) == 1
    payload = torch.load(parts[0], weights_only=True)
    record = payload["records"][0]
    assert record["topk_vals"] is None
    assert record["topk_ids"] is None
    assert torch.equal(
        record["target_lm_head_hidden_states"], target_hidden.float()
    )
    print("PASS missing_draft_logits_captures_target_only")


def test_nonzero_rank_captures_nothing(tmp):
    cap = Ag2DraftCapture(f"{tmp}/r", max_steps=4, topk=4, flush_every=2)
    for rank in (1, 2):
        inp = make_inputs()
        target_hidden = inp.pop("target_lm_head_hidden_states")
        cap.stage_target_lm_head_inputs(rank=rank, hidden_states=target_hidden)
        cap.collect(rank=rank, num_reqs=3, **inp)
    assert cap.steps_captured == 0 and not list(Path(tmp).glob("r.*.pt"))
    print("PASS nonzero_rank_captures_nothing")


def test_positive_capture_and_budget(tmp):
    cap = Ag2DraftCapture(f"{tmp}/c", max_steps=3, flush_every=2, topk=4)
    inputs = [make_inputs(seed=s) for s in range(5)]
    for inp in inputs:
        target_hidden = inp.pop("target_lm_head_hidden_states")
        inp["expected_target_lm_head_hidden_states"] = target_hidden
        cap.stage_target_lm_head_inputs(rank=0, hidden_states=target_hidden)
        expected = inp.pop("expected_target_lm_head_hidden_states")
        cap.collect(rank=0, num_reqs=3, **inp)
        inp["expected_target_lm_head_hidden_states"] = expected
    assert cap.done and cap.steps_captured == 3, "budget must stop capture"

    parts = sorted(Path(tmp).glob("c.rank0.part*.pt"))
    assert len(parts) == 2, f"expected 2 parts, got {parts}"
    records = []
    for p in parts:
        payload = torch.load(p, weights_only=True)
        assert payload["schema"] == "ag2-draft-capture-v3"
        records.extend(payload["records"])
    assert len(records) == 3

    for step, rec in enumerate(records):
        state = inputs[step]["idx_mapping"].long()
        ref_vals, ref_ids = torch.topk(inputs[step]["draft_logits"][state], 4, dim=-1)
        assert torch.equal(rec["topk_ids"], ref_ids), "topk ids mismatch"
        assert torch.allclose(rec["topk_vals"], ref_vals), "topk vals mismatch"
        assert torch.equal(rec["num_rejected"], inputs[step]["num_rejected"])
        assert torch.equal(
            rec["next_prefill_tokens"],
            inputs[step]["next_prefill_tokens"][state],
        )
        assert torch.equal(
            rec["target_lm_head_hidden_states"],
            inputs[step]["expected_target_lm_head_hidden_states"].float(),
        )
        assert rec["target_lm_head_hidden_states"].shape == (9, 8)
        assert rec["retained_hidden"].shape == (3, 8)
        assert rec["step"] == step
    print("PASS positive_capture_and_budget")


def test_validator_detects_corruption(tmp):
    """Deliberately corrupted record must fail the topk comparison."""
    cap = Ag2DraftCapture(f"{tmp}/n", max_steps=1, flush_every=1, topk=4)
    inp = make_inputs(seed=42)
    target_hidden = inp.pop("target_lm_head_hidden_states")
    cap.stage_target_lm_head_inputs(rank=0, hidden_states=target_hidden)
    cap.collect(rank=0, num_reqs=3, **inp)
    part = next(Path(tmp).glob("n.rank0.part*.pt"))
    payload = torch.load(part, weights_only=True)
    rec = payload["records"][0]
    tampered = rec["topk_ids"].clone()
    tampered[0, 0, 0] += 1
    ref_ids = torch.topk(inp["draft_logits"][inp["idx_mapping"].long()], 4, dim=-1).indices
    assert not torch.equal(tampered, ref_ids), (
        "validator failed to detect deliberate corruption"
    )
    print("PASS validator_detects_corruption")


def test_target_hidden_validator_detects_corruption(tmp):
    cap = Ag2DraftCapture(f"{tmp}/h", max_steps=1, flush_every=1, topk=4)
    inp = make_inputs(seed=17)
    inp["draft_logits"] = None
    target_hidden = inp.pop("target_lm_head_hidden_states")
    cap.stage_target_lm_head_inputs(rank=0, hidden_states=target_hidden)
    cap.collect(rank=0, num_reqs=3, **inp)
    part = next(Path(tmp).glob("h.rank0.part*.pt"))
    payload = torch.load(part, weights_only=True)
    captured = payload["records"][0]["target_lm_head_hidden_states"]
    tampered = captured.clone()
    tampered[0, 0] += 1
    assert torch.equal(captured, target_hidden.float())
    assert not torch.equal(tampered, target_hidden.float())
    print("PASS target_hidden_validator_detects_corruption")


def test_missing_staged_target_hidden_fails_closed(tmp):
    cap = Ag2DraftCapture(f"{tmp}/m", max_steps=1, flush_every=1, topk=4)
    inp = make_inputs(seed=19)
    inp.pop("target_lm_head_hidden_states")
    try:
        cap.collect(rank=0, num_reqs=3, **inp)
    except RuntimeError as exc:
        assert "missing the exact target lm_head input" in str(exc)
    else:
        raise AssertionError("capture accepted an unattributed hidden-state tensor")
    print("PASS missing_staged_target_hidden_fails_closed")


if __name__ == "__main__":
    test_disabled_env()
    with tempfile.TemporaryDirectory() as tmp:
        test_missing_draft_logits_captures_target_only(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_nonzero_rank_captures_nothing(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_positive_capture_and_budget(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_validator_detects_corruption(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_target_hidden_validator_detects_corruption(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_missing_staged_target_hidden_fails_closed(tmp)
    print("ALL CONTROLS PASS")
