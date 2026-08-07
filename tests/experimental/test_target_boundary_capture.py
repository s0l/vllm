from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu import target_boundary_capture as module


def _batch(req_ids: list[str], counts: list[int]) -> SimpleNamespace:
    total = sum(counts)
    cu = [0]
    for count in counts:
        cu.append(cu[-1] + count)
    return SimpleNamespace(
        num_reqs=len(req_ids),
        req_ids=req_ids,
        cu_num_logits_np=cu,
        logits_indices=torch.arange(total),
        positions=torch.arange(100, 100 + total),
        input_ids=torch.arange(200, 200 + total),
        expanded_idx_mapping=torch.arange(total),
        expanded_local_pos=torch.zeros(total, dtype=torch.int32),
        idx_mapping=torch.arange(len(req_ids)),
        prefill_len_np=[101, 101, 102],
        query_start_loc=torch.tensor(cu),
        num_scheduled_tokens=torch.tensor(counts).numpy(),
        num_computed_tokens_np=torch.zeros(len(req_ids), dtype=torch.int32).numpy(),
        seq_lens=torch.tensor([100 + count for count in counts]),
        prompt_lens=None,
        num_tokens=total,
        num_tokens_after_padding=total,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        is_prefilling_np=torch.ones(len(req_ids), dtype=torch.bool).numpy(),
    )


def test_capture_is_filtered_first_output_only_and_mutation_sensitive(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    capture = module.TargetBoundaryCapture(
        str(tmp_path / "target"), "wanted-", 2
    )
    batch = _batch(["dummy", "wanted-a", "wanted-b"], [0, 1, 2])
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    logits = torch.arange(21, dtype=torch.bfloat16).reshape(3, 7)
    capture.capture(hidden, logits, batch)
    capture.capture(hidden + 1000, logits + 1000, batch)

    payload = torch.load(
        tmp_path / "target.part0000.pt", map_location="cpu", weights_only=False
    )
    assert [item["req_id"] for item in payload["requests"]] == [
        "wanted-a",
        "wanted-b",
    ]
    assert payload["source_rows"].tolist() == [0, 1, 2]
    assert torch.equal(payload["hidden_states"], hidden)
    assert torch.equal(payload["raw_logits"], logits)
    assert not (tmp_path / "target.part0001.pt").exists()

    mutated = logits.clone()
    mutated[2, 3] += 1
    assert not torch.equal(mutated, payload["raw_logits"])


def test_non_owner_rank_does_not_write_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    capture = module.TargetBoundaryCapture(
        str(tmp_path / "target"), "wanted-", 1
    )
    batch = _batch(["wanted-a"], [1])
    capture.capture(torch.ones(1, 4), torch.ones(1, 7), batch)
    assert not (tmp_path / "target.part0000.pt").exists()


def test_partial_prefill_is_not_mislabeled_as_first_emission(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    capture = module.TargetBoundaryCapture(str(tmp_path / "target"), "wanted-", 1)
    partial = _batch(["wanted-a"], [1])
    partial.prefill_len_np = [1513]
    partial.positions = torch.tensor([1343])
    capture.capture(torch.ones(1, 4), torch.ones(1, 7), partial)
    assert capture.seen == set()
    assert not (tmp_path / "target.part0000.pt").exists()

    final = _batch(["wanted-a"], [1])
    final.prefill_len_np = [1513]
    final.positions = torch.tensor([1512])
    capture.capture(torch.ones(1, 4), torch.ones(1, 7), final)
    assert capture.seen == {"wanted-a"}
    assert (tmp_path / "target.part0000.pt").exists()
