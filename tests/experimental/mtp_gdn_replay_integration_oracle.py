"""Exercise the actual Qwen GDN journal/commit methods without model startup."""

from __future__ import annotations

import json

import torch
from torch import nn

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    gather_gdn_replay_acceptance,
)

MAX_REQS = 8
QUERY_LEN = 4
K_HEADS = 6
V_HEADS = 18
HEAD_DIM = 128
POOL_BLOCKS = 14


def _make_layer(device: torch.device) -> QwenGatedDeltaNetAttention:
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer._ag2_mtp_replay_commit = True
    layer.num_spec = QUERY_LEN - 1
    layer.local_num_k_heads = K_HEADS
    layer.local_num_v_heads = V_HEADS
    layer.head_k_dim = HEAD_DIM
    layer.head_v_dim = HEAD_DIM
    layer.A_log = nn.Parameter(
        torch.randn(V_HEADS, device=device, dtype=torch.float32).mul_(0.1)
    )
    layer.dt_bias = nn.Parameter(
        torch.randn(V_HEADS, device=device, dtype=torch.float32).mul_(0.1)
    )
    max_tokens = MAX_REQS * QUERY_LEN
    layer._ag2_mtp_journal_k = torch.empty(
        max_tokens, K_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    layer._ag2_mtp_journal_v = torch.empty(
        max_tokens, V_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    layer._ag2_mtp_journal_a = torch.empty(
        max_tokens, V_HEADS, device=device, dtype=torch.bfloat16
    )
    layer._ag2_mtp_journal_b = torch.empty_like(layer._ag2_mtp_journal_a)
    layer._ag2_mtp_journal_query_start = torch.empty(
        MAX_REQS + 1, device=device, dtype=torch.int32
    )
    layer._ag2_mtp_journal_state_ids = torch.empty(
        MAX_REQS, device=device, dtype=torch.int32
    )
    layer._ag2_mtp_journal_batch_indices = torch.empty_like(
        layer._ag2_mtp_journal_state_ids
    )
    conv = torch.empty(0, device=device)
    ssm = torch.randn(
        POOL_BLOCKS,
        V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    ).mul_(0.02)
    layer.kv_cache = (conv, ssm)
    return layer


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "differing": int(torch.count_nonzero(delta).item()),
        "max_abs": float(delta.max().item()),
    }


def _reference_commit(
    layer: QwenGatedDeltaNetAttention,
    expected_pool: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state_ids: torch.Tensor,
    query_start: torch.Tensor,
    accepted: torch.Tensor,
) -> None:
    for row in range(state_ids.shape[0]):
        state_id = int(state_ids[row].item())
        take = int(accepted[row].item())
        if state_id == 0 or take == 0:
            continue
        start = int(query_start[row].item())
        end = start + take
        cu = torch.tensor([0, take], device=key.device, dtype=torch.int32)
        indices = state_ids[row].view(1, 1).expand(1, take)
        fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log,
            a=a[start:end],
            b=b[start:end],
            dt_bias=layer.dt_bias,
            q=key[:, start:end],
            k=key[:, start:end],
            v=value[:, start:end],
            initial_state=expected_pool,
            inplace_final_state=True,
            store_output=False,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )


@torch.inference_mode()
def _round(
    layer: QwenGatedDeltaNetAttention,
    *,
    state_ids: torch.Tensor,
    batch_indices: torch.Tensor,
    query_lens: list[int],
    accepted_by_row: torch.Tensor,
    idx_mapping: torch.Tensor,
) -> dict[str, float | int]:
    device = state_ids.device
    query_start = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    total = int(query_start[-1].item())
    key = torch.randn(
        1, total, K_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    value = torch.randn(
        1, total, V_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    a = torch.randn(total, V_HEADS, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    repeated_ids = state_ids[:, None].expand(-1, QUERY_LEN)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=state_ids.shape[0],
        num_spec_decode_tokens=total,
        num_actual_tokens=total,
        spec_query_start_loc=query_start,
        spec_state_indices_tensor=repeated_ids,
        spec_batch_indices=batch_indices,
    )
    layer._record_mtp_replay_journal(
        key=key,
        value=value,
        a=a,
        b=b,
        attn_metadata=metadata,
    )

    accepted_by_req = torch.ones(MAX_REQS, device=device, dtype=torch.int32)
    req_slots = idx_mapping.index_select(0, batch_indices.long())
    valid_reqs = req_slots >= 0
    accepted_by_req.index_copy_(
        0,
        req_slots[valid_reqs].long(),
        accepted_by_row[valid_reqs],
    )
    effective_accepted = accepted_by_row.clone()
    effective_accepted.masked_fill_(~valid_reqs, 0)
    effective_accepted = torch.minimum(
        effective_accepted,
        query_start[1:] - query_start[:-1],
    )
    initial_pool = layer.kv_cache[1].clone()
    expected = initial_pool.clone()
    _reference_commit(
        layer,
        expected,
        key,
        value,
        a,
        b,
        state_ids,
        query_start,
        effective_accepted,
    )
    expected_first = initial_pool.clone()
    _reference_commit(
        layer,
        expected_first,
        key,
        value,
        a,
        b,
        state_ids,
        query_start,
        torch.ones_like(accepted_by_row),
    )
    before_unmapped = layer.kv_cache[1].clone()
    replay_accepted = torch.empty_like(accepted_by_row)
    gather_gdn_replay_acceptance(
        layer._ag2_mtp_journal_batch_indices,
        idx_mapping,
        accepted_by_req,
        layer._ag2_mtp_journal_query_start,
        replay_accepted,
        state_ids.shape[0],
    )
    layer.commit_mtp_replay_journal(replay_accepted, state_ids.shape[0])
    active = torch.unique(state_ids[state_ids > 0]).long()
    matching_lengths: dict[str, list[int]] = {}
    for row, state_id_tensor in enumerate(state_ids):
        state_id = int(state_id_tensor.item())
        if state_id == 0:
            continue
        matches = []
        for candidate in range(QUERY_LEN + 1):
            candidate_pool = initial_pool.clone()
            candidate_accepted = torch.zeros_like(effective_accepted)
            candidate_accepted[row] = candidate
            _reference_commit(
                layer,
                candidate_pool,
                key,
                value,
                a,
                b,
                state_ids,
                query_start,
                candidate_accepted,
            )
            if torch.equal(
                layer.kv_cache[1][state_id],
                candidate_pool[state_id],
            ):
                matches.append(candidate)
        matching_lengths[str(state_id)] = matches
    all_ids = torch.arange(POOL_BLOCKS, device=device)
    inactive_mask = torch.ones(POOL_BLOCKS, device=device, dtype=torch.bool)
    inactive_mask[active] = False
    return {
        **_error(layer.kv_cache[1], expected),
        "accepted_requested": accepted_by_row.tolist(),
        "accepted_effective": effective_accepted.tolist(),
        "per_active_differing": {
            str(int(block_id.item())): _error(
                layer.kv_cache[1][block_id.long()],
                expected[block_id.long()],
            )["differing"]
            for block_id in active
        },
        "vs_first_token_differing": _error(
            layer.kv_cache[1][active], expected_first[active]
        )["differing"],
        "actual_matching_lengths": matching_lengths,
        "unmapped_differing": int(
            torch.count_nonzero(
                layer.kv_cache[1][all_ids[inactive_mask]]
                - before_unmapped[all_ids[inactive_mask]]
            ).item()
        ),
    }


@torch.inference_mode()
def _capture_record_graph(
    layer: QwenGatedDeltaNetAttention,
    num_reqs: int,
    state_ids: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]]:
    device = state_ids.device
    total = num_reqs * QUERY_LEN
    key = torch.randn(
        1, total, K_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    value = torch.randn(
        1, total, V_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    a = torch.randn(total, V_HEADS, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    query_start = torch.arange(
        0, total + 1, QUERY_LEN, device=device, dtype=torch.int32
    )
    batch_indices = torch.arange(num_reqs, device=device, dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_reqs,
        num_spec_decode_tokens=total,
        num_actual_tokens=total,
        spec_query_start_loc=query_start,
        spec_state_indices_tensor=state_ids[:, None].expand(-1, QUERY_LEN),
        spec_batch_indices=batch_indices,
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        layer._record_mtp_replay_journal(
            key=key, value=value, a=a, b=b, attn_metadata=metadata
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer._record_mtp_replay_journal(
            key=key, value=value, a=a, b=b, attn_metadata=metadata
        )
    return graph, (
        key,
        value,
        a,
        b,
        query_start,
        state_ids,
        batch_indices,
    )


@torch.inference_mode()
def _replay_captured_record(
    layer: QwenGatedDeltaNetAttention,
    captured: tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]],
    accepted: torch.Tensor,
) -> dict[str, float | int]:
    graph, tensors = captured
    key, value, a, b, query_start, state_ids, _batch_indices = tensors
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    a.copy_(torch.randn_like(a))
    b.copy_(torch.randn_like(b))
    initial = torch.randn_like(layer.kv_cache[1]).mul_(0.02)
    layer.kv_cache[1].copy_(initial)
    expected = initial.clone()
    _reference_commit(
        layer,
        expected,
        key,
        value,
        a,
        b,
        state_ids,
        query_start,
        accepted,
    )
    graph.replay()
    idx_mapping = torch.arange(MAX_REQS, device=key.device, dtype=torch.int32)
    accepted_by_req = torch.ones(MAX_REQS, device=key.device, dtype=torch.int32)
    accepted_by_req[: accepted.shape[0]].copy_(accepted)
    replay_accepted = torch.empty_like(accepted)
    gather_gdn_replay_acceptance(
        layer._ag2_mtp_journal_batch_indices,
        idx_mapping,
        accepted_by_req,
        layer._ag2_mtp_journal_query_start,
        replay_accepted,
        accepted.shape[0],
    )
    layer.commit_mtp_replay_journal(replay_accepted, accepted.shape[0])
    active = state_ids.long()
    return _error(layer.kv_cache[1][active], expected[active])


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(913)
    device = torch.device("cuda")
    layer = _make_layer(device)
    idx_mapping = torch.tensor(
        [5, 2, 7, 1, 3, 0, 6, -1], device=device, dtype=torch.int32
    )
    first = _round(
        layer,
        state_ids=torch.tensor(
            [7, 2, 11, 4, 0, 13], device=device, dtype=torch.int32
        ),
        batch_indices=torch.tensor(
            [3, 0, 4, 1, 2, 7], device=device, dtype=torch.int32
        ),
        query_lens=[4, 4, 4, 4, 0, 4],
        accepted_by_row=torch.tensor(
            [1, 2, 3, 4, 1, 4], device=device, dtype=torch.int32
        ),
        idx_mapping=idx_mapping,
    )
    # Reuse physical slots with a different batch permutation and fresh inputs.
    second = _round(
        layer,
        state_ids=torch.tensor([4, 7, 2, 11], device=device, dtype=torch.int32),
        batch_indices=torch.tensor([5, 2, 0, 6], device=device, dtype=torch.int32),
        query_lens=[4, 4, 4, 4],
        accepted_by_row=torch.tensor(
            [4, 1, 3, 2], device=device, dtype=torch.int32
        ),
        idx_mapping=idx_mapping,
    )
    # Capture different graph shapes in one layer, then replay the first after
    # the second. Python capture-time scalars must not leak across graph keys.
    graph_x1 = _capture_record_graph(
        layer,
        1,
        torch.tensor([13], device=device, dtype=torch.int32),
    )
    graph_x4 = _capture_record_graph(
        layer,
        4,
        torch.tensor([7, 2, 11, 4], device=device, dtype=torch.int32),
    )
    graph_x1_result = _replay_captured_record(
        layer,
        graph_x1,
        torch.tensor([4], device=device, dtype=torch.int32),
    )
    graph_x4_result = _replay_captured_record(
        layer,
        graph_x4,
        torch.tensor([1, 2, 3, 4], device=device, dtype=torch.int32),
    )
    passed = (
        first["differing"] == 0
        and first["unmapped_differing"] == 0
        and second["differing"] == 0
        and second["unmapped_differing"] == 0
        and graph_x1_result["differing"] == 0
        and graph_x4_result["differing"] == 0
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "first": first,
                "slot_reuse": second,
                "graph_x1_after_x4_capture": graph_x1_result,
                "graph_x4": graph_x4_result,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
