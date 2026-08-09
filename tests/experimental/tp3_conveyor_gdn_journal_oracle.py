"""Exact GPU oracle for two-bank GDN replay used by the TP3 conveyor POC."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

import vllm.forward_context as forward_context_module
import vllm.v1.worker.ubatching as ubatching_module
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

QUERY_LEN = 4
K_HEADS = 6
V_HEADS = 18
HEAD_DIM = 128
POOL_BLOCKS = 12
MAX_REQS = 8


def _make_layer(device: torch.device) -> QwenGatedDeltaNetAttention:
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer._ag2_mtp_replay_commit = True
    layer._ag2_mtp_journal_banks = 2
    layer._ag2_mtp_journal_num_reqs = [0, 0]
    layer._ag2_mtp_journal_conveyor_step = True
    layer.num_spec = QUERY_LEN - 1
    layer.local_num_k_heads = K_HEADS
    layer.local_num_v_heads = V_HEADS
    layer.head_k_dim = HEAD_DIM
    layer.head_v_dim = HEAD_DIM
    layer.A_log = nn.Parameter(torch.randn(V_HEADS, device=device).mul_(0.1))
    layer.dt_bias = nn.Parameter(torch.randn(V_HEADS, device=device).mul_(0.1))
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
        MAX_REQS + 2, device=device, dtype=torch.int32
    )
    layer._ag2_mtp_journal_state_ids = torch.empty(
        MAX_REQS, device=device, dtype=torch.int32
    )
    layer._ag2_mtp_journal_batch_indices = torch.empty_like(
        layer._ag2_mtp_journal_state_ids
    )
    layer.kv_cache = (
        torch.empty(0, device=device),
        torch.randn(
            POOL_BLOCKS,
            V_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            device=device,
            dtype=torch.float32,
        ).mul_(0.02),
    )
    return layer


def _bank(tensor: torch.Tensor, bank: int) -> torch.Tensor:
    half = tensor.shape[0] // 2
    return tensor[bank * half : (bank + 1) * half]


def _record_wave(
    layer: QwenGatedDeltaNetAttention,
    bank: int,
    request_offset: int,
    state_ids: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    device = state_ids.device
    num_reqs = state_ids.shape[0]
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
    ubatching_module.dbo_current_ubatch_id = lambda: bank
    forward_context_module.get_forward_context = lambda: SimpleNamespace(
        additional_kwargs={"ag2_ubatch_request_offset": request_offset}
    )
    layer._record_mtp_replay_journal(
        key=key, value=value, a=a, b=b, attn_metadata=metadata
    )
    return key, value, a, b, query_start


def _reference_commit(
    layer: QwenGatedDeltaNetAttention,
    pool: torch.Tensor,
    state_ids: torch.Tensor,
    accepted: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
) -> None:
    key, value, a, b, query_start = tensors
    for row, state_id_tensor in enumerate(state_ids):
        take = int(accepted[row])
        start = int(query_start[row])
        state_id = state_id_tensor.view(1, 1).expand(1, take)
        fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log,
            a=a[start : start + take],
            b=b[start : start + take],
            dt_bias=layer.dt_bias,
            q=key[:, start : start + take],
            k=key[:, start : start + take],
            v=value[:, start : start + take],
            initial_state=pool,
            inplace_final_state=True,
            store_output=False,
            cu_seqlens=torch.tensor([0, take], device=pool.device, dtype=torch.int32),
            ssm_state_indices=state_id,
            sequence_lengths=torch.tensor([take], device=pool.device),
            use_qk_l2norm_in_kernel=True,
        )


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(1909)
    device = torch.device("cuda")
    layer = _make_layer(device)
    state_ids = [
        torch.tensor([2, 5, 7], device=device, dtype=torch.int32),
        torch.tensor([3, 8, 10], device=device, dtype=torch.int32),
    ]
    wave_inputs = [
        _record_wave(layer, 0, 0, state_ids[0]),
        _record_wave(layer, 1, 3, state_ids[1]),
    ]
    accepted = torch.tensor([1, 4, 2, 3, 1, 4], device=device, dtype=torch.int32)
    expected = layer.kv_cache[1].clone()
    _reference_commit(layer, expected, state_ids[0], accepted[:3], wave_inputs[0])
    _reference_commit(layer, expected, state_ids[1], accepted[3:], wave_inputs[1])

    idx_mapping = torch.arange(6, device=device, dtype=torch.int32)
    accepted_by_req = accepted.clone()
    replay_accepted = torch.empty(MAX_REQS, device=device, dtype=torch.int32)
    for bank in range(2):
        rows = layer._ag2_mtp_journal_num_reqs[bank]
        gathered = gather_gdn_replay_acceptance(
            _bank(layer._ag2_mtp_journal_batch_indices, bank),
            idx_mapping,
            accepted_by_req,
            _bank(layer._ag2_mtp_journal_query_start, bank),
            replay_accepted,
            rows,
        )
        layer.commit_mtp_replay_journal(gathered, rows, bank=bank)

    torch.cuda.synchronize()
    delta = (layer.kv_cache[1] - expected).abs()
    differing = int(torch.count_nonzero(delta))

    # Monolithic <=M128 fallback still uses the candidate process, but only
    # bank 0 records rows. Empty bank 1 must not launch a zero-row GPU gather.
    layer._ag2_mtp_journal_num_reqs[:] = [0, 0]
    layer._ag2_mtp_journal_conveyor_step = False
    layer.kv_cache[1].copy_(torch.randn_like(layer.kv_cache[1]).mul_(0.02))
    one_bank_inputs = _record_wave(layer, 0, 0, state_ids[0])
    one_bank_expected = layer.kv_cache[1].clone()
    _reference_commit(
        layer,
        one_bank_expected,
        state_ids[0],
        accepted[:3],
        one_bank_inputs,
    )
    # CUDA Graph replay does not rerun the Python recorder, so its scalar may
    # retain a different capture-time row count. Monolithic postprocess must
    # use the current prepare_attn count instead.
    layer._ag2_mtp_journal_num_reqs[0] = 8
    monolithic_runtime_rows = 3
    gathered = gather_gdn_replay_acceptance(
        layer._ag2_mtp_journal_batch_indices,
        idx_mapping,
        accepted_by_req,
        layer._ag2_mtp_journal_query_start,
        replay_accepted,
        monolithic_runtime_rows,
    )
    layer.commit_mtp_replay_journal(gathered, monolithic_runtime_rows, bank=0)
    torch.cuda.synchronize()
    one_bank_delta = (layer.kv_cache[1] - one_bank_expected).abs()
    one_bank_differing = int(torch.count_nonzero(one_bank_delta))
    stale_capture_rows = layer._ag2_mtp_journal_num_reqs[0]
    # Mixed/prefill fallback must retain the original full journal capacity,
    # not inherit one conveyor half (MAX_REQS / 2).
    monolithic_capacity_rows = MAX_REQS - 1
    _record_wave(
        layer,
        0,
        0,
        torch.arange(monolithic_capacity_rows, device=device, dtype=torch.int32),
    )
    print(
        {
            "passed": differing == 0 and one_bank_differing == 0,
            "differing": differing,
            "max_abs": float(delta.max()),
            "one_active_bank_differing": one_bank_differing,
            "stale_capture_rows": stale_capture_rows,
            "monolithic_runtime_rows": monolithic_runtime_rows,
            "monolithic_capacity_rows": monolithic_capacity_rows,
            "global_batch_indices": [
                _bank(layer._ag2_mtp_journal_batch_indices, b)[:3].tolist()
                for b in range(2)
            ],
        }
    )
    if differing or one_bank_differing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
