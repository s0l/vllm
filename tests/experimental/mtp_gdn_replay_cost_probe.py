"""Measure the exact existing-kernel upper bound for journal commit."""

from __future__ import annotations

import json
import statistics

import torch

from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    gather_gdn_replay_acceptance,
)

QUERY_LEN = 4
NUM_K_HEADS = 6
NUM_V_HEADS = 18
HEAD_DIM = 128
MODEL_GDN_LAYERS = 48
WARMUPS = 20
REPEATS = 100


@torch.inference_mode()
def measure(batch: int, device: torch.device) -> dict[str, float | int]:
    dtype = torch.bfloat16
    total_tokens = batch * QUERY_LEN
    k = torch.randn(
        1,
        total_tokens,
        NUM_K_HEADS,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        1,
        total_tokens,
        NUM_V_HEADS,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(total_tokens, NUM_V_HEADS, device=device, dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(NUM_V_HEADS, device=device, dtype=torch.float32).mul_(0.1)
    dt_bias = torch.randn_like(a_log).mul_(0.1)
    replay_states = torch.randn(
        batch + 1,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    ).mul_(0.02)
    live_indices = (
        torch.arange(1, batch + 1, device=device, dtype=torch.int32)
        .view(batch, 1)
        .expand(-1, QUERY_LEN)
    )
    state_indices = live_indices.contiguous()
    query_start = torch.arange(
        0,
        total_tokens + 1,
        QUERY_LEN,
        device=device,
        dtype=torch.int32,
    )
    sequence_lengths = torch.full((batch,), QUERY_LEN, device=device, dtype=torch.int32)

    def replay() -> None:
        fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=k,
            k=k,
            v=v,
            initial_state=replay_states,
            inplace_final_state=True,
            store_output=False,
            cu_seqlens=query_start,
            ssm_state_indices=state_indices,
            sequence_lengths=sequence_lengths,
            use_qk_l2norm_in_kernel=True,
        )

    snapshot_states = torch.randn(
        1 + batch * QUERY_LEN,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    ).mul_(0.02)
    snapshot_indices = torch.arange(
        1,
        1 + batch * QUERY_LEN,
        device=device,
        dtype=torch.int32,
    ).view(batch, QUERY_LEN)
    for row in range(batch):
        snapshot_states[snapshot_indices[row, 0]].copy_(replay_states[row + 1])

    def verify_store() -> None:
        fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=k,
            k=k,
            v=v,
            initial_state=snapshot_states,
            inplace_final_state=True,
            cu_seqlens=query_start,
            ssm_state_indices=snapshot_indices,
            use_qk_l2norm_in_kernel=True,
        )

    def verify_no_store() -> None:
        fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=k,
            k=k,
            v=v,
            initial_state=replay_states,
            inplace_final_state=True,
            store_final_state=False,
            cu_seqlens=query_start,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    batch_indices = torch.arange(batch, device=device, dtype=torch.int32)
    idx_mapping = torch.arange(batch - 1, -1, -1, device=device, dtype=torch.int32)
    accepted_by_slot = torch.full((batch,), QUERY_LEN, device=device, dtype=torch.int32)
    gathered_accepted = torch.empty_like(accepted_by_slot)

    def gather_acceptance() -> None:
        gather_gdn_replay_acceptance(
            batch_indices,
            idx_mapping,
            accepted_by_slot,
            query_start,
            gathered_accepted,
            batch,
        )

    def timed(fn) -> tuple[float, float]:
        for _ in range(WARMUPS):
            fn()
        torch.cuda.synchronize(device)
        samples_ms = []
        for _ in range(REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))
        samples_ms.sort()
        return (
            statistics.median(samples_ms),
            samples_ms[int(0.95 * (len(samples_ms) - 1))],
        )

    median, p95 = timed(replay)
    verify_store_median, verify_store_p95 = timed(verify_store)
    verify_no_store_median, verify_no_store_p95 = timed(verify_no_store)
    gather_median, gather_p95 = timed(gather_acceptance)
    return {
        "batch": batch,
        "one_layer_median_ms": median,
        "one_layer_p95_ms": p95,
        "serial_48_layer_median_ms": median * MODEL_GDN_LAYERS,
        "serial_48_layer_p95_ms": p95 * MODEL_GDN_LAYERS,
        "verify_store_48_layer_median_ms": verify_store_median * MODEL_GDN_LAYERS,
        "verify_no_store_48_layer_median_ms": (
            verify_no_store_median * MODEL_GDN_LAYERS
        ),
        "net_replay_path_48_layer_median_ms": (
            (verify_no_store_median + median) * MODEL_GDN_LAYERS
        ),
        "net_vs_snapshot_48_layer_median_ms": (
            (verify_no_store_median + median - verify_store_median) * MODEL_GDN_LAYERS
        ),
        "verify_store_48_layer_p95_ms": verify_store_p95 * MODEL_GDN_LAYERS,
        "verify_no_store_48_layer_p95_ms": verify_no_store_p95 * MODEL_GDN_LAYERS,
        "one_batch_acceptance_gather_median_ms": gather_median,
        "one_batch_acceptance_gather_p95_ms": gather_p95,
        "state_bytes": replay_states.numel() * replay_states.element_size(),
        "snapshot_state_bytes": (
            snapshot_states.numel() * snapshot_states.element_size()
        ),
        "journal_input_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in (k, v, a, b)
        ),
    }


def main() -> None:
    torch.manual_seed(212)
    device = torch.device("cuda")
    rows = [measure(batch, device) for batch in (1, 22)]
    print(
        json.dumps(
            {
                "shape": {
                    "query_len": QUERY_LEN,
                    "accepted": QUERY_LEN,
                    "k_heads": NUM_K_HEADS,
                    "v_heads": NUM_V_HEADS,
                    "head_dim": HEAD_DIM,
                    "gdn_layers": MODEL_GDN_LAYERS,
                    "warmups": WARMUPS,
                    "repeats": REPEATS,
                },
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
