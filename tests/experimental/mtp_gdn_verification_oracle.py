"""Compare multi-token GDN verification with sequential BF16 decode semantics."""

from __future__ import annotations

import argparse
import json

import torch

from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)


def tensor_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - reference.float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-dtype", choices=("bf16", "fp32"), default="bf16"
    )
    args = parser.parse_args()
    torch.manual_seed(212)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    state_dtype = (
        torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
    )
    query_len = 3
    num_k_heads = 6
    num_v_heads = 18
    head_dim = 128

    q = torch.randn(
        1, query_len, num_k_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, query_len, num_v_heads, head_dim, device=device, dtype=dtype
    )
    a = torch.randn(query_len, num_v_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(num_v_heads, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn_like(a_log) * 0.1
    initial = (
        torch.randn(
            num_v_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=state_dtype,
        )
        * 0.02
    )

    multi_state = torch.zeros(
        query_len + 1,
        num_v_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=state_dtype,
    )
    multi_state[1] = initial
    multi_indices = torch.arange(
        1, query_len + 1, device=device, dtype=torch.int32
    ).view(1, query_len)
    multi_out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=multi_state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, query_len], device=device, dtype=torch.int32),
        ssm_state_indices=multi_indices,
        num_accepted_tokens=torch.ones(1, device=device, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
    )

    sequential_state = torch.zeros_like(multi_state)
    sequential_state[1] = initial
    sequential_outputs = []
    sequential_states = []
    one_index = torch.ones(1, device=device, dtype=torch.int32)
    one_cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    for token_idx in range(query_len):
        output, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a[token_idx : token_idx + 1],
            b=b[token_idx : token_idx + 1],
            dt_bias=dt_bias,
            q=q[:, token_idx : token_idx + 1],
            k=k[:, token_idx : token_idx + 1],
            v=v[:, token_idx : token_idx + 1],
            initial_state=sequential_state,
            inplace_final_state=True,
            cu_seqlens=one_cu,
            ssm_state_indices=one_index,
            use_qk_l2norm_in_kernel=True,
        )
        sequential_outputs.append(output)
        sequential_states.append(sequential_state[1].clone())
    sequential_out = torch.cat(sequential_outputs, dim=1)
    sequential_state_history = torch.stack(sequential_states)

    output_error = tensor_error(multi_out, sequential_out)
    state_error = tensor_error(multi_state[1:4], sequential_state_history)
    result = {
        "shape": {
            "query_len": query_len,
            "num_k_heads": num_k_heads,
            "num_v_heads": num_v_heads,
            "head_dim": head_dim,
            "state_dtype": str(state_dtype),
        },
        "output_error": output_error,
        "state_error": state_error,
        "bit_exact_output": torch.equal(multi_out, sequential_out),
        "bit_exact_state": torch.equal(
            multi_state[1:4], sequential_state_history
        ),
    }
    print(json.dumps(result, indent=2))
    if not result["bit_exact_output"] or not result["bit_exact_state"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
