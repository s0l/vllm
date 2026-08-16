"""Compare the patched no-store verifier against the installed exact kernel."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update as installed_update,
)


QUERY_LEN = 4
NUM_REQUESTS = 1
NUM_K_HEADS = 6
NUM_V_HEADS = 18
HEAD_DIM = 128


def _load_candidate():
    source = (
        Path("/workspace/vllm-src")
        / "vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py"
    )
    name = "ag2_mtp_gdn_no_store_candidate"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.fused_sigmoid_gating_delta_rule_update


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "differing": int(torch.count_nonzero(delta).item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(212)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    total_tokens = NUM_REQUESTS * QUERY_LEN
    q = torch.randn(
        1,
        total_tokens,
        NUM_K_HEADS,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    k = torch.randn_like(q)
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
    a_log = torch.randn(
        NUM_V_HEADS, device=device, dtype=torch.float32
    ).mul_(0.1)
    dt_bias = torch.randn_like(a_log).mul_(0.1)
    initial = torch.randn(
        NUM_REQUESTS,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    ).mul_(0.02)
    query_start = torch.arange(
        0,
        total_tokens + 1,
        QUERY_LEN,
        device=device,
        dtype=torch.int32,
    )
    state_indices = torch.arange(
        1,
        NUM_REQUESTS * QUERY_LEN + 1,
        device=device,
        dtype=torch.int32,
    ).view(NUM_REQUESTS, QUERY_LEN)
    accepted_previous = torch.ones(
        NUM_REQUESTS, device=device, dtype=torch.int32
    )

    base = torch.zeros(
        1 + NUM_REQUESTS * QUERY_LEN,
        NUM_V_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    for req_idx in range(NUM_REQUESTS):
        base[state_indices[req_idx, 0]].copy_(initial[req_idx])

    installed_pool = base.clone()
    installed_out, _ = installed_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=installed_pool,
        inplace_final_state=True,
        cu_seqlens=query_start,
        ssm_state_indices=state_indices,
        num_accepted_tokens=accepted_previous,
        use_qk_l2norm_in_kernel=True,
    )

    candidate_update = _load_candidate()
    default_pool = base.clone()
    default_out, _ = candidate_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=default_pool,
        inplace_final_state=True,
        cu_seqlens=query_start,
        ssm_state_indices=state_indices,
        num_accepted_tokens=accepted_previous,
        use_qk_l2norm_in_kernel=True,
    )
    default_output_error = _error(default_out, installed_out)
    default_state_error = _error(default_pool, installed_pool)
    expected_final_state = installed_pool.clone()
    del installed_pool, default_pool
    torch.cuda.empty_cache()

    no_store_pool = base.clone()
    no_store_out, returned_state = candidate_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=no_store_pool,
        inplace_final_state=True,
        store_final_state=False,
        cu_seqlens=query_start,
        ssm_state_indices=state_indices,
        num_accepted_tokens=accepted_previous,
        use_qk_l2norm_in_kernel=True,
    )
    state_only_pool = base.clone()
    state_only_out, _ = candidate_update(
        A_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=state_only_pool,
        inplace_final_state=True,
        store_output=False,
        cu_seqlens=query_start,
        ssm_state_indices=state_indices,
        num_accepted_tokens=accepted_previous,
        use_qk_l2norm_in_kernel=True,
    )

    result = {
        "default_output": default_output_error,
        "default_state": default_state_error,
        "no_store_output": _error(no_store_out, installed_out),
        "no_store_input_unchanged": torch.equal(
            no_store_pool, base
        ),
        "no_store_returns_input": returned_state.data_ptr()
        == no_store_pool.data_ptr(),
        "state_only_state": _error(state_only_pool, expected_final_state),
        "state_only_reuses_q": state_only_out.data_ptr() == q.data_ptr(),
    }
    passed = bool(
        result["default_output"]["differing"] == 0
        and result["default_state"]["differing"] == 0
        and result["no_store_output"]["differing"] == 0
        and result["no_store_input_unchanged"]
        and result["no_store_returns_input"]
        and result["state_only_state"]["differing"] == 0
        and result["state_only_reuses_q"]
    )
    print(json.dumps({"passed": passed, **result}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
