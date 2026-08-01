# SPDX-License-Identifier: Apache-2.0
"""Compare one-shot and resumed execution of the vendored FLA GDN kernel.

This is a deterministic contract control for chunked-prefill lifecycle state.
The same projected inputs are evaluated either as one complete sequence or as
two consecutive segments connected by the first segment's final FP32 state.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    left_float = left.float()
    delta = left_float - right.float()
    return {
        "exact": bool(torch.equal(left, right)),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(left_float).clamp_min(1e-30)
        ),
    }


def _run(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    assert final_state is not None
    return output, final_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1488)
    parser.add_argument("--splits", default="1,32,64,86,128,256,1024")
    parser.add_argument("--seed", type=int, default=212)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    heads = 16
    key_dim = value_dim = 128
    shape = (1, args.length, heads)

    q = F.normalize(
        torch.randn(*shape, key_dim, device=device, dtype=dtype).float(), dim=-1
    ).to(dtype)
    k = F.normalize(
        torch.randn(*shape, key_dim, device=device, dtype=dtype).float(), dim=-1
    ).to(dtype)
    v = torch.randn(*shape, value_dim, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(*shape, device=device, dtype=torch.float32))
    beta = torch.sigmoid(
        torch.randn(*shape, device=device, dtype=torch.float32)
    ).to(dtype)
    zero_state = torch.zeros(
        1,
        heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )

    full_output, full_state = _run(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=zero_state
    )
    splits = [int(value) for value in args.splits.split(",")]
    results: list[dict[str, object]] = []
    prefix_states: dict[int, torch.Tensor] = {}
    for split in splits:
        if not 0 < split < args.length:
            raise ValueError(f"split must be within the sequence: {split}")
        first_output, first_state = _run(
            q=q[:, :split],
            k=k[:, :split],
            v=v[:, :split],
            g=g[:, :split],
            beta=beta[:, :split],
            initial_state=zero_state,
        )
        prefix_states[split] = first_state
        second_output, resumed_state = _run(
            q=q[:, split:],
            k=k[:, split:],
            v=v[:, split:],
            g=g[:, split:],
            beta=beta[:, split:],
            initial_state=first_state,
        )
        # Production defaults mamba_ssm_cache_dtype=auto to the BF16 model
        # dtype.  The FLA kernel returns FP32 final state, but the cache write
        # rounds it to BF16 before a later chunk gathers it again.
        cached_first_state = first_state.to(dtype)
        cached_second_output, cached_resumed_state = _run(
            q=q[:, split:],
            k=k[:, split:],
            v=v[:, split:],
            g=g[:, split:],
            beta=beta[:, split:],
            initial_state=cached_first_state,
        )
        resumed_output = torch.cat((first_output, second_output), dim=1)
        results.append(
            {
                "split": split,
                "output": _difference(full_output, resumed_output),
                "suffix_output": _difference(
                    full_output[:, split:], second_output
                ),
                "final_state": _difference(full_state, resumed_state),
                "bf16_cache_handoff": {
                    "suffix_output": _difference(
                        full_output[:, split:], cached_second_output
                    ),
                    "final_state": _difference(
                        full_state, cached_resumed_state
                    ),
                },
                "probe_1400": _difference(
                    full_output[:, 1400], resumed_output[:, 1400]
                ),
            }
        )

    # Live chunked prefill resumes multiple requests as one variable-length
    # packed FLA invocation.  A sequence can be exact when resumed alone yet
    # still be corrupted by packed chunk metadata or a preceding sequence's
    # offset.  Pack every tested suffix with its authoritative prefix state and
    # compare each row with the corresponding slice of the one-shot reference.
    suffix_lengths = [args.length - split for split in splits]
    packed_cu = torch.tensor(
        [0, *torch.tensor(suffix_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    packed_q = torch.cat([q[:, split:] for split in splits], dim=1)
    packed_k = torch.cat([k[:, split:] for split in splits], dim=1)
    packed_v = torch.cat([v[:, split:] for split in splits], dim=1)
    packed_g = torch.cat([g[:, split:] for split in splits], dim=1)
    packed_beta = torch.cat([beta[:, split:] for split in splits], dim=1)
    packed_initial_state = torch.cat(
        [prefix_states[split] for split in splits], dim=0
    )
    packed_output, packed_final_state = chunk_gated_delta_rule(
        q=packed_q,
        k=packed_k,
        v=packed_v,
        g=packed_g,
        beta=packed_beta,
        initial_state=packed_initial_state,
        output_final_state=True,
        cu_seqlens=packed_cu,
        use_qk_l2norm_in_kernel=False,
    )
    assert packed_final_state is not None
    packed_cached_output, packed_cached_final_state = chunk_gated_delta_rule(
        q=packed_q,
        k=packed_k,
        v=packed_v,
        g=packed_g,
        beta=packed_beta,
        initial_state=packed_initial_state.to(dtype),
        output_final_state=True,
        cu_seqlens=packed_cu,
        use_qk_l2norm_in_kernel=False,
    )
    assert packed_cached_final_state is not None
    packed_results: list[dict[str, object]] = []
    for index, split in enumerate(splits):
        packed_start = int(packed_cu[index].item())
        packed_end = int(packed_cu[index + 1].item())
        packed_results.append(
            {
                "split": split,
                "packed_offset": packed_start,
                "output": _difference(
                    full_output[:, split:],
                    packed_output[:, packed_start:packed_end],
                ),
                "final_state": _difference(
                    full_state,
                    packed_final_state[index : index + 1],
                ),
                "bf16_cache_handoff": {
                    "output": _difference(
                        full_output[:, split:],
                        packed_cached_output[:, packed_start:packed_end],
                    ),
                    "final_state": _difference(
                        full_state,
                        packed_cached_final_state[index : index + 1],
                    ),
                },
            }
        )

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "length": args.length,
                "results": results,
                "packed_cu_seqlens": packed_cu.tolist(),
                "packed_results": packed_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
