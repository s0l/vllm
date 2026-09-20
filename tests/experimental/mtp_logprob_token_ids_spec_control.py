#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact-image control for speculative ``logprob_token_ids`` coverage."""

from __future__ import annotations

import json

import numpy as np
import torch

from vllm import _C_stable_libtorch  # noqa: F401
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


def main() -> int:
    device = torch.device("cuda")
    req_state_idx = 2
    requested_token_id = 11
    token_id_state = LogprobTokenIdsState(max_num_reqs=4, device=device)
    token_id_state.add_request(
        req_state_idx,
        SamplingParams(logprob_token_ids=[requested_token_id]),
    )
    token_id_state.apply_staged_writes()

    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.enable_adaptive_verification = False
    rejection_sampler.sampler = type(
        "SamplerState",
        (),
        {
            "logprobs_mode": "raw_logprobs",
            "logprob_token_ids_state": token_id_state,
        },
    )()
    sampled = torch.tensor([[5, 6, 7]], device=device, dtype=torch.int64)
    num_sampled = torch.tensor([3], device=device, dtype=torch.int32)
    logits = torch.full((3, 17), -8.0, device=device)
    logits[:, 3] = 8.0
    logits[:, requested_token_id] = -2.0
    cu_num_logits_np = np.array([0, 3], dtype=np.int32)
    cu_num_logits = torch.from_numpy(cu_num_logits_np).to(device)
    expanded_idx_mapping = torch.full(
        (3,), req_state_idx, device=device, dtype=torch.int64
    )

    result = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        logits,
        cu_num_logits,
        cu_num_logits_np,
        max_num_logprobs=1,
        expanded_idx_mapping=expanded_idx_mapping,
        idx_mapping_np=np.array([req_state_idx], dtype=np.int32),
    )
    assert result is not None
    selected = result.logprob_token_ids[:, 0].tolist()
    requested = result.logprob_token_ids[:, 1].tolist()
    assert selected == [5, 6, 7]
    assert requested == [requested_token_id] * 3
    assert result.logprobs.shape == (3, 2)

    # The natural top-1 is deliberately token 3, so this control would fail if
    # speculative decode silently used top-k instead of the requested token.
    assert requested_token_id != 3
    print(
        json.dumps(
            {
                "passed": True,
                "positions": len(selected),
                "selected_token_ids": selected,
                "requested_token_ids": requested,
                "natural_top1_token_id": 3,
                "negative_control_nontrivial": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
