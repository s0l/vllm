# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone exact-image controls for the shared FP8 lm_head POC."""

import torch
from torch import nn

from vllm.model_executor.models.ag2_fp8_draft_head import (
    Ag2SharedFp8LMHeadMethod,
    install_shared_fp8_lm_head,
)


def main() -> None:
    torch.manual_seed(29)
    lm_head = nn.Module().cuda()
    source = (torch.randn(256, 128, device="cuda") * 0.07).to(torch.bfloat16)
    lm_head.register_parameter(
        "weight", nn.Parameter(source.clone(), requires_grad=False)
    )
    lm_head.quant_method = object()

    target = nn.Module()
    draft = nn.Module()
    target.lm_head = lm_head
    draft.lm_head = lm_head
    source_bytes, installed_bytes = install_shared_fp8_lm_head(draft.lm_head)

    assert target.lm_head is draft.lm_head
    assert target.lm_head.weight.dtype == torch.float8_e4m3fn
    assert isinstance(target.lm_head.quant_method, Ag2SharedFp8LMHeadMethod)
    assert source_bytes == source.numel() * 2
    assert installed_bytes == source.numel() + source.shape[0] * 4

    x = (torch.randn(3, 128, device="cuda") * 0.11).to(torch.bfloat16)
    reference = x @ source.T
    actual = target.lm_head.quant_method.apply(target.lm_head, x)
    diff = (actual.float() - reference.float()).abs()
    assert torch.isfinite(actual.float()).all()
    assert float(diff.mean()) < 0.02
    assert float(diff.max()) < 0.2

    original_scale = target.lm_head.ag2_fp8_weight_scale.clone()
    target.lm_head.ag2_fp8_weight_scale.mul_(2)
    corrupted = target.lm_head.quant_method.apply(target.lm_head, x)
    target.lm_head.ag2_fp8_weight_scale.copy_(original_scale)
    negative_diff = (corrupted.float() - reference.float()).abs()
    assert float(negative_diff.mean()) > float(diff.mean()) * 10

    assert install_shared_fp8_lm_head(target.lm_head) == (0, 0)
    print(
        "PASS shared module, replacement bytes, finite projection, "
        "numerical bound, negative scale control, idempotence"
    )


if __name__ == "__main__":
    main()
