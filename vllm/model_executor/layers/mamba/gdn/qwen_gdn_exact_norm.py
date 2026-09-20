# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""POC fixed-recipe GDN norm; no global compiler policy or executable cache."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from .qwen_gdn_exact_norm_kernels import gated_h15, gated_h18, sum_h15


def validate_inputs(core, projected, weight):
    if (
        core.ndim != 3
        or core.shape[1] not in (15, 18)
        or core.shape[2] != 128
        or not 1 <= core.shape[0] <= 4096
        or projected.shape != (core.shape[0], 6144)
        or weight.shape != (128,)
        or any(t.dtype != torch.bfloat16 for t in (core, projected, weight))
        or any(t.device != core.device for t in (projected, weight))
        or not all(t.is_contiguous() for t in (core, projected, weight))
    ):
        raise ValueError("unproved exact GDN norm input/recipe")


def exact_gdn_norm(
    core: torch.Tensor, projected: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    validate_inputs(core, projected, weight)
    if not core.is_cuda:
        raise ValueError("exact GDN norm is a CUDA-only recipe")
    rows, heads, _ = core.shape
    output = torch.empty((rows, 2304), dtype=core.dtype, device=core.device)
    count = rows * heads
    # These are math-recipe constants from admitted original target artifacts,
    # not an autotuning search. No mutable launch policy crosses invocations.
    if heads == 18:
        gated_h18[((count + 7) // 8,)](
            core,
            weight,
            projected,
            output,
            count,
            128,
            XBLOCK=8,
            num_warps=2,
            num_stages=1,
            enable_fp_fusion=True,
        )
    else:
        sums = torch.empty((count, 1), dtype=torch.float32, device=core.device)
        sum_h15[((count + 7) // 8,)](
            core,
            sums,
            count,
            128,
            XBLOCK=8,
            num_warps=2,
            num_stages=1,
            enable_fp_fusion=True,
        )
        gated_h15[((rows * 2304 + 1023) // 1024,)](
            core,
            sums,
            weight,
            projected,
            output,
            rows * 2304,
            rows,
            XBLOCK=1024,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=True,
        )
    return output


def exact_gdn_norm_fake(core, projected, weight):
    return core.new_empty((core.shape[0], 2304))


direct_register_custom_op(
    op_name="ready_exact_gdn_norm",
    op_func=exact_gdn_norm,
    fake_impl=exact_gdn_norm_fake,
)
