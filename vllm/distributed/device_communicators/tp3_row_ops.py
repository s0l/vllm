# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional, shape-dispatched boundaries for experimental TP3 row state.

Every call owns fresh outputs. Raw A/R survive until the successor consumes
C+(A+R); no tensor state or row partition survives into the next invocation.
"""

import torch

from vllm._custom_ops import create_fp4_output_tensors
from vllm.model_executor.kernels.linear.nvfp4.arc import ag2_nvfp4_arc_quantize
from vllm.utils.torch_utils import direct_register_custom_op

from .tp3_row_norm import RowNormSpec
from .tp3_row_plan import RowOwnerPlan, partition_rows
from .tp3_row_transport import RowSeam

ROW_MIN_M = 32


def _group(name):
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    if group.unique_name != name or group.world_size != 3:
        raise ValueError("row continuation requires its initialized TP3 group")
    return group


def _specs(role, config):
    if len(config) != 6:
        raise ValueError("three destination norm configurations are required")
    return tuple(RowNormSpec(role, config[2 * r], config[2 * r + 1]) for r in range(3))


def _validate(inputs, gamma, divisor=None, selected=()):
    shape, device = inputs[0].shape, inputs[0].device
    if len(shape) != 2 or shape[1] != 5120 or not 1 <= shape[0] <= 4096:
        raise ValueError("row boundary requires M1..4096/H5120")
    if any(
        x.shape != shape
        or x.dtype != torch.bfloat16
        or x.device != device
        or not x.is_contiguous()
        for x in inputs
    ):
        raise ValueError("incompatible raw row operands")
    if (
        gamma.shape != (5120,)
        or gamma.dtype != torch.bfloat16
        or gamma.device != device
        or not gamma.is_contiguous()
    ):
        raise ValueError("row boundary requires original BF16 gamma")
    if divisor is not None and (
        divisor.numel() != 1
        or divisor.dtype != torch.float32
        or divisor.device != device
    ):
        raise ValueError("row boundary requires one FP32 quantization multiplier")
    if divisor is not None and (
        len(selected) != 3
        or any(
            s.ndim != 1
            or s.dtype != torch.int32
            or s.device != device
            or s.numel() not in (0, 64, 256, 512)
            or not s.is_contiguous()
            for s in selected
        )
    ):
        raise ValueError("invalid destination ARC metadata")


def _reduce(value, group, selected):
    if value.shape[0] < ROW_MIN_M:
        # Preserve the accepted selector, including explicitly configured
        # backends, instead of duplicating its environment policy here.
        from vllm.distributed.parallel_state import tp3_unified_exact_reduce

        return tp3_unified_exact_reduce(value, group.unique_name), None
    plan = RowOwnerPlan(
        partition_rows(value.shape[0]), 5120, tuple(5120 + s.numel() for s in selected)
    )
    seam = RowSeam(plan, group.rank_in_group, group.device_group)
    reduced = torch.zeros_like(value)
    seam.owned(reduced).copy_(seam.reduce_owned(value))
    return reduced, seam


def _quantized(specs, inputs, gamma, divisor, selected, group, seam):
    rank = group.rank_in_group
    if seam is None:
        normalized, carry = specs[rank].apply(inputs, gamma)
        q, sf = ag2_nvfp4_arc_quantize(normalized, divisor, selected[rank])
        return q, sf, carry, normalized
    carry = None

    def normalized_for(destination):
        nonlocal carry
        normalized, current = specs[destination].apply(inputs, gamma)
        if destination == rank:
            carry = current
        return normalized

    q, sf = seam.quantized(normalized_for, divisor, selected, candidate=True)
    return q, sf, carry, None


def row_post(
    contribution: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    divisor: torch.Tensor,
    selected: list[torch.Tensor],
    config: list[int],
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate((contribution, residual), gamma, divisor, selected)
    specs, group = _specs("post", config), _group(group_name)
    attention, seam = _reduce(contribution, group, selected)
    q, sf, _, _ = _quantized(
        specs, (attention, residual), gamma, divisor, selected, group, seam
    )
    return q, sf, attention


def row_next(
    contribution: torch.Tensor,
    attention: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    divisor: torch.Tensor,
    selected: list[torch.Tensor],
    config: list[int],
    base_needed: bool,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate((contribution, attention, residual), gamma, divisor, selected)
    specs, group = _specs("next", config), _group(group_name)
    mlp, seam = _reduce(contribution, group, selected)
    q, sf, carry, normalized = _quantized(
        specs, (mlp, attention, residual), gamma, divisor, selected, group, seam
    )
    if base_needed:
        if seam is None:
            empty = torch.empty(0, dtype=torch.int32, device=contribution.device)
            bq, bsf = ag2_nvfp4_arc_quantize(normalized, divisor, empty)
        else:
            bq, bsf = seam.base_quantized(q, sf)
            # A base-only destination may otherwise alias a returned q tensor.
            if selected[group.rank_in_group].numel() == 0:
                bq = bq.clone()
    else:
        bq = torch.empty(0, dtype=torch.uint8, device=contribution.device)
        bsf = torch.empty(0, dtype=torch.float8_e4m3fn, device=contribution.device)
    return q, sf, bq, bsf, carry


def row_first(
    value: torch.Tensor, gamma: torch.Tensor, config: list[int], group_name: str
) -> torch.Tensor:
    _validate((value,), gamma)
    group = _group(group_name)
    return _specs("first", config)[group.rank_in_group].apply((value,), gamma)[0]


def row_terminal(
    contribution: torch.Tensor,
    attention: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    config: list[int],
    group_name: str,
) -> torch.Tensor:
    _validate((contribution, attention, residual), gamma)
    specs, group = _specs("terminal", config), _group(group_name)
    selected = [torch.empty(0, dtype=torch.int32, device=contribution.device)] * 3
    mlp, seam = _reduce(contribution, group, selected)
    inputs = (mlp, attention, residual)
    if seam is None:
        return specs[group.rank_in_group].apply(inputs, gamma)[0]
    # A contiguous row slice is still a view retaining the full-M allocation.
    owned = [seam.owned(spec.apply(inputs, gamma)[0]).clone() for spec in specs]
    return seam.publish_destinations(owned)


def _packed_fake(value, selected, group_name):
    rank = _group(group_name).rank_in_group
    q, sf = create_fp4_output_tensors(
        value.shape[0],
        5120,
        value.device,
        True,
        padded_n=5120 + selected[rank].shape[0],
    )
    return q, sf.view(torch.float8_e4m3fn)


def row_post_fake(contribution, residual, gamma, divisor, selected, config, group_name):
    q, sf = _packed_fake(contribution, selected, group_name)
    return q, sf, torch.empty_like(contribution)


def row_next_fake(
    contribution,
    attention,
    residual,
    gamma,
    divisor,
    selected,
    config,
    base_needed,
    group_name,
):
    q, sf = _packed_fake(contribution, selected, group_name)
    if base_needed:
        bq, bsf = create_fp4_output_tensors(
            contribution.shape[0], 5120, contribution.device, True
        )
        bsf = bsf.view(torch.float8_e4m3fn)
    else:
        bq = torch.empty(0, dtype=torch.uint8, device=contribution.device)
        bsf = torch.empty(0, dtype=torch.float8_e4m3fn, device=contribution.device)
    return q, sf, bq, bsf, torch.empty_like(contribution)


def row_first_fake(value, gamma, config, group_name):
    return torch.empty_like(value)


def row_terminal_fake(contribution, attention, residual, gamma, config, group_name):
    return torch.empty_like(contribution)


for _name, _real, _fake in (
    ("tp3_row_post", row_post, row_post_fake),
    ("tp3_row_next", row_next, row_next_fake),
    ("tp3_row_first", row_first, row_first_fake),
    ("tp3_row_terminal", row_terminal, row_terminal_fake),
):
    direct_register_custom_op(op_name=_name, op_func=_real, fake_impl=_fake)
