# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_embedding_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """Use the phase-stable reduction policy for vocabulary embeddings."""
    group = get_tp_group()
    if group.world_size == 1:
        return input_
    if group.use_custom_op_call:
        return torch.ops.vllm.embedding_all_reduce(input_, group_name=group.unique_name)
    from .parallel_state import embedding_all_reduce

    return embedding_all_reduce(input_, group.unique_name)


def tensor_model_parallel_gdn_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """Use the GDN-specific runtime reduction policy."""
    group = get_tp_group()
    if group.world_size == 1:
        return input_
    if group.use_custom_op_call:
        return torch.ops.vllm.gdn_all_reduce(
            input_,
            group_name=group.unique_name,
        )
    from .parallel_state import gdn_all_reduce

    return gdn_all_reduce(input_, group.unique_name)


def tensor_model_parallel_unified_exact_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    """Use the common exact TP3 arithmetic for every decoder reduction row."""
    group = get_tp_group()
    if group.world_size == 1:
        return input_
    if group.use_custom_op_call:
        return torch.ops.vllm.tp3_unified_exact_reduce(
            input_, group_name=group.unique_name
        )
    from .parallel_state import tp3_unified_exact_reduce

    return tp3_unified_exact_reduce(input_, group.unique_name)


def tensor_model_parallel_owner_residual_arc_prequant(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    input_scale_inv: torch.Tensor,
    selected_all: torch.Tensor,
    route0: torch.Tensor,
    route1: torch.Tensor,
    route2: torch.Tensor,
    inverse_order: torch.Tensor,
    route_counts: list[int],
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    group = get_tp_group()
    if group.world_size != 3:
        raise ValueError("owner residual prequant requires TP3")
    return torch.ops.vllm.tp3_owner_residual_arc_prequant(
        contribution,
        residual_owner,
        weight,
        input_scale_inv,
        selected_all,
        route0,
        route1,
        route2,
        inverse_order,
        route_counts,
        group.unique_name,
        eps,
    )


def tensor_model_parallel_owner_terminal_norm(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    group = get_tp_group()
    if group.world_size != 3:
        raise ValueError("owner terminal norm requires TP3")
    return torch.ops.vllm.tp3_owner_terminal_norm(
        contribution,
        residual_owner,
        weight,
        group.unique_name,
        eps,
    )


def tensor_model_parallel_owner_materialize_aux(
    contribution: torch.Tensor,
    residual_owner: torch.Tensor,
) -> torch.Tensor:
    """Materialize the canonical full hidden state consumed by MTP."""
    group = get_tp_group()
    if group.world_size != 3:
        raise ValueError("owner auxiliary materialization requires TP3")
    return torch.ops.vllm.tp3_owner_materialize_aux(
        contribution,
        residual_owner,
        group.unique_name,
    )


def tensor_model_parallel_v1_block5_fused_add_rms_norm(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.tp3_v1_block5_fused_add_rms_norm(value, residual, weight, eps)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
