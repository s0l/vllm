# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit P/M invocation ABI for the opt-in exact row target POC."""

import torch


def ready_compaction_enabled(config) -> bool:
    additional = config.additional_config
    value = (
        additional.get("ready_target_compaction", 0)
        if isinstance(additional, dict)
        else 0
    )
    if type(value) is not int or value not in (0, 1):
        raise ValueError("ready_target_compaction must be recipe version0 or1")
    return value == 1


def bind_ready_rows(model_inputs, *, enabled, actual_rows, physical_rows, full_graph):
    """Metadata-only; no sync, new CUDA allocation or caller dictionary mutation.

    FULL Graph has a fixed execution extent P. Compiled-only execution may
    consume actual M. Both paths pass the same tensor-valued argument lifetime.
    """
    if not enabled:
        return model_inputs
    if (
        type(actual_rows) is not int
        or type(physical_rows) is not int
        or type(full_graph) is not bool
        or not 1 <= actual_rows <= physical_rows <= 4096
        or model_inputs.get("intermediate_tensors") is not None
    ):
        raise ValueError("invalid ready target invocation or PP")
    positions = model_inputs.get("positions")
    if (
        not isinstance(positions, torch.Tensor)
        or positions.ndim not in (1, 2)
        or positions.shape[-1] != physical_rows
        or positions.dtype != torch.int64
        or positions.stride(-1) != 1
        or (positions.ndim == 2 and positions.shape[0] != 3)
    ):
        raise ValueError("invalid physical ready positions")
    for key in ("input_ids", "inputs_embeds"):
        value = model_inputs.get(key)
        if value is not None and (
            not isinstance(value, torch.Tensor)
            or value.ndim != (1 if key == "input_ids" else 2)
            or value.shape[0] != physical_rows
            or value.device != positions.device
        ):
            raise ValueError("invalid physical ready input")
    if (
        model_inputs.get("input_ids") is None
        and model_inputs.get("inputs_embeds") is None
    ):
        raise ValueError("ready target requires token ids or embeddings")
    rows = physical_rows if full_graph else actual_rows
    linear = positions[0] if positions.ndim == 2 else positions
    result = dict(model_inputs)
    result["ready_rows"] = linear[:rows]
    return result


def compact_inputs(input_ids, positions, inputs_embeds, ready_rows):
    # This adapter runs outside the compiled backbone but inside the owner's
    # complete Graph capture. Never propagate independent P/M constraints into
    # piecewise AOT, whose later pieces consume only M.
    parent = positions.shape[-1]
    useful = ready_rows.shape[0]
    torch._check(useful >= 1)
    torch._check(useful <= parent)
    if input_ids is not None:
        torch._check(input_ids.shape[0] == parent)
        input_ids = input_ids[:useful]
    if inputs_embeds is not None:
        torch._check(inputs_embeds.shape[0] == parent)
        inputs_embeds = inputs_embeds[:useful]
    return input_ids, positions[..., :useful], inputs_embeds, parent


def physical_output(output, parent):
    torch._check(output.shape[0] <= parent)
    if output.shape[0] == parent:
        return output
    # MTP's hidden-state copy still consumes P. Tail is not a real token and
    # must remain inert to verifier, slots and sampling; never uninitialized.
    return torch.cat(
        (output, output.new_zeros((parent - output.shape[0], output.shape[1])))
    )
