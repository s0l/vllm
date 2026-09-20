# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded source padding for checkpoint FP8-block MTP experts."""

from collections.abc import Iterable
from copy import copy, deepcopy
from math import lcm

import regex as re
import torch

from vllm.utils.math_utils import round_up


class MTPExpertLoader:
    def __init__(self, *, hidden, width, tp, block, layers, experts):
        if (
            any(
                type(v) is not int or v <= 0
                for v in (hidden, width, tp, block, layers, experts)
            )
            or hidden % block
            or width % block
        ):
            raise ValueError("invalid MTP FP8 block partition")
        self.hidden, self.width, self.tp, self.block = hidden, width, tp, block
        self.layers, self.experts = layers, experts
        # Fp8MoEMethod already refines scales losslessly to blocks >=32.
        # Preserve whole source blocks and let its loader expand the grid.
        self.physical = round_up(width, lcm(tp * 32, block))
        self.seen: set[tuple[int, int, str, str]] = set()
        self.finished = self.closed = False

    def transform(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if self.finished or self.closed:
            raise RuntimeError("MTP expert reload requires a new worker")
        try:
            for name, tensor in weights:
                if ".mlp.experts." not in "." + name:
                    yield name, tensor
                    continue
                match = re.fullmatch(
                    r"layers\.(\d+)\.mlp\.experts\.(\d+)\."
                    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)",
                    name,
                )
                if match is None:
                    raise ValueError("unsupported MTP expert source field")
                layer, expert = int(match[1]), int(match[2])
                projection, field = match[3], match[4]
                key = layer, expert, projection, field
                if not 0 <= layer < self.layers or not 0 <= expert < self.experts:
                    raise ValueError("MTP expert source index outside model geometry")
                if key in self.seen:
                    raise ValueError("duplicate MTP expert source field")
                down, scale = projection == "down_proj", field == "weight_scale_inv"
                unit = self.block if scale else 1
                h, w, p = (v // unit for v in (self.hidden, self.width, self.physical))
                shape = (h, w) if down else (w, h)
                dtypes = (
                    (torch.bfloat16, torch.float32) if scale else (torch.float8_e4m3fn,)
                )
                if tuple(tensor.shape) != shape or tensor.dtype not in dtypes:
                    raise ValueError("MTP expert source shape/dtype mismatch")
                if not torch.isfinite(tensor.float()).all() or (
                    scale and not (tensor > 0).all()
                ):
                    raise ValueError("invalid MTP expert source values")
                if p != w:
                    padded_shape = (h, p) if down else (p, h)
                    if scale:
                        padded = torch.ones(
                            padded_shape, dtype=tensor.dtype, device=tensor.device
                        )
                    else:
                        padded = torch.zeros(
                            padded_shape, dtype=torch.uint8, device=tensor.device
                        ).view(tensor.dtype)
                    if down:
                        padded[:, :w].copy_(tensor)
                    else:
                        padded[:w].copy_(tensor)
                    tensor = padded
                self.seen.add(key)
                yield name, tensor
        except BaseException:
            self.closed = True
            raise

    def finish(self):
        if self.finished or self.closed:
            raise RuntimeError("MTP expert reload requires a new worker")
        if len(self.seen) != self.layers * self.experts * 6:
            self.closed = True
            raise ValueError("incomplete MTP FP8 expert source coverage")
        self.finished = True


def partition_native_mtp(draft, *, start_layer, layers):
    """Isolate physical expert geometry/backend from target and source configs."""
    extra = draft.additional_config
    if not isinstance(extra, dict) or "flashnext_native_experts" not in extra:
        return draft, None
    entries = getattr(draft.quant_config, "quantized_layers", {})
    algorithms = [
        entries.get(f"mtp.layers.{start_layer + i}.mlp.experts", {})
        for i in range(layers)
    ]
    if any(
        info.get("quant_algo") not in ("FP8_BLOCK_SCALES", "FP8_PB_WO")
        or info.get("group_size", 128) != 128
        for info in algorithms
    ):
        raise ValueError("native FlashNext requires FP8 block128 MTP experts")
    text = draft.model_config.hf_text_config
    loader = MTPExpertLoader(
        hidden=text.hidden_size,
        width=text.moe_intermediate_size,
        tp=draft.parallel_config.tensor_parallel_size,
        block=128,
        layers=layers,
        experts=text.num_experts,
    )
    result = copy(draft)
    result.model_config = copy(draft.model_config)
    result.model_config.hf_config = deepcopy(draft.model_config.hf_config)
    result.model_config.hf_text_config = result.model_config.hf_config.get_text_config()
    result.model_config.hf_text_config.moe_intermediate_size = loader.physical
    result.kernel_config = copy(draft.kernel_config)
    result.kernel_config.moe_backend = "triton"
    return result, loader
