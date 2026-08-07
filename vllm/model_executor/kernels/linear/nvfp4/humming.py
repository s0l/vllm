# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.humming_utils import (
    prepare_humming_layer,
)
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

_STABLE_K64_LANGUAGE_SHAPES = {
    (5632, 5120),  # GDN qkvz
    (5120, 2048),  # GDN/full-attention output projection
    (6144, 5120),  # full-attention qkv under TP3/DCP3
    (11648, 5120),  # fused MLP gate/up
    (5120, 5824),  # row-parallel MLP down
}


def _use_stable_k64(logical_n: int, logical_k: int) -> bool:
    return (logical_n, logical_k) in _STABLE_K64_LANGUAGE_SHAPES


def _use_narrow_pad64(logical_n: int, logical_k: int) -> bool:
    return (logical_n, logical_k) == (32, 5120)


class HummingNvFp4LinearKernel(NvFp4LinearKernel):
    """Humming GEMM Kernel for NVFP4."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Humming only supported on CUDA"

        if not current_platform.has_device_capability(75):
            return False, "Humming only supported on SM75+"

        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Route through humming's compressed-tensors nvfp4 loader (same path as
        # the MoE oracle); the native group_tensor schema mishandles a scalar
        # global scale.
        quant_config = {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "type": "float",
            "num_bits": 4,
            "strategy": "group",
            "group_size": 16,
        }
        # CT pack-quantized reads `weight_packed`; the scheme renamed it to `weight`.
        if not hasattr(layer, "weight_packed"):
            layer.weight_packed = layer.weight
            del layer.weight

        # Humming requires both the physical N dimension and the amount of
        # internal N padding to be multiples of 8. TP3 can produce logical
        # widths that are not multiples of eight. Add zero-valued physical rows
        # up to the next
        # multiple of 8, then slice them from the result in apply_weights.
        logical_output_size = sum(layer.output_partition_sizes)
        logical_input_size = layer.input_size_per_partition
        physical_row_padding = (-logical_output_size) % 8
        if physical_row_padding:
            for name in ("weight_packed", "weight_scale"):
                value = getattr(layer, name)
                padding = value.new_zeros((physical_row_padding, *value.shape[1:]))
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(
                        torch.cat((value.detach(), padding), dim=0),
                        requires_grad=False,
                    ),
                )
            layer.humming_logical_output_size = logical_output_size
            layer.output_partition_sizes = list(layer.output_partition_sizes)
            layer.output_partition_sizes[-1] += physical_row_padding
            layer.output_size_per_partition += physical_row_padding
        # The CT linear scheme inverts the global scale (1/scale) for
        # marlin/cutlass; humming wants the original.
        layer.weight_global_scale = torch.nn.Parameter(
            1.0 / layer.weight_global_scale, requires_grad=False
        )
        use_narrow_pad64 = (
            envs.AG2_VLLM_NVFP4_HUMMING_NARROW_PAD64
            and _use_narrow_pad64(logical_output_size, logical_input_size)
        )
        prepare_humming_layer(
            layer,
            quant_config,
            pad_n_to_multiple=64 if use_narrow_pad64 else 256,
        )
        if use_narrow_pad64:
            logger.warning_once(
                "NVFP4 Humming narrow N padding reduced from 256 to 64 for "
                "the proven TP3 GDN ba geometry."
            )

        if (
            envs.AG2_VLLM_NVFP4_HUMMING_STABLE_K64
            and _use_stable_k64(logical_output_size, logical_input_size)
        ):
            from vllm.utils.humming import HummingMethod

            configs = copy.deepcopy(
                HummingMethod.get_default_tuning_configs(
                    layer,
                    use_batch_invariant=True,
                )
            )
            for _, _, config in configs:
                block_m, block_n, _ = config["block_shape"]
                warp_m, warp_n, _ = config["warp_shape"]
                config["block_shape"] = (block_m, block_n, 64)
                config["warp_shape"] = (warp_m, warp_n, 64)
                config["use_stream_k"] = False
            layer.ag2_humming_tuning_config = json.dumps(configs)
            logger.warning_once(
                "NVFP4 Humming stable K64 schedule enabled for proven TP3 "
                "language projection geometries; Stream-K remains disabled."
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.humming import HummingMethod

        flatten_inputs = x.view(-1, x.size(-1))
        output = HummingMethod.forward_layer(
            layer=layer,
            inputs=flatten_inputs,
            compute_config=layer.compute_config,
            tuning_config=getattr(layer, "ag2_humming_tuning_config", None),
        )
        output = output.view(*x.shape[:-1], output.size(-1))
        logical_output_size = getattr(layer, "humming_logical_output_size", None)
        if logical_output_size is not None:
            output = output[..., :logical_output_size]
        return output
