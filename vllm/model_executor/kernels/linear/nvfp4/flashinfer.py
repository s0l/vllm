# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_prepare_bf16_fp4_weights,
    flashinfer_scaled_fp4_mm,
    has_flashinfer,
    has_flashinfer_b12x_gemm,
    has_flashinfer_bf16_fp4,
)

from .arc import ag2_nvfp4_arc_quantize
from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class FlashInferCuteDslNvFp4W4A16LinearKernel(NvFp4LinearKernel):
    """BF16 x NVFP4 GEMM via FlashInfer's CuTe-DSL backend."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            if not current_platform.is_cuda():
                return False, "FlashInfer CuTe-DSL W4A16 requires CUDA"
            capability = current_platform.get_device_capability()
            if capability is None:
                return False, "CUDA compute capability is unavailable"
            compute_capability = capability.to_int()

        if compute_capability not in (100, 103) and not (
            compute_capability >= 120 and compute_capability < 130
        ):
            return False, "FlashInfer CuTe-DSL W4A16 requires sm_100 or sm_12x"
        if not has_flashinfer_bf16_fp4():
            return False, "FlashInfer CuTe-DSL BF16 x FP4 GEMM is unavailable"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        padded_weight, weights_padding_bytes = pad_nvfp4_weight_for_cutlass(
            layer.weight.data, alignment=64
        )
        swizzled_scale = swizzle_blockscale(layer.weight_scale.data)
        weight, weight_scale, global_scale = flashinfer_prepare_bf16_fp4_weights(
            padded_weight,
            swizzled_scale,
            layer.weight_global_scale.data.reshape(1),
            backend="cute-dsl",
        )
        assert global_scale is not None

        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.weight_global_scale = torch.nn.Parameter(
            global_scale, requires_grad=False
        )
        layer.weights_padding_cols = weights_padding_bytes

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(
                f"FlashInfer CuTe-DSL W4A16 requires BF16 input, got {x.dtype}"
            )

        output_size = layer.output_size_per_partition
        output_shape = [*x.shape[:-1], output_size]
        x_2d = x.reshape(-1, x.shape[-1])
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)
        if weights_padding_bytes:
            x_2d = torch.nn.functional.pad(x_2d, (0, weights_padding_bytes * 2))
        x_2d = x_2d.contiguous()

        out = torch.ops.vllm.flashinfer_mm_bf16_fp4(
            x_2d,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
        )
        out = slice_nvfp4_output(out, output_size)
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferCuteDslNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's cutedsl backend."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_device_capability_family(100):
            return False, "FlashInfer cutedsl requires sm_10x"
        if not has_flashinfer():
            return False, "FlashInfer required"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # cutedsl uses the same swizzled + padded layout as cutlass.
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutedsl",
        )

        x_fp4 = pad_nvfp4_activation_for_cutlass(
            x_fp4, getattr(layer, "weights_padding_cols", 0)
        )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend="cute-dsl",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferCutlassNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's CUTLASS wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            cutlass_fp4_supported,
        )

        if (
            cutlass_fp4_supported()
            and current_platform.has_device_capability(100)
            and has_flashinfer()
        ):
            return True, None
        return False, "FlashInfer + >=sm_100 required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def input_quant_key(self) -> QuantKey | None:
        """Advertise prequant input only for the default-off TP3 owner POC.

        ARC changes the activation K/layout per consumer.  Advertising this
        unconditionally would let generic fusion construct a canonical NVFP4
        activation and silently bypass the attached ARC sidecar.
        """
        if os.environ.get("AG2_VLLM_TP3_OWNER_PREQUANT", "0") == "1":
            return kNvfp4Dynamic
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            x_fp4, x_blockscale = qa.data, qa.scale
            x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_bytes)
            output_dtype = qa.orig_dtype
            output_shape = [*qa.orig_shape[:-1], output_size]
        else:
            assert isinstance(x, torch.Tensor)
            output_dtype = x.dtype
            output_shape = [*x.shape[:-1], output_size]
            x_fp4, x_blockscale = scaled_fp4_quant(
                x,
                layer.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend="flashinfer-cutlass",
                padded_n=x.shape[-1] + weights_padding_bytes * 2,
            )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend="cutlass",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferTrtllmNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if has_flashinfer():
            return True, None
        return False, "FlashInfer required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        epilogue_tile_m = 128

        layer.weight = torch.nn.Parameter(
            shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
            .reshape(weight_scale.shape)
            .view(torch.float8_e4m3fn),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-trtllm",
        )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend="trtllm",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferCudnnNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's cuDNN wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if has_flashinfer():
            return True, None
        return False, "FlashInfer required"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # cuDNN uses the same swizzled + padded layout as CUTLASS
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cudnn",
            padded_n=x.shape[-1] + weights_padding_bytes * 2,
        )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend="cudnn",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class FlashInferB12xNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via FlashInfer's b12x CuTe DSL warp-level MMA kernel (SM120+)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.has_device_capability(120) and has_flashinfer_b12x_gemm():
            return True, None
        return (
            False,
            (
                "FlashInfer b12x requires SM120+ and FlashInfer "
                "with Sm120BlockScaledDenseGemmKernel"
            ),
        )

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def input_quant_key(self, layer: torch.nn.Module | None = None) -> QuantKey | None:
        """Admit explicit consumers without enabling the legacy owner runtime."""
        if layer is not None and getattr(layer, "_ag2_row_prequant_input", False):
            return kNvfp4Dynamic
        if os.environ.get("AG2_VLLM_TP3_OWNER_PREQUANT", "0") == "1":
            return kNvfp4Dynamic
        return None

    def bind_row_prequant_input(self, layer: torch.nn.Module) -> None:
        """Bind one loaded consumer during initialization, before compilation."""
        if (
            getattr(layer, "input_quant_key", None) not in (None, kNvfp4Dynamic)
            or not hasattr(layer, "input_global_scale_inv")
            or layer.input_global_scale_inv.dtype != torch.float32
            or layer.input_global_scale_inv.numel() != 1
            or not hasattr(layer, "weights_padding_cols")
        ):
            raise ValueError("row prequant requires a loaded compatible B12x consumer")
        layer._ag2_row_prequant_input = True
        layer.input_quant_key = kNvfp4Dynamic

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        qa = as_quantized_activation(x, self.input_quant_key(layer))
        if qa is not None:
            x_fp4, x_blockscale = qa.data, qa.scale
            output_dtype = qa.orig_dtype
            output_shape = [*qa.orig_shape[:-1], output_size]
        else:
            assert isinstance(x, torch.Tensor)
            output_dtype = x.dtype
            output_shape = [*x.shape[:-1], output_size]
            selected = getattr(layer, "_ag2_nvfp4_arc_selected", None)
            if selected is not None:
                x_fp4, x_blockscale = ag2_nvfp4_arc_quantize(
                    x, layer.input_global_scale_inv, selected
                )
            else:
                x_fp4, x_blockscale = scaled_fp4_quant(
                    x,
                    layer.input_global_scale_inv,
                    is_sf_swizzled_layout=True,
                    backend="b12x",
                )

        x_fp4 = pad_nvfp4_activation_for_cutlass(
            x_fp4, getattr(layer, "weights_padding_cols", 0)
        )

        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            backend="b12x",
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
