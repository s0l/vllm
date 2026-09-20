# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repacked_nk,
    marlin_unpad_output,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    is_fp4_marlin_supported,
    prepare_fp4_layer_for_marlin,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

_gate_up_scratch: torch.Tensor | None = None


def set_nvfp4_marlin_gate_up_scratch(scratch: torch.Tensor | None) -> None:
    """Install the model-runner-owned physical gate/up destination.

    The buffer is process-local and shared by sequential gate/up projections.
    Keeping ownership in the runner makes its HBM cost visible before KV
    profiling; this module only resolves the current view used by the custom
    op.  Callers must not execute model forwards concurrently in one worker.
    """
    global _gate_up_scratch
    if scratch is not None:
        if scratch.ndim != 2 or scratch.shape[1] != 11648:
            raise ValueError("NVFP4 Marlin gate/up scratch must have shape (M, 11648)")
        if scratch.dtype != torch.bfloat16 or not scratch.is_contiguous():
            raise ValueError("NVFP4 Marlin gate/up scratch must be contiguous BF16")
    _gate_up_scratch = scratch


def get_nvfp4_marlin_gate_up_scratch() -> torch.Tensor | None:
    """Return the process-local runner-owned gate/up workspace, if installed."""
    return _gate_up_scratch


def _physical_output(x: torch.Tensor, padded_n: int) -> torch.Tensor:
    if envs.AG2_VLLM_NVFP4_MARLIN_GATE_UP_SCRATCH and padded_n == 11648:
        scratch = _gate_up_scratch
        if scratch is None:
            raise RuntimeError(
                "NVFP4 Marlin gate/up scratch is enabled but not configured"
            )
        if scratch.device != x.device or scratch.dtype != x.dtype:
            raise RuntimeError(
                "NVFP4 Marlin gate/up scratch does not match input device/dtype"
            )
        if x.shape[0] > scratch.shape[0]:
            raise RuntimeError(
                "NVFP4 Marlin gate/up scratch is too small: "
                f"rows={x.shape[0]} capacity={scratch.shape[0]}"
            )
        return scratch[: x.shape[0]]
    return x.new_empty((x.shape[0], padded_n))


class MarlinNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 weight-only GEMM via Marlin (W4A16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if is_fp4_marlin_supported():
            return True, None
        return False, "Marlin FP4 not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.warning_once(
            "Selected the Marlin NVFP4 weight-only (W4A16) kernel. This does "
            "not imply that the GPU lacks native FP4 support: W4A16 layers "
            "use BF16/FP16 activations, while the available native NVFP4 "
            "kernels require FP4 activations (W4A4). Marlin may be slower "
            "for compute-heavy workloads."
        )
        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            envs.AG2_VLLM_NVFP4_MARLIN_ISOLATE_PREFILL
            and not envs.AG2_VLLM_NVFP4_MARLIN_WHOLE_SLICE_PREFILL
            and is_forward_context_available()
        ):
            request_layout_cpu = get_forward_context().marlin_request_layout_cpu
            if request_layout_cpu is None:
                raise RuntimeError(
                    "NVFP4 Marlin prefill isolation is enabled without "
                    "scheduler request layout"
                )
            padded_n, _ = marlin_repacked_nk(layer.weight, num_bits=4)
            physical_output = _physical_output(x, padded_n)
            torch.ops.vllm.nvfp4_marlin_request_isolated(
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                bias,
                layer.output_size_per_partition,
                layer.input_size_per_partition,
                physical_output,
            )
            return marlin_unpad_output(
                physical_output, layer.output_size_per_partition, padded_n
            )
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=None,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )


def _nvfp4_marlin_request_isolated_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    bias: torch.Tensor | None,
    size_n: int,
    size_k: int,
    output: torch.Tensor,
) -> None:
    request_layout_cpu = get_forward_context().marlin_request_layout_cpu
    if request_layout_cpu is None:
        raise RuntimeError(
            "NVFP4 Marlin prefill isolation ran without scheduler request layout"
        )
    if input.ndim != 2:
        raise ValueError(
            "NVFP4 Marlin prefill isolation requires a two-dimensional input"
        )
    if request_layout_cpu.device.type != "cpu" or request_layout_cpu.ndim != 1:
        raise ValueError("Marlin request layout must be a one-dimensional CPU tensor")
    num_reqs = int(request_layout_cpu[0].item())
    num_decodes = int(request_layout_cpu[1].item())
    if not 0 <= num_decodes <= num_reqs:
        raise ValueError(f"invalid Marlin request layout: {num_decodes=} {num_reqs=}")
    boundaries = request_layout_cpu[2 : num_reqs + 3].tolist()
    if (
        len(boundaries) != num_reqs + 1
        or boundaries[0] != 0
        or any(left > right for left, right in zip(boundaries, boundaries[1:]))
        or boundaries[-1] > input.shape[0]
    ):
        raise ValueError(
            "invalid Marlin request boundaries: "
            f"rows={input.shape[0]} boundaries={boundaries}"
        )

    if num_decodes == num_reqs:
        apply_fp4_marlin_linear(
            input=input,
            weight=weight,
            weight_scale=weight_scale,
            weight_global_scale=weight_global_scale,
            workspace=workspace,
            size_n=size_n,
            size_k=size_k,
            bias=bias,
            output=output,
        )
        return None

    sections: list[torch.Tensor] = []
    decode_end = boundaries[num_decodes]
    if decode_end:
        sections.append(input[:decode_end])
    sections.extend(
        input[boundaries[index] : boundaries[index + 1]]
        for index in range(num_decodes, num_reqs)
        if boundaries[index] != boundaries[index + 1]
    )
    if boundaries[-1] < input.shape[0]:
        sections.append(input[boundaries[-1] :])
    if not sections:
        raise ValueError("Marlin request layout contains no executable rows")
    # Do not retain every section output and concatenate afterwards.  That
    # transiently needs the sum of all section outputs plus a second complete
    # output allocation.  Large mixed-prefill batches can then consume the
    # runtime headroom needed by later GDN temporaries.  Keep one complete
    # destination and at most one section output alive instead.
    padded_n, _ = marlin_repacked_nk(weight, num_bits=4)
    if output.shape != (input.shape[0], padded_n):
        raise ValueError(
            "NVFP4 Marlin prefill output has wrong physical shape: "
            f"expected={(input.shape[0], padded_n)} actual={output.shape}"
        )
    if output.dtype != input.dtype or output.device != input.device:
        raise ValueError("NVFP4 Marlin prefill output must match input dtype/device")
    if not output.is_contiguous():
        raise ValueError("NVFP4 Marlin prefill output must be contiguous")
    output_offset = 0
    for section in sections:
        next_offset = output_offset + section.shape[0]
        # marlin_gemm already supports a caller-owned ``c`` tensor.  Write
        # every request directly into its rows of the one full destination;
        # retaining a separate section output here consumes exactly the HBM
        # headroom that large GDN prefills need immediately afterwards.
        apply_fp4_marlin_linear(
            input=section,
            weight=weight,
            weight_scale=weight_scale,
            weight_global_scale=weight_global_scale,
            workspace=workspace,
            size_n=size_n,
            size_k=size_k,
            bias=bias,
            output=output[output_offset:next_offset],
        )
        output_offset = next_offset
    return None


def _nvfp4_marlin_request_isolated_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    workspace: torch.Tensor,
    bias: torch.Tensor | None,
    size_n: int,
    size_k: int,
    output: torch.Tensor,
) -> None:
    del input, weight, weight_scale, weight_global_scale, workspace, bias
    del size_n, size_k, output
    return None


direct_register_custom_op(
    op_name="nvfp4_marlin_request_isolated",
    op_func=_nvfp4_marlin_request_isolated_impl,
    mutates_args=["output"],
    fake_impl=_nvfp4_marlin_request_isolated_fake,
    tags=(torch.Tag.cudagraph_unsafe,),
)
