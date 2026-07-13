"""NVFP4 storage/operator seam for phase-switched Qwopus row linears.

This module is intentionally opt-in and is not wired into ``RowParallelLinear``
yet.  It owns the hard invariant missing from ordinary tensor parallelism: one
nonduplicated packed allocation must expose a TP2 prefill partition and a TP3
decode partition without repacking at the phase boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os

import torch
import torch.distributed as dist

from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.utils.torch_utils import direct_register_custom_op


@dataclass(frozen=True)
class DualViewSegment:
    global_start: int
    width: int
    decode: bool


def qwopus_mlp_row_segments(rank: int) -> tuple[DualViewSegment, ...]:
    """Exact aligned I=17408 ownership for physical ranks 0,1,2."""
    plans = {
        0: (
            DualViewSegment(0, 5824, True),
            DualViewSegment(5824, 2880, False),
        ),
        1: (
            DualViewSegment(8704, 2944, False),
            DualViewSegment(11648, 5760, True),
        ),
        2: (DualViewSegment(5824, 5824, True),),
    }
    try:
        return plans[rank]
    except KeyError as exc:
        raise ValueError("dual-view Qwopus layout requires physical rank 0..2") from exc


def validate_qwopus_mlp_row_plan() -> None:
    pair = [segment for rank in (0, 1) for segment in qwopus_mlp_row_segments(rank)]
    assert [(item.global_start, item.width) for item in pair] == [
        (0, 5824),
        (5824, 2880),
        (8704, 2944),
        (11648, 5760),
    ]
    decode = [
        next(item for item in qwopus_mlp_row_segments(rank) if item.decode)
        for rank in (0, 2, 1)
    ]
    assert [(item.global_start, item.width) for item in decode] == [
        (0, 5824),
        (5824, 5824),
        (11648, 5760),
    ]
    assert sum(item.width for item in pair) == 17408
    assert sum(item.width for item in decode) == 17408


def _make_kernel_layer(
    packed: torch.Tensor,
    scales: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.nn.Module, FlashInferCutlassNvFp4LinearKernel]:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.input_global_scale = torch.nn.Parameter(
        input_scale.float(), requires_grad=False
    )
    layer.weight_global_scale = torch.nn.Parameter(
        weight_scale.float(), requires_grad=False
    )
    layer.alpha = torch.nn.Parameter(
        layer.input_global_scale * layer.weight_global_scale,
        requires_grad=False,
    )
    layer.input_global_scale_inv = torch.nn.Parameter(
        1.0 / layer.input_global_scale, requires_grad=False
    )
    layer.input_size_per_partition = packed.shape[1] * 2
    layer.output_size_per_partition = packed.shape[0]
    layer.bias = None
    kernel = FlashInferCutlassNvFp4LinearKernel(NvFp4LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    return layer, kernel


class DualViewNvFp4RowLinear(torch.nn.Module):
    """Nonduplicated segmented row-linear with TP2 and TP3 execution views."""

    def __init__(
        self,
        *,
        rank: int,
        packed: torch.Tensor,
        scales: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        validate_qwopus_mlp_row_plan()
        if packed.shape[1] * 2 != 17408:
            raise ValueError("dual-view row POC currently supports Qwopus I=17408")
        if scales.shape[1] * 16 != 17408:
            raise ValueError("NVFP4 block scales do not match packed K dimension")
        self.rank = rank
        self.segments = qwopus_mlp_row_segments(rank)
        self.segment_layers = torch.nn.ModuleList()
        self.segment_kernels: list[FlashInferCutlassNvFp4LinearKernel] = []
        for segment in self.segments:
            local_packed = packed[
                :, segment.global_start // 2:
                (segment.global_start + segment.width) // 2
            ].contiguous()
            local_scales = scales[
                :, segment.global_start // 16:
                (segment.global_start + segment.width) // 16
            ].contiguous()
            layer, kernel = _make_kernel_layer(
                local_packed, local_scales, input_scale, weight_scale
            )
            self.segment_layers.append(layer)
            self.segment_kernels.append(kernel)

    @property
    def prefill_input_width(self) -> int:
        return sum(segment.width for segment in self.segments) if self.rank < 2 else 0

    @property
    def decode_input_width(self) -> int:
        return next(segment.width for segment in self.segments if segment.decode)

    def _apply_segment(self, index: int, value: torch.Tensor) -> torch.Tensor:
        return self.segment_kernels[index].apply_weights(
            self.segment_layers[index], value
        )

    def apply_prefill(
        self, input_parallel: torch.Tensor, pair_group: dist.ProcessGroup
    ) -> torch.Tensor | None:
        if self.rank == 2:
            return None
        if input_parallel.shape[-1] != self.prefill_input_width:
            raise ValueError("prefill input does not match persistent TP2 half")
        offset = 0
        output = None
        for index, segment in enumerate(self.segments):
            value = input_parallel[..., offset:offset + segment.width].contiguous()
            partial = self._apply_segment(index, value)
            output = partial if output is None else output.add_(partial)
            offset += segment.width
        assert output is not None
        dist.all_reduce(output, group=pair_group)
        return output

    def apply_decode(self, input_parallel: torch.Tensor) -> torch.Tensor:
        index = next(
            index for index, segment in enumerate(self.segments) if segment.decode
        )
        if input_parallel.shape[-1] != self.decode_input_width:
            raise ValueError("decode input does not match persistent TP3 view")
        output = self._apply_segment(index, input_parallel.contiguous())
        dist.all_reduce(output)
        return output


class DualViewNvFp4SwiGLUColumn(torch.nn.Module):
    """Column side of the same persistent layout for gate/up projections."""

    def __init__(
        self,
        *,
        rank: int,
        gate_packed: torch.Tensor,
        gate_scales: torch.Tensor,
        gate_input_scale: torch.Tensor,
        gate_weight_scale: torch.Tensor,
        up_packed: torch.Tensor,
        up_scales: torch.Tensor,
        up_input_scale: torch.Tensor,
        up_weight_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        validate_qwopus_mlp_row_plan()
        expected = (17408, 2560)
        if gate_packed.shape != expected or up_packed.shape != expected:
            raise ValueError("dual-view column POC requires Qwopus H=5120,I=17408")
        self.rank = rank
        self.segments = qwopus_mlp_row_segments(rank)
        self.gate_layers = torch.nn.ModuleList()
        self.up_layers = torch.nn.ModuleList()
        self.gate_kernels: list[FlashInferCutlassNvFp4LinearKernel] = []
        self.up_kernels: list[FlashInferCutlassNvFp4LinearKernel] = []
        for segment in self.segments:
            rows = slice(segment.global_start, segment.global_start + segment.width)
            gate_layer, gate_kernel = _make_kernel_layer(
                gate_packed[rows].contiguous(),
                gate_scales[rows].contiguous(),
                gate_input_scale,
                gate_weight_scale,
            )
            up_layer, up_kernel = _make_kernel_layer(
                up_packed[rows].contiguous(),
                up_scales[rows].contiguous(),
                up_input_scale,
                up_weight_scale,
            )
            self.gate_layers.append(gate_layer)
            self.up_layers.append(up_layer)
            self.gate_kernels.append(gate_kernel)
            self.up_kernels.append(up_kernel)

    def _activate_segment(self, index: int, value: torch.Tensor) -> torch.Tensor:
        gate = self.gate_kernels[index].apply_weights(
            self.gate_layers[index], value
        )
        up = self.up_kernels[index].apply_weights(self.up_layers[index], value)
        return torch.nn.functional.silu(gate) * up

    def apply_prefill(self, value: torch.Tensor) -> torch.Tensor | None:
        if self.rank == 2:
            return None
        return torch.cat(
            [self._activate_segment(index, value) for index in range(len(self.segments))],
            dim=-1,
        )

    def apply_decode(self, value: torch.Tensor) -> torch.Tensor:
        index = next(
            index for index, segment in enumerate(self.segments) if segment.decode
        )
        return self._activate_segment(index, value)


class DualViewNvFp4MLP(torch.nn.Module):
    """Complete exact SwiGLU MLP with TP2-prefill and TP3-decode views."""

    def __init__(
        self,
        *,
        column: DualViewNvFp4SwiGLUColumn,
        row: DualViewNvFp4RowLinear,
    ) -> None:
        super().__init__()
        if column.rank != row.rank:
            raise ValueError("dual-view column/row physical ranks differ")
        self.rank = row.rank
        self.column = column
        self.row = row

    def apply_prefill(
        self, hidden_states: torch.Tensor, pair_group: dist.ProcessGroup
    ) -> torch.Tensor | None:
        intermediate = self.column.apply_prefill(hidden_states)
        if intermediate is None:
            return None
        return self.row.apply_prefill(intermediate, pair_group)

    def apply_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.row.apply_decode(self.column.apply_decode(hidden_states))


_PAIR_GROUP: dist.ProcessGroup | None = None


def _gather_rank_tensors(value: torch.Tensor) -> list[torch.Tensor]:
    values = [torch.empty_like(value) for _ in range(3)]
    dist.all_gather(values, value)
    return values


def _full_column_from_fused(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ranks = _gather_rank_tensors(value)
    local = ranks[0].shape[0] // 2
    gate = torch.cat([item[:local] for item in ranks], dim=0)[:17408].contiguous()
    up = torch.cat([item[local:] for item in ranks], dim=0)[:17408].contiguous()
    return gate, up


def _full_row(value: torch.Tensor) -> torch.Tensor:
    logical_width = 8704 if value.shape[1] == 2912 else 1088
    return torch.cat(_gather_rank_tensors(value), dim=1)[:, :logical_width].contiguous()


def materialize_qwopus_dual_view_mlps(model: torch.nn.Module) -> int:
    """Replace processed TP3 Qwopus MLPs by persistent dual views.

    This is an opt-in server POC.  It reconstructs one logical layer at a time
    from already processed equal TP3 shards, immediately retains only this
    physical rank's dual-view segments, and releases the original TP3 module.
    """
    global _PAIR_GROUP
    if os.getenv("VLLM_EXPERIMENTAL_DUAL_VIEW_MLP", "0") != "1":
        return 0
    if dist.get_world_size() != 3:
        raise ValueError("Qwopus dual-view MLP requires world size 3")
    if _PAIR_GROUP is None:
        _PAIR_GROUP = dist.new_group([0, 1], backend="nccl")
    rank = dist.get_rank()
    converted = 0
    for module in model.modules():
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        if gate_up is None or down is None:
            continue
        if tuple(getattr(gate_up, "weight", torch.empty(0)).shape) != (11648, 2560):
            continue
        if tuple(getattr(down, "weight", torch.empty(0)).shape) != (5120, 2912):
            continue
        gate_weight, up_weight = _full_column_from_fused(gate_up.weight.data)
        gate_scales, up_scales = _full_column_from_fused(gate_up.weight_scale.data)
        down_weight = _full_row(down.weight.data)
        down_scales = _full_row(down.weight_scale.data)
        gate_input_scale = gate_up.input_scale.max()
        gate_weight_scale = gate_up.weight_scale_2.max()
        down_input_scale = down.input_scale.max()
        down_weight_scale = down.weight_scale_2.max()
        column = DualViewNvFp4SwiGLUColumn(
            rank=rank,
            gate_packed=gate_weight,
            gate_scales=gate_scales,
            gate_input_scale=gate_input_scale,
            gate_weight_scale=gate_weight_scale,
            up_packed=up_weight,
            up_scales=up_scales,
            up_input_scale=gate_input_scale,
            up_weight_scale=gate_weight_scale,
        )
        row = DualViewNvFp4RowLinear(
            rank=rank,
            packed=down_weight,
            scales=down_scales,
            input_scale=down_input_scale,
            weight_scale=down_weight_scale,
        )
        operator = DualViewNvFp4MLP(column=column, row=row)
        module.dual_view_operator = operator
        module.dual_view_pair_group = _PAIR_GROUP
        del module.gate_up_proj
        del module.down_proj
        converted += 1
        del gate_weight, up_weight, gate_scales, up_scales, down_weight, down_scales
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return converted


def dual_view_broadcast(value: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import get_tp_group

    output = value.clone()
    return get_tp_group().broadcast(output, src=0)


def dual_view_broadcast_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


direct_register_custom_op(
    op_name="dual_view_broadcast",
    op_func=dual_view_broadcast,
    fake_impl=dual_view_broadcast_fake,
)


def apply_materialized_dual_view_mlp(
    module: torch.nn.Module, value: torch.Tensor
) -> torch.Tensor:
    operator: DualViewNvFp4MLP = module.dual_view_operator
    threshold = int(os.getenv("VLLM_EXPERIMENTAL_DUAL_VIEW_MLP_THRESHOLD", "192"))
    if value.shape[0] < threshold:
        return operator.apply_decode(value)
    output = operator.apply_prefill(value, module.dual_view_pair_group)
    if operator.rank == 2:
        output = torch.empty_like(value)
    assert output is not None
    # Rank 0/1 already hold identical pair-reduced results. Broadcast supplies
    # GPU2 for the following residual/attention layer; this cost is measured.
    return torch.ops.vllm.dual_view_broadcast(output)
