# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental row publication on an explicitly owned TP process group."""

import torch
import torch.distributed as dist

from vllm.model_executor.kernels.linear.nvfp4.arc import ag2_nvfp4_arc_quantize
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    convert_swizzled_to_linear,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

from . import tp3_exact_reduce as exact
from .tp3_row_plan import RowOwnerPlan


class RowSeam:
    def __init__(self, plan: RowOwnerPlan, rank: int, group):
        plan._rank(rank)
        if dist.get_world_size(group) != 3 or dist.get_rank(group) != rank:
            raise ValueError("row seam requires matching rank in its TP3 group")
        self.plan, self.rank, self.group = plan, rank, group
        self.begin, self.count = plan.offsets[rank], plan.rows[rank]

    def owned(self, tensor):
        return tensor[self.begin : self.begin + self.count]

    def validate(self, contribution, residual):
        shape = (self.plan.total_rows, self.plan.hidden)
        if tuple(contribution.shape) != shape or tuple(residual.shape) != shape:
            raise ValueError("physical invocation shape mismatch before collective")
        if contribution.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
            raise ValueError("fixed exact protocol requires BF16")
        if contribution.device != residual.device or not contribution.is_contiguous():
            raise ValueError("incompatible contribution/residual storage")

    def reduce_owned(self, contribution):
        send, recv = self.plan.gather_splits(self.rank)
        received = torch.empty(
            sum(recv), dtype=contribution.dtype, device=contribution.device
        )
        dist.all_to_all_single(
            received, contribution.flatten(), list(recv), list(send), group=self.group
        )
        return exact.fixed_tp3_sum_fused(received.view(3, self.count, self.plan.hidden))

    def quantized(self, normalized, divisor, selected, *, candidate):
        if len(selected) != 3 or any(
            value.ndim != 1
            or value.dtype != torch.int32
            or value.numel() + self.plan.hidden != width
            for value, width in zip(selected, self.plan.augmented_k, strict=True)
        ):
            raise ValueError("destination ARC layout mismatch before collective")
        if not candidate:
            return ag2_nvfp4_arc_quantize(normalized, divisor, selected[self.rank])
        payloads = []
        destinations = (
            normalized if isinstance(normalized, tuple) else (normalized,) * 3
        )
        if len(destinations) != 3:
            raise ValueError("missing destination normalized rows")
        for peer in range(3):
            value = normalized(peer) if callable(normalized) else destinations[peer]
            owner = self.owned(value).contiguous()
            q, sf = ag2_nvfp4_arc_quantize(owner, divisor, selected[peer])
            linear = convert_swizzled_to_linear(
                sf, self.count, self.plan.augmented_k[peer], 16
            )
            payloads.append(
                torch.cat(
                    (q.flatten(), linear.contiguous().view(torch.uint8).flatten())
                )
            )
            del value, owner
        send, recv = self.plan.wire_splits(self.rank, alignment=16)
        outgoing = torch.cat(
            [
                torch.nn.functional.pad(p, (0, n - p.numel()))
                for p, n in zip(payloads, send, strict=True)
            ]
        )
        incoming = torch.empty(sum(recv), dtype=torch.uint8, device=outgoing.device)
        dist.all_to_all_single(
            incoming, outgoing, list(recv), list(send), group=self.group
        )
        qparts, sfparts = [], []
        width = self.plan.augmented_k[self.rank]
        logical = self.plan.publish_splits(self.rank, packed=True)[1]
        for n, packet, useful in zip(
            self.plan.rows, incoming.split(recv), logical, strict=True
        ):
            packet = packet[:useful]
            split = n * width // 2
            qparts.append(packet[:split].view(n, width // 2))
            sfparts.append(packet[split:].view(n, width // 16))
        return torch.cat(qparts), swizzle_blockscale(
            torch.cat(sfparts).view(torch.float8_e4m3fn)
        )

    def base_quantized(self, q, sf):
        """Extract canonical GDN base, without another network publication.

        A separate fresh base quantization is the oracle. Equal global scales
        are a precondition, not something this extraction establishes.
        """
        linear = convert_swizzled_to_linear(
            sf, self.plan.total_rows, self.plan.augmented_k[self.rank], 16
        )
        return q[:, : self.plan.hidden // 2].contiguous(), swizzle_blockscale(
            linear[:, : self.plan.hidden // 16].contiguous()
        )

    def publish_destinations(self, values):
        """Full rows per destination, preserving its own compiled norm result."""
        if len(values) != 3 or any(
            v.dtype != torch.bfloat16
            or tuple(v.shape) != (self.count, self.plan.hidden)
            for v in values
        ):
            raise ValueError("three destination-owned BF16 payloads required")
        send_bytes, recv_bytes = self.plan.publish_splits(self.rank, packed=False)
        send, recv = [n // 2 for n in send_bytes], [n // 2 for n in recv_bytes]
        outgoing = torch.cat([v.contiguous().flatten() for v in values])
        incoming = torch.empty(
            sum(recv), dtype=values[0].dtype, device=values[0].device
        )
        dist.all_to_all_single(incoming, outgoing, recv, send, group=self.group)
        return incoming.view(self.plan.total_rows, self.plan.hidden)
