# SPDX-License-Identifier: Apache-2.0
"""Static topology contract for the experimental folded Qwen3.5 MTP path.

The dense MTP layer is naturally tensor-parallel over ranks 0 and 1, while
its attention is evaluated by the existing TP3/DCP3 FlashInfer path.  This
module deliberately contains no communication or allocation: it is the
fail-closed, deterministic ownership contract consumed by those later stages.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op


_FOLDED_MTP_GROUPS: dict[str, dist.ProcessGroup] = {}


def _gemma_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Match Qwen3.5's GemmaRMSNorm checkpoint semantics exactly."""
    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    return (normalized * (1.0 + weight.float())).to(value.dtype)


def _folded_mtp_group(group_key: str) -> dist.ProcessGroup:
    try:
        return _FOLDED_MTP_GROUPS[group_key]
    except KeyError as exc:
        raise RuntimeError(f"unknown folded MTP communicator {group_key}") from exc


def _folded_mtp_pair_all_gather_impl(
    local: torch.Tensor,
    gathered: torch.Tensor,
    group_key: str,
) -> None:
    group = _folded_mtp_group(group_key)
    rows, width = local.shape
    shards = gathered.view(2, rows, width)
    dist.all_gather_into_tensor(shards, local, group=group)


def _folded_mtp_pair_all_gather_fake(
    local: torch.Tensor,
    gathered: torch.Tensor,
    group_key: str,
) -> None:
    return None


def _folded_mtp_pair_all_reduce_impl(
    value: torch.Tensor,
    group_key: str,
) -> None:
    dist.all_reduce(value, group=_folded_mtp_group(group_key))


def _folded_mtp_pair_all_reduce_fake(
    value: torch.Tensor,
    group_key: str,
) -> None:
    return None


def _folded_mtp_publish_impl(
    value: torch.Tensor,
    rank: int,
    group_key: str,
) -> None:
    group = _folded_mtp_group(group_key)
    if rank == 0:
        _wait_p2p([dist.P2POp(dist.isend, value, 2, group, 50)])
    elif rank == 2:
        _wait_p2p([dist.P2POp(dist.irecv, value, 0, group, 50)])


def _folded_mtp_publish_fake(
    value: torch.Tensor,
    rank: int,
    group_key: str,
) -> None:
    return None


def _wait_p2p(operations: list[dist.P2POp]) -> None:
    for work in dist.batch_isend_irecv(operations):
        work.wait()


def _folded_mtp_restore_impl(
    local_output: torch.Tensor,
    dense_output: torch.Tensor,
    output4: torch.Tensor,
    output8: torch.Tensor,
    rank: int,
    group_key: str,
) -> None:
    group = _folded_mtp_group(group_key)
    if rank == 0:
        dense_output[:, :8].copy_(local_output)
        _wait_p2p([dist.P2POp(dist.irecv, output4, 1, group, 30)])
        dense_output[:, 8:].copy_(output4)
    elif rank == 1:
        output4.copy_(local_output[:, :4])
        dense_output[:, :4].copy_(local_output[:, 4:])
        _wait_p2p(
            [
                dist.P2POp(dist.isend, output4, 0, group, 30),
                dist.P2POp(dist.irecv, output8, 2, group, 31),
            ]
        )
        dense_output[:, 4:].copy_(output8)
    else:
        _wait_p2p([dist.P2POp(dist.isend, local_output, 1, group, 31)])


def _folded_mtp_restore_fake(
    local_output: torch.Tensor,
    dense_output: torch.Tensor,
    output4: torch.Tensor,
    output8: torch.Tensor,
    rank: int,
    group_key: str,
) -> None:
    return None


def _folded_mtp_prepare_impl(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    attention_q: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    q4: torch.Tensor,
    q8: torch.Tensor,
    peer_k: torch.Tensor,
    peer_v: torch.Tensor,
    other_k: torch.Tensor,
    other_v: torch.Tensor,
    rank: int,
    group_key: str,
) -> None:
    group = _folded_mtp_group(group_key)
    if rank == 0:
        attention_q.copy_(q_local[:, :8])
        q4.copy_(q_local[:, 8:])
        full_k[:, :2].copy_(k_local)
        full_v[:, :2].copy_(v_local)
        _wait_p2p(
            [
                dist.P2POp(dist.isend, q4, 1, group, 10),
                dist.P2POp(dist.isend, k_local, 1, group, 11),
                dist.P2POp(dist.isend, v_local, 1, group, 12),
                dist.P2POp(dist.irecv, peer_k, 1, group, 13),
                dist.P2POp(dist.irecv, peer_v, 1, group, 14),
                dist.P2POp(dist.isend, k_local, 2, group, 15),
                dist.P2POp(dist.isend, v_local, 2, group, 16),
            ]
        )
        full_k[:, 2:].copy_(peer_k)
        full_v[:, 2:].copy_(peer_v)
    elif rank == 1:
        attention_q[:, 4:].copy_(q_local[:, :4])
        q8.copy_(q_local[:, 4:])
        full_k[:, 2:].copy_(k_local)
        full_v[:, 2:].copy_(v_local)
        _wait_p2p(
            [
                dist.P2POp(dist.irecv, q4, 0, group, 10),
                dist.P2POp(dist.irecv, peer_k, 0, group, 11),
                dist.P2POp(dist.irecv, peer_v, 0, group, 12),
                dist.P2POp(dist.isend, k_local, 0, group, 13),
                dist.P2POp(dist.isend, v_local, 0, group, 14),
                dist.P2POp(dist.isend, q8, 2, group, 17),
                dist.P2POp(dist.isend, k_local, 2, group, 18),
                dist.P2POp(dist.isend, v_local, 2, group, 19),
            ]
        )
        attention_q[:, :4].copy_(q4)
        full_k[:, :2].copy_(peer_k)
        full_v[:, :2].copy_(peer_v)
    else:
        _wait_p2p(
            [
                dist.P2POp(dist.irecv, peer_k, 0, group, 15),
                dist.P2POp(dist.irecv, peer_v, 0, group, 16),
                dist.P2POp(dist.irecv, q8, 1, group, 17),
                dist.P2POp(dist.irecv, other_k, 1, group, 18),
                dist.P2POp(dist.irecv, other_v, 1, group, 19),
            ]
        )
        attention_q.copy_(q8)
        full_k[:, :2].copy_(peer_k)
        full_v[:, :2].copy_(peer_v)
        full_k[:, 2:].copy_(other_k)
        full_v[:, 2:].copy_(other_v)


def _folded_mtp_prepare_fake(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    attention_q: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    q4: torch.Tensor,
    q8: torch.Tensor,
    peer_k: torch.Tensor,
    peer_v: torch.Tensor,
    other_k: torch.Tensor,
    other_v: torch.Tensor,
    rank: int,
    group_key: str,
) -> None:
    return None


def _folded_mtp_redistribute_impl(
    q_local: torch.Tensor,
    attention_q: torch.Tensor,
    q4: torch.Tensor,
    q8: torch.Tensor,
    rank: int,
    group_key: str,
) -> None:
    group = _folded_mtp_group(group_key)
    if rank == 0:
        attention_q.copy_(q_local[:, :8])
        q4.copy_(q_local[:, 8:])
        _wait_p2p([dist.P2POp(dist.isend, q4, 1, group, 40)])
    elif rank == 1:
        attention_q[:, 4:].copy_(q_local[:, :4])
        q8.copy_(q_local[:, 4:])
        _wait_p2p(
            [
                dist.P2POp(dist.irecv, q4, 0, group, 40),
                dist.P2POp(dist.isend, q8, 2, group, 41),
            ]
        )
        attention_q[:, :4].copy_(q4)
    else:
        _wait_p2p([dist.P2POp(dist.irecv, q8, 1, group, 41)])
        attention_q.copy_(q8)


def _folded_mtp_redistribute_fake(
    q_local: torch.Tensor,
    attention_q: torch.Tensor,
    q4: torch.Tensor,
    q8: torch.Tensor,
    rank: int,
    group_key: str,
) -> None:
    return None


direct_register_custom_op(
    "ag2_folded_mtp_restore",
    _folded_mtp_restore_impl,
    mutates_args=["dense_output", "output4", "output8"],
    fake_impl=_folded_mtp_restore_fake,
)
direct_register_custom_op(
    "ag2_folded_mtp_pair_all_gather",
    _folded_mtp_pair_all_gather_impl,
    mutates_args=["gathered"],
    fake_impl=_folded_mtp_pair_all_gather_fake,
)
direct_register_custom_op(
    "ag2_folded_mtp_pair_all_reduce",
    _folded_mtp_pair_all_reduce_impl,
    mutates_args=["value"],
    fake_impl=_folded_mtp_pair_all_reduce_fake,
)
direct_register_custom_op(
    "ag2_folded_mtp_publish",
    _folded_mtp_publish_impl,
    mutates_args=["value"],
    fake_impl=_folded_mtp_publish_fake,
)
direct_register_custom_op(
    "ag2_folded_mtp_prepare",
    _folded_mtp_prepare_impl,
    mutates_args=[
        "attention_q",
        "full_k",
        "full_v",
        "q4",
        "q8",
        "peer_k",
        "peer_v",
        "other_k",
        "other_v",
    ],
    fake_impl=_folded_mtp_prepare_fake,
)
direct_register_custom_op(
    "ag2_folded_mtp_redistribute",
    _folded_mtp_redistribute_impl,
    mutates_args=["attention_q", "q4", "q8"],
    fake_impl=_folded_mtp_redistribute_fake,
)


@dataclass(frozen=True)
class FoldedMTPTransfer:
    source: int
    destination: int
    tensor: str
    head_start: int
    head_end: int

    @property
    def head_count(self) -> int:
        return self.head_end - self.head_start


@dataclass(frozen=True)
class FoldedMTPGeometry:
    """Exact Qwopus Q24/KV4 natural-TP2 -> attention-TP3 mapping."""

    total_q_heads: int = 24
    total_kv_heads: int = 4
    dense_pair_size: int = 2
    attention_world_size: int = 3

    def __post_init__(self) -> None:
        if self.total_q_heads != 24 or self.total_kv_heads != 4:
            raise ValueError("folded MTP currently supports only Q24/KV4")
        if self.dense_pair_size != 2 or self.attention_world_size != 3:
            raise ValueError("folded MTP requires natural TP2 over TP3/DCP3")

    @property
    def dense_q_heads(self) -> int:
        return self.total_q_heads // self.dense_pair_size

    @property
    def dense_kv_heads(self) -> int:
        return self.total_kv_heads // self.dense_pair_size

    @property
    def attention_q_heads(self) -> int:
        return self.total_q_heads // self.attention_world_size

    def dense_q_range(self, rank: int) -> tuple[int, int]:
        if rank not in (0, 1):
            raise ValueError(f"rank {rank} is not a dense-pair owner")
        start = rank * self.dense_q_heads
        return start, start + self.dense_q_heads

    def attention_q_range(self, rank: int) -> tuple[int, int]:
        if not 0 <= rank < self.attention_world_size:
            raise ValueError(f"rank {rank} is not an attention owner")
        start = rank * self.attention_q_heads
        return start, start + self.attention_q_heads

    def input_transfers(self) -> tuple[FoldedMTPTransfer, ...]:
        """Minimal fixed transfers before the existing DCP3 attention call.

        KV indices are local TP2 indices: rank 0 owns global KV 0:2 and rank 1
        owns global KV 2:4.  Each DCP rank needs all four KV heads for the
        positions it owns.  Q is re-sliced from 12/12 into 8/8/8.
        """
        return (
            FoldedMTPTransfer(0, 1, "q", 8, 12),
            FoldedMTPTransfer(1, 2, "q", 16, 24),
            FoldedMTPTransfer(0, 1, "kv", 0, 2),
            FoldedMTPTransfer(1, 0, "kv", 2, 4),
            FoldedMTPTransfer(0, 2, "kv", 0, 2),
            FoldedMTPTransfer(1, 2, "kv", 2, 4),
        )

    def output_transfers(self) -> tuple[FoldedMTPTransfer, ...]:
        """Fixed transfers restoring attention 8/8/8 to dense TP2 12/12."""
        return (
            FoldedMTPTransfer(1, 0, "attention_output", 8, 12),
            FoldedMTPTransfer(2, 1, "attention_output", 16, 24),
        )

    def validate(self) -> None:
        """Prove exact, non-overlapping Q ownership at both boundaries."""
        dense = [self.dense_q_range(rank) for rank in range(self.dense_pair_size)]
        attention = [
            self.attention_q_range(rank)
            for rank in range(self.attention_world_size)
        ]
        for ranges in (dense, attention):
            cursor = 0
            for start, end in ranges:
                if start != cursor or end <= start:
                    raise ValueError(f"non-contiguous folded MTP Q layout: {ranges}")
                cursor = end
            if cursor != self.total_q_heads:
                raise ValueError(f"incomplete folded MTP Q layout: {ranges}")

        input_q = [
            transfer
            for transfer in self.input_transfers()
            if transfer.tensor == "q"
        ]
        output_q = [
            transfer
            for transfer in self.output_transfers()
            if transfer.tensor == "attention_output"
        ]
        input_layout = [
            (x.source, x.destination, x.head_start, x.head_end) for x in input_q
        ]
        if input_layout != [(0, 1, 8, 12), (1, 2, 16, 24)]:
            raise ValueError("folded MTP input Q transport drift")
        output_layout = [
            (x.source, x.destination, x.head_start, x.head_end) for x in output_q
        ]
        if output_layout != [(1, 0, 8, 12), (2, 1, 16, 24)]:
            raise ValueError("folded MTP output transport drift")


FOLDED_MTP_GEOMETRY = FoldedMTPGeometry()
FOLDED_MTP_GEOMETRY.validate()


class FoldedMTPDeviceTransport:
    """Preallocated, capture-stable QKV and attention-output redistribution.

    The communicator must be a dedicated three-rank device group created on
    every rank before CUDA Graph capture.  All ranks must call ``prepare`` and
    ``restore`` in the same order on every replay.
    """

    def __init__(
        self,
        *,
        rank: int,
        device: torch.device,
        max_rows: int,
        head_dim: int,
        group: dist.ProcessGroup,
        geometry: FoldedMTPGeometry = FOLDED_MTP_GEOMETRY,
    ) -> None:
        geometry.validate()
        if not 0 <= rank < geometry.attention_world_size:
            raise ValueError(f"rank {rank} is not an attention owner")
        if max_rows <= 0 or head_dim <= 0:
            raise ValueError("max_rows and head_dim must be positive")
        if dist.get_world_size(group) != geometry.attention_world_size:
            raise ValueError("folded MTP communicator must contain three ranks")
        if dist.get_rank(group) != rank:
            raise ValueError("folded MTP communicator rank/order mismatch")

        self.rank = rank
        self.device = device
        self.max_rows = max_rows
        self.head_dim = head_dim
        self.group = group
        self.group_key = f"{rank}:{id(group)}"
        _FOLDED_MTP_GROUPS[self.group_key] = group
        self.geometry = geometry
        dtype = torch.bfloat16
        self.attention_q = torch.empty(
            (max_rows, geometry.attention_q_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.redistributed_q = torch.empty_like(self.attention_q)
        self.full_k = torch.empty(
            (max_rows, geometry.total_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.full_v = torch.empty_like(self.full_k)
        self.dense_output = torch.empty(
            (max_rows, geometry.dense_q_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.dense_k_dummy = torch.empty(
            (max_rows, geometry.dense_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.dense_v_dummy = torch.empty_like(self.dense_k_dummy)
        self.q4 = torch.empty((max_rows, 4, head_dim), dtype=dtype, device=device)
        self.q8 = torch.empty((max_rows, 8, head_dim), dtype=dtype, device=device)
        self.kv2_k = torch.empty(
            (max_rows, geometry.dense_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.kv2_v = torch.empty_like(self.kv2_k)
        self.kv2_other_k = torch.empty_like(self.kv2_k)
        self.kv2_other_v = torch.empty_like(self.kv2_k)
        self.output4 = torch.empty_like(self.q4)
        self.output8 = torch.empty_like(self.q8)

    def _rows(self, value: torch.Tensor | None) -> int:
        if value is None or value.ndim != 3:
            raise ValueError("dense-pair input must be a rank-3 tensor")
        rows = value.shape[0]
        if not 0 < rows <= self.max_rows:
            raise ValueError(f"rows {rows} exceed folded MTP buffer capacity")
        if value.device != self.device or value.dtype != torch.bfloat16:
            raise ValueError("folded MTP tensors must be BF16 on the local device")
        return rows

    def prepare(
        self,
        q_local: torch.Tensor | None,
        k_local: torch.Tensor | None,
        v_local: torch.Tensor | None,
        *,
        rows: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert pair Q12/KV2 inputs to local DCP Q8/full-KV4 inputs."""
        if self.rank < self.geometry.dense_pair_size:
            actual_rows = self._rows(q_local)
            assert q_local is not None and k_local is not None and v_local is not None
            expected_q = (actual_rows, self.geometry.dense_q_heads, self.head_dim)
            expected_kv = (actual_rows, self.geometry.dense_kv_heads, self.head_dim)
            if tuple(q_local.shape) != expected_q:
                raise ValueError(f"invalid folded MTP Q shape {tuple(q_local.shape)}")
            if (
                tuple(k_local.shape) != expected_kv
                or tuple(v_local.shape) != expected_kv
            ):
                raise ValueError("invalid folded MTP K/V shape")
            for value in (k_local, v_local):
                if value.device != self.device or value.dtype != torch.bfloat16:
                    raise ValueError(
                        "folded MTP tensors must be BF16 on the local device"
                    )
        else:
            if q_local is not None or k_local is not None or v_local is not None:
                raise ValueError("attention satellite must not receive dense QKV")
            if rows is None or not 0 < rows <= self.max_rows:
                raise ValueError("attention satellite requires a valid row count")
            actual_rows = rows

        if q_local is None:
            q_local = self.dense_output[:actual_rows]
        if k_local is None:
            k_local = self.dense_k_dummy[:actual_rows]
        if v_local is None:
            v_local = self.dense_v_dummy[:actual_rows]

        q = self.attention_q[:actual_rows]
        key = self.full_k[:actual_rows]
        value = self.full_v[:actual_rows]
        q4 = self.q4[:actual_rows]
        q8 = self.q8[:actual_rows]
        peer_k = self.kv2_k[:actual_rows]
        peer_v = self.kv2_v[:actual_rows]
        other_k = self.kv2_other_k[:actual_rows]
        other_v = self.kv2_other_v[:actual_rows]

        torch.ops.vllm.ag2_folded_mtp_prepare(
            q_local,
            k_local,
            v_local,
            q,
            key,
            value,
            q4,
            q8,
            peer_k,
            peer_v,
            other_k,
            other_v,
            self.rank,
            self.group_key,
        )
        return q, key, value

    def restore(self, local_output: torch.Tensor) -> torch.Tensor | None:
        """Restore local DCP Q8 output to pair-owned Q12 output shards."""
        rows = self._rows(local_output)
        expected = (rows, self.geometry.attention_q_heads, self.head_dim)
        if tuple(local_output.shape) != expected:
            raise ValueError(
                f"invalid folded MTP attention output {tuple(local_output.shape)}"
            )
        if self.rank == 0:
            output = self.dense_output[:rows]
        elif self.rank == 1:
            output = self.dense_output[:rows]
        else:
            output = self.dense_output[:rows]
        torch.ops.vllm.ag2_folded_mtp_restore(
            local_output,
            output,
            self.output4[:rows],
            self.output8[:rows],
            self.rank,
            self.group_key,
        )
        return output if self.rank < self.geometry.dense_pair_size else None

    def redistribute_q(
        self,
        q_local: torch.Tensor | None,
        *,
        rows: int,
    ) -> torch.Tensor:
        """Convert pair-owned Q12 back to local attention-owned Q8."""
        if not 0 < rows <= self.max_rows:
            raise ValueError("invalid folded MTP redistribution row count")
        q = self.redistributed_q[:rows]
        q4 = self.q4[:rows]
        q8 = self.q8[:rows]
        if self.rank < self.geometry.dense_pair_size:
            actual_rows = self._rows(q_local)
            if actual_rows != rows or q_local is None:
                raise ValueError("invalid folded MTP dense Q rows")
            expected = (rows, self.geometry.dense_q_heads, self.head_dim)
            if tuple(q_local.shape) != expected:
                raise ValueError("invalid folded MTP dense Q shape")
        elif q_local is not None:
            raise ValueError("attention satellite must not own dense Q")

        if q_local is None:
            q_local = self.dense_output[:rows]
        torch.ops.vllm.ag2_folded_mtp_redistribute(
            q_local,
            q,
            q4,
            q8,
            self.rank,
            self.group_key,
        )
        return q


class FoldedMTPAttentionBridge(torch.nn.Module):
    """Connect the device transport to an existing DCP3 Attention layer.

    The wrapped layer remains the sole owner of attention metadata, slot
    mapping, KV update, FlashInfer workspace and output/LSE combination.  The
    bridge only changes the Q/K/V ownership immediately around that call.
    """

    def __init__(
        self,
        attention: torch.nn.Module,
        transport: FoldedMTPDeviceTransport,
    ) -> None:
        super().__init__()
        if getattr(attention, "num_heads", None) != 8:
            raise ValueError("folded MTP DCP3 Attention must own 8 Q heads")
        if getattr(attention, "num_kv_heads", None) != 4:
            raise ValueError("folded MTP DCP3 Attention must own all 4 KV heads")
        if getattr(attention, "head_size", None) != transport.head_dim:
            raise ValueError("folded MTP attention head size mismatch")
        self.attention = attention
        self.transport = transport

    def forward(
        self,
        q_local: torch.Tensor | None,
        k_local: torch.Tensor | None,
        v_local: torch.Tensor | None,
        *,
        rows: int,
    ) -> torch.Tensor | None:
        query, key, value = self.transport.prepare(
            q_local,
            k_local,
            v_local,
            rows=rows,
        )
        local_output = self.attention(query, key, value).view(
            rows,
            self.transport.geometry.attention_q_heads,
            self.transport.head_dim,
        )
        return self.transport.restore(local_output)


class FoldedMTPAttentionCore(torch.nn.Module):
    """Default-off folded core backed by vLLM's standard DCP3 Attention."""

    def __init__(
        self,
        *,
        vllm_config: object,
        prefix: str,
        transport_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        from vllm.distributed.parallel_state import (
            get_dcp_group,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.attention.head_partition import (
            make_attention_head_partition,
        )

        model_config = getattr(vllm_config, "model_config")
        config = model_config.hf_text_config
        parallel_config = getattr(vllm_config, "parallel_config")
        cache_config = getattr(vllm_config, "cache_config")
        quant_config = getattr(vllm_config, "quant_config")
        geometry = FoldedMTPGeometry(
            total_q_heads=config.num_attention_heads,
            total_kv_heads=config.num_key_value_heads,
            dense_pair_size=2,
            attention_world_size=3,
        )
        if get_tensor_model_parallel_world_size() != 3:
            raise ValueError("folded MTP Attention requires TP3")
        dcp_group = get_dcp_group()
        if dcp_group.world_size != 3:
            raise ValueError("folded MTP Attention requires DCP3")
        head_dim = config.head_dim
        if head_dim != 256:
            raise ValueError("folded MTP Attention requires head_dim=256")
        dcp_rank = dcp_group.rank_in_group
        partition = make_attention_head_partition(
            total_num_heads=geometry.total_q_heads,
            total_num_kv_heads=geometry.total_kv_heads,
            tp_size=geometry.attention_world_size,
            tp_rank=dcp_rank,
        )
        attention = Attention(
            geometry.attention_q_heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=geometry.total_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        if not attention.attn_backend.supports_dcp_full_kv_attention_heads:
            raise NotImplementedError(
                "folded MTP requires full-KV DCP backend support"
            )
        attention.dcp_full_kv_attention_heads = True
        attention.dcp_local_kv_head_indices = partition.kv_head_indices
        max_rows = parallel_config.tensor_parallel_size
        scheduler_config = getattr(vllm_config, "scheduler_config")
        max_rows = max(max_rows, scheduler_config.max_num_batched_tokens)
        device = torch.device("cuda", torch.cuda.current_device())
        transport = FoldedMTPDeviceTransport(
            rank=dcp_rank,
            device=device,
            max_rows=max_rows,
            head_dim=head_dim,
            group=transport_group,
            geometry=geometry,
        )
        self.bridge = FoldedMTPAttentionBridge(attention, transport)

    @property
    def attention(self) -> torch.nn.Module:
        return self.bridge.attention

    def forward(
        self,
        q_local: torch.Tensor | None,
        k_local: torch.Tensor | None,
        v_local: torch.Tensor | None,
        *,
        rows: int,
    ) -> torch.Tensor | None:
        return self.bridge(q_local, k_local, v_local, rows=rows)


class FoldedMTPNaturalTP2Predictor(torch.nn.Module):
    """Checkpoint-exact BF16 MTP trunk owned by ranks 0/1 over TP3/DCP3.

    Embedding and lm-head remain the existing TP3 modules. Rank 2 owns only
    the standard DCP3 Attention/KV state and receives the normalized trunk
    output for the unchanged TP3 draft head.
    """

    _WEIGHT_NAMES = {
        "fc": "fc.weight",
        "pre_embedding": "pre_fc_norm_embedding.weight",
        "pre_hidden": "pre_fc_norm_hidden.weight",
        "input_norm": "layers.0.input_layernorm.weight",
        "post_norm": "layers.0.post_attention_layernorm.weight",
        "final_norm": "norm.weight",
        "q_norm": "layers.0.self_attn.q_norm.weight",
        "k_norm": "layers.0.self_attn.k_norm.weight",
        "q": "layers.0.self_attn.q_proj.weight",
        "k": "layers.0.self_attn.k_proj.weight",
        "v": "layers.0.self_attn.v_proj.weight",
        "o": "layers.0.self_attn.o_proj.weight",
        "gate": "layers.0.mlp.gate_proj.weight",
        "up": "layers.0.mlp.up_proj.weight",
        "down": "layers.0.mlp.down_proj.weight",
    }

    def __init__(self, *, vllm_config: object, prefix: str) -> None:
        super().__init__()
        from vllm.distributed.parallel_state import (
            get_dcp_group,
            get_tensor_model_parallel_rank,
        )
        from vllm.model_executor.layers.rotary_embedding import get_rope

        config = getattr(vllm_config, "model_config").hf_text_config
        self.rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.head_dim = config.head_dim
        if (
            self.hidden_size != 5120
            or self.intermediate_size != 17408
            or config.num_attention_heads != 24
            or config.num_key_value_heads != 4
            or self.head_dim != 256
        ):
            raise ValueError("natural TP2 MTP currently requires Qwopus geometry")
        dcp = get_dcp_group()
        if dcp.world_size != 3 or dcp.rank_in_group != self.rank:
            raise ValueError("natural TP2 MTP requires aligned TP3/DCP3 ranks")
        all_group = dcp.make_sibling_device_group(
            group_desc="mtp_natural_tp2_attention"
        )
        pair_group: dist.ProcessGroup | None = None
        for ranks in dcp.group_ranks:
            global_pair = ranks[:2]
            group = dist.new_group(global_pair, backend="nccl")
            if dcp.rank in global_pair:
                pair_group = group
        self.all_group_key = f"natural-all:{self.rank}:{id(all_group)}"
        _FOLDED_MTP_GROUPS[self.all_group_key] = all_group
        self.pair_group_key = ""
        if self.rank < 2:
            assert pair_group is not None
            self.pair_group_key = f"natural-pair:{self.rank}:{id(pair_group)}"
            _FOLDED_MTP_GROUPS[self.pair_group_key] = pair_group

        dtype = getattr(vllm_config, "model_config").dtype
        local_hidden = self.hidden_size // 2
        local_intermediate = self.intermediate_size // 2

        def parameter(shape: tuple[int, ...]) -> torch.nn.Parameter:
            actual = shape if self.rank < 2 else (1,)
            return torch.nn.Parameter(
                torch.empty(actual, dtype=dtype), requires_grad=False
            )

        self.fc = parameter((local_hidden, self.hidden_size * 2))
        self.pre_embedding = parameter((self.hidden_size,))
        self.pre_hidden = parameter((self.hidden_size,))
        self.input_norm = parameter((self.hidden_size,))
        self.post_norm = parameter((self.hidden_size,))
        self.final_norm = parameter((self.hidden_size,))
        self.q_norm = parameter((self.head_dim,))
        self.k_norm = parameter((self.head_dim,))
        self.q = parameter((12 * self.head_dim * 2, self.hidden_size))
        self.k = parameter((2 * self.head_dim, self.hidden_size))
        self.v = parameter((2 * self.head_dim, self.hidden_size))
        self.o = parameter((self.hidden_size, 12 * self.head_dim))
        self.gate = parameter((local_intermediate, self.hidden_size))
        self.up = parameter((local_intermediate, self.hidden_size))
        self.down = parameter((self.hidden_size, local_intermediate))
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.attention_core = FoldedMTPAttentionCore(
            vllm_config=vllm_config,
            prefix=f"{prefix}.layers.0.self_attn.attn",
            transport_group=all_group,
        )
        self.epsilon = config.rms_norm_eps
        self._loaded: set[str] = set()

    def _norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _gemma_rms_norm(value, weight, self.epsilon)

    def _pair_all_gather(self, local: torch.Tensor) -> torch.Tensor:
        rows, width = local.shape
        gathered = torch.empty(
            (2, rows, width), dtype=local.dtype, device=local.device
        )
        torch.ops.vllm.ag2_folded_mtp_pair_all_gather(
            local, gathered, self.pair_group_key
        )
        return gathered.permute(1, 0, 2).reshape(rows, width * 2)

    def _pair_all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        torch.ops.vllm.ag2_folded_mtp_pair_all_reduce(
            value, self.pair_group_key
        )
        return value

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        rows = hidden_states.shape[0]
        if self.rank < 2:
            fused = torch.cat(
                (
                    self._norm(inputs_embeds, self.pre_embedding),
                    self._norm(hidden_states, self.pre_hidden),
                ),
                dim=-1,
            )
            hidden = self._pair_all_gather(F.linear(fused, self.fc))
            residual = hidden
            normalized = self._norm(hidden, self.input_norm)
            q_gate = F.linear(normalized, self.q)
            q_gate = q_gate.view(rows, 12, self.head_dim * 2)
            query, gate = q_gate.chunk(2, dim=-1)
            query = query.flatten(1)
            gate = gate.flatten(1)
            key = F.linear(normalized, self.k)
            value = F.linear(normalized, self.v)
            query = self._norm(
                query.view(rows, 12, self.head_dim), self.q_norm
            ).flatten(1)
            key = self._norm(
                key.view(rows, 2, self.head_dim), self.k_norm
            ).flatten(1)
            query, key = self.rotary_emb(positions, query, key)
            attention = self.attention_core(
                query.view(rows, 12, self.head_dim),
                key.view(rows, 2, self.head_dim),
                value.view(rows, 2, self.head_dim),
                rows=rows,
            )
            assert attention is not None
            attention = attention.flatten(1) * torch.sigmoid(gate)
            residual = residual + self._pair_all_reduce(F.linear(attention, self.o))
            normalized = self._norm(residual, self.post_norm)
            mlp = F.linear(
                F.silu(F.linear(normalized, self.gate))
                * F.linear(normalized, self.up),
                self.down,
            )
            output = self._norm(
                residual + self._pair_all_reduce(mlp), self.final_norm
            )
        else:
            self.attention_core(None, None, None, rows=rows)
            output = torch.empty_like(hidden_states)
        torch.ops.vllm.ag2_folded_mtp_publish(
            output, self.rank, self.all_group_key
        )
        return output

    def load_weight(self, name: str, source: torch.Tensor) -> bool:
        short = name.removeprefix("model.")
        owner = next(
            (key for key, expected in self._WEIGHT_NAMES.items() if short == expected),
            None,
        )
        if owner is None:
            return False
        target = getattr(self, owner)
        if self.rank < 2:
            local_hidden = self.hidden_size // 2
            local_intermediate = self.intermediate_size // 2
            if owner == "fc":
                value = source[
                    self.rank * local_hidden : (self.rank + 1) * local_hidden
                ]
            elif owner == "q":
                width = 12 * self.head_dim * 2
                value = source[self.rank * width : (self.rank + 1) * width]
            elif owner in ("k", "v"):
                width = 2 * self.head_dim
                value = source[self.rank * width : (self.rank + 1) * width]
            elif owner == "o":
                width = 12 * self.head_dim
                value = source[:, self.rank * width : (self.rank + 1) * width]
            elif owner in ("gate", "up"):
                value = source[
                    self.rank * local_intermediate :
                    (self.rank + 1) * local_intermediate
                ]
            elif owner == "down":
                value = source[
                    :, self.rank * local_intermediate :
                    (self.rank + 1) * local_intermediate
                ]
            else:
                value = source
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"natural TP2 MTP weight {name} shape {tuple(value.shape)} "
                    f"!= {tuple(target.shape)}"
                )
            target.data.copy_(value)
        self._loaded.add(short)
        return True

    def validate_loaded(self) -> None:
        expected = set(self._WEIGHT_NAMES.values())
        if self._loaded != expected:
            raise RuntimeError(
                f"natural TP2 MTP weights incomplete: missing={sorted(expected - self._loaded)}"
            )
