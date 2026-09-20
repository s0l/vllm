# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-capturable device-selected gather from the pinned E8 archive."""

from __future__ import annotations

import ctypes
from typing import Any, cast

import torch


class E8DemandLoader:
    def __init__(self, path: str, host_fields: dict[str, torch.Tensor], owner: int):
        self.library = ctypes.CDLL(path)
        self.owner = owner
        self._pointer = self.library.flash_e8_mapped_pointer
        self._pointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self._assign = self.library.flash_e8_assign_slots
        self._assign.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._pack = self.library.flash_e8_pack_records
        self._pack.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._assign_temporal = cast(
            Any, getattr(self.library, "flash_e8_assign_temporal_slots", None)
        )
        self._select_promotions = cast(
            Any, getattr(self.library, "flash_e8_select_temporal_promotions", None)
        )
        self._promote = cast(
            Any, getattr(self.library, "flash_e8_promote_records", None)
        )
        self._fill_resident = cast(
            Any, getattr(self.library, "flash_e8_fill_resident_records", None)
        )
        self._pack_compact = cast(
            Any, getattr(self.library, "flash_e8_pack_compact_records", None)
        )
        if self._assign_temporal is not None:
            self._assign_temporal.argtypes = [ctypes.c_void_p] * 5 + [
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self._select_promotions.argtypes = [ctypes.c_void_p] * 8 + [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self._promote.argtypes = [ctypes.c_void_p] * 4 + [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self._fill_resident.argtypes = [ctypes.c_void_p] * 5 + [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self._pack_compact.argtypes = [ctypes.c_void_p] * 4 + [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
        self.mapped: dict[str, tuple[int, int]] = {}
        for name in ("q_gu", "q_down"):
            value = host_fields[name]
            if (
                not value.is_pinned()
                or not value.is_contiguous()
                or value.shape[1] != owner
            ):
                raise ValueError("E8 demand source must be pinned layer-major storage")
            pointer = ctypes.c_void_p()
            if self._pointer(value.data_ptr(), ctypes.byref(pointer)):
                raise RuntimeError("E8 demand source has no mapped device pointer")
            record_bytes = value[0, 0].numel() * value.element_size()
            if pointer.value is None:
                raise RuntimeError("E8 demand source returned a null mapped pointer")
            self.mapped[name] = (pointer.value, record_bytes)

    @staticmethod
    def _stream() -> ctypes.c_void_p:
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def gather(
        self,
        *,
        layer: int,
        resident: int,
        counts: torch.Tensor,
        slots: torch.Tensor,
        slot_count: torch.Tensor,
        destination: dict[str, torch.Tensor],
        slot: int,
    ) -> None:
        stream = self._stream()
        if self._assign(
            counts.data_ptr(),
            slots.data_ptr(),
            slot_count.data_ptr(),
            self.owner,
            resident,
            stream,
        ):
            raise RuntimeError("E8 demand slot assignment launch failed")
        for name, (base, record_bytes) in self.mapped.items():
            layer_pointer = base + layer * self.owner * record_bytes
            if self._pack(
                layer_pointer,
                destination[name][slot].data_ptr(),
                slots.data_ptr(),
                self.owner,
                resident,
                record_bytes,
                stream,
            ):
                raise RuntimeError(f"E8 demand gather launch failed for {name}")

    def temporal_gather(
        self,
        *,
        layer: int,
        counts: torch.Tensor,
        resident_slots: torch.Tensor,
        cold_slots: torch.Tensor,
        cold_experts: torch.Tensor,
        cold_count: torch.Tensor,
        slot_experts: torch.Tensor,
        ages: torch.Tensor,
        clock: torch.Tensor,
        promotion_slots: torch.Tensor,
        resident_bank: dict[str, torch.Tensor],
    ) -> None:
        if self._assign_temporal is None:
            raise RuntimeError("E8 temporal demand ABI is unavailable")
        stream = self._stream()
        if self._assign_temporal(
            counts.data_ptr(),
            resident_slots.data_ptr(),
            cold_slots.data_ptr(),
            cold_experts.data_ptr(),
            cold_count.data_ptr(),
            self.owner,
            stream,
        ):
            raise RuntimeError("E8 temporal slot assignment launch failed")
        if self._select_promotions(
            counts.data_ptr(),
            resident_slots.data_ptr(),
            slot_experts.data_ptr(),
            ages.data_ptr(),
            clock.data_ptr(),
            cold_experts.data_ptr(),
            cold_count.data_ptr(),
            promotion_slots.data_ptr(),
            self.owner,
            slot_experts.numel(),
            stream,
        ):
            raise RuntimeError("E8 temporal promotion selection failed")
        capacity = cold_experts.numel()
        for name, (base, record_bytes) in self.mapped.items():
            layer_pointer = base + layer * self.owner * record_bytes
            if self._fill_resident(
                layer_pointer,
                resident_bank[name].data_ptr(),
                cold_experts.data_ptr(),
                promotion_slots.data_ptr(),
                cold_count.data_ptr(),
                layer,
                slot_experts.numel(),
                capacity,
                record_bytes,
                stream,
            ):
                raise RuntimeError(f"E8 temporal resident fill failed for {name}")

    def temporal_promote(
        self,
        *,
        layer: int,
        counts: torch.Tensor,
        resident_slots: torch.Tensor,
        slot_experts: torch.Tensor,
        ages: torch.Tensor,
        clock: torch.Tensor,
        cold_experts: torch.Tensor,
        cold_count: torch.Tensor,
        promotion_slots: torch.Tensor,
        staging: dict[str, torch.Tensor],
        resident_bank: dict[str, torch.Tensor],
        slot: int,
    ) -> None:
        stream = self._stream()
        if self._select_promotions(
            counts.data_ptr(),
            resident_slots.data_ptr(),
            slot_experts.data_ptr(),
            ages.data_ptr(),
            clock.data_ptr(),
            cold_experts.data_ptr(),
            cold_count.data_ptr(),
            promotion_slots.data_ptr(),
            self.owner,
            slot_experts.numel(),
            stream,
        ):
            raise RuntimeError("E8 temporal promotion selection failed")
        capacity = cold_experts.numel()
        resident = slot_experts.numel()
        for name, (_, record_bytes) in self.mapped.items():
            if self._promote(
                staging[name][slot].data_ptr(),
                resident_bank[name].data_ptr(),
                promotion_slots.data_ptr(),
                cold_count.data_ptr(),
                layer,
                resident,
                capacity,
                record_bytes,
                stream,
            ):
                raise RuntimeError(f"E8 temporal promotion failed for {name}")
