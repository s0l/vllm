# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent stream-ordered CPU experts with a leased prepared RAM source."""

import ctypes as ct
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vllm.utils.nvfp4_cpu_experts import CpuExpertConfig


@dataclass(frozen=True)
class StreamExpertConfig:
    cpu: CpuExpertConfig
    history_steps: int
    max_promotions: int
    scan_order: bool = False

    @classmethod
    def from_options(cls, options, tp):
        if options is None:
            return None
        if not isinstance(options, dict) or set(options) - {"scan_order"} != {
            "library",
            "sha256",
            "cores",
            "max_m",
            "history_steps",
            "max_promotions",
        }:
            raise ValueError("stream experts require explicit resources and placement")
        history, promotions = options["history_steps"], options["max_promotions"]
        scan_order = options.get("scan_order", False)
        if type(scan_order) is not bool:
            raise ValueError("scan_order requires a bool")
        if (
            type(history) is not int
            or not 1 <= history <= 256
            or type(promotions) is not int
            or not 0 <= promotions <= 24576
        ):
            raise ValueError("invalid stream expert placement budget")
        cpu = CpuExpertConfig.from_options(
            {key: options[key] for key in ("library", "sha256", "cores", "max_m")}, tp
        )
        return cls(cpu, history, promotions, scan_order)

    def create(self, geometry, rank, *, layers, experts, topk):
        cores = self.cpu.rank_cores(rank, geometry.tp)
        if not set(cores) <= os.sched_getaffinity(0):
            raise ValueError("stream expert cores are outside worker affinity")
        if not {"avx2", "fma"} <= set(Path("/proc/cpuinfo").read_text().split()):
            raise ValueError("stream experts require AVX2 and FMA")
        width = min(geometry.local, max(0, geometry.width - rank * geometry.local))
        if width < 64 or width % 64:
            raise ValueError("compact stream rank has no supported local columns")
        cpu = StreamExecutor(
            load_library(self.cpu.library, self.cpu.sha256),
            hidden=geometry.hidden,
            width=width,
            layers=layers,
            experts=experts,
            max_m=self.cpu.max_m,
            topk=topk,
            threads=len(cores),
        )
        try:
            if self.scan_order and not hasattr(
                cpu.lib, "fn_executor_source_scan_order"
            ):
                raise ValueError("native expert library lacks scan-order support")
            cpu.pin(cores)
            cpu.begin(1)
        except BaseException:
            cpu.close()
            raise
        return cpu


def pointer(array):
    if not array.flags.c_contiguous:
        raise ValueError("noncontiguous native buffer")
    return ct.c_void_p(array.ctypes.data)


def load_library(path, expected_sha):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha:
        raise ValueError("executor native identity mismatch")
    lib = ct.CDLL(str(Path(path).resolve()))
    signatures = {
        "fn_executor_create": ([ct.c_int] * 7, ct.c_void_p),
        "fn_executor_pin": ([ct.c_void_p, ct.c_void_p, ct.c_int], ct.c_int),
        "fn_executor_bind": (
            [ct.c_void_p, ct.c_int] + [ct.c_void_p] * 4 + [ct.c_float] * 2,
            ct.c_int,
        ),
        "fn_executor_unbind": ([ct.c_void_p, ct.c_int], ct.c_int),
        "fn_executor_epoch": ([ct.c_void_p, ct.c_uint64], ct.c_int),
        "fn_executor_publish": ([ct.c_void_p, ct.c_uint64], ct.c_int),
        "fn_executor_destroy": ([ct.c_void_p], ct.c_int),
        "fn_executor_source": (
            [ct.c_void_p, ct.c_uint64, ct.POINTER(ct.c_char_p), ct.c_void_p],
            ct.c_int,
        ),
        "fn_executor_source_compact": (
            [ct.c_void_p, ct.c_uint64, ct.POINTER(ct.c_char_p), ct.c_void_p, ct.c_int],
            ct.c_int,
        ),
        "fn_executor_source_layout": ([ct.c_void_p] * 2, ct.c_int),
        "fn_executor_source_stats": ([ct.c_void_p] * 2, ct.c_int),
        "fn_executor_source_hot": ([ct.c_void_p] * 2 + [ct.c_size_t], ct.c_int),
        "fn_executor_source_export": (
            [ct.c_void_p] * 2 + [ct.c_int, ct.c_void_p, ct.c_size_t],
            ct.c_int,
        ),
        "fn_executor_source_resident": ([ct.c_void_p] * 2 + [ct.c_size_t], ct.c_int),
        "fn_executor_source_drop_hot": (
            [ct.c_void_p] * 2 + [ct.c_size_t, ct.c_void_p],
            ct.c_int,
        ),
        "fn_executor_source_load": (
            [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int],
            ct.c_int,
        ),
        "fn_executor_source_borrow": (
            [ct.c_void_p] * 2 + [ct.c_int] + [ct.c_void_p] * 2,
            ct.c_int,
        ),
        "fn_executor_source_release": ([ct.c_void_p, ct.c_uint64], ct.c_int),
        "fn_executor_source_region": ([ct.c_void_p] * 2, ct.c_int),
        "fn_task_create": (
            [ct.c_void_p] + [ct.c_int] * 3 + [ct.c_void_p] * 6,
            ct.c_void_p,
        ),
        "fn_task_run": ([ct.c_void_p], ct.c_int),
        "fn_task_enqueue": ([ct.c_void_p] * 2, ct.c_int),
        "fn_task_wait_stream": ([ct.c_void_p] * 2, ct.c_int),
        "fn_task_stats": ([ct.c_void_p] * 2, ct.c_int),
        "fn_task_destroy": ([ct.c_void_p], ct.c_int),
    }
    for name, signature in {
        "fn_executor_source_scan_order": ([ct.c_void_p, ct.c_int], ct.c_int),
        "fn_executor_source_scores": ([ct.c_void_p] * 2 + [ct.c_size_t], ct.c_int),
        "fn_executor_source_replacement_stats": ([ct.c_void_p] * 2, ct.c_int),
    }.items():
        if hasattr(lib, name):
            signatures[name] = signature
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    if hasattr(lib, "fn_executor_create_q8"):
        lib.fn_executor_create_q8.argtypes, lib.fn_executor_create_q8.restype = (
            [ct.c_int] * 7,
            ct.c_void_p,
        )
        (
            lib.fn_executor_vnni_supported.argtypes,
            lib.fn_executor_vnni_supported.restype,
        ) = [], ct.c_int
    return lib


def require(code, operation):
    if code:
        raise RuntimeError(f"{operation} rejected: {code}")


class StreamExecutor:
    def __init__(
        self,
        lib,
        *,
        hidden,
        width,
        experts=512,
        layers=48,
        max_m=64,
        topk=10,
        threads=2,
        math_backend="w4a16",
    ):
        self.lib, self.h, self.i = lib, hidden, width
        self.experts, self.layers = experts, layers
        self.max_m, self.topk = max_m, topk
        if math_backend not in ("w4a16", "w4a8"):
            raise ValueError("unknown CPU expert math")
        create = (
            lib.fn_executor_create_q8
            if math_backend == "w4a8"
            else lib.fn_executor_create
        )
        self.math_backend = math_backend
        self.ptr = create(hidden, width, experts, layers, max_m, topk, threads)
        if not self.ptr:
            raise ValueError("executor scratch admission failed")
        self.owners, self.tasks, self.graphs = {}, [], set()
        self.epoch = 0
        self.scratch_bytes = max_m * topk * (6 * hidden + 10 * width + 8)

    def pin(self, cores):
        values = np.asarray(cores, dtype=np.int32)
        if values.ndim != 1:
            raise ValueError("invalid CPU affinity list")
        require(
            self.lib.fn_executor_pin(self.ptr, pointer(values), len(values)),
            "worker affinity",
        )

    def quiesce(self):
        for stream in {s for task in self.tasks for s in task.streams}:
            stream.synchronize()

    def begin(self, epoch):
        self.quiesce()
        require(self.lib.fn_executor_epoch(self.ptr, epoch), "registry begin")
        self.epoch = epoch

    def bind(self, layer, expert, row):
        if not (0 <= layer < self.layers and 0 <= expert < self.experts):
            raise ValueError("expert identity outside registry")
        if hasattr(row, "descriptor"):
            row.descriptor(row.lease.pool)
        for name, shape in {
            "w13_weight": (2 * self.i, self.h // 2),
            "w2_weight": (self.h, self.i // 2),
            "w13_weight_scale": (2 * self.i, self.h // 16),
            "w2_weight_scale": (self.h, self.i // 16),
        }.items():
            value = row[name]
            if value.shape != shape or value.dtype != np.uint8:
                raise ValueError("prepared registry schema: " + name)
            pointer(value)
        for name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            value = row[name]
            if (
                value.shape != ()
                or value.dtype != np.float32
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError("invalid prepared scalar: " + name)
        key = layer * self.experts + expert
        require(
            self.lib.fn_executor_bind(
                self.ptr,
                key,
                pointer(row["w13_weight"]),
                pointer(row["w2_weight"]),
                pointer(row["w13_weight_scale"]),
                pointer(row["w2_weight_scale"]),
                float(np.float32(row["w13_weight_scale_2"] / row["w13_input_scale"])),
                float(np.float32(row["w2_weight_scale_2"] / row["w2_input_scale"])),
            ),
            "registry bind",
        )
        self.owners[key] = row

    def unbind(self, layer, expert):
        key = layer * self.experts + expert
        require(self.lib.fn_executor_unbind(self.ptr, key), "registry unbind")
        self.owners.pop(key, None)

    def source(self, paths, identities, budget, *, archive_width=0):
        """Paths/layout must come from PreparedExpertArchive admission."""
        if self.graphs:
            raise ValueError("retire graphs before source re-admission")
        if len(paths) != self.layers or len(identities) != self.layers or budget <= 0:
            raise ValueError("source inventory/budget")
        names = (ct.c_char_p * self.layers)(*(str(p).encode() for p in paths))
        stamps = np.asarray(identities, dtype=np.uint64)
        if stamps.shape != (self.layers, 5):
            raise ValueError("source identity schema")
        self.quiesce()
        require(
            self.lib.fn_executor_source_compact(
                self.ptr, budget, names, pointer(stamps), archive_width
            ),
            "source admission",
        )
        self.owners.clear()

    def source_scan_order(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("scan order requires a bool")
        require(
            self.lib.fn_executor_source_scan_order(self.ptr, int(enabled)),
            "source scan order",
        )

    def source_stats(self):
        values = np.zeros(12, np.uint64)
        require(
            self.lib.fn_executor_source_stats(self.ptr, pointer(values)), "source stats"
        )
        keys = (
            "hits",
            "misses",
            "evictions",
            "read_bytes",
            "read_ns",
            "check_ns",
            "validations",
            "row_bytes",
            "stride",
            "capacity",
            "rows",
            "poisoned",
        )
        result = dict(zip(keys, map(int, values), strict=True))
        layout = np.zeros(4, np.uint64)
        require(
            self.lib.fn_executor_source_layout(self.ptr, pointer(layout)),
            "source layout",
        )
        result.update(
            zip(
                ("archive_width", "archive_row_bytes", "archive_stride", "compact_ns"),
                map(int, layout),
                strict=True,
            )
        )
        return result

    def source_scores(self, scores):
        values = np.ascontiguousarray(scores, dtype=np.float64)
        if (
            values.shape != (self.layers, self.experts)
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("invalid native source priorities")
        require(
            self.lib.fn_executor_source_scores(self.ptr, pointer(values), values.size),
            "source priorities",
        )

    def source_replacement_stats(self):
        values = np.zeros(3, np.uint64)
        require(
            self.lib.fn_executor_source_replacement_stats(self.ptr, pointer(values)),
            "source replacement stats",
        )
        return dict(
            zip(
                ("replacement_ns", "indexed_rows", "priority_metadata_bytes"),
                map(int, values),
                strict=True,
            )
        )

    def source_hot(self, keys, *, drop=False):
        values = np.zeros((self.layers, self.experts), np.uint8)
        for layer, expert in keys:
            if not (0 <= layer < self.layers and 0 <= expert < self.experts):
                raise ValueError("source HOT identity")
            values[layer, expert] = 1
        self.quiesce()
        require(
            self.lib.fn_executor_source_hot(self.ptr, pointer(values), values.size),
            "source HOT hints",
        )
        released = np.zeros(1, np.uint64)
        if drop:
            require(
                self.lib.fn_executor_source_drop_hot(
                    self.ptr, pointer(values), values.size, pointer(released)
                ),
                "source GPU duplicate release",
            )
        return int(released[0])

    def export(self, keys, destination):
        values = np.asarray(keys, dtype=np.int32)
        if values.shape != (len(keys), 2) or not 1 <= len(keys) <= 512:
            raise ValueError("source export keys")
        if (
            destination.dtype != np.uint8
            or destination.ndim != 2
            or len(destination) != len(keys)
        ):
            raise ValueError("source export destination")
        self.quiesce()
        require(
            self.lib.fn_executor_source_export(
                self.ptr,
                pointer(values),
                len(keys),
                pointer(destination),
                destination.nbytes,
            ),
            "source export",
        )

    def resident_bitmap(self):
        present = np.zeros(self.layers * self.experts, np.uint8)
        require(
            self.lib.fn_executor_source_resident(
                self.ptr, pointer(present), present.size
            ),
            "source resident snapshot",
        )
        return present

    def resident_keys(self):
        return np.flatnonzero(self.resident_bitmap()).tolist()

    def load_rows(self, layer, experts):
        values = np.asarray(experts, np.int32)
        if values.ndim != 1 or not len(values) or values.tolist() != list(experts):
            raise ValueError("source load identities")
        require(
            self.lib.fn_executor_source_load(
                self.ptr, layer, pointer(values), len(values)
            ),
            "source load",
        )

    def borrow_region(self):
        values = np.zeros(3, np.uint64)
        require(
            self.lib.fn_executor_source_region(self.ptr, pointer(values)),
            "source region lease",
        )
        return int(values[0]), int(values[1]), SourceLease(self, int(values[2]))

    def borrow_rows(self, keys, fields, row_bytes):
        """No source I/O or weight copy; each returned view retains its lease."""
        values = np.asarray(keys, np.int32)
        if values.ndim != 1 or not len(values) or values.tolist() != list(keys):
            raise ValueError("source borrow identities")
        actual_bytes = self.h * self.i * 27 // 16 + 24
        if row_bytes != actual_bytes or any(
            field["offset"] < 0
            or field["nbytes"] <= 0
            or field["offset"] + field["nbytes"] > actual_bytes
            or int(np.prod(field["shape"])) * np.dtype(field["dtype"]).itemsize
            != field["nbytes"]
            for field in fields.values()
        ):
            raise ValueError("source borrow layout")
        addresses, token = np.zeros(len(values), np.uint64), np.zeros(1, np.uint64)
        require(
            self.lib.fn_executor_source_borrow(
                self.ptr,
                pointer(values),
                len(values),
                pointer(addresses),
                pointer(token),
            ),
            "source resident borrow",
        )
        lease = SourceLease(self, int(token[0]))
        rows = []
        for address in addresses:
            backing = (ct.c_uint8 * actual_bytes).from_address(int(address))
            backing.source_lease = lease
            blob = np.ctypeslib.as_array(backing)
            row = {
                name: np.ndarray(
                    field["shape"],
                    dtype=field["dtype"],
                    buffer=blob,
                    offset=field["offset"],
                )
                for name, field in fields.items()
            }
            for value in row.values():
                value.setflags(write=False)
            rows.append(row)
        return rows

    def publish(self):
        require(self.lib.fn_executor_publish(self.ptr, self.epoch), "registry publish")

    def task(self, layer, x, ids, weights, output, epoch, status):
        task = StreamTask(self, layer, x, ids, weights, output, epoch, status)
        self.tasks.append(task)
        return task

    def retain_graph(self, graph):
        self.graphs.add(graph)

    def retire_graph(self, graph):
        self.quiesce()
        graph.reset()
        self.graphs.remove(graph)

    def close(self):
        if self.graphs:
            raise ValueError("retire captured task graphs before executor close")
        self.quiesce()
        for task in self.tasks:
            task.close()
        self.tasks.clear()
        require(self.lib.fn_executor_destroy(self.ptr), "executor destroy")
        self.ptr = None
        self.owners.clear()


class SourceLease:
    """Owned by NumPy/ctypes backing, never by the source's cache registry."""

    def __init__(self, owner, token):
        self.owner, self.token = owner, token

    def close(self):
        if self.token:
            require(
                self.owner.lib.fn_executor_source_release(self.owner.ptr, self.token),
                "source lease release",
            )
            self.token = 0

    def __del__(self):
        self.close()


class StreamTask:
    def __init__(self, owner, layer, x, ids, weights, output, epoch, status):
        m = len(x)
        for array, dtype, shape in (
            (x, np.uint16, (m, owner.h)),
            (ids, np.int32, (m, owner.topk)),
            (weights, np.float32, ids.shape),
            (output, np.float32, x.shape),
            (epoch, np.uint64, (1,)),
            (status, np.int32, (1,)),
        ):
            if array.dtype != dtype or array.shape != shape:
                raise ValueError("native task buffer schema")
            pointer(array)
        self.owner, self.status = owner, status
        self.buffers = x, ids, weights, output, epoch, status
        self.streams = set()
        self.ptr = owner.lib.fn_task_create(
            owner.ptr,
            layer,
            m,
            owner.topk,
            *(pointer(a) for a in self.buffers),
        )
        if not self.ptr:
            raise ValueError("task admission failed")

    def run(self):
        return self.owner.lib.fn_task_run(self.ptr)

    def enqueue(self, stream):
        self.streams.add(stream)
        require(
            self.owner.lib.fn_task_enqueue(self.ptr, stream.cuda_stream), "task enqueue"
        )

    def wait_on(self, stream):
        self.streams.add(stream)
        require(
            self.owner.lib.fn_task_wait_stream(self.ptr, stream.cuda_stream),
            "task wait",
        )

    def check(self):
        require(int(self.status[0]), "task completion")

    def stats(self):
        values = np.zeros(3, np.uint64)
        require(self.owner.lib.fn_task_stats(self.ptr, pointer(values)), "task stats")
        return dict(
            queue_ns=int(values[1] - values[0]), compute_ns=int(values[2] - values[1])
        )

    def close(self):
        if self.owner.graphs:
            raise ValueError("task retained by graph")
        if self.ptr:
            for stream in self.streams:
                stream.synchronize()
            require(self.owner.lib.fn_task_destroy(self.ptr), "task destroy")
            self.ptr = None
