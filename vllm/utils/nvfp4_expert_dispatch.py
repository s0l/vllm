# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded whole-expert placement from a pinned local service profile.

Estimates are conditional; unsupported GPU geometry keeps the all-GPU path.
The selected route IDs, weights and CPU arithmetic are never changed here.
"""

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DispatchConfig:
    path: str
    sha256: str

    @classmethod
    def from_options(cls, options):
        if options is None:
            return None
        if not isinstance(options, dict) or set(options) != {"path", "sha256"}:
            raise ValueError("dispatch requires a pinned service profile")
        path, sha = options["path"], options["sha256"]
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("invalid dispatch profile identity")
        value = cls(path, sha)
        value.read()
        return value

    def read(self):
        blob = Path(self.path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != self.sha256:
            raise ValueError("dispatch profile SHA mismatch")
        value = json.loads(blob)
        model = value["model"]
        if (
            value["schema"] != 1
            or value["status"] != "LOCAL_POC"
            or model["schema"] != 1
            or model["status"] != "SERVICE_CALIBRATION_ONLY"
            or any(
                model[key] != expected
                for key, expected in dict(
                    cpu_rows=64,
                    cpu_lanes=640,
                    cpu_experts=512,
                    wave_rows=32,
                    wave_slots=2,
                    max_gpu_bucket=1024,
                ).items()
            )
            or len(value["gpu_uuids"]) != 3
            or len(model["ranks"]) != 3
        ):
            raise ValueError("unsupported dispatch service geometry")
        if len(set(value["gpu_uuids"])) != 3:
            raise ValueError("duplicate dispatch device")
        for device_uuid in value["gpu_uuids"]:
            uuid.UUID(device_uuid.removeprefix("GPU-"))
        cores = [
            core
            for row in model["ranks"]
            for core in row["core_groups"] + row["host_affinity"]
        ]
        if any(type(core) is not int or core < 0 for core in cores) or len(
            set(cores)
        ) != len(cores):
            raise ValueError("overlapping or invalid dispatch cores")
        for index, row in enumerate(model["ranks"]):
            if (
                row["rank"] != index
                or row["width"] != (256, 192, 192)[index]
                or len(row["core_groups"]) != 4
                or len(set(row["core_groups"])) != 4
                or len(row["host_affinity"]) != 1
            ):
                raise ValueError("foreign dispatch CPU allocation")
            for key, count in (("cpu", 3), ("copy", 4), ("compute", 3)):
                coefficients = np.asarray(row[key]["coefficients"])
                if (
                    coefficients.shape != (count,)
                    or not np.isfinite(coefficients).all()
                    or (coefficients < 0).any()
                ):
                    raise ValueError("invalid service coefficients")
            tails = np.asarray(row["single"])
            if (
                tails.shape != (4, 2)
                or not np.isfinite(tails).all()
                or not np.array_equal(tails[:, 0], [1, 4, 16, 64])
                or (tails[:, 1] <= 0).any()
            ):
                raise ValueError("invalid CPU service tails")
        return value

    def bind_cpu_only(self, rank, *, width, cores, device_uuid):
        """Bind physical ownership without reusing calibrated cost coefficients."""
        value = self.read()
        if not 0 <= rank < 3:
            raise ValueError("CPU-only dispatch requires TP3")
        row = value["model"]["ranks"][rank]
        callers = {r["host_affinity"][0] for r in value["model"]["ranks"]}
        if (
            width != row["width"]
            or str(device_uuid).removeprefix("GPU-")
            != value["gpu_uuids"][rank].removeprefix("GPU-")
            or not cores
            or set(cores) & callers
            or not set(cores) <= os.sched_getaffinity(0)
            or row["host_affinity"][0] not in os.sched_getaffinity(0)
        ):
            raise ValueError("CPU-only dispatch physical allocation mismatch")
        return CpuOnlyPolicy(self.sha256, row["host_affinity"][0])

    def bind(self, rank, *, width, native_sha, cores, device_uuid):
        value = self.read()
        if not 0 <= rank < 3:
            raise ValueError("dispatch requires TP3")
        model, row = value["model"], value["model"]["ranks"][rank]
        expected = dict(
            native_sha=model["native_sha256"],
            width=row["width"],
            cores=row["core_groups"],
            device_uuid=value["gpu_uuids"][rank].removeprefix("GPU-"),
        )
        actual = dict(
            native_sha=native_sha,
            width=width,
            cores=list(cores),
            device_uuid=str(device_uuid).removeprefix("GPU-"),
        )
        mismatches = {
            key: dict(expected=expected[key], actual=actual[key])
            for key in expected
            if expected[key] != actual[key]
        }
        if mismatches:
            raise ValueError(
                "dispatch profile differs from physical runtime: "
                + json.dumps(mismatches)
            )
        host = row["host_affinity"][0]
        if host not in os.sched_getaffinity(0) or host in cores:
            raise ValueError("dispatch caller lacks its independent calibrated core")
        return DispatchPolicy(ServiceModel(model, rank), self.sha256, host)


class CpuOnlyPolicy:
    def __init__(self, profile_sha, caller_core):
        self.profile_sha = profile_sha
        self.caller_core = caller_core

    def __call__(self, *args, **kwargs):
        raise RuntimeError("CPU-only binding cannot use a calibrated mixed policy")


class DispatchPolicy:
    def __init__(self, service, profile_sha, caller_core):
        self.service, self.profile_sha, self.caller_core = (
            service,
            profile_sha,
            caller_core,
        )

    def __call__(self, layer, jobs, hot, ready, topk):
        if (
            not 0 <= layer < 48
            or len(jobs) > 512
            or any(not 0 <= e < 512 for e in jobs)
        ):
            raise ValueError("foreign expert job identity")
        began = time.perf_counter()
        result = choose(self.service, jobs, hot=hot, ready=ready, topk=topk)
        result.update(
            policy_ms=(time.perf_counter() - began) * 1000,
            profile_sha256=self.profile_sha,
        )
        return result


class ServiceModel:
    def __init__(self, model, rank):
        if model["schema"] != 1 or not 0 <= rank < 3:
            raise ValueError("foreign service model")
        self.model, self.rank = (model, model["ranks"][rank])

    @staticmethod
    def evaluate(fit, features):
        return sum((c * x for c, x in zip(fit["coefficients"], features, strict=True)))

    def gpu(self, jobs, hot):
        copy_free = compute_free = 0.0
        slot_free = [0.0, 0.0]
        ordered = list(jobs)
        for index, start in enumerate(range(0, len(ordered), 32)):
            keys = ordered[start : start + 32]
            lanes = sum(len(jobs[e]) for e in keys)
            bucket = 1 << (lanes - 1).bit_length()
            if bucket > self.model["max_gpu_bucket"]:
                return None
            missing = sum(e not in hot for e in keys)
            duration = self.evaluate(
                self.rank["copy"], [1, missing, int(index == 0), bucket]
            )
            copied = max(copy_free, slot_free[index % 2]) + duration
            done = max(copied, compute_free) + self.evaluate(
                self.rank["compute"], [1, len(keys), bucket]
            )
            copy_free, compute_free, slot_free[index % 2] = (copied, done, done)
        return compute_free


def choose(service, jobs, *, hot=(), ready=(), topk=10, extra_ms=None, force_gpu=()):
    if not 1 <= topk <= 10 or any(len(lanes) == 0 for lanes in jobs.values()):
        raise ValueError("invalid jobs/topk")
    lanes = [int(lane) for group in jobs.values() for lane in group]
    if len(lanes) != len(set(lanes)) or any(lane < 0 for lane in lanes):
        raise ValueError("duplicate/invalid lane ownership")
    hot, ready, force_gpu = set(hot), set(ready), set(force_gpu)
    baseline = service.gpu(jobs, hot)
    if baseline is None:
        return dict(cpu=[], status="UNKNOWN_GPU_GEOMETRY", gpu_ms=None)
    extra = 0 if extra_ms is None else extra_ms
    if not np.isfinite(extra) or extra < 0:
        raise ValueError("invalid measured extra cost")
    best = dict(
        cpu=[], cpu_ms=0.0, gpu_ms=baseline, predicted_ms=baseline, cpu_tokens=0
    )
    eligible = sorted(
        set(jobs) & ready - hot - force_gpu, key=lambda e: (len(jobs[e]), e)
    )

    def finish():
        best.update(
            gpu_control_ms=baseline,
            break_even_extra_ms=baseline - best["predicted_ms"],
            status="OPTIMISTIC_POC" if extra_ms is None else "MEASURED_EXTRA_POC",
            source_ready=not bool(set(jobs) - hot - ready),
            readiness_cost="READY"
            if not set(jobs) - hot - ready
            else "UNKNOWN_SOURCE_MISS",
        )
        return best

    if not eligible or extra >= baseline:
        return finish()
    one = service.rank["single"]
    legal_counts = [len(jobs[e]) for e in eligible if len(jobs[e]) <= 64]
    # Measured medians need not be monotone. Bound by ALL eligible single-job
    # tails rather than assuming the shortest job has the cheapest sample.
    minimum = (
        float(np.interp(legal_counts, [p[0] for p in one], [p[1] for p in one]).min())
        if legal_counts
        else np.inf
    )
    if minimum + extra >= baseline:
        return finish()
    selected: list[int] = []
    tokens: set[int] = set()
    counts: list[int] = []
    token_counts = [0]
    total_lanes = 0
    for expert in eligible:
        count = len(jobs[expert])
        more = {int(lane) // topk for lane in jobs[expert]}
        if len(tokens | more) > service.model["cpu_rows"]:
            continue
        if len(selected) + 1 > 512 or total_lanes + count > 640 or count > 64:
            continue
        selected.append(expert)
        counts.append(count)
        total_lanes += count
        tokens.update(more)
        token_counts.append(len(tokens))
    if not selected:
        return finish()
    # A removal epoch per expert describes every accepted short-job prefix.
    order = list(jobs)
    removal = {expert: index + 1 for index, expert in enumerate(selected)}
    n = len(selected) + 1
    gpu = (
        np.array([removal.get(expert, n) for expert in order])[None, :]
        > np.arange(n)[:, None]
    )
    ordinal = np.cumsum(gpu, axis=1) - 1
    waves = (len(order) + 31) // 32
    bins = np.arange(n)[:, None] * waves + ordinal // 32
    lengths = np.array([len(jobs[expert]) for expert in order])
    cold = np.array([expert not in hot for expert in order])
    flat_bins = bins[gpu]
    shape = n, waves
    number = np.bincount(flat_bins, minlength=n * waves).reshape(shape)
    useful = np.bincount(
        flat_bins, weights=np.broadcast_to(lengths, gpu.shape)[gpu], minlength=n * waves
    ).reshape(shape)
    missing = np.bincount(
        flat_bins, weights=np.broadcast_to(cold, gpu.shape)[gpu], minlength=n * waves
    ).reshape(shape)
    buckets = 1 << np.ceil(np.log2(np.maximum(useful, 1))).astype(np.int64)
    known = ~((buckets > service.model["max_gpu_bucket"]) & (number > 0)).any(axis=1)
    copy_free, compute_free = np.zeros(n), np.zeros(n)
    slots = np.zeros((n, 2))
    c = service.rank["copy"]["coefficients"]
    g = service.rank["compute"]["coefficients"]
    for wave in range(waves):
        present = number[:, wave] > 0
        copying = (
            c[0]
            + c[1] * missing[:, wave]
            + c[2] * int(wave == 0)
            + c[3] * buckets[:, wave]
        )
        computing = g[0] + g[1] * number[:, wave] + g[2] * buckets[:, wave]
        copied = np.maximum(copy_free, slots[:, wave % 2]) + np.where(
            present, copying, 0
        )
        done = np.maximum(copied, compute_free) + np.where(present, computing, 0)
        copy_free, compute_free = copied, done
        slots[:, wave % 2] = done
    a = service.rank["cpu"]["coefficients"]
    tail = np.interp(counts, [p[0] for p in one], [p[1] for p in one])
    cpu = np.zeros(n)
    cpu[1:] = np.maximum(tail, a[0] + a[1] * np.arange(1, n) + a[2] * np.cumsum(counts))
    cpu[1] = tail[0]
    predicted = np.maximum(cpu, compute_free) + extra
    predicted[~known] = np.inf
    # Preserve the reference's exact all-GPU value and first-minimum tie rule.
    predicted[0], compute_free[0] = baseline, baseline
    index = int(np.argmin(predicted))
    if index:
        best = dict(
            cpu=sorted(selected[:index]),
            cpu_ms=float(cpu[index]),
            gpu_ms=float(compute_free[index]),
            predicted_ms=float(predicted[index]),
            cpu_tokens=token_counts[index],
        )
    return finish()
