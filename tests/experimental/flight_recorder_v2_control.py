"""Deterministic exact-image control for the AG2 K3 flight recorder."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch


def load_recorder(source_root: Path):
    path = source_root / (
        "vllm/v1/worker/gpu/spec_decode/autoregressive/ag2_flight_recorder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ag2_flight_recorder_under_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/workspace/vllm-src"))
    args = parser.parse_args()
    module = load_recorder(args.source_root)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "flight")
        recorder = module.Ag2FlightRecorder(prefix, steps=4, num_drafts=3)
        external_req_ids = ["chatcmpl-first", "chatcmpl-second"]
        req_ids = [f"{external_req_ids[0]}-deadbeef", external_req_ids[1]]
        idx = torch.tensor([1, 3], dtype=torch.int64)
        last = torch.tensor([0, 101, 0, 202], dtype=torch.int64)

        recorder.record(
            num_reqs=2,
            req_ids=req_ids,
            draft_tokens=torch.tensor([[11, 12, 13], [21, 22, 23]]),
            num_sampled=torch.tensor([1, 3]),
            num_rejected=torch.tensor([3, 1]),
            last_sampled=last,
            idx_mapping=idx,
        )
        recorder.record(
            num_reqs=2,
            req_ids=req_ids,
            draft_tokens=torch.tensor([[14, 15, 16], [24, 25, 26]]),
            num_sampled=torch.tensor([2, 4]),
            num_rejected=torch.tensor([2, 0]),
            last_sampled=last,
            idx_mapping=idx,
        )

        mm = np.memmap(f"{prefix}.rank0.fr.npy", dtype=np.int64, mode="r")
        assert list(map(int, mm[:5])) == [module.MAGIC, 4, 1, 64, 3]
        record_len = 4 + 64 * 8
        record = mm[5 : 5 + record_len]
        assert list(map(int, record[:4])) == [0, int(record[1]), 2, 3]
        slots = record[4:].reshape(64, 8)
        assert list(map(int, slots[0])) == [
            1,
            module.stable_request_hash(external_req_ids[0]),
            1,
            3,
            101,
            11,
            12,
            13,
        ]
        assert list(map(int, slots[1])) == [
            3,
            module.stable_request_hash(external_req_ids[1]),
            3,
            1,
            202,
            21,
            22,
            23,
        ]
        assert module.external_request_id(req_ids[0]) == external_req_ids[0]
        assert module.stable_request_hash(req_ids[0]) == module.stable_request_hash(
            external_req_ids[0]
        )
        assert module.stable_request_hash(req_ids[0]) != module.stable_request_hash(
            req_ids[1]
        )

    print("PASS: v2 request identity and all K3 draft positions are exact")


if __name__ == "__main__":
    main()
