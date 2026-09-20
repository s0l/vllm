# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.model_executor.warmup import kernel_warmup
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    synchronize_flashinfer_autotune_cache,
)


class FakeWorld:
    def __init__(self, rank: int, leader_payload: bytes | None = None):
        self.rank_in_group = rank
        self.world_size = 1
        self.leader_payload = leader_payload
        self.barriers = 0

    def broadcast_object(self, value: bytes | None, src: int) -> bytes | None:
        assert src == 0
        return value if self.rank_in_group == 0 else self.leader_payload

    def barrier(self) -> None:
        self.barriers += 1


class FakeTuner:
    def __init__(self, payload: bytes = b'{"leader": true}'):
        self.payload = payload
        self.saved = 0
        self.cleared = 0
        self.loaded: list[bytes] = []
        self.load_ok = True

    def save_configs(self, path: str) -> None:
        self.saved += 1
        Path(path).write_bytes(self.payload)

    def clear_cache(self) -> None:
        self.cleared += 1

    def load_configs(self, path: str) -> bool:
        self.loaded.append(Path(path).read_bytes())
        return self.load_ok


@pytest.mark.parametrize("rank", [0, 1])
def test_flashinfer_autotune_sync_loads_leader_choices(tmp_path: Path, rank: int):
    payload = b'{"rank_zero_tactic": 7}'
    cache_path = tmp_path / f"rank-{rank}" / "autotune.json"
    cache_path.parent.mkdir()
    tuner = FakeTuner(payload)
    world = FakeWorld(rank, leader_payload=payload)

    assert synchronize_flashinfer_autotune_cache(
        cache_path=cache_path,
        world=world,
        tuner=tuner,
        save_leader=True,
    )

    assert tuner.saved == (rank == 0)
    assert tuner.cleared == 1
    assert tuner.loaded == [payload]
    assert cache_path.read_bytes() == payload
    assert world.barriers == 1


def test_flashinfer_autotune_sync_rejects_missing_leader_result(tmp_path: Path):
    cache_path = tmp_path / "autotune.json"
    tuner = FakeTuner()

    assert not synchronize_flashinfer_autotune_cache(
        cache_path=cache_path,
        world=FakeWorld(rank=1, leader_payload=None),
        tuner=tuner,
        save_leader=True,
    )
    assert tuner.cleared == 0
    assert tuner.loaded == []


def test_flashinfer_autotune_sync_loads_existing_without_resave(tmp_path: Path):
    payload = b'{"frozen_tactic": 11}'
    cache_path = tmp_path / "autotune.json"
    cache_path.write_bytes(payload)
    tuner = FakeTuner()
    world = FakeWorld(rank=0)
    before = cache_path.stat()

    assert synchronize_flashinfer_autotune_cache(
        cache_path=cache_path,
        world=world,
        tuner=tuner,
        save_leader=False,
    )
    assert tuner.saved == 0
    assert tuner.cleared == 1
    assert tuner.loaded == [payload]
    assert cache_path.read_bytes() == payload
    after = cache_path.stat()
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns


def test_flashinfer_autotune_skips_benchmark_on_cache_hit(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "autotune.json"
    cache_path.write_bytes(b'{"frozen_tactic": 11}')
    tuner = FakeTuner()
    world = FakeWorld(rank=0)
    dummy_run = MagicMock()
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        _dummy_run=dummy_run,
    )

    monkeypatch.setattr(kernel_warmup, "_flashinfer_autotune_skip_ops", lambda _: None)
    monkeypatch.setattr(
        kernel_warmup, "resolve_flashinfer_autotune_file", lambda _: cache_path
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )
    monkeypatch.setattr("flashinfer.autotuner.AutoTuner.get", lambda: tuner)

    kernel_warmup.flashinfer_autotune(runner)
    dummy_run.assert_not_called()
    assert tuner.saved == 0
    assert tuner.loaded == [b'{"frozen_tactic": 11}']


def test_flashinfer_autotune_sync_fails_closed_on_invalid_cache(tmp_path: Path):
    cache_path = tmp_path / "autotune.json"
    tuner = FakeTuner()
    tuner.load_ok = False

    with pytest.raises(RuntimeError, match="synchronized FlashInfer"):
        synchronize_flashinfer_autotune_cache(
            cache_path=cache_path,
            world=FakeWorld(rank=1, leader_payload=tuner.payload),
            tuner=tuner,
            save_leader=True,
        )


def test_flashinfer_accepted_checkpoint_bypasses_derived_identity(
    tmp_path: Path, monkeypatch
):
    payload = b'{"accepted_tactic": 17}'
    checkpoint = tmp_path / "accepted.json"
    checkpoint.write_bytes(payload)
    monkeypatch.setenv("AG2_FLASHINFER_ACCEPTED_AUTOTUNE_FILE", str(checkpoint))
    monkeypatch.setenv(
        "AG2_FLASHINFER_ACCEPTED_AUTOTUNE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    assert resolve_flashinfer_autotune_file(MagicMock()) == checkpoint.resolve()


def test_flashinfer_accepted_checkpoint_requires_complete_authority(
    tmp_path: Path, monkeypatch
):
    checkpoint = tmp_path / "accepted.json"
    checkpoint.write_text("{}")
    monkeypatch.setenv("AG2_FLASHINFER_ACCEPTED_AUTOTUNE_FILE", str(checkpoint))
    monkeypatch.delenv("AG2_FLASHINFER_ACCEPTED_AUTOTUNE_SHA256", raising=False)

    with pytest.raises(RuntimeError, match="requires both file and SHA256"):
        resolve_flashinfer_autotune_file(MagicMock())


def test_flashinfer_accepted_checkpoint_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch
):
    checkpoint = tmp_path / "accepted.json"
    checkpoint.write_text("{}")
    monkeypatch.setenv("AG2_FLASHINFER_ACCEPTED_AUTOTUNE_FILE", str(checkpoint))
    monkeypatch.setenv("AG2_FLASHINFER_ACCEPTED_AUTOTUNE_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        resolve_flashinfer_autotune_file(MagicMock())
