# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    synchronize_flashinfer_autotune_cache,
)


class FakeWorld:
    def __init__(self, rank: int, leader_payload: bytes | None = None):
        self.rank_in_group = rank
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
