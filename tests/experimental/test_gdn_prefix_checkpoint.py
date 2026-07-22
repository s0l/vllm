import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.mamba_utils import GDNPrefixCheckpointStore


class _TensorStore(GDNPrefixCheckpointStore):
    def __init__(
        self,
        tensors: dict[str, list[torch.Tensor]],
        limit: int = 2,
        advertised_limit: int | None = None,
    ):
        super().__init__(limit=limit, advertised_limit=advertised_limit)
        self.tensors = tensors

    def _state_tensors(self, req_id, *args):
        yield from self.tensors[req_id]


class TestGDNPrefixCheckpoint(unittest.TestCase):
    def test_exact_restore_and_wrong_key_miss(self):
        tensors = {
            "req": [
                torch.tensor([1.0, 2.0]),
                torch.tensor([3], dtype=torch.int32),
            ]
        }
        store = _TensorStore(tensors)
        store.save(b"exact", "req", None, None, None)

        tensors["req"][0].zero_()
        tensors["req"][1].zero_()
        store.restore(b"exact", "req", None, None, None)
        torch.testing.assert_close(tensors["req"][0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(
            tensors["req"][1], torch.tensor([3], dtype=torch.int32)
        )

        with self.assertRaisesRegex(RuntimeError, "missing in worker"):
            store.restore(b"wrong", "req", None, None, None)

    def test_worker_and_scheduler_lru_have_identical_touch_order(self):
        tensors = {"req": [torch.tensor([1.0])]}
        store = _TensorStore(tensors, limit=2)
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.gdn_checkpoint_limit = 2

        for key in (b"a", b"b"):
            store.save(key, "req", None, None, None)
            coordinator.register_gdn_checkpoint(key)

        store.restore(b"a", "req", None, None, None)
        self.assertTrue(coordinator.has_gdn_checkpoint(b"a"))

        store.save(b"c", "req", None, None, None)
        coordinator.register_gdn_checkpoint(b"c")

        self.assertTrue(store.contains(b"a") and store.contains(b"c"))
        self.assertFalse(store.contains(b"b"))
        self.assertTrue(coordinator.has_gdn_checkpoint(b"a", touch=False))
        self.assertTrue(coordinator.has_gdn_checkpoint(b"c", touch=False))
        self.assertFalse(coordinator.has_gdn_checkpoint(b"b", touch=False))

    def test_dcp_checkpoint_lookup_uses_newest_exact_effective_boundary(self):
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.gdn_checkpoint_limit = 8
        coordinator.hash_block_size = 2
        coordinator.dcp_world_size = 3
        coordinator.pcp_world_size = 1
        spec = MambaSpec(
            block_size=4,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            separate_pool=True,
        )
        block_hashes = [BlockHash(bytes([i])) for i in range(18)]
        # Effective recurrent boundaries are 12 tokens, i.e. every sixth hash.
        coordinator.register_gdn_checkpoint(block_hashes[5])
        coordinator.register_gdn_checkpoint(block_hashes[11])

        self.assertEqual(
            coordinator.find_gdn_checkpoint_boundary(block_hashes, 36, spec), 24
        )
        self.assertEqual(
            coordinator.find_gdn_checkpoint_boundary(block_hashes, 18, spec), 12
        )

    def test_lookup_does_not_touch_lru_until_restore_is_dispatched(self):
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.gdn_checkpoint_limit = 2
        coordinator.hash_block_size = 2
        coordinator.dcp_world_size = 1
        coordinator.pcp_world_size = 1
        spec = MambaSpec(
            block_size=2,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            separate_pool=True,
        )
        block_hashes = [BlockHash(b"a"), BlockHash(b"b")]
        coordinator.register_gdn_checkpoint(block_hashes[0])
        coordinator.register_gdn_checkpoint(block_hashes[1])

        # Merely considering a waiting request must not change scheduler LRU
        # order because workers receive no restore command for that request.
        self.assertEqual(
            coordinator.find_gdn_checkpoint_boundary(block_hashes[:1], 2, spec), 2
        )
        self.assertEqual(list(coordinator.gdn_checkpoint_keys), block_hashes)

        # Admission/restore is the synchronization point shared with workers.
        self.assertTrue(coordinator.has_gdn_checkpoint(block_hashes[0]))
        self.assertEqual(
            list(coordinator.gdn_checkpoint_keys),
            [block_hashes[1], block_hashes[0]],
        )

    def test_worker_snapshot_atomically_replaces_scheduler_lru(self):
        tensors = {"req": [torch.tensor([1.0])]}
        store = _TensorStore(tensors, limit=2)
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.gdn_checkpoint_limit = 2

        # Deliberately create a scheduler order/membership that differs from
        # the worker, then prove the output snapshot repairs both atomically.
        coordinator.register_gdn_checkpoint(b"stale")
        store.save(b"a", "req", None, None, None)
        store.save(b"b", "req", None, None, None)
        store.restore(b"a", "req", None, None, None)
        coordinator.sync_gdn_checkpoints(store.snapshot_keys())

        self.assertEqual(list(coordinator.gdn_checkpoint_keys), [b"b", b"a"])
        self.assertFalse(coordinator.has_gdn_checkpoint(b"stale", touch=False))

        with self.assertRaisesRegex(RuntimeError, "Invalid worker"):
            coordinator.sync_gdn_checkpoints((b"a", b"a"))

    def test_worker_retention_reserves_three_full_async_save_waves(self):
        tensors = {"req": [torch.tensor([1.0])]}
        store = _TensorStore(tensors, limit=8)
        store.advertised_limit = 2

        for key in (b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h"):
            store.save(key, "req", None, None, None)
        self.assertEqual(store.snapshot_keys(), (b"g", b"h"))

        # A restore queued from that snapshot remains physically available
        # after all three reserved advertised-size waves of newer saves.
        for key in (b"i", b"j", b"k", b"l", b"m", b"n"):
            store.save(key, "req", None, None, None)
        self.assertTrue(store.contains(b"g"))
        self.assertTrue(store.contains(b"h"))
        self.assertEqual(store.snapshot_keys(), (b"m", b"n"))

        with self.assertRaisesRegex(ValueError, "advertised_limit"):
            _TensorStore(tensors, limit=8, advertised_limit=0)

    def test_cold_reused_state_is_zeroed_but_restore_destination_is_not(self):
        tensors = {
            "cold": [torch.tensor([4.0, 5.0])],
            "hit": [torch.tensor([6.0, 7.0])],
        }
        requests = {
            "cold": SimpleNamespace(num_computed_tokens=0),
            "hit": SimpleNamespace(num_computed_tokens=0),
        }
        output = SimpleNamespace(
            num_scheduled_tokens={"cold": 1, "hit": 1},
            gdn_checkpoint_restore={"hit": b"key"},
        )
        store = _TensorStore(tensors)
        store.zero_cold_from_output(output, None, requests, None)
        torch.testing.assert_close(tensors["cold"][0], torch.zeros(2))
        torch.testing.assert_close(tensors["hit"][0], torch.tensor([6.0, 7.0]))

    def test_one_slot_state_is_not_freed_by_positional_cleanup(self):
        manager = object.__new__(MambaManager)
        manager.kv_cache_spec = MambaSpec(
            block_size=4,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            separate_pool=True,
        )
        manager.mamba_cache_mode = "align"
        manager.one_slot_align = True
        with patch.object(
            SingleTypeKVCacheManager, "remove_skipped_blocks"
        ) as base_cleanup:
            manager.remove_skipped_blocks("req", 8)
        base_cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
