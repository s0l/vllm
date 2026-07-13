import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.mamba_utils import GDNPrefixCheckpointStore


class _TensorStore(GDNPrefixCheckpointStore):
    def __init__(self, tensors: dict[str, list[torch.Tensor]], limit: int = 2):
        super().__init__(limit=limit)
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
