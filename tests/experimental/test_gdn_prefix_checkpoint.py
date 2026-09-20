# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.v1.core.gdn_checkpoint_policy import GDNCheckpointPolicy
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec
from vllm.v1.request import _parse_prefix_cache_hint_tokens
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
    def test_preview_lease_at_full_capacity_defers_new_save(self):
        policy = GDNCheckpointPolicy(4, 4)
        initial = policy.plan({b"leased": 10})
        policy.acknowledge(initial, initial.resident)
        blocked = policy.plan({b"new": 100}, protected={b"leased"})
        self.assertEqual(blocked.saves, ())
        policy.acknowledge(blocked, blocked.resident)
        self.assertIn(b"leased", policy.visible())
        # Cancellation releases the preview lease; admission can recover.
        recovery = policy.plan({b"new": 100})
        self.assertEqual(recovery.evictions, (b"leased",))
        policy.acknowledge(recovery, recovery.resident)
        self.assertEqual(policy.visible(), (b"new",))

    def test_v2_managed_save_executes_authoritative_plan(self):
        from vllm.v1.worker.gpu.elastic_gdn import V2GDNCheckpointManager

        class Store(_TensorStore):
            def _state_tensors_for_block_ids(self, *args, **kwargs):
                yield from self.tensors["req"]

        manager = V2GDNCheckpointManager()
        manager.store = Store({"req": [torch.tensor([3.0])]})
        manager._block_ids = {"req": ([1],)}
        policy = GDNCheckpointPolicy(4, 4)
        for key, value in ((b"a", 3.0), (b"b", 7.0)):
            plan = policy.plan({key: 100})
            manager.store.tensors["req"][0].fill_(value)
            output = SimpleNamespace(
                gdn_checkpoint_plan=(plan.sequence, plan.evictions, plan.resident),
                gdn_checkpoint_budget=(4, 4),
                gdn_checkpoint_save={"req": key},
            )
            keys = manager.save(output, None, None)
            policy.acknowledge(plan, keys)
            manager.store.tensors["req"][0].zero_()
            manager.store.restore_blocks(key, ([1],), None, None)
            self.assertEqual(manager.store.tensors["req"][0].item(), value)
            with self.assertRaisesRegex(RuntimeError, "sequence"):
                manager.save(output, None, None)
            self.assertEqual(manager.store.checkpoint_bytes, 4)

    def test_cost_aging_retains_long_prefix_through_short_request_churn(self):
        for cost_aware in (False, True):
            policy = GDNCheckpointPolicy(36 * 100, 100, cost_aware=cost_aware)
            plan = policy.plan({b"long": 31000})
            policy.acknowledge(plan, plan.resident)
            for i in range(128):
                plan = policy.plan({str(i).encode(): 2500})
                policy.acknowledge(plan, plan.resident)
                self.assertLessEqual(len(policy.entries) * 100, 3600)
            self.assertEqual(b"long" in policy.visible(), cost_aware)
            # Cost protection must age out under sustained unrelated demand.
            for i in range(128, 2000):
                plan = policy.plan({str(i).encode(): 2500})
                policy.acknowledge(plan, plan.resident)
            self.assertNotIn(b"long", policy.visible())

    def test_retired_key_cannot_be_resurrected_by_delayed_publication(self):
        policy = GDNCheckpointPolicy(8, 4)
        first = policy.plan({b"a": 1, b"b": 1})
        self.assertEqual(policy.visible(), ())
        second = policy.plan({b"c": 2})
        policy.acknowledge(first, first.resident)
        self.assertNotIn(b"a", policy.visible())
        self.assertNotIn(b"c", policy.visible())
        with self.assertRaisesRegex(RuntimeError, "Out-of-order"):
            policy.acknowledge(first, first.resident)
        with self.assertRaisesRegex(RuntimeError, "membership"):
            policy.acknowledge(second, (b"wrong",))
        policy.acknowledge(second, second.resident)
        self.assertIn(b"c", policy.visible())

    def test_managed_restore_precedes_eviction_at_full_budget(self):
        tensors = {"req": [torch.tensor([3.0])]}
        store = _TensorStore(tensors)
        policy = GDNCheckpointPolicy(4, 4)

        def execute(plan, saves):
            output = SimpleNamespace(
                gdn_checkpoint_plan=(plan.sequence, plan.evictions, plan.resident),
                gdn_checkpoint_budget=(4, 4),
                gdn_checkpoint_save=saves,
            )
            store.save_from_output(output, None, None, None)

        first = policy.plan({b"a": 100})
        execute(first, {"req": b"a"})
        policy.acknowledge(first, store.snapshot_keys())
        # A restore already queued before retirement still finds exact bytes.
        second = policy.plan({b"b": 10})
        tensors["req"][0].zero_()
        store.restore(b"a", "req", None, None, None)
        self.assertEqual(tensors["req"][0].item(), 3.0)
        tensors["req"][0].fill_(7)
        execute(second, {"req": b"b"})
        policy.acknowledge(second, store.snapshot_keys())
        self.assertFalse(store.contains(b"a"))
        self.assertEqual(store.checkpoint_bytes, 4)
        tensors["req"][0].zero_()
        store.restore(b"b", "req", None, None, None)
        self.assertEqual(tensors["req"][0].item(), 7.0)

    def test_managed_budget_rejects_oversize_before_copy_and_recovers(self):
        store = _TensorStore({"req": [torch.ones(2)]})
        store.managed_budget = (4, 4)
        with self.assertRaisesRegex(RuntimeError, "before allocation"):
            store.save(b"a", "req", None, None, None)
        self.assertEqual(store.checkpoint_bytes, 0)
        store.tensors["req"] = [torch.tensor([9.0])]
        store.save(b"a", "req", None, None, None)
        with self.assertRaises(RuntimeError):
            store.save(b"a", "req", None, None, None)
        self.assertEqual(store.checkpoint_bytes, 4)

    def test_wave_larger_than_budget_is_admitted_without_overallocation(self):
        policy = GDNCheckpointPolicy(8, 4)
        plan = policy.plan({b"a": 10, b"b": 20, b"c": 30})
        self.assertEqual(set(plan.saves), {b"b", b"c"})
        policy.acknowledge(plan, plan.resident)
        for candidates in ({b"c": 30, b"d": 40}, {}, {b"d": 40}):
            plan = policy.plan(candidates)
            policy.acknowledge(plan, plan.resident)
        self.assertEqual(set(policy.visible()), {b"c", b"d"})

    def test_attention_cow_disjoint_and_overlapping_sources(self):
        from tests.experimental.elastic_attention_cow_region_control import main

        main()

    def test_real_geometry_automatic_match_resume_and_bounded_splits(self):
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.mamba_block_pool = None
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.enable_dcp_fine_prefix = True
        scheduler = object.__new__(Scheduler)
        scheduler.mamba_has_prefill_checkpoint_blocks = False
        scheduler.mamba_prefill_checkpoint_alignment = 1
        scheduler.mamba_fine_grained_prefix_cache = False
        scheduler.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
        scheduler.hash_block_size = 192
        scheduler.block_size = 7488
        scheduler.mamba_state_update_alignment = 64
        scheduler.use_eagle = True
        scheduler.use_eagle_block_drop = True
        scheduler.mamba_partial_cache_hit = True
        scheduler.max_num_scheduled_tokens = 4096
        scheduler._long_prefill_chunk_cap = lambda request: 0
        for length in (192, 384, 3000, 3072, 7488, 16065):
            request = SimpleNamespace(
                num_computed_tokens=0,
                num_prompt_tokens=length,
                execution_prefill_len=length,
                shared_prefix_boundary=0,
                prefix_cache_hint_tokens=0,
            )
            tail = (length - 1) // 192 * 192
            expected = tuple(b for b in (tail - 192, tail) if b > 0)
            self.assertEqual(
                scheduler._fine_prefix_checkpoint_boundaries(request), expected
            )
            visited = []
            while request.num_computed_tokens < length:
                count = scheduler._mamba_block_aligned_split(
                    request, min(4096, length - request.num_computed_tokens)
                )
                self.assertGreater(count, 0)
                request.num_computed_tokens += count
                visited.append(request.num_computed_tokens)
            self.assertTrue(set(expected).issubset(visited))
            self.assertTrue(all(boundary % 64 == 0 for boundary in visited[:-1]))
            self.assertLessEqual(len(visited), length // 4096 + 7)

    def test_hint_geometry_rejected_before_scheduling(self):
        env = {"AG2_VLLM_DCP_FINE_PREFIX": "1"}
        for hint, error in ((336, "hash aligned"), (192, "overlap")):
            with (
                patch.dict(
                    os.environ,
                    {**env, "AG2_VLLM_DCP_FINE_PREFIX_HINT_TOKENS": str(hint)},
                    clear=True,
                ),
                self.assertRaisesRegex(ValueError, error),
            ):
                _parse_prefix_cache_hint_tokens(
                    {"ag2_prefix_cache_hint_tokens": hint},
                    hash_block_size=192,
                    use_eagle=True,
                )
        with patch.dict(
            os.environ,
            {**env, "AG2_VLLM_DCP_FINE_PREFIX_HINT_TOKENS": "384"},
            clear=True,
        ):
            self.assertEqual(
                _parse_prefix_cache_hint_tokens(
                    {"ag2_prefix_cache_hint_tokens": 384},
                    hash_block_size=192,
                    use_eagle=True,
                ),
                384,
            )
            with self.assertRaisesRegex(ValueError, "integer"):
                _parse_prefix_cache_hint_tokens({"ag2_prefix_cache_hint_tokens": 384.5})

    def test_restore_metadata_failure_does_not_partially_mutate(self):
        tensors = {"req": [torch.tensor([1.0]), torch.tensor([2.0])]}
        store = _TensorStore(tensors)
        store.save(b"key", "req", None, None, None)
        tensors["req"] = [torch.tensor([10.0]), torch.tensor([20.0, 30.0])]
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            store.restore(b"key", "req", None, None, None)
        self.assertEqual(tensors["req"][0].item(), 10.0)
        self.assertEqual(store.restores, 0)

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
        coordinator.mamba_block_pool = None
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
        coordinator.mamba_block_pool = None
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

    def test_dcp_fine_checkpoint_lookup_uses_hash_boundary(self):
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.mamba_block_pool = None
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.gdn_checkpoint_limit = 8
        coordinator.hash_block_size = 2
        coordinator.dcp_world_size = 3
        coordinator.pcp_world_size = 1
        coordinator.enable_dcp_fine_prefix = True
        spec = MambaSpec(
            block_size=4,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            separate_pool=True,
        )
        block_hashes = [BlockHash(bytes([i])) for i in range(18)]
        coordinator.register_gdn_checkpoint(block_hashes[4])

        self.assertEqual(
            coordinator.find_gdn_checkpoint_boundary(block_hashes, 18, spec), 10
        )
        self.assertEqual(
            coordinator.find_gdn_checkpoint_boundary(block_hashes, 8, spec), 0
        )

    def test_prefix_hint_is_server_bounded_and_fail_closed(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "requires"),
        ):
            _parse_prefix_cache_hint_tokens({"ag2_prefix_cache_hint_tokens": 336})

        env = {
            "AG2_VLLM_DCP_FINE_PREFIX": "1",
            "AG2_VLLM_DCP_FINE_PREFIX_HINT_TOKENS": "336",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                _parse_prefix_cache_hint_tokens({"ag2_prefix_cache_hint_tokens": 336}),
                336,
            )
            with self.assertRaisesRegex(ValueError, "server-configured"):
                _parse_prefix_cache_hint_tokens({"ag2_prefix_cache_hint_tokens": 168})
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                _parse_prefix_cache_hint_tokens({"ag2_prefix_cache_hint_tokens": True})

    def test_prefix_hint_uses_only_an_opted_in_server_default(self):
        env = {
            "AG2_VLLM_DCP_FINE_PREFIX": "1",
            "AG2_VLLM_DCP_FINE_PREFIX_HINT_TOKENS": "336",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(_parse_prefix_cache_hint_tokens(None), 0)
        env["AG2_VLLM_DCP_FINE_PREFIX_DEFAULT"] = "1"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(_parse_prefix_cache_hint_tokens(None), 336)

    def test_fine_hint_is_an_exact_scheduler_stop(self):
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.mamba_block_pool = None
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.enable_dcp_fine_prefix = True
        scheduler = object.__new__(Scheduler)
        scheduler.mamba_has_prefill_checkpoint_blocks = False
        scheduler.mamba_prefill_checkpoint_alignment = 1
        scheduler.mamba_fine_grained_prefix_cache = False
        scheduler.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
        scheduler.hash_block_size = 2
        scheduler.block_size = 12
        scheduler.use_eagle = False
        scheduler.use_eagle_block_drop = False
        scheduler.mamba_partial_cache_hit = True
        request = SimpleNamespace(
            num_computed_tokens=0,
            num_prompt_tokens=10,
            execution_prefill_len=10,
            num_tokens=10,
            shared_prefix_boundary=2,
            prefix_cache_hint_tokens=2,
        )

        self.assertEqual(scheduler._mamba_block_aligned_split(request, 10), 2)

    def test_eagle_fine_hint_materializes_resume_then_match_boundaries(self):
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.mamba_block_pool = None
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.enable_dcp_fine_prefix = True
        scheduler = object.__new__(Scheduler)
        scheduler.mamba_has_prefill_checkpoint_blocks = False
        scheduler.mamba_prefill_checkpoint_alignment = 1
        scheduler.mamba_fine_grained_prefix_cache = False
        scheduler.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
        scheduler.hash_block_size = 2
        scheduler.block_size = 12
        scheduler.use_eagle = True
        scheduler.use_eagle_block_drop = True
        scheduler.mamba_partial_cache_hit = True
        request = SimpleNamespace(
            num_computed_tokens=0,
            num_prompt_tokens=10,
            execution_prefill_len=10,
            num_tokens=10,
            shared_prefix_boundary=4,
            prefix_cache_hint_tokens=4,
        )

        self.assertEqual(scheduler._fine_prefix_resume_boundary(request), 2)
        self.assertEqual(scheduler._mamba_block_aligned_split(request, 10), 2)
        request.num_computed_tokens = 2
        self.assertEqual(scheduler._mamba_block_aligned_split(request, 8), 2)

    def test_attention_registers_shared_partial_boundary(self):
        manager = object.__new__(FullAttentionManager)
        manager.block_size = 12
        manager.kv_cache_group_id = 0
        manager.block_pool = MagicMock(hash_block_size=2)
        block = SimpleNamespace()
        manager.req_to_blocks = {"req": [block]}
        request = SimpleNamespace(
            request_id="req",
            num_prompt_tokens=9,
            shared_prefix_boundary=2,
        )

        manager._cache_partial_tail_block(request, num_tokens=2)

        manager.block_pool.cache_partial_block.assert_called_once_with(
            request=request,
            block=block,
            num_tokens=2,
            kv_cache_group_id=0,
            block_size=12,
        )

    def test_fine_checkpoint_save_filter_preserves_coarse_and_named_boundaries(self):
        spec = MambaSpec(
            block_size=4,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            separate_pool=True,
        )
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.mamba_block_pool = None
        coordinator.gdn_checkpoint_keys = OrderedDict()
        coordinator.enable_dcp_fine_prefix = True
        coordinator.hash_block_size = 2
        coordinator.dcp_world_size = 3
        coordinator.pcp_world_size = 1
        coordinator.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["gdn"], spec)]
        )
        scheduler = object.__new__(Scheduler)
        scheduler.mamba_has_prefill_checkpoint_blocks = False
        scheduler.mamba_prefill_checkpoint_alignment = 1
        scheduler.mamba_fine_grained_prefix_cache = False
        scheduler.kv_cache_manager = SimpleNamespace(coordinator=coordinator)
        scheduler.hash_block_size = 2
        scheduler.use_eagle = True
        scheduler.use_eagle_block_drop = True
        request = SimpleNamespace(
            num_prompt_tokens=19,
            execution_prefill_len=19,
            shared_prefix_boundary=4,
            prefix_cache_hint_tokens=4,
        )

        self.assertTrue(scheduler._should_save_gdn_checkpoint(request, 2))
        self.assertTrue(scheduler._should_save_gdn_checkpoint(request, 4))
        self.assertTrue(scheduler._should_save_gdn_checkpoint(request, 12))
        self.assertTrue(scheduler._should_save_gdn_checkpoint(request, 18))
        self.assertFalse(scheduler._should_save_gdn_checkpoint(request, 6))

    def test_lookup_does_not_touch_lru_until_restore_is_dispatched(self):
        coordinator = object.__new__(HybridKVCacheCoordinator)
        coordinator.mamba_block_pool = None
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
        coordinator.mamba_block_pool = None
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

    def test_worker_reports_exact_host_bytes_and_evictions(self):
        tensors = {
            "a": [
                torch.zeros(3, dtype=torch.float32),
                torch.zeros(2, dtype=torch.int16),
            ],
            "b": [torch.zeros(4, dtype=torch.float32)],
            "wrong": [torch.zeros(5, dtype=torch.float32)],
        }
        store = _TensorStore(tensors, limit=1)

        store.save(b"a", "a", None, None, None)
        self.assertEqual(store.checkpoint_bytes, 16)
        self.assertEqual(store.peak_checkpoint_bytes, 16)
        self.assertEqual(store.evictions, 0)

        store.save(b"b", "b", None, None, None)
        self.assertEqual(store.checkpoint_bytes, 16)
        self.assertEqual(store.peak_checkpoint_bytes, 32)
        self.assertEqual(store.evictions, 1)

        # Replacing an existing key must not double-count retained bytes.
        store.save(b"b", "b", None, None, None)
        self.assertEqual(store.checkpoint_bytes, 16)
        self.assertEqual(store.evictions, 1)

        with self.assertRaisesRegex(RuntimeError, "byte size changed"):
            store.save(b"wrong", "wrong", None, None, None)

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
