# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from unittest.mock import patch

from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import init_none_hash


class TestElasticCacheRetention(unittest.TestCase):
    def test_zero_allocation_does_not_walk_blocks(self):
        pool = self.pool(n=1000, frontier=1)

        class ForbiddenBlocks:
            def __getitem__(self, key):
                raise AssertionError("zero allocation accessed block storage")

        pool.blocks = ForbiddenBlocks()
        self.assertEqual(pool.get_new_blocks(0), [])
        self.assertEqual(pool.get_num_free_blocks(), 999)

    def test_single_allocation_does_not_walk_unused_capacity(self):
        for frontier, active in ((1, 1000), (25, 1000), (None, 1000), (None, 500)):
            pool = BlockPool(
                1000,
                True,
                192,
                prefer_low_id_allocations=True,
                active_num_gpu_blocks=active,
            )
            pool.cache_preservation_num_blocks = frontier

            class BoundedBlocks(list):
                def __getitem__(self, key):
                    if not isinstance(key, int) or key != 1:
                        raise AssertionError(
                            "walked beyond sufficient first free block"
                        )
                    return super().__getitem__(key)

            pool.blocks = BoundedBlocks(pool.blocks)
            self.assertEqual([b.block_id for b in pool.get_new_blocks(1)], [1])
            self.assertEqual(pool.get_num_free_blocks(), active - 2)

    def setUp(self):
        init_none_hash(sha256)

    def pool(self, n=69, frontier=25):
        pool = BlockPool(n, True, 192, prefer_low_id_allocations=True)
        pool.cache_preservation_num_blocks = frontier
        return pool

    def cache(self, pool, name, tokens, block_size=7488):
        req = make_request(name, [1] * tokens, 192, sha256, cache_salt=name)
        count = (tokens + block_size - 1) // block_size
        blocks = pool.get_new_blocks(count)
        full = tokens // block_size
        pool.cache_full_blocks(req, blocks, 0, full, block_size, 0)
        partial = tokens // 192 * 192
        if partial > full * block_size:
            pool.cache_partial_block(req, blocks[-1], partial, 0, block_size)
        return req, blocks

    def hit(self, pool, req, tokens):
        return pool.get_cached_block(req.block_hashes[tokens // 192 - 1], [0])

    def test_long_prefix_survives_matched_32_short_then_ages_out(self):
        pool = self.pool()
        req, long_blocks = self.cache(pool, "long", 30184)
        pool.free_blocks(reversed(long_blocks))
        pool.touch(long_blocks)
        pool.free_blocks(reversed(long_blocks))
        for wave in range(4):
            live = []
            for lane in range(8):
                _, blocks = self.cache(pool, f"short-{wave}-{lane}", 2284)
                live.extend(blocks)
            self.assertTrue(all(b.block_id < 25 for b in live))
            pool.free_blocks(live)
        for end in range(7488, 29953, 7488):
            self.assertIsNotNone(self.hit(pool, req, end))
        # Stable-state exactness: all retained keys still point to the same bytes.
        self.assertEqual(self.hit(pool, req, 29952), [long_blocks[3]])
        self.assertEqual(pool.get_num_free_blocks(), 68)
        for i in range(256):
            _, blocks = self.cache(pool, f"age-{i}", 2284)
            pool.free_blocks(blocks)
        self.assertIsNone(self.hit(pool, req, 7488))
        self.assertLessEqual(len(pool._cache_values), pool.num_gpu_blocks - 1)

    def test_equal_cost_evicts_leaf_before_root_and_shrinks(self):
        pool = self.pool(n=7, frontier=4)
        req, blocks = self.cache(pool, "chain", 3 * 7488)
        pool.free_blocks(reversed(blocks))
        chosen = pool.get_new_blocks(1)
        self.assertEqual(chosen, [blocks[-1]])
        self.assertIsNotNone(self.hit(pool, req, 7488))
        self.assertIsNone(self.hit(pool, req, 3 * 7488))
        self.assertTrue(pool.deactivate_tail_blocks(4))
        pool.free_blocks(chosen)
        pool.activate_tail_blocks(7)
        self.assertIsNotNone(self.hit(pool, req, 7488))

    def test_physical_id_control_reproduces_zero_prefix(self):
        pool = self.pool()
        req, blocks = self.cache(pool, "long", 30184)
        pool.free_blocks(blocks)
        pool.touch(blocks)
        pool.free_blocks(blocks)
        with patch.object(
            BlockPool, "_cache_eviction_key", lambda self, b: (b.block_id, 0, 0)
        ):
            for wave in range(4):
                live = []
                for lane in range(8):
                    _, short = self.cache(pool, f"short-{wave}-{lane}", 2284)
                    live.extend(short)
                pool.free_blocks(live)
        self.assertIsNone(self.hit(pool, req, 7488))

    def test_leases_override_priority_reset_and_recovery(self):
        pool = self.pool(n=7, frontier=4)
        _, blocks = self.cache(pool, "leased", 3 * 7488)
        pool.free_blocks(blocks)
        pool.touch(blocks[-1:])
        chosen = pool.get_new_blocks(2)
        self.assertNotIn(blocks[-1], chosen)
        self.assertFalse(pool.reset_prefix_cache())
        pool.free_blocks(chosen)
        pool.free_blocks(blocks[-1:])
        self.assertTrue(pool.reset_prefix_cache())
        self.assertEqual(pool._cache_values, {})
        self.assertEqual(pool._cache_inflation, 0)
        req, new = self.cache(pool, "recovery", 2284)
        pool.free_blocks(new)
        self.assertIsNotNone(self.hit(pool, req, 2112))

    def test_alias_move_and_eviction_preserve_charge(self):
        pool = self.pool()
        req, blocks = self.cache(pool, "partial", 7680)
        old = blocks[-1]
        value = pool._cache_values[old.block_id]
        dst = pool.get_new_blocks(1)[0]
        pool.move_block_hashes(old, dst)
        self.assertNotIn(old.block_id, pool._cache_values)
        self.assertEqual(pool._cache_values[dst.block_id], value)
        self.assertEqual(self.hit(pool, req, 7680), [dst])
        pool._maybe_evict_cached_block(dst)
        self.assertNotIn(dst.block_id, pool._cache_values)
        self.assertIsNone(self.hit(pool, req, 7680))

    def test_partial_to_full_promotion_replaces_cost_and_hash(self):
        pool = self.pool()
        req, blocks = self.cache(pool, "promotion", 2284)
        self.assertEqual(pool._cache_values[blocks[0].block_id][1], 2112)
        req.append_output_token_ids([1] * (7488 - 2284))
        pool.cache_full_blocks(req, blocks, 0, 1, 7488, 0)
        self.assertEqual(pool._cache_values[blocks[0].block_id][1], 7488)
        self.assertIsNone(self.hit(pool, req, 2112))
        self.assertEqual(self.hit(pool, req, 7488), blocks)
