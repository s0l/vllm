import unittest

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.rank_projected_block_pool import RankProjectedPhysicalBlockPool


class RankProjectedPhysicalBlockPoolTest(unittest.TestCase):
    def test_vector_capacity_allocation_free_and_generation(self) -> None:
        pool = RankProjectedPhysicalBlockPool((3, 2, 1))
        blocks = [KVCacheBlock(index) for index in range(5)]
        pool.allocate(blocks, [0, 1, 0, 2, 1])
        self.assertEqual((1, 0, 0), pool.get_num_free_blocks())
        self.assertEqual(
            (0, 0, 0),
            (
                blocks[0].physical_block_id(0, 3),
                blocks[1].physical_block_id(1, 3),
                blocks[3].physical_block_id(2, 3),
            ),
        )
        generations = [block.physical_generation for block in blocks]
        self.assertEqual([0] * 5, generations)

        pool.free(blocks)
        self.assertEqual((3, 2, 1), pool.get_num_free_blocks())
        replacement = KVCacheBlock(10)
        pool.allocate([replacement], [2])
        self.assertEqual(1, replacement.physical_generation)

    def test_exhaustion_is_atomic(self) -> None:
        pool = RankProjectedPhysicalBlockPool((2, 2, 1))
        blocks = [KVCacheBlock(index) for index in range(3)]
        before = pool.get_num_free_blocks()
        with self.assertRaisesRegex(ValueError, "rank 2.*exhausted"):
            pool.allocate(blocks, [0, 2, 2])
        self.assertEqual(before, pool.get_num_free_blocks())
        self.assertTrue(all(not block.is_rank_projected for block in blocks))

    def test_duplicate_and_referenced_free_fail_before_mutation(self) -> None:
        pool = RankProjectedPhysicalBlockPool((2, 2, 2))
        first = KVCacheBlock(1)
        second = KVCacheBlock(2)
        pool.allocate([first, second], [0, 1])
        first.ref_cnt = 1
        before = pool.get_num_free_blocks()
        with self.assertRaisesRegex(ValueError, "referenced"):
            pool.free([first, second])
        self.assertEqual(before, pool.get_num_free_blocks())
        self.assertTrue(first.is_rank_projected)
        self.assertTrue(second.is_rank_projected)


if __name__ == "__main__":
    unittest.main()
