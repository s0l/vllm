import unittest

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock


class RankProjectedKVCacheBlocksTest(unittest.TestCase):
    def test_attention_is_compact_and_scalar_group_is_replicated(self) -> None:
        attention = [KVCacheBlock(index) for index in range(4)]
        owners = (0, 1, 0, 2)
        physical_ids = (7, 11, 8, 13)
        for block, owner, physical_id in zip(attention, owners, physical_ids):
            block.set_rank_block_id(
                owner_rank=owner,
                physical_block_id=physical_id,
                world_size=3,
                generation=2,
            )
        gdn = [KVCacheBlock(40), KVCacheBlock(41)]
        blocks = KVCacheBlocks((attention, gdn))

        self.assertEqual(
            (
                ([7, 8], [40, 41]),
                ([11], [40, 41]),
                ([13], [40, 41]),
            ),
            blocks.get_rank_projected_block_ids(3),
        )
        with self.assertRaisesRegex(ValueError, "rank-projected"):
            blocks.get_block_ids()

    def test_identity_is_single_assignment_and_world_checked(self) -> None:
        block = KVCacheBlock(5)
        block.set_rank_block_id(
            owner_rank=1,
            physical_block_id=9,
            world_size=3,
            generation=4,
        )
        self.assertEqual((None, 9, None), block.rank_block_ids)
        self.assertEqual(4, block.physical_generation)
        with self.assertRaisesRegex(ValueError, "already set"):
            block.set_rank_block_id(
                owner_rank=0,
                physical_block_id=10,
                world_size=3,
                generation=5,
            )
        with self.assertRaisesRegex(ValueError, "world size"):
            block.physical_block_id(0, 2)

    def test_reset_requires_zero_references(self) -> None:
        block = KVCacheBlock(3, ref_cnt=1)
        block.set_rank_block_id(
            owner_rank=2,
            physical_block_id=17,
            world_size=3,
            generation=1,
        )
        with self.assertRaisesRegex(ValueError, "referenced"):
            block.reset_rank_block_id()
        block.ref_cnt = 0
        block.reset_rank_block_id()
        self.assertFalse(block.is_rank_projected)
        self.assertEqual(3, block.physical_block_id(0, 3))


if __name__ == "__main__":
    unittest.main()
