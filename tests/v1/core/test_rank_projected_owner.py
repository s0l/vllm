import unittest

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.rank_projected_block_pool import RankProjectedPhysicalBlockPool
from vllm.v1.core.rank_projected_owner import ElasticPageOwnerPolicy


class ElasticPageOwnerPolicyTest(unittest.TestCase):
    def test_measured_knots_and_extension_identity(self) -> None:
        policy = ElasticPageOwnerPolicy()
        self.assertEqual((103, 102, 51), policy.counts(4096 // 16))
        short = policy.owners(0, 4096 // 16)
        self.assertEqual((179, 179, 154), policy.counts(8192 // 16))
        self.assertEqual(short, policy.owners(0, 4096 // 16))
        self.assertEqual((683, 683, 682), policy.counts(32768 // 16))
        self.assertEqual((5462, 5461, 5461), policy.counts(262144 // 16))

    def test_request_range_allocation_uses_owner_local_ids(self) -> None:
        policy = ElasticPageOwnerPolicy()
        pool = RankProjectedPhysicalBlockPool((110, 110, 60))
        blocks = [KVCacheBlock(index) for index in range(256)]
        owners = pool.allocate_request_pages(blocks, start_page=0, policy=policy)
        self.assertEqual((103, 102, 51), tuple(owners.count(rank) for rank in range(3)))
        self.assertEqual((7, 8, 9), pool.get_num_free_blocks())
        self.assertEqual(
            tuple(owners),
            tuple(
                block.rank_block_ids.index(
                    block.physical_block_id(owners[index], 3)
                )
                for index, block in enumerate(blocks)
            ),
        )

    def test_bad_range_and_wrong_world_fail_closed(self) -> None:
        policy = ElasticPageOwnerPolicy()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            policy.owners(-1, 1)
        pool = RankProjectedPhysicalBlockPool((8, 8))
        with self.assertRaisesRegex(ValueError, "world size 3"):
            pool.allocate_request_pages([KVCacheBlock(0)], start_page=0, policy=policy)

    def test_local_lengths_and_luts_include_partial_owner_page(self) -> None:
        policy = ElasticPageOwnerPolicy()
        owners, ordinals, prefix = policy.build_luts(256)
        self.assertEqual(256, len(owners))
        self.assertEqual(256, len(ordinals))
        self.assertTrue(all(len(rank_prefix) == 257 for rank_prefix in prefix))
        for tokens in (1, 15, 16, 17, 4095, 4096):
            full_pages, remainder = divmod(tokens, 16)
            for rank in range(3):
                expected = prefix[rank][full_pages] * 16
                if remainder and owners[full_pages] == rank:
                    expected += remainder
                self.assertEqual(expected, policy.local_length(tokens, rank))
            self.assertEqual(
                tokens, sum(policy.local_length(tokens, rank) for rank in range(3))
            )

    def test_live_flashinfer_page_size_preserves_token_shares(self) -> None:
        policy = ElasticPageOwnerPolicy(page_size=64)
        for tokens in (1, 63, 64, 65, 4096, 8192, 32768, 262144):
            local = tuple(policy.local_length(tokens, rank) for rank in range(3))
            self.assertEqual(tokens, sum(local))
        self.assertEqual((26, 25, 13), policy.counts(4096 // 64))
        self.assertEqual((171, 171, 170), policy.counts(32768 // 64))


if __name__ == "__main__":
    unittest.main()
