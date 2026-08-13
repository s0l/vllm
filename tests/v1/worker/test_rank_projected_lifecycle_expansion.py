import unittest

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.gpu.model_runner import (
    _expand_rank_projected_block_copies,
    _expand_rank_projected_zero_ids,
)


class RankProjectedLifecycleExpansionTest(unittest.TestCase):
    def test_zeroing_expands_both_physical_pages_in_order(self) -> None:
        self.assertEqual(
            [6, 7, 14, 15], _expand_rank_projected_zero_ids([3, 7])
        )

    def test_cow_copies_both_physical_pages(self) -> None:
        copies = _expand_rank_projected_block_copies(
            [KVCacheBlockCopy(src_block_id=3, dst_block_id=9)]
        )
        self.assertEqual(
            [
                KVCacheBlockCopy(src_block_id=6, dst_block_id=18),
                KVCacheBlockCopy(src_block_id=7, dst_block_id=19),
            ],
            copies,
        )


if __name__ == "__main__":
    unittest.main()
