import unittest

import torch

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens


class RankProjectedDCPLocalizeTest(unittest.TestCase):
    def test_conservation_partial_pages_and_known_knots(self) -> None:
        lengths = torch.tensor(
            [0, 1, 15, 16, 17, 4095, 4096, 8192, 32768, 262144],
            dtype=torch.int32,
        )
        all_ranks = get_dcp_local_seq_lens(
            lengths, 3, None, 1, rank_projected=True
        )
        self.assertTrue(torch.equal(lengths, all_ranks.sum(dim=-1)))
        self.assertEqual([1648, 1632, 816], all_ranks[6].tolist())
        self.assertEqual([2864, 2864, 2464], all_ranks[7].tolist())
        self.assertEqual([10928, 10928, 10912], all_ranks[8].tolist())
        self.assertEqual([87392, 87376, 87376], all_ranks[9].tolist())

    def test_scalar_and_explicit_rank_views_match(self) -> None:
        lengths = torch.arange(0, 2049, 7, dtype=torch.int32)
        all_ranks = get_dcp_local_seq_lens(
            lengths, 3, None, 1, rank_projected=True
        )
        for rank in range(3):
            explicit = get_dcp_local_seq_lens(
                lengths, 3, rank, 1, rank_projected=True
            )
            self.assertTrue(torch.equal(explicit, all_ranks[:, rank]))

    def test_wrong_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "DCP3"):
            get_dcp_local_seq_lens(
                torch.tensor([16]), 2, 0, 1, rank_projected=True
            )


if __name__ == "__main__":
    unittest.main()
