import unittest

import torch

from vllm.v1.attention.backends.flashinfer import (
    _dcp_causal_paged_custom_mask,
    _dcp_pseudo_decode_rows,
    _dcp_pseudo_block_table_capacity,
    _flashinfer_seq_lens_and_blocks_for_paged_kv,
)


class RankProjectedFlashInferMetadataTest(unittest.TestCase):
    def test_live_pseudo_decode_scratch_covers_projected_table(self) -> None:
        self.assertEqual(
            2816,
            _dcp_pseudo_block_table_capacity(262144, 4096, 64, 3, True),
        )
        self.assertGreaterEqual(
            _dcp_pseudo_block_table_capacity(262144, 4096, 64, 3, True),
            2808,
        )

    def test_decode_page_counts_and_partial_last_page(self) -> None:
        seq_lens = torch.tensor([4097, 8192, 32767], dtype=torch.int32)
        qo_indptr = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        localized = []
        for rank in range(3):
            local, local_np, blocks_np = _flashinfer_seq_lens_and_blocks_for_paged_kv(
                seq_lens,
                qo_indptr,
                num_decodes=3,
                num_prefills=0,
                page_size=64,
                use_dcp=True,
                dcp_world_size=3,
                dcp_rank=rank,
                dcp_kv_cache_interleave_size=1,
                rank_projected_dcp=True,
            )
            self.assertTrue(torch.equal(local, torch.from_numpy(local_np)))
            self.assertTrue(
                torch.equal(
                    torch.from_numpy(blocks_np),
                    torch.div(local + 63, 64, rounding_mode="floor"),
                )
            )
            localized.append(local)
        self.assertTrue(torch.equal(seq_lens, torch.stack(localized).sum(dim=0)))

    def test_pseudo_rows_conserve_every_causal_boundary(self) -> None:
        seq_lens = torch.tensor([4100, 8200], dtype=torch.int32)
        qo_indptr = torch.tensor([0, 4, 12], dtype=torch.int32)
        local_rows = []
        row_maps = []
        for rank in range(3):
            row_map, local = _dcp_pseudo_decode_rows(
                seq_lens,
                qo_indptr,
                3,
                rank,
                1,
                rank_projected_dcp=True,
                page_size=64,
            )
            row_maps.append(row_map)
            local_rows.append(local)
        self.assertTrue(all(torch.equal(row_maps[0], value) for value in row_maps[1:]))
        global_rows = torch.stack(local_rows).sum(dim=0)
        self.assertEqual(
            [4097, 4098, 4099, 4100, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200],
            global_rows.tolist(),
        )

    def test_custom_mask_visible_prefix_matches_local_lengths(self) -> None:
        seq_lens = torch.tensor([33], dtype=torch.int32)
        qo_indptr = torch.tensor([0, 3], dtype=torch.int32)
        for rank in range(3):
            mask, local = _dcp_causal_paged_custom_mask(
                seq_lens,
                qo_indptr,
                3,
                rank,
                1,
                rank_projected_dcp=True,
                page_size=64,
            )
            rows = mask.view(3, int(local[0]))
            self.assertTrue(torch.all(rows[1:] >= rows[:-1]))
            self.assertEqual(int(local[0]), int(rows[-1].sum()))


if __name__ == "__main__":
    unittest.main()
