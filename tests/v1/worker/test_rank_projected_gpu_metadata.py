import unittest

import torch

from vllm.v1.core.rank_projected_owner import ElasticPageOwnerPolicy
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens


class RankProjectedGPUMetadataTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_live_block_id_range_fits_two_worker_blocks(self) -> None:
        device = torch.device("cuda:0")
        worker_blocks = 68
        logical_blocks = worker_blocks // 2
        kernel_pages_per_worker_block = 832 // 64

        for rank in range(3):
            with self.subTest(rank=rank):
                tables = BlockTables(
                    block_sizes=[832],
                    max_num_reqs=1,
                    max_num_batched_tokens=1,
                    max_num_blocks_per_group=[logical_blocks],
                    device=device,
                    kernel_block_sizes=[64],
                    cp_size=3,
                    cp_rank=rank,
                    cp_interleave=1,
                    rank_projected_groups=[True],
                )
                tables.append_block_ids(
                    0, (list(range(logical_blocks)),), overwrite=True
                )
                count = int(tables.num_blocks.np[0, 0])
                physical_pages = tables.host_block_tables[0][0, :count].tolist()
                self.assertEqual(len(physical_pages), len(set(physical_pages)))
                self.assertGreater(count, 0)
                self.assertGreaterEqual(min(physical_pages), 0)
                self.assertLess(
                    max(physical_pages),
                    worker_blocks * kernel_pages_per_worker_block,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_slot_and_length_metadata_direct_and_graph_replay(self) -> None:
        device = torch.device("cuda:0")
        tokens = 32768
        pages = tokens // 64
        policy = ElasticPageOwnerPolicy(page_size=64)
        owners, ordinals, _ = policy.build_luts(pages)

        for rank in range(3):
            with self.subTest(rank=rank):
                local_pages = owners.count(rank)
                tables = BlockTables(
                    # Exact Qwopus hybrid geometry after page unification: one
                    # worker KV block contains 13 FlashInfer pages and covers
                    # 2496 global tokens under DCP3.
                    block_sizes=[832],
                    max_num_reqs=1,
                    max_num_batched_tokens=tokens,
                    max_num_blocks_per_group=[14],
                    device=device,
                    kernel_block_sizes=[64],
                    cp_size=3,
                    cp_rank=rank,
                    cp_interleave=1,
                    rank_projected_groups=[True],
                )
                logical_ids = list(range(1000, 1000 + 14))
                tables.append_block_ids(0, (logical_ids,), overwrite=True)
                tables.apply_staged_writes()

                idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
                query_start = torch.tensor(
                    [0, tokens], dtype=torch.int32, device=device
                )
                positions = torch.arange(tokens, dtype=torch.int32, device=device)
                seq_lens = torch.tensor([tokens], dtype=torch.int32, device=device)
                local_seq_lens = torch.empty(1, dtype=torch.int32, device=device)

                def producer(
                    tables=tables,
                    idx_mapping=idx_mapping,
                    query_start=query_start,
                    positions=positions,
                    local_seq_lens=local_seq_lens,
                    seq_lens=seq_lens,
                    rank=rank,
                ) -> None:
                    tables.compute_slot_mappings(
                        idx_mapping, query_start, positions, tokens
                    )
                    prepare_dcp_local_seq_lens(
                        local_seq_lens,
                        seq_lens,
                        1,
                        3,
                        rank,
                        1,
                        tables.rank_projected_owner_lut,
                        tables.rank_projected_prefix_counts,
                        tables.rank_projected_page_size,
                    )

                side = torch.cuda.Stream()
                with torch.cuda.stream(side):
                    for _ in range(3):
                        producer()
                torch.cuda.current_stream().wait_stream(side)
                producer()
                torch.cuda.synchronize()

                expected = torch.full(
                    (tokens,), -1, dtype=torch.int64, device=device
                )
                owner_tensor = torch.tensor(owners, dtype=torch.int64, device=device)
                ordinal_tensor = torch.tensor(
                    ordinals, dtype=torch.int64, device=device
                )
                page_indices = positions.to(torch.int64) // 64
                owned = owner_tensor[page_indices] == rank
                expected_physical_ids = []
                for page, owner in enumerate(owners):
                    if owner != rank:
                        continue
                    superblock_start = page // 39 * 39
                    local_slot = owners[superblock_start:page].count(rank)
                    expected_physical_ids.append(
                        (1000 + page // 39) * 26 + local_slot
                    )
                physical_tensor = torch.tensor(
                    expected_physical_ids, dtype=torch.int64, device=device
                )
                expected[owned] = (
                    physical_tensor[ordinal_tensor[page_indices[owned]]]
                    * 64
                    + positions[owned].to(torch.int64) % 64
                )
                self.assertTrue(torch.equal(expected, tables.slot_mappings[0]))
                self.assertEqual(local_pages * 64, int(local_seq_lens.item()))

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    producer()
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(expected, tables.slot_mappings[0]))
                self.assertEqual(local_pages * 64, int(local_seq_lens.item()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_rejects_non_integral_kernel_page_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            BlockTables(
                block_sizes=[17],
                max_num_reqs=1,
                max_num_batched_tokens=16,
                max_num_blocks_per_group=[1],
                device=torch.device("cuda:0"),
                kernel_block_sizes=[16],
                cp_size=3,
                cp_rank=0,
                cp_interleave=1,
                rank_projected_groups=[True],
            )


if __name__ == "__main__":
    unittest.main()
