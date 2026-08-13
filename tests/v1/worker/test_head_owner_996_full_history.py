import unittest

import torch

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens


class HeadOwner996FullHistoryTest(unittest.TestCase):
    def test_full_attention_merge_preserves_and_validates_history_mode(self) -> None:
        def spec(full_history: bool) -> FullAttentionSpec:
            return FullAttentionSpec(
                block_size=64,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.float8_e4m3fn,
                dcp_full_history=full_history,
            )

        merged = FullAttentionSpec.merge([spec(True), spec(True)])
        self.assertTrue(merged.dcp_full_history)
        with self.assertRaisesRegex(AssertionError, "same attention spec"):
            FullAttentionSpec.merge([spec(True), spec(False)])

    def test_scheduler_manager_allocates_one_block_per_physical_page(self) -> None:
        def manager(full_history: bool) -> FullAttentionManager:
            spec = FullAttentionSpec(
                block_size=64,
                num_kv_heads=4,
                head_size=256,
                dtype=torch.float8_e4m3fn,
                dcp_full_history=full_history,
            )
            return FullAttentionManager(
                kv_cache_spec=spec,
                block_pool=BlockPool(
                    num_gpu_blocks=32,
                    enable_caching=False,
                    hash_block_size=64,
                    enable_kv_cache_events=False,
                ),
                enable_caching=False,
                kv_cache_group_id=0,
                scheduler_block_size=192,
                dcp_world_size=3,
            )

        self.assertEqual(64, manager(True).block_size)
        self.assertEqual(192, manager(False).block_size)

    def test_cpu_lengths_are_full_on_every_rank(self) -> None:
        seq_lens = torch.tensor([0, 1, 63, 64, 257], dtype=torch.int32)
        for rank in range(3):
            actual = get_dcp_local_seq_lens(
                seq_lens,
                dcp_size=3,
                dcp_rank=rank,
                cp_kv_cache_interleave_size=1,
                full_history=True,
            )
            self.assertTrue(torch.equal(seq_lens, actual))
            self.assertIsNot(seq_lens, actual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_slot_and_length_metadata_direct_and_graph_replay(self) -> None:
        device = torch.device("cuda:0")
        tokens = 512
        page_size = 64
        block_ids = list(range(100, 100 + tokens // page_size))

        results: list[torch.Tensor] = []
        for rank in range(3):
            with self.subTest(rank=rank):
                tables = BlockTables(
                    block_sizes=[page_size],
                    max_num_reqs=1,
                    max_num_batched_tokens=tokens,
                    max_num_blocks_per_group=[len(block_ids)],
                    device=device,
                    kernel_block_sizes=[page_size],
                    cp_size=3,
                    cp_rank=rank,
                    cp_interleave=1,
                    full_history_groups=[True],
                )
                tables.append_block_ids(0, (block_ids,), overwrite=True)
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
                        full_history=True,
                    )

                producer()
                torch.cuda.synchronize()
                expected = (
                    torch.tensor(block_ids, dtype=torch.int64, device=device)
                    .repeat_interleave(page_size)
                    * page_size
                    + positions.to(torch.int64) % page_size
                )
                self.assertTrue(torch.equal(expected, tables.slot_mappings[0]))
                self.assertEqual(tokens, int(local_seq_lens.item()))

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    producer()
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(expected, tables.slot_mappings[0]))
                self.assertEqual(tokens, int(local_seq_lens.item()))
                results.append(tables.slot_mappings[0].clone())

        self.assertTrue(torch.equal(results[0], results[1]))
        self.assertTrue(torch.equal(results[1], results[2]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_rejects_conflicting_ownership_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            BlockTables(
                block_sizes=[64],
                max_num_reqs=1,
                max_num_batched_tokens=64,
                max_num_blocks_per_group=[1],
                device=torch.device("cuda:0"),
                kernel_block_sizes=[64],
                cp_size=3,
                cp_rank=0,
                cp_interleave=1,
                rank_projected_groups=[True],
                full_history_groups=[True],
            )


if __name__ == "__main__":
    unittest.main()
