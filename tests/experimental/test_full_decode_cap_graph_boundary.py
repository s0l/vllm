# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


class TestFullDecodeCapGraphBoundary(unittest.TestCase):
    @staticmethod
    def _manager(cap: int, enabled: bool) -> CudaGraphManager:
        manager = object.__new__(CudaGraphManager)
        manager.compilation_config = MagicMock(
            cudagraph_capture_sizes=[1, 4, 8, 16, 32, 64],
            max_cudagraph_capture_size=64,
        )
        manager.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        manager.decode_query_len = 4
        manager.full_decode_query_lens = {1, 4}
        manager.full_decode_cap_query_lens = {4} if enabled else None
        manager.expand_dynamic_decode_query_lens = False
        manager.varlen_decode = False
        manager.defer_startup_graphs = False
        manager.dynamic_piecewise_coverage_sizes = ()
        manager.dynamic_piecewise_capture_sizes = ()
        manager.dynamic_piecewise_safety_sizes = ()
        manager.dynamic_full_capture_sizes = ()
        manager.max_num_reqs = 32
        manager.max_uniform_decode_reqs = cap
        manager.lora_capture_cases = [0]
        manager.vllm_config = MagicMock(spec=VllmConfig)
        manager.vllm_config.speculative_config = None
        manager._candidates = {}
        manager._capture_descs = {}
        manager._dynamic_graph_entries = {}
        manager._init_candidates()
        return manager

    def test_only_declared_full_query_len_gets_cap_boundary(self) -> None:
        manager = self._manager(23, enabled=True)
        full = manager._capture_descs[CUDAGraphMode.FULL]
        self.assertIn(
            (92, 23),
            {
                (desc.num_tokens, desc.num_reqs)
                for desc in full
                if desc.uniform_token_count == 4
            },
        )
        self.assertNotIn(
            92,
            {desc.num_tokens for desc in full if desc.uniform_token_count == 1},
        )
        self.assertEqual(manager.max_capture_tokens, 92)
        self.assertNotIn(
            92,
            {
                desc.num_tokens
                for desc in manager._capture_descs[CUDAGraphMode.PIECEWISE]
            },
        )

    def test_disabled_control_preserves_old_m64_ceiling(self) -> None:
        manager = self._manager(23, enabled=False)
        full = manager._capture_descs[CUDAGraphMode.FULL]
        qlen4_tokens = {
            desc.num_tokens for desc in full if desc.uniform_token_count == 4
        }
        self.assertEqual(max(qlen4_tokens), 64)
        self.assertEqual(manager.max_capture_tokens, 64)

    def test_resident_cap_mutation_moves_boundary(self) -> None:
        manager = self._manager(24, enabled=True)
        full = manager._capture_descs[CUDAGraphMode.FULL]
        self.assertIn(
            (96, 24),
            {
                (desc.num_tokens, desc.num_reqs)
                for desc in full
                if desc.uniform_token_count == 4
            },
        )
        self.assertEqual(manager.max_capture_tokens, 96)


if __name__ == "__main__":
    unittest.main()
