# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm.v1.attention.backends.flashinfer import (
    _dcp_pseudo_block_table_capacity,
)


class TestDCPPseudoCapacity(unittest.TestCase):
    def test_accepted_exp9_full_context_capacity(self) -> None:
        capacity = _dcp_pseudo_block_table_capacity(262144, 4096, 64, 3)
        self.assertEqual(capacity, 1434)
        self.assertGreaterEqual(capacity, 1404)

    def test_scheduler_tail_is_not_divided_by_dcp(self) -> None:
        # This is the historical failing geometry: the worker table has width
        # 702 while the old folded-token formula allocated only 688 entries.
        self.assertEqual(
            _dcp_pseudo_block_table_capacity(131072, 1024, 64, 3),
            702,
        )

    def test_invalid_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _dcp_pseudo_block_table_capacity(262144, 4096, 0, 3)


if __name__ == "__main__":
    unittest.main()
