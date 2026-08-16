# SPDX-License-Identifier: Apache-2.0

import unittest

from vllm.v1.attention.backends.gdn_attn import (
    _gdn_decode_cudagraph_capacity,
)


class TestGDNFullDecodeCapCapacity(unittest.TestCase):
    def test_owner_capacity_matches_full_qlen4_boundary(self) -> None:
        self.assertEqual(_gdn_decode_cudagraph_capacity(23, 3, 64, True), 92)

    def test_disabled_control_preserves_generic_m64_ceiling(self) -> None:
        self.assertEqual(_gdn_decode_cudagraph_capacity(23, 3, 64, False), 64)

    def test_resident_cap_mutation_moves_capacity(self) -> None:
        self.assertEqual(_gdn_decode_cudagraph_capacity(24, 3, 64, True), 96)


if __name__ == "__main__":
    unittest.main()
