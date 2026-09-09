# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import unittest
from unittest.mock import patch

import torch

from vllm.distributed.device_communicators.tp3_exact_reduce import (
    exact_direct_pair_owner_reduce,
    exact_head_owner_reduce_scatter,
    fixed_tp3_sum,
)
from vllm.distributed.parallel_state import _tp3_unified_exact_backend


class TestTP3UnifiedExactReduce(unittest.TestCase):
    def test_fixed_sum_contract(self) -> None:
        values = torch.tensor(
            [
                [[1.0, 0.125, -7.0]],
                [[0.5, -0.25, 2.0]],
                [[-0.125, 0.5, 4.0]],
            ],
            dtype=torch.bfloat16,
        )
        expected = (
            (values[0].float() + values[1].float()) + values[2].float()
        ).bfloat16()
        self.assertTrue(torch.equal(fixed_tp3_sum(values), expected))

    def test_fixed_sum_rejects_non_tp3(self) -> None:
        with self.assertRaisesRegex(ValueError, "three TP ranks"):
            fixed_tp3_sum(torch.zeros((2, 1, 4), dtype=torch.bfloat16))

    def test_head_owner_shape_and_dtype_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "heads divisible by 3"):
            exact_head_owner_reduce_scatter(
                torch.zeros((2, 8, 4), dtype=torch.bfloat16), object()
            )
        with self.assertRaisesRegex(ValueError, "requires BF16"):
            exact_head_owner_reduce_scatter(
                torch.zeros((2, 9, 4), dtype=torch.float32), object()
            )

    def test_head_owner_packs_owner_major_and_sums_fixed_order(self) -> None:
        local = torch.arange(2 * 9 * 2, dtype=torch.bfloat16).view(2, 9, 2)
        sources = torch.stack((local, local + 1, local - 2))
        captured_send = None

        def fake_all_to_all(receive, send, **_kwargs) -> None:
            nonlocal captured_send
            captured_send = send.clone()
            owner = 1
            received = sources[:, :, owner * 3 : (owner + 1) * 3, :]
            receive.copy_(received.contiguous().view(-1))

        with (
            patch("torch.distributed.get_world_size", return_value=3),
            patch("torch.distributed.all_to_all_single", side_effect=fake_all_to_all),
        ):
            actual = exact_head_owner_reduce_scatter(local, object())

        expected_send = local.view(2, 3, 3, 2).movedim(1, 0).contiguous().view(-1)
        self.assertTrue(torch.equal(captured_send, expected_send))
        expected = fixed_tp3_sum(sources[:, :, 3:6, :])
        self.assertTrue(torch.equal(actual, expected))

    def test_direct_pair_disseminates_only_completed_owner_halves(self) -> None:
        local = torch.arange(2 * 8, dtype=torch.bfloat16).view(2, 8)
        sources = torch.stack((local, local + 1, local - 2))
        calls = 0

        def fake_all_to_all(receive, send, **kwargs) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                self.assertEqual(kwargs["input_split_sizes"], [8, 8, 0])
                receive.copy_(sources[:, :, :4].contiguous().view(-1))
                return
            owned_left = fixed_tp3_sum(sources[:, :, :4]).reshape(-1)
            owned_right = fixed_tp3_sum(sources[:, :, 4:]).reshape(-1)
            self.assertEqual(kwargs["output_split_sizes"], [8, 8, 0])
            self.assertEqual(kwargs["input_split_sizes"], [8, 8, 8])
            self.assertTrue(torch.equal(send, owned_left.repeat(3)))
            receive.copy_(torch.cat((owned_left, owned_right)))

        with (
            patch("torch.distributed.get_world_size", return_value=3),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.all_to_all_single", side_effect=fake_all_to_all),
        ):
            actual = exact_direct_pair_owner_reduce(local, object())

        self.assertEqual(calls, 2)
        self.assertTrue(torch.equal(actual, fixed_tp3_sum(sources)))

    def test_auto_backend_crossover(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AG2_VLLM_TP3_UNIFIED_EXACT_BACKEND", None)
            os.environ.pop("AG2_VLLM_TP3_EXACT_OWNER_MIN_ROWS", None)
            self.assertEqual(_tp3_unified_exact_backend(1), "all_gather_fused")
            self.assertEqual(_tp3_unified_exact_backend(23), "all_gather_fused")
            self.assertEqual(_tp3_unified_exact_backend(24), "weighted_owner_992")

    def test_explicit_backend_and_invalid_values(self) -> None:
        with patch.dict(
            os.environ,
            {"AG2_VLLM_TP3_UNIFIED_EXACT_BACKEND": "all_gather"},
        ):
            self.assertEqual(_tp3_unified_exact_backend(128), "all_gather")
        with (
            patch.dict(
                os.environ,
                {"AG2_VLLM_TP3_UNIFIED_EXACT_BACKEND": "bad"},
            ),
            self.assertRaisesRegex(ValueError, "unknown TP3"),
        ):
            _tp3_unified_exact_backend(1)

        with (
            patch.dict(
                os.environ,
                {
                    "AG2_VLLM_TP3_UNIFIED_EXACT_BACKEND": "auto",
                    "AG2_VLLM_TP3_EXACT_OWNER_MIN_ROWS": "0",
                },
            ),
            self.assertRaisesRegex(ValueError, "must be positive"),
        ):
            _tp3_unified_exact_backend(1)


if __name__ == "__main__":
    unittest.main()
