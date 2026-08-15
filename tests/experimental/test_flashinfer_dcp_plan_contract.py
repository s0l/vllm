# SPDX-License-Identifier: Apache-2.0

import unittest

from vllm.v1.attention.backends.flashinfer import (
    BatchDCPPrefillWrapper,
    BatchDCPPseudoPrefillWrapper,
)


class TestFlashInferDCPPlanContract(unittest.TestCase):
    @staticmethod
    def _planned(wrapper_type):
        wrapper = object.__new__(wrapper_type)
        wrapper._ag2_plan_ready = True
        wrapper._ag2_window_left = -1
        wrapper._ag2_logits_soft_cap = 0.0
        wrapper._ag2_sm_scale = 0.125
        return wrapper

    def test_owned_plan_state_accepts_matching_runtime(self) -> None:
        for wrapper_type in (BatchDCPPrefillWrapper, BatchDCPPseudoPrefillWrapper):
            with self.subTest(wrapper_type=wrapper_type.__name__):
                self._planned(wrapper_type).assert_plan_contract(
                    window_left=-1,
                    logits_soft_cap=0.0,
                    sm_scale=0.125,
                )

    def test_unplanned_or_stale_state_fails_closed(self) -> None:
        for wrapper_type in (BatchDCPPrefillWrapper, BatchDCPPseudoPrefillWrapper):
            with self.subTest(wrapper_type=wrapper_type.__name__):
                wrapper = self._planned(wrapper_type)
                wrapper._ag2_plan_ready = False
                with self.assertRaises(AssertionError):
                    wrapper.assert_plan_contract(
                        window_left=-1,
                        logits_soft_cap=0.0,
                        sm_scale=0.125,
                    )

                wrapper._ag2_plan_ready = True
                with self.assertRaises(AssertionError):
                    wrapper.assert_plan_contract(
                        window_left=-1,
                        logits_soft_cap=0.0,
                        sm_scale=0.25,
                    )


if __name__ == "__main__":
    unittest.main()
