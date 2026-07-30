from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor


class ThinkingBudgetValidationTests(unittest.TestCase):
    def processor(self, *, use_v2: bool, reasoning_enabled: bool) -> InputProcessor:
        processor = InputProcessor.__new__(InputProcessor)
        processor.use_v2_model_runner = use_v2
        processor.model_config = MagicMock()
        processor.speculative_config = None
        processor.structured_outputs_config = None
        processor.renderer = SimpleNamespace(tokenizer=None)
        processor.vllm_config = SimpleNamespace(
            reasoning_config=(
                SimpleNamespace(enabled=True) if reasoning_enabled else None
            )
        )
        return processor

    def validate(self, processor: InputProcessor, params: SamplingParams) -> None:
        with patch.object(SamplingParams, "verify"):
            processor._validate_params(params, ("generate",))

    def test_v2_preserves_zero_budget(self):
        params = SamplingParams(thinking_token_budget=0)
        self.validate(
            self.processor(use_v2=True, reasoning_enabled=True),
            params,
        )
        self.assertEqual(params.thinking_token_budget, 0)

    def test_v2_rejects_positive_budget_without_reasoning_config(self):
        params = SamplingParams(thinking_token_budget=8192)
        with self.assertRaisesRegex(ValueError, "reasoning_config"):
            self.validate(
                self.processor(use_v2=True, reasoning_enabled=False),
                params,
            )

    def test_v2_absent_budget_remains_absent(self):
        params = SamplingParams()
        self.validate(
            self.processor(use_v2=True, reasoning_enabled=False),
            params,
        )
        self.assertIsNone(params.thinking_token_budget)

    def test_v1_preserves_supported_budget(self):
        params = SamplingParams(thinking_token_budget=2048)
        self.validate(
            self.processor(use_v2=False, reasoning_enabled=True),
            params,
        )
        self.assertEqual(params.thinking_token_budget, 2048)

    def test_v1_still_rejects_budget_without_reasoning_config(self):
        params = SamplingParams(thinking_token_budget=2048)
        with self.assertRaisesRegex(ValueError, "reasoning_config"):
            self.validate(
                self.processor(use_v2=False, reasoning_enabled=False),
                params,
            )


if __name__ == "__main__":
    unittest.main()
