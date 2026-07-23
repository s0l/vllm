# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.utils import ceil_to_multiple


def test_qwen3_vision_tp3_padding_covers_heads_and_mlp():
    assert ceil_to_multiple(16, 3) == 18
    assert ceil_to_multiple(4304, 3) == 4305
    assert ceil_to_multiple(4608, 3) == 4608
