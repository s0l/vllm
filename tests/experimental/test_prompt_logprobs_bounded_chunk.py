# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.sample import prompt_logprob


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Triton CUDA")
def test_prompt_logprobs_chunk_size_preserves_result(monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(212)
    num_tokens = 1103
    vocab_size = 257
    logits = torch.randn(num_tokens, vocab_size, generator=generator, device="cuda")
    prompt_token_ids = torch.randint(
        0, vocab_size, (num_tokens,), generator=generator, device="cuda"
    )
    row_ids = torch.arange(num_tokens, device="cuda").unsqueeze(1)

    def logits_fn(hidden_states: torch.Tensor) -> torch.Tensor:
        return logits[hidden_states[:, 0]]

    def compute(chunk_size: int):
        monkeypatch.setattr(prompt_logprob, "PROMPT_LOGPROBS_CHUNK_SIZE", chunk_size)
        return prompt_logprob.compute_prompt_logprobs_with_chunking(
            prompt_token_ids,
            row_ids,
            logits_fn,
            num_prompt_logprobs=1,
        )

    reference = compute(1024)
    bounded = compute(256)

    torch.testing.assert_close(bounded[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(bounded[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(bounded[2], reference[2], rtol=0, atol=0)
