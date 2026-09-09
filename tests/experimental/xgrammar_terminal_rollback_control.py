#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic xgrammar terminal/rollback control for Qwen tool tags.

Run inside the vLLM runtime image.  This deliberately exercises the exact
validate -> rollback -> accept sequence used by speculative decoding without
loading model weights or using a GPU.
"""

from __future__ import annotations

import json

import xgrammar as xgr
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

MODEL = "crushleorey/Qwopus3.6-27B-v2-NVFP4"
POST_TERMINAL_TOKENS = (271, 198)  # "\n\n", "\n"
SPEC = json.dumps(
    {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "<tool_call>\n<function=websearch__web_search_tool>\n",
            "content": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "regions": {"type": "string", "default": ""},
                        "max_urls": {"type": "integer", "default": 5},
                    },
                    "additionalProperties": False,
                },
                "style": "qwen_xml",
                "any_order": False,
            },
            "end": "\n</function>\n</tool_call>",
        },
    }
)
VALID_TEXT = (
    "<tool_call>\n"
    "<function=websearch__web_search_tool>\n"
    "<parameter=query>CUDA documentation</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def make_grammar(tokenizer, max_rollback_tokens: int = 2):
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=len(tokenizer)
    )
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)
    compiled = compiler.compile_structural_tag(SPEC)
    return XgrammarGrammar(
        matcher=xgr.GrammarMatcher(compiled, max_rollback_tokens=max_rollback_tokens),
        vocab_size=len(tokenizer),
        ctx=compiled,
    )


def accept_all(matcher, tokens: list[int]) -> tuple[int, int | None]:
    for index, token in enumerate(tokens):
        if not matcher.accept_token(token):
            return index, token
    return len(tokens), None


def check_reasoning_boundary(tokenizer) -> dict:
    prompt = tokenizer.encode("{", add_special_tokens=False)
    config = VllmConfig(
        model_config=ModelConfig(model=MODEL, tokenizer=MODEL),
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=2),
    )
    manager = StructuredOutputManager(config)
    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type":"object"}'),
    )
    sampling_params.structured_outputs._backend = "xgrammar"
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)
    request = Request(
        "reasoning-boundary",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    manager.grammar_init(request)
    structured = request.structured_output_request
    assert structured is not None
    while not structured._check_grammar_completion():
        pass
    grammar = structured.grammar
    assert isinstance(grammar, XgrammarGrammar)
    assert grammar.accept_tokens(request.request_id, prompt)

    marker = tokenizer.encode("\n", add_special_tokens=False)[0]

    class MarkerReasoner:
        def is_reasoning_end(self, input_ids):
            return marker in list(input_ids)

        def is_reasoning_end_streaming(self, input_ids, delta_ids):
            return marker in list(delta_ids)

    manager.reasoner_cls = MarkerReasoner
    structured.reasoner = MarkerReasoner()
    structured.reasoning_ended = False

    pre = tokenizer.encode(" ", add_special_tokens=False)[0]
    invalid = tokenizer.encode("z", add_special_tokens=False)[0]
    valid_if_skipped = tokenizer.encode('"key"', add_special_tokens=False)[0]
    assert grammar.validate_tokens([invalid]) == []
    assert grammar.validate_tokens([valid_if_skipped]) == [valid_if_skipped]

    accepted_calls: list[list[int]] = []
    original_accept = grammar.accept_tokens

    def record_accept(request_id: str, tokens: list[int]) -> bool:
        accepted_calls.append(list(tokens))
        return original_accept(request_id, tokens)

    grammar.accept_tokens = record_accept  # type: ignore[method-assign]
    drafts = [pre, marker, invalid, valid_if_skipped]
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1
    assert accepted_calls == [], accepted_calls
    assert not grammar.is_terminated()
    return {
        "drafts": [[token, tokenizer.decode([token])] for token in drafts],
        "mutating_accept_calls": accepted_calls,
    }


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    text_tokens = tokenizer.encode(VALID_TEXT, add_special_tokens=False)
    assert tokenizer.decode(text_tokens) == VALID_TEXT
    assert text_tokens[-1] == tokenizer.convert_tokens_to_ids("</tool_call>")
    valid_tokens = [*text_tokens, tokenizer.eos_token_id]

    baseline = make_grammar(tokenizer)
    accepted, rejected = accept_all(baseline.matcher, valid_tokens)
    assert accepted == len(valid_tokens), (accepted, rejected)
    assert baseline.matcher.is_terminated()

    # Reproduce vLLM validate_tokens(): a speculative tail reaches terminal
    # while the raw draft window still contains a post-terminal token. The
    # validator must return the terminal prefix without probing the matcher
    # after it has stopped, then restore the exact pre-validation state.
    for post_terminal in POST_TERMINAL_TOKENS:
        grammar = make_grammar(tokenizer)
        matcher = grammar.matcher
        prefix = valid_tokens[:-2]
        tail = valid_tokens[-2:]
        assert accept_all(matcher, prefix) == (len(prefix), None)
        assert not matcher.is_terminated()

        accepted_tail = grammar.validate_tokens([*tail, post_terminal])
        assert accepted_tail == tail, (
            post_terminal,
            accepted_tail,
        )
        assert not matcher.is_terminated(), (
            "rollback left matcher terminal",
            post_terminal,
        )
        replayed, replay_rejected = accept_all(matcher, tail)
        assert replayed == len(tail), (
            "valid replay rejected after terminal rollback",
            post_terminal,
            replayed,
            replay_rejected,
        )
        assert matcher.is_terminated()

    print(
        json.dumps(
            {
                "xgrammar_control": "passed",
                "valid_token_count": len(valid_tokens),
                "tail": [
                    [token, tokenizer.decode([token])] for token in valid_tokens[-4:]
                ],
                "post_terminal": [
                    [token, tokenizer.decode([token])] for token in POST_TERMINAL_TOKENS
                ],
                "reasoning_boundary": check_reasoning_boundary(tokenizer),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
