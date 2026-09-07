# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_exact_full_attention_reduces_rank_local_projection() -> None:
    """The exact path must not return RowParallelLinear's local partial."""
    source = Path("vllm/model_executor/models/qwen3_next.py").read_text()
    tree = ast.parse(source)
    attention = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3NextAttention"
    )
    forward = next(
        node
        for node in attention.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    guarded_calls = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != (
            "self._ag2_tp3_unified_exact_reduce and (not return_tp_partial)"
        ):
            continue
        guarded_calls.extend(
            child
            for statement in node.body
            for child in ast.walk(statement)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "tensor_model_parallel_unified_exact_all_reduce"
        )
    assert len(guarded_calls) == 1
