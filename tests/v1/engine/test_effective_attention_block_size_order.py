# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path


def test_scheduler_exists_before_effective_attention_block_size_init() -> None:
    source_path = Path(__file__).parents[3] / "vllm/v1/engine/core.py"
    tree = ast.parse(source_path.read_text())
    engine_core = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineCore"
    )
    init = next(
        node
        for node in engine_core.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    statements = [ast.unparse(node) for node in init.body]
    scheduler_assignment = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("self.scheduler =")
    )
    effective_size_init = statements.index(
        "self._initialize_effective_attention_block_size()"
    )

    assert scheduler_assignment < effective_size_init
