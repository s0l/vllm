# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path


def test_direct_prefill_output_reuse_excludes_speculative_batches() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    tree = ast.parse(source_path.read_text())
    guarded_reuse = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "spec_sequence_masks is None":
            continue
        guarded_reuse = any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "prefill_output"
                for target in child.targets
            )
            and "core_attn_out[" in ast.unparse(child.value)
            for child in ast.walk(node)
        )
        if guarded_reuse:
            break

    assert guarded_reuse, (
        "direct GDN output reuse would alias the later speculative index_copy_"
    )
