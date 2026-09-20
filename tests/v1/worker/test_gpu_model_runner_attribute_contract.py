# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
import textwrap

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_gpu_model_runner_reads_only_initialized_or_inherited_attributes() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(GPUModelRunner)))
    cls = tree.body[0]
    assert isinstance(cls, ast.ClassDef)

    reads: set[str] = set()
    writes: set[str] = set()
    methods = {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            target = writes if isinstance(node.ctx, (ast.Store, ast.Del)) else reads
            target.add(node.attr)

    inherited_lora_attributes = {
        "_set_active_loras",
        "load_lora_model",
        "lora_manager",
        "maybe_dummy_run_with_lora",
        "maybe_setup_dummy_loras",
    }
    assert reads - writes - methods == inherited_lora_attributes
