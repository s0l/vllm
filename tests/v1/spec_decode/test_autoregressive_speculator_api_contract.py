import ast
import inspect
import textwrap

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
    prepare_decode_inputs,
)


def test_prepare_decode_inputs_call_matches_current_signature() -> None:
    source = textwrap.dedent(inspect.getsource(AutoRegressiveSpeculator.propose))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "prepare_decode_inputs"
    ]
    assert len(calls) == 1
    call = calls[0]

    inspect.signature(prepare_decode_inputs).bind(
        *(object() for _ in call.args),
        **{keyword.arg: object() for keyword in call.keywords if keyword.arg},
    )
    assert ast.unparse(call.args[4]) == "self.sample_src_positions"
