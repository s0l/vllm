# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mutable QSA storage identity, typed views and the real AOT operator ABI."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import (
    qsa_cache_operands,
    qsa_cache_storage,
    qsa_cache_view,
)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16, torch.float32])
def test_storage_carrier_preserves_padding_offsets_and_lifetime(dtype):
    arena = torch.zeros(8192, dtype=torch.uint8)
    view = arena.view(dtype).as_strided((3, 2, 4), (256, 8, 1), 16)
    carrier = qsa_cache_storage(view)
    assert carrier is qsa_cache_storage(arena[7:12])
    assert carrier.numel() == arena.numel()
    rebuilt = qsa_cache_view(carrier, view)
    assert rebuilt.data_ptr() == view.data_ptr()
    assert rebuilt.stride() == view.stride() and rebuilt.dtype == view.dtype
    rebuilt.fill_(2)
    assert torch.equal(rebuilt.view(torch.uint8), view.view(torch.uint8))
    assert not arena[-1024:].count_nonzero()
    replacement = torch.zeros_like(carrier)
    qsa_cache_view(replacement, view).fill_(3)
    assert not torch.equal(replacement, arena)
    with pytest.raises(ValueError, match="bound storage extent"):
        qsa_cache_view(carrier[:-1], view)
    ref = weakref.ref(carrier)
    del carrier, rebuilt
    gc.collect()
    assert ref() is None
    assert qsa_cache_storage(view).data_ptr() == arena.data_ptr()


@pytest.mark.parametrize("auto_v2", [False, True])
@torch.inference_mode()
def test_compiled_qsa_operator_keeps_one_live_storage_across_owners(
    monkeypatch, auto_v2
):
    import vllm.models.qwen4_exp.nvidia.qsa as qsa

    arena = torch.zeros(8 * 4096, dtype=torch.uint8)
    typed = [
        arena.as_strided((8, 2, 8, 16), (4096, 16, 32, 1)).view(torch.float8_e4m3fn),
        arena.view(torch.bfloat16).as_strided((8, 8, 1, 4), (2048, 4, 4, 1), 16),
        arena.view(torch.bfloat16).as_strided((8, 16, 1, 4), (2048, 4, 4, 1), 128),
    ]
    carriers = [qsa_cache_storage(t) for t in typed]
    assert carriers[0] is carriers[1] is carriers[2]
    calls = []

    def owner(index):
        def consume(projected, positions, query, key, value, caches, selection, output):
            calls.append([t.untyped_storage().data_ptr() for t in caches])
            for expected, actual in zip(typed, caches, strict=True):
                assert actual.dtype == expected.dtype
                assert (
                    actual.shape == expected.shape
                    and actual.stride() == expected.stride()
                )
                assert actual.storage_offset() == expected.storage_offset()
            ids = [1 + index * 3, 2 + index * 3, 3 + index * 3]
            for delta, (cache, block) in enumerate(zip(caches, ids, strict=True)):
                cache[block].copy_((query[0] + delta).to(cache.dtype))
            output.copy_(query + caches[-1][ids[-1], 0, 0, 0].float())
            selection.copy_(positions)

        return SimpleNamespace(
            kv_cache=typed[0],
            indexer=SimpleNamespace(
                raw_key_cache=SimpleNamespace(kv_cache=typed[1]),
                compressed_key_cache=SimpleNamespace(kv_cache=typed[2]),
            ),
            _run_fp8_qsa=consume,
        )

    context = SimpleNamespace(no_compile_layers={str(i): owner(i) for i in range(2)})
    monkeypatch.setattr(qsa, "get_forward_context", lambda: context)
    lib = torch.library.Library("vllm", "IMPL", "CPU")
    lib.impl("qsa_fp8_owner", qsa._qsa_fp8_owner)

    def invoke(x, operands):
        unique, indices = qsa_cache_operands(operands)
        pos = torch.ones(1, dtype=torch.int64)
        selection = torch.empty_like(pos)
        for i in range(2):
            output = torch.empty_like(x)
            torch.ops.vllm.qsa_fp8_owner(
                x, pos, x, x, x, unique, indices, selection, output, str(i)
            )
            x = output
        return output, selection

    try:
        x = torch.ones(1)
        options = dict(enable_auto_functionalized_v2=auto_v2)
        with pytest.raises(
            Exception, match="input mutations on views with different dtypes"
        ):
            torch.compile(invoke, fullgraph=True, options=options)(x, typed)
        compiled = torch.compile(invoke, fullgraph=True, options=options)
        for value in (1.0, 3.0, 1.0):
            x.fill_(value)
            arena.zero_()
            expected = invoke(x, carriers)
            state = arena.clone()
            assert torch.equal(expected[0], 4 * x + 6)
            arena.zero_()
            start = len(calls)
            actual = compiled(x, carriers)
            assert calls[start:] == [[arena.data_ptr()] * 3] * 2
            assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
            assert torch.equal(arena, state)
            assert not arena[:4096].count_nonzero()
            assert not arena[-4096:].count_nonzero()
    finally:
        lib._destroy()
