from vllm.model_executor.kernels.linear.nvfp4.humming import (
    _use_narrow_pad64,
    _use_stable_k64,
)


def test_stable_k64_is_limited_to_proven_language_shapes():
    accepted = {
        (5632, 5120),
        (5120, 2048),
        (6144, 5120),
        (11648, 5120),
        (5120, 5824),
    }
    assert all(_use_stable_k64(*shape) for shape in accepted)
    assert not _use_stable_k64(32, 5120)
    assert not _use_stable_k64(5120, 5120)
    assert not _use_stable_k64(5632, 4096)


def test_narrow_padding_is_only_for_tp3_gdn_ba():
    assert _use_narrow_pad64(32, 5120)
    assert not _use_narrow_pad64(36, 5120)
    assert not _use_narrow_pad64(32, 4096)
    assert not _use_narrow_pad64(5120, 2048)
