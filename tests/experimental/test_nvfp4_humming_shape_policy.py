from vllm.model_executor.kernels.linear.nvfp4.humming import _use_narrow_pad64


def test_narrow_padding_is_only_for_tp3_gdn_ba():
    assert _use_narrow_pad64(32, 5120)
    assert not _use_narrow_pad64(36, 5120)
    assert not _use_narrow_pad64(32, 4096)
    assert not _use_narrow_pad64(5120, 2048)
