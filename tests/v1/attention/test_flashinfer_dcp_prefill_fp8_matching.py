from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder


def _builder(*, target: bool, prefill: bool, use_dcp: bool = True):
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder._dcp_match_fp8_new_tokens = target
    builder._dcp_prefill_match_fp8_new_tokens = prefill
    builder.use_dcp = use_dcp
    return builder


def test_prefill_matching_only_selects_native_causal_dcp_prefill():
    builder = _builder(target=False, prefill=True)
    assert builder._should_match_fp8_new_tokens(
        causal=True,
        num_prefills=2,
        use_dcp_pseudo_decode=False,
        uniform_target_decode=False,
    )
    assert not builder._should_match_fp8_new_tokens(
        causal=False,
        num_prefills=2,
        use_dcp_pseudo_decode=False,
        uniform_target_decode=False,
    )
    assert not builder._should_match_fp8_new_tokens(
        causal=True,
        num_prefills=0,
        use_dcp_pseudo_decode=False,
        uniform_target_decode=False,
    )
    assert not builder._should_match_fp8_new_tokens(
        causal=True,
        num_prefills=2,
        use_dcp_pseudo_decode=True,
        uniform_target_decode=False,
    )
    builder.use_dcp = False
    assert not builder._should_match_fp8_new_tokens(
        causal=True,
        num_prefills=2,
        use_dcp_pseudo_decode=False,
        uniform_target_decode=False,
    )


def test_target_matching_contract_is_unchanged():
    builder = _builder(target=True, prefill=False)
    assert builder._should_match_fp8_new_tokens(
        causal=True,
        num_prefills=1,
        use_dcp_pseudo_decode=False,
        uniform_target_decode=True,
    )
    assert not builder._should_match_fp8_new_tokens(
        causal=True,
        num_prefills=1,
        use_dcp_pseudo_decode=False,
        uniform_target_decode=False,
    )
