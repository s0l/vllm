# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.v1.attention.backends.flashinfer import _resolve_decode_split_plan


def test_spec_target_policy_preserves_single_token_full_graph() -> None:
    assert _resolve_decode_split_plan(
        num_decode_tokens=5,
        num_decodes=5,
        fixed_split_size=1,
        disable_split_kv=False,
        spec_target_only=True,
    ) == (-1, False, False)


def test_spec_target_policy_uses_replannable_wrapper_for_qlen4() -> None:
    assert _resolve_decode_split_plan(
        num_decode_tokens=20,
        num_decodes=5,
        fixed_split_size=1,
        disable_split_kv=False,
        spec_target_only=True,
    ) == (1, False, True)


def test_global_batch_invariant_policy_keeps_existing_graph_contract() -> None:
    assert _resolve_decode_split_plan(
        num_decode_tokens=5,
        num_decodes=5,
        fixed_split_size=2048,
        disable_split_kv=True,
        spec_target_only=False,
    ) == (2048, True, False)


def test_qlen1_policy_is_narrower_than_spec_target_policy() -> None:
    assert _resolve_decode_split_plan(
        num_decode_tokens=5,
        num_decodes=5,
        fixed_split_size=4096,
        disable_split_kv=False,
        spec_target_only=True,
        qlen1_fixed_split_size=2048,
        qlen1_disable_split_kv=True,
    ) == (2048, True, False)
    assert _resolve_decode_split_plan(
        num_decode_tokens=20,
        num_decodes=5,
        fixed_split_size=4096,
        disable_split_kv=False,
        spec_target_only=True,
        qlen1_fixed_split_size=2048,
        qlen1_disable_split_kv=True,
    ) == (4096, False, True)


@pytest.mark.parametrize(
    ("num_decode_tokens", "num_decodes"),
    [(1, 0), (5, 2)],
)
def test_decode_split_policy_rejects_nonuniform_geometry(
    num_decode_tokens: int,
    num_decodes: int,
) -> None:
    with pytest.raises(ValueError, match="uniform query length"):
        _resolve_decode_split_plan(
            num_decode_tokens=num_decode_tokens,
            num_decodes=num_decodes,
            fixed_split_size=1,
            disable_split_kv=False,
            spec_target_only=True,
        )
