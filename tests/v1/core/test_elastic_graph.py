# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm.v1.core.elastic_graph import (
    ElasticGraphCache,
    ElasticGraphError,
    ElasticPlanKind,
    GraphExecutionPolicy,
    GraphPrice,
    GraphResidency,
    LogicalDispatchKey,
    OwnerGraphExecutionPolicy,
    PhysicalReplayKey,
    ReclaimGroup,
    RuntimeGeneration,
    configured_compiled_piecewise_sizes,
    derive_short_decode_graph_inventory,
    require_plan_consensus,
    resolve_step_physical_keys,
)

GENERATION = RuntimeGeneration("test-generation")


def current_q4_piecewise_policy() -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="batched-causal-q1-v1",
        math_contract="pending-batched-q1-product-math-v1",
        owners=(
            OwnerGraphExecutionPolicy("mtp_decode", (1,), "PIECEWISE"),
            OwnerGraphExecutionPolicy("mtp_prefill", (), "PIECEWISE"),
            OwnerGraphExecutionPolicy(
                "target", (1,), "PIECEWISE", piecewise_query_len_min_tokens=32
            ),
        ),
    )


def approved_q4_full_policy() -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="approved-q4-full-control-v1",
        math_contract="approved-q4-full-math-v1",
        owners=(
            OwnerGraphExecutionPolicy("mtp_decode", (1,), "PIECEWISE"),
            OwnerGraphExecutionPolicy("mtp_prefill", (), "PIECEWISE"),
            OwnerGraphExecutionPolicy("target", (1, 4), "PIECEWISE"),
        ),
    )


def runtime_shape_policy(*owners: OwnerGraphExecutionPolicy) -> GraphExecutionPolicy:
    return GraphExecutionPolicy(
        verifier_contract="runtime-shape-control-v1",
        math_contract="runtime-shape-math-v1",
        owners=tuple(sorted(owners, key=lambda owner: owner.owner)),
    )


def key(
    owner: str,
    mode: str,
    tokens: int,
    physical_x: int,
    *,
    uniform: int | None = None,
) -> PhysicalReplayKey:
    return PhysicalReplayKey(
        logical=LogicalDispatchKey(
            owner=owner,
            mode=mode,
            token_bucket=tokens,
            logical_num_reqs=physical_x if mode == "FULL" else None,
            uniform_query_len=uniform,
        ),
        physical_num_reqs=physical_x,
        generation=GENERATION,
    )


def price(name: str, resident: int = 10, peak: int = 12) -> GraphPrice:
    return GraphPrice(resident, peak, name)


def publish(cache: ElasticGraphCache, graph_key: PhysicalReplayKey, *, pinned=False):
    graph_price = price(graph_key.identity)
    cache.publish_hot(graph_key, graph_price, pinned=pinned)
    cache.install_reclaim_group(
        ReclaimGroup(
            graph_price.reclaim_group,
            (graph_key,),
            graph_price.resident_bytes,
        )
    )


def test_same_piecewise_bucket_keeps_distinct_physical_x_variants() -> None:
    x8 = key("target", "PIECEWISE", 4096, 8)
    x32 = key("target", "PIECEWISE", 4096, 32)
    assert x8.logical == x32.logical
    assert x8 != x32

    cache = ElasticGraphCache(GENERATION)
    publish(cache, x8)
    publish(cache, x32)
    assert cache.entries[x8].hot
    assert cache.entries[x32].hot


def test_recapture_price_keeps_conservative_cold_envelope() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    cache.publish_hot(
        graph_key,
        price("private-pool", resident=44, peak=52),
        pinned=False,
    )

    cache.synchronize_hot(
        (
            (
                graph_key,
                price("private-pool", resident=2, peak=2),
                False,
                1,
            ),
        )
    )

    assert cache.entries[graph_key].price == price(
        "private-pool", resident=44, peak=52
    )
    assert cache._groups["private-pool"] == ReclaimGroup(
        "private-pool", (graph_key,), reclaimable_bytes=1, retained_bytes=1
    )


def test_recapture_rejects_changed_reclaim_identity() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    cache.publish_hot(graph_key, price("first"), pinned=False)

    with pytest.raises(ElasticGraphError, match="reclaim identity changed"):
        cache.publish_hot(graph_key, price("other"), pinned=False)


def test_scheduler_resolver_emits_complete_owner_specific_physical_set() -> None:
    keys = resolve_step_physical_keys((1, 3, 8, 32, 4), GENERATION)
    assert [(item.logical.owner, item.logical.mode) for item in keys] == [
        ("target", "FULL"),
        ("mtp_prefill", "PIECEWISE"),
        ("mtp_decode", "FULL"),
    ]
    assert [item.logical.token_bucket for item in keys] == [32, 32, 8]
    assert all(item.physical_num_reqs == 8 for item in keys)
    assert keys[1].logical.logical_num_reqs is None
    assert keys[1].logical.uniform_query_len is None


def test_exact_piecewise_target_preserves_semantic_uniform_query_len() -> None:
    keys = resolve_step_physical_keys(
        (0, 3, 40, 160, 4),
        GENERATION,
        4096,
        policy=current_q4_piecewise_policy(),
    )

    assert keys[0].logical.owner == "target"
    assert keys[0].logical.mode == "PIECEWISE"
    assert keys[0].logical.uniform_query_len == 4
    assert keys[1].logical.owner == "mtp_prefill"
    assert keys[1].logical.uniform_query_len is None


def test_piecewise_query_len_identity_starts_only_at_consumed_math_threshold() -> None:
    policy = current_q4_piecewise_policy()
    below = resolve_step_physical_keys(
        (0, 3, 1, 4, 4), GENERATION, 4096, policy=policy
    )
    below_mixed = resolve_step_physical_keys(
        (0, 3, 1, 4, 0), GENERATION, 4096, policy=policy
    )
    boundary = resolve_step_physical_keys(
        (0, 3, 8, 32, 4), GENERATION, 4096, policy=policy
    )
    boundary_mixed = resolve_step_physical_keys(
        (0, 3, 8, 32, 0), GENERATION, 4096, policy=policy
    )

    assert below[0].logical.uniform_query_len is None
    assert below == below_mixed
    assert boundary[0].logical.uniform_query_len == 4
    assert boundary != boundary_mixed

    cache = ElasticGraphCache(GENERATION)
    for graph_key in below_mixed:
        publish(cache, graph_key)
    plan = cache.plan(
        "same-math-small-q4",
        below,
        request_bytes=0,
        available_bytes=1_000,
    )
    assert plan.kind == ElasticPlanKind.USER
    assert plan.hot_hits == below
    assert not plan.cold_misses
    assert not plan.victim_keys


def test_current_policy_accepts_q1_full_and_rejects_q4_full() -> None:
    policy = current_q4_piecewise_policy()
    q1 = resolve_step_physical_keys(
        (1, 0, 32, 32, 1), GENERATION, 4096, policy=policy
    )
    assert [(item.logical.owner, item.logical.mode) for item in q1] == [
        ("target", "FULL")
    ]

    with pytest.raises(ElasticGraphError, match="FULL key violates"):
        resolve_step_physical_keys(
            (1, 3, 40, 160, 4), GENERATION, 4096, policy=policy
        )


def test_policy_payload_is_content_addressed_and_corruption_fails_closed() -> None:
    payload = current_q4_piecewise_policy().to_payload()
    assert GraphExecutionPolicy.from_payload(payload).fingerprint == payload[
        "fingerprint"
    ]
    payload["math_contract"] = "silently-mutated"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        GraphExecutionPolicy.from_payload(payload)


def test_policy_rejects_unknown_piecewise_padding_contract() -> None:
    with pytest.raises(ValueError, match="unknown PIECEWISE padding contract"):
        OwnerGraphExecutionPolicy(
            "target",
            (1,),
            "PIECEWISE",
            piecewise_padding_contract="wishful-padding-v1",
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema",), True, "schema must be an integer"),
        (("verifier_contract",), 7, "must be a non-empty string"),
        (("owners", 0, "full_exact_x"), 1, "must be a boolean"),
        (("owners", 0, "full_query_lens"), [1, "4"], "only integers"),
    ],
)
def test_policy_payload_rejects_coerced_types(
    path: tuple[object, ...], value: object, message: str
) -> None:
    payload = current_q4_piecewise_policy().to_payload()
    cursor: object = payload
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=message):
        GraphExecutionPolicy.from_payload(payload)


@pytest.mark.parametrize("x", [1, 3, 16, 32, 40])
def test_policy_specific_q4_authority_matrix(x: int) -> None:
    piecewise = resolve_step_physical_keys(
        (0, 3, x, 4 * x, 4),
        GENERATION,
        4096,
        policy=current_q4_piecewise_policy(),
    )
    assert [(key.logical.owner, key.logical.mode) for key in piecewise] == [
        ("target", "PIECEWISE"),
        ("mtp_prefill", "PIECEWISE"),
        ("mtp_decode", "FULL"),
    ]

    full = resolve_step_physical_keys(
        (1, 3, x, 4 * x, 4),
        GENERATION,
        4096,
        policy=approved_q4_full_policy(),
    )
    assert [(key.logical.owner, key.logical.mode) for key in full] == [
        ("target", "FULL"),
        ("mtp_prefill", "PIECEWISE"),
        ("mtp_decode", "FULL"),
    ]


def test_production_none_policy_cannot_resolve_short_decode() -> None:
    policy = GraphExecutionPolicy(
        verifier_contract="negative-none-v1",
        math_contract="negative-only-v1",
        owners=(OwnerGraphExecutionPolicy("target", (1,), "NONE"),),
    )
    with pytest.raises(ElasticGraphError, match="representation policy"):
        resolve_step_physical_keys(
            (0, 0, 1, 4, 0), GENERATION, 4096, policy=policy
        )


def test_full_target_exact_mtp_wave_is_not_rounded_into_compiled_class() -> None:
    keys = resolve_step_physical_keys(
        (1, 3, 40, 160, 4),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(256, 512, 1024, 2048, 4096),
    )

    assert [
        (item.logical.owner, item.logical.mode, item.logical.token_bucket)
        for item in keys
    ] == [
        ("target", "FULL", 160),
        ("mtp_prefill", "PIECEWISE", 160),
        ("mtp_decode", "FULL", 40),
    ]
    assert all(item.physical_num_reqs == 40 for item in keys)


def test_short_decode_inventory_is_derived_from_k_and_terminal_max_x() -> None:
    inventory = derive_short_decode_graph_inventory(
        max_x=40,
        num_spec_tokens=3,
        generation=GENERATION,
        max_num_batched_tokens=4096,
    )

    assert tuple(inventory) == (1, 2, 4, 8, 16, 32, 40)
    for x, keys in inventory.items():
        assert [item.logical.owner for item in keys] == [
            "target",
            "mtp_prefill",
            "mtp_decode",
        ]
        assert [item.logical.token_bucket for item in keys] == [4 * x, 4 * x, x]
        assert all(item.physical_num_reqs == x for item in keys)
    assert inventory[32][0].logical.token_bucket == 128
    assert inventory[40][0].logical.token_bucket == 160


def test_runtime_policy_resolves_k0_without_draft_owners() -> None:
    policy = runtime_shape_policy(
        OwnerGraphExecutionPolicy(
            "target",
            (1,),
            "PIECEWISE",
            piecewise_padding_contract="exact-batched-decode-m-v1",
            activation="always",
            token_source="step",
            execution_order=0,
        ),
        OwnerGraphExecutionPolicy(
            "draft",
            (4,),
            "NONE",
            activation="speculative",
            token_source="fixed_query",
            fixed_query_len=4,
            execution_order=1,
        ),
    )

    keys = resolve_step_physical_keys(
        (1, 0, 7, 7, 1), GENERATION, 4096, policy=policy
    )

    assert [(item.logical.owner, item.logical.mode, item.logical.token_bucket)
            for item in keys] == [("target", "FULL", 7)]


def test_runtime_policy_resolves_arbitrary_k_from_owner_formulas() -> None:
    policy = runtime_shape_policy(
        OwnerGraphExecutionPolicy(
            "mtp_decode",
            (1,),
            "PIECEWISE",
            activation="speculative",
            token_source="requests",
            execution_order=2,
        ),
        OwnerGraphExecutionPolicy(
            "mtp_prefill",
            (),
            "PIECEWISE",
            piecewise_padding_contract="exact-batched-decode-m-v1",
            activation="speculative",
            token_source="step",
            execution_order=1,
        ),
        OwnerGraphExecutionPolicy(
            "target",
            (1,),
            "PIECEWISE",
            piecewise_padding_contract="exact-batched-decode-m-v1",
            activation="always",
            token_source="step",
            execution_order=0,
        ),
    )

    keys = resolve_step_physical_keys(
        (0, 7, 5, 40, 8), GENERATION, 4096, policy=policy
    )

    assert [
        (item.logical.owner, item.logical.mode, item.logical.token_bucket)
        for item in keys
    ] == [
        ("target", "PIECEWISE", 40),
        ("mtp_prefill", "PIECEWISE", 40),
        ("mtp_decode", "FULL", 5),
    ]


def test_runtime_policy_resolves_dflash_query_geometry_without_mtp_names() -> None:
    policy = runtime_shape_policy(
        OwnerGraphExecutionPolicy(
            "dflash_query",
            (8,),
            "NONE",
            activation="speculative",
            token_source="fixed_query",
            fixed_query_len=8,
            execution_order=1,
        ),
        OwnerGraphExecutionPolicy(
            "target",
            (1,),
            "PIECEWISE",
            piecewise_padding_contract="exact-batched-decode-m-v1",
            activation="always",
            token_source="step",
            execution_order=0,
        ),
    )

    keys = resolve_step_physical_keys(
        (0, 7, 5, 40, 8), GENERATION, 4096, policy=policy
    )

    assert [
        (item.logical.owner, item.logical.mode, item.logical.token_bucket)
        for item in keys
    ] == [
        ("target", "PIECEWISE", 40),
        ("dflash_query", "FULL", 40),
    ]


def test_short_decode_inventory_supports_k0_and_terminal_max_x() -> None:
    inventory = derive_short_decode_graph_inventory(
        max_x=5,
        num_spec_tokens=0,
        generation=GENERATION,
        max_num_batched_tokens=4096,
    )

    assert tuple(inventory) == (1, 2, 4, 5)
    assert [
        keys[0].logical.token_bucket for keys in inventory.values()
    ] == [1, 2, 4, 5]


def test_short_decode_inventory_keeps_exact_graph_owners() -> None:
    compiled = (2, 4, 8, 16, 32, 64, 128, 160, 256, 512, 1024, 2048, 4096)
    inventory = derive_short_decode_graph_inventory(
        max_x=40,
        num_spec_tokens=3,
        generation=GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=compiled,
    )

    assert tuple(inventory) == (1, 2, 4, 8, 16, 32, 40)
    for x, keys in inventory.items():
        assert [(item.logical.owner, item.logical.mode) for item in keys] == [
            ("target", "PIECEWISE"),
            ("mtp_prefill", "PIECEWISE"),
            ("mtp_decode", "FULL"),
        ]
        assert all(item.physical_num_reqs == x for item in keys)


def test_short_decode_inventory_rejects_impossible_carrier() -> None:
    with pytest.raises(ElasticGraphError, match="exceeds the configured token carrier"):
        derive_short_decode_graph_inventory(
            max_x=40,
            num_spec_tokens=3,
            generation=GENERATION,
            max_num_batched_tokens=128,
        )


def test_short_decode_physical_x_selects_bounded_superset_and_terminal() -> None:
    from vllm.v1.core.elastic_graph import select_short_decode_physical_x

    inventory = (1, 2, 4, 8, 16, 32, 40)
    expected = [1, 2, 4, 4, *([8] * 4), *([16] * 8), *([32] * 16), *([40] * 8)]
    assert [
        select_short_decode_physical_x(x, inventory) for x in range(1, 41)
    ] == expected
    with pytest.raises(ElasticGraphError, match="exceeds"):
        select_short_decode_physical_x(41, inventory)


def test_scheduler_resolver_caps_mtp_prefill_piecewise_boundary() -> None:
    keys = resolve_step_physical_keys(
        (1, 3, 40, 3000, 75),
        GENERATION,
        max_num_batched_tokens=3000,
    )
    mtp_prefill = keys[1]
    assert mtp_prefill.logical.mode == "PIECEWISE"
    assert mtp_prefill.logical.token_bucket == 3000
    assert mtp_prefill.physical_num_reqs == 40


def test_compiled_k0_piecewise_carrier_has_no_physical_graph_key() -> None:
    assert resolve_step_physical_keys(
        (0, 0, 40, 4096, 0),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(4096,),
    ) == ()

    adjacent = resolve_step_physical_keys(
        (0, 0, 40, 2048, 0),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(4096,),
    )
    assert [(item.logical.owner, item.logical.token_bucket) for item in adjacent] == [
        ("target", 2048)
    ]


def test_compiled_piecewise_exemption_drops_speculative_piecewise_graphs_only() -> None:
    speculative = resolve_step_physical_keys(
        (0, 3, 40, 4096, 0),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(4096,),
    )
    assert [item.logical.owner for item in speculative] == ["mtp_decode"]
    full = resolve_step_physical_keys(
        (1, 0, 40, 4096, 102),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(4096,),
    )
    assert [item.logical.owner for item in full] == ["target"]


def test_compiled_piecewise_exemption_covers_bounded_unpriced_prefill_sizes() -> None:
    compiled = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    for tokens in compiled:
        assert resolve_step_physical_keys(
            (0, 0, 1, tokens, 0),
            GENERATION,
            max_num_batched_tokens=4096,
            compiled_piecewise_sizes=compiled,
        ) == ()
    assert resolve_step_physical_keys(
        (0, 0, 1, 1, 0),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=compiled,
    )


def test_compiled_mixed_carriers_do_not_remove_exact_k3_decode_graphs() -> None:
    compiled = (2, 4, 8, 16, 32, 64, 128, 160, 256, 512, 1024, 2048, 4096)
    for x in (1, 2, 4, 8, 16, 32, 40):
        exact = resolve_step_physical_keys(
            (0, 3, x, 4 * x, 4),
            GENERATION,
            max_num_batched_tokens=4096,
            compiled_piecewise_sizes=compiled,
        )
        assert [(item.logical.owner, item.logical.mode) for item in exact] == [
            ("target", "PIECEWISE"),
            ("mtp_prefill", "PIECEWISE"),
            ("mtp_decode", "FULL"),
        ]

        # The corresponding mixed/prefill power-of-two carrier has qlen=0 and
        # must retain only the actually consumed MTP-decode Graph owner.
        mixed_tokens = 1 << (4 * x - 1).bit_length()
        mixed_same_geometry = resolve_step_physical_keys(
            (0, 3, x, mixed_tokens, 0),
            GENERATION,
            max_num_batched_tokens=4096,
            compiled_piecewise_sizes=compiled,
        )
        assert [item.logical.owner for item in mixed_same_geometry] == [
            "mtp_decode"
        ]

    # An undeclared exact carrier is Graph-backed for the same reason.
    exact_x3 = resolve_step_physical_keys(
        (0, 3, 3, 12, 4),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=compiled,
    )
    assert [item.logical.owner for item in exact_x3] == [
        "target",
        "mtp_prefill",
        "mtp_decode",
    ]

    mixed = resolve_step_physical_keys(
        (0, 3, 1, 64, 0),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=compiled,
    )
    assert [(item.logical.owner, item.logical.mode) for item in mixed] == [
        ("mtp_decode", "FULL")
    ]


@pytest.mark.parametrize("value", [[4096, 4096], [True], "4096", [8192]])
def test_compiled_piecewise_config_rejects_malformed_or_out_of_range(value) -> None:
    config = SimpleNamespace(
        additional_config={"elastic_compiled_piecewise_sizes": value},
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )
    with pytest.raises(ElasticGraphError, match="elastic_compiled_piecewise_sizes"):
        configured_compiled_piecewise_sizes(config)


def test_full_family_and_piecewise_survive_arbitrary_hot_transitions() -> None:
    full_x1 = key("target", "FULL", 4, 1, uniform=4)
    full_x2 = key("target", "FULL", 8, 2, uniform=4)
    mixed = key("target", "PIECEWISE", 4096, 12)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, full_x1, pinned=True)
    publish(cache, full_x2, pinned=True)
    publish(cache, mixed)

    for index, graph_key in enumerate((full_x1, full_x2, mixed, full_x1)):
        plan = cache.plan(
            f"step-{index}",
            (graph_key,),
            request_bytes=1,
            available_bytes=1,
        )
        assert plan.kind == ElasticPlanKind.USER
        assert not plan.victim_keys
        cache.commit_user(plan)
        cache.release(plan.transaction_id)

    assert all(cache.entries[item].hot for item in (full_x1, full_x2, mixed))


def test_lru_pressure_evicts_only_oldest_unleased_piecewise_group() -> None:
    older = key("target", "PIECEWISE", 2048, 8)
    newer = key("target", "PIECEWISE", 4096, 8)
    pinned = key("target", "FULL", 4, 1, uniform=4)
    cold = key("mtp_prefill", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, older)
    publish(cache, newer)
    publish(cache, pinned, pinned=True)

    touch = cache.plan("touch-newer", (newer,), request_bytes=0, available_bytes=0)
    cache.commit_user(touch)
    cache.release(touch.transaction_id)
    cache.register(cold, price=price("cold", resident=9, peak=12))

    plan = cache.plan(
        "maintenance",
        (cold,),
        request_bytes=8,
        available_bytes=10,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == (older,)
    assert pinned not in plan.victim_keys


def test_two_outstanding_leases_and_cancellation_are_independent() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, graph_key)
    first = cache.plan("first", (graph_key,), request_bytes=0, available_bytes=0)
    second = cache.plan("second", (graph_key,), request_bytes=0, available_bytes=0)
    cache.commit_user(first)
    cache.commit_user(second)
    assert cache.entries[graph_key].leases == frozenset({"first", "second"})

    cache.cancel("first")
    assert cache.entries[graph_key].leases == frozenset({"second"})
    assert not cache.entries[graph_key].reclaimable
    cache.release("second")
    assert cache.entries[graph_key].reclaimable


def test_maintenance_is_separate_from_user_commit_and_failure_is_cold() -> None:
    cold = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    cold_price = price("cold", resident=9, peak=12)
    cache.register(cold, price=cold_price)
    plan = cache.plan("capture", (cold,), request_bytes=0, available_bytes=12)
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    with pytest.raises(ElasticGraphError, match="expected a user"):
        cache.commit_user(plan)

    cache.begin_maintenance(plan)
    assert cache.entries[cold].state == GraphResidency.CAPTURING
    cache.fail_maintenance(plan)
    assert cache.entries[cold].state == GraphResidency.COOLDOWN
    assert not cache.entries[cold].leases


def test_unknown_price_defers_without_mutation_or_eternal_implicit_loan() -> None:
    unknown = key("target", "PIECEWISE", 8192, 9)
    cache = ElasticGraphCache(GENERATION)
    before = dict(cache.entries)
    plan = cache.plan("unknown", (unknown,), request_bytes=0, available_bytes=1 << 40)
    assert plan.kind == ElasticPlanKind.DEFER
    assert plan.defer_reason == "missing_exact_price_or_class_envelope"
    assert dict(cache.entries) == before


def test_class_envelope_funds_first_capture_and_returns_to_exact_price() -> None:
    unknown = key("target", "PIECEWISE", 8192, 9)
    cache = ElasticGraphCache(GENERATION)
    envelope = price("class-envelope", resident=20, peak=30)
    plan = cache.plan(
        "enveloped",
        (unknown,),
        request_bytes=0,
        available_bytes=30,
        class_envelopes={unknown.logical: envelope},
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.capture_loan_bytes == 30
    cache.begin_maintenance(plan)
    exact = price("exact", resident=11, peak=14)
    cache.finish_maintenance(plan, {unknown: exact})
    assert cache.entries[unknown].price == exact


def test_capture_with_sufficient_budget_retains_unrelated_hot_graphs() -> None:
    full = key("target", "FULL", 4, 1, uniform=4)
    old_a = key("target", "PIECEWISE", 2048, 8)
    old_b = key("target", "PIECEWISE", 4096, 8)
    cold = key("target", "PIECEWISE", 8192, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, full, pinned=True)
    publish(cache, old_a)
    publish(cache, old_b)
    plan = cache.plan(
        "calibration",
        (cold,),
        request_bytes=32,
        available_bytes=1 << 20,
        owner_set_capture_envelope_bytes=1 << 20,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == ()
    assert set(cache.entries) == {full, old_a, old_b}


def test_owner_set_endpoint_subtracts_only_proven_reclaim_before_capture() -> None:
    old = key("target", "PIECEWISE", 4096, 8)
    cold = key("target", "PIECEWISE", 8192, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)

    # The 130-byte endpoint includes the old resident set.  Only the physical
    # 10-byte reclaim proof makes the cold endpoint fit the 120-byte boundary.
    plan = cache.plan(
        "reclaim-then-capture",
        (cold,),
        request_bytes=100,
        available_bytes=120,
        owner_set_capture_envelope_bytes=130,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == (old,)
    assert plan.reclaim_bytes == 10
    assert plan.capture_loan_bytes == 120


def test_bounded_hotset_stages_old_set_until_new_publication() -> None:
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "FULL", 16, 16, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)
    cache.register(new, price=price("new", resident=20, peak=24))

    plan = cache.plan(
        "replace",
        (new,),
        request_bytes=10,
        available_bytes=64,
        owner_set_capture_envelope_bytes=24,
        replace_unleased_on_miss=True,
        residency_cap_bytes=24,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.staged_hotset_replace
    assert plan.victim_keys == (old,)
    assert plan.capture_loan_bytes == 24

    cache.begin_maintenance(plan)
    assert cache.entries[old].hot
    assert cache.entries[old].leases == frozenset({"replace"})
    assert cache.entries[new].state == GraphResidency.CAPTURING

    cache.finish_maintenance(
        plan,
        {new: price("new", resident=20, peak=24)},
    )
    assert cache.entries[old].state == GraphResidency.COLD
    assert not cache.entries[old].leases
    assert cache.entries[new].hot
    assert cache.stats.evictions == 1


def test_bounded_hotset_capture_failure_preserves_old_set() -> None:
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "FULL", 16, 16, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)
    cache.register(new, price=price("new", resident=20, peak=24))
    plan = cache.plan(
        "replace-fail",
        (new,),
        request_bytes=10,
        available_bytes=64,
        owner_set_capture_envelope_bytes=24,
        replace_unleased_on_miss=True,
        residency_cap_bytes=24,
    )
    cache.begin_maintenance(plan)
    cache.fail_maintenance(plan)

    assert cache.entries[old].hot
    assert cache.entries[old].reclaimable
    assert cache.entries[new].state == GraphResidency.COOLDOWN
    assert cache.stats.evictions == 0


def test_bounded_hotset_rejects_unreclaimable_but_separates_capture_peak() -> None:
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "FULL", 16, 16, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old, pinned=True)
    cache.register(new, price=price("new", resident=20, peak=24))

    pinned = cache.plan(
        "pinned",
        (new,),
        request_bytes=10,
        available_bytes=64,
        owner_set_capture_envelope_bytes=24,
        replace_unleased_on_miss=True,
        residency_cap_bytes=24,
    )
    assert pinned.kind == ElasticPlanKind.DEFER
    assert pinned.defer_reason == "hotset_victim_not_reclaimable"
    assert cache.entries[old].hot

    empty = ElasticGraphCache(GENERATION)
    conservative_capture = empty.plan(
        "oversized",
        (new,),
        request_bytes=0,
        available_bytes=128,
        owner_set_capture_envelope_bytes=40,
        replace_unleased_on_miss=True,
        residency_cap_bytes=32,
    )
    assert conservative_capture.kind == ElasticPlanKind.MAINTENANCE
    assert conservative_capture.staged_hotset_replace
    assert conservative_capture.capture_loan_bytes == 40


def test_idle_unpin_makes_pinned_graph_reclaimable() -> None:
    pinned_key = key("target", "FULL", 7, 7, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, pinned_key, pinned=True)

    assert cache.unpin_idle((pinned_key,)) == (pinned_key,)
    assert cache.entries[pinned_key].reclaimable
    reclaim = cache.plan_reclaim_all(
        "cold-epoch",
        request_bytes=10,
        available_bytes=0,
    )
    assert reclaim.kind == ElasticPlanKind.RECLAIM
    assert reclaim.victim_keys == (pinned_key,)


def test_idle_unpin_rejects_leased_graph() -> None:
    pinned_key = key("target", "FULL", 9, 9, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, pinned_key, pinned=True)
    plan = cache.plan("user", (pinned_key,), request_bytes=10, available_bytes=20)
    cache.commit_user(plan)

    with pytest.raises(ElasticGraphError, match="cannot unpin an active"):
        cache.unpin_idle((pinned_key,))

    assert cache.entries[pinned_key].pinned
    assert cache.entries[pinned_key].leases == frozenset({"user"})


def test_bounded_hotset_retains_priced_entries_below_cap() -> None:
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "FULL", 16, 16, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)
    cache.register(new, price=price("new", resident=20, peak=24))

    plan = cache.plan(
        "retain-below-cap",
        (new,),
        request_bytes=10,
        available_bytes=64,
        owner_set_capture_envelope_bytes=24,
        replace_unleased_on_miss=True,
        residency_cap_bytes=32,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.staged_hotset_replace
    assert not plan.victim_keys
    assert plan.capture_loan_bytes == 34
    cache.begin_maintenance(plan)
    cache.finish_maintenance(plan, {new: price("new", resident=20, peak=24)})
    assert cache.entries[old].hot
    assert cache.entries[new].hot
    assert cache.stats.evictions == 0


def test_bounded_hotset_retains_unknown_exact_set_when_envelope_fits() -> None:
    """A miss is not an eviction trigger when conservative bytes fit."""
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "PIECEWISE", 12, 3)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)

    plan = cache.plan(
        "retain-unknown-below-cap",
        (new,),
        request_bytes=100,
        available_bytes=512,
        owner_set_capture_envelope_bytes=180,
        retained_transition_overlap_bytes=20,
        replace_unleased_on_miss=True,
        residency_cap_bytes=320,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.staged_hotset_replace
    assert plan.victim_keys == ()
    assert plan.capture_loan_bytes == 300
    assert cache.entries[old].hot


def test_bounded_hotset_unknown_set_replaces_only_on_proven_cap_deficit() -> None:
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "PIECEWISE", 12, 3)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)

    plan = cache.plan(
        "replace-unknown-over-cap",
        (new,),
        request_bytes=180,
        available_bytes=512,
        owner_set_capture_envelope_bytes=180,
        replace_unleased_on_miss=True,
        residency_cap_bytes=320,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == (old,)


def test_unknown_set_does_not_double_count_shared_resident_floor() -> None:
    """Destination envelope and current endpoint share retained CUDA state."""
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "PIECEWISE", 8, 2)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)

    plan = cache.plan(
        "retain-shared-floor",
        (new,),
        request_bytes=650,
        available_bytes=700,
        owner_set_capture_envelope_bytes=526,
        shared_resident_bytes=520,
        retained_transition_overlap_bytes=20,
        replace_unleased_on_miss=True,
        residency_cap_bytes=671,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == ()
    assert plan.capture_loan_bytes == 676


def test_bounded_hotset_retained_transition_composes_workspace_overlap() -> None:
    """The retained-endpoint max must not mask measured workspace overlap."""
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "FULL", 16, 16, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)
    cache.register(new, price=price("new", resident=20, peak=24))

    plan = cache.plan(
        "retain-with-overlap",
        (new,),
        request_bytes=10,
        available_bytes=96,
        owner_set_capture_envelope_bytes=30,
        retained_transition_overlap_bytes=32,
        replace_unleased_on_miss=True,
        residency_cap_bytes=64,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == ()
    # Without the explicit atomic term, request + peak (=34) masks the
    # overlap carried only by a smaller owner-set envelope (=30).
    assert plan.capture_loan_bytes == 66


def test_live_retained_transition_receipt_funds_observed_residency() -> None:
    """Replay the exact byte boundary from the failed short12 transaction."""
    current_endpoint = 185_335_808
    old_grant = 297_795_584
    workspace_unit = 33_554_432
    observed_residency = 309_067_776
    miss_peaks = (23_855_104, 44_040_192, 44_564_480)
    assert current_endpoint + sum(miss_peaks) == old_grant

    cache = ElasticGraphCache(GENERATION)
    old = key("old", "FULL", 2, 2, uniform=1)
    publish(cache, old)
    misses = (
        key("target", "PIECEWISE", 4, 1),
        key("mtp_prefill", "PIECEWISE", 4, 1),
        key("mtp_decode", "FULL", 1, 1, uniform=1),
    )
    for graph_key, peak in zip(misses, miss_peaks, strict=True):
        cache.register(
            graph_key,
            price=price(graph_key.logical.owner, resident=peak, peak=peak),
        )

    plan = cache.plan(
        "live-receipt-replay",
        misses,
        request_bytes=current_endpoint,
        available_bytes=1 << 30,
        owner_set_capture_envelope_bytes=old_grant,
        retained_transition_overlap_bytes=workspace_unit,
        replace_unleased_on_miss=True,
        residency_cap_bytes=640 << 20,
    )

    assert plan.victim_keys == ()
    assert plan.capture_loan_bytes == 331_350_016
    assert plan.capture_loan_bytes >= observed_residency


def test_bounded_hotset_over_cap_replaces_old_endpoint() -> None:
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "FULL", 16, 16, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)
    cache.register(new, price=price("new", resident=20, peak=24))
    plan = cache.plan(
        "absolute-endpoint",
        (new,),
        request_bytes=20,
        available_bytes=44,
        owner_set_capture_envelope_bytes=39,
        replace_unleased_on_miss=True,
        residency_cap_bytes=32,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    # The settled old+new residency would be 30 bytes, but request_bytes is
    # the authoritative current endpoint for cap admission.  20+20 exceeds
    # the declared 32-byte hotset, so this is replacement, not retention.
    assert plan.capture_loan_bytes == 39
    assert plan.staged_hotset_replace
    assert cache.entries[old].hot
    assert plan.victim_keys == (old,)


def test_bounded_hotset_reclaims_only_minimum_lru_groups_for_deficit() -> None:
    oldest = key("target", "FULL", 1, 1, uniform=1)
    newer = key("target", "FULL", 2, 2, uniform=1)
    newest = key("target", "FULL", 4, 4, uniform=1)
    cold = key("target", "FULL", 8, 8, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    for graph_key, resident in ((oldest, 10), (newer, 11), (newest, 12)):
        graph_price = price(graph_key.identity, resident=resident, peak=resident)
        cache.publish_hot(graph_key, graph_price, pinned=False)
        cache.install_reclaim_group(
            ReclaimGroup(
                graph_price.reclaim_group,
                (graph_key,),
                reclaimable_bytes=resident,
            )
        )
    cache.register(cold, price=price("cold", resident=8, peak=8))

    # Current 33 + new 8 exceeds cap 25 by 16. The two oldest groups reclaim
    # 21; the newest remains HOT. Whole-hotset replacement would reclaim 33.
    plan = cache.plan(
        "minimal-deficit",
        (cold,),
        request_bytes=33,
        available_bytes=64,
        owner_set_capture_envelope_bytes=8,
        replace_unleased_on_miss=True,
        residency_cap_bytes=25,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == (oldest, newer)
    assert plan.reclaim_bytes == 21
    assert newest not in plan.victim_keys


def test_bounded_hotset_rejects_absolute_endpoint_above_capacity() -> None:
    old = key("target", "FULL", 1, 1, uniform=1)
    new = key("target", "FULL", 16, 16, uniform=1)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, old)
    cache.register(new, price=price("new", resident=20, peak=24))
    plan = cache.plan(
        "endpoint-too-large",
        (new,),
        request_bytes=20,
        available_bytes=39,
        owner_set_capture_envelope_bytes=40,
        replace_unleased_on_miss=True,
        residency_cap_bytes=32,
    )
    assert plan.kind == ElasticPlanKind.DEFER
    assert plan.defer_reason == "insufficient_atomic_hotset_transition_bytes"
    assert cache.entries[old].hot


def test_bounded_hotset_first_publication_is_staged_for_cap_validation() -> None:
    desired = resolve_step_physical_keys((1, 3, 40, 40, 1), GENERATION, 4096)
    cache = ElasticGraphCache(GENERATION)

    plan = cache.plan(
        "first-publication",
        desired,
        request_bytes=0,
        available_bytes=512,
        owner_set_capture_envelope_bytes=200,
        replace_unleased_on_miss=True,
        residency_cap_bytes=256,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.staged_hotset_replace
    assert not plan.victim_keys
    assert plan.physical_keys == desired
    assert [item.logical.owner for item in plan.capture_order] == [
        "mtp_prefill",
        "target",
        "mtp_decode",
    ]
    assert [item.logical.mode for item in plan.capture_order] == [
        "PIECEWISE",
        "FULL",
        "FULL",
    ]


def test_m160_owner_set_preserves_dispatch_identity_but_captures_piecewise_first(
) -> None:
    desired = resolve_step_physical_keys(
        (1, 3, 40, 160, 4),
        GENERATION,
        max_num_batched_tokens=4096,
        compiled_piecewise_sizes=(256, 512, 1024, 2048, 4096),
    )
    cache = ElasticGraphCache(GENERATION)

    plan = cache.plan(
        "m160-first-publication",
        desired,
        request_bytes=0,
        available_bytes=512,
        owner_set_capture_envelope_bytes=200,
        replace_unleased_on_miss=True,
        residency_cap_bytes=256,
    )

    assert [item.logical.owner for item in plan.physical_keys] == [
        "target",
        "mtp_prefill",
        "mtp_decode",
    ]
    assert [item.logical.owner for item in plan.capture_order] == [
        "mtp_prefill",
        "target",
        "mtp_decode",
    ]
    assert [item.logical.mode for item in plan.capture_order] == [
        "PIECEWISE",
        "FULL",
        "FULL",
    ]


def test_bounded_hotset_a_b_a_retains_both_owner_sets_with_shared_floor() -> None:
    """A and B capture once each; the return to A is a pure HOT hit.

    The worker aggregate includes a large shared CUDA/runtime floor.  It must
    be counted once when predicting the settled A+B endpoint, otherwise the
    first B miss spuriously replaces A and turns A->B->A into recapture churn.
    """
    cache = ElasticGraphCache(GENERATION)
    a = key("target", "FULL", 4, 1, uniform=4)
    b = key("target", "FULL", 8, 2, uniform=4)
    shared_floor = 520
    cap = 671

    first_a = cache.plan(
        "capture-a",
        (a,),
        request_bytes=0,
        available_bytes=cap,
        owner_set_capture_envelope_bytes=650,
        replace_unleased_on_miss=True,
        residency_cap_bytes=cap,
    )
    assert first_a.kind == ElasticPlanKind.MAINTENANCE
    cache.begin_maintenance(first_a)
    cache.finish_maintenance(
        first_a,
        {a: price("a", resident=130, peak=130)},
    )

    first_b = cache.plan(
        "capture-b",
        (b,),
        request_bytes=650,
        available_bytes=700,
        owner_set_capture_envelope_bytes=526,
        shared_resident_bytes=shared_floor,
        replace_unleased_on_miss=True,
        residency_cap_bytes=cap,
    )
    assert first_b.kind == ElasticPlanKind.MAINTENANCE
    assert first_b.victim_keys == ()
    cache.begin_maintenance(first_b)
    cache.finish_maintenance(
        first_b,
        {b: price("b", resident=6, peak=6)},
    )

    return_a = cache.plan(
        "return-a",
        (a,),
        request_bytes=656,
        available_bytes=700,
        owner_set_capture_envelope_bytes=650,
        shared_resident_bytes=shared_floor,
        replace_unleased_on_miss=True,
        residency_cap_bytes=cap,
    )
    assert return_a.kind == ElasticPlanKind.USER
    assert return_a.hot_hits == (a,)
    assert return_a.cold_misses == ()
    assert cache.stats.cold_misses == 2
    assert cache.stats.promotions == 2
    assert cache.stats.evictions == 0


def test_observability_is_aggregate_and_trace_is_bounded() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, graph_key)
    for index in range(140):
        plan = cache.plan(
            f"hit-{index}", (graph_key,), request_bytes=0, available_bytes=0
        )
        cache.commit_user(plan)
        cache.release(plan.transaction_id)
    deferred = cache.plan(
        "unknown",
        (key("target", "PIECEWISE", 8192, 8),),
        request_bytes=0,
        available_bytes=0,
    )
    cache.observe_defer(deferred)
    assert cache.stats.hot_hits == 140
    assert cache.stats.deferrals == 1
    assert len(cache.trace) == 128


def test_kv_pressure_builds_explicit_reclaim_transaction() -> None:
    full = key("target", "FULL", 4, 1, uniform=4)
    piecewise = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, full, pinned=True)
    publish(cache, piecewise)
    plan = cache.plan_reclaim_all("kv-pressure", request_bytes=20, available_bytes=0)
    assert plan.kind == ElasticPlanKind.RECLAIM
    assert plan.victim_keys == (piecewise,)
    cache.begin_reclaim(plan)
    assert cache.entries[full].hot
    assert cache.entries[piecewise].state == GraphResidency.COLD


def test_insufficient_capacity_defers_without_evicting_pinned_or_leased() -> None:
    pinned = key("target", "FULL", 4, 1, uniform=4)
    leased = key("target", "PIECEWISE", 2048, 8)
    cold = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, pinned, pinned=True)
    publish(cache, leased)
    active = cache.plan("active", (leased,), request_bytes=0, available_bytes=0)
    cache.commit_user(active)
    cache.register(cold, price=price("cold", resident=9, peak=12))

    plan = cache.plan("blocked", (cold,), request_bytes=8, available_bytes=0)
    assert plan.kind == ElasticPlanKind.DEFER
    assert not plan.victim_keys
    assert cache.entries[pinned].hot
    assert cache.entries[leased].leases == frozenset({"active"})


def test_rank_consensus_rejects_stale_or_divergent_plan_before_mutation() -> None:
    graph_key = key("target", "FULL", 4, 1, uniform=4)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, graph_key, pinned=True)
    plan = cache.plan("same", (graph_key,), request_bytes=0, available_bytes=0)
    assert require_plan_consensus((plan, plan)) == plan.fingerprint
    divergent = replace(plan, request_bytes=1)
    with pytest.raises(ElasticGraphError, match="differs by rank"):
        require_plan_consensus((plan, divergent))

    stale_key = replace(graph_key, generation=RuntimeGeneration("stale"))
    with pytest.raises(ElasticGraphError, match="stale generation"):
        cache.plan("stale", (stale_key,), request_bytes=0, available_bytes=0)


def test_last_item_release_does_not_clear_idle_residency() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, graph_key)
    plan = cache.plan("tail", (graph_key,), request_bytes=0, available_bytes=0)
    cache.commit_user(plan)
    cache.release("tail")
    assert cache.entries[graph_key].hot
    assert cache.entries[graph_key].reclaimable


def test_deferred_free_blocks_reclaim_until_physical_completion() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cold = key("mtp_prefill", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    publish(cache, graph_key)
    cache._entries[graph_key] = replace(cache.entries[graph_key], deferred_free=True)
    cache.register(cold, price=price("cold", resident=9, peak=12))
    plan = cache.plan("tail", (cold,), request_bytes=1, available_bytes=0)
    assert plan.kind == ElasticPlanKind.DEFER
    assert not plan.victim_keys


def test_reclaim_group_is_atomic_not_sum_of_marginal_entry_prices() -> None:
    first = key("target", "PIECEWISE", 2048, 8)
    second = key("mtp_prefill", "PIECEWISE", 2048, 8)
    cold = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticGraphCache(GENERATION)
    shared = price("shared-pool", resident=10, peak=12)
    cache.publish_hot(first, shared, pinned=False)
    cache.publish_hot(second, shared, pinned=False)
    cache.install_reclaim_group(ReclaimGroup("shared-pool", (first, second), 14))
    cache.register(cold, price=price("cold", resident=9, peak=12))

    plan = cache.plan("group", (cold,), request_bytes=2, available_bytes=0)
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.reclaim_bytes == 14
    assert plan.victim_keys == (first, second)
