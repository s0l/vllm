# SPDX-License-Identifier: Apache-2.0
import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm.v1.core.elastic_graph import (
    DispatchRepresentation,
    ElasticAdmissionController,
    ElasticAdmissionLoan,
    ElasticGraphError,
    ElasticPlanKind,
    ElasticResidencyEntry,
    ElasticResidencyReceipt,
    ElasticRuntimeConfig,
    GraphExecutionPolicy,
    GraphPrice,
    GraphResidency,
    LogicalDispatchKey,
    OwnerGraphExecutionPolicy,
    PhysicalReplayKey,
    ReclaimGroup,
    RuntimeGeneration,
    build_execution_manifest,
    canonical_execution_request_order,
    configured_compiled_piecewise_sizes,
    derive_short_decode_graph_inventory,
    execution_manifest_phase_from_step_key,
    require_plan_consensus,
    resolve_step_physical_keys,
)

GENERATION = RuntimeGeneration("test-generation")


def test_canonical_execution_order_matches_decode_then_prefill_input_batch() -> None:
    scheduled = {"prefill-q1": 1, "decode-q4": 4, "prefill-q3": 3}

    assert canonical_execution_request_order(
        scheduled,
        is_prefilling_by_request={
            "prefill-q1": True,
            "decode-q4": False,
            "prefill-q3": True,
        },
        decode_query_len=4,
    ) == ("decode-q4", "prefill-q1", "prefill-q3")

    with pytest.raises(ValueError, match="lifecycle identity differs"):
        canonical_execution_request_order(
            scheduled,
            is_prefilling_by_request={"decode-q4": False},
            decode_query_len=4,
        )


def test_controller_fifo_loans_and_transaction_ids_are_public_and_replayable() -> None:
    controller = ElasticAdmissionController(GENERATION)

    assert controller.next_transaction_id() == "elastic-00000000000000000001"
    assert controller.next_transaction_id() == "elastic-00000000000000000002"
    controller.reserve_loan((1, 3, 8, 32, 1), 128)
    controller.reserve_loan(None, 0)
    controller.replace_latest_loan((0, 3, 16, 64, 0), 96)

    assert controller.snapshot.pending_loans == (
        ElasticAdmissionLoan((1, 3, 8, 32, 1), 128),
        ElasticAdmissionLoan((0, 3, 16, 64, 0), 96),
    )
    assert controller.max_pending_grant() == 128
    assert controller.settle_next_loan() == ElasticAdmissionLoan((1, 3, 8, 32, 1), 128)
    assert controller.cancel_latest_loan() == ElasticAdmissionLoan(
        (0, 3, 16, 64, 0), 96
    )
    assert controller.pending_loans == ()

    with pytest.raises(ValueError, match="non-negative"):
        controller.reserve_loan(None, True)
    with pytest.raises(ValueError, match="non-negative"):
        controller.reserve_loan(None, "128")  # type: ignore[arg-type]
    with pytest.raises(ElasticGraphError, match="missing elastic loan"):
        controller.settle_next_loan()


def test_controller_receipt_acceptance_is_atomic_and_transaction_bound() -> None:
    controller = ElasticAdmissionController(GENERATION)
    physical = key("target", "PIECEWISE", 32, 8)
    receipt = ElasticResidencyReceipt(
        generation=GENERATION,
        transaction_id="elastic-00000000000000000001",
        resident_bytes=88,
        floor_bytes=8,
        transition_floor_bytes=16,
        peak_bytes=96,
        cublas_workspace_bytes=4,
        entries=(
            ElasticResidencyEntry(
                key=physical,
                pinned=False,
                resident_bytes=80,
                local_pool_bytes=80,
                reclaimable_bytes=80,
            ),
        ),
    )

    assert controller.accept_residency_receipt(
        receipt,
        expected_transaction_id=receipt.transaction_id,
    ) == (0, 80)
    accepted = controller.snapshot
    assert controller.last_receipt == receipt

    stale = replace(
        receipt,
        generation=RuntimeGeneration("stale-generation"),
        entries=(
            replace(
                receipt.entries[0],
                key=replace(physical, generation=RuntimeGeneration("stale-generation")),
            ),
        ),
    )
    with pytest.raises(ElasticGraphError, match="generation differs"):
        controller.accept_residency_receipt(stale)
    with pytest.raises(ElasticGraphError, match="transaction differs"):
        controller.accept_residency_receipt(
            replace(receipt, transaction_id="out-of-order"),
            expected_transaction_id=receipt.transaction_id,
        )

    assert controller.snapshot == accepted
    assert controller.last_receipt == receipt


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


def execution_manifest_policy(
    *, compiled_target_sizes: tuple[int, ...] = ()
) -> GraphExecutionPolicy:
    return runtime_shape_policy(
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
            activation="speculative",
            token_source="step",
            execution_order=1,
        ),
        OwnerGraphExecutionPolicy(
            "target",
            (),
            "PIECEWISE",
            compiled_piecewise_sizes=compiled_target_sizes,
            activation="always",
            token_source="step",
            execution_order=0,
        ),
    )


def test_execution_manifest_keeps_q1_current_separate_from_q4_successor() -> None:
    policy = execution_manifest_policy()
    manifest, dispatch = build_execution_manifest(
        step_key=(0, 3, 1, 1, 0),
        request_ids=("request-0",),
        per_request_query_lens=(1,),
        per_request_is_prefilling=(True,),
        scheduled_draft_rows=(0,),
        requested_output_k=3,
        executed_drafter_k=3,
        phase="mixed",
        generation=GENERATION,
        policy=policy,
        max_num_batched_tokens=4096,
    )

    assert manifest.per_request_query_lens == (1,)
    assert manifest.scheduled_draft_rows == (0,)
    assert [invocation.owner for invocation in manifest.invocations] == [
        "target",
        "mtp_prefill",
        "mtp_decode",
    ]
    assert [invocation.live_num_tokens for invocation in manifest.invocations] == [
        1,
        1,
        1,
    ]
    assert all(
        item.representation == DispatchRepresentation.HOT_GRAPH for item in dispatch
    )
    assert {item.physical_key.logical.token_bucket for item in dispatch} == {1}

    successor = resolve_step_physical_keys(
        (0, 3, 1, 4, 4), GENERATION, 4096, policy=policy
    )
    assert {key.logical.token_bucket for key in successor} == {1, 4}
    assert set(successor) != {
        item.physical_key for item in dispatch if item.physical_key is not None
    }


def test_variable_decode_uses_piecewise_manifest_lane() -> None:
    step_key = (0, 3, 2, 8, 0)
    phase = execution_manifest_phase_from_step_key(step_key)
    manifest, dispatch = build_execution_manifest(
        step_key=step_key,
        request_ids=("request-q4", "request-q3"),
        per_request_query_lens=(4, 3),
        per_request_is_prefilling=(False, False),
        scheduled_draft_rows=(3, 2),
        requested_output_k=3,
        executed_drafter_k=3,
        phase=phase,
        generation=GENERATION,
        policy=execution_manifest_policy(compiled_target_sizes=(8,)),
        max_num_batched_tokens=4096,
    )

    assert phase == "mixed"
    assert manifest.per_request_is_prefilling == (False, False)
    assert [item.phase for item in manifest.invocations] == ["mixed"] * 3
    assert [item.live_num_tokens for item in manifest.invocations] == [7, 7, 2]
    assert [item.physical_num_tokens for item in manifest.invocations] == [8, 8, 2]
    assert dispatch[0].representation == DispatchRepresentation.COMPILED_ONLY
    assert all(
        item.representation == DispatchRepresentation.HOT_GRAPH for item in dispatch[1:]
    )


def test_execution_manifest_binds_single_final_replay_inside_mixed_rows() -> None:
    is_prefilling_by_request = {
        "decode": False,
        # The semantic prompt ended at position 3, but preemption extended the
        # execution-prefill boundary to 4. Position 3 is therefore the final
        # replay row, not ordinary decode.
        "resumed-final-replay": 3 < 4,
    }
    request_ids = canonical_execution_request_order(
        {"resumed-final-replay": 1, "decode": 1},
        is_prefilling_by_request=is_prefilling_by_request,
        decode_query_len=4,
    )
    manifest, _dispatch = build_execution_manifest(
        step_key=(0, 0, 2, 2, 0),
        request_ids=request_ids,
        per_request_query_lens=(1, 1),
        per_request_is_prefilling=tuple(
            is_prefilling_by_request[request_id] for request_id in request_ids
        ),
        scheduled_draft_rows=(0, 0),
        requested_output_k=0,
        executed_drafter_k=0,
        phase="mixed",
        generation=GENERATION,
        policy=execution_manifest_policy(),
        max_num_batched_tokens=4096,
    )

    assert manifest.request_ids == ("decode", "resumed-final-replay")
    assert manifest.per_request_is_prefilling == (False, True)
    assert (
        replace(
            manifest,
            per_request_is_prefilling=(False, False),
        ).fingerprint
        != manifest.fingerprint
    )


def test_manifest_phase_comes_from_uniform_marker_not_lifecycle_label() -> None:
    assert execution_manifest_phase_from_step_key(None) is None
    assert execution_manifest_phase_from_step_key((0, 3, 12, 32, 0)) == "mixed"
    assert execution_manifest_phase_from_step_key((0, 3, 12, 48, 4)) == "decode"


def test_mixed_q1_does_not_select_uniform_decode_full_graph() -> None:
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
            (4,),
            "PIECEWISE",
            activation="speculative",
            token_source="step",
            execution_order=1,
        ),
        OwnerGraphExecutionPolicy(
            "target",
            (1,),
            "PIECEWISE",
            activation="always",
            token_source="step",
            execution_order=0,
        ),
    )

    _manifest, mixed_dispatch = build_execution_manifest(
        step_key=(0, 3, 1, 1, 0),
        request_ids=("request-0",),
        per_request_query_lens=(1,),
        per_request_is_prefilling=(True,),
        scheduled_draft_rows=(0,),
        requested_output_k=3,
        executed_drafter_k=3,
        phase="mixed",
        generation=GENERATION,
        policy=policy,
        max_num_batched_tokens=4096,
    )
    _manifest, decode_dispatch = build_execution_manifest(
        step_key=(1, 3, 1, 1, 1),
        request_ids=("request-0",),
        per_request_query_lens=(1,),
        per_request_is_prefilling=(False,),
        scheduled_draft_rows=(0,),
        requested_output_k=3,
        executed_drafter_k=3,
        phase="decode",
        generation=GENERATION,
        policy=policy,
        max_num_batched_tokens=4096,
    )

    assert mixed_dispatch[0].physical_key is not None
    assert mixed_dispatch[0].physical_key.logical.mode == "PIECEWISE"
    assert decode_dispatch[0].physical_key is not None
    assert decode_dispatch[0].physical_key.logical.mode == "FULL"


def test_execution_manifest_rejects_successor_as_mixed_current() -> None:
    with pytest.raises(ElasticGraphError, match="cannot inherit a successor carrier"):
        build_execution_manifest(
            step_key=(0, 3, 1, 4, 4),
            request_ids=("request-0",),
            per_request_query_lens=(1,),
            per_request_is_prefilling=(True,),
            scheduled_draft_rows=(0,),
            requested_output_k=3,
            executed_drafter_k=3,
            phase="mixed",
            generation=GENERATION,
            policy=execution_manifest_policy(),
            max_num_batched_tokens=4096,
        )


def test_execution_manifest_declares_compiled_only_instead_of_graph_miss() -> None:
    manifest, dispatch = build_execution_manifest(
        step_key=(0, 0, 1, 4096, 0),
        request_ids=("request-0",),
        per_request_query_lens=(4096,),
        per_request_is_prefilling=(True,),
        scheduled_draft_rows=(0,),
        requested_output_k=0,
        executed_drafter_k=0,
        phase="mixed",
        generation=GENERATION,
        policy=execution_manifest_policy(compiled_target_sizes=(4096,)),
        max_num_batched_tokens=4096,
    )

    assert tuple(item.invocation for item in dispatch) == manifest.invocations
    assert len(dispatch) == 1
    assert dispatch[0].representation == DispatchRepresentation.COMPILED_ONLY
    assert dispatch[0].physical_key is None


@pytest.mark.parametrize(
    (
        "step_key",
        "query_lens",
        "draft_rows",
        "requested_k",
        "executed_k",
        "phase",
        "expected_target_bucket",
    ),
    (
        ((0, 3, 1, 1, 0), (1,), (0,), 3, 3, "mixed", 1),
        ((0, 3, 1, 2, 0), (2,), (0,), 3, 3, "mixed", 2),
        ((0, 3, 1, 4, 0), (3,), (1,), 3, 3, "mixed", 4),
        ((0, 3, 2, 8, 0), (1, 4), (0, 2), 3, 3, "mixed", 8),
        ((0, 3, 2, 8, 4), (4, 4), (3, 3), 3, 3, "decode", 8),
        ((0, 0, 1, 2, 0), (2,), (0,), 0, 0, "mixed", 2),
        # Dynamic scheduling may request one draft while the MTP module still
        # executes its configured depth of three and slices the publication.
        ((0, 1, 1, 2, 2), (2,), (1,), 1, 3, "decode", 2),
    ),
)
def test_execution_manifest_state_matrix(
    step_key,
    query_lens,
    draft_rows,
    requested_k,
    executed_k,
    phase,
    expected_target_bucket,
) -> None:
    manifest, dispatch = build_execution_manifest(
        step_key=step_key,
        request_ids=tuple(f"request-{index}" for index in range(len(query_lens))),
        per_request_query_lens=query_lens,
        per_request_is_prefilling=(False,) * len(query_lens),
        scheduled_draft_rows=draft_rows,
        requested_output_k=requested_k,
        executed_drafter_k=executed_k,
        phase=phase,
        generation=GENERATION,
        policy=execution_manifest_policy(),
        max_num_batched_tokens=4096,
    )

    assert manifest.per_request_query_lens == query_lens
    assert manifest.scheduled_draft_rows == draft_rows
    assert dispatch[0].invocation.owner == "target"
    assert dispatch[0].invocation.physical_num_tokens == expected_target_bucket
    assert len(dispatch) == (3 if requested_k else 1)
    restored = pickle.loads(pickle.dumps((manifest, dispatch)))
    assert restored == (manifest, dispatch)
    assert restored[0].fingerprint == manifest.fingerprint


def test_execution_manifest_fingerprint_detects_owner_geometry_mutations() -> None:
    manifest, _dispatch = build_execution_manifest(
        step_key=(0, 3, 2, 8, 0),
        request_ids=("request-0", "request-1"),
        per_request_query_lens=(1, 4),
        per_request_is_prefilling=(False, True),
        scheduled_draft_rows=(0, 2),
        requested_output_k=3,
        executed_drafter_k=3,
        phase="mixed",
        generation=GENERATION,
        policy=execution_manifest_policy(),
        max_num_batched_tokens=4096,
    )

    assert manifest.per_request_is_prefilling == (False, True)
    with pytest.raises(ValueError, match="prefill phases must align"):
        replace(manifest, per_request_is_prefilling=(False,))
    with pytest.raises(ValueError, match="prefill phases must align"):
        replace(manifest, per_request_is_prefilling=(False, 1))

    mutations = (
        replace(manifest, request_ids=("request-1", "request-0")),
        replace(manifest, per_request_query_lens=(2, 3)),
        replace(manifest, per_request_is_prefilling=(True, False)),
        replace(manifest, scheduled_draft_rows=(1, 2)),
        replace(
            manifest,
            scheduled_encoder_inputs=(("request-0", (0,)),),
        ),
        replace(
            manifest,
            requested_output_k=2,
            invocations=tuple(
                replace(invocation, requested_output_k=2)
                for invocation in manifest.invocations
            ),
        ),
        replace(
            manifest,
            executed_drafter_k=4,
            invocations=tuple(
                replace(invocation, executed_drafter_k=4)
                for invocation in manifest.invocations
            ),
        ),
        replace(
            manifest,
            active_lora_ids=(7,),
            invocations=tuple(
                replace(invocation, active_loras=1)
                for invocation in manifest.invocations
            ),
        ),
        replace(
            manifest,
            generation=RuntimeGeneration("stale"),
            invocations=tuple(
                replace(invocation, generation=RuntimeGeneration("stale"))
                for invocation in manifest.invocations
            ),
        ),
    )
    assert all(item.fingerprint != manifest.fingerprint for item in mutations)


def test_execution_manifest_rejects_underfilled_or_overlimit_identity() -> None:
    policy = execution_manifest_policy()
    with pytest.raises(ElasticGraphError, match="max_num_batched_tokens"):
        build_execution_manifest(
            step_key=(0, 3, 1, 4, 0),
            request_ids=("request-0",),
            per_request_query_lens=(5,),
            per_request_is_prefilling=(True,),
            scheduled_draft_rows=(0,),
            requested_output_k=3,
            executed_drafter_k=3,
            phase="mixed",
            generation=GENERATION,
            policy=policy,
            max_num_batched_tokens=4,
        )

    with pytest.raises(ElasticGraphError, match="scheduled target draft rows"):
        build_execution_manifest(
            step_key=(0, 3, 1, 1, 0),
            request_ids=("request-0",),
            per_request_query_lens=(1,),
            per_request_is_prefilling=(True,),
            scheduled_draft_rows=(4,),
            requested_output_k=3,
            executed_drafter_k=3,
            phase="mixed",
            generation=GENERATION,
            policy=policy,
            max_num_batched_tokens=4,
        )


def test_execution_manifest_binds_encoder_schedule_and_lora_identity() -> None:
    manifest, _dispatch = build_execution_manifest(
        step_key=(0, 3, 2, 8, 0),
        request_ids=("vision", "text"),
        per_request_query_lens=(5, 3),
        per_request_is_prefilling=(True, False),
        scheduled_draft_rows=(0, 1),
        scheduled_encoder_inputs={"vision": [1, 0, 1]},
        requested_output_k=3,
        executed_drafter_k=3,
        phase="mixed",
        generation=GENERATION,
        policy=execution_manifest_policy(),
        max_num_batched_tokens=4096,
    )

    assert manifest.scheduled_encoder_inputs == (("vision", (0, 1)),)
    assert manifest.active_lora_ids == ()
    with pytest.raises(ElasticGraphError, match="LoRA identity"):
        build_execution_manifest(
            step_key=(0, 3, 2, 8, 0),
            request_ids=("vision", "text"),
            per_request_query_lens=(5, 3),
            per_request_is_prefilling=(True, False),
            scheduled_draft_rows=(0, 1),
            active_lora_ids=(7,),
            requested_output_k=3,
            executed_drafter_k=3,
            phase="mixed",
            generation=GENERATION,
            policy=execution_manifest_policy(),
            max_num_batched_tokens=4096,
        )


def test_execution_manifest_bounded_geometry_surface() -> None:
    policy = execution_manifest_policy()
    for requested_k in range(4):
        executed_k = 3 if requested_k else 0
        for x in range(1, 9):
            for live_per_request in (1, 2, 3, 4, 17, 511):
                query_lens = tuple(live_per_request + (index % 2) for index in range(x))
                live_m = sum(query_lens)
                physical_m = 1 << (live_m - 1).bit_length()
                if physical_m > 4096:
                    continue
                manifest, dispatch = build_execution_manifest(
                    step_key=(0, requested_k, x, physical_m, 0),
                    request_ids=tuple(f"r-{index}" for index in range(x)),
                    per_request_query_lens=query_lens,
                    per_request_is_prefilling=(False,) * x,
                    scheduled_draft_rows=tuple(
                        min(index % 4, executed_k) for index in range(x)
                    ),
                    requested_output_k=requested_k,
                    executed_drafter_k=executed_k,
                    phase="mixed",
                    generation=GENERATION,
                    policy=policy,
                    max_num_batched_tokens=4096,
                )
                assert tuple(item.invocation for item in dispatch) == (
                    manifest.invocations
                )
                assert len(dispatch) == (3 if requested_k else 1)


def test_pre_mutation_maintenance_rollback_restores_prior_hot_set() -> None:
    controller = ElasticAdmissionController(GENERATION)
    source = key("target", "PIECEWISE", 2, 1)
    destination = key("target", "PIECEWISE", 4, 1)
    publish(controller, source)
    controller.register(destination, price=price("destination"))
    before_entries = dict(controller.entries)
    before_stats = controller.stats
    before_trace = controller.trace
    plan = controller.plan(
        "pre-mutation-rollback",
        (destination,),
        request_bytes=10,
        available_bytes=12,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE

    controller.begin_maintenance(plan)
    assert not controller.entries[source].hot
    controller.rollback_pre_mutation(plan)

    assert controller.entries == before_entries
    assert controller.stats == before_stats
    assert controller.trace == before_trace


def test_pre_mutation_user_rollback_releases_only_transaction_lease() -> None:
    controller = ElasticAdmissionController(GENERATION)
    current = key("target", "PIECEWISE", 2, 1)
    publish(controller, current)
    plan = controller.plan(
        "user-rollback",
        (current,),
        request_bytes=10,
        available_bytes=12,
    )
    before_stats = controller.stats
    before_trace = controller.trace
    controller.commit_user(plan)
    assert controller.entries[current].leases == frozenset({"user-rollback"})

    controller.rollback_pre_mutation(plan)

    assert controller.entries[current].hot
    assert not controller.entries[current].leases
    assert controller.stats == before_stats
    assert controller.trace == before_trace


def test_pre_mutation_reclaim_rollback_restores_evicted_hot_set() -> None:
    controller = ElasticAdmissionController(GENERATION)
    victim = key("target", "PIECEWISE", 2, 1)
    publish(controller, victim)
    before_entries = dict(controller.entries)
    before_stats = controller.stats
    before_trace = controller.trace
    plan = controller.plan_reclaim_all(
        "reclaim-rollback",
        request_bytes=0,
        available_bytes=0,
    )
    assert plan.kind == ElasticPlanKind.RECLAIM

    controller.begin_reclaim(plan)
    assert not controller.entries[victim].hot
    controller.rollback_pre_mutation(plan)

    assert controller.entries == before_entries
    assert controller.stats == before_stats
    assert controller.trace == before_trace


def test_maintenance_receipt_may_publish_captured_and_protected_hot_keys() -> None:
    controller = ElasticAdmissionController(GENERATION)
    captured = key("target", "PIECEWISE", 4, 1)
    protected = key("target", "PIECEWISE", 8, 1)
    controller.register(captured, price=price("captured"))
    publish(controller, protected)
    plan = controller.plan(
        "maintenance-receipt",
        (captured,),
        protected_keys=(protected,),
        request_bytes=10,
        available_bytes=32,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    receipt = ElasticResidencyReceipt(
        generation=GENERATION,
        transaction_id=plan.transaction_id,
        resident_bytes=20,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=20,
        cublas_workspace_bytes=0,
        entries=tuple(
            ElasticResidencyEntry(
                key=physical_key,
                resident_bytes=10,
                local_pool_bytes=10,
                pinned=False,
                reclaimable_bytes=10,
            )
            for physical_key in sorted(
                (captured, protected), key=lambda item: item.identity
            )
        ),
    )

    controller.validate_residency_receipt(
        receipt,
        expected_transaction_id=plan.transaction_id,
        required_hot_keys=(*plan.physical_keys, *plan.protected_keys),
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


def publish(
    cache: ElasticAdmissionController, graph_key: PhysicalReplayKey, *, pinned=False
):
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

    cache = ElasticAdmissionController(GENERATION)
    publish(cache, x8)
    publish(cache, x32)
    assert cache.entries[x8].hot
    assert cache.entries[x32].hot


def test_recapture_price_keeps_conservative_cold_envelope() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticAdmissionController(GENERATION)
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

    assert cache.entries[graph_key].price == price("private-pool", resident=44, peak=52)
    assert cache._groups["private-pool"] == ReclaimGroup(
        "private-pool", (graph_key,), reclaimable_bytes=1, retained_bytes=1
    )


def test_zero_reclaim_proof_is_retained_not_falsely_evictable() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticAdmissionController(GENERATION)
    receipt = ElasticResidencyReceipt(
        generation=GENERATION,
        transaction_id="zero-proof-receipt",
        resident_bytes=94,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=94,
        cublas_workspace_bytes=0,
        entries=(
            ElasticResidencyEntry(
                key=graph_key,
                pinned=False,
                resident_bytes=94,
                local_pool_bytes=0,
                reclaimable_bytes=0,
            ),
        ),
    )

    assert cache.accept_residency_receipt(receipt) == (94, 0)
    assert cache.entries[graph_key].pinned
    assert cache._groups == {}
    reclaim = cache.plan_reclaim_all(
        "zero-proof-tail",
        request_bytes=cache.resident_bytes,
        available_bytes=0,
    )
    assert reclaim.kind == ElasticPlanKind.DEFER
    assert reclaim.defer_reason == "no_reclaimable_piecewise_graphs"


def test_pinned_receipt_preserves_proof_for_explicit_idle_unpin() -> None:
    graph_key = key("target", "FULL", 8, 8, uniform=1)
    cache = ElasticAdmissionController(GENERATION)
    receipt = ElasticResidencyReceipt(
        generation=GENERATION,
        transaction_id="pinned-proof-receipt",
        resident_bytes=80,
        floor_bytes=0,
        transition_floor_bytes=0,
        peak_bytes=80,
        cublas_workspace_bytes=0,
        entries=(
            ElasticResidencyEntry(
                key=graph_key,
                pinned=True,
                resident_bytes=80,
                local_pool_bytes=80,
                reclaimable_bytes=79,
            ),
        ),
    )

    assert cache.accept_residency_receipt(receipt) == (80, 0)
    assert cache.entries[graph_key].pinned
    assert cache.unpin_idle() == (graph_key,)
    reclaim = cache.plan_reclaim_all(
        "pinned-proof-rebuild", request_bytes=80, available_bytes=0
    )
    assert reclaim.kind == ElasticPlanKind.RECLAIM
    assert reclaim.victim_keys == (graph_key,)


def test_recapture_rejects_changed_reclaim_identity() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticAdmissionController(GENERATION)
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
    below = resolve_step_physical_keys((0, 3, 1, 4, 4), GENERATION, 4096, policy=policy)
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

    cache = ElasticAdmissionController(GENERATION)
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
    q1 = resolve_step_physical_keys((1, 0, 32, 32, 1), GENERATION, 4096, policy=policy)
    assert [(item.logical.owner, item.logical.mode) for item in q1] == [
        ("target", "FULL")
    ]

    with pytest.raises(ElasticGraphError, match="FULL key violates"):
        resolve_step_physical_keys((1, 3, 40, 160, 4), GENERATION, 4096, policy=policy)


def test_policy_payload_is_content_addressed_and_corruption_fails_closed() -> None:
    payload = current_q4_piecewise_policy().to_payload()
    assert (
        GraphExecutionPolicy.from_payload(payload).fingerprint == payload["fingerprint"]
    )
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
        resolve_step_physical_keys((0, 0, 1, 4, 0), GENERATION, 4096, policy=policy)


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

    keys = resolve_step_physical_keys((1, 0, 7, 7, 1), GENERATION, 4096, policy=policy)

    assert [
        (item.logical.owner, item.logical.mode, item.logical.token_bucket)
        for item in keys
    ] == [("target", "FULL", 7)]


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

    keys = resolve_step_physical_keys((0, 7, 5, 40, 8), GENERATION, 4096, policy=policy)

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

    keys = resolve_step_physical_keys((0, 7, 5, 40, 8), GENERATION, 4096, policy=policy)

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
    assert [keys[0].logical.token_bucket for keys in inventory.values()] == [1, 2, 4, 5]


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
    assert (
        resolve_step_physical_keys(
            (0, 0, 40, 4096, 0),
            GENERATION,
            max_num_batched_tokens=4096,
            compiled_piecewise_sizes=(4096,),
        )
        == ()
    )

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
        assert (
            resolve_step_physical_keys(
                (0, 0, 1, tokens, 0),
                GENERATION,
                max_num_batched_tokens=4096,
                compiled_piecewise_sizes=compiled,
            )
            == ()
        )
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
        assert [item.logical.owner for item in mixed_same_geometry] == ["mtp_decode"]

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
    cache = ElasticAdmissionController(GENERATION)
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


def test_step5_cold_a_b_a_alternation_defers_until_true_idle() -> None:
    """A leased A cannot fund B; idle reclaim then permits B and A again."""
    a = key("target", "PIECEWISE", 4096, 8)
    b = key("target", "PIECEWISE", 8192, 16)
    controller = ElasticAdmissionController(GENERATION)
    publish(controller, a)
    controller.register(b, price=price("b"))

    active_a = controller.plan("active-a", (a,), request_bytes=0, available_bytes=0)
    controller.commit_user(active_a)
    blocked_b = controller.plan("blocked-b", (b,), request_bytes=8, available_bytes=0)
    assert blocked_b.kind == ElasticPlanKind.DEFER
    assert not blocked_b.victim_keys
    assert controller.entries[a].leases == frozenset({"active-a"})

    controller.cancel("active-a")
    reclaim_a = controller.plan_reclaim_all(
        "idle-a", request_bytes=0, available_bytes=0
    )
    assert reclaim_a.kind == ElasticPlanKind.RECLAIM
    assert reclaim_a.victim_keys == (a,)
    controller.begin_reclaim(reclaim_a)

    capture_b = controller.plan("capture-b", (b,), request_bytes=0, available_bytes=12)
    assert capture_b.kind == ElasticPlanKind.MAINTENANCE
    assert not capture_b.victim_keys
    controller.begin_maintenance(capture_b)
    controller.finish_maintenance(capture_b, {b: price("b")})

    capture_a = controller.plan(
        "capture-a-again", (a,), request_bytes=0, available_bytes=12
    )
    assert capture_a.kind == ElasticPlanKind.MAINTENANCE
    assert not capture_a.victim_keys
    controller.begin_maintenance(capture_a)
    controller.finish_maintenance(capture_a, {a: price(a.identity)})

    assert controller.entries[a].hot
    assert controller.entries[b].hot


def test_step5_x_sequence_idle_and_cold_x16_recovery() -> None:
    """Independent holdout for the exact KV6 lifecycle shape sequence."""
    xs = (32, 16, 17, 39)
    keys = {x: key("target", "PIECEWISE", 4096, x) for x in xs}
    controller = ElasticAdmissionController(GENERATION)

    for x in xs:
        graph_key = keys[x]
        plan = controller.plan(
            f"capture-x{x}",
            (graph_key,),
            request_bytes=0,
            available_bytes=12,
            destination_capture_endpoint_bytes=12,
        )
        assert plan.kind == ElasticPlanKind.MAINTENANCE
        assert not plan.victim_keys
        controller.begin_maintenance(plan)
        controller.finish_maintenance(
            plan,
            {graph_key: price(graph_key.identity)},
        )
        controller.install_reclaim_group(
            ReclaimGroup(
                graph_key.identity,
                (graph_key,),
                reclaimable_bytes=10,
            )
        )
        controller.retain_hot(plan.transaction_id, (graph_key,))
        controller.cancel(plan.transaction_id)

    reclaim = controller.plan_reclaim_all(
        "true-idle", request_bytes=0, available_bytes=0
    )
    assert reclaim.kind == ElasticPlanKind.RECLAIM
    assert set(reclaim.victim_keys) == set(keys.values())
    controller.begin_reclaim(reclaim)
    assert all(not controller.entries[item].hot for item in keys.values())

    x16 = keys[16]
    recovery = controller.plan(
        "cold-x16-recovery",
        (x16,),
        request_bytes=0,
        available_bytes=12,
        destination_capture_endpoint_bytes=12,
    )
    assert recovery.kind == ElasticPlanKind.MAINTENANCE
    assert not recovery.victim_keys
    controller.begin_maintenance(recovery)
    controller.finish_maintenance(recovery, {x16: price(x16.identity)})
    assert controller.entries[x16].hot


def test_step5_capture_cancellation_boundaries_preserve_physical_truth() -> None:
    """Cancellation before, during and after publication never invents HOT."""
    graph_key = key("target", "PIECEWISE", 4096, 16)
    controller = ElasticAdmissionController(GENERATION)
    controller.register(graph_key, price=price("capture-cancel"))

    before = controller.plan(
        "cancel-before", (graph_key,), request_bytes=0, available_bytes=12
    )
    controller.fail_maintenance(before)
    assert controller.entries[graph_key].state == GraphResidency.COLD

    during = controller.plan(
        "cancel-during", (graph_key,), request_bytes=0, available_bytes=12
    )
    controller.begin_maintenance(during)
    assert controller.entries[graph_key].state == GraphResidency.CAPTURING
    controller.fail_maintenance(during)
    assert controller.entries[graph_key].state == GraphResidency.COLD

    after = controller.plan(
        "cancel-after", (graph_key,), request_bytes=0, available_bytes=12
    )
    controller.begin_maintenance(after)
    controller.finish_maintenance(after, {graph_key: price("capture-cancel")})
    controller.retain_hot(after.transaction_id, (graph_key,))
    controller.cancel(after.transaction_id)
    assert controller.entries[graph_key].hot
    assert not controller.entries[graph_key].leases


def test_lru_pressure_evicts_only_oldest_unleased_piecewise_group() -> None:
    older = key("target", "PIECEWISE", 2048, 8)
    newer = key("target", "PIECEWISE", 4096, 8)
    pinned = key("target", "FULL", 4, 1, uniform=4)
    cold = key("mtp_prefill", "PIECEWISE", 4096, 8)
    cache = ElasticAdmissionController(GENERATION)
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
    cache = ElasticAdmissionController(GENERATION)
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


def test_capture_execute_is_separate_from_hot_hit_and_failure_is_cold() -> None:
    cold = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticAdmissionController(GENERATION)
    cold_price = price("cold", resident=9, peak=12)
    cache.register(cold, price=cold_price)
    plan = cache.plan("capture", (cold,), request_bytes=0, available_bytes=12)
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    with pytest.raises(ElasticGraphError, match="expected a user"):
        cache.commit_user(plan)

    cache.begin_maintenance(plan)
    assert cache.entries[cold].state == GraphResidency.CAPTURING
    cache.fail_maintenance(plan)
    assert cache.entries[cold].state == GraphResidency.COLD
    assert not cache.entries[cold].leases


def test_unknown_price_defers_without_mutation_or_eternal_implicit_loan() -> None:
    unknown = key("target", "PIECEWISE", 8192, 9)
    cache = ElasticAdmissionController(GENERATION)
    before = dict(cache.entries)
    plan = cache.plan("unknown", (unknown,), request_bytes=0, available_bytes=1 << 40)
    assert plan.kind == ElasticPlanKind.DEFER
    assert plan.defer_reason == "missing_exact_price_or_class_envelope"
    assert dict(cache.entries) == before


def test_class_envelope_funds_first_capture_and_returns_to_exact_price() -> None:
    unknown = key("target", "PIECEWISE", 8192, 9)
    cache = ElasticAdmissionController(GENERATION)
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
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, full, pinned=True)
    publish(cache, old_a)
    publish(cache, old_b)
    plan = cache.plan(
        "calibration",
        (cold,),
        request_bytes=32,
        available_bytes=1 << 20,
        destination_capture_endpoint_bytes=1 << 20,
        shared_resident_bytes=32,
    )
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == ()
    assert set(cache.entries) == {full, old_a, old_b}


def test_owner_set_endpoint_subtracts_only_proven_reclaim_before_capture() -> None:
    old = key("target", "PIECEWISE", 4096, 8)
    cold = key("target", "PIECEWISE", 8192, 8)
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, old)

    # The 130-byte endpoint includes the old resident set.  Only the physical
    # 10-byte reclaim proof makes the cold endpoint fit the 120-byte boundary.
    plan = cache.plan(
        "reclaim-then-capture",
        (cold,),
        request_bytes=100,
        available_bytes=120,
        destination_capture_endpoint_bytes=130,
        shared_resident_bytes=100,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == (old,)
    assert plan.reclaim_bytes == 10
    assert plan.capture_loan_bytes == 120


def test_hot_user_endpoint_deficit_defers_without_selecting_victims() -> None:
    current = key("target", "PIECEWISE", 4096, 8)
    unrelated = key("target", "PIECEWISE", 2048, 8)
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, current)
    publish(cache, unrelated)
    before = cache.snapshot

    plan = cache.plan(
        "hot-one-byte-short",
        (current,),
        request_bytes=100,
        available_bytes=100,
        destination_capture_endpoint_bytes=101,
        shared_resident_bytes=100,
    )

    assert plan.kind == ElasticPlanKind.DEFER
    assert plan.defer_reason == "hot_endpoint_requires_explicit_reclaim"
    assert plan.victim_keys == ()
    assert cache.snapshot == before


def test_post_transition_accounting_requires_destination_endpoint() -> None:
    cache = ElasticAdmissionController(GENERATION)

    with pytest.raises(ValueError, match="requires a destination capture endpoint"):
        cache.plan(
            "orphan-post-transition",
            (),
            request_bytes=0,
            available_bytes=1,
            post_transition_extra_bytes=1,
        )


def test_capture_loan_includes_retained_transition_overlap() -> None:
    cold = key("target", "PIECEWISE", 8192, 8)
    cache = ElasticAdmissionController(GENERATION)

    plan = cache.plan(
        "retained-overlap",
        (cold,),
        request_bytes=100,
        available_bytes=160,
        destination_capture_endpoint_bytes=130,
        retained_transition_overlap_bytes=20,
        shared_resident_bytes=100,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.capture_loan_bytes == 150


def test_graph_only_plan_reclaims_for_later_hot_consumer_peak() -> None:
    """Capture and HOT+MM fit separately after one exact victim selection."""
    old = key("target", "PIECEWISE", 4096, 8)
    cold = key("target", "PIECEWISE", 8192, 8)
    cache = ElasticAdmissionController(GENERATION)
    old_price = price("old", resident=100, peak=100)
    cache.publish_hot(old, old_price, pinned=False)
    cache.install_reclaim_group(ReclaimGroup("old", (old,), 100))

    plan = cache.plan(
        "capture-then-hot-mm",
        (cold,),
        request_bytes=100,
        available_bytes=150,
        destination_capture_endpoint_bytes=100,
        post_transition_extra_bytes=50,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == (old,)
    assert plan.reclaim_bytes == 100
    assert plan.capture_loan_bytes == 100


def test_graph_only_plan_distinguishes_cold_and_hot_endpoints() -> None:
    old = key("target", "PIECEWISE", 4096, 8)
    cold = key("target", "PIECEWISE", 8192, 8)
    cache = ElasticAdmissionController(GENERATION)
    old_price = price("old-asymmetric", resident=60, peak=60)
    cache.publish_hot(old, old_price, pinned=False)
    cache.install_reclaim_group(ReclaimGroup("old-asymmetric", (old,), 60))

    plan = cache.plan(
        "asymmetric-capture-then-hot-mm",
        (cold,),
        request_bytes=100,
        available_bytes=190,
        destination_capture_endpoint_bytes=140,
        post_transition_endpoint_bytes=100,
        post_transition_extra_bytes=50,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == (old,)
    assert plan.reclaim_bytes == 60
    assert plan.capture_loan_bytes == 180

    insufficient = ElasticAdmissionController(GENERATION)
    short_price = price("old-asymmetric-short", resident=59, peak=59)
    insufficient.publish_hot(old, short_price, pinned=False)
    insufficient.install_reclaim_group(ReclaimGroup("old-asymmetric-short", (old,), 59))
    entries_before = dict(insufficient.entries)
    rejected = insufficient.plan(
        "asymmetric-one-byte-short",
        (cold,),
        request_bytes=100,
        available_bytes=190,
        destination_capture_endpoint_bytes=140,
        post_transition_endpoint_bytes=100,
        post_transition_extra_bytes=50,
    )
    assert rejected.kind == ElasticPlanKind.DEFER
    assert insufficient.entries == entries_before


def test_owner_set_endpoint_adds_unshared_live_hotset_before_capture() -> None:
    cold = key("target", "FULL", 2, 2, uniform=1)
    cache = ElasticAdmissionController(GENERATION)

    plan = cache.plan(
        "cumulative-hotset",
        (cold,),
        request_bytes=600,
        available_bytes=800,
        destination_capture_endpoint_bytes=180,
        retained_transition_overlap_bytes=32,
        shared_resident_bytes=12,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.capture_loan_bytes == 800


def test_destination_intersection_cannot_exceed_either_physical_set() -> None:
    cold = key("target", "FULL", 2, 2, uniform=1)
    cache = ElasticAdmissionController(GENERATION)

    with pytest.raises(ValueError, match="current and destination sets"):
        cache.plan(
            "invalid-intersection",
            (cold,),
            request_bytes=600,
            available_bytes=800,
            destination_capture_endpoint_bytes=180,
            shared_resident_bytes=181,
        )


def test_x2_cold_endpoint_preserves_observed_transition_segment() -> None:
    cold = key("target", "FULL", 2, 2, uniform=1)
    cache = ElasticAdmissionController(GENERATION)
    current = 612_368_384
    endpoint = 180_355_072
    transition_segment = 20_971_520

    plan = cache.plan(
        "x2-underloan-regression",
        (cold,),
        request_bytes=current,
        available_bytes=633_339_904,
        destination_capture_endpoint_bytes=endpoint,
        retained_transition_overlap_bytes=transition_segment,
        shared_resident_bytes=endpoint,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.capture_loan_bytes == 633_339_904
    assert plan.capture_loan_bytes > max(current, endpoint)


def test_capture_envelope_merge_keeps_only_common_receipt_byte_proof() -> None:
    controller = ElasticAdmissionController(GENERATION)
    owner_key = (0, 3, 0, 0, 0)
    common = key("target", "PIECEWISE", 32, 8).identity
    first_only = key("mtp_prefill", "PIECEWISE", 32, 8).identity
    second_only = key("mtp_decode", "PIECEWISE", 8, 8).identity

    controller.record_capture_envelope(
        owner_key,
        (100, 8, 0),
        merge_max=True,
        resident_key_bytes=((common, 20), (first_only, 30)),
    )
    controller.record_capture_envelope(
        owner_key,
        (120, 10, 0),
        merge_max=True,
        resident_key_bytes=((common, 12), (second_only, 40)),
    )

    assert controller.capture_envelopes[owner_key] == (120, 10, 0)
    assert controller.capture_envelope_resident_key_bytes(owner_key) == ((common, 12),)

    # Direct replacement of the public compatibility view must not reuse stale
    # sidecar evidence for a different byte endpoint.
    controller.capture_envelopes[owner_key] = (121, 10, 0)
    assert controller.capture_envelope_resident_key_bytes(owner_key) is None

    with pytest.raises(ElasticGraphError, match="exceeds the destination endpoint"):
        controller.record_capture_envelope(
            (0, 3, 0, 32, 0),
            (10, 0, 0),
            resident_key_bytes=((common, 11),),
        )


def test_idle_unpin_makes_pinned_graph_reclaimable() -> None:
    pinned_key = key("target", "FULL", 7, 7, uniform=1)
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, pinned_key, pinned=True)

    assert cache.unpin_idle() == (pinned_key,)
    assert cache.entries[pinned_key].reclaimable
    reclaim = cache.plan_reclaim_all(
        "cold-epoch",
        request_bytes=10,
        available_bytes=0,
    )
    assert reclaim.kind == ElasticPlanKind.RECLAIM
    assert reclaim.victim_keys == (pinned_key,)


def test_idle_unpin_preserves_pinned_graph_without_physical_proof() -> None:
    pinned_key = key("target", "FULL", 7, 7, uniform=1)
    cache = ElasticAdmissionController(GENERATION)
    cache.publish_hot(pinned_key, price("retained-pool"), pinned=True)

    assert cache.unpin_idle() == ()
    assert cache.entries[pinned_key].pinned


def test_idle_unpin_rejects_leased_graph() -> None:
    pinned_key = key("target", "FULL", 9, 9, uniform=1)
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, pinned_key, pinned=True)
    plan = cache.plan("user", (pinned_key,), request_bytes=10, available_bytes=20)
    cache.commit_user(plan)

    with pytest.raises(ElasticGraphError, match="cannot unpin an active"):
        cache.unpin_idle()
    assert cache.entries[pinned_key].pinned
    assert cache.entries[pinned_key].leases == frozenset({"user"})


def test_pressure_reclaim_plans_pinned_and_evictable_without_early_mutation() -> None:
    pinned_key = key("target", "FULL", 39, 39, uniform=1)
    evictable_key = key("target", "PIECEWISE", 156, 39)
    controller = ElasticAdmissionController(GENERATION)
    publish(controller, pinned_key, pinned=True)
    publish(controller, evictable_key)
    controller.resident_bytes = 100
    controller.floor_bytes = 12

    plan = controller.plan_pressure_reclaim_all(
        "pressure-x39", request_bytes=100, available_bytes=0
    )

    assert plan.kind == ElasticPlanKind.PRESSURE_RECLAIM
    assert plan.victim_keys == (evictable_key, pinned_key)
    assert plan.reclaim_groups == ()
    assert plan.capture_loan_bytes == 12
    assert plan.reclaim_bytes == 88
    assert controller.entries[pinned_key].pinned
    assert controller.entries[evictable_key].reclaimable

    controller.begin_reclaim(plan)
    assert not controller.entries[pinned_key].hot
    assert not controller.entries[evictable_key].hot


def test_pressure_reclaim_rejects_leased_pinned_set_without_mutation() -> None:
    pinned_key = key("target", "FULL", 39, 39, uniform=1)
    controller = ElasticAdmissionController(GENERATION)
    publish(controller, pinned_key, pinned=True)
    controller.retain_hot("active", (pinned_key,))

    plan = controller.plan_pressure_reclaim_all(
        "leased", request_bytes=10, available_bytes=0
    )

    assert plan.kind == ElasticPlanKind.DEFER
    assert plan.defer_reason == "pressure_reclaim_has_active_graph_lease"
    assert not plan.victim_keys
    assert controller.entries[pinned_key].pinned
    assert controller.entries[pinned_key].leases == frozenset({"active"})


def test_pressure_reclaim_rejects_stale_protected_key_without_mutation() -> None:
    pinned_key = key("target", "FULL", 39, 39, uniform=1)
    controller = ElasticAdmissionController(GENERATION)
    publish(controller, pinned_key, pinned=True)
    stale = replace(
        pinned_key,
        generation=RuntimeGeneration("stale-idle-reclaim-generation"),
    )
    before = controller.snapshot

    with pytest.raises(ElasticGraphError, match="stale generation"):
        controller.plan_pressure_reclaim_all(
            "pressure-reclaim-stale-protected",
            request_bytes=1,
            available_bytes=0,
            protected_keys=(stale,),
        )

    assert controller.snapshot == before


def test_no_deficit_miss_never_selects_a_victim() -> None:
    old = key("target", "PIECEWISE", 8, 8)
    new = key("target", "PIECEWISE", 16, 16)
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, old)
    cache.register(new, price=price("new", resident=20, peak=24))

    plan = cache.plan(
        "capture-with-tail",
        (new,),
        request_bytes=10,
        available_bytes=40,
        destination_capture_endpoint_bytes=24,
    )

    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.victim_keys == ()
    assert plan.capture_order == (new,)
    assert cache.entries[old].hot


def test_residency_receipt_allows_pinned_proof_but_rejects_active_lease() -> None:
    graph_key = key("target", "PIECEWISE", 8, 8)
    pinned = ElasticResidencyEntry(
        key=graph_key,
        pinned=True,
        resident_bytes=16,
        local_pool_bytes=16,
        reclaimable_bytes=16,
    )
    assert pinned.pinned and pinned.reclaimable_bytes == 16
    with pytest.raises(ValueError, match="leased Graph entry"):
        ElasticResidencyEntry(
            key=graph_key,
            pinned=False,
            resident_bytes=16,
            local_pool_bytes=16,
            reclaimable_bytes=16,
            lease_ids=("active",),
        )

    stale = replace(graph_key, generation=RuntimeGeneration("stale"))
    entry = ElasticResidencyEntry(
        key=stale,
        pinned=False,
        resident_bytes=16,
        local_pool_bytes=16,
        reclaimable_bytes=16,
    )
    with pytest.raises(ValueError, match="mixes runtime generations"):
        ElasticResidencyReceipt(
            generation=GENERATION,
            transaction_id=None,
            resident_bytes=16,
            floor_bytes=0,
            transition_floor_bytes=0,
            peak_bytes=16,
            cublas_workspace_bytes=0,
            entries=(entry,),
        )

    with pytest.raises(ValueError, match="partial.*forbidden"):
        ElasticResidencyReceipt(
            generation=GENERATION,
            transaction_id=None,
            resident_bytes=0,
            floor_bytes=0,
            transition_floor_bytes=0,
            peak_bytes=0,
            cublas_workspace_bytes=0,
            entries=(),
            complete=False,
        )

    with pytest.raises(ValueError, match="local pool exceeds"):
        ElasticResidencyEntry(
            key=graph_key,
            pinned=False,
            resident_bytes=16,
            local_pool_bytes=17,
            reclaimable_bytes=0,
        )
    with pytest.raises(ValueError, match="reclaim proof exceeds the local pool"):
        ElasticResidencyEntry(
            key=graph_key,
            pinned=False,
            resident_bytes=16,
            local_pool_bytes=8,
            reclaimable_bytes=9,
        )
    with pytest.raises(ValueError, match="entries exceed aggregate"):
        ElasticResidencyReceipt(
            generation=GENERATION,
            transaction_id=None,
            resident_bytes=15,
            floor_bytes=0,
            transition_floor_bytes=0,
            peak_bytes=16,
            cublas_workspace_bytes=0,
            entries=(pinned,),
        )


def test_runtime_config_is_single_fail_closed_activation_authority() -> None:
    config = SimpleNamespace(
        additional_config={"elastic_gdn_backing": True},
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL")),
    )
    assert ElasticRuntimeConfig.from_vllm_config(config).enabled
    config.model_config.enforce_eager = True
    assert not ElasticRuntimeConfig.from_vllm_config(config).enabled


def test_measured_deficit_defers_while_only_victim_is_leased() -> None:
    active = key("target", "PIECEWISE", 8, 8)
    cold = key("target", "PIECEWISE", 16, 16)
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, active)
    cache.register(cold, price=price("cold", resident=20, peak=24))
    active_plan = cache.plan(
        "active-a", (active,), request_bytes=10, available_bytes=40
    )
    cache.commit_user(active_plan)

    deferred = cache.plan(
        "defer-b",
        (cold,),
        request_bytes=10,
        available_bytes=20,
        destination_capture_endpoint_bytes=30,
    )

    assert deferred.kind == ElasticPlanKind.DEFER
    assert deferred.defer_reason == "insufficient_reclaimable_graph_and_kv_bytes"
    assert cache.entries[active].hot
    assert cache.entries[active].leases == frozenset({"active-a"})


def test_true_idle_reclaim_then_capture_preserves_progress() -> None:
    old = key("target", "PIECEWISE", 8, 8)
    new = key("target", "PIECEWISE", 16, 16)
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, old)
    cache.release("absent")

    reclaim = cache.plan_reclaim_all(
        "idle-reclaim", request_bytes=10, available_bytes=0
    )
    assert reclaim.kind == ElasticPlanKind.RECLAIM
    cache.begin_reclaim(reclaim)
    cache.synchronize_hot(())

    cache.register(new, price=price("new", resident=20, peak=24))
    capture = cache.plan(
        "capture-b",
        (new,),
        request_bytes=0,
        available_bytes=24,
        destination_capture_endpoint_bytes=24,
    )
    assert capture.kind == ElasticPlanKind.MAINTENANCE
    assert capture.victim_keys == ()


def test_observability_is_aggregate_and_trace_is_bounded() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cache = ElasticAdmissionController(GENERATION)
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
    cache = ElasticAdmissionController(GENERATION)
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
    cache = ElasticAdmissionController(GENERATION)
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
    cache = ElasticAdmissionController(GENERATION)
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
    cache = ElasticAdmissionController(GENERATION)
    publish(cache, graph_key)
    plan = cache.plan("tail", (graph_key,), request_bytes=0, available_bytes=0)
    cache.commit_user(plan)
    cache.release("tail")
    assert cache.entries[graph_key].hot
    assert cache.entries[graph_key].reclaimable


def test_deferred_free_blocks_reclaim_until_physical_completion() -> None:
    graph_key = key("target", "PIECEWISE", 4096, 8)
    cold = key("mtp_prefill", "PIECEWISE", 4096, 8)
    cache = ElasticAdmissionController(GENERATION)
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
    cache = ElasticAdmissionController(GENERATION)
    shared = price("shared-pool", resident=10, peak=12)
    cache.publish_hot(first, shared, pinned=False)
    cache.publish_hot(second, shared, pinned=False)
    cache.install_reclaim_group(ReclaimGroup("shared-pool", (first, second), 14))
    cache.register(cold, price=price("cold", resident=9, peak=12))

    plan = cache.plan("group", (cold,), request_bytes=2, available_bytes=0)
    assert plan.kind == ElasticPlanKind.MAINTENANCE
    assert plan.reclaim_bytes == 14
    assert plan.victim_keys == (first, second)
