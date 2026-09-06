# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.core.elastic_graph import canonical_execution_request_order
from vllm.v1.worker.gpu.model_runner import (
    ExecuteModelState,
    GPUModelRunner,
    _elastic_input_staging_fingerprint,
    _elastic_mm_staging_fingerprint,
    _elastic_new_request_execution_prefill_len,
    _elastic_new_request_prompt_len,
    _prepare_elastic_local_staging_with_consensus,
    _stage_elastic_mm_inputs_with_consensus,
    _validate_elastic_materialized_input_batch,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState

pytestmark = pytest.mark.cpu_test


def test_complete_phase_sequence_is_checked_before_model_forward():
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(GPUModelRunner.execute_model)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    lines = lambda name: [
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == name
    ]
    checked = lines("validate_post_materialization_completion")
    forwards = lines("forward_start") + lines("run_fullgraph")
    assert len(checked) == 1
    assert forwards and checked[0] < min(forwards)


def test_local_staging_failure_still_enters_post_materialization_vote():
    prepare_inputs = Mock(side_effect=RuntimeError("rank-local copy failed"))
    working_set = Mock()
    working_set.require_post_materialization_consensus.side_effect = RuntimeError(
        "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH: all-rank rejection"
    )
    plan = Mock()

    with pytest.raises(RuntimeError, match="all-rank rejection"):
        _prepare_elastic_local_staging_with_consensus(
            prepare_inputs,
            plan,
            working_set,
        )

    working_set.require_post_materialization_consensus.assert_called_once()
    call = working_set.require_post_materialization_consensus.call_args
    assert call.args == (plan,)
    assert "rank-local copy failed" in call.kwargs["validation_error"]
    assert call.kwargs["phase"] == "post_materialization"


@pytest.mark.parametrize(
    "stage",
    ["sampling_indices", "prepare_attn", "preprocess_state", "lora"],
)
def test_asymmetric_local_staging_failure_prevents_first_collective(stage: str):
    first_model_collective = Mock()
    working_set = Mock()
    working_set.require_post_materialization_consensus.side_effect = RuntimeError(
        "ELASTIC_POST_MUTATION_OBSERVER_MISMATCH: all-rank rejection"
    )

    def fail_local_stage():
        raise RuntimeError(f"rank-local {stage} failed")

    with pytest.raises(RuntimeError, match="all-rank rejection"):
        _prepare_elastic_local_staging_with_consensus(
            fail_local_stage,
            Mock(),
            working_set,
        )
        first_model_collective()

    first_model_collective.assert_not_called()
    assert (
        stage
        in (
            working_set.require_post_materialization_consensus.call_args.kwargs[
                "validation_error"
            ]
        )
    )


def test_mm_model_inputs_stage_post_encoder_state_before_embedding_vote():
    sentinel = object()

    class EncoderDecoderState:
        def __init__(self):
            self.encoder_outputs = [sentinel]

        def prepare_inputs(self, _input_batch, _req_states):
            outputs = self.encoder_outputs
            self.encoder_outputs = []
            return {"encoder_outputs": outputs}

        def stage_mm_embeddings(self, _input_batch, _req_states):
            return sentinel

        def commit_staged_mm_encoder(self, _completed):
            return None

    state = EncoderDecoderState()
    working_set = Mock()

    staged, prepared = _stage_elastic_mm_inputs_with_consensus(
        state,
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        working_set,
    )
    assert staged is sentinel
    assert prepared == {"encoder_outputs": [sentinel]}
    assert state.encoder_outputs == []
    assert (
        working_set.require_post_materialization_consensus.call_args.kwargs["phase"]
        == "post_mm_materialization"
    )


def test_mm_staging_fingerprint_binds_tensor_geometry_and_mask():
    prepared = {"positions": torch.zeros(1)}
    first = (
        ([torch.zeros(2, 4, dtype=torch.float16)], torch.tensor([True, False])),
        prepared,
    )
    second = (
        ([torch.zeros(1, 8, dtype=torch.float16)], torch.tensor([True, False])),
        prepared,
    )

    assert _elastic_mm_staging_fingerprint(first) != _elastic_mm_staging_fingerprint(
        second
    )
    shifted_mask = (
        ([torch.zeros(2, 4, dtype=torch.float16)], torch.tensor([False, True])),
        prepared,
    )
    assert _elastic_mm_staging_fingerprint(first) != _elastic_mm_staging_fingerprint(
        shifted_mask
    )


def test_input_staging_fingerprint_expands_grouped_encoder_items():
    staged = (
        None,
        None,
        None,
        None,
        None,
        {},
        True,
        (
            ["hash-a", "hash-b"],
            [("image", 2, {"pixels": torch.zeros(2, 3, 4, 4)})],
        ),
        None,
    )

    assert len(_elastic_input_staging_fingerprint(staged)) == 64


def test_rank_local_cached_encoder_omission_rejects_declared_work():
    feature = SimpleNamespace(identifier="image-hash", modality="image", data=object())
    state = SimpleNamespace(
        encoder_cache=SimpleNamespace(mm_features={"req0": [feature]})
    )

    with pytest.raises(RuntimeError, match="differs from scheduler-declared work"):
        ModelState.validate_staged_mm_encoder(
            state,
            {"req0": [0]},
            ([], []),
        )

    ModelState.validate_staged_mm_encoder(
        state,
        {"req0": [0]},
        (["image-hash"], [("image", 1, object())]),
    )


def test_encoder_validator_skips_text_request_without_mm_cache_entry():
    feature = SimpleNamespace(identifier="image-hash", modality="image", data=object())
    state = SimpleNamespace(
        encoder_cache=SimpleNamespace(mm_features={"vision": [feature]})
    )

    ModelState.validate_staged_mm_encoder(
        state,
        {"vision": [0]},
        (["image-hash"], [("image", 1, object())]),
        ["text", "vision"],
    )


def test_mm_overlap_peak_does_not_double_count_retained_growth():
    assert GPUModelRunner._elastic_mm_overlap_peak(64, 40) == 104
    # The 40-byte observed delta may be W16 retained growth plus A24 transient;
    # the physical endpoint is still baseline64 + delta40, not final80 + delta40.
    assert GPUModelRunner._elastic_mm_overlap_peak(64, 16 + 24) == 104


def _batch(query_start_loc: list[int], cu_num_logits: list[int]):
    return SimpleNamespace(
        num_reqs=len(query_start_loc) - 1,
        query_start_loc_np=np.asarray(query_start_loc, dtype=np.int32),
        cu_num_logits_np=np.asarray(cu_num_logits, dtype=np.int32),
    )


def test_expected_sampling_indices_single_token_prefill():
    indices = GPUModelRunner._expected_sampling_indices(_batch([0, 1], [0, 1]))

    np.testing.assert_array_equal(indices, np.asarray([0], dtype=np.int64))


def test_expected_sampling_indices_mixed_speculative_rows():
    indices = GPUModelRunner._expected_sampling_indices(_batch([0, 1, 5], [0, 1, 4]))

    np.testing.assert_array_equal(indices, np.asarray([0, 2, 3, 4], dtype=np.int64))


def test_expected_sampling_indices_rejects_more_logits_than_query_rows():
    with pytest.raises(RuntimeError, match="invalid CPU sampling-index contract"):
        GPUModelRunner._expected_sampling_indices(_batch([0, 1], [0, 2]))


def test_execute_state_carries_elastic_transaction_through_sampling_boundary():
    state = ExecuteModelState(
        input_batch=None,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        dp_sync=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        routed_experts=None,
        num_spec_tokens_to_schedule=0,
        gdn_checkpoint_keys=None,
        elastic_external_memory_bytes=0,
        elastic_external_memory_floor_bytes=0,
        elastic_mm_activation_loan_bytes=0,
        elastic_dynamic_graph_step_started=False,
        elastic_transaction_id=None,
        elastic_step_plan=None,
        is_synthetic_warmup=True,
    )

    assert state.elastic_transaction_id is None
    assert state.is_synthetic_warmup


def test_elastic_resumed_final_replay_uses_execution_prefill_boundary():
    request = SimpleNamespace(
        prompt_token_ids=[11, 12, 13],
        prompt_embeds=None,
        prefill_token_ids=[11, 12, 13, 14],
        execution_prefill_len=4,
        num_computed_tokens=3,
    )

    prefill_len = _elastic_new_request_execution_prefill_len(request)
    assert _elastic_new_request_prompt_len(request) == 3
    assert prefill_len == 4
    assert canonical_execution_request_order(
        {"decode": 1, "resumed-final-replay": 1},
        is_prefilling_by_request={
            "decode": False,
            "resumed-final-replay": request.num_computed_tokens < prefill_len,
        },
        decode_query_len=4,
    ) == ("decode", "resumed-final-replay")


def test_elastic_new_request_requires_prompt_identity():
    request = SimpleNamespace(
        prompt_token_ids=None,
        prompt_embeds=None,
        prefill_token_ids=[11],
    )

    with pytest.raises(
        RuntimeError, match="Neither prompt_token_ids nor prompt_embeds"
    ):
        _elastic_new_request_prompt_len(request)


def test_elastic_new_request_accepts_prompt_embeds_only():
    request = SimpleNamespace(
        prompt_token_ids=None,
        prompt_embeds=torch.zeros((3, 8)),
        prefill_token_ids=None,
    )

    assert _elastic_new_request_prompt_len(request) == 3


def test_elastic_new_request_ids_and_embeds_share_one_prompt_boundary():
    request = SimpleNamespace(
        prompt_token_ids=[11, 12, 13],
        prompt_embeds=torch.zeros((3, 8)),
        prefill_token_ids=[11, 12, 13],
    )

    assert _elastic_new_request_prompt_len(request) == 3


def test_elastic_new_request_rejects_ids_embeds_length_mismatch():
    request = SimpleNamespace(
        prompt_token_ids=[11, 12],
        prompt_embeds=torch.zeros((3, 8)),
        prefill_token_ids=[11, 12],
    )

    with pytest.raises(RuntimeError, match="different lengths"):
        _elastic_new_request_prompt_len(request)


def test_elastic_new_request_uses_explicit_resumed_prefill_boundary():
    request = SimpleNamespace(
        prompt_token_ids=[11, 12, 13],
        prompt_embeds=None,
        prefill_token_ids=[11, 12, 13, 14, 15],
        execution_prefill_len=5,
        num_computed_tokens=3,
    )

    prefill_len = _elastic_new_request_execution_prefill_len(request)

    assert prefill_len == 5
    assert request.num_computed_tokens < prefill_len


def test_elastic_new_request_accepts_embeds_only_execution_boundary():
    request = SimpleNamespace(
        prompt_token_ids=None,
        prompt_embeds=torch.zeros((3, 8)),
        prefill_token_ids=[0, 0, 0],
        execution_prefill_len=3,
    )

    assert _elastic_new_request_execution_prefill_len(request) == 3


@pytest.mark.parametrize("execution_prefill_len", [None, True, 3.0, "3"])
def test_elastic_new_request_rejects_missing_or_noninteger_execution_boundary(
    execution_prefill_len,
):
    request = SimpleNamespace(
        prompt_token_ids=[11, 12, 13],
        prompt_embeds=None,
        prefill_token_ids=[11, 12, 13],
        execution_prefill_len=execution_prefill_len,
    )

    with pytest.raises(RuntimeError, match="integer execution_prefill_len"):
        _elastic_new_request_execution_prefill_len(request)


@pytest.mark.parametrize(
    ("execution_prefill_len", "message"),
    [(2, "smaller than prompt length"), (4, "exceeds transported token stream")],
)
def test_elastic_new_request_rejects_out_of_bounds_execution_boundary(
    execution_prefill_len,
    message,
):
    request = SimpleNamespace(
        prompt_token_ids=[11, 12, 13],
        prompt_embeds=None,
        prefill_token_ids=[11, 12, 13],
        execution_prefill_len=execution_prefill_len,
    )

    with pytest.raises(RuntimeError, match=message):
        _elastic_new_request_execution_prefill_len(request)


def test_elastic_new_request_rejects_missing_prefill_stream_before_admission():
    request = SimpleNamespace(
        prompt_token_ids=None,
        prompt_embeds=torch.zeros((3, 8)),
        prefill_token_ids=None,
        execution_prefill_len=3,
    )

    with pytest.raises(RuntimeError, match="omitted the transported prefill"):
        _elastic_new_request_execution_prefill_len(request)


def test_add_requests_wires_execution_prefill_boundary_to_request_state():
    req_states = SimpleNamespace(
        req_id_to_index={},
        add_request=Mock(),
        apply_staged_writes=Mock(),
    )

    def add_request(**kwargs):
        req_states.req_id_to_index[kwargs["req_id"]] = 7

    req_states.add_request.side_effect = add_request
    runner = SimpleNamespace(
        _remove_request=Mock(return_value=False),
        req_states=req_states,
        pooling_runner=None,
        encoder_cache=None,
        model_state=Mock(),
        block_tables=Mock(),
        lora_state=Mock(),
        is_last_pp_rank=False,
        sampler=None,
        prompt_logprobs_worker=None,
        adaptive_verification=None,
    )
    request = SimpleNamespace(
        req_id="resumed",
        prompt_token_ids=[11, 12, 13],
        prompt_embeds=None,
        prefill_token_ids=[11, 12, 13, 14, 15],
        execution_prefill_len=5,
        num_computed_tokens=3,
        sampling_params=None,
        pooling_params=None,
        mm_features=[],
        block_ids=([0],),
        lora_request=None,
    )

    GPUModelRunner.add_requests(
        runner,
        SimpleNamespace(scheduled_new_reqs=[request]),
    )

    req_states.add_request.assert_called_once_with(
        req_id="resumed",
        prompt_len=3,
        all_token_ids=[11, 12, 13, 14, 15],
        num_computed_tokens=3,
        max_tokens=1,
        execution_prefill_len=5,
    )


def test_add_requests_accepts_embeds_only_prompt_identity():
    req_states = SimpleNamespace(
        req_id_to_index={},
        add_request=Mock(),
        apply_staged_writes=Mock(),
    )

    def add_request(**kwargs):
        req_states.req_id_to_index[kwargs["req_id"]] = 9

    req_states.add_request.side_effect = add_request
    runner = SimpleNamespace(
        _remove_request=Mock(return_value=False),
        req_states=req_states,
        pooling_runner=None,
        encoder_cache=None,
        model_state=Mock(),
        block_tables=Mock(),
        lora_state=Mock(),
        is_last_pp_rank=False,
        sampler=None,
        prompt_logprobs_worker=None,
        adaptive_verification=None,
    )
    prompt_embeds = torch.zeros((3, 8))
    request = SimpleNamespace(
        req_id="embeds-only",
        prompt_token_ids=None,
        prompt_embeds=prompt_embeds,
        prefill_token_ids=[0, 0, 0],
        execution_prefill_len=3,
        num_computed_tokens=0,
        sampling_params=None,
        pooling_params=None,
        mm_features=[],
        block_ids=([0],),
        lora_request=None,
    )

    GPUModelRunner.add_requests(
        runner,
        SimpleNamespace(scheduled_new_reqs=[request]),
    )

    req_states.add_request.assert_called_once_with(
        req_id="embeds-only",
        prompt_len=3,
        all_token_ids=[0, 0, 0],
        num_computed_tokens=0,
        max_tokens=1,
        execution_prefill_len=3,
    )


def test_materialized_input_batch_preserves_mixed_row_prefill_phases():
    manifest = SimpleNamespace(
        request_ids=("decode", "resumed-final-replay"),
        per_request_query_lens=(1, 1),
        per_request_is_prefilling=(False, True),
        scheduled_draft_rows=(0, 0),
    )
    input_batch = SimpleNamespace(
        req_ids=["decode", "resumed-final-replay"],
        num_reqs=2,
        num_scheduled_tokens=np.asarray([1, 1], dtype=np.int32),
        is_prefilling_np=np.asarray([False, True], dtype=np.bool_),
        num_draft_tokens_per_req=None,
    )

    _validate_elastic_materialized_input_batch(manifest, input_batch)

    input_batch.is_prefilling_np = np.asarray([False, False], dtype=np.bool_)
    with pytest.raises(RuntimeError, match="planned_prefill=.*False, True"):
        _validate_elastic_materialized_input_batch(manifest, input_batch)
