# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner

pytestmark = pytest.mark.cpu_test


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
        finished_req_ids=set(),
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
