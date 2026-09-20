# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU controls for logical aliases of measured physical replay endpoints."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler


def scheduler(rows, physical, measured=None):
    result = object.__new__(Scheduler)
    result._elastic_admission_controller = SimpleNamespace(
        measured_bytes={} if measured is None else measured
    )
    result._elastic_graph_catalog = rows
    result._resolve_elastic_step_physical_keys = Mock(side_effect=physical.__getitem__)
    return result


def test_hot_alias_uses_maximum_complete_aggregate_without_mutation():
    rows = {
        (2,): {"hot_peak_bytes": 140, "cold_peak_bytes": 90},
        (4,): {"hot_peak_bytes": 120, "cold_peak_bytes": 180},
    }
    s = scheduler(rows, {(2,): ("draft1",), (4,): ("draft1",), (1024,): ("draft1",)})
    assert s._elastic_hot_replay_envelope((1024,)) == 140
    assert s._elastic_admission_controller.measured_bytes == {}
    assert len(rows) == 2


@pytest.mark.parametrize("measured", [0, 70, 200])
def test_exact_logical_measurement_takes_precedence(measured):
    s = scheduler({}, {}, {(1024,): measured})
    assert s._elastic_hot_replay_envelope((1024,)) == measured
    s._resolve_elastic_step_physical_keys.assert_not_called()


@pytest.mark.parametrize(
    "other", [("draft2",), ("draft1", "target"), (), ("other-generation-draft1",)]
)
def test_no_borrowing_across_physical_geometry_owner_set_or_generation(other):
    s = scheduler({(2,): {"hot_peak_bytes": 140}}, {(2,): other, (1024,): ("draft1",)})
    assert s._elastic_hot_replay_envelope((1024,)) == 0


@pytest.mark.parametrize("row", [{}, {"cold_peak_bytes": 140}, {"hot_peak_bytes": 0}])
def test_missing_hot_evidence_is_unknown_even_with_cold_measurement(row):
    s = scheduler({(2,): row}, {(2,): ("draft1",), (1024,): ("draft1",)})
    assert s._elastic_hot_replay_envelope((1024,)) == 0


def test_compiled_only_empty_set_does_not_alias_unrelated_empty_rows():
    s = scheduler({(2,): {"hot_peak_bytes": 140}}, {(2,): (), (1024,): ()})
    assert s._elastic_hot_replay_envelope((1024,)) == 0
