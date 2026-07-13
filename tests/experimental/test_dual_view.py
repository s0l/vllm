import pytest

from vllm.experimental.dual_view import (
    choose_dual_view_prefill_phase,
    qwopus_tp2_to_tp3_gdn_moves,
)


@pytest.mark.parametrize(
    "prefill,decode,waiting,step,expected",
    [
        (False, False, False, 1, None),
        (True, False, False, 1, True),
        (False, False, True, 1, True),
        (False, True, False, 1, False),
        (True, True, False, 1, False),
        (True, True, False, 2, True),
        (False, True, True, 3, False),
        (False, True, True, 4, True),
    ],
)
def test_choose_dual_view_prefill_phase(
    prefill, decode, waiting, step, expected
):
    assert choose_dual_view_prefill_phase(
        current_step=step,
        cadence=2,
        has_running_prefill=prefill,
        has_running_decode=decode,
        has_waiting_work=waiting,
    ) is expected


def test_choose_dual_view_prefill_phase_rejects_zero_cadence():
    with pytest.raises(ValueError):
        choose_dual_view_prefill_phase(
            current_step=1,
            cadence=0,
            has_running_prefill=True,
            has_running_decode=True,
            has_waiting_work=False,
        )


def test_qwopus_gdn_transition_moves_only_middle_heads_to_slow_rank():
    moves = qwopus_tp2_to_tp3_gdn_moves()
    assert [(m.source_rank, m.target_rank, m.key_head_start, m.key_head_count)
            for m in moves] == [(0, 2, 6, 2), (1, 2, 8, 3)]
    assert sum(move.key_head_count for move in moves) == 5
