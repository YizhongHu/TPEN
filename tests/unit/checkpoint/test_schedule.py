"""Unit tests for semantic checkpoint schedules."""

from __future__ import annotations

import pytest

from tpen.checkpoint import CheckpointSchedule, EveryNUpdates, ExplicitUpdates, TerminalOnly


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        (TerminalOnly(), [False, False, False]),
        (EveryNUpdates(3), [False, False, True]),
        (ExplicitUpdates([2, 5]), [False, True, False]),
    ],
)
def test_periodic_decisions_use_completed_updates(
    schedule: CheckpointSchedule, expected: list[bool]
) -> None:
    """Each schedule selects durable update counts, not event occurrences."""

    assert [schedule.should_run(update) for update in (1, 2, 3)] == expected


@pytest.mark.parametrize(
    "schedule",
    [TerminalOnly(), EveryNUpdates(3), ExplicitUpdates([2, 5])],
)
def test_terminal_decision_is_explicit_and_always_selected(
    schedule: CheckpointSchedule,
) -> None:
    """Periodic cadence cannot suppress a terminal publication."""

    assert schedule.should_run(0, terminal=True)
    assert schedule.should_run(1, terminal=True)


def test_every_n_updates_aligns_after_resume_without_run_local_state() -> None:
    schedule = EveryNUpdates(3)

    # A resumed trainer at four completed updates remains in the same phase;
    # its next selected update is six, not the third callback occurrence.
    assert [schedule.should_run(update) for update in range(4, 8)] == [
        False,
        False,
        True,
        False,
    ]


def test_explicit_updates_normalize_order_and_duplicates() -> None:
    schedule = ExplicitUpdates([5, 2, 5, 2])

    assert schedule.updates == frozenset({2, 5})
    assert [schedule.should_run(update) for update in range(1, 7)] == [
        False,
        True,
        False,
        False,
        True,
        False,
    ]


def test_zero_completed_updates_are_terminal_only() -> None:
    schedules = (TerminalOnly(), EveryNUpdates(1), ExplicitUpdates([1]))

    assert [schedule.should_run(0) for schedule in schedules] == [False, False, False]
    assert [schedule.should_run(0, terminal=True) for schedule in schedules] == [
        True,
        True,
        True,
    ]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: EveryNUpdates(0), "every_n"),
        (lambda: EveryNUpdates(), "every_n"),
        (lambda: EveryNUpdates(True), "every_n"),
        (lambda: ExplicitUpdates([0]), "updates entry"),
        (lambda: ExplicitUpdates([1, True]), "updates entry"),
    ],
)
def test_invalid_schedule_boundaries_fail_loudly(factory, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


@pytest.mark.parametrize(
    "schedule", [EveryNUpdates(2), ExplicitUpdates([2]), TerminalOnly()]
)
def test_negative_decision_inputs_fail_loudly(schedule: CheckpointSchedule) -> None:
    """Schedules reject invalid update values they own."""

    with pytest.raises(ValueError, match="completed_updates"):
        schedule.should_run(-1)
