"""Formal review round 2: CR cursor alignment and channel-map totality."""

from __future__ import annotations

import pytest

from tpen.execution.chain_resume import (
    ChannelEvent,
    ChannelMapError,
    ChannelParityError,
    ProcessEvidence,
    compare_channel_parity,
    e5_channel_map,
)


def _evidence(phase: str, pid: int, steps: tuple[int, ...]) -> ProcessEvidence:
    events = tuple(
        event
        for step in steps
        for event in (
            ChannelEvent(
                channel=spec.channel,
                step=step,
                event_kind=spec.event_kind,
                value=f"{spec.channel}:same",
            )
            for spec in e5_channel_map()
        )
    )
    return ProcessEvidence(
        phase=phase,
        pid=pid,
        seed=1,
        seed_source=f"{phase}:seed-source",
        command=("native", phase),
        returncode=0,
        events=events,
    )


def test_delayed_first_resumed_events_are_rejected() -> None:
    """RED at CR: equal values at different first steps are not parity."""

    reference = _evidence("reference", 101, (3,))
    resumed = _evidence("recovery", 102, (4,))
    with pytest.raises(ChannelParityError):
        compare_channel_parity(reference, resumed, resume_step=2)


def test_replayed_cursor_prefix_is_rejected_even_with_matching_step_three() -> None:
    """RED at CR: replayed cursor events cannot be filtered into a green result."""

    reference = _evidence("reference", 103, (3,))
    resumed = _evidence("recovery", 104, (0, 1, 2, 3))
    with pytest.raises(ChannelParityError):
        compare_channel_parity(reference, resumed, resume_step=2)


def test_exact_first_step_parity_is_accepted() -> None:
    """GREEN control: both fresh processes first emit the selected step."""

    reference = _evidence("reference", 105, (3,))
    resumed = _evidence("recovery", 106, (3,))
    result = compare_channel_parity(reference, resumed, resume_step=2)
    assert set(result) == {spec.channel for spec in e5_channel_map()}
    assert all(result.values())


def test_empty_explicit_channel_map_is_refused() -> None:
    """RED candidate at CR: explicit empty channel maps cannot mean total."""

    reference = _evidence("reference", 107, (3,))
    resumed = _evidence("recovery", 108, (3,))
    with pytest.raises(ChannelMapError):
        compare_channel_parity(reference, resumed, resume_step=2, channel_map=())
