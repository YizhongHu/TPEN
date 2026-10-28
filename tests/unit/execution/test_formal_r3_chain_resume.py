"""Formal review round 3: CR replay-guard phase gating and cursor attribution."""

from __future__ import annotations

import pytest

from tpen.execution.chain_resume import (
    ChannelEvent,
    ChannelParityError,
    ProcessEvidence,
    compare_channel_parity,
    e5_channel_map,
)


def _evidence(phase: str, pid: int, steps: tuple[int, ...]) -> ProcessEvidence:
    events = tuple(
        ChannelEvent(
            channel=spec.channel,
            step=step,
            event_kind=spec.event_kind,
            value=f"{spec.channel}:same",
        )
        for step in steps
        for spec in e5_channel_map()
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


def test_replayed_prefix_rejected_for_any_resumed_phase_label() -> None:
    """An unrecognized phase label must not disable the replay guard."""

    reference = _evidence("reference", 201, (3,))
    resumed = _evidence("restored", 202, (0, 1, 2, 3))
    with pytest.raises(ChannelParityError):
        compare_channel_parity(reference, resumed, resume_step=2)


def test_replayed_prefix_rejected_for_recovery_phase_control() -> None:
    """The recognized recovery label continues rejecting replay."""

    reference = _evidence("reference", 203, (3,))
    resumed = _evidence("recovery", 204, (0, 1, 2, 3))
    with pytest.raises(ChannelParityError):
        compare_channel_parity(reference, resumed, resume_step=2)


def test_trainer_cursor_restart_signature_names_the_cursor_channel() -> None:
    """A cursor-omission restart is attributed to trainer_cursor."""

    reference = _evidence("reference", 205, (3,))
    resumed = _evidence("recovery", 206, (0, 1, 2, 3))
    with pytest.raises(ChannelParityError, match="trainer_cursor"):
        compare_channel_parity(
            reference,
            resumed,
            resume_step=2,
            disabled_limbs=("trainer_cursor",),
        )
