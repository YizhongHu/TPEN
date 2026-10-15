"""Formal review round 3: CR replay-guard phase gating and cursor attribution.

Target: CR @ 596033ecff7ca37a5c8ff5e5cda635c9b9a3ab60.

Polarity at that head (preregistered):
- test_replayed_prefix_rejected_for_any_resumed_phase_label: EXPECTED RED.
  The replayed-prefix guard in ``compare_channel_parity`` applies only when
  ``resumed.phase.lower()`` is exactly ``recovery`` or ``resumed``; any other
  caller-supplied label (``restored``, ``successor``, ...) silently skips the
  guard, so a replayed prefix passes as parity — fail-open on free text.
- test_replayed_prefix_rejected_for_recovery_phase_control: EXPECTED GREEN
  (pin; the recognized label keeps rejecting the replay).
- test_trainer_cursor_restart_signature_names_the_cursor_channel: EXPECTED
  RED.  A cursor-omission restart (events from step 0) is detected only by the
  generic before-cursor guard on whichever channel sorts first
  (``model_logabs``); the adapter cannot attribute the mismatch to
  ``trainer_cursor``, because that channel's value IS the step number and the
  comparator aligns on step, making its value parity tautological.
"""

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
    """RED at CR: an unrecognized phase label must not disable the replay guard."""

    reference = _evidence("reference", 201, (3,))
    resumed = _evidence("restored", 202, (0, 1, 2, 3))
    with pytest.raises(ChannelParityError):
        compare_channel_parity(reference, resumed, resume_step=2)


def test_replayed_prefix_rejected_for_recovery_phase_control() -> None:
    """GREEN pin: the recognized 'recovery' label rejects the same replay."""

    reference = _evidence("reference", 203, (3,))
    resumed = _evidence("recovery", 204, (0, 1, 2, 3))
    with pytest.raises(ChannelParityError):
        compare_channel_parity(reference, resumed, resume_step=2)


def test_trainer_cursor_restart_signature_names_the_cursor_channel() -> None:
    """RED at CR: a cursor-omission restart must be attributed to trainer_cursor."""

    reference = _evidence("reference", 205, (3,))
    # The real signature of a trainer-cursor restore omission: the resumed
    # process restarts at step 0 and replays the prefix.
    resumed = _evidence("recovery", 206, (0, 1, 2, 3))
    with pytest.raises(ChannelParityError, match="trainer_cursor"):
        compare_channel_parity(
            reference,
            resumed,
            resume_step=2,
            disabled_limbs=("trainer_cursor",),
        )
