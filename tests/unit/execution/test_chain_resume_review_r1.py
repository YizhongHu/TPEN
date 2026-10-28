"""Review r1: CR adapter guard strength.

Target layer: CR @ ddd01920 only.

Polarity at the CR head:
- test_identical_seed_provenance_is_rejected: GREEN. The CR contract rejects
  identical seed provenance for separate reference and recovery processes.
- test_first_subsequent_event_uses_completed_step_cursor: EXPECTED GREEN
  (pins the resume_step = next_iteration - 1 cursor convention).
- test_channel_map_totality_fails_on_unknown_owner: EXPECTED GREEN (pin).
"""

from __future__ import annotations

import pytest

from tpen.execution.chain_resume import (
    E5_CHANNEL_MAP,
    ChannelEvent,
    ChannelMapError,
    ChannelParityError,
    ProcessEvidence,
    assert_fresh_processes,
    e5_channel_map,
    first_subsequent_event,
)
from tpen.execution.native_profile import SUPPORTED_STATE_OWNERS


def _evidence(phase: str, pid: int, seed_source: str, steps) -> ProcessEvidence:
    events = tuple(
        ChannelEvent(
            channel="model_logabs",
            step=step,
            event_kind="first_model_evaluation",
            value=float(step),
        )
        for step in steps
    )
    return ProcessEvidence(
        phase=phase,
        pid=pid,
        seed=1,
        seed_source=seed_source,
        command=("probe",),
        returncode=0,
        events=events,
    )


def test_identical_seed_provenance_is_rejected() -> None:
    reference = _evidence("reference", 101, "config:runtime.seed=1", (4, 5))
    resumed = _evidence("recovery", 102, "config:runtime.seed=1", (4, 5))
    with pytest.raises(ChannelParityError):
        assert_fresh_processes(reference, resumed)


def test_first_subsequent_event_uses_completed_step_cursor() -> None:
    spec = next(s for s in E5_CHANNEL_MAP if s.channel == "model_logabs")
    resumed = _evidence("recovery", 103, "checkpoint:train_resume", (4, 5))
    event = first_subsequent_event(resumed, spec, resume_step=3)
    assert event.step == 4


def test_channel_map_totality_fails_on_unknown_owner() -> None:
    with pytest.raises(ChannelMapError):
        e5_channel_map(frozenset(SUPPORTED_STATE_OWNERS) | {"callback_private"})
