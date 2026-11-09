"""Tests for the durable-step cadence a migrated callback schedules on.

`tpen.callback.Cadence` gates on `tpen.events.Occurrence.count`, which is
run-local and restarts at 1 after a checkpoint restore. Three of the callbacks
migrated to `tpen.callback.StatefulCallback` -- `GradientStats`, `SamplerHealth`
and `RuntimeEquivariance` -- were configured with ``every_n_steps: 5`` gating on
the DURABLE trainer step, so moving them onto an occurrence-count cadence would
have made a resumed run fire on a different schedule than an uninterrupted one:
the exact bug `tpen.callback.Checkpoint` documents and deliberately avoids.
`StepCadence` is what keeps that from happening.

Its window must also be identical to the legacy one it replaces, since this
migration changes the mechanism and nothing else. `_LegacySchedule` below is a
copy of the arithmetic in `_CallbackCore.should_run` plus the ``num_calls``
increment `_CallbackCore.handle` applied, and the equivalence test compares the
two across the whole option grid rather than at a few sampled points.
"""

from __future__ import annotations

import random

import pytest

from tpen.callback import StepCadence, StepCadenceGate
from tpen.callback.cadence import pop_step_cadence


class _LegacySchedule:
    """The legacy string path's scheduling decision, transcribed.

    Kept as a copy rather than an import so that deleting the legacy path
    (item ``85870732``) cannot quietly delete the reference this test compares
    against.
    """

    def __init__(
        self,
        every_n_steps: int | None = None,
        start_step: int = 0,
        max_calls: int | None = None,
        probability: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.every_n_steps = every_n_steps
        self.start_step = int(start_step)
        self.max_calls = max_calls
        self.probability = float(probability)
        self._rng = random.Random(seed)
        self.num_calls = 0

    def admits(self, step: int) -> bool:
        if self.max_calls is not None and self.num_calls >= self.max_calls:
            return False
        if self.every_n_steps is not None:
            if step < self.start_step:
                return False
            if (step - self.start_step) % self.every_n_steps != 0:
                return False
        if self.probability >= 1.0:
            admitted = True
        elif self.probability <= 0.0:
            admitted = False
        else:
            admitted = self._rng.random() < self.probability
        if admitted:
            self.num_calls += 1
        return admitted


def _remaining_options() -> list[dict[str, object]]:
    """Every option combination other than the interval, which parametrizes."""

    return [
        {
            "start_step": start_step,
            "max_calls": max_calls,
            "probability": probability,
            "seed": seed,
        }
        for start_step in (0, 1, 3, 10)
        for max_calls in (None, 0, 1, 4)
        for probability in (1.0, 0.0, 0.5, 0.25)
        # A seed of ``None`` draws from OS entropy, so two independent streams
        # could not agree even when the arithmetic matches. Seeds are fixed
        # here; `test_a_fully_scheduled_gate_never_touches_its_rng_stream`
        # covers the unseeded, fully-scheduled case separately.
        for seed in (0, 7, 123)
    ]


@pytest.mark.parametrize("every_n_steps", [None, 1, 2, 3, 5, 7])
def test_the_durable_window_matches_the_legacy_one_it_replaces(
    every_n_steps: int | None,
) -> None:
    """Exhaustive equivalence, so the claim rests on the grid and not on samples.

    The interval parametrizes and the remaining 192 combinations run inside, so
    a regression names the interval that broke it without adding four figures
    of test IDs to the suite.
    """

    mismatches = []
    for options in _remaining_options():
        options = {"every_n_steps": every_n_steps, **options}
        legacy = _LegacySchedule(**options)  # type: ignore[arg-type]
        gate = StepCadenceGate(pop_step_cadence(dict(options)))

        legacy_steps = [step for step in range(60) if legacy.admits(step)]
        gate_steps = [step for step in range(60) if gate.should_run(step)]

        if gate_steps != legacy_steps or gate.num_calls != legacy.num_calls:
            mismatches.append((options, legacy_steps, gate_steps))

    assert mismatches == [], f"{len(mismatches)} option sets diverged: {mismatches[:3]}"


def test_the_configured_every_n_steps_5_fires_on_the_same_steps_as_before() -> None:
    # The literal setting three migrated callbacks carry in
    # ``experiments/hooke/tpen-pair-v1/configs/train.yaml``.
    gate = StepCadenceGate(pop_step_cadence({"every_n_steps": 5}))

    assert [step for step in range(25) if gate.should_run(step)] == [0, 5, 10, 15, 20]


def test_a_restarted_occurrence_count_cannot_shift_the_schedule() -> None:
    # After a restore the trainer resumes at its durable cursor while the
    # run-local occurrence count starts again at 1. Gating on the step means the
    # resumed half fires exactly where the uninterrupted run would have.
    uninterrupted = StepCadenceGate(pop_step_cadence({"every_n_steps": 5}))
    fired = [step for step in range(20) if uninterrupted.should_run(step)]

    before = StepCadenceGate(pop_step_cadence({"every_n_steps": 5}))
    resumed = StepCadenceGate(pop_step_cadence({"every_n_steps": 5}))
    across_a_restore = [step for step in range(7) if before.should_run(step)]
    across_a_restore += [step for step in range(7, 20) if resumed.should_run(step)]

    assert across_a_restore == fired == [0, 5, 10, 15]


def test_a_fully_scheduled_gate_never_touches_its_rng_stream() -> None:
    # ``probability: 1.0`` is what the equivariance config sets. Consuming the
    # stream anyway would make the scheduling RNG's position depend on how many
    # steps had run, which matters the moment a probability below 1 is set.
    gate = StepCadenceGate(StepCadence(probability=1.0, seed=3))

    for step in range(10):
        assert gate.should_run(step)

    assert gate._rng.random() == random.Random(3).random()


def test_pop_step_cadence_consumes_exactly_the_five_scheduling_scalars() -> None:
    kwargs = {
        "every_n_steps": 5,
        "start_step": 1,
        "max_calls": 3,
        "probability": 0.5,
        "seed": 9,
        "fail_fast": True,
        "check_finite": False,
    }

    cadence = pop_step_cadence(kwargs)

    assert cadence == StepCadence(every_n=5, start=1, max_calls=3, probability=0.5, seed=9)
    # Everything else reaches the callback's own constructor untouched.
    assert kwargs == {"fail_fast": True, "check_finite": False}


def test_start_step_stays_inert_without_an_every_n_steps() -> None:
    # Bug-for-bug with the legacy path, which guarded its whole window --
    # ``start_step`` included -- behind ``if every_n_steps is not None``. No
    # shipped config sets one without the other, and preserving the quirk is
    # what keeps this migration from moving any callback's firing steps.
    gate = StepCadenceGate(pop_step_cadence({"start_step": 4}))

    assert [step for step in range(6) if gate.should_run(step)] == [0, 1, 2, 3, 4, 5]


def test_an_out_of_range_probability_is_still_rejected() -> None:
    # The legacy path validated this in ``_CallbackCore.__init__``. A migrated
    # callback routes ``probability`` past that constructor, so the check had to
    # move with it.
    with pytest.raises(ValueError, match=r"probability must be in \[0, 1\]"):
        pop_step_cadence({"probability": 1.5})


def test_a_zero_interval_is_rejected_rather_than_dividing_by_zero() -> None:
    with pytest.raises(ValueError, match="every_n must be at least 1"):
        pop_step_cadence({"every_n_steps": 0})
