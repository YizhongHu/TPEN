"""The restore-limb disable mechanism, verified rather than trusted.

An unverified disable is worth nothing. It can be a no-op, in which case the
mutation arm goes red for an unrelated reason; worse, it can INVERT into a
strengthening, in which case the arm goes green and is read as the mutant
failing to matter. Both have happened in this repository. So the disable is
checked here in both directions, against observed state:

* enabled  -> the limb's live state MATCHES the checkpointed state
* disabled -> the limb's live state DIFFERS from it, and specifically still
  equals what the fresh process itself generated

SCOPE NOTE, load-bearing. These are unit tests of the apply seam
(:func:`~tests.helpers.chain_resume_spike.restore_limbs.apply_limb_states`) and
they run in-process on purpose: their evidence is a direct reading of RNG state,
not a parity comparison, so the in-process vacuity that the
fresh-process rule exists to prevent cannot arise here. The fresh-process rule
governs *resume attempts under a parity gate*, and those live in
``test_fixture_stream_sensitivity.py``. The final test in this module closes the
loop anyway, asserting the same consumption witness out of a genuine subprocess
receipt, so the seam is proven on the real path and not only in a unit harness.
"""

from __future__ import annotations

import pytest

from tests.helpers.chain_resume_spike.fixture import (
    ContinuationSystem,
    restore_system,
    spawn_attempt,
)
from tests.helpers.chain_resume_spike.receipt import read_receipt
from tests.helpers.chain_resume_spike.restore_limbs import (
    LIMB_CHANNELS,
    RestoreLimb,
    RestorePolicy,
    live_limb_fingerprints,
)


def _committed_payload() -> tuple[dict, dict]:
    """Return a parent's payload and its per-limb fingerprints after some work."""

    parent = ContinuationSystem.fresh(seed=11_111)
    parent.run(3)
    payload = {"carried": parent.carried_state_dict(), "limbs": parent.limb_state_dict()}
    return payload, live_limb_fingerprints(parent)


def test_an_enabled_limb_is_observed_consumed() -> None:
    """Positive direction: with nothing disabled, every limb really is applied."""

    payload, parent_fingerprints = _committed_payload()
    child = ContinuationSystem.fresh(seed=22_222)

    application = restore_system(child, payload, RestorePolicy.all_enabled())

    assert all(application.consumed.values()), application.consumed
    for limb in RestoreLimb:
        assert application.live_fingerprints[limb] == parent_fingerprints[limb], (
            f"{limb.value} reports consumed but its live state does not match the parent"
        )


@pytest.mark.parametrize("limb", list(RestoreLimb), ids=lambda limb: limb.value)
def test_a_disabled_limb_is_observed_not_consumed(limb: RestoreLimb) -> None:
    """Negative direction, and the anti-inversion check.

    Two assertions, because the weaker one alone is satisfiable by an inverted
    disable. The disabled limb must (a) not match the checkpoint, and (b) still
    hold exactly what the fresh child generated for itself -- which is what a
    genuine skip leaves behind. A disable that quietly substituted some third
    state would fail (b) while passing (a).
    """

    payload, _ = _committed_payload()
    child = ContinuationSystem.fresh(seed=33_333)
    before = live_limb_fingerprints(child)

    application = restore_system(child, payload, RestorePolicy.only_disabled(limb))

    assert application.consumed[limb] is False, (
        f"{limb.value} was disabled yet its state matches the checkpoint: the "
        "disable did not take effect"
    )
    assert application.live_fingerprints[limb] == before[limb], (
        f"{limb.value} was disabled but its live state changed anyway; the disable "
        "is not a clean skip and may have inverted into something else"
    )
    # Every OTHER limb must still have been applied. One limb per arm.
    for other in RestoreLimb:
        if other is not limb:
            assert application.consumed[other] is True, (
                f"disabling {limb.value} also suppressed {other.value}, so the two "
                "clauses share a falsifier and mask each other"
            )


def test_the_consumption_witness_is_an_observation_not_a_flag_echo() -> None:
    """``consumed`` must disagree with the policy when reality disagrees with it.

    Constructed by handing the seam a payload whose limb state is ALREADY live
    in the child. Every limb is then genuinely consumed no matter what the
    policy says, so a ``consumed`` map that merely echoed ``policy`` would
    report ``False`` for the disabled limb while the state plainly matches.
    """

    system = ContinuationSystem.fresh(seed=44_444)
    system.run(2)
    # The payload IS this system's own current state, so restoring it -- or not
    # restoring it -- leaves identical bytes either way.
    payload = {"carried": system.carried_state_dict(), "limbs": system.limb_state_dict()}

    application = restore_system(
        system, payload, RestorePolicy.only_disabled(RestoreLimb.SAMPLER_GENERATOR)
    )

    assert RestoreLimb.SAMPLER_GENERATOR not in application.requested_enabled
    assert application.consumed[RestoreLimb.SAMPLER_GENERATOR] is True, (
        "consumed echoed the policy instead of observing the live state"
    )


def test_a_cadence_with_probability_one_depends_only_on_its_counter() -> None:
    """The production-relevant half of the cadence limb, isolated.

    ``tpen.callback.cadence.StepCadenceGate._draw`` returns early without
    touching its RNG when ``probability >= 1`` (lines 216-217), while
    ``num_calls`` advances on every admission (line 207). So a fully-scheduled,
    capped gate has NO live RNG dependence and a fully live counter dependence
    -- and an unrestored counter lets a capped callback fire again after a
    resume. This arm pins that asymmetry, which the main parity fixture (which
    runs at ``probability < 1``) deliberately does not exercise.
    """

    parent = ContinuationSystem.fresh(
        seed=55_555, cadence_max_calls=3, cadence_probability=1.0
    )
    parent.run(3)
    assert parent.cadence_num_calls == 3, "the cap was not reached, so it is not tested"
    # The cap is now spent: an uninterrupted parent admits nothing further.
    assert parent.step_once()["measurement_gate"] is False

    payload = {"carried": parent.carried_state_dict(), "limbs": parent.limb_state_dict()}

    restored = ContinuationSystem.fresh(
        seed=66_666, cadence_max_calls=3, cadence_probability=1.0
    )
    restore_system(restored, payload, RestorePolicy.all_enabled())
    assert restored.step_once()["measurement_gate"] is False, (
        "a restored capped cadence admitted a measurement the parent would not have"
    )

    unrestored = ContinuationSystem.fresh(
        seed=77_777, cadence_max_calls=3, cadence_probability=1.0
    )
    restore_system(
        unrestored, payload, RestorePolicy.only_disabled(RestoreLimb.CADENCE_RNG)
    )
    assert unrestored.step_once()["measurement_gate"] is True, (
        "the counter half of the cadence limb is not observable, so its mutation "
        "arm would be vacuous for a probability=1 gate"
    )


def test_restore_policy_rejects_a_shape_that_would_disable_nothing() -> None:
    """A policy built from the wrong type must fail loudly, not silently pass.

    Typeguard is scoped to the ``tpen`` package (``--typeguard-packages=tpen``),
    so the annotations on :class:`RestorePolicy` guard nothing at runtime here
    and the checks have to be explicit. A bare string would otherwise be
    accepted and would disable no limb at all, turning every mutation arm
    green.
    """

    with pytest.raises(TypeError, match="frozenset"):
        RestorePolicy(disabled={RestoreLimb.GLOBAL_RNG})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-RestoreLimb"):
        RestorePolicy(disabled=frozenset({"global_rng"}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RestoreLimb"):
        RestorePolicy.only_disabled("global_rng")  # type: ignore[arg-type]


def test_policy_tokens_round_trip_through_the_command_line_form() -> None:
    """Tokens survive the argv hop to the child process without losing a limb."""

    for policy in (
        RestorePolicy.all_enabled(),
        RestorePolicy.all_disabled(),
        *[RestorePolicy.only_disabled(limb) for limb in RestoreLimb],
    ):
        assert RestorePolicy.from_tokens(policy.to_tokens()) == policy


def test_the_subprocess_path_reports_the_same_consumption_witness(tmp_path) -> None:
    """Close the loop: the seam behaves identically on the real fresh-process path.

    The unit arms above read the seam directly. This one reads it out of a
    genuine subprocess receipt, so the mechanism is not proven only under a
    harness that bypasses the way attempts actually run.
    """

    root = tmp_path / "root"
    first = spawn_attempt(
        tmp_path / "a0",
        root=root,
        run_id="witness",
        attempt_index=0,
        steps=2,
        total_target=4,
        checkpoint_every=2,
    )
    assert first.exit_code == 0, first.log_path.read_text(encoding="utf-8")

    second = spawn_attempt(
        tmp_path / "a1",
        root=root,
        run_id="witness",
        attempt_index=1,
        steps=2,
        total_target=4,
        checkpoint_every=2,
        policy=RestorePolicy.only_disabled(RestoreLimb.GLOBAL_RNG),
    )
    assert second.exit_code == 0, second.log_path.read_text(encoding="utf-8")

    receipt = read_receipt(second.receipt_path)
    assert receipt.disabled_limbs == ("global_rng",)
    assert receipt.limb_application is not None
    consumed = receipt.limb_application["consumed"]
    assert consumed["global_rng"] is False
    assert consumed["sampler_generator"] is True
    assert consumed["cadence_rng"] is True
    # The channel map the attribution rests on travels with the limb, so a
    # reviewer can check it against the receipt without re-reading the source.
    assert LIMB_CHANNELS[RestoreLimb.GLOBAL_RNG] == frozenset({"noise_term", "parameters"})
