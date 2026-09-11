"""G0: continuation parity on SUBSEQUENT DRAWS, and per-limb attribution.

The gate this module pins is the one prior spike DS-A0 passed while its RNG was
entirely discarded. DS-A0 compared deterministic outputs, so nothing it asserted
depended on the restored stream, and its non-vacuity control inherited exactly
the same blindness. Four properties are therefore asserted together, and no one
of them is sufficient alone:

1. The green arm holds -- with every limb restored, the resumed process
   reproduces the uninterrupted run's subsequent draws.
2. Each single-limb arm DIVERGES.
3. Each single-limb arm diverges in a channel that limb actually feeds. Redness
   alone is produced by every limb and by an unrelated bug; only the channel
   says which mutant ran.
4. The resumed process is a genuinely fresh OS process whose own seed differs
   from the parent's, so property 1 cannot hold by reinitialization accident.

Arm A and arm B share one trajectory by construction: arm A is a single
uninterrupted process that runs K+L steps and commits a generation at K along
the way, and arm B restores THAT generation. Seeding two independent processes
and hoping they agree would compare two unrelated runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.chain_resume_spike.fixture import spawn_attempt
from tests.helpers.chain_resume_spike.parity import (
    assert_channel_map_is_disjoint_and_total,
    divergence_is_attributable_to,
    first_divergence,
)
from tests.helpers.chain_resume_spike.receipt import AttemptReceipt, read_receipt
from tests.helpers.chain_resume_spike.restore_limbs import (
    LIMB_CHANNELS,
    RestoreLimb,
    RestorePolicy,
)

#: Fixed total logical target for the chain.
TOTAL_TARGET = 8
#: Steps the first link runs before the comparison point.
RESUME_STEP = 4
#: Generations are committed this often.
CHECKPOINT_EVERY = 2


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory) -> tuple[AttemptReceipt, Path]:
    """Run K+L steps in one uninterrupted process; return its receipt and gen K.

    Module-scoped so every resume arm restores from the identical baseline
    rather than from a per-test one, following the precedent in
    ``tests/integration/training/test_train_runner.py``.
    """

    base = tmp_path_factory.mktemp("uninterrupted")
    launch = spawn_attempt(
        base / "attempt",
        root=base / "root",
        run_id="g0-reference",
        attempt_index=0,
        steps=TOTAL_TARGET,
        total_target=TOTAL_TARGET,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    assert launch.exit_code == 0, launch.log_path.read_text(encoding="utf-8")
    parent = base / "root" / f"step_{RESUME_STEP:06d}"
    assert parent.is_dir(), "the uninterrupted arm committed no generation at K"
    return read_receipt(launch.receipt_path), parent


def _resume(
    tmp_path: Path, parent: Path, policy: RestorePolicy | None = None
) -> AttemptReceipt:
    """Restore ``parent`` in a fresh process and run the remaining L steps."""

    launch = spawn_attempt(
        tmp_path / "resumed",
        root=tmp_path / "root",
        run_id="g0-reference",
        attempt_index=1,
        steps=TOTAL_TARGET - RESUME_STEP,
        total_target=TOTAL_TARGET,
        checkpoint_every=CHECKPOINT_EVERY,
        policy=policy,
        restore_from=parent,
    )
    assert launch.exit_code == 0, launch.log_path.read_text(encoding="utf-8")
    return read_receipt(launch.receipt_path)


def test_the_channel_map_partitions_every_observable_channel() -> None:
    """Every limb feeds at least one channel, and no two limbs share one.

    Runs before the parity arms because it is their precondition. A limb
    feeding no channel has a vacuous mutation arm -- the DS-A0 shape -- and two
    limbs sharing a channel could not be told apart by the divergence they
    produce, so each would mask the other.
    """

    assert_channel_map_is_disjoint_and_total()


def test_resume_reproduces_the_uninterrupted_subsequent_draws(
    uninterrupted, tmp_path
) -> None:
    """GREEN ARM. Run first: a red green-arm invalidates every mutation arm.

    Compared on the draws taken AFTER the restore -- next proposals, acceptance
    uniforms, walker positions, the parameter noise term and the cadence
    measurements -- never on serialized state bytes and never on deterministic
    outputs alone.
    """

    reference, parent = uninterrupted
    resumed = _resume(tmp_path, parent)

    divergence = first_divergence(reference.trace[RESUME_STEP:], resumed.trace)
    assert divergence is None, (
        "restored continuation did not reproduce the uninterrupted stream: "
        + divergence.describe()
    )


def test_the_resumed_attempt_is_a_fresh_process_with_its_own_distinct_seed(
    uninterrupted, tmp_path
) -> None:
    """The green arm must not hold because the resume re-derived the same seed.

    Without this, parity would be satisfiable by a reinitialization that
    happened to match -- which is indistinguishable, in the trace, from a
    genuine restore. The seed is drawn from OS entropy and recorded in the
    receipt precisely so this is a checked fact.
    """

    reference, parent = uninterrupted
    resumed = _resume(tmp_path, parent)

    assert resumed.pid != reference.pid, "the resume did not run in a fresh process"
    assert resumed.process_seed != reference.process_seed, (
        "the resumed process drew the SAME seed as its parent, so parity above "
        "cannot distinguish a restore from a reinitialization"
    )
    assert resumed.identity.parent_generation == RESUME_STEP
    assert resumed.identity.run_id == reference.identity.run_id
    assert resumed.identity.attempt_id != reference.identity.attempt_id


@pytest.mark.parametrize("limb", list(RestoreLimb), ids=lambda limb: limb.value)
def test_disabling_one_limb_diverges_in_that_limb_own_channel(
    uninterrupted, tmp_path, limb: RestoreLimb
) -> None:
    """One limb disabled per arm; divergence must land in a channel IT feeds.

    Disabling exactly one limb, with the others enabled, keeps two clauses from
    sharing a falsifier. Asserting the CHANNEL, not merely that the arm went
    red, is what proves the intended mutant is the one that ran.
    """

    reference, parent = uninterrupted
    resumed = _resume(tmp_path, parent, policy=RestorePolicy.only_disabled(limb))

    divergence = first_divergence(reference.trace[RESUME_STEP:], resumed.trace)
    assert divergence is not None, (
        f"disabling {limb.value} did not perturb the continuation at all, so the "
        "parity gate above is blind to that limb"
    )
    assert divergence_is_attributable_to(divergence, limb), (
        f"disabling {limb.value} diverged in channel {divergence.channel!r}, which "
        f"that limb does not feed (it feeds {sorted(LIMB_CHANNELS[limb])}). The arm "
        "is red for the wrong reason."
    )


def test_disabling_every_limb_diverges(uninterrupted, tmp_path) -> None:
    """The all-disabled arm, alongside the single-limb arms.

    A single-limb arm that passes for the wrong reason still has to account for
    itself here, where nothing at all is restored.
    """

    reference, parent = uninterrupted
    resumed = _resume(tmp_path, parent, policy=RestorePolicy.all_disabled())

    divergence = first_divergence(reference.trace[RESUME_STEP:], resumed.trace)
    assert divergence is not None, "restoring nothing still reproduced the stream"
    assert resumed.limb_application is not None
    assert not any(resumed.limb_application["consumed"].values()), (
        "the all-disabled arm reports a limb as consumed, so the disable did not "
        "actually take effect"
    )
