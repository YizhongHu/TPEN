"""A non-finite local-energy row is refused, unless masking is chosen on purpose.

The behaviour this replaces was not silent -- non-finite rows were excluded and
counted in ``local_energy_nonfinite_count``. It was still wrong to leave as the
only option, and the reason is statistical rather than procedural.

MASKING IS NOT A RANDOM SUBSAMPLE. Non-finite local energies occur where the
local energy is pathological: near nodes, at coalescence, in the tail, wherever
the wavefunction misbehaves. Those are exactly the regions carrying the physics
being measured. Dropping them selects a subsample SYSTEMATICALLY, so the energy
estimator is biased by an amount nobody has characterised -- and the count tells
you how many rows went, not what bias they took with them. A biased estimator
with a diagnostic attached is still a biased estimator, and it produces a
plausible number rather than a crash.

So masking stays reachable, because a scientist may knowingly want it, but only
by declaring it where the closed schema can see the declaration.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpen.training.trainer import VMCTrainer
from tpen.training.vmc import (
    DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    NONFINITE_LOCAL_ENERGY_POLICIES,
    compute_vmc_objective,
    resolve_nonfinite_local_energy_policy,
)


def _batch(bad: int = 0, total: int = 8):
    """Return logabs and local energies with ``bad`` non-finite rows."""

    logabs = torch.linspace(0.1, 1.0, total, dtype=torch.float64, requires_grad=True)
    energy = torch.full((total,), -2.9, dtype=torch.float64)
    for index in range(bad):
        energy[index] = math.inf if index % 2 == 0 else math.nan
    return logabs, energy


class TestTheObjectivePolicy:
    def test_fail_refuses_even_a_single_non_finite_row(self) -> None:
        """One bad row is enough. A threshold would be a number nobody chose."""

        logabs, energy = _batch(bad=1)
        with pytest.raises(ValueError, match="active policy is 'fail'"):
            compute_vmc_objective(logabs, energy, nonfinite_policy="fail")

    def test_mask_preserves_the_historical_behaviour_exactly(self) -> None:
        """The old estimator is unchanged, so choosing it is choosing the old run."""

        logabs, energy = _batch(bad=2)
        result = compute_vmc_objective(logabs, energy, nonfinite_policy="mask")
        assert result.metrics["local_energy_nonfinite_count"] == 2
        assert result.metrics["local_energy_n_finite"] == 6
        assert torch.isfinite(result.loss)

    def test_a_clean_batch_is_identical_under_both_policies(self) -> None:
        """Over-restriction control.

        A policy that refused clean batches too would satisfy every rejection
        test above and would only be discovered when a real run could not
        start. With no non-finite row the two policies must agree exactly.
        """

        logabs, energy = _batch(bad=0)
        failing = compute_vmc_objective(logabs, energy, nonfinite_policy="fail")
        masking = compute_vmc_objective(logabs, energy, nonfinite_policy="mask")
        torch.testing.assert_close(failing.loss, masking.loss)
        assert failing.metrics == masking.metrics

    def test_the_default_is_the_historical_behaviour(self) -> None:
        """Existing non-HI callers are unchanged by the policy's introduction.

        The helium-importance schema does NOT rely on this default -- it
        requires the declaration -- so HI cannot reach masking by omission
        while every other caller keeps working.
        """

        assert DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY == "mask"
        logabs, energy = _batch(bad=1)
        assert compute_vmc_objective(logabs, energy).metrics[
            "local_energy_nonfinite_count"
        ] == 1

    @pytest.mark.parametrize("policy", ["FAIL", "drop", "", None, 1])
    def test_an_unrecognised_policy_is_refused_rather_than_defaulted(self, policy) -> None:
        """A typo must not fall back to masking.

        Silently defaulting a misspelled policy would reintroduce the exact
        failure the policy exists to prevent: a run producing numbers under an
        estimator nobody chose.
        """

        with pytest.raises(ValueError, match="nonfinite local-energy policy"):
            resolve_nonfinite_local_energy_policy(policy)

    @pytest.mark.parametrize("policy", NONFINITE_LOCAL_ENERGY_POLICIES)
    def test_every_admitted_policy_round_trips(self, policy: str) -> None:
        assert resolve_nonfinite_local_energy_policy(policy) == policy

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            # Under "fail" the batch is refused for having ANY non-finite row,
            # which is reached before the count is examined -- so the policy
            # message is the correct one, not the empty-sample message.
            ("fail", "active policy is 'fail'"),
            ("mask", "no finite local-energy samples"),
        ],
    )
    def test_an_all_non_finite_batch_is_refused_under_every_policy(
        self, policy: str, expected: str
    ) -> None:
        """Refused under both, though for different and correct reasons.

        The empty-sample guard is retained under every policy: with no finite
        row there is no estimator at all, biased or otherwise. Under "fail" it
        is simply never reached, because a single non-finite row already
        refuses the step.

        Asserting the SPECIFIC message per policy rather than "it raises"
        keeps the test able to tell the two refusals apart -- otherwise a
        policy that failed for the wrong reason would still look correct.
        """

        logabs, energy = _batch(bad=8, total=8)
        with pytest.raises(ValueError, match=expected):
            compute_vmc_objective(logabs, energy, nonfinite_policy=policy)

    def test_the_empty_sample_guard_is_reachable_under_mask(self) -> None:
        """Control for the split above: the retained guard must still fire.

        If the empty-sample check had been made unreachable under both
        policies, the parametrized case would still pass on its "fail" arm and
        the loss of the guard would be invisible.
        """

        assert NONFINITE_LOCAL_ENERGY_POLICIES == ("fail", "mask")
        logabs, energy = _batch(bad=4, total=4)
        with pytest.raises(ValueError, match="no finite local-energy samples"):
            compute_vmc_objective(logabs, energy, nonfinite_policy="mask")


class TestTheActivePolicyTravelsWithTheCheckpoint:
    """Recording the policy is not enough; a resume must check it."""

    def test_the_trainer_records_the_active_policy(self) -> None:
        """Config records intent; this records what actually ran."""

        trainer = VMCTrainer(max_steps=1, nonfinite_local_energy_policy="fail")
        assert trainer.state_dict()["nonfinite_local_energy_policy"] == "fail"

    def test_resuming_under_a_different_policy_raises(self) -> None:
        """Otherwise one run would report under two different estimators.

        Follows the precedent of the update-method restore, which raises in
        both directions rather than skipping quietly.
        """

        trainer = VMCTrainer(max_steps=1, nonfinite_local_energy_policy="fail")
        with pytest.raises(ValueError, match="disagrees with the checkpoint"):
            trainer.load_state_dict(
                {
                    "next_iteration": 5,
                    "completed_updates": 5,
                    "nonfinite_local_energy_policy": "mask",
                }
            )

    def test_a_matching_policy_restores_normally(self) -> None:
        """Over-restriction control: the check must not block a valid resume."""

        trainer = VMCTrainer(max_steps=1, nonfinite_local_energy_policy="fail")
        trainer.load_state_dict(
            {
                "next_iteration": 5,
                "completed_updates": 5,
                "nonfinite_local_energy_policy": "fail",
            }
        )
        assert trainer.next_iteration == 5

    def test_a_legacy_checkpoint_is_read_as_mask_and_reported_as_such(self) -> None:
        """Absence is not neutral: every checkpoint written before this key masked.

        Waving a legacy checkpoint through because the old file happened to be
        silent would be the estimator change this check exists to catch, made
        invisible by the age of the artifact.
        """

        trainer = VMCTrainer(max_steps=1, nonfinite_local_energy_policy="fail")
        with pytest.raises(ValueError, match="absent, so the historical 'mask' is assumed"):
            trainer.load_state_dict({"next_iteration": 5, "completed_updates": 5})

    def test_a_legacy_checkpoint_resumes_under_the_historical_policy(self) -> None:
        """The other half: masking runs resume from legacy checkpoints unchanged."""

        trainer = VMCTrainer(max_steps=1, nonfinite_local_energy_policy="mask")
        trainer.load_state_dict({"next_iteration": 5, "completed_updates": 5})
        assert trainer.next_iteration == 5

    def test_the_policy_is_validated_at_construction_not_at_the_first_bad_row(self) -> None:
        """A misspelled policy must fail before a run produces any samples."""

        with pytest.raises(ValueError, match="nonfinite local-energy policy"):
            VMCTrainer(max_steps=1, nonfinite_local_energy_policy="faiil")
