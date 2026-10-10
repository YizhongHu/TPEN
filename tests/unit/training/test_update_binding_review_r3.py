"""Round-3 review probes for the one-bound-update-method lifecycle (PR 524).

Reviewer round 3. This file exists to settle M-R2-4, the question the round-3
verifier deliberately left open: does the R2-4 strengthening of the R1-5 pin
-- replacing ``isinstance(seen, Adam)`` with identity against the captured
runner-built instance -- actually buy anything, or is it decorative?

ADOPTED BY THE WRITER. The reviewer's file carried TWO tests; only the pin
below is adopted, and the omission is deliberate.

THE INSTRUMENT WAS DELETED ON ADOPTION, on the reviewer's own instruction.
Its other test was the R1-5 fixture with the assertion weakened back to the
pre-R2-4 ``isinstance`` form -- a measuring stick, not a pin. It is green at
the pristine head exactly like the real pin, so in the suite it would assert
nothing; its entire value was the DIFFERENCE it exposed under the round-3
mutants, and that measurement is now recorded rather than re-run. The
reviewer put it plainly: "a weakened twin that outlives its measurement reads
as a second pin and is not one."

WHAT IT MEASURED, kept because the result is the reason this file exists:
under MUT-BYPASS -- the runner bypassing its configured factory for a
same-type, same-hyperparameter, DIFFERENT instance -- the weakened
``isinstance`` form PASSED while the strengthened identity pin FAILED. So the
R2-4 strengthening is load-bearing, not decorative. That was the open
question (M-R2-4) the round-3 verifier flagged unsettled rather than skipping.

- The test below is the ADOPTABLE strengthening the measurement points at.
  A capture fixture that keeps only the last optimizer the factory built is
  blind to a runner that builds the carrier TWICE and hands `fit` the second
  build: the single slot holds the second, and an identity assertion against
  it passes. That is the shape this test closes. Counting the builds pins the
  runner's own claim (train.py: "ONE carrier is constructed for this run,
  here, and nothing below builds a second") directly on the foreign path,
  where no binding boundary exists to refuse the second build. Read the
  sibling fixtures for what THEY assert; this header does not describe them.
"""

from __future__ import annotations

import torch

from tpen.runner import Train
from tests.helpers.hooke_models import (
    build_tiny_hamiltonian_terms,
    build_tiny_spenn,
)
from tests.helpers.run_context import make_run_context
from tests.unit.training.test_update_binding import _FixedSampler

LEARNING_RATE = 0.01


class _ForeignTrainer:
    """The R1-5 foreign trainer: duck-typed contract, carrier-free return."""

    next_iteration = 0

    def __init__(self, seen: dict) -> None:
        self._seen = seen

    def resolve_update_state(self, *, model, optimizer):
        del model, optimizer
        # Truthy, and deliberately WITHOUT `.optimizer`, exactly as in R1-5.
        return object()

    def fit(self, *, model, sampler, hamiltonian_terms, optimizer, context, emit):
        del model, sampler, hamiltonian_terms, context, emit
        self._seen["optimizer"] = optimizer
        return None


def test_r3_the_runner_builds_exactly_one_carrier_for_a_foreign_trainer(
    tmp_path,
) -> None:
    """`Train.run` invokes the configured optimizer factory exactly once.

    The capture records EVERY build, not the last one, so this fails both
    when the runner bypasses the configured factory (zero builds captured,
    `fit` fed from elsewhere) and when it builds twice and feeds `fit` the
    second (the case the last-build capture in R1-5/R2-1 cannot see). On the
    foreign path there is no VMC binding boundary to refuse a second carrier,
    so this count is the only direct guard of the runner's one-carrier claim
    there.
    """

    seen: dict = {}
    builds: list = []

    def build_adam(params):
        builds.append(torch.optim.Adam(params, lr=LEARNING_RATE))
        return builds[-1]

    context = make_run_context(tmp_path / "counted-foreign-run")
    torch.manual_seed(0)

    result = Train(
        model=build_tiny_spenn(),
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=build_adam,
        trainer=_ForeignTrainer(seen),
    ).run(context)

    assert result.status == "completed"
    assert len(builds) == 1, (
        f"the runner invoked the configured optimizer factory {len(builds)} "
        "times; it must build exactly one carrier per run"
    )
    assert seen["optimizer"] is builds[0], (
        "the one carrier the runner built is not the one that reached fit"
    )
