"""Round-3 review probes for the one-bound-update-method lifecycle (PR 524).

Reviewer round 3, head dd1fa1464b38a71782e4dea29408a92ea89538f8. These two
tests exist to settle M-R2-4, the question the round-3 verifier deliberately
left open: does the R2-4 strengthening of the R1-5 pin -- replacing
``isinstance(seen, Adam)`` with identity against the captured runner-built
instance -- actually buy anything, or is it decorative?

The measurement design, so the two tests are read correctly:

- The first test is an INSTRUMENT, not a pin. It is the R1-5 fixture with the
  assertion weakened back to the pre-R2-4 ``isinstance`` form. At the pristine
  head it is green, like the real pin. Its value is under the round-3 mutants:
  a code mutant that the identity form catches while this form stays green is
  a regression class the R2-4 strengthening alone detects. If the writer
  adopts this file, keep the docstring's verdict with it or delete the test;
  a weakened twin that outlives its measurement reads as a second pin and is
  not one.

- The second test is the ADOPTABLE strengthening the measurement points at.
  The R1-5/R2-1 capture fixtures record only the LAST optimizer the factory
  built, so their identity assertions are blind to a runner that builds the
  carrier TWICE through the configured factory and hands `fit` the second
  build -- `built` is overwritten and ``seen is built`` passes. Counting the
  builds pins the runner's own claim (train.py: "ONE carrier is constructed
  for this run, here, and nothing below builds a second") directly on the
  foreign path, where no binding boundary exists to refuse the second build.
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


def test_m_r2_4_instrument_the_weakened_isinstance_form_of_the_r1_5_pin(
    tmp_path,
) -> None:
    """INSTRUMENT (M-R2-4): the R1-5 fixture with the pre-R2-4 assertion.

    This is what the R1-5 pin asserted before the writer's R2-4 fix: the
    optimizer reaching a foreign trainer's `fit` is an Adam, with no identity
    claim. Green at the pristine head. Under a mutant that delivers a
    same-type optimizer the runner's configured factory did not build, this
    stays green while the identity pin goes red -- that difference, measured,
    is M-R2-4's answer. Do not read this test alone as evidence of anything.
    """

    seen: dict = {}

    context = make_run_context(tmp_path / "weakened-foreign-run")
    torch.manual_seed(0)

    result = Train(
        model=build_tiny_spenn(),
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=lambda params: torch.optim.Adam(params, lr=LEARNING_RATE),
        trainer=_ForeignTrainer(seen),
    ).run(context)

    assert result.status == "completed"
    assert isinstance(seen["optimizer"], torch.optim.Adam), (
        "the weakened form only checks the TYPE of the carrier reaching fit"
    )


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
