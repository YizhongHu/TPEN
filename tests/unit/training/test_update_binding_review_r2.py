"""Round-2 review probes for the one-bound-update-method lifecycle (PR 524).

Reviewer round 2, head 47989a125b10778562c594a8db2a0204cb5a5335, base
dbb486ea10744810dc95a6d3d288c47efed012fb. This file is SELF-CONTAINED on
purpose: it imports nothing from the PR's own test files, so the identical
file runs unmodified at the base SHA (where those files do not exist) and at
the head. That is what lets each probe be measured on BOTH sides of the slice.

R2-1  The writer's open question at this head is mutant M-E: reverting the
      runner's carrier alias (`optimizer = update_state.optimizer`) survives
      the whole suite, "because every in-repo trainer satisfies
      `update_state.optimizer is optimizer`, so the alias is unreachable."
      The alias IS reachable -- by a foreign duck-typed trainer whose
      `resolve_update_state` returns an object WITH `.optimizer`. Before this
      PR, `Train.run` DISCARDED that return value unconditionally, so such a
      trainer's `fit` always received the runner-built optimizer. At the head,
      the attribute-guarded alias ADOPTS the foreign carrier instead, and
      restore and `fit` receive an optimizer the runner did not build. The
      acceptance clause "Preserve other configured trainers' existing runner
      path" is the same clause R1-5 was accepted under, and the in-repo safety
      is the same accident: no in-repo implementer happens to return such an
      object.

      This probe asserts the PRE-524 contract -- the runner-built optimizer is
      the one that reaches `fit`. That is what made it discriminating while
      R2-1 was open; the measurement that settled R2-1 lives in the record,
      not in this header.

      ADOPTED. The writer accepted R2-1 and removed the carrier alias outright
      rather than guarding it further, so this probe is no longer a red
      demonstration but the permanent pin against the alias being
      reintroduced. The probe BODY has been strengthened since it was written
      -- the capture records every build rather than only the last -- so do
      not read the code below as the form any earlier round measured.

R2-2  Control for "checkpoint/resume contracts are preserved": a STATELESS
      custom method (``update_state()`` is the base-class ``None``) still
      writes ``parameter_layout`` into its checkpoint, because the trainer's
      resolved state is the trainer-built wrapper -- and restore then calls
      ``rebuild_update_state``, which raises ``TypeError`` because the method
      has no update state to rebuild. Expected to behave IDENTICALLY at base
      and head: green on both arms means this is a pre-existing limitation of
      the stateless-method checkpoint path, not a regression from this slice.
      If the two arms diverge, that divergence is itself a finding.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import TrainResume, restore_checkpoint, save_checkpoint
from tpen.data.batch import ElectronBatch
from tpen.runner import Train
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    VMCUpdateMethod,
    VMCUpdateResult,
)
from tests.helpers.hooke_models import (
    build_tiny_hamiltonian_terms,
    build_tiny_spenn,
    tiny_pair_batch,
)
from tests.helpers.run_context import make_run_context
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext

LEARNING_RATE = 0.01


# ---------------------------------------------------------------------------
# Self-contained deterministic fixtures (duplicated, not imported, so the file
# runs at the base SHA where the PR's test files do not exist)
# ---------------------------------------------------------------------------


class _FixedWalkers:
    """Walkers wrapping one fixed batch."""

    def __init__(self, batch: ElectronBatch) -> None:
        self._batch = batch
        self.n_walkers = int(batch.batch_size)

    def make_batch(self) -> ElectronBatch:
        return self._batch


class _FixedSampler:
    """A sampler with no randomness and an honestly-empty checkpoint state."""

    def __init__(self, *, seed: int = 5, n_walkers: int = 4) -> None:
        generator = torch.Generator().manual_seed(seed)
        batch = tiny_pair_batch(n_walkers=n_walkers)
        positions = batch.positions + 0.01 * torch.randn(
            batch.positions.shape, generator=generator, dtype=batch.positions.dtype
        )
        self._batch = ElectronBatch(positions=positions, spins=batch.spins)

    def collect_samples(self, model, *, device=None):
        del model, device
        return _FixedWalkers(self._batch), None

    def mcmc_state_dict(self) -> dict[str, Any]:
        return {}

    def load_mcmc_state_dict(self, state: Any) -> None:
        del state


def _checkpoint_context(tmp_path) -> SimpleNamespace:
    """The public context boundary the checkpoint tests already use."""

    return SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "model": {"name": "tiny-tpen"},
                "optimizer": {"name": "adam"},
                "trainer": {"name": "vmc"},
                "sampler": {"name": "fixed"},
                "hamiltonian_terms": {"tiny": {}},
            }
        ),
        metadata=SimpleNamespace(device="cpu", dtype="float64"),
        run_dir=tmp_path,
    )


class _StatelessR2Method(VMCUpdateMethod[AutogradUpdateInput]):
    """A stateless custom method: ``update_state()`` is the base-class ``None``."""

    def __init__(self) -> None:
        self.update_calls = 0

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        self.update_calls += 1
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_probe")


# ---------------------------------------------------------------------------
# R2-1: the runner carrier alias is reachable by a foreign duck-typed trainer,
# and there it changes which optimizer reaches `fit`
# ---------------------------------------------------------------------------


def test_r2_1_a_foreign_resolve_return_with_a_carrier_keeps_the_runner_optimizer(
    tmp_path,
) -> None:
    """A foreign trainer returning an object WITH ``.optimizer`` keeps the
    runner-built optimizer, as it did before PR 524.

    Measured GREEN at base dbb486ea and RED at head 47989a12 by the reviewer:
    before the slice `Train.run` discarded `resolve_update_state`'s return, so
    whatever a duck-typed trainer returned, `fit` received the optimizer the
    runner built; the carrier alias adopted the foreign ``.optimizer``
    instead. That red arm proved the alias was NOT unreachable -- mutant M-E's
    survival was structural rather than a fixture accident -- on exactly the
    branch the R1-5 fix did not cover.

    The alias has since been removed, so this is now GREEN and serves as the
    pin that keeps it removed.
    """

    seen: dict[str, Any] = {}
    builds: list[Any] = []

    class _ForeignCarrierTrainer:
        """Implements the duck-typed runner contract and nothing else.

        Unlike R1-5's foreign trainer, its resolve return CARRIES an
        ``.optimizer`` -- a plausible shape for a trainer that manages its own
        update state and happens to expose its carrier.
        """

        next_iteration = 0

        def resolve_update_state(self, *, model, optimizer):
            del optimizer
            self.own_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            return SimpleNamespace(optimizer=self.own_optimizer)

        def fit(self, *, model, sampler, hamiltonian_terms, optimizer, context, emit):
            del model, sampler, hamiltonian_terms, context, emit
            seen["optimizer"] = optimizer
            return None

    def build_adam(params):
        # Every build, not the last. The single-slot-capture blindness is
        # recorded in the round-3 probe file's module docstring; pointing at
        # it BY CONTENT rather than by label is deliberate (round 5, R5-3) --
        # the label "R3-2" never appears in that file, and a tree grep for it
        # lands on an unrelated review lane's identically-numbered finding in
        # `test_update_observations.py`. A single-slot capture is satisfied by
        # a SECOND build comparing against itself, so it cannot see a runner
        # that builds twice.
        builds.append(torch.optim.Adam(params, lr=LEARNING_RATE))
        return builds[-1]

    context = make_run_context(tmp_path / "foreign-carrier-run")
    torch.manual_seed(0)
    trainer = _ForeignCarrierTrainer()

    result = Train(
        model=build_tiny_spenn(),
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=build_adam,
        trainer=trainer,
    ).run(context)

    assert result.status == "completed"
    assert builds, (
        "the runner never invoked the configured optimizer factory; it built "
        "its carrier somewhere else"
    )
    assert seen["optimizer"] is builds[0], (
        "the runner handed `fit` a carrier it did not build: the foreign "
        "trainer's `resolve_update_state` return was adopted instead of being "
        "discarded as it was before PR 524"
    )


# ---------------------------------------------------------------------------
# R2-2: the stateless-method checkpoint path behaves the same on both sides
# ---------------------------------------------------------------------------


def test_r2_2_a_stateless_method_checkpoint_resume_raises_in_rebuild(
    tmp_path,
) -> None:
    """A stateless custom method cannot resume the checkpoint it wrote.

    Its resolved state is the trainer-built wrapper, so ``state_dict`` writes
    ``parameter_layout``; restore then calls ``rebuild_update_state``, which
    asks the method for its own state and gets ``None``. Expected to raise the
    same ``TypeError`` at base AND head -- a pre-existing limitation of the
    stateless checkpoint path, asserted here so a silent change in either
    direction would surface as an arm divergence rather than pass unnoticed.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = _FixedSampler()
    method = _StatelessR2Method()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    trainer.resolve_update_state(model=model, optimizer=optimizer)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    assert method.update_calls == 1, "the probe's single step must actually have run"
    assert trainer.completed_updates == 0, "a declined update must not count"

    checkpoint = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=trainer.next_iteration,
        completed_updates=trainer.completed_updates,
        model=model,
        context=_checkpoint_context(tmp_path),
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
    )

    torch.manual_seed(123)
    resumed_model = build_tiny_spenn()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=LEARNING_RATE)
    resumed_trainer = VMCTrainer(
        max_steps=2, log_every_n_steps=1, update_method=_StatelessR2Method()
    )
    resumed_trainer.resolve_update_state(
        model=resumed_model, optimizer=resumed_optimizer
    )

    with pytest.raises(
        TypeError, match="returned no update state after rebind"
    ):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(checkpoint)},
            model=resumed_model,
            context=_checkpoint_context(tmp_path),
            optimizer=resumed_optimizer,
            trainer=resumed_trainer,
            sampler=_FixedSampler(),
        )
