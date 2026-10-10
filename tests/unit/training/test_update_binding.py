"""One bound update method per run, from construction through restore to fit.

The property under test is a LIFECYCLE one, not a numerical one: whatever
object is constructed before a checkpoint is restored must be the object that
performs the next update. A run that builds the method twice restores state
into the first instance and then steps with the second, so the checkpoint is
loaded into an object that is immediately discarded.

That is not hypothetical. Before this slice the trainer reused an instance only
when the SPEC that produced it was the same object, behind a ``spec is not
None`` guard -- so a run with no configured method, which is every run using the
default Adam path, fell through the guard and built a fresh
`LegacyAutogradUpdate` on each of the two selections a run performs. The default
adapter owns no persistent method state today, which is the only reason that was
survivable rather than a wrong-state resume.

Most tests here therefore drive the DEFAULT path deliberately. The configured
factory path is covered too, but as a control: it was already correct, and a
test that only covered it would have passed throughout the defect's lifetime.
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
    LegacyAutogradUpdate,
    ModelParameterBinding,
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
# Deterministic fixtures
# ---------------------------------------------------------------------------


class _FixedWalkers:
    """Walkers wrapping one fixed batch."""

    def __init__(self, batch: ElectronBatch) -> None:
        self._batch = batch
        self.n_walkers = int(batch.batch_size)

    def make_batch(self) -> ElectronBatch:
        return self._batch


class _FixedSampler:
    """A sampler with no randomness, and therefore no trajectory of its own.

    The resume-equivalence test below compares an uninterrupted run against an
    interrupted one. If the sampler contributed its own RNG stream, a mismatch
    could mean either a broken binding or a sampler that failed to restore its
    state, and the test could not tell those apart. Handing out the SAME batch
    every step removes that second explanation entirely, so any divergence can
    only come from the optimizer and method state this slice is about.

    It implements the checkpoint sampler contract (`mcmc_state_dict` /
    `load_mcmc_state_dict`) because `save_checkpoint` requires it; the state is
    genuinely empty rather than a stub that discards something real.
    """

    def __init__(self, *, seed: int = 5, n_walkers: int = 4) -> None:
        generator = torch.Generator().manual_seed(seed)
        batch = tiny_pair_batch(n_walkers=n_walkers)
        # Jitter the fixed positions once, deterministically, so the batch is
        # not the fixture's exact default and a degenerate zero batch cannot
        # make the comparison pass trivially.
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


def _fresh_run(*, seed: int = 0, max_steps: int = 1):
    """Build one seeded model/carrier/trainer triple with no configured method."""

    torch.manual_seed(seed)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = VMCTrainer(max_steps=max_steps, log_every_n_steps=1)
    return model, optimizer, trainer


def _fit(trainer, model, optimizer, *, sampler=None, context=None):
    """Drive one fit with the deterministic sampler and tiny Hamiltonian."""

    return trainer.fit(
        model=model,
        sampler=_FixedSampler() if sampler is None else sampler,
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=optimizer,
        context=_StubContext() if context is None else context,
        emit=lambda **_: None,
    )


class _CountingLegacyUpdate(LegacyAutogradUpdate):
    """A legacy adapter that OWNS persistent method state.

    The shipped adapter is stateless, so it cannot demonstrate the consequence
    of being rebuilt. This one carries a counter through
    ``method_state_dict``/``load_method_state_dict`` and records the value it
    was restored from, which makes "was the restored instance the one that
    stepped?" directly observable rather than inferred.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.updates_applied = 0
        self.restored_from: int | None = None

    def method_state_dict(self):
        return {"updates_applied": int(self.updates_applied)}

    def load_method_state_dict(self, state):
        self.restored_from = int(state["updates_applied"])
        self.updates_applied = int(state["updates_applied"])

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        result = super().update(update_input)
        if result.applied:
            self.updates_applied += 1
        return result


class _StatelessCustomMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A custom method with no owned state, exercising the `None` authority."""

    def __init__(self) -> None:
        self.update_calls = 0

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        self.update_calls += 1
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")


def _count_legacy_constructions(monkeypatch) -> dict[str, int]:
    """Count every `LegacyAutogradUpdate` construction, wherever it happens.

    Patching the class's own ``__init__`` rather than the construction helper is
    deliberate: the helper is new in this slice, so a counter attached to it
    would be unobservable in a tree without the fix and the red arm of a
    red/green pair would fail for the wrong reason. The class predates the
    change, so this counter means the same thing on both sides.
    """

    calls = {"n": 0}
    real_init = LegacyAutogradUpdate.__init__

    def counting_init(self, *args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(LegacyAutogradUpdate, "__init__", counting_init)
    return calls


# ---------------------------------------------------------------------------
# One construction per run
# ---------------------------------------------------------------------------


def test_the_default_adapter_is_constructed_once_per_runner_driven_run(
    tmp_path, monkeypatch
) -> None:
    """The real runner builds the default method once, not once per selection.

    This drives `tpen.runner.Train`, which is the only site that performs the
    full pre-restore resolve and then calls `fit`. A trainer-level test would
    not prove the runner still routes through one binding.
    """

    calls = _count_legacy_constructions(monkeypatch)
    context = make_run_context(tmp_path)
    torch.manual_seed(0)

    Train(
        model=build_tiny_spenn(),
        sampler=_FixedSampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=lambda params: torch.optim.Adam(params, lr=LEARNING_RATE),
        trainer=VMCTrainer(max_steps=1, log_every_n_steps=1),
    ).run(context)

    assert calls["n"] == 1, (
        "the default adapter was constructed more than once; the instance the "
        "runner resolved before restore is not the one that ran the loop"
    )


def test_the_method_resolved_before_restore_is_the_one_that_updates() -> None:
    """Resolve then fit must share ONE instance, on the default path.

    The identity is read through the public `VMCUpdateState`: a legacy adapter
    hands out its own `ModelParameterBinding`, and two separately constructed
    adapters over the same model hold two DIFFERENT binding objects. So an
    ``is`` comparison on that binding distinguishes one instance from two
    without reaching into a private attribute.
    """

    model, optimizer, trainer = _fresh_run()

    resolved = trainer.resolve_update_state(model=model, optimizer=optimizer)
    final_state = _fit(trainer, model, optimizer)

    assert final_state.update_state.model_parameters is resolved.model_parameters
    assert final_state.update_state.optimizer is optimizer
    assert trainer.completed_updates == 1, "the loop must still have run"


def test_direct_fit_binds_once_and_enters_the_same_loop(monkeypatch) -> None:
    """A caller that never resolved binds exactly once inside `fit`."""

    calls = _count_legacy_constructions(monkeypatch)
    model, optimizer, trainer = _fresh_run()

    final_state = _fit(trainer, model, optimizer)

    assert calls["n"] == 1
    assert trainer.completed_updates == 1
    assert final_state.update_state.optimizer is optimizer


def test_an_explicit_none_late_selector_rereads_the_constructor_spec() -> None:
    """``update_method=None`` re-reads the CONSTRUCTOR spec; it does not rebuild.

    SCOPED, after reviewer round 2 (R2-2). An earlier name and docstring here
    said ``None`` means "use what is bound" -- the exact wording round 1
    refuted in the trainer's own docstring, recurring verbatim in the test
    that is supposed to pin the behaviour. It is only true in THIS case, where
    the trainer was never given a constructor spec, so re-reading it yields
    ``None`` and nothing conflicts. A trainer configured with one method and
    bound from an explicit override RAISES on the same call; that is pinned by
    the adopted round-1 R1-2 probe in
    `tests/unit/training/test_update_binding_review_r1.py`.
    """

    model, optimizer, trainer = _fresh_run()

    first = trainer.bind_update_method(model=model, optimizer=optimizer, update_method=None)
    second = trainer.bind_update_method(model=model, optimizer=optimizer, update_method=None)

    assert first is second


def test_a_configured_factory_is_constructed_once(monkeypatch) -> None:
    """Control: the FACTORY path. Correct before this slice, and still correct.

    Recorded as a control rather than as evidence for the change. The previous
    spec-identity memo already covered this path, so this test would have passed
    throughout the defect's lifetime; it is here to prove the new binding did
    not regress it.
    """

    del monkeypatch
    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    factory_calls = {"n": 0}

    def factory(opt, *, model_parameters):
        factory_calls["n"] += 1
        return _CountingLegacyUpdate(opt, model_parameters=model_parameters)

    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=factory)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    _fit(trainer, model, optimizer)

    assert factory_calls["n"] == 1


# ---------------------------------------------------------------------------
# Restore: the restored instance performs the next update
# ---------------------------------------------------------------------------


def _save_resume_checkpoint(tmp_path, *, model, optimizer, trainer, sampler):
    """Write a real public train-resume checkpoint for this run."""

    return save_checkpoint(
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


def test_the_restored_stateful_method_is_the_one_that_performs_the_next_update(
    tmp_path,
) -> None:
    """A method that owns state must not be rebuilt between restore and fit.

    `_CountingLegacyUpdate` carries a counter across the checkpoint. If the
    trainer rebuilt the method after restore, the instance performing the first
    resumed update would have ``restored_from is None`` and a counter back at
    zero -- a resume that looks clean and is not.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = _FixedSampler()

    def factory(opt, *, model_parameters):
        return _CountingLegacyUpdate(opt, model_parameters=model_parameters)

    trainer = VMCTrainer(max_steps=2, log_every_n_steps=1, update_method=factory)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    _fit(trainer, model, optimizer, sampler=sampler)
    assert trainer.completed_updates == 2
    checkpoint = _save_resume_checkpoint(
        tmp_path, model=model, optimizer=optimizer, trainer=trainer, sampler=sampler
    )

    # A genuinely fresh run: different seed, new objects, nothing carried over
    # in memory. Everything it knows must come through the checkpoint.
    torch.manual_seed(123)
    resumed_model = build_tiny_spenn()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=LEARNING_RATE)
    resumed_sampler = _FixedSampler()
    resumed_trainer = VMCTrainer(max_steps=3, log_every_n_steps=1, update_method=factory)
    # The runner's order: bind BEFORE restore.
    resumed_trainer.resolve_update_state(model=resumed_model, optimizer=resumed_optimizer)
    restore_checkpoint(
        load={"mode": "train_resume", "path": str(checkpoint)},
        model=resumed_model,
        context=_checkpoint_context(tmp_path),
        optimizer=resumed_optimizer,
        trainer=resumed_trainer,
        sampler=resumed_sampler,
    )
    resumed_trainer.rebuild_update_state(model=resumed_model)
    final_state = _fit(resumed_trainer, resumed_model, resumed_optimizer, sampler=resumed_sampler)

    stepped = final_state.update_state
    bound = resumed_trainer.bind_update_method(
        model=resumed_model, optimizer=resumed_optimizer
    )
    assert isinstance(bound, _CountingLegacyUpdate)
    assert bound.restored_from == 2, "the stepping instance is not the restored one"
    assert bound.updates_applied == 3
    assert stepped.model_parameters is bound.model_parameters
    assert all(
        left is right
        for left, right in zip(
            bound.model_parameters.parameters,
            tuple(resumed_model.parameters()),
            strict=True,
        )
    )


def test_the_default_adapter_survives_restore_as_one_instance(tmp_path, monkeypatch) -> None:
    """The DEFAULT path, across a real restore, still builds exactly one method."""

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = _FixedSampler()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    _fit(trainer, model, optimizer, sampler=sampler)
    checkpoint = _save_resume_checkpoint(
        tmp_path, model=model, optimizer=optimizer, trainer=trainer, sampler=sampler
    )

    calls = _count_legacy_constructions(monkeypatch)
    torch.manual_seed(321)
    resumed_model = build_tiny_spenn()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=LEARNING_RATE)
    resumed_sampler = _FixedSampler()
    resumed_trainer = VMCTrainer(max_steps=2, log_every_n_steps=1)
    resolved = resumed_trainer.resolve_update_state(
        model=resumed_model, optimizer=resumed_optimizer
    )
    restore_checkpoint(
        load={"mode": "train_resume", "path": str(checkpoint)},
        model=resumed_model,
        context=_checkpoint_context(tmp_path),
        optimizer=resumed_optimizer,
        trainer=resumed_trainer,
        sampler=resumed_sampler,
    )
    resumed_trainer.rebuild_update_state(model=resumed_model)
    final_state = _fit(
        resumed_trainer, resumed_model, resumed_optimizer, sampler=resumed_sampler
    )

    assert calls["n"] == 1, "the resumed run built a second default adapter"
    # The rebind REPLACES the binding object, so the pre-restore binding is
    # deliberately NOT expected to survive; what must survive is the method,
    # and it is the method that owns the binding the loop finally used.
    assert final_state.update_state.optimizer is resolved.optimizer
    assert all(
        left is right
        for left, right in zip(
            final_state.update_state.model_parameters.parameters,
            tuple(resumed_model.parameters()),
            strict=True,
        )
    )


def test_adam_moments_and_trajectory_match_uninterrupted_execution(tmp_path) -> None:
    """Four steps straight through equal two, a checkpoint, and two more.

    This is the preservation claim the whole slice rests on, and it is checked
    by EXACT equality rather than a tolerance: with a fixed batch and float64
    CPU arithmetic, a resumed run that restored every Adam moment must follow
    the identical trajectory. A tolerance would hide precisely the kind of
    partially-restored state this test exists to catch.
    """

    terms = build_tiny_hamiltonian_terms()

    torch.manual_seed(0)
    straight_model = build_tiny_spenn()
    straight_optimizer = torch.optim.Adam(straight_model.parameters(), lr=LEARNING_RATE)
    straight_trainer = VMCTrainer(max_steps=4, log_every_n_steps=1)
    straight_sampler = _FixedSampler()
    straight_trainer.resolve_update_state(model=straight_model, optimizer=straight_optimizer)
    straight_trainer.fit(
        model=straight_model,
        sampler=straight_sampler,
        hamiltonian_terms=terms,
        optimizer=straight_optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    expected = [p.detach().clone() for p in straight_model.parameters()]

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = _FixedSampler()
    trainer = VMCTrainer(max_steps=2, log_every_n_steps=1)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )
    checkpoint = _save_resume_checkpoint(
        tmp_path, model=model, optimizer=optimizer, trainer=trainer, sampler=sampler
    )

    torch.manual_seed(777)
    resumed_model = build_tiny_spenn()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=LEARNING_RATE)
    resumed_sampler = _FixedSampler()
    resumed_trainer = VMCTrainer(max_steps=4, log_every_n_steps=1)
    resumed_trainer.resolve_update_state(model=resumed_model, optimizer=resumed_optimizer)
    restore_checkpoint(
        load={"mode": "train_resume", "path": str(checkpoint)},
        model=resumed_model,
        context=_checkpoint_context(tmp_path),
        optimizer=resumed_optimizer,
        trainer=resumed_trainer,
        sampler=resumed_sampler,
    )
    resumed_trainer.rebuild_update_state(model=resumed_model)
    resumed_trainer.fit(
        model=resumed_model,
        sampler=resumed_sampler,
        hamiltonian_terms=terms,
        optimizer=resumed_optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    assert resumed_trainer.next_iteration == straight_trainer.next_iteration == 4
    assert resumed_trainer.completed_updates == straight_trainer.completed_updates == 4
    # A non-vacuity control: the run must actually have MOVED, or two
    # never-updated models would compare equal and prove nothing.
    torch.manual_seed(0)
    untrained = [p.detach().clone() for p in build_tiny_spenn().parameters()]
    assert any(
        not torch.equal(before, after) for before, after in zip(untrained, expected, strict=True)
    ), "the uninterrupted run did not move any parameter"
    for reference, resumed in zip(expected, resumed_model.parameters(), strict=True):
        assert torch.equal(reference, resumed.detach())


# ---------------------------------------------------------------------------
# Refusals: all of them fire before sampling and before mutable restore
# ---------------------------------------------------------------------------


def test_a_second_model_is_refused_after_binding() -> None:
    """One trainer does not silently drive a second model."""

    model, optimizer, trainer = _fresh_run()
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    torch.manual_seed(1)
    other_model = build_tiny_spenn()

    with pytest.raises(ValueError, match="already bound to a different model"):
        trainer.resolve_update_state(model=other_model, optimizer=optimizer)


def test_a_second_carrier_is_refused_after_binding() -> None:
    """A run publishes one optimizer and mutates that same one.

    The message asserted here is the BINDING boundary's own, not
    `_resolve_method_state`'s "mismatched legacy optimizer ownership". The two
    guards used to share a string, and reviewer round 1 plus the independent
    verifier's M2 mutant together showed what that cost: the older check fires
    first on this path, so the binding check could be deleted outright and no
    test noticed. Asserting the distinct message is what pins WHICH guard
    refused.
    """

    model, optimizer, trainer = _fresh_run()
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    other_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with pytest.raises(ValueError, match="already bound to a different optimizer"):
        trainer.resolve_update_state(model=model, optimizer=other_optimizer)


def test_a_conflicting_late_selector_is_refused_after_binding() -> None:
    """A different method supplied after binding fails rather than rebuilding."""

    model, optimizer, trainer = _fresh_run()
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    late = LegacyAutogradUpdate(
        optimizer,
        model_parameters=ModelParameterBinding.from_parameters(tuple(model.parameters())),
    )

    with pytest.raises(ValueError, match="already bound from a different specification"):
        trainer.bind_update_method(model=model, optimizer=optimizer, update_method=late)


def test_the_bound_instance_itself_is_not_a_conflicting_selector() -> None:
    """Handing back the very method that is bound is not a late selection."""

    model, optimizer, trainer = _fresh_run()
    bound = trainer.bind_update_method(model=model, optimizer=optimizer)

    assert (
        trainer.bind_update_method(model=model, optimizer=optimizer, update_method=bound)
        is bound
    )


def test_a_refusal_happens_before_any_update_runs() -> None:
    """The guards fire at the binding boundary, not part-way through a loop."""

    model, optimizer, trainer = _fresh_run()
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    other_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with pytest.raises(ValueError, match="already bound to a different optimizer"):
        _fit(trainer, model, other_optimizer)

    assert trainer.completed_updates == 0
    assert trainer.next_iteration == 0


# ---------------------------------------------------------------------------
# Existing contracts that must keep working
# ---------------------------------------------------------------------------


def test_a_bound_instance_spec_is_retained_not_reconstructed() -> None:
    """An explicitly constructed method is bound as-is."""

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    method = LegacyAutogradUpdate(
        optimizer,
        model_parameters=ModelParameterBinding.from_parameters(tuple(model.parameters())),
    )
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    assert trainer.bind_update_method(model=model, optimizer=optimizer) is method
    assert trainer.bind_update_method(model=model, optimizer=optimizer) is method


def test_a_stateless_custom_method_is_bound_once_and_keeps_its_authority() -> None:
    """A method returning no `VMCUpdateState` still binds exactly once.

    Its authority is the supplied optimizer, as before; what changes is that the
    trainer now holds the instance rather than re-deriving it.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    method = _StatelessCustomMethod()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    resolved = trainer.resolve_update_state(model=model, optimizer=optimizer)
    assert resolved.optimizer is optimizer
    final_state = _fit(trainer, model, optimizer)

    assert method.update_calls == 1
    assert final_state.update_state.optimizer is optimizer
    assert trainer.completed_updates == 0, "a declined step must not count"
