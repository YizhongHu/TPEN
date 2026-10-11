"""Pins for the two parameter-binding guards on the update-method boundary.

The acceptance contract for this slice requires the trainer to "validate live
parameter/optimizer identity".  The OPTIMIZER half of that clause is pinned in
several places.  The PARAMETER half was not pinned anywhere: the independent
verifier at head 1f4f1e26 disabled both guards at once and every arm reported
identical pass counts, so a regression, inversion or outright deletion of
either one would have passed the whole suite.

Each test below drives exactly one guard, through the trainer entry point that
reaches it, and asserts the guard's own message.  A method that returns a
binding over parameters the live model does not own is the discriminating
input in both cases, because ``ModelParameterBinding.compare`` requires
parameter-reference IDENTITY and not merely a matching layout: two models of
the same architecture compare unequal.

These probes assert what the code beneath them does.  Per-round measurements,
job identifiers and review history live in the record -- PR 524's thread and
the Task Orchestrator notes on item b6b5b4f5 -- and deliberately not here.
"""

from __future__ import annotations

import pytest
import torch
import typeguard

from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
)
from types import SimpleNamespace

from tests.helpers.hooke_models import build_tiny_hamiltonian_terms, build_tiny_spenn
from tests.unit.training.test_update_binding import _FixedSampler, _StubContext

LEARNING_RATE = 0.01


class _CarrierSwappingMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A method that accepts the rebind correctly, then swaps its carrier.

    This is the shape the post-rebind ownership check exists for: everything
    about the parameter binding is honoured, so the parameter guard below it
    cannot fire, and only the optimizer-identity check can refuse. ``swap``
    is what makes one class serve as both the witness and its own control.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        binding: ModelParameterBinding,
        *,
        swap: bool,
    ) -> None:
        self._optimizer = optimizer
        self._binding = binding
        self._swap = swap
        self.rebind_calls = 0

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")

    def update_state(self) -> VMCUpdateState:
        return VMCUpdateState(optimizer=self._optimizer, model_parameters=self._binding)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self.rebind_calls += 1
        self._binding = model_parameters
        if self._swap:
            # A DIFFERENT Adam over the SAME live parameters: the run would
            # publish one carrier and mutate another.
            self._optimizer = torch.optim.Adam(
                [p for p in model_parameters.parameters], lr=LEARNING_RATE
            )


class _ForeignBindingMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A stateful method whose reported binding is not the live model's.

    ``rebind_model_parameters`` is honoured or ignored according to
    ``honour_rebind``; that switch is what lets one class drive both guards.
    The optimizer is reported unchanged throughout, so the optimizer-ownership
    checks that precede both parameter checks pass and cannot be mistaken for
    the refusal under test.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        binding: ModelParameterBinding,
        *,
        honour_rebind: bool,
    ) -> None:
        self._optimizer = optimizer
        self._binding = binding
        self._honour_rebind = honour_rebind
        self.rebind_calls = 0

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")

    def update_state(self) -> VMCUpdateState:
        return VMCUpdateState(optimizer=self._optimizer, model_parameters=self._binding)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self.rebind_calls += 1
        if self._honour_rebind:
            self._binding = model_parameters


def test_resolve_refuses_a_method_bound_to_another_models_parameters() -> None:
    """`_resolve_method_state`'s parameter check refuses a foreign binding.

    The method reports the run's own optimizer, so the ownership check ahead of
    this one passes and the refusal under test is the only one that can fire.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    foreign_model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    foreign_binding = ModelParameterBinding.from_parameters(tuple(foreign_model.parameters()))

    # Control: the foreign binding is unequal for the reason claimed -- the
    # parameter REFERENCES differ -- and not because the layouts differ.
    live_binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    assert live_binding.layout.compare(foreign_binding.layout)[0], (
        "the two models must share a layout, or this test would pass for the wrong reason"
    )
    assert not live_binding.compare(foreign_binding)[0]

    method = _ForeignBindingMethod(optimizer, foreign_binding, honour_rebind=True)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    with pytest.raises(
        ValueError, match="update method parameter binding does not match live model"
    ):
        trainer.resolve_update_state(model=model, optimizer=optimizer)


def test_rebuild_refuses_a_method_that_ignores_the_rebuilt_binding() -> None:
    """`rebuild_update_state`'s parameter check refuses a dropped rebind.

    Restore rebinds the method onto the reloaded model's parameters. A method
    that accepts the call and keeps its old references would leave the resumed
    loop stepping parameters the run no longer owns; this guard is what refuses
    it, and the assertion below is on that guard's own message.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _ForeignBindingMethod(optimizer, binding, honour_rebind=False)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    # Resolving succeeds: at this point the reported binding IS the live one.
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    # A restore replaces the model objects; the method ignores the rebind.
    torch.manual_seed(123)
    reloaded_model = build_tiny_spenn()

    with pytest.raises(
        ValueError, match="update method did not retain the rebuilt model binding"
    ):
        trainer.rebuild_update_state(model=reloaded_model)

    assert method.rebind_calls == 1, (
        "the guard must fire because the rebind was DROPPED, not because it was never attempted"
    )


def test_rebuild_refuses_a_method_that_swaps_its_carrier_during_rebind() -> None:
    """`rebuild_update_state`'s ownership check refuses a post-rebind swap.

    The method honours the parameter rebind exactly, so the parameter guard
    beneath this one cannot be what refuses: the only disagreement left is the
    optimizer identity. A run that accepted this would publish one carrier and
    mutate another across a resume.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _CarrierSwappingMethod(optimizer, binding, swap=True)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    torch.manual_seed(123)
    reloaded_model = build_tiny_spenn()

    with pytest.raises(ValueError, match="mismatched legacy optimizer ownership"):
        trainer.rebuild_update_state(model=reloaded_model)

    assert method.rebind_calls == 1, (
        "the guard must fire on a SWAP, not because the rebind was never attempted"
    )
    assert method._optimizer is not optimizer, (
        "the fixture did not actually swap the carrier, so the refusal proves nothing"
    )


def test_rebuild_accepts_a_method_that_retains_its_carrier_through_rebind() -> None:
    """Control for the test above: retaining the carrier is NOT refused.

    Without this, the swap test would be satisfied by a trainer that refuses
    every rebuild, which would pin nothing about carrier identity.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _CarrierSwappingMethod(optimizer, binding, swap=False)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    torch.manual_seed(123)
    reloaded_model = build_tiny_spenn()

    rebuilt = trainer.rebuild_update_state(model=reloaded_model)

    assert rebuilt.optimizer is optimizer
    assert method.rebind_calls == 1


def test_rebuild_refuses_to_run_before_the_method_has_been_resolved() -> None:
    """`rebuild_update_state` refuses when nothing has been resolved yet.

    The runner's order is resolve-then-restore-then-rebuild. Reaching rebuild
    first means the run never bound a method, and rebuilding from no authority
    would invent one. The accept path -- resolve first, then rebuild -- is
    exercised by the retain-carrier test in this file, so this is not satisfied
    by a trainer that refuses every rebuild.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)

    with pytest.raises(
        RuntimeError, match="update state must be resolved before it can be rebuilt"
    ):
        trainer.rebuild_update_state(model=model)


class _StructuralStateMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A method returning something SHAPED like VMCUpdateState but not one.

    ``nominal`` selects between the real dataclass and an ordinary object
    carrying the same two attribute names, so one class serves as both the
    witness and its control.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        binding: ModelParameterBinding,
        *,
        nominal: bool,
    ) -> None:
        self._optimizer = optimizer
        self._binding = binding
        self._nominal = nominal

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")

    def update_state(self):
        if self._nominal:
            return VMCUpdateState(optimizer=self._optimizer, model_parameters=self._binding)
        return SimpleNamespace(optimizer=self._optimizer, model_parameters=self._binding)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self._binding = model_parameters


def test_resolve_refuses_a_structurally_compatible_non_update_state() -> None:
    """Admission is NOMINAL: the declared type, not a matching shape.

    The substitute carries the same two attributes with the same values, so
    every check BELOW this one would pass on it. Only the type admission can
    refuse, which is what makes this a pin on that line rather than on the
    guards after it.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _StructuralStateMethod(optimizer, binding, nominal=False)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    # The suite runs with `--typeguard-packages=tpen`, so without this
    # suppression typeguard intercepts the substitute at the call boundary and
    # raises before the production check is reached. That would make this test
    # pass while measuring the TEST HARNESS rather than the guard, and it would
    # pass identically with the guard deleted -- which is what a first run of
    # this test actually showed. Suppressing runtime type checking here is what
    # makes the production refusal the only thing that can answer, which is the
    # configuration shipped code runs in.
    with typeguard.suppress_type_checks():
        with pytest.raises(
            TypeError, match="update_state must return VMCUpdateState or None"
        ):
            trainer.resolve_update_state(model=model, optimizer=optimizer)


def test_resolve_accepts_the_same_state_when_it_is_nominally_correct() -> None:
    """Control: the identical contents in a real VMCUpdateState ARE accepted.

    Without this, the refusal above would be satisfied by a trainer that
    rejects every stateful method, which would pin nothing about nominality.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _StructuralStateMethod(optimizer, binding, nominal=True)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    # Suppressed on the same terms as the refusal test, so the pair differs in
    # exactly one thing: whether the returned state is nominally a
    # VMCUpdateState.
    with typeguard.suppress_type_checks():
        resolved = trainer.resolve_update_state(model=model, optimizer=optimizer)

    assert resolved.optimizer is optimizer


def test_rebuild_refuses_when_the_method_is_bound_but_no_state_was_resolved() -> None:
    """The precondition is a DISJUNCTION, and this is its other arm.

    `bind_update_method` alone leaves the method bound and the state unresolved.
    A test that only ever reaches this guard with BOTH halves absent pins one
    disjunct: weakening the `or` to an `and` still refuses the both-absent case
    and silently admits this one. That weakening was measured surviving every
    committed test before this one existed.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)

    trainer.bind_update_method(model=model, optimizer=optimizer)

    with pytest.raises(
        RuntimeError, match="update state must be resolved before it can be rebuilt"
    ):
        trainer.rebuild_update_state(model=model)


class _NominalThenStructuralMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """Nominal at resolve, structurally-compatible after rebind.

    The substitute carries the SAME optimizer object and the rebuilt binding,
    so the identity and comparison checks below the admission check both pass
    on it. Only a nominal check can refuse it, which is what makes this a pin
    on that line.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, binding: ModelParameterBinding) -> None:
        self._optimizer = optimizer
        self._binding = binding
        self._rebound = False

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")

    def update_state(self):
        if self._rebound:
            return SimpleNamespace(optimizer=self._optimizer, model_parameters=self._binding)
        return VMCUpdateState(optimizer=self._optimizer, model_parameters=self._binding)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self._binding = model_parameters
        self._rebound = True


def test_rebuild_refuses_a_structurally_compatible_state_after_rebind() -> None:
    """Nominal admission is enforced on the RESTORE path, not only on fit.

    Resolve refuses a non-VMCUpdateState; without the same refusal here the
    two boundaries disagree, and a method that is nominal at resolve and
    structural after rebind has its substitute retained as the run's authority.
    Typeguard is suppressed for the same reason as the resolve-path pair: the
    suite instruments `tpen`, so without this the harness would answer instead
    of the production check and the test would pass with the guard deleted.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _NominalThenStructuralMethod(optimizer, binding)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    with typeguard.suppress_type_checks():
        trainer.resolve_update_state(model=model, optimizer=optimizer)

        torch.manual_seed(123)
        reloaded_model = build_tiny_spenn()
        with pytest.raises(
            TypeError, match="update_state must return VMCUpdateState or None"
        ):
            trainer.rebuild_update_state(model=reloaded_model)


class _SubclassedUpdateState(VMCUpdateState):
    """A lawful subclass of the declared authority type.

    Nothing in the contract requires an implementation to return the base
    class exactly; `isinstance` admits subclasses and that is the documented
    behaviour both admission checks rely on.
    """


class _SubclassStateMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A method whose owned state is a SUBCLASS of VMCUpdateState."""

    def __init__(self, optimizer: torch.optim.Optimizer, binding: ModelParameterBinding) -> None:
        self._optimizer = optimizer
        self._binding = binding

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")

    def update_state(self) -> VMCUpdateState:
        return _SubclassedUpdateState(optimizer=self._optimizer, model_parameters=self._binding)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self._binding = model_parameters


def test_resolve_accepts_a_lawful_subclass_of_the_update_state() -> None:
    """Admission is NOMINAL, not EXACT: a subclass is a VMCUpdateState.

    The refusal tests in this file pin that a non-VMCUpdateState is rejected.
    They do not pin the other direction, and the independent verifier measured
    that gap: tightening either `isinstance` check to an exact-type comparison
    survived every committed suite. Nothing noticed that lawful subclass
    implementations had been locked out. This is the discriminator for that.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _SubclassStateMethod(optimizer, binding)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)

    resolved = trainer.resolve_update_state(model=model, optimizer=optimizer)

    assert isinstance(resolved, _SubclassedUpdateState), (
        "the subclass must survive admission as itself, not be coerced to the base"
    )
    assert resolved.optimizer is optimizer


def test_rebuild_accepts_a_lawful_subclass_of_the_update_state() -> None:
    """The RESTORE boundary admits subclasses on the same terms as resolve.

    Both boundaries now carry an `isinstance` admission check, so both can be
    tightened to an exact-type comparison independently. One discriminator per
    boundary is what keeps that from happening unnoticed at either.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))

    method = _SubclassStateMethod(optimizer, binding)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    torch.manual_seed(123)
    reloaded_model = build_tiny_spenn()

    rebuilt = trainer.rebuild_update_state(model=reloaded_model)

    assert isinstance(rebuilt, _SubclassedUpdateState)
    assert rebuilt.optimizer is optimizer


class _DriftingMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """Passes validation once, then DRIFTS what it owns.

    Validation is not a one-time rite. The trainer re-resolves at every later
    boundary, so a method that satisfies the checks at resolve and then swaps
    its owned optimizer -- or its parameter binding -- must be refused when it
    next presents itself. `drift` selects which half moves; `armed` is flipped
    by the test AFTER the first successful validation, so the first pass is
    genuinely clean and the refusal can only come from re-validation.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        binding: ModelParameterBinding,
        *,
        drift: str,
    ) -> None:
        self._optimizer = optimizer
        self._binding = binding
        self._drift = drift
        self.armed = False

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")

    def update_state(self) -> VMCUpdateState:
        optimizer, binding = self._optimizer, self._binding
        if self.armed and self._drift == "optimizer":
            optimizer = torch.optim.Adam(
                [p for p in self._binding.parameters], lr=LEARNING_RATE
            )
        if self.armed and self._drift == "binding":
            binding = ModelParameterBinding.from_parameters(
                tuple(build_tiny_spenn().parameters())
            )
        return VMCUpdateState(optimizer=optimizer, model_parameters=binding)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self._binding = model_parameters


def _drift_fixture(drift: str):
    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    binding = ModelParameterBinding.from_parameters(tuple(model.parameters()))
    method = _DriftingMethod(optimizer, binding, drift=drift)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, update_method=method)
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    method.armed = True
    return model, optimizer, method, trainer


def test_a_later_resolve_refuses_an_optimizer_that_drifted_after_validation() -> None:
    """Re-validation catches a carrier swapped AFTER the first clean resolve.

    The independent verifier measured that caching the first validated state
    and trusting it at later boundaries survives every committed suite. Nothing
    was pinning that the trainer re-validates rather than reuses.
    """

    model, optimizer, _method, trainer = _drift_fixture("optimizer")

    with pytest.raises(ValueError, match="mismatched legacy optimizer ownership"):
        trainer.resolve_update_state(model=model, optimizer=optimizer)


def test_a_later_resolve_refuses_a_binding_that_drifted_after_validation() -> None:
    """The same for the parameter half of the authority."""

    model, optimizer, _method, trainer = _drift_fixture("binding")

    with pytest.raises(
        ValueError, match="update method parameter binding does not match live model"
    ):
        trainer.resolve_update_state(model=model, optimizer=optimizer)


def test_fit_refuses_an_optimizer_that_drifted_after_validation() -> None:
    """`fit` is the boundary that matters, and it re-validates too.

    The later-resolve tests above pin `_resolve_method_state`. This one drives
    the public `fit` entry point, so a cache introduced at the fit call site
    rather than inside the resolver is caught as well.
    """

    model, optimizer, _method, trainer = _drift_fixture("optimizer")

    with pytest.raises(ValueError, match="mismatched legacy optimizer ownership"):
        trainer.fit(
            model=model,
            sampler=_FixedSampler(),
            hamiltonian_terms=build_tiny_hamiltonian_terms(),
            optimizer=optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
        )


def test_a_later_resolve_accepts_a_method_that_did_not_drift() -> None:
    """Control: re-validation must not refuse an authority that stayed put.

    Without this, the three refusals above would be satisfied by a trainer
    that rejects every second resolve, which would pin nothing about drift.
    """

    model, optimizer, method, trainer = _drift_fixture("optimizer")
    method.armed = False

    resolved = trainer.resolve_update_state(model=model, optimizer=optimizer)

    assert resolved.optimizer is optimizer
