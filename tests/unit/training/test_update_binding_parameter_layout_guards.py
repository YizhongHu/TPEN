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
    LegacyAutogradUpdate,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
)
from types import SimpleNamespace

from tests.helpers.hooke_models import build_tiny_hamiltonian_terms, build_tiny_spenn
from tests.unit.training.test_update_binding import (
    _FixedSampler,
    _StatelessCustomMethod,
    _StubContext,
)

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


class _CountingSampler:
    """Wraps the deterministic sampler and COUNTS how often it is asked.

    The contract clause under test is temporal, not behavioural: a refusal
    that arrives after one sample is still the right refusal and still breaks
    the promise. Only a witness that observes WHETHER sampling happened can
    tell those apart, which is why this counts rather than asserting on the
    error alone.
    """

    def __init__(self) -> None:
        self._inner = _FixedSampler()
        self.collect_calls = 0

    def collect_samples(self, model, *, device=None):
        self.collect_calls += 1
        return self._inner.collect_samples(model, device=device)

    def mcmc_state_dict(self):
        return self._inner.mcmc_state_dict()

    def load_mcmc_state_dict(self, state) -> None:
        self._inner.load_mcmc_state_dict(state)


def test_a_wrong_binding_is_refused_BEFORE_any_sample_is_drawn() -> None:
    """ACCEPTANCE clause: the refusal must land BEFORE SAMPLING.

    The contract does not merely require that a wrong model/optimizer binding
    fails; it requires that it fails *before sampling or mutable restore*. A
    trainer that validates late raises the identical ValueError after having
    already drawn a batch, satisfies every error-matching test in this suite,
    and violates the clause. The independent verifier measured exactly that:
    delayed validation survived the focused, unit and integration selections
    while its external witnesses sampled once before the correct refusal.

    `collect_calls == 0` is the whole assertion. The ValueError is necessary
    but not sufficient, so it is checked and then set aside.
    """

    model, optimizer, _method, trainer = _drift_fixture("optimizer")
    sampler = _CountingSampler()

    with pytest.raises(ValueError, match="mismatched legacy optimizer ownership"):
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=build_tiny_hamiltonian_terms(),
            optimizer=optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
        )

    assert sampler.collect_calls == 0, (
        "the binding was refused only AFTER sampling; the contract requires "
        "the refusal to land before any sample is drawn"
    )


def test_the_counting_sampler_does_count_when_a_run_is_allowed_to_proceed() -> None:
    """Control for the witness itself, not for the trainer.

    `collect_calls == 0` proves nothing unless this sampler is known to reach
    a non-zero count on a run that is permitted to sample. Without this, a
    sampler that was never wired in at all would satisfy the clause test.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    sampler = _CountingSampler()

    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda **_: None,
    )

    assert sampler.collect_calls > 0


def test_a_drifted_PARAMETER_BINDING_is_refused_BEFORE_any_sample_is_drawn() -> None:
    """The clause names TWO paths and the optimizer one does not cover both.

    ACCEPTANCE: "a conflicting late selector or wrong model/OPTIMIZER BINDING
    fails before sampling or mutable restore." The sibling test above drives
    optimizer drift, so a trainer that keeps the type and optimizer guards
    eager and defers ONLY the parameter-identity check past the first sample
    satisfies it -- the optimizer refusal still lands early and nothing counts
    the sample drawn before the binding was checked.

    That is not hypothetical. The independent verifier built exactly that
    mutant: committed focused 42, unit 3066 + 5 skip and integration 6 all
    passed with the same phase outcomes as pristine, while its external
    binding-drift probe recorded one sample and zero updates before the
    correct ValueError.

    One witness pins one arm of a disjunction. This is the second disjunctive
    clause on this slice to need that said out loud; `:381` was the first.
    """

    model, optimizer, _method, trainer = _drift_fixture("binding")
    sampler = _CountingSampler()

    with pytest.raises(
        ValueError, match="update method parameter binding does not match live model"
    ):
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=build_tiny_hamiltonian_terms(),
            optimizer=optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
        )

    assert sampler.collect_calls == 0, (
        "the parameter binding was refused only AFTER sampling; the contract "
        "requires the refusal to land before any sample is drawn"
    )


def test_a_conflicting_late_SELECTOR_is_refused_BEFORE_any_sample_is_drawn() -> None:
    """The selector arm of the clause, at the sampling boundary.

    ACCEPTANCE names TWO failure classes: "a conflicting late SELECTOR or
    wrong model/optimizer binding fails before sampling or mutable restore."
    Three rounds of this lane pinned refinements of the SECOND class while the
    first had no timing witness at all.

    `test_a_conflicting_late_selector_is_refused_after_binding` pins that the
    refusal happens; it calls `bind_update_method` directly and observes
    nothing about when the refusal lands relative to sampling. An independent
    verifier (Cannon 52065488) built `delay_selector_sample`, which defers only
    the selector predicate past the first ordinary sample and SURVIVED F 43,
    I 8, B 3067 pass + 5 skip and J 12, all rc 0, while its external witness
    recorded one sample before the correct ValueError.

    Deleting the selector guard outright is killed by two direct binding tests,
    so the guard is not untested -- only its TIMING is. This witness therefore
    has to fail on the sample count, not on the absence of a refusal.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer.resolve_update_state(model=model, optimizer=optimizer)

    # A DIFFERENT method instance arriving late. Not the bound one -- handing
    # back the very instance that is bound is explicitly not a conflict.
    late = LegacyAutogradUpdate(
        optimizer,
        model_parameters=ModelParameterBinding.from_parameters(tuple(model.parameters())),
    )
    sampler = _CountingSampler()

    with pytest.raises(ValueError, match="already bound from a different specification"):
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=build_tiny_hamiltonian_terms(),
            optimizer=optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
            update_method=late,
        )

    assert sampler.collect_calls == 0, (
        "the conflicting late selector was refused only AFTER sampling; the "
        "contract requires the refusal to land before any sample is drawn"
    )


def _bound_trainer(stateless: bool = False):
    """One trainer already bound to its own model and carrier.

    `stateless` selects a method whose `update_state()` returns `None`. That is
    not a cosmetic variation: `_resolve_method_state` RETURNS EARLY on `None`,
    so neither the ownership guard nor the parameter guard ever runs, and the
    retained model/carrier checks in `bind_update_method` are the only defence
    left. An independent verifier measured all four retained-binding single
    deferrals crossing their boundary on this path (Cannon 52093726), which
    refuted a claim of mine that a single deferral is always caught by the
    method-state guard. It is caught only when a method HAS state.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = VMCTrainer(
        max_steps=1,
        log_every_n_steps=1,
        update_method=_StatelessCustomMethod() if stateless else None,
    )
    trainer.resolve_update_state(model=model, optimizer=optimizer)
    return model, optimizer, trainer


@pytest.mark.parametrize("stateless", [False, True], ids=["stateful", "stateless"])
def test_a_second_MODEL_is_refused_BEFORE_any_sample_is_drawn(stateless: bool) -> None:
    """The rebind guards are in the clause too, and their timing was unpinned.

    `test_a_second_model_is_refused_after_binding` calls `resolve_update_state`
    directly and asserts nothing about timing.
    `test_a_refusal_happens_before_any_update_runs` drives `fit` but asserts
    `completed_updates == 0` and `next_iteration == 0` -- COUNTERS, not
    samples. A guard deferred past `collect_samples` and raised before the
    update leaves both counters at zero and survives, while its docstring
    claims "the guards fire at the binding boundary, not part-way through a
    loop".

    That is the right property with the wrong witness, which is how this lane
    has lost four cells of this clause already. Sampling is the observable the
    contract names, so the assertion has to be on the sample count.
    """

    _model, optimizer, trainer = _bound_trainer(stateless)
    torch.manual_seed(1)
    other_model = build_tiny_spenn()
    sampler = _CountingSampler()

    with pytest.raises(ValueError, match="already bound to a different model"):
        trainer.fit(
            model=other_model,
            sampler=sampler,
            hamiltonian_terms=build_tiny_hamiltonian_terms(),
            optimizer=optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
        )

    assert sampler.collect_calls == 0, (
        "the second model was refused only AFTER sampling; the contract "
        "requires the refusal to land before any sample is drawn"
    )


@pytest.mark.parametrize("stateless", [False, True], ids=["stateful", "stateless"])
def test_a_second_CARRIER_is_refused_BEFORE_any_sample_is_drawn(stateless: bool) -> None:
    """The carrier half of the same rebind pair, at the same boundary."""

    model, _optimizer, trainer = _bound_trainer(stateless)
    other_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = _CountingSampler()

    with pytest.raises(ValueError, match="already bound to a different optimizer"):
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=build_tiny_hamiltonian_terms(),
            optimizer=other_optimizer,
            context=_StubContext(),
            emit=lambda **_: None,
        )

    assert sampler.collect_calls == 0, (
        "the second carrier was refused only AFTER sampling; the contract "
        "requires the refusal to land before any sample is drawn"
    )
