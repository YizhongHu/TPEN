"""Contract tests for the update seam's objective re-evaluation callable.

Each test here corresponds to one STATED PROPERTY of
:class:`tpen.training.update.ObjectiveReevaluation`.  The properties are
recorded in that class's own docstring; this module is their enforcement, and
the mapping is deliberately one test per property so a failure names the
property rather than a mechanism.

Two of the properties are DECLARED PRECONDITIONS rather than guarantees the
code could enforce on its own:

RNG-inertness currently holds BY ABSENCE.  Nothing on the recompute path draws
from an RNG, because ``tpen/physics/`` contains no generator at all.  The day
it acquires one -- a stochastic kinetic estimator being the obvious candidate
-- ``test_repeated_reevaluation_at_unchanged_parameters_is_bitwise_equal`` is
the test that must go red.  ``torch.random.fork_rng()`` was rejected precisely
so that failure stays loud instead of being silently pinned.

Under policy ``"mask"`` the estimator's subsample is parameter-dependent, so
the same-function property is claimed only at UNCHANGED parameters.  Whether a
line search over a mask-biased estimator is sound at all is an open question
this seam surfaces and does not answer.
"""

from __future__ import annotations

import copy
import pickle
import pytest
import torch

import tpen.training.update as update_module
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.sampling.metropolis import MetropolisSampler
from tpen.training.vmc import compute_vmc_objective
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    ObjectiveReevaluation,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
    vmc_objective_reevaluation,
)
from tests.helpers.hooke_models import (
    build_tiny_hamiltonian_terms,
    build_tiny_sampler,
    build_tiny_spenn,
)
from tests.helpers.vmc_scientific_oracle import (
    loss_tolerance_envelope,
    oracle_vmc_objective,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext


def _parameter_tolerance_envelope(terms: torch.Tensor) -> float:
    """Parameter-space tolerance: the loss envelope, widened for backprop.

    Same derivation and the same 10x multiple used by the DDP scientific
    oracle for its parameter comparisons: comparing model parameters after a
    real backward pass adds the tiny network's own op chain on top of the
    reduction envelope.  LBFGS with ``max_iter > 1`` runs that chain several
    times per step, which this multiple already covers for the shallow Hooke
    fixture.
    """

    return 10.0 * loss_tolerance_envelope(terms)


# ---------------------------------------------------------------------------
# Fixtures and test-authored update methods
# ---------------------------------------------------------------------------


class _CountingModel(torch.nn.Module):
    """Record the exact batch object handed to every forward call."""

    def __init__(self, weight: float = 0.75) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([weight], dtype=torch.float64))
        self.seen_batches: list[ElectronBatch] = []

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        self.seen_batches.append(batch)
        shape = (batch.batch_size,)
        # PER-ROW, deliberately.  An identical logabs on every row makes the
        # score-function objective identically zero for ANY row subset: mu is
        # the subset mean, so sum_i m_i (E_i - mu) is exactly 0 and the logabs
        # factors straight out.  A row-selection test built on a constant
        # logabs therefore compares 0.0 with 0.0 and cannot discriminate --
        # measured, not hypothesised (Cannon job 44953686).
        per_row = batch.positions.reshape(batch.batch_size, -1).sum(dim=-1)
        logabs = self.weight * per_row
        return WavefunctionOutput(logabs=logabs, sign=torch.ones(shape, dtype=batch.dtype))


class _ConstantTerm:
    """Hamiltonian term returning a fixed per-sample local energy."""

    def __init__(self, values: torch.Tensor) -> None:
        self.values = values
        self.calls = 0

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        del wavefunction
        self.calls += 1
        return LocalEnergyResult(total=self.values.expand(batch.batch_size).clone(), terms={})


class _VaryingTerm:
    """Per-row DISTINCT local energies, so a row subset changes the estimator.

    Needed for the row-selection test: with a constant local energy every
    subset shares one mean and the centred residuals vanish, so the objective
    would not move however the selection was applied.
    """

    def __init__(self, offset: float = 0.0) -> None:
        self.offset = float(offset)
        self.calls = 0

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        del wavefunction
        self.calls += 1
        values = torch.arange(batch.batch_size, dtype=batch.dtype) * 0.75 + self.offset
        return LocalEnergyResult(total=values, terms={})


class _AlwaysNonFiniteTerm:
    """Non-finite from the FIRST call.

    Distinct from `_NonFiniteAfterFirstCallTerm` for a reason worth stating:
    that one exists so a TRAINER step can succeed and only the re-evaluation
    meet a non-finite row.  A test that calls the factory DIRECTLY has no
    primary step, so its first call is the re-evaluation -- and the
    after-first-call term would still be finite there.  Using the wrong one
    silently produced a DID NOT RAISE on the propagation test (Cannon job
    44953686).
    """

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        del wavefunction
        total = torch.full((batch.batch_size,), 1.5, dtype=batch.dtype)
        total[0] = float("inf")
        return LocalEnergyResult(total=total, terms={})


class _NonFiniteAfterFirstCallTerm:
    """Finite for the primary objective, non-finite on every re-evaluation.

    This is what makes the policy-continuity property observable through the
    REAL construction path: the trainer's own step succeeds, and only the
    re-evaluation meets a non-finite row.  A term that was non-finite from the
    first call would make the trainer refuse the step before any update input
    existed, so the seam's policy could never be read at all.
    """

    def __init__(self) -> None:
        self.calls = 0

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        del wavefunction
        self.calls += 1
        total = torch.full((batch.batch_size,), 1.5, dtype=batch.dtype)
        if self.calls > 1:
            total[0] = float("inf")
        return LocalEnergyResult(total=total, terms={})


def _finite_primary(batch: ElectronBatch) -> torch.Tensor:
    """An all-finite primary local energy, matching the batch's shape."""

    return torch.full((batch.batch_size,), 1.5, dtype=batch.dtype)


def _batch(n_walkers: int = 3, n_electrons: int = 1) -> ElectronBatch:
    positions = torch.linspace(
        0.1, 1.0, n_walkers * n_electrons, dtype=torch.float64
    ).reshape(n_walkers, n_electrons, 1)
    spins = torch.ones((n_walkers, n_electrons), dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)


class _ReevaluatingLegacyUpdate(VMCUpdateMethod):
    """Legacy update behaviour, preceded by ``k`` discarded re-evaluations.

    Deliberately identical to :class:`LegacyAutogradUpdate` in everything it
    does to the model and the optimizer.  Its only difference is that it calls
    the re-evaluation ``k`` times and throws the results away, so a run using
    it must be bitwise indistinguishable from a legacy run -- in parameters,
    in walkers, and in sampler RNG state.  That is the resume guard: any
    divergence means merely LOOKING at the objective twice changed the run.
    """

    def __init__(self, optimizer, model_parameters: ModelParameterBinding, *, k: int = 3) -> None:
        self.k = int(k)
        self.calls: list[torch.Tensor] = []
        self._delegate = LegacyAutogradUpdate(optimizer, model_parameters=model_parameters)

    def update_state(self) -> VMCUpdateState:
        return self._delegate.update_state()

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self._delegate.rebind_model_parameters(model_parameters)

    def set_step_scopes(self, *, backward_scope=None, optimizer_scope=None) -> None:
        self._delegate.set_step_scopes(
            backward_scope=backward_scope, optimizer_scope=optimizer_scope
        )

    def state_dict(self):
        return self._delegate.state_dict()

    def load_state_dict(self, state) -> None:
        self._delegate.load_state_dict(state)

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        for _ in range(self.k):
            self.calls.append(update_input.reevaluate().detach().clone())
        return self._delegate.update(update_input)


class _TrueClosureLBFGSUpdate(VMCUpdateMethod):
    """A TRUE closure optimizer driven through the seam.

    Unlike the tests-only workaround this seam was created to retire, the
    closure here recomputes the objective and its gradient on every call.
    ``torch.optim.LBFGS`` mutates parameters in place between closure calls;
    each re-evaluation builds a NEW graph from the step's fixed sample, so the
    saved-tensor version error that made a replayed cached loss the only
    option no longer arises.
    """

    def __init__(self, optimizer, model_parameters: ModelParameterBinding) -> None:
        self.optimizer = optimizer
        self.model_parameters = model_parameters
        self.closure_calls = 0

    def update_state(self) -> VMCUpdateState:
        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self.model_parameters = model_parameters

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        def closure() -> torch.Tensor:
            self.closure_calls += 1
            self.optimizer.zero_grad(set_to_none=True)
            objective = update_input.reevaluate()
            objective.backward()
            return objective

        self.optimizer.step(closure)
        return VMCUpdateResult(applied=True, grad_norm=0.0)


class _CapturingUpdate(VMCUpdateMethod):
    """Capture the typed update input without touching the model."""

    def __init__(self, optimizer, model_parameters: ModelParameterBinding) -> None:
        self.optimizer = optimizer
        self.model_parameters = model_parameters
        self.captured: AutogradUpdateInput | None = None

    def update_state(self) -> VMCUpdateState:
        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        self.model_parameters = model_parameters

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        self.captured = update_input
        return VMCUpdateResult(applied=False, grad_norm=0.0)


def _run_trainer(
    *,
    update_method_factory,
    hamiltonian_terms=None,
    nonfinite_local_energy_policy: str = "mask",
    max_steps: int = 1,
    seed: int = 0,
):
    """Run the real ``VMCTrainer`` over the tiny Hooke stack.

    Returns ``(pre_step_model, model, sampler, method, state)``.
    """

    torch.manual_seed(seed)
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    assert sampler.seed is not None, (
        "the smoke fixture stopped supplying a sampler seed: comparing sampler "
        "RNG state across two runs is the whole point of the resume guard"
    )
    pre_step_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    method = update_method_factory(
        optimizer, ModelParameterBinding(parameters=tuple(model.parameters()))
    )
    trainer = VMCTrainer(
        max_steps=max_steps,
        update_method=method,
        nonfinite_local_energy_policy=nonfinite_local_energy_policy,
    )
    state = trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=(
            build_tiny_hamiltonian_terms() if hamiltonian_terms is None else hamiltonian_terms
        ),
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )
    return pre_step_model, model, sampler, method, state


# ---------------------------------------------------------------------------
# P1 FIXED-SAMPLE
# ---------------------------------------------------------------------------


def test_reevaluation_reuses_the_identical_step_sample() -> None:
    """Every re-evaluation forwards THE SAME batch object, never a fresh draw."""

    batch = _batch()
    model = _CountingModel()
    terms = [_ConstantTerm(torch.tensor([2.0], dtype=torch.float64))]
    reevaluate = vmc_objective_reevaluation(
        model=model,
        hamiltonian_terms=terms,
        batch=batch,
        primary_local_energy=torch.full((batch.batch_size,), 2.0, dtype=torch.float64),
    )

    for _ in range(3):
        reevaluate()

    assert model.seen_batches, "the re-evaluation never ran a forward pass"
    # Object identity, not tensor equality: an equal-but-fresh batch is
    # precisely the failure this property forbids, and equality could not
    # tell the two apart.
    assert all(seen is batch for seen in model.seen_batches)


def test_reevaluation_holds_no_sampler_and_therefore_cannot_resample() -> None:
    """The structural half of fixed-sample: no walker source is reachable."""

    batch = _batch()
    reevaluate = vmc_objective_reevaluation(
        model=_CountingModel(),
        hamiltonian_terms=[_ConstantTerm(torch.tensor([2.0], dtype=torch.float64))],
        batch=batch,
        primary_local_energy=_finite_primary(batch),
    )

    assert set(reevaluate.__dataclass_fields__) == {"recompute", "dtype", "device"}
    # The recompute closure is the only place a sampler could hide, so the
    # check reads the cells the code path actually captured rather than
    # trusting the field list alone.
    captured = [cell.cell_contents for cell in (reevaluate.recompute.__closure__ or ())]
    assert not any(isinstance(value, MetropolisSampler) for value in captured)
    assert not any(hasattr(value, "collect_samples") for value in captured)


# ---------------------------------------------------------------------------
# P2 RNG-INERT and P3 SAME-FUNCTION (at unchanged parameters)
# ---------------------------------------------------------------------------


def test_reevaluation_does_not_advance_sampler_or_global_rng_state() -> None:
    """Re-evaluating must leave the resume-relevant RNG state untouched.

    The sampler state is read through the PUBLIC ``mcmc_state_dict``, which is
    the resume payload itself, so this asserts on the same bytes a resumed run
    would restore rather than on a private attribute that merely correlates
    with them.
    """

    _, _, sampler, method, _ = _run_trainer(update_method_factory=_CapturingUpdate)
    update_input = method.captured
    assert update_input is not None

    torch.manual_seed(1234)
    global_before = torch.random.get_rng_state().clone()
    sampler_before = sampler.mcmc_state_dict()["generator_state"].clone()

    for _ in range(3):
        update_input.reevaluate()

    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert torch.equal(sampler.mcmc_state_dict()["generator_state"], sampler_before)


def test_repeated_reevaluation_at_unchanged_parameters_is_bitwise_equal() -> None:
    """SAME-FUNCTION: at fixed parameters the objective is one number.

    This is also the tripwire for the declared RNG-by-absence precondition:
    if ``tpen/physics/`` ever acquires a generator, this test goes red.
    """

    _, _, _, method, _ = _run_trainer(update_method_factory=_CapturingUpdate)
    update_input = method.captured
    assert update_input is not None

    values = [update_input.reevaluate().detach().clone() for _ in range(3)]

    assert torch.equal(values[0], values[1])
    assert torch.equal(values[0], values[2])


# ---------------------------------------------------------------------------
# P4 FRESH after in-place parameter mutation
# ---------------------------------------------------------------------------


def test_reevaluation_survives_in_place_parameter_mutation_that_breaks_the_primary_graph() -> None:
    """The exact defect this seam closes, with the old failure shown beside it."""

    _, model, _, method, _ = _run_trainer(update_method_factory=_CapturingUpdate)
    update_input = method.captured
    assert update_input is not None

    first = update_input.reevaluate()
    first.backward()

    # An in-place parameter mutation is what a closure optimizer does between
    # closure calls.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05)

    # The MATERIALIZED objective is now unusable: its retained graph's saved
    # tensors were versioned before the mutation.  This is the accepted
    # residual the seam exists to retire, asserted rather than described.
    with pytest.raises(RuntimeError, match="inplace operation"):
        update_input.objective.backward()

    # The re-evaluation builds a new graph, so it neither raises nor returns
    # the pre-mutation value.
    second = update_input.reevaluate()
    second.backward()
    assert not torch.equal(first.detach(), second.detach())


# ---------------------------------------------------------------------------
# P5 TRAJECTORY-INERT (the resume guard)
# ---------------------------------------------------------------------------


def test_reevaluating_k_times_leaves_walkers_and_sampler_rng_bitwise_unchanged() -> None:
    """A run that re-evaluates must be indistinguishable from one that does not.

    Parameters, walker positions, and sampler generator state are all compared
    bitwise.  Walkers are included on purpose: the generator state alone would
    still match if re-evaluation had perturbed the accept/reject decisions the
    walkers depend on.
    """

    _, legacy_model, legacy_sampler, _, legacy_state = _run_trainer(
        update_method_factory=lambda optimizer, binding: LegacyAutogradUpdate(
            optimizer, model_parameters=binding
        ),
        max_steps=3,
    )
    _, reeval_model, reeval_sampler, reeval_method, reeval_state = _run_trainer(
        update_method_factory=lambda optimizer, binding: _ReevaluatingLegacyUpdate(
            optimizer, binding, k=3
        ),
        max_steps=3,
    )

    assert len(reeval_method.calls) == 9, "the re-evaluating method did not re-evaluate"

    assert torch.equal(
        legacy_sampler.mcmc_state_dict()["generator_state"],
        reeval_sampler.mcmc_state_dict()["generator_state"],
    )
    assert torch.equal(legacy_state.samples.positions, reeval_state.samples.positions)
    for legacy_parameter, reeval_parameter in zip(
        legacy_model.parameters(), reeval_model.parameters(), strict=True
    ):
        assert torch.equal(legacy_parameter.detach(), reeval_parameter.detach())


# ---------------------------------------------------------------------------
# P6 a real closure optimizer, against an independent oracle
# ---------------------------------------------------------------------------


def test_true_closure_lbfgs_matches_an_independent_fresh_gradient_oracle() -> None:
    """Real ``LBFGS``, ``max_iter > 1``, genuine fresh-gradient closure.

    The shadow arm's objective comes from ``oracle_vmc_objective`` -- a
    separate implementation of the score-function reduction -- so the
    comparison discriminates.  Local energy is recomputed through production
    ``local_energy`` in both arms; that shared factor is stated rather than
    claimed as independent.
    """

    def factory(optimizer, binding):
        return _TrueClosureLBFGSUpdate(optimizer, binding)

    torch.manual_seed(0)
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    pre_step_model = copy.deepcopy(model)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.05, max_iter=4, history_size=4)
    method = factory(optimizer, ModelParameterBinding(parameters=tuple(model.parameters())))
    terms = build_tiny_hamiltonian_terms()
    trainer = VMCTrainer(max_steps=1, update_method=method)
    state = trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )

    assert method.closure_calls > 1, (
        "LBFGS called the closure once, so this never exercised true closure "
        "semantics -- raise max_iter or check the optimizer configuration"
    )

    shadow_model = copy.deepcopy(pre_step_model)
    shadow_optimizer = torch.optim.LBFGS(
        shadow_model.parameters(), lr=0.05, max_iter=4, history_size=4
    )
    shadow_terms = build_tiny_hamiltonian_terms()
    batch = state.batch

    def shadow_closure() -> torch.Tensor:
        shadow_optimizer.zero_grad(set_to_none=True)
        output = shadow_model(batch)
        energy = local_energy(shadow_terms, shadow_model, batch, return_terms=False)
        loss = oracle_vmc_objective([output.logabs], [energy]).loss
        loss.backward()
        return loss

    shadow_optimizer.step(shadow_closure)

    atol = _parameter_tolerance_envelope(state.local_energy.abs())
    for control_parameter, shadow_parameter in zip(
        model.parameters(), shadow_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            control_parameter.detach(), shadow_parameter.detach(), atol=atol, rtol=0.0
        )


def test_true_closure_lbfgs_differs_from_the_cached_loss_workaround() -> None:
    """The seam changed the science, not only the API surface.

    A cached-loss closure -- the only shape the previous seam permitted --
    returns a constant to LBFGS, so its line search has no curvature to read.
    If the fresh-gradient path landed on the same parameters, the seam would
    be decoration.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    pre_step_model = copy.deepcopy(model)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.05, max_iter=4, history_size=4)
    method = _TrueClosureLBFGSUpdate(
        optimizer, ModelParameterBinding(parameters=tuple(model.parameters()))
    )
    trainer = VMCTrainer(max_steps=1, update_method=method)
    state = trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )

    cached_model = copy.deepcopy(pre_step_model)
    cached_optimizer = torch.optim.LBFGS(
        cached_model.parameters(), lr=0.05, max_iter=4, history_size=4
    )
    cached_terms = build_tiny_hamiltonian_terms()
    output = cached_model(state.batch)
    energy = local_energy(cached_terms, cached_model, state.batch, return_terms=False)
    cached_optimizer.zero_grad(set_to_none=True)
    cached_loss = oracle_vmc_objective([output.logabs], [energy]).loss
    cached_loss.backward()
    replayed = cached_loss.detach()
    cached_optimizer.step(lambda: replayed)

    differs = any(
        not torch.equal(fresh.detach(), cached.detach())
        for fresh, cached in zip(model.parameters(), cached_model.parameters(), strict=True)
    )
    assert differs, (
        "the fresh-gradient closure landed on exactly the cached-loss "
        "workaround's parameters, so the seam bought nothing"
    )


# ---------------------------------------------------------------------------
# P7 POLICY-CONTINUOUS and P8 FAIL PROPAGATES
# ---------------------------------------------------------------------------


def test_reevaluation_uses_the_trainers_resolved_nonfinite_policy() -> None:
    """One step must never run under two estimators.

    THIS PROPERTY CHANGED SHAPE UNDER RULING 377f6b0f, and the reason is worth
    stating rather than quietly rewriting the assertion.  It used to be
    observed through OUTCOME: a mid-step non-finite row made the ``"fail"`` arm
    raise and the ``"mask"`` arm return a masked value, so the two arms
    differed only by the policy the seam carried.

    The pinned-row rule now PRE-EMPTS that.  A row that goes non-finite
    mid-step is refused under every policy, before `compute_vmc_objective` is
    reached, so both arms raise for the same reason and the outcome no longer
    discriminates.  Worse, the policy is now UNREACHABLE at compute time in the
    ordinary case: the selection contains exactly the rows that were finite
    when the mask was pinned, and any of those going bad raises, so the sliced
    tensors handed to `compute_vmc_objective` are always all-finite and neither
    policy branch can fire.

    So the property is now observed at the point where the policy is still
    live: it must ARRIVE at the decision point unchanged.  That is a real
    claim, not a weakened one -- a recompute that passed the module default
    here would still be running one step under two estimators the moment any
    future ruling makes the policy load-bearing again.
    """

    seen: list[str] = []

    def capture(*, primary_finite_mask, recomputed_local_energy, policy):
        del recomputed_local_energy
        seen.append(policy)
        return primary_finite_mask

    for policy in ("fail", "mask"):
        _, _, _, method, _ = _run_trainer(
            update_method_factory=_CapturingUpdate,
            hamiltonian_terms=[_ConstantTerm(torch.tensor([1.25], dtype=torch.float64))],
            nonfinite_local_energy_policy=policy,
        )
        assert method.captured is not None
        original = update_module.select_reevaluation_rows
        update_module.select_reevaluation_rows = capture
        try:
            method.captured.reevaluate()
        finally:
            update_module.select_reevaluation_rows = original

    assert seen == ["fail", "mask"], (
        "the trainer's resolved policy must reach the decision point unchanged; "
        f"got {seen}"
    )


def test_nonfinite_reevaluation_under_fail_propagates_out_of_optimizer_step() -> None:
    """P8: the seam propagates a refusal; it does not swallow or downgrade it.

    The refusal now comes from the PINNED-ROW rule of ruling 377f6b0f rather
    than from the ``"fail"`` policy -- the row is finite when the mask is
    pinned and non-finite at the trial point, so it is refused before the
    policy is consulted.  The property under test is unchanged and is the one
    that matters: whatever raises inside the closure must escape
    ``optimizer.step(closure)`` rather than be swallowed by LBFGS.
    """

    batch = _batch(n_walkers=4)
    model = _CountingModel()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.05, max_iter=4)
    reevaluate = vmc_objective_reevaluation(
        model=model,
        hamiltonian_terms=[_AlwaysNonFiniteTerm()],
        batch=batch,
        primary_local_energy=_finite_primary(batch),
        nonfinite_policy="fail",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        objective = reevaluate()
        objective.backward()
        return objective

    with pytest.raises(ValueError, match=r"row\(s\) \[0\]"):
        optimizer.step(closure)


def test_reevaluation_rejects_an_inadmissible_policy_at_construction() -> None:
    """A misspelled policy fails before a run starts, not at the first row."""

    with pytest.raises(ValueError, match="nonfinite local-energy policy must be one of"):
        vmc_objective_reevaluation(
            model=_CountingModel(),
            hamiltonian_terms=[_ConstantTerm(torch.tensor([2.0], dtype=torch.float64))],
            batch=_batch(),
            primary_local_energy=_finite_primary(_batch()),
            nonfinite_policy="mask_but_quietly",
        )


# ---------------------------------------------------------------------------
# Record invariants
# ---------------------------------------------------------------------------


def test_autograd_update_input_requires_a_reevaluation() -> None:
    """The unusable shape -- an objective with no way to recompute it -- is gone."""

    batch = _batch()
    with pytest.raises(TypeError, match="reevaluate"):
        AutogradUpdateInput(
            batch=batch,
            wavefunction=WavefunctionOutput(
                logabs=torch.zeros(batch.batch_size, dtype=torch.float64),
                sign=torch.ones(batch.batch_size, dtype=torch.float64),
            ),
            local_energy=torch.zeros(batch.batch_size, dtype=torch.float64),
            step=0,
            objective=torch.tensor(1.0, dtype=torch.float64),
        )  # type: ignore[call-arg]


def test_reevaluation_rejects_a_recomputed_objective_of_the_wrong_shape_or_dtype() -> None:
    """A silently degraded objective would corrupt curvature one call at a time."""

    non_scalar = ObjectiveReevaluation(
        recompute=lambda: torch.zeros(2, dtype=torch.float64),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="scalar objective"):
        non_scalar()

    wrong_dtype = ObjectiveReevaluation(
        recompute=lambda: torch.zeros((), dtype=torch.float32),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    # NARROW on purpose. `match="dtype"` matched THREE guard messages raised
    # from three different functions, so it confirmed only that something
    # mentioning dtype went wrong -- not that this recompute's own check fired.
    # Cannot-distinguish reads as coverage exactly like cannot-fail does.
    with pytest.raises(ValueError, match="must recompute an objective with dtype"):
        wrong_dtype()


def test_live_reevaluation_and_its_input_cannot_be_serialized() -> None:
    """A record closed over a live model must not cross an artifact boundary."""

    batch = _batch()
    reevaluate = vmc_objective_reevaluation(
        model=_CountingModel(),
        hamiltonian_terms=[_ConstantTerm(torch.tensor([2.0], dtype=torch.float64))],
        batch=batch,
        primary_local_energy=_finite_primary(batch),
    )
    with pytest.raises(RuntimeError, match="live ObjectiveReevaluation cannot be serialized"):
        pickle.dumps(reevaluate)

    detached = AutogradUpdateInput(
        batch=batch,
        wavefunction=WavefunctionOutput(
            logabs=torch.zeros(batch.batch_size, dtype=torch.float64),
            sign=torch.ones(batch.batch_size, dtype=torch.float64),
        ),
        local_energy=torch.zeros(batch.batch_size, dtype=torch.float64),
        step=0,
        objective=torch.tensor(1.0, dtype=torch.float64),
        reevaluate=reevaluate,
    )
    with pytest.raises(RuntimeError, match="live AutogradUpdateInput cannot be serialized"):
        pickle.dumps(detached)


# ---------------------------------------------------------------------------
# The mask decision point (open question 377f6b0f)
# ---------------------------------------------------------------------------


def test_the_decision_point_pins_the_step_mask_rather_than_re_masking() -> None:
    """RULING 377f6b0f PART 1. Replaces the None-returning status quo.

    This test previously asserted ``selection is None`` and existed so that a
    ruling would have to change the line deliberately.  It did; this is that
    change, and the assertion is inverted rather than deleted so the history
    reads as a decision.
    """

    selection = update_module.select_reevaluation_rows(
        primary_finite_mask=torch.tensor([True, True, False, True]),
        recomputed_local_energy=torch.ones(4, dtype=torch.float64),
        policy="mask",
    )
    assert selection is not None
    assert torch.equal(selection, torch.tensor([True, True, False, True]))


def test_a_row_selection_returned_by_the_decision_point_is_actually_applied() -> None:
    """The swap point is wired, not decorative.

    Whichever way ``377f6b0f`` is ruled, the implementer changes one function.
    This proves the recompute HONOURS that function's answer, so a ruling of
    fixed-mask-per-step is a change at the named seam and nothing else.  It
    also pins the exact input the decision receives: the primary step's finite
    mask, detached, plus the freshly recomputed local energy.
    """

    batch = _batch(n_walkers=4)
    model = _CountingModel()
    terms = [_VaryingTerm()]
    # Row 2 was non-finite when the primary objective was formed.
    primary = torch.tensor([1.5, 1.5, float("inf"), 1.5], dtype=torch.float64)
    reevaluate = vmc_objective_reevaluation(
        model=model,
        hamiltonian_terms=terms,
        batch=batch,
        primary_local_energy=primary,
    )

    seen: list[tuple[torch.Tensor, torch.Tensor, str]] = []

    # Deliberately NOT `primary_finite_mask`. Ruling 377f6b0f made that the
    # real function's own answer, so returning it here would compare the
    # recompute against itself and the difference assertion below could not
    # fail. Dropping row 0 as well keeps the injected answer distinguishable
    # from the production one, which is the only way this test can still show
    # that the recompute HONOURS what the decision point returns.
    def fixed_mask(*, primary_finite_mask, recomputed_local_energy, policy):
        seen.append((primary_finite_mask, recomputed_local_energy, policy))
        narrower = primary_finite_mask.clone()
        narrower[0] = False
        return narrower

    original = update_module.select_reevaluation_rows
    update_module.select_reevaluation_rows = fixed_mask
    try:
        pinned = reevaluate()
    finally:
        update_module.select_reevaluation_rows = original

    assert len(seen) == 1
    observed_mask, observed_energy, observed_policy = seen[0]
    assert torch.equal(observed_mask, torch.tensor([True, True, False, True]))
    assert observed_mask.dtype == torch.bool
    assert observed_energy.shape == (4,)
    assert observed_policy == "mask"

    # Asserted against an EXPLICIT three-row reduction, not merely "the two
    # differ".  The first version of this test only checked difference, and
    # difference was structurally impossible for its fixture -- so it went red
    # on a true statement while measuring nothing about the selection.
    reference_output = model(batch)
    reference_energy = local_energy(terms, model, batch, return_terms=False)
    mask = torch.tensor([False, True, False, True])
    expected = compute_vmc_objective(
        reference_output.logabs[mask],
        reference_energy[mask],
        nonfinite_policy="mask",
    ).loss
    assert torch.equal(pinned.detach(), expected.detach())

    unpinned = reevaluate()
    assert not torch.equal(pinned.detach(), unpinned.detach())


# ---------------------------------------------------------------------------
# Guards on NON-FACTORY construction (E2), and the construction-time mask (E3)
# ---------------------------------------------------------------------------


def _detached_input(
    batch: ElectronBatch,
    reevaluate: ObjectiveReevaluation,
) -> AutogradUpdateInput:
    """Build an input directly, bypassing the trainer and the factory."""

    return AutogradUpdateInput(
        batch=batch,
        wavefunction=WavefunctionOutput(
            logabs=torch.zeros(batch.batch_size, dtype=torch.float64),
            sign=torch.ones(batch.batch_size, dtype=torch.float64),
        ),
        local_energy=torch.zeros(batch.batch_size, dtype=torch.float64),
        step=0,
        objective=torch.tensor(1.0, dtype=torch.float64),
        reevaluate=reevaluate,
    )


def test_input_rejects_a_reevaluation_disagreeing_with_the_objective_dtype() -> None:
    """Guards the NON-FACTORY route, which is the only one that can reach here.

    Through `vmc_objective_reevaluation` these checks cannot fire: it derives
    dtype and device from the batch, and `VMCStepData.validate` already forces
    batch/logabs/objective agreement.  `ObjectiveReevaluation` is directly
    constructible though, and the suite's own `_reevaluation` helper builds one
    without the factory -- so the guard protects a real route and the defect
    was that nothing exercised it.  Recorded because "unreachable via the
    factory" answers a narrower question than "unreachable", and concluding
    the latter from the former would delete live protection.
    """

    batch = _batch()
    wrong_dtype = ObjectiveReevaluation(
        recompute=lambda: torch.zeros((), dtype=torch.float32),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="reevaluation and objective must share one dtype"):
        _detached_input(batch, wrong_dtype)


def test_input_rejects_a_reevaluation_disagreeing_with_the_objective_device() -> None:
    """The device half of the same guard, isolated by matching dtype.

    The device is never dereferenced by this check, only compared, so naming a
    device absent from the machine is a legitimate way to reach the branch on
    a CPU-only runner rather than a skip.
    """

    batch = _batch()
    wrong_device = ObjectiveReevaluation(
        recompute=lambda: torch.zeros((), dtype=torch.float64),
        dtype=torch.float64,
        device=torch.device("cuda"),
    )
    with pytest.raises(ValueError, match="reevaluation and objective must share one device"):
        _detached_input(batch, wrong_device)


def test_the_retained_primary_mask_is_fixed_at_construction() -> None:
    """The mask describes the PRIMARY step, not the caller's tensor now.

    Replaces a vacuous `assert not mask.requires_grad`: `torch.isfinite`
    returns a bool tensor, which can never require grad even from a
    grad-requiring input, so that assertion could not fail for any input. This
    asserts the property the code actually needs and that a plausible
    implementation would break -- storing `primary_local_energy` and computing
    `isfinite` lazily inside `recompute` would make the mask follow the
    caller's later mutations.
    """

    batch = _batch(n_walkers=4)
    primary = torch.tensor([1.5, 1.5, float("inf"), 1.5], dtype=torch.float64)
    reevaluate = vmc_objective_reevaluation(
        model=_CountingModel(),
        hamiltonian_terms=[_VaryingTerm()],
        batch=batch,
        primary_local_energy=primary,
    )

    # Mutate the caller's tensor IN PLACE after construction, in both
    # directions, so a lazily recomputed mask would differ in two entries
    # rather than one and could not coincide with the correct answer.
    with torch.no_grad():
        primary[0] = float("inf")
        primary[2] = 0.0

    seen: list[torch.Tensor] = []

    def capture(*, primary_finite_mask, recomputed_local_energy, policy):
        del recomputed_local_energy, policy
        seen.append(primary_finite_mask)
        return None

    original = update_module.select_reevaluation_rows
    update_module.select_reevaluation_rows = capture
    try:
        reevaluate()
    finally:
        update_module.select_reevaluation_rows = original

    assert len(seen) == 1
    assert seen[0].dtype == torch.bool
    assert torch.equal(seen[0], torch.tensor([True, True, False, True]))


# ---------------------------------------------------------------------------
# Ruling 377f6b0f: fixed row mask per step, refuse a pinned row that goes bad
#
# Four claims, and each dies to a mutant that kills IT AND NOTHING ELSE:
#   M1 pin-not-applied  (`return None`, RAISE KEPT)          -> only the first
#   M2 raise-suppressed (delete the raise)                   -> only the second
#   M4 repin-to-current (`return isfinite(recomputed)`)      -> only the third
#   M3 raise-too-broad  (drop the `primary_finite_mask &`)   -> only the fourth
# Getting there required a design change, not bookkeeping: the stability test
# asserts the row set is stable ACROSS TRIAL POINTS rather than equal to the
# primary mask, so M4 passes it and the third test owns the equality claim
# alone.  The obvious version had two tests dying to one mutant.
# ---------------------------------------------------------------------------


_F = torch.float64


def _select(primary: torch.Tensor, recomputed: torch.Tensor, policy: str = "mask"):
    """Call the decision point the way the recompute does."""

    return update_module.select_reevaluation_rows(
        primary_finite_mask=torch.isfinite(primary),
        recomputed_local_energy=recomputed,
        policy=policy,
    )


def test_the_row_set_is_stable_across_trial_points_within_one_step() -> None:
    """A closure optimizer moves the parameters; the scored rows must not."""

    primary = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=_F)
    first = _select(primary, torch.tensor([1.5, 2.5, 3.5, 4.5], dtype=_F))
    second = _select(primary, torch.tensor([9.0, -3.0, 0.25, 7.0], dtype=_F))

    assert first is not None and second is not None
    assert torch.equal(first, second)


def test_a_pinned_row_going_non_finite_raises_and_names_the_row() -> None:
    """The message must identify the rows, not merely be an exception.

    A bare ``pytest.raises(ValueError)`` would pass on an unrelated ValueError
    raised nearby, which is a defect this project has paid for before.
    """

    primary = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=_F)
    recomputed = torch.tensor([1.0, float("inf"), 3.0, float("nan")], dtype=_F)

    with pytest.raises(ValueError, match=r"row\(s\) \[1, 3\]"):
        _select(primary, recomputed)


def test_a_row_excluded_at_the_step_start_stays_excluded_when_it_recovers() -> None:
    """Pinning is per STEP, not per call: recovery must not widen the mask.

    Widening mid-search changes the estimator exactly as shrinking does, and it
    is the direction a naive ``isfinite(recomputed)`` gets wrong.
    """

    primary = torch.tensor([1.0, float("inf"), 3.0, 4.0], dtype=_F)
    recomputed = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=_F)

    selection = _select(primary, recomputed)

    assert selection is not None
    assert torch.equal(selection, torch.tensor([True, False, True, True]))


def test_a_row_already_non_finite_at_the_step_start_does_not_raise() -> None:
    """Only rows that WERE finite and went bad are a failure.

    A row non-finite from the outset was never pinned in, so it is excluded by
    the mask and must not trigger the refusal.  Dropping the
    ``primary_finite_mask &`` conjunct turns this benign case into a raise.
    """

    primary = torch.tensor([1.0, float("nan"), 3.0, 4.0], dtype=_F)
    recomputed = torch.tensor([1.0, float("nan"), 3.0, 4.0], dtype=_F)

    selection = _select(primary, recomputed)

    assert selection is not None
    assert torch.equal(selection, torch.tensor([True, False, True, True]))


def test_no_propagate_opt_out_is_reachable_from_this_seam() -> None:
    """The proof obligation is structural: there is no channel to enable it.

    Ruling 377f6b0f permits ``propagate-non-finite`` only where an optimizer's
    admission carries a test proving its search rejects non-finite trial values
    and terminates.  No such admission exists, so no channel does either.  This
    pins that, so a later change adding a boolean anyone can set fails here
    before it reaches review.

    DELIBERATELY HAS NO MUTANT.  It guards a CLASS of change rather than a
    behaviour, so a mutant for it would be theatre -- and inventing one to keep
    the table symmetrical is how a suite acquires tests that pass for reasons
    nobody checked.
    """

    import inspect

    signature = inspect.signature(update_module.select_reevaluation_rows)
    assert set(signature.parameters) == {
        "primary_finite_mask",
        "recomputed_local_energy",
        "policy",
    }


def test_the_default_mask_policy_now_refuses_rather_than_re_masking() -> None:
    """The behaviour change, pinned as a test rather than only described.

    Runs under the DEFAULT policy -- no ``"fail"`` anywhere -- because the
    change people will actually meet is under ``"mask"``.  Before the ruling
    this returned a value with row 0 quietly dropped mid-search.
    """

    batch = _batch(n_walkers=4)
    reevaluate = vmc_objective_reevaluation(
        model=_CountingModel(),
        # ALWAYS non-finite, not after-first-call: a direct factory call has no
        # primary evaluation, so its first call IS the re-evaluation. The
        # after-first-call term stays finite there and the refusal never fires.
        hamiltonian_terms=[_AlwaysNonFiniteTerm()],
        batch=batch,
        primary_local_energy=_finite_primary(batch),
    )
    with pytest.raises(ValueError, match=r"row\(s\) \[0\]"):
        reevaluate()


def test_the_pinned_row_raise_escapes_the_trainer_through_optimizer_step() -> None:
    """Testing the seam is not testing the seam's effect.

    Every other test here calls the decision point or the factory directly, so
    none shows the raise reaching a caller.  That is the whole point of the
    ruling: a refusal that never escapes is indistinguishable from the silent
    masking it replaces.  Every hop below is production code --
    ``VMCTrainer.fit`` -> update method -> ``optimizer.step(closure)`` ->
    ``reevaluate()`` -> ``select_reevaluation_rows``.
    """

    torch.manual_seed(0)
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.05, max_iter=4)
    method = _TrueClosureLBFGSUpdate(
        optimizer, ModelParameterBinding(parameters=tuple(model.parameters()))
    )
    trainer = VMCTrainer(max_steps=1, update_method=method)

    with pytest.raises(ValueError, match=r"row\(s\) \[0\]"):
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=[_NonFiniteAfterFirstCallTerm()],
            optimizer=optimizer,
            context=_StubContext(),
            emit=lambda name, *, state=None, payload=None, step=None: None,
        )

    assert method.closure_calls >= 1, (
        "the closure never ran, so this proved nothing about the escape path"
    )
