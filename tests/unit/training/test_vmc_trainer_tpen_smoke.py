"""Smoke test: one VMC trainer step over the real tiny TPEN stack."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tpen.artifacts import RunContext
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.events import DomainState, Ended, Occurrence, Started
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.sampling import SamplerStats
from tpen.training.events import (
    Backward,
    BuildBatch,
    CollectSamples,
    Forward,
    LocalEnergy,
    Metrics,
    Objective,
    OptimizerUpdate,
    TrainingIteration,
    TrainingIterationCompleted,
    UpdateCompleted,
    UpdateSkipped,
)
from tpen.training.state import TrainerState
from tpen.training.trainer import VMCTrainer, _parameter_norm
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn


class _StubContext(RunContext):
    """Minimal RunContext subclass: satisfies typing, logs to a list."""

    def __init__(self) -> None:
        self.callbacks = []
        self.loggers = []
        self.metadata = SimpleNamespace(device="cpu", dtype="float64")
        self.records: list[tuple[str, dict]] = []
        self.occurrences: list[Occurrence[Any]] = []
        self.states: list[DomainState | None] = []
        self.trace: list[tuple[str, object]] = []
        self._occurrence_counts = {}

    def log(self, metrics, *, step=None, namespace="run") -> None:
        self.records.append((namespace, dict(metrics)))

    def _dispatch_occurrence(
        self, occurrence: Occurrence[Any], *, state: DomainState | None = None
    ) -> None:
        # ``state`` is accepted and recorded but never asserted on here: this
        # double stands in for the dispatch sink, and the trainer now hands its
        # `TrainerState` to every typed boundary.
        self.occurrences.append(occurrence)
        self.states.append(state)
        self.trace.append(("typed", occurrence.event))


class _VacuumWalkers:
    """Zero-electron walkers: the vacuum has no coordinate degrees of freedom."""

    def __init__(self, n_walkers: int = 2) -> None:
        self.n_walkers = int(n_walkers)

    def make_batch(self) -> ElectronBatch:
        return ElectronBatch(
            positions=torch.zeros(self.n_walkers, 0, 3, dtype=torch.float64),
            spins=torch.zeros(self.n_walkers, 0, dtype=torch.float64),
        )


class _VacuumSampler:
    """Sampler stand-in returning one fixed zero-electron walker set."""

    def collect_samples(self, model, *, device=None):
        del model, device
        walkers = _VacuumWalkers()
        stats = SamplerStats(
            acceptance_rate=1.0,
            n_walkers=walkers.n_walkers,
            burn_in=0,
            n_steps=0,
            proposal_scale=0.0,
        )
        return walkers, stats


class _ConstantWavefunction(torch.nn.Module):
    """Constant wavefunction whose trainable parameter cannot reach ``logabs``.

    This reproduces the condition the trainer's vacuum branch guards: the loss
    is disconnected from the parameters, so ``loss.requires_grad`` is ``False``
    and no optimizer update is possible.
    """

    def __init__(self) -> None:
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        shape = batch.sample_shape
        return WavefunctionOutput(
            logabs=torch.zeros(shape, dtype=torch.float64),
            sign=torch.ones(shape, dtype=torch.float64),
        )


class _NormSpyAdam(torch.optim.Adam):
    """Adam that records the model's parameter norm on both sides of ``step``.

    Subclassing the real optimizer is deliberate: ``VMCTrainer.fit`` annotates
    ``optimizer`` as ``torch.optim.Optimizer`` and the suite runs typeguard over
    ``tpen``, so a duck-typed wrapper would be rejected at call time.
    """

    def __init__(self, model: torch.nn.Module, **kwargs: Any) -> None:
        super().__init__(model.parameters(), **kwargs)
        self._model = model
        self.norm_before_step: float | None = None
        self.norm_after_step: float | None = None

    def step(self, closure=None):
        self.norm_before_step = _parameter_norm(self._model)
        result = super().step(closure)
        self.norm_after_step = _parameter_norm(self._model)
        return result


def _train_metrics(context: _StubContext) -> dict:
    """Return the single ``train`` record logged into ``context``."""

    records = [metrics for namespace, metrics in context.records if namespace == "train"]
    assert len(records) == 1, f"expected exactly one train record, got {len(records)}"
    return records[0]


def _occurrence_labels(occurrences: list[Occurrence[Any]]) -> list[tuple[str, type]]:
    """Return one ``(boundary, concrete type)`` label per typed occurrence."""

    labels: list[tuple[str, type]] = []
    for occurrence in occurrences:
        event = occurrence.event
        if isinstance(event, Started):
            labels.append(("started", type(event.operation)))
        elif isinstance(event, Ended):
            labels.append(("ended", type(event.operation)))
        else:
            labels.append(("event", type(event)))
    return labels


def _record_legacy(context: _StubContext):
    """Return an ``emit`` callable that only records legacy event names."""

    def emit(name: str, *, state=None, payload=None, step=None) -> None:
        del state, payload, step
        context.trace.append(("legacy", name))

    return emit


def _fit_one_typed_step(context: _StubContext, *, emit=None) -> VMCTrainer:
    """Drive one real tiny-TPEN training iteration through ``context``."""

    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=context,
        emit=_record_legacy(context) if emit is None else emit,
    )
    return trainer


def _fit_vacuum_steps(context: _StubContext, *, max_steps: int = 1) -> VMCTrainer:
    """Drive ``max_steps`` zero-electron iterations that must skip their update."""

    model = _ConstantWavefunction()
    trainer = VMCTrainer(max_steps=max_steps, log_every_n_steps=1)
    trainer.fit(
        model=model,
        sampler=_VacuumSampler(),
        # The vacuum has no kinetic or interaction contribution to sum.
        hamiltonian_terms=[],
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        context=context,
        emit=_record_legacy(context),
    )
    return trainer


_FORBIDDEN_METRICS = {
    "reference_energy",
    "energy_error",
    "energy_abs_error",
    "abs_energy_error",
    "exact_energy",
    "expected_energy",
}


def test_one_vmc_step_is_finite_and_vmc_native() -> None:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, return_terms=True)

    state = trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )

    assert math.isfinite(float(state.loss))
    assert torch.isfinite(state.local_energy).all()
    # Native VMC metrics only -- no reference/exact comparison leaks in.
    assert _FORBIDDEN_METRICS.isdisjoint(state.metrics)
    for key in ("energy", "loss", "grad_norm"):
        assert key in state.metrics


def _fit_one_step(*, return_terms: bool, terms) -> object:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, return_terms=return_terms)
    return trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )


def test_vmc_trainer_uses_canonical_objective_metrics() -> None:
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    state = _fit_one_step(return_terms=False, terms=terms)

    # Physical training estimator is logged as `energy`, never `energy_mean`.
    assert "energy" in state.metrics
    assert "energy_mean" not in state.metrics
    for key in (
        "loss",
        "energy_variance",
        "energy_std",
        "energy_stderr",
        "local_energy_n_finite",
        "local_energy_n_total",
        "local_energy_finite_fraction",
        "local_energy_nonfinite_count",
    ):
        assert key in state.metrics
    # No per-term metrics when return_terms is disabled.
    assert not any(key.startswith("energy_term_") for key in state.metrics)


def test_vmc_trainer_logs_term_metrics_when_return_terms_enabled() -> None:
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    state = _fit_one_step(return_terms=True, terms=terms)

    # A list of terms falls back to snake-case class names for the metric keys.
    expected_names = ("kinetic_energy", "harmonic_trap", "electron_electron_interaction")
    for name in expected_names:
        prefix = f"energy_term_{name}"
        assert prefix in state.metrics
        assert f"{prefix}_variance" in state.metrics


def test_vmc_trainer_scopes_every_training_phase() -> None:
    legacy_events: list[tuple[str, int | None]] = []
    context = _StubContext()

    def emit(name: str, *, state=None, payload=None, step=None) -> None:
        del state
        context.trace.append(("legacy", name))
        legacy_events.append((name, step))
        # The legacy step is explicit; it is never duplicated in the payload.
        assert payload is None or "step" not in payload

    _fit_one_typed_step(context, emit=emit)

    # No legacy string emissions survive the typed lifecycle migration.
    assert legacy_events == []
    # Phases are sequential and non-nested inside one iteration scope.
    assert _occurrence_labels(context.occurrences) == [
        ("started", TrainingIteration),
        ("started", CollectSamples),
        ("ended", CollectSamples),
        ("started", BuildBatch),
        ("ended", BuildBatch),
        ("started", LocalEnergy),
        ("ended", LocalEnergy),
        ("started", Forward),
        ("ended", Forward),
        ("started", Objective),
        ("ended", Objective),
        ("started", Backward),
        ("ended", Backward),
        ("started", OptimizerUpdate),
        ("ended", OptimizerUpdate),
        ("event", UpdateCompleted),
        ("started", Metrics),
        ("ended", Metrics),
        ("event", TrainingIterationCompleted),
        ("ended", TrainingIteration),
    ]
    # One iteration means every concrete type is at its first occurrence.
    assert [occurrence.count for occurrence in context.occurrences] == [1] * 20
    completion_index = next(
        index
        for index, (kind, event) in enumerate(context.trace)
        if kind == "typed" and isinstance(event, TrainingIterationCompleted)
    )
    iteration_end_index = next(
        index
        for index, (kind, event) in enumerate(context.trace)
        if kind == "typed"
        and isinstance(event, Ended)
        and isinstance(event.operation, TrainingIteration)
    )
    assert completion_index < iteration_end_index


def test_vmc_trainer_step_end_failure_skips_completion_but_ends_iteration() -> None:
    model = build_tiny_spenn()

    class _FailingSampler:
        def collect_samples(self, model, *, device=None):
            del model, device
            raise RuntimeError("sample collection failed")

    sampler = _FailingSampler()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    context = _StubContext()

    with pytest.raises(RuntimeError, match="sample collection failed"):
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=terms,
            optimizer=optimizer,
            context=context,
            emit=lambda *args, **kwargs: pytest.fail("legacy emit site was used"),
        )

    assert not any(
        isinstance(occurrence.event, TrainingIterationCompleted)
        for occurrence in context.occurrences
    )
    assert isinstance(context.occurrences[-1].event, Ended)
    assert isinstance(context.occurrences[-1].event.operation, TrainingIteration)


def test_update_completed_is_emitted_after_the_optimizer_update_scope() -> None:
    context = _StubContext()
    trainer = _fit_one_typed_step(context)
    labels = _occurrence_labels(context.occurrences)

    # The update is only complete once its scope has closed.
    assert labels.index(("event", UpdateCompleted)) == (
        labels.index(("ended", OptimizerUpdate)) + 1
    )
    assert ("event", UpdateSkipped) not in labels
    # A completed update advances both counters together.
    assert (trainer.next_iteration, trainer.completed_updates) == (1, 1)


def test_skipped_update_advances_next_iteration_but_not_completed_updates() -> None:
    trainer = _fit_vacuum_steps(_StubContext(), max_steps=2)

    assert trainer.next_iteration == 2
    assert trainer.completed_updates == 0
    assert trainer.state_dict()["next_iteration"] == 2
    assert trainer.state_dict()["completed_updates"] == 0
    assert "parameter_layout" in trainer.state_dict()


def test_skipped_update_emits_update_skipped_without_an_optimizer_scope() -> None:
    context = _StubContext()

    _fit_vacuum_steps(context)
    labels = _occurrence_labels(context.occurrences)

    assert ("event", UpdateSkipped) in labels
    assert ("event", UpdateCompleted) not in labels
    # The skip path never opens an OptimizerUpdate scope at all.
    assert not any(operation is OptimizerUpdate for _, operation in labels)
    # Backward is likewise unreachable without a differentiable loss.
    assert not any(operation is Backward for _, operation in labels)


def test_param_norm_describes_the_pre_update_model() -> None:
    """One `train` record must describe exactly one model version."""

    context = _StubContext()
    model = build_tiny_spenn()
    optimizer = _NormSpyAdam(model, lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)

    trainer.fit(
        model=model,
        sampler=build_tiny_sampler(),
        hamiltonian_terms=[
            KineticEnergy(),
            HarmonicTrap(omega=0.5),
            ElectronElectronInteraction(),
        ],
        optimizer=optimizer,
        context=context,
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )

    # Precondition: the update actually moved the parameters. Without this the
    # equality below would also hold for a post-update reading and pin nothing.
    assert optimizer.norm_before_step != optimizer.norm_after_step

    metrics = _train_metrics(context)
    assert metrics["param_norm"] == optimizer.norm_before_step
    assert metrics["param_norm"] != optimizer.norm_after_step
    # Moving the computation earlier must not move the key: CSV/JSONL readers
    # and anyone diffing records see the same column order as before.
    #
    # Expressed as a CONTIGUOUS RUN rather than as "the last four keys". The
    # `[-4:]` form this started as was never the invariant it appeared to be:
    # a method that reports diagnostics has always had them merged in after
    # `optimizer_step`, so the proxy held only on the legacy path, which
    # happened to report nothing. The legacy adapter now reports an explicit
    # minimal record too, so the proxy holds nowhere and the property it stood
    # for has to be stated directly. This is a strengthening: it pins the
    # trainer-owned columns' order AND what may follow them.
    keys = list(metrics)
    start = keys.index("grad_norm")
    assert keys[start : start + 4] == [
        "grad_norm",
        "param_norm",
        "loss_has_grad",
        "optimizer_step",
    ]
    # Everything after that run is the attempt's own reporting: the trainer's
    # generic reason, then the method-owned record. Appended, never
    # interleaved, so the trainer-owned columns keep their positions.
    assert keys[start + 4 :] == [
        "update_reason",
        "update_record_method",
        "update_record_applied",
        "update_record_reason",
        "update_record_step",
        "update_record_grad_norm",
    ]


def test_skipped_update_logs_the_same_train_record_shape() -> None:
    """A vacuum skip emits the normal record; `optimizer_step` is the discriminator."""

    completed_context = _StubContext()
    _fit_one_typed_step(completed_context)
    skipped_context = _StubContext()
    _fit_vacuum_steps(skipped_context)

    completed = _train_metrics(completed_context)
    skipped = _train_metrics(skipped_context)

    # Same keys in the same order, so no consumer has to branch on whether an
    # iteration happened to apply an update.
    assert list(skipped) == list(completed)
    assert completed["optimizer_step"] is True
    assert skipped["optimizer_step"] is False
    # The skip path opens no optimizer scope, yet still reports a parameter
    # norm -- and on that path pre-update and post-update coincide anyway.
    assert skipped["param_norm"] == pytest.approx(0.0)
    assert skipped["grad_norm"] == 0.0


def test_the_state_carries_the_optimizer_step_discriminator() -> None:
    """The typed field beside the metric, and it must agree with the metric.

    A health callback observing update by-products cannot tell "no gradients
    because no update ran" from "no gradients because they were already cleared"
    without this field; `tpen.callback.GradientStats` reads it for exactly that.
    Because the callback reads only the state, an assignment lost from the loop
    body would make its check unfalsifiable while every metric stayed correct --
    so the agreement is asserted here, at the one place both spellings are set.
    """

    completed_context = _StubContext()
    _fit_one_typed_step(completed_context)
    skipped_context = _StubContext()
    _fit_vacuum_steps(skipped_context)

    completed = _train_metrics(completed_context)
    skipped = _train_metrics(skipped_context)

    def observed_state(context: _StubContext) -> TrainerState:
        """Return the state handed to the completed-iteration boundary.

        The trainer mutates one state object in place all run long, so this is
        only unambiguous because both fits above run a single step.
        """

        states = [
            state
            for occurrence, state in zip(context.occurrences, context.states, strict=True)
            if isinstance(occurrence.event, TrainingIterationCompleted)
        ]
        assert len(states) == 1, f"expected one completed iteration, got {len(states)}"
        state = states[0]
        assert isinstance(state, TrainerState)
        return state

    assert observed_state(completed_context).optimizer_step is True
    assert observed_state(skipped_context).optimizer_step is False
    # One fact, one spelling on each side of the durable boundary.
    assert observed_state(completed_context).optimizer_step is completed["optimizer_step"]
    assert observed_state(skipped_context).optimizer_step is skipped["optimizer_step"]


def test_load_state_dict_round_trips_both_progress_counters() -> None:
    trainer = VMCTrainer(max_steps=5)

    trainer.load_state_dict({"next_iteration": 4, "completed_updates": 3})

    assert trainer.next_iteration == 4
    assert trainer.completed_updates == 3
    state = trainer.state_dict()
    # No model/update authority is resolved in this counter-only test, so no
    # parameter layout is available; the exact key set still pins this schema.
    assert set(state) == {
        "next_iteration",
        "completed_updates",
        "nonfinite_local_energy_policy",
    }
    # `nonfinite_local_energy_policy` records the ACTIVE estimator beside the
    # progress counters. Exact equality is kept rather than loosened to a
    # subset check: an unexpected new key in trainer state is precisely what
    # these tests should catch, so weakening the assertion to accommodate a
    # change would remove the guard to fit the thing it guards against.
    assert state == {
        "next_iteration": 4,
        "completed_updates": 3,
        "nonfinite_local_energy_policy": "mask",
    }


def test_load_state_dict_rejects_legacy_trainer_state() -> None:
    """Pre-rename checkpoints must fail loudly, never resume from step 0."""

    trainer = VMCTrainer(max_steps=5)

    with pytest.raises(KeyError, match="next_iteration"):
        trainer.load_state_dict({"global_step": 3, "completed_steps": 3})
    with pytest.raises(KeyError, match="completed_updates"):
        trainer.load_state_dict({"next_iteration": 3})

    assert trainer.state_dict() == {
        "next_iteration": 0,
        "completed_updates": 0,
        "nonfinite_local_energy_policy": "mask",
    }


def test_load_state_dict_leaves_trainer_unmutated_on_a_non_integer_value() -> None:
    """A rejected value must not half-restore the resume cursor."""

    trainer = VMCTrainer(max_steps=5)
    trainer.load_state_dict({"next_iteration": 4, "completed_updates": 3})

    # `next_iteration` is coerced first, so a bad `completed_updates` is the
    # case that would leave a partially mutated trainer behind.
    with pytest.raises(ValueError):
        trainer.load_state_dict({"next_iteration": 5, "completed_updates": "x"})

    assert trainer.state_dict() == {
        "next_iteration": 4,
        "completed_updates": 3,
        "nonfinite_local_energy_policy": "mask",
    }
