"""Tests for the composable Evaluate runner."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import tpen.run as run_module
import tpen.runner as runner_module
import tpen.runner.evaluate as evaluate_runner_module
import tpen.runner.train as train_runner_module
from tpen.artifacts import RunContext
from tpen.callback import Callback, SubscriptionGroup
from tpen.checkpoint import RestoreReport
from tpen.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from tpen.evaluation import (
    EvaluationTask,
    Evaluator,
    HamiltonianTermSummary,
    LocalEnergyCalculator,
    LocalEnergySummary,
    MCMCGenerator,
    ReferenceEnergySummary,
    SamplerStatsSummary,
    WavefunctionCalculator,
)
from tpen.evaluation.events import (
    ComponentFailed,
    ComponentRun,
    EvaluationCompleted,
    EvaluationStarted,
    EvaluationTaskRun,
)
from tpen.events import Ended, Started, Subscription, ended, started
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.run import run_from_config
from tpen.runner import Evaluate, Train
from tpen.sampling import SamplerStats
from tpen.training.state import TrainerState
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    VMCUpdateMethod,
    VMCUpdateResult,
)
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn
from tests.helpers.run_context import RecordingLogger, make_run_context

FIXTURES = Path(__file__).resolve().parents[1] / "artifacts" / "hooke"


def test_evaluate_constructor_uses_evaluator_boundary() -> None:
    params = set(inspect.signature(Evaluate.__init__).parameters)
    assert params == {"self", "model", "load", "evaluator"}


def test_evaluate_requires_evaluator() -> None:
    with pytest.raises(ValueError, match="requires an evaluator"):
        Evaluate(model=None)


@pytest.mark.parametrize("fixture", ["exact_singlet_eval.yaml", "exact_triplet_eval.yaml"])
def test_evaluate_config_is_root_owned_and_uses_evaluator(fixture: str) -> None:
    cfg = OmegaConf.load(FIXTURES / fixture)
    assert "callbacks" in cfg and "loggers" in cfg
    assert "callbacks" not in cfg.runner
    assert "loggers" not in cfg.runner
    assert "exact_energy" not in cfg.system
    assert "phase" not in cfg.evaluation

    raw = OmegaConf.to_container(cfg, resolve=False)
    assert raw["runner"]["evaluator"] == "${evaluator}"
    assert raw["evaluator"]["namespace"] == "${evaluation.namespace}"
    assert raw["evaluator"]["tasks"] == ["${evaluation_tasks.energy}"]
    assert raw["evaluation_tasks"]["energy"]["generator"]["_target_"] == "tpen.evaluation.generators.MCMCGenerator"
    assert raw["evaluation_tasks"]["energy"]["summaries"][-1]["_target_"] == "tpen.evaluation.summaries.ReferenceEnergySummary"


def test_instantiate_runner_uses_normal_hydra_recursion_for_evaluator(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("_instantiate_runner must not special-case evaluation tasks")

    monkeypatch.setattr(run_module, "_instantiate_sequence", fail_if_called)
    cfg = OmegaConf.create(
        {
            "runner": {
                "_target_": "tpen.runner.Evaluate",
                "model": None,
                "evaluator": {
                    "_target_": "tpen.evaluation.Evaluator",
                    "namespace": "eval",
                    "tasks": [],
                },
            }
        }
    )

    runner = run_module._instantiate_runner(_runner_context(cfg))

    assert isinstance(runner, Evaluate)
    assert isinstance(runner.evaluator, Evaluator)


def test_train_config_with_evaluator_fails_as_normal_constructor_error() -> None:
    cfg = OmegaConf.create(
        {
            "runner": {
                "_target_": "tpen.runner.Train",
                "model": None,
                "sampler": None,
                "hamiltonian_terms": [],
                "optimizer": None,
                "trainer": None,
                "evaluator": {
                    "_target_": "tpen.evaluation.Evaluator",
                    "namespace": "eval",
                    "tasks": [],
                },
            }
        }
    )

    with pytest.raises(Exception, match="evaluator"):
        run_module._instantiate_runner(_runner_context(cfg))


def test_runtime_dtype_rejects_non_floating_dtype() -> None:
    with pytest.raises(ValueError, match="floating point"):
        runner_module._runtime_dtype("int64")


def test_train_asserts_eager_initialization_before_optimizer_construction(tmp_path: Path) -> None:
    class _OptimizerFactory:
        called = False

        def __call__(self, params):
            self.called = True
            raise AssertionError("optimizer should not be constructed")

    optimizer = _OptimizerFactory()
    recorder = _TypedOccurrenceRecorder()
    context, _ = _recording_context(tmp_path, [recorder])
    runner = Train(
        model=nn.LazyLinear(1),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=optimizer,
        trainer=object(),
    )

    with pytest.raises(RuntimeError, match="uninitialized"):
        runner.run(context)

    assert optimizer.called is False
    assert recorder.seen == []


def test_evaluation_started_is_emitted_after_model_ready(tmp_path: Path) -> None:
    recorder = _TypedOccurrenceRecorder()
    typed = _TypedOccurrenceRecorder()
    context, _ = _recording_context(tmp_path, [recorder, typed])
    runner = Evaluate(
        model=nn.LazyLinear(1),
        evaluator=_energy_evaluator(_StaticSampler(torch.zeros(1, 2, 1, dtype=torch.float64)), []),
    )

    with pytest.raises(RuntimeError, match="uninitialized"):
        runner.run(context)

    assert recorder.seen == []
    # The lazy model is rejected before evaluation begins, so the typed suite
    # boundary is never reached either.
    assert typed.seen == []


def test_train_rejects_model_only_load_mode(tmp_path: Path) -> None:
    runner = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"mode": "model_only", "path": "unused"},
    )

    with pytest.raises(ValueError, match="load.mode.*model_only"):
        runner.run(_recording_context(tmp_path, [])[0])


def test_evaluate_rejects_train_resume_load_mode(tmp_path: Path) -> None:
    runner = Evaluate(
        model=_QuadraticModel(),
        evaluator=_energy_evaluator(_StaticSampler(torch.zeros(1, 2, 1, dtype=torch.float64)), []),
        load={"mode": "train_resume", "path": "unused"},
    )

    with pytest.raises(ValueError, match="load.mode.*train_resume"):
        runner.run(_recording_context(tmp_path, [])[0])


def test_train_train_resume_calls_runner_owned_restore(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_restore_checkpoint_with_events(**kwargs):
        calls.append(kwargs)
        return RestoreReport(
            mode="train_resume", checkpoint_dir="ckpt", next_iteration=4, completed_updates=4
        )

    monkeypatch.setattr(
        train_runner_module,
        "restore_checkpoint_with_events",
        fake_restore_checkpoint_with_events,
    )
    runner = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"mode": "train_resume", "path": "ckpt"},
    )

    context, _ = _recording_context(tmp_path, [])
    result = runner.run(context)

    assert result.status == "completed"
    assert calls and calls[0]["model"] is runner.model
    assert calls[0]["trainer"] is runner.trainer
    assert calls[0]["sampler"] is runner.sampler
    assert calls[0]["emit"].__self__ is context


def test_train_rebuilds_update_state_after_resume_restore(monkeypatch, tmp_path: Path) -> None:
    """The updater is rebound only after the runner's restore returns."""

    calls = []

    class _RestoreAwareTrainer(_NoopTrainer):
        def resolve_update_state(self, *, model, optimizer):
            calls.append(("resolve", model, optimizer))

        def rebuild_update_state(self, *, model):
            calls.append(("rebuild", model))

        def fit(self, *, model, sampler, hamiltonian_terms, optimizer, context, emit):
            calls.append(("fit", model, optimizer))
            return super().fit(
                model=model,
                sampler=sampler,
                hamiltonian_terms=hamiltonian_terms,
                optimizer=optimizer,
                context=context,
                emit=emit,
            )

    def fake_restore_checkpoint_with_events(**kwargs):
        calls.append(("restore", kwargs["model"]))
        return RestoreReport(mode="train_resume", checkpoint_dir="ckpt")

    monkeypatch.setattr(
        train_runner_module,
        "restore_checkpoint_with_events",
        fake_restore_checkpoint_with_events,
    )
    runner = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_RestoreAwareTrainer(),
        load={"mode": "train_resume", "path": "ckpt"},
    )

    runner.run(_recording_context(tmp_path, [])[0])

    assert [entry[0] for entry in calls] == ["resolve", "restore", "rebuild", "fit"]


def test_train_rejects_legacy_optimizer_mismatch_before_resume_restore(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Runner validation precedes restore, which would otherwise mutate state."""

    model = nn.Linear(1, 1).double()
    owned_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = VMCTrainer(
        max_steps=0,
        update_method=LegacyAutogradUpdate(
            owned_optimizer,
            model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
        ),
    )
    restore_calls = []

    def fake_restore_checkpoint_with_events(**kwargs):
        restore_calls.append(kwargs)
        raise AssertionError("restore must not run before ownership validation")

    monkeypatch.setattr(
        train_runner_module,
        "restore_checkpoint_with_events",
        fake_restore_checkpoint_with_events,
    )
    runner = Train(
        model=model,
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=trainer,
        load={"mode": "train_resume", "path": "unused"},
    )

    with pytest.raises(ValueError, match="ownership"):
        runner.run(_recording_context(tmp_path, [])[0])

    assert restore_calls == []


def test_train_rejects_parameter_binding_mismatch_before_resume_restore(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The PARAMETER arm of the same clause, at the same boundary.

    ACCEPTANCE: "a conflicting late selector or wrong model/optimizer binding
    fails before sampling or mutable restore." That sentence carries TWO
    disjunctions, so it names four obligations, not two. Three were pinned:
    optimizer and parameter identity before sampling, and optimizer ownership
    before restore -- the sibling test directly above. Parameter identity
    before restore was not.

    An independent verifier measured the hole (Cannon 52043410). A mutant that
    masks only parameter identity during the runner's pre-restore resolve,
    keeping the type and optimizer checks eager and performing the real
    parameter check after restore returns, survived 43 focused tests, the full
    unit suite and 8 selected integration tests. Its external witness recorded
    ONE RESTORE AND A REAL MODEL-PARAMETER CHANGE before the correct
    `ValueError`. Restore mutates optimizer and model state, so a refusal that
    lands afterwards has already let the damage happen.

    The optimizer is deliberately the same object the runner builds, so the
    ownership guard above cannot fire and steal the kill: the factory returns
    the pre-built carrier. The foreign binding has the SAME layout as the live
    model, so the layout comparison passes and the refusal must come from
    parameter-reference identity -- `ModelParameterBinding.compare` tests
    `left is right` -- which is what "wrong model binding" means here.
    """

    model = nn.Linear(1, 1).double()
    foreign_model = nn.Linear(1, 1).double()
    owned_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = VMCTrainer(
        max_steps=0,
        update_method=LegacyAutogradUpdate(
            owned_optimizer,
            model_parameters=ModelParameterBinding(
                parameters=tuple(foreign_model.parameters())
            ),
        ),
    )
    restore_calls = []

    def fake_restore_checkpoint_with_events(**kwargs):
        restore_calls.append(kwargs)
        raise AssertionError("restore must not run before parameter-binding validation")

    monkeypatch.setattr(
        train_runner_module,
        "restore_checkpoint_with_events",
        fake_restore_checkpoint_with_events,
    )
    runner = Train(
        model=model,
        sampler=object(),
        hamiltonian_terms=[],
        # The runner builds exactly one carrier from this factory. Returning
        # the already-owned optimizer makes the ownership check pass by
        # identity, which isolates the parameter arm.
        optimizer=lambda params: owned_optimizer,
        trainer=trainer,
        load={"mode": "train_resume", "path": "unused"},
    )

    with pytest.raises(
        ValueError, match="update method parameter binding does not match live model"
    ):
        runner.run(_recording_context(tmp_path, [])[0])

    assert restore_calls == [], (
        "the parameter binding was refused only AFTER restore; the contract "
        "requires the refusal to land before any mutable restore"
    )


def test_train_rejects_a_conflicting_late_selector_before_resume_restore(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The selector arm of the clause, at the restore boundary.

    An independent verifier (Cannon 52065488) built `delay_selector_restore`,
    which defers only the selector predicate past restore. It passed F 43 and
    I 8 while its external witness recorded one restore that CHANGED REAL
    MODEL TENSORS and moved the carrier learning rate from 0.01 to 0.03125
    before the correct ValueError. Pristine restores zero times and leaves
    weights and lr untouched. Restore mutates state, so a refusal landing
    afterwards has already done the damage.

    The conflict is built WITHOUT a late argument, because the runner never
    passes one: the trainer is constructed carrying one spec and then
    explicitly pre-bound to a different method, so the runner's own resolve
    falls back to the constructor spec and conflicts with what is bound. The
    model and optimizer are deliberately the same objects the runner uses --
    the factory returns the pre-built carrier -- so the two binding guards that
    precede the selector check cannot fire first and steal the kill.
    """

    model = nn.Linear(1, 1).double()
    owned_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    binding = ModelParameterBinding(parameters=tuple(model.parameters()))
    configured = LegacyAutogradUpdate(owned_optimizer, model_parameters=binding)
    already_bound = LegacyAutogradUpdate(owned_optimizer, model_parameters=binding)
    trainer = VMCTrainer(max_steps=0, update_method=configured)
    trainer.bind_update_method(
        model=model,
        optimizer=owned_optimizer,
        update_method=already_bound,
    )
    restore_calls = []

    def fake_restore_checkpoint_with_events(**kwargs):
        restore_calls.append(kwargs)
        raise AssertionError("restore must not run before late-selector validation")

    monkeypatch.setattr(
        train_runner_module,
        "restore_checkpoint_with_events",
        fake_restore_checkpoint_with_events,
    )
    runner = Train(
        model=model,
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: owned_optimizer,
        trainer=trainer,
        load={"mode": "train_resume", "path": "unused"},
    )

    with pytest.raises(ValueError, match="already bound from a different specification"):
        runner.run(_recording_context(tmp_path, [])[0])

    assert restore_calls == [], (
        "the conflicting late selector was refused only AFTER restore; the "
        "contract requires the refusal to land before any mutable restore"
    )


class _StatelessRunnerMethod(VMCUpdateMethod[AutogradUpdateInput]):
    """A method with no owned state, for the stateless half of each pair.

    `_resolve_method_state` returns early when `update_state()` is `None`, so
    on this path neither the ownership guard nor the parameter guard runs and
    the retained model/carrier checks are the only defence. The stateful half
    of each pair below is shadowed by those method-state guards; the stateless
    half is not. Independent measurement (Cannon 52093726) showed all four
    retained-binding SINGLE deferrals crossing their boundary here.
    """

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        return VMCUpdateResult(applied=False, grad_norm=0.0, reason="declined_for_test")


def _retained_method(stateless: bool, optimizer, binding):
    if stateless:
        return _StatelessRunnerMethod()
    return LegacyAutogradUpdate(optimizer, model_parameters=binding)


def _recording_restore(monkeypatch):
    """Monkeypatch restore to record and refuse to proceed."""

    restore_calls = []

    def fake_restore_checkpoint_with_events(**kwargs):
        restore_calls.append(kwargs)
        raise AssertionError("restore must not run before binding validation")

    monkeypatch.setattr(
        train_runner_module,
        "restore_checkpoint_with_events",
        fake_restore_checkpoint_with_events,
    )
    return restore_calls


@pytest.mark.parametrize("stateless", [False, True], ids=["stateful", "stateless"])
def test_train_rejects_a_second_model_before_resume_restore(
    monkeypatch,
    tmp_path: Path,
    stateless: bool,
) -> None:
    """The model-rebind guard at the restore boundary.

    Deletion of this guard is killed by `test_a_second_model_is_refused_after_binding`,
    so the guard is tested; that test calls `resolve_update_state` directly and
    observes nothing about restore. Restore mutates model and optimizer state,
    so a refusal landing after it has already done the damage.
    """

    model = nn.Linear(1, 1).double()
    foreign_model = nn.Linear(1, 1).double()
    owned_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = VMCTrainer(
        max_steps=0,
        update_method=_retained_method(
            stateless,
            owned_optimizer,
            ModelParameterBinding(parameters=tuple(foreign_model.parameters())),
        ),
    )
    # Bind to the FOREIGN model first, so the runner's own model is the second.
    trainer.bind_update_method(model=foreign_model, optimizer=owned_optimizer)
    restore_calls = _recording_restore(monkeypatch)
    runner = Train(
        model=model,
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: owned_optimizer,
        trainer=trainer,
        load={"mode": "train_resume", "path": "unused"},
    )

    with pytest.raises(ValueError, match="already bound to a different model"):
        runner.run(_recording_context(tmp_path, [])[0])

    assert restore_calls == [], (
        "the second model was refused only AFTER restore; the contract "
        "requires the refusal to land before any mutable restore"
    )


@pytest.mark.parametrize("stateless", [False, True], ids=["stateful", "stateless"])
def test_train_rejects_a_second_carrier_before_resume_restore(
    monkeypatch,
    tmp_path: Path,
    stateless: bool,
) -> None:
    """The carrier half of the same pair, at the restore boundary."""

    model = nn.Linear(1, 1).double()
    owned_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    foreign_optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    trainer = VMCTrainer(
        max_steps=0,
        update_method=_retained_method(
            stateless,
            foreign_optimizer,
            ModelParameterBinding(parameters=tuple(model.parameters())),
        ),
    )
    # Bind to the FOREIGN carrier first; the runner then supplies its own.
    trainer.bind_update_method(model=model, optimizer=foreign_optimizer)
    restore_calls = _recording_restore(monkeypatch)
    runner = Train(
        model=model,
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: owned_optimizer,
        trainer=trainer,
        load={"mode": "train_resume", "path": "unused"},
    )

    with pytest.raises(ValueError, match="already bound to a different optimizer"):
        runner.run(_recording_context(tmp_path, [])[0])

    assert restore_calls == [], (
        "the second carrier was refused only AFTER restore; the contract "
        "requires the refusal to land before any mutable restore"
    )


def test_evaluate_model_only_calls_runner_owned_restore(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_restore_checkpoint_with_events(**kwargs):
        calls.append(kwargs)
        return RestoreReport(
            mode="model_only", checkpoint_dir="ckpt", next_iteration=4, completed_updates=4
        )

    monkeypatch.setattr(evaluate_runner_module, "restore_checkpoint_with_events", fake_restore_checkpoint_with_events)
    runner = Evaluate(
        model=_QuadraticModel(),
        evaluator=_energy_evaluator(
            _StaticSampler(torch.zeros(2, 2, 1, dtype=torch.float64)),
            {"constant": _ConstantEnergyTerm([1.0, 1.0])},
        ),
        load={"mode": "model_only", "path": "ckpt"},
    )

    context, _ = _recording_context(tmp_path, [])
    result = runner.run(context)

    assert result.status == "completed"
    assert calls and calls[0]["model"] is runner.model
    assert "sampler" not in calls[0]
    assert calls[0]["emit"].__self__ is context


def test_checkpoint_load_mode_none_does_not_call_restore(monkeypatch, tmp_path: Path) -> None:
    def fail_restore(**kwargs):
        raise AssertionError("restore_checkpoint should not be called")

    monkeypatch.setattr(train_runner_module, "restore_checkpoint_with_events", fail_restore)
    monkeypatch.setattr(evaluate_runner_module, "restore_checkpoint_with_events", fail_restore)

    train = Train(
        model=nn.Linear(1, 1).double(),
        sampler=object(),
        hamiltonian_terms=[],
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        trainer=_NoopTrainer(),
        load={"mode": "none"},
    )
    assert train.run(_recording_context(tmp_path, [])[0]).status == "completed"

    evaluate = Evaluate(
        model=_QuadraticModel(),
        evaluator=_energy_evaluator(
            _StaticSampler(torch.zeros(2, 2, 1, dtype=torch.float64)),
            {"constant": _ConstantEnergyTerm([1.0, 1.0])},
        ),
        load={"mode": "none"},
    )
    assert evaluate.run(_recording_context(tmp_path / "evaluate", [])[0]).status == "completed"


def test_evaluate_emits_lifecycle_events_through_run_context(tmp_path: Path) -> None:
    recorder = _AllOccurrenceRecorder()
    context, logger = _recording_context(tmp_path, [recorder])
    runner = Evaluate(
        model=build_tiny_spenn(),
        evaluator=_energy_evaluator(
            build_tiny_sampler(),
            [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()],
            return_terms=True,
        ),
    )

    result = runner.run(context)

    assert result.status == "completed"
    # The full typed lifecycle, in emission order. This replaces the identical
    # assertion over the legacy string sequence, which slice 2 deleted along
    # with the fourteen emit sites that produced it.
    assert recorder.labels() == [
        "EvaluationStarted",
        "Started[EvaluationTaskRun]",
        "Started[GeneratorRun]",
        "Ended[GeneratorRun]",
        "Started[CalculatorRun]",
        "Ended[CalculatorRun]",
        "Started[CalculatorRun]",
        "Ended[CalculatorRun]",
        "Started[SummaryRun]",
        "Ended[SummaryRun]",
        "Started[SummaryRun]",
        "Ended[SummaryRun]",
        "Started[SummaryRun]",
        "Ended[SummaryRun]",
        "Ended[EvaluationTaskRun]",
        "EvaluationCompleted",
    ]
    energy_records = [record.metrics for record in logger.by_namespace("eval/energy")]
    assert energy_records
    assert "local_energy_mean" in energy_records[-1]
    assert "reference_energy" not in energy_records[-1]


def test_evaluate_emits_the_typed_suite_boundaries(tmp_path: Path) -> None:
    """`EvaluationStarted`/`EvaluationCompleted` bracket the suite as POINT events.

    Deliberately not a scope: a scope's ``Ended`` fires from a ``finally``, which
    would pre-empt the run-level ``exception`` event that `EvaluationTiming`
    turns into ``eval/perf {failed: True}``. Nothing fires these on the failure
    path, which is exactly why that residual trigger has to stay.
    """

    recorder = _TypedOccurrenceRecorder()
    context, _ = _recording_context(tmp_path, [recorder])
    runner = Evaluate(
        model=build_tiny_spenn(),
        evaluator=_energy_evaluator(
            build_tiny_sampler(),
            [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()],
            return_terms=True,
        ),
    )

    result = runner.run(context)

    assert result.status == "completed"
    assert [type(event).__name__ for event in recorder.seen] == [
        "EvaluationStarted",
        "EvaluationCompleted",
    ]


def test_evaluate_logs_reference_and_term_metrics_from_task(tmp_path: Path) -> None:
    context, logger = _recording_context(tmp_path, [])
    sampler = _StaticSampler(
        torch.tensor(
            [
                [[0.0], [1.0]],
                [[2.0], [3.0]],
                [[4.0], [5.0]],
            ],
            dtype=torch.float64,
        )
    )
    runner = Evaluate(
        model=_QuadraticModel(),
        evaluator=_energy_evaluator(
            sampler,
            {
                "kinetic": _ConstantEnergyTerm([1.0, 2.0, 3.0]),
                "harmonic_trap": _ConstantEnergyTerm([4.0, 5.0, 6.0]),
            },
            return_terms=True,
            reference_energy=7.0,
        ),
    )

    result = runner.run(context)

    assert result.status == "completed"
    assert sampler.calls == 1
    metrics = logger.by_namespace("eval/energy")[0].metrics
    assert metrics["local_energy_mean"] == pytest.approx(7.0)
    assert metrics["energy_error"] == pytest.approx(0.0)
    assert metrics["energy_abs_error"] == pytest.approx(0.0)
    assert metrics["term/kinetic_mean"] == pytest.approx(2.0)
    assert metrics["term/harmonic_trap_mean"] == pytest.approx(5.0)
    assert metrics["sampler_n_walkers"] == 3


def _runner_context(cfg) -> RunContext:
    context = object.__new__(RunContext)
    context.cfg = cfg
    return context


class _AllOccurrenceRecorder(Callback):
    """Capture the whole typed evaluation lifecycle, in delivery order."""

    def __init__(self) -> None:
        super().__init__(
            typed_groups=(
                # One group: `ComponentRun` and a subclass selector in separate
                # groups would be rejected as overlapping (ADR-E002).
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(EvaluationStarted),
                        Subscription.of(EvaluationCompleted),
                        Subscription.of(ComponentFailed),
                        started(EvaluationTaskRun),
                        ended(EvaluationTaskRun),
                        started(ComponentRun),
                        ended(ComponentRun),
                    ),
                ),
            )
        )
        self.seen: list[object] = []

    def handle_occurrence_impl(self, occurrence, context) -> None:
        self.seen.append(occurrence.event)

    def labels(self) -> list[str]:
        return [
            f"{type(event).__name__}[{type(event.operation).__name__}]"
            if isinstance(event, (Started, Ended))
            else type(event).__name__
            for event in self.seen
        ]


class _TypedOccurrenceRecorder(Callback):
    """Capture only the evaluation domain's suite-level typed point events."""

    def __init__(self) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(EvaluationStarted),
                        Subscription.of(EvaluationCompleted),
                    ),
                ),
            )
        )
        self.seen: list[object] = []

    def handle_occurrence_impl(self, occurrence, context) -> None:
        self.seen.append(occurrence.event)


def _recording_context(tmp_path: Path, callbacks) -> tuple[RunContext, RecordingLogger]:
    """Return a real `RunContext` and the logger capturing its metric records.

    This replaced a `RunContext` subclass that skipped ``super().__init__`` and
    worked only because `Runner.emit` falls back when ``artifact_manager`` is
    missing. `Evaluate` now emits typed occurrences through the context itself,
    which needs the occurrence counters, the artifact manager, and the run
    clock, so the genuine article is the only usable option. Metric records are
    read off a logger rather than an overridden ``log``, which also keeps
    `RunContext.log` on its production path.
    """

    logger = RecordingLogger()
    return make_run_context(tmp_path, callbacks=list(callbacks), loggers=[logger]), logger


class _QuadraticModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        flat = batch.flatten_samples()
        logabs = -flat.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _StaticSampler:
    def __init__(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.calls = 0

    def collect_samples(self, model, *, device: str | torch.device | None = None):
        self.calls += 1
        positions = self.positions.to(device=device)
        stats = SamplerStats(
            acceptance_rate=1.0,
            n_walkers=positions.shape[0],
            burn_in=0,
            n_steps=1,
            proposal_scale=0.1,
        )
        return Walkers(positions=positions), stats


class _NoopTrainer:
    # `Train` requires the durable resume cursor rather than guessing it from
    # the final state, so even a no-op trainer has to report one. Completing
    # step 0 leaves the cursor at 1.
    next_iteration = 1

    def fit(self, *, model, sampler, hamiltonian_terms, optimizer, context, emit):
        return TrainerState(step=0, model=model, optimizer=optimizer, trainer=self, sampler=sampler)


class _ConstantEnergyTerm:
    name = "constant"

    def __init__(self, values) -> None:
        self.values = torch.as_tensor(values, dtype=torch.float64)

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        values = self.values.to(device=batch.device, dtype=batch.dtype)
        return LocalEnergyResult(total=values, terms={"internal": values})


def _energy_evaluator(
    sampler,
    terms,
    *,
    return_terms: bool = False,
    reference_energy: float | None = None,
) -> Evaluator:
    summaries = [LocalEnergySummary(), SamplerStatsSummary()]
    # HamiltonianTermSummary requires per-term energies; only include it when the
    # calculator actually produces them, else it (correctly) fails the task.
    if return_terms:
        summaries.insert(1, HamiltonianTermSummary())
    if reference_energy is not None:
        summaries.append(ReferenceEnergySummary(reference_energy=reference_energy))
    return Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir=Path("/tmp/rhu/spenn_eval_tests/energy"),
                generator=MCMCGenerator(sampler=sampler),
                calculators=[
                    WavefunctionCalculator(),
                    LocalEnergyCalculator(hamiltonian_terms=terms, return_terms=return_terms),
                ],
                summaries=summaries,
            )
        ],
    )


def _metrics(run_root: Path, namespace: str) -> dict:
    jsonl_files = list(run_root.glob("**/metrics.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one metrics.jsonl, found {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    matches = [record["metrics"] for record in records if record.get("namespace") == namespace]
    assert matches, f"no records for namespace {namespace}"
    return matches[-1]


@pytest.mark.parametrize(
    ("fixture", "exact_energy"),
    [("exact_singlet_eval.yaml", 2.0), ("exact_triplet_eval.yaml", 1.25)],
)
def test_hooke_eval_runner_matches_exact_energy(tmp_path, fixture: str, exact_energy: float) -> None:
    config_path = FIXTURES / fixture
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    exit_code = run_from_config(cfg, config_path=str(config_path), command="pytest")
    assert exit_code == 0

    metrics = _metrics(tmp_path, "eval/energy")
    energy_atol = float(cfg.validation.energy_atol)
    variance_max = float(cfg.validation.variance_max)

    assert metrics["reference_energy"] == pytest.approx(exact_energy)
    assert abs(metrics["energy_error"]) < energy_atol
    assert metrics["energy_abs_error"] < energy_atol
    assert metrics["local_energy_n_finite"] == metrics["local_energy_n_total"] == 512
    assert metrics["local_energy_finite_fraction"] == 1.0
    assert metrics["local_energy_nonfinite_count"] == 0
    assert abs(metrics["local_energy_mean"] - exact_energy) < energy_atol
    assert metrics["local_energy_variance"] < variance_max
    for term in ("kinetic", "harmonic_trap", "electron_electron"):
        assert f"term/{term}_mean" in metrics
        assert f"term/{term}_variance" in metrics
    assert "virial_residual" in metrics
    assert "virial_relative_residual" in metrics
    assert math.isfinite(metrics["virial_residual"])
    assert math.isfinite(metrics["virial_relative_residual"])
    # This sampled estimator has finite-walker noise, and neither this sample
    # nor the fixed-point unit fixture is an exact |psi|^2 expectation. A
    # restricted variational ansatz need not have a vanishing residual because
    # the identity requires dilation stationarity.
    assert metrics["sampler_n_walkers"] == 512
    assert "acceptance_rate" in {key.removeprefix("sampler_") for key in metrics}
    assert "wall_time_sec" in _metrics(tmp_path, "eval/perf")
    assert "time_sec" in _metrics(tmp_path, "diagnostics/energy")
    assert any("wall_time_sec" in record for record in _namespace_records(tmp_path, "runtime"))


def _namespace_records(run_root: Path, namespace: str) -> list[dict]:
    jsonl_files = list(run_root.glob("**/metrics.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one metrics.jsonl, found {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    return [record["metrics"] for record in records if record.get("namespace") == namespace]


def _only_run_dir_with_status(run_root: Path, run_name: str) -> Path:
    status_files = list((run_root / run_name).glob("**/status.json"))
    assert len(status_files) == 1, f"expected one status.json under {run_name}, found {status_files}"
    return status_files[0].parent


@pytest.mark.parametrize("fixture", ["exact_singlet_eval.yaml", "exact_triplet_eval.yaml"])
def test_hooke_eval_runner_writes_standard_artifacts(tmp_path, fixture: str) -> None:
    config_path = FIXTURES / fixture
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    assert run_from_config(cfg, config_path=str(config_path), command="pytest") == 0

    run_dirs = list(tmp_path.glob("hooke_exact/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    run_dir = run_dirs[0]
    for artifact in (
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.jsonl",
        "metrics.csv",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"


def test_hooke_exact_evaluation_stack_runs_from_yaml_fixture(tmp_path) -> None:
    """The full Generator->Calculator->Summary stack runs on exact Hooke from YAML.

    This is a correctness test for the evaluation implementation: the same
    deterministic task stack used for learned-model validation is exercised on
    the analytic Hooke singlet, where every diagnostic has a known answer.
    """

    config_path = FIXTURES / "hooke_exact_evaluation.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    exit_code = run_from_config(cfg, config_path=str(config_path), command="pytest")
    assert exit_code == 0

    # Exact local energy E_L = 2.0 everywhere finite, for every geometry task.
    # The exact singlet is nodeless (sign=+1) so every finite configuration has
    # a finite local energy. The variance tolerance is looser for the cusp task:
    # at r12=1e-5 the autograd Laplacian sums 1/r12 terms that cancel to give
    # E_L=2.0, and float64 catastrophic cancellation leaves a small residual.
    variance_tol = {
        "cusp": 1.0e-6,
        "tail": 1.0e-8,
        "stratified_geometry": 1.0e-8,
        "hooke_orbital": 1.0e-8,
        "energy": 1.0e-8,
    }
    tasks = ("cusp", "tail", "stratified_geometry", "hooke_orbital", "energy")
    for task in tasks:
        metrics = _metrics(tmp_path, f"hooke_exact/{task}")
        assert metrics["local_energy_finite_fraction"] == 1.0, task
        assert metrics["local_energy_nonfinite_count"] == 0, task
        assert metrics["local_energy_mean"] == pytest.approx(2.0, abs=1.0e-3), task
        assert metrics["local_energy_variance"] < variance_tol[task], task
        assert metrics["local_energy_q01"] == pytest.approx(2.0, abs=1.0e-3), task
        assert metrics["local_energy_q99"] == pytest.approx(2.0, abs=1.0e-3), task
        status_metrics = _metrics(tmp_path, f"hooke_exact/{task}/status")
        assert status_metrics["task_success"] is True, task
        assert status_metrics["task_failed"] is False, task

    # Opposite-spin cusp: even slope -> 1/2; near-coalescence C_{-1} -> 0.
    cusp = _metrics(tmp_path, "hooke_exact/cusp")
    assert cusp["cusp_even_slope_mean"] == pytest.approx(0.5, abs=1.0e-3)
    assert cusp["cusp_even_slope_abs_error"] < 1.0e-3
    assert cusp["cusp_odd_slant_mean_abs"] < 1.0e-8
    assert cusp["cusp_odd_slant_max_abs"] < 1.0e-8
    assert cusp["c_minus_1_abs_max"] < 1.0e-3

    tail = _metrics(tmp_path, "hooke_exact/tail")
    assert tail["stability_outlier_count"] == 0
    assert tail["stability_abs_threshold"] == pytest.approx(10.0)
    assert tail["nonfinite_logabs_count"] == 0

    for task in ("stratified_geometry", "hooke_orbital"):
        metrics = _metrics(tmp_path, f"hooke_exact/{task}")
        assert metrics["nonfinite_local_energy_count"] == 0, task
        assert metrics["large_abs_local_energy_count"] == 0, task
        assert metrics["nonfinite_logabs_count"] == 0, task
        assert metrics["local_energy_pathology_count"] == 0, task

    # Reference energy comparison: included by this exact-Hooke correctness
    # config. There is no phase gate; ordinary validation configs omit it.
    energy = _metrics(tmp_path, "hooke_exact/energy")
    assert energy["reference_energy"] == pytest.approx(2.0)
    assert energy["energy_abs_error"] < 1.0e-4

    suite_status = _metrics(tmp_path, "hooke_exact/status")
    assert suite_status["suite_success"] is True
    assert suite_status["suite_failed"] is False

    run_dir = _only_run_dir_with_status(tmp_path, "hooke_exact_stack")
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    failures = run_dir / "diagnostics" / "failures.jsonl"
    assert not failures.exists() or failures.read_text().strip() == ""

    index = json.loads((run_dir / "diagnostics" / "index.json").read_text())
    indexed = {task["name"]: task for task in index["tasks"]}
    assert set(indexed) == set(tasks)
    assert all(task["status"] == "success" for task in indexed.values())

    # Each task writes under its resolved task output directory and the index
    # records that same task-local location.
    for task in tasks:
        task_dir = run_dir / task
        assert (run_dir / task).is_dir(), f"missing task output dir: {task}"
        assert (task_dir / "sampled_eval_table.csv").is_file(), task
        assert Path(indexed[task]["output_dir"]) == task_dir
        assert any(Path(artifact["path"]) == task_dir / "sampled_eval_table.csv" for artifact in indexed[task]["artifacts"])
