"""T2 review probes for the three newly admitted checkpoint callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tpen.callback import FactorScalars, OptimizerState, SamplerHealth
from tpen.checkpoint.profiles import (
    CADENCE_SENSITIVE_CALLBACK_TARGETS,
    OBSERVATIONAL_CALLBACK_TARGETS,
    RESUME_SENSITIVE_CALLBACK_TARGETS,
)
from tpen.nn import ElectronElectronCusp
from tpen.events import Started
from tpen.training.events import TrainingIteration
from tpen.training.state import TrainerState
from tpen.training.trainer import VMCTrainer
from tpen.training.update import ModelParameterBinding
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    make_sampler_stats,
    training_state,
)
from tests.unit.callback.test_optimizer_state_review_r1 import (
    _OneElectronModel,
    _UnsupportedAdamCarrierUpdate,
    _DispatchingContext,
)


OPTIMIZER_STATE = "tpen.callback.OptimizerState"
SAMPLER_HEALTH = "tpen.callback.SamplerHealth"
FACTOR_SCALARS = "tpen.callback.FactorScalars"


@pytest.mark.parametrize(
    ("target", "observational", "resume_sensitive", "cadence_sensitive"),
    [
        (OPTIMIZER_STATE, True, False, False),
        (SAMPLER_HEALTH, False, True, True),
        (FACTOR_SCALARS, False, True, True),
    ],
)
def test_pr647_targets_have_exact_profile_classifications(
    target: str,
    observational: bool,
    resume_sensitive: bool,
    cadence_sensitive: bool,
) -> None:
    """Pin exact membership and exclusion across all callback profile sets."""

    assert (target in OBSERVATIONAL_CALLBACK_TARGETS) is observational
    assert (target in RESUME_SENSITIVE_CALLBACK_TARGETS) is resume_sensitive
    assert (target in CADENCE_SENSITIVE_CALLBACK_TARGETS) is cadence_sensitive


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_calls", 0), ("probability", 0.0)],
)
def test_optimizer_state_rejects_unsupported_method_at_iteration_entry_when_logging_is_disabled(
    field: str, value: object
) -> None:
    """Admission validation is unconditional, before sampling or completed logging."""

    model = _OneElectronModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    method = _UnsupportedAdamCarrierUpdate(
        optimizer, ModelParameterBinding.from_parameters(tuple(model.parameters()))
    )
    callback = OptimizerState({"all": ["linear"]}, **{field: value})
    context = _DispatchingContext()
    context.callbacks = [callback]
    trainer = VMCTrainer(max_steps=1, update_method=method)
    trainer.bind_update_method(model=model, optimizer=optimizer)
    state = TrainerState(
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        update_state=method.update_state(),
    )

    with pytest.raises(ValueError, match="LegacyAutogradUpdate"):
        with context.scope(TrainingIteration(step=0), state=state):
            pass
    assert not optimizer.state


@dataclass
class _FactorModel:
    factors: list[Any]


def _factor_model(*, invalid: bool) -> _FactorModel:
    factor = ElectronElectronCusp(trainable_range=True)
    if invalid:
        with torch.no_grad():
            factor.raw_same_range.fill_(float("nan"))
    return _FactorModel([factor])


@pytest.mark.parametrize("kind", ["sampler", "factors"])
def test_resume_rebuild_keeps_durable_cadence_and_fail_fast_at_selected_step(kind: str) -> None:
    """A rebuilt callback restarts occurrence count, not the durable-step schedule."""

    def build() -> Any:
        if kind == "sampler":
            return SamplerHealth(
                every_n_steps=2,
                start_step=0,
                fail_fast=True,
                max_acceptance_rate=0.9,
            )
        return FactorScalars(every_n_steps=2, start_step=0, fail_fast=True)

    def state(step: int, *, invalid: bool) -> TrainerState:
        return training_state(
            model=_factor_model(invalid=invalid),
            sampler_stats=make_sampler_stats(acceptance_rate=0.95 if invalid else 0.5),
        )

    uninterrupted = build()
    uninterrupted_context = RecordingContext()
    deliver_completed_iteration(uninterrupted, uninterrupted_context, state(0, invalid=False), step=0, count=1)
    deliver_completed_iteration(uninterrupted, uninterrupted_context, state(1, invalid=False), step=1, count=2)
    with pytest.raises((RuntimeError, ValueError)):
        deliver_completed_iteration(uninterrupted, uninterrupted_context, state(2, invalid=True), step=2, count=3)

    before_restore = build()
    resumed_context = RecordingContext()
    deliver_completed_iteration(before_restore, resumed_context, state(0, invalid=False), step=0, count=1)
    deliver_completed_iteration(before_restore, resumed_context, state(1, invalid=False), step=1, count=2)
    rebuilt = build()
    with pytest.raises((RuntimeError, ValueError)):
        deliver_completed_iteration(rebuilt, resumed_context, state(2, invalid=True), step=2, count=1)

    namespace = "checks/sampler" if kind == "sampler" else "train/factors"
    assert [record["step"] for record in uninterrupted_context.by_namespace(namespace)] == [0, 2]
    assert [record["step"] for record in resumed_context.by_namespace(namespace)] == [0, 2]
