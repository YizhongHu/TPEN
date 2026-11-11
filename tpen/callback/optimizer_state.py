"""Optional optimizer-state summaries grouped by model modules."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from tpen.artifacts import RunContext
from tpen.dependencies import require_torch
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Started, Subscription, started
from tpen.parameter_groups import resolve_parameter_groups
from tpen.training.events import TrainingIteration, TrainingIterationCompleted
from tpen.training.state import TrainerState
from tpen.training.update import LegacyAutogradUpdate

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class OptimizerState(StatefulCallback[TrainerState]):
    """Log Adam-style optimizer state for configured parameter groups.

    Parameters
    ----------
    groups : mapping[str, iterable[str]]
        Group names mapped to ``named_modules()`` paths or explicit
        ``trace_name`` values. Unselected parameters are reported under the
        ``ungrouped`` bucket.
    every_n_steps : int, optional
        Durable-step cadence consumed by ``pop_step_cadence``.

    Notes
    -----
    This is an observational callback. It supports only the nominal
    :class:`LegacyAutogradUpdate` method with an Adam or AdamW carrier and
    refuses other update methods rather than inferring equivalent semantics
    from a carrier optimizer. Missing state is expected before the first
    Adam step and is counted per parameter.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, groups: Mapping[str, Any], **kwargs: Any) -> None:
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            typed_groups=(
                # Validate at iteration entry, before sampling, independently
                # of the observational completed-step cadence.
                SubscriptionGroup(selectors=(started(TrainingIteration),)),
                SubscriptionGroup(selectors=(Subscription.of(TrainingIterationCompleted),)),
            ),
            **kwargs,
        )
        self.groups = dict(groups)
        self._steps = StepCadenceGate(cadence)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Measure optimizer state at the completed-update boundary."""

        event = occurrence.event
        if isinstance(event, Started) and isinstance(event.operation, TrainingIteration):
            _validate_update_method(state)
            return
        if not isinstance(event, TrainingIterationCompleted):
            return
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return

        torch = require_torch(feature="OptimizerState callback")
        update_state = state.update_state
        if update_state is None:
            raise ValueError("OptimizerState requires TrainerState.update_state")
        _validate_update_method(state)
        optimizer = update_state.optimizer
        model_parameters = tuple(update_state.model_parameters.parameters)
        live_ids = {id(parameter) for parameter in model_parameters}
        foreign = [parameter for parameter in optimizer.state if id(parameter) not in live_ids]
        if foreign:
            raise ValueError(
                "OptimizerState refuses optimizer state for parameter(s) outside the live model"
            )
        groups = resolve_parameter_groups(state.model, self.groups)
        for group, parameters in groups.items():
            context.log(
                _optimizer_metrics(torch, optimizer, parameters),
                step=step,
                namespace=f"train/optimizer_state/{group}",
            )


def _validate_update_method(state: TrainerState) -> None:
    """Require the explicitly supported update-method authority."""

    trainer = state.trainer
    method = getattr(trainer, "_resolved_update_method", None) if trainer is not None else None
    if method is not None and not isinstance(method, LegacyAutogradUpdate):
        raise ValueError(
            "OptimizerState supports only LegacyAutogradUpdate with Adam-style state; "
            f"got {type(method).__name__}"
        )
    optimizer = state.update_state.optimizer if state.update_state is not None else None
    if optimizer is None or type(optimizer).__name__ not in {"Adam", "AdamW"}:
        name = type(optimizer).__name__ if optimizer is not None else "None"
        raise ValueError(
            "OptimizerState supports only Adam or AdamW optimizer state; got " + name
        )


def _optimizer_metrics(
    torch: Any,
    optimizer: Any,
    parameters: tuple[tuple[str, Any], ...],
) -> dict[str, Any]:
    """Compute JSON/CSV-safe summaries for one group."""

    states = [optimizer.state.get(parameter) for _, parameter in parameters]
    present = [state for state in states if state is not None]
    steps = [_state_step(state) for state in present]
    exp_avg = [_state_tensor(state, "exp_avg") for state in present]
    exp_avg_sq = [_state_tensor(state, "exp_avg_sq") for state in present]
    step_min = min(steps) if steps else 0
    step_max = max(steps) if steps else 0
    return {
        "n_parameters": len(parameters),
        "n_parameter_elements": sum(int(parameter.numel()) for _, parameter in parameters),
        "n_no_optimizer_state": len(parameters) - len(present),
        "step": step_min if step_min == step_max else None,
        "step_min": step_min,
        "step_max": step_max,
        "step_mismatch": step_min != step_max,
        **_tensor_metrics(torch, "exp_avg", exp_avg),
        **_tensor_metrics(torch, "exp_avg_sq", exp_avg_sq),
    }


def _state_step(state: Mapping[str, Any]) -> int:
    """Read one Adam step scalar without averaging it."""

    value = state.get("step")
    if value is None or getattr(value, "numel", lambda: 0)() != 1:
        raise ValueError("OptimizerState requires a scalar Adam state['step']")
    return int(value.item())


def _state_tensor(state: Mapping[str, Any], key: str) -> Any:
    """Read one required Adam moment tensor."""

    value = state.get(key)
    if value is None or not hasattr(value, "reshape"):
        raise ValueError(f"OptimizerState requires Adam state[{key!r}]")
    return value


def _tensor_metrics(torch: Any, name: str, values: list[Any]) -> dict[str, Any]:
    """Summarize one optimizer moment while preserving invalidity evidence."""

    flat = torch.cat([value.detach().reshape(-1) for value in values]) if values else torch.empty(0)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    prefix = f"{name}_"
    return {
        f"{prefix}l2": float(finite_values.norm().item()) if finite_values.numel() else None,
        f"{prefix}rms": float(finite_values.square().mean().sqrt().item()) if finite_values.numel() else None,
        f"{prefix}max_abs": float(finite_values.abs().max().item()) if finite_values.numel() else None,
        f"{prefix}nonfinite_count": int((~finite).sum().item()),
    }


__all__ = ["OptimizerState"]
