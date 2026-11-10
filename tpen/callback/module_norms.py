"""Optional module-grouped parameter and gradient norm logging."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from tpen.artifacts import RunContext
from tpen.dependencies import require_torch
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.parameter_groups import resolve_parameter_groups
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class ModuleNorms(StatefulCallback[TrainerState]):
    """Log norms for configured module groups after each completed update.

    Parameters
    ----------
    groups : mapping[str, iterable[str]]
        Group names mapped to ``named_modules()`` paths or explicit
        ``trace_name`` values. Parameters not selected by a group are reported
        under the explicit ``ungrouped`` bucket.
    every_n_steps : int, optional
        Durable-step cadence consumed by ``pop_step_cadence``. The
        ``every_n_steps`` and ``start_step`` window reconstructs exactly;
        callback-local ``max_calls`` and probabilistic admission restart on
        resume and affect only which observational metrics are logged.

    Notes
    -----
    Parameters are read after the optimizer update, while gradients are the
    clipped gradients consumed by that update. Missing gradients are counted
    separately and are never folded into a zero gradient. Each group is logged
    under ``train/module_norms/<group>`` so a configured callback remains
    entirely absent from runs that do not list it.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, groups: Mapping[str, Any], **kwargs: Any) -> None:
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            typed_groups=(
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
        """Measure configured groups at the completed-iteration boundary."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return
        torch = require_torch(feature="ModuleNorms callback")
        groups = resolve_parameter_groups(state.model, self.groups)
        for group, parameters in groups.items():
            context.log(_norm_metrics(torch, parameters), step=step, namespace=f"train/module_norms/{group}")


def _norm_metrics(torch: Any, parameters: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Compute independent parameter and gradient summaries for one group."""

    values = [parameter.detach().reshape(-1) for _, parameter in parameters]
    gradients = [parameter.grad.detach().reshape(-1) for _, parameter in parameters if parameter.grad is not None]
    n_elements = sum(int(value.numel()) for value in values)
    finite_values = torch.cat(values) if values else torch.empty(0)
    finite_gradients = torch.cat(gradients) if gradients else torch.empty(0)
    finite_gradients = finite_gradients[torch.isfinite(finite_gradients)]
    n_no_gradient = len(parameters) - len(gradients)
    return {
        "n_parameters": len(parameters),
        "n_parameter_elements": n_elements,
        "n_no_gradient_parameters": n_no_gradient,
        "parameter_l2": float(finite_values.norm().item()) if finite_values.numel() else 0.0,
        "parameter_rms": float(finite_values.square().mean().sqrt().item()) if finite_values.numel() else 0.0,
        "parameter_max_abs": float(finite_values.abs().max().item()) if finite_values.numel() else 0.0,
        "gradient_l2": float(finite_gradients.norm().item()) if finite_gradients.numel() else 0.0,
        "gradient_rms": float(finite_gradients.square().mean().sqrt().item()) if finite_gradients.numel() else 0.0,
        "gradient_max_abs": float(finite_gradients.abs().max().item()) if finite_gradients.numel() else 0.0,
        "gradient_nonfinite_count": sum(
            int((~torch.isfinite(gradient)).sum().item()) for gradient in gradients
        ),
    }


__all__ = ["ModuleNorms"]
