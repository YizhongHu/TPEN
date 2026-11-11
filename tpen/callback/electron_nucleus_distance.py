"""Training-batch electron--nucleus distance diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import torch

from tpen.artifacts import RunContext
from tpen.data.batch import ElectronBatch, electron_nuclear_distances
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class ElectronNucleusDistanceQuantiles(StatefulCallback[TrainerState]):
    """Log per-electron nearest-nucleus distances from each training batch.

    The callback is optional: it emits no metrics unless a run configuration
    instantiates it.  At a completed iteration it summarizes the exact
    ``TrainerState.batch`` that supplied that iteration's model update.  For
    each electron, distances are first minimized over nuclei and then
    summarized across all configurations in the batch.

    This is an observational callback. Its ``every_n_steps`` and ``start_step``
    cadence is based on the durable training step and therefore reconstructs
    the same scheduled steps after resume. Callback-local ``max_calls`` and
    ``probability`` state restarts on resume, changing only which observational
    steps are logged, as with ``TrainStepTiming``.

    Parameters
    ----------
    quantiles : sequence of float
        Quantile levels to report.  Levels are supplied by the run
        configuration; core provides no defaults.  Quantiles use PyTorch's
        ``linear`` interpolation method, matching the naming convention used
        by :mod:`tpen.diagnostics.energy`.
    every_n_steps : int, optional
        Training-step cadence.
    start_step : int, optional
        First durable training step at which to emit a metric.
    max_calls, probability, seed
        Optional callback-local cadence controls. On resume, their counters
        and random stream restart, changing only which observational steps are
        logged, as with :class:`tpen.callback.TrainStepTiming`.

    Raises
    ------
    ValueError
        If ``quantiles`` is missing, empty, invalid, or contains duplicates.
        A batch without nuclear positions, or with zero nuclei, is also
        refused explicitly.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, *, quantiles: Sequence[float] | None = None, **kwargs: Any) -> None:
        if quantiles is None:
            raise ValueError("quantiles must be provided by the run configuration")
        self.quantiles = _validate_quantiles(quantiles)
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(TrainingIterationCompleted),)),
            ),
            **kwargs,
        )
        self._steps = StepCadenceGate(cadence)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Summarize the batch consumed by the completed training step."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return
        batch = state.batch
        if not isinstance(batch, ElectronBatch):
            raise ValueError(
                "ElectronNucleusDistanceQuantiles requires the completed "
                "iteration's ElectronBatch on TrainerState.batch"
            )
        context.log(
            summarize_electron_nucleus_distances(batch, quantiles=self.quantiles),
            step=step,
            namespace="train/electron_nucleus_distance",
        )


def summarize_electron_nucleus_distances(
    batch: ElectronBatch,
    *,
    quantiles: Sequence[float],
) -> dict[str, float]:
    """Return per-electron nearest-nucleus distance summaries.

    Parameters
    ----------
    batch : ElectronBatch
        The training batch to summarize.
    quantiles : sequence of float
        Configured levels in ``[0, 1]``.  PyTorch's linear interpolation is
        used for every level.

    Returns
    -------
    dict
        Keys are ``electron_<index>_nucleus_distance_min`` and
        ``electron_<index>_nucleus_distance_<q-label>``.
    """

    levels = _validate_quantiles(quantiles)
    nuclear_positions = batch.nuclear_positions
    if nuclear_positions is None:
        configuration = batch.atomic_configuration
        nuclear_positions = None if configuration is None else configuration.positions
    if nuclear_positions is None or int(nuclear_positions.shape[-2]) == 0:
        raise ValueError("electron-nucleus distance diagnostics require at least one nucleus")
    if nuclear_positions.ndim > 2:
        nuclear_positions = batch.flatten_samples().nuclear_positions

    distances = electron_nuclear_distances(batch, nuclear_positions=nuclear_positions)
    nearest = distances.min(dim=-1).values
    q_tensor = nearest.new_tensor(levels)
    metrics: dict[str, float] = {}
    # Quantile is taken over configurations, preserving the electron axis.
    q_values = _quantile_by_electron(nearest, q_tensor)
    minima = nearest.min(dim=0).values
    for electron, minimum in enumerate(minima):
        prefix = f"electron_{electron}_nucleus_distance"
        metrics[f"{prefix}_min"] = float(minimum.item())
        for level, value in zip(levels, q_values[electron], strict=True):
            metrics[f"{prefix}_{_quantile_label(level)}"] = float(value.item())
    return metrics


def _quantile_by_electron(values: Any, levels: Any) -> Any:
    """Compute configured linear quantiles across the flattened batch axis."""

    flattened = values.reshape(-1, values.shape[-1])
    return torch.quantile(flattened, levels, dim=0, interpolation="linear").transpose(0, 1)


def _validate_quantiles(values: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize configured quantile levels."""

    levels = tuple(float(value) for value in values)
    if not levels:
        raise ValueError("quantiles must contain at least one level")
    if any(not 0.0 <= level <= 1.0 for level in levels):
        raise ValueError(f"quantiles must lie in [0, 1], got {levels}")
    if len(set(levels)) != len(levels):
        raise ValueError(f"quantiles must not contain duplicates, got {levels}")
    labels = tuple(_quantile_label(level) for level in levels)
    if len(set(labels)) != len(labels):
        raise ValueError(
            "quantiles must have unique serialized labels, "
            f"got {dict(zip(levels, labels, strict=True))}"
        )
    return levels


def _quantile_label(value: float) -> str:
    """Use the established energy-diagnostic quantile labels."""

    known = {
        0.001: "q001",
        0.01: "q01",
        0.05: "q05",
        0.5: "q50",
        0.95: "q95",
        0.99: "q99",
        0.999: "q999",
    }
    for key, label in known.items():
        if math.isclose(value, key, rel_tol=0.0, abs_tol=1.0e-12):
            return label
    return "q" + f"{100.0 * value:g}".replace(".", "p")


__all__ = ["ElectronNucleusDistanceQuantiles", "summarize_electron_nucleus_distances"]
