"""Test-only training-batch logabs/sign capture callback for the amplitude spike.

This is explicitly a spike harness, not a production core feature: it is not
registered under ``tpen.callback`` and no study config references it by
dotted path. It follows the StatefulCallback typed-event/cadence pattern
documented by ``tpen.callback.factor_scalars.FactorScalars`` and
``tpen.callback.checkpoint.Checkpoint``, and reads ``state.wavefunction_output``
from the JUST-COMPLETED iteration, before the next forward -- the same
pre-update, no-extra-forward observation point the amplitude-risk design
specifies (note amplitude-risk-design-20261106, "Training cadence").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from tpen.artifacts import RunContext
from tpen.callback.base import StatefulCallback
from tpen.callback.cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingCompleted, TrainingIterationCompleted
from tpen.training.state import TrainerState


class TrainingBatchLogabsCapture(StatefulCallback[TrainerState]):
    """Buffer every completed iteration's pre-update logabs/sign, dump at the end.

    Not a core callback. Construct directly in a spike driver script and pass
    it in a run's ``callbacks`` list; do not add it to a checked-in study
    config under ``experiments/``.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, *, output_path: str | Path, every_n_steps: int = 1, **kwargs: Any) -> None:
        kwargs["every_n_steps"] = every_n_steps
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(TrainingIterationCompleted),
                        Subscription.of(TrainingCompleted),
                    )
                ),
            ),
            **kwargs,
        )
        self._steps = StepCadenceGate(cadence)
        self.output_path = Path(output_path)
        self._steps_buf: list[int] = []
        self._optimizer_step_buf: list[bool] = []
        self._logabs_rows: list[np.ndarray] = []
        self._sign_rows: list[np.ndarray] = []
        self._n_walkers: int | None = None
        self._flushed = False

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        event = occurrence.event
        if isinstance(event, TrainingIterationCompleted):
            step = int(event.iteration.step)
            if not self._steps.should_run(step):
                return
            output = state.wavefunction_output
            logabs = output.logabs.detach().to("cpu", dtype=torch.float64).reshape(-1).numpy()
            sign = output.sign.detach().to("cpu", dtype=torch.float64).reshape(-1).numpy()
            if self._n_walkers is None:
                self._n_walkers = int(logabs.size)
            elif int(logabs.size) != self._n_walkers:
                raise ValueError(
                    f"training-batch capture saw a changing batch size: "
                    f"{self._n_walkers} then {logabs.size} at step {step}"
                )
            self._steps_buf.append(step)
            self._optimizer_step_buf.append(bool(state.optimizer_step))
            self._logabs_rows.append(logabs.copy())
            self._sign_rows.append(sign.copy())
            return
        if isinstance(event, TrainingCompleted):
            self._flush()

    def _flush(self) -> None:
        if self._flushed:
            return
        self._flushed = True
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._steps_buf:
            np.savez(
                self.output_path,
                step=np.empty((0,), dtype=np.int64),
                optimizer_step=np.empty((0,), dtype=bool),
                logabs=np.empty((0, 0), dtype=np.float64),
                sign=np.empty((0, 0), dtype=np.float64),
            )
            return
        np.savez(
            self.output_path,
            step=np.asarray(self._steps_buf, dtype=np.int64),
            optimizer_step=np.asarray(self._optimizer_step_buf, dtype=bool),
            logabs=np.stack(self._logabs_rows, axis=0),
            sign=np.stack(self._sign_rows, axis=0),
        )


__all__ = ["TrainingBatchLogabsCapture"]
