"""R1: a failed health check must leave no checkpoint for the step it failed.

Regression record for item ``195c3ff3``, measured on Cannon as jobs ``38267440``
and ``38267443``. PR #175 moved the four health checks onto
`tpen.training.events.TrainingIterationCompleted` while `tpen.callback.Checkpoint`
still wrote at the earlier legacy ``step_end``, so the two swapped order: a
``fail_fast`` abort wrote a complete checkpoint AND advanced ``latest.json`` to
it, leaving the model that failed its own health check as the default resume
target of any later resume.

The repair puts the periodic checkpoint write on the SAME boundary the checks
fire on. That works because dispatch within one occurrence is the callback-list
loop (`tpen.artifacts.RunContext._dispatch_occurrence`) and an exception
propagates straight out of it, so a check listed before `Checkpoint` raises
first and `Checkpoint` never runs.

Nothing in the code declares that ordering, and that is exactly the objection
these tests answer. They pin the PROPERTY -- a failed ``fail_fast`` check leaves
no checkpoint for its step -- and not the mechanism. They fail if someone
reorders the production callback list, changes which typed event `Checkpoint`
subscribes to (an `UpdateCompleted` write lands BEFORE any check can run), or
wraps the dispatch loop in a per-callback ``try/except``.

Two choices here are load-bearing:

*The callback order is READ FROM THE PRODUCTION CONFIG*, not written out below,
so an edit to that file is what the assertion actually sees. Their configured
options are read from it too, so this exercises the shipped callbacks rather
than test-shaped lookalikes.

*The lever must be `DataIntegrity`.* It is the only check that can collide with
a checkpoint write: it runs at ``every_n_steps: 1``, while `GradientStats` and
`RuntimeEquivariance` fire at steps 0, 5, 10 and periodic checkpoints land on
completed-update multiples of five, i.e. loop steps 4, 9, 14. A probe built on
either of those produces identical listings in both arms and reads as a false
refutation.

The scenario reproduces the hardware probe exactly: the failure lands on loop
step 4, whose completed-update count of 5 is a cadence hit, so the checkpoint
under test is the same ``step_000005`` both arms of that probe compared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from tpen.callback import Checkpoint, DataIntegrity
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.training.events import (
    TrainingIteration,
    TrainingIterationCompleted,
    UpdateCompleted,
)
from tpen.training.state import TrainerState
from tests.helpers.run_context import make_run_context

PRODUCTION_TRAIN_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "hooke"
    / "tpen-pair-v1"
    / "configs"
    / "train.yaml"
)

_DATA_INTEGRITY = "tpen.callback.DataIntegrity"
_CHECKPOINT = "tpen.callback.Checkpoint"

# The failing loop step and the durable counters it carries. `completed_updates`
# is a multiple of the config's `schedule.every_n: 5`, so the periodic write is
# genuinely eligible -- without that this test would pass for the wrong reason.
FAILING_STEP = 4
NEXT_ITERATION = 5
COMPLETED_UPDATES = 5


class _Trainer:
    """Trainer stand-in reporting only the two required progress counters."""

    def state_dict(self) -> dict[str, int]:
        return {
            "next_iteration": NEXT_ITERATION,
            "completed_updates": COMPLETED_UPDATES,
        }


class _Sampler:
    def mcmc_state_dict(self) -> dict[str, bool]:
        return {"has_burned_in": True}


def _state() -> TrainerState:
    """Build state a healthy `DataIntegrity` passes and `Checkpoint` can save."""

    model = nn.Linear(2, 1, dtype=torch.float64)
    return TrainerState(
        step=FAILING_STEP,
        metrics={"loss": 0.5},
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        batch=ElectronBatch(
            positions=torch.zeros(4, 2, 3, dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]] * 4, dtype=torch.float64),
        ),
        local_energy=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        loss=torch.tensor(1.5, dtype=torch.float64),
        wavefunction_output=WavefunctionOutput(
            logabs=torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float64),
            sign=torch.ones(4, dtype=torch.float64),
        ),
    )


def _production_entries() -> dict[str, dict[str, Any]]:
    """Return the shipped options of the two callbacks this property spans.

    Read with `yaml.safe_load` rather than OmegaConf so the file's ``${run.dir}``
    interpolations stay inert strings; the one that matters is overridden below.
    """

    cfg = yaml.safe_load(PRODUCTION_TRAIN_CONFIG.read_text(encoding="utf-8"))
    entries = {
        str(entry["_target_"]): {
            key: value for key, value in entry.items() if key != "_target_"
        }
        for entry in cfg["callbacks"]
        if str(entry.get("_target_")) in {_DATA_INTEGRITY, _CHECKPOINT}
    }
    # A rename or a dropped entry would otherwise empty this silently and every
    # assertion below would pass while testing nothing.
    assert set(entries) == {_DATA_INTEGRITY, _CHECKPOINT}, (
        f"{PRODUCTION_TRAIN_CONFIG} no longer configures both callbacks: {sorted(entries)}"
    )
    return entries


def _production_order() -> list[str]:
    """Return the two targets in the order the production config lists them."""

    cfg = yaml.safe_load(PRODUCTION_TRAIN_CONFIG.read_text(encoding="utf-8"))
    return [
        str(entry["_target_"])
        for entry in cfg["callbacks"]
        if str(entry.get("_target_")) in {_DATA_INTEGRITY, _CHECKPOINT}
    ]


def _callbacks(
    checkpoint_dir: Path, *, healthy: bool, order: list[str] | None = None
) -> list[Any]:
    """Build the two shipped callbacks in the production config's order.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Replaces the config's unresolved ``${run.dir}/checkpoints``.
    healthy : bool
        When ``False``, the one lever: a negative ceiling no finite fraction can
        satisfy, so the check fails on data that is otherwise perfectly valid.
        Every other option stays exactly as shipped.
    order : list of str, optional
        Overrides the config's order. Only the reordering test passes this.
    """

    entries = _production_entries()
    data_integrity = dict(entries[_DATA_INTEGRITY])
    if not healthy:
        data_integrity["max_nonfinite_logabs_fraction"] = -1.0
    checkpoint = dict(entries[_CHECKPOINT])
    checkpoint["output_dir"] = checkpoint_dir
    checkpoint["schedule"] = instantiate(OmegaConf.create(checkpoint["schedule"]))
    checkpoint["payload"] = instantiate(OmegaConf.create(checkpoint["payload"]))

    build = {
        _DATA_INTEGRITY: lambda: DataIntegrity(**data_integrity),
        _CHECKPOINT: lambda: Checkpoint(**checkpoint),
    }
    return [build[target]() for target in (_production_order() if order is None else order)]


def _run_iteration(context: Any, state: TrainerState) -> None:
    """Replay the two typed boundaries one completed iteration emits.

    `UpdateCompleted` first, exactly as `tpen.training.VMCTrainer.fit` emits it
    after ``optimizer.step()`` returns, so the periodic checkpoint is armed and
    genuinely eligible when the completed-iteration boundary arrives.
    """

    iteration = TrainingIteration(step=FAILING_STEP)
    context.emit(UpdateCompleted(iteration=iteration), state=state)
    context.emit(TrainingIterationCompleted(iteration=iteration), state=state)


def test_a_failed_fail_fast_check_leaves_no_checkpoint_for_that_step(tmp_path) -> None:
    """The property. Independent of which mechanism currently delivers it."""

    checkpoint_dir = tmp_path / "checkpoints"
    context = make_run_context(
        tmp_path, callbacks=_callbacks(checkpoint_dir, healthy=False)
    )

    with pytest.raises(RuntimeError, match="DataIntegrity failed at step 4"):
        _run_iteration(context, _state())

    assert sorted(path.name for path in checkpoint_dir.glob("step_*")) == []
    # `latest.json` is the durability claim: a stray directory nothing points at
    # is inert, but a pointer makes the rejected model the default resume target.
    assert not (checkpoint_dir / "latest.json").exists()


def test_the_same_iteration_does_checkpoint_when_the_check_passes(tmp_path) -> None:
    """Control arm: the harness really would have written one.

    Without this, the assertion above passes just as well when the callbacks are
    misconfigured, the cadence never admits the step, or the state is missing a
    component -- every failure mode where nothing was ever going to be written.
    """

    checkpoint_dir = tmp_path / "checkpoints"
    context = make_run_context(
        tmp_path, callbacks=_callbacks(checkpoint_dir, healthy=True)
    )

    _run_iteration(context, _state())

    assert (checkpoint_dir / "step_000005" / "COMPLETE").exists()
    assert (checkpoint_dir / "latest.json").exists()


def test_checkpointing_before_the_check_is_what_reintroduces_the_defect(
    tmp_path,
) -> None:
    """The regression, reproduced by the one edit that causes it.

    Listing `Checkpoint` ahead of the check restores exactly the #175 behaviour
    the item measured on hardware. Recorded as a test so the property above is
    demonstrably load-bearing rather than incidentally true, and so the cost of
    reordering that list is written down somewhere executable.
    """

    checkpoint_dir = tmp_path / "checkpoints"
    context = make_run_context(
        tmp_path,
        callbacks=_callbacks(
            checkpoint_dir, healthy=False, order=[_CHECKPOINT, _DATA_INTEGRITY]
        ),
    )

    with pytest.raises(RuntimeError, match="DataIntegrity failed at step 4"):
        _run_iteration(context, _state())

    assert (checkpoint_dir / "step_000005" / "COMPLETE").exists()
    assert (checkpoint_dir / "latest.json").exists()
