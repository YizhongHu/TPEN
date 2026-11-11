"""The smoke grid's tiny workload must still leave the artifact the next stage needs.

One property, asserted behaviourally rather than by arithmetic: after the smoke's
``training.max_steps`` iterations, the run has actually written a terminal
checkpoint.

Why it needs a test at all. The scan's checkpoint stream is explicitly composed
with ``TerminalOnly``, so the terminal boundary cannot be suppressed by a
periodic cadence. This test drives the real callback built from the real composed
config -- the smoke's static overrides applied exactly as the launcher applies
them -- and checks the checkpoint directory is on disk. The config-surface test
also pins the absence of a legacy checkpoint cadence override.

The workload is replayed rather than trained: the point is the callback contract
at the smoke's step budget, and running the real trainer would need a GPU and
would test the trainer instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tpen.artifacts import RunContext
from tpen.callback import Checkpoint
from tpen.checkpoint import TerminalOnly, TrainResume
from tpen.events import Occurrence
from tpen.training.events import (
    TrainingCompleted,
    TrainingIteration,
    TrainingIterationCompleted,
    UpdateCompleted,
)
from tpen.training.state import TrainerState

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO_ROOT / "experiments" / "hooke" / "tpen-pair-scan-v1"
CONFIG_DIR = STUDY_DIR / "configs"
LIBRARY = REPO_ROOT / "experiments" / "hooke" / "choices" / "basis_levels.yaml"
CHECKPOINT_TARGET = "tpen.callback.Checkpoint"

# The two stages whose static overrides carry a train workload. Both are checked:
# the final-train stage writes the checkpoint ``07_final_eval`` restores, so a
# cadence mistake there breaks the final measurement rather than the scan.
TRAIN_STAGES = ("train", "final_train")


# ---------------------------------------------------------------------------
# Config composition
# ---------------------------------------------------------------------------
def _smoke_static_overrides(stage: str) -> dict[str, Any]:
    """Return the smoke grid's static overrides for one stage."""

    grid = OmegaConf.load(CONFIG_DIR / "smoke.yaml")
    overrides = OmegaConf.select(grid, f"static_overrides.{stage}")
    assert overrides is not None, f"smoke.yaml declares no static_overrides.{stage}"
    return dict(OmegaConf.to_container(overrides, resolve=True))


def _train_config(stage: str, run_dir: Path) -> DictConfig:
    """Compose train.yaml the way the launcher does, for one smoke stage.

    Merged with the basis library (``tpen.run.load_config`` reads exactly one YAML
    file), then given the stage's static overrides as a dotlist, exactly as
    ``run.py`` receives them. ``run.dir`` is filled in because the callback's
    ``output_dir`` interpolates it.
    """

    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_DIR / "train.yaml"), OmegaConf.load(LIBRARY)
    )
    dotlist = [f"{path}={value}" for path, value in _smoke_static_overrides(stage).items()]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    OmegaConf.update(cfg, "run.dir", str(run_dir), merge=False)
    return cfg


def _checkpoint_callback(cfg: DictConfig) -> Checkpoint:
    """Instantiate the config's own ``Checkpoint`` spec, kwargs and all."""

    specs = [spec for spec in cfg.callbacks if spec.get("_target_") == CHECKPOINT_TARGET]
    assert len(specs) == 1, "train.yaml must register exactly one Checkpoint callback"
    callback = instantiate(specs[0])
    assert isinstance(callback, Checkpoint)
    return callback


# ---------------------------------------------------------------------------
# Minimal loop replay
# ---------------------------------------------------------------------------
class _Trainer:
    """Trainer stub reporting the two durable progress counters."""

    def __init__(self, *, next_iteration: int, completed_updates: int) -> None:
        self.next_iteration = int(next_iteration)
        self.completed_updates = int(completed_updates)

    def state_dict(self) -> dict[str, int]:
        return {
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
        }


class _Sampler:
    """Sampler stub exposing the MCMC state ``save_checkpoint`` reads."""

    def mcmc_state_dict(self) -> dict[str, bool]:
        return {"has_burned_in": True}


class _Context(RunContext):
    """`RunContext` carrying only what ``save_checkpoint`` reads.

    Subclasses `tpen.artifacts.RunContext` rather than duck-typing a namespace
    because the typed-occurrence dispatch path annotates its context parameter and
    the suite runs typeguard over ``tpen``. Mirrors
    ``tests/unit/training/test_checkpoint_callback.py``.
    """

    def __init__(self, cfg: DictConfig, metadata: Any, run_dir: Path) -> None:
        self.cfg = cfg
        self.metadata = metadata
        self._run_dir = Path(run_dir)

    @property
    def run_dir(self) -> Path:
        """Return the stubbed run directory."""

        return self._run_dir

    def __repr__(self) -> str:
        return f"_Context(run_dir={self._run_dir!r})"


def _context(cfg: DictConfig, run_dir: Path) -> _Context:
    metadata = SimpleNamespace(
        run_id="smoke",
        device="cpu",
        dtype="float64",
        git_commit="deadbeef",
        git_branch="claude/scan-grid",
        dirty_worktree=False,
        command="pytest",
        extra={"torch_version": torch.__version__},
    )
    return _Context(cfg=cfg, metadata=metadata, run_dir=run_dir)


def _state(step: int) -> TrainerState:
    """Build loop state for one completed iteration that applied its update."""

    model = torch.nn.Linear(2, 1)
    return TrainerState(
        step=step,
        metrics={"loss": 0.5},
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1.0e-3),
        trainer=_Trainer(next_iteration=step + 1, completed_updates=step + 1),
        sampler=_Sampler(),
    )


def _run_loop(callback: Checkpoint, context: _Context, max_steps: int) -> None:
    """Replay ``max_steps`` iterations and the terminal boundary, in loop order.

    Mirrors `tpen.training.trainer.VMCTrainer.fit`: each iteration emits
    ``UpdateCompleted`` right after ``optimizer.step()`` returns and
    ``TrainingIterationCompleted`` at the end of the body, and the runner emits
    ``TrainingCompleted`` once the loop returns.
    """

    state = None
    for step in range(max_steps):
        state = _state(step)
        iteration = TrainingIteration(step=step)
        callback.handle_occurrence(
            Occurrence(event=UpdateCompleted(iteration=iteration), count=step + 1),
            context,
            state,
        )
        callback.handle_occurrence(
            Occurrence(event=TrainingIterationCompleted(iteration=iteration), count=step + 1),
            context,
            state,
        )
    assert state is not None, "the smoke budget must run at least one iteration"
    callback.handle_occurrence(Occurrence(event=TrainingCompleted(), count=1), context, state)


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage", TRAIN_STAGES)
def test_the_smoke_budget_writes_a_terminal_checkpoint(tmp_path: Path, stage: str) -> None:
    """A completed smoke row leaves a restorable checkpoint on disk.

    The directory name is the resume cursor, so the terminal write of a
    ``max_steps``-step run lands in ``step_{max_steps:06d}``. The COMPLETE marker
    is what the validation stage looks for; without it the row reports
    ``completed`` and validation dies.
    """

    max_steps = int(_smoke_static_overrides(stage)["training.max_steps"])
    run_dir = tmp_path / stage
    cfg = _train_config(stage, run_dir)
    callback = _checkpoint_callback(cfg)

    _run_loop(callback, _context(cfg, run_dir), max_steps)

    checkpoints = run_dir / "checkpoints"
    terminal = checkpoints / f"step_{max_steps:06d}"
    assert checkpoints.is_dir(), "the checkpoints directory must not be empty"
    assert terminal.is_dir(), sorted(path.name for path in checkpoints.iterdir())
    assert (terminal / "COMPLETE").is_file()
    assert (terminal / "model.pt").is_file()
    assert (checkpoints / "latest.json").is_file()


@pytest.mark.parametrize("stage", TRAIN_STAGES)
def test_the_terminal_write_is_ungated_so_any_step_budget_is_safe(
    tmp_path: Path, stage: str
) -> None:
    """``TerminalOnly`` is what makes the budget a free choice.

    The timing probe owns ``max_steps`` for the production grid. Pinning the
    semantic schedule, not just one outcome, prevents a later cadence from
    suppressing the artifact.
    """

    cfg = _train_config(stage, tmp_path / stage)
    callback = _checkpoint_callback(cfg)

    assert callback.terminal is True
    assert callback.periodic is False
    assert callback.schedule == TerminalOnly()
    assert callback.payload == TrainResume()
    # Every budget an operator could pick, including the awkward ones.
    for max_steps in (1, 3, 4, 7, 60, 97, 500):
        assert callback.schedule.should_run(max_steps, terminal=True) is True


def test_the_smoke_reduces_the_train_budget_it_claims_to_reduce() -> None:
    """Guards the tests above against passing on an unreduced workload.

    If the smoke's overrides stopped applying -- a renamed key, a dropped stage --
    the checks above would still pass while quietly exercising the 500-step
    production budget. So the reduction itself is asserted.
    """

    production = OmegaConf.load(CONFIG_DIR / "train.yaml")
    for stage in TRAIN_STAGES:
        overrides = _smoke_static_overrides(stage)
        cfg = _train_config(stage, Path("/tmp") / stage)

        assert overrides["training.max_steps"] < int(production.training.max_steps)
        assert overrides["sampler_params.n_walkers"] < int(production.sampler_params.n_walkers)
        # And the override actually reached the composed config.
        assert int(cfg.training.max_steps) == int(overrides["training.max_steps"])
        assert int(cfg.sampler_params.n_walkers) == int(overrides["sampler_params.n_walkers"])
