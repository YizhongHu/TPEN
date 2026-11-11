"""The durable `events.jsonl` boundary sequence of successful runs.

Every other test of the string event path pins either the *failure* sequence
(`tests/integration/test_run_fail_loudness.py`) or the writer in isolation, by
calling `RunContext.emit_event` directly
(`tests/unit/test_event_foundation.py`). Neither reaches the emitters, so seven
run-level names had no test tying them to a real run's durable stream:

    model_built            tpen/runner/train.py
    train_start            tpen/runner/train.py
    train_end              tpen/runner/train.py
    checkpoint_restored    tpen/runner/train.py, tpen/runner/evaluate.py
    step_start             tpen/training/trainer.py
    step_end               tpen/training/trainer.py
    run_end                tpen/runner/train.py, tpen/runner/evaluate.py

Deleting any of those emits left the suite green, which matters because
removing the last writer of an event is indistinguishable -- to a green suite --
from removing something nothing used. This module drives three real runs
through `run_from_config` and pins what each one records.

What is asserted is presence, relative order, and multiplicity of the run-level
boundary names, and nothing else. The stream is projected onto
`BOUNDARY_NAMES` before comparison, so a new event name added anywhere in the
lifecycle does not churn these tests; only a change to the boundaries
themselves does. Payload contents are deliberately not asserted -- they are
pinned where they are contractual, in the fail-loudness and unit tests.

Three runs, sharing one module-scoped fixture because each is a real VMC loop:

1. a fresh training run, covering every boundary except `checkpoint_restored`;
2. a training run resumed from run 1's mid-run checkpoint, which is the only
   way `checkpoint_restored` fires on the `Train` path;
3. an evaluation run restoring run 1's terminal checkpoint with
   ``mode: model_only``, covering the `Evaluate` copies of
   `checkpoint_restored` and `run_end`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from tpen.run import run_from_config

FIXTURE = Path(__file__).resolve().parent / "artifacts" / "training" / "events_sequence.yaml"

# Trainer steps the fixture configures. Read from the fixture rather than
# duplicated, so the expected step-boundary multiplicity cannot drift from it.
MAX_STEPS = int(OmegaConf.load(FIXTURE).trainer.max_steps)

# The run-level boundaries this module owns. Events outside this set (the load
# events, and any typed-path or future additions) are filtered out before
# comparison: they have their own tests, and including them here would make an
# unrelated addition break these assertions.
BOUNDARY_NAMES = frozenset(
    {
        "run_start",
        "model_built",
        "checkpoint_restored",
        "train_start",
        "step_start",
        "step_end",
        "train_end",
        "run_end",
    }
)

# The fixture's composed stream has cadence 2 over applied updates, so it writes
# exactly one mid-run checkpoint, whose `next_iteration` is 2. A checkpoint is
# named by that cursor, and `fit` resumes at it, so the two are one constant:
# resuming from `step_000002` leaves exactly trainer step 2 to run.
RESUMED_STEP = 2
RESUME_FROM = f"step_{RESUMED_STEP:06d}"
# The composed stream's terminal path names the resume cursor of a completed
# `MAX_STEPS` run.
TERMINAL_CHECKPOINT = f"step_{MAX_STEPS:06d}"


@dataclass(frozen=True)
class _Runs:
    """Run directories of the three runs the module-scoped fixture drives."""

    train: Path
    resume: Path
    evaluate: Path


def _events(run_dir: Path) -> list[dict]:
    """Return every record in a run's durable event stream, in written order."""

    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _boundary_sequence(run_dir: Path) -> list[str]:
    """Return the run-level boundary names a run recorded, in written order."""

    return [event["event"] for event in _events(run_dir) if event["event"] in BOUNDARY_NAMES]


def _execute(cfg: DictConfig, root: Path) -> Path:
    """Run one configured run under `root` and return its run directory."""

    cfg.run.root = str(root)
    # `raise_exceptions=True` so a broken fixture surfaces its own traceback
    # instead of an opaque exit code; the exit-code assertion still catches a
    # handled failure that returns 1 without raising.
    exit_code = run_from_config(
        cfg, config_path=str(FIXTURE), command="pytest", raise_exceptions=True
    )
    assert exit_code == 0
    run_dirs = list(root.glob("events_sequence/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir under {root}, found {run_dirs}"
    return run_dirs[0]


def _train_config(load_path: Path | None = None) -> DictConfig:
    """Return the fixture config, optionally resuming from `load_path`."""

    cfg = OmegaConf.load(FIXTURE)
    if load_path is not None:
        # Only `load` changes. The checkpoint verifies the model, optimizer,
        # trainer, sampler, and Hamiltonian config hashes, and every one of
        # those sections is untouched here, so the resume is accepted.
        cfg.load = {
            "path": str(load_path),
            "mode": "train_resume",
            "strict": True,
            "allow_protocol_mismatch": False,
        }
    return cfg


def _evaluate_config(load_path: Path) -> DictConfig:
    """Return an evaluation config sharing the fixture's hashed sections.

    Built from the training fixture rather than written out separately: a
    ``model_only`` restore verifies the ``model`` and ``hamiltonian_terms``
    config hashes, so deriving the evaluation config from the one that produced
    the checkpoint makes those sections identical by construction instead of by
    two files being kept in sync.
    """

    cfg = OmegaConf.load(FIXTURE)
    cfg.load = {
        "path": str(load_path),
        "mode": "model_only",
        "strict": True,
        "allow_protocol_mismatch": False,
    }
    cfg.runner = {
        "_target_": "tpen.runner.Evaluate",
        "model": "${model}",
        "load": "${load}",
        "evaluator": "${evaluator}",
    }
    cfg.evaluator = {
        "_target_": "tpen.evaluation.Evaluator",
        "namespace": "eval",
        "tasks": [
            {
                "name": "null_task",
                "namespace": "eval/null_task",
                "output_dir": "${run.dir}/null_task",
                "generator": {"_target_": "tests.helpers.evaluation_components.NullGenerator"},
                "calculators": [
                    {"_target_": "tests.helpers.evaluation_components.IdentityCalculator"}
                ],
                "summaries": [{"_target_": "tests.helpers.evaluation_components.MetricSummary"}],
            }
        ],
    }
    # The training checkpoint writers have nothing to observe in an evaluation
    # run, and `events.jsonl` is written by the context rather than by any
    # callback, so the evaluation run carries no callback battery at all.
    cfg.callbacks = []
    return cfg


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> _Runs:
    """Drive the fresh, resumed, and evaluation runs once for this module.

    Each run gets its own root so the run-directory glob stays unambiguous even
    when two runs start within the same timestamp second.
    """

    root = tmp_path_factory.mktemp("events_sequence")

    train_dir = _execute(_train_config(), root / "train")
    checkpoints = train_dir / "checkpoints"
    # Guard the fixture itself: the resume and evaluation runs are meaningless
    # if run 1 did not write the checkpoints they restore.
    for name in (RESUME_FROM, TERMINAL_CHECKPOINT):
        assert (checkpoints / name / "COMPLETE").is_file(), f"run 1 wrote no {name} checkpoint"

    resume_dir = _execute(_train_config(checkpoints / RESUME_FROM), root / "resume")
    evaluate_dir = _execute(_evaluate_config(checkpoints / TERMINAL_CHECKPOINT), root / "eval")
    return _Runs(train=train_dir, resume=resume_dir, evaluate=evaluate_dir)


def test_successful_train_run_records_every_boundary(runs: _Runs) -> None:
    """A fresh training run records the whole `Train` boundary sequence.

    Exact equality on the projected stream, so this fails on a deleted emit, a
    reordered pair, a duplicated boundary, and -- because
    `checkpoint_restored` is in `BOUNDARY_NAMES` but not in the expected list --
    on a run that restores a checkpoint it was never configured to restore.
    """

    assert _boundary_sequence(runs.train) == [
        "run_start",
        "model_built",
        "train_start",
        *(["step_start", "step_end"] * MAX_STEPS),
        "train_end",
        "run_end",
    ]


def test_resumed_train_run_records_checkpoint_restored(runs: _Runs) -> None:
    """A resumed training run records `checkpoint_restored` before `train_start`.

    This is the only path on which `Train` emits `checkpoint_restored` at all,
    and its position is contractual: it reports a restore that has already
    happened, so it must follow `model_built` (the optimizer it restores into is
    built there) and precede `train_start`.
    """

    assert _boundary_sequence(runs.resume) == [
        "run_start",
        "model_built",
        "checkpoint_restored",
        "train_start",
        # One iteration left: the restored cursor is `RESUMED_STEP`, not 0.
        "step_start",
        "step_end",
        "train_end",
        "run_end",
    ]


def test_resumed_train_run_steps_resume_at_the_restored_cursor(runs: _Runs) -> None:
    """The resumed run's step boundaries carry the restored cursor.

    Without this, a resume that silently restarted from step 0 but stopped early
    would still produce the sequence above; the durable `step` field is what
    distinguishes a real resume from a coincidence of counts.
    """

    steps = {
        event["step"] for event in _events(runs.resume) if event["event"] in ("step_start", "step_end")
    }
    assert steps == {RESUMED_STEP}


def test_successful_evaluation_run_records_its_boundaries(runs: _Runs) -> None:
    """An evaluation run records its own `checkpoint_restored` and `run_end`.

    `Evaluate` owns copies of both names that are distinct emit sites from
    `Train`'s, so deleting either one is invisible to the training tests above.
    """

    assert _boundary_sequence(runs.evaluate) == [
        "run_start",
        "checkpoint_restored",
        "run_end",
    ]
