"""End-to-end artifacts written from the typed run lifecycle.

WHY THESE ARE INTEGRATION TESTS AND NOT UNIT ONES. ``status.json`` and the
empty-suite ``diagnostics/index.json`` are written by `tpen.callback.Status` and
`tpen.callback.ArtifactIndex` from a subscription group declaring
`tpen.callback.cadence.SubscriptionGroup.stateless`, and those two are the first
consumers of that mechanism anywhere. Three separate things have to hold for a
byte to reach disk: `tpen.run.run_from_config` has to emit the boundary,
`tpen.artifacts.RunContext._dispatch_occurrence` has to route a state-free
occurrence to a `StatefulCallback` rather than skip it, and the callback has to
answer on ``handle_stateless_occurrence_impl``. A unit test that hands an
occurrence to the callback by hand proves only the third, and the first two are
exactly the ones that fail SILENTLY -- a missing emit and a skipped dispatch both
look like a passing suite and an absent file.

The failure cases matter most. Runs fail on a cluster far more often than a
green test suite is read, and before these landed nothing at any level asserted
that ``status.json`` records a failed run at all: the unit tests drove only the
success boundaries, and ``test_run_fail_loudness.py`` asserts ``error.json`` and
``events.jsonl`` but never ``status.json``.

A run can fail in TWO shapes, and only one of them raises. A crash reaches
`tpen.run_events.RunFailed`; an evaluation SUITE whose tasks failed raises
nothing at all, returns ``RunResult(status="failed")``, and reaches
`tpen.run_events.RunCompleted` like any success. The second shape used to write
``completed`` and exit ``0``, and it is the one automation reads: the
``_attempt_already_completed`` check in ``experiments/toolkit/task_state.py``
banks a row as finished on ``latest.json`` plus a ``completed`` ``status.json``,
so a failed evaluation was never retried.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from tpen.inspection import inspect_eval_run, inspect_run
from tpen.run import run_from_config

_STATUS_CALLBACK = {
    "_target_": "tpen.callback.Status",
    "output_path": "${run.dir}/status.json",
    # Off so the status boxes do not flood the test log; the artifact is what is
    # under test, and the terminal line has its own unit coverage.
    "terminal": False,
}

_ARTIFACT_INDEX_CALLBACK = {"_target_": "tpen.callback.ArtifactIndex"}
_EVALUATION_TIMING_CALLBACK = {"_target_": "tpen.callback.EvaluationTiming"}
_METADATA_CALLBACK = {
    "_target_": "tpen.callback.Metadata",
    "output_path": "${run.dir}/metadata.json",
}
_JSONL_LOGGER = {"_target_": "tpen.logging.JSONL", "path": "${run.dir}/metrics.jsonl"}


class _RaisingGenerator:
    """Evaluation generator that always raises, to drive a failed SUITE.

    A generator failure is the one component failure that makes its whole task
    ``failed`` outright (`tpen.evaluation.evaluator.Evaluator._evaluate_task`
    returns before any calculator runs), and one failed task aggregates the
    suite to ``failed``. The exception never escapes the evaluator, which is
    precisely why this shape reaches the COMPLETION boundary rather than the
    failure one.
    """

    def generate(self, *, model: object, context: object) -> object:
        """Raise instead of producing a bundle."""

        del model, context
        raise RuntimeError("generator exploded")


def _failing_task_cfg() -> dict[str, object]:
    return {
        "name": "boom",
        "namespace": "eval/boom",
        "output_dir": "${run.dir}/boom",
        "generator": {"_target_": f"{__name__}._RaisingGenerator"},
        "calculators": [],
        "summaries": [],
    }


def _cfg(tmp_path: Path, **extra) -> DictConfig:
    base = {
        "experiment": {"name": "s", "sector": "s", "run_name": "s"},
        "run": {"root": str(tmp_path), "run_id": None, "dir": None},
        "runtime": {"seed": 0},
        "terminal": {"enabled": False},
        "callbacks": [_STATUS_CALLBACK, _ARTIFACT_INDEX_CALLBACK],
        "evaluator": {
            "_target_": "tpen.evaluation.Evaluator",
            "namespace": "eval",
            "tasks": [],
        },
        "runner": {
            "_target_": "tpen.runner.Evaluate",
            # Evaluator.evaluate has a real torch.nn.Module contract even for
            # an empty task list; Identity is the minimal no-op model.
            "model": {"_target_": "torch.nn.Identity"},
            "evaluator": "${evaluator}",
        },
    }
    base.update(extra)
    return OmegaConf.create(base)


def _run_dir(tmp_path: Path) -> Path:
    run_dirs = list(tmp_path.glob("s/s/*"))
    assert len(run_dirs) == 1, run_dirs
    return run_dirs[0]


def _events(run_dir: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_a_successful_run_records_completed_and_an_empty_index(tmp_path: Path) -> None:
    """The success path, through the real harness, for both callbacks at once.

    The empty suite is the whole reason `ArtifactIndex` observes a run boundary:
    every run with at least one task rewrites the index from the task boundary,
    so this is the only shape where the run-level subscription is what puts the
    file on disk.
    """

    assert run_from_config(_cfg(tmp_path), config_path="s.yaml", command="pytest") == 0

    run_dir = _run_dir(tmp_path)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["current_event"] == "run_end"
    assert status["start_time"] is not None
    assert status["end_time"] is not None
    assert status["exception_type"] is None
    assert status["exception_message"] is None

    run_end = [event for event in _events(run_dir) if event["event"] == "run_end"]
    assert len(run_end) == 1
    assert run_end[0]["payload"] == {"status": "completed"}

    assert json.loads((run_dir / "diagnostics" / "index.json").read_text()) == {"tasks": []}


def test_a_real_completed_eval_run_has_eval_evidence_and_preserves_inspect_run_behavior(
    tmp_path: Path,
) -> None:
    """The new API reads the real eval artifacts without requiring checkpoints."""

    assert run_from_config(_cfg(tmp_path), config_path="s.yaml", command="pytest") == 0

    run_dir = _run_dir(tmp_path)
    evidence = inspect_eval_run(run_dir)
    assert evidence.verdict == "completed"
    assert evidence.run_status == "completed"
    assert evidence.outcome == "completed"
    assert evidence.resolved_config_identity is not None
    assert evidence.refusal is None
    assert evidence == type(evidence).from_mapping(evidence.to_dict())

    # This is intentionally pinned: the legacy checkpoint-oriented API still
    # cannot represent a completed evaluation with no checkpoint.
    with pytest.raises(ValueError):
        inspect_run(run_dir)


def test_a_real_failed_eval_run_has_failed_eval_evidence(tmp_path: Path) -> None:
    """A real failed evaluation suite maps to the failed eval verdict."""

    cfg = _cfg(
        tmp_path,
        callbacks=[_STATUS_CALLBACK, _METADATA_CALLBACK, _EVALUATION_TIMING_CALLBACK],
        evaluator={
            "_target_": "tpen.evaluation.Evaluator",
            "namespace": "eval",
            "tasks": [_failing_task_cfg()],
        },
    )
    assert run_from_config(cfg, config_path="s.yaml", command="pytest") == 1

    evidence = inspect_eval_run(_run_dir(tmp_path))
    assert evidence.verdict == "failed"
    assert evidence.run_status == "failed"
    assert evidence.outcome == "failed"
    assert evidence.refusal is None


def test_eval_evidence_refuses_missing_outcome_identity_and_matching_status(tmp_path: Path) -> None:
    """Each negative case changes one artifact from a real completed eval."""

    assert run_from_config(_cfg(tmp_path), config_path="s.yaml", command="pytest") == 0
    run_dir = _run_dir(tmp_path)

    (run_dir / "run-outcome.json").unlink()
    assert inspect_eval_run(run_dir).refusal["code"] == "missing-run-outcome"

    # Recreate a fresh real run for each independent one-file mutation.
    identity_root = tmp_path / "identity"
    assert run_from_config(_cfg(identity_root), config_path="s.yaml", command="pytest") == 0
    identity_dir = _run_dir(identity_root)
    (identity_dir / "run_start.json").unlink()
    assert inspect_eval_run(identity_dir).refusal["code"] == "missing-run-identity"

    status_root = tmp_path / "status"
    assert run_from_config(_cfg(status_root), config_path="s.yaml", command="pytest") == 0
    status_dir = _run_dir(status_root)
    status_path = status_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["status"] = "failed"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert inspect_eval_run(status_dir).refusal["code"] == "status-outcome-mismatch"


def test_a_failed_evaluation_suite_returns_nonzero_and_records_failed(tmp_path: Path) -> None:
    """A task failure is a failed result, not a successful run boundary."""

    cfg = _cfg(
        tmp_path,
        callbacks=[_STATUS_CALLBACK, _METADATA_CALLBACK, _EVALUATION_TIMING_CALLBACK],
        loggers=[_JSONL_LOGGER],
        evaluator={
            "_target_": "tpen.evaluation.Evaluator",
            "namespace": "eval",
            "tasks": [_failing_task_cfg()],
        },
    )

    assert run_from_config(cfg, config_path="s.yaml", command="pytest") == 1

    run_dir = _run_dir(tmp_path)
    status = json.loads((run_dir / "status.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert status["status"] == "failed"
    assert status["current_event"] == "run_end"
    assert metadata["status"] == "failed"

    run_end = [event for event in _events(run_dir) if event["event"] == "run_end"]
    assert len(run_end) == 1
    assert run_end[0]["payload"] == {"status": "failed"}

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    eval_perf = [record for record in metrics if record["namespace"] == "eval/perf"]
    assert eval_perf[-1]["metrics"]["failed"] is True
    suite_status = [record for record in metrics if record["namespace"] == "eval/status"]
    assert suite_status[-1]["metrics"]["suite_failed"] is True


def test_a_run_that_fails_inside_the_runner_records_failed(tmp_path: Path) -> None:
    """A failed run must still say so on disk, with the failure's identity.

    ``run_end`` and ``exception`` were two separate legacy strings, and mapping
    both onto `tpen.run_events.RunCompleted` would drop this case while leaving
    every success-path test green.
    """

    missing = tmp_path / "missing" / "latest.json"
    cfg = _cfg(
        tmp_path,
        load={
            "path": str(missing),
            "mode": "model_only",
            "strict": True,
            "allow_protocol_mismatch": False,
        },
        runner={
            "_target_": "tpen.runner.Evaluate",
            "model": None,
            "load": "${load}",
            "evaluator": "${evaluator}",
        },
    )

    assert run_from_config(cfg, config_path="s.yaml", command="pytest") == 1

    run_dir = _run_dir(tmp_path)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["current_event"] == "exception"
    # Not merely non-null: the identity is what makes the artifact worth reading,
    # and it now travels as typed fields on `RunFailed` rather than as a live
    # exception object in an untyped payload.
    assert status["exception_type"] == "FileNotFoundError"
    assert str(missing) in status["exception_message"]
    assert status["start_time"] is not None
    assert status["end_time"] is not None

    # The runner never returned, so the completion boundary never happened and
    # the index was never written -- the same as before this migration, where
    # ``run_end`` was the last statement of ``Evaluate.run`` rather than a
    # ``finally``.
    assert not (run_dir / "diagnostics" / "index.json").exists()


def test_a_run_that_fails_before_the_runner_exists_still_records_failed(
    tmp_path: Path,
) -> None:
    """`RunStarted` fires before the runner is built, and `RunFailed` after.

    `tpen.run.run_from_config` emits `RunStarted` and only then instantiates the
    runner, so a configuration error lands between the two boundaries: the run
    is recorded as ``running`` and then as ``failed``, and never reaches a
    completion boundary at all.
    """

    cfg = _cfg(tmp_path, runner={"_target_": "tpen.runner.Evaluate", "model": None})

    assert run_from_config(cfg, config_path="s.yaml", command="pytest") == 1

    status = json.loads((_run_dir(tmp_path) / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["exception_type"] == "ValueError"
    assert "evaluator" in status["exception_message"]
