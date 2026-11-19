"""Distinct-process RuntimeEquivariance reconstruction evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[3]
WORKER = ROOT / "tests" / "helpers" / "runtime_equivariance_recovery.py"


def _launch(root: Path, mode: str, *, failure: bool = False) -> dict[str, object]:
    """Launch one small CPU float64 recovery fixture in a fresh PID."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    command = [sys.executable, str(WORKER), "--mode", mode, "--root", str(root)]
    if failure:
        command.append("--failure")
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"recovery worker failed with exit code {exc.returncode}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        )
    return json.loads(completed.stdout)


def _eligible_steps(*, start: int, end: int, every: int, start_step: int) -> list[int]:
    """Derive eligible durable steps using ``StepCadence``'s window rule."""

    return [
        step
        for step in range(start, end)
        if step >= start_step and (step - start_step) % every == 0
    ]


def _assert_sequence(label: str, actual: object, expected: object) -> None:
    """Keep complete actual/expected receipts in a failed assertion."""

    assert actual == expected, (
        f"{label} mismatch\nactual: {actual!r}\nexpected: {expected!r}"
    )


def test_runtime_equivariance_train_resume_reconstructs_across_processes(tmp_path: Path) -> None:
    """A real TrainResume restore preserves checker schedule and permutation oracle."""

    prefix = _launch(tmp_path, "prefix")
    reference = _launch(tmp_path, "reference")
    successor = _launch(tmp_path, "successor")

    reference_schedule = reference["schedule"]
    prefix_schedule = prefix["schedule"]
    successor_schedule = successor["schedule"]
    every = int(reference_schedule["every_n_steps"])
    start_step = int(reference_schedule["start_step"])
    cursor = int(successor_schedule["run_start"])
    run_end = int(reference_schedule["run_end"])
    reference_steps = _eligible_steps(
        start=int(reference_schedule["run_start"]),
        end=run_end,
        every=every,
        start_step=start_step,
    )
    prefix_steps = _eligible_steps(
        start=int(prefix_schedule["run_start"]),
        end=int(prefix_schedule["run_end"]),
        every=every,
        start_step=start_step,
    )
    successor_steps = _eligible_steps(
        start=cursor,
        end=int(successor_schedule["run_end"]),
        every=every,
        start_step=start_step,
    )

    assert len({prefix["pid"], reference["pid"], successor["pid"]}) == 3
    assert successor["restored"] is True
    assert successor["trainer_next_iteration"] == successor_schedule["run_end"]
    assert successor["checker_types"] == [
        "FullModelEquivarianceChecker",
        "TraceEquivarianceChecker",
    ]
    assert all(
        event["event_step"] == event["state_step"]
        for event in successor["completed_events"]
    )
    checker_count = len(reference["checker_types"])
    expected_reference_step_sequence = [
        step for step in reference_steps for _ in range(checker_count)
    ]
    expected_prefix_step_sequence = [
        step for step in prefix_steps for _ in range(checker_count)
    ]
    expected_successor_step_sequence = [
        step for step in successor_steps for _ in range(checker_count)
    ]
    _assert_sequence(
        "reference eligible step sequence",
        [call["step"] for call in reference["selection_calls"]],
        expected_reference_step_sequence,
    )
    _assert_sequence(
        "prefix eligible step sequence",
        [call["step"] for call in prefix["selection_calls"]],
        expected_prefix_step_sequence,
    )
    _assert_sequence(
        "successor eligible step sequence",
        [call["step"] for call in successor["selection_calls"]],
        expected_successor_step_sequence,
    )

    expected_prefix_calls = [
        call for call in reference["selection_calls"] if call["step"] < cursor
    ]
    expected_successor_calls = [
        call for call in reference["selection_calls"] if call["step"] >= cursor
    ]
    _assert_sequence(
        "prefix selector calls",
        prefix["selection_calls"],
        expected_prefix_calls,
    )
    _assert_sequence(
        "successor selector calls",
        successor["selection_calls"],
        expected_successor_calls,
    )

    expected_prefix_records = [
        record for record in reference["records"] if record["step"] < cursor
    ]
    expected_successor_records = [
        record for record in reference["records"] if record["step"] >= cursor
    ]
    _assert_sequence("prefix records", prefix["records"], expected_prefix_records)
    _assert_sequence(
        "successor records", successor["records"], expected_successor_records
    )

    first_call_by_step = {}
    for call in reference["selection_calls"]:
        first_call_by_step.setdefault(call["step"], call["images"])
    assert len(reference_steps) >= 2
    reference_calls = reference["selection_calls"]
    assert first_call_by_step[reference_steps[0]] != first_call_by_step[reference_steps[1]], (
        "selector must be step-discriminating; "
        f"step {reference_steps[0]} and {reference_steps[1]} selected the same images: "
        f"{reference_calls!r}"
    )

    successor_event_steps = [
        event["event_step"] for event in successor["completed_events"]
    ]
    ineligible_step = next(
        step
        for step in range(cursor, int(successor_schedule["run_end"]))
        if step not in successor_steps
    )
    assert ineligible_step in successor_event_steps
    assert all(record["metrics"]["n_comparisons"] > 0 for record in successor["records"])
    assert all(record["metrics"]["passed"] is True for record in successor["records"])


def test_runtime_equivariance_failure_artifact_uses_event_step_after_restore(
    tmp_path: Path,
) -> None:
    """A deliberately failing restored check names the first post-cursor step."""

    prefix = _launch(tmp_path, "prefix")
    failure = _launch(tmp_path, "successor", failure=True)

    assert failure["restored"] is True
    successor_schedule = failure["schedule"]
    first_successor_step = _eligible_steps(
        start=int(successor_schedule["run_start"]),
        end=int(successor_schedule["run_end"]),
        every=int(successor_schedule["every_n_steps"]),
        start_step=int(successor_schedule["start_step"]),
    )[0]
    assert f"failed at step {first_successor_step}" in failure["failure_message"]
    expected_artifacts = [
        {
            "path": (
                "equivariance-artifacts/full_model/"
                f"step_{first_successor_step:06d}/failure.json"
            ),
            "step": first_successor_step,
        }
    ]
    _assert_sequence("failure artifact records", failure["artifact_records"], expected_artifacts)
    assert int(prefix["schedule"]["run_end"]) == int(successor_schedule["run_start"])
