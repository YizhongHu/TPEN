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


def test_runtime_equivariance_train_resume_reconstructs_across_processes(tmp_path: Path) -> None:
    """A real TrainResume restore preserves checker schedule and permutation oracle."""

    prefix = _launch(tmp_path, "prefix")
    reference = _launch(tmp_path, "reference")
    successor = _launch(tmp_path, "successor")

    assert len({prefix["pid"], reference["pid"], successor["pid"]}) == 3
    assert successor["restored"] is True
    assert successor["trainer_next_iteration"] == 5
    assert reference["selected"] == successor["selected"]
    assert reference["records"] == successor["records"]
    assert [record["step"] for record in successor["records"]] == [5, 5, 10, 10]
    assert all(record["metrics"]["n_comparisons"] > 0 for record in successor["records"][::2])
    assert all(record["metrics"]["passed"] is True for record in successor["records"])


def test_runtime_equivariance_failure_artifact_uses_event_step_after_restore(
    tmp_path: Path,
) -> None:
    """A deliberately failing restored check names the first post-cursor step."""

    _launch(tmp_path, "prefix")
    failure = _launch(tmp_path, "successor", failure=True)

    assert failure["restored"] is True
    assert "failed at step 5" in failure["failure_message"]
    assert failure["artifact_paths"] == [
        "equivariance-artifacts/full_model/step_000005/failure.json"
    ]
