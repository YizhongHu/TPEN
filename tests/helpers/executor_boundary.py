"""Shared executor-only subprocess harness for integration tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def executor_environment() -> dict[str, str]:
    """Build an environment whose child imports the checkout's executor code."""

    environment = os.environ.copy()
    for variable in (
        "PYTHONHOME",
        "PYTHONOPTIMIZE",
        "PYTHONNOUSERSITE",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONWARNINGS",
    ):
        environment.pop(variable, None)
    repository_root = Path(__file__).resolve().parents[2]
    executor_src = repository_root / "executor" / "src"
    executor_tests = repository_root / "executor" / "tests"
    environment["PYTHONPATH"] = os.pathsep.join((str(executor_src), str(executor_tests)))
    return environment


def executor_call(
    child_code: str,
    tmp_path: Path,
    action: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one executor-only child operation and decode its JSON result."""

    try:
        completed = subprocess.run(
            [sys.executable, "-c", child_code, action, json.dumps(dict(payload), sort_keys=True)],
            cwd=tmp_path,
            env=executor_environment(),
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"executor child timed out ({action}, timeout=120s):\n"
            f"stdout={error.stdout}\nstderr={error.stderr}"
        ) from error
    if completed.returncode != 0:
        raise AssertionError(
            f"executor child failed ({action}, rc={completed.returncode}):\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    if completed.stderr:
        raise AssertionError(f"executor child wrote stderr ({action}): {completed.stderr}")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise AssertionError(f"executor child result is not a mapping ({action}): {result!r}")
    boundary = result.get("_executor_boundary")
    if not isinstance(boundary, dict):
        raise AssertionError(f"executor child boundary is not a mapping ({action}): {boundary!r}")
    if boundary.get("tpen_imports") != []:
        raise AssertionError(
            f"executor child crossed tpen import boundary ({action}): {boundary!r}"
        )
    if boundary.get("tpen_modules") != []:
        raise AssertionError(f"executor child loaded tpen modules ({action}): {boundary!r}")
    result.pop("_executor_boundary", None)
    return result


__all__ = ["executor_call", "executor_environment"]
