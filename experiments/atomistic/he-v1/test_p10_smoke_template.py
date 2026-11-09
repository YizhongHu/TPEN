"""Pin the P10 smoke summary to product-shaped artifacts and the executor writer."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest


HERE = Path(__file__).parent
TEMPLATE = HERE / "templates" / "p10-smoke.pbs"
SUMMARY = HERE / "p10_summary.py"


def _load_summary():
    spec = importlib.util.spec_from_file_location("p10_summary_test_module", SUMMARY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _executor_constants() -> tuple[str, str]:
    executor_python = os.environ.get("P9C_EXECUTOR_PYTHON")
    if not executor_python:
        pytest.skip("named skip: P9C_EXECUTOR_PYTHON is unset")
    result = subprocess.run(
        [executor_python, "-c", "from tpen_executor.runners import CHILD_OBSERVATION_FILENAME, RUN_SHIM; print(CHILD_OBSERVATION_FILENAME); print(RUN_SHIM)"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    return lines[0], "\n".join(lines[1:])


def test_template_summary_reads_product_artifacts_from_distinct_locations(tmp_path: Path) -> None:
    """The actual summary function must distinguish attempt and attempt/core."""

    filename, _ = _executor_constants()
    summary = _load_summary()
    attempt = tmp_path / "attempt"
    core = attempt / "core"
    checkpoint = tmp_path / "checkpoint"
    core.mkdir(parents=True)
    checkpoint.mkdir()
    (checkpoint / "COMPLETE").write_text("")
    (attempt / filename).write_text(json.dumps({"hostname": "node", "cuda_visible_devices": "0", "environment": {"OMP_NUM_THREADS": "4"}}))
    (core / "resolved_config.yaml").write_text("trainer:\n  max_steps: 25\nsampler:\n  n_walkers: 16\n")
    manifest = {
        "schema": "tpen-executor/run-manifest-v2",
        "attempts": [{"output_dir": str(attempt), "placement_receipt": {"comparison": "equal", "declared": {}, "child": {"host": "node", "cuda_visible_devices": "0", "affinity": []}, "gpu": {"status": "none"}}, "outcome_receipt": {"status": "completed"}}],
        "segments": [{"task_id": "task", "completed_updates": 25, "next_iteration": 25, "checkpoint_dir": str(checkpoint)}],
    }
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_path / "home")
    try:
        result = summary.summarize(manifest_path, tmp_path / "terminal-summary.json", filename)
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
    assert result["child_observation"] == str(attempt / filename)
    assert result["resolved_trainer_max_steps"] == 25


def test_template_uses_summary_function_and_product_filename() -> None:
    """The PBS template must pass the subprocess-owned filename to the real reader."""

    filename, shim = _executor_constants()
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "p10_summary.py" in text
    assert '"$CHILD_OBSERVATION_FILENAME"' in text
    assert f"Path(run_roots[-1], {filename!r})" in shim


def test_run_root_guard_precedes_all_writes_and_keeps_temp_under_root() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    guard = text.index('ROOT_CANONICAL=$(realpath -e -- "$TPEN_RUN_ROOT")')
    first_write = min(text.index("mkdir -p"), text.index('> "$ROOT/'), text.index('source "$ROOT/'))
    assert guard < first_write
    assert 'case "$ROOT_CANONICAL" in' in text
    assert '"$APPROVED_RUNS_ROOT_CANONICAL"/*)' in text
    assert 'export TMPDIR="$ROOT/tmp"' in text
