"""Exercise the executor harness's pytest/JUnit summary reconciliation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RECONCILE = REPO_ROOT / "scripts" / "cannon" / "executor-harness" / "helpers" / "reconcile_counts.py"


def _junit(totals: dict[str, int], skip_reasons: list[str]) -> str:
    cases = ["<testcase classname='tests' name='case-{index}'/>" for index in range(totals["tests"])]
    for index, reason in enumerate(skip_reasons):
        cases[index] = (
            f"<testcase classname='tests' name='case-{index}'>"
            f"<skipped message='{reason}'/></testcase>"
        )
    attributes = " ".join(f"{key}='{value}'" for key, value in totals.items())
    return f"<testsuite {attributes}>{''.join(cases)}</testsuite>\n"


@pytest.mark.parametrize(
    ("summary", "totals", "skip_reasons", "expected"),
    [
        (
            "4 passed, 1 failed, 1 error, 1 skipped, 2 xfailed, 3 xpassed",
            {"tests": 12, "failures": 1, "errors": 1, "skipped": 3},
            ["skip", "xfail-one", "xfail-two"],
            {"passed": 7, "failures": 1, "errors": 1, "skips": 3},
        ),
        (
            "2 passed, 1 failed, 1 error, 1 skipped",
            {"tests": 5, "failures": 1, "errors": 1, "skipped": 1},
            ["skip"],
            {"passed": 2, "failures": 1, "errors": 1, "skips": 1},
        ),
    ],
)
def test_reconcile_counts_folds_pytest_summary(
    tmp_path: Path,
    summary: str,
    totals: dict[str, int],
    skip_reasons: list[str],
    expected: dict[str, int],
) -> None:
    junit = tmp_path / "results.xml"
    log = tmp_path / "pytest.log"
    counts = tmp_path / "counts.txt"
    skips = tmp_path / "skips.txt"
    collection = tmp_path / "collection.json"
    junit.write_text(_junit(totals, skip_reasons), encoding="utf-8")
    log.write_text(summary + "\n", encoding="utf-8")
    collection.write_text(
        json.dumps({"collected": [f"tests::case-{index}" for index in range(totals["tests"])]}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "LOCAL_FAKE": "1",
            "DRY_RUN": "1",
            "LOCAL_FAKE_ROOT": str(tmp_path),
            "NODE_TMP": str(tmp_path),
            "DRY_NODE_TMP": str(tmp_path),
            "RUN_ROOT": str(tmp_path),
            "TASK_ROOT": str(tmp_path),
            "HARNESS_ROOT": str(tmp_path),
            "USER": "test-user",
            "SLURM_JOB_ID": "test-job",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(RECONCILE), str(junit), str(log), str(counts), str(skips), str(collection)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    observed = {
        line.partition("=")[0]: int(line.partition("=")[2])
        for line in counts.read_text(encoding="utf-8").splitlines()
        if line.partition("=")[0] in {"passed", "failures", "errors", "skips"}
    }
    assert observed == expected
