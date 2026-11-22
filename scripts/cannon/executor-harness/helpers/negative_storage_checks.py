"""Negative tests for the shell and Python storage-root gates."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from path_guard import allowed_roots


ROOT = "/n/netscratch/kozinsky_lab/Lab/rhu/verify/test-job"
HELPER = Path(__file__).with_name("validate_inputs.sh")
BASE = {
    "USER": "rhu",
    "SLURM_JOB_ID": "12345",
    "DRY_RUN": "0",
    "NODE_TMP": "/scratch/rhu/12345",
    "DRY_NODE_TMP": "/scratch/rhu/12345",
    "HARNESS_ROOT": ROOT,
    "RUN_ROOT": ROOT,
    "TASK_ROOT": ROOT,
    "LOCAL_FAKE": "0",
    "LOCAL_FAKE_ROOT": "",
}


def shell_gate(environment: dict[str, str]) -> bool:
    command = f"source {HELPER!s}; validate_inputs"
    return subprocess.run(["bash", "-c", command], env=environment, check=False, capture_output=True).returncode == 0


def python_gate(environment: dict[str, str]) -> bool:
    old = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(environment)
        allowed_roots()
        return True
    except (RuntimeError, ValueError):
        return False
    finally:
        os.environ.clear()
        os.environ.update(old)


def main() -> int:
    cases = {
        "NODE_TMP=/tmp/x": {"NODE_TMP": "/tmp/x"},
        "RUN_ROOT=/n/netscratch/kozinsky_lab/Lab/x": {"RUN_ROOT": "/n/netscratch/kozinsky_lab/Lab/x"},
        "HARNESS_ROOT=/var/x": {"HARNESS_ROOT": "/var/x"},
    }
    failures: list[str] = []
    for label, changes in cases.items():
        environment = {**os.environ, **BASE, **changes}
        shell_ok = shell_gate(environment)
        python_ok = python_gate(environment)
        print(f"{label}\tvalidate_inputs={'PASS' if shell_ok else 'FAIL'}\tallowed_roots={'PASS' if python_ok else 'FAIL'}")
        if shell_ok or python_ok:
            failures.append(label)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
