#!/usr/bin/env python3
"""Rehearse the literal sbatch export line through an env -i DRY_RUN."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED = {
    "TASK_ROOT", "RUN_ROOT", "HEAD_SHA", "JOB_NAME", "MUTANT_SPEC", "DRY_RUN", "DRY_NODE_TMP",
    "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "USER", "HOME", "PATH",
    "PYTHONPATH", "ROOT_PYTHON", "EXECUTOR_PYTHON",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: rehearse_submit_line.py STAGE", file=sys.stderr)
        return 2
    stage = Path(sys.argv[1]).resolve()
    line_path = stage / "submission-line.txt"
    lines = [line.rstrip("\n") for line in line_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        print("literal_line_count=FAIL", file=sys.stderr)
        return 1
    line = lines[0]
    argv = shlex.split(line)
    if len(argv) < 3 or argv[0] != "sbatch" or Path(argv[-1]).name != "job.sbatch":
        print("literal_line_shape=FAIL", file=sys.stderr)
        return 1
    export_arg = next((arg for arg in argv[1:-1] if arg.startswith("--export=NONE,")), None)
    if export_arg is None:
        print("literal_line_export=FAIL", file=sys.stderr)
        return 1
    values: dict[str, str] = {}
    for entry in export_arg.removeprefix("--export=NONE,").split(","):
        key, sep, value = entry.partition("=")
        if not sep or not key or key in values:
            print("literal_line_bindings=FAIL", file=sys.stderr)
            return 1
        values[key] = value
    missing = sorted((REQUIRED - values.keys()) - {"SLURM_JOB_ID", "SLURM_JOB_PARTITION"})
    if missing:
        print(f"literal_line_missing={','.join(missing) or '<none>'}", file=sys.stderr)
        print(f"literal_line_task_root={values.get('TASK_ROOT', '<missing>')}", file=sys.stderr)
        print(f"literal_line_dry_run={values.get('DRY_RUN', '<missing>')}", file=sys.stderr)
        return 1
    if values.get("PATH") != "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin":
        print("literal_line_path=FAIL", file=sys.stderr)
        return 1
    if values.get("PYTHONPATH") != "":
        print("literal_line_pythonpath=FAIL", file=sys.stderr)
        return 1
    if not values.get("HEAD_SHA", "").isalnum() or len(values["HEAD_SHA"]) != 40:
        print("literal_line_head_sha=FAIL", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="p10-submit-rehearsal-") as temporary:
        rehearsal = Path(temporary) / "stage"
        shutil.copytree(stage, rehearsal)
        child_env = {key: value for key, value in values.items()}
        child_env.setdefault("SLURM_JOB_ID", "REHEARSAL")
        child_env.setdefault("SLURM_JOB_PARTITION", "test")
        child_env["TASK_ROOT"] = str(rehearsal)
        child_env["RUN_ROOT"] = str(rehearsal / "run-root")
        child_env["DRY_NODE_TMP"] = str(rehearsal / "dry-node-tmp")
        child_env["DRY_RUN"] = "1"
        child_env["LOCAL_FAKE"] = "1"
        child_env["LOCAL_FAKE_ROOT"] = str(rehearsal)
        child_env["MUTANT_SPEC"] = str(rehearsal / "mutants.json")
        child_env["ROOT_PYTHON"] = sys.executable
        child_env["EXECUTOR_PYTHON"] = sys.executable
        child_env["SLURM_JOB_NAME"] = child_env["JOB_NAME"]
        literal_path = child_env["PATH"]
        if shutil.which("mkdir", path=literal_path) is None:
            child_env["PATH"] = f"/bin:{literal_path}"
        result = subprocess.run(
            ["env", "-i", *[f"{key}={value}" for key, value in child_env.items()], "/bin/bash", "job.sbatch"],
            cwd=rehearsal, env={"PATH": os.environ["PATH"]}, text=True, capture_output=True,
        )
        log = result.stdout + result.stderr
    (stage / "evidence").mkdir(parents=True, exist_ok=True)
    (stage / "evidence" / "literal-submit-rehearsal.log").write_text(log, encoding="utf-8")
    print(f"literal_line={line}")
    print(f"parsed_var_count={len(values)}")
    print(f"literal_path={values['PATH']}")
    print(f"rehearsal_path={child_env['PATH']}")
    print(f"env_i_dry_run_rc={result.returncode}")
    print("rehearsal_task_root=temporary-copy")
    print("rehearsal_dry_run=1")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
