#!/usr/bin/env python3
"""Emit a fail-closed environment-variable audit table for the final harness."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


REQUIRED = {
    "TASK_ROOT", "HEAD_SHA", "UV", "NODE_TMP", "DRY_RUN", "DRY_NODE_TMP",
    "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "USER", "HOME", "PATH", "TMPDIR",
    "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE", "REVIEW_SOURCE", "REVIEW_ARM",
    "REVIEW_CAPTURE", "ROOT_PYTHON", "P10_ROOT_PYTHON", "EXECUTOR_PYTHON",
    "P8_EXECUTOR_PYTHON", "P5B_REVIEW_SCIENCE_PYTHON", "P9C_ROOT_PYTHON",
    "P9C_EXECUTOR_PYTHON", "P10_REPO_ROOT", "P10_EXECUTOR_REPO_ROOT",
    "C3_EXECUTOR_PYTHON",
}
PATTERNS = (
    re.compile(r"os\.environ(?:\.get\(|\[)\s*['\"]([A-Z][A-Z0-9_]*)"),
    re.compile(r"os\.getenv\(\s*['\"]([A-Z][A-Z0-9_]*)"),
    re.compile(r"env\[\s*['\"]([A-Z][A-Z0-9_]*)"),
    re.compile(r"\$\{?([A-Z][A-Z0-9_]*)"),
)


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: env_audit.py CHECKOUT [STAGED_ROOT]", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    staged = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else Path(os.environ["TASK_ROOT"]).resolve()

    def files_under(path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        return sorted(
            file for file in path.rglob("*")
            if file.is_file() and file.suffix in {".py", ".sbatch"}
            and "evidence" not in file.parts and "managed-python" not in file.parts
        )

    areas = {
        "staged": staged,
        "executor/tests": root / "executor" / "tests",
        "experiments/tests": root / "experiments",
        "checkout/tpen": root / "tpen",
        "checkout/executor/src": root / "executor" / "src",
    }
    print("area\tfile\tvariable\tobserved")
    found: set[str] = set()
    for area, path in areas.items():
        files = files_under(path)
        for file in files:
            try:
                text = file.read_text(encoding="utf-8")
            except UnicodeError as error:
                # AppleDouble or other transfer artefacts are not UTF-8 source;
                # report and skip them deterministically rather than aborting.
                print(f"NON_UTF8\t{area}\t{file}\t{type(error).__name__}")
                continue
            except OSError:
                continue
            for pattern in PATTERNS:
                for variable in pattern.findall(text):
                    found.add(variable)
                    print(f"{area}\t{file.relative_to(root) if file.is_relative_to(root) else file}\t{variable}\t{os.environ.get(variable, '<UNSET>')}")
    missing = sorted(REQUIRED - set(os.environ))
    print(f"SUMMARY\treferenced={len(found)}\trequired_missing={','.join(missing) or '<none>'}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
