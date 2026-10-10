from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET = Path(__file__).with_name("test_outcome_selection.py")


def test_single_file_outcome_selection_keeps_content_error_identity() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(TARGET), "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
