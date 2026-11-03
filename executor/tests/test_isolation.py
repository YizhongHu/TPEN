"""Verify importing the executor does not load the scientific core."""

import subprocess
import sys
from pathlib import Path


def test_import_isolated_from_core_and_torch() -> None:
    """A clean interpreter keeps ``tpen`` and ``torch`` unloaded."""
    src_path = Path(__file__).parents[1] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(src_path)!r}); import tpen_executor; "
            "assert 'tpen' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
