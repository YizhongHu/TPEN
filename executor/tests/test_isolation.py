"""Verify importing the executor does not load the scientific core."""

import subprocess
import sys


def test_import_isolated_from_core_and_torch() -> None:
    """A clean interpreter keeps ``tpen`` and ``torch`` unloaded."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); import tpen_executor; "
            "assert 'tpen' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
