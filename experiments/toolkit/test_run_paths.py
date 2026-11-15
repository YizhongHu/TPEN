"""Boundary tests for manifest run-path resolution."""

import pytest

from experiments.toolkit.run_paths import RunPathError, resolve_run_dir


def test_resolve_run_dir_reports_typed_error_for_missing_binding() -> None:
    with pytest.raises(RunPathError, match="required to resolve run directory"):
        resolve_run_dir({"results_root": "/tmp/results"})
