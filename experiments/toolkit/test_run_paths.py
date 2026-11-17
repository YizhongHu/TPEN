"""Boundary tests for manifest run-path resolution."""

import pytest

from experiments.toolkit.run_paths import RunPathError, resolve_run_dir


def test_resolve_run_dir_reports_typed_error_for_missing_binding() -> None:
    with pytest.raises(RunPathError, match="required to resolve run directory"):
        resolve_run_dir({"results_root": "/tmp/results"})


@pytest.mark.parametrize("field", ["run_dir", "run_layout"])
def test_resolve_run_dir_rejects_row_path_options(field: str) -> None:
    row = {
        "results_root": "/tmp/results",
        "experiment_name": "stage1",
        "sector": "he",
        "run_id": "run-0",
        field: "flat" if field == "run_layout" else "/tmp/elsewhere",
    }
    with pytest.raises(RunPathError, match="run_dir/run_layout"):
        resolve_run_dir(row)
