"""Tests for the study's new minimal final report.

``final_report.py`` is a rewrite rather than a port, so nothing in the v3 test
suite covers it. The reductions it performs are where a reporting bug would be
invisible: an aggregation that averages an invariant violation away, or a sign
error that turns a broken variational estimator into the winning cell.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]

while str(STUDY_DIR) in sys.path:
    sys.path.remove(str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR))
_STUDY_TOP_LEVEL_MODULES = {
    "collect",
    "final_collect",
    "final_eval",
    "final_plan",
    "final_report",
    "final_train",
    "launch",
    "plan",
    "plot",
    "select_champions",
    "stats",
    "train",
    "utils",
    "validate",
}
for _module_name in list(sys.modules):
    if _module_name.split(".", maxsplit=1)[0] in _STUDY_TOP_LEVEL_MODULES:
        del sys.modules[_module_name]


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"tpen_pair_scan_v1_report_{name}", STUDY_DIR / f"{name}.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_script("stats")
_load_script("plot")
final_report = _load_script("final_report")

ATTEMPT = "20260813T120000-0400"
COLLECT_ATTEMPT = "20260813T110000-0400"


def _energy_row(**overrides):
    row = {
        "final_run_id": "run",
        "report_row": "no-basis",
        "report_col": "SiLU",
        "winner_kind": "energy",
        "seed_index": "0",
        "energy_mean": "2.02",
        "energy_error": "0.02",
        "local_energy_var": "0.5",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Primary metric aggregation
# ---------------------------------------------------------------------------
def test_energy_by_axis_aggregates_seed_replicates_into_one_cell():
    rows = [
        _energy_row(final_run_id="a", seed_index="0", energy_mean="2.01", energy_error="0.01"),
        _energy_row(final_run_id="b", seed_index="1", energy_mean="2.03", energy_error="0.03"),
        _energy_row(final_run_id="c", seed_index="2", energy_mean="2.05", energy_error="0.05"),
    ]
    out = final_report.energy_by_axis_rows(rows)
    assert len(out) == 1
    assert out[0]["report_row"] == "no-basis"
    assert out[0]["report_col"] == "SiLU"
    assert out[0]["n_runs"] == 3
    assert out[0]["energy_median"] == "2.03"
    assert out[0]["energy_error_median"] == "0.03"


def test_energy_by_axis_separates_distinct_report_cells():
    rows = [
        _energy_row(report_row="no-basis", report_col="SiLU", energy_mean="2.01"),
        _energy_row(report_row="no-basis", report_col="Tanh", energy_mean="2.02"),
        _energy_row(report_row="hooke-total-shell", report_col="SiLU", energy_mean="2.03"),
    ]
    out = final_report.energy_by_axis_rows(rows)
    assert [(row["report_row"], row["report_col"]) for row in out] == [
        ("hooke-total-shell", "SiLU"),
        ("no-basis", "SiLU"),
        ("no-basis", "Tanh"),
    ]
    assert [row["n_runs"] for row in out] == [1, 1, 1]


def test_energy_below_the_variational_bound_is_counted_not_rewarded():
    # A variational estimator cannot go below 2.0 Ha. When it does, that is an
    # estimator defect, so the count must be surfaced rather than absorbed into
    # a median that would rank the broken cell best.
    rows = [
        _energy_row(final_run_id="a", energy_mean="1.80", energy_error="-0.20"),
        _energy_row(final_run_id="b", energy_mean="2.10", energy_error="0.10"),
        _energy_row(final_run_id="c", energy_mean="1.99", energy_error="-0.01"),
    ]
    out = final_report.energy_by_axis_rows(rows)
    assert out[0]["n_below_exact"] == 2


def test_no_run_is_counted_below_a_bound_it_sits_above():
    rows = [
        _energy_row(final_run_id="a", energy_mean="2.00"),
        _energy_row(final_run_id="b", energy_mean="2.50"),
    ]
    assert final_report.energy_by_axis_rows(rows)[0]["n_below_exact"] == 0


def test_energy_by_axis_ignores_unparseable_energies_without_dropping_the_row():
    rows = [
        _energy_row(final_run_id="a", energy_mean=""),
        _energy_row(final_run_id="b", energy_mean="2.04", energy_error="0.04"),
    ]
    out = final_report.energy_by_axis_rows(rows)
    assert out[0]["n_runs"] == 2
    assert out[0]["energy_median"] == "2.04"
    assert out[0]["n_below_exact"] == 0


def test_the_exact_hooke_energy_is_the_taut_value():
    assert final_report.EXACT_HOOKE_ENERGY == 2.0
    assert final_report.PRIMARY_METRIC == "eval/mcmc_energy/local_energy_mean"
    assert final_report.SECONDARY_METRIC == "eval/stratified_geometry/local_energy_variance"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
def _symmetry_row(**overrides):
    row = {
        "report_row": "no-basis",
        "report_col": "SiLU",
        "symmetry_task": "full_model_antisymmetry",
        "logabs_error_max": "1e-12",
        "sign_mismatch_count": "0",
    }
    row.update(overrides)
    return row


def _trace_row(**overrides):
    row = {
        "report_row": "no-basis",
        "report_col": "SiLU",
        "trace_kind": "trace_equivariance",
        "max_equivariance_error": "1e-12",
        "comparison_error_count": "0",
    }
    row.update(overrides)
    return row


def test_invariant_violations_aggregate_by_worst_case_not_by_average():
    # One violated seed among many clean ones is a violation of the model class.
    # Averaging would report 1e-1 as roughly 2e-2 and hide it.
    rows = [
        _symmetry_row(logabs_error_max="1e-12"),
        _symmetry_row(logabs_error_max="1e-12"),
        _symmetry_row(logabs_error_max="1e-12"),
        _symmetry_row(logabs_error_max="1e-12"),
        _symmetry_row(logabs_error_max="0.1"),
    ]
    out = final_report.invariants_by_axis_rows(rows, [])
    assert out[0]["antisymmetry_logabs_error_max"] == "0.1"


def test_trace_equivariance_error_also_aggregates_by_worst_case():
    rows = [_trace_row(max_equivariance_error="1e-9"), _trace_row(max_equivariance_error="3e-4")]
    out = final_report.invariants_by_axis_rows([], rows)
    assert out[0]["trace_equivariance_error_max"] == "0.0003"


def test_mismatch_counts_are_summed_across_seeds_not_maximized():
    rows = [
        _symmetry_row(sign_mismatch_count="2"),
        _symmetry_row(sign_mismatch_count="3"),
    ]
    out = final_report.invariants_by_axis_rows(rows, [])
    assert out[0]["antisymmetry_sign_mismatch_total"] == 5


def test_invariants_cover_cells_present_in_only_one_of_the_two_tables():
    symmetry = [_symmetry_row(report_col="SiLU")]
    trace = [_trace_row(report_col="Tanh")]
    out = final_report.invariants_by_axis_rows(symmetry, trace)
    assert [(row["report_row"], row["report_col"]) for row in out] == [
        ("no-basis", "SiLU"),
        ("no-basis", "Tanh"),
    ]
    assert out[0]["trace_equivariance_error_max"] == ""
    assert out[1]["antisymmetry_logabs_error_max"] == ""


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------
def test_failures_by_axis_names_the_dominant_task_scoped_mode():
    rows = [
        {"report_row": "b", "report_col": "c", "task": "cusp", "failure_mode": "nonfinite"},
        {"report_row": "b", "report_col": "c", "task": "cusp", "failure_mode": "nonfinite"},
        {"report_row": "b", "report_col": "c", "task": "tail", "failure_mode": "outlier"},
    ]
    out = final_report.failures_by_axis_rows(rows)
    assert out[0]["n_failure_rows"] == 3
    assert out[0]["dominant_failure_mode"] == "cusp:nonfinite"
    assert out[0]["dominant_failure_count"] == 2


def test_failures_by_axis_labels_an_unnamed_mode_rather_than_dropping_it():
    rows = [{"report_row": "b", "report_col": "c", "task": "", "failure_mode": ""}]
    out = final_report.failures_by_axis_rows(rows)
    assert out[0]["n_failure_rows"] == 1
    assert out[0]["dominant_failure_mode"] == "unlabelled"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _markdown(**overrides):
    kwargs = {
        "study": "tpen_pair_scan_v1",
        "attempt_id": ATTEMPT,
        "collect_attempt_id": COLLECT_ATTEMPT,
        "row_label": "basis",
        "col_label": "activation",
        "energy_axis_rows": final_report.energy_by_axis_rows([_energy_row()]),
        "invariant_rows": final_report.invariants_by_axis_rows([_symmetry_row()], [_trace_row()]),
        "failure_axis_rows": [],
        "cost_rows": [],
        "run_index_rows": [{"final_run_id": "run"}],
        "warnings": [],
    }
    kwargs.update(overrides)
    return final_report.report_markdown(**kwargs)


def test_report_markdown_records_its_provenance_and_axes():
    text = _markdown()
    assert f"`{ATTEMPT}`" in text
    assert f"`{COLLECT_ATTEMPT}`" in text
    assert "rows = `basis`" in text
    assert "columns = `activation`" in text
    assert final_report.PRIMARY_METRIC in text


def test_report_markdown_escalates_a_sub_bound_energy_to_a_prose_warning():
    # The count is in the table either way; the report must also say it in prose,
    # because a reader scanning for the lowest median would otherwise pick the
    # invalid cell.
    rows = final_report.energy_by_axis_rows([_energy_row(energy_mean="1.5", energy_error="-0.5")])
    text = _markdown(energy_axis_rows=rows)
    assert "variational bound" in text
    assert "invalid" in text


def test_report_markdown_stays_quiet_when_every_energy_respects_the_bound():
    text = _markdown()
    assert "variational bound" not in text


def test_report_markdown_surfaces_collection_warnings():
    text = _markdown(warnings=["missing source table: energy_by_run.csv"])
    assert "## Warnings" in text
    assert "missing source table: energy_by_run.csv" in text


def test_report_markdown_marks_an_empty_section_rather_than_omitting_it():
    text = _markdown(energy_axis_rows=[], invariant_rows=[])
    assert text.count("_no rows_") >= 2
    assert "## Invariants" in text


# ---------------------------------------------------------------------------
# End to end over a synthetic collect attempt
# ---------------------------------------------------------------------------
def _write_collect_attempt(results_root: Path) -> Path:
    from experiments.toolkit.artifacts import write_csv

    attempt = results_root / "08_final_collect" / COLLECT_ATTEMPT
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "manifest.yaml").write_text(
        "study: tpen_pair_scan_v1\n"
        f"attempt_id: {COLLECT_ATTEMPT}\n"
        "report_row_key: basis\n"
        "report_col_key: activation\n"
    )
    write_csv(attempt / "run_index.csv", [{"final_run_id": "run-a"}, {"final_run_id": "run-b"}])
    write_csv(
        attempt / "energy_by_run.csv",
        [
            _energy_row(final_run_id="run-a", seed_index="0"),
            _energy_row(final_run_id="run-b", seed_index="1", energy_mean="2.06", energy_error="0.06"),
        ],
    )
    write_csv(attempt / "symmetry_summary.csv", [_symmetry_row()])
    write_csv(attempt / "trace_summary.csv", [_trace_row()])
    write_csv(
        attempt / "training_curve_summary.csv",
        [
            {"report_row": "no-basis", "report_col": "SiLU", "step": "0", "energy_mean": "3.0"},
            {"report_row": "no-basis", "report_col": "SiLU", "step": "10", "energy_mean": "2.1"},
        ],
    )
    write_csv(attempt / "failure_modes.csv", [])
    write_csv(attempt / "architecture_summary.csv", [])
    write_csv(attempt / "cost_by_axis.csv", [{"report_row": "no-basis", "wall_time_sec": "10"}])
    (results_root / "08_final_collect" / "latest.json").write_text(
        f'{{"attempt_id": "{COLLECT_ATTEMPT}", "path": "{COLLECT_ATTEMPT}"}}\n'
    )
    return attempt


def test_build_report_writes_tables_figures_and_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    results_root = tmp_path / "results"
    _write_collect_attempt(results_root)
    record = final_report.build_report(results_root=results_root, attempt_id=ATTEMPT)

    attempt = results_root / "09_final_report" / ATTEMPT
    assert record["attempt_id"] == ATTEMPT
    assert record["final_collect_attempt_id"] == COLLECT_ATTEMPT
    assert record["report_row_key"] == "basis"
    assert record["report_col_key"] == "activation"
    assert record["n_final_runs"] == 2
    assert record["warnings"] == []
    for name in final_report.REPORT_TABLES:
        assert (attempt / "tables" / name).is_file(), name
    for name in final_report.REPORT_FIGURES:
        figure = attempt / "figures" / name
        assert figure.is_file(), name
        assert figure.stat().st_size > 0, name
    assert (attempt / "report.md").is_file()
    assert (attempt / "final_report.json").is_file()
    # The stage's latest pointer makes the attempt discoverable downstream.
    assert (results_root / "09_final_report" / "latest.json").is_file()


def test_build_report_defaults_to_the_latest_collect_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    results_root = tmp_path / "results"
    _write_collect_attempt(results_root)
    assert final_report.resolve_final_collect_attempt_id(results_root) == COLLECT_ATTEMPT


def test_build_report_refuses_a_results_root_with_no_collect_attempt(tmp_path):
    with pytest.raises(FileNotFoundError, match="08_final_collect"):
        final_report.build_report(results_root=tmp_path / "results", attempt_id=ATTEMPT)


def test_build_report_records_a_warning_for_a_missing_source_table(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    results_root = tmp_path / "results"
    attempt = _write_collect_attempt(results_root)
    (attempt / "symmetry_summary.csv").unlink()
    record = final_report.build_report(results_root=results_root, attempt_id=ATTEMPT)
    assert any("symmetry_summary.csv" in warning for warning in record["warnings"])
    # A partially collected lineage still renders, so triage is possible.
    assert (results_root / "09_final_report" / ATTEMPT / "report.md").is_file()
