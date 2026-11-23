"""Synthetic-fixture tests for Stage-1 descriptive analysis."""

from pathlib import Path

import pytest

from experiments.atomistic.stage1.analysis import (
    AnalysisError,
    join_training_history,
    summarize_collection,
    write_trajectory_figures,
)


def _collection(*, checkpoints=(0, 1), lineages=(0, 1)) -> dict:
    records = []
    for lineage in lineages:
        for checkpoint in checkpoints:
            for stream in range(4):
                mean = float(stream + 1 + lineage + checkpoint)
                records.append({
                    "system_id": "synthetic",
                    "lineage": lineage,
                    "checkpoint": checkpoint,
                    "stream": stream,
                    "statistics": {"mean": mean, "variance": float(stream + 1)},
                    "reference": {"energy_hartree": 0.0},
                })
    return {"schema": "stage1-collection/v1", "manifest_hash": "synthetic", "records": records}


def test_summary_uses_between_stream_term_and_reports_ranges() -> None:
    analysis = summarize_collection(_collection(lineages=(0,), checkpoints=(0,)))
    group = analysis["groups"][0]

    assert group["mean"] == pytest.approx(2.5)
    assert group["signed_error"] == pytest.approx(2.5)
    assert group["stream_error_range"] == pytest.approx(3.0)
    assert group["stream_variance_range"] == pytest.approx(3.0)
    # mean([1, 2, 3, 4]) + population variance([1, 2, 3, 4])
    assert group["pooled_variance"] == pytest.approx(3.75)


def test_checkpoint_aggregate_keeps_lineage_ranges_separate() -> None:
    analysis = summarize_collection(_collection(lineages=(0, 1), checkpoints=(10,)))
    checkpoint = analysis["checkpoints"][0]

    assert checkpoint["lineage_count"] == 2
    assert checkpoint["signed_error_min"] == pytest.approx(12.5)
    assert checkpoint["signed_error_max"] == pytest.approx(13.5)
    assert checkpoint["signed_error_range"] == pytest.approx(1.0)


def test_incomplete_or_duplicate_streams_are_not_silently_imputed() -> None:
    document = _collection(lineages=(0,), checkpoints=(0,))
    document["records"] = document["records"][:-1]
    with pytest.raises(AnalysisError, match="exactly streams"):
        summarize_collection(document)

    document = _collection(lineages=(0,), checkpoints=(0,))
    document["records"][-1]["stream"] = 2
    with pytest.raises(AnalysisError, match="duplicate stream"):
        summarize_collection(document)


def _training_rows(lineage: int = 0, count: int = 100) -> list[dict]:
    return [{
        "system_id": "synthetic", "lineage": lineage, "step": step,
        "namespace": "train",
        "metrics": {"energy": 10.0 + step, "energy_variance": 2.0 + step,
                    "optimizer_step": step != 1},
    } for step in range(count)]


def test_training_join_maps_zero_based_step_and_preserves_skips() -> None:
    analysis = summarize_collection(_collection(lineages=(0,), checkpoints=(0, 1, 100)))
    joined = join_training_history(analysis, _training_rows())
    comparisons = {(row["checkpoint"]): row for row in joined["comparisons"]}

    assert comparisons[0]["available"] is False
    assert comparisons[0]["reason"] == "step zero has no trailing pre-update window"
    assert comparisons[1]["available"] is True
    assert comparisons[1]["completed_steps"] == [1]
    assert comparisons[1]["stored_steps"] == [0]
    assert comparisons[1]["skipped_updates"] == 0
    assert comparisons[100]["available"] is True
    assert comparisons[100]["completed_steps"] == [1, 2, 3, 4, 5, *range(6, 101)]
    assert comparisons[100]["skipped_updates"] == 1
    assert comparisons[100]["training_mean"] == pytest.approx(59.5)
    assert comparisons[100]["training_variance"] == pytest.approx(51.5)


def test_training_join_marks_missing_window_unavailable() -> None:
    analysis = summarize_collection(_collection(lineages=(0,), checkpoints=(2,)))
    joined = join_training_history(analysis, _training_rows(count=1))
    comparison = joined["comparisons"][0]

    assert comparison["available"] is False
    assert comparison["missing_completed_steps"] == [2]


def test_figures_are_written_to_caller_directory(tmp_path: Path) -> None:
    analysis = summarize_collection(_collection(lineages=(0, 1), checkpoints=(0, 1)))
    paths = write_trajectory_figures(analysis, tmp_path / "figures")

    assert set(paths) == {"error_vs_step", "variance_vs_step"}
    assert all(Path(path).is_file() for path in paths.values())
    assert all(Path(path).stat().st_size > 0 for path in paths.values())
