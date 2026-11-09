"""Tests for the optional training-batch electron--nucleus distance callback."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from tpen.callback import ElectronNucleusDistanceQuantiles
from tpen.callback.electron_nucleus_distance import summarize_electron_nucleus_distances
from tpen.data.batch import ElectronBatch
from tpen.logging import CSV, JSONL
from tpen.training.events import TrainingIteration, TrainingIterationCompleted

from .support import RecordingContext, deliver_completed_iteration, training_state
from tests.helpers.run_context import make_run_context


def _batch(positions: list[list[list[float]]], nuclei: list[list[float]]) -> ElectronBatch:
    """Build a deterministic fixed-geometry electron batch."""

    nuclear_positions = (
        torch.empty((0, 3), dtype=torch.float64)
        if not nuclei
        else torch.tensor(nuclei, dtype=torch.float64)
    )
    return ElectronBatch(
        positions=torch.tensor(positions, dtype=torch.float64),
        nuclear_positions=nuclear_positions,
        nuclear_charges=torch.ones(len(nuclei), dtype=torch.float64),
    )


def test_summaries_match_independent_numpy_for_several_nuclei() -> None:
    batch = _batch(
        [
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[5.0, 0.0, 0.0], [0.0, 0.0, 6.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
    )
    levels = (0.25, 0.5, 0.75)
    actual = summarize_electron_nucleus_distances(batch, quantiles=levels)

    positions = np.asarray(batch.positions).reshape(-1, 2, 3)
    nuclei = np.asarray(batch.nuclear_positions)
    nearest = np.linalg.norm(positions[:, :, None, :] - nuclei[None, None, :, :], axis=-1).min(axis=-1)
    expected: dict[str, float] = {}
    for electron in range(2):
        prefix = f"electron_{electron}_nucleus_distance"
        expected[f"{prefix}_min"] = float(nearest[:, electron].min())
        for level, value in zip(levels, np.quantile(nearest[:, electron], levels, method="linear"), strict=True):
            expected[f"{prefix}_q{100 * level:g}".replace(".", "p")] = float(value)
    assert actual == pytest.approx(expected)


def test_callback_summarizes_the_batch_the_iteration_trained_on() -> None:
    other_batch = _batch([[[100.0, 0.0, 0.0]]], [[0.0, 0.0, 0.0]])
    trained_batch = _batch(
        [
            [[1.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0]],
        ],
        [[0.0, 0.0, 0.0]],
    )
    callback = ElectronNucleusDistanceQuantiles(quantiles=(0.5,))
    context = RecordingContext()
    state = training_state(batch=trained_batch, samples=other_batch)

    deliver_completed_iteration(callback, context, state, step=7)

    assert context.latest("train/electron_nucleus_distance") == {
        "electron_0_nucleus_distance_min": 1.0,
        "electron_0_nucleus_distance_q50": 3.0,
    }


@pytest.mark.parametrize("quantiles", [None, (), (-0.1,), (1.1,), (0.5, 0.5)])
def test_quantile_levels_are_required_and_validated(quantiles) -> None:
    with pytest.raises(ValueError):
        ElectronNucleusDistanceQuantiles(quantiles=quantiles)


def test_zero_nuclei_are_refused() -> None:
    batch = _batch([[[1.0, 0.0, 0.0]]], [])
    with pytest.raises(ValueError, match="at least one nucleus"):
        summarize_electron_nucleus_distances(batch, quantiles=(0.5,))


def test_real_csv_and_jsonl_loggers_preserve_cadence_and_existing_metrics(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    jsonl_path = tmp_path / "metrics.jsonl"
    callback = ElectronNucleusDistanceQuantiles(quantiles=(0.5,), every_n_steps=2)
    context = make_run_context(
        tmp_path,
        callbacks=[callback],
        loggers=[CSV(csv_path), JSONL(jsonl_path)],
    )
    state = training_state(batch=_batch([[[1.0, 0.0, 0.0]]], [[0.0, 0.0, 0.0]]))

    for step in range(1, 6):
        context.log({"existing_metric": float(step)}, step=step, namespace="train")
        context.emit(
            TrainingIterationCompleted(iteration=TrainingIteration(step=step)),
            state=state,
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows if row["key"] == "existing_metric"] == [1, 2, 3, 4, 5]
    assert [int(row["step"]) for row in rows if row["key"] == "electron_0_nucleus_distance_q50"] == [2, 4]
    assert {row["key"] for row in rows} == {"existing_metric", "electron_0_nucleus_distance_min", "electron_0_nucleus_distance_q50"}

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [record["step"] for record in records if "existing_metric" in record["metrics"]] == [1, 2, 3, 4, 5]
    assert [record["step"] for record in records if "electron_0_nucleus_distance_q50" in record["metrics"]] == [2, 4]


def test_distance_metric_is_absent_when_callback_is_not_configured(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    jsonl_path = tmp_path / "metrics.jsonl"
    context = make_run_context(tmp_path, loggers=[CSV(csv_path), JSONL(jsonl_path)])
    state = training_state(batch=_batch([[[1.0, 0.0, 0.0]]], [[0.0, 0.0, 0.0]]))

    for step in range(1, 4):
        context.log({"existing_metric": float(step)}, step=step, namespace="train")
        context.emit(
            TrainingIterationCompleted(iteration=TrainingIteration(step=step)),
            state=state,
        )

    assert "electron_nucleus_distance" not in csv_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert all(not any(key.startswith("electron_") for key in record["metrics"]) for record in records)

