"""Synthetic reviewer probes for the frozen Stage-1 analysis addendum."""

from pathlib import Path

import pytest

from experiments.atomistic.stage1.analysis import summarize_collection, write_trajectory_figures


def _collection(means: list[float], variances: list[float], reference: float = 0.0) -> dict:
    return {
        "schema": "stage1-collection/v1",
        "manifest_hash": "synthetic-review-r1",
        "records": [
            {
                "system_id": "synthetic",
                "lineage": 0,
                "checkpoint": 7,
                "stream": stream,
                "statistics": {"mean": mean, "variance": variance},
                "reference": {"energy_hartree": reference},
            }
            for stream, (mean, variance) in enumerate(zip(means, variances))
        ],
    }


def test_signed_error_is_mean_of_four_stream_means_minus_reference() -> None:
    # Errors are asymmetric and the nonzero reference prevents a zero-reference
    # shortcut from passing: means [8, 11, 14, 15], reference 10 -> 2.0.
    analysis = summarize_collection(_collection([8.0, 11.0, 14.0, 15.0], [1.0] * 4, reference=10.0))

    group = analysis["groups"][0]
    assert group["mean"] == pytest.approx(12.0)
    assert group["signed_error"] == pytest.approx(2.0)
    assert [row["error"] for row in group["streams"]] == [-2.0, 1.0, 4.0, 5.0]


def test_pooled_variance_is_population_with_between_stream_term() -> None:
    # Within-stream population variance is 2.5; between-stream population
    # variance is 5.0, so the required pooled value is 7.5.  This rejects both
    # the within-only mutant (2.5) and ddof=1/sample or SE substitutions.
    analysis = summarize_collection(_collection([0.0, 2.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0], reference=1.0))

    assert analysis["groups"][0]["pooled_variance"] == pytest.approx(7.5)


def test_figures_have_signed_absolute_variance_panels_and_all_lineage_spreads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Semantic red on the current PR: it lacks the required absolute-error panel."""
    import matplotlib.pyplot as plt

    calls: list[dict[str, object]] = []

    class Line:
        def get_color(self) -> str:
            return "synthetic-color"

    class Axis:
        def plot(self, steps, values, **kwargs):
            calls.append({"kind": "plot", "steps": list(steps), "values": list(values), "label": kwargs["label"]})
            return (Line(),)

        def fill_between(self, steps, lows, highs, **kwargs):
            calls.append({"kind": "spread", "steps": list(steps), "lows": list(lows), "highs": list(highs)})

        def set_xlabel(self, _value):
            pass

        def set_ylabel(self, _value):
            pass

        def grid(self, **_kwargs):
            pass

        def legend(self, **_kwargs):
            pass

    class Figure:
        def tight_layout(self):
            pass

        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"synthetic figure")

    monkeypatch.setattr(plt, "subplots", lambda **_kwargs: (Figure(), Axis()))
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    document = _collection([8.0, 11.0, 14.0, 15.0], [1.0] * 4, reference=10.0)
    document["records"] += [
        {**row, "lineage": 1, "statistics": {**row["statistics"], "mean": row["statistics"]["mean"] + 1.0}}
        for row in list(document["records"])
    ]
    analysis = summarize_collection(document)
    paths = write_trajectory_figures(analysis, tmp_path / "figures")

    # Current PR returns only error_vs_step and variance_vs_step: expected
    # semantic red, not HARNESS_INVALID, because all inputs are synthetic and
    # the plotting spy is a complete deterministic test double.
    assert set(paths) == {"signed_error_vs_step", "absolute_error_vs_step", "variance_vs_step"}
    assert len([call for call in calls if call["kind"] == "plot"]) == 6
    assert len([call for call in calls if call["kind"] == "spread"]) == 6
    assert all(Path(path).is_file() for path in paths.values())
