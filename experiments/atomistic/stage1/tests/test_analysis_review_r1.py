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
    """Check each panel's plotted stream and aggregate values, not just files."""
    import matplotlib.pyplot as plt

    calls: list[dict[str, object]] = []
    axes: list[object] = []

    class Line:
        def get_color(self) -> str:
            return "synthetic-color"

    class Axis:
        def __init__(self, panel: int):
            self.panel = panel

        def plot(self, steps, values, **kwargs):
            calls.append({
                "kind": "plot",
                "panel": self.panel,
                "steps": list(steps),
                "values": [tuple(point) for point in values],
                "label": kwargs["label"],
            })
            return (Line(),)

        def fill_between(self, steps, lows, highs, **kwargs):
            calls.append({
                "kind": "spread",
                "panel": self.panel,
                "steps": list(steps),
                "lows": list(lows),
                "highs": list(highs),
            })

        def set_xlabel(self, _value):
            pass

        def set_ylabel(self, _value):
            pass

        def grid(self, *_args, **_kwargs):
            pass

        def legend(self, **_kwargs):
            pass

    class Figure:
        def tight_layout(self):
            pass

        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"synthetic figure")

    def subplots(**_kwargs):
        axis = Axis(len(axes))
        axes.append(axis)
        return Figure(), axis

    monkeypatch.setattr(plt, "subplots", subplots)
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    # The first lineage has both negative and positive stream errors.  Its
    # signed aggregate is -0.25 while its absolute aggregate is 0.25, so a
    # signed/absolute panel swap cannot pass by coincidence.
    document = _collection([5.0, 9.0, 12.0, 13.0], [1.0] * 4, reference=10.0)
    document["records"] += [
        {**row, "lineage": 1, "statistics": {**row["statistics"], "mean": row["statistics"]["mean"] + 1.0}}
        for row in list(document["records"])
    ]
    analysis = summarize_collection(document)
    paths = write_trajectory_figures(analysis, tmp_path / "figures")

    assert set(paths) == {"error_vs_step", "absolute_error_vs_step", "variance_vs_step"}
    plot_calls = [call for call in calls if call["kind"] == "plot"]
    assert len(plot_calls) == 6
    assert len(axes) == 3

    signed = [call for call in plot_calls if call["panel"] == 0]
    absolute = [call for call in plot_calls if call["panel"] == 1]
    variance = [call for call in plot_calls if call["panel"] == 2]
    assert [call["values"][0] for call in signed] == [
        (-5.0, -1.0, 2.0, 3.0, -0.25),
        (-4.0, 0.0, 3.0, 4.0, 0.75),
    ]
    assert [call["values"][0] for call in absolute] == [
        (5.0, 1.0, 2.0, 3.0, 0.25),
        (4.0, 0.0, 3.0, 4.0, 0.75),
    ]
    # Every lineage contributes all four stream values plus its aggregate to
    # the variance panel as well; a three-stream plot has only four columns.
    assert [call["values"][0] for call in variance] == [
        (1.0, 1.0, 1.0, 1.0, 10.6875),
        (1.0, 1.0, 1.0, 1.0, 10.6875),
    ]
    assert [call["values"][0] for call in signed] != [call["values"][0] for call in absolute]
    assert len([call for call in calls if call["kind"] == "spread"]) == 6
    assert all(Path(path).is_file() for path in paths.values())
