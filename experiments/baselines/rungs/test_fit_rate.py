"""Coverage for the legacy timestamped-rung rate estimator."""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def _load_fit_rate(root: Path, monkeypatch: pytest.MonkeyPatch):
    """Load the CLI script after supplying its required probe-root argument."""
    source = Path(os.environ.get("FIT_RATE_PATH", HERE / "fit_rate.py"))
    spec = importlib.util.spec_from_file_location("fit_rate_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "argv", [str(source), str(root)])
    # The CLI summary is incidental to these unit tests; retain its execution
    # while keeping pytest output focused on the function contracts.
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def test_parse_reads_timestamped_steps_and_rejects_malformed_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "a1" / "train.log"
    log.parent.mkdir()
    log.write_text(
        "\n".join(
            [
                "2026-01-01T00:00:00.000Z\tINFO Step 50: start",
                "not a tab-separated record",
                "2026-01-01T00:00:01.000Z\tINFO Step 75 without colon",
                "not-a-timestamp\tINFO Step 90: bad time",
                "2026-01-01T00:00:02.000Z\tINFO Step 100: finish",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mod = _load_fit_rate(tmp_path, monkeypatch)
    points = mod.parse("a1")

    assert points is not None
    assert [step for step, _ in points] == [50, 100]
    assert points[1][1] - points[0][1] == pytest.approx(2.0)
    assert mod.parse("missing") is None


def test_fit_returns_known_least_squares_seconds_per_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_fit_rate(tmp_path, monkeypatch)
    points = [(0, 10.0), (2, 17.0), (4, 24.0), (6, 31.0)]

    assert mod.fit(points, cut=2) == pytest.approx(3.5)


@pytest.mark.parametrize(
    "points",
    [
        [(7, 1.0), (7, 2.0)],  # den == 0: all observations have one step.
        [(7, 1.0)],  # Fewer than two retained points cannot define a slope.
    ],
)
def test_fit_returns_none_without_a_defined_slope(
    points: list[tuple[int, float]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_fit_rate(tmp_path, monkeypatch)

    assert mod.fit(points, cut=0) is None


def test_fit_rate_constants_are_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_fit_rate(tmp_path, monkeypatch)

    assert mod.STEP.fullmatch("Step 123:")
    assert mod.PRIOR == pytest.approx(0.4012836)
