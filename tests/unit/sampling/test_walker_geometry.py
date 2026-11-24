"""Tests for walker-geometry diagnostics in the sampling namespace."""

from __future__ import annotations

import math

import pytest
import torch

from tpen.data.batch import Walkers
from tpen.sampling import SamplerStats, summarize_walker_geometry
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn

GEOMETRY_KEYS = (
    "position_mean_abs",
    "position_rms",
    "position_max_abs",
    "radius_mean",
    "radius_std",
    "radius_q50",
    "radius_q90",
    "radius_q99",
    "radius_max",
    "center_of_mass_rms",
)

PAIR_DISTANCE_KEYS = (
    "electron_distance_min",
    "electron_distance_q01",
    "electron_distance_q05",
    "electron_distance_mean",
    "electron_distance_q50",
    "electron_distance_q95",
    "electron_distance_q99",
    "electron_distance_max",
)


def _walkers(positions: torch.Tensor) -> Walkers:
    return Walkers(positions=positions)


def test_summary_returns_json_safe_scalars() -> None:
    generator = torch.Generator().manual_seed(11)
    walkers = _walkers(torch.randn(8, 2, 3, generator=generator, dtype=torch.float64))

    metrics = summarize_walker_geometry(walkers)

    for key in (*GEOMETRY_KEYS, *PAIR_DISTANCE_KEYS, "n_walkers", "n_electrons", "spatial_dim"):
        assert key in metrics, f"missing {key}"
    for key, value in metrics.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), key
        assert math.isfinite(float(value)), key
    assert metrics["n_walkers"] == 8
    assert metrics["n_electrons"] == 2
    assert metrics["spatial_dim"] == 3
    assert metrics["electron_distance_n_pairs"] == 8  # one pair per walker


def test_radius_metrics_for_known_batch() -> None:
    # Two walkers, one electron each, at distances 3 and 4 from the origin.
    positions = torch.tensor(
        [[[3.0, 0.0, 0.0]], [[0.0, 4.0, 0.0]]],
        dtype=torch.float64,
    )

    metrics = summarize_walker_geometry(_walkers(positions))

    assert metrics["radius_mean"] == pytest.approx(3.5)
    assert metrics["radius_std"] == pytest.approx(0.5)
    assert metrics["radius_q50"] == pytest.approx(3.5)
    assert metrics["radius_max"] == pytest.approx(4.0)
    assert metrics["position_max_abs"] == pytest.approx(4.0)
    assert metrics["position_rms"] == pytest.approx(math.sqrt((9.0 + 16.0) / 6.0))
    # One electron per walker: the center of mass is the electron itself.
    assert metrics["center_of_mass_rms"] == pytest.approx(math.sqrt((9.0 + 16.0) / 2.0))


def test_electron_distance_metrics_for_known_two_electron_batch() -> None:
    # Pair distances: walker 0 -> 2.0, walker 1 -> 6.0.
    positions = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 3.0, 0.0], [0.0, -3.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    metrics = summarize_walker_geometry(_walkers(positions))

    assert metrics["electron_distance_n_pairs"] == 2
    assert metrics["electron_distance_min"] == pytest.approx(2.0)
    assert metrics["electron_distance_max"] == pytest.approx(6.0)
    assert metrics["electron_distance_mean"] == pytest.approx(4.0)
    assert metrics["electron_distance_q50"] == pytest.approx(4.0)


def test_single_electron_system_has_no_pair_metrics() -> None:
    walkers = _walkers(torch.randn(4, 1, 3, dtype=torch.float64))

    metrics = summarize_walker_geometry(walkers)

    assert metrics["n_electrons"] == 1
    assert metrics["electron_distance_n_pairs"] == 0
    for key in PAIR_DISTANCE_KEYS:
        assert key not in metrics
    # Radius metrics still exist for a single electron.
    assert "radius_mean" in metrics


def test_zero_electron_system_returns_counts_only() -> None:
    walkers = _walkers(torch.zeros(4, 0, 3, dtype=torch.float64))

    metrics = summarize_walker_geometry(walkers)

    assert metrics == {
        "n_walkers": 4,
        "n_electrons": 0,
        "spatial_dim": 3,
        "electron_distance_n_pairs": 0,
    }


def test_non_3d_positions_rejected() -> None:
    # Walkers itself rejects the malformed shape before the summary runs.
    with pytest.raises(ValueError, match="spatial_dim"):
        summarize_walker_geometry(_walkers(torch.zeros(4, 3, dtype=torch.float64)))


def test_collect_samples_includes_geometry_and_metadata_stats() -> None:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()

    walkers, stats = sampler.collect_samples(model, emitter=None)

    assert isinstance(stats, SamplerStats)
    # Geometry is a field of the typed record, not a flattened attribute set.
    for key in (*GEOMETRY_KEYS, *PAIR_DISTANCE_KEYS):
        assert key in stats.geometry, f"missing geometry stat {key}"
    assert stats.n_walkers == walkers.batch_size
    assert stats.geometry["n_electrons"] == 2

    # The logging edge still emits every durable train/sampler key.
    metrics = stats.as_metrics()
    for key in (
        "acceptance_rate",
        "n_walkers",
        "burn_in",
        "n_steps",
        "proposal_scale",
        "seed",
        *GEOMETRY_KEYS,
        *PAIR_DISTANCE_KEYS,
    ):
        assert key in metrics, f"missing stat {key}"
    assert metrics["n_walkers"] == walkers.batch_size
    assert metrics["n_electrons"] == 2
    assert metrics["radius_max"] >= metrics["radius_q99"] >= metrics["radius_q50"] > 0.0
    assert metrics["electron_distance_min"] > 0.0
