"""Focused round-one review probes for electron-nucleus distance logging."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tpen.callback import ElectronNucleusDistanceQuantiles
from tpen.callback.electron_nucleus_distance import summarize_electron_nucleus_distances
from tpen.data.batch import ElectronBatch

from .support import RecordingContext, deliver_completed_iteration, training_state


def _batch(
    positions: torch.Tensor,
    nuclear_positions: torch.Tensor,
    nuclear_charges: torch.Tensor,
) -> ElectronBatch:
    """Construct a valid batch while preserving every sample axis."""

    return ElectronBatch(
        positions=positions.to(dtype=torch.float64),
        nuclear_positions=nuclear_positions.to(dtype=torch.float64),
        nuclear_charges=nuclear_charges.to(dtype=torch.float64),
    )


def _numpy_oracle(positions: torch.Tensor, nuclei: torch.Tensor) -> dict[str, float]:
    """Compute nearest-nucleus minima and linear quantiles independently."""

    p = positions.detach().cpu().numpy()
    r = nuclei.detach().cpu().numpy()
    distances = np.linalg.norm(p.reshape(-1, p.shape[-2], 3)[:, :, None, :] - r.reshape(-1, r.shape[-2], 3)[:, None, :, :], axis=-1)
    nearest = distances.min(axis=-1)
    levels = (0.25, 0.5, 0.75)
    expected: dict[str, float] = {}
    for electron in range(p.shape[-2]):
        prefix = f"electron_{electron}_nucleus_distance"
        expected[f"{prefix}_min"] = float(nearest[:, electron].min())
        for level, value in zip(
            levels,
            np.quantile(nearest[:, electron], levels, method="linear"),
            strict=True,
        ):
            expected[f"{prefix}_q{100 * level:g}".replace(".", "p")] = float(value)
    return expected


def test_sampled_nuclei_preserve_all_configuration_axes_and_match_flattened_control() -> None:
    """Exercise the exported sampled-nuclei ``ElectronBatch`` contract.

    The ordinary Metropolis training path carries fixed 2-D
    ``AtomicConfiguration`` positions.  This probe is deliberately bounded to
    the public helper contract for an ``ElectronBatch`` whose nuclei vary over
    multiple sample axes; it does not claim that normal training reaches this
    representation.
    """

    positions = torch.tensor(
        [
            [
                [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]],
                [[2.2, 0.0, 0.0], [3.6, 0.0, 0.0]],
                [[4.4, 0.0, 0.0], [5.8, 0.0, 0.0]],
            ],
            [
                [[0.6, 1.0, 0.0], [1.8, 1.0, 0.0]],
                [[2.7, 1.0, 0.0], [3.1, 1.0, 0.0]],
                [[4.9, 1.0, 0.0], [5.5, 1.0, 0.0]],
            ],
        ]
    )
    nuclei = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
                [[2.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
                [[4.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
            ],
        ]
    )
    charges = torch.ones((2, 3, 2))
    sampled = _batch(positions, nuclei, charges)
    levels = (0.25, 0.5, 0.75)

    # This is the direct public contract: every [2, 3] configuration carries
    # its own nuclei.  A failure here is a product regression, not a fixture
    # failure, because the equivalent flattened control below is valid.
    actual = summarize_electron_nucleus_distances(sampled, quantiles=levels)
    assert actual == pytest.approx(_numpy_oracle(positions, nuclei))

    flattened = _batch(
        positions.reshape(6, 2, 3),
        nuclei.reshape(6, 2, 3),
        charges.reshape(6, 2),
    )
    assert summarize_electron_nucleus_distances(flattened, quantiles=levels) == pytest.approx(actual)


def test_shared_nuclei_control_matches_numpy_for_multi_sample_positions() -> None:
    """The same multi-axis electron batch works with shared nuclei."""

    positions = torch.tensor(
        [
            [[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
            [[[4.0, 0.0, 0.0]], [[6.0, 0.0, 0.0]]],
        ]
    )
    nuclei = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    batch = _batch(positions, nuclei, torch.ones(2))
    levels = (0.25, 0.5, 0.75)
    actual = summarize_electron_nucleus_distances(batch, quantiles=levels)
    p = positions.detach().cpu().numpy().reshape(-1, 1, 3)
    r = nuclei.detach().cpu().numpy().reshape(1, -1, 3)
    nearest = np.linalg.norm(p - r, axis=-1).min(axis=-1)
    expected_values = np.quantile(nearest, levels, method="linear")
    expected = {
        "electron_0_nucleus_distance_min": float(nearest.min()),
        **{
            f"electron_0_nucleus_distance_q{100 * level:g}".replace(".", "p"): float(value)
            for level, value in zip(levels, expected_values, strict=True)
        },
    }
    assert actual == pytest.approx(expected)


def test_flattened_sampled_nuclei_control_matches_numpy() -> None:
    """The equivalent flattened representation is a positive harness control."""

    positions = torch.tensor(
        [
            [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]],
            [[2.2, 0.0, 0.0], [3.6, 0.0, 0.0]],
            [[4.4, 0.0, 0.0], [5.8, 0.0, 0.0]],
            [[0.6, 1.0, 0.0], [1.8, 1.0, 0.0]],
            [[2.7, 1.0, 0.0], [3.1, 1.0, 0.0]],
            [[4.9, 1.0, 0.0], [5.5, 1.0, 0.0]],
        ]
    )
    nuclei = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
            [[2.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
            [[4.0, 1.0, 0.0], [10.0, 1.0, 0.0]],
        ]
    )
    actual = summarize_electron_nucleus_distances(
        _batch(positions, nuclei, torch.ones((6, 2))),
        quantiles=(0.25, 0.5, 0.75),
    )
    assert actual == pytest.approx(_numpy_oracle(positions, nuclei))


def test_quantile_labels_do_not_silently_drop_colliding_requested_levels() -> None:
    """Distinct requested levels must remain distinguishable or be refused."""

    batch = _batch(
        torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]], [[9.0, 0.0, 0.0]]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.ones(1),
    )
    colliding_levels = (0.1234567, 0.1234568)

    try:
        metrics = summarize_electron_nucleus_distances(batch, quantiles=colliding_levels)
    except ValueError as exc:
        assert "label" in str(exc).lower() or "quantile" in str(exc).lower()
    else:
        values = [
            value
            for key, value in metrics.items()
            if key.startswith("electron_0_nucleus_distance_q")
        ]
        assert len(values) == len(colliding_levels), "a colliding key silently overwrote one quantile"
        assert values[0] != values[1]


def test_standard_quantile_levels_keep_the_documented_metric_names() -> None:
    """The ordinary study levels remain a positive naming control."""

    batch = _batch(
        torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.ones(1),
    )
    metrics = summarize_electron_nucleus_distances(batch, quantiles=(0.01, 0.5, 0.99))
    nearest = np.array([0.0, 1.0, 4.0])
    expected = {
        "electron_0_nucleus_distance_q01": float(np.quantile(nearest, 0.01, method="linear")),
        "electron_0_nucleus_distance_q50": float(np.quantile(nearest, 0.5, method="linear")),
        "electron_0_nucleus_distance_q99": float(np.quantile(nearest, 0.99, method="linear")),
    }
    assert {key: metrics[key] for key in expected} == pytest.approx(expected)


def _step_batch(step: int) -> ElectronBatch:
    """Give each durable step a distinct value for log/value comparison."""

    return _batch(
        torch.tensor([[[float(step), 0.0, 0.0]], [[float(step + 1), 0.0, 0.0]]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.ones(1),
    )


def _run_steps(callback: ElectronNucleusDistanceQuantiles, steps: list[int]) -> list[dict]:
    context = RecordingContext()
    for count, step in enumerate(steps, start=1):
        state = training_state(step=step + 1000, batch=_step_batch(step))
        deliver_completed_iteration(callback, context, state, step=step, count=count)
    return context.by_namespace("train/electron_nucleus_distance")


def test_resumed_callback_replays_durable_steps_and_logs_event_coordinates() -> None:
    """A fresh callback on a resumed suffix matches uninterrupted delivery."""

    uninterrupted = _run_steps(
        ElectronNucleusDistanceQuantiles(quantiles=(0.5,), every_n_steps=3, start_step=1),
        [1, 2, 3, 4, 5, 6, 7],
    )
    resumed = _run_steps(
        ElectronNucleusDistanceQuantiles(quantiles=(0.5,), every_n_steps=3, start_step=1),
        [4, 5, 6, 7],
    )

    expected_suffix = [record for record in uninterrupted if record["step"] in {4, 7}]
    assert [record["step"] for record in resumed] == [4, 7]
    assert resumed == expected_suffix
    assert [record["metrics"]["electron_0_nucleus_distance_q50"] for record in resumed] == [4.5, 7.5]
