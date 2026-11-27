"""Acceptance tests for the opt-in E1 score-estimator transform."""

from pathlib import Path

import pytest
import torch

from tpen.training.trainer import VMCTrainer
from tpen.training.vmc import (
    clip_local_energy_for_score,
    compute_vmc_objective,
    resolve_energy_clipping_k,
    resolve_energy_clipping_mode,
)


def test_none_is_bit_identical_to_the_historical_score_objective() -> None:
    """The opt-in flag must not perturb the default gradient or raw metrics."""

    logabs_a = torch.tensor([0.2, -0.4, 0.8, 1.1], dtype=torch.float64, requires_grad=True)
    logabs_b = logabs_a.detach().clone().requires_grad_()
    energy = torch.tensor([-1.0, 0.5, 7.0, 2.0], dtype=torch.float64)

    baseline = compute_vmc_objective(logabs_a, energy)
    explicit_none = compute_vmc_objective(
        logabs_b,
        energy,
        energy_clipping_mode="none",
        energy_clipping_k=5.0,
    )
    baseline.loss.backward()
    explicit_none.loss.backward()

    assert torch.equal(baseline.loss, explicit_none.loss)
    assert torch.equal(logabs_a.grad, logabs_b.grad)
    assert baseline.metrics == explicit_none.metrics


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_gradient_matches_frozen_raw_mean_centered_reference(
    dtype: torch.dtype,
) -> None:
    """The default gradient remains the inline, unclipped score estimator."""

    energy = torch.tensor([0.0] * 15 + [100.0], dtype=dtype)
    logabs = torch.linspace(-1.0, 1.0, 16, dtype=dtype, requires_grad=True)
    reference_logabs = logabs.detach().clone().requires_grad_()

    result = compute_vmc_objective(logabs, energy)
    # Frozen raw mean-centering reference: this deliberately does not call the
    # clipping implementation, so a default->psiformer mutant changes the
    # gradient assertion below before any diagnostic assertion can pass.
    reference_loss = 2.0 * torch.mean((energy - energy.mean()) * reference_logabs)
    reference_loss.backward()
    result.loss.backward()

    assert torch.equal(logabs.grad, reference_logabs.grad)
    assert torch.equal(result.loss, reference_loss)


def test_psiformer_uses_even_median_mean_ad_and_clipped_recentering() -> None:
    """The hand calculation pins the even-count median and second centering."""

    logabs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    energy = torch.tensor([0.0, 1.0, 2.0, 100.0], dtype=torch.float64)
    result = compute_vmc_objective(
        logabs,
        energy,
        energy_clipping_mode="psiformer",
        energy_clipping_k=1.0,
    )

    # Median=1.5, mean AD=25.25, so the upper bound is 26.75 and the clipped
    # energies are [0, 1, 2, 26.75], recentered by their mean 7.4375.
    clipped = torch.tensor([0.0, 1.0, 2.0, 26.75], dtype=torch.float64)
    expected_loss = 2.0 * torch.mean((clipped - clipped.mean()) * logabs)
    assert torch.equal(result.loss, expected_loss)
    assert result.metrics["energy"] == float(energy.mean())
    assert result.metrics["energy_variance"] == float(energy.var(unbiased=False))
    assert result.metrics["local_energy_clip_fraction"] == 0.25
    assert result.metrics["local_energy_clip_lower"] == -23.75
    assert result.metrics["local_energy_clip_upper"] == 26.75


@pytest.mark.parametrize(
    "values, expected, bounds, fraction",
    [
        ([0.0, 0.0, 0.0, 100.0], [0.0, 0.0, 0.0, 25.0], (-25.0, 25.0), 0.25),
        ([2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], (2.0, 2.0), 0.0),
        (
            [0.0, 1.0, 100.0],
            [0.0, 1.0, 103.0 / 3.0],
            (-97.0 / 3.0, 103.0 / 3.0),
            1.0 / 3.0,
        ),
    ],
)
def test_psiformer_handles_ties_all_equal_and_odd_count(
    values: list[float],
    expected: list[float],
    bounds: tuple[float, float],
    fraction: float,
) -> None:
    """Tie, zero-deviation, and odd-count inputs use the stated median rule."""

    energy = torch.tensor(values, dtype=torch.float64)
    before = energy.clone()
    clipped, metrics = clip_local_energy_for_score(energy, mode="psiformer", k=1.0)

    torch.testing.assert_close(clipped, torch.tensor(expected, dtype=torch.float64))
    assert torch.equal(energy, before)
    assert metrics["local_energy_clip_lower"] == pytest.approx(bounds[0])
    assert metrics["local_energy_clip_upper"] == pytest.approx(bounds[1])
    assert metrics["local_energy_clip_fraction"] == fraction


def test_clipping_changes_only_the_surrogate_and_logs_raw_metrics() -> None:
    """Raw energy observables are invariant while the score surrogate changes."""

    logabs = torch.tensor([0.2, -0.4, 0.8, 1.1], dtype=torch.float64, requires_grad=True)
    energy = torch.tensor([-1.0, 0.5, 7.0, 2.0], dtype=torch.float64)
    raw = compute_vmc_objective(logabs, energy)
    clipped = compute_vmc_objective(
        logabs.detach().clone().requires_grad_(),
        energy,
        energy_clipping_mode="psiformer",
        energy_clipping_k=1.0,
    )

    for key in ("energy", "energy_variance", "energy_std", "energy_stderr"):
        assert clipped.metrics[key] == raw.metrics[key]
    assert clipped.metrics["local_energy_clip_fraction"] > 0.0
    assert clipped.loss != raw.loss


@pytest.mark.parametrize("mode", ["unknown", "soft", "", None, 1])
def test_unknown_clipping_mode_is_rejected(mode: object) -> None:
    with pytest.raises(ValueError, match="energy clipping mode"):
        resolve_energy_clipping_mode(mode)
    with pytest.raises(ValueError, match="energy clipping mode"):
        VMCTrainer(max_steps=1, energy_clipping_mode=mode)


@pytest.mark.parametrize("k", [0, -1, float("inf"), float("nan"), True, "5"])
def test_invalid_clipping_k_is_rejected(k: object) -> None:
    with pytest.raises(ValueError, match="energy clipping k"):
        resolve_energy_clipping_k(k)
    with pytest.raises(ValueError, match="energy clipping k"):
        VMCTrainer(max_steps=1, energy_clipping_k=k)


def test_hooke2_opt_in_overlay_is_declared() -> None:
    overlay = Path("experiments/atomistic/stage1/configs/overlays/hooke2-e1-clipping.yaml")
    assert overlay.is_file()
    assert "energy_clipping_mode: psiformer" in overlay.read_text()
    assert "energy_clipping_k: 5.0" in overlay.read_text()
