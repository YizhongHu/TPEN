"""Round-2 assembled-model probes for the positive-singlet contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch
from tpen.physics.kinetic import KineticEnergy


_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS = {
    "tail": _ROOT / "experiments/atomistic/stage1/configs/hooke2/positive_singlet_tail.yaml",
    "comparator": _ROOT / "experiments/atomistic/stage1/configs/hooke2/positive_singlet_comparator.yaml",
}
_DTYPE = torch.float64


def _model_and_batch(arm: str, *, include_spins: bool = False):
    torch.manual_seed(12345)
    cfg = OmegaConf.load(_CONFIGS[arm])
    OmegaConf.resolve(cfg)
    cfg.model.embedding.include_spins = include_spins
    model = instantiate(cfg.model).to(dtype=_DTYPE)
    sampler = instantiate(cfg.sampler)
    return model, sampler, sampler.reset(n_walkers=1).make_batch()


def _generic_positions(*, dtype: torch.dtype = _DTYPE) -> torch.Tensor:
    com = torch.tensor([0.37, -0.23, 0.19], dtype=dtype)
    delta = torch.tensor([0.81, -0.44, 0.29], dtype=dtype)
    return torch.stack((com + delta / 2, com - delta / 2)).reshape(1, 2, 3)


def _radial_batch(model_batch: ElectronBatch, radius: float) -> ElectronBatch:
    com = torch.tensor([0.37, -0.23, 0.19], dtype=_DTYPE)
    direction = torch.tensor([1.0, -2.0, 2.0], dtype=_DTYPE)
    direction = direction / torch.linalg.vector_norm(direction)
    delta = radius * direction
    positions = torch.stack((com + delta / 2, com - delta / 2)).reshape(1, 2, 3)
    return replace(model_batch, positions=positions)


@pytest.mark.parametrize("arm", ["tail", "comparator"])
def test_assembled_models_are_spatially_symmetric_at_fixed_spins(arm: str) -> None:
    model, _, sampler_batch = _model_and_batch(arm)
    batch = replace(sampler_batch, positions=_generic_positions())
    spatial_only = replace(batch, positions=batch.positions[:, [1, 0]])
    simultaneous = replace(
        spatial_only,
        spins=spatial_only.spins[:, [1, 0]],
    )

    direct = model(batch).logabs
    torch.testing.assert_close(direct, model(spatial_only).logabs, rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(direct, model(simultaneous).logabs, rtol=0.0, atol=1.0e-12)


def test_boundary_average_covers_spin_enabled_spatial_exchange() -> None:
    """Exercise the model-boundary average when the trunk itself is asymmetric."""

    model, _, sampler_batch = _model_and_batch("tail", include_spins=True)
    batch = replace(sampler_batch, positions=_generic_positions())
    spatial_only = replace(batch, positions=batch.positions[:, [1, 0]])

    # The public assembled path is symmetric with spins held fixed (+,-).
    direct = model(batch).logabs
    torch.testing.assert_close(direct, model(spatial_only).logabs, rtol=0.0, atol=1.0e-12)

    # The same fixture is a negative control for the verifier mutant: without
    # TPENWaveFunction's boundary average, the spin-conditioned trunk is not
    # spatially symmetric.  This private one-pass helper is the un-averaged g.
    unaveraged = model._readout_output_once(batch).logabs
    unaveraged_swapped = model._readout_output_once(spatial_only).logabs
    assert torch.max(torch.abs(unaveraged - unaveraged_swapped)) > 1.0e-8


@pytest.mark.parametrize("arm", ["tail", "comparator"])
def test_assembled_residual_has_zero_fixed_spin_directional_derivative_at_coalescence(arm: str) -> None:
    model, _, sampler_batch = _model_and_batch(arm)
    com = torch.tensor([0.37, -0.23, 0.19], dtype=_DTYPE)
    positions = torch.stack((com, com)).reshape(1, 2, 3).requires_grad_(True)
    output = model(replace(sampler_batch, positions=positions))
    gradient = torch.autograd.grad(output.aux["g"].sum(), positions)[0]
    direction = torch.tensor([1.0, -2.0, 2.0], dtype=_DTYPE)
    direction = direction / torch.linalg.vector_norm(direction)
    directional = (gradient[0, 0] - gradient[0, 1]).dot(direction) / 2
    assert torch.abs(directional) < 1.0e-10


def test_same_seed_parameters_and_logabs_delta_match_the_tail() -> None:
    tail, _, sampler_batch = _model_and_batch("tail")
    comparator, _, _ = _model_and_batch("comparator")
    tail_parameters = dict(tail.named_parameters())
    comparator_parameters = dict(comparator.named_parameters())
    assert tail_parameters.keys() == comparator_parameters.keys()
    for name in tail_parameters:
        assert torch.equal(tail_parameters[name], comparator_parameters[name]), name

    for radius in (1.0e-6, 1.0, 50.0):
        batch = _radial_batch(sampler_batch, radius)
        delta = tail(batch).logabs - comparator(batch).logabs
        expected = 0.5 * torch.log1p(torch.tensor([0.25 * radius**2], dtype=_DTYPE))
        roundoff = torch.finfo(_DTYPE).eps * 128.0 * (delta.abs() + expected.abs() + 1.0)
        torch.testing.assert_close(delta, expected, rtol=0.0, atol=float(roundoff.max()))


def test_production_kinetic_tail_delta_matches_analytic_identity() -> None:
    tail, _, sampler_batch = _model_and_batch("tail")
    comparator, _, _ = _model_and_batch("comparator")
    kinetic = KineticEnergy()
    kappa = 0.25

    for radius in (1.0e-6, 1.0, 50.0):
        base = _radial_batch(sampler_batch, radius)
        positions = base.positions.detach().clone().requires_grad_(True)
        probe = replace(base, positions=positions)
        comparator_logabs = comparator(probe).logabs
        comparator_gradient = torch.autograd.grad(comparator_logabs.sum(), positions, create_graph=True)[0]
        delta = positions[0, 0] - positions[0, 1]
        denominator = 1.0 + kappa * delta.square().sum()
        relative_gradient = kappa * delta / denominator
        h_gradient = torch.stack((relative_gradient, -relative_gradient)).unsqueeze(0)
        r2 = delta.square().sum()
        laplacian = 2.0 * kappa * (3.0 + kappa * r2) / denominator.square()
        expected = -0.5 * (
            laplacian
            + 2.0 * (comparator_gradient * h_gradient).sum()
            + h_gradient.square().sum()
        )
        observed = kinetic.local_energy(tail, base).total - kinetic.local_energy(comparator, base).total
        # Subtracting two independently evaluated kinetic energies loses digits
        # at tiny r12; retain a bounded float64 cancellation allowance.
        roundoff = torch.maximum(
            torch.finfo(_DTYPE).eps * 4096.0 * (observed.abs() + expected.abs() + 1.0),
            torch.tensor(1.0e-9, dtype=_DTYPE),
        )
        assert torch.all((observed - expected.reshape(1)).abs() <= roundoff)


@pytest.mark.parametrize(
    "config_name",
    [
        "he/eval.yaml",
        "qho2e/eval.yaml",
        "h2/eval.yaml",
    ],
)
def test_sampler_batches_reject_non_hooke_physics_without_injected_labels(config_name: str) -> None:
    cfg = OmegaConf.load(_ROOT / "experiments/atomistic/stage1/configs" / config_name)
    OmegaConf.resolve(cfg)
    sampler_cfg = cfg.evaluation_sampler
    sampler_cfg.system = cfg.system_descriptor
    sampler = instantiate(sampler_cfg)
    batch = sampler.reset(n_walkers=1).make_batch()
    model, _, _ = _model_and_batch("tail")
    with pytest.raises(ValueError, match="requires exactly kinetic|requires a configured"):
        model(batch)
