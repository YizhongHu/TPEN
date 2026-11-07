"""Independent operator-review probes for the one-electron e-e cusp claim."""

import torch

from tpen.data.batch import ElectronBatch
from tpen.nn.cusp import ElectronElectronCusp


def _batch(n_electrons: int) -> ElectronBatch:
    generator = torch.Generator().manual_seed(20261107 + n_electrons)
    positions = torch.randn(64, n_electrons, 3, generator=generator, dtype=torch.float64)
    spin_values = [1] if n_electrons == 1 else [1, -1]
    spins = torch.tensor([spin_values] * 64, dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)


def test_electron_electron_cusp_is_inert_for_one_electron() -> None:
    factor = ElectronElectronCusp(trainable_range=True)
    values = factor(_batch(1))

    assert torch.isfinite(values).all()
    assert torch.equal(values, torch.zeros_like(values)), f"one-electron values={values}"

    values.sum().backward()
    for name in ("raw_same_range", "raw_opposite_range"):
        gradient = getattr(factor, name).grad
        assert gradient is None or torch.equal(gradient, torch.zeros_like(gradient)), (
            f"one-electron {name} gradient={gradient}"
        )


def test_electron_electron_cusp_is_nonzero_for_two_electrons() -> None:
    values = ElectronElectronCusp(trainable_range=True)(_batch(2))

    assert torch.isfinite(values).all()
    assert torch.any(values != 0), f"two-electron values={values}"
