"""Exact one-electron hydrogen-like wavefunction reference."""

from __future__ import annotations

import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.batch.geometry import electron_nuclear_distances


class HydrogenExact(nn.Module):
    """Fixed exact-form wavefunction for one-electron, charge-one hydrogen.

    psi(r) = exp(-alpha r), where r is the electron-nucleus distance and
    `alpha` is a fixed construction-time literal, never a trainable
    parameter. `alpha = 1` is the true nonrelativistic infinite-nuclear-mass
    ground state (`exact_energy` = -0.5 hartree); other values, e.g.
    `alpha = 1.2`, are a known-imperfect control with no single exact
    eigenvalue (the local energy is not constant).

    Parameters
    ----------
    atoms : AtomicConfiguration
        Fixed nuclear geometry authority (`main.typ`, "Electron-nucleus
        cusp"). Must hold exactly one charge-one nucleus: this formula is
        the Z=1 ground state and is not a general-Z hydrogenic solution.
    alpha : float, optional
        Fixed radial decay rate. Default 1.0 (the exact ground state).
    """

    exact_alpha: float = 1.0
    exact_energy: float = -0.5

    def __init__(self, atoms: object, alpha: float = 1.0) -> None:
        super().__init__()
        if not isinstance(atoms, AtomicConfiguration):
            raise TypeError(f"{type(self).__name__} requires an AtomicConfiguration, got {type(atoms).__name__}")
        if atoms.n_nuclei != 1:
            raise ValueError(f"{type(self).__name__} requires exactly one nucleus, got {atoms.n_nuclei}")
        if float(atoms.charges[0]) != 1.0:
            raise ValueError(f"{type(self).__name__} requires a charge-one nucleus (Z=1)")
        self.atoms = atoms
        self.alpha = float(alpha)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate `log|psi|` and `sign(psi)` for a batch of configurations."""

        flat = batch.flatten_samples()
        if flat.n_electrons != 1:
            raise ValueError(f"{type(self).__name__} requires exactly one electron, got {flat.n_electrons}")
        atoms = self.atoms.to(device=flat.device, dtype=flat.dtype)
        r = electron_nuclear_distances(flat, nuclear_positions=atoms.positions).reshape(flat.batch_size)
        logabs = -self.alpha * r
        sign = torch.ones_like(logabs)
        return WavefunctionOutput(logabs=logabs, sign=sign)


__all__ = ["HydrogenExact"]
