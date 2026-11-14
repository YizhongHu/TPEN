"""Exact Hooke atom wavefunction references.

The one-electron oracle is the three-dimensional harmonic-oscillator ground
state in atomic units.  The pair oracles below are separate, two-electron
Taut-solvable references and deliberately remain unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput


class OneElectronHookeExact(nn.Module):
    """Exact one-electron Hooke ground state in three spatial dimensions.

    Parameters
    ----------
    omega : float, optional
        Isotropic harmonic-trap frequency in atomic units.  The default is
        ``0.5``, the frequency used by the registered Hooke one-electron
        benchmark.

    Notes
    -----
    For one electron the normalized wavefunction is proportional to
    ``exp(-omega * |r|**2 / 2)``.  With the atomic-unit Hamiltonian
    ``-1/2 nabla^2 + omega**2 |r|**2 / 2`` its exact energy is ``3*omega/2``.
    """

    def __init__(self, omega: float = 0.5) -> None:
        super().__init__()
        if omega <= 0:
            raise ValueError("omega must be positive")
        self.omega = float(omega)

    @property
    def exact_energy(self) -> float:
        """Return the three-dimensional oscillator ground-state energy."""

        return 1.5 * self.omega

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate log amplitude and positive sign for one-electron samples."""

        pos = batch.flatten_samples().positions  # [B, 1, 3]
        radius_sq = pos[:, 0, :].square().sum(-1)
        # The normalization constant cancels from derivatives of log|psi|.
        logabs = -0.5 * self.omega * radius_sq
        sign = torch.ones_like(logabs)
        return WavefunctionOutput(logabs=logabs, sign=sign)


class HookeSingletExact(nn.Module):
    """Exact spatial wavefunction for the two-electron Hooke singlet (omega=1/2).

    psi(r1, r2) = (1 + r12/2) * exp(-r1^2/4 - r2^2/4)
    Exact energy: E = 2.0
    """

    omega: float = 0.5
    exact_energy: float = 2.0

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        pos = batch.flatten_samples().positions  # [B, 2, 3]
        r1_sq = pos[:, 0, :].square().sum(-1)
        r2_sq = pos[:, 1, :].square().sum(-1)
        r12 = (pos[:, 0, :] - pos[:, 1, :]).norm(dim=-1)
        logabs = torch.log1p(0.5 * r12) - 0.25 * (r1_sq + r2_sq)
        sign = torch.ones_like(logabs)
        return WavefunctionOutput(logabs=logabs, sign=sign)


class HookeTripletExact(nn.Module):
    """Exact spatial wavefunction for the two-electron Hooke triplet (omega=1/4).

    psi(r1, r2) = (z1 - z2) * (1 + r12/4) * exp(-r1^2/8 - r2^2/8)
    Exact energy: E = 1.25

    Tests must sample away from the node z1 = z2.
    """

    omega: float = 0.25
    exact_energy: float = 1.25

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        pos = batch.flatten_samples().positions  # [B, 2, 3]
        r1_sq = pos[:, 0, :].square().sum(-1)
        r2_sq = pos[:, 1, :].square().sum(-1)
        r12 = (pos[:, 0, :] - pos[:, 1, :]).norm(dim=-1)
        z_diff = pos[:, 0, 2] - pos[:, 1, 2]
        sign = torch.sign(z_diff)
        logabs = torch.log(torch.abs(z_diff)) + torch.log1p(0.25 * r12) - 0.125 * (r1_sq + r2_sq)
        return WavefunctionOutput(logabs=logabs, sign=sign)


__all__ = ["HookeSingletExact", "HookeTripletExact", "OneElectronHookeExact"]
