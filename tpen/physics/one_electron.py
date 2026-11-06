"""Analytic one-electron reference wavefunctions in three dimensions.

These parameter-free modules are measurement controls, not learned ansatzes.
The Gaussian is the oscillator ground state when its alpha equals omega;
the exponential is a Coulomb ground state when its decay equals the charge.
Other positive values provide deliberately imperfect, normalizable controls.
Neither module evaluates a local energy or calls a learned forward.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput


def _one_electron_positions(batch: ElectronBatch) -> torch.Tensor:
    """Flatten samples and require exactly one physical electron in space."""
    positions = batch.flatten_samples().positions
    if positions.shape[1:] != (1, 3):
        raise ValueError("one-electron references require positions shaped [batch, 1, 3]")
    return positions[:, 0]


class OneElectronGaussian(nn.Module):
    """Normalized Gaussian reference with log amplitude minus alpha*r**2/2.

    Parameters
    ----------
    alpha : float, optional
        Positive inverse squared length. At oscillator frequency omega=0.5,
        alpha=0.5 has local energy 0.75 atomic units everywhere.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Return the signed logarithm of the normalized reference."""
        positions = _one_electron_positions(batch)
        logabs = (
            0.75 * (math.log(self.alpha) - math.log(math.pi))
            - 0.5 * self.alpha * positions.square().sum(-1)
        )
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class OneElectronExponential(nn.Module):
    """Normalized hydrogenic reference with log amplitude minus decay*r.

    Parameters
    ----------
    decay : float, optional
        Positive inverse length. At nuclear charge one, decay=1 has local
        energy -0.5 atomic units for r>0. Derivatives at r=0 are undefined;
        this module intentionally does not regularize the nuclear cusp.
    """

    def __init__(self, decay: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(decay) or decay <= 0:
            raise ValueError("decay must be finite and positive")
        self.decay = float(decay)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Return the signed logarithm of the normalized reference."""
        positions = _one_electron_positions(batch)
        logabs = (
            1.5 * math.log(self.decay) - 0.5 * math.log(math.pi)
            - self.decay * positions.norm(dim=-1)
        )
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


__all__ = ["OneElectronGaussian", "OneElectronExponential"]
