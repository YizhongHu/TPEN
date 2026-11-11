"""Independent float64 NumPy oracle for dense stochastic reconfiguration.

This module is a *reference*, not a second copy of the subject.  It never
imports :mod:`tpen.training.score_geometry`, :mod:`tpen.training.qgt`, or
:mod:`tpen.training.sr`, which are the code these tests exist to check.  It
also deliberately differs from the subject structurally:

* it is NumPy, not Torch;
* it solves in **parameter space** with an LU solve
  (:func:`numpy.linalg.solve`), whereas the subject's preferred route is a
  **sample-space** Gram solve via symmetric eigendecomposition.

So a passing comparison exercises two algebraically equivalent but
structurally independent paths, in the spirit of the adoption analysis's
"tiny float64 NumPy parameter-space solve as the dense oracle".  A shared
helper would have made the comparison vacuous.

Conventions are restated here from first principles rather than imported, so
that a convention change in the subject shows up as a test failure instead of
propagating silently into the reference.

For a real wavefunction with scores ``O[k, i] = d log|psi(x_k)| / d theta_i``
and local energies ``E[k]`` over ``N`` samples:

    Obar = O - mean_k O
    A    = Obar / sqrt(N)
    eps  = c (E - mean_k E) / sqrt(N)
    S    = A^T A
    g    = A^T eps
    lam  = max(absolute + relative * trace(S) / P, minimum)
    dtheta = (S + lam I)^{-1} g
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DenseSROracleResult",
    "damping_shift",
    "design_matrix",
    "energy_gradient",
    "energy_residual",
    "parameter_qgt",
    "sr_direction",
]


@dataclass(frozen=True)
class DenseSROracleResult:
    """Every intermediate of one reference SR solve, for targeted assertions."""

    design: np.ndarray
    residual: np.ndarray
    qgt: np.ndarray
    gradient: np.ndarray
    shift: float
    direction: np.ndarray


def design_matrix(scores: np.ndarray) -> np.ndarray:
    """Return ``A = (O - mean) / sqrt(N)`` for an ``[N, P]`` score matrix."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be an [N, P] matrix")
    n_samples = scores.shape[0]
    if n_samples <= 0:
        raise ValueError("scores must contain at least one sample")
    centered = scores - scores.mean(axis=0, keepdims=True)
    return centered / np.sqrt(float(n_samples))


def energy_residual(energies: np.ndarray, *, scale: float = 2.0) -> np.ndarray:
    """Return ``eps = scale * (E - mean E) / sqrt(N)``."""

    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    n_samples = energies.shape[0]
    if n_samples <= 0:
        raise ValueError("energies must contain at least one sample")
    return float(scale) * (energies - energies.mean()) / np.sqrt(float(n_samples))


def parameter_qgt(scores: np.ndarray) -> np.ndarray:
    """Return the empirical parameter-space QGT ``S = A^T A``."""

    design = design_matrix(scores)
    return design.T @ design


def energy_gradient(
    scores: np.ndarray,
    energies: np.ndarray,
    *,
    scale: float = 2.0,
) -> np.ndarray:
    """Return the ordinary VMC energy gradient ``g = A^T eps``.

    With the default ``scale = 2`` this equals
    ``2 * mean_k[(E_k - Ebar) * (O_k - Obar)]``, the gradient of TPEN's
    score-function objective.
    """

    return design_matrix(scores).T @ energy_residual(energies, scale=scale)


def damping_shift(
    scores: np.ndarray,
    *,
    absolute: float = 0.0,
    relative: float = 1.0e-3,
    minimum: float = 0.0,
) -> float:
    """Return ``max(absolute + relative * trace(S) / P, minimum)``."""

    qgt = parameter_qgt(scores)
    n_parameters = qgt.shape[0]
    trace = float(np.trace(qgt))
    return float(max(absolute + relative * (trace / n_parameters), minimum))


def sr_direction(
    scores: np.ndarray,
    energies: np.ndarray,
    *,
    absolute: float = 0.0,
    relative: float = 1.0e-3,
    minimum: float = 0.0,
    scale: float = 2.0,
    rank_cutoff: float = 0.0,
) -> DenseSROracleResult:
    """Solve the damped parameter-space SR system and return every intermediate.

    Parameters
    ----------
    scores : numpy.ndarray
        Raw, uncentered ``[N, P]`` score matrix.
    energies : numpy.ndarray
        Local energies with ``N`` elements.
    absolute, relative, minimum : float, optional
        Damping terms, matching the subject's policy definition.
    scale : float, optional
        Energy-gradient scale ``c``.
    rank_cutoff : float, optional
        Relative eigenvalue cutoff.  ``0.0`` uses a plain LU solve, which is
        the structurally independent path; a nonzero cutoff necessarily uses
        an eigendecomposition, since truncation is defined spectrally.

    Returns
    -------
    DenseSROracleResult
        Design matrix, residual, QGT, gradient, shift, and direction.
    """

    design = design_matrix(scores)
    residual = energy_residual(energies, scale=scale)
    qgt = design.T @ design
    gradient = design.T @ residual
    shift = damping_shift(
        scores,
        absolute=absolute,
        relative=relative,
        minimum=minimum,
    )
    n_parameters = qgt.shape[0]
    damped = qgt + shift * np.eye(n_parameters, dtype=np.float64)

    if rank_cutoff <= 0.0:
        # LU solve: a different factorization from the subject's eigh, so an
        # agreement is not two implementations sharing one numerical routine.
        direction = np.linalg.solve(damped, gradient)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(qgt)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        threshold = rank_cutoff * float(eigenvalues.max()) if eigenvalues.size else 0.0
        projected = eigenvectors.T @ gradient
        scaled = np.where(
            eigenvalues >= threshold,
            projected / (eigenvalues + shift),
            0.0,
        )
        direction = eigenvectors @ scaled

    return DenseSROracleResult(
        design=design,
        residual=residual,
        qgt=qgt,
        gradient=gradient,
        shift=shift,
        direction=direction,
    )
