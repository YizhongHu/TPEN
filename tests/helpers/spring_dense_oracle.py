"""Independent NumPy parameter-space oracle for the SPRING recurrence.

This helper is deliberately independent of the Torch implementation.  It
imports no TPEN training module and does not call any production geometry,
operator, or solver helper.  The conventions are restated here so a change in
the subject's centering, normalization, residual scale, or route fails the
comparison instead of silently changing the expectation with it.

For raw score rows ``O[k, i]`` and local energies ``E[k]`` over ``N`` samples:

    A       = (O - mean(O, axis=0)) / sqrt(N)
    epsilon = 2 * (E - mean(E)) / sqrt(N)
    S       = A.T @ A
    g       = A.T @ epsilon
    lambda  = max(absolute + relative * trace(S) / P, minimum)

The parameter-space form of one SPRING step is the algebraic equivalent of
the sample-space form used in production:

    z_t = (S + lambda I)^(-1) (g_t - S mu z_(t-1)) + mu z_(t-1)

The recurrence returns the unscaled projected history ``z_t``.  Learning rate
and trust-cap application are intentionally left to the tests, because the
oracle's role is to validate the projected history before application policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DenseSPRINGOracleResult",
    "damping_shift",
    "design_matrix",
    "energy_residual",
    "spring_step",
    "spring_trajectory",
]


@dataclass(frozen=True)
class DenseSPRINGOracleResult:
    """Independent intermediates and the unscaled projected history."""

    design: np.ndarray
    residual: np.ndarray
    qgt: np.ndarray
    gradient: np.ndarray
    shift: float
    direction: np.ndarray
    history: np.ndarray


def design_matrix(scores: np.ndarray) -> np.ndarray:
    """Return ``A = (O - mean(O)) / sqrt(N)`` from raw score rows."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be an [N, P] matrix")
    n_samples = scores.shape[0]
    if n_samples <= 0:
        raise ValueError("scores must contain at least one sample")
    return (scores - scores.mean(axis=0, keepdims=True)) / np.sqrt(float(n_samples))


def energy_residual(energies: np.ndarray, *, scale: float = 2.0) -> np.ndarray:
    """Return ``epsilon = scale * (E - mean(E)) / sqrt(N)``."""

    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    n_samples = energies.shape[0]
    if n_samples <= 0:
        raise ValueError("energies must contain at least one sample")
    return float(scale) * (energies - energies.mean()) / np.sqrt(float(n_samples))


def damping_shift(
    scores: np.ndarray,
    *,
    absolute: float = 0.0,
    relative: float = 1.0e-3,
    minimum: float = 0.0,
) -> float:
    """Return the parameter-space damping shift from ``trace(S) / P``."""

    qgt = design_matrix(scores).T @ design_matrix(scores)
    n_parameters = qgt.shape[0]
    return float(max(absolute + relative * float(np.trace(qgt)) / n_parameters, minimum))


def spring_step(
    scores: np.ndarray,
    energies: np.ndarray,
    history: np.ndarray,
    *,
    history_decay: float,
    absolute: float = 0.0,
    relative: float = 1.0e-3,
    minimum: float = 0.0,
    scale: float = 2.0,
) -> DenseSPRINGOracleResult:
    """Solve one independent float64 parameter-space SPRING step."""

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
    previous = np.asarray(history, dtype=np.float64).reshape(-1)
    if previous.shape != (qgt.shape[0],):
        raise ValueError("history must have one value per parameter coordinate")
    damped_rhs = gradient - qgt @ (float(history_decay) * previous)
    direction = np.linalg.solve(qgt + shift * np.eye(qgt.shape[0]), damped_rhs)
    projected = direction + float(history_decay) * previous
    return DenseSPRINGOracleResult(
        design=design,
        residual=residual,
        qgt=qgt,
        gradient=gradient,
        shift=shift,
        direction=direction,
        history=projected,
    )


def spring_trajectory(
    score_steps: tuple[np.ndarray, ...] | list[np.ndarray],
    energy_steps: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    history_decay: float,
    absolute: float = 0.0,
    relative: float = 1.0e-3,
    minimum: float = 0.0,
    scale: float = 2.0,
) -> tuple[DenseSPRINGOracleResult, ...]:
    """Return independent results for a sequence of projected-history steps."""

    if len(score_steps) != len(energy_steps):
        raise ValueError("score_steps and energy_steps must have equal length")
    if not score_steps:
        return ()
    first = np.asarray(score_steps[0], dtype=np.float64)
    history = np.zeros(first.shape[1], dtype=np.float64)
    results: list[DenseSPRINGOracleResult] = []
    for scores, energies in zip(score_steps, energy_steps, strict=True):
        result = spring_step(
            scores,
            energies,
            history,
            history_decay=history_decay,
            absolute=absolute,
            relative=relative,
            minimum=minimum,
            scale=scale,
        )
        results.append(result)
        history = result.history
    return tuple(results)
