"""Sampler warmup and equilibration helpers."""

from __future__ import annotations

from tpen.data.batch import Walkers


def equilibrate(
    model,
    sampler,
    walkers: Walkers,
    n_steps: int,
    target_acceptance: tuple[float, float] | None = None,
) -> Walkers:
    """Run sampler burn-in and optionally adapt proposal scale.

    Parameters
    ----------
    model : callable
        Wavefunction model sampled by `sampler`.
    sampler : object
        Sampler exposing a ``sample(model, walkers, n_steps)`` method. If it
        owns a move with ``step_size``, this function can adapt that scale.
    walkers : Walkers
        Initial walker state.
    n_steps : int
        Number of burn-in steps.
    target_acceptance : tuple of float or None, optional
        Inclusive acceptance-rate interval. If supplied and the sampler has a
        mutable ``move.step_size``, the scale is adjusted once after burn-in.

    Returns
    -------
    Walkers
        Burned-in walker state with cached model values.
    """

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    # ``sample`` trusts populated caches; refresh or invalidate before reuse.
    walkers = sampler.sample(model, walkers, n_steps)
    if target_acceptance is not None and hasattr(getattr(sampler, "move", None), "step_size"):
        low, high = target_acceptance
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("target_acceptance must satisfy 0 <= low <= high <= 1")
        if sampler.acceptance_rate < low:
            sampler.move.step_size *= 0.9
        elif sampler.acceptance_rate > high:
            sampler.move.step_size *= 1.1
    return walkers


def warmup(model, sampler, walkers: Walkers, n_steps: int) -> Walkers:
    """Run a warmup phase using the sampler.

    Parameters
    ----------
    model : callable
        Wavefunction model sampled by `sampler`.
    sampler : object
        Sampler exposing ``sample``.
    walkers : Walkers
        Initial walker state.
    n_steps : int
        Number of warmup steps.

    Returns
    -------
    Walkers
        Warmed-up walker state.
    """

    return equilibrate(model, sampler, walkers, n_steps)
