"""Fixed-model trajectory collection with an explicit draw axis.

`MetropolisSampler.collect_samples` returns only the final state of its
persistent walkers, so nothing in the sampling package retains a time series.
This module adds the missing draw axis by calling the sampler's public
production entry point repeatedly and evaluating one observable per draw,
producing a `[draw, walker]` trajectory that the statistics package can
estimate an autocorrelation time from.

"Fixed model" is enforced, not assumed. Autocorrelation over training updates is
meaningless -- the target distribution is moving, so a lag-k correlation mixes
the chain's memory with the model's -- and a trajectory silently collected while
parameters drifted would produce a confident number for a quantity that does not
exist. The collector therefore evaluates in eval mode with every parameter
frozen and, by default, verifies that no parameter moved. It does not disable
autograd: the local energy needs position derivatives, so `torch.no_grad()`
would make the contract's first observable raise rather than run.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from tpen.data.batch.geometry import electron_nuclear_displacements
from tpen.data.batch.walkers import Walkers
from tpen.sampling.stats import SamplerStats
from tpen.events import EventEmitter
from tpen.sampling.protocols import SamplingCollector
from tpen.statistics.trajectory import ObservableTrajectory

__all__ = [
    "ModelDriftError",
    "SamplerDrawDiagnostics",
    "SamplerTrajectoryDiagnostics",
    "collect_observable_trajectory",
    "collect_observable_trajectory_with_diagnostics",
    "parameter_fingerprint",
]


class ModelDriftError(RuntimeError):
    """Raised when model parameters changed during a fixed-model collection."""


@dataclass(frozen=True)
class SamplerDrawDiagnostics:
    """Sampler health observed at one collector-visible walker state.

    The sampler may advance multiple Metropolis steps between collector draws.
    This record therefore describes only the accepted state returned by
    ``collect_samples`` and the aggregate acceptance rate for that call. It
    makes no claim about intermediate accepted states.

    Parameters
    ----------
    collection_index : int
        Zero-based index among every collector call, including discarded draws.
    region_index : int
        Zero-based index inside the discarded or retained region.
    acceptance_rate : float
        Scalar acceptance rate returned for this sampler call.
    n_walkers : int
        Number of persistent walker chains.
    burn_in : int
        Sampler-internal one-time burn-in step count.
    n_steps : int
        Sampler steps between this state and the preceding visible draw.
    proposal_scale : float
        Proposal scale used by the sampler call.
    seed : int or None
        Sampler-reported chain seed, when available.
    minimum_electron_nucleus_radius : float or None
        Raw minimum electron--nucleus radius in this returned walker state.
        ``None`` means the walkers carried no typed atomic configuration.
    """

    collection_index: int
    region_index: int
    acceptance_rate: float
    n_walkers: int
    burn_in: int
    n_steps: int
    proposal_scale: float
    seed: int | None
    minimum_electron_nucleus_radius: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "collection_index", int(self.collection_index))
        object.__setattr__(self, "region_index", int(self.region_index))
        object.__setattr__(self, "acceptance_rate", float(self.acceptance_rate))
        object.__setattr__(self, "n_walkers", int(self.n_walkers))
        object.__setattr__(self, "burn_in", int(self.burn_in))
        object.__setattr__(self, "n_steps", int(self.n_steps))
        object.__setattr__(self, "proposal_scale", float(self.proposal_scale))
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        if self.collection_index < 0 or self.region_index < 0:
            raise ValueError("sampler draw indices must be non-negative")
        if self.n_walkers < 1:
            raise ValueError("sampler draw diagnostics require at least one walker")
        if self.burn_in < 0 or self.n_steps < 1:
            raise ValueError("sampler burn-in must be non-negative and draw stride positive")
        if not 0.0 <= self.acceptance_rate <= 1.0:
            raise ValueError("sampler draw acceptance_rate must be in [0, 1]")
        if not math.isfinite(self.proposal_scale):
            raise ValueError("sampler draw proposal_scale must be finite")
        if self.minimum_electron_nucleus_radius is not None:
            radius = float(self.minimum_electron_nucleus_radius)
            if not math.isfinite(radius) or radius < 0.0:
                raise ValueError("electron-nucleus radii must be finite and non-negative")
            object.__setattr__(self, "minimum_electron_nucleus_radius", radius)

    @property
    def transition_count(self) -> int:
        """Return proposed walker transitions represented by this draw."""

        return self.n_walkers * self.n_steps

    def to_dict(self) -> dict[str, float | int | None]:
        """Return the JSON-safe fields for one sidecar row."""

        return {
            "collection_index": self.collection_index,
            "region_index": self.region_index,
            "acceptance_rate": self.acceptance_rate,
            "n_walkers": self.n_walkers,
            "burn_in": self.burn_in,
            "draw_stride": self.n_steps,
            "transition_count": self.transition_count,
            "proposal_scale": self.proposal_scale,
            "seed": self.seed,
            "minimum_electron_nucleus_radius": self.minimum_electron_nucleus_radius,
        }


@dataclass(frozen=True)
class SamplerTrajectoryDiagnostics:
    """Draw-resolved health for discarded and retained collector regions.

    ``discarded_draws`` are the collector's explicit ``discard_draws`` region,
    after any sampler-internal burn-in. ``retained_draws`` are the states that
    define the observable trajectory. Their minima are never pooled: discarded
    states are pre-measurement diagnostics and cannot contaminate the retained
    distribution statistic.

    The sampler advances ``draw_stride`` steps per visible draw, so accepted
    intermediate states remain unobserved. Consequently the retained minimum is
    explicitly the minimum over retained draws, not the minimum over every
    state reached by the sampler.
    """

    discarded_draws: tuple[SamplerDrawDiagnostics, ...]
    retained_draws: tuple[SamplerDrawDiagnostics, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "discarded_draws", tuple(self.discarded_draws))
        object.__setattr__(self, "retained_draws", tuple(self.retained_draws))
        if not self.retained_draws:
            raise ValueError("sampler trajectory diagnostics require retained draws")
        draws = self.discarded_draws + self.retained_draws
        expected_collection = tuple(range(len(draws)))
        if tuple(draw.collection_index for draw in draws) != expected_collection:
            raise ValueError("sampler trajectory collection indices must be contiguous")
        for region in (self.discarded_draws, self.retained_draws):
            if tuple(draw.region_index for draw in region) != tuple(range(len(region))):
                raise ValueError("sampler trajectory region indices must be contiguous")
        for field_name in ("n_walkers", "burn_in", "n_steps", "proposal_scale"):
            values = {getattr(draw, field_name) for draw in draws}
            if len(values) != 1:
                raise ValueError(f"sampler trajectory {field_name} changed between draws")
        radius_available = {
            draw.minimum_electron_nucleus_radius is not None for draw in draws
        }
        if len(radius_available) != 1:
            raise ValueError(
                "sampler trajectory electron-nucleus geometry availability changed between draws"
            )

    @property
    def n_walkers(self) -> int:
        """Return the persistent chain count."""

        return self.retained_draws[0].n_walkers

    @property
    def draw_stride(self) -> int:
        """Return sampler steps between collector-visible states."""

        return self.retained_draws[0].n_steps

    @property
    def burn_in(self) -> int:
        """Return the sampler-internal one-time burn-in step count."""

        return self.retained_draws[0].burn_in

    @property
    def proposal_scale(self) -> float:
        """Return the proposal scale shared by every draw."""

        return self.retained_draws[0].proposal_scale

    @property
    def intermediate_sampler_steps_observed(self) -> bool:
        """Return ``False`` because only stride-spaced states are visible."""

        return False

    def as_metrics(self) -> dict[str, float | int | bool]:
        """Return scalar trajectory-health metrics without replacing acceptance_rate."""

        retained = self.retained_draws
        discarded = self.discarded_draws
        metrics: dict[str, float | int | bool] = {
            "trajectory_retained_draw_count": len(retained),
            "trajectory_discarded_draw_count": len(discarded),
            "trajectory_n_walkers": self.n_walkers,
            "trajectory_draw_stride": self.draw_stride,
            "trajectory_sampler_burn_in": self.burn_in,
            "trajectory_proposal_scale": self.proposal_scale,
            "trajectory_retained_value_count": len(retained) * self.n_walkers,
            "trajectory_discarded_value_count": len(discarded) * self.n_walkers,
            "trajectory_retained_transition_count": sum(
                draw.transition_count for draw in retained
            ),
            "trajectory_discarded_transition_count": sum(
                draw.transition_count for draw in discarded
            ),
            "trajectory_retained_draw_acceptance_rate_mean": _weighted_acceptance_rate(
                retained
            ),
            "trajectory_intermediate_sampler_steps_observed": False,
        }
        if discarded:
            metrics["trajectory_discarded_draw_acceptance_rate_mean"] = (
                _weighted_acceptance_rate(discarded)
            )
        retained_minimum = _minimum_radius(retained)
        if retained_minimum is not None:
            metrics["trajectory_retained_draw_minimum_electron_nucleus_radius"] = (
                retained_minimum
            )
        discarded_minimum = _minimum_radius(discarded)
        if discarded_minimum is not None:
            metrics["trajectory_discarded_draw_minimum_electron_nucleus_radius"] = (
                discarded_minimum
            )
        return metrics

    def to_dict(self) -> dict[str, object]:
        """Return the versioned draw-series sidecar payload."""

        return {
            "schema": "sampler_trajectory_diagnostics/v1",
            "n_walkers": self.n_walkers,
            "draw_stride": self.draw_stride,
            "sampler_burn_in": self.burn_in,
            "proposal_scale": self.proposal_scale,
            "intermediate_sampler_steps_observed": False,
            "intermediate_sampler_steps_unobserved_reason": (
                "collect_samples returns only the stride-spaced state; intermediate "
                "accepted states between collector-visible draws are unobserved"
            ),
            "sampler_internal_burn_in_states_observed": False,
            "discarded_draw_acceptance_rate_series": [
                draw.acceptance_rate for draw in self.discarded_draws
            ],
            "retained_draw_acceptance_rate_series": [
                draw.acceptance_rate for draw in self.retained_draws
            ],
            "discarded_draws": [draw.to_dict() for draw in self.discarded_draws],
            "retained_draws": [draw.to_dict() for draw in self.retained_draws],
            "metrics": self.as_metrics(),
        }


def parameter_fingerprint(model: torch.nn.Module) -> str:
    """Return a sha256 over a model's parameter and buffer values.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose state should be fingerprinted.

    Returns
    -------
    str
        Hex digest covering every parameter and buffer, in sorted name order,
        including each tensor's name, shape and raw bytes.
    """

    digest = hashlib.sha256()
    named = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in sorted(named, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        values = tensor.detach().cpu().contiguous().to(torch.float64).numpy()
        digest.update(values.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


@contextmanager
def _frozen_eval(model: torch.nn.Module) -> Iterator[None]:
    """Run a block with `model` in eval mode and every parameter frozen.

    Deliberately does NOT use `torch.no_grad()`. The first observable this
    collector must serve is the local energy, whose kinetic term differentiates
    ``logabs`` twice with respect to positions (`tpen.physics.kinetic` calls
    ``requires_grad_(True)`` then ``torch.autograd.grad(..., create_graph=True)``).
    Under `torch.no_grad()` no graph is recorded, so that call raises rather than
    merely running slower, and the energy path -- the observable the acceptance
    contract names first -- becomes unreachable.

    "Fixed model" is a claim about parameters, not about autograd. It is enforced
    here by clearing ``requires_grad`` on every parameter, so no parameter
    gradient can accumulate while the trajectory is collected, and separately by
    the caller's parameter fingerprint check. Position derivatives stay available
    because they are what the observable is entitled to need.

    The caller detaches each draw before retaining it, so the per-draw graph is
    released immediately and collection does not accumulate autograd state.
    """

    was_training = model.training
    parameters = list(model.parameters())
    previously_required = [parameter.requires_grad for parameter in parameters]
    model.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        # Restore per-parameter flags rather than setting them all True: the
        # caller may legitimately have had some frozen before we were invoked.
        for parameter, required in zip(parameters, previously_required):
            parameter.requires_grad_(required)
        model.train(was_training)


def collect_observable_trajectory(
    sampler: SamplingCollector,
    model: torch.nn.Module,
    observable: Callable[[Walkers], torch.Tensor],
    *,
    observable_name: str,
    n_draws: int,
    discard_draws: int = 0,
    device: torch.device | str | None = None,
    reset: bool = False,
    verify_model_unchanged: bool = True,
    emitter: EventEmitter | None,
) -> tuple[ObservableTrajectory, SamplerStats]:
    """Collect a trajectory while preserving the established two-value API.

    Use :func:`collect_observable_trajectory_with_diagnostics` when the caller
    also needs draw-resolved sampler health. Keeping this wrapper avoids a patch
    release break for callers that unpack the original two values.
    """

    trajectory, stats, _ = collect_observable_trajectory_with_diagnostics(
        sampler,
        model,
        observable,
        observable_name=observable_name,
        n_draws=n_draws,
        discard_draws=discard_draws,
        device=device,
        reset=reset,
        verify_model_unchanged=verify_model_unchanged,
        emitter=emitter,
    )
    return trajectory, stats


def collect_observable_trajectory_with_diagnostics(
    sampler: SamplingCollector,
    model: torch.nn.Module,
    observable: Callable[[Walkers], torch.Tensor],
    *,
    observable_name: str,
    n_draws: int,
    discard_draws: int = 0,
    device: torch.device | str | None = None,
    reset: bool = False,
    verify_model_unchanged: bool = True,
    emitter: EventEmitter | None,
) -> tuple[ObservableTrajectory, SamplerStats, SamplerTrajectoryDiagnostics]:
    """Collect a `[draw, walker]` observable trajectory from a fixed model.

    Each draw advances the sampler's persistent chain by its configured
    ``n_steps`` and evaluates `observable` on the resulting walkers. Walker
    columns are kept separate for the whole collection: they are independent
    chains and the statistics layer must never see them concatenated.

    Parameters
    ----------
    sampler : SamplingCollector
        Typed sampler exposing ``collect_samples(model, *, reset=..., device=..., emitter=...)``
        returning ``(Walkers, SamplerStats)``.
    model : torch.nn.Module
        Wavefunction model, held fixed for the whole collection.
    observable : callable
        Maps one `Walkers` state to a ``[n_walkers]`` tensor of per-walker
        values. Autocorrelation is observable-specific, so exactly one
        observable is collected per call.
    observable_name : str
        Name recorded on the trajectory, for example ``"local_energy"``.
    n_draws : int
        Number of draws to retain per walker.
    discard_draws : int, optional
        Draws to collect and discard before retention begins, on top of the
        sampler's own burn-in. Recorded on the trajectory as ``burn_in_draws``.
    device : torch.device, str, or None, optional
        Forwarded to the sampler.
    reset : bool, optional
        Force a fresh, re-seeded, re-burned-in chain before the first draw.
    verify_model_unchanged : bool, optional
        Fingerprint parameters before and after collection and raise if they
        differ. Defaults to ``True``.

    Returns
    -------
    trajectory : ObservableTrajectory
        Retained samples with explicit ``[draw, walker]`` axes. ``draw_stride``
        is the sampler's own ``n_steps`` between draws, read from the typed
        `SamplerStats` record rather than probed off the sampler object.
    stats : SamplerStats
        The typed record from the final draw.
    diagnostics : SamplerTrajectoryDiagnostics
        Draw-resolved acceptance, transitions, and observed-state geometry,
        partitioned into discarded and retained regions.

    Raises
    ------
    TypeError
        If `observable` does not return a tensor.
    AttributeError
        If `sampler` lacks ``collect_samples`` and runtime type checking is
        disabled. The argument must satisfy the `SamplingCollector` contract;
        when enabled, type checking rejects an incompatible sampler first.
    ValueError
        If `n_draws` is below one, `discard_draws` is negative, or the
        observable's shape does not match the walker count.
    ModelDriftError
        If `verify_model_unchanged` is set and any parameter moved.
    """

    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")
    if discard_draws < 0:
        raise ValueError(f"discard_draws must be non-negative, got {discard_draws}")

    fingerprint_before = parameter_fingerprint(model) if verify_model_unchanged else None

    columns: list[torch.Tensor] = []
    discarded_diagnostics: list[SamplerDrawDiagnostics] = []
    retained_diagnostics: list[SamplerDrawDiagnostics] = []
    stats: SamplerStats | None = None
    with _frozen_eval(model):
        for draw_index in range(discard_draws + n_draws):
            # `reset` applies to the first call only: re-seeding mid-collection
            # would restart the chain and destroy the draw axis being built.
            walkers, stats = sampler.collect_samples(
                model,
                reset=reset and draw_index == 0,
                device=device,
                emitter=emitter,
            )
            if not isinstance(stats, SamplerStats):
                raise TypeError(
                    "sampler.collect_samples must return a SamplerStats record, "
                    f"got {type(stats).__name__}"
                )
            if stats.n_steps < 1:
                raise ValueError(
                    f"sampler advances {stats.n_steps} steps per draw; consecutive draws "
                    "would be identical and the draw axis would carry no information"
                )
            values = observable(walkers)
            if not isinstance(values, torch.Tensor):
                raise TypeError(
                    f"observable must return a torch.Tensor, got {type(values).__name__}"
                )
            values = values.detach().reshape(-1)
            if values.numel() != walkers.batch_size:
                raise ValueError(
                    f"observable returned {values.numel()} values for {walkers.batch_size} "
                    "walkers; one value per walker is required to keep chains separate"
                )
            if draw_index < discard_draws:
                discarded_diagnostics.append(
                    _draw_diagnostics(
                        walkers,
                        stats,
                        collection_index=draw_index,
                        region_index=draw_index,
                    )
                )
            else:
                retained_diagnostics.append(
                    _draw_diagnostics(
                        walkers,
                        stats,
                        collection_index=draw_index,
                        region_index=draw_index - discard_draws,
                    )
                )
                columns.append(values.to(torch.float64).cpu())

    if verify_model_unchanged:
        fingerprint_after = parameter_fingerprint(model)
        if fingerprint_after != fingerprint_before:
            raise ModelDriftError(
                "model parameters changed during trajectory collection; "
                "autocorrelation over a moving target distribution is undefined. "
                f"before={fingerprint_before} after={fingerprint_after}"
            )

    # The loop above runs at least once because n_draws >= 1 was validated.
    if stats is None:
        raise TypeError("sampler.collect_samples did not return a SamplerStats record")
    if stats.n_steps < 1:
        raise ValueError(
            f"sampler advances {stats.n_steps} steps per draw; consecutive draws would be "
            "identical and the draw axis would carry no information"
        )
    trajectory = ObservableTrajectory(
        observable=observable_name,
        values=torch.stack(columns, dim=0),
        draw_stride=stats.n_steps,
        burn_in_draws=discard_draws,
    )
    diagnostics = SamplerTrajectoryDiagnostics(
        discarded_draws=tuple(discarded_diagnostics),
        retained_draws=tuple(retained_diagnostics),
    )
    return trajectory, stats, diagnostics


def _draw_diagnostics(
    walkers: Walkers,
    stats: SamplerStats,
    *,
    collection_index: int,
    region_index: int,
) -> SamplerDrawDiagnostics:
    """Build one typed diagnostic from the exact returned walker state."""

    if stats.reported_n_walkers != walkers.batch_size:
        raise ValueError(
            "sampler stats walker count disagrees with the returned walker state: "
            f"stats={stats.reported_n_walkers}, walkers={walkers.batch_size}"
        )
    minimum_radius: float | None = None
    if walkers.atomic_configuration is not None and walkers.positions.numel() > 0:
        # Raw geometry: no epsilon floor, threshold, or near-miss surrogate.
        distances = electron_nuclear_displacements(walkers.make_batch()).norm(dim=-1)
        if distances.numel() > 0:
            minimum_radius = float(distances.min().item())
    return SamplerDrawDiagnostics(
        collection_index=collection_index,
        region_index=region_index,
        acceptance_rate=stats.acceptance_rate,
        n_walkers=stats.reported_n_walkers,
        burn_in=stats.burn_in,
        n_steps=stats.n_steps,
        proposal_scale=stats.proposal_scale,
        seed=stats.seed,
        minimum_electron_nucleus_radius=minimum_radius,
    )


def _weighted_acceptance_rate(draws: tuple[SamplerDrawDiagnostics, ...]) -> float:
    """Return acceptance weighted by the represented transition count."""

    transitions = sum(draw.transition_count for draw in draws)
    if transitions < 1:
        raise ValueError("acceptance aggregation requires at least one transition")
    return sum(draw.acceptance_rate * draw.transition_count for draw in draws) / transitions


def _minimum_radius(draws: tuple[SamplerDrawDiagnostics, ...]) -> float | None:
    """Return the minimum raw radius in one region, preserving unavailability."""

    if not draws or draws[0].minimum_electron_nucleus_radius is None:
        return None
    return min(
        float(draw.minimum_electron_nucleus_radius)
        for draw in draws
        if draw.minimum_electron_nucleus_radius is not None
    )
