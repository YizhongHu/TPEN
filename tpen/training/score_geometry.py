"""Frozen score geometry: raw parameter-score blocks to a VMC design matrix.

This module owns the *matrix view* of a VMC step.  The payload type
:class:`~tpen.data.batch.MaterializedParameterLogScores` owns raw,
parameter-shaped, uncentered score blocks; nothing about a design matrix,
centering, or normalization belongs to that payload.  Turning those blocks
into the ``B x P`` object a quantum-geometric-tensor solve consumes is a
geometry concern, so it lives here.

The point of a single frozen geometry is that every downstream consumer --
dense parameter-space SR, sample-space minSR, and later a projected-history
method -- inherits *identical* centering, normalization, dtype, and sign
conventions.  Two consumers that each rolled their own centering would agree
on tiny examples and disagree exactly where it is expensive to notice.

Conventions frozen here
-----------------------
Scores are derivatives of the real log amplitude,
``O[k, i] = d log|psi(x_k)| / d theta_i``.  This is the sign convention
already produced by
:meth:`~tpen.nn.TPENWaveFunction.evaluate_materialized_parameter_score_request`;
it is restated rather than re-derived.

Centering subtracts a mean built from an explicitly reduced total and count,
never from a local ``.mean()`` call.  Normalization is ``1 / sqrt(count)`` on
the centered rows.  Writing ``Obar`` for centered scores and ``N`` for the
global count, the design matrix is

.. math::

    A = \\bar{O} / \\sqrt{N}, \\qquad
    S = A^{T} A = \\langle O_i O_j \\rangle - \\langle O_i \\rangle \\langle O_j \\rangle,

so ``S`` is the empirical quantum geometric tensor for a real wavefunction
with no further scaling.  The energy residual uses the same count and the
same reduced-total centering,

.. math::

    \\varepsilon_k = c \\, (E_k - \\bar{E}) / \\sqrt{N},

with ``c`` the energy-gradient scale.  With the default ``c = 2`` this makes
``A^T epsilon`` exactly the ordinary VMC energy gradient produced by
:func:`~tpen.training.vmc.compute_vmc_objective`, whose surrogate objective
carries the same factor of two.  That identity is what lets the Euclidean
limit of an SR step be compared against ordinary VMC without a fudge factor.

Distributed readiness
---------------------
Centering routes through the existing
:class:`~tpen.training.statistics.StatisticsReducer` seam rather than calling
``.mean()``.  Under the single-process
:class:`~tpen.training.statistics.IdentityStatisticsReducer` this is exactly
the local mean, so nothing changes today.  The reason to pay for the seam now
is that a later count-aware distributed reducer then makes centering correct
by construction: the mean is built from a global sum and a global count, so
ranks holding different sample counts cannot silently produce an unweighted
mean of per-rank means.  This module never creates a process group and never
performs a collective itself; it only asks the injected reducer for one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import prod
from typing import Any, Self

from tpen.data.batch import MaterializedParameterLogScores, ParameterLayout
from tpen.dependencies import require_torch
from tpen.training.statistics import (
    IdentityStatisticsReducer,
    StatisticsReducer,
    StatisticsSums,
    center_statistics,
)
from tpen.training.update import serialize_parameter_layout

torch = require_torch(feature="VMC score geometry")


# Bump when any frozen convention in this module changes meaning.  A method
# state envelope records this string so a resumed run cannot silently reuse
# a warm start built under different centering or normalization.
SCORE_CONVENTION_VERSION = "score-geometry-1"


@dataclass(frozen=True, kw_only=True)
class ScoreConventions:
    """Frozen numerical conventions shared by every score consumer.

    Parameters
    ----------
    energy_gradient_scale : float, optional
        Multiplicative factor ``c`` on the centered local-energy residual.
        The default ``2.0`` matches
        :func:`~tpen.training.vmc.compute_vmc_objective`, so ``A^T epsilon``
        reproduces the ordinary VMC energy gradient exactly.
    solve_dtype : torch.dtype or str or None, optional
        Real floating dtype the design matrix and residual are promoted to
        before any linear algebra.  ``None`` keeps the incoming score dtype.
        A dense reference is normally run in ``torch.float64``: the QGT is a
        Gram matrix and therefore squares the condition number of the scores,
        which is the one place in this engine where the extra precision is
        load-bearing rather than defensive.  A bare dtype name such as
        ``"float64"`` is accepted so a YAML config can spell it the same way
        it spells ``runtime.dtype``.

    Notes
    -----
    Centering is deliberately not optional.  An uncentered mode would be a
    second, quietly wrong geometry that agrees with this one only when the
    score mean happens to vanish.
    """

    energy_gradient_scale: float = 2.0
    solve_dtype: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_gradient_scale", float(self.energy_gradient_scale))
        # A config supplies a dtype as the bare name "float64", matching how
        # `runtime.dtype` is spelled everywhere else in this project's YAML.
        # Coercing with getattr(torch, name) is the convention already used by
        # the sampler, the evaluator, and checkpoint restore.
        if isinstance(self.solve_dtype, str):
            resolved = getattr(torch, self.solve_dtype, None)
            if not isinstance(resolved, torch.dtype):
                raise ValueError(
                    f"ScoreConventions.solve_dtype {self.solve_dtype!r} is not a torch dtype name"
                )
            object.__setattr__(self, "solve_dtype", resolved)
        self.validate()

    def validate(self) -> Self:
        """Validate the scale factor and the optional solve dtype."""

        scale = self.energy_gradient_scale
        if scale != scale or scale in (float("inf"), float("-inf")):
            raise ValueError("ScoreConventions.energy_gradient_scale must be finite")
        if scale == 0.0:
            raise ValueError("ScoreConventions.energy_gradient_scale must be nonzero")
        if self.solve_dtype is not None:
            if not isinstance(self.solve_dtype, torch.dtype):
                raise TypeError("ScoreConventions.solve_dtype must be a torch.dtype or None")
            if not self.solve_dtype.is_floating_point:
                raise TypeError("ScoreConventions.solve_dtype must be a real floating dtype")
            if self.solve_dtype.is_complex:
                raise TypeError("ScoreConventions.solve_dtype must not be complex")
        return self

    def fingerprint(self) -> dict[str, Any]:
        """Return JSON-safe convention metadata for a state envelope."""

        return {
            "version": SCORE_CONVENTION_VERSION,
            "energy_gradient_scale": self.energy_gradient_scale,
            "solve_dtype": None if self.solve_dtype is None else str(self.solve_dtype),
        }


@dataclass(frozen=True, kw_only=True)
class ScoreGeometry:
    """A centered, normalized design matrix and its parameter layout.

    Parameters
    ----------
    design : torch.Tensor
        The ``[count, total_numel]`` matrix ``A = centered_scores / sqrt(count)``.
        Column ``j`` corresponds to flat parameter coordinate ``j`` in layout
        slot order.
    layout : ParameterLayout
        Layout the columns were flattened from.
    count : int
        The reduced global sample count used for both centering and
        normalization.  Under a distributed reducer this exceeds the number
        of local rows in `design`.
    conventions : ScoreConventions
        The conventions this geometry was built under.

    Notes
    -----
    ``design`` holds *rank-local rows* against a *global* count.  Keeping
    those two facts in one object is deliberate: a consumer that later gathers
    rows across ranks must not re-derive the count from ``design.shape[0]``.
    """

    design: Any
    layout: ParameterLayout
    count: int
    conventions: ScoreConventions

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_local_samples(self) -> int:
        """Return the number of score rows held locally."""

        return int(self.design.shape[0])

    @property
    def n_parameters(self) -> int:
        """Return the flat parameter dimension ``P``."""

        return int(self.design.shape[1])

    @property
    def dtype(self) -> Any:
        """Return the design-matrix dtype."""

        return self.design.dtype

    @property
    def device(self) -> Any:
        """Return the design-matrix device."""

        return self.design.device

    def validate(self) -> Self:
        """Validate matrix rank-2 shape, layout agreement, and count."""

        if not isinstance(self.design, torch.Tensor):
            raise TypeError("ScoreGeometry.design must be a torch.Tensor")
        if not self.design.is_floating_point():
            raise TypeError("ScoreGeometry.design must have a real floating dtype")
        if self.design.ndim != 2:
            raise ValueError(
                f"ScoreGeometry.design must be a matrix, got shape {tuple(self.design.shape)}"
            )
        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("ScoreGeometry.layout must be a ParameterLayout")
        self.layout.validate()
        if int(self.design.shape[1]) != self.layout.total_numel:
            raise ValueError(
                "ScoreGeometry.design must have one column per flat layout coordinate, "
                f"got {int(self.design.shape[1])} for layout total_numel "
                f"{self.layout.total_numel}"
            )
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("ScoreGeometry.count must be a positive integer")
        if not isinstance(self.conventions, ScoreConventions):
            raise TypeError("ScoreGeometry.conventions must be a ScoreConventions")
        self.conventions.validate()
        return self

    def fingerprint(self) -> dict[str, Any]:
        """Return JSON-safe layout-plus-convention metadata."""

        return layout_convention_fingerprint(self.layout, self.conventions)

    def unflatten(self, vector: Any) -> tuple[Any, ...]:
        """Split a flat ``P``-vector into parameter-shaped blocks.

        Parameters
        ----------
        vector : torch.Tensor
            Flat tensor with ``layout.total_numel`` elements.

        Returns
        -------
        tuple of torch.Tensor
            One block per layout slot, each with that slot's exact shape.
        """

        return unflatten_to_layout(vector, layout=self.layout)


def flatten_parameter_score_blocks(
    scores: MaterializedParameterLogScores,
    *,
    sample_shape: tuple[int, ...] | None = None,
) -> Any:
    """Flatten parameter-shaped score blocks into one ``[B, P]`` matrix.

    Parameters
    ----------
    scores : MaterializedParameterLogScores
        Raw, uncentered score blocks.  Block ``i`` has shape
        ``[*sample_shape, *layout.slots[i].shape]``.
    sample_shape : tuple of int, optional
        Expected leading sample shape.  When given it is enforced, which
        catches a caller that paired scores with a different forward pass.

    Returns
    -------
    torch.Tensor
        Matrix with ``B = prod(sample_shape)`` rows and one column per flat
        parameter coordinate, in layout slot order.

    Notes
    -----
    Column order is the layout's slot order followed by each slot's row-major
    element order.  That ordering is the only thing making a flat ``P``-vector
    meaningful, so it is fixed here and reused by :func:`unflatten_to_layout`
    rather than being re-implemented per consumer.
    """

    if not isinstance(scores, MaterializedParameterLogScores):
        raise TypeError(
            "flatten_parameter_score_blocks requires MaterializedParameterLogScores"
        )
    scores.validate(sample_shape=sample_shape)
    resolved_shape = scores.sample_shape if sample_shape is None else tuple(sample_shape)
    n_samples = int(prod(resolved_shape)) if resolved_shape else 1
    if not scores.blocks:
        raise ValueError(
            "flatten_parameter_score_blocks requires at least one parameter slot"
        )
    columns = [
        block.reshape(n_samples, slot.numel)
        for slot, block in zip(scores.layout.slots, scores.blocks, strict=True)
    ]
    return torch.cat(columns, dim=1)


def unflatten_to_layout(vector: Any, *, layout: ParameterLayout) -> tuple[Any, ...]:
    """Split a flat ``P``-vector into layout-shaped blocks.

    This is the exact inverse of the column ordering used by
    :func:`flatten_parameter_score_blocks`.

    Parameters
    ----------
    vector : torch.Tensor
        Flat tensor with ``layout.total_numel`` elements.
    layout : ParameterLayout
        Target layout.

    Returns
    -------
    tuple of torch.Tensor
        One block per slot, each with that slot's exact shape.
    """

    if not isinstance(vector, torch.Tensor):
        raise TypeError("unflatten_to_layout.vector must be a torch.Tensor")
    if not isinstance(layout, ParameterLayout):
        raise TypeError("unflatten_to_layout.layout must be a ParameterLayout")
    layout.validate()
    if vector.ndim != 1:
        raise ValueError(
            f"unflatten_to_layout.vector must be flat, got shape {tuple(vector.shape)}"
        )
    if int(vector.numel()) != layout.total_numel:
        raise ValueError(
            "unflatten_to_layout.vector must have one element per flat layout coordinate, "
            f"got {int(vector.numel())} for layout total_numel {layout.total_numel}"
        )
    blocks: list[Any] = []
    offset = 0
    for slot in layout.slots:
        blocks.append(vector[offset : offset + slot.numel].reshape(slot.shape))
        offset += slot.numel
    return tuple(blocks)


def layout_convention_fingerprint(
    layout: ParameterLayout,
    conventions: ScoreConventions,
) -> dict[str, Any]:
    """Return a JSON-safe fingerprint of a layout and its conventions.

    The digest lets a resumed run reject a checkpoint whose parameter layout
    or numerical conventions no longer match, without storing the whole
    layout twice.  It reuses
    :func:`~tpen.training.update.serialize_parameter_layout` so a layout has
    exactly one serialized form in the codebase.
    """

    if not isinstance(conventions, ScoreConventions):
        raise TypeError("layout_convention_fingerprint requires a ScoreConventions")
    serialized_layout = serialize_parameter_layout(layout)
    payload = {
        "layout": serialized_layout,
        "conventions": conventions.fingerprint(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "total_numel": layout.total_numel,
        "n_slots": len(layout.slots),
        "conventions": conventions.fingerprint(),
        "digest": digest,
    }


def build_score_geometry(
    scores: MaterializedParameterLogScores,
    *,
    sample_shape: tuple[int, ...] | None = None,
    conventions: ScoreConventions | None = None,
    reducer: StatisticsReducer | None = None,
) -> ScoreGeometry:
    """Center and normalize raw score blocks into a design matrix.

    Parameters
    ----------
    scores : MaterializedParameterLogScores
        Raw, uncentered, rank-local score blocks.
    sample_shape : tuple of int, optional
        Expected leading sample shape, enforced when supplied.
    conventions : ScoreConventions, optional
        Frozen conventions.  Defaults to :class:`ScoreConventions`.
    reducer : StatisticsReducer, optional
        Reducer supplying the global count and score sum.  Defaults to
        :class:`~tpen.training.statistics.IdentityStatisticsReducer`.

    Returns
    -------
    ScoreGeometry
        The centered, ``1/sqrt(count)``-normalized design matrix.

    Raises
    ------
    ValueError
        If the reduced count is not positive, or the layout is empty.
    """

    rows = flatten_parameter_score_blocks(scores, sample_shape=sample_shape)
    return build_score_geometry_from_rows(
        rows,
        layout=scores.layout,
        conventions=conventions,
        reducer=reducer,
    )


def build_score_geometry_from_rows(
    rows: Any,
    *,
    layout: ParameterLayout,
    conventions: ScoreConventions | None = None,
    reducer: StatisticsReducer | None = None,
) -> ScoreGeometry:
    """Center and normalize an already-flattened ``[B, P]`` score matrix.

    This is the constructor to use when a consumer must select a subset of
    rows -- dropping samples with a non-finite local energy, for instance --
    before centering.  Selecting rows after centering would center against a
    mean that includes the very rows being discarded.

    Parameters
    ----------
    rows : torch.Tensor
        Raw, uncentered ``[B, P]`` score matrix in layout column order.
    layout : ParameterLayout
        Layout the columns correspond to.
    conventions : ScoreConventions, optional
        Frozen conventions.  Defaults to :class:`ScoreConventions`.
    reducer : StatisticsReducer, optional
        Reducer supplying the global count and score sum.  Defaults to
        :class:`~tpen.training.statistics.IdentityStatisticsReducer`.

    Returns
    -------
    ScoreGeometry
        The centered, ``1/sqrt(count)``-normalized design matrix.
    """

    resolved_conventions = ScoreConventions() if conventions is None else conventions
    resolved_conventions.validate()
    resolved_reducer = IdentityStatisticsReducer() if reducer is None else reducer
    if not isinstance(resolved_reducer, StatisticsReducer):
        raise TypeError("build_score_geometry_from_rows.reducer must be a StatisticsReducer")
    if not isinstance(rows, torch.Tensor):
        raise TypeError("build_score_geometry_from_rows.rows must be a torch.Tensor")
    if rows.ndim != 2:
        raise ValueError(
            "build_score_geometry_from_rows.rows must be a matrix, got shape "
            f"{tuple(rows.shape)}"
        )

    if resolved_conventions.solve_dtype is not None:
        rows = rows.to(dtype=resolved_conventions.solve_dtype)

    # The reducer owns the count and the sum; centering stays here because a
    # reducer must never see raw score rows.  Under IdentityStatisticsReducer
    # this is the local mean; under a count-aware distributed reducer it is
    # the global mean, with no change at this call site.
    local = StatisticsSums(count=int(rows.shape[0]), sums=(rows.sum(dim=0),))
    reduced = resolved_reducer.reduce(local)
    count = reduced.count
    if count <= 0:
        raise ValueError("build_score_geometry_from_rows requires a positive reduced count")

    centered = center_statistics(rows, count=count, total=reduced.sums[0])
    design = centered / (float(count) ** 0.5)
    return ScoreGeometry(
        design=design,
        layout=layout,
        count=count,
        conventions=resolved_conventions,
    )


def build_energy_residual(
    local_energy: Any,
    *,
    geometry: ScoreGeometry,
    reducer: StatisticsReducer | None = None,
) -> Any:
    """Center and normalize local energies into the residual vector.

    Parameters
    ----------
    local_energy : torch.Tensor
        Per-sample local energies.  Any sample shape is accepted and
        flattened; its element count must match the geometry's local rows.
    geometry : ScoreGeometry
        Geometry supplying the count, dtype, device, and conventions.  Using
        the geometry's own count is what keeps ``A`` and ``epsilon`` on one
        normalization.
    reducer : StatisticsReducer, optional
        Reducer supplying the global count and energy sum.  Defaults to
        :class:`~tpen.training.statistics.IdentityStatisticsReducer`.

    Returns
    -------
    torch.Tensor
        Flat residual ``epsilon = scale * (E - Ebar) / sqrt(count)``, such
        that ``geometry.design.T @ epsilon`` is the ordinary VMC energy
        gradient under the default scale.

    Raises
    ------
    ValueError
        If the sample count disagrees with the geometry, or the reduced count
        differs from the geometry's count.
    """

    if not isinstance(local_energy, torch.Tensor):
        raise TypeError("build_energy_residual.local_energy must be a torch.Tensor")
    if not local_energy.is_floating_point():
        raise TypeError("build_energy_residual.local_energy must have a real floating dtype")
    if not isinstance(geometry, ScoreGeometry):
        raise TypeError("build_energy_residual.geometry must be a ScoreGeometry")
    resolved_reducer = IdentityStatisticsReducer() if reducer is None else reducer
    if not isinstance(resolved_reducer, StatisticsReducer):
        raise TypeError("build_energy_residual.reducer must be a StatisticsReducer")

    values = local_energy.reshape(-1).to(dtype=geometry.dtype, device=geometry.device)
    if int(values.numel()) != geometry.n_local_samples:
        raise ValueError(
            "build_energy_residual requires one local energy per local score row, "
            f"got {int(values.numel())} energies for {geometry.n_local_samples} rows"
        )

    local = StatisticsSums(count=int(values.numel()), sums=(values.sum(),))
    reduced = resolved_reducer.reduce(local)
    if reduced.count != geometry.count:
        raise ValueError(
            "build_energy_residual reduced count must match the score geometry count, "
            f"got {reduced.count} and {geometry.count}"
        )

    centered = center_statistics(values, count=reduced.count, total=reduced.sums[0])
    scale = geometry.conventions.energy_gradient_scale
    return centered * (scale / (float(reduced.count) ** 0.5))


__all__ = [
    "SCORE_CONVENTION_VERSION",
    "ScoreConventions",
    "ScoreGeometry",
    "build_energy_residual",
    "build_score_geometry",
    "build_score_geometry_from_rows",
    "flatten_parameter_score_blocks",
    "layout_convention_fingerprint",
    "unflatten_to_layout",
]
