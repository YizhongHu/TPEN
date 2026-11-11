"""Quantum-geometric-tensor actions, dense forms, damping, and dense solves.

This module owns the linear algebra of the empirical quantum geometric tensor
(QGT) for a real wavefunction, against the frozen design matrix produced by
:mod:`tpen.training.score_geometry`.  It owns no optimizer, no parameters, and
no training loop; those belong to the update method.

Two equivalent routes to one update
-----------------------------------
Writing ``A`` for the ``[B, P]`` design matrix, ``epsilon`` for the normalized
energy residual, ``g = A^T epsilon`` for the energy gradient, and ``lambda``
for the damping shift:

*Parameter space* forms ``S = A^T A`` (``P x P``) and solves
``(S + lambda I_P) delta = g``.

*Sample space* forms the Gram matrix ``T = A A^T`` (``B x B``), solves
``(T + lambda I_B) y = epsilon``, and maps back with ``delta = A^T y``.

These are the same vector, by the push-through identity
``(A^T A + lambda I)^{-1} A^T = A^T (A A^T + lambda I)^{-1}``.  Sample space is
the cheaper route precisely when ``B << P``, which is the ordinary VMC regime;
this is the "minSR" route.  Both are implemented because their agreement is
the sharpest available check that the conventions are right: they are
algebraically equivalent but structurally different, so a centering,
normalization, or transpose error breaks the agreement.

The equivalence is only exact if both routes use an *identical* scalar shift,
which is why :class:`DampingPolicy` anchors its relative term to a single
quantity (see that class).  Deriving the relative shift from
``mean(diag(S))`` in one route and ``mean(diag(T))`` in the other would divide
the same trace by ``P`` and by ``B``, silently making the two routes different
algorithms that agree only when ``B == P``.

Rank truncation is likewise safe in both routes.  ``S`` and ``T`` share their
nonzero spectrum.  In parameter space ``g = A^T epsilon`` lies in the range of
``A^T``, so ``S``'s null modes carry no component of ``g``.  In sample space a
null mode ``y`` of ``T`` satisfies ``||A^T y||^2 = y^T A A^T y = 0``, so
``A^T`` annihilates it.  Discarding null modes therefore changes neither
route's answer, and a matched relative cutoff retains the same modes in both.

Distributed readiness
---------------------
Actions are the primary interface and dense matrices are a convenience built
on them, because the action form is what a later sample-sharded distributed
solve needs: ``A^T u`` is a ``P``-element sum reduction while ``A v`` stays
row-local.  Every cross-sample sum here therefore goes through the injected
:class:`~tpen.training.statistics.StatisticsReducer` rather than being assumed
local.  This module never creates a process group.

One asymmetry is deliberate and documented on the method: the sample-space
Gram is defined on the rows the operator actually holds.  A globally correct
Gram under sharding needs cross-rank score rows, which is a gather this lane
does not own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from tpen.dependencies import require_torch
from tpen.training.score_geometry import ScoreGeometry
from tpen.training.update import json_safe_metric_name, json_safe_scalar
from tpen.training.statistics import IdentityStatisticsReducer, StatisticsReducer

torch = require_torch(feature="VMC quantum geometric tensor")


@dataclass(frozen=True, kw_only=True)
class DampingPolicy:
    """Regularization shift added to the QGT before a solve.

    Parameters
    ----------
    absolute : float, optional
        Constant shift added to the diagonal.
    relative : float, optional
        Shift proportional to ``trace(S) / P``, the mean parameter-space
        diagonal entry.  This makes the regularization scale-aware: doubling
        every score doubles the trace and doubles this term.
    minimum : float, optional
        Floor applied to the resulting shift.  A strictly positive floor is
        the simplest way to keep a solve well posed when the trace collapses,
        for example on a step whose scores are nearly constant.

    Notes
    -----
    The relative term is anchored to ``trace(S) / P`` in *both* the
    parameter-space and the sample-space route, even though the sample-space
    matrix is ``B x B``.  The shift is a property of the regularized problem,
    not of the matrix chosen to represent it; anchoring it per-route would
    make the two routes disagree whenever ``B != P``.
    """

    absolute: float = 0.0
    relative: float = 1.0e-3
    minimum: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "absolute", float(self.absolute))
        object.__setattr__(self, "relative", float(self.relative))
        object.__setattr__(self, "minimum", float(self.minimum))
        self.validate()

    def validate(self) -> Self:
        """Validate that every term is finite and non-negative."""

        for name in ("absolute", "relative", "minimum"):
            value = getattr(self, name)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"DampingPolicy.{name} must be finite")
            if value < 0.0:
                raise ValueError(f"DampingPolicy.{name} must be non-negative")
        return self

    def shift(self, *, trace: float, n_parameters: int) -> float:
        """Return the scalar diagonal shift for one solve.

        Parameters
        ----------
        trace : float
            ``trace(S) = ||A||_F^2``, already reduced across every sample.
        n_parameters : int
            The flat parameter dimension ``P``.
        """

        if type(n_parameters) is not int or n_parameters <= 0:
            raise ValueError("DampingPolicy.shift requires a positive parameter count")
        trace = float(trace)
        if trace != trace or trace < 0.0:
            raise ValueError("DampingPolicy.shift requires a non-negative finite trace")
        return max(self.absolute + self.relative * (trace / n_parameters), self.minimum)

    def fingerprint(self) -> dict[str, float]:
        """Return JSON-safe damping metadata for telemetry or a state envelope."""

        return {
            "absolute": self.absolute,
            "relative": self.relative,
            "minimum": self.minimum,
        }


@dataclass(frozen=True, kw_only=True)
class SolveDiagnostics:
    """Observable outcome of one regularized QGT solve.

    Parameters
    ----------
    space : str
        ``"parameter"`` or ``"sample"``, the route actually taken.
    shift : float
        The scalar diagonal shift applied.
    trace : float
        ``trace(S)``, reduced across samples.
    n_modes : int
        Eigenmodes available before truncation: ``P`` or ``B``.
    retained_modes : int
        Eigenmodes kept after relative truncation.
    max_eigenvalue : float
        Largest eigenvalue of the undamped matrix.
    min_retained_eigenvalue : float
        Smallest retained eigenvalue of the undamped matrix, or ``0.0`` when
        nothing was retained.
    dtype : str
        The dtype the solve ACTUALLY ran in, read off the matrix rather than
        taken from the configuration.

        This is deliberately an observed value and not a restatement of
        :attr:`~tpen.training.score_geometry.ScoreConventions.solve_dtype`.
        The configured value already reaches the method-state fingerprint, but
        a configuration records an intention: it cannot show that some future
        edit did not quietly downcast on the way here. A silent precision
        reduction inside a preconditioner does not raise -- it degrades the
        optimization trajectory, which surfaces as bad physics or a suspect
        hyperparameter and is debugged far from its cause. Reporting what ran
        makes that observable per step.
    """

    space: str
    shift: float
    trace: float
    n_modes: int
    retained_modes: int
    max_eigenvalue: float
    min_retained_eigenvalue: float
    dtype: str

    @property
    def truncated_modes(self) -> int:
        """Return how many eigenmodes the relative cutoff discarded."""

        return self.n_modes - self.retained_modes

    def as_metrics(self, *, prefix: str = "qgt") -> dict[str, Any]:
        """Return JSON-safe telemetry keys for a training metrics record.

        The docstring's "JSON-safe" claim used to be FALSE for exactly the
        solves worth recording. These values come out of an eigendecomposition
        of a regularized QGT: an ill-conditioned or overflowing solve can make
        ``trace``, ``max_eigenvalue`` or ``shift`` non-finite, and a bare
        ``float()`` passed that straight to the JSONL sink, which calls
        ``json.dumps(..., allow_nan=False)`` and RAISES. A diagnostic recorded
        to explain a bad solve therefore killed the run that produced it.

        Routing through :func:`~tpen.training.update.json_safe_scalar` makes a
        finite value pass through unchanged -- so no existing record changes --
        while a non-finite one is NAMED rather than crashing the sink or being
        fabricated into a zero.
        """

        prefix = json_safe_metric_name(prefix)

        return {
            f"{prefix}_space": json_safe_scalar(self.space),
            f"{prefix}_shift": json_safe_scalar(self.shift),
            f"{prefix}_trace": json_safe_scalar(self.trace),
            f"{prefix}_modes": int(self.n_modes),
            f"{prefix}_retained_modes": int(self.retained_modes),
            f"{prefix}_truncated_modes": int(self.truncated_modes),
            f"{prefix}_max_eigenvalue": json_safe_scalar(self.max_eigenvalue),
            f"{prefix}_min_retained_eigenvalue": json_safe_scalar(
                self.min_retained_eigenvalue
            ),
            f"{prefix}_dtype": json_safe_scalar(str(self.dtype)),
        }


class QGTOperator:
    """Actions and dense forms of the empirical QGT for one score geometry.

    Parameters
    ----------
    geometry : ScoreGeometry
        Centered, normalized design matrix and its layout.
    reducer : StatisticsReducer, optional
        Reducer used for every cross-sample sum.  Defaults to
        :class:`~tpen.training.statistics.IdentityStatisticsReducer`.

    Notes
    -----
    The operator holds the geometry by reference and is valid only for the one
    step that produced it.  It caches nothing across steps, so there is no
    stale-curvature failure mode to reason about.
    """

    def __init__(
        self,
        geometry: ScoreGeometry,
        *,
        reducer: StatisticsReducer | None = None,
    ) -> None:
        if not isinstance(geometry, ScoreGeometry):
            raise TypeError("QGTOperator.geometry must be a ScoreGeometry")
        geometry.validate()
        resolved_reducer = IdentityStatisticsReducer() if reducer is None else reducer
        if not isinstance(resolved_reducer, StatisticsReducer):
            raise TypeError("QGTOperator.reducer must be a StatisticsReducer")
        self.geometry = geometry
        self.reducer = resolved_reducer

    @property
    def n_parameters(self) -> int:
        """Return the flat parameter dimension ``P``."""

        return self.geometry.n_parameters

    @property
    def n_local_samples(self) -> int:
        """Return the number of score rows held locally."""

        return self.geometry.n_local_samples

    def jv(self, vector: Any) -> Any:
        """Apply ``A v``: a parameter vector to local sample space.

        No reduction is required.  Row ``k`` of ``A v`` depends only on row
        ``k`` of ``A``, so under sample sharding this stays rank-local.
        """

        vector = self.as_parameter_vector(vector, name="jv.vector")
        return self.geometry.design @ vector

    def jt_u(self, sample_vector: Any) -> Any:
        """Apply ``A^T u``: sample space to a parameter vector.

        This is the reduced direction.  Under sample sharding each rank holds
        a partial sum over its own rows, so the result goes through the
        reducer as a single ``P``-element sum.
        """

        sample_vector = self.as_sample_vector(sample_vector, name="jt_u.sample_vector")
        local = self.geometry.design.transpose(0, 1) @ sample_vector
        return self.reducer.reduce_sum(local)

    def qgt_action(self, vector: Any) -> Any:
        """Apply ``S v = A^T (A v)`` without forming ``S``.

        This is the matrix-free parameter-space action.  Its cost is
        ``O(B P)`` per call with one ``P``-element reduction, against
        ``O(P^2)`` storage for the dense form.
        """

        return self.jt_u(self.jv(vector))

    def sample_gram_action(self, sample_vector: Any) -> Any:
        """Apply ``T u = A (A^T u)`` without forming ``T``.

        Note that the inner ``A^T u`` is reduced, so this action is globally
        correct under sample sharding even though :meth:`dense_sample_gram`
        is not.
        """

        return self.jv(self.jt_u(sample_vector))

    def trace(self) -> Any:
        """Return the reduced ``trace(S) = trace(T) = ||A||_F^2`` as a scalar tensor."""

        local = self.geometry.design.pow(2).sum()
        return self.reducer.reduce_sum(local)

    def dense_parameter_qgt(self) -> Any:
        """Return the reduced dense ``S = A^T A`` with shape ``[P, P]``.

        The result is explicitly symmetrized.  ``A^T A`` is symmetric in exact
        arithmetic but not bitwise so in floating point, and
        :func:`torch.linalg.eigh` reads only one triangle; symmetrizing makes
        the solve independent of which triangle that is.
        """

        local = self.geometry.design.transpose(0, 1) @ self.geometry.design
        reduced = self.reducer.reduce_sum(local)
        return _symmetrize(reduced)

    def dense_sample_gram(self) -> Any:
        """Return the dense ``T = A A^T`` over the locally held rows.

        Returns
        -------
        torch.Tensor
            Symmetric ``[B_local, B_local]`` matrix.

        Notes
        -----
        Unlike every other method here, this one is **not** made global by the
        reducer.  A global sample-space Gram contains inner products between
        score rows on *different* ranks, so it requires gathering the rows
        themselves, not summing a local matrix.  That gather is out of scope
        for this lane, so under a genuinely sharded reducer this method
        returns a block-diagonal piece of the true Gram rather than the Gram.
        It is correct as written for a single process, which is the only
        configuration this lane admits; :meth:`sample_gram_action` is the form
        that stays correct under sharding.
        """

        design = self.geometry.design
        return _symmetrize(design @ design.transpose(0, 1))

    def energy_gradient(self, residual: Any) -> Any:
        """Return ``g = A^T epsilon``, the ordinary VMC energy gradient.

        Under the default conventions this equals the gradient of
        :func:`~tpen.training.vmc.compute_vmc_objective` with respect to the
        parameters, flattened in layout order.
        """

        return self.jt_u(residual)

    def as_parameter_vector(self, vector: Any, *, name: str) -> Any:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"QGTOperator.{name} must be a torch.Tensor")
        if vector.ndim != 1 or int(vector.numel()) != self.n_parameters:
            raise ValueError(
                f"QGTOperator.{name} must be a flat vector of length {self.n_parameters}, "
                f"got shape {tuple(vector.shape)}"
            )
        return vector.to(dtype=self.geometry.dtype, device=self.geometry.device)

    def as_sample_vector(self, vector: Any, *, name: str) -> Any:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"QGTOperator.{name} must be a torch.Tensor")
        if vector.ndim != 1 or int(vector.numel()) != self.n_local_samples:
            raise ValueError(
                f"QGTOperator.{name} must be a flat vector of length "
                f"{self.n_local_samples}, got shape {tuple(vector.shape)}"
            )
        return vector.to(dtype=self.geometry.dtype, device=self.geometry.device)


def solve_parameter_space(
    operator: QGTOperator,
    residual: Any,
    *,
    damping: DampingPolicy,
    rank_cutoff: float = 0.0,
) -> tuple[Any, SolveDiagnostics]:
    """Solve ``(S + lambda I) delta = A^T epsilon`` densely in parameter space.

    Parameters
    ----------
    operator : QGTOperator
        Operator over the step's design matrix.
    residual : torch.Tensor
        Normalized energy residual ``epsilon``.
    damping : DampingPolicy
        Regularization policy supplying ``lambda``.
    rank_cutoff : float, optional
        Relative eigenvalue cutoff.  Modes with
        ``w < rank_cutoff * max(w)`` are discarded.  ``0.0`` keeps every mode.

    Returns
    -------
    tuple of (torch.Tensor, SolveDiagnostics)
        The flat ``P``-element update direction and its diagnostics.
    """

    matrix = operator.dense_parameter_qgt()
    gradient = operator.energy_gradient(residual)
    trace = float(operator.trace().item())
    shift = damping.shift(trace=trace, n_parameters=operator.n_parameters)
    direction, retained, max_eigenvalue, min_retained = _eigh_solve(
        matrix,
        gradient,
        shift=shift,
        rank_cutoff=rank_cutoff,
    )
    diagnostics = SolveDiagnostics(
        dtype=str(matrix.dtype),
        space="parameter",
        shift=shift,
        trace=trace,
        n_modes=int(matrix.shape[0]),
        retained_modes=retained,
        max_eigenvalue=max_eigenvalue,
        min_retained_eigenvalue=min_retained,
    )
    return direction, diagnostics


def solve_sample_space(
    operator: QGTOperator,
    residual: Any,
    *,
    damping: DampingPolicy,
    rank_cutoff: float = 0.0,
) -> tuple[Any, SolveDiagnostics]:
    """Solve ``(T + lambda I) y = epsilon`` and map back with ``delta = A^T y``.

    This is the minSR route.  It is algebraically identical to
    :func:`solve_parameter_space` and is preferred when ``B << P``, where it
    replaces a ``P x P`` factorization with a ``B x B`` one.

    Parameters
    ----------
    operator : QGTOperator
        Operator over the step's design matrix.
    residual : torch.Tensor
        Normalized energy residual ``epsilon``.
    damping : DampingPolicy
        Regularization policy.  The shift is anchored to ``trace(S) / P``,
        exactly as in the parameter-space route, so the two agree.
    rank_cutoff : float, optional
        Relative eigenvalue cutoff, matched to the parameter-space route.

    Returns
    -------
    tuple of (torch.Tensor, SolveDiagnostics)
        The flat ``P``-element update direction and its diagnostics.
    """

    matrix = operator.dense_sample_gram()
    residual = operator.as_sample_vector(residual, name="solve_sample_space.residual")
    trace = float(operator.trace().item())
    # Anchored on the parameter count, not the sample count: see DampingPolicy.
    shift = damping.shift(trace=trace, n_parameters=operator.n_parameters)
    sample_solution, retained, max_eigenvalue, min_retained = _eigh_solve(
        matrix,
        residual,
        shift=shift,
        rank_cutoff=rank_cutoff,
    )
    direction = operator.jt_u(sample_solution)
    diagnostics = SolveDiagnostics(
        dtype=str(matrix.dtype),
        space="sample",
        shift=shift,
        trace=trace,
        n_modes=int(matrix.shape[0]),
        retained_modes=retained,
        max_eigenvalue=max_eigenvalue,
        min_retained_eigenvalue=min_retained,
    )
    return direction, diagnostics


def _symmetrize(matrix: Any) -> Any:
    """Return the exact symmetric part of a numerically symmetric matrix."""

    return 0.5 * (matrix + matrix.transpose(0, 1))


def _eigh_solve(
    matrix: Any,
    right_hand_side: Any,
    *,
    shift: float,
    rank_cutoff: float,
) -> tuple[Any, int, float, float]:
    """Solve ``(M + shift I) x = b`` by symmetric eigendecomposition.

    An eigendecomposition rather than a Cholesky or LU factorization is the
    deliberate choice for a dense *reference*: it makes rank truncation and
    the retained spectrum directly observable, which is what the singular and
    rank-deficient acceptance cases need.  Speed is not the objective of this
    path.

    Returns
    -------
    tuple
        ``(solution, retained_modes, max_eigenvalue, min_retained_eigenvalue)``.
    """

    if float(rank_cutoff) < 0.0 or float(rank_cutoff) >= 1.0:
        raise ValueError("rank_cutoff must satisfy 0.0 <= rank_cutoff < 1.0")
    if not torch.isfinite(matrix).all():
        raise ValueError("QGT solve received a non-finite matrix")
    if not torch.isfinite(right_hand_side).all():
        raise ValueError("QGT solve received a non-finite right-hand side")

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    # eigh returns ascending eigenvalues; a PSD matrix can still produce tiny
    # negative values from rounding, so clamp before any relative comparison.
    eigenvalues = eigenvalues.clamp_min(0.0)
    max_eigenvalue = float(eigenvalues[-1].item()) if eigenvalues.numel() else 0.0

    threshold = float(rank_cutoff) * max_eigenvalue
    retained_mask = eigenvalues >= threshold if threshold > 0.0 else None

    if retained_mask is not None:
        retained_modes = int(retained_mask.sum().item())
        retained_values = eigenvalues[retained_mask]
        min_retained = float(retained_values.min().item()) if retained_values.numel() else 0.0
    else:
        retained_modes = int(eigenvalues.numel())
        min_retained = float(eigenvalues[0].item()) if eigenvalues.numel() else 0.0

    # Numerical-rank guard, taken RELATIVE to the spectrum and applied to the
    # smallest mode that survives truncation.
    #
    # An absolute test such as ``min(eigenvalues + shift) <= 0`` is not usable
    # here, and was the original form of this check. Whether a numerically zero
    # eigenvalue comes back from a symmetric eigensolver as ``+1e-17`` or
    # ``-1e-17`` is a rounding accident that differs between backends: on the
    # same rank-deficient matrix NumPy returned -1.7e-16 while torch returned a
    # small positive value. An absolute test therefore fires or stays silent by
    # coin flip, and when it stays silent it lets through a direction amplified
    # by 1/epsilon -- confidently wrong rather than loudly refused.
    #
    # The tolerance is the standard numerical-rank threshold, matching the
    # convention used by ``numpy.linalg.matrix_rank``: ``n * eps * largest``.
    # Damping and rank truncation are both legitimate ways to make an otherwise
    # singular problem well posed, so a positive shift or a cutoff that removes
    # the offending modes correctly passes this guard.
    rank_tolerance = (
        eigenvalues.numel() * torch.finfo(matrix.dtype).eps * max_eigenvalue
    )
    if min_retained + shift <= rank_tolerance:
        raise ValueError(
            "QGT solve is singular: the smallest retained eigenvalue "
            f"({min_retained:.3e}) plus damping ({shift:.3e}) is within the "
            f"numerical-rank tolerance ({rank_tolerance:.3e}) of a spectrum whose "
            f"largest eigenvalue is {max_eigenvalue:.3e}. Supply positive damping "
            "or a rank cutoff."
        )

    projected = eigenvectors.transpose(0, 1) @ right_hand_side
    scaled = projected / (eigenvalues + shift)
    if retained_mask is not None:
        scaled = torch.where(retained_mask, scaled, torch.zeros_like(scaled))

    solution = eigenvectors @ scaled
    return solution, retained_modes, max_eigenvalue, min_retained


__all__ = [
    "DampingPolicy",
    "QGTOperator",
    "SolveDiagnostics",
    "solve_parameter_space",
    "solve_sample_space",
]
