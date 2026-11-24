"""Stochastic reconfiguration and minSR as a typed VMC update method.

This is the first consumer of the
:class:`~tpen.training.update.ScoreUpdateInput` seam.  It composes the frozen
geometry of :mod:`tpen.training.score_geometry` with the linear algebra of
:mod:`tpen.training.qgt` into one
:class:`~tpen.training.update.VMCUpdateMethod`, and adds exactly the policy a
score consumer needs and the layers below deliberately do not have: which
solve route to take, what to do about non-finite samples, how far a single
step is allowed to move, and what to report.

Relationship to the ordinary VMC gradient
-----------------------------------------
The energy gradient this method forms, ``g = A^T epsilon``, is the same vector
that :func:`~tpen.training.vmc.compute_vmc_objective` produces through
autograd, including its factor of two and its exclusion of non-finite
local-energy samples.  The update direction is the preconditioned
``delta = (S + lambda I)^{-1} g``, and the parameter step is
``theta <- theta - lr * delta``.  As ``S`` approaches a multiple of the
identity, ``delta`` becomes proportional to ``g`` and the method reduces to
ordinary gradient descent on the VMC objective.  That limit is a real
acceptance check, not a decorative remark, and it only holds because the
residual scale is shared with the objective.

No silent fallback
------------------
This method never quietly reverts to a first-order step.  A degenerate or
non-finite step is reported as ``applied=False`` with a telemetry reason, and
a genuinely inconsistent input raises.  There is no code path in which an
``SR`` configuration silently trains with Adam.

Gradient observability
----------------------
The method writes its *preconditioned direction* into each parameter's
``.grad`` and steps a plain SGD optimizer, so the applied displacement is
exactly ``lr * delta``.  A callback that reads raw ``.grad`` after the step --
:class:`~tpen.callback.health.gradient_stats.GradientStats` does exactly this,
by design, because :class:`~tpen.training.update.LegacyAutogradUpdate`
deliberately omits a post-step ``zero_grad`` -- therefore observes the natural
gradient rather than the Euclidean one under this method.  That is a real
change in what such a callback means, so both norms are reported separately in
telemetry, and :attr:`~tpen.training.update.VMCUpdateResult.grad_norm` is the
*Euclidean* energy-gradient norm, which keeps the trainer's headline metric
comparable across Adam and SR runs.  Reconciling the callback itself is
integration work owned by the Hydra/telemetry layer, not by this module.

Distributed posture
-------------------
Every cross-sample sum is taken through an injected
:class:`~tpen.training.statistics.StatisticsReducer`.  This module creates no
process group, imports no runtime wrapper, and never consumes a DDP-reduced
gradient as a score row.  A distributed claim for this method belongs to a
later slice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
import numbers
from typing import Any, Literal, Self

from tpen.dependencies import require_torch
from tpen.nn.forward import MaterializedParameterScoreRequest
from tpen.training.qgt import (
    DampingPolicy,
    QGTOperator,
    SolveDiagnostics,
    solve_parameter_space,
    solve_sample_space,
)
from tpen.training.score_geometry import (
    NONFINITE_SCORE_ROW_POLICY,
    ScoreConventions,
    build_energy_residual,
    build_score_geometry_from_rows,
    flatten_parameter_score_blocks,
    layout_convention_fingerprint,
    select_score_update_rows,
    unflatten_to_layout,
)
from tpen.training.statistics import IdentityStatisticsReducer, StatisticsReducer
from tpen.training.vmc import (
    DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    resolve_nonfinite_local_energy_policy,
)
from tpen.training.update import (
    ModelParameterBinding,
    ScoreUpdateInput,
    UpdateDiagnostics,
    UpdateMethodDescription,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
    flatten_settings,
    json_safe_metric_name,
    json_safe_scalar,
)

torch = require_torch(feature="VMC stochastic reconfiguration")


# Bump when the persisted state envelope changes shape or meaning.
# sr-state-2 adds `nonfinite_local_energy_policy` and `nonfinite_score_row_policy`.
SR_STATE_VERSION = "sr-state-2"

SolveSpace = Literal["auto", "parameter", "sample"]

# Exact authority-factor vocabulary for SR/minSR grid mapping.
SR_GRID_FACTOR_KEYS: frozenset[str] = frozenset(
    {"eta", "relative_damping", "squared_update_norm_bound"}
)


@dataclass(frozen=True, kw_only=True)
class SRPolicy:
    """Solve-route, regularization, and trust policy for an SR/minSR step.

    Parameters
    ----------
    solve_space : {"auto", "parameter", "sample"}, optional
        Which algebraically equivalent route to take.  ``"parameter"`` forms
        the ``P x P`` QGT; ``"sample"`` is minSR and forms the ``B x B`` Gram
        matrix; ``"auto"`` picks whichever matrix is smaller.  Because the
        routes agree exactly, this is a cost choice, not a numerical one --
        which is precisely why ``"auto"`` is a safe default.
    damping : DampingPolicy, optional
        Regularization applied identically in both routes.
    rank_cutoff : float, optional
        Relative eigenvalue cutoff for rank truncation.  ``0.0`` retains
        every mode.
    learning_rate : float, optional
        Step size applied to the preconditioned direction.  Retained here as
        well as in the optimizer so that a telemetry record is self-describing.
    max_update_norm : float or None, optional
        Trust cap on the Euclidean norm ``||Delta theta||_2`` of the applied
        parameter displacement ``lr * delta``.  ``None`` disables this cap.
        Mutually exclusive with ``squared_update_norm_bound``.  A cap is a
        scale limit, not a direction change: the direction is rescaled, never
        clipped per-coordinate.
    squared_update_norm_bound : float or None, optional
        Trust cap on the squared Euclidean norm ``||Delta theta||_2^2`` of the
        applied parameter displacement ``lr * delta``.  ``None`` disables this
        cap.  Mutually exclusive with ``max_update_norm``.  Enforcing
        ``||Delta theta||_2^2 <= C`` rescales the direction proportionally by
        ``sqrt(C) / ||lr * delta||_2`` when exceeded, never substituting the
        unsquared condition ``||Delta theta||_2 <= C``.
    score_chunk_size : int or None, optional
        Sample chunk size passed to the score-bearing forward request.
        ``None`` leaves the choice to the model.  This is a memory control:
        materializing the ``[B, P]`` score block is the dominant allocation of
        an SR step, and chunking trades a larger forward for a smaller peak.

    Notes
    -----
    ``learning_rate`` must agree with the owned optimizer's ``lr``; the update
    method checks this rather than trusting the caller, because two learning
    rates that silently disagree would make every reported step size wrong.
    """

    solve_space: SolveSpace = "auto"
    damping: DampingPolicy = DampingPolicy()
    rank_cutoff: float = 0.0
    learning_rate: float = 1.0e-2
    max_update_norm: float | None = None
    squared_update_norm_bound: float | None = None
    score_chunk_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rank_cutoff",
            _coerce_real_float(self.rank_cutoff, field_name="SRPolicy.rank_cutoff"),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _coerce_real_float(self.learning_rate, field_name="SRPolicy.learning_rate"),
        )
        if self.max_update_norm is not None:
            object.__setattr__(
                self,
                "max_update_norm",
                _coerce_real_float(
                    self.max_update_norm, field_name="SRPolicy.max_update_norm"
                ),
            )
        if self.squared_update_norm_bound is not None:
            object.__setattr__(
                self,
                "squared_update_norm_bound",
                _coerce_real_float(
                    self.squared_update_norm_bound,
                    field_name="SRPolicy.squared_update_norm_bound",
                ),
            )
        self.validate()

    def validate(self) -> Self:
        """Validate route selection, cutoff range, step-size positivity, and trust caps."""

        if self.solve_space not in ("auto", "parameter", "sample"):
            raise ValueError(
                "SRPolicy.solve_space must be 'auto', 'parameter', or 'sample', "
                f"got {self.solve_space!r}"
            )
        if not isinstance(self.damping, DampingPolicy):
            raise TypeError("SRPolicy.damping must be a DampingPolicy")
        self.damping.validate()
        rank_cutoff = _coerce_real_float(self.rank_cutoff, field_name="SRPolicy.rank_cutoff")
        if not _is_finite(rank_cutoff) or not 0.0 <= rank_cutoff < 1.0:
            raise ValueError("SRPolicy.rank_cutoff must satisfy 0.0 <= rank_cutoff < 1.0")
        learning_rate = _coerce_real_float(
            self.learning_rate, field_name="SRPolicy.learning_rate"
        )
        if not learning_rate > 0.0 or not _is_finite(learning_rate):
            raise ValueError("SRPolicy.learning_rate must be finite and positive")
        if (
            self.max_update_norm is not None
            and self.squared_update_norm_bound is not None
        ):
            raise ValueError(
                "SRPolicy cannot set both max_update_norm and squared_update_norm_bound; "
                "specify at most one displacement cap"
            )
        if self.max_update_norm is not None:
            max_update_norm = _coerce_real_float(
                self.max_update_norm, field_name="SRPolicy.max_update_norm"
            )
            if not max_update_norm > 0.0 or not _is_finite(max_update_norm):
                raise ValueError("SRPolicy.max_update_norm must be finite and positive, or None")
        if self.squared_update_norm_bound is not None:
            squared_bound = _coerce_real_float(
                self.squared_update_norm_bound,
                field_name="SRPolicy.squared_update_norm_bound",
            )
            if not squared_bound > 0.0 or not _is_finite(squared_bound):
                raise ValueError(
                    "SRPolicy.squared_update_norm_bound must be finite and positive, or None"
                )
        if self.score_chunk_size is not None:
            if type(self.score_chunk_size) is not int or self.score_chunk_size < 1:
                raise ValueError("SRPolicy.score_chunk_size must be a positive integer, or None")
        return self

    def resolve_space(self, *, n_parameters: int, n_samples: int) -> str:
        """Return the concrete route for one step.

        ``"auto"`` chooses the smaller matrix: sample space when there are
        fewer samples than parameters, which is the ordinary VMC regime.
        """

        if self.solve_space != "auto":
            return self.solve_space
        return "sample" if n_samples < n_parameters else "parameter"

    def trust_scale(self, *, direction_norm: float) -> float:
        """Return the multiplicative trust factor for one update direction.

        Parameters
        ----------
        direction_norm : float
            Euclidean norm ``||delta||_2`` of the unscaled update direction.

        Returns
        -------
        float
            Scalar in ``(0, 1]`` applied proportionally to ``delta``.  Returns
            ``1.0`` when no cap is active, when ``direction_norm == 0.0``, or
            when the proposed displacement already satisfies the active bound.
        """

        norm = _coerce_real_float(direction_norm, field_name="direction_norm")
        if not _is_finite(norm) or norm < 0.0:
            raise ValueError(f"direction_norm must be finite and non-negative, got {norm!r}")
        if norm == 0.0:
            return 1.0
        proposed = self.learning_rate * norm
        if self.max_update_norm is not None:
            return 1.0 if proposed <= self.max_update_norm else self.max_update_norm / proposed
        if self.squared_update_norm_bound is not None:
            # Enforce ||Delta theta||_2^2 <= squared_update_norm_bound (never ||Delta theta||_2 <= C).
            proposed_sq = proposed * proposed
            if proposed_sq <= self.squared_update_norm_bound:
                return 1.0
            return min(1.0, math.sqrt(self.squared_update_norm_bound) / proposed)
        return 1.0

    def fingerprint(self) -> dict[str, Any]:
        """Return JSON-safe policy metadata for telemetry or a state envelope."""

        return {
            "solve_space": self.solve_space,
            "damping": self.damping.fingerprint(),
            "rank_cutoff": self.rank_cutoff,
            "learning_rate": self.learning_rate,
            "max_update_norm": self.max_update_norm,
            "squared_update_norm_bound": self.squared_update_norm_bound,
            "score_chunk_size": self.score_chunk_size,
        }


@dataclass(frozen=True, kw_only=True)
class SRGridHyperparameters:
    """Validated SR/minSR authority-grid factor triple.

    Parameters
    ----------
    eta : float
        Positive finite learning rate mapped to :attr:`SRPolicy.learning_rate`.
    relative_damping : float
        Positive finite relative diagonal damping mapped to
        :attr:`DampingPolicy.relative`.
    squared_update_norm_bound : float
        Positive finite upper bound ``C`` on the squared Euclidean displacement
        norm ``||Delta theta||_2^2 <= C``, mapped to
        :attr:`SRPolicy.squared_update_norm_bound`.
    """

    eta: float
    relative_damping: float
    squared_update_norm_bound: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "eta",
            _coerce_positive_finite_float(
                self.eta, field_name="SRGridHyperparameters.eta"
            ),
        )
        object.__setattr__(
            self,
            "relative_damping",
            _coerce_positive_finite_float(
                self.relative_damping,
                field_name="SRGridHyperparameters.relative_damping",
            ),
        )
        object.__setattr__(
            self,
            "squared_update_norm_bound",
            _coerce_positive_finite_float(
                self.squared_update_norm_bound,
                field_name="SRGridHyperparameters.squared_update_norm_bound",
            ),
        )
        self.validate()

    def validate(self) -> Self:
        """Validate that all three authority factors are non-bool positive finite floats."""

        _coerce_positive_finite_float(
            self.eta, field_name="SRGridHyperparameters.eta"
        )
        _coerce_positive_finite_float(
            self.relative_damping,
            field_name="SRGridHyperparameters.relative_damping",
        )
        _coerce_positive_finite_float(
            self.squared_update_norm_bound,
            field_name="SRGridHyperparameters.squared_update_norm_bound",
        )
        return self

    @classmethod
    def from_mapping(cls, factors: Mapping[str, Any]) -> SRGridHyperparameters:
        """Parse and strictly validate an SR authority-factor mapping.

        Parameters
        ----------
        factors : Mapping[str, Any]
            Mapping containing the exact key set
            ``{"eta", "relative_damping", "squared_update_norm_bound"}``.

        Returns
        -------
        SRGridHyperparameters
            Validated factor record.

        Raises
        ------
        TypeError
            If ``factors`` is not a mapping, or if any factor value is a
            boolean or non-real value.
        ValueError
            If any required key is missing, any unknown or extra key is
            present, or any numeric value is non-finite or not strictly
            positive.
        """

        if not isinstance(factors, Mapping):
            raise TypeError(
                "SR grid factors must be a Mapping with keys "
                f"{sorted(SR_GRID_FACTOR_KEYS)}, got {type(factors).__name__}"
            )
        provided_keys = set(factors.keys())
        missing = sorted(SR_GRID_FACTOR_KEYS - provided_keys)
        unknown = sorted(
            (key for key in provided_keys if key not in SR_GRID_FACTOR_KEYS),
            key=repr,
        )
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {missing!r}")
            if unknown:
                details.append(f"unknown keys: {unknown!r}")
            raise ValueError(
                "SR grid factors must have exact keys "
                f"{sorted(SR_GRID_FACTOR_KEYS)!r} ({'; '.join(details)})"
            )
        # Validate and coerce before constructing so bool/None/non-real values
        # consistently raise TypeError with exact field diagnostics under both
        # Typeguard-ON and suppressed-typecheck execution.
        eta = _coerce_positive_finite_float(
            factors["eta"], field_name="SRGridHyperparameters.eta"
        )
        relative_damping = _coerce_positive_finite_float(
            factors["relative_damping"],
            field_name="SRGridHyperparameters.relative_damping",
        )
        squared_update_norm_bound = _coerce_positive_finite_float(
            factors["squared_update_norm_bound"],
            field_name="SRGridHyperparameters.squared_update_norm_bound",
        )
        return cls(
            eta=eta,
            relative_damping=relative_damping,
            squared_update_norm_bound=squared_update_norm_bound,
        )

    def to_factors(self) -> dict[str, float]:
        """Return the exact authority-factor dictionary for readback."""

        return {
            "eta": self.eta,
            "relative_damping": self.relative_damping,
            "squared_update_norm_bound": self.squared_update_norm_bound,
        }


def sr_policy_from_factors(
    factors: Mapping[str, Any] | SRGridHyperparameters,
    *,
    solve_space: SolveSpace = "auto",
    damping_minimum: float = 1.0e-12,
    rank_cutoff: float = 0.0,
    score_chunk_size: int | None = None,
) -> SRPolicy:
    """Map validated SR authority factors onto a concrete :class:`SRPolicy`.

    Parameters
    ----------
    factors : Mapping[str, Any] or SRGridHyperparameters
        Exact authority factor mapping or validated record specifying ``eta``,
        ``relative_damping``, and ``squared_update_norm_bound``.
    solve_space : {"auto", "parameter", "sample"}, optional
        QGT solve route forwarded to :class:`SRPolicy`.
    damping_minimum : float, optional
        Non-negative finite floor forwarded to :attr:`DampingPolicy.minimum`.
    rank_cutoff : float, optional
        Relative eigenvalue cutoff forwarded to :class:`SRPolicy`.
    score_chunk_size : int or None, optional
        Sample chunk size forwarded to :class:`SRPolicy`.

    Returns
    -------
    SRPolicy
        Concrete policy with ``learning_rate=eta``,
        ``damping=DampingPolicy(relative=relative_damping, absolute=0.0,
        minimum=damping_minimum)``, ``max_update_norm=None``, and
        ``squared_update_norm_bound=squared_update_norm_bound``.
    """

    if isinstance(factors, SRGridHyperparameters):
        resolved = factors.validate()
    else:
        resolved = SRGridHyperparameters.from_mapping(factors)
    minimum = _coerce_real_float(damping_minimum, field_name="damping_minimum")
    if not _is_finite(minimum) or minimum < 0.0:
        raise ValueError(
            f"damping_minimum must be finite and non-negative, got {minimum!r}"
        )
    return SRPolicy(
        solve_space=solve_space,
        damping=DampingPolicy(
            absolute=0.0,
            relative=resolved.relative_damping,
            minimum=minimum,
        ),
        rank_cutoff=rank_cutoff,
        learning_rate=resolved.eta,
        max_update_norm=None,
        squared_update_norm_bound=resolved.squared_update_norm_bound,
        score_chunk_size=score_chunk_size,
    )


def build_sr_update_from_factors(
    factors: Mapping[str, Any] | SRGridHyperparameters,
    *,
    model_parameters: ModelParameterBinding,
    solve_space: SolveSpace = "auto",
    damping_minimum: float = 1.0e-12,
    rank_cutoff: float = 0.0,
    score_chunk_size: int | None = None,
    conventions: ScoreConventions | None = None,
    reducer: StatisticsReducer | None = None,
    nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
) -> tuple[torch.optim.SGD, StochasticReconfigurationUpdate]:
    """Construct a matched plain-SGD optimizer and SR update from authority factors.

    Parameters
    ----------
    factors : Mapping[str, Any] or SRGridHyperparameters
        Exact authority factor mapping or validated record specifying ``eta``,
        ``relative_damping``, and ``squared_update_norm_bound``.
    model_parameters : ModelParameterBinding
        Bound live parameter domain owned by the constructed SR method.
    solve_space : {"auto", "parameter", "sample"}, optional
        QGT solve route forwarded to :func:`sr_policy_from_factors`.
    damping_minimum : float, optional
        Non-negative finite floor forwarded to :attr:`DampingPolicy.minimum`.
    rank_cutoff : float, optional
        Relative eigenvalue cutoff forwarded to :class:`SRPolicy`.
    score_chunk_size : int or None, optional
        Sample chunk size forwarded to :class:`SRPolicy`.
    conventions : ScoreConventions or None, optional
        Score conventions forwarded to :class:`StochasticReconfigurationUpdate`.
    reducer : StatisticsReducer or None, optional
        Statistics reducer forwarded to :class:`StochasticReconfigurationUpdate`.
    nonfinite_local_energy_policy : str, optional
        Non-finite local-energy policy forwarded to
        :class:`StochasticReconfigurationUpdate`.

    Returns
    -------
    tuple of (torch.optim.SGD, StochasticReconfigurationUpdate)
        Plain SGD optimizer whose learning rate equals ``eta`` and the bound
        SR update method exposing exact readback on ``policy``,
        ``policy.fingerprint()``, and ``describe().settings``.
    """

    if not isinstance(model_parameters, ModelParameterBinding):
        raise TypeError(
            "build_sr_update_from_factors requires model_parameters to be a "
            "ModelParameterBinding"
        )
    policy = sr_policy_from_factors(
        factors,
        solve_space=solve_space,
        damping_minimum=damping_minimum,
        rank_cutoff=rank_cutoff,
        score_chunk_size=score_chunk_size,
    )
    optimizer = torch.optim.SGD(
        model_parameters.parameters,
        lr=policy.learning_rate,
    )
    method = StochasticReconfigurationUpdate(
        optimizer,
        model_parameters=model_parameters,
        policy=policy,
        conventions=conventions,
        reducer=reducer,
        nonfinite_local_energy_policy=nonfinite_local_energy_policy,
    )
    return optimizer, method


@dataclass(frozen=True, kw_only=True)
class SRTelemetry(UpdateDiagnostics):
    """Observable record of one SR/minSR update attempt.

    Every field is a Python scalar so the record can go straight into a
    JSON-safe metrics stream.
    """

    applied: bool
    reason: str
    step: int
    n_samples: int
    n_finite_samples: int
    n_parameters: int
    energy_gradient_norm: float
    update_direction_norm: float
    applied_update_norm: float
    trust_scale: float
    diagnostics: SolveDiagnostics | None = None
    n_nonfinite_energy_samples: int = 0
    n_nonfinite_score_samples: int = 0
    nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY
    nonfinite_score_row_policy: str = NONFINITE_SCORE_ROW_POLICY

    def as_metrics(self, *, prefix: str = "sr") -> dict[str, Any]:
        """Return JSON-safe telemetry keys for a training metrics record.

        EVERY FLOAT HERE ROUTES THROUGH `json_safe_scalar`, and that is
        load-bearing rather than defensive. This method predates the explicit
        result contract and used a bare ``float()``, which meant a NON-FINITE
        norm reached ``tpen.logging.jsonl`` -- which calls
        ``json.dumps(..., allow_nan=False)`` and RAISES.

        This path is the one most likely to produce such a value, not the
        least: a non-finite direction is an EXPECTED outcome here, with its own
        skip reason, so the record written to explain the bad step was the
        record that crashed the run writing it. A finite value passes through
        unchanged, so no existing record changes meaning.
        """

        prefix = json_safe_metric_name(prefix)

        metrics: dict[str, Any] = {
            f"{prefix}_applied": bool(self.applied),
            f"{prefix}_reason": json_safe_scalar(self.reason),
            f"{prefix}_step": int(self.step),
            f"{prefix}_samples": int(self.n_samples),
            f"{prefix}_finite_samples": int(self.n_finite_samples),
            f"{prefix}_nonfinite_energy_samples": int(self.n_nonfinite_energy_samples),
            f"{prefix}_nonfinite_score_samples": int(self.n_nonfinite_score_samples),
            f"{prefix}_nonfinite_local_energy_policy": json_safe_scalar(
                self.nonfinite_local_energy_policy
            ),
            f"{prefix}_nonfinite_score_row_policy": json_safe_scalar(
                self.nonfinite_score_row_policy
            ),
            f"{prefix}_parameters": int(self.n_parameters),
            f"{prefix}_energy_gradient_norm": json_safe_scalar(self.energy_gradient_norm),
            f"{prefix}_update_direction_norm": json_safe_scalar(self.update_direction_norm),
            f"{prefix}_applied_update_norm": json_safe_scalar(self.applied_update_norm),
            f"{prefix}_trust_scale": json_safe_scalar(self.trust_scale),
        }
        if self.diagnostics is not None:
            metrics.update(self.diagnostics.as_metrics(prefix=f"{prefix}_qgt"))
        return metrics


class StochasticReconfigurationUpdate(VMCUpdateMethod[ScoreUpdateInput]):
    """Dense SR / sample-space minSR update over materialized parameter scores.

    Parameters
    ----------
    optimizer : torch.optim.SGD
        The optimizer that applies the preconditioned direction.  It must be
        plain SGD -- see Notes.
    model_parameters : ModelParameterBinding
        The parameter domain this method owns and updates.
    policy : SRPolicy, optional
        Solve, damping, and trust policy.
    conventions : ScoreConventions, optional
        Frozen score conventions.  Defaults to :class:`ScoreConventions`.
    reducer : StatisticsReducer, optional
        Reducer for every cross-sample sum.  Defaults to
        :class:`~tpen.training.statistics.IdentityStatisticsReducer`.

    Notes
    -----
    The optimizer is required to be ``torch.optim.SGD`` with no momentum, no
    dampening, no weight decay, no Nesterov, and no ``maximize``.  Under those
    settings, and only those, ``optimizer.step()`` applies exactly
    ``theta <- theta - lr * grad``, which is what makes the applied
    displacement provably equal to the reported one.  Accepting a general
    optimizer would let a momentum buffer or a decoupled decay term silently
    change the update, and accepting Adam specifically is the failure this
    slice exists to rule out.  Rejecting it loudly at construction is the
    whole point.

    An SGD optimizer is used rather than a hand-rolled ``param.add_`` because
    it makes the method's persistent state a standard optimizer payload, so
    checkpoint save and restore already work through the trainer's existing
    ownership path.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        model_parameters: ModelParameterBinding,
        policy: SRPolicy | None = None,
        conventions: ScoreConventions | None = None,
        reducer: StatisticsReducer | None = None,
        nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    ) -> None:
        self.nonfinite_local_energy_policy = resolve_nonfinite_local_energy_policy(
            nonfinite_local_energy_policy
        )
        self.nonfinite_score_row_policy = NONFINITE_SCORE_ROW_POLICY
        resolved_policy = SRPolicy() if policy is None else policy
        if not isinstance(resolved_policy, SRPolicy):
            raise TypeError("StochasticReconfigurationUpdate.policy must be an SRPolicy")
        resolved_policy.validate()
        _validate_plain_sgd(optimizer, learning_rate=resolved_policy.learning_rate)
        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError(
                "StochasticReconfigurationUpdate.model_parameters must be a "
                "ModelParameterBinding"
            )
        resolved_conventions = ScoreConventions() if conventions is None else conventions
        if not isinstance(resolved_conventions, ScoreConventions):
            raise TypeError(
                "StochasticReconfigurationUpdate.conventions must be a ScoreConventions"
            )
        resolved_reducer = IdentityStatisticsReducer() if reducer is None else reducer
        if not isinstance(resolved_reducer, StatisticsReducer):
            raise TypeError(
                "StochasticReconfigurationUpdate.reducer must be a StatisticsReducer"
            )

        self.optimizer = optimizer
        self.model_parameters = model_parameters
        self.policy = resolved_policy
        self.conventions = resolved_conventions
        self.reducer = resolved_reducer
        self.completed_updates = 0
        # The RESULT is the record; `last_telemetry` below reads out of it.
        self._last_result: VMCUpdateResult | None = None

    @property
    def last_telemetry(self) -> SRTelemetry | None:
        """Return this attempt's diagnostics, for callers predating the result.

        COMPATIBILITY ONLY, AND NOT A SECOND AUTHORITY. It is derived from
        :attr:`last_result`, so it cannot disagree with what ``update``
        returned and cannot survive into a later attempt on its own.
        """

        result = self._last_result
        if result is None:
            return None
        diagnostics = result.diagnostics
        assert diagnostics is None or isinstance(diagnostics, SRTelemetry)
        return diagnostics

    def forward_request(self) -> MaterializedParameterScoreRequest:
        """Request raw per-sample parameter score blocks from the forward pass.

        Returning the request rather than performing it keeps this method free
        of the model: the trainer owns the single forward, and the score blocks
        arrive in the same packet as the value output, so no second forward or
        derivative recomputation is needed.

        ``score_chunk_size`` is forwarded here because chunking is a memory
        decision about materializing a ``[B, P]`` score block, which is a
        property of this method's step, not of the model.
        """

        return MaterializedParameterScoreRequest(chunk_size=self.policy.score_chunk_size)

    def norm_semantics(self) -> str:
        """Report the norm as the ENERGY-GRADIENT norm, not the step's size.

        Stated explicitly because this method writes a PRECONDITIONED
        direction into ``.grad`` before stepping the carrier, so ``.grad`` is
        not a gradient at all by the time an observer reads it. The reported
        norm is the norm of the energy gradient; the size of what was actually
        applied is a separate field of this method's diagnostics
        (``applied_update_norm``).
        """

        return "energy_gradient_l2_preconditioned_direction_in_grad"

    def declared_nonfinite_local_energy_policy(self) -> str:
        """Return the non-finite local-energy policy enforced by this SR method."""

        return self.nonfinite_local_energy_policy

    def describe(self) -> UpdateMethodDescription:
        """Add SR's resolved policy to the generic description."""

        base = super().describe()
        settings = dict(base.settings)
        settings.update(flatten_settings(self.policy.fingerprint(), prefix="policy_"))
        settings.update(flatten_settings(self.conventions.fingerprint(), prefix="conventions_"))
        settings["reducer_class"] = type(self.reducer).__qualname__
        settings["nonfinite_local_energy_policy"] = self.nonfinite_local_energy_policy
        settings["nonfinite_score_row_policy"] = self.nonfinite_score_row_policy
        return replace(base, settings=settings)

    def update_state(self) -> VMCUpdateState:
        """Return the single optimizer and parameter binding this method owns."""

        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Adopt restored model references after a checkpoint load."""

        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError(
                "StochasticReconfigurationUpdate model_parameters must be a "
                "ModelParameterBinding"
            )
        self.model_parameters = model_parameters

    def method_state_dict(self) -> Mapping[str, Any]:
        """Return the versioned method-state envelope.

        The envelope carries the layout-plus-convention fingerprint alongside
        the schedule counter, so a resume can reject a checkpoint whose
        parameter layout or numerical conventions no longer match instead of
        producing a plausible wrong step.  This slice has no warm start to
        persist; when an iterative solver adds one, it belongs in this envelope
        under the same fingerprint.

        The optimizer payload is deliberately NOT included.  The checkpoint
        already persists the optimizer separately, and this method's
        :meth:`update_state` names that same object as the single authority.
        Carrying it here too would create two sources of truth for one payload
        and restore it twice on every resume.
        """

        return {
            "version": SR_STATE_VERSION,
            "fingerprint": layout_convention_fingerprint(
                self.model_parameters.layout,
                self.conventions,
            ),
            "policy": self.policy.fingerprint(),
            "nonfinite_local_energy_policy": self.nonfinite_local_energy_policy,
            "nonfinite_score_row_policy": self.nonfinite_score_row_policy,
            "completed_updates": self.completed_updates,
        }

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the envelope, rejecting layout, convention, or policy drift."""

        self.validate_method_state_dict(state)
        completed = state.get("completed_updates", 0)
        self.completed_updates = completed

    def validate_method_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate the envelope without mutating the live update method."""

        if not isinstance(state, Mapping):
            raise TypeError("StochasticReconfigurationUpdate state must be a mapping")
        version = state.get("version")
        if version != SR_STATE_VERSION:
            raise ValueError(
                f"unsupported SR state version {version!r}, expected {SR_STATE_VERSION!r}"
            )
        expected = layout_convention_fingerprint(self.model_parameters.layout, self.conventions)
        recorded = state.get("fingerprint")
        if not isinstance(recorded, Mapping) or recorded.get("digest") != expected["digest"]:
            raise ValueError(
                "SR checkpoint layout/convention fingerprint does not match the live model"
            )
        if state.get("policy") != self.policy.fingerprint():
            raise ValueError("SR checkpoint policy does not match the live method")
        recorded_energy_policy = state.get("nonfinite_local_energy_policy")
        if recorded_energy_policy != self.nonfinite_local_energy_policy:
            raise ValueError(
                "SR checkpoint nonfinite_local_energy_policy does not match the live method: "
                f"recorded {recorded_energy_policy!r}, live {self.nonfinite_local_energy_policy!r}"
            )
        recorded_score_policy = state.get("nonfinite_score_row_policy")
        if recorded_score_policy != self.nonfinite_score_row_policy:
            raise ValueError(
                "SR checkpoint nonfinite_score_row_policy does not match the live method: "
                f"recorded {recorded_score_policy!r}, live {self.nonfinite_score_row_policy!r}"
            )
        completed = state.get("completed_updates", 0)
        if type(completed) is not int or completed < 0:
            raise ValueError("SR state completed_updates must be a non-negative integer")

    def update(self, update_input: ScoreUpdateInput) -> VMCUpdateResult:
        """Apply one SR/minSR step and report the outcome.

        Parameters
        ----------
        update_input : ScoreUpdateInput
            Live step record carrying raw uncentered score blocks and the
            model's direct parameter binding.

        Returns
        -------
        VMCUpdateResult
            ``applied`` reports whether parameters moved; ``grad_norm`` is the
            Euclidean energy-gradient norm, not the natural-gradient norm.

        Raises
        ------
        TypeError
            If the input is not a :class:`ScoreUpdateInput`.
        ValueError
            If the wavefunction is complex, or the score binding disagrees
            with the bound parameter domain.
        RuntimeError
            If ``nonfinite_local_energy_policy == "fail"`` and any local energy
            is non-finite, if any pinned finite local-energy row carries a
            non-finite parameter score under :data:`NONFINITE_SCORE_ROW_POLICY`,
            or if a nonzero-electron batch has no finite local-energy sample.
        """

        if not isinstance(update_input, ScoreUpdateInput):
            raise TypeError("StochasticReconfigurationUpdate requires ScoreUpdateInput")
        update_input.validate()
        self._validate_real_wavefunction(update_input)
        self._validate_binding(update_input)

        local_energy = update_input.local_energy.reshape(-1)
        n_samples = int(local_energy.numel())
        n_parameters = update_input.parameter_scores.layout.total_numel

        rows = flatten_parameter_score_blocks(
            update_input.parameter_scores,
            sample_shape=tuple(update_input.wavefunction.logabs.shape),
        )

        # Pin the finite row mask at the first (local-energy) evaluation and
        # refuse any pinned row with non-finite parameter scores rather than
        # silently re-masking (consolidated 377f6b0f / 02859027 ruling).
        selection = select_score_update_rows(
            local_energy=local_energy,
            rows=rows,
            nonfinite_local_energy_policy=self.nonfinite_local_energy_policy,
            method_name="stochastic reconfiguration",
        )
        finite_mask = selection.finite_mask
        n_finite = selection.n_finite
        n_nonfinite_energy = selection.n_nonfinite_energy
        n_nonfinite_score = selection.n_nonfinite_score_total

        if update_input.batch.n_electrons == 0:
            # The zero-electron vacuum has no sampled coordinate degrees of
            # freedom, so there is no update to make.  This mirrors the legacy
            # adapter's behavior rather than inventing a second convention.
            return self._skip(
                reason="zero_electron_batch",
                step=update_input.step,
                n_samples=n_samples,
                n_finite=n_finite,
                n_parameters=n_parameters,
                n_nonfinite_energy=n_nonfinite_energy,
                n_nonfinite_score=n_nonfinite_score,
            )
        if n_finite == 0:
            raise RuntimeError(
                "cannot compute an SR update: no finite local-energy sample remains "
                "for a nonzero-electron batch"
            )
        if n_finite < 2:
            # One sample centers to exactly zero, so the QGT and the gradient
            # both vanish.  Reporting this is more useful than applying a
            # provably zero step.
            return self._skip(
                reason="insufficient_finite_samples",
                step=update_input.step,
                n_samples=n_samples,
                n_finite=n_finite,
                n_parameters=n_parameters,
                n_nonfinite_energy=n_nonfinite_energy,
                n_nonfinite_score=n_nonfinite_score,
            )

        if n_finite != n_samples:
            rows = rows[finite_mask]
            local_energy = local_energy[finite_mask]

        geometry = build_score_geometry_from_rows(
            rows,
            layout=update_input.parameter_scores.layout,
            conventions=self.conventions,
            reducer=self.reducer,
        )
        residual = build_energy_residual(
            local_energy,
            geometry=geometry,
            reducer=self.reducer,
        )
        operator = QGTOperator(geometry, reducer=self.reducer)

        space = self.policy.resolve_space(
            n_parameters=operator.n_parameters,
            n_samples=geometry.count,
        )
        solve = solve_parameter_space if space == "parameter" else solve_sample_space
        direction, diagnostics = solve(
            operator,
            residual,
            damping=self.policy.damping,
            rank_cutoff=self.policy.rank_cutoff,
        )

        energy_gradient = operator.energy_gradient(residual)
        energy_gradient_norm = float(torch.linalg.vector_norm(energy_gradient).item())

        if not torch.isfinite(direction).all():
            return self._skip(
                reason="nonfinite_update_direction",
                step=update_input.step,
                n_samples=n_samples,
                n_finite=n_finite,
                n_parameters=n_parameters,
                energy_gradient_norm=energy_gradient_norm,
                diagnostics=diagnostics,
                n_nonfinite_energy=n_nonfinite_energy,
                n_nonfinite_score=n_nonfinite_score,
            )

        direction_norm = float(torch.linalg.vector_norm(direction).item())
        trust_scale = self._trust_scale(direction_norm)
        applied_norm = self.policy.learning_rate * direction_norm * trust_scale

        self._apply(direction * trust_scale)
        self.completed_updates += 1
        telemetry = SRTelemetry(
            applied=True,
            reason="applied",
            step=update_input.step,
            n_samples=n_samples,
            n_finite_samples=n_finite,
            n_nonfinite_energy_samples=n_nonfinite_energy,
            n_nonfinite_score_samples=n_nonfinite_score,
            nonfinite_local_energy_policy=self.nonfinite_local_energy_policy,
            nonfinite_score_row_policy=self.nonfinite_score_row_policy,
            n_parameters=n_parameters,
            energy_gradient_norm=energy_gradient_norm,
            update_direction_norm=direction_norm,
            applied_update_norm=applied_norm,
            trust_scale=trust_scale,
            diagnostics=diagnostics,
        )
        return self._record_result(
            applied=True,
            grad_norm=energy_gradient_norm,
            reason=telemetry.reason,
            diagnostics=telemetry,
        )

    def _trust_scale(self, direction_norm: float) -> float:
        """Return the factor capping the applied displacement."""

        return self.policy.trust_scale(direction_norm=direction_norm)

    def _apply(self, direction: Any) -> None:
        """Write the preconditioned direction into ``.grad`` and step."""

        blocks = unflatten_to_layout(direction, layout=self.model_parameters.layout)
        for parameter, block in zip(self.model_parameters.parameters, blocks, strict=True):
            parameter.grad = block.detach().to(
                dtype=parameter.dtype,
                device=parameter.device,
            )
        self.optimizer.step()

    def _skip(
        self,
        *,
        reason: str,
        step: int,
        n_samples: int,
        n_finite: int,
        n_parameters: int,
        energy_gradient_norm: float = 0.0,
        diagnostics: SolveDiagnostics | None = None,
        n_nonfinite_energy: int = 0,
        n_nonfinite_score: int = 0,
    ) -> VMCUpdateResult:
        """Record a non-applied step with an explicit reason."""

        telemetry = SRTelemetry(
            applied=False,
            reason=reason,
            step=step,
            n_samples=n_samples,
            n_finite_samples=n_finite,
            n_nonfinite_energy_samples=n_nonfinite_energy,
            n_nonfinite_score_samples=n_nonfinite_score,
            nonfinite_local_energy_policy=self.nonfinite_local_energy_policy,
            nonfinite_score_row_policy=self.nonfinite_score_row_policy,
            n_parameters=n_parameters,
            energy_gradient_norm=energy_gradient_norm,
            update_direction_norm=0.0,
            applied_update_norm=0.0,
            trust_scale=1.0,
            diagnostics=diagnostics,
        )
        return self._record_result(
            applied=False,
            grad_norm=energy_gradient_norm,
            reason=reason,
            diagnostics=telemetry,
        )

    def _validate_real_wavefunction(self, update_input: ScoreUpdateInput) -> None:
        """Reject a complex wavefunction rather than silently ignoring its phase.

        The whole geometry here is the real-``log|psi|`` QGT.  For a complex
        wavefunction the QGT acquires the imaginary part of the score
        covariance and the energy gradient acquires a term this method never
        forms.  Dropping the phase would therefore produce a confident,
        plausible, wrong step.
        """

        if update_input.wavefunction.phase is not None:
            raise ValueError(
                "StochasticReconfigurationUpdate supports real wavefunctions only; "
                "the step carries a complex phase, whose quantum geometric tensor "
                "this method does not form"
            )

    def _validate_binding(self, update_input: ScoreUpdateInput) -> None:
        """Require the step's scores to describe this method's parameter domain."""

        if not update_input.parameter_scores.layout.compare(self.model_parameters.layout)[0]:
            raise ValueError(
                "SR update scores do not match the bound parameter layout; the update "
                "method and the score-producing model have diverged"
            )
        bound = self.model_parameters.parameters
        step_parameters = update_input.parameter_binding.parameters
        if len(bound) != len(step_parameters) or any(
            left is not right for left, right in zip(bound, step_parameters, strict=True)
        ):
            raise ValueError(
                "SR update parameter binding does not reference the bound live "
                "parameters; a rebind was missed after a checkpoint restore"
            )


def _validate_plain_sgd(optimizer: Any, *, learning_rate: float) -> None:
    """Require a momentum-free, decay-free SGD whose ``lr`` matches the policy."""

    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError(
            "StochasticReconfigurationUpdate requires a torch.optim.SGD optimizer so the "
            "applied step is exactly -lr * preconditioned_direction; got "
            f"{type(optimizer).__name__}. An adaptive optimizer such as Adam would "
            "rescale the natural gradient and is never an acceptable substitute here."
        )
    for index, group in enumerate(optimizer.param_groups):
        for key, forbidden in (("momentum", 0), ("dampening", 0), ("weight_decay", 0)):
            value = group.get(key, 0)
            if float(value) != float(forbidden):
                raise ValueError(
                    f"StochasticReconfigurationUpdate requires SGD param group {index} to "
                    f"have {key}={forbidden}, got {value}"
                )
        for key in ("nesterov", "maximize"):
            if bool(group.get(key, False)):
                raise ValueError(
                    f"StochasticReconfigurationUpdate requires SGD param group {index} to "
                    f"have {key} disabled"
                )
        group_lr = float(group.get("lr"))
        if group_lr != float(learning_rate):
            raise ValueError(
                f"StochasticReconfigurationUpdate SGD param group {index} has lr={group_lr}, "
                f"which disagrees with SRPolicy.learning_rate={learning_rate}"
            )


def _coerce_real_float(value: Any, *, field_name: str) -> float:
    """Coerce a non-boolean real scalar to ``float``."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{field_name} must be a real number (not bool), got {type(value).__name__}"
        )
    return float(value)


def _coerce_positive_finite_float(value: Any, *, field_name: str) -> float:
    """Validate and coerce a non-boolean, finite, strictly positive real scalar."""

    coerced = _coerce_real_float(value, field_name=field_name)
    if not _is_finite(coerced) or not coerced > 0.0:
        raise ValueError(
            f"{field_name} must be finite and strictly positive, got {coerced!r}"
        )
    return coerced


def _is_finite(value: float) -> bool:
    """Return whether a Python float is finite."""

    return value == value and value not in (float("inf"), float("-inf"))


__all__ = [
    "SR_GRID_FACTOR_KEYS",
    "SR_STATE_VERSION",
    "SRGridHyperparameters",
    "SRPolicy",
    "SRTelemetry",
    "SolveSpace",
    "StochasticReconfigurationUpdate",
    "build_sr_update_from_factors",
    "sr_policy_from_factors",
]
