"""Dense Hamiltonian-tangent Linear Method as a typed VMC update method.

This module implements the real-wavefunction Toulouse--Umrigar dense Linear
Method (Umrigar et al., *PRL* **98**, 110201 (2007); Toulouse & Umrigar,
*J. Chem. Phys.* **126**, 084102 (2007), **128**, 174101 (2008)) over the
materialized single-pass Hamiltonian-tangent step contract
(:class:`~tpen.training.step_requests.MaterializedHamiltonianTangentStepRequest`
and :class:`~tpen.training.step_requests.HamiltonianTangentUpdateInput`).

Mathematical formulation
------------------------
On a pinned finite sample set of global size :math:`N \\ge 2`, with raw
log-amplitude scores :math:`O_{b,i} = \\partial_{\\theta_i} \\log|\\psi(R_b)|`,
local energies :math:`E_{L,b}`, and local-energy derivatives
:math:`g^E_{b,j} = \\partial_{\\theta_j} E_L(R_b)`, define empirical means
:math:`E_0 = \\langle E_L \\rangle`, :math:`\\mu_O = \\langle O \\rangle`,
:math:`\\mu_{g^E} = \\langle g^E \\rangle`, centered quantities
:math:`\\bar{E}_L = E_L - E_0` and :math:`\\bar{O} = O - \\mu_O`, and the
centered Hamiltonian-applied tangent matrix
:math:`\\mathcal{H}_{b,j} = \\bar{E}_{L,b}\\bar{O}_{b,j} + g^E_{b,j}`.

The augmented overlap matrix :math:`S` and energy-centered **non-symmetric**
Hamiltonian matrix :math:`\\Delta H = H - E_0 S` are

.. math::

    S = \\begin{pmatrix} 1 & 0^\\top \\\\ 0 & S_{\\text{param}} \\end{pmatrix},
    \\qquad
    \\Delta H = \\begin{pmatrix} 0 & h_r^\\top \\\\ h_c & \\Delta H_{1:,1:} \\end{pmatrix},

with :math:`S_{\\text{param}} = \\frac{1}{N}\\bar{O}^\\top \\bar{O}` (explicitly
symmetrized), :math:`h_c = \\frac{1}{N}\\bar{O}^\\top \\bar{E}_L`,
:math:`h_r = h_c + \\mu_{g^E}`, and
:math:`\\Delta H_{1:,1:} = \\frac{1}{N}\\bar{O}^\\top \\mathcal{H}`.
:math:`\\Delta H` is **never** symmetrized so that the Toulouse--Umrigar
finite-sample zero-variance principle holds identically on any finite sample.

The stabilized energy-centered generalized eigenproblem

.. math::

    \\big(\\Delta H + c_{\\text{diag}} P_{\\text{deriv}}\\big) c
    = \\Delta\\lambda \\big(S + c_{\\text{ovlp}} P_{\\text{deriv}}\\big) c,
    \\qquad \\lambda = E_0 + \\Delta\\lambda,

is invariant under arbitrary constant energy shifts :math:`E_L \\mapsto E_L + C`
for all :math:`c_{\\text{diag}} \\ge 0` and :math:`c_{\\text{ovlp}} \\ge 0`.
Rank truncation applies strictly to the **raw** overlap eigenvalues before
adding :math:`c_{\\text{ovlp}}` so null modes of :math:`S_{\\text{param}}`
(gauge, structurally disconnected, and :math:`N < P` sample-null modes) are
projected out cleanly before symmetric whitening and dense eigensolution.

Explicit ownership and non-goals
--------------------------------
- Real wavefunctions only (``wavefunction.phase is None``); complex phases,
  non-floating tensors, and graph-bearing inputs are refused loudly.
- Full-parameter dense matrices only; no silent partial-parameter subset or
  hidden iterative fallback.
- Explicit dense working-set byte admission enforced before matrix or
  ``[N, P]`` flattening allocation.
- Two-pass additive statistics route through an injected
  :class:`~tpen.training.statistics.StatisticsReducer` with no process-group
  creation here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import reduce
import math
from math import prod
from typing import Any, Self

from tpen.data.batch import ParameterLayout
from tpen.dependencies import require_torch
from tpen.training.score_geometry import (
    ScoreConventions,
    flatten_parameter_score_blocks,
    layout_convention_fingerprint,
    unflatten_to_layout,
)
from tpen.training.statistics import (
    IdentityStatisticsReducer,
    StatisticsReducer,
    StatisticsSums,
    center_statistics,
)
from tpen.training.step_requests import (
    HamiltonianTangentUpdateInput,
    MaterializedHamiltonianTangentStepRequest,
    MaterializedLocalEnergyDerivatives,
)
from tpen.training.update import (
    ModelParameterBinding,
    UpdateDiagnostics,
    UpdateMethodDescription,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
    flatten_settings,
    json_safe_metric_name,
    json_safe_scalar,
)
from tpen.training.vmc import (
    DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    resolve_nonfinite_local_energy_policy,
)

torch = require_torch(feature="VMC dense linear method")


# Persisted method-state envelope version recorded in trainer checkpoints.
LM_STATE_VERSION = "lm-state-1"

# Fixed pinned tangent-row policy enforcing ruling 377f6b0f / 02859027:
# non-finite scores or local-energy derivatives on a pinned finite-E_L row
# always refuse the step rather than silently re-masking onto a second subsample.
NONFINITE_TANGENT_ROW_POLICY = "fail_on_pinned"

# Worst-case simultaneously live dense solver footprint multiplier:
# W [P^2] + M [(P+1)^2] + complex evecs [2*(P+1)^2] + LAPACK dgeev workspace [~4*(P+1)^2].
DENSE_LM_WORKING_SET_FACTOR = 8

# Default 1 GiB dense working-set guard (admits up to P <= 4,095 in float64).
DEFAULT_MAX_DENSE_BYTES = 1024**3

# Exact factor vocabulary from experiments/atomistic/he-importance/optimizer_grid_authority.json.
LM_GRID_FACTOR_KEYS: frozenset[str] = frozenset(
    {"derivative_block_shift", "overlap_metric_trust_radius"}
)

# Relative eigenvalue tie tolerance used for deterministic eigenpair selection.
_EIGENVALUE_TIE_REL_TOL = 1.0e-12

# Relative tolerance on local energy-descent check h_c . delta_theta <= tol * ||h_c|| * ||delta_theta||.
_DESCENT_REL_TOL = 1.0e-12


class LinearMethodMemoryLimitError(ValueError):
    """Raised when a dense Linear Method parameter layout or step exceeds its memory budget."""


def _is_finite(value: float) -> bool:
    """Return whether a Python float is finite."""

    return math.isfinite(value)


def _validate_positive_int(value: Any, *, field_name: str) -> int:
    """Validate that ``value`` is a positive ``int`` (excluding ``bool``)."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return value


def _validate_nonnegative_int(value: Any, *, field_name: str) -> int:
    """Validate that ``value`` is a non-negative ``int`` (excluding ``bool``)."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")
    return value


def _validate_real_float(
    value: Any,
    *,
    field_name: str,
    min_value: float | None = None,
    min_inclusive: bool = True,
    max_value: float | None = None,
    max_inclusive: bool = True,
) -> float:
    """Validate that ``value`` is a finite real scalar within bounds (excluding ``bool``)."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number, got {value!r}")
    numeric = float(value)
    if not _is_finite(numeric):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if min_value is not None:
        if min_inclusive and numeric < min_value:
            raise ValueError(f"{field_name} must be >= {min_value}, got {numeric!r}")
        if not min_inclusive and numeric <= min_value:
            raise ValueError(f"{field_name} must be > {min_value}, got {numeric!r}")
    if max_value is not None:
        if max_inclusive and numeric > max_value:
            raise ValueError(f"{field_name} must be <= {max_value}, got {numeric!r}")
        if not max_inclusive and numeric >= max_value:
            raise ValueError(f"{field_name} must be < {max_value}, got {numeric!r}")
    return numeric


def _dtype_element_size(dtype: Any, *, field_name: str = "dtype") -> int:
    """Return the byte width of a real floating-point ``torch.dtype``."""

    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"{field_name} must be a torch.dtype, got {type(dtype).__name__}")
    if not dtype.is_floating_point or dtype.is_complex:
        raise TypeError(f"{field_name} must be a real floating torch.dtype, got {dtype}")
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize
    return int(torch.empty((), dtype=dtype).element_size())


def _resolve_layout_input_dtype(layout: ParameterLayout) -> torch.dtype:
    """Resolve the promoted real floating dtype across every slot of a layout."""

    if not isinstance(layout, ParameterLayout):
        raise TypeError(f"layout must be a ParameterLayout, got {type(layout).__name__}")
    layout.validate()
    if not layout.slots or layout.total_numel <= 0:
        raise ValueError("Linear Method requires a non-empty parameter layout")
    for slot in layout.slots:
        if not isinstance(slot.dtype, torch.dtype) or not slot.dtype.is_floating_point or slot.dtype.is_complex:
            raise TypeError(
                f"Linear Method requires real floating parameter slots, got {slot.dtype}"
            )
    return reduce(torch.promote_types, (slot.dtype for slot in layout.slots))


def _validate_real_floating_tensor(tensor: Any, *, field_name: str) -> torch.Tensor:
    """Validate that ``tensor`` is a real floating-point ``torch.Tensor``."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(
            f"{field_name} must have a real floating dtype, got {tensor.dtype}"
        )
    return tensor


def dense_matrix_working_bytes(
    n_parameters: int,
    solve_dtype: torch.dtype = torch.float64,
) -> int:
    """Return the worst-case dense Linear Method solver working set in bytes.

    Computes :math:`B_{\\text{dense}}(P, b) = 8 (P + 1)^2 b`, covering the
    simultaneously live whitening map ``W``, reduced eigenproblem matrix ``M``,
    complex eigenvector matrix, and LAPACK ``dgeev`` workspace at full retained
    rank :math:`r = P`.

    Parameters
    ----------
    n_parameters : int
        Flat trainable parameter count :math:`P \\ge 1`.
    solve_dtype : torch.dtype, optional
        Real floating solve precision (defaults to ``torch.float64``).

    Returns
    -------
    int
        Estimated worst-case dense solver working set in bytes.
    """

    p_count = _validate_positive_int(n_parameters, field_name="n_parameters")
    byte_width = _dtype_element_size(solve_dtype, field_name="solve_dtype")
    return DENSE_LM_WORKING_SET_FACTOR * (p_count + 1) * (p_count + 1) * byte_width


def dense_step_working_bytes(
    n_parameters: int,
    n_samples: int,
    solve_dtype: torch.dtype = torch.float64,
    input_dtype: torch.dtype | None = None,
) -> int:
    """Return the full-step dense Linear Method working set in bytes.

    Computes

    .. math::

        B_{\\text{step}}(N, P, b, b_{\\text{in}})
        = 2 N P b_{\\text{in}}
        + \\max\\big(3 N P b + 2 P^2 b,\\; 8 (P + 1)^2 b\\big),

    accounting for the caller-held materialized tangent packet
    ``(parameter_scores, local_energy_derivatives)`` alongside the peak of the
    assembly phase versus the dense eigensolve phase.

    Parameters
    ----------
    n_parameters : int
        Flat trainable parameter count :math:`P \\ge 1`.
    n_samples : int
        Flattened batch size :math:`N \\ge 0`.
    solve_dtype : torch.dtype, optional
        Real floating solve precision (defaults to ``torch.float64``).
    input_dtype : torch.dtype or None, optional
        Promoted input tensor dtype.  ``None`` uses ``solve_dtype``.

    Returns
    -------
    int
        Estimated full-step peak working set in bytes.
    """

    p_count = _validate_positive_int(n_parameters, field_name="n_parameters")
    sample_count = _validate_nonnegative_int(n_samples, field_name="n_samples")
    solve_width = _dtype_element_size(solve_dtype, field_name="solve_dtype")
    input_width = (
        solve_width
        if input_dtype is None
        else _dtype_element_size(input_dtype, field_name="input_dtype")
    )
    input_bytes = 2 * sample_count * p_count * input_width
    assembly_bytes = 3 * sample_count * p_count * solve_width + 2 * p_count * p_count * solve_width
    solve_bytes = DENSE_LM_WORKING_SET_FACTOR * (p_count + 1) * (p_count + 1) * solve_width
    return input_bytes + max(assembly_bytes, solve_bytes)


@dataclass(frozen=True, kw_only=True)
class LinearMethodPolicy:
    """Stabilization, subspace, rescaling, trust, and memory policy for dense LM.

    Parameters
    ----------
    derivative_block_shift : float, optional
        Diagonal shift :math:`c_{\\text{diag}} \\ge 0` (Hartree) added strictly
        to :math:`\\Delta H_{1:,1:}`.
    overlap_shift : float, optional
        Diagonal metric shift :math:`c_{\\text{ovlp}} \\ge 0` (dimensionless)
        added to retained overlap eigenvalues after pre-shift rank truncation.
    rank_cutoff : float, optional
        Relative eigenvalue threshold :math:`0 \\le \\text{rank\\_cutoff} < 1`
        applied strictly to **raw** overlap eigenvalues before adding
        ``overlap_shift``.
    rescaling_xi : float, optional
        Toulouse--Umrigar tangent rescaling parameter :math:`\\xi \\in [0, 1]`.
        ``0.5`` is Umrigar's norm-preserving rescaling; ``1.0`` is unrescaled;
        ``0.0`` is minimal-change rescaling.
    min_c0 : float, optional
        Minimum unit-norm whitened overlap weight :math:`y_{k,0}^{\\mathbb{R}}`
        required for an eigenpair to be admissible (:math:`0 < c_{0,\\min} < 1`).
    imag_eigenvalue_tol : float, optional
        Relative tolerance :math:`\\ge 0` for treating an eigenvalue as real.
    imag_eigenvector_tol : float, optional
        Euclidean tolerance :math:`\\ge 0` on the imaginary part of a
        phase-aligned unit eigenvector.
    require_descent : bool, optional
        When ``True``, decline any step whose local directional derivative
        exceeds the numerical rounding tolerance.
    step_scale : float, optional
        Multiplicative factor :math:`\\eta_{\\text{LM}} > 0` applied to the
        rescaled direction; must agree with the plain-SGD carrier ``lr``.
    overlap_metric_trust_radius : float or None, optional
        Optional trust cap on the regularized overlap step norm
        :math:`\\|\\Delta\\theta\\|_{\\tilde{S}}`.
    max_update_norm : float or None, optional
        Optional Euclidean cap :math:`\\|\\Delta\\theta\\|_2 \\le C`.  Mutually
        exclusive with ``squared_update_norm_bound``.
    squared_update_norm_bound : float or None, optional
        Optional squared Euclidean cap :math:`\\|\\Delta\\theta\\|_2^2 \\le C`.
        Mutually exclusive with ``max_update_norm``.
    tangent_chunk_size : int or None, optional
        Positive sample chunk bound forwarded to
        :class:`~tpen.training.step_requests.MaterializedHamiltonianTangentStepRequest`.
    max_dense_bytes : int, optional
        Hard byte ceiling on estimated dense working-set memory (defaults to
        1 GiB).
    max_parameters : int or None, optional
        Optional hard parameter-count ceiling.
    """

    derivative_block_shift: float = 1.0e-2
    overlap_shift: float = 0.0
    rank_cutoff: float = 1.0e-12
    rescaling_xi: float = 0.5
    min_c0: float = 1.0e-4
    imag_eigenvalue_tol: float = 1.0e-8
    imag_eigenvector_tol: float = 1.0e-6
    require_descent: bool = True
    step_scale: float = 1.0
    overlap_metric_trust_radius: float | None = None
    max_update_norm: float | None = None
    squared_update_norm_bound: float | None = None
    tangent_chunk_size: int | None = None
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES
    max_parameters: int | None = None

    def __post_init__(self) -> None:
        # Validate every field BEFORE coercing int -> float so bool inputs are
        # always rejected rather than silently converted to 0.0 / 1.0.
        object.__setattr__(
            self,
            "derivative_block_shift",
            _validate_real_float(
                self.derivative_block_shift,
                field_name="LinearMethodPolicy.derivative_block_shift",
                min_value=0.0,
                min_inclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "overlap_shift",
            _validate_real_float(
                self.overlap_shift,
                field_name="LinearMethodPolicy.overlap_shift",
                min_value=0.0,
                min_inclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "rank_cutoff",
            _validate_real_float(
                self.rank_cutoff,
                field_name="LinearMethodPolicy.rank_cutoff",
                min_value=0.0,
                min_inclusive=True,
                max_value=1.0,
                max_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "rescaling_xi",
            _validate_real_float(
                self.rescaling_xi,
                field_name="LinearMethodPolicy.rescaling_xi",
                min_value=0.0,
                min_inclusive=True,
                max_value=1.0,
                max_inclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "min_c0",
            _validate_real_float(
                self.min_c0,
                field_name="LinearMethodPolicy.min_c0",
                min_value=0.0,
                min_inclusive=False,
                max_value=1.0,
                max_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "imag_eigenvalue_tol",
            _validate_real_float(
                self.imag_eigenvalue_tol,
                field_name="LinearMethodPolicy.imag_eigenvalue_tol",
                min_value=0.0,
                min_inclusive=True,
            ),
        )
        object.__setattr__(
            self,
            "imag_eigenvector_tol",
            _validate_real_float(
                self.imag_eigenvector_tol,
                field_name="LinearMethodPolicy.imag_eigenvector_tol",
                min_value=0.0,
                min_inclusive=True,
            ),
        )
        if type(self.require_descent) is not bool:
            raise TypeError(
                "LinearMethodPolicy.require_descent must be a bool, "
                f"got {type(self.require_descent).__name__}"
            )
        object.__setattr__(
            self,
            "step_scale",
            _validate_real_float(
                self.step_scale,
                field_name="LinearMethodPolicy.step_scale",
                min_value=0.0,
                min_inclusive=False,
            ),
        )
        if self.overlap_metric_trust_radius is not None:
            object.__setattr__(
                self,
                "overlap_metric_trust_radius",
                _validate_real_float(
                    self.overlap_metric_trust_radius,
                    field_name="LinearMethodPolicy.overlap_metric_trust_radius",
                    min_value=0.0,
                    min_inclusive=False,
                ),
            )
        if self.max_update_norm is not None:
            object.__setattr__(
                self,
                "max_update_norm",
                _validate_real_float(
                    self.max_update_norm,
                    field_name="LinearMethodPolicy.max_update_norm",
                    min_value=0.0,
                    min_inclusive=False,
                ),
            )
        if self.squared_update_norm_bound is not None:
            object.__setattr__(
                self,
                "squared_update_norm_bound",
                _validate_real_float(
                    self.squared_update_norm_bound,
                    field_name="LinearMethodPolicy.squared_update_norm_bound",
                    min_value=0.0,
                    min_inclusive=False,
                ),
            )
        if self.tangent_chunk_size is not None:
            object.__setattr__(
                self,
                "tangent_chunk_size",
                _validate_positive_int(
                    self.tangent_chunk_size,
                    field_name="LinearMethodPolicy.tangent_chunk_size",
                ),
            )
        object.__setattr__(
            self,
            "max_dense_bytes",
            _validate_positive_int(
                self.max_dense_bytes,
                field_name="LinearMethodPolicy.max_dense_bytes",
            ),
        )
        if self.max_parameters is not None:
            object.__setattr__(
                self,
                "max_parameters",
                _validate_positive_int(
                    self.max_parameters,
                    field_name="LinearMethodPolicy.max_parameters",
                ),
            )
        self.validate()

    def validate(self) -> Self:
        """Validate all numerical bounds, types, and mutual-exclusivity constraints."""

        _validate_real_float(
            self.derivative_block_shift,
            field_name="LinearMethodPolicy.derivative_block_shift",
            min_value=0.0,
            min_inclusive=True,
        )
        _validate_real_float(
            self.overlap_shift,
            field_name="LinearMethodPolicy.overlap_shift",
            min_value=0.0,
            min_inclusive=True,
        )
        _validate_real_float(
            self.rank_cutoff,
            field_name="LinearMethodPolicy.rank_cutoff",
            min_value=0.0,
            min_inclusive=True,
            max_value=1.0,
            max_inclusive=False,
        )
        _validate_real_float(
            self.rescaling_xi,
            field_name="LinearMethodPolicy.rescaling_xi",
            min_value=0.0,
            min_inclusive=True,
            max_value=1.0,
            max_inclusive=True,
        )
        _validate_real_float(
            self.min_c0,
            field_name="LinearMethodPolicy.min_c0",
            min_value=0.0,
            min_inclusive=False,
            max_value=1.0,
            max_inclusive=False,
        )
        _validate_real_float(
            self.imag_eigenvalue_tol,
            field_name="LinearMethodPolicy.imag_eigenvalue_tol",
            min_value=0.0,
            min_inclusive=True,
        )
        _validate_real_float(
            self.imag_eigenvector_tol,
            field_name="LinearMethodPolicy.imag_eigenvector_tol",
            min_value=0.0,
            min_inclusive=True,
        )
        if type(self.require_descent) is not bool:
            raise TypeError("LinearMethodPolicy.require_descent must be a bool")
        _validate_real_float(
            self.step_scale,
            field_name="LinearMethodPolicy.step_scale",
            min_value=0.0,
            min_inclusive=False,
        )
        if self.overlap_metric_trust_radius is not None:
            _validate_real_float(
                self.overlap_metric_trust_radius,
                field_name="LinearMethodPolicy.overlap_metric_trust_radius",
                min_value=0.0,
                min_inclusive=False,
            )
        if self.max_update_norm is not None:
            _validate_real_float(
                self.max_update_norm,
                field_name="LinearMethodPolicy.max_update_norm",
                min_value=0.0,
                min_inclusive=False,
            )
        if self.squared_update_norm_bound is not None:
            _validate_real_float(
                self.squared_update_norm_bound,
                field_name="LinearMethodPolicy.squared_update_norm_bound",
                min_value=0.0,
                min_inclusive=False,
            )
        if self.max_update_norm is not None and self.squared_update_norm_bound is not None:
            raise ValueError(
                "LinearMethodPolicy.max_update_norm and "
                "LinearMethodPolicy.squared_update_norm_bound are mutually exclusive"
            )
        if self.tangent_chunk_size is not None:
            _validate_positive_int(
                self.tangent_chunk_size,
                field_name="LinearMethodPolicy.tangent_chunk_size",
            )
        _validate_positive_int(
            self.max_dense_bytes,
            field_name="LinearMethodPolicy.max_dense_bytes",
        )
        if self.max_parameters is not None:
            _validate_positive_int(
                self.max_parameters,
                field_name="LinearMethodPolicy.max_parameters",
            )
        return self

    def fingerprint(self) -> dict[str, Any]:
        """Return JSON-safe policy identity for telemetry and checkpoint envelopes."""

        return {
            "derivative_block_shift": self.derivative_block_shift,
            "overlap_shift": self.overlap_shift,
            "rank_cutoff": self.rank_cutoff,
            "rescaling_xi": self.rescaling_xi,
            "min_c0": self.min_c0,
            "imag_eigenvalue_tol": self.imag_eigenvalue_tol,
            "imag_eigenvector_tol": self.imag_eigenvector_tol,
            "require_descent": self.require_descent,
            "step_scale": self.step_scale,
            "overlap_metric_trust_radius": self.overlap_metric_trust_radius,
            "max_update_norm": self.max_update_norm,
            "squared_update_norm_bound": self.squared_update_norm_bound,
            "tangent_chunk_size": self.tangent_chunk_size,
            "max_dense_bytes": self.max_dense_bytes,
            "max_parameters": self.max_parameters,
        }


def validate_dense_memory_budget(
    *,
    n_parameters: int,
    solve_dtype: torch.dtype,
    policy: LinearMethodPolicy,
    n_samples: int | None = None,
    input_dtype: torch.dtype | None = None,
) -> int:
    """Enforce parameter-count and dense working-set byte ceilings before allocation.

    Parameters
    ----------
    n_parameters : int
        Flat trainable parameter count :math:`P \\ge 1`.
    solve_dtype : torch.dtype
        Real floating solve precision.
    policy : LinearMethodPolicy
        Active policy carrying ``max_dense_bytes`` and optional ``max_parameters``.
    n_samples : int or None, optional
        When ``None``, validates the matrix-only preflight envelope
        :func:`dense_matrix_working_bytes`.  When provided, validates the
        full-step envelope :func:`dense_step_working_bytes`.
    input_dtype : torch.dtype or None, optional
        Promoted input tensor dtype used when ``n_samples`` is provided.

    Returns
    -------
    int
        Estimated working-set bytes that passed admission.

    Raises
    ------
    LinearMethodMemoryLimitError
        If ``n_parameters`` exceeds ``policy.max_parameters`` or the estimated
        working set exceeds ``policy.max_dense_bytes``.
    """

    if not isinstance(policy, LinearMethodPolicy):
        raise TypeError(
            "validate_dense_memory_budget.policy must be a LinearMethodPolicy, "
            f"got {type(policy).__name__}"
        )
    policy.validate()
    p_count = _validate_positive_int(n_parameters, field_name="n_parameters")
    if n_samples is None:
        estimated_bytes = dense_matrix_working_bytes(p_count, solve_dtype)
    else:
        estimated_bytes = dense_step_working_bytes(
            p_count,
            n_samples,
            solve_dtype,
            input_dtype,
        )

    if policy.max_parameters is not None and p_count > policy.max_parameters:
        raise LinearMethodMemoryLimitError(
            f"dense Linear Method refused P={p_count} parameters: exceeds "
            f"policy.max_parameters={policy.max_parameters} "
            f"(estimated working set {estimated_bytes} bytes in {solve_dtype})"
        )
    if estimated_bytes > policy.max_dense_bytes:
        scope = (
            f"P={p_count}"
            if n_samples is None
            else f"P={p_count}, N={n_samples}"
        )
        raise LinearMethodMemoryLimitError(
            f"dense Linear Method refused {scope} in {solve_dtype}: estimated "
            f"working set {estimated_bytes} bytes exceeds "
            f"policy.max_dense_bytes={policy.max_dense_bytes} bytes"
        )
    return estimated_bytes


@dataclass(frozen=True, kw_only=True)
class LMGridHyperparameters:
    """Validated two-factor Linear Method hyperparameter cell from the authority grid.

    Parameters
    ----------
    derivative_block_shift : float
        Strictly positive finite diagonal shift (Hartree) on :math:`\\Delta H_{1:,1:}`.
    overlap_metric_trust_radius : float
        Strictly positive finite trust radius on :math:`\\|\\Delta\\theta\\|_{\\tilde{S}}`.
    """

    derivative_block_shift: float
    overlap_metric_trust_radius: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "derivative_block_shift",
            _validate_real_float(
                self.derivative_block_shift,
                field_name="LMGridHyperparameters.derivative_block_shift",
                min_value=0.0,
                min_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "overlap_metric_trust_radius",
            _validate_real_float(
                self.overlap_metric_trust_radius,
                field_name="LMGridHyperparameters.overlap_metric_trust_radius",
                min_value=0.0,
                min_inclusive=False,
            ),
        )

    @classmethod
    def from_mapping(cls, factors: Any) -> Self:
        """Parse an exact authority-grid factor mapping into :class:`LMGridHyperparameters`.

        Parameters
        ----------
        factors : Mapping
            Mapping containing exactly :data:`LM_GRID_FACTOR_KEYS`.

        Returns
        -------
        LMGridHyperparameters
            Validated factor bundle.

        Raises
        ------
        TypeError
            If ``factors`` is not a mapping, or if either value is a ``bool``
            or non-real object.
        ValueError
            If keys are missing or unexpected, or if either value is non-finite
            or non-positive.
        """

        if not isinstance(factors, Mapping):
            raise TypeError(
                "LMGridHyperparameters.from_mapping requires a Mapping, "
                f"got {type(factors).__name__}"
            )
        provided_keys = set(factors.keys())
        if provided_keys != LM_GRID_FACTOR_KEYS:
            missing = sorted(LM_GRID_FACTOR_KEYS - provided_keys)
            extra = sorted((key for key in provided_keys if key not in LM_GRID_FACTOR_KEYS), key=str)
            raise ValueError(
                "LMGridHyperparameters factor keys must equal "
                f"{sorted(LM_GRID_FACTOR_KEYS)}; missing={missing}, extra={extra}"
            )
        return cls(
            derivative_block_shift=factors["derivative_block_shift"],
            overlap_metric_trust_radius=factors["overlap_metric_trust_radius"],
        )

    def to_factors(self) -> dict[str, float]:
        """Return the exact authority-grid factor dictionary."""

        return {
            "derivative_block_shift": self.derivative_block_shift,
            "overlap_metric_trust_radius": self.overlap_metric_trust_radius,
        }

    def to_policy(
        self,
        *,
        tangent_chunk_size: int | None = None,
        max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES,
        max_parameters: int | None = None,
        require_descent: bool = True,
        min_c0: float = 1.0e-4,
        imag_eigenvalue_tol: float = 1.0e-8,
        imag_eigenvector_tol: float = 1.0e-6,
    ) -> LinearMethodPolicy:
        """Build the canonical :class:`LinearMethodPolicy` for this grid cell."""

        return LinearMethodPolicy(
            derivative_block_shift=self.derivative_block_shift,
            overlap_shift=0.0,
            rank_cutoff=1.0e-12,
            rescaling_xi=0.5,
            min_c0=min_c0,
            imag_eigenvalue_tol=imag_eigenvalue_tol,
            imag_eigenvector_tol=imag_eigenvector_tol,
            require_descent=require_descent,
            step_scale=1.0,
            overlap_metric_trust_radius=self.overlap_metric_trust_radius,
            max_update_norm=None,
            squared_update_norm_bound=None,
            tangent_chunk_size=tangent_chunk_size,
            max_dense_bytes=max_dense_bytes,
            max_parameters=max_parameters,
        )


def lm_policy_from_factors(
    factors: Any,
    *,
    tangent_chunk_size: int | None = None,
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    max_parameters: int | None = None,
    require_descent: bool = True,
    min_c0: float = 1.0e-4,
    imag_eigenvalue_tol: float = 1.0e-8,
    imag_eigenvector_tol: float = 1.0e-6,
) -> LinearMethodPolicy:
    """Build a :class:`LinearMethodPolicy` from an authority-grid factor mapping.

    Parameters
    ----------
    factors : Mapping or LMGridHyperparameters
        Exact ``{"derivative_block_shift", "overlap_metric_trust_radius"}``
        cell mapping or already-validated :class:`LMGridHyperparameters`.
    tangent_chunk_size : int or None, optional
        Chunk bound forwarded to the tangent step request.
    max_dense_bytes : int, optional
        Hard dense working-set byte ceiling.
    max_parameters : int or None, optional
        Optional parameter-count ceiling.
    require_descent : bool, optional
        Whether to enforce the local energy-descent guard.
    min_c0 : float, optional
        Minimum unit-norm overlap weight for admissible real eigenpairs.
    imag_eigenvalue_tol : float, optional
        Relative tolerance for eigenvalue reality checks.
    imag_eigenvector_tol : float, optional
        Euclidean tolerance on phase-aligned eigenvector imaginary norm.

    Returns
    -------
    LinearMethodPolicy
        Resolved dense Linear Method policy.
    """

    hyper = (
        factors
        if isinstance(factors, LMGridHyperparameters)
        else LMGridHyperparameters.from_mapping(factors)
    )
    return hyper.to_policy(
        tangent_chunk_size=tangent_chunk_size,
        max_dense_bytes=max_dense_bytes,
        max_parameters=max_parameters,
        require_descent=require_descent,
        min_c0=min_c0,
        imag_eigenvalue_tol=imag_eigenvalue_tol,
        imag_eigenvector_tol=imag_eigenvector_tol,
    )


def build_linear_method_update_from_factors(
    factors: Any,
    *,
    model_parameters: ModelParameterBinding,
    optimizer: Any = None,
    conventions: ScoreConventions | None = None,
    reducer: StatisticsReducer | None = None,
    nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    tangent_chunk_size: int | None = None,
    max_dense_bytes: int = DEFAULT_MAX_DENSE_BYTES,
    max_parameters: int | None = None,
    require_descent: bool = True,
    min_c0: float = 1.0e-4,
    imag_eigenvalue_tol: float = 1.0e-8,
    imag_eigenvector_tol: float = 1.0e-6,
) -> LinearMethodUpdate:
    """Build a :class:`LinearMethodUpdate` from an authority-grid factor mapping.

    When ``optimizer`` is omitted, constructs a plain momentum-free
    ``torch.optim.SGD`` carrier bound to ``model_parameters.parameters`` with
    ``lr=policy.step_scale`` (``1.0``).

    Parameters
    ----------
    factors : Mapping or LMGridHyperparameters
        Exact authority-grid factor mapping or :class:`LMGridHyperparameters`.
    model_parameters : ModelParameterBinding
        Live parameter binding for the model under optimization.
    optimizer : torch.optim.SGD or None, optional
        Optional caller-supplied plain-SGD carrier matching ``step_scale``.
    conventions : ScoreConventions or None, optional
        Numerical conventions (defaults to ``ScoreConventions(solve_dtype=torch.float64)``).
    reducer : StatisticsReducer or None, optional
        Statistics reducer (defaults to :class:`IdentityStatisticsReducer`).
    nonfinite_local_energy_policy : {"fail", "mask"}, optional
        Active local-energy non-finite row policy.
    tangent_chunk_size : int or None, optional
        Optional per-chunk tangent evaluation bound.
    max_dense_bytes : int, optional
        Hard dense working-set byte ceiling.
    max_parameters : int or None, optional
        Optional hard parameter-count ceiling.
    require_descent : bool, optional
        Whether to enforce the local energy-descent guard.
    min_c0 : float, optional
        Minimum unit-norm overlap weight for admissible real eigenpairs.
    imag_eigenvalue_tol : float, optional
        Relative tolerance for eigenvalue reality checks.
    imag_eigenvector_tol : float, optional
        Euclidean tolerance on phase-aligned eigenvector imaginary norm.

    Returns
    -------
    LinearMethodUpdate
        Validated dense Linear Method update instance.
    """

    if not isinstance(model_parameters, ModelParameterBinding):
        raise TypeError(
            "build_linear_method_update_from_factors.model_parameters must be "
            f"a ModelParameterBinding, got {type(model_parameters).__name__}"
        )
    policy = lm_policy_from_factors(
        factors,
        tangent_chunk_size=tangent_chunk_size,
        max_dense_bytes=max_dense_bytes,
        max_parameters=max_parameters,
        require_descent=require_descent,
        min_c0=min_c0,
        imag_eigenvalue_tol=imag_eigenvalue_tol,
        imag_eigenvector_tol=imag_eigenvector_tol,
    )
    resolved_optimizer = (
        torch.optim.SGD(model_parameters.parameters, lr=policy.step_scale)
        if optimizer is None
        else optimizer
    )
    return LinearMethodUpdate(
        resolved_optimizer,
        model_parameters=model_parameters,
        policy=policy,
        conventions=conventions,
        reducer=reducer,
        nonfinite_local_energy_policy=nonfinite_local_energy_policy,
    )


def flatten_local_energy_derivative_blocks(
    derivatives: MaterializedLocalEnergyDerivatives,
    *,
    sample_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Flatten parameter-shaped local-energy derivative blocks into ``[N, P]``.

    Column ordering is identical to
    :func:`~tpen.training.score_geometry.flatten_parameter_score_blocks` (layout
    slot order followed by row-major element order within each slot).

    Parameters
    ----------
    derivatives : MaterializedLocalEnergyDerivatives
        Raw uncentered local-energy parameter derivatives.
    sample_shape : tuple of int or None, optional
        Expected leading sample shape, enforced when provided.

    Returns
    -------
    torch.Tensor
        Flattened 2D matrix with shape ``(N, P)``.
    """

    if not isinstance(derivatives, MaterializedLocalEnergyDerivatives):
        raise TypeError(
            "flatten_local_energy_derivative_blocks requires "
            "MaterializedLocalEnergyDerivatives"
        )
    derivatives.validate(sample_shape=sample_shape)
    resolved_shape = derivatives.sample_shape if sample_shape is None else tuple(sample_shape)
    n_samples = int(prod(resolved_shape)) if resolved_shape else 1
    if not derivatives.blocks:
        raise ValueError(
            "flatten_local_energy_derivative_blocks requires at least one parameter slot"
        )
    columns: list[torch.Tensor] = []
    for slot, block in zip(derivatives.layout.slots, derivatives.blocks, strict=True):
        _validate_real_floating_tensor(
            block,
            field_name=f"derivatives.blocks[{slot.ordinal}]",
        )
        columns.append(block.reshape(n_samples, slot.numel))
    return torch.cat(columns, dim=1)


@dataclass(frozen=True, kw_only=True)
class TangentRowSelection:
    """Pinned finite-row mask and non-finite row counts for one tangent step.

    Parameters
    ----------
    finite_mask : torch.Tensor
        1D boolean tensor of shape ``(n_samples,)`` identifying pinned finite
        local-energy rows.
    n_samples : int
        Total flattened sample count.
    n_finite_samples : int
        Count of pinned finite samples.
    n_nonfinite_energy_samples : int
        Count of rows with non-finite local energy.
    n_nonfinite_score_samples : int
        Count of rows with at least one non-finite score coordinate.
    n_nonfinite_derivative_samples : int
        Count of rows with at least one non-finite local-energy derivative.
    nonfinite_local_energy_policy : str
        Active resolved local-energy policy (``"fail"`` or ``"mask"``).
    nonfinite_tangent_row_policy : str
        Fixed pinned tangent-row policy (:data:`NONFINITE_TANGENT_ROW_POLICY`).
    """

    finite_mask: torch.Tensor
    n_samples: int
    n_finite_samples: int
    n_nonfinite_energy_samples: int
    n_nonfinite_score_samples: int
    n_nonfinite_derivative_samples: int
    nonfinite_local_energy_policy: str
    nonfinite_tangent_row_policy: str = NONFINITE_TANGENT_ROW_POLICY


def select_tangent_update_rows(
    local_energy: torch.Tensor,
    score_rows: torch.Tensor,
    derivative_rows: torch.Tensor,
    *,
    nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
) -> TangentRowSelection:
    """Pin finite local-energy rows and enforce ``fail_on_pinned`` on scores and derivatives.

    Implements the consolidated ``377f6b0f`` / ``02859027`` non-finite row
    contract uniformly across ``(local_energy, score_rows, derivative_rows)``:

    1. Pin ``energy_finite = torch.isfinite(local_energy)``.
    2. Under ``nonfinite_local_energy_policy == "fail"``, refuse immediately if
       any local-energy entry is non-finite.
    3. Under BOTH ``"fail"`` and ``"mask"``, refuse loudly if any **pinned**
       row (where ``energy_finite`` is ``True``) carries a non-finite score or
       non-finite local-energy derivative.  Silent re-masking of pinned rows is
       strictly forbidden.

    Parameters
    ----------
    local_energy : torch.Tensor
        1D per-sample local energies of shape ``(N,)``.
    score_rows : torch.Tensor
        2D flattened score matrix of shape ``(N, P)``.
    derivative_rows : torch.Tensor
        2D flattened local-energy derivative matrix of shape ``(N, P)``.
    nonfinite_local_energy_policy : {"fail", "mask"}, optional
        Active local-energy non-finite row policy.

    Returns
    -------
    TangentRowSelection
        Pinned boolean mask and per-array non-finite counts.

    Raises
    ------
    RuntimeError
        If ``nonfinite_local_energy_policy == "fail"`` and any local energy is
        non-finite, or if any pinned finite-``E_L`` row has a non-finite score
        or local-energy derivative.
    """

    resolved_policy = resolve_nonfinite_local_energy_policy(nonfinite_local_energy_policy)
    _validate_real_floating_tensor(local_energy, field_name="local_energy")
    _validate_real_floating_tensor(score_rows, field_name="score_rows")
    _validate_real_floating_tensor(derivative_rows, field_name="derivative_rows")

    if local_energy.ndim != 1:
        raise ValueError(
            f"select_tangent_update_rows.local_energy must be 1D, got shape {tuple(local_energy.shape)}"
        )
    if score_rows.ndim != 2:
        raise ValueError(
            f"select_tangent_update_rows.score_rows must be 2D, got shape {tuple(score_rows.shape)}"
        )
    if derivative_rows.ndim != 2:
        raise ValueError(
            "select_tangent_update_rows.derivative_rows must be 2D, "
            f"got shape {tuple(derivative_rows.shape)}"
        )
    if score_rows.shape != derivative_rows.shape:
        raise ValueError(
            "select_tangent_update_rows score_rows and derivative_rows must share one shape, "
            f"got {tuple(score_rows.shape)} and {tuple(derivative_rows.shape)}"
        )
    if int(local_energy.shape[0]) != int(score_rows.shape[0]):
        raise ValueError(
            "select_tangent_update_rows local_energy length must match score_rows row count, "
            f"got {int(local_energy.shape[0])} and {int(score_rows.shape[0])}"
        )
    if local_energy.device != score_rows.device or derivative_rows.device != score_rows.device:
        raise ValueError(
            "select_tangent_update_rows local_energy, score_rows, and derivative_rows "
            "must share one device"
        )

    n_samples = int(local_energy.numel())
    energy_finite = torch.isfinite(local_energy)
    n_finite_samples = int(energy_finite.sum().item())
    n_nonfinite_energy = n_samples - n_finite_samples

    if resolved_policy == "fail" and n_nonfinite_energy > 0:
        bad_preview = (~energy_finite).nonzero(as_tuple=False).flatten()[:10].tolist()
        raise RuntimeError(
            f"linear method refused a step: {n_nonfinite_energy} of {n_samples} "
            f"local-energy samples are non-finite (first rows {bad_preview}) and the "
            "active nonfinite_local_energy_policy is 'fail'"
        )

    score_finite = torch.isfinite(score_rows).all(dim=1)
    deriv_finite = torch.isfinite(derivative_rows).all(dim=1)
    n_nonfinite_scores = int((~score_finite).sum().item())
    n_nonfinite_derivs = int((~deriv_finite).sum().item())

    pinned_bad_scores = energy_finite & ~score_finite
    if bool(pinned_bad_scores.any()):
        n_bad = int(pinned_bad_scores.sum().item())
        bad_preview = pinned_bad_scores.nonzero(as_tuple=False).flatten()[:10].tolist()
        raise RuntimeError(
            f"linear method refused a step under nonfinite_tangent_row_policy="
            f"{NONFINITE_TANGENT_ROW_POLICY!r} (nonfinite_local_energy_policy="
            f"{resolved_policy!r}): {n_bad} pinned finite-energy row(s) have "
            f"non-finite parameter scores (first rows {bad_preview})"
        )

    pinned_bad_derivs = energy_finite & ~deriv_finite
    if bool(pinned_bad_derivs.any()):
        n_bad = int(pinned_bad_derivs.sum().item())
        bad_preview = pinned_bad_derivs.nonzero(as_tuple=False).flatten()[:10].tolist()
        raise RuntimeError(
            f"linear method refused a step under nonfinite_tangent_row_policy="
            f"{NONFINITE_TANGENT_ROW_POLICY!r} (nonfinite_local_energy_policy="
            f"{resolved_policy!r}): {n_bad} pinned finite-energy row(s) have "
            f"non-finite local-energy derivatives (first rows {bad_preview})"
        )

    return TangentRowSelection(
        finite_mask=energy_finite,
        n_samples=n_samples,
        n_finite_samples=n_finite_samples,
        n_nonfinite_energy_samples=n_nonfinite_energy,
        n_nonfinite_score_samples=n_nonfinite_scores,
        n_nonfinite_derivative_samples=n_nonfinite_derivs,
        nonfinite_local_energy_policy=resolved_policy,
        nonfinite_tangent_row_policy=NONFINITE_TANGENT_ROW_POLICY,
    )


@dataclass(frozen=True, kw_only=True)
class DenseLinearMethodMatrices:
    """Assembled energy-centered dense Linear Method matrices for one step.

    Parameters
    ----------
    count : int
        Reduced global finite sample count :math:`N \\ge 2`.
    energy_mean : float
        Reduced sample mean local energy :math:`E_0 = \\langle E_L \\rangle`.
    S_param : torch.Tensor
        Symmetric ``(P, P)`` parameter-space overlap matrix
        :math:`\\langle \\bar{O}_i \\bar{O}_j \\rangle`.
    h_c : torch.Tensor
        First column ``(P,)`` of :math:`\\Delta H`,
        :math:`\\langle \\bar{O}_i \\bar{E}_L \\rangle`.
    h_r : torch.Tensor
        First row ``(P,)`` of :math:`\\Delta H`,
        :math:`h_c + \\langle g^E \\rangle`.
    dH11 : torch.Tensor
        Non-symmetric ``(P, P)`` derivative block of :math:`\\Delta H`,
        :math:`\\langle \\bar{O}_i (\\bar{E}_L \\bar{O}_j + g^E_j) \\rangle`.
    dtype : str
        Observed string dtype of the assembled solve matrices (e.g.
        ``"torch.float64"``).
    score_mean_norm_sq : float, optional
        Pass-1 squared Euclidean norm :math:`\\|\\mu_O\\|_2^2` used for the
        scale-equivariant centering-roundoff zero-signal floor.  Defaults to
        ``0.0`` for directly constructed matrix fixtures without raw scores.
    """

    count: int
    energy_mean: float
    S_param: torch.Tensor
    h_c: torch.Tensor
    h_r: torch.Tensor
    dH11: torch.Tensor
    dtype: str
    score_mean_norm_sq: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "energy_mean",
            _validate_real_float(
                self.energy_mean,
                field_name="DenseLinearMethodMatrices.energy_mean",
            ),
        )
        object.__setattr__(
            self,
            "score_mean_norm_sq",
            _validate_real_float(
                self.score_mean_norm_sq,
                field_name="DenseLinearMethodMatrices.score_mean_norm_sq",
                min_value=0.0,
                min_inclusive=True,
            ),
        )
        self.validate()

    @property
    def n_parameters(self) -> int:
        """Return the flat parameter dimension ``P``."""

        return int(self.S_param.shape[0])

    def validate(self) -> Self:
        """Validate matrix shapes, real floating dtypes, device agreement, and count."""

        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 2:
            raise ValueError(
                f"DenseLinearMethodMatrices.count must be an integer >= 2, got {self.count!r}"
            )
        _validate_real_float(
            self.energy_mean,
            field_name="DenseLinearMethodMatrices.energy_mean",
        )
        _validate_real_float(
            self.score_mean_norm_sq,
            field_name="DenseLinearMethodMatrices.score_mean_norm_sq",
            min_value=0.0,
            min_inclusive=True,
        )
        for name in ("S_param", "h_c", "h_r", "dH11"):
            _validate_real_floating_tensor(
                getattr(self, name),
                field_name=f"DenseLinearMethodMatrices.{name}",
            )
        if self.S_param.ndim != 2 or self.S_param.shape[0] != self.S_param.shape[1] or self.S_param.shape[0] < 1:
            raise ValueError(
                "DenseLinearMethodMatrices.S_param must be square (P, P) with P >= 1, "
                f"got shape {tuple(self.S_param.shape)}"
            )
        p_count = int(self.S_param.shape[0])
        if tuple(self.h_c.shape) != (p_count,):
            raise ValueError(
                f"DenseLinearMethodMatrices.h_c must have shape {(p_count,)}, "
                f"got {tuple(self.h_c.shape)}"
            )
        if tuple(self.h_r.shape) != (p_count,):
            raise ValueError(
                f"DenseLinearMethodMatrices.h_r must have shape {(p_count,)}, "
                f"got {tuple(self.h_r.shape)}"
            )
        if tuple(self.dH11.shape) != (p_count, p_count):
            raise ValueError(
                f"DenseLinearMethodMatrices.dH11 must have shape {(p_count, p_count)}, "
                f"got {tuple(self.dH11.shape)}"
            )
        shared_dtype = self.S_param.dtype
        shared_device = self.S_param.device
        for name, tensor in (("h_c", self.h_c), ("h_r", self.h_r), ("dH11", self.dH11)):
            if tensor.dtype != shared_dtype:
                raise ValueError(
                    f"DenseLinearMethodMatrices.{name} dtype {tensor.dtype} "
                    f"does not match S_param dtype {shared_dtype}"
                )
            if tensor.device != shared_device:
                raise ValueError(
                    f"DenseLinearMethodMatrices.{name} device {tensor.device} "
                    f"does not match S_param device {shared_device}"
                )
        if not isinstance(self.dtype, str) or self.dtype != str(shared_dtype):
            raise ValueError(
                f"DenseLinearMethodMatrices.dtype must match str(S_param.dtype)={str(shared_dtype)!r}, "
                f"got {self.dtype!r}"
            )
        return self


def assemble_dense_linear_method_matrices(
    local_energy: torch.Tensor,
    score_rows: torch.Tensor,
    derivative_rows: torch.Tensor,
    *,
    solve_dtype: torch.dtype | None = None,
    reducer: StatisticsReducer | None = None,
) -> DenseLinearMethodMatrices:
    """Assemble energy-centered dense Linear Method matrices in two additive passes.

    Every cross-sample sum routes through ``reducer`` (defaulting to
    :class:`~tpen.training.statistics.IdentityStatisticsReducer`).  Pass 1
    reduces first moments ``(sum(E_L), sum(O), sum(g^E))`` and global count
    ``N``; Pass 2 centers locally via
    :func:`~tpen.training.statistics.center_statistics`, forms
    :math:`\\mathcal{H} = \\bar{E}_L \\bar{O} + g^E`, and reduces the unnormalized
    second-moment products before dividing by ``N``.  Only ``S_param`` is
    symmetrized; ``dH11``, ``h_c``, and ``h_r`` remain strictly non-symmetric.

    Parameters
    ----------
    local_energy : torch.Tensor
        1D finite local-energy samples of shape ``(n_local,)``.
    score_rows : torch.Tensor
        2D finite uncentered parameter scores of shape ``(n_local, P)``.
    derivative_rows : torch.Tensor
        2D finite uncentered local-energy derivatives of shape ``(n_local, P)``.
    solve_dtype : torch.dtype or None, optional
        Real floating precision for assembly.  ``None`` promotes across the
        three input tensors.
    reducer : StatisticsReducer or None, optional
        Additive statistics reducer.

    Returns
    -------
    DenseLinearMethodMatrices
        Assembled detached matrices in ``solve_dtype``.
    """

    _validate_real_floating_tensor(local_energy, field_name="local_energy")
    _validate_real_floating_tensor(score_rows, field_name="score_rows")
    _validate_real_floating_tensor(derivative_rows, field_name="derivative_rows")

    if local_energy.ndim != 1:
        raise ValueError(
            "assemble_dense_linear_method_matrices.local_energy must be 1D, "
            f"got shape {tuple(local_energy.shape)}"
        )
    if score_rows.ndim != 2 or score_rows.shape[1] < 1:
        raise ValueError(
            "assemble_dense_linear_method_matrices.score_rows must have shape (N, P) "
            f"with P >= 1, got {tuple(score_rows.shape)}"
        )
    if derivative_rows.ndim != 2 or derivative_rows.shape != score_rows.shape:
        raise ValueError(
            "assemble_dense_linear_method_matrices.derivative_rows must match score_rows shape "
            f"{tuple(score_rows.shape)}, got {tuple(derivative_rows.shape)}"
        )
    if int(local_energy.shape[0]) != int(score_rows.shape[0]):
        raise ValueError(
            "assemble_dense_linear_method_matrices local_energy length must match score_rows "
            f"row count, got {int(local_energy.shape[0])} and {int(score_rows.shape[0])}"
        )
    if local_energy.device != score_rows.device or derivative_rows.device != score_rows.device:
        raise ValueError(
            "assemble_dense_linear_method_matrices inputs must share one device"
        )

    resolved_reducer = IdentityStatisticsReducer() if reducer is None else reducer
    if not isinstance(resolved_reducer, StatisticsReducer):
        raise TypeError(
            "assemble_dense_linear_method_matrices.reducer must be a StatisticsReducer"
        )

    if solve_dtype is None:
        target_dtype = torch.promote_types(
            torch.promote_types(local_energy.dtype, score_rows.dtype),
            derivative_rows.dtype,
        )
    else:
        _dtype_element_size(solve_dtype, field_name="solve_dtype")
        target_dtype = solve_dtype

    n_local = int(score_rows.shape[0])
    n_parameters = int(score_rows.shape[1])
    device = score_rows.device

    with torch.no_grad():
        e_loc = local_energy.detach().to(dtype=target_dtype)
        o_loc = score_rows.detach().to(dtype=target_dtype)
        ge_loc = derivative_rows.detach().to(dtype=target_dtype)

        if n_local > 0 and (
            not bool(torch.isfinite(e_loc).all())
            or not bool(torch.isfinite(o_loc).all())
            or not bool(torch.isfinite(ge_loc).all())
        ):
            raise ValueError(
                "assemble_dense_linear_method_matrices requires finite local_energy, "
                "score_rows, and derivative_rows"
            )

        # Pass 1: reduce count and first-moment sums.
        if n_local == 0:
            local_e_sum = torch.zeros((), dtype=target_dtype, device=device)
            local_o_sum = torch.zeros((n_parameters,), dtype=target_dtype, device=device)
            local_ge_sum = torch.zeros((n_parameters,), dtype=target_dtype, device=device)
        else:
            local_e_sum = e_loc.sum()
            local_o_sum = o_loc.sum(dim=0)
            local_ge_sum = ge_loc.sum(dim=0)

        reduced_first = resolved_reducer.reduce(
            StatisticsSums(
                count=n_local,
                sums=(local_e_sum, local_o_sum, local_ge_sum),
            )
        )
        global_count = reduced_first.count
        if global_count < 2:
            raise ValueError(
                "assemble_dense_linear_method_matrices requires reduced global count >= 2, "
                f"got {global_count}"
            )
        total_e, total_o, total_ge = reduced_first.sums
        global_count_f = float(global_count)
        energy_mean = float((total_e / global_count_f).item())
        score_mean = total_o / global_count_f
        mu_ge = total_ge / global_count_f
        score_mean_norm_sq = float((score_mean * score_mean).sum().item())
        del score_mean

        # Pass 2: center locally using reduced first moments and form second-moment sums.
        if n_local == 0:
            local_hc = torch.zeros((n_parameters,), dtype=target_dtype, device=device)
            local_s = torch.zeros((n_parameters, n_parameters), dtype=target_dtype, device=device)
            local_dh11 = torch.zeros((n_parameters, n_parameters), dtype=target_dtype, device=device)
        else:
            e_bar = center_statistics(e_loc, count=global_count, total=total_e)
            o_bar = center_statistics(o_loc, count=global_count, total=total_o)
            # Centered Hamiltonian-applied tangent H_tan = E_bar[:, None] * O_bar + g^E.
            # Using raw g^E is algebraically exact because <O_bar_i> == 0 identically.
            h_tan = e_bar.unsqueeze(1) * o_bar + ge_loc
            local_hc = o_bar.transpose(0, 1) @ e_bar
            local_s = o_bar.transpose(0, 1) @ o_bar
            local_dh11 = o_bar.transpose(0, 1) @ h_tan
            del e_bar, o_bar, h_tan
        del e_loc, o_loc, ge_loc

        reduced_hc = resolved_reducer.reduce_sum(local_hc)
        del local_hc
        reduced_s = resolved_reducer.reduce_sum(local_s)
        del local_s
        reduced_dh11 = resolved_reducer.reduce_sum(local_dh11)
        del local_dh11

        h_c = reduced_hc / global_count_f
        del reduced_hc
        h_r = h_c + mu_ge
        s_param = reduced_s / global_count_f
        del reduced_s
        # Symmetrize ONLY the overlap matrix S_param; NEVER symmetrize dH11 or h_r/h_c.
        s_param = 0.5 * (s_param + s_param.transpose(0, 1))
        dh11 = reduced_dh11 / global_count_f
        del reduced_dh11

    return DenseLinearMethodMatrices(
        count=global_count,
        energy_mean=energy_mean,
        S_param=s_param,
        h_c=h_c,
        h_r=h_r,
        dH11=dh11,
        dtype=str(target_dtype),
        score_mean_norm_sq=score_mean_norm_sq,
    )


@dataclass(frozen=True, kw_only=True)
class LinearMethodSolveDiagnostics:
    """Observable diagnostics for one dense Linear Method generalized eigensolve.

    Contains 12 always-present solve metadata fields and 13 nullable
    eigenpair/step/min-retained fields.  On skip outcomes where a quantity was
    never computed, its field is strictly ``None`` (emitted as JSON ``null`` or
    an empty CSV cell via :func:`~tpen.training.update.json_safe_scalar`), never
    a fabricated ``0.0``.
    """

    # Always-present fields (12)
    dtype: str
    derivative_block_shift: float
    overlap_shift: float
    overlap_modes: int
    overlap_retained_modes: int
    overlap_truncated_modes: int
    overlap_max_eigenvalue: float
    energy_mean: float
    n_admissible_real_eigenpairs: int
    n_complex_eigenpairs: int
    rescaling_xi: float
    estimated_dense_bytes: int

    # Optional eigenpair / step / min-retained fields (13, ALL float | None = None)
    overlap_min_retained_eigenvalue: float | None = None
    selected_delta_eigenvalue: float | None = None
    selected_eigenvalue: float | None = None
    selected_eigenvalue_imag: float | None = None
    selected_eigenvector_imag_norm: float | None = None
    selected_c0: float | None = None
    rescaling_denominator: float | None = None
    raw_overlap_step_norm: float | None = None
    rescaled_overlap_step_norm: float | None = None
    reg_overlap_step_norm: float | None = None
    applied_overlap_step_norm: float | None = None
    applied_reg_overlap_step_norm: float | None = None
    directional_derivative: float | None = None

    def as_metrics(self, *, prefix: str = "lm_solve") -> dict[str, Any]:
        """Return flat JSON-safe and CSV-safe diagnostic metrics."""

        safe_prefix = json_safe_metric_name(prefix)
        return {
            f"{safe_prefix}_dtype": json_safe_scalar(str(self.dtype)),
            f"{safe_prefix}_derivative_block_shift": json_safe_scalar(
                self.derivative_block_shift
            ),
            f"{safe_prefix}_overlap_shift": json_safe_scalar(self.overlap_shift),
            f"{safe_prefix}_overlap_modes": int(self.overlap_modes),
            f"{safe_prefix}_overlap_retained_modes": int(self.overlap_retained_modes),
            f"{safe_prefix}_overlap_truncated_modes": int(self.overlap_truncated_modes),
            f"{safe_prefix}_overlap_max_eigenvalue": json_safe_scalar(
                self.overlap_max_eigenvalue
            ),
            f"{safe_prefix}_overlap_min_retained_eigenvalue": json_safe_scalar(
                self.overlap_min_retained_eigenvalue
            ),
            f"{safe_prefix}_energy_mean": json_safe_scalar(self.energy_mean),
            f"{safe_prefix}_n_admissible_real_eigenpairs": int(
                self.n_admissible_real_eigenpairs
            ),
            f"{safe_prefix}_n_complex_eigenpairs": int(self.n_complex_eigenpairs),
            f"{safe_prefix}_rescaling_xi": json_safe_scalar(self.rescaling_xi),
            f"{safe_prefix}_estimated_dense_bytes": int(self.estimated_dense_bytes),
            f"{safe_prefix}_selected_delta_eigenvalue": json_safe_scalar(
                self.selected_delta_eigenvalue
            ),
            f"{safe_prefix}_selected_eigenvalue": json_safe_scalar(
                self.selected_eigenvalue
            ),
            f"{safe_prefix}_selected_eigenvalue_imag": json_safe_scalar(
                self.selected_eigenvalue_imag
            ),
            f"{safe_prefix}_selected_eigenvector_imag_norm": json_safe_scalar(
                self.selected_eigenvector_imag_norm
            ),
            f"{safe_prefix}_selected_c0": json_safe_scalar(self.selected_c0),
            f"{safe_prefix}_rescaling_denominator": json_safe_scalar(
                self.rescaling_denominator
            ),
            f"{safe_prefix}_raw_overlap_step_norm": json_safe_scalar(
                self.raw_overlap_step_norm
            ),
            f"{safe_prefix}_rescaled_overlap_step_norm": json_safe_scalar(
                self.rescaled_overlap_step_norm
            ),
            f"{safe_prefix}_reg_overlap_step_norm": json_safe_scalar(
                self.reg_overlap_step_norm
            ),
            f"{safe_prefix}_applied_overlap_step_norm": json_safe_scalar(
                self.applied_overlap_step_norm
            ),
            f"{safe_prefix}_applied_reg_overlap_step_norm": json_safe_scalar(
                self.applied_reg_overlap_step_norm
            ),
            f"{safe_prefix}_directional_derivative": json_safe_scalar(
                self.directional_derivative
            ),
        }


@dataclass(frozen=True, kw_only=True)
class LinearMethodSolveOutcome:
    """Result returned by :func:`solve_dense_linear_method`.

    Parameters
    ----------
    accepted : bool
        ``True`` if a finite, descent-checked rescaled displacement was formed.
    reason : str
        ``"applied"`` when ``accepted=True``, or the explicit skip reason token.
    delta_theta : torch.Tensor or None
        Rescaled displacement :math:`\\Delta\\theta_{\\text{rescaled}}`
        (pre-``step_scale``, pre-``trust_scale``) when ``accepted=True``, else
        ``None``.
    diagnostics : LinearMethodSolveDiagnostics
        Solve diagnostics with ``None`` for any unavailable field.
    update_direction_norm : float or None, optional
        Observed Euclidean norm :math:`\\|\\Delta\\theta_{\\text{rescaled}}\\|_2`
        when a real eigenpair was unwhitened and rescaled (including on
        ``non_descent_step`` and ``nonfinite_update_direction``), else ``None``.
    """

    accepted: bool
    reason: str
    delta_theta: torch.Tensor | None
    diagnostics: LinearMethodSolveDiagnostics
    update_direction_norm: float | None = None


def _solve_dense_linear_method_core(
    *,
    count: int,
    energy_mean: float,
    score_mean_norm_sq: float,
    matrix_box: list[torch.Tensor | None],
    h_c: torch.Tensor,
    h_r: torch.Tensor,
    policy: LinearMethodPolicy,
    energy_gradient_scale: float,
) -> LinearMethodSolveOutcome:
    """Solve the stabilized dense Linear Method eigenproblem with staged deallocation.

    ``matrix_box`` carries ``[S_param, dH11]`` and each slot is cleared to
    ``None`` immediately before its final local reference is deleted so that
    callers releasing ownership (such as :meth:`LinearMethodUpdate.update`)
    reclaim ``S_param``, ``U``, ``dH11``, and ``tmp_HW`` prior to
    ``torch.linalg.eig(M)``.
    """

    s_param = matrix_box[0]
    dh11_check = matrix_box[1]
    assert s_param is not None and dh11_check is not None
    n_parameters = int(s_param.shape[0])
    solve_dtype = s_param.dtype
    dtype_str = str(solve_dtype)

    if (
        not bool(torch.isfinite(s_param).all())
        or not bool(torch.isfinite(h_c).all())
        or not bool(torch.isfinite(h_r).all())
        or not bool(torch.isfinite(dh11_check).all())
    ):
        raise ValueError("solve_dense_linear_method received non-finite matrices")
    del dh11_check

    estimated_dense_bytes = validate_dense_memory_budget(
        n_parameters=n_parameters,
        solve_dtype=solve_dtype,
        policy=policy,
    )

    with torch.no_grad():
        # Step 1: Symmetric overlap eigendecomposition S_param = U diag(w) U^T.
        s_sym = 0.5 * (s_param + s_param.transpose(0, 1))
        matrix_box[0] = None
        del s_param
        w, u_mat = torch.linalg.eigh(s_sym)
        del s_sym
        # Clamp tiny negative rounding artifacts on the PSD overlap matrix.
        w = w.clamp_min(0.0)

        finfo = torch.finfo(solve_dtype)
        w_max = float(w[-1].item())
        w_sum = float(w.sum().item())
        # Scale-equivariant centering-roundoff floor using exact variance identity
        # <||O||_2^2> = tr(S_param) + ||mu_O||_2^2.
        raw_score_scale = w_sum + score_mean_norm_sq
        zero_factor = float(max(count, n_parameters, 16)) * finfo.eps
        tau_zero = max(float(finfo.tiny), (zero_factor * zero_factor) * raw_score_scale)

        if not _is_finite(w_max) or w_max <= tau_zero:
            matrix_box[1] = None
            del w, u_mat
            diagnostics = LinearMethodSolveDiagnostics(
                dtype=dtype_str,
                derivative_block_shift=policy.derivative_block_shift,
                overlap_shift=policy.overlap_shift,
                overlap_modes=n_parameters,
                overlap_retained_modes=0,
                overlap_truncated_modes=n_parameters,
                overlap_max_eigenvalue=w_max,
                energy_mean=energy_mean,
                n_admissible_real_eigenpairs=0,
                n_complex_eigenpairs=0,
                rescaling_xi=policy.rescaling_xi,
                estimated_dense_bytes=estimated_dense_bytes,
            )
            return LinearMethodSolveOutcome(
                accepted=False,
                reason="degenerate_overlap_subspace",
                delta_theta=None,
                diagnostics=diagnostics,
                update_direction_norm=None,
            )

        # Step 2: Apply rank_cutoff strictly to RAW overlap eigenvalues BEFORE
        # adding overlap_shift c_ovlp, projecting out ker(S_param) cleanly.
        tau_rank = max(policy.rank_cutoff, float(n_parameters) * finfo.eps) * w_max
        retained_mask = w > tau_rank
        r = int(retained_mask.sum().item())
        if r == 0:
            matrix_box[1] = None
            del w, u_mat
            diagnostics = LinearMethodSolveDiagnostics(
                dtype=dtype_str,
                derivative_block_shift=policy.derivative_block_shift,
                overlap_shift=policy.overlap_shift,
                overlap_modes=n_parameters,
                overlap_retained_modes=0,
                overlap_truncated_modes=n_parameters,
                overlap_max_eigenvalue=w_max,
                energy_mean=energy_mean,
                n_admissible_real_eigenpairs=0,
                n_complex_eigenpairs=0,
                rescaling_xi=policy.rescaling_xi,
                estimated_dense_bytes=estimated_dense_bytes,
            )
            return LinearMethodSolveOutcome(
                accepted=False,
                reason="empty_retained_subspace",
                delta_theta=None,
                diagnostics=diagnostics,
                update_direction_norm=None,
            )

        w_r = w[retained_mask]
        u_r = u_mat[:, retained_mask]
        # Record raw unshifted minimum retained overlap eigenvalue.
        overlap_min_retained = float(w_r.min().item())
        # Regularize ONLY retained modes with overlap_shift c_ovlp.
        w_reg_r = w_r + policy.overlap_shift
        inv_sqrt_w_reg = w_reg_r.rsqrt()
        # Whitening map W = U_r @ diag(w_reg_r^{-1/2}) [P, r].
        w_map = u_r * inv_sqrt_w_reg.unsqueeze(0)
        # Release full U and w immediately before forming M.
        del u_mat, w, retained_mask

        # Step 3: Build symmetrically whitened (r + 1) x (r + 1) matrix M.
        h_c_tilde = inv_sqrt_w_reg * (u_r.transpose(0, 1) @ h_c)
        h_r_tilde = inv_sqrt_w_reg * (u_r.transpose(0, 1) @ h_r)
        del u_r, inv_sqrt_w_reg

        dh11 = matrix_box[1]
        matrix_box[1] = None
        assert dh11 is not None
        tmp_hw = dh11 @ w_map
        hw = w_map.transpose(0, 1) @ tmp_hw
        # Release dH11 and tmp_HW BEFORE calling torch.linalg.eig(M).
        del dh11, tmp_hw
        if policy.derivative_block_shift != 0.0:
            hw.diagonal().add_(policy.derivative_block_shift / w_reg_r)

        m_mat = torch.zeros((r + 1, r + 1), dtype=solve_dtype, device=w_map.device)
        m_mat[0, 1:] = h_r_tilde
        m_mat[1:, 0] = h_c_tilde
        m_mat[1:, 1:] = hw
        del hw, h_c_tilde, h_r_tilde

        # Step 4: Dense non-symmetric eigensolve M y = dlam y.
        try:
            evals, evecs = torch.linalg.eig(m_mat)
        except torch.linalg.LinAlgError:
            del m_mat, w_map, w_r, w_reg_r
            diagnostics = LinearMethodSolveDiagnostics(
                dtype=dtype_str,
                derivative_block_shift=policy.derivative_block_shift,
                overlap_shift=policy.overlap_shift,
                overlap_modes=n_parameters,
                overlap_retained_modes=r,
                overlap_truncated_modes=n_parameters - r,
                overlap_max_eigenvalue=w_max,
                overlap_min_retained_eigenvalue=overlap_min_retained,
                energy_mean=energy_mean,
                n_admissible_real_eigenpairs=0,
                n_complex_eigenpairs=0,
                rescaling_xi=policy.rescaling_xi,
                estimated_dense_bytes=estimated_dense_bytes,
            )
            return LinearMethodSolveOutcome(
                accepted=False,
                reason="eigensolve_failed",
                delta_theta=None,
                diagnostics=diagnostics,
                update_direction_norm=None,
            )
        del m_mat

        evals_finite = bool(
            torch.isfinite(evals.real).all() and torch.isfinite(evals.imag).all()
        )
        evecs_finite = bool(
            torch.isfinite(evecs.real).all() and torch.isfinite(evecs.imag).all()
        )
        if not evals_finite or not evecs_finite:
            del evals, evecs, w_map, w_r, w_reg_r
            diagnostics = LinearMethodSolveDiagnostics(
                dtype=dtype_str,
                derivative_block_shift=policy.derivative_block_shift,
                overlap_shift=policy.overlap_shift,
                overlap_modes=n_parameters,
                overlap_retained_modes=r,
                overlap_truncated_modes=n_parameters - r,
                overlap_max_eigenvalue=w_max,
                overlap_min_retained_eigenvalue=overlap_min_retained,
                energy_mean=energy_mean,
                n_admissible_real_eigenpairs=0,
                n_complex_eigenpairs=0,
                rescaling_xi=policy.rescaling_xi,
                estimated_dense_bytes=estimated_dense_bytes,
            )
            return LinearMethodSolveOutcome(
                accepted=False,
                reason="nonfinite_eigensolve",
                delta_theta=None,
                diagnostics=diagnostics,
                update_direction_norm=None,
            )

        # Step 5: Complex eigenpair filtering, phase alignment, and c0 admissibility.
        n_complex = 0
        # Tuple fields: (index, lam_re, lam_im, vec_imag_norm, c0, y_unit)
        candidates: list[tuple[int, float, float, float, float, torch.Tensor]] = []
        for idx in range(r + 1):
            lam_re = float(evals[idx].real.item())
            lam_im = float(evals[idx].imag.item())
            imag_tol = policy.imag_eigenvalue_tol * max(1.0, abs(lam_re))
            if abs(lam_im) > imag_tol:
                n_complex += 1
                continue

            y_k = evecs[:, idx]
            y0 = y_k[0]
            abs_y0 = float(torch.abs(y0).item())
            if abs_y0 == 0.0 or not _is_finite(abs_y0):
                continue

            # Rotate by conjugate phase of y_{k,0} so the leading entry is real > 0.
            phase_rot = torch.conj(y0) / abs_y0
            y_tilde = y_k * phase_rot
            vec_imag_norm = float(torch.linalg.vector_norm(y_tilde.imag).item())
            if not _is_finite(vec_imag_norm) or vec_imag_norm > policy.imag_eigenvector_tol:
                continue

            y_real = y_tilde.real.to(dtype=solve_dtype)
            y_real_norm = float(torch.linalg.vector_norm(y_real).item())
            if y_real_norm == 0.0 or not _is_finite(y_real_norm):
                continue
            y_unit = y_real / y_real_norm
            if float(y_unit[0].item()) < 0.0:
                y_unit = -y_unit
            c0_val = float(y_unit[0].item())
            if c0_val < policy.min_c0:
                continue
            candidates.append((idx, lam_re, lam_im, vec_imag_norm, c0_val, y_unit))

        del evals, evecs

        n_admissible = len(candidates)
        if n_admissible == 0:
            del w_map, w_r, w_reg_r
            diagnostics = LinearMethodSolveDiagnostics(
                dtype=dtype_str,
                derivative_block_shift=policy.derivative_block_shift,
                overlap_shift=policy.overlap_shift,
                overlap_modes=n_parameters,
                overlap_retained_modes=r,
                overlap_truncated_modes=n_parameters - r,
                overlap_max_eigenvalue=w_max,
                overlap_min_retained_eigenvalue=overlap_min_retained,
                energy_mean=energy_mean,
                n_admissible_real_eigenpairs=0,
                n_complex_eigenpairs=n_complex,
                rescaling_xi=policy.rescaling_xi,
                estimated_dense_bytes=estimated_dense_bytes,
            )
            return LinearMethodSolveOutcome(
                accepted=False,
                reason="no_admissible_real_eigenpair",
                delta_theta=None,
                diagnostics=diagnostics,
                update_direction_norm=None,
            )

        # Deterministic selection: minimum Re(dlam); within relative tie tolerance,
        # prefer larger c0 (stronger overlap with current state), then smaller Re(dlam),
        # then smaller eigenpair index.
        min_lam_re = min(cand[1] for cand in candidates)
        tie_tol = _EIGENVALUE_TIE_REL_TOL * max(1.0, abs(min_lam_re))
        near_min = [cand for cand in candidates if cand[1] - min_lam_re <= tie_tol]
        _, sel_dlam, sel_dlam_im, sel_vec_imag_norm, sel_c0, y_unit = min(
            near_min,
            key=lambda cand: (-cand[4], cand[1], cand[0]),
        )
        del candidates, near_min

        # Step 6: Unwhiten, compute exact O(r) overlap norms, and apply xi-rescaling.
        y1 = y_unit[1:]
        delta_raw = (w_map @ y1) / sel_c0
        del w_map

        y1_sq = y1 * y1
        inv_c0_sq = 1.0 / (sel_c0 * sel_c0)
        d_tilde = max(0.0, float(y1_sq.sum().item()) * inv_c0_sq)
        if policy.overlap_shift == 0.0:
            d_raw = d_tilde
        else:
            d_raw = max(0.0, float(((w_r / w_reg_r) * y1_sq).sum().item()) * inv_c0_sq)
        del w_r, w_reg_r, y1, y1_sq, y_unit

        raw_overlap_step_norm = math.sqrt(d_raw)
        raw_reg_overlap_step_norm = math.sqrt(d_tilde)

        xi = policy.rescaling_xi
        sqrt_one_plus_d_tilde = math.sqrt(1.0 + d_tilde)
        denom_num = (1.0 - xi) * (1.0 + d_tilde) + xi * sqrt_one_plus_d_tilde
        denom_den = (1.0 - xi) + xi * sqrt_one_plus_d_tilde
        rescaling_denom = denom_num / denom_den

        delta_rescaled = delta_raw / rescaling_denom
        del delta_raw
        rescaled_overlap_step_norm = raw_overlap_step_norm / rescaling_denom
        reg_overlap_step_norm = raw_reg_overlap_step_norm / rescaling_denom
        selected_eigenvalue = energy_mean + sel_dlam

        direction_norm = float(torch.linalg.vector_norm(delta_rescaled).item())
        proposed = delta_rescaled * policy.step_scale
        proposed_norm = float(torch.linalg.vector_norm(proposed).item())

        direction_is_finite = (
            bool(torch.isfinite(delta_rescaled).all())
            and _is_finite(direction_norm)
            and bool(torch.isfinite(proposed).all())
            and _is_finite(proposed_norm)
            and _is_finite(rescaling_denom)
            and _is_finite(rescaled_overlap_step_norm)
            and _is_finite(reg_overlap_step_norm)
        )
        if not direction_is_finite:
            diagnostics = LinearMethodSolveDiagnostics(
                dtype=dtype_str,
                derivative_block_shift=policy.derivative_block_shift,
                overlap_shift=policy.overlap_shift,
                overlap_modes=n_parameters,
                overlap_retained_modes=r,
                overlap_truncated_modes=n_parameters - r,
                overlap_max_eigenvalue=w_max,
                overlap_min_retained_eigenvalue=overlap_min_retained,
                energy_mean=energy_mean,
                n_admissible_real_eigenpairs=n_admissible,
                n_complex_eigenpairs=n_complex,
                rescaling_xi=policy.rescaling_xi,
                estimated_dense_bytes=estimated_dense_bytes,
                selected_delta_eigenvalue=sel_dlam,
                selected_eigenvalue=selected_eigenvalue,
                selected_eigenvalue_imag=sel_dlam_im,
                selected_eigenvector_imag_norm=sel_vec_imag_norm,
                selected_c0=sel_c0,
                rescaling_denominator=rescaling_denom,
                raw_overlap_step_norm=raw_overlap_step_norm,
                rescaled_overlap_step_norm=rescaled_overlap_step_norm,
                reg_overlap_step_norm=reg_overlap_step_norm,
                applied_overlap_step_norm=None,
                applied_reg_overlap_step_norm=None,
                directional_derivative=None,
            )
            return LinearMethodSolveOutcome(
                accepted=False,
                reason="nonfinite_update_direction",
                delta_theta=None,
                diagnostics=diagnostics,
                update_direction_norm=direction_norm,
            )

        # Step 7: Local energy-descent guard h_c . delta_proposed <= tol * ||h_c|| * ||delta_proposed||.
        hc_dot_proposed = float(torch.dot(h_c, proposed).item())
        hc_norm = float(torch.linalg.vector_norm(h_c).item())
        del proposed
        if not _is_finite(hc_dot_proposed) or not _is_finite(hc_norm):
            diagnostics = LinearMethodSolveDiagnostics(
                dtype=dtype_str,
                derivative_block_shift=policy.derivative_block_shift,
                overlap_shift=policy.overlap_shift,
                overlap_modes=n_parameters,
                overlap_retained_modes=r,
                overlap_truncated_modes=n_parameters - r,
                overlap_max_eigenvalue=w_max,
                overlap_min_retained_eigenvalue=overlap_min_retained,
                energy_mean=energy_mean,
                n_admissible_real_eigenpairs=n_admissible,
                n_complex_eigenpairs=n_complex,
                rescaling_xi=policy.rescaling_xi,
                estimated_dense_bytes=estimated_dense_bytes,
                selected_delta_eigenvalue=sel_dlam,
                selected_eigenvalue=selected_eigenvalue,
                selected_eigenvalue_imag=sel_dlam_im,
                selected_eigenvector_imag_norm=sel_vec_imag_norm,
                selected_c0=sel_c0,
                rescaling_denominator=rescaling_denom,
                raw_overlap_step_norm=raw_overlap_step_norm,
                rescaled_overlap_step_norm=rescaled_overlap_step_norm,
                reg_overlap_step_norm=reg_overlap_step_norm,
                applied_overlap_step_norm=None,
                applied_reg_overlap_step_norm=None,
                directional_derivative=None,
            )
            return LinearMethodSolveOutcome(
                accepted=False,
                reason="nonfinite_update_direction",
                delta_theta=None,
                diagnostics=diagnostics,
                update_direction_norm=direction_norm,
            )

        directional_derivative = energy_gradient_scale * hc_dot_proposed
        descent_tol = _DESCENT_REL_TOL * hc_norm * proposed_norm
        if policy.require_descent and hc_dot_proposed > descent_tol:
            diagnostics = LinearMethodSolveDiagnostics(
                dtype=dtype_str,
                derivative_block_shift=policy.derivative_block_shift,
                overlap_shift=policy.overlap_shift,
                overlap_modes=n_parameters,
                overlap_retained_modes=r,
                overlap_truncated_modes=n_parameters - r,
                overlap_max_eigenvalue=w_max,
                overlap_min_retained_eigenvalue=overlap_min_retained,
                energy_mean=energy_mean,
                n_admissible_real_eigenpairs=n_admissible,
                n_complex_eigenpairs=n_complex,
                rescaling_xi=policy.rescaling_xi,
                estimated_dense_bytes=estimated_dense_bytes,
                selected_delta_eigenvalue=sel_dlam,
                selected_eigenvalue=selected_eigenvalue,
                selected_eigenvalue_imag=sel_dlam_im,
                selected_eigenvector_imag_norm=sel_vec_imag_norm,
                selected_c0=sel_c0,
                rescaling_denominator=rescaling_denom,
                raw_overlap_step_norm=raw_overlap_step_norm,
                rescaled_overlap_step_norm=rescaled_overlap_step_norm,
                reg_overlap_step_norm=reg_overlap_step_norm,
                applied_overlap_step_norm=None,
                applied_reg_overlap_step_norm=None,
                directional_derivative=directional_derivative,
            )
            return LinearMethodSolveOutcome(
                accepted=False,
                reason="non_descent_step",
                delta_theta=None,
                diagnostics=diagnostics,
                update_direction_norm=direction_norm,
            )

        diagnostics = LinearMethodSolveDiagnostics(
            dtype=dtype_str,
            derivative_block_shift=policy.derivative_block_shift,
            overlap_shift=policy.overlap_shift,
            overlap_modes=n_parameters,
            overlap_retained_modes=r,
            overlap_truncated_modes=n_parameters - r,
            overlap_max_eigenvalue=w_max,
            overlap_min_retained_eigenvalue=overlap_min_retained,
            energy_mean=energy_mean,
            n_admissible_real_eigenpairs=n_admissible,
            n_complex_eigenpairs=n_complex,
            rescaling_xi=policy.rescaling_xi,
            estimated_dense_bytes=estimated_dense_bytes,
            selected_delta_eigenvalue=sel_dlam,
            selected_eigenvalue=selected_eigenvalue,
            selected_eigenvalue_imag=sel_dlam_im,
            selected_eigenvector_imag_norm=sel_vec_imag_norm,
            selected_c0=sel_c0,
            rescaling_denominator=rescaling_denom,
            raw_overlap_step_norm=raw_overlap_step_norm,
            rescaled_overlap_step_norm=rescaled_overlap_step_norm,
            reg_overlap_step_norm=reg_overlap_step_norm,
            applied_overlap_step_norm=None,
            applied_reg_overlap_step_norm=None,
            directional_derivative=directional_derivative,
        )
        return LinearMethodSolveOutcome(
            accepted=True,
            reason="applied",
            delta_theta=delta_rescaled,
            diagnostics=diagnostics,
            update_direction_norm=direction_norm,
        )


def solve_dense_linear_method(
    matrices: DenseLinearMethodMatrices,
    *,
    policy: LinearMethodPolicy,
    energy_gradient_scale: float = 2.0,
) -> LinearMethodSolveOutcome:
    """Solve the stabilized energy-centered dense Linear Method eigenproblem.

    Parameters
    ----------
    matrices : DenseLinearMethodMatrices
        Assembled overlap and energy-centered non-symmetric Hamiltonian blocks.
    policy : LinearMethodPolicy
        Stabilization shifts, pre-shift rank cutoff, rescaling, admissibility,
        and dense memory-budget policy.
    energy_gradient_scale : float, optional
        Positive scalar converting :math:`h_c` to the VMC energy gradient
        :math:`g_{\\text{VMC}} = c_{\\text{scale}} h_c` (defaults to ``2.0``)
        for ``directional_derivative`` reporting.  Does not affect the
        eigenproblem itself.

    Returns
    -------
    LinearMethodSolveOutcome
        Accepted rescaled displacement ``delta_theta`` (pre-``step_scale``,
        pre-``trust_scale``) or explicit skip outcome with nullable diagnostics.
    """

    if not isinstance(matrices, DenseLinearMethodMatrices):
        raise TypeError(
            "solve_dense_linear_method.matrices must be DenseLinearMethodMatrices, "
            f"got {type(matrices).__name__}"
        )
    matrices.validate()
    if not isinstance(policy, LinearMethodPolicy):
        raise TypeError(
            "solve_dense_linear_method.policy must be a LinearMethodPolicy, "
            f"got {type(policy).__name__}"
        )
    policy.validate()
    resolved_scale = _validate_real_float(
        energy_gradient_scale,
        field_name="energy_gradient_scale",
        min_value=0.0,
        min_inclusive=False,
    )
    matrix_box: list[torch.Tensor | None] = [matrices.S_param, matrices.dH11]
    return _solve_dense_linear_method_core(
        count=matrices.count,
        energy_mean=matrices.energy_mean,
        score_mean_norm_sq=matrices.score_mean_norm_sq,
        matrix_box=matrix_box,
        h_c=matrices.h_c,
        h_r=matrices.h_r,
        policy=policy,
        energy_gradient_scale=resolved_scale,
    )


# Reference used by LinearMethodUpdate.update() to release S_param/dH11 at
# refcount 0 before eig(M) while still honoring monkeypatches in unit tests.
_DEFAULT_SOLVE_DENSE_LINEAR_METHOD = solve_dense_linear_method


@dataclass(frozen=True, kw_only=True)
class LinearMethodTelemetry(UpdateDiagnostics):
    """Observable detached record of one dense Linear Method update attempt.

    Norm semantics match :class:`~tpen.training.sr.SRTelemetry` and
    :class:`~tpen.training.spring.SPRINGTelemetry`:

    - ``energy_gradient_norm`` is the Euclidean VMC energy-gradient norm
      :math:`c_{\\text{scale}}\\|h_c\\|_2` (``0.0`` only on pre-assembly skips).
    - ``update_direction_norm`` is the pre-``step_scale``, pre-``trust_scale``
      Euclidean norm :math:`\\|\\Delta\\theta_{\\text{rescaled}}\\|_2` whenever
      a real eigenpair was unwhitened and rescaled (``0.0`` on earlier skips).
    - ``applied_update_norm`` is
      :math:`\\eta_{\\text{LM}} \\alpha_{\\text{trust}} \\|\\Delta\\theta_{\\text{rescaled}}\\|_2`
      on applied steps and ``0.0`` on any skip.
    """

    applied: bool
    reason: str
    step: int
    n_samples: int
    n_finite_samples: int
    n_nonfinite_energy_samples: int
    n_nonfinite_score_samples: int
    n_nonfinite_derivative_samples: int
    nonfinite_local_energy_policy: str
    nonfinite_tangent_row_policy: str
    n_parameters: int
    energy_gradient_norm: float
    update_direction_norm: float
    applied_update_norm: float
    trust_scale: float
    overlap_trust_scale: float
    euclidean_trust_scale: float
    diagnostics: LinearMethodSolveDiagnostics | None = None

    def as_metrics(self, *, prefix: str = "lm") -> dict[str, Any]:
        """Return flat JSON-safe and CSV-safe telemetry entries."""

        safe_prefix = json_safe_metric_name(prefix)
        metrics: dict[str, Any] = {
            f"{safe_prefix}_applied": bool(self.applied),
            f"{safe_prefix}_reason": json_safe_scalar(self.reason),
            f"{safe_prefix}_step": int(self.step),
            f"{safe_prefix}_samples": int(self.n_samples),
            f"{safe_prefix}_finite_samples": int(self.n_finite_samples),
            f"{safe_prefix}_nonfinite_energy_samples": int(self.n_nonfinite_energy_samples),
            f"{safe_prefix}_nonfinite_score_samples": int(self.n_nonfinite_score_samples),
            f"{safe_prefix}_nonfinite_derivative_samples": int(
                self.n_nonfinite_derivative_samples
            ),
            f"{safe_prefix}_nonfinite_local_energy_policy": json_safe_scalar(
                self.nonfinite_local_energy_policy
            ),
            f"{safe_prefix}_nonfinite_tangent_row_policy": json_safe_scalar(
                self.nonfinite_tangent_row_policy
            ),
            f"{safe_prefix}_parameters": int(self.n_parameters),
            f"{safe_prefix}_energy_gradient_norm": json_safe_scalar(
                self.energy_gradient_norm
            ),
            f"{safe_prefix}_update_direction_norm": json_safe_scalar(
                self.update_direction_norm
            ),
            f"{safe_prefix}_applied_update_norm": json_safe_scalar(
                self.applied_update_norm
            ),
            f"{safe_prefix}_trust_scale": json_safe_scalar(self.trust_scale),
            f"{safe_prefix}_overlap_trust_scale": json_safe_scalar(
                self.overlap_trust_scale
            ),
            f"{safe_prefix}_euclidean_trust_scale": json_safe_scalar(
                self.euclidean_trust_scale
            ),
        }
        if self.diagnostics is not None:
            metrics.update(self.diagnostics.as_metrics(prefix=f"{prefix}_solve"))
        return metrics


class LinearMethodUpdate(VMCUpdateMethod[HamiltonianTangentUpdateInput]):
    """Dense Hamiltonian-tangent Linear Method VMC update method.

    Parameters
    ----------
    optimizer : torch.optim.SGD
        Plain momentum-free, decay-free SGD carrier whose ``lr`` equals
        ``policy.step_scale``.
    model_parameters : ModelParameterBinding
        Authoritative live trainable-parameter binding for the model.
    policy : LinearMethodPolicy or None, optional
        Stabilization, rank-cutoff, rescaling, trust, and memory policy.
    conventions : ScoreConventions or None, optional
        Numerical conventions.  Defaults to
        ``ScoreConventions(energy_gradient_scale=2.0, solve_dtype=torch.float64)``.
    reducer : StatisticsReducer or None, optional
        Additive statistics reducer (defaults to :class:`IdentityStatisticsReducer`).
    nonfinite_local_energy_policy : {"fail", "mask"}, optional
        Active local-energy non-finite row policy.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        model_parameters: ModelParameterBinding,
        policy: LinearMethodPolicy | None = None,
        conventions: ScoreConventions | None = None,
        reducer: StatisticsReducer | None = None,
        nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    ) -> None:
        self.nonfinite_local_energy_policy = resolve_nonfinite_local_energy_policy(
            nonfinite_local_energy_policy
        )
        self.nonfinite_tangent_row_policy = NONFINITE_TANGENT_ROW_POLICY

        resolved_policy = LinearMethodPolicy() if policy is None else policy
        if not isinstance(resolved_policy, LinearMethodPolicy):
            raise TypeError(
                "LinearMethodUpdate.policy must be a LinearMethodPolicy, "
                f"got {type(resolved_policy).__name__}"
            )
        resolved_policy.validate()
        _validate_plain_sgd(optimizer, step_scale=resolved_policy.step_scale)

        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError(
                "LinearMethodUpdate.model_parameters must be a ModelParameterBinding, "
                f"got {type(model_parameters).__name__}"
            )
        effective_input_dtype = _validate_model_parameters(model_parameters)

        resolved_conventions = (
            ScoreConventions(energy_gradient_scale=2.0, solve_dtype=torch.float64)
            if conventions is None
            else conventions
        )
        if not isinstance(resolved_conventions, ScoreConventions):
            raise TypeError(
                "LinearMethodUpdate.conventions must be a ScoreConventions, "
                f"got {type(resolved_conventions).__name__}"
            )
        resolved_conventions.validate()
        if resolved_conventions.energy_gradient_scale <= 0.0:
            raise ValueError(
                "LinearMethodUpdate requires conventions.energy_gradient_scale > 0.0, "
                f"got {resolved_conventions.energy_gradient_scale!r}"
            )

        resolved_reducer = IdentityStatisticsReducer() if reducer is None else reducer
        if not isinstance(resolved_reducer, StatisticsReducer):
            raise TypeError(
                "LinearMethodUpdate.reducer must be a StatisticsReducer, "
                f"got {type(resolved_reducer).__name__}"
            )

        effective_solve_dtype = (
            effective_input_dtype
            if resolved_conventions.solve_dtype is None
            else resolved_conventions.solve_dtype
        )
        assert model_parameters.layout is not None
        # Preflight dense memory admission BEFORE any sampling or tangent pass.
        validate_dense_memory_budget(
            n_parameters=model_parameters.layout.total_numel,
            solve_dtype=effective_solve_dtype,
            policy=resolved_policy,
        )

        self.optimizer = optimizer
        self.model_parameters = model_parameters
        self.policy = resolved_policy
        self.conventions = resolved_conventions
        self.reducer = resolved_reducer
        self.completed_updates = 0
        self._last_result: VMCUpdateResult | None = None

    @property
    def last_telemetry(self) -> LinearMethodTelemetry | None:
        """Return the diagnostics from the most recent attempt, or ``None``."""

        result = self._last_result
        if result is None:
            return None
        diagnostics = result.diagnostics
        assert diagnostics is None or isinstance(diagnostics, LinearMethodTelemetry)
        return diagnostics

    def step_request(self) -> MaterializedHamiltonianTangentStepRequest:
        """Request materialized parameter scores and local-energy derivatives in one pass."""

        return MaterializedHamiltonianTangentStepRequest(
            chunk_size=self.policy.tangent_chunk_size
        )

    def declared_nonfinite_local_energy_policy(self) -> str:
        """Return the resolved non-finite local-energy policy declared by this method."""

        return self.nonfinite_local_energy_policy

    def norm_semantics(self) -> str:
        """Report ``grad_norm`` as the Euclidean energy-gradient norm."""

        return "energy_gradient_l2_preconditioned_direction_in_grad"

    def describe(self) -> UpdateMethodDescription:
        """Extend the generic description with LM policy, conventions, and row policies."""

        base = super().describe()
        settings = dict(base.settings)
        settings.update(flatten_settings(self.policy.fingerprint(), prefix="policy_"))
        settings.update(flatten_settings(self.conventions.fingerprint(), prefix="conventions_"))
        settings["reducer_class"] = type(self.reducer).__qualname__
        settings["nonfinite_local_energy_policy"] = self.nonfinite_local_energy_policy
        settings["nonfinite_tangent_row_policy"] = self.nonfinite_tangent_row_policy
        return replace(base, settings=settings)

    def update_state(self) -> VMCUpdateState:
        """Return the plain-SGD carrier and live parameter binding owned by this method."""

        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Adopt restored live model parameter references after validating dtype and budget."""

        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError(
                "LinearMethodUpdate.model_parameters must be a ModelParameterBinding, "
                f"got {type(model_parameters).__name__}"
            )
        effective_input_dtype = _validate_model_parameters(model_parameters)
        effective_solve_dtype = (
            effective_input_dtype
            if self.conventions.solve_dtype is None
            else self.conventions.solve_dtype
        )
        assert model_parameters.layout is not None
        validate_dense_memory_budget(
            n_parameters=model_parameters.layout.total_numel,
            solve_dtype=effective_solve_dtype,
            policy=self.policy,
        )
        self.model_parameters = model_parameters

    def method_state_dict(self) -> dict[str, Any]:
        """Return the JSON-safe versioned ``lm-state-1`` checkpoint envelope."""

        assert self.model_parameters.layout is not None
        return {
            "version": LM_STATE_VERSION,
            "fingerprint": layout_convention_fingerprint(
                self.model_parameters.layout,
                self.conventions,
            ),
            "policy": self.policy.fingerprint(),
            "nonfinite_local_energy_policy": self.nonfinite_local_energy_policy,
            "nonfinite_tangent_row_policy": self.nonfinite_tangent_row_policy,
            "completed_updates": self.completed_updates,
        }

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore ``lm-state-1`` fail-closed, rejecting any identity or policy drift."""

        if not isinstance(state, Mapping):
            raise TypeError("LinearMethodUpdate state must be a mapping")
        version = state.get("version")
        if version != LM_STATE_VERSION:
            raise ValueError(
                f"unsupported Linear Method state version {version!r}, "
                f"expected {LM_STATE_VERSION!r}"
            )
        assert self.model_parameters.layout is not None
        expected = layout_convention_fingerprint(
            self.model_parameters.layout,
            self.conventions,
        )
        recorded = state.get("fingerprint")
        if not isinstance(recorded, Mapping) or recorded.get("digest") != expected["digest"]:
            raise ValueError(
                "Linear Method checkpoint layout/convention fingerprint does not match "
                "the live model"
            )
        if state.get("policy") != self.policy.fingerprint():
            raise ValueError(
                "Linear Method checkpoint policy does not match the live method"
            )
        if state.get("nonfinite_local_energy_policy") != self.nonfinite_local_energy_policy:
            raise ValueError(
                "Linear Method checkpoint nonfinite_local_energy_policy "
                f"{state.get('nonfinite_local_energy_policy')!r} does not match "
                f"the live method ({self.nonfinite_local_energy_policy!r})"
            )
        if state.get("nonfinite_tangent_row_policy") != self.nonfinite_tangent_row_policy:
            raise ValueError(
                "Linear Method checkpoint nonfinite_tangent_row_policy "
                f"{state.get('nonfinite_tangent_row_policy')!r} does not match "
                f"the live method ({self.nonfinite_tangent_row_policy!r})"
            )
        if "completed_updates" not in state:
            raise ValueError("Linear Method state is missing required key 'completed_updates'")
        completed = state["completed_updates"]
        if type(completed) is not int or completed < 0:
            raise ValueError(
                "Linear Method state completed_updates must be a non-negative integer"
            )
        self.completed_updates = completed

    def update(self, update_input: HamiltonianTangentUpdateInput) -> VMCUpdateResult:
        """Assemble dense LM matrices, solve the stabilized eigenproblem, and step SGD."""

        if not isinstance(update_input, HamiltonianTangentUpdateInput):
            raise TypeError("LinearMethodUpdate requires HamiltonianTangentUpdateInput")
        update_input.validate()
        self._validate_real_and_detached(update_input)
        self._validate_binding(update_input)

        local_energy = update_input.local_energy.reshape(-1)
        n_samples = int(local_energy.numel())
        layout = update_input.parameter_scores.layout
        n_parameters = layout.total_numel
        effective_input_dtype = _resolve_layout_input_dtype(layout)
        effective_solve_dtype = (
            effective_input_dtype
            if self.conventions.solve_dtype is None
            else self.conventions.solve_dtype
        )

        # Enforce full-step dense memory budget BEFORE flattening [N, P] blocks
        # or selecting rows, guaranteeing zero [N, P] allocation prior to refusal.
        validate_dense_memory_budget(
            n_parameters=n_parameters,
            n_samples=n_samples,
            solve_dtype=effective_solve_dtype,
            input_dtype=effective_input_dtype,
            policy=self.policy,
        )

        with torch.no_grad():
            sample_shape = tuple(update_input.wavefunction.logabs.shape)
            score_rows = flatten_parameter_score_blocks(
                update_input.parameter_scores,
                sample_shape=sample_shape,
            )
            derivative_rows = flatten_local_energy_derivative_blocks(
                update_input.local_energy_derivatives,
                sample_shape=sample_shape,
            )

            selection = select_tangent_update_rows(
                local_energy,
                score_rows,
                derivative_rows,
                nonfinite_local_energy_policy=self.nonfinite_local_energy_policy,
            )

            # Degenerate batch guards in exact SR/SPRING order.
            if update_input.batch.n_electrons == 0:
                del score_rows, derivative_rows
                return self._skip(
                    reason="zero_electron_batch",
                    step=update_input.step,
                    selection=selection,
                    n_parameters=n_parameters,
                )
            if selection.n_finite_samples == 0:
                del score_rows, derivative_rows
                raise RuntimeError(
                    "cannot compute a linear-method update: no finite local-energy "
                    "sample remains for a nonzero-electron batch"
                )
            if selection.n_finite_samples < 2:
                del score_rows, derivative_rows
                return self._skip(
                    reason="insufficient_finite_samples",
                    step=update_input.step,
                    selection=selection,
                    n_parameters=n_parameters,
                )

            if selection.n_finite_samples != selection.n_samples:
                score_rows = score_rows[selection.finite_mask]
                derivative_rows = derivative_rows[selection.finite_mask]
                local_energy = local_energy[selection.finite_mask]

            matrices = assemble_dense_linear_method_matrices(
                local_energy,
                score_rows,
                derivative_rows,
                solve_dtype=effective_solve_dtype,
                reducer=self.reducer,
            )
            # Release flattened [N, P] matrices immediately after P x P assembly.
            del score_rows, derivative_rows

            h_c = matrices.h_c
            energy_gradient_norm = (
                self.conventions.energy_gradient_scale
                * float(torch.linalg.vector_norm(h_c).item())
            )

            if solve_dense_linear_method is _DEFAULT_SOLVE_DENSE_LINEAR_METHOD:
                # Hand S_param and dH11 through a mutable box and release `matrices`
                # so S_param and dH11 hit refcount 0 prior to torch.linalg.eig(M).
                matrix_box: list[torch.Tensor | None] = [matrices.S_param, matrices.dH11]
                h_r = matrices.h_r
                count = matrices.count
                energy_mean = matrices.energy_mean
                score_mean_norm_sq = matrices.score_mean_norm_sq
                del matrices
                outcome = _solve_dense_linear_method_core(
                    count=count,
                    energy_mean=energy_mean,
                    score_mean_norm_sq=score_mean_norm_sq,
                    matrix_box=matrix_box,
                    h_c=h_c,
                    h_r=h_r,
                    policy=self.policy,
                    energy_gradient_scale=self.conventions.energy_gradient_scale,
                )
                del h_r, matrix_box
            else:
                outcome = solve_dense_linear_method(
                    matrices,
                    policy=self.policy,
                    energy_gradient_scale=self.conventions.energy_gradient_scale,
                )
                del matrices

            if not outcome.accepted:
                return self._skip(
                    reason=outcome.reason,
                    step=update_input.step,
                    selection=selection,
                    n_parameters=n_parameters,
                    energy_gradient_norm=energy_gradient_norm,
                    update_direction_norm=(
                        0.0
                        if outcome.update_direction_norm is None
                        else outcome.update_direction_norm
                    ),
                    diagnostics=outcome.diagnostics,
                )

            assert outcome.delta_theta is not None
            delta_rescaled = outcome.delta_theta
            direction_norm = (
                float(torch.linalg.vector_norm(delta_rescaled).item())
                if outcome.update_direction_norm is None
                else outcome.update_direction_norm
            )
            assert outcome.diagnostics.rescaled_overlap_step_norm is not None
            assert outcome.diagnostics.reg_overlap_step_norm is not None

            overlap_trust_scale, euclidean_trust_scale = self._trust_scales(
                direction_norm=direction_norm,
                reg_overlap_step_norm=outcome.diagnostics.reg_overlap_step_norm,
            )
            trust_scale = min(overlap_trust_scale, euclidean_trust_scale)

            applied_scale = self.policy.step_scale * trust_scale
            applied_update_norm = applied_scale * direction_norm
            applied_overlap_step_norm = (
                applied_scale * outcome.diagnostics.rescaled_overlap_step_norm
            )
            applied_reg_overlap_step_norm = (
                applied_scale * outcome.diagnostics.reg_overlap_step_norm
            )
            applied_directional_derivative = (
                self.conventions.energy_gradient_scale
                * float(torch.dot(h_c, delta_rescaled * applied_scale).item())
            )
            # Plain SGD applies theta_{t+1} = theta_t - step_scale * grad.
            # Storing grad = -(trust_scale * delta_rescaled) moves parameters by
            # exactly +step_scale * trust_scale * delta_rescaled with no division.
            grad_vector = -(trust_scale * delta_rescaled)

            final_diagnostics = replace(
                outcome.diagnostics,
                applied_overlap_step_norm=applied_overlap_step_norm,
                applied_reg_overlap_step_norm=applied_reg_overlap_step_norm,
                directional_derivative=applied_directional_derivative,
            )

        self._apply(grad_vector)
        self.completed_updates += 1
        telemetry = LinearMethodTelemetry(
            applied=True,
            reason="applied",
            step=update_input.step,
            n_samples=selection.n_samples,
            n_finite_samples=selection.n_finite_samples,
            n_nonfinite_energy_samples=selection.n_nonfinite_energy_samples,
            n_nonfinite_score_samples=selection.n_nonfinite_score_samples,
            n_nonfinite_derivative_samples=selection.n_nonfinite_derivative_samples,
            nonfinite_local_energy_policy=selection.nonfinite_local_energy_policy,
            nonfinite_tangent_row_policy=selection.nonfinite_tangent_row_policy,
            n_parameters=n_parameters,
            energy_gradient_norm=energy_gradient_norm,
            update_direction_norm=direction_norm,
            applied_update_norm=applied_update_norm,
            trust_scale=trust_scale,
            overlap_trust_scale=overlap_trust_scale,
            euclidean_trust_scale=euclidean_trust_scale,
            diagnostics=final_diagnostics,
        )
        return self._record_result(
            applied=True,
            grad_norm=energy_gradient_norm,
            reason=telemetry.reason,
            diagnostics=telemetry,
        )

    def _trust_scales(
        self,
        *,
        direction_norm: float,
        reg_overlap_step_norm: float,
    ) -> tuple[float, float]:
        """Return ``(overlap_trust_scale, euclidean_trust_scale)`` in ``(0, 1]``."""

        proposed_reg_ovlp = self.policy.step_scale * reg_overlap_step_norm
        ovlp_radius = self.policy.overlap_metric_trust_radius
        if ovlp_radius is None or proposed_reg_ovlp == 0.0 or proposed_reg_ovlp <= ovlp_radius:
            overlap_scale = 1.0
        else:
            overlap_scale = ovlp_radius / proposed_reg_ovlp

        proposed_euclid = self.policy.step_scale * direction_norm
        if self.policy.max_update_norm is not None:
            if proposed_euclid == 0.0 or proposed_euclid <= self.policy.max_update_norm:
                euclid_scale = 1.0
            else:
                euclid_scale = self.policy.max_update_norm / proposed_euclid
        elif self.policy.squared_update_norm_bound is not None:
            proposed_sq = proposed_euclid * proposed_euclid
            if proposed_sq == 0.0 or proposed_sq <= self.policy.squared_update_norm_bound:
                euclid_scale = 1.0
            else:
                euclid_scale = (
                    math.sqrt(self.policy.squared_update_norm_bound) / proposed_euclid
                )
        else:
            euclid_scale = 1.0

        return overlap_scale, euclid_scale

    def _apply(self, grad_vector: torch.Tensor) -> None:
        """Write the negated preconditioned direction into ``.grad`` and step plain SGD."""

        assert self.model_parameters.layout is not None
        blocks = unflatten_to_layout(grad_vector, layout=self.model_parameters.layout)
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
        selection: TangentRowSelection,
        n_parameters: int,
        energy_gradient_norm: float = 0.0,
        update_direction_norm: float = 0.0,
        diagnostics: LinearMethodSolveDiagnostics | None = None,
    ) -> VMCUpdateResult:
        """Record an atomic declined step without mutating parameters or optimizer state."""

        telemetry = LinearMethodTelemetry(
            applied=False,
            reason=reason,
            step=step,
            n_samples=selection.n_samples,
            n_finite_samples=selection.n_finite_samples,
            n_nonfinite_energy_samples=selection.n_nonfinite_energy_samples,
            n_nonfinite_score_samples=selection.n_nonfinite_score_samples,
            n_nonfinite_derivative_samples=selection.n_nonfinite_derivative_samples,
            nonfinite_local_energy_policy=selection.nonfinite_local_energy_policy,
            nonfinite_tangent_row_policy=selection.nonfinite_tangent_row_policy,
            n_parameters=n_parameters,
            energy_gradient_norm=energy_gradient_norm,
            update_direction_norm=update_direction_norm,
            applied_update_norm=0.0,
            trust_scale=1.0,
            overlap_trust_scale=1.0,
            euclidean_trust_scale=1.0,
            diagnostics=diagnostics,
        )
        return self._record_result(
            applied=False,
            grad_norm=energy_gradient_norm,
            reason=reason,
            diagnostics=telemetry,
        )

    def _validate_real_and_detached(
        self,
        update_input: HamiltonianTangentUpdateInput,
    ) -> None:
        """Reject complex phases, non-real tensors, and graph-bearing step inputs."""

        if update_input.wavefunction.phase is not None:
            raise ValueError(
                "LinearMethodUpdate supports real wavefunctions only; "
                "the step carries a complex phase that this method does not form"
            )
        _validate_real_floating_tensor(
            update_input.local_energy,
            field_name="update_input.local_energy",
        )
        for idx, block in enumerate(update_input.parameter_scores.blocks):
            _validate_real_floating_tensor(
                block,
                field_name=f"update_input.parameter_scores.blocks[{idx}]",
            )
        for idx, block in enumerate(update_input.local_energy_derivatives.blocks):
            _validate_real_floating_tensor(
                block,
                field_name=f"update_input.local_energy_derivatives.blocks[{idx}]",
            )
        if (
            update_input.wavefunction.logabs.requires_grad
            or update_input.wavefunction.sign.requires_grad
            or update_input.local_energy.requires_grad
            or any(block.requires_grad for block in update_input.parameter_scores.blocks)
            or any(
                block.requires_grad
                for block in update_input.local_energy_derivatives.blocks
            )
        ):
            raise ValueError(
                "LinearMethodUpdate requires fully detached step tensors; "
                "received a graph-bearing tensor"
            )

    def _validate_binding(self, update_input: HamiltonianTangentUpdateInput) -> None:
        """Require the tangent packet to match this method's live parameter binding."""

        assert self.model_parameters.layout is not None
        if not update_input.parameter_scores.layout.compare(self.model_parameters.layout)[0]:
            raise ValueError(
                "Linear Method tangent packet layout does not match the bound parameter layout"
            )
        bound = self.model_parameters.parameters
        step_parameters = update_input.parameter_binding.parameters
        if len(bound) != len(step_parameters) or any(
            left is not right for left, right in zip(bound, step_parameters, strict=True)
        ):
            raise ValueError(
                "Linear Method parameter binding does not reference the bound live "
                "parameters; a rebind was missed after a checkpoint restore"
            )


def _validate_model_parameters(model_parameters: ModelParameterBinding) -> torch.dtype:
    """Validate non-empty real floating parameter binding and return promoted dtype."""

    model_parameters.validate()
    assert model_parameters.layout is not None
    for idx, parameter in enumerate(model_parameters.parameters):
        if not parameter.is_floating_point() or parameter.is_complex():
            raise TypeError(
                f"LinearMethodUpdate parameter {idx} must have a real floating dtype, "
                f"got {parameter.dtype}"
            )
    return _resolve_layout_input_dtype(model_parameters.layout)


def _validate_plain_sgd(optimizer: Any, *, step_scale: float) -> None:
    """Require a plain momentum-free, decay-free SGD whose ``lr`` matches ``step_scale``."""

    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError(
            "LinearMethodUpdate requires a torch.optim.SGD optimizer so the applied "
            "step is exactly +step_scale * trust_scale * delta_theta_rescaled; got "
            f"{type(optimizer).__name__}. An adaptive optimizer such as Adam is never "
            "an acceptable substitute here."
        )
    for index, group in enumerate(optimizer.param_groups):
        for key in ("momentum", "dampening", "weight_decay"):
            value = group.get(key, 0.0)
            if float(value) != 0.0:
                raise ValueError(
                    f"LinearMethodUpdate requires SGD param group {index} to have "
                    f"{key}=0, got {value}"
                )
        for key in ("nesterov", "maximize"):
            if bool(group.get(key, False)):
                raise ValueError(
                    f"LinearMethodUpdate requires SGD param group {index} to have "
                    f"{key} disabled"
                )
        group_lr = float(group.get("lr"))
        if group_lr != float(step_scale):
            raise ValueError(
                f"LinearMethodUpdate SGD param group {index} has lr={group_lr}, "
                f"which disagrees with LinearMethodPolicy.step_scale={step_scale}"
            )


__all__ = [
    "DEFAULT_MAX_DENSE_BYTES",
    "DENSE_LM_WORKING_SET_FACTOR",
    "LM_GRID_FACTOR_KEYS",
    "LM_STATE_VERSION",
    "NONFINITE_TANGENT_ROW_POLICY",
    "DenseLinearMethodMatrices",
    "LMGridHyperparameters",
    "LinearMethodMemoryLimitError",
    "LinearMethodPolicy",
    "LinearMethodSolveDiagnostics",
    "LinearMethodSolveOutcome",
    "LinearMethodTelemetry",
    "LinearMethodUpdate",
    "TangentRowSelection",
    "assemble_dense_linear_method_matrices",
    "build_linear_method_update_from_factors",
    "dense_matrix_working_bytes",
    "dense_step_working_bytes",
    "flatten_local_energy_derivative_blocks",
    "lm_policy_from_factors",
    "select_tangent_update_rows",
    "solve_dense_linear_method",
    "validate_dense_memory_budget",
]
