"""Canonical short-range Kato cusp factors for electron-electron and electron-nucleus pairs.

Both cusp factors here are `tpen.nn.factor.LogAmplitudeFactor` instances and
compose directly through `TPENWaveFunction.factors`; `ElectronElectronCusp`
additionally remains a `tpen.nn.envelope.Envelope` so it stays usable in the
legacy `AdditiveEnvelope` compatibility stack. Electron-nucleus cusp laws
separate the charge-fixed Kato first radial slope (`LinearElectronNucleusCuspLaw`,
the exact compatibility default) from an optional trainable regular curvature
term (`CurvatureElectronNucleusCuspLaw`) that contributes only at
second order, so it can never perturb the enforced ``u'(0+) = -Z`` slope. That
curvature term carries the electron-nucleus counterpart of
`ElectronElectronCusp`'s trainable ``range_parameter``; its outer-tail
consequence is stated on the class and is executable through
`CurvatureElectronNucleusCuspLaw.outer_tail_slope`.

`TailSafeElectronNucleusCuspLaw` is the same functional form under a different
coordinate system, ``c = d (Z - kappa)``, chosen so the outer radial slope is
``-kappa < 0`` for EVERY nucleus rather than only where a caller remembers to
keep ``c/d < Z``. The two laws store disjoint parameter names on purpose, so a
checkpoint written by one cannot be reinterpreted by the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, electron_nuclear_displacements, pairwise_distances
from tpen.data.equivariant_state import JsonScalar, compare_tensor_blocks
from tpen.data.indices import permute_particle_axis
from tpen.data.permutation import Permutation
from tpen.dependencies import require_torch, require_torch_functional, require_torch_nn
from tpen.nn.envelope import Envelope
from tpen.nn.factor import LogAmplitudeFactor, _inverse_softplus

torch = require_torch(feature="TPEN cusp modules")
nn = require_torch_nn(feature="TPEN cusp modules")
F = require_torch_functional(feature="TPEN cusp modules")


@dataclass(frozen=True)
class ElectronNucleusCuspEvaluation:
    """Typed analytic radial data for one electron-nucleus cusp factor.

    All pair fields retain the explicit ``[batch, electron, nucleus]`` axes.
    ``slope_residual`` is the provider's already-cancelled value for
    ``(u'(r) + Z) / r``; consumers must not reconstruct it by subtraction.
    """

    displacement: torch.Tensor
    distance: torch.Tensor
    pair_value: torch.Tensor
    radial_first_derivative: torch.Tensor
    radial_second_derivative: torch.Tensor
    slope_residual: torch.Tensor
    nuclear_charges: torch.Tensor
    origin_radial_slope: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_electrons(self) -> int:
        """Return the explicit electron count from the displacement axis."""

        return int(self.displacement.shape[1])

    @property
    def n_nuclei(self) -> int:
        """Return the explicit nucleus count from the charge axis."""

        return int(self.nuclear_charges.shape[0])

    def validate(self) -> "ElectronNucleusCuspEvaluation":
        """Validate shape, dtype/device, finiteness, and axis contracts."""

        if not isinstance(self.displacement, torch.Tensor) or self.displacement.ndim != 4:
            raise ValueError("displacement must have shape [batch, electrons, nuclei, spatial_dim]")
        pair_fields = (
            self.distance,
            self.pair_value,
            self.radial_first_derivative,
            self.radial_second_derivative,
            self.slope_residual,
        )
        expected = self.displacement.shape[:3]
        if any(not isinstance(field, torch.Tensor) or field.shape != expected for field in pair_fields):
            raise ValueError("cusp pair fields must have shape [batch, electrons, nuclei]")
        if (
            not isinstance(self.nuclear_charges, torch.Tensor)
            or not isinstance(self.origin_radial_slope, torch.Tensor)
            or self.nuclear_charges.shape != (expected[2],)
            or self.origin_radial_slope.shape != (expected[2],)
        ):
            raise ValueError("nuclear charges and origin slopes must have shape [nuclei]")
        tensors = (self.displacement, *pair_fields, self.nuclear_charges, self.origin_radial_slope)
        if any(field.device != self.displacement.device or field.dtype != self.displacement.dtype for field in tensors):
            raise ValueError("all cusp evaluation tensors must share dtype and device")
        if any(not torch.isfinite(field).all() for field in tensors):
            raise ValueError("cusp evaluation tensors must be finite")
        return self

    def permute(self, permutation: Permutation) -> "ElectronNucleusCuspEvaluation":
        """Return the evaluation under an active electron permutation."""

        if len(permutation) != self.n_electrons:
            raise ValueError("electron permutation has incompatible size")
        return type(self)(
            displacement=permute_particle_axis(self.displacement, permutation, axis=1),
            distance=permute_particle_axis(self.distance, permutation, axis=1),
            pair_value=permute_particle_axis(self.pair_value, permutation, axis=1),
            radial_first_derivative=permute_particle_axis(self.radial_first_derivative, permutation, axis=1),
            radial_second_derivative=permute_particle_axis(self.radial_second_derivative, permutation, axis=1),
            slope_residual=permute_particle_axis(self.slope_residual, permutation, axis=1),
            nuclear_charges=self.nuclear_charges.clone(),
            origin_radial_slope=self.origin_radial_slope.clone(),
        )

    def compare(
        self,
        other: "ElectronNucleusCuspEvaluation",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, Mapping[str, JsonScalar]]:
        """Compare every semantic tensor field and return error metrics."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks(
            [self.displacement, self.distance, self.pair_value, self.radial_first_derivative,
             self.radial_second_derivative, self.slope_residual, self.nuclear_charges, self.origin_radial_slope],
            [other.displacement, other.distance, other.pair_value, other.radial_first_derivative,
             other.radial_second_derivative, other.slope_residual, other.nuclear_charges, other.origin_radial_slope],
            atol=atol,
            rtol=rtol,
        )

    def local_energy_pair(self) -> torch.Tensor:
        """Return the analytic kinetic-plus-Coulomb contribution per pair."""

        return -0.5 * self.radial_second_derivative - self.slope_residual


def rational_pair_cusp(
    distance: torch.Tensor,
    coefficient: torch.Tensor | float,
    range_parameter: torch.Tensor | float,
) -> torch.Tensor:
    """Return the rational pair cusp term.

    Parameters
    ----------
    distance : torch.Tensor
        Pair distances.
    coefficient : torch.Tensor or float
        Short-range cusp coefficient ``a``.
    range_parameter : torch.Tensor or float
        Positive range parameter ``b``.

    Returns
    -------
    torch.Tensor
        Values of ``a r / (1 + b r)`` with the same shape as `distance`.
    """

    output = coefficient * distance / (1.0 + range_parameter * distance)
    assert output.shape == distance.shape
    return output


class ElectronNucleusCuspLaw(ABC):
    """Typed radial law for a generic electron-nucleus cusp factor.

    A law computes the additive pairwise log-amplitude value for one
    electron-nucleus pair from raw (unclamped) distances and nuclear charges.
    Concrete laws are independently tested against the Kato cusp condition
    (the required ``d/dr`` slope at coalescence); this base class enforces no
    formula, only the typed value contract.

    Every law must fix its first-order Kato slope at ``r = 0`` by nuclear
    charge alone (``d/dr v_A(0) = -Z_A``, matching `LinearElectronNucleusCuspLaw`
    below). A law may separately expose an *optional* trainable regular radial
    component ``w_A(r)`` contributing only to second-order (and higher)
    curvature -- i.e. satisfying ``w_A(0) = 0`` and ``d/dr w_A(0) = 0`` -- so
    curvature can be learned without perturbing the enforced cusp condition.
    `CurvatureElectronNucleusCuspLaw` implements this ``w_A`` term.
    """

    @abstractmethod
    def value(self, distance: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """Return the pairwise cusp value with the same shape as `distance`.

        Parameters
        ----------
        distance : torch.Tensor
            Raw, unclamped electron-nucleus pair distances.
        charges : torch.Tensor
            Nuclear charges, broadcastable against `distance`.

        Returns
        -------
        torch.Tensor
            Cusp value with the same shape as `distance`.
        """

    def analytic_terms(
        self, distance: torch.Tensor, charges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return value, first derivative, second derivative, and residual."""

        raise NotImplementedError(f"{type(self).__name__} has no analytic cusp capability")

    def origin_radial_slope(self, charges: torch.Tensor) -> torch.Tensor:
        """Return the independently specified coalescence radial slope."""

        raise NotImplementedError(f"{type(self).__name__} has no analytic cusp capability")


class LinearElectronNucleusCuspLaw(ElectronNucleusCuspLaw):
    """Compatibility law reproducing the existing He linear cusp ``-Z r``.

    This is the He linear cusp formerly hard-coded by the retired
    ``NuclearConfinement`` envelope, preserved here as a
    `ElectronNucleusCuspLaw` so existing systems have a like-for-like generic
    counterpart.

    WHY THIS LAW STILL EXISTS, since it is mathematically redundant. It is
    identical to `CurvatureElectronNucleusCuspLaw(curvature_coefficient=0.0,
    trainable=False)`, because ``w_A(r) = 0 * r^2 / (1 + d r)`` vanishes. Do
    NOT collapse the two on that basis: this law has NO PARAMETERS AT ALL, so
    it cannot express the ``c = 0`` degeneracy that the curved law's own
    defaults (``curvature_coefficient=0.0, trainable=True``) land in, where
    ``d w_A / d d`` is proportional to ``c`` and a trainable range therefore
    receives an identically zero gradient and can never move. A parameter-free
    law makes "I want fixed ``-Z r``" unable to be said wrongly; the curved law
    relies on `tests/unit/experiments/test_he_v1_config.py` asserting
    ``c != 0`` to catch the same mistake after the fact. Trap-proof by
    construction is not the same as trap-caught by assertion.
    """

    def value(self, distance: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """Return ``-Z r`` broadcast to the shape of `distance`."""

        return -charges * distance

    def analytic_terms(self, distance: torch.Tensor, charges: torch.Tensor):
        value = self.value(distance, charges)
        zero = torch.zeros_like(distance)
        return value, -torch.broadcast_to(charges, distance.shape), zero, zero

    def origin_radial_slope(self, charges: torch.Tensor) -> torch.Tensor:
        """Return the charge-fixed slope without consulting ``value``."""

        return -charges


class CurvatureElectronNucleusCuspLaw(nn.Module, ElectronNucleusCuspLaw):
    """Linear Kato slope plus an optional trainable second-order curvature term.

    NAME STATES THE FUNCTIONAL FORM, NOT THE TRAINABILITY. This class was
    called ``TrainableCurvatureElectronNucleusCuspLaw``, which named the wrong
    axis: what distinguishes it from `LinearElectronNucleusCuspLaw` is the
    CURVATURE TERM, while trainability is the internal ``trainable`` flag below
    and is available in both states. ``trainable=False`` gives the curved law
    with FROZEN curvature -- a third configuration the old name made sound
    contradictory.

    Computes ``v_A(r) = -Z_A r + w_A(r)`` with ``w_A(r) = c r^2 / (1 + d r)``,
    where ``d`` is the strictly positive, optionally trainable RANGE parameter
    of the electron-nucleus cusp -- the counterpart of `ElectronElectronCusp`'s
    ``range_parameter`` -- and ``c`` is an unconstrained-sign curvature
    coefficient. ``w_A`` satisfies ``w_A(0) = 0`` and ``d/dr w_A(0) = 0`` for
    any ``c`` and positive ``d``: the curvature term contributes only from
    second order, so the charge-fixed Kato slope ``d/dr v_A(0) = -Z_A`` stays
    exact regardless of ``c`` or ``d``, before and after any optimizer step.

    Functional-form decision, recorded here because it moves outer-tail
    diagnostics by construction. Kato constrains only the ``r -> 0`` slope, so
    a saturating Pade law ``-Z r / (1 + a r)`` satisfies it too; but that law
    tends to the constant ``-Z / a``, i.e. ``log |psi|`` stops decreasing and
    the factor stops confining. This law is deliberately NON-saturating:
    ``w_A(r) -> (c / d) r`` at large ``r``, so ``v_A`` keeps growing linearly
    and its outer-tail slope is ``-Z_A + c / d`` (see `outer_tail_slope`). Two
    consequences bind consumers:

    - At ``c = 0`` the law is exactly ``-Z_A r`` and the outer-tail slope is
      exactly ``-Z_A``. At ``c != 0`` the tail slope is SHIFTED by ``c / d``,
      so outer-tail tolerances calibrated against the pure ``-Z r`` law
      must not be applied unchanged to a trained curvature law.
    - A decaying (normalizable) tail needs a negative slope, i.e.
      ``c / d < Z_A``. **THIS CLASS DOES NOT ENFORCE THAT BOUND, so its tail is
      unbounded and training can walk across the sign change into a growing,
      non-normalizable tail with nothing raising.** A caller that trains ``c``
      and ``d`` owns choosing an initialization, and if needed a constraint,
      that keeps ``c / d`` well below the nuclear charge.

      If you do not specifically need this law's unconstrained sign, prefer
      `TailSafeElectronNucleusCuspLaw`, which coordinates the same functional
      form as ``c = d (Z - kappa)`` so the outer slope is ``-kappa < 0`` for
      every nucleus by construction. The bound there is structural rather than
      a caller's responsibility. Stated here, at the point of use, because a
      reader who lands on this class should learn its limit without opening
      anything else.

    Gradient reachability, which matters when enabling training from a config:
    ``d`` enters the value only through ``w_A``, so at exactly ``c = 0`` the
    gradient with respect to ``d`` is identically zero. ``c`` itself still
    receives gradient there and moves off zero, which unlocks ``d`` from the
    next step; a configuration that wants the range trained from the first step
    should initialize `curvature_coefficient` nonzero.

    Parameters
    ----------
    curvature_coefficient : float, optional
        Initial value of the (unconstrained-sign) curvature coefficient ``c``.
    curvature_range : float, optional
        Positive initial value of the range parameter ``d``.
    trainable : bool, optional
        Whether ``c`` and ``d`` are optimized. When ``False`` they are fixed
        non-persistent buffers at the constructor values, so the module
        contributes no checkpoint state; when ``True`` it contributes
        ``raw_curvature_coefficient`` and ``raw_curvature_range`` to the state
        dict, which a ``strict=True`` restore will not accept against a
        differently configured law.
    eps : float, optional
        Positivity offset for the trainable range parameter.
    """

    def __init__(
        self,
        curvature_coefficient: float = 0.0,
        curvature_range: float = 1.0,
        trainable: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if curvature_range <= 0.0:
            raise ValueError(f"curvature_range must be positive, got {curvature_range}")
        self.trainable = bool(trainable)
        self.eps = eps
        if self.trainable:
            self.raw_curvature_coefficient = nn.Parameter(
                torch.tensor(float(curvature_coefficient), dtype=torch.float64)
            )
            self.raw_curvature_range = nn.Parameter(_inverse_softplus(float(curvature_range) - eps))
        else:
            self.register_buffer(
                "_curvature_coefficient",
                torch.tensor(float(curvature_coefficient), dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_curvature_range",
                torch.tensor(float(curvature_range), dtype=torch.float64),
                persistent=False,
            )

    @property
    def curvature_coefficient(self) -> torch.Tensor:
        """Return the (unconstrained-sign) curvature coefficient ``c``."""

        if self.trainable:
            return self.raw_curvature_coefficient
        return self._curvature_coefficient

    @property
    def curvature_range(self) -> torch.Tensor:
        """Return the positive curvature range parameter ``d``."""

        if self.trainable:
            return F.softplus(self.raw_curvature_range) + self.eps
        return self._curvature_range

    def outer_tail_slope(self, charges: torch.Tensor | float) -> torch.Tensor:
        """Return the large-``r`` limit of ``d/dr v_A``, namely ``-Z + c / d``.

        This makes the recorded outer-tail consequence of the chosen functional
        form executable: a tail diagnostic calibrated against the pure ``-Z r``
        law expects ``-Z``, while this law asymptotes to ``-Z + c / d``.

        Parameters
        ----------
        charges : torch.Tensor or float
            Nuclear charges ``Z``.

        Returns
        -------
        torch.Tensor
            Asymptotic radial slope with the shape of `charges`.
        """

        charge_tensor = torch.as_tensor(charges)
        if not torch.is_floating_point(charge_tensor):
            charge_tensor = charge_tensor.to(dtype=torch.get_default_dtype())
        coefficient = self.curvature_coefficient.to(device=charge_tensor.device, dtype=charge_tensor.dtype)
        range_parameter = self.curvature_range.to(device=charge_tensor.device, dtype=charge_tensor.dtype)
        return -charge_tensor + coefficient / range_parameter

    def value(self, distance: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """Return ``-Z r + c r^2 / (1 + d r)`` broadcast to the shape of `distance`."""

        linear = -charges * distance
        coefficient = self.curvature_coefficient.to(device=distance.device, dtype=distance.dtype)
        range_parameter = self.curvature_range.to(device=distance.device, dtype=distance.dtype)
        curvature = coefficient * distance.square() / (1.0 + range_parameter * distance)
        return linear + curvature

    def analytic_terms(self, distance: torch.Tensor, charges: torch.Tensor):
        """Return closed-form radial derivatives and the cancelled residual."""

        coefficient = self.curvature_coefficient.to(device=distance.device, dtype=distance.dtype)
        range_parameter = self.curvature_range.to(device=distance.device, dtype=distance.dtype)
        denominator = 1.0 + range_parameter * distance
        value = self.value(distance, charges)
        first = -charges + coefficient * distance * (2.0 + range_parameter * distance) / denominator.square()
        second = 2.0 * coefficient / denominator.pow(3)
        # This is deliberately algebraically cancelled; never form (first + Z) / r.
        residual = coefficient * (2.0 + range_parameter * distance) / denominator.square()
        return value, first, second, residual

    def origin_radial_slope(self, charges: torch.Tensor) -> torch.Tensor:
        """Return the charge-fixed slope independently of the value expression."""

        return -charges

    def scalar_diagnostics(self) -> dict[str, float]:
        """Return ``c`` and ``d`` as constrained values, with the raws beside them.

        ``d`` is stored as ``raw_curvature_range`` behind a softplus, so the raw
        and constrained axes are different: a raw value can crawl while the
        effective range has settled, or sit nearly still while the constrained
        value moves. Both are reported so a convergence assessment reads the
        physical parameter and can still see whether the optimizer is touching
        the stored one at all -- which is the observable that distinguishes a
        genuinely converged range from the ``c = 0`` zero-gradient trap.
        """

        scalars = {
            "curvature_coefficient": float(self.curvature_coefficient.item()),
            "curvature_range": float(self.curvature_range.item()),
        }
        if self.trainable:
            scalars["raw_curvature_coefficient"] = float(self.raw_curvature_coefficient.item())
            scalars["raw_curvature_range"] = float(self.raw_curvature_range.item())
        return scalars


class TailSafeElectronNucleusCuspLaw(nn.Module, ElectronNucleusCuspLaw):
    r"""Curved Kato law whose outer tail is guaranteed decaying, for every nucleus.

    NAME STATES THE PROPERTY, NOT THE VERSION. What distinguishes this from
    `CurvatureElectronNucleusCuspLaw` is not that it is newer -- it is that its
    asymptotic slope cannot come out non-decaying, whatever the optimizer does
    and whatever the nuclear charge is. Choose between the two on that fact
    alone; neither name needs a changelog to disambiguate.

    Same functional form as `CurvatureElectronNucleusCuspLaw`,
    ``v_A(r) = -Z_A r + c_A r^2 / (1 + d r)``, but coordinated differently:

    .. math::

        d      &= \varepsilon + \mathrm{softplus}(\tilde{d}) \\
        \kappa &= m + \mathrm{softplus}(\tilde{\kappa}) \\
        c_A    &= d \, (Z_A - \kappa)

    with ``eps = 1e-12`` and margin ``m = 0.1`` by default. Both ``d`` and
    ``kappa`` are trainable; ``c`` is DERIVED and is never stored.

    **Why the tail is safe.** The old law's outer slope is ``-Z_A + c/d``,
    which is negative only while ``c/d < Z_A`` -- a bound that law records but
    does not enforce, so training can walk across it and produce a
    non-normalizable, growing tail. Substituting ``c_A = d (Z_A - kappa)``
    gives ``c_A / d = Z_A - kappa`` and therefore

        outer slope = ``-Z_A + (Z_A - kappa)`` = ``-kappa``

    identically. Since ``kappa >= m > 0`` by construction, the slope is
    strictly negative for EVERY nucleus regardless of its charge, at every
    point in training. The bound is structural, not a constraint a caller has
    to remember.

    **Why ``c`` is derived per charge rather than stored.** ``kappa = Z - c/d``
    depends on the charge, so a single stored ``c`` can only be tail-safe for
    one ``Z``. The consolidated authority's expression for helium is
    ``c = d (2 - kappa)``, where the ``2`` is the NUCLEAR CHARGE and not a
    literal -- see ``theory-report-2026-08-31``, "Let kappa = Z - c/d".
    Hardcoding it would ship a helium-only law that silently mis-scales on any
    other atom. `curvature_coefficient` therefore takes ``charges``, and every
    caller that already receives charges gets the right ``c`` for free.

    **The margin ``m`` is a design choice, not a measurement.** It is a
    conservative positive floor keeping ``kappa`` away from zero, chosen by the
    designer. It is NOT a learned physical decay constant and must not be
    reported as one; the learned quantity is ``kappa`` itself, which is free
    above the floor.

    **Identifiability of the range parameter.** Differentiating the value with
    respect to ``d`` gives, after the two terms combine,

    .. math::

        \frac{\partial v_A}{\partial d}
            = (Z_A - \kappa) \frac{r^2}{(1 + d r)^2}
            = \frac{c_A}{d} \frac{r^2}{(1 + d r)^2},

    since ``c_A = d (Z_A - kappa)`` by definition. Every path by which ``d``
    reaches the value carries that factor, so **the gradient on the range
    parameter is proportional to** ``c/d``.

    This is an IDENTIFIABILITY LIMIT, not a defect and not a property of any
    particular ``kappa``. As ``c -> 0`` the term ``c r^2 / (1 + d r)`` vanishes
    and ``d`` is shaping something that is not there, so a small curvature
    coefficient and a well-conditioned range cannot be had at the same time --
    in this parametrization or in the ``(c, d)`` one, which is the same
    function. Anything that "fixes" the conditioning by moving ``kappa`` has
    moved ``c``, because they are the same fact.

    Concretely, the he-importance control level sits at ``c/d = 0.01``, so
    ``d``'s gradient there is a hundredth of its value at ``c/d = 1``. That
    factor is exact. What it implies for LEARNABILITY is a separate and
    unmeasured question: `torch.optim.Adam` divides by ``sqrt(v)`` and is
    scale-invariant per parameter in the deterministic limit, so a uniformly
    smaller gradient does not give a proportionally smaller step. The real
    exposure is signal-to-noise -- a suppressed signal against unchanged
    sampling noise makes ``d`` random-walk rather than descend -- and no
    measurement here establishes where that boundary lies.

    **Migration is refused, structurally.** This class stores ``raw_range`` and
    ``raw_kappa``. `CurvatureElectronNucleusCuspLaw` stores
    ``raw_curvature_coefficient`` and ``raw_curvature_range``. The key sets are
    DISJOINT and the class names differ, so an old checkpoint cannot be
    reinterpreted in the new coordinates by accident -- there is no version tag
    for a reader to overlook. `_load_from_state_dict` additionally raises a
    named error when it sees the old keys, so the failure explains the
    migration instead of surfacing as an opaque unexpected-key list.

    That disjointness is what makes the refusal reliable. The old law
    contributes NOTHING to the state dict when ``trainable=False`` (its
    coordinates are non-persistent buffers), so absence of keys is not evidence
    of format, and any scheme that tried to tell the two apart by inspecting
    stored bytes alone would be undecidable for the frozen case.

    Parameters
    ----------
    curvature_range : float, optional
        Initial ``d``. Must be positive.
    tail_slope : float, optional
        Initial ``kappa``, the magnitude of the outer radial slope. Must exceed
        `tail_slope_margin`. The resulting outer slope is ``-tail_slope``.
    trainable : bool, optional
        Whether ``d`` and ``kappa`` are optimized. When ``False`` they are
        fixed non-persistent buffers and this module contributes no checkpoint
        state.
    range_eps : float, optional
        Positivity offset for ``d``.
    tail_slope_margin : float, optional
        The floor ``m`` below which ``kappa`` cannot fall. Positive.

    See Also
    --------
    CurvatureElectronNucleusCuspLaw : the unconstrained-sign predecessor, whose
        tail is NOT bounded.
    """

    def __init__(
        self,
        curvature_range: float = 1.0,
        tail_slope: float = 1.0,
        trainable: bool = True,
        range_eps: float = 1e-12,
        tail_slope_margin: float = 0.1,
    ) -> None:
        super().__init__()
        if curvature_range <= 0.0:
            raise ValueError(f"curvature_range must be positive, got {curvature_range}")
        if tail_slope_margin <= 0.0:
            raise ValueError(f"tail_slope_margin must be positive, got {tail_slope_margin}")
        if tail_slope <= tail_slope_margin:
            raise ValueError(
                f"tail_slope must exceed tail_slope_margin, got {tail_slope} <= {tail_slope_margin}. "
                "The margin is the floor the parametrization enforces, so a requested value at or "
                "below it is not representable and is refused rather than silently clamped"
            )
        self.trainable = bool(trainable)
        self.range_eps = float(range_eps)
        self.tail_slope_margin = float(tail_slope_margin)
        if self.trainable:
            self.raw_range = nn.Parameter(_inverse_softplus(float(curvature_range) - self.range_eps))
            self.raw_kappa = nn.Parameter(_inverse_softplus(float(tail_slope) - self.tail_slope_margin))
        else:
            self.register_buffer(
                "_range",
                torch.tensor(float(curvature_range), dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_kappa",
                torch.tensor(float(tail_slope), dtype=torch.float64),
                persistent=False,
            )

    @classmethod
    def from_curvature_coefficient(
        cls,
        curvature_coefficient: float,
        curvature_range: float,
        charge: float,
        **kwargs: object,
    ) -> "TailSafeElectronNucleusCuspLaw":
        """Build the law by INVERTING a requested ``(c, d)`` at a given charge.

        The migration path for a caller that thinks in the old coordinates.
        ``kappa = Z - c / d`` is the exact inverse of the forward map, so the
        constructed law reproduces the requested ``c`` at that charge.

        Parameters
        ----------
        curvature_coefficient : float
            The requested ``c``.
        curvature_range : float
            The requested ``d``. Must be positive.
        charge : float
            The nuclear charge ``Z`` at which ``c`` was specified. REQUIRED,
            because ``c`` alone does not determine ``kappa``.
        **kwargs : object
            Forwarded to ``__init__`` (``trainable``, ``range_eps``,
            ``tail_slope_margin``).

        Returns
        -------
        TailSafeElectronNucleusCuspLaw

        Raises
        ------
        ValueError
            When the request is not representable, i.e. ``Z - c/d`` does not
            exceed the margin. That is exactly the region the old law permitted
            and this one forbids -- a tail that decays too slowly or not at
            all. It RAISES rather than clamping, because silently returning a
            different law than the one asked for is how a mis-migrated
            checkpoint would go unnoticed.
        """

        if curvature_range <= 0.0:
            raise ValueError(f"curvature_range must be positive, got {curvature_range}")
        tail_slope = float(charge) - float(curvature_coefficient) / float(curvature_range)
        margin = float(kwargs.get("tail_slope_margin", 0.1))
        if tail_slope <= margin:
            raise ValueError(
                f"c={curvature_coefficient} and d={curvature_range} at Z={charge} give "
                f"kappa = Z - c/d = {tail_slope}, which does not exceed the margin {margin}. "
                "That request is not tail-safe: the old law permitted it, this law refuses it. "
                "Choose a smaller c/d ratio rather than lowering the margin"
            )
        return cls(curvature_range=curvature_range, tail_slope=tail_slope, **kwargs)  # type: ignore[arg-type]

    @property
    def curvature_range(self) -> torch.Tensor:
        """Return the positive range parameter ``d``."""

        if self.trainable:
            return F.softplus(self.raw_range) + self.range_eps
        return self._range

    @property
    def tail_slope_magnitude(self) -> torch.Tensor:
        """Return ``kappa``, the magnitude of the outer radial slope."""

        if self.trainable:
            return F.softplus(self.raw_kappa) + self.tail_slope_margin
        return self._kappa

    def curvature_coefficient(self, charges: torch.Tensor | float) -> torch.Tensor:
        """Return ``c_A = d (Z_A - kappa)``, one value per charge.

        Unlike `CurvatureElectronNucleusCuspLaw.curvature_coefficient`, this is
        a METHOD taking charges rather than a stored property, because ``c``
        depends on the charge. That difference is deliberate and is what stops
        the law being helium-only.

        Parameters
        ----------
        charges : torch.Tensor or float
            Nuclear charges ``Z``.

        Returns
        -------
        torch.Tensor
            ``c`` with the shape of `charges`.
        """

        charge_tensor = torch.as_tensor(charges)
        if not torch.is_floating_point(charge_tensor):
            charge_tensor = charge_tensor.to(dtype=torch.get_default_dtype())
        range_parameter = self.curvature_range.to(
            device=charge_tensor.device, dtype=charge_tensor.dtype
        )
        kappa = self.tail_slope_magnitude.to(
            device=charge_tensor.device, dtype=charge_tensor.dtype
        )
        return range_parameter * (charge_tensor - kappa)

    def outer_tail_slope(self, charges: torch.Tensor | float) -> torch.Tensor:
        """Return the large-``r`` limit of ``d/dr v_A``, which is ``-kappa``.

        Charge-INDEPENDENT by construction, which is the whole point: the
        predecessor's ``-Z + c/d`` varies with the nucleus and can change sign,
        while this is ``-kappa < 0`` for every nucleus in the system.

        Computed through `curvature_coefficient` rather than by returning
        ``-kappa`` directly, so the identity ``-Z + c/d == -kappa`` is
        EXERCISED on the real code path instead of asserted in a docstring. A
        future edit that broke the relation would show up here rather than
        remaining hidden behind a shortcut.

        Parameters
        ----------
        charges : torch.Tensor or float
            Nuclear charges ``Z``.

        Returns
        -------
        torch.Tensor
            Asymptotic radial slope with the shape of `charges`.
        """

        charge_tensor = torch.as_tensor(charges)
        if not torch.is_floating_point(charge_tensor):
            charge_tensor = charge_tensor.to(dtype=torch.get_default_dtype())
        coefficient = self.curvature_coefficient(charge_tensor)
        range_parameter = self.curvature_range.to(
            device=charge_tensor.device, dtype=charge_tensor.dtype
        )
        return -charge_tensor + coefficient / range_parameter

    def value(self, distance: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """Return ``-Z r + c(Z) r^2 / (1 + d r)`` broadcast to `distance`."""

        linear = -charges * distance
        coefficient = self.curvature_coefficient(charges).to(
            device=distance.device, dtype=distance.dtype
        )
        range_parameter = self.curvature_range.to(device=distance.device, dtype=distance.dtype)
        curvature = coefficient * distance.square() / (1.0 + range_parameter * distance)
        return linear + curvature

    def analytic_terms(self, distance: torch.Tensor, charges: torch.Tensor):
        """Return closed-form radial derivatives and the cancelled residual."""

        coefficient = self.curvature_coefficient(charges).to(
            device=distance.device, dtype=distance.dtype
        )
        range_parameter = self.curvature_range.to(device=distance.device, dtype=distance.dtype)
        denominator = 1.0 + range_parameter * distance
        value = self.value(distance, charges)
        first = -charges + coefficient * distance * (2.0 + range_parameter * distance) / denominator.square()
        second = 2.0 * coefficient / denominator.pow(3)
        # This is deliberately algebraically cancelled; never form (first + Z) / r.
        residual = coefficient * (2.0 + range_parameter * distance) / denominator.square()
        return value, first, second, residual

    def origin_radial_slope(self, charges: torch.Tensor) -> torch.Tensor:
        """Return the charge-fixed slope independently of the value expression.

        Unchanged from the predecessor: the curvature term still contributes
        only from second order, so ``d/dr v_A(0) = -Z_A`` exactly, for any
        ``d`` and ``kappa``.
        """

        return -charges

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Refuse an old-coordinate checkpoint by name rather than by symptom.

        Raises unconditionally when legacy keys are present, INCLUDING under
        ``strict=False``, where the default behaviour would be to ignore them
        silently and leave this law at its freshly initialized values. A
        wavefunction quietly restored to the wrong parameters is precisely the
        ambiguous migration this slice exists to prevent, and it would produce
        plausible numbers rather than an error.
        """

        legacy = [
            name
            for name in ("raw_curvature_coefficient", "raw_curvature_range")
            if prefix + name in state_dict
        ]
        if legacy:
            raise RuntimeError(
                f"{type(self).__name__} at prefix {prefix!r} was given "
                f"{sorted(legacy)}, which belong to CurvatureElectronNucleusCuspLaw. "
                "The two laws use different coordinates: this one stores 'raw_range' and "
                "'raw_kappa' and derives c = d * (Z - kappa) per charge, so the old raw "
                "values have no meaning here and are refused rather than reinterpreted. "
                "To carry an old checkpoint forward, rebuild the law with "
                "TailSafeElectronNucleusCuspLaw.from_curvature_coefficient(c, d, charge), "
                "which inverts the request exactly and raises if it was never tail-safe"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def scalar_diagnostics(self) -> dict[str, float]:
        """Return the constrained coordinates with their raws beside them.

        ``c`` is deliberately ABSENT: it is charge-dependent, so there is no
        single value to report here, and emitting one would reintroduce the
        helium-only reading this parametrization removes. Read it through
        `curvature_coefficient` with the charges in hand.
        """

        scalars = {
            "curvature_range": float(self.curvature_range.item()),
            "tail_slope_magnitude": float(self.tail_slope_magnitude.item()),
        }
        if self.trainable:
            scalars["raw_range"] = float(self.raw_range.item())
            scalars["raw_kappa"] = float(self.raw_kappa.item())
        return scalars


class ElectronNucleusCusp(LogAmplitudeFactor):
    """Generic electron-nucleus Kato cusp factor for arbitrary nuclei.

    This factor is constructed directly from an `AtomicConfiguration` -- the
    sole authority for nuclear geometry -- and never infers nuclear context
    from a batch. It uses raw, unclamped pair distances so the cusp condition
    is observed exactly at coalescence.

    Parameters
    ----------
    atoms : AtomicConfiguration
        Constructor-owned fixed nuclear geometry authority.
    law : ElectronNucleusCuspLaw or None, optional
        Cusp radial law. Defaults to `LinearElectronNucleusCuspLaw`, the He
        linear-cusp compatibility law.
    """

    def __init__(self, atoms: object, law: object = None) -> None:
        super().__init__()
        if not isinstance(atoms, AtomicConfiguration):
            raise TypeError(f"{type(self).__name__} requires an AtomicConfiguration, got {type(atoms).__name__}")
        self.atoms = atoms
        resolved_law = law if law is not None else LinearElectronNucleusCuspLaw()
        if not isinstance(resolved_law, ElectronNucleusCuspLaw):
            raise TypeError(
                f"{type(self).__name__} law must be an ElectronNucleusCuspLaw, got {type(resolved_law).__name__}"
            )
        self.law = resolved_law

    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the summed electron-nucleus cusp contribution."""

        atoms = self.atoms.to(device=batch.device, dtype=batch.dtype)
        distance = electron_nuclear_displacements(batch, nuclear_positions=atoms.positions).norm(dim=-1)
        charges = atoms.charges.reshape(1, 1, -1)
        value = self.law.value(distance, charges)
        if value.shape != distance.shape:
            raise ValueError(
                f"{type(self.law).__name__}.value must have shape {tuple(distance.shape)}, got {tuple(value.shape)}"
            )
        return value.sum(dim=(1, 2))

    def analytic_evaluation(self, batch: ElectronBatch) -> ElectronNucleusCuspEvaluation:
        """Return opt-in closed-form radial data for local-energy consumers.

        Ordinary :meth:`forward` and :meth:`factor_value` never call this
        capability, keeping the value-only path free of derivative work.
        """

        flat_batch = batch.flatten_samples()
        atoms = self.atoms.to(device=flat_batch.device, dtype=flat_batch.dtype)
        displacement = electron_nuclear_displacements(flat_batch, nuclear_positions=atoms.positions)
        distance = displacement.norm(dim=-1)
        charges = atoms.charges.reshape(1, 1, -1)
        pair_value, first, second, residual = self.law.analytic_terms(distance, charges)
        origin_slope = self.law.origin_radial_slope(atoms.charges)
        evaluation = ElectronNucleusCuspEvaluation(
            displacement=displacement,
            distance=distance,
            pair_value=pair_value,
            radial_first_derivative=first,
            radial_second_derivative=second,
            slope_residual=residual,
            nuclear_charges=atoms.charges,
            origin_radial_slope=origin_slope,
        )
        return evaluation

    def scalar_diagnostics(self) -> dict[str, float]:
        """Return the law's trainable scalars, and the tail slope they imply.

        The factor owns no scalars of its own -- the radial law does -- so this
        delegates and prefixes. `outer_tail_slope` is included when the law
        exposes it because the asymptotic slope is the quantity a tail gate is
        centered on, and recomputing it downstream from ``c`` and ``d`` would
        duplicate the law's own definition.
        """

        law_scalars = getattr(self.law, "scalar_diagnostics", None)
        if not callable(law_scalars):
            return {}
        scalars = {f"law.{name}": value for name, value in law_scalars().items()}
        tail_slope = getattr(self.law, "outer_tail_slope", None)
        if callable(tail_slope):
            charges = self.atoms.charges.reshape(-1)
            slopes = tail_slope(charges).reshape(-1)
            for index, slope in enumerate(slopes.tolist()):
                scalars[f"law.outer_tail_slope.{index}"] = float(slope)
        return scalars


class ElectronElectronCusp(Envelope, LogAmplitudeFactor):
    """Spin-aware analytic electron-electron cusp envelope.

    Parameters
    ----------
    enabled : bool, optional
        Whether this envelope contributes to the output.
    same_spin_coefficient : float, optional
        Short-range coefficient for equal-spin electron pairs.
    opposite_spin_coefficient : float, optional
        Short-range coefficient for opposite-spin electron pairs.
    spinless_coefficient : float or None, optional
        Coefficient used when `ElectronBatch.spins` is absent. If ``None``,
        `same_spin_coefficient` is used.
    range_parameter : float, optional
        Default positive range parameter.
    same_range_parameter : float or None, optional
        Equal-spin range parameter. If ``None``, `range_parameter` is used.
    opposite_range_parameter : float or None, optional
        Opposite-spin range parameter. If ``None``, `range_parameter` is used.
    trainable_range : bool, optional
        Whether to optimize the range parameters through a softplus
        parametrization.
    eps : float, optional
        Distance floor retained for backwards compatibility. The default is
        the analytic unfloored distance (``0.0``). A non-zero value is
        unproven and may be inconsistent: the cusp factor and Coulomb
        potential do not necessarily apply the same floor, yielding a hybrid
        Hamiltonian: a clipped potential evaluated with the boundary condition
        of an unclipped Coulomb potential. Inspect and validate before using a
        non-zero value. The
        finite-eps electron-electron case is UNMEASURED.
    range_eps : float, optional
        Independent positive offset for the softplus range parametrization.
        Defaults effectively to ``1e-12`` so range parameters remain strictly
        positive when ``eps=0.0``. For backwards compatibility, an old-style
        call that supplies a non-zero ``eps`` and omits ``range_eps`` uses
        that value for both historical roles; pass ``range_eps`` explicitly to
        opt out of that coupling.

    Warning
    -------
    The electron-nucleus measurement found ``E(r; eps) = Z/r - Z/eps -
    Z^2/2`` for ``0 < r < eps`` across three eps scales and three directions,
    with normalized error <= ``1.11e-16``. The electron-electron finite-eps
    case is UNMEASURED; identical clamps might yield a constant offset rather
    than a divergence, but this has not been tested and must not be assumed
    benign.
    """

    def __init__(
        self,
        enabled: bool = True,
        same_spin_coefficient: float = 0.25,
        opposite_spin_coefficient: float = 0.5,
        spinless_coefficient: float | None = None,
        range_parameter: float = 1.0,
        same_range_parameter: float | None = None,
        opposite_range_parameter: float | None = None,
        trainable_range: bool = False,
        eps: float = 0.0,
        range_eps: float | None = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.same_spin_coefficient = float(same_spin_coefficient)
        self.opposite_spin_coefficient = float(opposite_spin_coefficient)
        if spinless_coefficient is None:
            spinless_coefficient = same_spin_coefficient
        self.spinless_coefficient = float(spinless_coefficient)
        self.trainable_range = bool(trainable_range)
        self.eps = float(eps)
        if range_eps is None:
            # Preserve the old single-eps behavior for explicit non-zero
            # calls, while the new default keeps a fixed positivity offset.
            self.range_eps = self.eps if self.eps != 0.0 else 1.0e-12
        else:
            self.range_eps = float(range_eps)
        same_range = range_parameter if same_range_parameter is None else same_range_parameter
        opposite_range = range_parameter if opposite_range_parameter is None else opposite_range_parameter
        if self.trainable_range:
            self.raw_same_range = nn.Parameter(_inverse_softplus(float(same_range) - self.range_eps))
            self.raw_opposite_range = nn.Parameter(_inverse_softplus(float(opposite_range) - self.range_eps))
        else:
            self.register_buffer("same_range", torch.tensor(float(same_range)), persistent=False)
            self.register_buffer("opposite_range", torch.tensor(float(opposite_range)), persistent=False)

    @property
    def same_range_parameter(self) -> torch.Tensor:
        """Return the positive same-spin range parameter."""

        if self.trainable_range:
            return F.softplus(self.raw_same_range) + self.range_eps
        return self.same_range

    @property
    def opposite_range_parameter(self) -> torch.Tensor:
        """Return the positive opposite-spin range parameter."""

        if self.trainable_range:
            return F.softplus(self.raw_opposite_range) + self.range_eps
        return self.opposite_range

    def scalar_diagnostics(self) -> dict[str, float]:
        """Return both range parameters as constrained values, raws beside them.

        Same reasoning as the electron-nucleus law: the softplus makes the raw
        and effective axes different, so a convergence assessment that read only
        the raws could mistake a settled range for a moving one.
        """

        scalars = {
            "same_range_parameter": float(self.same_range_parameter.item()),
            "opposite_range_parameter": float(self.opposite_range_parameter.item()),
        }
        if self.trainable_range:
            scalars["raw_same_range"] = float(self.raw_same_range.item())
            scalars["raw_opposite_range"] = float(self.raw_opposite_range.item())
        return scalars

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the electron-electron cusp contribution."""

        distances = pairwise_distances(batch.positions, eps=self.eps).squeeze(-1)
        assert distances.shape == (batch.batch_size, batch.n_electrons, batch.n_electrons)
        tri = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        if batch.spins is None:
            contribution = rational_pair_cusp(distances, self.spinless_coefficient, self.same_range_parameter)
        else:
            spins = batch.spins.to(device=batch.device, dtype=batch.dtype)
            same_spin = spins.unsqueeze(2) == spins.unsqueeze(1)
            coefficients = torch.where(
                same_spin,
                torch.full_like(distances, self.same_spin_coefficient),
                torch.full_like(distances, self.opposite_spin_coefficient),
            )
            ranges = torch.where(
                same_spin,
                self.same_range_parameter,
                self.opposite_range_parameter,
            )
            contribution = rational_pair_cusp(distances, coefficients, ranges)
        output = contribution.masked_fill(~tri, 0.0).sum(dim=(1, 2))
        assert output.shape == (batch.batch_size,)
        return output


__all__ = [
    "ElectronElectronCusp",
    "ElectronNucleusCusp",
    "ElectronNucleusCuspEvaluation",
    "ElectronNucleusCuspLaw",
    "LinearElectronNucleusCuspLaw",
    "CurvatureElectronNucleusCuspLaw",
    "TailSafeElectronNucleusCuspLaw",
    "rational_pair_cusp",
]
