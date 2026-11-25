"""Native pure-PyTorch structured KFAC update method and HI grid mapping.

This module owns :class:`NativeKFACUpdate`, its policy/telemetry records, and
the strict helium-importance authority grid mapping.  It consumes detached
:class:`~tpen.training.step_requests.StructuredCurvatureUpdateInput` payloads
produced by :class:`~tpen.training.step_requests.StructuredCurvatureStepRequest`
without adding any external KFAC dependency.

Mathematical conventions frozen in TO ``9daaf9ef`` §4
-----------------------------------------------------
- ``KroneckerFactorBlock``: uncentered KFAC-expand second moments ``(A, G)``
  preserving the homogeneous bias column ``1`` appended last.
- ``DenseFactorBlock``: centered empirical score covariance ``F`` matching
  ``S = O^T O`` so global-scale gauge modes carry a zero eigenvalue.
- Energy half-gradient (``scale = 1.0``): ascent-gradient sign
  ``g_half = + (1 / N_fin) sum_b (E_b - E_bar) grad log|psi(R_b)|``; writing
  ``parameter.grad = nu * d`` into plain ``torch.optim.SGD(lr=eta)`` applies
  ``theta <- theta - eta * nu * d`` with ``d = (F_cached + lambda I)^{-1} g_half``.
- Trace-normalized relative damping ``lambda_b = relative_damping * max(tau_b, 1e-12)``
  and exact eigen-basis Kronecker inversion (rejecting factored Tikhonov).
- Step-0 warm EMA initialization, update-before-refresh cadence, exact
  predicted-KL cap against current EMA factors, and binary ``method_state.pt``
  tensor persistence for bit-exact mid-refresh resume parity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Self

from tpen.dependencies import require_torch
from tpen.training.step_requests import (
    StructuredCurvatureStepRequest,
    StructuredCurvatureUpdateInput,
)
from tpen.training.update import (
    ModelParameterBinding,
    ScopeFactory,
    UpdateDiagnostics,
    UpdateMethodDescription,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
    flatten_settings,
    json_safe_metric_name,
    json_safe_scalar,
    serialize_parameter_layout,
)
from tpen.training.vmc import (
    DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    resolve_nonfinite_local_energy_policy,
)

torch = require_torch(feature="native structured KFAC")

# Bump whenever the persisted JSON/binary KFAC state schema changes.
KFAC_STATE_VERSION = "kfac-state-1"
KFAC_CONVENTION_VERSION = "kfac-native-expand-uncentered-kron-centered-dense-halfgrad-1"
KFAC_DAMPING_TRACE_FLOOR = 1.0e-12
KFAC_KL_DENOM_FLOOR = 1.0e-30

KFAC_GRID_FACTOR_KEYS: tuple[str, ...] = (
    "eta",
    "relative_damping",
    "factor_ema",
    "curvature_factor_update_interval",
    "inverse_refresh_interval",
    "separate_momentum",
    "predicted_kl_cap",
)


def _is_finite_float(value: float) -> bool:
    """Return whether a Python float is finite."""

    return math.isfinite(value)


def _require_real_float(
    value: Any,
    *,
    field_name: str,
    strictly_positive: bool = False,
    min_value: float | None = None,
    max_exclusive: float | None = None,
) -> float:
    """Validate and coerce a real numeric scalar, rejecting booleans and non-reals."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a real number, not bool ({value!r})")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number, got {type(value).__name__}")
    numeric = float(value)
    if not _is_finite_float(numeric):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if strictly_positive and not numeric > 0.0:
        raise ValueError(f"{field_name} must be positive, got {numeric!r}")
    if min_value is not None and numeric < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}, got {numeric!r}")
    if max_exclusive is not None and numeric >= max_exclusive:
        raise ValueError(f"{field_name} must be < {max_exclusive}, got {numeric!r}")
    return numeric


def _require_positive_int(value: Any, *, field_name: str) -> int:
    """Validate a positive Python integer, rejecting booleans and floats."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer, not bool ({value!r})")
    if type(value) is not int:
        raise TypeError(f"{field_name} must be a positive integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value!r}")
    return value


def _resolve_solve_dtype(value: Any, *, field_name: str) -> torch.dtype:
    """Resolve ``torch.float32`` / ``torch.float64`` or its string name."""

    candidate = value
    if isinstance(candidate, str):
        dtype_name = candidate.removeprefix("torch.")
        candidate = getattr(torch, dtype_name, None)
    if not isinstance(candidate, torch.dtype) or candidate not in (torch.float32, torch.float64):
        raise ValueError(
            f"{field_name} must be torch.float32 or torch.float64, got {value!r}"
        )
    return candidate


@dataclass(frozen=True, kw_only=True)
class KFACPolicy:
    """Numerical, cadence, damping, and trust-region policy for :class:`NativeKFACUpdate`.

    Parameters
    ----------
    learning_rate : float, optional
        Step size ``eta`` applied to the preconditioned direction via plain SGD.
    relative_damping : float, optional
        Trace-normalized relative damping multiplier ``lambda_rel > 0``.
    factor_ema : float, optional
        Exponential moving average decay in ``[0.0, 1.0)`` for curvature factors.
    curvature_factor_update_interval : int, optional
        Positive step cadence for updating EMA curvature factors.
    inverse_refresh_interval : int, optional
        Positive step cadence for refreshing cached eigendecompositions.
    predicted_kl_cap : float or None, optional
        Positive KL trust-region cap ``kl_cap``, or ``None`` to disable capping.
    solve_dtype : torch.dtype or str, optional
        Real floating dtype (``torch.float64`` default, or ``torch.float32``)
        for curvature accumulation, eigendecomposition, and solves.
    max_dense_block_numel : int, optional
        Maximum allowed scalar element count for any :class:`DenseFactorBlock`.
    damping_trace_floor : float, optional
        Positive floor ``1e-12`` applied to ``tau_b`` before scaling by
        ``relative_damping`` so zero blocks at ``n = 2`` stay strictly SPD.
    """

    learning_rate: float = 0.03
    relative_damping: float = 1.0e-3
    factor_ema: float = 0.95
    curvature_factor_update_interval: int = 1
    inverse_refresh_interval: int = 10
    predicted_kl_cap: float | None = 1.0e-3
    solve_dtype: Any = torch.float64
    max_dense_block_numel: int = 2048
    damping_trace_floor: float = KFAC_DAMPING_TRACE_FLOOR

    def __post_init__(self) -> None:
        lr = _require_real_float(
            self.learning_rate, field_name="KFACPolicy.learning_rate", strictly_positive=True
        )
        rel_damp = _require_real_float(
            self.relative_damping,
            field_name="KFACPolicy.relative_damping",
            strictly_positive=True,
        )
        ema = _require_real_float(
            self.factor_ema,
            field_name="KFACPolicy.factor_ema",
            min_value=0.0,
            max_exclusive=1.0,
        )
        factor_interval = _require_positive_int(
            self.curvature_factor_update_interval,
            field_name="KFACPolicy.curvature_factor_update_interval",
        )
        refresh_interval = _require_positive_int(
            self.inverse_refresh_interval,
            field_name="KFACPolicy.inverse_refresh_interval",
        )
        kl_cap = (
            None
            if self.predicted_kl_cap is None
            else _require_real_float(
                self.predicted_kl_cap,
                field_name="KFACPolicy.predicted_kl_cap",
                strictly_positive=True,
            )
        )
        resolved_dtype = _resolve_solve_dtype(
            self.solve_dtype, field_name="KFACPolicy.solve_dtype"
        )
        max_dense = _require_positive_int(
            self.max_dense_block_numel, field_name="KFACPolicy.max_dense_block_numel"
        )
        trace_floor = _require_real_float(
            self.damping_trace_floor,
            field_name="KFACPolicy.damping_trace_floor",
            strictly_positive=True,
        )
        object.__setattr__(self, "learning_rate", lr)
        object.__setattr__(self, "relative_damping", rel_damp)
        object.__setattr__(self, "factor_ema", ema)
        object.__setattr__(self, "curvature_factor_update_interval", factor_interval)
        object.__setattr__(self, "inverse_refresh_interval", refresh_interval)
        object.__setattr__(self, "predicted_kl_cap", kl_cap)
        object.__setattr__(self, "solve_dtype", resolved_dtype)
        object.__setattr__(self, "max_dense_block_numel", max_dense)
        object.__setattr__(self, "damping_trace_floor", trace_floor)

    def validate(self) -> Self:
        """Re-validate all policy bounds and dtypes."""

        _require_real_float(
            self.learning_rate, field_name="KFACPolicy.learning_rate", strictly_positive=True
        )
        _require_real_float(
            self.relative_damping,
            field_name="KFACPolicy.relative_damping",
            strictly_positive=True,
        )
        _require_real_float(
            self.factor_ema,
            field_name="KFACPolicy.factor_ema",
            min_value=0.0,
            max_exclusive=1.0,
        )
        _require_positive_int(
            self.curvature_factor_update_interval,
            field_name="KFACPolicy.curvature_factor_update_interval",
        )
        _require_positive_int(
            self.inverse_refresh_interval,
            field_name="KFACPolicy.inverse_refresh_interval",
        )
        if self.predicted_kl_cap is not None:
            _require_real_float(
                self.predicted_kl_cap,
                field_name="KFACPolicy.predicted_kl_cap",
                strictly_positive=True,
            )
        _resolve_solve_dtype(self.solve_dtype, field_name="KFACPolicy.solve_dtype")
        _require_positive_int(
            self.max_dense_block_numel, field_name="KFACPolicy.max_dense_block_numel"
        )
        _require_real_float(
            self.damping_trace_floor,
            field_name="KFACPolicy.damping_trace_floor",
            strictly_positive=True,
        )
        return self

    def fingerprint(self) -> dict[str, Any]:
        """Return JSON-safe policy identity for checkpoint continuity and telemetry."""

        return {
            "learning_rate": self.learning_rate,
            "relative_damping": self.relative_damping,
            "factor_ema": self.factor_ema,
            "curvature_factor_update_interval": self.curvature_factor_update_interval,
            "inverse_refresh_interval": self.inverse_refresh_interval,
            "predicted_kl_cap": self.predicted_kl_cap,
            "solve_dtype": str(self.solve_dtype),
            "max_dense_block_numel": self.max_dense_block_numel,
            "damping_trace_floor": self.damping_trace_floor,
        }


@dataclass(frozen=True, kw_only=True)
class KFACGridHyperparameters:
    """Validated helium-importance authority grid hyperparameters for KFAC.

    Maps the exact 7 factors declared in the HI optimizer grid authority:
    ``eta``, ``relative_damping``, ``factor_ema``,
    ``curvature_factor_update_interval``, ``inverse_refresh_interval``,
    ``separate_momentum`` (must be ``False``), and ``predicted_kl_cap``.
    """

    eta: float
    relative_damping: float
    factor_ema: float
    curvature_factor_update_interval: int
    inverse_refresh_interval: int
    separate_momentum: bool
    predicted_kl_cap: float

    def __post_init__(self) -> None:
        eta = _require_real_float(
            self.eta, field_name="KFACGridHyperparameters.eta", strictly_positive=True
        )
        rel_damp = _require_real_float(
            self.relative_damping,
            field_name="KFACGridHyperparameters.relative_damping",
            strictly_positive=True,
        )
        ema = _require_real_float(
            self.factor_ema,
            field_name="KFACGridHyperparameters.factor_ema",
            min_value=0.0,
            max_exclusive=1.0,
        )
        factor_interval = _require_positive_int(
            self.curvature_factor_update_interval,
            field_name="KFACGridHyperparameters.curvature_factor_update_interval",
        )
        refresh_interval = _require_positive_int(
            self.inverse_refresh_interval,
            field_name="KFACGridHyperparameters.inverse_refresh_interval",
        )
        if type(self.separate_momentum) is not bool:
            raise TypeError(
                "KFACGridHyperparameters.separate_momentum must be a bool, "
                f"got {type(self.separate_momentum).__name__}"
            )
        if self.separate_momentum:
            raise ValueError(
                "KFACGridHyperparameters refuses separate_momentum=True; the native "
                "KFAC path requires separate_momentum=False with zero separate momentum"
            )
        kl_cap = _require_real_float(
            self.predicted_kl_cap,
            field_name="KFACGridHyperparameters.predicted_kl_cap",
            strictly_positive=True,
        )
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "relative_damping", rel_damp)
        object.__setattr__(self, "factor_ema", ema)
        object.__setattr__(self, "curvature_factor_update_interval", factor_interval)
        object.__setattr__(self, "inverse_refresh_interval", refresh_interval)
        object.__setattr__(self, "separate_momentum", False)
        object.__setattr__(self, "predicted_kl_cap", kl_cap)

    @classmethod
    def from_factors(cls, factors: Mapping[str, Any]) -> "KFACGridHyperparameters":
        """Construct validated KFAC hyperparameters from an exact HI factor mapping."""

        if not isinstance(factors, Mapping):
            raise TypeError(
                f"KFAC grid factors must be a Mapping, got {type(factors).__name__}"
            )
        keys = set(factors.keys())
        expected = set(KFAC_GRID_FACTOR_KEYS)
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        if missing:
            raise ValueError(f"KFAC grid factors missing required key(s): {missing!r}")
        if unknown:
            raise ValueError(f"KFAC grid factors contain unknown key(s): {unknown!r}")
        return cls(
            eta=factors["eta"],
            relative_damping=factors["relative_damping"],
            factor_ema=factors["factor_ema"],
            curvature_factor_update_interval=factors["curvature_factor_update_interval"],
            inverse_refresh_interval=factors["inverse_refresh_interval"],
            separate_momentum=factors["separate_momentum"],
            predicted_kl_cap=factors["predicted_kl_cap"],
        )

    def to_policy(
        self,
        *,
        solve_dtype: Any = torch.float64,
        max_dense_block_numel: int = 2048,
        damping_trace_floor: float = KFAC_DAMPING_TRACE_FLOOR,
    ) -> KFACPolicy:
        """Convert validated HI grid hyperparameters into a :class:`KFACPolicy`."""

        return KFACPolicy(
            learning_rate=self.eta,
            relative_damping=self.relative_damping,
            factor_ema=self.factor_ema,
            curvature_factor_update_interval=self.curvature_factor_update_interval,
            inverse_refresh_interval=self.inverse_refresh_interval,
            predicted_kl_cap=self.predicted_kl_cap,
            solve_dtype=solve_dtype,
            max_dense_block_numel=max_dense_block_numel,
            damping_trace_floor=damping_trace_floor,
        )

    def to_factor_dict(self) -> dict[str, Any]:
        """Return exact readback dictionary keyed by :data:`KFAC_GRID_FACTOR_KEYS`."""

        return {
            "eta": self.eta,
            "relative_damping": self.relative_damping,
            "factor_ema": self.factor_ema,
            "curvature_factor_update_interval": self.curvature_factor_update_interval,
            "inverse_refresh_interval": self.inverse_refresh_interval,
            "separate_momentum": self.separate_momentum,
            "predicted_kl_cap": self.predicted_kl_cap,
        }


def kfac_policy_from_factors(
    factors: Mapping[str, Any],
    *,
    solve_dtype: Any = torch.float64,
    max_dense_block_numel: int = 2048,
    damping_trace_floor: float = KFAC_DAMPING_TRACE_FLOOR,
) -> KFACPolicy:
    """Map an exact 7-factor KFAC HI grid cell to a validated :class:`KFACPolicy`."""

    return KFACGridHyperparameters.from_factors(factors).to_policy(
        solve_dtype=solve_dtype,
        max_dense_block_numel=max_dense_block_numel,
        damping_trace_floor=damping_trace_floor,
    )


def build_kfac_update_from_factors(
    factors: Mapping[str, Any],
    *,
    model_parameters: ModelParameterBinding,
    optimizer: Any | None = None,
    solve_dtype: Any = torch.float64,
    max_dense_block_numel: int = 2048,
    damping_trace_floor: float = KFAC_DAMPING_TRACE_FLOOR,
    nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
) -> "NativeKFACUpdate":
    """Build a :class:`NativeKFACUpdate` directly from one HI authority grid cell."""

    if not isinstance(model_parameters, ModelParameterBinding):
        raise TypeError(
            "build_kfac_update_from_factors.model_parameters must be a ModelParameterBinding"
        )
    policy = kfac_policy_from_factors(
        factors,
        solve_dtype=solve_dtype,
        max_dense_block_numel=max_dense_block_numel,
        damping_trace_floor=damping_trace_floor,
    )
    resolved_optimizer = (
        torch.optim.SGD(model_parameters.parameters, lr=policy.learning_rate)
        if optimizer is None
        else optimizer
    )
    return NativeKFACUpdate(
        resolved_optimizer,
        model_parameters=model_parameters,
        policy=policy,
        nonfinite_local_energy_policy=nonfinite_local_energy_policy,
    )


@dataclass(frozen=True, kw_only=True)
class KFACTelemetry(UpdateDiagnostics):
    """Observable JSON-safe record of one :class:`NativeKFACUpdate` attempt."""

    applied: bool
    reason: str
    step: int
    curvature_step: int
    n_samples: int
    n_finite_samples: int
    n_parameters: int
    n_kronecker_blocks: int
    n_dense_blocks: int
    factors_updated: bool
    inverse_refreshed: bool
    energy_gradient_norm: float
    update_direction_norm: float
    applied_update_norm: float
    predicted_kl: float
    trust_scale: float
    vg_sum: float
    damped_quadratic_term: float
    solve_dtype: str

    def as_metrics(self, *, prefix: str = "kfac") -> dict[str, Any]:
        """Return flat JSON-safe telemetry keys for training metrics sinks."""

        prefix = json_safe_metric_name(prefix)
        return {
            f"{prefix}_applied": bool(self.applied),
            f"{prefix}_reason": json_safe_scalar(self.reason),
            f"{prefix}_step": int(self.step),
            f"{prefix}_curvature_step": int(self.curvature_step),
            f"{prefix}_samples": int(self.n_samples),
            f"{prefix}_finite_samples": int(self.n_finite_samples),
            f"{prefix}_parameters": int(self.n_parameters),
            f"{prefix}_kronecker_blocks": int(self.n_kronecker_blocks),
            f"{prefix}_dense_blocks": int(self.n_dense_blocks),
            f"{prefix}_factors_updated": bool(self.factors_updated),
            f"{prefix}_inverse_refreshed": bool(self.inverse_refreshed),
            f"{prefix}_energy_gradient_norm": json_safe_scalar(self.energy_gradient_norm),
            f"{prefix}_update_direction_norm": json_safe_scalar(self.update_direction_norm),
            f"{prefix}_applied_update_norm": json_safe_scalar(self.applied_update_norm),
            f"{prefix}_predicted_kl": json_safe_scalar(self.predicted_kl),
            f"{prefix}_trust_scale": json_safe_scalar(self.trust_scale),
            f"{prefix}_vg_sum": json_safe_scalar(self.vg_sum),
            f"{prefix}_damped_quadratic_term": json_safe_scalar(self.damped_quadratic_term),
            f"{prefix}_solve_dtype": json_safe_scalar(self.solve_dtype),
        }


class NativeKFACUpdate(VMCUpdateMethod[StructuredCurvatureUpdateInput]):
    """Native pure-PyTorch structured KFAC update method.

    Parameters
    ----------
    optimizer : torch.optim.SGD
        Plain SGD optimizer (zero momentum, dampening, weight decay, Nesterov,
        and ``maximize=False``) with ``lr == policy.learning_rate``.
    model_parameters : ModelParameterBinding
        Authoritative live trainable-parameter binding.
    policy : KFACPolicy or None, optional
        KFAC numerical, EMA, refresh, damping, and KL-cap policy.
    nonfinite_local_energy_policy : {"fail", "mask"}, optional
        Active non-finite local-energy policy.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        model_parameters: ModelParameterBinding,
        policy: KFACPolicy | None = None,
        nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    ) -> None:
        self.nonfinite_local_energy_policy = resolve_nonfinite_local_energy_policy(
            nonfinite_local_energy_policy
        )
        resolved_policy = KFACPolicy() if policy is None else policy
        if not isinstance(resolved_policy, KFACPolicy):
            raise TypeError("NativeKFACUpdate.policy must be a KFACPolicy")
        resolved_policy.validate()
        _validate_plain_sgd(optimizer, learning_rate=resolved_policy.learning_rate)
        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError("NativeKFACUpdate.model_parameters must be a ModelParameterBinding")

        self.optimizer = optimizer
        self.model_parameters = model_parameters
        self.policy = resolved_policy
        self.completed_updates = 0
        self.curvature_step = 0
        self._optimizer_scope: ScopeFactory | None = None
        self._last_result: VMCUpdateResult | None = None

        # Structured curvature state initialized on curvature_step == 0 and
        # persisted in binary ``method_state.pt`` for bit-exact resume parity.
        self._block_signature: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] | None = None
        self._kronecker_ema: list[dict[str, torch.Tensor]] | None = None
        self._kronecker_eigs: list[dict[str, Any]] | None = None
        self._dense_ema: list[dict[str, torch.Tensor]] | None = None
        self._dense_eigs: list[dict[str, Any]] | None = None

    @property
    def last_telemetry(self) -> KFACTelemetry | None:
        """Return diagnostics from the most recent update attempt."""

        result = self._last_result
        if result is None:
            return None
        diagnostics = result.diagnostics
        assert diagnostics is None or isinstance(diagnostics, KFACTelemetry)
        return diagnostics

    def forward_request(self) -> None:
        """Return ``None`` because KFAC declares a step-level request."""

        return None

    def step_request(self) -> StructuredCurvatureStepRequest:
        """Request detached structured curvature factors and energy half-gradient."""

        return StructuredCurvatureStepRequest(
            solve_dtype=self.policy.solve_dtype,
            max_dense_block_numel=self.policy.max_dense_block_numel,
        )

    def norm_semantics(self) -> str:
        """Report the norm as the L2 norm of the energy half-gradient."""

        return "energy_half_gradient_l2_preconditioned_direction_in_grad"

    def set_step_scopes(
        self,
        *,
        backward_scope: ScopeFactory | None = None,
        optimizer_scope: ScopeFactory | None = None,
    ) -> None:
        """Install optional typed optimizer event scope for one trainer fit."""

        del backward_scope
        self._optimizer_scope = optimizer_scope

    def describe(self) -> UpdateMethodDescription:
        """Add KFAC policy, convention identity, and HI factors to description."""

        base = super().describe()
        settings = dict(base.settings)
        settings.update(flatten_settings(self.policy.fingerprint(), prefix="policy_"))
        settings["convention"] = KFAC_CONVENTION_VERSION
        settings["separate_momentum"] = False
        return replace(base, settings=settings)

    def update_state(self) -> VMCUpdateState:
        """Return the single plain-SGD carrier and live binding owned by KFAC."""

        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Adopt restored live model parameter references after checkpoint load."""

        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError("NativeKFACUpdate.model_parameters must be a ModelParameterBinding")
        if not self.model_parameters.layout.compare(model_parameters.layout)[0]:
            raise ValueError("NativeKFACUpdate model parameter layout changed during rebind")
        self.model_parameters = model_parameters

    def method_state_dict(self) -> dict[str, Any]:
        """Return compact JSON-safe KFAC metadata persisted in ``trainer.json``.

        Large float64 EMA factors and cached eigendecompositions are never
        serialized into JSON; they live in the optional binary ``method_state.pt``
        sidecar returned by :meth:`method_tensor_state_dict`.
        """

        return {
            "version": KFAC_STATE_VERSION,
            "convention": KFAC_CONVENTION_VERSION,
            "parameter_layout": serialize_parameter_layout(self.model_parameters.layout),
            "policy": self.policy.fingerprint(),
            "completed_updates": self.completed_updates,
            "curvature_step": self.curvature_step,
            "has_curvature_state": self._block_signature is not None,
            "block_signature": (
                None
                if self._block_signature is None
                else [
                    {"kind": kind, "ordinals": list(ordinals), "dims": list(dims)}
                    for kind, ordinals, dims in self._block_signature
                ]
            ),
        }

    def _parse_method_state(
        self, state: Mapping[str, Any]
    ) -> tuple[int, int, tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] | None]:
        """Validate and parse compact KFAC JSON metadata without mutating live state."""

        if not isinstance(state, Mapping):
            raise TypeError("NativeKFACUpdate state must be a mapping")
        if state.get("version") != KFAC_STATE_VERSION:
            raise ValueError(
                f"unsupported KFAC state version {state.get('version')!r}, "
                f"expected {KFAC_STATE_VERSION!r}"
            )
        if state.get("convention") != KFAC_CONVENTION_VERSION:
            raise ValueError(
                f"KFAC checkpoint convention {state.get('convention')!r} disagrees with "
                f"live convention {KFAC_CONVENTION_VERSION!r}"
            )
        if state.get("parameter_layout") != serialize_parameter_layout(
            self.model_parameters.layout
        ):
            raise ValueError("KFAC checkpoint parameter layout does not match the live model")
        if state.get("policy") != self.policy.fingerprint():
            raise ValueError("KFAC checkpoint policy does not match the live method")

        completed_updates = state.get("completed_updates")
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("KFAC state completed_updates must be a non-negative integer")
        curvature_step = state.get("curvature_step")
        if type(curvature_step) is not int or curvature_step < 0:
            raise ValueError("KFAC state curvature_step must be a non-negative integer")
        has_curvature_state = state.get("has_curvature_state")
        if type(has_curvature_state) is not bool:
            raise ValueError("KFAC state has_curvature_state must be a bool")
        if has_curvature_state != (curvature_step > 0):
            raise ValueError(
                f"KFAC state has_curvature_state={has_curvature_state} disagrees with "
                f"curvature_step={curvature_step}"
            )

        raw_sig = state.get("block_signature")
        parsed_sig = _parse_block_signature(raw_sig, expect_present=has_curvature_state)
        return completed_updates, curvature_step, parsed_sig

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore compact KFAC metadata, rejecting any version/policy/layout drift."""

        self.validate_method_state_dict(state)
        completed_updates, curvature_step, parsed_sig = self._parse_method_state(state)

        self.completed_updates = completed_updates
        self.curvature_step = curvature_step
        self._block_signature = parsed_sig
        # Clear any stale in-memory curvature tensors; binary sidecar restore
        # populates them when has_curvature_state is True.
        self._kronecker_ema = None
        self._kronecker_eigs = None
        self._dense_ema = None
        self._dense_eigs = None

    def validate_method_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate compact KFAC metadata without mutating live state."""

        self._parse_method_state(state)

    def method_tensor_state_dict(self) -> dict[str, Any] | None:
        """Return detached CPU curvature tensors for binary ``method_state.pt``."""

        if self._block_signature is None:
            return None
        assert self._kronecker_ema is not None
        assert self._kronecker_eigs is not None
        assert self._dense_ema is not None
        assert self._dense_eigs is not None
        return {
            "version": KFAC_STATE_VERSION,
            "convention": KFAC_CONVENTION_VERSION,
            "curvature_step": self.curvature_step,
            "solve_dtype": str(self.policy.solve_dtype),
            "block_signature": [
                {"kind": kind, "ordinals": list(ordinals), "dims": list(dims)}
                for kind, ordinals, dims in self._block_signature
            ],
            "kronecker_ema": [
                {"A": entry["A"].detach().cpu(), "G": entry["G"].detach().cpu()}
                for entry in self._kronecker_ema
            ],
            "kronecker_eigs": [
                {
                    "d_A": entry["d_A"].detach().cpu(),
                    "Q_A": entry["Q_A"].detach().cpu(),
                    "d_G": entry["d_G"].detach().cpu(),
                    "Q_G": entry["Q_G"].detach().cpu(),
                    "damping": float(entry["damping"]),
                }
                for entry in self._kronecker_eigs
            ],
            "dense_ema": [
                {"F": entry["F"].detach().cpu()} for entry in self._dense_ema
            ],
            "dense_eigs": [
                {
                    "d_F": entry["d_F"].detach().cpu(),
                    "Q_F": entry["Q_F"].detach().cpu(),
                    "damping": float(entry["damping"]),
                }
                for entry in self._dense_eigs
            ],
        }

    def _parse_method_tensor_state(
        self,
        state: Mapping[str, Any] | None,
        *,
        expected_curvature_step: int,
        expected_block_signature: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] | None,
    ) -> tuple[
        list[dict[str, torch.Tensor]] | None,
        list[dict[str, Any]] | None,
        list[dict[str, torch.Tensor]] | None,
        list[dict[str, Any]] | None,
    ]:
        """Validate and materialize ``method_state.pt`` tensors without mutating live state."""

        if expected_block_signature is None:
            if state is not None:
                raise ValueError(
                    "NativeKFACUpdate received unexpected binary method_state.pt payload "
                    "for curvature_step == 0"
                )
            return None, None, None, None

        if state is None:
            raise ValueError(
                "NativeKFACUpdate requires binary method_state.pt curvature tensor state "
                f"for curvature_step={expected_curvature_step}, but none was provided"
            )
        if not isinstance(state, Mapping):
            raise TypeError("NativeKFACUpdate binary tensor state must be a mapping")
        if state.get("version") != KFAC_STATE_VERSION:
            raise ValueError(
                f"unsupported KFAC binary tensor state version {state.get('version')!r}, "
                f"expected {KFAC_STATE_VERSION!r}"
            )
        if state.get("convention") != KFAC_CONVENTION_VERSION:
            raise ValueError(
                "KFAC binary tensor state convention does not match live method"
            )
        if state.get("curvature_step") != expected_curvature_step:
            raise ValueError(
                f"KFAC binary tensor state curvature_step {state.get('curvature_step')!r} "
                f"disagrees with trainer metadata curvature_step={expected_curvature_step}"
            )
        if state.get("solve_dtype") != str(self.policy.solve_dtype):
            raise ValueError(
                f"KFAC binary tensor state solve_dtype {state.get('solve_dtype')!r} "
                f"disagrees with live policy solve_dtype={str(self.policy.solve_dtype)!r}"
            )
        recorded_sig = _parse_block_signature(state.get("block_signature"), expect_present=True)
        if recorded_sig != expected_block_signature:
            raise ValueError(
                "KFAC binary tensor state block_signature disagrees with trainer metadata"
            )

        kron_specs = [spec for spec in expected_block_signature if spec[0] == "kronecker"]
        dense_specs = [spec for spec in expected_block_signature if spec[0] == "dense"]

        raw_kron_ema = state.get("kronecker_ema")
        raw_kron_eigs = state.get("kronecker_eigs")
        raw_dense_ema = state.get("dense_ema")
        raw_dense_eigs = state.get("dense_eigs")
        if (
            not isinstance(raw_kron_ema, list)
            or not isinstance(raw_kron_eigs, list)
            or not isinstance(raw_dense_ema, list)
            or not isinstance(raw_dense_eigs, list)
        ):
            raise ValueError("KFAC binary tensor state lists are missing or malformed")
        if len(raw_kron_ema) != len(kron_specs) or len(raw_kron_eigs) != len(kron_specs):
            raise ValueError("KFAC binary tensor state Kronecker block count mismatch")
        if len(raw_dense_ema) != len(dense_specs) or len(raw_dense_eigs) != len(dense_specs):
            raise ValueError("KFAC binary tensor state Dense block count mismatch")

        target_device = self._target_device()
        solve_dtype = self.policy.solve_dtype

        restored_kron_ema: list[dict[str, torch.Tensor]] = []
        restored_kron_eigs: list[dict[str, Any]] = []
        for (_kind, _ords, dims), ema_entry, eig_entry in zip(
            kron_specs, raw_kron_ema, raw_kron_eigs, strict=True
        ):
            out_dim, in_dim = dims
            if not isinstance(ema_entry, Mapping) or not isinstance(eig_entry, Mapping):
                raise ValueError("KFAC binary Kronecker state entries must be mappings")
            A = _validate_restored_tensor(
                ema_entry.get("A"), shape=(in_dim, in_dim), dtype=solve_dtype, device=target_device, label="A_ema"
            )
            G = _validate_restored_tensor(
                ema_entry.get("G"), shape=(out_dim, out_dim), dtype=solve_dtype, device=target_device, label="G_ema"
            )
            d_A = _validate_restored_tensor(
                eig_entry.get("d_A"),
                shape=(in_dim,),
                dtype=solve_dtype,
                device=target_device,
                label="d_A",
                non_negative=True,
            )
            Q_A = _validate_restored_tensor(
                eig_entry.get("Q_A"), shape=(in_dim, in_dim), dtype=solve_dtype, device=target_device, label="Q_A"
            )
            d_G = _validate_restored_tensor(
                eig_entry.get("d_G"),
                shape=(out_dim,),
                dtype=solve_dtype,
                device=target_device,
                label="d_G",
                non_negative=True,
            )
            Q_G = _validate_restored_tensor(
                eig_entry.get("Q_G"), shape=(out_dim, out_dim), dtype=solve_dtype, device=target_device, label="Q_G"
            )
            damping = _require_real_float(
                eig_entry.get("damping"), field_name="kronecker_eigs.damping", strictly_positive=True
            )
            restored_kron_ema.append({"A": A, "G": G})
            restored_kron_eigs.append(
                {"d_A": d_A, "Q_A": Q_A, "d_G": d_G, "Q_G": Q_G, "damping": damping}
            )

        restored_dense_ema: list[dict[str, torch.Tensor]] = []
        restored_dense_eigs: list[dict[str, Any]] = []
        for (_kind, _ords, dims), ema_entry, eig_entry in zip(
            dense_specs, raw_dense_ema, raw_dense_eigs, strict=True
        ):
            (dim,) = dims
            if not isinstance(ema_entry, Mapping) or not isinstance(eig_entry, Mapping):
                raise ValueError("KFAC binary Dense state entries must be mappings")
            F = _validate_restored_tensor(
                ema_entry.get("F"), shape=(dim, dim), dtype=solve_dtype, device=target_device, label="F_ema"
            )
            d_F = _validate_restored_tensor(
                eig_entry.get("d_F"),
                shape=(dim,),
                dtype=solve_dtype,
                device=target_device,
                label="d_F",
                non_negative=True,
            )
            Q_F = _validate_restored_tensor(
                eig_entry.get("Q_F"), shape=(dim, dim), dtype=solve_dtype, device=target_device, label="Q_F"
            )
            damping = _require_real_float(
                eig_entry.get("damping"), field_name="dense_eigs.damping", strictly_positive=True
            )
            restored_dense_ema.append({"F": F})
            restored_dense_eigs.append({"d_F": d_F, "Q_F": Q_F, "damping": damping})

        return restored_kron_ema, restored_kron_eigs, restored_dense_ema, restored_dense_eigs

    def validate_method_tensor_state_dict(
        self,
        state: Mapping[str, Any] | None,
        *,
        method_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate KFAC metadata and binary curvature state together without mutating live state."""

        if method_state is not None:
            _completed_updates, expected_cstep, expected_sig = self._parse_method_state(
                method_state
            )
        else:
            expected_cstep = self.curvature_step
            expected_sig = self._block_signature
        self._parse_method_tensor_state(
            state,
            expected_curvature_step=expected_cstep,
            expected_block_signature=expected_sig,
        )

    def load_method_state_and_tensors(
        self,
        method_state: Mapping[str, Any],
        method_tensor_state: Mapping[str, Any] | None,
    ) -> None:
        """Atomically validate and commit KFAC JSON metadata and ``method_state.pt`` tensors."""

        completed_updates, curvature_step, parsed_sig = self._parse_method_state(method_state)
        kron_ema, kron_eigs, dense_ema, dense_eigs = self._parse_method_tensor_state(
            method_tensor_state,
            expected_curvature_step=curvature_step,
            expected_block_signature=parsed_sig,
        )
        self.completed_updates = completed_updates
        self.curvature_step = curvature_step
        self._block_signature = parsed_sig
        self._kronecker_ema = kron_ema
        self._kronecker_eigs = kron_eigs
        self._dense_ema = dense_ema
        self._dense_eigs = dense_eigs

    def load_method_tensor_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """Restore binary ``method_state.pt`` EMA factors and cached eigendecompositions."""

        kron_ema, kron_eigs, dense_ema, dense_eigs = self._parse_method_tensor_state(
            state,
            expected_curvature_step=self.curvature_step,
            expected_block_signature=self._block_signature,
        )
        self._kronecker_ema = kron_ema
        self._kronecker_eigs = kron_eigs
        self._dense_ema = dense_ema
        self._dense_eigs = dense_eigs

    def update(self, update_input: StructuredCurvatureUpdateInput) -> VMCUpdateResult:
        """Update EMA factors, optionally refresh eigenbases, and apply one KFAC step."""

        if not isinstance(update_input, StructuredCurvatureUpdateInput):
            raise TypeError(
                "NativeKFACUpdate requires StructuredCurvatureUpdateInput, "
                f"got {type(update_input).__name__}"
            )
        update_input.validate()
        if update_input.wavefunction.phase is not None:
            raise ValueError(
                "NativeKFACUpdate supports real wavefunctions only (phase must be None)"
            )
        self._validate_binding(update_input)
        # Re-verify plain-SGD invariants before mutating EMA or stepping so
        # post-init or post-restore optimizer mutation cannot silently disagree
        # with policy.learning_rate and predicted_kl_cap.
        _validate_plain_sgd(self.optimizer, learning_rate=self.policy.learning_rate)
        if update_input.solve_dtype != self.policy.solve_dtype:
            raise ValueError(
                f"StructuredCurvatureUpdateInput.solve_dtype={update_input.solve_dtype} "
                f"disagrees with KFACPolicy.solve_dtype={self.policy.solve_dtype}"
            )

        local_energy = update_input.local_energy.reshape(-1)
        n_samples = int(local_energy.numel())
        n_parameters = self.model_parameters.layout.total_numel
        energy_finite = torch.isfinite(local_energy)
        n_finite = int(energy_finite.sum().item())

        if self.nonfinite_local_energy_policy == "fail" and not bool(energy_finite.all()):
            n_bad = int((~energy_finite).sum().item())
            raise RuntimeError(
                f"NativeKFACUpdate refused a step: {n_bad} of {n_samples} local-energy "
                "samples are non-finite and the active policy is 'fail'"
            )
        if update_input.n_finite_samples != n_finite:
            raise ValueError(
                f"StructuredCurvatureUpdateInput.n_finite_samples={update_input.n_finite_samples} "
                f"disagrees with observed finite local-energy count {n_finite}"
            )

        if update_input.batch.n_electrons == 0:
            return self._skip(
                reason="zero_electron_batch",
                step=update_input.step,
                n_samples=n_samples,
                n_finite=n_finite,
                n_parameters=n_parameters,
            )
        if n_finite == 0:
            raise RuntimeError(
                "cannot compute a KFAC update: no finite local-energy sample remains "
                "for a nonzero-electron batch"
            )
        if n_finite < 2:
            return self._skip(
                reason="insufficient_finite_samples",
                step=update_input.step,
                n_samples=n_samples,
                n_finite=n_finite,
                n_parameters=n_parameters,
            )
        if not update_input.curvature_available:
            raise ValueError(
                "NativeKFACUpdate received curvature_available=False on a non-degenerate batch"
            )

        if self.curvature_step > 0 and (
            self._kronecker_ema is None
            or self._kronecker_eigs is None
            or self._dense_ema is None
            or self._dense_eigs is None
        ):
            raise RuntimeError(
                "NativeKFACUpdate curvature tensor state was not restored for "
                f"curvature_step={self.curvature_step}"
            )

        # Atomic pre-validation BEFORE mutating any EMA, eigencache, or parameter.
        kron_blocks = update_input.kronecker_blocks
        dense_blocks = update_input.dense_blocks
        half_grads = update_input.energy_half_gradient

        incoming_signature = _signature_from_blocks(kron_blocks, dense_blocks)
        if self._block_signature is not None and incoming_signature != self._block_signature:
            raise ValueError(
                "StructuredCurvatureUpdateInput block topology disagrees with KFAC EMA state"
            )

        for idx, blk in enumerate(kron_blocks):
            if not bool(torch.isfinite(blk.A).all()) or not bool(torch.isfinite(blk.G).all()):
                raise RuntimeError(
                    f"NativeKFACUpdate refused non-finite Kronecker factor block {idx} on pinned rows"
                )
        for idx, blk in enumerate(dense_blocks):
            if blk.F.numel() > 0 and not bool(torch.isfinite(blk.F).all()):
                raise RuntimeError(
                    f"NativeKFACUpdate refused non-finite Dense factor block {idx} on pinned rows"
                )
        for idx, grad in enumerate(half_grads):
            if grad.numel() > 0 and not bool(torch.isfinite(grad).all()):
                raise RuntimeError(
                    f"NativeKFACUpdate refused non-finite energy_half_gradient[{idx}] on pinned rows"
                )

        # Compute next EMA and optional eigenbasis refresh in local temporaries
        # so any numerical failure leaves live KFAC state completely unmutated.
        cstep = self.curvature_step
        factors_updated = False
        inverse_refreshed = False

        if cstep == 0:
            next_kron_ema = [
                {"A": blk.A.detach().clone(), "G": blk.G.detach().clone()}
                for blk in kron_blocks
            ]
            next_dense_ema = [{"F": blk.F.detach().clone()} for blk in dense_blocks]
            factors_updated = True
        elif cstep % self.policy.curvature_factor_update_interval == 0:
            assert self._kronecker_ema is not None and self._dense_ema is not None
            alpha = self.policy.factor_ema
            beta = 1.0 - alpha
            next_kron_ema = [
                {
                    "A": alpha * prev["A"] + beta * blk.A.detach(),
                    "G": alpha * prev["G"] + beta * blk.G.detach(),
                }
                for prev, blk in zip(self._kronecker_ema, kron_blocks, strict=True)
            ]
            next_dense_ema = [
                {"F": alpha * prev["F"] + beta * blk.F.detach()}
                for prev, blk in zip(self._dense_ema, dense_blocks, strict=True)
            ]
            factors_updated = True
        else:
            assert self._kronecker_ema is not None and self._dense_ema is not None
            next_kron_ema = self._kronecker_ema
            next_dense_ema = self._dense_ema

        # Update-before-refresh cadence: refresh at cstep == 0, R, 2R, ...
        if cstep % self.policy.inverse_refresh_interval == 0:
            next_kron_eigs = [
                self._decompose_kronecker(entry["A"], entry["G"]) for entry in next_kron_ema
            ]
            next_dense_eigs = [
                self._decompose_dense(entry["F"]) for entry in next_dense_ema
            ]
            inverse_refreshed = True
        else:
            assert self._kronecker_eigs is not None and self._dense_eigs is not None
            next_kron_eigs = self._kronecker_eigs
            next_dense_eigs = self._dense_eigs

        # Eigen-basis preconditioned solve and exact predicted-KL quadratic form.
        slots = self.model_parameters.layout.slots
        directions_by_ordinal: list[torch.Tensor | None] = [None] * len(slots)
        q_ema_accum = 0.0
        vg_accum = 0.0
        damped_quad_accum = 0.0
        dir_sq_accum = 0.0
        grad_sq_accum = 0.0

        for grad in half_grads:
            grad_sq_accum += float((grad * grad).sum().item())

        for blk, ema_entry, eig_entry in zip(
            kron_blocks, next_kron_ema, next_kron_eigs, strict=True
        ):
            w_ord = blk.parameter_ordinals[0]
            w_shape = slots[w_ord].shape
            g_w = half_grads[w_ord].reshape(blk.out_dim, -1)
            if blk.has_bias:
                b_ord = blk.parameter_ordinals[1]
                g_b = half_grads[b_ord].reshape(blk.out_dim, 1)
                M_b = torch.cat((g_w, g_b), dim=1)
            else:
                M_b = g_w

            d_G = eig_entry["d_G"]
            Q_G = eig_entry["Q_G"]
            d_A = eig_entry["d_A"]
            Q_A = eig_entry["Q_A"]
            lam = float(eig_entry["damping"])

            proj = Q_G.transpose(0, 1) @ M_b @ Q_A
            denom = d_G.unsqueeze(1) * d_A.unsqueeze(0) + lam
            D_b = Q_G @ (proj / denom) @ Q_A.transpose(0, 1)
            if not bool(torch.isfinite(D_b).all()):
                raise RuntimeError(
                    "NativeKFACUpdate produced a non-finite Kronecker preconditioned direction"
                )

            if blk.has_bias:
                b_ord = blk.parameter_ordinals[1]
                directions_by_ordinal[w_ord] = D_b[:, :-1].reshape(w_shape)
                directions_by_ordinal[b_ord] = D_b[:, -1].reshape(slots[b_ord].shape)
            else:
                directions_by_ordinal[w_ord] = D_b.reshape(w_shape)

            # Exact Kronecker quadratic form tr(D_b^T G_ema D_b A_ema) without
            # materializing G_ema (x) A_ema.
            q_b = float((D_b * (ema_entry["G"] @ D_b @ ema_entry["A"])).sum().item())
            vg_b = float((D_b * M_b).sum().item())
            d_sq_b = float((D_b * D_b).sum().item())
            q_ema_accum += q_b
            vg_accum += vg_b
            damped_quad_accum += lam * d_sq_b
            dir_sq_accum += d_sq_b

        for blk, ema_entry, eig_entry in zip(
            dense_blocks, next_dense_ema, next_dense_eigs, strict=True
        ):
            if blk.dim == 0:
                for ordinal in blk.parameter_ordinals:
                    directions_by_ordinal[ordinal] = torch.zeros(
                        slots[ordinal].shape,
                        device=blk.F.device,
                        dtype=self.policy.solve_dtype,
                    )
                continue
            g_vec = torch.cat(
                [half_grads[ordinal].reshape(-1) for ordinal in blk.parameter_ordinals],
                dim=0,
            )
            d_F = eig_entry["d_F"]
            Q_F = eig_entry["Q_F"]
            lam = float(eig_entry["damping"])
            d_vec = Q_F @ ((Q_F.transpose(0, 1) @ g_vec) / (d_F + lam))
            if not bool(torch.isfinite(d_vec).all()):
                raise RuntimeError(
                    "NativeKFACUpdate produced a non-finite Dense preconditioned direction"
                )
            offset = 0
            for ordinal in blk.parameter_ordinals:
                slot = slots[ordinal]
                directions_by_ordinal[ordinal] = d_vec[offset : offset + slot.numel].reshape(
                    slot.shape
                )
                offset += slot.numel

            q_b = float((d_vec * (ema_entry["F"] @ d_vec)).sum().item())
            vg_b = float((d_vec * g_vec).sum().item())
            d_sq_b = float((d_vec * d_vec).sum().item())
            q_ema_accum += q_b
            vg_accum += vg_b
            damped_quad_accum += lam * d_sq_b
            dir_sq_accum += d_sq_b

        if any(direction is None for direction in directions_by_ordinal):
            raise RuntimeError("NativeKFACUpdate failed to populate every parameter direction")

        eta = self.policy.learning_rate
        eta_sq = eta * eta
        if not (
            _is_finite_float(q_ema_accum)
            and _is_finite_float(vg_accum)
            and _is_finite_float(damped_quad_accum)
            and _is_finite_float(dir_sq_accum)
            and _is_finite_float(grad_sq_accum)
        ):
            raise RuntimeError("NativeKFACUpdate produced non-finite quadratic-form scalars")

        q_ema = eta_sq * q_ema_accum
        predicted_kl = 0.5 * max(q_ema, 0.0)
        vg_sum = eta_sq * vg_accum
        damped_quadratic_term = eta_sq * damped_quad_accum

        if self.policy.predicted_kl_cap is None:
            trust_scale = 1.0
        else:
            trust_scale = min(
                1.0,
                math.sqrt(
                    self.policy.predicted_kl_cap / max(predicted_kl, KFAC_KL_DENOM_FLOOR)
                ),
            )

        energy_gradient_norm = math.sqrt(max(grad_sq_accum, 0.0))
        direction_norm = math.sqrt(max(dir_sq_accum, 0.0))
        applied_update_norm = eta * trust_scale * direction_norm

        # Commit curvature state and apply parameter step atomically.
        self._block_signature = incoming_signature
        self._kronecker_ema = next_kron_ema
        self._kronecker_eigs = next_kron_eigs
        self._dense_ema = next_dense_ema
        self._dense_eigs = next_dense_eigs

        for parameter, direction in zip(
            self.model_parameters.parameters, directions_by_ordinal, strict=True
        ):
            assert direction is not None
            parameter.grad = (trust_scale * direction).detach().to(
                dtype=parameter.dtype, device=parameter.device
            )

        scope = (
            nullcontext()
            if self._optimizer_scope is None
            else self._optimizer_scope(update_input.step)
        )
        with scope:
            self.optimizer.step()

        self.curvature_step = cstep + 1
        self.completed_updates += 1

        telemetry = KFACTelemetry(
            applied=True,
            reason="applied",
            step=update_input.step,
            curvature_step=cstep,
            n_samples=n_samples,
            n_finite_samples=n_finite,
            n_parameters=n_parameters,
            n_kronecker_blocks=len(kron_blocks),
            n_dense_blocks=len(dense_blocks),
            factors_updated=factors_updated,
            inverse_refreshed=inverse_refreshed,
            energy_gradient_norm=energy_gradient_norm,
            update_direction_norm=direction_norm,
            applied_update_norm=applied_update_norm,
            predicted_kl=predicted_kl,
            trust_scale=trust_scale,
            vg_sum=vg_sum,
            damped_quadratic_term=damped_quadratic_term,
            solve_dtype=str(self.policy.solve_dtype),
        )
        return self._record_result(
            applied=True,
            grad_norm=energy_gradient_norm,
            reason=telemetry.reason,
            diagnostics=telemetry,
        )

    def _decompose_kronecker(
        self, A_ema: torch.Tensor, G_ema: torch.Tensor
    ) -> dict[str, Any]:
        """Compute symmetrized eigendecompositions and trace-normalized damping."""

        A_sym = 0.5 * (A_ema + A_ema.transpose(0, 1))
        G_sym = 0.5 * (G_ema + G_ema.transpose(0, 1))
        d_A, Q_A = torch.linalg.eigh(A_sym)
        d_G, Q_G = torch.linalg.eigh(G_sym)
        d_A = d_A.clamp_min(0.0)
        d_G = d_G.clamp_min(0.0)
        tau_A = float(d_A.mean().item()) if d_A.numel() > 0 else 0.0
        tau_G = float(d_G.mean().item()) if d_G.numel() > 0 else 0.0
        tau = tau_G * tau_A
        damping = self.policy.relative_damping * max(tau, self.policy.damping_trace_floor)
        return {
            "d_A": d_A.detach(),
            "Q_A": Q_A.detach(),
            "d_G": d_G.detach(),
            "Q_G": Q_G.detach(),
            "damping": float(damping),
        }

    def _decompose_dense(self, F_ema: torch.Tensor) -> dict[str, Any]:
        """Compute symmetrized eigendecomposition and trace-normalized damping."""

        if F_ema.numel() == 0:
            return {
                "d_F": F_ema.new_empty((0,)),
                "Q_F": F_ema.new_empty((0, 0)),
                "damping": float(self.policy.relative_damping * self.policy.damping_trace_floor),
            }
        F_sym = 0.5 * (F_ema + F_ema.transpose(0, 1))
        d_F, Q_F = torch.linalg.eigh(F_sym)
        d_F = d_F.clamp_min(0.0)
        tau = float(d_F.mean().item())
        damping = self.policy.relative_damping * max(tau, self.policy.damping_trace_floor)
        return {
            "d_F": d_F.detach(),
            "Q_F": Q_F.detach(),
            "damping": float(damping),
        }

    def _skip(
        self,
        *,
        reason: str,
        step: int,
        n_samples: int,
        n_finite: int,
        n_parameters: int,
    ) -> VMCUpdateResult:
        """Record a degenerate-batch skip without mutating EMA or refresh state."""

        n_kron = 0 if self._kronecker_ema is None else len(self._kronecker_ema)
        n_dense = 0 if self._dense_ema is None else len(self._dense_ema)
        telemetry = KFACTelemetry(
            applied=False,
            reason=reason,
            step=step,
            curvature_step=self.curvature_step,
            n_samples=n_samples,
            n_finite_samples=n_finite,
            n_parameters=n_parameters,
            n_kronecker_blocks=n_kron,
            n_dense_blocks=n_dense,
            factors_updated=False,
            inverse_refreshed=False,
            energy_gradient_norm=0.0,
            update_direction_norm=0.0,
            applied_update_norm=0.0,
            predicted_kl=0.0,
            trust_scale=1.0,
            vg_sum=0.0,
            damped_quadratic_term=0.0,
            solve_dtype=str(self.policy.solve_dtype),
        )
        return self._record_result(
            applied=False,
            grad_norm=0.0,
            reason=reason,
            diagnostics=telemetry,
        )

    def _target_device(self) -> torch.device:
        if self.model_parameters.parameters:
            return self.model_parameters.parameters[0].device
        return torch.device("cpu")

    def _validate_binding(self, update_input: StructuredCurvatureUpdateInput) -> None:
        if not update_input.parameter_binding.layout.compare(self.model_parameters.layout)[0]:
            raise ValueError(
                "NativeKFACUpdate step layout disagrees with bound parameter layout"
            )
        bound = self.model_parameters.parameters
        incoming = update_input.parameter_binding.parameters
        if len(bound) != len(incoming) or any(
            left is not right for left, right in zip(bound, incoming, strict=True)
        ):
            raise ValueError(
                "NativeKFACUpdate step parameter binding does not reference bound live parameters"
            )


def _signature_from_blocks(
    kron_blocks: tuple[Any, ...], dense_blocks: tuple[Any, ...]
) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]:
    """Build deterministic structural block signature for continuity checks."""

    entries: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for blk in kron_blocks:
        entries.append(("kronecker", tuple(blk.parameter_ordinals), (int(blk.out_dim), int(blk.in_dim))))
    for blk in dense_blocks:
        entries.append(("dense", tuple(blk.parameter_ordinals), (int(blk.dim),)))
    return tuple(entries)


def _parse_block_signature(
    raw: Any, *, expect_present: bool
) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] | None:
    """Parse and validate a serialized KFAC block signature."""

    if not expect_present:
        if raw is not None:
            raise ValueError("KFAC block_signature must be None when has_curvature_state is False")
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("KFAC block_signature must be a non-empty list when has_curvature_state is True")
    parsed: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("KFAC block_signature entries must be mappings")
        kind = entry.get("kind")
        ordinals = entry.get("ordinals")
        dims = entry.get("dims")
        if kind not in ("kronecker", "dense"):
            raise ValueError(f"KFAC block_signature has invalid kind {kind!r}")
        if not isinstance(ordinals, list) or not ordinals or any(
            type(o) is not int or o < 0 for o in ordinals
        ):
            raise ValueError("KFAC block_signature ordinals must be a non-empty list of non-negative ints")
        if not isinstance(dims, list) or any(type(d) is not int or d < 0 for d in dims):
            raise ValueError("KFAC block_signature dims must be a list of non-negative ints")
        expected_dim_len = 2 if kind == "kronecker" else 1
        if len(dims) != expected_dim_len:
            raise ValueError(f"KFAC block_signature {kind} dims length must be {expected_dim_len}")
        parsed.append((kind, tuple(ordinals), tuple(dims)))
    return tuple(parsed)


def _validate_restored_tensor(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    label: str,
    non_negative: bool = False,
) -> torch.Tensor:
    """Validate shape, dtype, finiteness, and optional non-negativity of a restored tensor."""

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"KFAC binary tensor state entry {label!r} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(
            f"KFAC binary tensor state {label!r} expected shape {shape}, got {tuple(value.shape)}"
        )
    if value.dtype != dtype:
        raise ValueError(
            f"KFAC binary tensor state {label!r} expected dtype {dtype}, got {value.dtype}"
        )
    if value.numel() > 0 and not bool(torch.isfinite(value).all()):
        raise ValueError(f"KFAC binary tensor state {label!r} contains non-finite values")
    if non_negative and value.numel() > 0 and not bool((value >= 0).all()):
        raise ValueError(f"KFAC binary tensor state {label!r} must be non-negative")
    return value.detach().to(device=device, dtype=dtype).clone()


def _validate_plain_sgd(optimizer: Any, *, learning_rate: float) -> None:
    """Require plain momentum-free, decay-free SGD whose ``lr`` matches the policy."""

    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError(
            "NativeKFACUpdate requires a torch.optim.SGD optimizer so the applied step "
            f"is exactly -eta * nu * d; got {type(optimizer).__name__}"
        )
    for index, group in enumerate(optimizer.param_groups):
        for key in ("momentum", "dampening", "weight_decay"):
            value = group.get(key, 0.0)
            if float(value) != 0.0:
                raise ValueError(
                    f"NativeKFACUpdate requires SGD param group {index} to have {key}=0, got {value}"
                )
        for key in ("nesterov", "maximize"):
            if bool(group.get(key, False)):
                raise ValueError(
                    f"NativeKFACUpdate requires SGD param group {index} to have {key}=False"
                )
        group_lr = float(group.get("lr"))
        if group_lr != float(learning_rate):
            raise ValueError(
                f"NativeKFACUpdate SGD param group {index} has lr={group_lr}, "
                f"which disagrees with KFACPolicy.learning_rate={learning_rate}"
            )


__all__ = [
    "KFAC_CONVENTION_VERSION",
    "KFAC_DAMPING_TRACE_FLOOR",
    "KFAC_GRID_FACTOR_KEYS",
    "KFAC_KL_DENOM_FLOOR",
    "KFAC_STATE_VERSION",
    "KFACGridHyperparameters",
    "KFACPolicy",
    "KFACTelemetry",
    "NativeKFACUpdate",
    "build_kfac_update_from_factors",
    "kfac_policy_from_factors",
]
