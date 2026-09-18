"""SPRING: a projected-history recurrence over the landed minSR geometry.

SPRING consumes the same centered, ``1 / sqrt(N)``-scaled score geometry and
the same sample-space solver as :mod:`tpen.training.sr`.  Its only additional
state is a parameter-space projected history.  For ``A`` the frozen design
matrix, ``epsilon`` the scaled energy residual, and ``z`` the previous
history, one step is

.. math::

    z_t = A^T (A A^T + \\lambda I)^{-1}
        (\\epsilon_t - A \\mu z_{t-1}) + \\mu z_{t-1}.

The persisted history is the **unscaled** ``z_t`` produced by this recurrence:
it is kept separately from the applied displacement, before the learning-rate
factor and before any trust-cap rescale.  Because it is a parameter-space
vector, it is invariant to per-step row masking and is one canonical vector
that a later DDP extension can replicate.  This module creates no process
group and does not implement DDP runtime behavior.

SPRING is not a closure optimizer.  It consumes one score-bearing forward
packet and performs one solve per update; it never re-evaluates the model and
does not participate in the objective-reevaluation capability seam.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from tpen.dependencies import require_torch
from tpen.nn.forward import MaterializedParameterScoreRequest
from tpen.training.qgt import QGTOperator, SolveDiagnostics, solve_sample_space
from tpen.training.score_geometry import (
    ScoreConventions,
    build_energy_residual,
    build_score_geometry_from_rows,
    flatten_parameter_score_blocks,
    layout_convention_fingerprint,
    unflatten_to_layout,
)
from tpen.training.statistics import IdentityStatisticsReducer, StatisticsReducer
from tpen.training.sr import SRPolicy
from tpen.training.update import (
    ModelParameterBinding,
    ScoreUpdateInput,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
)
from tpen.training.vmc import (
    DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    resolve_nonfinite_local_energy_policy,
)

torch = require_torch(feature="VMC SPRING projected-history update")


# Bump when the persisted state envelope changes shape or meaning.
SPRING_STATE_VERSION = "spring-state-1"


@dataclass(frozen=True, kw_only=True)
class SPRINGPolicy:
    """SR/minSR policy plus the required projected-history decay.

    Parameters
    ----------
    base : SRPolicy
        Composed SR policy.  SPRING takes its damping, learning rate, trust
        cap, score chunking, and numerical conventions from this object; it
        does not duplicate those fields.
    history_decay : float
        ``mu`` in the projected-history recurrence.  It is required so a
        policy cannot silently choose a history schedule.
    """

    base: SRPolicy
    history_decay: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "history_decay", float(self.history_decay))
        self.validate()

    def validate(self) -> Self:
        """Validate the composed base policy and decay range."""

        if not isinstance(self.base, SRPolicy):
            raise TypeError("SPRINGPolicy.base must be an SRPolicy")
        self.base.validate()
        if not _is_finite(self.history_decay) or not 0.0 <= self.history_decay < 1.0:
            raise ValueError(
                "SPRINGPolicy.history_decay must be finite and satisfy "
                "0.0 <= history_decay < 1.0"
            )
        return self

    def fingerprint(self) -> dict[str, Any]:
        """Return the complete JSON-safe policy identity."""

        return {
            "base": self.base.fingerprint(),
            "history_decay": self.history_decay,
        }


@dataclass(frozen=True, kw_only=True)
class SPRINGTelemetry:
    """Observable record of one SPRING update attempt.

    The fields mirror SR's telemetry and add the history norm, decay in force,
    and whether the canonical unscaled history advanced on this step.
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
    history_norm: float
    history_decay: float
    history_advanced: bool
    diagnostics: SolveDiagnostics | None = None

    def as_metrics(self, *, prefix: str = "spring") -> dict[str, Any]:
        """Return JSON-safe telemetry keys for a training metrics record."""

        metrics: dict[str, Any] = {
            f"{prefix}_applied": bool(self.applied),
            f"{prefix}_reason": self.reason,
            f"{prefix}_step": int(self.step),
            f"{prefix}_samples": int(self.n_samples),
            f"{prefix}_finite_samples": int(self.n_finite_samples),
            f"{prefix}_parameters": int(self.n_parameters),
            f"{prefix}_energy_gradient_norm": float(self.energy_gradient_norm),
            f"{prefix}_update_direction_norm": float(self.update_direction_norm),
            f"{prefix}_applied_update_norm": float(self.applied_update_norm),
            f"{prefix}_trust_scale": float(self.trust_scale),
            f"{prefix}_history_norm": float(self.history_norm),
            f"{prefix}_history_decay": float(self.history_decay),
            f"{prefix}_history_advanced": bool(self.history_advanced),
        }
        if self.diagnostics is not None:
            metrics.update(self.diagnostics.as_metrics(prefix=f"{prefix}_qgt"))
        return metrics


class SPRINGUpdate(VMCUpdateMethod[ScoreUpdateInput]):
    """Apply the SPRING projected-history recurrence over landed minSR.

    Parameters
    ----------
    optimizer : torch.optim.SGD
        Plain SGD that applies the direction exactly.
    model_parameters : ModelParameterBinding
        Live parameter references and their authoritative layout.
    policy : SPRINGPolicy
        Composed SR policy and required history decay.
    conventions : ScoreConventions, optional
        Frozen score conventions shared with SR.
    reducer : StatisticsReducer, optional
        Reducer for every cross-sample sum.  No process group is created.
    nonfinite_local_energy_policy : {"fail", "mask"}, optional
        Same local-energy policy as SR.

    Notes
    -----
    The history is a single parameter-space vector.  Row masking changes the
    current geometry and residual but never changes the representation of the
    history, which is why the state is canonical and DDP-replicable.  The
    optimizer payload is not part of :meth:`method_state_dict`; the trainer's
    optimizer checkpoint is its sole owner.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        model_parameters: ModelParameterBinding,
        policy: SPRINGPolicy,
        conventions: ScoreConventions | None = None,
        reducer: StatisticsReducer | None = None,
        nonfinite_local_energy_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
    ) -> None:
        self.nonfinite_local_energy_policy = resolve_nonfinite_local_energy_policy(
            nonfinite_local_energy_policy
        )
        if not isinstance(policy, SPRINGPolicy):
            raise TypeError("SPRINGUpdate.policy must be an SPRINGPolicy")
        policy.validate()
        _validate_plain_sgd(optimizer, learning_rate=policy.base.learning_rate)
        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError("SPRINGUpdate.model_parameters must be a ModelParameterBinding")
        resolved_conventions = ScoreConventions() if conventions is None else conventions
        if not isinstance(resolved_conventions, ScoreConventions):
            raise TypeError("SPRINGUpdate.conventions must be a ScoreConventions")
        resolved_reducer = IdentityStatisticsReducer() if reducer is None else reducer
        if not isinstance(resolved_reducer, StatisticsReducer):
            raise TypeError("SPRINGUpdate.reducer must be a StatisticsReducer")

        self.optimizer = optimizer
        self.model_parameters = model_parameters
        self.policy = policy
        self.conventions = resolved_conventions
        self.reducer = resolved_reducer
        self.completed_updates = 0
        self.history = self._zero_history()
        self.last_telemetry: SPRINGTelemetry | None = None

    def forward_request(self) -> MaterializedParameterScoreRequest:
        """Request the one raw score packet consumed by a SPRING step."""

        return MaterializedParameterScoreRequest(chunk_size=self.policy.base.score_chunk_size)

    def update_state(self) -> VMCUpdateState:
        """Return the one optimizer and live parameter binding this method owns."""

        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Adopt restored live parameter references after checkpoint load."""

        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError("SPRINGUpdate.model_parameters must be a ModelParameterBinding")
        if not self.model_parameters.layout.compare(model_parameters.layout)[0]:
            raise ValueError("SPRING model parameter layout changed during rebind")
        self.model_parameters = model_parameters

    def method_state_dict(self) -> Mapping[str, Any]:
        """Return the JSON-safe versioned SPRING state envelope.

        The optimizer payload is deliberately absent: the trainer persists the
        optimizer separately.  This envelope owns the method counter and the
        unscaled, parameter-space history vector.
        """

        return {
            "version": SPRING_STATE_VERSION,
            "fingerprint": layout_convention_fingerprint(
                self.model_parameters.layout,
                self.conventions,
            ),
            "policy": self.policy.fingerprint(),
            "completed_updates": self.completed_updates,
            "history": [float(value) for value in self.history.detach().cpu().tolist()],
        }

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state, rejecting version, identity, and history-shape drift."""

        if not isinstance(state, Mapping):
            raise TypeError("SPRINGUpdate state must be a mapping")
        version = state.get("version")
        if version != SPRING_STATE_VERSION:
            raise ValueError(
                f"unsupported SPRING state version {version!r}, expected {SPRING_STATE_VERSION!r}"
            )
        expected = layout_convention_fingerprint(self.model_parameters.layout, self.conventions)
        recorded = state.get("fingerprint")
        if not isinstance(recorded, Mapping) or recorded.get("digest") != expected["digest"]:
            raise ValueError(
                "SPRING checkpoint layout/convention fingerprint does not match the live model"
            )
        if state.get("policy") != self.policy.fingerprint():
            raise ValueError("SPRING checkpoint policy does not match the live method")
        completed = state.get("completed_updates")
        if type(completed) is not int or completed < 0:
            raise ValueError("SPRING state completed_updates must be a non-negative integer")

        recorded_history = state.get("history")
        if not isinstance(recorded_history, (list, tuple)):
            raise TypeError("SPRING state history must be a flat JSON list")
        if len(recorded_history) != self.model_parameters.layout.total_numel:
            raise ValueError(
                "SPRING state history length must equal the live layout total_numel "
                f"({self.model_parameters.layout.total_numel}), got {len(recorded_history)}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not _is_finite(float(value))
            for value in recorded_history
        ):
            raise ValueError("SPRING state history must contain only finite numeric values")
        self.history = torch.tensor(
            [float(value) for value in recorded_history],
            dtype=self._history_dtype(),
            device=self._history_device(),
        )
        self.completed_updates = completed

    def update(self, update_input: ScoreUpdateInput) -> VMCUpdateResult:
        """Apply one SPRING step without a second score gather or solve."""

        if not isinstance(update_input, ScoreUpdateInput):
            raise TypeError("SPRINGUpdate requires ScoreUpdateInput")
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

        energy_finite = torch.isfinite(local_energy)
        if self.nonfinite_local_energy_policy == "fail" and not bool(energy_finite.all()):
            n_bad = int((~energy_finite).sum().item())
            raise RuntimeError(
                f"SPRING refused a step: {n_bad} of {int(energy_finite.numel())} local-energy "
                "samples are non-finite and the active policy is 'fail'. Masking them would "
                "drop a systematically selected subsample, biasing the estimator by an "
                "uncharacterised amount"
            )
        finite_mask = energy_finite & torch.isfinite(rows).all(dim=1)
        n_finite = int(finite_mask.sum().item())

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
                "cannot compute a SPRING update: no finite local-energy sample remains "
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

        previous = self.history.to(dtype=geometry.dtype, device=geometry.device)
        decayed = previous * self.policy.history_decay
        corrected_residual = residual - operator.jv(decayed)
        direction, diagnostics = solve_sample_space(
            operator,
            corrected_residual,
            damping=self.policy.base.damping,
            rank_cutoff=self.policy.base.rank_cutoff,
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
            )

        projected_history = direction + decayed
        if not torch.isfinite(projected_history).all():
            return self._skip(
                reason="nonfinite_projected_history",
                step=update_input.step,
                n_samples=n_samples,
                n_finite=n_finite,
                n_parameters=n_parameters,
                energy_gradient_norm=energy_gradient_norm,
                diagnostics=diagnostics,
            )
        history_norm = float(torch.linalg.vector_norm(projected_history).item())
        if not _is_finite(history_norm):
            return self._skip(
                reason="nonfinite_projected_history",
                step=update_input.step,
                n_samples=n_samples,
                n_finite=n_finite,
                n_parameters=n_parameters,
                energy_gradient_norm=energy_gradient_norm,
                diagnostics=diagnostics,
            )

        direction_norm = history_norm
        trust_scale = self._trust_scale(direction_norm)
        applied_norm = self.policy.base.learning_rate * direction_norm * trust_scale

        self._apply(projected_history * trust_scale)
        self.history = projected_history.detach().clone()
        self.completed_updates += 1
        self.last_telemetry = SPRINGTelemetry(
            applied=True,
            reason="applied",
            step=update_input.step,
            n_samples=n_samples,
            n_finite_samples=n_finite,
            n_parameters=n_parameters,
            energy_gradient_norm=energy_gradient_norm,
            update_direction_norm=direction_norm,
            applied_update_norm=applied_norm,
            trust_scale=trust_scale,
            history_norm=history_norm,
            history_decay=self.policy.history_decay,
            history_advanced=True,
            diagnostics=diagnostics,
        )
        return VMCUpdateResult(applied=True, grad_norm=energy_gradient_norm)

    def _trust_scale(self, direction_norm: float) -> float:
        """Return the base policy's factor capping applied displacement."""

        cap = self.policy.base.max_update_norm
        if cap is None or direction_norm == 0.0:
            return 1.0
        proposed = self.policy.base.learning_rate * direction_norm
        return 1.0 if proposed <= cap else cap / proposed

    def _apply(self, direction: Any) -> None:
        """Write the applied direction into ``.grad`` and step plain SGD."""

        blocks = unflatten_to_layout(direction, layout=self.model_parameters.layout)
        for parameter, block in zip(self.model_parameters.parameters, blocks, strict=True):
            parameter.grad = block.detach().to(dtype=parameter.dtype, device=parameter.device)
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
    ) -> VMCUpdateResult:
        """Record a non-applied step without changing canonical history."""

        self.last_telemetry = SPRINGTelemetry(
            applied=False,
            reason=reason,
            step=step,
            n_samples=n_samples,
            n_finite_samples=n_finite,
            n_parameters=n_parameters,
            energy_gradient_norm=energy_gradient_norm,
            update_direction_norm=0.0,
            applied_update_norm=0.0,
            trust_scale=1.0,
            history_norm=self._history_norm(),
            history_decay=self.policy.history_decay,
            history_advanced=False,
            diagnostics=diagnostics,
        )
        return VMCUpdateResult(applied=False, grad_norm=energy_gradient_norm)

    def _zero_history(self) -> Any:
        """Build the canonical zero parameter-space history vector."""

        return torch.zeros(
            self.model_parameters.layout.total_numel,
            dtype=self._history_dtype(),
            device=self._history_device(),
        )

    def _history_dtype(self) -> Any:
        """Return the dtype used by SR's geometry for persistent history."""

        if self.conventions.solve_dtype is not None:
            return self.conventions.solve_dtype
        if self.model_parameters.parameters:
            return self.model_parameters.parameters[0].dtype
        return torch.float64

    def _history_device(self) -> Any:
        """Return the live model device for the canonical history vector."""

        if self.model_parameters.parameters:
            return self.model_parameters.parameters[0].device
        return torch.device("cpu")

    def _history_norm(self) -> float:
        """Return the finite scalar norm used by skip telemetry."""

        return float(torch.linalg.vector_norm(self.history).item())

    def _validate_real_wavefunction(self, update_input: ScoreUpdateInput) -> None:
        """Reject a complex wavefunction whose phase this geometry does not form."""

        if update_input.wavefunction.phase is not None:
            raise ValueError(
                "SPRINGUpdate supports real wavefunctions only; the step carries a complex "
                "phase whose QGT this method does not form"
            )

    def _validate_binding(self, update_input: ScoreUpdateInput) -> None:
        """Require scores to describe this method's live parameter domain."""

        if not update_input.parameter_scores.layout.compare(self.model_parameters.layout)[0]:
            raise ValueError("SPRING scores do not match the bound parameter layout")
        bound = self.model_parameters.parameters
        step_parameters = update_input.parameter_binding.parameters
        if len(bound) != len(step_parameters) or any(
            left is not right for left, right in zip(bound, step_parameters, strict=True)
        ):
            raise ValueError("SPRING score binding does not reference the bound live parameters")


def _validate_plain_sgd(optimizer: Any, *, learning_rate: float) -> None:
    """Require a plain SGD whose update is exactly ``theta -= lr * grad``.

    This is intentionally a private third copy following the landed
    ``block_ng.py`` precedent.  Consolidation belongs to a later slice so this
    implementation does not alter SR or block-NG behavior.
    """

    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError("SPRINGUpdate requires torch.optim.SGD to apply its direction exactly")
    for index, group in enumerate(optimizer.param_groups):
        for key in ("momentum", "dampening", "weight_decay"):
            if float(group.get(key, 0.0)) != 0.0:
                raise ValueError(f"SPRING SGD param group {index} must have {key}=0")
        for key in ("nesterov", "maximize"):
            if bool(group.get(key, False)):
                raise ValueError(f"SPRING SGD param group {index} must have {key}=False")
        if float(group.get("lr")) != float(learning_rate):
            raise ValueError("SPRING SGD learning rate must match SPRINGPolicy.base.learning_rate")


def _is_finite(value: float) -> bool:
    """Return whether a Python float is finite."""

    return value == value and value not in (float("inf"), float("-inf"))


__all__ = [
    "SPRING_STATE_VERSION",
    "SPRINGPolicy",
    "SPRINGTelemetry",
    "SPRINGUpdate",
]
