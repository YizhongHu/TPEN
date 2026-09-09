"""Exact per-tensor block-diagonal natural-gradient VMC updates.

The empirical Fisher is deliberately split at tensor boundaries: for each
trainable tensor ``f``, this module forms the dense block
``F_f = S_f.T @ S_f / N`` from centered per-sample log-amplitude scores.  It
does not approximate a tensor's own block, cache curvature, or communicate
between tensors.  The approximation is solely the omission of cross-tensor
Fisher blocks.

One score-bearing wavefunction evaluation supplies the complete live packet for
one update: :meth:`BlockDiagonalNaturalGradientUpdate.update` consumes that
packet and never re-evaluates the wavefunction after parameters mutate.  This
keeps the scores and energy gradient at one fixed parameter value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tpen.dependencies import require_torch
from tpen.nn import MaterializedParameterScoreRequest
from tpen.training.update import (
    ModelParameterBinding,
    ScoreUpdateInput,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
    serialize_parameter_layout,
)

torch = require_torch(feature="block-diagonal natural-gradient updates")


BLOCK_NG_STATE_VERSION = 1


@dataclass(frozen=True, kw_only=True)
class BlockNGPolicy:
    """Numerical policy for :class:`BlockDiagonalNaturalGradientUpdate`.

    Parameters
    ----------
    damping : float
        Required positive-or-zero diagonal shift for every Fisher block.  It
        has no default because a silent choice would hide conditioning policy.
    learning_rate : float
        SGD learning rate used to apply the preconditioned direction.
    solve_dtype : torch.dtype or str, optional
        Dtype used for score centering, block construction, and the eigensolve.
        ``float64`` is the default; ``float32`` is selectable for a deliberately
        lower-precision run.  A bare dtype name is accepted at this policy's
        configuration seam so Hydra YAML resolves to the same concrete object
        as programmatic construction.
    score_chunk_size : int, optional
        Chunk size forwarded to the materialized-score forward request.
    """

    damping: float
    learning_rate: float
    solve_dtype: Any = torch.float64
    score_chunk_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "damping", float(self.damping))
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        if isinstance(self.solve_dtype, str):
            dtype_name = self.solve_dtype.removeprefix("torch.")
            resolved = getattr(torch, dtype_name, None)
            if not isinstance(resolved, torch.dtype):
                raise ValueError(
                    f"BlockNGPolicy.solve_dtype {self.solve_dtype!r} is not a torch dtype name"
                )
            object.__setattr__(self, "solve_dtype", resolved)
        self.validate()

    def validate(self) -> "BlockNGPolicy":
        """Validate numerical and score-materialization settings."""

        for name in ("damping", "learning_rate"):
            value = getattr(self, name)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"BlockNGPolicy.{name} must be finite")
        if self.damping < 0.0:
            raise ValueError("BlockNGPolicy.damping must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("BlockNGPolicy.learning_rate must be positive")
        if self.solve_dtype not in (torch.float32, torch.float64):
            raise ValueError("BlockNGPolicy.solve_dtype must be torch.float32 or torch.float64")
        if self.score_chunk_size is not None and (
            type(self.score_chunk_size) is not int or self.score_chunk_size <= 0
        ):
            raise ValueError("BlockNGPolicy.score_chunk_size must be a positive integer or None")
        return self

    def fingerprint(self) -> dict[str, Any]:
        """Return the JSON-safe numerical policy identity for resume checks."""

        return {
            "damping": self.damping,
            "learning_rate": self.learning_rate,
            "solve_dtype": str(self.solve_dtype),
            "score_chunk_size": self.score_chunk_size,
        }


@dataclass(frozen=True, kw_only=True)
class BlockNGTelemetry:
    """Observable result of one block-NG update.

    ``solve_dtype`` is observed from the configured linear-algebra path and is
    retained so experiment records distinguish float64 and float32 solves.
    """

    applied: bool
    step: int
    n_samples: int
    n_blocks: int
    energy_gradient_norm: float
    update_direction_norm: float
    solve_dtype: str

    def as_metrics(self) -> dict[str, float | int | str | bool]:
        """Return bounded telemetry under method-owned metric names."""

        return {
            "block_ng_applied": self.applied,
            "block_ng_step": self.step,
            "block_ng_n_samples": self.n_samples,
            "block_ng_n_blocks": self.n_blocks,
            "block_ng_energy_gradient_norm": self.energy_gradient_norm,
            "block_ng_update_direction_norm": self.update_direction_norm,
            "block_ng_solve_dtype": self.solve_dtype,
        }


class BlockDiagonalNaturalGradientUpdate(VMCUpdateMethod[ScoreUpdateInput]):
    """Apply an exact dense Fisher solve independently to every tensor.

    Parameters
    ----------
    optimizer : torch.optim.SGD
        Plain SGD optimizer used only to apply the computed direction.
    model_parameters : ModelParameterBinding
        Live parameter references and their authoritative tensor layout.
    policy : BlockNGPolicy
        Required damping, learning-rate, and solve-dtype policy.

    Notes
    -----
    Fisher blocks are rebuilt from the incoming score packet on every call.
    This method intentionally has no curvature EMA, refresh cadence, or
    inverse/eigenvector cache.  The trainer performs exactly one score request
    for a step; this method consumes its packet without evaluating the model a
    second time, before or after applying the parameter update.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        model_parameters: ModelParameterBinding,
        policy: BlockNGPolicy,
    ) -> None:
        if not isinstance(policy, BlockNGPolicy):
            raise TypeError("BlockDiagonalNaturalGradientUpdate.policy must be BlockNGPolicy")
        policy.validate()
        _validate_plain_sgd(optimizer, learning_rate=policy.learning_rate)
        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError(
                "BlockDiagonalNaturalGradientUpdate.model_parameters must be "
                "a ModelParameterBinding"
            )
        self.optimizer = optimizer
        self.model_parameters = model_parameters
        self.policy = policy
        self.completed_updates = 0
        self.last_telemetry: BlockNGTelemetry | None = None

    def forward_request(self) -> MaterializedParameterScoreRequest:
        """Request the raw per-sample score blocks consumed by this method."""

        return MaterializedParameterScoreRequest(chunk_size=self.policy.score_chunk_size)

    def update_state(self) -> VMCUpdateState:
        """Return the optimizer and live binding owned by this update method."""

        return VMCUpdateState(optimizer=self.optimizer, model_parameters=self.model_parameters)

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Adopt restored live parameters after checkpoint load."""

        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError("Block-NG model parameters must be a ModelParameterBinding")
        self.model_parameters = model_parameters

    def method_state_dict(self) -> dict[str, Any]:
        """Return persistent state needed to continue the same block-NG method.

        The optimizer owns its own tensor state in ``optimizer.pt``.  This
        envelope instead records the method's update count and the exact
        layout and numerical policy under which its directions are defined.
        A resume that silently changes any of these facts is not parity.
        """

        return {
            "version": BLOCK_NG_STATE_VERSION,
            "parameter_layout": serialize_parameter_layout(self.model_parameters.layout),
            "policy": self.policy.fingerprint(),
            "completed_updates": self.completed_updates,
        }

    def load_method_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore persistent method state, rejecting a different method identity."""

        if not isinstance(state, Mapping):
            raise TypeError("Block-NG method state must be a mapping")
        if state.get("version") != BLOCK_NG_STATE_VERSION:
            raise ValueError(
                f"unsupported Block-NG state version {state.get('version')!r}, "
                f"expected {BLOCK_NG_STATE_VERSION!r}"
            )
        if state.get("parameter_layout") != serialize_parameter_layout(
            self.model_parameters.layout
        ):
            raise ValueError("Block-NG checkpoint parameter layout does not match the live model")
        if state.get("policy") != self.policy.fingerprint():
            raise ValueError("Block-NG checkpoint policy does not match the live method")
        completed_updates = state.get("completed_updates")
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("Block-NG completed_updates must be a non-negative integer")
        self.completed_updates = completed_updates

    def update(self, update_input: ScoreUpdateInput) -> VMCUpdateResult:
        """Build current Fisher blocks, solve them, and apply one SGD step."""

        if not isinstance(update_input, ScoreUpdateInput):
            raise TypeError("BlockDiagonalNaturalGradientUpdate requires ScoreUpdateInput")
        update_input.validate()
        self._validate_binding(update_input)
        if update_input.wavefunction.phase is not None:
            raise ValueError("BlockDiagonalNaturalGradientUpdate supports real wavefunctions only")

        local_energy = update_input.local_energy.reshape(-1)
        n_samples = int(local_energy.numel())
        if n_samples < 2:
            raise ValueError("Block-NG requires at least two samples to form centered scores")
        if not bool(torch.isfinite(local_energy).all()):
            raise RuntimeError("Block-NG refuses non-finite local-energy samples")

        directions, gradients = build_block_ng_directions(
            update_input.parameter_scores.blocks,
            local_energy,
            damping=self.policy.damping,
            solve_dtype=self.policy.solve_dtype,
        )
        if not all(bool(torch.isfinite(direction).all()) for direction in directions):
            raise RuntimeError("Block-NG produced a non-finite update direction")

        for parameter, direction in zip(self.model_parameters.parameters, directions, strict=True):
            parameter.grad = direction.reshape_as(parameter).to(
                dtype=parameter.dtype, device=parameter.device
            )
        self.optimizer.step()
        self.completed_updates += 1

        flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
        flat_direction = torch.cat([direction.reshape(-1) for direction in directions])
        grad_norm = float(torch.linalg.vector_norm(flat_gradient).item())
        self.last_telemetry = BlockNGTelemetry(
            applied=True,
            step=update_input.step,
            n_samples=n_samples,
            n_blocks=len(directions),
            energy_gradient_norm=grad_norm,
            update_direction_norm=float(torch.linalg.vector_norm(flat_direction).item()),
            solve_dtype=str(self.policy.solve_dtype),
        )
        return VMCUpdateResult(applied=True, grad_norm=grad_norm)

    def _validate_binding(self, update_input: ScoreUpdateInput) -> None:
        """Require scores and live references to match the bound tensor layout."""

        if not update_input.parameter_scores.layout.compare(self.model_parameters.layout)[0]:
            raise ValueError("Block-NG scores do not match the bound parameter layout")
        if any(
            left is not right
            for left, right in zip(
                self.model_parameters.parameters,
                update_input.parameter_binding.parameters,
                strict=True,
            )
        ):
            raise ValueError("Block-NG score binding does not reference the bound live parameters")


def build_block_ng_directions(
    score_blocks: tuple[Any, ...],
    local_energy: Any,
    *,
    damping: float,
    solve_dtype: Any = torch.float64,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return one natural-gradient direction and energy gradient per tensor.

    Parameters
    ----------
    score_blocks : tuple of torch.Tensor
        Raw per-sample score tensors, each shaped ``sample_shape + parameter.shape``.
    local_energy : torch.Tensor
        One finite local energy for each flattened sample.
    damping : float
        Diagonal Fisher shift.
    solve_dtype : torch.dtype, optional
        Dtype for all block algebra.

    Returns
    -------
    tuple
        ``(directions, gradients)`` in parameter-layout order.  Each direction
        is flat within its tensor block.
    """

    policy = BlockNGPolicy(damping=damping, learning_rate=1.0, solve_dtype=solve_dtype)
    del policy
    if not isinstance(local_energy, torch.Tensor):
        raise TypeError("build_block_ng_directions.local_energy must be a torch.Tensor")
    energies = local_energy.reshape(-1).to(dtype=solve_dtype)
    n_samples = int(energies.numel())
    if n_samples < 2:
        raise ValueError("build_block_ng_directions requires at least two samples")

    directions = []
    gradients = []
    for block in score_blocks:
        if not isinstance(block, torch.Tensor):
            raise TypeError("build_block_ng_directions.score_blocks must contain tensors")
        rows = block.reshape(n_samples, -1).to(dtype=solve_dtype)
        if not bool(torch.isfinite(rows).all()):
            raise RuntimeError("Block-NG refuses non-finite score samples")
        centered_scores = rows - rows.mean(dim=0, keepdim=True)
        # Algebraically Sc.T @ 1 == 0, but centering is numerically
        # load-bearing in float32: the frozen fixture's maximum relative error
        # is 0.00893 with this subtraction and 0.01021 without it.
        centered_energy = energies - energies.mean()
        gradient = (2.0 / n_samples) * (centered_scores.transpose(0, 1) @ centered_energy)
        fisher = (centered_scores.transpose(0, 1) @ centered_scores) / n_samples
        eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
        inverse = (eigenvalues + damping).reciprocal()
        direction = eigenvectors @ (inverse * (eigenvectors.transpose(0, 1) @ gradient))
        directions.append(direction)
        gradients.append(gradient)
    return tuple(directions), tuple(gradients)


def _validate_plain_sgd(optimizer: Any, *, learning_rate: float) -> None:
    """Require SGD settings that implement exactly ``theta -= lr * grad``."""

    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError("Block-NG requires torch.optim.SGD to apply its direction exactly")
    for index, group in enumerate(optimizer.param_groups):
        for key in ("momentum", "dampening", "weight_decay"):
            if float(group.get(key, 0.0)) != 0.0:
                raise ValueError(f"Block-NG SGD param group {index} must have {key}=0")
        for key in ("nesterov", "maximize"):
            if bool(group.get(key, False)):
                raise ValueError(f"Block-NG SGD param group {index} must have {key}=False")
        if float(group.get("lr")) != learning_rate:
            raise ValueError("Block-NG SGD learning rate must match BlockNGPolicy.learning_rate")


__all__ = [
    "BLOCK_NG_STATE_VERSION",
    "BlockDiagonalNaturalGradientUpdate",
    "BlockNGPolicy",
    "BlockNGTelemetry",
    "build_block_ng_directions",
]
