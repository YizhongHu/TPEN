"""Physics-free structured curvature capture for native KFAC forward requests.

This module owns the opt-in single-forward curvature capture session and typed
factor block payloads consumed by :class:`~tpen.nn.forward.StructuredCurvatureForwardRequest`.
It imports zero symbols from :mod:`tpen.physics` and performs zero
``torch.func``/``vmap`` or ``is_grads_batched=True`` transforms, keeping
:class:`~tpen.nn.activation.SoftplusBeta` trunks safe by construction.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod
from typing import Any, Literal, Self

from tpen.data.batch import (
    ElectronBatch,
    ParameterBinding,
    WavefunctionOutput,
    WavefunctionPacket,
)
from tpen.data.equivariant_state import compare_tensor_blocks
from tpen.data.indices import no_repeated_particle_mask
from tpen.data.permutation import Permutation
from tpen.dependencies import require_torch, require_torch_nn

torch = require_torch(feature="structured curvature capture")
nn = require_torch_nn(feature="structured curvature capture")

_ACTIVE_CAPTURE_SESSION: ContextVar["_StructuredCurvatureCaptureSession | None"] = ContextVar(
    "_ACTIVE_CAPTURE_SESSION",
    default=None,
)


def active_structured_curvature_session() -> "_StructuredCurvatureCaptureSession | None":
    """Return the active structured-curvature capture session, or ``None``."""

    return _ACTIVE_CAPTURE_SESSION.get()


def softplus_derivative(raw: torch.Tensor) -> torch.Tensor:
    """Return PyTorch's exact ``F.softplus`` (beta=1, threshold=20) derivative."""

    return torch.where(raw > 20.0, torch.ones_like(raw), torch.sigmoid(raw))


def _validate_solve_dtype(solve_dtype: Any, *, field_name: str) -> torch.dtype:
    """Validate that a curvature solve dtype is float32 or float64."""

    if not isinstance(solve_dtype, torch.dtype) or solve_dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"{field_name} must be torch.float32 or torch.float64, got {solve_dtype!r}"
        )
    return solve_dtype


def _validate_positive_int(value: Any, *, field_name: str) -> int:
    """Validate a positive Python integer, explicitly refusing booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return value


def _validate_ordinals(ordinals: tuple[int, ...], *, field_name: str) -> tuple[int, ...]:
    """Validate non-empty unique non-negative parameter ordinals."""

    if not ordinals:
        raise ValueError(f"{field_name} must be non-empty")
    for ordinal in ordinals:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError(f"{field_name} entries must be non-negative integers, got {ordinal!r}")
    if len(set(ordinals)) != len(ordinals):
        raise ValueError(f"{field_name} entries must be unique, got {ordinals!r}")
    return ordinals


def _require_finite(tensor: torch.Tensor, *, label: str) -> None:
    """Fail loudly on any non-finite value on pinned finite rows."""

    if tensor.numel() > 0 and not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(
            f"structured curvature capture encountered non-finite values in {label} on pinned finite rows"
        )


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Enforce exact symmetry on a square factor matrix."""

    return 0.5 * (matrix + matrix.transpose(-1, -2))


@dataclass(frozen=True, eq=False, kw_only=True)
class KroneckerFactorBlock:
    """Detached two-factor uncentered KFAC-expand Kronecker block ``G (x) A``.

    Parameters
    ----------
    parameter_ordinals : tuple of int
        One ordinal ``(w_ord,)`` for weight-only layers or two ordinals
        ``(w_ord, b_ord)`` when a bias column is appended last in ``A``.
    has_bias : bool
        Whether the final column of ``A`` is the homogeneous bias coordinate.
    out_dim : int
        Row dimension ``d_G`` of the matricized block.
    in_dim : int
        Column dimension ``d_A`` of the matricized block (including bias if any).
    A : torch.Tensor
        Detached ``[d_A, d_A]`` input second-moment factor in ``solve_dtype``.
    G : torch.Tensor
        Detached ``[d_G, d_G]`` cotangent second-moment factor in ``solve_dtype``.
    active_locations : int
        Number of active shared locations ``L`` per sample (``0`` when support
        is empty, e.g., 3-body paths at ``n = 2``).
    """

    parameter_ordinals: tuple[int, ...]
    has_bias: bool
    out_dim: int
    in_dim: int
    A: torch.Tensor
    G: torch.Tensor
    active_locations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_ordinals", tuple(self.parameter_ordinals))
        self.validate()

    def validate(self) -> Self:
        """Validate dimensions, ordinals, dtypes, device agreement, and detachment."""

        _validate_ordinals(self.parameter_ordinals, field_name="KroneckerFactorBlock.parameter_ordinals")
        if type(self.has_bias) is not bool:
            raise TypeError("KroneckerFactorBlock.has_bias must be a bool")
        expected_len = 2 if self.has_bias else 1
        if len(self.parameter_ordinals) != expected_len:
            raise ValueError(
                f"KroneckerFactorBlock with has_bias={self.has_bias} requires "
                f"{expected_len} parameter ordinal(s), got {len(self.parameter_ordinals)}"
            )
        _validate_positive_int(self.out_dim, field_name="KroneckerFactorBlock.out_dim")
        _validate_positive_int(self.in_dim, field_name="KroneckerFactorBlock.in_dim")
        if isinstance(self.active_locations, bool) or not isinstance(self.active_locations, int) or self.active_locations < 0:
            raise ValueError("KroneckerFactorBlock.active_locations must be a non-negative integer")
        if not isinstance(self.A, torch.Tensor) or not isinstance(self.G, torch.Tensor):
            raise TypeError("KroneckerFactorBlock.A and G must be torch.Tensor values")
        if tuple(self.A.shape) != (self.in_dim, self.in_dim):
            raise ValueError(
                f"KroneckerFactorBlock.A expected shape {(self.in_dim, self.in_dim)}, got {tuple(self.A.shape)}"
            )
        if tuple(self.G.shape) != (self.out_dim, self.out_dim):
            raise ValueError(
                f"KroneckerFactorBlock.G expected shape {(self.out_dim, self.out_dim)}, got {tuple(self.G.shape)}"
            )
        _validate_solve_dtype(self.A.dtype, field_name="KroneckerFactorBlock.A.dtype")
        if self.G.dtype != self.A.dtype:
            raise ValueError("KroneckerFactorBlock A and G must share one solve_dtype")
        if self.G.device != self.A.device:
            raise ValueError("KroneckerFactorBlock A and G must share one device")
        return self

    def detach(self) -> Self:
        """Return a detached copy of both Kronecker factors."""

        return type(self)(
            parameter_ordinals=self.parameter_ordinals,
            has_bias=self.has_bias,
            out_dim=self.out_dim,
            in_dim=self.in_dim,
            A=self.A.detach(),
            G=self.G.detach(),
            active_locations=self.active_locations,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move both Kronecker factors to a target device or solve dtype."""

        return type(self)(
            parameter_ordinals=self.parameter_ordinals,
            has_bias=self.has_bias,
            out_dim=self.out_dim,
            in_dim=self.in_dim,
            A=self.A.to(device=device, dtype=dtype),
            G=self.G.to(device=device, dtype=dtype),
            active_locations=self.active_locations,
        )

    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare metadata exactly and Kronecker factors numerically."""

        if (
            type(self) is not type(other)
            or self.parameter_ordinals != other.parameter_ordinals
            or self.has_bias != other.has_bias
            or self.out_dim != other.out_dim
            or self.in_dim != other.in_dim
            or self.active_locations != other.active_locations
        ):
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks((self.A, self.G), (other.A, other.G), atol=atol, rtol=rtol)

    def __getstate__(self) -> dict[str, Any]:
        """Refuse serialization while either factor carries an autograd graph."""

        if self.A.requires_grad or self.G.requires_grad:
            raise RuntimeError("graph-bearing KroneckerFactorBlock cannot be serialized")
        return self.__dict__


@dataclass(frozen=True, eq=False, kw_only=True)
class DenseFactorBlock:
    """Detached centered empirical score-covariance block ``F``.

    Parameters
    ----------
    parameter_ordinals : tuple of int
        Ordered layout ordinals flattened into this dense Fisher block.
    dim : int
        Total scalar dimension covered by ``parameter_ordinals``.
    F : torch.Tensor
        Detached ``[dim, dim]`` centered empirical score covariance in
        ``solve_dtype``.
    """

    parameter_ordinals: tuple[int, ...]
    dim: int
    F: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_ordinals", tuple(self.parameter_ordinals))
        self.validate()

    def validate(self) -> Self:
        """Validate ordinals, matrix shape, dtype, and tensor type."""

        _validate_ordinals(self.parameter_ordinals, field_name="DenseFactorBlock.parameter_ordinals")
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 0:
            raise ValueError("DenseFactorBlock.dim must be a non-negative integer")
        if not isinstance(self.F, torch.Tensor):
            raise TypeError("DenseFactorBlock.F must be a torch.Tensor")
        if tuple(self.F.shape) != (self.dim, self.dim):
            raise ValueError(
                f"DenseFactorBlock.F expected shape {(self.dim, self.dim)}, got {tuple(self.F.shape)}"
            )
        _validate_solve_dtype(self.F.dtype, field_name="DenseFactorBlock.F.dtype")
        return self

    def detach(self) -> Self:
        """Return a detached copy of this dense block."""

        return type(self)(
            parameter_ordinals=self.parameter_ordinals,
            dim=self.dim,
            F=self.F.detach(),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move this dense block to a target device or solve dtype."""

        return type(self)(
            parameter_ordinals=self.parameter_ordinals,
            dim=self.dim,
            F=self.F.to(device=device, dtype=dtype),
        )

    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare metadata exactly and dense factor numerically."""

        if (
            type(self) is not type(other)
            or self.parameter_ordinals != other.parameter_ordinals
            or self.dim != other.dim
        ):
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks((self.F,), (other.F,), atol=atol, rtol=rtol)

    def __getstate__(self) -> dict[str, Any]:
        """Refuse serialization while the factor carries an autograd graph."""

        if self.F.requires_grad:
            raise RuntimeError("graph-bearing DenseFactorBlock cannot be serialized")
        return self.__dict__


@dataclass(frozen=True, eq=False, kw_only=True)
class StructuredCurvatureForwardPacket(WavefunctionPacket):
    """Detached structured curvature factors plus energy half-gradient for one forward.

    This ephemeral packet carries detached ``solve_dtype`` curvature blocks,
    detached ``energy_half_gradient`` blocks, detached value output, and a live
    :class:`~tpen.data.batch.ParameterBinding`. It unconditionally refuses
    serialization.
    """

    parameter_binding: ParameterBinding
    solve_dtype: torch.dtype
    kronecker_blocks: tuple[KroneckerFactorBlock, ...]
    dense_blocks: tuple[DenseFactorBlock, ...]
    energy_half_gradient: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kronecker_blocks", tuple(self.kronecker_blocks))
        object.__setattr__(self, "dense_blocks", tuple(self.dense_blocks))
        object.__setattr__(self, "energy_half_gradient", tuple(self.energy_half_gradient))
        self.validate()

    def __getstate__(self) -> dict[str, Any]:
        """Unconditionally refuse serialization of a live curvature packet."""

        if (
            self.output.logabs.requires_grad
            or any(b.A.requires_grad or b.G.requires_grad for b in self.kronecker_blocks)
            or any(b.F.requires_grad for b in self.dense_blocks)
            or any(g.requires_grad for g in self.energy_half_gradient)
        ):
            raise RuntimeError("graph-bearing StructuredCurvatureForwardPacket cannot be serialized")
        raise RuntimeError("live StructuredCurvatureForwardPacket cannot be serialized")

    def validate(self) -> Self:
        """Validate exhaustive disjoint parameter coverage, shapes, dtypes, and devices."""

        if not isinstance(self.output, WavefunctionOutput):
            raise TypeError("StructuredCurvatureForwardPacket.output must be a WavefunctionOutput")
        self.output.validate()
        if not isinstance(self.parameter_binding, ParameterBinding):
            raise TypeError(
                "StructuredCurvatureForwardPacket.parameter_binding must be a ParameterBinding"
            )
        self.parameter_binding.validate()
        _validate_solve_dtype(
            self.solve_dtype, field_name="StructuredCurvatureForwardPacket.solve_dtype"
        )
        slots = self.parameter_binding.layout.slots
        if not slots:
            raise ValueError("StructuredCurvatureForwardPacket requires at least one trainable parameter")
        if len(self.energy_half_gradient) != len(slots):
            raise ValueError(
                "StructuredCurvatureForwardPacket.energy_half_gradient must have one tensor per slot"
            )
        binding_device = self.parameter_binding.parameters[0].device
        if self.output.logabs.device != binding_device:
            raise ValueError("StructuredCurvatureForwardPacket output and binding must share one device")
        for slot, grad in zip(slots, self.energy_half_gradient, strict=True):
            if not isinstance(grad, torch.Tensor):
                raise TypeError("energy_half_gradient entries must be torch.Tensor values")
            if tuple(grad.shape) != slot.shape:
                raise ValueError(
                    f"energy_half_gradient[{slot.ordinal}] expected shape {slot.shape}, got {tuple(grad.shape)}"
                )
            if grad.dtype != self.solve_dtype:
                raise ValueError(
                    f"energy_half_gradient[{slot.ordinal}] expected dtype {self.solve_dtype}, got {grad.dtype}"
                )
            if grad.device != binding_device:
                raise ValueError("energy_half_gradient tensors must share the parameter device")

        covered: set[int] = set()
        for block in self.kronecker_blocks:
            if not isinstance(block, KroneckerFactorBlock):
                raise TypeError("kronecker_blocks must contain KroneckerFactorBlock instances")
            block.validate()
            if block.A.dtype != self.solve_dtype or block.A.device != binding_device:
                raise ValueError("KroneckerFactorBlock dtype/device disagrees with packet")
            for ordinal in block.parameter_ordinals:
                if ordinal >= len(slots):
                    raise ValueError(f"KroneckerFactorBlock ordinal {ordinal} out of range")
                if ordinal in covered:
                    raise ValueError(f"Duplicate coverage of parameter ordinal {ordinal}")
                covered.add(ordinal)
            weight_slot = slots[block.parameter_ordinals[0]]
            expected_in = prod(weight_slot.shape[1:]) + (1 if block.has_bias else 0)
            if len(weight_slot.shape) < 2 or weight_slot.shape[0] != block.out_dim or expected_in != block.in_dim:
                raise ValueError(
                    f"KroneckerFactorBlock dimensions {(block.out_dim, block.in_dim)} disagree with "
                    f"weight slot shape {weight_slot.shape} (has_bias={block.has_bias})"
                )
            if block.has_bias:
                bias_slot = slots[block.parameter_ordinals[1]]
                if bias_slot.shape != (block.out_dim,):
                    raise ValueError(
                        f"KroneckerFactorBlock bias slot shape {bias_slot.shape} disagrees with out_dim={block.out_dim}"
                    )

        for block in self.dense_blocks:
            if not isinstance(block, DenseFactorBlock):
                raise TypeError("dense_blocks must contain DenseFactorBlock instances")
            block.validate()
            if block.F.dtype != self.solve_dtype or block.F.device != binding_device:
                raise ValueError("DenseFactorBlock dtype/device disagrees with packet")
            expected_dim = 0
            for ordinal in block.parameter_ordinals:
                if ordinal >= len(slots):
                    raise ValueError(f"DenseFactorBlock ordinal {ordinal} out of range")
                if ordinal in covered:
                    raise ValueError(f"Duplicate coverage of parameter ordinal {ordinal}")
                covered.add(ordinal)
                expected_dim += slots[ordinal].numel
            if expected_dim != block.dim:
                raise ValueError(
                    f"DenseFactorBlock dim {block.dim} disagrees with covered numel {expected_dim}"
                )

        missing = sorted(set(range(len(slots))) - covered)
        if missing:
            raise ValueError(
                f"StructuredCurvatureForwardPacket does not exhaustively cover parameter ordinals: missing {missing}"
            )
        return self

    def as_output(self) -> WavefunctionOutput:
        """Return the composed detached value output."""

        return self.output

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Validate device compatibility against live parameters and cast solve tensors."""

        self.parameter_binding.to(device=device, dtype=None)
        target_solve_dtype = self.solve_dtype if dtype is None else _validate_solve_dtype(
            dtype, field_name="StructuredCurvatureForwardPacket.to(dtype)"
        )
        return type(self)(
            output=self.output.to(device=device),
            parameter_binding=self.parameter_binding,
            solve_dtype=target_solve_dtype,
            kronecker_blocks=tuple(
                block.to(device=device, dtype=target_solve_dtype) for block in self.kronecker_blocks
            ),
            dense_blocks=tuple(
                block.to(device=device, dtype=target_solve_dtype) for block in self.dense_blocks
            ),
            energy_half_gradient=tuple(
                grad.to(device=device, dtype=target_solve_dtype) for grad in self.energy_half_gradient
            ),
        )

    def detach(self) -> Self:
        """Return a detached copy preserving the live parameter binding."""

        return type(self)(
            output=_detach_output(self.output),
            parameter_binding=self.parameter_binding,
            solve_dtype=self.solve_dtype,
            kronecker_blocks=tuple(block.detach() for block in self.kronecker_blocks),
            dense_blocks=tuple(block.detach() for block in self.dense_blocks),
            energy_half_gradient=tuple(grad.detach() for grad in self.energy_half_gradient),
        )

    def permute(self, permutation: Permutation) -> Self:
        """Apply fermionic value permutation while cloning particle-invariant curvature payloads."""

        return type(self)(
            output=self.output.permute(permutation),
            parameter_binding=self.parameter_binding,
            solve_dtype=self.solve_dtype,
            kronecker_blocks=tuple(
                KroneckerFactorBlock(
                    parameter_ordinals=b.parameter_ordinals,
                    has_bias=b.has_bias,
                    out_dim=b.out_dim,
                    in_dim=b.in_dim,
                    A=b.A.clone(),
                    G=b.G.clone(),
                    active_locations=b.active_locations,
                )
                for b in self.kronecker_blocks
            ),
            dense_blocks=tuple(
                DenseFactorBlock(
                    parameter_ordinals=b.parameter_ordinals,
                    dim=b.dim,
                    F=b.F.clone(),
                )
                for b in self.dense_blocks
            ),
            energy_half_gradient=tuple(grad.clone() for grad in self.energy_half_gradient),
        )

    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare output, binding, curvature blocks, and energy half-gradient."""

        if (
            type(self) is not type(other)
            or self.solve_dtype != other.solve_dtype
            or not self.parameter_binding.compare(other.parameter_binding)[0]
            or len(self.kronecker_blocks) != len(other.kronecker_blocks)
            or len(self.dense_blocks) != len(other.dense_blocks)
        ):
            return False, {"max_abs_error": float("inf")}
        comparisons = [
            self.output.compare(other.output, atol=atol, rtol=rtol),
            compare_tensor_blocks(
                self.energy_half_gradient, other.energy_half_gradient, atol=atol, rtol=rtol
            ),
        ]
        for left, right in zip(self.kronecker_blocks, other.kronecker_blocks, strict=True):
            comparisons.append(left.compare(right, atol=atol, rtol=rtol))
        for left, right in zip(self.dense_blocks, other.dense_blocks, strict=True):
            comparisons.append(left.compare(right, atol=atol, rtol=rtol))
        return all(ok for ok, _ in comparisons), {
            "max_abs_error": max((m["max_abs_error"] for _, m in comparisons), default=0.0)
        }


@dataclass
class _PendingKroneckerCapture:
    """Intermediate Kronecker capture awaiting its cotangent from Backward #1."""

    parameter_ordinals: tuple[int, ...]
    has_bias: bool
    out_dim: int
    in_dim: int
    A: torch.Tensor
    active_locations: int
    output_tensor: torch.Tensor
    cotangent_kind: Literal["linear", "mixing_bov"]
    mask_info: tuple[int, int, torch.Tensor] | None = None


@dataclass
class _PendingLinearBiasOnlyCapture:
    """Intermediate frozen-weight/trainable-bias linear capture awaiting cotangent."""

    parameter_ordinal: int
    out_dim: int
    output_tensor: torch.Tensor
    mask_info: tuple[int, int, torch.Tensor] | None = None


@dataclass
class _PendingPathAggregationCapture:
    """Intermediate PathAggregation capture awaiting pre-activation cotangent."""

    parameter_ordinal: int
    channels: int
    paths: int
    h_detached: torch.Tensor
    output_tensor: torch.Tensor


@dataclass
class _PendingPfaffianReadoutCapture:
    """Intermediate PfaffianReadout capture awaiting d(logabs)/d(psi)."""

    parameter_ordinal: int
    channels: int
    channel_pfaffians: torch.Tensor
    output_tensor: torch.Tensor


class _StructuredCurvatureCaptureSession:
    """Opt-in single-forward curvature capture scope for native KFAC.

    Activated only during :func:`evaluate_structured_curvature_forward`.
    Outside this scope, every TPEN module executes its exact historical forward
    with zero overhead.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        binding: ParameterBinding,
        batch_size: int,
        solve_dtype: torch.dtype,
        max_dense_block_numel: int,
    ) -> None:
        self.model = model
        self.binding = binding
        self.batch_size = int(batch_size)
        self.solve_dtype = _validate_solve_dtype(solve_dtype, field_name="solve_dtype")
        self.max_dense_block_numel = _validate_positive_int(
            max_dense_block_numel, field_name="max_dense_block_numel"
        )
        self._id_to_ordinal: dict[int, int] = {
            id(parameter): ordinal for ordinal, parameter in enumerate(binding.parameters)
        }
        self._claimed_ordinals: set[int] = set()
        self._distinct_mask_stack: list[tuple[int, int]] = []
        self._hooks: list[Any] = []
        self._token: Any = None

        self._completed_kronecker: list[KroneckerFactorBlock] = []
        self._completed_dense: list[DenseFactorBlock] = []
        self._pending_kronecker: list[_PendingKroneckerCapture] = []
        self._pending_bias_only: list[_PendingLinearBiasOnlyCapture] = []
        self._pending_path_agg: list[_PendingPathAggregationCapture] = []
        self._pending_pfaffian: list[_PendingPfaffianReadoutCapture] = []

    def __enter__(self) -> Self:
        if _ACTIVE_CAPTURE_SESSION.get() is not None:
            raise RuntimeError("nested _StructuredCurvatureCaptureSession is not supported")
        self._register_linear_hooks()
        self._token = _ACTIVE_CAPTURE_SESSION.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_type, exc_val, exc_tb
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        if self._token is not None:
            _ACTIVE_CAPTURE_SESSION.reset(self._token)
            self._token = None

    @contextmanager
    def distinct_tuple_mask_scope(self, *, n_particles: int, order: int) -> Iterator[None]:
        """Mask repeated-particle positions for ``order >= 2`` linear layers inside this scope."""

        if order < 2:
            yield
            return
        self._distinct_mask_stack.append((int(n_particles), int(order)))
        try:
            yield
        finally:
            self._distinct_mask_stack.pop()

    def _claim_ordinal(self, parameter: nn.Parameter, *, label: str) -> int:
        ordinal = self._id_to_ordinal.get(id(parameter))
        if ordinal is None:
            raise ValueError(
                f"structured curvature capture encountered unbound trainable parameter for {label}"
            )
        if ordinal in self._claimed_ordinals:
            raise ValueError(
                f"structured curvature capture encountered duplicate ownership of parameter ordinal {ordinal} ({label})"
            )
        self._claimed_ordinals.add(ordinal)
        return ordinal

    def _register_linear_hooks(self) -> None:
        from tpen.nn.initialization import SeededLinear

        for module in self.model.modules():
            if not isinstance(module, (nn.Linear, SeededLinear)):
                continue
            weight_trainable = isinstance(module.weight, nn.Parameter) and module.weight.requires_grad
            bias_trainable = isinstance(module.bias, nn.Parameter) and module.bias.requires_grad
            if not (weight_trainable or bias_trainable):
                continue
            handle = module.register_forward_hook(self._linear_forward_hook)
            self._hooks.append(handle)

    def _linear_forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            raise ValueError(
                f"structured curvature linear hook expected single tensor input/output on {type(module).__name__}"
            )
        a = inputs[0]
        y = output
        if a.ndim < 2 or int(a.shape[0]) != self.batch_size:
            raise ValueError(
                f"structured curvature linear input expected leading batch_size={self.batch_size}, "
                f"got shape {tuple(a.shape)}"
            )
        if y.ndim != a.ndim or tuple(y.shape[:-1]) != tuple(a.shape[:-1]):
            raise ValueError(
                f"structured curvature linear output shape {tuple(y.shape)} disagrees with input {tuple(a.shape)}"
            )

        weight = module.weight
        bias = module.bias
        weight_trainable = isinstance(weight, nn.Parameter) and weight.requires_grad
        bias_trainable = isinstance(bias, nn.Parameter) and bias.requires_grad

        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])
        device = weight.device

        mask_info: tuple[int, int, torch.Tensor] | None = None
        if self._distinct_mask_stack:
            n_particles, order = self._distinct_mask_stack[-1]
            if a.ndim < order + 2 or tuple(a.shape[-1 - order : -1]) != (n_particles,) * order:
                raise ValueError(
                    f"distinct-tuple mask expected trailing particle axes {(n_particles,) * order} "
                    f"before channel axis, got input shape {tuple(a.shape)}"
                )
            mask_flat = no_repeated_particle_mask(n_particles, order, device=a.device).reshape(-1)
            n_extra = prod(a.shape[1 : -1 - order])
            tuple_size = n_particles ** order
            n_distinct = int(mask_flat.sum().item())
            active_locations = n_extra * n_distinct
            mask_info = (n_extra, tuple_size, mask_flat)
        else:
            active_locations = prod(a.shape[1:-1])

        if not weight_trainable and bias_trainable:
            assert isinstance(bias, nn.Parameter)
            b_ord = self._claim_ordinal(bias, label=f"{type(module).__name__}.bias")
            if out_features > self.max_dense_block_numel:
                raise ValueError(
                    f"frozen-weight linear bias block dim {out_features} exceeds "
                    f"max_dense_block_numel={self.max_dense_block_numel}"
                )
            if active_locations == 0 or not y.requires_grad:
                self._completed_dense.append(
                    DenseFactorBlock(
                        parameter_ordinals=(b_ord,),
                        dim=out_features,
                        F=torch.zeros((out_features, out_features), device=device, dtype=self.solve_dtype),
                    )
                )
            else:
                self._pending_bias_only.append(
                    _PendingLinearBiasOnlyCapture(
                        parameter_ordinal=b_ord,
                        out_dim=out_features,
                        output_tensor=y,
                        mask_info=mask_info,
                    )
                )
            return

        assert isinstance(weight, nn.Parameter)
        w_ord = self._claim_ordinal(weight, label=f"{type(module).__name__}.weight")
        if bias_trainable:
            assert isinstance(bias, nn.Parameter)
            b_ord = self._claim_ordinal(bias, label=f"{type(module).__name__}.bias")
            ordinals: tuple[int, ...] = (w_ord, b_ord)
            has_bias = True
            d_a = in_features + 1
        else:
            ordinals = (w_ord,)
            has_bias = False
            d_a = in_features

        if active_locations == 0:
            self._completed_kronecker.append(
                KroneckerFactorBlock(
                    parameter_ordinals=ordinals,
                    has_bias=has_bias,
                    out_dim=out_features,
                    in_dim=d_a,
                    A=torch.zeros((d_a, d_a), device=device, dtype=self.solve_dtype),
                    G=torch.zeros((out_features, out_features), device=device, dtype=self.solve_dtype),
                    active_locations=0,
                )
            )
            return

        a_det = a.detach()
        if mask_info is not None:
            n_extra, tuple_size, mask_flat = mask_info
            a_active = a_det.reshape(self.batch_size, n_extra, tuple_size, in_features)[
                :, :, mask_flat, :
            ].reshape(self.batch_size * active_locations, in_features)
        else:
            a_active = a_det.reshape(self.batch_size * active_locations, in_features)
        a_solve = a_active.to(device=device, dtype=self.solve_dtype)
        _require_finite(a_solve, label=f"{type(module).__name__} activation a")
        if has_bias:
            ones = torch.ones((a_solve.shape[0], 1), device=device, dtype=self.solve_dtype)
            a_solve = torch.cat((a_solve, ones), dim=1)
        A = _symmetrize((a_solve.transpose(0, 1) @ a_solve) / float(self.batch_size * active_locations))
        _require_finite(A, label=f"{type(module).__name__} Kronecker factor A")

        if not y.requires_grad:
            self._completed_kronecker.append(
                KroneckerFactorBlock(
                    parameter_ordinals=ordinals,
                    has_bias=has_bias,
                    out_dim=out_features,
                    in_dim=d_a,
                    A=A,
                    G=torch.zeros((out_features, out_features), device=device, dtype=self.solve_dtype),
                    active_locations=active_locations,
                )
            )
            return

        self._pending_kronecker.append(
            _PendingKroneckerCapture(
                parameter_ordinals=ordinals,
                has_bias=has_bias,
                out_dim=out_features,
                in_dim=d_a,
                A=A,
                active_locations=active_locations,
                output_tensor=y,
                cotangent_kind="linear",
                mask_info=mask_info,
            )
        )

    def record_mixing_kronecker(
        self,
        *,
        weight: torch.Tensor,
        flattened_input: torch.Tensor,
        contribution: torch.Tensor,
        label: str,
    ) -> None:
        """Capture one vectorized unary/binary mixing path before scatter_add_."""

        if not isinstance(weight, nn.Parameter) or not weight.requires_grad:
            return
        w_ord = self._claim_ordinal(weight, label=label)
        out_dim = int(weight.shape[0])
        in_dim = prod(weight.shape[1:])
        device = weight.device
        if flattened_input.ndim != 3 or int(flattened_input.shape[0]) != self.batch_size or int(flattened_input.shape[1]) != in_dim:
            raise ValueError(
                f"{label} flattened_input expected shape {(self.batch_size, in_dim, 'V')}, "
                f"got {tuple(flattened_input.shape)}"
            )
        n_supports = int(flattened_input.shape[2])
        if tuple(contribution.shape) != (self.batch_size, out_dim, n_supports):
            raise ValueError(
                f"{label} contribution expected shape {(self.batch_size, out_dim, n_supports)}, "
                f"got {tuple(contribution.shape)}"
            )

        if n_supports == 0:
            self._completed_kronecker.append(
                KroneckerFactorBlock(
                    parameter_ordinals=(w_ord,),
                    has_bias=False,
                    out_dim=out_dim,
                    in_dim=in_dim,
                    A=torch.zeros((in_dim, in_dim), device=device, dtype=self.solve_dtype),
                    G=torch.zeros((out_dim, out_dim), device=device, dtype=self.solve_dtype),
                    active_locations=0,
                )
            )
            return

        a_solve = (
            flattened_input.detach()
            .permute(0, 2, 1)
            .reshape(self.batch_size * n_supports, in_dim)
            .to(device=device, dtype=self.solve_dtype)
        )
        _require_finite(a_solve, label=f"{label} input a")
        A = _symmetrize((a_solve.transpose(0, 1) @ a_solve) / float(self.batch_size * n_supports))
        _require_finite(A, label=f"{label} Kronecker factor A")

        if not contribution.requires_grad:
            self._completed_kronecker.append(
                KroneckerFactorBlock(
                    parameter_ordinals=(w_ord,),
                    has_bias=False,
                    out_dim=out_dim,
                    in_dim=in_dim,
                    A=A,
                    G=torch.zeros((out_dim, out_dim), device=device, dtype=self.solve_dtype),
                    active_locations=n_supports,
                )
            )
            return

        self._pending_kronecker.append(
            _PendingKroneckerCapture(
                parameter_ordinals=(w_ord,),
                has_bias=False,
                out_dim=out_dim,
                in_dim=in_dim,
                A=A,
                active_locations=n_supports,
                output_tensor=contribution,
                cotangent_kind="mixing_bov",
            )
        )

    def record_path_aggregation(
        self,
        *,
        weight: torch.Tensor,
        interaction_block: torch.Tensor,
        aggregated: torch.Tensor,
        label: str,
    ) -> None:
        """Capture PathAggregation inputs and pre-activation outputs for algebraic O_U."""

        if not isinstance(weight, nn.Parameter) or not weight.requires_grad:
            return
        w_ord = self._claim_ordinal(weight, label=label)
        channels = int(weight.shape[0])
        paths = int(weight.shape[1])
        dim = channels * paths
        if dim > self.max_dense_block_numel:
            raise ValueError(
                f"{label} dense block numel {dim} exceeds max_dense_block_numel={self.max_dense_block_numel}"
            )
        device = weight.device
        if dim == 0 or not aggregated.requires_grad:
            self._completed_dense.append(
                DenseFactorBlock(
                    parameter_ordinals=(w_ord,),
                    dim=dim,
                    F=torch.zeros((dim, dim), device=device, dtype=self.solve_dtype),
                )
            )
            return
        h_det = interaction_block.detach()
        _require_finite(h_det, label=f"{label} input h")
        self._pending_path_agg.append(
            _PendingPathAggregationCapture(
                parameter_ordinal=w_ord,
                channels=channels,
                paths=paths,
                h_detached=h_det,
                output_tensor=aggregated,
            )
        )

    def record_pfaffian_readout(
        self,
        *,
        channel_weights: torch.Tensor,
        channel_pfaffians: torch.Tensor,
        psi: torch.Tensor,
    ) -> None:
        """Capture PfaffianReadout channel Pfaffians and pre-logabs psi."""

        if not isinstance(channel_weights, nn.Parameter) or not channel_weights.requires_grad:
            return
        w_ord = self._claim_ordinal(channel_weights, label="PfaffianReadout.channel_weights")
        channels = int(channel_weights.numel())
        if channels > self.max_dense_block_numel:
            raise ValueError(
                f"PfaffianReadout.channel_weights numel {channels} exceeds "
                f"max_dense_block_numel={self.max_dense_block_numel}"
            )
        device = channel_weights.device
        pf_det = channel_pfaffians.detach()
        _require_finite(pf_det, label="PfaffianReadout.channel_pfaffians")
        if channels == 0 or not psi.requires_grad:
            self._completed_dense.append(
                DenseFactorBlock(
                    parameter_ordinals=(w_ord,),
                    dim=channels,
                    F=torch.zeros((channels, channels), device=device, dtype=self.solve_dtype),
                )
            )
            return
        self._pending_pfaffian.append(
            _PendingPfaffianReadoutCapture(
                parameter_ordinal=w_ord,
                channels=channels,
                channel_pfaffians=pf_det,
                output_tensor=psi,
            )
        )

    def record_local_dense_scores(
        self,
        *,
        parameters: Sequence[nn.Parameter],
        scores: torch.Tensor,
        label: str,
    ) -> None:
        """Record exact locally computed per-sample scores for post-readout factors."""

        trainable = tuple(p for p in parameters if isinstance(p, nn.Parameter) and p.requires_grad)
        if not trainable:
            return
        ordinals = tuple(self._claim_ordinal(p, label=label) for p in trainable)
        dim = sum(p.numel() for p in trainable)
        if dim > self.max_dense_block_numel:
            raise ValueError(
                f"{label} dense block numel {dim} exceeds max_dense_block_numel={self.max_dense_block_numel}"
            )
        device = trainable[0].device
        scores_solve = scores.detach().to(device=device, dtype=self.solve_dtype)
        if tuple(scores_solve.shape) != (self.batch_size, dim):
            raise ValueError(
                f"{label} local scores expected shape {(self.batch_size, dim)}, got {tuple(scores_solve.shape)}"
            )
        _require_finite(scores_solve, label=f"{label} score rows O_b")
        centered = scores_solve - scores_solve.mean(dim=0, keepdim=True)
        F = _symmetrize((centered.transpose(0, 1) @ centered) / float(self.batch_size))
        _require_finite(F, label=f"{label} dense factor F")
        self._completed_dense.append(
            DenseFactorBlock(parameter_ordinals=ordinals, dim=dim, F=F)
        )

    def captured_output_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return ordered graph-bearing outputs for the single shared backward pass."""

        outputs: list[torch.Tensor] = []
        outputs.extend(item.output_tensor for item in self._pending_kronecker)
        outputs.extend(item.output_tensor for item in self._pending_bias_only)
        outputs.extend(item.output_tensor for item in self._pending_path_agg)
        outputs.extend(item.output_tensor for item in self._pending_pfaffian)
        return tuple(outputs)

    def finalize(
        self,
        cotangents: Sequence[torch.Tensor | None],
    ) -> tuple[tuple[KroneckerFactorBlock, ...], tuple[DenseFactorBlock, ...]]:
        """Assemble all Kronecker and Dense factor blocks from Backward #1 cotangents."""

        expected_count = (
            len(self._pending_kronecker)
            + len(self._pending_bias_only)
            + len(self._pending_path_agg)
            + len(self._pending_pfaffian)
        )
        if len(cotangents) != expected_count:
            raise RuntimeError(
                f"structured curvature cotangent count mismatch: expected {expected_count}, got {len(cotangents)}"
            )

        cursor = 0
        kron_blocks = list(self._completed_kronecker)
        dense_blocks = list(self._completed_dense)

        for item in self._pending_kronecker:
            grad = cotangents[cursor]
            cursor += 1
            device = item.A.device
            if grad is None:
                G = torch.zeros((item.out_dim, item.out_dim), device=device, dtype=self.solve_dtype)
            elif item.cotangent_kind == "mixing_bov":
                g_solve = (
                    grad.detach()
                    .permute(0, 2, 1)
                    .reshape(self.batch_size * item.active_locations, item.out_dim)
                    .to(device=device, dtype=self.solve_dtype)
                )
                _require_finite(g_solve, label="mixing cotangent g")
                G = _symmetrize((g_solve.transpose(0, 1) @ g_solve) / float(self.batch_size))
                _require_finite(G, label="mixing Kronecker factor G")
            else:
                g_det = grad.detach()
                if item.mask_info is not None:
                    n_extra, tuple_size, mask_flat = item.mask_info
                    g_active = g_det.reshape(self.batch_size, n_extra, tuple_size, item.out_dim)[
                        :, :, mask_flat, :
                    ].reshape(self.batch_size * item.active_locations, item.out_dim)
                else:
                    g_active = g_det.reshape(self.batch_size * item.active_locations, item.out_dim)
                g_solve = g_active.to(device=device, dtype=self.solve_dtype)
                _require_finite(g_solve, label="linear cotangent g")
                G = _symmetrize((g_solve.transpose(0, 1) @ g_solve) / float(self.batch_size))
                _require_finite(G, label="linear Kronecker factor G")
            kron_blocks.append(
                KroneckerFactorBlock(
                    parameter_ordinals=item.parameter_ordinals,
                    has_bias=item.has_bias,
                    out_dim=item.out_dim,
                    in_dim=item.in_dim,
                    A=item.A,
                    G=G,
                    active_locations=item.active_locations,
                )
            )

        for item in self._pending_bias_only:
            grad = cotangents[cursor]
            cursor += 1
            device = self.binding.parameters[item.parameter_ordinal].device
            if grad is None:
                F = torch.zeros((item.out_dim, item.out_dim), device=device, dtype=self.solve_dtype)
            else:
                g_det = grad.detach()
                if item.mask_info is not None:
                    n_extra, tuple_size, mask_flat = item.mask_info
                    g_active = g_det.reshape(self.batch_size, n_extra, tuple_size, item.out_dim)[
                        :, :, mask_flat, :
                    ].reshape(self.batch_size, -1, item.out_dim)
                else:
                    g_active = g_det.reshape(self.batch_size, -1, item.out_dim)
                g_solve = g_active.to(device=device, dtype=self.solve_dtype)
                _require_finite(g_solve, label="linear bias-only cotangent g")
                scores = g_solve.sum(dim=1)
                _require_finite(scores, label="linear bias-only score rows O_b")
                centered = scores - scores.mean(dim=0, keepdim=True)
                F = _symmetrize((centered.transpose(0, 1) @ centered) / float(self.batch_size))
                _require_finite(F, label="linear bias-only dense factor F")
            dense_blocks.append(
                DenseFactorBlock(
                    parameter_ordinals=(item.parameter_ordinal,),
                    dim=item.out_dim,
                    F=F,
                )
            )

        for item in self._pending_path_agg:
            grad = cotangents[cursor]
            cursor += 1
            dim = item.channels * item.paths
            device = self.binding.parameters[item.parameter_ordinal].device
            if grad is None:
                F = torch.zeros((dim, dim), device=device, dtype=self.solve_dtype)
            else:
                g_solve = grad.detach().reshape(self.batch_size, item.channels, -1).to(
                    device=device, dtype=self.solve_dtype
                )
                h_solve = item.h_detached.reshape(self.batch_size, item.channels, item.paths, -1).to(
                    device=device, dtype=self.solve_dtype
                )
                _require_finite(g_solve, label="PathAggregation cotangent g")
                scores = torch.einsum("bcl,bcpl->bcp", g_solve, h_solve).reshape(self.batch_size, dim)
                _require_finite(scores, label="PathAggregation score rows O_b")
                centered = scores - scores.mean(dim=0, keepdim=True)
                F = _symmetrize((centered.transpose(0, 1) @ centered) / float(self.batch_size))
                _require_finite(F, label="PathAggregation dense factor F")
            dense_blocks.append(
                DenseFactorBlock(
                    parameter_ordinals=(item.parameter_ordinal,),
                    dim=dim,
                    F=F,
                )
            )

        for item in self._pending_pfaffian:
            grad = cotangents[cursor]
            cursor += 1
            device = self.binding.parameters[item.parameter_ordinal].device
            if grad is None:
                F = torch.zeros((item.channels, item.channels), device=device, dtype=self.solve_dtype)
            else:
                g_psi = grad.detach().reshape(self.batch_size).to(device=device, dtype=self.solve_dtype)
                _require_finite(g_psi, label="PfaffianReadout cotangent g_psi")
                pf_solve = item.channel_pfaffians.to(device=device, dtype=self.solve_dtype)
                scores = g_psi.unsqueeze(1) * pf_solve
                _require_finite(scores, label="PfaffianReadout score rows O_b")
                centered = scores - scores.mean(dim=0, keepdim=True)
                F = _symmetrize((centered.transpose(0, 1) @ centered) / float(self.batch_size))
                _require_finite(F, label="PfaffianReadout dense factor F")
            dense_blocks.append(
                DenseFactorBlock(
                    parameter_ordinals=(item.parameter_ordinal,),
                    dim=item.channels,
                    F=F,
                )
            )

        # Enforce exhaustive parameter coverage with zero silent identity fallback.
        expected_ordinals = set(range(len(self.binding.parameters)))
        uncovered = sorted(expected_ordinals - self._claimed_ordinals)
        if uncovered:
            id_to_name = {id(p): name for name, p in self.model.named_parameters()}
            names = [
                id_to_name.get(id(self.binding.parameters[o]), f"<ordinal {o}>")
                for o in uncovered
            ]
            raise ValueError(
                "StructuredCurvatureForwardRequest found unrecognized trainable parameter(s) "
                f"without an explicit local KFAC capture rule (ordinals={uncovered}, names={names}); "
                "refusing silent identity fallback or vmap traversal across SoftplusBeta trunks"
            )

        kron_blocks.sort(key=lambda blk: blk.parameter_ordinals[0])
        dense_blocks.sort(key=lambda blk: blk.parameter_ordinals[0])
        return tuple(kron_blocks), tuple(dense_blocks)


def _detach_output(output: WavefunctionOutput) -> WavefunctionOutput:
    """Return a value-only detached WavefunctionOutput."""

    return WavefunctionOutput(
        logabs=output.logabs.detach(),
        sign=output.sign.detach(),
        phase=None if output.phase is None else output.phase.detach(),
        aux={
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in output.aux.items()
        },
    )


def _resolve_model_binding(model: nn.Module) -> ParameterBinding:
    """Validate that a model exposes a live, consistent ParameterBinding."""

    binding = getattr(model, "parameter_binding", None)
    if not isinstance(binding, ParameterBinding):
        raise TypeError(
            "StructuredCurvatureForwardRequest requires model.parameter_binding to be a ParameterBinding"
        )
    try:
        binding.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError("parameter binding/layout is no longer valid") from exc
    if not binding.parameters:
        raise ValueError("StructuredCurvatureForwardRequest requires at least one trainable parameter")
    current = tuple(p for p in model.parameters() if p.requires_grad)
    if len(current) != len(binding.parameters) or any(
        left is not right for left, right in zip(current, binding.parameters, strict=True)
    ):
        raise ValueError(
            "parameter binding/layout mismatch, freezing, or reordering; refusing curvature forward"
        )
    return binding


def evaluate_structured_curvature_forward(
    model: nn.Module,
    *,
    batch: ElectronBatch,
    sample_weights: torch.Tensor,
    solve_dtype: torch.dtype = torch.float64,
    max_dense_block_numel: int = 2048,
    forward_fn: Any = None,
) -> StructuredCurvatureForwardPacket:
    """Evaluate one forward plus two backward passes for structured KFAC capture.

    Parameters
    ----------
    model : torch.nn.Module
        Wavefunction module exposing a live :attr:`parameter_binding`.
    batch : ElectronBatch
        Pinned finite electron batch (must be non-empty).
    sample_weights : torch.Tensor
        Detached 1D per-sample weights shaped ``(batch.batch_size,)`` used for
        the weighted energy half-gradient backward pass.
    solve_dtype : torch.dtype, optional
        Real floating dtype (``torch.float32`` or ``torch.float64``) for all
        captured curvature factors and energy half-gradient blocks.
    max_dense_block_numel : int, optional
        Maximum allowed scalar element count for any :class:`DenseFactorBlock`.
    forward_fn : callable or None, optional
        Optional bound forward callable (defaults to ``model(batch)``).
    """

    if torch.is_inference_mode_enabled():
        raise RuntimeError("StructuredCurvatureForwardRequest is not supported in inference mode")
    if not isinstance(batch, ElectronBatch):
        raise TypeError(
            f"StructuredCurvatureForwardRequest requires ElectronBatch, got {type(batch).__name__}"
        )
    if not isinstance(sample_weights, torch.Tensor):
        raise TypeError("StructuredCurvatureForwardRequest.sample_weights must be a torch.Tensor")
    if sample_weights.requires_grad:
        raise ValueError("StructuredCurvatureForwardRequest.sample_weights must be detached")
    if not sample_weights.is_floating_point() or sample_weights.is_complex():
        raise TypeError("StructuredCurvatureForwardRequest.sample_weights must have a real floating dtype")
    if sample_weights.ndim != 1:
        raise ValueError(
            f"StructuredCurvatureForwardRequest.sample_weights must be 1D, got shape {tuple(sample_weights.shape)}"
        )
    if batch.batch_size < 1:
        raise ValueError("StructuredCurvatureForwardRequest requires a non-empty ElectronBatch")
    if int(sample_weights.shape[0]) != batch.batch_size:
        raise ValueError(
            f"StructuredCurvatureForwardRequest.sample_weights length {sample_weights.shape[0]} "
            f"disagrees with batch.batch_size={batch.batch_size}"
        )
    if sample_weights.device != batch.device:
        raise ValueError("StructuredCurvatureForwardRequest sample_weights and batch must share one device")
    _require_finite(sample_weights, label="sample_weights")
    _validate_solve_dtype(solve_dtype, field_name="solve_dtype")
    _validate_positive_int(max_dense_block_numel, field_name="max_dense_block_numel")

    binding = _resolve_model_binding(model)
    invoke_forward = model if forward_fn is None else forward_fn

    with torch.enable_grad():
        with _StructuredCurvatureCaptureSession(
            model=model,
            binding=binding,
            batch_size=batch.batch_size,
            solve_dtype=solve_dtype,
            max_dense_block_numel=max_dense_block_numel,
        ) as session:
            output = invoke_forward(batch)
            if not isinstance(output, WavefunctionOutput):
                raise TypeError(
                    f"StructuredCurvatureForwardRequest expected WavefunctionOutput, got {type(output).__name__}"
                )
            if output.phase is not None:
                raise ValueError(
                    "StructuredCurvatureForwardRequest supports real wavefunctions only (phase must be None)"
                )
            logabs = output.logabs
            if tuple(logabs.shape) != (batch.batch_size,):
                raise ValueError(
                    f"StructuredCurvatureForwardRequest expected flat logabs shape {(batch.batch_size,)}, "
                    f"got {tuple(logabs.shape)}"
                )
            _require_finite(logabs.detach(), label="WavefunctionOutput.logabs")
            if not logabs.requires_grad:
                raise RuntimeError(
                    "StructuredCurvatureForwardRequest requires a differentiable logabs output"
                )

            captured_outputs = session.captured_output_tensors()
            if captured_outputs:
                cotangents = torch.autograd.grad(
                    logabs.sum(),
                    captured_outputs,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )
            else:
                cotangents = ()

            weights_cast = sample_weights.detach().to(device=logabs.device, dtype=logabs.dtype)
            raw_half_grads = torch.autograd.grad(
                (weights_cast * logabs).sum(),
                binding.parameters,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

            kronecker_blocks, dense_blocks = session.finalize(cotangents)

    energy_half_gradient: list[torch.Tensor] = []
    for parameter, raw_grad in zip(binding.parameters, raw_half_grads, strict=True):
        if raw_grad is None:
            grad_solve = torch.zeros(parameter.shape, device=parameter.device, dtype=solve_dtype)
        else:
            grad_solve = raw_grad.detach().to(device=parameter.device, dtype=solve_dtype)
        _require_finite(grad_solve, label="energy_half_gradient")
        energy_half_gradient.append(grad_solve)

    return StructuredCurvatureForwardPacket(
        output=_detach_output(output),
        parameter_binding=binding,
        solve_dtype=solve_dtype,
        kronecker_blocks=kronecker_blocks,
        dense_blocks=dense_blocks,
        energy_half_gradient=tuple(energy_half_gradient),
    )


__all__ = [
    "DenseFactorBlock",
    "KroneckerFactorBlock",
    "StructuredCurvatureForwardPacket",
    "active_structured_curvature_session",
    "evaluate_structured_curvature_forward",
    "softplus_derivative",
]
