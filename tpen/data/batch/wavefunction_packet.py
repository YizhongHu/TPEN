"""Immutable typed wavefunction forward-packet contracts.

The compatibility :class:`WavefunctionOutput` remains the ordinary value
surface.  Rich forward results compose that value with one exact derivative
payload so callers cannot accidentally request a capability and receive a
packet that omits it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Self

import torch

from tpen.data.batch.wavefunction_output import WavefunctionOutput
from tpen.data.equivariant_state import compare_tensor_blocks
from tpen.data.indices import permute_particle_axis
from tpen.data.permutation import Permutation


@dataclass(frozen=True, eq=False, kw_only=True)
class CoordinateLogGradient:
    """Store the exact coordinate gradient of real ``logabs`` values.

    Parameters
    ----------
    values : torch.Tensor
        Real tensor with shape
        ``[*sample_shape, n_electrons, spatial_dim]``.
    """

    values: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Return the leading sample dimensions."""

        return tuple(self.values.shape[:-2])

    @property
    def n_electrons(self) -> int:
        """Return the size of the electron axis."""

        return int(self.values.shape[-2])

    @property
    def spatial_dim(self) -> int:
        """Return the coordinate dimension."""

        return int(self.values.shape[-1])

    @property
    def device(self) -> torch.device:
        """Return the tensor device."""

        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the real floating tensor dtype."""

        return self.values.dtype

    def validate(
        self,
        *,
        sample_shape: tuple[int, ...] | None = None,
        n_electrons: int | None = None,
        spatial_dim: int | None = None,
    ) -> Self:
        """Validate type, rank, and optional shape metadata."""

        if not isinstance(self.values, torch.Tensor):
            raise TypeError("CoordinateLogGradient.values must be a torch.Tensor")
        if self.values.ndim < 2:
            raise ValueError(
                "CoordinateLogGradient.values must have shape "
                "[*sample_shape, n_electrons, spatial_dim]"
            )
        if not self.values.is_floating_point():
            raise TypeError("CoordinateLogGradient.values must have a real floating dtype")
        if self.n_electrons < 1:
            raise ValueError("CoordinateLogGradient requires at least one electron")
        if self.spatial_dim < 1:
            raise ValueError("CoordinateLogGradient spatial_dim must be at least one")
        if sample_shape is not None and self.sample_shape != tuple(sample_shape):
            raise ValueError(
                f"CoordinateLogGradient expected sample shape {tuple(sample_shape)}, "
                f"got {self.sample_shape}"
            )
        if n_electrons is not None and self.n_electrons != n_electrons:
            raise ValueError(
                f"CoordinateLogGradient expected {n_electrons} electrons, got {self.n_electrons}"
            )
        if spatial_dim is not None and self.spatial_dim != spatial_dim:
            raise ValueError(
                f"CoordinateLogGradient expected spatial_dim {spatial_dim}, got {self.spatial_dim}"
            )
        return self

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move the gradient to a new device or real floating dtype."""

        return type(self)(values=self.values.to(device=device, dtype=dtype))

    def detach(self) -> Self:
        """Return the gradient detached from its autograd graph."""

        return type(self)(values=self.values.detach())

    def permute(self, permutation: Permutation) -> Self:
        """Permute the explicit electron axis using the active convention."""

        return type(self)(values=permute_particle_axis(self.values, permutation, axis=-2))

    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare coordinate-gradient values."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks([self.values], [other.values], atol=atol, rtol=rtol)


@dataclass(frozen=True, eq=False, kw_only=True)
class ParameterSlot:
    """Describe one statically ordered trainable-parameter slot.

    Parameters
    ----------
    ordinal : int
        Zero-based position in the owning layout.
    shape : tuple of int
        Exact parameter shape.
    numel : int
        Product of `shape`, stored explicitly for layout validation.
    dtype : torch.dtype
        Real floating parameter dtype.
    """

    ordinal: int
    shape: tuple[int, ...]
    numel: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        self.validate()

    def validate(self) -> Self:
        """Validate ordinal, shape, element count, and dtype."""

        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool) or self.ordinal < 0:
            raise ValueError("ParameterSlot.ordinal must be a non-negative integer")
        if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in self.shape):
            raise ValueError("ParameterSlot.shape entries must be non-negative integers")
        if not isinstance(self.numel, int) or isinstance(self.numel, bool) or self.numel < 0:
            raise ValueError("ParameterSlot.numel must be a non-negative integer")
        expected_numel = prod(self.shape)
        if self.numel != expected_numel:
            raise ValueError(
                f"ParameterSlot.numel must equal prod(shape)={expected_numel}, got {self.numel}"
            )
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise TypeError("ParameterSlot.dtype must be a real floating torch.dtype")
        return self

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return static metadata with an optionally replaced dtype.

        `device` is accepted for a uniform semantic ``to`` surface but has no
        effect because a static slot carries no device-bound value.
        """

        del device
        return type(self)(
            ordinal=self.ordinal,
            shape=self.shape,
            numel=self.numel,
            dtype=self.dtype if dtype is None else dtype,
        )

    def detach(self) -> Self:
        """Return this tensor-free immutable metadata."""

        return self

    def permute(self, permutation: Permutation) -> Self:
        """Return this particle-invariant static metadata."""

        del permutation
        return self

    def compare(
        self,
        other: Self,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> tuple[bool, dict[str, float]]:
        """Compare exact slot metadata."""

        del atol, rtol
        close = (
            type(self) is type(other)
            and self.ordinal == other.ordinal
            and self.shape == other.shape
            and self.numel == other.numel
            and self.dtype == other.dtype
        )
        return close, {"max_abs_error": 0.0 if close else float("inf")}


@dataclass(frozen=True, eq=False, kw_only=True)
class ParameterLayout:
    """Store the immutable ordered metadata for trainable parameters."""

    slots: tuple[ParameterSlot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))
        self.validate()

    @property
    def total_numel(self) -> int:
        """Return the total number of scalar parameters in the layout."""

        return sum(slot.numel for slot in self.slots)

    def validate(self) -> Self:
        """Validate exact slot types and dense ordinal order."""

        for ordinal, slot in enumerate(self.slots):
            if not isinstance(slot, ParameterSlot):
                raise TypeError("ParameterLayout.slots must contain only ParameterSlot values")
            slot.validate()
            if slot.ordinal != ordinal:
                raise ValueError(
                    "ParameterLayout slot ordinals must be dense and match tuple order; "
                    f"expected {ordinal}, got {slot.ordinal}"
                )
        return self

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Return static layout metadata with optionally replaced dtypes."""

        return type(self)(slots=tuple(slot.to(device=device, dtype=dtype) for slot in self.slots))

    def detach(self) -> Self:
        """Return this tensor-free immutable layout."""

        return self

    def permute(self, permutation: Permutation) -> Self:
        """Return this particle-invariant static layout."""

        del permutation
        return self

    def compare(
        self,
        other: Self,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> tuple[bool, dict[str, float]]:
        """Compare ordered slot metadata exactly."""

        del atol, rtol
        if type(self) is not type(other) or len(self.slots) != len(other.slots):
            return False, {"max_abs_error": float("inf")}
        close = all(left.compare(right)[0] for left, right in zip(self.slots, other.slots))
        return close, {"max_abs_error": 0.0 if close else float("inf")}


@dataclass(frozen=True, eq=False, kw_only=True)
class ParameterBinding:
    """Bind a layout to an ordered tuple of direct live parameters.

    Parameters are retained by identity.  No names, module-member paths, or
    reconstruction metadata are stored.
    """

    layout: ParameterLayout
    parameters: tuple[torch.nn.Parameter, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        self.validate()

    def validate(self) -> Self:
        """Validate direct parameter types and their exact layout agreement."""

        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("ParameterBinding.layout must be a ParameterLayout")
        self.layout.validate()
        if len(self.parameters) != len(self.layout.slots):
            raise ValueError(
                "ParameterBinding.parameters must have one direct reference per layout slot"
            )
        for slot, parameter in zip(self.layout.slots, self.parameters):
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError("ParameterBinding.parameters must contain direct torch.nn.Parameter references")
            if not parameter.requires_grad:
                raise ValueError("ParameterBinding parameters must require gradients")
            if tuple(parameter.shape) != slot.shape:
                raise ValueError(
                    f"ParameterBinding slot {slot.ordinal} expected shape {slot.shape}, "
                    f"got {tuple(parameter.shape)}"
                )
            if parameter.numel() != slot.numel:
                raise ValueError(
                    f"ParameterBinding slot {slot.ordinal} expected numel {slot.numel}, "
                    f"got {parameter.numel()}"
                )
            if parameter.dtype != slot.dtype:
                raise ValueError(
                    f"ParameterBinding slot {slot.ordinal} expected dtype {slot.dtype}, "
                    f"got {parameter.dtype}"
                )
        return self

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Validate a no-op materialization request without replacing references.

        Moving or casting a live parameter is model-owned.  This method returns
        the same binding only when every requested target already matches.
        """

        target_device = None if device is None else torch.device(device)
        for parameter in self.parameters:
            if target_device is not None and parameter.device != target_device:
                raise ValueError("ParameterBinding.to cannot move direct live parameter references")
            if dtype is not None and parameter.dtype != dtype:
                raise ValueError("ParameterBinding.to cannot cast direct live parameter references")
        return self

    def detach(self) -> Self:
        """Reject detachment because it would stop being a live binding."""

        raise RuntimeError("ParameterBinding cannot detach direct live parameter references")

    def permute(self, permutation: Permutation) -> Self:
        """Return this particle-invariant live binding."""

        del permutation
        return self

    def compare(
        self,
        other: Self,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> tuple[bool, dict[str, float]]:
        """Compare layout metadata and parameter-reference order exactly."""

        del atol, rtol
        if type(self) is not type(other) or not self.layout.compare(other.layout)[0]:
            return False, {"max_abs_error": float("inf")}
        close = len(self.parameters) == len(other.parameters) and all(
            left is right for left, right in zip(self.parameters, other.parameters)
        )
        return close, {"max_abs_error": 0.0 if close else float("inf")}


class ParameterScore(ABC):
    """Nominal capability for typed real-logabs parameter scores."""

    @abstractmethod
    def validate(self, *, sample_shape: tuple[int, ...] | None = None) -> Self:
        """Validate the semantic score payload."""

    @abstractmethod
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move the score payload to a device or dtype."""

    @abstractmethod
    def detach(self) -> Self:
        """Detach the score payload from autograd."""

    @abstractmethod
    def permute(self, permutation: Permutation) -> Self:
        """Apply the semantic particle permutation."""

    @abstractmethod
    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare two score payloads."""


@dataclass(frozen=True, eq=False, kw_only=True)
class MaterializedParameterLogScores(ParameterScore):
    """Store raw, uncentered real-logabs scores in parameter-shaped blocks.

    Block ``i`` has exact shape
    ``[*sample_shape, *layout.slots[i].shape]``.
    """

    layout: ParameterLayout
    blocks: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        self.validate()

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Return the shared leading sample shape.

        An empty layout has no tensor from which to infer a sample shape and
        therefore reports ``()``; callers can still validate an explicit
        shape with :meth:`validate`.
        """

        if not self.blocks:
            return ()
        slot_shape = self.layout.slots[0].shape
        prefix_ndim = self.blocks[0].ndim - len(slot_shape)
        return tuple(self.blocks[0].shape[:prefix_ndim])

    @property
    def device(self) -> torch.device | None:
        """Return the shared block device, or ``None`` for an empty layout."""

        return None if not self.blocks else self.blocks[0].device

    def validate(self, *, sample_shape: tuple[int, ...] | None = None) -> Self:
        """Validate layout order and every sample-plus-parameter block shape."""

        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("MaterializedParameterLogScores.layout must be a ParameterLayout")
        self.layout.validate()
        if len(self.blocks) != len(self.layout.slots):
            raise ValueError(
                "MaterializedParameterLogScores.blocks must have one block per layout slot"
            )
        expected_sample_shape = None if sample_shape is None else tuple(sample_shape)
        inferred_sample_shape: tuple[int, ...] | None = expected_sample_shape
        shared_device: torch.device | None = None
        for slot, block in zip(self.layout.slots, self.blocks):
            if not isinstance(block, torch.Tensor):
                raise TypeError("MaterializedParameterLogScores.blocks must contain torch.Tensor values")
            if block.ndim < len(slot.shape):
                raise ValueError(
                    f"Parameter score block {slot.ordinal} cannot end with parameter shape {slot.shape}"
                )
            prefix_ndim = block.ndim - len(slot.shape)
            block_sample_shape = tuple(block.shape[:prefix_ndim])
            parameter_shape = tuple(block.shape[prefix_ndim:])
            if parameter_shape != slot.shape:
                raise ValueError(
                    f"Parameter score block {slot.ordinal} expected trailing shape {slot.shape}, "
                    f"got {parameter_shape}"
                )
            if inferred_sample_shape is None:
                inferred_sample_shape = block_sample_shape
            elif block_sample_shape != inferred_sample_shape:
                raise ValueError(
                    f"Parameter score block {slot.ordinal} expected sample shape "
                    f"{inferred_sample_shape}, got {block_sample_shape}"
                )
            if block.dtype != slot.dtype:
                raise ValueError(
                    f"Parameter score block {slot.ordinal} expected dtype {slot.dtype}, got {block.dtype}"
                )
            if shared_device is None:
                shared_device = block.device
            elif block.device != shared_device:
                raise ValueError("Materialized parameter score blocks must share one device")
        return self

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move every score block and keep layout dtype metadata aligned."""

        return type(self)(
            layout=self.layout.to(device=device, dtype=dtype),
            blocks=tuple(block.to(device=device, dtype=dtype) for block in self.blocks),
        )

    def detach(self) -> Self:
        """Detach every materialized score block."""

        return type(self)(layout=self.layout, blocks=tuple(block.detach() for block in self.blocks))

    def permute(self, permutation: Permutation) -> Self:
        """Clone particle-invariant parameter score blocks."""

        del permutation
        return type(self)(layout=self.layout, blocks=tuple(block.clone() for block in self.blocks))

    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare layout metadata exactly and score blocks numerically."""

        if type(self) is not type(other) or not self.layout.compare(other.layout)[0]:
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks(self.blocks, other.blocks, atol=atol, rtol=rtol)


@dataclass(frozen=True, eq=False, kw_only=True)
class WavefunctionPacket(ABC):
    """Nominal base for frozen packets that compose a value output."""

    output: WavefunctionOutput

    @abstractmethod
    def validate(self) -> Self:
        """Validate the packet and all exact payload fields."""

    @abstractmethod
    def as_output(self) -> WavefunctionOutput:
        """Return the ordinary compatibility value view."""

    @abstractmethod
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move the packet and every exact payload."""

    @abstractmethod
    def detach(self) -> Self:
        """Detach the packet and every exact payload."""

    @abstractmethod
    def permute(self, permutation: Permutation) -> Self:
        """Apply the packet's semantic particle permutation."""

    @abstractmethod
    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare the value view and every exact payload."""


@dataclass(frozen=True, eq=False, kw_only=True)
class CoordinateForwardPacket(WavefunctionPacket):
    """Compose an ordinary value output with its coordinate log-gradient."""

    coordinates: CoordinateLogGradient

    def __post_init__(self) -> None:
        self.validate()

    def __getstate__(self):
        """Reject persistence while the value or derivative graph is live."""

        _reject_graph_bearing_output(self.output)
        if self.coordinates.values.requires_grad:
            raise RuntimeError("graph-bearing CoordinateForwardPacket cannot be serialized")
        return self.__dict__

    def validate(self) -> Self:
        """Validate value/coordinate sample, device, and dtype agreement."""

        if not isinstance(self.output, WavefunctionOutput):
            raise TypeError("CoordinateForwardPacket.output must be a WavefunctionOutput")
        if not isinstance(self.coordinates, CoordinateLogGradient):
            raise TypeError("CoordinateForwardPacket.coordinates must be a CoordinateLogGradient")
        self.output.validate()
        self.coordinates.validate(sample_shape=tuple(self.output.logabs.shape))
        if self.coordinates.device != self.output.logabs.device:
            raise ValueError("CoordinateForwardPacket output and coordinates must share one device")
        if self.coordinates.dtype != self.output.logabs.dtype:
            raise ValueError("CoordinateForwardPacket output and coordinates must share one dtype")
        return self

    def as_output(self) -> WavefunctionOutput:
        """Return the exact composed compatibility output."""

        return self.output

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move both the value view and coordinate payload."""

        return type(self)(
            output=self.output.to(device=device, dtype=dtype),
            coordinates=self.coordinates.to(device=device, dtype=dtype),
        )

    def detach(self) -> Self:
        """Detach both the value view and coordinate payload."""

        return type(self)(output=_detach_output(self.output), coordinates=self.coordinates.detach())

    def permute(self, permutation: Permutation) -> Self:
        """Apply fermionic value and electron-gradient permutation semantics."""

        return type(self)(
            output=self.output.permute(permutation),
            coordinates=self.coordinates.permute(permutation),
        )

    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare the value output and coordinate payload."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf")}
        return _combine_comparisons(
            self.output.compare(other.output, atol=atol, rtol=rtol),
            self.coordinates.compare(other.coordinates, atol=atol, rtol=rtol),
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class ParameterScoreForwardPacket(WavefunctionPacket):
    """Compose an ordinary value output with materialized parameter scores."""

    parameter_scores: MaterializedParameterLogScores

    def __post_init__(self) -> None:
        self.validate()

    def __getstate__(self):
        """Reject persistence while the value or score graph is live."""

        _reject_graph_bearing_output(self.output)
        if any(block.requires_grad for block in self.parameter_scores.blocks):
            raise RuntimeError("graph-bearing ParameterScoreForwardPacket cannot be serialized")
        return self.__dict__

    def validate(self) -> Self:
        """Validate value/score sample shape and device agreement."""

        if not isinstance(self.output, WavefunctionOutput):
            raise TypeError("ParameterScoreForwardPacket.output must be a WavefunctionOutput")
        if not isinstance(self.parameter_scores, MaterializedParameterLogScores):
            raise TypeError(
                "ParameterScoreForwardPacket.parameter_scores must be "
                "MaterializedParameterLogScores"
            )
        self.output.validate()
        self.parameter_scores.validate(sample_shape=tuple(self.output.logabs.shape))
        if (
            self.parameter_scores.device is not None
            and self.parameter_scores.device != self.output.logabs.device
        ):
            raise ValueError("ParameterScoreForwardPacket output and scores must share one device")
        return self

    def as_output(self) -> WavefunctionOutput:
        """Return the exact composed compatibility output."""

        return self.output

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move both the value view and materialized scores."""

        return type(self)(
            output=self.output.to(device=device, dtype=dtype),
            parameter_scores=self.parameter_scores.to(device=device, dtype=dtype),
        )

    def detach(self) -> Self:
        """Detach both the value view and every score block."""

        return type(self)(
            output=_detach_output(self.output),
            parameter_scores=self.parameter_scores.detach(),
        )

    def permute(self, permutation: Permutation) -> Self:
        """Apply fermionic value semantics and invariant score semantics."""

        return type(self)(
            output=self.output.permute(permutation),
            parameter_scores=self.parameter_scores.permute(permutation),
        )

    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare the value output and materialized score payload."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf")}
        return _combine_comparisons(
            self.output.compare(other.output, atol=atol, rtol=rtol),
            self.parameter_scores.compare(other.parameter_scores, atol=atol, rtol=rtol),
        )


def _detach_output(output: WavefunctionOutput) -> WavefunctionOutput:
    """Detach explicit value fields and tensor-valued diagnostic ``aux`` entries."""

    return WavefunctionOutput(
        logabs=output.logabs.detach(),
        sign=output.sign.detach(),
        phase=None if output.phase is None else output.phase.detach(),
        aux={
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in output.aux.items()
        },
    )


def _reject_graph_bearing_output(output: WavefunctionOutput) -> None:
    """Reject serializing any direct tensor in a live output graph."""

    values = (output.logabs, output.sign, output.phase, *output.aux.values())
    if any(isinstance(value, torch.Tensor) and value.requires_grad for value in values):
        raise RuntimeError("graph-bearing wavefunction packet cannot be serialized")


def _combine_comparisons(
    *comparisons: tuple[bool, dict[str, float]],
) -> tuple[bool, dict[str, float]]:
    """Combine exact typed comparison results."""

    return all(close for close, _ in comparisons), {
        "max_abs_error": max((metrics["max_abs_error"] for _, metrics in comparisons), default=0.0)
    }


__all__ = [
    "CoordinateForwardPacket",
    "CoordinateLogGradient",
    "MaterializedParameterLogScores",
    "ParameterBinding",
    "ParameterLayout",
    "ParameterScore",
    "ParameterScoreForwardPacket",
    "ParameterSlot",
    "WavefunctionPacket",
]
