"""Typed step-level request, derivative-payload, and producer contracts.

This module owns the single-pass VMC step request/produce seam that sits above
pure wavefunction forward requests (:mod:`tpen.nn.forward`) and below concrete
update algorithms.  Requests use nominal double dispatch: the trainer resolves
a method's :class:`VMCStepRequest` and calls ``request.produce(producer)``,
which delegates to exactly one typed method on a narrow producer protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    WavefunctionOutput,
    slice_flat_batch,
)
from tpen.data.equivariant_state import compare_tensor_blocks
from tpen.data.permutation import Permutation
from tpen.dependencies import require_torch
from tpen.nn.forward import (
    DenseFactorBlock,
    KroneckerFactorBlock,
    ParameterScoreRequest,
)
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.training.update import (
    AutogradUpdateInput,
    ScoreUpdateInput,
    VMCStepData,
    VMCUpdateMethod,
)
from tpen.training.vmc import VMCObjectiveResult

torch = require_torch(feature="VMC step requests")

InputT = TypeVar("InputT")


def _validate_step(step: int) -> None:
    """Validate that a VMC step index is a non-negative integer."""

    if type(step) is not int or step < 0:
        raise ValueError("VMC update step must be a non-negative integer")


def _validate_positive_optional_int(value: int | None, *, field_name: str) -> None:
    """Validate that an optional chunk bound is ``None`` or a positive ``int``."""

    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 1
    ):
        raise ValueError(
            f"{field_name} must be a positive integer or None, got {value!r}"
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class MaterializedLocalEnergyDerivatives:
    """Store raw, uncentered local-energy parameter derivatives in layout order.

    Block ``i`` stores :math:`D_i(R_s) = \\partial_{\\theta_i} E_L(R_s)` with
    exact shape ``[*sample_shape, *layout.slots[i].shape]``.

    Parameters
    ----------
    layout : ParameterLayout
        Ordered static metadata for the trainable parameter slots.
    blocks : tuple of torch.Tensor
        One derivative tensor per slot in ``layout``.  Direct construction
        permits graph-bearing tensors so detachment and serialization guards
        remain independently testable; :func:`evaluate_hamiltonian_tangent_step`
        always returns detached blocks.
    """

    layout: ParameterLayout
    blocks: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        self.validate()

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Return the shared leading sample shape, or ``()`` for an empty layout."""

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
            raise TypeError(
                "MaterializedLocalEnergyDerivatives.layout must be a ParameterLayout"
            )
        self.layout.validate()
        if len(self.blocks) != len(self.layout.slots):
            raise ValueError(
                "MaterializedLocalEnergyDerivatives.blocks must have one block per layout slot"
            )
        expected_sample_shape = None if sample_shape is None else tuple(sample_shape)
        inferred_sample_shape: tuple[int, ...] | None = expected_sample_shape
        shared_device: torch.device | None = None
        for slot, block in zip(self.layout.slots, self.blocks, strict=True):
            if not isinstance(block, torch.Tensor):
                raise TypeError(
                    "MaterializedLocalEnergyDerivatives.blocks must contain torch.Tensor values"
                )
            if block.ndim < len(slot.shape):
                raise ValueError(
                    f"Local energy derivative block {slot.ordinal} cannot end with "
                    f"parameter shape {slot.shape}"
                )
            prefix_ndim = block.ndim - len(slot.shape)
            block_sample_shape = tuple(block.shape[:prefix_ndim])
            parameter_shape = tuple(block.shape[prefix_ndim:])
            if parameter_shape != slot.shape:
                raise ValueError(
                    f"Local energy derivative block {slot.ordinal} expected trailing shape "
                    f"{slot.shape}, got {parameter_shape}"
                )
            if inferred_sample_shape is None:
                inferred_sample_shape = block_sample_shape
            elif block_sample_shape != inferred_sample_shape:
                raise ValueError(
                    f"Local energy derivative block {slot.ordinal} expected sample shape "
                    f"{inferred_sample_shape}, got {block_sample_shape}"
                )
            if block.dtype != slot.dtype:
                raise ValueError(
                    f"Local energy derivative block {slot.ordinal} expected dtype "
                    f"{slot.dtype}, got {block.dtype}"
                )
            if shared_device is None:
                shared_device = block.device
            elif block.device != shared_device:
                raise ValueError(
                    "Materialized local energy derivative blocks must share one device"
                )
        return self

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Move every derivative block and keep layout dtype metadata aligned."""

        return type(self)(
            layout=self.layout.to(device=device, dtype=dtype),
            blocks=tuple(block.to(device=device, dtype=dtype) for block in self.blocks),
        )

    def detach(self) -> Self:
        """Detach every materialized derivative block."""

        return type(self)(
            layout=self.layout,
            blocks=tuple(block.detach() for block in self.blocks),
        )

    def permute(self, permutation: Permutation) -> Self:
        """Clone particle-invariant local-energy derivative blocks.

        Under a particle-symmetric Hamiltonian and fermionic wavefunction
        :math:`|\\psi|`, both :math:`E_L(R)` and its parameter derivatives
        :math:`\\partial_\\theta E_L(R)` are invariant under electron
        permutations, mirroring :meth:`WavefunctionOutput.permute`.
        """

        del permutation
        return type(self)(
            layout=self.layout,
            blocks=tuple(block.clone() for block in self.blocks),
        )

    def compare(
        self,
        other: Self,
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare layout metadata exactly and derivative blocks numerically."""

        if type(self) is not type(other) or not self.layout.compare(other.layout)[0]:
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks(self.blocks, other.blocks, atol=atol, rtol=rtol)

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization while any derivative block carries a live graph."""

        if any(block.requires_grad for block in self.blocks):
            raise RuntimeError(
                "graph-bearing MaterializedLocalEnergyDerivatives cannot be serialized"
            )
        return self.__dict__


@dataclass(frozen=True, kw_only=True)
class HamiltonianTangentUpdateInput(VMCStepData):
    """Typed update input carrying materialized scores and local-energy tangents.

    Parameters
    ----------
    step : int
        Zero-based training iteration index.
    parameter_scores : MaterializedParameterLogScores
        Raw uncentered parameter scores :math:`O_j(R_s) = \\partial_{\\theta_j} \\log|\\psi(R_s)|`.
    local_energy_derivatives : MaterializedLocalEnergyDerivatives
        Raw uncentered local-energy derivatives :math:`D_j(R_s) = \\partial_{\\theta_j} E_L(R_s)`.
    parameter_binding : ParameterBinding
        Direct live parameter references aligned with both materialized payloads.
    """

    step: int
    parameter_scores: MaterializedParameterLogScores
    local_energy_derivatives: MaterializedLocalEnergyDerivatives
    parameter_binding: ParameterBinding

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> Self:
        """Validate scores, local-energy derivatives, and live parameter binding."""

        super().validate()
        _validate_step(self.step)
        if not isinstance(self.parameter_scores, MaterializedParameterLogScores):
            raise TypeError(
                "HamiltonianTangentUpdateInput.parameter_scores must be "
                "MaterializedParameterLogScores"
            )
        if not isinstance(
            self.local_energy_derivatives, MaterializedLocalEnergyDerivatives
        ):
            raise TypeError(
                "HamiltonianTangentUpdateInput.local_energy_derivatives must be "
                "MaterializedLocalEnergyDerivatives"
            )
        if not isinstance(self.parameter_binding, ParameterBinding):
            raise TypeError(
                "HamiltonianTangentUpdateInput.parameter_binding must be a ParameterBinding"
            )
        self.parameter_binding.validate()
        expected_sample_shape = tuple(self.wavefunction.logabs.shape)
        self.parameter_scores.validate(sample_shape=expected_sample_shape)
        self.local_energy_derivatives.validate(sample_shape=expected_sample_shape)
        if not self.parameter_scores.layout.compare(self.parameter_binding.layout)[0]:
            raise ValueError(
                "HamiltonianTangentUpdateInput parameter scores and binding layouts do not match"
            )
        if not self.local_energy_derivatives.layout.compare(
            self.parameter_binding.layout
        )[0]:
            raise ValueError(
                "HamiltonianTangentUpdateInput local energy derivatives and binding "
                "layouts do not match"
            )
        if self.parameter_binding.parameters:
            binding_device = self.parameter_binding.parameters[0].device
            if any(
                parameter.device != binding_device
                for parameter in self.parameter_binding.parameters
            ):
                raise ValueError(
                    "HamiltonianTangentUpdateInput parameter binding parameters must "
                    "share one device"
                )
            if self.parameter_scores.device != binding_device:
                raise ValueError(
                    "HamiltonianTangentUpdateInput parameter scores and binding must "
                    "share one device"
                )
            if self.local_energy_derivatives.device != binding_device:
                raise ValueError(
                    "HamiltonianTangentUpdateInput local energy derivatives and binding "
                    "must share one device"
                )
            if self.wavefunction.logabs.device != binding_device:
                raise ValueError(
                    "HamiltonianTangentUpdateInput wavefunction and binding must "
                    "share one device"
                )
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization, prioritizing graph-bearing errors over live-binding errors."""

        super().__getstate__()
        if any(block.requires_grad for block in self.parameter_scores.blocks):
            raise RuntimeError(
                "graph-bearing HamiltonianTangentUpdateInput cannot be serialized"
            )
        if any(block.requires_grad for block in self.local_energy_derivatives.blocks):
            raise RuntimeError(
                "graph-bearing HamiltonianTangentUpdateInput cannot be serialized"
            )
        raise RuntimeError("live HamiltonianTangentUpdateInput cannot be serialized")


@dataclass(frozen=True, kw_only=True)
class StructuredCurvatureUpdateInput(VMCStepData):
    """Typed update input carrying detached KFAC curvature factors and half-gradient.

    Parameters
    ----------
    step : int
        Zero-based training iteration index.
    parameter_binding : ParameterBinding
        Direct live parameter references aligned with every curvature block.
    solve_dtype : torch.dtype
        Real floating dtype (``torch.float32`` or ``torch.float64``) of all
        curvature factors and ``energy_half_gradient`` tensors.
    kronecker_blocks : tuple of KroneckerFactorBlock
        Detached uncentered KFAC-expand Kronecker blocks ``(A, G)``.
    dense_blocks : tuple of DenseFactorBlock
        Detached centered empirical score-covariance blocks ``F``.
    energy_half_gradient : tuple of torch.Tensor
        Detached ``solve_dtype`` energy half-gradient blocks (``scale = 1.0``)
        aligned with ``parameter_binding.layout.slots``.
    n_finite_samples : int
        Count of pinned finite local-energy samples used to form the curvature
        factors and half-gradient.
    curvature_available : bool, optional
        ``True`` for normal non-degenerate steps; ``False`` only for degenerate
        skip steps (``n_electrons == 0`` or ``n_finite_samples < 2``) where no
        curvature backward pass ran.
    """

    step: int
    parameter_binding: ParameterBinding
    solve_dtype: torch.dtype
    kronecker_blocks: tuple[KroneckerFactorBlock, ...]
    dense_blocks: tuple[DenseFactorBlock, ...]
    energy_half_gradient: tuple[torch.Tensor, ...]
    n_finite_samples: int
    curvature_available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "kronecker_blocks", tuple(self.kronecker_blocks))
        object.__setattr__(self, "dense_blocks", tuple(self.dense_blocks))
        object.__setattr__(self, "energy_half_gradient", tuple(self.energy_half_gradient))
        self.validate()

    def validate(self) -> Self:
        """Validate step data, binding, curvature block partition, and half-gradient."""

        super().validate()
        _validate_step(self.step)
        if not isinstance(self.parameter_binding, ParameterBinding):
            raise TypeError(
                "StructuredCurvatureUpdateInput.parameter_binding must be a ParameterBinding"
            )
        self.parameter_binding.validate()
        if not isinstance(self.solve_dtype, torch.dtype) or self.solve_dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                "StructuredCurvatureUpdateInput.solve_dtype must be torch.float32 or torch.float64"
            )
        if (
            type(self.n_finite_samples) is not int
            or self.n_finite_samples < 0
            or self.n_finite_samples > self.batch.batch_size
        ):
            raise ValueError(
                "StructuredCurvatureUpdateInput.n_finite_samples must be in [0, batch.batch_size]"
            )
        if type(self.curvature_available) is not bool:
            raise TypeError("StructuredCurvatureUpdateInput.curvature_available must be a bool")

        if not self.curvature_available:
            if self.kronecker_blocks or self.dense_blocks or self.energy_half_gradient:
                raise ValueError(
                    "StructuredCurvatureUpdateInput with curvature_available=False must carry empty blocks"
                )
            return self

        slots = self.parameter_binding.layout.slots
        if not slots:
            raise ValueError(
                "StructuredCurvatureUpdateInput requires at least one trainable parameter"
            )
        if len(self.energy_half_gradient) != len(slots):
            raise ValueError(
                "StructuredCurvatureUpdateInput.energy_half_gradient must have one tensor per slot"
            )
        binding_device = self.parameter_binding.parameters[0].device
        if self.wavefunction.logabs.device != binding_device:
            raise ValueError(
                "StructuredCurvatureUpdateInput wavefunction and binding must share one device"
            )
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
                raise ValueError("KroneckerFactorBlock dtype/device disagrees with update input")
            for ordinal in block.parameter_ordinals:
                if ordinal >= len(slots):
                    raise ValueError(f"KroneckerFactorBlock ordinal {ordinal} out of range")
                if ordinal in covered:
                    raise ValueError(f"Duplicate curvature coverage of parameter ordinal {ordinal}")
                covered.add(ordinal)

        for block in self.dense_blocks:
            if not isinstance(block, DenseFactorBlock):
                raise TypeError("dense_blocks must contain DenseFactorBlock instances")
            block.validate()
            if block.F.dtype != self.solve_dtype or block.F.device != binding_device:
                raise ValueError("DenseFactorBlock dtype/device disagrees with update input")
            for ordinal in block.parameter_ordinals:
                if ordinal >= len(slots):
                    raise ValueError(f"DenseFactorBlock ordinal {ordinal} out of range")
                if ordinal in covered:
                    raise ValueError(f"Duplicate curvature coverage of parameter ordinal {ordinal}")
                covered.add(ordinal)

        missing = sorted(set(range(len(slots))) - covered)
        if missing:
            raise ValueError(
                f"StructuredCurvatureUpdateInput does not exhaustively cover parameter ordinals: missing {missing}"
            )
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Unconditionally refuse serialization, prioritizing graph-bearing errors first."""

        super().__getstate__()
        if (
            any(b.A.requires_grad or b.G.requires_grad for b in self.kronecker_blocks)
            or any(b.F.requires_grad for b in self.dense_blocks)
            or any(g.requires_grad for g in self.energy_half_gradient)
        ):
            raise RuntimeError(
                "graph-bearing StructuredCurvatureUpdateInput cannot be serialized"
            )
        raise RuntimeError("live StructuredCurvatureUpdateInput cannot be serialized")


@dataclass(frozen=True, kw_only=True)
class ProducedVMCStep(Generic[InputT]):
    """Frozen ephemeral carrier returned by :meth:`VMCStepRequest.produce`.

    Parameters
    ----------
    update_input : InputT
        Exact typed input consumed by the update method's ``update()`` call.
    objective : VMCObjectiveResult
        Canonical VMC objective and scalar metrics for the iteration.
    local_energy : torch.Tensor
        Per-sample total local energy (detached in normal trainer execution).
    term_energies : dict of str to torch.Tensor or None
        Optional per-term local-energy mapping, or ``None`` when
        ``return_terms=False``.
    wavefunction_output : WavefunctionOutput
        Wavefunction output for the iteration.  For
        :class:`AutogradStepRequest`, this is the exact live object carried by
        ``update_input.wavefunction``; for score and tangent requests, every
        tensor inside this record and ``update_input`` is detached.
    """

    update_input: InputT
    objective: VMCObjectiveResult
    local_energy: torch.Tensor
    term_energies: dict[str, torch.Tensor] | None
    wavefunction_output: WavefunctionOutput

    def __post_init__(self) -> None:
        if not isinstance(self.objective, VMCObjectiveResult):
            raise TypeError(
                "ProducedVMCStep.objective must be a VMCObjectiveResult, "
                f"got {type(self.objective).__name__}"
            )
        if not isinstance(self.local_energy, torch.Tensor):
            raise TypeError(
                "ProducedVMCStep.local_energy must be a torch.Tensor, "
                f"got {type(self.local_energy).__name__}"
            )
        if self.term_energies is not None:
            if not isinstance(self.term_energies, dict):
                raise TypeError(
                    "ProducedVMCStep.term_energies must be a dict or None, "
                    f"got {type(self.term_energies).__name__}"
                )
            for name, value in self.term_energies.items():
                if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                    raise TypeError(
                        "ProducedVMCStep.term_energies must map str names to torch.Tensor values"
                    )
        if not isinstance(self.wavefunction_output, WavefunctionOutput):
            raise TypeError(
                "ProducedVMCStep.wavefunction_output must be a WavefunctionOutput, "
                f"got {type(self.wavefunction_output).__name__}"
            )

    def __getstate__(self) -> dict[str, Any]:
        """Unconditionally refuse serialization of a live ephemeral step carrier."""

        raise RuntimeError("live ProducedVMCStep cannot be serialized")


class VMCStepRequest(Generic[InputT], ABC):
    """Nominal strategy declaring what data a VMC training step must produce."""

    @abstractmethod
    def produce(self, producer: "VMCStepProducer", /) -> ProducedVMCStep[InputT]:
        """Produce the typed step record through direct double dispatch."""

    @abstractmethod
    def describe_label(self) -> str:
        """Return the canonical sink-safe description label for this request."""


@dataclass(frozen=True, kw_only=True)
class AutogradStepRequest(VMCStepRequest[AutogradUpdateInput]):
    """Request the historical live autograd objective and re-evaluation input."""

    def produce(
        self,
        producer: "AutogradStepProducer",
        /,
    ) -> ProducedVMCStep[AutogradUpdateInput]:
        """Delegate to the producer's autograd step implementation."""

        return producer.produce_autograd_step(self)

    def describe_label(self) -> str:
        """Return the historical ``"value"`` forward-request label."""

        return "value"


@dataclass(frozen=True, kw_only=True)
class ParameterScoreStepRequest(VMCStepRequest[ScoreUpdateInput]):
    """Request a detached parameter-score step produced from a forward request.

    Parameters
    ----------
    forward_request : ParameterScoreRequest
        Pure wavefunction forward request used to compute parameter scores.
    """

    forward_request: ParameterScoreRequest

    def __post_init__(self) -> None:
        if not isinstance(self.forward_request, ParameterScoreRequest):
            raise TypeError(
                "ParameterScoreStepRequest.forward_request must be a ParameterScoreRequest, "
                f"got {type(self.forward_request).__name__}"
            )

    def produce(
        self,
        producer: "ParameterScoreStepProducer",
        /,
    ) -> ProducedVMCStep[ScoreUpdateInput]:
        """Delegate to the producer's parameter-score step implementation."""

        return producer.produce_parameter_score_step(self)

    def describe_label(self) -> str:
        """Return the qualified name of the wrapped forward request."""

        return (
            f"{type(self.forward_request).__module__}."
            f"{type(self.forward_request).__qualname__}"
        )


class HamiltonianTangentStepRequest(VMCStepRequest[InputT], ABC):
    """Abstract nominal strategy base for Hamiltonian-tangent step requests."""


@dataclass(frozen=True, kw_only=True)
class MaterializedHamiltonianTangentStepRequest(
    HamiltonianTangentStepRequest[HamiltonianTangentUpdateInput]
):
    """Request materialized parameter scores and local-energy derivatives in one pass.

    Parameters
    ----------
    chunk_size : int or None, optional
        Positive sample chunk size bounding peak autograd intermediate memory
        during the single-pass local-energy and reverse-mode tangent loop.
        ``None`` defers chunk sizing to the trainer's
        ``local_energy_chunk_size`` (or the full flattened batch when unset).

    Notes
    -----
    Memory envelope: ``chunk_size`` bounds **only** the peak autograd
    intermediate memory held during per-chunk differentiation.  The returned
    materialized payload always occupies :math:`O(2 N P)` memory across
    ``(parameter_scores, local_energy_derivatives)`` for :math:`N` flattened
    samples and :math:`P` trainable parameters.  Callers that require dense
    memory admission limits enforce those bounds at the consumer layer.
    """

    chunk_size: int | None = None

    def __post_init__(self) -> None:
        _validate_positive_optional_int(
            self.chunk_size,
            field_name="MaterializedHamiltonianTangentStepRequest.chunk_size",
        )

    def produce(
        self,
        producer: "HamiltonianTangentStepProducer",
        /,
    ) -> ProducedVMCStep[HamiltonianTangentUpdateInput]:
        """Delegate to the producer's materialized tangent step implementation."""

        return producer.produce_materialized_hamiltonian_tangent_step(self)

    def describe_label(self) -> str:
        """Return the qualified class label for this materialized tangent request."""

        return "tpen.training.step_requests.MaterializedHamiltonianTangentStepRequest"


@dataclass(frozen=True, kw_only=True)
class StructuredCurvatureStepRequest(VMCStepRequest[StructuredCurvatureUpdateInput]):
    """Request structured KFAC curvature factors and energy half-gradient in one forward.

    Parameters
    ----------
    solve_dtype : torch.dtype, optional
        Real floating dtype (``torch.float32`` or ``torch.float64``) used for
        curvature factor accumulation and energy half-gradient blocks.
    max_dense_block_numel : int, optional
        Maximum allowed scalar element count for any :class:`DenseFactorBlock`.
    """

    solve_dtype: torch.dtype = torch.float64
    max_dense_block_numel: int = 2048

    def __post_init__(self) -> None:
        if not isinstance(self.solve_dtype, torch.dtype) or self.solve_dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                "StructuredCurvatureStepRequest.solve_dtype must be torch.float32 or torch.float64"
            )
        if (
            isinstance(self.max_dense_block_numel, bool)
            or not isinstance(self.max_dense_block_numel, int)
            or self.max_dense_block_numel < 1
        ):
            raise ValueError(
                "StructuredCurvatureStepRequest.max_dense_block_numel must be a positive integer"
            )

    def produce(
        self,
        producer: "StructuredCurvatureStepProducer",
        /,
    ) -> ProducedVMCStep[StructuredCurvatureUpdateInput]:
        """Delegate to the producer's structured-curvature step implementation."""

        return producer.produce_structured_curvature_step(self)

    def describe_label(self) -> str:
        """Return the qualified class label for this structured-curvature request."""

        return "tpen.training.step_requests.StructuredCurvatureStepRequest"


@runtime_checkable
class AutogradStepProducer(Protocol):
    """Narrow method-only producer targeted by :class:`AutogradStepRequest`."""

    def produce_autograd_step(
        self,
        request: AutogradStepRequest,
    ) -> ProducedVMCStep[AutogradUpdateInput]:
        """Produce one live autograd step record."""
        ...


@runtime_checkable
class ParameterScoreStepProducer(Protocol):
    """Narrow method-only producer targeted by :class:`ParameterScoreStepRequest`."""

    def produce_parameter_score_step(
        self,
        request: ParameterScoreStepRequest,
    ) -> ProducedVMCStep[ScoreUpdateInput]:
        """Produce one detached parameter-score step record."""
        ...


@runtime_checkable
class HamiltonianTangentStepProducer(Protocol):
    """Narrow method-only producer targeted by materialized tangent requests."""

    def produce_materialized_hamiltonian_tangent_step(
        self,
        request: MaterializedHamiltonianTangentStepRequest,
    ) -> ProducedVMCStep[HamiltonianTangentUpdateInput]:
        """Produce one detached materialized Hamiltonian-tangent step record."""
        ...


@runtime_checkable
class StructuredCurvatureStepProducer(Protocol):
    """Narrow method-only producer targeted by :class:`StructuredCurvatureStepRequest`."""

    def produce_structured_curvature_step(
        self,
        request: StructuredCurvatureStepRequest,
    ) -> ProducedVMCStep[StructuredCurvatureUpdateInput]:
        """Produce one detached structured-curvature step record."""
        ...


@runtime_checkable
class VMCStepProducer(
    AutogradStepProducer,
    ParameterScoreStepProducer,
    HamiltonianTangentStepProducer,
    StructuredCurvatureStepProducer,
    Protocol,
):
    """Composite method-only step producer supporting all core step requests."""


def resolve_step_request(update_method: VMCUpdateMethod[Any]) -> VMCStepRequest[Any]:
    """Resolve and validate the single typed step request for an update method.

    Parameters
    ----------
    update_method : VMCUpdateMethod
        Update method whose :meth:`~tpen.training.update.VMCUpdateMethod.step_request`
        and legacy :meth:`~tpen.training.update.VMCUpdateMethod.forward_request`
        declarations are validated for consistency.

    Returns
    -------
    VMCStepRequest
        Validated nominal step request instance.

    Raises
    ------
    TypeError
        If ``update_method`` is not a :class:`VMCUpdateMethod`, or if
        ``step_request()`` does not return a :class:`VMCStepRequest`.
    ValueError
        If ``forward_request()`` returns a non-``None`` request that conflicts
        with ``step_request()`` (including a non-:class:`ParameterScoreRequest`
        when ``step_request()`` was overridden).
    """

    if not isinstance(update_method, VMCUpdateMethod):
        raise TypeError(
            "resolve_step_request requires a VMCUpdateMethod, "
            f"got {type(update_method).__name__}"
        )
    request = update_method.step_request()
    if not isinstance(request, VMCStepRequest):
        raise TypeError(
            "VMCUpdateMethod.step_request must return a VMCStepRequest, "
            f"got {type(request).__name__}"
        )
    legacy_request = update_method.forward_request()
    if legacy_request is not None:
        # Check the legacy request type BEFORE wrapping it so a conflicting
        # non-ParameterScoreRequest raises a ValueError conflict rather than a
        # TypeError from ParameterScoreStepRequest.__post_init__.
        if not isinstance(legacy_request, ParameterScoreRequest):
            raise ValueError(
                "VMCUpdateMethod declares conflicting step_request() and "
                "non-ParameterScoreRequest forward_request() "
                f"({type(legacy_request).__name__})"
            )
        expected = ParameterScoreStepRequest(forward_request=legacy_request)
        if request != expected:
            raise ValueError(
                "VMCUpdateMethod declares conflicting step_request() and "
                f"forward_request(): step_request() returned {request!r}, "
                f"but forward_request() implies {expected!r}"
            )
    return request


def _resolve_live_parameter_binding(model: Any) -> ParameterBinding:
    """Validate and return the model's live trainable-parameter binding."""

    binding = getattr(model, "parameter_binding", None)
    if not isinstance(binding, ParameterBinding):
        raise TypeError(
            "MaterializedHamiltonianTangentStepRequest requires model.parameter_binding "
            "to be a ParameterBinding"
        )
    try:
        binding.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError("parameter binding/layout is no longer valid") from exc
    if not binding.parameters:
        raise ValueError(
            "MaterializedHamiltonianTangentStepRequest requires at least one "
            "trainable parameter"
        )
    parameters_fn = getattr(model, "parameters", None)
    if not callable(parameters_fn):
        raise TypeError(
            "MaterializedHamiltonianTangentStepRequest requires model.parameters() "
            "to be callable"
        )
    current = tuple(parameter for parameter in parameters_fn() if parameter.requires_grad)
    if len(current) != len(binding.parameters) or not all(
        current_parameter is bound_parameter
        for current_parameter, bound_parameter in zip(
            current, binding.parameters, strict=True
        )
    ):
        raise ValueError(
            "parameter binding/layout mismatch, freezing, or reordering; "
            "refusing Hamiltonian tangent evaluation"
        )
    return binding


def _resolve_effective_chunk_size(
    *,
    chunk_size: int | None,
    trainer_chunk_size: int | None,
    batch_size: int,
) -> int:
    """Resolve the active chunk bound for materialized tangent evaluation."""

    _validate_positive_optional_int(chunk_size, field_name="chunk_size")
    _validate_positive_optional_int(trainer_chunk_size, field_name="trainer_chunk_size")
    if chunk_size is not None and trainer_chunk_size is not None:
        return min(chunk_size, trainer_chunk_size)
    if chunk_size is not None:
        return chunk_size
    if trainer_chunk_size is not None:
        return trainer_chunk_size
    return batch_size


def evaluate_hamiltonian_tangent_step(
    *,
    model: Any,
    hamiltonian_terms: Mapping[Any, Any] | Sequence[Any],
    batch: ElectronBatch,
    step: int = 0,
    chunk_size: int | None = None,
    trainer_chunk_size: int | None = None,
    return_terms: bool = False,
    local_energy_fn: Callable[..., Any] = local_energy,
) -> tuple[HamiltonianTangentUpdateInput, dict[str, torch.Tensor] | None]:
    """Evaluate local energy, scores, and local-energy tangents in one pass per chunk.

    Parameters
    ----------
    model : callable
        Wavefunction model exposing a live :attr:`parameter_binding` and
        ``parameters()``.
    hamiltonian_terms : Mapping or Sequence of HamiltonianTerm
        Hamiltonian contributions passed to ``local_energy_fn``.
    batch : ElectronBatch
        Electron configuration batch.  All leading sample axes are flattened to
        1D shape ``(batch.batch_size,)`` across every returned tensor and block.
    step : int, optional
        Zero-based iteration index stored on :class:`HamiltonianTangentUpdateInput`.
    chunk_size : int or None, optional
        Request-level positive sample chunk size.
    trainer_chunk_size : int or None, optional
        Trainer-level positive sample chunk size (combined with ``chunk_size``
        by taking the stricter ``min``).
    return_terms : bool, optional
        Whether to return the detached per-term local-energy mapping alongside
        the update input.
    local_energy_fn : callable, optional
        Local-energy evaluator seam injected by the trainer (defaults to
        :func:`tpen.physics.hamiltonian.local_energy`).

    Returns
    -------
    tuple of (HamiltonianTangentUpdateInput, dict or None)
        Detached materialized tangent update input and optional detached
        per-term local-energy mapping (``None`` when ``return_terms=False``).

    Notes
    -----
    This helper executes outside :class:`~tpen.nn.TPENWaveFunction`, so it does
    not trigger ``TPENWaveFunction._log_zeroed_score_parameters``, and it uses
    ``local_energy_fn`` (``_NAIVE_EVALUATOR`` by default) rather than
    :class:`~tpen.physics.hamiltonian.AnalyticCuspEvaluator`.
    """

    if torch.is_inference_mode_enabled():
        raise RuntimeError(
            "MaterializedHamiltonianTangentStepRequest is not supported in inference mode"
        )
    if not isinstance(batch, ElectronBatch):
        raise TypeError(
            "evaluate_hamiltonian_tangent_step requires an ElectronBatch, "
            f"got {type(batch).__name__}"
        )
    _validate_step(step)
    binding = _resolve_live_parameter_binding(model)

    flat = batch.flatten_samples()
    # Detach incoming coordinates so a pure coordinate-potential Hamiltonian
    # cannot spuriously mark local energy as requiring grad from batch inputs.
    flat = replace(flat, positions=flat.positions.detach())
    step_batch = (
        replace(batch, positions=batch.positions.detach())
        if batch.positions.requires_grad
        else batch
    )
    effective_chunk_size = _resolve_effective_chunk_size(
        chunk_size=chunk_size,
        trainer_chunk_size=trainer_chunk_size,
        batch_size=flat.batch_size,
    )

    parameters = binding.parameters
    if flat.batch_size == 0:
        empty_output = WavefunctionOutput(
            logabs=torch.empty((0,), device=flat.device, dtype=flat.dtype),
            sign=torch.ones((0,), device=flat.device, dtype=flat.dtype),
        )
        empty_local_energy = torch.empty((0,), device=flat.device, dtype=flat.dtype)
        # Allocate 0-row blocks directly from each live parameter so per-slot
        # dtype and device metadata match binding.layout exactly.
        empty_scores = MaterializedParameterLogScores(
            layout=binding.layout,
            blocks=tuple(
                parameter.new_empty((0, *parameter.shape)) for parameter in parameters
            ),
        )
        empty_derivatives = MaterializedLocalEnergyDerivatives(
            layout=binding.layout,
            blocks=tuple(
                parameter.new_empty((0, *parameter.shape)) for parameter in parameters
            ),
        )
        update_input = HamiltonianTangentUpdateInput(
            batch=step_batch,
            wavefunction=empty_output,
            local_energy=empty_local_energy,
            step=step,
            parameter_scores=empty_scores,
            local_energy_derivatives=empty_derivatives,
            parameter_binding=binding,
        )
        return update_input, ({} if return_terms else None)

    logabs_chunks: list[torch.Tensor] = []
    sign_chunks: list[torch.Tensor] = []
    total_chunks: list[torch.Tensor] = []
    term_chunks: dict[str, list[torch.Tensor]] = {}
    term_order: tuple[str, ...] | None = None
    score_block_chunks: list[list[torch.Tensor]] = [[] for _ in parameters]
    derivative_block_chunks: list[list[torch.Tensor]] = [[] for _ in parameters]

    for start in range(0, flat.batch_size, effective_chunk_size):
        chunk = slice_flat_batch(
            flat, start, min(start + effective_chunk_size, flat.batch_size)
        )
        chunk_count = chunk.batch_size
        with torch.enable_grad():
            # Always request return_terms=True internally so any kinetic term's
            # live WavefunctionOutput is reused without a duplicate forward pass.
            chunk_result = local_energy_fn(
                hamiltonian_terms,
                model,
                chunk,
                return_terms=True,
            )
            if not isinstance(chunk_result, LocalEnergyResult):
                raise TypeError(
                    "MaterializedHamiltonianTangentStepRequest requires "
                    "local_energy_fn(..., return_terms=True) to return "
                    f"LocalEnergyResult, got {type(chunk_result).__name__}"
                )
            if not isinstance(chunk_result.total, torch.Tensor):
                raise TypeError("local-energy chunk total must be a torch.Tensor")
            if tuple(chunk_result.total.shape) != (chunk_count,):
                raise ValueError(
                    "local-energy chunk total must have shape "
                    f"{(chunk_count,)}, got {tuple(chunk_result.total.shape)}"
                )
            if not isinstance(chunk_result.terms, Mapping):
                raise TypeError("local-energy chunk terms must be a mapping")
            chunk_term_keys = tuple(chunk_result.terms)
            if term_order is None:
                term_order = chunk_term_keys
            elif chunk_term_keys != term_order:
                raise ValueError(
                    "chunked Hamiltonian-tangent local-energy terms changed between chunks: "
                    f"expected {term_order!r}, got {chunk_term_keys!r}"
                )

            chunk_output = chunk_result.wavefunction_output
            if chunk_output is None:
                # Pure coordinate-potential Hamiltonians do not call the model
                # inside local_energy; evaluate it once for this chunk.
                chunk_output = model(chunk)
            if not isinstance(chunk_output, WavefunctionOutput):
                raise TypeError(
                    "MaterializedHamiltonianTangentStepRequest requires "
                    f"WavefunctionOutput, got {type(chunk_output).__name__}"
                )
            if tuple(chunk_output.logabs.shape) != (chunk_count,):
                raise ValueError(
                    "Hamiltonian-tangent wavefunction logabs must have shape "
                    f"{(chunk_count,)}, got {tuple(chunk_output.logabs.shape)}"
                )
            if chunk_output.phase is not None:
                raise ValueError(
                    "MaterializedHamiltonianTangentStepRequest does not support "
                    "complex wavefunctions (wavefunction_output.phase must be None)"
                )

            logabs_flat = chunk_output.logabs
            eloc_flat = chunk_result.total
            if not logabs_flat.requires_grad:
                raise RuntimeError(
                    "tangent request requires a differentiable logabs output"
                )

            per_sample_scores: list[list[torch.Tensor]] = [[] for _ in parameters]
            per_sample_derivatives: list[list[torch.Tensor]] = [[] for _ in parameters]
            eloc_requires_grad = bool(eloc_flat.requires_grad)

            for sample_index in range(chunk_count):
                has_more_samples = sample_index + 1 < chunk_count
                score_grads = torch.autograd.grad(
                    logabs_flat[sample_index],
                    parameters,
                    retain_graph=(eloc_requires_grad or has_more_samples),
                    create_graph=False,
                    allow_unused=True,
                )
                for param_index, (parameter, grad) in enumerate(
                    zip(parameters, score_grads, strict=True)
                ):
                    per_sample_scores[param_index].append(
                        parameter.new_zeros(parameter.shape)
                        if grad is None
                        else grad.detach()
                    )

                if eloc_requires_grad:
                    de_grads = torch.autograd.grad(
                        eloc_flat[sample_index],
                        parameters,
                        retain_graph=has_more_samples,
                        create_graph=False,
                        allow_unused=True,
                    )
                    for param_index, (parameter, grad) in enumerate(
                        zip(parameters, de_grads, strict=True)
                    ):
                        per_sample_derivatives[param_index].append(
                            parameter.new_zeros(parameter.shape)
                            if grad is None
                            else grad.detach()
                        )
                else:
                    for param_index, parameter in enumerate(parameters):
                        per_sample_derivatives[param_index].append(
                            parameter.new_zeros(parameter.shape)
                        )

            for param_index in range(len(parameters)):
                score_block_chunks[param_index].append(
                    torch.stack(per_sample_scores[param_index], dim=0).detach()
                )
                derivative_block_chunks[param_index].append(
                    torch.stack(per_sample_derivatives[param_index], dim=0).detach()
                )

            logabs_chunks.append(chunk_output.logabs.detach().reshape(-1))
            sign_chunks.append(chunk_output.sign.detach().reshape(-1))
            total_chunks.append(chunk_result.total.detach())
            if return_terms:
                for term_name, term_value in chunk_result.terms.items():
                    term_chunks.setdefault(term_name, []).append(term_value.detach())

        # Release all graph-bearing chunk references before starting the next chunk.
        del (
            chunk_result,
            chunk_output,
            logabs_flat,
            eloc_flat,
            per_sample_scores,
            per_sample_derivatives,
            chunk,
        )

    # Reassemble with empty aux (mirroring evaluate_local_energy_in_chunks) so
    # no per-chunk diagnostic tensor can retain a derivative graph.
    wavefunction_output = WavefunctionOutput(
        logabs=torch.cat(logabs_chunks, dim=0),
        sign=torch.cat(sign_chunks, dim=0),
    )
    total_local_energy = torch.cat(total_chunks, dim=0)
    assembled_terms = (
        {name: torch.cat(chunks, dim=0) for name, chunks in term_chunks.items()}
        if return_terms
        else None
    )
    parameter_scores = MaterializedParameterLogScores(
        layout=binding.layout,
        blocks=tuple(torch.cat(chunks, dim=0) for chunks in score_block_chunks),
    )
    local_energy_derivatives = MaterializedLocalEnergyDerivatives(
        layout=binding.layout,
        blocks=tuple(torch.cat(chunks, dim=0) for chunks in derivative_block_chunks),
    )
    update_input = HamiltonianTangentUpdateInput(
        batch=step_batch,
        wavefunction=wavefunction_output,
        local_energy=total_local_energy,
        step=step,
        parameter_scores=parameter_scores,
        local_energy_derivatives=local_energy_derivatives,
        parameter_binding=binding,
    )
    return update_input, assembled_terms


__all__ = [
    "AutogradStepProducer",
    "AutogradStepRequest",
    "HamiltonianTangentStepProducer",
    "HamiltonianTangentStepRequest",
    "HamiltonianTangentUpdateInput",
    "MaterializedHamiltonianTangentStepRequest",
    "MaterializedLocalEnergyDerivatives",
    "ParameterScoreStepProducer",
    "ParameterScoreStepRequest",
    "ProducedVMCStep",
    "StructuredCurvatureStepProducer",
    "StructuredCurvatureStepRequest",
    "StructuredCurvatureUpdateInput",
    "VMCStepProducer",
    "VMCStepRequest",
    "evaluate_hamiltonian_tangent_step",
    "resolve_step_request",
]
