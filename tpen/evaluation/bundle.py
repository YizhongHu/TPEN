"""Typed primitive outputs shared across evaluation components.

``EvaluationBundle`` is intentionally not a generic dict.

A field belongs here only if it is a reusable primitive output produced by
calculators and consumed by multiple summaries/tasks. Derived diagnostic metrics
do not belong in the bundle.

Good bundle fields include:

- generated configurations
- wavefunction values
- local-energy values
- derivative values
- electron-nucleus radial values
- transform comparison values
- trace records

Do not add fields such as ``cusp_even_slope``, ``c_minus_1_abs``,
``tail_outlier_count``, ``pfaffian_near_zero_count``, or ``feature_rms_q95``.
Those are summary outputs, not shared primitive calculator outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from tpen.data.batch import ElectronBatch
from tpen.data.batch.geometry import electron_nuclear_displacements
from tpen.data.equivariant_state import compare_tensor_blocks
from tpen.data.indices import permutation_index, permute_particle_axis
from tpen.data.permutation import Permutation
from tpen.evaluation.trajectory_records import TrajectoryRecordArtifact
from tpen.physics.receipt import ExecutionReceipt
from tpen.trace import Trace


@dataclass(frozen=True)
class GeneratedConfigurations:
    """Electron configurations produced by an evaluation generator.

    Metadata is bookkeeping only. Scientific quantities computed from a model
    or Hamiltonian belong in calculator outputs, not in the metadata mapping.
    A streamed trajectory artifact is carried through its dedicated typed field
    so it cannot masquerade as bookkeeping or an arbitrary dictionary.
    """

    batch: ElectronBatch
    metadata: Mapping[str, Any]
    trajectory_records: TrajectoryRecordArtifact | None = None


@dataclass(frozen=True)
class WavefunctionValues:
    """Wavefunction values evaluated on generated configurations."""

    logabs: torch.Tensor
    sign: torch.Tensor
    components: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class LocalEnergyValues:
    """Local-energy values evaluated on generated configurations.

    ``receipt`` is the declared execution receipt of the evaluator that
    actually produced ``local_energy``, when that evaluator records one (see
    `tpen.physics.hamiltonian.RecordingLocalEnergyEvaluator`). ``None`` when
    the producing evaluator does not record a receipt.
    """

    local_energy: torch.Tensor
    finite_mask: torch.Tensor
    term_energies: Mapping[str, torch.Tensor] | None = None
    receipt: ExecutionReceipt | None = None


@dataclass(frozen=True)
class DerivativeValues:
    """Radial derivative values used by geometry summaries."""

    radial_dlogabs: torch.Tensor
    r12: torch.Tensor
    direction_id: torch.Tensor
    antipodal_pair_id: torch.Tensor | None = None
    direction_sign: torch.Tensor | None = None


@dataclass(frozen=True)
class ElectronNucleusRadialValues:
    """Typed electron-nucleus radial log-amplitude derivatives.

    Every field has shape ``[batch, n_electrons, n_nuclei]``. Distances use
    the raw geometry and must therefore be strictly positive for this
    diagnostic. ``finite_mask`` records derivative availability explicitly;
    nonfinite derivative values are permitted only when marked unavailable.
    """

    distance: torch.Tensor
    radial_dlogabs: torch.Tensor
    finite_mask: torch.Tensor

    def validate(self, batch: ElectronBatch) -> "ElectronNucleusRadialValues":
        """Validate shape, geometry, device, dtype, and availability semantics."""

        flat = batch.flatten_samples()
        if flat.nuclear_positions is None or flat.nuclear_charges is None:
            raise ValueError("ElectronNucleusRadialValues requires batch nuclear positions and charges")
        expected_shape = (flat.batch_size, flat.n_electrons, flat.nuclear_positions.shape[-2])
        for name, value in (("distance", self.distance), ("radial_dlogabs", self.radial_dlogabs)):
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"ElectronNucleusRadialValues.{name} must have shape {expected_shape}")
            if value.device != flat.device or value.dtype != flat.dtype:
                raise ValueError(
                    f"ElectronNucleusRadialValues.{name} must match batch dtype/device"
                )
        if tuple(self.finite_mask.shape) != expected_shape or self.finite_mask.dtype != torch.bool:
            raise ValueError(
                "ElectronNucleusRadialValues.finite_mask must be bool with shape "
                f"{expected_shape}"
            )
        if self.finite_mask.device != flat.device:
            raise ValueError("ElectronNucleusRadialValues.finite_mask must match batch device")
        if not torch.isfinite(self.distance).all() or torch.any(self.distance <= 0):
            raise ValueError("ElectronNucleusRadialValues.distance must be finite and strictly positive")
        expected_distance = electron_nuclear_displacements(flat).norm(dim=-1)
        if not torch.equal(self.distance, expected_distance):
            raise ValueError("ElectronNucleusRadialValues.distance must equal raw batch geometry")
        if not torch.equal(self.finite_mask, torch.isfinite(self.radial_dlogabs)):
            raise ValueError(
                "ElectronNucleusRadialValues.finite_mask must equal isfinite(radial_dlogabs)"
            )
        return self

    def permute(self, permutation: Permutation) -> "ElectronNucleusRadialValues":
        """Apply an electron permutation to every electron-indexed field."""

        if len(permutation) != self.distance.shape[1]:
            raise ValueError(
                f"Permutation of size {len(permutation)} is incompatible with "
                f"{self.distance.shape[1]} electrons"
            )
        return type(self)(
            distance=permute_particle_axis(self.distance, permutation, axis=1),
            radial_dlogabs=permute_particle_axis(self.radial_dlogabs, permutation, axis=1),
            finite_mask=permute_particle_axis(self.finite_mask, permutation, axis=1),
        )

    def compare(
        self,
        other: "ElectronNucleusRadialValues",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, int | float]]:
        """Compare raw distances, availability masks, and finite derivatives."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf"), "finite_mask_mismatch_count": 0}
        self_shapes = (self.distance.shape, self.radial_dlogabs.shape, self.finite_mask.shape)
        other_shapes = (other.distance.shape, other.radial_dlogabs.shape, other.finite_mask.shape)
        self_devices = (self.distance.device, self.radial_dlogabs.device, self.finite_mask.device)
        other_devices = (other.distance.device, other.radial_dlogabs.device, other.finite_mask.device)
        self_dtypes = (self.distance.dtype, self.radial_dlogabs.dtype, self.finite_mask.dtype)
        other_dtypes = (other.distance.dtype, other.radial_dlogabs.dtype, other.finite_mask.dtype)
        if self_shapes != other_shapes or self_devices != other_devices or self_dtypes != other_dtypes:
            return False, {"max_abs_error": float("inf"), "finite_mask_mismatch_count": 0}
        mask_mismatch_count = int((self.finite_mask != other.finite_mask).sum().item())
        distance_close, distance_metrics = compare_tensor_blocks(
            [self.distance],
            [other.distance],
            atol=atol,
            rtol=rtol,
        )
        shared_finite = self.finite_mask & other.finite_mask
        derivative_close, derivative_metrics = compare_tensor_blocks(
            [self.radial_dlogabs[shared_finite]],
            [other.radial_dlogabs[shared_finite]],
            atol=atol,
            rtol=rtol,
        )
        max_abs_error = max(
            float(distance_metrics["max_abs_error"]),
            float(derivative_metrics["max_abs_error"]),
        )
        return (
            distance_close and derivative_close and mask_mismatch_count == 0,
            {
                "max_abs_error": max_abs_error,
                "finite_mask_mismatch_count": mask_mismatch_count,
            },
        )


@dataclass(frozen=True)
class AtlasNumericalProvenance:
    """Numerical identity attached to a generated atlas boundary."""

    dtype: str
    device: str
    evaluation_dtype: str
    evaluation_device: str
    seed: int

    def validate(self, batch: ElectronBatch) -> "AtlasNumericalProvenance":
        """Require provenance to describe the evaluated batch exactly."""

        flat = batch.flatten_samples()
        if self.dtype != "float64":
            raise ValueError(
                "AtlasNumericalProvenance.dtype must pin boundary generation to 'float64'"
            )
        if self.device != "cpu":
            raise ValueError(
                "AtlasNumericalProvenance.device must pin boundary generation to 'cpu'"
            )
        expected_dtype = str(flat.dtype).removeprefix("torch.")
        expected_device = str(flat.device)
        if self.evaluation_dtype != expected_dtype or self.evaluation_device != expected_device:
            raise ValueError(
                "AtlasNumericalProvenance evaluation dtype/device must describe the evaluated batch"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("AtlasNumericalProvenance.seed must be an integer")
        return self


@dataclass(frozen=True)
class AtlasDerivativeValues:
    """One named atlas scalar and its pathwise first and second derivatives."""

    value: torch.Tensor
    first_derivative: torch.Tensor
    second_derivative: torch.Tensor
    value_finite_mask: torch.Tensor
    first_derivative_finite_mask: torch.Tensor
    second_derivative_finite_mask: torch.Tensor

    def validate(self, batch: ElectronBatch) -> "AtlasDerivativeValues":
        """Validate scalar shapes, dtype/device, and exact finite masks."""

        flat = batch.flatten_samples()
        expected_shape = (flat.batch_size,)
        for name, value in (
            ("value", self.value),
            ("first_derivative", self.first_derivative),
            ("second_derivative", self.second_derivative),
        ):
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"AtlasDerivativeValues.{name} must have shape {expected_shape}")
            if value.device != flat.device or value.dtype != flat.dtype:
                raise ValueError(f"AtlasDerivativeValues.{name} must match batch dtype/device")
        for name, mask, value in (
            ("value_finite_mask", self.value_finite_mask, self.value),
            (
                "first_derivative_finite_mask",
                self.first_derivative_finite_mask,
                self.first_derivative,
            ),
            (
                "second_derivative_finite_mask",
                self.second_derivative_finite_mask,
                self.second_derivative,
            ),
        ):
            if tuple(mask.shape) != expected_shape or mask.dtype != torch.bool:
                raise ValueError(f"AtlasDerivativeValues.{name} must be bool with shape {expected_shape}")
            if mask.device != flat.device or not torch.equal(mask, torch.isfinite(value)):
                raise ValueError(
                    f"AtlasDerivativeValues.{name} must match value device and isfinite status"
                )
        return self

    def compare(
        self,
        other: "AtlasDerivativeValues",
        *,
        atol: float,
        rtol: float,
    ) -> tuple[bool, float, int]:
        """Compare masks exactly and numeric values only where both are finite."""

        if type(self) is not type(other):
            return False, float("inf"), 0
        close = True
        max_abs_error = 0.0
        mask_mismatch_count = 0
        for left, right, left_mask, right_mask in (
            (self.value, other.value, self.value_finite_mask, other.value_finite_mask),
            (
                self.first_derivative,
                other.first_derivative,
                self.first_derivative_finite_mask,
                other.first_derivative_finite_mask,
            ),
            (
                self.second_derivative,
                other.second_derivative,
                self.second_derivative_finite_mask,
                other.second_derivative_finite_mask,
            ),
        ):
            if left.shape != right.shape or left.device != right.device or left.dtype != right.dtype:
                return False, float("inf"), mask_mismatch_count
            mismatch = int((left_mask != right_mask).sum().item())
            mask_mismatch_count += mismatch
            shared = left_mask & right_mask
            block_close, metrics = compare_tensor_blocks(
                [left[shared]], [right[shared]], atol=atol, rtol=rtol
            )
            close = close and block_close and mismatch == 0
            max_abs_error = max(max_abs_error, float(metrics["max_abs_error"]))
        return close, max_abs_error, mask_mismatch_count


@dataclass(frozen=True)
class HeliumAtlasValues:
    """Typed primitive values for helium coalescence, curvature, and tail atlases.

    Scalar wavefunction and Hamiltonian fields are invariant to electron
    relabelling. ``per_electron_kinetic`` and its domain status carry the sole
    explicit electron axis and therefore permute semantically.
    """

    requested_coordinate: torch.Tensor
    realized_coordinate: torch.Tensor
    coordinate_representability_boundary_radius: torch.Tensor
    is_coordinate_representability_boundary: torch.Tensor
    is_exact_zero_sentinel: torch.Tensor
    ideal_unfloored_ee_inverse_distance: torch.Tensor | None
    ideal_unfloored_ee_positive_separation_domain_mask: torch.Tensor | None
    ideal_unfloored_ee_reciprocal_evaluation_defined_mask: torch.Tensor | None
    ideal_unfloored_ee_reciprocal_failure_radius: torch.Tensor | None
    is_ideal_unfloored_ee_reciprocal_failure_boundary: torch.Tensor | None
    derivatives: Mapping[str, AtlasDerivativeValues]
    total_local_energy: torch.Tensor
    total_local_energy_finite_mask: torch.Tensor
    hamiltonian_terms: Mapping[str, torch.Tensor]
    hamiltonian_term_finite_masks: Mapping[str, torch.Tensor]
    per_electron_kinetic: torch.Tensor
    per_electron_kinetic_domain_mask: torch.Tensor
    per_electron_kinetic_status: tuple[tuple[str, ...], ...]
    cancellation_abs_sum: torch.Tensor
    cancellation_residual: torch.Tensor
    cancellation_ratio: torch.Tensor
    cancellation_abs_sum_finite_mask: torch.Tensor
    cancellation_residual_finite_mask: torch.Tensor
    cancellation_ratio_finite_mask: torch.Tensor
    domain_status: tuple[str, ...]
    provenance: AtlasNumericalProvenance

    def validate(self, batch: ElectronBatch) -> "HeliumAtlasValues":
        """Validate typed shapes, semantic masks, and numerical provenance."""

        flat = batch.flatten_samples()
        sample_shape = (flat.batch_size,)
        electron_shape = (flat.batch_size, flat.n_electrons)
        for name, value in (
            ("requested_coordinate", self.requested_coordinate),
            ("realized_coordinate", self.realized_coordinate),
        ):
            if tuple(value.shape) != sample_shape or value.device != flat.device or value.dtype != flat.dtype:
                raise ValueError(
                    f"HeliumAtlasValues.{name} must match batch dtype/device with shape {sample_shape}"
                )
            if not torch.isfinite(value).all() or torch.any(value < 0):
                raise ValueError(f"HeliumAtlasValues.{name} must be finite and nonnegative")
        for name, mask in (
            (
                "is_coordinate_representability_boundary",
                self.is_coordinate_representability_boundary,
            ),
            ("is_exact_zero_sentinel", self.is_exact_zero_sentinel),
        ):
            _validate_bool_mask(mask, shape=sample_shape, device=flat.device, name=name)
        _validate_boundary_radius(
            self.coordinate_representability_boundary_radius,
            mask=self.is_coordinate_representability_boundary,
            shape=sample_shape,
            device=flat.device,
            dtype=flat.dtype,
            name="coordinate_representability_boundary_radius",
        )
        if not torch.equal(
            self.coordinate_representability_boundary_radius[
                self.is_coordinate_representability_boundary
            ],
            self.realized_coordinate[self.is_coordinate_representability_boundary],
        ):
            raise ValueError(
                "coordinate_representability_boundary_radius must equal the realized "
                "coordinate at the boundary"
            )
        if torch.any(
            self.is_coordinate_representability_boundary & self.is_exact_zero_sentinel
        ):
            raise ValueError(
                "coordinate-representability boundary rows and exact-zero sentinels must be distinct"
            )
        if not torch.all(self.requested_coordinate[self.is_exact_zero_sentinel] == 0):
            raise ValueError("exact-zero sentinels require requested_coordinate == 0")
        if not torch.all(self.realized_coordinate[self.is_exact_zero_sentinel] == 0):
            raise ValueError("exact-zero sentinels require realized_coordinate == 0")

        ideal_fields = (
            self.ideal_unfloored_ee_inverse_distance,
            self.ideal_unfloored_ee_positive_separation_domain_mask,
            self.ideal_unfloored_ee_reciprocal_evaluation_defined_mask,
            self.ideal_unfloored_ee_reciprocal_failure_radius,
            self.is_ideal_unfloored_ee_reciprocal_failure_boundary,
        )
        if any(value is None for value in ideal_fields) and not all(
            value is None for value in ideal_fields
        ):
            raise ValueError(
                "ideal unfloored e-e values, positive-separation domain, reciprocal-evaluation "
                "domain, reciprocal-failure radius, and boundary mask must be present together"
            )
        if self.ideal_unfloored_ee_inverse_distance is not None:
            ideal = self.ideal_unfloored_ee_inverse_distance
            positive_domain = self.ideal_unfloored_ee_positive_separation_domain_mask
            evaluation_defined = (
                self.ideal_unfloored_ee_reciprocal_evaluation_defined_mask
            )
            reciprocal_radius = self.ideal_unfloored_ee_reciprocal_failure_radius
            reciprocal_boundary = self.is_ideal_unfloored_ee_reciprocal_failure_boundary
            assert positive_domain is not None
            assert evaluation_defined is not None
            assert reciprocal_radius is not None
            assert reciprocal_boundary is not None
            if tuple(ideal.shape) != sample_shape or ideal.device != flat.device or ideal.dtype != flat.dtype:
                raise ValueError(
                    "ideal_unfloored_ee_inverse_distance must match batch dtype/device and sample shape"
                )
            _validate_bool_mask(
                positive_domain,
                shape=sample_shape,
                device=flat.device,
                name="ideal_unfloored_ee_positive_separation_domain_mask",
            )
            if not torch.equal(positive_domain, self.realized_coordinate > 0):
                raise ValueError(
                    "ideal_unfloored_ee_positive_separation_domain_mask must be true exactly "
                    "at positive realized separation"
                )
            expected_ideal = self.realized_coordinate.reciprocal()
            if not torch.equal(ideal, expected_ideal):
                raise ValueError(
                    "ideal_unfloored_ee_inverse_distance must equal the unfloored reciprocal "
                    "of realized_coordinate"
                )
            _validate_bool_mask(
                evaluation_defined,
                shape=sample_shape,
                device=flat.device,
                name="ideal_unfloored_ee_reciprocal_evaluation_defined_mask",
            )
            if not torch.equal(evaluation_defined, torch.isfinite(ideal)):
                raise ValueError(
                    "ideal_unfloored_ee_reciprocal_evaluation_defined_mask must be true "
                    "exactly where the ideal reciprocal is finite"
                )
            _validate_bool_mask(
                reciprocal_boundary,
                shape=sample_shape,
                device=flat.device,
                name="is_ideal_unfloored_ee_reciprocal_failure_boundary",
            )
            _validate_boundary_radius(
                reciprocal_radius,
                mask=reciprocal_boundary,
                shape=sample_shape,
                device=flat.device,
                dtype=flat.dtype,
                name="ideal_unfloored_ee_reciprocal_failure_radius",
            )
            if not torch.equal(
                reciprocal_radius[reciprocal_boundary],
                self.realized_coordinate[reciprocal_boundary],
            ):
                raise ValueError(
                    "ideal_unfloored_ee_reciprocal_failure_radius must equal the realized "
                    "coordinate at the transition"
                )
            if torch.any(reciprocal_boundary & self.is_exact_zero_sentinel):
                raise ValueError(
                    "ideal unfloored e-e reciprocal-failure boundaries and zero sentinels must be distinct"
                )
            if not torch.all(~torch.isfinite(ideal[reciprocal_boundary])):
                raise ValueError(
                    "ideal unfloored e-e reciprocal-failure boundary must mark nonfinite reciprocal values"
                )

        if not self.derivatives:
            raise ValueError("HeliumAtlasValues.derivatives must not be empty")
        for name, values in self.derivatives.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("HeliumAtlasValues derivative names must be non-empty strings")
            if not isinstance(values, AtlasDerivativeValues):
                raise TypeError(f"HeliumAtlasValues derivative {name!r} must be AtlasDerivativeValues")
            values.validate(flat)

        _validate_float64_sample(
            self.total_local_energy,
            shape=sample_shape,
            device=flat.device,
            name="total_local_energy",
        )
        _validate_bool_mask(
            self.total_local_energy_finite_mask,
            shape=sample_shape,
            device=flat.device,
            name="total_local_energy_finite_mask",
        )
        if not torch.equal(self.total_local_energy_finite_mask, torch.isfinite(self.total_local_energy)):
            raise ValueError("total_local_energy_finite_mask must equal isfinite(total_local_energy)")
        if not self.hamiltonian_terms or set(self.hamiltonian_terms) != set(
            self.hamiltonian_term_finite_masks
        ):
            raise ValueError("Hamiltonian values and finite masks must have identical non-empty names")
        for name, value in self.hamiltonian_terms.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Hamiltonian term names must be non-empty strings")
            _validate_float64_sample(value, shape=sample_shape, device=flat.device, name=name)
            mask = self.hamiltonian_term_finite_masks[name]
            _validate_bool_mask(mask, shape=sample_shape, device=flat.device, name=f"{name}_finite_mask")
            if not torch.equal(mask, torch.isfinite(value)):
                raise ValueError(f"Hamiltonian term {name!r} finite mask must equal isfinite")

        _validate_float64_sample(
            self.per_electron_kinetic,
            shape=electron_shape,
            device=flat.device,
            name="per_electron_kinetic",
        )
        _validate_bool_mask(
            self.per_electron_kinetic_domain_mask,
            shape=electron_shape,
            device=flat.device,
            name="per_electron_kinetic_domain_mask",
        )
        if len(self.per_electron_kinetic_status) != flat.batch_size or any(
            len(row) != flat.n_electrons for row in self.per_electron_kinetic_status
        ):
            raise ValueError(
                "per_electron_kinetic_status must have [batch, n_electrons] string structure"
            )
        kinetic_finite = torch.isfinite(self.per_electron_kinetic)
        if not torch.equal(self.per_electron_kinetic_domain_mask, kinetic_finite):
            raise ValueError(
                "per_electron_kinetic_domain_mask must be true exactly for finite attributions"
            )
        for index, statuses in enumerate(self.per_electron_kinetic_status):
            for electron, status in enumerate(statuses):
                is_finite = bool(kinetic_finite[index, electron].item())
                if (is_finite and status != "defined") or (
                    not is_finite and not status.startswith("undefined_")
                ):
                    raise ValueError(
                        "per_electron_kinetic_status must explicitly distinguish defined and undefined values"
                    )

        for name, value, mask in (
            (
                "cancellation_abs_sum",
                self.cancellation_abs_sum,
                self.cancellation_abs_sum_finite_mask,
            ),
            (
                "cancellation_residual",
                self.cancellation_residual,
                self.cancellation_residual_finite_mask,
            ),
            (
                "cancellation_ratio",
                self.cancellation_ratio,
                self.cancellation_ratio_finite_mask,
            ),
        ):
            _validate_float64_sample(value, shape=sample_shape, device=flat.device, name=name)
            _validate_bool_mask(mask, shape=sample_shape, device=flat.device, name=f"{name}_finite_mask")
            if not torch.equal(mask, torch.isfinite(value)):
                raise ValueError(f"{name}_finite_mask must equal isfinite({name})")
        if len(self.domain_status) != flat.batch_size or any(
            not isinstance(value, str) or not value for value in self.domain_status
        ):
            raise ValueError("domain_status must contain one non-empty string per sample")
        sample_finite = torch.ones(sample_shape, device=flat.device, dtype=torch.bool)
        for values in self.derivatives.values():
            sample_finite &= values.value_finite_mask
            sample_finite &= values.first_derivative_finite_mask
            sample_finite &= values.second_derivative_finite_mask
        sample_finite &= self.total_local_energy_finite_mask
        for mask in self.hamiltonian_term_finite_masks.values():
            sample_finite &= mask
        kinetic_was_requested = any(
            status != "undefined_no_kinetic_registry_term"
            for statuses in self.per_electron_kinetic_status
            for status in statuses
        )
        if kinetic_was_requested:
            sample_finite &= self.per_electron_kinetic_domain_mask.all(dim=1)
        expected_domain_status = tuple(
            "exact_zero_sentinel"
            if bool(self.is_exact_zero_sentinel[index].item())
            else "ideal_unfloored_ee_reciprocal_and_coordinate_representability_boundary"
            if bool(self.is_coordinate_representability_boundary[index].item())
            and self.is_ideal_unfloored_ee_reciprocal_failure_boundary is not None
            and bool(
                self.is_ideal_unfloored_ee_reciprocal_failure_boundary[index].item()
            )
            else "coordinate_representability_boundary"
            if bool(self.is_coordinate_representability_boundary[index].item())
            else "ideal_unfloored_ee_reciprocal_failure_boundary"
            if self.is_ideal_unfloored_ee_reciprocal_failure_boundary is not None
            and bool(
                self.is_ideal_unfloored_ee_reciprocal_failure_boundary[index].item()
            )
            else "finite"
            if bool(sample_finite[index].item())
            else "computed_nonfinite_retained"
            for index in range(flat.batch_size)
        )
        if self.domain_status != expected_domain_status:
            raise ValueError(
                "domain_status must exactly encode sentinel, boundary, finite, and retained-nonfinite semantics"
            )
        self.provenance.validate(flat)
        return self

    def permute(self, permutation: Permutation) -> "HeliumAtlasValues":
        """Apply the electron permutation to per-electron kinetic attribution."""

        if len(permutation) != self.per_electron_kinetic.shape[1]:
            raise ValueError(
                f"Permutation of size {len(permutation)} is incompatible with "
                f"{self.per_electron_kinetic.shape[1]} electrons"
            )
        order = tuple(permutation_index(permutation).tolist())
        return type(self)(
            requested_coordinate=self.requested_coordinate.clone(),
            realized_coordinate=self.realized_coordinate.clone(),
            coordinate_representability_boundary_radius=(
                self.coordinate_representability_boundary_radius.clone()
            ),
            is_coordinate_representability_boundary=(
                self.is_coordinate_representability_boundary.clone()
            ),
            is_exact_zero_sentinel=self.is_exact_zero_sentinel.clone(),
            ideal_unfloored_ee_inverse_distance=(
                None
                if self.ideal_unfloored_ee_inverse_distance is None
                else self.ideal_unfloored_ee_inverse_distance.clone()
            ),
            ideal_unfloored_ee_positive_separation_domain_mask=(
                None
                if self.ideal_unfloored_ee_positive_separation_domain_mask is None
                else self.ideal_unfloored_ee_positive_separation_domain_mask.clone()
            ),
            ideal_unfloored_ee_reciprocal_evaluation_defined_mask=(
                None
                if self.ideal_unfloored_ee_reciprocal_evaluation_defined_mask is None
                else self.ideal_unfloored_ee_reciprocal_evaluation_defined_mask.clone()
            ),
            ideal_unfloored_ee_reciprocal_failure_radius=(
                None
                if self.ideal_unfloored_ee_reciprocal_failure_radius is None
                else self.ideal_unfloored_ee_reciprocal_failure_radius.clone()
            ),
            is_ideal_unfloored_ee_reciprocal_failure_boundary=(
                None
                if self.is_ideal_unfloored_ee_reciprocal_failure_boundary is None
                else self.is_ideal_unfloored_ee_reciprocal_failure_boundary.clone()
            ),
            derivatives=dict(self.derivatives),
            total_local_energy=self.total_local_energy.clone(),
            total_local_energy_finite_mask=self.total_local_energy_finite_mask.clone(),
            hamiltonian_terms={name: value.clone() for name, value in self.hamiltonian_terms.items()},
            hamiltonian_term_finite_masks={
                name: value.clone() for name, value in self.hamiltonian_term_finite_masks.items()
            },
            per_electron_kinetic=permute_particle_axis(
                self.per_electron_kinetic, permutation, axis=1
            ),
            per_electron_kinetic_domain_mask=permute_particle_axis(
                self.per_electron_kinetic_domain_mask, permutation, axis=1
            ),
            per_electron_kinetic_status=tuple(
                tuple(statuses[index] for index in order)
                for statuses in self.per_electron_kinetic_status
            ),
            cancellation_abs_sum=self.cancellation_abs_sum.clone(),
            cancellation_residual=self.cancellation_residual.clone(),
            cancellation_ratio=self.cancellation_ratio.clone(),
            cancellation_abs_sum_finite_mask=self.cancellation_abs_sum_finite_mask.clone(),
            cancellation_residual_finite_mask=self.cancellation_residual_finite_mask.clone(),
            cancellation_ratio_finite_mask=self.cancellation_ratio_finite_mask.clone(),
            domain_status=self.domain_status,
            provenance=self.provenance,
        )

    def compare(
        self,
        other: "HeliumAtlasValues",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, int | float]]:
        """Compare semantic status exactly and finite numeric values tolerantly."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf"), "status_mismatch_count": 0}
        if (self.ideal_unfloored_ee_inverse_distance is None) != (
            other.ideal_unfloored_ee_inverse_distance is None
        ):
            return False, {"max_abs_error": float("inf"), "status_mismatch_count": 1}
        if (
            tuple(self.derivatives) != tuple(other.derivatives)
            or tuple(self.hamiltonian_terms) != tuple(other.hamiltonian_terms)
            or self.domain_status != other.domain_status
            or self.per_electron_kinetic_status != other.per_electron_kinetic_status
            or self.provenance != other.provenance
        ):
            return False, {"max_abs_error": float("inf"), "status_mismatch_count": 1}
        status_mismatch_count = 0
        max_abs_error = 0.0
        close = True
        for left, right in (
            (self.requested_coordinate, other.requested_coordinate),
            (self.realized_coordinate, other.realized_coordinate),
        ):
            block_close, metrics = compare_tensor_blocks([left], [right], atol=atol, rtol=rtol)
            close = close and block_close
            max_abs_error = max(max_abs_error, float(metrics["max_abs_error"]))
        for left, right in (
            (
                self.is_coordinate_representability_boundary,
                other.is_coordinate_representability_boundary,
            ),
            (self.is_exact_zero_sentinel, other.is_exact_zero_sentinel),
            (self.total_local_energy_finite_mask, other.total_local_energy_finite_mask),
            (self.per_electron_kinetic_domain_mask, other.per_electron_kinetic_domain_mask),
            (self.cancellation_abs_sum_finite_mask, other.cancellation_abs_sum_finite_mask),
            (self.cancellation_residual_finite_mask, other.cancellation_residual_finite_mask),
            (self.cancellation_ratio_finite_mask, other.cancellation_ratio_finite_mask),
        ):
            status_mismatch_count += int((left != right).sum().item())
        if self.ideal_unfloored_ee_inverse_distance is not None:
            assert self.ideal_unfloored_ee_positive_separation_domain_mask is not None
            assert self.ideal_unfloored_ee_reciprocal_evaluation_defined_mask is not None
            assert other.ideal_unfloored_ee_inverse_distance is not None
            assert other.ideal_unfloored_ee_positive_separation_domain_mask is not None
            assert other.ideal_unfloored_ee_reciprocal_evaluation_defined_mask is not None
            assert self.ideal_unfloored_ee_reciprocal_failure_radius is not None
            assert other.ideal_unfloored_ee_reciprocal_failure_radius is not None
            assert self.is_ideal_unfloored_ee_reciprocal_failure_boundary is not None
            assert other.is_ideal_unfloored_ee_reciprocal_failure_boundary is not None
            status_mismatch_count += int(
                (
                    self.is_ideal_unfloored_ee_reciprocal_failure_boundary
                    != other.is_ideal_unfloored_ee_reciprocal_failure_boundary
                )
                .sum()
                .item()
            )
            self_ideal_finite = torch.isfinite(self.ideal_unfloored_ee_inverse_distance)
            other_ideal_finite = torch.isfinite(other.ideal_unfloored_ee_inverse_distance)
            status_mismatch_count += int(
                (
                    self.ideal_unfloored_ee_positive_separation_domain_mask
                    != other.ideal_unfloored_ee_positive_separation_domain_mask
                )
                .sum()
                .item()
            )
            status_mismatch_count += int(
                (
                    self.ideal_unfloored_ee_reciprocal_evaluation_defined_mask
                    != other.ideal_unfloored_ee_reciprocal_evaluation_defined_mask
                )
                .sum()
                .item()
            )
            status_mismatch_count += int((self_ideal_finite != other_ideal_finite).sum().item())
            shared_ideal = self_ideal_finite & other_ideal_finite
            ideal_close, ideal_metrics = compare_tensor_blocks(
                [self.ideal_unfloored_ee_inverse_distance[shared_ideal]],
                [other.ideal_unfloored_ee_inverse_distance[shared_ideal]],
                atol=atol,
                rtol=rtol,
            )
            close = close and ideal_close
            max_abs_error = max(max_abs_error, float(ideal_metrics["max_abs_error"]))
            shared_reciprocal_radius = (
                self.is_ideal_unfloored_ee_reciprocal_failure_boundary
                & other.is_ideal_unfloored_ee_reciprocal_failure_boundary
            )
            radius_close, radius_metrics = compare_tensor_blocks(
                [
                    self.ideal_unfloored_ee_reciprocal_failure_radius[
                        shared_reciprocal_radius
                    ]
                ],
                [
                    other.ideal_unfloored_ee_reciprocal_failure_radius[
                        shared_reciprocal_radius
                    ]
                ],
                atol=atol,
                rtol=rtol,
            )
            close = close and radius_close
            max_abs_error = max(max_abs_error, float(radius_metrics["max_abs_error"]))
        shared_coordinate_radius = (
            self.is_coordinate_representability_boundary
            & other.is_coordinate_representability_boundary
        )
        coordinate_radius_close, coordinate_radius_metrics = compare_tensor_blocks(
            [self.coordinate_representability_boundary_radius[shared_coordinate_radius]],
            [other.coordinate_representability_boundary_radius[shared_coordinate_radius]],
            atol=atol,
            rtol=rtol,
        )
        close = close and coordinate_radius_close
        max_abs_error = max(
            max_abs_error, float(coordinate_radius_metrics["max_abs_error"])
        )
        for name in self.derivatives:
            derivative_close, error, mismatches = self.derivatives[name].compare(
                other.derivatives[name], atol=atol, rtol=rtol
            )
            close = close and derivative_close
            max_abs_error = max(max_abs_error, error)
            status_mismatch_count += mismatches
        numeric = [
            (
                self.total_local_energy,
                other.total_local_energy,
                self.total_local_energy_finite_mask & other.total_local_energy_finite_mask,
            ),
            (
                self.per_electron_kinetic,
                other.per_electron_kinetic,
                self.per_electron_kinetic_domain_mask & other.per_electron_kinetic_domain_mask,
            ),
            (
                self.cancellation_abs_sum,
                other.cancellation_abs_sum,
                self.cancellation_abs_sum_finite_mask & other.cancellation_abs_sum_finite_mask,
            ),
            (
                self.cancellation_residual,
                other.cancellation_residual,
                self.cancellation_residual_finite_mask & other.cancellation_residual_finite_mask,
            ),
            (
                self.cancellation_ratio,
                other.cancellation_ratio,
                self.cancellation_ratio_finite_mask & other.cancellation_ratio_finite_mask,
            ),
        ]
        for name in self.hamiltonian_terms:
            left_mask = self.hamiltonian_term_finite_masks[name]
            right_mask = other.hamiltonian_term_finite_masks[name]
            status_mismatch_count += int((left_mask != right_mask).sum().item())
            numeric.append(
                (
                    self.hamiltonian_terms[name],
                    other.hamiltonian_terms[name],
                    left_mask & right_mask,
                )
            )
        for left, right, shared_mask in numeric:
            block_close, metrics = compare_tensor_blocks(
                [left[shared_mask]], [right[shared_mask]], atol=atol, rtol=rtol
            )
            close = close and block_close
            max_abs_error = max(max_abs_error, float(metrics["max_abs_error"]))
        return close and status_mismatch_count == 0, {
            "max_abs_error": max_abs_error,
            "status_mismatch_count": status_mismatch_count,
        }


def _validate_bool_mask(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    name: str,
) -> None:
    """Validate one explicitly named boolean atlas status tensor."""

    if tuple(value.shape) != shape or value.dtype != torch.bool or value.device != device:
        raise ValueError(f"HeliumAtlasValues.{name} must be bool on {device} with shape {shape}")


def _validate_float64_sample(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    name: str,
) -> None:
    """Keep Hamiltonian cancellation diagnostics in float64 end to end."""

    if tuple(value.shape) != shape or value.dtype != torch.float64 or value.device != device:
        raise ValueError(f"HeliumAtlasValues.{name} must be float64 on {device} with shape {shape}")


def _validate_boundary_radius(
    value: torch.Tensor,
    *,
    mask: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> None:
    """Require a finite nonnegative radius exactly on each named boundary."""

    if tuple(value.shape) != shape or value.device != device or value.dtype != dtype:
        raise ValueError(
            f"HeliumAtlasValues.{name} must match batch dtype/device with shape {shape}"
        )
    if not torch.equal(torch.isfinite(value), mask):
        raise ValueError(f"HeliumAtlasValues.{name} must be finite exactly on its boundary")
    if not torch.isnan(value[~mask]).all():
        raise ValueError(f"HeliumAtlasValues.{name} must use explicit NaN outside its boundary")
    if torch.any(value[mask] < 0):
        raise ValueError(
            f"HeliumAtlasValues.{name} boundary values must be nonnegative realized "
            "coordinates"
        )


class TransformName(str, Enum):
    """Stable names for supported transform-comparison calculators."""

    FULL_MODEL_ANTISYMMETRY = "full_model_antisymmetry"
    SPATIAL_EXCHANGE_SYMMETRY = "spatial_exchange_symmetry"
    ROTATION_CONSISTENCY = "rotation_consistency"


class TransformKind(str, Enum):
    """Semantic kinds for transform-specific summary dispatch."""

    FULL_MODEL_ANTISYMMETRY = "full_model_antisymmetry"
    SPATIAL_EXCHANGE = "spatial_exchange"
    ROTATION_CONSISTENCY = "rotation_consistency"


@dataclass(frozen=True)
class TransformComparisonValues:
    """Raw values and typed identity for one transformed model comparison."""

    original_logabs: torch.Tensor
    transformed_logabs: torch.Tensor
    original_sign: torch.Tensor
    transformed_sign: torch.Tensor
    logabs_abs_error: torch.Tensor
    sign_mismatch: torch.Tensor
    sample_index: torch.Tensor
    original_positions: torch.Tensor
    transformed_positions: torch.Tensor
    transform_name: TransformName
    transform_kind: TransformKind
    finite: torch.Tensor
    metadata: Mapping[str, Any]
    local_energy_abs_error: torch.Tensor | None = None


@dataclass(frozen=True)
class TraceKeySummary:
    """Typed aggregate diagnostics for one trace key."""

    key: str
    count: int
    mean_abs_error: float
    max_abs_error: float
    failure_count: int
    missing_count: int
    extra_count: int
    sample_count: int = 0


@dataclass(frozen=True)
class TraceComparisonValues:
    """Raw trace-comparison records produced by trace equivariance checks."""

    max_abs_error: torch.Tensor
    mean_abs_error: torch.Tensor
    failure_count: int
    compared_entry_count: int
    comparison_error_count: int
    missing_key_count: int
    extra_key_count: int
    records: Sequence[Mapping[str, Any]]
    key_summaries: Sequence[TraceKeySummary] = ()
    compared_sample_count: int = 0


@dataclass(frozen=True)
class FeatureTraceValues:
    """Raw feature magnitude records collected from trace entries."""

    records: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class ReadoutTraceValues:
    """Raw readout conditioning records collected from trace entries."""

    records: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class FactorResponseArmValues:
    """Row-aligned fixed-configuration values for one declared factor arm."""

    label: str
    parameter_scales: Mapping[str, float]
    realized_parameters: Mapping[str, float]
    local_energy: torch.Tensor
    logabs: torch.Tensor
    sign: torch.Tensor
    term_energies: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class FactorResponseValues:
    """Common-configuration factor responses with explicit comparison semantics."""

    comparison_kind: str
    baseline_label: str
    arms: Sequence[FactorResponseArmValues]
    model_state_restored: bool


@dataclass(frozen=True)
class EvaluationBundle:
    """Reusable primitive outputs for one evaluation task."""

    generated: GeneratedConfigurations
    wavefunction: WavefunctionValues | None = None
    local_energy: LocalEnergyValues | None = None
    derivatives: Mapping[str, DerivativeValues] | None = None
    electron_nucleus_radial: ElectronNucleusRadialValues | None = None
    helium_atlas: HeliumAtlasValues | None = None
    trace: Trace | None = None
    transform: TransformComparisonValues | None = None
    trace_comparison: TraceComparisonValues | None = None
    feature_trace: FeatureTraceValues | None = None
    readout_trace: ReadoutTraceValues | None = None
    factor_response: FactorResponseValues | None = None


__all__ = [
    "AtlasDerivativeValues",
    "AtlasNumericalProvenance",
    "DerivativeValues",
    "ElectronNucleusRadialValues",
    "EvaluationBundle",
    "FactorResponseArmValues",
    "FactorResponseValues",
    "FeatureTraceValues",
    "GeneratedConfigurations",
    "HeliumAtlasValues",
    "LocalEnergyValues",
    "ReadoutTraceValues",
    "TraceKeySummary",
    "TraceComparisonValues",
    "TransformKind",
    "TransformName",
    "TransformComparisonValues",
    "WavefunctionValues",
]
