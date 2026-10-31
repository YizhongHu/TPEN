"""Electron coordinate batch state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any

import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch.base import _coerce_optional_tensor
from tpen.data.indices import permute_particle_axis
from tpen.data.permutation import Permutation


@dataclass
class ElectronBatch:
    """Store batched electron coordinates and optional context.

    ``nuclear_positions``/``nuclear_charges`` remain the raw, possibly
    per-sample pipeline-transport tensors consumed by physics/nn code (e.g.
    evaluation-orbit generators that vary nuclear geometry per sample); they
    are not collapsed into `AtomicConfiguration` because that type is
    strictly unbatched (one fixed system). ``atomic_configuration`` is an
    additive typed reference to the constructor/config-owned fixed system,
    carried through the pipeline for identity/comparison; when both are
    given for an unbatched (non-per-sample) batch, they must agree exactly
    so the typed reference is never a competing authority over the raw
    tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Electron coordinates with shape
        ``[*sample_shape, n_electrons, spatial_dim]``.
    system : object or None, optional
        Optional system metadata, such as electron count and nuclear data.
    nuclear_positions : torch.Tensor or None, optional
        Nuclear coordinates with shape ``[n_nuclei, dim]`` or
        ``[*sample_shape, n_nuclei, dim]``.
    nuclear_charges : torch.Tensor or None, optional
        Nuclear charges with shape ``[n_nuclei]`` or
        ``[*sample_shape, n_nuclei]``.
    atomic_configuration : AtomicConfiguration or None, optional
        Fixed, unbatched nuclear geometry owned by the constructor/config,
        carried alongside the raw nuclear tensors for typed pipeline
        transport.
    spins : torch.Tensor or None, optional
        Spin labels with shape ``[*sample_shape, n_electrons]`` and entries
        exactly equal to ``+1`` or ``-1``.
    aux : dict, optional
        Auxiliary metadata passed through model, sampler, or physics code.
    """

    positions: torch.Tensor
    system: Any | None = None
    nuclear_positions: torch.Tensor | None = None
    nuclear_charges: torch.Tensor | None = None
    atomic_configuration: AtomicConfiguration | None = None
    spins: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = self.positions if isinstance(self.positions, torch.Tensor) else torch.as_tensor(self.positions)
        self.nuclear_positions = _coerce_optional_tensor(self.nuclear_positions, dtype=self.positions.dtype)
        self.nuclear_charges = _coerce_optional_tensor(self.nuclear_charges, dtype=self.positions.dtype)
        self.spins = _coerce_optional_tensor(self.spins, dtype=self.positions.dtype)
        if self.positions.ndim < 3:
            raise ValueError(
                "ElectronBatch.positions must have shape [*sample_shape, n_electrons, spatial_dim], "
                f"got {tuple(self.positions.shape)}"
            )
        if self.spins is not None:
            if tuple(self.spins.shape) != (*self.sample_shape, self.n_electrons):
                raise ValueError("ElectronBatch.spins must have shape [*sample_shape, n_electrons]")
            if not torch.all((self.spins == 1) | (self.spins == -1)):
                raise ValueError("ElectronBatch.spins entries must be exactly +1 or -1")
        if self.system is not None and hasattr(self.system, "n_electrons") and self.system.n_electrons is not None:
            if self.system.n_electrons != self.n_electrons:
                raise ValueError("ElectronBatch.positions disagree with system.n_electrons")
        if self.nuclear_positions is not None:
            nuclear_shape = tuple(self.nuclear_positions.shape)
            unbatched = self.nuclear_positions.ndim == 2 and nuclear_shape[-1] == self.spatial_dim
            sampled = (
                self.nuclear_positions.ndim == len(self.sample_shape) + 2
                and nuclear_shape[:-2] == self.sample_shape
                and nuclear_shape[-1] == self.spatial_dim
            )
            if not (unbatched or sampled):
                raise ValueError(
                    "ElectronBatch.nuclear_positions must have shape [n_nuclei, dim] "
                    "or [*sample_shape, n_nuclei, dim]"
                )
        if self.nuclear_charges is not None:
            charge_shape = tuple(self.nuclear_charges.shape)
            unbatched = self.nuclear_charges.ndim == 1
            sampled = self.nuclear_charges.ndim == len(self.sample_shape) + 1 and charge_shape[:-1] == self.sample_shape
            if not (unbatched or sampled):
                raise ValueError("ElectronBatch.nuclear_charges must have shape [n_nuclei] or [*sample_shape, n_nuclei]")
        if self.nuclear_positions is not None and self.nuclear_charges is not None:
            if self.nuclear_positions.shape[-2] != self.nuclear_charges.shape[-1]:
                raise ValueError("ElectronBatch.nuclear_positions and nuclear_charges must agree on n_nuclei")
        if self.atomic_configuration is not None:
            if not isinstance(self.atomic_configuration, AtomicConfiguration):
                raise TypeError("ElectronBatch.atomic_configuration must be an AtomicConfiguration or None")
            if self.atomic_configuration.device != self.positions.device:
                raise ValueError("ElectronBatch.atomic_configuration must be on the positions device")
            if self.nuclear_positions is not None and self.nuclear_positions.ndim == 2:
                if not torch.equal(self.nuclear_positions, self.atomic_configuration.positions):
                    raise ValueError("ElectronBatch.atomic_configuration must match unbatched nuclear_positions")
            if self.nuclear_charges is not None and self.nuclear_charges.ndim == 1:
                if not torch.equal(self.nuclear_charges, self.atomic_configuration.charges):
                    raise ValueError("ElectronBatch.atomic_configuration must match unbatched nuclear_charges")

    def validate(self) -> "ElectronBatch":
        """Validate this batch using the constructor invariants.

        Returns
        -------
        ElectronBatch
            This batch, for fluent runtime validation.
        """

        ElectronBatch(
            positions=self.positions,
            system=self.system,
            nuclear_positions=self.nuclear_positions,
            nuclear_charges=self.nuclear_charges,
            atomic_configuration=self.atomic_configuration,
            spins=self.spins,
            aux=self.aux,
        )
        return self

    def validity_metrics(self) -> dict[str, int | float | bool]:
        """Return JSON-safe, explicit runtime validity metrics for this batch.

        Metrics are semantic and field-specific (no generic object traversal):
        configuration/electron/dimension counts and finite/valid fractions for
        positions, spins, and nuclear positions.
        """

        positions_total = int(self.positions.numel())
        positions_finite = int(torch.isfinite(self.positions).sum().item())
        n_configurations = 1
        for size in self.sample_shape:
            n_configurations *= int(size)
        metrics: dict[str, int | float | bool] = {
            "n_configurations": n_configurations,
            "n_electrons": int(self.n_electrons),
            "spatial_dim": int(self.spatial_dim),
            "positions_total_count": positions_total,
            "positions_finite_count": positions_finite,
            "positions_nonfinite_fraction": (
                float((positions_total - positions_finite) / positions_total) if positions_total else 0.0
            ),
        }
        if self.spins is not None:
            spins_total = int(self.spins.numel())
            spins_valid = int(((self.spins == 1) | (self.spins == -1)).sum().item())
            metrics["spins_total_count"] = spins_total
            metrics["spins_valid_count"] = spins_valid
            metrics["spins_invalid_fraction"] = (
                float((spins_total - spins_valid) / spins_total) if spins_total else 0.0
            )
        if self.nuclear_positions is not None:
            nuclear_total = int(self.nuclear_positions.numel())
            nuclear_finite = int(torch.isfinite(self.nuclear_positions).sum().item())
            metrics["nuclear_positions_total_count"] = nuclear_total
            metrics["nuclear_positions_finite_count"] = nuclear_finite
            metrics["nuclear_positions_nonfinite_fraction"] = (
                float((nuclear_total - nuclear_finite) / nuclear_total) if nuclear_total else 0.0
            )
        return metrics

    @property
    def device(self) -> torch.device:
        """Return the device of the position tensor.

        Returns
        -------
        torch.device
            Device on which `positions` is stored.
        """

        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the position tensor.

        Returns
        -------
        torch.dtype
            Data type used by `positions`.
        """

        return self.positions.dtype

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Return the sample axes before particle and coordinate axes.

        Returns
        -------
        tuple of int
            Leading sample shape of `positions`.
        """

        return tuple(self.positions.shape[:-2])

    @property
    def n_configurations(self) -> int:
        """Return the flattened number of electron configurations.

        Returns
        -------
        int
            Product of all sample axes.
        """

        return prod(self.sample_shape)

    @property
    def batch_size(self) -> int:
        """Return the flattened number of electron configurations.

        Returns
        -------
        int
            Product of all sample axes.
        """

        return self.n_configurations

    @property
    def n_electrons(self) -> int:
        """Return the number of electrons per configuration.

        Returns
        -------
        int
            Size of the electron axis.
        """

        return self.positions.shape[-2]

    @property
    def spatial_dim(self) -> int:
        """Return the spatial dimension of each electron coordinate.

        Returns
        -------
        int
            Size of the final coordinate axis.
        """

        return self.positions.shape[-1]

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ElectronBatch":
        """Move tensor fields to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target floating-point dtype. If ``None``, the current dtype is
            preserved.

        Returns
        -------
        ElectronBatch
            Batch with tensor fields moved to the requested device or dtype.
        """

        return replace(
            self,
            positions=self.positions.to(device=device, dtype=dtype),
            nuclear_positions=None if self.nuclear_positions is None else self.nuclear_positions.to(device=device, dtype=dtype),
            nuclear_charges=None if self.nuclear_charges is None else self.nuclear_charges.to(device=device, dtype=dtype),
            atomic_configuration=None if self.atomic_configuration is None else self.atomic_configuration.to(device=device, dtype=dtype),
            spins=None if self.spins is None else self.spins.to(device=device, dtype=dtype),
        )

    def flatten_samples(self) -> "ElectronBatch":
        """Return a 3D batch by flattening all sample axes.

        Returns
        -------
        ElectronBatch
            Batch whose tensor fields use a single leading configuration axis.
        """

        positions = self.positions.reshape(self.n_configurations, self.n_electrons, self.spatial_dim)
        if self.nuclear_positions is None or self.nuclear_positions.ndim == 2:
            nuclear_positions = self.nuclear_positions
        else:
            nuclear_positions = self.nuclear_positions.reshape(self.n_configurations, *self.nuclear_positions.shape[-2:])
        if self.nuclear_charges is None or self.nuclear_charges.ndim == 1:
            nuclear_charges = self.nuclear_charges
        else:
            nuclear_charges = self.nuclear_charges.reshape(self.n_configurations, self.nuclear_charges.shape[-1])
        spins = None if self.spins is None else self.spins.reshape(self.n_configurations, self.n_electrons)
        aux = {
            key: _flatten_aux_value(value, n_configurations=self.n_configurations, n_electrons=self.n_electrons)
            for key, value in self.aux.items()
        }
        return replace(
            self,
            positions=positions,
            nuclear_positions=nuclear_positions,
            nuclear_charges=nuclear_charges,
            spins=spins,
            aux=aux,
        )

    def permute(self, permutation: Permutation) -> "ElectronBatch":
        """Return a copy with electron-indexed fields permuted.

        Parameters
        ----------
        permutation : Permutation
            Particle-label permutation acting on the electron axis.

        Returns
        -------
        ElectronBatch
            Batch with positions and spin labels transformed by the active
            permutation convention.
        """

        if len(permutation) != self.n_electrons:
            raise ValueError(
                f"Permutation of size {len(permutation)} is incompatible with "
                f"{self.n_electrons} electrons"
            )
        positions = permute_particle_axis(self.positions, permutation, axis=-2)
        spins = None if self.spins is None else permute_particle_axis(self.spins, permutation, axis=-1)
        aux = {
            key: _permute_aux_value(value, permutation=permutation, n_electrons=self.n_electrons)
            for key, value in self.aux.items()
        }
        return replace(self, positions=positions, spins=spins, aux=aux)


def _flatten_aux_value(value: Any, *, n_configurations: int, n_electrons: int) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim < 3 or int(value.shape[-2]) != n_electrons:
        return value
    if prod(value.shape[:-2]) != n_configurations:
        return value
    return value.reshape(n_configurations, n_electrons, *value.shape[-1:])


def _permute_aux_value(value: Any, *, permutation: Permutation, n_electrons: int) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim < 2 or int(value.shape[-2]) != n_electrons:
        return value
    return permute_particle_axis(value, permutation, axis=-2)


def slice_flat_batch(batch: ElectronBatch, start: int, end: int) -> ElectronBatch:
    """Slice a flattened electron batch along its sample axis."""

    positions = batch.positions[start:end]
    spins = None if batch.spins is None else batch.spins[start:end]
    nuclear_positions = batch.nuclear_positions
    if nuclear_positions is not None and nuclear_positions.ndim == 3 and nuclear_positions.shape[0] == batch.batch_size:
        nuclear_positions = nuclear_positions[start:end]
    nuclear_charges = batch.nuclear_charges
    if nuclear_charges is not None and nuclear_charges.ndim == 2 and nuclear_charges.shape[0] == batch.batch_size:
        nuclear_charges = nuclear_charges[start:end]
    aux: dict[str, Any] = {}
    for key, value in batch.aux.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (batch.batch_size,):
            aux[key] = value[start:end]
        else:
            aux[key] = value
    return ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        atomic_configuration=batch.atomic_configuration,
        spins=spins,
        aux=aux,
    )


__all__ = ["ElectronBatch", "slice_flat_batch"]
