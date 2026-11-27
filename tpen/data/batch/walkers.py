"""Monte Carlo walker state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch.base import _coerce_optional_tensor
from tpen.data.batch.electron_batch import ElectronBatch
from tpen.data.batch.wavefunction_output import WavefunctionOutput


def _move_aux_value(value: Any, *, device: torch.device | str | None, dtype: torch.dtype | None) -> Any:
    """Move tensor leaves in auxiliary state without changing metadata."""

    if isinstance(value, torch.Tensor):
        tensor_dtype = dtype if value.is_floating_point() or value.is_complex() else None
        return value.to(device=device, dtype=tensor_dtype)
    if isinstance(value, dict):
        return {key: _move_aux_value(item, device=device, dtype=dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_aux_value(item, device=device, dtype=dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_aux_value(item, device=device, dtype=dtype) for item in value)
    return value


@dataclass
class Walkers:
    """Store Monte Carlo walker state and cached model values.

    Parameters
    ----------
    positions : torch.Tensor
        Walker electron coordinates with shape
        ``[batch, n_electrons, spatial_dim]``.
    logabs : torch.Tensor or None, optional
        Cached log absolute wavefunction values with shape ``[batch]``.
    sign : torch.Tensor or None, optional
        Cached real wavefunction signs with shape ``[batch]``.
    spins : torch.Tensor or None, optional
        Fixed spin labels with shape ``[batch, n_electrons]`` and entries
        exactly equal to ``+1`` or ``-1``.
    atomic_configuration : AtomicConfiguration or None, optional
        Fixed nuclear geometry owned by the sampler/constructor, carried by
        reference (never reconstructed) for typed pipeline transport. There
        is no separate partial positions/charges pair to go out of sync.
    aux : dict, optional
        Auxiliary sampler state and metadata.
    """

    positions: torch.Tensor
    logabs: torch.Tensor | None = None
    sign: torch.Tensor | None = None
    spins: torch.Tensor | None = None
    atomic_configuration: AtomicConfiguration | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = self.positions if isinstance(self.positions, torch.Tensor) else torch.as_tensor(self.positions)
        self.logabs = _coerce_optional_tensor(self.logabs, dtype=self.positions.dtype)
        self.sign = _coerce_optional_tensor(self.sign, dtype=self.positions.dtype)
        self.spins = _coerce_optional_tensor(self.spins, dtype=self.positions.dtype)
        if self.positions.ndim != 3:
            raise ValueError("Walkers.positions must have shape [batch, n_electrons, spatial_dim]")
        if self.spins is not None:
            if tuple(self.spins.shape) != tuple(self.positions.shape[:2]):
                raise ValueError("Walkers.spins must have shape [batch, n_electrons]")
            if not torch.all((self.spins == 1) | (self.spins == -1)):
                raise ValueError("Walkers.spins entries must be exactly +1 or -1")
        if self.logabs is not None and tuple(self.logabs.shape) != (self.positions.shape[0],):
            raise ValueError("Walkers.logabs must have shape [batch]")
        if self.sign is not None and tuple(self.sign.shape) != (self.positions.shape[0],):
            raise ValueError("Walkers.sign must have shape [batch]")
        if self.atomic_configuration is not None:
            if not isinstance(self.atomic_configuration, AtomicConfiguration):
                raise TypeError("Walkers.atomic_configuration must be an AtomicConfiguration or None")
            if self.atomic_configuration.spatial_dim != self.positions.shape[-1]:
                raise ValueError("Walkers.atomic_configuration spatial_dim must match positions")
            if self.atomic_configuration.device != self.positions.device:
                raise ValueError("Walkers.atomic_configuration must be on the positions device")

    def validate(self) -> "Walkers":
        """Validate this walker state using the constructor invariants.

        Returns
        -------
        Walkers
            This walker state, for fluent runtime validation.
        """

        Walkers(
            positions=self.positions,
            logabs=self.logabs,
            sign=self.sign,
            spins=self.spins,
            atomic_configuration=self.atomic_configuration,
            aux=self.aux,
        )
        return self

    @property
    def device(self) -> torch.device:
        """Return the device of the walker positions.

        Returns
        -------
        torch.device
            Device on which `positions` is stored.
        """

        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the walker positions.

        Returns
        -------
        torch.dtype
            Data type used by `positions`.
        """

        return self.positions.dtype

    @property
    def batch_size(self) -> int:
        """Return the number of walkers.

        Returns
        -------
        int
            Size of the leading walker axis.
        """

        return self.positions.shape[0]

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "Walkers":
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
        Walkers
            Walker state with tensor fields moved to the requested device or
            dtype.
        """

        return replace(
            self,
            positions=self.positions.to(device=device, dtype=dtype),
            logabs=None if self.logabs is None else self.logabs.to(device=device, dtype=dtype),
            sign=None if self.sign is None else self.sign.to(device=device, dtype=dtype),
            spins=None if self.spins is None else self.spins.to(device=device, dtype=dtype),
            atomic_configuration=None if self.atomic_configuration is None else self.atomic_configuration.to(device=device, dtype=dtype),
            aux=_move_aux_value(self.aux, device=device, dtype=dtype),
        )

    def clone(self) -> "Walkers":
        """Return a walker state with independent tensor storage.

        Returns
        -------
        Walkers
            Cloned walker state. Tensor fields are cloned, while auxiliary
            metadata is shallow-copied. `atomic_configuration` is immutable
            and carried by reference, not cloned.
        """

        return replace(
            self,
            positions=self.positions.clone(),
            logabs=None if self.logabs is None else self.logabs.clone(),
            sign=None if self.sign is None else self.sign.clone(),
            spins=None if self.spins is None else self.spins.clone(),
            aux=dict(self.aux),
        )

    def detach(self) -> "Walkers":
        """Return a walker state detached from autograd graphs.

        Returns
        -------
        Walkers
            Detached walker state with the same values and shallow-copied
            auxiliary metadata. `atomic_configuration` is already
            constructed detached and no-grad, so it is carried by reference.
        """

        return replace(
            self,
            positions=self.positions.detach(),
            logabs=None if self.logabs is None else self.logabs.detach(),
            sign=None if self.sign is None else self.sign.detach(),
            spins=None if self.spins is None else self.spins.detach(),
            aux=dict(self.aux),
        )

    def with_positions(self, positions: torch.Tensor, *, invalidate_cache: bool = True) -> "Walkers":
        """Return walker state using replacement positions.

        Parameters
        ----------
        positions : torch.Tensor
            Replacement positions with shape ``[batch, n_electrons,
            spatial_dim]``.
        invalidate_cache : bool, optional
            Whether to clear cached wavefunction values. This should usually
            remain ``True`` whenever positions change.

        Returns
        -------
        Walkers
            Walker state with replacement positions.
        """

        positions = positions if isinstance(positions, torch.Tensor) else torch.as_tensor(positions)
        if tuple(positions.shape) != tuple(self.positions.shape):
            raise ValueError(f"Replacement positions must have shape {tuple(self.positions.shape)}, got {tuple(positions.shape)}")
        return replace(
            self,
            positions=positions,
            logabs=None if invalidate_cache else self.logabs,
            sign=None if invalidate_cache else self.sign,
            aux=dict(self.aux),
        )

    def make_batch(self) -> ElectronBatch:
        """Return an electron batch view of the walker state.

        Returns
        -------
        ElectronBatch
            Batch carrying positions, spins, system metadata, and auxiliary
            metadata from the walkers.
        """

        return ElectronBatch(
            positions=self.positions,
            system=self.aux.get("system"),
            nuclear_positions=None if self.atomic_configuration is None else self.atomic_configuration.positions,
            nuclear_charges=None if self.atomic_configuration is None else self.atomic_configuration.charges,
            atomic_configuration=self.atomic_configuration,
            spins=self.spins,
            aux=dict(self.aux),
        )

    def update_cache(self, model) -> "Walkers":
        """Evaluate a model and store detached wavefunction values.

        Parameters
        ----------
        model : callable
            Wavefunction callable returning `WavefunctionOutput`.

        Returns
        -------
        Walkers
            Walker state with cached ``logabs`` and ``sign`` values.
        """

        output = model(self.make_batch())
        if not isinstance(output, WavefunctionOutput):
            raise TypeError(f"Wavefunction model must return WavefunctionOutput, got {type(output)!r}")
        logabs = output.logabs
        sign = output.sign
        if logabs.shape != (self.batch_size,):
            raise ValueError(f"Model logabs must have shape [{self.batch_size}], got {tuple(logabs.shape)}")
        if sign.shape != (self.batch_size,):
            raise ValueError(f"Model sign must have shape [{self.batch_size}], got {tuple(sign.shape)}")
        return replace(self, logabs=logabs.detach(), sign=sign.detach(), aux=dict(self.aux))


__all__ = ["Walkers"]
