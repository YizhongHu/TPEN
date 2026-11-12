"""Potential-energy Hamiltonian terms.

This module keeps two API generations side by side for the electron-nucleus
and nucleus-nucleus terms. `ElectronNucleusPotential` and
`NucleusNucleusPotential` are the canonical Hamiltonian API: each is
constructed once from a fixed `AtomicConfiguration` -- the sole nuclear-
geometry authority for that term -- and one shared geometry applies to every
sample in a batch. `ElectronNucleusInteraction` and `NucleusNucleusInteraction`
are the legacy, supported minor-release compatibility surface: they read
nuclear geometry from batch-transported metadata
(`ElectronBatch.nuclear_positions`/`nuclear_charges`) and, unlike the
constructor-owned `*Potential` API, additionally accept *per-configuration*
nuclear geometry that varies within one batch (shape
``[batch, n_nuclei, spatial_dim]``/``[batch, n_nuclei]``) -- a broader batch
geometry capability the canonical API does not need. Neither legacy class is
deprecated in this minor version; there is no runtime warning yet.

Each pair (`ElectronNucleusInteraction`/`ElectronNucleusPotential`,
`NucleusNucleusInteraction`/`NucleusNucleusPotential`) shares one private,
pure Coulomb arithmetic kernel (`_electron_nucleus_coulomb_potential`,
`_nucleus_nucleus_coulomb_energy`). The kernels take already-normalized
tensors; each class keeps its own distinct shape validation, error contract,
and geometry-authority semantics around the shared arithmetic.
"""

from __future__ import annotations

import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, electron_nuclear_distances, pairwise_distances
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.physics.operators import (
    ELECTRON_ELECTRON_COULOMB,
    ELECTRON_NUCLEUS_COULOMB,
    HARMONIC_TRAP,
    NUCLEUS_NUCLEUS_COULOMB,
    register_operator_geometry,
    register_operator,
)
from tpen.physics.system import HamiltonianTermDescriptor


@register_operator(HARMONIC_TRAP)
class HarmonicTrap:
    """Hamiltonian term for a harmonic confinement potential.

    .. math:: V_\\mathrm{trap} = \\tfrac{1}{2}\\omega^2 \\sum_i r_i^2

    Parameters
    ----------
    omega : float, optional
        Trap frequency.
    """

    name = "harmonic_trap"
    def __init__(self, omega: float = 1.0) -> None:
        self.omega = omega

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        positions = batch.flatten_samples().positions
        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        value = 0.5 * (self.omega**2) * positions.square().sum(dim=(1, 2))
        if value.shape != (positions.shape[0],):
            raise ValueError(f"harmonic-trap energy must have shape {(positions.shape[0],)}, got {tuple(value.shape)}")
        return LocalEnergyResult(total=value, terms={self.name: value})

    def describe_physical_operator(self) -> HamiltonianTermDescriptor:
        """Declare the configured trap frequency; no nuclear resource."""

        return HamiltonianTermDescriptor(
            operator_id=self.operator_id,
            slot=self.name,
            parameters={"omega": float(self.omega)},
        )


@register_operator(ELECTRON_ELECTRON_COULOMB)
class ElectronElectronInteraction:
    """Hamiltonian term for Coulomb electron-electron repulsion.

    .. math:: V_\\mathrm{ee} = \\sum_{i<j} \\frac{1}{r_{ij}}

    Parameters
    ----------
    eps : float, optional
        Distance floor retained for backwards compatibility. A non-zero value
        is unproven and may be inconsistent: the cusp factor and Coulomb
        potential do not necessarily apply the same floor, yielding a hybrid
        Hamiltonian: a clipped potential evaluated with the boundary condition
        of an unclipped Coulomb potential. Inspect and validate before using a
        non-zero value. The
        finite-eps electron-electron case is UNMEASURED.

    Warning
    -------
    A positive floor removes the potential divergence and introduces a
    divergence into the total; it is not numerical safety. The electron-nucleus
    measurement found ``E(r; eps) = Z/r - Z/eps - Z^2/2`` for ``0 < r < eps``
    across three eps scales and three directions (normalized error <=
    ``1.11e-16``). The electron-electron finite-eps case is UNMEASURED; a
    constant offset from identical clamps has not been tested and must not be
    assumed benign.
    """

    name = "electron_electron"
    def __init__(self, eps: float = 0.0) -> None:
        self.eps = eps

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        positions = batch.flatten_samples().positions
        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        distances = pairwise_distances(positions, eps=self.eps).squeeze(-1)
        expected_distances = (positions.shape[0], positions.shape[1], positions.shape[1])
        if distances.shape != expected_distances:
            raise ValueError(f"pairwise distances must have shape {expected_distances}, got {tuple(distances.shape)}")
        tri = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        value = distances.reciprocal().masked_fill(~tri, 0.0).sum(dim=(1, 2))
        if value.shape != (positions.shape[0],):
            raise ValueError(f"electron-electron energy must have shape {(positions.shape[0],)}, got {tuple(value.shape)}")
        return LocalEnergyResult(total=value, terms={self.name: value})

    def describe_physical_operator(self) -> HamiltonianTermDescriptor:
        """Declare the configured distance floor; no nuclear resource."""

        return HamiltonianTermDescriptor(
            operator_id=self.operator_id,
            slot=self.name,
            parameters={"eps": float(self.eps)},
        )


@register_operator(ELECTRON_NUCLEUS_COULOMB)
class ElectronNucleusInteraction:
    """Hamiltonian term for Coulomb electron-nucleus attraction.

    .. math:: V_\\mathrm{en} = -\\sum_{i,A} \\frac{Z_A}{|r_i - R_A|}

    Parameters
    ----------
    nuclear_positions : torch.Tensor or None, optional
        Deprecated compatibility metadata. When supplied, it must agree
        exactly with the nuclear positions carried by every evaluated batch.
    nuclear_charges : torch.Tensor or None, optional
        Deprecated compatibility metadata paired with ``nuclear_positions``.
    eps : float, optional
        Distance floor retained for backwards compatibility. A non-zero value
        is unproven and may be inconsistent: the cusp factor and Coulomb
        potential do not necessarily apply the same floor, yielding a hybrid
        Hamiltonian: a clipped potential evaluated with the boundary condition
        of an unclipped Coulomb potential. Inspect and validate before using a
        non-zero value. The
        finite-eps electron-electron case is UNMEASURED.

    Warning
    -------
    A positive floor removes the potential divergence and introduces a
    divergence into the total; it is not numerical safety. Measurement found
    ``E(r; eps) = Z/r - Z/eps - Z^2/2`` for ``0 < r < eps`` across three eps
    scales and three directions, with normalized error <= ``1.11e-16``. The
    electron-electron finite-eps case is UNMEASURED; a constant offset from
    identical clamps has not been tested and must not be assumed benign.
    """

    name = "electron_nucleus"
    def __init__(
        self,
        nuclear_positions: torch.Tensor | None = None,
        nuclear_charges: torch.Tensor | None = None,
        eps: float = 0.0,
    ) -> None:
        if (nuclear_positions is None) != (nuclear_charges is None):
            raise ValueError("nuclear_positions and nuclear_charges must be provided together")
        self.nuclear_positions = None if nuclear_positions is None else torch.as_tensor(nuclear_positions).detach().clone()
        self.nuclear_charges = None if nuclear_charges is None else torch.as_tensor(nuclear_charges).detach().clone()
        if self.nuclear_positions is not None and self.nuclear_positions.ndim != 2:
            raise ValueError("nuclear_positions must have shape [n_nuclei, spatial_dim]")
        if self.nuclear_charges is not None and self.nuclear_charges.ndim != 1:
            raise ValueError("nuclear_charges must have shape [n_nuclei]")
        if self.nuclear_positions is not None and self.nuclear_positions.shape[0] != self.nuclear_charges.shape[0]:
            raise ValueError("nuclear_positions and nuclear_charges must agree on n_nuclei")
        self.eps = eps

    def declared_nuclear_geometry(self) -> AtomicConfiguration | None:
        """Return legacy constructor geometry, if the term declared one."""

        if self.nuclear_positions is None:
            return None
        assert self.nuclear_charges is not None
        return AtomicConfiguration(self.nuclear_positions, self.nuclear_charges)

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        flat = batch.flatten_samples()
        positions = flat.positions
        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        _require_nuclear_context(flat)
        self._validate_legacy_context(flat)
        distances = electron_nuclear_distances(flat, eps=self.eps)
        potential = _electron_nucleus_coulomb_potential(distances, flat.nuclear_charges)
        expected_potential = (positions.shape[0], positions.shape[1])
        if potential.shape != expected_potential:
            raise ValueError(f"electron-nucleus potential must have shape {expected_potential}, got {tuple(potential.shape)}")
        value = -potential.sum(dim=1)
        if value.shape != (positions.shape[0],):
            raise ValueError(f"electron-nucleus energy must have shape {(positions.shape[0],)}, got {tuple(value.shape)}")
        return LocalEnergyResult(total=value, terms={self.name: value})

    def _validate_legacy_context(self, batch: ElectronBatch) -> None:
        """Reject legacy constructor metadata that disagrees with a batch."""

        if self.nuclear_positions is None:
            return
        assert self.nuclear_charges is not None
        positions = self.nuclear_positions.to(device=batch.device, dtype=batch.dtype)
        charges = self.nuclear_charges.to(device=batch.device, dtype=batch.dtype)
        if not torch.equal(positions, batch.nuclear_positions) or not torch.equal(charges, batch.nuclear_charges):
            raise ValueError("legacy ElectronNucleusInteraction nuclear metadata must agree exactly with batch context")


@register_operator(NUCLEUS_NUCLEUS_COULOMB)
class NucleusNucleusInteraction:
    r"""Hamiltonian term for Born--Oppenheimer nuclear repulsion.

    .. math:: V_\mathrm{nn} = \sum_{A<B} \frac{Z_A Z_B}{|R_A - R_B|}

    Nuclear coordinates and charges are read from the explicit metadata on
    ``ElectronBatch``.  Shared ``[n_nuclei, spatial_dim]`` metadata and
    per-configuration ``[batch, n_nuclei, spatial_dim]`` metadata are both
    supported.
    """

    name = "nucleus_nucleus"
    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        """Return the pairwise nuclear repulsion for each batch sample."""

        flat = batch.flatten_samples()
        positions = flat.nuclear_positions
        charges = flat.nuclear_charges
        if positions is None:
            raise ValueError("NucleusNucleusInteraction requires batch.nuclear_positions")
        if charges is None:
            raise ValueError("NucleusNucleusInteraction requires batch.nuclear_charges")

        positions = positions.to(device=flat.device, dtype=flat.dtype)
        charges = charges.to(device=flat.device, dtype=flat.dtype)
        if positions.ndim == 2:
            if positions.shape[-1] != flat.spatial_dim:
                raise ValueError("nuclear positions must match batch spatial dimension")
            positions = positions.unsqueeze(0).expand(flat.batch_size, -1, -1)
        elif positions.ndim == 3:
            if positions.shape[0] != flat.batch_size or positions.shape[-1] != flat.spatial_dim:
                raise ValueError("batched nuclear positions must match batch size and spatial dimension")
        else:
            raise ValueError("nuclear positions must have shape [n_nuclei, dim] or [batch, n_nuclei, dim]")

        if charges.ndim == 1:
            charges = charges.unsqueeze(0).expand(flat.batch_size, -1)
        elif charges.ndim != 2 or charges.shape[0] != flat.batch_size:
            raise ValueError("nuclear charges must have shape [n_nuclei] or [batch, n_nuclei]")
        if positions.shape[1] != charges.shape[1]:
            raise ValueError("nuclear positions and charges must agree on n_nuclei")
        if not torch.isfinite(positions).all() or not torch.isfinite(charges).all():
            raise ValueError("nuclear positions and charges must be finite")

        value = _nucleus_nucleus_coulomb_energy(positions, charges, check_collisions=True)
        return LocalEnergyResult(total=value, terms={self.name: value})


@register_operator(ELECTRON_NUCLEUS_COULOMB)
class ElectronNucleusPotential:
    """Hamiltonian term for Coulomb electron-nucleus attraction.

    .. math:: V_\\mathrm{en} = -\\sum_{i,A} \\frac{Z_A}{|r_i - R_A|}

    Unlike `ElectronNucleusInteraction`, this term is constructed directly
    from an `AtomicConfiguration`, which is the sole authority for nuclear
    geometry: values are computed from `atoms`, not from batch-transported
    tensors. A batch's own transported nuclear context (`nuclear_positions`,
    `nuclear_charges`, `atomic_configuration`), when present, must agree with
    `atoms` exactly, or construction-time authority and pipeline transport
    have silently diverged and evaluation fails loudly instead of picking one
    side.

    Parameters
    ----------
    atoms : AtomicConfiguration
        Fixed nuclear geometry authority for this term.
    eps : float, optional
        Distance floor retained for backwards compatibility. A non-zero value
        is unproven and may be inconsistent: the cusp factor and Coulomb
        potential do not necessarily apply the same floor, yielding a hybrid
        Hamiltonian: a clipped potential evaluated with the boundary condition
        of an unclipped Coulomb potential. Inspect and validate before using a
        non-zero value. The
        finite-eps electron-electron case is UNMEASURED.

    Warning
    -------
    A positive floor removes the potential divergence and introduces a
    divergence into the total; it is not numerical safety. Measurement found
    ``E(r; eps) = Z/r - Z/eps - Z^2/2`` for ``0 < r < eps`` across three eps
    scales and three directions, with normalized error <= ``1.11e-16``. The
    electron-electron finite-eps case is UNMEASURED; a constant offset from
    identical clamps has not been tested and must not be assumed benign.
    """

    name = "electron_nucleus"
    def __init__(self, atoms: object, eps: float = 0.0) -> None:
        if not isinstance(atoms, AtomicConfiguration):
            raise TypeError(f"{type(self).__name__} requires an AtomicConfiguration, got {type(atoms).__name__}")
        self.atoms = atoms
        self.eps = eps

    def declared_nuclear_geometry(self) -> AtomicConfiguration:
        """Return the construction-time geometry authority for this term."""

        return self.atoms

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        flat = batch.flatten_samples()
        positions = flat.positions
        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        _validate_batch_atoms_context(self.atoms, flat, term_name=type(self).__name__)
        atoms = self.atoms.to(device=flat.device, dtype=flat.dtype)
        distances = electron_nuclear_distances(flat, eps=self.eps, nuclear_positions=atoms.positions)
        potential = _electron_nucleus_coulomb_potential(distances, atoms.charges)
        value = -potential.sum(dim=1)
        if value.shape != (positions.shape[0],):
            raise ValueError(f"electron-nucleus energy must have shape {(positions.shape[0],)}, got {tuple(value.shape)}")
        return LocalEnergyResult(total=value, terms={self.name: value})

    def describe_physical_operator(self) -> HamiltonianTermDescriptor:
        """Declare the distance floor and the construction-time nuclear resource."""

        return HamiltonianTermDescriptor(
            operator_id=self.operator_id,
            slot=self.name,
            parameters={"eps": float(self.eps)},
            resource_id=self.atoms.content_id(),
            resource=self.atoms,
        )


register_operator_geometry(
    ElectronNucleusInteraction,
    ElectronNucleusInteraction.declared_nuclear_geometry,
)
register_operator_geometry(
    ElectronNucleusPotential,
    ElectronNucleusPotential.declared_nuclear_geometry,
)


@register_operator(NUCLEUS_NUCLEUS_COULOMB)
class NucleusNucleusPotential:
    r"""Hamiltonian term for Born--Oppenheimer nuclear repulsion.

    .. math:: V_\mathrm{nn} = \sum_{A<B} \frac{Z_A Z_B}{|R_A - R_B|}

    Unlike `NucleusNucleusInteraction`, this term is constructed directly
    from an `AtomicConfiguration`, which is the sole authority for nuclear
    geometry: the pairwise sum is computed once from `atoms` and broadcast
    across the batch. A batch's own transported nuclear context, when
    present, must agree with `atoms` exactly, or evaluation fails loudly.

    Parameters
    ----------
    atoms : AtomicConfiguration
        Fixed nuclear geometry authority for this term.
    """

    name = "nucleus_nucleus"
    def __init__(self, atoms: object) -> None:
        if not isinstance(atoms, AtomicConfiguration):
            raise TypeError(f"{type(self).__name__} requires an AtomicConfiguration, got {type(atoms).__name__}")
        self.atoms = atoms

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        """Return the pairwise nuclear repulsion for each batch sample."""

        flat = batch.flatten_samples()
        _validate_batch_atoms_context(self.atoms, flat, term_name=type(self).__name__)
        atoms = self.atoms.to(device=flat.device, dtype=flat.dtype)
        pair_sum = _nucleus_nucleus_coulomb_energy(
            atoms.positions.unsqueeze(0), atoms.charges.unsqueeze(0), check_collisions=False
        )[0]
        value = pair_sum.expand(flat.batch_size)
        return LocalEnergyResult(total=value, terms={self.name: value})

    def describe_physical_operator(self) -> HamiltonianTermDescriptor:
        """Declare the construction-time nuclear resource; no free parameters."""

        return HamiltonianTermDescriptor(
            operator_id=self.operator_id,
            slot=self.name,
            resource_id=self.atoms.content_id(),
            resource=self.atoms,
        )


def _electron_nucleus_coulomb_potential(distances: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
    """Return ``sum_A Z_A / |r_i - R_A|`` per electron.

    Shared Coulomb arithmetic behind `ElectronNucleusInteraction` (batch-transported
    charges) and `ElectronNucleusPotential` (`AtomicConfiguration`-owned charges).

    Parameters
    ----------
    distances : torch.Tensor
        Electron-nucleus distances, shape ``[batch, n_electrons, n_nuclei]``.
    charges : torch.Tensor
        Nuclear charges, shape ``[n_nuclei]`` (shared across the batch) or
        ``[batch, n_nuclei]`` (per-sample).

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch, n_electrons]``.
    """

    if charges.ndim == 1:
        charge_view = charges.reshape(1, 1, -1)
    elif charges.ndim == 2:
        charge_view = charges.unsqueeze(1)
    else:
        raise ValueError("nuclear charges must have shape [n_nuclei] or [batch, n_nuclei]")
    return (charge_view / distances).sum(dim=-1)


def _nucleus_nucleus_coulomb_energy(positions: torch.Tensor, charges: torch.Tensor, *, check_collisions: bool) -> torch.Tensor:
    """Return ``sum_{A<B} Z_A Z_B / |R_A - R_B|`` per batch sample.

    Shared Coulomb arithmetic behind `NucleusNucleusInteraction` (batch-transported
    geometry) and `NucleusNucleusPotential` (`AtomicConfiguration`-owned geometry).

    Parameters
    ----------
    positions : torch.Tensor
        Nuclear positions, shape ``[batch, n_nuclei, spatial_dim]``.
    charges : torch.Tensor
        Nuclear charges, shape ``[batch, n_nuclei]``.
    check_collisions : bool
        Whether to raise when any nuclear pair distance is non-positive. Legacy
        batch-transported geometry may vary per sample and is checked; canonical
        `AtomicConfiguration`-owned geometry is validated at construction and is
        not re-checked here.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``[batch]``.
    """

    batch_size, n_nuclei = positions.shape[0], positions.shape[1]
    if n_nuclei < 2:
        return torch.zeros(batch_size, device=positions.device, dtype=positions.dtype)
    distances = torch.linalg.norm(positions.unsqueeze(2) - positions.unsqueeze(1), dim=-1)
    pair_mask = torch.triu(torch.ones((n_nuclei, n_nuclei), device=positions.device, dtype=torch.bool), diagonal=1)
    if check_collisions and (distances[:, pair_mask] <= 0).any():
        raise ValueError("nuclear positions contain colliding nuclei")
    pair_values = charges.unsqueeze(2) * charges.unsqueeze(1) / distances
    return pair_values[:, pair_mask].sum(dim=1)


def _require_nuclear_context(batch: ElectronBatch) -> None:
    """Require the typed nuclear context owned by an electron batch."""

    if batch.nuclear_positions is None:
        raise ValueError("ElectronNucleusInteraction requires batch.nuclear_positions")
    if batch.nuclear_charges is None:
        raise ValueError("ElectronNucleusInteraction requires batch.nuclear_charges")


def _validate_batch_atoms_context(atoms: AtomicConfiguration, batch: ElectronBatch, *, term_name: str) -> None:
    """Require any transported nuclear context on `batch` to agree with `atoms`.

    `atoms` is the sole construction-time authority; this only guards against
    the batch's own (optional) transported nuclear tensors or typed
    `atomic_configuration` reference silently diverging from it.
    """

    if batch.atomic_configuration is not None:
        is_close, _ = atoms.compare(batch.atomic_configuration)
        if not is_close:
            raise ValueError(f"{term_name} atoms must agree exactly with batch.atomic_configuration")
    if batch.nuclear_positions is not None:
        positions = batch.nuclear_positions
        reference = atoms.positions.to(device=positions.device, dtype=positions.dtype)
        if positions.ndim == 2:
            candidate = positions
        else:
            candidate = positions[0]
            if not torch.all(positions == candidate.unsqueeze(0)):
                raise ValueError(f"{term_name} requires batch.nuclear_positions constant across the batch")
        if candidate.shape != reference.shape or not torch.equal(candidate, reference):
            raise ValueError(f"{term_name} atoms must agree exactly with batch.nuclear_positions")
    if batch.nuclear_charges is not None:
        charges = batch.nuclear_charges
        reference = atoms.charges.to(device=charges.device, dtype=charges.dtype)
        if charges.ndim == 1:
            candidate = charges
        else:
            candidate = charges[0]
            if not torch.all(charges == candidate.unsqueeze(0)):
                raise ValueError(f"{term_name} requires batch.nuclear_charges constant across the batch")
        if candidate.shape != reference.shape or not torch.equal(candidate, reference):
            raise ValueError(f"{term_name} atoms must agree exactly with batch.nuclear_charges")


__all__ = [
    "ElectronElectronInteraction",
    "ElectronNucleusInteraction",
    "ElectronNucleusPotential",
    "HarmonicTrap",
    "NucleusNucleusInteraction",
    "NucleusNucleusPotential",
]
