"""Autograd-based kinetic-energy estimators."""

from __future__ import annotations

import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.hamiltonian import LocalEnergyResult
from tpen.physics.operators import KINETIC_ENERGY, register_operator
from tpen.physics.system import HamiltonianTermDescriptor


def _extract_logabs(output: WavefunctionOutput) -> torch.Tensor:
    if not isinstance(output, WavefunctionOutput):
        raise TypeError(f"Wavefunction model must return WavefunctionOutput, got {type(output)!r}")
    return output.logabs


def autograd_laplacian(model, batch: ElectronBatch) -> torch.Tensor:
    """Return the Laplacian of ``log|psi|``.

    Parameters
    ----------
    model : callable
        Wavefunction model returning `WavefunctionOutput`.
    batch : ElectronBatch
        Electron batch with positions shaped ``[batch, n_electrons,
        spatial_dim]`` after flattening.

    Returns
    -------
    torch.Tensor
        Laplacian values with shape ``[batch]``.
    """

    batch = batch.flatten_samples()
    positions = batch.positions.detach().clone().requires_grad_(True)
    if positions.ndim != 3:
        raise ValueError("batch.positions must flatten to [batch, n_electrons, spatial_dim]")
    probe_batch = ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
        atomic_configuration=batch.atomic_configuration,
        spins=batch.spins,
        aux=dict(batch.aux),
    )
    output = model(probe_batch)
    logabs = _extract_logabs(output)
    if logabs.shape != (batch.batch_size,):
        raise ValueError(f"logabs must have shape {(batch.batch_size,)}, got {tuple(logabs.shape)}")
    grad = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
    if grad.shape != positions.shape:
        raise ValueError(f"logabs gradient must have shape {tuple(positions.shape)}, got {tuple(grad.shape)}")
    flat_grad = grad.reshape(grad.shape[0], -1)
    laplacian = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
    for idx in range(flat_grad.shape[1]):
        second = torch.autograd.grad(flat_grad[:, idx].sum(), positions, create_graph=True, retain_graph=True)[0]
        laplacian = laplacian + second.reshape(second.shape[0], -1)[:, idx]
    if laplacian.shape != (batch.batch_size,):
        raise ValueError(f"laplacian must have shape {(batch.batch_size,)}, got {tuple(laplacian.shape)}")
    return laplacian


def kinetic_energy_from_logabs(model, batch: ElectronBatch) -> torch.Tensor:
    """Return local kinetic energy from the log-amplitude.

    Parameters
    ----------
    model : callable
        Wavefunction model returning `WavefunctionOutput`.
    batch : ElectronBatch
        Electron batch with positions shaped ``[batch, n_electrons,
        spatial_dim]`` after flattening.

    Returns
    -------
    torch.Tensor
        Kinetic local-energy contribution with shape ``[batch]``.
    """

    value, _ = _kinetic_energy_and_output(model, batch)
    return value


def _kinetic_energy_and_output(
    model,
    batch: ElectronBatch,
) -> tuple[torch.Tensor, WavefunctionOutput]:
    """Return kinetic energy and the exact signed-log output it differentiated."""

    kinetic, output, _ = _kinetic_energy_output_and_per_electron(model, batch)
    return kinetic, output


def _kinetic_energy_output_and_per_electron(
    model,
    batch: ElectronBatch,
) -> tuple[torch.Tensor, WavefunctionOutput, torch.Tensor]:
    """Return total and per-electron kinetic energy from one derivative pass."""

    batch = batch.flatten_samples()
    positions = batch.positions.detach().clone().requires_grad_(True)
    if positions.ndim != 3:
        raise ValueError("batch.positions must flatten to [batch, n_electrons, spatial_dim]")
    probe_batch = _kinetic_probe_batch(batch, positions, 0, batch.batch_size)
    output = model(probe_batch)
    logabs = _extract_logabs(output)
    if logabs.shape != (batch.batch_size,):
        raise ValueError(f"logabs must have shape {(batch.batch_size,)}, got {tuple(logabs.shape)}")
    grad = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
    if grad.shape != positions.shape:
        raise ValueError(f"logabs gradient must have shape {tuple(positions.shape)}, got {tuple(grad.shape)}")
    flat_grad = grad.reshape(grad.shape[0], -1)
    laplacian = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
    electron_laplacians = [
        torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
        for _ in range(batch.n_electrons)
    ]
    for idx in range(flat_grad.shape[1]):
        second = torch.autograd.grad(flat_grad[:, idx].sum(), positions, create_graph=True, retain_graph=True)[0]
        diagonal = second.reshape(second.shape[0], -1)[:, idx]
        laplacian = laplacian + diagonal
        electron = idx // batch.spatial_dim
        electron_laplacians[electron] = electron_laplacians[electron] + diagonal
    if batch.n_electrons == 0:
        # Preserve the typed per-electron contract for the vacuum sector.
        electron_laplacian = torch.empty(
            (batch.batch_size, 0),
            device=grad.device,
            dtype=grad.dtype,
        )
    else:
        electron_laplacian = torch.stack(electron_laplacians, dim=1)
    kinetic = -0.5 * (laplacian + flat_grad.pow(2).sum(dim=1))
    per_electron = -0.5 * (electron_laplacian + grad.pow(2).sum(dim=-1))
    if kinetic.shape != (batch.batch_size,):
        raise ValueError(
            f"kinetic local energy must have shape {(batch.batch_size,)}, "
            f"got {tuple(kinetic.shape)}"
        )
    expected_attribution_shape = (batch.batch_size, batch.n_electrons)
    if tuple(per_electron.shape) != expected_attribution_shape:
        raise ValueError(
            "per-electron kinetic attribution must have shape "
            f"{expected_attribution_shape}, got {tuple(per_electron.shape)}"
        )
    return kinetic, output, per_electron


def per_electron_kinetic_from_logabs_reference(
    model,
    batch: ElectronBatch,
) -> torch.Tensor:
    """Return slow per-electron kinetic attribution for diagnostic validation.

    This readable implementation deliberately evaluates one sample and one
    Cartesian coordinate at a time. It is the independent numerical reference
    for the bounded atlas path and is not used by training.
    """

    flat = batch.flatten_samples()
    rows: list[torch.Tensor] = []
    for sample_index in range(flat.batch_size):
        positions = (
            flat.positions[sample_index : sample_index + 1]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        probe_batch = _kinetic_probe_batch(
            flat,
            positions,
            sample_index,
            sample_index + 1,
        )
        logabs = _extract_logabs(model(probe_batch))
        gradient = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
        electron_values: list[torch.Tensor] = []
        for electron in range(flat.n_electrons):
            laplacian = torch.zeros((), device=flat.device, dtype=flat.dtype)
            for coordinate in range(flat.spatial_dim):
                second = torch.autograd.grad(
                    gradient[0, electron, coordinate],
                    positions,
                    retain_graph=True,
                )[0]
                laplacian = laplacian + second[0, electron, coordinate]
            gradient_squared = gradient[0, electron].square().sum()
            electron_values.append(-0.5 * (laplacian + gradient_squared))
        rows.append(torch.stack(electron_values).detach())
    if not rows:
        return torch.empty(
            (0, flat.n_electrons),
            device=flat.device,
            dtype=flat.dtype,
        )
    return torch.stack(rows)


def per_electron_kinetic_from_logabs(
    model,
    batch: ElectronBatch,
) -> torch.Tensor:
    """Return per-electron attribution from the shared kinetic derivative pass."""

    _, _, per_electron = _kinetic_energy_output_and_per_electron(model, batch)
    return per_electron.detach()


def _kinetic_probe_batch(
    batch: ElectronBatch,
    positions: torch.Tensor,
    start: int,
    end: int,
) -> ElectronBatch:
    """Build a graph-carrying kinetic probe without changing batch semantics."""

    nuclear_positions = batch.nuclear_positions
    if nuclear_positions is not None and nuclear_positions.ndim == 3:
        nuclear_positions = nuclear_positions[start:end]
    nuclear_charges = batch.nuclear_charges
    if nuclear_charges is not None and nuclear_charges.ndim == 2:
        nuclear_charges = nuclear_charges[start:end]
    spins = None if batch.spins is None else batch.spins[start:end]
    return ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        atomic_configuration=batch.atomic_configuration,
        spins=spins,
        aux=dict(batch.aux),
    )


@register_operator(KINETIC_ENERGY)
class KineticEnergy:
    """Hamiltonian term for the quantum kinetic energy operator."""

    name = "kinetic"
    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        value, output, per_electron = _kinetic_energy_output_and_per_electron(
            wavefunction,
            batch,
        )
        return LocalEnergyResult(
            total=value,
            terms={self.name: value},
            wavefunction_output=output,
            per_electron_kinetic=per_electron,
        )

    def describe_physical_operator(self) -> HamiltonianTermDescriptor:
        """Declare the fixed-coefficient kinetic operator; no free parameters."""

        return HamiltonianTermDescriptor(operator_id=self.operator_id, slot=self.name)


__all__ = [
    "KineticEnergy",
    "autograd_laplacian",
    "kinetic_energy_from_logabs",
    "per_electron_kinetic_from_logabs",
    "per_electron_kinetic_from_logabs_reference",
]
