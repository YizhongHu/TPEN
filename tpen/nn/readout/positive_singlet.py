"""Opt-in positive scalar readouts for the two-electron Hooke singlet."""

from __future__ import annotations

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.real import Feature
from tpen.dependencies import require_torch, require_torch_nn
from tpen.physics.operators import ELECTRON_ELECTRON_COULOMB, HARMONIC_TRAP, KINETIC_ENERGY
from tpen.physics.system import SystemDescriptor

torch = require_torch(feature="positive singlet readout")
nn = require_torch_nn(feature="positive singlet readout")


_HOOKE_SINGLET_OPERATORS = frozenset(
    {KINETIC_ENERGY, HARMONIC_TRAP, ELECTRON_ELECTRON_COULOMB}
)


def validate_hooke_singlet_system(system: object) -> SystemDescriptor:
    """Validate and return the exact physical sector for the positive ansatz."""

    if not isinstance(system, SystemDescriptor):
        raise ValueError("positive singlet ansatz requires a configured SystemDescriptor")
    particle = system.particle
    if (
        particle.n_electrons != 2
        or particle.n_spin_up != 1
        or particle.n_spin_down != 1
    ):
        raise ValueError("positive singlet ansatz requires the Hooke 2e singlet sector")
    operators = frozenset(term.operator_id for term in system.terms)
    if len(system.terms) != 3 or operators != _HOOKE_SINGLET_OPERATORS:
        raise ValueError(
            "positive singlet ansatz requires exactly kinetic + harmonic_trap + electron_electron terms"
        )
    return system


def validate_hooke_singlet_batch(
    batch: ElectronBatch,
    *,
    system_descriptor: SystemDescriptor | None,
) -> None:
    """Validate the narrow sector admitted by the positive-singlet ansatz."""

    declared = system_descriptor if system_descriptor is not None else batch.system
    validate_hooke_singlet_system(declared)
    if batch.system is not None:
        validate_hooke_singlet_system(batch.system)
    if batch.n_electrons != 2:
        raise ValueError("positive singlet ansatz requires exactly two electrons")
    if batch.spins is None:
        raise ValueError("positive singlet ansatz requires explicit opposite-spin labels")
    spins = batch.spins.reshape(-1, 2)
    if not torch.all((spins[:, 0] + spins[:, 1]) == 0):
        raise ValueError("positive singlet ansatz requires one up and one down electron")
class PositiveSingletReadout(nn.Module):
    """Unsigned, exchange-symmetric scalar log-amplitude readout."""

    def __init__(
        self,
        *,
        channels: int,
        system_descriptor: SystemDescriptor | None = None,
    ) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.system_descriptor = (
            None if system_descriptor is None else validate_hooke_singlet_system(system_descriptor)
        )
        self.scalar_readout = nn.Linear(self.channels, 1)

    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        """Return ``log|A| = g_theta`` and a strictly positive sign."""

        validate_hooke_singlet_batch(batch, system_descriptor=self.system_descriptor)
        if 2 not in features:
            raise KeyError("PositiveSingletReadout requires an order-2 Feature block")
        pair = features.blocks[2]
        if pair.ndim != 4 or pair.shape[1] != self.channels or pair.shape[-2:] != (2, 2):
            raise ValueError(
                "PositiveSingletReadout expects an order-2 block shaped "
                f"[batch, {self.channels}, 2, 2], got {tuple(pair.shape)}"
            )
        symmetric_pair = 0.5 * (pair + pair.transpose(-1, -2))
        # Select one off-diagonal pair after symmetrizing.  Pooling the whole
        # block would make the explicit exchange symmetrization unobservable
        # to a mutation because a full mean is already permutation-invariant.
        pooled = symmetric_pair[..., 0, 1]
        logabs = self.scalar_readout(pooled).squeeze(-1)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs), aux={"g": logabs})


HookeSingletPositiveReadout = PositiveSingletReadout
HookeSingletReadout = PositiveSingletReadout


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


__all__ = [
    "HookeSingletPositiveReadout",
    "HookeSingletReadout",
    "PositiveSingletReadout",
    "validate_hooke_singlet_system",
    "validate_hooke_singlet_batch",
]
