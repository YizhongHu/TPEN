"""Opt-in positive scalar readouts for the two-electron Hooke singlet."""

from __future__ import annotations

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.real import Feature
from tpen.dependencies import require_torch, require_torch_nn

torch = require_torch(feature="positive singlet readout")
nn = require_torch_nn(feature="positive singlet readout")


def validate_hooke_singlet_batch(batch: ElectronBatch, *, system_name: str) -> None:
    """Validate the narrow sector admitted by the positive-singlet ansatz."""

    if system_name.lower() not in {"hooke_pair", "hooke-2e", "hooke2", "hooke_2e"}:
        raise ValueError("positive singlet ansatz is only defined for system_name='hooke_pair'")
    if batch.n_electrons != 2:
        raise ValueError("positive singlet ansatz requires exactly two electrons")
    if batch.spins is None:
        raise ValueError("positive singlet ansatz requires explicit opposite-spin labels")
    spins = batch.spins.reshape(-1, 2)
    if not torch.all((spins[:, 0] + spins[:, 1]) == 0):
        raise ValueError("positive singlet ansatz requires one up and one down electron")
    declared = batch.system if batch.system is not None else batch.aux.get("system_name")
    if declared is None:
        return
    label = declared if isinstance(declared, str) else getattr(declared, "name", None)
    if label is not None and "hooke" not in str(label).lower():
        raise ValueError("positive singlet ansatz refuses non-Hooke systems")
    particle = getattr(declared, "particle", None)
    if label is None and hasattr(declared, "terms"):
        operators = " ".join(
            f"{getattr(getattr(term, 'operator_id', None), 'namespace', '')}."
            f"{getattr(getattr(term, 'operator_id', None), 'name', '')}"
            for term in declared.terms
        ).lower()
        if "harmonic" not in operators and "hooke" not in operators:
            raise ValueError("positive singlet ansatz refuses non-Hooke systems")
    if particle is not None and (
        getattr(particle, "n_electrons", None) != 2
        or getattr(particle, "n_spin_up", None) != 1
        or getattr(particle, "n_spin_down", None) != 1
    ):
        raise ValueError("positive singlet ansatz requires the Hooke 2e singlet sector")


class PositiveSingletReadout(nn.Module):
    """Unsigned, exchange-symmetric scalar log-amplitude readout."""

    def __init__(self, *, channels: int, system_name: str = "hooke_pair") -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.system_name = str(system_name)
        if self.system_name.lower() not in {"hooke_pair", "hooke-2e", "hooke2", "hooke_2e"}:
            raise ValueError("positive singlet ansatz is only defined for system_name='hooke_pair'")
        self.scalar_readout = nn.Linear(self.channels, 1)

    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        """Return ``log|A| = g_theta`` and a strictly positive sign."""

        validate_hooke_singlet_batch(batch, system_name=self.system_name)
        if 2 not in features:
            raise KeyError("PositiveSingletReadout requires an order-2 Feature block")
        pair = features.blocks[2]
        if pair.ndim != 4 or pair.shape[1] != self.channels or pair.shape[-2:] != (2, 2):
            raise ValueError(
                "PositiveSingletReadout expects an order-2 block shaped "
                f"[batch, {self.channels}, 2, 2], got {tuple(pair.shape)}"
            )
        symmetric_pair = 0.5 * (pair + pair.transpose(-1, -2))
        pooled = symmetric_pair.mean(dim=(-1, -2))
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
    "validate_hooke_singlet_batch",
]
