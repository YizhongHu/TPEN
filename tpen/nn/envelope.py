"""Legacy additive envelope factors for wavefunction log-amplitudes.

`Envelope` and `AdditiveEnvelope` are a supported minor-release
compatibility surface: their constructor, forward behavior, Hydra target,
and `ModuleList` state-dict keys must not change, and neither carries a
runtime deprecation warning in this minor version. `GaussianConfinement` and
`HookeGaussianConfinement` are the concrete decay/confinement envelopes built
on this legacy interface.

The canonical, new-generation types --
`tpen.nn.factor.LogAmplitudeFactor`/`AdditiveCusp` and
`tpen.nn.cusp.ElectronNucleusCusp`/`ElectronElectronCusp` -- now live in their
own modules; this module re-exports them lazily (see `__getattr__` below) so
every import path that previously resolved through `tpen.nn.envelope` keeps
working unchanged. They compose independently and do not replace the legacy
envelope stack above. `TPENWaveFunction` sums both generations in one
post-readout factor pipeline (see `tpen/nn/tpen_wave_function.py`), and
`TPENWaveFunction.factors` is the canonical composition seam for
`LogAmplitudeFactor` terms.

`FeatureEnvelope` is reserved for a future typed feature-space transform (a
distinct concept from the multiplicative coordinate `Envelope` above). It must
never be introduced as an alias or rename of `Envelope`.
"""

from __future__ import annotations

from collections.abc import Iterable

from tpen.data.batch import ElectronBatch
from tpen.dependencies import require_torch, require_torch_functional, require_torch_nn
from tpen.nn.factor import _check_factor_tensor as _check_envelope_tensor
from tpen.nn.factor import _inverse_softplus

torch = require_torch(feature="TPEN envelope modules")
nn = require_torch_nn(feature="TPEN envelope modules")
F = require_torch_functional(feature="TPEN envelope modules")


class Envelope(nn.Module):
    """Template for additive log-amplitude envelope factors.

    An envelope accepts an :class:`ElectronBatch` and returns a scalar
    contribution to ``log |psi|`` for each flattened configuration. Smooth
    confinement tails and short-range cusp factors both use this interface.

    Parameters
    ----------
    enabled : bool, optional
        Whether this envelope contributes to the output.
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return a flattened-batch envelope contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Electron batch whose sample axes may be higher rank.

        Returns
        -------
        torch.Tensor
            Envelope contribution with shape ``[batch]`` after sample
            flattening.
        """

        flat_batch = batch.flatten_samples()
        if not self.enabled:
            return torch.zeros(flat_batch.batch_size, device=flat_batch.device, dtype=flat_batch.dtype)
        output = self.envelope_value(flat_batch)
        _check_envelope_tensor(output, flat_batch, name=type(self).__name__)
        return output

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the enabled envelope contribution for a flattened batch.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Envelope contribution with shape ``[batch]``.
        """

        raise NotImplementedError("Envelope.envelope_value must be implemented by subclasses")


class AdditiveEnvelope(Envelope):
    """Envelope that sums a sequence of envelope components.

    Parameters
    ----------
    envelopes : iterable of torch.nn.Module, optional
        Envelope modules whose outputs are added. Each component must accept an
        :class:`ElectronBatch` and return a tensor of shape ``[batch]``.
    enabled : bool, optional
        Whether this envelope contributes to the output.
    """

    def __init__(self, envelopes: Iterable[nn.Module] = (), enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.envelopes = nn.ModuleList(tuple(envelopes))

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the sum of all component envelope contributions."""

        total = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        for index, envelope in enumerate(self.envelopes):
            value = envelope(batch)
            _check_envelope_tensor(value, batch, name=f"envelopes[{index}]")
            total = total + value
        return total


class GaussianConfinement(Envelope):
    """Smooth Gaussian envelope for harmonically trapped systems.

    This contributes

    ``log |psi| <- log |psi| - coefficient * sum_i |r_i|^2``.

    For a Hooke or harmonic-oscillator tail with frequency ``omega``, the fixed
    ground-state Gaussian coefficient is ``omega / 2``.

    Parameters
    ----------
    enabled : bool, optional
        Whether this envelope contributes to the output.
    coefficient : float, optional
        Nonnegative coefficient multiplying ``sum_i |r_i|^2``.
    trainable : bool, optional
        Whether to optimize the coefficient through a softplus
        parametrization. A trainable coefficient is strictly positive.
    """

    def __init__(
        self,
        enabled: bool = True,
        coefficient: float = 0.0,
        trainable: bool = False,
    ) -> None:
        super().__init__(enabled=enabled)
        if coefficient < 0.0:
            raise ValueError(f"coefficient must be nonnegative, got {coefficient}")
        self.trainable = bool(trainable)
        if self.trainable:
            self.raw_coefficient = nn.Parameter(_inverse_softplus(float(coefficient)))
        else:
            self.register_buffer(
                "_coefficient",
                torch.tensor(float(coefficient), dtype=torch.float64),
                persistent=False,
            )

    @property
    def coefficient(self) -> torch.Tensor:
        """Return the nonnegative harmonic-confinement coefficient."""

        if self.trainable:
            return F.softplus(self.raw_coefficient)
        return self._coefficient

    def envelope_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the smooth harmonic envelope contribution."""

        radius_squared = batch.positions.square().sum(dim=(1, 2))
        output = -self.coefficient.to(device=batch.device, dtype=batch.dtype) * radius_squared
        assert output.shape == (batch.batch_size,)
        return output

    def scalar_diagnostics(self) -> dict[str, float]:
        """Expose the physical confinement strength to FactorScalars."""
        values = {"coefficient": float(self.coefficient.detach().item())}
        if self.trainable:
            values["raw_coefficient"] = float(self.raw_coefficient.detach().item())
        return values


class HookeGaussianConfinement(GaussianConfinement):
    """Gaussian ground-state envelope for the Hooke / harmonic oscillator.

    This is :class:`GaussianConfinement` parametrized by the oscillator
    frequency ``omega`` instead of a raw coefficient. The fixed ground-state
    Gaussian uses ``coefficient = omega / 2``, contributing

    ``log |psi| <- log |psi| - (omega / 2) * sum_i |r_i|^2``.

    It supplies the common output-side asymptotic prior shared by every main
    architecture choice in the pair-stability study.

    Parameters
    ----------
    omega : float
        Positive oscillator frequency.
    enabled : bool, optional
        Whether this envelope contributes to the output.
    trainable : bool, optional
        Whether to optimize the coefficient through a softplus parametrization.
    """

    def __init__(self, *, omega: float, enabled: bool = True, trainable: bool = False) -> None:
        if omega <= 0.0:
            raise ValueError(f"omega must be positive, got {omega}")
        super().__init__(enabled=enabled, coefficient=float(omega) / 2.0, trainable=trainable)
        self.omega = float(omega)


# Names that moved to `tpen.nn.factor` / `tpen.nn.cusp`, kept resolvable from
# this module (attribute access, `from tpen.nn.envelope import ...`, and
# `import *`) via lazy re-export so this module never has to import those
# modules eagerly (which import `Envelope` from here).
_FACTOR_MODULE_NAMES = frozenset({"LogAmplitudeFactor", "AdditiveCusp"})
_CUSP_MODULE_NAMES = frozenset(
    {
        "ElectronNucleusCuspLaw",
        "LinearElectronNucleusCuspLaw",
        "ElectronNucleusCusp",
        "ElectronElectronCusp",
        "rational_pair_cusp",
    }
)


def __getattr__(name: str) -> object:
    if name in _FACTOR_MODULE_NAMES:
        import tpen.nn.factor as _factor

        return getattr(_factor, name)
    if name in _CUSP_MODULE_NAMES:
        import tpen.nn.cusp as _cusp

        return getattr(_cusp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdditiveCusp",
    "AdditiveEnvelope",
    "ElectronElectronCusp",
    "ElectronNucleusCusp",
    "ElectronNucleusCuspLaw",
    "Envelope",
    "GaussianConfinement",
    "HookeGaussianConfinement",
    "LinearElectronNucleusCuspLaw",
    "LogAmplitudeFactor",
    "rational_pair_cusp",
]
