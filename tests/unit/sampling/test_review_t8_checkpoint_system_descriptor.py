"""Delta-review detectors for sampler checkpoint serialization of system metadata."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import WavefunctionOutput
from tpen.physics.system import SystemDescriptor
from tpen.sampling.metropolis import MetropolisSampler


_ROOT = Path(__file__).resolve().parents[3]
_QHO_CONFIG = _ROOT / "experiments/atomistic/stage1/configs/qho2e/eval.yaml"
_LEGACY_STATE_KEYS = {
    "walkers",
    "atomic_configuration",
    "has_burned_in",
    "acceptance_rate",
    "generator_state",
    "generator_device",
}


class _ConstantWavefunction(torch.nn.Module):
    def forward(self, batch):
        size = batch.positions.shape[0]
        value = torch.zeros(size, dtype=batch.positions.dtype, device=batch.positions.device)
        return WavefunctionOutput(logabs=value, sign=torch.ones_like(value))


def _descriptor(*, omega: float = 0.5) -> SystemDescriptor:
    cfg = OmegaConf.load(_QHO_CONFIG)
    # Set the source term before resolving the descriptor interpolation; after
    # resolution, ``system_descriptor.terms`` is an independent container.
    cfg.hamiltonian_terms.harmonic_trap.omega = omega
    OmegaConf.resolve(cfg)
    return instantiate(OmegaConf.create(cfg.system_descriptor))


def _sampler(*, system: SystemDescriptor | None = None) -> MetropolisSampler:
    cfg = OmegaConf.load(_QHO_CONFIG)
    OmegaConf.resolve(cfg)
    sampler_cfg = OmegaConf.create(cfg.evaluation_sampler)
    sampler = instantiate(sampler_cfg)
    if system is not None:
        sampler.system = system
    return sampler


@pytest.mark.parametrize(
    "with_system",
    [False, True],
    ids=["system-none", "real-system-descriptor"],
)
def test_sampler_state_torch_save_load_preserves_metadata_and_continuation(with_system: bool) -> None:
    system = _descriptor() if with_system else None
    sampler = _sampler(system=system)
    sampler.reset(n_walkers=1)
    state = sampler.mcmc_state_dict()

    if not with_system:
        # The default path retains the historical state shape, so checkpoints
        # written before descriptor binding remain loadable without migration.
        assert set(state) == _LEGACY_STATE_KEYS
        checkpoint_state = {key: state[key] for key in _LEGACY_STATE_KEYS}
    else:
        checkpoint_state = state

    stream = BytesIO()
    torch.save(checkpoint_state, stream)
    stream.seek(0)
    restored_state = torch.load(stream, weights_only=False)

    restored = _sampler()
    restored.load_mcmc_state_dict(restored_state)
    assert restored.walkers is not None
    assert (restored.walkers.make_batch().system is not None) is with_system
    if with_system:
        assert isinstance(restored.system, SystemDescriptor)
        assert restored.system.system_id() == system.system_id()

    continued = restored.sample(_ConstantWavefunction(), restored.walkers, n_steps=1)
    assert (continued.make_batch().system is not None) is with_system


def test_sampler_rejects_mismatched_configured_system_on_restore() -> None:
    source = _sampler(system=_descriptor(omega=0.5))
    source.reset(n_walkers=1)
    stream = BytesIO()
    torch.save(source.mcmc_state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=False)

    mismatched = _sampler(system=_descriptor(omega=0.7))
    with pytest.raises(ValueError, match="does not match the restored checkpoint's system"):
        mismatched.load_mcmc_state_dict(state)

