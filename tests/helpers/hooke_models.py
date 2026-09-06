"""Tiny real-model builders for Hooke pair smoke tests.

Single source of truth: everything is instantiated from the smoke training
fixture ``tests/integration/artifacts/hooke/pair_train.yaml`` (a copy of the experiments
config), so unit tests exercise the exact model/sampler the integration run uses.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch
from tpen.nn import TPENWaveFunction
from tpen.sampling.metropolis import MetropolisSampler

PAIR_TRAIN_CONFIG = Path(__file__).resolve().parents[1] / "integration" / "artifacts" / "hooke" / "pair_train.yaml"

# The order-1-OUTPUT path weights of the one-layer pair fixture: 16 of its 39
# trainable parameters. Order-1 features reach ``logabs`` ONLY through the
# odd-electron Pfaffian padding block
# (`tpen.nn.readout.pfaffian._odd_padding_block`), so at an EVEN electron count
# that whole subtree dead-ends at the readout and autograd never reaches it.
#
# The set is PARITY- and DEPTH-dependent, not a property of "two electrons":
# an odd count connects it, and a second layer would reconnect layer 0's copy.
# Any reasoning keyed to ``n == 2`` is wrong for that reason.
#
# Spelled out as a literal and owned by the module that owns the fixture.
# Deriving it from the model would re-apply the same reachability reasoning the
# score seam applies, and would then agree with itself for any value.
INACTIVE_PAIR_PARAMETERS = frozenset(
    [f"stack.layers.0.mixing.weights.g{index}" for index in range(15)]
    + ["stack.layers.0.path_aggregation.weights.o1"]
)


def _config() -> OmegaConf:
    return OmegaConf.load(PAIR_TRAIN_CONFIG)


def build_tiny_spenn() -> TPENWaveFunction:
    """Instantiate the tiny `TPENWaveFunction` from the smoke fixture config."""

    cfg = _config()
    model = instantiate(cfg.model)
    dtype = getattr(torch, str(cfg.runtime.dtype))
    device = torch.device(str(cfg.runtime.device))
    return model.to(device=device, dtype=dtype)


def build_tiny_sampler() -> MetropolisSampler:
    """Instantiate the fixed-spin Metropolis sampler from the smoke fixture config."""

    return instantiate(_config().sampler)


def build_tiny_hamiltonian_terms() -> dict:
    """Instantiate the named Hooke Hamiltonian terms from the smoke fixture config."""

    return dict(instantiate(_config().hamiltonian_terms))


def tiny_pair_batch(n_walkers: int = 4) -> ElectronBatch:
    """Return a tiny 2-electron batch with fixed (up, down) spins."""

    generator = torch.Generator().manual_seed(0)
    positions = torch.randn(n_walkers, 2, 3, generator=generator, dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0]] * n_walkers, dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)
