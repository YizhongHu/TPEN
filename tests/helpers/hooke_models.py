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
# "Order-1-OUTPUT" is EXACT, not a gloss. The fixture's mixing carries 29 paths
# and each declares an output order: g0..g14 are the fifteen at output order 1
# and are exactly the fifteen mixing names below, while g15..g28 are the
# FOURTEEN at output order 2 and stay LIVE at every parity. So roughly half the
# tensor-product mixing dies at an even count -- NOT all of it -- and the
# fifteen are identified by output order rather than by any name pattern.
#
# PARITY: measured 16 disconnected at n=2, 4 and 6, byte-identical each time,
# and 0 at n=3, 5 and 7. Any reasoning keyed to ``n == 2`` is wrong.
#
# DEPTH: a second layer DOES reconnect layer 0, because g15 and g16 carry
# order-1 input to order-2 output (m=2, m1=1, m2=1) and the readout consumes
# order-2 unconditionally at every parity. BUT THE COUNT IS INVARIANT AND
# MERELY RELOCATES: measured 16 disconnected at one, two and three layers, at
# ``stack.layers.0``, ``.1`` and ``.2`` respectively. ADDING LAYERS IS
# THEREFORE NOT A MITIGATION -- a deeper model still serves exactly sixteen
# silently-zeroed parameters, just at ``stack.layers.<last>``.
#
# Spelled out as a literal and owned by the module that owns the fixture.
# Deriving it from the model would re-apply the same reachability reasoning the
# score seam applies, and would then agree with itself for any value.
INACTIVE_PAIR_PARAMETERS = frozenset(
    [f"stack.layers.0.mixing.weights.g{index}" for index in range(15)]
    + ["stack.layers.0.path_aggregation.weights.o1"]
)


def pair_parameters_by_name(model: TPENWaveFunction) -> dict[str, torch.nn.Parameter]:
    """Return trainable parameters by name, refusing a name-shape mismatch loudly.

    `INACTIVE_PAIR_PARAMETERS` was measured against a layer that mixes with
    `EquivariantMixing` DIRECTLY. A `CompositeMixing` model spells the same
    weights ``stack.layers.0.mixing.producers.0.weights.gN``, so a bare lookup
    raises ``KeyError('stack.layers.0.mixing.weights.g0')`` and names nothing
    that would let a reader work out why.

    The miss is also PARTIAL, which is harder to diagnose than a clean total
    one: ``path_aggregation.weights.o1`` exists under BOTH constructions, so 15
    of the 16 names miss and one hits.

    Parameters
    ----------
    model : TPENWaveFunction
        Model whose trainable parameter names are checked against the literal.

    Returns
    -------
    dict of str to torch.nn.Parameter
        Every named parameter, once the literal is known to apply.

    Raises
    ------
    KeyError
        If any name in `INACTIVE_PAIR_PARAMETERS` is absent, with the count
        missing, the first missing name, the likely cause, and the mixing
        parameter names actually present.
    """

    named = dict(model.named_parameters())
    missing = tuple(sorted(name for name in INACTIVE_PAIR_PARAMETERS if name not in named))
    if missing:
        present = tuple(sorted(name for name in named if ".mixing." in name))
        raise KeyError(
            f"INACTIVE_PAIR_PARAMETERS does not describe this model: {len(missing)} of "
            f"{len(INACTIVE_PAIR_PARAMETERS)} names absent, first {missing[0]!r}. "
            "The literal was measured against a layer using EquivariantMixing directly; "
            "a CompositeMixing model spells the same weights as "
            "'stack.layers.0.mixing.producers.0.weights.gN'. "
            f"Mixing parameters actually present: {present}"
        )
    return named


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
