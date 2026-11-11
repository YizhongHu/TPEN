"""Behavioral hydrogen-sampler controls for ``MetropolisSampler``.

Moved here from ``experiments/atomistic/h-v2/test_oracle_vs_tpen.py`` and
``test_stub_oracle_contract.py`` (contract amendment 2026-11-06): the
experiment boundary restricts experiment code/tests to configuration,
``run_from_config``, or the ``tpen`` command, and these checks reach
``MetropolisSampler``'s private hooks directly. The sampler is built inline
from the frozen H-v2 sampler literals (``tests/helpers/hydrogen_contract.SAMPLER_SETTINGS``)
rather than via the Hydra config.

Per the amendment's coverage hand-off: the Gamma(3, rate 2) mixing check below
keeps the frozen numeric sampler settings as literals rather than reading them
from configuration. A configuration-bound, survival-curve-strengthened
successor returns in the sampler tail qualification layer directly above
pull request 592, once the exact hydrogen module exists to carry it.
"""

from __future__ import annotations

import torch

from tpen.data.batch import Walkers
from tpen.sampling.metropolis import MetropolisSampler

from tests.helpers.hydrogen_contract import (
    ExactHydrogenState,
    RandomSmoothWaveFunction,
    SAMPLER_SETTINGS,
    gamma3_cdf,
    hydrogen_atoms,
)


def _hydrogen_sampler(**overrides) -> MetropolisSampler:
    atoms = hydrogen_atoms()
    settings = dict(SAMPLER_SETTINGS)
    settings.update(overrides)
    return MetropolisSampler(
        n_electrons=1,
        spatial_dim=3,
        n_up=1,
        n_down=0,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
        dtype=torch.float64,
        **settings,
    )


def test_sampler_accept_exponent_matches_independent_rederivation() -> None:
    """Behaviorally pin the sampler's |psi|^2 (not |psi|) acceptance exponent."""

    model = RandomSmoothWaveFunction(seed=7)
    sampler = _hydrogen_sampler()
    walkers = sampler.reset()

    saved_state = sampler._generator.get_state().clone()
    current_logabs, _ = sampler._evaluate(model, walkers)
    proposals, log_q_ratio = sampler._propose(model, walkers)
    proposal_walkers = Walkers(
        positions=proposals,
        spins=walkers.spins,
        atomic_configuration=walkers.atomic_configuration,
        aux=dict(walkers.aux),
    )
    proposed_logabs, _ = sampler._evaluate(model, proposal_walkers)
    expected_log_accept_ratio = torch.nan_to_num(
        2.0 * (proposed_logabs - current_logabs) + log_q_ratio, nan=-torch.inf
    )

    sampler._generator.set_state(saved_state)
    stepped = sampler.step(model, walkers)
    assert torch.equal(stepped.aux["log_accept_ratio"], expected_log_accept_ratio)


def test_sampler_accept_decision_matches_independent_rederivation() -> None:
    """Pin the accept DECISION, resulting positions, and cached logabs -- not only the ratio."""

    model = RandomSmoothWaveFunction(seed=7)
    sampler = _hydrogen_sampler()
    walkers = sampler.reset()

    saved_state = sampler._generator.get_state().clone()
    current_logabs, _ = sampler._evaluate(model, walkers)
    proposals, log_q_ratio = sampler._propose(model, walkers)
    proposal_walkers = Walkers(
        positions=proposals,
        spins=walkers.spins,
        atomic_configuration=walkers.atomic_configuration,
        aux=dict(walkers.aux),
    )
    proposed_logabs, _ = sampler._evaluate(model, proposal_walkers)
    expected_log_accept_ratio = torch.nan_to_num(
        2.0 * (proposed_logabs - current_logabs) + log_q_ratio, nan=-torch.inf
    )
    uniforms = torch.rand(
        expected_log_accept_ratio.shape, dtype=expected_log_accept_ratio.dtype, generator=sampler._generator
    )
    expected_accepted = torch.log(uniforms.clamp_min(1e-12)) < torch.clamp(expected_log_accept_ratio, max=0.0)

    sampler._generator.set_state(saved_state)
    stepped = sampler.step(model, walkers)
    assert torch.equal(stepped.aux["accepted"], expected_accepted)
    assert torch.equal(
        stepped.positions, torch.where(expected_accepted.view(-1, 1, 1), proposals, walkers.positions)
    )
    assert torch.equal(stepped.logabs, torch.where(expected_accepted, proposed_logabs, current_logabs))


def test_real_sampler_radii_match_gamma32_from_narrow_and_diffuse_starts(record_property) -> None:
    """The predeclared empirical-CDF absolute-difference band is 0.02."""

    model = ExactHydrogenState(1.0).double()
    radii_at = (0.5, 1.0, 1.5, 2.0, 3.0)
    for scale in (0.1, 5.0):
        sampler = _hydrogen_sampler(initial_scale=scale)
        walkers = sampler.reset()
        for _ in range(1000):
            walkers = sampler.step(model, walkers)
        retained = []
        accepted = []
        for _ in range(200):
            for _ in range(20):
                walkers = sampler.step(model, walkers)
                accepted.append(float(walkers.aux["accepted"].double().mean()))
            retained.append(walkers.positions.reshape(walkers.positions.shape[0], -1).norm(dim=-1))
        radii = torch.cat(retained)
        differences = [abs(float((radii <= r).double().mean()) - gamma3_cdf(r)) for r in radii_at]
        record_property(f"scale_{scale}_cdf_differences", differences)
        record_property(f"scale_{scale}_acceptance_rate", sum(accepted) / len(accepted))
        print(f"scale={scale} cdf_differences={differences} acceptance_rate={sum(accepted) / len(accepted)}")
        assert max(differences) < 0.02
