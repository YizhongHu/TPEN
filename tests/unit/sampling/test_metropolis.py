"""Unit tests for Metropolis sampling helpers."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from tpen.data.permutation import Permutation
from tpen.sampling import MALASampler
from tpen.sampling.metropolis import MetropolisSampler
from tpen.sampling.moves import GaussianMove


class ShiftMove(nn.Module):
    def propose(
        self,
        walkers: Walkers,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del generator
        positions = walkers.positions + 1.0
        log_q_ratio = torch.zeros(walkers.batch_size, device=walkers.device, dtype=walkers.dtype)
        return positions, log_q_ratio


class NoGradLinearModel(nn.Module):
    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        assert not torch.is_grad_enabled()
        logabs = batch.positions.sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class QuadraticLogAbsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.saw_position_grad = False

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        if torch.is_grad_enabled() and batch.positions.requires_grad:
            self.saw_position_grad = True
        logabs = -0.5 * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class ShiftedQuadraticLogAbsModel(QuadraticLogAbsModel):
    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        output = super().forward(batch)
        return WavefunctionOutput(logabs=output.logabs + self.shift, sign=output.sign)


class CountingLinearModel(NoGradLinearModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        self.calls += 1
        return super().forward(batch)


def test_metropolis_sampler_uses_logabs_ratio_and_caches_values() -> None:
    sampler = MetropolisSampler(move=ShiftMove(), n_walkers=3, n_electrons=2, spatial_dim=1, dtype=torch.float64)
    walkers = sampler.initialize(n_walkers=3, device="cpu")
    spins = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    walkers = Walkers(positions=torch.zeros_like(walkers.positions), spins=spins, aux=walkers.aux)

    stepped = sampler.step(NoGradLinearModel(), walkers)

    assert torch.equal(stepped.positions, torch.ones_like(stepped.positions))
    assert torch.equal(stepped.spins, spins)
    assert torch.equal(stepped.logabs, torch.full((3,), 2.0, dtype=torch.float64))
    assert torch.equal(stepped.sign, torch.ones(3, dtype=torch.float64))
    assert torch.equal(stepped.aux["accepted"], torch.ones(3, dtype=torch.bool))
    assert sampler.acceptance_rate == 1.0


def test_metropolis_initialize_builds_spins_from_partition() -> None:
    sampler = MetropolisSampler(n_walkers=5, n_electrons=2, spatial_dim=3, n_up=1, n_down=1, dtype=torch.float64)

    walkers = sampler.initialize(device="cpu")

    assert walkers.positions.shape == (5, 2, 3)
    assert walkers.spins is not None
    assert torch.equal(walkers.spins, torch.tensor([[1.0, -1.0]], dtype=torch.float64).expand(5, -1))


def test_metropolis_initialize_without_partition_has_no_spins() -> None:
    sampler = MetropolisSampler(n_walkers=4, n_electrons=2, spatial_dim=2, dtype=torch.float64)

    walkers = sampler.initialize(device="cpu")

    assert walkers.positions.shape == (4, 2, 2)
    assert walkers.spins is None


def test_sampler_propagates_fixed_nuclear_context_through_batches_and_restore() -> None:
    nuclear_positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
    sampler = MetropolisSampler(
        n_walkers=3,
        n_electrons=2,
        spatial_dim=3,
        seed=7,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        dtype=torch.float64,
    )

    walkers = sampler.reset(device="cpu")
    batch = walkers.make_batch()
    assert torch.equal(batch.nuclear_positions, nuclear_positions)
    assert torch.equal(batch.nuclear_charges, nuclear_charges)
    stepped = sampler.step(NoGradLinearModel(), walkers)
    assert torch.equal(stepped.make_batch().nuclear_positions, nuclear_positions)
    assert torch.equal(stepped.make_batch().nuclear_charges, nuclear_charges)
    permuted = batch.permute(Permutation((1, 0)))
    assert torch.equal(permuted.nuclear_positions, nuclear_positions)
    assert torch.equal(permuted.nuclear_charges, nuclear_charges)

    resumed = MetropolisSampler(n_walkers=3, n_electrons=2, spatial_dim=3, seed=7, dtype=torch.float64)
    resumed.load_mcmc_state_dict(sampler.mcmc_state_dict(), device="cpu")
    restored = resumed.walkers
    assert restored is not None
    assert torch.equal(restored.make_batch().nuclear_positions, nuclear_positions)
    assert torch.equal(restored.make_batch().nuclear_charges, nuclear_charges)


def test_sampler_rejects_partial_or_malformed_nuclear_context() -> None:
    import pytest

    positions = torch.zeros(1, 3, dtype=torch.float64)
    charges = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="provided together"):
        MetropolisSampler(nuclear_positions=positions)
    with pytest.raises(ValueError, match="spatial_dim"):
        MetropolisSampler(nuclear_positions=positions, nuclear_charges=charges, spatial_dim=2)
    with pytest.raises(ValueError, match="n_nuclei"):
        MetropolisSampler(nuclear_positions=positions, nuclear_charges=torch.ones(2))


def test_gaussian_single_electron_move_preserves_shape_and_changes_one_electron_per_walker() -> None:
    torch.manual_seed(0)
    positions = torch.zeros(5, 3, 2, dtype=torch.float64)
    walkers = Walkers(positions=positions)
    move = GaussianMove(step_size=0.1, move_all=False)

    proposals, log_q_ratio = move.propose(walkers)
    changed = (proposals != positions).any(dim=-1)

    assert proposals.shape == positions.shape
    assert log_q_ratio.shape == (5,)
    assert torch.allclose(log_q_ratio, torch.zeros(5, dtype=torch.float64))
    assert torch.equal(changed.sum(dim=1), torch.ones(5, dtype=torch.int64))


def _tiny_sampler() -> MetropolisSampler:
    return MetropolisSampler(
        n_walkers=8,
        burn_in=3,
        n_steps=2,
        proposal_scale=0.3,
        seed=123,
        n_electrons=2,
        spatial_dim=1,
        n_up=1,
        n_down=1,
        dtype=torch.float64,
    )


def test_sampler_local_rng_does_not_mutate_global_state() -> None:
    before = torch.get_rng_state()
    sampler = _tiny_sampler()
    sampler.collect_samples(NoGradLinearModel())
    after = torch.get_rng_state()

    assert torch.equal(before, after)


def test_sampler_seed_is_reproducible_across_instances() -> None:
    model = NoGradLinearModel()
    first, _ = _tiny_sampler().collect_samples(model)
    second, _ = _tiny_sampler().collect_samples(model)

    assert torch.equal(first.positions, second.positions)


def test_collect_samples_burns_in_once_and_advances_chain() -> None:
    sampler = _tiny_sampler()
    model = NoGradLinearModel()

    first, _ = sampler.collect_samples(model)
    assert sampler.has_burned_in is True
    second, _ = sampler.collect_samples(model)

    # The chain advanced (no re-burn, no reset) so the second draw differs.
    assert not torch.equal(first.positions, second.positions)


def test_collect_samples_reset_restarts_the_chain() -> None:
    model = NoGradLinearModel()
    sampler = _tiny_sampler()

    sampler.collect_samples(model)
    reset_draw, _ = sampler.collect_samples(model, reset=True)
    fresh_draw, _ = _tiny_sampler().collect_samples(model)

    # reset=True re-seeds, so it matches a fresh sampler's first draw.
    assert torch.equal(reset_draw.positions, fresh_draw.positions)


def test_collect_samples_refreshes_retained_cache_before_changed_model_decision() -> None:
    stale = MetropolisSampler(n_walkers=8, burn_in=0, n_steps=1, seed=123, dtype=torch.float64)
    fresh = MetropolisSampler(n_walkers=8, burn_in=0, n_steps=1, seed=123, dtype=torch.float64)
    base = QuadraticLogAbsModel()
    shifted = ShiftedQuadraticLogAbsModel(-5.0)

    for sampler in (stale, fresh):
        walkers = sampler.reset(device="cpu")
        sampler._walkers = walkers.update_cache(base)
        sampler._has_burned_in = True
    expected, _ = fresh.collect_samples(shifted)
    actual, _ = stale.collect_samples(shifted)

    assert torch.equal(actual.positions, expected.positions)
    assert torch.equal(actual.aux["accepted"], expected.aux["accepted"])


def test_retained_cache_refresh_is_one_boundary_evaluation_not_per_step() -> None:
    sampler = MetropolisSampler(n_walkers=4, burn_in=0, n_steps=2, seed=123, dtype=torch.float64)
    model = CountingLinearModel()
    sampler.collect_samples(model, reset=True)
    assert model.calls == 2

    model.calls = 0
    sampler.collect_samples(model)
    assert model.calls == 3  # one retained-cache refresh plus two proposals


def test_sampler_state_dict_roundtrip_continues_same_chain() -> None:
    model = NoGradLinearModel()
    sampler = _tiny_sampler()
    sampler.collect_samples(model)
    state = sampler.mcmc_state_dict()

    resumed = _tiny_sampler()
    resumed.load_mcmc_state_dict(state)

    expected, _ = sampler.collect_samples(model)
    actual, _ = resumed.collect_samples(model)

    assert resumed.has_burned_in is True
    assert torch.equal(expected.positions, actual.positions)


def test_sampler_state_load_maps_checkpoint_device_to_runtime_device() -> None:
    model = NoGradLinearModel()
    sampler = _tiny_sampler()
    sampler.collect_samples(model)
    state = sampler.mcmc_state_dict()
    state["generator_device"] = "cuda"

    resumed = _tiny_sampler()
    resumed.load_mcmc_state_dict(state, device="cpu")
    actual, _ = resumed.collect_samples(model, device="cpu")

    assert resumed.has_burned_in is True
    assert actual.device == torch.device("cpu")


def test_initialize_rejects_mismatched_device() -> None:
    import pytest

    sampler = _tiny_sampler()
    with pytest.raises(ValueError, match="reset"):
        sampler.initialize(device="meta")


def test_step_rejects_mismatched_walker_device() -> None:
    import pytest

    sampler = _tiny_sampler()
    walkers = Walkers(positions=torch.empty(2, 2, 1, device="meta", dtype=torch.float64))

    with pytest.raises(ValueError, match="reset"):
        sampler.step(NoGradLinearModel(), walkers)


def _tiny_mala() -> MALASampler:
    return MALASampler(
        n_walkers=8,
        burn_in=3,
        n_steps=2,
        proposal_scale=0.2,
        seed=7,
        n_electrons=2,
        spatial_dim=1,
        dtype=torch.float64,
    )


def test_mala_sampler_local_rng_does_not_mutate_global_state() -> None:
    before = torch.get_rng_state()
    _tiny_mala().collect_samples(QuadraticLogAbsModel())
    after = torch.get_rng_state()

    assert torch.equal(before, after)


def test_mala_sampler_seed_is_reproducible_across_instances() -> None:
    model = QuadraticLogAbsModel()
    first, _ = _tiny_mala().collect_samples(model)
    second, _ = _tiny_mala().collect_samples(model)

    assert torch.equal(first.positions, second.positions)


def test_mala_sampler_initializes_configured_spin_partition() -> None:
    sampler = MALASampler(
        n_walkers=3,
        n_electrons=4,
        spatial_dim=2,
        n_up=1,
        n_down=3,
        dtype=torch.float64,
    )

    walkers = sampler.initialize()

    assert walkers.spins is not None
    expected = torch.tensor([[1.0, -1.0, -1.0, -1.0]], dtype=torch.float64).expand(3, -1)
    torch.testing.assert_close(walkers.spins, expected)


def test_mala_sampler_uses_logabs_gradients_and_caches_valid_walkers() -> None:
    model = QuadraticLogAbsModel()
    nuclear_positions = torch.tensor([[0.0]], dtype=torch.float64)
    nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
    sampler = MALASampler(
        proposal_scale=0.05,
        n_walkers=4,
        n_electrons=2,
        spatial_dim=1,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        dtype=torch.float64,
    )
    walkers = sampler.reset()

    stepped = sampler.step(model, walkers)

    assert model.saw_position_grad
    assert stepped.positions.shape == walkers.positions.shape
    assert stepped.logabs is not None and stepped.logabs.shape == (4,)
    assert stepped.sign is not None and stepped.sign.shape == (4,)
    assert stepped.aux["accepted"].shape == (4,)
    assert stepped.aux["log_accept_ratio"].shape == (4,)
    assert stepped.atomic_configuration is not None
    assert torch.equal(stepped.atomic_configuration.positions, nuclear_positions)
    assert torch.equal(stepped.atomic_configuration.charges, nuclear_charges)
    assert torch.isfinite(stepped.logabs).all()
    assert 0.0 <= sampler.acceptance_rate <= 1.0


def test_sampler_with_no_atomic_configuration_stays_legacy_hooke_none() -> None:
    sampler = MetropolisSampler(n_walkers=3, n_electrons=2, spatial_dim=2, dtype=torch.float64)

    walkers = sampler.reset(device="cpu")

    assert sampler.atomic_configuration is None
    assert sampler.nuclear_positions is None
    assert sampler.nuclear_charges is None
    assert walkers.atomic_configuration is None
    assert walkers.make_batch().nuclear_positions is None
    assert walkers.make_batch().nuclear_charges is None

    state = sampler.mcmc_state_dict()
    resumed = MetropolisSampler(n_walkers=3, n_electrons=2, spatial_dim=2, dtype=torch.float64)
    resumed.load_mcmc_state_dict(state, device="cpu")

    assert resumed.atomic_configuration is None


def test_load_mcmc_state_dict_adopts_restored_configuration_when_unset() -> None:
    nuclear_positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
    configured = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        dtype=torch.float64,
    )
    configured.reset(device="cpu")
    state = configured.mcmc_state_dict()

    unset = MetropolisSampler(n_walkers=2, n_electrons=2, spatial_dim=3, dtype=torch.float64)
    assert unset.atomic_configuration is None

    unset.load_mcmc_state_dict(state, device="cpu")

    assert unset.atomic_configuration is not None
    assert unset.atomic_configuration == configured.atomic_configuration
    assert torch.equal(unset.nuclear_positions, nuclear_positions)
    assert torch.equal(unset.nuclear_charges, nuclear_charges)
    # The restored walkers are normalized onto the single resolved reference.
    assert unset.walkers.atomic_configuration is unset.atomic_configuration


def test_load_mcmc_state_dict_serializes_atomic_configuration_before_any_reset() -> None:
    nuclear_positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
    configured = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        dtype=torch.float64,
    )

    # No reset()/initialize() call yet: no walkers exist on this sampler.
    assert configured.walkers is None
    state = configured.mcmc_state_dict()
    assert state["walkers"] is None
    assert state["atomic_configuration"] is not None

    unset = MetropolisSampler(n_walkers=2, n_electrons=2, spatial_dim=3, dtype=torch.float64)
    unset.load_mcmc_state_dict(state, device="cpu")

    assert unset.atomic_configuration is not None
    assert torch.equal(unset.nuclear_positions, nuclear_positions)
    assert torch.equal(unset.nuclear_charges, nuclear_charges)


def test_load_mcmc_state_dict_requires_exact_agreement_when_configured() -> None:
    positions_a = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    charges_a = torch.tensor([2.0], dtype=torch.float64)
    positions_b = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    charges_b = torch.tensor([3.0], dtype=torch.float64)

    source = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=positions_a,
        nuclear_charges=charges_a,
        dtype=torch.float64,
    )
    source.reset(device="cpu")
    state = source.mcmc_state_dict()

    mismatched = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=positions_b,
        nuclear_charges=charges_b,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="does not match"):
        mismatched.load_mcmc_state_dict(state, device="cpu")

    # Nearly-equal-but-not-exact must also be rejected: exact agreement, not tolerant compare.
    almost = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=positions_a + 1e-12,
        nuclear_charges=charges_a,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="does not match"):
        almost.load_mcmc_state_dict(state, device="cpu")

    # A checkpoint with no configuration must not disturb an already-configured sampler
    # (contract: "preserve configured atomic configuration on restore").
    unset_checkpoint_state = dict(state)
    unset_checkpoint_state["atomic_configuration"] = None
    unset_checkpoint_state["walkers"] = replace(state["walkers"].to(device="cpu"), atomic_configuration=None)
    configured_again = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=positions_a,
        nuclear_charges=charges_a,
        dtype=torch.float64,
    )
    configured_again.load_mcmc_state_dict(unset_checkpoint_state, device="cpu")
    assert configured_again.atomic_configuration == AtomicConfiguration(positions=positions_a, charges=charges_a)


def test_load_mcmc_state_dict_rejects_walkers_atomic_configuration_mismatching_canonical_entry() -> None:
    positions_a = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    charges_a = torch.tensor([2.0], dtype=torch.float64)
    positions_b = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    charges_b = torch.tensor([3.0], dtype=torch.float64)

    source = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=positions_a,
        nuclear_charges=charges_a,
        dtype=torch.float64,
    )
    source.reset(device="cpu")
    state = source.mcmc_state_dict()
    # Simulate a hand-built/malformed checkpoint where the walkers carry a
    # different configuration than the canonical top-level entry.
    divergent_configuration = AtomicConfiguration(positions=positions_b, charges=charges_b)
    state["walkers"] = replace(state["walkers"], atomic_configuration=divergent_configuration)

    resumed = MetropolisSampler(n_walkers=2, n_electrons=2, spatial_dim=3, dtype=torch.float64)
    with pytest.raises(ValueError, match="does not match its canonical"):
        resumed.load_mcmc_state_dict(state, device="cpu")


def test_reset_preserves_configured_atomic_configuration() -> None:
    nuclear_positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
    sampler = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        dtype=torch.float64,
    )
    before = sampler.atomic_configuration

    sampler.reset(device="cpu")
    sampler.reset(device="cpu")

    assert sampler.atomic_configuration is before
    assert sampler.walkers is not None
    assert sampler.walkers.atomic_configuration is before


def test_sampler_atomic_configuration_moves_with_mixed_device_restore() -> None:
    nuclear_positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    nuclear_charges = torch.tensor([2.0], dtype=torch.float64)
    source = MetropolisSampler(
        n_walkers=2,
        n_electrons=2,
        spatial_dim=3,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        dtype=torch.float64,
    )
    source.reset(device="cpu")
    state = source.mcmc_state_dict()
    state["generator_device"] = "cuda"

    resumed = MetropolisSampler(n_walkers=2, n_electrons=2, spatial_dim=3, dtype=torch.float64)
    resumed.load_mcmc_state_dict(state, device="cpu")

    assert resumed.atomic_configuration is not None
    assert resumed.atomic_configuration.device == torch.device("cpu")
    assert resumed.walkers is not None
    assert resumed.walkers.atomic_configuration.device == torch.device("cpu")
    assert torch.equal(resumed.nuclear_positions, nuclear_positions)


def test_walkers_atomic_configuration_to_moves_dtype_and_device() -> None:
    configuration = AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )
    walkers = Walkers(
        positions=torch.zeros(2, 2, 3, dtype=torch.float64),
        atomic_configuration=configuration,
    )

    moved = walkers.to(dtype=torch.float32)

    assert moved.atomic_configuration is not None
    assert moved.atomic_configuration.dtype == torch.float32
    assert moved.positions.dtype == torch.float32
    # The source configuration is immutable and unaffected by the move.
    assert configuration.dtype == torch.float64


def test_walkers_rejects_atomic_configuration_device_mismatch() -> None:
    configuration = AtomicConfiguration(
        positions=torch.zeros(1, 3, dtype=torch.float64),
        charges=torch.ones(1, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="positions device"):
        Walkers(
            positions=torch.zeros(2, 2, 3, dtype=torch.float64, device="meta"),
            atomic_configuration=configuration,
        )


def test_electron_batch_rejects_atomic_configuration_mismatching_unbatched_nuclear_tensors() -> None:
    configuration = AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        charges=torch.tensor([2.0], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="atomic_configuration must match"):
        ElectronBatch(
            positions=torch.zeros(2, 2, 3, dtype=torch.float64),
            nuclear_positions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
            nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
            atomic_configuration=configuration,
        )
