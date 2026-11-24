"""Tests for fixed-model observable trajectory collection."""

import pytest
import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch.electron_batch import ElectronBatch
from tpen.data.batch.walkers import Walkers
from tpen.data.batch.wavefunction_output import WavefunctionOutput
from tpen.physics.kinetic import autograd_laplacian
from tpen.sampling.stats import SamplerStats
from tpen.sampling.trajectory import (
    ModelDriftError,
    collect_observable_trajectory,
    collect_observable_trajectory_with_diagnostics,
    parameter_fingerprint,
)


def _stats(n_walkers: int, n_steps: int, seed: int = 7) -> SamplerStats:
    """Build the typed sampler record returned by the local fakes."""
    return SamplerStats(0.5, n_walkers, 0, n_steps, 1.0, seed=seed)


class FakeSampler:
    """Persistent sampler fake whose positions encode draw order."""

    def __init__(self, n_walkers: int = 2, n_steps: int = 4, misleading_n_steps: int | None = None) -> None:
        self.n_walkers = n_walkers
        self.returned_n_steps = n_steps
        self.n_steps = n_steps if misleading_n_steps is None else misleading_n_steps
        self.calls: list[dict[str, object]] = []
        self.draw_index = 0

    def collect_samples(
        self,
        model: torch.nn.Module,
        *,
        reset: bool,
        device: torch.device | str | None,
        emitter=None,
    ) -> tuple[Walkers, SamplerStats]:
        """Return real typed records with deterministic, varying walker values."""
        self.calls.append({"reset": reset, "device": device, "emitter": emitter})
        draw = self.draw_index
        self.draw_index += 1
        positions = torch.arange(self.n_walkers, dtype=torch.float32).reshape(self.n_walkers, 1, 1)
        positions = positions + draw * 10
        return Walkers(positions=positions), _stats(self.n_walkers, self.returned_n_steps, seed=draw)


class TinyModel(torch.nn.Module):
    """Small model exposing one parameter for fixed-model checks."""

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))


def _position_observable(walkers: Walkers) -> torch.Tensor:
    """Read the fake sampler's per-draw, per-walker values."""
    return walkers.positions[:, 0, 0]


def test_collection_preserves_draw_axis_and_order() -> None:
    """Keep each sampler result as one draw rather than flattening chains."""
    sampler = FakeSampler(n_walkers=2)

    trajectory, _ = collect_observable_trajectory(
        sampler,
        TinyModel(),
        _position_observable,
        observable_name="position",
        n_draws=3,
    )

    # float64 on purpose: ObservableTrajectory promotes every trajectory so the
    # lag sums downstream do not lose their tail to float32 rounding.
    expected = torch.tensor([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], dtype=torch.float64)
    assert trajectory.values.shape == (3, 2)
    torch.testing.assert_close(trajectory.values, expected)


def test_discard_draws_drops_only_leading_draws() -> None:
    """Discard warmup draws while retaining the exact tail and metadata."""
    sampler = FakeSampler(n_walkers=2)

    trajectory, _ = collect_observable_trajectory(
        sampler,
        TinyModel(),
        _position_observable,
        observable_name="position",
        n_draws=2,
        discard_draws=2,
    )

    assert len(sampler.calls) == 4
    torch.testing.assert_close(
        trajectory.values,
        torch.tensor([[20.0, 21.0], [30.0, 31.0]], dtype=torch.float64),
    )
    assert trajectory.burn_in_draws == 2


def test_draw_stride_comes_from_returned_sampler_stats() -> None:
    """Use typed diagnostics, even when sampler duck-typed metadata lies."""
    sampler = FakeSampler(n_steps=9, misleading_n_steps=99)

    trajectory, _ = collect_observable_trajectory(
        sampler,
        TinyModel(),
        _position_observable,
        observable_name="position",
        n_draws=1,
    )

    assert trajectory.draw_stride == 9


def test_reset_is_forwarded_only_to_first_collection_call() -> None:
    """Avoid re-seeding later draws, which would destroy temporal order."""
    sampler = FakeSampler()

    collect_observable_trajectory(
        sampler,
        TinyModel(),
        _position_observable,
        observable_name="position",
        n_draws=3,
        reset=True,
        device="cpu",
    )

    assert [call["reset"] for call in sampler.calls] == [True, False, False]
    assert [call["device"] for call in sampler.calls] == ["cpu", "cpu", "cpu"]


def test_model_drift_is_rejected_by_default_but_can_be_disabled() -> None:
    """Reject autocorrelation over a moving target unless explicitly opted out."""
    model = TinyModel()

    def mutating_observable(walkers: Walkers) -> torch.Tensor:
        model.weight.data.add_(1.0)
        return _position_observable(walkers)

    with pytest.raises(ModelDriftError, match="moving target distribution"):
        collect_observable_trajectory(
            FakeSampler(), model, mutating_observable, observable_name="position", n_draws=2
        )

    completed, _ = collect_observable_trajectory(
        FakeSampler(), model, mutating_observable, observable_name="position", n_draws=2,
        verify_model_unchanged=False,
    )
    assert completed.n_draws == 2


@pytest.mark.parametrize("initial_training", [True, False])
def test_collection_freezes_parameters_but_leaves_autograd_enabled(initial_training: bool) -> None:
    """Freeze the model without disabling the autograd the energy depends on.

    This previously asserted ``torch.is_grad_enabled()`` was False, which
    codified a defect rather than a contract: the local energy's kinetic term
    differentiates twice with respect to positions, so a collector that disables
    autograd cannot serve the acceptance contract's first observable at all.
    Freezing is expressed on the parameters instead.
    """
    model = TinyModel()
    model.train(initial_training)
    seen: list[tuple[bool, bool, bool]] = []

    def observable(walkers: Walkers) -> torch.Tensor:
        any_parameter_trainable = any(p.requires_grad for p in model.parameters())
        seen.append((model.training, torch.is_grad_enabled(), any_parameter_trainable))
        return _position_observable(walkers)

    collect_observable_trajectory(FakeSampler(), model, observable, observable_name="position", n_draws=2)

    # eval mode, autograd live, no parameter able to accumulate a gradient.
    assert seen == [(False, True, False), (False, True, False)]
    assert model.training is initial_training
    # Parameter flags are restored, not forced on.
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("sampler", [object(), None])
def test_sampler_without_collect_samples_raises_type_error(sampler: object) -> None:
    """Require the public sampler collection contract."""
    with pytest.raises(TypeError, match="collect_samples"):
        collect_observable_trajectory(sampler, TinyModel(), _position_observable, observable_name="x", n_draws=1)


def test_non_tensor_observable_result_raises_type_error() -> None:
    """Do not silently coerce an untyped observable result."""
    with pytest.raises(TypeError, match="torch.Tensor"):
        collect_observable_trajectory(
            FakeSampler(), TinyModel(), lambda walkers: [1.0, 2.0], observable_name="x", n_draws=1
        )


def test_wrong_observable_length_raises_value_error() -> None:
    """Require one scalar value for every walker to preserve chain columns."""
    with pytest.raises(ValueError, match="one value per walker"):
        collect_observable_trajectory(
            FakeSampler(n_walkers=2), TinyModel(), lambda walkers: torch.ones(1), observable_name="x", n_draws=1
        )


@pytest.mark.parametrize(
    ("n_draws", "discard_draws", "match"),
    [(0, 0, "n_draws"), (1, -1, "discard_draws")],
)
def test_collection_rejects_invalid_draw_counts(n_draws: int, discard_draws: int, match: str) -> None:
    """Reject requests that cannot produce a valid trajectory."""
    with pytest.raises(ValueError, match=match):
        collect_observable_trajectory(
            FakeSampler(), TinyModel(), _position_observable, observable_name="x",
            n_draws=n_draws, discard_draws=discard_draws,
        )


def test_zero_sampler_steps_explains_identical_consecutive_draws() -> None:
    """Reject a nominal draw axis that has no temporal separation."""
    with pytest.raises(ValueError, match="consecutive draws would be identical"):
        collect_observable_trajectory(
            FakeSampler(n_steps=0), TinyModel(), _position_observable, observable_name="x", n_draws=1
        )


def test_parameter_fingerprint_is_stable_changes_with_values_and_ignores_registration_order() -> None:
    """Hash semantic names and values, not module insertion order."""
    first = torch.nn.Module()
    first.register_parameter("alpha", torch.nn.Parameter(torch.tensor(1.0)))
    first.register_parameter("beta", torch.nn.Parameter(torch.tensor(2.0)))
    second = torch.nn.Module()
    second.register_parameter("beta", torch.nn.Parameter(torch.tensor(2.0)))
    second.register_parameter("alpha", torch.nn.Parameter(torch.tensor(1.0)))

    assert parameter_fingerprint(first) == parameter_fingerprint(first)
    assert parameter_fingerprint(first) == parameter_fingerprint(second)
    first.alpha.data.add_(1.0)
    assert parameter_fingerprint(first) != parameter_fingerprint(second)


def test_returned_stats_are_from_final_draw() -> None:
    """Return the final typed diagnostics record alongside the trajectory."""
    sampler = FakeSampler(n_steps=3)

    _, stats = collect_observable_trajectory(
        sampler, TinyModel(), _position_observable, observable_name="x", n_draws=3
    )

    assert stats == _stats(2, 3, seed=2)


class AtomicDrawSampler:
    """Return prescribed one-electron states and per-draw acceptance rates."""

    def __init__(self) -> None:
        self.states = (0.10, 0.50, 0.20, 0.80)
        self.acceptance_rates = (0.1, 0.2, 0.4, 0.6)
        self.index = 0
        self.atoms = AtomicConfiguration(
            positions=torch.zeros((1, 1), dtype=torch.float64),
            charges=torch.ones(1, dtype=torch.float64),
        )

    def collect_samples(
        self,
        model: torch.nn.Module,
        *,
        reset: bool,
        device: torch.device | str | None,
        emitter=None,
    ) -> tuple[Walkers, SamplerStats]:
        radius = self.states[self.index]
        acceptance_rate = self.acceptance_rates[self.index]
        self.index += 1
        walkers = Walkers(
            positions=torch.tensor([[[radius]]], dtype=torch.float64),
            atomic_configuration=self.atoms,
        )
        return walkers, SamplerStats(
            acceptance_rate=acceptance_rate,
            n_walkers=1,
            burn_in=100,
            n_steps=20,
            proposal_scale=0.5,
            seed=9,
        )


def test_draw_diagnostics_separate_discarded_and_retained_minima() -> None:
    """Reduce exact retained states, never the terminal snapshot or discarded region."""

    _, final_stats, diagnostics = collect_observable_trajectory_with_diagnostics(
        AtomicDrawSampler(),
        TinyModel(),
        _position_observable,
        observable_name="radius",
        n_draws=3,
        discard_draws=1,
    )

    metrics = diagnostics.as_metrics()
    assert final_stats.acceptance_rate == pytest.approx(0.6)
    # The scalar keeps its established final-call meaning. The new draw series
    # and aggregate have their own identities.
    assert metrics["trajectory_retained_draw_acceptance_rate_mean"] == pytest.approx(0.4)
    assert diagnostics.to_dict()["retained_draw_acceptance_rate_series"] == [0.2, 0.4, 0.6]
    assert metrics["trajectory_retained_draw_minimum_electron_nucleus_radius"] == pytest.approx(0.2)
    assert metrics["trajectory_discarded_draw_minimum_electron_nucleus_radius"] == pytest.approx(0.1)
    # The terminal state is radius 0.8 and cannot stand in for the retained
    # minimum. The stride remains beside the result and limits its claim.
    assert diagnostics.retained_draws[-1].minimum_electron_nucleus_radius == pytest.approx(0.8)
    assert metrics["trajectory_draw_stride"] == 20
    assert metrics["trajectory_intermediate_sampler_steps_observed"] is False
    assert metrics["trajectory_retained_transition_count"] == 3 * 20
    assert metrics["trajectory_discarded_transition_count"] == 20


def test_draw_geometry_unavailable_is_omitted_instead_of_zero() -> None:
    """Walkers without typed nuclei cannot fabricate an electron-nucleus minimum."""

    _, _, diagnostics = collect_observable_trajectory_with_diagnostics(
        FakeSampler(n_walkers=2, n_steps=4),
        TinyModel(),
        _position_observable,
        observable_name="position",
        n_draws=2,
    )

    metrics = diagnostics.as_metrics()
    assert "trajectory_retained_draw_minimum_electron_nucleus_radius" not in metrics
    assert all(
        draw.minimum_electron_nucleus_radius is None for draw in diagnostics.retained_draws
    )


class WrongWalkerCountSampler(FakeSampler):
    """Return typed stats that disagree with the actual walker state."""

    def collect_samples(
        self,
        model: torch.nn.Module,
        *,
        reset: bool,
        device: torch.device | str | None,
        emitter=None,
    ) -> tuple[Walkers, SamplerStats]:
        walkers, stats = super().collect_samples(model, reset=reset, device=device, emitter=emitter)
        return walkers, _stats(walkers.batch_size + 1, stats.n_steps)


def test_draw_diagnostics_reject_sampler_stats_walker_count_mismatch() -> None:
    """Counts and acceptance weighting must describe the returned chains."""

    with pytest.raises(ValueError, match="walker count disagrees"):
        collect_observable_trajectory_with_diagnostics(
            WrongWalkerCountSampler(),
            TinyModel(),
            _position_observable,
            observable_name="position",
            n_draws=1,
        )


class GaussianWavefunction(torch.nn.Module):
    """Analytic ``log|psi| = -alpha/2 * sum(r^2)`` with an exact Laplacian.

    Real enough to exercise `tpen.physics.kinetic`: it is differentiated twice
    with respect to positions by the same autograd path the local energy uses.
    Its Laplacian of ``logabs`` is ``-alpha * n_electrons * spatial_dim``,
    independent of position, so the oracle is exact rather than statistical.
    """

    def __init__(self, alpha: float = 0.75) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        squared = (batch.positions.to(torch.float64) ** 2).sum(dim=(-2, -1))
        logabs = -0.5 * self.alpha * squared
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def test_collection_supports_an_observable_needing_position_derivatives() -> None:
    """Serve the local energy's autograd path, not just gradient-free values.

    This is the regression test for a defect that three independent clean
    verifications missed: the collector ran the observable under
    ``torch.no_grad()``, so every fixture that passed was a gradient-free toy.
    `tpen.physics.kinetic.autograd_laplacian` calls
    ``torch.autograd.grad(..., create_graph=True)``, which RAISES under
    ``no_grad``. Energy is the acceptance contract's first observable, so the
    contract's primary path was unreachable while the suite stayed green.
    """
    alpha, n_electrons, spatial_dim = 0.75, 1, 1
    model = GaussianWavefunction(alpha)

    def laplacian_observable(walkers: Walkers) -> torch.Tensor:
        batch = ElectronBatch(positions=walkers.positions.to(torch.float64))
        return autograd_laplacian(model, batch)

    trajectory, _ = collect_observable_trajectory(
        FakeSampler(n_walkers=2),
        model,
        laplacian_observable,
        observable_name="laplacian",
        n_draws=3,
    )

    # Exact and position-independent: every draw and walker sees -alpha*N*D.
    expected = torch.full((3, 2), -alpha * n_electrons * spatial_dim, dtype=torch.float64)
    torch.testing.assert_close(trajectory.values, expected)


def test_frozen_collection_leaves_no_parameter_gradient_behind() -> None:
    """Differentiate through the model without accumulating parameter grads.

    Freezing is what makes "fixed model" true while autograd stays live. If the
    collector merely enabled grad without clearing ``requires_grad``, a caller
    that later stepped an optimizer would apply gradients accumulated during
    evaluation -- silently training on its own diagnostic.
    """
    model = GaussianWavefunction()

    def laplacian_observable(walkers: Walkers) -> torch.Tensor:
        batch = ElectronBatch(positions=walkers.positions.to(torch.float64))
        return autograd_laplacian(model, batch)

    collect_observable_trajectory(
        FakeSampler(n_walkers=2), model, laplacian_observable, observable_name="laplacian", n_draws=2
    )

    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in model.parameters())
