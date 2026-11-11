"""The two routes to the VMC energy gradient must agree, with clipping off.

`tpen.training.score_geometry` documents that ``geometry.design.T @ epsilon``
reproduces the ordinary VMC energy gradient, and `ScoreConventions` says its
default ``energy_gradient_scale = 2.0`` "matches
:func:`~tpen.training.vmc.compute_vmc_objective`". Both defaults are 2.0.

That agreement was DOCUMENTED IN TWO PLACES AND TESTED IN NEITHER. The existing
suites compare score against score -- `test_score_geometry.py` against a dense
oracle, `test_sr_update.py` SR against minSR -- so nothing compared the two
ROUTES: forming a design matrix and contracting it, versus building one scalar
loss and calling backward once. They agree only if the centering, the
``1/sqrt(count)`` normalization and the scale convention are all right, which is
exactly the set of things that can silently drift apart.

The scores here are ANALYTIC, not autograd-derived, so the two sides of every
comparison come from independent computations rather than from one differentiator
used twice.
"""

from __future__ import annotations

import pytest
import torch

from tpen.data.batch import (
    MaterializedParameterLogScores,
    ParameterLayout,
    ParameterSlot,
)
from tpen.training.score_geometry import (
    ScoreConventions,
    build_energy_residual,
    build_score_geometry,
    unflatten_to_layout,
)
from tpen.training.vmc import compute_vmc_objective

TOLERANCE = 1.0e-12
N_SAMPLES = 64


class _QuadraticLogAmplitude(torch.nn.Module):
    r"""A tiny model whose parameter scores are known in closed form.

    ``log|psi|(x) = alpha . x + beta . f(x)`` with
    ``f(x) = (x_0^2, x_1^2, x_0 x_1)``. Linear in the parameters, so

        d log|psi| / d alpha = x        and        d log|psi| / d beta = f(x)

    exactly. That is what lets the score route be built WITHOUT autograd: if
    both routes were differentiated by the same engine, a convention error in
    the geometry could cancel against the same error in the loss and the test
    would agree for the wrong reason.

    Linear-in-parameters is not a weakening. The parity claim is about the
    centering and normalization conventions relating scores to the gradient,
    and those are independent of how the scores were produced.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor([0.3, -0.7], dtype=torch.float64))
        self.beta = torch.nn.Parameter(torch.tensor([0.45, 0.1, -0.2], dtype=torch.float64))

    @staticmethod
    def features(positions: torch.Tensor) -> torch.Tensor:
        """Return ``f(x)``, which is also ``d log|psi| / d beta``."""

        return torch.stack(
            [
                positions[:, 0].square(),
                positions[:, 1].square(),
                positions[:, 0] * positions[:, 1],
            ],
            dim=-1,
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        linear = (positions * self.alpha).sum(dim=-1)
        quadratic = (self.features(positions) * self.beta).sum(dim=-1)
        return linear + quadratic


def _layout() -> ParameterLayout:
    return ParameterLayout(
        slots=(
            ParameterSlot(ordinal=0, shape=(2,), numel=2, dtype=torch.float64),
            ParameterSlot(ordinal=1, shape=(3,), numel=3, dtype=torch.float64),
        )
    )


def _fixture(seed: int = 20260906):
    """Return positions and local energies, independent of any model."""

    generator = torch.Generator().manual_seed(seed)
    positions = torch.randn(N_SAMPLES, 2, dtype=torch.float64, generator=generator)
    local_energy = torch.randn(N_SAMPLES, dtype=torch.float64, generator=generator) - 2.9
    return positions, local_energy


def _analytic_scores(positions: torch.Tensor) -> MaterializedParameterLogScores:
    """Build raw per-sample score blocks in closed form."""

    return MaterializedParameterLogScores(
        layout=_layout(),
        blocks=(positions.clone(), _QuadraticLogAmplitude.features(positions)),
    )


def _score_route(
    positions: torch.Tensor,
    local_energy: torch.Tensor,
    *,
    conventions: ScoreConventions | None = None,
) -> tuple[torch.Tensor, ...]:
    """Gradient via the design matrix: ``A^T epsilon``, unflattened to layout."""

    geometry = build_score_geometry(
        _analytic_scores(positions), sample_shape=(N_SAMPLES,), conventions=conventions
    )
    residual = build_energy_residual(local_energy, geometry=geometry)
    return unflatten_to_layout(geometry.design.T @ residual, layout=_layout())


def _autograd_route(
    positions: torch.Tensor,
    local_energy: torch.Tensor,
    *,
    clip_norm: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Gradient via one scalar loss and one backward pass."""

    model = _QuadraticLogAmplitude()
    result = compute_vmc_objective(model(positions), local_energy)
    result.loss.backward()
    if clip_norm is not None:
        # Mirrors what `LegacyAutogradUpdate` does between backward and step
        # when `gradient_clip_norm` is set: an in-place rescale of the
        # already-computed gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    return (model.alpha.grad.detach().clone(), model.beta.grad.detach().clone())


class TestTheTwoRoutesAgree:
    def test_score_contraction_equals_the_autograd_gradient(self) -> None:
        """The parity criterion, with clipping disabled."""

        positions, local_energy = _fixture()
        for expected, actual in zip(
            _autograd_route(positions, local_energy),
            _score_route(positions, local_energy),
        ):
            torch.testing.assert_close(actual, expected, rtol=TOLERANCE, atol=TOLERANCE)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_agreement_is_not_an_accident_of_one_draw(self, seed: int) -> None:
        """Independent fixtures, so agreement is not a property of one sample."""

        positions, local_energy = _fixture(seed=seed)
        for expected, actual in zip(
            _autograd_route(positions, local_energy),
            _score_route(positions, local_energy),
        ):
            torch.testing.assert_close(actual, expected, rtol=TOLERANCE, atol=TOLERANCE)


class TestTheParityCanFail:
    """Controls. Without these, agreement would be unfalsifiable."""

    def test_a_mismatched_scale_convention_breaks_parity_by_exactly_that_factor(
        self,
    ) -> None:
        """The two 2.0 defaults are COUPLED, and nothing else guards the coupling.

        `ScoreConventions.energy_gradient_scale` and
        `compute_vmc_objective`'s `scale_factor` are separately defaulted 2.0 in
        separate modules, and the docstring of the first asserts it matches the
        second. Changing one alone must break parity -- by exactly the ratio, so
        this also shows the disagreement is the scale and not something else.
        """

        positions, local_energy = _fixture()
        halved = _score_route(
            positions, local_energy, conventions=ScoreConventions(energy_gradient_scale=1.0)
        )
        for reference, scaled in zip(_score_route(positions, local_energy), halved):
            torch.testing.assert_close(scaled * 2.0, reference, rtol=TOLERANCE, atol=TOLERANCE)

        for expected, actual in zip(_autograd_route(positions, local_energy), halved):
            assert not torch.allclose(actual, expected, rtol=1e-6, atol=1e-6), (
                "halving the score scale must break parity; if it does not, the "
                "test is not sensitive to the convention it claims to check"
            )

    def test_clipping_breaks_parity_which_is_why_the_criterion_disables_it(self) -> None:
        """Explains the 'with clipping disabled' qualifier rather than assuming it.

        Gradient clipping rescales the autograd gradient AFTER it is computed
        and has no counterpart on the score route, so the two cannot agree when
        it is active. That is a property of clipping, not a defect -- but it
        means a parity check run with clipping on would fail for a reason that
        says nothing about the conventions under test.

        The clip norm is chosen well below the gradient's own norm so it
        certainly binds; a clip that never binds would leave this test green
        while demonstrating nothing.
        """

        positions, local_energy = _fixture()
        unclipped = _autograd_route(positions, local_energy)
        norm = torch.sqrt(sum(block.square().sum() for block in unclipped))
        assert norm > 0.0

        clipped = _autograd_route(positions, local_energy, clip_norm=float(norm) / 10.0)
        assert any(
            not torch.allclose(after, before, rtol=1e-6, atol=1e-6)
            for before, after in zip(unclipped, clipped)
        ), "the chosen clip norm did not bind, so this control demonstrates nothing"

        for expected, actual in zip(clipped, _score_route(positions, local_energy)):
            assert not torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


class TestTheScoresThemselvesAreRight:
    def test_analytic_scores_match_autograd_per_sample(self) -> None:
        """Pin the closed form, so a parity failure localises.

        The parity tests above use analytic scores precisely so the two routes
        do not share a differentiator. That independence is only worth having if
        the closed form is correct, so it is checked ONCE, here, against
        per-sample autograd -- and nowhere else, so the parity tests keep their
        independence.
        """

        positions, _ = _fixture()
        model = _QuadraticLogAmplitude()
        logabs = model(positions)

        alpha_rows = []
        beta_rows = []
        for index in range(N_SAMPLES):
            alpha_grad, beta_grad = torch.autograd.grad(
                logabs[index], (model.alpha, model.beta), retain_graph=True
            )
            alpha_rows.append(alpha_grad)
            beta_rows.append(beta_grad)

        scores = _analytic_scores(positions)
        torch.testing.assert_close(
            scores.blocks[0], torch.stack(alpha_rows), rtol=TOLERANCE, atol=TOLERANCE
        )
        torch.testing.assert_close(
            scores.blocks[1], torch.stack(beta_rows), rtol=TOLERANCE, atol=TOLERANCE
        )
