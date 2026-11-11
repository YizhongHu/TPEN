"""Capability tests for the bounded two-coefficient log-Jastrow factor.

The authority specifies the TWO-ELECTRON expression. The module generalizes it
to a sum over unordered pairs, so the first test below pins the generalization
against an independently written two-electron formula rather than against the
module's own loop.
"""

import pytest
import torch

from tpen.data.batch import ElectronBatch
from tpen.nn.cusp import rational_pair_cusp
from tpen.nn.jastrow import BoundedTwoCoefficientJastrow

THETA_S = 0.37
THETA_T = -0.21
LENGTH = 1.0


def _q(squared: torch.Tensor | float, length: float = LENGTH) -> torch.Tensor:
    """Bounded distance, written out here rather than imported.

    Deliberately a SECOND implementation of ``q``. Importing
    `tpen.nn.basis.bounded_distance` would make the expected value and the
    subject share a source, and a test whose two sides come from one
    implementation cannot detect that implementation being wrong.
    """

    squared_tensor = torch.as_tensor(squared, dtype=torch.float64)
    return squared_tensor / (length * length + squared_tensor)


def _factor(trainable: bool = True) -> BoundedTwoCoefficientJastrow:
    return BoundedTwoCoefficientJastrow(
        mean_coefficient=THETA_S,
        difference_coefficient=THETA_T,
        length=LENGTH,
        trainable=trainable,
    )


def _batch(positions: list) -> ElectronBatch:
    return ElectronBatch(positions=torch.tensor(positions, dtype=torch.float64))


CENTRE = torch.tensor([0.4, -0.2, 0.6], dtype=torch.float64)

# Six axis directions. The spherical average is what the Kato condition
# constrains, and an axis sextet integrates exactly through second order, which
# is enough to kill the first-order anisotropy this is looking for.
DIRECTIONS = torch.tensor(
    [
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ],
    dtype=torch.float64,
)


def _radial_derivative_of(function, radius: float) -> float:
    """Return ``d/dr function(r)`` by autograd at a single separation.

    Autograd rather than a difference quotient: a fixed-step quotient has a
    noise floor around ``eps * |f| / step`` that grows as the signal shrinks,
    which is exactly the regime these tests probe.
    """

    r = torch.tensor(float(radius), dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(function(r).sum(), r)
    return float(grad.item())


def _averaged_radial_derivative(factor, radius: float) -> float:
    """Return the magnitude of the direction-averaged radial derivative of J.

    Two electrons are placed symmetrically about `CENTRE`, separated by
    ``radius`` along each direction in turn, and the derivative is taken with
    respect to that separation.
    """

    total = 0.0
    for direction in DIRECTIONS:
        def separated(r, direction=direction):
            offset = 0.5 * r * direction
            positions = torch.stack([CENTRE - offset, CENTRE + offset]).unsqueeze(0)
            return factor(ElectronBatch(positions=positions))

        total += _radial_derivative_of(separated, radius)
    return abs(total / len(DIRECTIONS))


class TestTheTwoElectronCaseMatchesTheAuthority:
    def test_pair_sum_reduces_to_the_authority_expression_at_n_equals_two(self) -> None:
        """The generalization must be exactly the authority's formula at n = 2."""

        r1 = torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64)
        r2 = torch.tensor([-0.4, 0.1, 0.9], dtype=torch.float64)
        batch = _batch([[r1.tolist(), r2.tolist()]])

        q1 = _q(r1.dot(r1))
        q2 = _q(r2.dot(r2))
        q12 = _q((r1 - r2).dot(r1 - r2))
        # J = theta_s q12 (q1 + q2)/2 + theta_t q12 (q1 - q2)^2, written out.
        expected = THETA_S * q12 * (q1 + q2) / 2.0 + THETA_T * q12 * (q1 - q2) ** 2

        torch.testing.assert_close(_factor()(batch), expected.reshape(1))

    def test_three_electrons_are_a_sum_over_unordered_pairs(self) -> None:
        """Each pair counted ONCE; a double count or a diagonal term shows here."""

        positions = [[0.3, -0.7, 0.2], [-0.4, 0.1, 0.9], [1.1, 0.5, -0.3]]
        batch = _batch([positions])
        vectors = [torch.tensor(p, dtype=torch.float64) for p in positions]
        singles = [_q(v.dot(v)) for v in vectors]

        expected = torch.zeros((), dtype=torch.float64)
        for i in range(3):
            for j in range(i + 1, 3):
                separation = vectors[i] - vectors[j]
                pair = _q(separation.dot(separation))
                expected = expected + pair * (
                    THETA_S * (singles[i] + singles[j]) / 2.0
                    + THETA_T * (singles[i] - singles[j]) ** 2
                )

        torch.testing.assert_close(_factor()(batch), expected.reshape(1))


class TestSymmetryAndSignStructure:
    def test_exchanging_two_electrons_leaves_the_factor_unchanged(self) -> None:
        """Both summands are symmetric in i and j, so J is permutation-invariant."""

        batch = _batch([[[0.3, -0.7, 0.2], [-0.4, 0.1, 0.9], [1.1, 0.5, -0.3]]])
        factor = _factor()
        # Plain index rather than a Permutation object: the property under
        # test is invariance of J, and routing it through another module would
        # make a failure ambiguous between the two.
        swapped = ElectronBatch(positions=batch.positions[:, [1, 0, 2], :])
        torch.testing.assert_close(factor(batch), factor(swapped))

    def test_zero_coefficients_make_the_factor_the_identity(self) -> None:
        """Both coefficients start at zero, so an untrained model is unchanged.

        ``J = 0`` means ``exp(J) = 1``. This is what makes the factor safe to
        add to an existing model without perturbing it before training.
        """

        factor = BoundedTwoCoefficientJastrow(length=LENGTH)
        batch = _batch([[[0.3, -0.7, 0.2], [-0.4, 0.1, 0.9]]])
        torch.testing.assert_close(factor(batch), torch.zeros(1, dtype=torch.float64))

    def test_the_factor_is_finite_so_it_can_never_introduce_a_node(self) -> None:
        """A finite J gives a strictly positive exp(J), which has no zeros.

        The factor therefore cannot move a node or flip a sign; the readout
        remains solely responsible for the sign structure. Checked over
        configurations that include coincidence and the nucleus, where an
        unbounded correlation factor would be most likely to diverge.
        """

        batch = _batch(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
                [[1e4, 0.0, 0.0], [-1e4, 0.0, 0.0]],
            ]
        )
        values = _factor()(batch)
        assert torch.isfinite(values).all()


class TestBehaviourAtCoincidence:
    def test_the_gradient_vanishes_exactly_where_the_electrons_coincide(self) -> None:
        """The Kato condition is on the SPHERICAL AVERAGE, and this is stronger.

        ``q(r) = r^2/(l^2 + r^2)`` has ``q(0) = 0`` and ``q'(0) = 0``, so every
        summand carries a factor that is flat at coalescence and the Cartesian
        gradient of J is EXACTLY zero there. A term with a zero gradient
        contributes nothing to the spherical average of the radial derivative,
        so the analytic electron-electron cusp is untouched.

        Note what is NOT claimed. The Kato condition constrains the spherical
        average of the derivative of the FULL log-amplitude; the stronger claim
        that every directional derivative of the wavefunction vanishes at
        coalescence is false and is not what this asserts. The scope here is
        this factor's own contribution.
        """

        positions = torch.tensor(
            [[[0.4, -0.2, 0.6], [0.4, -0.2, 0.6]]], dtype=torch.float64, requires_grad=True
        )
        _factor()(ElectronBatch(positions=positions)).sum().backward()

        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all(), "sqrt in the distance would NaN here"
        torch.testing.assert_close(positions.grad, torch.zeros_like(positions.grad))

    def test_the_spherical_average_of_the_radial_derivative_vanishes_linearly(self) -> None:
        """The Kato-shaped statement: the average vanishes IN THE LIMIT, as O(r).

        The claim is about a limit, not about a value at any finite separation.
        At separation ``r`` the averaged radial derivative of J is proportional
        to ``r`` -- it is SMALL, not zero -- because ``q(r) = r^2/(l^2 + r^2)``
        has ``q'(r) = 2r l^2/(l^2 + r^2)^2``. What distinguishes a factor that
        respects the cusp from one that alters it is therefore the RATE, and a
        ratio test states that scale-free.

        An earlier version of this test asserted an absolute floor of 1e-6 at
        ``r = 1e-4``. That was wrong arithmetic, not a wrong factor: the true
        value there is about ``0.27 r``, i.e. 2.7e-5, so the threshold demanded
        behaviour that only appears near ``r = 4e-6``. Recorded because the
        failure read exactly like a defect in the module.

        Uses autograd rather than finite differences. A difference quotient
        with a fixed step has a noise floor of roughly ``eps * |J| / step``,
        which grows as the signal shrinks and eventually swamps precisely the
        limit being measured.
        """

        radii = (1e-2, 1e-3, 1e-4)
        averages = [_averaged_radial_derivative(_factor(), radius) for radius in radii]

        # Each tenfold reduction in r must reduce the average about tenfold.
        for coarse, fine in zip(averages, averages[1:]):
            assert fine == pytest.approx(coarse / 10.0, rel=0.05), (
                f"averages {averages} do not fall linearly in r; a factor that "
                "altered the cusp would tend to a nonzero constant instead"
            )

        assert averages[-1] < averages[0] / 50.0

    def test_the_instrument_would_see_a_factor_that_did_alter_the_cusp(self) -> None:
        """Discriminating control: prove the ratio test above is not vacuous.

        The analytic electron-electron cusp term ``a r / (1 + b r)`` has radial
        derivative tending to ``a != 0`` at coalescence -- that nonzero limit IS
        the cusp. Measured with the SAME instrument, it does NOT fall with r.

        Without this, the test above would pass equally for an instrument that
        returned something proportional to r whatever it was handed.
        """

        coefficient = 0.5
        radii = (1e-2, 1e-3, 1e-4)
        values = [
            abs(
                _radial_derivative_of(
                    lambda r: rational_pair_cusp(r, coefficient, 1.0), radius
                )
            )
            for radius in radii
        ]

        for value in values:
            assert value == pytest.approx(coefficient, rel=0.05), (
                "a genuine cusp term must keep a nonzero radial derivative at "
                f"coalescence; got {values}"
            )
        # And it emphatically does not shrink the way J does.
        assert values[-1] > values[0] / 2.0


class TestDegenerateElectronCounts:
    @pytest.mark.parametrize("n_electrons", [0, 1])
    def test_fewer_than_two_electrons_gives_exactly_zero(self, n_electrons: int) -> None:
        """No unordered pair exists, so the sum is empty rather than undefined.

        A masked upper triangle keeps the shape static, so this returns zeros
        instead of failing on an empty index.
        """

        batch = ElectronBatch(positions=torch.zeros(3, n_electrons, 3, dtype=torch.float64))
        torch.testing.assert_close(_factor()(batch), torch.zeros(3, dtype=torch.float64))


class TestTrainability:
    def test_both_coefficients_receive_gradient_from_the_first_step(self) -> None:
        """Neither coefficient sits in a zero-gradient trap at its initial value.

        Both start at exactly zero, and each multiplies a term that is
        generically nonzero, so both derivatives are nonzero immediately. That
        is the property the electron-nucleus curvature law had to work around.
        """

        factor = BoundedTwoCoefficientJastrow(length=LENGTH)
        factor(_batch([[[0.3, -0.7, 0.2], [-0.4, 0.1, 0.9]]])).sum().backward()

        for name, parameter in (("theta_s", factor.theta_s), ("theta_t", factor.theta_t)):
            assert parameter.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().item() > 0.0, f"{name} sits at a zero-gradient point"

    def test_a_configuration_that_really_does_kill_the_gradient(self) -> None:
        """Discriminating control for the assertion above.

        ``grad > 0`` says nothing on its own unless some configuration makes it
        false; otherwise it would pass for any nonzero number the autograd
        engine happened to produce. Coincident electrons are that
        configuration: every ``q_ij`` is exactly zero, so both terms and both
        derivatives vanish and neither coefficient can learn from such a
        sample.

        Compared RELATIVELY against the healthy configuration rather than
        against an absolute floor. The degenerate value is zero up to
        round-off, and an exact-zero assertion on a computed quantity fires or
        not by coin flip.
        """

        healthy = BoundedTwoCoefficientJastrow(length=LENGTH)
        healthy(_batch([[[0.3, -0.7, 0.2], [-0.4, 0.1, 0.9]]])).sum().backward()

        degenerate = BoundedTwoCoefficientJastrow(length=LENGTH)
        point = [0.4, -0.2, 0.6]
        degenerate(_batch([[point, point]])).sum().backward()

        for name in ("theta_s", "theta_t"):
            live = getattr(healthy, name).grad.abs().item()
            dead = getattr(degenerate, name).grad.abs().item()
            assert live > 0.0
            assert dead < live * 1e-6, (
                f"{name}: coincident electrons should carry no gradient, got "
                f"{dead} against {live} at a generic configuration"
            )

    def test_trainable_contributes_state_and_frozen_contributes_none(self) -> None:
        """Frozen coordinates are non-persistent buffers, matching the cusp laws."""

        assert set(_factor(trainable=True).state_dict()) == {"theta_s", "theta_t"}
        assert set(_factor(trainable=False).state_dict()) == set()

    def test_a_frozen_factor_still_evaluates_to_the_same_value(self) -> None:
        """Freezing changes the gradient path, not the function."""

        batch = _batch([[[0.3, -0.7, 0.2], [-0.4, 0.1, 0.9]]])
        torch.testing.assert_close(_factor(trainable=True)(batch), _factor(trainable=False)(batch))


class TestConstructorValidation:
    @pytest.mark.parametrize("length", [0.0, -1.0])
    def test_a_non_positive_length_is_refused(self, length: float) -> None:
        with pytest.raises(ValueError, match="length must be positive"):
            BoundedTwoCoefficientJastrow(length=length)

    def test_diagnostics_report_both_coefficients(self) -> None:
        diagnostics = _factor().scalar_diagnostics()
        assert diagnostics["jastrow_mean_coefficient"] == pytest.approx(THETA_S)
        assert diagnostics["jastrow_difference_coefficient"] == pytest.approx(THETA_T)
