"""Capability tests for the tail-safe electron-nucleus cusp law.

The property under test is that the outer radial slope is negative for EVERY
nucleus by construction, not merely for the charge someone had in mind. The
predecessor law, `CurvatureElectronNucleusCuspLaw`, is correct for the charge
it was tuned at and unbounded elsewhere; that difference is what these tests
have to be able to see.
"""

import pytest
import torch

from tpen.nn.cusp import CurvatureElectronNucleusCuspLaw, TailSafeElectronNucleusCuspLaw

# Charges spanning hydrogen to iron. Deliberately NOT only helium: a
# helium-only law passes every single-charge test.
CHARGES = (1.0, 2.0, 3.0, 6.0, 26.0)


def _law(tail_slope: float = 1.99, curvature_range: float = 1.0, **kwargs):
    return TailSafeElectronNucleusCuspLaw(
        curvature_range=curvature_range, tail_slope=tail_slope, **kwargs
    )


class TestTheOuterTailIsSafeForEveryCharge:
    @pytest.mark.parametrize("charge", CHARGES)
    def test_outer_slope_equals_minus_kappa_independently_of_charge(self, charge: float) -> None:
        """Expectation comes from a source INDEPENDENT of the charge under test.

        This is the point of the parametrization. ``kappa`` is read off the law
        itself and does not depend on ``charge``, while the subject
        ``outer_tail_slope(charge)`` is computed through
        ``c = d (Z - kappa)`` and does. If the expected value were recomputed
        from the same charge fed to the subject, both sides would move together
        and the test could not fail -- it would confirm the formula against
        itself for any formula whatsoever.
        """

        law = _law()
        expected = -law.tail_slope_magnitude  # no `charge` on this line, on purpose
        actual = law.outer_tail_slope(torch.tensor(charge, dtype=torch.float64))
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("charge", CHARGES)
    def test_the_measured_far_field_slope_matches_the_analytic_one(self, charge: float) -> None:
        """A second, independent instrument: finite differences on ``value``.

        The test above trusts `outer_tail_slope`. This one ignores it entirely
        and measures the slope of the function itself far from the nucleus, so
        an error shared between `outer_tail_slope` and `curvature_coefficient`
        cannot hide in both.
        """

        law = _law()
        charges = torch.tensor(charge, dtype=torch.float64)
        far, step = 1.0e7, 1.0e3
        measured = (
            law.value(torch.tensor(far + step, dtype=torch.float64), charges)
            - law.value(torch.tensor(far, dtype=torch.float64), charges)
        ) / step
        torch.testing.assert_close(
            measured, -law.tail_slope_magnitude, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("charge", CHARGES)
    def test_the_slope_is_strictly_negative_so_the_tail_decays(self, charge: float) -> None:
        """The physical claim: normalizable, for every nucleus."""

        assert _law().outer_tail_slope(torch.tensor(charge, dtype=torch.float64)).item() < 0.0

    def test_the_predecessor_is_not_safe_at_the_same_coordinates(self) -> None:
        """The discriminating control: prove the property is not vacuous.

        Without this, every assertion above would pass equally for a law that
        was safe by accident. The old law with ``c/d`` above the charge has a
        POSITIVE outer slope -- a growing, non-normalizable tail -- which is
        exactly the state the new parametrization cannot represent.
        """

        unsafe = CurvatureElectronNucleusCuspLaw(curvature_coefficient=5.0, curvature_range=1.0)
        assert unsafe.outer_tail_slope(torch.tensor(2.0, dtype=torch.float64)).item() > 0.0

        with pytest.raises(ValueError, match="not tail-safe"):
            TailSafeElectronNucleusCuspLaw.from_curvature_coefficient(
                curvature_coefficient=5.0, curvature_range=1.0, charge=2.0
            )


class TestTheKatoSlopeIsUntouched:
    @pytest.mark.parametrize("charge", CHARGES)
    def test_the_origin_slope_is_exactly_minus_z(self, charge: float) -> None:
        """The curvature term still contributes only from second order."""

        charges = torch.tensor(charge, dtype=torch.float64)
        torch.testing.assert_close(_law().origin_radial_slope(charges), -charges)

    @pytest.mark.parametrize("charge", CHARGES)
    def test_the_measured_slope_at_the_origin_is_minus_z(self, charge: float) -> None:
        """Measured from ``value``, so the claim is about the function.

        ``origin_radial_slope`` returns ``-Z`` by construction and would keep
        returning it even if `value` drifted. This checks the two agree.
        """

        law = _law()
        charges = torch.tensor(charge, dtype=torch.float64)
        radius = torch.tensor(1.0e-7, dtype=torch.float64)
        measured = law.value(radius, charges) / radius
        torch.testing.assert_close(measured, -charges, rtol=1e-6, atol=1e-6)

    def test_the_value_is_zero_at_the_nucleus(self) -> None:
        zero = torch.tensor(0.0, dtype=torch.float64)
        torch.testing.assert_close(
            _law().value(zero, torch.tensor(2.0, dtype=torch.float64)), zero
        )

    def test_analytic_terms_agree_with_autograd(self) -> None:
        """The closed forms are the ones actually used; check them numerically."""

        law = _law()
        charges = torch.tensor(2.0, dtype=torch.float64)
        radius = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

        value, first, second, residual = law.analytic_terms(radius, charges)
        (grad,) = torch.autograd.grad(law.value(radius, charges), radius, create_graph=True)
        (curvature,) = torch.autograd.grad(grad, radius)

        torch.testing.assert_close(value, law.value(radius, charges))
        torch.testing.assert_close(first, grad)
        torch.testing.assert_close(second, curvature)
        # The residual is the already-cancelled (u' + Z)/r; never reconstructed
        # by subtraction, so it is checked against the definition instead.
        torch.testing.assert_close(residual, (first + charges) / radius)


class TestTheInversion:
    @pytest.mark.parametrize("charge", CHARGES)
    def test_round_trip_reproduces_the_requested_coefficient(self, charge: float) -> None:
        """``from_curvature_coefficient`` must be the exact inverse at that Z."""

        requested_c, requested_d = 0.01, 1.0
        law = TailSafeElectronNucleusCuspLaw.from_curvature_coefficient(
            curvature_coefficient=requested_c, curvature_range=requested_d, charge=charge
        )
        charges = torch.tensor(charge, dtype=torch.float64)
        torch.testing.assert_close(
            law.curvature_coefficient(charges),
            torch.tensor(requested_c, dtype=torch.float64),
            rtol=1e-9,
            atol=1e-12,
        )
        torch.testing.assert_close(
            law.curvature_range,
            torch.tensor(requested_d, dtype=torch.float64),
            rtol=1e-9,
            atol=1e-12,
        )

    def test_the_shipped_hi_control_level_inverts_to_the_configured_kappa(self) -> None:
        """The he-importance config's (c, d) = (.01, 1) at Z = 2 gives kappa = 1.99.

        Pins the migration arithmetic that the config comment asserts, so the
        two cannot drift apart silently.
        """

        law = TailSafeElectronNucleusCuspLaw.from_curvature_coefficient(
            curvature_coefficient=0.01, curvature_range=1.0, charge=2.0
        )
        assert law.tail_slope_magnitude.item() == pytest.approx(1.99, abs=1e-12)

    @pytest.mark.parametrize("coefficient", [2.0, 1.95, 100.0])
    def test_an_unrepresentable_request_raises_rather_than_clamping(
        self, coefficient: float
    ) -> None:
        """Silently returning a different law is how a bad migration goes unseen."""

        with pytest.raises(ValueError, match="not tail-safe"):
            TailSafeElectronNucleusCuspLaw.from_curvature_coefficient(
                curvature_coefficient=coefficient, curvature_range=1.0, charge=2.0
            )

    def test_a_non_positive_range_is_refused_by_the_classmethod(self) -> None:
        with pytest.raises(ValueError, match="curvature_range must be positive"):
            TailSafeElectronNucleusCuspLaw.from_curvature_coefficient(
                curvature_coefficient=0.01, curvature_range=0.0, charge=2.0
            )


class TestConstructorValidation:
    @pytest.mark.parametrize("tail_slope", [0.1, 0.05, 0.0, -1.0])
    def test_a_tail_slope_at_or_below_the_margin_is_refused(self, tail_slope: float) -> None:
        with pytest.raises(ValueError, match="tail_slope must exceed tail_slope_margin"):
            _law(tail_slope=tail_slope)

    def test_a_tail_slope_just_above_the_margin_is_accepted(self) -> None:
        """The over-restriction control: the boundary is exclusive, not a moat."""

        assert _law(tail_slope=0.100001).tail_slope_magnitude.item() > 0.1

    @pytest.mark.parametrize("curvature_range", [0.0, -1.0])
    def test_a_non_positive_range_is_refused(self, curvature_range: float) -> None:
        with pytest.raises(ValueError, match="curvature_range must be positive"):
            _law(curvature_range=curvature_range)

    def test_a_non_positive_margin_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tail_slope_margin must be positive"):
            _law(tail_slope_margin=0.0)


class TestTrainability:
    def test_both_coordinates_receive_gradient_at_the_configured_level(self) -> None:
        """Neither coordinate is stuck at the shipped initialization.

        The predecessor had a real trap here: at exactly ``c = 0`` the gradient
        with respect to ``d`` is identically zero. The analogous degeneracy sits
        at ``kappa = Z``, and the shipped level (kappa = 1.99, Z = 2) is off it.
        """

        law = _law(tail_slope=1.99)
        charges = torch.tensor(2.0, dtype=torch.float64)
        law.value(torch.tensor(0.8, dtype=torch.float64), charges).backward()

        for name in ("raw_range", "raw_kappa"):
            grad = getattr(law, name).grad
            assert grad is not None, f"{name} received no gradient"
            assert torch.isfinite(grad).all()
            assert grad.abs().item() > 0.0, f"{name} sits at a zero-gradient point"

    def test_kappa_receives_gradient_even_where_the_coefficient_vanishes(self) -> None:
        """Strictly better than the predecessor, and worth pinning.

        At ``kappa = Z`` the derived ``c`` is zero and the ``d`` gradient
        vanishes, mirroring the old ``c = 0`` trap. But ``kappa`` still moves,
        so the law recovers on its own rather than sitting frozen.
        """

        law = _law(tail_slope=2.0)
        charges = torch.tensor(2.0, dtype=torch.float64)
        law.value(torch.tensor(0.8, dtype=torch.float64), charges).backward()

        assert law.raw_kappa.grad is not None
        assert law.raw_kappa.grad.abs().item() > 0.0
        assert law.curvature_coefficient(charges).abs().item() == pytest.approx(0.0, abs=1e-12)

    def test_the_range_gradient_really_does_die_at_the_degenerate_point(self) -> None:
        """Discriminating control for the ``grad > 0`` assertions above.

        Those assertions are worth nothing unless some configuration makes them
        false, and this is it. ``c = d (Z - kappa)`` and its ``d``-derivative
        both carry a factor of ``(Z - kappa)``, so at ``kappa = Z`` the range
        receives no gradient at all -- the exact analogue of the predecessor's
        ``c = 0`` trap, which is why the shipped config sits at 1.99 rather
        than 2.

        Stated RELATIVELY against the healthy configuration. The degenerate
        value is zero only up to the softplus round-trip's round-off, so an
        exact-zero assertion would fire or not by coin flip.
        """

        charges = torch.tensor(2.0, dtype=torch.float64)
        radius = torch.tensor(0.8, dtype=torch.float64)

        healthy = _law(tail_slope=1.99)
        healthy.value(radius, charges).backward()

        degenerate = _law(tail_slope=2.0)  # kappa == Z
        degenerate.value(radius, charges).backward()

        live = healthy.raw_range.grad.abs().item()
        dead = degenerate.raw_range.grad.abs().item()
        assert live > 0.0
        assert dead < live * 1e-6, (
            f"raw_range should receive no gradient at kappa == Z, got {dead} "
            f"against {live} at kappa = 1.99"
        )

    def test_frozen_contributes_no_checkpoint_state(self) -> None:
        assert set(_law(trainable=False).state_dict()) == set()

    def test_freezing_does_not_change_the_function(self) -> None:
        charges = torch.tensor(2.0, dtype=torch.float64)
        radius = torch.tensor(0.8, dtype=torch.float64)
        torch.testing.assert_close(
            _law(trainable=True).value(radius, charges),
            _law(trainable=False).value(radius, charges),
        )


class TestMigrationIsRefusedNotReinterpreted:
    def test_the_two_laws_have_disjoint_parameter_names(self) -> None:
        """This disjointness IS the migration guarantee, so it is pinned.

        If the key sets ever overlapped, an old checkpoint could load into the
        new law and be silently reinterpreted in coordinates that mean
        something different, producing plausible numbers rather than an error.
        """

        old = set(CurvatureElectronNucleusCuspLaw(trainable=True).state_dict())
        new = set(_law(trainable=True).state_dict())
        assert old == {"raw_curvature_coefficient", "raw_curvature_range"}
        assert new == {"raw_range", "raw_kappa"}
        assert not (old & new)

    @pytest.mark.parametrize("strict", [True, False])
    def test_loading_old_coordinates_raises_and_names_the_migration(self, strict: bool) -> None:
        """Refused under strict=False too, where the default is silent success.

        ``strict=False`` would ordinarily ignore the unrecognised keys and
        leave the law at its freshly initialized values -- a wavefunction
        restored to the wrong parameters with nothing raising. That is the
        ambiguous migration this slice exists to prevent, so the refusal must
        not depend on the caller passing strict=True.
        """

        legacy = CurvatureElectronNucleusCuspLaw(
            curvature_coefficient=0.01, curvature_range=1.0, trainable=True
        ).state_dict()

        with pytest.raises(RuntimeError, match="from_curvature_coefficient"):
            _law(trainable=True).load_state_dict(legacy, strict=strict)

    def test_a_new_checkpoint_round_trips_into_the_new_law(self) -> None:
        """The over-restriction control: the refusal must not reject valid state."""

        source = _law(tail_slope=1.5, curvature_range=2.0)
        target = _law()
        target.load_state_dict(source.state_dict())

        torch.testing.assert_close(target.curvature_range, source.curvature_range)
        torch.testing.assert_close(target.tail_slope_magnitude, source.tail_slope_magnitude)

    def test_new_coordinates_are_refused_by_the_old_law(self) -> None:
        """The other direction, so neither law can quietly absorb the other."""

        with pytest.raises(RuntimeError):
            CurvatureElectronNucleusCuspLaw(trainable=True).load_state_dict(
                _law(trainable=True).state_dict(), strict=True
            )


class TestDiagnostics:
    def test_diagnostics_report_the_coordinates_and_their_raws(self) -> None:
        diagnostics = _law(tail_slope=1.99, curvature_range=1.0).scalar_diagnostics()
        assert diagnostics["tail_slope_magnitude"] == pytest.approx(1.99)
        assert diagnostics["curvature_range"] == pytest.approx(1.0)
        assert "raw_range" in diagnostics and "raw_kappa" in diagnostics

    def test_the_charge_dependent_coefficient_is_absent_from_diagnostics(self) -> None:
        """Reporting one ``c`` would reintroduce the helium-only reading.

        ``c`` differs per nucleus, so there is no single scalar to emit and a
        diagnostic that emitted one would be quietly wrong on any other atom.
        """

        assert "curvature_coefficient" not in _law().scalar_diagnostics()

    def test_frozen_diagnostics_omit_the_raw_axis(self) -> None:
        diagnostics = _law(trainable=False).scalar_diagnostics()
        assert "raw_range" not in diagnostics and "raw_kappa" not in diagnostics
