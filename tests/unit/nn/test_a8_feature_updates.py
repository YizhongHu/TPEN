"""The three admitted A8 feature-update arms, and what they are not.

The study admits exactly three: ``x + u``, ``u``, and ``x + g(R) u``. All three
are composed from primitives that already existed:

===================  ====================================================
arm                  spelling
===================  ====================================================
residual             ``update: ResidualUpdater``
replacement          ``update: ReplaceUpdater``
Gaussian residual    ``update: ResidualUpdater`` plus
                     ``update_envelope: GaussianCoordinateEnvelope``
===================  ====================================================

No new primitive was needed. An earlier attempt in this slice added a second
Gaussian gate before noticing ``GaussianCoordinateEnvelope`` already computed
``exp(-sum_i |r_i|^2 / (2 sigma^2))`` and that ``CoordinateEnvelope`` returns
``type(features)(blocks)``, so it works unchanged at the update seam as well as
the feature seam. That duplicate is gone. The lesson is recorded in this
docstring rather than in a commit nobody will read again: searching for
implementations of the ``update_envelope`` SEAM found nothing, because the
existing primitive is reached through the ``feature_envelope`` seam. Searching
for the seam is not searching for the concept.

Most of what follows tests what the Gaussian arm is NOT, because substitution
rather than breakage is the failure mode here. A norm gate, an RMS
normalization and this envelope all multiply by something in ``(0, 1)`` and
look alike in a training curve. Only one reads the electron coordinates.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tpen.data.batch import ElectronBatch
from tpen.data.real import Feature, Update
from tpen.nn import GaussianCoordinateEnvelope, ReplaceUpdater, ResidualUpdater
from tpen.nn.context import TPENForwardContext


def _context(positions):
    batch = ElectronBatch(
        positions=torch.tensor(positions, dtype=torch.float64),
        nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
    )
    return TPENForwardContext(batch=batch)


def _blocks(value, batch_size=1, channels=2, n_particles=2):
    return [
        torch.zeros(batch_size, 0, dtype=torch.float64),
        torch.full((batch_size, channels, n_particles), value, dtype=torch.float64),
        torch.full((batch_size, channels, n_particles, n_particles), value, dtype=torch.float64),
    ]


def _update(value=1.0, **kwargs):
    return Update(_blocks(value, **kwargs))


def _feature(value=5.0, **kwargs):
    return Feature(_blocks(value, **kwargs))


class TestGaussianGateValue:
    def test_matches_the_closed_form(self) -> None:
        """Electrons at (1,0,0) and (0,2,0): r1^2 + r2^2 = 5."""

        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        gate = envelope.scalar(_context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]).batch)
        assert gate.shape == (1,)
        assert gate.item() == pytest.approx(math.exp(-5.0 / 2.0))

    def test_is_one_at_the_nucleus(self) -> None:
        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        gate = envelope.scalar(_context([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]).batch)
        assert gate.item() == pytest.approx(1.0)

    def test_decays_far_from_the_nucleus(self) -> None:
        """The question the arm asks: does the correction fade out there?"""

        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        near = envelope.scalar(_context([[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]]).batch)
        far = envelope.scalar(_context([[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]]).batch)
        assert near.item() > 0.9
        assert far.item() < 1e-20

    def test_sigma_is_one_bohr_for_the_study(self) -> None:
        """The literal control fixes the width; this pins the default."""

        assert GaussianCoordinateEnvelope().sigma == 1.0

    def test_the_gate_is_permutation_invariant(self) -> None:
        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        first = envelope.scalar(_context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]).batch)
        swapped = envelope.scalar(_context([[[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]]).batch)
        assert swapped.item() == pytest.approx(first.item())


class TestTheGateIsAnchoredAtTheOrigin:
    """A stated limitation, pinned so it cannot change silently.

    ``GaussianCoordinateEnvelope`` measures ``sum_i |r_i|^2`` from the ORIGIN,
    not from the nucleus. That is exactly the authority's formula, because the
    literal control protocol places the nucleus at the origin -- so this is not
    a defect and is deliberately not "fixed" here.

    It is pinned because the two only differ when the atom is moved, and a
    future study that translates the system would silently change the model
    rather than translate it. Anyone who moves the nucleus must read this.
    """

    def test_translating_the_atom_changes_the_gate(self) -> None:
        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        at_origin = envelope.scalar(_context([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).batch)
        translated = envelope.scalar(_context([[[6.0, 0.0, 0.0], [5.0, 1.0, 0.0]]]).batch)
        assert translated.item() != pytest.approx(at_origin.item())

    def test_the_nucleus_is_at_the_origin_in_the_control_config(self) -> None:
        """Which is what makes the origin anchoring correct for this study."""

        from omegaconf import OmegaConf

        cfg = OmegaConf.load("experiments/atomistic/he-importance/configs/train.yaml")
        assert OmegaConf.select(cfg, "system.nuclei.positions") == [[0.0, 0.0, 0.0]]


class TestWhatTheGaussianArmIsNot:
    def test_the_gate_does_not_depend_on_the_update(self) -> None:
        """NOT exp(-u^2) and not any function of the update's magnitude.

        Scaling u by 3 must scale the output by exactly 3. A norm gate would
        not satisfy this.
        """

        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        context = _context([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        small = envelope(_update(1.0), context)
        large = envelope(_update(3.0), context)
        for lhs, rhs in zip(small.blocks[1:], large.blocks[1:]):
            assert torch.allclose(3.0 * lhs, rhs)

    def test_one_scalar_is_shared_across_orders_channels_and_tuples(self) -> None:
        """Not per-electron, not per-channel, not per-tuple."""

        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        gated = envelope(_update(1.0), _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]))
        expected = math.exp(-5.0 / 2.0)
        for block in gated.blocks[1:]:
            assert torch.allclose(block, torch.full_like(block, expected))

    def test_the_envelope_returns_an_update_not_a_feature(self) -> None:
        """It must be usable at the update seam, which is how the arm is spelled."""

        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        out = envelope(_update(1.0), _context([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))
        assert isinstance(out, Update)

    def test_the_norm_gated_updater_is_not_exported(self) -> None:
        """It is not an A8 arm and must not be reachable as ``tpen.nn.*``."""

        import tpen.nn

        assert not hasattr(tpen.nn, "NormGatedUpdater")
        assert "NormGatedUpdater" not in tpen.nn.__all__

    def test_the_envelope_has_no_parameters(self) -> None:
        """One configuration-level scalar, not a learned gate."""

        assert list(GaussianCoordinateEnvelope().parameters()) == []


class TestTheThreeArms:
    def test_residual_is_x_plus_u(self) -> None:
        out = ResidualUpdater()(_feature(5.0), _update(1.0))
        assert torch.allclose(out.blocks[1], torch.full_like(out.blocks[1], 6.0))

    def test_replacement_is_u(self) -> None:
        out = ReplaceUpdater()(_feature(5.0), _update(1.0))
        assert torch.allclose(out.blocks[1], torch.full_like(out.blocks[1], 1.0))

    def test_gaussian_residual_is_x_plus_gate_times_u(self) -> None:
        context = _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        gated = GaussianCoordinateEnvelope(sigma=1.0)(_update(1.0), context)
        out = ResidualUpdater()(_feature(5.0), gated)
        expected = 5.0 + math.exp(-5.0 / 2.0)
        assert torch.allclose(out.blocks[1], torch.full_like(out.blocks[1], expected))

    def test_all_three_are_reachable_from_tpen_nn(self) -> None:
        """A config names these by dotted path; unexported means unusable."""

        import tpen.nn

        for name in ("ResidualUpdater", "ReplaceUpdater", "GaussianCoordinateEnvelope"):
            assert hasattr(tpen.nn, name), name
            assert name in tpen.nn.__all__, name

    def test_the_three_arms_are_distinguishable(self) -> None:
        """Otherwise the scan compares one thing three times."""

        x, u = _feature(5.0), _update(1.0)
        context = _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])

        residual = ResidualUpdater()(x, u).blocks[1]
        replacement = ReplaceUpdater()(x, u).blocks[1]
        gaussian = ResidualUpdater()(x, GaussianCoordinateEnvelope(sigma=1.0)(u, context)).blocks[1]

        assert not torch.allclose(residual, replacement)
        assert not torch.allclose(residual, gaussian)
        assert not torch.allclose(replacement, gaussian)


class TestGradients:
    def test_the_gate_passes_gradient_to_the_update(self) -> None:
        block = torch.full((1, 2, 2), 1.0, dtype=torch.float64, requires_grad=True)
        update = Update([torch.zeros(1, 0, dtype=torch.float64), block])
        context = _context([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
        GaussianCoordinateEnvelope(sigma=1.0)(update, context).blocks[1].sum().backward()
        assert block.grad is not None
        assert torch.allclose(block.grad, torch.full_like(block.grad, math.exp(-5.0 / 2.0)))

    def test_the_gate_carries_gradient_to_the_coordinates(self) -> None:
        """The gate is part of the wavefunction, so the local energy needs this."""

        positions = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float64, requires_grad=True
        )
        batch = ElectronBatch(
            positions=positions,
            nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
        )
        GaussianCoordinateEnvelope(sigma=1.0).scalar(batch).sum().backward()
        assert positions.grad is not None
        assert not torch.allclose(positions.grad, torch.zeros_like(positions.grad))
