"""Minimal B pair ingress: bounded distances and the ordered pair channel.

CD3 adds exactly ``q_i`` to the one-body input and ``(q_i, q_j, q_ij)`` to the
ordered ``(i, j)`` pair input, with ``q(r) = r^2 / (l^2 + r^2)`` and ``l = 1``
bohr. Nothing else -- no extra radial or angular family.

The acceptance contract's falsifier is not about the feature maths:

    a zero-row rank changes a valid global result or requires special caller
    branching

so the neutral-semantics cases below are the load-bearing ones, and the closed
forms exist mostly so that a neutral-semantics test cannot pass against a basis
that quietly computes nothing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tpen.data.batch import ElectronBatch
from tpen.data.permutation import Permutation
from tpen.nn.basis import BoundedDistanceBasis, RawCoordinateBasis, bounded_distance

LENGTH = 1.0


def _basis(spatial_dim: int = 3, include_spin: bool = True, length: float = LENGTH):
    return BoundedDistanceBasis(
        inner=RawCoordinateBasis(spatial_dim=spatial_dim, include_spin=include_spin),
        length=length,
    )


def _batch(positions, spins=None):
    # as_tensor, NOT tensor(list): a caller passing torch.zeros(2, 0, 3) through
    # .tolist() gets [[], []], which collapses to shape (2, 0) and loses the
    # spatial axis. ElectronBatch then rejects it for ndim < 3 -- correctly, but
    # the failure reads like "zero electrons are unsupported" when it is really
    # "this helper destroyed the shape".
    positions = torch.as_tensor(positions, dtype=torch.float64)
    if spins is None:
        spins = torch.ones(positions.shape[:-1], dtype=torch.float64)
        spins[..., 1::2] = -1.0
    return ElectronBatch(
        positions=positions,
        nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
        spins=spins,
    )


class TestTheScalarTransform:
    def test_matches_the_closed_form(self) -> None:
        q = bounded_distance(torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64), LENGTH)
        torch.testing.assert_close(q, torch.tensor([0.0, 0.5, 0.75], dtype=torch.float64))

    def test_is_bounded_in_zero_one(self) -> None:
        """Bounded is the whole point: it adds no exponential tail slope."""

        q = bounded_distance(torch.logspace(-6, 6, 50, dtype=torch.float64), LENGTH)
        assert (q >= 0).all() and (q < 1).all()

    def test_is_smooth_at_zero_separation(self) -> None:
        """The reason the argument is r^2 and not r.

        ``sqrt`` has an infinite gradient at the origin, so a chain rule through
        it produces NaN exactly where two particles coincide -- the
        configuration a wavefunction most needs to survive. Taking r^2 keeps the
        expression polynomial in the coordinates.
        """

        coordinate = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        bounded_distance(coordinate.square().sum(), LENGTH).backward()
        assert coordinate.grad is not None
        assert torch.isfinite(coordinate.grad).all()

    def test_rejects_a_nonpositive_length(self) -> None:
        with pytest.raises(ValueError, match="length must be positive"):
            bounded_distance(torch.zeros(1), 0.0)


class TestTheOrderedPairChannel:
    def test_pair_entries_are_q_i_q_j_q_ij(self) -> None:
        """Two electrons at unit distance from the origin, separated by 2."""

        features = _basis()(_batch([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]))
        pair = features.pair
        assert pair is not None
        assert pair.shape == (1, 2, 2, 3)

        # q_1 = q_2 = 1/(1+1) = 0.5 ; q_12 = 4/(1+4) = 0.8
        entry = pair[0, 0, 1]
        torch.testing.assert_close(entry, torch.tensor([0.5, 0.5, 0.8], dtype=torch.float64))

    def test_the_channel_is_ordered_not_symmetrised(self) -> None:
        """Entry (i,j) must be (q_i, q_j, q_ij), with q_i and q_j distinguishable.

        Uses electrons at DIFFERENT radii so the first two channels differ; with
        equal radii a symmetrised implementation would be indistinguishable
        from an ordered one.
        """

        features = _basis()(_batch([[[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]))
        pair = features.pair
        ij, ji = pair[0, 0, 1], pair[0, 1, 0]

        assert ij[0] != ij[1], "q_i and q_j coincide; this case cannot detect symmetrisation"
        torch.testing.assert_close(ij[0], ji[1])
        torch.testing.assert_close(ij[1], ji[0])
        torch.testing.assert_close(ij[2], ji[2])

    def test_the_diagonal_is_defined_rather_than_a_hole(self) -> None:
        """(i,i) is (q_i, q_i, 0). No branching, no NaN."""

        features = _basis()(_batch([[[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]))
        diagonal = features.pair[0, 0, 0]
        torch.testing.assert_close(diagonal[2], torch.tensor(0.0, dtype=torch.float64))
        torch.testing.assert_close(diagonal[0], diagonal[1])
        assert torch.isfinite(features.pair).all()

    def test_the_pair_channel_is_exactly_three_wide(self) -> None:
        """Only q_i, q_j, q_ij. No extra radial or angular family."""

        assert _basis().pair_features == 3
        assert _basis()(_batch([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])).pair.shape[-1] == 3


class TestNeutralSemanticsForZeroRows:
    """The contract's falsifier: no special caller branching, no changed result."""

    def test_a_zero_electron_batch_produces_well_formed_empty_features(self) -> None:
        features = _basis()(_batch(torch.zeros(2, 0, 3)))
        assert features.one_body.shape == (2, 0, _basis().out_features)
        assert features.pair.shape == (2, 0, 0, 3)

    def test_a_one_electron_batch_has_no_off_diagonal_pair(self) -> None:
        """One electron means zero ordered pairs, and the shape says so."""

        features = _basis()(_batch([[[1.0, 0.0, 0.0]]]))
        assert features.pair.shape == (1, 1, 1, 3)
        torch.testing.assert_close(
            features.pair[0, 0, 0, 2], torch.tensor(0.0, dtype=torch.float64)
        )

    def test_a_zero_row_batch_needs_no_branch_and_stays_finite(self) -> None:
        """The same call shape works for 0, 1 and 2 electrons.

        This is the property the contract asks for stated directly: a caller
        does not test the electron count before calling.
        """

        basis = _basis()
        for count in (0, 1, 2):
            batch = _batch(torch.zeros(1, count, 3))
            features = basis(batch)
            assert features.pair.shape == (1, count, count, 3)
            assert torch.isfinite(features.one_body).all()
            assert torch.isfinite(features.pair).all()


class TestRawFeaturesArePreserved:
    def test_the_inner_one_body_block_is_unchanged(self) -> None:
        """Cartesian and spin survive verbatim; q_i is APPENDED, not blended."""

        batch = _batch([[[1.0, 2.0, 3.0], [0.0, 3.0, 0.0]]])
        inner = RawCoordinateBasis(spatial_dim=3, include_spin=True)
        augmented = BoundedDistanceBasis(inner=inner, length=LENGTH)

        raw = inner(batch).one_body
        combined = augmented(batch).one_body

        torch.testing.assert_close(combined[..., : raw.shape[-1]], raw)
        assert combined.shape[-1] == raw.shape[-1] + 1

    def test_out_features_counts_the_appended_column_once(self) -> None:
        inner = RawCoordinateBasis(spatial_dim=3, include_spin=True)
        assert BoundedDistanceBasis(inner=inner).out_features == inner.out_features + 1

    def test_the_spin_channel_is_not_duplicated(self) -> None:
        """A second spin column would widen the input silently rather than fail."""

        batch = _batch([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        inner = RawCoordinateBasis(spatial_dim=3, include_spin=True)
        combined = BoundedDistanceBasis(inner=inner)(batch).one_body
        # spatial_dim + spin + q_i == 5, not 6.
        assert combined.shape[-1] == 5

    def test_q_ij_does_not_leak_into_the_one_body_channels(self) -> None:
        """The authority forbids broadcasting pair info as a substitute.

        Moving electron j while holding i fixed changes q_ij, so if any pair
        quantity had been broadcast into the one-body block, electron i's
        one-body row would move too.
        """

        basis = _basis()
        near = basis(_batch([[[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]]])).one_body[0, 0]
        far = basis(_batch([[[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])).one_body[0, 0]
        torch.testing.assert_close(near, far)


class TestEquivariance:
    def test_permuting_electrons_permutes_both_pair_axes(self) -> None:
        """``ElectronBasisFeatures.permute`` already handles this; check it holds
        for features this basis actually produces rather than assuming it."""

        batch = _batch([[[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
        swap = Permutation((1, 0))
        basis = _basis()

        direct = basis(batch.permute(swap))
        permuted = basis(batch).permute(swap)

        torch.testing.assert_close(direct.one_body, permuted.one_body)
        torch.testing.assert_close(direct.pair, permuted.pair)

    def test_the_swap_actually_changes_the_features(self) -> None:
        """Otherwise the equivariance check above is vacuous."""

        batch = _batch([[[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
        basis = _basis()
        assert not torch.allclose(basis(batch).one_body, basis(batch.permute(Permutation((1, 0)))).one_body)


class TestGradients:
    def test_gradients_reach_the_coordinates_through_both_channels(self) -> None:
        positions = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float64, requires_grad=True
        )
        batch = ElectronBatch(
            positions=positions,
            nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        )
        features = _basis()(batch)
        (features.one_body.sum() + features.pair.sum()).backward()

        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()
        assert not torch.allclose(positions.grad, torch.zeros_like(positions.grad))

    def test_gradients_are_finite_at_coincident_electrons(self) -> None:
        """The configuration that a sqrt-based q would turn into NaN."""

        positions = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
        batch = ElectronBatch(
            positions=positions,
            nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        )
        features = _basis()(batch)
        (features.one_body.sum() + features.pair.sum()).backward()
        assert torch.isfinite(positions.grad).all()
