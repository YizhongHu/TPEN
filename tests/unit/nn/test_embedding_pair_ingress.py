"""The embedding CONSUMES the ordered-pair channel, at order 2 only.

The producing half is ``BoundedDistanceBasis``; this is the consumption seam
CD3 names. Pair channels enter at order 2 because that is the only order whose
tuple *is* an ordered pair -- the authority defines pair features and nothing
wider, so no triple analogue is invented for higher orders.

The failures worth guarding are both SILENT, and in opposite directions:

- a configured pair width with no pair tensor, which would surface later as a
  confusing shape error;
- an unconsumed pair tensor, which would silently discard typed features a
  caller went to the trouble of producing. That is the same class of defect as
  broadcasting pair information into one-body channels undocumented, merely
  inverted.

Both raise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tpen.data.batch import ElectronBatch
from tpen.nn.basis import BoundedDistanceBasis, RawCoordinateBasis
from tpen.nn.embedding import Embedding

PAIR_CHANNELS = 3


def _batch(n_walkers: int = 2, n_electrons: int = 2):
    generator = torch.Generator().manual_seed(11)
    spins = torch.ones(n_walkers, n_electrons, dtype=torch.float64)
    spins[..., 1::2] = -1.0
    return ElectronBatch(
        positions=torch.randn(n_walkers, n_electrons, 3, generator=generator, dtype=torch.float64),
        nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
        spins=spins,
    )


def _basis():
    return BoundedDistanceBasis(
        inner=RawCoordinateBasis(spatial_dim=3, include_spin=True), length=1.0
    )


def _embedding(pair_input_channels: int = PAIR_CHANNELS, max_order: int = 2, **kwargs):
    basis = _basis()
    return Embedding(
        max_order=max_order,
        spatial_dim=3,
        out_channels=4,
        hidden_channels=8,
        num_hidden_layers=1,
        include_spins=True,
        in_features=basis.out_features,
        pair_input_channels=pair_input_channels,
        **kwargs,
    ).to(torch.float64)


class TestWidths:
    def test_order_two_is_widened_by_exactly_the_pair_channel(self) -> None:
        with_pair = _embedding(PAIR_CHANNELS)
        without = _embedding(0)
        assert (
            with_pair._order_in_channels(2) - without._order_in_channels(2) == PAIR_CHANNELS
        )

    def test_order_one_is_untouched(self) -> None:
        """Order 1 has no pair; widening it would be an invented convention."""

        assert _embedding(PAIR_CHANNELS)._order_in_channels(1) == _embedding(0)._order_in_channels(1)

    def test_a_pair_width_needs_an_order_that_can_hold_a_pair(self) -> None:
        with pytest.raises(ValueError, match="requires max_order >= 2"):
            _embedding(PAIR_CHANNELS, max_order=1)

    def test_a_negative_pair_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _embedding(-1)


class TestTheSeamCarriesPairFeatures:
    def test_a_forward_with_pair_features_succeeds(self) -> None:
        features = _basis()(_batch())
        out = _embedding()(features)
        assert torch.isfinite(out.blocks[2]).all()

    def test_the_pair_channel_actually_changes_the_order_two_block(self) -> None:
        """Consumed, not merely accepted.

        Same embedding, same one-body features, only the pair contents differ.
        If the pair tensor were concatenated and then ignored -- or dropped --
        these would agree.
        """

        embedding = _embedding()
        features = _basis()(_batch())

        perturbed = type(features)(
            one_body=features.one_body,
            pair=features.pair + 0.5,
            metadata=dict(features.metadata),
        )

        base = embedding(features).blocks[2]
        moved = embedding(perturbed).blocks[2]
        assert not torch.allclose(base, moved)

    def test_the_one_body_block_is_unaffected_by_the_pair_channel(self) -> None:
        """Order 1 must not see pair information; that would be the broadcast
        the authority forbids, arriving by a different route."""

        embedding = _embedding()
        features = _basis()(_batch())
        perturbed = type(features)(
            one_body=features.one_body,
            pair=features.pair + 0.5,
            metadata=dict(features.metadata),
        )
        torch.testing.assert_close(
            embedding(features).blocks[1], embedding(perturbed).blocks[1]
        )


class TestBothSilentMismatchesRaise:
    def test_configured_pair_width_with_no_pair_tensor_raises(self) -> None:
        plain = RawCoordinateBasis(spatial_dim=3, include_spin=True)(_batch())
        embedding = Embedding(
            max_order=2,
            spatial_dim=3,
            out_channels=4,
            hidden_channels=8,
            num_hidden_layers=1,
            include_spins=True,
            in_features=plain.one_body.shape[-1],
            pair_input_channels=PAIR_CHANNELS,
        ).to(torch.float64)
        with pytest.raises(ValueError, match="carry no pair channel"):
            embedding(plain)

    def test_an_unconsumed_pair_tensor_raises(self) -> None:
        """Silently discarding typed features is the inverse defect."""

        features = _basis()(_batch())
        with pytest.raises(ValueError, match="silently discarded"):
            _embedding(0)(features)

    def test_a_wrong_pair_width_raises(self) -> None:
        features = _basis()(_batch())
        with pytest.raises(ValueError, match="expected pair width"):
            _embedding(PAIR_CHANNELS + 1)(features)


class TestGradients:
    def test_gradient_reaches_the_coordinates_through_the_pair_channel(self) -> None:
        positions = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
        batch = ElectronBatch(
            positions=positions,
            nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]] * 2, dtype=torch.float64),
        )
        _embedding()(_basis()(batch)).blocks[2].sum().backward()
        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()
        assert not torch.allclose(positions.grad, torch.zeros_like(positions.grad))

    def test_the_added_pair_weight_columns_have_reachable_gradient(self) -> None:
        """The authority requires the added columns stay trainable.

        Reaching the first layer's weight is not enough: the gradient must
        reach the SLICE of it that the pair channel feeds, which is the tail
        columns. A zero gradient there would mean the pair channel is wired but
        dead.
        """

        embedding = _embedding()
        embedding(_basis()(_batch())).blocks[2].sum().backward()

        first_weight = next(
            p for name, p in embedding.order_mlps["2"].named_parameters() if name.endswith("weight")
        )
        pair_columns = first_weight.grad[..., -PAIR_CHANNELS:]
        assert pair_columns.abs().sum() > 0
