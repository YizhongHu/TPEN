"""Matched B seed starts: added input columns are zero, so B00 == B10 at step 0.

The authority requires copying shared raw-input weights exactly and setting the
added input columns to zero, so that a bounded-distance cell begins as the SAME
FUNCTION as its raw counterpart. The point is scientific rather than tidy: it
removes initial-function change as the first explanation for any later
difference between the two cells, making this a learned-feature experiment
rather than a comparison of random initialisation recipes.

The decisive test is FUNCTION EQUALITY, not column positions
---------------------------------------------------------
Order 2's input is ``[v_i, v_j, pair]``, so appending one column per particle
inserts a zero at the end of EACH PARTICLE BLOCK rather than at the end of the
row. A "zero the last k columns" implementation is wrong for every order above
1 while looking right for order 1.

Checking the intended column indices would re-apply exactly the reasoning that
produced the mapping, so an off-by-one in my derivation would agree with itself
and pass. Comparing the two models' OUTPUTS cannot: it fails for any mapping
error whatever, because a misplaced weight changes the function.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tpen.data.batch import ElectronBatch
from tpen.nn.basis import BoundedDistanceBasis, RawCoordinateBasis
from tpen.nn.embedding import Embedding, matched_start_embedding, transplant_raw_input_columns

PAIR_CHANNELS = 3


def _batch(n_walkers: int = 3, n_electrons: int = 2):
    generator = torch.Generator().manual_seed(5)
    spins = torch.ones(n_walkers, n_electrons, dtype=torch.float64)
    spins[..., 1::2] = -1.0
    return ElectronBatch(
        positions=torch.randn(n_walkers, n_electrons, 3, generator=generator, dtype=torch.float64),
        nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
        spins=spins,
    )


def _raw_basis():
    return RawCoordinateBasis(spatial_dim=3, include_spin=True)


def _augmented_basis():
    return BoundedDistanceBasis(inner=_raw_basis(), length=1.0)


def _embedding(in_features: int, pair_input_channels: int, seed: int = 0, max_order: int = 2):
    torch.manual_seed(seed)
    return Embedding(
        max_order=max_order,
        spatial_dim=3,
        out_channels=4,
        hidden_channels=8,
        num_hidden_layers=1,
        include_spins=True,
        in_features=in_features,
        pair_input_channels=pair_input_channels,
    ).to(torch.float64)


def _pair() -> tuple[Embedding, Embedding]:
    """Return (raw cell, bounded-distance cell) built from DIFFERENT seeds.

    Different seeds on purpose: if both were seeded identically the shared
    weights would already agree and the transplant would be untestable.
    """

    raw = _embedding(_raw_basis().out_features, 0, seed=0)
    augmented = _embedding(_augmented_basis().out_features, PAIR_CHANNELS, seed=99)
    return raw, augmented


class TestTheTwoCellsStartAsOneFunction:
    def test_outputs_agree_exactly_after_transplant(self) -> None:
        """The decisive check: same function, every order."""

        raw, augmented = _pair()
        batch = _batch()

        before_raw = raw(_raw_basis()(batch))
        before_aug = augmented(_augmented_basis()(batch))
        assert not torch.allclose(before_raw.blocks[1], before_aug.blocks[1]), (
            "the two cells already agree before transplanting; this fixture cannot "
            "detect whether the transplant did anything"
        )

        transplant_raw_input_columns(raw, augmented)

        after_raw = raw(_raw_basis()(batch))
        after_aug = augmented(_augmented_basis()(batch))
        for order in range(1, len(after_raw.blocks)):
            torch.testing.assert_close(
                after_raw.blocks[order],
                after_aug.blocks[order],
                msg=f"order {order} differs after transplant",
            )

    def test_order_two_specifically_agrees(self) -> None:
        """Order 2 is where a "zero the last k columns" bug would survive.

        Order 1 has one particle block, so the added column IS last and a naive
        implementation happens to be right. Order 2 is the discriminating case.
        """

        raw, augmented = _pair()
        batch = _batch()
        transplant_raw_input_columns(raw, augmented)
        torch.testing.assert_close(
            raw(_raw_basis()(batch)).blocks[2],
            augmented(_augmented_basis()(batch)).blocks[2],
        )

    def test_a_naive_tail_zeroing_would_NOT_have_agreed(self) -> None:
        """Positive control for the test above: prove the trap is real.

        Emulates "copy the source block into the target's leading columns and
        zero the tail" -- the plausible wrong implementation -- and requires it
        to produce a DIFFERENT function. Without this, the agreement tests
        could be passing against a mapping that happens not to matter.
        """

        raw, augmented = _pair()
        batch = _batch()

        first = next(
            p for name, p in augmented.order_mlps["2"].named_parameters() if name.endswith("weight")
        )
        source_first = next(
            p for name, p in raw.order_mlps["2"].named_parameters() if name.endswith("weight")
        )
        with torch.no_grad():
            for name, target_param in augmented.order_mlps["2"].named_parameters():
                source_param = dict(raw.order_mlps["2"].named_parameters())[name]
                if source_param.shape == target_param.shape:
                    target_param.copy_(source_param)
            first.zero_()
            width = source_first.shape[1]
            first[:, :width].copy_(source_first)

        assert not torch.allclose(
            raw(_raw_basis()(batch)).blocks[2],
            augmented(_augmented_basis()(batch)).blocks[2],
        ), "tail-zeroing produced the same function; order 2 cannot discriminate here"


class TestTheAddedColumnsStayTrainable:
    def test_the_added_columns_are_zero_but_receive_gradient(self) -> None:
        """Zeroed is not frozen. The authority requires a reachable gradient."""

        raw, augmented = _pair()
        transplant_raw_input_columns(raw, augmented)

        first = next(
            p for name, p in augmented.order_mlps["2"].named_parameters() if name.endswith("weight")
        )
        pair_columns = first[:, -PAIR_CHANNELS:]
        torch.testing.assert_close(pair_columns, torch.zeros_like(pair_columns))

        augmented(_augmented_basis()(_batch())).blocks[2].sum().backward()
        assert first.grad is not None
        assert first.grad[:, -PAIR_CHANNELS:].abs().sum() > 0, (
            "the zeroed pair columns receive no gradient; they are dead rather than "
            "merely starting at zero"
        )

    def test_the_per_particle_added_column_also_receives_gradient(self) -> None:
        """The interleaved zeros, not just the pair block at the end."""

        raw, augmented = _pair()
        transplant_raw_input_columns(raw, augmented)
        width = augmented.particle_input_channels

        first = next(
            p for name, p in augmented.order_mlps["2"].named_parameters() if name.endswith("weight")
        )
        augmented(_augmented_basis()(_batch())).blocks[2].sum().backward()

        # q_i sits at the end of the first particle block.
        assert first.grad[:, width - 1].abs().sum() > 0


class TestRefusals:
    def test_a_narrower_target_is_refused(self) -> None:
        raw, augmented = _pair()
        with pytest.raises(ValueError, match="at least as wide"):
            transplant_raw_input_columns(augmented, raw)

    def test_a_source_with_a_pair_channel_is_refused(self) -> None:
        """The source is the RAW cell by definition."""

        _, augmented = _pair()
        other = _embedding(_augmented_basis().out_features, PAIR_CHANNELS, seed=1)
        with pytest.raises(ValueError, match="no pair channel"):
            transplant_raw_input_columns(augmented, other)

    def test_mismatched_max_order_is_refused(self) -> None:
        raw = _embedding(_raw_basis().out_features, 0, seed=0, max_order=1)
        _, augmented = _pair()
        with pytest.raises(ValueError, match="equal max_order"):
            transplant_raw_input_columns(raw, augmented)


class TestMatchedStartFactory:
    def test_returns_augmented_and_leaves_source_unchanged(self) -> None:
        raw, augmented = _pair()
        source_before = {name: value.detach().clone() for name, value in raw.named_parameters()}

        returned = matched_start_embedding(raw=raw, augmented=augmented)

        assert returned is augmented
        for name, value in raw.named_parameters():
            assert torch.equal(value, source_before[name])

    def test_refuses_width_mismatch(self) -> None:
        raw = _embedding(_raw_basis().out_features, 0, seed=0)
        narrower = _embedding(_raw_basis().out_features - 1, 0, seed=1)
        with pytest.raises(ValueError, match="at least as wide"):
            matched_start_embedding(raw=raw, augmented=narrower)
