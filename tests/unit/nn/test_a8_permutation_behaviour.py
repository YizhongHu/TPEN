"""Permutation behaviour is identical across the three A8 arms.

CD2's plan row lists "permutation behavior" among the contracts an A8 arm must
bind. TPEN's equivariance is established upstream of the feature update, so the
claim here is narrow and precise: **switching A8 arm must not change how the
model responds to an electron permutation.**

That framing is deliberate. Asserting a particular sign convention would be
asserting something about the READOUT, not about A8, and a failure would be
attributed to the wrong slice. What is asserted instead:

1. ``logabs`` is invariant under permutation, for each arm. This is
   convention-free -- ``|psi|`` cannot depend on electron labelling whatever the
   antisymmetry convention is -- so it is safe to assert outright.
2. All three arms agree with each other on the sign response. This is the actual
   CD2 claim and needs no convention at all: whatever the model does, every arm
   must do the same thing.

The two together catch an arm that broke equivariance without this module having
to encode a physics convention it is not the owner of.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch
from tpen.data.permutation import Permutation
from tpen.nn import GaussianCoordinateEnvelope, ReplaceUpdater, ResidualUpdater

CONTROL_CONFIG = Path("experiments/atomistic/he-importance/configs/train.yaml")
A8_ARMS = ("residual", "replacement", "gaussian")

# Helium is two electrons, so the only nontrivial permutation is the swap.
SWAP = Permutation((1, 0))


def _cfg():
    cfg = OmegaConf.load(CONTROL_CONFIG)
    OmegaConf.resolve(cfg)
    return cfg


def _model_for(arm: str, shared_state=None):
    cfg = _cfg()
    model = instantiate(cfg.model).to(getattr(torch, str(cfg.runtime.dtype)))
    layer = model.stack.layers[0]
    if arm == "residual":
        layer.update = ResidualUpdater()
    elif arm == "replacement":
        layer.update = ReplaceUpdater()
    elif arm == "gaussian":
        layer.update = ResidualUpdater()
        layer.update_envelope = GaussianCoordinateEnvelope(sigma=1.0)
    else:  # pragma: no cover
        raise ValueError(arm)
    if shared_state is not None:
        model.load_state_dict(shared_state)
    return model


def _batch(n_walkers: int = 4):
    generator = torch.Generator().manual_seed(7)
    return ElectronBatch(
        positions=torch.randn(n_walkers, 2, 3, generator=generator, dtype=torch.float64),
        nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
        spins=torch.tensor([[1, -1]] * n_walkers, dtype=torch.float64),
    )


class TestMagnitudeIsPermutationInvariant:
    """``|psi|`` cannot depend on how the electrons are labelled."""

    @pytest.mark.parametrize("arm", A8_ARMS)
    def test_logabs_is_invariant(self, arm: str) -> None:
        model = _model_for(arm)
        batch = _batch()
        direct = model(batch).logabs.detach()
        permuted = model(batch.permute(SWAP)).logabs.detach()
        torch.testing.assert_close(direct, permuted)

    def test_the_permutation_actually_changes_the_batch(self) -> None:
        """Otherwise every invariance assertion above is vacuous.

        A swap that silently returned the same tensor would make all three
        parametrized cases pass while testing nothing.
        """

        batch = _batch()
        assert not torch.equal(batch.positions, batch.permute(SWAP).positions)


class TestTheArmsAgreeOnPermutationResponse:
    """The actual CD2 claim: A8 must not change permutation behaviour."""

    def test_all_three_arms_share_one_sign_response(self) -> None:
        """Convention-free. Whatever the readout does, every arm does the same.

        Weights are shared across arms so the only difference is the arm, which
        the shared state-dict key set makes possible -- see
        ``test_a8_parameter_layout.py`` for why that sharing is also a
        checkpoint hazard.
        """

        batch = _batch()
        shared = _model_for("residual").state_dict()

        responses = {}
        for arm in A8_ARMS:
            model = _model_for(arm, shared_state=shared)
            direct = model(batch)
            permuted = model(batch.permute(SWAP))
            responses[arm] = (direct.sign.detach() * permuted.sign.detach())

        torch.testing.assert_close(responses["residual"], responses["replacement"])
        torch.testing.assert_close(responses["residual"], responses["gaussian"])

    def test_the_arms_are_still_behaviourally_distinct(self) -> None:
        """Agreeing on permutation response is not being the same model.

        Without this, the agreement above could be read as the arms doing
        nothing at all.
        """

        batch = _batch()
        shared = _model_for("residual").state_dict()
        outputs = {
            arm: _model_for(arm, shared_state=shared)(batch).logabs.detach() for arm in A8_ARMS
        }
        assert not torch.allclose(outputs["residual"], outputs["replacement"])
        assert not torch.allclose(outputs["residual"], outputs["gaussian"])


class TestTheGateItselfIsInvariant:
    def test_the_gaussian_gate_does_not_see_electron_labels(self) -> None:
        """A sum over electrons, so relabelling cannot move it.

        Pinned on the real batch shape rather than only on the unit fixture, so
        a broadcasting change that made the gate per-electron would be caught
        here as well as in the unit test.
        """

        envelope = GaussianCoordinateEnvelope(sigma=1.0)
        batch = _batch()
        torch.testing.assert_close(
            envelope.scalar(batch), envelope.scalar(batch.permute(SWAP))
        )
