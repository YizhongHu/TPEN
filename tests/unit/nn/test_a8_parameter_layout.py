"""Deterministic parameter registration and layout across the three A8 arms.

L1b/CD2's falsifier is "equivalent builds register different parameter order or
names". This module is that falsifier, plus the A8-specific consequence.

Two DIFFERENT identities, and neither subsumes the other
--------------------------------------------------------
``ParameterLayout`` slots carry ``ordinal``, ``shape``, ``numel`` and ``dtype``
-- and NO NAME. ``state_dict`` restore, by contrast, keys entirely on names. So:

- a rename that preserves order, shape and dtype leaves the LAYOUT identical and
  breaks a ``strict=True`` restore;
- a reorder that preserves names breaks the layout and leaves the state dict
  loadable.

The contract says "order or names", so both are checked here, separately.

Why this matters before any DDP wrapping
----------------------------------------
The acceptance contract requires layout and names to be pre-wrap invariants.
Nothing here wraps a model -- that needs a distributed runtime this slice does
not require -- but a layout that is already nondeterministic single-process
cannot become deterministic under a wrapper, so this is the part that can be
established now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.nn import GaussianCoordinateEnvelope, ReplaceUpdater, ResidualUpdater

CONTROL_CONFIG = Path("experiments/atomistic/he-importance/configs/train.yaml")


def _control_cfg():
    cfg = OmegaConf.load(CONTROL_CONFIG)
    OmegaConf.resolve(cfg)
    return cfg


def _build(cfg=None):
    """Instantiate the control model. Separate calls are independent builds."""

    cfg = _control_cfg() if cfg is None else cfg
    return instantiate(cfg.model).to(getattr(torch, str(cfg.runtime.dtype)))


def _names(model) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _layout_signature(model) -> list[tuple[int, tuple[int, ...], int, str]]:
    """Positional identity: ordinal, shape, numel, dtype. Deliberately nameless."""

    return [(s.ordinal, tuple(s.shape), s.numel, str(s.dtype)) for s in model.parameter_layout.slots]


def _with_arm(arm: str):
    """Build the control model wired for one A8 arm."""

    model = _build()
    layer = model.stack.layers[0]
    if arm == "residual":
        layer.update = ResidualUpdater()
    elif arm == "replacement":
        layer.update = ReplaceUpdater()
    elif arm == "gaussian":
        layer.update = ResidualUpdater()
        layer.update_envelope = GaussianCoordinateEnvelope(sigma=1.0)
    else:  # pragma: no cover - guards a typo in a parametrize list
        raise ValueError(arm)
    return model


A8_ARMS = ("residual", "replacement", "gaussian")


class TestEquivalentBuildsAgree:
    """The contract's falsifier, both halves of it."""

    def test_two_builds_register_the_same_parameter_names_in_the_same_order(self) -> None:
        assert _names(_build()) == _names(_build())

    def test_two_builds_produce_the_same_layout(self) -> None:
        assert _layout_signature(_build()) == _layout_signature(_build())

    def test_the_layout_is_dense_and_ordinal_ordered(self) -> None:
        slots = _build().parameter_layout.slots
        assert [s.ordinal for s in slots] == list(range(len(slots)))

    def test_there_is_something_to_compare(self) -> None:
        """A model with no trainable parameters would pass everything above."""

        model = _build()
        assert len(_names(model)) > 0
        assert len(model.parameter_layout.slots) == len(_names(model))


class TestTheComparisonCanActuallyFail:
    """Positive controls. Determinism tests that cannot detect a difference
    are indistinguishable from tests of nothing."""

    def test_a_different_width_changes_both_identities(self) -> None:
        cfg = _control_cfg()
        cfg.model.embedding.out_channels = 16
        cfg.model.layers[0].mixing.channels = 16
        cfg.model.layers[0].path_aggregation.channels = 16
        cfg.model.readout.channels = 16
        narrow = _build(cfg)
        wide = _build()

        assert _layout_signature(narrow) != _layout_signature(wide)

    def test_renaming_is_invisible_to_the_layout_but_not_to_the_state_dict(self) -> None:
        """The two identities are genuinely independent, demonstrated not asserted.

        A parameter moved to a differently named attribute of the same shape
        leaves the LAYOUT untouched -- slots carry no name -- while the
        state_dict key changes. This is why the contract says "order OR names"
        and why both are checked.
        """

        model = _build()
        before_layout = _layout_signature(model)
        before_keys = set(model.state_dict().keys())

        victim = next(n for n, p in model.named_parameters() if p.requires_grad)
        owner_path, _, leaf = victim.rpartition(".")
        owner = model.get_submodule(owner_path) if owner_path else model
        parameter = getattr(owner, leaf)
        delattr(owner, leaf)
        setattr(owner, f"{leaf}_renamed", parameter)

        assert _layout_signature(model) == before_layout, "layout should be name-blind"
        assert set(model.state_dict().keys()) != before_keys, "state dict should notice"


class TestTheA8ArmsShareOneLayout:
    """Measured property with a consequence, not just a fact.

    None of the three A8 arms contributes a trainable parameter:
    ``ResidualUpdater`` holds a float ``step``, ``ReplaceUpdater`` holds
    nothing, and ``GaussianCoordinateEnvelope`` wraps a parameterless
    ``GaussianDecayGate``. So switching arms perturbs neither the layout nor the
    state-dict keys.
    """

    @pytest.mark.parametrize("arm", A8_ARMS)
    def test_each_arm_builds(self, arm: str) -> None:
        assert len(_names(_with_arm(arm))) > 0

    def test_all_three_arms_share_one_parameter_layout(self) -> None:
        signatures = {arm: _layout_signature(_with_arm(arm)) for arm in A8_ARMS}
        assert signatures["residual"] == signatures["replacement"] == signatures["gaussian"]

    def test_all_three_arms_share_one_state_dict_key_set(self) -> None:
        keys = {arm: set(_with_arm(arm).state_dict().keys()) for arm in A8_ARMS}
        assert keys["residual"] == keys["replacement"] == keys["gaussian"]

    def test_no_a8_arm_contributes_a_trainable_parameter(self) -> None:
        """The reason for the two properties above, pinned directly."""

        assert list(ResidualUpdater().parameters()) == []
        assert list(ReplaceUpdater().parameters()) == []
        assert list(GaussianCoordinateEnvelope(sigma=1.0).parameters()) == []

    def test_the_arms_are_nonetheless_behaviourally_distinct(self) -> None:
        """Sharing a layout must not be mistaken for being the same model.

        Without this, "all three arms agree" reads as evidence the arms are
        interchangeable. They are not -- they differ in output, and that is the
        whole point of the coordinate.
        """

        from tpen.data.batch import ElectronBatch

        generator = torch.Generator().manual_seed(0)
        batch = ElectronBatch(
            positions=torch.randn(4, 2, 3, generator=generator, dtype=torch.float64),
            nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
            spins=torch.tensor([[1, -1]] * 4, dtype=torch.float64),
        )

        base = _build()
        state = base.state_dict()
        outputs = {}
        for arm in A8_ARMS:
            model = _with_arm(arm)
            # Same weights across arms, so any difference is the arm itself.
            model.load_state_dict(state)
            outputs[arm] = model(batch).logabs.detach().clone()

        assert not torch.allclose(outputs["residual"], outputs["replacement"])
        assert not torch.allclose(outputs["residual"], outputs["gaussian"])
        assert not torch.allclose(outputs["replacement"], outputs["gaussian"])


class TestTheSharedLayoutIsAlsoAHazard:
    """A restore cannot tell the arms apart, and that is worth stating loudly.

    Because all three arms share one state-dict key set, a ``strict=True``
    model-only restore of a checkpoint trained under one arm into a model built
    for another SUCCEEDS SILENTLY. Nothing in the state dict records which A8
    arm produced it.

    The study varies A8 across arms, so a mis-restored checkpoint is silently
    wrong science rather than a crash. The remedy is not in this slice -- it is
    checkpoint metadata recording the arm, which belongs to CD12/lane L5 and is
    tracked as an open reference-firewall-adjacent surface there. This test
    exists to make the hazard impossible to discover by accident later.
    """

    def test_a_cross_arm_strict_restore_succeeds_silently(self) -> None:
        trained_under_residual = _with_arm("residual").state_dict()
        built_for_gaussian = _with_arm("gaussian")

        # No exception, no warning, no signal of any kind.
        result = built_for_gaussian.load_state_dict(trained_under_residual, strict=True)
        assert not result.missing_keys
        assert not result.unexpected_keys
