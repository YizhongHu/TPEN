"""Initialization comes from NAMED STREAMS, not from global-RNG draw order.

The literal control protocol requires embedding and channel-MLP affine layers to
be seeded from named module streams, and path-aggregation matrices to use
named-stream Xavier uniform. It states the falsifier directly:

    No construction-order global-RNG accident is a seed definition.

That is a precise claim and it has a precise test. Under global RNG, a module's
initial values depend on how many draws were consumed BEFORE it was built, so
widening an unrelated module silently re-initializes everything constructed
after it. Under named streams, each module draws from a generator seeded by
``blake2b(seed:stream)`` and is invariant to everything else.

Why both arms are measured
--------------------------
The invariance test alone cannot tell "named streams work" from "this model
happens not to be order-sensitive". So every invariance case is paired with the
LEGACY build -- the same config with the initializers removed -- which
``PathAggregation``'s own docstring says falls back to "the legacy PyTorch
global-RNG Xavier initializer". The legacy arm must SHIFT where the seeded arm
holds. Without that pairing the suite would pass just as well against a model
with no randomness at all.

This is the same bidirectional discipline used for the config schema: prove the
rule closes, and prove the check can detect it not closing.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from hydra.utils import instantiate
from omegaconf import OmegaConf

CONTROL_CONFIG = Path("experiments/atomistic/he-importance/configs/train.yaml")


def _cfg():
    cfg = OmegaConf.load(CONTROL_CONFIG)
    OmegaConf.resolve(cfg)
    return cfg


def _strip_initializers(cfg):
    """Return the config with every named stream removed: the legacy arm."""

    cfg = copy.deepcopy(cfg)
    cfg.model.embedding.pop("initializer", None)
    cfg.model.layers[0].path_aggregation.pop("initializer", None)
    return cfg


def _build(cfg):
    return instantiate(cfg.model).to(getattr(torch, str(cfg.runtime.dtype)))


def _values(model, prefix: str) -> list[torch.Tensor]:
    """Snapshot the initial values of every parameter under ``prefix``."""

    return [p.detach().clone() for n, p in model.named_parameters() if n.startswith(prefix)]


def _same(left, right) -> bool:
    return len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right))


def _widen_embedding(cfg):
    """Change ONLY the embedding, which is unrelated to path aggregation."""

    cfg = copy.deepcopy(cfg)
    cfg.model.embedding.hidden_channels = 256
    return cfg


PATH_AGG = "stack.layers.0.path_aggregation"


class TestTheControlConfigDeclaresItsStreams:
    def test_both_seeded_surfaces_declare_an_initializer(self) -> None:
        """A missing initializer is a SILENT fallback to global RNG.

        PathAggregation documents the fallback at its own ``initializer``
        argument. Nothing warns, so absence has to be asserted here.
        """

        cfg = _cfg()
        for path in ("model.embedding.initializer", "model.layers[0].path_aggregation.initializer"):
            block = OmegaConf.select(cfg, path)
            assert block is not None, f"{path} is absent; that silently means global RNG"
            assert block._target_ == "tpen.nn.initialization.TorchInitializer"
            assert block.seed == cfg.runtime.seed

    def test_the_streams_are_distinct(self) -> None:
        """Two surfaces sharing one stream name would draw identical sequences."""

        cfg = _cfg()
        streams = {
            str(OmegaConf.select(cfg, "model.embedding.initializer.stream")),
            str(OmegaConf.select(cfg, "model.layers[0].path_aggregation.initializer.stream")),
        }
        assert len(streams) == 2


class TestReproducibility:
    def test_two_builds_produce_identical_values(self) -> None:
        cfg = _cfg()
        assert _same(_values(_build(cfg), PATH_AGG), _values(_build(cfg), PATH_AGG))
        assert _same(_values(_build(cfg), "embedding"), _values(_build(cfg), "embedding"))

    def test_a_different_seed_produces_different_values(self) -> None:
        """Positive control: identical-values tests are vacuous without it."""

        cfg = _cfg()
        other = copy.deepcopy(cfg)
        other.model.embedding.initializer.seed = 1
        other.model.layers[0].path_aggregation.initializer.seed = 1

        assert not _same(_values(_build(cfg), PATH_AGG), _values(_build(other), PATH_AGG))
        assert not _same(_values(_build(cfg), "embedding"), _values(_build(other), "embedding"))

    def test_there_are_parameters_under_both_prefixes(self) -> None:
        """Empty lists compare equal, so the comparisons need something in them."""

        model = _build(_cfg())
        assert _values(model, PATH_AGG)
        assert _values(model, "embedding")


class TestTheAuthoritysFalsifier:
    """No construction-order global-RNG accident is a seed definition."""

    def test_widening_the_embedding_does_not_disturb_path_aggregation(self) -> None:
        cfg = _cfg()
        baseline = _values(_build(cfg), PATH_AGG)
        widened = _values(_build(_widen_embedding(cfg)), PATH_AGG)
        assert _same(baseline, widened), (
            "path-aggregation initial values moved when an unrelated module changed "
            "width; initialization is following global-RNG draw order"
        )

    def test_the_legacy_arm_DOES_get_disturbed(self) -> None:
        """The paired arm. Without it the test above proves nothing.

        Same config, initializers removed, so both surfaces fall back to the
        legacy global-RNG initializer. Widening the embedding consumes a
        different number of draws before path aggregation is built, so its
        weights move. This is the failure the named streams prevent, shown
        actually happening rather than asserted.
        """

        legacy = _strip_initializers(_cfg())
        torch.manual_seed(0)
        baseline = _values(_build(legacy), PATH_AGG)
        torch.manual_seed(0)
        widened = _values(_build(_widen_embedding(legacy)), PATH_AGG)
        assert not _same(baseline, widened), (
            "the legacy arm was NOT order-sensitive, so the invariance test above "
            "cannot distinguish named streams from a model with no randomness"
        )


class TestGlobalRngIsNotMutated:
    def test_building_the_control_model_leaves_global_rng_untouched(self) -> None:
        """A build that advances global RNG makes any later draw order-dependent.

        This is the property that lets a sampler seeded from the same base seed
        be reproducible regardless of what the model construction did.
        """

        cfg = _cfg()
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        _build(cfg)
        assert torch.equal(before, torch.get_rng_state())

    def test_the_legacy_arm_DOES_mutate_global_rng(self) -> None:
        """Paired arm again: proves the check above can detect mutation."""

        legacy = _strip_initializers(_cfg())
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        _build(legacy)
        assert not torch.equal(before, torch.get_rng_state())
