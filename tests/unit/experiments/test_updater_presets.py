"""Static contracts for the composable updater presets.

The acceptance these tests exist for is that a preset "resolves to explicit
objects without registries/factories/string dispatch".  That is checked two
ways, deliberately: structurally, that every configured block names a
``_target_`` rather than a lookup key; and behaviourally, that instantiating
the preset yields the exact classes, so a preset cannot pass by naming
something plausible that resolves to nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.training.optim import make_optimizer, make_update_method
from tpen.training.block_ng import BlockDiagonalNaturalGradientUpdate, BlockNGPolicy
from tpen.training.qgt import DampingPolicy
from tpen.training.score_geometry import ScoreConventions
from tpen.training.sr import SRPolicy, StochasticReconfigurationUpdate
from tpen.training.update import LegacyAutogradUpdate, ModelParameterBinding

ROOT = Path(__file__).resolve().parents[3]
PRESETS = ROOT / "experiments" / "configs" / "updater"
LEGACY = PRESETS / "legacy_adam.yaml"
SR_DENSE = PRESETS / "sr_dense.yaml"
MINSR = PRESETS / "minsr.yaml"
BLOCK_NG = PRESETS / "block_ng.yaml"

ALL_PRESETS = (LEGACY, SR_DENSE, MINSR, BLOCK_NG)
SR_PRESETS = (SR_DENSE, MINSR)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict), f"{path.name} must be a mapping"
    return value


def _parameters() -> tuple[torch.nn.Parameter, ...]:
    """Return a small live parameter domain standing in for a model."""

    return (torch.nn.Parameter(torch.zeros(3, dtype=torch.float64)),)


def _build(path: Path):
    """Instantiate a preset into a live (optimizer, update_method) pair."""

    cfg = OmegaConf.create(_load(path))
    parameters = _parameters()
    optimizer = make_optimizer(cfg.optimizer, parameters)
    method = make_update_method(
        OmegaConf.select(cfg, "trainer.update_method"),
        optimizer=optimizer,
        model_parameters=ModelParameterBinding(parameters=parameters),
    )
    return optimizer, method


@pytest.mark.parametrize("path", ALL_PRESETS, ids=lambda p: p.stem)
def test_every_preset_supplies_exactly_the_two_expected_blocks(path: Path) -> None:
    """A preset is a drop-in pair, so it must not carry unrelated config roots.

    In particular it must not carry `callbacks` or `loggers`: those are owned
    at the config root by `RunContext`, and a preset that shipped its own would
    silently take over that ownership when merged.
    """

    preset = _load(path)

    assert set(preset) == {"optimizer", "trainer"}
    assert set(preset["trainer"]) == {"update_method"}
    assert "callbacks" not in preset
    assert "loggers" not in preset


def test_legacy_preset_keeps_adam_and_selects_the_default_updater() -> None:
    """The legacy preset must remain the historical Adam behaviour."""

    preset = _load(LEGACY)
    assert preset["optimizer"]["_target_"] == "torch.optim.Adam"
    # `null` is the value that selects LegacyAutogradUpdate; it is not an
    # unset placeholder, and a preset that omitted the key entirely would not
    # override an SR block it was merged over.
    assert "update_method" in preset["trainer"]
    assert preset["trainer"]["update_method"] is None

    optimizer, method = _build(LEGACY)
    assert isinstance(optimizer, torch.optim.Adam)
    assert method is None, "legacy must resolve to no explicit method"


def test_legacy_remains_the_trainer_default_when_no_method_is_configured() -> None:
    """With no configured method the trainer still builds the legacy adapter."""

    from tpen.training.trainer import VMCTrainer

    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    trainer = VMCTrainer(max_steps=1)

    selected = trainer._select_update_method(
        model=model, optimizer=optimizer, update_method=None
    )

    assert isinstance(selected, LegacyAutogradUpdate)
    assert selected.forward_request() is None, "legacy must not request a score payload"


@pytest.mark.parametrize(
    ("path", "expected_space"),
    [(SR_DENSE, "parameter"), (MINSR, "sample")],
    ids=["sr_dense", "minsr"],
)
def test_sr_presets_resolve_to_exact_objects(path: Path, expected_space: str) -> None:
    """Each SR preset yields the concrete classes, not merely a valid mapping."""

    optimizer, method = _build(path)

    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(method, StochasticReconfigurationUpdate)
    assert isinstance(method.policy, SRPolicy)
    assert isinstance(method.policy.damping, DampingPolicy)
    assert isinstance(method.conventions, ScoreConventions)
    assert method.policy.solve_space == expected_space
    # The dtype is spelled as the bare name "float64" in YAML, matching how
    # runtime.dtype is spelled; it must arrive as a real torch dtype.
    assert method.conventions.solve_dtype == torch.float64
    # An SR method must ask for the score payload, or it would be handed an
    # autograd input it cannot consume.
    assert method.forward_request() is not None


def test_block_ng_preset_resolves_to_exact_objects() -> None:
    """Block-NG uses the same factory shape as SR without a name registry."""

    optimizer, method = _build(BLOCK_NG)

    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(method, BlockDiagonalNaturalGradientUpdate)
    assert isinstance(method.policy, BlockNGPolicy)
    assert method.policy.solve_dtype is torch.float64
    assert method.forward_request() is not None


def test_block_ng_float32_config_resolves_to_its_own_dtype_object() -> None:
    """A float32 configuration must not be silently coerced to float64."""

    cfg = OmegaConf.create(_load(BLOCK_NG))
    cfg.trainer.update_method.policy.solve_dtype = "float32"
    parameters = _parameters()
    optimizer = make_optimizer(cfg.optimizer, parameters)
    method = make_update_method(
        cfg.trainer.update_method,
        optimizer=optimizer,
        model_parameters=ModelParameterBinding(parameters=parameters),
    )

    assert isinstance(method, BlockDiagonalNaturalGradientUpdate)
    assert method.policy.solve_dtype is torch.float32
    assert method.policy.solve_dtype is not torch.float64


def test_block_ng_hydra_config_refuses_an_unknown_solve_dtype_name() -> None:
    """Hydra must not turn an unknown dtype spelling into a valid solve policy."""

    cfg = OmegaConf.create(_load(BLOCK_NG))
    cfg.trainer.update_method.policy.solve_dtype = "not_a_torch_dtype"
    parameters = _parameters()
    optimizer = make_optimizer(cfg.optimizer, parameters)

    with pytest.raises(InstantiationException, match="not a torch dtype name"):
        make_update_method(
            cfg.trainer.update_method,
            optimizer=optimizer,
            model_parameters=ModelParameterBinding(parameters=parameters),
        )


def test_block_ng_preset_names_targets_and_keeps_learning_rates_equal() -> None:
    """The preset must carry explicit construction and one agreed step size."""

    preset = _load(BLOCK_NG)
    method = preset["trainer"]["update_method"]

    assert method["_target_"] == "tpen.training.block_ng.BlockDiagonalNaturalGradientUpdate"
    assert method["_partial_"] is True
    assert method["policy"]["_target_"] == "tpen.training.block_ng.BlockNGPolicy"
    assert method["policy"]["learning_rate"] == preset["optimizer"]["lr"]


@pytest.mark.parametrize("path", SR_PRESETS, ids=lambda p: p.stem)
def test_sr_presets_name_targets_rather_than_lookup_keys(path: Path) -> None:
    """Structural check: every configured block is an explicit `_target_`.

    This is the "no registries, no factories, no string dispatch" acceptance
    read literally. A registry-based config would carry a bare name like
    ``method: sr`` somewhere; walking the tree proves no such key exists.
    """

    preset = _load(path)
    method = preset["trainer"]["update_method"]

    assert method["_target_"] == "tpen.training.sr.StochasticReconfigurationUpdate"
    assert method["_partial_"] is True, "the method needs the live optimizer, so it must be partial"
    assert method["policy"]["_target_"] == "tpen.training.sr.SRPolicy"
    assert method["policy"]["damping"]["_target_"] == "tpen.training.qgt.DampingPolicy"
    assert method["conventions"]["_target_"] == (
        "tpen.training.score_geometry.ScoreConventions"
    )

    # Every nested mapping in the updater block either names a target or is a
    # plain value bag belonging to one that does.
    def _walk(node: object, trail: str) -> None:
        if isinstance(node, dict):
            has_target = "_target_" in node
            assert has_target, f"{trail} is a mapping with no _target_"
            for key, value in node.items():
                if isinstance(value, dict):
                    _walk(value, f"{trail}.{key}")

    _walk(method, "trainer.update_method")


@pytest.mark.parametrize("path", SR_PRESETS, ids=lambda p: p.stem)
def test_sr_preset_learning_rate_agrees_between_the_two_blocks(path: Path) -> None:
    """The optimizer lr and the policy lr are one number spelled twice."""

    preset = _load(path)
    optimizer_lr = preset["optimizer"]["lr"]
    policy_lr = preset["trainer"]["update_method"]["policy"]["learning_rate"]

    assert optimizer_lr == policy_lr

    _, method = _build(path)
    assert method.policy.learning_rate == pytest.approx(optimizer_lr)


@pytest.mark.parametrize("path", SR_PRESETS, ids=lambda p: p.stem)
def test_sr_preset_rejects_a_desynchronized_learning_rate(path: Path) -> None:
    """Editing one lr and not the other fails loudly at construction.

    This is the failure a user is most likely to introduce by hand, and it is
    silent in the worst way if unchecked: the run proceeds and every reported
    step size is wrong.
    """

    cfg = OmegaConf.create(_load(path))
    cfg.optimizer.lr = float(cfg.optimizer.lr) * 2.0
    parameters = _parameters()
    optimizer = make_optimizer(cfg.optimizer, parameters)

    with pytest.raises(ValueError, match="disagrees with SRPolicy.learning_rate"):
        make_update_method(
            cfg.trainer.update_method,
            optimizer=optimizer,
            model_parameters=ModelParameterBinding(parameters=parameters),
        )


@pytest.mark.parametrize("path", SR_PRESETS, ids=lambda p: p.stem)
def test_pointing_an_sr_preset_at_adam_is_refused(path: Path) -> None:
    """Swapping in Adam under an SR preset raises instead of quietly training."""

    cfg = OmegaConf.create(_load(path))
    parameters = _parameters()
    adam = torch.optim.Adam(parameters, lr=float(cfg.optimizer.lr))

    with pytest.raises(TypeError, match="Adam"):
        make_update_method(
            cfg.trainer.update_method,
            optimizer=adam,
            model_parameters=ModelParameterBinding(parameters=parameters),
        )


def test_sr_and_minsr_presets_differ_only_in_the_solve_route() -> None:
    """The two SR presets must not drift apart on anything but the route.

    They are supposed to be the same algorithm reached two ways. If a future
    edit changed damping in one and not the other, the pair would stop being a
    cost choice and quietly become two different methods.
    """

    dense = _load(SR_DENSE)["trainer"]["update_method"]
    sample = _load(MINSR)["trainer"]["update_method"]

    assert dense["policy"].pop("solve_space") == "parameter"
    assert sample["policy"].pop("solve_space") == "sample"
    assert dense == sample
    assert _load(SR_DENSE)["optimizer"] == _load(MINSR)["optimizer"]


@pytest.mark.parametrize("path", ALL_PRESETS, ids=lambda p: p.stem)
def test_preset_optimizer_block_is_partial_and_omits_params(path: Path) -> None:
    """The optimizer is bound to the live model by the runner, not by YAML."""

    optimizer_cfg = _load(path)["optimizer"]

    assert optimizer_cfg["_partial_"] is True
    assert "params" not in optimizer_cfg


def not_an_update_method(optimizer, *, model_parameters):
    """A factory with the right signature that returns the wrong thing.

    Module-level so Hydra can name it with `_target_`. It accepts the exact
    arguments `make_update_method` passes, so the call SUCCEEDS and the type
    check is what rejects it -- a target whose call merely raised would test
    Hydra, not the guard.
    """

    del optimizer, model_parameters
    return object()


def test_make_update_method_rejects_a_factory_returning_the_wrong_type() -> None:
    """A `_target_` naming something that is not an update method fails loudly."""

    parameters = _parameters()
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    bogus = OmegaConf.create(
        {
            "_target_": (
                "tests.unit.experiments.test_updater_presets.not_an_update_method"
            ),
            "_partial_": True,
        }
    )

    with pytest.raises(TypeError, match="must return a VMCUpdateMethod"):
        make_update_method(
            bogus,
            optimizer=optimizer,
            model_parameters=ModelParameterBinding(parameters=parameters),
        )


def test_instantiating_a_preset_needs_no_params_or_optimizer_key() -> None:
    """Hydra alone cannot finish the job, which is why make_* helpers exist."""

    cfg = OmegaConf.create(_load(SR_DENSE))
    partial_method = instantiate(cfg.trainer.update_method)

    # Instantiation yields a factory, not a method: the optimizer and the live
    # parameter binding are still missing at this point.
    assert callable(partial_method)
    assert not isinstance(partial_method, StochasticReconfigurationUpdate)
