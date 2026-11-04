"""Pre-review round 1 controls for the AB-4b resolver (PR #573, item 006033a4).

Modules are loaded by path under review-unique ``sys.modules`` names, like
``test_train_config.py``.  Exceptions raised *through* the resolver are
compared against the module objects the resolver itself imports
(``train_config._context_mapping_api()``), never against a second by-path
copy, because separately loaded modules have distinct exception classes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import types
from typing import Any

import pytest
from omegaconf import OmegaConf


def _load(name: str, filename: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load("he_importance_stage_coordinate_for_ab4b_review_r1", "stage_coordinate.py")
optimizer_grid = _load("he_importance_optimizer_grid_for_ab4b_review_r1", "optimizer_grid.py")
train_config = _load("he_importance_train_config_for_ab4b_review_r1", "train_config.py")
ab_universe = _load("he_importance_ab_universe_for_ab4b_review_r1", "ab_universe.py")

_PRECHANGE_PIN = "040a5b8e87befc59bb21a9c64f8ec1782c8fa283"
_A_COORDINATES = tuple(sorted(ab_universe.render_a_packages()[0]["literal_configuration"]))


def _slot(package: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    """Return the rendered package's optimizer slot for ``method`` (``adam`` or ``O*``)."""

    return next(slot for slot in package["optimizer_slots"] if slot["method"] == method)


def _ab_cell(
    tmp_path: Path,
    package: Mapping[str, Any],
    axis: str,
    *,
    optimizer: Mapping[str, Any] | None = None,
    cell: Any = None,
) -> Any:
    """Materialize one rendered package from its OWN fixed-Adam slot.

    Unlike the PR helper, nothing is retyped: the identity's optimizer levels
    and the optimizer cell both come from the package's rendered slot.
    """

    adam = _slot(package, "adam")
    identity_optimizer = (
        dict(optimizer) if optimizer is not None else {"method": adam["method"], "levels": dict(adam["levels"])}
    )
    optimizer_cell = cell if cell is not None else stage_coordinate.OptimizerCell(adam["method"], adam["status"])
    return stage_coordinate.materialize_stage(
        axis,
        [
            {
                "scientific_identity": {
                    "context": {
                        "axis": axis,
                        "literal_configuration": dict(package["literal_configuration"]),
                    },
                    "optimizer": identity_optimizer,
                },
                "payload": {"updates": stage_coordinate.stage_definition(axis).updates},
                "topology": {},
            }
        ],
        optimizer_cell,
        tmp_path,
    )[0]


def _rebuilt(cell: Any, mutate: Callable[[dict[str, Any]], Any]) -> Any:
    """Return ``cell`` with ``mutate`` applied to a thawed manifest and the hash rebound."""

    manifest = stage_coordinate._thaw(cell.manifest)
    mutate(manifest)
    digest = stage_coordinate.content_hash(manifest)
    return replace(
        cell,
        manifest=stage_coordinate._freeze(manifest),
        content_hash=digest,
        output_path=cell.output_path.parent / digest,
    )


def _literal(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["scientific_identity"]["context"]["literal_configuration"]


# T1 ------------------------------------------------------------------------
@pytest.mark.parametrize("axis", ("A", "B"))
def test_resolver_forwards_exactly_the_package_labels_to_ab3(
    tmp_path: Path, axis: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clause 2: the resolved model is AB-3's composition for the package's own labels.

    A spy on the composition entry point the resolver actually imports records
    Every package must forward exactly its literal configuration (plus
    ``package`` and the model-initialization seed) on the committed base model.
    """

    composer = train_config._context_mapping_api()
    real = composer.compose_architecture_package
    calls: list[tuple[Any, Any, dict[str, Any]]] = []

    def spy(model_cfg: Any, *, package: str | None = None, **kwargs: Any) -> dict[str, Any]:
        snapshot = OmegaConf.to_container(model_cfg, resolve=False) if OmegaConf.is_config(model_cfg) else model_cfg
        calls.append((snapshot, package, dict(kwargs)))
        return real(model_cfg, package=package, **kwargs)

    monkeypatch.setattr(composer, "compose_architecture_package", spy)
    assert not hasattr(train_config, "_AB_CONTROL")
    base_model = OmegaConf.to_container(train_config._compose_base_config().model, resolve=False)

    packages = ab_universe.render_ab_universe()[axis]
    assert len(packages) == {"A": 254, "B": 60}[axis]
    for index, package in enumerate(packages):
        cell = _ab_cell(tmp_path / axis / str(index), package, axis)
        before = len(calls)
        resolved = train_config.resolve_train_config(cell)
        assert len(calls) == before + 1, "resolver must call the AB-3 entry point exactly once"
        model_arg, package_arg, kwargs = calls[-1]
        literal = dict(package["literal_configuration"])
        expected_package = literal.pop("package", None)
        assert (axis == "B") == (expected_package is not None)
        assert package_arg == expected_package
        assert kwargs == {**literal, "seed": cell.seed_streams["model_initialization"]}
        assert model_arg == base_model
        assert resolved.model is not None


# T2 ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "axis, mutate, pattern",
    [
        ("A", lambda lit: lit.__setitem__("package", "B01"),
         r"for A must be closed: extra coordinate\(s\) \['package'\]$"),
        ("B", lambda lit: lit.pop("package"),
         r"for B must be closed: missing coordinate\(s\) \['package'\]$"),
        ("A", lambda lit: lit.pop("activation"),
         r"for A must be closed: missing coordinate\(s\) \['activation'\]$"),
        ("B", lambda lit: lit.__setitem__("unconsumed", "x"),
         r"for B must be closed: extra coordinate\(s\) \['unconsumed'\]$"),
        ("A", lambda lit: (lit.__setitem__("channel", lit.pop("channels"))),
         r"missing coordinate\(s\) \['channels'\]; extra coordinate\(s\) \['channel'\]$"),
    ],
    ids=["a-with-package", "b-without-package", "missing-activation", "extra-key", "misspelled-key"],
)
def test_closed_literal_mutants_are_refused_naming_the_coordinate(
    tmp_path: Path, axis: str, mutate: Callable[[dict[str, Any]], Any], pattern: str
) -> None:
    """Clause 3: the literal set is closed on both axes and the refusal names the coordinate."""

    package = ab_universe.render_ab_universe()[axis][0]
    cell = _ab_cell(tmp_path, package, axis)
    rebuilt = _rebuilt(cell, lambda manifest: mutate(_literal(manifest)))
    with pytest.raises(train_config.ContextResolutionError, match=pattern):
        train_config.resolve_train_config(rebuilt)


# T3 ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "axis, coordinate",
    [("A", coordinate) for coordinate in _A_COORDINATES] + [("B", "package")],
)
def test_near_miss_in_every_coordinate_is_wrapped_with_its_cause(
    tmp_path: Path, axis: str, coordinate: str
) -> None:
    """Clause 3: a trailing-space near miss in ANY coordinate is a ContextResolutionError
    whose chained cause is AB-3's UnknownContextLabel naming the bad label."""

    composer = train_config._context_mapping_api()
    package = ab_universe.render_ab_universe()[axis][0]
    cell = _ab_cell(tmp_path, package, axis)
    near_miss = package["literal_configuration"][coordinate] + " "
    rebuilt = _rebuilt(cell, lambda manifest: _literal(manifest).__setitem__(coordinate, near_miss))
    with pytest.raises(train_config.ContextResolutionError) as info:
        train_config.resolve_train_config(rebuilt)
    assert isinstance(info.value.__cause__, composer.UnknownContextLabel)
    assert repr(near_miss) in str(info.value.__cause__)


# T4 (expected RED at eeb302f2; evidence for the value-shape finding) -------
def test_mapping_valued_coordinate_is_refused_as_context_resolution_error(tmp_path: Path) -> None:
    """Clause 3 discipline over value shape, not only key set.

    A mapping-valued coordinate passes the manifest firewall and the closed-key
    check, then reaches AB-3 where an unhashable label raises TypeError, which
    the resolver's narrow wrap does not convert.  Expected to FAIL at the
    reviewed head with ``TypeError: unhashable type: 'dict'``.
    """

    package = ab_universe.render_a_packages()[0]
    cell = _ab_cell(tmp_path, package, "A")
    rebuilt = _rebuilt(cell, lambda manifest: _literal(manifest).__setitem__("channels", {"c1": 32, "c2": 32}))
    with pytest.raises(train_config.ContextResolutionError):
        train_config.resolve_train_config(rebuilt)


# T5 ------------------------------------------------------------------------
def test_unrelated_composition_error_is_not_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The near-miss wrap converts ONLY UnknownContextLabel; other AB-3 errors pass through unchanged."""

    composer = train_config._context_mapping_api()
    package = ab_universe.render_a_packages()[0]
    cell = _ab_cell(tmp_path, package, "A")
    real_compose = train_config._compose_base_config

    def truncated_factors() -> Any:
        cfg = real_compose()
        cfg.model.factors = [OmegaConf.to_container(cfg.model.factors[0], resolve=False)]
        return cfg

    monkeypatch.setattr(train_config, "_compose_base_config", truncated_factors)
    with pytest.raises(composer.ContextMappingError) as info:
        train_config.resolve_train_config(cell)
    assert not isinstance(info.value, composer.UnknownContextLabel)
    assert not isinstance(info.value, train_config.TrainConfigResolutionError)
    assert "e-e and e-n factors" in str(info.value)


# T6 ------------------------------------------------------------------------
@pytest.mark.parametrize("axis", ("A", "B"))
def test_o_star_slot_from_the_rendered_package_is_refused_before_composition(
    tmp_path: Path, axis: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clause 4 with the renderer's own O* slot shape (``levels: None``), not a retyped row."""

    package = ab_universe.render_ab_universe()[axis][0]
    o_star = _slot(package, "O*")
    assert o_star["status"] == "unavailable" and o_star["levels"] is None
    pending = _ab_cell(
        tmp_path,
        package,
        axis,
        optimizer={"method": o_star["method"], "levels": None},
        cell=stage_coordinate.OptimizerCell(o_star["method"], o_star["status"], o_star["unavailable_reason"]),
    )
    monkeypatch.setattr(
        train_config, "_compose_base_config", lambda: pytest.fail("O* row reached config composition")
    )
    with pytest.raises(train_config.UnavailableOptimizerError, match=re.escape(o_star["unavailable_reason"])):
        train_config.resolve_train_config(pending)


@pytest.mark.parametrize("axis", ("A", "B"))
def test_fixed_adam_slot_from_the_rendered_package_resolves_with_its_levels(tmp_path: Path, axis: str) -> None:
    """Clause 2/4 control: the package's rendered fixed-Adam slot is what resolves, level by level."""

    package = ab_universe.render_ab_universe()[axis][0]
    levels = _slot(package, "adam")["levels"]
    resolved = train_config.resolve_train_config(_ab_cell(tmp_path, package, axis))
    assert resolved.optimizer._target_ == "torch.optim.Adam"
    assert resolved.optimizer.lr == levels["lr"]
    assert list(resolved.optimizer.betas) == [levels["beta1"], levels["beta2"]]
    assert resolved.optimizer.eps == levels["epsilon"]
    assert resolved.optimizer.weight_decay == levels["weight_decay"]


# T7 ------------------------------------------------------------------------
def _load_prechange_resolver() -> tuple[types.ModuleType, str]:
    source = subprocess.run(
        ["git", "show", f"{_PRECHANGE_PIN}:experiments/atomistic/he-importance/train_config.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    module = types.ModuleType("he_importance_train_config_prechange_for_ab4b_review_r1")
    module.__file__ = train_config.__file__
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, source


def _assert_matches_prechange(cells: Any, resolver: Any, prechange: types.ModuleType) -> None:
    """Verbatim semantics of the PR's oracle comparison (copied so the control is self-contained)."""

    for cell in cells:
        try:
            expected = prechange.resolve_train_config(cell)
        except Exception as expected_error:  # noqa: BLE001
            try:
                resolver(cell)
            except Exception as observed_error:  # noqa: BLE001
                assert type(observed_error).__name__ == type(expected_error).__name__
                assert str(observed_error) == str(expected_error)
            else:
                raise AssertionError("current resolver accepted a prechange-refused cell")
        else:
            observed = resolver(cell)
            assert OmegaConf.to_container(observed, resolve=True) == OmegaConf.to_container(expected, resolve=True)


def test_prechange_oracle_refusal_arm_is_a_live_control(tmp_path: Path) -> None:
    """Clause 1: the oracle detects a changed refusal text AND a lifted refusal on a real Q row."""

    prechange, source = _load_prechange_resolver()
    assert "_AB_CONTEXT_KEYS" not in source, "oracle must be the pre-change source, not the current file"
    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path / "materialized", availability=optimizer_grid.roster_availability(), stages=("Q",)
    )
    refused = [c for c in cells if c.manifest["scientific_identity"]["optimizer_cell"]["status"] != "available"]
    accepted = [c for c in cells if c.manifest["scientific_identity"]["optimizer_cell"]["status"] == "available"]
    assert refused and accepted
    _assert_matches_prechange(refused[:1], train_config.resolve_train_config, prechange)  # sanity: green

    def changed_text(source_cell: Any) -> Any:
        try:
            return train_config.resolve_train_config(source_cell)
        except train_config.UnavailableOptimizerError as error:
            raise train_config.UnavailableOptimizerError(str(error) + " (mutant)") from None

    with pytest.raises(AssertionError):
        _assert_matches_prechange(refused[:1], changed_text, prechange)

    def lifted(source_cell: Any) -> Any:
        return train_config.resolve_train_config(accepted[0])

    with pytest.raises(AssertionError, match="accepted a prechange-refused cell"):
        _assert_matches_prechange(refused[:1], lifted, prechange)
