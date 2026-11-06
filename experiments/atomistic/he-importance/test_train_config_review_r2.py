"""Pre-review round 2 controls for the AB-4b resolver (PR #573, head 22394bae).

Covers the round-1 fixes directly: the value-type refusal (F3) names the
coordinate and fires before any config composition; the A/B-shaped context
refusals (F4) name the problem and never borrow the O-literal text; and no
O-raw/O-composite context is routed into the A/B branch.  Modules are loaded
by path under review-unique ``sys.modules`` names, as in ``test_train_config.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import pytest


def _load(name: str, filename: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load("he_importance_stage_coordinate_for_ab4b_review_r2", "stage_coordinate.py")
optimizer_grid = _load("he_importance_optimizer_grid_for_ab4b_review_r2", "optimizer_grid.py")
train_config = _load("he_importance_train_config_for_ab4b_review_r2", "train_config.py")
ab_universe = _load("he_importance_ab_universe_for_ab4b_review_r2", "ab_universe.py")


def _ab_cell(tmp_path: Path, package: Mapping[str, Any], axis: str) -> Any:
    """Materialize one rendered package from its own fixed-Adam slot."""

    adam = next(slot for slot in package["optimizer_slots"] if slot["method"] == "adam")
    return stage_coordinate.materialize_stage(
        axis,
        [
            {
                "scientific_identity": {
                    "context": {"axis": axis, "literal_configuration": dict(package["literal_configuration"])},
                    "optimizer": {"method": adam["method"], "levels": dict(adam["levels"])},
                },
                "payload": {"updates": stage_coordinate.stage_definition(axis).updates},
                "topology": {},
            }
        ],
        stage_coordinate.OptimizerCell(adam["method"], adam["status"]),
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


def _context(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["scientific_identity"]["context"]


# T8 (F3) -------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [32, None, True, 0.5, ["32", "32"], {"c1": 32, "c2": 32}],
    ids=["int", "none", "bool", "float", "list", "mapping"],
)
def test_non_str_coordinate_is_refused_naming_it_before_composition(
    tmp_path: Path, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every non-str coordinate value is a ContextResolutionError that names the
    coordinate and is raised before ``_compose_base_config`` (so before AB-3)."""

    package = ab_universe.render_a_packages()[0]
    cell = _ab_cell(tmp_path, package, "A")
    rebuilt = _rebuilt(cell, lambda manifest: _context(manifest)["literal_configuration"].__setitem__("channels", value))
    monkeypatch.setattr(
        train_config, "_compose_base_config", lambda: pytest.fail("non-str coordinate reached config composition")
    )
    with pytest.raises(train_config.ContextResolutionError, match=r"coordinate 'channels' must be an exact str label"):
        train_config.resolve_train_config(rebuilt)


# T9 (F4) -------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate, pattern",
    [
        (lambda ctx: ctx.__setitem__("unconsumed_coordinate", "x"),
         r"A/B shape must have exactly the keys \['axis', 'literal_configuration'\]; "
         r"declared keys=\['axis', 'literal_configuration', 'unconsumed_coordinate'\]"),
        (lambda ctx: ctx.pop("axis"),
         r"A/B shape must have exactly the keys \['axis', 'literal_configuration'\]; "
         r"declared keys=\['literal_configuration'\]"),
        (lambda ctx: ctx.__setitem__("axis", "a"), r"A/B shape has unknown axis 'a'"),
        (lambda ctx: ctx.__setitem__("axis", "AB"), r"A/B shape has unknown axis 'AB'"),
        (lambda ctx: ctx.__setitem__("axis", None), r"A/B shape has unknown axis None"),
        (lambda ctx: ctx.__setitem__("literal_configuration", "(32,32)"),
         r"requires literal_configuration to be a mapping, got str"),
    ],
    ids=["extra-context-key", "axis-missing", "axis-lowercase", "axis-AB", "axis-none", "literal-not-mapping"],
)
def test_ab_shaped_context_refusals_name_the_problem(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], Any], pattern: str
) -> None:
    """An A/B-shaped context that fails the outer shape is refused with an A/B
    message naming the problem, never with the O-literal text."""

    package = ab_universe.render_a_packages()[0]
    cell = _ab_cell(tmp_path, package, "A")
    rebuilt = _rebuilt(cell, lambda manifest: mutate(_context(manifest)))
    with pytest.raises(train_config.ContextResolutionError, match=pattern) as info:
        train_config.resolve_train_config(rebuilt)
    assert "O-raw or O-composite" not in str(info.value)


def test_near_miss_refusal_message_names_the_label(tmp_path: Path) -> None:
    """The wrap message itself (not only its chained cause) names the bad label."""

    package = ab_universe.render_a_packages()[0]
    cell = _ab_cell(tmp_path, package, "A")
    rebuilt = _rebuilt(
        cell, lambda manifest: _context(manifest)["literal_configuration"].__setitem__("activation", "SiLU,SiLU ")
    )
    with pytest.raises(
        train_config.ContextResolutionError,
        match=r"unknown A/B coordinate label: unknown activation context label: 'SiLU,SiLU '",
    ):
        train_config.resolve_train_config(rebuilt)


# T10 (F4 routing boundary) -------------------------------------------------
def test_o_contexts_are_not_routed_into_the_ab_branch(tmp_path: Path) -> None:
    """Intended O rows resolve; a drifted O-composite keeps the O-literal refusal
    text; only a context that carries an A/B key reaches the A/B refusal."""

    cells = stage_coordinate.materialize_intended_configurations(
        tmp_path / "materialized", availability=optimizer_grid.roster_availability(), stages=("Q",)
    )
    adam_cells = [c for c in cells if c.manifest["scientific_identity"]["optimizer_cell"]["status"] == "available"]
    raw = next(c for c in adam_cells if c.manifest["scientific_identity"]["context"]["name"] == "O-raw")
    composite = next(c for c in adam_cells if c.manifest["scientific_identity"]["context"]["name"] == "O-composite")
    assert train_config.resolve_train_config(raw).model is not None
    assert train_config.resolve_train_config(composite).model is not None

    drifted = _rebuilt(composite, lambda manifest: _context(manifest)["overrides"].__setitem__("input_features", "B10"))
    with pytest.raises(
        train_config.ContextResolutionError, match="exactly the authority's O-raw or O-composite literal"
    ) as info:
        train_config.resolve_train_config(drifted)
    assert "A/B shape" not in str(info.value)

    hybrid = _rebuilt(composite, lambda manifest: _context(manifest).__setitem__("axis", "A"))
    with pytest.raises(train_config.ContextResolutionError, match=r"A/B shape must have exactly the keys"):
        train_config.resolve_train_config(hybrid)
