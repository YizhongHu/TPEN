"""The Stage-Q optimizer grid: declared cell enumeration and availability.

This module owns the executable transcription of consolidated-design
sections 2.7 (the 120-cell VMC-first optimizer grid) and 2.8 (the two
Stage-Q optimizer contexts, O-raw and O-composite). It reads the literal,
reviewable ``optimizer_grid_authority.json`` transcription, expands it into
the declared cell universe, and renders the committed
``intended_configurations.json`` Stage-Q inventory from that universe.

Two concerns are kept structurally separate, per the lane's design
decisions (Task Orchestrator item 993de93d, note
``lane-design-decisions-2026-10-09``):

- **Declaration** (:func:`declared_cells`) states which optimizer cells the
  design names. It takes no roster input and its cell identities never
  move.
- **Runnability** (:func:`roster_availability`, :func:`stage_q_report`) is a
  separate, build-time judgement derived from
  ``tpen.hi_schema.HI_METHOD_ROSTER``. A roster change can flip which cells
  are reported runnable; it can never add, remove, or rename a declared
  cell.

Only the standard library is imported here, plus a sibling study module
(``stage_coordinate``) loaded by file path. The one deliberate exception is
the lazy ``tpen.hi_schema`` import inside :func:`roster_availability`, which
mirrors the same checkout-boundary pattern already used by
``train_config.py``'s ``_optimizer_entry``: experiments code may not import
``tpen`` at module scope, but a caller that already has the checkout root on
``sys.path`` may ask this module to resolve the live roster for it.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any

_AUTHORITY_PATH = Path(__file__).with_name("optimizer_grid_authority.json")
_INTENDED_CONFIGURATIONS_PATH = Path(__file__).with_name("intended_configurations.json")
_STAGE_CODE = "Q"

_FAMILY_REQUIRED_KEYS = frozenset({"family", "varied", "fixed", "declared_cells"})
_FAMILY_OPTIONAL_KEYS = frozenset({"units"})
_EXPECTED_FAMILIES = ("adam", "sr", "kfac", "spring", "linear_method")
_CONTEXT_REQUIRED_KEYS = frozenset({"name", "overrides", "unspecified_coordinates"})
_SCALAR_TYPES = (str, int, float, bool)


class OptimizerGridError(ValueError):
    """The authority, its expansion, or a roster mapping is malformed."""


def _load_stage_coordinate() -> Any:
    """Load the sibling ``stage_coordinate`` module by file path.

    ``experiments/atomistic/he-importance`` is a hyphenated directory with no
    ``__init__.py``; a plain ``import stage_coordinate`` only works when this
    file's directory happens to be on ``sys.path``. Loading by explicit file
    location, the same technique this package's own test modules use, avoids
    depending on that.
    """

    name = "he_importance_stage_coordinate_for_optimizer_grid"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("stage_coordinate.py"))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive only
        raise OptimizerGridError("could not load the sibling stage_coordinate module")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: stage_coordinate's frozen dataclasses resolve
    # their own module via ``sys.modules``, which requires the entry to
    # already exist by the time the class body executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load_stage_coordinate()


@dataclass(frozen=True)
class GridCell:
    """One declared optimizer cell: a family plus its complete literal levels.

    Parameters
    ----------
    family
        Method name, spelled exactly as in ``HI_METHOD_ROSTER``.
    levels
        The complete literal constructor values for this cell: every varied
        factor's selected value merged with every fixed factor from the
        authority. Nothing here is an inherited library default.
    cell_id
        Availability-free content identity:
        ``stage_coordinate.content_hash({"family": family, "levels": levels})``.
    lexical_id
        A deterministic, human-readable identifier built from the family
        name and each level's sorted factor name and ``json.dumps`` value.
    """

    family: str
    levels: Mapping[str, Any]
    cell_id: str
    lexical_id: str


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OptimizerGridError(f"{label} must be a mapping")
    return value


def _require_scalar(value: Any, label: str) -> Any:
    """Require one of the literal JSON scalar types (``bool`` included).

    ``bool`` is listed explicitly in ``_SCALAR_TYPES`` rather than relying on
    its being an ``int`` subclass, because a fixed constructor value such as
    KFAC's ``separate_momentum: false`` must be accepted on its own terms.
    """

    if not isinstance(value, _SCALAR_TYPES):
        raise OptimizerGridError(f"{label} must be a literal scalar")
    return value


def _validate_level_list(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value:
        raise OptimizerGridError(f"{label} must be a non-empty list of literal levels")
    for index, item in enumerate(value):
        _require_scalar(item, f"{label}[{index}]")
    return tuple(value)


def _validate_fixed(value: Any, label: str) -> Mapping[str, Any]:
    fixed = _require_mapping(value, label)
    for key, nested in fixed.items():
        if not isinstance(key, str):
            raise OptimizerGridError(f"{label} keys must be strings")
        _require_scalar(nested, f"{label}.{key}")
    return fixed


def _validate_family(entry: Any) -> Mapping[str, Any]:
    mapping = _require_mapping(entry, "family entry")
    keys = frozenset(mapping)
    if not _FAMILY_REQUIRED_KEYS <= keys <= (_FAMILY_REQUIRED_KEYS | _FAMILY_OPTIONAL_KEYS):
        raise OptimizerGridError(
            f"family entry keys mismatch: required={sorted(_FAMILY_REQUIRED_KEYS)}, "
            f"optional={sorted(_FAMILY_OPTIONAL_KEYS)}, actual={sorted(keys)}"
        )
    family = mapping["family"]
    if not isinstance(family, str) or not family:
        raise OptimizerGridError("family name must be a non-empty string")
    varied = _require_mapping(mapping["varied"], f"{family}.varied")
    for factor, levels in varied.items():
        if not isinstance(factor, str):
            raise OptimizerGridError(f"{family}.varied factor names must be strings")
        _validate_level_list(levels, f"{family}.varied.{factor}")
    fixed = _validate_fixed(mapping["fixed"], f"{family}.fixed")
    if set(varied) & set(fixed):
        raise OptimizerGridError(f"{family} declares the same factor as both varied and fixed")
    declared_cells = mapping["declared_cells"]
    if isinstance(declared_cells, bool) or not isinstance(declared_cells, int) or declared_cells <= 0:
        raise OptimizerGridError(f"{family}.declared_cells must be a positive integer")
    units = mapping.get("units", {})
    units_mapping = _require_mapping(units, f"{family}.units")
    for key, value in units_mapping.items():
        if not isinstance(key, str) or key not in varied | fixed:
            raise OptimizerGridError(f"{family}.units names an undeclared factor {key!r}")
        if not isinstance(value, str) or not value:
            raise OptimizerGridError(f"{family}.units.{key} must be a non-empty string")
    return mapping


def _validate_context(entry: Any) -> Mapping[str, Any]:
    mapping = _require_mapping(entry, "context entry")
    if frozenset(mapping) != _CONTEXT_REQUIRED_KEYS:
        raise OptimizerGridError(
            f"context entry keys mismatch: expected={sorted(_CONTEXT_REQUIRED_KEYS)}, "
            f"actual={sorted(mapping)}"
        )
    name = mapping["name"]
    if not isinstance(name, str) or not name:
        raise OptimizerGridError("context name must be a non-empty string")
    overrides = _require_mapping(mapping["overrides"], f"context {name}.overrides")
    for key, value in overrides.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            raise OptimizerGridError(f"context {name}.overrides must map non-empty strings to non-empty strings")
    marker = mapping["unspecified_coordinates"]
    if not isinstance(marker, str) or "section-2.3" not in marker:
        raise OptimizerGridError(
            f"context {name}.unspecified_coordinates must name the section-2.3 control protocol"
        )
    return mapping


def load_authority() -> Mapping[str, Any]:
    """Load and strictly validate the literal section-2.7/2.8 transcription.

    Raises
    ------
    OptimizerGridError
        If the authority is missing a required top-level key, a family
        entry has the wrong shape, a family name is missing or repeated, or
        a context entry has the wrong shape.
    """

    raw = json.loads(_AUTHORITY_PATH.read_text(encoding="utf-8"))
    top_level = _require_mapping(raw, "optimizer grid authority")
    expected_top = frozenset({"source", "notes", "families", "contexts"})
    if frozenset(top_level) != expected_top:
        raise OptimizerGridError(
            f"authority top-level keys mismatch: expected={sorted(expected_top)}, actual={sorted(top_level)}"
        )
    source = _require_mapping(top_level["source"], "authority.source")
    if not isinstance(source.get("document"), str) or not source["document"]:
        raise OptimizerGridError("authority.source.document must be a non-empty string")
    if not isinstance(source.get("sections"), list) or not source["sections"]:
        raise OptimizerGridError("authority.source.sections must be a non-empty list")
    notes = top_level["notes"]
    if not isinstance(notes, list) or not notes or not all(isinstance(note, str) and note for note in notes):
        raise OptimizerGridError("authority.notes must be a non-empty list of non-empty strings")

    families_raw = top_level["families"]
    if not isinstance(families_raw, list) or not families_raw:
        raise OptimizerGridError("authority.families must be a non-empty list")
    families = tuple(_validate_family(entry) for entry in families_raw)
    family_names = tuple(entry["family"] for entry in families)
    if len(set(family_names)) != len(family_names):
        raise OptimizerGridError("authority.families repeats a family name")
    if frozenset(family_names) != frozenset(_EXPECTED_FAMILIES):
        raise OptimizerGridError(
            f"authority.families must declare exactly {sorted(_EXPECTED_FAMILIES)}, got {sorted(family_names)}"
        )

    contexts_raw = top_level["contexts"]
    if not isinstance(contexts_raw, list) or len(contexts_raw) != 2:
        raise OptimizerGridError("authority.contexts must declare exactly the two section-2.8 contexts")
    contexts = tuple(_validate_context(entry) for entry in contexts_raw)
    context_names = {entry["name"] for entry in contexts}
    if context_names != {"O-raw", "O-composite"}:
        raise OptimizerGridError(f"authority.contexts must be named O-raw and O-composite, got {sorted(context_names)}")

    return MappingProxyType(
        {
            "source": MappingProxyType(dict(source)),
            "notes": tuple(notes),
            "families": families,
            "contexts": contexts,
        }
    )


def _lexical_id(family: str, levels: Mapping[str, Any]) -> str:
    """Build a deterministic id from the family and each sorted factor value."""

    parts = [family]
    for factor in sorted(levels):
        parts.append(f"{factor}={json.dumps(levels[factor], sort_keys=True)}")
    return "__".join(parts)


def declared_cells() -> tuple[GridCell, ...]:
    """Return the full, declared, availability-free 120-cell expansion.

    Expansion order is deterministic: families in authority-file order, and
    within a family, the Cartesian product of its varied factors in
    authority-file order. Raises :class:`OptimizerGridError` if any family's
    expansion length disagrees with its declared ``declared_cells``, or if
    the total disagrees with the consolidated-design total of 120.
    """

    authority = load_authority()
    cells: list[GridCell] = []
    for family_entry in authority["families"]:
        family = family_entry["family"]
        varied = family_entry["varied"]
        fixed = family_entry["fixed"]
        factor_names = tuple(varied)
        value_lists = tuple(varied[name] for name in factor_names)
        expanded = 0
        for combination in product(*value_lists):
            levels = dict(fixed)
            levels.update(zip(factor_names, combination))
            frozen_levels = MappingProxyType(levels)
            cell_id = stage_coordinate.content_hash({"family": family, "levels": dict(levels)})
            cells.append(
                GridCell(
                    family=family,
                    levels=frozen_levels,
                    cell_id=cell_id,
                    lexical_id=_lexical_id(family, levels),
                )
            )
            expanded += 1
        if expanded != family_entry["declared_cells"]:
            raise OptimizerGridError(
                f"{family} expands to {expanded} cells, authority declares {family_entry['declared_cells']}"
            )
    if len(cells) != 120:
        raise OptimizerGridError(f"declared optimizer grid must total 120 cells, expanded {len(cells)}")
    return tuple(cells)


def roster_availability(roster: Any = None) -> dict[str, "stage_coordinate.OptimizerCell"]:
    """Resolve every declared family's build-time availability.

    Parameters
    ----------
    roster
        An iterable of objects with ``method``, ``admitted``, and
        ``requires`` attributes (the shape of
        ``tpen.hi_schema.MethodAvailability``). When ``None``, the live
        ``tpen.hi_schema.HI_METHOD_ROSTER`` is imported lazily -- the one
        deliberate checkout-boundary exception this module takes, mirroring
        ``train_config.py``'s ``_optimizer_entry``. When a roster is
        supplied explicitly it is used exactly as given, which is how a test
        flips availability without monkeypatching the ``tpen`` package.

    Returns
    -------
    dict[str, stage_coordinate.OptimizerCell]
        One entry per declared family.

    Raises
    ------
    OptimizerGridError
        If a declared family has no matching roster entry. A missing entry
        is always an error; it is never treated as available.
    """

    if roster is None:
        from importlib import import_module

        schema = import_module("tpen.hi_schema")
        roster = schema.HI_METHOD_ROSTER

    by_method = {entry.method: entry for entry in roster}
    families = tuple(family_entry["family"] for family_entry in load_authority()["families"])
    resolved: dict[str, Any] = {}
    for family in families:
        entry = by_method.get(family)
        if entry is None:
            raise OptimizerGridError(
                f"family {family!r} has no HI_METHOD_ROSTER entry; a missing entry is never available"
            )
        if entry.admitted:
            resolved[family] = stage_coordinate.OptimizerCell(family, "available")
        else:
            resolved[family] = stage_coordinate.OptimizerCell(family, "unavailable", entry.requires)
    return resolved


def render_intended_configurations() -> list[dict[str, Any]]:
    """Render the exact Stage-Q entry list that ``intended_configurations.json`` commits.

    Each of the 120 declared cells becomes one entry carrying its family and
    complete levels (no availability status, no reason), expanded against
    both section-2.8 optimizer contexts. Q's ``updates`` is read from the
    committed stage-coordinate authority, never retyped here.
    """

    stage = stage_coordinate.stage_definition(_STAGE_CODE)
    if stage.updates is None:
        raise OptimizerGridError(f"stage {_STAGE_CODE!r} has no fixed training horizon")
    contexts = load_authority()["contexts"]
    entries: list[dict[str, Any]] = []
    for cell in declared_cells():
        levels = dict(cell.levels)
        configurations = [
            {
                "scientific_identity": {
                    "context": {
                        "name": context["name"],
                        "overrides": dict(context["overrides"]),
                    },
                    "optimizer": {"method": cell.family, "levels": dict(levels)},
                },
                "payload": {"updates": stage.updates},
                "topology": {},
            }
            for context in contexts
        ]
        entries.append(
            {
                "stage": _STAGE_CODE,
                "optimizer": {"method": cell.family, "levels": levels},
                "configurations": configurations,
            }
        )
    return entries


def render_intended_configurations_text() -> str:
    """Return the exact committed-file text for ``intended_configurations.json``."""

    return json.dumps(render_intended_configurations(), indent=2) + "\n"


def stage_q_report(*, roster: Any = None) -> dict[str, Any]:
    """Return the machine-readable Stage-Q inventory report.

    Every count is computed from :func:`declared_cells` and
    :func:`roster_availability`; nothing here is a literal runnable count,
    and nothing here asserts that all 120 cells can launch.
    """

    cells = declared_cells()
    availability = roster_availability(roster)
    per_family: dict[str, dict[str, Any]] = {}
    for cell in cells:
        state = availability[cell.family]
        bucket = per_family.setdefault(
            cell.family, {"declared": 0, "runnable": 0, "unavailable": 0, "reason": None}
        )
        bucket["declared"] += 1
        if state.status == "available":
            bucket["runnable"] += 1
        else:
            bucket["unavailable"] += 1
            bucket["reason"] = state.reason
    return {
        "stage": _STAGE_CODE,
        "declared": sum(bucket["declared"] for bucket in per_family.values()),
        "runnable": sum(bucket["runnable"] for bucket in per_family.values()),
        "unavailable": sum(bucket["unavailable"] for bucket in per_family.values()),
        "excluded_stages": {
            "O1": (
                "independently derivable from the same 120 cells x 2 contexts; deferred to a "
                "follow-up slice, not declared here (lane-design-decisions-2026-10-09, item 2)"
            ),
            "R": "needs eight architecture packages selected after A/B, which do not exist yet",
            "F": "needs one frozen selected procedure, which does not exist yet",
        },
        "families": per_family,
    }


def main(argv: list[str] | None = None) -> None:
    """Print the Stage-Q report, and on ``--write`` regenerate the committed file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite intended_configurations.json from the authority.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(stage_q_report(), indent=2, sort_keys=True))
    if args.write:
        _INTENDED_CONFIGURATIONS_PATH.write_text(render_intended_configurations_text(), encoding="utf-8")


if __name__ == "__main__":
    main()


__all__ = [
    "GridCell",
    "OptimizerGridError",
    "declared_cells",
    "load_authority",
    "render_intended_configurations",
    "render_intended_configurations_text",
    "roster_availability",
    "stage_q_report",
]
