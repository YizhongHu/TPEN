"""Enforce the ``experiments/README.md`` boundary for this study.

This test reads study files from disk and parses them with :mod:`ast`.  It does
not import the study: ``he-importance`` is not an importable package name, and
the study's imports would pull in its training dependencies.

The inventory is pinned to ``dev`` at 3143e43ac170df33f61f2002b1011cd85c9cfc37
on 2026-10-07.  The two inventories are intentionally explicit so that a new
crossing, or the removal of a declared crossing, fails review until the
inventory is updated with its disposition.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
INVENTORY_DATE = "2026-10-07"
INVENTORY_HEAD = "3143e43ac170df33f61f2002b1011cd85c9cfc37"


@dataclass(frozen=True)
class Crossing:
    """One source-level tpen boundary crossing found by the AST walk."""

    line: int
    target: str
    sanctioned: bool = False


# Each key is (relative file, line, dotted target).  Each value is the
# one-line disposition recorded for that exact crossing.
EXPECTED_PRODUCTION_INVENTORY = {
    ("launch.py", 22, "tpen.accelerator"): (
        "item ffb269d1: removed by the next stack layer."
    ),
    ("launch.py", 23, "tpen.artifacts"): (
        "item ffb269d1: removed by the next stack layer."
    ),
    ("launch.py", 24, "tpen.distributed"): (
        "item ffb269d1: removed by the next stack layer."
    ),
    ("train_config.py", 187, "tpen.hi_schema"): (
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
    ("train_config.py", 258, "tpen.hi_schema"): (
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
}

EXPECTED_TEST_INVENTORY = {
    ("test_launch.py", 16, "tpen.artifacts"): (
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 17, "tpen.distributed"): (
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 19, "tpen.runner"): (
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 388, "tpen.run"): (
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 430, "tpen.run"): (
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 486, "tpen.run"): (
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 591, "tpen.run"): (
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_stage_coordinate.py", 594, "tpen.hi.train"): (
        "item e923ec4e: retained until the facade slice removes this crossing."
    ),
    ("test_train_config.py", 176, "tpen.hi_schema"): (
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
    ("test_train_config.py", 188, "tpen.hi_schema"): (
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
    ("test_train_config.py", 281, "tpen.hi_schema"): (
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
}


def _is_tpen_name(name: str) -> bool:
    """Return whether *name* names ``tpen`` or one of its submodules."""

    return name == "tpen" or name.startswith("tpen.")


def _is_sanctioned_import(node: ast.ImportFrom) -> bool:
    """Recognize the one launcher import allowed by ``experiments/README.md``."""

    return (
        node.module == "tpen.run"
        and len(node.names) == 1
        and node.names[0].name == "run_from_config"
    )


def _dynamic_import_name(node: ast.Call) -> str | None:
    """Return the supported dynamic-import spelling, if *node* uses one."""

    if isinstance(node.func, ast.Name) and node.func.id in {"import_module", "__import__"}:
        return node.func.id
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
    ):
        return "importlib.import_module"
    return None


def _detect_crossings(source: str) -> tuple[Crossing, ...]:
    """Detect static imports and literal dynamic imports in *source*."""

    crossings: list[Crossing] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_tpen_name(alias.name):
                    crossings.append(Crossing(node.lineno, alias.name))
            continue

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_tpen_name(module):
                target = module
                if _is_sanctioned_import(node):
                    target = f"{module}.{node.names[0].name}"
                crossings.append(
                    Crossing(node.lineno, target, sanctioned=_is_sanctioned_import(node))
                )
            continue

        if not isinstance(node, ast.Call) or _dynamic_import_name(node) is None:
            continue
        if not node.args:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            if _is_tpen_name(argument.value):
                crossings.append(Crossing(node.lineno, argument.value))

    return tuple(sorted(crossings, key=lambda crossing: (crossing.line, crossing.target)))


def _scan_study() -> tuple[tuple[Path, ...], tuple[tuple[str, Crossing], ...]]:
    """Read and parse every Python file, returning only unsanctioned crossings."""

    paths = tuple(sorted(STUDY_DIR.rglob("*.py")))
    all_crossings: list[tuple[str, Crossing]] = []
    for path in paths:
        relative_path = path.relative_to(STUDY_DIR).as_posix()
        source = path.read_text(encoding="utf-8")
        all_crossings.extend(
            (relative_path, crossing) for crossing in _detect_crossings(source)
        )

    # These are controls, not optional diagnostics: an empty walk must fail
    # instead of making the declared inventory pass vacuously.
    assert len(paths) > 0, "boundary scanner visited no Python files"
    assert len(all_crossings) > 0, "boundary scanner found no tpen crossings"

    violations = tuple(
        (relative_path, crossing)
        for relative_path, crossing in all_crossings
        if not crossing.sanctioned
    )
    assert len(violations) > 0, "boundary scanner found no unsanctioned tpen crossings"
    return paths, violations


def _inventory_keys(
    scanned: tuple[tuple[str, Crossing], ...], *, tests: bool
) -> set[tuple[str, int, str]]:
    """Project scanner results into one of the two named inventories."""

    return {
        (relative_path, crossing.line, crossing.target)
        for relative_path, crossing in scanned
        if relative_path.startswith("test_") is tests
    }


def test_inventory_metadata_is_pinned_and_dated() -> None:
    assert INVENTORY_DATE == "2026-10-07"
    assert INVENTORY_HEAD == "3143e43ac170df33f61f2002b1011cd85c9cfc37"


def test_scanner_has_nonempty_inputs_and_crossings() -> None:
    visited, crossings = _scan_study()
    assert len(visited) > 0
    assert len(crossings) > 0


def test_production_inventory_is_exact() -> None:
    _, crossings = _scan_study()
    assert _inventory_keys(crossings, tests=False) == set(EXPECTED_PRODUCTION_INVENTORY)


def test_test_file_inventory_is_exact() -> None:
    _, crossings = _scan_study()
    assert _inventory_keys(crossings, tests=True) == set(EXPECTED_TEST_INVENTORY)


def test_dynamic_import_module_crossing_is_detected() -> None:
    source = '''
from importlib import import_module
module = import_module("tpen.anything")
'''
    crossings = _detect_crossings(source)
    assert any(crossing.target == "tpen.anything" for crossing in crossings)
