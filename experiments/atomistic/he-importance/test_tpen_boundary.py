"""Enforce the ``experiments/README.md`` boundary for this study.

This test reads study files from disk and parses them with :mod:`ast`.  It does
not import the study: ``he-importance`` is not an importable package name, and
the study's imports would pull in its training dependencies.

The inventory is pinned to ``dev`` at 3143e43ac170df33f61f2002b1011cd85c9cfc37
on 2026-10-07.  The two inventories are intentionally explicit: every measured
crossing must be declared, and every declared entry for a file already present
must be measured.  Entries for files introduced by pending PR 516 are admitted
only with an explicit pending PR and source SHA.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parent
INVENTORY_DATE = "2026-10-07"
INVENTORY_HEAD = "3143e43ac170df33f61f2002b1011cd85c9cfc37"
PENDING_516_SHA = "1ec5f658a19c227ad7e30d1e8d7e82f0b50dd879"


@dataclass(frozen=True)
class Crossing:
    """One source-level tpen boundary crossing found by the AST walk."""

    line: int
    target: str
    sanctioned: bool = False


@dataclass(frozen=True)
class InventoryEntry:
    """Disposition and optional admission metadata for one declared crossing."""

    disposition: str
    pending_pr: int | None = None
    pending_sha: str | None = None


# Each key is (relative file, line, dotted target).  Each value is the
# one-line disposition recorded for that exact crossing.
EXPECTED_PRODUCTION_INVENTORY = {
    ("train_config.py", 187, "tpen.hi_schema"): InventoryEntry(
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
    ("train_config.py", 258, "tpen.hi_schema"): InventoryEntry(
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
    ("run_stage_q.py", 30, "tpen.accelerator"): InventoryEntry(
        "pending PR 516: reconcile only if this crossing differs from its measured SHA.",
        pending_pr=516,
        pending_sha=PENDING_516_SHA,
    ),
    ("run_stage_q.py", 31, "tpen.distributed"): InventoryEntry(
        "pending PR 516: reconcile only if this crossing differs from its measured SHA.",
        pending_pr=516,
        pending_sha=PENDING_516_SHA,
    ),
    ("run_stage_q.py", 100, "tpen.hi_schema"): InventoryEntry(
        "pending PR 516 and item df8f8b31: reconcile only if this crossing differs from its measured SHA.",
        pending_pr=516,
        pending_sha=PENDING_516_SHA,
    ),
}

EXPECTED_TEST_INVENTORY = {
    ("test_launch.py", 16, "tpen.artifacts"): InventoryEntry(
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 17, "tpen.distributed"): InventoryEntry(
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 19, "tpen.runner"): InventoryEntry(
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 388, "tpen.run"): InventoryEntry(
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 430, "tpen.run"): InventoryEntry(
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 486, "tpen.run"): InventoryEntry(
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_launch.py", 591, "tpen.run"): InventoryEntry(
        "item 0df7f0cd: test-only crossing retained as an explicit follow-up entry."
    ),
    ("test_stage_coordinate.py", 594, "tpen.hi.train"): InventoryEntry(
        "item e923ec4e: retained until the facade slice removes this crossing."
    ),
    ("test_train_config.py", 176, "tpen.hi_schema"): InventoryEntry(
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
    ("test_train_config.py", 188, "tpen.hi_schema"): InventoryEntry(
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
    ("test_train_config.py", 281, "tpen.hi_schema"): InventoryEntry(
        "item df8f8b31: remedy blocked on pending HI authority ruling 99591859."
    ),
    ("test_run_stage_q.py", 99, "tpen.hi_schema"): InventoryEntry(
        "pending PR 516 and item df8f8b31: reconcile only if this crossing differs from its measured SHA.",
        pending_pr=516,
        pending_sha=PENDING_516_SHA,
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
        arguments = [*node.args]
        arguments.extend(
            keyword.value for keyword in node.keywords if keyword.arg == "name"
        )
        for argument in arguments:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if _is_tpen_name(argument.value):
                    crossings.append(Crossing(node.lineno, argument.value))
                    break

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


def _validate_inventory_admission(
    measured: set[tuple[str, int, str]],
    declared: dict[tuple[str, int, str], InventoryEntry],
) -> tuple[tuple[tuple[str, int, str], int, str | None], ...]:
    """Apply the three-part admission rule and report absent-file allowances."""

    undeclared = measured - set(declared)
    assert not undeclared, f"undeclared tpen crossings: {sorted(undeclared)}"

    admitted_pending: list[tuple[tuple[str, int, str], int, str | None]] = []
    for key, entry in declared.items():
        file_path = STUDY_DIR / key[0]
        if file_path.is_file():
            # pending_pr never excuses a stale entry for a file already in the
            # tree: an existing file must contain its declared crossing.
            assert key in measured, f"declared crossing not measured: {key}"
            continue

        assert entry.pending_pr is not None, (
            f"absent-file entry lacks pending_pr admission: {key}"
        )
        admitted_pending.append((key, entry.pending_pr, entry.pending_sha))

    return tuple(sorted(admitted_pending))


def _inventory_keys(
    scanned: tuple[tuple[str, Crossing], ...], *, tests: bool
) -> set[tuple[str, int, str]]:
    """Project scanner results into one of the two named inventories."""

    return {
        (relative_path, crossing.line, crossing.target)
        for relative_path, crossing in scanned
        if Path(relative_path).name.startswith("test_") is tests
    }


def _assert_pending_516_partition(
    measured: set[tuple[str, int, str]],
    declared: dict[tuple[str, int, str], InventoryEntry],
    pending: tuple[tuple[tuple[str, int, str], int, str | None], ...],
) -> tuple[int, int]:
    """Require each declared PR-516 crossing to be pending or measured once."""

    pending_by_key = {key: (pending_pr, pending_sha) for key, pending_pr, pending_sha in pending}
    pending_keys = set(pending_by_key)
    measured_keys = {
        key for key, entry in declared.items() if entry.pending_pr == 516 and key in measured
    }
    declared_516 = {
        key for key, entry in declared.items() if entry.pending_pr == 516
    }
    assert pending_keys | measured_keys == declared_516
    assert not pending_keys & measured_keys

    for key in declared_516:
        file_exists = (STUDY_DIR / key[0]).is_file()
        assert (key in pending_keys) is (not file_exists)
        if key in pending_keys:
            assert pending_by_key[key] == (516, PENDING_516_SHA)

    return len(pending_keys), len(measured_keys)


def test_inventory_metadata_is_pinned_and_dated() -> None:
    assert INVENTORY_DATE == "2026-10-07"
    assert INVENTORY_HEAD == "3143e43ac170df33f61f2002b1011cd85c9cfc37"
    assert PENDING_516_SHA == "1ec5f658a19c227ad7e30d1e8d7e82f0b50dd879"


def test_scanner_has_nonempty_inputs_and_crossings() -> None:
    visited, crossings = _scan_study()
    assert len(visited) > 0
    assert len(crossings) > 0


def test_production_inventory_obeys_admission_rule() -> None:
    _, crossings = _scan_study()
    measured = _inventory_keys(crossings, tests=False)
    pending = _validate_inventory_admission(measured, EXPECTED_PRODUCTION_INVENTORY)
    pending_count, measured_count = _assert_pending_516_partition(
        measured, EXPECTED_PRODUCTION_INVENTORY, pending
    )
    assert pending_count + measured_count == 3


def test_test_file_inventory_obeys_admission_rule() -> None:
    _, crossings = _scan_study()
    measured = _inventory_keys(crossings, tests=True)
    pending = _validate_inventory_admission(measured, EXPECTED_TEST_INVENTORY)
    pending_count, measured_count = _assert_pending_516_partition(
        measured, EXPECTED_TEST_INVENTORY, pending
    )
    assert pending_count + measured_count == 1


def test_dynamic_import_module_crossing_is_detected() -> None:
    source = '''
from importlib import import_module
module = import_module("tpen.anything")
'''
    crossings = _detect_crossings(source)
    assert any(crossing.target == "tpen.anything" for crossing in crossings)


def test_dynamic_import_module_keyword_crossing_is_detected() -> None:
    source = '''
from importlib import import_module
module = import_module(name="tpen.anything")
'''
    crossings = _detect_crossings(source)
    assert any(crossing.target == "tpen.anything" for crossing in crossings)


def test_function_local_static_import_crossing_is_detected() -> None:
    source = '''
def resolve():
    from tpen.anything import value
    return value
'''
    crossings = _detect_crossings(source)
    assert any(crossing.target == "tpen.anything" for crossing in crossings)


def test_absent_file_pending_pr_entry_is_admitted_and_reported() -> None:
    key = ("pending_future.py", 1, "tpen.anything")
    declared = {
        key: InventoryEntry(
            "pending PR 516 admission.", pending_pr=516, pending_sha=PENDING_516_SHA
        )
    }
    assert _validate_inventory_admission(set(), declared) == (
        (key, 516, PENDING_516_SHA),
    )


def test_existing_file_with_absent_crossing_rejects_pending_pr_allowance() -> None:
    key = ("test_tpen_boundary.py", 1, "tpen.never_imported")
    declared = {
        key: InventoryEntry(
            "stale entry must fail.", pending_pr=516, pending_sha=PENDING_516_SHA
        )
    }
    try:
        _validate_inventory_admission(set(), declared)
    except AssertionError:
        pass
    else:
        raise AssertionError("pending_pr must not excuse a stale entry for an existing file")


def test_undeclared_measured_crossing_always_fails() -> None:
    measured = {("existing.py", 1, "tpen.unlisted")}
    try:
        _validate_inventory_admission(measured, {})
    except AssertionError:
        pass
    else:
        raise AssertionError("every measured crossing must have a declared entry")
