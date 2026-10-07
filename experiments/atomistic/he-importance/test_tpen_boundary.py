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
import tempfile


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
    ("launch.py", 22, "tpen.accelerator"): InventoryEntry(
        "item ffb269d1: removed by the next stack layer."
    ),
    ("launch.py", 23, "tpen.artifacts"): InventoryEntry(
        "item ffb269d1: removed by the next stack layer."
    ),
    ("launch.py", 24, "tpen.distributed"): InventoryEntry(
        "item ffb269d1: removed by the next stack layer."
    ),
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


# CONTROL CAPABILITY MATRIX
# Each detector capability has a property-named negative control.  Keep this
# list synchronized with the implementation above: a capability mutation must
# kill the control named on its row, with pytest cache disabled during audits.
#
# static ast.Import / module scope       -> test_capability_static_import_statement
# static import alias iteration          -> test_capability_multiple_static_import_aliases
# static ast.ImportFrom                  -> test_capability_static_from_import
# bare tpen name                         -> test_capability_bare_tpen_name
# dotted tpen name                       -> test_capability_dotted_tpen_name
# function-local nesting                 -> test_capability_function_local_static_import
# class-body nesting                     -> test_capability_class_body_static_import
# recursive AST traversal                -> test_capability_recursive_ast_walk
# bare import_module dynamic form        -> test_capability_dynamic_bare_import_module
# qualified importlib dynamic form       -> test_capability_dynamic_qualified_importlib
# __import__ dynamic form                -> test_capability_dynamic_dunder_import
# positional dynamic argument            -> test_capability_dynamic_positional_argument
# keyword dynamic argument               -> test_capability_dynamic_keyword_argument
# dynamic argument iteration             -> test_capability_dynamic_argument_iteration
# recursive *.py discovery               -> test_capability_recursive_python_file_discovery
# *.py file filtering                    -> test_capability_python_file_filter
# sanctioned runner carve-out            -> test_capability_sanctioned_runner_carveout
# production/test partition              -> test_capability_production_test_partition
# non-empty file scan                    -> test_capability_nonempty_scan / test_capability_empty_scan_rejected
# non-empty crossing scan                -> test_capability_no_crossings_rejected
# non-empty unsanctioned scan            -> test_capability_no_unsanctioned_crossings_rejected
# literal dynamic target                 -> test_capability_dynamic_literal_string
# measured/declared admission            -> test_capability_undeclared_measured_crossing
# existing-file stale admission          -> test_capability_existing_stale_entry
# absent-file pending admission           -> test_capability_absent_pending_entry
# absent-file pending_pr requirement      -> test_capability_absent_without_pending_pr
# pending/measured partition             -> test_capability_pending_entry_partition


def test_inventory_metadata_is_pinned_and_dated() -> None:
    assert INVENTORY_DATE == "2026-10-07"
    assert INVENTORY_HEAD == "3143e43ac170df33f61f2002b1011cd85c9cfc37"
    assert PENDING_516_SHA == "1ec5f658a19c227ad7e30d1e8d7e82f0b50dd879"


def test_capability_static_import_statement() -> None:
    crossings = _detect_crossings("import tpen\n")
    assert [(crossing.line, crossing.target) for crossing in crossings] == [(1, "tpen")]


def test_capability_multiple_static_import_aliases() -> None:
    crossings = _detect_crossings("import tpen.one, tpen.two\n")
    assert [crossing.target for crossing in crossings] == ["tpen.one", "tpen.two"]


def test_capability_static_from_import() -> None:
    crossings = _detect_crossings("from tpen.accelerator import AcceleratorKind\n")
    assert [(crossing.line, crossing.target) for crossing in crossings] == [
        (1, "tpen.accelerator")
    ]


def test_capability_bare_tpen_name() -> None:
    crossings = _detect_crossings("import tpen\n")
    assert [(crossing.line, crossing.target) for crossing in crossings] == [(1, "tpen")]


def test_capability_dotted_tpen_name() -> None:
    crossings = _detect_crossings("import tpen.accelerator\n")
    assert [(crossing.line, crossing.target) for crossing in crossings] == [
        (1, "tpen.accelerator")
    ]


def test_capability_class_body_static_import() -> None:
    source = '''
class Resolver:
    from tpen.anything import value
'''
    crossings = _detect_crossings(source)
    assert [(crossing.line, crossing.target) for crossing in crossings] == [(3, "tpen.anything")]


def test_capability_recursive_ast_walk() -> None:
    source = '''
def resolve():
    class Nested:
        import tpen.deep
'''
    crossings = _detect_crossings(source)
    assert [(crossing.line, crossing.target) for crossing in crossings] == [(4, "tpen.deep")]


def test_capability_dynamic_qualified_importlib() -> None:
    source = '''
import importlib
module = importlib.import_module("tpen.anything")
'''
    crossings = _detect_crossings(source)
    assert [(crossing.line, crossing.target) for crossing in crossings] == [
        (3, "tpen.anything")
    ]


def test_capability_dynamic_dunder_import() -> None:
    source = 'module = __import__("tpen.anything", fromlist=["value"])\n'
    crossings = _detect_crossings(source)
    assert [(crossing.line, crossing.target) for crossing in crossings] == [
        (1, "tpen.anything")
    ]


def test_capability_dynamic_positional_argument() -> None:
    source = 'module = import_module("tpen.anything")\n'
    crossings = _detect_crossings(source)
    assert [(crossing.line, crossing.target) for crossing in crossings] == [
        (1, "tpen.anything")
    ]


def test_capability_dynamic_literal_string() -> None:
    source = 'module = import_module("tpen.literal")\n'
    crossings = _detect_crossings(source)
    assert crossings[0].target == "tpen.literal"


def test_capability_dynamic_argument_iteration() -> None:
    source = 'module = import_module("stdlib", "tpen.second")\n'
    crossings = _detect_crossings(source)
    assert crossings[0].target == "tpen.second"


def test_capability_recursive_python_file_discovery() -> None:
    global STUDY_DIR
    original_study_dir = STUDY_DIR
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        nested = root / "one" / "two"
        nested.mkdir(parents=True)
        (nested / "deep.py").write_text("import tpen.deep\n", encoding="utf-8")
        STUDY_DIR = root
        try:
            _, crossings = _scan_study()
        finally:
            STUDY_DIR = original_study_dir
    assert [(path, crossing.line, crossing.target) for path, crossing in crossings] == [
        ("one/two/deep.py", 1, "tpen.deep")
    ]


def test_capability_python_file_filter() -> None:
    global STUDY_DIR
    original_study_dir = STUDY_DIR
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "kept.py").write_text("import tpen.kept\n", encoding="utf-8")
        (root / "ignored.txt").write_text("import tpen.ignored\n", encoding="utf-8")
        STUDY_DIR = root
        try:
            _, crossings = _scan_study()
        finally:
            STUDY_DIR = original_study_dir
    assert [(path, crossing.target) for path, crossing in crossings] == [
        ("kept.py", "tpen.kept")
    ]


def test_capability_sanctioned_runner_carveout() -> None:
    crossings = _detect_crossings("from tpen.run import run_from_config\n")
    assert len(crossings) == 1
    assert crossings[0].sanctioned
    assert crossings[0].target == "tpen.run.run_from_config"
    _, scanned = _scan_study()
    assert not any(
        path == "launch.py" and crossing.line == 25 for path, crossing in scanned
    )


def test_capability_production_test_partition() -> None:
    scanned = (
        ("production.py", Crossing(1, "tpen.production")),
        ("nested/test_case.py", Crossing(2, "tpen.test")),
    )
    assert _inventory_keys(scanned, tests=False) == {
        ("production.py", 1, "tpen.production")
    }
    assert _inventory_keys(scanned, tests=True) == {
        ("nested/test_case.py", 2, "tpen.test")
    }


def test_capability_nonempty_scan() -> None:
    visited, crossings = _scan_study()
    assert len(visited) > 0
    assert len(crossings) > 0


def test_capability_empty_scan_rejected() -> None:
    global STUDY_DIR
    original_study_dir = STUDY_DIR
    with tempfile.TemporaryDirectory() as temporary_root:
        STUDY_DIR = Path(temporary_root)
        try:
            try:
                _scan_study()
            except AssertionError as exc:
                assert "visited no Python files" in str(exc)
            else:
                raise AssertionError("an empty study must be rejected")
        finally:
            STUDY_DIR = original_study_dir


def test_capability_no_crossings_rejected() -> None:
    global STUDY_DIR
    original_study_dir = STUDY_DIR
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "plain.py").write_text("value = 1\n", encoding="utf-8")
        STUDY_DIR = root
        try:
            try:
                _scan_study()
            except AssertionError as exc:
                assert "found no tpen crossings" in str(exc)
            else:
                raise AssertionError("a study with no crossings must be rejected")
        finally:
            STUDY_DIR = original_study_dir


def test_capability_no_unsanctioned_crossings_rejected() -> None:
    global STUDY_DIR
    original_study_dir = STUDY_DIR
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "launcher.py").write_text(
            "from tpen.run import run_from_config\n", encoding="utf-8"
        )
        STUDY_DIR = root
        try:
            try:
                _scan_study()
            except AssertionError as exc:
                assert "found no unsanctioned tpen crossings" in str(exc)
            else:
                raise AssertionError("a study with only sanctioned crossings must be rejected")
        finally:
            STUDY_DIR = original_study_dir


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


def test_capability_dynamic_bare_import_module() -> None:
    source = '''
from importlib import import_module
module = import_module("tpen.anything")
'''
    crossings = _detect_crossings(source)
    assert any(crossing.target == "tpen.anything" for crossing in crossings)


def test_capability_dynamic_keyword_argument() -> None:
    source = '''
from importlib import import_module
module = import_module(name="tpen.anything")
'''
    crossings = _detect_crossings(source)
    assert any(crossing.target == "tpen.anything" for crossing in crossings)


def test_capability_function_local_static_import() -> None:
    source = '''
def resolve():
    from tpen.anything import value
    return value
'''
    crossings = _detect_crossings(source)
    assert any(crossing.target == "tpen.anything" for crossing in crossings)


def test_capability_absent_pending_entry() -> None:
    key = ("pending_future.py", 1, "tpen.anything")
    declared = {
        key: InventoryEntry(
            "pending PR 516 admission.", pending_pr=516, pending_sha=PENDING_516_SHA
        )
    }
    assert _validate_inventory_admission(set(), declared) == (
        (key, 516, PENDING_516_SHA),
    )


def test_capability_pending_entry_partition() -> None:
    global STUDY_DIR
    original_study_dir = STUDY_DIR
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        existing_key = ("existing.py", 1, "tpen.existing")
        pending_key = ("pending.py", 1, "tpen.pending")
        (root / existing_key[0]).write_text("import tpen.existing\n", encoding="utf-8")
        declared = {
            existing_key: InventoryEntry("measured", pending_pr=516, pending_sha=PENDING_516_SHA),
            pending_key: InventoryEntry("pending", pending_pr=516, pending_sha=PENDING_516_SHA),
        }
        STUDY_DIR = root
        try:
            pending = _validate_inventory_admission({existing_key}, declared)
            assert _assert_pending_516_partition(
                {existing_key}, declared, pending
            ) == (1, 1)
        finally:
            STUDY_DIR = original_study_dir


def test_capability_pending_entry_partition_rejects_gaps_and_overlap() -> None:
    gap_key = ("test_tpen_boundary.py", 1, "tpen.future")
    overlap_key = ("future.py", 1, "tpen.future")
    cases = (
        (gap_key, set(), ()),
        (overlap_key, {overlap_key}, ((overlap_key, 516, PENDING_516_SHA),)),
    )
    for key, measured, pending in cases:
        declared = {
            key: InventoryEntry("pending", pending_pr=516, pending_sha=PENDING_516_SHA)
        }
        try:
            _assert_pending_516_partition(measured, declared, pending)
        except AssertionError:
            pass
        else:
            raise AssertionError("pending/measured entries must form a strict partition")


def test_capability_absent_without_pending_pr() -> None:
    key = ("absent_no_pending.py", 1, "tpen.real")
    declared = {key: InventoryEntry("missing traceability metadata.")}
    try:
        _validate_inventory_admission(set(), declared)
    except AssertionError as exc:
        assert "pending_pr" in str(exc)
    else:
        raise AssertionError("an absent-file entry without pending_pr must be rejected")


def test_capability_existing_stale_entry() -> None:
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


def test_capability_undeclared_measured_crossing() -> None:
    measured = {("existing.py", 1, "tpen.unlisted")}
    try:
        _validate_inventory_admission(measured, {})
    except AssertionError:
        pass
    else:
        raise AssertionError("every measured crossing must have a declared entry")
