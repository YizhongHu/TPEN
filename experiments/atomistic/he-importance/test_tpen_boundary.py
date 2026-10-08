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
# The second value is the four additional unsanctioned crossings introduced by
# PR 516; keep both merge orders green until that PR lands.
EXPECTED_UNSANCTIONED_COUNTS = {16, 20}
PENDING_516_POLICY = (
    "the control is deliberately 516-scoped; a new pending PR needs its own pin"
)


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


def _detect_crossings(source: str, *, filename: str = "<unknown>") -> tuple[Crossing, ...]:
    """Detect static imports and literal dynamic imports in *source*."""

    crossings: list[Crossing] = []
    tree = ast.parse(source, filename=filename)
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
            keyword.value
            for keyword in node.keywords
            if keyword.arg in {"name", "package"}
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
            (relative_path, crossing)
            for crossing in _detect_crossings(source, filename=relative_path)
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
    if len(violations) not in EXPECTED_UNSANCTIONED_COUNTS:
        declared = set(EXPECTED_PRODUCTION_INVENTORY) | set(EXPECTED_TEST_INVENTORY)
        measured = {
            (relative_path, crossing.line, crossing.target)
            for relative_path, crossing in violations
        }
        existing_declared = {
            key for key in declared if (STUDY_DIR / key[0]).is_file()
        }
        added = sorted(measured - declared)
        removed = sorted(existing_declared - measured)
        raise AssertionError(
            "boundary scanner found an unexpected number of unsanctioned tpen "
            f"crossings: actual={len(violations)}, "
            f"expected={sorted(EXPECTED_UNSANCTIONED_COUNTS)}, "
            f"added={added}, removed={removed}; reconcile by removing the import "
            "per experiments/README.md or declaring it with a disposition under "
            "rule (a); if an import merely shifted lines, update the stale sibling "
            "entry under rule (b), then rerun the boundary control"
        )
    return paths, violations


def _validate_inventory_admission(
    measured: set[tuple[str, int, str]],
    declared: dict[tuple[str, int, str], InventoryEntry],
) -> tuple[tuple[tuple[str, int, str], int, str | None], ...]:
    """Apply the three-part admission rule and report absent-file allowances."""

    undeclared = measured - set(declared)
    assert not undeclared, (
        f"undeclared tpen crossings: {sorted(undeclared)}; remove the import per "
        "experiments/README.md or declare it with a disposition (rule (a)); if the "
        "import merely shifted lines, update the stale sibling entry under rule (b)"
    )

    admitted_pending: list[tuple[tuple[str, int, str], int, str | None]] = []
    for key, entry in declared.items():
        file_path = STUDY_DIR / key[0]
        if file_path.is_file():
            # pending_pr never excuses a stale entry for a file already in the
            # tree: an existing file must contain its declared crossing.
            assert key in measured, (
                f"declared crossing not measured: {key}; if the import merely shifted "
                "lines, update the stale sibling entry under rule (b); otherwise "
                "remove the import per experiments/README.md or declare it with a "
                "disposition (rule (a))"
            )
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
    assert pending_keys | measured_keys == declared_516, (
        f"pending_keys={sorted(pending_keys)}, measured_keys={sorted(measured_keys)}, "
        f"declared_516={sorted(declared_516)}; {PENDING_516_POLICY}"
    )
    assert not pending_keys & measured_keys, (
        f"pending_keys={sorted(pending_keys)} overlaps measured_keys="
        f"{sorted(measured_keys)}; {PENDING_516_POLICY}"
    )

    for key in declared_516:
        file_exists = (STUDY_DIR / key[0]).is_file()
        assert (key in pending_keys) is (not file_exists), (
            f"key={key}, pending_keys={sorted(pending_keys)}, file_exists={file_exists}; "
            f"{PENDING_516_POLICY}"
        )
        if key in pending_keys:
            assert pending_by_key[key] == (516, PENDING_516_SHA), (
                f"key={key}, pending_by_key={pending_by_key[key]}, "
                f"expected=(516, {PENDING_516_SHA!r}); {PENDING_516_POLICY}"
            )

    return len(pending_keys), len(measured_keys)


# CONTROL CAPABILITY MATRIX
# Each detector capability owes BOTH directions: an accept-direction control
# must fail when matching is broken, and a reject-direction control must fail
# when the restriction is loosened. Keep this list synchronized with the
# implementation above; audit mutations with pytest cache disabled.
#
# static ast.Import / module scope       -> test_capability_static_import_statement
# static import alias iteration          -> test_capability_multiple_static_import_aliases
# static ast.ImportFrom                  -> test_capability_static_from_import
# bare tpen name                         -> test_capability_bare_tpen_name
# dotted tpen name                       -> test_capability_dotted_tpen_name
# tpen-name rejection                    -> test_capability_tpen_name_rejects_non_tpen
# function-local nesting                 -> test_capability_function_local_static_import
# class-body nesting                     -> test_capability_class_body_static_import
# recursive AST traversal                -> test_capability_recursive_ast_walk
# bare import_module dynamic form        -> test_capability_dynamic_bare_import_module
# qualified importlib dynamic form       -> test_capability_dynamic_qualified_importlib
# __import__ dynamic form                -> test_capability_dynamic_dunder_import
# unknown dynamic callable rejection     -> test_capability_dynamic_rejects_unknown_callable
# positional dynamic argument            -> test_capability_dynamic_positional_argument
# name/package keyword dynamic arguments (exact target + unsanctioned) -> test_capability_dynamic_keyword_argument
# irrelevant keyword rejection            -> test_capability_dynamic_rejects_non_name_keyword
# dynamic argument iteration             -> test_capability_dynamic_argument_iteration
# disk source reading                    -> test_capability_reads_source_from_disk
# AST source parsing                     -> test_capability_ast_source_parsing / test_capability_ast_parse_diagnostic_names_relative_file
# recursive *.py discovery               -> test_capability_recursive_python_file_discovery
# *.py file filtering                    -> test_capability_python_file_filter
# sanctioned runner carve-out            -> test_capability_sanctioned_runner_carveout
# sanctioned module restriction           -> test_capability_sanctioned_import_rejects_wrong_module
# sanctioned name-count restriction      -> test_capability_sanctioned_import_rejects_extra_names
# sanctioned symbol restriction          -> test_capability_sanctioned_import_rejects_wrong_symbol
# qualified receiver restriction         -> test_capability_dynamic_rejects_non_importlib_receiver
# qualified attribute restriction        -> test_capability_dynamic_rejects_non_import_module_attribute
# production/test partition              -> test_capability_production_test_partition
# non-empty file scan                    -> test_capability_nonempty_scan / test_capability_empty_scan_rejected
# non-empty crossing scan                -> test_capability_no_crossings_rejected
# unsanctioned filter partition (including keyword form) -> test_capability_unsanctioned_filter_partition
# added crossing count diagnostic        -> test_capability_inventory_count_diagnostic_added
# removed crossing count diagnostic      -> test_capability_inventory_count_diagnostic_removed
# unchanged count reaches admission      -> test_capability_inventory_count_diagnostic_line_shift
# zero unsanctioned crossings allowed    -> test_capability_no_unsanctioned_crossings_allowed
# literal dynamic target                 -> test_capability_dynamic_literal_string
# nonliteral dynamic target rejection     -> test_capability_dynamic_rejects_nonliteral
# measured/declared admission            -> test_capability_undeclared_measured_crossing
# existing-file stale admission          -> test_capability_existing_stale_entry
# absent-file pending admission           -> test_capability_absent_pending_entry
# absent-file pending_pr requirement      -> test_capability_absent_without_pending_pr
# pending metadata pinning                -> test_capability_pending_entry_rejects_mispinned_metadata
# pending/measured partition             -> test_capability_pending_entry_partition
# pending-516 diagnostic scope           -> test_capability_pending_516_partition_diagnostic_is_516_scoped
# tpen prefix semantics                   -> test_capability_tpen_name_rejects_substring_matches
# admission diagnostics                  -> test_capability_undeclared_measured_crossing / test_capability_existing_stale_entry


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
    crossings = _detect_crossings("import tpen\nimport tpen.accelerator\n")
    assert [crossing.target for crossing in crossings] == ["tpen", "tpen.accelerator"]
    assert crossings[0].target == "tpen"


def test_capability_dotted_tpen_name() -> None:
    crossings = _detect_crossings("import tpen.accelerator\n")
    assert [(crossing.line, crossing.target) for crossing in crossings] == [
        (1, "tpen.accelerator")
    ]


def test_capability_tpen_name_rejects_non_tpen() -> None:
    assert _detect_crossings("import numpy\n") == ()
    assert _detect_crossings("from numpy import array\n") == ()


def test_capability_tpen_name_rejects_substring_matches() -> None:
    assert _is_tpen_name("tpen")
    assert _is_tpen_name("tpen.anything")
    assert not _is_tpen_name("mytpen")
    assert not _is_tpen_name("other.tpen")


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


def test_capability_dynamic_rejects_unknown_callable() -> None:
    assert _detect_crossings('load_anything("tpen.anything")\n') == ()


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


def test_capability_dynamic_rejects_nonliteral() -> None:
    source = 'module_name = "tpen.variable"\nimport_module(module_name)\n'
    assert _detect_crossings(source) == ()


def test_capability_dynamic_argument_iteration() -> None:
    source = 'module = import_module("stdlib", "tpen.second")\n'
    crossings = _detect_crossings(source)
    assert crossings[0].target == "tpen.second"


def test_capability_reads_source_from_disk() -> None:
    global STUDY_DIR
    global EXPECTED_UNSANCTIONED_COUNTS
    original_study_dir = STUDY_DIR
    original_expected_counts = EXPECTED_UNSANCTIONED_COUNTS
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "source.py").write_text("import tpen.disk\n", encoding="utf-8")
        STUDY_DIR = root
        EXPECTED_UNSANCTIONED_COUNTS = {1}
        try:
            _, crossings = _scan_study()
        finally:
            STUDY_DIR = original_study_dir
            EXPECTED_UNSANCTIONED_COUNTS = original_expected_counts
    assert [(path, crossing.target) for path, crossing in crossings] == [
        ("source.py", "tpen.disk")
    ]


def test_capability_ast_source_parsing() -> None:
    crossings = _detect_crossings("import tpen.parsed\n")
    assert crossings[0].target == "tpen.parsed"


def test_capability_ast_parse_diagnostic_names_relative_file() -> None:
    global STUDY_DIR
    original_study_dir = STUDY_DIR
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "broken.py").write_text("if :\n", encoding="utf-8")
        STUDY_DIR = root
        try:
            try:
                _scan_study()
            except SyntaxError as exc:
                assert exc.filename == "broken.py"
            else:
                raise AssertionError("a syntax error must be reported by its relative file")
        finally:
            STUDY_DIR = original_study_dir


def test_capability_recursive_python_file_discovery() -> None:
    global STUDY_DIR
    global EXPECTED_UNSANCTIONED_COUNTS
    original_study_dir = STUDY_DIR
    original_expected_counts = EXPECTED_UNSANCTIONED_COUNTS
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        nested = root / "one" / "two"
        nested.mkdir(parents=True)
        (nested / "deep.py").write_text("import tpen.deep\n", encoding="utf-8")
        STUDY_DIR = root
        EXPECTED_UNSANCTIONED_COUNTS = {1}
        try:
            _, crossings = _scan_study()
        finally:
            STUDY_DIR = original_study_dir
            EXPECTED_UNSANCTIONED_COUNTS = original_expected_counts
    assert [(path, crossing.line, crossing.target) for path, crossing in crossings] == [
        ("one/two/deep.py", 1, "tpen.deep")
    ]


def test_capability_python_file_filter() -> None:
    global STUDY_DIR
    global EXPECTED_UNSANCTIONED_COUNTS
    original_study_dir = STUDY_DIR
    original_expected_counts = EXPECTED_UNSANCTIONED_COUNTS
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "kept.py").write_text("import tpen.kept\n", encoding="utf-8")
        (root / "ignored.txt").write_text("import tpen.ignored\n", encoding="utf-8")
        STUDY_DIR = root
        EXPECTED_UNSANCTIONED_COUNTS = {1}
        try:
            _, crossings = _scan_study()
        finally:
            STUDY_DIR = original_study_dir
            EXPECTED_UNSANCTIONED_COUNTS = original_expected_counts
    assert [(path, crossing.target) for path, crossing in crossings] == [
        ("kept.py", "tpen.kept")
    ]


def test_capability_sanctioned_runner_carveout() -> None:
    launch_source = (STUDY_DIR / "launch.py").read_text(encoding="utf-8")
    sanctioned = [
        crossing
        for crossing in _detect_crossings(launch_source, filename="launch.py")
        if crossing.sanctioned
    ]
    assert len(sanctioned) == 1
    assert sanctioned[0].target == "tpen.run.run_from_config"
    _, scanned = _scan_study()
    assert not any(
        path == "launch.py" and crossing.target == "tpen.run.run_from_config"
        for path, crossing in scanned
    )


def test_capability_sanctioned_import_rejects_wrong_module() -> None:
    crossings = _detect_crossings("from tpen.other import run_from_config\n")
    assert len(crossings) == 1
    assert not crossings[0].sanctioned
    assert crossings[0].target == "tpen.other"


def test_capability_sanctioned_import_rejects_extra_names() -> None:
    crossings = _detect_crossings(
        "from tpen.run import run_from_config, prepare_run_context\n"
    )
    assert len(crossings) == 1
    assert not crossings[0].sanctioned
    assert crossings[0].target == "tpen.run"


def test_capability_sanctioned_import_rejects_wrong_symbol() -> None:
    crossings = _detect_crossings("from tpen.run import prepare_run_context\n")
    assert len(crossings) == 1
    assert not crossings[0].sanctioned
    assert crossings[0].target == "tpen.run"


def test_capability_dynamic_rejects_non_importlib_receiver() -> None:
    crossings = _detect_crossings('other.import_module("tpen.anything")\n')
    assert crossings == ()


def test_capability_dynamic_rejects_non_import_module_attribute() -> None:
    crossings = _detect_crossings('importlib.load_module("tpen.anything")\n')
    assert crossings == ()


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
    assert len(crossings) in EXPECTED_UNSANCTIONED_COUNTS


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


def test_capability_unsanctioned_filter_partition() -> None:
    global STUDY_DIR
    global EXPECTED_UNSANCTIONED_COUNTS
    original_study_dir = STUDY_DIR
    original_expected_counts = EXPECTED_UNSANCTIONED_COUNTS
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "sanctioned.py").write_text(
            "from tpen.run import run_from_config\n", encoding="utf-8"
        )
        (root / "unsanctioned.py").write_text(
            "from tpen.accelerator import AcceleratorKind\n", encoding="utf-8"
        )
        (root / "keyword.py").write_text(
            'importlib.import_module(".r3_hidden", package="tpen")\n',
            encoding="utf-8",
        )
        STUDY_DIR = root
        EXPECTED_UNSANCTIONED_COUNTS = {2}
        try:
            _, crossings = _scan_study()
        finally:
            STUDY_DIR = original_study_dir
            EXPECTED_UNSANCTIONED_COUNTS = original_expected_counts
    assert [(path, crossing.target) for path, crossing in crossings] == [
        ("keyword.py", "tpen"),
        ("unsanctioned.py", "tpen.accelerator"),
    ]


def test_capability_no_unsanctioned_crossings_allowed() -> None:
    global STUDY_DIR
    global EXPECTED_UNSANCTIONED_COUNTS
    original_study_dir = STUDY_DIR
    original_expected_counts = EXPECTED_UNSANCTIONED_COUNTS
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "launcher.py").write_text(
            "from tpen.run import run_from_config\n", encoding="utf-8"
        )
        STUDY_DIR = root
        EXPECTED_UNSANCTIONED_COUNTS = {0}
        try:
            _, crossings = _scan_study()
        finally:
            STUDY_DIR = original_study_dir
            EXPECTED_UNSANCTIONED_COUNTS = original_expected_counts
    assert crossings == ()


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


def test_capability_inventory_count_diagnostic_added() -> None:
    global STUDY_DIR
    global EXPECTED_UNSANCTIONED_COUNTS
    original_study_dir = STUDY_DIR
    original_expected_counts = EXPECTED_UNSANCTIONED_COUNTS
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "added.py").write_text(
            "from tpen.added import Added\n", encoding="utf-8"
        )
        STUDY_DIR = root
        EXPECTED_UNSANCTIONED_COUNTS = {0}
        try:
            try:
                _scan_study()
            except AssertionError as error:
                message = str(error)
            else:
                assert False, "added crossing did not trip the public count gate"
        finally:
            STUDY_DIR = original_study_dir
            EXPECTED_UNSANCTIONED_COUNTS = original_expected_counts
    assert "actual=1" in message
    assert "expected=[0]" in message
    assert "('added.py', 1, 'tpen.added')" in message
    assert "reconcile" in message


def test_capability_inventory_count_diagnostic_removed() -> None:
    global STUDY_DIR
    global EXPECTED_UNSANCTIONED_COUNTS
    original_study_dir = STUDY_DIR
    original_expected_counts = EXPECTED_UNSANCTIONED_COUNTS
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "launch.py").write_text(
            "\nfrom tpen.artifacts import RunResult\n", encoding="utf-8"
        )
        STUDY_DIR = root
        EXPECTED_UNSANCTIONED_COUNTS = {2}
        try:
            try:
                _scan_study()
            except AssertionError as error:
                message = str(error)
            else:
                assert False, "removed crossing did not trip the public count gate"
        finally:
            STUDY_DIR = original_study_dir
            EXPECTED_UNSANCTIONED_COUNTS = original_expected_counts
    assert "actual=1" in message
    assert "expected=[2]" in message
    assert "('launch.py', 22, 'tpen.accelerator')" in message
    assert "('launch.py', 24, 'tpen.distributed')" in message
    assert "reconcile" in message


def test_capability_inventory_count_diagnostic_line_shift() -> None:
    global STUDY_DIR
    global EXPECTED_UNSANCTIONED_COUNTS
    original_study_dir = STUDY_DIR
    original_expected_counts = EXPECTED_UNSANCTIONED_COUNTS
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        (root / "launch.py").write_text(
            "\nfrom tpen.accelerator import AcceleratorIdentity\n"
            "from tpen.artifacts import RunResult\n"
            "from tpen.distributed import ExecutionTopology\n",
            encoding="utf-8",
        )
        STUDY_DIR = root
        EXPECTED_UNSANCTIONED_COUNTS = {3}
        try:
            _, crossings = _scan_study()
            measured = _inventory_keys(crossings, tests=False)
            try:
                _validate_inventory_admission(measured, EXPECTED_PRODUCTION_INVENTORY)
            except AssertionError as error:
                message = str(error)
            else:
                assert False, "line-shifted sibling entry was admitted"
        finally:
            STUDY_DIR = original_study_dir
            EXPECTED_UNSANCTIONED_COUNTS = original_expected_counts
    assert "('launch.py', 2, 'tpen.accelerator')" in message
    assert "rule (a)" in message
    assert "rule (b)" in message


def test_capability_dynamic_bare_import_module() -> None:
    source = '''
from importlib import import_module
module = import_module("tpen.anything")
'''
    crossings = _detect_crossings(source)
    assert any(crossing.target == "tpen.anything" for crossing in crossings)


def test_capability_dynamic_keyword_argument() -> None:
    for source, expected_target in (
        ('import_module(name="tpen.anything")\n', "tpen.anything"),
        ('import_module(".anything", package="tpen")\n', "tpen"),
        ('importlib.import_module(".anything", package="tpen")\n', "tpen"),
    ):
        assert _detect_crossings(source) == (
            Crossing(1, expected_target, sanctioned=False),
        )


def test_capability_dynamic_rejects_non_name_keyword() -> None:
    source = 'import_module(label="tpen.anything")\n'
    assert _detect_crossings(source) == ()


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


def test_capability_pending_entry_rejects_mispinned_metadata() -> None:
    global STUDY_DIR
    original_study_dir = STUDY_DIR
    with tempfile.TemporaryDirectory() as temporary_root:
        STUDY_DIR = Path(temporary_root)
        cases = (
            (515, PENDING_516_SHA),
            (516, "wrong-source-sha"),
        )
        try:
            for pending_pr, pending_sha in cases:
                key = ("future.py", 1, "tpen.future")
                declared = {
                    key: InventoryEntry(
                        "mispinned pending declaration",
                        pending_pr=pending_pr,
                        pending_sha=pending_sha,
                    )
                }
                pending = _validate_inventory_admission(set(), declared)
                try:
                    _assert_pending_516_partition(set(), declared, pending)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(
                        "pending admission must reject wrong PR or source SHA"
                    )
        finally:
            STUDY_DIR = original_study_dir


def test_capability_pending_516_partition_diagnostic_is_516_scoped() -> None:
    key = ("future.py", 1, "tpen.future")
    declared = {
        key: InventoryEntry("future pending crossing", pending_pr=530, pending_sha="sha")
    }
    pending = _validate_inventory_admission(set(), declared)
    try:
        _assert_pending_516_partition(set(), declared, pending)
    except AssertionError as exc:
        message = str(exc)
        assert "pending_keys" in message
        assert "declared_516" in message
        assert "516-scoped" in message
        assert "new pending PR needs its own pin" in message
    else:
        raise AssertionError("a non-516 pending PR must not enter the 516 partition")


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
    except AssertionError as exc:
        message = str(exc)
        assert "declared crossing not measured" in message
        assert "stale sibling entry under rule (b)" in message
        assert "experiments/README.md" in message
    else:
        raise AssertionError("pending_pr must not excuse a stale entry for an existing file")


def test_capability_undeclared_measured_crossing() -> None:
    measured = {("existing.py", 1, "tpen.unlisted")}
    try:
        _validate_inventory_admission(measured, {})
    except AssertionError as exc:
        message = str(exc)
        assert "undeclared tpen crossings" in message
        assert "remove the import per experiments/README.md" in message
        assert "declare it with a disposition" in message
        assert "stale sibling entry under rule (b)" in message
    else:
        raise AssertionError("every measured crossing must have a declared entry")
