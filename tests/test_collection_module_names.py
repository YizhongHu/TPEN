from __future__ import annotations

from collections import defaultdict
import fnmatch
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTEST_DEFAULT_TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")
ALWAYS_EXCLUDED_DIRS = frozenset(
    {".git", ".venv", ".paseo", ".agent-worktrees", ".agent-scratch"}
)


def _norecursedirs() -> tuple[tuple[str, ...], ...]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = config["tool"]["pytest"]["ini_options"].get("norecursedirs", [])
    return tuple(tuple(Path(entry).parts) for entry in entries)


def _is_excluded(relative: Path, norecursedirs: tuple[tuple[str, ...], ...]) -> bool:
    parts = relative.parts
    if any(part in ALWAYS_EXCLUDED_DIRS for part in parts):
        return True
    if parts[:2] == (".claude", "worktrees"):
        return True

    for pattern_parts in norecursedirs:
        if len(pattern_parts) == 1:
            if any(fnmatch.fnmatch(part, pattern_parts[0]) for part in parts):
                return True
        elif parts[: len(pattern_parts)] == pattern_parts:
            return True
    return False


def _is_environment_root(path: Path) -> bool:
    """Return whether a directory contains a Python environment marker."""

    return (path / "pyvenv.cfg").is_file() or (path / "conda-meta" / "history").is_file()


def _test_files() -> list[Path]:
    norecursedirs = _norecursedirs()
    test_files: list[Path] = []
    for current, directories, filenames in os.walk(REPO_ROOT):
        current_path = Path(current)
        relative_current = Path(current).relative_to(REPO_ROOT)
        directories[:] = [
            directory
            for directory in directories
            if not _is_excluded(relative_current / directory, norecursedirs)
            and not _is_environment_root(current_path / directory)
        ]
        test_files.extend(
            Path(current) / filename
            for filename in filenames
            # Keep this census aligned with pytest's default python_files.
            if any(fnmatch.fnmatch(filename, pattern) for pattern in PYTEST_DEFAULT_TEST_FILE_PATTERNS)
        )
    return sorted(test_files)


def _duplicate_basename_groups() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in _test_files():
        groups[path.name].append(path)
    if not groups:
        raise AssertionError(
            "Duplicate-basename walk found no test_*.py files; "
            "the repository scan is disabled."
        )
    return {name: paths for name, paths in groups.items() if len(paths) > 1}


def _is_package_rooted(path: Path) -> bool:
    """Return whether a test file is below a regular package directory."""

    directory = path.parent
    while directory != REPO_ROOT:
        if REPO_ROOT not in directory.parents:
            return False
        if not (directory / "__init__.py").is_file():
            return False
        if directory.parent == REPO_ROOT:
            return True
        directory = directory.parent
    return False


def _non_package_rooted_duplicate_groups(
    duplicate_groups: dict[str, list[Path]],
) -> dict[str, list[Path]]:
    """Keep duplicate groups whose selected files are all outside packages."""

    result: dict[str, list[Path]] = {}
    for name, paths in duplicate_groups.items():
        selected = [path for path in paths if not _is_package_rooted(path)]
        if len(selected) > 1:
            result[name] = selected
    return result


def test_duplicate_test_module_names_collect_without_collision() -> None:
    duplicate_groups = _duplicate_basename_groups()
    if not duplicate_groups:
        pytest.skip(
            "No duplicate test-file basenames remain; there are no collision targets to collect."
        )

    duplicate_files = [
        path
        for paths in sorted(duplicate_groups.values(), key=lambda paths: paths[0].name)
        for path in paths
    ]
    selection = [
        "--collect-only",
        "-q",
        *(str(path.relative_to(REPO_ROOT)) for path in duplicate_files),
    ]
    module_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *selection,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert module_result.returncode == 0, (
        "pytest module invocation could not collect the dynamically discovered "
        "duplicate-basename files\n"
        f"stdout:\n{module_result.stdout}\n"
        f"stderr:\n{module_result.stderr}"
    )
    console_pytest = Path(sys.executable).parent / "pytest"
    if not console_pytest.is_file():
        pytest.skip(
            f"pytest console script is not available beside {sys.executable}; "
            "cannot verify console/module collection agreement."
        )
    console_result = subprocess.run(
        [str(console_pytest), *selection],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert console_result.returncode == 0, (
        "pytest console invocation could not collect the dynamically discovered "
        "duplicate-basename files\n"
        f"console invocation ({console_result.returncode}) stdout:\n{console_result.stdout}\n"
        f"console invocation stderr:\n{console_result.stderr}"
    )

    non_package_groups = _non_package_rooted_duplicate_groups(duplicate_groups)
    if not non_package_groups:
        classification = ", ".join(
            f"{name}: {sum(not _is_package_rooted(path) for path in paths)}"
            f"/{len(paths)} outside regular packages"
            for name, paths in sorted(duplicate_groups.items())
        )
        pytest.skip(
            "No duplicate-basename group has at least two files outside regular "
            "packages; the pythonpath-sensitive console arm has no collision "
            f"targets (evidence: {classification})."
        )
    non_package_files = [
        path
        for paths in sorted(non_package_groups.values(), key=lambda paths: paths[0].name)
        for path in paths
    ]
    non_package_selection = [
        "--collect-only",
        "-q",
        *(str(path.relative_to(REPO_ROOT)) for path in non_package_files),
    ]
    non_package_console_result = subprocess.run(
        [str(console_pytest), *non_package_selection],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert non_package_console_result.returncode == 0, (
        "pytest console invocation could not collect duplicate files outside "
        "regular packages; this arm protects repository-root discovery when "
        "pythonpath = [\".\"] is absent\n"
        f"console invocation ({non_package_console_result.returncode}) stdout:\n"
        f"{non_package_console_result.stdout}\n"
        f"console invocation stderr:\n{non_package_console_result.stderr}"
    )


def _put(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _configure_collection_tree(root: Path) -> None:
    _put(
        root,
        "pyproject.toml",
        '[tool.pytest.ini_options]\n'
        'pythonpath = ["."]\n'
        "consider_namespace_packages = true\n"
        "norecursedirs = []\n",
    )


@pytest.mark.parametrize(
    "env_name, marker",
    [
        (".venv-gpu", "pyvenv.cfg"),
        (".venv-rocm", "pyvenv.cfg"),
        ("custom-env", "pyvenv.cfg"),
        ("conda-env", "conda-meta/history"),
    ],
)
def test_duplicate_walk_excludes_environments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    marker: str,
) -> None:
    _configure_collection_tree(tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    expected = [
        _put(tmp_path, f"{directory}/test_review.py", "def test_ok(): pass\n")
        for directory in ("suite_a", "suite_b")
    ]
    _put(tmp_path, f"{env_name}/{marker}")
    for vendor in ("vendor_a", "vendor_b"):
        _put(
            tmp_path,
            f"{env_name}/lib/python3.12/site-packages/{vendor}/test_review.py",
            'raise RuntimeError("virtualenv test must not be imported")\n',
        )

    try:
        groups = _duplicate_basename_groups()
    except pytest.skip.Exception as exc:
        pytest.fail(f"environment exclusion regression test skipped: {exc}")
    assert groups == {"test_review.py": expected}


def test_namespace_gap_remains_a_root_sensitive_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_collection_tree(tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    _put(tmp_path, "tests/__init__.py")
    _put(tmp_path, "tests/conftest.py")
    _put(tmp_path, "tests/test_anchor.py", "def test_ok(): pass\n")
    _put(tmp_path, "experiments/pkg/__init__.py")
    expected = [
        _put(tmp_path, f"experiments/pkg/{leaf}/test_anchor.py", "def test_ok(): pass\n")
        for leaf in ("left", "right")
    ]

    try:
        groups = _duplicate_basename_groups()
        sensitive_groups = _non_package_rooted_duplicate_groups(groups)
    except pytest.skip.Exception as exc:
        pytest.fail(f"namespace-gap regression test skipped: {exc}")
    assert sensitive_groups == {
        "test_anchor.py": expected,
    }


def test_duplicate_pytest_default_suffix_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_collection_tree(tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    expected = [
        _put(tmp_path, f"{directory}/foo_test.py", "def test_ok(): pass\n")
        for directory in ("suite_a", "suite_b")
    ]

    assert _duplicate_basename_groups() == {"foo_test.py": expected}


def test_real_norecursedirs_excludes_hidden_checkout(
    tmp_path: Path,
) -> None:
    real_config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    norecursedirs = real_config["tool"]["pytest"]["ini_options"]["norecursedirs"]
    _put(
        tmp_path,
        "pyproject.toml",
        "[tool.pytest.ini_options]\n"
        'pythonpath = ["."]\n'
        "consider_namespace_packages = true\n"
        f"norecursedirs = {norecursedirs!r}\n",
    )
    _put(tmp_path, "tests/__init__.py")
    _put(tmp_path, ".claude/worktrees/agent-x/tests/__init__.py")
    _put(tmp_path, "tests/test_anchor.py", "def test_ok(): pass\n")
    _put(tmp_path, ".claude/worktrees/agent-x/tests/test_anchor.py", "def test_hidden(): pass\n")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.parametrize("marker", ["pyvenv.cfg", "conda-meta/history"])
def test_environment_marker_at_repo_root_does_not_silence_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    _configure_collection_tree(tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    expected = [
        _put(tmp_path, f"{name}/test_review.py", "def test_ok(): pass\n")
        for name in ("suite_a", "suite_b")
    ]
    _put(tmp_path, marker)

    if not (Path(sys.executable).parent / "pytest").is_file():
        # This environment condition is unrelated to a root marker hiding duplicates.
        pytest.skip("console script unavailable; cannot drive the full guard")

    try:
        test_duplicate_test_module_names_collect_without_collision()
    except pytest.skip.Exception as exc:
        pytest.fail(f"root marker hid real duplicate targets: {exc}")

    assert _duplicate_basename_groups() == {"test_review.py": expected}
