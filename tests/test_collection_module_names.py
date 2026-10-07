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


def _test_files() -> list[Path]:
    norecursedirs = _norecursedirs()
    test_files: list[Path] = []
    for current, directories, filenames in os.walk(REPO_ROOT):
        relative_current = Path(current).relative_to(REPO_ROOT)
        directories[:] = [
            directory
            for directory in directories
            if not _is_excluded(relative_current / directory, norecursedirs)
        ]
        test_files.extend(
            Path(current) / filename
            for filename in filenames
            if filename.startswith("test_") and filename.endswith(".py")
        )
    return sorted(test_files)


def _duplicate_basename_groups() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in _test_files():
        groups[path.name].append(path)
    return {name: paths for name, paths in groups.items() if len(paths) > 1}


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

    assert module_result.returncode == console_result.returncode == 0, (
        "pytest collection did not succeed consistently for the dynamically discovered "
        "duplicate-basename files\n"
        f"module invocation ({module_result.returncode}) stdout:\n{module_result.stdout}\n"
        f"module invocation stderr:\n{module_result.stderr}\n"
        f"console invocation ({console_result.returncode}) stdout:\n{console_result.stdout}\n"
        f"console invocation stderr:\n{console_result.stderr}"
    )
