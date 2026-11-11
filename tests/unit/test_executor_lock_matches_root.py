"""Check that the executor lock follows the root project's locked versions."""

import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _packages(lock: dict) -> dict[str, str]:
    """Return package versions keyed by package name."""
    return {package["name"]: package["version"] for package in lock["package"]}


def test_executor_lock_matches_root_versions() -> None:
    """The executor dev group must track root-lock versions exactly.

    The executor project is intentionally separate, but its development
    dependencies must use the root lock's versions.  ``uv lock --check`` only
    validates one lock against its own project metadata, so it cannot detect
    drift between the two locks.
    """
    with (ROOT / "uv.lock").open("rb") as root_file:
        root_lock = tomllib.load(root_file)
    with (ROOT / "executor" / "uv.lock").open("rb") as executor_file:
        executor_lock = tomllib.load(executor_file)
    with (ROOT / "executor" / "pyproject.toml").open("rb") as project_file:
        executor_project = tomllib.load(project_file)

    root_packages = _packages(root_lock)
    executor_packages = _packages(executor_lock)
    mismatches: list[str] = []
    for name, executor_version in executor_packages.items():
        root_version = root_packages.get(name)
        if root_version is not None and executor_version != root_version:
            mismatches.append(f"{name}: executor {executor_version} vs root {root_version}")

    executor_only = sorted(set(executor_packages) - set(root_packages))
    if executor_only != ["tpen-executor"]:
        mismatches.append(f"unexpected executor-only packages: {executor_only!r}")

    for constraint in executor_lock["manifest"]["constraints"]:
        name = constraint["name"]
        specifier = constraint["specifier"]
        expected = specifier.removeprefix("==")
        root_version = root_packages.get(name)
        if root_version != expected:
            mismatches.append(f"{name}: executor {expected} vs root {root_version}")

    parsl_extra = executor_project["project"]["optional-dependencies"]["parsl"]
    if parsl_extra != ["parsl==2026.8.10"]:
        mismatches.append(f"parsl extra is not the approved pin: {parsl_extra!r}")
    for requirement in parsl_extra:
        name, separator, expected = requirement.partition("==")
        if not separator or not name or not expected:
            mismatches.append(f"invalid parsl extra requirement: {requirement!r}")
            continue
        executor_version = executor_packages.get(name)
        root_version = root_packages.get(name)
        if executor_version != expected:
            mismatches.append(f"{name}: executor lock {executor_version} vs extra {expected}")
        if root_version != expected:
            mismatches.append(f"{name}: root {root_version} vs extra {expected}")

    assert not mismatches, "executor/root lock mismatches: " + ", ".join(mismatches)
