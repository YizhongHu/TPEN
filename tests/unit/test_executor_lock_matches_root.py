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

    root_packages = _packages(root_lock)
    executor_packages = _packages(executor_lock)
    mismatches: list[str] = []
    for name, executor_version in executor_packages.items():
        root_version = root_packages.get(name)
        if root_version is not None and executor_version != root_version:
            mismatches.append(f"{name}: executor {executor_version} vs root {root_version}")

    for constraint in executor_lock["manifest"]["constraints"]:
        name = constraint["name"]
        specifier = constraint["specifier"]
        expected = specifier.removeprefix("==")
        root_version = root_packages.get(name)
        if root_version != expected:
            mismatches.append(f"{name}: executor {expected} vs root {root_version}")

    assert not mismatches, "executor/root lock mismatches: " + ", ".join(mismatches)
