"""Build and validate an immutable control-project prefix."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from tpen_executor._codec import canonical_bytes, fields

from .application import BalsamControlRefusal


MANIFEST_SCHEMA = "tpen-balsam-control/prefix-manifest-v1"


def _inside(path: Path, parent: Path) -> bool:
    """Return whether path is equal to or below parent after resolution."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _file_digest(path: Path) -> str:
    """Hash one regular file without following a user-selected directory."""

    if not path.is_file() or path.is_symlink():
        raise BalsamControlRefusal(f"identity input is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    """Hash every regular file below a source root in stable path order."""

    if not root.is_dir() or root.is_symlink():
        raise BalsamControlRefusal(f"identity source root is unavailable: {root}")
    result = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        if path.is_symlink():
            raise BalsamControlRefusal(f"identity source is a symlink: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        result.update(len(relative).to_bytes(8, "big"))
        result.update(relative)
        result.update(len(content).to_bytes(8, "big"))
        result.update(content)
    return result.hexdigest()


def _source_build_digest(root: Path, *, include_lock: bool) -> str:
    """Hash only the files that define one source build."""

    result = hashlib.sha256()
    inputs = [root / "pyproject.toml"]
    if include_lock and (root / "uv.lock").exists():
        inputs.append(root / "uv.lock")
    inputs.append(root / "src")
    for path in inputs:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        result.update(len(relative).to_bytes(8, "big"))
        result.update(relative)
        if path.name == "src":
            content = bytes.fromhex(_tree_digest(path))
        else:
            content = path.read_bytes()
        result.update(len(content).to_bytes(8, "big"))
        result.update(content)
    return result.hexdigest()


def _executable_identity(path: Path) -> dict[str, str]:
    """Return the resolved executable path and its current bytes digest."""

    if not path.is_absolute():
        raise BalsamControlRefusal(f"qualified executable must be absolute: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BalsamControlRefusal(f"qualified executable is unavailable: {path}") from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise BalsamControlRefusal(f"qualified executable is unavailable: {path}")
    return {"path": str(resolved), "digest": _file_digest(resolved)}


def _selected_project_inputs(project_dir: Path, lock: Path) -> tuple[Path, Path]:
    """Resolve the project and require the supplied lock to be its uv.lock."""

    if not project_dir.is_absolute() or not project_dir.is_dir() or project_dir.is_symlink():
        raise BalsamControlRefusal(f"uv project must be an absolute directory: {project_dir}")
    selected_project = project_dir / "pyproject.toml"
    selected_lock = project_dir / "uv.lock"
    if lock.resolve() != selected_lock.resolve():
        raise BalsamControlRefusal("supplied lock is not the selected project uv.lock")
    return selected_project, selected_lock


def _validate_cache_dir(cache_dir: Path | None) -> Path:
    """Require an explicit, absolute cache outside the home directory."""

    if cache_dir is None or not cache_dir.is_absolute():
        raise BalsamControlRefusal("cache_dir must be an absolute non-Home directory")
    resolved = cache_dir.resolve()
    if _inside(resolved, Path.home()):
        raise BalsamControlRefusal("cache_dir must be outside the home directory")
    if cache_dir.exists() and cache_dir.is_symlink():
        raise BalsamControlRefusal("cache_dir must not be a symlink")
    return resolved


def prefix_manifest(
    project_dir: Path,
    lock: Path,
    *,
    python: Path,
    uv: Path,
    prefix: Path,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the content/version-bound manifest for one final prefix."""

    if not prefix.is_absolute() or _inside(prefix, Path.home()):
        raise BalsamControlRefusal("control prefix must be absolute and outside the home directory")
    selected_project, selected_lock = _selected_project_inputs(project_dir, lock)
    cache = _validate_cache_dir(cache_dir)
    executor_root = project_dir.parents[1].resolve()
    selected_prefix = prefix.resolve()
    if _inside(selected_prefix, project_dir.resolve()) or _inside(selected_prefix, executor_root):
        raise BalsamControlRefusal("prefix must not overlap project or executor inputs")
    if _inside(cache, project_dir.resolve()) or _inside(cache, executor_root):
        raise BalsamControlRefusal("cache_dir must not overlap project or executor inputs")
    if _inside(selected_prefix, cache) or _inside(cache, selected_prefix):
        raise BalsamControlRefusal("prefix and cache_dir must not overlap")
    if not (executor_root / "pyproject.toml").is_file() or not (executor_root / "src").is_dir():
        raise BalsamControlRefusal("executor source-build inputs are unavailable")
    return {
        "schema": MANIFEST_SCHEMA,
        "project_dir": str(project_dir.resolve()),
        "project_digest": _file_digest(selected_project),
        "lock_path": str(selected_lock.resolve()),
        "lock_digest": _file_digest(selected_lock),
        "executor_source_digest": _source_build_digest(executor_root, include_lock=True),
        "control_source_digest": _source_build_digest(project_dir, include_lock=True),
        "python": _executable_identity(python),
        "uv": _executable_identity(uv),
        "cache_dir": str(cache),
        "prefix": str(prefix.resolve()),
    }


def _read_manifest(path: Path) -> Mapping[str, Any]:
    """Read and validate one exact manifest."""

    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise BalsamControlRefusal("control prefix manifest is unreadable") from error
    fields(data, set(prefix_manifest_fields()), MANIFEST_SCHEMA, "prefix manifest")
    return data


def prefix_manifest_fields() -> tuple[str, ...]:
    """Return the exact manifest field names."""

    return (
        "schema",
        "project_dir",
        "project_digest",
        "lock_path",
        "lock_digest",
        "executor_source_digest",
        "control_source_digest",
        "python",
        "uv",
        "cache_dir",
        "prefix",
    )


def build_prefix(
    project_dir: Path,
    lock: Path,
    *,
    python: Path,
    uv: Path,
    prefix: Path,
    cache_dir: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    """Build one final prefix, refusing drift or implicit toolchain changes.

    Existing prefixes are reusable only when their manifest is byte-for-byte
    equivalent to the requested identity.  A non-empty unmanifested prefix is
    never adopted, and managed Python downloads are disabled explicitly.
    """

    manifest = prefix_manifest(
        project_dir,
        lock,
        python=python,
        uv=uv,
        prefix=prefix,
        cache_dir=cache_dir,
    )
    manifest_path = prefix / "tpen-control-manifest.json"
    if prefix.exists():
        if not prefix.is_dir():
            raise BalsamControlRefusal("control prefix is not a directory")
        if manifest_path.exists():
            current = _read_manifest(manifest_path)
            if current != manifest:
                raise BalsamControlRefusal("control prefix identity drifted")
            return dict(manifest)
        if any(prefix.iterdir()):
            raise BalsamControlRefusal("unmanifested control prefix cannot be reused")
    else:
        prefix.mkdir(parents=True)
    selected_cache = Path(manifest["cache_dir"])
    selected_cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("XDG_CACHE_HOME", None)
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(prefix),
            "UV_CACHE_DIR": str(selected_cache),
            "UV_NO_MANAGED_PYTHON": "1",
        }
    )
    completed = runner(
        (
            str(uv),
            "sync",
            "--locked",
            "--no-editable",
            "--project",
            str(project_dir),
            "--python",
            str(python),
        ),
        check=False,
        capture_output=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise BalsamControlRefusal("control prefix build failed; interactive resolution is required")
    manifest_path.write_bytes(canonical_bytes(manifest))
    return dict(manifest)
