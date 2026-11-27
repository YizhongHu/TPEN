"""Build and validate an immutable control-project prefix."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix_manifest(project: Path, lock: Path, executor_artifact: Path, *, python: Path, uv: Path, prefix: Path) -> dict[str, Any]:
    """Build the content/version-bound manifest for one final prefix."""

    if not prefix.is_absolute() or _inside(prefix, Path.home()):
        raise BalsamControlRefusal("control prefix must be absolute and outside the home directory")
    for path in (python, uv):
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise BalsamControlRefusal(f"qualified executable is unavailable: {path}")
    return {
        "schema": MANIFEST_SCHEMA,
        "project_digest": _file_digest(project),
        "lock_digest": _file_digest(lock),
        "executor_artifact_digest": _file_digest(executor_artifact),
        "python": str(python),
        "uv": str(uv),
        "prefix": str(prefix),
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
        "project_digest",
        "lock_digest",
        "executor_artifact_digest",
        "python",
        "uv",
        "prefix",
    )


def build_prefix(
    project: Path,
    lock: Path,
    executor_artifact: Path,
    *,
    python: Path,
    uv: Path,
    prefix: Path,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    """Build one final prefix, refusing drift or implicit toolchain changes.

    Existing prefixes are reusable only when their manifest is byte-for-byte
    equivalent to the requested identity.  A non-empty unmanifested prefix is
    never adopted, and managed Python downloads are disabled explicitly.
    """

    manifest = prefix_manifest(project, lock, executor_artifact, python=python, uv=uv, prefix=prefix)
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
    environment = dict(os.environ)
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(prefix),
            "UV_NO_MANAGED_PYTHON": "1",
        }
    )
    completed = runner(
        (str(uv), "sync", "--locked", "--no-editable", "--project", str(project), "--python", str(python)),
        check=False,
        capture_output=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise BalsamControlRefusal("control prefix build failed; interactive resolution is required")
    manifest_path.write_bytes(canonical_bytes(manifest))
    return dict(manifest)
