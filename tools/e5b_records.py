"""Durable replace publication for E5 qualification markers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from tpen.execution.config import canonical_payload_bytes


def write_replace_json(path: Path, data: Mapping[str, Any]) -> None:
    """Replace a marker after syncing its bytes, then sync its directory.

    E5 status markers may be updated across starts. They do not use E3's
    exclusive receipt publication, which must preserve the first receipt.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(canonical_payload_bytes(data) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
