"""Tracked-lock resolution helpers shared by staging and the job."""

from __future__ import annotations


def expected_torch_version(lock: dict, extra: str) -> str:
    """Return the unique Linux Torch version declared by one optional extra."""
    project = next((p for p in lock.get("package", []) if p.get("source", {}).get("editable") == "."), None)
    entries = (project or {}).get("optional-dependencies", {}).get(extra, [])
    torch = [entry for entry in entries if entry.get("name") == "torch" and (not entry.get("marker") or "sys_platform != 'darwin'" in entry.get("marker", ""))]
    if len(torch) != 1 or "version" not in torch[0]:
        raise ValueError(f"lock has no unique Linux torch resolution for extra {extra}")
    return torch[0]["version"]
