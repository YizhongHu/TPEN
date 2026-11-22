"""Fail-closed path checks for every harness-side write."""

from __future__ import annotations

import os
from pathlib import Path

LAB_ROOT = "/n/netscratch/kozinsky_lab/Lab/rhu"


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _local_fake(path: str) -> bool:
    fake_root = os.environ.get("LOCAL_FAKE_ROOT", "")
    return os.environ.get("LOCAL_FAKE") == "1" and os.environ.get("DRY_RUN") == "1" and bool(fake_root) and _under(path, fake_root)


def _validate_storage_roots(node_tmp: str, run_root: str, task_root: str, harness_root: str) -> None:
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    user = os.environ.get("USER", "")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not user or not job_id:
        raise RuntimeError("USER and SLURM_JOB_ID are required before trusting storage roots")
    if not dry_run:
        expected = f"/scratch/{user}/{job_id}"
        if node_tmp != expected:
            raise RuntimeError(f"NODE_TMP must equal {expected}, got {node_tmp}")
    else:
        dry_node_tmp = os.environ.get("DRY_NODE_TMP", node_tmp)
        if not (_under(dry_node_tmp, LAB_ROOT) or _local_fake(dry_node_tmp)):
            raise RuntimeError(f"DRY_NODE_TMP is outside the approved root: {dry_node_tmp}")
    for name, path in (("HARNESS_ROOT", harness_root), ("RUN_ROOT", run_root), ("TASK_ROOT", task_root)):
        if not (_under(path, LAB_ROOT) or _local_fake(path)):
            raise RuntimeError(f"{name} is outside the approved root: {path}")


def allowed_roots() -> tuple[Path, ...]:
    """Return the only roots the job may write to."""

    node_tmp = os.environ.get("NODE_TMP")
    run_root = os.environ.get("RUN_ROOT")
    if not node_tmp or not run_root:
        raise RuntimeError("NODE_TMP and RUN_ROOT are required before any write")
    task_root = os.environ.get("TASK_ROOT", run_root)
    harness_root = os.environ.get("HARNESS_ROOT", task_root)
    _validate_storage_roots(node_tmp, run_root, task_root, harness_root)
    return (Path(node_tmp), Path(run_root), Path(task_root))


def assert_write_path(path: Path, *, allow_missing: bool = True) -> Path:
    """Check that *path* is below an allowed root without resolving interpreters."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise RuntimeError(f"relative write path rejected: {candidate}")
    # lexical containment is intentional: resolve() would follow a path that may
    # not exist yet and is not appropriate for interpreter paths.
    normalized = Path(os.path.normpath(str(candidate)))
    for root in allowed_roots():
        root_normalized = Path(os.path.normpath(str(root)))
        if normalized == root_normalized or root_normalized in normalized.parents:
            return candidate
    raise RuntimeError(f"write outside allowed roots: {candidate}")


def write_text(path: Path, text: str) -> None:
    """Assert, create the parent, then write one text artifact."""

    assert_write_path(path)
    parent = path.parent
    assert_write_path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
