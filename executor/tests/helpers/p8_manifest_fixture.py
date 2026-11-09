"""Generator for the checked-in P8 cross-project manifest fixture."""

from __future__ import annotations

import sys
from pathlib import Path

from tpen_executor._codec import digest
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId
from tpen_executor.results import AcceptedSegment, ManifestAttempt, OutcomeReceipt, RunManifest, StartReceipt


GROUP_DIGEST = "76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf"


def cross_project_manifest() -> RunManifest:
    """Return the deterministic v2 manifest used by toolkit tests."""

    attempt_id = AttemptId("attempt-1")
    output_dir = "/runs/group-1/task-a/attempt-1"
    start = StartReceipt(
        "task-a",
        attempt_id,
        AllocationId("allocation-1"),
        BackendJobId("job-1"),
        GROUP_DIGEST,
        "2026-11-04T00:00:00Z",
        output_dir,
    )
    outcome = OutcomeReceipt(
        "task-a",
        attempt_id,
        GROUP_DIGEST,
        "2026-11-04T00:01:00Z",
        "completed",
        output_dir,
        digest({"energy": -2.0}),
    )
    attempt = ManifestAttempt("task-a", attempt_id, start, outcome, output_dir)
    segment = AcceptedSegment(
        "task-a",
        attempt_id,
        None,
        f"{output_dir}/core/checkpoint/step_000001",
        1,
        1,
    )
    return RunManifest("group-1", GROUP_DIGEST, (attempt,), (segment,))


def write_fixture(path: str | Path) -> None:
    """Write canonical JSON plus a terminal newline for a toolkit fixture."""

    Path(path).write_bytes(cross_project_manifest().encode() + b"\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python executor/tests/helpers/p8_manifest_fixture.py PATH")
    write_fixture(sys.argv[1])
