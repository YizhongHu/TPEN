"""Build phase-2 checkpoint evidence from explicitly named phase-1 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA, HE_H2_EVAL_CHECKPOINTS, HOOKE_SYSTEMS,
    MANIFEST_SCHEMA, SMOKE_EVAL_ROSTER_MARKER, bind_evaluation_manifest,
    hooke_profile_selection, manifest_hash, smoke_eval_roster,
)
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.checkpoint.artifact import require_complete_checkpoint_dir


def build_evidence(document: Mapping[str, Any]) -> dict[str, Any]:
    """Hash complete checkpoints selected by the phase-1 rows' protocol.

    Parameters
    ----------
    document
        Phase-1 manifest, including its content hash.

    Returns
    -------
    dict
        The protocol's checkpoint-evidence envelope. No run is modified.
    """
    rows = document.get("rows")
    if (document.get("schema") != MANIFEST_SCHEMA or document.get("phase") != 1
            or not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) for row in rows)):
        raise ValueError("expected a phase-1 manifest")
    if document.get("manifest_hash") != manifest_hash(rows):
        raise ValueError("phase-1 manifest_hash mismatch")
    if any(row.get("evaluation_roster") not in {None, SMOKE_EVAL_ROSTER_MARKER}
           for row in rows):
        raise ValueError("unknown evaluation_roster marker")
    train = [row for row in rows if row.get("kind") == "train"]
    rosters = {}
    for row in train:
        if row.get("evaluation_roster") == SMOKE_EVAL_ROSTER_MARKER:
            roster = smoke_eval_roster(row)
        elif row["system"] in HOOKE_SYSTEMS:
            roster = hooke_profile_selection(row["science_profile"], row["system"])[2]
        elif row["system"] in {"he", "h2"}:
            roster = HE_H2_EVAL_CHECKPOINTS
        else:
            raise ValueError("unsupported evaluation system")
        rosters[row["row_id"]] = roster
    checkpoints = sorted({step for roster in rosters.values() for step in roster})
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": document["manifest_hash"],
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row]) for row in train
        },
        "checkpoints": {
            f"{row['system']}:{row['lineage']}:{step}": "0" * 64
            for row in train for step in rosters[row["row_id"]]
        },
    }
    # Ask the protocol to validate identities and the complete roster before
    # touching any run directory. Placeholder digests are never published.
    bind_evaluation_manifest(phase1_rows=rows, checkpoints=checkpoints,
                             checkpoint_evidence=evidence)
    for row in train:
        run_dir = resolve_run_dir(row)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        smoke = row.get("evaluation_roster") == SMOKE_EVAL_ROSTER_MARKER
        expected_status = "planned_yield" if smoke else "completed"
        if status.get("status") != expected_status:
            raise ValueError(f"{run_dir}: expected status {expected_status}")
        if smoke:
            outcome = json.loads((run_dir / "run-outcome.json").read_text(encoding="utf-8"))
            stop_at = row["execution"]["stop_at"]
            if (outcome.get("status") != "planned_yield"
                    or outcome.get("completed_updates") != stop_at
                    or outcome.get("next_iteration") != stop_at):
                raise ValueError(f"{run_dir}: planned_yield does not match stop_at")
        for step in rosters[row["row_id"]]:
            checkpoint = require_complete_checkpoint_dir(
                run_dir / "checkpoints" / f"step_{step:06d}",
            )
            with (checkpoint / "model.pt").open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            evidence["checkpoints"][f"{row['system']}:{row['lineage']}:{step}"] = digest
    return evidence


def main() -> int:
    """Write checkpoint evidence without launching evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = build_evidence(json.loads(args.phase1_manifest.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
