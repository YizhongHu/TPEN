"""Bounded T2 probes for the HI S6B-R2 qualification seam."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

completion = importlib.import_module(
    "experiments.atomistic.he-importance.full_horizon_completion"
)
qualification = importlib.import_module("experiments.polaris.qualify.qualification")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _delta(core: Path, mode: str) -> dict[str, object]:
    return {
        "attempt_core_dir": str(core),
        "attempt_id": core.parent.name,
        "group_digest": core.parent.parent.parent.name,
        "load": {
            "mode": mode,
            "path": None if mode == "fresh" else str(
                core.parent.parent.parent / "previous" / "core" / "checkpoint"
            ),
        },
        "schema": "tpen-executor/attempt-delta-v1",
        "task_id": core.parent.parent.name,
    }


def _completion(core: Path, row_id: str) -> dict[str, object]:
    return {
        "attempt_chain": [],
        "completed_at": "2026-11-25T14:00:00.000000Z",
        "completed_updates": 1,
        "core_acceptance_digest": "a" * 64,
        "final_attempt_id": core.parent.name,
        "final_checkpoint": {
            "content_id": "f" * 64,
            "path": str(core / "checkpoints" / "final"),
        },
        "group_digest": core.parent.parent.parent.name,
        "schema": "tpen-executor/full-horizon-completion-v1",
        "science_digest": row_id,
        "target_iterations": 1,
        "task_id": core.parent.parent.name,
    }


def _qualification_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    root = tmp_path / "root"
    root.mkdir()
    row_id = "ab" * 32
    core = root / "groups" / ("b" * 64) / "task-7" / "attempt-S" / "core"
    core.mkdir(parents=True)
    (core.parent.parent / "full-horizon-completion.json").write_bytes(
        _canonical_bytes(_completion(core, row_id))
    )
    campaign = root / "campaign"
    (campaign / "cells" / row_id).mkdir(parents=True)
    (campaign / "cells" / row_id / "selection_receipt.json").write_bytes(
        _canonical_bytes({"selection": {"selected": {"context": "O-raw"}}, "walkers": 1})
    )
    (core / "attempt_report.json").write_bytes(
        _canonical_bytes({"row_id": row_id, "attempt_delta": _delta(core, "fresh")})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(_canonical_bytes({"runtime_kind": "polaris-gpu-qualification", "output_root": str(campaign), "cell": {"context": "O-raw", "walkers": 1, "updates": 1, "checkpoint_every": 1}}))
    return campaign, manifest, core, row_id


def test_qualification_cell_complete_consumes_matching_real_full_h_record(tmp_path: Path) -> None:
    campaign, manifest, _core, _row = _qualification_fixture(tmp_path)
    evidence = tmp_path / "evidence"
    qualification.qualify(manifest, evidence, artifact_root=campaign.parent)
    verdict = json.loads((evidence / "qualification.json").read_text(encoding="utf-8"))
    assert next(row for row in verdict["predicates"] if row["name"] == "cell_complete")["status"] == "PASS"


def test_qualification_does_not_short_circuit_later_invalid_same_row_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, manifest, core, row = _qualification_fixture(tmp_path)
    invalid_core = core.parents[2] / "task-invalid" / "attempt-invalid" / "core"
    invalid_core.mkdir(parents=True)
    (invalid_core.parent.parent / "full-horizon-completion.json").write_bytes(_canonical_bytes({"schema": "malformed"}))
    invalid = invalid_core / "attempt_report.json"
    invalid.write_bytes(_canonical_bytes({"row_id": row, "attempt_delta": _delta(invalid_core, "fresh")}))
    real_rglob = Path.rglob

    def ordered_rglob(root: Path, pattern: str):
        if root == campaign.parent and pattern == "attempt_report.json":
            return iter([core / "attempt_report.json", invalid])
        return real_rglob(root, pattern)

    monkeypatch.setattr(Path, "rglob", ordered_rglob)
    with pytest.raises(completion.FullHorizonCompletionError):
        qualification.qualify(manifest, tmp_path / "evidence", artifact_root=campaign.parent)


def test_qualification_legacy_complete_survives_unrelated_delta_report(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    row = "row-a"
    (root / "cells" / row).mkdir(parents=True)
    (root / "cells" / row / "selection_receipt.json").write_bytes(_canonical_bytes({"selection": {"selected": {"context": "O-raw"}}}))
    (root / "Q" / row / "COMPLETE").parent.mkdir(parents=True)
    (root / "Q" / row / "COMPLETE").write_text("completed\n", encoding="utf-8")
    (root / "unrelated" / "attempt_report.json").parent.mkdir(parents=True)
    (root / "unrelated" / "attempt_report.json").write_bytes(_canonical_bytes({"row_id": "row-b"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(_canonical_bytes({"runtime_kind": "polaris-gpu-qualification", "output_root": str(root), "cell": {"context": "O-raw", "walkers": 1, "updates": 1, "checkpoint_every": 1}}))
    evidence = tmp_path / "evidence"
    qualification.qualify(manifest, evidence)
    verdict = json.loads((evidence / "qualification.json").read_text(encoding="utf-8"))
    assert next(row for row in verdict["predicates"] if row["name"] == "cell_complete")["status"] == "PASS"
