"""Bounded T2 probes for the HI S6B-R2 qualification seam."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

from tpen_executor.configured_program import AttemptDelta, ConfiguredLoad
from tpen_executor._codec import canonical_bytes

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "executor" / "tests"))
from test_b1r2c_full_horizon import _publish, _single_fixture  # noqa: E402


completion = importlib.import_module(
    "experiments.atomistic.he-importance.full_horizon_completion"
)
qualification = importlib.import_module("experiments.polaris.qualify.qualification")


def _delta(core: Path, mode: str) -> AttemptDelta:
    checkpoint = None if mode == "fresh" else str(core.parent.parent.parent / "previous" / "core" / "checkpoint")
    return AttemptDelta(
        str(core), ConfiguredLoad(mode, checkpoint), core.parent.parent.parent.name,
        core.parent.parent.name, core.parent.name,
    )


def _qualification_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    root = tmp_path / "root"
    root.mkdir()
    store, group, fence, verdict, task = _single_fixture(root)
    _publish(store, group, fence, verdict, task)
    row_id = task.science_digest
    campaign = root / "campaign"
    (campaign / "cells" / row_id).mkdir(parents=True)
    (campaign / "cells" / row_id / "selection_receipt.json").write_bytes(canonical_bytes({"selection": {"selected": {"context": "O-raw"}}, "walkers": 1}))
    delta = _delta(Path(fence.output_dir) / "core", "fresh")
    report = {"row_id": row_id, "attempt_delta": json.loads(delta.canonical_bytes)}
    (Path(fence.output_dir) / "core" / "attempt_report.json").write_bytes(canonical_bytes(report))
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(canonical_bytes({"runtime_kind": "polaris-gpu-qualification", "output_root": str(campaign), "cell": {"context": "O-raw", "walkers": 1, "updates": 1, "checkpoint_every": 1}}))
    return campaign, manifest, Path(fence.output_dir) / "core", row_id


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
    (invalid_core.parent.parent / "full-horizon-completion.json").write_bytes(canonical_bytes({"schema": "malformed"}))
    invalid = invalid_core / "attempt_report.json"
    invalid.write_bytes(canonical_bytes({"row_id": row, "attempt_delta": json.loads(_delta(invalid_core, "fresh").canonical_bytes)}))
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
    (root / "cells" / row / "selection_receipt.json").write_bytes(canonical_bytes({"selection": {"selected": {"context": "O-raw"}}}))
    (root / "Q" / row / "COMPLETE").parent.mkdir(parents=True)
    (root / "Q" / row / "COMPLETE").write_text("completed\n", encoding="utf-8")
    (root / "unrelated" / "attempt_report.json").parent.mkdir(parents=True)
    (root / "unrelated" / "attempt_report.json").write_bytes(canonical_bytes({"row_id": "row-b"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(canonical_bytes({"runtime_kind": "polaris-gpu-qualification", "output_root": str(root), "cell": {"context": "O-raw", "walkers": 1, "updates": 1, "checkpoint_every": 1}}))
    evidence = tmp_path / "evidence"
    qualification.qualify(manifest, evidence)
    verdict = json.loads((evidence / "qualification.json").read_text(encoding="utf-8"))
    assert next(row for row in verdict["predicates"] if row["name"] == "cell_complete")["status"] == "PASS"
