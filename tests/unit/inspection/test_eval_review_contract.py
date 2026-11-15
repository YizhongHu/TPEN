"""Reviewer contracts for checkpoint-free evaluation evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.inspection import (
    EVAL_RUN_EVIDENCE_SCHEMA,
    EvalRunEvidence,
    inspect_eval_run,
)
from tpen.run_events import RUN_OUTCOME_SCHEMA


IDENTITY = {"schema": "tpen-config-identity-v1", "sha256": "a" * 64}


def _outcome(status: str, *, checkpoint_ref: object = None) -> dict[str, object]:
    """Return the closed wire shape emitted by the run-outcome writer."""

    return {
        "schema": RUN_OUTCOME_SCHEMA,
        "status": status,
        "run_dir": None,
        "error": None,
        "checkpoint_ref": checkpoint_ref,
        "next_iteration": None,
        "completed_updates": None,
        "target_iterations": None,
        "exception_type": None,
        "publication_stage": None,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_eval_sidecars(
    run_dir: Path,
    *,
    status: str = "completed",
    identity: object = IDENTITY,
    outcome: str | None = "completed",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "status.json", {"status": status})
    _write_json(run_dir / "run_start.json", {"resolved_config_identity": identity})
    if outcome is not None:
        _write_json(run_dir / "run-outcome.json", _outcome(outcome))


@pytest.mark.parametrize(
    "changes",
    [
        {"resolved_config_identity": None},
        {"outcome": None},
        {"run_status": "failed"},
        {"run_status": "failed", "outcome": "failed"},
        {"run_status": "completed", "outcome": "failed"},
    ],
)
def test_completed_from_mapping_requires_identity_and_matching_completed_status(
    changes: dict[str, object],
) -> None:
    """A completed record cannot be decoded from incomplete or disagreeing facts."""

    record: dict[str, object] = {
        "schema": EVAL_RUN_EVIDENCE_SCHEMA,
        "run_dir": "run",
        "verdict": "completed",
        "run_status": "completed",
        "outcome": "completed",
        "resolved_config_identity": IDENTITY,
        "refusal": None,
    }
    record.update(changes)

    with pytest.raises((TypeError, ValueError)):
        EvalRunEvidence.from_mapping(record)


def test_completed_from_mapping_round_trips() -> None:
    record = EvalRunEvidence(
        EVAL_RUN_EVIDENCE_SCHEMA,
        "run",
        "completed",
        "completed",
        "completed",
        IDENTITY,
        None,
    )

    assert EvalRunEvidence.from_mapping(record.to_dict()) == record


def test_symlink_cycle_is_refused_as_an_artifact_problem(tmp_path: Path) -> None:
    """A cyclic input path must not escape as Path.resolve's RuntimeError."""

    cycle = tmp_path / "cycle"
    cycle.symlink_to(cycle, target_is_directory=True)

    evidence = inspect_eval_run(cycle)

    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "run-dir-io-error"
    assert evidence.refusal["reason"]


@pytest.mark.parametrize("sidecar", ["status.json", "run_start.json", "run-outcome.json"])
@pytest.mark.parametrize("encoding", ["ordinary-json", "invalid-utf8"])
def test_malformed_sidecars_return_refusal(tmp_path: Path, sidecar: str, encoding: str) -> None:
    run_dir = tmp_path / "run"
    _write_eval_sidecars(run_dir)
    path = run_dir / sidecar
    if encoding == "ordinary-json":
        path.write_text("{\n", encoding="utf-8")
    else:
        path.write_bytes(b"\xff\xfe\xff")

    evidence = inspect_eval_run(run_dir)

    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"].startswith("malformed-")


def test_failed_status_without_outcome_is_failed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_eval_sidecars(run_dir, status="failed", outcome=None)

    evidence = inspect_eval_run(run_dir)

    assert evidence.verdict == "failed"
    assert evidence.run_status == "failed"


@pytest.mark.parametrize(
    ("status", "outcome", "verdict"),
    [
        ("failed", "failed", "failed"),
        ("unknown", "unknown", "incomplete"),
        ("refused", "refused", "incomplete"),
    ],
)
def test_matching_explicit_outcome_statuses_map_to_operational_verdict(
    tmp_path: Path, status: str, outcome: str, verdict: str
) -> None:
    run_dir = tmp_path / status
    _write_eval_sidecars(run_dir, status=status, outcome=outcome)

    evidence = inspect_eval_run(run_dir)

    assert evidence.verdict == verdict
    assert evidence.run_status == status
    assert evidence.outcome == outcome
