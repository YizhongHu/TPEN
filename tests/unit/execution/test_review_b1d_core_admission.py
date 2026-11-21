"""Unrun T2 proposals: trust and identity at the admission mapping boundary."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
from typeguard import suppress_type_checks

import tpen.inspection as inspection
from tpen.checkpoint.hashing import resolved_config_identity
from tpen.execution.core_admission import evaluate_core_admission
from tpen.execution.recovery import (
    ScienceIntent,
    SuccessorAdmissionDecodeRefusal,
    decode_successor_admission_request,
    select_successor_admission,
)

from .test_continuation_admission import _checkpoint, _request
from .test_core_admission import (
    _context,
    _planned_yield_run,
    _write_outcome,
    _write_run_start,
)


def _assert_unknown(mapping):
    assert mapping is not None
    assert tuple(mapping[key] for key in ("outcome", "admission", "checkpoint", "quiescence")) == (
        "unknown", "unknown", "unknown", "unknown"
    )
    assert mapping["profile"] is None


def test_review_request_for_another_leaf_is_unknown(tmp_path: Path) -> None:
    """Two individually valid leaves must not be combined into an admission."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _planned_yield_run(first)
    second_ref = _planned_yield_run(second)
    request = _request(second_ref)
    assert inspection.inspect_run(first).continuation_eligible
    assert inspection.inspect_run(second).continuation_eligible
    assert select_successor_admission(
        ScienceIntent({"input": "cx04"}, target_iterations=8), request
    ).accepted
    _assert_unknown(evaluate_core_admission(_context(first, request)))


def test_review_changed_run_identity_is_unknown(tmp_path: Path) -> None:
    """Fresh inspection identity, not just request self-consistency, binds."""
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)
    identity = resolved_config_identity({"study": "a-different-study"})
    (tmp_path / "run_start.json").write_text(
        json.dumps({"resolved_config_identity": identity}), encoding="utf-8"
    )
    evidence = inspection.inspect_run(tmp_path)
    assert evidence.continuation_eligible
    assert evidence.resolved_config_identity != request.predecessor.resolved_config_identity
    _assert_unknown(evaluate_core_admission(_context(tmp_path, request)))


def test_review_predecessor_config_mismatch_is_unknown(tmp_path: Path) -> None:
    """A selector identity refusal is uncertainty at the mapping boundary."""
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)
    request = replace(request, predecessor=replace(
        request.predecessor,
        resolved_config_identity=resolved_config_identity({"study": "other"}),
    ))
    _assert_unknown(evaluate_core_admission(_context(tmp_path, request)))


@pytest.mark.parametrize("outcome", ["completed", "failed", "refused"])
def test_review_contradictory_terminal_outcome_is_unknown(tmp_path: Path, outcome: str) -> None:
    """The primitive's refused inspection must dominate its retained outcome."""
    ref = _checkpoint(tmp_path)
    _write_run_start(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"unknown"}\n', encoding="utf-8")
    _write_outcome(tmp_path, outcome, ref)
    evidence = inspection.inspect_run(tmp_path)
    assert evidence.verdict == "refused"
    assert evidence.refusal["code"] == "status-outcome-mismatch"
    assert evidence.outcome == outcome
    _assert_unknown(evaluate_core_admission(_context(tmp_path)))


@pytest.mark.parametrize("status", ["completed", "failed"])
@pytest.mark.parametrize("artifact", ["missing", "invalid-json"])
def test_review_untyped_terminal_evidence_is_unknown(tmp_path: Path, status: str, artifact: str) -> None:
    """Legacy status cannot replace a trustworthy typed outcome."""
    _checkpoint(tmp_path)
    _write_run_start(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    if artifact == "invalid-json":
        (tmp_path / "run-outcome.json").write_text("{broken", encoding="utf-8")
    evidence = inspection.inspect_run(tmp_path)
    assert evidence.schema != inspection.RUN_EVIDENCE_SCHEMA_V3
    assert evidence.verdict == status
    _assert_unknown(evaluate_core_admission(_context(tmp_path)))


def test_review_semantic_wire_decode_failure_is_unknown_without_typeguard(tmp_path: Path) -> None:
    """Production cannot rely on pytest instrumentation catching assignment errors."""
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)
    wire = request.to_dict()
    wire["core_evidence"]["checkpoint"] = None
    assert isinstance(decode_successor_admission_request(wire), SuccessorAdmissionDecodeRefusal)
    context = replace(_context(tmp_path, request), successor_request=wire)
    # Intentional local exception to typecheck=on: exercise the uninstrumented
    # production branch; Typeguard can intercept the union assignment earlier.
    with suppress_type_checks():
        result = evaluate_core_admission(context)
    _assert_unknown(result)


def test_review_catalog_io_failure_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject only the filesystem failure, retaining the real inspection path."""
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)

    def unavailable(_path):
        raise PermissionError("temporary catalog access failure")

    monkeypatch.setattr(inspection, "read_publications", unavailable)
    evidence = inspection.inspect_run(tmp_path)
    assert evidence.verdict == "refused"
    assert evidence.refusal["code"] == "corrupt-publication-catalog"
    _assert_unknown(evaluate_core_admission(_context(tmp_path, request)))


def test_review_missing_catalog_is_refused_without_writes(tmp_path: Path) -> None:
    """Green control: real missing publication remains a typed refusal."""
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)
    (tmp_path / "checkpoints" / "publications.jsonl").unlink()
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = evaluate_core_admission(_context(tmp_path, request))
    assert result["outcome"] == "successor"
    assert result["admission"] == "refused"
    assert result["reason"] == "unpublished-checkpoint-ref"
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
