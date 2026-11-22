"""Unrun delta proposals for the remaining admission classification edges."""

from dataclasses import replace
from pathlib import Path

import pytest

import tpen.execution.recovery as recovery
import tpen.inspection as inspection
from tpen.execution.core_admission import evaluate_core_admission
from tpen.execution.recovery import AdmissionRefusalCode, ScienceIntent

from .test_continuation_admission import _request
from .test_core_admission import _context, _planned_yield_run
from .test_review_b1d_core_admission import _assert_unknown


def test_delta_selector_identity_refusal_is_unknown(tmp_path: Path) -> None:
    """Binding checks cannot replace handling the selector's identity result."""
    ref = _planned_yield_run(tmp_path)
    request = replace(_request(ref), target_iterations=ref.next_iteration - 1)
    assert inspection.inspect_run(tmp_path).continuation_eligible
    result = recovery.select_successor_admission(
        ScienceIntent({"input": "cx04"}, target_iterations=request.target_iterations), request
    )
    assert result.refusal is not None
    assert result.refusal.code is AdmissionRefusalCode.IDENTITY_MISMATCH
    _assert_unknown(evaluate_core_admission(_context(tmp_path, request)))


def test_delta_selector_catalog_io_failure_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful inspection does not guarantee the subsequent read works."""
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)
    reads = []

    def unavailable(path):
        reads.append(path)
        raise PermissionError("selector catalog read unavailable")

    # Patch only the selector's imported read alias, not inspection's alias.
    monkeypatch.setattr(recovery, "read_publications", unavailable)
    assert inspection.inspect_run(tmp_path).continuation_eligible
    result = evaluate_core_admission(_context(tmp_path, request))
    assert len(reads) == 1
    _assert_unknown(result)


def test_delta_missing_model_remains_typed_stale_refusal(tmp_path: Path) -> None:
    """A known absent payload is a stale publication, not an access error."""
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)
    (ref.checkpoint_dir / "model.pt").unlink()
    evidence = inspection.inspect_run(tmp_path)
    assert evidence.verdict == "refused"
    assert evidence.refusal["code"] == "stale-checkpoint-ref"
    result = evaluate_core_admission(_context(tmp_path, request))
    assert result is not None
    assert (result["outcome"], result["admission"], result["checkpoint"]) == (
        "successor", "refused", "invalid"
    )
    assert result["reason"] == "stale-checkpoint-ref"


def test_delta_inspection_unavailable_does_not_change_wire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Green control for the intentionally process-local diagnostic flag."""
    _planned_yield_run(tmp_path)

    def unavailable(_path):
        raise PermissionError("inspection catalog read unavailable")

    monkeypatch.setattr(inspection, "read_publications", unavailable)
    evidence = inspection.inspect_run(tmp_path)
    assert evidence.inspection_unavailable is True
    assert evidence.refusal["code"] == "corrupt-publication-catalog"
    wire = evidence.to_dict()
    assert "inspection_unavailable" not in wire
    assert wire == replace(evidence, inspection_unavailable=False).to_dict()
    decoded = inspection.RunEvidence.from_mapping(wire)
    assert decoded.to_dict() == wire
    assert decoded.inspection_unavailable is False
