"""Behaviour controls for the core admission mapping boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpen.checkpoint.hashing import resolved_config_identity
from tpen.execution import core_admission
from tpen.execution.core_admission import (
    CORE_ADMISSION_SCHEMA,
    CoreAdmissionContextV1,
    encode_core_admission,
    evaluate_core_admission,
)
from tpen.execution.recovery import SuccessorAdmissionRequest

from .test_continuation_admission import _checkpoint, _request


MUTANTS = (
    (
        "planned-yield-to-completed",
        "tpen/execution/core_admission.py",
        'if evidence.outcome == "completed":',
        'if evidence.outcome in {"completed", "planned_yield"}:',
        "test_planned_yield_admits_real_published_checkpoint",
    ),
    (
        "profile-refusal-to-admitted",
        "tpen/execution/core_admission.py",
        'if profile_result.verdict == "refused":',
        'if profile_result.verdict == "supported":',
        "test_profile_refusal_is_a_typed_successor_refusal",
    ),
    (
        "ignore-stale-or-missing-reference",
        "tpen/execution/core_admission.py",
        '"unpublished-checkpoint-ref",',
        '"not-unpublished-checkpoint-ref",',
        "test_stale_inspection_is_not_admitted_from_request_bytes",
    ),
    (
        "exception-to-refused",
        "tpen/execution/core_admission.py",
        'return _unknown(f"inspection-error:{type(exc).__name__}")',
        'return _mapping("successor", "refused", "invalid", None, "exception")',
        "test_inspection_exception_is_unknown",
    ),
    (
        "failed-to-unknown",
        "tpen/execution/core_admission.py",
        'if evidence.outcome in {"failed", "refused"}:',
        'if evidence.outcome == "refused":',
        "test_failed_maps_to_terminal_refused",
    ),
    (
        "request-binding-bypass",
        "tpen/execution/core_admission.py",
        "if not _request_binds_to_inspection(context, evidence, request):",
        "if True:",
        "test_review_request_for_another_leaf_is_unknown",
    ),
    (
        "inspection-validity-bypass",
        "tpen/execution/core_admission.py",
        'if evidence.verdict == "refused" or evidence.refusal is not None:',
        "if False:",
        "test_review_contradictory_terminal_outcome_is_unknown",
    ),
    (
        "typed-v3-only-bypass",
        "tpen/execution/core_admission.py",
        'if evidence.schema != RUN_EVIDENCE_SCHEMA_V3:',
        "if False:",
        "test_review_untyped_terminal_evidence_is_unknown",
    ),
    (
        "unavailable-to-refusal",
        "tpen/execution/core_admission.py",
        "if evidence.inspection_unavailable:",
        "if False:",
        "test_review_catalog_io_failure_is_unknown",
    ),
    (
        "quiescence-not-unknown",
        "tpen/execution/core_admission.py",
        '"quiescence": "unknown",',
        '"quiescence": "valid",',
        "test_canonical_mapping_has_executor_unknown_quiescence",
    ),
)


def _identity() -> dict[str, str]:
    return resolved_config_identity({"study": "cx04"})


def _write_run_start(run_dir: Path) -> None:
    run_dir.joinpath("run_start.json").write_text(
        json.dumps({"resolved_config_identity": _identity()}) + "\n", encoding="utf-8"
    )


def _write_outcome(run_dir: Path, status: str, ref: object | None = None) -> None:
    ref_mapping = None if ref is None else ref.to_dict()  # type: ignore[union-attr]
    data = {
        "schema": "tpen.run-outcome/v1",
        "status": status,
        "run_dir": str(run_dir),
        "error": None,
        "checkpoint_ref": ref_mapping,
        "next_iteration": None if ref is None else ref.next_iteration,  # type: ignore[union-attr]
        "completed_updates": None if ref is None else ref.completed_updates,  # type: ignore[union-attr]
        "target_iterations": 8,
        "exception_type": None,
        "publication_stage": None,
    }
    run_dir.joinpath("run-outcome.json").write_text(json.dumps(data) + "\n", encoding="utf-8")


def _context(
    run_dir: Path | None,
    request: SuccessorAdmissionRequest | None = None,
    *,
    config: dict[str, object] | None = None,
    task_id: str = "task-1",
) -> CoreAdmissionContextV1:
    digest = _identity()["sha256"]
    return CoreAdmissionContextV1(
        task_id=task_id,
        lineage_id="lineage-1",
        core_run_dir=run_dir,
        science_input_digest=(request.science_input_digest if request is not None else digest),
        successor_request=request,
        unresolved_config={"study": "cx04"} if config is None else config,
        profile_name="tpen.native-recovery/v1",
    )


def _planned_yield_run(tmp_path: Path):
    ref = _checkpoint(tmp_path)
    _write_run_start(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"planned_yield"}\n', encoding="utf-8")
    _write_outcome(tmp_path, "planned_yield", ref)
    return ref


def test_no_core_leaf_returns_none_for_start_fresh() -> None:
    assert evaluate_core_admission(_context(None)) is None


def test_completed_maps_to_completed(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    _write_run_start(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    _write_outcome(tmp_path, "completed", ref)
    mapping = evaluate_core_admission(_context(tmp_path))
    assert mapping is not None
    assert mapping["outcome"] == "completed"
    assert mapping["admission"] == "unknown"
    assert mapping["checkpoint"] == "unknown"


def test_planned_yield_admits_real_published_checkpoint(tmp_path: Path) -> None:
    ref = _planned_yield_run(tmp_path)
    mapping = evaluate_core_admission(_context(tmp_path, _request(ref)))
    assert mapping is not None
    assert mapping["outcome"] == "successor"
    assert mapping["admission"] == "admitted"
    assert mapping["checkpoint"] == "valid"
    assert mapping["profile"]["verdict"] == "supported"


def test_profile_refusal_is_a_typed_successor_refusal(tmp_path: Path) -> None:
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)
    mapping = evaluate_core_admission(
        _context(tmp_path, request, config={"callbacks": [{"_target_": "not.allowed"}]})
    )
    assert mapping is not None
    assert mapping["outcome"] == "successor"
    assert mapping["admission"] == "refused"
    assert mapping["checkpoint"] == "invalid"
    assert mapping["profile"]["refusal"]["code"] == "unsupported-callback"
    assert mapping["reason"] == "unsupported-callback"


def test_stale_inspection_is_not_admitted_from_request_bytes(tmp_path: Path) -> None:
    ref = _planned_yield_run(tmp_path)
    request = _request(ref)
    (tmp_path / "checkpoints" / "publications.jsonl").unlink()
    mapping = evaluate_core_admission(_context(tmp_path, request))
    assert mapping == {
        "schema": CORE_ADMISSION_SCHEMA,
        "outcome": "successor",
        "admission": "refused",
        "checkpoint": "invalid",
        "quiescence": "unknown",
        "profile": None,
        "reason": "unpublished-checkpoint-ref",
    }


def test_inspection_exception_is_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(core_admission, "inspect_run", lambda _path: (_ for _ in ()).throw(OSError("boom")))
    mapping = evaluate_core_admission(_context(tmp_path))
    assert mapping is not None
    assert mapping["outcome"] == "unknown"
    assert mapping["admission"] == "unknown"
    assert mapping["reason"] == "inspection-error:OSError"


def test_failed_maps_to_terminal_refused(tmp_path: Path) -> None:
    _write_run_start(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"failed"}\n', encoding="utf-8")
    _write_outcome(tmp_path, "failed")
    mapping = evaluate_core_admission(_context(tmp_path))
    assert mapping is not None
    assert mapping["outcome"] == "terminal"
    assert mapping["admission"] == "refused"
    assert mapping["checkpoint"] == "absent"


def test_typed_unknown_maps_all_unknown(tmp_path: Path) -> None:
    _write_run_start(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"unknown"}\n', encoding="utf-8")
    _write_outcome(tmp_path, "unknown")
    mapping = evaluate_core_admission(_context(tmp_path))
    assert mapping is not None
    assert (mapping["outcome"], mapping["admission"], mapping["checkpoint"]) == (
        "unknown",
        "unknown",
        "unknown",
    )


def test_malformed_or_untyped_evidence_is_unknown(tmp_path: Path) -> None:
    ref = _checkpoint(tmp_path)
    _write_run_start(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"planned_yield"}\n', encoding="utf-8")
    (tmp_path / "run-outcome.json").write_text("{\"schema\":\"tpen.run-outcome/v1\"}\n", encoding="utf-8")
    mapping = evaluate_core_admission(_context(tmp_path, _request(ref)))
    assert mapping is not None
    assert mapping["outcome"] == "unknown"
    assert mapping["admission"] == "unknown"
    assert mapping["checkpoint"] == "unknown"


def test_identity_mismatch_is_unknown(tmp_path: Path) -> None:
    ref = _planned_yield_run(tmp_path)
    mapping = evaluate_core_admission(_context(tmp_path, _request(ref), task_id="other-task"))
    assert mapping is not None
    assert mapping["outcome"] == "unknown"
    assert mapping["reason"] == "identity-mismatch"


def test_canonical_mapping_has_executor_unknown_quiescence(tmp_path: Path) -> None:
    _write_run_start(tmp_path)
    (tmp_path / "status.json").write_text('{"status":"unknown"}\n', encoding="utf-8")
    _write_outcome(tmp_path, "unknown")
    mapping = evaluate_core_admission(_context(tmp_path))
    assert mapping is not None
    assert encode_core_admission(mapping) == json.dumps(
        mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert mapping["schema"] == CORE_ADMISSION_SCHEMA
    assert mapping["quiescence"] == "unknown"


def test_mutant_table_has_unique_compile_valid_anchors_and_existing_killers() -> None:
    source = inspect.getsource(core_admission)
    review_source = Path(__file__).with_name("test_review_b1d_core_admission.py").read_text()
    test_sources = Path(__file__).read_text() + review_source
    for name, target, before, after, killer in MUTANTS:
        assert target == "tpen/execution/core_admission.py"
        assert source.count(before) == 1, name
        compile(source.replace(before, after), target, "exec")
        assert f"def {killer}(" in test_sources, killer

