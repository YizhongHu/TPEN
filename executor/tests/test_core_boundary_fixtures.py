"""Executor parity tests for core-owned public evidence fixtures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from tpen.inspection import RunEvidence
from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATHS = (
    REPO_ROOT / "tests" / "contracts" / "fixtures" / "core-run-evidence-v2.json",
    REPO_ROOT / "tests" / "contracts" / "fixtures" / "core-run-evidence-v1-refused.json",
)
V3_FIXTURE_PATHS = (
    REPO_ROOT / "tests" / "contracts" / "fixtures" / "core-run-evidence-v3-planned-yield.json",
    REPO_ROOT / "tests" / "contracts" / "fixtures" / "core-run-evidence-v3-unknown.json",
)
CORE_REFUSAL_CODES = (
    "missing-run-dir", "missing-status", "multiple-status", "malformed-status", "status-io-error",
    "malformed-run-outcome", "status-outcome-mismatch", "missing-run-identity", "checkpoint-selection-mismatch",
    "missing-checkpoint-ref", "stale-checkpoint-ref", "model-only-checkpoint", "outcome-checkpoint-mismatch",
    "unpublished-checkpoint-ref", "unknown-checkpoint-profile", "malformed-run-identity", "missing-checkpoint",
    "incomplete-checkpoint", "no-train-resume-checkpoint", "checkpoint-io-error", "corrupt-checkpoint",
    "corrupt-publication-catalog", "torn-publication-catalog",
)


def _fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain_mapping_or_none(value: Mapping[str, object] | None) -> dict[str, object] | None:
    return None if value is None else dict(value)


@pytest.mark.parametrize("fixture_path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_executor_decodes_each_core_owned_fixture_identically(fixture_path: Path) -> None:
    payload = _fixture(fixture_path)
    evidence = CoreRunEvidence.from_mapping(payload)
    assert evidence.schema == payload["schema"]
    assert evidence.run_dir == payload["run_dir"]
    assert evidence.verdict == payload["verdict"]
    assert evidence.run_status == payload["run_status"]
    assert (evidence.checkpoint is None) is (payload["checkpoint"] is None)
    assert _plain_mapping_or_none(evidence.checkpoint) == payload["checkpoint"]
    assert _plain_mapping_or_none(evidence.refusal) == payload["refusal"]
    if payload["schema"] == "tpen.run-evidence/v2":
        assert "resolved_config_identity" in payload
        assert (evidence.resolved_config_identity is None) is (
            payload["resolved_config_identity"] is None
        )
        assert _plain_mapping_or_none(evidence.resolved_config_identity) == payload[
            "resolved_config_identity"
        ]
    else:
        assert payload["schema"] == "tpen.run-evidence/v1"
        assert "resolved_config_identity" not in payload
        assert evidence.resolved_config_identity is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: {**payload, "schema": "tpen.run-evidence/v999"},
        lambda payload: {**payload, "unexpected": True},
    ],
    ids=["unknown-schema", "extra-field"],
)
def test_executor_refuses_unknown_schema_and_extra_field(mutate) -> None:
    with pytest.raises(EvidenceUnavailable):
        CoreRunEvidence.from_mapping(mutate(_fixture(FIXTURE_PATHS[0])))


@pytest.mark.parametrize("fixture_path", V3_FIXTURE_PATHS, ids=lambda path: path.stem)
def test_executor_decodes_both_core_v3_fixtures(fixture_path: Path) -> None:
    payload = _fixture(fixture_path)
    evidence = CoreRunEvidence.from_mapping(payload)
    assert evidence.schema == "tpen.run-evidence/v3"
    assert evidence.outcome == payload["outcome"]
    assert evidence.checkpoint_ref == payload["checkpoint_ref"]
    assert evidence.continuation_eligible is (payload["outcome"] == "planned_yield")


def test_v3_decoder_refuses_status_outcome_mismatch() -> None:
    payload = _fixture(V3_FIXTURE_PATHS[0])
    payload["run_status"] = "completed"
    with pytest.raises(EvidenceUnavailable, match="run_status must equal outcome"):
        CoreRunEvidence.from_mapping(payload)


def test_v3_decoder_refuses_extra_key() -> None:
    payload = _fixture(V3_FIXTURE_PATHS[0])
    payload["unexpected"] = True
    with pytest.raises(EvidenceUnavailable, match="unexpected keys"):
        CoreRunEvidence.from_mapping(payload)


def test_v3_eligibility_requires_planned_yield() -> None:
    payload = _fixture(V3_FIXTURE_PATHS[0])
    payload["outcome"] = "completed"
    payload["run_status"] = "completed"
    payload["verdict"] = "completed"
    evidence = CoreRunEvidence.from_mapping(payload)
    assert evidence.continuation_eligible is False


@pytest.mark.parametrize("code", CORE_REFUSAL_CODES)
def test_v3_refusal_codes_match_core(code: str) -> None:
    payload = {
        "schema": "tpen.run-evidence/v3",
        "run_dir": "/normalized/core",
        "verdict": "refused",
        "run_status": "completed",
        "checkpoint": None,
        "refusal": {"code": code, "reason": 17},
        "resolved_config_identity": {"schema": "tpen-config-identity-v1", "sha256": "a" * 64},
        "outcome": "unknown",
        "checkpoint_ref": None,
    }
    core = RunEvidence.from_mapping(payload)
    executor = CoreRunEvidence.from_mapping(payload)
    assert dict(executor.refusal) == core.refusal
    assert executor.schema == core.schema
    assert executor.run_dir == core.run_dir
    assert executor.verdict == core.verdict
    assert executor.run_status == core.run_status
    assert executor.checkpoint is None
    assert dict(executor.resolved_config_identity) == core.resolved_config_identity
    assert executor.outcome == core.outcome
    assert executor.checkpoint_ref is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: {**payload, "checkpoint": []},
        lambda payload: {**payload, "checkpoint": {"unexpected": 1}},
        lambda payload: {**payload, "run_dir": 3},
        lambda payload: {**payload, "verdict": "completed", "checkpoint": None, "refusal": None, "outcome": "completed", "run_status": "completed"},
        lambda payload: {**payload, "verdict": "refused", "checkpoint": dict(payload["checkpoint"]), "refusal": {"code": "x", "reason": "y"}},
    ],
    ids=["nested-type", "nested-keys", "run-dir-type", "completed-without-checkpoint", "refused-with-checkpoint"],
)
def test_v3_malformed_cases_have_core_and_executor_same_verdict(mutate) -> None:
    payload = _fixture(V3_FIXTURE_PATHS[0])
    with pytest.raises(Exception):
        RunEvidence.from_mapping(mutate(payload))
    with pytest.raises(EvidenceUnavailable):
        CoreRunEvidence.from_mapping(mutate(payload))
