"""Executor parity tests for core-owned public evidence fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "contracts" / "fixtures" / "core-run-evidence-v2.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_executor_decodes_each_core_owned_fixture_identically() -> None:
    payload = _fixture()
    evidence = CoreRunEvidence.from_mapping(payload)
    assert evidence.schema == payload["schema"]
    assert evidence.run_dir == payload["run_dir"]
    assert evidence.verdict == payload["verdict"]
    assert evidence.run_status == payload["run_status"]
    assert dict(evidence.checkpoint or {}) == payload["checkpoint"]
    assert evidence.refusal == payload["refusal"]
    assert dict(evidence.resolved_config_identity or {}) == payload["resolved_config_identity"]


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
        CoreRunEvidence.from_mapping(mutate(_fixture()))
