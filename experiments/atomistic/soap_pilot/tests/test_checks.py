"""Offline R1/Rev3 checker controls, including missing and tampered evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.atomistic.soap_pilot.checks import (
    Verdict,
    check_disconnected_optimizer_state,
    check_model_float64,
    check_named_outputs,
    check_resume_metric_keys,
    check_step_zero,
)


def test_resume_requires_equal_metric_key_sets() -> None:
    result = check_resume_metric_keys({1: {"logical_state": "a", "energy": 1.0}}, {1: {"logical_state": "a", "variance": 1.0}})
    assert result.verdict is Verdict.PRODUCT_FAILURE
    assert "uninterrupted-only" in result.detail


def test_missing_evidence_never_passes() -> None:
    assert check_model_float64(None, None).verdict is Verdict.UNMEASURED
    assert check_named_outputs(None).verdict is Verdict.UNMEASURED
    assert check_step_zero(None).verdict is Verdict.UNMEASURED


def test_step_zero_amendment_controls() -> None:
    result = check_step_zero({
        "energy": {"adam": 2.0, "soap": 2.0, "minsr64": 2.0, "minsr128": 2.0},
        "energy_variance": {"adam": 0.1, "soap": 0.1, "minsr64": 0.1, "minsr128": 0.1},
        "grad_norm": {"adam": 8.0, "soap": 8.0, "minsr64": 8.0, "minsr128": 8.0},
        "model_sha256": {"adam": "a", "soap": "a", "minsr64": "a", "minsr128": "a"},
        "construction": {"autograd": (8.0, 8.0), "qgt": (8.0, 8.0)},
    })
    assert result.passed


def test_disconnected_count_is_tensor_count() -> None:
    assert check_disconnected_optimizer_state({"n_no_optimizer_state": 16, "n_no_optimizer_state_elements": 492000}).passed
    assert check_disconnected_optimizer_state({"n_no_optimizer_state": 492000, "n_no_optimizer_state_elements": 16}).verdict is Verdict.PRODUCT_FAILURE


def test_named_output_hash_and_complete(tmp_path: Path) -> None:
    output = tmp_path / "metrics.jsonl"
    output.write_text("step=1\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    result = check_named_outputs({"complete": True, "files": [{"path": str(output), "sha256": digest}]}, tmp_path)
    assert result.passed
    output.write_text("tampered\n", encoding="utf-8")
    assert check_named_outputs({"complete": True, "files": [{"path": str(output), "sha256": digest}]}, tmp_path).verdict is Verdict.PRODUCT_FAILURE
