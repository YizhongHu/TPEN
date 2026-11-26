"""Offline R1/Rev3 checker controls, including missing and tampered evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.atomistic.soap_pilot.checks import (
    Verdict,
    check_callbacks,
    check_checkpoint,
    check_config_identity,
    check_disconnected_optimizer_state,
    check_eval_metrics,
    check_model_float64,
    check_named_outputs,
    check_parent_join,
    check_path_containment,
    check_profile,
    check_rendered_diff,
    check_resume_metric_keys,
    check_resume_parity,
    check_seed_domain,
    check_state_cursor_tolerance,
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


def test_callback_policy_and_semantic_refusals() -> None:
    callbacks = [
        {"_target_": "tpen.callback.OptimizerState"},
        {"_target_": "tpen.callback.TrainPhaseTiming", "accelerator_synchronize": True},
        {"_target_": "tpen.callback.TrainStepTiming", "accelerator_synchronize": True},
        {"_target_": "tpen.callback.RunTiming", "accelerator_synchronize": True},
    ]
    assert check_callbacks({"callbacks": callbacks}, family="soap", train_timing_synchronize=True).passed
    assert check_callbacks({"callbacks": callbacks}, family="minsr64", train_timing_synchronize=True).verdict is Verdict.PRODUCT_FAILURE
    bad = [dict(item) for item in callbacks]
    bad[-1]["accelerator_synchronize"] = False
    assert check_callbacks({"callbacks": bad}, family="soap", train_timing_synchronize=True).verdict is Verdict.PRODUCT_FAILURE


def test_parent_profile_config_checkpoint_hash_controls() -> None:
    parent = {"campaign_id": "c", "invocation_id": "i", "manifest_hash": "m", "train_row_hash": "r", "row_id": "id", "source_sha": "s", "rendered_config_sha256": "x"}
    assert check_parent_join(parent, dict(parent)).passed
    foreign = dict(parent, row_id="foreign")
    assert check_parent_join(parent, foreign).verdict is Verdict.PRODUCT_FAILURE
    assert check_profile({"family": "soap", "updates": 100}, {"family": "soap", "updates": 100}).passed
    assert check_profile({"family": "soap"}, {"family": "soap", "extra": 1}).verdict is Verdict.PRODUCT_FAILURE
    raw = b"runtime: {dtype: float64}\n"
    assert check_config_identity({"runtime": {"dtype": "float64"}}, rendered_bytes=raw, expected_sha256=hashlib.sha256(raw).hexdigest()).passed
    assert check_config_identity({"runtime": {"dtype": "float64"}}, rendered_bytes=b"tamper", expected_sha256=hashlib.sha256(raw).hexdigest()).verdict is Verdict.PRODUCT_FAILURE
    evidence = {"step": 100, "complete": True, "publication_id": "pub", "model_sha256": "a" * 64, "files": {"model.pt": "model.pt", "COMPLETE": "COMPLETE"}}
    assert check_checkpoint(100, evidence).passed
    assert check_checkpoint(20, evidence).verdict is Verdict.PRODUCT_FAILURE


def test_seed_metric_nan_dtype_state_cursor_and_tolerance_controls() -> None:
    rows = [{"run_id": "adam", "seed": 1, "sampler_seed": 2}, {"run_id": "soap", "seed": 1, "sampler_seed": 2}]
    assert check_seed_domain(rows, matched_groups=[{"adam", "soap"}]).passed
    assert check_seed_domain(rows).verdict is Verdict.PRODUCT_FAILURE
    assert check_resume_parity({1: {"state": "a", "cursor": 1, "energy": 1.0}}, {1: {"state": "a", "cursor": 1, "energy": 1.0000001}}, tolerances={"energy": (1e-5, 0.0)}).passed
    assert check_resume_parity({1: {"state": "a", "cursor": 1, "energy": 1.0}}, {1: {"state": "a", "cursor": 2, "energy": 1.0}}).verdict is Verdict.PRODUCT_FAILURE
    assert check_resume_parity({1: {"state": "a", "cursor": 1, "energy": float("nan")}}, {1: {"state": "a", "cursor": 1, "energy": 1.0}}).verdict is Verdict.PRODUCT_FAILURE
    assert check_state_cursor_tolerance({"state": "a", "cursor": 1, "energy": 1.001}, {"state": "a", "cursor": 1, "energy": 1.0}, tolerances={"energy": (1e-2, 0.0)}).passed
    assert check_state_cursor_tolerance({"state": "b", "cursor": 1}, {"state": "a", "cursor": 1}).verdict is Verdict.PRODUCT_FAILURE


def test_metric_key_extra_missing_eval_metric_and_render_diff_controls() -> None:
    assert check_resume_metric_keys({1: {"logical_state": "a", "energy": 1.0}}, {1: {"logical_state": "a", "energy": 1.0, "extra": 2.0}}).verdict is Verdict.PRODUCT_FAILURE
    assert check_eval_metrics({"configured_metric_keys": ["energy", "variance"], "metric_keys": ["energy"]}).verdict is Verdict.PRODUCT_FAILURE
    assert check_eval_metrics({"configured_metric_keys": ["energy"], "metric_keys": ["energy"]}).passed
    config = {"runtime": {"dtype": "float64"}}
    assert check_rendered_diff(config, {"complete": True, "unexpected": [], "runtime_dtype": "float64"}).passed
    assert check_rendered_diff(config, {"complete": True, "unexpected": ["callbacks"]}).verdict is Verdict.PRODUCT_FAILURE
    assert check_rendered_diff(config, {"complete": False, "unexpected": []}).verdict is Verdict.UNMEASURED


def test_missing_complete_writer_and_path_escape_controls(tmp_path: Path) -> None:
    output = tmp_path / "metrics.jsonl"
    output.write_text("step=1\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert check_named_outputs({"complete": False, "files": [{"path": str(output), "sha256": digest}]}, tmp_path).verdict is Verdict.UNMEASURED
    assert check_named_outputs({"complete": True, "files": [{"path": str(output), "sha256": digest}], "present_undeclared": ["other"]}, tmp_path).verdict is Verdict.PRODUCT_FAILURE
    assert check_path_containment(output, tmp_path).passed
    assert check_path_containment(tmp_path / "sibling" / "x", tmp_path).verdict is Verdict.HARNESS_INVALID
