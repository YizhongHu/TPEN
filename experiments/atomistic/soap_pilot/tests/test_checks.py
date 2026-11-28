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


def test_step_zero_requires_exact_families_and_constructions() -> None:
    """The clean four-family fixture passes; each family deletion is unmeasured."""

    clean = {
        "energy": {family: 2.0 for family in ("adam", "soap", "minsr64", "minsr128")},
        "energy_variance": {family: 0.1 for family in ("adam", "soap", "minsr64", "minsr128")},
        "grad_norm": {family: 8.0 for family in ("adam", "soap", "minsr64", "minsr128")},
        "model_sha256": {family: "a" for family in ("adam", "soap", "minsr64", "minsr128")},
        "construction": {"autograd": (8.0, 8.0), "qgt": (8.0, 8.0)},
    }
    assert check_step_zero(clean).passed
    for family in ("adam", "soap", "minsr64", "minsr128"):
        missing = {key: dict(value) for key, value in clean.items() if key != "construction"}
        for values in missing.values():
            values.pop(family)
        missing["construction"] = dict(clean["construction"])
        assert check_step_zero(missing).verdict is Verdict.UNMEASURED

    malformed = {key: dict(value) if isinstance(value, dict) else value for key, value in clean.items()}
    malformed["energy"]["unexpected"] = 2.0
    assert check_step_zero(malformed).verdict is Verdict.HARNESS_INVALID
    malformed_construction = {key: dict(value) if isinstance(value, dict) else value for key, value in clean.items()}
    malformed_construction["construction"] = {"autograd": (8.0, 8.0)}
    assert check_step_zero(malformed_construction).verdict is Verdict.HARNESS_INVALID


def test_disconnected_count_is_tensor_count() -> None:
    records = [
        {"step": 100, "namespace": "train/optimizer_state/embedding", "metrics": {"n_no_optimizer_state": 8, "n_no_optimizer_state_elements": 492000}},
        {"step": 100, "namespace": "train/optimizer_state/readout", "metrics": {"n_no_optimizer_state": 8, "n_no_optimizer_state_elements": 1}},
    ]
    assert check_disconnected_optimizer_state(records, groups=("embedding", "readout"), terminal_step=100).passed
    wrong = [dict(record, metrics={"n_no_optimizer_state": 492000}) for record in records]
    assert check_disconnected_optimizer_state(wrong, groups=("embedding", "readout"), terminal_step=100).verdict is Verdict.PRODUCT_FAILURE


def test_empty_group_claims_cannot_prove_parameter_bearing_group_empty() -> None:
    records = [
        {
            "step": 100,
            "namespace": "train/optimizer_state/embedding",
            "metrics": {"n_no_optimizer_state": 16},
        }
    ]
    groups = ("embedding", "factors")

    # Without a complete inventory, the caller's claim must not turn an
    # omitted parameter-bearing group into a vacuous PASS.
    without_inventory = check_disconnected_optimizer_state(
        records,
        groups=groups,
        terminal_step=100,
        empty_groups=("factors",),
    )
    assert without_inventory.verdict is not Verdict.PASS

    # A complete inventory that says the claimed group has parameters must
    # reject the contradictory empty-group claim as well.
    against_inventory = check_disconnected_optimizer_state(
        records,
        groups=groups,
        terminal_step=100,
        group_inventory={
            "embedding": {"parameters": 1},
            "factors": {"parameters": 1},
        },
        empty_groups=("factors",),
    )
    assert against_inventory.verdict is not Verdict.PASS


def _optimizer_records(step: int, *, groups: tuple[str, ...] = ("embedding", "readout")) -> list[dict[str, object]]:
    """Make compact tensor-count records for coordinate and duplicate controls."""

    return [
        {
            "step": step,
            "namespace": f"train/optimizer_state/{group}",
            "metrics": {"n_no_optimizer_state": 8},
        }
        for group in groups
    ]


def test_partial_converted_records_never_fall_forward_to_exact_step() -> None:
    partial = [
        _optimizer_records(99, groups=("embedding",))[0],
        *_optimizer_records(100),
    ]
    result = check_disconnected_optimizer_state(
        partial, groups=("embedding", "readout"), terminal_step=100
    )
    assert result.verdict is Verdict.UNMEASURED
    assert "fallback is forbidden" in result.detail

    assert check_disconnected_optimizer_state(
        _optimizer_records(99), groups=("embedding", "readout"), terminal_step=100
    ).passed

    both = [*_optimizer_records(99), *_optimizer_records(100)]
    assert check_disconnected_optimizer_state(
        both, groups=("embedding", "readout"), terminal_step=100
    ).verdict is Verdict.HARNESS_INVALID


def test_duplicate_optimizer_records_are_harness_invalid_before_selection() -> None:
    duplicate_and_missing = [
        *_optimizer_records(99, groups=("embedding",)),
        *_optimizer_records(99, groups=("embedding",)),
    ]
    result = check_disconnected_optimizer_state(
        duplicate_and_missing,
        groups=("embedding", "readout"),
        terminal_step=100,
    )
    assert result.verdict is Verdict.HARNESS_INVALID
    assert "duplicate" in result.detail


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
