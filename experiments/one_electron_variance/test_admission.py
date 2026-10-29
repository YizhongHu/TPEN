"""Missing, changed or non-independent evidence cannot admit scientific output."""

import hashlib
import json

import pytest

from experiments.one_electron_variance.admission import (
    CONTROL_CHECKS, EVIDENCE_KINDS, validate_science_admission,
)
from experiments.one_electron_variance.protocol import DESIGN_SHA256
from experiments.one_electron_variance.analysis import _verify_sampling_receipt


def _reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _gate(tmp_path, *, omit_control=None, verifier="independent-test-fixture"):
    # Deliberately synthetic fixtures, never an operational admission receipt.
    raw = tmp_path / "synthetic-raw-evidence.txt"
    raw.write_text("synthetic unit-test receipt, not a qualification\n")
    evidence = {}
    for kind in EVIDENCE_KINDS:
        receipt = {"kind": kind, "source_sha": "a" * 40, "status": "passed",
                   "artifacts": [_reference(raw)]}
        if kind == "instrument_controls":
            receipt.update(checks={key: True for key in CONTROL_CHECKS if key != omit_control},
                           max_exact_ordinary_grid_variance=1e-16)
        if kind == "independent_verification":
            receipt["verifier_id"] = verifier
        if kind == "release":
            receipt["ref"] = "refs/heads/main"
        if kind == "timing":
            receipt["systems"] = {system: {"fits_six_hours_with_twenty_percent_margin": True}
                                  for system in ("hooke", "hydrogen")}
        if kind == "memory":
            receipt.update(full_a100_verified=True, full_4096_batch_and_stream_verified=True)
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(receipt))
        evidence[kind] = _reference(path)
    path = tmp_path / "admission.json"
    path.write_text(json.dumps({"schema": "one-electron-admission/v1", "source_sha": "a" * 40,
                               "released_main_sha": "a" * 40, "design_sha256": DESIGN_SHA256,
                               "writer_id": "writer-test-fixture", "evidence": evidence}))
    return path


def test_bound_receipts_and_changed_evidence(tmp_path):
    gate = _gate(tmp_path)
    assert validate_science_admission(gate, "a" * 40)["max_exact_ordinary_grid_variance"] == 1e-16
    (tmp_path / "synthetic-raw-evidence.txt").write_text("changed after admission\n")
    with pytest.raises(ValueError, match="evidence changed"):
        validate_science_admission(gate, "a" * 40)


@pytest.mark.parametrize("kind", CONTROL_CHECKS)
def test_each_instrument_control_is_required(tmp_path, kind):
    with pytest.raises(ValueError, match="controls are incomplete"):
        validate_science_admission(_gate(tmp_path, omit_control=kind), "a" * 40)


def test_self_verification_and_unreleased_source_are_refused(tmp_path):
    gate = _gate(tmp_path, verifier="writer-test-fixture")
    with pytest.raises(ValueError, match="distinct named verifier"):
        validate_science_admission(gate, "a" * 40)
    with pytest.raises(ValueError, match="released-source"):
        validate_science_admission(gate, "b" * 40)


def test_bare_pass_does_not_establish_qualification(tmp_path):
    gate = tmp_path / "bare-pass.json"
    gate.write_text(json.dumps({"status": "passed", "instrument_controls": "passed",
                               "source_sha": "a" * 40, "max_exact_ordinary_grid_variance": 1e-16}))
    with pytest.raises(ValueError, match="released-source"):
        validate_science_admission(gate, "a" * 40)


@pytest.mark.parametrize("mutation", ["seed", "burn_in", "draw_stride", "missing_draw"])
def test_sampler_receipt_binds_observed_settings(tmp_path, mutation):
    receipt = {"schema": "sampler_trajectory_diagnostics/v1", "n_walkers": 4096,
               "draw_stride": 20, "sampler_burn_in": 1000, "proposal_scale": .5,
               "discarded_draws": [], "retained_draws": [
                   {"seed": 1100000, "collection_index": i, "region_index": i,
                    "n_walkers": 4096, "draw_stride": 20, "burn_in": 1000,
                    "proposal_scale": .5, "transition_count": 81920} for i in range(64)]}
    path = tmp_path / "sampler_trajectory_diagnostics.json"
    path.write_text(json.dumps(receipt))
    _verify_sampling_receipt(tmp_path, 1100000)
    if mutation == "missing_draw":
        receipt["retained_draws"].pop()
    else:
        receipt["retained_draws"][0][mutation] += 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="sampler"):
        _verify_sampling_receipt(tmp_path, 1100000)
