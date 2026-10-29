"""Validate the immutable evidence references behind scientific admission.

This checks receipt identity and completeness, not the authority or honesty of
the people producing receipts. The operator's release and independent review
remain external workflow decisions; a bare caller-supplied pass is insufficient.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments.one_electron_variance.protocol import DESIGN_SHA256

CONTROL_CHECKS = (
    "ordinary_scalar_oracles", "stress_scalar_oracles", "imperfect_moments",
    "sampler_radial_tails", "wrong_target_rejected", "wrong_kinetic_rejected",
    "missing_potential_rejected", "synthetic_rise_detected", "synthetic_null_detected",
    "strict_model_restore", "complete_trajectory_records",
)
EVIDENCE_KINDS = (
    "instrument_controls", "gpu_correctness", "independent_verification",
    "timing", "memory", "release",
)


def _checked_path(reference: dict, base: Path) -> Path:
    path = Path(reference["path"])
    if not path.is_absolute():
        path = base / path
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != reference["sha256"]:
        raise ValueError(f"admission evidence changed: {path}")
    return path


def validate_science_admission(path: Path, source_sha: str) -> dict:
    """Require exact-source, hashed control, release, review and resource receipts."""
    gate = json.loads(path.read_text())
    if (gate.get("schema") != "one-electron-admission/v1"
            or gate.get("source_sha") != source_sha
            or gate.get("released_main_sha") != source_sha
            or gate.get("design_sha256") != DESIGN_SHA256):
        raise ValueError("science requires exact released-source admission")
    references = gate.get("evidence", {})
    if set(references) != set(EVIDENCE_KINDS):
        raise ValueError("science admission lacks required evidence kinds")
    receipts = {}
    for kind in EVIDENCE_KINDS:
        receipt_path = _checked_path(references[kind], path.parent)
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("kind") != kind or receipt.get("source_sha") != source_sha
                or receipt.get("status") != "passed" or not receipt.get("artifacts")):
            raise ValueError(f"invalid {kind} receipt")
        for reference in receipt["artifacts"]:
            _checked_path(reference, receipt_path.parent)
        receipts[kind] = receipt
    controls = receipts["instrument_controls"]
    if any(controls.get("checks", {}).get(key) is not True for key in CONTROL_CHECKS):
        raise ValueError("instrument controls are incomplete")
    variance = float(controls["max_exact_ordinary_grid_variance"])
    if not math.isfinite(variance) or not 0 <= variance <= 1e-10:
        raise ValueError("exact-reference variance did not pass qualification")
    independent = receipts["independent_verification"]
    if (not independent.get("verifier_id") or not gate.get("writer_id")
            or independent["verifier_id"] == gate["writer_id"]):
        raise ValueError("independent verification requires a distinct named verifier")
    if receipts["release"].get("ref") != "refs/heads/main":
        raise ValueError("release receipt must identify main")
    if any(receipts["timing"].get("systems", {}).get(system, {}).get(
            "fits_six_hours_with_twenty_percent_margin") is not True
           for system in ("hooke", "hydrogen")):
        raise ValueError("both systems must pass the fixed timing budget")
    if (receipts["memory"].get("full_a100_verified") is not True
            or receipts["memory"].get("full_4096_batch_and_stream_verified") is not True):
        raise ValueError("memory admission must cover full batches and streams on a full A100")
    return {"gate": gate, "receipts": receipts, "max_exact_ordinary_grid_variance": variance}
