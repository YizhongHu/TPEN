"""Final T2-delta controls for the SOAP P3 reviewer.

These tests intentionally contain expected refusals for gaps introduced or
exposed by the delta.  A red result is diagnostic evidence; this file does
not alter the implementation under review.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import uuid

import pytest

from experiments.atomistic.soap_pilot import composer
from experiments.atomistic.soap_pilot.p3_render import (
    P3RenderError,
    admit_row_and_command,
    finalize_eval_row,
    render_eval_template,
    render_r2_train_packet,
)
from experiments.atomistic.soap_pilot.rows import CheckpointEvidence, make_parent_join


SOURCE = "7b88f833babcb82148c8f975a08b53970838be6a"
ROOT = Path(__file__).resolve().parents[4]


def _execution_state(tmp_path: Path, *, kind: str = "train") -> dict[str, str]:
    cache = tmp_path / "uv-cache"
    cache.mkdir(exist_ok=True)
    return {
        "UV_CACHE_DIR": str(cache),
        "APPROVAL_ID": "soap-review-approval",
        "EXPECTED_QUEUE": "capacity" if kind == "train" else "debug",
    }


def _publication(checkpoint_dir: Path, model_bytes: bytes = b"soap100-model") -> CheckpointEvidence:
    checkpoint_dir.mkdir(parents=True)
    model = checkpoint_dir / "model.pt"
    model.write_bytes(model_bytes)
    complete = checkpoint_dir / "COMPLETE"
    complete.write_text("complete\n", encoding="utf-8")
    return CheckpointEvidence(
        100,
        "soap100-publication",
        hashlib.sha256(model_bytes).hexdigest(),
        True,
        {"model.pt": str(model), "COMPLETE": str(complete)},
    )


def _soap_eval_fixture(tmp_path: Path):
    train = render_r2_train_packet(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )["r2-soap-100"]
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap100", row=train.row)
    template = render_eval_template(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )
    return train, parent, template


def test_finalized_soap_eval_is_admitted_through_real_row_path(tmp_path: Path) -> None:
    """A valid final eval row must pass the same admission path as train rows."""

    train, parent, template = _soap_eval_fixture(tmp_path)
    checkpoint_dir = (
        tmp_path / train.row["experiment_name"] / train.row["sector"]
        / train.row["run_id"] / "checkpoints" / "step_000100"
    )
    publication = _publication(checkpoint_dir)
    eval_row = finalize_eval_row(
        template,
        train_row=train.row,
        parent=parent,
        publication=publication,
        checkpoint_dir=checkpoint_dir,
    )

    # This is the reviewer-designed expectation.  The current generic audit
    # is train-only and is expected to make this diagnostic test red.
    admitted = admit_row_and_command(
        eval_row, python="/usr/bin/python3", checkout_root=str(ROOT),
        execution_state=_execution_state(tmp_path, kind="eval"),
    )
    assert "experiments/atomistic/stage1/eval.py" in admitted.command


def test_train_admission_control_remains_green(tmp_path: Path) -> None:
    """The same gateway still admits a valid train packet."""

    train = render_r2_train_packet(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )["r2-soap-100"]
    admitted = admit_row_and_command(
        train.row, python="/usr/bin/python3", checkout_root=str(ROOT),
        execution_state=_execution_state(tmp_path),
    )
    assert "experiments/atomistic/stage1/train.py" in admitted.command


def test_recipe_freeze_is_independent_of_composer_source_at_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-import composer drift must still be rejected by the frozen gate."""

    original = composer._optimizer_and_trainer

    def drifted(profile):
        optimizer, trainer = original(profile)
        if profile.family == "soap":
            optimizer = {**optimizer, "lr": 0.123}
        return optimizer, trainer

    monkeypatch.setattr(composer, "_optimizer_and_trainer", drifted)

    # The already-imported module retains its original snapshot and therefore
    # catches this ordinary post-import source drift.
    from experiments.atomistic.soap_pilot import p3_render as imported_p3

    with pytest.raises(P3RenderError, match="recipe|drift|enumerated"):
        imported_p3.render_r2_train_packet(
            output_dir=tmp_path / "after-import" / "inputs", results_root=str(tmp_path / "after-import"),
            source_sha=SOURCE, device="cpu"
        )

    # Fresh import is intentionally isolated under a unique module key.  If
    # the snapshot is built from live composer output, this accepted packet is
    # the diagnostic red result: the expected independent refusal is missing.
    module_name = f"experiments.atomistic.soap_pilot._p3_render_probe_{uuid.uuid4().hex}"
    path = Path(imported_p3.__file__).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fresh
    try:
        spec.loader.exec_module(fresh)
        with pytest.raises(fresh.P3RenderError, match="recipe|drift|enumerated"):
            packet = fresh.render_r2_train_packet(
                output_dir=tmp_path / "before-import" / "inputs", results_root=str(tmp_path / "before-import"),
                source_sha=SOURCE, device="cpu"
            )
            fresh.admit_row_and_command(
                packet["r2-soap-100"].row, python="/usr/bin/python3", checkout_root=str(ROOT)
            )
    finally:
        sys.modules.pop(spec.name, None)


def test_eval_publication_is_bound_to_train_output_when_optional_metadata_absent(tmp_path: Path) -> None:
    """A sibling foreign output must not satisfy the actual SOAP parent join."""

    train, parent, template = _soap_eval_fixture(tmp_path)
    actual_root = tmp_path / train.row["experiment_name"] / train.row["sector"] / train.row["run_id"]
    foreign_checkpoint = (
        tmp_path / train.row["experiment_name"] / train.row["sector"]
        / "foreign-run" / "checkpoints" / "step_000100"
    )
    publication = _publication(foreign_checkpoint)

    # Deliberately omit parent_run_id and parent_output_dir: these are the
    # optional fields whose absence must not permit a foreign sibling output.
    with pytest.raises(P3RenderError, match="parent|output|publication|checkpoint"):
        finalize_eval_row(
            template,
            train_row=train.row,
            parent=parent,
            publication=publication,
            checkpoint_dir=foreign_checkpoint,
        )
    assert actual_root != foreign_checkpoint.parent.parent

