"""Reviewer round-one controls for P3 admission and fail-closed policy.

These tests intentionally include expected refusals for gaps found during
review.  A red result is useful diagnostic evidence; this module does not fix
the implementation under review.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from experiments.atomistic.soap_pilot import composer
from experiments.atomistic.soap_pilot.composer import R2_PROFILES, TARGET_PROFILES, render_profile
from experiments.atomistic.soap_pilot.p3_render import (
    P3RenderError,
    admit_row_and_command,
    fail_closed_recipe_diff,
    finalize_eval_row,
    render_eval_template,
    render_pbs_audit,
    render_r2_train_packet,
)
from experiments.atomistic.soap_pilot.rows import CheckpointEvidence, make_parent_join


SOURCE = "7b88f833babcb82148c8f975a08b53970838be6a"
ROOT = Path(__file__).resolve().parents[4]


def _complete_publication(tmp_path: Path, train_row) -> tuple[Path, CheckpointEvidence]:
    checkpoint_dir = (
        tmp_path / train_row["experiment_name"] / train_row["sector"]
        / train_row["run_id"] / "checkpoints" / "step_000100"
    )
    checkpoint_dir.mkdir(parents=True)
    model = checkpoint_dir / "model.pt"
    model.write_bytes(b"review-publication-model")
    complete = checkpoint_dir / "COMPLETE"
    complete.write_text("complete\n", encoding="utf-8")
    publication = CheckpointEvidence(
        100,
        "soap100-publication",
        hashlib.sha256(model.read_bytes()).hexdigest(),
        True,
        {"model.pt": str(model), "COMPLETE": str(complete)},
    )
    return checkpoint_dir, publication


def test_adam_train_row_cannot_be_published_as_soap100_eval(tmp_path: Path) -> None:
    """A consistent parent join does not make an Adam row a SOAP publication."""

    packet = render_r2_train_packet(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )
    train = packet["r2-adam-100"]
    checkpoint_dir, publication = _complete_publication(tmp_path, train.row)
    parent = make_parent_join(campaign_id="review", invocation_id="adam-100", row=train.row)
    template = render_eval_template(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )

    with pytest.raises(P3RenderError, match="SOAP|family|optimizer"):
        finalize_eval_row(
            template,
            train_row=train.row,
            parent=parent,
            publication=publication,
            checkpoint_dir=checkpoint_dir,
        )


def test_soap100_control_accepts_real_complete_publication(tmp_path: Path) -> None:
    """The same complete fixture is accepted for the genuine SOAP train row."""

    train = render_r2_train_packet(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )["r2-soap-100"]
    checkpoint_dir, publication = _complete_publication(tmp_path, train.row)
    row = finalize_eval_row(
        render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"),
        train_row=train.row,
        parent=make_parent_join(campaign_id="review", invocation_id="soap-100", row=train.row),
        publication=publication,
        checkpoint_dir=checkpoint_dir,
    )
    assert row["checkpoint_sha256"] == publication.model_sha256


def test_publication_named_files_must_match_checkpoint_directory(tmp_path: Path) -> None:
    """A publication cannot name model files from an unrelated directory."""

    train = render_r2_train_packet(
        output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
    )["r2-soap-100"]
    checkpoint_dir, publication = _complete_publication(tmp_path, train.row)
    foreign = tmp_path / "foreign" / "model.pt"
    foreign.parent.mkdir()
    foreign.write_bytes(b"foreign-directory-model")
    publication = replace(publication, files={**publication.files, "model.pt": str(foreign)})

    with pytest.raises(P3RenderError, match="publication|checkpoint|file|directory"):
        finalize_eval_row(
            render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"),
            train_row=train.row,
            parent=make_parent_join(campaign_id="review", invocation_id="soap-100", row=train.row),
            publication=publication,
            checkpoint_dir=checkpoint_dir,
        )


def test_optimizer_recipe_drift_refuses_render_and_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing only SOAP's optimizer rate is unenumerated recipe drift."""

    original = composer._optimizer_and_trainer

    def drifted(profile):
        optimizer, trainer = original(profile)
        if profile.family == "soap":
            optimizer = {**optimizer, "lr": 0.123}
        return optimizer, trainer

    monkeypatch.setattr(composer, "_optimizer_and_trainer", drifted)
    with pytest.raises(P3RenderError, match="recipe|drift|enumerated"):
        packet = render_r2_train_packet(
            output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"
        )
        admit_row_and_command(packet["r2-soap-100"].row, python="/usr/bin/python3", checkout_root=str(ROOT))


def test_recipe_diff_refuses_unknown_null_valued_key() -> None:
    """An absent key and an explicit null are different recipe states."""

    with pytest.raises(P3RenderError, match="recipe|extra"):
        fail_closed_recipe_diff({}, {"extra": None})


def test_pbs_audit_refuses_disabled_gpu_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quoted guard needles are insufficient when guard_gpus returns early."""

    baseline = render_pbs_audit()
    assert "guard_gpus" in baseline["lr4_guards"]
    wrapper = Path(baseline["wrapper"]).resolve()
    original = Path.read_text

    def disabled_guard(path: Path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.resolve() == wrapper:
            text = text.replace("guard_gpus() {\n", "guard_gpus() {\n  return 0\n", 1)
        return text

    monkeypatch.setattr(Path, "read_text", disabled_guard)
    with pytest.raises(P3RenderError, match="guard|GPU|disabled|return"):
        render_pbs_audit()


@pytest.mark.parametrize("profile", TARGET_PROFILES)
def test_target_profiles_have_stage1_schedule_and_one_initial_checkpoint(tmp_path: Path, profile) -> None:
    """Target rows use Stage-1 zero-origin identity and positive YAML schedule."""

    rendered = render_profile(
        profile,
        output_dir=tmp_path / "inputs",
        results_root=str(tmp_path),
        source_sha=SOURCE,
        device="cpu",
    )
    admitted = admit_row_and_command(rendered.row, python="/usr/bin/python3", checkout_root=str(ROOT))
    assert admitted.row["checkpoint_steps"] == [0, 1000, 5000]
    assert rendered.rendered.values["checkpoint_steps"] == [1000, 5000]
    assert sum(
        callback.get("_target_") == "tpen.callback.InitialCheckpoint"
        for callback in rendered.rendered.values["callbacks"]
    ) == 1
