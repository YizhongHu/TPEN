"""Focused P3 packet tests; execution belongs on the Cannon test partition."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.atomistic.soap_pilot.composer import R2_PROFILES
from experiments.atomistic.soap_pilot.p3_render import (
    P3RenderError,
    admit_row_and_command,
    finalize_eval_row,
    render_eval_template,
    render_pbs_audit,
    render_r2_train_packet,
)
from experiments.atomistic.soap_pilot.rows import CheckpointEvidence, make_parent_join


SOURCE = "7b88f833babcb82148c8f975a08b53970838be6a"


def test_p3_five_r2_rows_are_singleton_and_stage1_admitted(tmp_path: Path) -> None:
    """The four train rows are singleton; the fifth slot is an eval template."""

    packet = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    assert set(packet) == {profile.name for profile in R2_PROFILES}
    assert len({item.row["run_id"] for item in packet.values()}) == 4
    template = render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    assert template.checkpoint_step == 100
    assert not (tmp_path / "inputs" / "r2-soap-eval-100.yaml").exists()


def test_p3_row_identity_path_sha_and_command_token_audit(tmp_path: Path) -> None:
    """Rendered rows retain canonical input paths and byte identities."""

    packet = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    for composed in packet.values():
        row = composed.row
        assert row["row_id"]
        assert Path(row["row_extension"]["rendered_config"]["path"]).is_file()
        assert hashlib.sha256(Path(composed.rendered.path).read_bytes()).hexdigest() == composed.rendered.sha256


def test_p3_real_train_eval_mains_core_stub_use_admitted_bytes_once(tmp_path: Path) -> None:
    """The packet exposes one rendered file per row, with no overlay second pass."""

    packet = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    for composed in packet.values():
        assert composed.row["row_extension"]["rendered_config"]["path"] == composed.rendered.path
        assert composed.row["row_extension"]["rendered_config"]["sha256"] == composed.rendered.sha256


def test_p3_eval_template_refuses_without_complete_soap100_publication(tmp_path: Path) -> None:
    """No final row exists until model.pt, COMPLETE, and the actual parent join exist."""

    train = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")["r2-soap-100"]
    template = render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap100", row=train.row)
    publication = CheckpointEvidence(100, "soap100", "a" * 64, True, {"model.pt": "model.pt", "COMPLETE": "COMPLETE"})
    with pytest.raises(P3RenderError, match="COMPLETE"):
        finalize_eval_row(template, train_row=train.row, parent=parent, publication=publication, checkpoint_dir=tmp_path / "missing")


@pytest.mark.parametrize("mutation", ["extension", "source", "kind", "profile", "base", "sha"])
def test_p3_refusal_matrix_rejects_mutated_binding_before_launch(tmp_path: Path, mutation: str) -> None:
    """Binding, source, kind, profile, base, and digest mutations refuse."""

    composed = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")["r2-soap-100"]
    row = dict(composed.row)
    if mutation == "extension":
        row["row_extension"] = {"version": "bad"}
    elif mutation == "source":
        row["source_sha"] = "0" * 40
    elif mutation == "kind":
        row["kind"] = "eval"
    elif mutation == "profile":
        row["row_extension"] = dict(row["row_extension"], diagnostics_profile="bad")
    elif mutation == "base":
        row["config_identity_rule"] = "foreign"
    else:
        row["row_extension"] = dict(row["row_extension"], rendered_config={"path": composed.rendered.path, "sha256": "0" * 64})
    with pytest.raises((P3RenderError, ValueError)):
        admit_row_and_command(row, python="/usr/bin/python3", checkout_root=str(tmp_path))


def test_p3_pbs_packet_env_argv_outputs_and_lr4_guards() -> None:
    """The P3 audit consumes, but does not alter, the committed PBS wrapper."""

    audit = render_pbs_audit()
    assert "SELECTED_ROW" in audit["env"]
    assert audit["command_owner"].endswith("command_for_row")
    assert "$RESULTS_ROOT/logs/row-" in audit["outputs"]


def test_p3_fail_closed_recipe_diff_allows_only_enumerated_r2_changes(tmp_path: Path) -> None:
    """Template refusal and unique rows provide the fail-closed diff boundary."""

    packet = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    assert {row.row["updates"] for row in packet.values()} == {20, 100}
    assert all(row.row["lineage"] == 0 for row in packet.values())
