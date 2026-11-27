"""Focused P3 packet tests; execution belongs on the Cannon test partition."""

from __future__ import annotations

import hashlib
import base64
import json
from dataclasses import replace
from pathlib import Path
import shlex
import sys

import pytest
import yaml

from experiments.atomistic.soap_pilot.composer import R2_PROFILES
from experiments.atomistic.soap_pilot.p3_render import (
    P3RenderError,
    admit_row_and_command,
    fail_closed_recipe_diff,
    finalize_eval_row,
    render_eval_template,
    render_pbs_audit,
    render_r2_train_packet,
)
from experiments.atomistic.soap_pilot.rows import CheckpointEvidence, make_parent_join, manifest_hash, row_id
from experiments.atomistic.stage1.launch import _binding_token, command_for_row


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
        assert row_id(row) == row["row_id"] == manifest_hash([{key: value for key, value in row.items() if key != "row_id"}])
        token_row = json.loads(base64.urlsafe_b64decode(_binding_token(row) + "=" * (-len(_binding_token(row)) % 4)))
        assert token_row == row
        assert Path(row["row_extension"]["rendered_config"]["path"]).is_file()
        assert hashlib.sha256(Path(composed.rendered.path).read_bytes()).hexdigest() == composed.rendered.sha256
        assert row["seed"] == 2435329684536531677
        assert row["sampler_seed"] == 7066630035130326805
        assert row["checkpoint_steps"] == ([0, 100] if row["updates"] == 100 else [0, 20])


def test_p3_real_train_eval_mains_core_stub_use_admitted_bytes_once(tmp_path: Path) -> None:
    """Real train/eval mains reach one stubbed core call from launcher argv."""

    from experiments.atomistic.stage1 import eval as stage1_eval
    from experiments.atomistic.stage1 import train as stage1_train
    from omegaconf import OmegaConf

    packet = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    train = packet["r2-soap-100"]
    root = Path(__file__).resolve().parents[4]
    train_calls: list[dict[str, object]] = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(stage1_train, "run_from_config", lambda cfg, **kwargs: train_calls.append({"cfg": OmegaConf.to_container(cfg, resolve=True), **kwargs}) or 0)
        train_argv = shlex.split(command_for_row(train.row, python=sys.executable, checkout_root=str(root)))
        monkeypatch.setattr(sys, "argv", [train_argv[0], *train_argv[2:]])
        assert stage1_train.main() == 0
    finally:
        monkeypatch.undo()
    assert len(train_calls) == 1
    assert train_calls[0]["config_path"] == train.rendered.path
    train_cfg = train_calls[0]["cfg"]
    assert train_cfg["sampler"]["n_walkers"] == 512  # type: ignore[index]
    assert sum(callback.get("_target_") == "tpen.callback.InitialCheckpoint" for callback in train_cfg["callbacks"]) == 1  # type: ignore[index]

    checkpoint_dir = tmp_path / train.row["experiment_name"] / train.row["sector"] / train.row["run_id"] / "checkpoints" / "step_000100"
    checkpoint_dir.mkdir(parents=True)
    model = checkpoint_dir / "model.pt"
    model.write_bytes(b"model-bytes")
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    publication = CheckpointEvidence(100, "soap100-publication", hashlib.sha256(model.read_bytes()).hexdigest(), True, {"model.pt": str(model), "COMPLETE": str(checkpoint_dir / "COMPLETE")})
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap100", row=train.row)
    template = render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    eval_row = finalize_eval_row(template, train_row=train.row, parent=parent, publication=publication, checkpoint_dir=checkpoint_dir)
    eval_calls: list[dict[str, object]] = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(stage1_eval, "run_from_config", lambda cfg, **kwargs: eval_calls.append({"cfg": OmegaConf.to_container(cfg, resolve=True), **kwargs}) or 0)
        eval_argv = shlex.split(command_for_row(eval_row, python=sys.executable, checkout_root=str(root)))
        assert eval_argv[-2] == "--binding"
        monkeypatch.setattr(sys, "argv", [eval_argv[0], *eval_argv[2:]])
        assert stage1_eval.main() == 0
    finally:
        monkeypatch.undo()
    assert len(eval_calls) == 1
    assert eval_calls[0]["config_path"] == str(Path(eval_row["row_extension"]["rendered_config"]["path"]).resolve())
    assert eval_calls[0]["cfg"]["runner"]["load"]["mode"] == "model_only"  # type: ignore[index]


def test_p3_eval_template_refuses_without_complete_soap100_publication(tmp_path: Path) -> None:
    """No final row exists until model.pt, COMPLETE, and the actual parent join exist."""

    train = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")["r2-soap-100"]
    template = render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap100", row=train.row)
    publication = CheckpointEvidence(100, "soap100", "a" * 64, True, {"model.pt": "model.pt", "COMPLETE": "COMPLETE"})
    with pytest.raises(P3RenderError, match="COMPLETE"):
        finalize_eval_row(template, train_row=train.row, parent=parent, publication=publication, checkpoint_dir=tmp_path / train.row["experiment_name"] / train.row["sector"] / train.row["run_id"] / "checkpoints" / "step_000100")


def test_p3_complete_soap100_finalizes_eval_and_uses_eval_driver(tmp_path: Path) -> None:
    """A complete publication produces an eval row and launcher-owned argv."""

    train = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")["r2-soap-100"]
    checkpoint_dir = tmp_path / train.row["experiment_name"] / train.row["sector"] / train.row["run_id"] / "checkpoints" / "step_000100"
    checkpoint_dir.mkdir(parents=True)
    model = checkpoint_dir / "model.pt"
    model.write_bytes(b"complete-model")
    (checkpoint_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    publication = CheckpointEvidence(100, "soap100-publication", hashlib.sha256(model.read_bytes()).hexdigest(), True, {"model.pt": str(model), "COMPLETE": str(checkpoint_dir / "COMPLETE")})
    row = finalize_eval_row(render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"), train_row=train.row, parent=make_parent_join(campaign_id="soap-r2", invocation_id="soap100", row=train.row), publication=publication, checkpoint_dir=checkpoint_dir)
    command = command_for_row(row, python="/usr/bin/python3", checkout_root=str(Path(__file__).resolve().parents[4]))
    assert "experiments/atomistic/stage1/eval.py" in command
    assert "--binding" in command
    assert row["checkpoint_sha256"] == publication.model_sha256


@pytest.mark.parametrize("case", ["missing-model", "sha-mismatch", "missing-complete", "foreign-parent", "wrong-step", "duplicate-publication", "missing-publisher", "existing-destination"])
def test_p3_eval_publication_refusal_matrix_has_zero_core_calls(tmp_path: Path, case: str) -> None:
    """Every publication/parent refusal happens before any core invocation."""

    train = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")["r2-soap-100"]
    checkpoint_dir = tmp_path / train.row["experiment_name"] / train.row["sector"] / train.row["run_id"] / "checkpoints" / "step_000100"
    checkpoint_dir.mkdir(parents=True)
    model = checkpoint_dir / "model.pt"
    model.write_bytes(b"model")
    complete = checkpoint_dir / "COMPLETE"
    complete.write_text("ok\n", encoding="utf-8")
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    publication = CheckpointEvidence(100, "soap100", model_sha, True, {"model.pt": str(model), "COMPLETE": str(complete)})
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap100", row=train.row)
    if case == "missing-model":
        model.unlink()
    elif case == "sha-mismatch":
        publication = CheckpointEvidence(100, "soap100", "a" * 64, True, {"model.pt": str(model), "COMPLETE": str(complete)})
    elif case == "missing-complete":
        complete.unlink()
    elif case == "foreign-parent":
        parent = replace(parent, campaign_id="foreign", invocation_id="foreign", row_id="foreign-parent")
    elif case == "wrong-step":
        publication = CheckpointEvidence(20, "soap100", model_sha, True, {"model.pt": str(model), "COMPLETE": str(complete)})
    elif case == "duplicate-publication":
        publication = CheckpointEvidence(100, "duplicate", model_sha, True, {"model.pt": str(model), "COMPLETE": str(complete)})
    elif case == "missing-publisher":
        publication = CheckpointEvidence(100, "", model_sha, False, {"model.pt": str(model), "COMPLETE": str(complete)})
    else:
        (tmp_path / "inputs" / "r2-soap-eval-100.yaml").write_text("existing\n", encoding="utf-8")
    calls: list[object] = []
    expected_messages = {
        "missing-model": "COMPLETE with model.pt",
        "sha-mismatch": "hash disagrees",
        "missing-complete": "COMPLETE with model.pt",
        "foreign-parent": "parent does not match train row",
        "wrong-step": "checkpoint-100",
        "duplicate-publication": "identity is missing or non-unique",
        "missing-publisher": "identity is missing or non-unique",
        "existing-destination": "destination already exists",
    }
    with pytest.raises(P3RenderError, match=expected_messages[case]):
        finalize_eval_row(render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu"), train_row=train.row, parent=parent, publication=publication, checkpoint_dir=checkpoint_dir)
    assert calls == []


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


@pytest.mark.parametrize("mutation", ["nested-runner", "load-resume", "outside", "symlink"])
def test_p3_extended_refusal_matrix_has_zero_core_calls(tmp_path: Path, mutation: str) -> None:
    """Runner/load/path substitutions refuse before a core callback is possible."""

    composed = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")["r2-soap-100"]
    row = dict(composed.row)
    path = Path(composed.rendered.path)
    if mutation in {"nested-runner", "load-resume"}:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if mutation == "nested-runner":
            document["runner"]["unexpected"] = True
        else:
            document["load"] = {"mode": "resume", "path": "foreign"}
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        row["row_extension"] = dict(row["row_extension"], rendered_config={"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        row["row_id"] = row_id(row)
    else:
        outside = tmp_path / "outside.yaml"
        outside.write_bytes(path.read_bytes())
        replacement = outside
        if mutation == "symlink":
            replacement = tmp_path / "inputs" / "escape.yaml"
            replacement.symlink_to(outside)
        row["row_extension"] = dict(row["row_extension"], rendered_config={"path": str(replacement), "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()})
        row["row_id"] = row_id(row)
    calls: list[object] = []
    with pytest.raises((P3RenderError, ValueError)):
        admit_row_and_command(row, python="/usr/bin/python3", checkout_root=str(tmp_path))
    assert calls == []


def test_p3_pbs_packet_env_argv_outputs_and_lr4_guards() -> None:
    """The P3 audit consumes, but does not alter, the committed PBS wrapper."""

    audit = render_pbs_audit()
    assert set(audit["env"]) >= {"MODE", "SOURCE_SHA", "CHECKOUT_ROOT", "UV_CACHE_DIR", "RESULTS_ROOT", "SELECTED_ROW", "APPROVAL_ID", "EXPECTED_QUEUE"}
    assert "command_for_row(row, python=python, checkout_root=checkout)" in audit["argv"]
    assert 'row-${row_id}-${PBS_JOBID}.log' in audit["log"]
    assert 'row-${row_id}-${PBS_JOBID}.status' in audit["status"]
    assert set(audit["lr4_guards"]) == {
        "assert_pbs_context", "validate_selected_row", "assert_contained_path", "guard_gpus", "derive_cpu_layout",
    }
    for name in audit["lr4_guards"]:
        evidence = audit["guard_evidence"][name]
        assert evidence["line_range"][0] <= evidence["line_range"][1]
        assert evidence["checks"]
        assert all(":" in check for check in evidence["checks"])
    state_evidence = audit["execution_state_evidence"]
    assert all(state_evidence[name]["checks"] for name in ("cache", "approval", "queue"))
    assert all(":" in check for item in state_evidence.values() for check in item["checks"])


def test_p3_fail_closed_recipe_diff_allows_only_enumerated_r2_changes(tmp_path: Path) -> None:
    """Only enumerated scale/destination changes pass the recipe diff."""

    packet = render_r2_train_packet(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=SOURCE, device="cpu")
    assert {row.row["updates"] for row in packet.values()} == {20, 100}
    assert all(row.row["lineage"] == 0 for row in packet.values())
    expected = {"runtime": {"seed": 1, "dtype": "float64"}, "optimizer": {"lr": 1.0}, "callbacks": ["RunTiming"], "output": {"root": "base"}, "schedule": [100]}
    allowed = {"runtime.seed", "output.root", "schedule"}
    actual = {"runtime": {"seed": 2, "dtype": "float32"}, "optimizer": {"lr": 2.0}, "callbacks": ["InitialCheckpoint"], "output": {"root": "r2"}, "schedule": [20, 100]}
    with pytest.raises(P3RenderError, match="unenumerated"):
        fail_closed_recipe_diff(expected, actual, allowed_paths=allowed)
    fail_closed_recipe_diff(expected, {"runtime": {"seed": 2, "dtype": "float64"}, "optimizer": {"lr": 1.0}, "callbacks": ["RunTiming"], "output": {"root": "r2"}, "schedule": [20, 100]}, allowed_paths=allowed)
