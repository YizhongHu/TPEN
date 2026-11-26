"""Golden and negative tests for the immutable pilot row contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest

from experiments.atomistic.soap_pilot.composer import R2_PROFILES, TARGET_PROFILES, render_eval, render_profile
from experiments.atomistic.soap_pilot.rows import (
    CampaignMember,
    ComparisonGroup,
    CheckpointEvidence,
    RowContractError,
    make_eval_row,
    make_parent_join,
    manifest_hash,
    row_id,
    validate_row,
    validate_seed_domain,
)
from experiments.atomistic.stage1.collect import CollectionError, collect
from experiments.atomistic.stage1.launch import _binding_token
from experiments.atomistic.stage1.protocol import manifest_hash as stage1_manifest_hash, validate_gateway_binding


def test_golden_train_render_and_row_identity(tmp_path: Path) -> None:
    """The rendered bytes, extension, and row identity form one immutable unit."""

    results = tmp_path / "results"
    rendered = render_profile(R2_PROFILES[0], output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    assert rendered.rendered.sha256 == rendered.row["row_extension"]["rendered_config"]["sha256"]
    assert rendered.row["row_id"] == row_id(rendered.row)
    validate_row(rendered.row, expected_kind="train")


def test_row_tamper_is_deliberate_red_control(tmp_path: Path) -> None:
    """A changed rendered digest must fail closed (deliberate-red control)."""

    results = tmp_path / "results"
    rendered = render_profile(R2_PROFILES[1], output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    tampered = dict(rendered.row)
    tampered["row_extension"] = dict(tampered["row_extension"])
    tampered["row_extension"]["rendered_config"] = dict(tampered["row_extension"]["rendered_config"])
    tampered["row_extension"]["rendered_config"]["sha256"] = "0" * 64
    tampered["row_id"] = row_id(tampered)
    with pytest.raises(RowContractError, match="rendered_config.sha256|row_id"):
        validate_row(tampered, expected_kind="train")


def test_parent_join_and_comparison_seed_domain(tmp_path: Path) -> None:
    results = tmp_path / "results"
    rendered = render_profile(R2_PROFILES[0], output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="adam-0", row=rendered.row)
    assert parent.manifest_hash and parent.train_row_hash
    assert parent.train_row_hash == manifest_hash([rendered.row])
    group = ComparisonGroup("soap-r2", "hooke-lineage0", rendered.row["system"], 0, (
        CampaignMember("adam-0", "adam", "r2", None, rendered.row["seed"], rendered.row["sampler_seed"]),
        CampaignMember("soap-0", "soap", "r2", None, rendered.row["seed"], rendered.row["sampler_seed"]),
    ))
    group.validate()


def test_comparison_rejects_seed_collision() -> None:
    group = ComparisonGroup("soap-r2", "bad", "hooke_pair_singlet_omega0.5", 0, (
        CampaignMember("a", "adam", "r2", None, 4, 4),
        CampaignMember("b", "soap", "r2", None, 4, 4),
    ))
    with pytest.raises(RowContractError, match="distinct"):
        group.validate()


@pytest.mark.parametrize("profile", [*R2_PROFILES, *TARGET_PROFILES])
def test_every_roster_train_profile_has_golden_render_and_callback_policy(tmp_path: Path, profile) -> None:
    """Every immutable R2/target roster row is rendered and self-identifying."""

    results = tmp_path / "results"
    rendered = render_profile(profile, output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    validate_row(rendered.row, expected_kind="train")
    assert rendered.rendered.path.startswith(str((results / "inputs").resolve()))
    callbacks = rendered.rendered.values["callbacks"]
    targets = [callback["_target_"] for callback in callbacks]
    assert targets.count("tpen.callback.OptimizerState") == (1 if profile.family in {"adam", "soap"} else 0)
    for target in ("tpen.callback.TrainPhaseTiming", "tpen.callback.TrainStepTiming"):
        callback = next(callback for callback in callbacks if callback["_target_"] == target)
        assert callback["accelerator_synchronize"] is profile.train_timing_synchronize
    run_timing = next(callback for callback in callbacks if callback["_target_"] == "tpen.callback.RunTiming")
    assert run_timing["accelerator_synchronize"] is True


def test_r2_20_is_distinct_unsynchronized_cross_check(tmp_path: Path) -> None:
    results = tmp_path / "results"
    rendered = render_profile(R2_PROFILES[-1], output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    assert rendered.rendered.values["trainer"]["max_steps"] == 20
    assert rendered.rendered.values["checkpoint_steps"] == [20]
    assert all(
        callback["accelerator_synchronize"] is False
        for callback in rendered.rendered.values["callbacks"]
        if callback["_target_"] in {"tpen.callback.TrainPhaseTiming", "tpen.callback.TrainStepTiming"}
    )


def test_strict_fresh_eval_golden_join(tmp_path: Path) -> None:
    results = tmp_path / "results"
    train = render_profile(R2_PROFILES[1], output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap-100", row=train.row)
    checkpoint = CheckpointEvidence(100, "soap-100-step-100", "a" * 64, True, {"model.pt": "/tmp/model.pt", "COMPLETE": "/tmp/COMPLETE"})
    evaluation = render_eval(output_dir=results / "inputs", results_root=str(results), source_sha=train.row["source_sha"], checkpoint_dir=str(results / "soap" / "checkpoints" / "step_000100"), checkpoint_sha256="a" * 64, parent_row=train.row, device="cpu")
    assert evaluation.rendered.values["load"] == {"path": str(results / "soap" / "checkpoints" / "step_000100"), "mode": "model_only", "strict": True, "allow_protocol_mismatch": False}
    assert evaluation.row["checkpoint_file"] == str((results / "soap" / "checkpoints" / "step_000100" / "model.pt").resolve())
    assert evaluation.rendered.values["trajectory_identity"]["checkpoint_file"] == evaluation.row["checkpoint_file"]
    assert evaluation.row["walkers"] == evaluation.rendered.values["evaluation_sampler"]["n_walkers"] == 4096
    assert evaluation.row["draws"] == evaluation.rendered.values["evaluation_tasks"]["mcmc_energy"]["generator"]["n_draws"] == 256
    joined = make_eval_row(
        train_row=train.row,
        parent=parent,
        checkpoint=checkpoint,
        rendered=evaluation.rendered,
        row_base={
            "run_id": "joined-eval",
            "stream": 0,
            "draws": 256,
            "config": evaluation.row["config"],
            "seed": evaluation.row["seed"],
            "walkers": evaluation.row["walkers"],
            "checkpoint_dir": evaluation.row["checkpoint_dir"],
        },
    )
    assert joined["depends_on"] == train.row["row_id"]
    assert joined["phase1_manifest_hash"] == parent.manifest_hash


def test_parent_checkpoint_profile_and_path_refusals(tmp_path: Path) -> None:
    results = tmp_path / "results"
    rendered = render_profile(R2_PROFILES[0], output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    row = dict(rendered.row)
    row["run_id"] = "foreign"
    with pytest.raises(RowContractError, match="row_id|rendered_config"):
        validate_row(row, expected_kind="train")
    escaped = dict(rendered.row)
    escaped["row_extension"] = dict(escaped["row_extension"])
    escaped["row_extension"]["rendered_config"] = dict(escaped["row_extension"]["rendered_config"])
    escaped["row_extension"]["rendered_config"]["path"] = str(tmp_path / "outside.yaml")
    escaped["row_id"] = row_id(escaped)
    with pytest.raises(RowContractError, match="root|existing|escape"):
        validate_row(escaped, expected_kind="train")


def test_seed_domain_rejects_undeclared_collision() -> None:
    rows = [{"run_id": "a", "seed": 1, "sampler_seed": 2}, {"run_id": "b", "seed": 1, "sampler_seed": 2}]
    with pytest.raises(RowContractError, match="collision"):
        validate_seed_domain(rows)


def test_rendered_rows_pass_stage1_admission_and_registry_tamper_refuses(tmp_path: Path) -> None:
    """Exercise the real Stage-1 gateway for every rendered train/eval row."""

    results = tmp_path / "results"
    train_rows = [
        render_profile(profile, output_dir=results / "inputs", results_root=str(results),
                       source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu").row
        for profile in R2_PROFILES
    ]
    checkpoint_dir = results / "checkpoint" / "step_000100"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    evaluation = render_eval(
        output_dir=results / "inputs", results_root=str(results),
        source_sha=train_rows[1]["source_sha"], checkpoint_dir=str(checkpoint_dir),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        parent_row=train_rows[1], device="cpu",
    ).row

    for row in [*train_rows, evaluation]:
        admitted = validate_gateway_binding(
            binding=_binding_token(row), expected_kinds={row["kind"]}
        )
        assert admitted["system"] == "hooke2"

    bad_registry = dict(train_rows[0], system="hooke1")
    with pytest.raises(Exception, match="config path|known system|identity"):
        validate_gateway_binding(binding=_binding_token(bad_registry), expected_kinds={"train"})
    bad_digest = dict(train_rows[0], config_sha256="0" * 64)
    with pytest.raises(Exception, match="identity"):
        validate_gateway_binding(binding=_binding_token(bad_digest), expected_kinds={"train"})


def test_collect_accepts_real_stage1_parent_eval_join_and_rejects_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the real Stage-1 collector and prove its parent hash boundary."""

    results = tmp_path / "results"
    train = render_profile(
        R2_PROFILES[1], output_dir=results / "inputs", results_root=str(results),
        source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu",
    ).row
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap-100", row=train)
    checkpoint_dir = results / "checkpoint" / "step_000100"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    eval_composed = render_eval(
        output_dir=results / "inputs", results_root=str(results), source_sha=train["source_sha"],
        checkpoint_dir=str(checkpoint_dir), checkpoint_sha256=checkpoint_sha,
        parent_row=train, device="cpu",
    )
    eval_rows = []
    for stream in range(4):
        row = dict(eval_composed.row)
        row["stream"] = stream
        row["walkers"] = 1
        row["draws"] = 1
        row["row_id"] = row_id(row)
        validate_row(row, expected_kind="eval")
        eval_rows.append(row)
        run_dir = results / row["experiment_name"] / row["sector"] / row["run_id"]
        (run_dir / "mcmc_energy").mkdir(parents=True, exist_ok=True)
        (run_dir / "mcmc_energy" / "sampled_eval_table.metadata.json").write_text("{}\n", encoding="utf-8")

    record = SimpleNamespace(
        record_schema="trajectory_records/v2", resolved_config_identity={"schema": "tpen-config-identity-v1", "sha256": "a" * 64},
        draw_count=1, walker_count=1, row_count=1, model_provenance="checkpoint",
        system_id=train["system_id"], execution_method_id="method", source_git_sha=train["source_sha"],
        finite_count=1, mean=2.0, variance=0.0, record_content_id="record", csv_sha256="b" * 64,
        checkpoint_sha256=checkpoint_sha,
    )
    monkeypatch.setattr("experiments.atomistic.stage1.collect.inspect_eval_run", lambda run_dir: {"verdict": "completed", "resolved_config_identity": record.resolved_config_identity, "source_git_sha": train["source_sha"]})
    monkeypatch.setattr("experiments.atomistic.stage1.collect.inspect_trajectory_record", lambda *args, **kwargs: record)
    monkeypatch.setattr("experiments.atomistic.stage1.collect.read_energy_grid", lambda record: [[2.0]])
    document = {"schema": "stage1/v4", "rows": [train, *eval_rows]}
    document["manifest_hash"] = stage1_manifest_hash(document["rows"])
    collected = collect(document)
    assert len(collected["records"]) == 4
    assert parent.train_row_hash == manifest_hash([train])

    mutated_train = dict(train, run_id="mutated")
    mutated_train["row_id"] = row_id(mutated_train)
    assert stage1_manifest_hash([mutated_train]) != parent.manifest_hash
    mutated_eval = dict(eval_rows[0], depends_on=mutated_train["row_id"])
    mutated_eval["phase1_train_row_hash"] = manifest_hash([mutated_train])
    mutated_eval["row_id"] = row_id(mutated_eval)
    assert mutated_eval["row_id"] != eval_rows[0]["row_id"]
    bad_document = {"schema": "stage1/v4", "rows": [mutated_train, *eval_rows]}
    bad_document["manifest_hash"] = stage1_manifest_hash(bad_document["rows"])
    with pytest.raises(CollectionError, match="phase-2 row|manifest"):
        collect(bad_document)
