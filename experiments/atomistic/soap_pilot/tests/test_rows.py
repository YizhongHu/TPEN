"""Golden and negative tests for the immutable pilot row contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
import yaml

from experiments.atomistic.soap_pilot.composer import (
    R2_PROFILES,
    TARGET_PROFILES,
    _TRAIN_BASE,
    _OVERLAY_BASE,
    _resolve,
    render_eval,
    render_profile,
)
from experiments.atomistic.soap_pilot.p3_render import finalize_eval_row, render_eval_template
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
from experiments.atomistic.stage1.protocol import (
    SYSTEM_SPECS,
    manifest_hash as stage1_manifest_hash,
    planned_physical_system_ids,
    validate_gateway_binding,
)


def _finalized_eval(tmp_path: Path, train: dict) -> dict:
    """Build the supported concrete eval row from named publication evidence."""

    checkpoint_dir = (
        tmp_path / train["experiment_name"] / train["sector"] / train["run_id"]
        / "checkpoints" / "step_000100"
    )
    checkpoint_dir.mkdir(parents=True)
    model = checkpoint_dir / "model.pt"
    model.write_bytes(b"soap100-model")
    complete = checkpoint_dir / "COMPLETE"
    complete.write_text("published\n", encoding="utf-8")
    publication = CheckpointEvidence(
        100, "soap100-test-publication", hashlib.sha256(model.read_bytes()).hexdigest(),
        True, {"model.pt": str(model), "COMPLETE": str(complete)},
    )
    return finalize_eval_row(
        render_eval_template(output_dir=tmp_path / "inputs", results_root=str(tmp_path), source_sha=train["source_sha"], device="cpu"),
        train_row=train,
        parent=make_parent_join(campaign_id="soap-r2", invocation_id="soap-100", row=train),
        publication=publication,
        checkpoint_dir=checkpoint_dir,
    )


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


_INTENTIONAL_TOP_LEVEL_TRANSFORMATIONS = {
    # Protocol: “The overlay owns train-side runtime/sampler/optimizer/trainer
    # leaves plus diagnostic callbacks and loggers.”
    "runtime",
    "sampler",
    "optimizer",
    "trainer",
    # Protocol: “Base callbacks are retained and the diagnostics are appended
    # explicitly; checkpoint retention therefore stays manifest-owned.”
    "callbacks",
    # Protocol: “base-config -> recorder-before-checkpoint diagnostics-v2
    # overlay -> manifest overrides”.
    "run",
    # Protocol: “base-config -> recorder-before-checkpoint diagnostics-v2
    # overlay -> manifest overrides”. — profile optimizer/trainer bindings.
    "runner",
    "checkpoint_steps",
    "stage1_overlay",
}


def _resolved_overlay_union(rendered: Mapping[str, object]) -> dict[str, object]:
    """Resolve base+overlay references against one rendered config."""

    base = yaml.safe_load(_TRAIN_BASE.read_text(encoding="utf-8"))
    overlay = yaml.safe_load(_OVERLAY_BASE.read_text(encoding="utf-8"))
    expected = {**deepcopy(base), **deepcopy(overlay)}
    return {key: _resolve(value, rendered) for key, value in expected.items()}


@pytest.mark.parametrize("profile", [*R2_PROFILES, *TARGET_PROFILES], ids=lambda profile: profile.name)
def test_every_rendered_train_top_level_section_matches_base_overlay_or_literal_transform(
    tmp_path: Path, profile,
) -> None:
    """Fail closed on any dropped/added top-level base+overlay section."""

    results = tmp_path / "results"
    rendered = render_profile(
        profile,
        output_dir=results / "inputs",
        results_root=str(results),
        source_sha="7b88f833babcb82148c8f975a08b53970838be6a",
        device="cpu",
    )
    actual = rendered.rendered.values
    expected = _resolved_overlay_union(actual)
    expected_keys = set(expected)
    actual_keys = set(actual)
    dropped = sorted(expected_keys - actual_keys)
    added = sorted(actual_keys - expected_keys)
    unexpected_added = [key for key in added if key != "stage1_overlay"]
    assert not dropped and not unexpected_added, (
        "unexpected top-level section drift: "
        f"dropped={dropped}, added={unexpected_added}"
    )

    # Every non-transformed section is compared byte-for-byte after resolving
    # only the references that the composer itself resolves.
    for section in sorted(expected_keys - _INTENTIONAL_TOP_LEVEL_TRANSFORMATIONS):
        assert actual[section] == expected[section], (
            f"unenumerated top-level section change in {section!r}"
        )

    overlay = yaml.safe_load(_OVERLAY_BASE.read_text(encoding="utf-8"))
    assert actual["loggers"] == _resolve(overlay["loggers"], actual)


@pytest.mark.parametrize("profile", [*R2_PROFILES, *TARGET_PROFILES], ids=lambda profile: profile.name)
def test_every_rendered_train_callback_hydra_instantiates_on_cpu(tmp_path: Path, profile) -> None:
    """Every rendered callback has a constructible Hydra configuration."""

    results = tmp_path / "results"
    rendered = render_profile(
        profile,
        output_dir=results / "inputs",
        results_root=str(results),
        source_sha="7b88f833babcb82148c8f975a08b53970838be6a",
        device="cpu",
    )
    callbacks = rendered.rendered.values["callbacks"]
    assert callbacks

    for index, callback in enumerate(callbacks):
        try:
            instance = instantiate(OmegaConf.create(callback))
        except Exception as exc:
            target = callback.get("_target_", "<missing target>")
            raise AssertionError(f"{profile.name} callback {index} ({target}) failed to instantiate") from exc
        assert instance is not None

    if profile.family in {"adam", "soap"}:
        overlay = yaml.safe_load(_OVERLAY_BASE.read_text(encoding="utf-8"))
        overlay_optimizer_state = next(
            item for item in overlay["callbacks"] if item["_target_"] == "tpen.callback.OptimizerState"
        )
        rendered_optimizer_state = next(
            item for item in callbacks if item["_target_"] == "tpen.callback.OptimizerState"
        )
        assert rendered_optimizer_state["groups"] == overlay_optimizer_state["groups"]


@pytest.mark.parametrize("profile", [*R2_PROFILES, *TARGET_PROFILES], ids=lambda profile: profile.name)
def test_every_rendered_train_logger_hydra_instantiates_on_cpu(tmp_path: Path, profile) -> None:
    """Every derived diagnostics logger is constructible without training."""

    results = tmp_path / "results"
    rendered = render_profile(
        profile,
        output_dir=results / "inputs",
        results_root=str(results),
        source_sha="7b88f833babcb82148c8f975a08b53970838be6a",
        device="cpu",
    )
    loggers = rendered.rendered.values["loggers"]
    assert loggers == [{"_target_": "tpen.logging.JSONL", "path": str(results / "stage1_hooke2" / "atomistic" / profile.name.replace("-", "_") / "metrics.jsonl")}]
    for index, logger in enumerate(loggers):
        try:
            instance = instantiate(OmegaConf.create(logger))
        except Exception as exc:
            target = logger.get("_target_", "<missing target>")
            raise AssertionError(f"{profile.name} logger {index} ({target}) failed to instantiate") from exc
        assert instance is not None


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


@pytest.mark.parametrize("parent_row", [None, {"row_id": "synthetic"}])
def test_composer_eval_refuses_without_publication_parent(tmp_path: Path, parent_row) -> None:
    """Concrete eval composition is reserved for the gated P3 finalizer."""

    with pytest.raises(ValueError, match="p3_render|SOAP100 publication"):
        render_eval(
            output_dir=tmp_path / "inputs", results_root=str(tmp_path),
            source_sha="7b88f833babcb82148c8f975a08b53970838be6a",
            checkpoint_dir=str(tmp_path / "checkpoint"), checkpoint_sha256="a" * 64,
            parent_row=parent_row, device="cpu",
        )


def test_strict_fresh_eval_golden_join(tmp_path: Path) -> None:
    results = tmp_path / "results"
    train = render_profile(R2_PROFILES[1], output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu").row
    evaluation = _finalized_eval(results, train)
    assert evaluation["checkpoint_file"] == str((results / train["experiment_name"] / train["sector"] / train["run_id"] / "checkpoints" / "step_000100" / "model.pt").resolve())
    assert evaluation["walkers"] == 4096
    assert evaluation["draws"] == 256
    assert evaluation["depends_on"] == train["row_id"]
    assert evaluation["phase1_manifest_hash"] == make_parent_join(campaign_id="soap-r2", invocation_id="soap-100", row=train).manifest_hash


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
    """Exercise gateway admission for all train rows and a gated eval row."""

    results = tmp_path / "results"
    train_profiles = [*R2_PROFILES, *TARGET_PROFILES]
    train_rows = [
        render_profile(profile, output_dir=results / "inputs", results_root=str(results),
                       source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu").row
        for profile in train_profiles
    ]
    evaluation = _finalized_eval(results, train_rows[1])
    hooke2 = next(spec for spec in SYSTEM_SPECS if spec.name == "hooke2")
    for row in [*train_rows, evaluation]:
        admitted = validate_gateway_binding(binding=_binding_token(row), expected_kinds={row["kind"]})
        assert admitted["system"] == hooke2.name
        assert admitted["system_id"] == hooke2.system_id
    bad_registry = dict(train_rows[0], system="hooke1")
    with pytest.raises(Exception, match="config path|known system|identity|row_id"):
        validate_gateway_binding(binding=_binding_token(bad_registry), expected_kinds={"train"})
    bad_digest = dict(train_rows[0], config_sha256="0" * 64)
    bad_digest["row_id"] = row_id(bad_digest)
    with pytest.raises(Exception, match="identity"):
        validate_gateway_binding(binding=_binding_token(bad_digest), expected_kinds={"train"})


def test_collect_accepts_real_stage1_parent_eval_join_and_rejects_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the real collector and preserve parent and physical-id boundaries."""

    results = tmp_path / "results"
    train = render_profile(R2_PROFILES[1], output_dir=results / "inputs", results_root=str(results), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu").row
    evaluation = _finalized_eval(results, train)
    expected_physical_id = planned_physical_system_ids()["hooke2"]
    eval_rows = []
    for stream in range(4):
        row = dict(evaluation, stream=stream, walkers=1, draws=1)
        row["row_id"] = row_id(row)
        validate_row(row, expected_kind="eval")
        eval_rows.append(row)
        run_dir = results / row["experiment_name"] / row["sector"] / row["run_id"]
        (run_dir / "mcmc_energy").mkdir(parents=True, exist_ok=True)
        (run_dir / "mcmc_energy" / "sampled_eval_table.metadata.json").write_text("{}\n", encoding="utf-8")
    record = SimpleNamespace(
        record_schema="trajectory_records/v2", resolved_config_identity={"schema": "tpen-config-identity-v1", "sha256": "a" * 64},
        draw_count=1, walker_count=1, row_count=1, model_provenance="checkpoint", system_id=expected_physical_id,
        execution_method_id="method", source_git_sha=train["source_sha"], finite_count=1, mean=2.0, variance=0.0,
        record_content_id="record", csv_sha256="b" * 64, checkpoint_sha256=evaluation["checkpoint_sha256"],
    )
    monkeypatch.setattr("experiments.atomistic.stage1.collect.inspect_eval_run", lambda run_dir: {"verdict": "completed", "resolved_config_identity": record.resolved_config_identity, "source_git_sha": train["source_sha"]})
    monkeypatch.setattr("experiments.atomistic.stage1.collect.inspect_trajectory_record", lambda *args, **kwargs: record)
    monkeypatch.setattr("experiments.atomistic.stage1.collect.read_energy_grid", lambda record: [[2.0]])
    document = {"schema": "stage1/v4", "rows": [train, *eval_rows]}
    document["manifest_hash"] = stage1_manifest_hash(document["rows"])
    collected = collect(document)
    assert len(collected["records"]) == 4
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="soap-100", row=train)
    assert parent.train_row_hash == manifest_hash([train])
    for invalid_physical_id in (train["system_id"], "0" * 64):
        invalid_rows = [dict(row) for row in eval_rows]
        invalid_rows[0]["expected_physical_system_id"] = invalid_physical_id
        invalid_rows[0]["row_id"] = row_id(invalid_rows[0])
        invalid_document = {"schema": "stage1/v4", "rows": [train, *invalid_rows]}
        invalid_document["manifest_hash"] = stage1_manifest_hash(invalid_document["rows"])
        with pytest.raises(CollectionError, match="physical system identity"):
            collect(invalid_document)
    mutated_train = dict(train, run_id="mutated")
    mutated_train["row_id"] = row_id(mutated_train)
    mutated_eval = dict(eval_rows[0], depends_on=mutated_train["row_id"], phase1_train_row_hash=manifest_hash([mutated_train]))
    mutated_eval["row_id"] = row_id(mutated_eval)
    bad_document = {"schema": "stage1/v4", "rows": [mutated_train, *eval_rows]}
    bad_document["manifest_hash"] = stage1_manifest_hash(bad_document["rows"])
    with pytest.raises(CollectionError, match="phase-2 row|manifest"):
        collect(bad_document)

