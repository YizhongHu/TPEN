"""Golden and negative tests for the immutable pilot row contract."""

from __future__ import annotations

from pathlib import Path
import pytest

from experiments.atomistic.soap_pilot.composer import R2_PROFILES, TARGET_PROFILES, render_eval, render_profile
from experiments.atomistic.soap_pilot.rows import (
    CampaignMember,
    ComparisonGroup,
    RowContractError,
    make_parent_join,
    manifest_hash,
    row_id,
    validate_row,
    validate_seed_domain,
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

