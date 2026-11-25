"""Golden and negative tests for the immutable pilot row contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.atomistic.soap_pilot.composer import R2_PROFILES, render_profile
from experiments.atomistic.soap_pilot.rows import (
    CampaignMember,
    ComparisonGroup,
    RowContractError,
    make_parent_join,
    row_id,
    validate_row,
)


def test_golden_train_render_and_row_identity(tmp_path: Path) -> None:
    """The rendered bytes, extension, and row identity form one immutable unit."""

    rendered = render_profile(R2_PROFILES[0], output_dir=tmp_path, results_root=str(tmp_path / "results"), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    assert rendered.rendered.sha256 == rendered.row["row_extension"]["rendered_config"]["sha256"]
    assert rendered.row["row_id"] == row_id(rendered.row)
    validate_row(rendered.row, expected_kind="train")


def test_row_tamper_is_deliberate_red_control(tmp_path: Path) -> None:
    """A changed rendered digest must fail closed (deliberate-red control)."""

    rendered = render_profile(R2_PROFILES[1], output_dir=tmp_path, results_root=str(tmp_path / "results"), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    tampered = dict(rendered.row)
    tampered["row_extension"] = dict(tampered["row_extension"])
    tampered["row_extension"]["rendered_config"] = dict(tampered["row_extension"]["rendered_config"])
    tampered["row_extension"]["rendered_config"]["sha256"] = "0" * 64
    tampered["row_id"] = row_id(tampered)
    with pytest.raises(RowContractError, match="rendered_config.sha256|row_id"):
        validate_row(tampered, expected_kind="train")


def test_parent_join_and_comparison_seed_domain(tmp_path: Path) -> None:
    rendered = render_profile(R2_PROFILES[0], output_dir=tmp_path, results_root=str(tmp_path / "results"), source_sha="7b88f833babcb82148c8f975a08b53970838be6a", device="cpu")
    parent = make_parent_join(campaign_id="soap-r2", invocation_id="adam-0", row=rendered.row)
    assert parent.manifest_hash == parent.train_row_hash or parent.manifest_hash != ""
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
