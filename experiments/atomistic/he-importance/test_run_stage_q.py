"""Tests for the single-cell Stage-Q mechanics launcher."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest


stage_coordinate = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
runner = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")


def _cells(tmp_path: Path) -> tuple[object, ...]:
    return stage_coordinate.materialize_intended_configurations(tmp_path)


def test_q_materialization_has_two_seed_labeled_cells_and_selection_is_explicit(tmp_path: Path) -> None:
    cells = _cells(tmp_path)
    q_cells = [cell for cell in cells if cell.manifest["stage"] == "Q"]
    assert [cell.manifest["seed_identity"]["label"] for cell in q_cells] == [810_001, 810_002]

    with pytest.raises(runner.StageQSelectionError, match="ambiguous"):
        runner.select_cell(cells)

    selected = runner.select_cell(cells, seed_label=810_002)
    assert selected.selected.as_dict() == {
        "stage": "Q",
        "architecture": "control",
        "optimizer": "adam",
        "seed_label": 810_002,
        "status": "available",
    }
    assert any(item.seed_label == 810_001 for item in selected.discarded)


def test_selection_refuses_zero_matches_and_never_first_matches(tmp_path: Path) -> None:
    cells = _cells(tmp_path)
    with pytest.raises(runner.StageQSelectionError, match="zero admitted"):
        runner.select_cell(cells, seed_label=999_999)


def test_q_cadence_is_accepted_at_the_horizon_and_only_train_is_emitted(tmp_path: Path) -> None:
    selected = runner.select_cell(_cells(tmp_path), seed_label=810_001)
    packet = runner._q_training_packet(selected.cell)
    assert packet.checkpoint_cadence.every_n_updates == 1_000
    assert packet.checkpoint_cadence.selected_updates == (1_000, 2_000)


def test_shared_analysis_packets_are_discarded_after_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = runner.select_cell(_cells(tmp_path), seed_label=810_001)
    original = stage_coordinate.materialize_job_packets
    observed: dict[str, object] = {}

    def materialize_and_witness(*args: object, **kwargs: object) -> object:
        packets = original(*args, **kwargs)
        observed["ranking"] = packets.ranking
        observed["independent"] = packets.independent_sampler_test
        return replace(packets, ranking=(object(),), independent_sampler_test=(object(),))

    monkeypatch.setattr(stage_coordinate, "materialize_job_packets", materialize_and_witness)
    packet = runner._q_training_packet(selected.cell)

    assert packet.cell is selected.cell
    assert observed["ranking"]
    assert observed["independent"]


def test_evaluation_horizon_polarity_is_refused(tmp_path: Path) -> None:
    selected = runner.select_cell(_cells(tmp_path), seed_label=810_001)
    with pytest.raises(stage_coordinate.MaterializationError, match="exceed the stage horizon"):
        stage_coordinate.materialize_job_packets(
            (selected.cell,),
            stage_coordinate.CheckpointCadence(1_000, (100_000,)),
            {"statistic": "logabs_variance"},
            (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            {"walkers": 1, "burn_in_sweeps": 0},
            ddp_provenance={"launcher": "test"},
        )


def test_cli_emits_no_analysis_packets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = runner.build_parser().parse_args(
        ["--dry-run", "--seed-label", "810001", "--output-root", str(tmp_path)]
    )
    assert runner.execute(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["train_packet"]["ranking_packets"] == 0
    assert report["train_packet"]["independent_sampler_packets"] == 0


def test_roster_design_receipt_is_live_and_names_every_unavailable_arm() -> None:
    receipt = runner.roster_design_receipt("Q")
    from tpen.hi_schema import HI_METHOD_ROSTER

    assert receipt["admitted_methods"] == tuple(
        entry.method for entry in HI_METHOD_ROSTER if entry.admitted and entry.target
    )
    assert set(receipt["unavailable_methods"]) == {
        entry.method for entry in HI_METHOD_ROSTER if not entry.admitted or not entry.target
    }
    intended_count = stage_coordinate.stage_definition("Q").maximum_lineages
    assert receipt["intended_count_from_authority"] == intended_count
    assert receipt["difference"]["intended_count_minus_admitted_method_count"] == (
        intended_count - len(receipt["admitted_methods"])
    )


def test_selection_receipt_contains_selected_and_discarded_cells(tmp_path: Path) -> None:
    selected = runner.select_cell(_cells(tmp_path), seed_label=810_001)
    receipt = selected.receipt()
    assert receipt["selected"]["seed_label"] == 810_001
    assert any(item["seed_label"] == 810_002 for item in receipt["discarded"])
    json.dumps(receipt)
