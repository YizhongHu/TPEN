"""Tests for the single-cell Stage-Q mechanics launcher."""

from __future__ import annotations

import builtins
import importlib
import io
import json
from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


stage_coordinate = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
runner = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")
stage = stage_coordinate


def _cells(tmp_path: Path) -> tuple[object, ...]:
    return stage_coordinate.materialize_intended_configurations(tmp_path)


def _args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values = {
        "output_root": tmp_path,
        "stage": "Q",
        "architecture": "control",
        "optimizer": "adam",
        "seed_label": 810001,
        "dry_run": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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
    assert "facility" not in report
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


def test_execution_facts_cross_launch_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCUDA:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            assert index == 0
            return SimpleNamespace(uuid="GPU-0")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCUDA()))
    facts = runner._execution_facts()
    selected = runner.select_cell(_cells(tmp_path), seed_label=810_001)
    received: dict[str, object] = {}

    def recording_runner(config: object, *, topology: object) -> int:
        received["topology"] = topology
        return 0

    assert runner.launch.launch_train(
        selected.cell, facts, runner=recording_runner
    ) == 0
    assert received["topology"] == facts
    assert facts == {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": facts["host"],
        "pid": facts["pid"],
        "device": "cuda:0",
        "job_id": facts["job_id"],
        "device_identity": {"kind": "cuda", "index": 0, "uuid": "GPU-0"},
    }


def test_g1_reversed_inventory_is_hash_stable_and_zero_multiple_unavailable_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "stage": "Q",
        "architecture": "control",
        "optimizer": "adam",
        "seed_label": 810001,
        "status": "available",
    }
    cells = stage.materialize_intended_configurations(tmp_path)
    forward = runner.select_cell(cells, seed_label=810001)
    reverse = runner.select_cell(tuple(reversed(cells)), seed_label=810001)
    assert forward.selected.as_dict() == expected
    assert reverse.selected.as_dict() == expected
    assert forward.cell.content_hash == reverse.cell.content_hash

    with pytest.raises(runner.StageQSelectionError, match="zero admitted"):
        runner.select_cell(cells, seed_label=999999)
    with pytest.raises(runner.StageQSelectionError, match="ambiguous"):
        runner.select_cell(cells, seed_label=None)
    with pytest.raises(runner.StageQSelectionError, match="ambiguous"):
        runner.select_cell(cells + (cells[0],), seed_label=810001)

    original_roster = runner._roster
    monkeypatch.setattr(
        runner,
        "_roster",
        lambda: tuple(entry for entry in original_roster() if entry.method != "adam"),
    )
    with pytest.raises(runner.StageQSelectionError, match="zero admitted"):
        runner.select_cell(cells, seed_label=810001)

    unavailable = stage.materialize_stage(
        "Q",
        [{"scientific_identity": {"architecture": "control", "optimizer": "adam"},
          "payload": {"updates": 2000}, "topology": {}}],
        stage.OptimizerCell("adam", "unavailable", reason="sentinel"),
        tmp_path / "unavailable",
    )
    monkeypatch.setattr(runner, "_roster", original_roster)
    with pytest.raises(runner.StageQSelectionError, match="zero admitted"):
        runner.select_cell(unavailable, seed_label=810001)


def test_runtime_roster_admission_negative_controls_are_not_receipt_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = runner.roster_design_receipt("Q")
    assert "adam" in baseline["admitted_methods"]
    assert baseline["unavailable_methods"]
    cells = stage.materialize_intended_configurations(tmp_path)
    original = runner._roster()
    controls = (
        (False, "module.Adam", "not-admitted"),
        (True, None, "missing-target"),
    )
    for admitted, target, reason in controls:
        sentinels = tuple(
            entry if entry.method != "adam" else SimpleNamespace(
                method="adam", admitted=admitted, target=target, requires=reason
            )
            for entry in original
        )
        monkeypatch.setattr(runner, "_roster", lambda sentinels=sentinels: sentinels)
        receipt = runner.roster_design_receipt("Q")
        assert receipt["admitted_methods"] == ()
        assert receipt["unavailable_methods"]["adam"] == reason
        with pytest.raises(runner.StageQSelectionError, match="zero admitted"):
            runner.select_cell(cells, seed_label=810001)


def test_g2_roster_receipt_is_live_and_authority_difference_is_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = (
        SimpleNamespace(method="sentinel", admitted=True, target="module.Target", requires="ready"),
        SimpleNamespace(method="not-admitted", admitted=False, target="module.No", requires="blocked"),
        SimpleNamespace(method="missing-target", admitted=True, target=None, requires="missing"),
    )
    authority = SimpleNamespace(maximum_lineages=17)
    expected_admitted = ("sentinel",)
    expected_unavailable = {"not-admitted": "blocked", "missing-target": "missing"}
    monkeypatch.setattr(runner, "_roster", lambda: entries)
    monkeypatch.setattr(stage, "stage_definition", lambda code: authority)
    receipt = runner.roster_design_receipt("Q")
    assert receipt["admitted_methods"] == expected_admitted
    assert receipt["unavailable_methods"] == expected_unavailable
    assert receipt["intended_count_from_authority"] == 17
    assert receipt["difference"]["intended_count_minus_admitted_method_count"] == 16


def test_g3_q_cadence_refuses_evaluation_horizon_and_execute_dry_run_does_not_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = runner.select_cell(
        stage.materialize_intended_configurations(tmp_path), seed_label=810001
    )
    with pytest.raises(stage.MaterializationError, match="exceed the stage horizon"):
        stage.materialize_job_packets(
            (selected.cell,), stage.CheckpointCadence(1000, (100000,)),
            {"statistic": "logabs_variance"},
            (stage.RankingStatistic.LOGABS_VARIANCE,),
            {"walkers": 1, "burn_in_sweeps": 0}, ddp_provenance={"launcher": "test"},
        )
    monkeypatch.setattr(runner.launch, "launch_train", lambda *a, **k: pytest.fail("dry run launched"))
    opened: list[Path] = []
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "evaluation.yaml":
            opened.append(path)
            raise AssertionError("execute read evaluation manifest")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    assert runner.execute(_args(tmp_path)) == 0
    assert not opened


def test_cadence_acceptance_and_refusal_are_both_explicit() -> None:
    assert stage.CheckpointCadence(1000, (1000, 2000)).selected_updates == (1000, 2000)
    with pytest.raises(stage.MaterializationError, match="lie on the dense cadence"):
        stage.CheckpointCadence(1000, (1500,))
    with pytest.raises(stage.MaterializationError, match="at least one"):
        stage.CheckpointCadence(1000, ())


def test_r1_o1_select_cell_refuses_before_consuming_cells() -> None:
    class ExplodingCells:
        def __iter__(self):
            raise AssertionError("non-Q selection consumed cell input")

    with pytest.raises(runner.StageQSelectionError, match="only stage 'Q'"):
        runner.select_cell(ExplodingCells(), stage="O1", seed_label=820001)


def test_r1_o1_execute_refuses_before_artifact_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="O1", seed_label=820001, dry_run=False)
    facts = {
        "global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1,
        "node_rank": 0, "node_size": 1, "host": "host", "pid": 1,
        "device": "cuda:0", "job_id": "job", "device_identity": None,
    }
    calls: list[object] = []
    monkeypatch.setattr(runner, "_execution_facts", lambda: facts)
    monkeypatch.setattr(runner.launch, "launch_train", lambda *a, **k: calls.append(a) or 0)
    error: BaseException | None = None
    try:
        runner.execute(args)
    except BaseException as caught:
        error = caught
    assert isinstance(error, runner.StageQSelectionError), (
        f"O1 execute must refuse before packet resolution/launch, got {error!r}"
    )
    assert not calls
    assert not list(tmp_path.rglob("selection_receipt.json"))


def test_g4_execute_forwards_exact_facts_and_emits_completion_only_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_packet_type = stage.TrainingPacket
    facts = {
        "global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1,
        "node_rank": 0, "node_size": 1, "host": "sentinel-host", "pid": 123,
        "device": "cuda:sentinel", "job_id": "job-sentinel",
        "device_identity": {"kind": "cuda", "index": 0, "uuid": "GPU-sentinel"},
    }
    monkeypatch.setattr(runner, "_execution_facts", lambda: dict(facts))
    original = runner.launch.launch_train
    observed: dict[str, object] = {}

    def launch_wrapper(source: object, topology: object) -> int:
        observed["topology"] = topology
        observed["packet_type"] = type(source)

        def recording_runner(config: object, *, topology: object) -> int:
            observed["runner_topology"] = topology
            output = Path(source.cell.output_path) / "resolved_config.yaml"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("horizon: 2000\n", encoding="utf-8")
            observed["config"] = config
            return 0

        return original(source, topology, runner=recording_runner)

    monkeypatch.setattr(runner.launch, "launch_train", launch_wrapper)
    assert runner.execute(_args(tmp_path, dry_run=False)) == 0
    assert observed["packet_type"] is expected_packet_type
    assert observed["topology"] == facts
    assert observed["runner_topology"] == facts
    materialized = stage.materialize_intended_configurations(tmp_path)
    selected = next(cell for cell in materialized if cell.manifest["seed_identity"]["label"] == 810001)
    expected_filenames = {"selection_receipt.json", "resolved_config.yaml", "COMPLETE"}
    assert {path.name for path in selected.output_path.iterdir()} == expected_filenames
    resolved = selected.output_path / "resolved_config.yaml"
    assert resolved.read_text(encoding="utf-8") == "horizon: 2000\n"
    assert not list(tmp_path.rglob("ranking_packet.json"))
    assert not list(tmp_path.rglob("independent_sampler_packet.json"))


def test_g4_nonzero_runner_never_emits_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    facts = {
        "global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1,
        "node_rank": 0, "node_size": 1, "host": "host", "pid": 1,
        "device": "cuda:0", "job_id": "job", "device_identity": None,
    }
    monkeypatch.setattr(runner, "_execution_facts", lambda: facts)
    original = runner.launch.launch_train

    def failing_launch(source: object, topology: object) -> int:
        return original(source, topology, runner=lambda config, **kwargs: 7)

    monkeypatch.setattr(runner.launch, "launch_train", failing_launch)
    assert runner.execute(_args(tmp_path, dry_run=False)) == 7
    assert not list(tmp_path.rglob("COMPLETE"))


def test_g4_success_without_snapshot_raises_and_never_emits_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts = {
        "global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1,
        "node_rank": 0, "node_size": 1, "host": "host", "pid": 1,
        "device": "cuda:0", "job_id": "job", "device_identity": None,
    }
    monkeypatch.setattr(runner, "_execution_facts", lambda: facts)
    original = runner.launch.launch_train

    def snapshotless_launch(source: object, topology: object) -> int:
        return original(source, topology, runner=lambda config, **kwargs: 0)

    monkeypatch.setattr(runner.launch, "launch_train", snapshotless_launch)
    with pytest.raises(RuntimeError, match="success without"):
        runner.execute(_args(tmp_path, dry_run=False))
    assert not list(tmp_path.rglob("COMPLETE"))


@pytest.mark.parametrize("launch_result", [0, 7])
def test_reused_complete_snapshot_refuses_before_launch_and_preserves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch_result: int
) -> None:
    """A prior attempt is evidence, not permission to reuse its artifact path."""
    expected_facts = {
        "global_rank": 0, "global_size": 1, "local_rank": 0, "local_size": 1,
        "node_rank": 0, "node_size": 1, "host": "host", "pid": 1,
        "device": "cuda:0", "job_id": "job", "device_identity": None,
    }
    monkeypatch.setattr(runner, "_execution_facts", lambda: dict(expected_facts))
    cells = stage.materialize_intended_configurations(tmp_path)
    selected = next(cell for cell in cells if cell.manifest["seed_identity"]["label"] == 810001)
    selected.output_path.mkdir(parents=True, exist_ok=True)
    sentinels = {
        selected.output_path / "selection_receipt.json": b'{"sentinel":"old-selection"}\n',
        selected.output_path / "resolved_config.yaml": b"old-resolved-config\n",
        selected.output_path / "COMPLETE": b"old-complete\n",
    }
    for path, contents in sentinels.items():
        path.write_bytes(contents)
    before = {path: path.read_bytes() for path in sentinels}

    calls = []

    def launch_double(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        return launch_result

    monkeypatch.setattr(runner.launch, "launch_train", launch_double)

    refusal = None
    result = None
    try:
        result = runner.execute(_args(tmp_path, dry_run=False))
    except BaseException as caught:
        refusal = caught

    assert not calls, "reusing a COMPLETE artifact path must refuse before launch"
    if refusal is None:
        assert isinstance(result, int) and result != 0, (
            "reusing a COMPLETE artifact path must return a nonzero refusal"
        )
    else:
        message = str(refusal).lower()
        assert any(token in message for token in ("reuse", "already", "complete", "artifact", "exists")), (
            f"unexpected reuse refusal: {refusal!r}"
        )
    assert {path: path.read_bytes() for path in sentinels} == before


def _r2_facts() -> dict[str, object]:
    """Return deterministic execution facts for R2 boundary probes."""

    return {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": "r2-sentinel-host",
        "pid": 123,
        "device": "cuda:0",
        "job_id": "r2-sentinel-job",
        "device_identity": {"kind": "cuda", "index": 0, "uuid": "GPU-r2"},
    }


def test_r2_g1_execute_non_q_refuses_before_materialization_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The execute-only Q guard must precede materialization and all output."""

    materialize_calls: list[object] = []

    def exploding_materializer(*args: object, **kwargs: object) -> object:
        materialize_calls.append((args, kwargs))
        raise AssertionError("non-Q execute reached materialization")

    monkeypatch.setattr(stage, "materialize_intended_configurations", exploding_materializer)
    monkeypatch.setattr(runner, "_execution_facts", lambda: _r2_facts())
    monkeypatch.setattr(runner.launch, "launch_train", lambda *a, **k: pytest.fail("launched"))

    with pytest.raises(runner.StageQSelectionError, match="only stage 'Q'"):
        runner.execute(_args(tmp_path, stage="O1", dry_run=False, seed_label=820001))

    assert not materialize_calls
    assert not list(tmp_path.rglob("selection_receipt.json"))


def test_r2_g1_execute_guard_rejects_both_directions(tmp_path: Path) -> None:
    """The positive Q path remains callable while non-Q remains refused."""

    assert runner.execute(_args(tmp_path, dry_run=True)) == 0
    with pytest.raises(runner.StageQSelectionError, match="only stage 'Q'"):
        runner.execute(_args(tmp_path / "non-q", stage="O1"))


@pytest.mark.parametrize("opener_name", ["path", "builtins", "io"])
def test_r2_g2_train_execute_never_reads_evaluation_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opener_name: str
) -> None:
    """Dry and real injected-launch arms reject every supported open route."""

    evaluation_path = Path(runner.train_config.__file__).with_name("manifests").joinpath(
        "evaluation.yaml"
    )
    assert evaluation_path.is_file()
    evaluation_text = evaluation_path.read_text(encoding="utf-8")
    assert "evaluated_checkpoints:" in evaluation_text
    assert "100000" in evaluation_text

    def reject_evaluation(path: object) -> None:
        if str(path).endswith("evaluation.yaml"):
            raise AssertionError(f"evaluation manifest read through {opener_name}")

    if opener_name == "path":
        original = Path.open

        def guarded_path_open(self: Path, *args: object, **kwargs: object):
            reject_evaluation(self)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", guarded_path_open)
    elif opener_name == "builtins":
        original = builtins.open

        def guarded_builtin_open(file: object, *args: object, **kwargs: object):
            reject_evaluation(file)
            return original(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    else:
        original = io.open

        def guarded_io_open(file: object, *args: object, **kwargs: object):
            reject_evaluation(file)
            return original(file, *args, **kwargs)

        monkeypatch.setattr(io, "open", guarded_io_open)

    assert runner.execute(_args(tmp_path / "dry")) == 0

    facts = _r2_facts()
    monkeypatch.setattr(runner, "_execution_facts", lambda: dict(facts))
    original_launch = runner.launch.launch_train

    def injected_launch(source: object, topology: object) -> int:
        def fake_runner(config: object, *, topology: object) -> int:
            output = Path(source.cell.output_path) / "resolved_config.yaml"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("r2: injected\n", encoding="utf-8")
            return 0

        return original_launch(source, topology, runner=fake_runner)

    monkeypatch.setattr(runner.launch, "launch_train", injected_launch)
    assert runner.execute(_args(tmp_path / "real", dry_run=False)) == 0
    assert list((tmp_path / "real").rglob("COMPLETE"))


@pytest.mark.parametrize("kind", ["file", "empty-directory", "symlink", "dangling-symlink"])
def test_r2_g3_occupied_output_path_rejects_every_path_kind_and_preserves_evidence(
    tmp_path: Path, kind: str
) -> None:
    """Every filesystem occupant blocks a claim without mutation."""

    path = tmp_path / "occupied"
    sentinel = b"prior-evidence\n"
    if kind == "file":
        path.write_bytes(sentinel)
    elif kind == "empty-directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target"
        target.write_bytes(sentinel)
        path.symlink_to(target)
    else:
        path.symlink_to(tmp_path / "missing-target")

    original_target = os.readlink(path) if path.is_symlink() else None
    original_target_bytes = path.resolve().read_bytes() if kind == "symlink" else None

    with pytest.raises(runner.StageQSelectionError, match="already occupied"):
        runner._refuse_occupied_output_path(path)
    if kind == "file":
        assert path.read_bytes() == sentinel
    elif kind == "empty-directory":
        assert path.is_dir() and not any(path.iterdir())
    else:
        assert path.is_symlink()
        assert os.readlink(path) == original_target
        if kind == "symlink":
            assert path.resolve().read_bytes() == original_target_bytes == sentinel
        else:
            assert not path.resolve().exists()


def test_r2_g3_unoccupied_output_path_is_accepted(tmp_path: Path) -> None:
    """A fresh leaf is claimed without rejecting or overwriting it."""

    path = tmp_path / "new-output"
    runner._refuse_occupied_output_path(path)
    assert not path.is_file() and not path.is_symlink()
    assert path.is_dir()


def _cooperative_execute_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    reserve_directory: bool,
    final_launch_outcome: int,
) -> tuple[dict[str, object], Path]:
    """Interleave a second real execute immediately before the first write."""

    args = _args(tmp_path, dry_run=False)
    planned_cells = stage.materialize_intended_configurations(tmp_path)
    selected_output = next(
        cell.output_path
        for cell in planned_cells
        if cell.manifest["seed_identity"]["label"] == 810001
    )
    state: dict[str, object] = {
        "reentered": False,
        "reserved": False,
        "launches": [],
        "first_status": None,
        "second_status": None,
        "after_second": None,
        "after_first": None,
    }
    facts = _r2_facts()
    monkeypatch.setattr(runner, "_execution_facts", lambda: dict(facts))

    def fake_launch(source: object, topology: object) -> int:
        launches = state["launches"]
        assert isinstance(launches, list)
        label = f"launch-{len(launches) + 1}"
        launches.append(label)
        output = Path(source.cell.output_path)
        (output / "resolved_config.yaml").write_text(label + "\n", encoding="utf-8")
        return final_launch_outcome if label == "launch-1" else 0

    monkeypatch.setattr(runner.launch, "launch_train", fake_launch)
    original_check = runner._refuse_occupied_output_path

    def check_and_optionally_reserve(path: Path) -> None:
        original_check(path)
        if (
            reserve_directory
            and os.environ.get("TPEN_R2_ATOMIC_REPAIR") != "1"
            and not state["reserved"]
            and not path.exists()
        ):
            path.mkdir(parents=True, exist_ok=False)
            state["reserved"] = True

    monkeypatch.setattr(runner, "_refuse_occupied_output_path", check_and_optionally_reserve)
    original_write = runner._write_json

    def output_bytes(path: Path) -> dict[str, bytes | None]:
        return {
            name: (path / name).read_bytes() if (path / name).exists() else None
            for name in ("selection_receipt.json", "resolved_config.yaml", "COMPLETE")
        }

    def interleave_before_first_write(path: Path, payload: object) -> None:
        if not state["reentered"]:
            state["reentered"] = True
            try:
                try:
                    state["second_status"] = "success" if runner.execute(args) == 0 else "nonzero"
                except runner.StageQSelectionError:
                    state["second_status"] = "StageQSelectionError"
                state["after_second"] = output_bytes(selected_output)
            finally:
                state["reentered"] = False
        original_write(path, payload)

    monkeypatch.setattr(runner, "_write_json", interleave_before_first_write)
    try:
        state["first_status"] = "success" if runner.execute(args) == 0 else "nonzero"
    except runner.StageQSelectionError:
        state["first_status"] = "StageQSelectionError"

    state["after_first"] = output_bytes(selected_output)
    return state, selected_output


@pytest.mark.parametrize(
    ("reserve_directory", "final_launch_outcome"),
    [(False, 0), (False, 7), (True, 0), (True, 7)],
    ids=(
        "ordinary-launch-success",
        "ordinary-launch-failure",
        "atomic-launch-success",
        "atomic-launch-failure",
    ),
)
def test_r2_r4_cooperative_execute_race_has_one_winner_and_preserves_winner_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reserve_directory: bool,
    final_launch_outcome: int,
) -> None:
    """The real race oracle accepts either invocation as the atomic winner."""

    state, output_path = _cooperative_execute_race(
        tmp_path,
        monkeypatch,
        reserve_directory=reserve_directory,
        final_launch_outcome=final_launch_outcome,
    )
    statuses = [state["first_status"], state["second_status"]]
    assert statuses.count("StageQSelectionError") == 1
    assert sum(status in {"success", "nonzero"} for status in statuses) == 1
    launches = state["launches"]
    assert isinstance(launches, list) and len(launches) == 1
    after_second = state["after_second"]
    after_first = state["after_first"]
    assert isinstance(after_second, dict) and isinstance(after_first, dict)
    if state["second_status"] != "StageQSelectionError":
        assert state["first_status"] == "StageQSelectionError"
        assert after_first == after_second
    else:
        assert after_first["resolved_config.yaml"] == b"launch-1\n"
        if final_launch_outcome == 0:
            assert after_first["COMPLETE"] == b"completed\n"
        else:
            assert after_first["COMPLETE"] is None
