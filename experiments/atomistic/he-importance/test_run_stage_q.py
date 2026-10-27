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
optimizer_grid = importlib.import_module("experiments.atomistic.he-importance.optimizer_grid")
runner = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")
stage = stage_coordinate


def _fake_adam(
    params: object,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    del params, lr, betas, eps, weight_decay


@pytest.fixture(autouse=True)
def _runtime_torch_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the runtime-signature guard testable when the optional extra is absent."""

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(optim=SimpleNamespace(Adam=_fake_adam)),
        )


def _cells(tmp_path: Path) -> tuple[object, ...]:
    return stage_coordinate.materialize_intended_configurations(
        tmp_path, availability=optimizer_grid.roster_availability()
    )


def _args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values = {
        "output_root": tmp_path,
        "stage": "Q",
        "context": "O-raw",
        "optimizer": "adam",
        "seed_label": 810001,
        "walkers": stage_coordinate.COMMITTED_WALKERS,
        "dry_run": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_q_materialization_has_two_seed_labeled_cells_and_selection_is_explicit(tmp_path: Path) -> None:
    cells = _cells(tmp_path)
    q_cells = [cell for cell in cells if cell.manifest["stage"] == "Q"]
    labels = [cell.manifest["seed_identity"]["label"] for cell in q_cells]
    assert len(q_cells) == 480
    assert labels.count(810_001) == labels.count(810_002) == 240

    with pytest.raises(runner.StageQSelectionError, match="ambiguous"):
        runner.select_cell(cells)

    selected = runner.select_cell(cells, seed_label=810_002)
    assert selected.selected.as_dict() == {
        "stage": "Q",
        "context": "O-raw",
        "optimizer": {
            "method": "adam",
            "levels": {
                "lr": 0.005,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "parameter_groups": 1,
                "schedule": "constant",
            },
        },
        "seed_label": 810_002,
        "status": "available",
    }
    assert any(item.seed_label == 810_001 for item in selected.discarded)


def test_selection_refuses_zero_matches_and_never_first_matches(tmp_path: Path) -> None:
    cells = _cells(tmp_path)
    with pytest.raises(runner.StageQSelectionError, match="zero admitted"):
        runner.select_cell(cells, seed_label=999_999)


def _q_adam_cells(cells: tuple[object, ...], *, context: str) -> list[object]:
    return [
        cell
        for cell in cells
        if cell.manifest["stage"] == "Q"
        and cell.manifest["scientific_identity"]["context"]["name"] == context
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["seed_identity"]["label"] == 810001
    ]


def test_truthfulness_guard_refuses_each_nonmatching_adam_level_cell(tmp_path: Path) -> None:
    cells = _q_adam_cells(_cells(tmp_path), context="O-raw")
    expected = {"lr": 0.005, "beta2": 0.999}
    nonmatching = [
        cell
        for cell in cells
        if any(cell.manifest["scientific_identity"]["optimizer"]["levels"][key] != value
               for key, value in expected.items())
    ]
    assert len(nonmatching) == 7

    for cell in nonmatching:
        levels = cell.manifest["scientific_identity"]["optimizer"]["levels"]
        with pytest.raises(runner.StageQSelectionError, match="lr|beta2") as refusal:
            runner.select_cell((cell,), seed_label=810001)
        assert any(field in str(refusal.value) for field in ("lr", "beta2")), levels


def test_o_composite_selection_consumes_the_authority_literal(tmp_path: Path) -> None:
    composite = _q_adam_cells(_cells(tmp_path), context="O-composite")
    target = next(
        cell
        for cell in composite
        if cell.manifest["scientific_identity"]["optimizer"]["levels"]
        == {"lr": 0.005, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
            "weight_decay": 0.0, "parameter_groups": 1, "schedule": "constant"}
    )
    selection = runner.select_cell((target,), context="O-composite", seed_label=810001)
    assert selection.selected.context == "O-composite"
    assert selection.selected.seed_label == 810001


def test_o_composite_selection_has_eight_adam_authority_cells_and_schema_validates(
    tmp_path: Path,
) -> None:
    composite = _q_adam_cells(_cells(tmp_path), context="O-composite")
    authority = optimizer_grid.load_authority()
    adam = next(family for family in authority["families"] if family["family"] == "adam")
    expected_pairs = {
        (lr, beta2)
        for lr in adam["varied"]["lr"]
        for beta2 in adam["varied"]["beta2"]
    }
    seeded = [
        cell
        for cell in composite
        if cell.manifest["seed_identity"]["label"] == 810001
    ]
    assert len(seeded) == len(expected_pairs)
    observed_pairs = {
        (
            cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"],
            cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"],
        )
        for cell in seeded
    }
    assert observed_pairs == expected_pairs
    for lr, beta2 in expected_pairs:
        matches = [
            cell
            for cell in seeded
            if cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == lr
            and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == beta2
        ]
        assert len(matches) == 1
        selection = runner.select_cell(
            matches,
            context="O-composite",
            lr=lr,
            beta2=beta2,
            seed_label=810001,
        )
        assert selection.selected.context == "O-composite"
        assert selection.selected.seed_label == 810001

    target = next(
        cell
        for cell in seeded
        if cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
    resolved = runner.train_config.resolve_train_config(target)
    import tpen.hi_schema as schema

    schema.validate_hi_train_config(resolved)
    assert resolved.model.basis._target_ == "tpen.nn.basis.BoundedDistanceBasis"
    assert resolved.model.basis.inner._target_ == "tpen.nn.RawCoordinateBasis"
    assert resolved.model.basis.length == 1.0
    assert resolved.model.embedding._target_ == "tpen.nn.embedding.matched_start_embedding"
    assert resolved.model.embedding.raw.in_features == 4
    assert resolved.model.embedding.raw.pair_input_channels == 0
    assert resolved.model.embedding.augmented.in_features == 5
    assert resolved.model.embedding.augmented.pair_input_channels == 3
    assert [factor._target_ for factor in resolved.model.factors] == [
        "tpen.nn.ElectronElectronCusp",
        "tpen.nn.ElectronNucleusCusp",
        "tpen.nn.jastrow.BoundedTwoCoefficientJastrow",
    ]
    layer = resolved.model.layers[0]
    assert layer.mixing._target_ == "tpen.nn.CompositeMixing"
    assert [producer._target_ for producer in layer.mixing.producers] == [
        "tpen.nn.LinearEquivariantMixing",
        "tpen.nn.EquivariantMixing",
    ]
    assert layer.mixing.activation._target_ == "tpen.nn.ChannelPreservingMLPActivation"
    assert layer.path_aggregation.activation._target_ == "tpen.nn.ChannelPreservingMLPActivation"
    assert layer.update._target_ == "tpen.nn.ResidualUpdater"


def test_o_composite_seed_paths_are_independently_enumerated(
    tmp_path: Path,
) -> None:
    target = next(
        cell
        for cell in _q_adam_cells(_cells(tmp_path), context="O-composite")
        if cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
        and cell.manifest["seed_identity"]["label"] == 810001
    )
    resolved = runner.train_config.resolve_train_config(target)
    model = runner.OmegaConf.to_container(resolved.model, resolve=False)

    def walk(value: object, path: str = "model") -> set[str]:
        if isinstance(value, dict):
            found = set()
            if isinstance(value.get("initializer"), dict):
                found.add(f"{path}.initializer.seed")
            for key, nested in value.items():
                found.update(walk(nested, f"{path}.{key}"))
            return found
        if isinstance(value, (tuple, list)):
            found = set()
            for index, nested in enumerate(value):
                found.update(walk(nested, f"{path}.{index}"))
            return found
        return set()

    paths = walk(model)
    assert paths == {
        "model.embedding.raw.initializer.seed",
        "model.embedding.augmented.initializer.seed",
        "model.layers.0.mixing.activation.initializer.seed",
        "model.layers.0.path_aggregation.initializer.seed",
        "model.layers.0.path_aggregation.activation.initializer.seed",
    }
    expected = target.seed_streams["model_initialization"]
    assert all(runner.OmegaConf.select(resolved, path) == expected for path in paths)


def test_o_composite_jastrow_removal_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = runner.train_config.resolve_train_config

    def drop_jastrow(source: object) -> object:
        resolved = original(source)
        resolved.model.factors.pop()
        resolved.runner.model.factors.pop()
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", drop_jastrow)
    with pytest.raises(runner.StageQSelectionError, match="factors.*dd964863"):
        runner.select_cell(
            _q_adam_cells(_cells(tmp_path), context="O-composite"),
            context="O-composite",
            seed_label=810001,
        )


def test_o_composite_launch_path_matches_selection_path(tmp_path: Path) -> None:
    target = next(
        cell
        for cell in _q_adam_cells(_cells(tmp_path), context="O-composite")
        if cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
        and cell.manifest["seed_identity"]["label"] == 810001
    )
    selection = runner.select_cell(
        (target,), context="O-composite", seed_label=810001
    )
    packet, selected = runner._resolve_selected_config(selection)
    launched = runner.launch.prepare_train_launch(packet, _r2_facts())
    assert runner.OmegaConf.to_container(launched.config, resolve=True) == runner.OmegaConf.to_container(
        selected, resolve=True
    )


def test_o_composite_launch_config_divergence_is_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = next(
        cell
        for cell in _q_adam_cells(_cells(tmp_path), context="O-composite")
        if cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
        and cell.manifest["seed_identity"]["label"] == 810001
    )
    selection = runner.select_cell(
        (target,), context="O-composite", seed_label=810001
    )
    packet, selected = runner._resolve_selected_config(selection)
    original_prepare = runner.launch.prepare_train_launch

    def divergent_prepare(source: object, topology: object) -> object:
        plan = original_prepare(source, topology)
        runner.OmegaConf.update(
            plan.config,
            "model.layers.0.update._target_",
            "tpen.nn.ReplacementUpdater",
            merge=False,
        )
        return plan

    monkeypatch.setattr(runner.launch, "prepare_train_launch", divergent_prepare)
    launched = runner.launch.prepare_train_launch(packet, _r2_facts())
    with pytest.raises(AssertionError):
        assert runner.OmegaConf.to_container(launched.config, resolve=True) == runner.OmegaConf.to_container(
            selected, resolve=True
        )


def test_truthfulness_guard_leaves_exactly_one_mechanics_cell(tmp_path: Path) -> None:
    selection = runner.select_cell(_cells(tmp_path), seed_label=810001)
    levels = selection.selected.optimizer["levels"]
    assert selection.selected.context == "O-raw"
    assert selection.selected.optimizer["method"] == "adam"
    assert levels["lr"] == 0.005
    assert levels["beta2"] == 0.999
    assert len([item for item in _q_adam_cells(_cells(tmp_path), context="O-raw")
                 if item.manifest["scientific_identity"]["optimizer"]["levels"] == levels]) == 1


def test_truthfulness_guard_reads_consumed_lr_from_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = next(
        cell
        for cell in _q_adam_cells(_cells(tmp_path), context="O-raw")
        if cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
    original = runner.train_config.resolve_train_config

    def resolve_with_different_lr(source: object) -> object:
        resolved = original(source)
        resolved.optimizer.lr = 0.006
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_with_different_lr)
    with pytest.raises(runner.StageQSelectionError, match="lr"):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize(
    ("axis", "value"),
    [("context", "O-unadmitted"), ("optimizer", "kfac"), ("optimizer", "sr")],
)
def test_r3_selection_axes_reject_unadmitted_values_before_all_execution_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    value: str,
) -> None:
    """Context and optimizer are both required admission coordinates."""

    cells = _cells(tmp_path)
    selection_kwargs = {axis: value, "seed_label": 810001}
    initial_message = "dd964863" if axis == "context" else "zero admitted"
    with pytest.raises(runner.StageQSelectionError, match=initial_message):
        runner.select_cell(cells, **selection_kwargs)

    for name, target in (
        ("_q_training_packet", runner),
        ("_execution_facts", runner),
    ):
        monkeypatch.setattr(
            target,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(f"{_name} was reached"),
        )
    monkeypatch.setattr(
        runner.train_config,
        "resolve_train_config",
        lambda *args, **kwargs: pytest.fail("config resolution was reached"),
    )
    monkeypatch.setattr(
        runner,
        "_refuse_occupied_output_path",
        lambda *args, **kwargs: pytest.fail("output claim was reached"),
    )
    monkeypatch.setattr(
        runner.launch,
        "launch_train",
        lambda *args, **kwargs: pytest.fail("launch was reached"),
    )

    for dry_run in (True, False):
        args = _args(
            tmp_path / f"{axis}-{value}-{dry_run}",
            dry_run=dry_run,
            **selection_kwargs,
        )
        expected_message = "dd964863" if axis == "context" else "zero admitted"
        with pytest.raises(runner.StageQSelectionError, match=expected_message):
            runner.execute(args)
        assert not list(args.output_root.rglob("selection_receipt.json"))
        assert not list(args.output_root.rglob("resolved_config.yaml"))
        assert not list(args.output_root.rglob("COMPLETE"))


@pytest.mark.parametrize("dry_run", [True, False])
def test_r3_default_axes_reach_frozen_exact_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    """The valid Q/O-raw/Adam axis remains a real executable selection."""

    cells = _cells(tmp_path)
    expected_cell = next(
        cell
        for cell in cells
        if cell.manifest["stage"] == "Q"
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
        and cell.manifest["seed_identity"]["label"] == 810001
    )
    expected_descriptor = {
        "stage": "Q",
        "context": "O-raw",
        "optimizer": expected_cell.manifest["scientific_identity"]["optimizer"],
        "seed_label": 810001,
        "status": "available",
    }
    expected_type = type(expected_cell)
    expected_hash = expected_cell.content_hash
    expected_manifest = expected_cell.manifest
    selected = runner.select_cell(cells, seed_label=810001)
    assert selected.selected.as_dict() == expected_descriptor
    assert type(selected.cell) is expected_type
    assert selected.cell.content_hash == expected_hash
    assert selected.cell.manifest == expected_manifest

    observed: dict[str, object] = {}
    if not dry_run:
        monkeypatch.setattr(runner, "_execution_facts", lambda: {"device": "cuda:0"})

        def injected_launch(packet: object, facts: object, **kwargs: object) -> int:
            observed["packet"] = packet
            observed["facts"] = facts
            return 7

        monkeypatch.setattr(runner.launch, "launch_train", injected_launch)

    assert runner.execute(
        _args(tmp_path / f"valid-{dry_run}", dry_run=dry_run)
    ) == (0 if dry_run else 7)
    if dry_run:
        report = json.loads(capsys.readouterr().out)
        assert report["selection"]["selected"] == expected_descriptor
        assert report["train_packet"]["content_hash"] == expected_hash
    else:
        packet = observed["packet"]
        assert type(packet.cell) is expected_type
        assert packet.cell.content_hash == expected_hash
        assert packet.cell.manifest == expected_manifest


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


def test_cli_2048_walkers_rebuilds_identity_and_consumes_resolved_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = runner.build_parser().parse_args(
        ["--dry-run", "--walkers", "2048", "--seed-label", "810001", "--output-root", str(tmp_path)]
    )
    assert runner.execute(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["walkers"] == 2_048
    assert report["resolved_config"]["sampler"]["n_walkers"] == 2_048
    assert report["resolved_config"]["runner"]["sampler"]["n_walkers"] == 2_048

    committed_root = tmp_path / "committed"
    committed_args = runner.build_parser().parse_args(
        ["--dry-run", "--seed-label", "810001", "--output-root", str(committed_root)]
    )
    assert runner.execute(committed_args) == 0
    committed_report = json.loads(capsys.readouterr().out)
    assert committed_report["walkers"] == 4_096
    assert committed_report["train_packet"]["content_hash"] != report["train_packet"]["content_hash"]


def test_cli_refuses_unsupported_walker_count(tmp_path: Path) -> None:
    args = runner.build_parser().parse_args(
        ["--dry-run", "--walkers", "1000", "--seed-label", "810001", "--output-root", str(tmp_path)]
    )
    with pytest.raises(runner.StageQSelectionError, match="unsupported Stage-Q walker count"):
        runner.execute(args)


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

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=FakeCUDA(), optim=SimpleNamespace(Adam=_fake_adam)),
    )
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
        "context": "O-raw",
        "optimizer": {
            "method": "adam",
            "levels": {
                "lr": 0.005,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.0,
                "parameter_groups": 1,
                "schedule": "constant",
            },
        },
        "seed_label": 810001,
        "status": "available",
    }
    cells = _cells(tmp_path)
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
        runner.select_cell(cells + (forward.cell,), seed_label=810001)

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
        [{"scientific_identity": {
            "context": {"name": "O-raw", "overrides": {}},
            "optimizer": {"method": "adam", "levels": {
                "lr": 0.005, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
                "weight_decay": 0.0, "parameter_groups": 1, "schedule": "constant",
            }},
        },
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
    cells = _cells(tmp_path)
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
        _cells(tmp_path), seed_label=810001
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

    def launch_wrapper(source: object, topology: object, **kwargs: object) -> int:
        observed["topology"] = topology
        observed["packet_type"] = type(source)
        guarded_runner = kwargs["runner"]

        def recording_run_from_config(config: object, *, topology: object) -> int:
            observed["runner_topology"] = topology
            output = Path(config.run.dir) / "resolved_config.yaml"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("horizon: 2000\n", encoding="utf-8")
            observed["config"] = config
            return 0

        monkeypatch.setattr(runner.launch, "run_from_config", recording_run_from_config)
        assert callable(guarded_runner)
        return original(source, topology, runner=guarded_runner)

    monkeypatch.setattr(runner.launch, "launch_train", launch_wrapper)
    assert runner.execute(_args(tmp_path, dry_run=False)) == 0
    assert observed["packet_type"] is expected_packet_type
    assert observed["topology"] == facts
    assert observed["runner_topology"] == facts
    materialized = _cells(tmp_path)
    selected = next(
        cell for cell in materialized
        if cell.manifest["seed_identity"]["label"] == 810001
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
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
    monkeypatch.setattr(runner.launch, "run_from_config", lambda config, *, topology: 7)
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
    monkeypatch.setattr(runner.launch, "run_from_config", lambda config, *, topology: 0)
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
    cells = _cells(tmp_path)
    selected = next(
        cell for cell in cells
        if cell.manifest["seed_identity"]["label"] == 810001
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
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
    def injected_run_from_config(config: object, *, topology: object) -> int:
        output = Path(config.run.dir) / "resolved_config.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("r2: injected\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner.launch, "run_from_config", injected_run_from_config)
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
    planned_cells = _cells(tmp_path)
    selected_output = next(
        cell.output_path
        for cell in planned_cells
        if cell.manifest["seed_identity"]["label"] == 810001
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
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

    def fake_launch(source: object, topology: object, **kwargs: object) -> int:
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


def test_selection_resolves_only_the_requested_level_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = _q_adam_cells(_cells(tmp_path), context="O-raw")
    original = runner.train_config.resolve_train_config
    resolved_sources: list[object] = []

    def resolve_declared_levels(source: object) -> object:
        resolved_sources.append(source)
        resolved = original(source)
        levels = source.cell.manifest["scientific_identity"]["optimizer"]["levels"]
        runner.OmegaConf.update(resolved, "optimizer.lr", levels["lr"], merge=False)
        runner.OmegaConf.update(resolved, "runner.optimizer.lr", levels["lr"], merge=False)
        runner.OmegaConf.update(
            resolved,
            "optimizer.betas",
            [levels["beta1"], levels["beta2"]],
            merge=False,
            force_add=True,
        )
        runner.OmegaConf.update(
            resolved,
            "runner.optimizer.betas",
            [levels["beta1"], levels["beta2"]],
            merge=False,
            force_add=True,
        )
        runner.OmegaConf.update(
            resolved, "optimizer.eps", levels["epsilon"], merge=False, force_add=True
        )
        runner.OmegaConf.update(
            resolved, "runner.optimizer.eps", levels["epsilon"], merge=False, force_add=True
        )
        runner.OmegaConf.update(
            resolved,
            "optimizer.weight_decay",
            levels["weight_decay"],
            merge=False,
            force_add=True,
        )
        runner.OmegaConf.update(
            resolved,
            "runner.optimizer.weight_decay",
            levels["weight_decay"],
            merge=False,
            force_add=True,
        )
        return resolved

    monkeypatch.setattr(
        runner.train_config, "resolve_train_config", resolve_declared_levels
    )
    selected = runner.select_cell(cells, seed_label=810001, lr=0.005, beta2=0.999)
    assert selected.selected.optimizer["levels"]["lr"] == 0.005
    assert selected.selected.optimizer["levels"]["beta2"] == 0.999
    assert len(resolved_sources) == 1


def test_mutant_without_level_consumption_is_red_for_seven_of_eight_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The level mapping is load-bearing: a no-op mutant rejects 7/8 cells."""

    cells = _q_adam_cells(_cells(tmp_path), context="O-raw")
    fixed = {
        "lr": 0.005,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "parameter_groups": 1,
        "schedule": "constant",
    }
    original = runner.train_config._declared_adam_levels

    def validation_preserving_mutant(manifest: object) -> dict[str, object]:
        original(manifest)
        return fixed

    monkeypatch.setattr(
        runner.train_config, "_declared_adam_levels", validation_preserving_mutant
    )

    refused = 0
    for cell in cells:
        levels = cell.manifest["scientific_identity"]["optimizer"]["levels"]
        try:
            runner.select_cell(
                (cell,),
                lr=levels["lr"],
                beta2=levels["beta2"],
                seed_label=810001,
            )
        except runner.StageQSelectionError:
            refused += 1
    assert refused == 7


def test_truthfulness_guard_refuses_consumed_non_adam_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = next(
        cell
        for cell in _q_adam_cells(_cells(tmp_path), context="O-raw")
        if cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
    original = runner.train_config.resolve_train_config

    def resolve_non_adam(source: object) -> object:
        resolved = original(source)
        resolved.optimizer._target_ = "torch.optim.SGD"
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_non_adam)
    with pytest.raises(runner.StageQSelectionError, match="optimizer._target_"):
        runner.select_cell((target,), seed_label=810001)


def test_truthfulness_guard_refuses_extra_declared_level_key(
    tmp_path: Path,
) -> None:
    target = next(
        cell
        for cell in _q_adam_cells(_cells(tmp_path), context="O-raw")
        if cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
    manifest = stage._thaw(target.manifest)
    manifest["scientific_identity"]["optimizer"]["levels"]["sentinel_extra"] = 1
    stage.validate_materialized_manifest(manifest)
    mutated_hash = stage.content_hash(manifest)
    mutated = replace(
        target,
        manifest=stage._freeze(manifest),
        content_hash=mutated_hash,
        output_path=target.output_path.parent / mutated_hash,
    )
    with pytest.raises(runner.StageQSelectionError, match="sentinel_extra"):
        runner.select_cell((mutated,), seed_label=810001)


def _rehash_cell(cell: object, mutate: object) -> object:
    manifest = stage._thaw(cell.manifest)
    mutate(manifest)
    stage.validate_materialized_manifest(manifest)
    mutated_hash = stage.content_hash(manifest)
    return replace(
        cell,
        manifest=stage._freeze(manifest),
        content_hash=mutated_hash,
        output_path=cell.output_path.parent / mutated_hash,
    )


def _mechanics_cell(cells: tuple[object, ...], *, seed_label: int = 810001) -> object:
    return next(
        cell
        for cell in cells
        if cell.manifest["stage"] == "Q"
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
        and cell.manifest["seed_identity"]["label"] == seed_label
    )


def test_truthfulness_guard_refuses_o_raw_b11_declared_override(tmp_path: Path) -> None:
    target = _mechanics_cell(_cells(tmp_path))

    def declare_b11(manifest: dict[str, object]) -> None:
        manifest["scientific_identity"]["context"]["overrides"]["input_features"] = "B11"

    mutated = _rehash_cell(target, declare_b11)
    with pytest.raises(runner.StageQSelectionError, match="input_features.*B11.*B00.*dd964863"):
        runner.select_cell((mutated,), seed_label=810001)


def test_truthfulness_guard_refuses_consumed_replacement_updater(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_replacement(source: object) -> object:
        resolved = original(source)
        for path in (
            "model.layers.0.update._target_",
            "runner.model.layers.0.update._target_",
        ):
            runner.OmegaConf.update(
                resolved, path, "tpen.nn.ReplacementUpdater", merge=False
            )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_replacement)
    with pytest.raises(runner.StageQSelectionError, match="feature_update.*ReplacementUpdater.*dd964863"):
        runner.select_cell((target,), seed_label=810001)


def test_execute_refuses_consumed_replacement_updater_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts = _r2_facts()
    monkeypatch.setattr(runner, "_execution_facts", lambda: dict(facts))
    original = runner.train_config.resolve_train_config

    def resolve_replacement(source: object) -> object:
        resolved = original(source)
        for path in (
            "model.layers.0.update._target_",
            "runner.model.layers.0.update._target_",
        ):
            runner.OmegaConf.update(
                resolved, path, "tpen.nn.ReplacementUpdater", merge=False
            )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_replacement)
    with pytest.raises(runner.StageQSelectionError, match="feature_update.*ReplacementUpdater"):
        runner.execute(_args(tmp_path, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


def test_truthfulness_guard_refuses_runner_optimizer_lr_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_runner_lr(source: object) -> object:
        resolved = original(source)
        runner.OmegaConf.update(resolved, "runner.optimizer.lr", 0.015, merge=False)
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_runner_lr)
    with pytest.raises(runner.StageQSelectionError, match="runner.optimizer.lr"):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_late_launch_resolution_drift_before_run_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walkers: int
) -> None:
    original_select = runner.select_cell
    original_resolve = runner.train_config.resolve_train_config
    calls = {"resolve": 0, "run_from_config": 0}

    def one_candidate_select(cells: object, **kwargs: object) -> object:
        materialized = tuple(cells)
        return original_select((_mechanics_cell(materialized),), **kwargs)

    def resolve_with_late_drift(source: object) -> object:
        calls["resolve"] += 1
        resolved = original_resolve(source)
        if calls["resolve"] == 3:
            runner.OmegaConf.update(resolved, "runner.optimizer.lr", 0.015, merge=False)
            runner.OmegaConf.update(resolved, "optimizer.lr", 0.015, merge=False)
        return resolved

    monkeypatch.setattr(runner, "select_cell", one_candidate_select)
    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_with_late_drift)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    monkeypatch.setattr(
        runner.launch,
        "run_from_config",
        lambda config, *, topology: calls.__setitem__("run_from_config", calls["run_from_config"] + 1),
    )
    with pytest.raises(runner.StageQSelectionError, match="lr"):
        runner.execute(_args(tmp_path, walkers=walkers, dry_run=False))
    assert calls == {"resolve": 3, "run_from_config": 0}


def test_execute_guarded_runner_delegates_recorded_config_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    facts = _r2_facts()
    observed: dict[str, object] = {}
    monkeypatch.setattr(runner, "_execution_facts", lambda: dict(facts))

    def recording_run_from_config(config: object, *, topology: object) -> int:
        observed["calls"] = int(observed.get("calls", 0)) + 1
        observed["config"] = runner.OmegaConf.to_container(config, resolve=True)
        output = Path(config.run.dir) / "resolved_config.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("guarded: true\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner.launch, "run_from_config", recording_run_from_config)
    assert runner.execute(_args(tmp_path, dry_run=False)) == 0
    report = json.loads(capsys.readouterr().out)
    assert observed["calls"] == 1
    assert observed["config"] == report["resolved_config"]


def test_truthfulness_guard_rejects_non_residual_base_and_consumed_updater(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original_compose = runner.train_config._compose_base_config

    def compose_non_residual() -> object:
        config = original_compose()
        runner.OmegaConf.update(
            config,
            "model.layers.0.update._target_",
            "tpen.nn.ReplacementUpdater",
            merge=False,
        )
        return config

    monkeypatch.setattr(runner.train_config, "_compose_base_config", compose_non_residual)
    with pytest.raises(
        runner.StageQSelectionError,
        match="feature_update.*ReplacementUpdater.*dd964863",
    ):
        runner.select_cell((target,), seed_label=810001)


def _update_model_views(config: object, updates: dict[str, object]) -> None:
    for prefix in ("model", "runner.model"):
        for field, value in updates.items():
            runner.OmegaConf.update(
                config,
                f"{prefix}.{field}",
                value,
                merge=False,
                force_add=True,
            )


def test_truthfulness_guard_refuses_runner_only_updater_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_runner_replacement(source: object) -> object:
        resolved = original(source)
        runner.OmegaConf.update(
            resolved,
            "runner.model.layers.0.update._target_",
            "tpen.nn.ReplacementUpdater",
            merge=False,
        )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_runner_replacement)
    with pytest.raises(runner.StageQSelectionError, match="runner.model.layers.*ReplacementUpdater"):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_runner_only_updater_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walkers: int
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_runner_replacement(source: object) -> object:
        resolved = original(source)
        runner.OmegaConf.update(
            resolved,
            "runner.model.layers.0.update._target_",
            "tpen.nn.ReplacementUpdater",
            merge=False,
        )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_runner_replacement)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(runner.StageQSelectionError, match="runner.model.layers.*ReplacementUpdater"):
        runner.execute(_args(tmp_path, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


def _bounded_distance_basis() -> dict[str, object]:
    return {
        "_target_": "tpen.nn.basis.BoundedDistanceBasis",
        "inner": {
            "_target_": "tpen.nn.RawCoordinateBasis",
            "spatial_dim": 3,
            "include_spin": True,
        },
        "length": 1.0,
    }


def test_truthfulness_guard_refuses_bounded_distance_b00_counterexample_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_bounded(source: object) -> object:
        resolved = original(source)
        _update_model_views(
            resolved,
            {
                "basis": _bounded_distance_basis(),
                "embedding.in_features": 5,
                "embedding.pair_input_channels": 3,
            },
        )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_bounded)
    with pytest.raises(
        runner.StageQSelectionError,
        match="input_features.*BoundedDistanceBasis.*dd964863",
    ):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_bounded_distance_b00_counterexample_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walkers: int
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_bounded(source: object) -> object:
        resolved = original(source)
        _update_model_views(
            resolved,
            {
                "basis": _bounded_distance_basis(),
                "embedding.in_features": 5,
                "embedding.pair_input_channels": 3,
            },
        )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_bounded)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(runner.StageQSelectionError, match="input_features.*BoundedDistanceBasis.*dd964863"):
        runner.execute(_args(tmp_path, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


def test_truthfulness_guard_refuses_bounded_distance_basis_only_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_bounded(source: object) -> object:
        resolved = original(source)
        _update_model_views(resolved, {"basis": _bounded_distance_basis()})
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_bounded)
    with pytest.raises(
        runner.StageQSelectionError,
        match="input_features.*BoundedDistanceBasis.*dd964863",
    ):
        runner.select_cell((target,), seed_label=810001)


def test_truthfulness_guard_refuses_explicit_null_betas_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_null_betas(source: object) -> object:
        resolved = original(source)
        for path in ("optimizer.betas", "runner.optimizer.betas"):
            runner.OmegaConf.update(resolved, path, None, merge=False, force_add=True)
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_null_betas)
    with pytest.raises(runner.StageQSelectionError, match="optimizer.betas"):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_explicit_null_betas_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walkers: int
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_null_betas(source: object) -> object:
        resolved = original(source)
        for path in ("optimizer.betas", "runner.optimizer.betas"):
            runner.OmegaConf.update(resolved, path, None, merge=False, force_add=True)
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_null_betas)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(runner.StageQSelectionError, match="optimizer.betas"):
        runner.execute(_args(tmp_path, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


def test_model_fingerprint_normalizes_only_initializer_seeds(tmp_path: Path) -> None:
    cells = [
        cell
        for cell in _cells(tmp_path)
        if cell.manifest["stage"] == "Q"
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    ]
    first = next(cell for cell in cells if cell.manifest["seed_identity"]["label"] == 810001)
    second = next(cell for cell in cells if cell.manifest["seed_identity"]["label"] == 810002)
    first_model = runner.OmegaConf.to_container(
        runner.train_config.resolve_train_config(first).runner.model,
        resolve=True,
    )
    second_model = runner.OmegaConf.to_container(
        runner.train_config.resolve_train_config(second).runner.model,
        resolve=True,
    )
    assert runner._model_fingerprint(first_model) == runner._model_fingerprint(second_model)
    assert runner._model_fingerprint(first_model) == runner._CONTROL_MODEL_FINGERPRINT

    changed = json.loads(json.dumps(first_model))
    changed["embedding"]["hidden_channels"] = 64
    assert runner._model_fingerprint(changed) != runner._model_fingerprint(first_model)


def test_truthfulness_guard_accepts_committed_control_at_both_seed_labels(
    tmp_path: Path,
) -> None:
    cells = _cells(tmp_path)
    for seed_label in (810001, 810002):
        selected = runner.select_cell(cells, seed_label=seed_label)
        assert selected.selected.seed_label == seed_label


def test_truthfulness_guard_refuses_spinless_embedding_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_spinless(source: object) -> object:
        resolved = original(source)
        _update_model_views(resolved, {"embedding.include_spins": False})
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_spinless)
    with pytest.raises(runner.StageQSelectionError, match="include_spins.*dd964863"):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_spinless_embedding_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walkers: int
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_spinless(source: object) -> object:
        resolved = original(source)
        _update_model_views(resolved, {"embedding.include_spins": False})
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_spinless)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(runner.StageQSelectionError, match="include_spins.*dd964863"):
        runner.execute(_args(tmp_path, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


def test_truthfulness_guard_refuses_auxiliary_embedding_channels_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_aux(source: object) -> object:
        resolved = original(source)
        _update_model_views(
            resolved, {"embedding.aux_feature_channels": {"extra": 1}}
        )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_aux)
    with pytest.raises(runner.StageQSelectionError, match="aux_feature_channels.*dd964863"):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_auxiliary_embedding_channels_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walkers: int
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_aux(source: object) -> object:
        resolved = original(source)
        _update_model_views(
            resolved, {"embedding.aux_feature_channels": {"extra": 1}}
        )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_aux)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(runner.StageQSelectionError, match="aux_feature_channels.*dd964863"):
        runner.execute(_args(tmp_path, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


def _wrapped_spinless_embedding(resolved: object) -> dict[str, object]:
    embedding = runner.OmegaConf.to_container(
        runner.OmegaConf.select(resolved, "model.embedding"), resolve=True
    )
    embedding["include_spins"] = False
    return {
        "_target_": "hydra.utils.instantiate",
        "_recursive_": False,
        "config": embedding,
    }


def test_truthfulness_guard_refuses_wrapped_embedding_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_wrapped(source: object) -> object:
        resolved = original(source)
        _update_model_views(
            resolved, {"embedding": _wrapped_spinless_embedding(resolved)}
        )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_wrapped)
    with pytest.raises(runner.StageQSelectionError, match="embedding._target_.*instantiate.*dd964863"):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_wrapped_embedding_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walkers: int
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_wrapped(source: object) -> object:
        resolved = original(source)
        _update_model_views(
            resolved, {"embedding": _wrapped_spinless_embedding(resolved)}
        )
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_wrapped)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(runner.StageQSelectionError, match="embedding._target_.*instantiate.*dd964863"):
        runner.execute(_args(tmp_path, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


def test_truthfulness_guard_refuses_non_input_control_change_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _mechanics_cell(_cells(tmp_path))
    original = runner.train_config.resolve_train_config

    def resolve_hidden_width(source: object) -> object:
        resolved = original(source)
        _update_model_views(resolved, {"embedding.hidden_channels": 64})
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_hidden_width)
    with pytest.raises(runner.StageQSelectionError, match="fingerprint.*expected.*actual.*dd964863"):
        runner.select_cell((target,), seed_label=810001)


@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_non_input_control_change_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, walkers: int
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_hidden_width(source: object) -> object:
        resolved = original(source)
        _update_model_views(resolved, {"embedding.hidden_channels": 64})
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_hidden_width)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(runner.StageQSelectionError, match="fingerprint.*expected.*actual.*dd964863"):
        runner.execute(_args(tmp_path, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


@pytest.mark.parametrize(
    ("seed_label", "walkers"),
    [
        (810001, stage_coordinate.COMMITTED_WALKERS),
        (810001, stage_coordinate.Q_MECHANICS_WALKERS),
        (810002, stage_coordinate.COMMITTED_WALKERS),
        (810002, stage_coordinate.Q_MECHANICS_WALKERS),
    ],
)
def test_execute_accepts_committed_control_at_both_seed_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_label: int,
    walkers: int,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)

    def recording_run_from_config(config: object, *, topology: object) -> int:
        observed["calls"] = int(observed.get("calls", 0)) + 1
        output = Path(config.run.dir) / "resolved_config.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("guarded: true\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner.launch, "run_from_config", recording_run_from_config)
    assert runner.execute(
        _args(tmp_path, seed_label=seed_label, walkers=walkers, dry_run=False)
    ) == 0
    assert observed["calls"] == 1


def _with_seed_stream(cell: object, stream: str, value: object) -> object:
    streams = dict(cell.seed_streams)
    streams[stream] = value
    return replace(cell, seed_streams=streams)


def _update_initializer_seeds(config: object, value: object) -> None:
    paths = (
        "embedding.initializer.seed",
        "layers.0.path_aggregation.initializer.seed",
    )
    for prefix in ("model", "runner.model"):
        for path in paths:
            runner.OmegaConf.update(
                config,
                f"{prefix}.{path}",
                value,
                merge=False,
            )


def _update_sampler_seeds(config: object, value: object) -> None:
    for path in ("sampler.seed", "runner.sampler.seed"):
        runner.OmegaConf.update(config, path, value, merge=False)


def _remove_sampler_seed_from_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    original = runner.train_config._compose_base_config

    def compose_without_sampler_seed() -> object:
        config = original()
        runner.OmegaConf.set_struct(config.sampler, False)
        del config.sampler["seed"]
        return config

    monkeypatch.setattr(
        runner.train_config,
        "_compose_base_config",
        compose_without_sampler_seed,
    )


@pytest.mark.parametrize("seed_label", [810001, 810002])
def test_truthfulness_guard_refuses_wrong_carrier_stream_at_selection(
    tmp_path: Path, seed_label: int
) -> None:
    target = next(
        cell for cell in _cells(tmp_path)
        if cell.manifest["stage"] == "Q"
        and cell.manifest["seed_identity"]["label"] == seed_label
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
    mutated = _with_seed_stream(target, "model_initialization", 7)
    with pytest.raises(
        runner.StageQSelectionError,
        match="cell.seed_streams.*model_initialization.*7.*dd964863",
    ):
        runner.select_cell((mutated,), seed_label=seed_label)


@pytest.mark.parametrize("seed_label", [810001, 810002])
@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_wrong_carrier_stream_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_label: int,
    walkers: int,
) -> None:
    target = next(
        cell for cell in _cells(tmp_path)
        if cell.manifest["stage"] == "Q"
        and cell.manifest["seed_identity"]["label"] == seed_label
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
    mutated = _with_seed_stream(target, "model_initialization", 7)
    monkeypatch.setattr(
        runner.stage_coordinate,
        "materialize_intended_configurations",
        lambda *args, **kwargs: (mutated,),
    )
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(
        runner.StageQSelectionError,
        match="cell.seed_streams.*model_initialization.*7.*dd964863",
    ):
        runner.execute(_args(tmp_path, seed_label=seed_label, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("another integer", 7),
        ("null", None),
        ("factory mapping", {"_target_": "torch.seed"}),
    ],
)
@pytest.mark.parametrize("seed_label", [810001, 810002])
def test_truthfulness_guard_refuses_consumed_initializer_seed_at_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    value: object,
    seed_label: int,
) -> None:
    target = _mechanics_cell(_cells(tmp_path), seed_label=seed_label)
    original = runner.train_config.resolve_train_config

    def resolve_with_seed_mutant(source: object) -> object:
        resolved = original(source)
        _update_initializer_seeds(resolved, value)
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_with_seed_mutant)
    with pytest.raises(
        runner.StageQSelectionError,
        match="runner.model.*initializer.seed.*dd964863",
    ) as refusal:
        runner.select_cell((target,), seed_label=seed_label)
    assert label in {"another integer", "null", "factory mapping"}
    assert str(refusal.value)


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("another integer", 7),
        ("null", None),
        ("factory mapping", {"_target_": "torch.seed"}),
    ],
)
@pytest.mark.parametrize("seed_label", [810001, 810002])
@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_consumed_initializer_seed_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    value: object,
    seed_label: int,
    walkers: int,
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_with_seed_mutant(source: object) -> object:
        resolved = original(source)
        _update_initializer_seeds(resolved, value)
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_with_seed_mutant)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(
        runner.StageQSelectionError,
        match="runner.model.*initializer.seed.*dd964863",
    ):
        runner.execute(_args(tmp_path, seed_label=seed_label, walkers=walkers, dry_run=False))
    assert label in {"another integer", "null", "factory mapping"}
    assert not list(tmp_path.rglob("selection_receipt.json"))


@pytest.mark.parametrize("seed_label", [810001, 810002])
def test_truthfulness_guard_refuses_wrong_sampler_seed_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed_label: int
) -> None:
    target = _mechanics_cell(_cells(tmp_path), seed_label=seed_label)
    original = runner.train_config.resolve_train_config

    def resolve_with_sampler_mutant(source: object) -> object:
        resolved = original(source)
        _update_sampler_seeds(resolved, 7)
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_with_sampler_mutant)
    with pytest.raises(
        runner.StageQSelectionError,
        match="runner.sampler.seed.*7.*dd964863",
    ):
        runner.select_cell((target,), seed_label=seed_label)


@pytest.mark.parametrize("seed_label", [810001, 810002])
@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_wrong_sampler_seed_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_label: int,
    walkers: int,
) -> None:
    original = runner.train_config.resolve_train_config

    def resolve_with_sampler_mutant(source: object) -> object:
        resolved = original(source)
        _update_sampler_seeds(resolved, 7)
        return resolved

    monkeypatch.setattr(runner.train_config, "resolve_train_config", resolve_with_sampler_mutant)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(
        runner.StageQSelectionError,
        match="runner.sampler.seed.*7.*dd964863",
    ):
        runner.execute(_args(tmp_path, seed_label=seed_label, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))


@pytest.mark.parametrize("seed_label", [810001, 810002])
def test_truthfulness_guard_refuses_missing_sampler_seed_at_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed_label: int
) -> None:
    _remove_sampler_seed_from_composition(monkeypatch)
    target = _mechanics_cell(_cells(tmp_path), seed_label=seed_label)
    with pytest.raises(
        runner.StageQSelectionError,
        match="runner.sampler.seed.*missing.*dd964863",
    ):
        runner.select_cell((target,), seed_label=seed_label)


@pytest.mark.parametrize("seed_label", [810001, 810002])
@pytest.mark.parametrize("walkers", [stage_coordinate.COMMITTED_WALKERS, stage_coordinate.Q_MECHANICS_WALKERS])
def test_execute_refuses_missing_sampler_seed_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_label: int,
    walkers: int,
) -> None:
    _remove_sampler_seed_from_composition(monkeypatch)
    monkeypatch.setattr(runner, "_execution_facts", _r2_facts)
    with pytest.raises(
        runner.StageQSelectionError,
        match="runner.sampler.seed.*missing.*dd964863",
    ):
        runner.execute(_args(tmp_path, seed_label=seed_label, walkers=walkers, dry_run=False))
    assert not list(tmp_path.rglob("selection_receipt.json"))
