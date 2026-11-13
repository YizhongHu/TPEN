"""Acceptance tests for TO baff945c: route the approved per-lineage science
training config through ``attached.build_h_v2_train_plan``, ``run_job.py``
and ``launch.py``, and lift ``h-v2-science-config-unavailable`` for L1-L3.

Kept in its own file (not ``test_attached.py``) so it does not collide with
PR 624's edits to that file (note ``acceptance-contract`` on this item).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

HERE = Path(__file__).parent


def _h_v2_module(name: str):
    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _h_v2_module("protocol")
attached = _h_v2_module("attached")
launch_config = _h_v2_module("launch_config")
launch = _h_v2_module("launch")
run_job = _h_v2_module("run_job")


# Field-by-field expected science values, transcribed from design doc A1
# (lineage L1 shown there; L2/L3 differ only in seed/run_id per A1's header).
_EXPECTED_SCIENCE_NON_SEED = {
    "runtime": {"device": "cuda", "dtype": "float64"},
    "sampler": {"n_walkers": 4096, "burn_in": 100, "n_steps": 10, "proposal_scale": 0.5},
    "trainer": {"max_steps": 100000, "return_terms": True},
    "model_embedding": {"out_channels": 32, "hidden_channels": 128},
    "model_mixing_channels": 32,
    "model_path_aggregation_channels": 32,
    "model_readout_channels": 32,
}

_PIPELINE_SEEDS = (910001, 910002)
_PIPELINE_WALKERS = 256
_PIPELINE_STEPS = 1000
_PIPELINE_CHANNELS = 8
_PIPELINE_HIDDEN = 32

_SEED_ROSTER = json.loads((HERE / "contract_inputs.json").read_text(encoding="utf-8"))["measurement"]["seed_roster"]
_TRAIN_SEEDS = _SEED_ROSTER["training_initialization_seeds"]
_SAMPLER_SEEDS = _SEED_ROSTER["training_sampler_seeds"]


@pytest.mark.parametrize("lineage,index", [("L1", 0), ("L2", 1), ("L3", 2)])
def test_science_training_config_matches_a1_field_by_field(lineage: str, index: int) -> None:
    config, identity = protocol.science_training_config(lineage, index)

    assert config["runtime"]["seed"] == _TRAIN_SEEDS[index]
    assert config["sampler"]["seed"] == _SAMPLER_SEEDS[index]
    assert config["runtime"]["device"] == _EXPECTED_SCIENCE_NON_SEED["runtime"]["device"]
    assert config["runtime"]["dtype"] == _EXPECTED_SCIENCE_NON_SEED["runtime"]["dtype"]
    assert config["sampler"]["n_walkers"] == _EXPECTED_SCIENCE_NON_SEED["sampler"]["n_walkers"]
    assert config["sampler"]["burn_in"] == _EXPECTED_SCIENCE_NON_SEED["sampler"]["burn_in"]
    assert config["sampler"]["n_steps"] == _EXPECTED_SCIENCE_NON_SEED["sampler"]["n_steps"]
    assert config["sampler"]["proposal_scale"] == _EXPECTED_SCIENCE_NON_SEED["sampler"]["proposal_scale"]
    assert config["trainer"]["max_steps"] == _EXPECTED_SCIENCE_NON_SEED["trainer"]["max_steps"]
    assert config["trainer"]["return_terms"] == _EXPECTED_SCIENCE_NON_SEED["trainer"]["return_terms"]
    assert config["model"]["embedding"]["out_channels"] == _EXPECTED_SCIENCE_NON_SEED["model_embedding"]["out_channels"]
    assert (
        config["model"]["embedding"]["hidden_channels"]
        == _EXPECTED_SCIENCE_NON_SEED["model_embedding"]["hidden_channels"]
    )
    assert config["model"]["layers"][0]["mixing"]["channels"] == _EXPECTED_SCIENCE_NON_SEED["model_mixing_channels"]
    assert (
        config["model"]["layers"][0]["path_aggregation"]["channels"]
        == _EXPECTED_SCIENCE_NON_SEED["model_path_aggregation_channels"]
    )
    assert config["model"]["readout"]["channels"] == _EXPECTED_SCIENCE_NON_SEED["model_readout_channels"]

    # Carried requirement 1: the hydrogen contract must accept it.
    protocol.assert_hydrogen_contract(config, evaluation=False)

    assert identity["lineage"] == lineage
    assert identity["lineage_index"] == index
    assert set(identity) == {"lineage", "lineage_index", "overlay_sha256", "base_train_sha256", "contract_inputs_sha256"}


def test_science_config_carries_the_complete_a1_logging_cadence() -> None:
    science, _ = protocol.science_training_config("L1", 0)
    assert science["trainer"]["log_every_n_steps"] == protocol.PILOT_A1_CADENCE
    for callback in science["callbacks"]:
        if callback["_target_"] in {"tpen.callback.DataIntegrity", "tpen.callback.GradientStats"}:
            assert callback["every_n_steps"] == protocol.PILOT_A1_CADENCE


def test_science_config_carries_no_pipeline_default_values() -> None:
    config, _ = protocol.science_training_config("L1", 0)
    assert config["sampler"]["n_walkers"] != _PIPELINE_WALKERS
    assert config["trainer"]["max_steps"] != _PIPELINE_STEPS
    assert config["model"]["embedding"]["out_channels"] != _PIPELINE_CHANNELS
    assert config["model"]["embedding"]["hidden_channels"] != _PIPELINE_HIDDEN
    assert config["model"]["layers"][0]["mixing"]["channels"] != _PIPELINE_CHANNELS
    assert config["model"]["layers"][0]["path_aggregation"]["channels"] != _PIPELINE_CHANNELS
    assert config["model"]["readout"]["channels"] != _PIPELINE_CHANNELS
    assert config["runtime"]["seed"] not in _PIPELINE_SEEDS
    assert config["sampler"]["seed"] not in _PIPELINE_SEEDS


def test_science_evaluation_roster_is_single_contract_parameter() -> None:
    expected = tuple(range(0, 20001, 1000)) + tuple(range(25000, 100001, 5000))
    updates = protocol.science_evaluated_checkpoint_updates()
    assert updates == expected
    assert len(updates) == 37
    assert protocol.science_eval_seed(2, len(updates) - 1, 3) == 1_000_000 + 20_000 + 360 + 3


def test_science_and_pilot_partition_parameters_are_separate() -> None:
    assert launch_config.RESOURCES.partition == "seas_gpu,kozinsky_gpu,gpu"
    assert launch_config.RESOURCES.constraint == "a100"
    assert launch_config.PILOT_RESOURCES.partition == "seas_gpu,kozinsky_gpu"
    assert launch_config.PILOT_PAYLOAD_RESOURCES["capacity-pilot"] is launch_config.PILOT_RESOURCES


def test_science_scale_overlay_key_set_equals_allowlist_exactly() -> None:
    overlay = protocol.load_yaml(protocol.SCIENCE_SCALE_OVERLAY_PATH)
    assert set(overlay) == protocol.SCIENCE_SCALE_OVERLAY_KEYS


def test_science_scale_overlay_key_set_drift_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant control: an overlay missing an allowlisted key must be refused."""

    drifted = tmp_path / "drifted_overlay.yaml"
    bad = dict(yaml.safe_load(protocol.SCIENCE_SCALE_OVERLAY_PATH.read_text(encoding="utf-8")))
    del bad["trainer.max_steps"]
    drifted.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setattr(protocol, "SCIENCE_SCALE_OVERLAY_PATH", drifted)
    with pytest.raises(ValueError, match="allowlist"):
        protocol.science_training_config("L1", 0)


def test_build_h_v2_train_plan_routes_science_config_and_overrides(tmp_path: Path) -> None:
    config, identity = protocol.science_training_config("L1", 0)
    plan, overrides_by_task, resolved = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff",
        results_root=tmp_path / "results",
        lineage="L1",
        config=config,
        science_config_identity=identity,
    )
    task = plan.tasks[0]
    overrides = overrides_by_task[task.logical_task_id]
    assert f"runtime.seed={_TRAIN_SEEDS[0]}" in overrides
    assert f"sampler.n_walkers={_EXPECTED_SCIENCE_NON_SEED['sampler']['n_walkers']}" in overrides
    assert f"trainer.max_steps={_EXPECTED_SCIENCE_NON_SEED['trainer']['max_steps']}" in overrides
    assert not any(f"sampler.n_walkers={_PIPELINE_WALKERS}" == item for item in overrides)
    assert not any(f"trainer.max_steps={_PIPELINE_STEPS}" == item for item in overrides)
    assert resolved["sampler"]["n_walkers"] == _EXPECTED_SCIENCE_NON_SEED["sampler"]["n_walkers"]

    assert task.metadata["payload"] == "science"
    carried_identity = task.metadata["science_config_identity"]
    assert attached.SCIENCE_CONFIG_IDENTITY_KEYS <= set(carried_identity)
    assert carried_identity["lineage"] == "L1"
    assert carried_identity["lineage_index"] == 0


def test_build_h_v2_train_plan_pipeline_check_payload_is_unaffected(tmp_path: Path) -> None:
    """No regression: a bare call (no config) still builds the frozen pipeline-check config."""

    plan, overrides_by_task, resolved = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results", lineage="pipeline-check-L1"
    )
    task = plan.tasks[0]
    assert task.metadata["payload"] == "pipeline-check"
    assert "science_config_identity" not in task.metadata
    assert resolved["sampler"]["n_walkers"] == _PIPELINE_WALKERS
    assert resolved["trainer"]["max_steps"] == _PIPELINE_STEPS


def test_build_h_v2_accepted_receipt_refuses_science_payload_missing_identity() -> None:
    with pytest.raises(attached.HV2ReceiptRefusal) as excinfo:
        attached.build_h_v2_accepted_receipt(
            "/nonexistent/run-manifest.json",
            expected_task_ids=("h-v2-train-L1",),
            expected_source_commit="deadbeef",
            stage=attached.TRAIN_STAGE,
            payload="science",
            science_config_identity={"lineage": "L1"},
        )
    assert excinfo.value.code == "science-config-identity-missing"


def test_checkpoint_dir_for_eval_refuses_incomplete_science_receipt(tmp_path: Path) -> None:
    run_root = tmp_path
    receipts_dir = run_root / "attached" / "receipts"
    receipts_dir.mkdir(parents=True)
    task_id = run_job._train_task_id("L1", pipeline_check=False)
    receipt_path = receipts_dir / f"{task_id}.h-v2.json"
    receipt_path.write_text(
        json.dumps(
            {
                "payload": "science",
                "science_config_identity": {"lineage": "L1"},  # incomplete
                "segments": [{"checkpoint_dir": str(tmp_path / "ckpt")}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(run_job.JobError, match="missing science config identity"):
        run_job.checkpoint_dir_for_eval(run_root, "L1", pipeline_check=False)


def test_checkpoint_dir_for_eval_accepts_complete_science_receipt(tmp_path: Path) -> None:
    run_root = tmp_path
    receipts_dir = run_root / "attached" / "receipts"
    receipts_dir.mkdir(parents=True)
    task_id = run_job._train_task_id("L1", pipeline_check=False)
    receipt_path = receipts_dir / f"{task_id}.h-v2.json"
    checkpoint_dir = tmp_path / "ckpt"
    identity = {
        "lineage": "L1",
        "lineage_index": 0,
        "overlay_sha256": "a" * 64,
        "base_train_sha256": "b" * 64,
        "contract_inputs_sha256": "c" * 64,
        "materialized_config_sha256": "d" * 64,
    }
    receipt_path.write_text(
        json.dumps({"payload": "science", "science_config_identity": identity, "segments": [{"checkpoint_dir": str(checkpoint_dir)}]}),
        encoding="utf-8",
    )
    assert run_job.checkpoint_dir_for_eval(run_root, "L1", pipeline_check=False) == checkpoint_dir


def test_run_job_train_refuses_unapproved_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_job, "check_identity", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUN", str(tmp_path))
    monkeypatch.setenv("SHA", "deadbeef")
    monkeypatch.setenv("SCIENCE_PYTHON", sys.executable)
    monkeypatch.setenv("EXECUTOR_PY", sys.executable)
    monkeypatch.setattr(
        attached,
        "probe_admitted_a1_modules",
        lambda config, candidates, **kwargs: ([], []),
    )

    class _MustNotBeCalled(AssertionError):
        pass

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise _MustNotBeCalled("build_h_v2_train_plan must not be reached for an unapproved lineage")

    monkeypatch.setattr(attached, "build_h_v2_train_plan", _forbidden)

    for lineage in ("L0", "L4", "pilot", ""):
        with pytest.raises(run_job.JobError, match=run_job.SCIENCE_CONFIG_UNAVAILABLE_CODE):
            run_job.run("train", lineage, pipeline_check=False)


def test_run_job_train_builds_science_config_for_approved_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_job, "check_identity", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUN", str(tmp_path))
    monkeypatch.setenv("SHA", "deadbeef")
    monkeypatch.setenv("SCIENCE_PYTHON", sys.executable)
    monkeypatch.setenv("EXECUTOR_PY", sys.executable)
    monkeypatch.setattr(
        attached,
        "probe_admitted_a1_modules",
        lambda config, candidates, **kwargs: ([], []),
    )

    captured: dict[str, Any] = {}

    class _StopEarly(RuntimeError):
        pass

    def _capture(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _StopEarly("stop before prepare/activate -- this test only checks routing")

    monkeypatch.setattr(attached, "build_h_v2_train_plan", _capture)

    with pytest.raises(_StopEarly):
        run_job.run("train", "L2", pipeline_check=False)

    assert captured["lineage"] == "L2"
    assert captured["config"]["runtime"]["seed"] == _TRAIN_SEEDS[1]
    assert captured["science_config_identity"]["lineage"] == "L2"
    assert captured["science_config_identity"]["lineage_index"] == 1


def test_launch_science_dry_run_prints_six_lines_with_science_job_names(tmp_path: Path) -> None:
    results = launch.submit_all(
        payload="science",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:00:00",
        e_wall="08:00:00",
        dry_run=True,
    )
    assert set(results) == set(launch_config.LINEAGES)
    for lineage, (train_id, eval_id) in results.items():
        assert train_id is None and eval_id is None


def test_launch_science_dry_run_line_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    launch.submit_all(
        payload="science",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:00:00",
        e_wall="08:00:00",
        dry_run=True,
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 6
    for lineage in launch_config.LINEAGES:
        assert any(f"--job-name=h-v2-science-{lineage}-train" in line for line in lines)
        assert any(f"--job-name=h-v2-science-{lineage}-eval" in line for line in lines)
        assert any("--dependency=afterok" in line and f"h-v2-science-{lineage}-eval" in line for line in lines)
    assert not any("pipeline-check" in line for line in lines)


def test_launch_science_refuses_unapproved_lineage_before_any_sbatch(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _recording_runner(command: list) -> None:
        calls.append(list(command))
        raise AssertionError("sbatch must never be called for a refused lineage")

    for lineage in ("L0", "L4", "pilot", ""):
        with pytest.raises(launch.LaunchError, match=run_job.SCIENCE_CONFIG_UNAVAILABLE_CODE):
            launch.submit_lineage(
                lineage,
                payload="science",
                science_python=sys.executable,
                executor_python=sys.executable,
                run_root=tmp_path,
                checkout_root=tmp_path / "checkout",
                t_wall="08:00:00",
                e_wall="08:00:00",
                runner=_recording_runner,
            )
    assert calls == []


def test_launch_pipeline_check_payload_job_names_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    launch.submit_all(
        payload="pipeline-check",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:00:00",
        e_wall="08:00:00",
        dry_run=True,
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 6
    for lineage in launch_config.LINEAGES:
        assert any(f"--job-name=h-v2-pipeline-check-{lineage}-train" in line for line in lines)
