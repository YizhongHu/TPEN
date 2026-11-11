"""T2 round-1 proposals for PR 629 at a246c509; run on Cannon only.

The real-receipt fixture requires H_V2_SCIENCE_PYTHON and
H_V2_EXECUTOR_PYTHON from the approved cluster harness. It drives the real
attached activation and reader with a two-update CPU test configuration;
no science result or GPU qualification is implied. No core/executor import,
mock manifest, fake inspector, or fabricated accepted receipt is used.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

HERE = Path(__file__).parent
REPO_ROOT = HERE.resolve().parents[2]


def _module(name):
    key = f"h_v2_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


protocol = _module("protocol")
attached = _module("attached")
launch = _module("launch")
job = _module("run_job")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _science_plan(tmp_path, lineage, index):
    config, identity = protocol.science_training_config(lineage, index)
    return attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results",
        lineage=lineage, config=config, science_config_identity=identity,
    )


@pytest.mark.parametrize("lineage,index", [("L1", 0), ("L2", 1), ("L3", 2)])
def test_resolved_handoff_matches_every_a1_nonlogging_leaf(tmp_path, lineage, index):
    """Kill science-only optimizer/architecture drift and late override leaks."""
    plan, overrides, config = _science_plan(tmp_path, lineage, index)
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")
    resolved = attached._resolve_bound_config(
        attached._merge_config_with_overrides(prepared.effective_config, prepared.overrides)
    )
    protocol.assert_hydrogen_contract(resolved)

    expected = yaml.safe_load((HERE / "testdata" / "science_a1_nonlogging.yaml").read_text())
    roster = json.loads((HERE / "contract_inputs.json").read_text())["measurement"]["seed_roster"]
    expected["runtime"]["seed"] = roster["training_initialization_seeds"][index]
    expected["sampler"]["seed"] = roster["training_sampler_seeds"][index]
    # A1 permits these labels, the run-root mechanism and per-lineage run ID.
    expected["experiment"]["run_name"] = "tpen_h_v2_train"
    expected["run"].update(
        root=str(tmp_path / "results" / "attached"),
        run_id=f"h-v2-train-{lineage}", layout="flat",
    )
    expected = OmegaConf.to_container(OmegaConf.create(expected), resolve=True)
    actual = copy.deepcopy(resolved)
    actual.pop("callbacks")
    actual.pop("loggers")
    actual["trainer"].pop("log_every_n_steps")
    # runner.trainer is a resolved copy of the same interpolated trainer.
    actual["runner"]["trainer"].pop("log_every_n_steps", None)
    assert actual == expected

    base = protocol.training_config()
    for callback in base["callbacks"]:
        if callback["_target_"] in {"tpen.callback.DataIntegrity", "tpen.callback.GradientStats"}:
            callback["every_n_steps"] = protocol.PILOT_A1_CADENCE
    assert config["callbacks"] == base["callbacks"]
    assert config["loggers"] == base["loggers"]
    assert config["trainer"]["log_every_n_steps"] == base["trainer"]["log_every_n_steps"]
    checkpoints = [c for c in config["callbacks"] if c["_target_"] == "tpen.callback.Checkpoint"]
    assert checkpoints == [{
        "_target_": "tpen.callback.Checkpoint", "terminal": True,
        "schedule": {"_target_": "tpen.checkpoint.EveryNUpdates", "every_n": 1000},
        "payload": {"_target_": "tpen.checkpoint.TrainResume"},
        "output_dir": "${run.dir}/checkpoints",
    }]

    # Verify bytes independently, rather than only requiring six key names.
    assert prepared.payload == "science"
    assert prepared.science_config_identity == {
        "lineage": lineage, "lineage_index": index,
        "overlay_sha256": _sha(HERE / "configs" / "science_scale_overlay.yaml"),
        "base_train_sha256": _sha(HERE / "configs" / "train.yaml"),
        "contract_inputs_sha256": _sha(HERE / "contract_inputs.json"),
        "materialized_config_sha256": _sha(prepared.effective_config),
    }


@pytest.mark.parametrize("lineage,index", [("L1", 0), ("L2", 1), ("L3", 2)])
def test_science_seeds_are_read_from_ci_not_matching_literals(tmp_path, monkeypatch, lineage, index):
    """Kill duplicated seed literals that happen to match today's CI roster."""
    document = json.loads((HERE / "contract_inputs.json").read_text())
    roster = document["measurement"]["seed_roster"]
    roster["training_initialization_seeds"] = [173001, 173002, 173003]
    roster["training_sampler_seeds"] = [183001, 183002, 183003]
    alternate = tmp_path / "contract_inputs.json"
    alternate.write_text(json.dumps(document))
    monkeypatch.setattr(protocol, "CONTRACT_INPUTS_PATH", alternate)
    config, identity = protocol.science_training_config(lineage, index)
    assert config["runtime"]["seed"] == roster["training_initialization_seeds"][index]
    assert config["sampler"]["seed"] == roster["training_sampler_seeds"][index]
    assert identity["contract_inputs_sha256"] == _sha(alternate)


@pytest.mark.parametrize("lineage,index", [("L1", 0), ("L2", 1), ("L3", 2)])
def test_science_prepare_rejects_resolved_physics_drift(tmp_path, lineage, index):
    """Kill bypassing the hydrogen check on the final bound science config."""
    plan, overrides, config = _science_plan(tmp_path, lineage, index)
    task_id = plan.tasks[0].logical_task_id
    overrides[task_id] += ("system.nuclei.charges=[2.0]",)
    with pytest.raises(ValueError, match="charge-one"):
        attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")


def test_in_job_missing_payload_refuses_before_run(monkeypatch):
    """F2: kill implicit science selection when the payload token is absent."""
    calls = []
    monkeypatch.setattr(job, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert job.main(["train", "L1"]) == 2
    assert calls == []


@pytest.mark.parametrize("stage", ["train", "eval"])
def test_science_wrapper_carries_an_explicit_parseable_payload(tmp_path, monkeypatch, stage):
    """F2: kill a wrapper which drops payload, or a CLI which ignores it."""
    command = launch.build_sbatch_command(
        stage, "L1", payload="science", science_python=sys.executable,
        executor_python=sys.executable, run_root=tmp_path,
        checkout_root=REPO_ROOT, time_value="08:00:00", dependency=None, commit="a" * 40,
    )
    wrap = next(token.split("=", 1)[1] for token in command if token.startswith("--wrap="))
    tokens = shlex.split(wrap)
    script = str(REPO_ROOT / "experiments" / "atomistic" / "h-v2" / "run_job.py")
    argv = tokens[tokens.index(script) + 1:]
    assert argv[:2] == [stage, "L1"]
    assert argv[2:] == ["--payload", "science"]
    calls = []
    monkeypatch.setattr(job, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert job.main(argv) == 0
    assert len(calls) == 1


@pytest.mark.parametrize("dry_run", [False, True])
def test_mixed_science_roster_refuses_before_any_submission(tmp_path, capsys, dry_run):
    """F3: kill validating each lineage only after earlier jobs were submitted."""
    calls = []

    def runner(command):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, f"{len(calls)};cluster\n", "")

    with pytest.raises(launch.LaunchError, match=job.SCIENCE_CONFIG_UNAVAILABLE_CODE):
        launch.submit_all(
            payload="science", science_python=sys.executable, executor_python=sys.executable,
            run_root=tmp_path, checkout_root=REPO_ROOT, t_wall="08:00:00", e_wall="08:00:00",
            lineages=("L1", "L9"), runner=runner, dry_run=dry_run,
        )
    assert calls == []
    assert capsys.readouterr().out == ""


@pytest.fixture(scope="module")
def real_science_receipt(tmp_path_factory):
    """Produce wire bytes through real dispatch, inspection and receipt writing."""
    science_python = os.environ.get("H_V2_SCIENCE_PYTHON")
    executor_python = os.environ.get("H_V2_EXECUTOR_PYTHON")
    assert science_python and executor_python, (
        "Cannon harness must set H_V2_SCIENCE_PYTHON and H_V2_EXECUTOR_PYTHON; "
        "a missing runtime is not a passing producer-consumer test"
    )
    root = tmp_path_factory.mktemp("science-receipt-r1")
    config, identity = protocol.science_training_config("L1", 0)
    # Acceptance-contract-detail permits a CPU science-shaped test only.
    config["runtime"]["device"] = "cpu"
    config["sampler"].update(n_walkers=4, burn_in=1, n_steps=1)
    config["trainer"]["max_steps"] = 2
    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=root / "handoff", results_root=root / "results",
        lineage="L1", config=config, science_config_identity=identity,
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=root / "handoff")
    allocation, worker = job.write_local_allocation_and_worker(
        root / "allocation", science_python=science_python,
        placeholder_config=prepared.effective_config, backend_job_id="t2-review-cpu-test",
    )
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()
    receipt_path = root / "attached" / "receipts" / f"{prepared.task_id}.h-v2.json"
    attached.activate_h_v2_run(
        prepared, science_python=science_python, executor_python=executor_python,
        allocation=allocation, worker_config=worker, store=root / "store",
        activation_receipt_path=root / "activation.json", h_v2_receipt_path=receipt_path,
        expected_source_commit=source_sha, dispatcher="local", timeout=180.0, cwd=REPO_ROOT,
    )
    return json.loads(receipt_path.read_text()), prepared


def _publish_real_receipt(root, receipt):
    path = root / "attached" / "receipts" / "h-v2-train-L1.h-v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))


def test_real_producer_preserves_payload_and_all_config_identity(real_science_receipt, tmp_path):
    """F1: kill payload shadowing and identity loss at prepare/activate/write."""
    receipt, prepared = real_science_receipt
    assert receipt["payload"] == "science"
    assert receipt["science_config_identity"] == prepared.science_config_identity
    _publish_real_receipt(tmp_path, receipt)
    checkpoint = job.checkpoint_dir_for_eval(tmp_path, "L1", pipeline_check=False)
    assert checkpoint == Path(receipt["segments"][0]["checkpoint_dir"])
    assert (checkpoint / "COMPLETE").is_file()


@pytest.mark.parametrize("missing", [
    "identity", "lineage", "lineage_index", "overlay_sha256", "base_train_sha256",
    "contract_inputs_sha256", "materialized_config_sha256", "payload",
])
def test_consumer_refuses_incomplete_real_producer_receipt(real_science_receipt, tmp_path, missing):
    """F1: kill identity guards that rely on an unvalidated payload discriminator."""
    receipt, _ = real_science_receipt
    _publish_real_receipt(tmp_path, receipt)
    # The unmodified producer output must reach the same consumer first.
    assert job.checkpoint_dir_for_eval(tmp_path, "L1", pipeline_check=False).is_dir()
    broken = copy.deepcopy(receipt)
    if missing == "identity":
        broken.pop("science_config_identity")
    elif missing == "payload":
        broken.pop("payload")
    else:
        broken["science_config_identity"].pop(missing)
    _publish_real_receipt(tmp_path, broken)
    with pytest.raises(job.JobError):
        job.checkpoint_dir_for_eval(tmp_path, "L1", pipeline_check=False)
