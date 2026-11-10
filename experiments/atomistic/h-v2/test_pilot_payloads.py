"""Acceptance tests for TO 63690220: the pilot and part (g) TRAIN payloads.

Acceptance-contract (note ``acceptance-contract`` on 63690220), amended by
``acceptance-contract-amendment-20261110-r2`` and
``-r3``, and the run-root amendment. Scope: study-only, TRAIN stage only;
EVAL is item E.
"""

from __future__ import annotations

import hashlib
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

_SEED_ROSTER = json.loads((HERE / "contract_inputs.json").read_text(encoding="utf-8"))["measurement"]["seed_roster"]
_NON_SCIENCE = _SEED_ROSTER["non_science_seed_namespaces"]
_SCIENCE_TRAIN_SEEDS = set(_SEED_ROSTER["training_initialization_seeds"])
_SCIENCE_SAMPLER_SEEDS = set(_SEED_ROSTER["training_sampler_seeds"])


# ---------------------------------------------------------------------------
# protocol.pilot_training_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,max_steps,namespace",
    [("gpu-route-g", 1000, "pipeline_gpu_smoke_g"), ("capacity-pilot", 20000, "capacity_pilot")],
)
def test_pilot_training_config_matches_a1_field_by_field(payload: str, max_steps: int, namespace: str) -> None:
    config, identity = protocol.pilot_training_config(payload)

    # Science scale overlay, UNCHANGED and reused as-is.
    assert config["runtime"]["device"] == "cuda"
    assert config["sampler"]["n_walkers"] == 4096
    assert config["model"]["embedding"]["out_channels"] == 32
    assert config["model"]["embedding"]["hidden_channels"] == 128
    assert config["model"]["layers"][0]["mixing"]["channels"] == 32
    assert config["model"]["layers"][0]["path_aggregation"]["channels"] == 32
    assert config["model"]["readout"]["channels"] == 32

    # Payload overlay: only trainer.max_steps and A1 cadence differ by payload.
    assert config["trainer"]["max_steps"] == max_steps
    assert config["trainer"]["log_every_n_steps"] == protocol.PILOT_A1_CADENCE
    data_integrity = next(cb for cb in config["callbacks"] if cb["_target_"] == "tpen.callback.DataIntegrity")
    gradient_stats = next(cb for cb in config["callbacks"] if cb["_target_"] == "tpen.callback.GradientStats")
    assert data_integrity["every_n_steps"] == protocol.PILOT_A1_CADENCE
    assert gradient_stats["every_n_steps"] == protocol.PILOT_A1_CADENCE

    # Every other leaf equals the pipeline-check/scale-overlay base exactly.
    pipeline = protocol.training_config()
    assert config["runtime"]["dtype"] == pipeline["runtime"]["dtype"]
    assert config["sampler"]["burn_in"] == pipeline["sampler"]["burn_in"]
    assert config["trainer"]["return_terms"] == pipeline["trainer"]["return_terms"]

    # Seeds equal CI, from the declared namespace, never a science seed.
    assert config["runtime"]["seed"] == _NON_SCIENCE[namespace]["training"][0]
    assert config["sampler"]["seed"] == _NON_SCIENCE[namespace]["sampler"][0]
    assert config["runtime"]["seed"] not in _SCIENCE_TRAIN_SEEDS
    assert config["sampler"]["seed"] not in _SCIENCE_SAMPLER_SEEDS

    # RunTiming + TrainStepTiming appended by the protocol-owned helper.
    assert any(cb["_target_"] == "tpen.callback.RunTiming" for cb in config["callbacks"])
    step_timing = next(cb for cb in config["callbacks"] if cb["_target_"] == "tpen.callback.TrainStepTiming")
    assert step_timing["every_n_steps"] == protocol.PILOT_A1_CADENCE
    assert step_timing["accelerator_synchronize"] is False

    protocol.assert_hydrogen_contract(config, evaluation=False)

    assert identity["payload"] == payload
    assert identity["lineage"] == protocol.PILOT_PAYLOADS[payload]["lineage"]
    assert identity["seed_namespace"] == namespace


def test_pilot_payload_overlay_key_sets_equal_allowlist_exactly() -> None:
    for spec in protocol.PILOT_PAYLOADS.values():
        overlay = protocol.load_yaml(spec["overlay_path"])
        assert set(overlay) == protocol.PILOT_PAYLOAD_OVERLAY_KEYS


def test_pilot_payload_overlay_key_set_drift_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant control: an overlay missing an allowlisted key must be refused."""

    drifted = tmp_path / "drifted.yaml"
    bad = dict(yaml.safe_load(protocol.PILOT_PAYLOADS["capacity-pilot"]["overlay_path"].read_text(encoding="utf-8")))
    del bad["trainer.max_steps"]
    drifted.write_text(yaml.safe_dump(bad), encoding="utf-8")
    monkeypatch.setitem(protocol.PILOT_PAYLOADS["capacity-pilot"], "overlay_path", drifted)
    with pytest.raises(ValueError, match="allowlist"):
        protocol.pilot_training_config("capacity-pilot")


def test_pilot_training_config_never_carries_a_science_seed() -> None:
    """Mutant: science seed leaks."""

    for payload in protocol.PILOT_PAYLOADS:
        config, _ = protocol.pilot_training_config(payload)
        assert config["runtime"]["seed"] not in _SCIENCE_TRAIN_SEEDS
        assert config["sampler"]["seed"] not in _SCIENCE_SAMPLER_SEEDS


def test_pilot_training_config_refuses_wrong_namespace_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutant control: wrong seed namespace must change the resolved seed."""

    monkeypatch.setitem(protocol.PILOT_PAYLOADS["gpu-route-g"], "seed_namespace", "capacity_pilot")
    config, identity = protocol.pilot_training_config("gpu-route-g")
    assert identity["seed_namespace"] == "capacity_pilot"
    assert config["runtime"]["seed"] == _NON_SCIENCE["capacity_pilot"]["training"][0]
    assert config["runtime"]["seed"] != _NON_SCIENCE["pipeline_gpu_smoke_g"]["training"][0]


# ---------------------------------------------------------------------------
# candidate module admission (amendment r3)
# ---------------------------------------------------------------------------


def test_candidate_modules_to_append_excludes_base_config_leaves() -> None:
    candidates = protocol.candidate_modules_to_append()
    assert "tpen.callback.DataIntegrity" not in candidates
    assert "tpen.callback.GradientStats" not in candidates
    assert "tpen.callback.InitialCheckpoint" in candidates
    assert "tpen.callback.ParameterActivity" in candidates
    assert "tpen.callback.ElectronNucleusDistanceQuantiles" in candidates
    # Amendment r4 rule 2: ModuleNorms is NEVER a probed/appended candidate.
    assert "tpen.callback.ModuleNorms" not in candidates
    assert "tpen.callback.ModuleNorms" not in protocol.A1_TRAINING_MODULE_CANDIDATES


def test_endq_candidate_carries_quantiles_not_output_path() -> None:
    """Mutant: ENDQ quantiles dropped."""

    entry = protocol._build_candidate_callback("tpen.callback.ElectronNucleusDistanceQuantiles")
    assert entry["quantiles"] == list(protocol.ENDQ_A1_QUANTILES)
    assert "output_path" not in entry
    assert entry["every_n_steps"] == protocol.PILOT_A1_CADENCE


def test_unspecified_modules_names_module_norms_with_reason() -> None:
    """Mutant: ModuleNorms appended with placeholder groups."""

    unspecified = protocol.unspecified_modules()
    assert unspecified == [{"target": "tpen.callback.ModuleNorms", "reason": "6c29fe1d"}]


def test_append_and_remove_candidate_module_round_trips() -> None:
    config, _ = protocol.pilot_training_config("gpu-route-g")
    before = len(config["callbacks"])
    protocol.append_candidate_module(config, "tpen.callback.InitialCheckpoint")
    assert len(config["callbacks"]) == before + 1
    protocol.remove_candidate_module(config, "tpen.callback.InitialCheckpoint")
    assert len(config["callbacks"]) == before


class _FakeRefusal(Exception):
    code = "unsupported-callback"
    reason = "not in the declared profile"


def test_probe_admitted_a1_modules_keeps_only_admitted_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parametrized (by stub) on the admitted set -- mirrors a stack head that
    admits only DataIntegrity/GradientStats versus one (dev) that admits more.
    """

    config, _ = protocol.pilot_training_config("gpu-route-g")
    admitted = {"tpen.callback.InitialCheckpoint", "tpen.callback.ParameterActivity"}

    def _fake_check_profile(science_python: Any, probe_path: Any, **kwargs: Any) -> None:
        del science_python, kwargs
        loaded = yaml.safe_load(Path(probe_path).read_text(encoding="utf-8"))
        for callback in loaded["callbacks"]:
            target = callback["_target_"]
            if target in protocol.A1_TRAINING_MODULE_CANDIDATES and target not in admitted | {
                "tpen.callback.DataIntegrity",
                "tpen.callback.GradientStats",
            }:
                raise attached.ActivationRefusal(_FakeRefusal.code, _FakeRefusal.reason)

    monkeypatch.setattr(attached, "check_profile", _fake_check_profile)
    monkeypatch.setattr(attached, "assert_candidate_instantiates", lambda *args, **kwargs: None)
    enabled, skipped = attached.probe_admitted_a1_modules(
        config,
        protocol.candidate_modules_to_append(),
        science_python=sys.executable,
        probe_config_path=tmp_path / "probe.yaml",
    )
    assert set(enabled) == admitted
    assert set(skipped) == set(protocol.candidate_modules_to_append()) - admitted
    targets_in_config = {cb["_target_"] for cb in config["callbacks"]}
    assert admitted <= targets_in_config
    assert not (set(skipped) & targets_in_config)


def test_probe_admitted_a1_modules_admits_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stack-head-shaped stub: the recovery profile admits nothing beyond the base config."""

    config, _ = protocol.pilot_training_config("capacity-pilot")

    def _refuse_everything(science_python: Any, probe_path: Any, **kwargs: Any) -> None:
        del science_python, probe_path, kwargs
        raise attached.ActivationRefusal(_FakeRefusal.code, _FakeRefusal.reason)

    monkeypatch.setattr(attached, "check_profile", _refuse_everything)
    enabled, skipped = attached.probe_admitted_a1_modules(
        config,
        protocol.candidate_modules_to_append(),
        science_python=sys.executable,
        probe_config_path=tmp_path / "probe.yaml",
    )
    assert enabled == []
    assert set(skipped) == set(protocol.candidate_modules_to_append())


def test_probe_raises_on_instantiation_failure_not_swallowed_as_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutant: probe swallows non-refusal errors as skips.

    Amendment r4 rule 3: only an ActivationRefusal from check_profile counts
    as "not admitted"; a misconfigured-but-admitted candidate (the exact
    ENDQ/ModuleNorms bug this amendment fixes) must RAISE, never be recorded
    in ``skipped``.
    """

    config, _ = protocol.pilot_training_config("gpu-route-g")
    monkeypatch.setattr(attached, "check_profile", lambda *args, **kwargs: None)

    def _raise_misconfigured(*args: Any, **kwargs: Any) -> None:
        raise attached.HV2ReceiptRefusal("candidate-module-misconfigured", "boom")

    monkeypatch.setattr(attached, "assert_candidate_instantiates", _raise_misconfigured)
    with pytest.raises(attached.HV2ReceiptRefusal) as excinfo:
        attached.probe_admitted_a1_modules(
            config,
            protocol.candidate_modules_to_append(),
            science_python=sys.executable,
            probe_config_path=tmp_path / "probe.yaml",
        )
    assert excinfo.value.code == "candidate-module-misconfigured"


def test_assert_candidate_instantiates_refuses_a_misconfigured_candidate(tmp_path: Path) -> None:
    """Regression + mutant control for the ENDQ/dropped-quantiles bug amendment r4 fixed.

    Exercises the REAL ``tpen.callback.ElectronNucleusDistanceQuantiles``
    class (requires torch/tpen in *science_python*; run on Cannon/dev per the
    receipt, like every other torch-backed test in this directory).
    """

    config, _ = protocol.pilot_training_config("capacity-pilot")
    broken = {"_target_": "tpen.callback.ElectronNucleusDistanceQuantiles", "every_n_steps": 1}
    config["callbacks"].append(broken)
    probe_path = tmp_path / "probe.yaml"
    probe_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(attached.HV2ReceiptRefusal) as excinfo:
        attached.assert_candidate_instantiates(
            sys.executable, probe_path, "tpen.callback.ElectronNucleusDistanceQuantiles"
        )
    assert excinfo.value.code == "candidate-module-misconfigured"


@pytest.mark.parametrize(
    "target",
    [
        "tpen.callback.InitialCheckpoint",
        "tpen.callback.ParameterActivity",
        "tpen.callback.ElectronNucleusDistanceQuantiles",
    ],
)
def test_assert_candidate_instantiates_accepts_the_real_candidate_entry(target: str, tmp_path: Path) -> None:
    """Amendment r4 rule 4: each appended candidate instantiates against its real class."""

    config, _ = protocol.pilot_training_config("capacity-pilot")
    protocol.append_candidate_module(config, target)
    probe_path = tmp_path / "probe.yaml"
    probe_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    attached.assert_candidate_instantiates(sys.executable, probe_path, target)


# ---------------------------------------------------------------------------
# checkpoint inventory
# ---------------------------------------------------------------------------


def _write_fake_checkpoint(root: Path, update: int) -> Path:
    step_dir = root / f"step_{update:06d}"
    step_dir.mkdir(parents=True)
    (step_dir / "manifest.json").write_text(json.dumps({"step": update}), encoding="utf-8")
    (step_dir / "model.pt").write_bytes(f"model-{update}".encode())
    (step_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return step_dir


def test_build_checkpoint_inventory_lists_every_committed_checkpoint_with_sha256(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    for update in (0, 1000, 2000):
        _write_fake_checkpoint(root, update)
    # An in-progress write must never be inventoried.
    (root / "step_003000.tmp").mkdir()

    inventory = attached.build_checkpoint_inventory(root)
    assert [entry["update"] for entry in inventory] == [0, 1000, 2000]
    for entry in inventory:
        model_path = Path(entry["path"]) / "model.pt"
        expected = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert entry["files"]["model.pt"] == expected


def test_build_checkpoint_inventory_omits_an_incomplete_checkpoint_mutant(tmp_path: Path) -> None:
    """Mutant: inventory omits a checkpoint (here: a checkpoint missing COMPLETE is correctly excluded)."""

    root = tmp_path / "checkpoints"
    _write_fake_checkpoint(root, 0)
    incomplete = root / "step_001000"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text("{}", encoding="utf-8")
    inventory = attached.build_checkpoint_inventory(root)
    assert [entry["update"] for entry in inventory] == [0]


def _sha(data: bytes = b"x") -> str:
    return hashlib.sha256(data).hexdigest()


def _write_core_dir(core_dir: Path, *, git_commit: str = "a" * 40) -> None:
    """Minimal real-shaped core run directory (mirrors ``test_attached._write_core_dir``)."""

    core_dir.mkdir(parents=True, exist_ok=True)
    (core_dir / "resolved_config.yaml").write_text("load:\n  mode: model_only\n", encoding="utf-8")
    (core_dir / "metadata.json").write_text(json.dumps({"git_commit": git_commit}), encoding="utf-8")


def _manifest_fixture(
    manifest_path: Path, *, task_id: str, attempt_id: str, output_dir: Path, checkpoint_dir: str
) -> None:
    """Minimal real-shaped run-manifest-v2 fixture (mirrors ``test_attached._manifest_fixture``)."""

    group_digest = _sha(b"group")
    payload = {
        "schema": "tpen-executor/run-manifest-v2",
        "group_id": "group-1",
        "group_digest": group_digest,
        "attempts": [
            {
                "schema": "tpen-executor/manifest-attempt-v1",
                "task_id": task_id,
                "attempt_id": attempt_id,
                "output_dir": str(output_dir),
                "start_receipt": {
                    "schema": "tpen-executor/start-receipt-v1",
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "allocation_id": "alloc-1",
                    "backend_job_id": "job-1",
                    "group_digest": group_digest,
                    "started_at_utc": "2026-01-01T00:00:00Z",
                    "output_dir": str(output_dir),
                },
                "outcome_receipt": {
                    "schema": "tpen-executor/outcome-receipt-v1",
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "group_digest": group_digest,
                    "finished_at_utc": "2026-01-01T00:01:00Z",
                    "status": "completed",
                    "output_dir": str(output_dir),
                    "result_digest": None,
                },
            }
        ],
        "segments": [
            {
                "schema": "tpen-executor/accepted-segment-v1",
                "task_id": task_id,
                "attempt_id": attempt_id,
                "predecessor_checkpoint": None,
                "checkpoint_dir": checkpoint_dir,
                "next_iteration": 1000,
                "completed_updates": 1000,
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_evidence(checkpoint_dir: str, next_iteration: int, completed_updates: int) -> dict[str, Any]:
    return {
        "verdict": "completed",
        "run_status": "completed",
        "checkpoint": {
            "checkpoint_dir": checkpoint_dir,
            "schema_version": 1,
            "next_iteration": next_iteration,
            "completed_updates": completed_updates,
        },
    }


def test_build_h_v2_accepted_receipt_records_checkpoint_inventory_from_real_producer(
    tmp_path: Path,
) -> None:
    """Consumer test fed the REAL producer output (harness r3 addendum2)."""

    output_dir = tmp_path / "attempt"
    core_dir = output_dir / "core"
    _write_core_dir(core_dir, git_commit="d" * 40)
    checkpoint_root = tmp_path / "run" / "checkpoints"
    _write_fake_checkpoint(checkpoint_root, 0)
    _write_fake_checkpoint(checkpoint_root, 1000)
    checkpoint_dir = checkpoint_root / "step_001000"
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(
        manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir, checkpoint_dir=str(checkpoint_dir)
    )

    payload = attached.build_h_v2_accepted_receipt(
        manifest_path,
        expected_task_ids=("t1",),
        expected_source_commit="d" * 40,
        stage=attached.TRAIN_STAGE,
        inspect_run=lambda path: _completed_evidence(str(checkpoint_dir), 1000, 1000),
        payload="capacity-pilot",
        science_config_identity={"lineage": "P"},
        enabled_modules=["tpen.callback.InitialCheckpoint"],
        skipped_not_admitted=["tpen.callback.ModuleNorms"],
    )
    inventory = payload["checkpoint_inventory"]
    assert [entry["update"] for entry in inventory] == [0, 1000]
    for entry in inventory:
        for name, sha in entry["files"].items():
            actual = hashlib.sha256((Path(entry["path"]) / name).read_bytes()).hexdigest()
            assert sha == actual
    assert payload["enabled_modules"] == ["tpen.callback.InitialCheckpoint"]
    assert payload["skipped_not_admitted"] == ["tpen.callback.ModuleNorms"]


def test_accepted_receipt_defaults_enabled_and_skipped_to_empty_lists(tmp_path: Path) -> None:
    """Mutant control for 'skipped module not recorded in receipt': a bare call
    (no modules actually skipped) must record [] rather than omitting the field,
    so a later caller that DOES skip something cannot be masked by a dict.get()
    default indistinguishable from 'nothing was skipped'."""

    output_dir = tmp_path / "attempt"
    core_dir = output_dir / "core"
    _write_core_dir(core_dir, git_commit="e" * 40)
    manifest_path = tmp_path / "run-manifest.json"
    _manifest_fixture(
        manifest_path, task_id="t1", attempt_id="a1", output_dir=output_dir, checkpoint_dir="/fake/checkpoint"
    )
    payload = attached.build_h_v2_accepted_receipt(
        manifest_path,
        expected_task_ids=("t1",),
        expected_source_commit="e" * 40,
        inspect_run=lambda path: _completed_evidence("/fake/checkpoint", 1000, 1000),
    )
    assert payload["enabled_modules"] == []
    assert payload["skipped_not_admitted"] == []


# ---------------------------------------------------------------------------
# launch_config profiles
# ---------------------------------------------------------------------------


def test_gpu_test_resources_carry_no_a100_constraint_and_the_gpu_test_partition() -> None:
    """Mutant: g profile on seas_gpu."""

    resources = launch_config.PILOT_PAYLOAD_RESOURCES["gpu-route-g"]
    assert resources.partition == "gpu_test"
    assert resources.constraint == ""
    assert "seas_gpu" not in resources.partition
    assert "kozinsky_gpu" not in resources.partition


def test_capacity_pilot_uses_the_approved_a100_resources() -> None:
    assert launch_config.PILOT_PAYLOAD_RESOURCES["capacity-pilot"] is launch_config.RESOURCES


# ---------------------------------------------------------------------------
# launch.py dry-run / cap / run-root prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", ["gpu-route-g", "capacity-pilot"])
def test_dry_run_prints_exactly_one_sbatch_line_with_expected_partition_gres_time(
    payload: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prefix = tmp_path / "runs"
    run_root = prefix / "h-v2" / payload / launch_config.PILOT_PAYLOAD_LINEAGES[payload]
    train_id = launch.submit_pilot_payload(
        payload,
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=run_root,
        checkout_root=run_root / "checkout",
        run_root_prefix=str(prefix),
        dry_run=True,
    )
    assert train_id is None
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    (line,) = lines
    resources = launch_config.PILOT_PAYLOAD_RESOURCES[payload]
    assert f"--partition={resources.partition}" in line
    assert f"--gres=gpu:{resources.gpus}" in line
    cap_seconds = launch_config.PILOT_PAYLOAD_MAX_WALL_SECONDS[payload]
    assert f"--time={launch._format_wall_seconds(cap_seconds)}" in line
    assert "--dependency" not in line
    # Orchestrator addition 2026-11-10 (Slurm requeue + ParameterActivity
    # ResolvedConfigIdentityError, f8f5fc94): every pilot/part-(g) sbatch
    # must carry --no-requeue.
    assert "--no-requeue" in line
    assert resources.no_requeue is True


def test_submit_pilot_payload_refuses_time_over_its_own_cap(tmp_path: Path) -> None:
    """Mutant: cap above 01:00:00 accepted."""

    prefix = tmp_path / "runs"
    run_root = prefix / "h-v2" / "gpu-route-g" / "G"
    with pytest.raises(launch.LaunchError, match="must be at most"):
        launch.submit_pilot_payload(
            "gpu-route-g",
            science_python=sys.executable,
            executor_python=sys.executable,
            run_root=run_root,
            checkout_root=run_root / "checkout",
            run_root_prefix=str(prefix),
            t_wall="01:00:01",
            dry_run=True,
        )


def test_submit_pilot_payload_accepts_time_at_exactly_its_cap(tmp_path: Path) -> None:
    prefix = tmp_path / "runs"
    run_root = prefix / "h-v2" / "gpu-route-g" / "G"
    train_id = launch.submit_pilot_payload(
        "gpu-route-g",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=run_root,
        checkout_root=run_root / "checkout",
        run_root_prefix=str(prefix),
        t_wall="01:00:00",
        dry_run=True,
    )
    assert train_id is None


def test_submit_pilot_payload_refuses_a_run_root_outside_the_prefix(tmp_path: Path) -> None:
    """Mutant: run-root prefix check removed."""

    prefix = tmp_path / "runs"
    outside = tmp_path / "elsewhere" / "h-v2" / "gpu-route-g" / "G"

    def _forbidden(command: list) -> None:
        raise AssertionError("sbatch must never be called for a run root outside the prefix")

    with pytest.raises(launch.LaunchError, match="durable prefix"):
        launch.submit_pilot_payload(
            "gpu-route-g",
            science_python=sys.executable,
            executor_python=sys.executable,
            run_root=outside,
            checkout_root=outside / "checkout",
            run_root_prefix=str(prefix),
            runner=_forbidden,
        )


def test_main_requires_run_root_prefix_for_a_pilot_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(launch_config.PILOT_RUN_ROOT_PREFIX_ENV, raising=False)
    run_root = tmp_path / "runs" / "h-v2" / "gpu-route-g" / "G"
    with pytest.raises(launch.LaunchError, match="run-root-prefix"):
        launch.main(
            [
                "--run-root",
                str(run_root),
                "--payload",
                "gpu-route-g",
                "--stage",
                "train",
                "--science-python",
                sys.executable,
                "--executor-python",
                sys.executable,
                "--commit",
                "deadbeef",
                "--dry-run",
            ]
        )


def test_main_requires_explicit_stage_train_for_a_pilot_payload(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "h-v2" / "gpu-route-g" / "G"
    with pytest.raises(launch.LaunchError, match="--stage train is required"):
        launch.main(
            [
                "--run-root",
                str(run_root),
                "--payload",
                "gpu-route-g",
                "--science-python",
                sys.executable,
                "--executor-python",
                sys.executable,
                "--commit",
                "deadbeef",
                "--run-root-prefix",
                str(tmp_path / "runs"),
                "--dry-run",
            ]
        )


# ---------------------------------------------------------------------------
# run_job.py in-job identity / stratum / dispatch
# ---------------------------------------------------------------------------


def _fake_subprocess_run_for_device(device_name: str):
    """Fake ``subprocess.run`` reporting a fixed delivered GPU *device_name*."""

    class _Result:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def _run(command, **kwargs):
        if command[0] == "git":
            return _Result(0, "deadbeef\n")
        if command[0] == "hostname":
            return _Result(0, "fake-host\n")
        if "import torch" in " ".join(command):
            return _Result(0, f"{device_name}\n")
        return _Result(0, "")  # e.g. nvidia-smi -L

    return _run


def test_check_identity_uses_the_payloads_own_expected_stratum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutant: stratum check removed."""

    (tmp_path / "checkout").mkdir()
    monkeypatch.setattr(
        run_job.subprocess, "run", _fake_subprocess_run_for_device(launch_config.APPROVED_GPU_TEST_STRATUM)
    )
    # The science/pipeline-check route's stratum must be refused for a MIG device.
    with pytest.raises(run_job.JobError, match="does not match the approved stratum"):
        run_job.check_identity(tmp_path, "deadbeef", sys.executable)
    # gpu-route-g's own stratum must be accepted for the same delivered device.
    run_job.check_identity(
        tmp_path, "deadbeef", sys.executable, expected_stratum=launch_config.APPROVED_GPU_TEST_STRATUM
    )


def test_run_job_pilot_payload_refuses_eval_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_job, "check_identity", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUN", str(tmp_path))
    monkeypatch.setenv("SHA", "deadbeef")
    monkeypatch.setenv("SCIENCE_PYTHON", sys.executable)
    monkeypatch.setenv("EXECUTOR_PY", sys.executable)
    with pytest.raises(run_job.JobError, match="only runs stage 'train'"):
        run_job.run("eval", "G", pilot_payload="gpu-route-g")


def test_run_job_pilot_payload_refuses_a_foreign_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_job, "check_identity", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUN", str(tmp_path))
    monkeypatch.setenv("SHA", "deadbeef")
    monkeypatch.setenv("SCIENCE_PYTHON", sys.executable)
    monkeypatch.setenv("EXECUTOR_PY", sys.executable)
    for lineage in ("L1", "P", "pipeline-check-G", ""):
        with pytest.raises(run_job.JobError, match="refuses lineage"):
            run_job.run("train", lineage, pilot_payload="gpu-route-g")


def test_run_job_pilot_payload_builds_config_and_probes_module_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_job, "check_identity", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUN", str(tmp_path))
    monkeypatch.setenv("SHA", "deadbeef")
    monkeypatch.setenv("SCIENCE_PYTHON", sys.executable)
    monkeypatch.setenv("EXECUTOR_PY", sys.executable)
    monkeypatch.setattr(attached, "probe_admitted_a1_modules", lambda config, candidates, **kwargs: ([], list(candidates)))

    captured: dict[str, Any] = {}

    class _StopEarly(RuntimeError):
        pass

    def _capture(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _StopEarly("stop before prepare/activate -- this test only checks routing")

    monkeypatch.setattr(attached, "build_h_v2_train_plan", _capture)

    with pytest.raises(_StopEarly):
        run_job.run("train", "G", pilot_payload="gpu-route-g")

    assert captured["lineage"] == "gpu-route-g"
    assert captured["payload_label"] == "gpu-route-g"
    assert captured["config"]["trainer"]["max_steps"] == 1000
    assert captured["science_config_identity"]["payload"] == "gpu-route-g"


def test_build_sbatch_command_omits_no_requeue_flag_when_resources_set_it_false(tmp_path: Path) -> None:
    """Control for the 'gpu_test profile no_requeue=False' mutant.

    Orchestrator addition 2026-11-10: a Slurm requeue into the same run dir
    with ParameterActivity enabled raises ResolvedConfigIdentityError (exec
    fix tracked on f8f5fc94); the gpu_test profile and the pilot payloads
    must keep ``no_requeue=True`` so every emitted sbatch line carries
    ``--no-requeue`` (asserted on the real profiles in
    ``test_dry_run_prints_exactly_one_sbatch_line_...`` above). This proves
    the line ITSELF would go missing if that field were ever flipped --
    i.e. that a real 'no_requeue=False' mutant on
    ``launch_config.GPU_TEST_RESOURCES``/``RESOURCES`` is caught by that
    assertion, not merely by this synthetic ``resources=`` override.
    """

    import dataclasses

    mutated = dataclasses.replace(launch_config.GPU_TEST_RESOURCES, no_requeue=False)
    prefix = tmp_path / "runs"
    run_root = prefix / "h-v2" / "gpu-route-g" / "G"
    command = launch.build_sbatch_command(
        "train",
        "G",
        payload="gpu-route-g",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=run_root,
        checkout_root=run_root / "checkout",
        time_value="01:00:00",
        dependency=None,
        resources=mutated,
    )
    assert "--no-requeue" not in command
