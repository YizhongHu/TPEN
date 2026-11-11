"""PR 637 T2 round-1 regression proposals; execute on Cannon CPU only.

Reviewed head: 9713e76d946278464f77f7d6ecb019ec2d64c2a7, base f1c4600c.
These tests were authored but NOT run by the reviewer. No xfails: documented
regressions should be red until repaired. Each test names its target mutant.
The real receipt fixture requires the approved two-interpreter CPU harness.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

HERE = Path(__file__).parent
REPO_ROOT = HERE.resolve().parents[2]
PAYLOADS = [("gpu-route-g", "G", "pipeline_gpu_smoke_g", 1000),
            ("capacity-pilot", "P", "capacity_pilot", 20000)]
BASE_MODULES = {"tpen.callback.DataIntegrity", "tpen.callback.GradientStats"}
OPTIONAL_MODULES = {"tpen.callback.InitialCheckpoint", "tpen.callback.ParameterActivity",
                    "tpen.callback.ElectronNucleusDistanceQuantiles", "tpen.callback.SamplerHealth",
                    "tpen.callback.FactorScalars", "tpen.callback.ModuleNorms",
                    "tpen.callback.OptimizerState", "tpen.callback.RawTrainingBatchRecorder"}


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


@pytest.mark.parametrize("payload,lineage,namespace,steps", PAYLOADS)
def test_resolved_pilot_matches_independent_a1(tmp_path, payload, lineage, namespace, steps):
    """Kill pilot-only changes to Adam lr, sampler stride, or any model leaf."""
    config, identity = protocol.pilot_training_config(payload)
    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff", results_root=tmp_path / "results",
        lineage=payload, config=config, science_config_identity=identity, payload_label=payload,
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")
    actual = copy.deepcopy(attached._resolve_bound_config(
        attached._merge_config_with_overrides(prepared.effective_config, prepared.overrides)
    ))
    expected = yaml.safe_load((HERE / "testdata" / "science_a1_nonlogging.yaml").read_text())
    roster = json.loads((HERE / "contract_inputs.json").read_text())["measurement"]["seed_roster"]
    expected["runtime"]["seed"] = roster["non_science_seed_namespaces"][namespace]["training"][0]
    expected["sampler"]["seed"] = roster["non_science_seed_namespaces"][namespace]["sampler"][0]
    expected["trainer"].update(max_steps=steps, log_every_n_steps=1)
    expected["experiment"]["run_name"] = "tpen_h_v2_train"
    expected["run"].update(root=str(tmp_path / "results" / "attached"),
                            run_id=f"h-v2-train-{payload}", layout="flat")
    expected = OmegaConf.to_container(OmegaConf.create(expected), resolve=True)
    actual.pop("callbacks")
    actual.pop("loggers")
    assert actual == expected
    assert prepared.science_config_identity == {
        "payload": payload, "lineage": lineage, "seed_namespace": namespace,
        "scale_overlay_sha256": _sha(HERE / "configs" / "science_scale_overlay.yaml"),
        "payload_overlay_sha256": _sha(protocol.PILOT_PAYLOADS[payload]["overlay_path"]),
        "base_train_sha256": _sha(HERE / "configs" / "train.yaml"),
        "contract_inputs_sha256": _sha(HERE / "contract_inputs.json"),
        "materialized_config_sha256": _sha(prepared.effective_config),
    }


@pytest.mark.parametrize("payload,lineage,namespace,steps", PAYLOADS)
def test_pilot_reads_seed_values_from_ci(tmp_path, monkeypatch, payload, lineage, namespace, steps):
    """Kill duplicated seed literals that happen to match the current CI."""
    document = json.loads((HERE / "contract_inputs.json").read_text())
    seeds = document["measurement"]["seed_roster"]["non_science_seed_namespaces"][namespace]
    seeds["training"] = [1999001]
    seeds["sampler"] = [1999005]
    path = tmp_path / "contract_inputs.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(protocol, "CONTRACT_INPUTS_PATH", path)
    config, identity = protocol.pilot_training_config(payload)
    assert (config["runtime"]["seed"], config["sampler"]["seed"]) == (1999001, 1999005)
    assert identity["contract_inputs_sha256"] == _sha(path)


def test_endq_levels_are_the_frozen_a1_values():
    """Kill changing the shared quantile constant (old test shares its oracle)."""
    assert protocol._build_candidate_callback("tpen.callback.ElectronNucleusDistanceQuantiles") == {
        "_target_": "tpen.callback.ElectronNucleusDistanceQuantiles", "every_n_steps": 1,
        "quantiles": [0.0, 0.001, 0.01, 0.5, 0.99, 0.999],
    }


@pytest.mark.parametrize("code", ["profile-unavailable", "profile-mismatch", "malformed-json"])
def test_probe_propagates_checker_failures(tmp_path, monkeypatch, code):
    """F3: kill treating checker failure as a supported refusal/optional skip."""
    config, _ = protocol.pilot_training_config("gpu-route-g")
    failure = attached.ActivationRefusal(code, "checker did not supply a usable verdict")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(attached, "check_profile", fail)
    with pytest.raises(attached.ActivationRefusal) as caught:
        attached.probe_admitted_a1_modules(
            config, protocol.candidate_modules_to_append(), science_python=sys.executable,
            probe_config_path=tmp_path / "probe.yaml",
        )
    assert caught.value is failure


@pytest.mark.parametrize("admit_optional", [False, True])
def test_job_module_lists_partition_actual_config(tmp_path, monkeypatch, admit_optional):
    """F4: kill base-module omission and dropped/reversed activation receipt lists."""
    for name, value in {"RUN": str(tmp_path), "SHA": "a" * 40,
                        "SCIENCE_PYTHON": sys.executable, "EXECUTOR_PY": sys.executable}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(job, "check_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(attached, "assert_candidate_instantiates", lambda *args, **kwargs: None)

    def check(science_python, path, **kwargs):
        targets = {cb["_target_"] for cb in yaml.safe_load(Path(path).read_text())["callbacks"]}
        if not admit_optional and targets & OPTIONAL_MODULES:
            raise attached.ActivationRefusal("profile-refused", "unsupported callback")

    captured = {}

    def capture(prepared, **kwargs):
        captured.update(kwargs)
        captured["config"] = yaml.safe_load(Path(prepared.effective_config).read_text())

    monkeypatch.setattr(attached, "check_profile", check)
    monkeypatch.setattr(attached, "activate_h_v2_run", capture)
    job.run("train", "G", pilot_payload="gpu-route-g")
    configured = {cb["_target_"] for cb in captured["config"]["callbacks"]}
    enabled, skipped = set(captured["enabled_modules"]), set(captured["skipped_not_admitted"])
    assert enabled == configured & (BASE_MODULES | OPTIONAL_MODULES)
    assert enabled == BASE_MODULES | (OPTIONAL_MODULES if admit_optional else set())
    assert skipped == (set() if admit_optional else OPTIONAL_MODULES)
    assert not (enabled & skipped)
    assert captured["skipped_unspecified"] == []
    assert ("tpen.callback.ModuleNorms" in configured) is admit_optional


def _checkpoint(root, iteration, updates):
    path = root / f"step_{iteration:06d}"
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"synthetic model bytes; filesystem-only fixture")
    (path / "COMPLETE").write_text("complete\n")
    (path / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "kind": "tpen.checkpoint", "next_iteration": iteration,
        "completed_updates": updates, "created_at_unix": 0.0,
        "files": {"model": "model.pt"}, "hashes": {}, "runtime": {}, "provenance": {},
    }))
    return path


def test_inventory_uses_completed_updates_not_directory_cursor(tmp_path):
    """F1: kill deriving the scientific update coordinate from step_<iteration>."""
    _checkpoint(tmp_path, 0, 0)
    later = _checkpoint(tmp_path, 1007, 1000)
    inventory = attached.build_checkpoint_inventory(tmp_path)
    assert [(entry["update"], Path(entry["path"]).name) for entry in inventory] == [
        (0, "step_000000"), (1000, later.name),
    ]


def test_inventory_covers_all_file_names_and_streams_hashes(tmp_path, monkeypatch):
    """Kill hashing only model.pt, omitting a side file, or reading whole blobs."""
    checkpoint = _checkpoint(tmp_path, 0, 0)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer" * 300000)
    expected = {path.name: _sha(path) for path in checkpoint.iterdir() if path.is_file()}
    # Even COMPLETE-bearing .tmp dirs and manifest-less dirs are not committed.
    temporary = _checkpoint(tmp_path, 1000, 1000)
    temporary.rename(tmp_path / "step_001000.tmp")
    incomplete = tmp_path / "step_002000"
    incomplete.mkdir()
    (incomplete / "COMPLETE").write_text("complete\n")

    def reject_full_read(self):
        raise AssertionError("inventory must stream file bytes")

    monkeypatch.setattr(Path, "read_bytes", reject_full_read)
    inventory = attached.build_checkpoint_inventory(tmp_path)
    assert len(inventory) == 1
    assert inventory[0]["files"] == expected


def _launch_args(root, prefix, payload):
    return ["--run-root", str(root), "--run-root-prefix", str(prefix), "--commit", "a" * 40,
            "--payload", payload, "--stage", "train", "--science-python", sys.executable,
            "--executor-python", sys.executable, "--dry-run"]


@pytest.mark.parametrize("suffix", ["h-v2/capacity-pilot/P", "h-v2/gpu-route-g/P", "unrelated"])
def test_cli_refuses_wrong_payload_lineage_root(tmp_path, capsys, suffix):
    """F5: kill checking only a broad supplied prefix, without payload/lineage."""
    prefix = tmp_path / "runs"
    with pytest.raises(launch.LaunchError):
        launch.main(_launch_args(prefix / suffix, prefix, "gpu-route-g"))
    assert "sbatch" not in capsys.readouterr().out


@pytest.mark.parametrize("payload,lineage,namespace,steps", PAYLOADS)
def test_cli_uses_t_wall_within_payload_cap(tmp_path, monkeypatch, capsys, payload, lineage, namespace, steps):
    """F6: kill ignoring a smaller requested TRAIN wall time and using the cap."""
    prefix = tmp_path / "runs"
    monkeypatch.setenv("T_WALL", "00:30:00")
    assert launch.main(_launch_args(prefix / "h-v2" / payload / lineage, prefix, payload)) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("sbatch ")]
    assert len(lines) == 1
    assert "--time=00:30:00" in shlex.split(lines[0])


@pytest.mark.parametrize("payload,lineage,namespace,steps", PAYLOADS)
def test_cli_resource_contract_is_literal(tmp_path, monkeypatch, capsys, payload, lineage, namespace, steps):
    """K2: kill untyped GRES; also pin caps/CPU/memory independently of constants."""
    prefix = tmp_path / "runs"
    monkeypatch.delenv("T_WALL", raising=False)
    launch.main(_launch_args(prefix / "h-v2" / payload / lineage, prefix, payload))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("sbatch ")]
    assert len(lines) == 1
    argv = shlex.split(lines[0])
    assert "--no-requeue" in argv
    if payload == "gpu-route-g":
        assert {"--partition=gpu_test", "--gres=gpu:nvidia_a100_3g.20gb:1",
                "--cpus-per-task=4", "--mem=32G", "--time=01:00:00"} <= set(argv)
        assert not any(token.startswith("--constraint=") for token in argv)
    else:
        assert {"--gres=gpu:1", "--constraint=a100", "--time=08:00:00"} <= set(argv)


@pytest.mark.parametrize("payload,lineage,wrong_device", [
    ("gpu-route-g", "G", "NVIDIA A100-SXM4-80GB"),
    ("capacity-pilot", "P", "NVIDIA A100-SXM4-40GB MIG 3g.20gb"),
])
def test_wrong_device_exits_one_before_building_payload(tmp_path, monkeypatch, payload, lineage, wrong_device):
    """Kill swapped per-payload strata, a late identity check, or swallowed mismatch."""
    for name, value in {"RUN": str(tmp_path), "SHA": "a" * 40,
                        "SCIENCE_PYTHON": sys.executable, "EXECUTOR_PY": sys.executable}.items():
        monkeypatch.setenv(name, value)

    def run(command, **kwargs):
        stdout = "a" * 40 if command[0] == "git" else "fake-host"
        if "import torch" in " ".join(command):
            stdout = wrong_device
        return subprocess.CompletedProcess(command, 0, stdout, "")

    built = []
    monkeypatch.setattr(job.subprocess, "run", run)
    monkeypatch.setattr(protocol, "pilot_training_config", lambda *args: built.append(args))
    assert job.main(["train", lineage, "--payload", payload]) == 1
    assert not built, "identity refusal happened after payload construction"


@pytest.fixture(scope="module", params=["gpu-route-g", "capacity-pilot"])
def real_pilot_receipt(request, tmp_path_factory):
    """Run the actual CPU producer, handoff, executor and TRAIN receipt writer."""
    science_python = os.environ.get("H_V2_SCIENCE_PYTHON")
    executor_python = os.environ.get("H_V2_EXECUTOR_PYTHON")
    assert science_python and executor_python, "approved Cannon CPU harness interpreters required"
    root = tmp_path_factory.mktemp("pilot-receipt-r1")
    payload = request.param
    config, identity = protocol.pilot_training_config(payload)
    config["runtime"]["device"] = "cpu"
    config["sampler"].update(n_walkers=4, burn_in=1, n_steps=1)
    config["trainer"]["max_steps"] = 2
    enabled, skipped = attached.probe_admitted_a1_modules(
        config, protocol.candidate_modules_to_append(), science_python=science_python,
        probe_config_path=root / "probe.yaml", cwd=REPO_ROOT,
    )
    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=root / "handoff", results_root=root / "results", lineage=payload,
        config=config, science_config_identity=identity, payload_label=payload,
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=root / "handoff")
    allocation, worker = job.write_local_allocation_and_worker(
        root / "allocation", science_python=science_python,
        placeholder_config=prepared.effective_config, backend_job_id="t2-review-cpu-test",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    receipt_path = root / "train.h-v2.json"
    attached.activate_h_v2_run(
        prepared, science_python=science_python, executor_python=executor_python,
        allocation=allocation, worker_config=worker, store=root / "store",
        activation_receipt_path=root / "activation.json", h_v2_receipt_path=receipt_path,
        expected_source_commit=commit, dispatcher="local", timeout=180.0, cwd=REPO_ROOT,
        enabled_modules=enabled, skipped_not_admitted=skipped,
        skipped_unspecified=protocol.unspecified_modules(),
    )
    return json.loads(receipt_path.read_text()), prepared, payload


def test_real_train_receipt_binds_overlay_and_inventory(real_pilot_receipt):
    """K1: kill dropping/corrupting overlay SHA at protocol, plan, prepare or write."""
    receipt, prepared, payload = real_pilot_receipt
    identity = receipt["science_config_identity"]
    assert receipt["payload"] == payload
    assert identity == prepared.science_config_identity
    assert identity["payload_overlay_sha256"] == _sha(protocol.PILOT_PAYLOADS[payload]["overlay_path"])
    assert identity["scale_overlay_sha256"] == _sha(HERE / "configs" / "science_scale_overlay.yaml")
    assert identity["base_train_sha256"] == _sha(HERE / "configs" / "train.yaml")
    assert identity["contract_inputs_sha256"] == _sha(HERE / "contract_inputs.json")
    assert identity["materialized_config_sha256"] == _sha(prepared.effective_config)
    root = Path(receipt["segments"][0]["checkpoint_dir"]).parent
    expected = []
    for path in root.glob("step_*"):
        if path.name.endswith(".tmp") or not (path / "COMPLETE").is_file():
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        expected.append({"update": manifest["completed_updates"], "path": str(path),
                         "files": {str(p.relative_to(path)): _sha(p) for p in path.rglob("*") if p.is_file()}})
    assert receipt["checkpoint_inventory"] == sorted(expected, key=lambda entry: entry["update"])
    assert expected, "empty synthetic evidence cannot satisfy the real consumer check"


@pytest.mark.parametrize("mode,payload_rc", [("ok", 0), ("fail", 0), ("hang", 7)])
@pytest.mark.parametrize("payload,lineage", [("gpu-route-g", "G"), ("capacity-pilot", "P")])
def test_wrapper_gpu_sidecar_is_bounded_and_preserves_exit(tmp_path, mode, payload_rc, payload, lineage):
    """F2: kill absent sampling, fail-fast monitoring, leaked children or unbounded wait.

    Executes ONLY a generated wrapper with fake payload and nvidia-smi programs;
    no scheduler, CUDA, real GPU query, or TPEN training is called by this test.
    """
    run = tmp_path / "run"
    checkout = run / "checkout"
    script = checkout / "experiments" / "atomistic" / "h-v2" / "run_job.py"
    script.parent.mkdir(parents=True)
    script.write_text("import sys, time\ntime.sleep(6.2)\nsys.exit(" + str(payload_rc) + ")\n")
    (run / "logs").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.jsonl"
    smi = bin_dir / "nvidia-smi"
    smi.write_text(f"#!{sys.executable}\n" + """
import json, os, signal, sys, time
from pathlib import Path
mode = os.environ['FAKE_SMI_MODE']
while True:
    with open(os.environ['FAKE_SMI_CALLS'], 'a') as stream:
        stream.write(json.dumps({'argv': sys.argv[1:], 'pid': os.getpid(), 'time': time.monotonic()}) + '\\n')
    if mode == 'fail':
        sys.exit(19)
    if mode == 'hang':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(300)
    print('123, 47', flush=True)
    if not any(arg in ('-l', '--loop') or arg.startswith('--loop=') for arg in sys.argv[1:]):
        break
    time.sleep(5)
""")
    smi.chmod(0o755)
    command = launch.build_sbatch_command(
        "train", lineage, payload=payload, science_python=sys.executable,
        executor_python=sys.executable, run_root=run, checkout_root=checkout,
        time_value="01:00:00", dependency=None, commit="a" * 40,
        resources=launch.launch_config.PILOT_PAYLOAD_RESOURCES[payload],
    )
    wrap = next(token.split("=", 1)[1] for token in command if token.startswith("--wrap="))
    env = dict(os.environ, RUN=str(run), SHA="a" * 40, FAKE_SMI_MODE=mode,
               FAKE_SMI_CALLS=str(calls), PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    process = subprocess.Popen(["bash", "-c", wrap], env=env, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=12)
        assert process.returncode == payload_rc, (stdout, stderr)
        assert calls.is_file(), "wrapper never started the GPU sidecar"
        records = [json.loads(line) for line in calls.read_text().splitlines()]
        assert records
        for record in records:
            args = " ".join(record["argv"])
            assert "memory.used" in args and "utilization.gpu" in args
        if mode == "ok":
            assert len(records) >= 2, "one initial sample is not a five-second series"
            assert 4 <= records[1]["time"] - records[0]["time"] <= 7
            assert any("123, 47" in path.read_text() for path in (run / "logs").rglob("*") if path.is_file())
        # A detached or TERM-resistant monitor must not survive payload completion.
        for pid in {record["pid"] for record in records}:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        # Test-owned process group only; contains no real job or scientific data.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=2)
