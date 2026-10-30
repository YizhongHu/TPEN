"""Focused controls for the tracked-lock Polaris GPU qualification route."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from tpen.checkpoint.save import save_checkpoint


ROOT = Path(__file__).parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage = _load("gpu_stage")
job = _load("gpu_job")
qualification = _load("qualification")


def test_cu126_lock_derivation_and_missing_entry_control() -> None:
    lock = stage.tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    assert stage.expected_torch_version(lock, "cu126") == "2.12.0+cu126"
    missing = copy.deepcopy(lock)
    project = next(package for package in missing["package"] if package.get("source", {}).get("editable") == ".")
    project["optional-dependencies"]["cu126"] = []
    try:
        stage.expected_torch_version(missing, "cu126")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a lock without the cu126 Linux torch entry")
    assert job.expected_torch_version(lock, "cu126") == "2.12.0+cu126"


def test_job_contract_has_single_profile_extra_and_visible_device() -> None:
    profile = json.loads((ROOT / "facility_profile.json").read_text(encoding="utf-8"))
    assert profile == {"runtime_kind": "polaris-gpu-qualification", "extra": "cu126"}
    script = (ROOT / "job.pbs").read_text(encoding="utf-8")
    helper_script = (ROOT / "gpu_job.py").read_text(encoding="utf-8")
    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert script.index("export CUDA_VISIBLE_DEVICES=0") < script.index("exec ")
    assert '"--offline", "--extra", manifest["extra"], "--locked"' in helper_script
    assert "LD_LIBRARY_PATH" not in script
    source = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.iterdir() if path.is_file())
    assert all(token not in source for token in ("overlay_root", "overlay_python", "libshim", "facility_site_packages"))


def test_prime_and_dry_sync_uses_profile_extra(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    paths = stage.gpu_layout(tmp_path / "run")
    paths["run_root"].mkdir()
    profile = {"runtime_kind": stage.RUNTIME_KIND, "extra": "cu128"}
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        Path(kwargs["env"]["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"]).mkdir(parents=True, exist_ok=True)
        return ""

    def fake_subprocess(args, **kwargs):
        calls.append(list(args))
        Path(kwargs["env"]["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(stage, "_run", fake_run)
    monkeypatch.setattr(stage.subprocess, "run", fake_subprocess)
    stage.prime_and_dry_sync(target, profile, paths, ["python", "-m", "uv"])
    assert len(calls) == 2
    for call in calls:
        assert call[call.index("--extra") + 1] == "cu128"
        assert "--locked" in call
    assert calls[1][calls[1].index("--offline") + 1] == "--extra"


def test_job_torch_probe_controls() -> None:
    good = {"torch": "2.12.0+cu126", "available": True, "count": 1, "name": "A100", "op": 2.0, "visible": "0", "file": "/local/scratch/123/env/lib/python/site-packages/torch/__init__.py"}
    env = Path("/local/scratch/123/env")
    assert good["torch"] == job.expected_torch_version({"package": [{"name": "x", "source": {"editable": "."}, "optional-dependencies": {"cu126": [{"name": "torch", "version": "2.12.0+cu126", "marker": "sys_platform != 'darwin'"}]}}]}, "cu126")
    assert good["file"].startswith(str(env))
    for mutant in ({**good, "torch": "2.11.0+cu128"}, {**good, "file": "/usr/lib/python/site-packages/torch/__init__.py"}, {**good, "visible": None}):
        path = Path(mutant["file"]).resolve()
        ok = (mutant["torch"] == "2.12.0+cu126" and mutant["available"] and mutant["count"] == 1 and "A100" in mutant["name"] and mutant["op"] == 2.0 and mutant["visible"] == "0" and (path == env or env in path.parents))
        assert not ok


def test_invalid_queue_is_rejected_before_work(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stage, "_harness_repo", lambda: (_ for _ in ()).throw(AssertionError("clone/work started")))
    args = type("Args", (), {"target_sha": "a" * 40, "walkers": 2048, "updates": 2000, "checkpoint_every": 1000, "nodes": 1, "cell_id": "cell", "queue": "prod", "evidence_root": str(tmp_path)})()
    try:
        stage.stage(args)
    except ValueError as error:
        assert "not allowed" in str(error)
    else:
        raise AssertionError("accepted an unprofiled queue")


def test_only_current_q_surface_is_admitted() -> None:
    for walkers in (2048, 4096):
        args = type("Args", (), {"target_sha": "a" * 40, "walkers": walkers, "updates": 2000,
                                  "checkpoint_every": 1000, "nodes": 1, "cell_id": "cell"})()
        stage._validate(args)
    for field, value in (("walkers", 1000), ("updates", 20), ("checkpoint_every", 20), ("nodes", 2)):
        values = {"target_sha": "a" * 40, "walkers": 2048, "updates": 2000,
                  "checkpoint_every": 1000, "nodes": 1, "cell_id": "cell"}
        values[field] = value
        try:
            stage._validate(type("Args", (), values)())
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid {field}={value}")


def test_load_env_preserves_quotes_and_dollar_literals(tmp_path: Path) -> None:
    env = tmp_path / "run.env"
    env.write_text("RUN_MANIFEST='a path/$X/with quote'\n", encoding="utf-8")
    assert job._load_env(env)["RUN_MANIFEST"] == "a path/$X/with quote"


def test_no_hydra_or_run_root_literals_in_launcher_sources(tmp_path: Path) -> None:
    files = [path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts and "tests" not in path.parts]
    sources = "\n".join(path.read_text(encoding="utf-8") for path in files)
    run_root_marker = "/" + "eagle" + "/"
    hydra_marker = "Hy" + "dra"
    assert run_root_marker not in sources
    assert hydra_marker not in sources and hydra_marker.lower() not in sources
    assert not re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", sources)
    planted = tmp_path / "planted.py"
    planted.write_text("x='" + "a" * 40 + "' " + run_root_marker + "run\n", encoding="utf-8")
    assert re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", planted.read_text())


def test_actual_run_stage_q_parser_rejects_unsupported_argument() -> None:
    target = REPO / job.CANONICAL_STAGE
    assert target.is_file(), target
    spec = importlib.util.spec_from_file_location("real_run_stage_q", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    try:
        module.build_parser().parse_args(["--not-a-real-override"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("real run_stage_q parser accepted an unsupported argument")


def test_verifier_writes_pass_and_copy_controls_fail(tmp_path: Path) -> None:
    output = tmp_path / "output" / "owner"
    output.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    context = SimpleNamespace(cfg=OmegaConf.create({}), run_dir=output,
                              metadata=SimpleNamespace(dtype="float32", device="cpu", run_id="fixture", run_name="fixture", extra={}, git_commit=None, git_branch=None, dirty_worktree=False, command=None))
    trainer = SimpleNamespace(state_dict=lambda: {"completed_updates": 1000, "next_iteration": 1000})
    for step in (1000, 2000):
        trainer.state_dict = lambda step=step: {"completed_updates": step, "next_iteration": step}
        save_checkpoint(output_dir=output / "checkpoints", next_iteration=step, completed_updates=step, model=model, context=context, trainer=trainer, save_optimizer=False, save_sampler=False, save_rng=False, save_trainer=True)
    (output / "selection_receipt.json").write_text(json.dumps({"selection": {"selected": {"context": "O-composite"}}, "walkers": 2048}) + "\n")
    (output / "resolved_config.yaml").write_text("context: O-composite\n")
    (output / "COMPLETE").write_text("completed\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runtime_kind": stage.RUNTIME_KIND, "cell": {"context": "O-composite", "walkers": 2048, "updates": 2000, "checkpoint_every": 1000}, "output_root": str(tmp_path / "output"), "evidence_dir": str(tmp_path / "evidence")}) + "\n")
    assert qualification.qualify(manifest, tmp_path / "evidence") == 0

    def control(name: str) -> Path:
        destination = tmp_path / "controls" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path / "output", destination)
        return destination

    bad = control("counters")
    for filename in ("manifest.json", "trainer.json"):
        path = bad / "owner" / "checkpoints" / "step_002000" / filename
        value = json.loads(path.read_text()); value["completed_updates"] = 1999; value["next_iteration"] = 1999; path.write_text(json.dumps(value))
    evidence = tmp_path / "evidence-counters"
    assert qualification.qualify(manifest, evidence, artifact_root=bad) != 0
    assert next(row for row in json.loads((evidence / "qualification.json").read_text())["predicates"] if row["name"] == "completed_updates")["status"] == "FAIL"
    missing = control("missing-complete")
    (missing / "owner" / "checkpoints" / "step_002000" / "COMPLETE").unlink()
    evidence = tmp_path / "evidence-complete"
    assert qualification.qualify(manifest, evidence, artifact_root=missing) != 0
    assert next(row for row in json.loads((evidence / "qualification.json").read_text())["predicates"] if row["name"] == "checkpoint_complete")["status"] == "FAIL"
    nan = control("nan")
    torch.save({"weight": torch.tensor([float("nan")])}, nan / "owner" / "checkpoints" / "step_002000" / "model.pt")
    evidence = tmp_path / "evidence-nan"
    assert qualification.qualify(manifest, evidence, artifact_root=nan) != 0
    assert next(row for row in json.loads((evidence / "qualification.json").read_text())["predicates"] if row["name"] == "final_tensors_finite")["status"] == "FAIL"
    wrong = control("wrong-cell")
    (wrong / "owner" / "selection_receipt.json").write_text(json.dumps({"selection": {"selected": {"context": "O-raw"}}, "walkers": 2048}))
    evidence = tmp_path / "evidence-cell"
    assert qualification.qualify(manifest, evidence, artifact_root=wrong) != 0
    assert next(row for row in json.loads((evidence / "qualification.json").read_text())["predicates"] if row["name"] == "resolved_cell_identity")["status"] == "FAIL"


def test_job_main_green_and_seam_controls(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "target"
    route = target / job.CANONICAL_STAGE
    route.parent.mkdir(parents=True)
    route.write_text("# route\n")
    (target / "uv.lock").write_text("version = 1\nrevision = 1\n\n[[package]]\nname = 'tpen'\nsource = { editable = '.' }\n")
    target_sha = "a" * 40
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "gpu_job.py").write_text("helper\n")
    manifest_path = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    run_env = tmp_path / "run.env"
    job_env = tmp_path / "job-env"
    values = {"RUN_MANIFEST": str(manifest_path), "RUN_PYTHON": sys.executable, "RUN_JOB_PY": str(harness / "gpu_job.py"), "RUN_PROJECT": str(target), "RUN_CACHE_DIR": str(tmp_path / "cache")}
    verify_profile = json.loads((ROOT.parent / "verify" / "facility_profile.json").read_text(encoding="utf-8"))
    manifest = {"runtime_kind": job.RUNTIME_KIND, "target_sha": target_sha, "required_ancestors": [], "nodes": 1, "target_repo": str(target), "job_helper": str(harness / "gpu_job.py"), "copied_file_sha256": {}, "staging_receipt": str(receipt), "evidence_dir": str(tmp_path / "evidence"), "output_root": str(tmp_path / "output"), "extra": "cu126", "expected_torch": "2.12.0+cu126", "cache_dir": str(tmp_path / "cache"), "verify_profile": verify_profile, "cell": {"stage": "Q", "context": "O-composite", "optimizer": "adam", "lr": 0.005, "beta2": 0.999, "seed_label": 810001, "walkers": 2048}}
    identity = {"prefix": str(job_env.resolve()), "executable": str((job_env / "bin" / "python").resolve()), "base_prefix": "/facility"}
    probe = {"torch": "2.12.0+cu126", "cuda": "12.6", "file": str(job_env / "lib/python/site-packages/torch/__init__.py"), "available": True, "count": 1, "name": "A100", "op": 2.0, "visible": "0"}
    def configure() -> None:
        job_env.mkdir(parents=True, exist_ok=True)
        (target / "uv.lock").write_text("version = 1\nrevision = 1\n\n[[package]]\nname = 'tpen'\nsource = { editable = '.' }\noptional-dependencies = { cu126 = [{ name = 'torch', version = '2.12.0+cu126' }] }\n")
        manifest_path.write_text(json.dumps(manifest))
        receipt.write_text(json.dumps({"runtime_kind": job.RUNTIME_KIND, "target_sha": target_sha, "cache_dir": str(tmp_path / "cache"), "extra": "cu126"}))
        run_env.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
        (tmp_path / "nodes").write_text("nid001.fqdn\n")
        monkeypatch.setenv("PBS_JOBID", "job")
        monkeypatch.setenv("PBS_NODEFILE", str(tmp_path / "nodes"))
        monkeypatch.setenv("JOB_ENV", str(job_env))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setattr(sys, "argv", ["gpu_job.py", str(run_env)])
        monkeypatch.setattr(job, "hostname", lambda: "nid001")
        monkeypatch.setattr(job, "nodefile_entries", lambda: ["nid001.fqdn"])
        monkeypatch.setattr(job, "git_head", lambda repo: target_sha)
        monkeypatch.setattr(job, "appledoubles", lambda repo: [])
        monkeypatch.setattr(job, "expected_torch_version", lambda lock, extra: "2.12.0+cu126")
        monkeypatch.setattr(job, "_uv_version", lambda uv: "uv 0.8.23")
        monkeypatch.setattr(job, "_venv_identity", lambda python: identity)
        monkeypatch.setattr(job, "_package_versions", lambda python: {"torch": "2.12.0+cu126"})
        def fake_probe(python, env):
            assert env.get("CUDA_VISIBLE_DEVICES") == "0"
            assert env.get("JOB_ENV") == str(job_env)
            return probe
        monkeypatch.setattr(job, "torch_probe", fake_probe)
        monkeypatch.setattr(job, "run_route", lambda *args: 0)
        monkeypatch.setattr(job, "run_qualification", lambda *args: 2)
        monkeypatch.setattr(job.subprocess, "check_output", lambda *args, **kwargs: "blob\n")
        monkeypatch.setattr(job.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))

    configure()
    assert job.main() == 2
    controls = [
        ("missing receipt", lambda: receipt.unlink(), 60),
        ("bad receipt", lambda: receipt.write_text("{}"), 61),
        ("route", lambda: route.unlink(), 64),
        ("login", lambda: monkeypatch.setattr(job, "hostname", lambda: "polaris-login-01"), 70),
        ("nodefile", lambda: monkeypatch.setattr(job, "nodefile_entries", lambda: ["other"]), 71),
        ("head", lambda: monkeypatch.setattr(job, "git_head", lambda repo: "b" * 40), 72),
        ("appledouble", lambda: monkeypatch.setattr(job, "appledoubles", lambda repo: ["._bad"]), 73),
        ("ancestor", lambda: (manifest.update(required_ancestors=["c" * 40]), monkeypatch.setattr(job.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1))), 79),
        ("copied", lambda: manifest.update(copied_file_sha256={"gpu_job.py": "0" * 64}), 81),
        ("kind", lambda: manifest.update(runtime_kind="wrong"), 84),
        ("topology", lambda: monkeypatch.setattr(job, "nodefile_entries", lambda: ["nid001", "nid002"]), 85),
        ("missing JOB_ENV", lambda: monkeypatch.setenv("JOB_ENV", ""), 76),
        ("sync", lambda: monkeypatch.setattr(job.subprocess, "run", lambda args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(9, args)) if "sync" in args else subprocess.CompletedProcess(args, 0)), 80),
        ("lock shape", lambda: (target / "uv.lock").write_text("not = [valid\n"), 78),
        ("probe", lambda: monkeypatch.setattr(job, "torch_probe", lambda python, env: {"torch": "wrong"}), 83),
        ("route rc", lambda: monkeypatch.setattr(job, "run_route", lambda *args: 7), 1),
    ]
    for name, mutate, expected in controls:
        monkeypatch.undo()
        route.parent.mkdir(parents=True, exist_ok=True)
        route.write_text("# route\n")
        manifest.update(runtime_kind=job.RUNTIME_KIND, required_ancestors=[], nodes=1, copied_file_sha256={})
        configure()
        mutate()
        manifest_path.write_text(json.dumps(manifest))
        assert job.main() == expected, name


def test_job_pbs_exports_child_environment_and_cuda_control(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    fake_python = root / "facility-python"
    observed = root / "observed"
    fake_python.write_text("#!/usr/bin/env bash\nprintf '%s|%s|%s|%s|%s' \"$CUDA_VISIBLE_DEVICES\" \"$JOB_ENV\" \"$1\" \"$2\" \"$RUN_PYTHON\" > '" + str(observed) + "'\n", encoding="utf-8")
    fake_python.chmod(0o755)
    job_env = root / "scratch" / "job" / "tpen-gpu-env"
    env_file = root / "run.env"
    env_file.write_text("RUN_PYTHON=" + str(fake_python) + "\nRUN_JOB_PY=" + str(root / "helper.py") + "\n", encoding="utf-8")
    (root / "helper.py").write_text("helper\n")
    script = ROOT / "job.pbs"
    executable = root / "job.pbs"
    executable.write_text(script.read_text(encoding="utf-8").replace("/local/scratch", str(root / "scratch")), encoding="utf-8")
    base_env = {"RUN_ENV_PATH": str(env_file), "PBS_JOBID": "job", "PATH": os.environ.get("PATH", "")}
    result = subprocess.run(["bash", str(executable)], env=base_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert observed.read_text().strip() == "0|" + str(job_env) + "|" + str(root / "helper.py") + "|" + str(env_file) + "|"
    mutant = root / "mutant.pbs"
    mutant.write_text(executable.read_text(encoding="utf-8").replace('exec "$RUN_PYTHON"', 'exec "$JOB_ENV/bin/python"'), encoding="utf-8")
    observed.unlink()
    result = subprocess.run(["bash", str(mutant)], env=base_env, text=True, capture_output=True)
    assert result.returncode != 0


def test_stage_real_clone_and_qsub_fakes(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "test"], check=True)
    (source / "uv.lock").write_text("version = 1\nrevision = 1\n\n[[package]]\nname = 'tpen'\nsource = { editable = '.' }\noptional-dependencies = { cu126 = [{ name = 'torch', version = '2.12.0+cu126' }] }\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "target"], check=True)
    target_sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(source), str(remote)], check=True)
    harness = tmp_path / "harness"
    harness.mkdir()
    subprocess.run(["git", "init", "-q", str(harness)], check=True)
    subprocess.run(["git", "-C", str(harness), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(harness), "config", "user.name", "test"], check=True)
    for name in stage.STORED_FILES:
        (harness / name).write_text("fixture\n")
    subprocess.run(["git", "-C", str(harness), "add", "."], check=True)
    subprocess.run(["git", "-C", str(harness), "commit", "-q", "-m", "harness"], check=True)
    monkeypatch.setattr(stage, "_harness_repo", lambda: harness)
    monkeypatch.setattr(stage.getpass, "getuser", lambda: "rhu")
    original_run = stage._run
    original_subprocess = stage.subprocess.run
    mode = {"value": "ok"}
    qstat_calls = {"count": 0}
    qsub_calls = []

    def fake_run(args, **kwargs):
        if "--version" in args and "uv" in args:
            return "uv 0.8.23"
        if args and args[0] == "qsub":
            qsub_calls.append(args)
            return "123.polaris"
        if "sync" in args and mode["value"] == "prime-fail":
            raise subprocess.CalledProcessError(9, args)
        if "sync" in args:
            env = kwargs["env"]
            Path(env["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            Path(env["UV_PROJECT_ENVIRONMENT"]).mkdir(parents=True, exist_ok=True)
            return ""
        return original_run(args, **kwargs)

    def fake_subprocess(args, **kwargs):
        if args and args[0] == "qstat":
            qstat_calls["count"] += 1
        if args[:2] == ["qstat", "-Qf"]:
            return subprocess.CompletedProcess(args, 0, "Queue: debug\nmax_queued = [u:PBS_GENERIC=1]\nqueued_jobs_threshold = [u:PBS_GENERIC=1]\n", "")
        if args == ["qstat", "-f"]:
            if mode["value"] == "gate-fail" and qstat_calls["count"] > 3:
                return subprocess.CompletedProcess(args, 0, "Job Id: 123.polaris\n    Job_Owner = rhu@polaris\n    job_state = Q\n    queue = debug\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["qstat", "-u"]:
            if mode["value"] == "gate-fail" and qstat_calls["count"] > 3:
                return subprocess.CompletedProcess(args, 0, "Job id User Time Use S Queue\n123.polaris rhu 00:00:00 Q debug\n", "")
            return subprocess.CompletedProcess(args, 0, "Job id User Time Use S Queue\n", "")
        if "--offline" in args:
            if mode["value"] == "dry-fail":
                raise subprocess.CalledProcessError(9, args)
            env = kwargs["env"]
            Path(env["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
            Path(env["UV_PROJECT_ENVIRONMENT"]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(args, 0)
        return original_subprocess(args, **kwargs)

    monkeypatch.setattr(stage, "_run", fake_run)
    monkeypatch.setattr(stage.subprocess, "run", fake_subprocess)
    def args(name: str, root: Path):
        return type("Args", (), {"target_sha": target_sha, "cell_id": name, "context": "O-composite", "optimizer": "adam", "lr": 0.005, "beta2": 0.999, "seed_label": 810001, "walkers": 2048, "updates": 2000, "checkpoint_every": 1000, "stage": "Q", "nodes": 1, "queue": "debug", "walltime": "00:20:00", "account": "HetRxnEnergy", "evidence_root": str(root), "required_ancestor": [], "remote": str(remote)})()

    root = tmp_path / "runs"
    try:
        assert stage.stage(args("green", root)) == 0
    except Exception as error:
        raise AssertionError(f"stage raised before qsub: {error}") from error
    run_root = root / "green"
    run_root = run_root.resolve()
    manifest = json.loads((run_root / "manifest.json").read_text())
    expected = ["qsub", "-V", "-A", "HetRxnEnergy", "-q", "debug", "-l", "select=1:system=polaris", "-l", "walltime=00:20:00", "-l", "filesystems=home:eagle", "-o", str(run_root / "pbs") + "/", "-e", str(run_root / "pbs") + "/", "-v", "RUN_ENV_PATH=" + str(run_root / "run.env"), str(run_root / "harness" / "job.pbs")]
    assert qsub_calls[-1] == expected and manifest["extra"] == "cu126", repr(qsub_calls[-1]) + " != " + repr(expected)
    run_env_text = (run_root / "run.env").read_text()
    for key, value in (("RUN_CACHE_DIR", manifest["cache_dir"]), ("RUN_PROJECT", manifest["target_repo"])):
        assert f"{key}=" in run_env_text and value in run_env_text
    assert json.loads((run_root / "staging-receipt.json").read_text()) == {"runtime_kind": stage.RUNTIME_KIND, "target_sha": target_sha, "cache_dir": manifest["cache_dir"], "extra": "cu126", "manifest": str(run_root / "manifest.json")}
    for name, failure in (("prime", "prime-fail"), ("dry", "dry-fail")):
        mode["value"] = failure
        failure_root = tmp_path / ("runs-" + name)
        try:
            stage.stage(args(name, failure_root))
        except (RuntimeError, subprocess.CalledProcessError):
            pass
        else:
            raise AssertionError(f"{failure} unexpectedly succeeded")
        assert not (failure_root / name / "staging-receipt.json").exists()
    mode["value"] = "ok"
    gate_root = tmp_path / "runs-gate"
    before = len(qsub_calls)
    qstat_calls["count"] = 0
    mode["value"] = "gate-fail"
    try:
        stage.stage(args("gate", gate_root))
    except RuntimeError:
        pass
    else:
        raise AssertionError("second queue gate unexpectedly admitted qsub")
    assert len(qsub_calls) == before and (gate_root / "gate" / "staging-receipt.json").is_file()
    decisions = list(gate_root.glob("**/*qstat-decision.json"))
    assert decisions and json.loads(decisions[-1].read_text())["allowed"] is False
