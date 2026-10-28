"""Two-sided self-tests for the actual B0 entry points."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from tpen.checkpoint.save import save_checkpoint


ROOT = Path(__file__).parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage = _load("gpu_stage")
qualification = _load("qualification")
job = _load("gpu_job")


def test_only_current_q_surface_is_admitted() -> None:
    for walkers in (2048, 4096):
        args = type("Args", (), {"target_sha": "a" * 40, "walkers": walkers,
            "updates": 2000, "checkpoint_every": 1000, "nodes": 1,
            "cell_id": "cell",})()
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


def test_job_script_has_literal_manifest_boundary_and_no_uv() -> None:
    script = (ROOT / "job.pbs").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "RUN_ENV_PATH" in script and 'source "$RUN_ENV_PATH"' in script
    assert 'exec "$RUN_PYTHON" "$RUN_JOB_PY"' in script
    assert "qsub" not in script and all(token not in script for token in ("-m uv", "uv sync", "uv pip"))


def test_job_script_exports_profile_before_exec(tmp_path: Path) -> None:
    run_dir = tmp_path / "run with spaces"
    run_dir.mkdir()
    fake_python = run_dir / "fake python.sh"
    fake_python.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$LD_LIBRARY_PATH|$CUDA_VISIBLE_DEVICES|$PYTHONDONTWRITEBYTECODE\"\n")
    fake_python.chmod(0o755)
    fake_job = run_dir / "job helper.py"
    fake_job.write_text("# fixture\n")
    run_env = run_dir / "run env"
    run_env.write_text("RUN_PYTHON=" + shlex.quote(str(fake_python)) + "\nRUN_JOB_PY=" + shlex.quote(str(fake_job)) + "\nRUN_LIBSHIM=libshim\nRUN_CUDA13_LIB=cuda13\nRUN_CUDA129_LIB=cuda129\n")
    script = ROOT / "job.pbs"
    for inherited, expected in (("", "libshim:cuda13:cuda129|0|1"), ("/opt/cray/libfabric/lib", "libshim:cuda13:cuda129:/opt/cray/libfabric/lib|0|1")):
        result = subprocess.run(["bash", str(script)], env={"RUN_ENV_PATH": str(run_env), "LD_LIBRARY_PATH": inherited}, check=True, text=True, capture_output=True)
        assert result.stdout.strip() == expected
    mutant = run_dir / "mutant.pbs"
    mutant.write_text(script.read_text().replace('export LD_LIBRARY_PATH=', '# export LD_LIBRARY_PATH=', 1))
    result = subprocess.run(["bash", str(mutant)], env={"RUN_ENV_PATH": str(run_env), "LD_LIBRARY_PATH": "/opt/cray/libfabric/lib"}, check=True, text=True, capture_output=True)
    assert result.stdout.strip() != "libshim:cuda13:cuda129:/opt/cray/libfabric/lib|0|1"


def test_verifier_writes_pass_and_copy_controls_fail(tmp_path: Path) -> None:
    output = tmp_path / "output" / "owner"
    output.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    context = SimpleNamespace(
        cfg=OmegaConf.create({}), run_dir=output,
        metadata=SimpleNamespace(dtype="float32", device="cpu", run_id="fixture", run_name="fixture", extra={}, git_commit=None, git_branch=None, dirty_worktree=False, command=None),
    )
    trainer = SimpleNamespace(state_dict=lambda: {"completed_updates": 1000, "next_iteration": 1000})
    save_checkpoint(output_dir=output / "checkpoints", next_iteration=1000, completed_updates=1000, model=model, context=context, trainer=trainer, save_optimizer=False, save_sampler=False, save_rng=False, save_trainer=True)
    trainer.state_dict = lambda: {"completed_updates": 2000, "next_iteration": 2000}
    save_checkpoint(output_dir=output / "checkpoints", next_iteration=2000, completed_updates=2000, model=model, context=context, trainer=trainer, save_optimizer=False, save_sampler=False, save_rng=False, save_trainer=True)
    (output / "selection_receipt.json").write_text(json.dumps({"selection": {"selected": {"context": "O-composite"}}, "walkers": 2048}) + "\n")
    (output / "resolved_config.yaml").write_text("context: O-composite\n")
    (output / "COMPLETE").write_text("completed\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runtime_kind": stage.RUNTIME_KIND, "cell": {"context": "O-composite", "walkers": 2048, "updates": 2000, "checkpoint_every": 1000}, "output_root": str(tmp_path / "output"), "evidence_dir": str(tmp_path / "evidence")}) + "\n")
    assert qualification.qualify(manifest, tmp_path / "evidence") == 0

    # Every negative arm is run against a copy of the writer-produced layout.
    def copy_control(name: str) -> Path:
        destination = tmp_path / "controls" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path / "output", destination)
        return destination

    h1 = copy_control("h1")
    for filename in ("manifest.json", "trainer.json"):
        path = h1 / "owner" / "checkpoints" / "step_002000" / filename
        value = json.loads(path.read_text())
        value["completed_updates"] = 1999
        value["next_iteration"] = 1999
        path.write_text(json.dumps(value) + "\n")
    assert qualification.qualify(manifest, tmp_path / "evidence-h1-updates", artifact_root=h1) != 0
    assert "completed_updates" in {row["name"] for row in json.loads((tmp_path / "evidence-h1-updates" / "qualification.json").read_text())["predicates"]}
    missing = copy_control("missing-complete")
    (missing / "owner" / "checkpoints" / "step_002000" / "COMPLETE").unlink()
    assert qualification.qualify(manifest, tmp_path / "evidence-h1", artifact_root=missing) != 0
    missing_complete = json.loads((tmp_path / "evidence-h1" / "qualification.json").read_text())
    assert next(row for row in missing_complete["predicates"] if row["name"] == "checkpoint_complete")["status"] == "FAIL"
    nan = copy_control("nan")
    torch.save({"weight": torch.tensor([float("nan")])}, nan / "owner" / "checkpoints" / "step_002000" / "model.pt")
    assert qualification.qualify(manifest, tmp_path / "evidence-nan", artifact_root=nan) != 0
    assert next(row for row in json.loads((tmp_path / "evidence-nan" / "qualification.json").read_text())["predicates"] if row["name"] == "final_tensors_finite")["status"] == "FAIL"
    wrong = copy_control("wrong-cell")
    (wrong / "owner" / "selection_receipt.json").write_text(json.dumps({"selection": {"selected": {"context": "O-raw"}}, "walkers": 2048}) + "\n")
    assert qualification.qualify(manifest, tmp_path / "evidence-cell", artifact_root=wrong) != 0
    assert next(row for row in json.loads((tmp_path / "evidence-cell" / "qualification.json").read_text())["predicates"] if row["name"] == "resolved_cell_identity")["status"] == "FAIL"


def test_no_hydra_or_run_root_literals_in_launcher_sources(tmp_path: Path) -> None:
    files = [path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    sources = "\n".join(path.read_text() for path in files)
    marker = "/" + "eagle" + "/"
    assert marker not in sources
    import re
    assert not re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", sources)
    planted = tmp_path / "_identity-scanner-fixture.tmp"
    try:
        planted.write_text("quoted='" + "a" * 40 + "'\n")
        assert re.search(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", planted.read_text())
    finally:
        planted.unlink()


def test_actual_run_stage_q_parser_rejects_unsupported_argument() -> None:
    target = ROOT.parents[2] / "atomistic" / "he-importance" / "run_stage_q.py"
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


def test_load_env_preserves_quotes_and_dollar_literals(tmp_path: Path) -> None:
    env = tmp_path / "run.env"
    env.write_text("RUN_MANIFEST='a path/$X/with quote'\n")
    assert job._load_env(env)["RUN_MANIFEST"] == "a path/$X/with quote"


def test_job_main_green_and_seam_controls(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "target"
    (target / "experiments/atomistic/he-importance").mkdir(parents=True)
    (target / job.CANONICAL_STAGE).write_text("# fixture route\n")
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(target), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(target), "add", "."], check=True)
    subprocess.run(["git", "-C", str(target), "commit", "-q", "-m", "fixture"], check=True)
    target_sha = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    run_env = tmp_path / "run.env"; manifest_path = tmp_path / "manifest.json"; receipt = tmp_path / "receipt.json"
    harness = tmp_path / "harness"; harness.mkdir(); (harness / "gpu_job.py").write_text("x")
    profile = {"overlay_python": sys.executable, "overlay_root": str(tmp_path), "facility_site_packages": str(tmp_path), "libshim": str(tmp_path), "cuda13_lib": str(tmp_path), "cuda129_lib": str(tmp_path)}
    manifest = {"runtime_kind": job.RUNTIME_KIND, "target_sha": target_sha, "required_ancestors": [], "nodes": 1, "target_repo": str(target), "job_helper": str(harness / "gpu_job.py"), "copied_file_sha256": {}, "staging_receipt": str(receipt), "runtime_profile": profile, "evidence_dir": str(tmp_path / "evidence"), "cell": {"stage": "Q", "context": "O-composite", "optimizer": "adam", "lr": 0.005, "beta2": 0.999, "seed_label": 810001, "walkers": 2048}, "output_root": str(tmp_path / "output")}
    def write_inputs() -> None:
        manifest_path.write_text(json.dumps(manifest)); receipt.write_text(json.dumps({"runtime_kind": job.RUNTIME_KIND, "target_sha": manifest["target_sha"], "overlay": {key: profile[key] for key in ("overlay_python", "overlay_root", "facility_site_packages")}}))
        run_env.write_text(f"RUN_MANIFEST={manifest_path}\nRUN_PYTHON={sys.executable}\nRUN_JOB_PY={harness / 'gpu_job.py'}\n")
        (tmp_path / "nodes").write_text("nid001.fqdn\n")
    def configure() -> None:
        write_inputs(); monkeypatch.setenv("RUN_ENV_PATH", str(run_env)); monkeypatch.setenv("PBS_JOBID", "job"); monkeypatch.setenv("PBS_NODEFILE", str(tmp_path / "nodes")); monkeypatch.setenv("LD_LIBRARY_PATH", ":".join((str(tmp_path), str(tmp_path), str(tmp_path), "/opt/cray/libfabric/lib")))
        monkeypatch.setattr(job, "hostname", lambda: "nid001"); monkeypatch.setattr(job, "nodefile_entries", lambda: ["nid001.fqdn"]); monkeypatch.setattr(job, "git_head", lambda repo: manifest["target_sha"]); monkeypatch.setattr(job, "appledoubles", lambda repo: []); monkeypatch.setattr(job, "overlay_snapshot", lambda profile: {"tree": "same"}); monkeypatch.setattr(job, "torch_probe", lambda python, env: {"torch": "2.8.0", "cuda": "12.9", "file": str(tmp_path / "site-packages" / "torch.py"), "available": True, "count": 1, "name": "A100", "op": 2.0, "visible": "0", "ld": env["LD_LIBRARY_PATH"]}); monkeypatch.setattr(job, "run_route", lambda *args: 0); monkeypatch.setattr(job, "run_qualification", lambda *args: 0)
    configure(); assert job.main() == 0
    controls = [("60", lambda: receipt.unlink(), 60), ("61", lambda: receipt.write_text("{}"), 61), ("64", lambda: (target / job.CANONICAL_STAGE).unlink(), 64), ("79", lambda: manifest.update(required_ancestors=["0" * 40]), 79), ("81", lambda: manifest.update(copied_file_sha256={"gpu_job.py": "0" * 64}), 81), ("82", lambda: manifest["runtime_profile"].update(libshim=str(tmp_path / "missing")), 82), ("84", lambda: manifest.update(runtime_kind="wrong"), 84), ("85", lambda: manifest.update(nodes=2), 85)]
    for name, mutate, expected in controls:
        (target / job.CANONICAL_STAGE).parent.mkdir(parents=True, exist_ok=True)
        (target / job.CANONICAL_STAGE).write_text("# fixture route\n")
        profile["libshim"] = str(tmp_path)
        manifest.update(runtime_kind=job.RUNTIME_KIND, target_sha=target_sha, required_ancestors=[], nodes=1, copied_file_sha256={}, job_helper=str(harness / "gpu_job.py"), staging_receipt=str(receipt), runtime_profile=profile); write_inputs(); mutate(); manifest_path.write_text(json.dumps(manifest)); assert job.main() == expected, name
    configure(); changed = iter(({"tree": "same"}, {"tree": "changed"})); monkeypatch.setattr(job, "overlay_snapshot", lambda profile: next(changed)); assert job.main() == 83


def test_stage_real_clone_and_qsub_fakes(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True); subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True); subprocess.run(["git", "-C", str(source), "config", "user.name", "test"], check=True)
    (source / "file").write_text("target\n"); subprocess.run(["git", "-C", str(source), "add", "file"], check=True); subprocess.run(["git", "-C", str(source), "commit", "-q", "-m", "target"], check=True)
    target_sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    remote = tmp_path / "remote.git"; subprocess.run(["git", "clone", "-q", "--bare", str(source), str(remote)], check=True)
    clean_harness = tmp_path / "harness"; clean_harness.mkdir(); subprocess.run(["git", "init", "-q", str(clean_harness)], check=True); subprocess.run(["git", "-C", str(clean_harness), "config", "user.email", "test@example.invalid"], check=True); subprocess.run(["git", "-C", str(clean_harness), "config", "user.name", "test"], check=True)
    for name in stage.STORED_FILES: (clean_harness / name).write_text("fixture\n")
    subprocess.run(["git", "-C", str(clean_harness), "add", "."], check=True); subprocess.run(["git", "-C", str(clean_harness), "commit", "-q", "-m", "harness"], check=True)
    overlay = tmp_path / "overlay"; site = overlay / "site-packages"; site.mkdir(parents=True); (overlay / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    profile = tmp_path / "profile.json"; profile.write_text(json.dumps({"runtime_kind": stage.RUNTIME_KIND, "overlay_python": sys.executable, "overlay_root": str(overlay), "facility_site_packages": str(site), "libshim": str(tmp_path), "cuda13_lib": str(tmp_path), "cuda129_lib": str(tmp_path), "provenance": "test"}))
    qsub_calls = []
    original_run = stage._run
    monkeypatch.setattr(stage, "_harness_repo", lambda: clean_harness); monkeypatch.setattr(stage, "_qstat", lambda: "")
    def fake_run(args, **kwargs):
        if args and args[0] == "qsub":
            qsub_calls.append(args); return "123.polaris"
        return original_run(args, **kwargs)
    monkeypatch.setattr(stage, "_run", fake_run)
    args = type("Args", (), {"target_sha": target_sha, "cell_id": "cell", "context": "O-composite", "optimizer": "adam", "lr": 0.005, "beta2": 0.999, "seed_label": 810001, "walkers": 2048, "updates": 2000, "checkpoint_every": 1000, "stage": "Q", "nodes": 1, "queue": "debug", "walltime": "00:30:00", "account": "HetRxnEnergy", "runtime_profile": str(profile), "evidence_root": str(tmp_path / "runs"), "required_ancestor": [], "remote": str(remote)})()
    assert stage.stage(args) == 0
    assert len(qsub_calls) == 1 and qsub_calls[0][1] == "-V" and qsub_calls[0].count("-l") == 3
    assert qsub_calls[0][qsub_calls[0].index("-l") + 1] == "select=1:system=polaris"
    manifest = json.loads((tmp_path / "runs" / "cell" / "manifest.json").read_text())
    assert Path(manifest["job_helper"]).name == "gpu_job.py"

    def isolated_profile(name: str, **changes: str) -> Path:
        value = json.loads(profile.read_text())
        value.update(changes)
        path = tmp_path / f"{name}.json"; path.write_text(json.dumps(value)); return path

    def isolated_args(name: str, runtime_profile: Path) -> object:
        values = vars(args).copy(); values.update(cell_id=name, runtime_profile=str(runtime_profile), evidence_root=str(tmp_path / f"runs-{name}"))
        return type("Args", (), values)()

    monkeypatch.setattr(stage, "_qstat", lambda: "busy")
    for name, runtime_profile in (("qstat", profile), ("missing-overlay", isolated_profile("missing-overlay", overlay_root=str(tmp_path / "missing"))), ("wrong-kind", isolated_profile("wrong-kind", runtime_kind="other")), ("unfilled", isolated_profile("unfilled", overlay_python=""))):
        before = len(qsub_calls)
        try:
            stage.stage(isolated_args(name, runtime_profile))
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f"stage accepted control {name}")
        assert len(qsub_calls) == before, name
