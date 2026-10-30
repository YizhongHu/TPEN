"""Focused controls for the tracked-lock Polaris GPU qualification route."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage = _load("gpu_stage")
job = _load("gpu_job")


def test_cu126_lock_derivation_and_missing_entry_control() -> None:
    lock = stage.tomllib.loads((ROOT.parents[2] / "uv.lock").read_text(encoding="utf-8"))
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
    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert script.index("export CUDA_VISIBLE_DEVICES=0") < script.index("exec ")
    assert 'uv sync --offline --extra "$RUN_EXTRA" --locked' in script
    assert "LD_LIBRARY_PATH" not in script
    source = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.iterdir() if path.is_file())
    assert all(token not in source for token in ("overlay_root", "overlay_python", "libshim", "facility_site_packages"))


def test_prime_and_dry_sync_uses_profile_extra(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    paths = stage.gpu_layout(tmp_path / "run")
    paths["run_root"].mkdir()
    profile = {"runtime_kind": stage.RUNTIME_KIND, "extra": "cu126"}
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
        assert call[call.index("--extra") + 1] == "cu126"
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
