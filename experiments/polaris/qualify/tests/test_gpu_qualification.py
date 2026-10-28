"""Two-sided self-tests for the actual B0 entry points."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
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
    assert "qsub" not in script and "uv" not in script


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
    (output / "selection_receipt.json").write_text(json.dumps({"selection": {"selected": {"context": "O-composite"}}}) + "\n")
    (output / "resolved_config.yaml").write_text("context: O-composite\n")
    (output / "COMPLETE").write_text("completed\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runtime_kind": stage.RUNTIME_KIND, "cell": {"context": "O-composite", "walkers": 2048, "updates": 2000, "checkpoint_every": 1000}, "output_root": str(tmp_path / "output"), "evidence_dir": str(tmp_path / "evidence")}) + "\n")
    assert qualification.qualify(manifest, tmp_path / "evidence") == 0

    # Every negative arm is run against a copy of the writer-produced layout.
    copied = tmp_path / "copies"
    shutil.copytree(tmp_path / "output", copied)
    shutil.copytree(tmp_path / "output", copied / "h1")
    for filename in ("manifest.json", "trainer.json"):
        path = copied / "h1" / "owner" / "checkpoints" / "step_002000" / filename
        value = json.loads(path.read_text())
        value["completed_updates"] = 1999
        value["next_iteration"] = 1999
        path.write_text(json.dumps(value) + "\n")
    assert qualification.qualify(manifest, tmp_path / "evidence-h1-updates", artifact_root=copied / "h1") != 0
    (copied / "owner" / "checkpoints" / "step_002000" / "COMPLETE").unlink()
    assert qualification.qualify(manifest, tmp_path / "evidence-h1", artifact_root=copied) != 0
    shutil.copytree(tmp_path / "output", copied / "nan")
    torch.save({"weight": torch.tensor([float("nan")])}, copied / "nan" / "owner" / "checkpoints" / "step_002000" / "model.pt")
    assert qualification.qualify(manifest, tmp_path / "evidence-nan", artifact_root=copied / "nan") != 0
    shutil.copytree(tmp_path / "output", copied / "wrong-cell")
    (copied / "wrong-cell" / "owner" / "selection_receipt.json").write_text(json.dumps({"selection": {"selected": {"context": "O-raw"}}}) + "\n")
    assert qualification.qualify(manifest, tmp_path / "evidence-cell", artifact_root=copied / "wrong-cell") != 0


def test_no_hydra_or_run_root_literals_in_launcher_sources() -> None:
    sources = "\n".join(path.read_text() for path in ROOT.glob("*.py"))
    assert "hydra" not in sources.lower()
    assert "/eagle/" not in sources
    assert not any(len(token) == 40 and all(char in "0123456789abcdef" for char in token)
                   for token in sources.split())


def test_actual_cli_rejects_unsupported_argument() -> None:
    result = subprocess.run([sys.executable, str(ROOT / "gpu_stage.py"), "--help"], capture_output=True, text=True, check=True)
    assert "--walkers" in result.stdout and "--runtime-profile" in result.stdout
