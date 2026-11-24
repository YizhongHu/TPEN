"""Regression tests for cwd-independent Stage-1 gateway config resolution."""

from __future__ import annotations

import hashlib
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import command_for_row, render, row_overrides
from experiments.atomistic.stage1.protocol import (
    HOOKE_SCIENCE_PROFILE,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
    hooke_evaluation_selection,
    overlay_sha256,
    resolve_config_path,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = "/n/netscratch/kozinsky_lab/Lab/rhu/stage1"
BASE_ROW = {
    "config": "configs/he/train.yaml",
    "seed": 123,
    "updates": 5,
    "walkers": 2,
    "source_sha": "c" * 40,
    "config_sha256": "b" * 64,
    "results_root": RESULTS_ROOT,
    "experiment_name": "stage1",
    "sector": "he",
}


@pytest.mark.parametrize("script", ["plan.py", "launch.py"])
def test_stage1_planning_scripts_import_by_path_outside_checkout(
    script: str, tmp_path: Path,
) -> None:
    """Both planning entrypoints bootstrap imports before loading protocol."""

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments/atomistic/stage1" / script), "--help"],
        cwd=unrelated,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def _row(kind: str, checkpoint_root: Path | None = None) -> dict[str, object]:
    """Return one small manifest row for a real rendered driver command."""

    row = {
        **BASE_ROW,
        "kind": kind,
        "row_id": f"he-{kind}-0",
        "run_id": f"{kind}-lineage-0",
    }
    if kind == "smoke":
        row.update(updates=3, walkers=2, checkpoint_steps=[3])
    elif kind == "eval":
        checkpoint_dir = (
            checkpoint_root / "checkpoint" if checkpoint_root is not None
            else Path(f"{RESULTS_ROOT}/stage1/he/train-lineage-0/checkpoints/step_000005")
        )
        checkpoint_file = checkpoint_dir / "model.pt"
        if checkpoint_root is not None:
            checkpoint_dir.mkdir(parents=True)
            payload = b"matching checkpoint bytes"
            checkpoint_file.write_bytes(payload)
            checkpoint_digest = hashlib.sha256(payload).hexdigest()
        else:
            checkpoint_digest = "d" * 64
        row.update(
            config="configs/he/eval.yaml",
            draws=2, checkpoint_dir=str(checkpoint_dir),
            checkpoint_file=str(checkpoint_file), checkpoint_sha256=checkpoint_digest,
        )
    else:
        row["checkpoint_steps"] = [5]
    return row


def _production_train_row(system: str, root: Path) -> dict[str, object]:
    """Build one production-shaped train row with an isolated output root."""

    config_root = REPO_ROOT / "experiments/atomistic/stage1/configs"
    if system in {"hooke1", "hooke2", "qho2e"}:
        overlay = "configs/overlays/hooke-diagnostics-v1.yaml"
        selection = hooke_evaluation_selection(system)[0]
        profile = HOOKE_SCIENCE_PROFILE
    else:
        overlay = "configs/overlays/diagnostics-v1.yaml"
        selection = None
        profile = None
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=config_root, systems=(system,)),
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1,
                              STAGE1_PROTOCOL.retention_cadence or 1)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(root),
        systems=(system,),
        phase=1,
        science_profile=profile,
        evaluation_selection=selection,
        overlay=overlay,
        overlay_sha256=overlay_sha256(overlay),
    )
    return next(row for row in rows if row["kind"] == "train")


def _production_eval_row(system: str, root: Path) -> dict[str, object]:
    """Adapt a production train identity into one isolated eval row."""

    row = dict(_production_train_row(system, root))
    checkpoint_dir = (root / "checkpoint").resolve()
    checkpoint_dir.mkdir(parents=True)
    payload = b"matching checkpoint bytes"
    checkpoint_file = checkpoint_dir / "model.pt"
    checkpoint_file.write_bytes(payload)
    row.update(
        kind="eval",
        config=f"configs/{system}/eval.yaml",
        row_id=f"{system}-eval-0",
        run_id=f"eval-{system}-0",
        draws=STAGE1_PROTOCOL.eval_draws,
        walkers=STAGE1_PROTOCOL.eval_walkers,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_file=str(checkpoint_file),
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
    )
    for key in ("checkpoint_steps", "overlay", "overlay_sha256"):
        row.pop(key, None)
    return row


def _run_rendered_row(
    row: dict[str, object], *, monkeypatch: pytest.MonkeyPatch,
    results_env: str | None,
) -> object:
    """Drive command_for_row's actual argv through the selected gateway main."""

    driver = (
        "experiments/atomistic/stage1/eval.py"
        if row["kind"] == "eval"
        else "experiments/atomistic/stage1/train.py"
    )
    argv = shlex.split(command_for_row(row, python=sys.executable, driver=driver))
    module = stage1_eval if row["kind"] == "eval" else stage1_train
    captured: list[object] = []
    monkeypatch.setattr(
        module, "run_from_config",
        lambda cfg, **_kwargs: (captured.append(cfg) or 0),
    )
    original_merge = module.OmegaConf.merge

    def spy_merge(*configs: object) -> object:
        """Require the row root to be bound before any composition merge."""

        assert configs
        assert module.OmegaConf.select(configs[0], "run.root") == str(row["results_root"])
        return original_merge(*configs)

    monkeypatch.setattr(module.OmegaConf, "merge", spy_merge)
    if results_env is None:
        monkeypatch.delenv("TPEN_STAGE1_RESULTS_ROOT", raising=False)
    else:
        monkeypatch.setenv("TPEN_STAGE1_RESULTS_ROOT", results_env)
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", [argv[1], *argv[2:]])
    assert module.main() == 0
    assert len(captured) == 1
    return captured[0]


def _contradictory_root_overrides(row: dict[str, object]) -> list[str]:
    """Replace the compatibility root override with a contradictory path."""

    return [
        "run.root=/contradictory/compatibility-root"
        if override.startswith("run.root=") else override
        for override in row_overrides(row)
    ]


def _rendered_argv(
    row: dict[str, object], driver: str, tmp_path: Path, checkout_root: Path,
) -> list[str]:
    """Extract the command that launch.render places in the sbatch script."""

    script = render(
        row=row,
        driver=driver,
        python=sys.executable,
        checkout_root=str(checkout_root),
        head_sha=str(row["source_sha"]),
        results_root=RESULTS_ROOT,
        scratch_base=str(tmp_path / "scratch"),
    )
    return shlex.split(script.splitlines()[-1])


def _make_probe_checkout(source: Path, destination: Path, marker: str) -> Path:
    """Make a light checkout containing the real drivers and protocol imports."""

    for relative in (
        Path("experiments/atomistic/stage1"),
        Path("experiments/baselines"),
        Path("experiments/toolkit"),
    ):
        shutil.copytree(source / relative, destination / relative)
    (destination / "tpen").mkdir(parents=True)
    (destination / "tpen/__init__.py").write_text("")
    (destination / "tpen/run.py").write_text(
        "def load_config(*args, **kwargs):\n    return None\n"
        "def run_from_config(*args, **kwargs):\n    return 0\n"
    )
    protocol = destination / "experiments/atomistic/stage1/protocol.py"
    protocol.write_text(
        protocol.read_text()
        + f"\nprint('STAGE1_PROTOCOL_ORIGIN={marker}:' + __file__)\n"
    )
    return destination


@pytest.mark.parametrize("system", ["hooke1", "hooke2", "qho2e", "he", "h2"])
@pytest.mark.parametrize("kind", ["train", "eval"])
def test_every_stage1_output_slot_is_mandatory(system: str, kind: str) -> None:
    """Every train/eval recipe requires the row builder to bind its root."""

    config = REPO_ROOT / "experiments/atomistic/stage1/configs" / system / f"{kind}.yaml"
    assert 'run: {root: "???"' in config.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("kind", "driver"),
    [
        ("smoke", "experiments/atomistic/stage1/train.py"),
        ("train", "experiments/atomistic/stage1/train.py"),
        ("eval", "experiments/atomistic/stage1/eval.py"),
    ],
)
def test_rendered_driver_imports_experiments_outside_checkout(
    kind: str, driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected execution checkout owns the driver and protocol import."""

    checkout_a = _make_probe_checkout(REPO_ROOT, tmp_path / "checkout-a", "A")
    checkout_b = _make_probe_checkout(REPO_ROOT, tmp_path / "checkout-b", "B")
    monkeypatch.chdir(checkout_a)
    monkeypatch.setattr(sys, "path", [str(checkout_a), *sys.path])
    argv = _rendered_argv(_row(kind), driver, tmp_path, checkout_b)
    result = subprocess.run(
        [*argv, "--help"],
        cwd=checkout_a,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert argv[1] == str(checkout_b / driver)
    assert f"STAGE1_PROTOCOL_ORIGIN=B:{checkout_b}" in result.stdout

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    control_argv = _rendered_argv(_row(kind), driver, tmp_path, checkout_a)
    control = subprocess.run(
        [*control_argv, "--help"],
        cwd=unrelated,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert control.returncode == 0, control.stderr
    assert control_argv[1] == str(checkout_a / driver)
    assert f"STAGE1_PROTOCOL_ORIGIN=A:{checkout_a}" in control.stdout


@pytest.mark.parametrize(
    ("kind", "driver"),
    [
        ("smoke", "experiments/atomistic/stage1/train.py"),
        ("train", "experiments/atomistic/stage1/train.py"),
        ("eval", "experiments/atomistic/stage1/eval.py"),
    ],
)
@pytest.mark.parametrize("submission_cwd", ["repo", "unrelated"])
@pytest.mark.parametrize("results_env", [None, "/contradictory/stage1-root"])
def test_rendered_driver_resolves_config_from_any_submission_cwd(
    kind: str,
    driver: str,
    submission_cwd: str,
    results_env: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rendered command reaches load_config with its bounded row budget."""

    row = _row(kind, checkpoint_root=tmp_path)
    argv = _rendered_argv(row, driver, tmp_path, REPO_ROOT)
    assert "--binding" in argv
    assert "--config" not in argv

    module = stage1_eval if kind == "eval" else stage1_train
    validation_paths: list[str] = []
    load_paths: list[str] = []
    run_calls: list[tuple[object, dict[str, object]]] = []

    original_validate = module.validate_gateway_binding

    def spy_validate(*args: object, **kwargs: object) -> object:
        validation_paths.append(str(resolve_config_path(str(row["config"]))))
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(module, "validate_gateway_binding", spy_validate)

    import tpen.run

    original_load_config = tpen.run.load_config

    def spy_load_config(config: str, overrides: list[str] | None = None) -> object:
        load_paths.append(config)
        return original_load_config(config, overrides)

    monkeypatch.setattr(tpen.run, "load_config", spy_load_config)
    monkeypatch.setattr(
        module,
        "run_from_config",
        lambda cfg, **kwargs: (run_calls.append((cfg, kwargs)) or 0),
    )
    if results_env is None:
        monkeypatch.delenv("TPEN_STAGE1_RESULTS_ROOT", raising=False)
    else:
        monkeypatch.setenv("TPEN_STAGE1_RESULTS_ROOT", results_env)
    monkeypatch.chdir(REPO_ROOT if submission_cwd == "repo" else tmp_path)
    monkeypatch.setattr(sys, "argv", [argv[1], *argv[2:]])

    assert module.main() == 0

    expected_path = str(resolve_config_path(str(row["config"])))
    assert validation_paths == [expected_path]
    assert load_paths == [expected_path]
    assert run_calls[0][1]["config_path"] == expected_path
    cfg = run_calls[0][0]
    if kind == "eval":
        assert cfg.evaluation_sampler.n_walkers == row["walkers"]
        assert cfg.evaluation_tasks.mcmc_energy.generator.n_draws == row["draws"]
    else:
        assert cfg.sampler.n_walkers == row["walkers"]
        assert cfg.trainer.max_steps == row["updates"]
        assert list(cfg.checkpoint_steps) == list(row["checkpoint_steps"])
    assert cfg.run.root == row["results_root"]


@pytest.mark.parametrize("system", ["hooke1", "hooke2", "qho2e", "he", "h2"])
@pytest.mark.parametrize("results_env", [None, "/contradictory/stage1-root"])
def test_production_train_rows_bind_output_before_resolution(
    system: str,
    results_env: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendered production train rows use their own root regardless of env state."""

    root = tmp_path / system
    row = _production_train_row(system, root)
    cfg = _run_rendered_row(row, monkeypatch=monkeypatch, results_env=results_env)
    assert cfg.run.root == str(root)


@pytest.mark.parametrize("system", ["hooke1", "hooke2", "qho2e", "he", "h2"])
@pytest.mark.parametrize("results_env", [None, "/contradictory/stage1-root"])
def test_production_eval_rows_bind_output_before_resolution(
    system: str,
    results_env: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendered production eval rows use their own root regardless of env state."""

    root = tmp_path / system
    row = _production_eval_row(system, root)
    cfg = _run_rendered_row(row, monkeypatch=monkeypatch, results_env=results_env)
    assert cfg.run.root == str(root)


@pytest.mark.parametrize("system", ["hooke1", "hooke2", "qho2e", "he"])
def test_train_builder_rebinds_output_after_compatibility_override(
    system: str, tmp_path: Path,
) -> None:
    """A legacy override cannot redirect the final train output root."""

    root = tmp_path / system
    row = _production_train_row(system, root)
    cfg = stage1_train.build_train_config(
        row, overrides=_contradictory_root_overrides(row),
    )
    assert cfg.run.root == str(root)


@pytest.mark.parametrize("system", ["hooke1", "hooke2", "qho2e", "he"])
def test_eval_builder_rebinds_output_after_compatibility_override(
    system: str, tmp_path: Path,
) -> None:
    """A legacy override cannot redirect the final eval output root."""

    root = tmp_path / system
    row = _production_eval_row(system, root)
    cfg = stage1_eval.build_eval_config(
        row, overrides=_contradictory_root_overrides(row),
    )
    assert cfg.run.root == str(root)


@pytest.mark.parametrize("submission_cwd", ["repo", "unrelated"])
def test_rendered_driver_refuses_missing_config_from_any_submission_cwd(
    submission_cwd: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing row config is rejected after the same path resolution step."""

    row = {**_row("smoke"), "config": "configs/he/does-not-exist.yaml"}
    argv = _rendered_argv(row, "experiments/atomistic/stage1/train.py", tmp_path, REPO_ROOT)
    monkeypatch.chdir(REPO_ROOT if submission_cwd == "repo" else tmp_path)
    monkeypatch.setattr(sys, "argv", [argv[1], *argv[2:]])

    with pytest.raises(FileNotFoundError, match="config path does not exist"):
        stage1_train.main()
