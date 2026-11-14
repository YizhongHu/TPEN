"""Executable launch-guard tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.atomistic.stage1.launch import command_for_row, render, row_overrides
from experiments.atomistic.stage1.protocol import ProtocolError, validate_gateway_binding


ROW = {"kind": "train", "config": "configs/he/train.yaml", "row_id": "he-train-0",
       "seed": 123, "updates": 50000, "walkers": 4096,
       "source_sha": "c" * 40, "config_sha256": "b" * 64,
       "results_root": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/he/train"}


def _run(script: str, tmp_path: Path, **slurm: str) -> subprocess.CompletedProcess[str]:
    """Execute a rendered script with an isolated, test-owned environment."""

    env = {"PATH": __import__("os").environ["PATH"], "USER": "tester", **slurm}
    return subprocess.run(["bash"], input=script, text=True, capture_output=True, env=env)


def test_render_contains_fail_closed_identity_and_cleanup_guards(tmp_path: Path) -> None:
    script = render(row=ROW, driver="driver.py", python="/usr/bin/python3",
                    head_sha="c" * 40,
                    results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
                    scratch_base=str(tmp_path))
    assert "STAGE1_GUARD missing SLURM_JOB_ID" in script
    assert '"$SLURM_JOB_PARTITION" != "$REQUESTED_PARTITION"' in script
    assert 'JOB_ROOT="$SCRATCH_BASE/$USER/$SLURM_JOB_ID"' in script
    assert "STAGE1_GUARD results root outside" in script
    assert '[[ "$JOB_ROOT" == "$SCRATCH_BASE/$USER/$SLURM_JOB_ID" ]]' in script
    assert "trap 'rm -rf -- \"$JOB_ROOT\"' EXIT" in script
    assert "exec " not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_render_fails_for_unapproved_partition_or_results_root() -> None:
    with pytest.raises(ValueError, match="partition"):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
               partition="gpu", scratch_base="/tmp/stage1")
    with pytest.raises(ValueError, match="Netscratch"):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root="/tmp/stage1", scratch_base="/tmp/stage1")


def test_partition_guard_is_exercised_by_fake_environment(tmp_path: Path) -> None:
    script = render(row=ROW, driver="/bin/true", python="/usr/bin/true",
                    head_sha="c" * 40,
                    results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
                    scratch_base=str(tmp_path))
    missing_job = _run(script, tmp_path)
    assert missing_job.returncode == 2
    assert "STAGE1_GUARD missing SLURM_JOB_ID" in missing_job.stderr
    mismatch = _run(script, tmp_path, SLURM_JOB_ID="123",
                    SLURM_JOB_PARTITION="gpu_test")
    assert mismatch.returncode == 2
    assert "STAGE1_GUARD partition delivered=gpu_test requested=test" in mismatch.stderr


def _rendered_script(scratch_base: str = "/tmp/stage1") -> str:
    return render(row=ROW, driver="/bin/true", python="/usr/bin/true",
                  head_sha="c" * 40,
                  results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
                  scratch_base=scratch_base)


def test_guard_fails_without_slurm_job_id(tmp_path: Path) -> None:
    result = _run(_rendered_script(str(tmp_path)), tmp_path)
    assert result.returncode == 2
    assert "STAGE1_GUARD missing SLURM_JOB_ID" in result.stderr


def test_rendered_prefix_guard_executes_under_fake_environment(tmp_path: Path) -> None:
    script = _rendered_script(str(tmp_path)).replace(
        "RESULTS_ROOT='/n/netscratch/kozinsky_lab/Lab/rhu/stage1'",
        "RESULTS_ROOT='/outside/approved/root'",
    )
    assert script != _rendered_script(str(tmp_path))
    result = _run(script, tmp_path, SLURM_JOB_ID="123", SLURM_JOB_PARTITION="test")
    assert result.returncode == 2
    assert "STAGE1_GUARD results root outside /n/netscratch/kozinsky_lab/Lab/rhu/" in result.stderr


def test_rendered_exit_trap_removes_job_root(tmp_path: Path) -> None:
    marker = tmp_path / "tester" / "123"
    script = _rendered_script(str(tmp_path))
    result = _run(script, tmp_path, SLURM_JOB_ID="123", SLURM_JOB_PARTITION="test")
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_rendered_script_preserves_preexisting_job_root(tmp_path: Path) -> None:
    root = tmp_path / "tester" / "123"
    root.mkdir(parents=True)
    sentinel = root / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    result = _run(_rendered_script(str(tmp_path)), tmp_path, SLURM_JOB_ID="123", SLURM_JOB_PARTITION="test")
    assert result.returncode == 2
    assert "STAGE1_GUARD job root already exists" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_render_rejects_shell_metacharacters_and_root_override(tmp_path: Path) -> None:
    for unsafe in ("$(touch marker)", "`touch marker`", "a'b", "a\nb"):
        with pytest.raises(ValueError):
            render(row={**ROW, "config": unsafe}, driver="driver.py",
                   python="/usr/bin/python3", head_sha="c" * 40,
                   results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
                   scratch_base=str(tmp_path))
    with pytest.raises(ValueError):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1", scratch_base="/")


def test_render_ignores_hostile_scratch_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE1_SCRATCH_BASE", "$(touch marker)")
    script = render(row=ROW, driver="/bin/true", python="/usr/bin/true",
                    head_sha="c" * 40,
                    results_root="/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
                    scratch_base=str(tmp_path))
    assert "STAGE1_SCRATCH_BASE" not in script
    assert f"SCRATCH_BASE='{tmp_path}'" in script


def test_forced_template_metacharacter_fails_without_side_effect(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    original = "RESULTS_ROOT='/n/netscratch/kozinsky_lab/Lab/rhu/stage1'"
    for hostile in (f"$(touch {marker})", f"`touch {marker}`"):
        script = _rendered_script(str(tmp_path)).replace(
            original, f"RESULTS_ROOT='{hostile}'",
        )
        assert script != _rendered_script(str(tmp_path))
        result = _run(script, tmp_path, SLURM_JOB_ID="123", SLURM_JOB_PARTITION="test")
        assert result.returncode == 2
        assert "STAGE1_GUARD results root outside" in result.stderr
        assert not marker.exists()


def test_gateway_rejects_omitted_or_altered_binding_before_run() -> None:
    from experiments.atomistic.stage1.launch import _binding_token, row_overrides
    token = _binding_token(ROW)
    overrides = row_overrides(ROW)
    validate_gateway_binding(binding=token, config=ROW["config"], overrides=overrides, kind="train")
    with pytest.raises(ProtocolError):
        validate_gateway_binding(binding=token, config=ROW["config"], overrides=overrides[:-1], kind="train")
    with pytest.raises(ProtocolError):
        validate_gateway_binding(binding=token, config=ROW["config"], overrides=[*overrides[:-1], "trainer.max_steps=1"], kind="train")
    with pytest.raises(ProtocolError):
        validate_gateway_binding(binding=token, config="configs/he/other.yaml",
                                 overrides=overrides, kind="train")
    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_path": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/he/train/model.pt",
        "checkpoint_sha256": "d" * 64,
        "resolved_config_identity": {"schema": "resolved-config-identity/v1", "row_id": "he-train-0"},
    }
    eval_token = _binding_token(eval_row)
    eval_overrides = row_overrides(eval_row)
    identity = _binding_token({"resolved_config_identity": eval_row["resolved_config_identity"]})
    validate_gateway_binding(binding=eval_token, config=eval_row["config"], overrides=eval_overrides, kind="eval",
                             expected_identity=identity, expected_checkpoint_sha256="d" * 64)
    with pytest.raises(ProtocolError):
        validate_gateway_binding(binding=eval_token, config=eval_row["config"], overrides=eval_overrides, kind="eval",
                                 expected_identity=None, expected_checkpoint_sha256="d" * 64)


@pytest.mark.parametrize("key", ["load.mode", "load.path", "runtime.seed"])
def test_gateway_rejects_conflicting_duplicate_override(key: str) -> None:
    from experiments.atomistic.stage1.launch import _binding_token, row_overrides

    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_path": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/he/train/model.pt",
        "checkpoint_sha256": "d" * 64,
        "resolved_config_identity": {"schema": "resolved-config-identity/v1", "row_id": "he-train-0"},
    }
    overrides = row_overrides(eval_row)
    identity = _binding_token({"resolved_config_identity": eval_row["resolved_config_identity"]})
    with pytest.raises(ProtocolError):
        validate_gateway_binding(
            binding=_binding_token(eval_row), config=eval_row["config"],
            overrides=[*overrides, f"{key}=conflict"], kind="eval",
            expected_identity=identity, expected_checkpoint_sha256="d" * 64,
        )


@pytest.mark.parametrize("override", ["++load.mode=full", "+runtime.seed=1", "~load.strict"])
def test_gateway_rejects_prefixed_override(override: str) -> None:
    from experiments.atomistic.stage1.launch import _binding_token, row_overrides

    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_path": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/he/train/model.pt",
        "checkpoint_sha256": "d" * 64,
        "resolved_config_identity": {"schema": "resolved-config-identity/v1", "row_id": "he-train-0"},
    }
    identity = _binding_token({"resolved_config_identity": eval_row["resolved_config_identity"]})
    with pytest.raises(ProtocolError, match="prefixed gateway overrides"):
        validate_gateway_binding(
            binding=_binding_token(eval_row), config=eval_row["config"],
            overrides=[*row_overrides(eval_row), override], kind="eval",
            expected_identity=identity, expected_checkpoint_sha256="d" * 64,
        )


def test_command_binds_every_kind() -> None:
    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_path": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/he/train/model.pt",
        "checkpoint_sha256": "d" * 64,
        "resolved_config_identity": {"schema": "resolved-config-identity/v1", "row_id": "he-train-0"},
    }
    for row in ({**ROW, "kind": "smoke"}, ROW, eval_row):
        command = command_for_row(row, python="/usr/bin/python3", driver="driver.py")
        assert f"runtime.seed={row['seed']}" in command
        assert str(row["walkers"]) in command
    eval_command = command_for_row(eval_row, python="/usr/bin/python3", driver="eval.py")
    assert "load.path=" in eval_command
    assert "load.mode=model_only" in eval_command
