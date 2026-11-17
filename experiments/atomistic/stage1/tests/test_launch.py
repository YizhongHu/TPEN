"""Executable launch-guard tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from experiments.atomistic.stage1.launch import (
    _binding_token,
    command_for_row,
    render as _render,
    row_overrides,
)
from experiments.atomistic.stage1.protocol import ProtocolError, validate_gateway_binding
from experiments.toolkit.run_paths import resolve_run_dir


ROW = {"kind": "train", "config": "configs/he/train.yaml", "row_id": "he-train-0",
       "seed": 123, "updates": 50000, "walkers": 4096,
       "source_sha": "c" * 40, "config_sha256": "b" * 64,
       "results_root": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1",
       "experiment_name": "stage1", "sector": "he", "run_id": "train-lineage-0"}

REPO_ROOT = Path(__file__).resolve().parents[4]


def render(*args: Any, **kwargs: Any) -> str:
    """Keep legacy test calls bound to this checkout explicitly."""

    kwargs.setdefault("checkout_root", str(REPO_ROOT))
    return _render(*args, **kwargs)


def _run(script: str, tmp_path: Path, **slurm: str) -> subprocess.CompletedProcess[str]:
    """Execute a rendered script with an isolated, test-owned environment."""

    env = {"PATH": __import__("os").environ["PATH"], "USER": "tester", **slurm}
    return subprocess.run(["bash"], input=script, text=True, capture_output=True, env=env)


def _fake_gpu_python(fake_bin: Path, device_count: str = "1") -> Path:
    """Create the rendered interpreter seam used by the in-job CUDA probe."""

    python = fake_bin / "python"
    python.write_text(
        f"#!/bin/sh\nprintf '%s\\n' '{device_count}'\n", encoding="utf-8",
    )
    python.chmod(0o750)
    return python


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
               partition="not-a-partition", scratch_base="/tmp/stage1")
    with pytest.raises(ValueError, match="Netscratch"):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root="/tmp/stage1", scratch_base="/tmp/stage1")


def test_gpu_render_requires_explicit_shape_and_cuda_binding() -> None:
    gpu_row = {**ROW, "device": "cuda"}
    with pytest.raises(ValueError, match="cpus_per_task"):
        render(row=gpu_row, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root=ROW["results_root"], partition="seas_gpu")
    script = render(
        row=gpu_row, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
        results_root=ROW["results_root"], partition="seas_gpu,kozinsky_gpu",
        cpus_per_task=8, mem="64G", time="08:00:00", scratch_base="/tmp/stage1",
    )
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --constraint=a100" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=08:00:00" in script
    assert "runtime.device=cuda" in script
    assert "expected exactly one visible GPU" in script


def test_gpu_render_rejects_cpu_binding_and_invalid_resources() -> None:
    with pytest.raises(ValueError, match="cuda"):
        render(row=ROW, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root=ROW["results_root"], partition="gpu",
               cpus_per_task=8, mem="64G", time="08:00:00")
    gpu_row = {**ROW, "device": "cuda"}
    for kwargs, message in [
        ({"cpus_per_task": 0, "mem": "64G", "time": "08:00:00"}, "cpus_per_task"),
        ({"cpus_per_task": 8, "mem": "64G", "time": "8:00:00"}, "time"),
        ({"cpus_per_task": 8, "mem": "2049G", "time": "08:00:00"}, "mem"),
    ]:
        with pytest.raises(ValueError, match=message):
            render(row=gpu_row, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
                   results_root=ROW["results_root"], partition="gpu", **kwargs)


@pytest.mark.parametrize("gpu_name", ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-PCIE-80GB"])
def test_gpu_partition_guard_accepts_one_requested_partition(
    tmp_path: Path, gpu_name: str,
) -> None:
    gpu_row = {**ROW, "device": "cuda"}
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in *mig.mode.current*) printf 'Disabled\\n';; "
        f"*) printf '{gpu_name}, 81920\\n';; esac\n",
        encoding="utf-8",
    )
    nvidia.chmod(0o750)
    python = _fake_gpu_python(fake_bin)
    script = render(
        row=gpu_row, driver="/bin/true", python=str(python), head_sha="c" * 40,
        results_root=ROW["results_root"], partition="seas_gpu,gpu",
        cpus_per_task=8, mem="64G", time="08:00:00", scratch_base=str(tmp_path),
    )
    result = _run(script, tmp_path, PATH=f"{fake_bin}:/usr/bin:/bin",
                   SLURM_JOB_ID="123", SLURM_JOB_PARTITION="gpu")
    assert result.returncode == 0, result.stderr
    assert "STAGE1_GUARD nvidia-smi=NVIDIA A100-SXM4-80GB, 81920" in result.stdout


@pytest.mark.parametrize("gpu_output", [
    "NVIDIA A100-SXM4-80GB MIG, 81920\\n",
    "NVIDIA A100-SXM4-40GB MIG 1g.5gb, 40960\\n",
    "NVIDIA A100-SXM4-40GB, 40960\\n",
    "NVIDIA A100-SXM4-80GB, 81920\\nNVIDIA A100-SXM4-80GB, 81920\\n",
])
def test_gpu_guard_rejects_mig_small_memory_and_multiple_gpus(
    tmp_path: Path, gpu_output: str,
) -> None:
    gpu_row = {**ROW, "device": "cuda"}
    script = render(
        row=gpu_row, driver="/bin/true", python="/usr/bin/true", head_sha="c" * 40,
        results_root=ROW["results_root"], partition="gpu",
        cpus_per_task=8, mem="64G", time="08:00:00", scratch_base=str(tmp_path),
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text(f"#!/bin/sh\nprintf '%b' '{gpu_output}'\n", encoding="utf-8")
    nvidia.chmod(0o750)
    python = _fake_gpu_python(fake_bin)
    script = render(
        row=gpu_row, driver="/bin/true", python=str(python), head_sha="c" * 40,
        results_root=ROW["results_root"], partition="gpu",
        cpus_per_task=8, mem="64G", time="08:00:00", scratch_base=str(tmp_path),
    )
    result = _run(script, tmp_path, PATH=f"{fake_bin}:/usr/bin:/bin",
                   SLURM_JOB_ID="123", SLURM_JOB_PARTITION="gpu")
    assert result.returncode == 2
    assert "STAGE1_GUARD" in result.stderr


def test_gpu_guard_rejects_delivered_partition_outside_requested_list_before_driver(
    tmp_path: Path,
) -> None:
    gpu_row = {**ROW, "device": "cuda"}
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = _fake_gpu_python(fake_bin)
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    nvidia.chmod(0o750)
    sentinel = tmp_path / "sentinel"
    driver = tmp_path / "driver"
    driver.write_text(f"#!/bin/sh\ntouch '{sentinel}'\n", encoding="utf-8")
    driver.chmod(0o750)
    script = render(
        row=gpu_row, driver=str(driver), python=str(python), head_sha="c" * 40,
        results_root=ROW["results_root"], partition="seas_gpu,gpu",
        cpus_per_task=8, mem="64G", time="08:00:00", scratch_base=str(tmp_path),
    )
    result = _run(script, tmp_path, PATH=f"{fake_bin}:/usr/bin:/bin",
                  SLURM_JOB_ID="123", SLURM_JOB_PARTITION="test")
    assert result.returncode == 2
    assert "partition delivered=test requested=seas_gpu,gpu" in result.stderr
    assert not sentinel.exists()


@pytest.mark.parametrize("device_count", ["0", "2"])
def test_gpu_guard_rejects_cuda_device_count_before_driver(
    tmp_path: Path, device_count: str,
) -> None:
    gpu_row = {**ROW, "device": "cuda"}
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = _fake_gpu_python(fake_bin, device_count)
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text(
        "#!/bin/sh\ncase \"$*\" in *mig.mode.current*) printf 'Disabled\\n';; "
        "*) printf 'NVIDIA A100-SXM4-80GB, 81920\\n';; esac\n",
        encoding="utf-8",
    )
    nvidia.chmod(0o750)
    script = render(
        row=gpu_row, driver="/bin/true", python=str(python), head_sha="c" * 40,
        results_root=ROW["results_root"], partition="gpu",
        cpus_per_task=8, mem="64G", time="08:00:00", scratch_base=str(tmp_path),
    )
    result = _run(script, tmp_path, PATH=f"{fake_bin}:/usr/bin:/bin",
                  SLURM_JOB_ID="123", SLURM_JOB_PARTITION="gpu")
    assert result.returncode == 2
    assert "exactly one CUDA device" in result.stderr


@pytest.mark.parametrize("mig_mode", ["Enabled", "[N/A]", "query-error"])
def test_gpu_guard_rejects_mig_mode_before_driver(tmp_path: Path, mig_mode: str) -> None:
    gpu_row = {**ROW, "device": "cuda"}
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = _fake_gpu_python(fake_bin)
    nvidia = fake_bin / "nvidia-smi"
    if mig_mode == "query-error":
        body = (
            "#!/bin/sh\ncase \"$*\" in *mig.mode.current*) exit 1;; "
            "*) printf 'NVIDIA A100-SXM4-80GB, 81920\\n';; esac\n"
        )
    else:
        body = (
            f"#!/bin/sh\ncase \"$*\" in *mig.mode.current*) printf '{mig_mode}\\n';; "
            "*) printf 'NVIDIA A100-SXM4-80GB, 81920\\n';; esac\n"
        )
    nvidia.write_text(body, encoding="utf-8")
    nvidia.chmod(0o750)
    script = render(
        row=gpu_row, driver="/bin/true", python=str(python), head_sha="c" * 40,
        results_root=ROW["results_root"], partition="gpu",
        cpus_per_task=8, mem="64G", time="08:00:00", scratch_base=str(tmp_path),
    )
    result = _run(script, tmp_path, PATH=f"{fake_bin}:/usr/bin:/bin",
                  SLURM_JOB_ID="123", SLURM_JOB_PARTITION="gpu")
    assert result.returncode == 2
    assert "MIG" in result.stderr


@pytest.mark.parametrize("duration", ["99:59:59", "100:00:00", "168:00:00"])
def test_gpu_resources_accept_two_or_three_digit_hours(duration: str) -> None:
    gpu_row = {**ROW, "device": "cuda"}
    render(row=gpu_row, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
           results_root=ROW["results_root"], partition="gpu", cpus_per_task=8,
           mem="64G", time=duration)


@pytest.mark.parametrize("duration", ["168:00:01", "00:00:00", "1:00:00", "08:60:00", "08:00:60"])
def test_gpu_resources_reject_overlong_zero_and_malformed_duration(duration: str) -> None:
    gpu_row = {**ROW, "device": "cuda"}
    with pytest.raises(ValueError, match="time"):
        render(row=gpu_row, driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
               results_root=ROW["results_root"], partition="gpu", cpus_per_task=8,
               mem="64G", time=duration)


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
    gpu_row = {**ROW, "device": "cuda"}
    gpu_overrides = row_overrides(gpu_row)
    assert validate_gateway_binding(
        binding=_binding_token(gpu_row), config=gpu_row["config"],
        overrides=gpu_overrides, kind="train",
    ) == gpu_row
    with pytest.raises(ProtocolError):
        validate_gateway_binding(
            binding=_binding_token(gpu_row), config=gpu_row["config"],
            overrides=[override for override in gpu_overrides
                       if not override.startswith("runtime.device=")],
            kind="train",
        )
    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_dir": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010",
        "checkpoint_file": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010/model.pt",
        "checkpoint_sha256": "d" * 64,
    }
    eval_token = _binding_token(eval_row)
    eval_overrides = row_overrides(eval_row)
    validate_gateway_binding(binding=eval_token, config=eval_row["config"], overrides=eval_overrides, kind="eval",
                             expected_checkpoint_sha256="d" * 64)
    with pytest.raises(ProtocolError):
        validate_gateway_binding(binding=eval_token, config=eval_row["config"], overrides=eval_overrides, kind="eval",
                                 expected_checkpoint_sha256=None)


def test_train_gateway_rejects_missing_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.atomistic.stage1 import train

    row = {
        **ROW,
        "kind": "smoke",
        "config": "configs/he/does-not-exist.yaml",
    }
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config", row["config"],
            *row_overrides(row),
            "--binding", _binding_token(row),
            "--kind", "smoke",
        ],
    )
    with pytest.raises(FileNotFoundError, match="config path does not exist"):
        train.main()


@pytest.mark.parametrize("checkpoint_override", [None, "/other/checkpoint.pt"])
def test_gateway_rejects_missing_or_mismatched_checkpoint_file(
    checkpoint_override: str | None,
) -> None:
    from experiments.atomistic.stage1.launch import _binding_token, row_overrides

    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_dir": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010",
        "checkpoint_file": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010/model.pt",
        "checkpoint_sha256": "d" * 64,
    }
    overrides = row_overrides(eval_row)
    if checkpoint_override is None:
        overrides = [
            override for override in overrides
            if not override.startswith("trajectory_identity.checkpoint_file=")
        ]
    else:
        overrides = [
            (f"trajectory_identity.checkpoint_file={checkpoint_override}"
             if override.startswith("trajectory_identity.checkpoint_file=") else override)
            for override in overrides
        ]
    with pytest.raises(ProtocolError):
        validate_gateway_binding(
            binding=_binding_token(eval_row), config=eval_row["config"],
            overrides=overrides, kind="eval", expected_checkpoint_sha256="d" * 64,
        )


@pytest.mark.parametrize("checkpoint_file", ["other.pt", "nested/model.pt"])
def test_gateway_checkpoint_file_must_be_the_planned_model_file(
    tmp_path: Path, checkpoint_file: str,
) -> None:
    """A descendant of the step directory is not necessarily the loaded model."""

    checkpoint_dir = tmp_path / "checkpoints" / "step_000010"
    bad_file = checkpoint_dir / checkpoint_file
    eval_row = {
        **ROW,
        "kind": "eval",
        "results_root": str(tmp_path / "results"),
        "draws": 256,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str(bad_file),
        "checkpoint_sha256": "d" * 64,
    }
    overrides = row_overrides(eval_row)
    with pytest.raises(ProtocolError, match="checkpoint_file"):
        validate_gateway_binding(
            binding=_binding_token(eval_row),
            config=eval_row["config"], overrides=overrides, kind="eval",
            expected_checkpoint_sha256=eval_row["checkpoint_sha256"],
        )


def test_gateway_accepts_the_planned_step_model_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "step_000010"
    eval_row = {
        **ROW,
        "kind": "eval",
        "results_root": str(tmp_path / "results"),
        "draws": 256,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str(checkpoint_dir / "model.pt"),
        "checkpoint_sha256": "d" * 64,
    }
    assert validate_gateway_binding(
        binding=_binding_token(eval_row), config=eval_row["config"],
        overrides=row_overrides(eval_row), kind="eval",
        expected_checkpoint_sha256=eval_row["checkpoint_sha256"],
    ) == eval_row


def test_gateway_rejects_layout_override_that_disagrees_with_planner(tmp_path: Path) -> None:
    row = {
        **ROW,
        "results_root": str(tmp_path / "results"),
    }
    planned = resolve_run_dir(row)
    assert planned == Path(row["results_root"]) / row["experiment_name"] / row["sector"] / row["run_id"]
    with pytest.raises(ProtocolError, match="layout"):
        validate_gateway_binding(
            binding=_binding_token(row), config=row["config"],
            overrides=[*row_overrides(row), "run.layout=flat"], kind="train",
        )


def test_render_rejects_absolute_run_components_outside_results_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved_root = tmp_path / "approved-results"
    monkeypatch.setattr(
        "experiments.atomistic.stage1.launch.FACILITY_RESULTS_PREFIX",
        f"{tmp_path}/",
    )
    row = {
        **ROW,
        "results_root": str(approved_root),
        "source_sha": "c" * 40,
    }
    (approved_root / ROW["experiment_name"] / ROW["sector"] / ROW["run_id"]).mkdir(
        parents=True,
    )
    render(row=row, driver="driver.py", python="/usr/bin/python3",
           head_sha="c" * 40, results_root=str(approved_root),
           scratch_base=str(tmp_path / "scratch"))

    escaped_root = tmp_path / "escaped-results"
    escaped_root.mkdir()
    outside = tmp_path / "outside-results"
    outside.mkdir()
    (escaped_root / ROW["experiment_name"]).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="beneath results_root"):
        render(
            row={**row, "results_root": str(escaped_root)},
            driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
            results_root=str(escaped_root), scratch_base=str(tmp_path / "scratch"),
        )
    for field in ("experiment_name", "sector", "run_id"):
        with pytest.raises(ValueError, match=field):
            render(
                row={**row, field: str(tmp_path / "escaped")},
                driver="driver.py", python="/usr/bin/python3", head_sha="c" * 40,
                results_root=str(approved_root), scratch_base=str(tmp_path / "scratch"),
            )


@pytest.mark.parametrize("field,value", [
    ("experiment_name", ""), ("experiment_name", "."), ("experiment_name", ".."),
    ("sector", "nested/sector"), ("run_id", "../escaped"),
])
def test_gateway_rejects_non_component_run_identity(field: str, value: str) -> None:
    row = {**ROW, field: value}
    with pytest.raises(ProtocolError, match=field):
        validate_gateway_binding(
            binding=_binding_token(row), config=row["config"],
            overrides=row_overrides(row), kind="train",
        )


def test_gateway_rejects_effective_run_dir_outside_results_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved_root = tmp_path / "approved"
    row = {**ROW, "results_root": str(approved_root)}
    monkeypatch.setattr(
        "experiments.atomistic.stage1.protocol.resolve_run_dir",
        lambda _row: tmp_path / "outside" / "stage1" / "he" / "run-0",
    )
    with pytest.raises(ProtocolError, match="beneath results_root"):
        validate_gateway_binding(
            binding=_binding_token(row), config=row["config"],
            overrides=row_overrides(row), kind="train",
        )


def test_row_overrides_bind_nested_layout_even_when_config_inherits_flat() -> None:
    row = {**ROW, "config": "configs/he/inherited-flat.yaml"}
    overrides = row_overrides(row)
    assert "run.layout=nested" in overrides
    assert validate_gateway_binding(
        binding=_binding_token(row), config=row["config"], overrides=overrides, kind="train",
    ) == row


def test_gateway_rejects_relative_checkpoint_paths() -> None:
    from experiments.atomistic.stage1.launch import _binding_token, row_overrides

    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_dir": "relative/checkpoint",
        "checkpoint_file": "relative/checkpoint/model.pt",
        "checkpoint_sha256": "d" * 64,
    }
    with pytest.raises(ProtocolError, match="checkpoint_dir and checkpoint_file must be absolute"):
        validate_gateway_binding(
            binding=_binding_token(eval_row), config=eval_row["config"],
            overrides=row_overrides(eval_row), kind="eval",
            expected_checkpoint_sha256="d" * 64,
        )


def test_gateway_rejects_mismatched_checkpoint_expectation() -> None:
    from experiments.atomistic.stage1.launch import _binding_token, row_overrides

    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_dir": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010",
        "checkpoint_file": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010/model.pt",
        "checkpoint_sha256": "d" * 64,
    }
    with pytest.raises(ProtocolError):
        validate_gateway_binding(
            binding=_binding_token(eval_row), config=eval_row["config"],
            overrides=row_overrides(eval_row), kind="eval",
            expected_checkpoint_sha256="e" * 64,
        )


@pytest.mark.parametrize("key", ["load.mode", "load.path", "runtime.seed"])
def test_gateway_rejects_conflicting_duplicate_override(key: str) -> None:
    from experiments.atomistic.stage1.launch import _binding_token, row_overrides

    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_dir": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010",
        "checkpoint_file": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010/model.pt",
        "checkpoint_sha256": "d" * 64,
    }
    overrides = row_overrides(eval_row)
    with pytest.raises(ProtocolError):
        validate_gateway_binding(
            binding=_binding_token(eval_row), config=eval_row["config"],
            overrides=[*overrides, f"{key}=conflict"], kind="eval",
            expected_checkpoint_sha256="d" * 64,
        )


@pytest.mark.parametrize("override", ["++load.mode=full", "+runtime.seed=1", "~load.strict"])
def test_gateway_rejects_prefixed_override(override: str) -> None:
    from experiments.atomistic.stage1.launch import _binding_token, row_overrides

    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_dir": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010",
        "checkpoint_file": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010/model.pt",
        "checkpoint_sha256": "d" * 64,
    }
    with pytest.raises(ProtocolError, match="prefixed gateway overrides"):
        validate_gateway_binding(
            binding=_binding_token(eval_row), config=eval_row["config"],
            overrides=[*row_overrides(eval_row), override], kind="eval",
            expected_checkpoint_sha256="d" * 64,
        )


def test_command_binds_every_kind() -> None:
    eval_row = {
        **ROW, "kind": "eval", "draws": 256,
        "checkpoint_dir": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010",
        "checkpoint_file": "/n/netscratch/kozinsky_lab/Lab/rhu/stage1/stage1/he/train-lineage-0/checkpoints/step_000010/model.pt",
        "checkpoint_sha256": "d" * 64,
    }
    for row in ({**ROW, "kind": "smoke"}, ROW, eval_row):
        command = command_for_row(row, python="/usr/bin/python3", driver="driver.py")
        assert f"runtime.seed={row['seed']}" in command
        assert str(row["walkers"]) in command
    eval_command = command_for_row(eval_row, python="/usr/bin/python3", driver="eval.py")
    assert "load.path=" in eval_command
    assert "trajectory_identity.checkpoint_file=" in eval_command
    assert "load.mode=model_only" in eval_command
    assert "--expected-resolved-config-identity" not in eval_command
