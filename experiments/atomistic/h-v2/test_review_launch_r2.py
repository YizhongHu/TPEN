"""PR619 T2 round-2 probes for decisions F1/F3 completeness (G1-G3).

Per orchestrator note ``verification-8641d0ee-disposition``: the round-1
fix left a ``science`` payload reaching dispatch unconditionally, the
wrapper execed an inherited ``$SCIENCE_PYTHON`` rather than an explicit
validated interpreter, and no test killed the mutant where
``checkpoint_dir_for_eval`` silently falls back to a placeholder path. No
live scheduler is touched; every test either injects a stub ``sbatch``
runner or stays at the pure argument-building/validation layer.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SHA = "a" * 40


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


launch = _h_v2_module("launch")
job = _h_v2_module("run_job")


def _runner(calls: list[list[str]], job_ids: list[str]):
    def run(command: list[str]) -> subprocess.CompletedProcess:
        calls.append(list(command))
        job_id = job_ids[len(calls) - 1]
        return subprocess.CompletedProcess(command, 0, f"{job_id};cluster\n", "")

    return run


# --- G1: explicit payload choice -------------------------------------------


def test_g1_science_payload_refuses_before_any_submission(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    with pytest.raises(launch.LaunchError) as excinfo:
        launch.submit_all(
            payload="science",
            science_python=sys.executable,
            executor_python=sys.executable,
            run_root=tmp_path,
            checkout_root=tmp_path / "checkout",
            t_wall="08:00:00",
            e_wall="24:00:00",
            runner=calls.append,  # type: ignore[arg-type]
        )
    assert calls == []
    assert job.SCIENCE_CONFIG_UNAVAILABLE_CODE in str(excinfo.value)


def test_g1_science_payload_refuses_in_dry_run_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    with pytest.raises(launch.LaunchError) as excinfo:
        launch.submit_all(
            payload="science",
            science_python=sys.executable,
            executor_python=sys.executable,
            run_root=tmp_path,
            checkout_root=tmp_path / "checkout",
            t_wall="08:00:00",
            e_wall="24:00:00",
            runner=calls.append,  # type: ignore[arg-type]
            dry_run=True,
        )
    assert calls == []
    assert job.SCIENCE_CONFIG_UNAVAILABLE_CODE in str(excinfo.value)
    assert capsys.readouterr().out == ""


def test_g1_science_payload_refuses_through_main_with_zero_runner_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("T_WALL", "08:00:00")
    monkeypatch.setenv("E_WALL", "24:00:00")
    with pytest.raises(launch.LaunchError) as excinfo:
        launch.main([
            "--run-root", str(tmp_path),
            "--commit", SHA,
            "--payload", "science",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
        ])
    assert calls == []
    assert job.SCIENCE_CONFIG_UNAVAILABLE_CODE in str(excinfo.value)


def test_g1_pipeline_check_submits_two_labelled_commands_per_lineage_with_afterok(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    job_ids = [str(n) for n in range(1, 2 * len(launch.launch_config.LINEAGES) + 1)]
    results = launch.submit_all(
        payload="pipeline-check",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:00:00",
        e_wall="24:00:00",
        runner=_runner(calls, job_ids),
    )
    assert len(calls) == 2 * len(launch.launch_config.LINEAGES)
    assert set(results) == set(launch.launch_config.LINEAGES)
    for lineage in launch.launch_config.LINEAGES:
        train_command = next(c for c in calls if f"--job-name=h-v2-pipeline-check-{lineage}-train" in c)
        eval_command = next(c for c in calls if f"--job-name=h-v2-pipeline-check-{lineage}-eval" in c)
        train_id, _ = results[lineage]
        assert f"--dependency=afterok:{train_id}" in eval_command
        train_wrap = next(a for a in train_command if a.startswith("--wrap="))
        eval_wrap = next(a for a in eval_command if a.startswith("--wrap="))
        assert train_wrap.endswith("--pipeline-check")
        assert eval_wrap.endswith("--pipeline-check")


def test_g1_build_sbatch_command_rejects_unknown_payload(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchError):
        launch.build_sbatch_command(
            "train", "L1",
            payload="not-a-real-payload",
            science_python=sys.executable,
            executor_python=sys.executable,
            run_root=tmp_path, checkout_root=tmp_path / "checkout",
            time_value="08:00:00", dependency=None,
        )


# --- G2: explicit absolute science/executor interpreter ---------------------


@pytest.mark.parametrize("bad_path", ["relative/python", "./python", "python"])
def test_g2_relative_science_python_refuses_with_zero_runner_calls(
    tmp_path: Path, bad_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("T_WALL", "08:00:00")
    monkeypatch.setenv("E_WALL", "24:00:00")
    with pytest.raises(launch.LaunchError):
        launch.main([
            "--run-root", str(tmp_path),
            "--commit", SHA,
            "--payload", "pipeline-check",
            "--science-python", bad_path,
            "--executor-python", sys.executable,
        ])
    assert calls == []


def test_g2_missing_science_python_refuses_with_zero_runner_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("T_WALL", "08:00:00")
    monkeypatch.setenv("E_WALL", "24:00:00")
    missing = tmp_path / "no-such-python"
    with pytest.raises(launch.LaunchError):
        launch.main([
            "--run-root", str(tmp_path),
            "--commit", SHA,
            "--payload", "pipeline-check",
            "--science-python", str(missing),
            "--executor-python", sys.executable,
        ])
    assert calls == []


def test_g2_relative_executor_python_refuses_with_zero_runner_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("T_WALL", "08:00:00")
    monkeypatch.setenv("E_WALL", "24:00:00")
    with pytest.raises(launch.LaunchError):
        launch.main([
            "--run-root", str(tmp_path),
            "--commit", SHA,
            "--payload", "pipeline-check",
            "--science-python", sys.executable,
            "--executor-python", "relative/python",
        ])
    assert calls == []


def test_g2_wrapper_embeds_quoted_literal_interpreters_not_env_vars(tmp_path: Path) -> None:
    command = launch.build_sbatch_command(
        "train", "L1",
        payload="pipeline-check",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path, checkout_root=tmp_path / "checkout",
        time_value="08:00:00", dependency=None,
    )
    wrap = next(arg for arg in command if arg.startswith("--wrap="))[len("--wrap="):]
    assert "$SCIENCE_PYTHON" not in wrap
    assert "$EXECUTOR_PY" not in wrap
    assert sys.executable in wrap


def test_g2_distinct_science_and_executor_pythons_both_appear_literally(tmp_path: Path) -> None:
    science_python = str(tmp_path / "science" / "python")
    executor_python = str(tmp_path / "executor" / "python")
    Path(science_python).parent.mkdir(parents=True)
    Path(executor_python).parent.mkdir(parents=True)
    Path(science_python).write_text("", encoding="utf-8")
    Path(executor_python).write_text("", encoding="utf-8")
    command = launch.build_sbatch_command(
        "train", "L1",
        payload="pipeline-check",
        science_python=science_python,
        executor_python=executor_python,
        run_root=tmp_path, checkout_root=tmp_path / "checkout",
        time_value="08:00:00", dependency=None,
    )
    wrap = next(arg for arg in command if arg.startswith("--wrap="))[len("--wrap="):]
    assert science_python in wrap
    assert executor_python in wrap


# --- G3: checkpoint_dir_for_eval must never fall back silently -------------


def test_g3_checkpoint_dir_for_eval_raises_when_train_receipt_is_missing(tmp_path: Path) -> None:
    with pytest.raises(job.JobError):
        job.checkpoint_dir_for_eval(tmp_path, "L1")


def test_g3_checkpoint_dir_for_eval_raises_when_train_receipt_has_no_segments(
    tmp_path: Path,
) -> None:
    receipts_dir = tmp_path / "attached" / "receipts"
    receipts_dir.mkdir(parents=True)
    (receipts_dir / "h-v2-train-L1.h-v2.json").write_text('{"segments": []}', encoding="utf-8")
    with pytest.raises(job.JobError):
        job.checkpoint_dir_for_eval(tmp_path, "L1")
