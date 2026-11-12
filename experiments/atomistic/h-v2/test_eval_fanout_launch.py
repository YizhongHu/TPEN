"""launch.py stage selection for item E's EVAL fan-out (TO 8a5e926f).

Orchestrator round-1 correction: ``--stage train|eval|both``, the ``eval``
leg's sbatch wiring (``afterok`` on train for science's chained route), the
pilot stages (``--pilot-stage 1``/``2``/``2-core``, each a SEPARATE job,
never auto-chained), and part (g). No live scheduler is touched: every test
either injects a stub ``sbatch`` runner or drives a real subprocess under a
stripped environment (never a live cluster).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SHA = "a" * 40
TRAIN_SHA = "b" * 40


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
launch_config = _h_v2_module("launch_config")


def _runner(calls: list[list[str]], job_ids: list[str]):
    def run(command: list[str]) -> subprocess.CompletedProcess:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, returncode=0, stdout=f"{job_ids[len(calls) - 1]};cluster\n", stderr="")

    return run


# --- science: --stage train / eval / both (default) ------------------------


def test_science_both_default_dry_run_is_exactly_six_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """B3: the science dry run stays exactly 6 lines (3 lineages x train+eval)."""

    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "T_WALL": "08:00:00", "E_WALL": "24:00:00"}
    result = subprocess.run(
        [
            "env", "-i", *(f"{k}={v}" for k, v in env.items()),
            sys.executable, str(HERE / "launch.py"),
            "--run-root", str(tmp_path / "runs" / "h-v2" / "science"),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "science",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("sbatch --parsable") == 6, result.stdout
    assert result.stdout.count("--dependency=afterok:") == 3, result.stdout


def test_packed_eval_scales_host_resources_and_passes_pack_k(tmp_path: Path) -> None:
    """Packing is explicit and requests one CPU/memory slice per stream."""

    command = launch.build_eval_fanout_sbatch_command(
        "science",
        lineage="L1",
        expected_train_commit=TRAIN_SHA,
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path / "runs" / "h-v2" / "science",
        checkout_root=tmp_path / "checkout",
        time_value="04:00:00",
        dependency=None,
        commit=SHA,
        resources=launch_config.RESOURCES,
        pack_k=3,
    )
    assert "--cpus-per-task=12" in command
    assert "--mem=96G" in command
    assert "--gres=gpu:1" in command
    assert "--pack-k 3" in next(flag for flag in command if flag.startswith("--wrap="))


def test_packed_eval_rejects_out_of_range_pack_k(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchError, match="between 1 and 6"):
        launch.build_eval_fanout_sbatch_command(
            "science",
            lineage="L1",
            expected_train_commit=TRAIN_SHA,
            science_python=sys.executable,
            executor_python=sys.executable,
            run_root=tmp_path,
            checkout_root=tmp_path / "checkout",
            time_value="04:00:00",
            dependency=None,
            commit=SHA,
            resources=launch_config.RESOURCES,
            pack_k=7,
        )


def test_science_stage_train_dry_run_is_exactly_three_lines_no_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("T_WALL", "08:00:00")
    code = launch.main(
        [
            "--run-root", str(tmp_path / "runs" / "h-v2" / "science"),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "science",
            "--stage", "train",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.count("sbatch --parsable") == 3
    assert "eval-fanout" not in printed
    assert "--dependency=" not in printed


def test_science_stage_eval_requires_expected_train_commit_and_time(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchError, match="--expected-train-commit"):
        launch.main(
            [
                "--run-root", str(tmp_path),
                "--commit", SHA,
                "--payload", "science",
                "--stage", "eval",
                "--time", "04:00:00",
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )
    with pytest.raises(launch.LaunchError, match="--time"):
        launch.main(
            [
                "--run-root", str(tmp_path),
                "--commit", SHA,
                "--payload", "science",
                "--stage", "eval",
                "--expected-train-commit", TRAIN_SHA,
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )


def test_science_stage_eval_dry_run_is_one_line_per_lineage_no_afterok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """EVAL-only submission against an earlier TRAIN commit: standalone, no dependency."""

    code = launch.main(
        [
            "--run-root", str(tmp_path / "runs" / "h-v2" / "science"),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "science",
            "--stage", "eval",
            "--expected-train-commit", TRAIN_SHA,
            "--time", "04:00:00",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.count("sbatch --parsable") == 3
    assert "--dependency=" not in printed
    assert printed.count(f"--expected-train-commit {TRAIN_SHA}") == 3


def test_science_both_chains_eval_via_afterok_on_the_just_submitted_train_id(tmp_path: Path) -> None:
    """Mutant: 'eval submitted without afterok'."""

    calls: list[list[str]] = []
    job_ids = ["111", "222", "333", "444", "555", "666"]
    train_id, eval_id = launch.submit_lineage(
        "L1",
        payload="science",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:00:00",
        e_wall="24:00:00",
        commit=SHA,
        runner=_runner(calls, job_ids),
    )
    assert len(calls) == 2
    train_command, eval_command = calls
    assert f"--dependency=afterok:{train_id}" in eval_command
    assert not any(flag.startswith("--dependency=") for flag in train_command)
    eval_wrap = next(flag for flag in eval_command if flag.startswith("--wrap="))
    assert "eval-fanout" in eval_wrap
    assert "science" in eval_wrap
    assert "L1" in eval_wrap
    assert SHA in eval_wrap  # --expected-train-commit == this invocation's own commit


def test_submit_lineage_forwards_pack_k_on_the_submitted_eval_line(tmp_path: Path) -> None:
    """The production lineage submitter preserves packing on its EVAL command."""

    calls: list[list[str]] = []
    train_id, eval_id = launch.submit_lineage(
        "L1",
        payload="science",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:00:00",
        e_wall="24:00:00",
        commit=SHA,
        pack_k=3,
        runner=_runner(calls, ["train-1", "eval-1"]),
    )

    assert (train_id, eval_id) == ("train-1", "eval-1")
    eval_wrap = next(flag for flag in calls[1] if flag.startswith("--wrap="))
    assert "--pack-k 3" in eval_wrap


def test_submit_all_forwards_pack_k_on_each_submitted_eval_line(tmp_path: Path) -> None:
    """The production all-lineage submitter preserves packing on EVAL commands."""

    calls: list[list[str]] = []
    result = launch.submit_all(
        payload="science",
        science_python=sys.executable,
        executor_python=sys.executable,
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:00:00",
        e_wall="24:00:00",
        commit=SHA,
        lineages=("L1",),
        pack_k=4,
        runner=_runner(calls, ["train-1", "eval-1"]),
    )

    assert result == {"L1": ("train-1", "eval-1")}
    eval_wrap = next(flag for flag in calls[1] if flag.startswith("--wrap="))
    assert "--pack-k 4" in eval_wrap


# --- part (g): standalone EVAL-only job -------------------------------------


def test_gpu_route_g_stage_eval_dry_run_is_one_line_gpu_test_resources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = launch.main(
        [
            "--run-root", str(tmp_path / "runs" / "h-v2" / "gpu-route-g" / "G"),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "gpu-route-g",
            "--stage", "eval",
            "--expected-train-commit", TRAIN_SHA,
            "--time", "00:30:00",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.count("sbatch --parsable") == 1
    assert "--partition=gpu_test" in printed
    assert "eval-fanout gpu-route-g" in printed
    assert "--pilot-stage" not in printed


def test_gpu_route_g_stage_eval_refuses_a_pilot_stage_flag(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchError, match="--pilot-stage is not supported"):
        launch.main(
            [
                "--run-root", str(tmp_path / "runs" / "h-v2" / "gpu-route-g" / "G"),
                "--run-root-prefix", str(tmp_path / "runs"),
                "--commit", SHA,
                "--payload", "gpu-route-g",
                "--stage", "eval",
                "--pilot-stage", "1",
                "--expected-train-commit", TRAIN_SHA,
                "--time", "00:30:00",
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )


# --- capacity-pilot: two-stage, each stage a SEPARATE job -------------------


def test_capacity_pilot_stage1_dry_run_is_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = launch.main(
        [
            "--run-root", str(tmp_path / "runs" / "h-v2" / "capacity-pilot" / "P"),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "capacity-pilot",
            "--stage", "eval",
            "--pilot-stage", "1",
            "--expected-train-commit", TRAIN_SHA,
            "--time", "06:25:00",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.count("sbatch --parsable") == 1
    assert "eval-fanout capacity-pilot" in printed
    assert "--pilot-stage 1" in printed


def test_capacity_pilot_stage1_never_auto_submits_stage2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mutant: 'stage 2 chained automatically after stage 1'.

    Submitting stage 1 must print exactly one sbatch-shaped line; a mutant
    that automatically submits (or dry-run-prints) a second,
    stage-2-shaped command in the same invocation must be caught. Never
    reads the stage-2 selection record either -- only the stage-2 path does.
    """

    code = launch.main(
        [
            "--run-root", str(tmp_path / "runs" / "h-v2" / "capacity-pilot" / "P"),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "capacity-pilot",
            "--stage", "eval",
            "--pilot-stage", "1",
            "--expected-train-commit", TRAIN_SHA,
            "--time", "06:25:00",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.count("sbatch --parsable") == 1
    assert "--pilot-stage 2" not in printed
    selection_path = tmp_path / "runs" / "h-v2" / "capacity-pilot" / "P" / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
    assert not selection_path.exists()


def test_capacity_pilot_stage2_refuses_before_sbatch_when_no_selection_record(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchError, match="no SEL selection record"):
        launch.main(
            [
                "--run-root", str(tmp_path / "runs" / "h-v2" / "capacity-pilot" / "P"),
                "--run-root-prefix", str(tmp_path / "runs"),
                "--commit", SHA,
                "--payload", "capacity-pilot",
                "--stage", "eval",
                "--pilot-stage", "2",
                "--expected-train-commit", TRAIN_SHA,
                "--time", "04:05:00",
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )


def test_capacity_pilot_stage2_refuses_before_sbatch_for_an_out_of_range_c1(tmp_path: Path) -> None:
    """Refusal of an out-of-range c1 before any sbatch call."""

    run_root = tmp_path / "runs" / "h-v2" / "capacity-pilot" / "P"
    selection_path = run_root / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps({"c1": 25, "triple_means": {}, "b1_eval_receipt_sha256": "x"}), encoding="utf-8")
    with pytest.raises(launch.LaunchError, match="refusing before any sbatch call"):
        launch.main(
            [
                "--run-root", str(run_root),
                "--run-root-prefix", str(tmp_path / "runs"),
                "--commit", SHA,
                "--payload", "capacity-pilot",
                "--stage", "eval",
                "--pilot-stage", "2",
                "--expected-train-commit", TRAIN_SHA,
                "--time", "04:05:00",
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )


def test_capacity_pilot_stage2_core_dry_run_succeeds_with_a_valid_selection_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_root = tmp_path / "runs" / "h-v2" / "capacity-pilot" / "P"
    selection_path = run_root / "attached" / "receipts" / "h-v2-eval-capacity-pilot-selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps({"c1": 10, "triple_means": {}, "b1_eval_receipt_sha256": "x"}), encoding="utf-8")
    code = launch.main(
        [
            "--run-root", str(run_root),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "capacity-pilot",
            "--stage", "eval",
            "--pilot-stage", "2-core",
            "--expected-train-commit", TRAIN_SHA,
            "--time", "03:00:00",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.count("sbatch --parsable") == 1
    assert "--pilot-stage 2-core" in printed


def test_capacity_pilot_stage_both_is_refused() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args(["--payload", "capacity-pilot", "--stage", "bogus"])


# --- capacity-pilot-100k: stage-1-only, no stage 2 (amendment 20261112-r1) --


def test_capacity_pilot_100k_stage1_dry_run_is_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = launch.main(
        [
            "--run-root", str(tmp_path / "runs" / "h-v2" / "capacity-pilot-100k" / "P"),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "capacity-pilot-100k",
            "--stage", "eval",
            "--pilot-stage", "1",
            "--expected-train-commit", TRAIN_SHA,
            "--time", "02:00:00",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert printed.count("sbatch --parsable") == 1
    assert "eval-fanout capacity-pilot-100k" in printed
    assert "--pilot-stage 1" in printed


def test_capacity_pilot_100k_requires_pilot_stage(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchError, match="--pilot-stage"):
        launch.main(
            [
                "--run-root", str(tmp_path / "runs" / "h-v2" / "capacity-pilot-100k" / "P"),
                "--run-root-prefix", str(tmp_path / "runs"),
                "--commit", SHA,
                "--payload", "capacity-pilot-100k",
                "--stage", "eval",
                "--expected-train-commit", TRAIN_SHA,
                "--time", "02:00:00",
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )


@pytest.mark.parametrize("pilot_stage", ["2", "2-core"])
def test_capacity_pilot_100k_refuses_pilot_stage_2_before_any_sbatch_call(
    tmp_path: Path, pilot_stage: str
) -> None:
    """Mutant: the 100k payload has no stage 2 -- it must refuse client-side, never
    silently run capacity-pilot's own stage-2/2-core route."""

    with pytest.raises(launch.LaunchError, match="must be '1'"):
        launch.main(
            [
                "--run-root", str(tmp_path / "runs" / "h-v2" / "capacity-pilot-100k" / "P"),
                "--run-root-prefix", str(tmp_path / "runs"),
                "--commit", SHA,
                "--payload", "capacity-pilot-100k",
                "--stage", "eval",
                "--pilot-stage", pilot_stage,
                "--expected-train-commit", TRAIN_SHA,
                "--time", "02:00:00",
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )
    # Refused before any sbatch call: no selection-record read, no submission.
    selection_path = (
        tmp_path / "runs" / "h-v2" / "capacity-pilot-100k" / "P" / "attached" / "receipts"
        / "h-v2-eval-capacity-pilot-100k-selection.json"
    )
    assert not selection_path.exists()


def test_capacity_pilot_100k_refuses_a_run_root_outside_its_own_prefix(tmp_path: Path) -> None:
    """The 100k payload's run root is its OWN prefix (``h-v2/capacity-pilot-100k/P``),
    distinct from capacity-pilot's (``h-v2/capacity-pilot/P``)."""

    with pytest.raises(launch.LaunchError, match="run root"):
        launch.main(
            [
                "--run-root", str(tmp_path / "runs" / "h-v2" / "capacity-pilot" / "P"),
                "--run-root-prefix", str(tmp_path / "runs"),
                "--commit", SHA,
                "--payload", "capacity-pilot-100k",
                "--stage", "eval",
                "--pilot-stage", "1",
                "--expected-train-commit", TRAIN_SHA,
                "--time", "02:00:00",
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )


def test_pilot_payload_never_submits_stage_both(tmp_path: Path) -> None:
    with pytest.raises(launch.LaunchError, match="never submits 'both'"):
        launch.main(
            [
                "--run-root", str(tmp_path),
                "--commit", SHA,
                "--payload", "capacity-pilot",
                "--science-python", sys.executable,
                "--executor-python", sys.executable,
                "--dry-run",
            ]
        )
