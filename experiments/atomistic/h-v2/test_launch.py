"""Acceptance tests for the H-v2 study-side launch script (item 67f2eb57).

No scheduler is touched: every test injects a stub ``sbatch`` runner -- a
plain callable that records the exact command and returns a scripted
``subprocess.CompletedProcess`` -- per the acceptance contract's "study
tests, no scheduler needed" clause.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


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


class _StubRunner:
    """Records every command issued and returns a scripted result per call."""

    def __init__(self, job_ids: list[str] | None = None, *, fail_on: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self._job_ids = list(job_ids or [])
        self._fail_on = fail_on  # 1-indexed call number that returns non-zero

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(list(command))
        call_number = len(self.calls)
        if self._fail_on == call_number:
            return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="boom")
        job_id = self._job_ids[call_number - 1]
        return subprocess.CompletedProcess(command, returncode=0, stdout=f"{job_id};cluster\n", stderr="")


def test_one_lineage_issues_exactly_two_sbatch_commands_with_chained_afterok(tmp_path: Path) -> None:
    runner = _StubRunner(job_ids=["111", "222"])
    train_id, eval_id = launch.submit_lineage(
        "L1",
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:30:00",
        e_wall="25:24:00",
        runner=runner,
    )
    assert len(runner.calls) == 2
    assert train_id == "111"
    assert eval_id == "222"
    train_command, eval_command = runner.calls
    assert "--job-name=h-v2-sci-L1-train" in train_command
    assert "--job-name=h-v2-sci-L1-eval" in eval_command
    assert not any(flag.startswith("--dependency=") for flag in train_command)
    assert "--dependency=afterok:111" in eval_command


def test_every_lineage_gets_its_own_chained_pair(tmp_path: Path) -> None:
    # In-budget pair (F4: submit_all validates T_WALL + E_WALL <= 32h before
    # any submission); 08:00:00 + 23:00:00 = 31h.
    runner = _StubRunner(job_ids=["1", "2", "3", "4", "5", "6"])
    results = launch.submit_all(
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:00:00",
        e_wall="23:00:00",
        runner=runner,
    )
    assert len(runner.calls) == 2 * len(launch_config.LINEAGES)
    assert set(results) == set(launch_config.LINEAGES)
    for train_id, eval_id in results.values():
        assert train_id is not None
        assert eval_id is not None


def test_flags_equal_the_section_c_config_values(tmp_path: Path) -> None:
    runner = _StubRunner(job_ids=["1", "2"])
    launch.submit_lineage(
        "L1",
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:30:00",
        e_wall="25:24:00",
        runner=runner,
    )
    resources = launch_config.RESOURCES
    for command in runner.calls:
        assert f"--partition={resources.partition}" in command
        assert f"--constraint={resources.constraint}" in command
        assert f"--gres=gpu:{resources.gpus}" in command
        assert f"--nodes={resources.nodes}" in command
        assert f"--ntasks={resources.ntasks}" in command
        assert f"--cpus-per-task={resources.cpus_per_task}" in command
        assert f"--mem={resources.mem}" in command
        assert "--no-requeue" in command
    assert "--time=08:30:00" in runner.calls[0]
    assert "--time=25:24:00" in runner.calls[1]


def test_a_failed_first_submission_aborts_before_the_second_sbatch(tmp_path: Path) -> None:
    runner = _StubRunner(job_ids=["111"], fail_on=1)
    with pytest.raises(launch.LaunchError):
        launch.submit_lineage(
            "L1",
            run_root=tmp_path,
            checkout_root=tmp_path / "checkout",
            t_wall="08:30:00",
            e_wall="25:24:00",
            runner=runner,
        )
    assert len(runner.calls) == 1


def test_a_missing_job_id_from_the_first_sbatch_aborts_before_the_second(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, returncode=0, stdout="\n", stderr="")

    with pytest.raises(launch.LaunchError):
        launch.submit_lineage(
            "L1",
            run_root=tmp_path,
            checkout_root=tmp_path / "checkout",
            t_wall="08:30:00",
            e_wall="25:24:00",
            runner=runner,
        )
    assert len(calls) == 1


def test_a_failed_lineage_aborts_before_any_later_lineage(tmp_path: Path) -> None:
    # In-budget pair; see test_every_lineage_gets_its_own_chained_pair.
    runner = _StubRunner(job_ids=["111"], fail_on=1)
    with pytest.raises(launch.LaunchError):
        launch.submit_all(
            run_root=tmp_path,
            checkout_root=tmp_path / "checkout",
            t_wall="08:00:00",
            e_wall="23:00:00",
            runner=runner,
        )
    # Only L1's train sbatch was attempted; L2/L3 never ran.
    assert len(runner.calls) == 1


def test_dry_run_prints_commands_and_calls_no_runner(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[list[str]] = []
    train_id, eval_id = launch.submit_lineage(
        "L1",
        run_root=tmp_path,
        checkout_root=tmp_path / "checkout",
        t_wall="08:30:00",
        e_wall="25:24:00",
        runner=calls.append,  # type: ignore[arg-type]
        dry_run=True,
    )
    assert calls == []
    assert train_id is None and eval_id is None
    printed = capsys.readouterr().out
    assert printed.count("sbatch") == 2
    assert "--dependency=afterok:" in printed


def test_launch_script_never_cancels_or_uses_the_forbidden_entrypoint() -> None:
    source = (HERE / "launch.py").read_text(encoding="utf-8")
    assert "scancel" not in source
    assert "qdel" not in source
    assert "python -m tpen.run" not in source


def test_main_requires_t_wall_and_e_wall_when_not_a_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("T_WALL", raising=False)
    monkeypatch.delenv("E_WALL", raising=False)
    with pytest.raises(launch.LaunchError):
        launch.main(["--run-root", str(tmp_path), "--commit", "deadbeef"])
