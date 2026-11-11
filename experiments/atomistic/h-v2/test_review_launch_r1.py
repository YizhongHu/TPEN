"""PR619 T1 probes; execute on Cannon only, with no live scheduler calls.

The unmodified head is expected to fail the six finding probes. Controls
cover the receipt exit boundary, submission parsing, and quoted dry runs.
No runtime or mutation result is claimed by merely committing this file.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SHA = "a" * 40


def _study_module(name):
    key = f"h_v2_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # spec.name == key; matches this dir's own loader idiom
        spec.loader.exec_module(module)
    return sys.modules[key]


launch = _study_module("launch")
job = _study_module("run_job")


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    """Bypass facility identity only; keep real plan/handoff construction."""
    for key, value in {
        "RUN": str(tmp_path), "SHA": SHA, "SCIENCE_PYTHON": sys.executable,
        "EXECUTOR_PY": sys.executable, "SLURM_JOB_ID": "1234",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(job, "check_identity", lambda *args: None)
    return tmp_path


def _payload_kwargs():
    """G1/G2 (round 2): every submission-reaching call must now choose a
    payload and supply explicit interpreter paths. ``pipeline-check`` is the
    only payload these mechanics probes dispatch; ``test_f3`` (and
    ``run_job.py`` directly) still cover the ``science`` refusal itself.
    """

    return {
        "payload": "pipeline-check",
        "science_python": sys.executable,
        "executor_python": sys.executable,
    }


def _sidecar_bytes(prepared):
    return tuple(Path(path).read_bytes() for path in (
        prepared.handoff.executor_input, prepared.handoff.configured_runs,
    ))


def test_f1_submitted_script_finds_checkout_siblings_after_spooling(tmp_path):
    """Catch submitting a sibling-importing Python file directly to Slurm.

    Only the submitted script is copied, as Slurm does. The original checkout
    has an attached.py sentinel; reaching it (exit 73) proves sibling lookup
    used the checkout. It never runs science or needs a GPU. A wrapper using
    --wrap is also supported so the proposed fix can retain this oracle.
    """
    root = tmp_path / "run with spaces ' $(touch INJECTED)"
    checkout = root / "checkout"
    study = checkout / "experiments" / "atomistic" / "h-v2"
    study.mkdir(parents=True)
    for name in {"run_job.py", Path(launch.launch_config.JOB_SCRIPT_RELATIVE).name}:
        shutil.copy2(HERE / name, study / name)
    marker = tmp_path / "sibling-imported"
    (study / "attached.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('reached')\n"
        "raise SystemExit(73)\n", encoding="utf-8",
    )
    command = launch.build_sbatch_command(
        "train", "L1", run_root=root, checkout_root=checkout,
        time_value="08:00:00", dependency=None, **_payload_kwargs(),
    )
    spool = tmp_path / "spool" / "job1234"
    spool.mkdir(parents=True)
    script = spool / "slurm_script"
    wraps = [arg.split("=", 1)[1] for arg in command if arg.startswith("--wrap=")]
    if wraps:
        assert len(wraps) == 1
        script.write_text("#!/bin/sh\n" + wraps[0] + "\n", encoding="utf-8")
        args = []
    else:
        script_index = next(i for i in range(1, len(command)) if not command[i].startswith("-"))
        shutil.copy2(command[script_index], script)
        args = command[script_index + 1:]
    script.chmod(0o700)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    env = dict(os.environ, RUN=str(root), SHA=SHA, SCIENCE_PYTHON=sys.executable,
               EXECUTOR_PY=sys.executable, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run([str(script), *args], cwd=checkout, env=env,
                            capture_output=True, text=True, timeout=20, check=False)
    assert marker.is_file(), result.stderr
    assert result.returncode == 73, result.stderr
    assert not (checkout / "INJECTED").exists()


def test_f2_later_lineage_cannot_replace_live_earlier_handoff(job_env, monkeypatch):
    """Catch a shared handoff directory with a deterministic interleaving.

    Pause L1 before dispatch, prepare L2, then read the files L1 still owns.
    Real prepare_h_v2_run/write_attached_handoff perform every file write.

    ADAPTED for the F3 fix: TRAIN now refuses before dispatch unless
    ``pipeline_check`` is set (no approved per-lineage science config
    interface exists yet), so both calls pass it explicitly to reach
    dispatch at all; this finding (handoff isolation) is independent of
    which payload is dispatched. Task ids correspondingly carry the
    pipeline-check label the fix gives that payload.
    """
    seen = []

    def activate(prepared, **kwargs):
        seen.append(prepared)
        if prepared.task_id == "h-v2-train-pipeline-check-L1":
            before = _sidecar_bytes(prepared)
            job.run("train", "L2", pipeline_check=True)
            assert _sidecar_bytes(prepared) == before, "L2 replaced L1's bound sidecars"

    monkeypatch.setattr(job.attached, "activate_h_v2_run", activate)
    job.run("train", "L1", pipeline_check=True)
    assert {p.task_id for p in seen} == {
        "h-v2-train-pipeline-check-L1",
        "h-v2-train-pipeline-check-L2",
    }


def test_f3_science_lineages_do_not_silently_dispatch_pipeline_defaults(job_env, monkeypatch):
    """Catch three relabeled copies of the excluded-seed CPU pipeline run.

    A clear pre-dispatch JobError is acceptable until science configs exist.
    If dispatch proceeds, inspect the real prepared configs against C2/A1.
    """
    prepared = []
    def assemble(config, candidates, **kwargs):
        for target in candidates:
            job.protocol.append_candidate_module(config, target)
        return list(candidates), []

    monkeypatch.setattr(job.attached, "probe_admitted_a1_modules", assemble)
    monkeypatch.setattr(job.attached, "activate_h_v2_run", lambda value, **kw: prepared.append(value))
    for index, lineage in enumerate(("L1", "L2", "L3"), 1):
        count = len(prepared)
        try:
            job.run("train", lineage)
        except job.JobError as error:
            assert str(error)
            assert len(prepared) == count, "refusal came after dispatch"
            continue
        assert len(prepared) == count + 1
        config = prepared[-1].config
        observed = (config["runtime"]["seed"], config["sampler"]["seed"],
                    config["runtime"]["device"], config["trainer"]["max_steps"],
                    config["sampler"]["n_walkers"])
        assert observed == (73000 + index, 83000 + index, "cuda", 50000, 4096)


@pytest.mark.parametrize("t_wall,e_wall", [
    ("08:30:00", "25:24:00"), ("2-00:00:01", "00:01:00"),
])
def test_f4_over_budget_pair_refuses_before_any_submission(tmp_path, t_wall, e_wall):
    """Catch forwarding arbitrary durations without the C3 32-hour cap."""
    calls = []

    def runner(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, f"{len(calls)}\n", "")

    with pytest.raises(launch.LaunchError):
        launch.submit_all(run_root=tmp_path, checkout_root=tmp_path / "checkout",
                          t_wall=t_wall, e_wall=e_wall, runner=runner, **_payload_kwargs())
    assert calls == []


def test_c4_exact_budget_boundary_still_submits(tmp_path):
    """Passing control: 8 + 24 hours is exactly the permitted lineage cap."""
    calls = []

    def runner(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, f"{len(calls)}\n", "")

    result = launch.submit_all(run_root=tmp_path, checkout_root=tmp_path / "checkout",
                               t_wall="08:00:00", e_wall="1-00:00:00", runner=runner,
                               **_payload_kwargs())
    assert len(calls) == 6 and len(result) == 3


def test_f5_cli_identity_cannot_be_replaced_by_stale_environment(tmp_path, monkeypatch):
    """Catch a required --commit that never reaches the submitted job.

    Inspect the child environment at the real subprocess boundary. Accept
    either explicit export or early refusal; no scheduler process is started.
    """
    requested = tmp_path / "requested"
    monkeypatch.setenv("RUN", str(tmp_path / "stale"))
    monkeypatch.setenv("SHA", "b" * 40)
    monkeypatch.setenv("SCIENCE_PYTHON", sys.executable)
    monkeypatch.setenv("EXECUTOR_PY", sys.executable)
    monkeypatch.setenv("T_WALL", "08:00:00")
    monkeypatch.setenv("E_WALL", "24:00:00")
    observed = []

    def submit(command, **kwargs):
        assert command[0] == "sbatch", "unexpected subprocess"
        env = dict(kwargs.get("env") or os.environ)
        for arg in command:
            if arg.startswith("--export="):
                for entry in arg.split("=", 1)[1].split(","):
                    if "=" in entry:
                        key, value = entry.split("=", 1)
                        env[key] = value
        observed.append((env.get("RUN"), env.get("SHA")))
        return subprocess.CompletedProcess(command, 0, f"{len(observed)}\n", "")

    monkeypatch.setattr(launch.subprocess, "run", submit)
    try:
        launch.main([
            "--run-root", str(requested), "--commit", SHA,
            "--payload", "pipeline-check",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
        ])
    except launch.LaunchError:
        assert observed == [], "identity refusal must precede submission"
        return
    assert len(observed) == 6
    assert observed == [(str(requested.resolve()), SHA)] * 6


@pytest.mark.parametrize("device,accepted", [
    ("NVIDIA H200", False), ("NVIDIA A100-SXM4-80GB", True),
])
def test_f6_delivered_card_must_match_approved_stratum(tmp_path, monkeypatch, device, accepted):
    """Catch a CUDA-available-only gate, with an approved A100 control."""
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "seas_gpu")
    science_python = str(tmp_path / "science-python")
    # A real subprocess executes the production torch assertion against this
    # tiny module. Thus a fix inside the -c code is exercised, not stubbed out.
    (tmp_path / "torch.py").write_text(
        "class cuda:\n"
        "    @staticmethod\n    def is_available(): return True\n"
        f"    @staticmethod\n    def get_device_name(index): return {device!r}\n",
        encoding="utf-8",
    )
    actual_run = subprocess.run

    def host_command(command, **kwargs):
        if command[0] == science_python:
            return actual_run([sys.executable, *command[1:]], cwd=tmp_path, **kwargs)
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, SHA + "\n", "")
        if command[0] in {"hostname", "nvidia-smi"}:
            return subprocess.CompletedProcess(command, 0, device + "\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(job.subprocess, "run", host_command)
    if accepted:
        job.check_identity(tmp_path, SHA, science_python)
    else:
        with pytest.raises(job.JobError):
            job.check_identity(tmp_path, SHA, science_python)


@pytest.mark.parametrize("refuses", [False, True])
def test_c_receipt_refusal_controls_job_exit(job_env, monkeypatch, capsys, refuses):
    """Catch main returning 0 after the attached route's receipt refusal.

    ADAPTED for the F3 fix: TRAIN now refuses before dispatch unless
    ``--pipeline-check`` is passed (see test_f2's note); this control's own
    concern -- that a downstream receipt refusal still makes ``main``
    return non-zero -- is independent of that gate, so it opts in to reach
    dispatch. Also the control for the verifier's mutant (d) addition
    (run_job.py's main() swallowing an exception/refusal and exiting 0):
    the ``refuses=True`` arm must observe ``rc != 0``.
    """
    calls = []

    def activate(prepared, **kwargs):
        calls.append(prepared.task_id)
        if refuses:
            raise job.attached.HV2ReceiptRefusal("wrong-source-commit", "review sentinel refusal")

    monkeypatch.setattr(job.attached, "activate_h_v2_run", activate)
    rc = job.main(["train", "L1", "--pipeline-check"])
    assert calls == ["h-v2-train-pipeline-check-L1"]
    captured = capsys.readouterr()
    if refuses:
        assert rc != 0 and "review sentinel refusal" in captured.err
        assert ": accepted" not in captured.out
    else:
        assert rc == 0 and "h-v2-train-pipeline-check-L1: accepted" in captured.out


@pytest.mark.parametrize("stdout,rc,expected_calls", [
    ("111\n", 0, 2), ("111;cluster\n", 0, 2),
    (";cluster\n", 0, 1), ("111;cluster\n", 1, 1),
])
def test_c_job_id_parsing_and_failed_first_submit(tmp_path, stdout, rc, expected_calls):
    """Catch unstripped cluster suffixes or trusting stdout on a failed call."""
    calls = []

    def runner(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, rc, stdout, "submission refused" if rc else "")

    args = dict(run_root=tmp_path, checkout_root=tmp_path / "checkout",
                t_wall="08:00:00", e_wall="24:00:00", runner=runner, **_payload_kwargs())
    if expected_calls == 1:
        with pytest.raises(launch.LaunchError):
            launch.submit_lineage("L1", **args)
    else:
        assert launch.submit_lineage("L1", **args) == ("111", "111")
        assert "--dependency=afterok:111" in calls[1]
    assert len(calls) == expected_calls


def test_c_dry_run_shell_words_match_real_commands(tmp_path, capsys):
    """Catch omitted quoting/argv joining and accidental dry-run submission."""
    root = tmp_path / "spaces ' ; $(touch BAD)"
    args = dict(run_root=root, checkout_root=root / "checkout",
                t_wall="08:00:00", e_wall="24:00:00", **_payload_kwargs())

    def forbidden_runner(command):
        pytest.fail("dry run called its runner")

    launch.submit_lineage("L1", **args, dry_run=True, runner=forbidden_runner)
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 2
    dry = [shlex.split(line) for line in printed]
    actual = []

    def runner(command):
        actual.append(command)
        return subprocess.CompletedProcess(command, 0, "123\n", "")

    launch.submit_lineage("L1", **args, runner=runner)
    assert dry[0] == actual[0]
    assert [s for s in dry[1] if not s.startswith("--dependency=")] == [
        s for s in actual[1] if not s.startswith("--dependency=")
    ]
    assert "--dependency=afterok:123" in actual[1]


def test_c_resource_flags_follow_nondefault_config(tmp_path):
    """Catch hard-coded flags even when they equal today's default config."""
    resources = replace(launch.launch_config.RESOURCES, partition="review_partition",
                        constraint="review_card", gpus=2, nodes=2, ntasks=3,
                        cpus_per_task=7, mem="41G", no_requeue=False)
    command = launch.build_sbatch_command(
        "train", "L1", run_root=tmp_path, checkout_root=tmp_path / "checkout",
        time_value="01:00:00", dependency=None, resources=resources, **_payload_kwargs(),
    )
    expected = {"--partition=review_partition", "--constraint=review_card", "--gres=gpu:2",
                "--nodes=2", "--ntasks=3", "--cpus-per-task=7", "--mem=41G"}
    assert expected <= set(command)
    assert "--no-requeue" not in command
