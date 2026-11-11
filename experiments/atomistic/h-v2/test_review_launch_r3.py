"""PR619 T1 round-3 probes (K1-K5): completeness fixes past round 2.

Round 2 (G1-G3) left five defects: the submit-side ``launch.py`` and the
generated job wrapper both crash outside pytest (K1: ``experiments`` is a
namespace package that resolves only with the repository root on
``sys.path``, which pytest supplies for free and nothing else does); EVAL's
receipt lookup disagreed with the name TRAIN actually wrote under
``pipeline-check`` (K2); a dry run without ``T_WALL``/``E_WALL`` fell
through a dead placeholder path instead of refusing cleanly (K3, covered by
a production code change with no new test needed -- :mod:`test_launch`'s
``test_main_requires_t_wall_and_e_wall_when_not_a_dry_run`` already covers
the always-required behaviour); and no test drove ``check_identity``'s
SHA-mismatch raise for real (K5, every :func:`run_job.run` test bypasses it
via ``monkeypatch``).
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
REPO_ROOT = HERE.parents[2]
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


# --- K1: real import boundary outside pytest --------------------------------


def test_k1_launch_py_dry_run_gets_past_import_outside_pytest(tmp_path):
    """``launch.py``, run as a plain script from a stripped environment and a
    cwd that is not the repository root, must get past ``run_job.py``'s (and
    therefore ``attached.py``'s) module-level import of
    ``experiments.toolkit.*`` -- not merely under pytest, which supplies the
    repository root on ``sys.path`` for free via ``pythonpath = ["."]``.
    Fails at e8f47f7e with ``ModuleNotFoundError: No module named
    'experiments'`` (confirmed on Cannon before this fix); passes once
    ``launch.py`` puts its own checkout root on ``sys.path``.
    """

    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "T_WALL": "08:00:00",
        "E_WALL": "24:00:00",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "launch.py"),
            "--run-root", str(tmp_path / "runs" / "h-v2" / "pipeline-check"),
            "--run-root-prefix", str(tmp_path / "runs"),
            "--commit", SHA,
            "--payload", "pipeline-check",
            "--science-python", sys.executable,
            "--executor-python", sys.executable,
            "--dry-run",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("sbatch --parsable") == 6, result.stdout


def test_k1_wrapper_body_gets_past_import_outside_pytest(tmp_path):
    """The real generated wrapper body (:func:`launch._wrap_command`), run
    standalone with a bare ``env -i``-style environment (no ``PYTHONPATH``,
    matching what Slurm hands a submitted job) and the real ``attached.py``
    (not a stub), must get past ``run_job.py``'s module-level import. Fails
    at e8f47f7e with the same ``ModuleNotFoundError``; passes once the
    wrapper sets its own ``PYTHONPATH`` to the checkout root.
    """

    checkout = tmp_path / "checkout"
    checkout.symlink_to(REPO_ROOT)
    run_root = tmp_path / "run"
    run_root.mkdir()
    wrap = launch._wrap_command(
        "train",
        "L1",
        checkout_root=checkout,
        science_python=sys.executable,
        executor_python=sys.executable,
        payload="pipeline-check",
    )
    script = tmp_path / "wrapper.sh"
    script.write_text("#!/bin/sh\n" + wrap + "\n", encoding="utf-8")
    script.chmod(0o700)
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "RUN": str(run_root),
        "SHA": "0" * 40,  # deliberately wrong: still proves past-import, no GPU needed
    }
    result = subprocess.run(
        [str(script)], env=env, capture_output=True, text=True, timeout=60
    )
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert "ModuleNotFoundError" not in result.stdout, result.stdout
    assert "[h-v2] job=" in result.stdout, (result.stdout, result.stderr)


# --- K2: EVAL's receipt lookup must agree with TRAIN's real receipt name ----


def test_k2_pipeline_check_eval_reaches_unsupported_refusal_not_missing_receipt(
    tmp_path, monkeypatch
):
    """Feed EVAL the real TRAIN receipt name (consistently labelled under
    ``pipeline-check``) and confirm it reaches the documented
    ``h-v2-eval-route-unsupported`` refusal, not ``no TRAIN accepted
    receipt`` -- the bug the round-2 label change introduced
    (``h-v2-train-pipeline-check-<lineage>`` written vs. ``h-v2-train-<lineage>``
    read).
    """

    for key, value in {
        "RUN": str(tmp_path), "SHA": SHA, "SCIENCE_PYTHON": sys.executable,
        "EXECUTOR_PY": sys.executable, "SLURM_JOB_ID": "1234",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(job, "check_identity", lambda *args: None)

    task_id = job._train_task_id("L1", pipeline_check=True)
    receipts_dir = tmp_path / "attached" / "receipts"
    receipts_dir.mkdir(parents=True)
    (receipts_dir / f"{task_id}.h-v2.json").write_text(
        json.dumps({"segments": [{"checkpoint_dir": str(tmp_path / "ckpt")}]}),
        encoding="utf-8",
    )

    with pytest.raises(job.attached.ActivationRefusal) as excinfo:
        job.run("eval", "L1", pipeline_check=True)
    assert excinfo.value.code == job.attached.EVAL_ROUTE_UNSUPPORTED_CODE


def test_k2_checkpoint_dir_for_eval_agrees_with_train_task_id(tmp_path):
    """:func:`run_job.checkpoint_dir_for_eval` must read the receipt name
    :func:`run_job._train_task_id` builds, for both the default and the
    ``pipeline_check`` namespace -- the two must never drift apart again.
    """

    receipts_dir = tmp_path / "attached" / "receipts"
    receipts_dir.mkdir(parents=True)
    checkpoint = tmp_path / "ckpt"
    for pipeline_check in (False, True):
        task_id = job._train_task_id("L1", pipeline_check=pipeline_check)
        payload = {"segments": [{"checkpoint_dir": str(checkpoint)}]}
        if not pipeline_check:
            # TO baff945c review r1 (F1): a pipeline_check=False lookup now
            # requires payload="science" plus a complete config identity.
            payload["payload"] = "science"
            payload["science_config_identity"] = {
                "lineage": "L1", "lineage_index": 0,
                "overlay_sha256": "a" * 64, "base_train_sha256": "b" * 64,
                "contract_inputs_sha256": "c" * 64, "materialized_config_sha256": "d" * 64,
            }
        (receipts_dir / f"{task_id}.h-v2.json").write_text(json.dumps(payload), encoding="utf-8")
        assert job.checkpoint_dir_for_eval(
            tmp_path, "L1", pipeline_check=pipeline_check
        ) == checkpoint


# --- K5: check_identity's SHA-mismatch raise must be exercised for real -----


@pytest.mark.parametrize("delivered_sha,accepted", [("b" * 40, False), ("a" * 40, True)])
def test_k5_check_identity_enforces_checkout_sha_match(
    tmp_path, monkeypatch, delivered_sha, accepted
):
    """No existing test drives ``check_identity``'s checkout-HEAD-vs-expected
    -commit raise for real: every :func:`run_job.run` test bypasses it via
    ``monkeypatch.setattr(job, "check_identity", lambda *args: None)``. Kills
    a mutant that drops the ``if head.returncode != 0 or head.stdout.strip()
    != expected_commit: raise JobError(...)`` block. Includes a matching-SHA
    control so the mismatch case is not merely a tautology.
    """

    expected = "a" * 40
    device = "NVIDIA A100-SXM4-80GB"
    science_python = str(tmp_path / "science-python")
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
            return subprocess.CompletedProcess(command, 0, delivered_sha + "\n", "")
        if command[0] in {"hostname", "nvidia-smi"}:
            return subprocess.CompletedProcess(command, 0, device + "\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(job.subprocess, "run", host_command)
    if accepted:
        job.check_identity(tmp_path, expected, science_python)
    else:
        with pytest.raises(job.JobError, match="does not match expected commit"):
            job.check_identity(tmp_path, expected, science_python)
