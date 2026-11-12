"""Focused PR #675 review probes.

These tests are intentionally separate from the implementation suite.  They
cover comparator fail-closed behavior and enter ``run_eval_fanout(pack_k=2)``
through its public fan-out function.  The integration probe uses the public
``run_from_config`` route in each worker and is skipped unless the approved
Cannon science interpreter is supplied by the harness.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent


def _module(name: str):
    key = f"h_v2_pr675_r1_{name}"
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


job = _module("run_job")
eval_job = job.eval_job
protocol = eval_job.protocol

EVAL_SHA = "e" * 40
TRAIN_SHA = "t" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _checkpoint(root: Path, update: int) -> dict:
    path = root / f"step_{update:06d}"
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(f"model-{update}".encode())
    (path / "COMPLETE").write_bytes(b"")
    manifest = json.dumps({"completed_updates": update}).encode()
    (path / "manifest.json").write_bytes(manifest)
    return {
        "update": update,
        "path": str(path),
        "files": {name: _sha(path / name) for name in ("model.pt", "COMPLETE", "manifest.json")},
    }


def _train_receipt(root: Path, *, updates: tuple[int, ...] = (1,)) -> Path:
    """Make a valid-shaped receipt whose checkpoint bytes are independently hashable."""

    config, identity = protocol.pilot_training_config("gpu-route-g")
    config_path = root / "train-materialized.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    identity["materialized_config_sha256"] = _sha(config_path)
    receipt = {
        "schema": job.attached.H_V2_ACCEPTED_RECEIPT_SCHEMA,
        "payload": "gpu-route-g",
        "science_config_identity": identity,
        "segments": [
            {
                "source_commit": TRAIN_SHA,
                "resolved_config_sha256": _sha(config_path),
                "artifacts": {"resolved_config": {"path": str(config_path), "sha256": _sha(config_path)}},
            }
        ],
        "checkpoint_inventory": [_checkpoint(root / "checkpoints", update) for update in updates],
    }
    path = root / "attached" / "receipts" / "h-v2-train-gpu-route-g.h-v2.json"
    _json(path, receipt)
    return path


def _fake_completed_run(config: dict) -> dict:
    """Write the real producer-shaped files for controlled orchestration tests."""

    run_dir = Path(config["run"]["root"]) / config["run"]["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"git_commit": EVAL_SHA}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text('{"event":"completed"}\n', encoding="utf-8")
    trajectory = run_dir / "trajectory"
    trajectory.mkdir()
    csv = trajectory / "sampled_eval_table.csv"
    csv.write_text("local_energy\n-1\n0\n1\n", encoding="utf-8")
    (trajectory / "sampled_eval_table.metadata.json").write_text(
        json.dumps({"csv_sha256": _sha(csv)}), encoding="utf-8"
    )
    return {
        "resolved_config_sha256": _sha(run_dir / "resolved_config.yaml"),
        "trajectory_csv_sha256": _sha(csv),
        "artifacts": {
            "trajectory_csv": {"path": str(csv), "sha256": _sha(csv)},
            "trajectory_metadata": {
                "path": str(trajectory / "sampled_eval_table.metadata.json"),
                "sha256": _sha(trajectory / "sampled_eval_table.metadata.json"),
            },
            "metadata": {"path": str(run_dir / "metadata.json"), "sha256": _sha(run_dir / "metadata.json")},
            "status": {"path": str(run_dir / "status.json"), "sha256": _sha(run_dir / "status.json")},
            "resolved_config": {
                "path": str(run_dir / "resolved_config.yaml"),
                "sha256": _sha(run_dir / "resolved_config.yaml"),
            },
        },
    }


@pytest.mark.parametrize(
    "packed,single,tolerance",
    [
        ({"energy": -0.5, "variance": 0.1}, {"energy": -0.5, "variance": 0.1}, {"energy_abs": 1.0, "variance_relative": 1.0}),
        ({"seed": 7, "checkpoint": 1, "checkpoint_sha256": "ck", "lineage": "L1", "energy": float("nan"), "variance": 0.1}, {"seed": 7, "checkpoint": 1, "checkpoint_sha256": "ck", "lineage": "L1", "energy": -0.5, "variance": 0.1}, {"energy_abs": 1.0, "variance_relative": 1.0}),
        ({"seed": 7, "checkpoint": 1, "checkpoint_sha256": "ck", "lineage": "L1", "energy": -0.5, "variance": 0.1}, {"seed": 7, "checkpoint": 1, "checkpoint_sha256": "ck", "lineage": "L1", "energy": -0.5, "variance": 0.1}, {"energy_abs": float("nan"), "variance_relative": 1.0}),
    ],
    ids=["missing-identity", "nan-metric", "nan-tolerance"],
)
def test_packed_comparator_refuses_nonfinite_or_missing_inputs(packed, single, tolerance):
    """The comparator must not let NaN or two missing identity fields compare equal."""

    with pytest.raises(eval_job.EvalJobRefusal):
        eval_job.compare_packed_single_stream(packed, single, tolerance=tolerance)


def test_packed_comparator_accepts_finite_identical_pair() -> None:
    eval_job.compare_packed_single_stream(
        {"seed": 7, "checkpoint": 1, "checkpoint_sha256": "ck", "lineage": "L1", "energy": -0.5, "variance": 0.1},
        {"seed": 7, "checkpoint": 1, "checkpoint_sha256": "ck", "lineage": "L1", "energy": -0.5, "variance": 0.1},
        tolerance={"energy_abs": 0.0, "variance_relative": 0.0},
    )


def test_run_eval_fanout_pack_k2_publishes_overlapping_siblings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Enter the production k=2 branch and prove two worker calls overlap.

    This is a controlled orchestration probe; the separate integration test
    below records overlap of the actual child-process intervals.
    """

    _train_receipt(tmp_path)
    generations = [
        {"ordinal": 0, "update": 1, "stream": 0, "seed": 7001},
        {"ordinal": 1, "update": 1, "stream": 1, "seed": 7002},
    ]
    monkeypatch.setattr(eval_job, "gpu_route_g_fanout", lambda: generations)
    monkeypatch.setattr(eval_job, "probe_readout_trajectory_generator", lambda *_: False)
    monkeypatch.setattr(job, "check_identity", lambda *args, **kwargs: None)

    active = 0
    maximum = 0
    guard = threading.Lock()
    overlap_barrier = threading.Barrier(2)

    def run_generation_direct(**kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            overlap_barrier.wait(timeout=10)
            return _fake_completed_run(kwargs["config"])
        finally:
            with guard:
                active -= 1

    # Keep the production fan-out, snapshot, receipt, and task paths real;
    # this wrapper only supplies tiny controlled producer bytes.
    monkeypatch.setattr(eval_job, "run_generation_direct", run_generation_direct)
    job.run_eval_fanout(
        run_root=tmp_path,
        expected_commit=EVAL_SHA,
        science_python=sys.executable,
        payload="gpu-route-g",
        lineage="G",
        expected_train_commit=TRAIN_SHA,
        pack_k=2,
    )
    receipt = eval_job.read_eval_receipt(job._eval_receipt_path(tmp_path, "h-v2-eval-gpu-route-g"))
    assert maximum >= 2
    assert {g["seed"] for g in receipt["generations"]} == {7001, 7002}
    assert {g["task_id"] for g in receipt["generations"]} == {
        "h-v2-eval-gpu-route-g-ck000001-s0",
        "h-v2-eval-gpu-route-g-ck000001-s1",
    }
    for generation in receipt["generations"]:
        for artifact in generation["artifacts"].values():
            assert _sha(Path(artifact["path"])) == artifact["sha256"]


def test_run_eval_fanout_pack_k2_publishes_success_before_sibling_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI returns non-zero while retaining a successful sibling receipt."""

    _train_receipt(tmp_path)
    generations = [
        {"ordinal": 0, "update": 1, "stream": 0, "seed": 7001},
        {"ordinal": 1, "update": 1, "stream": 1, "seed": 7002},
    ]
    monkeypatch.setattr(eval_job, "gpu_route_g_fanout", lambda: generations)
    monkeypatch.setattr(eval_job, "probe_readout_trajectory_generator", lambda *_: False)
    def one_fails(**kwargs):
        if kwargs["config"]["runtime"]["seed"] == 7001:
            raise eval_job.EvalJobRefusal("injected-failure", "review probe")
        return _fake_completed_run(kwargs["config"])

    monkeypatch.setattr(eval_job, "run_generation_direct", one_fails)
    monkeypatch.setattr(job, "check_identity", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUN", str(tmp_path))
    monkeypatch.setenv("SHA", EVAL_SHA)
    monkeypatch.setenv("SCIENCE_PYTHON", sys.executable)
    rc = job.main(
        [
            "eval-fanout",
            "gpu-route-g",
            "--expected-train-commit",
            TRAIN_SHA,
            "--pack-k",
            "2",
        ]
    )
    assert rc != 0
    receipt = eval_job.read_eval_receipt(job._eval_receipt_path(tmp_path, "h-v2-eval-gpu-route-g"))
    assert len(receipt["generations"]) == 1
    assert receipt["generations"][0]["seed"] == 7002


@pytest.mark.integration
def test_run_eval_fanout_pack_k2_real_cpu_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run two tiny public ``run_from_config`` eval subprocesses concurrently."""

    science_python = os.environ.get("H_V2_SCIENCE_PYTHON")
    if not science_python:
        pytest.skip("named skip: H_V2_SCIENCE_PYTHON is unset")

    from omegaconf import OmegaConf
    from tpen.run import run_from_config

    train, _ = protocol.pilot_training_config("gpu-route-g")
    train["runtime"]["device"] = "cpu"
    train["run"].update(root=str(tmp_path), run_id="real-train", layout="flat")
    train["trainer"]["max_steps"] = 1
    train["sampler"].update(n_walkers=4, burn_in=1, n_steps=1)
    assert run_from_config(OmegaConf.create(train), command="pr675 real pair train", raise_exceptions=True) == 0
    checkpoint = next(tmp_path.rglob("checkpoints/step_*"))
    checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert checkpoint_manifest["completed_updates"] == 1

    checkout = HERE.parents[2]
    # run_eval_fanout passes this path as the child cwd; bind it to the exact
    # checkout so run_from_config can resolve the same Git identity as the
    # parent test, rather than running from an empty temporary directory.
    (tmp_path / "checkout").symlink_to(checkout, target_is_directory=True)
    expected_eval_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    original_science_evaluation_config = protocol.science_evaluation_config

    def eval_config(checkpoint_dir: Path, *, seed: int, enable_readout_trajectory: bool = False):
        config = original_science_evaluation_config(
            checkpoint_dir,
            seed=seed,
            root=tmp_path / "real-results",
            enable_readout_trajectory=enable_readout_trajectory,
        )
        config["runtime"]["device"] = "cpu"
        config["sampler"].update(n_walkers=4, burn_in=1, n_steps=1)
        config["run"]["run_id"] = f"real-{seed}"
        config["evaluator"]["tasks"][0]["generator"].update(n_draws=2, chunk_size=4)
        config["evaluator"]["tasks"][0]["summaries"][0]["max_samples"] = 8
        return config

    monkeypatch.setattr(protocol, "science_evaluation_config", eval_config)
    monkeypatch.setattr(eval_job, "probe_readout_trajectory_generator", lambda *_: False)
    monkeypatch.setattr(eval_job, "gpu_route_g_fanout", lambda: [
        {"ordinal": 0, "update": 1, "stream": 0, "seed": 7001},
        {"ordinal": 1, "update": 1, "stream": 1, "seed": 7002},
    ])
    # Use the real checkpoint and receipt shape; only the source commit field
    # is synthetic because this probe is about fan-out wiring, not TRAIN auth.
    receipt_path = _train_receipt(tmp_path, updates=(1,))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checkpoint_inventory"][0]["path"] = str(checkpoint)
    receipt["checkpoint_inventory"][0]["files"] = {
        name: _sha(checkpoint / name) for name in ("model.pt", "COMPLETE", "manifest.json")
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    popen_events: list[dict[str, float | int | None]] = []
    original_popen = eval_job.subprocess.Popen

    class TracedPopen(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            event: dict[str, float | int | None] = {"pid": self.pid, "start": time.monotonic(), "end": None}
            popen_events.append(event)
            self._pr675_event = event

        def communicate(self, *args, **kwargs):
            result = super().communicate(*args, **kwargs)
            self._pr675_event["end"] = time.monotonic()
            return result

    monkeypatch.setattr(eval_job.subprocess, "Popen", TracedPopen)
    job.run_eval_fanout(
        run_root=tmp_path,
        expected_commit=expected_eval_commit,
        science_python=science_python,
        payload="gpu-route-g",
        lineage="G",
        expected_train_commit=TRAIN_SHA,
        pack_k=2,
    )
    eval_receipt = eval_job.read_eval_receipt(job._eval_receipt_path(tmp_path, "h-v2-eval-gpu-route-g"))
    assert {g["seed"] for g in eval_receipt["generations"]} == {7001, 7002}
    assert all(Path(a["path"]).is_file() for g in eval_receipt["generations"] for a in g["artifacts"].values())
    assert len(popen_events) == 2
    assert len({event["pid"] for event in popen_events}) == 2
    assert all(event["end"] is not None for event in popen_events)
    assert max(float(event["start"]) for event in popen_events) < min(
        float(event["end"]) for event in popen_events
    )
    interval_path = Path(os.environ.get("PR675_PROCESS_INTERVAL_DIR", str(tmp_path))) / "real-popen-events.json"
    interval_path.parent.mkdir(parents=True, exist_ok=True)
    interval_path.write_text(
        json.dumps(popen_events, sort_keys=True, indent=2), encoding="utf-8"
    )
    for generation in eval_receipt["generations"]:
        resolved = yaml.safe_load(
            Path(generation["artifacts"]["resolved_config"]["path"]).read_text(encoding="utf-8")
        )
        assert resolved["runtime"]["seed"] == generation["seed"]
        for artifact in generation["artifacts"].values():
            assert _sha(Path(artifact["path"])) == artifact["sha256"]

