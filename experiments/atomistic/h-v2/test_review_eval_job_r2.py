"""PR 641 T2 delta probes at 0ea7ebaa; tests only, no product repairs.

Use the round-1 producer-shaped fixtures, the real receipt writer/reader and
the production selection commands. Synthetic energy grids isolate selection
algebra; they are not science-scale completeness evidence.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("h_v2_review_r1_helpers", HERE / "test_review_eval_job_r1.py")
assert _spec is not None and _spec.loader is not None
helpers = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = helpers
_spec.loader.exec_module(helpers)
job, ev, protocol, launch = helpers.job, helpers.ev, helpers.protocol, helpers.launch


@pytest.fixture(autouse=True)
def shared_modules(monkeypatch):
    monkeypatch.setattr(job, "protocol", protocol)


def _final_inputs(root):
    """Write B1, run real SEL, and write a bound B2 through its real producer."""
    b1_path, b1 = helpers._b1(root)
    job.run_pilot_sel(root)
    selection_path = root / "attached/receipts/h-v2-eval-capacity-pilot-selection.json"
    selection = json.loads(selection_path.read_text())
    selection["selection_record_sha256"] = helpers._sha(selection_path)
    generations = []
    for spec in ev.pilot_fanout("2", c1=selection["c1"]):
        width = 1 if spec["ordinal"] in (9, 10, 11) else 10
        path = root / f"b2-{spec['ordinal']}-{spec['stream']}.csv"
        path.write_text(f"local_energy\n{-width}\n0\n{width}\n")
        generations.append({**spec, "artifacts": {"trajectory_csv": {
            "path": str(path), "sha256": helpers._sha(path)}}})
    b2_path = job._eval_receipt_path(root, "h-v2-eval-capacity-pilot-stage2")
    b2 = ev.write_eval_receipt(
        b2_path, eval_commit=helpers.EVAL_SHA, train_commit=helpers.TRAIN_SHA,
        train_receipt_path=root / "train.json", train_receipt_sha256=b1["train_receipt_sha256"],
        payload="capacity-pilot", generations=generations,
        readout_generator_enabled=True, selection=selection,
    )
    return b1_path, b1, b2_path, b2


def _final_record(root):
    job.run_pilot_final_sel(root)
    return json.loads((root / "attached/receipts/h-v2-eval-capacity-pilot-final-selection.json").read_text())


def test_final_sel_valid_producer_consumer_control(tmp_path):
    """Passing control for the common receipt fixture and numerical oracle."""
    _final_inputs(tmp_path)
    record = _final_record(tmp_path)
    assert record["c1"] == record["c_star"] == 10
    assert record["interior_minimum"] is True
    assert record["pooled01_w_by_ordinal"] == {"8": 10, "9": 1, "10": 1, "11": 1, "12": 10}


def test_final_sel_centers_each_stream_before_pooling(tmp_path):
    """D1: changing one stream's location must not change its dispersion."""
    _, _, b2_path, b2 = _final_inputs(tmp_path)
    before = _final_record(tmp_path)
    for generation in b2["generations"]:
        if generation["stream"] == 1 and generation["ordinal"] in (9, 10, 11):
            artifact = generation["artifacts"]["trajectory_csv"]
            path = Path(artifact["path"])
            path.write_text("local_energy\n99\n100\n101\n")
            artifact["sha256"] = helpers._sha(path)
    helpers._json(b2_path, b2)
    after = _final_record(tmp_path)
    assert after["pooled01_w_by_ordinal"] == before["pooled01_w_by_ordinal"], (
        "final SEL centered the pooled energies instead of each stream independently"
    )
    assert after["c_star"] == 10


@pytest.mark.parametrize("fault", ["missing-stream1", "changed-csv", "wrong-selection", "wrong-train"])
def test_final_sel_refuses_unbound_or_partial_b2(tmp_path, fault):
    """D2: final SEL must validate both inputs, not merely each receipt schema."""
    _, _, b2_path, b2 = _final_inputs(tmp_path)
    assert _final_record(tmp_path)["c_star"] == 10
    if fault == "missing-stream1":
        b2["generations"] = [g for g in b2["generations"] if not (g["ordinal"] == 8 and g["stream"] == 1)]
    elif fault == "changed-csv":
        artifact = next(g for g in b2["generations"] if g["ordinal"] == 9 and g["stream"] == 1)["artifacts"]["trajectory_csv"]
        Path(artifact["path"]).write_text("local_energy\n-100\n0\n100\n")
    elif fault == "wrong-selection":
        b2["selection"]["b1_eval_receipt_sha256"] = "0" * 64
    else:
        b2["train_receipt_sha256"] = "0" * 64
    helpers._json(b2_path, b2)
    with pytest.raises((job.JobError, ev.EvalJobRefusal, ValueError)):
        job.run_pilot_final_sel(tmp_path)


@pytest.mark.parametrize("fault", ["parent-path", "wrong-update", "existing-symlink"])
def test_verified_snapshot_refuses_escape_or_wrong_generation(tmp_path, fault):
    """D3: verified bytes must remain inside a fresh snapshot at the named update."""
    entry = helpers._checkpoint(tmp_path / "source", 1000)
    destination = tmp_path / "snapshots" / "generation"
    if fault == "parent-path":
        outside = Path(entry["path"]).parent / "escaped.bin"
        outside.write_bytes(b"inventoried path escaping its checkpoint")
        entry["files"]["../escaped.bin"] = helpers._sha(outside)
    elif fault == "wrong-update":
        manifest = Path(entry["path"]) / "manifest.json"
        content = json.loads(manifest.read_text())
        content.update(completed_updates=2000, next_iteration=2001)
        helpers._json(manifest, content)
        entry["files"]["manifest.json"] = helpers._sha(manifest)
    else:
        destination.mkdir(parents=True)
        victim = tmp_path / "unrelated-model.pt"
        victim.write_bytes(b"must remain untouched")
        (destination / "model.pt").symlink_to(victim)
    with pytest.raises((ev.EvalJobRefusal, ValueError, FileExistsError)):
        ev.resolve_checkpoint({"checkpoint_inventory": [entry]}, update=1000, snapshot_dir=destination)


def test_snapshot_survives_source_swap_control(tmp_path):
    """Passing F2/K3 control: source mutation after copying cannot change the snapshot."""
    entry = helpers._checkpoint(tmp_path / "source", 1000)
    snapshot = ev.resolve_checkpoint({"checkpoint_inventory": [entry]}, update=1000,
                                     snapshot_dir=tmp_path / "snapshot")
    (Path(entry["path"]) / "model.pt").write_bytes(b"replacement")
    assert helpers._sha(snapshot / "model.pt") == entry["files"]["model.pt"]


def test_eval_identity_checks_resolved_artifact_bytes(tmp_path):
    """D4: two equal digest strings are insufficient when the bound bytes changed."""
    receipt, _ = helpers._train(tmp_path, updates=(1000,))
    kwargs = dict(expected_commit=helpers.TRAIN_SHA, expected_lineage="P", expected_payload="capacity-pilot")
    ev.verify_train_receipt_identity(receipt, **kwargs)
    artifact = receipt["segments"][0]["artifacts"]["resolved_config"]
    Path(artifact["path"]).write_text("model: changed\n")
    with pytest.raises((ev.EvalJobRefusal, ValueError)):
        ev.verify_train_receipt_identity(receipt, **kwargs)


def test_real_train_receipt_passes_eval_identity_gate(tmp_path, monkeypatch):
    """D4 producer/consumer: input YAML and resolved YAML have different hashes.

    Reuse the real one-update CPU TRAIN -> checkpoint -> EVAL smoke, but
    exercise the production identity gate that the round-1 smoke skipped.
    No mocked receipt or hash strings are involved in this test.
    """
    original = ev.load_train_receipt
    seen = []

    def validate_produced_receipt(path):
        receipt, digest = original(path)
        seen.append(receipt)
        commit = receipt["segments"][0]["source_commit"]
        ev.verify_train_receipt_identity(receipt, expected_commit=commit,
                                         expected_lineage="G", expected_payload="gpu-route-g")
        return receipt, digest

    monkeypatch.setattr(ev, "load_train_receipt", validate_produced_receipt)
    helpers.test_real_eval_restores_p_and_e_checkpoints(tmp_path, monkeypatch, False)
    assert seen, "the real TRAIN producer never reached the EVAL identity gate"


def test_historical_train_identity_is_not_rebound_to_current_ci(tmp_path, monkeypatch):
    """D4: a newer checkout's annotation must not invalidate authenticated old inputs."""
    receipt, _ = helpers._train(tmp_path, updates=(1000,))
    kwargs = dict(expected_commit=helpers.TRAIN_SHA, expected_lineage="P", expected_payload="capacity-pilot")
    ev.verify_train_receipt_identity(receipt, **kwargs)
    newer_ci = tmp_path / "newer-contract-inputs.json"
    newer_ci.write_bytes(protocol.CONTRACT_INPUTS_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(protocol, "CONTRACT_INPUTS_PATH", newer_ci)
    ev.verify_train_receipt_identity(receipt, **kwargs)


@pytest.mark.parametrize("fault", ["duplicate", "seed", "nonfinite"])
def test_sel_refuses_invalid_run_identity_or_values(tmp_path, fault):
    """D5: all ordinal labels being present is weaker than an exact valid B1 set."""
    _, receipt = helpers._b1(tmp_path)
    assert ev.compute_pilot_selection(receipt)["c1"] == 10
    if fault == "duplicate":
        receipt["generations"].append(copy.deepcopy(receipt["generations"][10]))
    elif fault == "seed":
        receipt["generations"][10]["seed"] += 1
    else:
        artifact = receipt["generations"][10]["artifacts"]["trajectory_csv"]
        path = Path(artifact["path"])
        path.write_text("local_energy\nnan\nnan\nnan\n")
        artifact["sha256"] = helpers._sha(path)
    with pytest.raises((ev.EvalJobRefusal, ValueError)):
        ev.compute_pilot_selection(receipt)


def test_sel_binds_the_receipt_bytes_it_parsed(tmp_path, monkeypatch):
    """D5: atomic receipt replacement during SEL must refuse or bind the old bytes."""
    path, receipt = helpers._b1(tmp_path)
    old_sha = helpers._sha(path)
    original = ev.compute_pilot_selection

    def replace_after_read(parsed):
        result = original(parsed)
        replacement = copy.deepcopy(receipt)
        replacement["eval_commit"] = "d" * 40
        helpers._json(path, replacement)
        return result

    monkeypatch.setattr(ev, "compute_pilot_selection", replace_after_read)
    try:
        job.run_pilot_sel(tmp_path)
    except (job.JobError, ev.EvalJobRefusal, ValueError):
        return
    selection = json.loads((tmp_path / "attached/receipts/h-v2-eval-capacity-pilot-selection.json").read_text())
    assert selection["b1_eval_receipt_sha256"] == old_sha


@pytest.mark.parametrize("stage", ["train", "eval", "both"])
def test_science_launch_requires_root_admission_on_every_path(tmp_path, monkeypatch, stage):
    """D6: omission of the guard input must not authorize a disposable run root."""
    monkeypatch.delenv(launch.launch_config.PILOT_RUN_ROOT_PREFIX_ENV, raising=False)
    monkeypatch.setenv("T_WALL", "08:00:00")
    monkeypatch.setenv("E_WALL", "24:00:00")
    submitted = []
    monkeypatch.setattr(launch, "_default_runner", lambda argv: (
        submitted.append(argv) or subprocess.CompletedProcess(argv, 0, "123\n", "")))
    args = ["--run-root", str(tmp_path / "ephemeral"), "--commit", helpers.EVAL_SHA,
            "--payload", "science", "--stage", stage, "--science-python", sys.executable,
            "--executor-python", sys.executable, "--dry-run"]
    if stage == "eval":
        args += ["--time", "24:00:00", "--expected-train-commit", helpers.TRAIN_SHA]
    with pytest.raises(launch.LaunchError):
        launch.main(args)
    assert submitted == []


def test_science_dry_run_is_exactly_six_lines(tmp_path, monkeypatch, capsys):
    """D7: count all stdout lines, not just occurrences of sbatch."""
    monkeypatch.setenv("T_WALL", "08:00:00")
    monkeypatch.setenv("E_WALL", "24:00:00")
    launch.main(["--run-root", str(tmp_path / "runs/h-v2/science"),
                 "--run-root-prefix", str(tmp_path / "runs"), "--commit", helpers.EVAL_SHA,
                 "--payload", "science", "--science-python", sys.executable,
                 "--executor-python", sys.executable, "--dry-run"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6 and all(line.startswith("sbatch ") for line in lines)


def test_science_seed_runtime_reads_or_refuses_changed_ci(tmp_path, monkeypatch):
    """D8: runtime must consume CI or fail closed, never silently ignore it."""
    data = json.loads(protocol.CONTRACT_INPUTS_PATH.read_text())
    formula = data["measurement"]["seed_roster"]["evaluation_stream_seed_formula"]
    assert "1000000" in formula
    data["measurement"]["seed_roster"]["evaluation_stream_seed_formula"] = formula.replace("1000000", "2000000", 1)
    changed = helpers._json(tmp_path / "contract-inputs.json", data)
    monkeypatch.setattr(protocol, "CONTRACT_INPUTS_PATH", changed)
    try:
        actual = protocol.science_eval_seed(0, 0, 0)
    except ValueError:
        return
    assert actual == 2_000_000, "runtime ignored the formula in CI"
