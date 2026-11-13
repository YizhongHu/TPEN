"""PR687 T1 probe for science roster propagation through the receipt path.

The test keeps numerical execution fake at the existing subprocess boundary.
The production fan-out, checkpoint snapshotting, generation records, and eval
receipt writer remain real. The final ``pytest.raises`` is an explicit
negative control: a disconnected stale roster is observed to fail the same
receipt-cardinality assertion as the intended consumer check.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
EVAL_SHA = "a" * 40


def _module(name: str):
    key = f"h_v2_pr687_r1_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


job = _module("run_job")
eval_job = job.eval_job
protocol = eval_job.protocol


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(root: Path, update: int) -> dict[str, object]:
    path = root / f"step_{update:06d}"
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(f"model-{update}".encode())
    (path / "COMPLETE").write_bytes(b"")
    (path / "manifest.json").write_text(json.dumps({"completed_updates": update}), encoding="utf-8")
    return {
        "update": update,
        "path": str(path),
        "files": {name: _sha(path / name) for name in ("model.pt", "COMPLETE", "manifest.json")},
    }


def _science_train_receipt(root: Path, *, commit: str) -> dict[str, object]:
    _, identity = protocol.science_training_config("L1", 0)
    materialized = root / "materialized.yaml"
    materialized.write_text("science-materialized\n", encoding="utf-8")
    identity["materialized_config_sha256"] = _sha(materialized)
    resolved = root / "resolved.yaml"
    resolved.write_text("science-resolved\n", encoding="utf-8")
    resolved_sha = _sha(resolved)
    inventory = [
        _checkpoint(root / "checkpoints", update)
        for update in protocol.science_evaluated_checkpoint_updates()
    ]
    return {
        "payload": "science",
        "science_config_identity": identity,
        "segments": [{
            "source_commit": commit,
            "resolved_config_sha256": resolved_sha,
            "artifacts": {"resolved_config": {"path": str(resolved), "sha256": resolved_sha}},
        }],
        "checkpoint_inventory": inventory,
    }


def _write_train_receipt(root: Path, *, commit: str) -> None:
    path = root / "attached" / "receipts" / "h-v2-train-L1.h-v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_science_train_receipt(root, commit=commit)), encoding="utf-8")


def _fake_numerical_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Produce real-shaped evaluation artifacts without running numerics."""

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        config = yaml.safe_load(Path(argv[-1]).read_text(encoding="utf-8"))
        run_dir = Path(config["run"]["root"]) / config["run"]["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        (run_dir / "metadata.json").write_text(json.dumps({"git_commit": EVAL_SHA}), encoding="utf-8")
        for task in config["evaluator"]["tasks"]:
            relative = task["output_dir"].removeprefix("${run.dir}/")
            output_dir = run_dir / relative
            output_dir.mkdir(parents=True, exist_ok=True)
            csv = output_dir / "sampled_eval_table.csv"
            csv.write_text("local_energy\n-1\n0\n1\n", encoding="utf-8")
            (output_dir / "sampled_eval_table.metadata.json").write_text(
                json.dumps({"csv_sha256": _sha(csv)}), encoding="utf-8"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(eval_job, "probe_readout_trajectory_generator", lambda *_: False)
    monkeypatch.setattr(eval_job.subprocess, "run", fake_run)


def _receipt(root: Path) -> dict[str, object]:
    path = job._eval_receipt_path(root, "h-v2-eval-science-L1")
    return eval_job.read_eval_receipt(path)


def _assert_receipt_follows_roster(receipt: dict[str, object], updates: tuple[int, ...]) -> None:
    generations = receipt["generations"]
    assert len(generations) == len(updates) * 4
    assert [(row["ordinal"], row["update"], row["stream"]) for row in generations] == [
        (ordinal, update, stream)
        for ordinal, update in enumerate(updates)
        for stream in range(4)
    ]


def test_science_fanout_receipt_tracks_37x4_roster_and_disconnect_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real fan-out/receipt propagation follows an altered in-horizon roster."""

    original_updates = protocol.science_evaluated_checkpoint_updates()
    assert len(original_updates) == 37
    _fake_numerical_execution(monkeypatch)

    baseline_root = tmp_path / "baseline"
    _write_train_receipt(baseline_root, commit=EVAL_SHA)
    job.run_eval_fanout(
        run_root=baseline_root,
        expected_commit=EVAL_SHA,
        science_python=sys.executable,
        payload="science",
        lineage="L1",
        expected_train_commit=EVAL_SHA,
    )
    baseline = _receipt(baseline_root)
    _assert_receipt_follows_roster(baseline, original_updates)
    assert baseline["generations"][-1]["ordinal"] == 36
    assert baseline["generations"][-1]["update"] == 100000

    changed_updates = (*original_updates[:-1], 97500)
    changed_contract = json.loads(protocol.CONTRACT_INPUTS_PATH.read_text(encoding="utf-8"))
    changed_contract["measurement"]["evaluated_checkpoints"]["updates"] = list(changed_updates)
    changed_contract_path = tmp_path / "changed-contract.json"
    changed_contract_path.write_text(json.dumps(changed_contract), encoding="utf-8")
    monkeypatch.setattr(protocol, "CONTRACT_INPUTS_PATH", changed_contract_path)

    changed_root = tmp_path / "changed"
    _write_train_receipt(changed_root, commit=EVAL_SHA)
    job.run_eval_fanout(
        run_root=changed_root,
        expected_commit=EVAL_SHA,
        science_python=sys.executable,
        payload="science",
        lineage="L1",
        expected_train_commit=EVAL_SHA,
    )
    changed = _receipt(changed_root)
    _assert_receipt_follows_roster(changed, changed_updates)
    assert changed["generations"][-1]["ordinal"] == 36
    assert changed["generations"][-1]["update"] == 97500

    # Negative control: disconnect the consumer from the contract and feed it
    # a stale 27-checkpoint roster. The intended output assertion is observed
    # to fail, so a hard-coded or disconnected fan-out cannot look green.
    stale_updates = original_updates[:27]
    stale_generations = tuple(
        {
            "ordinal": ordinal,
            "update": update,
            "stream": stream,
            "seed": protocol.science_eval_seed(0, ordinal, stream),
        }
        for ordinal, update in enumerate(stale_updates)
        for stream in range(4)
    )
    monkeypatch.setattr(eval_job, "science_fanout", lambda *, lineage_index: stale_generations)
    disconnected_root = tmp_path / "disconnected"
    _write_train_receipt(disconnected_root, commit=EVAL_SHA)
    job.run_eval_fanout(
        run_root=disconnected_root,
        expected_commit=EVAL_SHA,
        science_python=sys.executable,
        payload="science",
        lineage="L1",
        expected_train_commit=EVAL_SHA,
    )
    with pytest.raises(AssertionError):
        _assert_receipt_follows_roster(_receipt(disconnected_root), changed_updates)
