"""Focused controls for the timeout salvage inventory and pilot-only consumer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h_v2_salvage_helpers", HERE / "test_review_eval_job_r1.py")
assert spec is not None and spec.loader is not None
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)
job, eval_job, protocol = helpers.job, helpers.ev, helpers.protocol
attached = job.attached


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(
    root: Path,
    step: int,
    *,
    complete: bool = True,
    source_commit: str = "a" * 40,
    config_text: str = "payload: capacity-pilot-100k\n",
) -> Path:
    entry = root / f"step_{step:06d}"
    entry.mkdir(parents=True)
    files = {
        "model": "model.pt",
        "optimizer": "optimizer.pt",
        "trainer": "trainer.json",
        "sampler": "sampler.pt",
        "rng": "rng.pt",
        "resolved_config": "resolved_config.yaml",
    }
    (entry / "model.pt").write_bytes(f"model-{step}".encode())
    (entry / "optimizer.pt").write_bytes(f"optimizer-{step}".encode())
    (entry / "sampler.pt").write_bytes(f"sampler-{step}".encode())
    (entry / "rng.pt").write_bytes(f"rng-{step}".encode())
    (entry / "resolved_config.yaml").write_text(config_text, encoding="utf-8")
    (entry / "trainer.json").write_text(
        json.dumps({"completed_updates": step, "next_iteration": step}), encoding="utf-8"
    )
    hashes = {f"{logical}_sha256": _sha(entry / relative) for logical, relative in files.items()}
    (entry / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "tpen.checkpoint",
                "completed_updates": step,
                "next_iteration": step,
                "files": files,
                "hashes": hashes,
                "provenance": {"git_sha": source_commit},
            }
        ),
        encoding="utf-8",
    )
    if complete:
        (entry / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return entry


def _checkpoint_root(tmp_path: Path, payload: str = "capacity-pilot-100k") -> Path:
    return tmp_path / "checkpoints"


def test_salvage_excludes_incomplete_checkpoint_and_stops_at_first_gap(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)
    _write_checkpoint(root, 0)
    _write_checkpoint(root, 1000, complete=False)
    _write_checkpoint(root, 2000)
    inventory, _ = attached.build_salvage_inventory(root, expected_source_commit="a" * 40)
    assert [entry["update"] for entry in inventory] == [0]


def test_salvage_stops_at_first_complete_checkpoint_gap(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)
    _write_checkpoint(root, 0)
    _write_checkpoint(root, 1000)
    _write_checkpoint(root, 3000)
    inventory, _ = attached.build_salvage_inventory(root, expected_source_commit="a" * 40)
    assert [entry["update"] for entry in inventory] == [0, 1000]


def test_salvage_refuses_trainer_cursor_mismatch(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)
    entry = _write_checkpoint(root, 0)
    (entry / "trainer.json").write_text(
        json.dumps({"completed_updates": 0, "next_iteration": 1}), encoding="utf-8"
    )
    with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-cursor-mismatch"):
        attached.build_salvage_inventory(root, expected_source_commit="a" * 40)


def test_salvage_refuses_completed_updates_only_cursor_mismatch(tmp_path: Path) -> None:
    """Both trainer cursor fields must agree with the checkpoint step."""
    root = _checkpoint_root(tmp_path)
    entry = _write_checkpoint(root, 0)
    (entry / "trainer.json").write_text(
        json.dumps({"completed_updates": 1, "next_iteration": 0}), encoding="utf-8"
    )
    with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-cursor-mismatch"):
        attached.build_salvage_inventory(root, expected_source_commit="a" * 40)


def test_salvage_refuses_hash_mismatch(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)
    entry = _write_checkpoint(root, 0)
    (entry / "model.pt").write_bytes(b"tampered")
    with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-checkpoint-hash-mismatch"):
        attached.build_salvage_inventory(root, expected_source_commit="a" * 40)


def test_salvage_refuses_identity_commit_or_config_mismatch(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)
    _write_checkpoint(root, 0)
    _write_checkpoint(root, 1000, source_commit="b" * 40)
    with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-identity-mismatch"):
        attached.build_salvage_inventory(root)

    root = tmp_path / "config-mismatch"
    _write_checkpoint(root, 0)
    _write_checkpoint(root, 1000, config_text="payload: changed\n")
    with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-identity-mismatch"):
        attached.build_salvage_inventory(root, expected_source_commit="a" * 40)


def test_salvage_refuses_checkpoint_symlink_escape(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)
    entry = _write_checkpoint(root, 0)
    outside = tmp_path / "outside-model.pt"
    outside.write_bytes((entry / "model.pt").read_bytes())
    (entry / "model.pt").unlink()
    (entry / "model.pt").symlink_to(outside)
    with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-checkpoint-path"):
        attached.build_salvage_inventory(root, expected_source_commit="a" * 40)


def test_salvage_refuses_absolute_or_parent_manifest_member(tmp_path: Path) -> None:
    for index, relative in enumerate(("../outside-model.pt", str(tmp_path / "outside-model.pt"))):
        root = tmp_path / f"manifest-case-{index}" / "checkpoints"
        entry = _write_checkpoint(root, 0)
        manifest_path = entry / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["model"] = relative
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-checkpoint-path"):
            attached.build_salvage_inventory(root, expected_source_commit="a" * 40)


def test_salvage_refuses_symlinked_checkpoint_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-checkpoints"
    _write_checkpoint(real_root, 0)
    symlink_root = tmp_path / "checkpoints"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-checkpoint-root-symlink"):
        attached.build_salvage_inventory(symlink_root, expected_source_commit="a" * 40)


def test_salvage_refuses_symlinked_salvage_parent(tmp_path: Path) -> None:
    payload = "capacity-pilot-100k"
    checkpoint_root = (
        tmp_path
        / "attached"
        / "store"
        / f"h-v2-train-{payload}"
        / "groups"
        / "g"
        / "attempt"
        / "core"
        / "checkpoints"
    )
    _write_checkpoint(checkpoint_root, 0)
    outside = tmp_path / "outside-salvage"
    outside.mkdir()
    (tmp_path / "attached" / "salvage").symlink_to(outside, target_is_directory=True)
    with pytest.raises(attached.HV2ReceiptRefusal, match="salvage-parent-symlink"):
        attached.write_salvage_inventory(
            tmp_path,
            payload=payload,
            job_id="60007665",
            terminal_state="TIMEOUT",
            expected_source_commit="a" * 40,
        )


def test_salvage_writer_creates_new_atomic_subdirectory_without_touching_store(tmp_path: Path) -> None:
    payload = "capacity-pilot-100k"
    checkpoint_root = tmp_path / "attached" / "store" / f"h-v2-train-{payload}" / "groups" / "g" / "attempt" / "core" / "checkpoints"
    entry = _write_checkpoint(checkpoint_root, 0)
    sentinel = tmp_path / "attached" / "store" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("preserve", encoding="utf-8")
    before = {path: path.read_bytes() for path in entry.iterdir()}
    output_path, receipt = attached.write_salvage_inventory(
        tmp_path, payload=payload, job_id="60007665", terminal_state="TIMEOUT", expected_source_commit="a" * 40
    )
    assert output_path.parent.name == "salvage"
    assert receipt["source"] == "salvage-partial-inventory"
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert {path: path.read_bytes() for path in entry.iterdir()} == before


def test_salvage_writer_never_overwrites_an_existing_salvage_directory(tmp_path: Path) -> None:
    payload = "capacity-pilot-100k"
    checkpoint_root = (
        tmp_path
        / "attached"
        / "store"
        / f"h-v2-train-{payload}"
        / "groups"
        / "g"
        / "attempt"
        / "core"
        / "checkpoints"
    )
    _write_checkpoint(checkpoint_root, 0)
    salvage_dir = tmp_path / "attached" / "salvage"
    salvage_dir.mkdir(parents=True)
    sentinel = salvage_dir / "existing.json"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        attached.write_salvage_inventory(
            tmp_path,
            payload=payload,
            job_id="60007665",
            terminal_state="TIMEOUT",
            expected_source_commit="a" * 40,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_salvage_inventory_is_refused_for_science_payload(tmp_path: Path) -> None:
    with pytest.raises(job.JobError, match="refused for science"):
        job.run_eval_fanout(
            run_root=tmp_path,
            expected_commit="b" * 40,
            science_python=sys.executable,
            payload="science",
            lineage="L1",
            expected_train_commit="a" * 40,
            pilot_stage=None,
            salvage_inventory_path=tmp_path / "salvage.json",
        )


def test_salvage_consumer_refuses_incomplete_inventory_entry() -> None:
    with pytest.raises(eval_job.EvalJobRefusal, match="salvage-inventory-entry"):
        eval_job.verify_salvage_inventory_identity(
            {
                "source": "salvage-partial-inventory",
                "payload": "capacity-pilot-100k",
                "lineage": "P",
                "source_commit": "a" * 40,
                "inventory": [{"update": 0}],
            },
            expected_commit="a" * 40,
            expected_payload="capacity-pilot-100k",
            expected_lineage="P",
        )


def test_salvage_sel_window_follows_inventory_length(tmp_path: Path) -> None:
    generations = []
    for ordinal in range(7):
        path = tmp_path / f"trajectory-{ordinal}.csv"
        width = 1 if ordinal in (3, 4, 5) else 10
        path.write_text(f"local_energy\n{-width}\n0\n{width}\n", encoding="utf-8")
        generations.append(
            {
                "ordinal": ordinal,
                "stream": 0,
                "seed": protocol.pilot_eval_seed(ordinal, 0),
                "artifacts": {"trajectory_csv": {"path": str(path), "sha256": _sha(path)}},
            }
        )
    selection = eval_job.compute_pilot_selection(
        {"source": "salvage-partial-inventory", "ordinal_count": 7, "generations": generations},
        payload="capacity-pilot-100k",
    )
    assert selection["c1"] == 4
    record = eval_job.build_selection_record(
        c1=4,
        triple_means=selection["triple_means"],
        b1_eval_receipt_sha256="c" * 64,
        payload="capacity-pilot-100k",
        ordinal_count=7,
    )
    assert record["interior_minimum"] is True
    assert eval_job.verify_selection_record(
        record, b1_eval_receipt_sha256="c" * 64, payload="capacity-pilot-100k", ordinal_count=7
    ) == 4


def test_salvage_sel_refuses_inventory_length_7_with_only_six_b1_generations(tmp_path: Path) -> None:
    generations = []
    for ordinal in range(6):
        path = tmp_path / f"trajectory-{ordinal}.csv"
        path.write_text("local_energy\n0\n", encoding="utf-8")
        generations.append(
            {
                "ordinal": ordinal,
                "stream": 0,
                "seed": protocol.pilot_eval_seed(ordinal, 0),
                "artifacts": {"trajectory_csv": {"path": str(path), "sha256": _sha(path)}},
            }
        )
    with pytest.raises(eval_job.EvalJobRefusal, match="sel-incomplete-stage1"):
        eval_job.compute_pilot_selection(
            {"source": "salvage-partial-inventory", "ordinal_count": 7, "generations": generations},
            payload="capacity-pilot-100k",
        )


@pytest.mark.parametrize("ordinal_count", [1, 4])
def test_salvage_sel_refuses_inventory_too_short_for_center_window(ordinal_count: int) -> None:
    with pytest.raises(eval_job.EvalJobRefusal, match="sel-inventory-too-short"):
        eval_job.compute_pilot_selection(
            {"source": "salvage-partial-inventory", "ordinal_count": ordinal_count, "generations": []},
            payload="capacity-pilot-100k",
        )


def test_salvage_default_100k_fanout_still_covers_101_ordinals() -> None:
    generations = eval_job.pilot_fanout("1", payload="capacity-pilot-100k")
    assert len(generations) == 101
    assert generations[0]["ordinal"] == 0
    assert generations[-1]["ordinal"] == 100
    assert generations[-1]["update"] == 100_000


def test_salvage_fanout_and_eval_receipt_preserve_source_binding(tmp_path: Path) -> None:
    generations = eval_job.pilot_fanout("1", payload="capacity-pilot-100k", ordinal_count=7)
    assert len(generations) == 7
    receipt = eval_job.build_eval_receipt(
        eval_commit="c" * 40,
        train_commit="a" * 40,
        payload="capacity-pilot-100k",
        train_receipt_path=tmp_path / "salvage.json",
        train_receipt_sha256="a" * 64,
        generations=generations,
        readout_generator_enabled=False,
        source="salvage-partial-inventory",
        source_sha256="b" * 64,
        ordinal_count=7,
    )
    assert receipt["source"] == "salvage-partial-inventory"
    assert receipt["source_sha256"] == "b" * 64
    assert receipt["ordinal_count"] == 7


def test_default_eval_receipt_shape_has_no_salvage_length(tmp_path: Path) -> None:
    receipt = eval_job.build_eval_receipt(
        eval_commit="c" * 40,
        train_commit="a" * 40,
        payload="capacity-pilot-100k",
        train_receipt_path=tmp_path / "train.json",
        train_receipt_sha256="a" * 64,
        generations=[],
        readout_generator_enabled=False,
    )
    assert "ordinal_count" not in receipt
