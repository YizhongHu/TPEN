"""Public restart primitives preserve publication authority and exclusivity."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import CheckpointRef, TrainResume, save_checkpoint
from tpen.execution import (
    PredecessorDiscoveryRefusal,
    ReceiptExistsError,
    create_json_receipt,
    discover_published_checkpoint,
)
from tpen.execution import invocation


def _save(root: Path, step: int) -> CheckpointRef:
    """Produce real full-state checkpoints without a training run."""
    model = torch.nn.Linear(1, 1, bias=False)
    context = SimpleNamespace(
        cfg=OmegaConf.create({"trainer": {"max_steps": 10}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id="cx08", extra={}),
        run_dir=root.parent,
    )
    directory = save_checkpoint(
        output_dir=root,
        next_iteration=step,
        completed_updates=step,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        trainer=SimpleNamespace(state_dict=lambda: {"next_iteration": step, "completed_updates": step}),
        sampler=SimpleNamespace(mcmc_state_dict=lambda: {"walker": step}),
        payload=TrainResume(),
        context=context,
    )
    return CheckpointRef.from_directory(directory)


def _snapshot(root: Path) -> dict:
    """Capture listing, mtimes and bytes, including the root directory itself."""
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in [root, *root.rglob("*")]
    }


def test_discovery_empty_is_read_only(tmp_path: Path) -> None:
    before = _snapshot(tmp_path)
    assert discover_published_checkpoint(tmp_path) is None
    assert _snapshot(tmp_path) == before


def test_discovery_newest_published_is_read_only(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    _save(root, 1)
    _save(root, 2)
    expected = _save(root, 3)
    before = _snapshot(tmp_path)
    assert discover_published_checkpoint(root) == expected
    assert _snapshot(tmp_path) == before


def test_discovery_ignores_newer_unpublished_directory(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    expected = _save(root, 1)
    # Restore publication files to model a crash after the next directory rename.
    indexes = {path: path.read_bytes() for path in root.iterdir() if path.is_file()}
    newer = _save(root, 2)
    for path, data in indexes.items():
        path.write_bytes(data)
    before = _snapshot(tmp_path)
    assert discover_published_checkpoint(root) == expected
    assert newer.checkpoint_dir.is_dir()
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("index_state", ["empty", "missing", "wrong-location", "corrupt"])
def test_discovery_refuses_inconsistent_indexes_without_writes(tmp_path: Path, index_state: str) -> None:
    import json

    root = tmp_path / "checkpoints"
    _save(root, 1)
    catalog = root / "publications.jsonl"
    error = PredecessorDiscoveryRefusal
    if index_state == "empty":
        catalog.write_bytes(b"")
    elif index_state == "missing":
        catalog.rename(root / "saved-catalog")
    elif index_state == "wrong-location":
        row = json.loads(catalog.read_text())
        row["ref"]["checkpoint_dir"] = str(root / "step_elsewhere")
        catalog.write_text(json.dumps(row) + "\n")
    else:
        catalog.write_bytes(b"not-json\n")
        error = ValueError
    before = _snapshot(tmp_path)
    with pytest.raises(error):
        discover_published_checkpoint(root)
    assert _snapshot(tmp_path) == before


def test_discovery_ignores_logger_metrics_and_status_authority(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    expected = _save(root, 1)
    assert discover_published_checkpoint(root) == expected
    # Put misleading observations both beside and inside the checkpoint root.
    for directory in (tmp_path, root):
        (directory / "logger.jsonl").write_text('{"step":999,"next_iteration":999}\n')
        (directory / "logger.csv").write_text("step,next_iteration\n999,999\n")
        (directory / "metrics.jsonl").write_text('{"step":999}\n')
        (directory / "metrics.csv").write_text("step,energy\n999,-1\n")
        (directory / "metrics.json").write_text('{"step":999}')
        (directory / "status.json").write_text('{"status":"completed","next_iteration":999}')
        (directory / "run-outcome.json").write_text('{"outcome":"completed","next_iteration":999}')
    before = _snapshot(tmp_path)
    assert discover_published_checkpoint(root) == expected
    assert _snapshot(tmp_path) == before


def test_receipt_exact_bytes_and_second_create_refuses(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = {"z": [True, None, 2], "a": "é"}
    create_json_receipt(path, payload)
    assert path.read_bytes() == '{"a":"é","z":[true,null,2]}\n'.encode("utf-8")
    before = _snapshot(tmp_path)
    with pytest.raises(ReceiptExistsError):
        create_json_receipt(path, {"replacement": True})
    assert path.read_bytes() == before["receipt.json"][1]
    assert path.stat().st_mtime_ns == before["receipt.json"][0]
    assert list(tmp_path.iterdir()) == [path]


def test_receipt_is_complete_before_atomic_publication(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    original_link = invocation.os.link
    observed = []

    def inspect_link(source, destination):
        assert not path.exists(), "destination became visible before publication"
        assert Path(source).read_bytes() == b'{"value":7}\n'
        observed.append(Path(destination))
        original_link(source, destination)

    monkeypatch.setattr(invocation.os, "link", inspect_link)
    create_json_receipt(path, {"value": 7})
    assert observed == [path]
    assert path.read_bytes() == b'{"value":7}\n'
    assert list(tmp_path.iterdir()) == [path]



def test_receipt_temporary_creation_is_exclusive(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    temporary = tmp_path / ".receipt.json.collision.tmp"
    temporary.write_bytes(b"another writer owns these bytes")
    monkeypatch.setattr(invocation, "uuid4", lambda: SimpleNamespace(hex="collision"))
    before = _snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        create_json_receipt(path, {"value": 7})
    assert _snapshot(tmp_path) == before
    assert not path.exists()


def _race_create(path, barrier, results, contender):
    barrier.wait(timeout=60)
    try:
        create_json_receipt(path, {"winner": contender})
    except ReceiptExistsError:
        results.put(("exists", contender))
    else:
        results.put(("created", contender))


def test_receipt_two_processes_exactly_one_winner(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    results = context.Queue()
    path = tmp_path / "receipt.json"
    processes = [
        context.Process(target=_race_create, args=(path, barrier, results, contender))
        for contender in (0, 1)
    ]
    try:
        for process in processes:
            process.start()
        barrier.wait(timeout=60)
        outcomes = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0
        assert sorted(outcome for outcome, _ in outcomes) == ["created", "exists"]
        winner = next(contender for outcome, contender in outcomes if outcome == "created")
        assert path.read_bytes() == ('{"winner":' + str(winner) + '}\n').encode()
        assert list(tmp_path.iterdir()) == [path]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()
        results.join_thread()
