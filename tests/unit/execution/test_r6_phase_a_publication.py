"""Round-6 Phase-A A2: committed checkpoints outrank stale publication indexes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _save(root: Path, step: int):
    import torch
    from omegaconf import OmegaConf
    from tpen.checkpoint import TrainResume, save_checkpoint

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(state_dict=lambda: {"next_iteration": step, "completed_updates": step})
    sampler = SimpleNamespace(mcmc_state_dict=lambda: {"walker": step})
    context = SimpleNamespace(
        # The source configuration is immutable across checkpoint cursors.
        cfg=OmegaConf.create({"study": {"name": "r6"}, "trainer": {"max_steps": 4}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id=f"r6-{step}", extra={}),
        run_dir=root,
    )
    return save_checkpoint(
        output_dir=root / "checkpoints",
        next_iteration=step,
        completed_updates=step,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
        context=context,
    )


def _payload_hashes(checkpoint: Path):
    import hashlib

    return {
        path.relative_to(checkpoint): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }


def _arrange_stale_indexes(root: Path, *, keep_catalog: bool):
    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME

    old = _save(root, 1)
    catalog = old.parent / PUBLICATION_CATALOG_FILENAME
    old_catalog = catalog.read_bytes()
    new = _save(root, 2)
    # The second save is structurally valid, but its publication is hidden from
    # the main catalog to model a committed checkpoint with stale indexes.
    if keep_catalog:
        catalog.write_bytes(old_catalog)
    else:
        # Preserve the old catalog as a fixture artifact: an absent catalog is
        # distinct from an empty catalog and must exercise the latest.json path.
        catalog.replace(catalog.with_name("publications.jsonl.old-fixture"))
    (old.parent / "latest.json").write_text(
        json.dumps({"checkpoint_dir": old.name, "step": 1}), encoding="utf-8"
    )
    return old, new


@pytest.mark.parametrize("keep_catalog", [True, False], ids=["old-catalog", "old-latest-no-catalog"])
def test_r6_committed_checkpoint_reconciles_newest_unindexed_state(tmp_path, keep_catalog):
    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "run"
    old, new = _arrange_stale_indexes(root, keep_catalog=keep_catalog)
    before = _payload_hashes(new)
    old_before = _payload_hashes(old)
    selected = _committed_checkpoint(root)
    assert selected is not None
    assert selected.next_iteration == 2
    assert selected.checkpoint_dir == new
    after = _payload_hashes(new)
    old_after = _payload_hashes(old)
    assert after == before, "reconciliation changed immutable checkpoint payload bytes"
    assert old_after == old_before, "reconciliation changed the older immutable payload bytes"
    again = _committed_checkpoint(root)
    assert again is not None and again.content_id == selected.content_id
    assert old.exists()


def test_r6_fully_indexed_latest_checkpoint_is_a_green_control(tmp_path):
    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "run"
    _save(root, 1)
    newest = _save(root, 2)
    selected = _committed_checkpoint(root)
    assert selected is not None and selected.checkpoint_dir == newest


def test_r6_indexless_checkpoint_is_repaired_by_checkpoint_owner(tmp_path):
    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME
    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "run"
    newest = _save(root, 2)
    (newest.parent / PUBLICATION_CATALOG_FILENAME).unlink()
    (newest.parent / "latest.json").unlink()
    selected = _committed_checkpoint(root)
    assert selected is not None and selected.checkpoint_dir == newest
    assert (newest.parent / PUBLICATION_CATALOG_FILENAME).is_file()
    assert (newest.parent / "latest.json").is_file()


def test_r6_empty_catalog_is_a_separate_green_control(tmp_path):
    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME
    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "run"
    newest = _save(root, 2)
    catalog = newest.parent / PUBLICATION_CATALOG_FILENAME
    catalog.write_text("", encoding="utf-8")
    (newest.parent / "latest.json").unlink()
    selected = _committed_checkpoint(root)
    assert selected is not None and selected.checkpoint_dir == newest


def test_r6_committed_corrupt_catalog_refuses_instead_of_falling_back(tmp_path):
    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME
    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "run"
    _save(root, 1)
    (root / "checkpoints" / PUBLICATION_CATALOG_FILENAME).write_text(
        '{"schema":"corrupt"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        _committed_checkpoint(root)
