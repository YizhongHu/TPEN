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


def test_r6_read_only_discovery_ignores_unpublished_newer_dir(tmp_path):
    """A predecessor killed between a newer checkpoint directory's rename and
    its catalog/``latest.json`` publish (a normal crash/timeout state, not
    corruption) must not be refused forever: discovery selects the newest
    PUBLISHED checkpoint and leaves the unpublished directory untouched
    (orchestrator decision decision-unpublished-newer-dir-20261111 on item
    4023abbb, refining H1 addendum F1 -- refusing here would strand the
    lineage, since read-only discovery can never publish it either)."""

    from tpen.execution.invocation import _committed_checkpoint
    from tests.unit.execution.test_s15_resume_predecessor_leaf_measurement import _snapshot_leaf

    root = tmp_path / "run"
    old, new = _arrange_stale_indexes(root, keep_catalog=True)
    before = _snapshot_leaf(root)
    selected = _committed_checkpoint(root)
    after = _snapshot_leaf(root)
    assert selected is not None
    assert selected.checkpoint_dir == old
    assert selected.next_iteration == 1
    assert after == before, "read-only discovery mutated the sealed predecessor leaf"
    assert old.exists() and new.exists()


def test_r6_read_only_discovery_refuses_when_nothing_was_ever_published(tmp_path):
    """A predecessor whose publication catalog never existed at all (not
    merely behind the newest directory) still refuses: distinct from the
    ignore-and-select-published case above, there is nothing published here
    to select."""

    from tpen.execution.invocation import PredecessorDiscoveryRefusal, _committed_checkpoint

    root = tmp_path / "run"
    old, new = _arrange_stale_indexes(root, keep_catalog=False)
    before = _payload_hashes(new)
    old_before = _payload_hashes(old)
    with pytest.raises(PredecessorDiscoveryRefusal):
        _committed_checkpoint(root)
    after = _payload_hashes(new)
    old_after = _payload_hashes(old)
    assert after == before, "refused discovery changed immutable checkpoint payload bytes"
    assert old_after == old_before, "refused discovery changed the older immutable payload bytes"
    assert old.exists() and new.exists()


@pytest.mark.parametrize("keep_catalog", [True, False], ids=["old-catalog", "old-latest-no-catalog"])
def test_r6_committed_checkpoint_reconciles_newest_unindexed_state(tmp_path, keep_catalog):
    """The owning allocation explicitly authorizes repair of its own stale
    index (H1: discovery defaults to read-only; see the refusal test above)."""

    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "run"
    old, new = _arrange_stale_indexes(root, keep_catalog=keep_catalog)
    before = _payload_hashes(new)
    old_before = _payload_hashes(old)
    selected = _committed_checkpoint(root, allow_reconcile=True)
    assert selected is not None
    assert selected.next_iteration == 2
    assert selected.checkpoint_dir == new
    after = _payload_hashes(new)
    old_after = _payload_hashes(old)
    assert after == before, "reconciliation changed immutable checkpoint payload bytes"
    assert old_after == old_before, "reconciliation changed the older immutable payload bytes"
    # The index is now consistent, so read-only discovery accepts it too.
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
    # The owning allocation explicitly authorizes the repair (H1: discovery
    # defaults to read-only; see test_s15_resume_predecessor_leaf_measurement.py).
    selected = _committed_checkpoint(root, allow_reconcile=True)
    assert selected is not None and selected.checkpoint_dir == newest
    assert (newest.parent / PUBLICATION_CATALOG_FILENAME).is_file()
    assert (newest.parent / "latest.json").is_file()


def test_r6_empty_catalog_is_a_separate_green_control(tmp_path):
    """An empty catalog plus a missing ``latest.json`` is still an indexless
    predecessor for read-only discovery; the owning allocation may still
    repair it (H1 addendum)."""

    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME
    from tpen.execution.invocation import PredecessorDiscoveryRefusal, _committed_checkpoint

    root = tmp_path / "run"
    newest = _save(root, 2)
    catalog = newest.parent / PUBLICATION_CATALOG_FILENAME
    catalog.write_text("", encoding="utf-8")
    (newest.parent / "latest.json").unlink()
    with pytest.raises(PredecessorDiscoveryRefusal):
        _committed_checkpoint(root)
    selected = _committed_checkpoint(root, allow_reconcile=True)
    assert selected is not None and selected.checkpoint_dir == newest


def test_r6_read_only_discovery_refuses_empty_catalog_with_intact_pointer(tmp_path):
    """An empty catalog with ``latest.json`` intact is distinct from the
    indexless case above: the stale-index iteration comparison alone would
    pass (the pointer still names the newest directory), so this exercises
    the catalog-row cross-check itself (review-r2-decisions R2.1)."""

    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME
    from tpen.execution.invocation import PredecessorDiscoveryRefusal, _committed_checkpoint
    from tests.unit.execution.test_s15_resume_predecessor_leaf_measurement import _snapshot_leaf

    root = tmp_path / "run"
    _save(root, 1)
    newest = _save(root, 2)
    (newest.parent / PUBLICATION_CATALOG_FILENAME).write_text("", encoding="utf-8")
    before = _snapshot_leaf(root)
    with pytest.raises(PredecessorDiscoveryRefusal):
        _committed_checkpoint(root)
    after = _snapshot_leaf(root)
    assert after == before, "refused discovery mutated the sealed predecessor leaf"


def test_r6_read_only_discovery_refuses_catalog_missing_selected_row(tmp_path):
    """The pointer and newest directory agree on step 2, but the catalog was
    rolled back to only the older row: the row-identity cross-check must
    still refuse even though the stale-index iteration comparison would pass
    (review-r2-decisions R2.1)."""

    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME
    from tpen.execution.invocation import PredecessorDiscoveryRefusal, _committed_checkpoint
    from tests.unit.execution.test_s15_resume_predecessor_leaf_measurement import _snapshot_leaf

    root = tmp_path / "run"
    old = _save(root, 1)
    catalog = old.parent / PUBLICATION_CATALOG_FILENAME
    old_catalog = catalog.read_bytes()
    new = _save(root, 2)
    catalog.write_bytes(old_catalog)
    before = _snapshot_leaf(root)
    with pytest.raises(PredecessorDiscoveryRefusal):
        _committed_checkpoint(root)
    after = _snapshot_leaf(root)
    assert after == before, "refused discovery mutated the sealed predecessor leaf"
    assert old.exists() and new.exists()


def test_r6_read_only_discovery_refuses_payload_identity_mismatch_against_catalog_row(tmp_path):
    """The selected directory's payload no longer matches its own catalog
    row's content identity (its model bytes were replaced in place by
    another structurally valid payload at the same step): the cross-check
    must refuse rather than trust the directory name and step number alone
    (review-r2-decisions R2.1)."""

    from tpen.execution.invocation import PredecessorDiscoveryRefusal, _committed_checkpoint
    from tests.unit.execution.test_s15_resume_predecessor_leaf_measurement import _snapshot_leaf

    root = tmp_path / "run"
    checkpoint = _save(root, 2)
    model_path = checkpoint / "model.pt"
    mutated = bytearray(model_path.read_bytes())
    mutated[0] ^= 0xFF
    model_path.write_bytes(bytes(mutated))
    before = _snapshot_leaf(root)
    with pytest.raises(PredecessorDiscoveryRefusal):
        _committed_checkpoint(root)
    after = _snapshot_leaf(root)
    assert after == before, "refused discovery mutated the sealed predecessor leaf"


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
