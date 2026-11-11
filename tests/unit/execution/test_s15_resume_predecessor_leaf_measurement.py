"""S1.5 measurement: does successor checkpoint discovery mutate a sealed
predecessor leaf?

Per Task Orchestrator item 047558ca and lane note
``hazard-e3-discovery-mutates-sealed-predecessor-2026-10-31``, E3 checkpoint
discovery (``tpen.execution.invocation._committed_checkpoint``) may call
``tpen.checkpoint.catalog.reconcile_publication``, which writes the
publication catalog, ``latest.json``, and the publication receipt. If a
successor's discovery path is ever pointed at a predecessor allocation's
sealed leaf (the hazard under investigation, not yet exercised), that call
would silently mutate finished-allocation evidence.

This module measures the filesystem effect of running the real discovery
path -- no mocking of the catalog writer -- against a predecessor leaf in
three states, modeled directly on the committed, fully-indexed, and
indexless fixtures used by ``tests/unit/execution/test_r6_phase_a_publication.py``:

(a) healthy published leaf (fully indexed at save time);
(b) leaf whose publication catalog and ``latest.json`` are missing, as an
    allocation interrupted before its index write would leave it;
(c) leaf with a corrupt catalog, which discovery must refuse rather than
    repair.

Every case asserts the predecessor leaf's file set, content hashes, and
mtimes are unchanged by discovery. Case (b) is measured RED: reconciliation
creates the missing index files, mutating the predecessor leaf. It is kept
failing (``xfail(strict=True)``) as the measured evidence for H1, not fixed
here -- product-code changes are out of scope for this slice.
"""

from __future__ import annotations

import hashlib
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
        cfg=OmegaConf.create({"study": {"name": "s15"}, "trainer": {"max_steps": 4}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id=f"s15-{step}", extra={}),
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


def _snapshot_leaf(run_dir: Path) -> dict[Path, tuple[str, int]]:
    """Snapshot every file under a predecessor leaf: relative path -> (sha256, mtime_ns).

    Covers the whole ``run_dir`` tree (checkpoint payloads, catalog, latest
    pointer, receipts), not only the checkpoint directory, since the hazard
    under test is any write anywhere under a sealed predecessor's leaf.
    """

    return {
        path.relative_to(run_dir): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in run_dir.rglob("*")
        if path.is_file()
    }


def test_s15_healthy_predecessor_leaf_is_byte_identical_after_discovery(tmp_path):
    """Case (a): a fully-indexed sealed leaf. Discovery must not rewrite it."""

    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "predecessor-run"
    _save(root, 1)
    _save(root, 2)

    before = _snapshot_leaf(root)
    selected = _committed_checkpoint(root)
    after = _snapshot_leaf(root)

    assert selected is not None and selected.next_iteration == 2
    assert after == before, "discovery mutated a fully-indexed sealed predecessor leaf"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "H1 measured: successor discovery writes into predecessor leaf "
        "(reconcile_publication backfills the missing catalog/latest/receipt "
        "when pointed at an indexless sealed leaf); fix in S2 (item TBD)"
    ),
)
def test_s15_indexless_predecessor_leaf_is_byte_identical_after_discovery(tmp_path):
    """Case (b): a committed checkpoint whose indexes never landed (an
    allocation interrupted before publication). Discovery reconciles -- and
    in doing so writes into the sealed leaf.
    """

    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME
    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "predecessor-run"
    newest = _save(root, 2)
    (newest.parent / PUBLICATION_CATALOG_FILENAME).unlink()
    (newest.parent / "latest.json").unlink()

    before = _snapshot_leaf(root)
    selected = _committed_checkpoint(root)
    after = _snapshot_leaf(root)

    assert selected is not None and selected.checkpoint_dir == newest
    assert after == before, "discovery mutated an indexless sealed predecessor leaf"


def test_s15_corrupt_predecessor_catalog_refuses_without_mutation(tmp_path):
    """Case (c): a corrupt catalog. Discovery must refuse, not repair, and
    must not touch the leaf while refusing.
    """

    from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME
    from tpen.execution.invocation import _committed_checkpoint

    root = tmp_path / "predecessor-run"
    _save(root, 1)
    (root / "checkpoints" / PUBLICATION_CATALOG_FILENAME).write_text(
        '{"schema":"corrupt"}\n', encoding="utf-8"
    )

    before = _snapshot_leaf(root)
    with pytest.raises(ValueError):
        _committed_checkpoint(root)
    after = _snapshot_leaf(root)

    assert after == before, "discovery mutated a sealed predecessor leaf while refusing a corrupt catalog"
