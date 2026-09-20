"""Path spelling: restore is spelling-independent, publish is not. Measured once.

Nothing in ``tpen/checkpoint`` canonicalises a path -- no ``resolve``,
``realpath``, ``samefile``, ``abspath`` or ``absolute`` appears in any of its
seven modules, and paths are built and compared as bare ``Path(...)``. (Naming
trap: ``artifact.resolve_checkpoint_dir`` does POINTER resolution, root or
``latest.json`` to a step directory, NOT path canonicalisation.)

WHERE THAT CANNOT BITE. Equivalent spellings open the same file, so spelling is
harmless wherever a path is merely opened or stat-ed. ``write_latest`` stores
``checkpoint_dir.name`` only, ``resolve_checkpoint_dir`` rebuilds
``parent / basename``, and ``is_complete_checkpoint_dir`` tests a basename and
existence. **The resume pointer is therefore spelling-independent.**

WHERE IT DOES BITE, and it is deliberate. ``reference.py`` line 39 says the path
is "a location, not part of content_id", yet ``to_dict`` serializes
``checkpoint_dir`` alongside it and ``CheckpointCatalog.publish`` compares the
FULL serialized mapping. The intent is documented in ``publish``'s own
docstring: a relocation must conflict rather than create two locations for one
catalog identity, because an append-only catalog cannot rewrite the original.

THE DEFECT IS THE CONFLATION. A genuine relocation and the same directory
reached by an equivalent spelling are indistinguishable to that comparison,
because nothing resolves. The guard against ambiguity fires where there is no
ambiguity: one directory, one content identity, two spellings.

AND THE ASYMMETRY IS THE FINDING. Restore succeeds under the alternate spelling
and publish refuses, so a chain attempt reaching the same root by a different
spelling RESTORES FINE AND THEN CANNOT COMMIT ITS NEXT GENERATION -- it advances
zero further generations and stops on a ``ValueError``. Fail-closed is the safe
direction, but a chain that cannot continue is exactly what this program exists
to prevent. Every resume attempt here is a fresh OS process, and fresh processes
are where spelling diverges: different cwd, relative versus absolute, symlinked
scratch roots, a mount presenting differently in a later allocation.

NOT FIXED HERE. ``tpen/checkpoint`` is outside this lane's write surface. No
``resolve()`` is added to ``tpen/``. Attributed to the ``tpen/checkpoint`` owner
and to ``3b9b736a``. This arm runs against PRODUCTION ``publish``, ``reference``
and pointer code -- those modules are torch-free at runtime, so no replica is in
the path. Every other arm in this lane supplies absolute, already-resolved roots
so this fragility cannot contaminate the parity or G6 results.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.helpers.chain_resume_spike.fixture import (
    ContinuationSystem,
    publish_generation,
    spawn_publish_only,
)
from tests.helpers.chain_resume_spike.identity import new_attempt
from tpen.checkpoint.reference import CheckpointRef

CHECKPOINT_EVERY = 2


@pytest.fixture()
def committed_generation(tmp_path) -> tuple[Path, Path, str]:
    """Commit generation 1 under an ABSOLUTE root; return root, generation, content_id."""

    root = (tmp_path / "root").resolve()
    system = ContinuationSystem.fresh(seed=8_191)
    system.run(CHECKPOINT_EVERY)
    generation = publish_generation(
        root, system, new_attempt("spelling", 0, None), credited_steps=(0, 1)
    )
    return root, generation, CheckpointRef.from_directory(generation).content_id


def test_republishing_the_same_absolute_spelling_is_idempotent(
    committed_generation, tmp_path
) -> None:
    """CONTROL, and it is what makes the two arms below mean anything.

    ``publish`` is idempotent for an identical serialized mapping, so
    republishing the very same directory under the very same spelling must
    succeed. Without this, the refusals below would be equally consistent with
    "republishing anything conflicts", and would say nothing about spelling.
    """

    root, generation, content_id = committed_generation

    report = spawn_publish_only(
        tmp_path / "control", root=root, generation=generation, cwd=root.parent
    )

    assert report["published"] is True, (
        f"republishing an identical spelling was refused: {report['error']}"
    )
    assert report["error_type"] is None
    assert report["content_id"] == content_id


def test_a_relative_spelling_from_a_different_cwd_is_refused_by_publish(
    committed_generation, tmp_path
) -> None:
    """Same directory, same content identity, different spelling: publish refuses.

    The expected message is derived from the source (catalog.py:77-80), which
    interpolates the content id with NO surrounding quotes, and from the actual
    ref's own ``content_id`` -- not from a guess about either.
    """

    root, generation, content_id = committed_generation
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    relative_generation = Path(os.path.relpath(generation, workdir))
    relative_root = Path(os.path.relpath(root, workdir))
    assert not relative_generation.is_absolute()

    report = spawn_publish_only(
        tmp_path / "relative",
        root=relative_root,
        generation=relative_generation,
        cwd=workdir,
    )

    assert report["published"] is False, "the alternate spelling was accepted"
    assert report["error_type"] == "ValueError"
    assert report["error"] == (
        f"conflicting checkpoint publication for content_id {content_id}"
    ), f"unexpected refusal message: {report['error']!r}"
    # The identity really is path-independent; only the serialized mapping differs.
    assert report["content_id"] == content_id
    assert report["serialized_checkpoint_dir"] != str(generation)

    # THE ASYMMETRY, in the same arm and under the same spelling.
    assert report["pointer_error"] is None
    assert report["pointer_resolves_to"] == generation.name, (
        "restore should be spelling-independent; the pointer stores a basename"
    )


def test_a_symlinked_root_is_refused_by_publish_but_still_resolves(
    committed_generation, tmp_path
) -> None:
    """The same asymmetry through a symlink, which is how a scratch root usually differs.

    Skipped rather than silently passed where the filesystem cannot make one --
    an arm that arranges for the question not to arise and then reports green
    looks safer than the system it tests.
    """

    root, generation, content_id = committed_generation
    link = tmp_path / "linked-root"
    try:
        link.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"filesystem cannot create a directory symlink: {exc}")

    report = spawn_publish_only(
        tmp_path / "symlink",
        root=link,
        generation=link / generation.name,
        cwd=tmp_path,
    )

    assert report["published"] is False, "the symlinked spelling was accepted"
    assert report["error_type"] == "ValueError"
    assert report["error"] == (
        f"conflicting checkpoint publication for content_id {content_id}"
    )
    assert report["pointer_error"] is None
    assert report["pointer_resolves_to"] == generation.name


def test_the_probe_really_ran_in_a_different_working_directory(
    committed_generation, tmp_path
) -> None:
    """The arms above are only about spelling if the child's cwd actually differed.

    A parent that normalised the path before passing it, or a child that
    inherited the parent's cwd, would make the relative arm a re-run of the
    absolute one while still going red for some other reason.
    """

    root, generation, _ = committed_generation
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()

    report = spawn_publish_only(
        tmp_path / "cwd-check",
        root=Path(os.path.relpath(root, workdir)),
        generation=Path(os.path.relpath(generation, workdir)),
        cwd=workdir,
    )

    assert Path(report["cwd"]).resolve() == workdir.resolve()
    assert not Path(report["spelling"]).is_absolute()
