"""G6: publish generation 1, fail generation 2 at every real commit boundary.

Two things are asserted here, and the first is what keeps the second honest.

**The replica's boundary order still matches production.** ARM T writes its
payload files without ``torch``, so the payload writes are a replica. Everything
else in the sequence -- naming, the stale-tmp sweep, the manifest, the COMPLETE
marker, the rename, ``CheckpointRef``, the catalog, ``latest.json`` and the
publication receipt -- is production code called directly.
:func:`test_the_replica_boundary_order_matches_production_save_py` re-derives
the order from ``tpen/checkpoint/save.py`` itself, so a production reorder turns
this suite red instead of silently invalidating its claim.

**Generation 1 survives every way generation 2 can fail.** The fault points are
named against the real boundaries, and the ordering correction matters: TPEN
writes the catalog BEFORE ``latest.json`` (save.py:226 then save.py:229), so
``after_rename_before_catalog``, ``after_catalog_before_latest`` and
``after_latest_before_receipt`` are the three committed-but-unacknowledged
states production can actually be interrupted in.

**SELECTION IS ASSERTED ON THE SURFACE THAT GOVERNS RESUME.** Production reaches
its checkpoint through ``restore.py:118`` -> ``resolve_checkpoint_dir`` ->
``read_latest`` (artifact.py:68/71) and never lists directories:
``list_complete_checkpoints`` has no caller in ``tpen/checkpoint`` outside its
own definition and the package re-export. A clause proved only on the listing
would be measured on a surface that decides nothing about resume, so these tests
select through ``resume_generation`` and additionally PIN THE DISAGREEMENT
between the two surfaces in both pre-``latest.json`` windows.

**PROVENANCE OF THE VALIDATION BOUNDARY THESE ASSERTIONS REST ON.** The
selection, validity and torn-row assertions rest on ``CheckpointRef``
validation, which is closed in both directions BUT FROM DIFFERENT PLACES, and
the distinction changes the failure mode:

* CONSTRUCT direction -- **LOCAL** closure. ``reference.py`` ``__post_init__``
  (72-86) runs ``_nonnegative_int``, ``_nonempty_text``, ``_require_sha256`` and
  ``_freeze`` directly on the fields; ``_freeze`` (322-336) raises on anything
  that is not ``None``, ``str``, ``bool``, ``int`` or ``float``. The check is at
  the point of use.
* DESERIALIZE direction -- **DOWNSTREAM** closure. ``_thaw`` (338-343)
  VALIDATES NOTHING ITSELF. That path is safe only because
  ``deserialize_checkpoint_ref`` (296) routes through
  ``CheckpointRef.from_mapping``, which constructs the dataclass and re-runs the
  SAME ``__post_init__``.

A downstream closure holds only WHILE THE ROUTING HOLDS. A future caller
reaching ``_thaw`` directly, or a deserialize path refactored to build a
``CheckpointRef`` by any route bypassing ``__post_init__``, removes the
validation silently -- no validator is edited and no test of the validators goes
red. So these assertions depend on the ROUTE, not only on the checks.

Note also that ``_freeze``'s acceptance of ``None`` is NOT the caller-open-leaf
pattern: ``None`` is a legitimate terminal JSON scalar with nothing beneath it
to descend into. The discriminator is whether anything remains BELOW the leaf,
not the token itself.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from tests.helpers.chain_resume_spike import fixture as fixture_module
from tests.helpers.chain_resume_spike.faults import (
    COMMITTED_BUT_UNACKNOWLEDGED_POINTS,
    FaultAction,
    FaultPlan,
    MEASURED_POINTS,
    FaultPoint,
    InjectedFault,
)
from tests.helpers.chain_resume_spike.fixture import (
    BOUNDARY_ORDER,
    spawn_attempt,
    ContinuationSystem,
    newest_listed_generation,
    pointer_target_name,
    resume_generation,
    publish_generation,
)
from tests.helpers.chain_resume_spike.identity import new_attempt
from tests.helpers.chain_resume_spike.parity import first_divergence
from tests.helpers.chain_resume_spike.receipt import read_receipt
from tests.helpers.chain_resume_spike.restore_limbs import (
    RestoreLimb,
    live_limb_fingerprints,
)
from tpen.checkpoint.artifact import (
    is_complete_checkpoint_dir,
    list_complete_checkpoints,
    read_latest,
)
from tpen.checkpoint.catalog import (
    CheckpointCatalog,
    IncompletePublicationRecordError,
    publication_catalog_path,
    read_publications,
    reconcile_publication,
)
from tpen.checkpoint.receipt import publication_receipt_path

#: Anchors identifying each declared boundary inside ``save_checkpoint``'s body.
#:
#: Matched as plain substrings against the real source. Deliberately not a
#: regex over names: the point is to pin the ORDER of the actual calls, so an
#: anchor is the call text itself.
PRODUCTION_ANCHORS: tuple[tuple[str, str], ...] = (
    ("tmp_dir_named", 'tmp_dir = root / f"{final_dir.name}.tmp"'),
    ("stale_tmp_removed", "shutil.rmtree(tmp_dir)"),
    ("tmp_dir_created", "tmp_dir.mkdir(parents=True)"),
    ("resolved_config_written", "_write_resolved_config("),
    ("payload_written", "torch.save(model.state_dict()"),
    ("component_hashes_computed", "_component_file_hashes("),
    ("manifest_written", "manifest.write("),
    ("complete_marker_written", '(tmp_dir / "COMPLETE").write_text'),
    ("renamed", "tmp_dir.rename(final_dir)"),
    ("ref_built", "CheckpointRef.from_directory(final_dir)"),
    ("catalog_published", "catalog.publish(ref)"),
    ("latest_written", "write_latest(root, final_dir"),
    ("receipt_recorded", "record_publication_receipt("),
)

TOTAL_TARGET = 6
CHECKPOINT_EVERY = 2


def _save_checkpoint_body() -> list[tuple[int, str]]:
    """Return ``save_checkpoint``'s executable body lines, numbered.

    Bounded at both ends, and the bounds are load-bearing in both directions.

    The lower bound SKIPS THE DOCSTRING, located via :mod:`ast` rather than by
    text. ``save_checkpoint``'s docstring narrates its own commit sequence and
    contains the literal text ``tmp_dir.rename(final_dir)`` at line 89 -- a
    text scan matches the prose as readily as the call, and would silently
    pin the order of a comment. This exact false match is why the bound exists.

    The upper bound stops at the ``finally``, whose ``shutil.rmtree`` would
    otherwise collide with the stale-tmp sweep anchor. A whole-file scan would
    additionally match ``_component_file_hashes`` and
    ``record_publication_receipt`` at their imports and definitions rather than
    at the call sites whose order is the point.
    """

    path = Path("tpen/checkpoint/save.py")
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint"
    )
    statements = function.body
    # ``ast.get_docstring`` confirms the first statement really is the
    # docstring, so this does not skip a real statement if it is ever removed.
    first_executable = (
        statements[1] if ast.get_docstring(function) is not None else statements[0]
    )
    start = first_executable.lineno - 1
    end = next(
        index for index, line in enumerate(lines) if index > start and line == "    finally:"
    )
    return [(index + 1, lines[index]) for index in range(start, end)]


def test_the_replica_boundary_order_matches_production_save_py() -> None:
    """The fixture's declared sequence is production's sequence, re-derived.

    Also pins the ordering correction this lane made: ``catalog.publish``
    precedes ``write_latest`` in the real source. The lane design note
    originally had them the other way round, which would have named a state
    ``save_checkpoint`` never passes through.
    """

    body = _save_checkpoint_body()
    assert [name for name, _ in PRODUCTION_ANCHORS] == list(BOUNDARY_ORDER), (
        "the anchor table and BOUNDARY_ORDER have drifted apart"
    )

    located: list[tuple[str, int]] = []
    for name, anchor in PRODUCTION_ANCHORS:
        matches = [number for number, line in body if anchor in line]
        assert len(matches) == 1, (
            f"anchor for {name} matched {len(matches)} lines in save_checkpoint's body "
            f"(expected exactly 1): {matches}"
        )
        located.append((name, matches[0]))

    line_numbers = [number for _, number in located]
    assert line_numbers == sorted(line_numbers), (
        "tpen/checkpoint/save.py no longer performs these boundaries in the order "
        f"this fixture replicates: {located}"
    )
    # No separate catalog-before-latest assertion here: the ascending-order
    # assertion above already entails it, since both names are in the anchor
    # table. A second assertion of an implied fact reads like extra coverage and
    # is not.


def test_the_replica_executes_the_order_it_declares(tmp_path) -> None:
    """RUNTIME pin. The static test above compares two DESCRIPTIONS; this runs the code.

    The static test catches a drift in production's order and a drift in the
    declaration, and it MISSES a drift in the third thing -- the executable
    sequence inside ``publish_generation`` that the declaration stands for. A
    mutant that reorders the actual publish and latest calls survives the static
    test entirely, which was measured in the previous round.

    So ``publish_generation`` records each boundary AS IT EXECUTES, and the
    three orders are asserted equal in one place: observed == declared ==
    derived from production source.
    """

    root = tmp_path / "root"
    system = ContinuationSystem.fresh(seed=606)
    system.run(CHECKPOINT_EVERY)
    observed: list[str] = []
    publish_generation(
        root,
        system,
        new_attempt("replica-order", 0, None),
        credited_steps=(0, 1),
        boundary_log=observed,
    )

    assert observed == list(BOUNDARY_ORDER), (
        "the replica did not execute the boundary sequence it declares:\n"
        f"  observed: {observed}\n  declared: {list(BOUNDARY_ORDER)}"
    )
    production_order = [name for name, _ in PRODUCTION_ANCHORS]
    assert observed == production_order, (
        "the replica's executed order no longer matches production's source order"
    )


def _publish_two_generations(root: Path, fault: FaultPlan | None):
    """Commit generation 1 cleanly, then attempt generation 2 under ``fault``.

    Returns the uninterrupted reference trace tail, the system, and the
    exception raised by the second publish (or ``None``).
    """

    system = ContinuationSystem.fresh(seed=90_210)
    identity = new_attempt("g6", 0, None)

    system.run(CHECKPOINT_EVERY)
    publish_generation(root, system, identity, credited_steps=(0, 1))

    system.run(CHECKPOINT_EVERY)
    raised: BaseException | None = None
    try:
        publish_generation(root, system, identity, credited_steps=(0, 1, 2, 3), fault=fault)
    except BaseException as exc:  # noqa: BLE001 - the injected fault is the datum
        raised = exc
    return system, raised


@pytest.mark.parametrize(
    "point",
    [
        FaultPoint.DURING_PAYLOAD_WRITE,
        FaultPoint.AFTER_PAYLOAD_BEFORE_MANIFEST,
        FaultPoint.AFTER_MANIFEST_BEFORE_COMPLETE,
        FaultPoint.AFTER_COMPLETE_BEFORE_RENAME,
    ],
    ids=lambda point: point.value,
)
def test_a_precommit_fault_leaves_generation_one_selectable(tmp_path, point) -> None:
    """Before the rename, generation 2 must not become selectable at all.

    Selectability is decided by production's own
    ``list_complete_checkpoints`` / ``is_complete_checkpoint_dir``, which reject
    a ``.tmp`` name and any directory missing ``manifest.json`` or ``COMPLETE``.
    Nothing about that judgement is reimplemented here.
    """

    root = tmp_path / "root"
    _, raised = _publish_two_generations(root, FaultPlan(point, FaultAction.RAISE))

    assert isinstance(raised, InjectedFault), f"expected the injected fault, got {raised!r}"

    complete = list_complete_checkpoints(root)
    assert [path.name for path in complete] == ["step_000002"], (
        f"an interrupted pre-commit generation became selectable: {complete}"
    )
    assert resume_generation(root).name == "step_000002"
    # latest.json still names generation 1, so a resume that trusts the pointer
    # lands on the last good generation rather than on nothing.
    assert read_latest(root)["checkpoint_dir"] == "step_000002"
    # Exactly one publication row: the interrupted generation never published.
    assert len(read_publications(publication_catalog_path(root))) == 1


#: Spelled out member by member rather than as ``sorted(COMMITTED_BUT_
#: UNACKNOWLEDGED_POINTS)``. The fault-registry coverage scan reads
#: ``FaultPoint.MEMBER`` attributes from the AST, so a set alias would exercise
#: these three while leaving them invisible to the check that every registered
#: point is driven -- coverage in fact, a gap in the ledger. The assertion
#: below keeps the explicit list and the set from drifting apart.
_UNACKNOWLEDGED_ARMS = (
    FaultPoint.AFTER_RENAME_BEFORE_CATALOG,
    FaultPoint.AFTER_CATALOG_BEFORE_LATEST,
    FaultPoint.AFTER_LATEST_BEFORE_RECEIPT,
)
assert set(_UNACKNOWLEDGED_ARMS) == COMMITTED_BUT_UNACKNOWLEDGED_POINTS


@pytest.mark.parametrize(
    "point", _UNACKNOWLEDGED_ARMS, ids=lambda point: point.value
)
def test_a_committed_but_unacknowledged_generation_is_valid_and_reconcilable(
    tmp_path, point
) -> None:
    """After the rename, generation 2 IS committed; the acknowledgements may lag.

    The rename is the commit (save.py:211), and the catalog row, ``latest.json``
    and the publication receipt are three separate durable operations after it.
    ``reconcile_publication`` is production's repair primitive for exactly this
    family, and its documented semantics -- append-idempotent row, unconditional
    ``latest.json`` rewrite on mismatch, best-effort receipt backfill
    (catalog.py:189-208) -- are what this asserts, rather than a guess.
    """

    root = tmp_path / "root"
    _, raised = _publish_two_generations(root, FaultPlan(point, FaultAction.RAISE))
    assert isinstance(raised, InjectedFault)

    generation_two = root / "step_000004"
    assert is_complete_checkpoint_dir(generation_two), (
        "the rename committed the checkpoint, so it must be selectable even "
        "though an acknowledgement is missing"
    )
    # Listing always sees the committed directory: it is complete on disk.
    assert newest_listed_generation(root).name == "step_000004"

    # But the pointer is written LATER than the rename, so before it is written
    # the two surfaces disagree -- and resume follows the pointer.
    if point is FaultPoint.AFTER_LATEST_BEFORE_RECEIPT:
        assert pointer_target_name(root) == "step_000004"
        assert resume_generation(root).name == "step_000004"
    else:
        assert pointer_target_name(root) == "step_000002"
        assert resume_generation(root).name == "step_000002", (
            "resume must follow latest.json, which still names generation 1"
        )

    reconcile_publication(root, generation_two)

    published = read_publications(publication_catalog_path(root))
    assert [ref.next_iteration for ref in published] == [2, 4]
    assert read_latest(root)["checkpoint_dir"] == "step_000004"
    assert publication_receipt_path(root).is_file()
    # After reconciliation the two surfaces agree again.
    assert resume_generation(root).name == newest_listed_generation(root).name

    # Idempotent: reconciling again must not append a duplicate row.
    reconcile_publication(root, generation_two)
    assert len(read_publications(publication_catalog_path(root))) == 2


@pytest.mark.parametrize(
    "point",
    [FaultPoint.AFTER_RENAME_BEFORE_CATALOG, FaultPoint.AFTER_CATALOG_BEFORE_LATEST],
    ids=lambda point: point.value,
)
def test_listing_and_the_resume_pointer_disagree_before_latest_is_written(
    tmp_path, point
) -> None:
    """PIN THE DISAGREEMENT. This is a property, not an incident.

    In both pre-``latest.json`` windows a newer generation is complete on disk
    while the pointer still names the older one. Production resume reaches its
    checkpoint through ``restore.py:118`` -> ``resolve_checkpoint_dir`` ->
    ``read_latest`` (artifact.py:68/71) and NEVER lists:
    ``list_complete_checkpoints`` has no caller in ``tpen/checkpoint`` outside
    its own definition and the package re-export.

    Pinned because R1/R2/R3 will build chain controllers. A controller that
    picks up "the newest complete directory" selects generation 2 here, while
    the production resume path selects generation 1 and replays the tail. Both
    are defensible in isolation; together they are two controllers disagreeing
    about what the chain's state is. The safe behaviour observed here is safe
    BECAUSE OF WHICH SURFACE IS CONSULTED, not because of the directory state.
    """

    root = tmp_path / "root"
    _, raised = _publish_two_generations(root, FaultPlan(point, FaultAction.RAISE))
    assert isinstance(raised, InjectedFault)

    listed = newest_listed_generation(root)
    resumed = resume_generation(root)

    assert listed.name == "step_000004", "the committed generation is not on disk"
    assert resumed.name == "step_000002", "resume did not follow the pointer"
    assert listed != resumed, (
        "the two selection surfaces agreed, so this window no longer exercises "
        "the disagreement this test exists to pin"
    )
    assert pointer_target_name(root) == "step_000002"


def test_a_stale_tmp_directory_is_not_selectable_and_a_retry_does_not_collide(
    tmp_path,
) -> None:
    """An unwound fault leaves no tmp; a hard exit does, and the retry sweeps it.

    Both halves matter. ``save_checkpoint``'s ``finally`` (save.py:245) clears
    the tmp directory when the interpreter unwinds, but explicitly does NOT for
    a ``SIGKILL`` or a default-handler ``SIGTERM`` -- see its own note at
    save.py:255-260. The residue is what would collide with the next attempt, so
    the sweep at save.py:144 is what makes a retry possible at all.
    """

    root = tmp_path / "root"
    system = ContinuationSystem.fresh(seed=4_242)
    identity = new_attempt("g6-tmp", 0, None)
    system.run(CHECKPOINT_EVERY)
    publish_generation(root, system, identity, credited_steps=(0, 1))
    system.run(CHECKPOINT_EVERY)

    # A raise unwinds, so the finally clears the tmp directory.
    with pytest.raises(InjectedFault):
        publish_generation(
            root,
            system,
            identity,
            credited_steps=(0, 1, 2, 3),
            fault=FaultPlan(FaultPoint.AFTER_MANIFEST_BEFORE_COMPLETE, FaultAction.RAISE),
        )
    assert not (root / "step_000004.tmp").exists()

    # Now simulate the residue a non-unwinding death leaves behind, and prove
    # (a) it is not selectable and (b) the retry sweeps rather than collides.
    stale = root / "step_000004.tmp"
    stale.mkdir()
    (stale / "manifest.json").write_text("{}", encoding="utf-8")
    (stale / "COMPLETE").write_text("complete\n", encoding="utf-8")
    assert not is_complete_checkpoint_dir(stale), (
        "a .tmp directory carrying both markers must still be rejected by name"
    )
    assert [path.name for path in list_complete_checkpoints(root)] == ["step_000002"]

    committed = publish_generation(root, system, identity, credited_steps=(0, 1, 2, 3))
    assert committed.name == "step_000004"
    assert not stale.exists(), "the retry did not sweep the stale tmp directory"


#: Cold-start seed shared by a reference run and a chain's first link. Only the
#: COLD START is pinned; every resumed attempt still draws its own OS entropy.
G6_COLD_START_SEED = 5_150_926


def test_restoring_generation_one_after_a_failed_generation_two_reproduces_the_stream(
    tmp_path,
) -> None:
    """The point of preserving generation 1: it still CONTINUES correctly.

    Survival on disk is not the claim. The claim is that restoring generation 1
    and taking further draws reproduces what an uninterrupted run would have
    drawn -- so the comparison is on subsequent draws, as everywhere else.

    THE RESTORE RUNS IN A FRESH OS PROCESS. An in-process restore leaves the
    parent's live RNG objects in memory, so the comparison can pass without
    anything having been restored -- the same vacuity the G0 arms exist to rule
    out, and it was still present in this module in the previous round.
    """

    total, every = 4, CHECKPOINT_EVERY

    # ARM A: uninterrupted, from a pinned cold start so the two arms share an
    # initial condition. Two independent cold starts share no trajectory.
    reference_launch = spawn_attempt(
        tmp_path / "reference",
        root=tmp_path / "reference-root",
        run_id="g6-parity",
        attempt_index=0,
        steps=total,
        total_target=total,
        checkpoint_every=every,
        seed=G6_COLD_START_SEED,
    )
    assert reference_launch.exit_code == 0, reference_launch.log_path.read_text(
        encoding="utf-8"
    )
    reference = read_receipt(reference_launch.receipt_path)

    # ARM B: the same cold start, committing generation 1 and stopping.
    root = tmp_path / "root"
    first = spawn_attempt(
        tmp_path / "a0", root=root, run_id="g6-parity", attempt_index=0,
        steps=every, total_target=total, checkpoint_every=every,
        seed=G6_COLD_START_SEED,
    )
    assert first.exit_code == 0, first.log_path.read_text(encoding="utf-8")
    survivor = resume_generation(root)
    assert survivor.name == "step_000002"

    # Generation 2 fails before it is ever committed.
    faulted = spawn_attempt(
        tmp_path / "a1", root=root, run_id="g6-parity", attempt_index=1,
        steps=every, total_target=total, checkpoint_every=every,
        fault=FaultPlan(FaultPoint.DURING_PAYLOAD_WRITE, FaultAction.RAISE),
    )
    assert faulted.exit_code == 1
    assert resume_generation(root).name == "step_000002", (
        "the failed generation displaced the surviving one"
    )

    # Generation 1 still continues the stream, from a fresh process whose own
    # seed differs from both arms above.
    resumed = spawn_attempt(
        tmp_path / "a2", root=root, run_id="g6-parity", attempt_index=2,
        steps=total - every, total_target=total, checkpoint_every=every,
    )
    assert resumed.exit_code == 0, resumed.log_path.read_text(encoding="utf-8")
    resumed_receipt = read_receipt(resumed.receipt_path)
    assert resumed_receipt.process_seed != reference.process_seed

    divergence = first_divergence(reference.trace[every:], resumed_receipt.trace)
    assert divergence is None, (
        "generation 1 survived on disk but no longer continues the stream: "
        + divergence.describe()
    )


def test_a_torn_final_catalog_row_is_diagnosed_and_repairable(tmp_path) -> None:
    """``TORN_CATALOG_ROW``, extending the patterns in test_catalog_torn_row.py.

    That module already pins the reader's behaviour; f6bd2a8b is terminal and
    this does not re-assert a fixed bug as open. What is added here is the
    chain-level consequence: after the documented repair, the committed
    generation is selectable and its identity row is back.
    """

    # Named explicitly so the registry coverage scan sees this point driven.
    # Unlike the others this one is not injected mid-publish: a torn row is
    # damage to the index AFTER a clean commit, so it is produced by truncating
    # the catalog rather than by a FaultPlan.
    assert FaultPoint.TORN_CATALOG_ROW in MEASURED_POINTS

    root = tmp_path / "root"
    system = ContinuationSystem.fresh(seed=8_642)
    identity = new_attempt("g6-torn", 0, None)
    system.run(CHECKPOINT_EVERY)
    publish_generation(root, system, identity, credited_steps=(0, 1))
    system.run(CHECKPOINT_EVERY)
    publish_generation(root, system, identity, credited_steps=(0, 1, 2, 3))

    catalog_path = publication_catalog_path(root)
    rows = catalog_path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Tear the final row: truncate it and drop its terminating newline, which is
    # what a write interrupted mid-append leaves behind.
    catalog_path.write_text(rows[0] + rows[1][: len(rows[1]) // 2], encoding="utf-8")

    with pytest.raises(IncompletePublicationRecordError):
        read_publications(catalog_path)

    # The committed directory is untouched by the torn index; selection still works.
    assert resume_generation(root).name == "step_000004"

    # The repair the error message names: drop the unterminated final line, then
    # reconcile the newest complete directory that now has no row.
    catalog_path.write_text(rows[0], encoding="utf-8")
    reconcile_publication(root, root / "step_000004")
    assert [ref.next_iteration for ref in read_publications(catalog_path)] == [2, 4]


def test_reconciliation_does_not_rewrite_the_committed_payload_bytes(
    tmp_path,
) -> None:
    """ONE operation, sampled either side of it. Reconciliation repairs indexes
    only; the payload bytes must not move.

    ``reconcile_publication``'s contract says so explicitly (catalog.py:199-201).
    Pinned here because a repair that rewrote a committed scientific artifact
    would be a far worse failure than the missing row it fixed.

    WHAT IT DOES NOT ESTABLISH, and the earlier name asserted otherwise: this
    exercises ``reconcile_publication`` and nothing else, comparing bytes
    before and after that single call. **IMMUTABILITY UNDER ANY OTHER
    OPERATION IS NOT ESTABLISHED HERE**, and neither is immutability between
    the two samples. Re-scoped by the round-5 claims sweep.
    """

    root = tmp_path / "root"
    system = ContinuationSystem.fresh(seed=1_234)
    identity = new_attempt("g6-immutable", 0, None)
    system.run(CHECKPOINT_EVERY)
    publish_generation(root, system, identity, credited_steps=(0, 1))

    generation = root / "step_000002"
    before = {
        path.name: path.read_bytes() for path in sorted(generation.iterdir()) if path.is_file()
    }
    reconcile_publication(root, generation)
    after = {
        path.name: path.read_bytes() for path in sorted(generation.iterdir()) if path.is_file()
    }
    assert before == after, "reconciliation mutated the committed checkpoint payload"


def test_the_manifest_records_the_chain_provenance_a_resume_needs(tmp_path) -> None:
    """Parent generation and credited steps travel with the generation itself.

    Without this the next attempt cannot rebuild the ledger, and would have to
    infer how much work was already credited -- which is how a replayed tail
    gets counted twice.
    """

    root = tmp_path / "root"
    system = ContinuationSystem.fresh(seed=555)
    identity = new_attempt("g6-provenance", 2, 4)
    system.run(CHECKPOINT_EVERY)
    published = publish_generation(root, system, identity, credited_steps=(0, 1))

    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    assert provenance["run_id"] == "g6-provenance"
    assert provenance["attempt_index"] == 2
    assert provenance["parent_generation"] == 4
    assert provenance["credited_steps"] == [0, 1]
    assert re.fullmatch(r"step_\d{6}", published.name)


# ----------------------------------------------------------------------
# Process-death arms. Repeated, because they are race-sensitive.
# ----------------------------------------------------------------------

#: Race-sensitive arms run this many times. A single observation of a timing
#: dependent outcome is an anecdote, and a small spike sample is not an
#: availability figure either -- this bounds the anecdote, it does not remove it.
REPEATS = 3


@pytest.mark.parametrize("attempt", range(REPEATS), ids=lambda index: f"run{index}")
@pytest.mark.parametrize(
    "action",
    [FaultAction.OS_EXIT, FaultAction.SIGTERM_SELF, FaultAction.SIGKILL_FROM_PARENT],
    ids=lambda action: action.value,
)
def test_generation_one_survives_a_process_death_that_never_unwinds(
    tmp_path, action: FaultAction, attempt: int
) -> None:
    """The case ``save_checkpoint``'s own ``finally`` explicitly does not cover.

    save.py:255-260 states it plainly: a ``SIGKILL``, or a ``SIGTERM`` under
    Python's default handler, terminates the process WITHOUT unwinding, so no
    ``finally`` runs and the temporary directory is left behind. Interruption
    safety there is "the interpreter got a chance to unwind", not "the directory
    is never left behind". These arms drive exactly that gap in a real process
    and assert the property that must hold regardless: the previously committed
    generation stays selectable, and the residue never becomes selectable.

    ``SIGKILL_FROM_PARENT`` is delivered by the parent, not self-inflicted. A
    signal the child arranges for itself is a signal the child was alive enough
    to arrange; only an external kill tests the case where no handler can run.
    """

    del attempt  # the repeat index only distinguishes parametrized node ids
    root = tmp_path / "root"

    first = spawn_attempt(
        tmp_path / "a0", root=root, run_id="hardkill", attempt_index=0,
        steps=2, total_target=6, checkpoint_every=2,
    )
    assert first.exit_code == 0, first.log_path.read_text(encoding="utf-8")
    assert resume_generation(root).name == "step_000002"

    second = spawn_attempt(
        tmp_path / "a1", root=root, run_id="hardkill", attempt_index=1,
        steps=2, total_target=6, checkpoint_every=2,
        fault=FaultPlan(FaultPoint.AFTER_COMPLETE_BEFORE_RENAME, action),
        timeout=60.0,
    )
    # Scheduler-visible status and inner application status are separate facts.
    # This is the OS-level status; the attempt wrote no outcome at all, which is
    # the inner status and is recorded by the receipt's ABSENCE below.
    assert second.exit_code != 0, (
        f"a {action.value} death reported success at the OS level"
    )
    if action is FaultAction.SIGKILL_FROM_PARENT:
        assert second.killed_by_parent, "the parent never delivered the kill"
        assert second.exit_code == -9

    # No outcome was recorded, because the process never got to record one.
    # Distinguishing "no receipt" from "a receipt saying it failed" is the
    # difference between a death and a handled error.
    assert not second.receipt_path.exists(), (
        "a non-unwinding death still wrote a receipt, so the arm is not testing "
        "what it claims to test"
    )

    # The property that must hold anyway.
    assert resume_generation(root).name == "step_000002"
    assert [path.name for path in list_complete_checkpoints(root)] == ["step_000002"]
    assert read_latest(root)["checkpoint_dir"] == "step_000002"
    # Unconditional, not ``if residue.exists()``. All three of these actions
    # terminate without unwinding, so production's ``finally`` cannot run and
    # the residue is GUARANTEED, not incidental -- MEASURED for each action at
    # this revision. A conditional assertion here would quietly become vacuous
    # the day the residue stopped appearing, which is precisely the change a
    # reader of this test would most want to be told about.
    residue = root / "step_000004.tmp"
    assert residue.exists(), (
        f"a {action.value} death left no residue, so production's non-unwinding "
        "gap is no longer what this arm exercises"
    )
    assert not is_complete_checkpoint_dir(residue), (
        "the residue of a killed write became selectable"
    )

    # And the chain still continues: a fresh attempt sweeps any residue and
    # reaches the target.
    third = spawn_attempt(
        tmp_path / "a2", root=root, run_id="hardkill", attempt_index=2,
        steps=6, total_target=6, checkpoint_every=2,
    )
    assert third.exit_code == 0, third.log_path.read_text(encoding="utf-8")
    assert not residue.exists(), "the retry did not sweep the residue"


def test_committing_a_generation_does_not_disturb_any_random_stream(tmp_path) -> None:
    """SAVE IS RNG-NEUTRAL over the fixture's OWN enumerated streams.

    The population is ``RestoreLimb``, a closed enum of this fixture's three
    limbs, and the loop below covers it exactly. "Any random stream" in the
    node name is bounded by that enum -- a stream this fixture does not hold
    is outside the claim. Pinned directly, because G0 does not pin it.

    G0 compares draws taken after a restore, so it detects a mismatch between
    what was CAPTURED and what is live afterwards. It does not detect a save
    that perturbs the streams BEFORE capturing them: measured in the previous
    round, a mutant drawing all three streams before capture leaves G0 green,
    while the same draws after capture turn G0 red. Neutrality therefore needs
    its own assertion rather than being inferred from parity.

    It matters for the chain because generations are committed mid-run. A save
    that consumed entropy would make the trajectory depend on the checkpoint
    CADENCE, so a run interrupted at a different point would not reproduce.
    """

    root = tmp_path / "root"
    system = ContinuationSystem.fresh(seed=31_415)
    system.run(3)

    before = live_limb_fingerprints(system)
    publish_generation(
        root, system, new_attempt("neutrality", 0, None), credited_steps=(0, 1, 2)
    )
    after = live_limb_fingerprints(system)

    for limb in RestoreLimb:
        assert before[limb] == after[limb], (
            f"committing a generation advanced the {limb.value} stream; the "
            "trajectory would then depend on checkpoint cadence"
        )

    # And the stronger end-to-end form: the draws that FOLLOW a commit are the
    # draws that would have followed no commit at all.
    committed_tail = system.run(2)
    reference = ContinuationSystem.fresh(seed=31_415)
    reference.run(3)
    assert first_divergence(reference.run(2), committed_tail) is None, (
        "the post-commit trajectory diverged from an uncommitted one"
    )


# ----------------------------------------------------------------------
# C10: the ACTUAL operation order, observed where the operations happen.
# ----------------------------------------------------------------------
#
# The two tests above compare DECLARATIONS and EMITTED RECORDS. Both are
# self-reports, and a self-report is separable from the thing it reports: a
# mutant that moves the real ``catalog.publish`` past ``write_latest`` while
# leaving the ``_boundary`` calls where they were satisfies both of them and
# every other node in the suite. Measured, not hypothesised.
#
# So the two checks below never read a label. One derives the order from INSIDE
# a wrapper around the real production callable, so displacing the call
# necessarily displaces its record -- there is no separate emission statement to
# leave behind. The other reads the FILESYSTEM immediately before the pointer
# write executes, which is a property of the world that no breadcrumb can lie
# about -- and immediately before, not after, because the precondition is only
# meaningful while the pointer does not yet exist.


def _observe_real_operation_order(monkeypatch) -> list[str]:
    """Wrap the real publish and write_latest; return the order they took effect.

    Each record is emitted BY THE WRAPPER, so displacing a call displaces its
    record -- there is no separate emission statement to leave behind. Patching
    the names the fixture actually resolves -- ``CheckpointCatalog.publish`` on
    the class, and the module-level ``write_latest`` the fixture imported --
    means a reordering of the call sites reorders these records too.

    THE TWO RECORDS ARE NOT TAKEN AT THE SAME POINT, and the earlier wording,
    which said both were taken after the wrapped call returned, was FALSE of
    the second one. ``wrapped_publish`` records AFTER ``real_publish`` returns,
    so it reports COMPLETION. ``wrapped_write_latest`` records BEFORE
    ``real_write_latest`` is called, and deliberately so: the filesystem
    precondition it samples is only meaningful while the pointer does not yet
    exist, and sampling it after the return would read a world in which the
    pointer is already published.

    So the sequence establishes CATALOG COMPLETION BEFORE POINTER-CALL ENTRY --
    which is what the caller asserts, and what a reordering breaks -- and NOT
    the completion of both operations. THE POINTER WRITE'S OWN COMPLETION IS
    NOT OBSERVED HERE; the end-state node is what covers it.

    Returns the observed sequence, which the caller asserts against production's
    own order.
    """

    observed: list[str] = []
    real_publish = CheckpointCatalog.publish
    real_write_latest = fixture_module.write_latest

    def wrapped_publish(self, ref):
        result = real_publish(self, ref)
        observed.append("catalog_published")
        return result

    def wrapped_write_latest(root, checkpoint_dir, **kwargs):
        # THE FILESYSTEM PRECONDITION, checked at the production call site: at
        # the moment the pointer is about to exist, the catalog row for this
        # generation must ALREADY be on disk. Read from publications.jsonl, not
        # from anything this test recorded.
        rows = []
        catalog_path = publication_catalog_path(root)
        if catalog_path.is_file():
            rows = [ref.checkpoint_dir.name for ref in read_publications(catalog_path)]
        observed.append(f"latest_written(catalog_rows={sorted(rows)})")
        result = real_write_latest(root, checkpoint_dir, **kwargs)
        return result

    monkeypatch.setattr(CheckpointCatalog, "publish", wrapped_publish)
    monkeypatch.setattr(fixture_module, "write_latest", wrapped_write_latest)
    return observed


def test_the_real_publish_takes_effect_before_the_real_latest_write(
    tmp_path, monkeypatch
) -> None:
    """C10. Observed at the operations themselves, not at any label.

    This is the check that rejects a reversed execution EVEN WHEN the
    declaration and the emitted boundary records still show the expected order.
    Agreement among self-reports cannot establish the property, so this reads
    none of them.
    """

    observed = _observe_real_operation_order(monkeypatch)

    root = tmp_path / "root"
    system = ContinuationSystem.fresh(seed=909)
    system.run(CHECKPOINT_EVERY)
    declared: list[str] = []
    publish_generation(
        root,
        system,
        new_attempt("real-order", 0, None),
        credited_steps=(0, 1),
        boundary_log=declared,
    )

    assert len(observed) == 2, f"expected both operations to run once: {observed}"
    assert observed[0] == "catalog_published", (
        "the real catalog publication did not take effect before the real "
        f"latest-pointer write. Observed order: {observed}"
    )
    assert observed[1].startswith("latest_written"), observed

    # AND THE FILESYSTEM PROPERTY, read at the production call site: when the
    # pointer was about to be written, this generation's row was already there.
    assert observed[1] == "latest_written(catalog_rows=['step_000002'])", (
        "at the moment latest.json was published the catalog did not already "
        f"contain this generation's row: {observed[1]}"
    )

    # The declaration agreed here -- but that agreement is NOT what this test
    # rests on, and a mutant can preserve it while reversing the operations.
    assert declared.index("catalog_published") < declared.index("latest_written")


def test_no_completed_publish_or_post_commit_interruption_leaves_an_unbacked_pointer(
    tmp_path,
) -> None:
    """DURABLE state only, at the moments this samples. Instrument-free.

    WHAT IT ESTABLISHES: at each moment it looks, ``latest.json`` does not name
    a generation the catalog fails to carry. It samples exactly twice per
    scenario -- once after a COMPLETED publish, and once after an injected
    interruption in each post-commit window.

    WHAT IT DOES NOT ESTABLISH, and the earlier name asserted otherwise: **A
    TRANSIENT UNBACKED POINTER BETWEEN THOSE MOMENTS IS NOT EXCLUDED.** This
    node was previously called ``..._is_never_visible_...``, which claims a
    universal over every instant while the implementation examines one instant
    after the fact. A verifier produced exactly that transient state --
    ``latest.json`` naming ``step_000002`` with an empty catalog, mid-publish,
    via a write that did not route through the wrapped bindings -- and this node
    passed, correctly, because the state was gone before it looked.

    Closing the instantaneous window needs an observation with no installed
    instrument taken DURING a publish; that arm is designed in note
    ``c10-observable-order-design`` and is NOT BUILT.

    It remains instrument-free in the sense that matters for the order check
    beside it: nothing here is emitted by the code under test, so no mutant can
    displace a record -- there is no record.
    """

    root = tmp_path / "root"
    system = ContinuationSystem.fresh(seed=1_009)
    system.run(CHECKPOINT_EVERY)
    publish_generation(root, system, new_attempt("fs-order", 0, None), credited_steps=(0, 1))

    def _pointer_is_backed_by_a_row() -> bool:
        named = pointer_target_name(root)
        if named is None:
            return True  # no pointer yet: nothing to be unbacked
        catalog_path = publication_catalog_path(root)
        if not catalog_path.is_file():
            return False
        return named in {ref.checkpoint_dir.name for ref in read_publications(catalog_path)}

    assert _pointer_is_backed_by_a_row(), "a completed publish left an unbacked pointer"

    # Every post-commit interruption window, checked on disk. The pointer must
    # either still name the previous generation (itself backed) or name one the
    # catalog carries -- never a generation with no row.
    for point in _UNACKNOWLEDGED_ARMS:
        window_root = tmp_path / f"window-{point.value}"
        _publish_two_generations(window_root, FaultPlan(point, FaultAction.RAISE))
        named = pointer_target_name(window_root)
        catalog_path = publication_catalog_path(window_root)
        rows = (
            {ref.checkpoint_dir.name for ref in read_publications(catalog_path)}
            if catalog_path.is_file()
            else set()
        )
        assert named in rows, (
            f"after an interruption at {point.value} the pointer names {named!r} "
            f"but the catalog carries {sorted(rows)} - a pointer with no row"
        )
