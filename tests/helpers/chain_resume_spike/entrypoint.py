"""Child-process entrypoint for one chain-resume continuation attempt.

Run as ``python -m tests.helpers.chain_resume_spike.entrypoint``, never
imported and called in-process by a test. That is not a style choice: an
in-process resume leaves the parent's live RNG objects in memory, so a
continuation parity gate passes without anything having been restored at all.
Every attempt in this package is a fresh OS process for that reason.

The attempt seeds itself from OS entropy, *distinct from its parent by
construction*, and records the seed in its receipt so distinctness is auditable
rather than assumed. It then restores the newest valid committed generation --
selected by production's own
``tpen.checkpoint.artifact.list_complete_checkpoints`` -- under the restore
policy it was given, executes up to its step budget, commits generations on its
cadence, and writes a durable typed receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import traceback
from pathlib import Path

from tpen.checkpoint.artifact import require_complete_checkpoint_dir

from .faults import FaultPlan, read_fault_plan
from .fixture import (
    ContinuationSystem,
    resume_generation,
    publish_generation,
    read_generation_payload,
    read_generation_provenance,
    restore_system,
)
from .identity import CreditedLedger, new_attempt
from .outcomes import AttemptOutcome, OutcomeKind
from .receipt import AttemptReceipt, ExitKind, write_receipt
from .restore_limbs import RestorePolicy


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse this attempt's command line."""

    parser = argparse.ArgumentParser(description="One chain-resume continuation attempt")
    parser.add_argument("--root", required=True, help="Shared checkpoint root")
    parser.add_argument("--run-id", required=True, help="Stable logical run id")
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True, help="Step budget before yielding")
    parser.add_argument("--total-target", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, required=True)
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument(
        "--disable-limb",
        action="append",
        default=[],
        help="Limb token whose restore is skipped; repeatable",
    )
    parser.add_argument("--fault-plan-path", default=None)
    parser.add_argument(
        "--restore-from",
        default=None,
        help=(
            "Explicit committed generation to continue from. When omitted, the "
            "newest selectable generation under --root is used, via production's "
            "own list_complete_checkpoints. Naming the parent explicitly is what "
            "lets the G0 comparison restore an EARLIER generation of a run that "
            "has since committed later ones."
        ),
    )
    parser.add_argument(
        "--publish-only",
        default=None,
        help=(
            "Path-spelling probe. Publish an ALREADY-COMMITTED generation to the "
            "catalog under whatever spelling this path is written in, then exit. "
            "Runs no steps and commits nothing new."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the OS-entropy seed. Test-only, for a deliberate seed-collision control.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute one attempt and return its exit status."""

    args = _parse_args(argv)
    root = Path(args.root)
    receipt_path = Path(args.receipt_path)
    policy = RestorePolicy.from_tokens(tuple(args.disable_limb))
    fault: FaultPlan | None = (
        None if args.fault_plan_path is None else read_fault_plan(Path(args.fault_plan_path))
    )

    if args.publish_only is not None:
        return _publish_only(Path(args.publish_only), root, receipt_path)

    # Fresh OS entropy, drawn before anything is restored. Distinct from the
    # parent with overwhelming probability, and RECORDED, so that "the fresh
    # process did not accidentally reproduce the parent's stream" is a checked
    # fact rather than an assumption.
    process_seed = secrets.randbits(63) if args.seed is None else int(args.seed)
    system = ContinuationSystem.fresh(process_seed)

    ledger = CreditedLedger(total_target=args.total_target)
    limb_application = None
    parent_generation: int | None = None
    already_credited: set[int] = set()

    if args.restore_from is not None:
        # An explicitly named parent still goes through production's validity
        # check: ``require_complete_checkpoint_dir`` rejects a ``.tmp`` name, a
        # missing manifest and a missing COMPLETE marker alike.
        parent_dir = require_complete_checkpoint_dir(Path(args.restore_from))
    else:
        # THE PRODUCTION POINTER PATH, not a directory listing. ``restore.py:118``
        # resolves through ``latest.json`` and never lists, so a fixture that
        # listed here would select a generation production would not -- measured
        # to differ in both pre-latest fault windows.
        parent_dir = resume_generation(root)
    if parent_dir is not None:
        provenance = read_generation_provenance(parent_dir)
        payload = read_generation_payload(parent_dir)
        # Rebuild the ledger from the committed generation BEFORE executing, so
        # a step the chain already credited cannot be credited a second time --
        # ``CreditedLedger.credit`` raises rather than silently accepting it.
        for step in provenance.get("credited_steps", []):
            ledger.credit(int(step))
            already_credited.add(int(step))
        limb_application = restore_system(system, payload, policy)
        parent_generation = int(system.step)

    identity = new_attempt(args.run_id, args.attempt_index, parent_generation)

    trace: list[dict[str, object]] = []
    credited: list[int] = []
    replayed: list[int] = []
    exit_status = 0
    outcome: AttemptOutcome

    try:
        for _ in range(args.steps):
            if system.step >= args.total_target:
                break
            row = system.step_once()
            trace.append(row)
            step_index = int(row["step"])
            if step_index in already_credited:
                # Re-executed but NOT re-credited. Recorded separately so the
                # difference between an honest replay and a double count is
                # visible in the evidence rather than inferred.
                replayed.append(step_index)
            else:
                ledger.credit(step_index)
                credited.append(step_index)
            if system.step % args.checkpoint_every == 0:
                publish_generation(
                    root,
                    system,
                    identity,
                    credited_steps=tuple(ledger.credited_steps),
                    fault=fault,
                )
    except BaseException as exc:  # noqa: BLE001 - re-raised below after recording
        # Recorded, then re-raised. A fault arm's diagnostic must survive even
        # though the attempt is about to fail; swallowing it here would turn a
        # deliberate failure into a silent one.
        traceback.print_exc()
        outcome = AttemptOutcome(
            kind=OutcomeKind.TRANSIENT,
            reason="attempt_raised",
            credited_through=max(credited) if credited else None,
            next_parent_generation=_newest_step(root),
            detail=f"{type(exc).__name__}: {exc}",
        )
        _emit(
            receipt_path,
            identity,
            outcome,
            process_seed,
            trace,
            credited,
            replayed,
            policy,
            limb_application,
            fault,
            # EARNED: the attempt really did fail, and the non-zero status
            # below reflects that rather than being chosen by the harness.
            ExitKind.EARNED,
        )
        return 1

    if ledger.is_complete:
        outcome = AttemptOutcome(
            kind=OutcomeKind.FINAL,
            reason="total_target_reached",
            credited_through=max(credited) if credited else None,
            next_parent_generation=_newest_step(root),
        )
    else:
        outcome = AttemptOutcome(
            kind=OutcomeKind.YIELD,
            reason="step_budget_reached",
            credited_through=max(credited) if credited else None,
            next_parent_generation=_newest_step(root),
        )

    _emit(
        receipt_path,
        identity,
        outcome,
        process_seed,
        trace,
        credited,
        replayed,
        policy,
        limb_application,
        fault,
        ExitKind.EARNED,
    )
    return exit_status


def _publish_only(generation: Path, root: Path, output: Path) -> int:
    """Publish one already-committed generation and report what production did.

    The whole point of running this in a SEPARATE PROCESS with a caller-chosen
    working directory is that the *spelling* of ``generation`` is then genuinely
    this process's own -- a relative path resolved against a different cwd, or a
    symlinked root -- rather than a string the parent constructed. Nothing in
    ``tpen/checkpoint`` canonicalises a path, so spelling is preserved all the
    way into the catalog comparison.

    Reports the outcome as JSON instead of raising, so the caller can assert on
    the EXACT exception type and message rather than on a process exit code.
    """

    from tpen.checkpoint.catalog import CheckpointCatalog, publication_catalog_path
    from tpen.checkpoint.reference import CheckpointRef

    result: dict[str, object] = {"spelling": str(generation), "cwd": os.getcwd()}
    try:
        ref = CheckpointRef.from_directory(generation)
        result["content_id"] = ref.content_id
        result["serialized_checkpoint_dir"] = ref.to_dict()["checkpoint_dir"]
        CheckpointCatalog(publication_catalog_path(root)).publish(ref)
        result["published"] = True
        result["error_type"] = None
        result["error"] = None
    except Exception as exc:  # noqa: BLE001 - the exception IS the measurement
        result["published"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)

    # The asymmetry this probe exists to show: the pointer path is spelling
    # independent, so restore still selects, under the very same spelling that
    # publish refused.
    try:
        resolved = resume_generation(root)
        result["pointer_resolves_to"] = None if resolved is None else resolved.name
        result["pointer_error"] = None
    except Exception as exc:  # noqa: BLE001
        result["pointer_resolves_to"] = None
        result["pointer_error"] = f"{type(exc).__name__}: {exc}"

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


def _newest_step(root: Path) -> int | None:
    """Return the step the resume pointer currently names, or ``None``.

    Reads the same surface a resume would, so a receipt's
    ``next_parent_generation`` names what the next attempt will actually get.
    """

    newest = resume_generation(root)
    if newest is None:
        return None
    return int(json.loads((newest / "manifest.json").read_text(encoding="utf-8"))["next_iteration"])


def _emit(
    receipt_path: Path,
    identity: object,
    outcome: AttemptOutcome,
    process_seed: int,
    trace: list[dict[str, object]],
    credited: list[int],
    replayed: list[int],
    policy: RestorePolicy,
    limb_application: object,
    fault: FaultPlan | None,
    exit_kind: ExitKind,
) -> None:
    """Write this attempt's durable receipt."""

    write_receipt(
        receipt_path,
        AttemptReceipt(
            identity=identity,  # type: ignore[arg-type]
            outcome=outcome,
            process_seed=process_seed,
            pid=os.getpid(),
            # Absolute by construction, and recorded so no claim about this
            # attempt's environment has to be taken on trust.
            executable=sys.executable,
            trace=trace,  # type: ignore[arg-type]
            credited_steps=tuple(credited),
            replayed_steps=tuple(replayed),
            disabled_limbs=policy.to_tokens(),
            limb_application=(
                None if limb_application is None else limb_application.to_dict()  # type: ignore[attr-defined]
            ),
            fault=None if fault is None else fault.to_dict(),
            exit_kind=exit_kind,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
