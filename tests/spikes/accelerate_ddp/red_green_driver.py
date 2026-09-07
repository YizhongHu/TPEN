"""Run every named mutation arm: green, red under the mutant, green again.

Per arm, in order:

1. Confirm the anchor occurs EXACTLY ONCE. Refuse otherwise -- a mutation that
   silently fails to apply produces a FALSE GREEN, and one that applies twice
   produces an unattributable red.
2. Record the file's sha256 before touching it.
3. Run the node id expecting SUCCESS, so the arm's red is attributable to the
   mutation rather than to a test that was already failing.
4. Apply the mutation; run the same node id expecting FAILURE.
5. Restore the original bytes; verify sha256 equals the pre-mutation digest.
6. Run the node id again expecting SUCCESS.

The restore is in a ``finally`` block: a driver that died mid-arm would otherwise
leave the working tree mutated, and every later arm would then be measuring the
wreckage of this one.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from tests.spikes.accelerate_ddp.mutation_plan import ENVIRONMENT_ARMS, MUTATIONS

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_node(node_id: str) -> int:
    """Run one pytest node id and return its exit code."""

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", node_id, "-p", "no:randomly", "-q", "--no-header"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    tail = completed.stdout.strip().splitlines()[-3:]
    for line in tail:
        print(f"      | {line}")
    # A node id that selects NOTHING exits 4 (usage error) or reports "no tests
    # ran". Either way it must never be read as a pass: a stale parametrize label
    # would otherwise turn a whole arm into a silent no-op.
    if "no tests ran" in completed.stdout:
        print("      | REFUSE: node id selected no tests")
        return 99
    return completed.returncode


def main() -> int:
    ok = 0
    fail = 0
    print(f"MUTATION ARMS: {len(MUTATIONS)} source, {len(ENVIRONMENT_ARMS)} environment")

    for mutation in MUTATIONS:
        print(f"\n{'-' * 70}\nARM {mutation.name}  [{mutation.kind}]")
        print(f"  file  {mutation.path}")
        print(f"  node  {mutation.node_id}")
        path = REPO_ROOT / mutation.path
        original = path.read_text()

        occurrences = original.count(mutation.anchor)
        if occurrences != 1:
            print(f"  REFUSE: anchor occurs {occurrences} times, expected exactly 1")
            fail += 1
            continue

        digest_before = _sha256(path)
        arm_ok = True
        try:
            print("  [1/3] baseline expecting GREEN")
            if _run_node(mutation.node_id) != 0:
                print("  REFUSE: baseline is not green; a red here is not attributable")
                fail += 1
                continue

            print("  [2/3] mutant expecting RED")
            path.write_text(original.replace(mutation.anchor, mutation.replacement))
            mutant_rc = _run_node(mutation.node_id)
            if mutant_rc == 0:
                print("  FAIL: the gate PASSED under the mutant; it does not detect this defect")
                arm_ok = False
            elif mutant_rc == 99:
                print("  FAIL: node id selected nothing under the mutant")
                arm_ok = False
            else:
                print(f"  red observed (rc={mutant_rc})")
        finally:
            # Restore unconditionally: a mutated tree would poison every later arm.
            path.write_text(original)

        digest_after = _sha256(path)
        if digest_after != digest_before:
            print(f"  FAIL: restore is not byte-identical\n    before {digest_before}\n    after  {digest_after}")
            arm_ok = False
        else:
            print(f"  restored byte-identically (sha256 {digest_after[:16]}...)")

        print("  [3/3] restored expecting GREEN")
        if _run_node(mutation.node_id) != 0:
            print("  FAIL: the gate is not green after restore")
            arm_ok = False

        if arm_ok:
            ok += 1
            print(f"  ARM OK: {mutation.defect}")
        else:
            fail += 1

    print(f"\n{'=' * 70}\nENVIRONMENT ARMS (no source mutant; different restoration guarantee)")
    for arm in ENVIRONMENT_ARMS:
        print(f"  {arm['name']}: {arm['how']}")
        print(f"    observed red in {arm['observed_red_in']} -- NOT re-run by this driver")

    print(f"\nRED_GREEN_SUMMARY OK={ok} FAIL={fail} TOTAL={len(MUTATIONS)}")
    print("RED_GREEN_DRIVER_COMPLETE")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
