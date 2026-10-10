#!/usr/bin/env python3
"""Round-6 mutation applier (PR 524, head f2006235). Stdlib only.

Byte-identical payloads to round 5's applier (M-RS dropped: not re-run this
round). Every mutation asserts the exact original line content before writing,
so a drifted line fails loudly instead of mutating the wrong thing. Activity
is proven by sha256 difference plus the printed mutated line, and the mutated
sha256 must equal round 5's recorded value or the arm is void.
"""

import hashlib
import sys
from pathlib import Path

MUTANTS = {
    "M-RC": {
        "file": "tpen/runner/train.py",
        "line": 125,
        "expect": "                optimizer=optimizer,\n",
        "replace": "                optimizer=make_optimizer(self.optimizer, self.model.parameters()),\n",
        "expect_sha": "332f1eaec858af31794efe2f8a47c5a83b1db9f71131f724c28a883038bc9734",
    },
    "M-DB": {
        "file": "tpen/runner/train.py",
        "line": 144,
        "expect": "            optimizer=optimizer,\n",
        "replace": "            optimizer=make_optimizer(self.optimizer, self.model.parameters()),\n",
        "expect_sha": "f36ba3fd12dfd902f0c82c62250d61afda6cfc22140012496f2b18edfd3ea8cd",
    },
    "M-A": {
        "file": "tpen/training/trainer.py",
        "line": 492,
        "expect": "        if optimizer is not self._bound_optimizer:\n",
        "replace": "        if False and optimizer is not self._bound_optimizer:\n",
        "expect_sha": "f19f7227c56a19901b1fe65aee653b59ca08ffe22705afdcc43350e4779a7d44",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    name = sys.argv[1]
    spec = MUTANTS[name]
    path = Path(spec["file"])
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    idx = spec["line"] - 1
    if lines[idx] != spec["expect"]:
        print(
            f"MUTANT {name}: line {spec['line']} content drifted; refusing.\n"
            f"  got:  {lines[idx]!r}\n  want: {spec['expect']!r}"
        )
        return 2
    before = sha256(path)
    lines[idx] = spec["replace"]
    path.write_text("".join(lines), encoding="utf-8")
    after = sha256(path)
    if before == after:
        print(f"MUTANT {name}: INACTIVE (sha unchanged)")
        return 3
    print(f"MUTANT {name} ACTIVE on {spec['file']}: sha256 {before} -> {after}")
    print(f"MUTATED LINE {spec['line']}: {lines[idx].rstrip()}")
    if after != spec["expect_sha"]:
        print(
            f"MUTANT {name}: mutated sha {after} != round-5 recorded "
            f"{spec['expect_sha']}; ARM VOID."
        )
        return 4
    print(f"MUTANT {name}: byte-identical to round 5's payload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
