#!/usr/bin/env python3
"""Round-5 mutation applier (PR 524, head cbf5fcd4). Stdlib only.

Every mutation asserts the exact original line content before writing, so a
drifted line fails loudly instead of mutating the wrong thing. Activity is
proven by sha256 difference plus the printed mutated line.
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
    },
    "M-DB": {
        "file": "tpen/runner/train.py",
        "line": 144,
        "expect": "            optimizer=optimizer,\n",
        "replace": "            optimizer=make_optimizer(self.optimizer, self.model.parameters()),\n",
    },
    "M-RS": {
        "file": "tpen/runner/train.py",
        "line": 121,
        "expect": '        if mode == "train_resume":\n',
        "replace": '        if False and mode == "train_resume":\n',
    },
    "M-A": {
        "file": "tpen/training/trainer.py",
        "line": 492,
        "expect": "        if optimizer is not self._bound_optimizer:\n",
        "replace": "        if False and optimizer is not self._bound_optimizer:\n",
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
