"""Write a deterministic Stage-1 row manifest before any launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol import Protocol, build_manifest, manifest_hash


def main() -> int:
    """Expand the fixed Stage-1 protocol into JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 10000, 25000, 50000])
    parser.add_argument("--seeds", type=int, nargs=3, default=[73001, 73002, 73003])
    args = parser.parse_args()
    rows = build_manifest(Protocol(), checkpoints=args.checkpoints, seeds=args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "stage1/v1", "manifest_hash": manifest_hash(rows),
                                       "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
