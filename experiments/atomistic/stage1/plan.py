"""Write a deterministic Stage-1 manifest; no run is launched here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.atomistic.stage1.protocol import STAGE1_PROTOCOL, build_manifest, manifest_hash


def main() -> int:
    """Expand explicit CLI identities and checkpoint bindings into JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--checkpoint-bindings", type=Path, required=True,
                        help="JSON object keyed by system/lineage/checkpoint")
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--lineage-seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    raw = json.loads(args.checkpoint_bindings.read_text(encoding="utf-8"))
    bindings = {}
    for key, digest in raw.items():
        system, lineage, checkpoint = key.split(":")
        bindings[(system, int(lineage), int(checkpoint))] = digest
    rows = build_manifest(protocol=STAGE1_PROTOCOL, source_sha=args.source_sha,
                          config_sha256=args.config_sha256, checkpoints=args.checkpoints,
                          lineage_seeds=args.lineage_seeds,
                          checkpoint_sha256=bindings, results_root=args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "stage1/v2", "manifest_hash": manifest_hash(rows),
                                       "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
