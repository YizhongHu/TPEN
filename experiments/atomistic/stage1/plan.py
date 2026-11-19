"""Write a deterministic Stage-1 manifest; no run is launched here."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Direct script execution puts only this directory on sys.path.  Add the
# checkout root before importing the source-owned experiments package.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.atomistic.stage1.protocol import (
    SYSTEMS, STAGE1_PROTOCOL, bind_evaluation_manifest, build_manifest,
    config_identities, manifest_hash, overlay_sha256,
)


def main() -> int:
    """Expand explicit CLI identities and checkpoint bindings into JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--config-sha256", help="legacy digest input; two-phase modes recompute it")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--checkpoint-bindings", type=Path,
                        help="phase-2 evidence envelope, or one-shot digest map")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--overlay", help="canonical train-side diagnostics overlay")
    parser.add_argument("--overlay-sha256", help="expected exact overlay digest")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--phase1-manifest", type=Path,
                        help="phase-1 manifest to bind during phase 2")
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--lineage-seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    systems = tuple(args.systems) if args.systems else None
    if args.overlay_sha256 is not None and args.overlay is None:
        raise ValueError("--overlay-sha256 requires --overlay")
    effective_overlay_sha = (
        None if args.overlay is None else args.overlay_sha256 or overlay_sha256(args.overlay)
    )
    config_root = Path(__file__).resolve().parent / "configs"
    config_sha256: str | dict[str, str]
    if args.phase == 1:
        config_sha256 = config_identities(config_root=config_root, systems=systems)
        _compare_legacy_config_sha256(args.config_sha256, config_sha256)
        rows = build_manifest(protocol=STAGE1_PROTOCOL, source_sha=args.source_sha,
                              config_sha256=config_sha256, checkpoints=args.checkpoints,
                              lineage_seeds=args.lineage_seeds, checkpoint_sha256={},
                              results_root=args.results_root, systems=systems, phase=1,
                              device=args.device, overlay=args.overlay,
                              overlay_sha256=effective_overlay_sha)
    else:
        if args.phase1_manifest is not None:
            phase1_doc = json.loads(args.phase1_manifest.read_text(encoding="utf-8"))
            if phase1_doc.get("phase") != 1:
                raise ValueError("--phase1-manifest must contain a phase-1 manifest")
            phase1_rows = phase1_doc.get("rows")
            phase1_hash = phase1_doc.get("manifest_hash")
            if (not isinstance(phase1_rows, list)
                    or not isinstance(phase1_hash, str)
                    or phase1_hash != manifest_hash(phase1_rows)):
                raise ValueError("--phase1-manifest has an invalid manifest_hash")
            phase1_systems = tuple(dict.fromkeys(
                str(row["system"]) for row in phase1_rows if row.get("kind") == "train"
            ))
            config_sha256 = config_identities(config_root=config_root, systems=phase1_systems)
            _compare_legacy_config_sha256(args.config_sha256, config_sha256)
            eval_rows = bind_evaluation_manifest(
                phase1_rows=phase1_rows, checkpoints=args.checkpoints,
                checkpoint_evidence=_read_bindings(args.checkpoint_bindings, structured=True),
                source_sha=args.source_sha,
                config_sha256=config_sha256,
                protocol=STAGE1_PROTOCOL,
                phase1_manifest_hash=phase1_hash,
                device=args.device,
            )
            rows = list(phase1_rows) + eval_rows
        else:
            if args.config_sha256 is not None:
                config_sha256 = args.config_sha256
            else:
                config_sha256 = config_identities(config_root=config_root, systems=systems)
            rows = build_manifest(protocol=STAGE1_PROTOCOL, source_sha=args.source_sha,
                                  config_sha256=config_sha256, checkpoints=args.checkpoints,
                                  lineage_seeds=args.lineage_seeds,
                                  checkpoint_sha256=_read_bindings(args.checkpoint_bindings),
                                  results_root=args.results_root, systems=systems, phase=2,
                                  device=args.device, overlay=args.overlay,
                                  overlay_sha256=effective_overlay_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity_rules = {row.get("config_identity_rule") for row in rows}
    if len(identity_rules) != 1:
        raise ValueError("manifest rows must use one config identity rule")
    args.output.write_text(json.dumps({"schema": "stage1/v3", "phase": args.phase,
                                       "config_identity_rule": next(iter(identity_rules)),
                                       "manifest_hash": manifest_hash(rows),
                                       "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


def _compare_legacy_config_sha256(
    supplied: str | None, computed: dict[str, str],
) -> None:
    """Check a legacy scalar against every computed two-phase identity."""

    if supplied is not None and any(digest != supplied for digest in computed.values()):
        raise ValueError("--config-sha256 does not match the computed two-phase identity")


def _read_bindings(path: Path | None, *, structured: bool = False) -> dict:
    """Decode one-shot digest maps or the structured phase-2 evidence envelope."""

    if path is None:
        raise ValueError("phase 2 requires --checkpoint-bindings")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if structured:
        if not isinstance(raw, dict):
            raise ValueError("phase-2 checkpoint evidence must be a JSON object")
        return raw
    if not isinstance(raw, dict):
        raise ValueError("checkpoint bindings must be a JSON object")
    bindings = {}
    for key, digest in raw.items():
        parts = key.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid checkpoint binding key {key!r}")
        system, lineage, checkpoint = parts
        bindings[(system, int(lineage), int(checkpoint))] = digest
    return bindings


if __name__ == "__main__":
    raise SystemExit(main())
