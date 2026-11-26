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
    HE_H2_EVAL_CHECKPOINTS, HOOKE_A37, HOOKE_SYSTEMS,
    MANIFEST_SCHEMA, OVERLAY_PROFILE_V1, OVERLAY_PROFILES,
    QHO_Q21, SYSTEMS, STAGE1_PROTOCOL,
    bind_evaluation_manifest, build_manifest, config_identities, derive_smoke_eval_roster,
    manifest_hash,
    overlay_sha256, resolve_overlay_path, protocol_for_science_profile,
    HOOKE2_ACT50_ALL_PROFILES, HOOKE_SCIENCE_PROFILE,
)


def main() -> int:
    """Expand explicit CLI identities and checkpoint bindings into JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--config-sha256", help="legacy digest input; two-phase modes recompute it")
    parser.add_argument("--results-root")
    parser.add_argument("--checkpoint-bindings", type=Path,
                        help="phase-2 evidence envelope, or one-shot digest map")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--overlay", help="canonical train-side diagnostics overlay")
    parser.add_argument("--overlay-sha256", help="expected exact overlay digest")
    parser.add_argument("--overlay-profile", choices=OVERLAY_PROFILES,
                        default=None)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--phase1-manifest", type=Path,
                        help="phase-1 manifest to bind during phase 2")
    parser.add_argument("--checkpoints", type=int, nargs="+")
    parser.add_argument("--lineage-seeds", type=int, nargs="+")
    parser.add_argument("--science-profile")
    parser.add_argument("--evaluation-selection")
    args = parser.parse_args()
    systems = tuple(args.systems) if args.systems else None
    phase1_probe_rows = None
    if args.phase == 2 and args.phase1_manifest is not None:
        phase1_probe = json.loads(args.phase1_manifest.read_text(encoding="utf-8"))
        phase1_probe_rows = phase1_probe.get("rows", [])
        if isinstance(phase1_probe_rows, list):
            systems = tuple(dict.fromkeys(
                str(row["system"])
                for row in phase1_probe_rows
                if isinstance(row, dict) and row.get("kind") == "train"
            )) or systems
            phase1_probe_profiles = {
                row.get("science_profile")
                for row in phase1_probe_rows
                if isinstance(row, dict)
                and row.get("kind") == "train"
                and row.get("science_profile") is not None
            }
            if len(phase1_probe_profiles) > 1:
                raise ValueError("phase-1 rows must use one science profile")
            phase1_probe_profile = next(iter(phase1_probe_profiles), None)
            if (phase1_probe_profile is not None
                    and args.science_profile is not None
                    and args.science_profile != phase1_probe_profile):
                raise ValueError(
                    "explicit --science-profile conflicts with the phase-1 manifest"
                )
            if phase1_probe_profile is not None:
                args.science_profile = str(phase1_probe_profile)
    selected_for_profile = set(systems or SYSTEMS)
    if args.phase == 2 and args.science_profile is None \
            and selected_for_profile.issubset(HOOKE_SYSTEMS):
        args.science_profile = "hooke-hv2-v1"
    if args.phase == 1:
        if (args.source_sha is None or args.results_root is None
                or args.lineage_seeds is None):
            parser.error("phase 1 requires --source-sha, --results-root and --lineage-seeds")
        if systems is None or (
            selected_for_profile & HOOKE_SYSTEMS
            and not selected_for_profile.issubset(HOOKE_SYSTEMS)
        ):
            parser.error(
                "phase 1 refuses selections containing Hooke systems without the "
                "Hooke science profile; choose Hooke-only systems or a non-Hooke selection"
            )
        effective_protocol = protocol_for_science_profile(STAGE1_PROTOCOL, args.science_profile)
        if args.checkpoints is None:
            if effective_protocol.retention_cadence is None:
                parser.error("--checkpoints is required for protocols without retention_cadence")
            args.checkpoints = list(range(
                0, effective_protocol.lineage_updates + 1,
                effective_protocol.retention_cadence,
            ))
        if args.science_profile is None and selected_for_profile.issubset(HOOKE_SYSTEMS):
            args.science_profile = "hooke-hv2-v1"
    else:
        effective_protocol = protocol_for_science_profile(STAGE1_PROTOCOL, args.science_profile)
    if args.phase == 2 and args.checkpoints is None:
        if args.science_profile == "hooke-hv2-v1" and selected_for_profile == {"qho2e"}:
            args.checkpoints = list(QHO_Q21)
        elif args.science_profile == "hooke-hv2-v1":
            args.checkpoints = list(HOOKE_A37)
        elif args.science_profile in HOOKE2_ACT50_ALL_PROFILES:
            from experiments.atomistic.stage1.protocol import HOOKE2_ACT50_A27
            args.checkpoints = list(HOOKE2_ACT50_A27)
        elif selected_for_profile.issubset({"he", "h2"}):
            args.checkpoints = list(HE_H2_EVAL_CHECKPOINTS)
        else:
            parser.error("--checkpoints is required for phase 2")
    if args.phase == 2 and args.phase1_manifest is None and (
        args.source_sha is None or args.results_root is None or args.lineage_seeds is None
    ):
        parser.error("one-shot phase 2 requires --source-sha, --results-root and --lineage-seeds")
    if args.overlay_sha256 is not None and args.overlay is None:
        raise ValueError("--overlay-sha256 requires --overlay")
    effective_overlay_profile = args.overlay_profile or OVERLAY_PROFILE_V1
    effective_overlay_sha = (
        None if args.overlay is None else args.overlay_sha256 or overlay_sha256(args.overlay)
    )
    config_root = Path(__file__).resolve().parent / "configs"
    identity_profile = (
        args.science_profile if args.science_profile in HOOKE2_ACT50_ALL_PROFILES else None
    )
    config_sha256: str | dict[str, str]
    if args.phase == 1:
        config_sha256 = config_identities(
            config_root=config_root, systems=systems, science_profile=identity_profile,
        )
        _compare_legacy_config_sha256(args.config_sha256, config_sha256)
        rows = build_manifest(protocol=effective_protocol, source_sha=args.source_sha,
                              config_sha256=config_sha256, checkpoints=args.checkpoints,
                              lineage_seeds=args.lineage_seeds, checkpoint_sha256={},
                              results_root=args.results_root, systems=systems, phase=1,
                              device=args.device, overlay=args.overlay,
                              overlay_sha256=effective_overlay_sha,
                              overlay_profile=effective_overlay_profile,
                              science_profile=args.science_profile,
                              evaluation_selection=args.evaluation_selection,
                              include_smoke=False, auto_overlay=args.overlay is None)
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
            phase1_train_rows = [row for row in phase1_rows if row.get("kind") == "train"]
            phase1_profiles = {
                str(row.get("overlay_profile", OVERLAY_PROFILE_V1))
                for row in phase1_train_rows
            }
            if len(phase1_profiles) != 1:
                raise ValueError("phase-1 rows must use one diagnostics overlay profile")
            phase1_profile = next(iter(phase1_profiles))
            if args.overlay_profile is not None and args.overlay_profile != phase1_profile:
                raise ValueError(
                    "explicit --overlay-profile conflicts with the phase-1 manifest"
                )
            phase1_overlays = {row.get("overlay") for row in phase1_train_rows}
            phase1_overlay_shas = {row.get("overlay_sha256") for row in phase1_train_rows}
            if args.overlay is not None:
                if len(phase1_overlays) != 1 or len(phase1_overlay_shas) != 1:
                    raise ValueError(
                        "explicit --overlay conflicts with the phase-1 manifest"
                    )
                phase1_overlay = next(iter(phase1_overlays))
                phase1_overlay_sha = next(iter(phase1_overlay_shas))
                if phase1_overlay is None or phase1_overlay_sha is None:
                    raise ValueError(
                        "explicit --overlay conflicts with the phase-1 manifest"
                    )
                if (resolve_overlay_path(args.overlay).resolve()
                        != resolve_overlay_path(str(phase1_overlay)).resolve()
                        or (args.overlay_sha256 is not None
                            and args.overlay_sha256 != phase1_overlay_sha)
                        or overlay_sha256(args.overlay) != phase1_overlay_sha):
                    raise ValueError(
                        "explicit --overlay conflicts with the phase-1 manifest"
                    )
            phase1_science_profiles = {
                row.get("science_profile") for row in phase1_train_rows
                if row.get("science_profile") in {
                    HOOKE_SCIENCE_PROFILE, *HOOKE2_ACT50_ALL_PROFILES,
                }
            }
            if len(phase1_science_profiles) > 1:
                raise ValueError("phase-1 rows must use one science profile")
            phase1_science_profile = next(iter(phase1_science_profiles), args.science_profile)
            if any(row.get("evaluation_roster") is not None for row in phase1_rows):
                args.checkpoints = list(derive_smoke_eval_roster(
                    phase1_rows, science_roster=args.checkpoints,
                ))
            config_sha256 = config_identities(
                config_root=config_root, systems=phase1_systems,
                science_profile=(phase1_science_profile
                                 if phase1_science_profile in HOOKE2_ACT50_ALL_PROFILES else None),
            )
            _compare_legacy_config_sha256(args.config_sha256, config_sha256)
            eval_rows = bind_evaluation_manifest(
                phase1_rows=phase1_rows, checkpoints=args.checkpoints,
                checkpoint_evidence=_read_bindings(args.checkpoint_bindings, structured=True),
                source_sha=args.source_sha,
                config_sha256=config_sha256,
                protocol=protocol_for_science_profile(STAGE1_PROTOCOL, phase1_science_profile),
                phase1_manifest_hash=phase1_hash,
                device=args.device,
            )
            rows = list(phase1_rows) + eval_rows
        else:
            if args.config_sha256 is not None:
                config_sha256 = args.config_sha256
            else:
                config_sha256 = config_identities(
                    config_root=config_root, systems=systems,
                    science_profile=identity_profile,
                )
            rows = build_manifest(protocol=effective_protocol, source_sha=args.source_sha,
                                  config_sha256=config_sha256, checkpoints=args.checkpoints,
                                  lineage_seeds=args.lineage_seeds,
                                  checkpoint_sha256=_read_bindings(args.checkpoint_bindings),
                                  results_root=args.results_root, systems=systems, phase=2,
                                  device=args.device, overlay=args.overlay,
                                  overlay_sha256=effective_overlay_sha,
                                  overlay_profile=effective_overlay_profile,
                                  science_profile=args.science_profile,
                                  evaluation_selection=args.evaluation_selection,
                                  include_smoke=False, auto_overlay=args.overlay is None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity_rules = {row.get("config_identity_rule") for row in rows}
    if len(identity_rules) != 1:
        raise ValueError("manifest rows must use one config identity rule")
    args.output.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "phase": args.phase,
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
