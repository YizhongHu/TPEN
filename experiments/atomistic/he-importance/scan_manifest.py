"""Frozen O1 helium-importance scan manifest and its fail-closed loader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from collections.abc import Mapping
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
MANIFEST_SCHEMA = "tpen.hi.scan_manifest.v1"
MANIFEST_PATH = _HERE / "o1_scan_manifest_v1.json"
DESIGN_DOC_SHA256 = "76e6b7bcaf5454da120750e353fb268bbce07e6aeda6d2a601c86ddf7cc8046f"
MANAGER_RULING = "Paseo msg c92c3a08 (2026-11-14): SR/CAD/DOC"

_COORDINATE_SOURCES = {
    "stage": {"citation": "consolidated design sections 2.8 stage table; optimizer_grid_authority.json", "evidence_class": "RELAYED"},
    "optimizer": {"citation": "consolidated design section 2.7; optimizer_grid_authority.json", "evidence_class": "RELAYED"},
    "context": {"citation": "consolidated design section 2.8; optimizer_grid_authority.json", "evidence_class": "RELAYED"},
    "seed": {"citation": "operator-relayed eight seeds in consolidated design section 2.8; labels implemented in stage_coordinate.seed_labels", "evidence_class": ["RELAYED", "INFERRED"]},
    "horizon": {"citation": "committed stage_coordinate_authority.json payload.updates; consolidated design section 2.8", "evidence_class": "RELAYED"},
    "walkers": {"citation": "operator ruling 2026-10-24, clause 2: 4096 stays the literal control for later scientific stages", "evidence_class": "RELAYED"},
    "admission": {"citation": "tpen.hi_schema.HI_METHOD_ROSTER; consolidated design section 2.8; manager ruling c92c3a08", "evidence_class": "RELAYED"},
    "cadence": {"citation": "manager ruling c92c3a08: checkpoint cadence is operational and excluded from identity", "evidence_class": "RELAYED"},
}
_SOURCE_ALIASES = {"s": "stage", "o": "optimizer", "c": "context", "d": "seed", "h": "horizon", "w": "walkers", "a": "admission"}
_ROW_SOURCE_REFS = "s,o,c,d,h,w,a"


class ScanManifestError(ValueError):
    """A manifest is malformed, tampered, stale, or selects an unavailable row."""


def checkpoint_cadence_declaration(*, updates: int | None = None) -> dict[str, bool]:
    """Describe per-run cadence without admitting its operational value to identity."""

    del updates
    return {"operational": True, "recorded_per_run": True, "in_identity": False}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _authority_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((_HERE / name).read_bytes()).hexdigest()
        for name in ("optimizer_grid_authority.json", "stage_coordinate_authority.json")
    }


def _materialized_rows() -> dict[str, Any]:
    def load(name: str, filename: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, _HERE / filename)
        if spec is None or spec.loader is None:
            raise ScanManifestError(f"cannot load study module {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    stage = load("hi_mf1_manifest_stage_coordinate", "stage_coordinate.py")
    grid = load("hi_mf1_manifest_optimizer_grid", "optimizer_grid.py")
    availability = grid.roster_availability()
    declared = {
        (cell.family, _canonical(dict(cell.levels))): cell.cell_id
        for cell in grid.declared_cells()
    }
    contexts = [
        {"name": context["name"], "overrides": dict(context["overrides"]),
         "approval_sources": ["context"]}
        for context in grid.load_authority()["contexts"]
    ]
    with tempfile.TemporaryDirectory(prefix="hi-mf1-manifest-") as root:
        cells = stage.materialize_intended_configurations(
            Path(root), availability=availability, stages=("O1",)
        )
        horizons = {cell.manifest["payload"]["updates"] for cell in cells}
        walkers = {stage.walker_count(cell.manifest) for cell in cells}
        if len(horizons) != 1:
            raise ScanManifestError(f"O1 materialized horizons are not uniform: {sorted(horizons)!r}")
        if len(walkers) != 1:
            raise ScanManifestError(f"O1 materialized walker counts are not uniform: {sorted(walkers)!r}")
        horizon = next(iter(horizons))
        walker_count = next(iter(walkers))
        rows = []
        for cell in cells:
            identity = cell.manifest["scientific_identity"]
            optimizer = identity["optimizer"]
            optimizer_cell = identity["optimizer_cell"]
            cell_id = declared[(optimizer["method"], _canonical(dict(optimizer["levels"])))]
            rows.append(
                {
                    "row_id": cell.content_hash,
                    "cell_id": cell_id,
                    "context": identity["context"]["name"],
                    "seed_label": cell.manifest["seed_identity"]["label"],
                    "status": optimizer_cell["status"],
                    "approval_sources": _ROW_SOURCE_REFS,
                    **(
                        {"unavailable_reason": optimizer_cell["unavailable_reason"]}
                        if "unavailable_reason" in optimizer_cell
                        else {}
                    ),
                }
            )
    cells = [
        {
            "cell_id": cell.cell_id,
            "optimizer": {"method": cell.family, "levels": dict(cell.levels)},
            "approval_sources": ["optimizer", "admission", "horizon", "walkers"],
            "horizon": horizon,
            "walkers": walker_count,
        }
        for cell in grid.declared_cells()
    ]
    return {
        "rows": tuple(rows),
        "cells": tuple(cells),
        "contexts": tuple(contexts),
        "horizon": horizon,
        "walkers": walker_count,
    }


def render_manifest() -> dict[str, Any]:
    """Render the complete deterministic O1 manifest from live study authorities."""

    materialized = _materialized_rows()
    rows = materialized["rows"]
    if len(rows) != 1920:
        raise ScanManifestError(f"O1 manifest must contain 1920 rows, got {len(rows)}")
    admitted = sum(row["status"] == "available" for row in rows)
    body: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "stage": "O1",
        "horizon": materialized["horizon"],
        "walkers": materialized["walkers"],
        "checkpoint_cadence": checkpoint_cadence_declaration(),
        "provenance": {
            "design_document": {
                "path": "2026-09-04-tpen-helium-importance-scan-consolidated.md",
                "sha256": DESIGN_DOC_SHA256,
                "sections": ["2.7", "2.8", "2.9"],
                "tracked": False,
                "location": "operator checkout (untracked; commit decision pending with operator)",
            },
            "authority_sha256": _authority_hashes(),
            "manager_ruling": MANAGER_RULING,
            "lane_plan_approval": {
                "citation": "TO note operator-approved-materialization-2026-09-05 on 67987933",
                "evidence_class": "RELAYED",
            },
            "source_aliases": _SOURCE_ALIASES,
            "coordinate_sources": _COORDINATE_SOURCES,
        },
        "counts": {"rows": len(rows), "admitted": admitted, "unavailable": len(rows) - admitted},
        "cells": list(materialized["cells"]),
        "contexts": list(materialized["contexts"]),
        "rows": list(rows),
    }
    body["content_hash"] = content_hash(body)
    return body


def render_manifest_text() -> str:
    return json.dumps(render_manifest(), sort_keys=True, separators=(",", ":")) + "\n"


def _body_without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(manifest)
    body.pop("content_hash", None)
    return body


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load and authenticate a committed manifest, refusing any tampering."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScanManifestError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ScanManifestError(f"manifest schema is not {MANIFEST_SCHEMA!r}")
    declared = manifest.get("content_hash")
    if not isinstance(declared, str) or content_hash(_body_without_hash(manifest)) != declared:
        raise ScanManifestError("manifest content hash mismatch")
    if manifest.get("counts") != {"rows": 1920, "admitted": 128, "unavailable": 1792}:
        raise ScanManifestError("manifest counts do not match the frozen O1 design")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 1920:
        raise ScanManifestError("manifest rows must contain exactly 1920 entries")
    if len({row.get("row_id") for row in rows if isinstance(row, Mapping)}) != 1920:
        raise ScanManifestError("manifest row ids are not unique")
    return dict(manifest)


def find_row(manifest: Mapping[str, Any], row_id: str) -> Mapping[str, Any]:
    for row in manifest["rows"]:
        if isinstance(row, Mapping) and row.get("row_id") == row_id:
            return row
    raise ScanManifestError(f"unknown manifest row id {row_id!r}")


def validate_live_row(manifest: Mapping[str, Any], row_id: str) -> Mapping[str, Any]:
    """Re-materialize O1 and require the selected row's content identity to remain live."""

    row = find_row(manifest, row_id)
    live = {candidate["row_id"]: candidate for candidate in _materialized_rows()["rows"]}
    current = live.get(row_id)
    if current is None or dict(current) != dict(row):
        raise ScanManifestError("manifest row identity is stale or tampered")
    if row.get("status") != "available":
        reason = row.get("unavailable_reason", "method not admitted by HI_METHOD_ROSTER")
        raise ScanManifestError(f"manifest row {row_id!r} is unavailable: {reason}")
    return row


def write_manifest(path: Path = MANIFEST_PATH) -> None:
    path.write_text(render_manifest_text(), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    write_manifest()

