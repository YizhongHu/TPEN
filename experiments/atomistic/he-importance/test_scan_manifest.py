"""Tests for the frozen O1 scan manifest and manifest-backed selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("hi_mf1_scan_manifest_test", HERE / "scan_manifest.py")
assert spec is not None and spec.loader is not None
manifest_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = manifest_module
spec.loader.exec_module(manifest_module)


def test_committed_manifest_regenerates_byte_identically() -> None:
    path = HERE / "o1_scan_manifest_v1.json"
    assert path.stat().st_size < 1_000_000
    assert path.read_text(encoding="utf-8") == manifest_module.render_manifest_text()


def test_manifest_counts_sources_and_admission_reason() -> None:
    manifest = manifest_module.load_manifest()
    assert manifest["counts"] == {"rows": 1920, "admitted": 128, "unavailable": 1792}
    assert manifest["horizon"] == 50000
    assert manifest["walkers"] == 4096
    assert manifest["checkpoint_cadence"]["identity"] is False
    assert manifest["provenance"]["design_document"]["sha256"] == manifest_module.DESIGN_DOC_SHA256
    assert len(manifest["cells"]) == 120
    assert len(manifest["contexts"]) == 2
    sr = next(row for row in manifest["rows"] if row.get("status") == "unavailable" and row["cell_id"])
    assert "unavailable_reason" in sr
    assert "EXCLUDED from the helium-importance scan" in sr["unavailable_reason"] or sr["unavailable_reason"]


def test_manifest_tampering_and_unknown_rows_refuse(tmp_path: Path) -> None:
    manifest = manifest_module.load_manifest()
    broken = dict(manifest)
    broken["rows"] = list(manifest["rows"])
    broken["rows"][0] = dict(broken["rows"][0], seed_label=999)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(manifest_module.ScanManifestError, match="hash mismatch"):
        manifest_module.load_manifest(path)
    with pytest.raises(manifest_module.ScanManifestError, match="unknown manifest row"):
        manifest_module.find_row(manifest, "missing-row")

