"""Focused T2 reviewer probes for PR849's HI roster/manifest boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest


# The study directory contains a hyphen, so use the same explicit loader shape
# as the study's existing review tests rather than relying on package syntax.
import importlib.util
import sys


HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load("he_importance_stage_coordinate_for_roster_review_r1", "stage_coordinate.py")
optimizer_grid = _load("he_importance_optimizer_grid_for_roster_review_r1", "optimizer_grid.py")
scan_manifest = _load("he_importance_scan_manifest_for_roster_review_r1", "scan_manifest.py")
train_config = _load("he_importance_train_config_for_roster_review_r1", "train_config.py")
runner = _load("he_importance_run_stage_q_for_roster_review_r1", "run_stage_q.py")


def _manifest_row(manifest: Mapping[str, Any], *, status: str, method: str) -> Mapping[str, Any]:
    methods = {cell["cell_id"]: cell["optimizer"]["method"] for cell in manifest["cells"]}
    return next(
        row for row in manifest["rows"] if row["status"] == status and methods[row["cell_id"]] == method
    )


def _args(manifest_path: Path, row_id: str, output_root: Path) -> Any:
    return runner.build_parser().parse_args(
        ["--manifest", str(manifest_path), "--row", row_id, "--output-root", str(output_root), "--dry-run"]
    )


def test_real_manifest_writer_loads_and_executes_one_admitted_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """T1: the real writer output is consumed by the real loader and dry-run path."""

    path = tmp_path / "written-v2.json"
    scan_manifest.write_manifest(path)
    manifest = runner.scan_manifest.load_manifest(path)
    row = _manifest_row(manifest, status="available", method="adam")
    calls: list[str] = []
    monkeypatch.setattr(runner.launch, "launch_train", lambda *_args, **_kwargs: calls.append("launch"))
    monkeypatch.setattr(runner.launch, "run_from_config", lambda *_args, **_kwargs: calls.append("run"))

    output_root = tmp_path / "dry-run"
    assert runner.execute(_args(path, row["row_id"], output_root)) == 0
    assert calls == []
    report = json.loads(capsys.readouterr().out)
    assert report["train_packet"]["content_hash"] == row["row_id"]
    assert report["resolved_config"]["run"]["dir"] == str(output_root / "O1" / row["row_id"])


def test_manifest_writer_detector_rejects_semantically_wrong_rehashed_output(tmp_path: Path) -> None:
    """T1 fault control: a wrong but self-consistent file must not be admitted."""

    path = tmp_path / "wrong-v2.json"
    scan_manifest.write_manifest(path)
    wrong = json.loads(path.read_text(encoding="utf-8"))
    wrong["rows"][0] = dict(wrong["rows"][0], seed_label=wrong["rows"][0]["seed_label"] + 1)
    wrong["content_hash"] = scan_manifest.content_hash(scan_manifest._body_without_hash(wrong))
    path.write_text(json.dumps(wrong), encoding="utf-8")
    row_id = wrong["rows"][0]["row_id"]

    with pytest.raises(runner.scan_manifest.ScanManifestError, match=r"manifest .* stale or tampered"):
        runner.execute(_args(path, row_id, tmp_path / "must-stay-empty"))
    assert not (tmp_path / "must-stay-empty").exists()


def test_v1_cli_refuses_before_materialization_and_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1: v1 is refused at the CLI boundary and an occupied legacy output is untouched."""

    source = scan_manifest.MANIFEST_V1_PATH
    path = tmp_path / "legacy-v1.json"
    path.write_bytes(source.read_bytes())
    manifest = runner.scan_manifest.load_manifest(path)
    row = _manifest_row(manifest, status="available", method="adam")
    output_root = tmp_path / "legacy-output"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("legacy output\n", encoding="utf-8")

    def fail_materialization(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("v1 reached materialization")

    monkeypatch.setattr(runner.stage_coordinate, "materialize_intended_configurations", fail_materialization)
    with pytest.raises(runner.scan_manifest.ScanManifestError, match=r"manifest v1 launch is refused.*v2"):
        runner.execute(_args(path, row["row_id"], output_root))
    assert sentinel.read_text(encoding="utf-8") == "legacy output\n"
    assert tuple(output_root.iterdir()) == (sentinel,)


def _path_hash_records(cells: tuple[Any, ...], root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(cell.output_path.relative_to(root)), cell.content_hash) for cell in cells))


def test_roster_status_reason_flip_preserves_every_o1_path_and_hash(tmp_path: Path) -> None:
    """T2: availability is sidecar state; full O1 path and hash identities stay fixed."""

    unavailable_roster = optimizer_grid.roster_availability()
    sr = unavailable_roster["sr"]
    assert sr.status == "unavailable"
    baseline = dict(unavailable_roster)
    flipped = dict(unavailable_roster)
    flipped["sr"] = stage_coordinate.OptimizerCell("sr", "available")
    root = tmp_path / "same-root"
    cells_a = stage_coordinate.materialize_intended_configurations(root, availability=baseline, stages=("O1",))
    cells_b = stage_coordinate.materialize_intended_configurations(root, availability=flipped, stages=("O1",))

    assert _path_hash_records(cells_a, root) == _path_hash_records(cells_b, root)
    assert all(cell.output_path == root / "O1" / cell.content_hash for cell in cells_a)
    assert all(cell.output_path == root / "O1" / cell.content_hash for cell in cells_b)
    assert any(cell.availability != other.availability for cell, other in zip(cells_a, cells_b))
    assert sr.reason


def test_roster_identity_detector_catches_parent_path_corruption(tmp_path: Path) -> None:
    """T2 fault control: preserving leaf/hash cannot hide a corrupted parent path."""

    availability = optimizer_grid.roster_availability()
    root = tmp_path / "root"
    cells = stage_coordinate.materialize_intended_configurations(root, availability=availability, stages=("O1",))
    corrupted = replace(cells[0], output_path=root / "CORRUPTED" / cells[0].content_hash)
    assert _path_hash_records(cells, root) != _path_hash_records((corrupted, *cells[1:]), root)


class _OriginalManifestGuard(dict[str, Any]):
    """Expose an authentic snapshot via items, but fail any original reread."""

    def items(self):  # type: ignore[no-untyped-def]
        return super().items()

    def get(self, key: str, default: Any = None) -> Any:
        if key == "scientific_identity":
            raise AssertionError("original manifest reread")
        return super().get(key, default)


def test_resolver_uses_owned_manifest_snapshot_for_optimizer_entry(tmp_path: Path) -> None:
    """T3: optimizer resolution must not consume a mutable original after snapshot."""

    availability = optimizer_grid.roster_availability()
    cells = stage_coordinate.materialize_intended_configurations(tmp_path / "clean", availability=availability, stages=("O1",))
    cell = next(cell for cell in cells if cell.availability["method"] == "adam" and cell.availability["status"] == "available")
    baseline = train_config.resolve_train_config(cell)

    guarded = _OriginalManifestGuard(stage_coordinate._thaw(cell.manifest))
    changed = replace(cell, manifest=guarded)
    try:
        resolved = train_config.resolve_train_config(changed)
    except AssertionError as error:
        pytest.fail(str(error))
    else:
        assert resolved == baseline
