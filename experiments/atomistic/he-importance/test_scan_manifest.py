"""Tests for the frozen O1 scan manifest and manifest-backed selection."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import signal
import sys
import time
from copy import deepcopy

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("hi_mf1_scan_manifest_test", HERE / "scan_manifest.py")
assert spec is not None and spec.loader is not None
manifest_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = manifest_module
spec.loader.exec_module(manifest_module)

_PER_TEST_TIMEOUT_SECONDS = 120


@contextmanager
def _timeout_guard(seconds: int, nodeid: str):
    """Bound one test without adding a pytest-timeout dependency."""

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_alarm = signal.alarm(0)

    def raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"per-test timeout ({seconds}s): {nodeid}")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_alarm:
            signal.alarm(previous_alarm)


@pytest.fixture(autouse=True)
def _per_test_timeout(request: pytest.FixtureRequest):
    with _timeout_guard(_PER_TEST_TIMEOUT_SECONDS, request.node.nodeid):
        yield


def test_per_test_timeout_fires() -> None:
    with pytest.raises(TimeoutError, match="test_per_test_timeout_fires"):
        with _timeout_guard(1, "test_per_test_timeout_fires"):
            time.sleep(2)


def test_committed_manifest_regenerates_byte_identically() -> None:
    path = HERE / "o1_scan_manifest_v1.json"
    assert path.stat().st_size < 1_000_000
    committed = path.read_bytes()
    rendered = manifest_module.render_manifest_text().encode("utf-8")
    committed_hash = hashlib.sha256(committed).hexdigest()
    rendered_hash = hashlib.sha256(rendered).hexdigest()
    if committed_hash != rendered_hash or len(committed) != len(rendered):
        first_difference = next(
            (index for index, (left, right) in enumerate(zip(committed, rendered)) if left != right),
            min(len(committed), len(rendered)),
        )
        window_start = max(0, first_difference - 24)
        window_end = first_difference + 24
        raise AssertionError(
            "manifest bytes differ: "
            f"committed_sha256={committed_hash} rendered_sha256={rendered_hash} "
            f"committed_length={len(committed)} rendered_length={len(rendered)} "
            f"first_difference={first_difference} "
            f"committed_window={committed[window_start:window_end]!r} "
            f"rendered_window={rendered[window_start:window_end]!r}"
        )


def test_manifest_counts_sources_and_admission_reason() -> None:
    manifest = manifest_module.load_manifest()
    assert manifest["counts"] == {"rows": 1920, "admitted": 128, "unavailable": 1792}
    assert manifest["horizon"] == 50000
    assert manifest["walkers"] == 4096
    assert manifest["checkpoint_cadence"] == {
        "operational": True, "recorded_per_run": True, "in_identity": False
    }
    assert manifest["provenance"]["design_document"]["sha256"] == manifest_module.DESIGN_DOC_SHA256
    assert manifest["provenance"]["design_document"]["path"] == (
        "2026-09-04-tpen-helium-importance-scan-consolidated.md"
    )
    assert manifest["provenance"]["design_document"]["tracked"] is False
    assert manifest["provenance"]["design_document"]["location"] == (
        "operator checkout (untracked; commit decision pending with operator)"
    )
    assert manifest["provenance"]["lane_plan_approval"]["citation"] == (
        "TO note operator-approved-materialization-2026-09-05 on 67987933"
    )
    assert len(manifest["cells"]) == 120
    assert len(manifest["contexts"]) == 2
    rendered = manifest_module.render_manifest()
    assert rendered["counts"] == manifest["counts"]
    assert [row["status"] for row in rendered["rows"]] == [row["status"] for row in manifest["rows"]]
    sources = manifest["provenance"]["coordinate_sources"]
    assert set(sources) >= {"stage", "optimizer", "context", "seed", "horizon", "walkers", "admission"}
    assert {sources[key]["evidence_class"] if isinstance(sources[key]["evidence_class"], str) else tuple(sources[key]["evidence_class"]) for key in sources} >= {"RELAYED"}
    aliases = manifest["provenance"]["source_aliases"]
    assert set(aliases.values()) >= {"stage", "optimizer", "context", "seed", "horizon", "walkers", "admission"}
    for row in manifest["rows"]:
        assert set(aliases[ref] for ref in row["approval_sources"].split(",")) == set(aliases.values()) - {"cadence"}
    for cell in manifest["cells"]:
        assert set(cell["approval_sources"]) == {"optimizer", "admission", "horizon", "walkers"}
        assert cell["horizon"] == manifest["horizon"]
        assert cell["walkers"] == manifest["walkers"]
    for context in manifest["contexts"]:
        assert context["approval_sources"] == ["context"]
    roster = manifest_module._materialized_rows()
    for row in manifest["rows"]:
        if row["status"] == "unavailable":
            assert row["unavailable_reason"] == next(
                candidate["unavailable_reason"]
                for candidate in roster["rows"]
                if candidate["row_id"] == row["row_id"]
            )


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


def _runner():
    import importlib

    return importlib.import_module("experiments.atomistic.he-importance.run_stage_q")


@pytest.fixture(autouse=True)
def _manifest_tests_never_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep manifest selection tests behind an explicit no-training firewall."""

    runner = _runner()

    def fail_launch(*_args, **_kwargs):
        raise AssertionError("manifest test crossed the launch boundary")

    monkeypatch.setattr(runner.launch, "launch_train", fail_launch)
    monkeypatch.setattr(runner.launch, "run_from_config", fail_launch)


def test_manifest_launch_firewall_is_active() -> None:
    runner = _runner()
    with pytest.raises(AssertionError, match="launch boundary"):
        runner.launch.launch_train(None, None)


def _row(manifest: dict, *, status: str, method: str) -> dict:
    methods = {cell["cell_id"]: cell["optimizer"]["method"] for cell in manifest["cells"]}
    return next(row for row in manifest["rows"] if row["status"] == status and methods[row["cell_id"]] == method)


def _args(runner, manifest_path: Path, row_id: str, output_root: Path, *extra: str):
    return runner.build_parser().parse_args(
        ["--manifest", str(manifest_path), "--row", row_id, "--output-root", str(output_root), "--dry-run", *extra]
    )


@pytest.mark.parametrize("method", ["sr", "kfac"])
def test_unavailable_rows_refuse_before_output_or_model(
    tmp_path: Path, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    manifest = manifest_module.load_manifest()
    row = _row(manifest, status="unavailable", method=method)
    path = tmp_path / f"{method}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner.train_config, "resolve_train_config", lambda *_args, **_kwargs: pytest.fail("model/config constructed"))
    with pytest.raises(runner.scan_manifest.ScanManifestError, match=row["row_id"]) as excinfo:
        runner.execute(_args(runner, path, row["row_id"], tmp_path / method))
    assert not (tmp_path / method).exists()
    assert row["unavailable_reason"] in str(excinfo.value)


def test_manifest_tamper_stale_and_unknown_refuse_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    manifest = manifest_module.load_manifest()
    admitted = _row(manifest, status="available", method="adam")
    tampered = deepcopy(manifest)
    tampered["rows"][0]["seed_label"] += 1
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.scan_manifest.ScanManifestError, match="content hash mismatch"):
        runner.execute(_args(runner, tampered_path, tampered["rows"][0]["row_id"], tmp_path / "tampered-out"))
    assert not (tmp_path / "tampered-out").exists()
    with pytest.raises(runner.scan_manifest.ScanManifestError, match="unknown manifest row"):
        runner.execute(_args(runner, HERE / "o1_scan_manifest_v1.json", "missing-row", tmp_path / "unknown-out"))
    assert not (tmp_path / "unknown-out").exists()
    original = runner.scan_manifest._materialized_rows()
    drifted = deepcopy(original)
    drifted["rows"] = list(drifted["rows"])
    drifted_row = next(candidate for candidate in drifted["rows"] if candidate["row_id"] == admitted["row_id"])
    drifted_row["seed_label"] += 1
    monkeypatch.setattr(runner.scan_manifest, "_materialized_rows", lambda: drifted)
    with pytest.raises(runner.scan_manifest.ScanManifestError, match="stale or tampered"):
        runner.execute(_args(runner, HERE / "o1_scan_manifest_v1.json", admitted["row_id"], tmp_path / "stale-out"))
    assert not (tmp_path / "stale-out").exists()


def test_manifest_positive_consumes_same_runner_limbs_and_horizon(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _runner()
    manifest = manifest_module.load_manifest()
    row = _row(manifest, status="available", method="adam")
    path = HERE / "o1_scan_manifest_v1.json"
    manifest_report = tmp_path / "manifest-report"
    assert runner.execute(_args(runner, path, row["row_id"], manifest_report)) == 0
    manifest_report_json = json.loads(capsys.readouterr().out)
    cell = next(cell for cell in manifest["cells"] if cell["cell_id"] == row["cell_id"])
    levels = cell["optimizer"]["levels"]
    flag_args = runner.build_parser().parse_args([
        "--stage", "O1", "--context", row["context"], "--optimizer", "adam",
        "--lr", str(levels["lr"]), "--beta2", str(levels["beta2"]),
        "--seed-label", str(row["seed_label"]), "--walkers", "4096",
        "--output-root", str(tmp_path / "flag-report"), "--dry-run",
    ])
    assert runner.execute(flag_args) == 0
    flag_report_json = json.loads(capsys.readouterr().out)
    assert manifest_report_json["walkers"] == 4096
    assert manifest_report_json["resolved_config"]["runner"]["trainer"]["max_steps"] == 50000
    assert manifest_report_json["resolved_config"]["runner"]["sampler"] == flag_report_json["resolved_config"]["runner"]["sampler"]
    assert manifest_report_json["resolved_config"]["runner"]["trainer"] == flag_report_json["resolved_config"]["runner"]["trainer"]


def test_every_admitted_manifest_row_consumes_materialized_w4096(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every admitted row binds its materialized identity and consumed sampler to W=4096."""

    runner = _runner()
    manifest = manifest_module.load_manifest()
    cells = runner.stage_coordinate.materialize_intended_configurations(
        tmp_path / "cells",
        availability=runner.optimizer_grid.roster_availability(),
        stages=("O1",),
    )
    cells_by_row_id = {cell.content_hash: cell for cell in cells}
    live_rows = manifest_module._materialized_rows()
    monkeypatch.setattr(runner.scan_manifest, "_materialized_rows", lambda: live_rows)
    monkeypatch.setattr(
        runner.stage_coordinate,
        "materialize_intended_configurations",
        lambda *_args, **_kwargs: cells,
    )

    admitted_rows = [row for row in manifest["rows"] if row["status"] == "available"]
    assert len(admitted_rows) == 128
    for row in admitted_rows:
        cell = cells_by_row_id[row["row_id"]]
        cell_record = next(candidate for candidate in manifest["cells"] if candidate["cell_id"] == row["cell_id"])
        assert runner.stage_coordinate.walker_count(cell.manifest) == cell_record["walkers"] == 4096
        assert runner.execute(
            _args(runner, HERE / "o1_scan_manifest_v1.json", row["row_id"], tmp_path / row["row_id"])
        ) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["resolved_config"]["sampler"]["n_walkers"] == 4096
        assert report["resolved_config"]["runner"]["sampler"]["n_walkers"] == 4096


@pytest.mark.parametrize("flag", [
    "--stage", "--context", "--optimizer", "--lr", "--beta2", "--seed-label",
    "--walkers", "--literal-configuration",
])
def test_manifest_rejects_explicit_flag_selection(tmp_path: Path, flag: str) -> None:
    runner = _runner()
    manifest = manifest_module.load_manifest()
    row = _row(manifest, status="available", method="adam")
    value = "{}" if flag == "--literal-configuration" else {
        "--stage": "O1", "--context": row["context"], "--optimizer": "adam",
        "--lr": "0.005", "--beta2": "0.999", "--seed-label": str(row["seed_label"]),
        "--walkers": "4096",
    }[flag]
    args = _args(runner, HERE / "o1_scan_manifest_v1.json", row["row_id"], tmp_path / "out", flag, value)
    with pytest.raises(runner.StageQSelectionError, match="mutually exclusive"):
        runner.execute(args)
    assert not (tmp_path / "out").exists()


def test_cadence_is_operational_not_identity() -> None:
    manifest = manifest_module.load_manifest()
    identity = [(row["row_id"], row["cell_id"], row["context"], row["seed_label"], row["status"]) for row in manifest["rows"]]
    altered = deepcopy(manifest)
    altered["checkpoint_cadence"] = manifest_module.checkpoint_cadence_declaration(updates=777)
    assert identity == [(row["row_id"], row["cell_id"], row["context"], row["seed_label"], row["status"]) for row in altered["rows"]]
    assert manifest_module.checkpoint_cadence_declaration(updates=1000) == manifest_module.checkpoint_cadence_declaration(updates=777)
    assert manifest_module.content_hash(manifest_module._body_without_hash(manifest)) == manifest["content_hash"]


def test_manifest_cadence_value_cannot_enter_rendered_identity() -> None:
    baseline = manifest_module.render_manifest_text()
    for cadence in (1, 777, 50_000):
        assert manifest_module.checkpoint_cadence_declaration(updates=cadence) == {
            "operational": True, "recorded_per_run": True, "in_identity": False
        }
    assert baseline == manifest_module.render_manifest_text()

