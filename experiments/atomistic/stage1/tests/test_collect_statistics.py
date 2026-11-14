"""Hand-computable collector statistics and join guards."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.atomistic.stage1 import collect as collector
from experiments.atomistic.stage1.protocol import manifest_hash


def _manifest(tmp_path: Path, streams: int = 4) -> dict:
    rows = []
    for stream in range(streams):
        rows.append({"row_id": f"row-{stream}", "kind": "eval", "system_id": "he_atom",
                     "lineage": 0, "checkpoint": 10, "stream": stream,
                     "results_root": str(tmp_path), "run_id": f"run-{stream}",
                     "draws": 2, "walkers": 2, "source_sha": "a" * 40,
                     "checkpoint_sha256": "c" * 64})
    return {"schema": "stage1/v2", "manifest_hash": manifest_hash(rows), "rows": rows}


def _install_single_fake(monkeypatch, tmp_path: Path, **changes):
    document = _manifest(tmp_path, streams=1)
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    base = dict(record_schema="trajectory_records/v2", draw_count=2, walker_count=2, row_count=4,
                finite_count=4, mean=2.5, variance=1.25, csv_sha256="d" * 64,
                system_id="he_atom", execution_method_id="method", resolved_config_identity=identity,
                model_provenance="checkpoint", checkpoint_sha256="c" * 64, record_content_id="e" * 64)
    base.update(changes)
    monkeypatch.setattr(collector, "inspect_run", lambda path: {
        "verdict": "completed", "resolved_config_identity": identity, "source_git_sha": "a" * 40})
    monkeypatch.setattr(collector, "_record_path", lambda path: path / "m.json")
    monkeypatch.setattr(collector, "inspect_trajectory_record", lambda path, **kwargs: SimpleNamespace(**base))
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    return document


def test_happy_path_statistics_and_four_stream_aggregate(monkeypatch, tmp_path: Path) -> None:
    document = _manifest(tmp_path)
    values = [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]],
              [[3.0, 4.0], [5.0, 6.0]], [[4.0, 5.0], [6.0, 7.0]]]
    monkeypatch.setattr(collector, "inspect_run", lambda path: {
        "verdict": "completed", "resolved_config_identity": {"schema": "tpen-config-identity-v1", "sha256": "b" * 64},
        "source_git_sha": "a" * 40})
    def inspect(path, **kwargs):
        index = int(Path(path).name.rsplit("-", 1)[-1])
        return SimpleNamespace(record_schema="trajectory_records/v2", draw_count=2, walker_count=2,
                               row_count=4, finite_count=4, mean=sum(sum(x) for x in values[index]) / 4,
                               variance=1.25, csv_sha256="d" * 64, system_id="he_atom",
                               path=Path(f"m-{index}"),
                               execution_method_id="method", resolved_config_identity={"schema": "tpen-config-identity-v1", "sha256": "b" * 64},
                               model_provenance="checkpoint", checkpoint_sha256="c" * 64,
                               record_content_id="e" * 64)
    monkeypatch.setattr(collector, "_record_path", lambda path: path / ("m-" + path.name))
    monkeypatch.setattr(collector, "inspect_trajectory_record", inspect)
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: values[int(record.path.name.rsplit("-", 1)[-1])])
    monkeypatch.setattr(collector, "_reference", lambda system_id, registry_path: (-2.9, "transcribed"))
    result = collector.collect(document)
    stats = result["records"][0]["statistics"]
    assert stats["mean"] == 2.5
    assert stats["variance"] == 1.25
    assert stats["draw_mean_variance"] == 1.0
    assert stats["draw_axis_uncertainty"] == 2 ** -0.5
    assert result["aggregates"][0]["mean_of_stream_means"] == 4.0
    assert result["aggregates"][0]["spread_of_stream_means"] == 1.25 ** 0.5
    assert result["records"][0]["reference"]["error"] == pytest.approx(5.4)


@pytest.mark.parametrize("identity", [None, {"schema": "wrong", "sha256": "b" * 64}])
def test_missing_or_malformed_run_identity_is_refused(monkeypatch, tmp_path: Path, identity) -> None:
    document = _manifest(tmp_path, streams=1)
    monkeypatch.setattr(collector, "inspect_run", lambda path: {"verdict": "completed", "resolved_config_identity": identity})
    with pytest.raises(collector.CollectionError, match="identity"):
        collector.collect(document)


@pytest.mark.parametrize("changes,needle", [
    pytest.param({"model_provenance": "constructed"}, "provenance", id="provenance"),
    pytest.param({"resolved_config_identity": {"schema": "tpen-config-identity-v1", "sha256": "z" * 64}}, "identity", id="identity"),
    pytest.param({"draw_count": 3}, "shape", id="draws"), pytest.param({"walker_count": 3}, "shape", id="walkers"),
    pytest.param({"system_id": "other"}, "system", id="system"), pytest.param({"finite_count": 3}, "non-finite", id="non-finite"),
])
def test_join_guards_refuse_invalid_record(monkeypatch, tmp_path: Path, changes, needle) -> None:
    with pytest.raises(collector.CollectionError, match=needle):
        collector.collect(_install_single_fake(monkeypatch, tmp_path, **changes))


def test_missing_record_is_refused(monkeypatch, tmp_path: Path) -> None:
    document = _manifest(tmp_path, streams=1)
    monkeypatch.setattr(collector, "inspect_run", lambda path: {
        "verdict": "completed", "resolved_config_identity": {"schema": "tpen-config-identity-v1", "sha256": "b" * 64},
        "source_git_sha": "a" * 40})
    with pytest.raises(collector.CollectionError, match="missing trajectory"):
        collector.collect(document)


def test_unknown_and_unsourced_references_are_refused(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="unknown"):
        collector._reference("not-a-system", None)

    registry = tmp_path / "systems.yaml"
    registry.write_text("systems:\n  - id: x\n    confidence: unverified\n", encoding="utf-8")
    with pytest.raises(Exception, match="missing"):
        collector._reference("x", registry)


def test_preserved_legacy_collectors_are_unchanged() -> None:
    import hashlib
    import subprocess
    for path in ("experiments/atomistic/h-v2/collect.py", "experiments/atomistic/he-v1/collect.py"):
        expected = subprocess.check_output(["git", "rev-parse", f"f0704811:{path}"], text=True).strip()
        actual = hashlib.sha1(Path(path).read_bytes()).hexdigest()
        assert expected  # the immutable base blob is present in the exact-head clone
        assert actual == hashlib.sha1(subprocess.check_output(["git", "show", expected])).hexdigest()
