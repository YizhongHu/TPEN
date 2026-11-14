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
                     "results_root": str(tmp_path), "experiment_name": "stage1",
                     "sector": "he", "run_id": f"run-{stream}",
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
    assert result["records"][0]["identity"]["physical_system_id"] == "he_atom"


def _install_multi_fake(monkeypatch, tmp_path: Path, *, physical_ids, registry_ids=None):
    document = _manifest(tmp_path)
    if registry_ids is not None:
        for row, system_id in zip(document["rows"], registry_ids):
            row["system_id"] = system_id
        document["manifest_hash"] = manifest_hash(document["rows"])
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    monkeypatch.setattr(collector, "inspect_run", lambda path: {
        "verdict": "completed", "resolved_config_identity": identity, "source_git_sha": "a" * 40})
    monkeypatch.setattr(collector, "_record_path", lambda path: path / ("m-" + path.name))

    def inspect(path, **kwargs):
        index = int(Path(path).name.rsplit("-", 1)[-1])
        return SimpleNamespace(record_schema="trajectory_records/v2", draw_count=2, walker_count=2,
                               row_count=4, finite_count=4, mean=2.5, variance=1.25,
                               csv_sha256="d" * 64, system_id=physical_ids[index],
                               execution_method_id="method", resolved_config_identity=identity,
                               model_provenance="checkpoint", checkpoint_sha256="c" * 64,
                               record_content_id="e" * 64)

    monkeypatch.setattr(collector, "inspect_trajectory_record", inspect)
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(collector, "_reference", lambda system_id, registry_path: (-2.9, "transcribed"))
    return document


def test_physical_system_id_must_be_consistent_within_registry_id(monkeypatch, tmp_path: Path) -> None:
    document = _install_multi_fake(monkeypatch, tmp_path, physical_ids=["physical-a", "physical-b", "physical-a", "physical-a"])
    with pytest.raises(collector.CollectionError, match="physical system"):
        collector.collect(document)


def test_physical_system_id_cannot_be_shared_by_registry_ids(monkeypatch, tmp_path: Path) -> None:
    document = _install_multi_fake(
        monkeypatch, tmp_path, physical_ids=["physical-a"] * 4,
        registry_ids=["he_atom", "h2_molecule", "he_atom", "he_atom"],
    )
    with pytest.raises(collector.CollectionError, match="physical system"):
        collector.collect(document)


def test_missing_physical_system_id_is_refused(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(collector.CollectionError, match="physical system"):
        collector.collect(_install_single_fake(monkeypatch, tmp_path, system_id=None))


def test_stream_aggregate_requires_the_complete_four_stream_set(
    monkeypatch, tmp_path: Path
) -> None:
    """A partial stream set must not be reported as a science aggregate."""
    document = _manifest(tmp_path, streams=3)
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    monkeypatch.setattr(
        collector,
        "inspect_run",
        lambda path: {
            "verdict": "completed",
            "resolved_config_identity": identity,
            "source_git_sha": "a" * 40,
        },
    )

    def inspect(path, **kwargs):
        index = int(Path(path).name.rsplit("-", 1)[-1])
        return SimpleNamespace(
            record_schema="trajectory_records/v2",
            draw_count=2,
            walker_count=2,
            row_count=4,
            finite_count=4,
            mean=2.5,
            variance=1.25,
            csv_sha256="d" * 64,
            system_id="he_atom",
            path=Path(f"m-{index}"),
            execution_method_id="method",
            resolved_config_identity=identity,
            model_provenance="checkpoint",
            checkpoint_sha256="c" * 64,
            record_content_id="e" * 64,
        )

    monkeypatch.setattr(collector, "_record_path", lambda path: path / ("m-" + path.name))
    monkeypatch.setattr(collector, "inspect_trajectory_record", inspect)
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(collector, "_reference", lambda system_id, registry_path: (-2.9, "transcribed"))

    with pytest.raises(collector.CollectionError, match="four streams"):
        collector.collect(document)


@pytest.mark.parametrize("stream_indices", [[0, 1, 2, 2], [0, 1, 2, 4]])
def test_stream_aggregate_rejects_duplicate_or_gapped_indices(
    monkeypatch, tmp_path: Path, stream_indices
) -> None:
    """Duplicate or gapped stream indices must not form an aggregate."""
    document = _manifest(tmp_path)
    rows = document["rows"]
    for row, stream in zip(rows, stream_indices):
        row["stream"] = stream
    document["manifest_hash"] = manifest_hash(rows)
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    monkeypatch.setattr(
        collector,
        "inspect_run",
        lambda path: {
            "verdict": "completed",
            "resolved_config_identity": identity,
            "source_git_sha": "a" * 40,
        },
    )
    monkeypatch.setattr(collector, "_record_path", lambda path: path / "m.json")
    monkeypatch.setattr(
        collector,
        "inspect_trajectory_record",
        lambda path, **kwargs: SimpleNamespace(
            record_schema="trajectory_records/v2",
            draw_count=2,
            walker_count=2,
            row_count=4,
            finite_count=4,
            mean=2.5,
            variance=1.25,
            csv_sha256="d" * 64,
            system_id="he_atom",
            execution_method_id="method",
            resolved_config_identity=identity,
            model_provenance="checkpoint",
            checkpoint_sha256="c" * 64,
            record_content_id="e" * 64,
        ),
    )
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(collector, "_reference", lambda system_id, registry_path: (-2.9, "transcribed"))

    with pytest.raises(collector.CollectionError, match="four streams"):
        collector.collect(document)


def test_duplicate_record_sidecars_are_refused(monkeypatch, tmp_path: Path) -> None:
    """A fixed-path lookup must not silently choose among duplicate records."""
    document = _manifest(tmp_path, streams=1)
    run_dir = tmp_path / "run-0"
    task_dir = run_dir / "mcmc_energy"
    task_dir.mkdir(parents=True)
    (task_dir / "sampled_eval_table.metadata.json").write_text("{}", encoding="utf-8")
    (task_dir / "sampled_eval_table.duplicate.metadata.json").write_text("{}", encoding="utf-8")
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    monkeypatch.setattr(
        collector,
        "inspect_run",
        lambda path: {
            "verdict": "completed",
            "resolved_config_identity": identity,
            "source_git_sha": "a" * 40,
        },
    )
    monkeypatch.setattr(
        collector,
        "inspect_trajectory_record",
        lambda path, **kwargs: SimpleNamespace(
            record_schema="trajectory_records/v2",
            draw_count=2,
            walker_count=2,
            row_count=4,
            finite_count=4,
            mean=2.5,
            variance=1.25,
            csv_sha256="d" * 64,
            system_id="he_atom",
            path=Path("values.csv"),
            execution_method_id="method",
            resolved_config_identity=identity,
            model_provenance="checkpoint",
            checkpoint_sha256="c" * 64,
            record_content_id="e" * 64,
        ),
    )
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(collector, "_reference", lambda system_id, registry_path: (-2.9, "transcribed"))

    with pytest.raises(collector.CollectionError, match="duplicate"):
        collector.collect(document)


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
    pytest.param({"finite_count": 3}, "non-finite", id="non-finite"),
])
def test_join_guards_refuse_invalid_record(monkeypatch, tmp_path: Path, changes, needle) -> None:
    with pytest.raises(collector.CollectionError, match=needle):
        collector.collect(_install_single_fake(monkeypatch, tmp_path, **changes))


@pytest.mark.parametrize("changes", [
    pytest.param({"mean": 2.6}, id="mean"),
    pytest.param({"variance": 1.5}, id="variance"),
])
def test_inspection_statistics_must_reconcile_with_grid(monkeypatch, tmp_path: Path, changes) -> None:
    with pytest.raises(collector.CollectionError, match="do not reconcile"):
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
    import subprocess
    expected_blobs = {
        "experiments/atomistic/h-v2/collect.py": "005bee8a144c7e0ae74338080ae53647a2c12078",
        "experiments/atomistic/he-v1/collect.py": "8c6f66907949c9a8b116290792753921a1eac222",
    }
    for path, expected in expected_blobs.items():
        actual = subprocess.check_output(["git", "hash-object", path], text=True).strip()
        assert actual == expected
