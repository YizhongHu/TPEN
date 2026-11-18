"""Hand-computable collector statistics and join guards."""

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.atomistic.stage1 import collect as collector
from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    Protocol,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    manifest_hash,
)


_TWO_PHASE_CONFIG_ROOT = Path(__file__).parents[1] / "configs"
_TWO_PHASE_SOURCE_SHA = "a" * 40
_TWO_PHASE_PHYSICAL_ID = "c" * 64


def _manifest(tmp_path: Path, streams: int = 4, draws: int = 2, walkers: int = 2) -> dict:
    rows = []
    for stream in range(streams):
        rows.append({"row_id": f"row-{stream}", "kind": "eval", "system_id": "he_atom",
                     "expected_physical_system_id": "he_atom",
                     "lineage": 0, "checkpoint": 10, "stream": stream,
                     "results_root": str(tmp_path), "experiment_name": "stage1",
                     "sector": "he", "run_id": f"run-{stream}",
                     "draws": draws, "walkers": walkers, "source_sha": "a" * 40,
                     "checkpoint_sha256": "c" * 64})
    return {"schema": "stage1/v3", "manifest_hash": manifest_hash(rows), "rows": rows}


def _install_single_fake(monkeypatch, tmp_path: Path, *, streams: int = 1, draws: int = 2,
                          walkers: int = 2, **changes):
    document = _manifest(tmp_path, streams=streams, draws=draws, walkers=walkers)
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    row_count = draws * walkers
    base = dict(record_schema="trajectory_records/v2", draw_count=draws, walker_count=walkers,
                row_count=row_count, finite_count=row_count, mean=2.5, variance=1.25, csv_sha256="d" * 64,
                system_id="he_atom", execution_method_id="method", resolved_config_identity=identity,
                model_provenance="checkpoint", checkpoint_sha256="c" * 64, record_content_id="e" * 64)
    base.update(changes)
    monkeypatch.setattr(collector, "inspect_eval_run", lambda path: {
        "verdict": "completed", "resolved_config_identity": identity, "source_git_sha": "a" * 40})
    monkeypatch.setattr(collector, "_record_path", lambda path: path / "m.json")
    monkeypatch.setattr(collector, "inspect_trajectory_record", lambda path, **kwargs: SimpleNamespace(**base))
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    return document


def test_happy_path_statistics_and_four_stream_aggregate(monkeypatch, tmp_path: Path) -> None:
    document = _manifest(tmp_path)
    values = [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]],
              [[3.0, 4.0], [5.0, 6.0]], [[4.0, 5.0], [6.0, 7.0]]]
    monkeypatch.setattr(collector, "inspect_eval_run", lambda path: {
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
    monkeypatch.setattr(collector, "inspect_eval_run", lambda path: {
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
        "inspect_eval_run",
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
        "inspect_eval_run",
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
    run_dir = collector.resolve_run_dir(document["rows"][0])
    task_dir = run_dir / "mcmc_energy"
    task_dir.mkdir(parents=True)
    (task_dir / "sampled_eval_table.metadata.json").write_text("{}", encoding="utf-8")
    (task_dir / "sampled_eval_table.duplicate.metadata.json").write_text("{}", encoding="utf-8")
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    monkeypatch.setattr(
        collector,
        "inspect_eval_run",
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

    with pytest.raises(collector.CollectionError, match="^duplicate trajectory metadata sidecars"):
        collector.collect(document)


@pytest.mark.parametrize("identity", [None, {"schema": "wrong", "sha256": "b" * 64}])
def test_missing_or_malformed_run_identity_is_refused(monkeypatch, tmp_path: Path, identity) -> None:
    document = _manifest(tmp_path, streams=1)
    monkeypatch.setattr(collector, "inspect_eval_run", lambda path: {"verdict": "completed", "resolved_config_identity": identity})
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


def test_writer_statistics_are_forwarded_without_reducing_again(
    monkeypatch, tmp_path: Path
) -> None:
    """A realistic grid may round differently from the writer's reduction tree."""
    document = _manifest(tmp_path, streams=1)
    values = [
        [0.75 + ((draw * 64 + walker) % 17 - 8) * 1.0e-5 for walker in range(64)]
        for draw in range(4)
    ]
    draw_sums = [sum(draw) for draw in values]
    draw_square_sums = [sum(value * value for value in draw) for draw in values]
    total = sum(draw_sums)
    total_square = sum(draw_square_sums)
    collector_mean = total / 256
    collector_variance = total_square / 256 - collector_mean * collector_mean
    writer_mean = math.nextafter(collector_mean, math.inf)
    writer_variance = math.nextafter(collector_variance, math.inf)
    document = _install_single_fake(
        monkeypatch,
        tmp_path,
        streams=4,
        draws=4,
        walkers=64,
        mean=writer_mean,
        variance=writer_variance,
    )
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: values)
    result = collector.collect(document)
    stats = result["records"][0]["statistics"]
    assert stats["mean"] == writer_mean
    assert stats["variance"] == writer_variance


def test_writer_statistics_exact_recompute_matches(monkeypatch, tmp_path: Path) -> None:
    """Control: an exactly recomputed sidecar scalar passes identically on the base collector.

    Mirrors ``collect.py``'s (pre-fix) base reduction loop exactly: a plain
    ``sum()`` over the full pooled list of 256 values is not bit-identical to
    draw-ordered partial sums accumulated with ``+=`` (CPython's ``sum()``
    uses a compensated summation algorithm since 3.12), so the per-draw
    partial sums must be accumulated the same way the collector does.
    """
    values = [
        [0.75 + ((draw * 64 + walker) % 17 - 8) * 1.0e-5 for walker in range(64)]
        for draw in range(4)
    ]
    draw_sums = {index: sum(draw) for index, draw in enumerate(values)}
    draw_square_sums = {index: sum(value * value for value in draw) for index, draw in enumerate(values)}
    total = 0.0
    total_square = 0.0
    for index in range(len(values)):
        total += draw_sums[index]
        total_square += draw_square_sums[index]
    collector_mean = total / 256
    collector_variance = total_square / 256 - collector_mean * collector_mean
    document = _install_single_fake(
        monkeypatch,
        tmp_path,
        streams=4,
        draws=4,
        walkers=64,
        mean=collector_mean,
        variance=collector_variance,
    )
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: values)
    result = collector.collect(document)
    stats = result["records"][0]["statistics"]
    assert stats["mean"] == collector_mean
    assert stats["variance"] == collector_variance


def test_tampered_csv_is_refused_after_inspection(monkeypatch, tmp_path: Path) -> None:
    """A post-inspection energy edit remains fail-closed at the reader boundary."""
    document = _manifest(tmp_path, streams=1)
    csv_path = tmp_path / "values.csv"
    csv_path.write_text(
        "draw_index,walker_index,local_energy\n"
        "0,0,1.0\n0,1,2.0\n1,0,3.0\n1,1,4.0\n",
        encoding="utf-8",
    )
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    record = SimpleNamespace(
        record_schema="trajectory_records/v2", draw_count=2, walker_count=2,
        row_count=4, finite_count=4, mean=2.5, variance=1.25,
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        path=csv_path, system_id="he_atom", execution_method_id="method",
        resolved_config_identity=identity, model_provenance="checkpoint",
        checkpoint_sha256="c" * 64, record_content_id="e" * 64,
    )
    monkeypatch.setattr(collector, "inspect_eval_run", lambda path: {
        "verdict": "completed", "resolved_config_identity": identity,
        "source_git_sha": "a" * 40,
    })
    monkeypatch.setattr(collector, "_record_path", lambda path: path / "m.json")

    def inspect_and_tamper(path, **kwargs):
        csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("1.0", "9.0"), encoding="utf-8")
        return record

    monkeypatch.setattr(collector, "inspect_trajectory_record", inspect_and_tamper)
    with pytest.raises(collector.CollectionError, match="changed after inspection"):
        collector.collect(document)


def test_missing_record_is_refused(monkeypatch, tmp_path: Path) -> None:
    document = _manifest(tmp_path, streams=1)
    monkeypatch.setattr(collector, "inspect_eval_run", lambda path: {
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


def _two_phase_document(tmp_path: Path) -> dict:
    """Build a real planner-shaped phase-1 + phase-2 document (no disk I/O)."""

    protocol = Protocol(smoke_updates=1, smoke_walkers=2, lineage_updates=1, lineage_walkers=2,
                        lineages=3, eval_streams=4, eval_walkers=2, eval_draws=2)
    config_sha256 = config_identities(config_root=_TWO_PHASE_CONFIG_ROOT, systems=("he",))
    phase1_rows = build_manifest(
        protocol=protocol, source_sha=_TWO_PHASE_SOURCE_SHA, config_sha256=config_sha256,
        checkpoints=[7], lineage_seeds=[11, 12, 13], checkpoint_sha256={},
        results_root=str(tmp_path), systems=("he",), phase=1,
    )
    phase1_hash = manifest_hash(phase1_rows)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"he:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {f"he:{lineage}:7": _TWO_PHASE_PHYSICAL_ID for lineage in range(protocol.lineages)},
    }
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows, checkpoints=[7], checkpoint_evidence=evidence,
        expected_physical_system_id={"he": _TWO_PHASE_PHYSICAL_ID}, source_sha=_TWO_PHASE_SOURCE_SHA,
        config_sha256=config_sha256, protocol=protocol, phase1_manifest_hash=phase1_hash,
    )
    rows = list(phase1_rows) + eval_rows
    return {"schema": "stage1/v3", "manifest_hash": manifest_hash(rows), "rows": rows}


def _install_two_phase_fakes(monkeypatch) -> None:
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    monkeypatch.setattr(collector, "inspect_eval_run", lambda path: {
        "verdict": "completed", "resolved_config_identity": identity,
        "source_git_sha": _TWO_PHASE_SOURCE_SHA})
    monkeypatch.setattr(collector, "_record_path", lambda path: path / "m.json")
    monkeypatch.setattr(collector, "inspect_trajectory_record", lambda path, **kwargs: SimpleNamespace(
        record_schema="trajectory_records/v2", draw_count=2, walker_count=2, row_count=4,
        finite_count=4, mean=2.5, variance=1.25, csv_sha256="d" * 64,
        system_id=_TWO_PHASE_PHYSICAL_ID, execution_method_id="method",
        resolved_config_identity=identity, model_provenance="checkpoint",
        checkpoint_sha256=_TWO_PHASE_PHYSICAL_ID, record_content_id="e" * 64))
    monkeypatch.setattr(collector, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(collector, "_reference", lambda system_id, registry_path: (-2.9, "transcribed"))


def test_computed_two_phase_producer_output_passes_collection(monkeypatch, tmp_path: Path) -> None:
    """A real planner phase-1 + phase-2 manifest, with computed config identity, collects."""

    document = _two_phase_document(tmp_path)
    _install_two_phase_fakes(monkeypatch)
    result = collector.collect(document)
    assert len(result["records"]) == 12
    assert len(result["aggregates"]) == 3


def test_two_phase_collection_refuses_a_flipped_phase1_train_row_hash(monkeypatch, tmp_path: Path) -> None:
    document = _two_phase_document(tmp_path)
    _install_two_phase_fakes(monkeypatch)
    rows = document["rows"]
    eval_row = next(row for row in rows if row["kind"] == "eval")
    eval_row["phase1_train_row_hash"] = "f" * 64
    document["manifest_hash"] = manifest_hash(rows)
    with pytest.raises(collector.CollectionError, match="not bound to its phase-1 train row"):
        collector.collect(document)


def test_two_phase_collection_refuses_a_changed_config_sha256(monkeypatch, tmp_path: Path) -> None:
    document = _two_phase_document(tmp_path)
    _install_two_phase_fakes(monkeypatch)
    rows = document["rows"]
    eval_row = next(row for row in rows if row["kind"] == "eval")
    eval_row["config_sha256"] = "d" * 64
    document["manifest_hash"] = manifest_hash(rows)
    with pytest.raises(collector.CollectionError, match="config identity mismatch"):
        collector.collect(document)


def test_preserved_legacy_collectors_are_unchanged() -> None:
    import subprocess
    expected_blobs = {
        "experiments/atomistic/h-v2/collect.py": "005bee8a144c7e0ae74338080ae53647a2c12078",
        "experiments/atomistic/he-v1/collect.py": "8c6f66907949c9a8b116290792753921a1eac222",
    }
    for path, expected in expected_blobs.items():
        actual = subprocess.check_output(["git", "hash-object", path], text=True).strip()
        assert actual == expected
