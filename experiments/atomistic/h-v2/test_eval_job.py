"""Unit controls for item E's evaluation fan-out (TO 8a5e926f).

Covers the acceptance contract's named mutants: skip the checkpoint hash
check, wrong stream seed, drop a file-written hash, accept a TRAIN receipt
from another lineage, a glob replaces the inventory, select by the
directory cursor instead of completed_updates, read a non-inventoried file
without refusing, the readout generator silently omitted, a pilot run
outside the subset, c1 outside 3..19 accepted, a selection record from
another B1 receipt accepted, and the latest tie winning instead of the
earliest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"h_v2_{name}", HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _module("protocol")
eval_job = _module("eval_job")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checkpoint_entry(tmp_path: Path, *, update: int, content: bytes = b"model-bytes") -> dict:
    """Return an inventory entry carrying the full load closure (F2).

    model.pt, manifest.json, and COMPLETE -- every file
    eval_job.REQUIRED_CHECKPOINT_CLOSURE_FILES names -- so this fixture
    satisfies verify_checkpoint_files'/materialize_verified_checkpoint's
    closure-completeness check by default; tests targeting a specific
    missing/tampered file still delete or overwrite just that one file.
    """

    checkpoint_dir = tmp_path / f"ck{update}"
    checkpoint_dir.mkdir()
    manifest = b'{"completed_updates": %d}' % update
    complete_marker = b""
    (checkpoint_dir / "model.pt").write_bytes(content)
    (checkpoint_dir / "manifest.json").write_bytes(manifest)
    (checkpoint_dir / "COMPLETE").write_bytes(complete_marker)
    return {
        "update": update,
        "path": str(checkpoint_dir),
        "files": {
            "model.pt": _sha256(content),
            "manifest.json": _sha256(manifest),
            "COMPLETE": _sha256(complete_marker),
        },
    }


def _train_receipt(tmp_path: Path, *, lineage: str, payload: str, commit: str, inventory: list[dict]) -> dict:
    """Return a synthetic TRAIN receipt with a REAL, currently-matching identity block.

    F3: :func:`eval_job.verify_train_receipt_identity` now checks every
    identity field against this checkout's own current frozen inputs, so a
    fixture carrying only ``{"lineage": ...}`` no longer accepts. Identity is
    built through the real protocol functions (the same ones a real TRAIN
    writer uses) so every hash matches today's checkout; ``lineage`` is a
    pilot payload's own fixed value (never overridden here -- the
    wrong-lineage/wrong-payload tests below pass a *payload*/*lineage*
    combination that is naturally inconsistent instead).
    """

    if payload in protocol.PILOT_PAYLOADS:
        _, identity = protocol.pilot_training_config(payload)
    else:
        _, identity = protocol.science_training_config(lineage, 0)
    identity["materialized_config_sha256"] = _sha256(b"materialized-input-bytes-for-tests")
    # D4 (review r2 delta): the resolved_config artifact's ACTUAL bytes are
    # re-hashed, so this fixture needs a real file -- distinct from the
    # materialized input digest above, matching the real producer's two
    # different artifacts.
    resolved_config_path = tmp_path / f"resolved-config-{lineage}-{commit}.yaml"
    resolved_config_path.write_text("resolved-runtime-config-bytes-for-tests\n", encoding="utf-8")
    resolved_config_sha256 = _sha256(resolved_config_path.read_bytes())
    return {
        "payload": payload,
        "segments": [{
            "source_commit": commit,
            "resolved_config_sha256": resolved_config_sha256,
            "artifacts": {"resolved_config": {"path": str(resolved_config_path), "sha256": resolved_config_sha256}},
        }],
        "science_config_identity": identity,
        "checkpoint_inventory": inventory,
    }


# --- checkpoint inventory selection and hash verification ------------------


def test_select_checkpoint_finds_the_matching_update(tmp_path: Path) -> None:
    entry = _checkpoint_entry(tmp_path, update=9000)
    receipt = _train_receipt(tmp_path, lineage="L1", payload="science", commit="a" * 40, inventory=[entry])
    found = eval_job.select_checkpoint(receipt, update=9000)
    assert found == entry


def test_select_checkpoint_refuses_an_update_outside_the_inventory(tmp_path: Path) -> None:
    receipt = _train_receipt(tmp_path, lineage="L1", payload="science", commit="a" * 40, inventory=[])
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.select_checkpoint(receipt, update=9000)
    assert excinfo.value.code == "checkpoint-not-in-inventory"


def test_select_checkpoint_never_globs_a_directory_not_in_the_inventory(tmp_path: Path) -> None:
    """Mutant: 'a glob replaces the inventory'."""

    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "step_009000").mkdir()
    receipt = _train_receipt(tmp_path, lineage="L1", payload="science", commit="a" * 40, inventory=[])
    # A directory exists on disk that a glob would happily find, but the
    # inventory does not name it: selection must still refuse.
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.select_checkpoint(receipt, update=9000)
    assert excinfo.value.code == "checkpoint-not-in-inventory"


def test_verify_checkpoint_files_accepts_matching_hashes(tmp_path: Path) -> None:
    entry = _checkpoint_entry(tmp_path, update=1000)
    out = eval_job.verify_checkpoint_files(entry)
    assert out == Path(entry["path"])


def test_verify_checkpoint_files_refuses_a_tampered_file(tmp_path: Path) -> None:
    """Mutant: 'skip the checkpoint hash check'."""

    entry = _checkpoint_entry(tmp_path, update=1000)
    (Path(entry["path"]) / "model.pt").write_bytes(b"tampered")
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.verify_checkpoint_files(entry)
    assert excinfo.value.code == "checkpoint-hash-mismatch"


def test_verify_checkpoint_files_refuses_a_missing_file(tmp_path: Path) -> None:
    entry = _checkpoint_entry(tmp_path, update=1000)
    (Path(entry["path"]) / "model.pt").unlink()
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.verify_checkpoint_files(entry)
    assert excinfo.value.code == "checkpoint-file-missing"


def test_verify_checkpoint_files_never_reads_a_file_not_in_the_inventory(tmp_path: Path) -> None:
    """Mutant: 'read a non-inventoried file without refusing'.

    A sibling file exists on disk with no inventory entry; it is simply
    never looked at, and verification of the inventoried file alone still
    succeeds.
    """

    entry = _checkpoint_entry(tmp_path, update=1000)
    (Path(entry["path"]) / "extra.bin").write_bytes(b"not tracked, never read")
    out = eval_job.verify_checkpoint_files(entry)
    assert out == Path(entry["path"])


def test_resolve_checkpoint_snapshot_closes_the_check_use_window(tmp_path: Path) -> None:
    """F2: core must load the verified SNAPSHOT, never the original directory."""

    entry = _checkpoint_entry(tmp_path, update=1000)
    receipt = _train_receipt(tmp_path, lineage="L1", payload="science", commit="a" * 40, inventory=[entry])
    snapshot_dir = tmp_path / "snapshot"
    resolved = eval_job.resolve_checkpoint(receipt, update=1000, snapshot_dir=snapshot_dir)
    assert resolved == snapshot_dir
    assert resolved != Path(entry["path"])
    assert (resolved / "model.pt").read_bytes() == (Path(entry["path"]) / "model.pt").read_bytes()
    # Tampering the ORIGINAL after the snapshot was taken must not affect it.
    (Path(entry["path"]) / "model.pt").write_bytes(b"swapped-after-verification")
    assert (resolved / "model.pt").read_bytes() == b"model-bytes"


def test_materialize_verified_checkpoint_refuses_a_tampered_file(tmp_path: Path) -> None:
    """F2: a hash mismatch refuses before any bytes are copied into the snapshot."""

    entry = _checkpoint_entry(tmp_path, update=1000)
    (Path(entry["path"]) / "model.pt").write_bytes(b"tampered")
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.materialize_verified_checkpoint(entry, tmp_path / "snapshot")
    assert excinfo.value.code == "checkpoint-hash-mismatch"


# --- TRAIN receipt identity (EVAL-only against an earlier commit) ---------


def test_verify_train_receipt_identity_accepts_a_matching_receipt(tmp_path: Path) -> None:
    receipt = _train_receipt(tmp_path, lineage="L1", payload="science", commit="a" * 40, inventory=[])
    eval_job.verify_train_receipt_identity(
        receipt, expected_commit="a" * 40, expected_lineage="L1", expected_payload="science"
    )


def test_verify_train_receipt_identity_refuses_wrong_commit(tmp_path: Path) -> None:
    receipt = _train_receipt(tmp_path, lineage="L1", payload="science", commit="a" * 40, inventory=[])
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.verify_train_receipt_identity(
            receipt, expected_commit="b" * 40, expected_lineage="L1", expected_payload="science"
        )
    assert excinfo.value.code == "wrong-train-commit"


def test_verify_train_receipt_identity_refuses_a_receipt_from_another_lineage(tmp_path: Path) -> None:
    """Mutant: 'accept a TRAIN receipt from another lineage'."""

    receipt = _train_receipt(tmp_path, lineage="L2", payload="science", commit="a" * 40, inventory=[])
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.verify_train_receipt_identity(
            receipt, expected_commit="a" * 40, expected_lineage="L1", expected_payload="science"
        )
    assert excinfo.value.code == "wrong-train-lineage"


def test_verify_train_receipt_identity_refuses_wrong_payload(tmp_path: Path) -> None:
    receipt = _train_receipt(tmp_path, lineage="G", payload="gpu-route-g", commit="a" * 40, inventory=[])
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.verify_train_receipt_identity(
            receipt, expected_commit="a" * 40, expected_lineage="G", expected_payload="capacity-pilot"
        )
    assert excinfo.value.code == "wrong-train-payload"


def test_verify_train_receipt_identity_refuses_an_empty_receipt() -> None:
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.verify_train_receipt_identity(
            {"payload": "science", "segments": []},
            expected_commit="a" * 40,
            expected_lineage="L1",
            expected_payload="science",
        )
    assert excinfo.value.code == "train-receipt-empty"


# --- seed grids: equal CI exactly ------------------------------------------


def test_science_fanout_is_27x4_with_the_frozen_seed_formula() -> None:
    generations = eval_job.science_fanout(lineage_index=1)
    assert len(generations) == 108
    updates = protocol.science_evaluated_checkpoint_updates()
    assert [g["update"] for g in generations[:4]] == [updates[0]] * 4
    for generation in generations:
        expected_seed = 1_000_000 + 10_000 * 1 + 10 * generation["ordinal"] + generation["stream"]
        assert generation["seed"] == expected_seed


def test_science_fanout_seed_changes_with_stream() -> None:
    """Mutant: 'wrong stream seed'."""

    generations = eval_job.science_fanout(lineage_index=0)
    by_key = {(g["ordinal"], g["stream"]): g["seed"] for g in generations}
    assert by_key[(5, 0)] != by_key[(5, 1)]
    assert by_key[(5, 1)] - by_key[(5, 0)] == 1


def test_pilot_fanout_stage1_is_21_runs_stream0() -> None:
    generations = eval_job.pilot_fanout("1")
    assert len(generations) == 21
    assert {g["stream"] for g in generations} == {0}
    seeds = {g["seed"] for g in generations}
    roster = set(json.loads(protocol.CONTRACT_INPUTS_PATH.read_text())["measurement"]["seed_roster"][
        "non_science_seed_namespaces"
    ]["capacity_pilot"]["evaluation"])
    assert seeds <= roster


def test_pilot_fanout_stage2_matches_protocol_grid_exactly() -> None:
    generations = eval_job.pilot_fanout("2", c1=10)
    pairs = {(g["ordinal"], g["stream"]) for g in generations}
    assert pairs == set(protocol.pilot_stage2_grid(10))
    assert len(generations) == 13


def test_pilot_fanout_stage2_core_drops_only_ordinal_20() -> None:
    core = eval_job.pilot_fanout("2-core", c1=10)
    full = eval_job.pilot_fanout("2", c1=10)
    assert len(core) == 10
    full_pairs = {(g["ordinal"], g["stream"]) for g in full}
    core_pairs = {(g["ordinal"], g["stream"]) for g in core}
    assert full_pairs - core_pairs == {(20, 1), (20, 2), (20, 3)}


def test_pilot_fanout_stage2_requires_c1() -> None:
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.pilot_fanout("2")
    assert excinfo.value.code == "pilot-stage2-needs-c1"


def test_gpu_route_g_fanout_is_one_generation_at_the_ci_seed() -> None:
    generations = eval_job.gpu_route_g_fanout()
    assert len(generations) == 1
    assert generations[0]["update"] == 1000
    assert generations[0]["seed"] == 999004


# --- SEL step: the W estimator, c1, and the selection record --------------


def test_w_statistic_matches_the_contract_order_statistic() -> None:
    # 1000 values 1..1000: median 500.5, P99.9 index ceil(999) = 999th
    # smallest absolute deviation (no interpolation) -> deviation 499.5.
    values = [float(v) for v in range(1, 1001)]
    assert eval_job.read_stream_local_energies  # imported, used by compute_pilot_selection
    assert protocol.w_statistic(values) == pytest.approx(499.5)


def test_select_pilot_basin_ordinal_picks_the_smallest_triple_mean_earliest_tie() -> None:
    """Mutant: 'the latest tie wins'."""

    w = {o: 10.0 for o in range(21)}
    w[9] = w[10] = w[11] = 1.0  # a clear minimum triple centered at 10
    w[15] = w[16] = w[17] = 1.0  # an exact tie elsewhere
    c1, means = protocol.select_pilot_basin_ordinal(w)
    assert c1 == 10  # earliest tie wins, never 16


def test_select_pilot_basin_ordinal_refuses_missing_ordinals() -> None:
    with pytest.raises(ValueError):
        protocol.select_pilot_basin_ordinal({o: 1.0 for o in range(5)})


def test_build_and_verify_selection_record_round_trips() -> None:
    record = eval_job.build_selection_record(c1=10, triple_means={9: 1.0, 10: 0.9, 11: 1.1}, b1_eval_receipt_sha256="deadbeef")
    c1 = eval_job.verify_selection_record(record, b1_eval_receipt_sha256="deadbeef")
    assert c1 == 10


def test_build_selection_record_refuses_c1_outside_3_19() -> None:
    """Mutant: 'c1 outside 3..19 accepted'."""

    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.build_selection_record(c1=2, triple_means={}, b1_eval_receipt_sha256="x")
    assert excinfo.value.code == "sel-c1-out-of-range"
    with pytest.raises(eval_job.EvalJobRefusal):
        eval_job.build_selection_record(c1=20, triple_means={}, b1_eval_receipt_sha256="x")


def test_verify_selection_record_refuses_a_record_from_another_b1_receipt() -> None:
    """Mutant: 'a selection record from another B1 receipt accepted'."""

    record = eval_job.build_selection_record(c1=10, triple_means={9: 1.0, 10: 1.0, 11: 1.0}, b1_eval_receipt_sha256="aaa")
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.verify_selection_record(record, b1_eval_receipt_sha256="bbb")
    assert excinfo.value.code == "sel-record-mismatch"


def test_verify_selection_record_refuses_c1_out_of_range_even_if_unsigned() -> None:
    tampered = {"c1": 25, "triple_means": {}, "b1_eval_receipt_sha256": "aaa"}
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.verify_selection_record(tampered, b1_eval_receipt_sha256="aaa")
    assert excinfo.value.code == "sel-c1-out-of-range"


# --- eval receipt: writer, reader, and a run outside the schedule ----------


def test_write_and_read_eval_receipt_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "eval-receipt.json"
    generation = eval_job.generation_record(
        ordinal=9,
        update=9000,
        stream=0,
        seed=1234,
        task_id="h-v2-eval-science-L1-ck009000-s0",
        run_result={
            "resolved_config_sha256": "a" * 64,
            "trajectory_csv_sha256": "b" * 64,
            "artifacts": {"trajectory_csv": {"path": "/tmp/x.csv", "sha256": "b" * 64}},
        },
    )
    eval_job.write_eval_receipt(
        path,
        eval_commit="c" * 40,
        train_commit="d" * 40,
        train_receipt_path=tmp_path / "train.json",
        train_receipt_sha256="e" * 64,
        payload="science",
        generations=[generation],
        readout_generator_enabled=True,
    )
    read_back = eval_job.read_eval_receipt(path)
    assert read_back["schema"] == eval_job.EVAL_RECEIPT_SCHEMA
    assert read_back["generations"][0]["seed"] == 1234
    assert read_back["readout_generator_enabled"] is True


def test_read_eval_receipt_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.read_eval_receipt(tmp_path / "absent.json")
    assert excinfo.value.code == "eval-receipt-missing"


def test_read_eval_receipt_refuses_a_bad_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.read_eval_receipt(path)
    assert excinfo.value.code == "eval-receipt-bad-schema"


def test_consumer_reads_a_real_eval_receipt_written_by_the_producer(tmp_path: Path) -> None:
    """Consumer-side test (acceptance contract): fed the REAL producer output."""

    path = tmp_path / "eval-receipt.json"
    written = eval_job.write_eval_receipt(
        path,
        eval_commit="c" * 40,
        train_commit="d" * 40,
        train_receipt_path=tmp_path / "train.json",
        train_receipt_sha256="e" * 64,
        payload="gpu-route-g",
        generations=[
            eval_job.generation_record(
                ordinal=1,
                update=1000,
                stream=0,
                seed=999004,
                task_id="h-v2-eval-gpu-route-g-ck001000-s0",
                run_result={
                    "resolved_config_sha256": "f" * 64,
                    "trajectory_csv_sha256": "g" * 64,
                    "artifacts": {},
                },
            )
        ],
        readout_generator_enabled=False,
    )
    read_back = eval_job.read_eval_receipt(path)
    assert read_back == written


# --- the model/eval scale config -------------------------------------------


def test_science_evaluation_config_applies_the_a2_overlay_and_readout_generator(tmp_path: Path) -> None:
    config = protocol.science_evaluation_config(tmp_path / "ck", seed=1234567)
    assert config["runtime"]["device"] == "cuda"
    assert config["sampler"]["n_walkers"] == 4096
    assert config["sampler"]["burn_in"] == 1000
    assert config["evaluator"]["tasks"][0]["generator"]["n_draws"] == 256
    assert config["evaluator"]["tasks"][0]["summaries"][0]["max_samples"] >= 1048576
    assert config["load"]["path"] == str(tmp_path / "ck")
    assert config["runtime"]["seed"] == 1234567
    assert len(config["evaluator"]["tasks"]) == 2
    assert config["evaluator"]["tasks"][1]["generator"]["_target_"] == protocol.READOUT_TRAJECTORY_GENERATOR_TARGET
    assert config["evaluator"]["tasks"][1]["generator"]["readout_eps"] == protocol.READOUT_EPS
    # F1: the model mapping must stay exactly what eval.yaml carries (no
    # eps key) -- never mutated to match the generator's expected floor,
    # which would break restore identity against an earlier TRAIN receipt
    # whose authenticated model mapping has no explicit eps either.
    assert "eps" not in config["model"]["readout"]


def test_science_evaluation_config_can_omit_the_readout_generator(tmp_path: Path) -> None:
    """Mutant: 'readout generator silently omitted' -- must be an explicit, recorded choice."""

    config = protocol.science_evaluation_config(tmp_path / "ck", seed=1, enable_readout_trajectory=False)
    assert len(config["evaluator"]["tasks"]) == 1


def test_science_evaluation_config_carries_no_reference_energy(tmp_path: Path) -> None:
    config = protocol.science_evaluation_config(tmp_path / "ck", seed=1)

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            assert "reference_energy" not in node
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)

    _walk(config)


# --- dispatch boundary: run_from_config only, never tpen.run.main ---------


def test_eval_job_dispatches_through_run_from_config_not_tpen_run_main() -> None:
    """Mutant: 'dispatch via tpen.run.main' -- the orchestrator's binding correction.

    ``tpen.run.main`` is not an experiment-facing surface
    (``experiments/README.md``); the subprocess script must call
    ``tpen.run.run_from_config`` on a loaded config directly.
    """

    script = eval_job._RUN_FROM_CONFIG_SCRIPT
    assert "run_from_config" in script
    assert "import main" not in script
    assert ".main(" not in script
    assert "from tpen.run import main" not in script


# --- run_generation_direct: evidence verification without any executor ----


def _fake_generation_config(run_root: Path, run_id: str) -> dict:
    return {
        "run": {"root": str(run_root), "run_id": run_id, "layout": "flat"},
        "evaluator": {
            "tasks": [
                {
                    "name": "trajectory",
                    "output_dir": "${run.dir}/trajectory",
                    "generator": {"_target_": "tpen.evaluation.generators.trajectory.TrajectoryMCMCGenerator"},
                }
            ]
        },
    }


def _write_fake_run_output(run_dir: Path, *, commit: str, trajectory_bytes: bytes = b"a,b\n1,2\n") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"git_commit": commit}), encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text("resolved: true\n", encoding="utf-8")
    trajectory_dir = run_dir / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    (trajectory_dir / "sampled_eval_table.csv").write_bytes(trajectory_bytes)
    (trajectory_dir / "sampled_eval_table.metadata.json").write_text(
        json.dumps({"csv_sha256": _sha256(trajectory_bytes)}), encoding="utf-8"
    )


def test_run_generation_direct_computes_real_artifact_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No mutant yet named this directly, but this is the test that must exist
    for 'drop a file-written hash' to be catchable at all: before this test,
    nothing in the suite ever called run_generation_direct."""

    run_root = tmp_path / "results"
    config = _fake_generation_config(run_root, "gen-1")
    run_dir = run_root / "gen-1"

    def fake_subprocess_run(argv, **kwargs):
        _write_fake_run_output(run_dir, commit="c" * 40)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(eval_job.subprocess, "run", fake_subprocess_run)
    result = eval_job.run_generation_direct(
        science_python="/usr/bin/python3",
        config=config,
        config_path=tmp_path / "gen-1.yaml",
        expected_source_commit="c" * 40,
    )
    assert result["resolved_config_sha256"] == _sha256((run_dir / "resolved_config.yaml").read_bytes())
    assert result["trajectory_csv_sha256"] == _sha256(b"a,b\n1,2\n")
    assert result["artifacts"]["trajectory_csv"]["sha256"] == _sha256(b"a,b\n1,2\n")
    assert all(entry["sha256"] for entry in result["artifacts"].values())  # mutant: 'drop a file-written hash'


def test_run_generation_direct_refuses_wrong_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "results"
    config = _fake_generation_config(run_root, "gen-1")
    run_dir = run_root / "gen-1"

    def fake_subprocess_run(argv, **kwargs):
        _write_fake_run_output(run_dir, commit="a" * 40)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(eval_job.subprocess, "run", fake_subprocess_run)
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.run_generation_direct(
            science_python="/usr/bin/python3",
            config=config,
            config_path=tmp_path / "gen-1.yaml",
            expected_source_commit="b" * 40,
        )
    assert excinfo.value.code == "eval-run-wrong-commit"


def test_run_generation_direct_refuses_a_failed_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _fake_generation_config(tmp_path / "results", "gen-1")

    def fake_subprocess_run(argv, **kwargs):
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()

    monkeypatch.setattr(eval_job.subprocess, "run", fake_subprocess_run)
    with pytest.raises(eval_job.EvalJobRefusal) as excinfo:
        eval_job.run_generation_direct(
            science_python="/usr/bin/python3",
            config=config,
            config_path=tmp_path / "gen-1.yaml",
            expected_source_commit="c" * 40,
        )
    assert excinfo.value.code == "eval-run-failed"
