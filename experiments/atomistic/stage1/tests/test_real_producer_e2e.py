"""Real Stage-1 producer chain from planning through collection.

This test is intentionally small, but it does not synthesize any run
artifacts.  The train and eval artifacts are written by the public TPEN run
entry point and are inspected through the experiments toolkit boundary.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.collect import CollectionError, collect
from experiments.atomistic.stage1.launch import command_for_row, row_overrides
from experiments.atomistic.stage1.protocol import (
    HOOKE_EVAL_SELECTION,
    HOOKE_SCIENCE_PROFILE,
    Protocol,
    SYSTEMS,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    manifest_hash,
    overlay_sha256,
    physical_system_id_from_config,
    validate_gateway_binding,
)
from experiments.atomistic.stage1.train import prepare_initial_checkpoint_config
from experiments.baselines.systems import load_registry
from experiments.toolkit.run_paths import resolve_run_dir
from experiments.toolkit.trajectory_records import (
    inspect_eval_run,
    inspect_trajectory_record,
)
from tpen.run import load_config, run_from_config


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
HOOKE_OVERLAY = CONFIG_ROOT / "overlays" / "hooke-diagnostics-v1.yaml"
REPO_ROOT = Path(__file__).resolve().parents[4]
EXPECTED_REGISTRY = {
    "hooke1": ("hooke_one_electron_omega0.5", 0.75, "transcribed"),
    "hooke2": ("hooke_pair_singlet_omega0.5", 2.0, "transcribed"),
    "qho2e": ("qho_pair_noninteracting_singlet_omega0.5", 1.5, "transcribed"),
    "he": ("he_atom", -2.903724377034119598, "transcribed"),
    "h2": ("h2_molecule", -1.17447, "transcribed"),
}


def _config_sha256() -> str:
    """Hash the four science-default config pairs used by this test."""

    digest = hashlib.sha256()
    for system in SYSTEMS:
        for kind in ("train", "eval"):
            digest.update((CONFIG_ROOT / system / f"{kind}.yaml").read_bytes())
    return digest.hexdigest()


def _source_sha() -> str:
    """Read the exact checkout identity recorded by real TPEN runs."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
    ).strip()


def _run_real_config(*, row: dict[str, object], config_path: Path,
                     extra_overrides: list[str], command: str) -> None:
    """Run one manifest row through the public configured-run entry point."""

    overrides = row_overrides(row) + extra_overrides
    validate_gateway_binding(
        binding=_binding_token(row),
        config=str(row["config"]),
        overrides=overrides,
        kind=str(row["kind"]),
        expected_checkpoint_sha256=(
            None if row["kind"] != "eval" else str(row["checkpoint_sha256"])
        ),
    )
    # command_for_row is the launcher's rendered command contract.  The test
    # invokes its public run_from_config equivalent directly so no scheduler or
    # shell is needed inside pytest.
    assert str(row["config"]) in command_for_row(
        row, python=sys.executable, driver="experiments/atomistic/stage1/train.py"
        if row["kind"] == "train" else "experiments/atomistic/stage1/eval.py"
    )
    cfg = load_config(str(config_path), overrides)
    if row["kind"] == "train":
        prepare_initial_checkpoint_config(cfg)
    assert run_from_config(
        cfg, config_path=str(config_path), command=command, raise_exceptions=True
    ) == 0


def _binding_token(row: dict[str, object]) -> str:
    """Import the launch binding seam without widening the public test API."""

    from experiments.atomistic.stage1.launch import _binding_token as encode

    return encode(row)


@pytest.mark.integration
@pytest.mark.stage1_real_producer
def test_real_train_eval_inspect_collect_chain_for_all_stage1_systems(
    tmp_path: Path, record_property, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Produce and collect tiny real runs for the Stage-1 system roster."""

    source_sha = _source_sha()
    config_sha256 = _config_sha256()
    protocol = Protocol(
        smoke_updates=1,
        smoke_walkers=2,
        lineage_updates=1,
        lineage_walkers=2,
        lineages=3,
        eval_streams=4,
        eval_walkers=2,
        eval_draws=2,
    )
    placeholder_sha = "0" * 64
    bindings = {
        (system, lineage, checkpoint): placeholder_sha
        for system in SYSTEMS
        for lineage in range(protocol.lineages)
        for checkpoint in (0, 1)
    }
    planned = build_manifest(
        protocol=protocol,
        source_sha=source_sha,
        config_sha256=config_sha256,
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256=bindings,
        results_root=str(tmp_path),
    )
    train_rows = {
        system: next(row for row in planned
                     if row["kind"] == "train" and row["system"] == system
                     and row["lineage"] == 0)
        for system in SYSTEMS
    }

    eval_rows = [
        row for row in planned
        if row["kind"] == "eval" and row["lineage"] == 0 and row["checkpoint"] == 1
    ]
    assert len(eval_rows) == len(SYSTEMS) * protocol.eval_streams
    assert all(Path(str(row["checkpoint_file"])).is_absolute() for row in eval_rows)

    # The checkpoint content is necessarily produced by training, so the
    # planned checkpoint directory is fixed first and its digest is bound to
    # eval rows immediately after the real producer publishes model.pt.
    checkpoint_digests: dict[str, str] = {}
    total_started = time.perf_counter()
    for system, row in train_rows.items():
        system_started = time.perf_counter()
        _run_real_config(
            row=row,
            config_path=CONFIG_ROOT / system / "train.yaml",
            extra_overrides=["sampler.burn_in=1", "sampler.n_steps=1"],
            command=f"stage1 real-producer train {system}",
        )
        checkpoint_dir = resolve_run_dir(row) / "checkpoints" / "step_000001"
        assert checkpoint_dir == Path(str(next(
            eval_row for eval_row in eval_rows if eval_row["system"] == system
        )["checkpoint_dir"]))
        assert (checkpoint_dir / "COMPLETE").is_file()
        checkpoint_file = checkpoint_dir / "model.pt"
        assert checkpoint_file.is_file()
        checkpoint_digests[system] = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()

        for eval_row in eval_rows:
            if eval_row["system"] != system:
                continue
            eval_row["checkpoint_sha256"] = checkpoint_digests[system]
            _run_real_config(
                row=eval_row,
                config_path=CONFIG_ROOT / system / "eval.yaml",
                extra_overrides=[
                    "evaluation_sampler.burn_in=1",
                    "evaluation_sampler.n_steps=1",
                ],
                command=f"stage1 real-producer eval {system} stream {eval_row['stream']}",
            )
            run_dir = resolve_run_dir(eval_row)
            evidence = inspect_eval_run(run_dir)
            assert evidence["verdict"] == "completed"
            identity = evidence["resolved_config_identity"]
            run_start = json.loads((run_dir / "run_start.json").read_text(encoding="utf-8"))
            assert run_start["resolved_config_identity"] == identity
            assert run_start["resolved_config_identity"]["schema"] == "tpen-config-identity-v1"
            metadata_path = run_dir / "mcmc_energy" / "sampled_eval_table.metadata.json"
            inspected = inspect_trajectory_record(
                metadata_path,
                expected_resolved_config_identity=identity,
                expected_checkpoint_sha256=checkpoint_digests[system],
            )
            assert inspected.record_schema == "trajectory_records/v2"
            assert inspected.model_provenance == "checkpoint"
            assert inspected.checkpoint_sha256 == checkpoint_digests[system]
            assert inspected.resolved_config_identity == identity
            assert json.loads(metadata_path.read_text(encoding="utf-8"))["checkpoint_sha256"] == (
                checkpoint_digests[system]
            )
        system_seconds = time.perf_counter() - system_started
        record_property(f"{system}_runtime_seconds", round(system_seconds, 3))
        print(f"{system} real-producer runtime: {system_seconds:.3f}s")

    for row in eval_rows:
        assert row["checkpoint_sha256"] == checkpoint_digests[str(row["system"])]

    total_seconds = time.perf_counter() - total_started
    record_property("total_runtime_seconds", round(total_seconds, 3))
    print(f"total real-producer runtime: {total_seconds:.3f}s")

    document = {
        "schema": "stage1/v3",
        "manifest_hash": manifest_hash(eval_rows),
        "rows": eval_rows,
    }
    collected = collect(document)
    assert len(collected["records"]) == len(SYSTEMS) * protocol.eval_streams
    assert len(collected["aggregates"]) == len(SYSTEMS)
    registry = {
        entry["id"]: entry for entry in load_registry()["systems"]
    }
    assert set(EXPECTED_REGISTRY) == set(SYSTEMS)
    assert {expected[0] for expected in EXPECTED_REGISTRY.values()} <= set(registry)
    assert all(
        record["identity"]["physical_system_id"]
        == record["identity"]["expected_physical_system_id"]
        for record in collected["records"]
    )
    for record in collected["records"]:
        system = next(system for system, expected in EXPECTED_REGISTRY.items()
                      if expected[0] == record["system_id"])
        expected_id, expected_reference, expected_confidence = EXPECTED_REGISTRY[system]
        assert record["system_id"] == expected_id
        assert registry[expected_id]["reference_energy_hartree"] == expected_reference
        assert registry[expected_id]["confidence"] == expected_confidence
        assert record["reference"]["energy_hartree"] == expected_reference
        assert record["reference"]["confidence"] == expected_confidence
        mean = record["statistics"]["mean"]
        signed_error = mean - expected_reference
        assert record["reference"]["error"] == signed_error
        assert record["reference"]["absolute_error"] == abs(signed_error)

    assert {
        (aggregate["system_id"], aggregate["lineage"], aggregate["checkpoint"])
        for aggregate in collected["aggregates"]
    } == {(expected[0], 0, 1) for expected in EXPECTED_REGISTRY.values()}
    assert all(aggregate["stream_count"] == protocol.eval_streams
               for aggregate in collected["aggregates"])

    wrong_physical_id = json.loads(json.dumps(document))
    wrong_physical_id["rows"][0]["expected_physical_system_id"] = "f" * 64
    wrong_physical_id["manifest_hash"] = manifest_hash(wrong_physical_id["rows"])
    with pytest.raises(CollectionError, match="physical system identity"):
        collect(wrong_physical_id)

    wrong_checkpoint = json.loads(json.dumps(document))
    wrong_checkpoint["rows"][0]["checkpoint_sha256"] = "0" * 64
    wrong_checkpoint["manifest_hash"] = manifest_hash(wrong_checkpoint["rows"])
    with pytest.raises(CollectionError, match="checkpoint"):
        collect(wrong_checkpoint)


@pytest.mark.integration
@pytest.mark.stage1_real_producer
@pytest.mark.parametrize("system", ["hooke1", "hooke2"])
def test_bounded_real_hooke_overlay_chain_preserves_diagnostics_and_readout(
    system: str, tmp_path: Path, record_property, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a labeled, bounded Hooke qualification train/eval/collect chain.

    This is deliberately not a science-profile row: the one-update protocol
    owns a separate bounded identity and never masquerades as the 100k row.
    """

    started = time.perf_counter()
    source_sha = _source_sha()
    protocol = Protocol(
        smoke_updates=1,
        smoke_walkers=2,
        lineage_updates=1,
        lineage_walkers=2,
        lineages=3,
        eval_streams=4,
        eval_walkers=2,
        eval_draws=2,
    )
    phase1_rows = build_manifest(
        protocol=protocol,
        source_sha=source_sha,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=(system,)),
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path / "hooke-bounded-qualification"),
        systems=(system,),
        phase=1,
        device="cpu",
        overlay=str(HOOKE_OVERLAY),
        overlay_sha256=overlay_sha256(HOOKE_OVERLAY),
    )
    train_row = next(
        row for row in phase1_rows
        if row["kind"] == "train" and row["lineage"] == 0
    )
    train_argv = [
        "stage1-train", "--config", str(train_row["config"]), "--kind", "train",
        *row_overrides(train_row), "sampler.burn_in=1", "sampler.n_steps=1",
        "--binding", _binding_token(train_row),
        "--overlay", str(train_row["overlay"]),
        "--overlay-sha256", str(train_row["overlay_sha256"]),
    ]
    with monkeypatch.context() as isolated:
        isolated.setattr(sys, "argv", train_argv)
        assert stage1_train.main() == 0

    train_dir = resolve_run_dir(train_row)
    train_records = [
        json.loads(line)
        for line in (train_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    namespaces = {record.get("namespace") for record in train_records}
    assert "train/amplitude_risk" in namespaces
    assert "train/module_norms/readout" in namespaces
    checkpoint_file = train_dir / "checkpoints" / "step_000001" / "model.pt"
    assert checkpoint_file.is_file()
    checkpoint_sha = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()

    evidence = {
        "schema": "stage1-checkpoint-evidence/v1",
        "phase1_manifest_hash": manifest_hash(phase1_rows),
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:1": (
                checkpoint_sha if lineage == 0 else "0" * 64
            )
            for lineage in range(protocol.lineages)
        },
    }
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=[1],
        checkpoint_evidence=evidence,
        expected_physical_system_id={
            system: physical_system_id_from_config(CONFIG_ROOT / system / "eval.yaml")
        },
        protocol=protocol,
        source_sha=source_sha,
        device="cpu",
    )
    produced_eval_rows = [row for row in eval_rows if row["lineage"] == 0]
    for eval_row in produced_eval_rows:
        overrides = row_overrides(eval_row) + [
            "evaluation_sampler.burn_in=1", "evaluation_sampler.n_steps=1",
        ]
        validate_gateway_binding(
            binding=_binding_token(eval_row), config=str(eval_row["config"]),
            overrides=overrides, kind="eval",
            expected_checkpoint_sha256=checkpoint_sha,
        )
        eval_cfg = load_config(str(CONFIG_ROOT / system / "eval.yaml"), overrides)
        assert run_from_config(
            eval_cfg, config_path=str(CONFIG_ROOT / system / "eval.yaml"),
            command=f"stage1 bounded Hooke qualification eval {system}",
            raise_exceptions=True,
        ) == 0
        run_dir = resolve_run_dir(eval_row)
        assert inspect_eval_run(run_dir)["verdict"] == "completed"

    collected = collect({
        "schema": "stage1/v3",
        "manifest_hash": manifest_hash(phase1_rows + produced_eval_rows),
        "rows": phase1_rows + produced_eval_rows,
    })
    assert len(collected["records"]) == protocol.eval_streams
    assert len(collected["aggregates"]) == 1
    record_property(
        f"{system}_bounded_qualification_runtime_seconds",
        round(time.perf_counter() - started, 3),
    )
