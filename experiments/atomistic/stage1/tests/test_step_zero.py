"""Focused Stage-1 contract tests for the retained initial checkpoint."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.atomistic.stage1.protocol import (
    Protocol,
    ProtocolError,
    SYSTEMS,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    evaluation_seed,
    manifest_hash,
)
from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.collect import collect
from experiments.atomistic.stage1.launch import _binding_token, row_overrides
from experiments.toolkit.run_paths import resolve_run_dir
from experiments.toolkit.trajectory_records import inspect_eval_run, inspect_trajectory_record
from experiments.atomistic.stage1.train import prepare_initial_checkpoint_config
from tpen.run import load_config


CONFIG_ROOT = Path(__file__).parents[1] / "configs"


@pytest.fixture
def tiny_protocol() -> Protocol:
    """Return a small protocol with the production four-stream shape."""

    return Protocol(
        smoke_updates=1,
        smoke_walkers=1,
        lineage_updates=7,
        lineage_walkers=2,
        lineages=1,
        eval_streams=4,
        eval_walkers=2,
        eval_draws=2,
    )


def _phase1_rows(protocol: Protocol, checkpoints: list[int], tmp_path: Path) -> list[dict]:
    """Build phase-1 rows for one lineage and all Stage-1 systems."""

    return build_manifest(
        protocol=protocol,
        source_sha="a" * 40,
        config_sha256={system: "b" * 64 for system in SYSTEMS},
        checkpoints=checkpoints,
        lineage_seeds=[11],
        checkpoint_sha256={},
        results_root=str(tmp_path / "results"),
        phase=1,
    )


def test_zero_is_admitted_once_and_negative_or_duplicate_steps_are_rejected(
    tiny_protocol: Protocol, tmp_path: Path,
) -> None:
    """Zero is a coordinate, while negatives and duplicate coordinates fail."""

    rows = _phase1_rows(tiny_protocol, [0, 7], tmp_path)
    train_rows = [row for row in rows if row["kind"] == "train"]
    assert all(row["checkpoint_steps"] == [0, 7] for row in train_rows)

    for checkpoints, message in (([-1, 7], "non-negative"), ([0, 0, 7], "duplicates")):
        with pytest.raises(ProtocolError, match=message):
            _phase1_rows(tiny_protocol, checkpoints, tmp_path)


def test_phase2_binds_zero_digest_and_emits_four_streams(
    tiny_protocol: Protocol, tmp_path: Path,
) -> None:
    """The initial checkpoint is phase-2-bound like a positive checkpoint."""

    phase1_rows = _phase1_rows(tiny_protocol, [0, 7], tmp_path)
    train_rows = [row for row in phase1_rows if row["kind"] == "train"]
    phase1_hash = manifest_hash(phase1_rows)
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=[0],
        checkpoint_evidence={
            "schema": "stage1-checkpoint-evidence/v1",
            "phase1_manifest_hash": phase1_hash,
            "phase1_train_row_hashes": {
                f"{row['system']}:{row['lineage']}": manifest_hash([row])
                for row in train_rows
            },
            "checkpoints": {
                f"{system}:0:0": "c" * 64
                for system in SYSTEMS
            },
        },
        source_sha="a" * 40,
        config_sha256={system: "b" * 64 for system in SYSTEMS},
        protocol=tiny_protocol,
        phase1_manifest_hash=phase1_hash,
    )

    zero_rows = [row for row in eval_rows if row["checkpoint"] == 0]
    assert len(zero_rows) == len(SYSTEMS) * tiny_protocol.eval_streams
    assert {row["checkpoint_sha256"] for row in zero_rows} == {"c" * 64}
    assert all("step_000000" in row["checkpoint_dir"] for row in zero_rows)
    assert train_rows


def test_positive_row_identity_does_not_depend_on_zero_prefix(
    tiny_protocol: Protocol, tmp_path: Path,
) -> None:
    """Adding zero does not shift positive checkpoint coordinates or seeds."""

    def planned(checkpoints: list[int]) -> list[dict]:
        return build_manifest(
            protocol=tiny_protocol,
            source_sha="a" * 40,
            config_sha256={system: "b" * 64 for system in SYSTEMS},
            checkpoints=checkpoints,
            lineage_seeds=[11],
            checkpoint_sha256={
                (system, 0, checkpoint): "c" * 64
                for system in SYSTEMS
                for checkpoint in checkpoints
            },
            results_root=str(tmp_path / "results"),
            phase=2,
        )

    without_zero = planned([7])
    with_zero = planned([0, 7])
    positive_without = [
        (row["row_id"], row["seed"])
        for row in without_zero
        if row["kind"] == "eval" and row["checkpoint"] == 7
    ]
    positive_with = [
        (row["row_id"], row["seed"])
        for row in with_zero
        if row["kind"] == "eval" and row["checkpoint"] == 7
    ]
    assert positive_with == positive_without
    expected_he_seed = next(
        row["seed"] for row in without_zero
        if row["kind"] == "eval" and row["system"] == "he"
        and row["checkpoint"] == 7 and row["stream"] == 2
    )
    assert expected_he_seed == 1582742142991873901
    assert evaluation_seed(system="he", lineage=0, checkpoint=7, stream=2) == expected_he_seed


def test_phase2_refuses_missing_zero_digest(tiny_protocol: Protocol, tmp_path: Path) -> None:
    """A zero row cannot be created without its producer digest."""

    phase1_rows = _phase1_rows(tiny_protocol, [0, 7], tmp_path)
    phase1_hash = manifest_hash(phase1_rows)
    train_rows = [row for row in phase1_rows if row["kind"] == "train"]
    evidence = {
        "schema": "stage1-checkpoint-evidence/v1",
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in train_rows
        },
        "checkpoints": {},
    }
    with pytest.raises(ProtocolError, match="missing digest"):
        bind_evaluation_manifest(
            phase1_rows=phase1_rows,
            checkpoints=[0],
            checkpoint_evidence=evidence,
            protocol=tiny_protocol,
            phase1_manifest_hash=phase1_hash,
        )


@pytest.mark.parametrize("system", SYSTEMS)
def test_zero_is_removed_from_explicit_schedule_and_routes_existing_callback(system: str) -> None:
    """The gateway never instantiates ExplicitUpdates with zero."""

    cfg = load_config(str(CONFIG_ROOT / system / "train.yaml"), ["checkpoint_steps=[0,7]"])
    callback_count = len(cfg.callbacks)

    prepare_initial_checkpoint_config(cfg)

    assert list(cfg.checkpoint_steps) == [7]
    assert list(cfg.callbacks[4].schedule.updates) == [7]
    assert len(cfg.callbacks) == callback_count + 1
    assert cfg.callbacks[-1]._target_ == "tpen.callback.InitialCheckpoint"


def test_positive_only_config_is_unchanged_by_zero_preparation(system: str = "he") -> None:
    """The no-zero path remains the existing schedule/configuration path."""

    cfg = load_config(str(CONFIG_ROOT / system / "train.yaml"), ["checkpoint_steps=[7]"])
    before = list(cfg.checkpoint_steps)
    callback_count = len(cfg.callbacks)

    prepare_initial_checkpoint_config(cfg)

    assert list(cfg.checkpoint_steps) == before == [7]
    assert list(cfg.callbacks[4].schedule.updates) == [7]
    assert len(cfg.callbacks) == callback_count


@pytest.mark.integration
@pytest.mark.stage1_real_producer
def test_zero_and_positive_checkpoint_real_producer_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Produce, evaluate, inspect, and collect zero plus one on Cannon."""

    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[4],
    ).strip()
    protocol = Protocol(
        smoke_updates=1,
        smoke_walkers=1,
        lineage_updates=1,
        lineage_walkers=2,
        lineages=1,
        eval_streams=4,
        eval_walkers=2,
        eval_draws=2,
    )
    systems = ("he",)
    config_sha256 = config_identities(config_root=CONFIG_ROOT, systems=systems)
    phase1_rows = build_manifest(
        protocol=protocol,
        source_sha=source_sha,
        config_sha256=config_sha256,
        checkpoints=[0, 1],
        lineage_seeds=[11],
        checkpoint_sha256={},
        results_root=str(tmp_path / "results"),
        systems=systems,
        phase=1,
        device="cpu",
    )
    train_row = next(row for row in phase1_rows if row["kind"] == "train")
    train_overrides = row_overrides(train_row) + [
        "sampler.burn_in=1", "sampler.n_steps=1",
    ]

    def run_train() -> None:
        monkeypatch.setattr(
            sys, "argv", [
                "stage1-train", "--config", str(CONFIG_ROOT / "he" / "train.yaml"),
                "--kind", "train", *train_overrides,
                "--binding", _binding_token(train_row),
            ],
        )
        assert stage1_train.main() == 0

    run_train()
    run_dir = resolve_run_dir(train_row)
    digests = {
        checkpoint: hashlib.sha256(
            (run_dir / "checkpoints" / f"step_{checkpoint:06d}" / "model.pt").read_bytes()
        ).hexdigest()
        for checkpoint in (0, 1)
    }
    assert all(
        (run_dir / "checkpoints" / f"step_{checkpoint:06d}" / "COMPLETE").is_file()
        for checkpoint in (0, 1)
    )
    phase1_hash = manifest_hash(phase1_rows)
    phase2_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=[0, 1],
        checkpoint_evidence={
            "schema": "stage1-checkpoint-evidence/v1",
            "phase1_manifest_hash": phase1_hash,
            "phase1_train_row_hashes": {
                f"{train_row['system']}:{train_row['lineage']}": manifest_hash([train_row]),
            },
            "checkpoints": {
                f"he:0:{checkpoint}": digest for checkpoint, digest in digests.items()
            },
        },
        protocol=protocol,
        phase1_manifest_hash=phase1_hash,
        source_sha=source_sha,
        config_sha256=config_sha256,
        device="cpu",
    )
    eval_rows = [row for row in phase2_rows if row["kind"] == "eval"]
    for row in eval_rows:
        overrides = row_overrides(row) + [
            "evaluation_sampler.burn_in=1", "evaluation_sampler.n_steps=1",
        ]
        monkeypatch.setattr(
            sys, "argv", [
                "stage1-eval", "--config", str(CONFIG_ROOT / "he" / "eval.yaml"),
                "--kind", "eval", *overrides,
                "--binding", _binding_token(row),
                "--expected-checkpoint-sha256", str(row["checkpoint_sha256"]),
            ],
        )
        assert stage1_eval.main() == 0
        evaluated = inspect_eval_run(resolve_run_dir(row))
        assert evaluated["verdict"] == "completed"
        metadata = resolve_run_dir(row) / "mcmc_energy" / "sampled_eval_table.metadata.json"
        inspected = inspect_trajectory_record(
            metadata,
            expected_checkpoint_sha256=str(row["checkpoint_sha256"]),
        )
        assert inspected.model_provenance == "checkpoint"
        assert inspected.checkpoint_sha256 == row["checkpoint_sha256"]

    document = {
        "schema": "stage1/v3",
        "manifest_hash": manifest_hash(phase2_rows),
        "rows": phase2_rows,
    }
    collected = collect(document)
    assert len(collected["records"]) == protocol.eval_streams * 2
    assert len(collected["aggregates"]) == 2
    assert {aggregate["checkpoint"] for aggregate in collected["aggregates"]} == {0, 1}
    assert all(aggregate["stream_count"] == protocol.eval_streams
               for aggregate in collected["aggregates"])
