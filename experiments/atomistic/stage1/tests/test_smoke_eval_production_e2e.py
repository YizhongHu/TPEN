"""Production-gateway smoke-roster chains at fixture-only artifact scale."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate

from experiments.atomistic.stage1 import eval as eval_module
from experiments.atomistic.stage1 import train as train_module
from experiments.atomistic.stage1.collect import CollectionError, collect
from experiments.atomistic.stage1.launch import _binding_token, command_for_row
from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HOOKE2_ACT50_A27,
    HOOKE2_ACT50_NOLAYER_PROFILE,
    HOOKE2_ACT50_SIGMOID_PROFILE,
    HOOKE_A37,
    HOOKE_SCIENCE_PROFILE,
    SMOKE_EVAL_ROSTER_MARKER,
    STAGE1_PROTOCOL,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    derive_smoke_request,
    manifest_hash,
    smoke_eval_roster,
)
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.artifacts import RunResult
from tpen.checkpoint.hashing import checkpoint_hashes, resolved_config_identity
from tpen.checkpoint.manifest import CHECKPOINT_KIND, CHECKPOINT_SCHEMA_VERSION, CheckpointManifest
from tpen.checkpoint.payload import MODEL_ONLY_PAYLOAD
from tpen.evaluation.trajectory_records import TrajectoryRecordBatch, TrajectoryRecordStreamWriter
from tpen.physics.hamiltonian import NaiveLocalEnergyEvaluator
from tpen.statistics.trajectory import ObservableTrajectory


ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs"


def _source_sha() -> str:
    """Use the exact checkout identity without embedding a source hash."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parents[2], text=True,
    ).strip()


def _family_train_row(tmp_path: Path, *, profile: str, system: str) -> dict[str, object]:
    """Build one production-shaped science row for a fixture-only chain."""

    checkpoints = list(range(
        0,
        50_001 if profile in {
            HOOKE2_ACT50_SIGMOID_PROFILE, HOOKE2_ACT50_NOLAYER_PROFILE,
        } else STAGE1_PROTOCOL.lineage_updates + 1,
        1_000,
    ))
    config_profile = profile if profile != HOOKE_SCIENCE_PROFILE else None
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha=_source_sha(),
        config_sha256=config_identities(
            config_root=CONFIG_ROOT, systems=(system,), science_profile=config_profile,
        ),
        checkpoints=checkpoints,
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path / "science"),
        systems=(system,),
        phase=1,
        science_profile=profile,
        evaluation_selection=("A27" if config_profile is not None else "A37"),
        include_smoke=False,
        auto_overlay=True,
    )
    return next(row for row in rows if row["kind"] == "train" and row["lineage"] == 0)


def _write_fixture_checkpoint(cfg: object, checkpoint_dir: Path, *, step: int) -> str:
    """Publish a model-only checkpoint fixture accepted by the eval gateway."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = f"fixture-model-{step}".encode()
    model = checkpoint_dir / "model.pt"
    model.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    hashes = checkpoint_hashes(cfg)
    hashes["model"] = digest
    CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        kind=CHECKPOINT_KIND,
        next_iteration=step,
        completed_updates=step,
        created_at_unix=0.0,
        files={"model": "model.pt"},
        hashes=hashes,
        runtime={},
        provenance={},
        payload=MODEL_ONLY_PAYLOAD.to_manifest(),
    ).write(checkpoint_dir / "manifest.json")
    (checkpoint_dir / "COMPLETE").write_text("fixture\n", encoding="utf-8")
    return digest


def _fixture_train_runner(rows: dict[str, dict[str, object]], seen: list[int]):
    """Return a runner that records PlannedYield and writes fixture checkpoints."""

    def run(cfg, *, runtime_callbacks=(), **_kwargs) -> int:
        run_id = str(cfg.run.run_id)
        row = rows[run_id]
        callbacks = tuple(runtime_callbacks)
        assert len(callbacks) == 1
        callback = callbacks[0]
        assert type(callback).__name__ == "PlannedYield"
        assert callback.target_iteration == row["execution"]["stop_at"]
        assert callback.target_iterations == row["updates"]
        seen.append(int(callback.target_iteration))
        run_dir = resolve_run_dir(row)
        for step in (0, int(row["execution"]["stop_at"])):
            _write_fixture_checkpoint(
                cfg, run_dir / "checkpoints" / f"step_{step:06d}", step=step,
            )
        return 0

    return run


def _fixture_eval_runner(rows: dict[str, dict[str, object]]):
    """Return a runner that writes real trajectory-record fixture artifacts."""

    def run(cfg, **_kwargs) -> int:
        row = rows[str(cfg.run.run_id)]
        run_dir = resolve_run_dir(row)
        run_dir.mkdir(parents=True, exist_ok=True)
        identity = resolved_config_identity(cfg)
        (run_dir / "run_start.json").write_text(
            json.dumps({
                "resolved_config_identity": identity,
                "git": {"sha": row["source_sha"]},
            }), encoding="utf-8",
        )
        (run_dir / "status.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8",
        )
        (run_dir / "run-outcome.json").write_text(
            json.dumps(RunResult(status="completed").to_dict()), encoding="utf-8",
        )
        descriptor = instantiate(cfg.system_descriptor)
        terms = instantiate(cfg.hamiltonian_terms)
        receipt = NaiveLocalEnergyEvaluator().describe_execution_receipt(
            terms, descriptor.system_id(),
        )
        walkers = int(row["walkers"])
        draws = int(row["draws"])
        electrons = int(cfg.system.n_particles)
        dimensions = int(cfg.system.spatial_dim)
        values = torch.arange(1, draws * walkers + 1, dtype=torch.float64).reshape(draws, walkers)

        def draw(index: int) -> TrajectoryRecordBatch:
            local = values[index]
            return TrajectoryRecordBatch(
                draw_index=torch.full((walkers,), index, dtype=torch.int64),
                walker_index=torch.arange(walkers, dtype=torch.int64),
                positions=torch.zeros(walkers, electrons, dimensions, dtype=torch.float64),
                local_energy=local,
                term_energies={name: local / len(terms) for name in terms},
                logabs=torch.zeros(walkers), sign=torch.ones(walkers),
                finite_mask=torch.ones(walkers, dtype=torch.bool),
            )

        task_dir = run_dir / "mcmc_energy"
        task_dir.mkdir(parents=True, exist_ok=True)
        writer = TrajectoryRecordStreamWriter(
            task_dir / "sampled_eval_table.csv", observable="local_energy",
            n_draws=draws, n_walkers=walkers, term_names=tuple(terms),
            system_descriptor=descriptor, execution_receipt=receipt,
            resolved_config_identity=identity, model_provenance="checkpoint",
            checkpoint_sha256=str(row["checkpoint_sha256"]), first_draw=draw(0),
        )
        for index in range(1, draws):
            writer.append(draw(index))
        writer.finalize(ObservableTrajectory(
            "local_energy", values, draw_stride=1, burn_in_draws=0,
        ))
        return 0

    return run


@pytest.mark.parametrize(
    ("profile", "system", "science_roster"),
    (
        (HOOKE_SCIENCE_PROFILE, "hooke1", HOOKE_A37),
        (HOOKE2_ACT50_SIGMOID_PROFILE, "hooke2", HOOKE2_ACT50_A27),
        (HOOKE2_ACT50_NOLAYER_PROFILE, "hooke2", HOOKE2_ACT50_A27),
    ),
)
def test_smoke_roster_real_gateway_chain(
    profile: str, system: str, science_roster: tuple[int, ...],
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property,
) -> None:
    """Run derive, planner CLI, train/eval gateways, and real collection."""

    source = _family_train_row(tmp_path, profile=profile, system=system)
    smoke_row = derive_smoke_request(
        source, results_root=str(tmp_path / "smoke"), stop_at=1_000,
    )
    phase1_doc = {
        "schema": "stage1/v4", "phase": 1,
        "manifest_hash": manifest_hash([smoke_row]), "rows": [smoke_row],
    }
    phase1_path = tmp_path / "phase1.json"
    phase1_path.write_text(json.dumps(phase1_doc, indent=2) + "\n", encoding="utf-8")

    train_rows = {str(smoke_row["run_id"]): smoke_row}
    train_seen: list[int] = []
    monkeypatch.setattr(train_module, "run_from_config", _fixture_train_runner(train_rows, train_seen))
    train_command = command_for_row(
        smoke_row, python=sys.executable, driver="experiments/atomistic/stage1/train.py",
    )
    assert "--binding" in train_command
    monkeypatch.setattr(sys, "argv", ["stage1-train", "--binding", _binding_token(smoke_row)])
    assert train_module.main() == 0
    assert train_seen == [1_000]

    checkpoint_sha = {
        checkpoint: hashlib.sha256(
            (resolve_run_dir(smoke_row) / "checkpoints" / f"step_{checkpoint:06d}" / "model.pt").read_bytes()
        ).hexdigest()
        for checkpoint in (0, 1_000)
    }
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": manifest_hash([smoke_row]),
        "phase1_train_row_hashes": {
            f"{smoke_row['system']}:{smoke_row['lineage']}": manifest_hash([smoke_row])
        },
        "checkpoints": {
            f"{smoke_row['system']}:{smoke_row['lineage']}:{checkpoint}": digest
            for checkpoint, digest in checkpoint_sha.items()
        },
    }
    evidence_path = tmp_path / "checkpoint-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    phase2_path = tmp_path / "phase2.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "plan.py"), "--output", str(phase2_path),
         "--phase1-manifest", str(phase1_path), "--checkpoint-bindings", str(evidence_path)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    phase2_doc = json.loads(phase2_path.read_text(encoding="utf-8"))
    assert {row.get("evaluation_roster") for row in phase2_doc["rows"]} == {
        SMOKE_EVAL_ROSTER_MARKER,
    }
    assert {row["checkpoint"] for row in phase2_doc["rows"] if row["kind"] == "eval"} == {0, 1_000}
    assert set(smoke_eval_roster(smoke_row, science_roster=science_roster)) == {0, 1_000}

    eval_rows = [row for row in phase2_doc["rows"] if row["kind"] == "eval"]
    for row in eval_rows:
        row["draws"] = 2
        row["walkers"] = 2
        row["checkpoint_sha256"] = checkpoint_sha[int(row["checkpoint"])]
    eval_by_run = {str(row["run_id"]): row for row in eval_rows}
    monkeypatch.setattr(eval_module, "run_from_config", _fixture_eval_runner(eval_by_run))
    eval_command = command_for_row(
        eval_rows[0], python=sys.executable, driver="experiments/atomistic/stage1/eval.py",
    )
    assert "--binding" in eval_command
    for row in eval_rows:
        monkeypatch.setattr(sys, "argv", ["stage1-eval", "--binding", _binding_token(row)])
        assert eval_module.main() == 0

    rows = [smoke_row, *eval_rows]
    document = {"schema": "stage1/v4", "manifest_hash": manifest_hash(rows), "rows": rows}
    collected = collect(document)
    assert len(collected["records"]) == 2 * 4
    assert {record["checkpoint"] for record in collected["records"]} == {0, 1_000}

    science_row = copy.deepcopy(eval_rows[0])
    science_row["checkpoint"] = science_roster[-1]
    science_row["row_id"] = f"{science_row['row_id']}-science-grid"
    science_row["run_id"] = f"{science_row['run_id']}-science-grid"
    refused = [*rows, science_row]
    with pytest.raises(CollectionError, match="smoke-prefix-v1"):
        collect({
            "schema": "stage1/v4", "manifest_hash": manifest_hash(refused), "rows": refused,
        })
