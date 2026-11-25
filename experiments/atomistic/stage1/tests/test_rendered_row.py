"""Focused admission tests for materialized Stage-1 rows."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from experiments.atomistic.stage1 import protocol


ROOT = Path(__file__).parents[1]


def _write_rendered(path: Path, payload: dict[str, object]) -> str:
    """Write one deterministic YAML mapping and return its exact digest."""

    raw = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _train_row(tmp_path: Path) -> tuple[dict[str, object], Path]:
    """Build a small valid v1 materialized train row and its input file."""

    results_root = tmp_path / "results"
    rendered_path = results_root / "inputs" / "train.yaml"
    rendered_path.parent.mkdir(parents=True)
    overlay = ROOT / protocol.HOOKE_OVERLAY_RELATIVE_PATH
    overlay_digest = protocol.overlay_sha256(overlay)
    row: dict[str, object] = {
        "row_id": "hooke2-materialized-train",
        "system": "hooke2", "system_id": "hooke_pair_singlet_omega0.5",
        "kind": "train", "seed": 17, "source_sha": "a" * 40,
        "config_sha256": protocol.config_identities(
            config_root=ROOT / "configs", systems=("hooke2",)
        )["hooke2"], "config_identity_rule": protocol.CONFIG_IDENTITY_RULE,
        "config": "configs/hooke2/train.yaml", "results_root": str(results_root),
        "experiment_name": "stage1", "sector": "hooke2", "run_id": "materialized-train",
        "updates": 10, "walkers": 2, "device": "cpu", "lineage": 0,
        "sampler_seed": 23, "checkpoint_steps": [0, 1, 10],
        "overlay": protocol.HOOKE_OVERLAY_RELATIVE_PATH, "overlay_sha256": overlay_digest,
        "row_extension": {
            "version": protocol.ROW_EXTENSION_VERSION,
            "rendered_config": {"path": str(rendered_path), "sha256": ""},
            "diagnostics_profile": protocol.OVERLAY_PROFILE_V1,
        },
    }
    run_dir = protocol.resolve_run_dir(row)
    payload: dict[str, object] = {
        "run": {"root": str(results_root), "run_id": row["run_id"], "dir": str(run_dir), "layout": "nested"},
        "experiment": {"name": "stage1", "sector": "hooke2"},
        "runtime": {"seed": 17, "device": "cpu", "dtype": "float64"},
        "system": {"id": row["system_id"]},
        "stage1_overlay": {
            "path": protocol.HOOKE_OVERLAY_RELATIVE_PATH,
            "sha256": overlay_digest,
            "composition_order": protocol.OVERLAY_COMPOSITION_ORDER,
        },
        "model": {"_target_": "example.Model"}, "hamiltonian_terms": {},
        "optimizer": {"_target_": "torch.optim.Adam"},
        "sampler": {"_target_": "example.Sampler", "n_walkers": 2, "seed": 23},
        "trainer": {"_target_": "example.Trainer", "max_steps": 10},
        "checkpoint_steps": [1, 10],
        "runner": {
            "_target_": "tpen.runner.Train", "model": {"_target_": "example.Model"},
            "sampler": {"_target_": "example.Sampler", "n_walkers": 2, "seed": 23},
            "hamiltonian_terms": {}, "optimizer": {"_target_": "torch.optim.Adam"},
            "trainer": {"_target_": "example.Trainer", "max_steps": 10},
        },
        "callbacks": [
            {"_target_": "tpen.callback.Checkpoint", "terminal": False,
             "schedule": {"_target_": "tpen.checkpoint.ExplicitUpdates", "updates": [1, 10]},
             "payload": {"_target_": "tpen.checkpoint.TrainResume"},
             "output_dir": str(run_dir / "checkpoints")},
            {"_target_": "tpen.callback.InitialCheckpoint", "output_dir": str(run_dir / "checkpoints")},
        ],
    }
    row["row_extension"]["rendered_config"]["sha256"] = _write_rendered(rendered_path, payload)  # type: ignore[index]
    return row, rendered_path


def test_materialized_train_reads_one_checked_mapping_and_retains_identity(tmp_path: Path) -> None:
    row, rendered_path = _train_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    assert admission.rendered_path == rendered_path.resolve()
    assert admission.rendered_file_sha256 == row["row_extension"]["rendered_config"]["sha256"]  # type: ignore[index]
    protocol.validate_materialized_admission(cfg, row, admission)


def test_materialized_train_rejects_runner_copy_mismatch(tmp_path: Path) -> None:
    row, _ = _train_row(tmp_path)
    cfg, admission = protocol.load_materialized_config(row, expected_kind="train")
    cfg.runner.sampler.n_walkers = 3
    with pytest.raises(protocol.ProtocolError, match="runner.sampler"):
        protocol.validate_materialized_admission(cfg, row, admission)


def test_materialized_train_rejects_native_smoke_derivation(tmp_path: Path) -> None:
    row, _ = _train_row(tmp_path)
    with pytest.raises(protocol.ProtocolError, match="native smoke adapter"):
        protocol.derive_smoke_request(row, results_root=str(tmp_path / "other"), stop_at=1)
