"""Acceptance tests for positive-singlet native eval selection."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.run import load_config
from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token, command_for_row
from experiments.atomistic.stage1.protocol import (
    Protocol,
    build_manifest,
    bind_evaluation_manifest,
    config_identities,
    decode_binding,
    manifest_hash,
    physical_system_id_from_config,
    resolve_run_dir,
)


ROOT = Path(__file__).parents[1] / "configs"
HOOKE2 = ROOT / "hooke2"
ARMS = ("positive_singlet_tail", "positive_singlet_comparator")


def _phase2_rows(
    *, config_variant: str | None, results_root: str = "/external/stage1",
) -> list[dict[str, object]]:
    """Build the smallest native phase-2 manifest for one Hooke-2 arm."""

    protocol = Protocol(
        smoke_updates=1, smoke_walkers=2, lineage_updates=3,
        lineage_walkers=2, lineages=1, eval_streams=1, eval_walkers=2,
        eval_draws=2,
    )
    config_sha = config_identities(config_root=ROOT, systems=("hooke2",))
    rows = build_manifest(
        protocol=protocol, source_sha="a" * 40, config_sha256=config_sha,
        checkpoints=[0, 3], lineage_seeds=[11],
        checkpoint_sha256={("hooke2", 0, 0): "b" * 64,
                           ("hooke2", 0, 3): "c" * 64},
        results_root=results_root, systems=("hooke2",), phase=2,
        expected_physical_system_id={
            "hooke2": physical_system_id_from_config(HOOKE2 / "eval.yaml"),
        },
    )
    if config_variant is not None:
        for row in rows:
            row["config_identity_rule"] = "legacy-config-sha256"
            row["config"] = (
                f"configs/hooke2/{config_variant}.yaml"
                if row["kind"] == "train"
                else f"configs/hooke2/{config_variant}_eval.yaml"
            )
            row["row_id"] = manifest_hash([
                {key: value for key, value in row.items() if key != "row_id"}
            ])
    return rows


def _bound_variant_eval_row() -> dict[str, object]:
    """Bind one variant train row through the native phase-2 adapter."""

    protocol = Protocol(
        smoke_updates=1, smoke_walkers=2, lineage_updates=3,
        lineage_walkers=2, lineages=1, eval_streams=1, eval_walkers=2,
        eval_draws=2,
    )
    phase1 = build_manifest(
        protocol=protocol, source_sha="a" * 40,
        config_sha256=config_identities(config_root=ROOT, systems=("hooke2",)),
        checkpoints=[0, 3], lineage_seeds=[11], checkpoint_sha256={},
        results_root="/external/stage1", systems=("hooke2",), phase=1,
    )
    train = next(row for row in phase1 if row["kind"] == "train")
    train["config"] = "configs/hooke2/positive_singlet_tail.yaml"
    train["row_id"] = manifest_hash([
        {key: value for key, value in train.items() if key != "row_id"}
    ])
    phase1_hash = manifest_hash(phase1)
    evidence = {
        "schema": "stage1-checkpoint-evidence/v1",
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            "hooke2:0": manifest_hash([train]),
        },
        "checkpoints": {"hooke2:0:3": "c" * 64},
    }
    return next(row for row in bind_evaluation_manifest(
        phase1_rows=phase1, checkpoints=[3], checkpoint_evidence=evidence,
        expected_physical_system_id={"hooke2": "d" * 64},
        config_sha256=config_identities(config_root=ROOT, systems=("hooke2",)),
        protocol=protocol, phase1_manifest_hash=phase1_hash,
    ) if row["kind"] == "eval")


def test_positive_singlet_train_and_eval_graphs_strictly_match() -> None:
    """Both opt-in eval graphs must accept the corresponding train state dict."""

    for arm in ARMS:
        train = load_config(str(HOOKE2 / f"{arm}.yaml"))
        evaluation = load_config(str(HOOKE2 / f"{arm}_eval.yaml"))
        assert OmegaConf.to_container(train.model, resolve=True) == OmegaConf.to_container(
            evaluation.model, resolve=True,
        )
        train_model = instantiate(train.model)
        eval_model = instantiate(evaluation.model)
        eval_model.load_state_dict(train_model.state_dict(), strict=True)


def test_positive_singlet_eval_exposure_matches_generic_eval() -> None:
    """Variant evals retain the generic sampler and measurement exposure."""

    generic = load_config(str(HOOKE2 / "eval.yaml"))
    for arm in ARMS:
        variant = load_config(str(HOOKE2 / f"{arm}_eval.yaml"))
        assert OmegaConf.to_container(
            variant.evaluation_sampler, resolve=True,
        ) == OmegaConf.to_container(generic.evaluation_sampler, resolve=True)
        assert OmegaConf.to_container(
            variant.evaluation_tasks, resolve=True,
        ) == OmegaConf.to_container(generic.evaluation_tasks, resolve=True)
        assert OmegaConf.to_container(
            variant.evaluator, resolve=True,
        ) == OmegaConf.to_container(generic.evaluator, resolve=True)


def test_native_eval_rows_and_launch_preserve_positive_singlet_selection() -> None:
    """Native phase-2 rows and rendered bindings carry the selected eval graph."""

    for arm in ARMS:
        rows = _phase2_rows(config_variant=arm)
        train = next(row for row in rows if row["kind"] == "train")
        evaluation = next(row for row in rows if row["kind"] == "eval")
        assert train["config"] == f"configs/hooke2/{arm}.yaml"
        assert evaluation["config"] == f"configs/hooke2/{arm}_eval.yaml"
        command = command_for_row(
            evaluation, python="/usr/bin/python3", driver="driver.py",
        )
        token = command.rsplit(" ", 1)[-1]
        assert decode_binding(token)["config"] == evaluation["config"]


def test_native_phase2_binding_selects_tail_eval_config() -> None:
    """The native phase-1-to-phase-2 adapter follows the selected train arm."""

    assert _bound_variant_eval_row()["config"] == (
        "configs/hooke2/positive_singlet_tail_eval.yaml"
    )


def test_native_tail_train_checkpoint_eval_has_finite_local_energy(
    tmp_path: Path, monkeypatch,
) -> None:
    """A tiny native tail train feeds a strict native eval and finite energy."""

    rows = _phase2_rows(
        config_variant="positive_singlet_tail",
        results_root=str(tmp_path / "results"),
    )
    train = next(row for row in rows if row["kind"] == "train")
    evaluation = next(row for row in rows if row["kind"] == "eval" and row["checkpoint"] == 3)
    monkeypatch.setattr(stage1_train.sys, "argv", [
        "stage1-train", "--binding", _binding_token(train),
    ])
    assert stage1_train.main() == 0

    checkpoint_file = Path(str(evaluation["checkpoint_file"]))
    assert checkpoint_file.is_file()
    evaluation["checkpoint_sha256"] = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    evaluation["row_id"] = manifest_hash([
        {key: value for key, value in evaluation.items() if key != "row_id"}
    ])
    monkeypatch.setattr(stage1_eval.sys, "argv", [
        "stage1-eval", "--binding", _binding_token(evaluation),
    ])
    assert stage1_eval.main() == 0

    table = resolve_run_dir(evaluation) / "mcmc_energy" / "sampled_eval_table.csv"
    with table.open(newline="", encoding="utf-8") as handle:
        energies = [float(row["local_energy"]) for row in csv.DictReader(handle)]
    assert energies and all(math.isfinite(value) for value in energies)


def test_generic_pfaffian_eval_selection_is_unchanged() -> None:
    """The default Hooke-2 row continues to select the Pfaffian eval graph."""

    rows = _phase2_rows(config_variant=None)
    evaluation = next(row for row in rows if row["kind"] == "eval")
    assert evaluation["config"] == "configs/hooke2/eval.yaml"
    generic = load_config(str(HOOKE2 / "eval.yaml"))
    assert generic.model.embedding.include_spins is True
    assert generic.model.readout._target_ == "tpen.nn.readout.PfaffianReadout"
