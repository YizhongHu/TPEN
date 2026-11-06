"""Frozen study identities and measurement wiring, run on the cluster."""

import json

import numpy as np
import pytest
import yaml

from experiments.one_electron_variance.analysis import Profile, classify_lineage, load_profile
from experiments.one_electron_variance.protocol import (
    CHECKPOINTS, evaluation_config, evaluation_seed, prepare, train_config,
)


@pytest.mark.parametrize("system,energy", [("hooke", .75), ("hydrogen", -.5)])
def test_fixed_protocol_and_strict_restore_model_identity(tmp_path, system, energy):
    cfg = train_config(system, 1, tmp_path)
    assert cfg["trainer"]["max_steps"] == 50000
    assert cfg["system"]["n_particles"] == 1
    assert cfg["system"]["reference_energy"] == energy
    assert cfg["runtime"]["seed"] == 73002
    assert cfg["sampler"]["seed"] == 83002
    assert cfg["sampler"]["n_walkers"] == 4096
    assert cfg["model"]["readout"]["eps"] == 1e-12
    assert cfg["model"]["readout"]["trainable"] is True
    assert cfg["model"]["basis"] is None
    assert cfg["model"]["embedding"]["out_channels"] == 32
    assert cfg["model"]["embedding"]["hidden_channels"] == 128
    assert "ElectronElectron" not in yaml.safe_dump(cfg)
    for step in CHECKPOINTS:
        for stream in (0, 1):
            evaluation = evaluation_config(system, 1, step, stream, tmp_path)
            assert evaluation["model"] == cfg["model"]
            assert evaluation["hamiltonian_terms"] == cfg["hamiltonian_terms"]
            assert evaluation["load"]["strict"] is True
            assert evaluation["sampler"]["burn_in"] == 1000
            assert evaluation["sampler"]["n_steps"] == 20
            task = evaluation["evaluator"]["tasks"][0]
            assert task["generator"]["n_draws"] == 64
            assert task["summaries"][0]["max_samples"] == 262144


def test_seed_table_and_plan_are_complete_without_attempts(tmp_path):
    seeds = {evaluation_seed(system, j, step, b) for system in ("hooke", "hydrogen")
             for j in range(3) for step in CHECKPOINTS for b in (0, 1)}
    assert len(seeds) == 108
    assert seeds.isdisjoint({73001, 73002, 73003, 83001, 83002, 83003})
    train, evaluation = prepare(tmp_path, "a" * 40)
    assert (train.n_tasks, evaluation.n_tasks) == (6, 108)
    assert len({task.logical_task_id for plan in (train, evaluation) for task in plan.tasks}) == 114
    manifest = json.loads((tmp_path / "protocol.json").read_text())
    assert manifest["admission"] == "not_admitted"
    assert len(manifest["configs"]) == 114
    with pytest.raises(FileExistsError):
        prepare(tmp_path, "a" * 40)


def _synthetic(*, rise=True, drift=0.):
    # Equal signs in every half-prefix keep the physical mean fixed exactly.
    signs = np.tile(np.array([-1., 1.])[:, None], (32, 32))
    profiles = {}
    for step in CHECKPOINTS:
        variance = 2e-4 if step <= 5000 else 1e-5
        late = step >= 40000
        if late and rise:
            variance = 8e-4
        for stream in (0, 1):
            profiles[step, stream] = Profile(.75 + (drift if late else 0.) + np.sqrt(variance) * signs)
    return profiles


def test_classifier_detects_constant_mean_fall_then_eighty_fold_rise(tmp_path):
    result = classify_lineage(_synthetic(), exact_energy=.75, floor=1e-10, bootstrap_seed=42,
                              energy_uncertainty_adequate=True, training_adequate=True)
    assert result["outcome"] == "full_signature"
    assert result["streams"][0]["ratio_50000_over_25000"] == pytest.approx(80)
    (tmp_path / "synthetic-positive.json").write_text(json.dumps(result, indent=2))


def test_classifier_distinguishes_null_from_energy_drift(tmp_path):
    null = classify_lineage(_synthetic(rise=False), exact_energy=.75, floor=1e-10, bootstrap_seed=42,
                            energy_uncertainty_adequate=True, training_adequate=True)
    drift = classify_lineage(_synthetic(drift=.02), exact_energy=.75, floor=1e-10, bootstrap_seed=42)
    assert null["outcome"] == "conditional_finite_budget_null"
    assert drift["outcome"] == "variance_recurrence_energy_unresolved"
    (tmp_path / "synthetic-negative.json").write_text(json.dumps(null, indent=2))


def test_energy_equivalence_does_not_silently_assume_tail_adequacy():
    result = classify_lineage(_synthetic(), exact_energy=.75, floor=1e-10, bootstrap_seed=42)
    assert result["outcome"] == "variance_recurrence_energy_unresolved"
    result = classify_lineage(_synthetic(rise=False), exact_energy=.75, floor=1e-10, bootstrap_seed=42)
    assert result["outcome"] == "inconclusive"


def test_classifier_refuses_missing_profiles_nonfinite_or_unequal_budget():
    profiles = _synthetic()
    profiles.pop((50000, 1))
    with pytest.raises(ValueError, match="nine checkpoints"):
        classify_lineage(profiles, exact_energy=.75, floor=1e-10, bootstrap_seed=42)
    with pytest.raises(ValueError, match="finite complete"):
        Profile(np.array([[1., 2.], [3., np.nan]]))
    profiles = _synthetic()
    profiles[50000, 1] = Profile(np.ones((32, 32)))
    with pytest.raises(ValueError, match="identical evaluation budgets"):
        classify_lineage(profiles, exact_energy=.75, floor=1e-10, bootstrap_seed=42)


def test_science_collector_rejects_final_draw_or_selected_coordinate_receipt(tmp_path):
    (tmp_path / "sampled_eval_table.metadata.json").write_text(json.dumps({
        "schema": "trajectory_records/v1", "row_semantics": "selected_coordinates",
    }))
    with pytest.raises(ValueError, match="complete-trajectory"):
        load_profile(tmp_path)
