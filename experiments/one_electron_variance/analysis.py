"""Whole-trajectory endpoints for the preregistered one-electron experiment.

Every uncertainty resample retains all draws of a walker. Selected coordinates
and final-draw snapshots cannot enter the scientific collector. A valid finite
trajectory is not evidence that the population variance exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf

from experiments.one_electron_variance.protocol import (
    CHECKPOINTS, DESIGN_SHA256, EXACT_ENERGY, WINDOWS,
    evaluation_config, evaluation_seed, train_config,
)


@dataclass(frozen=True)
class Profile:
    """One complete draw-by-walker energy trajectory and its physical context."""

    energy: np.ndarray
    floor_count: int = 0
    physical_diagnostics: dict | None = None

    def __post_init__(self) -> None:
        if self.energy.ndim != 2 or min(self.energy.shape) < 2 or not np.isfinite(self.energy).all():
            raise ValueError("profile requires a finite complete draw-by-walker array")
        if not 0 <= self.floor_count <= self.energy.size:
            raise ValueError("invalid floor count")


def file_hash(path: Path) -> str:
    """Hash an artifact without loading all its bytes at once."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_profile(directory: Path) -> Profile:
    """Refuse missing, nonfinite, selected, reordered, or shortened science records."""
    path = directory / "sampled_eval_table.csv"
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    if (metadata["schema"] != "trajectory_records/v1"
            or metadata["row_semantics"] != "complete_draw_walker_grid"
            or (metadata["draw_count"], metadata["walker_count"], metadata["row_count"]) != (64, 4096, 262144)
            or metadata["n_electrons"] != 1 or metadata["spatial_dim"] != 3
            or metadata["nonfinite_count"] != 0
            or metadata["csv_sha256"] != file_hash(path)):
        raise ValueError("invalid complete-trajectory receipt")
    readout_path = directory / "raw_readout.csv"
    readout_receipt = json.loads(readout_path.with_suffix(".json").read_text())
    if (readout_receipt["row_count"] != 262144 or readout_receipt["readout_eps"] != 1e-12
            or readout_receipt["trajectory_sha256"] != metadata["csv_sha256"]
            or readout_receipt["csv_sha256"] != file_hash(readout_path)):
        raise ValueError("readout sidecar does not bind this complete trajectory")
    energies = np.empty((64, 4096), dtype=np.float64)
    radii, virials = [], []
    if metadata["term_names"] not in (["kinetic", "harmonic_trap"], ["kinetic", "electron_nucleus"]):
        raise ValueError("unexpected one-electron Hamiltonian")
    count = floor_count = 0
    with path.open(newline="") as handle, readout_path.open(newline="") as readout:
        for row, diagnostic in zip(csv.DictReader(handle), csv.DictReader(readout), strict=True):
            draw, walker = divmod(count, 4096)
            for record in (row, diagnostic):
                if (int(record["sample_index"]), int(record["draw_index"]), int(record["walker_index"])) != (count, draw, walker):
                    raise ValueError("records are not the complete ordered draw/walker grid")
            energy = float(row["local_energy"])
            terms = [float(row[f"term/{name}"]) for name in metadata["term_names"]]
            values = [energy, *terms, float(row["logabs"]), float(diagnostic["raw_channel_sum"]),
                      *(float(row[f"position/electron_0/axis_{a}"]) for a in range(3))]
            if (row["finite"] != "True" or not all(map(math.isfinite, values))
                    or float(row["sign"]) not in (-1., 1.)
                    or not math.isclose(energy, math.fsum(terms), rel_tol=1e-12, abs_tol=1e-10)):
                raise ValueError("nonfinite or inconsistent energy primitive")
            if count >= energies.size:
                raise ValueError("extra trajectory rows")
            energies[draw, walker] = energy
            radii.append(math.sqrt(sum(float(row[f"position/electron_0/axis_{a}"]) ** 2 for a in range(3))))
            virials.append(2 * terms[0] + (-2 if "harmonic_trap" in metadata["term_names"] else 1) * terms[1])
            active = float(diagnostic["raw_channel_sum"]) ** 2 < 1e-12
            if diagnostic["floor_active"] != str(active):
                raise ValueError("incorrect readout-floor flag")
            floor_count += active
            count += 1
    if count != energies.size or floor_count != readout_receipt["floor_count"]:
        raise ValueError("incomplete trajectory or floor receipt")
    if not math.isclose(float(energies.var()), metadata["variance"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("whole-trajectory variance does not reconcile")
    radius = np.asarray(radii)
    physical = {"mean_radius": float(radius.mean()), "mean_squared_radius": float(np.square(radius).mean()),
                "radial_cdf": {str(r): float(np.mean(radius <= r)) for r in (.1, .5, 1., 2., 4., 8.)},
                "virial_mean": float(np.mean(virials)), "virial_is_pointwise_oracle": False}
    return Profile(energies, floor_count, physical)


def describe(profile: Profile, exact_energy: float) -> dict:
    """Report raw and bounded tail statistics without removing any records."""
    values = profile.energy.reshape(-1)
    residual = values - exact_energy
    squared = (values - values.mean()) ** 2
    n_top = max(1, math.ceil(values.size / 100))
    total = float(squared.sum())
    prefixes = {}
    for fraction in (.25, .5, 1.):
        part = profile.energy[:max(1, int(fraction * profile.energy.shape[0]))]
        prefixes[str(fraction)] = {"mean": float(part.mean()), "variance": float(part.var()), "count": part.size}
    tail = {}
    for cutoff in (.1, 1., 10.):
        indicator = (np.abs(profile.energy - exact_energy) > cutoff).astype(float)
        centered = indicator - indicator.mean(axis=0, keepdims=True)
        denominator = float((centered * centered).sum())
        dependence = None if denominator == 0 else float((centered[:-1] * centered[1:]).sum() / denominator)
        tail[str(cutoff)] = {
            "positive_frequency": float(np.mean(residual > cutoff)),
            "negative_frequency": float(np.mean(residual < -cutoff)),
            "bounded_residual_second_moment": float(np.minimum(residual ** 2, cutoff ** 2).mean()),
            "lag_one_indicator_correlation": dependence,
        }
    return {"mean": float(values.mean()), "variance": float(values.var()), "count": values.size,
            "median_absolute_deviation": float(np.median(np.abs(values - np.median(values)))),
            "interquartile_range": float(np.quantile(values, .75) - np.quantile(values, .25)),
            "q99": float(np.quantile(values, .99)), "q999": float(np.quantile(values, .999)),
            "absolute_residual_q99": float(np.quantile(np.abs(residual), .99)),
            "absolute_residual_q999": float(np.quantile(np.abs(residual), .999)),
            "top_one_percent_squared_deviation_share": 0. if total == 0 else float(np.partition(squared, -n_top)[-n_top:].sum() / total),
            "prefixes": prefixes, "tails": tail, "floor_count": profile.floor_count,
            "physical_diagnostics": profile.physical_diagnostics}


def diagnostic_indices(profile: Profile, exact_energy: float) -> dict[str, list[int]]:
    """Select described diagnostic coordinates; these are never prevalence samples."""
    size = profile.energy.size
    evenly = np.linspace(0, size - 1, min(size, 32), dtype=int)
    # Stable index order breaks equal-magnitude ties reproducibly.
    largest = np.argsort(-np.abs(profile.energy.reshape(-1) - exact_energy), kind="stable")[:32]
    return {"evenly_indexed": evenly.tolist(), "largest_absolute_residual": largest.tolist()}


def _geometric_mean(values: list[float]) -> float:
    return 0. if any(value == 0 for value in values) else math.exp(math.fsum(map(math.log, values)) / len(values))


def _interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [.05, .95]).tolist()


def classify_lineage(profiles: dict[tuple[int, int], Profile], *, exact_energy: float,
                     floor: float, bootstrap_seed: int,
                     energy_uncertainty_adequate: bool = False, training_adequate: bool = False) -> dict:
    """Apply both-stream variance, energy-equivalence, and prefix requirements."""
    if set(profiles) != {(step, stream) for step in CHECKPOINTS for stream in (0, 1)}:
        raise ValueError("exactly nine checkpoints and two fresh streams are required")
    if not math.isfinite(floor) or floor < 1e-10:
        raise ValueError("floor must come from pre-training qualification")
    if len({profile.energy.shape for profile in profiles.values()}) != 1:
        raise ValueError("profiles must have identical evaluation budgets")
    rng = np.random.default_rng(bootstrap_seed)
    descriptions = {key: describe(profile, exact_energy) for key, profile in profiles.items()}
    streams = []
    stability = True
    for step in (*WINDOWS["middle"], 25000, *WINDOWS["late"]):
        v0, v1 = [descriptions[step, b]["variance"] for b in (0, 1)]
        stability &= max(v0, v1, floor) / max(min(v0, v1), floor) <= 2
        for b in (0, 1):
            record = descriptions[step, b]
            half = record["prefixes"]["0.5"]
            stability &= max(half["variance"], record["variance"], floor) / max(min(half["variance"], record["variance"]), floor) <= 2
            stability &= abs(half["mean"] - record["mean"]) < .005
    for b in (0, 1):
        variances = {name: _geometric_mean([descriptions[step, b]["variance"] for step in steps])
                     for name, steps in WINDOWS.items()}
        early, middle, late = [variances[name] for name in ("early", "middle", "late")]
        decrease = early / max(middle, floor)
        increase = late / max(middle, floor)
        sustained = sum(descriptions[step, b]["variance"] >= 10 * max(middle, floor) for step in WINDOWS["late"])
        signature = bool(early > floor and decrease >= 2 and increase >= 10 and late - middle >= 1e-4 and sustained >= 2)
        boot = {}
        for step in (*WINDOWS["middle"], 25000, *WINDOWS["late"]):
            # The mean of an entire walker trajectory is its bootstrap unit.
            means = profiles[step, b].energy.mean(axis=0)
            boot[step] = rng.choice(means, size=(1000, means.size), replace=True).mean(axis=1)
        middle_energy = np.mean([boot[step] for step in WINDOWS["middle"]], axis=0)
        late_energy = np.mean([boot[step] for step in WINDOWS["late"]], axis=0)
        intervals = {"middle_to_late": _interval(late_energy - middle_energy),
                     "25000_to_50000": _interval(boot[50000] - boot[25000]),
                     "middle_reference_error": _interval(middle_energy - exact_energy),
                     "late_reference_error": _interval(late_energy - exact_energy)}
        equivalence = all(low > -.005 and high < .005 for low, high in intervals.values())
        streams.append({"stream": b, "window_variances": variances, "early_decline_ratio": decrease,
                        "late_increase_ratio": increase, "late_absolute_increase": late - middle,
                        "late_checkpoint_count": sustained, "variance_signature": signature,
                        "already_at_floor": early <= floor,
                        "ratio_50000_over_25000": descriptions[50000, b]["variance"] / max(descriptions[25000, b]["variance"], floor),
                        "energy_intervals_90_percent": intervals, "energy_equivalence": equivalence})
    recurrence = all(row["variance_signature"] for row in streams)
    floor_seen = any(profile.floor_count for profile in profiles.values())
    energy_ok = all(row["energy_equivalence"] for row in streams) and stability and energy_uncertainty_adequate
    if recurrence:
        outcome = "full_signature" if energy_ok else "variance_recurrence_energy_unresolved"
    elif all(row["late_increase_ratio"] < 2 for row in streams) and energy_ok and training_adequate:
        outcome = "conditional_finite_budget_null"
    else:
        outcome = "inconclusive"
    # A floor event remains a measured stack diagnostic, not clean physical evidence.
    return {"outcome": outcome, "streams": streams, "prefix_and_stream_stability": bool(stability),
            "energy_uncertainty_adequate": energy_uncertainty_adequate,
            "training_adequate": training_adequate,
            "readout_floor_seen": floor_seen, "floor": floor,
            "population_moment_existence": "unresolved",
            "profiles": {f"{step}:{b}": row for (step, b), row in descriptions.items()}}


def _verify_run(directory: Path, source_sha: str) -> dict:
    if json.loads((directory / "status.json").read_text())["status"] != "completed":
        raise ValueError("run did not complete")
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["git_commit"] != source_sha or metadata["dirty_worktree"]:
        raise ValueError("run source differs from the qualified clean source")
    return yaml.safe_load((directory / "resolved_config.yaml").read_text())


def _verify_protocol_config(actual: dict, expected: dict, directory: Path) -> None:
    """Bind the consumed scientific settings to the frozen protocol."""
    expected["run"]["dir"] = str(directory)
    resolved = OmegaConf.to_container(OmegaConf.create(expected), resolve=True)
    for key in ("runtime", "system", "model", "sampler", "hamiltonian_terms",
                "optimizer", "trainer", "evaluator", "load"):
        if actual.get(key) != resolved.get(key):
            raise ValueError(f"consumed {key} differs from the frozen protocol")


def _verify_training(directory: Path, *, source_sha: str, lineage: int, floor: float,
                     expected_config: dict) -> bool:
    cfg = _verify_run(directory, source_sha)
    _verify_protocol_config(cfg, expected_config, directory)
    if (cfg["runtime"]["seed"] != 73001 + lineage or cfg["sampler"]["seed"] != 83001 + lineage
            or cfg["sampler"]["n_walkers"] != 4096 or cfg["trainer"]["max_steps"] != 50000):
        raise ValueError("training identity or budget differs from the protocol")
    for step in range(0, 50001, 2500):
        checkpoint = directory / "checkpoints" / f"step_{step:06d}"
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if (not (checkpoint / "COMPLETE").is_file() or manifest["completed_updates"] != step
                or manifest["next_iteration"] != step or manifest["provenance"]["git_sha"] != source_sha
                or manifest["provenance"]["git_dirty"] is not False):
            raise ValueError("missing or mislabeled completed-update checkpoint")
        for component, relative in manifest["files"].items():
            path = checkpoint / relative
            if not path.resolve().is_relative_to(checkpoint.resolve()):
                raise ValueError("checkpoint component escapes its directory")
            if file_hash(path) != manifest["hashes"][f"{component}_sha256"]:
                raise ValueError("checkpoint content does not match its manifest")
    count = 0
    initial_variance = None
    with (directory / "metrics.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("namespace") != "train":
                continue
            metrics = record["metrics"]
            if (record["step"] != count or metrics["optimizer_step"] is not True
                    or metrics["local_energy_n_finite"] != 4096 or metrics["local_energy_n_total"] != 4096):
                raise ValueError("training log contains a skipped, missing, or invalid update")
            if count == 0:
                initial_variance = metrics["energy_variance"]
            count += 1
    if count != 50000:
        raise ValueError("training log does not contain 50000 applied updates")
    activity = json.loads((directory / "parameter_activity.json").read_text())
    return bool(initial_variance > floor and any(row["gradient"] == "nonzero" for row in activity["parameters"]))


def collect(root: Path, *, qualification: Path, interpretation: Path | None = None) -> dict:
    """Collect the fixed six lineages only after an independently checked gate receipt."""
    gate = json.loads(qualification.read_text())
    if gate.get("status") != "passed" or gate.get("instrument_controls") != "passed":
        raise ValueError("a complete qualification receipt is required")
    protocol = json.loads((root / "protocol.json").read_text())
    if gate["source_sha"] != protocol["source_sha"] or protocol["design_sha256"] != DESIGN_SHA256:
        raise ValueError("qualification and frozen protocol source disagree")
    assessments = {} if interpretation is None else json.loads(interpretation.read_text())
    exact_variance = float(gate["max_exact_ordinary_grid_variance"])
    if not math.isfinite(exact_variance) or not 0 <= exact_variance <= 1e-10:
        raise ValueError("exact-reference variance did not pass qualification")
    floor = max(1e-10, 100 * exact_variance)
    results = {}
    for system, exact in EXACT_ENERGY.items():
        rows = []
        for lineage in range(3):
            profiles = {}
            train_dir = root / "one_electron_variance" / system / f"{system}-lineage{lineage}"
            try:
                adequate = _verify_training(
                    train_dir, source_sha=gate["source_sha"], lineage=lineage, floor=floor,
                    expected_config=train_config(system, lineage, root),
                )
                for step in CHECKPOINTS:
                    for stream in (0, 1):
                        directory = train_dir.with_name(f"{system}-lineage{lineage}-update{step:06d}-stream{stream}")
                        cfg = _verify_run(directory, gate["source_sha"])
                        _verify_protocol_config(
                            cfg, evaluation_config(system, lineage, step, stream, root), directory,
                        )
                        if (cfg["sampler"]["seed"] != evaluation_seed(system, lineage, step, stream)
                                or cfg["load"]["path"] != str(train_dir / "checkpoints" / f"step_{step:06d}")
                                or cfg["load"]["strict"] is not True):
                            raise ValueError("evaluation identity or restore differs from protocol")
                        profiles[step, stream] = load_profile(directory / "trajectory")
                assessment = assessments.get(f"{system}:{lineage}", {})
                accepted = assessment.get("energy_uncertainty_adequate") is True and bool(assessment.get("rationale"))
                result = classify_lineage(profiles, exact_energy=exact, floor=floor,
                                          bootstrap_seed=2_000_000 + 100 * lineage + (system == "hydrogen"),
                                          energy_uncertainty_adequate=accepted, training_adequate=adequate)
                result["interpretation"] = assessment
                result["diagnostic_indices_not_prevalence_samples"] = {
                    f"{step}:{stream}": diagnostic_indices(profiles[step, stream], exact)
                    for step in (25000, 50000) for stream in (0, 1)}
                rows.append(result)
            except (ValueError, KeyError, OSError) as exc:
                rows.append({"outcome": "incomplete", "reason": str(exc)})
        results[system] = {"lineages": rows, "replicated_full_signature": sum(row["outcome"] == "full_signature" for row in rows) >= 2,
                           "all_three_conditional_null": all(row["outcome"] == "conditional_finite_budget_null" for row in rows)}
    return {"qualification_sha256": file_hash(qualification),
            "interpretation_sha256": None if interpretation is None else file_hash(interpretation), "systems": results,
            "estimand": "equal_budget_whole_trajectory_variance_under_each_learned_Born_target"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--interpretation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root, qualification=args.qualification, interpretation=args.interpretation)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
