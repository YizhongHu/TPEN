"""Sampler tail qualification: full radial and local-energy survival curves.

Compares the real sampler, driven through the exact-hydrogen evaluation
route (`tpen.run.run_from_config` -> `tpen.physics.hydrogen.HydrogenExact`
-> the real `Evaluate` runner, `TrajectoryMCMCGenerator`, calculators,
summaries, and trajectory record writer -- the same route as
`test_hydrogen_exact_canary.py`), against the independent analytic oracle
(`analytic_oracle.py`) over the FULL radial and local-energy survival
curves, both tails, rather than the two-percentile/``r<=3`` comparison this
layer replaces (`contract_inputs.json`
``tail_characterization.sampler_tail_qualification.superseded_stub``): a
0.02-band CDF check up to ``r=3`` (about the 94th percentile of the outer
radius) never examines the inner radial tail, which is exactly where the
alpha=1.2 local-energy tail (``E_L = -0.72 + 0.2/r``) comes from.

Targets (``contract_inputs.json``
``tail_characterization.sampler_tail_qualification``):

- alpha=1: the exact ground state. Every local energy is exactly -0.5 Ha, so
  this stream is a sampler-mechanics instrument control -- only its radial
  tail is informative.
- alpha=1.2: the known-imperfect qualification target. Both its radial and
  local-energy tails are gated.

Band. Every gated comparison -- a radial lower-tail CDF level, a radial
upper-tail survival level, a fixed local-energy exceedance threshold, or a
local-energy exceedance level -- picks its threshold so that the ORACLE's
own exact survival probability at that threshold equals the declared level
exactly, by construction (``oracle.radial_lower_tail_quantile`` and friends
are the exact inverses of the oracle's own CDF/survival functions, each unit
tested against independent numeric integration in ``test_protocol.py``).  So
``S_exact == level`` always, and the per-comparison statistic is::

    z = (log(S_hat) - log(S_exact)) / se

where ``S_hat`` is the empirical fraction of the stream's 4096x256
draw-by-walker grid that falls in the gated event, and ``se`` is the
whole-walker (not per-record) bootstrap standard error of ``log(S_hat)``:
walkers, not individual draw records, are resampled with replacement, since
draws within one MCMC walker are correlated. A gated level with a zero
empirical count fails outright (``log(0)`` is undefined). The qualification
passes only if ``max(|z|)`` over every gated comparison, in both streams, is
at most ``c`` -- the contract's own precomputed Bonferroni critical value
(``band.c_value``), which already accounts for the family of ``K`` gated
comparisons (``band.k_value``).

Support rule: a level gates only when its EXACT expected count
(``level * 4096 * 256``) is at least 100; every level in the contract's
``radial_levels``/``alpha_1_2_only`` lists already clears this (the
deepest, 0.0001, has an expected count of about 105 --
``support_rule.deepest_gated_level_expected_count``), so no level is
dropped here, but the check is applied rather than assumed.

K cross-check: this test's own measured comparison count (``k_measured`` in
the printed report) matched the contract's declared ``band.k_value`` of 122
on the Cannon qualification run (job 58808266, 2026-11-07). The gate below
uses the contract's literal ``c_value`` directly, never recomputing it.

Seeds. `contract_inputs.json`'s `sampler_tail_qualification` seed namespace
provides two `sampler` seeds and two `evaluation` seeds; this test assigns
the `sampler` pair to the alpha=1 instrument-control streams (sampler
mechanics only -- no interesting energy target) and the `evaluation` pair to
the alpha=1.2 qualification-target streams, consistent with the other
namespaces' naming (e.g. `pipeline_cpu_learned_e`, where `evaluation` is
likewise the fresh-evaluation stream pair of the scientifically interesting
run).

**Experiment boundary.** Reaches core only through configuration and
`tpen.run.run_from_config`, per the operator ruling enforced by
`test_import_boundary.py`.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

from tpen.run import run_from_config

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "configs" / "sampler_tail_qualification_eval.yaml"
INPUTS_PATH = HERE / "contract_inputs.json"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"h_v2_tail_qualification_{name}", HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = _module("analytic_oracle")

_CONTRACT = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
_TAIL = _CONTRACT["tail_characterization"]["sampler_tail_qualification"]
_SHAPE = _TAIL["shape_and_seeds"]["fresh_evaluation_sampler"]
_SEEDS = _CONTRACT["measurement"]["seed_roster"]["non_science_seed_namespaces"]["sampler_tail_qualification"]
_BAND = _TAIL["band"]

N_WALKERS = _SHAPE["walkers"]
N_DRAWS = _SHAPE["draws_per_stream"]
N_STREAMS = _SHAPE["streams"]
N_ROWS_PER_STREAM = N_WALKERS * N_DRAWS

RADIAL_LEVELS = _TAIL["radial_levels"]
MIN_COUNT = _TAIL["support_rule"]["minimum_expected_count_per_stream"]
FIXED_THRESHOLDS = _TAIL["alpha_1_2_only"]["fixed_exceedance_thresholds_hartree"]
EXCEEDANCE_LEVELS = _TAIL["alpha_1_2_only"]["exceedance_levels"]
K_CONTRACT = _BAND["k_value"]
C_VALUE = _BAND["c_value"]
MIN_RESAMPLES = _CONTRACT["measurement"]["fresh_evaluation"]["uncertainty"]["minimum_resamples"]

#: `sampler` namespace seeds drive the alpha=1 instrument-control streams;
#: `evaluation` namespace seeds drive the alpha=1.2 qualification target.
ALPHA_SEEDS = {1.0: _SEEDS["sampler"], 1.2: _SEEDS["evaluation"]}

#: Internal bootstrap-resampling RNG seed. Not a physics seed (no sampler,
#: training, or evaluation draw depends on it) and not part of the
#: `sampler_tail_qualification` roster namespace; it only orders which
#: walkers a given resample happens to pick.
_BOOTSTRAP_SEED = 20261108


def _load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _run_stream(tmp_path: Path, *, alpha: float, stream: int) -> dict[str, np.ndarray]:
    seed = ALPHA_SEEDS[alpha][stream]
    config = _load_config()
    config["model"]["alpha"] = alpha
    config["runtime"]["seed"] = seed
    run_id = f"tail-qual-alpha{alpha}-stream{stream}"
    config["run"]["root"] = str(tmp_path)
    config["run"]["run_id"] = run_id

    assert config["sampler"]["n_walkers"] == _SHAPE["walkers"]
    assert config["sampler"]["burn_in"] == _SHAPE["burn_in"]
    assert config["sampler"]["n_steps"] == _SHAPE["stride"]
    generator_config = config["evaluator"]["tasks"][0]["generator"]
    assert generator_config["n_draws"] == _SHAPE["draws_per_stream"]

    status = run_from_config(
        OmegaConf.create(config), command=f"h-v2 sampler tail qualification {run_id}", raise_exceptions=True
    )
    assert status == 0

    run_dir = Path(config["run"]["root"]) / config["experiment"]["name"] / config["experiment"]["sector"] / run_id
    with (run_dir / "trajectory" / "sampled_eval_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == N_ROWS_PER_STREAM

    radii = np.empty(len(rows), dtype=np.float64)
    energies = np.empty(len(rows), dtype=np.float64)
    walkers = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        x = float(row["position/electron_0/axis_0"])
        y = float(row["position/electron_0/axis_1"])
        z = float(row["position/electron_0/axis_2"])
        radii[index] = math.sqrt(x * x + y * y + z * z)
        energies[index] = float(row["local_energy"])
        walkers[index] = int(row["walker_index"])
        assert row["finite"] == "True"

    return {"radii": radii, "energies": energies, "walkers": walkers}


def _gated_z(indicator: np.ndarray, walkers: np.ndarray, exact_level: float, *, rng: np.random.Generator) -> tuple[float, int]:
    """Return ``(z, hat_count)`` for one gated comparison; `z = inf` on a zero count."""

    hat_count = int(indicator.sum())
    if hat_count == 0:
        return math.inf, 0
    n = indicator.size
    s_hat = hat_count / n
    counts_per_walker = np.bincount(walkers, weights=indicator.astype(np.int64), minlength=N_WALKERS)
    resample_indices = rng.integers(0, N_WALKERS, size=(MIN_RESAMPLES, N_WALKERS))
    resample_counts = counts_per_walker[resample_indices].sum(axis=1)
    valid = resample_counts > 0
    if int(valid.sum()) < 2:
        return math.inf, hat_count
    log_boot = np.log(resample_counts[valid] / n)
    se = float(np.std(log_boot, ddof=1))
    if se == 0.0:
        return math.inf, hat_count
    z = (math.log(s_hat) - math.log(exact_level)) / se
    return z, hat_count


def _energy_exceedance_resample_counts(
    energies: np.ndarray, walkers: np.ndarray, resample_indices: np.ndarray, thresholds: list[float]
) -> np.ndarray:
    """Weighted-exceedance counts per whole-walker resample, median recomputed each time.

    `band.se_method` (``contract_inputs.json``) requires the median to be
    recomputed inside every whole-walker resample, not fixed once on the
    full-sample median (see PR 600 review round 1, finding F1). Returns an
    ``(R, len(thresholds))`` matrix so one shared resample matrix, built once
    per stream, can be reused by every energy comparison at that stream.
    """

    order = np.argsort(energies, kind="stable")
    e_sorted = energies[order]
    walkers_sorted = walkers[order]
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    r = resample_indices.shape[0]
    counts = np.empty((r, thresholds_arr.size), dtype=np.float64)
    for b in range(r):
        multiplicity = np.bincount(resample_indices[b], minlength=N_WALKERS).astype(np.float64)
        weight_sorted = multiplicity[walkers_sorted]
        cumulative = np.cumsum(weight_sorted)
        total = cumulative[-1]
        # Lower weighted median (first order statistic reaching half the total
        # weight), not the two-middle-values average `S_hat`'s np.median uses;
        # at N=1,048,576 the two conventions differ by one order-statistic
        # spacing, immaterial for se (round-2 review, finding G4).
        median_index = int(np.searchsorted(cumulative, total / 2.0, side="left"))
        median_index = min(median_index, e_sorted.size - 1)
        median_b = e_sorted[median_index]
        deviation_sorted = np.abs(e_sorted - median_b)
        mask = deviation_sorted[:, None] > thresholds_arr[None, :]
        counts[b] = weight_sorted @ mask
    return counts


def _se_from_resample_counts(counts_column: np.ndarray, n: int) -> float:
    valid = counts_column > 0
    if int(valid.sum()) < 2:
        return math.inf
    log_boot = np.log(counts_column[valid] / n)
    se = float(np.std(log_boot, ddof=1))
    return se if se > 0.0 else math.inf


def test_energy_exceedance_resample_counts_recomputes_the_median_per_resample() -> None:
    """Hand-computed check that the median is recomputed inside each resample.

    Two walkers, one draw each: energies 0.0 and 10.0. Three resamples pick
    different walker multiplicities, so the weighted median swings between
    0.0 and 10.0 depending on the resample -- exactly what ``band.se_method``
    requires and what the fixed-median indicator (the reverted mutant this
    test must catch) cannot reproduce, since a *single* full-sample median
    (``np.median([0.0, 10.0]) == 5.0``) puts every deviation at the gated
    threshold of 5.0 and reports a count of 0 in every resample.
    """

    energies = np.array([0.0, 10.0])
    walkers = np.array([0, 1])
    resample_indices = np.array([[0, 1], [0, 0], [1, 1]])
    thresholds = [5.0]

    counts = _energy_exceedance_resample_counts(energies, walkers, resample_indices, thresholds)

    assert counts[:, 0].tolist() == [1.0, 0.0, 0.0]

    sample_median = float(np.median(energies))
    fixed_indicator = np.abs(energies - sample_median) > thresholds[0]
    fixed_counts_per_resample = [
        float(np.bincount(resample_indices[b], weights=fixed_indicator.astype(np.int64), minlength=2).sum())
        for b in range(resample_indices.shape[0])
    ]
    assert fixed_counts_per_resample == [0.0, 0.0, 0.0]
    assert counts[:, 0].tolist() != fixed_counts_per_resample


@pytest.mark.h_v2_sampler_tail_qualification
def test_sampler_reproduces_exact_radial_and_local_energy_tail_curves(tmp_path_factory: pytest.TempPathFactory) -> None:
    start = time.monotonic()
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    comparisons: list[dict[str, Any]] = []
    medians: list[dict[str, Any]] = []

    for alpha in (1.0, 1.2):
        for stream in range(N_STREAMS):
            tmp_path = tmp_path_factory.mktemp(f"tail-qual-a{alpha}-s{stream}")
            data = _run_stream(tmp_path, alpha=alpha, stream=stream)
            radii = data["radii"]
            walkers = data["walkers"]
            n = radii.size

            medians.append({"alpha": alpha, "stream": stream, "radial_median": float(np.median(radii))})

            for level in RADIAL_LEVELS:
                if level * n < MIN_COUNT:
                    continue
                lower_threshold = oracle.radial_lower_tail_quantile(level, alpha)
                z, count = _gated_z(radii <= lower_threshold, walkers, level, rng=rng)
                comparisons.append(
                    {"alpha": alpha, "stream": stream, "side": "lower_tail_cdf", "level": level, "count": count, "z": z}
                )

                upper_threshold = oracle.radial_upper_tail_quantile(level, alpha)
                z, count = _gated_z(radii > upper_threshold, walkers, level, rng=rng)
                comparisons.append(
                    {"alpha": alpha, "stream": stream, "side": "upper_tail_survival", "level": level, "count": count, "z": z}
                )

            if alpha != 1.2:
                continue

            energies = data["energies"]
            sample_median_energy = float(np.median(energies))
            medians.append({"alpha": alpha, "stream": stream, "energy_median": sample_median_energy})
            deviation = np.abs(energies - sample_median_energy)

            gated_specs: list[dict[str, Any]] = []
            for threshold in FIXED_THRESHOLDS:
                exact_level = oracle.exceedance_probability(threshold, alpha)
                if exact_level * n < MIN_COUNT:
                    continue
                gated_specs.append(
                    {
                        "side": "fixed_exceedance_threshold_hartree",
                        "threshold_hartree": threshold,
                        "exact_level": exact_level,
                        "extra": {"threshold_hartree": threshold, "exact_level": exact_level},
                    }
                )

            for level in EXCEEDANCE_LEVELS:
                if level * n < MIN_COUNT:
                    continue
                threshold = oracle.energy_exceedance_threshold(level, alpha)
                gated_specs.append(
                    {
                        "side": "exceedance_level",
                        "threshold_hartree": threshold,
                        "exact_level": level,
                        "extra": {"level": level},
                    }
                )

            # One shared whole-walker resample matrix per stream: band.se_method
            # requires the median recomputed inside every resample, so all
            # energy comparisons at this stream reuse the same resample draw
            # (finding F1).
            resample_indices = rng.integers(0, N_WALKERS, size=(MIN_RESAMPLES, N_WALKERS))
            resample_counts = _energy_exceedance_resample_counts(
                energies, walkers, resample_indices, [spec["threshold_hartree"] for spec in gated_specs]
            )
            for column, spec in enumerate(gated_specs):
                indicator = deviation > spec["threshold_hartree"]
                count = int(indicator.sum())
                if count == 0:
                    z = math.inf
                else:
                    se = _se_from_resample_counts(resample_counts[:, column], n)
                    z = math.inf if not math.isfinite(se) else (math.log(count / n) - math.log(spec["exact_level"])) / se
                comparisons.append({"alpha": alpha, "stream": stream, "side": spec["side"], "count": count, "z": z, **spec["extra"]})

    wall_time_seconds = time.monotonic() - start
    max_abs_z = max(abs(entry["z"]) for entry in comparisons)
    failing = [entry for entry in comparisons if entry["count"] == 0 or abs(entry["z"]) > C_VALUE]

    report = {
        "k_contract": K_CONTRACT,
        "k_measured": len(comparisons),
        "c": C_VALUE,
        "max_abs_z": max_abs_z,
        "wall_time_seconds": wall_time_seconds,
        "medians_reported_not_gated": medians,
        "comparisons": comparisons,
    }
    print(json.dumps(report, indent=2, default=str))

    assert len(comparisons) == K_CONTRACT, (
        f"measured K={len(comparisons)} != contract band.k_value={K_CONTRACT}; c_value is only valid for the contract's K"
    )
    assert not failing, (
        f"{len(failing)} of {len(comparisons)} gated comparison(s) exceeded c={C_VALUE} "
        f"or had a zero count: {failing}"
    )
