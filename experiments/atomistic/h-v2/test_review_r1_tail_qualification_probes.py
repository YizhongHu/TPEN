"""Review r1 (PR 600, TO item 028867dc) green controls for the sampler tail
qualification: config/contract pinning, min-resamples pinning, and
independent-integration checks of the radial and local-energy tail oracle
functions in the exact gated regime, computed without calling the oracle's
own CDF/survival machinery.

**Experiment boundary.** No `tpen` import anywhere in this file's own source
(not even transitively at module scope): `test_sampler_tail_qualification.py`
is loaded by path, exactly as it loads `analytic_oracle`, so its `from
tpen.run import run_from_config` does execute when that module is loaded --
`run_from_config` is the one import the experiment boundary allows -- and its
`pytestmark` only marks its own collected tests.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "configs" / "sampler_tail_qualification_eval.yaml"
CONTRACT_PATH = HERE / "contract_inputs.json"


def _module(name: str, filename: str | None = None):
    spec = importlib.util.spec_from_file_location(
        f"h_v2_review_r1_tail_probes_{name}", HERE / (filename or f"{name}.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = _module("analytic_oracle")
tail_qualification = _module("tail_qualification", "test_sampler_tail_qualification.py")

_CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
_TAIL = _CONTRACT["tail_characterization"]["sampler_tail_qualification"]
_SHAPE = _TAIL["shape_and_seeds"]["fresh_evaluation_sampler"]
_SEEDS = _CONTRACT["measurement"]["seed_roster"]["non_science_seed_namespaces"]["sampler_tail_qualification"]


def test_config_shape_matches_contract() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["sampler"]["n_walkers"] == _SHAPE["walkers"]
    assert config["sampler"]["burn_in"] == _SHAPE["burn_in"]
    assert config["sampler"]["n_steps"] == _SHAPE["stride"]
    generator = config["evaluator"]["tasks"][0]["generator"]
    assert generator["n_draws"] == _SHAPE["draws_per_stream"]
    assert generator["discard_draws"] == 0
    assert config["runtime"]["seed"] in _SEEDS["sampler"] + _SEEDS["evaluation"]

    assert tail_qualification.ALPHA_SEEDS == {1.0: _SEEDS["sampler"], 1.2: _SEEDS["evaluation"]}
    all_seeds = _SEEDS["sampler"] + _SEEDS["evaluation"]
    assert len(set(all_seeds)) == len(all_seeds) == 4


def test_min_resamples_matches_contract() -> None:
    assert (
        tail_qualification.MIN_RESAMPLES
        == _CONTRACT["measurement"]["fresh_evaluation"]["uncertainty"]["minimum_resamples"]
    )


@pytest.mark.parametrize("alpha", [1.0, 1.2])
@pytest.mark.parametrize("level", [0.001, 0.0001])
def test_radial_quantiles_match_independent_integration_in_the_gated_tails(alpha: float, level: float) -> None:
    rate = 2.0 * alpha

    def pdf(radius: float) -> float:
        return (rate**3) * radius * radius * math.exp(-rate * radius) / 2.0

    def simpson(low: float, high: float, n: int) -> float:
        h = (high - low) / n
        total = pdf(low) + pdf(high)
        for i in range(1, n):
            total += pdf(low + i * h) * (4.0 if i % 2 else 2.0)
        return total * h / 3.0

    r_lo = oracle.radial_lower_tail_quantile(level, alpha)
    assert simpson(0.0, r_lo, 20000) == pytest.approx(level, rel=1e-4)

    r_hi = oracle.radial_upper_tail_quantile(level, alpha)
    assert 1.0 - simpson(0.0, r_hi, 200000) == pytest.approx(level, rel=1e-3)


@pytest.mark.parametrize(
    "threshold_or_level,is_level",
    [(0.5, False), (1.0, False), (0.001, True), (0.0001, True)],
)
def test_exceedance_probability_matches_independent_integration_in_the_tail(
    threshold_or_level: float, is_level: bool
) -> None:
    alpha = 1.2
    rate = 2.0 * alpha

    def pdf(radius: float) -> float:
        return (rate**3) * radius * radius * math.exp(-rate * radius) / 2.0

    def cdf_bisect(probability: float) -> float:
        low, high = 0.0, 1.0 / alpha
        n = 20000

        def simpson_cdf(r: float) -> float:
            if r <= 0.0:
                return 0.0
            h = r / n
            total = pdf(0.0) + pdf(r)
            for i in range(1, n):
                total += pdf(i * h) * (4.0 if i % 2 else 2.0)
            return total * h / 3.0

        while simpson_cdf(high) < probability:
            high *= 2.0
        for _ in range(100):
            mid = (low + high) / 2.0
            if simpson_cdf(mid) < probability:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    median_radius = cdf_bisect(0.5)
    median_energy = -0.72 + 0.2 / median_radius

    def energy(radius: float) -> float:
        return -0.72 + 0.2 / radius

    threshold = oracle.energy_exceedance_threshold(threshold_or_level, alpha) if is_level else threshold_or_level

    # E(r) = -0.72 + 0.2/r is monotone decreasing in r; the exceedance region
    # |E - median_energy| > threshold is two radial intervals bounded by the
    # two crossings of E(r) = median_energy +/- threshold.
    lower_crossing = 0.2 / (median_energy + threshold + 0.72)
    upper_inverse = median_energy - threshold + 0.72
    upper_crossing = 0.2 / upper_inverse if upper_inverse > 0.0 else math.inf

    def simpson_interval(low: float, high: float, n: int = 200000) -> float:
        h = (high - low) / n
        total = pdf(low) + pdf(high)
        for i in range(1, n):
            total += pdf(low + i * h) * (4.0 if i % 2 else 2.0)
        return total * h / 3.0

    lower_mass = simpson_interval(0.0, lower_crossing) if lower_crossing > 0.0 else 0.0
    upper_mass = 0.0
    if math.isfinite(upper_crossing):
        upper_mass = simpson_interval(upper_crossing, upper_crossing + 200.0 / alpha)
    numeric_level = lower_mass + upper_mass

    exact_level = oracle.exceedance_probability(threshold, alpha)
    assert exact_level == pytest.approx(numeric_level, rel=1e-3)
