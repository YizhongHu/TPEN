"""The checked-in scan grids: expansion, blinding, overrides, seeds, selection.

``configs/grid.yaml`` and ``configs/smoke.yaml`` are the only place the study's
288 jobs, its seed policy, and its champion selector are written down. Every test
here guards a failure that would produce a green scan answering a different
question than the one asked, and several of them exist because the failure has
already happened once somewhere in this stack:

expansion
    96 configs x 3 paired seed rows = 288 rows with unique ids. A duplicate id is
    two rows writing into one durable directory.

override paths
    ``load_config`` applies overrides with
    ``OmegaConf.merge(cfg, OmegaConf.from_dotlist(...))`` (``tpen/run.py:88-90``),
    which SILENTLY CREATES an unknown key. A mistyped axis, seed, or static
    override path therefore no-ops on every row with no error anywhere, so each
    one is checked to exist in the base config it targets.

blinding
    The basis library self-references its own levels by name while blinding rekeys
    it by slot. The round-trip through ``unblind.json`` is what makes a blinded
    plan recoverable.

seeds
    Nine scan seeds and three final blocks that must not collide. A collision
    would silently make a "fresh" final replicate a rerun of a scan row.

selection
    The primary metric must be the MCMC energy, ``min``, on the selection seed
    rows only. Selecting on the fixed-prior mean would rank on a non-variational
    quantity; letting the holdout seed vote would restore the winner's-curse bias
    that the split-sample exists to remove.

Nothing here imports ``tpen`` (``experiments/README.md``). The one property that
genuinely needs the production code -- that the smoke's terminal checkpoint is
written -- is asserted in ``tests/unit/experiments/test_scan_smoke_workload.py``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import pytest
from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
CONFIGS = STUDY_DIR / "configs"
REPO_ROOT = STUDY_DIR.parents[2]
GRID = CONFIGS / "grid.yaml"
SMOKE = CONFIGS / "smoke.yaml"
BASIS_LIBRARY = REPO_ROOT / "experiments" / "hooke" / "choices" / "basis_levels.yaml"

ATTEMPT = "20260813T120000-0400"

# The measured expansion of each checked-in grid. Hard-coded rather than derived
# from the grid's own axes: a test that recomputes the product from the file under
# test cannot detect an axis being added, dropped, or duplicated.
GRID_JOBS = 288
GRID_CONFIGS = 96
SMOKE_JOBS = 24
SMOKE_CONFIGS = 8

BASIS_LEVELS = ("no-basis", "hooke-axiswise-v1", "hooke-total-shell", "hooke-cartesian-box")

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))


def _load_script(name: str) -> ModuleType:
    """Load one study script under its own module name."""

    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tpen_pair_scan_v1_grid_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan = _load_script("plan")
select_champions = _load_script("select_champions")
# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

layout = sibling(__file__, 'utils.layout')
_tpen_utils_config = sibling(__file__, 'utils.config')
load_composed_config = _tpen_utils_config.load_composed_config


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan from the repo root: the grids name repo-relative config paths."""

    monkeypatch.chdir(REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _grid_data(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _plan(tmp_path: Path, grid: Path, *extra: str) -> Path:
    """Plan one grid into ``tmp_path`` and return the results root."""

    results_root = tmp_path / "results"
    code = plan.main(
        [
            "--grid",
            str(grid),
            "--results-root",
            str(results_root),
            "--attempt-id",
            ATTEMPT,
            *extra,
        ]
    )
    assert code == 0
    return results_root


def _manifest(results_root: Path) -> dict[str, Any]:
    return json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())


def _composed(stage_config: str) -> Any:
    """Return one base config composed with the basis library, as plan.py does."""

    return load_composed_config(
        REPO_ROOT / "experiments" / "hooke" / "tpen-pair-scan-v1" / "configs" / stage_config,
        [{"path": str(BASIS_LIBRARY), "provides": ["choices.basis"]}],
        required_paths=["choices.basis"],
        repo_root=REPO_ROOT,
    )


def _has_path(config: Any, dotted: str) -> bool:
    """Return whether a dotted path exists, WITHOUT resolving interpolations.

    Resolution is deliberately avoided: several base-config values interpolate
    ``run.dir``, which is null until a launcher sets it, so a resolving check
    would report "missing" for keys that are present.
    """

    node = OmegaConf.to_container(config, resolve=False)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# 1. Expansion
# ---------------------------------------------------------------------------
def test_the_full_grid_expands_to_288_rows_with_unique_run_ids(tmp_path: Path) -> None:
    manifest = _manifest(_plan(tmp_path, GRID))
    run_ids = [job["run_id"] for job in manifest["jobs"]]

    assert manifest["n_jobs"] == GRID_JOBS
    assert len(run_ids) == GRID_JOBS
    assert len(set(run_ids)) == GRID_JOBS
    # 96 configurations, each on the same three seed rows.
    assert len({job["config_id"] for job in manifest["jobs"]}) == GRID_CONFIGS
    assert len({job["major_id"] for job in manifest["jobs"]}) == len(BASIS_LEVELS)
    assert sorted({job["scan_seed"] for job in manifest["jobs"]}) == [0, 1, 2]


def test_the_smoke_grid_expands_smaller_but_keeps_every_least_proven_surface(tmp_path: Path) -> None:
    """Reduced in grid size only, and never in the surfaces a smoke exists to prove.

    All four basis levels: two of them threw at construction until
    ``include_gaussian_factor`` was written out, one was crippled by an off-by-one
    ``box_size``, and blinding has to reslot the library's self-references before
    any of them resolves inside an array task. The ``Gaussian`` activation: it is
    one of the two levels with ``Gamma(0) != 0``, which writes an invariant
    constant onto the tuple entries mixing never updates.
    """

    smoke = _grid_data(SMOKE)
    manifest = _manifest(_plan(tmp_path, SMOKE, "--no-blind"))

    assert manifest["n_jobs"] == SMOKE_JOBS
    assert len({job["run_id"] for job in manifest["jobs"]}) == SMOKE_JOBS
    assert len({job["config_id"] for job in manifest["jobs"]}) == SMOKE_CONFIGS
    assert tuple(smoke["major_grid"]["basis"]) == BASIS_LEVELS
    assert {job["major_choices"]["basis"] for job in manifest["jobs"]} == set(BASIS_LEVELS)
    assert "Gaussian" in {job["minor_choices"]["activation"] for job in manifest["jobs"]}
    assert smoke["scan_seed_rows"] == _grid_data(GRID)["scan_seed_rows"]


def test_both_grids_declare_the_same_stage_stack_and_differ_only_where_intended() -> None:
    """The smoke is the same pipeline: same configs, same stages, same selector.

    CLAUDE.md allows a smoke to reduce grid size and explicitly-requested scale
    controls, and nothing else. Anything else that differs -- a config path, a
    seed policy, a selector -- means the smoke stopped rehearsing the real run.
    """

    grid = _grid_data(GRID)
    smoke = _grid_data(SMOKE)
    shared = (
        "study",
        "config",
        "validation_config",
        "results_root",
        "choice_libraries",
        "config_snapshots",
        "scan_seed_axis",
        "scan_seed_rows",
        "blinding",
        "axis_id_labels",
        "axis_overrides",
        "choice_validation",
        "seed_overrides",
        "final_seed_sequences",
        "champions",
        "champion_reference_metrics",
    )
    for key in shared:
        assert smoke[key] == grid[key], key
    # The only permitted differences.
    assert smoke["final_replicates"] == 1
    assert grid["final_replicates"] == 9
    assert "static_overrides" in smoke
    assert "static_overrides" not in grid


def test_the_full_grid_plans_one_champion_per_basis_bucket_and_nine_replicates() -> None:
    """4 buckets x 1 champion kind x 9 replicates = 36 final train + 36 final eval."""

    grid = _grid_data(GRID)
    buckets = len(grid["major_grid"]["basis"])
    kinds = len(grid["champions"])

    assert buckets == 4
    assert kinds == 1
    assert buckets * kinds * grid["final_replicates"] == 36


# ---------------------------------------------------------------------------
# 2. Blinding
# ---------------------------------------------------------------------------
def test_blinding_round_trips_through_unblind_json(tmp_path: Path) -> None:
    """A blinded plan is recoverable, and carries no semantic basis value anywhere.

    Blinding is enabled by default for this study, so the artifact that maps slots
    back to levels is the only route from a planned row to the arm it ran.
    """

    results_root = _plan(tmp_path, GRID)
    manifest = _manifest(results_root)
    unblind = json.loads((results_root / "00_grid" / ATTEMPT / "unblind.json").read_text())
    axis = unblind["axes"]["basis"]
    slot_to_value = axis["slot_to_value"]
    value_to_slot = axis["value_to_slot"]

    # The two directions are mutual inverses, and cover exactly the four levels.
    assert {slot_to_value[slot] for slot in slot_to_value} == set(BASIS_LEVELS)
    assert value_to_slot == {value: slot for slot, value in slot_to_value.items()}
    assert sorted(slot_to_value) == ["B00", "B01", "B02", "B03"]

    planned_slots = {job["major_choices"]["basis"] for job in manifest["jobs"]}
    assert planned_slots == set(slot_to_value)
    # No row leaks a semantic level, in its choices or in its compiled command.
    for job in manifest["jobs"]:
        assert job["major_choices"]["basis"] not in BASIS_LEVELS
        for level in BASIS_LEVELS:
            assert f"basis_slot={level}" not in job["command"]

    # And the round trip reproduces the unblinded expansion exactly.
    plain_manifest = _manifest(_plan(tmp_path / "plain", GRID, "--no-blind"))
    recovered = sorted(slot_to_value[job["major_choices"]["basis"]] for job in manifest["jobs"])
    assert recovered == sorted(job["major_choices"]["basis"] for job in plain_manifest["jobs"])


def test_the_blinded_snapshot_resolves_every_basis_level_through_its_slot(tmp_path: Path) -> None:
    """The layer-5 reslot fix still holds for this grid, on every level.

    The library spells ``in_features`` as
    ``${tpen.basis_feature_dim:${choices.basis.<level>.basis}}``, which is what
    stops the embedding input width drifting from the basis feeding it. Blinding
    rekeys ``choices.basis`` by slot, so a verbatim copy leaves that interpolation
    pointing at a key that no longer exists and every array task dies on
    ``InterpolationKeyError``.
    """

    results_root = _plan(tmp_path, GRID)
    snapshot = OmegaConf.load(results_root / "00_grid" / ATTEMPT / "train_config.yaml")
    raw = OmegaConf.to_container(snapshot, resolve=False)

    assert sorted(raw["choices"]["basis"]) == ["B00", "B01", "B02", "B03"]
    for slot, level in raw["choices"]["basis"].items():
        in_features = level["in_features"]
        if in_features is None:  # the no-basis level
            continue
        assert f"choices.basis.{slot}.basis" in in_features
        for name in BASIS_LEVELS:
            assert f"choices.basis.{name}." not in in_features


# ---------------------------------------------------------------------------
# 3. Override paths exist in the configs they target
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grid_path", [GRID, SMOKE], ids=["grid", "smoke"])
def test_every_axis_override_path_exists_in_both_base_configs(grid_path: Path) -> None:
    train = _composed("train.yaml")
    evaluation = _composed("eval.yaml")

    for axis, path in _grid_data(grid_path)["axis_overrides"].items():
        assert _has_path(train, path), (axis, path, "train.yaml")
        assert _has_path(evaluation, path), (axis, path, "eval.yaml")


@pytest.mark.parametrize("grid_path", [GRID, SMOKE], ids=["grid", "smoke"])
def test_every_seed_override_path_exists_in_the_stage_config_it_drives(grid_path: Path) -> None:
    """Seed paths are checked per stage: train stages against train.yaml, eval against eval.yaml."""

    train = _composed("train.yaml")
    evaluation = _composed("eval.yaml")
    stage_configs = {
        "scan_train": train,
        "validation": evaluation,
        "final_train": train,
        "final_eval": evaluation,
    }

    policy = _grid_data(grid_path)["seed_overrides"]
    assert set(policy) == set(stage_configs)
    for stage, overrides in policy.items():
        for path in overrides:
            assert _has_path(stage_configs[stage], path), (stage, path)


def test_every_smoke_static_override_path_exists_in_the_stage_config_it_targets() -> None:
    """A static override on a key the config lacks is a silently inert workload.

    This is the reason the smoke's workload is small: an override that missed its
    key would leave the production budget in place and the "smoke" would run the
    full 500-step job while reporting as a smoke.
    """

    train = _composed("train.yaml")
    evaluation = _composed("eval.yaml")
    stage_configs = {
        "train": train,
        "validation": evaluation,
        "final_train": train,
        "final_eval": evaluation,
    }

    static = _grid_data(SMOKE)["static_overrides"]
    assert set(static) <= set(stage_configs)
    assert static, "the smoke must reduce the workload somewhere"
    for stage, overrides in static.items():
        for path in overrides:
            assert _has_path(stage_configs[stage], path), (stage, path)


def test_every_selected_and_reference_metric_names_a_task_the_eval_config_runs() -> None:
    """A metric whose task is not in the suite is a column that never arrives.

    Selection would then fall silently through the whole ladder to the fallback,
    which reads as "the runs were bad" rather than as "the metric was never
    emitted".
    """

    evaluation = _composed("eval.yaml")
    raw = OmegaConf.to_container(evaluation, resolve=False)
    namespace = raw["evaluation"]["namespace"]
    configured = {
        str(task).rsplit(".", 1)[-1].rstrip("}")
        for task in raw["evaluator"]["tasks"]
    }
    grid = _grid_data(GRID)
    metrics = [
        *grid["champions"][0]["metrics"],
        grid["champions"][0]["fallback_metric"],
        *(entry["metric"] for entry in grid["champion_reference_metrics"]),
    ]

    assert {"mcmc_energy", "stratified_geometry", "full_model_antisymmetry"} <= configured
    for metric in metrics:
        prefix, task, _key = metric.split("/", 2)
        assert prefix == namespace, metric
        assert task in configured, metric


# ---------------------------------------------------------------------------
# 4. Seeds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grid_path", [GRID, SMOKE], ids=["grid", "smoke"])
def test_scan_seed_triples_are_disjoint_within_and_across_rows(grid_path: Path) -> None:
    """Three independent streams per row, and no value reused anywhere.

    A repeated value silently fuses two streams that the paired-row design needs
    separable: validation has to reproduce a train row's MODEL while drawing a
    fresh sampler stream.
    """

    rows = _grid_data(grid_path)["scan_seed_rows"]
    names = ("training_model_seed", "training_sampler_seed", "validation_sampler_seed")
    values: list[int] = []
    for row in rows:
        triple = [row[name] for name in names]
        assert len(set(triple)) == len(names), row
        values.extend(triple)

    assert len(rows) == 3
    assert [row["seed_index"] for row in rows] == [0, 1, 2]
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("grid_path", [GRID, SMOKE], ids=["grid", "smoke"])
def test_final_seed_blocks_never_collide_with_the_scan_seeds_or_each_other(grid_path: Path) -> None:
    """A collision would make a "fresh" final replicate a rerun of a scan row.

    The final replicates exist to re-measure a champion on independent streams. A
    final seed that equals a scan seed reproduces the row selection already saw,
    so the "independent" measurement would inherit exactly the noise excursion
    that won.
    """

    data = _grid_data(grid_path)
    replicates = int(data["final_replicates"])
    scan_values = {
        row[name]
        for row in data["scan_seed_rows"]
        for name in ("training_model_seed", "training_sampler_seed", "validation_sampler_seed")
    }

    blocks = {}
    for name, spec in data["final_seed_sequences"].items():
        blocks[name] = {int(spec["start"]) + index * int(spec["step"]) for index in range(replicates)}
        assert len(blocks[name]) == replicates, name
        assert not (blocks[name] & scan_values), name

    names = sorted(blocks)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            assert not (blocks[name] & blocks[other]), (name, other)


# ---------------------------------------------------------------------------
# 5. Choice validation fails at plan time
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("axis", "block", "typo"),
    [
        ("basis", "major_grid", "hooke-axiswise-v11"),
        ("activation", "minor_grid", "Gaussain"),
    ],
)
def test_a_misspelled_axis_value_fails_at_plan_time(
    tmp_path: Path, axis: str, block: str, typo: str
) -> None:
    """One character wrong must stop the plan, not 288 jobs later.

    Without ``choice_validation`` the value would reach ``run_parameters`` as an
    override, and ``${choices.<axis>.<typo>...}`` would fail once per array task
    -- after submission, with the failure attributed to the runs rather than to
    the grid.
    """

    data = _grid_data(GRID)
    data[block][axis] = [typo, *data[block][axis][1:]]
    mutated = tmp_path / "typo_grid.yaml"
    OmegaConf.save(OmegaConf.create(data), mutated)

    with pytest.raises(ValueError) as excinfo:
        _plan(tmp_path, mutated)
    assert typo in str(excinfo.value)
    assert axis in str(excinfo.value)


def test_choice_validation_covers_every_axis_that_has_a_choice_library() -> None:
    """``lr`` and ``channels`` are numeric and have no library; the other two do.

    Blinding also requires it: ``plan.py`` refuses to blind a major axis with no
    ``choice_validation`` entry, because there would be no library to rekey.
    """

    data = _grid_data(GRID)
    assert set(data["choice_validation"]) == {"basis", "activation"}
    assert set(data["major_grid"]) <= set(data["choice_validation"])
    assert data["choice_validation"]["basis"]["choices_path"] == "choices.basis"
    assert data["choice_validation"]["activation"]["choices_path"] == "choices.activation"


# ---------------------------------------------------------------------------
# 6. Champion selection
# ---------------------------------------------------------------------------
#
# The selector is data, so these tests drive the real `select_champions.select`
# over a synthetic `03_collect` table whose numbers are chosen to separate the
# decisions under test. In every table below the ranking implied by
# `eval/stratified_geometry/local_energy_mean` is the OPPOSITE of the ranking
# implied by `eval/mcmc_energy/local_energy_mean`, which is what makes a swap back
# to the non-variational metric detectable rather than merely wrong.
# ---------------------------------------------------------------------------
SELECTION_SEEDS = ("0", "1")
HOLDOUT_SEED = "2"
# The configuration seed rows {0, 1} favour, and the one seed row 2 favours. They
# must differ, or a seed-2 leak would be invisible.
BEST_ON_SELECTION = "lr-3e-4_ch-8_act-SiLU"
BEST_ON_HOLDOUT = "lr-1e-3_ch-32_act-Tanh"


def _collection(results_root: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a synthetic ``03_collect`` attempt for the planned grid."""

    collect_dir = results_root / "03_collect" / "C1"
    _write_csv(collect_dir / "summary.csv", rows)
    (collect_dir / "source_grid_attempt.json").write_text(
        json.dumps({"grid_attempt_id": ATTEMPT}) + "\n"
    )
    layout.write_latest(results_root / "03_collect", "C1")


def _summary_rows(
    manifest: dict[str, Any],
    *,
    mcmc: Any,
    variance: Any,
    abs_error: Any,
) -> list[dict[str, Any]]:
    """Build summary rows from callables of ``(minor_id, seed_index)``."""

    rows = []
    for job in manifest["jobs"]:
        seed = int(job["scan_seed"])
        minor_id = job["minor_id"]
        energy = float(mcmc(minor_id, seed))
        rows.append(
            {
                "run_id": job["run_id"],
                "status": "completed",
                **{key: str(value) for key, value in job["choices"].items()},
                "major_id": job["major_id"],
                "minor_id": job["minor_id"],
                "config_id": job["config_id"],
                "eval/mcmc_energy/local_energy_mean": str(energy),
                "eval/mcmc_energy/energy_abs_error": str(abs_error(minor_id, seed)),
                "eval/stratified_geometry/local_energy_variance": str(variance(minor_id, seed)),
                # Deliberately anti-correlated with the MCMC energy: the fixed
                # prior has no variational floor, so the model that dips furthest
                # below 2.0 there is the worst-behaved one.
                "eval/stratified_geometry/local_energy_mean": str(2.0 - (energy - 2.0) * 10.0),
            }
        )
    return rows


def _planned_with_summary(tmp_path: Path, **builders: Any) -> Path:
    results_root = _plan(tmp_path, GRID)
    _collection(results_root, _summary_rows(_manifest(results_root), **builders))
    return results_root


def test_a_clear_winner_is_selected_on_the_variational_metric(tmp_path: Path) -> None:
    """Rung one decides, and the metric it decides on is the MCMC energy.

    ``MCMCGenerator`` draws from |psi|^2, so its mean local energy IS
    <psi|H|psi>/<psi|psi> and the variational principle bounds it below by the
    exact 2.0 Ha. The fixed-prior mean is E_q[E_L] for an arbitrary q -- unbounded
    below, and equal to 2.0 for ANY q at the exact eigenstate -- so ``min`` on it
    rewards the worst-behaved wavefunction. This test's table ranks the two in
    opposite orders, so selecting on the wrong one picks a different champion.
    """

    results_root = _planned_with_summary(
        tmp_path,
        # The clear winner: far below the field, with a tight spread.
        mcmc=lambda minor_id, seed: (2.01 if minor_id == BEST_ON_SELECTION else 2.50) + 0.001 * seed,
        variance=lambda minor_id, seed: 0.01 + 0.001 * seed,
        abs_error=lambda minor_id, seed: abs(
            (2.01 if minor_id == BEST_ON_SELECTION else 2.50) + 0.001 * seed - 2.0
        ),
    )

    report = select_champions.select(results_root=results_root, select_attempt_id="S1")["report"]
    champions = _read_csv(results_root / "04_select" / "S1" / "champions.csv")

    assert report["group_by"] == ["basis"]
    assert len(champions) == len(BASIS_LEVELS)
    assert {row["minor_id"] for row in champions} == {BEST_ON_SELECTION}
    assert {row["metric"] for row in champions} == {
        "eval/mcmc_energy/local_energy_mean_seed_median"
    }
    for decisions in report["decisions_by_group"].values():
        assert "eval/mcmc_energy/local_energy_mean" in decisions["energy"][0]
        assert "clearly wins" in decisions["energy"][0]


def test_the_holdout_seed_never_influences_the_champion_it_measures(tmp_path: Path) -> None:
    """Split-sample selection, and the assertion that the split is load-bearing.

    Seed row 2 is made overwhelmingly favourable to a DIFFERENT configuration than
    seed rows 0 and 1 prefer. If the holdout row were allowed to vote -- by
    aggregating all seeds, or by listing seed 2 among the selection seeds -- the
    champion identity would change. That is exactly the winner's-curse channel the
    split closes: with 24 configs per bucket the argmin is biased low by ~1.16
    sigma, and comparing buckets by their argmins favours the noisier bucket.
    """

    results_root = _planned_with_summary(
        tmp_path,
        mcmc=lambda minor_id, seed: (
            0.5  # a huge, isolated excursion on the holdout row only
            if (seed == 2 and minor_id == BEST_ON_HOLDOUT)
            else (2.01 if minor_id == BEST_ON_SELECTION else 2.50)
        ),
        variance=lambda minor_id, seed: 0.01,
        abs_error=lambda minor_id, seed: 0.01 if minor_id == BEST_ON_SELECTION else 0.50,
    )

    result = select_champions.select(results_root=results_root, select_attempt_id="S1")
    report = result["report"]
    champions = _read_csv(results_root / "04_select" / "S1" / "champions.csv")
    split = report["split_sample"]["energy"]

    assert split["enabled"] is True
    assert split["selection_seeds"] == list(SELECTION_SEEDS)
    assert split["holdout_seeds"] == [HOLDOUT_SEED]
    # The champion is the configuration the SELECTION seeds prefer, in every
    # bucket, and never the one the holdout row favours.
    assert {row["minor_id"] for row in champions} == {BEST_ON_SELECTION}
    assert BEST_ON_HOLDOUT not in {row["minor_id"] for row in champions}
    assert report["overall_champion"].endswith(BEST_ON_SELECTION)
    # The cross-bucket champion is chosen on the same sample as the per-bucket
    # ones. Its seed count is the only observable that distinguishes "selected on
    # {0,1}" from "selected on {0,1,2}", because a three-row median absorbs one
    # excursion and would leave the identity unchanged.
    assert int(report["overall_metric_seed_n"]) == len(SELECTION_SEEDS)
    assert int(report["secondary_metric_seed_n"]) == len(SELECTION_SEEDS)

    # The champion's own metric is re-read on the row that had no vote.
    for row in champions:
        assert row["holdout_seeds"] == HOLDOUT_SEED
        assert row["holdout_metric"] == "eval/mcmc_energy/local_energy_mean_seed_median"
        assert float(row["holdout_metric_value"]) == pytest.approx(2.01)
        assert int(row["holdout_metric_seed_n"]) == 1
        # Two seed rows chose it; one measured it. This is the count that catches a
        # leak the median would absorb: a seed-median over three rows is robust to
        # one excursion, so identity alone is not a sensitive test of the split.
        assert int(row["metric_seed_n"]) == len(SELECTION_SEEDS)


def test_perturbing_the_holdout_row_changes_no_selection_decision(tmp_path: Path) -> None:
    """The exhaustive form: seed row 2 cannot move ANY selection output.

    Two selections over the same collection, differing only in the holdout row's
    numbers, must agree on every champion, every ladder decision, and both
    cross-bucket champions. Any path that reads the holdout while deciding -- a
    per-bucket aggregate over all seeds, a cross-bucket champion taken from the
    full sample, a seed filter that silently does nothing -- changes one of those
    and fails here, including paths a champion-identity assertion would miss
    because a three-row median absorbs one excursion.

    The final assertion is the anti-vacuity guard: the perturbation must still
    reach the REPORTED holdout value, or the test would also pass on a
    configuration that never reads seed row 2 at all.
    """

    def summary(perturbed: bool) -> Any:
        def mcmc(minor_id: str, seed: int) -> float:
            if perturbed and seed == 2:
                # Absurdly favourable, and favourable to a different configuration.
                return -1.0e6 if minor_id == BEST_ON_HOLDOUT else 1.0e6
            return 2.01 if minor_id == BEST_ON_SELECTION else 2.50

        return {
            "mcmc": mcmc,
            "variance": lambda minor_id, seed: (
                -1.0e6 if (perturbed and seed == 2 and minor_id == BEST_ON_HOLDOUT) else 0.01
            ),
            "abs_error": lambda minor_id, seed: (
                -1.0e6
                if (perturbed and seed == 2 and minor_id == BEST_ON_HOLDOUT)
                else (0.01 if minor_id == BEST_ON_SELECTION else 0.50)
            ),
        }

    reports = {}
    champions = {}
    for label, perturbed in (("base", False), ("perturbed", True)):
        results_root = _plan(tmp_path / label, GRID)
        _collection(results_root, _summary_rows(_manifest(results_root), **summary(perturbed)))
        reports[label] = select_champions.select(
            results_root=results_root, select_attempt_id="S1"
        )["report"]
        champions[label] = _read_csv(results_root / "04_select" / "S1" / "champions.csv")

    holdout_columns = set(select_champions.HOLDOUT_COLUMNS)
    decided = [
        [{key: value for key, value in row.items() if key not in holdout_columns} for row in rows]
        for rows in (champions["base"], champions["perturbed"])
    ]

    assert decided[0] == decided[1]
    for key in (
        "overall_champion",
        "overall_metric",
        "overall_metric_value",
        "secondary_champion",
        "secondary_metric",
        "decisions_by_group",
        "bucket_distributions",
    ):
        assert reports["base"][key] == reports["perturbed"][key], key

    # The perturbation did reach the artifact -- through the reported holdout
    # measurement, which is the only place it is allowed to appear.
    base_holdout = {row["holdout_metric_value"] for row in champions["base"]}
    perturbed_holdout = {row["holdout_metric_value"] for row in champions["perturbed"]}
    assert base_holdout != perturbed_holdout
    assert all(float(value) == pytest.approx(1.0e6) for value in perturbed_holdout)


def test_a_seed_that_selects_the_champion_cannot_also_evaluate_it(tmp_path: Path) -> None:
    """An overlapping split is a leak, and fails loudly rather than being trimmed."""

    data = _grid_data(GRID)
    data["champions"][0]["selection_seeds"] = [0, 1, 2]
    mutated = tmp_path / "leaky_grid.yaml"
    OmegaConf.save(OmegaConf.create(data), mutated)
    results_root = _plan(tmp_path, mutated)
    _collection(
        results_root,
        _summary_rows(
            _manifest(results_root),
            mcmc=lambda minor_id, seed: 2.0 + 0.01 * seed,
            variance=lambda minor_id, seed: 0.01,
            abs_error=lambda minor_id, seed: 0.01,
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        select_champions.select(results_root=results_root, select_attempt_id="S1")
    assert "overlap" in str(excinfo.value)


def test_overlapping_error_bars_fall_through_to_the_deterministic_fallback(
    tmp_path: Path,
) -> None:
    """Every rung ties within its error bars, so the configured fallback decides.

    The fallback is |mean - 2.0| against the exact Hooke energy, never wall time:
    wall time is machine- and load-dependent, so a wall-time fallback makes
    champion identity irreproducible across independent selections of the same
    collected data.

    The overlap test itself is NOT a calibrated threshold -- it is a 1-stderr
    non-overlap rule on a 2-seed spread -- which is why the fallback has to be a
    quantity someone chose rather than whatever the ladder happened to leave.
    """

    # Identical medians on rungs one and two, with a wide per-seed spread so no
    # pair of error bars separates; only the fallback metric distinguishes.
    results_root = _planned_with_summary(
        tmp_path,
        mcmc=lambda minor_id, seed: 2.2 + (0.5 if seed % 2 else -0.5),
        variance=lambda minor_id, seed: 0.05 + (0.02 if seed % 2 else -0.02),
        abs_error=lambda minor_id, seed: (
            (0.10 if minor_id == BEST_ON_SELECTION else 0.20) + (0.5 if seed % 2 else -0.5)
        ),
    )

    report = select_champions.select(results_root=results_root, select_attempt_id="S1")["report"]
    champions = _read_csv(results_root / "04_select" / "S1" / "champions.csv")

    for decisions in report["decisions_by_group"].values():
        ladder = decisions["energy"]
        assert any("overlap the leader" in line for line in ladder[:-1])
        assert "fallback" in ladder[-1]
        assert "energy_abs_error" in ladder[-1]
        assert "wall_time" not in ladder[-1]
    assert {row["minor_id"] for row in champions} == {BEST_ON_SELECTION}
    assert {row["metric"] for row in champions} == {
        "eval/mcmc_energy/energy_abs_error_seed_median"
    }


def test_each_bucket_reports_its_distribution_alongside_its_champion(tmp_path: Path) -> None:
    """A champion is one draw; the bucket's distribution is the check on it.

    If the basis ranking flips between champion and median, the champion ranking
    was noise. The distribution is computed on the same selection rows the
    champion was chosen from, so the two are comparable.
    """

    results_root = _planned_with_summary(
        tmp_path,
        mcmc=lambda minor_id, seed: 2.01 if minor_id == BEST_ON_SELECTION else 2.50,
        variance=lambda minor_id, seed: 0.01,
        abs_error=lambda minor_id, seed: 0.01 if minor_id == BEST_ON_SELECTION else 0.50,
    )

    report = select_champions.select(results_root=results_root, select_attempt_id="S1")["report"]
    buckets = report["bucket_distributions"]["energy"]

    assert len(buckets) == len(BASIS_LEVELS)
    for distributions in buckets.values():
        summary = distributions["mcmc_energy"]
        # 24 minor configurations per basis bucket; one of them is the champion.
        assert summary["n"] == 24
        assert float(summary["best"]) == pytest.approx(2.01)
        assert float(summary["median"]) == pytest.approx(2.50)
        assert float(summary["q1"]) <= float(summary["median"]) <= float(summary["q3"])
        assert len(summary["best_k"]) == 3
        assert summary["best_k"][0]["config_id"].endswith(BEST_ON_SELECTION)
    # The non-variational fixed-prior mean is reported for the record and never
    # selected on.
    assert "stratified_energy_mean" in next(iter(buckets.values()))


def test_the_selector_never_ranks_on_the_non_variational_mean_or_on_wall_time() -> None:
    """The two ways this study's selection can be wrong, pinned as data.

    ``eval/stratified_geometry/local_energy_mean`` is E_q[E_L] on a FIXED prior
    (``StratifiedGeometryGenerator.generate`` begins ``del model``), so it has no
    variational floor and ``min`` on it prefers the wavefunction that dips
    furthest below 2.0. ``train/runtime/wall_time_sec`` was the previous
    fallback and makes champion identity machine-dependent.
    """

    for grid_path in (GRID, SMOKE):
        spec = _grid_data(grid_path)["champions"][0]
        reference_labels = {
            entry["label"]: entry["metric"]
            for entry in _grid_data(grid_path)["champion_reference_metrics"]
        }
        ranked = [*spec["metrics"], spec["fallback_metric"]]

        assert spec["selector"] == "metric_ladder"
        assert spec["mode"] == "min"
        assert spec["fallback_mode"] == "min"
        assert spec["metrics"][0] == "eval/mcmc_energy/local_energy_mean"
        assert spec["metrics"][1] == "eval/stratified_geometry/local_energy_variance"
        assert spec["fallback_metric"] == "eval/mcmc_energy/energy_abs_error"
        for metric in ranked:
            assert metric != "eval/stratified_geometry/local_energy_mean", grid_path
            assert not metric.startswith("train/"), grid_path
        # Recorded for the report, and only there.
        assert reference_labels["stratified_energy_mean"] == (
            "eval/stratified_geometry/local_energy_mean"
        )


# ---------------------------------------------------------------------------
# 7. The smoke's terminal checkpoint, at the config level
# ---------------------------------------------------------------------------
def test_the_smoke_declares_a_terminal_only_checkpoint_stream() -> None:
    """The scan's checkpoint selection policy is explicit and ungated.

    The scan's checkpoint stream is explicitly ``TerminalOnly`` in ``train.yaml``
    and therefore has no checkpoint cadence to divide the smoke budget. The
    behavioural check that the write actually happens is in
    ``tests/unit/experiments/test_scan_smoke_workload.py``; this one keeps the
    static override surface from creating a legacy cadence key.
    """

    static = _grid_data(SMOKE)["static_overrides"]
    for stage, overrides in static.items():
        max_steps = overrides.get("training.max_steps")
        assert isinstance(max_steps, int) and max_steps > 0, stage
        assert "checkpoint.every_n_steps" not in overrides
        # Every other cadence in the stage must also divide the budget, or the
        # smoke silently exercises fewer windows than it claims.
        for path in ("checks.every_n_steps", "status.every_n_steps", "training.log_every_n_steps"):
            if path in overrides:
                assert max_steps % int(overrides[path]) == 0, (stage, path)
