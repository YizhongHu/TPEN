"""Contracts the tpen-pair-scan-v1 fork must hold, independent of stage logic.

Three properties live here because none of them is a property of one stage:

1. The ``experiments/`` import rule (``experiments/README.md``): nothing under
   this study may import ``tpen`` except ``tpen.run.run_from_config`` in a
   launcher-style script.
2. The choice-library merge. The scan's configs are deliberately not
   self-contained -- ``choices.basis`` lives in one shared table -- so the
   planner owes every compiled command a merged config. The dangerous failure is
   not a crash but a SILENT partial resolution, so these tests assert both that
   the merge happens and that its absence is loud.
3. The fork's disposition: the modules the fork drops are absent rather than
   inert, and the modules it retargets no longer name a metric the TPEN eval
   suite does not produce.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
CONFIGS = STUDY_DIR / "configs"
REPO_ROOT = STUDY_DIR.parents[2]
BASIS_LIBRARY = REPO_ROOT / "experiments" / "hooke" / "choices" / "basis_levels.yaml"
V3_STUDY_DIR = STUDY_DIR.parent / "pair_stability_v3"

while str(STUDY_DIR) in sys.path:
    sys.path.remove(str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR))
_STUDY_TOP_LEVEL_MODULES = {
    "collect",
    "final_collect",
    "final_eval",
    "final_plan",
    "final_report",
    "final_train",
    "launch",
    "plan",
    "plot",
    "select_champions",
    "stats",
    "train",
    "utils",
    "validate",
}
for _module_name in list(sys.modules):
    if _module_name.split(".", maxsplit=1)[0] in _STUDY_TOP_LEVEL_MODULES:
        del sys.modules[_module_name]


def _load_script(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tpen_pair_scan_v1_contract_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

study_config = sibling(__file__, 'utils.config')

launch = _load_script("launch")
plan = _load_script("plan")
final_collect = _load_script("final_collect")

ATTEMPT = "20260813T120000-0400"

# The library declaration a grid must carry for the scan configs to be runnable.
BASIS_LIBRARY_SPECS = [
    {"path": "experiments/hooke/choices/basis_levels.yaml", "provides": ["choices.basis"]}
]


# ---------------------------------------------------------------------------
# Grid fixture
# ---------------------------------------------------------------------------
def _grid_data(results_root: Path, *, choice_libraries: object | None = "default") -> dict:
    """Return the fork's grid shape: basis x activation major, lr x channels minor."""

    grid = {
        "study": "tpen_pair_scan_v1",
        "config": "experiments/hooke/tpen-pair-scan-v1/configs/train.yaml",
        "validation_config": "experiments/hooke/tpen-pair-scan-v1/configs/eval.yaml",
        "results_root": str(results_root),
        "config_snapshots": {"train": "train_config.yaml", "validation": "validation_config.yaml"},
        "major_grid": {
            "basis": ["no-basis", "hooke-total-shell"],
            "activation": ["SiLU", "Tanh"],
        },
        "minor_grid": {"lr": [1.0e-3], "channels": [8]},
        "scan_seed_axis": "seed_index",
        "scan_seed_rows": [
            {
                "seed_index": 0,
                "training_model_seed": 0,
                "training_sampler_seed": 10,
                "validation_sampler_seed": 20,
            }
        ],
        "blinding": {
            "enabled_by_default": False,
            "slot_prefixes": {"basis": "B", "activation": "A"},
        },
        "axis_id_labels": {
            "basis": "b",
            "activation": "act",
            "lr": "lr",
            "channels": "ch",
            "seed_index": "seed",
        },
        "axis_overrides": {
            "basis": "run_parameters.basis_slot",
            "activation": "run_parameters.activation_slot",
            "lr": "run_parameters.lr",
            "channels": "run_parameters.channels",
        },
        "choice_validation": {
            "basis": {
                "choices_path": "choices.basis",
                "tags_path": "choices.basis.{value}.tags",
            },
            "activation": {
                "choices_path": "choices.activation",
                "tags_path": "choices.activation.{value}.tags",
            },
        },
        "seed_overrides": {
            "scan_train": {
                "run_parameters.training_model_seed": "training_model_seed",
                "run_parameters.training_sampler_seed": "training_sampler_seed",
            },
            "validation": {
                "run_parameters.training_model_seed": "training_model_seed",
                "run_parameters.validation_sampler_seed": "validation_sampler_seed",
            },
        },
        "champions": [
            {
                "name": "energy",
                "selector": "metric_ladder",
                "tasks": ["mcmc_energy"],
                "metric_template": "eval/{task}/local_energy_mean",
                "mode": "min",
            }
        ],
        "final_replicates": 1,
    }
    if choice_libraries == "default":
        grid["choice_libraries"] = [
            {
                "path": "experiments/hooke/choices/basis_levels.yaml",
                "provides": "choices.basis",
            }
        ]
    elif choice_libraries is not None:
        grid["choice_libraries"] = choice_libraries
    return grid


def _write_grid(tmp_path: Path, **kwargs) -> Path:
    grid_path = tmp_path / "grid.yaml"
    OmegaConf.save(OmegaConf.create(_grid_data(tmp_path / "results", **kwargs)), grid_path)
    return grid_path


def _plan(monkeypatch: pytest.MonkeyPatch, grid_path: Path, *extra: str) -> None:
    # The grid's config paths are repo-root relative, exactly as the real grid's
    # will be, so the planner runs from the repo root.
    monkeypatch.chdir(REPO_ROOT)
    assert plan.main(["--grid", str(grid_path), "--attempt-id", ATTEMPT, *extra]) == 0


# ---------------------------------------------------------------------------
# 1. The experiments/ import rule
# ---------------------------------------------------------------------------
def _study_python_files() -> list[Path]:
    return sorted(path for path in STUDY_DIR.rglob("*.py") if path.is_file())


def _tpen_imports(path: Path) -> list[str]:
    """Return every dotted ``tpen`` name a module imports."""

    imported: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tpen" or alias.name.startswith("tpen."):
                    imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "tpen" or module.startswith("tpen."):
                imported.extend(f"{module}.{alias.name}" for alias in node.names)
    return imported


def test_study_python_files_are_discovered_before_the_import_rule_is_checked():
    # Guards the rule below against passing because the glob found nothing.
    paths = _study_python_files()
    names = {path.name for path in paths}
    assert len(paths) >= 15, paths
    assert {"plan.py", "launch.py", "final_collect.py", "final_report.py"} <= names


def test_no_study_module_imports_tpen_outside_the_sanctioned_launcher_entrypoint():
    # experiments/README.md: code under experiments/ must not import tpen; the
    # single exception is tpen.run.run_from_config for launcher-style scripts.
    offending = {
        str(path.relative_to(STUDY_DIR)): [
            name for name in _tpen_imports(path) if name != "tpen.run.run_from_config"
        ]
        for path in _study_python_files()
    }
    assert {path: names for path, names in offending.items() if names} == {}


def _config_targets(config_path: Path) -> list[str]:
    """Return every ``_target_`` string a config declares, at any depth."""

    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    targets: list[str] = []

    def collect(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "_target_":
                    targets.append(str(value))
                else:
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(data)
    return targets


@pytest.mark.parametrize("name", ["train.yaml", "eval.yaml"])
def test_stage_configs_reference_tpen_only_through_target_strings(name):
    # Run configs may name tpen classes, but only as Hydra _target_ strings.
    targets = _config_targets(CONFIGS / name)
    assert targets, name
    assert [target for target in targets if target.startswith("tpen.")], name


@pytest.mark.parametrize("name", ["grid.yaml", "smoke.yaml"])
def test_grid_configs_declare_no_component_targets(name):
    # The grid configs are launcher METADATA: plan.py reads them, run.py never
    # does, and nothing instantiates them. A `_target_` here would be a component
    # spec written where no instantiation happens -- silently inert, and a
    # standing invitation to move real wiring out of the stage configs, where the
    # train/eval resolved-model identity check would have caught a mistake.
    assert _config_targets(CONFIGS / name) == []


def test_every_study_config_is_covered_by_one_of_the_two_target_rules():
    # Guards both rules above against a new config file slipping in unchecked.
    assert {path.name for path in CONFIGS.glob("*.yaml")} == {
        "train.yaml",
        "eval.yaml",
        "grid.yaml",
        "smoke.yaml",
    }


# ---------------------------------------------------------------------------
# 2. The choice-library merge
# ---------------------------------------------------------------------------
def test_scan_configs_are_not_self_contained_on_their_own():
    # The premise the merge exists for. If this ever stops holding, the merge
    # machinery is dead weight and the tests below prove nothing.
    for name in ("train.yaml", "eval.yaml"):
        raw = OmegaConf.load(CONFIGS / name)
        assert OmegaConf.select(raw, "choices.basis") is None, name


def test_loading_a_scan_config_without_the_library_fails_loudly_not_partially():
    # The intended failure mode: a dangling interpolation, not a null basis.
    raw = OmegaConf.load(CONFIGS / "train.yaml")
    with pytest.raises(Exception) as excinfo:
        OmegaConf.select(raw, "model.basis")
    assert "choices.basis" in str(excinfo.value)


def test_load_composed_config_populates_every_basis_level():
    composed = study_config.load_composed_config(
        CONFIGS / "train.yaml",
        study_config.choice_library_specs(BASIS_LIBRARY_SPECS),
        required_paths=["choices.basis"],
        repo_root=REPO_ROOT,
    )
    levels = OmegaConf.select(composed, "choices.basis")
    assert set(levels.keys()) == {
        "no-basis",
        "hooke-axiswise-v1",
        "hooke-total-shell",
        "hooke-cartesian-box",
    }
    # The merged config resolves the slot the base config selects by default.
    assert (
        OmegaConf.select(composed, "run_parameters.basis_slot") in levels
    ), OmegaConf.select(composed, "run_parameters.basis_slot")


def test_require_choice_paths_rejects_a_config_that_never_got_the_merge():
    raw = OmegaConf.load(CONFIGS / "train.yaml")
    with pytest.raises(ValueError, match="choices.basis"):
        study_config.require_choice_paths(raw, ["choices.basis"], context="unmerged train config")


def test_require_choice_paths_rejects_a_scalar_at_a_required_path():
    # A choice library is a table of levels. A scalar there is a merge that went
    # wrong, not a merge that succeeded, and it must not pass merely by being
    # non-None.
    cfg = OmegaConf.create({"choices": {"basis": 4}})
    with pytest.raises(ValueError, match="missing required choice path"):
        study_config.require_choice_paths(cfg, ["choices.basis"], context="scalar basis")


def test_require_choice_paths_rejects_a_list_at_a_required_path():
    cfg = OmegaConf.create({"choices": {"basis": ["no-basis"]}})
    with pytest.raises(ValueError, match="missing required choice path"):
        study_config.require_choice_paths(cfg, ["choices.basis"], context="list basis")


def test_require_choice_paths_rejects_an_empty_choice_table():
    # A merge against a fragment defining an empty `choices.basis:` would
    # otherwise look like a successful merge.
    cfg = OmegaConf.create({"choices": {"basis": {}}})
    with pytest.raises(ValueError, match="empty choice path"):
        study_config.require_choice_paths(cfg, ["choices.basis"], context="empty table")


def test_load_composed_config_without_libraries_reports_the_missing_path():
    with pytest.raises(ValueError, match="choices.basis"):
        study_config.load_composed_config(
            CONFIGS / "train.yaml", [], required_paths=["choices.basis"], repo_root=REPO_ROOT
        )


def test_resolve_library_path_rejects_a_declared_fragment_that_is_absent():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        study_config.resolve_library_path("experiments/hooke/choices/no_such_library.yaml", repo_root=REPO_ROOT)


def test_choice_library_specs_normalizes_paths_and_provides():
    specs = study_config.choice_library_specs(
        ["a.yaml", {"path": "b.yaml", "provides": "choices.basis"}, {"path": "c.yaml", "provides": ["x", "y"]}]
    )
    assert specs == [
        {"path": "a.yaml", "provides": []},
        {"path": "b.yaml", "provides": ["choices.basis"]},
        {"path": "c.yaml", "provides": ["x", "y"]},
    ]
    assert study_config.choice_library_provides(specs) == ["choices.basis", "x", "y"]


def test_choice_library_specs_rejects_an_entry_without_a_path():
    with pytest.raises(ValueError, match="non-empty path"):
        study_config.choice_library_specs([{"provides": "choices.basis"}])


def test_required_choice_paths_unions_library_provides_and_choice_validation(tmp_path):
    grid = _grid_data(tmp_path / "results")
    assert plan.required_choice_paths(grid) == ["choices.basis", "choices.activation"]
    # Dropping the library declaration must not drop the requirement: the
    # choice_validation entry still demands the path.
    without_library = _grid_data(tmp_path / "results", choice_libraries=None)
    assert "choices.basis" in plan.required_choice_paths(without_library)


def test_a_library_provide_is_required_even_with_no_choice_validation_entry(tmp_path):
    # The two declarations are independent halves of the union. A grid may merge
    # a fragment it does not validate grid values against -- that path must
    # still be required, or the merge for it is unenforced.
    grid = _grid_data(tmp_path / "results")
    grid["choice_libraries"] = [
        {"path": "experiments/hooke/choices/basis_levels.yaml", "provides": "choices.unvalidated"}
    ]
    paths = plan.required_choice_paths(grid)
    assert paths[0] == "choices.unvalidated"
    assert paths == ["choices.unvalidated", "choices.basis", "choices.activation"]


def test_planned_grid_attempt_snapshot_carries_the_merged_basis_library(tmp_path, monkeypatch):
    grid_path = _write_grid(tmp_path)
    _plan(monkeypatch, grid_path)
    snapshot = tmp_path / "results" / "00_grid" / ATTEMPT / "train_config.yaml"
    levels = OmegaConf.select(OmegaConf.load(snapshot), "choices.basis")
    assert levels is not None
    assert {"no-basis", "hooke-total-shell"} <= set(levels.keys())


def test_planned_validation_snapshot_carries_the_merged_basis_library(tmp_path, monkeypatch):
    grid_path = _write_grid(tmp_path)
    _plan(monkeypatch, grid_path)
    snapshot = tmp_path / "results" / "00_grid" / ATTEMPT / "validation_config.yaml"
    levels = OmegaConf.select(OmegaConf.load(snapshot), "choices.basis")
    assert levels is not None
    assert {"no-basis", "hooke-total-shell"} <= set(levels.keys())


def test_every_planned_command_runs_the_merged_snapshot_not_the_source_config(tmp_path, monkeypatch):
    # This is the property that makes the merge unavoidable: no compiled command
    # may reference configs/train.yaml, whose choices.basis is absent.
    grid_path = _write_grid(tmp_path)
    _plan(monkeypatch, grid_path)
    manifest = OmegaConf.to_container(
        OmegaConf.load(tmp_path / "results" / "00_grid" / ATTEMPT / "manifest.json"), resolve=True
    )
    snapshot = str(tmp_path / "results" / "00_grid" / ATTEMPT / "train_config.yaml")
    assert manifest["jobs"]
    for job in manifest["jobs"]:
        assert f"--config {snapshot}" in job["command"], job["command"]
        assert "configs/train.yaml" not in job["command"], job["command"]
    assert manifest["config"] == snapshot
    assert manifest["validation_config"] == str(
        tmp_path / "results" / "00_grid" / ATTEMPT / "validation_config.yaml"
    )


def test_planned_manifest_records_the_libraries_it_merged(tmp_path, monkeypatch):
    grid_path = _write_grid(tmp_path)
    _plan(monkeypatch, grid_path)
    manifest = OmegaConf.to_container(
        OmegaConf.load(tmp_path / "results" / "00_grid" / ATTEMPT / "manifest.json"), resolve=True
    )
    assert manifest["choice_libraries"] == [
        {"path": "experiments/hooke/choices/basis_levels.yaml", "provides": ["choices.basis"]}
    ]
    assert manifest["required_choice_paths"] == ["choices.basis", "choices.activation"]


def test_planning_without_the_declared_library_fails_instead_of_planning(tmp_path, monkeypatch):
    grid_path = _write_grid(tmp_path, choice_libraries=None)
    monkeypatch.chdir(REPO_ROOT)
    with pytest.raises(ValueError, match="choices.basis"):
        plan.main(["--grid", str(grid_path), "--attempt-id", ATTEMPT])
    assert not (tmp_path / "results" / "00_grid" / ATTEMPT / "manifest.json").exists()


def test_planning_with_a_missing_library_file_fails_instead_of_planning(tmp_path, monkeypatch):
    grid_path = _write_grid(
        tmp_path,
        choice_libraries=[{"path": "experiments/hooke/choices/absent.yaml", "provides": "choices.basis"}],
    )
    monkeypatch.chdir(REPO_ROOT)
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        plan.main(["--grid", str(grid_path), "--attempt-id", ATTEMPT])


def _write_attempt(tmp_path: Path, grid_data: dict, config_snapshot_data: dict) -> None:
    plan.write_grid_attempt(
        results_root=tmp_path / "results",
        attempt_id=ATTEMPT,
        created_at="2026-08-13T12:00:00-04:00",
        config=CONFIGS / "train.yaml",
        grid=tmp_path / "grid.yaml",
        grid_data=grid_data,
        jobs=[],
        config_snapshot_data=config_snapshot_data,
    )


def test_write_grid_attempt_rejects_a_train_snapshot_that_lost_the_library(tmp_path):
    # The on-disk check on the TRAIN snapshot, isolated: no validation_config, so
    # only the train verification can fire.
    grid_data = _grid_data(tmp_path / "results")
    grid_data.pop("validation_config")
    grid_data["required_choice_paths"] = ["choices.basis"]
    with pytest.raises(ValueError, match="train config snapshot"):
        _write_attempt(tmp_path, grid_data, {"train": OmegaConf.load(CONFIGS / "train.yaml")})


def test_write_grid_attempt_rejects_a_validation_snapshot_that_lost_the_library(tmp_path):
    # The on-disk check on the VALIDATION snapshot, isolated: the train snapshot
    # is correctly merged, so only the validation verification can fire.
    grid_data = _grid_data(tmp_path / "results")
    grid_data["required_choice_paths"] = ["choices.basis"]
    merged_train = study_config.load_composed_config(
        CONFIGS / "train.yaml",
        study_config.choice_library_specs(BASIS_LIBRARY_SPECS),
        required_paths=["choices.basis"],
        repo_root=REPO_ROOT,
    )
    with pytest.raises(ValueError, match="validation config snapshot"):
        _write_attempt(
            tmp_path,
            grid_data,
            {"train": merged_train, "validation": OmegaConf.load(CONFIGS / "eval.yaml")},
        )


def test_write_grid_attempt_accepts_snapshots_that_carry_the_library(tmp_path):
    # Keeps the two rejection tests honest: the same call path succeeds when both
    # snapshots are properly merged.
    grid_data = _grid_data(tmp_path / "results")
    grid_data["required_choice_paths"] = ["choices.basis"]
    specs = study_config.choice_library_specs(BASIS_LIBRARY_SPECS)
    merged = {
        stage: study_config.load_composed_config(
            CONFIGS / name,
            specs,
            required_paths=["choices.basis"],
            repo_root=REPO_ROOT,
        )
        for stage, name in (("train", "train.yaml"), ("validation", "eval.yaml"))
    }
    _write_attempt(tmp_path, grid_data, merged)
    assert (tmp_path / "results" / "00_grid" / ATTEMPT / "manifest.json").is_file()


def test_composition_is_validated_before_any_durable_artifact_is_written(tmp_path, monkeypatch):
    # A required path that no `choice_validation` entry covers, so `validate_grid`
    # cannot catch it, and no validation config, so only the TRAIN composition
    # check can. It must fire before the grid attempt directory exists: a failed
    # plan must leave no half-written lineage for a later stage to pick up.
    grid = _grid_data(tmp_path / "results")
    grid.pop("validation_config")
    grid["choice_libraries"] = [
        {"path": "experiments/hooke/choices/basis_levels.yaml", "provides": "choices.unvalidated"}
    ]
    grid_path = tmp_path / "grid.yaml"
    OmegaConf.save(OmegaConf.create(grid), grid_path)
    monkeypatch.chdir(REPO_ROOT)
    with pytest.raises(ValueError, match="choices.unvalidated"):
        plan.main(["--grid", str(grid_path), "--attempt-id", ATTEMPT])
    assert not (tmp_path / "results" / "00_grid" / ATTEMPT).exists()


def test_blinding_reslots_the_basis_library_self_reference(tmp_path, monkeypatch):
    # The basis library's in_features references its own level BY NAME
    # (`${choices.basis.hooke-total-shell.basis}`). Blinding rekeys the library
    # by slot, so a verbatim copy would leave that interpolation dangling and the
    # run would die inside a Slurm array task.
    grid_path = _write_grid(tmp_path)
    _plan(monkeypatch, grid_path, "--blind", "--blind-seed", "811")
    snapshot = OmegaConf.load(tmp_path / "results" / "00_grid" / ATTEMPT / "train_config.yaml")
    levels = OmegaConf.select(snapshot, "choices.basis")
    slots = set(levels.keys())
    assert slots == {"B00", "B01"}
    raw = OmegaConf.to_container(snapshot, resolve=False)
    for slot in slots:
        in_features = raw["choices"]["basis"][slot]["in_features"]
        if in_features is None:
            continue
        assert f"choices.basis.{slot}." in in_features, in_features
        assert "hooke-total-shell" not in in_features, in_features
        assert "no-basis" not in in_features, in_features


def test_blinding_rejects_a_library_reference_to_an_unscanned_level(tmp_path):
    # No slot exists for a level the grid does not scan, so the reference cannot
    # be rewritten and blinding must refuse rather than emit a dangling key.
    config = OmegaConf.create(
        {
            "run_parameters": {"basis_slot": "kept"},
            "choices": {
                "basis": {
                    "kept": {"in_features": "${choices.basis.unscanned.width}"},
                    "unscanned": {"width": 4},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="unscanned"):
        plan._materialize_slot_config(
            config,
            validation_specs={"basis": {"choices_path": "choices.basis"}},
            maps={
                "basis": {
                    "slot_to_value": {"B00": "kept"},
                    "value_to_slot": {"kept": "B00"},
                }
            },
            axis_override_paths={"basis": "run_parameters.basis_slot"},
        )


def test_basis_library_is_a_fragment_shared_with_the_integration_fixtures():
    # The merge is only worth having if there is exactly one table; a study-local
    # copy would defeat the point.
    library = OmegaConf.load(BASIS_LIBRARY)
    assert set(OmegaConf.to_container(library, resolve=False)) == {"choices"}
    assert not (CONFIGS / "basis_levels.yaml").exists()


# ---------------------------------------------------------------------------
# 3. Fork disposition: drops absent, metrics retargeted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "relative_path",
    [
        "parity.py",
        "sync.py",
        "test_pair_stability_v3_parity.py",
        "test_sync.py",
        "configs/pilot.yaml",
        "configs/pilot_smoke.yaml",
    ],
)
def test_dropped_files_are_absent_from_the_fork(relative_path):
    assert not (STUDY_DIR / relative_path).exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "parity.py",
        "sync.py",
        "test_pair_stability_v3_parity.py",
        "test_sync.py",
        "configs/pilot.yaml",
        "configs/pilot_smoke.yaml",
        "final_report.py",
    ],
)
def test_the_frozen_v3_study_still_holds_every_dropped_file(relative_path):
    # D9 freezes pair_stability_v3 as historical provenance. The fork drops these
    # files; it does not delete them from the study they came from. This also
    # keeps the test above honest: it fails if the paths were never real.
    assert (V3_STUDY_DIR / relative_path).is_file()


def _defined_names(path: Path) -> set[str]:
    """Return the top-level function and class names a module defines."""

    return {
        node.name
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_the_forked_report_is_a_rewrite_not_a_copy_of_the_v3_report():
    forked = STUDY_DIR / "final_report.py"
    original = V3_STUDY_DIR / "final_report.py"
    assert forked.read_text() != original.read_text()
    # A retarget would have kept the v3 figure/section builders. A rewrite shares
    # only the stage entry points.
    shared = _defined_names(forked) & _defined_names(original)
    assert shared <= {"main", "parse_args", "build_report"}, sorted(shared)
    # Named rather than a size ratio: these are v3's report-figure and markdown
    # builders, the surface a retarget would necessarily have carried over. The
    # first assertion keeps the second honest if v3 is ever renamed.
    v3_only = {
        "_save_energy_variance_scatter",
        "_save_cusp_winner_grid",
        "_save_architecture_normalization_line_grid",
        "_report_markdown",
    }
    assert v3_only <= _defined_names(original), sorted(v3_only - _defined_names(original))
    assert not (v3_only & _defined_names(forked))


def _live_strings_and_identifiers(path: Path) -> set[str]:
    """Return a module's non-docstring string constants and identifier names.

    Comments are absent from the AST and module/class/function docstrings are
    excluded deliberately: a comment or docstring that explains WHY a v3 metric
    was dropped is documentation, not a live reference to it. What must not
    survive is a metric name the code still reads, writes, or iterates.
    """

    tree = ast.parse(path.read_text())
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            names.add(node.value)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


@pytest.mark.parametrize(
    "absent",
    [
        "feature_trace_stability",
        "readout_trace_stability",
        "spatial_exchange_symmetry",
        "rotation_consistency",
        "eval/energy/",
        "update_normalization",
        "feature_normalization",
        "basis_update",
    ],
)
def test_no_forked_module_names_a_metric_or_axis_the_tpen_suite_lacks(absent):
    # Test modules are excluded: they must be able to name a dropped metric in
    # order to assert it is gone.
    offenders = [
        str(path.relative_to(STUDY_DIR))
        for path in _study_python_files()
        if not path.name.startswith("test_")
        and any(absent in name for name in _live_strings_and_identifiers(path))
    ]
    assert offenders == []


def test_the_dropped_metric_scan_would_notice_a_live_reference():
    # Guards the parametrized test above: it must be able to see a name that IS
    # live in the study, otherwise its empty result proves nothing.
    live = _live_strings_and_identifiers(STUDY_DIR / "final_collect.py")
    assert any("eval/mcmc_energy/" in name for name in live)
    assert "full_model_antisymmetry" in live


def test_final_collect_projects_exactly_the_tpen_eval_suite():
    assert final_collect.SYMMETRY_TASKS == ("full_model_antisymmetry",)
    assert final_collect.TRACE_TASKS == ("trace_equivariance",)
    assert "eval/mcmc_energy/local_energy_mean" in final_collect.EVAL_EXACT_METRICS
    assert not [
        metric for metric in final_collect.EVAL_EXACT_METRICS if metric.startswith("eval/energy/")
    ]


def test_final_collect_report_columns_are_named_for_the_axes_they_hold():
    # v3's `basis_class` / `normalization` headers would carry basis and
    # activation values here, which is exactly the mislabelling that bites at
    # write-up time.
    assert final_collect.REPORT_ROW_COLUMN == "report_row"
    assert final_collect.REPORT_COL_COLUMN == "report_col"
    for columns in (
        final_collect.RUN_INDEX_COLUMNS,
        final_collect.ENERGY_BY_RUN_COLUMNS,
        final_collect.ARCHITECTURE_SUMMARY_COLUMNS,
        final_collect.TRACE_COLUMNS,
    ):
        assert "report_row" in columns
        assert "report_col" in columns
        assert "basis_class" not in columns
        assert "normalization" not in columns
        assert "basis_update" not in columns


def test_report_axes_come_from_the_recorded_grid_not_from_hardcoded_names():
    manifest = {"major_axes": ["basis", "activation"], "minor_axes": ["lr", "channels"]}
    assert final_collect._report_axes(manifest) == ("basis", "activation")
    # One major axis falls back to the first minor axis for the column key.
    assert final_collect._report_axes({"major_axes": ["basis"], "minor_axes": ["lr"]}) == ("basis", "lr")


def test_deprecated_smoke_message_points_at_this_study_not_the_v3_one():
    assert "tpen-pair-scan-v1" in launch.DEPRECATED_SMOKE_MESSAGE
    assert "pair_stability_v3" not in launch.DEPRECATED_SMOKE_MESSAGE
