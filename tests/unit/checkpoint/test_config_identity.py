"""Golden and metamorphic tests for the public resolved-config identity."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from experiments.toolkit.config_identity import expected_resolved_config_identity
from omegaconf import OmegaConf

from tpen.checkpoint.hashing import (
    RESOLVED_CONFIG_IDENTITY_SCHEMA,
    ResolvedConfigIdentityError,
    _delete_known_operational_slots,
    _identity_resolved_container,
    canonical_resolved_config_bytes,
    resolved_config_identity,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]

# These are checked-in study inputs, not standalone identity-bound run
# configurations.  Keep this list closed and named: removing an entry is a
# deliberate follow-up when the corresponding study composition is repaired.
_IDENTITY_CONFIG_EXCLUSIONS = {
    "experiments/hooke/pair_stability_v3/configs/pair_stability.yaml": (
        "legacy SpENN targets are unavailable in the dev checkout; the real run path "
        "fails during full-tree resolution before identity is written"
    ),
    "experiments/hooke/pair_stability_v3/configs/pair_validation.yaml": (
        "legacy SpENN targets are unavailable in the dev checkout; the real run path "
        "fails during full-tree resolution before identity is written"
    ),
    "experiments/hooke/tpen-pair-scan-v1/configs/grid.yaml": (
        "launcher metadata, not a tpen run config; its basis merge has no system.omega"
    ),
    "experiments/hooke/tpen-pair-scan-v1/configs/smoke.yaml": (
        "launcher metadata, not a tpen run config; its basis merge has no system.omega"
    ),
}


def _identity_config_paths() -> tuple[Path, ...]:
    """Return every checked-in config tree covered by the identity boundary."""

    roots = (
        _REPO_ROOT / "tests" / "integration" / "artifacts",
        _REPO_ROOT / "configs",
    )
    paths = {path for root in roots if root.is_dir() for path in root.rglob("*.yaml")}
    paths.update(_REPO_ROOT.glob("experiments/**/configs/**/*.yaml"))
    paths.update(_REPO_ROOT.glob("experiments/**/configs/**/*.yml"))
    return tuple(
        sorted(
            path
            for path in paths
            if str(path.relative_to(_REPO_ROOT)) not in _IDENTITY_CONFIG_EXCLUSIONS
        )
    )


def _guard_config(config_path: Path) -> dict[str, object]:
    """Materialize each checked-in file as its real launch boundary does.

    The scan intentionally includes config fragments that are not runnable as
    standalone YAML: the pair scan merges the shared basis table, and the He
    evaluator receives its trajectory/checkpoint identity from ``eval.py``.
    Missing provider/run inputs are replaced with deterministic guard values so
    this test exercises the identity boundary rather than unrelated Hydra
    composition failures.
    """

    config = OmegaConf.load(config_path)
    if "tpen-pair-scan-v1" in config_path.parts or config_path.name == "basis_levels_train.yaml":
        basis = OmegaConf.load(_REPO_ROOT / "experiments" / "hooke" / "choices" / "basis_levels.yaml")
        config = OmegaConf.merge(config, basis)
    plain = OmegaConf.to_container(config, resolve=False, throw_on_missing=False)

    def replace_missing(value: object, path: tuple[object, ...] = ()) -> object:
        if isinstance(value, dict):
            return {key: replace_missing(item, path + (key,)) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_missing(item, path + (index,)) for index, item in enumerate(value)]
        if value == "???":
            dotted = ".".join(str(part) for part in path)
            return f"identity-guard-missing:{dotted}"
        return value

    if not isinstance(plain, dict):
        raise TypeError(f"config guard expected a mapping: {config_path}")
    return replace_missing(plain)  # type: ignore[return-value]


def _config() -> dict[str, object]:
    return {
        "run": {
            "root": "/tmp/a",
            "run_id": "x",
            "dir": "/tmp/a/x",
            "layout": "layout-a",
            "timezone": "America/New_York",
        },
        "callbacks": [
            {"_target_": "tpen.callback.ConfigSnapshot", "output_path": "/tmp/a/x/cb.json", "name": "save"},
            {"_target_": "tpen.callback.Status", "output_path": "/tmp/a/x/other.json", "name": "other"},
        ],
        "loggers": [{"_target_": "tpen.logging.JSONL", "path": "/tmp/a/x/log.txt", "name": "main"}],
        "evaluator": {"_target_": "tpen.evaluation.Evaluator", "tasks": [{"output_dir": "/tmp/a/x/trajectory", "metric": "energy"}]},
        "model": {"layers": 4},
        "science": {"charge": 2},
    }


@pytest.mark.parametrize(
    "config_path",
    _identity_config_paths(),
    ids=lambda path: str(path.relative_to(_REPO_ROOT)),
)
def test_identity_accepts_every_in_repo_config(config_path: Path) -> None:
    """Operational run paths in checked-in configs must never abort launch.

    This is deliberately a broad guard over the config trees used by
    integration artifacts and experiment studies.  Genuine science leaks stay
    covered by the focused refusal tests below; this guard prevents a closed
    operational-slot amendment from regressing a real run at startup.
    """

    config = _guard_config(config_path)
    assert resolved_config_identity(config)["schema"] == RESOLVED_CONFIG_IDENTITY_SCHEMA
    assert expected_resolved_config_identity(config)["schema"] == RESOLVED_CONFIG_IDENTITY_SCHEMA


def test_identity_config_exclusions_are_closed_and_explicit() -> None:
    """Stale/metadata inputs cannot silently become identity test coverage."""

    expected = {
        "experiments/hooke/pair_stability_v3/configs/pair_stability.yaml",
        "experiments/hooke/pair_stability_v3/configs/pair_validation.yaml",
        "experiments/hooke/tpen-pair-scan-v1/configs/grid.yaml",
        "experiments/hooke/tpen-pair-scan-v1/configs/smoke.yaml",
    }
    assert set(_IDENTITY_CONFIG_EXCLUSIONS) == expected
    assert all(reason for reason in _IDENTITY_CONFIG_EXCLUSIONS.values())
    covered = {str(path.relative_to(_REPO_ROOT)) for path in _identity_config_paths()}
    assert covered.isdisjoint(expected)


def test_identity_is_sensitive_to_resolved_resolver_values() -> None:
    """Resolved resolver output is part of the public identity contract."""

    value = ["first"]
    OmegaConf.register_new_resolver(
        "identity_probe",
        lambda: value[0],
        replace=True,
    )
    try:
        first = resolved_config_identity(
            OmegaConf.create({"science": {"value": "${identity_probe:}"}})
        )
        value[0] = "second"
        second = resolved_config_identity(
            OmegaConf.create({"science": {"value": "${identity_probe:}"}})
        )
    finally:
        OmegaConf.clear_resolver("identity_probe")
    assert first["schema"] == RESOLVED_CONFIG_IDENTITY_SCHEMA
    assert second["schema"] == RESOLVED_CONFIG_IDENTITY_SCHEMA
    assert first != second


def test_golden_identity_vector() -> None:
    assert canonical_resolved_config_bytes(_config()) == (
        b'{"callbacks":[{"_target_":"tpen.callback.ConfigSnapshot","name":"save"},'
        b'{"_target_":"tpen.callback.Status","name":"other"}],'
        b'"evaluator":{"_target_":"tpen.evaluation.Evaluator","tasks":[{"metric":"energy"}]},'
        b'"experiment":{"run_name":"tpen_run"},'
        b'"loggers":[{"_target_":"tpen.logging.JSONL","name":"main"}],"model":{"layers":4},"run":{},'
        b'"science":{"charge":2}}'
    )
    assert resolved_config_identity(_config()) == {
        "schema": RESOLVED_CONFIG_IDENTITY_SCHEMA,
        "sha256": "b58ece8d8f791a7954277ae4879695fb1fa196b8badd56539b891e996d46fc6a",
    }


def test_identity_is_invariant_to_all_declared_run_fields_and_paths() -> None:
    original = resolved_config_identity(_config())
    changed = _config()
    changed["run"] = {
        "root": "/different/root",
        "run_id": "another-run",
        "dir": "/different/root/another-run",
        "layout": "different-layout",
        "timezone": "UTC",
    }
    changed["callbacks"] = [
        {"_target_": "tpen.callback.ConfigSnapshot", "output_path": "/elsewhere/callback.json", "name": "save"},
        {"_target_": "tpen.callback.Status", "name": "other", "output_path": "/elsewhere/other.json"},
    ]
    changed["loggers"] = [{"_target_": "tpen.logging.JSONL", "path": "/elsewhere/log.txt", "name": "main"}]
    changed["evaluator"] = {"_target_": "tpen.evaluation.Evaluator", "tasks": [{"output_dir": "/elsewhere/trajectory", "metric": "energy"}]}
    assert resolved_config_identity(changed) == original


def test_identity_keeps_unproduced_run_like_fields_science_relevant() -> None:
    base = _config()
    changed = copy.deepcopy(base)
    changed["run"]["timestamp"] = "later"  # type: ignore[index]
    changed["run"]["start_time_unix"] = 99.0  # type: ignore[index]
    assert resolved_config_identity(changed) != resolved_config_identity(base)


def test_identity_changes_for_each_science_field() -> None:
    base = _config()
    for path in (("model", "layers"), ("science", "charge"), ("evaluator", "tasks", 0, "metric")):
        changed = copy.deepcopy(base)
        cursor: object = changed
        for key in path[:-1]:
            cursor = cursor[key]  # type: ignore[index]
        cursor[path[-1]] = "changed"  # type: ignore[index]
        assert resolved_config_identity(changed) != resolved_config_identity(base), path


def test_identity_resolves_run_derived_evaluator_copies_before_scrubbing() -> None:
    intended = {
        "experiment": {"name": "eval"},
        "run": {"dir": None, "run_id": None},
        "runner": {"_target_": "tpen.runner.Evaluate", "evaluator": "${evaluator}"},
        "evaluator": {
            "_target_": "tpen.evaluation.Evaluator",
            "tasks": [{"name": "trajectory", "output_dir": "${run.dir}/trajectory", "metric": "energy"}],
        },
        "science": {"charge": 1},
    }
    actual = copy.deepcopy(intended)
    actual["run"] = {"dir": "/different/run", "run_id": "different"}
    prepared = _identity_resolved_container(intended)
    assert prepared["runner"]["evaluator"]["tasks"][0]["output_dir"] == (
        "__tpen_identity_run_sentinel_7b1e4c93f0a64b9d__:dir__/trajectory"
    )
    assert resolved_config_identity(intended) == resolved_config_identity(actual)


def test_identity_applies_core_run_name_default_on_both_sides() -> None:
    intended = {"experiment": {"name": "eval"}, "science": {"charge": 1}}
    actual = {"experiment": {"name": "eval", "run_name": "eval"}, "run": {"run_id": "id", "dir": "/tmp/id"}, "science": {"charge": 1}}
    assert resolved_config_identity(intended) == resolved_config_identity(actual)


def test_identity_handles_tuple_operational_slots_like_lists() -> None:
    listed = _config()
    tupled = copy.deepcopy(listed)
    tupled["callbacks"] = tuple(tupled["callbacks"])  # type: ignore[assignment]
    tupled["loggers"] = tuple(tupled["loggers"])  # type: ignore[assignment]
    listed_scrubbed = copy.deepcopy(listed)
    tupled_scrubbed = copy.deepcopy(tupled)
    _delete_known_operational_slots(listed_scrubbed)
    _delete_known_operational_slots(tupled_scrubbed)
    assert list(tupled_scrubbed["callbacks"]) == listed_scrubbed["callbacks"]
    assert list(tupled_scrubbed["loggers"]) == listed_scrubbed["loggers"]
    assert resolved_config_identity(tupled) == resolved_config_identity(listed)


@pytest.mark.parametrize(
    "identity_function",
    [resolved_config_identity, expected_resolved_config_identity],
    ids=["core", "study"],
)
@pytest.mark.parametrize(
    "config",
    [
        {"science": {"seed": "${run.run_id}"}},
        {"science": {"inputs": ["${run.dir}/potential.dat"]}},
        {
            "callbacks": [
                {
                    "_target_": "tpen.callback.ConfigSnapshot",
                    "output_path": "${run.dir}/config.yaml",
                }
            ],
            "science": {"input": "${callbacks.0.output_path}"},
        },
    ],
    ids=["direct", "list-element", "indirect-chain"],
)
def test_identity_rejects_run_field_interpolation_into_science(
    identity_function, config: dict[str, object]
) -> None:
    with pytest.raises(ResolvedConfigIdentityError):
        identity_function(config)


def test_run_derived_science_collision_configs_are_refused() -> None:
    configs = [
        {"science": {"value": "${run.dir}/a"}},
        {"science": {"value": "${run.dir}/b"}},
    ]
    for config in configs:
        with pytest.raises(ResolvedConfigIdentityError):
            resolved_config_identity(config)
        with pytest.raises(ResolvedConfigIdentityError):
            expected_resolved_config_identity(config)


@pytest.mark.parametrize(
    ("intended", "run_side"),
    [("  padded-run  ", "padded-run"), (12345, "12345")],
)
def test_identity_matches_core_run_name_normalization(intended: object, run_side: str) -> None:
    base = {"experiment": {"name": "eval", "run_name": intended}, "science": {"charge": 1}}
    normalized = {"experiment": {"name": "eval", "run_name": run_side}, "science": {"charge": 1}}
    assert resolved_config_identity(base) == resolved_config_identity(normalized)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_identity_rejects_nonfinite_values(value: float) -> None:
    config = {"science": {"value": value}}
    with pytest.raises(TypeError, match="non-finite"):
        resolved_config_identity(config)


def test_identity_requires_mapping_and_string_keys() -> None:
    with pytest.raises(TypeError, match="requires a mapping"):
        resolved_config_identity(["not", "a", "mapping"])
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        resolved_config_identity({1: "not-json-config"})
