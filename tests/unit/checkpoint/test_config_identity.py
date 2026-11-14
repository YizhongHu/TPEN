"""Golden and metamorphic tests for the public resolved-config identity."""

from __future__ import annotations

import copy
import importlib
import inspect
import pkgutil
from collections.abc import Iterable
from pathlib import Path

import pytest
from experiments.toolkit.config_identity import expected_resolved_config_identity
from omegaconf import OmegaConf

from tpen.checkpoint.hashing import (
    _CALLBACK_OUTPUT_FIELDS,
    _LOGGER_PATH_FIELDS,
    _NON_OPERATIONAL_PATH_FIELDS,
    RESOLVED_CONFIG_IDENTITY_SCHEMA,
    ResolvedConfigIdentityError,
    _delete_known_operational_slots,
    _identity_resolved_container,
    canonical_resolved_config_bytes,
    resolved_config_identity,
)
from tpen.checkpoint.operational_slots import OPERATIONAL_SLOT_MANIFEST


_REPO_ROOT = Path(__file__).resolve().parents[3]

_PATH_LIKE_PARAMETER_NAMES = frozenset(
    {
        "artifact_dir",
        "dir",
        "directory",
        "file",
        "filename",
        "log_dir",
        "output_dir",
        "output_path",
        "path",
        "root",
    }
)
_CALLBACK_LOGGER_ALIAS_TARGETS = {
    "tpen.callback.RawTrainingBatchRecorder": (
        "tpen.callback.TrainingBatchRecorder",
    ),
}


def _public_module_names(package_name: str, package: object) -> tuple[str, ...]:
    """Return the package and every importable submodule below it."""

    module_names = [package_name]
    package_path = getattr(package, "__path__", None)
    if package_path is not None:
        module_names.extend(
            module_info.name
            for module_info in pkgutil.walk_packages(
                package_path, package_name + "."
            )
        )
    return tuple(dict.fromkeys(module_names))


def _public_export_names(module: object) -> tuple[str, ...]:
    """Include normal, ``__all__``, and lazy ``__getattr__`` exports."""

    names: set[str] = {
        name for name in dir(module) if not name.startswith("_")
    }
    exported = getattr(module, "__all__", ())
    if isinstance(exported, Iterable) and not isinstance(exported, (str, bytes)):
        names.update(name for name in exported if isinstance(name, str))
    return tuple(sorted(names))


def _callback_logger_path_fields(
    cls: type[object], roots: tuple[type[object], ...]
) -> set[str]:
    """Return path-like constructor parameters for one public class."""

    if cls in roots or not any(issubclass(cls, root) for root in roots):
        return set()
    parameters = inspect.signature(cls.__init__).parameters.values()
    return {
        parameter.name
        for parameter in parameters
        if parameter.name in _PATH_LIKE_PARAMETER_NAMES
    }


def _callback_logger_path_slots() -> dict[str, set[str]]:
    """Enumerate constructor path slots from the callback/logger packages."""

    from tpen.callback import Callback, StatefulCallback
    from tpen.logging import Logger

    packages = ("tpen.callback", "tpen.logging")
    discovered: dict[str, set[str]] = {}
    discovered_classes: dict[str, type[object]] = {}

    def record(target: str, cls: type[object], fields: set[str]) -> None:
        prior_cls = discovered_classes.setdefault(target, cls)
        if prior_cls is not cls:
            raise AssertionError(
                f"conflicting callback/logger classes for target {target}: "
                f"{prior_cls!r} != {cls!r}"
            )
        prior = discovered.setdefault(target, fields)
        if prior != fields:
            raise AssertionError(
                f"conflicting path fields for callback/logger target {target}: "
                f"{prior} != {fields}"
            )

    for package_name in packages:
        package = importlib.import_module(package_name)
        module_names = _public_module_names(package_name, package)
        for module_name in module_names:
            module = importlib.import_module(module_name)
            class_names = _public_export_names(module)
            module_exports = set(getattr(module, "__all__", ()))
            for class_name in class_names:
                try:
                    cls = getattr(module, class_name)
                except AttributeError:
                    continue
                if not inspect.isclass(cls):
                    continue
                fields = _callback_logger_path_fields(
                    cls, (Callback, StatefulCallback, Logger)
                )
                if not fields or (
                    module_name != package_name
                    and cls.__module__ != module_name
                    and class_name not in module_exports
                ):
                    continue
                target = f"{module_name}.{class_name}"
                record(target, cls, fields)

    expanded = dict(discovered)
    for target, aliases in _CALLBACK_LOGGER_ALIAS_TARGETS.items():
        if target in discovered:
            expanded.update({alias: set(discovered[target]) for alias in aliases})
    for target, fields in tuple(expanded.items()):
        if target.count(".") == 2 and (
            target.startswith("tpen.callback.") or target.startswith("tpen.logging.")
        ):
            expanded.setdefault("spenn." + target.removeprefix("tpen."), set(fields))
    return expanded


def _assert_callback_logger_path_table_closed(
    discovered: dict[str, set[str]], declared: dict[str, set[str]]
) -> None:
    """Reject missing classifications and stale table rows."""

    missing = {
        target: fields
        for target, fields in discovered.items()
        if target not in declared or declared[target] != fields
    }
    extra = {
        target: fields
        for target, fields in declared.items()
        if target not in discovered or fields != discovered[target]
    }
    assert not missing, f"unclassified callback/logger path slots: {missing}"
    assert not extra, f"closed table has stale/extra path slots: {extra}"


def _declared_callback_logger_path_slots() -> dict[str, set[str]]:
    """Return the operational and explicitly non-operational closed lists."""

    declared: dict[str, set[str]] = {
        target: set(fields)
        for target, fields in {**_CALLBACK_OUTPUT_FIELDS, **_LOGGER_PATH_FIELDS}.items()
    }
    declared.update(
        {
            target: set(fields)
            for target, fields in _NON_OPERATIONAL_PATH_FIELDS.items()
        }
    )
    return declared

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


def test_callback_logger_path_census_has_no_unclassified_slots() -> None:
    """Every callback/logger destination is operational or named as science."""

    _assert_callback_logger_path_table_closed(
        _callback_logger_path_slots(), _declared_callback_logger_path_slots()
    )


def test_callback_logger_path_census_rejects_unclassified_path_like_class() -> None:
    """The census guard must fail for a newly discovered path-like class."""

    discovered = _callback_logger_path_slots()
    discovered["tpen.callback.review_fixture.UnclassifiedPathCallback"] = {"path"}
    with pytest.raises(AssertionError, match="unclassified callback/logger path slots"):
        _assert_callback_logger_path_table_closed(
            discovered, _declared_callback_logger_path_slots()
        )


def test_callback_logger_path_census_keeps_package_and_module_targets() -> None:
    """Lazy package aliases and concrete module targets remain exact targets."""

    discovered = _callback_logger_path_slots()
    expected = {
        "tpen.callback.ParameterActivity",
        "tpen.callback.parameter_activity.ParameterActivity",
        "tpen.callback.RawTrainingBatchRecorder",
        "tpen.callback.raw_training_batch_recorder.RawTrainingBatchRecorder",
        "tpen.callback.raw_training_batch_recorder.TrainingBatchRecorder",
        "tpen.callback.TrainingBatchRecorder",
    }
    assert expected <= discovered.keys()
    assert discovered["tpen.callback.ParameterActivity"] == {"output_path"}
    assert discovered["tpen.callback.parameter_activity.ParameterActivity"] == {
        "output_path"
    }


def test_callback_logger_path_census_finds_synthetic_alias_and_submodule_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public package or submodule alias cannot evade the census."""

    callback_package = importlib.import_module("tpen.callback")
    callback_module = importlib.import_module("tpen.callback.parameter_activity")
    parameter_activity = callback_package.ParameterActivity
    submodule_alias = type(
        "ReviewSubmoduleAlias",
        (parameter_activity,),
        {"__module__": callback_module.__name__},
    )
    monkeypatch.setattr(callback_package, "ReviewAlias", parameter_activity, raising=False)
    monkeypatch.setattr(
        callback_package,
        "__all__",
        (*callback_package.__all__, "ReviewAlias"),
    )
    monkeypatch.setattr(
        callback_module,
        "ReviewSubmoduleAlias",
        submodule_alias,
        raising=False,
    )

    discovered = _callback_logger_path_slots()
    assert discovered["tpen.callback.ReviewAlias"] == {"output_path"}
    assert discovered["tpen.callback.parameter_activity.ReviewSubmoduleAlias"] == {
        "output_path"
    }


def test_callback_logger_path_table_is_closed_and_non_operational_reasons_exist() -> None:
    """The closed table cannot silently gain a science field or stale target."""

    assert _declared_callback_logger_path_slots() == _callback_logger_path_slots()
    assert all(
        reason.strip()
        for fields in _NON_OPERATIONAL_PATH_FIELDS.values()
        for reason in fields.values()
    )


def test_operational_slot_manifest_derives_identity_and_routing_registries() -> None:
    """Hashing and routing must consume the same owner-defined manifest."""

    from tpen.run import _ARTIFACT_SINK_FIELDS, _SINK_FREE_TYPES

    assert _CALLBACK_OUTPUT_FIELDS == {
        target: set(slot.fields)
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.kind == "callback" and slot.fields
    }
    assert _LOGGER_PATH_FIELDS == {
        target: set(slot.fields)
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.kind == "logger" and slot.fields
    }
    assert _NON_OPERATIONAL_PATH_FIELDS == {
        target: dict(slot.non_operational_reasons)
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.non_operational_reasons
    }
    assert _ARTIFACT_SINK_FIELDS == {
        target: slot.fields
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.routed and slot.fields
    }
    assert _SINK_FREE_TYPES == frozenset(
        target
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.routed and slot.sink_free
    )


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


@pytest.mark.parametrize(
    ("relative_path", "expected_sha256"),
    [
        # The local worker environment cannot resolve the locked dependency
        # set.  The orchestrator must replace this sentinel with the exact
        # base-commit digest from a Cannon probe before independent review.
        (
            "experiments/atomistic/he-importance/configs/train.yaml",
            None,
        ),
    ],
)
def test_he_importance_identity_golden_from_base_commit(
    relative_path: str, expected_sha256: str | None
) -> None:
    """The real He-importance config keeps its v1 identity byte-identical."""

    if expected_sha256 is None:
        pytest.skip("Cannon must fill the base-commit golden before review")
    config = _guard_config(_REPO_ROOT / relative_path)
    assert resolved_config_identity(config) == {
        "schema": RESOLVED_CONFIG_IDENTITY_SCHEMA,
        "sha256": expected_sha256,
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
