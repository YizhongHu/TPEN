"""Tests for CLI override serialization, including Hydra round-trips."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser

STUDY_DIR = Path(__file__).resolve().parent

while str(STUDY_DIR) in sys.path:
    sys.path.remove(str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR))
for module_name in list(sys.modules):
    if module_name.split(".", maxsplit=1)[0] == "utils":
        del sys.modules[module_name]

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

_tpen_utils_overrides = sibling(__file__, 'utils.overrides')
axis_value_overrides = _tpen_utils_overrides.axis_value_overrides
format_override_value = _tpen_utils_overrides.format_override_value
rewrite_cli_overrides = _tpen_utils_overrides.rewrite_cli_overrides


def _parse_back(key: str, formatted: str) -> object:
    override = OverridesParser.create().parse_overrides([f"{key}={formatted}"])[0]
    return override.value()


@pytest.mark.parametrize(
    "value",
    [
        5,
        5.5,
        True,
        False,
        None,
        "hello",
        "hello world",
        [0.0],
        [1, 2, 3],
        {"a": 1, "b": 2},
    ],
)
def test_format_override_value_round_trips_through_hydra(value: object) -> None:
    formatted = format_override_value(value)

    assert _parse_back("key", formatted) == value


def test_format_override_value_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError, match="Unsupported override value type"):
        format_override_value(object())


def test_rewrite_cli_overrides_replaces_exact_key_matches() -> None:
    command = ["python", "train.py", "training.max_steps=2000", "sampler_params.n_walkers=1024"]

    rewritten = rewrite_cli_overrides(command, {"training.max_steps": 2, "new.key": True})

    assert rewritten == [
        "python",
        "train.py",
        "sampler_params.n_walkers=1024",
        "training.max_steps=2",
        "new.key=true",
    ]


def test_rewrite_cli_overrides_bool_and_list_values_parse_correctly_in_hydra() -> None:
    command = ["python", "train.py"]
    overrides = {
        "checks.enabled": False,
        "evaluation_tasks.final_cusp.generator.center_of_mass_radii": [0.0],
    }

    rewritten = rewrite_cli_overrides(command, overrides)

    for part in rewritten[2:]:
        key, _, formatted = part.partition("=")
        assert _parse_back(key, formatted) == overrides[key]


def test_axis_value_overrides_support_stage_specific_and_mapped_specs() -> None:
    point = {
        "max_steps": 100,
        "mechanism": "feature_gaussian_norm",
        "channels": 8,
    }
    specs = {
        "max_steps": {
            "train": "training.max_steps",
            "final_train": "training.max_steps",
        },
        "mechanism": {
            "run_parameters.update_normalization_slot": {
                "baseline": "no-update-normalization",
                "feature_gaussian_norm": "no-update-normalization",
                "update_gaussian_norm": "update-gaussian-norm",
            },
            "run_parameters.feature_normalization_slot": {
                "baseline": "no-feature-normalization",
                "feature_gaussian_norm": "feature-gaussian-norm",
                "update_gaussian_norm": "no-feature-normalization",
            },
        },
        "channels": "run_parameters.channels",
    }

    train_overrides = axis_value_overrides(
        point,
        axes=("max_steps", "mechanism", "channels"),
        override_specs=specs,
        stage="train",
    )
    validation_overrides = axis_value_overrides(
        point,
        axes=("max_steps", "mechanism", "channels"),
        override_specs=specs,
        stage="validation",
    )

    assert train_overrides == [
        "training.max_steps=100",
        "run_parameters.update_normalization_slot=no-update-normalization",
        "run_parameters.feature_normalization_slot=feature-gaussian-norm",
        "run_parameters.channels=8",
    ]
    assert validation_overrides == [
        "run_parameters.update_normalization_slot=no-update-normalization",
        "run_parameters.feature_normalization_slot=feature-gaussian-norm",
        "run_parameters.channels=8",
    ]
