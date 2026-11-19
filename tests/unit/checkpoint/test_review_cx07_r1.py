"""CX-07 reviewer proposals; authored without execution at 8fe3cd9b.

The refusal and recovery-evidence cases are expected to fail at that head.
They are proposed acceptance assertions, not a measured runtime receipt.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest

from tpen.checkpoint.profiles import NATIVE_RECOVERY_PROFILE, check_profile
from tpen.execution.native_profile import (
    UnsupportedMutableStateError,
    validate_native_recovery_profile,
)


ROOT = Path(__file__).parents[3]
HI_CONFIG = ROOT / "experiments/atomistic/he-importance/configs/train.yaml"


def _hi_config():
    return OmegaConf.to_container(OmegaConf.load(HI_CONFIG), resolve=False)


def _runtime(config):
    return next(
        entry for entry in config["callbacks"]
        if entry["_target_"] == "tpen.callback.RuntimeEquivariance"
    )


def _resource_usage(config):
    return next(
        entry for entry in config["callbacks"]
        if entry["_target_"] == "tpen.callback.ResourceUsage"
    )


def _assert_refused(config, validator):
    if validator == "profile":
        assert check_profile(config, NATIVE_RECOVERY_PROFILE).verdict == "refused"
    else:
        with pytest.raises(UnsupportedMutableStateError):
            validate_native_recovery_profile(config)


@pytest.mark.parametrize("validator", ["profile", "native"])
@pytest.mark.parametrize("location", ["checker-seed", "stride", "device"])
def test_scalar_alias_cannot_hide_unaudited_constructor(validator, location):
    """An unresolved scalar-looking alias must not launder a target mapping."""
    config = _hi_config()
    if location == "checker-seed":
        # Keep HI's exact ${runtime.seed} syntax; change only its referent.
        config["runtime"]["seed"] = {"_target_": "builtins.int", "_args_": [19]}
    elif location == "stride":
        config["hidden_stride"] = {"_target_": "builtins.int", "_args_": [5]}
        _runtime(config)["every_n_steps"] = "${hidden_stride}"
    else:
        config["runtime"]["device"] = {"_target_": "builtins.str", "_args_": ["cpu"]}
    # Do not instantiate anything: admission itself must be syntax-only.
    _assert_refused(config, validator)


@pytest.mark.parametrize("validator", ["profile", "native"])
def test_checker_seed_rejects_attempt_dependent_resolver(validator):
    """A fresh allocation's environment is not a reconstructed checker seed."""
    config = _hi_config()
    _runtime(config)["checkers"][0]["seed"] = "${oc.env:CX07_CHECKER_SEED,19}"
    _assert_refused(config, validator)


@pytest.mark.parametrize("validator", ["profile", "native"])
@pytest.mark.parametrize("form", ["direct", "aliased"])
def test_allocator_device_rejects_embedded_interpolation(validator, form):
    """An interpolation anywhere in an audited device scalar is refused."""
    config = _hi_config()
    embedded = "cuda:${oc.env:CX07_DEVICE_INDEX,0}"
    if form == "direct":
        _resource_usage(config)["allocator_probe"]["device"] = embedded
    else:
        config["runtime"]["device"] = embedded
    _assert_refused(config, validator)


@pytest.mark.parametrize("validator", ["profile", "native"])
def test_allocator_device_literal_and_hi_reference_are_admitted(validator):
    """Literal devices and HI's full same-config device reference remain valid."""
    config = _hi_config()
    resource = _resource_usage(config)
    assert resource["allocator_probe"]["device"] == "${runtime.device}"

    if validator == "profile":
        assert check_profile(config, NATIVE_RECOVERY_PROFILE).verdict == "supported"
    else:
        validate_native_recovery_profile(config)

    resource["allocator_probe"]["device"] = "cuda:0"
    if validator == "profile":
        assert check_profile(config, NATIVE_RECOVERY_PROFILE).verdict == "supported"
    else:
        validate_native_recovery_profile(config)


def test_real_hi_scalar_references_remain_admitted():
    """Green control: keep HI's real runtime seed/device and routed path."""
    config = _hi_config()
    assert type(config["runtime"]["seed"]) is int
    assert isinstance(config["runtime"]["device"], str)
    assert check_profile(config, NATIVE_RECOVERY_PROFILE).verdict == "supported"
    validate_native_recovery_profile(config)


def test_recovery_fixture_observes_the_checker_step_and_selection(tmp_path, monkeypatch):
    """The reported oracle must describe actual checker calls at event steps."""
    from tests.helpers import runtime_equivariance_recovery as worker
    from tpen.equivariance import checks

    observed = []
    original = checks.select_nonidentity_permutations

    def record_selection(*args, **kwargs):
        selected = original(*args, **kwargs)
        observed.append((kwargs["step"], [list(p.image) for p in selected]))
        return selected

    monkeypatch.setattr(checks, "select_nonidentity_permutations", record_selection)
    result = worker.run("reference", tmp_path, failure=False)
    assert [step for step, _ in observed] == [
        call["step"] for call in result["selection_calls"]
    ]
    assert [
        images for _, images in observed
    ] == [call["images"] for call in result["selection_calls"]]


def test_recovery_fixture_exercises_both_checker_comparisons(tmp_path):
    """A three-particle batch alone does not make an empty trace meaningful."""
    from tests.helpers import runtime_equivariance_recovery as worker

    result = worker.run("reference", tmp_path, failure=False)
    assert {record["namespace"] for record in result["records"]} == {
        "checks/equivariance/full_model", "checks/equivariance/trace",
    }
    assert all(record["metrics"]["n_comparisons"] > 0 for record in result["records"])
