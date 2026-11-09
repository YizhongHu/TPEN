"""Independent P6 core behavior and import-boundary probes."""

from __future__ import annotations

import ast
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from tpen.checkpoint import profiles
from tpen.execution import native_profile

PROFILE = "tpen.native-recovery/v1"
ROOT = Path(__file__).parents[3]


def _outcome(validator, config):
    """Capture exact public error class name and text or a normal return."""
    try:
        value = validator(config)
    except Exception as exc:
        return type(exc).__name__, str(exc)
    return "returned", value


LEGACY_INPUTS = [
    None, [], "bad", {},
    {"callbacks": None}, {"callbacks": []}, {"callbacks": ()},
    {"callbacks": ""}, {"callbacks": False}, {"callbacks": 0},
    {"callbacks": {}}, {"callbacks": [None]},
    {"callbacks": [{}]}, {"callbacks": [{"_target_": ""}]},
    {"callbacks": [{"_target_": 7}]},
    {"callbacks": [{"_target_": "review_forbidden.PrivateState", "state": 7}]},
    {"callbacks": [{"_target_": "tpen.callback.Status"}]},
    {"loggers": [{"_target_": "tpen.logging.JSONL"}]},
    {"loggers": [{"_target_": "review_forbidden.Logger"}]},
    {"loggers": "not-a-sequence"},
]


# Frozen once from the predecessor validator at 5ba0db26; no repository or
# subprocess lookup is needed when this test runs.
LEGACY_EXPECTED_OUTCOMES = [
    ("UnsupportedMutableStateError", "native profile requires a mapping config"),
    ("UnsupportedMutableStateError", "native profile requires a mapping config"),
    ("UnsupportedMutableStateError", "native profile requires a mapping config"),
    ("returned", None),
    ("returned", None),
    ("returned", None),
    ("returned", None),
    ("returned", None),
    ("returned", None),
    ("returned", None),
    ("returned", None),
    ("UnsupportedMutableStateError", "callbacks[0] must be a mapping"),
    ("UnsupportedMutableStateError", "callbacks[0] has no explicit _target_"),
    ("UnsupportedMutableStateError", "callbacks[0] has no explicit _target_"),
    ("UnsupportedMutableStateError", "callbacks[0] has no explicit _target_"),
    (
        "UnsupportedMutableStateError",
        "callbacks[0] 'review_forbidden.PrivateState' is not in the native "
        "mutable-state profile",
    ),
    ("returned", None),
    ("returned", None),
    (
        "UnsupportedMutableStateError",
        "loggers[0] 'review_forbidden.Logger' is not in the native "
        "mutable-state profile",
    ),
    ("UnsupportedMutableStateError", "loggers must be a sequence"),
]


@pytest.mark.parametrize("config, expected", zip(LEGACY_INPUTS, LEGACY_EXPECTED_OUTCOMES))
def test_core_t1_existing_validator_preserves_exact_outcomes(config, expected):
    assert _outcome(native_profile.validate_native_recovery_profile, config) == expected


def test_core_t2_constants_facade_and_chain_resume_remain_exact():
    import tpen.execution as facade
    from tpen.execution import chain_resume

    for key in ("SUPPORTED_CALLBACK_TARGETS", "SUPPORTED_LOGGER_TARGETS", "SUPPORTED_STATE_OWNERS"):
        assert getattr(native_profile, key) is getattr(profiles, key)
        assert getattr(facade, key) is getattr(profiles, key)
    assert facade.UnsupportedMutableStateError is native_profile.UnsupportedMutableStateError
    assert facade.validate_native_recovery_profile is native_profile.validate_native_recovery_profile
    assert chain_resume.NATIVE_PROFILE_VERSION == "tpen.execution.native_profile:SUPPORTED_STATE_OWNERS@e5"
    assert profiles.SUPPORTED_CALLBACK_TARGETS == frozenset({
        "tpen.callback.ConfigSnapshot",
        "tpen.callback.ResolvedConfigSnapshot",
        "tpen.callback.Metadata",
        "tpen.callback.RunTiming",
        "tpen.callback.TrainStepTiming",
        "tpen.callback.TrainPhaseTiming",
        "tpen.callback.Status",
        "tpen.callback.Checkpoint",
        "tpen.callback.DataIntegrity",
        "tpen.callback.GradientStats",
        "tpen.callback.ElectronNucleusDistanceQuantiles",
    })
    assert profiles.SUPPORTED_LOGGER_TARGETS == frozenset({
        "tpen.logging.CSV", "tpen.logging.JSONL",
    })
    assert profiles.SUPPORTED_STATE_OWNERS == frozenset({
        "model", "optimizer", "trainer_cursor", "sampler_mcmc", "process_rng",
    })
    assert profiles.NATIVE_PROFILE.callback_state_supported is False
    assert profiles.REFUSAL_CODES == {
        "unknown-profile", "unsupported-callback", "unsupported-callback-cadence",
        "unsupported-logger", "malformed-config",
    }
    assert not [n for n in ast.walk(ast.parse((ROOT / "tpen/checkpoint/profiles.py").read_text()))
                if isinstance(n, ast.Constant) and type(n.value) is float]


@pytest.mark.parametrize("first", [
    "tpen.checkpoint", "tpen.checkpoint.profiles",
    "tpen.execution.native_profile", "tpen.execution.chain_resume",
])
def test_core_t3_fresh_import_orders_have_no_cycle_or_executor_dependency(first):
    code = """
import importlib, importlib.abc, sys, time, json
class DenyExecutor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tpen_executor' or fullname.startswith('tpen_executor.'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, DenyExecutor())
started = time.perf_counter()
importlib.import_module(sys.argv[1])
from tpen.checkpoint.profiles import SUPPORTED_STATE_OWNERS as owners
from tpen.execution.native_profile import SUPPORTED_STATE_OWNERS as native
from tpen.execution import NATIVE_PROFILE_VERSION
assert owners is native
print(json.dumps({'first': sys.argv[1], 'seconds': time.perf_counter()-started,
                 'modules': len(sys.modules), 'torch_loaded': 'torch' in sys.modules}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, first], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    print(result.stdout.strip())


@pytest.mark.parametrize("config,code", [
    (None, "malformed-config"), ([], "malformed-config"),
    ({"callbacks": False}, "malformed-config"),
    ({"callbacks": {}}, "malformed-config"),
    ({"callbacks": "bad"}, "malformed-config"),
    ({"callbacks": [None]}, "malformed-config"),
    ({"callbacks": [{}]}, "malformed-config"),
    ({"callbacks": [{"_target_": 1}]}, "malformed-config"),
    ({"callbacks": [{"_target_": "review_forbidden.PrivateState", "state": {"counter": 2}}]}, "unsupported-callback"),
    ({"loggers": [{"_target_": "review_forbidden.Logger"}]}, "unsupported-logger"),
])
def test_core_t4_new_checker_refuses_malformed_and_private_state(config, code):
    result = profiles.check_profile(config, PROFILE)
    assert result.verdict == "refused"
    assert result.refusal["code"] == code
    assert result.state_owners == profiles.SUPPORTED_STATE_OWNERS


def test_core_t5_omegaconf_syntax_never_resolves_or_imports_targets(monkeypatch):
    calls = []
    OmegaConf.register_new_resolver("review_p6", lambda *args: calls.append(args), replace=True)
    class DenyTarget(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("review_forbidden"):
                raise AssertionError(fullname)
    finder = DenyTarget()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    try:
        config = OmegaConf.create({
            "callbacks": [{"_target_": "tpen.callback.Status", "path": "${review_p6:argument}"}],
            "loggers": [{"_target_": "tpen.logging.JSONL"}],
            "model": {"_target_": "review_forbidden.Constructor", "value": "${review_p6:model}"},
        })
        assert profiles.check_profile(config, PROFILE).verdict == "supported"
        config.callbacks[0]._target_ = "review_forbidden.PrivateState"
        assert profiles.check_profile(config, PROFILE).refusal["code"] == "unsupported-callback"
        config.callbacks = "${review_p6:callbacks}"
        assert profiles.check_profile(config, PROFILE).refusal["code"] == "malformed-config"
        assert calls == []
    finally:
        OmegaConf.clear_resolver("review_p6")


@pytest.mark.parametrize("text,expected", [
    ("{}\n", "supported"),
    ("callbacks: [\n", "malformed-config"),
    ("callbacks: false\n", "malformed-config"),
    ("callbacks:\n  - _target_: review_forbidden.Callback\n", "unsupported-callback"),
    ("loggers:\n  - _target_: review_forbidden.Logger\n", "unsupported-logger"),
])
def test_core_t6_public_module_cli_emits_one_typed_record(tmp_path, text, expected):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    result = subprocess.run(
        [sys.executable, "-m", "tpen.checkpoint.profiles", str(path), "--profile", PROFILE],
        cwd=ROOT, text=True, capture_output=True, timeout=30,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0, result.stderr
    record = profiles.ProfileCheck.from_mapping(json.loads(result.stdout))
    assert record.profile == PROFILE
    if expected == "supported":
        assert record.verdict == "supported" and record.refusal is None
    else:
        assert record.verdict == "refused" and record.refusal["code"] == expected


def test_core_t7_unknown_profile_and_missing_config_are_refusals(tmp_path):
    path = tmp_path / "missing.yaml"
    for profile, code in ((PROFILE, "malformed-config"), ("review.unknown/v1", "unknown-profile")):
        result = subprocess.run(
            [sys.executable, "-m", "tpen.checkpoint.profiles", str(path), "--profile", profile],
            cwd=ROOT, text=True, capture_output=True, timeout=30,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["refusal"]["code"] == code
