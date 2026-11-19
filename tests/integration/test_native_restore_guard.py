"""Preconstruction native-profile admission through both public restore entries."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
import hashlib
import importlib
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from experiments.atomistic.stage1.protocol import compose_train_overlay, overlay_sha256
from tpen.checkpoint import CheckpointRef, CheckpointRestored
from tpen.checkpoint.profiles import (
    UnsupportedMutableStateError, validate_native_recovery_profile,
)
from tpen.execution.recovery import (
    CheckpointEvidence, OperationalRestoreRequest, RecoveryPhase, ScienceIntent, select_recovery,
)
from tpen.training.trainer import VMCTrainer

run_module = importlib.import_module("tpen.run")
FIXTURE = Path("tests/integration/artifacts/execution/native_recovery.yaml")


def test_execution_exports_the_core_owned_validator():
    from tpen.execution import native_profile

    assert native_profile.validate_native_recovery_profile is validate_native_recovery_profile
    assert native_profile.UnsupportedMutableStateError is UnsupportedMutableStateError


def _equal(left, right, path="state"):
    assert type(left) is type(right), f"{path}: types differ"
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype, f"{path}: tensor dtypes differ"
        assert left.shape == right.shape, f"{path}: tensor shapes differ"
        assert torch.equal(left, right), f"{path}: tensor values differ"
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype, f"{path}: array dtypes differ"
        assert left.shape == right.shape, f"{path}: array shapes differ"
        assert np.array_equal(left, right), f"{path}: array values differ"
    elif is_dataclass(left) and not isinstance(left, type):
        # Walkers' generated equality cannot reduce its tensor-valued fields.
        for field in fields(left):
            _equal(getattr(left, field.name), getattr(right, field.name),
                   f"{path}.{field.name}")
    elif isinstance(left, Mapping):
        assert left.keys() == right.keys(), f"{path}: mapping keys differ"
        for key in left:
            _equal(left[key], right[key], f"{path}[{key!r}]")
    elif isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        assert len(left) == len(right), f"{path}: sequence lengths differ"
        for index, (first, second) in enumerate(zip(left, right)):
            _equal(first, second, f"{path}[{index}]")
    else:
        assert left == right, f"{path}: values differ"


def _files(root):
    return {
        str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(),
                                     path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    cfg = OmegaConf.load(FIXTURE)
    cfg.run.root = str(tmp_path / "source")
    cfg.run.run_id = "source"
    cfg.trainer.max_steps = 2
    live = {}
    original = VMCTrainer.fit

    def capture(self, **kwargs):
        live.update(kwargs)
        live["trainer"] = self
        return original(self, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(VMCTrainer, "fit", capture)
        assert run_module.run_from_config(cfg, raise_exceptions=True) == 0
    checkpoint, = (tmp_path / "source").rglob("step_000001")
    intent = ScienceIntent({"fixture": "cx09"}, target_iterations=2)
    evidence = CheckpointEvidence.from_ref(
        CheckpointRef.from_directory(checkpoint),
        science_input_digest=intent.science_input_digest, target_iterations=2,
    )
    request = OperationalRestoreRequest(
        select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    )
    cfg.run.root = str(tmp_path / "successor")
    cfg.run.run_id = "successor"
    return cfg, request, checkpoint, live


def _entry(cfg, request, entry):
    if entry == "request":
        return {"restore_request": request}
    # Exercise the supported indirection rather than only a literal load map.
    cfg.load = request.load
    cfg.runner.load = "${load}"
    return {}


@pytest.mark.parametrize("entry", ["request", "configured"])
def test_admitted_profile_restores_once(source, monkeypatch, entry):
    cfg, request, checkpoint, _ = source
    kwargs = _entry(cfg, request, entry)
    profiles = importlib.import_module("tpen.checkpoint.profiles")
    calls, reports, cursors = [], [], []
    original_validate = profiles.validate_native_recovery_profile
    original_fit = VMCTrainer.fit

    def validate(config):
        calls.append(config)
        return original_validate(config)

    def fit(self, **state):
        cursors.append((self.next_iteration, self.completed_updates))
        _equal(state["model"].state_dict(), torch.load(checkpoint / "model.pt", weights_only=False))
        _equal(state["optimizer"].state_dict(), torch.load(checkpoint / "optimizer.pt", weights_only=False))
        _equal(state["sampler"].mcmc_state_dict(), torch.load(checkpoint / "sampler.pt", weights_only=False))
        return original_fit(self, **state)

    class Readback:
        def handle_occurrence(self, occurrence, context):
            if isinstance(occurrence.event, CheckpointRestored):
                reports.append(occurrence.event.report)

    monkeypatch.setattr(profiles, "validate_native_recovery_profile", validate)
    monkeypatch.setattr(VMCTrainer, "fit", fit)
    assert run_module.run_from_config(
        cfg, **kwargs, runtime_callbacks=(Readback(),), raise_exceptions=True,
    ) == 0
    assert len(calls) == 1
    assert cursors == [(1, 1)]
    assert len(reports) == 1
    assert all(getattr(reports[0], "loaded_" + owner)
               for owner in ("model", "optimizer", "trainer", "sampler", "rng"))


@pytest.mark.parametrize("entry", ["request", "configured"])
@pytest.mark.parametrize("refusal", ["unsupported-callback", "unsupported-callback-cadence"])
@pytest.mark.parametrize("raise_exceptions", [True, False])
def test_refusal_precedes_all_mutation(source, tmp_path, monkeypatch, entry, refusal, raise_exceptions):
    cfg, request, _, live = source
    kwargs = _entry(cfg, request, entry)
    if refusal == "unsupported-callback":
        cfg.callbacks.append({"_target_": "tpen.callback.FailureLog"})
    else:
        cfg.callbacks.append({"_target_": "tpen.callback.DataIntegrity", "max_calls": 1})

    def snapshot():
        return deepcopy({
            "model": live["model"].state_dict(),
            "optimizer": live["optimizer"].state_dict(),
            "sampler": live["sampler"].mcmc_state_dict(),
            "trainer": live["trainer"].state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        })

    before, files, config = snapshot(), _files(tmp_path), OmegaConf.to_yaml(cfg)
    entered = []
    original_prepare = run_module.prepare_run_context

    def prepare(*args, **kwargs):
        entered.append(True)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(run_module, "prepare_run_context", prepare)
    if raise_exceptions:
        with pytest.raises(UnsupportedMutableStateError) as error:
            run_module.run_from_config(cfg, **kwargs, raise_exceptions=True)
        assert error.value.code == refusal
    else:
        assert run_module.run_from_config(cfg, **kwargs) == 1
    assert entered == [], "profile refusal must precede context/callback/model construction"
    _equal(before, snapshot())
    assert _files(tmp_path) == files  # Includes checkpoint/catalog bytes and mtimes.
    assert not (tmp_path / "successor").exists()
    assert OmegaConf.to_yaml(cfg) == config


@pytest.mark.parametrize("mode", ["none", "model_only"])
@pytest.mark.parametrize("unsupported", ["callback", "cadence"])
def test_unsupported_profile_does_not_gate_non_resume(source, tmp_path, monkeypatch, mode, unsupported):
    cfg, _, checkpoint, _ = source
    callback = ({"_target_": "tpen.callback.FailureLog"} if unsupported == "callback"
                else {"_target_": "tpen.callback.DataIntegrity", "max_calls": 1})
    cfg.callbacks.append(callback)
    cfg.runner.load = {"mode": mode, "path": str(checkpoint)}
    profiles = importlib.import_module("tpen.checkpoint.profiles")

    def unexpected(config):
        pytest.fail("non-resume path must not invoke recovery admission")

    monkeypatch.setattr(profiles, "validate_native_recovery_profile", unexpected)
    reports = []

    class Readback:
        def handle_occurrence(self, occurrence, context):
            if isinstance(occurrence.event, CheckpointRestored):
                reports.append(occurrence.event.report)

    if mode == "model_only":
        cfg.runner = {
            "_target_": "tpen.runner.Evaluate", "model": "${model}",
            "load": {"mode": "model_only", "path": str(checkpoint)},
            "evaluator": {
                "_target_": "tpen.evaluation.Evaluator", "namespace": "eval",
                "tasks": [{
                    "name": "null_task", "namespace": "eval/null_task",
                    # Keep this fixture independent of run-populated science identity.
                    "output_dir": str(tmp_path / "null_task"),
                    "generator": {"_target_": "tests.helpers.evaluation_components.NullGenerator"},
                    "calculators": [{"_target_": "tests.helpers.evaluation_components.IdentityCalculator"}],
                    "summaries": [{"_target_": "tests.helpers.evaluation_components.MetricSummary"}],
                }],
            },
        }
        cfg.callbacks = [callback]
    assert run_module.run_from_config(
        cfg, runtime_callbacks=(Readback(),), raise_exceptions=True,
    ) == 0
    assert len(reports) == (1 if mode == "model_only" else 0)
    if reports:
        assert reports[0].mode == "model_only" and reports[0].loaded_model



@pytest.mark.parametrize("system", ["he", "h2"])
@pytest.mark.parametrize("entry", ["request", "configured"])
def test_stage1_diagnostics_composition_passes_restore_admission(source, monkeypatch, system, entry):
    _, request, _, _ = source
    base = Path("experiments/atomistic/stage1/configs")
    cfg = OmegaConf.load(base / system / "train.yaml")
    overlay = base / "overlays/diagnostics-v1.yaml"
    compose_train_overlay(cfg, overlay=overlay.resolve(), expected_sha256=overlay_sha256(overlay.resolve()))
    validate_native_recovery_profile(cfg)
    kwargs = _entry(cfg, request, entry)
    reached = []

    class Admitted(Exception):
        pass

    def after_guard(*args, **kwargs):
        reached.append(True)
        raise Admitted("stop before constructing the 50k-update science run")

    monkeypatch.setattr(run_module, "prepare_run_context", after_guard)
    with pytest.raises(Admitted):
        run_module.run_from_config(cfg, **kwargs, raise_exceptions=True)
    assert reached == [True]


def _positional_load(cfg):
    cfg.runner = {
        "_target_": "tpen.runner.Train",
        "_args_": ["${model}", "${sampler}", "${hamiltonian_terms}",
                    "${optimizer}", "${trainer}", "${load}"],
    }


def _rng_snapshot():
    return deepcopy((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def _assert_early_refusal(cfg, kwargs, code, raise_exceptions, tmp_path, monkeypatch):
    entered = []

    def forbidden(*args, **kwargs):
        entered.append(True)
        raise AssertionError("refusal crossed the preconstruction boundary")

    train = importlib.import_module("tpen.runner.train")
    monkeypatch.setattr(run_module, "prepare_run_context", forbidden)
    monkeypatch.setattr(run_module, "_instantiate_runner", forbidden)
    monkeypatch.setattr(train, "restore_checkpoint_with_events", forbidden)
    before, files = _rng_snapshot(), _files(tmp_path)
    if raise_exceptions:
        with pytest.raises(UnsupportedMutableStateError) as error:
            run_module.run_from_config(cfg, **kwargs, raise_exceptions=True)
        assert error.value.code == code
    else:
        assert run_module.run_from_config(cfg, **kwargs) == 1
    assert entered == []
    _equal(before, _rng_snapshot())
    assert _files(tmp_path) == files
    assert not (tmp_path / "successor").exists()


@pytest.mark.parametrize("raise_exceptions", [True, False])
def test_positional_resume_refuses_before_construction(tmp_path, monkeypatch, raise_exceptions):
    cfg = OmegaConf.load(FIXTURE)
    cfg.run.root = str(tmp_path / "successor")
    cfg.load = {"mode": "train_resume", "path": str(tmp_path / "unused")}
    _positional_load(cfg)
    cfg.callbacks.append({"_target_": "tpen.callback.FailureLog"})
    _assert_early_refusal(cfg, {}, "unsupported-callback", raise_exceptions, tmp_path, monkeypatch)


@pytest.mark.parametrize("positional", [False, True])
def test_plain_load_reference_is_admitted(tmp_path, monkeypatch, positional):
    cfg = OmegaConf.load(FIXTURE)
    cfg.load = {"mode": "${resume_mode}"}
    cfg.resume_mode = "train_resume"
    cfg.runner.load = "${load}"
    if positional:
        _positional_load(cfg)
    reached = []

    class Admitted(Exception):
        pass

    def after_guard(*args, **kwargs):
        reached.append(True)
        raise Admitted

    monkeypatch.setattr(run_module, "prepare_run_context", after_guard)
    with pytest.raises(Admitted):
        run_module.run_from_config(cfg, raise_exceptions=True)
    assert reached == [True]


@pytest.mark.parametrize("entry", ["request", "configured"])
@pytest.mark.parametrize("raise_exceptions", [True, False])
def test_native_refusal_precedes_hi_resolvers(source, tmp_path, monkeypatch, entry, raise_exceptions):
    cfg, request, _, _ = source
    kwargs = _entry(cfg, request, entry)
    cfg.schema = "tpen.hi.train.v1"
    cfg.callbacks.append({"_target_": "tpen.callback.FailureLog"})
    calls = []
    marker = tmp_path / "resolver-ran"

    def side_effect():
        calls.append(True)
        marker.write_text("resolver executed")
        random.random()
        return 1

    name = "cx09_r4_side_effect"
    OmegaConf.register_new_resolver(name, side_effect)
    try:
        cfg.probe = "${cx09_r4_side_effect:}"
        _assert_early_refusal(cfg, kwargs, "unsupported-callback", raise_exceptions, tmp_path, monkeypatch)
        assert calls == []
        assert not marker.exists()
    finally:
        OmegaConf.clear_resolver(name)


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("load", ["${cx09_unregistered:}", {"mode": "${cx09_unregistered:}"},
                                  "${missing}", "${load}"])
@pytest.mark.parametrize("raise_exceptions", [True, False])
def test_indeterminate_load_refuses_without_resolution(tmp_path, monkeypatch, positional, load, raise_exceptions):
    cfg = OmegaConf.load(FIXTURE)
    cfg.run.root = str(tmp_path / "successor")
    cfg.load = load
    cfg.runner.load = "${load}"
    if positional:
        _positional_load(cfg)
    _assert_early_refusal(cfg, {}, "malformed-config", raise_exceptions, tmp_path, monkeypatch)
