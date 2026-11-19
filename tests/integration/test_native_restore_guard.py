"""Preconstruction native-profile admission through both public restore entries."""

from copy import deepcopy
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


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _equal(first, second)
    else:
        assert left == right


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
        cfg.callbacks.append({"_target_": "tpen.callback.ResourceUsage"})
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
def test_unsupported_profile_does_not_gate_non_resume(source, monkeypatch, mode, unsupported):
    cfg, _, checkpoint, _ = source
    callback = ({"_target_": "tpen.callback.ResourceUsage"} if unsupported == "callback"
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
                    "output_dir": "${run.dir}/null_task",
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
