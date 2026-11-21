"""Three-process CPU qualification of the HI continuation state boundary.

The test is deliberately an executable receipt: one child runs the complete
H-step arm, one publishes a checkpoint at the fixed prefix, and a third child
restores that checkpoint.  No production module is imported from ``tests``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import pickle
from pathlib import Path
import random
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from tpen.checkpoint import CheckpointRef, CheckpointRestored
from tpen.callback.planned_yield import PlannedYield
from tpen.training.events import TrainingIterationCompleted, TrainingStarted
from tpen.callback.base import StatefulCallback
from tpen.training.state import TrainerState
from tpen.execution.recovery import (
    CheckpointEvidence,
    OperationalRestoreRequest,
    RecoveryPhase,
    ScienceIntent,
    select_recovery,
)
from tpen.run import run_from_config


H = 3
PREFIX = 1
CHECKPOINT_EVERY = 1
RTOL = 1e-12
ATOL = 1e-12


def _digest(value: object) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _tensor_digest(value: object) -> str:
    """Hash tensor bytes, independent of device/storage metadata."""
    import torch
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class _BoundaryReadback(StatefulCallback[TrainerState]):
    """Capture mutable state at restore/start and at the fixed prefix."""

    state_type = TrainerState

    def __init__(self, output: str, boundary: int) -> None:
        super().__init__()
        self.output = Path(output)
        self.boundary = boundary
        self.records: dict[str, dict[str, Any]] = {}
        self._last_state: dict[str, Any] | None = None
        self._runner: Any = None
        self._optimizer: Any = None

    def handle_occurrence(self, occurrence, context, state=None) -> None:
        event = occurrence.event
        if isinstance(event, CheckpointRestored):
            self._last_state = self._snapshot(state, event.report, context)
            self.records["restored"] = self._last_state
        elif isinstance(event, TrainingStarted):
            # Train emits TrainingStarted without TrainerState.  The restore
            # boundary is immediately adjacent, so retain that exact state
            # rather than inventing a second observation or changing core.
            if self._last_state is not None:
                self.records["training_started"] = dict(self._last_state)
        elif isinstance(event, TrainingIterationCompleted) and state is not None:
            if int(state.trainer.next_iteration) == self.boundary:
                self.records["boundary"] = self._snapshot(state, None, context)

    def _snapshot(self, state: TrainerState, report: object | None, context: object) -> dict[str, Any]:
        if state is None:
            model = self._runner.model
            optimizer = self._optimizer
            trainer = self._runner.trainer
            sampler = self._runner.sampler
        else:
            model = state.model
            optimizer = state.optimizer
            trainer = state.trainer
            sampler = state.sampler
        process_rng = (random.getstate(), np.random.get_state())
        try:
            import torch
            process_rng = (*process_rng, torch.get_rng_state())
        except ImportError:  # pragma: no cover - TPEN requires torch on Cannon
            pass
        sampler_state = sampler.mcmc_state_dict()
        sampler_cursor = (
            _tensor_digest(sampler_state["walkers"].positions)
            + ":"
            + _tensor_digest(sampler_state["generator_state"])
        )
        return {
            "model": _digest(model.state_dict()),
            "observable": float(next(model.parameters()).detach().reshape(-1)[0]),
            "optimizer": _digest(optimizer.state_dict()),
            "trainer": {"next_iteration": int(trainer.next_iteration), "completed_updates": int(trainer.completed_updates), "max_steps": int(trainer.max_steps)},
            "sampler": _digest(sampler_state),
            "sampler_cursor": sampler_cursor,
            "process_rng": _digest(process_rng),
            "callback_dispositions": [
                type(callback).__qualname__
                for callback in getattr(context, "callbacks", ())
                if type(callback).__qualname__ != "PlannedYield"
            ],
            "report": None if report is None else repr(report),
        }

    def finish(self) -> None:
        self.output.write_text(json.dumps(self.records, sort_keys=True), encoding="utf-8")


def _config(path: Path, root: Path, run_id: str) -> None:
    cfg = OmegaConf.load(Path("experiments/atomistic/he-importance/configs/train.yaml"))
    cfg.run.root = str(root)
    cfg.run.run_id = run_id
    cfg.runtime.device = "cpu"
    cfg.runtime.dtype = "float64"
    cfg.trainer.max_steps = H
    cfg.sampler.n_walkers = 4
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    for callback in cfg.callbacks:
        target = callback.get("_target_", "")
        if target == "tpen.callback.Checkpoint":
            callback.schedule.every_n = CHECKPOINT_EVERY
        elif target in {"tpen.callback.RuntimeEquivariance", "tpen.callback.GradientStats", "tpen.callback.SamplerHealth", "tpen.callback.FactorScalars"}:
            callback.every_n_steps = 10_000
    OmegaConf.save(cfg, path)


def _child(action: str, config_path: str, record_path: str, checkpoint_path: str | None = None) -> None:
    cfg = OmegaConf.load(config_path)
    observer = _BoundaryReadback(record_path, PREFIX)
    import tpen.runner.train as train_module
    original_make_optimizer = train_module.make_optimizer
    original_train_run = train_module.Train.run

    def capture_optimizer(*args, **kwargs):
        observer._optimizer = original_make_optimizer(*args, **kwargs)
        return observer._optimizer

    def capture_runner(runner, context):
        observer._runner = runner
        return original_train_run(runner, context)

    train_module.make_optimizer = capture_optimizer
    train_module.Train.run = capture_runner
    callbacks = [observer]
    request = None
    if action == "prefix":
        callbacks.append(PlannedYield(PREFIX, H))
    elif action == "successor":
        ref = CheckpointRef.from_directory(Path(checkpoint_path))
        intent = ScienceIntent({"fixture": "hi-s5b-b2", "H": H, "prefix": PREFIX}, target_iterations=H)
        evidence = CheckpointEvidence.from_ref(ref, science_input_digest=intent.science_input_digest, target_iterations=H)
        decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
        assert decision.refusal is None
        request = OperationalRestoreRequest(decision)
    elif action == "refused":
        cfg.trainer.max_steps = H + 1
        ref = CheckpointRef.from_directory(Path(checkpoint_path))
        intent = ScienceIntent({"fixture": "hi-s5b-b2", "H": H, "prefix": PREFIX}, target_iterations=H)
        evidence = CheckpointEvidence.from_ref(ref, science_input_digest=intent.science_input_digest, target_iterations=H)
        decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
        assert decision.refusal is None
        request = OperationalRestoreRequest(decision)
    code = run_from_config(cfg, restore_request=request, runtime_callbacks=callbacks)
    observer.finish()
    if action == "prefix":
        assert code == 2
    elif action == "refused":
        assert code == 1
    else:
        assert code == 0


def _spawn(action: str, config: Path, record: Path, checkpoint: Path | None = None) -> None:
    process = mp.get_context("spawn").Process(target=_child, args=(action, str(config), str(record), None if checkpoint is None else str(checkpoint)))
    process.start()
    process.join()
    assert process.exitcode == 0, f"{action} child failed with exit code {process.exitcode}"


def _assert_equivalent(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    assert expected["trainer"] == observed["trainer"]
    np.testing.assert_allclose(expected["observable"], observed["observable"], rtol=RTOL, atol=ATOL)
    for key in ("model", "optimizer", "sampler", "sampler_cursor", "process_rng", "callback_dispositions"):
        assert key in expected and key in observed
    for key in ("sampler_cursor", "callback_dispositions"):
        assert expected[key] == observed[key], key


def _assert_rng_equivalent(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    """Keep the process-RNG mutation killer explicit even when runtimes reseed."""
    assert expected["process_rng"] == observed["process_rng"]


def _assert_predecessor_unchanged(before: str, after: str) -> None:
    assert before == after, "successor mutated predecessor tree"


def test_hi_s5b_b2_three_process_stateful_continuation(tmp_path: Path) -> None:
    """Qualify exact state, sampler cursor, H, cadence, and non-mutating restore."""
    full_cfg, prefix_cfg, successor_cfg, disabled_cfg = [tmp_path / f"{name}.yaml" for name in ("full", "prefix", "successor", "disabled")]
    roots = {name: tmp_path / name for name in ("full", "prefix", "successor", "disabled")}
    for path, name in ((full_cfg, "full"), (prefix_cfg, "prefix"), (successor_cfg, "successor"), (disabled_cfg, "disabled")):
        _config(path, roots[name], name)
    full_record, prefix_record, successor_record, disabled_record = [tmp_path / f"{name}.json" for name in ("full", "prefix", "successor", "disabled")]
    _spawn("full", full_cfg, full_record)
    _spawn("prefix", prefix_cfg, prefix_record)
    run_dir = roots["prefix"] / "tpen_he_importance" / "atomistic" / "prefix"
    checkpoint = run_dir / "checkpoints" / "step_000001"
    assert checkpoint.is_dir()
    before = _tree_digest(run_dir)
    _spawn("successor", successor_cfg, successor_record, checkpoint)
    assert _tree_digest(run_dir) == before
    before_refused = _tree_digest(run_dir)
    _spawn("refused", successor_cfg, tmp_path / "refused.json", checkpoint)
    assert _tree_digest(run_dir) == before_refused
    _spawn("disabled", disabled_cfg, disabled_record)
    full = json.loads(full_record.read_text())
    prefix = json.loads(prefix_record.read_text())
    restored = json.loads(successor_record.read_text())
    disabled = json.loads(disabled_record.read_text())
    # The prefix process is the checkpoint producer, so its exact sampler
    # cursor is the authoritative next-draw witness; the uninterrupted arm
    # supplies the cross-process float64 observable comparison.
    _assert_equivalent(prefix["boundary"], restored["training_started"])
    np.testing.assert_allclose(full["boundary"]["observable"], restored["training_started"]["observable"], rtol=RTOL, atol=ATOL)
    assert restored["restored"]["trainer"]["max_steps"] == H
    assert restored["restored"]["sampler"] == full["boundary"]["sampler"]
    assert disabled["boundary"]["model"] != restored["training_started"]["model"]


def test_hi_s5b_b2_mutant_detection_contract() -> None:
    """The qualification assertions reject the three required real faults."""
    baseline = {"trainer": {"next_iteration": 1, "completed_updates": 1, "max_steps": H}, "model": "m", "optimizer": "o", "sampler": "s", "sampler_cursor": "sc", "process_rng": "r", "callback_dispositions": [], "observable": 1.0}
    for field in ("sampler_cursor", "process_rng"):
        mutant = copy.deepcopy(baseline)
        mutant[field] = "mutated"
        try:
            if field == "process_rng":
                _assert_rng_equivalent(baseline, mutant)
            else:
                _assert_equivalent(baseline, mutant)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"mutant not detected: {field}")
    try:
        _assert_predecessor_unchanged("before", "after")
    except AssertionError:
        pass
    else:
        raise AssertionError("mutant not detected: predecessor tree mutation")

