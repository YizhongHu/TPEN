"""Three-process CPU qualification of the HI continuation state boundary.

The test is deliberately an executable receipt: one child runs the complete
H-step arm, one publishes a checkpoint at the fixed prefix, and a third child
restores that checkpoint.  No production module is imported from ``tests``.
"""

from __future__ import annotations

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


def _semantic_digest(value: object) -> str:
    """Digest state content while ignoring pickle/container metadata."""
    import torch

    def canonical(item: object) -> object:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return ("torch", str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes())
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return ("numpy", str(array.dtype), tuple(array.shape), array.tobytes())
        if isinstance(item, dict):
            return tuple(sorted((str(key), canonical(val)) for key, val in item.items()))
        if isinstance(item, (list, tuple)):
            return tuple(canonical(val) for val in item)
        return item

    return _digest(canonical(value))


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
        self._checkpoint_output_dir: Path | None = None

    def handle_occurrence(self, occurrence, context, state=None) -> None:
        event = occurrence.event
        if isinstance(event, CheckpointRestored):
            self._last_state = self._snapshot(state, event.report, context)
            self.records["restored"] = self._last_state
        elif isinstance(event, TrainingStarted):
            # Train emits TrainingStarted without TrainerState.  For a restore
            # this is adjacent to CheckpointRestored; the disabled control has
            # no restore event, so take a genuine fresh-start snapshot here.
            self.records["training_started"] = (
                dict(self._last_state)
                if self._last_state is not None
                else self._snapshot(None, None, context)
            )
        elif isinstance(event, TrainingIterationCompleted) and state is not None:
            iteration = int(state.trainer.next_iteration)
            samples = state.samples
            self.records.setdefault("iterations", {})[str(iteration)] = {
                "samples": _tensor_digest(samples.positions),
                "local_energy": state.local_energy.detach().cpu().tolist(),
                "next_iteration": int(state.trainer.next_iteration),
                "completed_updates": int(state.trainer.completed_updates),
            }
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
        walkers = sampler_state["walkers"]
        sampler_cursor = None if walkers is None else (
            _tensor_digest(walkers.positions)
            + ":"
            + _tensor_digest(sampler_state["generator_state"])
        )
        sampler_semantic = {
            "cursor": sampler_cursor,
            "has_burned_in": bool(sampler_state["has_burned_in"]),
            "generator_state": _tensor_digest(sampler_state["generator_state"]),
        }
        if walkers is not None:
            sampler_semantic["walkers"] = _tensor_digest(walkers.positions)
        checkpoint_callbacks = [
            callback
            for callback in getattr(context, "callbacks", ())
            if type(callback).__qualname__ == "Checkpoint"
        ]
        assert len(checkpoint_callbacks) == 1, "expected exactly one live Checkpoint callback"
        self._checkpoint_output_dir = Path(checkpoint_callbacks[0].output_dir)
        schedule = getattr(checkpoint_callbacks[0], "schedule", None)
        every_n = getattr(schedule, "every_n", None)
        assert isinstance(every_n, int) and not isinstance(every_n, bool) and every_n > 0, (
            "live Checkpoint schedule must expose a positive integer every_n"
        )
        cadence = int(every_n)
        return {
            "model": _semantic_digest(model.state_dict()),
            "observable": float(next(model.parameters()).detach().reshape(-1)[0]),
            "optimizer": _semantic_digest(optimizer.state_dict()),
            "trainer": {"next_iteration": int(trainer.next_iteration), "completed_updates": int(trainer.completed_updates), "max_steps": int(trainer.max_steps)},
            "sampler": _digest(sampler_state),
            "sampler_cursor": sampler_cursor,
            "sampler_semantic": sampler_semantic,
            "process_rng": _semantic_digest(process_rng),
            "callback_dispositions": [
                type(callback).__qualname__
                for callback in getattr(context, "callbacks", ())
                if type(callback).__qualname__ != "PlannedYield"
            ],
            "report": None if report is None else repr(report),
            "horizon": int(trainer.max_steps),
            "checkpoint_cadence": cadence,
        }

    def finish(self) -> None:
        if self._checkpoint_output_dir is not None:
            self.records["checkpoint_steps"] = sorted(
                int(path.name.removeprefix("step_"))
                for path in self._checkpoint_output_dir.glob("step_*")
                if path.is_dir()
            )
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
        if action == "successor":
            # Perturb process RNG after construction/config hashing but before
            # the restore boundary. A real restore replaces this state; a
            # skipped apply_rng_state leaves the perturbation observable.
            random.seed(1)
            np.random.seed(1)
            import torch
            torch.manual_seed(1)
        return observer._optimizer

    def capture_runner(runner, context):
        observer._runner = runner
        if action == "disabled":
            # Preserve the effective H/cadence while stopping exactly at the
            # fresh TrainingStarted boundary, before any update or sampling.
            def fresh_start(**_kwargs):
                return TrainerState(
                    model=runner.model,
                    optimizer=observer._optimizer,
                    trainer=runner.trainer,
                    sampler=runner.sampler,
                )

            runner.trainer.fit = fresh_start
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
    assert expected["horizon"] == observed["horizon"] == H
    assert expected["checkpoint_cadence"] == observed["checkpoint_cadence"] == CHECKPOINT_EVERY
    np.testing.assert_allclose(expected["observable"], observed["observable"], rtol=RTOL, atol=ATOL)
    for key in ("model", "optimizer", "sampler", "sampler_cursor", "sampler_semantic", "process_rng", "callback_dispositions"):
        assert key in expected and key in observed
    # These semantic digests are the mutable state at the restore boundary;
    # raw pickle digests include process-local/container metadata. Presence
    # alone would let a restore-disabled mutant pass undetected.
    for key in ("model", "optimizer", "sampler_semantic", "process_rng", "callback_dispositions"):
        assert expected[key] == observed[key], key


def test_hi_s5b_b2_three_process_stateful_continuation(tmp_path: Path) -> None:
    """Qualify exact state, sampler cursor, H, cadence, and non-mutating restore."""
    full_cfg, prefix_cfg, successor_cfg, disabled_cfg, refused_cfg = [tmp_path / f"{name}.yaml" for name in ("full", "prefix", "successor", "disabled", "refused")]
    roots = {name: tmp_path / name for name in ("full", "prefix", "successor", "disabled", "refused")}
    for path, name in ((full_cfg, "full"), (prefix_cfg, "prefix"), (successor_cfg, "successor"), (disabled_cfg, "disabled"), (refused_cfg, "refused")):
        _config(path, roots[name], name)
    full_record, prefix_record, successor_record, disabled_record, refused_record = [tmp_path / f"{name}.json" for name in ("full", "prefix", "successor", "disabled", "refused")]
    _spawn("full", full_cfg, full_record)
    _spawn("prefix", prefix_cfg, prefix_record)
    run_dir = roots["prefix"] / "tpen_he_importance" / "atomistic" / "prefix"
    checkpoint = run_dir / "checkpoints" / "step_000001"
    assert checkpoint.is_dir()
    before = _tree_digest(run_dir)
    _spawn("successor", successor_cfg, successor_record, checkpoint)
    assert _tree_digest(run_dir) == before
    before_refused = _tree_digest(run_dir)
    _spawn("refused", refused_cfg, refused_record, checkpoint)
    assert _tree_digest(run_dir) == before_refused
    _spawn("disabled", disabled_cfg, disabled_record)
    full = json.loads(full_record.read_text())
    prefix = json.loads(prefix_record.read_text())
    restored = json.loads(successor_record.read_text())
    disabled = json.loads(disabled_record.read_text())
    refused = json.loads(refused_record.read_text())
    # The raw sampler pickle includes non-semantic serialization metadata, so
    # the exact next-draw witness is the semantic cursor: walker positions and
    # the sampler-local generator state.  Check both the checkpoint producer
    # and the uninterrupted arm against the restored successor; comparing only
    # with the prefix would miss a pre-boundary divergence in the prefix arm.
    _assert_equivalent(prefix["boundary"], restored["training_started"])
    _assert_equivalent(full["boundary"], restored["training_started"])
    full_iteration = full["iterations"][str(PREFIX + 1)]
    restored_iteration = restored["iterations"][str(PREFIX + 1)]
    assert len(restored["iterations"]) == H - PREFIX
    assert restored_iteration["completed_updates"] == PREFIX + 1
    assert restored_iteration["next_iteration"] == PREFIX + 1
    final_restored_iteration = restored["iterations"][str(H)]
    assert final_restored_iteration["completed_updates"] == H
    assert final_restored_iteration["next_iteration"] == H
    full_post_prefix_checkpoints = [step for step in full["checkpoint_steps"] if step > PREFIX]
    restored_checkpoints = [step for step in restored["checkpoint_steps"] if step > PREFIX]
    assert restored_checkpoints == full_post_prefix_checkpoints
    assert full_iteration["samples"] == restored_iteration["samples"]
    np.testing.assert_allclose(full_iteration["local_energy"], restored_iteration["local_energy"], rtol=RTOL, atol=ATOL)
    assert restored["restored"]["trainer"]["max_steps"] == H
    assert disabled["training_started"]["model"] != restored["training_started"]["model"]
    assert disabled["training_started"]["trainer"] == {"next_iteration": 0, "completed_updates": 0, "max_steps": H}
    assert disabled["training_started"]["optimizer"] != restored["training_started"]["optimizer"]
    assert disabled["training_started"]["checkpoint_cadence"] == CHECKPOINT_EVERY
    assert "restored" not in disabled and "boundary" not in disabled
    assert refused == {}
    refused_error = roots["refused"] / "tpen_he_importance" / "atomistic" / "refused" / "error.json"
    error = json.loads(refused_error.read_text(encoding="utf-8"))
    assert "target_iterations=3" in error["exception_message"]
    assert "max_steps=4" in error["exception_message"]

