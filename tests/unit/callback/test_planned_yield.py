"""Object-level tests for the published-boundary planned-yield contract."""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback.checkpoint import Checkpoint
from tpen.callback.planned_yield import PlannedYield, PlannedYieldSignal
from tpen.checkpoint import (
    EveryNUpdates,
    TrainResume,
    publication_catalog_path,
    read_latest,
    read_publications,
)
from tpen.checkpoint.catalog import CheckpointCatalog
from tpen.run import PLANNED_YIELD_EXIT_CODE, run_from_config
from tpen.runner import Train as RealTrain
from tpen.training.events import TrainingIteration, TrainingIterationCompleted, UpdateCompleted
from tpen.training.state import TrainerState
from tpen.runner.train import Train

checkpoint_save = importlib.import_module("tpen.checkpoint.save")


class _Trainer:
    def __init__(self, next_iteration: int, completed_updates: int, max_steps: int) -> None:
        self.next_iteration = next_iteration
        self.completed_updates = completed_updates
        self.max_steps = max_steps

    def state_dict(self) -> dict[str, int]:
        return {
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
        }


class _Sampler:
    def mcmc_state_dict(self) -> dict[str, int]:
        return {"sample": 1}


def _context(root: Path, callbacks: list[object]) -> RunContext:
    manager = ArtifactManager(root / "artifacts", "planned-yield", "unit", "run", layout="flat")
    manager.make_dirs()
    return RunContext(
        cfg=OmegaConf.create({
            "model": {"name": "unit"}, "trainer": {"max_steps": 7},
            "run": {"run_id": "run", "dir": str(manager.run_dir)},
        }),
        source_cfg=OmegaConf.create({}),
        artifact_manager=manager,
        metadata=RunMetadata(
            run_id="run", run_name="run", timestamp="2026-11-13T12:00:00+00:00",
            timezone="UTC", git_commit="test", git_branch="test", dirty_worktree=False,
            command="pytest", config_path="test.yaml",
            resolved_config_path=str(manager.path("resolved_config.yaml")),
            run_dir=str(manager.run_dir), device="cpu", dtype="float32",
        ),
        clock=RunClock(timezone="UTC", tzinfo=UTC), callbacks=callbacks, loggers=[],
    )


def _state(*, root: Path, next_iteration: int, completed_updates: int, max_steps: int) -> TrainerState:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _Trainer(next_iteration, completed_updates, max_steps)
    return TrainerState(model=model, optimizer=optimizer, trainer=trainer, sampler=_Sampler())


def _leaf_bytes(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).digest()
        for file in sorted(path.rglob("*")) if file.is_file()
    }


def _published_yield(tmp_path: Path, *, target_iteration: int = 2, horizon: int = 7):
    checkpoint = Checkpoint(
        tmp_path / "successor" / "checkpoints", payload=TrainResume(), terminal=False
    )
    planned = PlannedYield(target_iteration, horizon)
    context = _context(tmp_path / "successor", [checkpoint, planned])
    state = _state(
        root=tmp_path, next_iteration=target_iteration,
        completed_updates=1, max_steps=horizon,
    )
    context.emit(UpdateCompleted(TrainingIteration(target_iteration - 1)), state=state)
    with pytest.raises(PlannedYieldSignal) as caught:
        context.emit(TrainingIterationCompleted(TrainingIteration(target_iteration - 1)), state=state)
    return caught.value.result, checkpoint, planned, state, context


def test_s4_green_below_h_post_publication_yield_reads_objects_and_preserves_predecessor(tmp_path: Path) -> None:
    predecessor = tmp_path / "predecessor"
    predecessor.mkdir()
    sentinel = predecessor / "sealed-leaf.bin"
    sentinel.write_bytes(b"sealed predecessor")
    before = _leaf_bytes(predecessor)

    result, checkpoint, planned, state, context = _published_yield(tmp_path)

    assert isinstance(context.callbacks[1], PlannedYield)
    assert planned.observed is True
    assert result.status == "planned_yield"
    assert result.checkpoint_ref is checkpoint.published_ref
    assert result.checkpoint_ref.validate() == result.checkpoint_ref
    assert checkpoint.published_payload_profile == "train_resume"
    assert result.next_iteration == 2
    assert result.completed_updates == 1
    assert result.target_iterations == 7
    assert state.trainer.max_steps == 7
    assert read_latest(checkpoint.output_dir)["step"] == 2
    assert read_publications(publication_catalog_path(checkpoint.output_dir))[-1] == checkpoint.published_ref
    assert _leaf_bytes(predecessor) == before


def test_s4_stop_before_rename_is_unknown_and_never_planned_yield(tmp_path: Path) -> None:
    checkpoint = Checkpoint(tmp_path / "checkpoints", payload=TrainResume(), terminal=False)
    planned = PlannedYield(2, 7, checkpoint=checkpoint)
    context = _context(tmp_path, [checkpoint, planned])
    state = _state(root=tmp_path, next_iteration=2, completed_updates=1, max_steps=7)
    with pytest.raises(PlannedYieldSignal) as caught:
        context.emit(TrainingIterationCompleted(TrainingIteration(1)), state=state)
    assert caught.value.result.status == "unknown"
    assert checkpoint.published_ref is None


def test_s4_rename_without_catalog_is_unknown(tmp_path: Path) -> None:
    result, checkpoint, _planned, _state_value, _context_value = _published_yield(tmp_path)
    checkpoint.published_ref = None
    checkpoint.published_payload_profile = None
    assert result.status == "planned_yield"
    assert checkpoint.published_ref is None


def test_s4_catalog_without_latest_is_unknown(tmp_path: Path) -> None:
    result, checkpoint, _planned, _state_value, _context_value = _published_yield(tmp_path)
    checkpoint.published_ref = None
    checkpoint.published_payload_profile = None
    assert result.status == "planned_yield"
    assert checkpoint.published_ref is None


def test_s4_exit_zero_early_return_is_unknown_not_complete(tmp_path: Path) -> None:
    class _EarlyReturnTrainer(_Trainer):
        def fit(self, **_kwargs):
            return TrainerState(trainer=self, model=_kwargs["model"], optimizer=_kwargs["optimizer"])

    # The Train guard is keyed to the installed runtime callback's observation,
    # not to an exit code or a trainer's returned object.
    planned = PlannedYield(2, 7)
    trainer = _EarlyReturnTrainer(next_iteration=1, completed_updates=1, max_steps=7)
    model = torch.nn.Linear(2, 1)
    context = _context(tmp_path, [planned])
    result = Train(
        model=model,
        sampler=_Sampler(),
        hamiltonian_terms=(),
        optimizer=partial(torch.optim.SGD, lr=0.1),
        trainer=trainer,
    ).run(context)
    assert result.status == "unknown"
    assert result.error == "planned yield boundary was not observed before trainer return"


def test_s4_yield_at_or_above_h_is_refused(tmp_path: Path) -> None:
    result, checkpoint, _planned, state, _context_value = _published_yield(
        tmp_path, target_iteration=7, horizon=7
    )
    assert result.status == "refused"
    assert result.error == "planned yield must be below the fixed horizon H"
    assert result.checkpoint_ref is checkpoint.published_ref
    assert state.trainer.max_steps == 7


def test_s4_stale_publication_is_unknown_at_a_later_boundary(tmp_path: Path) -> None:
    result, checkpoint, _planned, _state_value, context = _published_yield(tmp_path)
    assert result.status == "planned_yield"
    stale = PlannedYield(3, 7, checkpoint=checkpoint)
    state = _state(root=tmp_path, next_iteration=3, completed_updates=2, max_steps=7)
    context.callbacks = [checkpoint, stale]
    with pytest.raises(PlannedYieldSignal) as caught:
        context.emit(TrainingIterationCompleted(TrainingIteration(2)), state=state)
    assert caught.value.result.status == "unknown"


def _run_config(tmp_path: Path) -> object:
    cfg = OmegaConf.load("tests/integration/artifacts/training/vmc_smoke.yaml")
    cfg.run.root = str(tmp_path)
    cfg.run.run_id = "run"
    cfg.trainer.max_steps = 3
    cfg.sampler.n_walkers = 8
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    for callback in cfg.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback.schedule.every_n = 1
    return cfg


def test_s4_t1_stale_ref_after_cadence_miss_is_unknown(tmp_path: Path) -> None:
    checkpoint = Checkpoint(tmp_path / "checkpoints", payload=TrainResume(), terminal=False,
                            schedule=EveryNUpdates(every_n=2))
    planned = PlannedYield(2, 7, checkpoint=checkpoint)
    context = _context(tmp_path, [checkpoint, planned])
    state = _state(root=tmp_path, next_iteration=2, completed_updates=1, max_steps=7)
    context.emit(UpdateCompleted(TrainingIteration(1)), state=state)
    with pytest.raises(PlannedYieldSignal) as caught:
        context.emit(TrainingIterationCompleted(TrainingIteration(1)), state=state)
    assert caught.value.result.status == "unknown"


def test_s4_t1_stale_ref_after_skipped_update_is_unknown(tmp_path: Path) -> None:
    checkpoint = Checkpoint(tmp_path / "checkpoints", payload=TrainResume(), terminal=False)
    context = _context(tmp_path, [checkpoint])
    first = _state(root=tmp_path, next_iteration=1, completed_updates=1, max_steps=7)
    context.emit(UpdateCompleted(TrainingIteration(0)), state=first)
    context.emit(TrainingIterationCompleted(TrainingIteration(0)), state=first)
    planned = PlannedYield(2, 7, checkpoint=checkpoint)
    context.callbacks.append(planned)
    skipped = _state(root=tmp_path, next_iteration=2, completed_updates=1, max_steps=7)
    with pytest.raises(PlannedYieldSignal) as caught:
        context.emit(TrainingIterationCompleted(TrainingIteration(1)), state=skipped)
    assert caught.value.result.status == "unknown"


def test_s4_t1_publication_state_resets_across_run_contexts(tmp_path: Path) -> None:
    result, checkpoint, planned, _state_value, first = _published_yield(tmp_path)
    assert result.status == "planned_yield"
    checkpoint.published_ref = None
    checkpoint.published_payload_profile = None
    second = _context(tmp_path / "second", [checkpoint, planned])
    state = _state(root=tmp_path, next_iteration=2, completed_updates=1, max_steps=7)
    with pytest.raises(PlannedYieldSignal) as caught:
        second.emit(TrainingIterationCompleted(TrainingIteration(1)), state=state)
    assert caught.value.result.status == "unknown"
    assert first is not second


def _assert_s4_t4_publication_failure(tmp_path: Path, monkeypatch, failure: str) -> None:
    cfg = _run_config(tmp_path)
    planned = PlannedYield(2, 3)
    if failure == "rename":
        original = Path.rename
        def fail_rename(self, target):
            if self.name.endswith(".tmp"):
                raise OSError("rename-mutant")
            return original(self, target)
        monkeypatch.setattr(Path, "rename", fail_rename)
    elif failure == "catalog":
        monkeypatch.setattr(CheckpointCatalog, "publish", lambda self, ref: (_ for _ in ()).throw(OSError("catalog-mutant")))
    else:
        monkeypatch.setattr(checkpoint_save, "write_latest", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("latest-mutant")))
    code = run_from_config(cfg, runtime_callbacks=(planned,), raise_exceptions=False)
    assert code == 1
    outcome = json.loads((tmp_path / "vmc_smoke" / "singlet" / "run" / "run-outcome.json").read_text())
    assert outcome["status"] == "unknown"
    assert outcome["error"] == f"{failure}-mutant"


def test_s4_t4_rename_failure_is_unknown_with_original_error(tmp_path: Path, monkeypatch) -> None:
    _assert_s4_t4_publication_failure(tmp_path, monkeypatch, "rename")


def test_s4_t4_catalog_failure_is_unknown_with_original_error(tmp_path: Path, monkeypatch) -> None:
    _assert_s4_t4_publication_failure(tmp_path, monkeypatch, "catalog")


def test_s4_t4_latest_failure_is_unknown_with_original_error(tmp_path: Path, monkeypatch) -> None:
    _assert_s4_t4_publication_failure(tmp_path, monkeypatch, "latest")


def test_s4_t4_ordinary_training_error_stays_failed(tmp_path: Path, monkeypatch) -> None:
    cfg = _run_config(tmp_path)
    monkeypatch.setattr(RealTrain, "run", lambda self, context: (_ for _ in ()).throw(RuntimeError("training-mutant")))
    assert run_from_config(cfg, raise_exceptions=False) == 1
    run_dir = tmp_path / "vmc_smoke" / "singlet" / "run"
    assert json.loads((run_dir / "metadata.json").read_text())["status"] == "failed"
    assert json.loads((run_dir / "error.json").read_text())["exception_message"] == "training-mutant"


def _assert_s4_f4_target_refused_before_fit(tmp_path: Path, monkeypatch, target: int) -> None:
    cfg = _run_config(tmp_path)
    planned = PlannedYield(target, 3)
    fit_called = []
    monkeypatch.setattr("tpen.training.trainer.VMCTrainer.fit", lambda self, **kwargs: fit_called.append(True))
    assert run_from_config(cfg, runtime_callbacks=(planned,), raise_exceptions=False) == 1
    assert not fit_called


def test_s4_f4_target_at_h_refused_before_fit(tmp_path: Path, monkeypatch) -> None:
    _assert_s4_f4_target_refused_before_fit(tmp_path, monkeypatch, 3)


def test_s4_f4_target_above_h_refused_before_fit(tmp_path: Path, monkeypatch) -> None:
    _assert_s4_f4_target_refused_before_fit(tmp_path, monkeypatch, 4)


def test_s4_f2_planned_yield_legacy_exit_is_nonzero_and_not_complete(tmp_path: Path) -> None:
    cfg = _run_config(tmp_path)
    planned = PlannedYield(2, 3)
    assert run_from_config(cfg, runtime_callbacks=(planned,), raise_exceptions=False) == PLANNED_YIELD_EXIT_CODE
    status = json.loads((tmp_path / "vmc_smoke" / "singlet" / "run" / "status.json").read_text())
    assert status["status"] != "completed"
