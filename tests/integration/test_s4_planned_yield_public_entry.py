"""S4 public-entry and routed planned-yield coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from tpen.artifacts import ArtifactDestination
from tpen.callback.planned_yield import PlannedYield
from tpen.checkpoint import CheckpointRef
from tpen.execution.recovery import (
    CheckpointEvidence,
    OperationalRestoreRequest,
    RecoveryPhase,
    ScienceIntent,
    select_recovery,
)
from tpen.run import PLANNED_YIELD_EXIT_CODE, run_from_config
from tpen.run_events import RunCompleted
from tpen.runner import Train


_FIXTURE = Path("tests/integration/artifacts/training/vmc_smoke.yaml")


def _cfg(root: Path, run_id: str, *, max_steps: int = 3):
    cfg = OmegaConf.load(_FIXTURE)
    cfg.run.root = str(root)
    cfg.run.run_id = run_id
    cfg.trainer.max_steps = max_steps
    cfg.sampler.n_walkers = 8
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    for callback in cfg.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback.schedule.every_n = 1
    return cfg


def _leaf_snapshot(root: Path) -> dict[str, tuple[str, int]]:
    return {
        str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }


class _RunCompletedReadback:
    """Real callback object used to observe the public typed boundary."""

    def __init__(self) -> None:
        self.events: list[RunCompleted] = []
        self.live_max_steps: int | None = None

    def handle_occurrence(self, occurrence, context) -> None:
        if isinstance(occurrence.event, RunCompleted):
            self.events.append(occurrence.event)


def test_s4_t3_routed_restore_yield_public_entry(tmp_path: Path, monkeypatch) -> None:
    import tpen.run as run_module

    qualified = f"{_RunCompletedReadback.__module__}.{_RunCompletedReadback.__qualname__}"
    monkeypatch.setattr(run_module, "_SINK_FREE_TYPES", run_module._SINK_FREE_TYPES | {qualified})

    source_destination = ArtifactDestination(tmp_path / "source-physical")
    source = _cfg(tmp_path / "logical", "source")
    assert run_from_config(source, artifact_destination=source_destination, raise_exceptions=True) == 0
    predecessor = source_destination.physical_root
    checkpoint = predecessor / "checkpoints" / "step_000001"
    ref = CheckpointRef.from_directory(checkpoint)
    intent = ScienceIntent({"fixture": "s4-t3"}, target_iterations=3)
    evidence = CheckpointEvidence.from_ref(
        ref, science_input_digest=intent.science_input_digest, target_iterations=3
    )
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    assert decision.refusal is None
    request = OperationalRestoreRequest(decision)
    before = _leaf_snapshot(predecessor)

    successor = _cfg(tmp_path / "logical", "successor")
    readback = _RunCompletedReadback()
    captured_h: list[int] = []
    original_run = Train.run

    def observe_run(self, context):
        captured_h.append(self.trainer.max_steps)
        return original_run(self, context)

    monkeypatch.setattr(Train, "run", observe_run)
    destination = ArtifactDestination(tmp_path / "successor-physical")
    planned = PlannedYield(2, 3)
    code = run_from_config(
        successor,
        restore_request=request,
        artifact_destination=destination,
        runtime_callbacks=(planned, readback),
        raise_exceptions=True,
    )
    assert code == PLANNED_YIELD_EXIT_CODE
    assert captured_h == [3]
    assert len(readback.events) == 1
    result = readback.events[0].result
    assert result.status == "planned_yield"
    assert isinstance(result.checkpoint_ref, CheckpointRef)
    assert (result.next_iteration, result.completed_updates, result.target_iterations) == (2, 2, 3)
    assert json.loads((destination.physical_root / "run-outcome.json").read_text())["status"] == "planned_yield"
    assert _leaf_snapshot(predecessor) == before


def test_s4_f3_yield_callback_admitted_in_routed_mode(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path / "logical", "routed")
    destination = ArtifactDestination(tmp_path / "physical")
    code = run_from_config(
        cfg,
        artifact_destination=destination,
        runtime_callbacks=(PlannedYield(1, 3),),
        raise_exceptions=True,
    )
    assert code == PLANNED_YIELD_EXIT_CODE
    assert json.loads((destination.physical_root / "run-outcome.json").read_text())["status"] == "planned_yield"
