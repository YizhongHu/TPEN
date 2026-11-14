"""Real constructed routed run (S3 round 3, review-r2 F / R2.6).

The HI-based ``experiments/atomistic/he-importance/test_s3_artifact_routing_constructed_run.py``
stubs final runner construction entirely, so it never reads a live trainer or
publishes a checkpoint. This module instead goes through the real public
entry (``tpen.run.run_from_config``) with the core integration fixture
(``tests/integration/artifacts/training/vmc_smoke.yaml``), a real ``Train``
and ``VMCTrainer``, and a claimed ``ArtifactDestination`` -- then resumes a
second routed allocation through a real S2 ``OperationalRestoreRequest`` --
exactly the coverage review-r2 R2.6 found missing.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from tpen.artifacts import ArtifactDestination, resolve_run_clock
from tpen.checkpoint import CheckpointRef, CheckpointRestored
from tpen.checkpoint.hashing import populate_resolved_config_defaults
from tpen.execution.recovery import (
    CheckpointEvidence,
    OperationalRestoreRequest,
    RecoveryPhase,
    ScienceIntent,
    select_recovery,
)
from tpen.run import run_from_config
from tpen.runner import Train

_FIXTURE = Path("tests/integration/artifacts/training/vmc_smoke.yaml")


def _fixture_cfg(root: Path, run_id: str) -> OmegaConf:
    cfg = OmegaConf.load(_FIXTURE)
    cfg.run.root = str(root)
    cfg.run.run_id = run_id
    cfg.trainer.max_steps = 2
    cfg.sampler.n_walkers = 8
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    for callback in cfg.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback.schedule.every_n = 1
    return cfg


def _expected_resolved_yaml(cfg: OmegaConf, run_dir: Path) -> str:
    """Reconstruct the admitted C ``prepare_run_context`` must record.

    Mirrors ``prepare_run_context``'s own resolution exactly: the only leaf
    it is entitled to add over the supplied config is the one admitted
    ``run.dir`` -- everything else must reach the emitted resolved config
    unchanged.
    """

    run_clock = resolve_run_clock(cfg)
    expected = OmegaConf.create(populate_resolved_config_defaults(cfg, timezone=run_clock.timezone))
    OmegaConf.update(expected, "run.dir", str(run_dir), merge=False, force_add=True)
    return OmegaConf.to_yaml(expected, resolve=True)


def test_real_constructed_routed_run_preserves_c_and_resumes_byte_identically(
    tmp_path: Path, monkeypatch
) -> None:
    logical_root = tmp_path / "logical"
    cfg = _fixture_cfg(logical_root, "cell")
    # Nested layout (the fixture's default): root/experiment/sector/run_id.
    expected_run_dir = logical_root / "vmc_smoke" / "singlet" / "cell"
    expected_c = _expected_resolved_yaml(cfg, expected_run_dir)

    physical_first = tmp_path / "allocation-1"
    destination_first = ArtifactDestination(physical_root=physical_first)

    captured: dict[str, object] = {}
    original_run = Train.run

    def observe_run(self, context):
        captured["context"] = context
        captured["live_max_steps"] = self.trainer.max_steps
        return original_run(self, context)

    monkeypatch.setattr(Train, "run", observe_run)

    assert run_from_config(cfg, artifact_destination=destination_first, raise_exceptions=True) == 0

    context = captured["context"]
    # C -- the recorded run.dir -- is the constant logical value, never the
    # physical one; the live trainer's horizon matches the supplied H.
    assert context.cfg.run.dir == str(expected_run_dir)
    assert captured["live_max_steps"] == 2
    # Physical writes land at the claimed leaf, not the logical directory.
    assert context.run_dir == physical_first
    assert not expected_run_dir.exists()

    resolved_path = physical_first / "resolved_config.yaml"
    assert resolved_path.is_file()
    emitted = OmegaConf.load(resolved_path)
    assert OmegaConf.to_yaml(emitted, resolve=True) == expected_c

    (checkpoint,) = physical_first.rglob("step_000001")
    assert (physical_first / "metrics.csv").is_file()
    assert (physical_first / "metrics.jsonl").is_file()

    before = sorted(p.relative_to(physical_first) for p in physical_first.rglob("*"))
    before_bytes = {
        str(p): (physical_first / p).read_bytes() for p in before if (physical_first / p).is_file()
    }

    intent = ScienceIntent({"fixture": "s3r3-vmc-smoke"}, target_iterations=2)
    evidence = CheckpointEvidence.from_ref(
        CheckpointRef.from_directory(checkpoint),
        science_input_digest=intent.science_input_digest,
        target_iterations=2,
    )
    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)
    assert decision.refusal is None
    request = OperationalRestoreRequest(decision)

    second_cfg = _fixture_cfg(logical_root, "cell")
    physical_second = tmp_path / "allocation-2"
    destination_second = ArtifactDestination(physical_root=physical_second)

    reports = []

    class _Readback:
        def handle_occurrence(self, occurrence: object, context: object) -> None:
            if isinstance(occurrence.event, CheckpointRestored):
                reports.append(occurrence.event.report)

    # (R3.1) review-r3: runtime-sink admission is by EXACT type only, with no
    # attribute-probe inference. `_Readback` is a test double with no
    # production registry entry, so it must be explicitly admitted the same
    # way a real sink-free type is -- membership in the closed registry --
    # rather than relying on it merely lacking an output-path-shaped
    # attribute.
    import tpen.run as run_module

    qualified_readback = f"{_Readback.__module__}.{_Readback.__qualname__}"
    monkeypatch.setattr(
        run_module, "_SINK_FREE_TYPES", run_module._SINK_FREE_TYPES | {qualified_readback}
    )

    assert (
        run_from_config(
            second_cfg,
            artifact_destination=destination_second,
            restore_request=request,
            runtime_callbacks=(_Readback(),),
            raise_exceptions=True,
        )
        == 0
    )
    assert len(reports) == 1
    assert reports[0].next_iteration == 1
    assert (physical_second / "resolved_config.yaml").is_file()
    assert (physical_second / "checkpoints").is_dir()

    after = sorted(p.relative_to(physical_first) for p in physical_first.rglob("*"))
    after_bytes = {
        str(p): (physical_first / p).read_bytes() for p in after if (physical_first / p).is_file()
    }
    assert after == before
    assert after_bytes == before_bytes
