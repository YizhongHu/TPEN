"""S3 acceptance: real HI resolved config through operational artifact routing.

Exercises the real ``train_config.resolve_train_config`` output -- the exact
flat-layout, intermediate-stage-segment shape that ``ArtifactManager``'s bare
``root/run_id`` recompute does NOT reproduce (F1) -- through the real
``tpen.run.run_from_config`` -> ``prepare_run_context`` route with a claimed
physical destination. Only final runner construction is stubbed (as
``test_run_stage_q_core_route.py`` does), so the full HI sink set
(``ConfigSnapshot``, ``ResolvedConfigSnapshot``, ``Metadata``, ``Status``,
health/equivariance callbacks, ``Checkpoint``, ``CSV``/``JSONL`` loggers) is
constructed and routed for real.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip("torch")

from tpen.run import run_from_config

# Keep the experiment boundary sanctioned: recover everything else from the
# already-imported defining module instead of adding tpen.* imports here.
tpen_run = sys.modules[run_from_config.__module__]
ArtifactDestination = tpen_run.ArtifactDestination
RunResult = tpen_run.RunResult
Runner = tpen_run.Runner


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load("s3_routing_stage_coordinate", "stage_coordinate.py")
train_config = _load("s3_routing_train_config", "train_config.py")


def _q_cell(tmp_path: Path) -> object:
    """Materialize one real Q-stage cell -- a fixed-horizon, flat-layout leaf."""

    levels = {
        "lr": 0.005,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "parameter_groups": 1,
        "schedule": "constant",
    }
    return stage_coordinate.materialize_stage(
        "Q",
        [
            {
                "scientific_identity": {
                    "context": {
                        "name": "O-raw",
                        "overrides": {"input_features": "B00", "feature_update": "residual"},
                    },
                    "optimizer": {"method": "adam", "levels": levels},
                },
                "payload": {"updates": stage_coordinate.stage_definition("Q").updates},
                "topology": {},
            }
        ],
        stage_coordinate.OptimizerCell("adam", "available", None),
        tmp_path,
    )[0]


class _StubRunner(Runner):
    """Final runner stub -- no model is built or trained; see module docstring."""

    def run(self, context: object) -> RunResult:
        del context
        return RunResult(status="completed")


def test_real_hi_resolved_config_routes_every_sink_under_the_claimed_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _q_cell(tmp_path / "cells")
    cfg = train_config.resolve_train_config(cell)

    admitted_run_dir = Path(str(OmegaConf.select(cfg, "run.dir")))
    admitted_run_root = Path(str(OmegaConf.select(cfg, "run.root")))
    admitted_run_id = str(OmegaConf.select(cfg, "run.run_id"))
    h = int(OmegaConf.select(cfg, "trainer.max_steps"))

    # Pin the exact F1 scenario: HI's admitted directory includes an
    # intermediate stage segment that a bare flat-layout root/run_id
    # recompute would miss.
    assert admitted_run_dir != admitted_run_root / admitted_run_id
    assert admitted_run_dir.parent.name == cell.manifest["stage"]

    captured: dict[str, object] = {}

    def spy_instantiate(context: object) -> Runner:
        captured["context"] = context
        return _StubRunner()

    monkeypatch.setattr(tpen_run, "_instantiate_runner", spy_instantiate)

    physical_root = tmp_path / "allocation-1"
    physical_root.mkdir()
    destination = ArtifactDestination(physical_root=physical_root)

    result = run_from_config(cfg, artifact_destination=destination, raise_exceptions=True)
    assert result == 0

    context = captured["context"]
    # C -- including run.dir and the live trainer horizon -- is the admitted
    # value, never the recomputed one, and H is unchanged by routing.
    assert context.cfg.run.dir == str(admitted_run_dir)
    assert int(OmegaConf.select(context.cfg, "trainer.max_steps")) == h
    # Physical writes land at the claimed leaf, not the admitted directory.
    assert context.run_dir == physical_root
    assert context.run_dir != admitted_run_dir

    sink_fields = ("output_path", "output_dir", "path", "artifact_dir")
    saw_any_sink = False
    for sink in [*context.callbacks, *context.loggers]:
        for field in sink_fields:
            value = getattr(sink, field, None)
            if value is None:
                continue
            value_path = Path(value)
            assert physical_root == value_path or physical_root in value_path.parents
            saw_any_sink = True
    assert saw_any_sink

    resolved_snapshot_path = physical_root / "resolved_config.yaml"
    assert resolved_snapshot_path.is_file()
    emitted_run = OmegaConf.load(resolved_snapshot_path)
    assert emitted_run.run.dir == str(admitted_run_dir)
    assert int(emitted_run.trainer.max_steps) == h

    checkpoints_dir = physical_root / "checkpoints"
    assert checkpoints_dir.is_dir()

    # The admitted logical directory itself is never written to.
    assert not admitted_run_dir.exists()


def test_a_second_allocation_of_the_same_cell_preserves_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _q_cell(tmp_path / "cells")
    cfg = train_config.resolve_train_config(cell)

    monkeypatch.setattr(tpen_run, "_instantiate_runner", lambda context: _StubRunner())

    first_physical = tmp_path / "allocation-1"
    first_physical.mkdir()
    run_from_config(
        cfg,
        artifact_destination=ArtifactDestination(physical_root=first_physical),
        raise_exceptions=True,
    )
    before = sorted(p.relative_to(first_physical) for p in first_physical.rglob("*"))
    before_bytes = {
        str(p): (first_physical / p).read_bytes()
        for p in before
        if (first_physical / p).is_file()
    }

    second_physical = tmp_path / "allocation-2"
    second_physical.mkdir()
    run_from_config(
        cfg,
        artifact_destination=ArtifactDestination(physical_root=second_physical),
        raise_exceptions=True,
    )

    after = sorted(p.relative_to(first_physical) for p in first_physical.rglob("*"))
    after_bytes = {
        str(p): (first_physical / p).read_bytes()
        for p in after
        if (first_physical / p).is_file()
    }
    assert after == before
    assert after_bytes == before_bytes
