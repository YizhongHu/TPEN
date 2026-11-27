"""Bounded physics, real planner admission and native segmented continuation."""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

from experiments.atomistic.stage1 import protocol, restart
from experiments.atomistic.stage1.tests.test_segmented_act50_profiles import _rows, STUDY
from experiments.atomistic.stage1.train import build_train_config
from tpen.callback.raw_training_batch_recorder import TrainingHistoryReader
from tpen.run import run_from_config


def _bounded_physics_run(cfg, **kwargs):
    """Substitute cheap physics only after the real Stage-1 admission boundary.

    This is an integration test, not a production model/throughput claim. Keep
    the admitted horizon, run identity, checkpoint schedule and recorder intact.
    """
    bounded = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bounded.model = {
        "_target_": "tests.helpers.trainable_wavefunctions.TrainableHookeSingletAnsatz",
    }
    bounded.runtime.device = "cpu"
    bounded.sampler.n_walkers = 2
    bounded.sampler.burn_in = 1
    bounded.sampler.n_steps = 1
    # Model-specific diagnostic groups do not exist on the small test ansatz.
    omitted = {"ModuleNorms", "OptimizerState", "ParameterActivity", "AmplitudeRiskFraction"}
    bounded.callbacks = [cb for cb in bounded.callbacks
                         if cb.get("_target_", "").split(".")[-1] not in omitted]
    return run_from_config(bounded, **kwargs)


def test_planner_segmented_native_continuation_preserves_history(tmp_path: Path) -> None:
    """Two retained prefixes cross an unscheduled last iteration (999).

    The science H remains 50000; both bounded executions yield rather than
    pretending to finish science. Production recorder cadence stays 50/1000.
    """
    source = _rows(protocol, "hooke2-sigmoid50k-seg-v1", str(tmp_path / "source"))[0]
    first = protocol.derive_smoke_request(source, results_root=str(tmp_path / "seg"), stop_at=1000)
    second = {**first, "execution": {"stop_at": 2000}}
    whole = protocol.derive_smoke_request(source, results_root=str(tmp_path / "whole"), stop_at=2000)
    config_path = STUDY / "configs/hooke2/train.yaml"
    for row in (first, whole):
        assert restart.execute_train(
            cfg=build_train_config(row), row=row, config_path=config_path,
            run=_bounded_physics_run,
        ) == 2
    history = protocol.resolve_run_dir(first) / "training_history"
    parent_bytes = {p.relative_to(history): p.read_bytes()
                    for p in history.rglob("*") if p.is_file()}
    parent = TrainingHistoryReader(history)
    parent.validate(require_complete=True)
    assert set(parent.read()["iteration_step"]) == set(range(0, 1000, 50))
    assert restart.execute_train(
        cfg=build_train_config(second), row=second, config_path=config_path,
        run=_bounded_physics_run,
    ) == 2
    assert {name: (history / name).read_bytes() for name in parent_bytes} == parent_bytes
    segments = list((history / "segments").iterdir())
    assert len(segments) == 1
    assert segments[0].name.startswith("seg_") and "step_" in segments[0].name
    reader = TrainingHistoryReader(history)
    reader.validate(require_complete=True)
    actual = reader.read()
    expected = TrainingHistoryReader(protocol.resolve_run_dir(whole) / "training_history").read()
    assert actual.keys() == expected.keys()
    for column in actual:
        np.testing.assert_array_equal(actual[column], expected[column], err_msg=column)
    identities = list(zip(actual["iteration_step"], actual["walker_index"]))
    assert len(identities) == len(set(identities)) == 40 * 2
    assert set(actual["iteration_step"]) == set(range(0, 2000, 50))

    # Whitespace is valid JSON but changes the header's hash-bound bytes.
    # A root reader must reject a damaged child, not fall back to its parent.
    child_header = segments[0] / "history.json"
    original_header = child_header.read_bytes()
    child_header.write_bytes(original_header + b" ")
    try:
        with pytest.raises(ValueError):
            TrainingHistoryReader(history).validate(require_complete=True)
        with pytest.raises(ValueError):
            TrainingHistoryReader(history).read()
    finally:
        child_header.write_bytes(original_header)
    TrainingHistoryReader(history).validate(require_complete=True)


def test_planner_v2_continuation_refuses_before_inspection(tmp_path: Path) -> None:
    """Legacy planner row is refused before inspecting or running a continuation."""
    row = _rows(protocol, "hooke2-sigmoid50k-v1", str(tmp_path))[0]
    protocol.resolve_run_dir(row).mkdir(parents=True)
    def forbidden(*args, **kwargs):
        pytest.fail("v2 continuation inspected or launched")
    with pytest.raises(restart.RestartRefusal, match="RawTrainingBatchRecorder|recorder"):
        restart.execute_train(
            cfg=build_train_config(row), row=row,
            config_path=STUDY / "configs/hooke2/train.yaml",
            inspector=forbidden, run=forbidden,
        )
