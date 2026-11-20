"""Stage-1 overlay integration discriminator for amplitude-risk cadence."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import WavefunctionOutput
from tpen.training.events import TrainingIteration, TrainingIterationCompleted
from tests.helpers.run_context import RecordingLogger, make_run_context
from tests.unit.callback.support import training_state


OVERLAY = (
    Path(__file__).resolve().parents[3]
    / "experiments/atomistic/stage1/configs/overlays/hooke-diagnostics-v1.yaml"
)


def test_stage1_hooke_amplitude_cadence_uses_0_99_100_pre_update_events(
    tmp_path: Path,
) -> None:
    """The composed Hooke callback fires at 0 and 100, using each batch once."""

    overlay = OmegaConf.load(OVERLAY)
    amplitude_cfg = next(
        entry for entry in overlay.callbacks
        if entry.get("_target_") == "tpen.callback.AmplitudeRiskFraction"
    )
    callback = instantiate(amplitude_cfg)
    logger = RecordingLogger()
    context = make_run_context(tmp_path, callbacks=[callback], loggers=[logger])
    state = training_state(model=lambda *_args, **_kwargs: pytest.fail("unexpected forward"))
    for step, values in (
        (0, [0.0, 0.0, -10.0]),
        (99, [100.0, 100.0, -10.0]),
        (100, [0.0, 0.0, 0.0]),
    ):
        state.wavefunction_output = WavefunctionOutput(
            logabs=torch.tensor(values, dtype=torch.float64),
            sign=torch.ones(3, dtype=torch.float64),
        )
        context.emit(
            TrainingIterationCompleted(iteration=TrainingIteration(step=step)),
            state=state,
        )

    records = logger.by_namespace("train/amplitude_risk")
    assert [record.step for record in records] == [0, 100]
    assert [record.metrics["total_count"] for record in records] == [3, 3]
    assert [record.metrics["low_amplitude_count.-3.0"] for record in records] == [1, 0]
