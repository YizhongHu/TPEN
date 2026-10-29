"""Readout replay must retain the channel sum and every draw, or refuse."""

from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.evaluation.generators.readout_trajectory import ReadoutTrajectoryMCMCGenerator
from tpen.evaluation.protocols import EvaluationContext
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import HarmonicTrap
from tpen.sampling.metropolis import MetropolisSampler


class CancelledChannels(nn.Module):
    """Two large channels cancel into a floored, nonzero raw sum."""

    def __init__(self, *, missing=False, mismatch=False):
        super().__init__()
        self.readout = SimpleNamespace(eps=1e-12)
        self.missing = missing
        self.mismatch = mismatch

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        radius_squared = batch.positions.square().sum(dim=(1, 2))
        raw = torch.full_like(radius_squared, 5e-7)
        logabs = -.25 * radius_squared + .5 * math.log(1e-12)
        if self.mismatch and not torch.is_grad_enabled():
            logabs = logabs + 1
        aux = {"channel_pfaffians": torch.stack((torch.ones_like(raw), raw - 1), dim=1)}
        if not self.missing:
            aux["pfaffian"] = raw
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(raw), aux=aux)


def _context(directory):
    return EvaluationContext(namespace="eval", artifact_level="records", task_failure_policy="fail_fast",
                             device=torch.device("cpu"), dtype=torch.float64, seed=11,
                             run_dir=directory, task_output_dir=directory, metadata={})


def _generator():
    return ReadoutTrajectoryMCMCGenerator(
        sampler=MetropolisSampler(n_walkers=4, burn_in=2, n_steps=2, n_electrons=1,
                                  spatial_dim=3, seed=11, dtype=torch.float64),
        hamiltonian_terms={"kinetic": KineticEnergy(), "trap": HarmonicTrap(omega=.5)},
        n_draws=3, reset=True, readout_eps=1e-12, replay_chunk_size=5,
        allow_nucleus_free_records=True,
    )


def test_sum_floor_and_every_retained_row_are_recorded(tmp_path):
    generated = _generator().generate(model=CancelledChannels().eval(), context=_context(tmp_path))
    assert generated.trajectory_records.row_count == 12
    with (tmp_path / "raw_readout.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert [int(row["sample_index"]) for row in rows] == list(range(12))
    assert all(float(row["raw_channel_sum"]) == 5e-7 for row in rows)
    assert all(row["floor_active"] == "True" for row in rows)
    assert json.loads((tmp_path / "raw_readout.json").read_text())["floor_count"] == 12
    metadata_path = tmp_path / "sampled_eval_table.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema"] == "trajectory_records/v2"
    assert metadata["atomic_configuration"] is None
    assert metadata["atomic_configuration_id"] is None
    # Missing geometry is different from a recorded absence of nuclei.
    del metadata["atomic_configuration"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="atomic_configuration"):
        generated.trajectory_records.validate(check_files=True)


def test_nucleus_free_recording_requires_explicit_admission(tmp_path):
    generator = _generator()
    generator.allow_nucleus_free_records = False
    with pytest.raises(ValueError, match="require Walkers.atomic_configuration"):
        generator.generate(model=CancelledChannels().eval(), context=_context(tmp_path))


@pytest.mark.parametrize("mode,reason", [("missing", "channel-summed"), ("mismatch", "disagrees")])
def test_missing_aux_or_replay_mismatch_refuses_complete_receipt(tmp_path, mode, reason):
    model = CancelledChannels(**{mode: True}).eval()
    with pytest.raises(ValueError, match=reason):
        _generator().generate(model=model, context=_context(tmp_path))
    assert (tmp_path / "sampled_eval_table.csv").is_file()
    assert not (tmp_path / "raw_readout.json").exists()


def test_wrong_floor_is_refused_before_sampling(tmp_path):
    model = CancelledChannels().eval()
    model.readout.eps = 1e-8
    with pytest.raises(ValueError, match="floor does not match"):
        _generator().generate(model=model, context=_context(tmp_path))
    assert not (tmp_path / "sampled_eval_table.csv").exists()
