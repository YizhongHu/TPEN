"""Tests for the optional readout trajectory replay generator."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.evaluation.generators import ReadoutTrajectoryMCMCGenerator, TrajectoryMCMCGenerator
from tpen.evaluation.protocols import EvaluationContext
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import HarmonicTrap
from tpen.sampling.metropolis import MetropolisSampler


class _FixedReadout(nn.Module):
    def __init__(self, *, mismatch: bool = False) -> None:
        super().__init__()
        self.readout = SimpleNamespace(eps=1.0e-12)
        self.mismatch = mismatch

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        radius_squared = batch.positions.square().sum(dim=(1, 2))
        raw = torch.full_like(radius_squared, 5.0e-7)
        logabs = -0.25 * radius_squared + 0.5 * torch.log(torch.full_like(raw, 1.0e-12))
        if self.mismatch and not torch.is_grad_enabled():
            logabs = logabs + 1.0
        return WavefunctionOutput(
            logabs=logabs,
            sign=torch.ones_like(raw),
            aux={"pfaffian": raw},
        )


def _context(directory) -> EvaluationContext:
    return EvaluationContext(
        namespace="eval",
        artifact_level="records",
        task_failure_policy="fail_fast",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=11,
        run_dir=directory,
        task_output_dir=directory,
        metadata={},
    )


def _kwargs() -> dict[str, object]:
    return {
        "sampler": MetropolisSampler(
            n_walkers=4,
            burn_in=2,
            n_steps=2,
            n_electrons=1,
            spatial_dim=3,
            nuclear_positions=torch.zeros(1, 3, dtype=torch.float64),
            nuclear_charges=torch.ones(1, dtype=torch.float64),
            seed=11,
            dtype=torch.float64,
        ),
        "hamiltonian_terms": {"kinetic": KineticEnergy(), "trap": HarmonicTrap(omega=0.5)},
        "n_draws": 3,
        "reset": True,
    }


def test_replay_adds_sidecar_without_changing_trajectory_csv(tmp_path):
    model = _FixedReadout().eval()
    plain_dir = tmp_path / "plain"
    replay_dir = tmp_path / "replay"
    plain_dir.mkdir()
    replay_dir.mkdir()
    plain = TrajectoryMCMCGenerator(**_kwargs()).generate(model=model, context=_context(plain_dir))
    replay = ReadoutTrajectoryMCMCGenerator(
        **_kwargs(), readout_eps=1.0e-12, replay_chunk_size=5
    ).generate(model=model, context=_context(replay_dir))
    assert plain.trajectory_records is not None
    assert replay.trajectory_records is not None
    assert (plain_dir / "sampled_eval_table.csv").read_bytes() == (
        replay_dir / "sampled_eval_table.csv"
    ).read_bytes()
    rows = list(csv.DictReader((replay_dir / "raw_readout.csv").open(newline="")))
    assert len(rows) == replay.trajectory_records.row_count == 12
    assert [int(row["sample_index"]) for row in rows] == list(range(12))
    assert all(row["floor_active"] == "True" for row in rows)
    assert json.loads((replay_dir / "raw_readout.json").read_text())["floor_count"] == 12


@pytest.mark.parametrize("mode,reason", [("mismatch", "disagrees")])
def test_replay_refuses_logabs_mismatch(tmp_path, mode, reason):
    with pytest.raises(ValueError, match=reason):
        ReadoutTrajectoryMCMCGenerator(
            **_kwargs(), readout_eps=1.0e-12
        ).generate(model=_FixedReadout(**{mode: True}).eval(), context=_context(tmp_path))
    assert not (tmp_path / "raw_readout.json").exists()


def test_replay_refuses_non_record_artifacts(tmp_path):
    context = _context(tmp_path)
    context = EvaluationContext(
        **{**context.__dict__, "artifact_level": "summaries"}
    )
    with pytest.raises(ValueError, match="require records"):
        ReadoutTrajectoryMCMCGenerator(**_kwargs(), readout_eps=1.0e-12).generate(
            model=_FixedReadout().eval(), context=context
        )
