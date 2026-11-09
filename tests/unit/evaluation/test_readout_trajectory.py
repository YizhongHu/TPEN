"""Tests for the optional readout trajectory replay generator."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
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
    def __init__(
        self,
        *,
        mismatch: bool = False,
        sign_mismatch: bool = False,
        raw_value: float = 5.0e-7,
        readout_eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        self.readout = SimpleNamespace(eps=readout_eps)
        self.mismatch = mismatch
        self.sign_mismatch = sign_mismatch
        self.raw_value = raw_value

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        radius_squared = batch.positions.square().sum(dim=(1, 2))
        raw = torch.full_like(radius_squared, self.raw_value)
        logabs = -0.25 * radius_squared + torch.log(torch.full_like(raw, self.raw_value))
        if self.mismatch and not torch.is_grad_enabled():
            logabs = logabs + 1.0
        return WavefunctionOutput(
            logabs=logabs,
            sign=torch.full_like(raw, -1.0 if self.sign_mismatch else 1.0),
            aux={"pfaffian": raw},
        )


class _DriftingReadout(_FixedReadout):
    def __init__(self) -> None:
        super().__init__(raw_value=5.0e-6)
        self.drift = nn.Parameter(torch.zeros(()))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        if not torch.is_grad_enabled():
            with torch.no_grad():
                self.drift.add_(1.0)
        return super().forward(batch)


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


def test_ordinary_model_raw_sum_agrees_with_logabs_sign_and_has_no_floor_hits(tmp_path):
    positions = torch.tensor(
        [[[0.2, 0.0, 0.0]], [[-0.4, 0.1, 0.0]], [[0.7, -0.2, 0.3]]], dtype=torch.float64
    )
    directory = tmp_path / "ordinary"
    directory.mkdir()
    _generator(_FixedTrajectorySampler(positions), chunk_size=2).generate(
        model=_FixedReadout(raw_value=5.0e-6).eval(), context=_context(directory)
    )
    rows = list(csv.DictReader((directory / "raw_readout.csv").open(newline="")))
    assert all(row["floor_active"] == "False" for row in rows)
    for row in rows:
        radius_squared = sum(
            float(row[f"position/electron_0/axis_{axis}"]) ** 2 for axis in range(3)
        )
        assert float(row["raw_channel_sum"]) == pytest.approx(5.0e-6)
        assert float(row["logabs"]) == pytest.approx(
            torch.log(torch.tensor(5.0e-6)).item() - 0.25 * radius_squared,
            rel=1.0e-12,
            abs=1.0e-12,
        )
        assert float(row["sign"]) == 1.0


def test_forced_floor_raw_sums_and_flags_match_independent_oracle(tmp_path):
    positions = torch.tensor(
        [[[-1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=torch.float64
    )
    directory = tmp_path / "floor"
    directory.mkdir()
    _generator(_FixedTrajectorySampler(positions), chunk_size=2).generate(
        model=_FixedReadout().eval(), context=_context(directory)
    )
    rows = list(csv.DictReader((directory / "raw_readout.csv").open(newline="")))
    expected_raw = [5.0e-7] * len(rows)
    expected_flags = [value * value < 1.0e-12 for value in expected_raw]
    assert [float(row["raw_channel_sum"]) for row in rows] == pytest.approx(expected_raw)
    assert [row["floor_active"] == "True" for row in rows] == expected_flags


@pytest.mark.parametrize(
    ("model", "generator_eps", "reason"),
    [
        (_FixedReadout(readout_eps=2.0e-12).eval(), 1.0e-12, "does not match"),
        (_FixedReadout(sign_mismatch=True).eval(), 1.0e-12, "disagrees"),
    ],
)
def test_replay_refuses_incompatible_readout_configuration_or_sign(
    tmp_path, model, generator_eps, reason
):
    with pytest.raises(ValueError, match=reason):
        ReadoutTrajectoryMCMCGenerator(
            **_kwargs(), readout_eps=generator_eps
        ).generate(model=model, context=_context(tmp_path))


def test_replay_refuses_training_mode(tmp_path):
    with pytest.raises(ValueError, match="eval-mode"):
        ReadoutTrajectoryMCMCGenerator(**_kwargs(), readout_eps=1.0e-12).generate(
            model=_FixedReadout().train(), context=_context(tmp_path)
        )


def test_replay_refuses_parameter_fingerprint_change(tmp_path):
    with pytest.raises(ValueError, match="changed"):
        ReadoutTrajectoryMCMCGenerator(**_kwargs(), readout_eps=1.0e-12).generate(
            model=_DriftingReadout().eval(), context=_context(tmp_path)
        )


def test_replay_refuses_incomplete_replay(monkeypatch, tmp_path):
    from tpen.evaluation.trajectory_records import TrajectoryRecordArtifact

    original_generate = TrajectoryMCMCGenerator.generate

    def truncated_generate(self, *, model, context):
        generated = original_generate(self, model=model, context=context)
        assert generated.trajectory_records is not None
        return replace(
            generated,
            trajectory_records=replace(
                generated.trajectory_records,
                row_count=generated.trajectory_records.row_count - 1,
            ),
        )

    monkeypatch.setattr(TrajectoryRecordArtifact, "validate", lambda self, **kwargs: self)
    monkeypatch.setattr(TrajectoryMCMCGenerator, "generate", truncated_generate)
    with pytest.raises(ValueError, match="omitted trajectory rows"):
        ReadoutTrajectoryMCMCGenerator(**_kwargs(), readout_eps=1.0e-12).generate(
            model=_FixedReadout().eval(), context=_context(tmp_path)
        )
