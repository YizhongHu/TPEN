"""Focused review probes for readout trajectory replay."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.batch.walkers import Walkers
from tpen.data.real import Feature, zero_block
from tpen.evaluation.generators import ReadoutTrajectoryMCMCGenerator
from tpen.evaluation.protocols import EvaluationContext
from tpen.nn.envelope import GaussianConfinement
from tpen.nn.readout import PfaffianReadout
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import HarmonicTrap
from tpen.sampling.stats import SamplerStats


class _FixedTrajectorySampler:
    """Small deterministic sampler that still exercises record collection."""

    def __init__(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.atomic_configuration = AtomicConfiguration(
            positions=torch.zeros(1, positions.shape[-1], dtype=positions.dtype),
            charges=torch.ones(1, dtype=positions.dtype),
        )

    def collect_samples(self, model, *, reset: bool, device: torch.device):
        del model, reset
        positions = self.positions.to(device)
        walkers = Walkers(
            positions=positions,
            spins=torch.ones(positions.shape[:2], device=device, dtype=positions.dtype),
            atomic_configuration=self.atomic_configuration.to(device=device),
        )
        return walkers, SamplerStats(
            acceptance_rate=1.0,
            n_walkers=positions.shape[0],
            burn_in=0,
            n_steps=1,
            proposal_scale=0.0,
            geometry={},
            seed=17,
        )


class _PfaffianBoundaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readout = PfaffianReadout(eps=1.0e-12, pair_channels=1).to(dtype=torch.float32)
        self.confinement = GaussianConfinement(coefficient=0.25)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        x = batch.positions[:, 0, 0]
        # The exact boundary value is intentional: float32 square and eps
        # tensor conversion are the arithmetic the readout uses.
        raw = torch.where(
            x < 0,
            torch.full_like(x, 1.0e-6),
            torch.where(torch.abs(x) < 0.5, torch.full_like(x, 1.001e-6), torch.full_like(x, 0.999e-6)),
        )
        features = Feature(
            [
                zero_block(batch.batch_size, dtype=batch.dtype),
                raw.reshape(-1, 1, 1),
                torch.zeros(batch.batch_size, 1, 1, 1, dtype=batch.dtype),
            ]
        )
        output = self.readout(features, batch)
        return replace(output, logabs=output.logabs + self.confinement(batch))


class _LinearPfaffianModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=True, dtype=torch.float32)
        nn.init.constant_(self.linear.weight, 0.03125)
        nn.init.constant_(self.linear.bias, 0.75)
        self.readout = PfaffianReadout(eps=1.0e-12, pair_channels=1).to(dtype=torch.float32)
        self.confinement = GaussianConfinement(coefficient=0.25)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        raw = self.linear(batch.positions[:, 0]).reshape(-1)
        features = Feature(
            [
                zero_block(batch.batch_size, dtype=batch.dtype),
                raw.reshape(-1, 1, 1),
                torch.zeros(batch.batch_size, 1, 1, 1, dtype=batch.dtype),
            ]
        )
        output = self.readout(features, batch)
        return replace(output, logabs=output.logabs + self.confinement(batch))


def _context(directory, *, dtype: torch.dtype = torch.float32) -> EvaluationContext:
    return EvaluationContext(
        namespace="eval",
        artifact_level="records",
        task_failure_policy="fail_fast",
        device=torch.device("cpu"),
        dtype=dtype,
        seed=17,
        run_dir=directory,
        task_output_dir=directory,
        metadata={},
    )


def _generator(sampler, *, chunk_size: int) -> ReadoutTrajectoryMCMCGenerator:
    return ReadoutTrajectoryMCMCGenerator(
        sampler=sampler,
        hamiltonian_terms={"kinetic": KineticEnergy(), "trap": HarmonicTrap(omega=0.5)},
        n_draws=2,
        reset=True,
        readout_eps=1.0e-12,
        replay_chunk_size=chunk_size,
    )


def test_float32_floor_flags_follow_pfaffian_tensor_arithmetic_and_bind_hashes(tmp_path):
    positions = torch.tensor(
        [[[-1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=torch.float32
    )
    model = _PfaffianBoundaryModel().eval()
    output = model(
        ElectronBatch(
            positions=positions,
            spins=torch.ones(3, 1, dtype=torch.float32),
            atomic_configuration=AtomicConfiguration(
                positions=torch.zeros(1, 3), charges=torch.ones(1)
            ),
        )
    )
    raw = output.aux["pfaffian"]
    expected_flags = raw.square() < torch.as_tensor(model.readout.eps, dtype=raw.dtype)
    assert raw.dtype is torch.float32
    assert expected_flags.tolist() == [False, False, True]

    directory = tmp_path / "boundary"
    directory.mkdir()
    generated = _generator(_FixedTrajectorySampler(positions), chunk_size=5).generate(
        model=model, context=_context(directory)
    )
    sidecar = json.loads((directory / "raw_readout.json").read_text())
    rows = list(csv.DictReader((directory / "raw_readout.csv").open(newline="")))
    assert [row["floor_active"] == "True" for row in rows[:3]] == expected_flags.tolist()
    assert sidecar["trajectory_sha256"] == hashlib.sha256(
        generated.trajectory_records.path.read_bytes()
    ).hexdigest()
    assert sidecar["csv_sha256"] == hashlib.sha256(
        (directory / "raw_readout.csv").read_bytes()
    ).hexdigest()
    assert sidecar["row_count"] == len(rows) == generated.trajectory_records.row_count


def test_float32_replay_accepts_different_chunk_sizes_and_hash_is_independent(tmp_path):
    positions = torch.tensor(
        [
            [[0.1, -0.2, 0.3]],
            [[-0.4, 0.5, -0.6]],
            [[0.7, -0.8, 0.9]],
            [[-1.0, 1.1, -1.2]],
            [[1.3, -1.4, 1.5]],
        ],
        dtype=torch.float32,
    )
    outputs = []
    for chunk_size in (1, 5):
        directory = tmp_path / f"chunk-{chunk_size}"
        directory.mkdir()
        generated = _generator(_FixedTrajectorySampler(positions), chunk_size=chunk_size).generate(
            model=_LinearPfaffianModel().eval(), context=_context(directory)
        )
        sidecar = json.loads((directory / "raw_readout.json").read_text())
        trajectory_bytes = generated.trajectory_records.path.read_bytes()
        raw_bytes = (directory / "raw_readout.csv").read_bytes()
        rows = list(csv.DictReader((directory / "raw_readout.csv").open(newline="")))
        assert sidecar["trajectory_sha256"] == hashlib.sha256(trajectory_bytes).hexdigest()
        assert sidecar["csv_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
        assert sidecar["row_count"] == 10
        outputs.append(rows)
    assert [[row["sample_index"], row["draw_index"], row["walker_index"], row["floor_active"]] for row in outputs[0]] == [
        [row["sample_index"], row["draw_index"], row["walker_index"], row["floor_active"]]
        for row in outputs[1]
    ]
    assert [float(row["raw_channel_sum"]) for row in outputs[0]] == pytest.approx(
        [float(row["raw_channel_sum"]) for row in outputs[1]], rel=1.0e-6, abs=1.0e-7
    )
