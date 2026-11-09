"""Optional readout replay diagnostics for retained trajectory records."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import replace

import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.evaluation.bundle import GeneratedConfigurations
from tpen.evaluation.generators.trajectory import TrajectoryMCMCGenerator
from tpen.evaluation.protocols import EvaluationContext
from tpen.sampling.trajectory import parameter_fingerprint


class ReadoutTrajectoryMCMCGenerator(TrajectoryMCMCGenerator):
    """Replay retained coordinates and record the unfloored readout sum.

    This is an opt-in evaluation generator selected by Hydra. It delegates
    trajectory collection to :class:`TrajectoryMCMCGenerator`, then replays
    exactly the serialized coordinates. The replay shares the learned forward
    pass and is not an independent energy oracle.

    Parameters
    ----------
    readout_eps : float
        Expected squared-amplitude floor, checked against ``model.readout``.
    replay_chunk_size : int, optional
        Maximum number of records in one replay forward pass.
    **kwargs
        Forwarded to :class:`TrajectoryMCMCGenerator`.

    Raises
    ------
    ValueError
        If the floor or replay chunk size is not positive.
    """

    name = "readout_trajectory_mcmc"

    def __init__(self, *, readout_eps: float, replay_chunk_size: int = 4096, **kwargs) -> None:
        super().__init__(**kwargs)
        if replay_chunk_size < 1 or not 0 < readout_eps < float("inf"):
            raise ValueError("readout floor and replay chunk size must be positive")
        self.readout_eps = float(readout_eps)
        self.replay_chunk_size = int(replay_chunk_size)

    def generate(
        self, *, model: torch.nn.Module | None, context: EvaluationContext
    ) -> GeneratedConfigurations:
        """Collect records, replay every row, and write the diagnostic sidecar."""

        if context.artifact_level != "records":
            raise ValueError("readout trajectory diagnostics require records")
        if model is None:
            raise TypeError("readout trajectory diagnostics require a model")
        if getattr(getattr(model, "readout", None), "eps", None) != self.readout_eps:
            raise ValueError("configured readout floor does not match the model")
        if model.training:
            raise ValueError("readout replay requires the evaluation runner's eval-mode model")

        fingerprint = parameter_fingerprint(model)
        generated = super().generate(model=model, context=context)
        records = generated.trajectory_records
        if records is None:
            raise ValueError("readout replay requires complete trajectory records")
        records.validate(check_files=True)
        flat = generated.batch.flatten_samples()
        if flat.spins is not None and not torch.equal(
            flat.spins, flat.spins[:1].expand_as(flat.spins)
        ):
            raise ValueError("readout replay requires a fixed spin population")

        path = context.task_output_dir / "raw_readout.csv"
        columns = [
            f"position/electron_{electron}/axis_{axis}"
            for electron in range(records.n_electrons)
            for axis in range(records.spatial_dim)
        ]
        row_count = floor_count = 0
        with records.path.open(newline="") as source, path.open("x", newline="") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(
                target,
                fieldnames=(
                    "sample_index",
                    "draw_index",
                    "walker_index",
                    "raw_channel_sum",
                    "floor_active",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            while rows := list(itertools.islice(reader, self.replay_chunk_size)):
                positions = torch.tensor(
                    [[float(row[column]) for column in columns] for row in rows],
                    device=flat.device,
                    dtype=flat.dtype,
                ).reshape(len(rows), records.n_electrons, records.spatial_dim)
                spins = None if flat.spins is None else flat.spins[:1].expand(len(rows), -1)
                batch = ElectronBatch(
                    positions=positions,
                    spins=spins,
                    system=flat.system,
                    atomic_configuration=flat.atomic_configuration,
                )
                with torch.no_grad():
                    output = model(batch)
                if not isinstance(output, WavefunctionOutput):
                    raise TypeError("readout replay requires WavefunctionOutput")
                raw = output.aux.get("pfaffian")
                if not isinstance(raw, torch.Tensor) or raw.shape != (len(rows),):
                    raise ValueError("readout replay requires the raw channel-summed pfaffian")
                expected_log = torch.tensor(
                    [float(row["logabs"]) for row in rows], device=flat.device, dtype=flat.dtype
                )
                expected_sign = torch.tensor(
                    [float(row["sign"]) for row in rows], device=flat.device, dtype=flat.dtype
                )
                if not (
                    torch.isfinite(raw).all()
                    and torch.isfinite(output.logabs).all()
                    and torch.allclose(output.logabs, expected_log, atol=1e-12, rtol=1e-12)
                    and torch.equal(output.sign, expected_sign)
                    and torch.equal(raw.sign(), expected_sign)
                ):
                    raise ValueError("readout replay disagrees with the retained signed log amplitude")
                floor_active = raw.square() < torch.as_tensor(
                    self.readout_eps, dtype=raw.dtype, device=raw.device
                )
                for row, value, active in zip(
                    rows,
                    raw.detach().cpu().tolist(),
                    floor_active.detach().cpu().tolist(),
                    strict=True,
                ):
                    draw, walker = divmod(row_count, records.n_walkers)
                    expected_identity = (row_count, draw, walker)
                    actual_identity = (
                        int(row["sample_index"]),
                        int(row["draw_index"]),
                        int(row["walker_index"]),
                    )
                    if actual_identity != expected_identity:
                        raise ValueError("readout replay input is not the complete ordered grid")
                    writer.writerow(
                        {
                            "sample_index": row_count,
                            "draw_index": draw,
                            "walker_index": walker,
                            "raw_channel_sum": value,
                            "floor_active": bool(active),
                        }
                    )
                    floor_count += int(active)
                    row_count += 1

        if row_count != records.row_count:
            raise ValueError("readout replay omitted trajectory rows")
        if parameter_fingerprint(model) != fingerprint:
            raise ValueError("model changed during trajectory or readout replay")
        receipt = {
            "schema": "raw-readout-replay/v1",
            "row_count": row_count,
            "trajectory_sha256": records.csv_sha256,
            "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "readout_eps": self.readout_eps,
            "floor_count": floor_count,
            "method": "same_forward_coordinate_replay",
            "independent_energy_oracle": False,
        }
        with path.with_suffix(".json").open("x") as handle:
            handle.write(json.dumps(receipt, indent=2) + "\n")
        return replace(generated, metadata={**generated.metadata, "raw_readout_replay": receipt})


__all__ = ["ReadoutTrajectoryMCMCGenerator"]
