"""Select, resolve, and optionally run exactly one Stage-Q mechanics cell.

The command deliberately owns selection rather than relying on inventory order.
It materializes the committed inventory, filters the live admitted method roster,
and requires exactly one ``(stage, architecture, optimizer, seed_label)`` match.
The default Q/control/Adam filters therefore require an explicit seed label while
the authority contains both Q seed labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf

# Direct ``python path/to/run_stage_q.py`` execution starts with the script
# directory on ``sys.path`` rather than the checkout root. Add the root before
# loading the hyphenated experiment namespace and before L1 checks its boundary.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tpen.accelerator import AcceleratorIdentity, AcceleratorKind
from tpen.distributed import ExecutionTopology

stage_coordinate = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
launch = importlib.import_module("experiments.atomistic.he-importance.launch")
train_config = importlib.import_module("experiments.atomistic.he-importance.train_config")

DEFAULT_STAGE = "Q"
DEFAULT_ARCHITECTURE = "control"
DEFAULT_OPTIMIZER = "adam"
Q_CHECKPOINT_EVERY_N_UPDATES = 1_000


class StageQSelectionError(ValueError):
    """The inventory does not identify exactly one admitted Stage-Q cell."""


@dataclass(frozen=True)
class CellDescriptor:
    """The auditable selection coordinates of one materialized cell."""

    stage: str
    architecture: str
    optimizer: str
    seed_label: int
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "architecture": self.architecture,
            "optimizer": self.optimizer,
            "seed_label": self.seed_label,
            "status": self.status,
        }


@dataclass(frozen=True)
class CellSelection:
    """One selected cell plus the complete discard record."""

    cell: Any
    selected: CellDescriptor
    discarded: tuple[CellDescriptor, ...]

    def receipt(self) -> dict[str, object]:
        return {
            "rule": (
                "exactly one admitted cell matching stage + architecture + optimizer + seed_label; "
                "refuse zero or multiple matches"
            ),
            "selected": self.selected.as_dict(),
            "discarded": [descriptor.as_dict() for descriptor in self.discarded],
        }


def _descriptor(cell: Any) -> CellDescriptor:
    manifest = cell.manifest
    identity = manifest["scientific_identity"]
    optimizer_cell = identity["optimizer_cell"]
    return CellDescriptor(
        stage=str(manifest["stage"]),
        architecture=str(identity["architecture"]),
        optimizer=str(identity["optimizer"]),
        seed_label=int(manifest["seed_identity"]["label"]),
        status=str(optimizer_cell["status"]),
    )


def _roster() -> tuple[Any, ...]:
    from tpen.hi_schema import HI_METHOD_ROSTER

    return tuple(HI_METHOD_ROSTER)


def roster_design_receipt(stage: str) -> dict[str, object]:
    """Report live roster admission against the authority's intended count."""

    roster = _roster()
    admitted = tuple(entry.method for entry in roster if entry.admitted and entry.target)
    unavailable = {
        entry.method: entry.requires for entry in roster if not entry.admitted or not entry.target
    }
    intended_count = stage_coordinate.stage_definition(stage).maximum_lineages
    if intended_count is None:
        raise StageQSelectionError(f"stage {stage!r} has no intended lineage count")
    return {
        "admitted_methods": admitted,
        "unavailable_methods": unavailable,
        "intended_count_from_authority": intended_count,
        "difference": {
            "intended_count_minus_admitted_method_count": intended_count - len(admitted),
            "dimensions": (
                "authority design runlets minus admitted method arms; "
                "reported without reconciliation"
            ),
        },
    }


def select_cell(
    cells: Iterable[Any],
    *,
    stage: str = DEFAULT_STAGE,
    architecture: str = DEFAULT_ARCHITECTURE,
    optimizer: str = DEFAULT_OPTIMIZER,
    seed_label: int | None = None,
) -> CellSelection:
    """Select one admitted cell and refuse ambiguity instead of first-match fallback."""

    admitted_methods = {
        entry.method for entry in _roster() if entry.admitted and entry.target
    }
    materialized = tuple(cells)
    descriptors = tuple(_descriptor(cell) for cell in materialized)
    requested = [
        (cell, descriptor)
        for cell, descriptor in zip(materialized, descriptors)
        if descriptor.stage == stage
        and descriptor.architecture == architecture
        and descriptor.optimizer == optimizer
        and descriptor.optimizer in admitted_methods
        and descriptor.status == "available"
        and (seed_label is None or descriptor.seed_label == seed_label)
    ]
    if len(requested) != 1:
        candidates = [descriptor.as_dict() for _, descriptor in requested]
        if not candidates:
            raise StageQSelectionError(
                "selection matched zero admitted cells for "
                f"stage={stage!r}, architecture={architecture!r}, optimizer={optimizer!r}, "
                f"seed_label={seed_label!r}"
            )
        raise StageQSelectionError(
            "selection is ambiguous; pass --seed-label explicitly: "
            + json.dumps(candidates, sort_keys=True)
        )
    selected_cell, selected_descriptor = requested[0]
    discarded = tuple(
        descriptor for descriptor in descriptors if descriptor != selected_descriptor
    )
    return CellSelection(selected_cell, selected_descriptor, discarded)


def _q_training_packet(cell: Any) -> Any:
    definition = stage_coordinate.stage_definition(cell.manifest["stage"])
    if definition.updates is None:
        raise StageQSelectionError("selected stage has no fixed training horizon")
    cadence = stage_coordinate.CheckpointCadence(
        Q_CHECKPOINT_EVERY_N_UPDATES,
        tuple(
            range(
                Q_CHECKPOINT_EVERY_N_UPDATES,
                definition.updates + 1,
                Q_CHECKPOINT_EVERY_N_UPDATES,
            )
        ),
    )
    packets = stage_coordinate.materialize_job_packets(
        (cell,),
        cadence,
        {"statistic": "logabs_variance"},
        (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
        {"walkers": 1, "burn_in_sweeps": 0},
        ddp_provenance={"launcher": "single-process-cli"},
    )
    if len(packets.train) != 1:
        raise StageQSelectionError("Stage-Q mechanics route did not produce exactly one train packet")
    # The shared materializer has one API for all HI stages and consequently
    # constructs analysis packets in memory. Stage Q is mechanics-only: retain
    # only ``train`` here; ranking and independent-sampler packets are discarded
    # in memory and are never written, scheduled, or run by this entry point.
    return packets.train[0]


def _topology() -> ExecutionTopology:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Stage-Q Polaris launch requires one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    identity = AcceleratorIdentity(
        kind=AcceleratorKind.CUDA,
        index=0,
        uuid=str(properties.uuid),
    )
    return ExecutionTopology.single_process(
        device="cuda:0",
        device_identity=identity,
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """Convert frozen packet mappings into ordinary strict-JSON containers."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, Path):
        return str(value)
    return value


def execute(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    cells = stage_coordinate.materialize_intended_configurations(output_root)
    selection = select_cell(
        cells,
        stage=args.stage,
        architecture=args.architecture,
        optimizer=args.optimizer,
        seed_label=args.seed_label,
    )
    packet = _q_training_packet(selection.cell)
    resolved = train_config.resolve_train_config(packet)
    report = {
        "facility": "Polaris (ALCF), PBS Pro",
        "selection": selection.receipt(),
        "roster_vs_design": roster_design_receipt(args.stage),
        "train_packet": {
            "content_hash": packet.cell.content_hash,
            "checkpoint_cadence": _jsonable(packet.checkpoint_cadence.science_parameters()),
            "ranking_packets": 0,
            "independent_sampler_packets": 0,
        },
        "resolved_config": OmegaConf.to_container(resolved, resolve=True),
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.dry_run:
        return 0

    output_path = Path(packet.cell.output_path)
    _write_json(output_path / "selection_receipt.json", report)
    exit_code = launch.launch_train(packet, _topology())
    if exit_code == 0:
        resolved_snapshot = output_path / "resolved_config.yaml"
        if not resolved_snapshot.is_file():
            raise RuntimeError(f"runner returned success without {resolved_snapshot}")
        (output_path / "COMPLETE").write_text("completed\n", encoding="utf-8")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--architecture", default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--optimizer", default=DEFAULT_OPTIMIZER)
    parser.add_argument("--seed-label", type=int)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return execute(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI ramp
    raise SystemExit(main())


__all__ = [
    "CellDescriptor",
    "CellSelection",
    "StageQSelectionError",
    "build_parser",
    "main",
    "roster_design_receipt",
    "select_cell",
]
