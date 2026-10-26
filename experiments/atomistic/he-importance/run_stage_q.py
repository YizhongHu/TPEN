"""Select, resolve, and optionally run exactly one Stage-Q mechanics cell.

The command deliberately owns selection rather than relying on inventory order.
It materializes the committed inventory, filters the live admitted method roster,
and requires exactly one ``(stage, context, optimizer, seed_label)`` match after
checking that the declared optimizer cell is truthful about the resolver output.
The default Q/O-raw/Adam filters therefore require an explicit seed label while
the authority contains both Q seed labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import importlib
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

from omegaconf import OmegaConf

# Direct ``python path/to/run_stage_q.py`` execution starts with the script
# directory on ``sys.path`` rather than the checkout root. Add the root before
# loading the hyphenated experiment namespace and before L1 checks its boundary.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

stage_coordinate = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
launch = importlib.import_module("experiments.atomistic.he-importance.launch")
optimizer_grid = importlib.import_module("experiments.atomistic.he-importance.optimizer_grid")
train_config = importlib.import_module("experiments.atomistic.he-importance.train_config")

DEFAULT_STAGE = "Q"
DEFAULT_CONTEXT = "O-raw"
DEFAULT_OPTIMIZER = "adam"
Q_CHECKPOINT_EVERY_N_UPDATES = 1_000


class StageQSelectionError(ValueError):
    """The inventory does not identify exactly one admitted Stage-Q cell."""


@dataclass(frozen=True)
class CellDescriptor:
    """The auditable selection coordinates of one materialized cell."""

    stage: str
    context: str
    optimizer: Mapping[str, object]
    seed_label: int
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "context": self.context,
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
                "exactly one truthful admitted cell matching stage + context + optimizer + seed_label; "
                "refuse zero or multiple matches"
            ),
            "selected": self.selected.as_dict(),
            "discarded": [descriptor.as_dict() for descriptor in self.discarded],
        }


def _descriptor(cell: Any) -> CellDescriptor:
    manifest = cell.manifest
    identity = manifest["scientific_identity"]
    context = identity["context"]
    optimizer = identity["optimizer"]
    optimizer_cell = identity["optimizer_cell"]
    return CellDescriptor(
        stage=str(manifest["stage"]),
        context=str(context["name"]),
        optimizer={
            "method": str(optimizer["method"]),
            "levels": dict(optimizer["levels"]),
        },
        seed_label=int(manifest["seed_identity"]["label"]),
        status=str(optimizer_cell["status"]),
    )


def _roster() -> tuple[Any, ...]:
    from tpen.hi_schema import HI_METHOD_ROSTER

    return tuple(HI_METHOD_ROSTER)


def _require_q_stage(stage: str) -> None:
    """Refuse non-Q stages before this mechanics-only route does any work.

    Parameters
    ----------
    stage : str
        Requested stage code.  This runner admits only the committed Stage-Q
        mechanics cell and must reject every other stage before materializing
        selection input or resolving a training packet.

    Raises
    ------
    StageQSelectionError
        If *stage* is not ``"Q"``.
    """

    if stage != DEFAULT_STAGE:
        raise StageQSelectionError(
            f"Stage-Q runner accepts only stage {DEFAULT_STAGE!r}; got {stage!r}"
        )


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
    context: str = DEFAULT_CONTEXT,
    optimizer: str = DEFAULT_OPTIMIZER,
    seed_label: int | None = None,
) -> CellSelection:
    """Select one truthful admitted cell and refuse first-match fallback.

    The resolver's consumed block is read before the exact-one decision. Each
    candidate is then checked against that block, so a declared-but-unconsumed
    optimizer level cannot win merely by appearing first in the inventory.
    """

    _require_q_stage(stage)
    admitted_methods = {
        entry.method for entry in _roster() if entry.admitted and entry.target
    }
    materialized = tuple(cells)
    descriptors = tuple(_descriptor(cell) for cell in materialized)
    requested = [
        (cell, descriptor)
        for cell, descriptor in zip(materialized, descriptors)
        if descriptor.stage == stage
        and descriptor.context == context
        and descriptor.optimizer["method"] == optimizer
        and descriptor.optimizer["method"] in admitted_methods
        and descriptor.status == "available"
        and (seed_label is None or descriptor.seed_label == seed_label)
    ]
    if not requested:
        raise StageQSelectionError(
            "selection matched zero admitted cells for "
            f"stage={stage!r}, context={context!r}, optimizer={optimizer!r}, "
            f"seed_label={seed_label!r}"
        )
    if context != DEFAULT_CONTEXT:
        raise StageQSelectionError(
            "truthfulness guard refused field 'context': "
            f"declared value {context!r}, consumed value {DEFAULT_CONTEXT!r}; "
            "resolver does not consume context/levels (item dd964863)"
        )

    truthful: list[tuple[Any, CellDescriptor]] = []
    refusals: list[str] = []
    for cell, descriptor in requested:
        try:
            # Resolve each candidate's own packet. The resolver may eventually
            # consume declared optimizer levels, so resolving requested[0]
            # once would make that first candidate win by construction.
            resolved = train_config.resolve_train_config(_q_training_packet(cell))
            _require_truthful_cell(cell, resolved)
        except StageQSelectionError as error:
            refusals.append(str(error))
        else:
            truthful.append((cell, descriptor))

    if len(truthful) != 1:
        if not truthful:
            detail = refusals[0] if refusals else "no candidate survived the truthfulness guard"
            raise StageQSelectionError(
                "truthfulness guard rejected every selected candidate: " + detail
            )
        candidates = [descriptor.as_dict() for _, descriptor in truthful]
        raise StageQSelectionError(
            "selection is ambiguous; pass --seed-label explicitly: "
            + json.dumps(candidates, sort_keys=True)
        )
    selected_cell, selected_descriptor = truthful[0]
    discarded = tuple(
        descriptor for descriptor in descriptors if descriptor != selected_descriptor
    )
    return CellSelection(selected_cell, selected_descriptor, discarded)


def _adam_signature_defaults() -> dict[str, object]:
    """Read Adam's constructor defaults from the runtime torch installation."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise StageQSelectionError(
            "truthfulness guard requires the runtime torch.optim.Adam signature"
        ) from error

    signature = inspect.signature(torch.optim.Adam)
    defaults: dict[str, object] = {}
    for name in ("betas", "eps", "weight_decay"):
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise StageQSelectionError(
                f"truthfulness guard cannot read torch.optim.Adam default for {name!r}"
            )
        defaults[name] = parameter.default
    return defaults


def _scheduler_configured(resolved: Any) -> bool:
    """Return whether the consumed config declares a scheduler block."""

    config = OmegaConf.to_container(resolved, resolve=True)

    def contains_scheduler(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                key in {"scheduler", "lr_scheduler"} and nested is not None
                or contains_scheduler(nested)
                for key, nested in value.items()
            )
        if isinstance(value, (tuple, list)):
            return any(contains_scheduler(nested) for nested in value)
        return False

    return contains_scheduler(config)


def _authority_o_raw_context() -> Mapping[str, Any]:
    """Return the exact O-raw coordinate mapping from the loaded authority."""

    authority = optimizer_grid.load_authority()
    for context in authority["contexts"]:
        if context["name"] == DEFAULT_CONTEXT:
            return {
                "name": context["name"],
                "overrides": dict(context["overrides"]),
            }
    raise StageQSelectionError("optimizer-grid authority has no O-raw context")


def _resolved_config_entries(value: Any, key: str, path: str = "") -> tuple[tuple[str, Any], ...]:
    """Find resolved configuration entries named *key* without importing TPEN."""

    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for nested_key, nested in value.items():
            nested_path = f"{path}.{nested_key}" if path else str(nested_key)
            if nested_key == key:
                found.append((nested_path, nested))
            found.extend(_resolved_config_entries(nested, key, nested_path))
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            found.extend(_resolved_config_entries(nested, key, f"{path}[{index}]"))
    return tuple(found)


def _control_residual_updater_target() -> Any:
    """Read the residual updater class path from the resolved control config."""

    control = train_config._compose_base_config()
    OmegaConf.resolve(control)
    control_config = OmegaConf.to_container(control, resolve=True)
    layers = control_config["model"]["layers"]
    targets = tuple(layer["update"]["_target_"] for layer in layers)
    if not targets or any(target != targets[0] for target in targets):
        raise StageQSelectionError(
            "truthfulness guard cannot identify one residual updater in the control config"
        )
    return targets[0]


def _context_refusal(field: str, declared: Any, consumed: Any) -> StageQSelectionError:
    return StageQSelectionError(
        f"truthfulness guard refused field '{field}': declared value {declared!r}, "
        f"consumed value {consumed!r}; context/levels are not consumed by the resolver "
        "(item dd964863)"
    )


def _require_truthful_cell(cell: Any, resolved: Any) -> None:
    """Refuse a cell whose declared context or Adam levels are not consumed."""

    identity = cell.manifest["scientific_identity"]
    declared_context = identity["context"]
    authority_context = _authority_o_raw_context()
    declared_context_keys = set(declared_context)
    authority_context_keys = set(authority_context)
    for field in sorted(declared_context_keys | authority_context_keys):
        if field == "overrides":
            continue
        declared_value = declared_context.get(field)
        consumed_value = authority_context.get(field)
        if declared_value != consumed_value:
            raise _context_refusal(f"context.{field}", declared_value, consumed_value)
    if declared_context_keys != authority_context_keys:
        raise _context_refusal(
            "context", dict(declared_context), dict(authority_context)
        )
    declared_overrides = declared_context["overrides"]
    authority_overrides = authority_context["overrides"]
    for field in sorted(set(declared_overrides) | set(authority_overrides)):
        declared_value = declared_overrides.get(field)
        consumed_value = authority_overrides.get(field)
        if declared_value != consumed_value:
            raise _context_refusal(
                f"context.overrides.{field}", declared_value, consumed_value
            )

    resolved_container = OmegaConf.to_container(resolved, resolve=True)
    input_feature_entries = _resolved_config_entries(resolved_container, "input_features")
    if input_feature_entries:
        for path, consumed_value in input_feature_entries:
            if consumed_value != authority_overrides["input_features"]:
                raise _context_refusal(
                    "context.overrides.input_features",
                    authority_overrides["input_features"],
                    f"{path}={consumed_value}",
                )
    else:
        # Source check: configs/train.yaml resolves model.embedding directly to
        # tpen.nn.Embedding with no input_features, bounded-distance, or extra-
        # term selector, and the HI schema has no such resolved model key. B00
        # is therefore the only input-feature value consumable by this route.
        pass

    layers = resolved_container.get("model", {}).get("layers", ())
    residual_target = _control_residual_updater_target()
    if not layers:
        raise _context_refusal(
            "context.overrides.feature_update", "residual", None
        )
    for index, layer in enumerate(layers):
        consumed_target = layer.get("update", {}).get("_target_")
        if consumed_target != residual_target:
            raise _context_refusal(
                f"context.overrides.feature_update.model.layers[{index}].update._target_",
                "residual",
                consumed_target,
            )

    declared_optimizer = identity["optimizer"]
    declared_method = declared_optimizer["method"]
    if declared_method != "adam":
        raise StageQSelectionError(
            "truthfulness guard refused field 'optimizer.method': "
            f"declared value {declared_method!r}, consumed value 'adam'"
        )

    declared = declared_optimizer["levels"]
    optimizer_cfg = OmegaConf.select(resolved, "optimizer")
    runner_optimizer_cfg = OmegaConf.select(resolved, "runner.optimizer")
    if optimizer_cfg is None or runner_optimizer_cfg is None:
        raise StageQSelectionError(
            "truthfulness guard refused field 'runner.optimizer': declared root "
            f"value {optimizer_cfg!r}, consumed value {runner_optimizer_cfg!r}"
        )
    root_optimizer = OmegaConf.to_container(optimizer_cfg, resolve=True)
    runner_optimizer = OmegaConf.to_container(runner_optimizer_cfg, resolve=True)
    for field in sorted(set(root_optimizer) | set(runner_optimizer)):
        declared_value = root_optimizer.get(field)
        consumed_value = runner_optimizer.get(field)
        if declared_value != consumed_value:
            raise StageQSelectionError(
                "truthfulness guard refused field "
                f"'runner.optimizer.{field}': declared root value {declared_value!r}, "
                f"consumed runner value {consumed_value!r}"
            )
    consumed_target = runner_optimizer.get("_target_")
    if consumed_target != "torch.optim.Adam":
        raise StageQSelectionError(
            "truthfulness guard refused field 'optimizer._target_': "
            "declared value 'torch.optim.Adam' for method 'adam', "
            f"consumed value {consumed_target!r}"
        )
    defaults = _adam_signature_defaults()
    configured_betas = runner_optimizer.get("betas")
    betas = configured_betas if configured_betas is not None else defaults["betas"]
    def configured_or_default(name: str, default: object) -> object:
        value = runner_optimizer.get(name)
        return default if value is None else value

    consumed = {
        "lr": runner_optimizer.get("lr"),
        "beta1": betas[0],
        "beta2": betas[1],
        "epsilon": configured_or_default("eps", defaults["eps"]),
        "weight_decay": configured_or_default("weight_decay", defaults["weight_decay"]),
        "parameter_groups": 1,
        "schedule": "constant" if not _scheduler_configured(resolved) else "scheduler",
    }
    declared_keys = set(declared)
    consumed_keys = set(consumed)
    if declared_keys != consumed_keys:
        missing = sorted(consumed_keys - declared_keys)
        extra = sorted(declared_keys - consumed_keys)
        details = []
        if missing:
            details.append(f"missing declared key(s) {missing!r}")
        if extra:
            details.append(f"extra declared key(s) {extra!r}")
        raise StageQSelectionError(
            "truthfulness guard refused optimizer level keys: " + "; ".join(details)
        )
    for field, consumed_value in consumed.items():
        declared_value = declared.get(field)
        if declared_value != consumed_value:
            raise StageQSelectionError(
                "truthfulness guard refused optimizer level "
                f"'{field}': declared value {declared_value!r}, "
                f"consumed value {consumed_value!r}"
            )


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


def _execution_facts() -> dict[str, object]:
    """Collect one-process launcher facts without importing TPEN topology types."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Stage-Q launch requires one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    return {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": "cuda:0",
        "job_id": os.environ.get("PBS_JOBID") or os.environ.get("PBS_JOB_ID"),
        "device_identity": {
            "kind": "cuda",
            "index": 0,
            "uuid": str(properties.uuid),
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refuse_occupied_output_path(path: Path) -> None:
    """Atomically claim an unoccupied cell leaf before writing evidence.

    Parameters
    ----------
    path : pathlib.Path
        Deterministic cell artifact leaf for the current execution attempt.
        Parents may be created non-exclusively, but the leaf is claimed with
        ``mkdir(exist_ok=False)``.  A path is occupied when it names an
        existing file, directory, or symlink, including a dangling symlink.
        A claimed path is the writer's exclusive ownership marker; an
        occupied path is retained as prior evidence.

    Raises
    ------
    StageQSelectionError
        If *path* cannot be claimed because it is occupied.  The function
        never removes or overwrites an occupant.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise StageQSelectionError(
            f"selected cell artifact path is already occupied: {path}"
        ) from error


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
    _require_q_stage(args.stage)
    output_root = Path(args.output_root).expanduser().resolve()
    cells = stage_coordinate.materialize_intended_configurations(
        output_root, availability=optimizer_grid.roster_availability()
    )
    selection = select_cell(
        cells,
        stage=args.stage,
        context=getattr(args, "context", DEFAULT_CONTEXT),
        optimizer=args.optimizer,
        seed_label=args.seed_label,
    )
    if args.walkers == stage_coordinate.COMMITTED_WALKERS:
        cell = selection.cell
    elif args.walkers == stage_coordinate.Q_MECHANICS_WALKERS:
        cell = stage_coordinate.with_q_mechanics_walkers(selection.cell)
    else:
        raise StageQSelectionError(f"unsupported Stage-Q walker count: {args.walkers}")
    selection = replace(selection, cell=cell)
    packet = _q_training_packet(selection.cell)
    resolved = train_config.resolve_train_config(packet)
    _require_truthful_cell(selection.cell, resolved)
    recorded_resolved = OmegaConf.to_container(resolved, resolve=True)
    report = {
        "selection": selection.receipt(),
        "walkers": stage_coordinate.walker_count(selection.cell.manifest),
        "roster_vs_design": roster_design_receipt(args.stage),
        "train_packet": {
            "content_hash": packet.cell.content_hash,
            "checkpoint_cadence": _jsonable(packet.checkpoint_cadence.science_parameters()),
            "ranking_packets": 0,
            "independent_sampler_packets": 0,
        },
        "resolved_config": recorded_resolved,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.dry_run:
        return 0

    output_path = Path(packet.cell.output_path)
    _refuse_occupied_output_path(output_path)
    _write_json(output_path / "selection_receipt.json", report)

    def _guarded_runner(config: Any, *, topology: object) -> int:
        """Validate launch.py's late resolution before delegating to TPEN."""

        _require_truthful_cell(selection.cell, config)
        launched_resolved = OmegaConf.to_container(config, resolve=True)
        if launched_resolved != recorded_resolved:
            raise StageQSelectionError(
                "launch truthfulness guard refused field 'resolved_config': "
                f"recorded value {recorded_resolved!r}, launched value {launched_resolved!r}"
            )
        return launch.run_from_config(config, topology=topology)

    exit_code = launch.launch_train(
        packet, _execution_facts(), runner=_guarded_runner
    )
    if exit_code == 0:
        resolved_snapshot = output_path / "resolved_config.yaml"
        if not resolved_snapshot.is_file():
            raise RuntimeError(f"runner returned success without {resolved_snapshot}")
        (output_path / "COMPLETE").write_text("completed\n", encoding="utf-8")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument("--optimizer", default=DEFAULT_OPTIMIZER)
    parser.add_argument("--seed-label", type=int)
    parser.add_argument("--walkers", type=int, default=stage_coordinate.COMMITTED_WALKERS)
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
